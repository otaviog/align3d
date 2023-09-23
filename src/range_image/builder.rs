use crate::{bilateral::BilateralFilter, image::RgbdFrame};

use super::RangeImage;

#[derive(Debug, Clone)]
/// Builder for multiple range images from RGB-D data.
pub struct RangeImageBuilder {
    with_normals: bool,
    with_intensity: bool,
    bilateral_filter: Option<BilateralFilter<u16>>,
    // bilateral_data: Array2Recycle<u16>,
    pyramid_levels: usize,
    blur_sigma: f32,
}

impl Default for RangeImageBuilder {
    /// Creates a new builder with default parameters.
    fn default() -> Self {
        Self {
            with_normals: true,
            with_intensity: true,
            bilateral_filter: None,
            pyramid_levels: 3,
            blur_sigma: 1.0,
        }
    }
}

impl RangeImageBuilder {
    /// Enables bilateral filtering of the depth map.
    /// The filter is applied to the original scale depth map before the range image is created.
    pub fn with_bilateral_filter(mut self, filter: Option<BilateralFilter<u16>>) -> Self {
        self.bilateral_filter = filter;
        self
    }

    /// Computes the normals of the range image.
    /// See [`RangeImage::compute_normals`].
    pub fn with_normals(mut self, value: bool) -> Self {
        self.with_normals = value;
        self
    }

    /// Computes the intensity of the range image.
    /// See [`RangeImage::compute_intensity`] and [`RangeImage::intensity_map`].
    pub fn with_intensity(mut self, value: bool) -> Self {
        self.with_intensity = value;
        self
    }

    /// Sets the number of pyramid levels to use, this corresponds to the output length of [`build`].
    /// See [`RangeImage::pyramid`].
    pub fn pyramid_levels(mut self, levels: usize) -> Self {
        self.pyramid_levels = levels;
        self
    }

    /// Sets the sigma value for the Gaussian blur applied when build a range image pyramid.
    /// See [`RangeImage::pyramid`].
    pub fn blur_sigma(mut self, sigma: f32) -> Self {
        self.blur_sigma = sigma;
        self
    }

    /// Builds the range images from the given RGB-D frame.
    ///
    /// # Arguments
    ///
    /// * `frame` - The RGB-D frame to build the range images from.
    ///
    /// # Returns
    ///
    /// A vector of range images, the length of the vector depends on the number of pyramid levels.
    pub fn build(&self, mut frame: RgbdFrame) -> Vec<RangeImage> {
        if let Some(filter) = &self.bilateral_filter {
            frame.image.depth = filter.filter(&frame.image.depth);
        }
        let mut first_image = RangeImage::from_rgbd_frame(&frame);
        if self.with_normals {
            first_image.compute_normals();
        }
        let mut range_images = first_image.pyramid(self.pyramid_levels, self.blur_sigma);
        for range_image in range_images.iter_mut() {
            if self.with_intensity {
                range_image.compute_intensity();
                range_image.compute_intensity_map();
            }
        }

        range_images
    }
}
