use crate::{bilateral::BilateralFilter, io::rgbd_image::RgbdFrame, Array2Recycle};

use super::RangeImage;

#[derive(Debug, Clone)]
/// Builder for multiple range images from RGB-D data.
pub struct RangeImageBuilder {
    with_normals: bool,
    with_luma: bool,
    bilateral_filter: Option<BilateralFilter<u16>>,
    bilateral_data: Array2Recycle<u16>,
    pyramid_levels: usize,
}

impl Default for RangeImageBuilder {
    /// Creates a new builder with default parameters.
    fn default() -> Self {
        Self {
            with_normals: false,
            with_luma: false,
            bilateral_filter: None,
            bilateral_data: Array2Recycle::Empty,
            pyramid_levels: 3,
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
    pub fn with_luma(mut self, value: bool) -> Self {
        self.with_luma = value;
        self
    }

    /// Sets the number of pyramid levels to use, this corresponds to the output length of [`build`].
    pub fn pyramid_levels(mut self, levels: usize) -> Self {
        self.pyramid_levels = levels;
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
            frame.image.depth = filter.filter(&frame.image.depth, Array2Recycle::Empty);
        }

        let mut range_images = if self.pyramid_levels > 1 {
            RangeImage::from_pyramid(&frame.pyramid(self.pyramid_levels))
        } else {
            vec![RangeImage::from_rgbd_frame(&frame)]
        };

        for range_image in range_images.iter_mut() {
            if self.with_normals {
                range_image.compute_normals();
            }
            if self.with_luma {
                range_image.compute_intensity();
                range_image.intensity_map();
            }
        }

        range_images
    }
}
