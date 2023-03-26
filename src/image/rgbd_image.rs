use ndarray::{Array2, Array3};

use crate::{camera::Camera, sampling::Downsample, bilateral::BilateralFilter};

use super::{py_scale_down, IntoImageRgb8};

/// A convinence struct that holds a color image, a depth image and its depth scale.
pub struct RgbdImage {
    pub color: Array3<u8>,
    pub depth: Array2<u16>,
    pub depth_scale: Option<f64>,
}

impl RgbdImage {
    pub fn new(color: Array3<u8>, depth: Array2<u16>) -> Self {
        Self {
            color,
            depth,
            depth_scale: None,
        }
    }

    pub fn with_depth_scale(color: Array3<u8>, depth: Array2<u16>, depth_scale: f64) -> Self {
        Self {
            color,
            depth,
            depth_scale: Some(depth_scale),
        }
    }

    pub fn width(&self) -> usize {
        self.color.shape()[1]
    }

    pub fn height(&self) -> usize {
        self.color.shape()[0]
    }
}

impl Downsample for RgbdImage {
    type Output = RgbdImage;
    fn downsample(&self, sigma: f32) -> RgbdImage {
        let resized_color = py_scale_down(&self.color.clone().into_image_rgb8(), sigma);
        let depth_filter = BilateralFilter::default();

        let resized_depth = depth_filter.scale_down(
            &self.depth,
        );

        RgbdImage {
            color: resized_color,
            depth: resized_depth,
            depth_scale: self.depth_scale,
        }
    }
}

pub struct RgbdFrame {
    pub camera: Camera,
    pub image: RgbdImage,
    // pub timestamp: Option<f64>,
}

impl RgbdFrame {
    pub fn new(camera: Camera, image: RgbdImage) -> Self {
        Self { camera, image }
    }

    pub fn into_parts(self) -> (Camera, RgbdImage) {
        (self.camera, self.image)
    }
}

impl Downsample for RgbdFrame {
    type Output = RgbdFrame;

    fn downsample(&self, scale: f32) -> RgbdFrame {
        RgbdFrame {
            camera: self.camera.scale(0.5),
            image: self.image.downsample(scale),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{
        image::IntoImageRgb8, io::core::RgbdDataset, sampling::Downsample,
        unit_test::sample_rgbd_dataset1,
    };

    #[rstest]
    fn test_downsample(sample_rgbd_dataset1: impl RgbdDataset) {
        let image = sample_rgbd_dataset1.get_item(0).unwrap().image;
        let scale_05 = image.downsample(0.5);
        assert_eq!([3, 240, 320], scale_05.color.shape());
        assert_eq!([240, 320], scale_05.depth.shape());
        assert_eq!(320, scale_05.width());
        assert_eq!(240, scale_05.height());
        scale_05
            .color
            .into_image_rgb8()
            .save("scale_05_color.png")
            .unwrap();
    }
}
