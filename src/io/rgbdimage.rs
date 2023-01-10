use image::{imageops::FilterType, ImageBuffer, Luma, Rgb};
use ndarray::{Array2, Array3};
use nshare::{ToNdarray2, ToNdarray3};

use crate::sampling::Downsampleble;

pub struct RGBDImage {
    pub color: Array3<u8>,
    pub depth: Array2<u16>,
    pub depth_scale: Option<f64>,
}

impl RGBDImage {
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
        self.color.shape()[2]
    }

    pub fn height(&self) -> usize {
        self.color.shape()[1]
    }
}

impl Downsampleble<RGBDImage> for RGBDImage {
    fn downsample(&self, scale: f64) -> RGBDImage {
        let (height, width) = (self.height() as u32, self.width() as u32);
        let resized_color = {
            let raw = self.color.as_slice().unwrap();
            let color_buffer = ImageBuffer::<Rgb<u8>, &[u8]>::from_raw(width, height, raw).unwrap();
            let resized_image = image::imageops::resize(
                &color_buffer,
                (width as f64 * scale) as u32,
                (height as f64 * scale) as u32,
                FilterType::Gaussian,
            );
            resized_image.into_ndarray3()
        };

        let resized_depth = {
            let raw = self.depth.as_slice().unwrap();
            let image_buffer =
                ImageBuffer::<Luma<u16>, &[u16]>::from_raw(width, height, raw).unwrap();
            let resized_image = image::imageops::resize(
                &image_buffer,
                (width as f64 * scale) as u32,
                (height as f64 * scale) as u32,
                FilterType::Nearest,
            );
            resized_image.into_ndarray2()
        };

        RGBDImage {
            color: resized_color,
            depth: resized_depth,
            depth_scale: self.depth_scale,
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{
        io::dataset::RGBDDataset, sampling::Downsampleble, unit_test::sample_rgbd_dataset1,
    };

    #[rstest]
    fn test_downsample(sample_rgbd_dataset1: impl RGBDDataset) {
        let (_, image) = sample_rgbd_dataset1.get_item(0).unwrap();
        let scale_05 = image.downsample(0.5);
        assert_eq!([3, 240, 320], scale_05.color.shape());
        assert_eq!([240, 320], scale_05.depth.shape());
        assert_eq!(320, scale_05.width());
        assert_eq!(240, scale_05.height());
    }
}
