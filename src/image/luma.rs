use image::GrayImage;
use ndarray::{s, Array1, Array2, ArrayView3};

/// Trait to convert an ndarray::Array* to an image::GrayImage
pub trait IntoLumaImage {
    fn to_luma_image(&self) -> GrayImage;
}

impl IntoLumaImage for Array2<f32> {
    /// Convert an ndarray::Array2<u16> to an image::GrayImage
    fn to_luma_image(&self) -> GrayImage {
        let (height, width) = (self.shape()[0], self.shape()[1]);

        let max = *self
            .iter()
            .max_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap())
            .unwrap();
        let min = *self
            .iter()
            .min_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap())
            .unwrap();

        let u8_image = self.map(|x| (((*x as f32 - min) / (max - min)) * 255.0) as u8);

        GrayImage::from_vec(width as u32, height as u32, u8_image.into_raw_vec()).unwrap()
    }
}

impl IntoLumaImage for Array2<u16> {
    /// Convert an ndarray::Array2<u16> to an image::GrayImage
    fn to_luma_image(&self) -> GrayImage {
        let (height, width) = (self.shape()[0], self.shape()[1]);

        let max = *self.iter().max().unwrap() as f32;
        let min = *self.iter().filter(|x| **x != 0).min().unwrap() as f32;

        let u8_image = self.map(|x| (((*x as f32 - min) / (max - min)) * 255.0) as u8);

        GrayImage::from_vec(width as u32, height as u32, u8_image.into_raw_vec()).unwrap()
    }
}

impl IntoLumaImage for Array2<u8> {
    /// Convert an ndarray::Array2<u16> to an image::GrayImage
    fn to_luma_image(&self) -> GrayImage {
        let (height, width) = (self.shape()[0], self.shape()[1]);

        GrayImage::from_vec(width as u32, height as u32, self.clone().into_raw_vec()).unwrap()
    }
}

pub fn rgb_to_luma(r: u8, g: u8, b: u8) -> f32 {
    const DIV: f32 = 1.0 / 255.0;
    (r as f32 * 0.3 + g as f32 * 0.59 + b as f32 * 0.11) * DIV
}

pub fn rgb_to_luma_u8(r: u8, g: u8, b: u8) -> u8 {
    (r as f32 * 0.3 + g as f32 * 0.59 + b as f32 * 0.11) as u8
}

pub trait IntoLumaArray<T> {
    fn to_luma_array(&self) -> Array2<T>;
}

impl IntoLumaArray<u8> for ArrayView3<'_, u8> {
    fn to_luma_array(&self) -> Array2<u8> {
        let (c, h, w) = self.dim();
        assert_eq!(c, 3);

        let grayscale: Array1<u8> = self
            .slice(s![0, .., ..])
            .iter()
            .zip(self.slice(s![1, .., ..]).iter())
            .zip(self.slice(s![2, .., ..]).iter())
            .map(|((r, g), b)| rgb_to_luma_u8(*r, *g, *b))
            .collect();

        grayscale.into_shape((h, w)).unwrap()
    }
}
