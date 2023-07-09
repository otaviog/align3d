use image::{flat::SampleLayout, imageops::blur, ImageBuffer, Rgb, RgbImage};
use nalgebra::Vector3;
use ndarray::{Array2, Array3, ShapeBuilder};

/// Trait to convert into ndarray::Array3, this is different than nshare version
/// because it uses the shape [height, width, channels] instead of [channels, height, width].
pub trait IntoArray3 {
    fn into_array3(self) -> Array3<u8>;
}

impl IntoArray3 for image::RgbImage {
    fn into_array3(self) -> Array3<u8> {
        let SampleLayout {
            channels,
            channel_stride,
            height,
            height_stride,
            width,
            width_stride,
        } = self.sample_layout();
        let shape = (height as usize, width as usize, channels as usize);
        let strides = (height_stride, width_stride, channel_stride);
        Array3::from_shape_vec(shape.strides(strides), self.into_raw()).unwrap()
    }
}

/// Trait to convert objects into image::RgbImage
pub trait IntoImageRgb8 {
    fn into_image_rgb8(self) -> RgbImage;
}

impl IntoImageRgb8 for Array3<u8> {
    fn into_image_rgb8(self) -> RgbImage {
        let (height, width, channels) = self.dim();
        if channels != 3 {
            panic!("Array3 must have 3 channels");
        }
        RgbImage::from_raw(width as u32, height as u32, self.into_raw_vec()).unwrap()
    }
}

pub trait ToImageRgb8 {
    fn to_image_rgb8(self) -> RgbImage;
}

impl ToImageRgb8 for Array2<Vector3<u8>> {
    fn to_image_rgb8(self) -> RgbImage {
        let (height, width) = self.dim();
        RgbImage::from_fn(width as u32, height as u32, |x, y| {
            let c = self[(y as usize, x as usize)];
            image::Rgb([c[0], c[1], c[2]])
        })
    }
}

pub fn py_scale_down(src_img: &ImageBuffer<Rgb<u8>, Vec<u8>>, sigma: f32) -> Array3<u8> {
    let (src_height, src_width) = (src_img.height() as usize, src_img.width() as usize);
    let src_img = blur(src_img, sigma).into_array3();

    let (dst_height, dst_width) = (src_height / 2, src_width / 2);
    Array3::from_shape_fn((dst_height, dst_width, 3), |(i_dst, j_dst, c)| {
        num::clamp(src_img[[i_dst * 2, j_dst * 2, c]], 0, 255)
    })
}

#[cfg(test)]
mod tests {
    use ndarray::Array3;
    use rstest::rstest;

    use super::{py_scale_down, IntoImageRgb8};
    use crate::unit_test::bloei_rgb;

    #[rstest]
    fn verify_downsample_rgb_image(bloei_rgb: Array3<u8>) {
        let downsample = py_scale_down(&bloei_rgb.into_image_rgb8(), 1.0);
        assert_eq!(downsample.shape(), &[300, 225, 3]);
        downsample
            .into_image_rgb8()
            .save("tests/outputs/downsample_rgb_test.png")
            .unwrap();
    }
}
