use image::{flat::SampleLayout, RgbImage, imageops::blur};
use ndarray::{Array3, ArrayView3, ShapeBuilder};

use crate::utils::math::integer_fractional;

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

pub fn scale_down_rgb8(
    src_img: &RgbImage,
    sigma: f32
) -> Array3<u8> {
    let (src_height, src_width) = (src_img.height() as usize, src_img.width() as usize);
    let src_img = blur(src_img, sigma).into_array3();

    let (dst_height, dst_width) = (src_height / 2, src_width / 2);
    Array3::<u8>::from_shape_fn((dst_height, dst_width, 3), |(i_dst, j_dst, c)| {
        let v = src_img[[i_dst * 2, j_dst * 2, c]];
        num::clamp(v as u8, 0, 255)
    })


    // let mut target_image = Array3::zeros((dst_height, dst_width, 3));
    // for i_dst in 0..dst_height {
    //     let i_src = i_dst * 2;
    //     for j_dst in 0..dst_width {
    //         let j_src = j_dst * 2;
    //         for c in 0..channels {
    //             let v = src_image[[i_src, j_src, c]]];
    //             target_image[[i_dst, j_dst, c]] = num::clamp(v as u8, 0, 255);
    //         }
    //     }
    // }
    // target_image
}


#[cfg(test)]
mod tests {
    use image::{ImageBuffer, RgbImage};
    use ndarray::Array3;
    use rstest::rstest;

    use super::{scale_down_rgb8, IntoImageRgb8};
    use crate::unit_test::bloei_rgb;

    #[rstest]
    fn verify_downsample_rgb_image(bloei_rgb: Array3<u8>) {
        let downsample = scale_down_rgb8(&bloei_rgb.into_image_rgb8(), 1.0);
        assert_eq!(downsample.shape(), &[300, 225, 3]);
        downsample
            .into_image_rgb8()
            .save("tests/outputs/downsample_rgb_test.png")
            .unwrap();
    }
}
