use ndarray::{Array3, ArrayView3};

use crate::utils::math::integer_fractional;

pub trait IntoImageRgb {
    fn to_image_rgb8(&self) -> image::RgbImage;
}

impl IntoImageRgb for ArrayView3<'_, u8> {
    fn to_image_rgb8(&self) -> image::RgbImage {
        // TODO: Make it more rustacean and less imperative.
        let (_, height, width) = self.dim();
        let mut image_rgb8 = image::RgbImage::new(width as u32, height as u32);
        for i in 0..height {
            for j in 0..width {
                let pixel = image::Rgb([self[[0, i, j]], self[[1, i, j]], self[[2, i, j]]]);
                image_rgb8.put_pixel(j as u32, i as u32, pixel);
            }
        }
        image_rgb8
    }
}

pub fn resize_image_rgb8(
    src_img: &ArrayView3<u8>,
    dst_width: usize,
    dst_height: usize,
) -> Array3<u8> {
    // TODO: Make it more rustacean and less imperative. And faster. 
    // As we working with downsampling by half ways, we can use nearest neighbors.
    let (channels, src_height, src_width) = src_img.dim();
    let mut target_image = Array3::zeros((channels, dst_height, dst_width));

    let height_ratio = src_height as f32 / dst_height as f32;
    let width_ratio = src_width as f32 / dst_width as f32;

    for i_dst in 0..dst_height {
        let (i_src, i_frac) = integer_fractional(i_dst as f32 * height_ratio);
        for j_dst in 0..dst_width {
            let (j_src, j_frac) = integer_fractional(j_dst as f32 * width_ratio);
            for c in 0..channels {
                let (v00, v01, v10, v11) = {
                    let v00 = src_img[[c, i_src, j_src]];
                    let v01 = if j_src + 1 < src_width {
                        src_img[[c, i_src, j_src + 1]]
                    } else {
                        v00
                    };
                    let v10 = if i_src + 1 < src_height {
                        src_img[[c, i_src + 1, j_src]]
                    } else {
                        v00
                    };
                    let v11 = if i_src + 1 < src_height && j_src + 1 < src_width {
                        src_img[[c, i_src + 1, j_src + 1]]
                    } else {
                        v00
                    };
                    (v00, v01, v10, v11)
                };
                let v0 = v00 as f32 * (1.0 - j_frac) + v01 as f32 * j_frac;
                let v1 = v10 as f32 * (1.0 - j_frac) + v11 as f32 * j_frac;
                let v = v0 * (1.0 - i_frac) + v1 * i_frac;
                target_image[[c, i_dst, j_dst]] = num::clamp(v as u8, 0, 255);
            }
        }
    }
    target_image
}

pub fn resize_image_rgb82(
    src_img: &ArrayView3<u8>,
    dst_width: usize,
    dst_height: usize,
) -> Array3<u8> {
    // TODO: Make it more rustacean and less imperative. And faster. 
    // As we working with downsampling by half ways, we can use nearest neighbors.
    let (src_height, src_width, channels) = src_img.dim();
    let mut target_image = Array3::zeros((dst_height, dst_width, channels));

    let height_ratio = src_height as f32 / dst_height as f32;
    let width_ratio = src_width as f32 / dst_width as f32;

    for i_dst in 0..dst_height {
        let (i_src, i_frac) = integer_fractional(i_dst as f32 * height_ratio);
        for j_dst in 0..dst_width {
            let (j_src, j_frac) = integer_fractional(j_dst as f32 * width_ratio);
            for c in 0..channels {
                let (v00, v01, v10, v11) = {
                    let v00 = src_img[[i_src, j_src, c]];
                    let v01 = if j_src + 1 < src_width {
                        src_img[[i_src, j_src + 1, c]]
                    } else {
                        v00
                    };
                    let v10 = if i_src + 1 < src_height {
                        src_img[[i_src + 1, j_src, c]]
                    } else {
                        v00
                    };
                    let v11 = if i_src + 1 < src_height && j_src + 1 < src_width {
                        src_img[[i_src + 1, j_src + 1, c]]
                    } else {
                        v00
                    };
                    (v00, v01, v10, v11)
                };
                let v0 = v00 as f32 * (1.0 - j_frac) + v01 as f32 * j_frac;
                let v1 = v10 as f32 * (1.0 - j_frac) + v11 as f32 * j_frac;
                let v = v0 * (1.0 - i_frac) + v1 * i_frac;
                target_image[[i_dst, j_dst, c]] = num::clamp(v as u8, 0, 255);
            }
        }
    }
    target_image
}


#[cfg(test)]
mod tests {
    use ndarray::Array3;
    use rstest::rstest;

    use super::*;
    use crate::unit_test::bloei_rgb;

    #[rstest]
    fn verify_downsample_rgb_image(bloei_rgb: Array3<u8>) {
        let downsampled = resize_image_rgb8(&bloei_rgb.view(), 200, 300);
        assert_eq!(downsampled.shape(), &[3, 300, 200]);
        bloei_rgb
            .view()
            .to_image_rgb8()
            .save("tests/outputs/downsample_rgb_test.png")
            .unwrap();
    }
}
