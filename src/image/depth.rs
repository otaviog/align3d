use ndarray::{Array2, ArrayView2};

use crate::Array2Recycle;

use crate::utils::math::integer_fractional;

/// Resize a depth image using bilinear interpolation and not considering 0 values (empty depth pixels)
///
/// # Arguments
///
/// * `depth_image` - The depth image to resize.
/// * `dst_height` - The target height of the resized image.
/// * `dst_width` - The target width of the resized image.
/// * `data` - An Array Recycle to reuse memory.
///
/// # Returns
///
/// The resized depth image.
pub fn resize_depth_image(
    depth_image: &ArrayView2<u16>,
    dst_width: usize,
    dst_height: usize,
    data: Array2Recycle<u16>,
) -> Array2<u16> {
    let (src_height, src_width) = (depth_image.shape()[0], depth_image.shape()[1]);
    let mut dst_image = data.get((dst_height, dst_width));

    let height_ratio = src_height as f32 / dst_height as f32;
    let width_ratio = src_width as f32 / dst_width as f32;

    for i_dst in 0..dst_height {
        let (i_src, i_frac) = integer_fractional(i_dst as f32 * height_ratio);

        for j_dst in 0..dst_width {
            let (j_src, j_frac) = integer_fractional(j_dst as f32 * width_ratio);

            let (v00, v01, v10, v11) = {
                let v00 = depth_image[[i_src, j_src]];
                if v00 == 0 {
                    dst_image[[i_dst, j_dst]] = 0;
                    continue;
                }

                let v01 = if j_dst + 1 < src_width {
                    let value = depth_image[[i_src, j_src + 1]];
                    if value == 0 {
                        v00
                    } else {
                        value
                    }
                } else {
                    v00
                };
                let v10 = if i_dst + 1 < src_height {
                    let value = depth_image[[i_src + 1, j_src]];
                    if value == 0 {
                        v00
                    } else {
                        value
                    }
                } else {
                    v00
                };

                let v11 = if i_dst + 1 < src_height && j_dst + 1 < src_width {
                    let value = depth_image[[i_src + 1, j_src + 1]];
                    if value == 0 {
                        v00
                    } else {
                        value
                    }
                } else {
                    v00
                };

                (v00, v01, v10, v11)
            };

            let u0_frac = v00 as f32 * (1.0 - j_frac) + v01 as f32 * j_frac;
            let u1_frac = v10 as f32 * (1.0 - j_frac) + v11 as f32 * j_frac;
            let v = u0_frac * (1.0 - i_frac) + u1_frac * i_frac;

            dst_image[[i_dst, j_dst]] = v as u16;
        }
    }

    dst_image
}

#[cfg(test)]
mod tests {
    use super::resize_depth_image;
    use crate::image::IntoLumaImage;
    use crate::Array2Recycle;
    use crate::{io::core::RgbdDataset, unit_test::sample_rgbd_dataset1};
    use rstest::rstest;
    #[rstest]
    fn should_resize_depth_images(sample_rgbd_dataset1: impl RgbdDataset) {
        let depth_image = sample_rgbd_dataset1.get_item(5).unwrap().image.depth;

        let data = Array2Recycle::Empty;
        let resized = resize_depth_image(&depth_image.view(), 320, 240, data);
        depth_image
            .to_luma_image()
            .save("tests/outputs/depth.png")
            .unwrap();
        resized
            .to_luma_image()
            .save("tests/outputs/depth_resized.png")
            .unwrap();
    }
}
