use super::image::rgb_to_luma;
use ndarray::{s, Array2, ArrayView2, ArrayView3};
use nshare::ToNdarray2;

/// Stores a grayscale image with float and interpolation operations.
#[derive(Debug, Clone)]
pub struct IntensityMap {
    map: Array2<f32>,
    shape: (usize, usize),
}

// The H is gradient divisor constant.
const H: f32 = 0.005;
const H_INV: f32 = 1.0 / H;
const BORDER_SIZE: usize = 2;

impl IntensityMap {
    /// Returns the shape of the map (height, width).
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Creates a map with all zeros.
    pub fn zeros(shape: (usize, usize)) -> Self {
        Self {
            map: Array2::zeros((shape.0 + BORDER_SIZE, shape.1 + BORDER_SIZE)),
            shape,
        }
    }

    /// Fills the map with the values from a given image.
    /// It will try to reuse allocated map.
    ///
    /// # Arguments
    /// * image: The image data to be converted in a intensity map.
    ///   Its values are divided by 255.0.
    pub fn fill(&mut self, image: &ArrayView2<u8>) {
        let (in_height, in_width) = {
            let dim = image.shape();
            (dim[0], dim[1])
        };

        let (map_grid_height, map_grid_width) = self.map.dim();

        if in_height >= map_grid_height && in_width >= map_grid_width {
            self.map = Array2::zeros((in_height + BORDER_SIZE, in_width + BORDER_SIZE));
        }

        self.shape = (in_height, in_width);

        // Fills the image.
        self.map
            .slice_mut(s!(..in_height, ..in_width))
            .iter_mut()
            .zip(image.iter())
            .for_each(|(dst, src)| {
                *dst = *src as f32 / 255.0;
            });

        // Fills the border X:
        for col in 0..in_width - 1 {
            let border = self.map[(in_height - 1, col)];
            for k in 0..2 {
                self.map[(in_height + k, col)] = border;
            }
        }

        for row in 0..in_height - 1 {
            let border = self.map[(row, in_width - 1)];
            for k in 0..2 {
                self.map[(row, in_width + k)] = border;
            }
        }

        let last_elem = image[(in_height - 1, in_width - 1)] as f32 / 255.0;
        for k in 0..2 {
            self.map[(in_height + k, in_width + k)] = last_elem;
        }
    }

    /// Constructor to create a map filled with an image.
    /// See `fill`.
    pub fn from_luma_image(image: &ArrayView2<u8>) -> Self {
        let shape = {
            let sh = image.shape();
            (sh[0], sh[1])
        };

        let mut map = Self::zeros(shape);
        map.fill(image);
        map
    }

    /// Constructor to create a map filled with a RGBimage.
    /// See `fill`.
    pub fn from_rgb_image(image: &ArrayView3<u8>) -> Self {
        // TODO: remove unnecessary copies.

        let luma = match image.dim() {
            (width, height, 3) => {
                let mut luma = Array2::<u8>::zeros((width, height));
                for y in 0..height {
                    for x in 0..width {
                        let r = image[[x, y, 0]];
                        let g = image[[x, y, 1]];
                        let b = image[[x, y, 2]];
                        luma[[x, y]] = (rgb_to_luma(r, g, b) * 255.0) as u8;
                    }
                }
                luma

                //let shape = image.shape();
                //let color = image.view().into_shape((shape[1] * shape[2], 3)).unwrap();
                //color
                //    .axis_iter(Axis(0))
                //    .map(|rgb| (rgb_to_luma(rgb[0], rgb[1], rgb[2]) * 255.0) as u8)
                //    .collect::<Array1<u8>>()
                //    .into_shape((shape[1], shape[2]))
                //    .unwrap()
            }
            (3, width, height) => {
                let mut luma = Array2::<u8>::zeros((width, height));
                for y in 0..height {
                    for x in 0..width {
                        let r = image[[0, x, y]];
                        let g = image[[1, x, y]];
                        let b = image[[2, x, y]];
                        luma[[x, y]] = (rgb_to_luma(r, g, b) * 255.0) as u8;
                    }
                }
                luma
            }
            _ => panic!("Invalid image shape"),
        };

        Self::from_luma_image(&luma.view())
    }

    /// Returns the intensity value with bilinear interpolation if
    /// u or v are not round numbers.
    ///
    /// # Arguments:
    ///
    /// * `u`: The "x" coordinate. Range is [0..width].
    /// * `v`: The "y" coordinate. Range is [0..height].
    ///
    /// # Returns:
    ///
    /// Bilinear interpolated value.
    pub fn bilinear(&self, u: f32, v: f32) -> f32 {
        let ui = u as usize;
        let vi = v as usize;

        let u_frac = u - ui as f32;
        let v_frac = v - vi as f32;

        let (val00, val10, val01, val11) = {
            (
                self.map[(vi, ui)],
                self.map[(vi, ui + 1)],
                self.map[[vi + 1, ui]],
                self.map[(vi + 1, ui + 1)],
            )
        };

        let u0_interp = val00 * (1.0 - u_frac) + val10 * u_frac;
        let u1_interp = val01 * (1.0 - u_frac) + val11 * u_frac;
        u0_interp * (1.0 - v_frac) + u1_interp * v_frac
    }

    pub fn bilinear_grad(&self, u: f32, v: f32) -> (f32, f32, f32) {
        let value = self.bilinear(u, v);
        let uh = self.bilinear(u + H, v);
        let vh = self.bilinear(u, v + H);

        (value, (uh - value) / H, (vh - value) / H)
    }

    /// Returns the intensity value with bilinear interpolation if
    /// u or v are not round numbers, and `u` and `v` numerical gradients.
    ///
    /// # Arguments:
    ///
    /// * `u`: The "x" coordinate. Range is [0..1].
    /// * `v`: The "y" coordinate. Range is [0..1].
    ///
    /// # Returns:
    ///
    /// * Bilinear interpolated value.
    /// * `u`'s gradient.
    /// * `v`'s gradient.
    pub fn bilinear_grad2(&self, u: f32, v: f32) -> (f32, f32, f32) {
        let ui = u as usize;
        let vi = v as usize;

        let u_frac = u - ui as f32;
        let v_frac = v - vi as f32;

        let value = {
            let (val00, val10, val01, val11) = {
                (
                    self.map[(vi, ui)],
                    self.map[(vi, ui + 1)],
                    self.map[[vi + 1, ui]],
                    self.map[(vi + 1, ui + 1)],
                )
            };

            let u0_interp = val00 * (1.0 - u_frac) + val10 * u_frac;
            let u1_interp = val01 * (1.0 - u_frac) + val11 * u_frac;
            u0_interp * (1.0 - v_frac) + u1_interp * v_frac
        };

        let i01 = {
            let v = v + H;
            let vi = v as usize;
            let v_frac = v - vi as f32;

            let (val00, val10, val01, val11) = {
                // Consider board padding
                (
                    self.map[(vi, ui)],
                    self.map[(vi, ui + 1)],
                    self.map[[vi + 1, ui]],
                    self.map[(vi + 1, ui + 1)],
                )
            };

            let u0_interp = val00 * (1.0 - u_frac) + val10 * u_frac;
            let u1_interp = val01 * (1.0 - u_frac) + val11 * u_frac;
            u0_interp * (1.0 - v_frac) + u1_interp * v_frac
        };

        let i10 = {
            let u = u + H;
            let ui = u as usize;
            let u_frac = u - ui as f32;

            let (val00, val10, val01, val11) = {
                // Consider board padding
                let vi = vi + 1;
                let ui = ui + 1;
                (
                    self.map[(vi, ui)],
                    self.map[(vi, ui + 1)],
                    self.map[[vi + 1, ui]],
                    self.map[(vi + 1, ui + 1)],
                )
            };

            let u0_interp = val00 * (1.0 - u_frac) + val10 * u_frac;
            let u1_interp = val01 * (1.0 - u_frac) + val11 * u_frac;
            u0_interp * (1.0 - v_frac) + u1_interp * v_frac
        };

        (value, (value - i10) * H_INV, (i01 - value) * H_INV)
    }
}

impl ToNdarray2 for IntensityMap {
    type Out = Array2<f32>;
    fn into_ndarray2(self) -> Self::Out {
        self.map
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use rstest::rstest;

    use super::IntensityMap;
    use crate::unit_test::bloei_luma8;

    #[rstest]
    fn border_should_repeat(bloei_luma8: Array2<u8>) {
        let map = IntensityMap::from_luma_image(&bloei_luma8.view());
        let width = bloei_luma8.shape()[1];
        let height = bloei_luma8.shape()[0];
        assert_eq!(
            map.bilinear(0.0, (height - 1) as f32 + 0.25),
            bloei_luma8[(height - 1, 0)] as f32 / 255.0
        );
        assert_eq!(
            map.bilinear((width - 1) as f32 + 0.25, 0.0),
            bloei_luma8[(0, width - 1)] as f32 / 255.0
        );
    }

    #[rstest]
    fn round_uv_should_match_image(bloei_luma8: Array2<u8>) {
        let map = IntensityMap::from_luma_image(&bloei_luma8.view());

        for (y, x) in [(20, 0), (33, 44), (12, 48)] {
            assert_eq!(
                map.bilinear(x as f32, y as f32),
                bloei_luma8[(y, x)] as f32 / 255.0
            );
        }
    }

    #[rstest]
    fn values(bloei_luma8: Array2<u8>) {
        let map = IntensityMap::from_luma_image(&bloei_luma8.view());
        for ((y, x), img_value) in bloei_luma8.indexed_iter() {
            let (value, _du, _dv) = map.bilinear_grad(x as f32, y as f32);
            assert_eq!(*img_value as f32 / 255.0, value);
        }
    }
}
