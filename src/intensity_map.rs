use super::color::rgb_to_luma;
use nalgebra::zero;
use ndarray::{s, Array1, Array2, Array3, Axis};

/// Stores a grayscale image with float and interpolation operations.
pub struct IntensityMap {
    map: Array2<f32>,
    shape: (usize, usize),
}

// The H is gradient divisor constant.
const H: f32 = 0.0005;
const H_INV: f32 = 1.0 / H;

impl IntensityMap {
    /// Returns the shape of the map (height, width).
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Creates a map with all zeros.
    pub fn zeros(shape: (usize, usize)) -> Self {
        Self {
            map: Array2::zeros((shape.0 + 1, shape.1 + 1)), // Adds border
            shape: shape,
        }
    }

    /// Fills the map with the values from a given image.
    /// It will try to reuse allocated map.
    ///
    /// # Arguments
    /// * image: The image data to be converted in a intensity map.
    ///   Its values are divided by 255.0.
    pub fn fill(&mut self, image: &Array2<u8>) {
        let (in_height, in_width) = {
            let dim = image.shape();
            (dim[0], dim[1])
        };

        let (map_grid_height, map_grid_width) = {
            let dim = self.map.shape();
            (dim[0], dim[1])
        };

        if in_height >= map_grid_height && in_width >= map_grid_width {
            self.map = Array2::zeros((in_height + 1, in_width + 1));
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
            self.map[(in_height, col)] = self.map[(in_height - 1, col)];
        }

        for row in 0..in_height - 1 {
            self.map[(row, in_width)] = self.map[(row, in_width - 1)];
        }

        self.map[(in_height, in_width)] = image[(in_height - 1, in_width - 1)] as f32 / 255.0;
    }

    /// Constructor to create a map filled with an image.
    /// See `fill`.
    pub fn from_luma_image(image: &Array2<u8>) -> Self {
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
    pub fn from_rgb_image(image: &Array3<u8>) -> Self {
        // TODO: remove unnecessary copies.
        let shape = image.shape();
        let color = image.view().into_shape((shape[0] * shape[1], 3)).unwrap();
        let luma = color
            .axis_iter(Axis(0))
            .map(|rgb| (rgb_to_luma(rgb[0], rgb[1], rgb[2]) * 255.0) as u8)
            .collect::<Array1<u8>>()
            .into_shape((shape[0], shape[1]))
            .unwrap();
        Self::from_luma_image(&luma)
    }

    /// Returns the intensity value with bilinear interpolation if
    /// u or v are not round numbers.
    ///
    /// # Arguments:
    ///
    /// * `u`: The "x" coordinate. Range is [0..1].
    /// * `v`: The "y" coordinate. Range is [0..1].
    ///
    /// # Returns:
    ///
    /// Bilinear interpolated value.
    pub fn bilinear(&self, u: f32, v: f32) -> f32 {
        // Hope that rustc knows how to optimize this.
        // Not in the mood to do another one
        self.bilinear_grad(u, v).0
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
    pub fn bilinear_grad(&self, u: f32, v: f32) -> (f32, f32, f32) {
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
                // Consider bord padding
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
                // Consider bord padding
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

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use rstest::rstest;

    use super::IntensityMap;
    use crate::unit_test::bloei_luma8;

    #[rstest]
    fn border_should_repeat(bloei_luma8: Array2<u8>) {
        let map = IntensityMap::from_luma_image(&bloei_luma8);
        let width = bloei_luma8.shape()[1];
        let _height = bloei_luma8.shape()[0];
        assert_eq!(
            map.bilinear(0.0, (width - 1) as f32),
            bloei_luma8[(width - 1, 0)] as f32 / 255.0
        );
        assert_eq!(
            map.bilinear(0.0, (width - 1) as f32 + 0.1),
            bloei_luma8[(width - 1, 0)] as f32 / 255.0
        );
    }

    #[rstest]
    fn round_uv_should_match_image(bloei_luma8: Array2<u8>) {
        let map = IntensityMap::from_luma_image(&bloei_luma8);

        for (y, x) in [(20, 0), (33, 44), (12, 48)] {
            assert_eq!(
                map.bilinear(x as f32, y as f32),
                bloei_luma8[(y, x)] as f32 / 255.0
            );
        }
    }
}
