use ndarray::{Array2, Array4, Axis};
use num::{clamp, ToPrimitive};
use std::{
    cmp::{max, min},
};

/// Bilateral grid. A data structure for representing images
/// within its intensity space. 
/// 
/// More information: Chen, J., Paris, S., 
/// & Durand, F. (2007). Real-time edge-aware image processing with 
/// the bilateral grid. ACM Transactions on Graphics (TOG), 26(3), 103-es.
pub struct BilateralGrid<I> {
    /// The grid data. Shape is [H W Z 2], where the last dimension contains,
    /// in order, the color value and the counter.
    pub data: Array4<f64>,
    sigma_space: f64,
    sigma_color: f64,
    color_min: I,
    space_pad: usize,
    color_pad: usize,
}

impl<I> BilateralGrid<I>
where
    I: num::Bounded
        + Ord
        + Copy
        + std::ops::Sub
        + ToPrimitive
        + std::convert::From<<I as std::ops::Sub>::Output>
        + num::NumCast,
{
    pub fn from_image(image: &Array2<I>, sigma_space: f64, sigma_color: f64) -> Self {
        let space_pad = 2;
        let color_pad = 2;

        let (image_height, image_width) = image.dim();

        let grid_height = ((image_height - 1) as f64 / sigma_space) as usize + 1 + 2 * space_pad;
        let grid_width = ((image_width - 1) as f64 / sigma_space) as usize + 1 + 2 * space_pad;

        let (color_min, color_max) = {
            let mut mi = I::max_value();
            let mut ma = I::min_value();
            image.iter().for_each(|v| {
                mi = min(mi, *v);
                ma = max(ma, *v);
            });
            (mi, ma)
        };

        let grid_depth = {
            let diff: I = (color_max - color_min).into();
            (diff.to_f64().unwrap() / sigma_color) as usize + 1 + 2 * color_pad
        };

        let mut grid = Array4::<f64>::zeros((grid_height, grid_width, grid_depth, 2));
        for row in 0..image_height {
            let grid_row = (row as f64 / sigma_space + 0.5) as usize + space_pad;

            for col in 0..image_width {
                let grid_col = (col as f64 / sigma_space + 0.5) as usize + space_pad;

                let color = image[(row, col)];
                let channel = {
                    let diff: I = (color - color_min).into();
                    (diff.to_f64().unwrap() / sigma_color + 0.5) as usize + color_pad
                };
                grid[(grid_row, grid_col, channel, 0)] += color.to_f64().unwrap();
                grid[(grid_row, grid_col, channel, 1)] += 1.0;
            }
        }

        Self {
            data: grid,
            sigma_color,
            sigma_space,
            color_min,
            space_pad,
            color_pad,
        }
    }

    pub fn normalize(&mut self) {
        let dim = self.dim();
        self.data
            .view_mut()
            .into_shape((dim.0 * dim.1 * dim.2, 2))
            .unwrap()
            .axis_iter_mut(Axis(0))
            .for_each(|mut color_count| {
                let count = color_count[1];
                if count > 0.0 {
                    color_count[0] /= count;
                    color_count[1] = 1.0;
                }
            });
    }

    pub fn slice(&self, image: &Array2<I>, dst_image: &mut Array2<I>) {
        let (image_height, image_width) = image.dim();

        for row in 0..image_height {
            for col in 0..image_width {
                let color = image[(row, col)];
                
                let trilinear = self.trilinear(
                    row as f64 / self.sigma_space + self.space_pad as f64,
                    col as f64 / self.sigma_space + self.space_pad as f64,
                    {
                        let diff: I = (color - self.color_min).into();
                        diff.to_f64().unwrap() / self.sigma_color + self.color_pad as f64
                    },
                );

                unsafe {
                    dst_image[(row, col)] = num::cast::cast(trilinear).unwrap_unchecked();
                }
            }
        }
    }

    pub fn trilinear(&self, row: f64, col: f64, channel: f64) -> f64 {
        let (height, width, depth, _) = self.data.dim();

        let z_index = clamp(channel as usize, 0, depth - 1);
        let zz_index: usize = clamp((channel + 1.0) as usize, 0, depth - 1);
        let z_alpha = channel - z_index as f64;

        let y_index = clamp(row as usize, 0, height - 1);
        let yy_index: usize = clamp((row + 1.0) as usize, 0, height - 1);
        let y_alpha = row - y_index as f64;

        let x_index = clamp(col as usize, 0, width - 1);
        let xx_index: usize = clamp((col + 1.0) as usize, 0, width - 1);
        let x_alpha = col - x_index as f64;

        #[rustfmt::skip]
        let value = 
        {
              (1.0 - y_alpha) * (1.0 - x_alpha) * (1.0 - z_alpha) *  self.data[(y_index,  x_index , z_index,  0)]
            + (1.0 - y_alpha) * x_alpha         * (1.0 - z_alpha) *  self.data[(y_index,  xx_index, z_index,  0)]
            + y_alpha         * (1.0 - x_alpha) * (1.0 - z_alpha) *  self.data[(yy_index, x_index , z_index,  0)]
            + y_alpha         * x_alpha         * (1.0 - z_alpha) *  self.data[(yy_index, xx_index, z_index,  0)]
            + (1.0 - y_alpha) * (1.0 - x_alpha) * z_alpha         *  self.data[(y_index,  x_index , zz_index, 0)]
            + (1.0 - y_alpha) * x_alpha         * z_alpha         *  self.data[(y_index,  xx_index, zz_index, 0)]
            + y_alpha         * (1.0 - x_alpha) * z_alpha         *  self.data[(yy_index, x_index , zz_index, 0)]
            + y_alpha         * x_alpha         * z_alpha         *  self.data[(yy_index, xx_index, zz_index, 0)]
        };
        value
    }

    pub fn dim(&self) -> (usize, usize, usize, usize) {
        self.data.dim()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use rstest::{rstest, fixture};
    use crate::unit_test::bloei_luma16;

    use super::BilateralGrid;

    #[fixture]
    fn bilateral_grid(bloei_luma16: Array2<u16>) -> BilateralGrid<u16>{
        BilateralGrid::from_image(&bloei_luma16, 4.5, 30.0)
    }

    #[rstest]
    fn verify_grid_creation(bilateral_grid: BilateralGrid<u16>) {
        assert_eq!(bilateral_grid.dim(), (138, 104, 173, 2));
    }

    #[rstest]
    fn verify_slice(bloei_luma16: Array2<u16>, mut bilateral_grid: BilateralGrid<u16>) {
        let mut dest_image = bloei_luma16.clone();
        bilateral_grid.normalize();
        bilateral_grid.slice(&bloei_luma16, &mut dest_image);
        
        assert_eq!(dest_image.dim(), (600, 450));
        assert_eq!(dest_image[(421, 123)], 2266);
    }
}