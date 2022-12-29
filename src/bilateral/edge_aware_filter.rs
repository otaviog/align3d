use std::{marker::PhantomData, mem::swap};

use ndarray::{Array2, Array4, Axis};
use num::ToPrimitive;

use crate::memory::Array2Recycle;

use super::BilateralGrid;

/// Bilateral filter using Bilateral Grid.
///
/// Based on the code in https://gist.github.com/ginrou/02e945562607fad170a1.
pub struct BilateralFilter<I> {
    _phantom: PhantomData<I>,
    /// The space (XY) down sample factor.
    pub sigma_space: f64,
    /// The intensity down sample factor.
    pub sigma_color: f64,
}

impl<I> Default for BilateralFilter<I>
where
    I: num::Bounded
        + Ord
        + Copy
        + std::ops::Sub
        + ToPrimitive
        + std::convert::From<<I as std::ops::Sub>::Output>
        + num::NumCast,
{
    fn default() -> Self {
        BilateralFilter::new(4.50000000225, 29.9999880000072)
    }
}

impl<I> BilateralFilter<I>
where
    I: num::Bounded
        + Ord
        + Copy
        + std::ops::Sub
        + ToPrimitive
        + std::convert::From<<I as std::ops::Sub>::Output>
        + num::NumCast,
{
    pub fn new(sigma_space: f64, sigma_color: f64) -> Self {
        Self {
            _phantom: PhantomData::default(),
            sigma_space,
            sigma_color,
        }
    }

    fn convolution(grid: &mut BilateralGrid<I>) {
        let mut data_ptr = grid.data.as_mut_ptr();

        let mut buffer = Array4::zeros(grid.dim());
        let mut buffer_ptr: *mut f64 = buffer.as_mut_ptr();

        let (grid_height, grid_width, grid_depth, _) = grid.dim();

        let row_stride = grid.data.stride_of(Axis(0));
        let col_stride = grid.data.stride_of(Axis(1));
        let channel_stride = grid.data.stride_of(Axis(2));
        for plane_offset in &[row_stride, col_stride, channel_stride] {
            let plane_offset = *plane_offset;
            for _ in 0..2 {
                swap(&mut data_ptr, &mut buffer_ptr);
                for row in 1..grid_height - 1 {
                    for col in 1..grid_width - 1 {
                        let mut b_ptr = unsafe {
                            buffer_ptr.offset(row as isize * row_stride + col as isize * col_stride)
                        };
                        let mut d_ptr = unsafe {
                            data_ptr.offset(row as isize * row_stride + col as isize * col_stride)
                        };

                        for _channel in 1..grid_depth {
                            let (prev_value, curr_value, next_value) = {
                                unsafe {
                                    let prev = (
                                        *b_ptr.offset(-plane_offset),
                                        *b_ptr.offset(-plane_offset + 1),
                                    );
                                    let curr = (*b_ptr, *b_ptr.add(1));
                                    let next = (
                                        *b_ptr.add(plane_offset as usize),
                                        *b_ptr.add((plane_offset + 1) as usize),
                                    );

                                    (prev, curr, next)
                                }
                            };

                            let set = (
                                (prev_value.0 + next_value.0 + 2.0 * curr_value.0) / 4.0,
                                (prev_value.1 + next_value.1 + 2.0 * curr_value.1) / 4.0,
                            );

                            unsafe {
                                *d_ptr = set.0;
                                *d_ptr.add(1) = set.1;

                                b_ptr = b_ptr.offset(channel_stride);
                                d_ptr = d_ptr.offset(channel_stride);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Filters the image. It will try to reuse buffers if possible.
    ///
    /// # Arguments:
    ///
    /// * `image`: Input image.
    ///
    /// # Returns:
    ///
    /// * The filtered image.
    pub fn filter(&self, image: &Array2<I>, result: Array2Recycle<I>) -> Array2<I>
    where
        I: num::Zero,
    {
        let mut result = result.get(image.dim());

        let mut grid = BilateralGrid::from_image(image, self.sigma_space, self.sigma_color);
        BilateralFilter::convolution(&mut grid);

        grid.normalize();
        grid.slice(image, &mut result);
        result
    }
}
