use ndarray::{Array2, Array4, Axis};
use num::clamp;
use std::{cmp::{max, min}, mem::swap};

pub struct BilateralGrid {
    pub grid: Array4<f64>,
    sigma_space: f64,
    sigma_color: f64,
    color_min: u16,
    space_pad: usize,
    color_pad: usize
}

impl BilateralGrid {
    pub fn from_image(image: &Array2<u16>, sigma_space: f64, sigma_color: f64) -> Self {
        let space_pad = 2;
        let color_pad = 2;

        let (image_height, image_width) = image.dim();

        let grid_height = ((image_height - 1) as f64 / sigma_space) as usize + 1 + 2 * space_pad;
        let grid_width = ((image_width - 1) as f64 / sigma_space) as usize + 1 + 2 * space_pad;

        let (color_min, color_max) = {
            let mut mi = std::u16::MAX;
            let mut ma = std::u16::MIN;
            image.iter().for_each(|v| {
                mi = min(mi, *v);
                ma = max(ma, *v);
            });
            (mi, ma)
        };
        let grid_depth =
            ((color_max - color_min) as f64 / sigma_color) as usize + 1 + 2 * color_pad;

        let mut grid = Array4::<f64>::zeros((grid_height, grid_width, grid_depth, 2));
        for row in 0..image_height {
            let grid_row = (row as f64 / sigma_space + 0.5) as usize + space_pad;

            for col in 0..image_width {
                let grid_col = (col as f64 / sigma_space + 0.5) as usize + space_pad;

                let color = image[(row, col)];
                let channel = ((color - color_min) as f64 / sigma_color + 0.5) as usize + color_pad;
                grid[(grid_row, grid_col, channel, 0)] += color as f64;
                grid[(grid_row, grid_col, channel, 1)] += 1.0;
            }
        }

        Self {
            grid: grid,
            sigma_color: sigma_color,
            sigma_space: sigma_space,
            color_min: color_min,
            space_pad: space_pad,
            color_pad: color_pad
        }
    }

    pub fn slice(&self, image: &Array2<u16>, dst_image: &mut Array2<u16>) {
        let (image_height, image_width) = image.dim();

        for row in 0..image_height {
            for col in 0..image_width {
                let color = image[(row, col)];
                dst_image[(row, col)] = self.trilinear(
                    row as f64 / self.sigma_space + self.space_pad as f64,
                    col as f64 / self.sigma_space + self.space_pad as f64,
                    (color - self.color_min) as f64 / self.sigma_color + self.color_pad as f64,
                ) as u16;
            }
        }
    }

    pub fn trilinear(&self, row: f64, col: f64, channel: f64) -> f64 {
        let (height, width, depth, _) = self.grid.dim();

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
              (1.0 - y_alpha) * (1.0 - x_alpha) * (1.0 - z_alpha) *  self.grid[(y_index,  x_index , z_index,  0)]
            + (1.0 - y_alpha) * x_alpha         * (1.0 - z_alpha) *  self.grid[(y_index,  xx_index, z_index,  0)]
            + y_alpha         * (1.0 - x_alpha) * (1.0 - z_alpha) *  self.grid[(yy_index, x_index , z_index,  0)]
            + y_alpha         * x_alpha         * (1.0 - z_alpha) *  self.grid[(yy_index, xx_index, z_index,  0)]
            + (1.0 - y_alpha) * (1.0 - x_alpha) * z_alpha         *  self.grid[(y_index,  x_index , zz_index, 0)]
            + (1.0 - y_alpha) * x_alpha         * z_alpha         *  self.grid[(y_index,  xx_index, zz_index, 0)]
            + y_alpha         * (1.0 - x_alpha) * z_alpha         *  self.grid[(yy_index, x_index , zz_index, 0)]
            + y_alpha         * x_alpha         * z_alpha         *  self.grid[(yy_index, xx_index, zz_index, 0)]
        };

        value
    }
 
    pub fn dim(&self) -> (usize, usize, usize, usize) {
        self.grid.dim()
    }
}

pub fn bilateral_filter(
    image: &Array2<u16>,
    sigma_space: f64,
    sigma_color: f64,
    dst_image: &mut Array2<u16>,
) {
    
    let mut grid = BilateralGrid::from_image(image, sigma_space, sigma_color);

    {
    let mut data_ptr = grid.grid.as_mut_ptr();

    let mut buffer = Array4::zeros(grid.dim());
    let mut buffer_ptr: *mut f64 = buffer.as_mut_ptr();

    let (grid_height, grid_width, grid_depth, _) = grid.dim();

    let row_stride = grid.grid.stride_of(Axis(0));
    let col_stride = grid.grid.stride_of(Axis(1));
    let channel_stride = grid.grid.stride_of(Axis(2));

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
    let dim = grid.dim();
    grid.grid.view_mut()
        .into_shape((dim.0*dim.1*dim.2, 2))
        .unwrap()
        .axis_iter_mut(Axis(0))
        .for_each(|mut color_count| {
            let count = color_count[1];
            if count > 0.0 {
                color_count[0] /= count;
                color_count[1] = 1.0;
            }
        });

    grid.slice(&image, dst_image);
}
