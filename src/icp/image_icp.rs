use itertools::izip;
use nalgebra::Vector3;
use ndarray::Axis;
use num::Float;
use rayon::prelude::{ParallelBridge, ParallelIterator};

use crate::{
    optim::{GaussNewton, GaussNewtonBatch},
    range_image::RangeImage,
    transform::{LieGroup, Transform},
    trig,
};

use super::{
    cost_function::{ColorDistance, PointPlaneDistance},
    icp_params::IcpParams,
};

pub struct ImageIcp<'target_lt> {
    pub params: IcpParams,
    target: &'target_lt RangeImage,
    pub initial_transform: Transform,
}

impl<'target_lt> ImageIcp<'target_lt> {
    pub fn new(params: IcpParams, target: &'target_lt RangeImage) -> Self {
        Self {
            params,
            target,
            initial_transform: Transform::eye(),
        }
    }

    /// Aligns the source point cloud to the target point cloud.
    ///
    /// # Arguments
    ///
    /// * `source` - The source point cloud.
    ///
    /// # Returns
    ///
    /// * The transformation that aligns the source point cloud to the target point cloud.
    pub fn align(&self, source: &RangeImage) -> Transform {
        let intensity_map = self
            .target
            .intensity_map
            .as_ref()
            .expect("Please, the target image should have a intensity map.");
        let target_normals = self
            .target
            .normals
            .as_ref()
            .expect("Please, the target image should have normals.");
        let source_colors = source
            .intensities
            .as_ref()
            .expect("Please, the source image should have intensity colors.");

        let mut optim_transform = self.initial_transform.clone();

        let geometric_distance = PointPlaneDistance {};
        let color_distance = ColorDistance {};

        let max_color_distance_sqr =
            self.params.max_color_distance * self.params.max_color_distance;
        let max_distance_sqr = self.params.max_distance * self.params.max_distance;

        let mut geom_optim = GaussNewton::<6>::new();
        let mut color_optim = GaussNewton::<6>::new();

        let mut best_residual = Float::infinity();
        let mut best_transform = optim_transform.clone();
        let batch_size = 2048;

        // let normal_angle = self.params.max_normal_angle.cos();

        for _ in 0..self.params.max_iterations {
            izip!(
                source
                    .mask
                    .view()
                    .to_shape(source.len())
                    .unwrap()
                    .axis_chunks_iter(Axis(0), batch_size),
                source
                    .points
                    .view()
                    .to_shape(source.len())
                    .unwrap()
                    .axis_chunks_iter(Axis(0), batch_size),
                source_colors
                    .view()
                    .to_shape(source.len())
                    .unwrap()
                    .axis_chunks_iter(Axis(0), batch_size)
            )
            .par_bridge()
            .map(|(mask_chunk, point_chunk, color_chunk)| {
                let mut color_batch = GaussNewtonBatch::new(batch_size);
                let mut geom_batch = GaussNewtonBatch::new(batch_size);

                izip!(mask_chunk, point_chunk, color_chunk)
                    .filter_map(|(&mask, point, color)| {
                        if mask != 0 {
                            Some((point, color))
                        } else {
                            None
                        }
                    })
                    .enumerate()
                    .for_each(|(batch_index, (&point, &color))| {
                        let p = optim_transform.transform_vector(&point);
                        let (u, v) = self.target.camera.project(&p);
                        let (u_int, v_int) = ((u + 0.5) as i32, (v + 0.5) as i32);
                        if let Some(target_point) =
                            self.target.get_point(v_int as usize, u_int as usize)
                        {
                            let geom_sqr_distance = (target_point - p).norm_squared();
                            if geom_sqr_distance > max_distance_sqr {
                                return; // exit closure
                            }
                            let target_normal = target_normals[(v_int as usize, u_int as usize)];
                            if trig::angle_between_normals(&p, &target_normal)
                                >= self.params.max_normal_angle
                            //if target_normal.dot(&p) >= normal_angle
                            {
                                return; // exit closure
                            }
                            let (residual, jacobian) =
                                geometric_distance.jacobian(&p, &target_point, &target_normal);

                            geom_batch.assign(batch_index, geom_sqr_distance, residual, &jacobian);

                            // Color part.
                            let (target_color, du, dv) = intensity_map.bilinear_grad(u, v);
                            let source_color = color as f32 * 0.00392156862745098; // / 255.0;
                            let ((dfx, dcx), (dfy, dcy)) = self.target.camera.project_grad(&p);
                            let color_gradient =
                                Vector3::new(du * dfx, dv * dfy, du * dcx + dv * dcy);
                            let (color_residual, color_jacobian) = color_distance.jacobian(
                                &p,
                                &color_gradient,
                                source_color,
                                target_color,
                            );
                            if color_residual * color_residual <= max_color_distance_sqr {
                                color_batch.assign(
                                    batch_index,
                                    geom_sqr_distance,
                                    color_residual,
                                    &color_jacobian,
                                );
                            }
                        }
                    });
                (geom_batch, color_batch)
            })
            .collect::<Vec<_>>()
            .iter()
            .for_each(|(geom, color)| {
                color_optim.step_batch(&color);
                geom_optim.step_batch(&geom)
            });

            geom_optim.combine(&color_optim, self.params.weight, self.params.color_weight);
            let residual = geom_optim.mean_squared_residual();
            let update = geom_optim.solve().unwrap();
            optim_transform = &Transform::exp(&LieGroup::Se3(update)) * &optim_transform;

            geom_optim.reset();
            color_optim.reset();

            if residual < best_residual {
                best_residual = residual;
                best_transform = optim_transform.clone();
            }
        }
        best_transform
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use rstest::rstest;

    use super::ImageIcp;
    use crate::{
        icp::icp_params::IcpParams,
        metrics::TransformMetrics,
        unit_test::{sample_range_img_ds2, TestRangeImageDataset},
    };

    #[rstest]
    fn test_align(sample_range_img_ds2: TestRangeImageDataset) {
        let rimage0 = sample_range_img_ds2.get(0).unwrap();
        let rimage1 = sample_range_img_ds2.get(1).unwrap();

        let gt_transform = sample_range_img_ds2.get_ground_truth(1, 0);

        let now = Instant::now();
        let actual = ImageIcp::new(
            IcpParams {
                max_iterations: 5,
                ..Default::default()
            },
            &rimage0,
        )
        .align(&rimage1);
        println!("Align computed in {:?}", now.elapsed());
        let angle_diff = TransformMetrics::new(&actual, &gt_transform).angle.abs();
        println!("Result metric: {}", angle_diff);
        assert!(angle_diff < 0.01);
    }
}
