use nalgebra::Vector3;
use ndarray::s;
use num::Float;

use crate::{
    camera::PointSpace,
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
        let mut color_batch = GaussNewtonBatch::new(self.target.len());

        let mut best_residual = Float::infinity();
        let mut best_transform = optim_transform.clone();
        for _ in 0..self.params.max_iterations {
            source.point_cloud_view().iter().for_each(
                |(linear_index, source_point, source_normal)| {
                    let source_point = optim_transform.transform_vector(&source_point);
                    let (u, v) = self
                        .target
                        .camera
                        .project_point(&PointSpace::Camera(source_point))
                        .unwrap();
                    let (u_int, v_int) = ((u + 0.5) as i32, (v + 0.5) as i32);
                    if u_int < 0
                        || u_int >= self.target.width() as i32
                        || v_int < 0
                        || v_int >= self.target.height() as i32
                    {
                        return; // exit closure
                    }

                    if let Some(target_point) =
                        self.target.get_point(v_int as usize, u_int as usize)
                    {
                        let geom_sqr_distance = (target_point - source_point).norm_squared();
                        if geom_sqr_distance > max_distance_sqr {
                            return; // exit closure
                        }
                        let source_normal = optim_transform.transform_normal(&source_normal);
                        let target_normal = {
                            let elem = target_normals.slice(s![v_int, u_int, ..]);
                            Vector3::new(elem[0], elem[1], elem[2])
                        };

                        if trig::angle_between_vecs(&source_normal, &target_normal)
                            >= self.params.max_normal_angle
                        {
                            return; // exit closure
                        }

                        let (residual, jacobian) = geometric_distance.jacobian(
                            &source_point,
                            &target_point,
                            &target_normal,
                        );

                        geom_optim.step(residual, &jacobian);

                        // Color part.
                        let (target_color, du, dv) = intensity_map.bilinear_grad(u, v);
                        let source_color = source_colors[linear_index];
                        let source_color = source_color as f32 / 255.0;

                        let ((dfx, dcx), (dfy, dcy)) =
                            self.target.camera.project_grad(&source_point);
                        let color_gradient = Vector3::new(du * dfx, dv * dfy, du * dcx + dv * dcy);

                        let (color_residual, color_jacobian) = color_distance.jacobian(
                            &source_point,
                            &color_gradient,
                            source_color,
                            target_color,
                        );

                        if color_residual * color_residual <= max_color_distance_sqr {
                            color_batch.assign(
                                linear_index,
                                geom_sqr_distance,
                                color_residual,
                                &color_jacobian,
                            );
                        }
                    }
                },
            );
            color_optim.step_batch(&color_batch);
            geom_optim.combine(&color_optim, self.params.weight, self.params.color_weight);
            let residual = geom_optim.mean_squared_residual();
            let update = geom_optim.solve().unwrap();
            optim_transform = &Transform::exp(&LieGroup::Se3(update)) * &optim_transform;

            geom_optim.reset();
            color_optim.reset();
            color_batch.reset();

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
    use nalgebra::{Quaternion, Vector3};
    use rstest::rstest;

    use super::ImageIcp;
    use crate::{
        icp::icp_params::IcpParams,
        metrics::TransformMetrics,
        transform::Transform,
        unit_test::{sample_range_img_ds2, TestRangeImageDataset},
    };

    #[rstest]
    fn test_align(sample_range_img_ds2: TestRangeImageDataset) {
        let rimage0 = sample_range_img_ds2.get(0).unwrap();
        let rimage1 = sample_range_img_ds2.get(5).unwrap();

        let result = ImageIcp::new(IcpParams::default(), &rimage0).align(&rimage1);
        println!("Result: {:?}", result);

        let expected = Transform::new(
            &Vector3::new(0.00838648, 0.0061255624, 0.00545656),
            &Quaternion::new(-0.0002255599, 0.00024208902, 0.99999976, 0.0005938519),
        );
        assert_eq!(TransformMetrics::new(&result, &expected).angle, 0.0);
    }
}
