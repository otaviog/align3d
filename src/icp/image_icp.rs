use nalgebra::Vector3;
use ndarray::s;
use num::Float;

use crate::{
    camera::PointSpace, optim::GaussNewton, range_image::RangeImage, transform::Transform, trig,
};

use super::{
    cost_function::{ColorDistance, PointPlaneDistance},
    icp_params::IcpParams,
};

pub struct ImageIcp<'target_lt> {
    pub params: IcpParams,
    target: &'target_lt mut RangeImage,
    pub initial_transform: Transform,
}

impl<'target_lt> ImageIcp<'target_lt> {
    pub fn new(params: IcpParams, target: &'target_lt mut RangeImage) -> Self {
        Self {
            params,
            target,
            initial_transform: Transform::eye(),
        }
    }

    /// Aligns the source point cloud to the target point cloud.
    pub fn align(&mut self, source: &RangeImage) -> Transform {
        let intensity_map = self.target.intensity_map();

        let target_normals = self.target.normals.as_ref().unwrap();
        let source_colors = source.intensities.as_ref().unwrap();

        let mut optim_transform = Transform::eye();

        let geometric_distance = PointPlaneDistance {};
        let color_distance = ColorDistance {};

        let mut geom_optim = GaussNewton::new();
        let mut color_optim = GaussNewton::new();
        let mut best_residual = Float::infinity();
        let mut best_transform = optim_transform.clone();
        for _ in 0..self.params.max_iterations {
            source.point_cloud_view().iter().for_each(
                |(linear_index, source_point, source_normal)| {
                    let source_point = optim_transform.transform_vector(&source_point);
                    let (x, y) = self
                        .target
                        .camera
                        .project_point(&PointSpace::Camera(source_point))
                        .unwrap();
                    let (xu, yu) = ((x + 0.5) as usize, (y + 0.5) as usize);

                    if let Some(target_point) = self.target.get_point(yu, xu) {
                        if (target_point - source_point).norm_squared() > 0.5 {
                            return; // exit closure
                        }
                        let src_normal = optim_transform.transform_normal(&source_normal);
                        let target_normal = {
                            let elem = target_normals.slice(s![yu, xu, ..]);
                            Vector3::new(elem[0], elem[1], elem[2])
                        };

                        if trig::angle_between_normals(&src_normal, &target_normal)
                            > self.params.max_normal_angle
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
                        let (target_color, du, dv) = intensity_map.bilinear_grad(x, y);
                        let source_color = source_colors[linear_index] as f32 / 255.0;

                        let ((dfx, dcx), (dfy, dcy)) =
                            self.target.camera.project_grad(&source_point);
                        let color_gradient = Vector3::new(du * dfx, dv * dfy, du * dcx + dv * dcy);

                        let (color_residual, color_jacobian) = color_distance.jacobian(
                            &source_point,
                            &color_gradient,
                            source_color,
                            target_color,
                        );
                        if color_residual * color_residual < 2.75 {
                            color_optim.step(color_residual, &color_jacobian);
                        }
                    }
                },
            );

            geom_optim.combine(&color_optim, self.params.weight, self.params.color_weight);
            let residual = geom_optim.mean_squared_residual();
            println!("Residual: {}", residual);

            let update = geom_optim.solve();
            optim_transform = &Transform::se3_exp(&update) * &optim_transform;
            geom_optim.reset();
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
        let mut rimage0 = sample_range_img_ds2.get_item(0).unwrap();
        let rimage1 = sample_range_img_ds2.get_item(5).unwrap();

        let result = ImageIcp::new(IcpParams::default(), &mut rimage0).align(&rimage1);
        println!("Result: {:?}", result);

        let expected = Transform::new(
            &Vector3::new(0.00838648, 0.0061255624, 0.00545656),
            Quaternion::new(-0.0002255599, 0.00024208902, 0.99999976, 0.0005938519),
        );
        assert_eq!(TransformMetrics::new(&result, &expected).angle, 0.0);
    }
}
