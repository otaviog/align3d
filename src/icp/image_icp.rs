use nalgebra::Vector3;
use ndarray::{s, Array1, Array2};
use num::Float;

use crate::{
    camera::PointSpace, intensity_map, optim::GaussNewton, range_image::RangeImage,
    transform::Transform, trig,
};

use super::icp_params::IcpParams;

pub struct PointPlaneDistance {}

impl PointPlaneDistance {
    /// Computes the residual and the Jacobian of the point-plane distance.
    ///
    /// # Arguments
    ///
    /// * source_point - 3D point in the source frame.
    /// * target_point - 3D point in the target frame.
    /// * target_normal - Normal of the plane in the target frame.
    ///
    /// # Returns
    ///
    /// * $J^tr$ and $J^tJ$ of the point-plane distance,
    /// where the first is a (6) vector and the former is a (6, 6)
    /// matrix as a (36) vector.
    pub fn jacobian(
        &self,
        source_point: Vector3<f32>,
        target_point: Vector3<f32>,
        target_normal: Vector3<f32>,
    ) -> (f32, [f32; 6]) {
        let twist = source_point.cross(&target_normal);
        let jacobian = [
            target_normal[0],
            target_normal[1],
            target_normal[2],
            twist[0],
            twist[1],
            twist[2],
        ];

        let residual = (target_point - source_point).dot(&target_normal);

        (residual, jacobian)
    }
}

pub struct ColorDistance {}

impl ColorDistance {
    pub fn jacobian(
        &self,
        source_point: Vector3<f32>,
        target_normal: Vector3<f32>,
        source_color: f32,
        target_color: f32,
    ) -> (f32, [f32; 6]) {
        let twist = source_point.cross(&target_normal);
        let jacobian = [
            target_normal[0],
            target_normal[1],
            target_normal[2],
            twist[0],
            twist[1],
            twist[2],
        ];

        let residual = source_color - target_color;
        let residual = residual;

        (residual, jacobian)
    }
}

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
                            > 18.0.to_radians()
                        {
                            return; // exit closure
                        }

                        let (residual, jacobian) =
                            geometric_distance.jacobian(source_point, target_point, target_normal);

                        geom_optim.step(residual, &jacobian);

                        // Color part.
                        let (target_color, du, dv) = intensity_map.bilinear_grad(x, y);
                        let ((dfx, dcx), (dfy, dcy)) =
                            self.target.camera.project_grad(&source_point);
                        let color_gradient = Vector3::new(du * dfx, dv * dfy, du * dcx + dv * dcy);
                        let source_color = source_colors[linear_index] as f32 / 255.0;
                        let (color_residual, color_jacobian) = color_distance.jacobian(
                            source_point,
                            color_gradient,
                            source_color,
                            target_color,
                        );

                        color_optim.step(color_residual, &color_jacobian);
                    }
                },
            );

            geom_optim.combine(&color_optim, self.params.weight, 0.25);
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
        metrics::transform_difference,
        transform::Transform,
        unit_test::{sample_range_img_ds2, TestRangeImageDataset},
    };

    #[rstest]
    fn test_align(sample_range_img_ds2: TestRangeImageDataset) {
        let mut rimage0 = sample_range_img_ds2.get_item(0).unwrap();
        let rimage1 = sample_range_img_ds2.get_item(5).unwrap();

        let result = ImageIcp::new(
            IcpParams {
                max_iterations: 10,
                weight: 0.05,
            },
            &mut rimage0,
        )
        .align(&rimage1);
        println!("Result: {:?}", result);

        let inv_expected = Transform::new(
            &Vector3::new(0.00022050377, 7.3633055e-5, -1.51071e-5),
            Quaternion::new(2.059626e-5, 0.00888227, 0.0008264509, 0.99996024),
        )
        .inverse();
        assert_eq!(transform_difference(&result, &inv_expected), 0.0);
    }
}
