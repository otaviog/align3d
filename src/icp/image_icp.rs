use nalgebra::{Cholesky, Vector3, Vector6};
use ndarray::{s, Array1, Array2, Axis};
use nshare::ToNalgebra;

use crate::{
    camera::{Camera, PointSpace},
    imagepointcloud::ImagePointCloud,
    intensity_map::IntensityMap,
    optim::GaussNewton,
    transform::Transform,
};

use super::icp_params::ICPParams;

pub struct PointPlaneDistance {
    weight: f32,
}

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

        let residual_weight = {
            let residual = (target_point - source_point).dot(&target_normal);
            residual*residual
        };

        (residual_weight, jacobian)
    }
}

pub struct ColorDistance {
    pub weight: f32,
}

impl ColorDistance {
    pub fn jacobian(
        &self,
        source_point: Vector3<f32>,
        source_color: f32,
        target_color: f32,
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

        let residual = target_color - source_color;
        let residual = residual * residual;

        (residual, jacobian)
    }
}

pub struct ImageICP<'target_lt, 'it_map> {
    pub params: ICPParams,
    camera: Camera,
    target: &'target_lt ImagePointCloud,
    intensity_map: &'it_map IntensityMap,
}

impl<'target_lt, 'it_map> ImageICP<'target_lt, 'it_map> {
    pub fn new(
        params: ICPParams,
        camera: Camera,
        target: &'target_lt ImagePointCloud,
        intensity_map: &'it_map IntensityMap,
    ) -> Self {
        Self {
            params,
            camera,
            target,
            intensity_map,
        }
    }

    /// Aligns the source point cloud to the target point cloud.
    pub fn align(&self, source: &ImagePointCloud) -> Transform {
        let target_normals = self.target.normals.as_ref().unwrap();
        let source_colors = source.intensities.as_ref().unwrap();

        let mut optim_transform = Transform::eye();

        let geometric_distance = PointPlaneDistance {
            weight: 1.0, // self.params.weight,
        };
        let color_distance = ColorDistance {
            weight: 1.0, //self.params.weight * 0.01,
        };

        let mut optim = GaussNewton::new();

        for _ in 0..self.params.max_iterations {
            let num_points = source.len();
            let mut residuals = Array1::<f32>::zeros(num_points);
            let mut jacobians = Array2::<f32>::zeros((num_points, 6));
            source.point_view().iter().enumerate().for_each(
                |(point_index, (pcl_index, src_point))| {
                    let source_point = optim_transform.transform_vector(&src_point);
                    let (x, y) = self
                        .camera
                        .project_point(&PointSpace::Camera(source_point))
                        .unwrap();
                    let (xu, yu) = (x as usize, y as usize);

                    if let Some(target_point) = self.target.get_point(yu, xu) {
                        // Geometric part
                        let target_normal = {
                            let elem = target_normals.slice(s![yu, xu, ..]);
                            Vector3::new(elem[0], elem[1], elem[2])
                        };

                        let (residual, jacobian) =
                            geometric_distance.jacobian(source_point, target_point, target_normal);

                        residuals[point_index] = residual;
                        jacobians
                            .row_mut(point_index)
                            .iter_mut()
                            .zip(jacobian)
                            .for_each(|(jdst, jsrc)| {
                                *jdst = jsrc;
                            });
                    }
                },
            );

            optim.step(&residuals, &jacobians, 0.005);
            let update = optim.solve();
            optim_transform = &Transform::se3_exp(&update) * &optim_transform;
         
        }
        optim_transform
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::ImageICP;
    use crate::{
        icp::icp_params::ICPParams,
        intensity_map::IntensityMap,
        unit_test::{sample_imrgbd_dataset1, TestRGBDDataset},
    };

    #[rstest]
    fn test_icp(sample_imrgbd_dataset1: TestRGBDDataset) {
        let (cam, pcl0) = sample_imrgbd_dataset1.get_item(0).unwrap();
        let (_, pcl1) = sample_imrgbd_dataset1.get_item(5).unwrap();

        let (height, width, _) = pcl0.colors.as_ref().unwrap().dim();
        let rgb = pcl0
            .colors
            .as_ref()
            .unwrap()
            .clone()
            .to_shape((3, height, width))
            .unwrap()
            .into_owned();
        let _ = ImageICP::new(
            ICPParams {
                max_iterations: 5,
                weight: 0.05,
            },
            cam,
            &pcl0,
            &IntensityMap::from_rgb_image(&rgb),
        )
        .align(&pcl1);
    }
}
