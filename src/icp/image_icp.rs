use nalgebra::{Cholesky, Vector3, Vector6};
use ndarray::{s, Array2, Axis};
use nshare::ToNalgebra;

use crate::{
    camera::Camera, imagepointcloud::ImagePointCloud, intensity_map::IntensityMap,
    transform::Transform,
};

use super::icp_params::ICPParams;

pub struct PointPlaneDistance {
    weight: f32,
}

impl PointPlaneDistance {
    pub fn jacobian(
        &self,
        source_point: Vector3<f32>,
        target_point: Vector3<f32>,
        target_normal: Vector3<f32>,
    ) -> ([f32; 6], [f32; 36]) {
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
        let residual = residual * self.weight;

        let residual_row = [
            jacobian[0] * residual,
            jacobian[1] * residual,
            jacobian[2] * residual,
            jacobian[3] * residual,
            jacobian[4] * residual,
            jacobian[5] * residual,
        ];

        let mut jacobian_row = [0.0; 36];
        for i in 0..6 {
            for j in 0..6 {
                jacobian_row[i * 6 + j] = jacobian[i] * self.weight * jacobian[j];
            }
        }
        (residual_row, jacobian_row)
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
    ) -> ([f32; 6], [f32; 36]) {
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
        let residual = residual * residual * self.weight;

        let residual_row = [
            jacobian[0] * residual,
            jacobian[1] * residual,
            jacobian[2] * residual,
            jacobian[3] * residual,
            jacobian[4] * residual,
            jacobian[5] * residual,
        ];

        let mut jacobian_row = [0.0; 36];
        for i in 0..6 {
            for j in 0..6 {
                jacobian_row[i * 6 + j] = jacobian[i] * self.weight * jacobian[j];
            }
        }
        (residual_row, jacobian_row)
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

    pub fn align(&self, source: &ImagePointCloud) -> Transform {
        let target_normals = self.target.normals.as_ref().unwrap();
        let source_colors = source.intensities.as_ref().unwrap();

        let mut optim_transform = Transform::eye();

        let mut jt_r_array = Array2::<f32>::zeros((self.target.valid_points_count(), 6));
        let mut jt_j_array = Array2::<f32>::zeros((self.target.valid_points_count(), 6 * 6));

        let geometric_distance = PointPlaneDistance {
            weight: self.params.weight,
        };
        let color_distance = ColorDistance {
            weight: self.params.weight * 0.01,
        };
        for _ in 0..self.params.max_iterations {
            source.point_view().iter().enumerate().for_each(
                |(point_count, (pcl_index, src_point))| {
                    let source_point = &optim_transform * &src_point;
                    let (x, y) = self.camera.project(&source_point);
                    let (xu, yu) = (x as usize, y as usize);

                    if let Some(target_point) = self.target.get_point(yu, xu) {
                        // Geometric part
                        let target_normal = {
                            let elem = target_normals.slice(s![yu, xu, ..]);
                            Vector3::new(elem[0], elem[1], elem[2])
                        };
                        let (jt_r, jt_j) =
                            geometric_distance.jacobian(source_point, target_point, target_normal);

                        // Color part.
                        let (source_color, du, dv) = self.intensity_map.bilinear_grad(x, y);
                        let ((dfx, dcx), (dfy, dcy)) = self.camera.project_grad(&source_point);
                        let gradk = Vector3::new(du * dfx, dv * dfy, du * dcx + dv * dcy);
                        let color = source_colors[pcl_index];
                        let (color_jt_r, color_jt_j) =
                            color_distance.jacobian(source_point, source_color, color, gradk);

                        jt_r_array
                            .row_mut(point_count)
                            .iter_mut()
                            .zip(jt_r.iter())
                            .zip(color_jt_r.iter())
                            .for_each(|((dst, src), src_c)| {
                                *dst = *src + *src_c;
                            });

                        jt_j_array
                            .row_mut(point_count)
                            .iter_mut()
                            .zip(jt_j.iter())
                            .zip(color_jt_j.iter())
                            .for_each(|((dst, src), src_c)| {
                                *dst = *src + *src_c;
                            });
                    }
                },
            );
            let jt_r = jt_r_array.sum_axis(Axis(0));
            let jt_j = {
                let sum = jt_j_array.sum_axis(Axis(0));
                sum.into_shape((6, 6))
                    .unwrap()
                    .into_nalgebra()
                    .fixed_slice::<6, 6>(0, 0)
                    .into_owned()
            };

            let update = {
                let v = Cholesky::new(jt_j).unwrap().solve(&jt_r.into_nalgebra());
                Vector6::<f32>::new(v[0], v[1], v[2], v[3], v[4], v[5])
            };

            optim_transform = &Transform::from_se3_exp(&update) * &optim_transform;
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

        let _ = ImageICP::new(
            ICPParams {
                max_iterations: 5,
                weight: 0.05,
            },
            cam,
            &pcl0,
            &IntensityMap::from_rgb_image(pcl0.colors.as_ref().unwrap()),
        )
        .align(&pcl1);
    }
}
