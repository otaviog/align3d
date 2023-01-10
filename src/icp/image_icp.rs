use nalgebra::{Cholesky, Vector3, Vector6};
use ndarray::{s, Array2, Axis};
use nshare::ToNalgebra;

use crate::{camera::Camera, imagepointcloud::ImagePointCloud, transform::Transform};

use super::icp_params::ICPParams;

pub struct PointPlaneDistance {
    weight: f32,
}

impl PointPlaneDistance {
    pub fn update(
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

pub struct ImageICP<'target_lt> {
    pub params: ICPParams,
    camera: Camera,
    target: &'target_lt ImagePointCloud,
}

impl<'target_lt> ImageICP<'target_lt> {
    pub fn new(params: ICPParams, camera: Camera, target: &'target_lt ImagePointCloud) -> Self {
        Self {
            params,
            camera,
            target,
        }
    }

    pub fn align(&self, source: &ImagePointCloud) -> Transform {
        if let None = self.target.normals {
            return Transform::eye();
        }

        let target_normals = self.target.normals.as_ref().unwrap();
        // let target_intensity = self.target.features.as_ref().unwrap();
        let mut optim_transform = Transform::eye();

        let mut jt_r_array = Array2::<f32>::zeros((self.target.valid_points_count(), 6));
        let mut jt_j_array = Array2::<f32>::zeros((self.target.valid_points_count(), 6 * 6));

        let distance_func = PointPlaneDistance {
            weight: self.params.weight,
        };
        for _ in 0..self.params.max_iterations {
            source
                .point_view()
                .iter()
                .enumerate()
                .for_each(|(point_idx, point)| {
                    let proj_point = &optim_transform * &point;
                    let (x, y) = self.camera.project(proj_point);
                    let (x, y) = (x as usize, y as usize);

                    if let Some(target_point) = self.target.get_point(y, x) {
                        let target_normal = {
                            let elem = target_normals.slice(s![y, x, ..]);
                            Vector3::new(elem[0], elem[1], elem[2])
                        };
                        let (jt_r, jt_j) =
                            distance_func.update(proj_point, target_point, target_normal);
                        let mut row_accessor = jt_r_array.row_mut(point_idx);
                        for i in 0..6 {
                            row_accessor[i] = jt_r[i];
                        }

                        let mut row_accessor = jt_j_array.row_mut(point_idx);
                        for i in 0..(6 * 6) {
                            row_accessor[i] = jt_j[i];
                        }

                        // let tgt_int = target_intensity[[y, x, 0]];
                        // 
                        // let project_grad = self.camera.project_grad(
                        //     proj_point
                        // );
                        // let grad = Vector3::new();
                    }
                });
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
        )
        .align(&pcl1);
    }
}
