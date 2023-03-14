use crate::kdtree::KdTree;
use crate::pointcloud::PointCloud;
use crate::transform::Transform;
use crate::Array1Recycle;
use nalgebra::{Cholesky, Vector3, Vector6};
use ndarray::{Array2, Array3, Axis};
use nshare::ToNalgebra;
use super::icp_params::ICPParams;

pub struct ICP<'target_lt> {
    pub params: ICPParams,
    target: &'target_lt PointCloud,
    kdtree: KdTree,
}

impl<'target_lt> ICP<'target_lt> {
    pub fn new(params: ICPParams, target: &'target_lt PointCloud) -> Self {
        Self {
            params,
            target,
            kdtree: KdTree::new(&target.points.view()),
        }
    }

    pub fn align(&self, source: &PointCloud) -> Transform {
        if let None = self.target.normals {
            return Transform::eye();
        }

        let target_normals = self.target.normals.as_ref().unwrap();
        let mut optim_transform = Transform::eye();

        let mut jt_j_array = Array3::<f32>::zeros((self.target.len(), 6, 6));
        let mut jt_r_array = Array2::<f32>::zeros((self.target.len(), 6));

        for _ in 0..self.params.max_iterations {
            let current_source_points = &optim_transform * &source.points;
            let nearest = self
                .kdtree
                .nearest::<3>(&current_source_points.view(), Array1Recycle::Empty);

            current_source_points
                .axis_iter(Axis(0))
                .enumerate()
                .for_each(|(idx, source_point)| {
                    let source_point =
                        Vector3::new(source_point[0], source_point[1], source_point[2]);
                    let (target_point, target_normal) = {
                        let row_point = self.target.points.row(idx);
                        let row_normal = target_normals.row(idx);

                        (
                            Vector3::new(row_point[0], row_point[1], row_point[2]),
                            Vector3::new(row_normal[0], row_normal[1], row_normal[2]),
                        )
                    };

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
                    let nearest_idx = nearest[idx];
                    let mut jt_r_row = jt_r_array.row_mut(nearest_idx);

                    let residual = residual * self.params.weight;
                    for i in 0..6 {
                        jt_r_row[i] = jacobian[i] * residual;
                    }

                    for i in 0..6 {
                        for j in 0..6 {
                            jt_j_array[[nearest_idx, i, j]] =
                                jacobian[i] * self.params.weight * jacobian[j];
                        }
                    }
                });

            let jt_j = jt_j_array
                .sum_axis(Axis(0))
                .into_nalgebra()
                .fixed_slice::<6, 6>(0, 0)
                .into_owned();
            let jt_r = jt_r_array.sum_axis(Axis(0));

            let update = Cholesky::new(jt_j).unwrap().solve(&jt_r.into_nalgebra());
            let update = Vector6::new(
                update[0], update[1], update[2], update[3], update[4], update[5],
            );

            optim_transform = &Transform::se3_exp(&update) * &optim_transform;

            // Resets the Jacobians for the next iteration.
            jt_r_array.fill(0.0);
            jt_j_array.fill(0.0);
        }

        optim_transform
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    use crate::{
        imagepointcloud::ImagePointCloud,
        io::{core::RGBDDataset, write_ply},
        pointcloud::PointCloud,
    };

    #[fixture]
    fn sample1() -> (PointCloud, PointCloud) {
        use crate::io::slamtb::SlamTbDataset;

        let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();

        let (cam1, rgbd_image1) = dataset.get_item(0).unwrap().into_parts();
        let (cam2, rgbd_image2) = dataset.get_item(3).unwrap().into_parts();
        let mut source = ImagePointCloud::from_rgbd_image(&cam1, &rgbd_image1);
        let mut target = ImagePointCloud::from_rgbd_image(&cam2, &rgbd_image2);

        source.compute_normals();
        target.compute_normals();

        (PointCloud::from(&source), PointCloud::from(&target))
    }

    #[rstest]
    fn test_icp(sample1: (PointCloud, PointCloud)) {
        let (source_pcl, target_pcl) = sample1;

        let transform = ICP::new(ICPParams::default(), &target_pcl).align(&source_pcl);

        let aligned_source_pcl = &transform * &source_pcl;
        write_ply("tests/data/out-icp1.ply", &aligned_source_pcl.into())
            .expect("Unable to write result");
    }
}


