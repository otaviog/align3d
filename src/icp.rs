use super::kdtree::KdTree;
use super::pointcloud::PointCloud;
use super::transform::Transform;
use nalgebra::{Cholesky, Vector6};
use ndarray::prelude::s;
use ndarray::{Array2, Array3, Axis};
use nshare::ToNalgebra;

pub struct ICPParams {
    pub max_iterations: usize,
    pub weight: f32,
}

impl ICPParams {
    pub fn default() -> Self {
        Self {
            max_iterations: 15,
            weight: 0.5,
        }
    }

    pub fn max_iterations(&'_ mut self, value: usize) -> &'_ mut ICPParams {
        self.max_iterations = value;
        self
    }

    pub fn weight(&'_ mut self, value: f32) -> &'_ mut ICPParams {
        self.weight = value;
        self
    }
}

pub struct ICP<'a> {
    pub params: ICPParams,
    target: &'a PointCloud,
    kdtree: KdTree,
}

impl<'a> ICP<'a> {
    pub fn new(params: ICPParams, target: &'a PointCloud) -> Self {
        Self {
            params,
            target,
            kdtree: KdTree::new(&target.points),
        }
    }

    pub fn align(&self, source: &PointCloud) -> Transform {
        if let None = self.target.normals {}

        let target_normals = self.target.normals.as_ref().unwrap();
        let mut jt_j_array = Array3::<f32>::zeros((self.target.len(), 3, 3));
        let mut jt_r_array = Array2::<f32>::zeros((self.target.len(), 6));
        let mut optim_transform = Transform::eye();

        for _ in 0..self.params.max_iterations {
            let curr_source_points = &optim_transform * &source.points;
            let nearest = self.kdtree.nearest(&curr_source_points);

            for (idx, source_point) in source.points.rows().into_iter().enumerate() {
                let target_normal = &target_normals.slice(s![idx, ..]).into_nalgebra();
                let source_point = source_point.into_nalgebra();

                let twist = source_point.cross(&target_normal);
                let jacobian = [
                    target_normal[0],
                    target_normal[1],
                    target_normal[2],
                    twist[0],
                    twist[1],
                    twist[2],
                ];

                let residual = 0.0f32;
                let nearest_idx = nearest[idx];
                jt_r_array[[nearest_idx, 0]] += jacobian[0] * residual * self.params.weight;
                for i in 0..6 {
                    for j in 0..6 {
                        jt_j_array[[nearest_idx, i, j]] =
                            jacobian[i] * self.params.weight * jacobian[j];
                    }
                }
            }
            let jt_j = jt_j_array.sum_axis(Axis(0));
            let jt_r = jt_r_array.sum_axis(Axis(0));

            let update = Cholesky::new(jt_j.into_nalgebra())
                .unwrap()
                .solve(&jt_r.into_nalgebra());
            // let up2 = Vector6::from_row_slice(update.rows(0, 6));
            let update = Vector6::new(
                update[0], update[1], update[2], update[3], update[4], update[5],
            );

            optim_transform = &Transform::from_se3_exp(&update) * &optim_transform;
        }

        optim_transform
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    use crate::{
        imagepointcloud::ImagePointCloud, io::{dataset::RGBDDataset, write_ply}, pointcloud::PointCloud,
    };

    #[fixture]
    fn sample1() -> (PointCloud, PointCloud) {
        use crate::io::slamtb::SlamTbDataset;

        let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();

        let (cam1, rgbd_image1) = dataset.get_item(0).unwrap();
        let (cam2, rgbd_image2) = dataset.get_item(0).unwrap();
        let mut source = ImagePointCloud::from_rgbd_image(cam1, rgbd_image1);
        let mut target = ImagePointCloud::from_rgbd_image(cam2, rgbd_image2);

        source.compute_normals();
        target.compute_normals();

        (
            source.into(),
            target.into()
        )
    }

    #[rstest]
    fn test_icp(sample1: (PointCloud, PointCloud)) {
        let (source_pcl, target_pcl) = sample1;

        let transform = ICP::new(ICPParams::default(), &target_pcl).align(&source_pcl);

        let aligned_source_pcl = &transform * &source_pcl;
        write_ply("tests/data/out-icp1.ply", &aligned_source_pcl.into()).expect("Unable to write result");
    }
}

// pub fn compute_jacobian(normal, source_point, target_normal, weight)  scalar_t jacobian[6];
//     jacobian[0] = normal[0];
//     jacobian[1] = normal[1];
//     jacobian[2] = normal[2];
//
//     const Vector<scalar_t, 3> rot_twist = Tsrc_point.cross(normal);
//     jacobian[3] = rot_twist[0];
//     jacobian[4] = rot_twist[1];
//     jacobian[5] = rot_twist[2];
//
//     for (int k = 0; k < 6; ++k) {
//       Jtr_partial[k] += jacobian[k] * residual * weight;
//     }
//
// #pragma unroll
//     for (int krow = 0; krow < 6; ++krow) {
// #pragma unroll
//       for (int kcol = 0; kcol < 6; ++kcol) {
//         JtJ_partial[krow][kcol] += jacobian[kcol] * weight * jacobian[krow];
//       }
//     }
//
// }
