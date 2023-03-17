use super::cost_function::PointPlaneDistance;
use super::icp_params::IcpParams;
use crate::pointcloud::PointCloud;
use crate::transform::Transform;
use crate::trig;
use crate::{kdtree::KdTree, optim::GaussNewton};
use itertools::izip;
use nalgebra::Vector3;
use ndarray::Axis;
use nshare::ToNalgebra;
use num::Float;

/// Standard Iterative Closest Point (ICP) algorithm for aligning two point clouds.
/// This implementation uses the point-to-plane distance.
pub struct Icp<'target_lt> {
    // Parameters of the ICP algorithm.
    pub params: IcpParams,
    // Initial transformation to start the algorithm. Default is the identity.
    pub initial_transform: Transform,
    target: &'target_lt PointCloud,
    kdtree: KdTree,
}

impl<'target_lt> Icp<'target_lt> {
    /// Create a new ICP instance.
    ///
    /// # Arguments
    ///
    /// * params - Parameters of the ICP algorithm.
    /// * target - Target point cloud.
    pub fn new(params: IcpParams, target: &'target_lt PointCloud) -> Self {
        Self {
            params,
            initial_transform: Transform::eye(),
            target,
            kdtree: KdTree::new(&target.points.view()),
        }
    }

    /// Aligns the source point cloud to the target point cloud.
    ///
    /// # Arguments
    ///
    /// * source - Source point cloud.
    ///
    /// # Returns
    ///
    /// The transformation that aligns the source point cloud to the target point cloud.
    pub fn align(&self, source: &PointCloud) -> Transform {
        if let None = self.target.normals {
            return Transform::eye();
        }

        let mut optim_transform = Transform::eye();
        let mut optim = GaussNewton::new();
        let geom_cost = PointPlaneDistance {};

        let mut best_residual = Float::infinity();
        let mut best_transform = optim_transform.clone();
        for _ in 0..self.params.max_iterations {
            for (source_point, source_normal) in izip!(
                source.points.axis_iter(Axis(0)),
                source.normals.as_ref().unwrap().axis_iter(Axis(0))
            ) {
                let source_point = optim_transform.transform_vector(&Vector3::new(
                    source_point[0],
                    source_point[1],
                    source_point[2],
                ));
                let source_normal = optim_transform.transform_normal(&Vector3::new(
                    source_normal[0],
                    source_normal[1],
                    source_normal[2],
                ));

                let (found_index, found_distance) = self.kdtree.nearest3d(&source_point);

                if found_distance > self.params.max_distance * self.params.max_distance {
                    // continue;
                }

                let target_normal = self
                    .target
                    .normals
                    .as_ref()
                    .unwrap()
                    .row(found_index)
                    .into_nalgebra()
                    .fixed_slice::<3, 1>(0, 0)
                    .into_owned();
                if trig::angle_between_normals(&source_normal, &target_normal)
                    > self.params.max_normal_angle
                {
                    continue;
                }

                let target_point = self
                    .target
                    .points
                    .row(found_index)
                    .into_nalgebra()
                    .fixed_slice::<3, 1>(0, 0)
                    .into_owned();

                let (residual, jacobian) =
                    geom_cost.jacobian(&source_point, &target_point, &target_normal);

                optim.step(residual, &jacobian);
            }

            let residual = optim.mean_squared_residual();
            println!("Residual: {}", residual);

            let update = optim.solve();
            optim_transform = &Transform::se3_exp(&update) * &optim_transform;
            optim.reset();
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
    use super::*;
    use rstest::*;

    use crate::{
        io::{core::RgbdDataset, write_ply},
        pointcloud::PointCloud,
        range_image::RangeImage,
    };

    #[fixture]
    fn sample1() -> (PointCloud, PointCloud) {
        use crate::io::slamtb::SlamTbDataset;

        let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();

        let (cam1, rgbd_image1) = dataset.get_item(0).unwrap().into_parts();
        let (cam2, rgbd_image2) = dataset.get_item(3).unwrap().into_parts();
        let mut source = RangeImage::from_rgbd_image(&cam1, &rgbd_image1);
        let mut target = RangeImage::from_rgbd_image(&cam2, &rgbd_image2);

        source.compute_normals();
        target.compute_normals();

        (PointCloud::from(&source), PointCloud::from(&target))
    }

    /// Test the ICP algorithm.
    #[rstest]
    fn test_icp(sample1: (PointCloud, PointCloud)) {
        let (source_pcl, target_pcl) = sample1;

        let mut params = IcpParams::default();
        params.weight = 0.25;
        let transform = Icp::new(params, &target_pcl).align(&source_pcl);

        let aligned_source_pcl = &transform * &source_pcl;
        write_ply("tests/data/out-icp1.ply", &aligned_source_pcl.into())
            .expect("Unable to write result");
    }
}
