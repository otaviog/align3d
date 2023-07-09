use super::cost_function::PointPlaneDistance;
use super::icp_params::IcpParams;
use crate::{
    kdtree::R3dTree,
    optim::GaussNewton,
    pointcloud::PointCloud,
    transform::{LieGroup, Transform},
    trig,
};
use itertools::izip;
use num::Float;

/// Standard Iterative Closest Point (ICP) algorithm for aligning two point clouds.
/// This implementation uses the point-to-plane distance.
pub struct Icp<'target_lt> {
    // Parameters of the ICP algorithm.
    pub params: IcpParams,
    // Initial transformation to start the algorithm. Default is the identity.
    pub initial_transform: Transform,
    target: &'target_lt PointCloud,
    kdtree: R3dTree,
}

impl<'target> Icp<'target> {
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
            kdtree: R3dTree::new(&target.points.view()),
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
        let target_normals = self
            .target
            .normals
            .as_ref()
            .expect("Please, the target point cloud should have normals.");
        let source_normals = source
            .normals
            .as_ref()
            .expect("Please, the source point cloud should have normals.");
        let mut optim_transform = Transform::eye();
        let mut optimizer = GaussNewton::<6>::new();
        let geom_cost = PointPlaneDistance {};

        let max_distance_sqr = self.params.max_distance * self.params.max_distance;

        let mut best_residual = Float::infinity();
        let mut best_transform = optim_transform.clone();
        for _ in 0..self.params.max_iterations {
            for (source_point, source_normal) in izip!(source.points.iter(), source_normals.iter())
            {
                let source_point = optim_transform.transform_vector(source_point);
                let source_normal = optim_transform.transform_normal(source_normal);

                let (found_index, found_sqr_distance) = self.kdtree.nearest(&source_point);
                if found_sqr_distance > max_distance_sqr {
                    continue;
                }

                let target_normal = target_normals[found_index];

                if trig::angle_between_vectors(&source_normal, &target_normal)
                    > self.params.max_normal_angle
                {
                    continue;
                }

                let target_point = self.target.points[found_index];

                let (residual, jacobian) =
                    geom_cost.jacobian(&source_point, &target_point, &target_normal);

                optimizer.step(residual, &jacobian);
            }

            let residual = optimizer.mean_squared_residual();
            optimizer.weight(self.params.weight);
            let update = optimizer.solve().unwrap();
            optim_transform = &Transform::exp(&LieGroup::Se3(update)) * &optim_transform;
            optimizer.reset();

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
        metrics::TransformMetrics,
        unit_test::{sample_pcl_ds1, TestPclDataset},
    };

    /// Test the ICP algorithm.
    #[rstest]
    fn test_icp(sample_pcl_ds1: TestPclDataset) {
        let target_pcl = sample_pcl_ds1.get(0);
        let source_pcl = sample_pcl_ds1.get(1);

        let actual = Icp::new(
            IcpParams {
                max_iterations: 5,
                ..Default::default()
            },
            &target_pcl,
        )
        .align(&source_pcl);
        let gt_transform = sample_pcl_ds1.get_ground_truth(1, 0);
        assert!(TransformMetrics::new(&actual, &gt_transform).angle.abs() < 0.1);
    }
}
