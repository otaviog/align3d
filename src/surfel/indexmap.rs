use nalgebra::Vector3;
use ndarray::Array2;

use crate::{camera::PinholeCamera, utils::window_iter::window};
use rayon::prelude::*;

/// IndexMap is a 2D array of indices of surfels in the surfel model.
/// This is used for fast neighbor search during fusion of a new frame.
pub struct IndexMap {
    /// The map is a 2D array of indices of surfels in the surfel model.
    pub map: Array2<i64>,
    /// The map is scaled by a factor of `scale` so we avoid collisions between surfels.
    pub scale: usize,
}

impl IndexMap {
    /// Create a new IndexMap with the given width, height and scale.
    pub fn new(width: usize, height: usize, scale: usize) -> Self {
        IndexMap {
            map: Array2::zeros((height * scale, width * scale)),
            scale,
        }
    }

    pub fn render_indices<T: Iterator<Item = (usize, Vector3<f32>)>>(
        &mut self,
        model_points: T,
        camera: &PinholeCamera,
    ) {
        self.map.fill(-1);
        for (id, point) in model_points {
            if let Some((u, v)) = camera.project_to_image(&point) {
                let (u, v) = (u as usize * self.scale, v as usize * self.scale);
                self.map[(v, u)] = id as i64;
            }
        }
    }

    pub fn render_indices_par<T>(&mut self, model_points: T, camera: &PinholeCamera)
    where
        T: ParallelIterator<Item = (usize, Vector3<f32>)>,
    {

        self.map.fill(-1);
        for (v, u, id) in model_points.filter_map(|(id, point)| {
            if let Some((u, v)) = camera.project_to_image(&point) {
                let (u, v) = (u as usize * self.scale, v as usize * self.scale);
                Some((v, u, id as i64))
            } else {
                None
            }
        }).collect::<Vec<_>>() {
            self.map[(v, u)] = id;
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "IndexMap: {}x{}x{} (w x h x scale), mean: {}",
            self.map.shape()[1] / self.scale,
            self.map.shape()[0] / self.scale,
            self.scale,
            self.map.mean().unwrap()
        )
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::{
        unit_test::{sample_pcl_ds1, TestPclDataset},
        utils::access::ToVector3,
    };
    use ndarray::Axis;
    use rstest::rstest;

    #[rstest]
    fn test_indexmap_par(sample_pcl_ds1: TestPclDataset) {
        let pcl = sample_pcl_ds1.get(13);
        let mut indexmap = IndexMap::new(640, 480, 4);
        let (intrinsics, extrinsics) = sample_pcl_ds1.camera(0);
        let camera = PinholeCamera::new(intrinsics, extrinsics.unwrap());

        indexmap.render_indices_par(
            pcl.points
                .axis_iter(Axis(0))
                .into_par_iter()
                .enumerate()
                .map(|(id, point)| (id, point.to_vector3())),
            &camera,
        );
        
        let start = Instant::now();
        indexmap.render_indices_par(
            pcl.points
                .axis_iter(Axis(0))
                .into_par_iter()
                .enumerate()
                .map(|(id, point)| (id, point.to_vector3())),
            &camera,
        );

        println!(
            "{}, time: {} seconds",
            indexmap.summary(),
            start.elapsed().as_secs_f64()
        );
    }

    #[rstest]
    fn test_indexmap(sample_pcl_ds1: TestPclDataset) {
        let pcl = sample_pcl_ds1.get(13);
        let mut indexmap = IndexMap::new(640, 480, 4);
        let (intrinsics, extrinsics) = sample_pcl_ds1.camera(0);
        let camera = PinholeCamera::new(intrinsics, extrinsics.unwrap());

        indexmap.render_indices(
            pcl.points
                .axis_iter(Axis(0))
                .enumerate()
                .map(|(id, point)| (id, point.to_vector3())),
            &camera,
        );
        
        let start = Instant::now();
        indexmap.render_indices(
            pcl.points
                .axis_iter(Axis(0))
                .enumerate()
                .map(|(id, point)| (id, point.to_vector3())),
            &camera,
        );

        println!(
            "{}, time: {} seconds",
            indexmap.summary(),
            start.elapsed().as_secs_f64()
        );
    }
}
