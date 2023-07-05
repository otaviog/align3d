use nalgebra::Vector3;
use ndarray::Array2;

use crate::{camera::PinholeCamera, utils::window_iter::window};
use rayon::prelude::*;

pub struct IndexMap {
    pub map: Array2<i64>,
    pub scale: usize,
}

impl IndexMap {
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
            if let Some((u, v)) = camera.project_if_visible(&point) {
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
            if let Some((u, v)) = camera.project_if_visible(&point) {
                let (u, v) = (u as usize * self.scale, v as usize * self.scale);
                Some((v, u, id as i64))
            } else {
                None
            }
        }).collect::<Vec<_>>() {
            self.map[(v, u)] = id;
        }
        
        
        // .collect::<Vec<_>>().iter().for_each(|(v, u, id)| {
        //     self.map[(v, u)] = id;
        // });
    }

    // pub fn iter_indexed<'a>(&self) -> impl Iterator<Item = (usize, usize, i64)> + 'a {
    //     self.map.view().indexed_iter().map(|((v, u), id)| (v, u, *id))
    // }

    pub fn get(&self, u: usize, v: usize) -> Option<usize> {
        let id = self.map[(v * self.scale, u * self.scale)];
        if id >= 0 {
            Some(id as usize)
        } else {
            None
        }
    }

    pub fn window<'a>(&'a self, u: usize, v: usize, n: usize) -> impl Iterator<Item = i64> + 'a {
        window(self.map.view(), u, v, n).filter(|id| *id >= 0)
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
