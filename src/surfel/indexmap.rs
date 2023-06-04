use nalgebra::Vector3;
use ndarray::Array2;

use crate::camera::PinholeCamera;

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

    pub fn render_indices(
        &mut self,
        model_points: impl Iterator<Item = (usize, Vector3<f32>)>,
        camera: &PinholeCamera,
    ) {
        self.map.fill(-1);
        for (id, point) in model_points {
            if let Some((u, v)) = camera.project_if_visible(&point) {
                let (u, v) = (
                    u as usize * self.scale,
                    v as usize * self.scale,
                );
                self.map[(v, u)] = id as i64;
            }
        }
    }

    pub fn get(&self, u: usize, v: usize) -> Option<usize> {
        let id = self.map[(v * self.scale, u * self.scale)];
        if id >= 0 {
            Some(id as usize)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{unit_test::{sample_pcl_ds1, TestPclDataset}, utils::access::ToVector3};
    use ndarray::Axis;
    use rstest::rstest;

    #[rstest]
    fn test_indexmap(sample_pcl_ds1: TestPclDataset) {
        let pcl = sample_pcl_ds1.get(0);
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
    }
}
