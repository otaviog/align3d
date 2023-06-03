use core::panic;

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
        panic!("Not implemented: scale");
        self.map.fill(-1);
        for (id, point) in model_points {
            if let Some((u, v)) = camera.project_if_visible(&point) {
                let (u, v) = ((u + 0.5) as usize, (v + 0.5) as usize);
                self.map[(u, v)] = id as i64;
            }
        }
    }

    pub fn get(&self, u: usize, v: usize) -> Option<usize> {
        let id = self.map[(u, v)];
        if id >= 0 {
            Some(id as usize)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::UnitVector3;

    //use crate::{transform::TransformBuilder, surfel::surfel_model::{SurfelModel, Surfel, RimageSurfelBuilder}};

    use super::*;

    #[test]
    fn test_indexmap() {
        // let mut camera = FullCamera::from_simple_intrinsic(
        //     525.0,
        //     525.0,
        //     319.5,
        //     239.5,
        //     TransformBuilder::default()
        //         .axis_angle(
        //             UnitVector3::new_normalize(Vector3::new(4.0, 1.0, 0.0)),
        //             35.0_f32.to_radians(),
        //         )
        //         .build(),
        //     640,
        //     480,
        // );
        //
        // let mut indexmap = IndexMap::new(640, 480, 4);
        //
        // let builder = RimageSurfelBuilder::new(&camera);
        //
        // let model = SurfelModel::new(3);
        //
        // let model_points =
        //     Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        //         .unwrap();
        // indexmap.render_indices(&model_points, &camera);
        // assert_eq!(indexmap.map[(320, 240)], 0);
        // assert_eq!(indexmap.map[(321, 241)], 1);
        // assert_eq!(indexmap.map[(319, 239)], 2);
    }
}
