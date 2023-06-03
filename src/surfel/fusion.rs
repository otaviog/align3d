use nalgebra::Vector2;
use ndarray::s;

use super::{
    indexmap::IndexMap,
    surfel_model::SurfelModel,
    surfel_type::{RimageSurfelBuilder, Surfel},
};
use crate::utils::access::ToVector3;
use crate::{camera::PinholeCamera, range_image::RangeImage};

pub struct SurfelFusion {
    indexmap: IndexMap,
    model: SurfelModel,
    timestamp: i32,
}

impl SurfelFusion {
    pub fn new(model: SurfelModel, map_width: usize, map_height: usize, map_scale: usize) -> Self {
        SurfelFusion {
            indexmap: IndexMap::new(map_width, map_height, map_scale),
            model,
            timestamp: 0,
        }
    }

    pub fn integrate(&mut self, range_image: &RangeImage, camera: &PinholeCamera) {
        let model_reader = self.model.read().unwrap();
        self.indexmap
            .render_indices(model_reader.position_iter(), camera);

        let surfel_builder = RimageSurfelBuilder::new(camera);
        let range_normals = range_image.normals.as_ref().unwrap();
        let range_colors = range_image.colors.as_ref().unwrap();

        let mut update_list = Vec::<(usize, Surfel)>::new();
        let mut add_list = Vec::<Surfel>::new();

        for v in 0..range_image.height() {
            for u in 0..range_image.width() {
                if !range_image.is_valid(u, v) {
                    continue;
                }

                if let Some(id) = self.indexmap.get(u, v) {
                    if let Some(model_surfel) = model_reader.get(id) {
                        let ri_point = range_image.points.slice(s![u, v, ..]).to_vector3();
                        let ri_normal = range_normals.slice(s![u, v, ..]).to_vector3();
                        let ri_color = range_colors.slice(s![u, v, ..]).to_vector3();

                        let ri_surfel = surfel_builder.build(
                            ri_point,
                            ri_normal,
                            ri_color,
                            Vector2::new(u as f32, v as f32),
                            self.timestamp,
                        );

                        if (model_surfel.position - ri_point).norm() < 0.1 {
                            update_list.push((id, model_surfel.merge(&ri_surfel, 0.5, 0.5)));
                        } else {
                            add_list.push(ri_surfel);
                        }
                    }
                }
            }
        }

        let mut free_list = Vec::<usize>::new();
        for (id, age, conf) in model_reader.age_confidence_iter() {
            if conf < 100.0 {
                free_list.push(id);
            }
        }

        drop(model_reader);

        let mut model_writer = self.model.write().unwrap();
        for (id, surfel) in update_list {
            model_writer.update(id, surfel);
        }

        for surfel in add_list {
            model_writer.add(surfel);
        }

        for id in free_list {
            model_writer.free(id);
        }
        drop(model_writer);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use nalgebra::{UnitVector3, Vector3};
    use ndarray::{
        parallel::prelude::{IntoParallelIterator, ParallelIterator},
        Array2, Axis,
    };

    use crate::{
        camera::CameraIntrinsics,
        transform::{Transform, TransformBuilder},
    };

    use super::*;

    fn column_vec(array: &Array2<f32>, col: usize) -> Vector3<f32> {
        let row = array.row(col);
        Vector3::new(row[0], row[1], row[2])
    }

    #[test]
    fn test_surfel_fusion() {
        let mut camera = CameraIntrinsics::from_simple_intrinsic(525.0, 525.0, 319.5, 239.5);
        camera.size(640, 480);
        camera.camera_to_world = Some(
            TransformBuilder::default()
                .axis_angle(
                    UnitVector3::new_normalize(Vector3::new(4.0, 1.0, 0.0)),
                    35.0_f32.to_radians(),
                )
                .build(),
        );

        let points = Array2::<f32>::zeros((600_000, 3));
        let start = Instant::now();
        let transform = camera.camera_to_world.clone().unwrap();
        points.axis_iter(Axis(0)).into_par_iter().for_each(|p| {
            //let v: Vector3<f32> = column_vec(&points, 0);
            let v = unsafe {
                let ptr = p.as_ptr();
                transform.transform_vector(&Vector3::new(*ptr, *ptr.add(1), *ptr.add(2)))
            };
            let is_visible = camera.is_visible(v);
            std::hint::black_box(is_visible);
        });
        let duration = start.elapsed();
        println!("Time elapsed in is_visible() is: {:?}", duration);
    }
}
