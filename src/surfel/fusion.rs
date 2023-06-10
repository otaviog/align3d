use std::{rc::Rc, sync::Mutex, sync::Arc};

use nalgebra::Vector2;
use ndarray::s;

use super::{
    indexmap::IndexMap,
    surfel_model::{SurfelModel, SurfelModelWriteCommands},
    surfel_type::{RimageSurfelBuilder},
};
use crate::utils::access::ToVector3;
use crate::{camera::PinholeCamera, range_image::RangeImage};

pub struct SurfelFusion {
    indexmap: IndexMap,
    timestamp: i32,
}

impl SurfelFusion {
    pub fn new(map_width: usize, map_height: usize, map_scale: usize) -> Self {
        SurfelFusion {
            indexmap: IndexMap::new(map_width, map_height, map_scale),
            timestamp: 0,
        }
    }

    pub fn integrate(
        &mut self,
        model: &mut SurfelModel,
        range_image: &RangeImage,
        camera: &PinholeCamera,
    ) {
        let mut write_commands = SurfelModelWriteCommands::new();
        {
            let model_reader = model.read().unwrap();
            self.indexmap
                .render_indices(model_reader.position_iter(), camera);

            let surfel_builder = RimageSurfelBuilder::new(&camera.intrinsics);
            let range_normals = range_image.normals.as_ref().unwrap();
            let range_colors = range_image.colors.as_ref().unwrap();

            for v in 0..range_image.height() {
                for u in 0..range_image.width() {
                    if !range_image.is_valid(u, v) {
                        continue;
                    }

                    let ri_point = range_image.points.slice(s![v, u, ..]).to_vector3();
                    let ri_normal = range_normals.slice(s![v, u, ..]).to_vector3();
                    let ri_color = range_colors.slice(s![v, u, ..]).to_vector3();

                    let ri_surfel = surfel_builder.build(
                        ri_point,
                        ri_normal,
                        ri_color,
                        Vector2::new(u as f32, v as f32),
                        self.timestamp,
                    );

                    if let Some(id) = self.indexmap.get(u, v) {
                        if let Some(model_surfel) = model_reader.get(id) {
                            if (model_surfel.position - ri_point).norm() < 0.1 {
                                write_commands.update.push((id, model_surfel.merge(&ri_surfel, 0.5, 0.5)));
                            } else {
                                write_commands.add.push(ri_surfel);
                            }
                        }
                    } else {
                        write_commands.add.push(ri_surfel);
                    }
                }

                for (id, _age, conf) in model_reader.age_confidence_iter() {
                    if conf < 100.0 {
                        write_commands.free.push(id);
                    }
                }

            }
        }


        model.write_commands = Arc::new(Mutex::new(write_commands));
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use std::time::Instant;

    use crate::{
        unit_test::{sample_range_img_ds2, TestRangeImageDataset},
        viz::Manager,
    };

    use super::*;

    #[rstest]
    #[ignore]
    fn test_surfel_fusion(sample_range_img_ds2: TestRangeImageDataset) {
        let manager = Manager::default();

        let mut model = SurfelModel::new(&manager.memory_allocator, 400_000);

        let mut surfel_fusion = SurfelFusion::new(640, 480, 4);

        let mut duration_accum = 0.0;
        const NUM_ITER: usize = 5;
        for i in 0..NUM_ITER {
            let range_image = sample_range_img_ds2.get(i).unwrap();
            let (intrinsics, transform) = sample_range_img_ds2.camera(i).clone();
            let start = Instant::now();

            let camera = PinholeCamera::new(intrinsics, transform.unwrap());
            surfel_fusion.integrate(&mut model, &range_image, &camera);
            duration_accum += start.elapsed().as_secs_f64();
        }

        println!(
            "Time elapsed in is_visible() is: {:?}",
            duration_accum / NUM_ITER as f64
        );
    }
}
