use nalgebra::Vector2;
use ndarray::Array2;
use rayon::prelude::*;

use super::{indexmap::IndexMap, surfel_model::SurfelModel, surfel_type::SurfelBuilder};
use crate::surfel::Surfel;
use crate::transform::TransformableMove;
use crate::trig::angle_between_normals;
use crate::utils::window_iter::window;
use crate::{camera::PinholeCamera, range_image::RangeImage};

pub struct SurfelFusionParameters {
    pub confidence_remove_threshold: f32,
    pub age_remove_threshold: u32,
}

impl Default for SurfelFusionParameters {
    fn default() -> Self {
        SurfelFusionParameters {
            confidence_remove_threshold: 15.0,
            age_remove_threshold: 10,
        }
    }
}

pub struct SurfelFusion {
    indexmap: IndexMap,
    timestamp: u32,
    params: SurfelFusionParameters,
}

#[derive(Debug, Clone, Copy)]
pub struct FusionSummary {
    pub num_added: usize,
    pub num_updated: usize,
    pub num_removed: usize,
}

impl SurfelFusion {
    pub fn new(
        map_width: usize,
        map_height: usize,
        map_scale: usize,
        params: SurfelFusionParameters,
    ) -> Self {
        SurfelFusion {
            indexmap: IndexMap::new(map_width, map_height, map_scale),
            timestamp: 0,
            params,
        }
    }

    pub fn integrate<'a, 'x, 'y>(
        &mut self,
        model: &'a mut SurfelModel,
        range_image: &RangeImage,
        camera: &PinholeCamera,
    ) -> FusionSummary where 'a: 'x, 'a: 'y, 'x: 'y {
        enum SurfelUpdate {
            Update(usize, Surfel),
            Add(Surfel),
            Discard,
        }

        //self.indexmap.render_indices_par(model.position_par_iter(), camera);
        self.indexmap.render_indices(model.position_iter(), camera);
        let mut model_map =
            Array2::<Option<(usize, Surfel)>>::from_elem(self.indexmap.map.dim(), None);
        self.indexmap.map.indexed_iter().for_each(|((v, u), id)| {
            if *id > -1 {
                model_map[(v, u)] = Some((*id as usize, model.get(*id as usize).unwrap()));
            }
        });

        let surfel_builder = SurfelBuilder::new(&camera.intrinsics);

        let updates = range_image
            .indexed_iter()
            .par_bridge()
            .map(|(v, u, point, normal, color)| {
                let ri_surfel = camera
                    .camera_to_world
                    .transform(surfel_builder.from_range_pixel(
                        point,
                        normal,
                        color,
                        Vector2::new(u as f32, v as f32),
                        self.timestamp,
                    ));

                let ray_cam_ri = ri_surfel.position - camera.camera_to_world.translation();
                let ray_cam_ri_norm = 1.0 / ray_cam_ri.norm();
                if let Some((index, _ray_dist, model_surfel)) = window(
                    model_map.view(),
                    u * self.indexmap.scale,
                    v * self.indexmap.scale,
                    8,
                )
                .filter_map(|v| v)
                .filter_map(|(index, model_surfel)| {
                    if (ri_surfel.position - model_surfel.position).norm()
                        > (model_surfel.radius + ri_surfel.radius) * 5.0
                    {
                        return None;
                    }

                    if angle_between_normals(&ri_surfel.normal, &model_surfel.normal)
                        < 30.0_f32.to_radians()
                    {
                        let ray_dist =
                            model_surfel.position.cross(&ray_cam_ri).norm() * ray_cam_ri_norm;
                        Some((index, ray_dist, model_surfel))
                    } else {
                        None
                    }
                })
                .min_by_key(|(_index, ray_distance, _model_surfel)| {
                    ordered_float::OrderedFloat(*ray_distance)
                }) {
                    if ri_surfel.radius >= model_surfel.radius * 1.5 {
                        //if true {
                        return SurfelUpdate::Update(index, model_surfel.merge(&ri_surfel));
                    }
                } else {
                    return SurfelUpdate::Add(ri_surfel);
                }
                SurfelUpdate::Discard
            })
            .collect::<Vec<_>>();

        let free_list = model
            .age_confidence_iter()
            .filter_map(|(id, age, conf)| {
                if (self.timestamp - age) > self.params.age_remove_threshold
                    && conf < self.params.confidence_remove_threshold
                {
                    Some(id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        self.timestamp += 1;

        //let mut vk_data = model.vk_data.lock().unwrap();

        //let mut writer = vk_data.write(model.data);

        let mut update_count = 0;
        let mut add_count = 0;
        {
            let mut cpu_writer = model.write_cpu();

            {
                for surfel_update in &updates {
                    match surfel_update {
                        SurfelUpdate::Update(id, surfel) => {
                            update_count += 1;
                            cpu_writer.update(*id, surfel);
                        }
                        SurfelUpdate::Add(surfel) => {
                            add_count += 1;
                            //writer.add(&surfel);
                        }
                        SurfelUpdate::Discard => {}
                    }
                }

                for id in &free_list {
                    //writer.free(*id);
                }
            }
        }

        {
            let mut gpu_writer = model.lock_gpu();
            let mut writer = gpu_writer.write();
            for surfel_update in &updates {
                match surfel_update {
                    SurfelUpdate::Update(id, surfel) => {
                        update_count += 1;
                        writer.update(*id, surfel);
                    }
                    SurfelUpdate::Add(surfel) => {
                        add_count += 1;
                        //writer.add(&surfel);
                    }
                    SurfelUpdate::Discard => {}
                }
            }

            for id in &free_list {
                //writer.free(*id);
            }
        }
        'x: {
            let mut writer_t = model.write::<'a, 'x>();
            'y: {
                let writer = writer_t.get::<'y>();
                // for surfel_update in &updates {
                //     match surfel_update {
                //         SurfelUpdate::Update(id, surfel) => {
                //             update_count += 1;
                //             writer.update(*id, surfel);
                //         }
                //         SurfelUpdate::Add(surfel) => {
                //             add_count += 1;
                //             //writer.add(&surfel);
                //         }
                //         SurfelUpdate::Discard => {}
                //     }
                // }
                drop(writer);
            }

            drop(writer_t);
        }
        return FusionSummary {
            num_added: add_count,
            num_updated: update_count,
            num_removed: free_list.len(),
        };
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use std::{fs::File, time::Instant};

    use crate::{
        unit_test::{sample_range_img_ds2, TestRangeImageDataset},
        viz::Manager,
    };

    use super::*;

    #[rstest]
    //#[ignore]
    fn test_surfel_fusion(sample_range_img_ds2: TestRangeImageDataset) {
        let manager = Manager::default();

        let mut model = SurfelModel::new(&manager.memory_allocator, 1_800_000);

        let mut surfel_fusion = SurfelFusion::new(640, 480, 4, SurfelFusionParameters::default());

        let mut duration_accum = 0.0;
        const NUM_ITER: usize = 4;
        let guard = pprof::ProfilerGuardBuilder::default()
            .frequency(1000)
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()
            .unwrap();
        for i in 0..NUM_ITER {
            let range_image = sample_range_img_ds2.get(i).unwrap();
            let (intrinsics, transform) = sample_range_img_ds2.camera(i).clone();

            let camera = PinholeCamera::new(intrinsics, transform.unwrap());
            let start = Instant::now();

            // guard
            let summary = surfel_fusion.integrate(&mut model, &range_image, &camera);

            duration_accum += start.elapsed().as_secs_f64();
            println!(
                "Iteration: {}, Summary: {:?}, Time elapsed: {:?}",
                i,
                summary,
                start.elapsed()
            );
        }

        if let Ok(report) = guard.report().build() {
            let file = File::create("flamegraph.svg").unwrap();
            report.flamegraph(file).unwrap();
        }
        println!(
            "Time elapsed in is_visible() is: {:?}",
            duration_accum / NUM_ITER as f64
        );
    }
}
