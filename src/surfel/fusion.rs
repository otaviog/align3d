use std::collections::HashMap;

use bitarray::BitArray;
use itertools::izip;
use nalgebra::Vector2;
use ndarray::{Array2, Axis};

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

    pub fn integrate(
        &mut self,
        model: &mut SurfelModel,
        range_image: &RangeImage,
        camera: &PinholeCamera,
    ) -> FusionSummary {
        self.indexmap.render_indices(model.position_iter(), camera);
        let mut model_map =
            Array2::<Option<(usize, Surfel)>>::from_elem(self.indexmap.map.dim(), None);
        self.indexmap.map.indexed_iter().for_each(|((v, u), id)| {
            if *id > -1 {
                model_map[(v, u)] = Some((*id as usize, model.get(*id as usize).unwrap()));
            }
        });

        enum SurfelUpdateCommand {
            Update(usize, Surfel),
            Add(Surfel),
        }

        let surfel_builder = SurfelBuilder::new(&camera.intrinsics);
        let normals = range_image.normals.as_ref().unwrap();
        let colors = range_image.colors.as_ref().unwrap();

        const PROCESS_CHUNK_SIZE: usize = 1024 * 10;
        let add_update_list: Vec<SurfelUpdateCommand> = izip!(
            range_image
                .mask
                .view()
                .to_shape(range_image.len())
                .unwrap()
                .axis_chunks_iter(Axis(0), PROCESS_CHUNK_SIZE),
            range_image
                .points
                .view()
                .to_shape(range_image.len())
                .unwrap()
                .axis_chunks_iter(Axis(0), PROCESS_CHUNK_SIZE),
            normals
                .view()
                .to_shape(range_image.len())
                .unwrap()
                .axis_chunks_iter(Axis(0), PROCESS_CHUNK_SIZE),
            colors
                .view()
                .to_shape(range_image.len())
                .unwrap()
                .axis_chunks_iter(Axis(0), PROCESS_CHUNK_SIZE)
        )
        .enumerate()
        .par_bridge()
        .map(
            |(chunk_i, (mask_chunk, point_chunk, normal_chunk, color_chunk))| {
                izip!(
                    mask_chunk.iter(),
                    point_chunk.iter(),
                    normal_chunk.iter(),
                    color_chunk.iter()
                )
                .enumerate()
                .filter_map(|(i, (&mask, point, normal, color))| {
                    if mask != 0 {
                        let point_index = chunk_i * PROCESS_CHUNK_SIZE + i;
                        let u = point_index % range_image.width();
                        let v = point_index / range_image.width();
                        Some((u, v, point, normal, color))
                    } else {
                        None
                    }
                })
                .map(|(u, v, &point, &normal, &color)| {
                    let ri_surfel =
                        camera
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

                    let found_merge_surfel = window(
                        model_map.view(),
                        u * self.indexmap.scale,
                        v * self.indexmap.scale,
                        8,
                    )
                    .flatten()
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
                    });

                    if let Some((found_surfel_index, _ray_dist, model_surfel)) = found_merge_surfel
                    {
                        // if ri_surfel.radius >= model_surfel.radius * 1.5
                        SurfelUpdateCommand::Update(
                            found_surfel_index,
                            model_surfel.merge(&ri_surfel),
                        )
                    } else {
                        SurfelUpdateCommand::Add(ri_surfel)
                    }
                })
                .collect::<Vec<_>>()
            },
        )
        .flatten()
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

        let mut update_count = 0;
        let mut add_count = 0;

        struct SurfelGpuUpdateCommand {
            model_index: usize,
            surfel: Surfel,
        }

        let mut gpu_update_commands = Vec::with_capacity(add_update_list.len());
        {
            let mut cpu_writer = model.get_cpu_writer();

            {
                for surfel_update in &add_update_list {
                    match surfel_update {
                        SurfelUpdateCommand::Update(model_index, surfel) => {
                            update_count += 1;
                            cpu_writer.update(*model_index, surfel);
                            gpu_update_commands.push(SurfelGpuUpdateCommand {
                                model_index: *model_index,
                                surfel: *surfel,
                            });
                        }
                        SurfelUpdateCommand::Add(surfel) => {
                            add_count += 1;
                            let new_index = cpu_writer.add(surfel);
                            gpu_update_commands.push(SurfelGpuUpdateCommand {
                                model_index: new_index,
                                surfel: *surfel,
                            });
                        }
                    }
                }

                for id in &free_list {
                    cpu_writer.free(*id);
                }
            }
        }

        {
            let mut gpu_writer = model.lock_gpu();
            let mut writer = gpu_writer.get_writer();

            for surfel_update in &gpu_update_commands {
                writer.update(surfel_update.model_index, &surfel_update.surfel);
            }

            for id in &free_list {
                writer.unmask(*id);
            }
        }

        FusionSummary {
            num_added: add_count,
            num_updated: update_count,
            num_removed: free_list.len(),
        }
    }

    pub fn integrate_with_sparse_features(
        &mut self,
        model: &mut SurfelModel,
        range_image: &RangeImage,
        camera: &PinholeCamera,
        sparse_features: HashMap<(usize, usize), BitArray<64>>,
    ) -> FusionSummary {
        self.indexmap.render_indices(model.position_iter(), camera);
        let mut model_map =
            Array2::<Option<(usize, Surfel)>>::from_elem(self.indexmap.map.dim(), None);
        self.indexmap.map.indexed_iter().for_each(|((v, u), id)| {
            if *id > -1 {
                model_map[(v, u)] = Some((*id as usize, model.get(*id as usize).unwrap()));
            }
        });
        const PROCESS_CHUNK_SIZE: usize = 1024 * 10;

        struct SourceSurfel {
            surfel: Surfel,
            u: usize,
            v: usize,
            sparse_feature: Option<BitArray<64>>,
        }

        let surfel_builder = SurfelBuilder::new(&camera.intrinsics);
        let normals = range_image.normals.as_ref().unwrap();
        let colors = range_image.colors.as_ref().unwrap();
        let source_surfels = izip!(
            range_image.mask.view().to_shape(range_image.len()).unwrap(),
            range_image
                .points
                .view()
                .to_shape(range_image.len())
                .unwrap(),
            normals.view().to_shape(range_image.len()).unwrap(),
            colors.view().to_shape(range_image.len()).unwrap()
        )
        .enumerate()
        .filter_map(|(point_index, (mask, point, normal, color))| {
            if mask != 0 {
                let u = point_index % range_image.width();
                let v = point_index / range_image.width();
                let surfel = camera
                    .camera_to_world
                    .transform(surfel_builder.from_range_pixel(
                        point,
                        normal,
                        color,
                        Vector2::new(u as f32, v as f32),
                        self.timestamp,
                    ));

                Some(SourceSurfel {
                    surfel,
                    u,
                    v,
                    sparse_feature: sparse_features.get(&(u, v)).map(|v| v.clone()),
                })
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

        enum SurfelUpdateCommand {
            Update(usize, Surfel, Option<BitArray<64>>),
            Add(Surfel, Option<BitArray<64>>),
        }
        struct SurfelGpuUpdateCommand {
            model_index: usize,
            surfel: Surfel,
        }

        let add_update_list = source_surfels
            .chunks(PROCESS_CHUNK_SIZE)
            .par_bridge()
            .map(|source_chunk| {
                source_chunk
                    .iter()
                    .map(|ri_surfel| {
                        let ray_cam_ri =
                            ri_surfel.surfel.position - camera.camera_to_world.translation();
                        let ray_cam_ri_norm = 1.0 / ray_cam_ri.norm();

                        let u = ri_surfel.u;
                        let v = ri_surfel.v;

                        let found_merge_surfel = window(
                            model_map.view(),
                            u * self.indexmap.scale,
                            v * self.indexmap.scale,
                            8,
                        )
                        .flatten()
                        .filter_map(|(index, model_surfel)| {
                            if (ri_surfel.surfel.position - model_surfel.position).norm()
                                > (model_surfel.radius + ri_surfel.surfel.radius) * 5.0
                            {
                                return None;
                            }

                            if angle_between_normals(&ri_surfel.surfel.normal, &model_surfel.normal)
                                < 30.0_f32.to_radians()
                            {
                                let ray_dist = model_surfel.position.cross(&ray_cam_ri).norm()
                                    * ray_cam_ri_norm;
                                Some((index, ray_dist, model_surfel))
                            } else {
                                None
                            }
                        })
                        .min_by_key(
                            |(_index, ray_distance, _model_surfel)| {
                                ordered_float::OrderedFloat(*ray_distance)
                            },
                        );

                        if let Some((found_surfel_index, _ray_dist, model_surfel)) =
                            found_merge_surfel
                        {
                            // Check if necessary: if ri_surfel.radius >= model_surfel.radius * 1.5
                            SurfelUpdateCommand::Update(
                                found_surfel_index,
                                model_surfel.merge(&ri_surfel.surfel),
                                ri_surfel.sparse_feature.clone(),
                            )
                        } else {
                            SurfelUpdateCommand::Add(
                                ri_surfel.surfel,
                                ri_surfel.sparse_feature.clone(),
                            )
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
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

        let mut update_count = 0;
        let mut add_count = 0;

        let mut gpu_update_commands = Vec::with_capacity(add_update_list.len());
        let mut sparse_features = Vec::with_capacity(add_update_list.len() / 4);
        {
            let mut cpu_writer = model.get_cpu_writer();
            {
                for surfel_update in &add_update_list {
                    let (model_index, surfel, sparse_feature) = match surfel_update {
                        SurfelUpdateCommand::Update(model_index, surfel, sparse_feature) => {
                            update_count += 1;
                            cpu_writer.update(*model_index, &surfel);
                            (*model_index, surfel, sparse_feature)
                        }
                        SurfelUpdateCommand::Add(surfel, sparse_feature) => {
                            add_count += 1;
                            let model_index = cpu_writer.add(&surfel);
                            (model_index, surfel, sparse_feature)
                        }
                    };

                    gpu_update_commands.push(SurfelGpuUpdateCommand {
                        model_index,
                        surfel: surfel.clone(),
                    });

                    if let Some(v) = sparse_feature {
                        sparse_features.push((model_index, v.clone()));
                    }
                }

                for id in &free_list {
                    cpu_writer.free(*id);
                }
            }
        }

        {
            let mut gpu_writer = model.lock_gpu();
            let mut writer = gpu_writer.get_writer();

            for surfel_update in &gpu_update_commands {
                writer.update(surfel_update.model_index, &surfel_update.surfel);
            }

            for id in &free_list {
                writer.unmask(*id);
            }
        }

        for (model_index, v) in sparse_features {
            model.sparse_features.insert(model_index, v.clone());
        }

        FusionSummary {
            num_added: add_count,
            num_updated: update_count,
            num_removed: free_list.len(),
        }
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
    #[ignore]
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
            let camera = sample_range_img_ds2.pinhole_camera(i);
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

    #[rstest]
    #[ignore]
    fn test_surfel_fusion_sparse(sample_range_img_ds2: TestRangeImageDataset) {
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
            let camera = sample_range_img_ds2.pinhole_camera(i);
            let start = Instant::now();

            // guard
            let summary = surfel_fusion.integrate_with_sparse_features(
                &mut model,
                &range_image,
                &camera,
                HashMap::new(),
            );

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
