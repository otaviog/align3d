use nalgebra::{Vector2, Vector3};
use ndarray::{s, Array2};
// use rerun::{MsgSender, RecordingStreamBuilder};

use super::{indexmap::IndexMap, surfel_model::SurfelModel, surfel_type::SurfelBuilder};
use crate::surfel::Surfel;
use crate::transform::TransformableMove;
use crate::trig::angle_between_normals;
use crate::utils::access::ToVector3;
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

pub struct RangeImage2 {
    pub points: Array2<Vector3<f32>>,
    pub normals: Array2<Vector3<f32>>,
    pub colors: Array2<Vector3<u8>>,
    pub valid_points: Array2<u8>,
    pub valid_points_count: usize,
}

impl From<&RangeImage> for RangeImage2 {
    fn from(range_image: &RangeImage) -> Self {
        let mut points: Array2<Vector3<f32>> =
            Array2::zeros((range_image.height(), range_image.width()));
        let mut normals: Array2<Vector3<f32>> =
            Array2::zeros((range_image.height(), range_image.width()));
        let mut colors: Array2<Vector3<u8>> =
            Array2::zeros((range_image.height(), range_image.width()));

        range_image.mask.indexed_iter().for_each(|((v, u), m)| {
            if *m > 0 {
                let point = range_image.get_point(v, u);
                let normal = range_image
                    .normals
                    .as_ref()
                    .unwrap()
                    .slice(s![v, u, ..])
                    .to_vector3();
                let color = range_image
                    .colors
                    .as_ref()
                    .unwrap()
                    .slice(s![v, u, ..])
                    .to_vector3();
                points[[v, u]] = point.unwrap();
                normals[[v, u]] = normal;
                colors[[v, u]] = color;
            }
        });

        RangeImage2 {
            points,
            normals,
            colors,
            valid_points: range_image.mask.clone(),
            valid_points_count: range_image.valid_points_count(),
        }
    }
}

impl<'a> RangeImage2 {
    pub fn indexed_iter(
        &'a self,
    ) -> impl Iterator<Item = (usize, usize, Vector3<f32>, Vector3<f32>, Vector3<u8>)> + 'a {
        self.valid_points
            .indexed_iter()
            .filter_map(move |((v, u), m)| {
                if *m > 0 {
                    let point = self.points[[v, u]];
                    let normal = self.normals[[v, u]];
                    let color = self.colors[[v, u]];
                    Some((v, u, point, normal, color))
                } else {
                    None
                }
            })
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
        range_image: &RangeImage2,
        camera: &PinholeCamera,
    ) -> FusionSummary {
        let mut update_list = Vec::with_capacity(range_image.valid_points_count);
        let mut add_list = Vec::with_capacity(range_image.valid_points_count);
        let mut free_list = Vec::with_capacity(40000);

        //let recording = RecordingStreamBuilder::new("minimal").connect(rerun::default_server_addr())?;

        {
            self.indexmap.render_indices(model.position_iter(), camera);
            let mut model_map =
                Array2::<Option<(usize, Surfel)>>::from_elem(self.indexmap.map.dim(), None);
            self.indexmap.map.indexed_iter().for_each(|((v, u), id)| {
                if *id > -1 {
                    model_map[(v, u)] = Some((*id as usize, model.get(*id as usize).unwrap()));
                }
            });

            let surfel_builder = SurfelBuilder::new(&camera.intrinsics);
            range_image
                .indexed_iter()
                .for_each(|(v, u, point, normal, color)| {
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

                        if angle_between_normals(&ri_surfel.normal, &model_surfel.normal) < 30.0_f32.to_radians()
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
                        //if  ri_surfel.radius >= model_surfel.radius*1.5 {
                        if true {
                            update_list.push((index, model_surfel.merge(&ri_surfel)));
                        }
                    } else {
                        add_list.push(ri_surfel);
                    }
                });

            for (id, age, conf) in model.age_confidence_iter() {
                if (self.timestamp - age) > self.params.age_remove_threshold
                    && conf < self.params.confidence_remove_threshold
                {
                    free_list.push(id);
                }
            }

            self.timestamp += 1;
        }

        let mut writer = model.write().unwrap();
        for (id, surfel) in &update_list {
            writer.update(*id, &surfel);
        }

        for surfel in &add_list {
            writer.add(surfel);
        }

        for id in &free_list {
            writer.free(*id);
        }

        return FusionSummary {
            num_added: add_list.len(),
            num_updated: update_list.len(),
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
            let range_image = RangeImage2::from(&sample_range_img_ds2.get(i).unwrap());
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
