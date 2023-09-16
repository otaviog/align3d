use std::thread;

use akaze::Akaze;
use align3d::{
    bilateral::BilateralFilter,
    bin_utils::dataset::load_dataset,
    camera::{CameraIntrinsics, PinholeCamera},
    icp::{multiscale::MultiscaleAlign, MsIcpParams},
    image::{IntoImageRgb8, RgbdFrame},
    io::dataset::{SubsetDataset},
    metrics::TransformMetrics,
    range_image::{RangeImage, RangeImageBuilder},
    surfel::{SurfelFusion, SurfelFusionParameters, SurfelModel},
    trajectory::Trajectory,
    trajectory_builder::TrajectoryBuilder,
    transform::Transform,
    viz::{node::MakeNode, GeoViewer, Manager},
};
use clap::Parser;
use vulkano::memory::allocator::MemoryAllocator;

#[derive(Parser)]
struct Args {
    /// Format of the dataset: slamtb, ilrgbd, or tum
    format: String,
    /// Path to the dataset directory
    dataset: String,
    /// Maximum number of frames to process
    max_frames: Option<usize>,
    /// Shows the point clouds with the predicted odometry
    #[clap(long, short, action)]
    show: bool,
}

enum OdometryMode {
    FrameToFrame,
    FrameToModel,
    GroundTruth,
}

struct SubmapSlam {
    intrinsics: CameraIntrinsics,
    range_processing: RangeImageBuilder,
    icp_params: MsIcpParams,
    traj_builder: TrajectoryBuilder,
    model: SurfelModel,
    fusion: SurfelFusion,
    frame_count: usize,
    prev_range_image: Option<Vec<RangeImage>>,
    gt_trajectory: Option<Trajectory>,
}

impl SubmapSlam {
    fn create(
        memory_allocator: &(impl MemoryAllocator + ?Sized),
        intrinsics: CameraIntrinsics,
        gt_trajectory: Option<Trajectory>,
    ) -> Self {
        let (width, height) = (intrinsics.width, intrinsics.height);
        SubmapSlam {
            intrinsics,
            range_processing: RangeImageBuilder::default()
                .with_bilateral_filter(Some(BilateralFilter::default()))
                .with_normals(true)
                .with_intensity(true),
            icp_params: MsIcpParams::default(),
            traj_builder: TrajectoryBuilder::default(),
            model: SurfelModel::new(memory_allocator, 4_000_000),
            fusion: SurfelFusion::new(width, height, 4, SurfelFusionParameters::default()),
            frame_count: 0,
            prev_range_image: None,
            gt_trajectory,
        }
    }

    fn get_transform(&mut self, range_image: &Vec<RangeImage>, mode: OdometryMode) -> Transform {
        match mode {
            OdometryMode::FrameToFrame => {
                if let Some(prev_range_image) = self.prev_range_image.as_ref() {
                    let icp =
                        MultiscaleAlign::new(self.icp_params.clone(), prev_range_image).unwrap();
                    icp.align(range_image)
                } else {
                    Transform::eye()
                }
            }
            OdometryMode::FrameToModel => {
                if let Some(camera_to_world) = self.traj_builder.current_camera_to_world() {
                    let camera = PinholeCamera::new(self.intrinsics.clone(), camera_to_world);
                    let model_frame = self.model.render_to_range_image(&camera);

                    let mut model_frame = model_frame.pyramid(3, 0.5);
                    for frame in &mut model_frame {
                        frame.compute_intensity();
                        frame.compute_intensity_map();
                    }

                    let icp = MultiscaleAlign::new(self.icp_params.clone(), &model_frame).unwrap();
                    icp.align(range_image)
                } else {
                    Transform::eye()
                }
            }
            OdometryMode::GroundTruth => {
                if self.frame_count > 0 && self.gt_trajectory.is_some() {
                    self.gt_trajectory
                        .as_ref()
                        .unwrap()
                        .get_relative_transform(self.frame_count, self.frame_count - 1)
                        .unwrap()
                } else {
                    Transform::eye()
                }
            }
        }
    }

    fn process_frame(&mut self, rgbd_frame: RgbdFrame) {
        let range_image = self.range_processing.build(rgbd_frame);

        let transform = self.get_transform(&range_image, OdometryMode::FrameToFrame);
        if self.frame_count > 0 && self.gt_trajectory.is_some() {
            let gt_transform = self
                .gt_trajectory
                .as_ref()
                .unwrap()
                .get_relative_transform(self.frame_count, self.frame_count - 1)
                .unwrap();
            let metrics = TransformMetrics::new(&gt_transform, &transform);
            println!(
                "Frame {} relative error: {} translation, {} rotation (degrees)",
                self.frame_count,
                metrics.translation,
                metrics.angle.to_degrees()
            );
        }
        self.traj_builder
            .accumulate(&transform, Some(self.frame_count as f32));
        self.fusion.integrate(
            &mut self.model,
            range_image.first().unwrap(),
            &PinholeCamera::new(
                self.intrinsics.clone(),
                self.traj_builder.current_camera_to_world().unwrap(),
            ),
        );

        self.prev_range_image = Some(range_image);
        self.frame_count += 1;
    }
}

fn main() {
    let args = Args::parse();
    let dataset = {
        let mut dataset = load_dataset(args.format, args.dataset).unwrap();
        if let Some(max_frames) = args.max_frames {
            dataset = Box::new(SubsetDataset::new(dataset, (0..max_frames).collect()));
        }
        dataset
    };

    let mut manager = Manager::default();
    let (camera_intrinsics, _) = dataset.camera(0);
    let mut slam = SubmapSlam::create(
        &manager.memory_allocator,
        camera_intrinsics.clone(),
        dataset.trajectory(),
    );

    let render_model = slam.model.vk_data.clone();
    let node = render_model.make_node(&mut manager);
    let mut geo_viewer = GeoViewer::from_manager(manager);

    let slam_thread = thread::spawn(move || {
        for i in 0..dataset.len() {
            slam.process_frame(dataset.get(i).expect("Failed to load frame {i}"));
        }
    });

    geo_viewer.add_node(node);
    geo_viewer.run();
    slam_thread.join().unwrap();
}
