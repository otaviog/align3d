use align3d::{
    bilateral::BilateralFilter,
    icp::{multiscale::MultiscaleAlign, MsIcpParams},
    io::dataset::SubsetDataset,
    metrics::TransformMetrics,
    range_image::{RangeImage, RangeImageBuilder},
    trajectory::TrajectoryBuilder,
    viz::rgbd_dataset_viewer::RgbdDatasetViewer, transform::Transform,
};

use clap::Parser;
use examples::load_dataset;
use kdam::tqdm;

#[derive(Parser)]
struct Args {
    /// Format of the dataset: ilrgbd, or tum
    format: String,
    /// Path to the dataset directory
    dataset: String,
    /// Maximum number of frames to process
    max_frames: Option<usize>,
    /// Shows the point clouds with the predicted odometry
    #[clap(long, short, action)]
    show: bool,
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

    let range_processing =
        RangeImageBuilder::default().with_bilateral_filter(Some(BilateralFilter::default()));

    let icp_params = MsIcpParams::default();

    let mut trajectory_build = TrajectoryBuilder::with_start(Transform::eye(), 0.0);
    let mut last_frame: Vec<RangeImage> = range_processing.build(dataset.get(0).unwrap());

    for i in tqdm!(
        1..dataset.len(),
        total = dataset.len() - 1,
        desc = "Processing frames"
    ) {
        let current_frame = range_processing.build(dataset.get(i).unwrap());
        let icp = MultiscaleAlign::new(icp_params.clone(), &last_frame).unwrap();
        let transform = icp.align(&current_frame);
        trajectory_build.accumulate(&transform, Some(i as f32));
        last_frame = current_frame;
    }

    let pred_trajectory = trajectory_build.build();
    let gt_trajectory = &dataset.trajectory().unwrap().first_frame_at_origin();

    let metrics = TransformMetrics::mean_trajectory_error(&pred_trajectory, gt_trajectory).unwrap();
    println!("Mean trajectory error: {metrics}");

    if args.show {
        RgbdDatasetViewer::new(dataset)
            .with_trajectory(pred_trajectory)
            .run();
    }
}
