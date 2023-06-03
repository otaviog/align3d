use align3d::{
    bilateral::BilateralFilter,
    bin_utils::dataset::create_dataset_from_string,
    icp::{multiscale::MultiscaleAlign, MsIcpParams},
    io::dataset::SubsetDataset,
    metrics::TransformMetrics,
    range_image::{RangeImage, RangeImageBuilder},
    trajectory_builder::TrajectoryBuilder,
    viz::rgbd_dataset_viewer::RgbdDatasetViewer,
};
use clap::Parser;
use kdam::tqdm;

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

fn main() {
    let args = Args::parse();
    let dataset = {
        let mut dataset = create_dataset_from_string(args.format, args.dataset).unwrap();
        if let Some(max_frames) = args.max_frames {
            dataset = Box::new(SubsetDataset::new(dataset, (0..max_frames).collect()));
        }
        dataset
    };

    let range_processing = RangeImageBuilder::default()
        .with_bilateral_filter(Some(BilateralFilter::default()))
        .with_normals(true)
        .with_intensity(true);

    let icp_params = MsIcpParams::default();

    let mut traj_builder = TrajectoryBuilder::default();
    let mut last_frame: Vec<RangeImage> = range_processing.build(dataset.get(0).unwrap());

    for i in tqdm!(
        1..dataset.len(),
        total = dataset.len() - 1,
        desc = "Processing frames"
    ) {
        let current_frame = range_processing.build(dataset.get(i).unwrap());
        let icp = MultiscaleAlign::new(icp_params.clone(), &last_frame).unwrap();
        let transform = icp.align(&current_frame);
        traj_builder.accumulate(&transform, Some(i as f32));
        last_frame = current_frame;
    }

    let pred_trajectory = traj_builder.build();
    let gt_trajectory = &dataset.trajectory().unwrap().first_frame_at_origin();

    let metrics = TransformMetrics::mean_trajectory_error(&pred_trajectory, gt_trajectory).unwrap();
    println!("Mean trajectory error: {metrics}");

    if args.show {
        RgbdDatasetViewer::new(dataset)
            .with_trajectory(pred_trajectory)
            .run();
    }
}
