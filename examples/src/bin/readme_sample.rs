use align3d::{
    bilateral::BilateralFilter,
    icp::{multiscale::MultiscaleAlign, MsIcpParams},
    io::dataset::{IndoorLidarDataset, RgbdDataset, SubsetDataset},
    metrics::TransformMetrics,
    range_image::RangeImageBuilder,
    trajectory::TrajectoryBuilder,
    viz::rgbd_dataset_viewer::RgbdDatasetViewer, transform::Transform,
};

fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    // Opens the dataset
    let dataset = Box::new(SubsetDataset::new(
        Box::new(IndoorLidarDataset::load("../datasets/apartment")?),
        (0..20).collect(),
    ));
    // Prepares range image processing
    let range_image_build = RangeImageBuilder::default()
        .with_bilateral_filter(Some(BilateralFilter::default()))
        .pyramid_levels(3);

    // Initialize ICP params for MultiScale alignment.
    let icp_params = MsIcpParams::default();
    let mut trajectory_build = TrajectoryBuilder::with_start(Transform::eye(), 0.0);

    // Frame-to-frame odometry
    let mut prev_frame = range_image_build.build(dataset.get(0).unwrap());
    for i in 1..dataset.len() {
        let current_frame = range_image_build.build(dataset.get(i).unwrap());

        let icp = MultiscaleAlign::new(icp_params.clone(), &prev_frame).unwrap();
        let transform = icp.align(&current_frame);
        trajectory_build.accumulate(&transform, Some(i as f32));

        prev_frame = current_frame;
    }

    // Computes the mean trajectory errors
    let pred_trajectory = trajectory_build.build();
    let gt_trajectory = &dataset
        .trajectory()
        .expect("Dataset has no trajectory")
        .first_frame_at_origin();
    let metrics = TransformMetrics::mean_trajectory_error(&pred_trajectory, gt_trajectory)?;
    println!("Mean trajectory error: {metrics}");
    
    // Shows the point clouds
    RgbdDatasetViewer::new(dataset)
        .with_trajectory(pred_trajectory)
        .run();
    Ok(())
}
