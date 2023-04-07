use align3d::{
    bilateral::BilateralFilter,
    icp::{multiscale::MultiscaleAlign, MsIcpParams},
    io::dataset::{IndoorLidarDataset, RgbdDataset, SubsetDataset},
    metrics::TransformMetrics,
    range_image::RangeImageBuilder,
    trajectory_builder::TrajectoryBuilder,
    viz::rgbd_dataset_viewer::RgbdDatasetViewer,
};

fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let dataset = Box::new(SubsetDataset::new(
        Box::new(IndoorLidarDataset::load("tests/data/indoor_lidar/bedroom")?),
        (0..20).collect(),
    ));
    let range_image_transform = RangeImageBuilder::default()
        .with_intensity(true)
        .with_normals(true)
        .with_bilateral_filter(Some(BilateralFilter::default()))
        .pyramid_levels(3);
    let icp_params = MsIcpParams::default();
    let mut traj_builder = TrajectoryBuilder::default();
    let mut prev_frame = range_image_transform.build(dataset.get(0).unwrap());
    for i in 1..dataset.len() {
        let current_frame = range_image_transform.build(dataset.get(i).unwrap());
        let icp = MultiscaleAlign::new(icp_params.clone(), &prev_frame).unwrap();
        let transform = icp.align(&current_frame);
        traj_builder.accumulate(&transform, Some(i as f32));
        prev_frame = current_frame;
    }
    let pred_trajectory = traj_builder.build();
    let gt_trajectory = &dataset
        .trajectory()
        .expect("Dataset has no trajectory")
        .first_frame_at_origin();
    let metrics = TransformMetrics::mean_trajectory_error(&pred_trajectory, &gt_trajectory)?;
    println!("Mean trajectory error: {}", metrics);
    RgbdDatasetViewer::new(dataset)
        .with_trajectory(pred_trajectory.clone())
        .run();

    Ok(())
}
