# Align3D

A crate for aligning 3D data using primary Iterative Closest Point (ICP) algorithms. 
All in Rust. 

Future versions should include more 3D visualization and SLAM related functionality.

## Sample use

The following code do the following:

* loads the IndoorLidarDataset;
* computes the odometry for 20 frames;
* display the metrics comparing with the ground truth;
* and shows the alignment results.

```rust
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
    // Loads the dataset
    let dataset = Box::new(SubsetDataset::new(
        Box::new(IndoorLidarDataset::load("tests/data/indoor_lidar/bedroom")?),
        (0..20).collect(),
    ));
    // RangeImageBuilder is a class to compose processing when loading
    // RGB-D frames (or `RangeImage`).
    let range_image_transform = RangeImageBuilder::default()
        .with_intensity(true) // Transforms RGB into intensity
        .with_normals(true) // Computes the normals
        .with_bilateral_filter(Some(BilateralFilter::default())) // Apply bilateral filter
        .pyramid_levels(3); // Computes 3-level gaussina pyramid.
    // Default Icp parameters.
    let icp_params = MsIcpParams::default(); 
    // TrajectoryBuilder accumulates the per-frame alignment to form 
    // the odometry of the camera poses.
    let mut traj_builder = TrajectoryBuilder::default(); 

    // Use the `.build()` method to create a RangeImage pyramid. 
    let mut prev_frame = range_image_transform.build(dataset.get(0).unwrap());

    // Iterate the dataset
    for i in 1..dataset.len() {
        let current_frame = range_image_transform.build(dataset.get(i).unwrap());
        // ICP alignment
        let icp = MultiscaleAlign::new(icp_params.clone(), &prev_frame).unwrap();
        let transform = icp.align(&current_frame);

        // Accumulates for getting the odometry.
        traj_builder.accumulate(&transform, Some(i as f32));
        prev_frame = current_frame;
    }

    // **Compute the metrics in relation to the ground truth**
    // Gets our trajectory.
    let pred_trajectory = traj_builder.build();
    // Gets the ground truth trajectory.
    let gt_trajectory = &dataset
        .trajectory()
        .expect("Dataset has no trajectory")
        .first_frame_at_origin();
    // Compute the metrics.
    let metrics = TransformMetrics::mean_trajectory_error(&pred_trajectory, &gt_trajectory)?;
    println!("Mean trajectory error: {}", metrics);

    // **Visualization part**
    RgbdDatasetViewer::new(dataset)
        .with_trajectory(pred_trajectory.clone())
        .run();

    Ok(())
}
```

It should display the following terminal output:

```txt
Mean trajectory error: angle: 1.91°, translation: 0.03885
```

and show a window like this:

![](resources/imgs/2023-04-07-16-26-03.png)

(move the camera using WASD controls)

# Release Plan

This crate is build for me to learn Rust (and it's being a worthwhile experience).
In the future, if reaches a performance and quality level it will be published as create.

Nevertheless, here's the current release plan

## v0.2 - Improve ICP

* [x] Bug fixes in PCL ICP.
* [ ] Optimize Image Icp performance

## v0.3 - Improve Visualization

* [ ] Optimize 3D rendering pipeline
* [ ] Improve UX of the 3D viewer

## v0.4 - Integration of RangeImages

* [ ] Either implement surfel or TSDF fusion
* [ ] Visualization
* [ ] Sparse registration based on point features for aligning submaps

## v0.5 - Pose Graph optimization

* [ ] Pose Graph build
* [ ] PGO Optimization