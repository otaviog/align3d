use std::f32::consts::PI;

use align3d::{
    bilateral::BilateralFilter,
    icp::{multiscale::MultiscaleAlign, IcpParams, MsIcpParams},
    io::{core::RgbdDataset, slamtb_dataset::SlamTbDataset},
    metrics::TransformMetrics,
    range_image::RangeImageBuilder,
    transform::Transform,
    viz::GeoViewer,
};

use nalgebra::Matrix4;

fn main() {
    const SOURCE_IDX: usize = 0;
    const TARGET_IDX: usize = 7;

    let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();
    let rgbd_transform = RangeImageBuilder::default()
        .with_intensity(true)
        .with_normals(true)
        .with_bilateral_filter(Some(BilateralFilter::default()))
        .pyramid_levels(3);
    let source_pcl = rgbd_transform.build(dataset.get_item(SOURCE_IDX).unwrap());
    let mut target_pcl = rgbd_transform.build(dataset.get_item(TARGET_IDX).unwrap());

    let params = MsIcpParams::repeat(
        3,
        &IcpParams {
            weight: 0.0,
            color_weight: 1.0,
            max_normal_angle: PI / 10.0,
            max_color_distance: 2.75,
            max_distance: 0.5,
            ..Default::default()
        },
    )
    .customize(|level, mut params| {
        match level {
            0 => params.max_iterations = 20, // 0 is the last level run
            1 => params.max_iterations = 10,
            2 => params.max_iterations = 5,
            3 => params.max_iterations = 5,
            _ => {}
        };
    });

    let icp = MultiscaleAlign::new(&target_pcl, params).unwrap();
    let result = icp.align(&source_pcl);

    let gt_transform = dataset
        .trajectory()
        .unwrap()
        .get_relative_transform(SOURCE_IDX as f32, TARGET_IDX as f32)
        .unwrap();
    println!(
        "Start with metrics: {:}",
        TransformMetrics::new(&gt_transform, &Transform::eye())
    );
    println!(
        "After metrics: {:}",
        TransformMetrics::new(&gt_transform, &result)
    );

    let mut viewer = GeoViewer::new();
    viewer.add_node(&source_pcl[0]);
    viewer.add_node(&target_pcl[0]);
    let source_t_node = viewer.add_node(&source_pcl[0]);
    source_t_node.borrow_mut().properties_mut().transformation = Matrix4::from(&result);
    viewer.run();
}
