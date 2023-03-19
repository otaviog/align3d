use align3d::{
    bilateral::BilateralFilter,
    icp::{multiscale::MultiscaleAlign, IcpParams, MsIcpParams},
    io::{core::RgbdDataset, slamtb::SlamTbDataset},
    metrics::TransformMetrics,
    range_image::{RangeImageBuilder},
    viz::GeoViewer,
};

use nalgebra::Matrix4;

fn main() {
    const SOURCE_IDX: usize = 0;
    const TARGET_IDX: usize = 6;

    let dataset = SlamTbDataset::load("tests/data/rgbd/sample2").unwrap();
    let rgbd_transform = RangeImageBuilder::default()
        .with_luma(true)
        .with_normals(true)
        .with_bilateral_filter(Some(BilateralFilter::default()))
        .pyramid_levels(3);
    let source_pcl = rgbd_transform.build(dataset.get_item(SOURCE_IDX).unwrap());
    let mut target_pcl = rgbd_transform.build(dataset.get_item(TARGET_IDX).unwrap());

    let params = MsIcpParams::repeat(
        3,
        &IcpParams {
            weight: 1.0,
            color_weight: 0.1,
            ..Default::default()
        },
    )
    .customize(|level, mut params| {
        match level {
            0 => params.max_iterations = 15, // 0 is the last level run
            1 => params.max_iterations = 20,
            2 => params.max_iterations = 30,
            3 => params.max_iterations = 30,
            _ => {}
        };
    });

    let mut icp = MultiscaleAlign::new(&mut target_pcl, params).unwrap();
    let result = icp.align(&source_pcl);

    let gt_transform = dataset
        .trajectory()
        .unwrap()
        .get_relative_transform(SOURCE_IDX as f32, TARGET_IDX as f32)
        .unwrap();
    println!(
        "Metrics: {:}",
        TransformMetrics::new(&gt_transform, &result)
    );

    let mut viewer = GeoViewer::new();
    viewer.add_range_image(&source_pcl[0]);
    viewer.add_range_image(&target_pcl[0]);
    let source_t_node = viewer.add_range_image(&source_pcl[0]);
    source_t_node.borrow_mut().properties.transformation = Matrix4::from(&result);
    viewer.run();
}

