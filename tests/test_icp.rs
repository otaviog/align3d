use align3d::{
    bilateral::BilateralFilter,
    icp::{Icp, IcpParams},
    io::dataset::{RgbdDataset, SlamTbDataset},
    pointcloud::PointCloud,
    range_image::RangeImageBuilder,
    viz::GeoViewer,
};

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    const SOURCE_IDX: usize = 0;
    const TARGET_IDX: usize = 4;

    let dataset = SlamTbDataset::load("tests/data/rgbd/sample2").unwrap();
    let frame_transform = RangeImageBuilder::default()
        .with_bilateral_filter(Some(BilateralFilter::default()))
        .with_normals(true)
        .pyramid_levels(1);
    let target_pcl = PointCloud::from(&frame_transform.build(dataset.get(TARGET_IDX).unwrap())[0]);
    let source_pcl = PointCloud::from(&frame_transform.build(dataset.get(SOURCE_IDX).unwrap())[0]);

    let icp = Icp::new(
        IcpParams {
            max_iterations: 15,
            max_distance: 0.5,
            max_normal_angle: 25_f32.to_radians(),
            ..Default::default()
        },
        &target_pcl,
    );
    let result = icp.align(&source_pcl);

    let mut viewer = GeoViewer::new();
    viewer.add(&target_pcl);
    viewer.add(&source_pcl);
    viewer.add(&(&result * &source_pcl));
    viewer.run();

    Ok(())
}
