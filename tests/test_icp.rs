use align3d::{
    icp::{Icp, IcpParams},
    io::{core::RgbdDataset, slamtb::SlamTbDataset},
    pointcloud::PointCloud,
    range_image::RangeImage, viz::GeoViewer,
};


pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();
    let frame = dataset.get_item(0).unwrap();

    let target_pcl: PointCloud = {
        let mut pcl = RangeImage::from_rgbd_frame(&frame);
        pcl.compute_normals();
        PointCloud::from(&pcl)
    };

    let frame = dataset.get_item(5).unwrap();
    let source_pcl: PointCloud = {
        let mut pcl = RangeImage::from_rgbd_frame(&frame);
        pcl.compute_normals();
        PointCloud::from(&pcl)
    };

    let icp = Icp::new(
        IcpParams {
            max_iterations: 10,
            weight: 0.01,
            ..Default::default()
        },
        &target_pcl,
    );
    let result = icp.align(&source_pcl);

    let mut viewer = GeoViewer::new();
    viewer.add_point_cloud(&target_pcl);
    viewer.add_point_cloud(&source_pcl);
    viewer.add_point_cloud(&(&result * &source_pcl));
    viewer.run();

    Ok(())
}
