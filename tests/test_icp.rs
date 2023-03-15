use align3d::{
    icp::{Icp, IcpParams},
    io::{core::RgbdDataset, slamtb::SlamTbDataset},
    pointcloud::PointCloud,
    range_image::RangeImage,
};

use rerun::Session;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();
    let frame = dataset.get_item(0).unwrap();

    let pcl0: PointCloud = {
        let mut pcl = RangeImage::from_rgbd_frame(&frame);
        pcl.compute_normals();
        PointCloud::from(&pcl)
    };

    let frame = dataset.get_item(5).unwrap();
    let pcl1: PointCloud = {
        let mut pcl = RangeImage::from_rgbd_frame(&frame);
        pcl.compute_normals();
        PointCloud::from(&pcl)
    };

    let icp = Icp::new(
        IcpParams {
            max_iterations: 10,
            weight: 0.01,
        },
        &pcl0,
    );
    let result = icp.align(&pcl1);

    let mut session = Session::new();
    pcl0.rerun_msg("pcl0")?.send(&mut session)?;
    pcl1.rerun_msg("pcl1")?.send(&mut session)?;
    (&result * &pcl1)
        .rerun_msg("pcl1_transformed")?
        .send(&mut session)?;

    session.show()?;

    Ok(())
}
