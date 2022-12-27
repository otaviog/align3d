use align3d::{
    imagepointcloud::ImagePointCloud,
    io::{dataset::RGBDDataset, slamtb::SlamTbDataset},
    pointcloud::PointCloud, icp::ICP,
};

pub fn main() {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();
    let item = dataset.get_item(0).unwrap();

    let pcl0: PointCloud = {
        let mut pcl = ImagePointCloud::from_rgbd_image(item.0, item.1);
        pcl.compute_normals();
        pcl.into()
    };

    let item = dataset.get_item(5).unwrap();
    let pcl1: PointCloud = {
        let mut pcl = ImagePointCloud::from_rgbd_image(item.0, item.1);
        pcl.compute_normals();
        pcl.into()
    };

    let icp = ICP::new(ICPParams { max_iterations: 10, weight: 0.001 }, &pcl0);
    let result = icp.align(&pcl1);

}
