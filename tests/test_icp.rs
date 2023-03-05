use std::{cell::RefCell, rc::Rc};

use align3d::{
    icp::{ICPParams, ICP},
    imagepointcloud::ImagePointCloud,
    io::{dataset::RGBDDataset, slamtb::SlamTbDataset},
    pointcloud::PointCloud,
    viz::{geometry::VkPointCloudNode, scene::Scene, Manager, Window},
};
use nalgebra::Matrix4;
use rerun::Session;

pub fn main_1() {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();
    let item = dataset.get_item(0).unwrap();

    let pcl0: PointCloud = {
        let mut pcl = ImagePointCloud::from_rgbd_image(&item.0, &item.1);
        pcl.compute_normals();
        PointCloud::from(&pcl)
    };

    let item = dataset.get_item(5).unwrap();
    let pcl1: PointCloud = {
        let mut pcl = ImagePointCloud::from_rgbd_image(&item.0, &item.1);
        pcl.compute_normals();
        PointCloud::from(&pcl)
    };

    let icp = ICP::new(
        ICPParams {
            max_iterations: 10,
            weight: 0.01,
        },
        &pcl0,
    );
    let result = icp.align(&pcl1);

    let mut manager = Manager::default();
    let node0 = VkPointCloudNode::load(&manager, &pcl0);
    let node00 = node0.borrow().new_node();
    node00.borrow_mut().properties.transformation = Matrix4::from(result);

    let mut scene = Scene::default();
    scene
        .add(VkPointCloudNode::load(&manager, &pcl1))
        .add(node0)
        .add(node00);

    let mut window = Window::create(&mut manager, Rc::new(RefCell::new(scene)));
    window.show();
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();
    let item = dataset.get_item(0).unwrap();

    let pcl0: PointCloud = {
        let mut pcl = ImagePointCloud::from_rgbd_image(&item.0, &item.1);
        pcl.compute_normals();
        PointCloud::from(&pcl)
    };

    let item = dataset.get_item(5).unwrap();
    let pcl1: PointCloud = {
        let mut pcl = ImagePointCloud::from_rgbd_image(&item.0, &item.1);
        pcl.compute_normals();
        PointCloud::from(&pcl)
    };

    let icp = ICP::new(
        ICPParams {
            max_iterations: 10,
            weight: 0.01,
        },
        &pcl0,
    );
    let result = icp.align(&pcl1);

    let mut session = Session::new();
    pcl0.rerun_msg("pcl0")?
        .send(&mut session)?;
    pcl1.rerun_msg("pcl1")?
        .send(&mut session)?;
    (&result * &pcl1).rerun_msg("pcl1_transformed")?
        .send(&mut session)?;
    
    session.show()?;

    Ok(())
}
