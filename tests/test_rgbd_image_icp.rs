use std::{cell::RefCell, rc::Rc};

use align3d::{
    icp::{ICPParams, ImageICP},
    imagepointcloud::ImagePointCloud,
    io::{dataset::RGBDDataset, slamtb::SlamTbDataset},
    pointcloud::PointCloud,
    viz::{geometry::VkPointCloudNode, scene::Scene, Manager, Window},
};
use nalgebra::Matrix4;

fn main() {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();

    let (cam, pcl0) = {
        let item = dataset.get_item(0).unwrap();
        let mut pcl = ImagePointCloud::from_rgbd_image(&item.0, item.1);
        pcl.compute_normals();
        (item.0, pcl)
    };

    let pcl1 = {
        let item = dataset.get_item(14).unwrap();
        let mut pcl = ImagePointCloud::from_rgbd_image(&item.0, item.1);
        pcl.compute_normals();
        pcl
    };

    let icp = ImageICP::new(
        ICPParams {
            max_iterations: 10,
            weight: 0.01,
        },
        cam,
        &pcl0,
    );
    let result = icp.align(&pcl1);

    let mut manager = Manager::default();
    let node0 = VkPointCloudNode::load(&manager, 
        &PointCloud::from(&pcl0));
    let node00 = node0.borrow().new_node();
    node00.borrow_mut().properties.transformation = Matrix4::from(result);

    let mut scene = Scene::default();
    scene
        .add(VkPointCloudNode::load(&manager, &PointCloud::from(&pcl1)))
        .add(node0)
        .add(node00);

    let mut window = Window::create(&mut manager, Rc::new(RefCell::new(scene)));
    window.show();
}
