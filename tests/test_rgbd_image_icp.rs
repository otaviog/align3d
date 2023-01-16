use std::{cell::RefCell, rc::Rc};

use align3d::{
    icp::{ICPParams, ImageICP},
    imagepointcloud::ImagePointCloud,
    intensity_map::IntensityMap,
    io::{dataset::RGBDDataset, slamtb::SlamTbDataset},
    pointcloud::PointCloud,
    viz::{geometry::VkPointCloudNode, node::Node, scene::Scene, Manager, Window},
};
use nalgebra::Matrix4;
use winit::event::VirtualKeyCode;

fn main() {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();

    let (cam, pcl0, intensity_map) = {
        let item = dataset.get_item(0).unwrap();
        let mut pcl = ImagePointCloud::from_rgbd_image(&item.0, &item.1);
        pcl.compute_normals().compute_intensity();

        let intensity_map = IntensityMap::from_rgb_image(&item.1.color);

        (item.0, pcl, intensity_map)
    };

    let pcl1 = {
        let item = dataset.get_item(14).unwrap();
        let mut pcl = ImagePointCloud::from_rgbd_image(&item.0, &item.1);
        pcl.compute_normals();
        pcl.compute_intensity();
        pcl
    };

    let icp = ImageICP::new(
        ICPParams {
            max_iterations: 10,
            weight: 0.01,
        },
        cam,
        &pcl0,
        &intensity_map,
    );
    let result = icp.align(&pcl1);

    let mut manager = Manager::default();
    let node0 = VkPointCloudNode::load(&manager, &PointCloud::from(&pcl0));
    let node00 = node0.borrow().new_node();
    node00.borrow_mut().properties.transformation = Matrix4::from(result);

    let node1 = VkPointCloudNode::load(&manager, &PointCloud::from(&pcl1));

    let mut scene = Scene::default();
    scene
        .add(node1.clone())
        .add(node0.clone())
        .add(node00.clone());

    let mut window = Window::create(&mut manager, Rc::new(RefCell::new(scene)));
    window.on_key = Some(Box::new(move |vkeycode, _window| {
        if let Some(node) = match vkeycode {
            VirtualKeyCode::Key1 => Some(node1.clone()),
            VirtualKeyCode::Key2 => Some(node00.clone()),
            VirtualKeyCode::Key3 => Some(node0.clone()),
            _ => None,
        } {
            let mut node = node.borrow_mut();
            let is_visible = node.properties().visible;
            node.properties_mut().set_visible(!is_visible);
        }
    }));
    window.show();
}
