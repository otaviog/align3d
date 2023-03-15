use std::{cell::RefCell, rc::Rc};

use align3d::{
    bilateral::BilateralFilter,
    icp::{multiscale::MultiscaleAlign, IcpParams},
    io::{
        core::RgbdDataset,
        rgbd_image::{RgbdFrame},
        slamtb::SlamTbDataset,
    },
    pointcloud::PointCloud,
    range_image::RangeImage,
    viz::{geometry::VkPointCloudNode, node::Node, scene::Scene, Manager, Window},
    Array2Recycle,
};

use nalgebra::Matrix4;
use winit::event::VirtualKeyCode;

fn process_frame(mut frame: RgbdFrame) -> Vec<RangeImage> {
    frame.image.depth = {
        let filter = BilateralFilter::default();
        filter.filter(&frame.image.depth, Array2Recycle::Empty)
    };

    let mut pyramid = RangeImage::from_pyramid(&frame.pyramid(3));

    for pcl in pyramid.iter_mut() {
        pcl.compute_normals().compute_intensity();
    }

    pyramid
}

fn main() {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample2").unwrap();
    let trajectory = dataset.trajectory().unwrap();
    const SOURCE_IDX: usize = 0;
    const TARGET_IDX: usize = 6;

    let source_pcl = process_frame(dataset.get_item(SOURCE_IDX).unwrap());

    let target_pcl = process_frame(dataset.get_item(TARGET_IDX).unwrap());

    let icp = MultiscaleAlign::new(
        &target_pcl,
        IcpParams {
            max_iterations: 20,
            weight: 0.5,
        },
    );

    let result = icp.align(&source_pcl);

    let mut manager = Manager::default();
    let source_node = VkPointCloudNode::load(&manager, &PointCloud::from(&source_pcl[0]));
    let source_t_node = source_node.borrow().new_node();
    let gt_mat: Matrix4<f32> = Matrix4::from(
        &trajectory
            .get_relative_transform(TARGET_IDX as f32, SOURCE_IDX as f32)
            .unwrap(),
    );

    println!("GT: {}", gt_mat);
    source_t_node.borrow_mut().properties.transformation = Matrix4::from(&result);

    let target_node = VkPointCloudNode::load(&manager, &PointCloud::from(&target_pcl[0]));

    let mut scene = Scene::default();
    scene
        .add(target_node.clone())
        .add(source_node.clone())
        .add(source_t_node.clone());

    let mut window = Window::create(&mut manager, Rc::new(RefCell::new(scene)));
    window.on_key = Some(Box::new(move |vkeycode, _window| {
        if let Some(node) = match vkeycode {
            VirtualKeyCode::Key1 => Some(source_node.clone()),
            VirtualKeyCode::Key2 => Some(target_node.clone()),
            VirtualKeyCode::Key3 => Some(source_t_node.clone()),
            _ => None,
        } {
            let mut node = node.borrow_mut();
            let is_visible = node.properties().visible;
            node.properties_mut().set_visible(!is_visible);
        }
    }));
    window.show();
}
