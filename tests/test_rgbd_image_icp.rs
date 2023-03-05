use std::{cell::RefCell, rc::Rc};

use align3d::{
    bilateral::BilateralFilter,
    camera::Camera,
    icp::{ICPParams, ImageICP},
    imagepointcloud::ImagePointCloud,
    intensity_map::IntensityMap,
    io::{dataset::RGBDDataset, rgbdimage::RGBDImage, slamtb::SlamTbDataset},
    pointcloud::PointCloud,
    viz::{geometry::VkPointCloudNode, node::Node, scene::Scene, Manager, Window},
    Array2Recycle,
};

use winit::event::VirtualKeyCode;

fn process_frame(item: &mut (Camera, RGBDImage)) -> ImagePointCloud {
    item.1.depth = {
        let filter = BilateralFilter::default();
        filter.filter(&item.1.depth, Array2Recycle::Empty)
    };
    let mut pcl = ImagePointCloud::from_rgbd_image(&item.0, &item.1);

    pcl.compute_normals().compute_intensity();

    pcl
}

fn main() {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample2").unwrap();
    let trajectory = dataset.trajectory().unwrap();
    const SOURCE_IDX: usize = 0;
    const TARGET_IDX: usize = 14;

    let (cam, source_pcl, intensity_map) = {
        let mut item = dataset.get_item(SOURCE_IDX).unwrap();
        let pcl = process_frame(&mut item);
        let intensity_map = IntensityMap::from_rgb_image(&item.1.color);
        (item.0, pcl, intensity_map)
    };

    let target_pcl = process_frame(&mut dataset.get_item(TARGET_IDX).unwrap());

    let icp = ImageICP::new(
        ICPParams {
            max_iterations: 2,
            weight: 0.5,
        },
        cam,
        &source_pcl,
        &intensity_map,
    );
    let result = icp.align(&target_pcl);

    let mut manager = Manager::default();
    let source_node = VkPointCloudNode::load(&manager, &PointCloud::from(&source_pcl));
    let source_t_node = source_node.borrow().new_node();
    source_t_node.borrow_mut().properties.transformation = result.into(); // Matrix4::from(result);
                                                                                    //source_t_node.borrow_mut().properties.transformation =
                                                                                    //    trajectory.get_relative_transform(TARGET_IDX as f32, SOURCE_IDX as f32).unwrap().into();

    let target_node = VkPointCloudNode::load(&manager, &PointCloud::from(&target_pcl));

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
