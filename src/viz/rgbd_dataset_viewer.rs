use std::{cell::RefCell, rc::Rc};

use crate::{
    io::dataset::RgbdDataset, pointcloud::PointCloud, range_image::RangeImage,
};

use super::{
    node::{IntoVulkanWorldSpace, MakeNode},
    scene::Scene,
    Manager, Window,
};

pub struct RgbdDatasetViewer {
    pub dataset: Box<dyn RgbdDataset>,
}

impl RgbdDatasetViewer {
    pub fn new(dataset: Box<dyn RgbdDataset>) -> Self {
        Self { dataset }
    }

    pub fn run(&self) {
        let mut manager = Manager::default();
        let mut scene = Scene::default();
        let trajectory = self.dataset.trajectory().unwrap().first_frame_at_origin();

        for i in 0..self.dataset.len() {
            let rgbd_frame = self.dataset.get(i as usize).unwrap();
            let mut ri = RangeImage::from_rgbd_frame(&rgbd_frame);
            ri.compute_normals();
            let pcl: PointCloud = PointCloud::from(&ri);
            let node = pcl.make_node(&mut manager);

            let transform = trajectory[i as usize].clone();
            node.borrow_mut().properties_mut().transformation =
                transform.into_vulkan_coordinate_system();

            scene.add(node.clone());
        }

        let mut window = Window::create(&mut manager, Rc::new(RefCell::new(scene)));
        window.show();
    }
}
