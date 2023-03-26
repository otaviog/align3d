use std::{cell::RefCell, rc::Rc};

use nalgebra::Matrix4;

use crate::{
    bilateral::BilateralFilter, io::core::RgbdDataset, range_image::RangeImage, Array2Recycle,
};

use super::{node::MakeNode, scene::Scene, Manager, Window};

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
        let trajectory = self.dataset.trajectory().unwrap();

        for i in 0..self.dataset.len() {
            let transform = trajectory.get_relative_transform(0.0, i as f32).unwrap();
            let (camera, mut frame) = self.dataset.get_item(i).unwrap().into_parts();

            frame.depth = {
                let filter = BilateralFilter::default();
                filter.filter(&frame.depth, Array2Recycle::Empty)
            };

            let mut point_cloud = RangeImage::from_rgbd_image(&camera, &frame);

            point_cloud.compute_normals();
            point_cloud.compute_intensity();

            let node = point_cloud.make_node(&mut manager);

            node.borrow_mut().properties_mut().transformation = Matrix4::from(&transform);
            scene.add(node.clone());
        }

        let mut window = Window::create(&mut manager, Rc::new(RefCell::new(scene)));
        window.show();
    }
}
