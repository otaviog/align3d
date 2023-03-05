use std::{rc::Rc, cell::RefCell};

use crate::{io::dataset::RGBDDataset, pointcloud::PointCloud, imagepointcloud::ImagePointCloud, bilateral::BilateralFilter, Array2Recycle};

use super::{Manager, scene::Scene, geometry::VkPointCloudNode, Window};

pub struct RgbdDatasetViewer {
    pub dataset: Box<dyn RGBDDataset>,
}

impl RgbdDatasetViewer {
    pub fn new(dataset: Box<dyn RGBDDataset>) -> Self {
        Self { dataset }
    }

    pub fn run(&self) {
        let mut manager = Manager::default();
        let mut scene = Scene::default();
        let trajectory = self.dataset.trajectory().unwrap();
        
        for i in 0..self.dataset.len() {            
            let transform = trajectory.get_relative_transform(0.0, i as f32).unwrap();
            let (camera, mut frame) = self.dataset.get_item(i).unwrap();

            frame.depth = {
                let filter = BilateralFilter::default();
                filter.filter(&frame.depth, Array2Recycle::Empty)
            };

            let mut point_cloud = ImagePointCloud::from_rgbd_image(&camera, &frame);

            point_cloud.compute_normals();
            // point_cloud.compute_intensity();
            
            let node = VkPointCloudNode::load(&manager, &PointCloud::from(&point_cloud));
            
            node.borrow_mut().properties.transformation = transform.into();
            scene.add(node.clone());
        }
        
        let mut window = Window::create(&mut manager, Rc::new(RefCell::new(scene)));
        window.show();
    }
}


