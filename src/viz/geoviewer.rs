use std::{cell::RefCell, rc::Rc};

use crate::{pointcloud::PointCloud, range_image::RangeImage};

use super::{geometry::VkPointCloudNode, scene::Scene, Manager, Window};

pub struct GeoViewer {
    scene: Rc<RefCell<Scene>>,
    manager: Manager,
}

impl GeoViewer {
    pub fn new() -> Self {
        Self {
            scene: Rc::new(RefCell::new(Scene::default())),
            manager: Manager::default(),
        }
    }

    pub fn add_point_cloud(&mut self, point_cloud: &PointCloud) -> Rc<RefCell<VkPointCloudNode>> {
        let node = VkPointCloudNode::load(&self.manager, &point_cloud);
        self.scene.borrow_mut().add(node.clone());
        node
    }

    pub fn add_range_image(&mut self, range_image: &RangeImage) -> Rc<RefCell<VkPointCloudNode>> {
        self.add_point_cloud(&PointCloud::from(range_image))
    }

    pub fn run(mut self) {
        let mut window = Window::create(&mut self.manager, self.scene.clone());
        window.on_key = Some(Box::new(move |vkeycode, _window| {
            let num_key = vkeycode as u32;
            if num_key <= 10 {
                let scene = self.scene.borrow_mut();
                if let Some(node) = scene.nodes.get(num_key as usize).map(|n| n.clone()) {
                    let mut node = node.borrow_mut();
                    let is_visible = node.properties().visible;
                    node.properties_mut().set_visible(!is_visible);
                }
            }
        }));
        window.show();
    }
}
