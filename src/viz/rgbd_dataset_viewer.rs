use crate::{
    io::dataset::RgbdDataset, range_image::RangeImage, trajectory::Trajectory,
};

use super::{
    node::{IntoVulkanWorldSpace, MakeNode, NodeRef, node_ref, Node},
    scene::Scene,
    Manager, Window,
};

pub struct RgbdDatasetViewer {
    pub dataset: Box<dyn RgbdDataset>,
    pub trajectory: Trajectory,
}

impl RgbdDatasetViewer {
    pub fn new(dataset: Box<dyn RgbdDataset>) -> Self {
        let trajectory = dataset.trajectory().unwrap();
        Self { 
            dataset,
            trajectory
        }
    }

    pub fn with_trajectory(mut self, trajectory: Trajectory) -> Self {
        self.trajectory = trajectory;
        self
    }

    pub fn run(&self) {
        let mut manager = Manager::default();

        let scene = self.make_node(&mut manager);
        let mut window = Window::create(&mut manager, scene);
        window.show();
    }
}

impl MakeNode for RgbdDatasetViewer {
    type Node = Scene;
    fn make_node(&self, manager: &mut Manager) -> NodeRef<dyn Node> {
        let mut scene = Scene::default();

        for i in 0..self.dataset.len() {
            let rgbd_frame = self.dataset.get(i as usize).unwrap();
            let mut ri = RangeImage::from_rgbd_frame(&rgbd_frame);
            ri.compute_normals();
            let node = ri.make_node(manager);

            let transform = self.trajectory[i as usize].clone();
            node.borrow_mut().properties_mut().transformation =
                transform.into_vulkan_coordinate_system();

            scene.add(node.clone());
        }

        node_ref(scene)
    }
}