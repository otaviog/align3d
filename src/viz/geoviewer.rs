use std::{cell::RefCell, rc::Rc};

use super::{
    node::{MakeNode, Node, NodeRef},
    scene::Scene,
    Manager, Window,
};

pub struct GeoViewer {
    scene: NodeRef<Scene>,
    manager: Manager,
}

impl Default for GeoViewer {
    fn default() -> Self {
        Self::new()
    }
}

impl GeoViewer {
    pub fn new() -> Self {
        Self {
            scene: Rc::new(RefCell::new(Scene::default())),
            manager: Manager::default(),
        }
    }

    pub fn from_manager(manager: Manager) -> Self {
        Self {
            scene: Rc::new(RefCell::new(Scene::default())),
            manager,
        }
    }

    pub fn add_node(&mut self, node: NodeRef<dyn Node>) {
        self.scene.borrow_mut().add(node);
    }

    pub fn add<GeomType>(&mut self, node: &GeomType) -> NodeRef<dyn Node>
    where
        GeomType: MakeNode,
    {
        let node = node.make_node(&mut self.manager);
        self.scene.borrow_mut().add(node.clone());

        node
    }

    pub fn run(mut self) {
        let mut window = Window::create(&mut self.manager, self.scene.clone());
        window.on_key = Some(Box::new(move |vkeycode, _window| {
            let num_key = vkeycode as u32;
            if num_key <= 10 {
                let scene = self.scene.borrow_mut();
                if let Some(node) = scene.nodes.get(num_key as usize).cloned() {
                    let mut node = node.borrow_mut();
                    let is_visible = node.properties().visible;
                    node.properties_mut().set_visible(!is_visible);
                }
            }
        }));
        window.show();
    }
}
