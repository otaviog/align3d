use super::{
    node::{node_ref, MakeNode, Node, NodeRef},
    scene::Scene,
    Manager, Window,
};

pub struct GeoViewer {
    scene: NodeRef<Scene>,
    manager: Manager,
    window: Option<Window>,
}

impl Default for GeoViewer {
    fn default() -> Self {
        Self::new()
    }
}

impl GeoViewer {
    pub fn new() -> Self {
        Self {
            scene: node_ref(Scene::default()),
            manager: Manager::default(),
            window: None,
        }
    }

    pub fn from_manager(manager: Manager) -> Self {
        Self {
            scene: node_ref(Scene::default()),
            manager,
            window: None,
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

    pub fn run(&mut self) {
        self.window
            .replace(Window::create(&mut self.manager, self.scene.clone()));
        let window = self.window.as_mut().unwrap();
        let scene = self.scene.clone();

        window.on_key = Some(Box::new(move |vkeycode, _window| {
            let num_key = vkeycode as u32;
            if num_key <= 10 {
                if let Some(node) = scene.borrow_mut().nodes.get(num_key as usize).cloned() {
                    let mut node = node.borrow_mut();
                    let is_visible = node.properties().visible;
                    node.properties_mut().set_visible(!is_visible);
                }
            }
        }));
        window.show();
    }
}
