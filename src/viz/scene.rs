use super::{
    controllers::FrameStepInfo,
    node::{node_ref, CommandBuffersContext, Node, NodeProperties, NodeRef},
};

#[derive(Clone, Default)]
pub struct Scene {
    node_properties: NodeProperties,
    pub nodes: Vec<NodeRef<dyn Node>>,
}

impl Scene {
    pub fn add(&mut self, node: NodeRef<dyn Node>) -> &mut Self {
        let new_sphere = self
            .node_properties
            .get_bounding_sphere()
            .add(&node.borrow().properties().get_bounding_sphere());

        self.node_properties.bounding_sphere(new_sphere);
        self.nodes.push(node);
        self
    }
}

impl Node for Scene {
    fn properties(&self) -> &NodeProperties {
        &self.node_properties
    }

    fn properties_mut(&mut self) -> &mut NodeProperties {
        &mut self.node_properties
    }

    fn new_instance(&self) -> NodeRef<dyn Node> {
        node_ref(Self {
            nodes: self
                .nodes
                .iter()
                .map(|node| node.borrow_mut().new_instance())
                .collect(),
            node_properties: self.node_properties,
        })
    }

    fn collect_command_buffers(
        &self,
        context: &mut CommandBuffersContext,
        window_state: &FrameStepInfo,
    ) {
        // Save the view matrices.
        let saved_view_matrix = context.view_matrix;
        let saved_view_normals_matrix = context.view_normals_matrix;

        // Transform with this parent node transformation.
        context.view_matrix *= self.node_properties.transformation;
        context.view_normals_matrix =
            nalgebra_glm::inverse_transpose(context.view_matrix.fixed_slice::<3, 3>(0, 0).into());

        // Traverse subnodes:
        for node in self.nodes.iter() {
            let node = node.borrow_mut();
            if node.properties().visible {
                node.collect_command_buffers(context, window_state);
            }
        }

        // Resets the matrices to the originals:
        context.view_normals_matrix = saved_view_normals_matrix;
        context.view_matrix = saved_view_matrix;
    }
}
