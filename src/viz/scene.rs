use std::sync::Arc;

use super::{
    controllers::FrameStepInfo,
    node::{CommandBuffersContext, Mat4x4, Node, NodeProperties},
};
use crate::bounds::Sphere3Df;

#[derive(Clone, Default)]
pub struct Scene {
    node_properties: NodeProperties,
    pub nodes: Vec<Arc<dyn Node>>,
}

impl Node for Scene {
    fn transformation(&self) -> &Mat4x4 {
        self.node_properties.transformation()
    }

    fn bounding_sphere(&self) -> &Sphere3Df {
        self.node_properties.bounding_sphere()
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
            nalgebra_glm::inverse_transpose(context.view_matrix.fixed_slice::<3, 3>(3, 3).into());

        // Traverse subnodes:
        for node in self.nodes.iter() {
            node.collect_command_buffers(context, window_state);
        }

        // Resets the matrices to the originals:
        context.view_normals_matrix = saved_view_normals_matrix;
        context.view_matrix = saved_view_matrix;
    }
}
