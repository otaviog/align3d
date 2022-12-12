use std::{collections::HashMap, sync::Arc};

use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    device::Device,
    pipeline::GraphicsPipeline,
};

use super::node::{CommandBuffersContext, Mat4x4, Node, NodeProperties};
use crate::bounds::Box3Df;

#[derive(Clone)]
pub struct Scene {
    node_properties: NodeProperties,
    pub nodes: Vec<Arc<dyn Node>>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            node_properties: NodeProperties::default(),
            nodes: Vec::new(),
        }
    }
}

impl Node for Scene {
    fn transformation(&self) -> &Mat4x4 {
        self.node_properties.transformation()
    }

    fn bounding_box(&self) -> &Box3Df {
        self.node_properties.bounding_box()
    }

    fn collect_command_buffers(&self, context: &mut CommandBuffersContext) {
        let parent_object_matrix = context.object_matrix.clone();
        context.object_matrix = context.object_matrix * self.node_properties.transformation;
        for node in self.nodes.iter() {
            node.collect_command_buffers(context);
        }
        context.object_matrix = parent_object_matrix;
    }
}
