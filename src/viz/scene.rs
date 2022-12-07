use std::{collections::HashMap, sync::Arc};

use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    pipeline::GraphicsPipeline,
};

use super::node::{Mat4x4, Node, NodeProperties};
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

    fn collect_command_buffers(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pipelines: &mut HashMap<String, Arc<GraphicsPipeline>>,
    ) {
        for node in self.nodes.iter() {
            node.collect_command_buffers(builder, pipelines);
        }
    }
}
