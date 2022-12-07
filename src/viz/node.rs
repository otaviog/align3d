
use std::{sync::Arc, collections::HashMap};

use crate::bounds::Box3Df;
use nalgebra_glm;
use vulkano::{command_buffer::{PrimaryAutoCommandBuffer, AutoCommandBufferBuilder}, pipeline::GraphicsPipeline};

pub type Mat4x4 = nalgebra_glm::Mat4x4;

pub trait Node {
    fn transformation(&self) -> &Mat4x4;
    fn bounding_box(&self) -> &Box3Df;
    fn collect_command_buffers(&self, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pipelines: &mut HashMap<String, Arc<GraphicsPipeline>>);
}

#[derive(Clone)]
pub struct NodeProperties {
    pub transformation: Mat4x4,
    pub bounding_box: Box3Df
}

impl NodeProperties {
    
    pub fn default() -> Self {
        Self {
            transformation: Mat4x4::identity(),
            bounding_box: Box3Df::empty()
        }
    }
}

impl NodeProperties {
    pub fn transformation(&self) -> &Mat4x4 {
        &self.transformation
    }

    pub fn bounding_box(&self) -> &Box3Df {
        &self.bounding_box
    }
}


#[cfg(test)]
mod tests {
    use super::NodeProperties;

    #[test]
    fn test_basic_behavior() {
        let node = NodeProperties::default();
        assert!(node.bounding_box().is_empty());
    }
}
