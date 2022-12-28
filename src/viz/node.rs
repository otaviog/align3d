use std::{collections::HashMap, sync::Arc};

use crate::bounds::Sphere3Df;
use nalgebra_glm::{self, Mat3};
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    device::Device,
    pipeline::GraphicsPipeline,
    render_pass::RenderPass,
};

use super::controllers::FrameStepInfo;

pub type Mat4x4 = nalgebra_glm::Mat4x4;

pub struct CommandBuffersContext<'a> {
    pub device: Arc<Device>,
    pub builder: &'a mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    pub pipelines: &'a mut HashMap<String, Arc<GraphicsPipeline>>,
    pub render_pass: Arc<RenderPass>,
    pub view_normals_matrix: nalgebra_glm::Mat3,
    pub view_matrix: nalgebra_glm::Mat4,
    pub projection_matrix: nalgebra_glm::Mat4,
}

impl<'a> CommandBuffersContext<'a> {
    pub fn new(
        device: Arc<Device>,
        builder: &'a mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pipelines: &'a mut HashMap<String, Arc<GraphicsPipeline>>,
        render_pass: Arc<RenderPass>,

        view_matrix: nalgebra_glm::Mat4,
        projection_matrix: nalgebra_glm::Mat4,
    ) -> Self {
        let view_normals_matrix: Mat3 =
            nalgebra_glm::inverse_transpose(view_matrix.fixed_slice::<3, 3>(0, 0).into());

        Self {
            device: device,
            builder: builder,
            pipelines: pipelines,
            render_pass: render_pass,
            view_matrix: view_matrix,
            projection_matrix: projection_matrix,
            view_normals_matrix: view_normals_matrix,
        }
    }
}

pub trait Node {
    fn transformation(&self) -> &Mat4x4;
    fn bounding_sphere(&self) -> &Sphere3Df;
    fn collect_command_buffers(
        &self,
        context: &mut CommandBuffersContext,
        window_state: &FrameStepInfo,
    );
}

#[derive(Clone)]
pub struct NodeProperties {
    pub transformation: Mat4x4,
    pub bounding_sphere: Sphere3Df,
}

impl Default for NodeProperties {
    fn default() -> Self {
        Self {
            transformation: Mat4x4::identity(),
            bounding_sphere: Sphere3Df::empty(),
        }
    }
}

impl NodeProperties {
    pub fn transformation(&self) -> &Mat4x4 {
        &self.transformation
    }

    pub fn bounding_sphere(&self) -> &Sphere3Df {
        &self.bounding_sphere
    }
}

#[cfg(test)]
mod tests {
    use super::NodeProperties;

    #[test]
    fn test_basic_behavior() {
        let node = NodeProperties::default();
        assert!(node.bounding_sphere().is_empty());
    }
}
