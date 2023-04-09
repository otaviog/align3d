use std::{cell::RefCell, collections::HashMap, rc::Rc, sync::Arc};

use crate::{bounds::Sphere3Df, transform::{Transformable, Transform}};
use nalgebra::Matrix4;
use nalgebra_glm::{self, Mat3};
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    device::Device,
    pipeline::GraphicsPipeline,
    render_pass::RenderPass,
};

use super::{controllers::FrameStepInfo, Manager};

pub type NodeRef<T> = Rc<RefCell<T>>;
pub fn node_ref<T>(node: T) -> NodeRef<T>
where
    T: Node,
{
    Rc::new(RefCell::new(node))
}

pub type Mat4x4 = nalgebra_glm::Mat4x4;

pub trait IntoVulkanWorldSpace {
    fn into_vulkan_coordinate_system(self) -> Mat4x4;
}

impl IntoVulkanWorldSpace for Transform {
    fn into_vulkan_coordinate_system(self) -> Mat4x4 {
        let matrix: Matrix4<f32> = self.into();
        let inv_axis_matrix = Matrix4::new(
            1.0, 0.0, 0.0, 0.0, 
            0.0, -1.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 1.0,
        );
        inv_axis_matrix * matrix
    }
}

#[derive(Clone, Copy)]
pub struct NodeProperties {
    pub transformation: Mat4x4,
    pub bounding_sphere: Sphere3Df, // TODO transform it into private
    pub visible: bool,
}

impl Default for NodeProperties {
    fn default() -> Self {
        Self {
            transformation: Mat4x4::identity(),
            bounding_sphere: Sphere3Df::empty(),
            visible: true,
        }
    }
}

impl NodeProperties {
    pub fn transformation(&mut self, value: Mat4x4) -> &mut Self {
        self.transformation = value;
        self
    }

    pub fn bounding_sphere(&mut self, value: Sphere3Df) -> &mut Self {
        self.bounding_sphere = value;
        self
    }

    pub fn get_bounding_sphere(&self) -> Sphere3Df {
        self.transformation.transform(&self.bounding_sphere)
    }

    pub fn set_visible(&mut self, value: bool) -> &mut Self {
        self.visible = value;
        self
    }
}
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
            device,
            builder,
            pipelines,
            render_pass,
            view_matrix,
            projection_matrix,
            view_normals_matrix,
        }
    }
}

pub trait Node {
    fn new_instance(&self) -> NodeRef<dyn Node>;

    fn properties_mut(&mut self) -> &mut NodeProperties;

    fn properties(&self) -> &NodeProperties;

    fn collect_command_buffers(
        &self,
        context: &mut CommandBuffersContext,
        window_state: &FrameStepInfo,
    );
}

pub trait MakeNode {
    type Node;
    fn make_node(&self, manager: &mut Manager) -> NodeRef<dyn Node>;
}

#[cfg(test)]
mod tests {
    use super::NodeProperties;

    #[test]
    fn test_basic_behavior() {
        let node = NodeProperties::default();
        assert!(node.bounding_sphere.is_empty());
    }
}
