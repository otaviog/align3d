use std::sync::Arc;

use nalgebra::Vector3;
use nalgebra_glm::{Mat3x3, Mat4x4};
use ndarray::Axis;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    memory::allocator::{MemoryAllocator, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::Subpass,
};

use crate::{
    bounds::Sphere3Df,
    pointcloud::PointCloud,
    viz::{
        controllers::FrameStepInfo,
        node::{CommandBuffersContext, Node, NodeProperties},
        Manager,
    },
};

use super::datatypes::{ColorU8, NormalF32, PositionF32};

pub struct VkPointCloud {
    pub points: Arc<CpuAccessibleBuffer<[PositionF32]>>,
    pub normals: Arc<CpuAccessibleBuffer<[NormalF32]>>,
    pub colors: Arc<CpuAccessibleBuffer<[ColorU8]>>,
    number_of_points: usize,
}

impl VkPointCloud {
    pub fn from_pointcloud(
        memory_allocator: &(impl MemoryAllocator + ?Sized),
        pointcloud: &PointCloud,
    ) -> Arc<Self> {
        let buffer_usage = BufferUsage {
            vertex_buffer: true,
            ..Default::default()
        };
        let number_of_points = pointcloud.len();
        Arc::new(Self {
            points: CpuAccessibleBuffer::from_iter(
                memory_allocator,
                buffer_usage,
                false,
                pointcloud
                    .points
                    .axis_iter(Axis(0))
                    .map(|v| PositionF32::new(v[0], v[1], v[2])),
            )
            .unwrap(),
            normals: CpuAccessibleBuffer::from_iter(
                memory_allocator,
                buffer_usage,
                false,
                pointcloud
                    .normals
                    .as_ref()
                    .unwrap()
                    .axis_iter(Axis(0))
                    .map(|v| NormalF32::new(v[0], v[1], v[2])),
            )
            .unwrap(),
            colors: CpuAccessibleBuffer::from_iter(
                memory_allocator,
                buffer_usage,
                false,
                pointcloud
                    .colors
                    .as_ref()
                    .unwrap()
                    .axis_iter(Axis(0))
                    .map(|v| ColorU8::new(v[2], v[1], v[0])),
            )
            .unwrap(),
            number_of_points,
        })
    }

    pub fn len(&self) -> usize {
        self.number_of_points
    }

    pub fn is_empty(&self) -> bool {
        self.number_of_points == 0
    }
}

pub struct VkPointCloudNode {
    pub properties: NodeProperties,
    point_cloud: Arc<VkPointCloud>,
}

impl VkPointCloudNode {
    pub fn new(point_cloud: Arc<VkPointCloud>) -> Arc<Self> {
        let points = point_cloud.points.read().unwrap();

        Arc::new(Self {
            properties: NodeProperties {
                bounding_sphere: Sphere3Df::from_point_iter(
                    points
                        .iter()
                        .map(|p| Vector3::new(p.position[0], p.position[1], p.position[2])),
                ),
                ..Default::default()
            },
            point_cloud: point_cloud.clone(),
        })
    }

    pub fn load(manager: &Manager, point_cloud: &PointCloud) -> Arc<Self> {
        Self::new(VkPointCloud::from_pointcloud(
            &manager.memory_allocator,
            &point_cloud,
        ))
    }

    pub fn new_node(&self) -> Arc<Self> {
        Arc::new(Self {
            properties: self.properties,
            point_cloud: self.point_cloud.clone(),
        })
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "resources/shaders/vkpointcloud/pointcloud.vert",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

mod gs {
    vulkano_shaders::shader! {
        ty: "geometry",
        path: "resources/shaders/vkpointcloud/pointcloud.geom",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        }
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "resources/shaders/vkpointcloud/pointcloud.frag"
    }
}

impl Node for VkPointCloudNode {
    fn properties(&self) -> &NodeProperties {
        &self.properties
    }

    fn properties_mut(&mut self) -> &mut NodeProperties {
        &mut self.properties
    }

    fn collect_command_buffers(
        &self,
        context: &mut CommandBuffersContext,
        window_state: &FrameStepInfo,
    ) {
        let pipeline = context
            .pipelines
            .entry("VkPointCloud".to_string())
            .or_insert_with(|| {
                let vs = vs::load(context.device.clone()).unwrap();
                let gs = gs::load(context.device.clone()).unwrap();
                let fs = fs::load(context.device.clone()).unwrap();
                GraphicsPipeline::start()
                    .render_pass(Subpass::from(context.render_pass.clone(), 0).unwrap())
                    .vertex_input_state(
                        BuffersDefinition::new()
                            .vertex::<PositionF32>()
                            .vertex::<NormalF32>()
                            .vertex::<ColorU8>(),
                    )
                    .input_assembly_state(
                        InputAssemblyState::new().topology(PrimitiveTopology::PointList),
                    )
                    .vertex_shader(vs.entry_point("main").unwrap(), ())
                    .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                        Viewport {
                            origin: [0.0, 0.0],
                            dimensions: window_state.viewport_size,
                            depth_range: 0.0..1.0,
                        },
                    ]))
                    .geometry_shader(gs.entry_point("main").unwrap(), ())
                    .fragment_shader(fs.entry_point("main").unwrap(), ())
                    .depth_stencil_state(DepthStencilState::simple_depth_test())
                    .build(context.device.clone())
                    .unwrap()
            });
        let memory_allocator =
            Arc::new(StandardMemoryAllocator::new_default(context.device.clone()));

        let uniform_buffer_subbuffer = {
            let uniform_data = vs::ty::Data {
                normal_worldview: Mat3x3::identity().into(),
                worldview: context.view_matrix.into(),
                projection_worldview: (context.projection_matrix * context.view_matrix).into(),
                _dummy0: [0; 12],
            };

            CpuAccessibleBuffer::from_data(
                &memory_allocator,
                BufferUsage {
                    uniform_buffer: true,
                    ..Default::default()
                },
                false,
                uniform_data,
            )
            .unwrap()
        };
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(context.device.clone());

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer)],
        )
        .unwrap();

        context
            .builder
            .bind_pipeline_graphics(pipeline.clone())
            .bind_vertex_buffers(
                0,
                (
                    self.point_cloud.points.clone(),
                    self.point_cloud.normals.clone(),
                    self.point_cloud.colors.clone(),
                ),
            )
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .draw(self.point_cloud.len() as u32, 1, 0, 0)
            .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use crate::pointcloud::PointCloud;
    use crate::unit_test::sample_teapot_pointcloud;
    use rstest::*;
    use vulkano::memory::allocator::StandardMemoryAllocator;

    use crate::viz::{Manager, OffscreenRenderer};

    use super::*;

    #[fixture]
    fn offscreen_renderer() -> (Manager, OffscreenRenderer) {
        let mut manager = Manager::default();
        println!("Using device: {}", manager.get_device_name());
        let renderer = OffscreenRenderer::new(&mut manager, 640, 480);
        (manager, renderer)
    }

    #[rstest]
    fn test_creation(
        offscreen_renderer: (Manager, OffscreenRenderer),
        sample_teapot_pointcloud: PointCloud,
    ) {
        let (manager, mut offscreen_renderer) = offscreen_renderer;
        let mem_alloc = StandardMemoryAllocator::new_default(manager.device.clone());
        let node = VkPointCloudNode::new(VkPointCloud::from_pointcloud(
            &mem_alloc,
            &sample_teapot_pointcloud,
        ));
        offscreen_renderer.render(node);
    }
}
