use std::{cell::RefCell, rc::Rc, sync::Arc};

use nalgebra::Vector3;

use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocator, MemoryUsage, StandardMemoryAllocator,
    },
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::Subpass,
};

use crate::{
    pointcloud::PointCloud,
    range_image::RangeImage,
    viz::{
        controllers::FrameStepInfo,
        misc::get_normal_matrix,
        node::{node_ref, CommandBuffersContext, MakeNode, Node, NodeProperties, NodeRef},
        sphere3d::Sphere3Df,
        Manager,
    },
};

use super::datatypes::{ColorU8, NormalF32, PositionF32};

pub struct VkPointCloud {
    pub points: Subbuffer<[PositionF32]>,
    pub normals: Subbuffer<[NormalF32]>,
    pub colors: Subbuffer<[ColorU8]>,
    number_of_points: usize,
}

impl VkPointCloud {
    pub fn from_pointcloud(
        memory_allocator: &(impl MemoryAllocator + ?Sized),
        pointcloud: &PointCloud,
    ) -> Arc<Self> {
        let create_info = BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        };
        let alloc_info = AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        };

        let number_of_points = pointcloud.len();

        Arc::new(Self {
            points: Buffer::from_iter(
                memory_allocator,
                create_info.clone(),
                alloc_info.clone(),
                pointcloud
                    .points
                    .iter()
                    .map(|v| PositionF32::new(v[0], v[1], v[2])),
            )
            .unwrap(),
            normals: Buffer::from_iter(
                memory_allocator,
                create_info.clone(),
                alloc_info.clone(),
                pointcloud
                    .normals
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|v| NormalF32::new(v[0], v[1], v[2])),
            )
            .unwrap(),
            colors: Buffer::from_iter(
                memory_allocator,
                create_info,
                alloc_info,
                pointcloud
                    .colors
                    .as_ref()
                    .unwrap()
                    .iter()
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
    pub fn new(point_cloud: Arc<VkPointCloud>) -> NodeRef<Self> {
        let points = point_cloud.points.read().unwrap();

        Rc::new(RefCell::new(Self {
            properties: NodeProperties {
                bounding_sphere: Sphere3Df::from_point_iter(
                    points
                        .iter()
                        .map(|p| Vector3::new(p.position[0], p.position[1], p.position[2])),
                ),
                ..Default::default()
            },
            point_cloud: point_cloud.clone(),
        }))
    }

    pub fn new_node(&self) -> NodeRef<Self> {
        Rc::new(RefCell::new(Self {
            properties: self.properties,
            point_cloud: self.point_cloud.clone(),
        }))
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "resources/shaders/vkpointcloud/pointcloud.vert",
    }
}

mod gs {
    vulkano_shaders::shader! {
        ty: "geometry",
        path: "resources/shaders/vkpointcloud/pointcloud.geom",
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

    fn new_instance(&self) -> NodeRef<dyn Node> {
        node_ref(VkPointCloudNode {
            properties: self.properties,
            point_cloud: self.point_cloud.clone(),
        })
    }

    fn collect_command_buffers(
        &self,
        context: &mut CommandBuffersContext,
        window_state: &FrameStepInfo,
    ) {
        if !self.properties.visible {
            return;
        }

        let pipeline = context
            .pipelines
            .entry("VkPointCloud".to_string())
            .or_insert_with(|| {
                let vs = vs::load(context.device.clone()).unwrap();
                let gs = gs::load(context.device.clone()).unwrap();
                let fs = fs::load(context.device.clone()).unwrap();

                GraphicsPipeline::start()
                    .vertex_input_state([
                        PositionF32::per_vertex(),
                        NormalF32::per_vertex(),
                        ColorU8::per_vertex(),
                    ])
                    .vertex_shader(vs.entry_point("main").unwrap(), ())
                    .input_assembly_state(
                        InputAssemblyState::new().topology(PrimitiveTopology::PointList),
                    )
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
                    .render_pass(Subpass::from(context.render_pass.clone(), 0).unwrap())
                    .build(context.device.clone())
                    .unwrap()
            });

        let memory_allocator =
            Arc::new(StandardMemoryAllocator::new_default(context.device.clone()));

        let uniform_buffer = SubbufferAllocator::new(
            memory_allocator,
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
        );

        let uniform_buffer_subbuffer = {
            let view_matrix = context.view_matrix * self.properties.transformation;
            let projection_worldview = context.projection_matrix * view_matrix;

            let uniform_data = vs::Data {
                normal_worldview: get_normal_matrix(&view_matrix),
                worldview: view_matrix.into(),
                projection_worldview: projection_worldview.into(),
            };

            let subbuffer = uniform_buffer.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = uniform_data;

            subbuffer
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

impl MakeNode for PointCloud {
    type Node = VkPointCloudNode;

    fn make_node(&self, manager: &mut Manager) -> NodeRef<dyn Node> {
        VkPointCloudNode::new(VkPointCloud::from_pointcloud(
            &manager.memory_allocator,
            self,
        ))
    }
}

impl MakeNode for RangeImage {
    type Node = VkPointCloudNode;

    fn make_node(&self, manager: &mut Manager) -> NodeRef<dyn Node> {
        VkPointCloudNode::new(VkPointCloud::from_pointcloud(
            &manager.memory_allocator,
            &PointCloud::from(self),
        ))
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
        println!("Using device: {}", manager.device_name());
        let renderer = OffscreenRenderer::new(&mut manager, 640, 480);
        (manager, renderer)
    }

    #[ignore]
    #[rstest]
    fn test_creation(
        offscreen_renderer: (Manager, OffscreenRenderer),
        sample_teapot_pointcloud: PointCloud,
    ) {
        let (manager, mut offscreen_renderer) = offscreen_renderer;
        let mem_alloc = StandardMemoryAllocator::new_default(manager.device);
        let node = VkPointCloudNode::new(VkPointCloud::from_pointcloud(
            &mem_alloc,
            &sample_teapot_pointcloud,
        ));
        offscreen_renderer.render(node);
    }
}
