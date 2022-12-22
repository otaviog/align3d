use std::sync::Arc;

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
            input_assembly::InputAssemblyState,
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
        controllers::WindowState,
        node::{CommandBuffersContext, Mat4x4, Node, NodeProperties},
    }, io::Geometry,
};

use super::datatypes::{ColorU8, NormalF32, PositionF32};

pub struct VkMesh {
    node_properties: NodeProperties,
    pub points: Arc<CpuAccessibleBuffer<[PositionF32]>>,
    pub normals: Arc<CpuAccessibleBuffer<[PositionF32]>>,
    pub colors: Arc<CpuAccessibleBuffer<[ColorU8]>>,
    number_of_points: usize,
}

impl VkMesh {
    pub fn from_geometry(
        memory_allocator: &(impl MemoryAllocator + ?Sized),
        geometry: Geometry,
    ) -> Self {
        let buffer_usage = BufferUsage {
            vertex_buffer: true,
            ..Default::default()
        };
        let number_of_points = geometry.len();
        Self {
            node_properties: NodeProperties {
                bounding_sphere: Sphere3Df::from_points(&geometry.points),
                ..Default::default()
            },
            points: CpuAccessibleBuffer::from_iter(
                memory_allocator,
                buffer_usage,
                false,
                geometry
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
                    .unwrap()
                    .axis_iter(Axis(0))
                    .map(|v| PositionF32::new(v[0], v[1], v[2])),
            )
            .unwrap(),
            colors: CpuAccessibleBuffer::from_iter(
                memory_allocator,
                buffer_usage,
                false,
                pointcloud
                    .colors
                    .unwrap()
                    .axis_iter(Axis(0))
                    .map(|v| ColorU8::new(v[0], v[1], v[2])),
            )
            .unwrap(),
            number_of_points: number_of_points,
        }
    }

    pub fn len(&self) -> usize {
        self.number_of_points
    }
}

impl Node for VkMesh {
    fn transformation(&self) -> &Mat4x4 {
        self.node_properties.transformation()
    }

    fn bounding_sphere(&self) -> &crate::bounds::Sphere3Df {
        self.node_properties.bounding_sphere()
    }

    fn collect_command_buffers(
        &self,
        context: &mut CommandBuffersContext,
        window_state: &WindowState,
    ) {
        let pipeline = context
            .pipelines
            .entry("VkPointCloud".to_string())
            .or_insert_with(|| {
                let vs = vs::load(context.device.clone()).unwrap();
                let fs = fs::load(context.device.clone()).unwrap();
                GraphicsPipeline::start()
                    .render_pass(Subpass::from(context.render_pass.clone(), 0).unwrap())
                    .vertex_input_state(
                        BuffersDefinition::new()
                            .vertex::<PositionF32>()
                            .vertex::<NormalF32>(), //.vertex::<ColorU8>(),
                    )
                    .input_assembly_state(
                        InputAssemblyState::new() //.topology(
                        //vulkano::pipeline::graphics::input_assembly::PrimitiveTopology::PointList)
                    )
                    .vertex_shader(vs.entry_point("main").unwrap(), ())
                    .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                        Viewport {
                            origin: [0.0, 0.0],
                            dimensions: window_state.window_size,
                            depth_range: 0.0..1.0,
                        },
                    ]))
                    .fragment_shader(fs.entry_point("main").unwrap(), ())
                    .depth_stencil_state(DepthStencilState::simple_depth_test())
                    .build(context.device.clone())
                    .unwrap()
            });
        let memory_allocator =
            Arc::new(StandardMemoryAllocator::new_default(context.device.clone()));

        let proj = cgmath::perspective(cgmath::Rad(std::f32::consts::FRAC_PI_2), 1.0, 0.01, 100.0);
        let uniform_buffer_subbuffer = {
            let uniform_data = vs::ty::Data {
                world: Mat4x4::identity().into(),
                view: context.view_matrix.into(),
                //proj: context.projection_matrix.into(),
                //view: view.into(),
                proj: proj.into(),
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
                    self.points.clone(),
                    self.normals.clone(),
                    //self.colors.clone(),
                ),
            )
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .draw(self.len() as u32, 1, 0, 0)
            //.draw(3 as u32, 1, 0, 0)
            .unwrap();
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "resources/shaders/vkpointcloud/vert.glsl",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "resources/shaders/vkpointcloud/frag.glsl"
    }
}

#[cfg(test)]
mod tests {
    use crate::pointcloud::PointCloud;
    use crate::unit_test::sample_teapot_pointcloud;
    use rstest::*;
    use vulkano::memory::allocator::StandardMemoryAllocator;

    use crate::viz::Manager;

    use super::*;

    #[fixture]
    fn vk_manager() -> Manager {
        Manager::default()
    }

    #[rstest]
    fn test_creation(vk_manager: Manager, sample_teapot_pointcloud: PointCloud) {
        let mem_alloc = StandardMemoryAllocator::new_default(vk_manager.device.clone());
        VkPointCloud::from_pointcloud(&mem_alloc, sample_teapot_pointcloud);
    }
}
