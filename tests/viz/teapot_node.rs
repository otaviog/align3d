use std::sync::Arc;

use align3d::{
    bounds::Box3Df,
    viz::node::{Mat4x4, Node, NodeProperties},
    viz::{geometry::Array3f32, node::CommandBuffersContext},
};
use cgmath::{Matrix4, Point3, Vector3};
use nalgebra_glm::Mat4;
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

pub struct TeaPotNode {
    node_properties: NodeProperties,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Array3f32]>>,
}

impl TeaPotNode {
    pub fn new(memory_allocator: &(impl MemoryAllocator + ?Sized)) -> Self {
        let vertices = [
            Array3f32::new(-0.5, -0.25, 0.0),
            Array3f32::new(0.0, 0.5, 0.0),
            Array3f32::new(0.25, -0.1, 0.0),
        ];
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            memory_allocator,
            BufferUsage {
                vertex_buffer: true,
                ..Default::default()
            },
            false,
            vertices,
        )
        .unwrap();

        TeaPotNode {
            node_properties: Default::default(),
            vertex_buffer: vertex_buffer,
        }
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "tests/data/shaders/vert.glsl",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "tests/data/shaders/frag.glsl"
    }
}

impl Node for TeaPotNode {
    fn transformation(&self) -> &Mat4x4 {
        self.node_properties.transformation()
    }

    fn bounding_box(&self) -> &Box3Df {
        self.node_properties.bounding_box()
    }

    fn collect_command_buffers(&self, context: &mut CommandBuffersContext) {
        let pipeline = context
            .pipelines
            .entry("Triangle".to_string())
            .or_insert_with(|| {
                let vs = vs::load(context.device.clone()).unwrap();
                let fs = fs::load(context.device.clone()).unwrap();
                GraphicsPipeline::start()
                    .render_pass(Subpass::from(context.render_pass.clone(), 0).unwrap())
                    .vertex_input_state(BuffersDefinition::new().vertex::<Array3f32>())
                    .input_assembly_state(InputAssemblyState::new())
                    .vertex_shader(vs.entry_point("main").unwrap(), ())
                    .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                        Viewport {
                            origin: [0.0, 0.0],
                            dimensions: [640.0, 480.0],
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

        let view = Matrix4::look_at_rh(
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        let proj = cgmath::perspective(cgmath::Rad(std::f32::consts::FRAC_PI_2), 1.0, 0.01, 100.0);
        let uniform_buffer_subbuffer = {
            let uniform_data = vs::ty::Data {
                world: Mat4::identity().into(),
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
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .draw(3 as u32, 1, 0, 0)
            .unwrap();
    }
}
