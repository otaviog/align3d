use std::{collections::HashMap, sync::Arc};

use align3d::{
    bounds::Box3Df,
    viz::node::{Mat4x4, Node, NodeProperties},
    viz::{
        geometry::{Array2f32, Array3f32},
        node::CommandBuffersContext,
    },
};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    device::Device,
    memory::allocator::{MemoryAllocator, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState, render_pass, vertex_input::BuffersDefinition,
            viewport::ViewportState,
        },
        GraphicsPipeline,
    },
    render_pass::Subpass,
};

pub struct TriangleNode {
    node_properties: NodeProperties,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Array2f32]>>,
}

impl TriangleNode {
    pub fn new(memory_allocator: &(impl MemoryAllocator + ?Sized)) -> Self {
        let vertices = [
            Array2f32::new(-0.5, -0.25),
            Array2f32::new(0.0, 0.5),
            Array2f32::new(0.25, -0.1),
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

        TriangleNode {
            node_properties: Default::default(),
            vertex_buffer: vertex_buffer,
        }
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
        #version 450

        layout(location = 0) in vec2 position;

        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
        }
    "
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
        #version 450

        layout(location = 0) out vec4 f_color;

        void main() {
            f_color = vec4(1.0, 0.0, 0.0, 1.0);
        }
    "
    }
}

impl Node for TriangleNode {
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
                    .vertex_input_state(BuffersDefinition::new().vertex::<Array2f32>())
                    .input_assembly_state(InputAssemblyState::new())
                    .vertex_shader(vs.entry_point("main").unwrap(), ())
                    .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
                    .fragment_shader(fs.entry_point("main").unwrap(), ())
                    .build(context.device.clone())
                    .unwrap()
            });

        context
            .builder
            .bind_pipeline_graphics(pipeline.clone())
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .draw(3 as u32, 1, 0, 0)
            .unwrap();
    }
}
