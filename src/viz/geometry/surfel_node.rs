use nalgebra::Vector3;
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        BufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    memory::allocator::StandardMemoryAllocator,
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
    bounds::Sphere3Df,
    surfel::{AttrColorMaskAge, SurfelModel, AttrPositionConfidence, AttrNormalRadius},
    viz::{
        controllers::FrameStepInfo,
        misc::get_normal_matrix,
        node::{node_ref, CommandBuffersContext, MakeNode, Mat4x4, Node, NodeProperties, NodeRef},
        Manager,
    },
};
use std::sync::{Arc, Mutex};

pub struct SurfelNode {
    pub properties: NodeProperties,
    model: Arc<Mutex<SurfelModel>>,
}

impl SurfelNode {
    pub fn new(surfel_model: Arc<Mutex<SurfelModel>>) -> NodeRef<Self> {
        node_ref(Self {
            properties: NodeProperties {
                bounding_sphere: Sphere3Df {
                    center: Vector3::new(0.0, 0.0, 0.0),
                    radius: 3.0,
                },
                visible: true,
                transformation: Mat4x4::identity(),
            },
            model: surfel_model,
        })
    }
}

impl MakeNode for Arc<Mutex<SurfelModel>> {
    type Node = SurfelNode;

    fn make_node(&self, _manager: &mut Manager) -> NodeRef<dyn Node> {
        SurfelNode::new(self.clone())
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "resources/shaders/surfel/surfel_model.vert",
    }
}

mod gs {
    vulkano_shaders::shader! {
        ty: "geometry",
        path: "resources/shaders/surfel/surfel_model.geom",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "resources/shaders/surfel/surfel_model.frag"
    }
}

impl Node for SurfelNode {
    fn properties(&self) -> &NodeProperties {
        &self.properties
    }

    fn properties_mut(&mut self) -> &mut NodeProperties {
        &mut self.properties
    }

    fn new_instance(&self) -> NodeRef<dyn Node> {
        // node_ref(SurfelNode {
        //     properties: self.properties,
        //     point_cloud: self.point_cloud.clone(),
        // })
        panic!("SurfelNode::new_instance() not implemented")
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
            .entry("SurfelModel".to_string())
            .or_insert_with(|| {
                let vs = vs::load(context.device.clone()).unwrap();
                let gs = gs::load(context.device.clone()).unwrap();
                let fs = fs::load(context.device.clone()).unwrap();
                GraphicsPipeline::start()
                    .render_pass(Subpass::from(context.render_pass.clone(), 0).unwrap())
                    .vertex_input_state([
                        AttrPositionConfidence::per_vertex(),
                        AttrNormalRadius::per_vertex(),
                        AttrColorMaskAge::per_vertex(),
                    ])
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
                    .render_pass(Subpass::from(context.render_pass.clone(), 0).unwrap())
                    .build(context.device.clone())
                    .unwrap()
            });
        let mut model = self.model.lock().unwrap();
        model.swap_graphics(context.device.clone(), context.queue.clone());

        let memory_allocator =
            Arc::new(StandardMemoryAllocator::new_default(context.device.clone()));

        let uniform_buffer = SubbufferAllocator::new(
            memory_allocator.clone(),
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
                    model.graphics.position.clone(),
                    model.graphics.normal.clone(),
                    model.graphics.color_n_mask.clone(),
                ),
            )
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .draw(model.size() as u32, 1, 0, 0)
            .unwrap();
            

        drop(model);
    }
}

#[cfg(test)]
mod tests {
    use rstest::*;

    use crate::viz::{Manager, OffscreenRenderer};

    #[fixture]
    fn offscreen_renderer() -> (Manager, OffscreenRenderer) {
        let mut manager = Manager::default();
        println!("Using device: {}", manager.device_name());
        let renderer = OffscreenRenderer::new(&mut manager, 640, 480);
        (manager, renderer)
    }

    // #[ignore]
    // #[rstest]
    // fn test_creation(
    //     offscreen_renderer: (Manager, OffscreenRenderer),
    //     sample_teapot_pointcloud: PointCloud,
    // ) {
    //     let (manager, mut offscreen_renderer) = offscreen_renderer;
    //     let mem_alloc = StandardMemoryAllocator::new_default(manager.device.clone());
    //     let node = VkPointCloudNode::new(VkPointCloud::from_pointcloud(
    //         &mem_alloc,
    //         &sample_teapot_pointcloud,
    //     ));
    //     offscreen_renderer.render(node);
    // }
}
