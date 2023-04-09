use std::{cell::RefCell, rc::Rc, sync::Arc};

use ndarray::{Array, Axis};
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
    io::Geometry,
    viz::{
        controllers::FrameStepInfo,
        node::{node_ref, CommandBuffersContext, MakeNode, Node, NodeProperties, NodeRef},
        Manager,
    },
};

use super::datatypes::{ColorU8, NormalF32, PositionF32};

/// Triangular mesh in GPU.
pub struct VkMesh {
    /// Vertex points.
    pub points: Arc<CpuAccessibleBuffer<[PositionF32]>>,
    // Indices.
    pub indices: Arc<CpuAccessibleBuffer<[u32]>>,
    /// Vertex normals.
    pub normals: Option<Arc<CpuAccessibleBuffer<[PositionF32]>>>,
    /// RGB colors.
    pub colors: Option<Arc<CpuAccessibleBuffer<[ColorU8]>>>,
    bounding_sphere: Sphere3Df,
    number_of_vertex: usize,
    number_of_faces: usize,
}

impl VkMesh {
    /// Constructs from a geometry structure. The structure must
    /// contain some faces.
    ///
    /// # Arguments
    ///
    /// * `memory_allocator` - Vulkan's memory allocator.
    /// * `geometry` - Target geometry.
    pub fn from_geometry(
        memory_allocator: &(impl MemoryAllocator + ?Sized),
        geometry: &Geometry,
    ) -> Arc<Self> {
        let vertex_buffer_usage = BufferUsage {
            vertex_buffer: true,
            ..Default::default()
        };

        let number_of_points = geometry.len_vertices();
        let number_of_faces = geometry.len_faces();
        Arc::new(Self {
            points: CpuAccessibleBuffer::from_iter(
                memory_allocator,
                vertex_buffer_usage,
                false,
                geometry
                    .points
                    .axis_iter(Axis(0))
                    .map(|v| PositionF32::new(v[0], v[1], v[2])),
            )
            .unwrap(),
            indices: CpuAccessibleBuffer::from_iter(
                memory_allocator,
                BufferUsage {
                    index_buffer: true,
                    ..Default::default()
                },
                false,
                Array::from_iter(geometry.faces.as_ref().unwrap().iter().cloned())
                    .iter()
                    .rev()
                    .map(|v| *v as u32),
            )
            .unwrap(),
            normals: Some(
                CpuAccessibleBuffer::from_iter(
                    memory_allocator,
                    vertex_buffer_usage,
                    false,
                    geometry
                        .normals
                        .as_ref()
                        .unwrap()
                        .axis_iter(Axis(0))
                        .map(|v| PositionF32::new(v[0], v[1], v[2])),
                )
                .unwrap(),
            ),
            colors: Some(
                CpuAccessibleBuffer::from_iter(
                    memory_allocator,
                    vertex_buffer_usage,
                    false,
                    geometry
                        .colors
                        .as_ref()
                        .unwrap()
                        .axis_iter(Axis(0))
                        .map(|v| ColorU8::new(v[0], v[1], v[2])),
                )
                .unwrap(),
            ),
            bounding_sphere: Sphere3Df::from_points(&geometry.points),
            number_of_vertex: number_of_points,
            number_of_faces,
        })
    }

    /// Number of vertices.
    pub fn len_vertex(&self) -> usize {
        self.number_of_vertex
    }

    /// Number of faces.
    pub fn len_face(&self) -> usize {
        self.number_of_faces
    }

    /// Bounding sphere of its points.
    pub fn bounding_sphere(&self) -> &Sphere3Df {
        &self.bounding_sphere
    }
}

/// A rendering node for VkMeshes.
pub struct VkMeshNode {
    node_properties: NodeProperties,
    /// Mesh instance.
    pub mesh: Arc<VkMesh>,
}

impl VkMeshNode {
    /// Creates a new node with a mesh.
    ///
    /// # Arguments
    ///
    /// * `mesh`: The mesh buffer instance.
    pub fn new(mesh: Arc<VkMesh>) -> NodeRef<Self> {
        Rc::new(RefCell::new(Self {
            node_properties: NodeProperties {
                bounding_sphere: *mesh.bounding_sphere(),
                ..Default::default()
            },
            mesh: mesh.clone(),
        }))
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "resources/shaders/vkmesh/mesh.vert",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "resources/shaders/vkmesh/mesh.frag"
    }
}

impl Node for VkMeshNode {
    fn new_instance(&self) -> NodeRef<dyn Node> {
        node_ref(VkMeshNode {
            node_properties: self.node_properties,
            mesh: self.mesh.clone(),
        })
    }

    fn properties(&self) -> &NodeProperties {
        &self.node_properties
    }

    fn properties_mut(&mut self) -> &mut NodeProperties {
        &mut self.node_properties
    }

    fn collect_command_buffers(
        &self,
        context: &mut CommandBuffersContext,
        frame_info: &FrameStepInfo,
    ) {
        let pipeline = context
            .pipelines
            .entry("VkMesh".to_string())
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
                        InputAssemblyState::new().topology(PrimitiveTopology::TriangleList),
                    )
                    .vertex_shader(vs.entry_point("main").unwrap(), ())
                    .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                        Viewport {
                            origin: [0.0, 0.0],
                            dimensions: frame_info.viewport_size,
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

        let uniform_buffer_subbuffer = {
            let uniform_data = vs::ty::Data {
                worldview: context.view_matrix.into(),
                worldview_normals: context.view_normals_matrix.into(),
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
                    self.mesh.points.clone(),
                    self.mesh.normals.as_ref().unwrap().clone(),
                    self.mesh.colors.as_ref().unwrap().clone(),
                ),
            )
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .bind_index_buffer(self.mesh.indices.clone())
            .draw_indexed((self.mesh.len_face() * 3) as u32, 1, 0, 0, 0)
            .unwrap();
    }
}

impl MakeNode for Geometry {
    type Node = VkMeshNode;

    fn make_node(&self, manager: &mut Manager) -> NodeRef<dyn Node> {
        VkMeshNode::new(VkMesh::from_geometry(&manager.memory_allocator, self))
    }
}

#[cfg(test)]
mod tests {
    use crate::{unit_test::sample_teapot_geometry, viz::OffscreenRenderer};
    use rstest::*;
    use vulkano::memory::allocator::StandardMemoryAllocator;

    use crate::viz::Manager;

    use super::*;

    #[rstest]
    fn test_creation(sample_teapot_geometry: Geometry) {
        let mut vk_manager = Manager::default();
        let mem_alloc = StandardMemoryAllocator::new_default(vk_manager.device.clone());
        let mut render = OffscreenRenderer::new(&mut vk_manager, 640, 480);

        let node = VkMeshNode::new(VkMesh::from_geometry(&mem_alloc, &sample_teapot_geometry));
        render.render(node);
    }
}
