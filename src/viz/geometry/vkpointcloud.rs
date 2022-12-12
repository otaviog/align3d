use std::{collections::HashMap, sync::Arc};

use ndarray::Axis;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    memory::allocator::MemoryAllocator,
    pipeline::GraphicsPipeline, device::Device,
};

use crate::{pointcloud::PointCloud, bounds::Box3Df, viz::node::{Mat4x4, Node, NodeProperties, CommandBuffersContext}};

use super::datatypes::{Array3f32, Array3u8};

pub struct VkPointCloud {
    node_properties: NodeProperties,
    pub points: Arc<CpuAccessibleBuffer<[Array3f32]>>,
    pub normals: Arc<CpuAccessibleBuffer<[Array3f32]>>,
    pub colors: Arc<CpuAccessibleBuffer<[Array3u8]>>,
}

impl VkPointCloud {
    pub fn from_pointcloud(
        memory_allocator: &(impl MemoryAllocator + ?Sized),
        pointcloud: PointCloud,
    ) -> Self {
        let buffer_usage = BufferUsage {
            vertex_buffer: true,
            ..Default::default()
        };

        VkPointCloud {
            node_properties: Default::default(),
            points: CpuAccessibleBuffer::from_iter(
                memory_allocator,
                buffer_usage,
                false,
                pointcloud
                    .points
                    .axis_iter(Axis(0))
                    .map(|v| Array3f32::new(v[0], v[1], v[2])),
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
                    .map(|v| Array3f32::new(v[0], v[1], v[2])),
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
                    .map(|v| Array3u8::new(v[0], v[1], v[2])),
            )
            .unwrap(),
        }
    }
}

impl Node for VkPointCloud {
    fn transformation(&self) -> &Mat4x4 {
        self.node_properties.transformation()
    }

    fn bounding_box(&self) -> &Box3Df {
        self.node_properties.bounding_box()
    }

    fn collect_command_buffers(&self,
        context: &mut CommandBuffersContext
    ) {
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
        let pcl = VkPointCloud::from_pointcloud(&mem_alloc, sample_teapot_pointcloud);
    }
}
