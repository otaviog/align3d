use std::sync::Arc;

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    memory::allocator::MemoryAllocator,
};

use crate::pointcloud::{self, PointCloud};

struct VkPointCloud {
    points: Arc<CpuAccessibleBuffer>,
    normals: Arc<CpuAccessibleBuffer>,
    colors: Arc<CpuAccessibleBuffer>,
}

impl From<PointCloud> for VkPointCloud {
    fn from(pointCloud: PointCloud) -> Self {}
}

impl VkPointCloud {
    pub fn from_pointcloud(
        &memory_allocator: &impl MemoryAllocator, // TODO: Pass/get from context.
        pointcloud: PointCloud,
    ) -> Self {
        VkPointCloud {
            points: CpuAccessibleBuffer::from_data(
                &memory_allocator,
                BufferUsage {
                    vertex_buffer: true,
                    ..Default::default()
                },
                false,
                pointcloud.points.as_ptr(),
            ),
            normals: CpuAccessibleBuffer::from_data(
                &memory_allocator,
                BufferUsage {
                    vertex_buffer: true,
                    ..Default::default()
                },
                false,
                pointcloud.normals.as_ptr(),
            ),
            colors: CpuAccessibleBuffer::from_data(
                &memory_allocator,
                BufferUsage {
                    vertex_buffer: true,
                    ..Default::default()
                },
                false,
                pointcloud.colors.as_ptr(),
            ),
        }
    }
}

impl Node for VkPointCloud {
    fn collect_command_buffers(builder: PrimaryAutoCommandBuffer, pipelines: HashMap<String, Arc<GraphicsPipeline> ) {

    }
}

#[cfg(test)]
mod tests {
    fn test_creation() {
        
    }
}