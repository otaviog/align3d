use std::sync::Arc;

use align3d::{viz::geometry::VkPointCloud};
use vulkano::memory::allocator::StandardMemoryAllocator;

mod test_data;
use test_data::sample_teapot_pointcloud;

fn main() {
    let sample_teapot_pointcloud = sample_teapot_pointcloud();

    let mut manager = align3d::viz::Manager::default();

    let memory_allocator = StandardMemoryAllocator::new_default(manager.device.clone());

    let mut window = align3d::viz::Window::create(
        &mut manager,
        Arc::new(VkPointCloud::from_pointcloud(
            &memory_allocator,
            sample_teapot_pointcloud,
        )),
    );

    window.show();
}
