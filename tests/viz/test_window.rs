use std::sync::Arc;

use teapot_node::TeaPotNode;
use vulkano::memory::allocator::StandardMemoryAllocator;

mod triangle_node;
use triangle_node::TriangleNode;

mod teapot_node;

fn main() {
    let mut manager = align3d::viz::Manager::default();

    let memory_allocator = StandardMemoryAllocator::new_default(manager.device.clone());

    let mut window =
        align3d::viz::Window::create(&mut manager, Arc::new(TeaPotNode::new(&memory_allocator)));

    window.show();
}
