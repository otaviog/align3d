use align3d::viz::{
    geometry::{VkMesh, VkMeshNode},
    Manager, Window,
};

mod test_data;
use test_data::sample_teapot;

fn main() {
    let mut manager = Manager::default();
    let geometry = VkMesh::from_geometry(&manager.memory_allocator, &sample_teapot());

    Window::create(&mut manager, VkMeshNode::new(geometry)).show();
}
