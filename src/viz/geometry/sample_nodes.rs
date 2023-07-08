use nalgebra::Vector3;
use ndarray::Array1;

use crate::{
    io::read_off,
    mesh::compute_normals,
    viz::{node::NodeRef, Manager},
};

use super::{VkMesh, VkMeshNode};

pub fn teapot_node(manager: &Manager) -> NodeRef<VkMeshNode> {
    let geometry = {
        let mut geometry = read_off("tests/data/teapot.off").unwrap();

        geometry.normals = Some(compute_normals(
            &geometry.points.view(),
            &geometry.faces.as_ref().unwrap().view(),
        ));
        geometry.colors = Some({
            let mut colors = Array1::<Vector3<u8>>::zeros(geometry.len_vertices());
            colors.iter_mut().for_each(|rgb| {
                rgb[0] = 255;
            });
            colors
        });
        geometry
    };
    VkMeshNode::new(VkMesh::from_geometry(&manager.memory_allocator, &geometry))
}
