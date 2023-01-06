use std::{rc::Rc, cell::RefCell};

use ndarray::{Array2, Axis};

use crate::{io::read_off, mesh::compute_normals, viz::Manager};

use super::{VkMeshNode, VkMesh};

pub fn teapot_node(manager: &Manager) -> Rc<RefCell<VkMeshNode>> {
    let geometry = {
        let mut geometry = read_off("tests/data/teapot.off").unwrap();

        geometry.normals = Some(compute_normals(
            &geometry.points,
            geometry.faces.as_ref().unwrap(),
        ));
        geometry.colors = Some({
            let mut colors = Array2::<u8>::zeros((geometry.len_vertices(), 3));
            colors.axis_iter_mut(Axis(0)).for_each(|mut rgb| {
                rgb[0] = 255;
            });
            colors
        });
        geometry
    };
    VkMeshNode::new(VkMesh::from_geometry(&manager.memory_allocator, &geometry))
}