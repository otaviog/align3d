use nalgebra::Vector3;
use ndarray::Array1;
use rstest::fixture;

use crate::{
    io::{read_off, Geometry},
    mesh::compute_normals,
};

#[fixture]
pub fn sample_teapot_geometry() -> Geometry {
    let mut geometry = read_off("tests/data/teapot.off").unwrap();
    let num_vertices = geometry.len_vertices();

    geometry.normals = Some(compute_normals(
        &geometry.points.view(),
        &geometry.faces.as_ref().unwrap().view(),
    ));
    geometry.colors = Some({
        let mut colors = Array1::<Vector3<u8>>::zeros(num_vertices);
        colors.iter_mut().for_each(|rgb| {
            rgb[0] = 255;
        });
        colors
    });

    geometry
}
