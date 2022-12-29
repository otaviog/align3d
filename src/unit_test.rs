use ndarray::{Array2, Axis};
use nshare::ToNdarray2;
use rstest::*;

use crate::io::{read_off, Geometry};
use crate::mesh::compute_normals;
use crate::pointcloud::PointCloud;

#[fixture]
pub fn sample_teapot_geometry() -> Geometry {
    let mut geometry = read_off("tests/data/teapot.off").unwrap();
    let num_vertices = geometry.len_vertices();

    geometry.normals = Some(compute_normals(
        &geometry.points,
        geometry.faces.as_ref().unwrap(),
    ));
    geometry.colors = Some({
        let mut colors = Array2::<u8>::zeros((num_vertices, 3));
        colors.axis_iter_mut(Axis(0)).for_each(|mut rgb| {
            rgb[0] = 255;
        });
        colors
    });

    geometry
}

#[fixture]
pub fn sample_teapot_pointcloud() -> PointCloud {
    let mut geometry = read_off("tests/data/teapot.off").unwrap();
    let num_vertices = geometry.len_vertices();
    geometry.normals = Some(Array2::<f32>::zeros((num_vertices, 3)));
    geometry.colors = Some(Array2::<u8>::zeros((num_vertices, 3)));

    PointCloud::from_geometry(geometry)
}

#[fixture]
pub fn bloei_luma16() -> Array2<u16> {
    let mut image = image::io::Reader::open("tests/data/images/bloei.jpg")
        .unwrap()
        .decode()
        .unwrap()
        .into_luma16()
        .into_ndarray2();

    image.iter_mut().for_each(|v| {
        *v /= std::u16::MAX / 5000;
    });
    image
}
