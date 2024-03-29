use align3d::bilateral::BilateralFilter;
use align3d::io::dataset::{RgbdDataset, SlamTbDataset};
use align3d::pointcloud::PointCloud;
use align3d::range_image::RangeImage;
use nalgebra::Vector3;
use ndarray::Array1;
use rstest::*;

use align3d::io::{read_off, Geometry};
use align3d::mesh::compute_normals;

#[fixture]
pub fn sample_teapot() -> Geometry {
    let mut geometry = read_off("tests/data/teapot.off").unwrap();

    geometry.normals = Some(compute_normals(
        &geometry.points.view(),
        &geometry.faces.as_ref().unwrap().view(),
    ));
    geometry.colors = Some({
        let mut colors = Array1::<Vector3<u8>>::zeros(geometry.len_vertices());
        colors.iter_mut().for_each(|rgb| {
            rgb[1] = 255;
        });
        colors
    });

    geometry
}

#[fixture]
pub fn sample_rgbd_pointcloud() -> PointCloud {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();
    let mut frame = dataset.get(0).unwrap();

    frame.image.depth = {
        let filter = BilateralFilter::default();
        filter.filter(&frame.image.depth)
    };

    let mut point_cloud = RangeImage::from_rgbd_frame(&frame);
    point_cloud.compute_normals();
    PointCloud::from(&point_cloud)
}
