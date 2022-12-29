use std::ops::Deref;

use align3d::bilateral::BilateralFilter;
use align3d::imagepointcloud::ImagePointCloud;
use align3d::io::dataset::RGBDDataset;
use align3d::io::slamtb::SlamTbDataset;
use ndarray::{Array2, Axis};
use rstest::*;

use align3d::io::{read_off, Geometry};
use align3d::mesh::compute_normals;
use align3d::pointcloud::PointCloud;

#[fixture]
pub fn sample_teapot() -> Geometry {
    let mut geometry = read_off("tests/data/teapot.off").unwrap();

    geometry.normals = Some(compute_normals(
        &geometry.points,
        geometry.faces.as_ref().unwrap(),
    ));
    geometry.colors = Some({
        let mut colors = Array2::<u8>::zeros((geometry.len_vertices(), 3));
        colors.axis_iter_mut(Axis(0)).for_each(|mut rgb| {
            rgb[1] = 255;
        });
        colors
    });

    geometry
}

#[fixture]
pub fn sample_teapot_pointcloud() -> PointCloud {
    PointCloud::from_geometry(sample_teapot())
}

#[fixture]
pub fn sample_rgbd_pointcloud() -> PointCloud {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();
    let mut item = dataset.get_item(0).unwrap();

    item.1.depth = {
        let mut filter = BilateralFilter::default();
        filter.filter(&item.1.depth);
        filter.result.unwrap()
    };

    let mut point_cloud = ImagePointCloud::from_rgbd_image(item.0, item.1);
    point_cloud.compute_normals();
    PointCloud::from(&point_cloud)
}