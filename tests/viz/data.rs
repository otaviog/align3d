use align3d::bilateral::BilateralFilter;
use align3d::io::dataset::{RgbdDataset, SlamTbDataset};
use align3d::range_image::RangeImage;
use align3d::Array2Recycle;
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

pub fn sample_teapot_pointcloud() -> PointCloud {
    PointCloud::from_geometry(sample_teapot())
}

pub fn sample_rgbd_pointcloud() -> PointCloud {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();
    let mut frame = dataset.get(0).unwrap();

    frame.image.depth = {
        let filter = BilateralFilter::default();
        filter.filter(&frame.image.depth, Array2Recycle::Empty)
    };

    let mut point_cloud = RangeImage::from_rgbd_frame(&frame);
    point_cloud.compute_normals();
    PointCloud::from(&point_cloud)
}
