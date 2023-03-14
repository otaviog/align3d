use ndarray::{Array2, Axis, Array3};
use nshare::{ToNdarray2, ToNdarray3};
use rstest::*;

use crate::bilateral::BilateralFilter;
use crate::camera::Camera;
use crate::imagepointcloud::ImagePointCloud;
use crate::io::core::RGBDDataset;
use crate::io::rgbdimage::RGBDFrame;
use crate::io::slamtb::SlamTbDataset;
use crate::io::{read_off, Geometry};
use crate::mesh::compute_normals;
use crate::pointcloud::PointCloud;
use crate::Array2Recycle;

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
pub fn bloei_rgb() -> Array3<u8> {
    image::io::Reader::open("tests/data/images/bloei.jpg")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8()
        .into_ndarray3()
}

#[fixture]
pub fn bloei_luma8() -> Array2<u8> {
    image::io::Reader::open("tests/data/images/bloei.jpg")
        .unwrap()
        .decode()
        .unwrap()
        .into_luma8()
        .into_ndarray2()
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

#[fixture]
pub fn sample_rgbd_dataset1() -> impl RGBDDataset {
    SlamTbDataset::load("tests/data/rgbd/sample1").unwrap()
}

pub struct TestRGBDFrameDataset {
    dataset: Box<dyn RGBDDataset>,
}

impl TestRGBDFrameDataset {
    pub fn get_item(
        &self,
        index: usize,
    ) -> Result<RGBDFrame, crate::io::core::DatasetError> {
        let (cam, mut rgbd_image) = self.dataset.get_item(index)?.into_parts();
        rgbd_image.depth = {
            let filter = BilateralFilter::default();
            filter.filter(&rgbd_image.depth, Array2Recycle::Empty)
        };
        Ok(RGBDFrame::new(cam, rgbd_image))
    }

    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}

#[fixture]
pub fn sample_rgbd_frame_dataset1() -> TestRGBDFrameDataset {
    TestRGBDFrameDataset {
        dataset: Box::new(SlamTbDataset::load("tests/data/rgbd/sample1").unwrap()),
    }
}

pub struct TestImagePointCloudDataset {
    dataset: Box<dyn RGBDDataset>,
}

impl TestImagePointCloudDataset {
    pub fn get_item(
        &self,
        index: usize,
    ) -> Result<(Camera, ImagePointCloud), crate::io::core::DatasetError> {
        let (cam, mut rgbd_image) = self.dataset.get_item(index)?.into_parts();
        rgbd_image.depth = {
            let filter = BilateralFilter::default();
            filter.filter(&rgbd_image.depth, Array2Recycle::Empty)
        };
        let mut pcl = ImagePointCloud::from_rgbd_image(&cam, &rgbd_image);
        pcl.compute_normals();
        pcl.compute_intensity();
        Ok((cam, pcl))
    }

    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}

#[fixture]
pub fn sample_imrgbd_dataset1() -> TestImagePointCloudDataset {
    TestImagePointCloudDataset {
        dataset: Box::new(SlamTbDataset::load("tests/data/rgbd/sample1").unwrap()),
    }
}
