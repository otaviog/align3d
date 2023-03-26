use ndarray::{Array2, Array3, Axis};
use nshare::ToNdarray2;
use rstest::*;

use crate::bilateral::BilateralFilter;

use crate::image::{IntoArray3, RgbdFrame};
use crate::{
    io::{core::RgbdDataset, read_off, slamtb_dataset::SlamTbDataset, Geometry},
    mesh::compute_normals,
    pointcloud::PointCloud,
    range_image::RangeImage,
    Array2Recycle,
};

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
        .into_array3()
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
pub fn sample_rgbd_dataset1() -> impl RgbdDataset {
    SlamTbDataset::load("tests/data/rgbd/sample1").unwrap()
}

pub struct TestRgbdFrameDataset {
    dataset: Box<dyn RgbdDataset>,
}

impl TestRgbdFrameDataset {
    pub fn get_item(&self, index: usize) -> Result<RgbdFrame, crate::io::core::DatasetError> {
        let (cam, mut rgbd_image) = self.dataset.get_item(index)?.into_parts();
        rgbd_image.depth = {
            let filter = BilateralFilter::default();
            filter.filter(&rgbd_image.depth, Array2Recycle::Empty)
        };
        Ok(RgbdFrame::new(cam, rgbd_image))
    }

    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}

#[fixture]
pub fn sample_rgbd_frame_dataset1() -> TestRgbdFrameDataset {
    TestRgbdFrameDataset {
        dataset: Box::new(SlamTbDataset::load("tests/data/rgbd/sample1").unwrap()),
    }
}

pub struct TestRangeImageDataset {
    dataset: Box<dyn RgbdDataset>,
}

impl TestRangeImageDataset {
    pub fn get_item(&self, index: usize) -> Result<RangeImage, crate::io::core::DatasetError> {
        let (cam, mut rgbd_image) = self.dataset.get_item(index)?.into_parts();
        rgbd_image.depth = {
            let filter = BilateralFilter::default();
            filter.filter(&rgbd_image.depth, Array2Recycle::Empty)
        };
        let mut range_img = RangeImage::from_rgbd_image(&cam, &rgbd_image);
        range_img.compute_normals();
        range_img.compute_intensity();
        Ok(range_img)
    }

    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}

#[fixture]
pub fn sample_range_img_ds1() -> TestRangeImageDataset {
    TestRangeImageDataset {
        dataset: Box::new(SlamTbDataset::load("tests/data/rgbd/sample1").unwrap()),
    }
}

#[fixture]
pub fn sample_range_img_ds2() -> TestRangeImageDataset {
    TestRangeImageDataset {
        dataset: Box::new(SlamTbDataset::load("tests/data/rgbd/sample2").unwrap()),
    }
}
