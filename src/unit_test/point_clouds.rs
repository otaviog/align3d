use ndarray::Array1;
use rstest::fixture;

use crate::{camera::CameraIntrinsics, io::read_off, pointcloud::PointCloud, transform::Transform};

use super::{sample_range_img_ds1, TestRangeImageDataset};

#[fixture]
pub fn sample_teapot_pointcloud() -> PointCloud {
    let mut geometry = read_off("tests/data/teapot.off").unwrap();
    let num_vertices = geometry.len_vertices();
    geometry.normals = Some(Array1::zeros(num_vertices));
    geometry.colors = Some(Array1::zeros(num_vertices));

    PointCloud::from_geometry(geometry)
}

pub struct TestPclDataset {
    dataset: TestRangeImageDataset,
}

impl TestPclDataset {
    pub fn get(&self, index: usize) -> PointCloud {
        let range_image = self
            .dataset
            .get(index)
            .expect("Error when loading range image to point cloud.");
        PointCloud::from(&range_image)
    }

    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    pub fn get_ground_truth(&self, source_index: usize, target_index: usize) -> Transform {
        self.dataset.get_ground_truth(source_index, target_index)
    }

    pub fn camera(&self, index: usize) -> (CameraIntrinsics, Option<Transform>) {
        self.dataset.camera(index)
    }
}

#[fixture]
pub fn sample_pcl_ds1() -> TestPclDataset {
    TestPclDataset {
        dataset: sample_range_img_ds1(),
    }
}
