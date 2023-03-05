use image::ImageError;

use super::rgbdimage::RGBDImage;
use crate::{camera::Camera, trajectory::Trajectory};
use std::io::Error;

#[derive(Debug)]
pub enum DatasetError {
    Io(Error),
    Parser(String),
    Image(ImageError),
}

impl From<Error> for DatasetError {
    fn from(err: Error) -> Self {
        DatasetError::Io(err)
    }
}

impl From<ImageError> for DatasetError {
    fn from(err: ImageError) -> Self {
        DatasetError::Image(err)
    }
}

pub struct RGBDDatasetItem {
    pub camera: Camera,
    pub rgbd_image: RGBDImage,
    pub timestamp: f64,
}

pub trait RGBDDataset {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn get_item(&self, index: usize) -> Result<(Camera, RGBDImage), DatasetError>;
    fn trajectory(&self) -> Option<Trajectory>;
}

pub struct SubsetDataset<D: RGBDDataset> {
    dataset: D,
    indices: Vec<usize>,
}

impl<D: RGBDDataset> SubsetDataset<D> {
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        Self { dataset, indices }
    }
}

impl<D: RGBDDataset> RGBDDataset for SubsetDataset<D> {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get_item(&self, index: usize) -> Result<(Camera, RGBDImage), DatasetError> {
        self.dataset.get_item(self.indices[index])
    }

    fn trajectory(&self) -> Option<Trajectory> {
        let orig_trajectory = self.dataset.trajectory()?;
        let mut trajectory = Trajectory::new();
        for (i, index) in self.indices.iter().enumerate() {
            trajectory.push(orig_trajectory.camera_to_world[*index].clone(), i as f32);
        }
        Some(trajectory)
    }
}
