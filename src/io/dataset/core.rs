use image::ImageError;

use crate::{trajectory::Trajectory, image::RgbdFrame};
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

pub trait RgbdDataset {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn get(&self, index: usize) -> Result<RgbdFrame, DatasetError>;
    fn trajectory(&self) -> Option<Trajectory>;
}

pub struct SubsetDataset<D: RgbdDataset> {
    dataset: D,
    indices: Vec<usize>,
}

impl<D: RgbdDataset> SubsetDataset<D> {
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        Self { dataset, indices }
    }
}

impl<D: RgbdDataset> RgbdDataset for SubsetDataset<D> {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, index: usize) -> Result<RgbdFrame, DatasetError> {
        self.dataset.get(self.indices[index])
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