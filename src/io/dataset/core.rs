use image::ImageError;

use crate::{image::RgbdFrame, trajectory::Trajectory};
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

impl std::error::Error for DatasetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DatasetError::Io(err) => Some(err),
            DatasetError::Parser(_) => None,
            DatasetError::Image(err) => Some(err),
        }
    }
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            DatasetError::Io(err) => write!(f, "IO error: {err}"),
            DatasetError::Parser(err) => write!(f, "Parser error: {err}"),
            DatasetError::Image(err) => write!(f, "Image error: {err}"),
        }
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

pub struct SubsetDataset {
    dataset: Box<dyn RgbdDataset>,
    indices: Vec<usize>,
}

impl SubsetDataset {
    pub fn new(dataset: Box<dyn RgbdDataset>, indices: Vec<usize>) -> Self {
        Self { dataset, indices }
    }
}

impl RgbdDataset for SubsetDataset {
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
        let mut trajectory = Trajectory::default();
        for (i, index) in self.indices.iter().enumerate() {
            trajectory.push(orig_trajectory.camera_to_world[*index].clone(), i as f32);
        }
        Some(trajectory)
    }
}
