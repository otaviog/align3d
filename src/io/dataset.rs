use image::ImageError;

use super::rgbdimage::RGBDImage;
use crate::camera::Camera;
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
    fn get_item(&self, index: usize) -> Result<(Camera, RGBDImage), DatasetError>;
}
