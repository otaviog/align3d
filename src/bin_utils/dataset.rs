use crate::{
    error::A3dError,
    io::dataset::{IndoorLidarDataset, RgbdDataset, SlamTbDataset, TumRgbdDataset},
};

pub fn load_dataset(format: String, path: String) -> Result<Box<dyn RgbdDataset + Send>, A3dError> {
    match format.as_str() {
        "slamtb" => Ok(Box::new(SlamTbDataset::load(&path).unwrap())),
        "ilrgbd" => Ok(Box::new(IndoorLidarDataset::load(&path).unwrap())),
        "tum" => Ok(Box::new(TumRgbdDataset::load(&path).unwrap())),
        _ => Err(A3dError::invalid_parameter(format!(
            "Invalid dataset format: {format}"
        ))),
    }
}
