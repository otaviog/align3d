use crate::{
    error::Error,
    io::dataset::{IndoorLidarDataset, RgbdDataset, SlamTbDataset, TumRgbdDataset},
};

pub fn create_dataset_from_string(format: String, path: String) -> Result<Box<dyn RgbdDataset>, Error> {
    match format.as_str() {
        "slamtb" => Ok(Box::new(SlamTbDataset::load(&path).unwrap())),
        "ilrgbd" => Ok(Box::new(IndoorLidarDataset::load(&path).unwrap())),
        "tum" => Ok(Box::new(TumRgbdDataset::load(&path).unwrap())),
        _ => Err(Error::invalid_parameter(format!(
            "Invalid dataset format: {format}"
        ))),
    }
}
