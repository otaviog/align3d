use ndarray::{Array2, Array3};

pub struct RGBDImage {
    pub color: Array3<u8>,
    pub depth: Array2<u16>,
    pub depth_scale: Option<f64>,
}

impl RGBDImage {
    pub fn new(color: Array3<u8>, depth: Array2<u16>) -> Self {
        Self {
            color,
            depth,
            depth_scale: None,
        }
    }

    pub fn with_depth_scale(color: Array3<u8>, depth: Array2<u16>, depth_scale: f64) -> Self {
        Self {
            color,
            depth,
            depth_scale: Some(depth_scale),
        }
    }

    pub fn width(&self) -> usize {
        self.color.shape()[2]
    }

    pub fn height(&self) -> usize {
        self.color.shape()[1]
    }
}
