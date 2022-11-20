use serde_derive::Deserialize;
use std::path::{Path, PathBuf};

use super::{
    dataset::{DatasetError, RGBDDataset},
    rgbdimage::RGBDImage,
};
use crate::{
    camera::{Camera, CameraBuilder},
    transform::Transform,
};

use nshare::{ToNdarray2, ToNdarray3};

pub struct SlamTbDataset {
    cameras: Vec<Camera>,
    rgb_images: Vec<String>,
    depth_images: Vec<String>,
    depth_scales: Vec<f64>,
    base_dir: PathBuf,
}

#[derive(Deserialize, Debug)]
struct KCam {
    matrix: Vec<Vec<f64>>,
    undist_coeff: Vec<f32>,
    image_size: (i32, i32),
}

#[derive(Deserialize, Debug)]
struct RTCam {
    matrix: Vec<Vec<f32>>,
}

#[derive(Deserialize, Debug)]
struct Info {
    kcam: KCam,
    depth_scale: f64,
    depth_bias: f64,
    depth_max: f64,
    rt_cam: RTCam,
    timestamp: usize,
}

#[derive(Deserialize, Debug)]
struct Frame {
    info: Info,
    depth_image: String,
    rgb_image: String,
}

#[derive(Deserialize, Debug)]
struct Document {
    root: Vec<Frame>,
}

impl SlamTbDataset {
    pub fn load(base_dir: &str) -> Result<Self, DatasetError> {
        let buffer = std::io::BufReader::new(std::fs::File::open(
            Path::new(base_dir).join("frames.json"),
        )?);
        serde_json::from_reader(buffer)
            .map(|doc: Document| {
                let mut cameras = Vec::new();
                let mut rgb_images = Vec::new();
                let mut depth_images = Vec::new();
                let mut depth_scales = Vec::new();

                for frame in doc.root.iter() {
                    let info = &frame.info;

                    let fx = info.kcam.matrix[0][0];
                    let fy = info.kcam.matrix[1][1];
                    let cx = info.kcam.matrix[0][2];
                    let cy = info.kcam.matrix[1][2];

                    let extrinsics = if info.rt_cam.matrix.len() == 4 {
                        Transform::from_matrix4(&nalgebra::Matrix4::<f32>::from_fn(|r, c| {
                            info.rt_cam.matrix[r][c]
                        }))
                    } else {
                        Transform::eye()
                    };

                    cameras.push(
                        CameraBuilder::from_simple_intrinsics(fx, fy, cx, cy)
                            .camera_to_world(Some(extrinsics))
                            .build(),
                    );
                    rgb_images.push(frame.rgb_image.clone());
                    depth_images.push(frame.depth_image.clone());
                    depth_scales.push(info.depth_scale);
                }
                Self {
                    cameras,
                    rgb_images,
                    depth_images,
                    depth_scales,
                    base_dir: PathBuf::from(base_dir),
                }
            })
            .map_err(|err| DatasetError::Parser(err.to_string()))
    }
}

impl RGBDDataset for SlamTbDataset {
    fn len(&self) -> usize {
        self.rgb_images.len().min(self.depth_images.len())
    }

    fn get_item(&self, index: usize) -> Result<(Camera, RGBDImage), DatasetError> {
        let rgb_image = image::open(self.base_dir.join(&self.rgb_images[index]))?
            .into_rgb8()
            .into_ndarray3();
        let depth_image = image::open(self.base_dir.join(&self.depth_images[index]))?
            .into_luma16()
            .into_ndarray2();
        Ok((
            self.cameras[index].clone(),
            RGBDImage::with_depth_scale(rgb_image, depth_image, self.depth_scales[index]),
        ))
    }
}

#[cfg(test)]
mod tests {
    //use rstest::*;

    use super::*;
    use crate::io::dataset::RGBDDataset;

    #[test]
    fn test_load() {
        let rgbd_dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();

        let (camera, image) = rgbd_dataset.get_item(0).unwrap();

        assert_eq!(camera.fx, 544.4732666015625);
        assert_eq!(camera.fy, 544.4732666015625);
        assert_eq!(camera.cx, 320.0);
        assert_eq!(camera.cy, 240.0);

        assert_eq!(image.height(), 480);
        assert_eq!(image.width(), 640);
    }
}
