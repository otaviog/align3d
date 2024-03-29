use std::path::{Path, PathBuf};

use super::core::{DatasetError, RgbdDataset};
use crate::{
    camera::CameraIntrinsics,
    image::{IntoArray3, RgbdFrame, RgbdImage},
    trajectory::Trajectory,
    transform::Transform,
};

use nshare::ToNdarray2;

pub struct SlamTbDataset {
    cameras: Vec<CameraIntrinsics>,
    extrinsic_cameras: Vec<Transform>,
    rgb_images: Vec<String>,
    depth_images: Vec<String>,
    depth_scales: Vec<f64>,
    base_dir: PathBuf,
}

mod json {
    use serde_derive::Deserialize;
    #[derive(Deserialize, Debug)]
    pub struct KCam {
        pub matrix: Vec<Vec<f64>>,
        #[serde(rename = "undist_coeff")]
        pub _undist_coeff: Vec<f32>,
        pub image_size: (i32, i32),
    }

    #[derive(Deserialize, Debug)]
    pub struct RTCam {
        pub matrix: Vec<Vec<f32>>,
    }

    #[derive(Deserialize, Debug)]
    pub struct Info {
        pub kcam: KCam,
        pub depth_scale: f64,
        pub depth_bias: f64,
        pub depth_max: f64,
        pub rt_cam: RTCam,
        pub timestamp: f64,
    }

    #[derive(Deserialize, Debug)]
    pub struct Frame {
        pub info: Info,
        pub depth_image: String,
        pub rgb_image: String,
    }

    #[derive(Deserialize, Debug)]
    pub struct Document {
        pub root: Vec<Frame>,
    }
}

impl SlamTbDataset {
    pub fn load(base_dir: &str) -> Result<Self, DatasetError> {
        let buffer = std::io::BufReader::new(std::fs::File::open(
            Path::new(base_dir).join("frames.json"),
        )?);
        serde_json::from_reader(buffer)
            .map(|doc: json::Document| {
                let mut cameras = Vec::new();
                let mut extrinsic_cameras = Vec::new();
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

                    cameras.push(CameraIntrinsics::from_simple_intrinsic(
                        fx,
                        fy,
                        cx,
                        cy,
                        info.kcam.image_size.0 as usize,
                        info.kcam.image_size.1 as usize,
                    ));
                    extrinsic_cameras.push(extrinsics);
                    rgb_images.push(frame.rgb_image.clone());
                    depth_images.push(frame.depth_image.clone());
                    depth_scales.push(info.depth_scale);
                }
                Self {
                    cameras,
                    extrinsic_cameras,
                    rgb_images,
                    depth_images,
                    depth_scales,
                    base_dir: PathBuf::from(base_dir),
                }
            })
            .map_err(|err| DatasetError::Parser(err.to_string()))
    }
}

impl RgbdDataset for SlamTbDataset {
    fn len(&self) -> usize {
        self.rgb_images.len().min(self.depth_images.len())
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, index: usize) -> Result<RgbdFrame, DatasetError> {
        let rgb_image = image::open(self.base_dir.join(&self.rgb_images[index]))?
            .into_rgb8()
            .into_array3();

        let rgb_image = rgb_image.as_standard_layout().into_owned();
        let depth_image = image::open(self.base_dir.join(&self.depth_images[index]))?
            .into_luma16()
            .into_ndarray2();
        Ok(RgbdFrame::new(
            self.cameras[index].clone(),
            RgbdImage::with_depth_scale(rgb_image, depth_image, self.depth_scales[index]),
            Some(self.extrinsic_cameras[index].clone()),
        ))
    }

    fn trajectory(&self) -> Option<Trajectory> {
        let mut trajectory = Trajectory::default();
        for (i, cam) in self.extrinsic_cameras.iter().enumerate() {
            trajectory.push(cam.clone(), i as f32);
        }
        Some(trajectory)
    }

    fn camera(&self, index: usize) -> (CameraIntrinsics, Option<Transform>) {
        (
            self.cameras[index].clone(),
            Some(self.extrinsic_cameras[index].clone()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load() {
        let rgbd_dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();

        let (camera, image, _) = rgbd_dataset.get(0).unwrap().into_parts();

        assert_eq!(camera.fx, 544.4732666015625);
        assert_eq!(camera.fy, 544.4732666015625);
        assert_eq!(camera.cx, 320.0);
        assert_eq!(camera.cy, 240.0);

        assert_eq!(image.height(), 480);
        assert_eq!(image.width(), 640);
    }
}
