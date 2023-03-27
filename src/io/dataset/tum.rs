use std::{io::BufRead, path::PathBuf};

use itertools::Itertools;
use nalgebra::Vector3;
use nshare::ToNdarray2;

use crate::{
    camera::Camera,
    image::{IntoArray3, RgbdFrame, RgbdImage},
    trajectory::Trajectory,
    transform::Transform,
};

use super::{DatasetError, RgbdDataset};

pub struct TumRgbdDataset {
    rgb_images: Vec<String>,
    depth_images: Vec<String>,
    trajectory: Trajectory,
}

fn load_trajectory(filepath: &str) -> Result<Trajectory, DatasetError> {
    let file = std::fs::File::open(filepath)?;
    let reader = std::io::BufReader::new(file);
    let trajectory = Ok(reader
        .lines()
        .map_ok(|line| line.trim().to_string())
        .filter(|line| !line.as_ref().unwrap().trim().starts_with("#"))
        .enumerate()
        .map(|(n, line)| {
            let line = line.unwrap();
            let tokens: Vec<f32> = line
                .split_whitespace()
                .map(|token| token.trim().parse::<f32>().unwrap())
                .collect();
            (
                Transform::new(
                    &Vector3::new(tokens[1], tokens[2], tokens[3]),
                    nalgebra::Quaternion::new(tokens[7], tokens[4], tokens[5], tokens[6]),
                ),
                n as f32,
            )
        })
        .collect::<Trajectory>());

    trajectory
}

impl TumRgbdDataset {
    pub fn load(base_dirpath: &str) -> Result<Self, DatasetError> {
        let rgb_images = glob::glob(
            PathBuf::from(base_dirpath)
                .join("rgb")
                .join("*.png")
                .to_str()
                .unwrap(),
        )
        .unwrap()
        .map(|x| x.unwrap().to_str().unwrap().to_string())
        .collect::<Vec<String>>();

        let depth_images = glob::glob(
            PathBuf::from(base_dirpath)
                .join("depth")
                .join("*.png")
                .to_str()
                .unwrap(),
        )
        .unwrap()
        .map(|x| x.unwrap().to_str().unwrap().to_string())
        .collect::<Vec<String>>();

        let trajectory = load_trajectory(
            PathBuf::from(base_dirpath)
                .join("groundtruth.txt")
                .to_str()
                .unwrap(),
        )
        .unwrap();
        Ok(TumRgbdDataset {
            rgb_images,
            depth_images,
            trajectory,
        })
    }
}

impl RgbdDataset for TumRgbdDataset {
    fn get(&self, index: usize) -> Result<RgbdFrame, DatasetError> {
        let rgb_image = image::open(&self.rgb_images[index])?
            .into_rgb8()
            .into_array3();

        let depth_image = image::open(&self.depth_images[index])?
            .into_luma16()
            .into_ndarray2();
        let mut rgbd_image = RgbdImage::new(rgb_image, depth_image);
        rgbd_image.depth_scale = Some(1.0 / 5000.0);
        let camera = Camera {
            fx: 525.0,
            fy: 525.0,
            cx: 319.5,
            cy: 239.5,
            camera_to_world: Some(self.trajectory[index].clone()),
        };

        Ok(RgbdFrame::new(camera, rgbd_image))
    }

    fn len(&self) -> usize {
        self.rgb_images.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn trajectory(&self) -> Option<Trajectory> {
        Some(self.trajectory.clone())
    }
}
