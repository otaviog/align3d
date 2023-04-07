use std::{io::BufRead, path::Path};

use glob::PatternError;
use itertools::Itertools;
use nalgebra::Matrix4;
use nshare::ToNdarray2;

use crate::{
    camera::Camera,
    image::{IntoArray3, RgbdFrame, RgbdImage},
    trajectory::Trajectory,
    transform::Transform,
};

use super::core::{DatasetError, RgbdDataset};

/// Parser for the IndoorLidar dataset. Available at:
///  http://redwood-data.org/indoor_lidar_rgbd/index.html.
///	 Jaesik Park and Qian-Yi Zhou and Vladlen Koltun,
///	 Colored Point Cloud Registration Revisited. ICCV, 2017.
pub struct IndoorLidarDataset {
    rgb_images: Vec<String>,
    depth_images: Vec<String>,
    trajectory: Trajectory,
}

impl From<PatternError> for DatasetError {
    fn from(err: glob::PatternError) -> Self {
        DatasetError::Parser(err.to_string())
    }
}

impl IndoorLidarDataset {
    pub fn load(base_dir: &str) -> Result<Self, DatasetError> {
        let rgb_images: Vec<String> = glob::glob(&format!("{base_dir}/image/*.jpg"))?
            .map(|x| x.unwrap().to_str().unwrap().to_string())
            .collect();
        let depth_images: Vec<String> = glob::glob(&format!("{base_dir}/depth/*.png"))?
            .map(|x| x.unwrap().to_str().unwrap().to_string())
            .collect();

        if rgb_images.len() != depth_images.len() {
            return Err(DatasetError::Parser(
                "Number of RGB and depth images do not match".to_string(),
            ));
        }

        let log_filename = Path::new(base_dir).file_stem().unwrap().to_str().unwrap();
        let file = std::fs::File::open(format!("{base_dir}/{log_filename}.log"))?;
        let reader = std::io::BufReader::new(file);
        let trajectory = reader
            .lines()
            .map_ok(|line| line.trim().to_string())
            .filter(|line| !line.as_ref().unwrap().is_empty())
            .map(|line| line.unwrap())
            .chunks(5)
            .into_iter()
            .enumerate()
            .map(|(n, lines)| {
                let mut matrix = Matrix4::zeros();
                let lines = lines.skip(1);
                for (i, line) in lines.enumerate() {
                    let iter = line.split_whitespace();
                    for (j, token) in iter.enumerate() {
                        matrix[(i, j)] = token.parse::<f32>().unwrap();
                    }
                }

                (Transform::from_matrix4(&matrix), n as f32)
            })
            .collect::<Trajectory>();
        Ok(IndoorLidarDataset {
            rgb_images,
            depth_images,
            trajectory,
        })
    }
}

impl RgbdDataset for IndoorLidarDataset {
    fn len(&self) -> usize {
        self.rgb_images.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, idx: usize) -> Result<RgbdFrame, DatasetError> {
        let rgb_image = image::open(&self.rgb_images[idx])?
            .into_rgb8()
            .into_array3();

        let depth_image = image::open(&self.depth_images[idx])?
            .into_luma16()
            .into_ndarray2();
        let rgbd_image = RgbdImage::with_depth_scale(rgb_image, depth_image, 0.001);
        let camera = Camera {
            fx: 525.0,
            fy: 525.0,
            cx: 319.5,
            cy: 239.5,
            camera_to_world: Some(self.trajectory[idx].clone()),
        };

        Ok(RgbdFrame::new(camera, rgbd_image))
    }

    fn trajectory(&self) -> Option<Trajectory> {
        Some(self.trajectory.clone())
    }
}
