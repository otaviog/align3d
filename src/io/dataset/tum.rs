use std::{io::BufRead, path::PathBuf};

use nalgebra::{Quaternion, Vector3};
use nshare::ToNdarray2;

use crate::{
    camera::CameraIntrinsics,
    image::{IntoArray3, RgbdFrame, RgbdImage},
    trajectory::Trajectory,
    transform::Transform,
};

use super::{DatasetError, RgbdDataset};

pub struct TumRgbdDataset {
    base_dir: PathBuf,
    rgb_images: Vec<String>,
    depth_images: Vec<String>,
    trajectory: Trajectory,
}

fn read_file_list(filepath: &PathBuf) -> Result<Vec<(f64, String)>, DatasetError> {
    let file = std::fs::File::open(filepath)?;
    let reader = std::io::BufReader::new(file);
    let file_list: Vec<(f64, String)> = reader
        .lines()
        .filter_map(|line| line.ok())
        .filter(|line| !line.trim().starts_with('#'))
        .map(|line| {
            let tokens: Vec<&str> = line.split(&[',', '\t', ' ']).collect();
            (
                tokens[0].trim().parse::<f64>().unwrap(),
                tokens[1].trim().to_string(),
            )
        })
        .collect();

    Ok(file_list)
}

fn associate<T1: Clone, T2: Clone>(
    first_list: &[(f64, T1)],
    second_list: &[(f64, T2)],
) -> Vec<(f64, T1, f64, T2)> {
    let mut first_list = first_list.iter().peekable();
    let mut second_list = second_list.iter().peekable();
    let mut result = Vec::<(f64, T1, f64, T2)>::new();
    while let (Some((first_time, first_value)), Some((second_time, second_value))) =
        (first_list.peek(), second_list.peek())
    {
        if (first_time - second_time).abs() < 0.02 {
            result.push((
                *first_time,
                first_value.clone(),
                *second_time,
                second_value.clone(),
            ));
            first_list.next();
            second_list.next();
        } else if first_time < second_time {
            first_list.next();
        } else {
            second_list.next();
        }
    }

    result
}

fn load_trajectory(filepath: &str) -> Result<Vec<(f64, Transform)>, DatasetError> {
    let file = std::fs::File::open(filepath)?;
    let reader = std::io::BufReader::new(file);
    let trajectory = reader
        .lines()
        .filter_map(|line| line.ok())
        .filter(|line| !line.trim().starts_with('#'))
        .map(|line| {
            let tokens: Vec<f64> = line
                .split_whitespace()
                .map(|token| token.trim().parse::<f64>().unwrap())
                .collect();
            (
                tokens[0],
                Transform::new(
                    &Vector3::new(tokens[1] as f32, tokens[2] as f32, tokens[3] as f32),
                    &Quaternion::new(
                        tokens[7] as f32,
                        tokens[4] as f32,
                        tokens[5] as f32,
                        tokens[6] as f32,
                    ),
                ),
            )
        })
        .collect::<Vec<(f64, Transform)>>();

    Ok(trajectory)
}

impl TumRgbdDataset {
    pub fn load(base_dirpath: &str) -> Result<Self, DatasetError> {
        let rgb_files = read_file_list(&PathBuf::from(base_dirpath).join("rgb.txt"))?;
        let depth_files = read_file_list(&PathBuf::from(base_dirpath).join("depth.txt"))?;
        let depth_rgb_assoc = associate(&depth_files, &rgb_files);
        let rgb_images = depth_rgb_assoc
            .iter()
            .map(|entry| entry.3.clone())
            .collect::<Vec<String>>();
        let depth_images = depth_rgb_assoc
            .iter()
            .map(|entry| entry.1.clone())
            .collect::<Vec<String>>();

        let trajectory = load_trajectory(
            PathBuf::from(base_dirpath)
                .join("groundtruth.txt")
                .to_str()
                .unwrap(),
        )?;

        let depth_traj_assoc = associate(&depth_files, &trajectory);

        let trajectory = depth_traj_assoc
            .iter()
            .map(|entry| (entry.3.clone(), entry.2 as f32))
            .collect::<Trajectory>();

        Ok(TumRgbdDataset {
            base_dir: PathBuf::from(base_dirpath),
            rgb_images,
            depth_images,
            trajectory,
        })
    }
}

impl RgbdDataset for TumRgbdDataset {
    fn get(&self, index: usize) -> Result<RgbdFrame, DatasetError> {
        let rgb_image = image::open(self.base_dir.join(&self.rgb_images[index]))?
            .into_rgb8()
            .into_array3();

        let depth_image = image::open(self.base_dir.join(&self.depth_images[index]))?
            .into_luma16()
            .into_ndarray2();
        let mut rgbd_image = RgbdImage::new(rgb_image, depth_image);
        rgbd_image.depth_scale = Some(1.0 / 5000.0);
        let camera = CameraIntrinsics {
            fx: 525.0,
            fy: 525.0,
            cx: 319.5,
            cy: 239.5,
            width: Some(640),
            height: Some(480)
        };

        Ok(RgbdFrame::new(camera, rgbd_image, Some(self.trajectory[index].clone())))
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

#[cfg(test)]
mod test {
    use super::*;

    #[ignore]
    #[test]
    fn test_load() {
        // Ignored: requires the TUM RGB-D dataset to be downloaded.
        let dataset = TumRgbdDataset::load("tests/data/rgbd_dataset_freiburg1_xyz").expect("
        Please, link the folder data/rgbd_dataset_freiburg1_xyz to the corresponding in the TUM RGB-D dataset folder");
        assert_eq!(dataset.len(), 797);
        let _item = dataset.get(0).unwrap();
    }
}
