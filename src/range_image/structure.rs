use crate::camera::CameraIntrinsics;

use crate::image::{py_scale_down, rgb_to_luma_u8, IntoImageRgb8, RgbdFrame, RgbdImage};
use crate::intensity_map::IntensityMap;

use nalgebra::Vector3;

use ndarray::{ArcArray2, Array1, Array2, Array3, Axis};

use crate::io::Geometry;
use crate::pointcloud::PointCloud;

use super::resize::{resize_range_normals, resize_range_points};

/// A point cloud that comes from an image-based measurement. It representation holds its grid structure.
pub struct RangeImage {
    /// 3D points in the camera frame, as array with shape: (height, width, 3)
    pub points: Array3<f32>,
    /// Mask of valid points, as array with shape: (height, width)
    pub mask: Array2<u8>,
    /// Normals of the points, as array with shape: (height, width, 3)
    pub normals: Option<Array3<f32>>,
    /// Colors of the points, as array with shape: (height, width, 3)
    pub colors: Option<Array3<u8>>,
    /// Camera parameters that originated the image.
    pub camera: CameraIntrinsics,
    /// Intensities of the points, as array with shape: (height*width)
    pub intensities: Option<Array1<u8>>,
    /// Intensity map of the points, as array with shape: (height, width)
    pub intensity_map: Option<IntensityMap>,
    valid_points: usize,
}

impl RangeImage {
    /// Creates a new range image from a depth image and camera parameters.
    ///
    /// # Arguments
    ///
    /// * `camera` - Camera parameters.
    /// * rgbd_image - Rgbd image.
    pub fn from_rgbd_image(camera: &CameraIntrinsics, rgbd_image: &RgbdImage) -> Self {
        let (width, height) = (rgbd_image.width(), rgbd_image.height());
        let depth_scale = rgbd_image.depth_scale.unwrap() as f32;
        let mut points = Array3::zeros((height, width, 3));
        let mut mask = Array2::<u8>::zeros((height, width));
        let mut colors = Array3::<u8>::zeros((height, width, 3));
        let mut valid_points = 0;

        for x in 0..width {
            for y in 0..height {
                let z = rgbd_image.depth[[y, x]];
                if z > 0 {
                    let z = rgbd_image.depth[[y, x]] as f32 * depth_scale;
                    let point3d = camera.backproject(x as f32, y as f32, z);
                    points[[y, x, 0]] = point3d[0];
                    points[[y, x, 1]] = point3d[1];
                    points[[y, x, 2]] = point3d[2];
                    mask[[y, x]] = 1;
                    valid_points += 1;
                }

                //colors.slice_mut(s![y, x, ..]).assign(&rgbd_image.color.slice(s![y, x, ..]));
                colors[[y, x, 0]] = rgbd_image.color[[y, x, 0]];
                colors[[y, x, 1]] = rgbd_image.color[[y, x, 1]];
                colors[[y, x, 2]] = rgbd_image.color[[y, x, 2]];
            }
        }

        Self {
            points,
            mask,
            normals: None,
            colors: Some(colors),
            camera: camera.clone(),
            intensities: None,
            intensity_map: None,
            valid_points,
        }
    }

    /// Creates a new range image from a depth image and camera parameters.
    pub fn from_rgbd_frame(frame: &RgbdFrame) -> Self {
        Self::from_rgbd_image(&frame.camera, &frame.image)
    }

    /// Width of the image.
    pub fn width(&self) -> usize {
        self.points.shape()[1]
    }

    /// Height of the image.
    pub fn height(&self) -> usize {
        self.points.shape()[0]
    }

    /// Number of valid points in the image, this is not the same as
    /// width*height, since some points may be invalid (i.e, 0 valued depth).
    pub fn valid_points_count(&self) -> usize {
        self.valid_points
    }

    /// Returns the number of points in the image, which is width*height.
    pub fn len(&self) -> usize {
        let shape = self.points.shape();
        shape[0] * shape[1]
    }

    /// Returns true if the image is empty, i.e, it has no points.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the 3D point at the given row and column.
    /// If the point is invalid, returns None.
    ///
    /// # Arguments
    ///
    /// * `row` - Row of the point.
    /// * `col` - Column of the point.
    ///
    /// # Returns
    ///
    /// * `Option<Vector3<f32>>` - 3D point at `row` and `col`.
    pub fn get_point(&self, row: usize, col: usize) -> Option<nalgebra::Vector3<f32>> {
        if col < self.width() && row < self.height() && self.mask[(row, col)] == 1 {
            Some(Vector3::new(
                self.points[(row, col, 0)],
                self.points[(row, col, 1)],
                self.points[(row, col, 2)],
            ))
        } else {
            None
        }
    }

    /// Updates the image with normals computed from the 3D points.
    pub fn compute_normals(&mut self) -> &mut Self {
        let height = self.height();
        let width = self.width();

        let ratio_threshold = 2f32;
        let ratio_threshold_squared = ratio_threshold * ratio_threshold;

        let mut normals = Array3::<f32>::zeros((height, width, 3));

        for row in 0..height {
            for col in 0..width {
                if self.mask[(row, col)] != 1 {
                    continue;
                };

                let center = nalgebra::Vector3::<f32>::new(
                    self.points[(row, col, 0)],
                    self.points[(row, col, 1)],
                    self.points[(row, col, 2)],
                );
                let left = self
                    .get_point(row, (col as i32 - 1) as usize)
                    .unwrap_or_else(nalgebra::Vector3::<f32>::zeros);
                let right = self
                    .get_point(row, col + 1)
                    .unwrap_or_else(nalgebra::Vector3::<f32>::zeros);

                let left_dist_squared = (left - center).norm_squared();
                let right_dist_squared = (right - center).norm_squared();
                let left_right_ratio = left_dist_squared / right_dist_squared;

                let left_to_right = if left_right_ratio < ratio_threshold_squared
                    && left_right_ratio > 1f32 / ratio_threshold_squared
                {
                    right - left
                } else if left_dist_squared < right_dist_squared {
                    center - left
                } else {
                    right - center
                };

                let bottom = self
                    .get_point(row + 1, col)
                    .unwrap_or_else(nalgebra::Vector3::<f32>::zeros);
                let top = self
                    .get_point((row as i32 - 1) as usize, col)
                    .unwrap_or_else(nalgebra::Vector3::<f32>::zeros);

                let bottom_dist_squared = (bottom - center).norm_squared();
                let top_dist_squared = (top - center).norm_squared();
                let bottom_top_ratio = bottom_dist_squared / top_dist_squared;

                let bottom_to_top = if bottom_top_ratio < ratio_threshold_squared
                    && bottom_top_ratio > 1f32 / ratio_threshold_squared
                {
                    top - bottom
                } else if bottom_dist_squared < top_dist_squared {
                    center - bottom
                } else {
                    top - center
                };

                let normal = left_to_right.cross(&bottom_to_top).normalize();

                let normal_magnitude = normal.magnitude();
                if normal_magnitude > 1e-6_f32 {
                    normals[(row, col, 0)] = normal[0] / normal_magnitude;
                    normals[(row, col, 1)] = normal[1] / normal_magnitude;
                    normals[(row, col, 2)] = normal[2] / normal_magnitude;
                }
            }
        }

        self.normals = Some(normals);

        self
    }

    /// By default, range image have only the RGB colors, this method
    /// will convert them into luma values, which are used as color optimization term in ICP.
    pub fn compute_intensity(&mut self) -> &mut Self {
        let color = self
            .colors
            .as_ref()
            .unwrap()
            .view()
            .into_shape((self.len(), 3))
            .unwrap();
        self.intensities = Some(
            color
                .axis_iter(Axis(0))
                .map(|rgb| rgb_to_luma_u8(rgb[0], rgb[1], rgb[2]))
                .collect(),
        );

        self
    }

    /// Generates the intensity map from the intensity array. This method is called by RangeImage
    /// that are targets in Image ICP.
    pub fn compute_intensity_map(&mut self) -> &mut Self {
        if self.intensities.is_none() {
            self.compute_intensity();
        }

        self.intensity_map = Some(IntensityMap::from_luma_image(
            &self
                .intensities
                .as_ref()
                .unwrap()
                .view()
                .into_shape((self.height(), self.width()))
                .unwrap(),
        ));

        self
    }

    /// Downscales the range image by a factor of 2 using a Gaussian pyramid for rgb image.
    /// And a simple averaging for the 3D points and normals.
    ///
    /// # Arguments
    ///
    /// * `sigma` - The start sigma value for the Gaussian pyramid.
    ///
    /// # Returns
    ///
    /// A new range image with the downscaled points, normals, and colors by a factor of 2.
    pub fn pyr_scale_down(&self, sigma: f32) -> RangeImage {
        let (width, height) = (self.width() / 2, self.height() / 2);
        let (points, mask) =
            resize_range_points(&self.points.view(), &self.mask.view(), width, height);

        let normals = self
            .normals
            .as_ref()
            .map(|normals| resize_range_normals(&normals.view(), &self.mask.view(), width, height));

        // TODO: Figure out how to not clone the colors, keep all immutable
        // and still convert into_image_rgb8
        let colors = if let Some(colors) = self.colors.clone() {
            let colors = colors.into_image_rgb8();
            let down_colors = py_scale_down(&colors, sigma);
            Some(down_colors)
        } else {
            None
        };
        let valid_points = mask.iter().map(|x| (*x == 1) as usize).sum();
        RangeImage {
            points,
            mask,
            normals,
            colors,
            camera: self.camera.scale(0.5),
            intensities: None,
            intensity_map: None,
            valid_points,
        }
    }

    pub fn pyramid(self, levels: usize, sigma: f32) -> Vec<RangeImage> {
        let mut pyramid = vec![self];

        for _ in 0..levels - 1 {
            let prev = pyramid.last().unwrap();
            pyramid.push(prev.pyr_scale_down(sigma));
        }

        pyramid
    }

    pub fn is_valid(&self, u: usize, v:usize) -> bool {
        self.mask[(v, u)] != 0
    }
}

impl From<&RangeImage> for PointCloud {
    fn from(image_pcl: &RangeImage) -> PointCloud {
        let num_total_points = image_pcl.len();

        let mask = image_pcl
            .mask
            .view()
            .into_shape((num_total_points,))
            .unwrap();
        let num_valid_points = mask.iter().map(|x| *x as usize).sum();

        // TODO: Improve mask and make a generic function/macro.
        let v: Vec<f32> = image_pcl
            .points
            .view()
            .into_shape((num_total_points, 3))
            .unwrap()
            .axis_iter(Axis(0))
            .enumerate()
            .filter(|(idx, _)| mask[*idx] != 0)
            .flat_map(|(_, v)| [v[0], v[1], v[2]])
            .collect();
        let points = Array2::from_shape_vec((num_valid_points, 3), v).unwrap();

        let normals = image_pcl.normals.as_ref().map(|normals| {
            Array2::from_shape_vec(
                (num_valid_points, 3),
                normals
                    .view()
                    .into_shape((num_total_points, 3))
                    .unwrap()
                    .axis_iter(Axis(0))
                    .enumerate()
                    .filter(|(idx, _)| mask[*idx] != 0)
                    .flat_map(|(_, v)| [v[0], v[1], v[2]])
                    .collect(),
            )
            .unwrap()
        });

        let colors = image_pcl.colors.as_ref().map(|colors| {
            ArcArray2::from_shape_vec(
                (num_valid_points, 3),
                colors
                    .view()
                    .into_shape((num_total_points, 3))
                    .unwrap()
                    .axis_iter(Axis(0))
                    .enumerate()
                    .filter(|(idx, _)| mask[*idx] != 0)
                    .flat_map(|(_, v)| [v[0], v[1], v[2]])
                    .collect(),
            )
            .unwrap()
        });

        PointCloud {
            points,
            normals,
            colors,
        }
    }
}

impl From<&RangeImage> for Geometry {
    fn from(image_pcl: &RangeImage) -> Geometry {
        let pcl = PointCloud::from(image_pcl);
        pcl.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        image::IntoLumaImage,
        io::{
            dataset::{RgbdDataset, SlamTbDataset},
            write_ply,
        },
    };
    use nshare::ToNdarray2;
    use rstest::*;

    #[fixture]
    fn sample1() -> SlamTbDataset {
        SlamTbDataset::load("tests/data/rgbd/sample1").unwrap()
    }

    #[rstest]
    fn should_backproject_rgbd_image(sample1: SlamTbDataset) {
        use crate::io::write_ply;

        let (cam, rgbd_image, _) = sample1.get(0).unwrap().into_parts();
        let im_pcl = RangeImage::from_rgbd_image(&cam, &rgbd_image);

        assert_eq!(480, im_pcl.height());
        assert_eq!(640, im_pcl.width());

        write_ply(
            "tests/outputs/out-back-projection.ply",
            &Geometry::from(&im_pcl),
        )
        .expect("Error while writing results");
    }

    #[rstest]
    fn should_compute_normals(sample1: SlamTbDataset) {
        let (cam, rgbd_image, _) = sample1.get(0).unwrap().into_parts();

        let mut im_pcl = RangeImage::from_rgbd_image(&cam, &rgbd_image);
        im_pcl.compute_normals();
        write_ply(
            "tests/outputs/out-range-image-normals.ply",
            &Geometry::from(&im_pcl),
        )
        .expect("Error while writing the results");

        {
            let normals = im_pcl.normals.as_ref().unwrap();
            assert_eq!(480, normals.shape()[0]);
            assert_eq!(640, normals.shape()[1]);

            let v = Vector3::new(
                normals[[44, 42, 0]],
                normals[[44, 42, 1]],
                normals[[44, 42, 2]],
            );
            assert_eq!(v.norm(), 1.0);
        }
    }

    #[rstest]
    fn should_convert_into_pointcloud(sample1: SlamTbDataset) {
        let (cam, rgbd_image, _) = sample1.get(0).unwrap().into_parts();
        let im_pcl = RangeImage::from_rgbd_image(&cam, &rgbd_image);

        let pcl = PointCloud::from(&im_pcl);
        assert_eq!(pcl.len(), 270213);
    }

    #[rstest]
    fn verify_pyramid(sample1: SlamTbDataset) {
        let mut pyramid = RangeImage::from_rgbd_frame(&sample1.get(0).unwrap()).pyramid(3, 1.0);

        for im in pyramid.iter_mut() {
            im.compute_normals();
            im.compute_intensity();
            im.compute_intensity_map();
        }

        for (i, im) in pyramid.iter().enumerate() {
            write_ply(
                format!("tests/outputs/out-range-image-{i}.ply"),
                &Geometry::from(im),
            )
            .expect("Error while writing the results");
        }

        for (i, im) in pyramid.iter_mut().enumerate() {
            let imap = im.intensity_map.as_ref().unwrap().clone().into_ndarray2();
            imap.to_luma_image()
                .save(format!("tests/outputs/out-range-image-{i}.png"))
                .expect("Error while writing the results");
        }
        assert_eq!(pyramid.len(), 3);
    }
}
