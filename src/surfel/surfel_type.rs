use nalgebra::{Vector2, Vector3};

use crate::{
    camera::CameraIntrinsics,
    range_image::RangeImage,
    transform::{Transform, TransformableMove},
};

/// A surfel is a point in 3D space with a radius, normal, color.
/// We add age and confidence, which are used to merge surfels or remove them during fusion.
#[derive(Debug, Clone, Copy)]
pub struct Surfel {
    /// Position of the surfel.
    pub position: Vector3<f32>,
    /// Normal of the surfel.
    pub normal: Vector3<f32>,
    /// Color of the surfel.
    pub color: Vector3<u8>,
    /// Radius of the surfel.
    pub radius: f32,
    /// Confidence of the surfel.
    pub confidence: f32,
    /// Age of the surfel.
    pub age: u32,
}

impl Surfel {
    /// Merge two surfels into one weighting them by their confidence.
    ///
    /// # Arguments
    ///
    /// * `other` - The other surfel to merge with.
    ///
    /// # Returns
    ///
    /// A new surfel that is the merge of the two.
    pub fn merge(&self, other: &Surfel) -> Self {
        let total_confidence = self.confidence + other.confidence;
        let weight1 = self.confidence / (total_confidence);
        let weight2 = other.confidence / (total_confidence);

        let color = Vector3::<f32>::new(
            self.color[0] as f32 * weight1,
            self.color[1] as f32 * weight1,
            self.color[2] as f32 * weight1,
        ) + Vector3::<f32>::new(
            other.color[0] as f32 * weight2,
            other.color[1] as f32 * weight2,
            other.color[2] as f32 * weight2,
        );

        Self {
            position: self.position * weight1 + other.position * weight2,
            normal: self.normal * weight1 + other.normal * weight2,
            color: Vector3::new(
                color[0].min(255.0) as u8,
                color[1].min(255.0) as u8,
                color[2].min(255.0) as u8,
            ),
            radius: self.radius * weight1 + other.radius * weight2,
            confidence: self.confidence * weight1 + other.confidence * weight2,
            age: self.age.max(other.age),
        }
    }
}

/// A builder for surfels. Building surlfes requires knowledge of the camera intrinsics.
/// This builder caches the necessary information to build surfels.
pub struct SurfelBuilder {
    camera_center: Vector2<f32>,
    inv_mean_focal_length: f32,
    max_center_distance: f32,
}

impl SurfelBuilder {
    /// Create a new surfel builder from the camera intrinsics.
    ///
    /// # Arguments
    ///
    /// * `camera` - The camera intrinsics.
    ///
    /// # Returns
    ///
    /// A new surfel builder.
    pub fn new(camera: &CameraIntrinsics) -> Self {
        let camera_center = Vector2::new(camera.cx as f32, camera.cy as f32);
        let inv_mean_focal_length = 1.0 / ((camera.fx + camera.fy) as f32 * 0.5);
        let max_center_distance = (camera_center - Vector2::new(0.0, 0.0)).norm();
        SurfelBuilder {
            camera_center,
            inv_mean_focal_length,
            max_center_distance,
        }
    }

    /// Create a surfel from a range pixel. See
    /// Kähler, O., Prisacariu, V. A., & Murray, D. W. (2016).
    /// Real-time large-scale dense 3D reconstruction with loop closure.
    /// In Computer Vision–ECCV 2016
    ///
    /// # Arguments
    ///
    /// * `range_point` - The pixel's 3D point.
    /// * `range_normal` - The pixel's 3D normal.
    /// * `color` - The color.
    /// * `range_coord` - The pixel's 2D coordinates [u, v].
    /// * `age` - The age of the surfel.
    ///
    /// # Returns
    ///
    /// A new surfel.
    pub fn from_range_pixel(
        &self,
        range_point: Vector3<f32>,
        range_normal: Vector3<f32>,
        color: Vector3<u8>,
        range_coord: Vector2<f32>,
        age: u32,
    ) -> Surfel {
        let _1_sqrt_2: f32 = 1.0f32 / 2.0f32.sqrt();
        let radius = _1_sqrt_2 * range_point[2] * self.inv_mean_focal_length;
        let radius = (radius / range_normal[2].abs()).min(2.0 * radius);

        let constant_weight: f32 = 2.0 * 0.6f32.powf(2.0);
        let weight = (range_coord - self.camera_center).norm() / self.max_center_distance;
        let weight = (-(weight * weight) / constant_weight).exp() * weight;

        Surfel {
            position: range_point,
            normal: range_normal,
            color,
            radius,
            confidence: weight,
            age,
        }
    }

    /// Create a surfel from a range image, see `from_range_pixel`.
    ///
    /// # Arguments
    ///
    /// * `rimage` - The range image.
    ///
    /// # Returns
    ///
    /// A vector of surfels expanded from the valid range image points.
    pub fn from_range_image(&self, rimage: &RangeImage) -> Vec<Surfel> {
        let mut surfels = Vec::new();
        let normals = rimage.normals.as_ref().unwrap();
        let colors = rimage.colors.as_ref().unwrap();
        for ((row, col), mask) in rimage.mask.indexed_iter() {
            if *mask == 0 {
                continue;
            }
            let range_point = rimage.points[(row, col)];
            let range_normal = normals[(row, col)];
            let range_color = colors[(row, col)];

            surfels.push(self.from_range_pixel(
                range_point,
                range_normal,
                range_color,
                Vector2::new(col as f32, row as f32),
                0,
            ));
        }

        surfels
    }
}

impl TransformableMove<Surfel> for Transform {
    fn transform(&self, surfel: Surfel) -> Surfel {
        Surfel {
            position: self.transform_vector(&surfel.position),
            normal: self.transform_normal(&surfel.normal),
            color: surfel.color,
            radius: surfel.radius,
            confidence: surfel.confidence,
            age: surfel.age,
        }
    }
}
