use nalgebra::{Vector2, Vector3};
use ndarray::s;

use crate::{camera::PinholeCamera, range_image::RangeImage, utils::access::ToVector3};

pub struct Surfel {
    pub position: Vector3<f32>,
    pub normal: Vector3<f32>,
    pub color: Vector3<u8>,
    pub radius: f32,
    pub confidence: f32,
    pub age: i32,
}

impl Surfel {
    pub fn merge(&self, other: &Surfel, weight1: f32, weight2: f32) -> Self {
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

pub struct RimageSurfelBuilder {
    camera_center: Vector2<f32>,
    inv_mean_focal_length: f32,
    max_center_distance: f32,
}

impl RimageSurfelBuilder {
    pub fn new(camera: &PinholeCamera) -> Self {
        let intrinsics = &camera.intrinsics;
        let camera_center = Vector2::new(intrinsics.cx as f32, intrinsics.cy as f32);
        let inv_mean_focal_length = 1.0 / ((intrinsics.fx + intrinsics.fy) as f32 * 0.5);
        let max_center_distance = (camera_center - Vector2::new(0.0, 0.0)).norm();
        RimageSurfelBuilder {
            camera_center,
            inv_mean_focal_length,
            max_center_distance,
        }
    }

    pub fn build(
        &self,
        range_point: Vector3<f32>,
        range_normal: Vector3<f32>,
        color: Vector3<u8>,
        range_coord: Vector2<f32>,
        age: i32,
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

    pub fn build_from_rimage(&self, rimage: &RangeImage) -> Vec<Surfel> {
        let mut surfels = Vec::new();
        let normals = rimage.normals.as_ref().unwrap();
        let colors = rimage.colors.as_ref().unwrap();
        for ((row, col), mask) in rimage.mask.indexed_iter() {
            if *mask == 0 {
                continue;
            }
            let range_point = rimage.points.slice(s![row, col, ..]).to_vector3();
            let range_normal = normals.slice(s![row, col, ..]).to_vector3();
            let range_color = colors.slice(s![row, col, ..]).to_vector3();

            surfels.push(self.build(
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
