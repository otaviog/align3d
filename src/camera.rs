use nalgebra::Vector3;

use super::transform::Transform;

#[derive(Clone, Debug)]
pub struct Camera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub camera_to_world: Option<Transform>,
}

pub struct CameraBuilder(Camera);

impl CameraBuilder {
    pub fn from_simple_intrinsics(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self(Camera {
            fx,
            fy,
            cx,
            cy,
            camera_to_world: None,
        })
    }

    pub fn camera_to_world(&'_ mut self, value: Option<Transform>) -> &'_ mut CameraBuilder {
        self.0.camera_to_world = value;
        self
    }

    pub fn build(&self) -> Camera {
        self.0.clone()
    }
}

impl Camera {
    pub fn backproject(&self, x: f32, y: f32, z: f32) -> Vector3<f32> {
        Vector3::new(
            (x - self.cx as f32) * z / self.fx as f32,
            (y - self.cy as f32) * z / self.fy as f32,
            z,
        )
    }

    pub fn project(&self, point: Vector3<f32>) -> (f32, f32) {
        (
            (point.x * self.fx as f32 + self.cx as f32) / point.z,
            (point.y * self.fy as f32 + self.cy as f32) / point.z,
        )
    }
}
