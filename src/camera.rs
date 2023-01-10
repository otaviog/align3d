use nalgebra::Vector3;

use super::transform::Transform;

/// Camera intrinsic parameters.
#[derive(Clone, Debug)]
pub struct Camera {
    /// Focal length and pixel scale in the X-axis.
    pub fx: f64,
    /// Focal length and pixel scale in the Y-axis.
    pub fy: f64,
    /// Camera X-center.
    pub cx: f64,
    /// Camera Y-center.
    pub cy: f64,
    pub camera_to_world: Option<Transform>,
}

pub struct CameraBuilder(Camera);

impl CameraBuilder {
    /// Creates a camera using the focal length with pixel
    /// scales (fx, fy) and camera center (cx, cy).
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
    /// Project a 3D point into image space.
    /// 
    /// # Arguments
    /// 
    /// * point: The 3D point.
    /// 
    /// # Returns
    /// 
    /// * (x and y) coordinates.
    pub fn project(&self, point: Vector3<f32>) -> (f32, f32) {
        (
            (point.x * self.fx as f32 + self.cx as f32) / point.z,
            (point.y * self.fy as f32 + self.cy as f32) / point.z,
        )
    }

    pub fn backproject(&self, x: f32, y: f32, z: f32) -> Vector3<f32> {
        Vector3::new(
            (x - self.cx as f32) * z / self.fx as f32,
            (y - self.cy as f32) * z / self.fy as f32,
            z,
        )
    }

    pub fn scale(&self, scale: f64) -> Self {
        if self.camera_to_world.is_some() {
            panic!("Not implemented: align3d::Camera.scale() with camera_to_world.");
        }

        Self {
            fx: self.fx * scale,
            fy: self.fy * scale,
            cx: self.cx * scale,
            cy: self.cy * scale,
            camera_to_world: None,
        }
    }
}
