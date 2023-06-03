use nalgebra::Vector3;

use super::transform::Transform;

/// Camera intrinsic parameters.
#[derive(Clone, Debug)]
pub struct CameraIntrinsics {
    /// Focal length and pixel scale in the X-axis.
    pub fx: f64,
    /// Focal length and pixel scale in the Y-axis.
    pub fy: f64,
    /// Camera X-center.
    pub cx: f64,
    /// Camera Y-center.
    pub cy: f64,
    pub width: Option<usize>,
    pub height: Option<usize>,
}

impl CameraIntrinsics {
    pub fn from_simple_intrinsic(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            width: None,
            height: None,
        }
    }

    /// Project a 3D point into image space.
    ///
    /// # Arguments
    ///
    /// * point: The 3D point.
    ///
    /// # Returns
    ///
    /// * (x and y) coordinates.
    pub fn project(&self, point: &Vector3<f32>) -> (f32, f32) {
        (
            point[0] * self.fx as f32 / point[2] + self.cx as f32,
            point[1] * self.fy as f32 / point[2] + self.cy as f32,
        )
    }

    pub fn project_grad(&self, point: &Vector3<f32>) -> ((f32, f32), (f32, f32)) {
        let z = point[2];
        let zz = z * z;
        (
            (self.fx as f32 / z, -point[0] * self.fx as f32 / zz),
            (self.fy as f32 / z, -point[1] * self.fy as f32 / zz),
        )
    }

    pub fn backproject(&self, x: f32, y: f32, z: f32) -> Vector3<f32> {
        Vector3::new(
            (x - self.cx as f32) * z / self.fx as f32,
            (y - self.cy as f32) * z / self.fy as f32,
            z,
        )
    }

    /// Scale the camera parameters according to the given scale.
    ///
    /// # Arguments
    ///
    /// * scale: The scale factor.
    ///
    /// # Returns
    ///
    /// * A new camera with scaled parameters.
    pub fn scale(&self, scale: f64) -> Self {
        Self {
            fx: self.fx * scale,
            fy: self.fy * scale,
            cx: self.cx * scale,
            cy: self.cy * scale,
            width: self.width,
            height: self.height,
        }
    }

    pub fn size(&mut self, width: usize, height: usize) {
        self.width = Some(width);
        self.height = Some(height);
    }
}

pub enum PointSpace {
    Camera(Vector3<f32>),
    World(Vector3<f32>),
}

#[derive(Clone, Debug)]
pub struct PinholeCamera {
    pub intrinsics: CameraIntrinsics,
    pub camera_to_world: Transform,
    world_to_camera: Transform,
    pub width: usize,
    pub height: usize,
}

impl PinholeCamera {
    pub fn new(
        intrinsics: CameraIntrinsics,
        camera_to_world: Transform,
        width: usize,
        height: usize,
    ) -> Self {
        Self {
            intrinsics,
            world_to_camera: camera_to_world.inverse(),
            camera_to_world,
            width,
            height,
        }
    }

    /// Project a 3D point into image space.
    ///
    /// # Arguments
    ///
    /// * point: The 3D point.
    ///
    /// # Returns
    ///
    /// * (x and y) coordinates.
    pub fn project(&self, point: &Vector3<f32>) -> (f32, f32) {
        self.intrinsics
            .project(&self.world_to_camera.transform_vector(point))
    }

    pub fn project_if_visible(&self, point: &Vector3<f32>) -> Option<(f32, f32)> {
        let (x, y) = self.project(point);

        if x >= 0.0 && x < self.width as f32 * 2.0 && y >= 0.0 && y < self.height as f32 * 2.0 {
            Some((x, y))
        } else {
            None
        }
    }
}
