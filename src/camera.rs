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
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl CameraIntrinsics {
    /// Create a new camera intrinsic parameters.
    ///
    /// # Arguments
    ///
    /// * fx: Focal length and pixel scale in the X-axis.
    /// * fy: Focal length and pixel scale in the Y-axis.
    /// * cx: Camera X-center.
    /// * cy: Camera Y-center.
    /// * width: Image width in pixels.
    /// * height: Image height in pixels.
    ///
    /// # Returns
    ///
    /// * A new camera intrinsic parameters.
    pub fn from_simple_intrinsic(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        width: usize,
        height: usize,
    ) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
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
        (
            point[0] * self.fx as f32 / point[2] + self.cx as f32,
            point[1] * self.fy as f32 / point[2] + self.cy as f32,
        )
    }

    /// Return the Jacobian of the projection.
    /// The Jacobian is the matrix of partial derivatives of the projection function.
    ///
    /// # Arguments
    ///
    /// * point: The 3D point.
    ///
    /// # Returns
    ///
    /// * ((dx/dfx, dx/dfy), (dy/dfx, dy/dfy)
    pub fn project_grad(&self, point: &Vector3<f32>) -> ((f32, f32), (f32, f32)) {
        let z = point[2];
        let zz = z * z;
        (
            (self.fx as f32 / z, -point[0] * self.fx as f32 / zz),
            (self.fy as f32 / z, -point[1] * self.fy as f32 / zz),
        )
    }

    /// Backproject a 2D point into 3D space.
    ///
    /// # Arguments
    ///
    /// * x: The x coordinate.
    /// * y: The y coordinate.
    /// * z: The depth.
    ///
    /// # Returns
    ///
    /// * The 3D point.
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
        self.width = width;
        self.height = height;
    }
}
/// A pinhole camera. It is defined by its intrinsic parameters and its pose in the world.
#[derive(Clone, Debug)]
pub struct PinholeCamera {
    pub intrinsics: CameraIntrinsics,
    pub camera_to_world: Transform,
    world_to_camera: Transform,
}

impl PinholeCamera {
    /// Create a new pinhole camera.
    ///
    /// # Arguments
    ///
    /// * intrinsics: The camera intrinsic parameters.
    /// * camera_to_world: The camera pose in the world.
    ///
    /// # Returns
    ///
    /// * A new pinhole camera.
    pub fn new(intrinsics: CameraIntrinsics, camera_to_world: Transform) -> Self {
        Self {
            intrinsics,
            world_to_camera: camera_to_world.inverse(),
            camera_to_world,
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

    /// Returns the projected 3D point if it is visible in the image.
    ///
    /// # Arguments
    ///
    /// * point: The 3D point.
    ///
    /// # Returns
    ///
    /// * (x and y) coordinates if the point is visible, None otherwise.
    pub fn project_if_visible(&self, point: &Vector3<f32>) -> Option<(f32, f32)> {
        let (x, y) = self.project(point);

        if x >= 0.0
            && x < self.intrinsics.width as f32
            && y >= 0.0
            && y < self.intrinsics.height as f32
        {
            Some((x, y))
        } else {
            None
        }
    }
}
