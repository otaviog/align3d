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
        let z = point[2];
        (
            point[0] * self.fx as f32 / z + self.cx as f32,
            point[1] * self.fy as f32 / z + self.cy as f32,
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
    width_f32: f32,
    height_f32: f32,
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
        let width_f32 = intrinsics.width as f32;
        let height_f32 = intrinsics.height as f32;
        Self {
            intrinsics,
            world_to_camera: camera_to_world.inverse(),
            camera_to_world,
            width_f32,
            height_f32,
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
    pub fn project(&self, point: &Vector3<f32>) -> (f32, f32, f32) {
        let point = self.world_to_camera.transform_vector(point);
        let (u, v) = self.intrinsics.project(&point);
        (u, v, point[2])
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
    pub fn project_to_image(&self, point: &Vector3<f32>) -> Option<(f32, f32, f32)> {
        let (x, y, z) = self.project(point);
        let (x, y) = (x.round(), y.round());

        if x >= 0.0 && x < self.width_f32 && y >= 0.0 && y < self.height_f32 {
            //Some((x, self.height_f32 - y, z))
            Some((x, y, z))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::transform::Transform;

    #[test]
    pub fn test_project() {
        let camera = super::PinholeCamera::new(
            super::CameraIntrinsics::from_simple_intrinsic(50.0, 50.0, 0.0, 0.0, 100, 100),
            Transform::eye(),
        );

        let point = nalgebra::Vector3::new(1.0, 1.0, 1.0);
        let (x, y, _) = camera.project(&point);
        assert_eq!(x, 50.0);
        assert_eq!(y, 50.0);

        let point = nalgebra::Vector3::new(1.0, 1.5, 1.0);
        let (x, y, _) = camera.project(&point);
        assert_eq!(x, 50.0);
        assert_eq!(y, 75.0);
    }

    #[test]
    pub fn test_project_to_image() {
        let camera = super::PinholeCamera::new(
            super::CameraIntrinsics::from_simple_intrinsic(50.0, 50.0, 0.0, 0.0, 100, 100),
            Transform::eye(),
        );

        let point = nalgebra::Vector3::new(1.0, 1.0, 1.0);
        let (x, y, _) = camera.project_to_image(&point).unwrap();
        assert_eq!(x, 50.0);
        assert_eq!(y, 50.0);

        let point = nalgebra::Vector3::new(1.0, 1.5, 1.0);
        let (x, y, _) = camera.project_to_image(&point).unwrap();
        assert_eq!(x, 50.0);
        assert_eq!(y, 75.0);
    }
}
