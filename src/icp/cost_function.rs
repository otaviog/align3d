use nalgebra::Vector3;

pub struct PointPlaneDistance {}

fn se3_jacobian(source_point: &Vector3<f32>, target_normal: &Vector3<f32>) -> [f32; 6] {
    let twist = source_point.cross(target_normal);
    [
        target_normal[0],
        target_normal[1],
        target_normal[2],
        twist[0],
        twist[1],
        twist[2],
    ]
}

impl PointPlaneDistance {
    /// Computes the residual and the Jacobian of the point-plane distance.
    ///
    /// # Arguments
    ///
    /// * source_point - 3D point in the source frame.
    /// * target_point - 3D point in the target frame.
    /// * target_normal - Normal of the plane in the target frame.
    ///
    /// # Returns
    ///
    /// * $J^tr$ and $J^tJ$ of the point-plane distance,
    /// where the first is a (6) vector and the former is a (6, 6)
    /// matrix as a (36) vector.
    pub fn jacobian(
        &self,
        source_point: &Vector3<f32>,
        target_point: &Vector3<f32>,
        target_normal: &Vector3<f32>,
    ) -> (f32, [f32; 6]) {
        let residual = (target_point - source_point).dot(target_normal);
        (residual, se3_jacobian(source_point, target_normal))
    }
}

pub struct ColorDistance {}

impl ColorDistance {
    pub fn jacobian(
        &self,
        source_point: &Vector3<f32>,
        target_normal: &Vector3<f32>,
        source_color: f32,
        target_color: f32,
    ) -> (f32, [f32; 6]) {
        (
            source_color - target_color,
            se3_jacobian(source_point, target_normal),
        )
    }
}
