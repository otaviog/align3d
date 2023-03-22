use nalgebra::{RealField, Vector3};

/// Returns the angle between two normals in radians.
///
/// # Arguments
///
/// * lfs - Left normal.
/// * rfs - Right normal.
///
/// # Returns
///
/// * Angle between the two normals in radians.
pub fn angle_between_normals<T: RealField>(lfs: &Vector3<T>, rfs: &Vector3<T>) -> T {
    lfs.dot(rfs).acos().abs()
}

pub fn angle_between_vecs(lfs: &Vector3<f32>, rfs: &Vector3<f32>) -> f32 {
    (lfs.dot(rfs) / (lfs.norm() * rfs.norm())).acos().abs()
}
