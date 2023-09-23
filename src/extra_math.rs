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
