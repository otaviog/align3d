use crate::transform::Transform;

pub fn transform_difference(lfs: &Transform, rhs: &Transform) -> f32 {
    let lfs_inv = lfs.inverse();
    let diff = &lfs_inv * rhs;
    diff.angle() + diff.translation().norm()
}
