use crate::transform::Transform;

/// Metrics for comparing two transforms.
#[derive(Clone, Debug)]
pub struct TransformMetrics {
    /// Angle between the two transforms in radians.
    pub angle: f32,
    /// Translation vector size between the two transforms.
    pub translation: f32,
}

impl TransformMetrics {
    /// Creates a new `TransformMetrics` from two transforms.
    pub fn new(lfs: &Transform, rhs: &Transform) -> Self {
        let lfs_inv = lfs.inverse();
        let diff = &lfs_inv * rhs;

        Self {
            angle: diff.angle(),
            translation: diff.translation().norm(),
        }
    }

    /// Returns the total error of the two transforms.
    pub fn total(&self) -> f32 {
        self.angle + self.translation
    }
}

impl std::fmt::Display for TransformMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "angle: {:.2}°, translation: {:.5}",
            self.angle.to_degrees(),
            self.translation
        )
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Quaternion, Vector3};

    use super::*;

    #[test]
    fn test_transform_metrics() {
        let sample0 = Transform::new(
            &Vector3::new(0.00022050377, 7.3633055e-5, -1.51071e-5),
            Quaternion::new(2.059626e-5, 0.00888227, 0.0008264509, 0.99996024),
        );
        let sample1 = Transform::new(
            &Vector3::new(0.00022050377, 7.3633055e-5, -1.51071e-5),
            Quaternion::new(2.059626e-5, 0.00888227, 0.0008264509, 0.99996024),
        );

        let metrics = TransformMetrics::new(&sample0, &sample1);

        assert_eq!(metrics.angle, 0.0);
        assert_eq!(metrics.translation, 0.0);
        assert_eq!(metrics.total(), 0.0);
    }
}
