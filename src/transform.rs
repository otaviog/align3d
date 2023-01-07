use nalgebra::{self, Rotation3};

use nalgebra::{Isometry3, Matrix4, Translation3, UnitQuaternion, Vector3, Vector6};
use ndarray::Axis;
use ndarray::{self, Array2};

use std::ops;

pub struct Rotation(Rotation3<f32>);

impl ops::Mul<&ndarray::Array2<f32>> for &Rotation {
    type Output = ndarray::Array2<f32>;

    fn mul(self, rhs: &ndarray::Array2<f32>) -> Self::Output {
        let mut result = ndarray::Array2::<f32>::zeros((rhs.len_of(Axis(0)), 3));

        for (in_iter, mut out_iter) in rhs.axis_iter(Axis(0)).zip(result.axis_iter_mut(Axis(0))) {
            let v = self.0 * Vector3::new(in_iter[0], in_iter[1], in_iter[2]);
            out_iter[0] = v[0];
            out_iter[1] = v[1];
            out_iter[2] = v[2];
        }

        result
    }
}

#[derive(Clone, Debug)]
pub struct Transform(Isometry3<f32>);

impl Transform {
    pub fn eye() -> Self {
        Self(Isometry3::<f32>::from_parts(
            Translation3::new(0.0, 0.0, 0.0),
            UnitQuaternion::new(Vector3::<f32>::zeros()),
        ))
    }

    pub fn from_se3_exp(translation_so3: &Vector6<f32>) -> Self {
        let translation =
            Translation3::new(translation_so3[0], translation_so3[1], translation_so3[2]);
        let so3 = Vector3::new(translation_so3[3], translation_so3[4], translation_so3[5]);

        Self(Isometry3::<f32>::from_parts(
            translation,
            UnitQuaternion::from_scaled_axis(so3),
        ))
    }

    pub fn from_matrix4(matrix: &Matrix4<f32>) -> Self {
        let translation = Translation3::new(matrix[(0, 3)], matrix[(1, 3)], matrix[(2, 3)]);
        let so3 = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
            &matrix.fixed_slice::<3, 3>(0, 0).into_owned(),
        ));
        Self(Isometry3::<f32>::from_parts(translation, so3))
    }

    pub fn transform(&self, mut rhs: Array2<f32>) -> Array2<f32> {
        for mut point in rhs.axis_iter_mut(Axis(0)) {
            let v = self.0 * Vector3::new(point[0], point[1], point[2]);
            point[0] = v[0];
            point[1] = v[1];
            point[2] = v[2];
        }

        rhs
    }

    pub fn ortho_rotation(&self) -> Rotation {
        Rotation(
            self.0.rotation.to_rotation_matrix()
        )
    }
}

impl ops::Mul<&ndarray::Array2<f32>> for &Transform {
    type Output = ndarray::Array2<f32>;

    fn mul(self, rhs: &ndarray::Array2<f32>) -> Self::Output {
        let mut result = ndarray::Array2::<f32>::zeros((rhs.len_of(Axis(0)), 3));

        for (in_iter, mut out_iter) in rhs.axis_iter(Axis(0)).zip(result.axis_iter_mut(Axis(0))) {
            let v = self.0 * Vector3::new(in_iter[0], in_iter[1], in_iter[2]);
            out_iter[0] = v[0];
            out_iter[1] = v[1];
            out_iter[2] = v[2];
        }

        result
    }
}

impl ops::Mul<&Vector3<f32>> for &Transform {
    type Output = Vector3<f32>;

    fn mul(self, rhs: &Vector3<f32>) -> Self::Output {
        self.0 * rhs
    }
}

impl ops::Mul<&Transform> for &Transform {
    type Output = Transform;

    fn mul(self, rhs: &Transform) -> Self::Output {
        Transform(self.0 * rhs.0)
    }
}

impl From<Transform> for Matrix4<f32> {
    fn from(transform: Transform) -> Self {
        transform.0.into()
    }
}

#[cfg(test)]
mod tests {
    use super::Transform;
    use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
    use ndarray::array;

    use ndarray::prelude::*;

    fn assert_array(f1: &Array2<f32>, f2: &Array2<f32>) -> bool {
        if f1.shape() != f2.shape() {
            return false;
        }

        let shape = f1.shape();
        let size = shape[0] * shape[1];
        f1.clone()
            .into_shape((size, 1))
            .iter()
            .zip(f2.clone().into_shape((size, 1)).iter())
            .all(|(v1, v2)| (v1[[0, 0]] - v2[[0, 0]]).abs() < 1e-5)
    }

    #[test]
    fn test_mul_op() {
        let transform = Transform::eye();
        let points = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
        let mult_result = &transform * &points;

        assert_eq!(mult_result, points);

        let transform = Transform(Isometry3::from_parts(
            Translation3::<f32>::new(0., 0., 3.),
            UnitQuaternion::<f32>::from_scaled_axis(Vector3::y() * std::f32::consts::PI),
        ));

        assert!(assert_array(
            &(&transform * &array![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
            &array![[-1.0, 2.0, 0.0], [-1.0, 2.0, 0.0]]
        ));
    }

    #[test]
    fn test_transform() {
        let transform = Transform(Isometry3::from_parts(
            Translation3::<f32>::new(0., 0., 3.),
            UnitQuaternion::<f32>::from_scaled_axis(Vector3::y() * std::f32::consts::PI),
        ));
        let mut points = array![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
        points = transform.transform(points);

        assert!(assert_array(
            &points,
            &array![[-1.0, 2.0, 0.0], [-1.0, 2.0, 0.0]]
        ));
    }
}
