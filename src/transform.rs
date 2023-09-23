use nalgebra::{
    Isometry3, Matrix3, Matrix4, Quaternion, Rotation3, Translation3, UnitQuaternion, UnitVector3,
    Vector3, Vector6,
};
use ndarray::Array1;

use std::ops;

pub enum LieGroup {
    Se3(Vector6<f32>),
    So3(Vector3<f32>),
}

/// A Rigid Body Transform in 3D space.
/// This wraps Isometry3 from nalgebra and provides methods for working with
/// Align3d's data structures.
#[derive(Clone, Debug)]
pub struct Transform(pub Isometry3<f32>);

impl Default for Transform {
    /// Create a new transform with zero translation and zero rotation.
    fn default() -> Self {
        Self::eye()
    }
}

impl Transform {
    /// Create a new transform with zero translation and zero rotation.
    pub fn eye() -> Self {
        Self(Isometry3::<f32>::from_parts(
            Translation3::new(0.0, 0.0, 0.0),
            UnitQuaternion::new(Vector3::<f32>::zeros()),
        ))
    }

    /// Create a new transform from a translation vector and a rotation quaternion.
    pub fn new(xyz: &Vector3<f32>, rotation: &Quaternion<f32>) -> Self {
        Self(Isometry3::<f32>::from_parts(
            Translation3::new(xyz[0], xyz[1], xyz[2]),
            UnitQuaternion::from_quaternion(*rotation),
        ))
    }

    fn exp_so3(omega: &Vector3<f32>) -> (f32, UnitQuaternion<f32>) {
        // https://github.com/strasdat/Sophus/blob/main-1.x/sophus/so3.hpp
        const EPSILON: f32 = 1e-8;
        let theta_sq = omega.norm_squared();

        let (theta, imag_factor, real_factor) = if theta_sq < EPSILON * EPSILON {
            let theta_po4 = theta_sq * theta_sq;
            (
                0.0,
                0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_po4,
                1.0 - (1.0 / 8.0) * theta_sq + (1.0 / 384.0) * theta_po4,
            )
        } else {
            let theta = theta_sq.sqrt();
            let half_theta = 0.5 * theta;
            (theta, half_theta.sin() / theta, half_theta.cos())
        };

        (
            theta,
            UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                real_factor,
                imag_factor * omega[0],
                imag_factor * omega[1],
                imag_factor * omega[2],
            )),
        )
    }

    /// Create a transform from a 6D vector of the form [x, y, z, rx, ry, rz] where x, y, and z are the translation part
    /// and rx,ry, and rz are the rotation part in the form of a scaled axis.
    ///
    /// # Arguments
    ///
    /// * xyz_so3 - 6D vector of the form [x, y, z, rx, ry, rz]
    ///
    /// # Returns
    ///
    /// * Transform
    pub fn exp(params: &LieGroup) -> Self {
        const EPSILON: f32 = 1e-8;

        match params {
            LieGroup::Se3(xyz_so3) => {
                let omega = Vector3::new(xyz_so3[3], xyz_so3[4], xyz_so3[5]);
                let (theta, quat) = Self::exp_so3(&omega);
                let theta_sq = theta * theta;
                let xyz = {
                    let left_jacobian = {
                        let big_omega = omega.cross_matrix();

                        if theta_sq < EPSILON {
                            Matrix3::identity() + (big_omega * 0.5)
                        } else {
                            let big_omega_squared = big_omega * big_omega;
                            Matrix3::identity()
                                + (1.0 - theta.cos()) / theta_sq * big_omega
                                + (theta - theta.sin()) / (theta_sq * theta) * big_omega_squared
                        }
                    };

                    left_jacobian * Vector3::new(xyz_so3[0], xyz_so3[1], xyz_so3[2])
                };
                Self(Isometry3::<f32>::from_parts(xyz.into(), quat))
            }
            &LieGroup::So3(so3) => {
                let omega = Vector3::new(so3[3], so3[4], so3[5]);
                let (_, quat) = Self::exp_so3(&omega);
                Self(Isometry3::<f32>::from_parts(
                    Translation3::new(0.0, 0.0, 0.0),
                    quat,
                ))
            }
        }
    }

    /// Create a transform from a 4x4 matrix homogeneous matrix.
    pub fn from_matrix4(matrix: &Matrix4<f32>) -> Self {
        let translation = Translation3::new(matrix[(0, 3)], matrix[(1, 3)], matrix[(2, 3)]);
        let so3 = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
            &matrix.fixed_slice::<3, 3>(0, 0).into_owned(),
        ));
        Self(Isometry3::<f32>::from_parts(translation, so3))
    }

    /// Transforms a 3D point.
    ///
    /// # Arguments
    ///
    /// * rhs - 3D point.
    ///
    /// # Returns
    ///
    /// * 3D point transformed.
    pub fn transform_vector(&self, rhs: &Vector3<f32>) -> Vector3<f32> {
        self.0.rotation * rhs + self.0.translation.vector
    }

    /// Transforms a 3D normal. That's use only the rotation part of the transform.
    ///
    /// # Arguments
    ///
    /// * rhs - 3D normal.
    ///
    /// # Returns
    ///
    /// * 3D normal transformed.
    pub fn transform_normal(&self, rhs: &Vector3<f32>) -> Vector3<f32> {
        self.0.rotation * rhs
    }

    /// Transforms an array of 3D vectors/points.
    ///
    /// # Arguments
    ///
    /// * rhs - Array of 3D vectors/points of shape (N, 3). It'll reuse this array as result's storage.
    ///
    /// # Returns
    ///
    /// * Array of 3D points of shape (N, 3) transformed.
    pub fn transform_vectors(&self, mut rhs: Array1<Vector3<f32>>) -> Array1<Vector3<f32>> {
        for point in rhs.iter_mut() {
            *point = self.transform_vector(point);
        }

        rhs
    }

    /// Transforms an array of 3D normals, i.e., it only uses the rotation part of the transform.
    ///
    /// # Arguments
    ///
    /// * rhs - Array of 3D normals of shape (N, 3). It'll reuse this array as result's storage.
    ///
    /// # Returns
    ///
    /// * Array of 3D normals of shape (N, 3) transformed.
    pub fn transform_normals(&self, mut rhs: Array1<Vector3<f32>>) -> Array1<Vector3<f32>> {
        for point in rhs.iter_mut() {
            *point = self.transform_normal(point);
        }

        rhs
    }

    /// Inverts the transform.
    pub fn inverse(&self) -> Self {
        Self(self.0.inverse())
    }

    /// Returns the rotation angle in radians.
    pub fn angle(&self) -> f32 {
        self.0.rotation.angle()
    }

    /// Returns the translation part.
    pub fn translation(&self) -> Vector3<f32> {
        self.0.translation.vector
    }
}

impl ops::Mul<&Transform> for &Transform {
    type Output = Transform;

    /// Composes two transforms.
    ///
    /// # Arguments
    ///
    /// * rhs - Transform to compose with, i.e. self * rhs, where rhs is applied first.
    ///
    /// # Returns
    ///
    /// * Composed transform.
    fn mul(self, rhs: &Transform) -> Self::Output {
        Transform(self.0 * rhs.0)
    }
}

// impl Into<Matrix4<f32>> for Transform {
//     /// Converts a transform to a 4x4 matrix.
//     fn into(self) -> Matrix4<f32> {
//         self.0.into()
//     }
// }

impl From<&Transform> for Matrix4<f32> {
    /// Converts a transform to a 4x4 matrix.
    fn from(transform: &Transform) -> Self {
        transform.0.into()
    }
}

impl From<&Matrix4<f32>> for Transform {
    /// Converts a 4x4 matrix to a transform.
    fn from(matrix: &Matrix4<f32>) -> Self {
        Transform::from_matrix4(matrix)
    }
}

pub struct TransformBuilder {
    pub translation: Vector3<f32>,
    pub rotation: UnitQuaternion<f32>,
}

impl Default for TransformBuilder {
    fn default() -> Self {
        Self {
            translation: Vector3::new(0.0, 0.0, 0.0),
            rotation: UnitQuaternion::default(),
        }
    }
}

/// Easy to use builder for transforms.
impl TransformBuilder {
    /// Sets the translation.
    pub fn translation(&mut self, translation: Vector3<f32>) -> &mut Self {
        self.translation = translation;
        self
    }

    /// Sets the rotation.
    ///
    /// # Arguments
    ///
    /// * axis - Rotation axis.
    /// * angle - Rotation angle in radians.
    pub fn axis_angle(&mut self, axis: UnitVector3<f32>, angle: f32) -> &mut Self {
        self.rotation = UnitQuaternion::from_axis_angle(&axis, angle);
        self
    }

    /// Generates a transform from the builder.
    pub fn build(&self) -> Transform {
        Transform(Isometry3::from_parts(
            Translation3::from(self.translation),
            self.rotation,
        ))
    }
}

/// A trait for any object that can be transform stuff (i.e., bounding spheres, point clouds).
pub trait Transformable<Type> {
    fn transform(&self, value: &Type) -> Type;
}

pub trait TransformableMove<Type> {
    fn transform(&self, value: Type) -> Type;
}

#[cfg(test)]
mod tests {
    use crate::transform::LieGroup;
    use crate::unit_test::access::FlattenVector3;

    use super::Transform;
    use nalgebra::Vector6;
    use nalgebra::{Isometry3, Matrix4, Translation3, UnitQuaternion, Vector3, Vector4};
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
        let points = array![
            Vector3::new(1., 2., 3.),
            Vector3::new(4., 5., 6.),
            Vector3::new(7., 8., 9.)
        ];
        let mult_result = transform.transform_vectors(points.clone());

        assert_eq!(mult_result, points);

        let transform = Transform(Isometry3::from_parts(
            Translation3::<f32>::new(0., 0., 3.),
            UnitQuaternion::<f32>::from_scaled_axis(Vector3::y() * std::f32::consts::PI),
        ));

        assert!(assert_array(
            &transform
                .transform_vectors(Array1::from_iter([
                    Vector3::new(1.0, 2.0, 3.0),
                    Vector3::new(1.0, 2.0, 3.0)
                ]))
                .flatten_vector3(),
            &array![[-1.0, 2.0, 0.0], [-1.0, 2.0, 0.0]]
        ));
    }

    #[test]
    fn test_transform() {
        let transform = Transform(Isometry3::from_parts(
            Translation3::<f32>::new(0., 0., 3.),
            UnitQuaternion::<f32>::from_scaled_axis(Vector3::y() * std::f32::consts::PI),
        ));
        let mut points = array![Vector3::new(1.0, 2.0, 3.0), Vector3::new(1.0, 2.0, 3.0)];
        points = transform.transform_vectors(points);

        assert!(assert_array(
            &points.flatten_vector3(),
            &array![[-1.0, 2.0, 0.0], [-1.0, 2.0, 0.0]]
        ));
    }

    #[test]
    fn test_exp() {
        let transform = Transform::exp(&LieGroup::Se3(Vector6::new(1.0, 2.0, 3.0, 0.4, 0.5, 0.3)));

        assert!(assert_array(
            &transform
                .transform_vectors(array![Vector3::new(5.5, 6.4, 7.8)])
                .flatten_vector3(),
            &array![[8.9848175, 6.9635687, 9.880962]]
        ));

        let se3 = Transform::exp(&LieGroup::Se3(Vector6::new(1.0, 2.0, 3.0, 0.4, 0.5, 0.3)));
        let matrix = Matrix4::from(&se3);
        let test_mult = matrix * Vector4::new(1.0, 2.0, 3.0, 1.0);
        assert_eq!(
            test_mult,
            Vector4::new(3.5280778, 2.8378963, 5.8994026, 1.0000)
        );
        let test_mult = se3.transform_vector(&Vector3::new(1.0, 2.0, 3.0));
        assert!(
            (test_mult - Vector3::new(3.5280778, 2.8378963, 5.8994026))
                .norm()
                .abs()
                < 1e-5
        );
    }

    #[test]
    fn test_compose() {
        let transform1 = Transform(Isometry3::from_parts(
            Translation3::<f32>::new(0., 0., 3.),
            UnitQuaternion::<f32>::identity(),
        ));
        let transform2 = Transform(Isometry3::from_parts(
            Translation3::<f32>::new(0., 0., 3.),
            UnitQuaternion::<f32>::from_scaled_axis(Vector3::y() * std::f32::consts::PI / 2.0),
        ));

        let transform = &transform1 * &transform2;
        assert!(assert_array(
            &transform
                .transform_vectors(array![
                    Vector3::new(1.0, 2.0, 3.0),
                    Vector3::new(1.0, 2.0, 3.0)
                ])
                .flatten_vector3(),
            &array![[2.9999998, 2.0, 5.0], [2.9999998, 2.0, 5.0]]
        ));
    }
}
