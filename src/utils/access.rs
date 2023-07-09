use nalgebra::Vector3;
use ndarray::{Array1, Array2, ArrayBase, Ix1, ViewRepr};

use crate::error::A3dError;

pub trait ToVector3<T> {
    fn to_vector3(&self) -> Vector3<T>;
}

impl<T> ToVector3<T> for ArrayBase<ViewRepr<&T>, Ix1>
where
    T: Clone,
{
    fn to_vector3(&self) -> Vector3<T> {
        Vector3::new(self[0].clone(), self[1].clone(), self[2].clone())
    }
}

pub trait FlattenVector3<T: Copy> {
    type Output;
    fn flatten_vector3(&self) -> Self::Output;
}

impl<T: Copy> FlattenVector3<T> for Array1<Vector3<T>> {
    type Output = Array2<T>;
    fn flatten_vector3(&self) -> Self::Output {
        Array2::from_shape_fn((self.len(), 3), |(i, j)| self[i][j])
    }
}

pub trait UnflattenVector3<T: Copy> {
    type Output;
    fn unflatten_vector3(&self) -> Result<Self::Output, A3dError>;
}

impl<T: Copy> UnflattenVector3<T> for Array2<T> {
    type Output = Array1<Vector3<T>>;
    fn unflatten_vector3(&self) -> Result<Self::Output, A3dError> {
        let dim = self.dim();
        if dim.1 != 3 {
            return Err(A3dError::Assertion(
                "Invalid dim size. Must be 3 2nd dimension".to_string(),
            ));
        }

        Ok(Array1::from_shape_fn(dim.0, |i| {
            Vector3::new(self[(i, 0)], self[(i, 1)], self[(i, 2)])
        }))
    }
}
