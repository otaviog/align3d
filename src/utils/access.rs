use nalgebra::Vector3;
use ndarray::{ArrayBase, Ix1, ViewRepr};

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
