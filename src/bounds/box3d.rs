use nalgebra::ClosedAdd;
use nalgebra::Scalar;
use nalgebra::Vector3;
use num::Zero;

#[derive(Clone)]
pub struct Box3D<T>
where
    T: Scalar + Zero + ClosedAdd,
{
    pub min: Vector3<T>,
    pub max: Vector3<T>,
}

impl<T> Box3D<T>
where
    T: Scalar + Zero + ClosedAdd,
{
    ///
    /// # Arguments
    ///
    /// * `start_point`: The minimum point in the X, Y, and Z axis.
    /// * `size`: The size of in the X, Y, and Z axis.
    pub fn from_extents(start_point: Vector3<T>, size: Vector3<T>) -> Self {
        Box3D {
            min: start_point.clone(),
            max: start_point + &size,
            //max: size
        }
    }

    pub fn empty() -> Self {
        Self {
            min: Vector3::<T>::zero(),
            max: Vector3::<T>::zero(),
        }
    }

    pub fn is_empty(&self) -> bool {
        true
    }
}

pub type Box3Df = Box3D<f32>;
