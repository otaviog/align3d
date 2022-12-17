use nalgebra::Vector3;

#[derive(Clone, Copy)]
pub struct Sphere3Df {
    pub center: Vector3<f32>,
    pub radius: f32,
}

impl Sphere3Df {
    pub fn empty() -> Self {
        Self {
            center: Vector3::zeros(),
            radius: -1.0,
        }
    }
}
