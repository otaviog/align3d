use nalgebra::Vector3;
use ndarray::{Array2, Axis};

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

    pub fn from_points(points: &Array2<f32>) -> Self {
        let center = points.mean_axis(Axis(0)).unwrap();
        let radius = points
            .axis_iter(Axis(0))
            .map(|row| {
                let sub = &row - &center;
                sub.dot(&sub)
            })
            .reduce(f32::max)
            .unwrap()
            .sqrt();

        Self {
            center: Vector3::new(center[0], center[1], center[2]),
            radius,
        }
    }

    pub fn from_point_iter<I>(point_iter: I) -> Self
    where
        I: Iterator<Item = Vector3<f32>> + Clone
    {
        let mut count = 0;
        let center = point_iter.clone().fold(Vector3::zeros(), |sum, p| {
            count += 1;
            sum + p
        });
        let center = center / count as f32;
        Self {
            center,
            radius: point_iter
                .map(|p| center.dot(&p))
                .reduce(f32::max)
                .unwrap()
                .sqrt(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.radius < 0.0
    }
}
