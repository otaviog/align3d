use nalgebra::Vector3;
use ndarray::ArrayView1;

use crate::{
    transform::{Transform, Transformable},
    viz::node::Mat4x4,
};

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

    pub fn from_points(points: &ArrayView1<Vector3<f32>>) -> Self {
        let center: Vector3<f32> =
            nalgebra::convert(points.iter().fold(Vector3::<f64>::zeros(), |accum, point| {
                let point: Vector3<f64> = nalgebra::convert(*point);
                accum + point
            }));

        let radius = points
            .iter()
            .map(|point| {
                let sub = point - center;
                sub.dot(&sub)
            })
            .reduce(f32::max)
            .unwrap()
            .sqrt();

        Self { center, radius }
    }

    pub fn from_point_iter<I>(point_iter: I) -> Self
    where
        I: Iterator<Item = Vector3<f32>> + Clone,
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

    pub fn add(&self, other: &Self) -> Self {
        if self.radius < 0.0 {
            return *other;
        }

        let center = (self.center + other.center) / 2.0;
        let radius = (self.center - center).norm() + self.radius.max(other.radius);
        Self { center, radius }
    }
}

impl Transformable<Sphere3Df> for Transform {
    fn transform(&self, sphere: &Sphere3Df) -> Sphere3Df {
        Sphere3Df {
            center: self.transform_vector(&sphere.center),
            radius: sphere.radius,
        }
    }
}

impl Transformable<Sphere3Df> for Mat4x4 {
    fn transform(&self, sphere: &Sphere3Df) -> Sphere3Df {
        Sphere3Df {
            center: self.transform_vector(&sphere.center),
            radius: sphere.radius,
        }
    }
}
