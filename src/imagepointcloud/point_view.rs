use nalgebra::Vector3;
use ndarray::iter::AxisIter;
use ndarray::{ArrayView2, Axis};
use std::iter::Enumerate;

use super::ImagePointCloud;

pub struct PointView<'a> {
    points: ArrayView2<'a, f32>,
    mask: ArrayView2<'a, u8>,
}

pub struct PointViewIterator<'a> {
    iter: Enumerate<
        std::iter::Zip<
            AxisIter<'a, f32, ndarray::Dim<[usize; 1]>>,
            AxisIter<'a, u8, ndarray::Dim<[usize; 1]>>,
        >,
    >,
}

impl<'a> PointView<'a> {
    pub fn iter(&'a self) -> PointViewIterator<'a> {
        PointViewIterator {
            iter: self
                .points
                .axis_iter(Axis(0))
                .zip(self.mask.axis_iter(Axis(0)))
                .enumerate(),
        }
    }
}

impl<'a> Iterator for PointViewIterator<'a> {
    type Item = (usize, Vector3<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (i, (v, m)) = self.iter.next()?;
            if m[0] > 0 {
                return Some((i, Vector3::new(v[0], v[1], v[2])));
            };
        }
    }
}

impl ImagePointCloud {
    pub fn point_view<'a>(&'a self) -> PointView<'a> {
        let total_points = self.len();
        let points = self.points.view().into_shape((total_points, 3)).unwrap();
        let mask = self.mask.view().into_shape((total_points, 1)).unwrap();
        PointView { points, mask }
    }
}
