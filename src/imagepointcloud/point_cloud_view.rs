use itertools::izip;
use nalgebra::Vector3;
use ndarray::iter::AxisIter;
use ndarray::{ArrayView2, Axis};
use std::iter::{Enumerate, Zip};

use super::ImagePointCloud;

pub struct PointCloudView<'a> {
    points: ArrayView2<'a, f32>,
    normals: ArrayView2<'a, f32>,
    mask: ArrayView2<'a, u8>,
}

pub struct PointCloudViewIterator<'a> {
    iter: Enumerate<
        Zip<
            Zip<
                AxisIter<'a, f32, ndarray::Dim<[usize; 1]>>,
                AxisIter<'a, f32, ndarray::Dim<[usize; 1]>>,
            >,
            AxisIter<'a, u8, ndarray::Dim<[usize; 1]>>,
        >,
    >,
}

impl<'a> PointCloudView<'a> {
    pub fn iter(&'a self) -> PointCloudViewIterator<'a> {
        PointCloudViewIterator {
            iter: self
                .points
                .axis_iter(Axis(0))
                .zip(self.normals.axis_iter(Axis(0)))
                .zip(self.mask.axis_iter(Axis(0)))
                .enumerate(),
        }
    }
}

impl<'a> Iterator for PointCloudViewIterator<'a> {
    type Item = (usize, Vector3<f32>, Vector3<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (i, ((v, n), m)) = self.iter.next()?;
            if m[0] > 0 {
                return Some((
                    i,
                    Vector3::new(v[0], v[1], v[2]),
                    Vector3::new(n[0], n[1], n[2]),
                ));
            };
        }
    }
}

impl ImagePointCloud {
    pub fn point_cloud_view<'a>(&'a self) -> PointCloudView<'a> {
        let total_points = self.len();
        let points = self.points.view().into_shape((total_points, 3)).unwrap();
        let normals = self
            .normals
            .as_ref()
            .unwrap()
            .view()
            .into_shape((total_points, 3))
            .unwrap();
        let mask = self.mask.view().into_shape((total_points, 1)).unwrap();
        PointCloudView {
            points,
            normals,
            mask,
        }
    }
}
