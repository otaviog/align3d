use nalgebra::Vector3;
use ndarray::{iter::Iter, ArrayView2};
use std::iter::Zip;

use super::RangeImage;

pub struct PointCloudView<'a> {
    points: ArrayView2<'a, Vector3<f32>>,
    normals: ArrayView2<'a, Vector3<f32>>,
    mask: ArrayView2<'a, u8>,
}

type NdArrayIter2<'a, T> = Iter<'a, T, ndarray::Dim<[usize; 2]>>;

pub struct PointCloudViewIterator<'a> {
    iter: Zip<
        Zip<NdArrayIter2<'a, Vector3<f32>>, NdArrayIter2<'a, Vector3<f32>>>,
        NdArrayIter2<'a, u8>,
    >,
    linear_index: usize,
}

impl<'a> PointCloudView<'a> {
    pub fn iter(&'a self) -> PointCloudViewIterator<'a> {
        PointCloudViewIterator {
            iter: self
                .points
                .iter()
                .zip(self.normals.iter())
                .zip(self.mask.iter()),
            linear_index: 0,
        }
    }
}

impl<'a> Iterator for PointCloudViewIterator<'a> {
    type Item = (usize, Vector3<f32>, Vector3<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let ((point, normal), mask) = self.iter.next()?;
            let linear_index = self.linear_index;
            self.linear_index += 1;
            if *mask > 0 {
                return Some((linear_index, *point, *normal));
            };
        }
    }
}

impl RangeImage {
    pub fn point_cloud_view(&'_ self) -> PointCloudView<'_> {
        PointCloudView {
            points: self.points.view(),
            normals: self.normals.as_ref().unwrap().view(),
            mask: self.mask.view(),
        }
    }
}
