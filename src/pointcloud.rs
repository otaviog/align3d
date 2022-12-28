use ndarray::prelude::*;
use ndarray::{ArcArray2, Array2};

use super::io::Geometry;

pub struct PointCloud {
    pub points: Array2<f32>,
    pub normals: Option<Array2<f32>>,
    pub colors: Option<ArcArray2<u8>>,
}

impl PointCloud {
    pub fn from_geometry(geometry: Geometry) -> Self {
        Self {
            points: geometry.points,
            normals: geometry.normals,
            colors: geometry.colors.map(|colors| colors.into()),
        }
    }

    pub fn zeros(len: usize) -> Self {
        Self {
            points: Array2::<f32>::zeros((len, 3)),
            normals: Some(Array2::<f32>::zeros((len, 3))),
            colors: Some(ArcArray2::<u8>::zeros((len, 3))),
        }
    }

    pub fn len(&self) -> usize {
        self.points.len_of(Axis(0))
    }
}

//impl<Idx> std::ops::Index<Idx> for PointCloud
//where Idx: std::slice::SliceIndex<[PointCloud]>
//{
//    fn index(&self, index: Idx) -> &Self::Output
//}

use crate::transform::Transform;

impl std::ops::Mul<&PointCloud> for &Transform {
    type Output = PointCloud;
    fn mul(self, rhs: &PointCloud) -> PointCloud {
        PointCloud {
            points: self * &rhs.points,
            normals: rhs
                .normals
                .as_ref()
                .map(|normals| &self.ortho_rotation() * &normals),
            colors: rhs.colors.clone(),
        }
    }
}

impl Into<Geometry> for PointCloud {
    fn into(self) -> Geometry {
        Geometry {
            points: self.points,
            normals: self.normals.map(|normals| normals),
            colors: self.colors.map(|colors| colors.into_owned()),
            faces: None,
            texcoords: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::io::read_off;
    use super::PointCloud;
    use rstest::*;
    #[fixture]
    fn sample_pcl1() -> PointCloud {
        PointCloud::from_geometry(read_off("tests/data/teapot.off").unwrap())
    }

    #[rstest]
    fn test_point_cloud_from_file(sample_pcl1: PointCloud) {
        assert_eq!(sample_pcl1.len(), 480);
    }
}