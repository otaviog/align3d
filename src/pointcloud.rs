use crate::{
    io::Geometry,
    transform::{Transform, Transformable},
};
use nalgebra::Vector3;
use ndarray::prelude::*;

pub struct PointCloud {
    pub points: Array1<Vector3<f32>>,
    pub normals: Option<Array1<Vector3<f32>>>,
    pub colors: Option<Array1<Vector3<u8>>>,
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
            points: Array1::zeros(len),
            normals: Some(Array1::zeros(len)),
            colors: Some(Array1::zeros(len)),
        }
    }

    pub fn len(&self) -> usize {
        self.points.len_of(Axis(0))
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

impl std::ops::Mul<&PointCloud> for &Transform {
    type Output = PointCloud;
    fn mul(self, rhs: &PointCloud) -> PointCloud {
        PointCloud {
            points: self.transform_vectors(rhs.points.clone()),
            normals: rhs
                .normals
                .as_ref()
                .map(|normals| self.transform_normals(normals.clone())),
            colors: rhs.colors.clone(),
        }
    }
}

impl Transformable<PointCloud> for Transform {
    fn transform(&self, pcl: &PointCloud) -> PointCloud {
        PointCloud {
            points: self.transform_vectors(pcl.points.clone()),
            normals: pcl
                .normals
                .as_ref()
                .map(|normals| self.transform_normals(normals.clone())),
            colors: pcl.colors.clone(),
        }
    }
}

impl From<PointCloud> for Geometry {
    fn from(pcl: PointCloud) -> Geometry {
        Geometry {
            points: pcl.points,
            normals: pcl.normals,
            colors: pcl.colors,
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
