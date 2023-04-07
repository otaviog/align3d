use crate::{
    io::Geometry,
    transform::{Transform, Transformable},
};
use ndarray::{prelude::*, ArcArray2, Array2};

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
            colors: pcl.colors.map(|colors| colors.into_owned()),
            faces: None,
            texcoords: None,
        }
    }
}

#[cfg(with_rerun)]
impl PointCloud {
    pub fn rerun_msg(&self, name: &str) -> Result<rerun::MsgSender, rerun::MsgSenderError> {
        use rerun::external::glam;

        let mut points = Vec::with_capacity(self.len());
        let mut colors = Vec::with_capacity(self.len());
        for (point, color) in self
            .points
            .outer_iter()
            .zip(self.colors.iter().flat_map(|colors| colors.outer_iter()))
        {
            points.push(rerun::components::Point3D::from(glam::Vec3::new(
                point[0], point[1], point[2],
            )));

            colors.push(rerun::components::ColorRGBA::from_rgb(
                color[0], color[1], color[2],
            ));
        }

        Ok(rerun::MsgSender::new(name)
            .with_component(&points)?
            .with_component(&colors)?)
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
