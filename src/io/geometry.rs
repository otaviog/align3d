use nalgebra::{Vector2, Vector3};
use ndarray::prelude::*;

/// Generic representation of attributes found in 3D model/object/geometry files.
pub struct Geometry {
    /// The 3D points. Shape is (Nx3).
    pub points: Array1<Vector3<f32>>,
    /// The RGB colors. Shape is (Nx3).
    pub colors: Option<Array1<Vector3<u8>>>,
    /// Per vertices normals. Shape is (Nx3)
    pub normals: Option<Array1<Vector3<f32>>>,
    /// The indices to connect vertices that make faces in the geometry.
    /// Shape is (Nx3), we always convert to triangles.
    pub faces: Option<Array2<usize>>,
    /// The texture coordinates.
    pub texcoords: Option<Array1<Vector2<f32>>>,
}

impl Geometry {
    /// Number of vertices.
    pub fn len_vertices(&self) -> usize {
        self.points.len()
    }

    /// Number of faces. Zero if `faces` is None.
    pub fn len_faces(&self) -> usize {
        self.faces.as_ref().map_or(0, |faces| faces.nrows())
    }
}

pub struct GeometryBuilder {
    geometry: Geometry,
}

impl GeometryBuilder {
    pub fn new(points: Array1<Vector3<f32>>) -> Self {
        Self {
            geometry: Geometry {
                points,
                colors: None,
                normals: None,
                faces: None,
                texcoords: None,
            },
        }
    }

    pub fn with_points(mut self, points: Array1<Vector3<f32>>) -> Self {
        self.geometry.points = points;
        self
    }

    pub fn with_colors(mut self, colors: Array1<Vector3<u8>>) -> Self {
        self.geometry.colors = Some(colors);
        self
    }

    pub fn with_normals(mut self, normals: Array1<Vector3<f32>>) -> Self {
        self.geometry.normals = Some(normals);
        self
    }

    pub fn with_faces(mut self, faces: Array2<usize>) -> Self {
        self.geometry.faces = Some(faces);
        self
    }

    pub fn with_texcoords(mut self, texcoords: Array1<Vector2<f32>>) -> Self {
        self.geometry.texcoords = Some(texcoords);
        self
    }

    pub fn build(self) -> Geometry {
        self.geometry
    }
}
