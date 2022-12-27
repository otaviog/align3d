use ndarray::prelude::*;

/// Generic representation of attributes found in 3D model/object/geometry files.
pub struct Geometry {
    /// The 3D points. Shape is (Nx3).
    pub points: Array2<f32>,
    /// The RGB colors. Shape is (Nx3).
    pub colors: Option<Array2<u8>>,
    /// Per vertices normals. Shape is (Nx3)
    pub normals: Option<Array2<f32>>,
    /// The indices to conect vertices that make faces in the geometry.
    /// Shape is (Nx3), we always convert to triangles.
    pub faces: Option<Array2<usize>>,
    /// The texture coordinates.
    pub texcoords: Option<Array2<f32>>,
}

impl Geometry {
    
    /// Number of vertices.
    pub fn len_vertices(&self) -> usize {
        self.points.nrows()
    }

    /// Number of faces. Zero if `faces` is None.
    pub fn len_faces(&self) -> usize {
        self.faces.as_ref().map_or(0, |faces| faces.nrows())
    }
}
