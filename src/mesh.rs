use nalgebra::Vector3;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

pub fn compute_normals(
    points: &ArrayView1<Vector3<f32>>,
    faces: &ArrayView2<usize>,
) -> Array1<Vector3<f32>> {
    let face_normals = faces
        .axis_iter(Axis(0))
        .map(|face| {
            let p0 = points[face[0]];
            let p1 = points[face[1]];
            let p2 = points[face[2]];
            let v0 = p1 - p0;
            let v1 = p2 - p0;

            let v0 = nalgebra::Vector3::new(v0[0], v0[1], v0[2]);
            let v1 = nalgebra::Vector3::new(v1[0], v1[1], v1[2]);

            let mut normal = v0.cross(&v1);
            let mag = normal.magnitude();
            if mag > 0.0 {
                normal /= mag;
            }

            normal
        })
        .collect::<Vec<_>>();

    let mut vertex_normals = Array1::<Vector3<f32>>::zeros(points.len());
    let mut vertex_face_count = Array1::<usize>::zeros(points.len());
    faces
        .axis_iter(Axis(0))
        .zip(face_normals)
        .for_each(|(face, face_normal)| {
            for f in [face[0], face[1], face[2]] {
                let normal_sum = vertex_normals[f] + face_normal;
                vertex_normals[f] = normal_sum;
                vertex_face_count[f] += 1;
            }
        });

    vertex_normals
        .iter_mut()
        .zip(vertex_face_count.iter())
        .for_each(|(normal, face_count)| {
            *normal /= *face_count as f32;
        });

    vertex_normals
}

#[cfg(test)]
mod tests {
    use crate::io::read_off;

    use super::compute_normals;

    #[test]
    fn test_compute_normals() {
        let geometry = read_off("tests/data/teapot.off").unwrap();
        let normals = compute_normals(&geometry.points.view(), &geometry.faces.unwrap().view());
        assert!(normals.len() == geometry.points.len());
    }
}
