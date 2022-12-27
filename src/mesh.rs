use ndarray::{Array1, Array2, Axis};

pub fn compute_normals(points: &Array2<f32>, faces: &Array2<usize>) -> Array2<f32> {
    let face_normals = faces
        .axis_iter(Axis(0))
        //.into_par_iter()
        .map(|face| {
            let p0 = points.row(face[0]);
            let p1 = points.row(face[1]);
            let p2 = points.row(face[2]);
            let v0 = &p1 - &p0;
            let v1 = &p2 - &p0;

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

    let mut vertex_normals = Array2::<f32>::zeros(points.raw_dim());
    let mut vertex_face_count = Array1::<usize>::zeros(points.len_of(Axis(0)));
    faces
        .axis_iter(Axis(0))
        .zip(face_normals)
        .for_each(|(face, face_normal)| {
            for f in [face[0], face[1], face[2]] {
                let face_array = [face_normal[0], face_normal[1], face_normal[2]];
                let face_normal = Array1::from_iter(face_array.iter());
                let normal_sum = &vertex_normals.row(f) + &face_normal;
                vertex_normals.row_mut(f).assign(&normal_sum);
                vertex_face_count[f] += 1;
            }
        });

    vertex_normals
        .axis_iter_mut(Axis(0))
        .zip(vertex_face_count.iter())
        .for_each(|(mut normal, face_count)| {
            normal /= *face_count as f32;
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
        let normals = compute_normals(&geometry.points, &geometry.faces.unwrap());
        assert!(normals.len() == geometry.points.len());
    }
}
