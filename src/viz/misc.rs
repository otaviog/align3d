use vulkano::padded::Padded;

pub fn get_normal_matrix(view_matrix: &nalgebra_glm::Mat4) -> [Padded::<[f32; 3], 4>; 3] {
    let normal_matrix = view_matrix.try_inverse().unwrap().transpose();
    let normal_matrix = normal_matrix.fixed_slice::<3, 4>(0, 0);
    let mut r = [Padded::<[f32; 3], 4>::from([0.0; 3]); 3];

    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = normal_matrix[(i, j)];
        }
    }

    r
}