use itertools::izip;

use nalgebra::{Cholesky, Matrix6, Vector6};
use ndarray::{Array1, Array2, Axis};
use num::Zero;

/// Implements the standard Gauss Newton optimization
pub struct GaussNewton {
    pub hessian: Matrix6<f32>,
    pub gradient: Vector6<f32>,
    mean_squared_residual: f32,
}

impl GaussNewton {
    pub fn new() -> Self {
        Self {
            hessian: Matrix6::zeros(),
            gradient: Vector6::zeros(),
            mean_squared_residual: 0.0,
        }
    }

    pub fn step(&mut self, residual: f32, jacobian: &[f32]) {
        self.mean_squared_residual += residual * residual;

        let jt_r = [[
            jacobian[0] * residual,
            jacobian[1] * residual,
            jacobian[2] * residual,
            jacobian[3] * residual,
            jacobian[4] * residual,
            jacobian[5] * residual,
        ]];

        let mut jt_j = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                jt_j[i][j] += jacobian[i] * jacobian[j];
            }
        }

        self.hessian += Matrix6::from_data(nalgebra::ArrayStorage(jt_j));
        self.gradient += Vector6::from_data(nalgebra::ArrayStorage(jt_r));
    }

    pub fn step_batch(
        &mut self,
        residual_array: &Array1<f32>,
        jacobian_array: &Array2<f32>,
        weight: f32,
    ) -> f32 {
        let size = residual_array.shape()[0];

        let mut mean_residual = 0.0;
        for (residual, jacobian) in izip!(residual_array.iter(), jacobian_array.axis_iter(Axis(0)))
        {
            let residual = *residual;
            mean_residual += residual * residual;

            self.step(residual, jacobian.as_slice().unwrap());
        }

        return mean_residual / size as f32;
    }

    pub fn solve(&self) -> Vector6<f32> {
        let v = Cholesky::new(self.hessian).unwrap().solve(&self.gradient);
        let update = Vector6::<f32>::new(v[0], v[1], v[2], v[3], v[4], v[5]);
        update
    }

    pub fn reset(&mut self) {
        self.hessian.set_zero();
        self.gradient.set_zero();
    }
}

#[cfg(test)]
mod tests {
    use nshare::ToNalgebra;

    #[test]
    fn test_gauss_newton() {
        use super::*;
        use ndarray::array;

        let mut gn = GaussNewton::new();

        let residual_array = array![1.0, 2.0, 3.0];
        let jacobian_array = array![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ];

        gn.step_batch(&residual_array, &jacobian_array, 1.0);

        let hessian = gn.hessian.clone();
        let gradient = gn.gradient.clone();

        let expected_hessian = array![
            [3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
            [6.0, 12.0, 18.0, 24.0, 30.0, 36.0],
            [9.0, 18.0, 27.0, 36.0, 45.0, 54.0],
            [12.0, 24.0, 36.0, 48.0, 60.0, 72.0],
            [15.0, 30.0, 45.0, 60.0, 75.0, 90.0],
            [18.0, 36.0, 54.0, 72.0, 90.0, 108.0],
        ]
        .into_nalgebra();

        let expected_gradient = array![6.0, 12.0, 18.0, 24.0, 30.0, 36.0].into_nalgebra();

        assert_eq!(hessian, expected_hessian);
        assert_eq!(gradient, expected_gradient);
    }
}
