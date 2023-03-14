use itertools::izip;

use nalgebra::{Cholesky, Vector6};
use ndarray::{Array1, Array2, Axis};
use nshare::ToNalgebra;

/// Implements the standard Gauss Newton optimization
pub struct GaussNewton {
    pub hessian: Array2<f32>,
    pub gradient: Array1<f32>,
}

impl GaussNewton {
    pub fn new() -> Self {
        Self {
            hessian: Array2::<f32>::zeros((6, 6)),
            gradient: Array1::<f32>::zeros(6),
        }
    }

    pub fn step(
        &mut self,
        residual_array: &Array1<f32>,
        jacobian_array: &Array2<f32>,
        weight: f32,
    ) -> f32 {
        let size = residual_array.shape()[0];
        // TODO: Don't zeroed every time
        let mut jt_r_array = Array2::<f32>::zeros((size, 6));
        let mut jt_j_array = Array2::<f32>::zeros((size, 6 * 6));

        let mut mean_residual = 0.0;
        for (residual, jacobian, mut dst_jt_r, mut dst_jtj) in izip!(
            residual_array.iter(),
            jacobian_array.axis_iter(Axis(0)),
            jt_r_array.axis_iter_mut(Axis(0)),
            jt_j_array.axis_iter_mut(Axis(0))
        ) {
            let residual = *residual;
            mean_residual += residual*residual;

            let jt_r = [
                jacobian[0] * residual,
                jacobian[1] * residual,
                jacobian[2] * residual,
                jacobian[3] * residual,
                jacobian[4] * residual,
                jacobian[5] * residual,
            ];

            let mut jt_j = [0.0; 36];
            for i in 0..6 {
                for j in 0..6 {
                    jt_j[i * 6 + j] += jacobian[i] * jacobian[j];
                }
            }

            // Todo push values
            dst_jt_r.assign(&Array1::<f32>::from_iter(jt_r.iter().cloned()));
            // dst_jt_r.assign(&jt_r);
            dst_jtj.assign(&Array1::<f32>::from_iter(jt_j.iter().cloned()));
        }

        let hessian = jt_j_array.sum_axis(Axis(0)).into_shape((6, 6)).unwrap();
        let gradient = jt_r_array.sum_axis(Axis(0));

        self.hessian += &(hessian * weight);
        self.gradient += &(gradient * weight);

        return mean_residual / size as f32;
    }

    pub fn solve(&self) -> Vector6<f32> {
        let hessian = self.hessian.clone().into_nalgebra();
        let gradient = self.gradient.clone().into_nalgebra();
        println!("Hessian: {:?}", hessian);
        println!("Gradient: {:?}", gradient);
        let v = Cholesky::new(hessian)
            .unwrap()
            .solve(&gradient);
        let update = Vector6::<f32>::new(v[0], v[1], v[2], v[3], v[4], v[5]);
        
        println!("Update: {:?}", update);
        update
    }

    pub fn reset(&mut self) {
        self.hessian = Array2::<f32>::zeros((6, 6));
        self.gradient = Array1::<f32>::zeros(6);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_gauss_newton() {
        use super::*;
        use nalgebra::{Vector3, Vector6};
        use ndarray::{array, s, Array1, Array2, Axis};
        use nshare::ToNalgebra;

        let mut gn = GaussNewton::new();

        let residual_array = array![1.0, 2.0, 3.0];
        let jacobian_array = array![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ];

        gn.step(&residual_array, &jacobian_array, 1.0);

        let hessian = gn.hessian.clone();
        let gradient = gn.gradient.clone();

        let expected_hessian = array![
            [3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
            [6.0, 12.0, 18.0, 24.0, 30.0, 36.0],
            [9.0, 18.0, 27.0, 36.0, 45.0, 54.0],
            [12.0, 24.0, 36.0, 48.0, 60.0, 72.0],
            [15.0, 30.0, 45.0, 60.0, 75.0, 90.0],
            [18.0, 36.0, 54.0, 72.0, 90.0, 108.0],
        ];

        let expected_gradient = array![6.0, 12.0, 18.0, 24.0, 30.0, 36.0];

        assert_eq!(hessian, expected_hessian);
        assert_eq!(gradient, expected_gradient);
    }
}