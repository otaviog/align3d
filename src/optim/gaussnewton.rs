use itertools::izip;

use nalgebra::{ArrayStorage, Cholesky, SMatrix, SVector, Const};
use ndarray::{Array1, Array2, Axis};
use num::Zero;

pub struct GaussNewtonBatch {
    jacobians: Array2<f32>,
    residuals: Array1<f32>,
    costs: Array1<f32>,
    dirty: Array1<bool>,
}

impl GaussNewtonBatch {
    pub fn new(max_entries: usize) -> Self {
        Self {
            jacobians: Array2::zeros((max_entries, 6)),
            residuals: Array1::zeros(max_entries),
            costs: Array1::zeros(max_entries),
            dirty: Array1::from_elem(max_entries, true),
        }
    }

    pub fn assign(&mut self, i: usize, cost: f32, residual: f32, jacobian: &[f32]) {
        if !self.dirty[i] {
            if self.costs[i] < cost {
                return;
            }
        }

        self.jacobians
            .row_mut(i)
            .assign(&Array1::from_vec(jacobian.to_vec()));
        self.residuals[i] = residual;
        self.dirty[i] = false;
        self.costs[i] = cost;
    }

    pub fn reset(&mut self) {
        self.dirty.fill(true);
    }
}

/// Implements the standard Gauss Newton optimization
pub struct GaussNewton<const DIM: usize> {
    hessian: SMatrix<f32, DIM, DIM>,
    gradient: SVector<f32, DIM>,
    squared_residual_sum: f32,
    count: usize,
}

impl<const DIM: usize> GaussNewton<DIM> {
    pub fn new() -> Self {
        Self {
            hessian: SMatrix::zeros(),
            gradient: SVector::zeros(),
            squared_residual_sum: 0.0,
            count: 0,
        }
    }

    pub fn reset(&mut self) {
        self.hessian.set_zero();
        self.gradient.set_zero();
        self.squared_residual_sum = 0.0;
        self.count = 0;
    }

    pub fn step(&mut self, residual: f32, jacobian: &[f32]) {
        self.squared_residual_sum += residual * residual;

        let jt_r = SMatrix::from_row_slice(jacobian) * residual;

        let mut jt_j = [[0.0; DIM]; DIM];
        for i in 0..DIM {
            for j in 0..DIM {
                jt_j[i][j] += jacobian[i] * jacobian[j];
            }
        }

        self.hessian += SMatrix::from_data(ArrayStorage(jt_j));
        self.gradient += jt_r;
        self.count += 1;
    }

    pub fn step_array(&mut self, residual_array: &Array1<f32>, jacobian_array: &Array2<f32>) {
        for (residual, jacobian) in izip!(residual_array.iter(), jacobian_array.axis_iter(Axis(0)))
        {
            let residual = *residual;
            self.step(residual, jacobian.as_slice().unwrap());
        }
    }

    pub fn step_batch(&mut self, batch: &GaussNewtonBatch) {
        for (dirty, residual, jacobian) in izip!(
            batch.dirty.iter(),
            batch.residuals.iter(),
            batch.jacobians.axis_iter(Axis(0))
        ) {
            if !*dirty {
                let residual = *residual;
                self.step(residual, jacobian.as_slice().unwrap());
            }
        }
    }

    pub fn solve(&self) -> Option<SVector<f32, DIM>> {
        if self.count == 0 {
            return None;
        }
        let hessian: SMatrix<f64, DIM, DIM> = nalgebra::convert(self.hessian);
        let gradient: SVector<f64, DIM> = nalgebra::convert(self.gradient);
        
        let update = Cholesky::<f64, Const<DIM>>::new(hessian)
                .unwrap()
                .solve(&gradient);
        Some(nalgebra::convert(update))
    }

    pub fn combine(&mut self, other: &Self, weight1: f32, weight2: f32) {
        self.hessian = self.hessian * (weight1 * weight1) + &other.hessian * (weight2 * weight2);
        self.gradient = self.gradient * weight1 + &other.gradient * weight2;
        self.squared_residual_sum =
            self.squared_residual_sum * weight1 + other.squared_residual_sum * weight2;
        self.count += other.count;
    }

    pub fn weight(&mut self, weight: f32) {
        self.hessian = self.hessian * (weight * weight);
        self.gradient = self.gradient * weight;
        self.squared_residual_sum = self.squared_residual_sum * weight;
    }

    pub fn mean_squared_residual(&self) -> f32 {
        self.squared_residual_sum / self.count as f32
    }
}

#[cfg(test)]
mod tests {
    use nshare::ToNalgebra;

    #[test]
    fn test_gauss_newton() {
        use super::*;
        use ndarray::array;

        let mut gn = GaussNewton::<6>::new();

        let residual_array = array![1.0, 2.0, 3.0];
        let jacobian_array = array![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ];

        gn.step_array(&residual_array, &jacobian_array);

        let hessian = gn.hessian;
        let gradient = gn.gradient;

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
