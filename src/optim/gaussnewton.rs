use itertools::izip;

use nalgebra::{ArrayStorage, Cholesky, Const, SMatrix, SVector};
use num::Zero;

#[derive(Clone)]
pub struct GaussNewtonBatch<const BATCH_SIZE: usize, const JACOBIAN_DIM: usize> {
    jacobians: [[f32; JACOBIAN_DIM]; BATCH_SIZE],
    residuals: [f32; BATCH_SIZE],
    costs: [f32; BATCH_SIZE],
    dirty: [bool; BATCH_SIZE],
}

impl<const BATCH_SIZE: usize, const JACOBIAN_DIM: usize>
    GaussNewtonBatch<BATCH_SIZE, JACOBIAN_DIM>
{
    pub fn new() -> Self {
        Self {
            jacobians: [[0.0f32; JACOBIAN_DIM]; BATCH_SIZE],
            residuals: [0.0f32; BATCH_SIZE],
            costs: [0.0f32; BATCH_SIZE],
            dirty: [true; BATCH_SIZE],
        }
    }

    pub fn assign(&mut self, i: usize, cost: f32, residual: f32, jacobian: &[f32]) {
        if !self.dirty[i] && self.costs[i] < cost {
            return;
        }

        for j in 0..JACOBIAN_DIM {
            self.jacobians[i][j] = jacobian[j];
        }
        self.residuals[i] = residual;
        self.dirty[i] = false;
        self.costs[i] = cost;
    }

    pub fn clear(&mut self) {
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

impl<const DIM: usize> Default for GaussNewton<DIM> {
    fn default() -> Self {
        Self::new()
    }
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

    pub fn step_batch<const BATCH_SIZE: usize>(
        &mut self,
        batch: &GaussNewtonBatch<BATCH_SIZE, DIM>,
    ) {
        for (dirty, residual, jacobian) in
            izip!(batch.dirty.iter(), batch.residuals.iter(), batch.jacobians)
        {
            if !*dirty {
                self.step(*residual, &jacobian);
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
        self.hessian = self.hessian * (weight1 * weight1) + other.hessian * (weight2 * weight2);
        self.gradient = self.gradient * weight1 + other.gradient * weight2;
        self.squared_residual_sum =
            self.squared_residual_sum * weight1 + other.squared_residual_sum * weight2;
        self.count += other.count;
    }

    pub fn weight(&mut self, weight: f32) {
        self.hessian *= weight * weight;
        self.gradient *= weight;
        self.squared_residual_sum *= weight;
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

        let mut batch = GaussNewtonBatch::<3, 6>::new();
        batch.assign(0, 1.0, 1.0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        batch.assign(1, 2.0, 2.0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        batch.assign(2, 3.0, 3.0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        gn.step_batch(&batch);

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
        assert_eq!(hessian, expected_hessian);

        let expected_gradient = array![6.0, 12.0, 18.0, 24.0, 30.0, 36.0].into_nalgebra();
        assert_eq!(gradient, expected_gradient);
    }
}
