use nalgebra::{Cholesky, Const, SMatrix, SVector};
use num::Zero;

/// Implements the standard Gauss Newton optimization
///
/// # Type parameters
///
/// * `DIM` - The dimension of the problem.
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
    /// Creates a new Gauss Newton optimizer.
    pub fn new() -> Self {
        Self {
            hessian: SMatrix::zeros(),
            gradient: SVector::zeros(),
            squared_residual_sum: 0.0,
            count: 0,
        }
    }

    /// Resets the optimizer.
    pub fn reset(&mut self) {
        self.hessian.set_zero();
        self.gradient.set_zero();
        self.squared_residual_sum = 0.0;
        self.count = 0;
    }

    /// Adds a new step to the optimizer.
    ///
    /// # Arguments
    ///
    /// * `residual` - The residual of the step.
    /// * `jacobian` - The jacobian of the step.
    pub fn step(&mut self, residual: f32, jacobian: &[f32; DIM]) {
        let mut jt_j = [[0.0; DIM]; DIM];
        for i in 0..DIM {
            let ival = jacobian[i];
            self.gradient[i] += ival * residual;

            jt_j[i][i] = ival * ival;
            for j in i + 1..DIM {
                let jval = jacobian[j];
                let mul = ival * jval;
                jt_j[i][j] = mul;
                jt_j[j][i] = mul;
            }
        }

        // self.hessian += SMatrix::from_data(ArrayStorage(jt_j));
        for (i, row) in jt_j.iter().enumerate().take(DIM) {
            for (j, value) in row.iter().enumerate().take(DIM) {
                self.hessian[(i, j)] += value;
            }
        }

        self.squared_residual_sum += residual * residual;

        // Improved test results.
        // for i in 0..DIM {
        //     self.gradient[i] += jacobian[i]*residual;
        // }

        self.count += 1;
    }

    /// Solve the current gauss newton system.
    ///
    /// # Returns
    ///
    /// The update vector.
    pub fn solve(&self) -> Option<SVector<f32, DIM>> {
        if self.count == 0 {
            return None;
        }
        let hessian: SMatrix<f64, DIM, DIM> = nalgebra::convert(self.hessian);
        let gradient: SVector<f64, DIM> = nalgebra::convert(self.gradient);

        Cholesky::<f64, Const<DIM>>::new(hessian)
            .map(|cholesky| nalgebra::convert(cholesky.solve(&gradient)))
    }

    /// Adds the values of another optimizer to this one.
    /// Use this to combine the state of sub optimizers.
    ///
    /// # Arguments
    ///
    /// * `other` - The other optimizer.
    pub fn add(&mut self, other: &Self) {
        self.hessian += other.hessian;
        self.gradient += other.gradient;
        self.squared_residual_sum += other.squared_residual_sum;
        self.count += other.count;
    }

    /// Adds the values of another optimizer to this one.
    /// Use this to combine the state of sub optimizers.
    /// # Arguments
    ///
    /// * `other` - The other optimizer.
    /// * `weight1` - The weight of this optimizer.
    /// * `weight2` - The weight of the other optimizer.
    pub fn add_weighted(&mut self, other: &Self, weight1: f32, weight2: f32) {
        self.hessian = self.hessian * (weight1 * weight1) + other.hessian * (weight2 * weight2);
        self.gradient = self.gradient * weight1 + other.gradient * weight2;
        self.squared_residual_sum =
            self.squared_residual_sum * weight1 + other.squared_residual_sum * weight2;
        self.count += other.count;
    }

    /// Weights the optimizer.
    pub fn weight(&mut self, weight: f32) {
        self.hessian *= weight * weight;
        self.gradient *= weight;
        self.squared_residual_sum *= weight;
    }

    /// Returns the mean squared residual.
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

        gn.step(1.0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        gn.step(2.0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        gn.step(3.0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

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
