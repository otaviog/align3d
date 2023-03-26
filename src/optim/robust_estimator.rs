pub trait RobustEstimator {
    fn estimate(&self, squared_residual: f32) -> f32;
    fn backward(&self, squared_residual: f32) -> f32;
}

pub struct HuberEstimator {
    pub delta: f32,
}

impl RobustEstimator for HuberEstimator {
    fn estimate(&self, squared_residual: f32) -> f32 {
        // if residual.abs() <= self.delta {
        //     residual * residual
        // } else {
        //     2.0 * self.delta * residual.abs() - self.delta * self.delta
        // }

        if squared_residual.abs() <= self.delta {
            return squared_residual;
        }

        2.0 * (squared_residual * self.delta).sqrt() - self.delta
    }

    fn backward(&self, residual: f32) -> f32 {
        1.0_f32.min(self.delta / residual.abs()).sqrt()
    }
}
