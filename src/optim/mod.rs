mod gaussnewton;
pub use gaussnewton::{GaussNewton, GaussNewtonBatch};

mod robust_estimator;
pub use robust_estimator::{HuberEstimator, RobustEstimator};
