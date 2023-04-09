use crate::{trajectory::Trajectory, transform::Transform};

/// Accumulates transforms and builds a trajectory.
#[derive(Clone, Debug)]
pub struct TrajectoryBuilder {
    trajectory: Trajectory,
    last: Transform,
    last_time: f32,
}

impl Default for TrajectoryBuilder {
    /// Creates a new `TrajectoryBuilder`.
    /// It'll contain a single pose at the origin.
    fn default() -> Self {
        let mut trajectory = Trajectory::default();
        trajectory.push(Transform::eye(), 0.0);
        Self {
            trajectory,
            last: Transform::eye(),
            last_time: 0.0,
        }
    }
}

impl TrajectoryBuilder {
    /// Accumulates the given transform and timestamp into the previous ones and adds
    /// it to the trajectory being build.
    pub fn accumulate(&mut self, now_to_previous: &Transform, timestamp: Option<f32>) {
        self.last = now_to_previous * &self.last;
        self.last_time = timestamp.unwrap_or(self.last_time + 1.0);
        self.trajectory.push(self.last.clone(), self.last_time);
    }

    /// Creates the trajectory at its current state.
    pub fn build(self) -> Trajectory {
        self.trajectory
    }
}
