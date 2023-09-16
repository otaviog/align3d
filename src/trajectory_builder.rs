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
        Self {
            trajectory: Trajectory::default(),
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

    /// Returns the current camera pose in the world frame.
    /// If the trajectory is empty, it returns `None`.
    pub fn current_camera_to_world(&self) -> Option<Transform> {
        if self.trajectory.is_empty() {
            None
        } else {
            Some(self.trajectory.last().unwrap().0.clone())
        }
    }
}
