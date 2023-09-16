use std::ops::Index;

use crate::transform::Transform;

/// Trajectory of camera poses. Use it to store or create trajectories while aligning scans.
#[derive(Clone, Debug)]
pub struct Trajectory {
    /// Camera poses, transforms points from camera to world.
    pub camera_to_world: Vec<Transform>,
    /// Timestamps of each pose.
    pub times: Vec<f32>,
}

impl Default for Trajectory {
    /// Empty trajectory.
    fn default() -> Self {
        Self {
            camera_to_world: Vec::new(),
            times: Vec::new(),
        }
    }
}

impl Trajectory {
    /// Adds a new pose to the trajectory.
    ///
    /// # Arguments
    ///
    /// * `camera_to_world` - Transform from camera to world.
    /// * `time` - Timestamp of the pose.
    pub fn push(&mut self, camera_to_world: Transform, time: f32) {
        self.camera_to_world.push(camera_to_world);
        self.times.push(time);
    }

    /// Returns the number of poses in the trajectory.
    pub fn len(&self) -> usize {
        self.camera_to_world.len()
    }

    /// Returns true if the trajectory is empty.
    pub fn is_empty(&self) -> bool {
        self.camera_to_world.is_empty()
    }

    /// Returns the relative transform between two poses.
    pub fn get_relative_transform(
        &self,
        from_index: usize,
        dest_index: usize,
    ) -> Option<Transform> {
        Some(&self.camera_to_world[dest_index].inverse() * &self.camera_to_world[from_index])
    }

    /// Returns the iterator over poses and timestamps.
    pub fn iter(&self) -> impl Iterator<Item = (Transform, f32)> + '_ {
        self.camera_to_world
            .iter()
            .zip(self.times.iter())
            .map(|(camera_to_world, time)| (camera_to_world.clone(), *time))
    }

    /// Creates a new trajectory with the poses transformed in such a way that the first pose is at origin.
    pub fn first_frame_at_origin(&self) -> Self {
        if self.camera_to_world.is_empty() {
            return self.clone();
        }

        let first_inv = self.camera_to_world[0].inverse();
        Self {
            camera_to_world: self
                .camera_to_world
                .iter()
                .map(|transform| &first_inv * transform)
                .collect::<Vec<Transform>>(),
            times: self.times.clone(),
        }
    }

    /// Creates a new trajectory with the given range.
    ///
    /// # Arguments
    ///
    /// * `start` - Inclusive start index of the range.
    /// * `end` - Exclusive end index of the range.
    ///
    /// # Returns
    ///
    /// New trajectory with the poses in the given range.
    pub fn slice(&self, start: usize, end: usize) -> Self {
        Self {
            camera_to_world: self.camera_to_world[start..end].to_vec(),
            times: self.times[start..end].to_vec(),
        }
    }

    /// Gets the last pose and timestamp.
    /// If the trajectory is empty, it returns `None`.
    pub fn last(&self) -> Option<(Transform, f32)> {
        if self.is_empty() {
            None
        } else {
            Some((
                self.camera_to_world[self.len() - 1].clone(),
                self.times[self.len() - 1],
            ))
        }
    }
}

impl FromIterator<(Transform, f32)> for Trajectory {
    /// Creates a new trajectory from the `(Transform, f32)` iterator.
    /// Use with the `collect::<Trajectory>` method.
    fn from_iter<T: IntoIterator<Item = (Transform, f32)>>(iter: T) -> Self {
        let mut trajectory = Trajectory::default();
        for (transform, time) in iter {
            trajectory.push(transform, time);
        }
        trajectory
    }
}

impl Index<usize> for Trajectory {
    type Output = Transform;
    /// Returns the pose at the given index.
    fn index(&self, index: usize) -> &Self::Output {
        &self.camera_to_world[index]
    }
}
