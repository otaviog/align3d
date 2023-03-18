use crate::transform::Transform;

#[derive(Clone, Debug)]
pub struct Trajectory {
    pub camera_to_world: Vec<Transform>,
    pub times: Vec<f32>,
}

impl Trajectory {
    pub fn new() -> Self {
        Self {
            camera_to_world: Vec::new(),
            times: Vec::new(),
        }
    }

    pub fn push(&mut self, camera_to_world: Transform, time: f32) {
        self.camera_to_world.push(camera_to_world);
        self.times.push(time);
    }

    pub fn len(&self) -> usize {
        self.camera_to_world.len()
    }

    pub fn is_empty(&self) -> bool {
        self.camera_to_world.is_empty()
    }

    pub fn get_relative_transform(&self, from_time: f32, to_time: f32) -> Option<Transform> {
        let (i_src, i_dst) = self.get_indices(from_time, to_time)?;
        Some(&self.camera_to_world[i_dst].inverse() * &self.camera_to_world[i_src])
    }

    pub fn get_indices(&self, time_src: f32, time_dst: f32) -> Option<(usize, usize)> {
        let i_src = self.times.iter().position(|t| *t == time_src)?;
        let i_dst = self.times.iter().position(|t| *t == time_dst)?;
        Some((i_src, i_dst))
    }

    pub fn iter(&self) -> impl Iterator<Item = (Transform, f32)> + '_ {
        self.camera_to_world
            .iter()
            .zip(self.times.iter())
            .map(|(camera_to_world, time)| (camera_to_world.clone(), *time))
    }
}
