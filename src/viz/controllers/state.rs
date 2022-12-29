use std::{collections::HashMap, time::Duration};

use winit::event::{ElementState, MouseButton, VirtualKeyCode};

use crate::bounds::Sphere3Df;

pub struct FrameStepInfo {
    pub viewport_size: [f32; 2],
    pub keyboard_state: HashMap<VirtualKeyCode, ElementState>,
    pub mouse_state: HashMap<MouseButton, ElementState>,
    pub elapsed_time: Duration,
}

impl FrameStepInfo {
    pub fn new(viewport_size: [f32; 2]) -> Self {
        Self {
            viewport_size,
            ..Default::default()
        }
    }
}

impl Default for FrameStepInfo {
    fn default() -> Self {
        Self {
            viewport_size: [0.0, 0.0],
            keyboard_state: HashMap::new(),
            mouse_state: HashMap::new(),
            elapsed_time: Duration::from_millis(24),
        }
    }
}

pub struct SceneState {
    pub world_bounds: Sphere3Df,
}

impl Default for SceneState {
    fn default() -> Self {
        Self {
            world_bounds: Sphere3Df::empty(),
        }
    }
}
