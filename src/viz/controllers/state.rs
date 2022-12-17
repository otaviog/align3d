use std::collections::HashMap;

use winit::event::{ElementState, MouseButton, VirtualKeyCode};

use crate::bounds::Sphere3Df;

pub struct WindowState {
    pub window_size: [f32; 2],
    pub keyboard_state: HashMap<VirtualKeyCode, ElementState>,
    pub mouse_state: HashMap<MouseButton, ElementState>,
    pub elapsed_time: f32,
}

impl WindowState {
    pub fn new() -> Self {
        Self {
            window_size: [0.0, 0.0],
            keyboard_state: HashMap::new(),
            mouse_state: HashMap::new(),
            elapsed_time: f32::INFINITY,
        }
    }
}
pub struct SceneState {
    pub world_bounds: Sphere3Df,
}

impl SceneState {
    pub fn new() -> Self {
        Self {
            world_bounds: Sphere3Df::empty(),
        }
    }
}