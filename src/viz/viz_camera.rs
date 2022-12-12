#![feature(result_option_inspect)]

use std::collections::HashMap;

use nalgebra_glm::{Vec2, Vec3};
use winit::event::{ElementState, MouseButton, ScanCode, VirtualKeyCode, WindowEvent};

use crate::{
    bounds::{Box3Df, Sphere3Df},
    camera::Camera,
};

pub struct VizCamera {
    pub eye: Vec3,
    pub view: Vec3,
    pub up: Vec3,
}

impl VizCamera {
    pub fn from_sphere_coordinates(center: Vec3, elevation: f32, azimuth: f32, distance: f32) {}
    pub fn right_vector(&self) -> Vec3 {
        self.view.cross(&self.up).normalize()
    }

    pub fn rotate_right_axis(&mut self, rad_angle: f32) {
        let right_vec = self.right_vector();
        self.view = nalgebra_glm::quat_rotate_vec3(
            &nalgebra_glm::quat_angle_axis(rad_angle, &right_vec),
            &self.view,
        )
        .normalize();

        let right_vec = self.view.cross(&self.up);
        self.up = right_vec.cross(&self.view).normalize();
    }

    pub fn rotate_up_axis(&mut self, rad_angle: f32) {
        self.view = nalgebra_glm::quat_rotate_vec3(
            &nalgebra_glm::quat_angle_axis(rad_angle, &self.up),
            &self.view,
        )
        .normalize();
    }

    pub fn translate_eye(&mut self, amount: f32) {
        self.eye += self.view * amount;
    }

    pub fn translate_right(&mut self, amount: f32) {
        self.eye += self.right_vector() * amount;
    }

    pub fn matrix(&self) -> nalgebra_glm::Mat4 {
        nalgebra_glm::look_at(&self.eye, &(self.eye + self.view), &self.up)
    }
}

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

pub trait VizCameraControl {
    fn key_event(&mut self, window_state: &WindowState, scene_bounds: &SceneState);
    fn cursor_moved(
        &mut self,
        x: f64,
        y: f64,
        window_state: &WindowState,
        scene_bounds: &SceneState,
    );
}

pub struct WSDVizCameraControl {
    pub viz_camera: VizCamera,
    pub velocity: f32,
    pub rotation_sensitivity: Vec2,
    cursor_last_position: Vec2,
}

impl WSDVizCameraControl {}

impl Default for WSDVizCameraControl {
    fn default() -> Self {
        Self {
            viz_camera: VizCamera {
                up: Vec3::new(0.0, 1.0, 0.0),
                view: Vec3::new(0.0, 0.0, 1.0),
                eye: Vec3::zeros(),
            },
            velocity: 0.25,
            rotation_sensitivity: Vec2::new(0.1, 0.1),
            cursor_last_position: Vec2::zeros(),
        }
    }
}

impl VizCameraControl for WSDVizCameraControl {
    fn key_event(&mut self, window_state: &WindowState, scene_state: &SceneState) {
        let move_increment =
            self.velocity * scene_state.world_bounds.radius * 2.0 * window_state.elapsed_time;

        match window_state.keyboard_state.get(&VirtualKeyCode::W) {
            Some(ElementState::Pressed) => {
                self.viz_camera.translate_eye(move_increment);
            }
            _ => {}
        }

        match window_state.keyboard_state.get(&VirtualKeyCode::W) {
            Some(ElementState::Pressed) => {
                self.viz_camera.translate_eye(-move_increment);
            }
            _ => {}
        }

        match window_state.keyboard_state.get(&VirtualKeyCode::S) {
            Some(ElementState::Pressed) => {
                self.viz_camera.translate_right(move_increment);
            }
            _ => {}
        }

        match window_state.keyboard_state.get(&VirtualKeyCode::D) {
            Some(ElementState::Pressed) => {
                self.viz_camera.translate_right(-move_increment);
            }
            _ => {}
        }
    }

    fn cursor_moved(
        &mut self,
        x: f64,
        y: f64,
        window_state: &WindowState,
        scene_state: &SceneState,
    ) {
        if let Some(ElementState::Pressed) = window_state.mouse_state.get(&MouseButton::Left) {
            let current_positon = Vec2::new(x as f32, y as f32);
            let mut difference = self.cursor_last_position - current_positon;
            difference[0] /= window_state.window_size[0] * self.rotation_sensitivity[0];
            difference[1] /= window_state.window_size[0] * self.rotation_sensitivity[1];
            self.viz_camera.rotate_up_axis(difference[0]);
            self.viz_camera.rotate_right_axis(difference[1]);
            self.cursor_last_position = current_positon;
        }
    }
}
