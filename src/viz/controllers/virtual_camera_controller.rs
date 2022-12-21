use nalgebra_glm::{Mat4, Vec2};
use winit::event::{ElementState, MouseButton, VirtualKeyCode};

use crate::viz::{virtual_camera::VirtualCameraSphericalBuilder, VirtualCamera};

use super::{SceneState, WindowState};

pub trait VirtualCameraControl {
    fn key_event(&mut self, window_state: &WindowState, scene_bounds: &SceneState);
    fn cursor_moved(
        &mut self,
        x: f64,
        y: f64,
        window_state: &WindowState,
        scene_bounds: &SceneState,
    );
    fn view_matrix(&self) -> Mat4;
    fn projection_matrix(&self) -> Mat4;
}

pub struct WASDVirtualCameraControl {
    pub camera: VirtualCamera,
    pub velocity: f32,
    pub rotation_sensitivity: Vec2,
    cursor_last_position: Vec2,
}

impl WASDVirtualCameraControl {
    pub fn fit_sphere_in_frustum(
        camera_builder: VirtualCameraSphericalBuilder,
        move_velocity: f32,
    ) -> Self {
        let move_velocity = move_velocity * camera_builder.sphere.radius;
        Self {
            camera: camera_builder.build(),
            velocity: move_velocity,
            ..Default::default()
        }
    }
}

impl Default for WASDVirtualCameraControl {
    fn default() -> Self {
        Self {
            camera: VirtualCamera::default(),
            velocity: 0.25,
            rotation_sensitivity: Vec2::new(0.1, 0.1),
            cursor_last_position: Vec2::zeros(),
        }
    }
}

impl VirtualCameraControl for WASDVirtualCameraControl {
    fn key_event(&mut self, window_state: &WindowState, scene_state: &SceneState) {
        let move_increment =
            self.velocity * scene_state.world_bounds.radius * 2.0 * window_state.elapsed_time.as_secs_f32();

        if let Some(ElementState::Pressed) = window_state.keyboard_state.get(&VirtualKeyCode::W) {
            self.camera.translate_eye(move_increment);
        }

        if let Some(ElementState::Pressed) = window_state.keyboard_state.get(&VirtualKeyCode::S) {
            self.camera.translate_eye(-move_increment);
        }

        if let Some(ElementState::Pressed) = window_state.keyboard_state.get(&VirtualKeyCode::A) {
            self.camera.translate_right(-move_increment);
        }

        if let Some(ElementState::Pressed) = window_state.keyboard_state.get(&VirtualKeyCode::D) {
            self.camera.translate_right(move_increment);
        }
    }

    fn cursor_moved(
        &mut self,
        x: f64,
        y: f64,
        window_state: &WindowState,
        _: &SceneState,
    ) {
        if let Some(ElementState::Pressed) = window_state.mouse_state.get(&MouseButton::Left) {
            let current_positon = Vec2::new(x as f32, y as f32);
            let mut difference = self.cursor_last_position - current_positon;
            difference[0] /= window_state.window_size[0] * self.rotation_sensitivity[0];
            difference[1] /= window_state.window_size[0] * self.rotation_sensitivity[1];
            self.camera.rotate_up_axis(difference[0]);
            self.camera.rotate_right_axis(difference[1]);
            self.cursor_last_position = current_positon;
        }
    }

    fn view_matrix(&self) -> Mat4 {
        self.camera.matrix()
    }

    fn projection_matrix(&self) -> Mat4 {
        self.camera.projection.matrix()
    }
}

#[cfg(test)]
mod test {
    use nalgebra::Vector3;

    use crate::{bounds::Sphere3Df, viz::virtual_camera::VirtualCameraSphericalBuilder};

    use super::{VirtualCameraControl, WASDVirtualCameraControl};

    #[test]
    pub fn test_should_instantiate_camera_controller() {
        let controller = WASDVirtualCameraControl::fit_sphere_in_frustum(
            VirtualCameraSphericalBuilder::fit(
                &Sphere3Df {
                    center: Vector3::new(2.0, 3.0, 4.0),
                    radius: 3.0,
                },
                std::f32::consts::PI / 2.0,
            ),
            0.05,
        );

        controller.view_matrix();
        controller.projection_matrix();
    }
}
