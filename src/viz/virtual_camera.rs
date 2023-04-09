use nalgebra::Vector3;
use nalgebra_glm::Vec3;

use crate::bounds::Sphere3Df;

use super::{virtual_projection::PerspectiveVirtualProjectionBuilder, VirtualProjection};

const VULKAN_UP: Vector3<f32> = Vector3::new(0.0, -1.0, 0.0);

/// Virtual camera to move around in the visualization.
pub struct VirtualCamera {
    /// Camera position point.
    pub eye: Vec3,
    /// Viewing vector. Always normalized.
    pub view: Vec3,
    /// Up vector. Always normalized.
    pub up: Vec3,
    /// Projection parameters.
    pub projection: VirtualProjection,
}

impl Default for VirtualCamera {
    fn default() -> VirtualCamera {
        Self {
            eye: Vec3::new(0.0, 0.0, -1.0),
            view: Vec3::new(0.0, 0.0, 1.0),
            up: VULKAN_UP,
            projection: VirtualProjection::default(),
        }
    }
}

impl VirtualCamera {
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

        let right_vec = self.view.cross(&VULKAN_UP);
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

pub struct VirtualCameraSphericalBuilder {
    pub sphere: Sphere3Df,
    elevation: f32,
    azimuth: f32,
    distance: f32,
    fov_y: f32,
    aspect_ratio: f32,
    near_plane_distance: f32,
    far_plane_distance: f32,
}

impl Default for VirtualCameraSphericalBuilder {
    fn default() -> Self {
        Self {
            sphere: Sphere3Df {
                center: Vector3::zeros(),
                radius: 1.0,
            },
            elevation: 0.0,
            azimuth: 0.0,
            distance: 1.0,
            fov_y: 45.0,
            aspect_ratio: 1.0,
            near_plane_distance: 1.0,
            far_plane_distance: 10.0,
        }
    }
}

impl VirtualCameraSphericalBuilder {
    pub fn fit(sphere: &Sphere3Df, fov_y: f32) -> Self {
        if sphere.is_empty() {
            panic!("Cannot fit empty sphere.");
        }
        let fov_y = fov_y / 2.0;
        let alpha = fov_y;
        let theta = std::f32::consts::FRAC_PI_2 - fov_y;

        let distance = alpha.cos() * ((theta.sin() * sphere.radius) / alpha.sin())
            + theta.cos() * sphere.radius;
        let near = distance - sphere.radius;

        Self {
            sphere: *sphere,
            distance,
            fov_y,
            near_plane_distance: near,
            ..Default::default()
        }
    }

    pub fn elevation(mut self, value: f32) -> Self {
        self.elevation = value;
        self
    }

    pub fn azimuth(mut self, value: f32) -> Self {
        self.azimuth = value;
        self
    }

    pub fn distance(mut self, value: f32) -> Self {
        self.distance = value;
        self
    }

    pub fn fov_y(mut self, value: f32) -> Self {
        self.fov_y = value;
        self
    }

    pub fn aspect_ratio(mut self, value: f32) -> Self {
        self.aspect_ratio = value;
        self
    }

    pub fn near_plane(mut self, value: f32) -> Self {
        self.near_plane_distance = value;
        self
    }

    pub fn far_plane(mut self, value: f32) -> Self {
        self.far_plane_distance = value;
        self
    }

    pub fn build(self) -> VirtualCamera {
        let theta = self.elevation;
        let phi = self.azimuth + std::f32::consts::PI * 1.5;

        let position = Vec3::new(
            phi.cos() * self.distance * theta.cos(),
            theta.sin() * self.distance,
            phi.sin() * self.distance * theta.cos(),
        ) + self.sphere.center;

        let view = (self.sphere.center - position).normalize();
        let right = view.cross(&Vec3::new(0.0, -1.0, 0.0)).normalize();
        let up = right.cross(&view).normalize();

        VirtualCamera {
            eye: position,
            view,
            up,
            projection: PerspectiveVirtualProjectionBuilder {
                fov_y: self.fov_y,
                aspect_ratio: self.aspect_ratio,
                near_plane: self.near_plane_distance,
                far_plane: self.far_plane_distance,
            }
            .build(),
        }
    }
}

#[cfg(test)]
mod test {
    use nalgebra::Vector3;

    use crate::bounds::Sphere3Df;

    use super::VirtualCameraSphericalBuilder;

    #[test]
    pub fn test_should_fit_view_bounds() {
        let sphere = Sphere3Df {
            center: Vector3::new(2.0, 3.0, 4.0),
            radius: 3.4,
        };
        let _camera = VirtualCameraSphericalBuilder::fit(&sphere, std::f32::consts::PI / 2.0)
            .azimuth(4.0)
            .elevation(1.0);
        //let camera = camera.build();

        //camera.matrix();
        //camera.projection.matrix();
    }
}
