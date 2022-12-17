use nalgebra::Vector3;
use nalgebra_glm::Vec3;

use crate::bounds::Sphere3Df;

use super::{virtual_projection::PerspectiveVirtualProjectionBuilder, VirtualProjection};

pub struct VirtualCamera {
    pub eye: Vec3,
    pub view: Vec3,
    pub up: Vec3,
    pub projection: VirtualProjection,
}

impl Default for VirtualCamera {
    fn default() -> VirtualCamera {
        Self {
            eye: Vec3::new(0.0, 0.0, 1.0),
            view: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            projection: VirtualProjection::default()
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

pub struct VirtualCameraSphericalBuilder {
    pub sphere: Sphere3Df,
    elevation_: f32,
    azimuth_: f32,
    distance_: f32,
    fov_y_: f32,
    aspect_ratio_: f32,
    near_plane_distance_: f32,
    far_plane_distance_: f32,
}

impl Default for VirtualCameraSphericalBuilder {
    fn default() -> Self {
        Self {
            sphere: Sphere3Df { center: Vector3::zeros(), radius: 1.0 },
            elevation_: 0.0,
            azimuth_: 0.0,
            distance_: 1.0,
            fov_y_: 45.0,
            aspect_ratio_: 1.0,
            near_plane_distance_: 1.0,
            far_plane_distance_: 10.0,
        }
    }
}

impl VirtualCameraSphericalBuilder {
    pub fn fit(sphere: &Sphere3Df, fov_y: f32) -> Self {
        let fov_y = fov_y / 2.0;
        let alpha = fov_y;
        let theta = std::f32::consts::FRAC_PI_2 - fov_y;

        let distance = alpha.cos() * ((theta.sin() * sphere.radius) / alpha.sin())
            + theta.cos() * sphere.radius;
        let near = distance - sphere.radius;

        Self {
            sphere: sphere.clone(),
            distance_: distance,
            fov_y_: fov_y,
            near_plane_distance_: near,
            ..Default::default()
        }
    }

    pub fn elevation(&'_ mut self, value: f32) -> &'_ mut Self {
        self.elevation_ = value;
        self
    }

    pub fn azimuth(&'_ mut self, value: f32) -> &'_ mut Self {
        self.azimuth_ = value;
        self
    }

    pub fn distance(&'_ mut self, value: f32) -> &'_ mut Self {
        self.distance_ = value;
        self
    }

    pub fn fov_y(&'_ mut self, value: f32) -> &'_ mut Self {
        self.fov_y_ = value;
        self
    }

    pub fn aspect_ratio(&'_ mut self, value: f32) -> &'_ mut Self {
        self.aspect_ratio_ = value;
        self
    }

    pub fn build(self) -> VirtualCamera {
        let theta = self.elevation_;
        let phi = self.azimuth_ + std::f32::consts::PI * 1.5;

        let mut position = Vec3::new(
            phi.cos() * self.distance_ * theta.cos(),
            theta.sin() * self.distance_,
            -phi.sin() * self.distance_ * theta.cos(),
        );
        position += &self.sphere.center;

        let view = (&self.sphere.center - &position).normalize();
        let right = view.cross(&Vec3::new(0.0, 1.0, 0.0)).normalize();
        let up = right.cross(&view).normalize();

        VirtualCamera {
            eye: position,
            view: view,
            up: up,
            projection: PerspectiveVirtualProjectionBuilder {
                fov_y: self.fov_y_,
                aspect_ratio: self.aspect_ratio_,
                near_plane_distance: self.near_plane_distance_,
                far_plane_distance: self.far_plane_distance_,
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

    pub fn test_should_fit_view_bounds() {
        let sphere = Sphere3Df {
            center: Vector3::new(2.0, 3.0, 4.0),
            radius: 3.4,
        };
        let camera = VirtualCameraSphericalBuilder::fit(&sphere, std::f32::consts::PI / 2.0)
            .azimuth(4.0)
            .elevation(1.0)
            .build();

        camera.matrix();
        camera.projection.matrix();
    }
}
