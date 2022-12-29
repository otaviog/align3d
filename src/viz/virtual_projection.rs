use nalgebra_glm::Mat4;

pub struct VirtualProjection {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
    pub far_left: f32,
    pub far_right: f32,
    pub far_bottom: f32,
    pub far_top: f32,
    pub near: f32,
    pub far: f32,
}

impl VirtualProjection {
    pub fn new(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        VirtualProjection {
            left,
            right,
            top,
            bottom,
            far_left: (left / near) * far,
            far_right: (right / near) * far,
            far_top: (top / near) * far,
            far_bottom: (bottom / near) * far,
            near,
            far,
        }
    }
}

impl Default for VirtualProjection {
    fn default() -> VirtualProjection {
        VirtualProjection::new(-1.0, 1.0, -1.0, 1.0, 1.0, 100.0)
    }
}

impl VirtualProjection {
    pub fn matrix(&self) -> Mat4 {
        let mut matrix = Mat4::zeros();

        matrix[(0, 0)] = 2.0 * self.near / (self.right - self.left);
        // matrix[(0, 1)] = 0.0;
        matrix[(0, 2)] = (self.right + self.left) / (self.right - self.left);
        // matrix[(0, 3)] = 0.0;

        // matrix[(1, 0)] = 0.0;
        matrix[(1, 1)] = 2.0 * self.near / (self.top - self.bottom);
        matrix[(1, 2)] = (self.top + self.bottom) / (self.top - self.bottom);
        // matrix[(1, 3)] = 0.0;

        // matrix[(2, 0)] = 0.0;
        // matrix[(2, 1)] = 0.0;
        matrix[(2, 2)] = -(self.far + self.near) / (self.far - self.near);
        matrix[(2, 3)] = -(2.0 * self.far * self.near) / (self.far - self.near);

        //matrix[(3, 0)] = 0.0;
        //matrix[(3, 1)] = 0.0;
        matrix[(3, 2)] = -1.0;
        //matrix[(3, 3)] = 0.0;

        matrix
    }
}

pub struct PerspectiveVirtualProjectionBuilder {
    pub fov_y: f32,
    pub aspect_ratio: f32,
    pub near_plane: f32,
    pub far_plane: f32,
}

impl PerspectiveVirtualProjectionBuilder {
    pub fn fov_y(&'_ mut self, value: f32) -> &'_ mut Self {
        self.fov_y = value;
        self
    }

    pub fn aspect_ratio(&'_ mut self, value: f32) -> &'_ mut Self {
        self.aspect_ratio = value;
        self
    }

    pub fn near_plane(&'_ mut self, value: f32) -> &'_ mut Self {
        self.near_plane = value;
        self
    }

    pub fn far_plane(&'_ mut self, value: f32) -> &'_ mut Self {
        self.far_plane = value;
        self
    }

    pub fn build(self) -> VirtualProjection {
        let top = (self.fov_y / 2.0).tan() * self.near_plane;
        let right = top * self.aspect_ratio;

        VirtualProjection::new(-right, right, -top, top, self.near_plane, self.far_plane)
    }
}
