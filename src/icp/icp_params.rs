use std::{
    f32::consts::PI,
    ops::{Index, IndexMut},
};

/// ICP parameters
#[derive(Debug, Clone, Copy)]
pub struct IcpParams {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Weight of the geometric distance.
    pub weight: f32,
    /// Weight of the color distance.
    pub color_weight: f32,
    /// Maximum distance between a point and its projection on the plane.
    pub max_point_to_plane_distance: f32,
    /// Maximum distance between two points to be considered as the same.
    pub max_distance: f32,
    /// Maximum angle between two normals to be considered as the same in radians.
    pub max_normal_angle: f32,

    pub max_color_distance: f32,
}

impl Default for IcpParams {
    /// Default ICP parameters. Its values are:
    ///
    /// ```rust
    /// # use align3d::icp::IcpParams;
    /// let params = IcpParams::default();
    /// println!("{:?}", params);
    /// ```
    fn default() -> Self {
        Self {
            max_iterations: 15,
            weight: 1.0,
            color_weight: 1.0e-1,
            max_point_to_plane_distance: 0.1,
            max_distance: 0.5,
            max_normal_angle: 18.0_f32.to_radians(),
            max_color_distance: 0.25,
        }
    }
}

impl IcpParams {
    pub fn max_iterations(&'_ mut self, value: usize) -> &'_ mut IcpParams {
        self.max_iterations = value;
        self
    }

    pub fn weight(&'_ mut self, value: f32) -> &'_ mut IcpParams {
        self.weight = value;
        self
    }
}

#[derive(Debug, Clone)]
pub struct MsIcpParams {
    pyramid: Vec<IcpParams>,
}

impl MsIcpParams {
    pub fn new(pyramid: Vec<IcpParams>) -> Self {
        Self { pyramid }
    }

    pub fn repeat(levels: usize, params: &IcpParams) -> Self {
        Self {
            pyramid: vec![*params; levels],
        }
    }

    pub fn customize<F>(mut self, mut f: F) -> Self
    where
        F: FnMut(usize, &mut IcpParams),
    {
        for (i, params) in self.pyramid.iter_mut().enumerate() {
            f(i, params);
        }

        self
    }

    pub fn len(&self) -> usize {
        self.pyramid.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pyramid.is_empty()
    }

    pub fn iter(&self) -> impl DoubleEndedIterator + ExactSizeIterator<Item = &IcpParams> {
        self.pyramid.iter()
    }
}

impl Index<usize> for MsIcpParams {
    type Output = IcpParams;

    fn index(&self, index: usize) -> &Self::Output {
        &self.pyramid[index]
    }
}

impl IndexMut<usize> for MsIcpParams {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.pyramid[index]
    }
}

impl Default for MsIcpParams {
    fn default() -> Self {
        Self::repeat(
            3,
            &IcpParams {
                weight: 1.0,
                color_weight: 1.0,
                max_normal_angle: PI / 10.0,
                max_color_distance: 2.75,
                max_distance: 0.5,
                ..Default::default()
            },
        )
        .customize(|level, mut params| {
            match level {
                0 => params.max_iterations = 20, // 0 is the last level run
                1 => params.max_iterations = 20,
                2 => params.max_iterations = 30,
                _ => {}
            };
        })
    }
}
