#[derive(Debug, Clone, Copy)]
pub struct IcpParams {
    pub max_iterations: usize,
    pub weight: f32,
}

impl IcpParams {
    pub fn default() -> Self {
        Self {
            max_iterations: 15,
            weight: 0.5,
        }
    }

    pub fn max_iterations(&'_ mut self, value: usize) -> &'_ mut IcpParams {
        self.max_iterations = value;
        self
    }

    pub fn weight(&'_ mut self, value: f32) -> &'_ mut IcpParams {
        self.weight = value;
        self
    }
}
