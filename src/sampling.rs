pub trait Downsample {
    type Output;
    fn downsample(&self, scale: f32) -> Self::Output;
}

