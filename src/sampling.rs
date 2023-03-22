pub trait Downsample {
    type Output;
    fn downsample(&self, scale: f64) -> Self::Output;
}

