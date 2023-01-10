pub trait Downsampleble<T> {
    fn downsample(&self, scale: f64) -> T;
}

