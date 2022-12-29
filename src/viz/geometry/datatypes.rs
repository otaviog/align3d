use bytemuck::{Pod, Zeroable};
use vulkano::impl_vertex;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Array2f32 {
    position: [f32; 2],
}
impl_vertex!(Array2f32, position);

impl Array2f32 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { position: [x, y] }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct PositionF32 {
    pub position: [f32; 3],
}
impl_vertex!(PositionF32, position);

impl PositionF32 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: [x, y, z],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct NormalF32 {
    normal: [f32; 3],
}
impl_vertex!(NormalF32, normal);

impl NormalF32 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { normal: [x, y, z] }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct ColorU8 {
    pub rgb: u32,
}
impl_vertex!(ColorU8, rgb);

impl ColorU8 {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self {
            rgb: ((r as u32) << 16) | ((g as u32) << 8) | (b as u32),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ColorU8;

    #[test]
    fn color_should_create_int_color() {
        let color = ColorU8::new(255, 155, 55);
        assert_eq!((color.rgb >> 16) & 0xff, 255);
        assert_eq!((color.rgb >> 8) & 0xff, 155);
        assert_eq!(color.rgb & 0xff, 55);
    }
}
