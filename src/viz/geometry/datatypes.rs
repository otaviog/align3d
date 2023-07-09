use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Array2f32 {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

impl Array2f32 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { position: [x, y] }
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct PositionF32 {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
}

impl PositionF32 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: [x, y, z],
        }
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct NormalF32 {
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
}

impl NormalF32 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { normal: [x, y, z] }
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct ColorU8 {
    #[format(R32_UINT)]
    pub rgb: u32,
}

impl ColorU8 {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self {
            rgb: ((r as u32) << 16) | ((g as u32) << 8) | (b as u32),
        }
    }

    pub fn into_parts(&self) -> (u8, u8, u8) {
        (
            ((self.rgb >> 16) & 0xff) as u8,
            ((self.rgb >> 8) & 0xff) as u8,
            (self.rgb & 0xff) as u8,
        )
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct ScalarF32 {
    #[format(R32_SFLOAT)]
    pub value: f32,
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct ScalarI32 {
    #[format(R32_SINT)]
    pub value: i32,
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct ScalarU32 {
    #[format(R32_UINT)]
    pub value: u32,
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
