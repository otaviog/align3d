use bytemuck::{Pod, Zeroable};
use vulkano::impl_vertex;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Array2f32 {
    position : [f32; 2],
}
impl_vertex!(Array2f32, position);

impl Array2f32 {
    pub fn new(x: f32, y:f32) -> Self {
        Self {
            position: [x, y]
        }
    }
}


#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct PositionF32 {
    position : [f32; 3],
}
impl_vertex!(PositionF32, position);

impl PositionF32 {
    pub fn new(x: f32, y:f32, z:f32) -> Self {
        Self {
            position: [x, y, z]
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct NormalF32 {
    normal : [f32; 3],
}
impl_vertex!(NormalF32, normal);

impl NormalF32 {
    pub fn new(x: f32, y:f32, z:f32) -> Self {
        Self {
            normal: [x, y, z]
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct ColorU8 {
    rgb : [u8; 3],
}
impl_vertex!(ColorU8, rgb);

impl ColorU8 {
    pub fn new(x: u8, y:u8, z:u8) -> Self {
        Self {
            rgb: [x, y, z]
        }
    }
}
