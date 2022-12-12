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
pub struct Array3f32 {
    position : [f32; 3],
}
impl_vertex!(Array3f32, position);

impl Array3f32 {
    pub fn new(x: f32, y:f32, z:f32) -> Self {
        Self {
            position: [x, y, z]
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Array3u8 {
    data : [u8; 3],
}
impl_vertex!(Array3u8);

impl Array3u8 {
    pub fn new(x: u8, y:u8, z:u8) -> Self {
        Self {
            data: [x, y, z]
        }
    }
}
