use nalgebra::Vector3;
use ndarray::parallel::prelude::ParallelIterator;
use vulkano::memory::{ExternalMemoryHandleType, ExternalMemoryHandleTypes};

use std::sync::Arc;
use vulkano::buffer::subbuffer::{BufferReadGuard, BufferWriteGuard};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferError, BufferCreateFlags};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::device::{Device, Queue};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage, MemoryAllocatePreference};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::sync::{self, GpuFuture};
use vulkano::{
    buffer::{BufferUsage, Subbuffer},
    memory::allocator::MemoryAllocator,
};

use super::Surfel;
use rayon::prelude::*;

struct Allocator {
    free_list: Vec<usize>,
    free_start: usize,
}

impl Allocator {
    pub fn new(pool_size: usize) -> Self {
        Self {
            free_list: (0..pool_size).collect(),
            free_start: 0,
        }
    }

    pub fn allocate(&mut self) -> Option<usize> {
        if self.free_start < self.free_list.len() {
            let index = self.free_list[self.free_start];
            self.free_start += 1;
            Some(index)
        } else {
            None
        }
    }

    pub fn free(&mut self, index: usize) {
        self.free_start -= 1;
        self.free_list[self.free_start] = index;
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct AttrPositionConfidence {
    #[format(R32G32B32A32_SFLOAT)]
    pub position_confidence: [f32; 4],
}

impl AttrPositionConfidence {
    pub fn new(position: Vector3<f32>, confidence: f32) -> Self {
        Self {
            position_confidence: [position.x, position.y, position.z, confidence],
        }
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct AttrNormalRadius {
    #[format(R32G32B32A32_SFLOAT)]
    pub normal_radius: [f32; 4],
}

impl AttrNormalRadius {
    pub fn new(normal: Vector3<f32>, radius: f32) -> Self {
        Self {
            normal_radius: [normal.x, normal.y, normal.z, radius],
        }
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct AttrColorMaskAge {
    #[format(R32G32_UINT)]
    pub rgbm: [u32; 2],
}

impl AttrColorMaskAge {
    pub fn new(r: u8, g: u8, b: u8, mask: u8, age: usize) -> Self {
        Self {
            rgbm: [
                ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | (mask as u32),
                0,
            ],
        }
    }

    pub fn into_parts(&self) -> (u8, u8, u8, u8, u32) {
        let rgbm = self.rgbm[0];
        (
            ((rgbm >> 24) & 0xff) as u8,
            ((rgbm >> 16) & 0xff) as u8,
            ((rgbm >> 8) & 0xff) as u8,
            (rgbm & 0xff) as u8,
            self.rgbm[1],
        )
    }

    pub fn mask(&self) -> u8 {
        (self.rgbm[0] & 0xff) as u8
    }

    pub fn set_mask(&mut self, mask: u8) {
        self.rgbm[0] = (self.rgbm[0] & 0xffffff00) | (mask as u32);
    }
}

pub struct SurfelModelWriter<'a> {
    position: BufferWriteGuard<'a, [AttrPositionConfidence]>,
    normal: BufferWriteGuard<'a, [AttrNormalRadius]>,
    color_n_mask: BufferWriteGuard<'a, [AttrColorMaskAge]>,
    allocator: &'a mut Allocator,
}

impl<'a> SurfelModelWriter<'a> {
    pub fn update(&mut self, index: usize, surfel: &Surfel) {
        self.position[index] = AttrPositionConfidence::new(surfel.position, surfel.confidence);
        self.normal[index] = AttrNormalRadius::new(surfel.normal, surfel.radius);
        self.color_n_mask[index] = AttrColorMaskAge::new(
            surfel.color[0],
            surfel.color[1],
            surfel.color[2],
            1,
            surfel.age as usize,
        );
    }

    pub fn add(&mut self, surfel: &Surfel) -> usize {
        let id = self.allocator.allocate().unwrap();
        self.update(id, surfel);
        id
    }

    pub fn free(&mut self, index: usize) {
        self.color_n_mask[index].set_mask(0);
        self.allocator.free(index);
    }
}

pub struct SurfelModelReader<'a> {
    position: BufferReadGuard<'a, [AttrPositionConfidence]>,
    normal: BufferReadGuard<'a, [AttrNormalRadius]>,
    color_n_mask: BufferReadGuard<'a, [AttrColorMaskAge]>,
    allocator: &'a Allocator,
}

impl<'a> SurfelModelReader<'a> {
    pub fn get(&self, index: usize) -> Option<Surfel> {
        let (r, g, b, m, age) = self.color_n_mask[index].into_parts();
        if m == 0 {
            return None;
        }

        let position = self.position[index].position_confidence;
        let normal = self.normal[index].normal_radius;

        Some(Surfel {
            position: Vector3::new(position[0], position[1], position[2]),
            normal: Vector3::new(normal[0], normal[1], normal[2]),
            color: Vector3::new(r, g, b),
            radius: normal[3],
            confidence: position[3],
            age: age as i32,
        })
    }

    pub fn get2(&self, index: usize) -> Surfel {
        let position = self.position[index].position_confidence;
        let normal = self.normal[index].normal_radius;
        let (r, g, b, m, age) = self.color_n_mask[index].into_parts();
        Surfel {
            position: Vector3::new(position[0], position[1], position[2]),
            normal: Vector3::new(normal[0], normal[1], normal[2]),
            color: Vector3::new(r, g, b),
            radius: normal[3],
            confidence: position[3],
            age: age as i32,
        }

        // let position = self.position[index].position;
        // let normal = self.normal[index].normal;
        // Surfel {
        //     position: Vector3::new(position[0], position[1], position[2]),
        //     normal: Vector3::new(normal[0], normal[1], normal[2]),
        //     color: Vector3::new(0, 0, 0),
        //     radius: 0.5,
        //     confidence: 0.2,
        //     age: 5,
        // }
    }

    pub fn position_iter(&'a self) -> impl Iterator<Item = (usize, Vector3<f32>)> + 'a {
        self.position
            .iter()
            .zip(self.color_n_mask.iter())
            .enumerate()
            .filter_map(move |(i, (p, m))| {
                if m.mask() == 0 {
                    None
                } else {
                    Some((
                        i,
                        Vector3::new(
                            p.position_confidence[0],
                            p.position_confidence[1],
                            p.position_confidence[2],
                        ),
                    ))
                }
            })
    }

    pub fn position_par_iter(&'a self) -> impl ParallelIterator<Item = (usize, Vector3<f32>)> + 'a {
        self.position
            .iter()
            .zip(self.color_n_mask.iter())
            .enumerate()
            .par_bridge()
            .filter_map(move |(i, (p, m))| {
                if m.mask() == 0 {
                    None
                } else {
                    Some((
                        i,
                        Vector3::new(
                            p.position_confidence[0],
                            p.position_confidence[1],
                            p.position_confidence[2],
                        ),
                    ))
                }
            })
    }

    pub fn age_confidence_iter(&'a self) -> impl Iterator<Item = (usize, i32, f32)> + 'a {
        self.color_n_mask
            .iter()
            .zip(self.position.iter())
            .enumerate()
            .filter_map(move |(i, (a, c))| {
                if a.mask() == 0 {
                    None
                } else {
                    Some((i, a.rgbm[1] as i32, c.position_confidence[3]))
                }
            })
    }
}

pub struct SurfelModelBuffers {
    pub position: Subbuffer<[AttrPositionConfidence]>,
    pub normal: Subbuffer<[AttrNormalRadius]>,
    pub color_n_mask: Subbuffer<[AttrColorMaskAge]>,
}


impl SurfelModelBuffers {
    pub fn new(
        memory_allocator: &(impl MemoryAllocator + ?Sized),
        size: usize,
        buffer_usage: BufferUsage,
        memory_usage: MemoryUsage,
    ) -> Self {
        let mut bff = BufferCreateFlags::default();
        
        let create_info = BufferCreateInfo {
            usage: buffer_usage,
            external_memory_handle_types: ExternalMemoryHandleTypes::HOST_ALLOCATION,
            ..Default::default()
        };
        let alloc_info = AllocationCreateInfo {
            usage: memory_usage,
            //allocate_preference: MemoryAllocatePreference::NeverAllocate,
            ..Default::default()
        };

        Self {
            position: Buffer::from_iter(
                memory_allocator,
                create_info.clone(),
                alloc_info.clone(),
                (0..size).map(|_| AttrPositionConfidence::new(Vector3::zeros(), 0.0)),
            )
            .unwrap(),
            normal: Buffer::from_iter(
                memory_allocator,
                create_info.clone(),
                alloc_info.clone(),
                (0..size).map(|_| AttrNormalRadius::new(Vector3::zeros(), 0.0)),
            )
            .unwrap(),
            color_n_mask: Buffer::from_iter(
                memory_allocator,
                create_info.clone(),
                alloc_info.clone(),
                (0..size).map(|_| AttrColorMaskAge::new(0, 0, 0, 0, 0)),
            )
            .unwrap(),
        }
    }
}

pub struct SurfelModel {
    pub graphics: SurfelModelBuffers,
    pub process: SurfelModelBuffers,
    allocator: Allocator,
    size: usize,
}

impl SurfelModel {
    pub fn new(memory_allocator: &(impl MemoryAllocator + ?Sized), size: usize) -> Self {
        SurfelModel {
            graphics: SurfelModelBuffers::new(
                memory_allocator,
                size,
                BufferUsage::TRANSFER_DST | BufferUsage::VERTEX_BUFFER,
                MemoryUsage::Upload,
            ),
            process: SurfelModelBuffers::new(
                memory_allocator,
                size,
                BufferUsage::TRANSFER_SRC | BufferUsage::UNIFORM_BUFFER,
                MemoryUsage::Upload,

            ),
            allocator: Allocator::new(size),
            size,
        }
    }

    pub fn read(&'_ self) -> Result<SurfelModelReader<'_>, BufferError> {
        Ok(SurfelModelReader {
            position: self.process.position.read()?,
            normal: self.process.normal.read()?,
            color_n_mask: self.process.color_n_mask.read()?,
            allocator: &self.allocator,
        })
    }

    pub fn write(&'_ mut self) -> Result<SurfelModelWriter<'_>, BufferError> {
        Ok(SurfelModelWriter {
            position: self.process.position.write()?,
            normal: self.process.normal.write()?,
            color_n_mask: self.process.color_n_mask.write()?,
            allocator: &mut self.allocator,
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn swap_graphics(&mut self, device: Arc<Device>, queue: Arc<Queue>) {
        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        );

        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.process.position.clone(),
                self.graphics.position.clone(),
            ))
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(
                self.process.normal.clone(),
                self.graphics.normal.clone(),
            ))
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(
                self.process.color_n_mask.clone(),
                self.graphics.color_n_mask.clone(),
            ))
            .unwrap();

        let command_buffer = builder.build().unwrap();
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
    }
}
