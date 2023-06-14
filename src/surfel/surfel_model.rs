use nalgebra::Vector3;
use std::collections::BTreeSet;
use std::sync::Arc;
use vulkano::buffer::subbuffer::{BufferReadGuard, BufferWriteGuard};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferError};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::device::{Device, Queue};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::sync::{self, GpuFuture};
use vulkano::{
    buffer::{BufferUsage, Subbuffer},
    memory::allocator::MemoryAllocator,
};

use crate::viz::geometry::{NormalF32, PositionF32};

use super::Surfel;

pub struct SurfelModelWriter<'a> {
    position: BufferWriteGuard<'a, [PositionF32]>,
    normal: BufferWriteGuard<'a, [NormalF32]>,
    color_n_mask: BufferWriteGuard<'a, [AttrColorMask]>,
    radius: BufferWriteGuard<'a, [f32]>,
    confidence: BufferWriteGuard<'a, [f32]>,
    age: BufferWriteGuard<'a, [i32]>,
    free_list: &'a mut BTreeSet<usize>,
}

impl<'a> SurfelModelWriter<'a> {
    pub fn update(&mut self, index: usize, surfel: &Surfel) {
        self.position[index] =
            PositionF32::new(surfel.position[0], surfel.position[1], surfel.position[2]);
        self.normal[index] = NormalF32::new(surfel.normal[0], surfel.normal[1], surfel.normal[2]);
        self.color_n_mask[index] =
            AttrColorMask::new(surfel.color[0], surfel.color[1], surfel.color[2], 1);
        self.radius[index] = surfel.radius;
        self.confidence[index] = surfel.confidence;
        self.age[index] = surfel.age;
    }

    pub fn add(&mut self, surfel: &Surfel) -> usize {
        let id = self.allocate();
        self.update(id, surfel);
        id
    }

    fn allocate(&mut self) -> usize {
        let found_index = self
            .free_list
            .iter()
            .next()
            .copied()
            .unwrap_or_else(|| panic!("No free surfel indices available"));
        self.free_list.remove(&found_index);
        found_index
    }

    pub fn free(&mut self, index: usize) {
        self.color_n_mask[index].rgbm = 0;
        self.free_list.insert(index);
    }
}

pub struct SurfelModelReader<'a> {
    position: BufferReadGuard<'a, [PositionF32]>,
    normal: BufferReadGuard<'a, [NormalF32]>,
    color_n_mask: BufferReadGuard<'a, [AttrColorMask]>,
    radius: BufferReadGuard<'a, [f32]>,
    confidence: BufferReadGuard<'a, [f32]>,
    age: BufferReadGuard<'a, [i32]>,
    free_list: &'a BTreeSet<usize>,
}

impl<'a> SurfelModelReader<'a> {
    pub fn get(&self, index: usize) -> Option<Surfel> {
        if self.free_list.contains(&index) {
            return None;
        }

        let position = self.position[index].position;
        let normal = self.normal[index].normal;
        let (r, g, b, m) = self.color_n_mask[index].into_parts();
        assert!(m == 1);
        Some(Surfel {
            position: Vector3::new(position[0], position[1], position[2]),
            normal: Vector3::new(normal[0], normal[1], normal[2]),
            color: Vector3::new(r, g, b),
            radius: self.radius[index],
            confidence: self.confidence[index],
            age: self.age[index],
        })
    }

    pub fn position_iter(&'a self) -> impl Iterator<Item = (usize, Vector3<f32>)> + 'a {
        self.position.iter().enumerate().filter_map(move |(i, p)| {
            if self.free_list.contains(&i) {
                None
            } else {
                Some((i, Vector3::new(p.position[0], p.position[1], p.position[2])))
            }
        })
    }

    pub fn age_confidence_iter(&'a self) -> impl Iterator<Item = (usize, i32, f32)> + 'a {
        self.age
            .iter()
            .zip(self.confidence.iter())
            .enumerate()
            .filter_map(move |(i, (a, c))| {
                if self.free_list.contains(&i) {
                    None
                } else {
                    Some((i, *a, *c))
                }
            })
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct AttrColorMask {
    #[format(R32_UINT)]
    pub rgbm: u32,
}

impl AttrColorMask {
    pub fn new(r: u8, g: u8, b: u8, mask: u8) -> Self {
        Self {
            rgbm: ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | (mask as u32),
        }
    }

    pub fn into_parts(&self) -> (u8, u8, u8, u8) {
        (
            ((self.rgbm >> 24) & 0xff) as u8,
            ((self.rgbm >> 16) & 0xff) as u8,
            ((self.rgbm >> 8) & 0xff) as u8,
            (self.rgbm & 0xff) as u8,
        )
    }
}

pub struct SurfelModelBuffers {
    pub position: Subbuffer<[PositionF32]>,
    pub normal: Subbuffer<[NormalF32]>,
    pub color_n_mask: Subbuffer<[AttrColorMask]>,
    pub radius: Subbuffer<[f32]>,
    pub confidence: Subbuffer<[f32]>,
    pub age: Subbuffer<[i32]>,
}

impl SurfelModelBuffers {
    pub fn new(
        memory_allocator: &(impl MemoryAllocator + ?Sized),
        size: usize,
        buffer_usage: BufferUsage,
        memory_usage: MemoryUsage,
    ) -> Self {
        let create_info = BufferCreateInfo {
            usage: buffer_usage,
            ..Default::default()
        };
        let alloc_info = AllocationCreateInfo {
            usage: memory_usage,
            ..Default::default()
        };

        Self {
            position: Buffer::from_iter(
                memory_allocator,
                create_info.clone(),
                alloc_info.clone(),
                (0..size).map(|_| PositionF32::new(0.0, 0.0, 0.0)),
            )
            .unwrap(),
            normal: Buffer::from_iter(
                memory_allocator,
                create_info.clone(),
                alloc_info.clone(),
                (0..size).map(|_| NormalF32::new(0.0, 0.0, 0.0)),
            )
            .unwrap(),
            color_n_mask: Buffer::from_iter(
                memory_allocator,
                create_info.clone(),
                alloc_info.clone(),
                (0..size).map(|_| AttrColorMask::new(0, 0, 0, 0)),
            )
            .unwrap(),
            radius: Buffer::from_iter(
                memory_allocator,
                create_info.clone(),
                alloc_info.clone(),
                (0..size).map(|_| 0.0),
            )
            .unwrap(),
            confidence: Buffer::from_iter(
                memory_allocator,
                create_info.clone(),
                alloc_info.clone(),
                (0..size).map(|_| 0.0),
            )
            .unwrap(),
            age: Buffer::from_iter(
                memory_allocator,
                create_info,
                alloc_info,
                (0..size).map(|_| 0),
            )
            .unwrap(),
        }
    }
}

pub struct SurfelModel {
    pub graphics: SurfelModelBuffers,
    pub process: SurfelModelBuffers,
    free_list: BTreeSet<usize>,
    size: usize,
}

impl SurfelModel {
    pub fn new(memory_allocator: &(impl MemoryAllocator + ?Sized), size: usize) -> Self {
        SurfelModel {
            graphics: SurfelModelBuffers::new(memory_allocator, size, BufferUsage::TRANSFER_DST | BufferUsage::VERTEX_BUFFER, MemoryUsage::Upload),
            process: SurfelModelBuffers::new(memory_allocator, size, BufferUsage::TRANSFER_SRC, MemoryUsage::Upload),
            free_list: BTreeSet::from_iter(0..size),
            size,
        }
    }

    pub fn read(&'_ self) -> Result<SurfelModelReader<'_>, BufferError> {
        Ok(SurfelModelReader {
            position: self.process.position.read()?,
            normal: self.process.normal.read()?,
            color_n_mask: self.process.color_n_mask.read()?,
            radius: self.process.radius.read()?,
            confidence: self.process.confidence.read()?,
            age: self.process.age.read()?,
            free_list: &self.free_list,
        })
    }

    pub fn write(&'_ mut self) -> Result<SurfelModelWriter<'_>, BufferError> {
        Ok(SurfelModelWriter {
            position: self.process.position.write()?,
            normal: self.process.normal.write()?,
            color_n_mask: self.process.color_n_mask.write()?,
            radius: self.process.radius.write()?,
            confidence: self.process.confidence.write()?,
            age: self.process.age.write()?,
            free_list: &mut self.free_list,
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
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(
                self.process.radius.clone(),
                self.graphics.radius.clone(),
            ))
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(
                self.process.confidence.clone(),
                self.graphics.confidence.clone(),
            ))
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(
                self.process.age.clone(),
                self.graphics.age.clone(),
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
