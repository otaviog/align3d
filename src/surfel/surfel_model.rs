use nalgebra::Vector3;
use std::collections::HashSet;
use std::sync::Arc;
use vulkano::buffer::cpu_access::{ReadLock, ReadLockError, WriteLock, WriteLockError};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    memory::allocator::MemoryAllocator,
};

use crate::viz::geometry::{ColorU8, NormalF32, PositionF32};

use super::Surfel;

pub struct SurfelModel {
    pub position: Arc<CpuAccessibleBuffer<[PositionF32]>>,
    pub normal: Arc<CpuAccessibleBuffer<[NormalF32]>>,
    pub color: Arc<CpuAccessibleBuffer<[ColorU8]>>,
    pub radius: Arc<CpuAccessibleBuffer<[f32]>>,
    pub confidence: Arc<CpuAccessibleBuffer<[f32]>>,
    pub age: Arc<CpuAccessibleBuffer<[i32]>>,
    pub mask: Arc<CpuAccessibleBuffer<[u8]>>,
    free_list: HashSet<usize>,
    size: usize,
}

pub struct SurfelModelWriter<'a> {
    position: WriteLock<'a, [PositionF32]>,
    normal: WriteLock<'a, [NormalF32]>,
    color: WriteLock<'a, [ColorU8]>,
    radius: WriteLock<'a, [f32]>,
    confidence: WriteLock<'a, [f32]>,
    age: WriteLock<'a, [i32]>,
    mask: WriteLock<'a, [u8]>,
    free_list: &'a mut HashSet<usize>,
}

impl<'a> SurfelModelWriter<'a> {
    pub fn update(&mut self, index: usize, surfel: Surfel) {
        self.position[index] =
            PositionF32::new(surfel.position[0], surfel.position[1], surfel.position[2]);
        self.normal[index] = NormalF32::new(surfel.normal[0], surfel.normal[1], surfel.normal[2]);
        self.color[index] = ColorU8::new(surfel.color[0], surfel.color[1], surfel.color[2]);
        self.radius[index] = surfel.radius;
        self.confidence[index] = surfel.confidence;
        self.age[index] = surfel.age;
        self.mask[index] = 1;
    }

    pub fn add(&mut self, surfel: Surfel) -> usize {
        let id = self.allocate();
        self.update(id, surfel);
        id
    }

    fn allocate(&mut self) -> usize {
        let found_index = self.free_list.iter().next().copied();
        found_index.unwrap_or_else(|| panic!("No free surfel indices available"))
    }

    pub fn free(&mut self, index: usize) {
        self.mask[index] = 0;
        self.free_list.insert(index);
    }
}

pub struct SurfelModelReader<'a> {
    position: ReadLock<'a, [PositionF32]>,
    normal: ReadLock<'a, [NormalF32]>,
    color: ReadLock<'a, [ColorU8]>,
    radius: ReadLock<'a, [f32]>,
    confidence: ReadLock<'a, [f32]>,
    age: ReadLock<'a, [i32]>,
    free_list: &'a HashSet<usize>,
}

impl<'a> SurfelModelReader<'a> {
    pub fn get(&self, index: usize) -> Option<Surfel> {
        if self.free_list.contains(&index) {
            return None;
        }

        let position = self.position[index].position;
        let normal = self.normal[index].normal;
        let (r, g, b) = self.color[index].into_parts();

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
        self.position
            .iter()
            .enumerate()
            .filter_map(move |(i, p)| {
                if self.free_list.contains(&i) {
                    None
                } else {
                    Some((i, Vector3::new(p.position[0], p.position[1], p.position[2])))
                }
            })
    }

    pub fn age_confidence_iter(&'a self) -> impl Iterator<Item = (usize, f32, f32)> + 'a {
        self.age
            .iter()
            .zip(self.confidence.iter())
            .enumerate()
            .filter_map(move |(i, (a, c))| {
                if self.free_list.contains(&i) {
                    None
                } else {
                    Some((i, *a as f32, *c))
                }
            })
    }
}

impl SurfelModel {
    pub fn new(memory_allocator: &(impl MemoryAllocator + ?Sized), size: usize) -> Self {
        let buffer_usage = BufferUsage {
            vertex_buffer: true,
            ..BufferUsage::default()
        };

        SurfelModel {
            position: CpuAccessibleBuffer::from_iter(
                memory_allocator,
                buffer_usage,
                false,
                (0..size).map(|_| PositionF32::new(0.0, 0.0, 0.0)),
            )
            .unwrap(),
            normal: CpuAccessibleBuffer::from_iter(
                memory_allocator,
                buffer_usage,
                false,
                (0..size).map(|_| NormalF32::new(0.0, 0.0, 0.0)),
            )
            .unwrap(),
            color: CpuAccessibleBuffer::from_iter(
                memory_allocator,
                buffer_usage,
                false,
                (0..size).map(|_| ColorU8::new(0, 0, 0)),
            )
            .unwrap(),
            radius: CpuAccessibleBuffer::from_iter(
                memory_allocator,
                buffer_usage,
                false,
                (0..size).map(|_| 0.0),
            )
            .unwrap(),
            confidence: CpuAccessibleBuffer::from_iter(
                memory_allocator,
                buffer_usage,
                false,
                (0..size).map(|_| 0.0),
            )
            .unwrap(),
            age: CpuAccessibleBuffer::from_iter(
                memory_allocator,
                buffer_usage,
                false,
                (0..size).map(|_| 0),
            )
            .unwrap(),
            free_list: HashSet::new(),
            mask: CpuAccessibleBuffer::from_iter(
                memory_allocator,
                buffer_usage,
                false,
                (0..size).map(|_| 0),
            )
            .unwrap(),
            size,
        }
    }

    pub fn read<'a>(&'a mut self) -> Result<SurfelModelReader<'a>, ReadLockError> {
        Ok(SurfelModelReader {
            position: self.position.read()?,
            normal: self.normal.read()?,
            color: self.color.read()?,
            radius: self.radius.read()?,
            confidence: self.confidence.read()?,
            age: self.age.read()?,
            free_list: &self.free_list,
        })
    }

    pub fn write<'a>(&'a mut self) -> Result<SurfelModelWriter<'a>, WriteLockError> {
        Ok(SurfelModelWriter {
            position: self.position.write()?,
            normal: self.normal.write()?,
            color: self.color.write()?,
            radius: self.radius.write()?,
            confidence: self.confidence.write()?,
            age: self.age.write()?,
            mask: self.mask.write()?,
            free_list: &mut self.free_list,
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }
}
