use nalgebra::Vector3;
use ndarray::parallel::prelude::ParallelIterator;
use vulkano::memory::ExternalMemoryHandleTypes;

use std::sync::Arc;
use vulkano::buffer::subbuffer::BufferWriteGuard;
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

use super::Surfel;
use rayon::prelude::*;

/// Simplest allocator that just keeps a list of free indices.
struct SimpleAllocator {
    free_list: Vec<usize>,
    free_start: usize,
}

impl SimpleAllocator {
    /// Create a new allocator with the given pool size.
    pub fn new(pool_size: usize) -> Self {
        Self {
            free_list: (0..pool_size).collect(),
            free_start: 0,
        }
    }

    /// Allocate a new index.
    pub fn allocate(&mut self) -> Option<usize> {
        if self.free_start < self.free_list.len() {
            let index = self.free_list[self.free_start];
            self.free_start += 1;
            Some(index)
        } else {
            None
        }
    }

    /// Free an index.
    pub fn free(&mut self, index: usize) {
        self.free_start -= 1;
        self.free_list[self.free_start] = index;
    }
}

/// Surfel model raw data
struct SurfelData {
    pub position: Vec<Vector3<f32>>,
    pub normal: Vec<Vector3<f32>>,
    pub color: Vec<Vector3<u8>>,
    pub radius: Vec<f32>,
    pub confidence: Vec<f32>,
    pub age: Vec<u32>,
    pub mask: Vec<bool>,
}

impl SurfelData {
    /// Create a new surfel model with the given size.
    pub fn new(size: usize) -> Self {
        Self {
            position: vec![Vector3::zeros(); size],
            normal: vec![Vector3::zeros(); size],
            color: vec![Vector3::zeros(); size],
            radius: vec![0.0; size],
            confidence: vec![0.0; size],
            age: vec![0; size],
            mask: vec![false; size],
        }
    }
}

/// Vertex containg position and confidence.
#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct VkSurfelPositionConf {
    #[format(R32G32B32A32_SFLOAT)]
    pub position_confidence: [f32; 4],
}

impl VkSurfelPositionConf {
    pub fn new(position: Vector3<f32>, confidence: f32) -> Self {
        Self {
            position_confidence: [position.x, position.y, position.z, confidence],
        }
    }
}

/// Vertex containing normal and radius.
#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct VkSurfelNormalRadius {
    #[format(R32G32B32A32_SFLOAT)]
    pub normal_radius: [f32; 4],
}

impl VkSurfelNormalRadius {
    pub fn new(normal: Vector3<f32>, radius: f32) -> Self {
        Self {
            normal_radius: [normal.x, normal.y, normal.z, radius],
        }
    }
}

/// Vertex containing color, mask and age.
#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct VkSurfelColorMaskAge {
    #[format(R32G32_UINT)]
    pub rgbmask_age: [u32; 2],
}

impl VkSurfelColorMaskAge {
    pub fn new(r: u8, g: u8, b: u8, mask: u8, age: u32) -> Self {
        Self {
            rgbmask_age: [
                ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | (mask as u32),
                age as u32,
            ],
        }
    }

    pub fn into_parts(&self) -> (u8, u8, u8, u8, u32) {
        let rgbm = self.rgbmask_age[0];
        (
            ((rgbm >> 24) & 0xff) as u8,
            ((rgbm >> 16) & 0xff) as u8,
            ((rgbm >> 8) & 0xff) as u8,
            (rgbm & 0xff) as u8,
            self.rgbmask_age[1],
        )
    }

    pub fn mask(&self) -> u8 {
        (self.rgbmask_age[0] & 0xff) as u8
    }

    pub fn set_mask(&mut self, mask: u8) {
        self.rgbmask_age[0] = (self.rgbmask_age[0] & 0xffffff00) | (mask as u32);
    }
}

/// Vulkan surfel model buffers. We use three buffers for position/confidence, normal/radius and color, mask, and age.
/// We pack them together to reduce the number of buffers to pass to vulkano.
pub struct VkSurfelData {
    /// Vertex buffer containing position and confidence.
    pub position_conf: Subbuffer<[VkSurfelPositionConf]>,
    /// Vertex buffer containing normal and radius.
    pub normal_radius: Subbuffer<[VkSurfelNormalRadius]>,
    /// Vertex buffer containing color, mask and age.
    pub color_mask_age: Subbuffer<[VkSurfelColorMaskAge]>,
}

impl VkSurfelData {
    /// Create a new surfel model in Vulkano with the given size.
    /// The buffers are allocated using the given memory allocator.
    ///
    /// # Arguments
    ///
    /// * `memory_allocator` - The memory allocator to use for buffer allocation.
    /// * `size` - The number of the surfels.
    /// * `buffer_usage` - The buffer usage.
    /// * `memory_usage` - The memory usage.
    pub fn new(
        memory_allocator: &(impl MemoryAllocator + ?Sized),
        size: usize,
        buffer_usage: BufferUsage,
        memory_usage: MemoryUsage,
    ) -> Self {
        let create_info = BufferCreateInfo {
            usage: buffer_usage,
            external_memory_handle_types: ExternalMemoryHandleTypes::HOST_ALLOCATION,
            ..Default::default()
        };
        let alloc_info = AllocationCreateInfo {
            usage: memory_usage,
            ..Default::default()
        };

        Self {
            position_conf: Buffer::from_iter(
                memory_allocator,
                create_info.clone(),
                alloc_info.clone(),
                (0..size).map(|_| VkSurfelPositionConf::new(Vector3::zeros(), 0.0)),
            )
            .unwrap(),
            normal_radius: Buffer::from_iter(
                memory_allocator,
                create_info.clone(),
                alloc_info.clone(),
                (0..size).map(|_| VkSurfelNormalRadius::new(Vector3::zeros(), 0.0)),
            )
            .unwrap(),
            color_mask_age: Buffer::from_iter(
                memory_allocator,
                create_info.clone(),
                alloc_info.clone(),
                (0..size).map(|_| VkSurfelColorMaskAge::new(0, 0, 0, 0, 0)),
            )
            .unwrap(),
        }
    }
}

/// Use this struct to write surfels to the surfel model. It'll store/free surfels from
/// the CPU and GPU memory at the same time.
pub struct SurfelModelWriter<'a> {
    position_conf: BufferWriteGuard<'a, [VkSurfelPositionConf]>,
    normal_radius: BufferWriteGuard<'a, [VkSurfelNormalRadius]>,
    color_mask_age: BufferWriteGuard<'a, [VkSurfelColorMaskAge]>,
    model: &'a mut SurfelData,
    allocator: &'a mut SimpleAllocator,
}

impl<'a> SurfelModelWriter<'a> {
    /// Update the surfel at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the surfel to update.
    /// * `surfel` - The new surfel data.
    pub fn update(&mut self, index: usize, surfel: &Surfel) {
        self.model.position[index] = surfel.position;
        self.model.normal[index] = surfel.normal;
        self.model.color[index] = surfel.color;
        self.model.radius[index] = surfel.radius;
        self.model.age[index] = surfel.age;
        self.model.confidence[index] = surfel.confidence;
        self.model.mask[index] = true;

        self.position_conf[index] = VkSurfelPositionConf::new(surfel.position, surfel.confidence);
        self.normal_radius[index] = VkSurfelNormalRadius::new(surfel.normal, surfel.radius);
        self.color_mask_age[index] = VkSurfelColorMaskAge::new(
            surfel.color[0],
            surfel.color[1],
            surfel.color[2],
            1,
            surfel.age,
        );
    }

    /// Add a new surfel to the model. It'll allocate a new surfel and update it with the given surfel.
    ///
    /// # Arguments
    ///
    /// * `surfel` - The surfel to add.
    ///
    /// # Returns
    ///
    /// The index of the new surfel.
    ///
    /// # Panics
    ///
    /// If there is no more space in the model to allocate new buffer.
    pub fn add(&mut self, surfel: &Surfel) -> usize {
        let id = self.allocator.allocate().unwrap();
        self.update(id, surfel);
        id
    }

    /// Free the surfel at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the surfel to free.
    pub fn free(&mut self, index: usize) {
        self.color_mask_age[index].set_mask(0);
        self.model.mask[index] = false;
        self.allocator.free(index);
    }
}

/// A surfel model is a collection of surfels. It's used to store surfels on the CPU and GPU.
/// It stores on GPU for fast rendering and on CPU for fast processing.
/// It has two copies on GPU, one for rendering and one for copying into the rendering one.
pub struct SurfelModel {
    /// The surfel data ready to render on GPU.
    pub graphics: VkSurfelData,
    process: VkSurfelData,
    data: SurfelData,
    allocator: SimpleAllocator,
    size: usize,
}

impl<'b> SurfelModel {
    /// Create a new surfel model in Vulkano with the given size.
    pub fn new(memory_allocator: &(impl MemoryAllocator + ?Sized), size: usize) -> Self {
        SurfelModel {
            graphics: VkSurfelData::new(
                memory_allocator,
                size,
                BufferUsage::TRANSFER_DST | BufferUsage::VERTEX_BUFFER,
                MemoryUsage::Upload,
            ),
            process: VkSurfelData::new(
                memory_allocator,
                size,
                BufferUsage::TRANSFER_SRC,
                MemoryUsage::Upload,
            ),
            data: SurfelData::new(size),
            allocator: SimpleAllocator::new(size),
            size,
        }
    }

    /// Get a writer to write surfels to the model.
    pub fn write(&'b mut self) -> Result<SurfelModelWriter<'b>, BufferError> {
        Ok(SurfelModelWriter {
            position_conf: self.process.position_conf.write()?,
            normal_radius: self.process.normal_radius.write()?,
            color_mask_age: self.process.color_mask_age.write()?,
            model: &mut self.data,
            allocator: &mut self.allocator,
        })
    }

    /// Gets the total number of surfels that this model can store.
    pub fn capacity(&self) -> usize {
        self.size
    }

    /// Copies the latest processed model into the graphics model.
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
                self.process.position_conf.clone(),
                self.graphics.position_conf.clone(),
            ))
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(
                self.process.normal_radius.clone(),
                self.graphics.normal_radius.clone(),
            ))
            .unwrap()
            .copy_buffer(CopyBufferInfo::buffers(
                self.process.color_mask_age.clone(),
                self.graphics.color_mask_age.clone(),
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

    /// Gets the surfel at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the surfel to get.
    ///
    /// # Returns
    ///
    /// The surfel at the given index if it exists, `None` otherwise.
    pub fn get(&self, index: usize) -> Option<Surfel> {
        if !self.data.mask[index] {
            return None;
        }

        Some(Surfel {
            position: self.data.position[index],
            normal: self.data.normal[index],
            color: self.data.color[index],
            radius: self.data.radius[index],
            confidence: self.data.confidence[index],
            age: self.data.age[index],
        })
    }

    /// Iterator over the allocated surfel position.
    pub fn position_iter<'a>(&'a self) -> impl Iterator<Item = (usize, Vector3<f32>)> + 'a {
        self.data
            .position
            .iter()
            .zip(self.data.mask.iter())
            .enumerate()
            .filter_map(move |(i, (p, m))| if *m { Some((i, *p)) } else { None })
    }

    /// Parallel iterator over the allocated surfel position.
    pub fn position_par_iter<'a>(
        &'a self,
    ) -> impl ParallelIterator<Item = (usize, Vector3<f32>)> + 'a {
        self.data
            .position
            .iter()
            .zip(self.data.mask.iter())
            .enumerate()
            .par_bridge()
            .filter_map(
                move |(i, (p, mask))| {
                    if *mask {
                        Some((i, *p))
                    } else {
                        None
                    }
                },
            )
    }

    /// Iterator over the allocated age and confidence of surfels.
    pub fn age_confidence_iter<'a>(&'a self) -> impl Iterator<Item = (usize, u32, f32)> + 'a {
        self.data
            .mask
            .iter()
            .zip(self.data.age.iter())
            .zip(self.data.confidence.iter())
            .enumerate()
            .filter_map(
                move |(i, ((mask, age), conf))| {
                    if *mask {
                        Some((i, *age, *conf))
                    } else {
                        None
                    }
                },
            )
    }
}
