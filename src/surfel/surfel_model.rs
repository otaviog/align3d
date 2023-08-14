use std::sync::Arc;

use nalgebra::Vector3;
use ndarray::parallel::prelude::ParallelIterator;
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
use rayon::prelude::*;

use vulkano::buffer::subbuffer::BufferWriteGuard;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferError};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::device::{Device, Queue};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage};
use vulkano::memory::ExternalMemoryHandleTypes;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::sync::{self, GpuFuture};
use vulkano::{
    buffer::{BufferUsage, Subbuffer},
    memory::allocator::MemoryAllocator,
};

use super::Surfel;

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

/// Use this struct to write surfels in CPU.
pub struct CpuSurfelWriter<'model> {
    model: &'model mut SurfelData,
    allocator: &'model mut SimpleAllocator,
}

impl<'model> CpuSurfelWriter<'model> {
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
        self.model.mask[index] = false;
        self.allocator.free(index);
    }
}

/// Vertex containing position and confidence. We store both in the same vertex to save space.
#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct VkSurfelPositionConf {
    /// Position and confidence.
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

/// Vertex containing normal and radius. We store both in the same vertex to save space.
#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct VkSurfelNormalRadius {
    /// Normal and radius.
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
                age,
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
                create_info,
                alloc_info,
                (0..size).map(|_| VkSurfelColorMaskAge::new(0, 0, 0, 0, 0)),
            )
            .unwrap(),
        }
    }

    /// Create a writer to update the surfel data in GPU.
    pub fn writer(&'_ mut self) -> Result<VkSurfelWriter<'_>, BufferError> {
        Ok(VkSurfelWriter {
            position_conf: self.position_conf.write()?,
            normal_radius: self.normal_radius.write()?,
            color_mask_age: self.color_mask_age.write()?,
        })
    }
}

/// Writer to update the surfel data in GPU.
pub struct VkSurfelWriter<'buffer> {
    position_conf: BufferWriteGuard<'buffer, [VkSurfelPositionConf]>,
    normal_radius: BufferWriteGuard<'buffer, [VkSurfelNormalRadius]>,
    color_mask_age: BufferWriteGuard<'buffer, [VkSurfelColorMaskAge]>,
}

impl<'buffer> VkSurfelWriter<'buffer> {
    /// Update the surfel at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the surfel to update.
    /// * `surfel` - The new surfel data.
    pub fn update(&mut self, index: usize, surfel: &Surfel) {
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

    /// Free the surfel at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the surfel to free.
    pub fn unmask(&mut self, index: usize) {
        self.color_mask_age[index].set_mask(0);
    }
}

/// Vulkan surfel model storage. We use two buffers for graphics and processing.
pub struct VkSurfelStorage {
    /// Surfel data for rendering; this is the data that is currently being rendered.
    pub graphics: VkSurfelData,
    // This is for processing; this is the data that is currently being processed.
    process: VkSurfelData,
}

impl VkSurfelStorage {
    pub fn new(memory_allocator: &(impl MemoryAllocator + ?Sized), size: usize) -> Self {
        Self {
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
        }
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
        let future = sync::now(device)
            .then_execute(queue, command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
    }

    pub fn capacity(&self) -> usize {
        self.process.position_conf.len() as usize
    }

    pub fn get_writer(&mut self) -> VkSurfelWriter {
        VkSurfelWriter {
            position_conf: self.process.position_conf.write().unwrap(),
            normal_radius: self.process.normal_radius.write().unwrap(),
            color_mask_age: self.process.color_mask_age.write().unwrap(),
        }
    }
}

/// A surfel model is a collection of surfels. It's used to store surfels on the CPU and GPU.
/// It stores on GPU for fast rendering and on CPU for fast processing.
/// It has two copies on GPU, one for rendering and one for copying into the rendering one.
pub struct SurfelModel {
    /// The surfel data ready to render on GPU.
    pub vk_data: Arc<Mutex<VkSurfelStorage>>,
    data: SurfelData,
    allocator: SimpleAllocator,
    size: usize,
}

impl SurfelModel {
    /// Create a new surfel model in Vulkano with the given size.
    pub fn new(memory_allocator: &(impl MemoryAllocator + ?Sized), size: usize) -> Self {
        SurfelModel {
            vk_data: Arc::new(Mutex::new(VkSurfelStorage::new(memory_allocator, size))),
            data: SurfelData::new(size),
            allocator: SimpleAllocator::new(size),
            size,
        }
    }

    /// Gets a writer to update the surfel data on CPU.
    pub fn get_cpu_writer(&'_ mut self) -> CpuSurfelWriter<'_> {
        CpuSurfelWriter {
            model: &mut self.data,
            allocator: &mut self.allocator,
        }
    }

    /// Gets a writer to update the surfel data on GPU.
    pub fn lock_gpu(&self) -> MappedMutexGuard<VkSurfelStorage> {
        MutexGuard::map(self.vk_data.lock(), |vk_data| vk_data)
    }

    /// Gets the total number of surfels that this model can store.
    pub fn capacity(&self) -> usize {
        self.size
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
    pub fn position_iter(&'_ self) -> impl Iterator<Item = (usize, Vector3<f32>)> + '_ {
        self.data
            .position
            .iter()
            .zip(self.data.mask.iter())
            .enumerate()
            .filter_map(move |(i, (p, m))| if *m { Some((i, *p)) } else { None })
    }

    /// Parallel iterator over the allocated surfel position.
    pub fn position_par_iter(&'_ self) -> impl ParallelIterator<Item = (usize, Vector3<f32>)> + '_ {
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
    pub fn age_confidence_iter(&'_ self) -> impl Iterator<Item = (usize, u32, f32)> + '_ {
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
