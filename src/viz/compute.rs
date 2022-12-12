use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Features, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};
use vulkano::VulkanLibrary;

fn smain() {
    let library = VulkanLibrary::new().expect("Vulkan not supported by the system");
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            application_name: Some("align3d::Viz Window".to_string()),
            ..Default::default()
        },
    )
    .expect("Failed to create Vulkan instance");

    let physical_device = instance
        .enumerate_physical_devices()
        .expect("Unable to enumerate physical devices")
        .next()
        .expect("No vulkan device available");

    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_, queue)| queue.queue_flags.graphics)
        .expect("No graphical queue queue family found") as u32;

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: DeviceExtensions {
                khr_storage_buffer_storage_class: true,
                ..DeviceExtensions::empty()
            },
            ..Default::default()
        },
    )
    .expect("failed to create device");

    let queue = queues.next().unwrap();

    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    let data_buffer = CpuAccessibleBuffer::from_iter(
        &memory_allocator,
        BufferUsage {
            storage_buffer: true,
            ..Default::default()
        },
        false,
        0..100,
    )
    .expect("Unable to create data buffer");
    let shader = cs::load(device.clone()).expect("Unable to create shader");
    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .expect("Failed to create compute pipeline");

    let descriptor_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        &descriptor_allocator,
        layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())],
    )
    .unwrap();

    let cmd_allocator = StandardCommandBufferAllocator::new(device.clone(), Default::default());
    let mut builder = AutoCommandBufferBuilder::primary(
        &cmd_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("Unable to create cmd builder");

    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            set,
        )
        .dispatch([100, 1, 1])
        .unwrap();

    let command_buffer = builder.build().unwrap();
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        println!("{} {}", n, val);
    }
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
    uint data[];
} buf;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    buf.data[idx] *= 12;
}"
    }
}

#[cfg(test)]
mod tests {
    #[test]
    pub fn show_window() {
        use super::smain;
        smain();

        let device;
        let pcl: VkPointCloud = pcl.from();
        
        let context = RenderingContext::from(device);

        context.scene.add_node(pcl).transform();
        
        window = context.create_window();

    }
}
