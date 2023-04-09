use std::sync::Arc;

use vulkano::{
    device::{
        physical::{PhysicalDeviceType, PhysicalDevice}, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, Queue, Features,
    },
    instance::{Instance, InstanceCreateInfo},
    VulkanLibrary, memory::allocator::StandardMemoryAllocator,
};

pub struct Manager {
    pub library: Arc<VulkanLibrary>,
    pub instance: Arc<Instance>,
    pub physical_device: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    pub queues: Box<dyn ExactSizeIterator<Item = Arc<Queue>>>,
    pub memory_allocator: StandardMemoryAllocator
}

impl Default for Manager {
    fn default() -> Self {
        let library = VulkanLibrary::new().expect("Vulkan is not supported by this system");
        let required_extensions = vulkano_win::required_extensions(&library);

        let instance = Instance::new(
            library.clone(),
            InstanceCreateInfo {
                application_name: Some("align3d".to_string()),
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("Failed to create Vulkan instance");

        let physical_device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .expect("Unable to enumerate physical devices")
            .filter(|p| {
                p.supported_extensions()
                    .contains(&physical_device_extensions)
            })
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    // Find the first first queue family that is suitable.
                    // If none is found, `None` is returned to `filter_map`,
                    // which disqualifies this physical device.
                    .position(|(_, q)| {
                        q.queue_flags.graphics
                            // && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|q| (p, q as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .expect("no device available");

        let (device, queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: physical_device_extensions,
                enabled_features: Features {
                    geometry_shader: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .expect("failed to create device");
        let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
        
        Self {
            library,
            instance,
            physical_device,
            device,
            queues: Box::new(queues),
            memory_allocator
        }
    }
}

impl Manager {
    pub fn device_name(&self) -> String {
        self.physical_device.properties().device_name.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[ignore]
    #[test]
    pub fn test_can_initialize() {
        let manager = Manager::default();
        println!("Using device {}", manager.device_name());
    }
}