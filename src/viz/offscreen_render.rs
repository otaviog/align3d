use std::{cell::RefCell, collections::HashMap, rc::Rc, sync::Arc};

use image::{ImageBuffer, Rgba, RgbaImage};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyImageToBufferInfo, RenderPassBeginInfo, SubpassContents,
    },
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, ImageDimensions, StorageImage},
    memory::allocator::{FreeListAllocator, GenericMemoryAllocator, StandardMemoryAllocator},
    pipeline::{graphics::viewport::Viewport, GraphicsPipeline},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    sync,
    sync::GpuFuture,
};

use super::{
    controllers::FrameStepInfo,
    node::{CommandBuffersContext, Node},
    Manager,
};

/// Renders nodes into images instead of a window.
pub struct OffscreenRenderer {
    pub device: Arc<Device>,
    pub pipelines: HashMap<String, Arc<GraphicsPipeline>>,
    pub render_pass: Arc<RenderPass>,
    pub queue: Arc<Queue>,
    pub framebuffer: Arc<Framebuffer>,
    pub memory_allocator: GenericMemoryAllocator<Arc<FreeListAllocator>>,
    framebuffer_image: Arc<StorageImage>,
    viewport: Viewport,
    command_buffer_allocator: StandardCommandBufferAllocator,
}

/// Render result image. This class holds a GPU buffer accessible from CPU
/// which can be mapped or converted into image.
pub struct RenderImage {
    image_buffer: Arc<CpuAccessibleBuffer<[u8]>>,
    width: u32,
    height: u32,
}

impl OffscreenRenderer {
    /// Builds a offscreen renderer
    ///
    /// # Arguments
    ///
    /// * `manager`: Manager to this renderer be associated.
    /// * `width`: Output image width.
    /// * `height`: Output image height.
    pub fn new(manager: &mut Manager, width: usize, height: usize) -> Self {
        let queue = manager.queues.next().unwrap();
        let memory_allocator = StandardMemoryAllocator::new_default(manager.device.clone());
        let (render_pass, framebuffer_image, framebuffer) = {
            let render_pass = vulkano::single_pass_renderpass!(
                manager.device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: Format::R8G8B8A8_UNORM,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            )
            .unwrap();

            let image = StorageImage::new(
                &memory_allocator,
                ImageDimensions::Dim2d {
                    width: width as u32,
                    height: height as u32,
                    array_layers: 1,
                },
                Format::R8G8B8A8_UNORM,
                Some(queue.queue_family_index()),
            )
            .unwrap();
            let view = ImageView::new_default(image.clone()).unwrap();
            let framebuffer = Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap();
            (render_pass, image, framebuffer)
        };

        Self {
            device: manager.device.clone(),
            pipelines: HashMap::<String, Arc<GraphicsPipeline>>::new(),
            render_pass,
            queue,
            framebuffer,
            framebuffer_image,
            viewport: Viewport {
                origin: [0.0, 0.0],
                dimensions: [width as f32, height as f32],
                depth_range: 0.0..1.0,
            },
            memory_allocator,
            command_buffer_allocator: StandardCommandBufferAllocator::new(
                manager.device.clone(),
                Default::default(),
            ),
        }
    }

    /// Draws the scene into a image
    ///
    /// # Arguments
    ///
    /// * `scene`: Target scene
    ///
    /// # Returns
    ///
    /// * A RenderImage object that contains a Vulkan buffer that can
    /// be transformed or copied into an image.
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    pub fn render(&mut self, scene: Rc<RefCell<dyn Node>>) -> RenderImage {
        let (width, height) = (
            self.viewport.dimensions[0] as usize,
            self.viewport.dimensions[1] as usize,
        );
        let image_buffer = CpuAccessibleBuffer::from_iter(
            &self.memory_allocator,
            BufferUsage {
                transfer_dst: true,
                ..Default::default()
            },
            false,
            (0..height * width * 4).map(|_| 0u8),
        )
        .expect("failed to create buffer");

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(self.framebuffer.clone())
                },
                SubpassContents::Inline,
            )
            .unwrap()
            .set_viewport(0, [self.viewport.clone()]);

        scene.borrow().collect_command_buffers(
            &mut CommandBuffersContext {
                device: self.device.clone(),
                builder: &mut builder,
                pipelines: &mut self.pipelines,
                render_pass: self.render_pass.clone(),
                view_normals_matrix: nalgebra_glm::Mat3::identity(),
                view_matrix: nalgebra_glm::Mat4::identity(),
                projection_matrix: nalgebra_glm::Mat4::identity(),
            },
            &FrameStepInfo::new(self.viewport.dimensions),
        );

        builder
            .end_render_pass()
            .unwrap()
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                self.framebuffer_image.clone(),
                image_buffer.clone(),
            ))
            .unwrap();

        let command_buffer = builder.build().unwrap();
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        RenderImage {
            image_buffer,
            width: width as u32,
            height: height as u32,
        }
    }
}

impl RenderImage {
    /// Returns a copy of the buffer into a RGBA Image.
    pub fn to_image(&self) -> RgbaImage {
        let image_buffer = self.image_buffer.read().unwrap();

        RgbaImage::from_fn(self.width, self.height, |x, y| {
            let offset = (y * self.height + x) as usize;
            Rgba::<u8>([
                image_buffer[offset],
                image_buffer[offset + 1],
                image_buffer[offset + 2],
                image_buffer[offset + 3],
            ])
        })
    }

    /// Maps without copying the buffer into an image.
    ///
    /// # Arguments
    ///
    /// * `f`: Function that uses the image.
    pub fn map<F>(&self, f: F)
    where
        F: Fn(ImageBuffer<Rgba<u8>, &[u8]>),
    {
        let image_buffer = self.image_buffer.read().unwrap();
        f(ImageBuffer::from_raw(self.width, self.height, &image_buffer[..]).unwrap());
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::viz::{geometry::sample_nodes::teapot_node, Manager};

    use super::OffscreenRenderer;
    use crate::viz::unit_test::vk_manager;

    #[rstest]
    pub fn test_render(mut vk_manager: Manager) {
        let mut renderer = OffscreenRenderer::new(&mut vk_manager, 1024, 768);

        let image = renderer.render(teapot_node(&vk_manager));
        assert_eq!(image.width, 1024);
        assert_eq!(image.height, 768);

        let owned_image = image.to_image();
        assert_eq!(owned_image.width(), 1024);
        assert_eq!(owned_image.height(), 768);

        image.map(|image| {
            assert_eq!(image.width(), 1024);
            assert_eq!(image.height(), 768);
        })
    }
}
