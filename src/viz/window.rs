use std::{sync::Arc, time::Instant};

use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents,
    },
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageUsage, SwapchainImage},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{graphics::viewport::Viewport, GraphicsPipeline},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::{
        acquire_next_image, AcquireError, Surface, Swapchain, SwapchainCreateInfo,
        SwapchainCreationError, SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::{Window as WWindow, WindowBuilder},
};

use super::{
    controllers::{FrameStepInfo, SceneState, VirtualCameraControl, WASDVirtualCameraControl},
    manager::Manager,
    node::CommandBuffersContext,
};
use super::{node::Node, virtual_camera::VirtualCameraSphericalBuilder};
use std::collections::HashMap;

pub struct Window {
    surface: Arc<Surface>,
    event_loop: Option<EventLoop<()>>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    scene: Arc<dyn Node>,
    command_buffer_allocator: StandardCommandBufferAllocator,
}

fn window_size_dependent_setup(
    memory_allocator: &StandardMemoryAllocator,
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].swapchain().image_extent();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(memory_allocator, dimensions, Format::D16_UNORM).unwrap(),
    )
    .unwrap();
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

impl Window {
    pub fn create(manager: &mut Manager, scene: Arc<dyn Node>) -> Self {
        let event_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            .build_vk_surface(&event_loop, manager.instance.clone())
            .unwrap();

        Self {
            surface,
            device: manager.device.clone(),
            queue: manager.queues.next().unwrap(),
            event_loop: Some(event_loop),
            scene: scene.clone(),
            command_buffer_allocator: StandardCommandBufferAllocator::new(
                manager.device.clone(),
                Default::default(),
            ),
        }
    }

    fn get_command_buffers(
        &self,
        framebuffer: Arc<Framebuffer>,
        viewport: &mut Viewport,
        pipelines: &mut HashMap<String, Arc<GraphicsPipeline>>,
        render_pass: Arc<RenderPass>,
        view_matrix: &nalgebra_glm::Mat4,
        projection_matrix: &nalgebra_glm::Mat4,
        window_state: &FrameStepInfo,
    ) -> PrimaryAutoCommandBuffer {
        // In order to draw, we have to build a *command buffer*. The command buffer object holds
        // the list of commands that are going to be executed.
        //
        // Building a command buffer is an expensive operation (usually a few hundred
        // microseconds), but it is known to be a hot path in the driver and is expected to be
        // optimized.
        //
        // Note that we have to pass a queue family when we create the command buffer. The command
        // buffer will only be executable on that given queue family.

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            // Before we can draw, we have to *enter a render pass*.
            .begin_render_pass(
                RenderPassBeginInfo {
                    // A list of values to clear the attachments with. This list contains
                    // one item for each attachment in the render pass. In this case,
                    // there is only one attachment, and we clear it with a blue color.
                    //
                    // Only attachments that have `LoadOp::Clear` are provided with clear
                    // values, any others should use `ClearValue::None` as the clear value.
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into()), Some(1f32.into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                // The contents of the first (and only) subpass. This can be either
                // `Inline` or `SecondaryCommandBuffers`. The latter is a bit more advanced
                // and is not covered here.
                SubpassContents::Inline,
            )
            .unwrap()
            // We are now inside the first subpass of the render pass. We add a draw command.
            //
            // The last two parameters contain the list of resources to pass to the shaders.
            // Since we used an `EmptyPipeline` object, the objects have to be `()`.
            .set_viewport(0, [viewport.clone()]);

        self.scene.collect_command_buffers(
            //&mut CommandBuffersContext {
            //    device: self.device.clone(),
            //    builder: &mut builder,
            //    pipelines: pipelines,
            //    render_pass: render_pass.clone(),
            //    object_matrix: nalgebra_glm::Mat4::identity(),
            //    view_matrix: view_matrix.clone(),
            //    projection_matrix: projection_matrix.clone(),
            //},
            &mut CommandBuffersContext::new(
                self.device.clone(),
                &mut builder,
                pipelines,
                render_pass,
                *view_matrix,
                *projection_matrix,
            ),
            window_state,
        );
        builder.end_render_pass().unwrap();

        // Finish building the command buffer by calling `build`.
        builder.build().unwrap()
    }

    pub fn show(&mut self, //: Arc<Self>
    ) {
        let dimensions = {
            let window = self
                .surface
                .object()
                .unwrap()
                .downcast_ref::<WWindow>()
                .unwrap();
            window.inner_size()
        };

        let (mut swapchain, images) = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&self.surface, Default::default())
                .unwrap();
            let image_format = Some(
                self.device
                    .physical_device()
                    .surface_formats(&self.surface, Default::default())
                    .unwrap()[0]
                    .0,
            );

            Swapchain::new(
                self.device.clone(),
                self.surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count,
                    image_format,
                    image_extent: dimensions.into(),
                    image_usage: ImageUsage {
                        color_attachment: true,
                        ..Default::default()
                    },
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions.width as f32, dimensions.height as f32],
            depth_range: 0.0..1.0,
        };

        let render_pass = vulkano::single_pass_renderpass!(
            self.device.clone(),
            attachments: {
                // `color` is a custom name we give to the first and only attachment.
                color: {
                    // `load: Clear` means that we ask the GPU to clear the content of this
                    // attachment at the start of the drawing.
                    load: Clear,
                    // `store: Store` means that we ask the GPU to store the output of the draw
                    // in the actual image. We could also ask it to discard the result.
                    store: Store,
                    // `format: <ty>` indicates the type of the format of the image. This has to
                    // be one of the types of the `vulkano::format` module (or alternatively one
                    // of your structs that implements the `FormatDesc` trait). Here we use the
                    // same format as the swapchain.
                    format: swapchain.image_format(),
                    // `samples: 1` means that we ask the GPU to use one sample to determine the value
                    // of each pixel in the color attachment. We could use a larger value (multisampling)
                    // for antialiasing. An example of this can be found in msaa-renderpass.rs.
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16_UNORM,
                    samples: 1,
                }
            },
            pass: {
                // We use the attachment named `color` as the one and only color attachment.
                color: [color],
                // No depth-stencil attachment is indicated with empty brackets.
                depth_stencil: {depth}
            }
        )
        .unwrap();

        let memory_allocator = StandardMemoryAllocator::new_default(self.device.clone());
        let mut framebuffers = window_size_dependent_setup(
            &memory_allocator,
            &images,
            render_pass.clone(),
            &mut viewport,
        );

        let mut recreate_swapchain = false;

        let mut previous_frame_end = Some(sync::now(self.device.clone()).boxed());
        let mut pipelines = HashMap::<String, Arc<GraphicsPipeline>>::new();

        let scene_sphere = self.scene.bounding_sphere();

        let mut camera_control = WASDVirtualCameraControl::new(
            VirtualCameraSphericalBuilder::fit(scene_sphere, std::f32::consts::FRAC_PI_2)
                .near_plane(0.05)
                .build(),
            0.05,
        );

        let mut window_state: FrameStepInfo = FrameStepInfo {
            viewport_size: [dimensions.width as f32, dimensions.height as f32],
            ..Default::default()
        };
        let scene_state: SceneState = SceneState {
            world_bounds: *self.scene.bounding_sphere(),
        };

        let event_loop = self.event_loop.take();
        let mut instant = Instant::now();
        event_loop
            .unwrap()
            .run_return(move |event, _, control_flow| {
                window_state.elapsed_time = instant.elapsed();
                match event {
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested,
                        ..
                    } => {
                        *control_flow = ControlFlow::Exit;
                    }
                    Event::WindowEvent {
                        event: WindowEvent::Resized(_),
                        ..
                    } => {
                        recreate_swapchain = true;
                    }
                    Event::WindowEvent {
                        event: WindowEvent::MouseInput { state, button, .. },
                        ..
                    } => {
                        window_state.mouse_state.insert(button, state);
                    }
                    Event::WindowEvent {
                        event: WindowEvent::CursorMoved { position, .. },
                        ..
                    } => {
                        camera_control.cursor_moved(
                            position.x as f64,
                            position.y as f64,
                            &window_state,
                            &scene_state,
                        );
                    }
                    Event::WindowEvent {
                        event: WindowEvent::KeyboardInput { input, .. },
                        ..
                    } => {
                        if let Some(vkeycode) = input.virtual_keycode {
                            window_state.keyboard_state.insert(vkeycode, input.state);
                        }
                        camera_control.key_event(&window_state, &scene_state);
                    }
                    Event::RedrawEventsCleared => {
                        // Do not draw frame when screen dimensions are zero.
                        // On Windows, this can occur from minimizing the application.
                        let window = self
                            .surface
                            .object()
                            .unwrap()
                            .downcast_ref::<WWindow>()
                            .unwrap();
                        let dimensions = window.inner_size();
                        if dimensions.width == 0 || dimensions.height == 0 {
                            return;
                        }

                        window_state.viewport_size =
                            [dimensions.width as f32, dimensions.height as f32];

                        // It is important to call this function from time to time, otherwise resources will keep
                        // accumulating and you will eventually reach an out of memory error.
                        // Calling this function polls various fences in order to determine what the GPU has
                        // already processed, and frees the resources that are no longer needed.
                        previous_frame_end.as_mut().unwrap().cleanup_finished();

                        // Whenever the window resizes we need to recreate everything dependent on the window size.
                        // In this example that includes the swapchain, the framebuffers and the dynamic state viewport.
                        if recreate_swapchain {
                            // Use the new dimensions of the window.

                            let (new_swapchain, new_images) =
                                match swapchain.recreate(SwapchainCreateInfo {
                                    image_extent: dimensions.into(),
                                    ..swapchain.create_info()
                                }) {
                                    Ok(r) => r,
                                    // This error tends to happen when the user is manually resizing the window.
                                    // Simply restarting the loop is the easiest way to fix this issue.
                                    Err(SwapchainCreationError::ImageExtentNotSupported {
                                        ..
                                    }) => return,
                                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                                };

                            swapchain = new_swapchain;
                            // Because framebuffers contains an Arc on the old swapchain, we need to
                            // recreate framebuffers as well.
                            framebuffers = window_size_dependent_setup(
                                &memory_allocator,
                                &new_images,
                                render_pass.clone(),
                                &mut viewport,
                            );
                            recreate_swapchain = false;
                        }

                        // Before we can draw on the output, we have to *acquire* an image from the swapchain. If
                        // no image is available (which happens if you submit draw commands too quickly), then the
                        // function will block.
                        // This operation returns the index of the image that we are allowed to draw upon.
                        //
                        // This function can block if no image is available. The parameter is an optional timeout
                        // after which the function call will return an error.
                        let (image_index, suboptimal, acquire_future) =
                            match acquire_next_image(swapchain.clone(), None) {
                                Ok(r) => r,
                                Err(AcquireError::OutOfDate) => {
                                    recreate_swapchain = true;
                                    return;
                                }
                                Err(e) => panic!("Failed to acquire next image: {:?}", e),
                            };

                        // acquire_next_image can be successful, but suboptimal. This means that the swapchain image
                        // will still work, but it may not display correctly. With some drivers this can be when
                        // the window resizes, but it may not cause the swapchain to become out of date.
                        if suboptimal {
                            recreate_swapchain = true;
                        };
                        let command_buffer = self.get_command_buffers(
                            framebuffers[image_index as usize].clone(),
                            &mut viewport,
                            &mut pipelines,
                            render_pass.clone(),
                            &camera_control.camera.matrix(),
                            &camera_control.projection_matrix(),
                            &window_state,
                        );

                        let future = previous_frame_end
                            .take()
                            .unwrap()
                            .join(acquire_future)
                            .then_execute(self.queue.clone(), command_buffer)
                            .unwrap()
                            // The color output is now expected to contain our triangle. But in order to show it on
                            // the screen, we have to *present* the image by calling `present`.
                            //
                            // This function does not actually present the image immediately. Instead it submits a
                            // present command at the end of the queue. This means that it will only be presented once
                            // the GPU has finished executing the command buffer that draws the triangle.
                            .then_swapchain_present(
                                self.queue.clone(),
                                SwapchainPresentInfo::swapchain_image_index(
                                    swapchain.clone(),
                                    image_index,
                                ),
                            )
                            .then_signal_fence_and_flush();

                        match future {
                            Ok(future) => {
                                previous_frame_end = Some(future.boxed());
                            }
                            Err(FlushError::OutOfDate) => {
                                recreate_swapchain = true;
                                previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                            }
                            Err(e) => {
                                panic!("Failed to flush future: {:?}", e);
                                // previous_frame_end = Some(sync::now(device.clone()).boxed());
                            }
                        }
                    }
                    _ => (),
                }
                instant = Instant::now();
            });
    }
}
