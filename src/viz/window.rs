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
    event::{ElementState, Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::{Window as WWindow, WindowBuilder},
};

use super::{
    controllers::{FrameStepInfo, SceneState, VirtualCameraControl, WASDVirtualCameraControl},
    manager::Manager,
    node::{CommandBuffersContext, NodeRef},
};
use super::{node::Node, virtual_camera::VirtualCameraSphericalBuilder};
use std::collections::HashMap;

pub type KeyCallback = Box<dyn FnMut(VirtualKeyCode, &FrameStepInfo)>;
pub struct Window {
    surface: Arc<Surface>,
    event_loop: Option<EventLoop<()>>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    scene: NodeRef<dyn Node>,
    command_buffer_allocator: StandardCommandBufferAllocator,
    pub on_key: Option<KeyCallback>,
    frame_counter: usize,
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
    pub fn create(manager: &mut Manager, scene: NodeRef<dyn Node>) -> Self {
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
            on_key: None,
            frame_counter: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
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
        // Builds the command buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Render pass
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into()), Some(1f32.into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassContents::Inline,
            )
            .unwrap()
            .set_viewport(0, [viewport.clone()]);

        (*self.scene).borrow().collect_command_buffers(
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

    pub fn show(&mut self) {
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
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.image_format(),
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
                color: [color],
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

        let scene_sphere = (self.scene).borrow().properties().get_bounding_sphere();

        let mut camera_control = WASDVirtualCameraControl::new(
            VirtualCameraSphericalBuilder::fit(&scene_sphere, std::f32::consts::FRAC_PI_2)
                .near_plane(0.05)
                .build(),
            0.05,
        );

        let mut window_state: FrameStepInfo = FrameStepInfo {
            viewport_size: [dimensions.width as f32, dimensions.height as f32],
            ..Default::default()
        };
        let scene_state: SceneState = SceneState {
            world_bounds: scene_sphere,
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
                            position.x,
                            position.y,
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

                        if let (Some(on_key), Some(vkeycode), ElementState::Pressed) =
                            (self.on_key.as_mut(), input.virtual_keycode, input.state)
                        {
                            on_key(vkeycode, &window_state);
                        }
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

                        // Clean up resources from the previous frame.
                        previous_frame_end.as_mut().unwrap().cleanup_finished();

                        // Swap chain recreation
                        if recreate_swapchain {
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
                                    Err(e) => panic!("Failed to recreate swapchain: {e:?}"),
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

                        let (image_index, suboptimal, acquire_future) =
                            match acquire_next_image(swapchain.clone(), None) {
                                Ok(r) => r,
                                Err(AcquireError::OutOfDate) => {
                                    recreate_swapchain = true;
                                    return;
                                }
                                Err(e) => panic!("Failed to acquire next image: {e:?}"),
                            };

                        if suboptimal {
                            recreate_swapchain = true;
                        };

                        let mut builder = AutoCommandBufferBuilder::primary(
                            &self.command_buffer_allocator,
                            self.queue.queue_family_index(),
                            CommandBufferUsage::OneTimeSubmit,
                        )
                        .unwrap();
                        let command_buffer = builder.build().unwrap();

                        let future = previous_frame_end
                        .take()
                        .unwrap()
                        .join(acquire_future)
                        .then_execute(self.queue.clone(), command_buffer)
                        .unwrap()
                        .then_signal_fence_and_flush();

                        let command_buffer = self.get_command_buffers(
                            framebuffers[image_index as usize].clone(),
                            &mut viewport,
                            &mut pipelines,
                            render_pass.clone(),
                            &camera_control.camera.matrix(),
                            &camera_control.projection_matrix(),
                            &window_state,
                        );

                        let future = Some(future.unwrap().boxed())
                            .take()
                            .unwrap()
                            //.join(acquire_future)
                            .then_execute(self.queue.clone(), command_buffer)
                            .unwrap()
                            .then_swapchain_present(
                                self.queue.clone(),
                                SwapchainPresentInfo::swapchain_image_index(
                                    swapchain.clone(),
                                    image_index,
                                ),
                            )
                            .then_signal_fence_and_flush();
                        self.frame_counter += 1;
                            
                        match future {
                            Ok(future) => {
                                previous_frame_end = Some(future.boxed());
                            }
                            Err(FlushError::OutOfDate) => {
                                recreate_swapchain = true;
                                previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                            }
                            Err(e) => {
                                panic!("Failed to flush future: {e:?}");
                            }
                        }
                    }
                    _ => (),
                }
                instant = Instant::now();
            });
    }
}
