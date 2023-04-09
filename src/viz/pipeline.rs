struct Pipeline {
    allocator: Arc<StandardMemoryAllocator>,
    queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    subpass: Subpass,
    intermediary: Arc<ImageView<AttachmentImage>>,
    command_buffer_allocator: StandardCommandBufferAllocator,
    pipelines: HashMap::<String, Arc<GraphicsPipeline>>;
}

impl Pipeline {
    pub fn create(manager: &mut Manager, scene: NodeRef<dyn Node>) -> Self {
        let event_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            .build_vk_surface(&event_loop, manager.instance.clone())
            .unwrap();

        Self {
            device: manager.device.clone(),
            queue: manager.queues.next().unwrap(),
            scene: scene.clone(),
            command_buffer_allocator: StandardCommandBufferAllocator::new(
                manager.device.clone(),
                Default::default(),
            ),
            pipelines: HashMap::<String, Arc<GraphicsPipeline>>::new()
        }
    }

    pub fn get_command_buffers(&self) {
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

    pub fn get_render_pass(&self) {
        Arc::new(
            vulkano::single_pass_renderpass!(
                self.device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: self.swapchain.format(),
                        samples: 1,
                    },
                    depth: {
                        load: Clear,
                        store: DontCare,
                        format: Format::D16Unorm,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {depth}
                }
            )
            .unwrap(),
        )
    }
}

