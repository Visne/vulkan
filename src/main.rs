mod shaders;

use std::default::Default;
use std::sync::Arc;
use std::time::Instant;

use cgmath::{Angle, Matrix4, Rad};
use default::default;
use gltf::mesh::util::{ReadColors, ReadIndices, ReadPositions};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents, SubpassEndInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
#[cfg(debug_assertions)]
use vulkano::instance::debug::{DebugUtilsMessengerCreateInfo, ValidationFeatureEnable};
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex as VertexTrait, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::EntryPoint;
use vulkano::swapchain::{
    acquire_next_image, PresentMode, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::{single_pass_renderpass, sync, Validated, VulkanError, VulkanLibrary};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use crate::shaders::{fragment_shader, vertex_shader};

#[allow(clippy::too_many_lines)]
fn main() {
    let event_loop = EventLoop::new().expect("Should create event loop");
    let instance = get_instance(&event_loop);

    let (physical_device, device, mut queues) = get_device(&instance);
    let (window, surface, mut viewport) = create_window(&instance, &event_loop);

    //let format = physical_device.surface_formats(&surface, default()).unwrap()[0].0;
    let format = Format::B8G8R8A8_SRGB; // TODO: Pick best format

    let (mut swapchain, images) =
        get_swapchain(&physical_device, &device, &surface, &window, format);
    let render_pass = get_render_pass(device.clone(), &swapchain);

    let vs = vertex_shader::load(device.clone()).unwrap().entry_point("main").unwrap();
    let fs = fragment_shader::load(device.clone()).unwrap().entry_point("main").unwrap();

    let pipeline = get_pipeline(device.clone(), render_pass.clone(), viewport.clone(), vs, fs);
    let queue = queues.next().expect("There should be exactly one queue");

    let mut bad_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let command_buffer_allocator = StandardCommandBufferAllocator::new(device.clone(), default());
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone(), default());

    let mut framebuffers = get_framebuffers(&images, &render_pass, memory_allocator.clone());

    let uniform_buffer = SubbufferAllocator::new(
        memory_allocator.clone(),
        SubbufferAllocatorCreateInfo {
            buffer_usage: BufferUsage::UNIFORM_BUFFER,
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..default()
        },
    );

    // let mut rng = rand::thread_rng();
    // let vertices = vec![
    //     Vertex { position: [-0.5, rng.gen::<f32>() * 2. - 1.], color: [1., 0., 0.] },
    //     Vertex { position: [0.0, rng.gen::<f32>() * 2. - 1.], color: [0., 1., 0.] },
    //     Vertex { position: [0.5, rng.gen::<f32>() * 2. - 1.], color: [0., 0., 1.] },
    // ];

    // let vertices = vec![
    //     Vertex { position: [-0.5, 0.5, 0.], color: [1., 0., 0.] },
    //     Vertex { position: [0.0, -0.5, 0.], color: [0., 1., 0.] },
    //     Vertex { position: [0.5, 0.5, 0.], color: [0., 0., 1.] },
    //     Vertex { position: [1., -0.5, 0.], color: [0., 0., 1.] },
    // ];

    let (document, buffers, _images) =
        gltf::import("resources/models/teapot_smile.gltf").expect("Model should be imported");
    let primitives = document.meshes().next().unwrap().primitives().next().unwrap();

    let reader = primitives.reader(|b| Some(buffers[b.index()].0.as_slice()));

    // let pos_index = primitives.attributes().find(|a| a.0 == Semantic::Positions).unwrap().1.index();
    // let indices_accessor = primitives.indices().unwrap();
    //
    // let vertices = buffers;

    let colors;
    let positions;
    let indices;

    if let ReadColors::RgbaU16(c) = reader.read_colors(0).unwrap() {
        colors = c.map(|x| x.map(|a| (a / u16::MAX) as f32));
    } else {
        unimplemented!();
    }
    if let ReadPositions::Standard(p) = reader.read_positions().unwrap() {
        positions = p;
    } else {
        unimplemented!();
    }
    if let ReadIndices::U16(i) = reader.read_indices().unwrap() {
        indices = i;
    } else {
        unimplemented!();
    }

    let color_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::VERTEX_BUFFER, ..default() },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..default()
        },
        colors,
    )
    .unwrap();

    let position_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::VERTEX_BUFFER, ..default() },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..default()
        },
        positions,
    )
    .unwrap();

    let index_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::INDEX_BUFFER, ..default() },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..default()
        },
        indices,
    )
    .unwrap();

    let rotation_start = Instant::now();

    event_loop.set_control_flow(ControlFlow::Poll);

    event_loop
        .run(move |event, target| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => target.exit(),

                WindowEvent::Resized(_) => {
                    bad_swapchain = true;
                    viewport.extent = window.inner_size().into();
                }

                _ => {}
            },

            Event::AboutToWait => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if bad_swapchain {
                    bad_swapchain = false;

                    let dimensions = window.inner_size().into();
                    let (new_swapchain, new_images) = swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: dimensions,
                            ..swapchain.create_info()
                        })
                        .expect("Failed to recreate swapchain: {e}");

                    swapchain = new_swapchain;

                    framebuffers =
                        get_framebuffers(&new_images, &render_pass, memory_allocator.clone());
                }

                #[allow(unused_variables)] // TODO: Remove after RustRover bug is fixed
                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            // Recreate swapchain right away
                            bad_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {e}"),
                    };

                if suboptimal {
                    // Recreate swapchain next frame
                    bad_swapchain = true;
                }

                let uniform_buffer_subbuffer = {
                    let elapsed = rotation_start.elapsed();
                    let rotation =
                        elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
                    let theta = Rad(rotation as f32);

                    let (s, c) = Rad::sin_cos(theta);
                    // #[rustfmt::skip]
                    // let model = Matrix4::new(
                    //     c,  0., -s, 0.,
                    //     0., 1., 0., 0.,
                    //     s,  0., c,  0.,
                    //     0.0, 0.0, 0.0, 1.,
                    // );

                    #[rustfmt::skip]
                    let model = Matrix4::new(
                        s, 0., c, 0.,
                        0., 1., 0., 0.,
                        -c, 0., s, 0.,
                        0., 1.5, -6.5, 1.,
                    );

                    #[rustfmt::skip]
                    let fixrot = Matrix4::new(
                        1., 0., 0., 0.,
                        0., 0., -1., 0.,
                        0., 1., 0., 0.,
                        0., 0., 0., 1.,
                    );

                    let model = model * fixrot;

                    // note: this teapot was meant for OpenGL where the origin is at the lower left
                    //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis

                    let aspect_ratio =
                        swapchain.image_extent()[0] as f32 / swapchain.image_extent()[1] as f32;

                    let vertical_fov = Rad(std::f32::consts::FRAC_PI_2);
                    let near = 0.01;
                    let far = 100.0;

                    let f = Rad::cot(vertical_fov / 2.);
                    let fa = f / aspect_ratio;
                    let f1 = (far + near) / (near - far);
                    let f2 = (2. * far * near) / (near - far);

                    #[rustfmt::skip]
                    let projection: Matrix4<f32> = Matrix4::new(
                        -fa, 0., 0., 0.,
                        0., f, 0., 0.,
                        0., 0., f1, -1.,
                        0., 0., f2, 0.,
                    );

                    // let view = Matrix4::look_at_rh(
                    //     Point3::new(0.3, 0.3, 1.0),
                    //     Point3::new(0.0, 0.0, 0.0),
                    //     Vector3::new(0.0, -1.0, 0.0),
                    // );
                    // let scale = Matrix4::from_scale(0.01);

                    let uniform_data = vertex_shader::Data {
                        //world: Matrix4::from(rotation).into(),
                        //view: (view * scale).into(),
                        model: model.into(),
                        projection: projection.into(),
                    };

                    let subbuffer = uniform_buffer.allocate_sized().unwrap();
                    *subbuffer.write().unwrap() = uniform_data;

                    subbuffer
                };

                let layout = pipeline.layout().set_layouts().get(0).unwrap();

                let set = PersistentDescriptorSet::new(
                    &descriptor_set_allocator,
                    layout.clone(),
                    [WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer)],
                    [],
                )
                .unwrap();

                let command_buffer = get_command_buffer(
                    &command_buffer_allocator,
                    &queue,
                    &pipeline,
                    &framebuffers[image_index as usize],
                    &viewport,
                    set,
                    &position_buffer,
                    &color_buffer,
                    &index_buffer,
                );

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer.clone())
                    .unwrap()
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => previous_frame_end = Some(future.boxed()),
                    Err(VulkanError::OutOfDate) => {
                        bad_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {e}");
                    }
                };
            }

            _ => {}
        })
        .expect("Event loop should not error");
}

#[cfg(debug_assertions)]
fn get_debug_messenger_info() -> Vec<DebugUtilsMessengerCreateInfo> {
    use colored::Colorize;
    use vulkano::instance::debug::{
        DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessengerCallback,
        DebugUtilsMessengerCallbackData,
    };

    let user_callback = |sev, ty, data: DebugUtilsMessengerCallbackData| {
        println!(
            "[{severity} {ty}][{prefix:.14}] {desc}",
            severity = match sev {
                DebugUtilsMessageSeverity::ERROR => "ERR".red(),
                DebugUtilsMessageSeverity::WARNING => "WRN".yellow(),
                DebugUtilsMessageSeverity::INFO => "INF".blue(),
                DebugUtilsMessageSeverity::VERBOSE => "VRB".white(),
                _ => unimplemented!("Add type"),
            },
            ty = match ty {
                DebugUtilsMessageType::GENERAL => "GNRL",
                DebugUtilsMessageType::VALIDATION => "VLDN",
                DebugUtilsMessageType::PERFORMANCE => "PERF",
                _ => unimplemented!("Add type"),
            }
            .white(),
            prefix = data.message_id_name.unwrap_or(""),
            desc = data.message,
        );
    };

    let message_severity = DebugUtilsMessageSeverity::ERROR
        | DebugUtilsMessageSeverity::WARNING
        | DebugUtilsMessageSeverity::INFO
        | DebugUtilsMessageSeverity::VERBOSE;

    let message_type = DebugUtilsMessageType::GENERAL
        | DebugUtilsMessageType::PERFORMANCE
        | DebugUtilsMessageType::VALIDATION;

    unsafe {
        vec![DebugUtilsMessengerCreateInfo {
            message_severity,
            message_type,
            ..DebugUtilsMessengerCreateInfo::user_callback(DebugUtilsMessengerCallback::new(
                user_callback,
            ))
        }]
    }
}

fn get_device(
    instance: &Arc<Instance>,
) -> (Arc<PhysicalDevice>, Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>> + Sized + Sized) {
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        khr_shader_non_semantic_info: true,
        ..DeviceExtensions::empty()
    };

    let physical_device = instance
        .enumerate_physical_devices()
        .expect("Could not enumerate physical devices")
        .find(|p| {
            p.supported_extensions().contains(&device_extensions)
                && p.properties().device_type == PhysicalDeviceType::DiscreteGpu
            // TODO: Pick best by type
        })
        .expect("No discrete devices available");

    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_, properties)| properties.queue_flags.contains(QueueFlags::GRAPHICS))
        .expect("Should find a graphical queue family with the specified queue flags")
        .try_into()
        .expect("Should fit into u32");

    let (device, queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo { queue_family_index, ..default() }],
            ..default()
        },
    )
    .expect("Should successfully create device");

    (physical_device, device, queues)
}

fn get_swapchain(
    physical_device: &Arc<PhysicalDevice>,
    device: &Arc<Device>,
    surface: &Arc<Surface>,
    window: &Arc<Window>,
    format: Format,
) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
    let caps = physical_device
        .surface_capabilities(surface, default())
        .expect("Should get surface capabilities");
    let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();

    assert!(device.enabled_extensions().khr_swapchain);
    Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count + 1,
            image_format: format,
            image_extent: window.inner_size().into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha,
            present_mode: PresentMode::Mailbox, // TODO: Mailbox might not be supported
            ..default()
        },
    )
    .expect("Should successfully create swapchain")
}

fn get_command_buffer(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: &Arc<Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffer: &Arc<Framebuffer>,
    viewport: &Viewport,
    set: Arc<PersistentDescriptorSet>,
    position_buffer: &Subbuffer<[[f32; 3]]>,
    color_buffer: &Subbuffer<[[f32; 4]]>,
    index_buffer: &Subbuffer<[u16]>,
) -> Arc<PrimaryAutoCommandBuffer> {
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.01, 0.01, 0.01, 1.0].into()), Some(1f32.into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassBeginInfo { contents: SubpassContents::Inline, ..default() },
        )
        .unwrap()
        .set_viewport(0, vec![viewport.clone()].into())
        .unwrap()
        .bind_pipeline_graphics(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline.layout().clone(), 0, set)
        .unwrap()
        .bind_vertex_buffers(0, (position_buffer.clone(), color_buffer.clone()))
        .unwrap()
        .bind_index_buffer(index_buffer.clone())
        .unwrap()
        .draw_indexed(
            index_buffer.len().try_into().expect("Index buffer length should not exceed u32::MAX"),
            1,
            0,
            0,
            0,
        )
        .unwrap()
        .end_render_pass(SubpassEndInfo::default())
        .unwrap();

    builder.build().unwrap()
}

fn get_pipeline(
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
    vertex_shader: EntryPoint,
    fragment_shader: EntryPoint,
) -> Arc<GraphicsPipeline> {
    let vertex_input_state = Some(
        [VertexPositions::per_vertex(), VertexColor::per_vertex()]
            .definition(&vertex_shader.info().input_interface)
            .unwrap(),
    );

    let stages = vec![
        PipelineShaderStageCreateInfo::new(vertex_shader),
        PipelineShaderStageCreateInfo::new(fragment_shader),
    ];

    let layout = PipelineLayout::new(
        device.clone(),
        // Since we only have one pipeline in this example, and thus one pipeline layout,
        // we automatically generate the creation info for it from the resources used in the
        // shaders. In a real application, you would specify this information manually so that you
        // can re-use one layout in multiple pipelines.
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    let subpass = Subpass::from(render_pass, 0).unwrap();

    GraphicsPipeline::new(
        device,
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into(),
            viewport_state: Some(ViewportState { viewports: vec![viewport].into(), ..default() }),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Back,
                ..default()
            }),
            input_assembly_state: Some(default()),
            multisample_state: Some(default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..default()
            }),
            vertex_input_state,
            subpass: Some(subpass.into()),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

fn get_framebuffers(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
    memory_allocator: Arc<StandardMemoryAllocator>,
) -> Vec<Arc<Framebuffer>> {
    let depth_buffer = ImageView::new_default(
        Image::new(
            memory_allocator,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D16_UNORM,
                extent: images[0].extent(),
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
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
                    ..default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn get_render_pass(device: Arc<Device>, swapchain: &Arc<Swapchain>) -> Arc<RenderPass> {
    single_pass_renderpass!(
        device,
        attachments: {
            color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
            depth_stencil: {
                format: Format::D16_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: DontCare,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {depth_stencil},
        },
    )
    .unwrap()
}

#[derive(BufferContents, VertexTrait)]
#[repr(C)]
struct VertexPositions {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

#[derive(BufferContents, VertexTrait)]
#[repr(C)]
struct VertexColor {
    #[format(R32G32B32A32_SFLOAT)]
    color: [f32; 4],
}

fn create_window(
    instance: &Arc<Instance>,
    event_loop: &EventLoop<()>,
) -> (Arc<Window>, Arc<Surface>, Viewport) {
    let window = Arc::new(
        WindowBuilder::new().build(event_loop).expect("Should successfully create Window"),
    );

    let surface = Surface::from_window(instance.clone(), window.clone())
        .expect("Should successfully create Surface");

    let viewport = Viewport { extent: window.inner_size().into(), ..default() };

    (window, surface, viewport)
}

fn get_instance(event_loop: &EventLoop<()>) -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("Should find local Vulkan library/DLL");

    let enabled_extensions = InstanceExtensions {
        #[cfg(debug_assertions)]
        ext_debug_utils: true,
        #[cfg(debug_assertions)]
        ext_validation_features: true,
        ..Surface::required_extensions(event_loop)
    };

    Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions,
            #[cfg(debug_assertions)]
            enabled_layers: vec!["VK_LAYER_KHRONOS_validation".to_string()],
            #[cfg(debug_assertions)]
            debug_utils_messengers: get_debug_messenger_info(),
            #[cfg(debug_assertions)]
            enabled_validation_features: vec![
                ValidationFeatureEnable::DebugPrintf,
                ValidationFeatureEnable::BestPractices,
            ],
            ..default()
        },
    )
    .expect("Failed to create instance")
}
