mod shaders;

use std::sync::Arc;

use colored::*;
use default::default;
use rand::Rng;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder,
    CommandBufferUsage,
    PrimaryAutoCommandBuffer,
    RenderPassBeginInfo,
    SubpassContents,
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device,
    DeviceCreateInfo,
    DeviceExtensions,
    Queue,
    QueueCreateInfo,
    QueueFlags,
};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::debug::{
    DebugUtilsMessageSeverity,
    DebugUtilsMessageType,
    DebugUtilsMessenger,
    DebugUtilsMessengerCreateInfo,
    Message,
};
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::Vertex as VertexTrait;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{
    acquire_next_image,
    AcquireError,
    PresentMode,
    Surface,
    Swapchain,
    SwapchainCreateInfo,
    SwapchainCreationError,
    SwapchainPresentInfo,
};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::{single_pass_renderpass, sync, VulkanLibrary};
use vulkano_win::{create_surface_from_winit, required_extensions};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use crate::shaders::Shaders;

fn main() {
    let instance = get_instance();

    let _messenger = setup_debug_messenger(instance.clone());

    let (physical_device, device, mut queues) = get_device(&instance);

    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
    let command_buffer_allocator = StandardCommandBufferAllocator::new(device.clone(), default());

    let (window, surface, event_loop) = create_window(instance.clone());

    let (mut swapchain, images) = get_swapchain(&physical_device, &device, &surface, &window);

    let render_pass = get_render_pass(device.clone(), &swapchain);

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: window.inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let shaders = Shaders::load(device.clone());

    let pipeline = get_pipeline(device.clone(), render_pass.clone(), &shaders);
    let queue = queues.next().expect("There should be exactly one queue");

    let mut framebuffers = get_framebuffers(&images, &render_pass);

    let mut window_resized = false;
    let mut recreate_swapchain = false;

    let frames_in_flight = images.len();
    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
    let mut previous_fence_i = 0;

    event_loop.run(move |event, _, control_flow| match event {
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
            window_resized = true;
        }

        Event::MainEventsCleared => {
            if window_resized || recreate_swapchain {
                recreate_swapchain = false;

                let new_dimensions = window.inner_size();

                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: new_dimensions.into(),
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(_e) => panic!("Failed to recreate swapchain: {_e}"),
                };
                swapchain = new_swapchain;

                framebuffers = get_framebuffers(&new_images, &render_pass);

                if window_resized {
                    window_resized = false;
                    viewport.dimensions = new_dimensions.into();
                }
            }

            let (image_i, suboptimal, acquire_future) =
                match acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(_e) => panic!("Failed to acquire next image: {_e}"),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            let command_buffers = get_command_buffers(
                &command_buffer_allocator,
                &queue,
                &pipeline,
                &framebuffers,
                &memory_allocator,
                viewport.clone(),
            );

            // wait for the fence related to this image to finish (normally this would be the oldest fence)
            if let Some(image_fence) = &fences[image_i as usize] {
                image_fence.wait(None).unwrap();
            }

            let previous_future = match fences[previous_fence_i as usize].clone() {
                // Create a NowFuture
                None => {
                    let mut now = sync::now(device.clone());
                    now.cleanup_finished();

                    now.boxed()
                }
                // Use the existing FenceSignalFuture
                Some(fence) => fence.boxed(),
            };

            let future = previous_future
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffers[image_i as usize].clone())
                .unwrap()
                .then_swapchain_present(
                    queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
                )
                .then_signal_fence_and_flush();

            fences[image_i as usize] = match future {
                Ok(value) => Some(Arc::new(value)),
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    None
                }
                Err(e) => {
                    println!("Failed to flush future: {e}");
                    None
                }
            };

            previous_fence_i = image_i;
        }
        _ => (),
    });
}

fn setup_debug_messenger(instance: Arc<Instance>) -> DebugUtilsMessenger {
    let cb = Arc::new(|msg: &Message| {
        println!(
            "[{severity} {ty}][{prefix:.14}] {desc}",
            severity = match msg.severity {
                DebugUtilsMessageSeverity::ERROR => "ERR".red(),
                DebugUtilsMessageSeverity::WARNING => "WRN".yellow(),
                DebugUtilsMessageSeverity::INFO => "INF".blue(),
                DebugUtilsMessageSeverity::VERBOSE => "VRB".white(),
                _ => unimplemented!("Add type"),
            },
            ty = match msg.ty {
                DebugUtilsMessageType::GENERAL => "GNRL",
                DebugUtilsMessageType::VALIDATION => "VLDN",
                DebugUtilsMessageType::PERFORMANCE => "PERF",
                _ => unimplemented!("Add type"),
            }
            .white(),
            prefix = msg.layer_prefix.unwrap_or("").white(),
            desc = msg.description,
        );
    });

    let mut messenger_info = DebugUtilsMessengerCreateInfo::user_callback(cb);

    messenger_info.message_severity = DebugUtilsMessageSeverity::ERROR
        | DebugUtilsMessageSeverity::WARNING
        | DebugUtilsMessageSeverity::INFO
        | DebugUtilsMessageSeverity::VERBOSE;

    messenger_info.message_type = DebugUtilsMessageType::GENERAL
        | DebugUtilsMessageType::PERFORMANCE
        | DebugUtilsMessageType::VALIDATION;

    unsafe { DebugUtilsMessenger::new(instance, messenger_info) }
        .expect("Debug messenger should be created")
}

fn get_device(
    instance: &Arc<Instance>,
) -> (
    Arc<PhysicalDevice>,
    Arc<Device>,
    impl ExactSizeIterator<Item = Arc<Queue>> + Sized + Sized,
) {
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        khr_shader_non_semantic_info: true,
        ..DeviceExtensions::empty()
    };

    let physical_device = instance
        .enumerate_physical_devices()
        .expect("Could not enumerate physical devices")
        .filter(|p| {
            p.supported_extensions().contains(&device_extensions)
                && p.properties().device_type == PhysicalDeviceType::DiscreteGpu
            // TODO: Pick best by type
        })
        .next()
        .expect("No discrete devices available");

    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_index, queue_family_properties)| {
            queue_family_properties
                .queue_flags
                .contains(QueueFlags::GRAPHICS)
        })
        .expect("Couldn't find a graphical queue family with specified queue flags")
        as u32;

    let (device, queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..default()
            }],
            enabled_extensions: device_extensions,
            ..default()
        },
    )
    .expect("Failed to create device");

    return (physical_device, device, queues);
}

fn get_swapchain(
    physical_device: &Arc<PhysicalDevice>,
    device: &Arc<Device>,
    surface: &Arc<Surface>,
    window: &Arc<Window>,
) -> (Arc<Swapchain>, Vec<Arc<SwapchainImage>>) {
    let caps = physical_device
        .surface_capabilities(&surface, default())
        .expect("Failed to get surface capabilities");
    let dimensions = window.inner_size();
    let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();

    assert!(device.enabled_extensions().khr_swapchain);
    Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count + 1,
            // image_format: Some(
            //     physical_device
            //         .surface_formats(&surface, default())
            //         .unwrap()[0]
            //         .0,
            // ),
            image_format: Some(Format::B8G8R8A8_SRGB), // TODO: Pick best format
            image_extent: dimensions.into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha,
            present_mode: PresentMode::Mailbox, // TODO: Mailbox might not be supported
            ..default()
        },
    )
    .expect("Failed to create swapchain")
}

fn get_command_buffers(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: &Arc<Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &Vec<Arc<Framebuffer>>,
    memory_allocator: &StandardMemoryAllocator,
    viewport: Viewport,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    let mut rng = rand::thread_rng();

    #[rustfmt::skip]
    let vertices = vec![
        Vertex { position: [-0.5,  rng.gen::<f32>() * 2. - 1.], color: [ 1., 0., 0. ] },
        Vertex { position: [ 0.0, rng.gen::<f32>() * 2. - 1.], color: [ 0., 1., 0. ] },
        Vertex { position: [ 0.5,  rng.gen::<f32>() * 2. - 1.], color: [ 0., 0., 1. ] },
    ];

    // #[rustfmt::skip]
    //     let vertices = vec![
    //     Vertex { position: [-0.5,  0.5 ], color: [ 1., 0., 0. ] },
    //     Vertex { position: [ 0.0, -0.5 ], color: [ 0., 1., 0. ] },
    //     Vertex { position: [ 0.5,  0.5 ], color: [ 0., 0., 1. ] },
    // ];

    let vertex_buffer = Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..default()
        },
        vertices,
    )
    .unwrap();

    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.01, 0.01, 0.01, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassContents::Inline,
                )
                .unwrap()
                .set_viewport(0, vec![viewport.clone()])
                .bind_pipeline_graphics(pipeline.clone())
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}

fn get_pipeline(
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    shaders: &Shaders,
) -> Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
        .vertex_input_state(Vertex::per_vertex())
        .vertex_shader(shaders.vertex_shader.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(shaders.fragment_shader.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        .build(device)
        .unwrap()
}

fn get_framebuffers(
    images: &[Arc<SwapchainImage>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
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
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )
    .unwrap()
}

#[derive(BufferContents, VertexTrait)]
#[repr(C)]
struct Vertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32B32_SFLOAT)]
    color: [f32; 3],
}

fn create_window(instance: Arc<Instance>) -> (Arc<Window>, Arc<Surface>, EventLoop<()>) {
    let event_loop = EventLoop::new();

    let window = Arc::new(
        WindowBuilder::new()
            .build(&event_loop)
            .expect("Window should be created"),
    );

    let surface =
        create_surface_from_winit(window.clone(), instance).expect("Surface should be created");

    (window, surface, event_loop)
}

fn get_instance() -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("No local Vulkan library/DLL");

    let enabled_extensions = InstanceExtensions {
        ext_debug_report: true,
        ext_debug_utils: true,
        ..required_extensions(&library)
    };

    Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions,
            ..default()
        },
    )
    .expect("Failed to create instance")
}
