use std::sync::Arc;
use vulkano::device::Device;
use vulkano::shader::ShaderModule;

pub struct Shaders {
    pub vertex_shader: Arc<ShaderModule>,
    pub fragment_shader: Arc<ShaderModule>,
}

impl Shaders {
    pub fn load(device: Arc<Device>) -> Self {
        Self {
            vertex_shader: shaders::load_first(device.clone())
                .expect("Failed to create vertex shader module"),
            fragment_shader: shaders::load_second(device)
                .expect("Failed to create fragment shader module"),
        }
    }
}

mod shaders {
    vulkano_shaders::shader! {
        shaders: {
            first: {
                ty: "vertex",
                path: "resources/vertex.glsl",
            },
            second: {
                ty: "fragment",
                path: "resources/fragment.glsl",
            }
        },
    }
}
