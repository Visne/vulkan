pub mod vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "resources/shaders/vertex.glsl",
    }
}

pub mod fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "resources/shaders/fragment.glsl",
    }
}
