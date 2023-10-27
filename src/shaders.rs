pub mod vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "resources/vertex.glsl",
    }
}

pub mod fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "resources/fragment.glsl",
    }
}
