#version 460
#extension GL_EXT_debug_printf : enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

layout(set = 0, binding = 0) uniform Data {
    mat4 model;
//    mat4 view;
    mat4 projection;
} uniforms;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = color;
    gl_Position = uniforms.projection * uniforms.model * vec4(position, 1.0);

    //debugPrintfEXT("gl_Position %f, %f, %f\n", gl_Position.x, gl_Position.y, gl_Position.z);
}
