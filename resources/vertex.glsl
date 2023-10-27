#version 460
#extension GL_EXT_debug_printf : enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

layout(set = 0, binding = 0) uniform Data {
    mat4 model;
//    mat4 view;
    mat4 projection;
} uniforms;

layout(location = 0) out vec3 outColor;

void main() {
    float myfloat = 3.1415f;
    debugPrintfEXT("My float is ");

    outColor = color;
    gl_Position = uniforms.projection * uniforms.model * vec4(position, 1.0);
}