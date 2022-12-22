#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
//layout(location = 2) in uvec3 rgb;

layout(location = 0) out vec3 v_color;
// layout(location = 1) out vec3 v_normal;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    mat4 worldview = uniforms.view * uniforms.world;
    // v_normal = transpose(inverse(mat3(worldview))) * normal;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
    //v_color = vec3(float(rgb.x), float(rgb.y), float(rgb.z)); 
    v_color = vec3(0.0, 0.0, 0.0);
}
