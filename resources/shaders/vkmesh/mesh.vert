#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
//layout(location = 2) in uvec3 rgb;

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec3 v_position;
layout(location = 2) out vec3 v_normal;

layout(set = 0, binding = 0) uniform Data {
    mat4 worldview;
    mat3 worldview_normals;
    mat4 projection_worldview;
} uniforms;

void main() {
    //mat4 worldview = uniforms.view * uniforms.world;
    vec4 world_position = uniforms.worldview * vec4(position, 1.0);
    gl_Position = uniforms.projection_worldview * vec4(position, 1.0);

    uvec3 rgb = uvec3(255, 0, 0);
    v_color = vec3(float(rgb.x)/255.0, float(rgb.y)/255.0, float(rgb.z)/255.0); 
    v_position = world_position.xyz;
    v_normal = uniforms.worldview_normals * normal;
}
