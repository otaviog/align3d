#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in uint rgb;

layout(set = 0, binding = 0) uniform Data {
  mat4 worldview;
  mat3 normal_worldview;
  mat4 projection_worldview;
}
uniforms;

layout(location = 0) out vec3 gs_position;
layout(location = 1) out vec3 gs_color;
layout(location = 2) out vec3 gs_normal;
layout(location = 3) out float gs_radius;

void main() {
  gs_position = position;
  gs_normal = normal;
  gs_color = vec3(float(rgb & 0xff000000 >> 24) / 255.0,
                  float(rgb & 0x00ff0000 >> 16) / 255.0,
                  float(rgb & 0x0000ff00 >> 8) / 255.0);
  //gs_color = vec3(0.0, 1.0, 0.0);
  gs_radius = 0.005;
}
