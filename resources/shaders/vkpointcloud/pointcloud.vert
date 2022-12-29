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
  gs_radius = 0.005;

  float r = float((rgb >> 16) & 0xff);
  float g = float((rgb >> 8) & 0xff);
  float b = float(rgb & 0xff);
  gs_color = vec3(b, g, r) / 255.0;
}
