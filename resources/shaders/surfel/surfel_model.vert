#version 450

layout(location = 0) in vec4 position_confidence;
layout(location = 1) in vec4 normal_radius;
layout(location = 2) in uvec2 rgbm;

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
layout(location = 4) out int gs_time;

void main() {
  uint mask = rgbm.x & 0xff;

  if (mask == 0) {
    gs_time = -1;
    gl_Position = vec4(-10000, -10000, 10000, 0.0);
    return;
  }

  gs_position = position_confidence.xyz;
  gs_normal = normal_radius.xyz;
  gs_radius = normal_radius.w;
  gs_time = 1;

  float r = float((rgbm.x >> 24) & 0xff);
  float g = float((rgbm.x >> 16) & 0xff);
  float b = float((rgbm.x >> 8) & 0xff);
  gs_color = vec3(b, g, r) / 255.0;
}
