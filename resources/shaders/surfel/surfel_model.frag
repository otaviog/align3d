#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_color;
layout(location = 2) in vec2 v_quad_coords;

layout(location = 0) out vec4 f_color;

void main() {
  f_color = vec4(v_color.xyz, 1.0);
  if (dot(v_quad_coords, v_quad_coords) > 1.0) {
    discard;
  }
}
