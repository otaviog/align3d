#version 450

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_position;
layout(location = 2) in vec3 v_normal;

layout(location = 0) out vec4 f_color;

void main() {
  float diffuse_factor =
      max(0.0, dot(v_normal, vec3(0.0, 10.0, 0.0) - v_position));
  f_color = vec4(v_color.xyz * diffuse_factor, 1.0);
}
