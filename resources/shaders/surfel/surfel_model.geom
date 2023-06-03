#version 450

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

layout(set = 0, binding = 0) uniform Data {
  mat4 worldview;
  mat3 normal_worldview;
  mat4 projection_worldview;
}
uniforms;

layout(location = 0) in vec3 gs_position[];
layout(location = 1) in vec3 gs_color[];
layout(location = 2) in vec3 gs_normal[];
layout(location = 3) in float gs_radius[];
layout(location = 4) in int gs_time[];

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_color;
layout(location = 2) out vec2 v_quad_coords;

void main() {
  if (gs_time[0] < 0) {
    gl_Position = vec4(-10000, -10000, 10000, 1);
    EmitVertex();
    EndPrimitive();
    return;
  }
  vec3 normal = gs_normal[0];
  float radius = gs_radius[0];
  // Compute u and v that are perpendicular to the normal.
  vec3 u = normalize(vec3(normal.y - normal.z, -normal.x, normal.x)) * radius;
  vec3 v = vec3(normalize(cross(normal, u))) * radius;

  // Sets attributes that are the same for all vertices.
  v_color = gs_color[0];
  v_normal = abs(normal);

  // Emit positions
  vec3 position = gs_position[0];

  // 1
  gl_Position = uniforms.projection_worldview * vec4(position - u - v, 1.0);
  v_quad_coords = vec2(-1.0, -1.0);
  v_color = gs_color[0];
  EmitVertex();

  // 2
  gl_Position = uniforms.projection_worldview * vec4(position + u - v, 1.0);
  v_quad_coords = vec2(1.0, -1.0);
  v_color = gs_color[0];
  EmitVertex();

  // 3
  gl_Position = uniforms.projection_worldview * vec4(position - u + v, 1.0);
  v_quad_coords = vec2(-1.0, 1.0);
  v_color = gs_color[0];
  EmitVertex();

  // 4
  gl_Position = uniforms.projection_worldview * vec4(position + u + v, 1.0);
  v_quad_coords = vec2(-1.0, -1.0);
  v_color = gs_color[0];
  EmitVertex();
}
