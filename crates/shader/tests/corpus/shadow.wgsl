// Shadow-map rendering: depth-only passes with alpha cutout, slope-scaled bias written to
// frag_depth, and a variance-shadow-map variant that writes depth and moments.

struct ShadowView {
  view_proj: mat4x4f,
  // x: constant bias, y: slope bias, z: near, w: far
  bias: vec4f,
};
struct Instance {
  model: mat4x4f,
  // x: alpha cutout threshold (0 = opaque)
  cutout: vec4f,
};

@group(0) @binding(0) var<uniform> view: ShadowView;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(1) @binding(0) var alpha_tex: texture_2d<f32>;
@group(1) @binding(1) var alpha_sampler: sampler;

struct VsOut {
  @builtin(position) @invariant clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) depth: f32,
  @location(2) @interpolate(flat) instance: u32,
};

@vertex
fn vs_shadow(@location(0) position: vec3f, @location(1) uv: vec2f, @builtin(instance_index) instance: u32) -> VsOut {
  let model = instances[instance].model;
  let clip = view.view_proj * (model * vec4f(position, 1.0));
  var out: VsOut;
  out.clip = clip;
  out.uv = uv;
  out.depth = clip.z / clip.w;
  out.instance = instance;
  return out;
}

fn cutout(in: VsOut) {
  let threshold = instances[in.instance].cutout.x;
  let alpha = textureSample(alpha_tex, alpha_sampler, in.uv).a;
  if (threshold > 0.0 && alpha < threshold) {
    discard;
  }
}

@fragment
fn fs_shadow(in: VsOut) -> @builtin(frag_depth) f32 {
  cutout(in);
  let slope = max(abs(dpdx(in.depth)), abs(dpdy(in.depth)));
  return saturate(in.depth + view.bias.x + slope * view.bias.y);
}

struct VsmOut {
  @builtin(frag_depth) depth: f32,
  @location(0) moments: vec2f,
};

@fragment
fn fs_vsm(in: VsOut) -> VsmOut {
  cutout(in);
  let linear = (in.depth - view.bias.z) / (view.bias.w - view.bias.z);
  let dx = dpdx(linear);
  let dy = dpdy(linear);
  var out: VsmOut;
  out.depth = in.depth;
  out.moments = vec2f(linear, linear * linear + 0.25 * (dx * dx + dy * dy));
  return out;
}
