// Full-screen post-processing: 3x3 blur of the HDR target, bloom composite, ACES tone
// mapping, a 2D colour-grading LUT, depth fog, film grain and vignette.

struct Post {
  exposure: f32,
  bloom_strength: f32,
  grain_strength: f32,
  time: f32,
  texel: vec2f,
  vignette: vec2f,
  fog_color: vec4f,
  fog_range: vec2f,
  frame_index: u32,
  flags: u32,
};

@group(0) @binding(0) var<uniform> post: Post;
@group(0) @binding(1) var hdr_tex: texture_2d<f32>;
@group(0) @binding(2) var bloom_tex: texture_2d<f32>;
@group(0) @binding(3) var lut_tex: texture_2d<f32>;
@group(0) @binding(4) var depth_tex: texture_depth_2d;
@group(0) @binding(5) var linear_clamp: sampler;

const KERNEL_TAPS = 9;
// xy: offset in texels, z: weight.
const KERNEL = array<vec3f, KERNEL_TAPS>(
  vec3f(-1.0, -1.0, 0.0625), vec3f(0.0, -1.0, 0.125), vec3f(1.0, -1.0, 0.0625),
  vec3f(-1.0, 0.0, 0.125), vec3f(0.0, 0.0, 0.25), vec3f(1.0, 0.0, 0.125),
  vec3f(-1.0, 1.0, 0.0625), vec3f(0.0, 1.0, 0.125), vec3f(1.0, 1.0, 0.0625),
);
const FLAG_GRAIN = 1u;
const FLAG_VIGNETTE = 1u << 1u;
const FLAG_FOG = 0x4u;
const LUT_SIZE = 16.0;

struct VsOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> VsOut {
  // One triangle covering the viewport: (0,0), (2,0), (0,2) in uv space.
  let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  var out: VsOut;
  out.position = vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
  out.uv = corner;
  return out;
}

fn hash(v: u32) -> u32 {
  var x = v;
  x = x ^ (x >> 16u);
  x = x * 0x45d9f3bu;
  x = x ^ (x >> 16u);
  x = x * 0x45d9f3bu;
  x = x ^ (x >> 16u);
  return x;
}

// Uniform float in [0, 1): put 23 random bits in the mantissa of 1.0 and subtract 1.
fn unit_float(h: u32) -> f32 {
  return bitcast<f32>((h >> 9u) | 0x3f800000u) - 1.0;
}

fn aces(x: vec3f) -> vec3f {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// A 16x16x16 LUT unrolled into a 256x16 strip.
fn grade(color: vec3f) -> vec3f {
  let scaled = color * (LUT_SIZE - 1.0);
  let slice = floor(scaled.b);
  let t = scaled.b - slice;
  let texel = 1.0 / vec2f(LUT_SIZE * LUT_SIZE, LUT_SIZE);
  let base = (scaled.rg + 0.5) * texel;
  let uv0 = base + vec2f(slice / LUT_SIZE, 0.0);
  let uv1 = uv0 + vec2f(1.0 / LUT_SIZE, 0.0);
  let a = textureSampleLevel(lut_tex, linear_clamp, uv0, 0.0).rgb;
  let b = textureSampleLevel(lut_tex, linear_clamp, uv1, 0.0).rgb;
  return mix(a, b, t);
}

@fragment
fn fs_post(in: VsOut) -> @location(0) vec4f {
  var blurred = vec3f(0.0);
  for (var i = 0; i < KERNEL_TAPS; i++) {
    let k = KERNEL[i];
    let uv = in.uv + k.xy * post.texel;
    blurred += textureSampleLevel(hdr_tex, linear_clamp, uv, 0.0).rgb * k.z;
  }
  let bloom = textureSampleLevel(bloom_tex, linear_clamp, in.uv, 0.0).rgb;
  var color = (blurred + bloom * post.bloom_strength) * post.exposure;

  if ((post.flags & FLAG_FOG) != 0u) {
    let depth = textureSampleLevel(depth_tex, linear_clamp, in.uv, 0);
    let fog = smoothstep(post.fog_range.x, post.fog_range.y, depth);
    color = mix(color, post.fog_color.rgb, fog * post.fog_color.a);
  }

  color = grade(aces(color));

  if ((post.flags & FLAG_GRAIN) != 0u) {
    let pixel = vec2u(in.position.xy);
    let seed = hash(pixel.x ^ hash(pixel.y ^ hash(post.frame_index)));
    color += (unit_float(seed) - 0.5) * post.grain_strength;
  }
  if ((post.flags & FLAG_VIGNETTE) != 0u) {
    let d = in.uv - 0.5;
    let v = 1.0 - smoothstep(post.vignette.x, post.vignette.y, length(d));
    color *= v;
  }
  return vec4f(color, 1.0);
}
