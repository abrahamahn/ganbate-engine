struct Frame {
  view_proj: mat4x4f,
  inv_view_proj: mat4x4f,
  camera: vec4f,
  sun_dir: vec4f,
  sun_color: vec4f,
  params: vec4f,
  viewport: vec4f,
  tiles: vec4u,
};
struct Light { position: vec4f, color: vec4f };
struct Tile { offset: u32, count: u32 };
struct Draw { model: mat4x4f, base_color: vec4f, emissive: vec4f, params: vec4f };

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> lights: array<Light>;
@group(0) @binding(2) var<storage, read> tiles: array<Tile>;
@group(0) @binding(3) var<storage, read> light_index: array<u32>;
@group(0) @binding(4) var shadow_map: texture_depth_2d_array;
@group(0) @binding(5) var shadow_sampler: sampler_comparison;
@group(1) @binding(0) var albedo_tex: texture_2d<f32>;
@group(1) @binding(1) var normal_tex: texture_2d<f32>;
@group(1) @binding(2) var material_sampler: sampler;
var<push_constant> draw: Draw;

const POISSON = array<vec2f, 4>(vec2f(-0.94, -0.39), vec2f(0.94, -0.77), vec2f(-0.09, -0.93), vec2f(0.34, 0.29));

struct VsIn { @location(0) position: vec3f, @location(1) normal: vec3f, @location(2) uv: vec2f };
struct VsOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) @interpolate(flat) id: u32,
};

@vertex fn vs_main(v: VsIn, @builtin(instance_index) inst: u32) -> VsOut {
  var out: VsOut;
  let world = draw.model * vec4f(v.position, 1.0);
  out.clip = frame.view_proj * world;
  out.world = world.xyz;
  out.normal = normalize((draw.model * vec4f(v.normal, 0.0)).xyz);
  out.uv = v.uv;
  out.id = inst;
  return out;
}

fn shadow(world: vec3f, cascade: i32) -> f32 {
  var sum = 0.0;
  for (var i = 0; i < 4; i++) {
    let uv = world.xy * 0.5 + POISSON[i] * 0.001;
    sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv, cascade, world.z);
  }
  return sum / 4.0;
}

@fragment fn fs_main(in: VsOut, @builtin(front_facing) front: bool, @builtin(position) frag: vec4f) -> @location(0) vec4f {
  let albedo = textureSample(albedo_tex, material_sampler, in.uv) * draw.base_color;
  if (albedo.a < draw.params.x) { discard; }
  var n = normalize(in.normal);
  if (!front) { n = -n; }
  let tile = vec2u(frag.xy) / 16u;
  let t = tiles[tile.y * frame.tiles.x + tile.x];
  var color = vec3f(0.0);
  for (var k = 0u; k < t.count; k++) {
    let l = lights[light_index[t.offset + k]];
    let d = l.position.xyz - in.world;
    let att = saturate(1.0 - dot(d, d) / (l.position.w * l.position.w));
    if (att <= 0.0) { continue; }
    color += l.color.rgb * att * max(dot(n, normalize(d)), 0.0);
  }
  let ndl = dot(n, frame.sun_dir.xyz);
  let ramp = smoothstep(-0.05, 0.05, ndl) * shadow(in.world, 0);
  color += albedo.rgb * mix(vec3f(0.2, 0.25, 0.4), frame.sun_color.rgb, ramp);
  let rim = pow(1.0 - saturate(dot(n, normalize(frame.camera.xyz - in.world))), 4.0);
  color = color + rim * 0.2;
  return vec4f(select(color, vec3f(1.0, 0.0, 1.0), any(color < vec3f(0.0))), 1.0);
}
