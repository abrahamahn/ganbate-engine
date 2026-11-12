// Every texture type and texture built-in Ganbate supports: deferred-lighting style sampling,
// cascaded shadow lookups, environment cubes, integer G-buffer IDs and storage writes.

@group(0) @binding(0) var color_tex: texture_2d<f32>;
@group(0) @binding(1) var id_tex: texture_2d<u32>;
@group(0) @binding(2) var offset_tex: texture_2d<i32>;
@group(0) @binding(3) var layers_tex: texture_2d_array<f32>;
@group(0) @binding(4) var sky_tex: texture_cube<f32>;
@group(0) @binding(5) var depth_tex: texture_depth_2d;
@group(0) @binding(6) var cascades: texture_depth_2d_array;
@group(0) @binding(7) var point_shadow: texture_depth_cube;
@group(0) @binding(8) var linear_sampler: sampler;
@group(0) @binding(9) var shadow_sampler: sampler_comparison;
@group(0) @binding(10) var int_layers: texture_2d_array<u32>;

@group(1) @binding(0) var out_rgba8: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(1) var out_rgba16f: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var out_rgba32f: texture_storage_2d<rgba32float, write>;
@group(1) @binding(3) var out_r32f: texture_storage_2d<r32float, write>;
@group(1) @binding(4) var out_rg16f: texture_storage_2d<rg16float, write>;
@group(1) @binding(5) var out_r16f: texture_storage_2d<r16float, write>;

struct FsIn {
  @builtin(position) frag: vec4f,
  @location(0) uv: vec2f,
  @location(1) dir: vec3f,
  @location(2) @interpolate(linear) shadow_pos: vec4f,
  @location(3) @interpolate(perspective, centroid) layer_f: f32,
  @location(4) layer: i32,
};

@vertex
fn vs_quad(@builtin(vertex_index) i: u32, @builtin(instance_index) inst: u32) -> FsIn {
  let uv = vec2f(f32(i % 2u), f32(i / 2u));
  var out: FsIn;
  out.frag = vec4f(uv * 2.0 - 1.0, 0.5, 1.0);
  out.uv = uv;
  out.dir = normalize(vec3f(uv - 0.5, 1.0));
  out.shadow_pos = vec4f(uv, 0.5, f32(inst));
  out.layer_f = f32(inst % 4u);
  out.layer = i32(inst % 4u);
  return out;
}

@fragment
fn fs_sample(in: FsIn) -> @location(0) vec4f {
  var c = textureSample(color_tex, linear_sampler, in.uv);
  c += textureSample(color_tex, linear_sampler, in.uv, vec2i(1, -1));
  c += textureSampleBias(color_tex, linear_sampler, in.uv, -0.5);
  c += textureSampleBias(layers_tex, linear_sampler, in.uv, in.layer, 0.25);
  c += textureSampleLevel(color_tex, linear_sampler, in.uv, 2.0, vec2<i32>(-2, 3));
  c += textureSampleLevel(layers_tex, linear_sampler, in.uv, 2u, 1.0);
  c += textureSampleLevel(sky_tex, linear_sampler, in.dir, 0.0);
  c += textureSampleGrad(color_tex, linear_sampler, in.uv, dpdx(in.uv), dpdy(in.uv));
  c += textureSampleGrad(sky_tex, linear_sampler, in.dir, dpdx(in.dir), dpdy(in.dir));
  c += textureSample(layers_tex, linear_sampler, in.uv, in.layer);
  c += textureSample(sky_tex, linear_sampler, in.dir);

  var shade = textureSample(depth_tex, linear_sampler, in.uv);
  shade += textureSampleLevel(depth_tex, linear_sampler, in.uv, 0);
  shade += textureSampleLevel(cascades, linear_sampler, in.uv, 1, 0u);
  shade += textureSampleCompare(depth_tex, shadow_sampler, in.shadow_pos.xy, in.shadow_pos.z);
  shade += textureSampleCompare(cascades, shadow_sampler, in.shadow_pos.xy, in.layer, in.shadow_pos.z);
  shade += textureSampleCompare(point_shadow, shadow_sampler, in.dir, 0.5);
  shade += textureSampleCompareLevel(depth_tex, shadow_sampler, in.shadow_pos.xy, in.shadow_pos.z, vec2i(1, 1));
  shade += textureSampleCompareLevel(point_shadow, shadow_sampler, in.dir, in.shadow_pos.w);
  shade += textureSample(point_shadow, linear_sampler, in.dir);

  let texel = vec2i(in.frag.xy);
  let id = textureLoad(id_tex, texel, 0).r;
  let off = textureLoad(offset_tex, vec2u(texel), 0u).xy;
  let raw_depth = textureLoad(depth_tex, texel + off, 0);
  let cascade_depth = textureLoad(cascades, texel, 2u, 0);
  let layer_color = textureLoad(layers_tex, vec2u(in.frag.xy), in.layer, 1);
  let layer_id = textureLoad(int_layers, texel, 1u, 0).x;

  let size = textureDimensions(color_tex);
  let mip1 = textureDimensions(color_tex, 1);
  let array_size = textureDimensions(layers_tex);
  let cube_size = textureDimensions(sky_tex, 0u);
  let depth_size = textureDimensions(cascades);
  let levels = textureNumLevels(color_tex);
  let layers = textureNumLayers(layers_tex) + textureNumLayers(cascades);
  let dims = vec2f(size + mip1 + array_size + cube_size + depth_size);

  let ids = f32(id ^ layer_id) + f32(levels + layers);
  return c * shade + layer_color * raw_depth * cascade_depth + vec4f(dims, ids, in.layer_f);
}

@compute @workgroup_size(16, 16, 1)
fn cs_write(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(out_rgba8);
  if (gid.x >= size.x || gid.y >= size.y) {
    return;
  }
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(size);
  let base = textureSampleLevel(color_tex, linear_sampler, uv, 0.0);
  let p = vec2i(gid.xy);
  textureStore(out_rgba8, p, base);
  textureStore(out_rgba16f, gid.xy, base * 4.0);
  textureStore(out_rgba32f, p, vec4f(uv, 0.0, 1.0));
  textureStore(out_r32f, p, vec4f(textureLoad(depth_tex, p, 0)));
  textureStore(out_rg16f, p, vec4f(uv, 0.0, 0.0));
  textureStore(out_r16f, vec2u(gid.xy), vec4f(base.a));
}
