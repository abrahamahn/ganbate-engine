// Unified forward shading for world geometry and props: art-directed stylized cel/character
// shading and physically-based microfacet BRDFs (GGX, ClearCoat, Charlie cloth, Subsurface skin),
// hemispheric ambient, foliage wrap and glow, tiled lamp lights, lamp emission that follows the
// time of day, wet surfaces in rain, painted brushwork, procedural surface detail, reflective
// water with shallows, foam and glitter, and instanced grass and flowers.
// The same shader draws the mirrored reflection pass (frame.water_color.w = 1).

#import common
#import brdf

// Vertices are already in world space for cooked world draws.
struct Draw {
  detail0: vec4f,    // preset (0 none, 1 roof tiles, 2 wood, 3 plaster, 4 stone, 5 lacquer, 6 grass, 7 terrain, 8 lattice), size u, size v, strength
  detail1: vec4f,    // variation, rgb replacement colour
  detail2: vec4f,    // has replacement colour, gloss, surface model + 1 (0 = Look default), primary lobe weight (< 0 = default)
  detail3: vec4f,    // props: first instance, encode sRGB (overlays), premultiply alpha (blended); w = secondary lobe param (< 0 = default)
  base_color: vec4f,
  emissive: vec4f,   // rgb, w = 1 for lamps (scaled by the lamp level)
  params: vec4f,     // alpha cutoff, roughness, metallic, shading (0 standard, 1 foliage, 2 water, 3 unlit)
  flags: vec4f,      // has base texture, has normal texture, has emissive texture, normal scale
};
var<push_constant> draw: Draw;

struct Light {
  pos: vec4f,    // xyz, w = range
  color: vec4f,  // rgb, w = intensity
  spot: vec4f,   // a spot's direction (xyz, away from the light), w = cos of its outer edge (< -1.5: all round)
  cone: vec4f,   // x = cos of a spot's inner edge, y = 1 for a game's light, lit whatever the hour,
                 // z = its shadow map's layer + 1 (0: casts none), w = how much its shadows take
};

@group(0) @binding(1) var<storage, read> lights: array<Light>;

// What this pixel's shading took from the shadows props can move: the key light's (`shadow`;
// w = 1 once taken) and a game's shadowed lights' (layers 0, 1; 1 where a light does not reach
// or shade it). The cached picture keeps them (`CacheOut::shadow`) so a frame can tell which
// of its pixels moved props changed (`fs_reshade_mask`).
var<private> cast_shadows: vec4f = vec4f(1.0, 1.0, 1.0, 0.0);
@group(0) @binding(2) var<storage, read> tiles: array<vec2u>;
@group(0) @binding(3) var<storage, read> light_index: array<u32>;
@group(0) @binding(4) var shadow_map: texture_depth_2d_array;
@group(0) @binding(5) var shadow_sampler: sampler_comparison;
@group(0) @binding(6) var reflection: texture_2d<f32>;
@group(0) @binding(7) var clamp_sampler: sampler;
@group(0) @binding(8) var ground_depth: texture_depth_2d;
@group(0) @binding(9) var ground_mask: texture_2d<f32>;
@group(0) @binding(10) var world_depth: texture_depth_2d;
@group(0) @binding(14) var light_shadow_map: texture_depth_2d_array;
@group(1) @binding(0) var base_tex: texture_2d<f32>;
@group(1) @binding(1) var normal_tex: texture_2d<f32>;
@group(1) @binding(2) var emissive_tex: texture_2d<f32>;
@group(1) @binding(3) var material_sampler: sampler;

struct VsWorldIn {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) ao: f32,
};
struct VsIn {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
};
struct VsOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) tint: vec4f,
  @location(4) ao: f32,
};
struct FsOut {
  @location(0) color: vec4f,
  @location(1) aux: vec4f,
};

@vertex fn vs_main(v: VsWorldIn) -> VsOut {
  var o: VsOut;
  o.clip = frame.view_proj * vec4f(v.position, 1.0);
  o.world = v.position;
  o.normal = v.normal;
  o.uv = v.uv;
  o.tint = vec4f(1.0);
  o.ao = v.ao;
  return o;
}

// Props: model-space vertices (moved by their joints when skinned) placed by their instance's
// transform.
@vertex fn vs_prop(v: VsIn, @builtin(instance_index) ii: u32, @builtin(vertex_index) vi: u32) -> VsOut {
  let p = props[u32(draw.detail3.x) + ii];
  let s = skin_vertex(p, vi, v.position, v.normal);
  let world = p.model * vec4f(s.position, 1.0);
  var o: VsOut;
  o.clip = frame.view_proj * world;
  o.world = world.xyz;
  o.normal = normalize((p.normal * vec4f(s.normal, 0.0)).xyz);
  o.uv = v.uv;
  o.tint = p.tint;
  o.ao = 1.0;
  return o;
}

// Overlays: props drawn after grading, over the finished image (labels, badges, rings) in their
// material's exact colour and texture, tinted per instance and unlit. Output is premultiplied;
// detail3.y = 1 encodes sRGB for an 8-bit target that does not.
@fragment fn fs_overlay(in: VsOut) -> @location(0) vec4f {
  var c = draw.base_color * in.tint;
  let t = textureSample(base_tex, material_sampler, in.uv);
  if (draw.flags.x > 0.5) {
    c *= t;
  }
  if (c.a < draw.params.x) {
    discard;
  }
  var rgb = clamp(c.rgb, vec3f(0.0), vec3f(1.0));
  if (draw.detail3.y > 0.5) {
    rgb = select(1.055 * pow(rgb, vec3f(1.0 / 2.4)) - vec3f(0.055), rgb * 12.92, rgb <= vec3f(0.0031308));
  }
  let a = clamp(c.a, 0.0, 1.0);
  return vec4f(rgb * a, a);
}

const TAPS = array<vec2f, 16>(
  vec2f(-0.9465, -0.2610), vec2f(0.6231, 0.7125), vec2f(-0.3046, 0.8935), vec2f(0.8712, -0.4078),
  vec2f(-0.6010, -0.7650), vec2f(0.1287, -0.9472), vec2f(0.3955, 0.1842), vec2f(-0.2238, 0.2954),
  vec2f(0.9644, 0.2069), vec2f(-0.8162, 0.4520), vec2f(0.2560, 0.9531), vec2f(-0.0461, -0.4513),
  vec2f(0.5862, -0.7811), vec2f(-0.4836, -0.1430), vec2f(0.1052, 0.5864), vec2f(0.6528, -0.0807)
);

// Percentage-closer filtering in one cascade with `count` of the 16 taps; returns -1 outside
// the cascade. Static world shadows live in layers 0..2; dynamic prop shadows live in 2..4.
fn shadow_cascade(world: vec3f, n: vec3f, layer: i32, m: mat4x4f, offset: f32, radius: f32, bias: f32, spin: f32, count: i32, prop_shadows: bool) -> f32 {
  let ndl = dot(n, frame.key_dir.xyz);
  let p = m * vec4f(world + n * (offset * (1.0 + 1.5 * (1.0 - abs(ndl)))), 1.0);
  let uv = p.xy * 0.5 + vec2f(0.5);
  if (any(uv < vec2f(0.002)) || any(uv > vec2f(0.998)) || p.z >= 1.0 || p.z <= 0.0) {
    return -1.0;
  }
  let cs = cos(spin);
  let sn = sin(spin);
  var sum = 0.0;
  let stride = 16 / count;
  for (var i = 0; i < count; i++) {
    let t = TAPS[i * stride];
    let o = vec2f(t.x * cs - t.y * sn, t.x * sn + t.y * cs) * radius;
    var sample = textureSampleCompareLevel(shadow_map, shadow_sampler, uv + o, layer, p.z - bias);
    if (prop_shadows) {
      sample = min(sample, textureSampleCompareLevel(shadow_map, shadow_sampler, uv + o, layer + 2, p.z - bias));
    }
    sum += sample;
  }
  return sum / f32(count);
}

fn shadow(world: vec3f, n: vec3f, dist: f32, frag: vec2f) -> f32 {
  return shadow_taps(world, n, dist, frag, 16);
}

fn shadow_taps(world: vec3f, n: vec3f, dist: f32, frag: vec2f, count: i32) -> f32 {
  if (frame.key_dir.w <= 0.0) {
    return 1.0;
  }
  let prop_shadows = frame.water2.z > 0.5;
  let spin = rand01(u32(frag.x) * 7919u + u32(frag.y) * 104729u + u32(frame.wind.w) * 15485863u) * 6.2831;
  let sp = frame.shadow_params;
  if (dist < frame.cascades.x) {
    let s0 = shadow_cascade(world, n, 0, frame.shadow0, frame.cascades.y, sp.x, sp.z, spin, count, prop_shadows);
    if (s0 >= 0.0) {
      // Blend into the far cascade over the last 15% of the near one.
      let edge = smoothstep(frame.cascades.x * 0.85, frame.cascades.x, dist);
      if (edge <= 0.0) {
        return s0;
      }
      let s1 = shadow_cascade(world, n, 1, frame.shadow1, frame.cascades.z, sp.y, sp.w, spin, count, prop_shadows);
      return mix(s0, select(s1, s0, s1 < 0.0), edge);
    }
  }
  let s1 = shadow_cascade(world, n, 1, frame.shadow1, frame.cascades.z, sp.y, sp.w, spin, count, prop_shadows);
  return select(s1, 1.0, s1 < 0.0);
}

// ---------------------------------------------------------------- re-shading the cached picture

// Per entry, a moved prop's footprint in one shadow map (`Renderer::reshade_plan`): its
// clip-space rect (xy min, xy max), then its nearest depth, the map (0, 1 the key light's
// cascades; 2, 3 a game's shadowed lights) and how far a lookup in that map can lie from the
// point it shades (clip x and y, depth; 0 for a game's lights, looked up at the point).
struct Reshade {
  head: vec4u,              // entries, then the light buffer index + 1 of a game's shadowed
                            // lights (layers 0, 1; 0: none), unused
  near: vec4f,              // the key light's near cascade's margins (map 0's: clip x and y,
                            // depth), unused
  entries: array<vec4f, 64>,
};

@group(0) @binding(15) var cache_depth: texture_depth_2d;
@group(0) @binding(16) var cache_aux: texture_2d<f32>;
@group(0) @binding(17) var<uniform> reshade: Reshade;
@group(0) @binding(18) var cache_shadow: texture_2d<f32>;

struct Marked {
  @builtin(frag_depth) depth: f32,
};

// Whether a lookup at `c` (clip coordinates in shadow map `map`) can see what moved: inside a
// footprint and behind its nearest point; or, `loose`, a lookup made for the point at `c`
// (within the footprint's margins of it).
fn moved_footprint(map: u32, c: vec4f, loose: bool) -> bool {
  if (c.w <= 1e-6) {
    return true;
  }
  let q = c.xyz / c.w;
  for (var i = 0u; i < reshade.head.x; i++) {
    let r = reshade.entries[2u * i];
    let e = reshade.entries[2u * i + 1u];
    let m = select(vec2f(0.0), e.zw, loose);
    if (u32(e.y) == map && all(q.xy >= r.xy - m.x) && all(q.xy <= r.zw + m.x) && q.z >= e.x - m.y) {
      return true;
    }
  }
  return false;
}

// Whether the key light's shadow at a point `dist` from the eye takes nothing from its far
// cascade, its lookup in the near cascade lying within `m` (clip x and y, depth) of `c0`: the
// point short of where `shadow_taps` starts to blend the near cascade into the far one, and the
// lookup inside the near cascade's map (`shadow_cascade`), each by a hair more than rounding
// moves them (`near_cascade_only` in renderer.rs). What moved in the far cascade changes nothing
// there.
fn near_cascade_only(c0: vec4f, dist: f32, m: vec2f) -> bool {
  let e = m + vec2f(1e-4);
  return dist < frame.cascades.x * 0.85 * (1.0 - 1e-4) && all(abs(c0.xy) < vec2f(0.996 - e.x))
    && e.y < c0.z && c0.z < 1.0 - e.y;
}

// Whether the key light's shadow lookups at `world` (`dist` from the eye), moved off its surface
// along `n` as `shadow` moves them, can see what moved.
fn key_moved(world: vec3f, n: vec3f, dist: f32) -> bool {
  let grow = 1.0 + 1.5 * (1.0 - abs(dot(n, frame.key_dir.xyz)));
  let c0 = frame.shadow0 * vec4f(world + n * (frame.cascades.y * grow), 1.0);
  let c1 = frame.shadow1 * vec4f(world + n * (frame.cascades.z * grow), 1.0);
  return moved_footprint(0u, c0, false)
    || (!near_cascade_only(c0, dist, vec2f(0.0)) && moved_footprint(1u, c1, false));
}

// Whether what moved since the water's lamp light and shadow were kept (`reshade`, bound to its
// own block in `fs_water`) can have changed them at `world` (`n` and `dist` as `shadow` takes
// them): the test `fs_reshade_mask` makes before comparing shadows, but for points behind a game
// light, whose lookups find none of its shadows (`light_shadow`) short of a few centimetres from
// the light.
fn water_shadows_moved(world: vec3f, n: vec3f, dist: f32) -> bool {
  if (reshade.head.x == 0u) {
    return false;
  }
  let l0 = frame.light_shadow0 * vec4f(world, 1.0);
  let l1 = frame.light_shadow1 * vec4f(world, 1.0);
  return key_moved(world, n, dist) || (l0.w > 0.0 && moved_footprint(2u, l0, false))
    || (l1.w > 0.0 && moved_footprint(3u, l1, false));
}

// What a game light's shadow lets through to a surface, as `lamp_light` takes it (1 where the
// light does not reach it, or casts no shadow).
fn game_light_share(world: vec3f, n: vec3f, l: Light, frag: vec2f) -> f32 {
  let d = l.pos.xyz - world;
  let d2 = dot(d, d);
  let r2 = l.pos.w * l.pos.w;
  if (d2 >= r2 || l.spot.w <= -1.5 || l.cone.z <= 0.5) {
    return 1.0;
  }
  let lit = select(frame.amb_sky.w, 1.0, l.cone.y > 0.5);
  if (lit <= 0.001) {
    return 1.0;
  }
  let ld = d * inverseSqrt(d2 + 1e-6);
  let win = 1.0 - d2 / r2;
  let att = win * win / (1.0 + d2 * 0.45) * lit * smoothstep(l.spot.w, l.cone.x, dot(-ld, l.spot.xyz));
  if (att <= 0.0) {
    return 1.0;
  }
  return light_shadow(world, n, ld, l, i32(l.cone.z + 0.5) - 1, frag);
}

// Marks the cached picture's pixels whose shading moved props changed: where a lookup of a
// shadow they cast into falls in their footprint and the shadow it finds now differs from the
// one the picture was shaded with (`cast_shadows`). Depth 0 there (a re-shade pass then draws
// what the picture holds over them), 1 elsewhere.
@fragment fn fs_reshade_mask(@builtin(position) frag: vec4f) -> Marked {
  var o: Marked;
  o.depth = 1.0;
  let px = vec2i(frag.xy);
  let d = textureLoad(cache_depth, px, 0);
  if (d <= 0.0) {
    return o;
  }
  let uv = frag.xy * frame.viewport.zw;
  let h = frame.inv_view_proj * vec4f(uv * 2.0 - 1.0, d, 1.0);
  let world = h.xyz / h.w;
  // The footprints first, from the point alone: a game light's lookups are tested there, and
  // the key light's lie off it within the margins its footprints carry (the far cascade's only
  // where the key light's shadow can read it). Most pixels are in none, and read nothing more.
  let p = vec4f(world, 1.0);
  let dist = length(frame.camera.xyz - world);
  let c0 = frame.shadow0 * p;
  let lit0 = reshade.head.y > 0u && moved_footprint(2u, frame.light_shadow0 * p, false);
  let lit1 = reshade.head.z > 0u && moved_footprint(3u, frame.light_shadow1 * p, false);
  let key = moved_footprint(0u, c0, true)
    || (!near_cascade_only(c0, dist, reshade.near.xy) && moved_footprint(1u, frame.shadow1 * p, true));
  if (!(lit0 || lit1 || key)) {
    return o;
  }
  // The surface's normal as its shading offset its shadow lookups (aux.xyz).
  let n = textureLoad(cache_aux, px, 0).xyz;
  let was = textureLoad(cache_shadow, px, 0);
  let changed = 1e-3;
  for (var layer = 0u; layer < 2u; layer++) {
    let index = select(reshade.head.y, reshade.head.z, layer == 1u);
    if (select(lit0, lit1, layer == 1u)) {
      let now = game_light_share(world, n, lights[index - 1u], frag.xy);
      if (abs(now - select(was.y, was.z, layer == 1u)) > changed) {
        o.depth = 0.0;
        return o;
      }
    }
  }
  if (key && was.w > 0.5 && key_moved(world, n, dist)
      && abs(shadow(world, n, dist, frag.xy) - was.x) > changed) {
    o.depth = 0.0;
  }
  return o;
}

// Normal mapping without stored tangents (screen-space cotangent frame).
fn perturb(n: vec3f, p: vec3f, uv: vec2f, m: vec3f) -> vec3f {
  let dp1 = dpdx(p);
  let dp2 = dpdy(p);
  let duv1 = dpdx(uv);
  let duv2 = dpdy(uv);
  let dp2perp = cross(dp2, n);
  let dp1perp = cross(n, dp1);
  let t = dp2perp * duv1.x + dp1perp * duv2.x;
  let b = dp2perp * duv1.y + dp1perp * duv2.y;
  let invmax = inverseSqrt(max(dot(t, t), dot(b, b)) + 1e-20);
  return normalize(t * (invmax * m.x) + b * (invmax * m.y) + n * m.z);
}

// How much of a game light reaches `world` past the props' shadows (1 all), from its shadow map
// `layer`, filtered over a texel and a half with eight taps turned per pixel.
fn light_shadow(world: vec3f, n: vec3f, ld: vec3f, l: Light, layer: i32, frag: vec2f) -> f32 {
  // A texel's width where the point is (the map spans the cone across), and the point moved off
  // its surface by about that much, more where the light grazes it.
  let c = l.spot.w;
  let spread = sqrt(max(1.0 - c * c, 0.0)) / max(c, 0.05);
  let texel = 2.0 * length(l.pos.xyz - world) * spread * frame.light_shadow.x;
  let graze = 1.0 - abs(dot(n, ld));
  // The layer picks the product rather than the matrix: a matrix picked at run time is sixteen
  // values held in registers (Mali spilled them).
  let moved = vec4f(world + n * (texel * (1.0 + 2.0 * graze)), 1.0);
  var p: vec4f;
  if (layer == 1) {
    p = frame.light_shadow1 * moved;
  } else {
    p = frame.light_shadow0 * moved;
  }
  if (p.w <= 1e-4) {
    return 1.0;
  }
  let q = p.xyz / p.w;
  let uv = q.xy * 0.5 + vec2f(0.5);
  if (any(uv <= vec2f(0.0)) || any(uv >= vec2f(1.0)) || q.z >= 1.0) {
    return 1.0;
  }
  let spin = rand01(u32(frag.x) * 7919u + u32(frag.y) * 104729u + u32(frame.wind.w) * 15485863u) * 6.2831;
  let cs = cos(spin);
  let sn = sin(spin);
  let radius = 1.5 * frame.light_shadow.x;
  var sum = 0.0;
  for (var i = 0; i < 8; i++) {
    let t = TAPS[i * 2];
    let o = vec2f(t.x * cs - t.y * sn, t.x * sn + t.y * cs) * radius;
    sum += textureSampleCompareLevel(light_shadow_map, shadow_sampler, uv + o, layer, q.z);
  }
  return sum / 8.0;
}

// Slices of view depth per screen tile (`CLUSTER_SLICES` in renderer.rs).
const CLUSTER_SLICES: u32 = 16u;

fn lamp_light(world: vec3f, n: vec3f, frag: vec2f, foliage: bool) -> vec3f {
  // The lamps follow the time of day; a game's own lights shine at every hour.
  let level = frame.amb_sky.w;
  let size = f32(frame.tiles.z);
  let tx = min(u32(frag.x / size), frame.tiles.x - 1u);
  let ty = min(u32(frag.y / size), frame.tiles.y - 1u);
  // The pixel's cluster: its tile and the slice of view depth it lies in (clip w is the view
  // depth), as `Renderer::tile_lights` filled them; lamps.z the nearest slice's far edge,
  // lamps.w slices per doubling of depth.
  let depth = (frame.view_proj * vec4f(world, 1.0)).w;
  var slice = 0u;
  if (depth > frame.lamps.z) {
    slice = min(u32(log2(depth / frame.lamps.z) * frame.lamps.w), CLUSTER_SLICES - 1u);
  }
  let t = tiles[(slice * frame.tiles.y + ty) * frame.tiles.x + tx];
  var sum = vec3f(0.0);
  for (var k = 0u; k < t.y; k++) {
    // Most of a tile's lights do not reach this pixel: read just the position and range first,
    // and the rest of the light only for one that does.
    let index = light_index[t.x + k];
    let at = lights[index].pos;
    let d = at.xyz - world;
    let d2 = dot(d, d);
    let r2 = at.w * at.w;
    if (d2 >= r2) {
      continue;
    }
    let l = lights[index];
    let lit = select(level, 1.0, l.cone.y > 0.5);
    if (lit <= 0.001) {
      continue;
    }
    let ld = d * inverseSqrt(d2 + 1e-6);
    let win = 1.0 - d2 / r2;
    var att = win * win / (1.0 + d2 * 0.45) * lit;
    if (l.spot.w > -1.5) {
      att *= smoothstep(l.spot.w, l.cone.x, dot(-ld, l.spot.xyz));
      if (l.cone.z > 0.5 && att > 0.0) {
        let layer = i32(l.cone.z + 0.5) - 1;
        let lit_share = light_shadow(world, n, ld, l, layer, frag);
        att *= 1.0 - l.cone.w * (1.0 - lit_share);
        if (layer == 0) {
          cast_shadows.y = lit_share;
        } else {
          cast_shadows.z = lit_share;
        }
      }
    }
    let nl = dot(n, ld);
    var w = smoothstep(-0.2, 0.3, nl);
    if (foliage) {
      w = (smoothstep(-0.8, 0.6, nl) * 0.8 + 0.2) * frame.lamps.x;
    }
    sum += l.color.rgb * (l.color.w * att * w);
  }
  return sum * frame.style2.z;
}

fn water_normal(world: vec3f, t: f32) -> vec3f {
  let s = frame.water.z;
  let p = world.xz * s;
  let w = frame.wind.xy * 0.05;
  var g = vec2f(0.0);
  g += vec2f(cos(p.x * 1.3 + p.y * 0.4 + t * 1.1), cos(p.y * 1.7 - p.x * 0.3 + t * 0.9)) * 0.45;
  g += vec2f(cos(p.x * 3.1 - p.y * 2.2 - t * 1.7), cos(p.y * 2.9 + p.x * 1.9 + t * 1.5)) * 0.25;
  let q = p * 2.5 + w * t;
  let e = 0.08;
  let nx = fbm2(q + vec2f(e, 0.0)) - fbm2(q - vec2f(e, 0.0));
  let nz = fbm2(q + vec2f(0.0, e)) - fbm2(q - vec2f(0.0, e));
  g += vec2f(nx, nz) * 4.0;
  // Rain rings.
  let rain = frame.water.w;
  if (rain > 0.0) {
    let cell = floor(world.xz * 2.2);
    let f = fract(world.xz * 2.2) - vec2f(0.5);
    let phase = fract(t * 0.9 + hash2(cell) * 7.0);
    let r = length(f);
    let ring = sin((r - phase * 0.6) * 40.0) * smoothstep(0.6 * phase + 0.08, 0.6 * phase, r) * (1.0 - phase);
    g += f / max(r, 1e-3) * (ring * rain * 1.5);
  }
  return normalize(vec3f(-g.x * 0.045, 1.0, -g.y * 0.045));
}

#import scene_detail

// Blended props (detail3.z = 1) write premultiplied colour; opaque draws keep alpha unused.
fn blended(c: vec4f) -> vec4f {
  if (draw.detail3.z > 0.5) {
    let a = clamp(c.a, 0.0, 1.0);
    return vec4f(c.rgb * a, a);
  }
  return c;
}

// The water's lamp light and key-light shadow depend on where its surface lies and which way it
// faces before its waves move it, so they hold while the cached picture holds: `fs_water_static`
// writes them (lamps rgb, shadow a) to `cache_water` when the picture changes, and `fs_water`
// (`shade_scene` with `WATER_STATICS` set) reads them there instead of looping over the lamps
// and filtering the shadow maps every frame, but where props moved since can have changed
// their shadows (`water_shadows_moved`), where it computes them.
@group(0) @binding(19) var cache_water: texture_2d<f32>;
const WATER_STATICS: bool = false;
// The lamp light and shadow a pixel's shading took (rgb, a), for `fs_water_static`.
var<private> lamps_and_shadow: vec4f;

// The scene's surface shading. `cut` keeps the alpha test and the reflection's clip under the
// water; without them (`fs_main_opaque`) the shader never discards, so the GPU can reject hidden
// fragments before shading them (early depth, or a tiler's hidden-surface removal).
// `reflects` is false in an entry point the mirrored reflection pass never draws with (it binds
// the main view's frame alone), and `water` false in one that never draws the water (the cached
// picture holds none: `draw_world`'s opaque and masked parts leave it out, and a water prop never
// settles). Each entry point passes constants, so the GPU's compiler drops the code they rule out
// and the registers and uniform slots that code took, while the bindings it reads stay in the
// layout every scene pipeline shares.
fn shade_scene(in: VsOut, front: bool, frag: vec4f, cut: bool, reflects: bool, water: bool) -> FsOut {
  // Texture fetches and derivatives first, in uniform control flow. The water (`fs_water`)
  // fetches its normal map only when it has one.
  let bias = frame.style2.x;
  let base_s = textureSampleBias(base_tex, material_sampler, in.uv, bias);
  var nrm_s = vec4f(0.0);
  if (!WATER_STATICS || draw.flags.y > 0.5) {
    nrm_s = textureSampleBias(normal_tex, material_sampler, in.uv, bias * 0.5);
  }
  let emi_s = textureSampleBias(emissive_tex, material_sampler, in.uv, bias);
  let dpx = dpdx(in.world);
  let dpy = dpdy(in.world);
  // The mirrored reflection pass reverses winding: there the faces seen are back faces.
  let reflection_pass = reflects && frame.water_color.w > 0.5;
  var n = normalize(in.normal);
  if (front == reflection_pass) {
    n = -n;
  }
  let geo_n = n;
  var mapped = n;
  if (draw.flags.y > 0.5) {
    let mirror = select(1.0, -1.0, reflection_pass);
    var m = nrm_s.xyz * 2.0 - vec3f(1.0);
    m = vec3f(m.xy * (draw.flags.w * frame.style2.y * mirror), max(m.z, 0.05));
    mapped = perturb(n, in.world, in.uv, normalize(m));
  }
  n = mapped;

  if (cut && reflection_pass && in.world.y < frame.water.x - 0.02) {
    discard;
  }
  var albedo = draw.base_color * in.tint;
  if (draw.flags.x > 0.5) {
    albedo *= base_s;
  }
  if (cut && albedo.a < draw.params.x) {
    discard;
  }
  // `fs_water` (`WATER_STATICS`) draws only the water: there its shading is a constant, and all
  // its branch never reads is dead code the GPU's compilers drop, the colour and glow fetches
  // (left in the source so the material's bind group keeps the scene's layout), surface detail,
  // brushwork and the derivatives they take, and the other surfaces' lighting. The normal map
  // stays: `hemi` and the lamps take the normal it tilts.
  let shading = select(i32(draw.params.w + 0.5), 2, WATER_STATICS);
  let to_cam = frame.camera.xyz - in.world;
  let dist = length(to_cam);
  let v = to_cam / max(dist, 1e-4);
  let l = frame.key_dir.xyz;
  let key = frame.key_color.rgb * frame.key_dir.w;
  let baked_ao = clamp(in.ao, 0.0, 1.0);
  let hemi = mix(frame.amb_ground.rgb, frame.amb_sky.rgb, n.y * 0.5 + 0.5)
    * mix(1.0, baked_ao, frame.lamps.y * 0.35);

  var out: FsOut;
  out.aux = vec4f(geo_n, 1.0 + baked_ao);

  var emissive = draw.emissive.rgb;
  if (draw.flags.z > 0.5) {
    emissive *= emi_s.rgb;
  }
  if (draw.emissive.w > 0.5) {
    emissive *= frame.amb_sky.w * frame.style2.w;
  }

  if (shading == 3) {
    out.color = blended(vec4f(albedo.rgb + emissive, albedo.a));
    return out;
  }

  // Procedural detail (tiles, planks, plaster, stone, lacquer, grassy ground) and brushwork.
  var rough_override = -1.0;
  var brush = 0.0;
  if (shading == 0 && draw.detail0.x > 0.5) {
    let d = surface_detail(in.world, geo_n, n, albedo.rgb, dpx, dpy);
    albedo = vec4f(d.albedo * d.cavity, albedo.a);
    n = d.n;
    rough_override = d.rough;
    // Joints, seams and lattice bars also shade a surface's own glow.
    emissive *= d.cavity;
  }
  if (shading <= 1) {
    let footprint = (length(dpx) + length(dpy)) / max(frame.paint.y, 1e-3);
    let amount = frame.paint.x * (1.0 - smoothstep(0.3, 0.9, footprint));
    if (amount > 0.0) {
      brush = (brush_noise(in.world, geo_n, frame.paint.y) - 0.5) * amount;
      albedo = vec4f(albedo.rgb * (1.0 + brush * 0.24), albedo.a);
    }
  }

  let foliage = shading == 1;
  var sh: f32;
  var lamps: vec3f;
  if (WATER_STATICS) {
    if (water_shadows_moved(in.world, geo_n, dist)) {
      sh = shadow(in.world, geo_n, dist, frag.xy);
      lamps = lamp_light(in.world, n, frag.xy, foliage);
    } else {
      let kept = textureLoad(cache_water, vec2i(frag.xy), 0);
      lamps = kept.rgb;
      sh = kept.a;
    }
  } else {
    sh = shadow(in.world, geo_n, dist, frag.xy);
    lamps = lamp_light(in.world, n, frag.xy, foliage);
  }
  cast_shadows.x = sh;
  cast_shadows.w = 1.0;
  lamps_and_shadow = vec4f(lamps, sh);

  if (water && shading == 2) {
    let t = frame.camera.w;
    let wn = water_normal(in.world, t);
    let cosv = clamp(dot(wn, v), 0.0, 1.0);
    let fres = 0.03 + 0.97 * pow(1.0 - cosv, 5.0);
    // The reflection pass's picture where there is one; the sky (stars, moon, clouds) only where
    // there is not, as it would be thrown away.
    var refl: vec3f;
    if (!reflection_pass && frame.water2.w > 0.5) {
      let uv = frag.xy * frame.viewport.zw + wn.xz * vec2f(0.06, 0.035);
      refl = textureSampleLevel(reflection, clamp_sampler, uv, 0.0).rgb;
    } else {
      refl = sky_color(reflect(-v, wn));
    }
    // Depth below the surface from the ground map: shallows turn clear and green, and foam
    // lines the shore and anything standing in the water.
    var below = 50.0;
    var foam = 0.0;
    let g = ground_texel(in.world);
    if (g.z >= 0.0) {
      let texel = vec2i(g.xy);
      let size = i32(frame.ground.z);
      let gd = textureLoad(ground_depth, clamp(texel, vec2i(0), vec2i(size - 1)), 0);
      if (gd < 1.0) {
        below = frame.water.x - ground_height(gd);
      }
      if (below > 0.02) {
        var near = 0.0;
        var offs = array<vec2i, 4>(vec2i(2, 0), vec2i(-2, 0), vec2i(0, 2), vec2i(0, -2));
        for (var k = 0; k < 4; k++) {
          let q = clamp(texel + offs[k], vec2i(0), vec2i(size - 1));
          let qd = textureLoad(ground_depth, q, 0);
          if (qd < 1.0 && ground_height(qd) > frame.water.x - 0.05) {
            near = 1.0;
          }
        }
        // Foam only where the water meets something: a thin, broken line.
        let churn = 0.35 + 0.65 * smoothstep(0.3, 0.7, noise2(in.world.xz * 4.3 + vec2f(t * 0.35, -t * 0.27)));
        foam = near * smoothstep(0.6, 0.05, below) * churn * frame.shallow.w;
      }
    } else {
      // Beyond the ground map, the coarser world map: shallows and foam along distant shores.
      let w = world_texel(in.world);
      if (w.z >= 0.0) {
        let size = i32(frame.world.z);
        let texel = clamp(vec2i(w.xy), vec2i(0), vec2i(size - 1));
        let wd = textureLoad(world_depth, texel, 0);
        if (wd < 1.0) {
          below = frame.water.x - world_height(wd);
        }
        if (below > 0.02) {
          var near = 0.0;
          var offs = array<vec2i, 4>(vec2i(1, 0), vec2i(-1, 0), vec2i(0, 1), vec2i(0, -1));
          for (var k = 0; k < 4; k++) {
            let q = clamp(texel + offs[k], vec2i(0), vec2i(size - 1));
            let qd = textureLoad(world_depth, q, 0);
            if (qd < 1.0 && world_height(qd) > frame.water.x - 0.05) {
              near = 1.0;
            }
          }
          let churn = 0.35 + 0.65 * smoothstep(0.3, 0.7, noise2(in.world.xz * 1.7 + vec2f(t * 0.3, -t * 0.22)));
          foam = near * smoothstep(1.0, 0.1, below) * churn * frame.shallow.w;
        }
      }
    }
    let shallows = (1.0 - smoothstep(0.0, 1.4, max(below, 0.0))) * 0.45;
    let water_col = mix(frame.water_color.rgb, frame.shallow.rgb, shallows);
    let deep = water_col * (hemi + key * 0.25) + lamps * 0.05;
    var col = mix(deep, refl, clamp(fres * 1.4 + 0.3, 0.0, 1.0) * frame.water.y);
    let r = reflect(-v, wn);
    col += key * (pow(max(dot(r, l), 0.0), 600.0) * 6.0 * sh);
    col += key * (pow(max(dot(r, l), 0.0), 40.0) * 0.08 * sh);
    // Glitter: brief sparks in the sun's (or moon's) path.
    let cell = floor(in.world.xz * 7.0);
    let spark = step(0.92, hash2(cell + vec2f(floor(t * 5.0) * 13.0, 0.0)));
    col += key * (spark * pow(max(dot(r, l), 0.0), 90.0) * 9.0 * frame.water2.x * sh);
    let foam_lit = key * (0.35 + 0.65 * sh) * 0.8 + hemi * 0.9 + lamps * 0.4;
    col = mix(col, vec3f(0.86, 0.93, 0.92) * foam_lit, clamp(foam, 0.0, 0.7));
    out.color = vec4f(col, 1.0);
    out.aux = vec4f(wn, 0.75);
    return out;
  }

  // Tide marks: what stands in the water is wet and stained just above it, the line broken.
  let preset = i32(draw.detail0.x + 0.5);
  if (frame.tide.x > 0.0 && shading == 0 && preset != 6 && preset != 7) {
    let above = in.world.y - frame.water.x + (noise2(in.world.xz * 0.9 + vec2f(in.world.y * 0.3)) - 0.5) * frame.tide.x * 0.35;
    let band = (1.0 - smoothstep(frame.tide.x * 0.55, frame.tide.x, above)) * step(-1.5, above);
    albedo = vec4f(albedo.rgb * mix(vec3f(1.0), frame.tide.yzw, band), albedo.a);
  }

  // Rain darkens and glosses upward-facing surfaces.
  let wet = frame.amb_ground.w * smoothstep(0.5, 0.9, geo_n.y);
  albedo = vec4f(albedo.rgb * (1.0 - 0.38 * wet), albedo.a);

  var rough = clamp(draw.params.y * (1.0 - 0.6 * wet), 0.05, 1.0);
  if (rough_override >= 0.0) {
    rough = min(rough, rough_override);
  }
  if (foliage) {
    out.aux = vec4f(geo_n, 0.5);
  }
  var model = i32(frame.surface_mode.x + 0.5);
  if (draw.detail2.z > 0.5) {
    model = i32(draw.detail2.z + 0.5) - 1;
  }
  var surf: SurfaceInput;
  surf.albedo = albedo.rgb;
  surf.n = n;
  surf.geo_n = geo_n;
  surf.v = v;
  surf.l = l;
  surf.key = key;
  surf.hemi = hemi;
  surf.lamps = lamps;
  surf.sh = sh;
  surf.rough = rough;
  surf.metallic = draw.params.z;
  surf.ao = baked_ao;
  surf.brush = brush;
  surf.foliage = foliage;
  surf.model = model;
  surf.lobe_weight = draw.detail2.w;
  surf.lobe_param = draw.detail3.w;
  var col = evaluate_surface(surf);

  // Wet sheen: the sky mirrored on puddled ground.
  if (wet > 0.0) {
    let r = reflect(-v, n);
    let f = 0.04 + 0.96 * pow(1.0 - clamp(dot(n, v), 0.0, 1.0), 5.0);
    col += (sky_base(r) * 0.6 + lamps * 0.05) * (wet * f);
  }

  col += emissive;
  if (reflection_pass) {
    col = mix(col, fog_color(-v), fog_amount(frame.camera.xyz, in.world, dist));
  }
  out.color = blended(vec4f(col, albedo.a));
  return out;
}

@fragment fn fs_main(in: VsOut, @builtin(front_facing) front: bool, @builtin(position) frag: vec4f) -> FsOut {
  return shade_scene(in, front, frag, true, true, true);
}

// Opaque world surfaces and props in the main view, drawn nearest first.
@fragment fn fs_main_opaque(in: VsOut, @builtin(front_facing) front: bool, @builtin(position) frag: vec4f) -> FsOut {
  return shade_scene(in, front, frag, false, false, true);
}

// The cached picture's shading: as `fs_main` and `fs_main_opaque` in the main view, with the
// shadows moved props can change; the picture holds no water.
struct CacheOut {
  @location(0) color: vec4f,
  @location(1) aux: vec4f,
  @location(2) shadow: vec4f,
};

fn cache_out(o: FsOut) -> CacheOut {
  var c: CacheOut;
  c.color = o.color;
  c.aux = o.aux;
  c.shadow = cast_shadows;
  return c;
}

@fragment fn fs_main_cache(in: VsOut, @builtin(front_facing) front: bool, @builtin(position) frag: vec4f) -> CacheOut {
  return cache_out(shade_scene(in, front, frag, true, false, false));
}

@fragment fn fs_main_opaque_cache(in: VsOut, @builtin(front_facing) front: bool, @builtin(position) frag: vec4f) -> CacheOut {
  return cache_out(shade_scene(in, front, frag, false, false, false));
}

// The water's lamp light and shadow, shaded as `fs_main_opaque` shades the water, into
// `cache_water`.
@fragment fn fs_water_static(in: VsOut, @builtin(front_facing) front: bool, @builtin(position) frag: vec4f) -> @location(0) vec4f {
  _ = shade_scene(in, front, frag, false, false, true);
  return lamps_and_shadow;
}

// The water in a frame that keeps the cached picture, its lamp light and shadow read from
// `cache_water` where no prop moved since they were kept (compiled with `WATER_STATICS`; its
// group binds `reshade` to the water's own block).
@fragment fn fs_water(in: VsOut, @builtin(front_facing) front: bool, @builtin(position) frag: vec4f) -> FsOut {
  return shade_scene(in, front, frag, false, false, true);
}

#import scene_grass
