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
  var m = frame.light_shadow0;
  if (layer == 1) {
    m = frame.light_shadow1;
  }
  // A texel's width where the point is (the map spans the cone across), and the point moved off
  // its surface by about that much, more where the light grazes it.
  let c = l.spot.w;
  let spread = sqrt(max(1.0 - c * c, 0.0)) / max(c, 0.05);
  let texel = 2.0 * length(l.pos.xyz - world) * spread * frame.light_shadow.x;
  let graze = 1.0 - abs(dot(n, ld));
  let p = m * vec4f(world + n * (texel * (1.0 + 2.0 * graze)), 1.0);
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

fn lamp_light(world: vec3f, n: vec3f, frag: vec2f, foliage: bool) -> vec3f {
  // The lamps follow the time of day; a game's own lights shine at every hour.
  let level = frame.amb_sky.w;
  let size = f32(frame.tiles.z);
  let tx = min(u32(frag.x / size), frame.tiles.x - 1u);
  let ty = min(u32(frag.y / size), frame.tiles.y - 1u);
  let t = tiles[ty * frame.tiles.x + tx];
  var sum = vec3f(0.0);
  for (var k = 0u; k < t.y; k++) {
    let l = lights[light_index[t.x + k]];
    let lit = select(level, 1.0, l.cone.y > 0.5);
    let d = l.pos.xyz - world;
    let d2 = dot(d, d);
    let r2 = l.pos.w * l.pos.w;
    if (d2 >= r2 || lit <= 0.001) {
      continue;
    }
    let ld = d * inverseSqrt(d2 + 1e-6);
    let win = 1.0 - d2 / r2;
    var att = win * win / (1.0 + d2 * 0.45) * lit;
    if (l.spot.w > -1.5) {
      att *= smoothstep(l.spot.w, l.cone.x, dot(-ld, l.spot.xyz));
      if (l.cone.z > 0.5 && att > 0.0) {
        let lit_share = light_shadow(world, n, ld, l, i32(l.cone.z + 0.5) - 1, frag);
        att *= 1.0 - l.cone.w * (1.0 - lit_share);
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

// ---------------------------------------------------------------- painted surface detail

// Coordinates on a surface's dominant plane.
fn plane_uv(p: vec3f, n: vec3f) -> vec2f {
  let a = abs(n);
  if (a.x > a.y && a.x > a.z) {
    return p.zy;
  }
  if (a.z > a.y) {
    return p.xy;
  }
  return p.xz;
}

// Brush strokes in 0..1: long soft dabs on the surface, fixed in the world so they never swim.
fn brush_noise(p: vec3f, n: vec3f, size: f32) -> f32 {
  let q = plane_uv(p, n) / size;
  let r = vec2f(q.x * 0.8 + q.y * 0.6, q.y * 0.8 - q.x * 0.6);
  return noise2(vec2f(r.x * 0.45, r.y * 1.8)) * 0.6 + noise2(vec2f(r.x * 1.1, r.y * 4.2) + vec2f(7.3, 1.9)) * 0.4;
}

struct Detail {
  albedo: vec3f,
  n: vec3f,
  cavity: f32,  // darkening in gutters, joints and seams
  rough: f32,   // replacement roughness, or < 0 to keep the material's
};

fn no_detail(albedo: vec3f, n: vec3f) -> Detail {
  var o: Detail;
  o.albedo = albedo;
  o.n = n;
  o.cavity = 1.0;
  o.rough = -1.0;
  return o;
}

// How many pattern cells of `size` metres along `axis` one pixel covers.
fn cells_per_pixel(axis: vec3f, size: f32, dpx: vec3f, dpy: vec3f) -> f32 {
  return (abs(dot(dpx, axis)) + abs(dot(dpy, axis))) / size;
}

// Rounded roof tiles in columns running down the slope, lapped row over row.
fn roof_tiles(p: vec3f, geo: vec3f, n: vec3f, albedo: vec3f, dpx: vec3f, dpy: vec3f) -> Detail {
  var o = no_detail(albedo, n);
  if (geo.y < 0.12) {
    return o;
  }
  let h = vec2f(geo.x, geo.z);
  let hl = length(h);
  var down = vec3f(0.0, 0.0, 1.0);
  if (hl > 0.03) {
    down = vec3f(h.x / hl, 0.0, h.y / hl);
  }
  let e = vec3f(-down.z, 0.0, down.x);
  var s = normalize(cross(e, geo));
  if (s.y > 0.0) {
    s = -s;
  }
  let width = draw.detail0.y;
  let row = draw.detail0.z;
  let fade = 1.0 - smoothstep(0.25, 0.6, max(cells_per_pixel(e, width, dpx, dpy), cells_per_pixel(s, row, dpx, dpy)));
  let relief = draw.detail0.w * fade;
  let u = dot(p, e) / width;
  let v = dot(p, s) / row;
  let x = fract(u) * 2.0 - 1.0;
  let crest = sqrt(max(1.0 - x * x, 0.0));
  let fv = fract(v);
  let lap = smoothstep(0.0, 0.18, fv);
  let lip = smoothstep(0.78, 1.0, fv);
  o.n = normalize(n + e * (clamp(x / max(crest, 0.2), -2.5, 2.5) * 0.3 * relief) + s * ((lip * 0.5 - (1.0 - lap) * 0.35) * relief));
  let gutter = smoothstep(0.0, 0.45, crest);
  // Far away the pattern averages out to a slightly darker roof.
  o.cavity = mix(0.86, mix(0.5, 1.0, lap) * mix(0.62, 1.0, gutter), fade);
  let id = vec2f(floor(u), floor(v));
  let r = hash2(id) - 0.5;
  let r2 = hash2(id + vec2f(17.0, 3.0)) - 0.5;
  o.albedo = albedo * (1.0 + r * draw.detail1.x * 2.0 * fade) * (1.0 + vec3f(r2, r2 * 0.6, -r2) * 0.12 * fade);
  return o;
}

// Planks with staggered butt joints and grain, along world x (or z for faces looking along x).
fn wood(p: vec3f, geo: vec3f, n: vec3f, albedo: vec3f, dpx: vec3f, dpy: vec3f) -> Detail {
  var o = no_detail(albedo, n);
  let a = abs(geo);
  var along = vec3f(1.0, 0.0, 0.0);
  if (a.x > a.z && a.x > a.y * 0.5) {
    along = vec3f(0.0, 0.0, 1.0);
  }
  let across = normalize(cross(geo, along) + vec3f(0.0, 1e-4, 0.0));
  let width = draw.detail0.y;
  let length_m = draw.detail0.z;
  let fade = 1.0 - smoothstep(0.3, 0.7, cells_per_pixel(across, width, dpx, dpy));
  let u = dot(p, across) / width;
  let plank = floor(u);
  let w = dot(p, along) / length_m + hash2(vec2f(plank, 7.0));
  let id = vec2f(plank, floor(w));
  let r = hash2(id) - 0.5;
  let fu = fract(u);
  let seam = smoothstep(0.0, 0.07, fu) * smoothstep(1.0, 0.93, fu);
  let fw = fract(w);
  let butt = smoothstep(0.0, 0.012, fw) * smoothstep(1.0, 0.988, fw);
  let along_m = dot(p, along);
  let g = noise2(vec2f(along_m * 0.9 / width, fu * 7.0 + r * 31.0)) * 0.7 + noise2(vec2f(along_m * 3.1 / width, fu * 23.0 + r * 11.0)) * 0.3;
  let relief = draw.detail0.w * fade;
  o.cavity = mix(1.0, mix(0.55, 1.0, seam * butt), relief);
  o.albedo = albedo * (1.0 + r * draw.detail1.x * 2.0 * fade) * (1.0 + (g - 0.5) * 0.35 * relief);
  o.n = normalize(n + across * ((smoothstep(0.85, 1.0, fu) - smoothstep(0.15, 0.0, fu)) * 0.25 * relief));
  return o;
}

// Soft mottling, plus faint rain streaks on walls.
fn plaster(p: vec3f, geo: vec3f, n: vec3f, albedo: vec3f) -> Detail {
  var o = no_detail(albedo, n);
  let size = draw.detail0.y;
  let m = fbm2(plane_uv(p, geo) / size);
  var streak = 0.0;
  if (abs(geo.y) < 0.4) {
    let side = dot(p, normalize(vec3f(-geo.z, 0.0, geo.x) + vec3f(1e-4, 0.0, 0.0)));
    streak = smoothstep(0.55, 0.85, noise2(vec2f(side * 3.0 / size, p.y * 0.35)));
  }
  let k = draw.detail0.w;
  o.albedo = albedo * (1.0 + (m - 0.5) * draw.detail1.x * 2.0 * k - streak * 0.08 * k);
  return o;
}

// Irregular stones (a jittered cell pattern) with dark joints.
fn stone(p: vec3f, geo: vec3f, n: vec3f, albedo: vec3f, dpx: vec3f, dpy: vec3f) -> Detail {
  var o = no_detail(albedo, n);
  let size = draw.detail0.y;
  let fade = 1.0 - smoothstep(0.25, 0.6, (length(dpx) + length(dpy)) / size);
  let uv = plane_uv(p, geo) / size;
  let c = floor(uv);
  let f = fract(uv);
  var d1 = 8.0;
  var d2 = 8.0;
  var id = vec2f(0.0);
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let cell = c + vec2f(f32(i), f32(j));
      let site = vec2f(f32(i), f32(j)) + vec2f(hash2(cell), hash2(cell + vec2f(31.0, 17.0))) * 0.8 + vec2f(0.1) - f;
      let d = dot(site, site);
      if (d < d1) {
        d2 = d1;
        d1 = d;
        id = cell;
      } else if (d < d2) {
        d2 = d;
      }
    }
  }
  let edge = sqrt(d2) - sqrt(d1);
  let joint = smoothstep(0.03, 0.14, edge);
  let relief = draw.detail0.w * fade;
  o.cavity = mix(1.0, mix(0.42, 1.0, joint), relief);
  o.albedo = albedo * (1.0 + (hash2(id) - 0.5) * draw.detail1.x * 2.0 * fade);
  return o;
}

// Grassy ground: the grass colour in large soft patches, so it meets the blades seamlessly.
fn grass_ground(p: vec3f, n: vec3f, albedo: vec3f) -> Detail {
  var o = no_detail(albedo, n);
  var base = albedo;
  if (draw.detail2.x > 0.5) {
    // Mostly the grass colour, keeping a little of the ground texture's markings.
    base = mix(albedo, draw.detail1.yzw, 0.75);
  }
  let m = fbm2(p.xz * 0.11);
  let m2 = noise2(p.xz * 0.9);
  let dry = vec3f(1.12, 1.06, 0.7);
  o.albedo = base * mix(vec3f(0.78), dry, smoothstep(0.35, 0.75, m)) * (0.92 + 0.16 * m2);
  return o;
}

// Terrain: grassy ground (as grass_ground) laid in broad meadow patches, giving way to beach
// sand near the water (wet and dark at the water line) and to bare rock in layers on steep
// slopes. Where sand and rock begin comes from terrain_cover, as in the ground map.
fn terrain_ground(p: vec3f, geo: vec3f, n: vec3f, albedo: vec3f, dpx: vec3f, dpy: vec3f) -> Detail {
  var o = grass_ground(p, n, albedo);
  let size = max(frame.terrain.w, 1.0);
  let broad = smoothstep(0.36, 0.66, fbm2(p.xz / size + vec2f(41.0, 13.0)));
  let meadow = frame.meadow.rgb * (0.9 + 0.2 * noise2(p.xz * 0.35));
  o.albedo = mix(o.albedo, meadow, broad * frame.sand.w);
  let c = terrain_cover(p, geo.y, frame.water.x, frame.terrain);
  if (c.x + c.y <= 0.0) {
    return o;
  }
  let above = p.y - frame.water.x;
  let dry = smoothstep(0.0, max(frame.terrain.x * 0.45, 0.05), above);
  let ripple = 0.94 + 0.06 * sin(dot(p.xz, vec2f(1.7, 0.6)) * 3.0 + noise2(p.xz * 0.8) * 4.0);
  let sand = mix(frame.wet_sand.rgb, frame.sand.rgb, dry) * ripple * (0.95 + 0.1 * noise2(p.xz * 2.3));
  // Rock: big angular facets (a jittered cell pattern on the face, stretched along it), each
  // its own tone, warmth and tilt, with thin dark cracks between them and faint sedimentary
  // bands; the cracks fade out where a facet is only a few pixels across.
  let face = plane_uv(p, geo);
  let q = vec2f(face.x / 6.5, (p.y + noise2(p.xz * 0.07) * 2.5) / 3.6);
  let cq = floor(q);
  let fq = fract(q);
  var d1 = 9.0;
  var d2 = 9.0;
  var id = vec2f(0.0);
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let cell = cq + vec2f(f32(i), f32(j));
      let site = vec2f(f32(i), f32(j)) + vec2f(hash2(cell), hash2(cell + vec2f(31.0, 17.0))) - fq;
      let d = dot(site, site);
      if (d < d1) {
        d2 = d1;
        d1 = d;
        id = cell;
      } else if (d < d2) {
        d2 = d;
      }
    }
  }
  let fade = 1.0 - smoothstep(0.08, 0.3, (length(dpx) + length(dpy)) / 3.6);
  let crack = mix(1.0, smoothstep(0.006, 0.045, sqrt(d2) - sqrt(d1)), fade);
  let tone = hash2(id) - 0.5;
  let warm = hash2(id + vec2f(5.0, 9.0)) - 0.5;
  let band = sin(p.y * 0.85 + noise2(p.xz * 0.05) * 3.0) * 0.07;
  let grain = noise2(face * 1.7) * 0.6 + noise2(face * 5.3) * 0.4;
  let rock = frame.rock.rgb * (1.0 + tone * 0.32 + band) * (vec3f(1.0) + vec3f(warm, warm * 0.3, -warm) * 0.14) * (0.9 + 0.2 * grain);
  let tilt = vec3f(hash2(id + vec2f(1.3, 0.0)) - 0.5, (hash2(id + vec2f(0.0, 4.1)) - 0.5) * 0.6, hash2(id + vec2f(2.7, 0.0)) - 0.5) * 0.7;
  o.albedo = mix(mix(o.albedo, sand, c.x), rock, c.y);
  o.cavity = mix(1.0, mix(0.62, 1.0, crack), c.y);
  o.n = normalize(mix(o.n, normalize(n + tilt), c.y));
  return o;
}

// A lattice of thin bars over the surface (a lantern's frame, paper ribs): cells `size u` wide
// and `size v` tall on the face's plane (a zero size leaves that direction's bars out), bars
// `strength` of a cell wide in the rule's colour (dark wood by default). The cavity term carries
// the bars, so they also shade a lamp's glow. Bars thinner than a pixel fade to their average
// cover rather than shimmer.
fn lattice(p: vec3f, geo: vec3f, n: vec3f, albedo: vec3f, dpx: vec3f, dpy: vec3f) -> Detail {
  var o = no_detail(albedo, n);
  let raw = vec2f(draw.detail0.y, draw.detail0.z);
  let used = vec2f(step(1e-4, raw.x), step(1e-4, raw.y));
  let size = max(raw, vec2f(1e-3));
  let q = plane_uv(p, geo) / size;
  let f = abs(fract(q) - vec2f(0.5));
  let half = 0.5 - draw.detail0.w * 0.5;
  // The pattern's footprint per pixel, from the world-space derivatives taken up front.
  let aa = max((abs(plane_uv(dpx, geo)) + abs(plane_uv(dpy, geo))) / size * 0.75, vec2f(1e-4));
  let bars = vec2f(smoothstep(half - aa.x, half + aa.x, f.x), smoothstep(half - aa.y, half + aa.y, f.y)) * used;
  // Where cells shrink under a few pixels the pattern averages to its coverage.
  let fade = 1.0 - smoothstep(0.15, 0.4, max(aa.x * used.x, aa.y * used.y));
  let average = draw.detail0.w * 0.6 * max(used.x, used.y);
  let cover = mix(average, max(bars.x, bars.y), fade);
  var wood = vec3f(0.06, 0.035, 0.02);
  if (draw.detail2.x > 0.5) {
    wood = draw.detail1.yzw;
  }
  o.albedo = mix(albedo, wood, cover);
  o.cavity = 1.0 - cover * 0.92;
  return o;
}

fn surface_detail(p: vec3f, geo: vec3f, n: vec3f, albedo: vec3f, dpx: vec3f, dpy: vec3f) -> Detail {
  let preset = i32(draw.detail0.x + 0.5);
  var base = albedo;
  if (draw.detail2.x > 0.5 && preset != 6 && preset != 7 && preset != 8) {
    base = draw.detail1.yzw;
  }
  if (preset == 1) {
    return roof_tiles(p, geo, n, base, dpx, dpy);
  }
  if (preset == 2) {
    return wood(p, geo, n, base, dpx, dpy);
  }
  if (preset == 3) {
    return plaster(p, geo, n, base);
  }
  if (preset == 4) {
    return stone(p, geo, n, base, dpx, dpy);
  }
  if (preset == 6) {
    return grass_ground(p, n, albedo);
  }
  if (preset == 7) {
    return terrain_ground(p, geo, n, albedo, dpx, dpy);
  }
  if (preset == 8) {
    return lattice(p, geo, n, albedo, dpx, dpy);
  }
  var o = no_detail(base, n);
  if (preset == 5) {
    o.rough = 0.12;
  }
  return o;
}

// Blended props (detail3.z = 1) write premultiplied colour; opaque draws keep alpha unused.
fn blended(c: vec4f) -> vec4f {
  if (draw.detail3.z > 0.5) {
    let a = clamp(c.a, 0.0, 1.0);
    return vec4f(c.rgb * a, a);
  }
  return c;
}

// The scene's surface shading. `cut` keeps the alpha test and the reflection's clip under the
// water; without them (`fs_main_opaque`) the shader never discards, so the GPU can reject hidden
// fragments before shading them (early depth, or a tiler's hidden-surface removal).
fn shade_scene(in: VsOut, front: bool, frag: vec4f, cut: bool) -> FsOut {
  // Texture fetches and derivatives first, in uniform control flow.
  let bias = frame.style2.x;
  let base_s = textureSampleBias(base_tex, material_sampler, in.uv, bias);
  let nrm_s = textureSampleBias(normal_tex, material_sampler, in.uv, bias * 0.5);
  let emi_s = textureSampleBias(emissive_tex, material_sampler, in.uv, bias);
  let dpx = dpdx(in.world);
  let dpy = dpdy(in.world);
  // The mirrored reflection pass reverses winding: there the faces seen are back faces.
  let reflection_pass = frame.water_color.w > 0.5;
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
  let shading = i32(draw.params.w + 0.5);
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

  let sh = shadow(in.world, geo_n, dist, frag.xy);
  let foliage = shading == 1;
  let lamps = lamp_light(in.world, n, frag.xy, foliage);

  if (shading == 2) {
    let t = frame.camera.w;
    let wn = water_normal(in.world, t);
    let cosv = clamp(dot(wn, v), 0.0, 1.0);
    let fres = 0.03 + 0.97 * pow(1.0 - cosv, 5.0);
    var refl = sky_color(reflect(-v, wn));
    if (!reflection_pass && frame.water2.w > 0.5) {
      let uv = frag.xy * frame.viewport.zw + wn.xz * vec2f(0.06, 0.035);
      refl = textureSampleLevel(reflection, clamp_sampler, uv, 0.0).rgb;
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
  return shade_scene(in, front, frag, true);
}

// Opaque world surfaces in the main view, drawn nearest first.
@fragment fn fs_main_opaque(in: VsOut, @builtin(front_facing) front: bool, @builtin(position) frag: vec4f) -> FsOut {
  return shade_scene(in, front, frag, false);
}

// ---------------------------------------------------------------- grass and flowers
// Instanced blades on a world-aligned grid around the camera, kept where the ground map says
// grass grows (its top surface is a grassy material), thinned and widened with distance.

const GRASS_SPACING = 0.13;

struct GrassSite {
  base: vec3f,
  keep: f32,     // 0 when the blade is not drawn
  widen: f32,    // blades grow wider where they are sparse
  seed: u32,
};

fn grass_site(ii: u32) -> GrassSite {
  var o: GrassSite;
  o.keep = 0.0;
  o.widen = 1.0;
  let cam = frame.camera.xyz;
  // The renderer draws side * side instances on a grid centred under the camera.
  let side = max(u32(frame.water2.y), 1u);
  let half = i32(side / 2u);
  let cx = i32(floor(cam.x / GRASS_SPACING)) + i32(ii % side) - half;
  let cz = i32(floor(cam.z / GRASS_SPACING)) + i32(ii / side) - half;
  let seed = hash_u((bitcast<u32>(cx) * 73856093u) ^ hash_u(bitcast<u32>(cz) * 19349663u + 7u));
  o.seed = seed;
  let jx = rand01(seed);
  let jz = rand01(seed + 1u);
  let x = (f32(cx) + jx) * GRASS_SPACING;
  let z = (f32(cz) + jz) * GRASS_SPACING;
  // Density: full near the camera, thinning with distance.
  let g = ground_texel(vec3f(x, 0.0, z));
  if (g.z < 0.0) {
    return o;
  }
  let size = i32(frame.ground.z);
  let texel = clamp(vec2i(g.xy), vec2i(0), vec2i(size - 1));
  let mask = textureLoad(ground_mask, texel, 0);
  if (mask.r < 0.5) {
    return o;
  }
  let gd = textureLoad(ground_depth, texel, 0);
  let y = ground_height(gd);
  o.base = vec3f(x, y, z);
  let d = length(o.base - cam);
  let density = clamp(pow(9.0 / max(d, 1e-3), 1.5), 0.05, 1.0) * frame.ground.w;
  if (rand01(seed + 2u) > density) {
    return o;
  }
  // Skip blades well outside the view.
  let c = frame.view_proj * vec4f(o.base + vec3f(0.0, frame.grass.x * 0.5, 0.0), 1.0);
  if (c.w <= 0.0 || any(abs(c.xy) > vec2f(c.w * 1.15 + 0.5))) {
    return o;
  }
  o.keep = 1.0;
  o.widen = min(inverseSqrt(max(density, 0.05)), 3.0);
  return o;
}

fn grass_wind(base: vec3f, seed: u32) -> vec3f {
  let w = frame.wind.xy;
  let speed = length(w);
  var dir = vec2f(0.7, 0.7);
  if (speed > 1e-3) {
    dir = w / speed;
  }
  let t = frame.camera.w;
  let phase = dot(base.xz, dir) * 0.45 + rand01(seed + 9u) * 1.5;
  let gust = 0.35 + 0.65 * (0.5 + 0.5 * sin(t * 1.3 - phase));
  let flutter = sin(t * 4.1 + rand01(seed + 10u) * 6.28) * 0.15;
  let amount = (0.12 + min(speed * 0.09, 0.5)) * frame.grass.w;
  return vec3f(dir.x, 0.0, dir.y) * ((gust + flutter) * amount);
}

struct GrassOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) color: vec3f,
  @location(3) t: f32,
};

fn hidden_grass() -> GrassOut {
  var o: GrassOut;
  o.clip = vec4f(0.0, 0.0, -1.0, 1.0);
  return o;
}

// Heights along a blade and the corners of its two quads and tip triangle.
const BLADE_T = array<f32, 4>(0.0, 0.45, 0.8, 1.0);
const BLADE_CORNERS = array<vec2f, 15>(
  vec2f(-1.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(-1.0, 0.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  vec2f(-1.0, 1.0), vec2f(1.0, 1.0), vec2f(1.0, 2.0), vec2f(-1.0, 1.0), vec2f(1.0, 2.0), vec2f(-1.0, 2.0),
  vec2f(-1.0, 2.0), vec2f(1.0, 2.0), vec2f(0.0, 3.0)
);

@vertex fn vs_grass(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> GrassOut {
  let site = grass_site(ii);
  if (site.keep <= 0.0) {
    return hidden_grass();
  }
  let seed = site.seed;
  let corner = BLADE_CORNERS[vi];
  let t = BLADE_T[u32(corner.y)];
  let yaw = rand01(seed + 3u) * 6.2831;
  let face = vec3f(cos(yaw), 0.0, sin(yaw));
  let right = vec3f(-face.z, 0.0, face.x);
  let height = frame.grass.x * (0.55 + 0.9 * rand01(seed + 4u));
  let width = frame.grass.y * (0.7 + 0.6 * rand01(seed + 5u)) * site.widen;
  let lean = face * (0.25 * (rand01(seed + 6u) - 0.3) * height);
  let bend = (lean + grass_wind(site.base, seed) * height) * (t * t);
  var p = site.base + vec3f(0.0, height * t, 0.0) + bend;
  p += right * (corner.x * 0.5 * width * (1.0 - 0.85 * t));
  // Lighting normal: the blade's face, bent towards the sky so meadows read soft and bright.
  let blade_n = normalize(cross(right, vec3f(0.0, height, 0.0) + bend * 2.0));
  var o: GrassOut;
  o.clip = frame.view_proj * vec4f(p, 1.0);
  o.world = p;
  o.normal = normalize(mix(blade_n, vec3f(0.0, 1.0, 0.0), 0.55));
  let patch = noise2(site.base.xz * 0.12);
  let tint = 0.85 + 0.3 * rand01(seed + 7u);
  let dry = mix(vec3f(1.0), vec3f(1.18, 1.08, 0.62), smoothstep(0.45, 0.8, patch));
  o.color = mix(frame.grass_base.rgb, frame.grass_tip.rgb, pow(t, 0.8)) * tint * dry;
  o.t = t;
  return o;
}

fn grass_light(world: vec3f, n_in: vec3f, albedo: vec3f, t: f32, frag: vec2f, front: bool) -> vec3f {
  let to_cam = frame.camera.xyz - world;
  let dist = length(to_cam);
  let v = to_cam / max(dist, 1e-4);
  var n = normalize(n_in);
  if (dot(n, v) < 0.0 && n.y < 0.9) {
    n = normalize(vec3f(-n.x, n.y, -n.z));
  }
  let l = frame.key_dir.xyz;
  let key = frame.key_color.rgb * frame.key_dir.w;
  let sh = shadow_taps(world, vec3f(0.0, 1.0, 0.0), dist, frag, 4);
  let ndl = dot(n, l);
  let lit = smoothstep(0.0, 1.0, (ndl + 0.55) / 1.55) * sh;
  let hemi = mix(frame.amb_ground.rgb, frame.amb_sky.rgb, n.y * 0.5 + 0.5);
  let dark = mix(vec3f(1.0), frame.shadow_tint.rgb, frame.shadow_tint.w);
  let ao = mix(0.38, 1.0, smoothstep(0.0, 0.75, t));
  let back = pow(clamp(dot(-l, v), 0.0, 1.0), 3.0) * frame.style.w * sh * t;
  let lamps = lamp_light(world, n, frag, true);
  let wet = 1.0 - 0.3 * frame.amb_ground.w;
  return albedo * wet * (key * lit + hemi * mix(dark, vec3f(1.0), lit) * ao + lamps * ao + key * back * 0.7);
}

@fragment fn fs_grass(in: GrassOut, @builtin(front_facing) front: bool, @builtin(position) frag: vec4f) -> FsOut {
  var out: FsOut;
  out.color = vec4f(grass_light(in.world, in.normal, in.color, in.t, frag.xy, front), 1.0);
  // aux.w = 0: no ambient occlusion or outlines on blades.
  out.aux = vec4f(0.0, 1.0, 0.0, 0.0);
  return out;
}

struct FlowerOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) color: vec3f,
  @location(2) uv: vec2f,
};

// Flower heads: small upturned discs on some of the grass sites, above the blades.
@vertex fn vs_flower(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FlowerOut {
  var o: FlowerOut;
  o.clip = vec4f(0.0, 0.0, -1.0, 1.0);
  let site = grass_site(ii);
  if (site.keep <= 0.0 || rand01(site.seed + 11u) > frame.grass.z) {
    return o;
  }
  let seed = site.seed;
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let c = corners[vi];
  let height = frame.grass.x * (0.75 + 0.6 * rand01(seed + 12u));
  let centre = site.base + vec3f(0.0, height, 0.0) + grass_wind(site.base, seed) * height;
  let to_cam = normalize(frame.camera.xyz - centre);
  // Tilted between facing up and facing the camera, so heads read from low and high views.
  let facing = normalize(mix(vec3f(0.0, 1.0, 0.0), to_cam, 0.5));
  var side = cross(facing, vec3f(0.0, 0.0, 1.0));
  if (length(side) < 1e-3) {
    side = vec3f(1.0, 0.0, 0.0);
  }
  side = normalize(side);
  let up = cross(side, facing);
  let r = 0.035 * (0.75 + 0.5 * rand01(seed + 13u)) * min(site.widen, 2.0);
  let p = centre + (side * c.x + up * c.y) * r;
  o.clip = frame.view_proj * vec4f(p, 1.0);
  o.world = p;
  let k = min(u32(rand01(seed + 14u) * 4.0), 3u);
  o.color = frame.flowers[k].rgb;
  o.uv = c;
  return o;
}

@fragment fn fs_flower(in: FlowerOut, @builtin(position) frag: vec4f) -> FsOut {
  let r = length(in.uv);
  let a = atan2(in.uv.y, in.uv.x);
  let petals = 0.62 + 0.38 * abs(cos(a * 2.5));
  if (r > petals) {
    discard;
  }
  let heart = smoothstep(0.32, 0.18, r);
  let albedo = mix(in.color, vec3f(1.0, 0.82, 0.25), heart);
  var out: FsOut;
  out.color = vec4f(grass_light(in.world, vec3f(0.0, 1.0, 0.0), albedo, 1.0, frag.xy, true), 1.0);
  out.aux = vec4f(0.0, 1.0, 0.0, 0.0);
  return out;
}
