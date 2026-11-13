// Shared by every stylized-renderer shader: the per-view frame block, hashing and noise, the
// painterly sky and the aerial-perspective fog. Prepended or imported via `#import common`.
#define_import_path common

struct Frame {
  view_proj: mat4x4f,
  inv_view_proj: mat4x4f,
  shadow0: mat4x4f,
  shadow1: mat4x4f,
  ground_vp: mat4x4f,   // top-down projection of the ground map (depth 0 at the top)
  camera: vec4f,        // xyz eye, w = seconds
  key_dir: vec4f,       // xyz towards the key light (sun or moon), w = intensity
  key_color: vec4f,     // rgb, w = unused
  amb_sky: vec4f,       // rgb ambient from above (already scaled), w = lamp level
  amb_ground: vec4f,    // rgb ambient from below, w = wetness
  shadow_tint: vec4f,   // rgb tint of the shaded side, w = shadow strength
  ramp: vec4f,          // terminator centre, softness, rim strength, rim power
  style: vec4f,         // specular, specular size, foliage wrap, foliage translucency
  style2: vec4f,        // texture lod bias, normal map strength, lamp light scale, lamp glow scale
  water: vec4f,         // plane height, reflection strength, ripple scale, rain
  water_color: vec4f,   // rgb deep water, w = 1 in the reflection pass
  zenith: vec4f,        // rgb, w = stars
  horizon: vec4f,       // rgb, w = cloud cover
  glow: vec4f,          // rgb, w = sun disc visibility
  glow_dir: vec4f,      // xyz, w = moon disc visibility
  sun_dir: vec4f,       // xyz, w = cloud scale
  moon_dir: vec4f,      // xyz, w = moon size
  cloud_lit: vec4f,     // rgb, w = star density
  cloud_shadow: vec4f,  // rgb, w = petals
  fog: vec4f,           // rgb, w = density
  fog2: vec4f,          // height base, height falloff, mist, fog strength
  wind: vec4f,          // xz wind velocity in xy, z = time scale, w = unused
  viewport: vec4f,      // width, height, 1 / width, 1 / height
  cascades: vec4f,      // cascade 0 far distance, normal offset 0, normal offset 1, near plane
  shadow_params: vec4f, // filter radius (uv) cascade 0, cascade 1, depth bias 0, depth bias 1
  ground: vec4f,        // ground map: top height, depth range (m), texels per side, grass density
  paint: vec4f,         // brush strength, brush stroke size (m), light shafts, unused
  shallow: vec4f,       // rgb shallow water, w = foam
  water2: vec4f,        // glitter, grass grid rows, 1 when prop shadows drawn, 1 when reflection drawn
  grass: vec4f,         // blade height, blade width, flower fraction, sway
  grass_base: vec4f,    // rgb blade root colour
  grass_tip: vec4f,     // rgb blade tip colour
  flowers: array<vec4f, 4>, // rgb flower colours
  world_vp: mat4x4f,    // top-down projection of the world map (the whole world, drawn once)
  world: vec4f,         // world map: top height, depth range (m), texels per side, 1 when drawn
  terrain: vec4f,       // beach height above the water (m), rock slope start, rock slope full, patch size (m)
  sand: vec4f,          // rgb dry sand, w = strength of the broad meadow patches
  wet_sand: vec4f,      // rgb sand at the water line, w = water plane height (mirrors water.x)
  rock: vec4f,          // rgb bare rock, w = shadow spin seed (mirrors wind.w)
  meadow: vec4f,        // rgb second grass tone of the broad patches, w = 1 in reflection pass
  lamps: vec4f,         // lamp light on foliage (scale), baked AO strength, cluster near (m), cluster slices per doubling
  tide: vec4f,          // tide-mark height above the water (m, 0 = none), rgb stain multiplier
  tiles: vec4u,         // tiles across, tiles down, tile size in pixels, light count
  light_shadow0: mat4x4f, // world to the shadow maps of a game's shadowed lights (layers 0, 1):
  light_shadow1: mat4x4f, // perspective from each light, depth 0 at it to 1 at its range
  light_shadow: vec4f,  // 1 / texels per side, 1-based light indices of layers 0 & 1, w = 1 when prop shadows drawn
  surface_mode: vec4f,  // default SurfaceModel (0..5), clearcoat, sheen, subsurface
  subsurface_color: vec4f, // rgb subsurface scatter tint, w = clearcoat roughness
  grass_grid: vec4f,    // the part of the grass grid drawn: first column, first row, columns, unused
};

@group(0) @binding(0) var<uniform> frame: Frame;

// Props: meshes a game creates at run time, in model space, placed per instance. A draw of n
// instances reads props[base + instance_index], `base` coming with the draw.
struct Prop {
  model: mat4x4f,
  normal: mat4x4f,  // inverse transpose of the model matrix (upper 3x3 used)
  tint: vec4f,      // multiplies the material's colour and alpha
  skin: vec4f,      // skinned: x = its first joint matrix in `joints`, y = its mesh's first record
                    // in `skins`, z = 1 (0: not skinned)
};
@group(0) @binding(11) var<storage, read> props: array<Prop>;
// A skinned prop's joints this frame (joint world-from-bind matrices, in the model's space), and
// each skinned mesh's vertices' four joints and their weights (two records a vertex).
@group(0) @binding(12) var<storage, read> joints: array<mat4x4f>;
@group(0) @binding(13) var<storage, read> skins: array<vec4f>;

struct Skinned {
  position: vec3f,
  normal: vec3f,
};

// A prop's vertex moved by its joints (linear blend skinning), in the model's space; as it is for
// a prop that is not skinned.
fn skin_vertex(p: Prop, vertex: u32, position: vec3f, normal: vec3f) -> Skinned {
  var o: Skinned;
  o.position = position;
  o.normal = normal;
  if (p.skin.z < 0.5) {
    return o;
  }
  let at = u32(p.skin.y) + vertex * 2u;
  let j = skins[at];
  let w = skins[at + 1u];
  if (w.x + w.y + w.z + w.w < 1e-6) {
    return o;
  }
  let first = u32(p.skin.x);
  let a = joints[first + u32(j.x)];
  let b = joints[first + u32(j.y)];
  let c = joints[first + u32(j.z)];
  let d = joints[first + u32(j.w)];
  let v = vec4f(position, 1.0);
  let n = vec4f(normal, 0.0);
  o.position = ((a * v) * w.x + (b * v) * w.y + (c * v) * w.z + (d * v) * w.w).xyz;
  o.normal = ((a * n) * w.x + (b * n) * w.y + (c * n) * w.z + (d * n) * w.w).xyz;
  return o;
}

const PI = 3.14159265;

// The water's aux.w: its normal (aux.xyz) follows the clock, so the AO of a water pixel is
// computed every frame (`fs_ssao`).
const WATER_AUX = 0.75;

fn hash_u(n: u32) -> u32 {
  var x = n;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  x = x ^ (x >> 16u);
  return x;
}

fn rand01(n: u32) -> f32 {
  return f32(hash_u(n) >> 8u) / 16777216.0;
}

fn hash2(p: vec2f) -> f32 {
  let q = bitcast<vec2u>(vec2i(floor(p)));
  return rand01(q.x * 1973u + hash_u(q.y * 9277u + 26699u));
}

fn hash3(p: vec3f) -> f32 {
  let q = bitcast<vec3u>(vec3i(floor(p)));
  return rand01(hash_u(q.x * 73856093u) ^ hash_u(q.y * 19349663u + 1u) ^ hash_u(q.z * 83492791u + 7u));
}

fn noise2(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (vec2f(3.0) - 2.0 * f);
  let a = hash2(i);
  let b = hash2(i + vec2f(1.0, 0.0));
  let c = hash2(i + vec2f(0.0, 1.0));
  let d = hash2(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm2(p: vec2f) -> f32 {
  var sum = 0.0;
  var amp = 0.5;
  var q = p;
  for (var i = 0; i < 5; i++) {
    sum += amp * noise2(q);
    q = vec2f(q.x * 1.6 - q.y * 1.2, q.x * 1.2 + q.y * 1.6) + vec2f(17.3, 9.1);
    amp *= 0.5;
  }
  return sum;
}

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// The ground map seen from above: where `p` falls in it (xy = texel, z = depth of `p`).
// Returns z < 0 when `p` is outside the map.
fn ground_texel(p: vec3f) -> vec3f {
  let q = frame.ground_vp * vec4f(p, 1.0);
  let uv = q.xy * 0.5 + vec2f(0.5);
  if (any(uv <= vec2f(0.0)) || any(uv >= vec2f(1.0))) {
    return vec3f(0.0, 0.0, -1.0);
  }
  return vec3f(uv * frame.ground.z, clamp(q.z, 0.0, 1.0));
}

// World height of a ground-map depth value.
fn ground_height(depth: f32) -> f32 {
  return frame.ground.x - depth * frame.ground.y;
}

// The world map seen from above, as ground_texel: xy = texel, z = depth of `p` (< 0 outside the
// map or before it is drawn).
fn world_texel(p: vec3f) -> vec3f {
  if (frame.world.w < 0.5) {
    return vec3f(0.0, 0.0, -1.0);
  }
  let q = frame.world_vp * vec4f(p, 1.0);
  let uv = q.xy * 0.5 + vec2f(0.5);
  if (any(uv <= vec2f(0.0)) || any(uv >= vec2f(1.0))) {
    return vec3f(0.0, 0.0, -1.0);
  }
  return vec3f(uv * frame.world.z, clamp(q.z, 0.0, 1.0));
}

// World height of a world-map depth value.
fn world_height(depth: f32) -> f32 {
  return frame.world.x - depth * frame.world.y;
}

// How much of a terrain surface is beach sand (x) and bare rock (y), 0..1. Sand lies within
// t.x metres above the water at `water`, rock where the slope (1 - n.y) passes t.y..t.z; both
// edges are broken by noise so they never run as ruled lines. The ground map uses the same
// test to keep grass off sand and rock.
fn terrain_cover(p: vec3f, ny: f32, water: f32, t: vec4f) -> vec2f {
  let wobble = noise2(p.xz * 0.23) * 0.65 + noise2(p.xz * 0.97 + vec2f(3.1, 7.7)) * 0.35 - 0.5;
  var sand = 0.0;
  if (t.x > 0.0) {
    sand = 1.0 - smoothstep(t.x * 0.55, t.x, p.y - water + wobble * t.x * 0.8);
  }
  let rock = smoothstep(t.y, t.z, 1.0 - abs(ny) + wobble * 0.16);
  return vec2f(sand, rock);
}

// The sky without sun, moon, stars or clouds: gradient, horizon glow towards the light, halo.
fn sky_base(dir: vec3f) -> vec3f {
  let h = dir.y;
  var col = mix(frame.horizon.rgb, frame.zenith.rgb, pow(clamp(h, 0.0, 1.0), 0.5));
  let below = frame.horizon.rgb * 0.62 + frame.zenith.rgb * 0.18;
  col = mix(col, below, smoothstep(0.0, -0.3, h));
  let g = frame.glow_dir.xyz;
  let flat_dir = normalize(vec2f(dir.x, dir.z) + vec2f(1e-5, 0.0));
  let flat_glow = normalize(vec2f(g.x, g.z) + vec2f(1e-5, 0.0));
  let az = max(dot(flat_dir, flat_glow), 0.0);
  let band = exp(-abs(h - max(g.y, -0.05)) * 5.0);
  col += frame.glow.rgb * (pow(az, 3.0) * band * 0.85);
  let mu = max(dot(dir, g), 0.0);
  col += frame.glow.rgb * (pow(mu, 10.0) * 0.3 + pow(mu, 90.0) * 0.5);
  return col;
}

fn clouds(dir: vec3f, base: vec3f) -> vec3f {
  let cover = frame.horizon.w;
  if (dir.y <= 0.0) {
    return base;
  }
  let t = frame.camera.w * frame.wind.z;
  let scale = frame.sun_dir.w;
  let uv = dir.xz / (dir.y + 0.12) * (0.55 * scale) + frame.wind.xy * (t * 0.0015);
  let density = fbm2(uv * 1.4);
  let threshold = mix(0.78, 0.18, cover);
  let c = smoothstep(threshold, threshold + 0.22, density) * smoothstep(0.0, 0.14, dir.y);
  if (c <= 0.0) {
    return base;
  }
  // Two-tone painted shading: brighter on the side facing the light.
  let g = frame.glow_dir.xyz;
  let towards = normalize(vec2f(g.x, g.z) + vec2f(1e-5, 0.0));
  let ahead = fbm2(uv * 1.4 + towards * 0.18);
  let lit = smoothstep(-0.05, 0.08, density - ahead) * 0.65 + 0.35 * smoothstep(-0.2, 0.6, g.y);
  var col = mix(frame.cloud_shadow.rgb, frame.cloud_lit.rgb, lit);
  let mu = max(dot(dir, g), 0.0);
  let edge = c * (1.0 - c) * 4.0;
  col += frame.glow.rgb * (pow(mu, 6.0) * edge * 1.2);
  return mix(base, col, c * 0.96);
}

fn stars(dir: vec3f) -> f32 {
  if (dir.y <= 0.0 || frame.zenith.w <= 0.0) {
    return 0.0;
  }
  let p = dir * 260.0;
  let cell = floor(p);
  let r = hash3(cell);
  let density = frame.cloud_lit.w;
  if (r < 1.0 - 0.012 * density) {
    return 0.0;
  }
  let off = vec3f(hash3(cell + vec3f(7.0)), hash3(cell + vec3f(13.0)), hash3(cell + vec3f(29.0)));
  let d = length(fract(p) - off);
  let twinkle = 0.75 + 0.25 * sin(frame.camera.w * (2.0 + r * 3.0) + r * 40.0);
  let size = 0.06 + 0.1 * hash3(cell + vec3f(41.0));
  return smoothstep(size, 0.0, d) * twinkle * frame.zenith.w * smoothstep(0.0, 0.2, dir.y);
}

// The full sky seen along `dir`.
fn sky_color(dir: vec3f) -> vec3f {
  var col = sky_base(dir);
  col += vec3f(0.85, 0.9, 1.0) * stars(dir) * 2.5;
  let m = frame.moon_dir.xyz;
  let moon_r = 0.028 * frame.moon_dir.w;
  let md = 1.0 - dot(dir, m);
  let moon_ang = sqrt(max(2.0 * md, 0.0));
  if (frame.glow_dir.w > 0.0 && moon_r > 0.0) {
    let disc = 1.0 - smoothstep(moon_r * 0.92, moon_r, moon_ang);
    if (disc > 0.0) {
      let surface = 0.82 + 0.18 * noise2((dir.xy - m.xy) * 220.0);
      col = mix(col, vec3f(1.0, 0.97, 0.9) * (2.6 * surface), disc * frame.glow_dir.w);
    }
    col += vec3f(0.55, 0.65, 0.95) * (exp(-moon_ang * 18.0) * 0.35 * frame.glow_dir.w);
  }
  if (frame.glow.w > 0.0) {
    let s = frame.sun_dir.xyz;
    let sd = sqrt(max(2.0 * (1.0 - dot(dir, s)), 0.0));
    col += vec3f(1.0, 0.92, 0.75) * (smoothstep(0.022, 0.017, sd) * 30.0 * frame.glow.w);
  }
  col = clouds(dir, col);
  return col;
}

// Aerial perspective and height fog between the camera and a point `dist` away along `dir`.
fn fog_amount(cam: vec3f, world: vec3f, dist: f32) -> f32 {
  let base = frame.fog2.x;
  let falloff = frame.fog2.y;
  let h0 = max(cam.y - base, 0.0);
  let h1 = max(world.y - base, 0.0);
  // Average density of exp(-falloff * h) along the segment.
  var avg = exp(-falloff * h0);
  let dh = h1 - h0;
  if (abs(falloff * dh) > 1e-4) {
    avg = (exp(-falloff * h0) - exp(-falloff * h1)) / (falloff * dh);
  }
  let density = frame.fog.w * frame.fog2.w;
  return 1.0 - exp(-density * dist * mix(0.35, 1.0, avg));
}

fn fog_color(dir: vec3f) -> vec3f {
  let g = frame.glow_dir.xyz;
  let mu = max(dot(dir, g), 0.0);
  let scatter = frame.glow.rgb * (pow(mu, 6.0) * 0.55);
  return mix(frame.fog.rgb, sky_base(dir), 0.45) + scatter;
}
