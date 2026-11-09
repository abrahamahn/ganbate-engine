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
  wet_sand: vec4f,      // rgb sand at the water line
  rock: vec4f,          // rgb bare rock
  meadow: vec4f,        // rgb second grass tone of the broad patches
  lamps: vec4f,         // lamp light on foliage (scale), baked AO strength, unused, unused
  tide: vec4f,          // tide-mark height above the water (m, 0 = none), rgb stain multiplier
  tiles: vec4u,         // tiles across, tiles down, tile size in pixels, light count
  light_shadow0: mat4x4f, // world to the shadow maps of a game's shadowed lights (layers 0, 1):
  light_shadow1: mat4x4f, // perspective from each light, depth 0 at it to 1 at its range
  light_shadow: vec4f,  // 1 / texels per side of those maps, unused, unused, unused
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
    let surface = 0.82 + 0.18 * noise2((dir.xy - m.xy) * 220.0);
    col = mix(col, vec3f(1.0, 0.97, 0.9) * (2.6 * surface), disc * frame.glow_dir.w);
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

// --- module: brdf ---
// Unified surface response library for Ganbate Engine: art-directed stylized cel/character
// shading and physically-based microfacet BRDFs (GGX, ClearCoat, Charlie cloth, Subsurface skin)
// sharing identical scene lighting, shadows, tiled lamps, and material inputs.
#define_import_path brdf
#import common

// Compile-time specialization switches (`ganbate_shader::CompileOptions`).
const ENABLE_STYLIZED: bool = true;
const ENABLE_PHOTOREAL: bool = true;
const FIXED_SURFACE_MODEL: i32 = -1;

struct SurfaceInput {
  albedo: vec3f,
  n: vec3f,
  geo_n: vec3f,
  v: vec3f,
  l: vec3f,
  key: vec3f,
  hemi: vec3f,
  lamps: vec3f,
  sh: f32,
  rough: f32,
  metallic: f32,
  ao: f32,
  brush: f32,
  foliage: bool,
  model: i32,
  lobe_weight: f32, // < 0 uses Look / model default
  lobe_param: f32,  // < 0 uses Look / model default
};

fn d_ggx(nh: f32, a2: f32) -> f32 {
  let d = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-6);
}

fn v_smith_ggx(nv: f32, nl: f32, a2: f32) -> f32 {
  let gv = nl * sqrt(nv * nv * (1.0 - a2) + a2);
  let gl = nv * sqrt(nl * nl * (1.0 - a2) + a2);
  return 0.5 / max(gv + gl, 1e-5);
}

fn f_schlick(vh: f32, f0: vec3f) -> vec3f {
  let m = clamp(1.0 - vh, 0.0, 1.0);
  let m2 = m * m;
  return f0 + (vec3f(1.0) - f0) * (m2 * m2 * m);
}

fn f_schlick_roughness(nv: f32, f0: vec3f, rough: f32) -> vec3f {
  let m = clamp(1.0 - nv, 0.0, 1.0);
  let m2 = m * m;
  let hi = max(vec3f(1.0 - rough), f0);
  return f0 + (hi - f0) * (m2 * m2 * m);
}

fn fd_burley(nv: f32, nl: f32, lh: f32, rough: f32) -> f32 {
  let f90 = 0.5 + 2.0 * rough * lh * lh;
  let fv = 1.0 + (f90 - 1.0) * pow(clamp(1.0 - nv, 0.0, 1.0), 5.0);
  let fl = 1.0 + (f90 - 1.0) * pow(clamp(1.0 - nl, 0.0, 1.0), 5.0);
  return (fv * fl) / PI;
}

fn d_charlie(nh: f32, rough: f32) -> f32 {
  let inv_r = 1.0 / max(rough, 0.05);
  let sin2h = max(1.0 - nh * nh, 1e-5);
  return (2.0 + inv_r) * pow(sin2h, 0.5 * inv_r) / (2.0 * PI);
}

fn sky_ibl(r_dir: vec3f, rough: f32, hemi: vec3f) -> vec3f {
  let crisp = sky_base(r_dir);
  return mix(crisp, hemi, rough * rough * 0.82);
}

// Model 0: Ganbate soft-cel ramp, shadow tint, toon specular, and upright-gated fresnel rim.
fn eval_stylized_ramp(s: SurfaceInput) -> vec3f {
  let ndl = dot(s.n, s.l);
  let half_l = ndl * 0.5 + 0.5 + s.brush * 0.45;
  var lit = smoothstep(frame.ramp.x - frame.ramp.y, frame.ramp.x + frame.ramp.y, half_l) * s.sh;
  var glow = vec3f(0.0);
  if (s.foliage) {
    let wrap = frame.style.z;
    let soft = smoothstep(0.0, 1.0, (ndl + wrap) / (1.0 + wrap));
    lit = mix(lit, soft * mix(0.3, 1.0, s.sh), 0.65);
    let back = pow(clamp(dot(-s.l, s.v), 0.0, 1.0), 3.0);
    glow = s.albedo * s.key * (back * frame.style.w * mix(0.25, 1.0, s.sh));
  }
  let dark = mix(vec3f(1.0), frame.shadow_tint.rgb, frame.shadow_tint.w);
  var ambient = s.hemi;
  if (s.foliage) {
    ambient = s.hemi * (1.0 + frame.style.w);
  }
  let diffuse = s.key * lit + ambient * mix(dark, vec3f(1.0), lit);
  var col = s.albedo * (diffuse + s.lamps) * (1.0 - 0.5 * s.metallic) + glow;

  let hv = normalize(s.l + s.v);
  let nh = max(dot(s.n, hv), 0.0);
  let size = frame.style.y * s.rough;
  let spec_t = 1.0 - size * 0.12;
  let gloss = 1.0 - s.rough;
  let spec = smoothstep(spec_t - 0.004, spec_t + 0.004, nh) * frame.style.x * (gloss * gloss * gloss) * lit;
  col += mix(vec3f(1.0), s.albedo, s.metallic) * s.key * spec;

  let fres = pow(1.0 - clamp(dot(s.n, s.v), 0.0, 1.0), frame.ramp.w);
  let rim_col = mix(frame.amb_sky.rgb * 1.6, s.key, clamp(ndl * 0.5 + 0.5, 0.0, 1.0));
  let upright = 1.0 - smoothstep(0.3, 0.75, s.geo_n.y);
  col += rim_col * (fres * frame.ramp.z * (0.35 + 0.65 * s.sh) * upright);
  return col;
}

// Model 1: Art-directed character cel shading with warm terminator scatter and anisotropic ribbon highlight.
fn eval_stylized_character(s: SurfaceInput) -> vec3f {
  let ndl = dot(s.n, s.l);
  let half_l = ndl * 0.5 + 0.5 + s.brush * 0.2;
  let sharpness = select(0.32, clamp(1.0 - s.lobe_weight, 0.08, 1.0), s.lobe_weight >= 0.0);
  let band = max(frame.ramp.y * sharpness, 0.005);
  let lit = smoothstep(frame.ramp.x - band, frame.ramp.x + band, half_l) * s.sh;
  let term_in = smoothstep(frame.ramp.x - 0.14, frame.ramp.x, half_l);
  let term_out = 1.0 - smoothstep(frame.ramp.x, frame.ramp.x + 0.12, half_l);
  let sss_band = term_in * term_out * s.sh * max(frame.surface_mode.w, 0.35);

  let dark = mix(vec3f(1.0), frame.shadow_tint.rgb, frame.shadow_tint.w);
  let diffuse = s.key * (lit + frame.subsurface_color.rgb * sss_band * 0.55)
    + s.hemi * mix(dark, vec3f(1.0), lit) * s.ao;
  var col = s.albedo * (diffuse + s.lamps) * (1.0 - 0.45 * s.metallic);

  // Tangent-aligned ribbon highlight for hair, eyes, and lacquered trim.
  let hv = normalize(s.l + s.v);
  var up_axis = vec3f(0.0, 1.0, 0.0);
  if (abs(s.n.y) > 0.92) {
    up_axis = vec3f(1.0, 0.0, 0.0);
  }
  let tangent = normalize(cross(up_axis, s.n));
  let aniso = select(0.55, clamp(s.lobe_param, 0.0, 1.0), s.lobe_param >= 0.0);
  let th = dot(tangent, hv);
  let sin_th = sqrt(max(1.0 - th * th, 0.0));
  let nh_eff = mix(max(dot(s.n, hv), 0.0), sin_th, aniso * 0.65);
  let gloss = 1.0 - s.rough;
  let spec_edge = 1.0 - (0.02 + 0.14 * s.rough);
  let spec = smoothstep(spec_edge - 0.01, spec_edge + 0.01, nh_eff) * max(frame.style.x, 0.45) * gloss * gloss * lit;
  col += mix(vec3f(1.0), s.albedo, s.metallic) * s.key * spec;

  let nv = clamp(dot(s.n, s.v), 0.0, 1.0);
  let rim = smoothstep(0.55, 0.88, 1.0 - nv) * clamp(ndl * 0.65 + 0.35, 0.0, 1.0);
  col += mix(frame.amb_sky.rgb * 1.4, s.key, 0.65) * (rim * max(frame.ramp.z, 0.25) * (0.4 + 0.6 * s.sh));
  return col;
}

// Model 2: Energy-conserving microfacet GGX + height-correlated Smith G2 + Burley diffuse + sky IBL.
fn eval_standard_ggx(s: SurfaceInput) -> vec3f {
  let r = clamp(s.rough, 0.045, 1.0);
  let a = r * r;
  let a2 = a * a;
  let nv = max(dot(s.n, s.v), 1e-4);
  let nl_raw = dot(s.n, s.l);
  let nl = max(nl_raw, 0.0);
  let hv = normalize(s.l + s.v);
  let nh = max(dot(s.n, hv), 0.0);
  let vh = max(dot(s.v, hv), 0.0);
  let lh = max(dot(s.l, hv), 0.0);

  let f0 = mix(vec3f(0.04), s.albedo, s.metallic);
  let f = f_schlick(vh, f0);
  let d = d_ggx(nh, a2);
  let vis = v_smith_ggx(nv, max(nl, 1e-4), a2);
  // Multi-scattering energy compensation for rough metals and dielectrics.
  let energy_comp = vec3f(1.0) + f0 * ((1.0 / max(1.0 - r * 0.42, 0.25)) - 1.0);
  let spec_brdf = d * vis * f * energy_comp;

  let kd = (vec3f(1.0) - f) * (1.0 - s.metallic);
  var diff_term = kd * s.albedo * fd_burley(nv, max(nl, 1e-4), lh, r);
  var key_vis = nl * s.sh;
  if (s.foliage) {
    let wrap = frame.style.z;
    key_vis = clamp((nl_raw + wrap) / (1.0 + wrap), 0.0, 1.0) * mix(0.25, 1.0, s.sh);
    let back = pow(clamp(dot(-s.l, s.v), 0.0, 1.0), 3.0) * frame.style.w * mix(0.25, 1.0, s.sh);
    diff_term += s.albedo * back * (1.0 / PI);
  }

  let direct = (diff_term + spec_brdf * step(0.0, nl_raw)) * s.key * key_vis;

  // Split-sum environment approximation with horizon & AO specular occlusion.
  let r_dir = reflect(-s.v, s.n);
  let horizon_occl = clamp(1.0 + dot(r_dir, s.geo_n), 0.0, 1.0);
  let spec_ao = clamp(pow(nv + s.ao, a) - 1.0 + s.ao, 0.0, 1.0) * horizon_occl * horizon_occl;
  let f_ibl = f_schlick_roughness(nv, f0, r);
  let kd_ibl = (vec3f(1.0) - f_ibl) * (1.0 - s.metallic);
  let env_diff = kd_ibl * s.albedo * s.hemi * s.ao;
  let env_spec = sky_ibl(r_dir, r, s.hemi) * f_ibl * (1.0 - r * 0.55) * spec_ao;
  let lamp_contrib = s.albedo * (1.0 - 0.75 * s.metallic) * s.lamps + f0 * s.lamps * (1.0 - r * 0.6);

  return direct + env_diff + env_spec + lamp_contrib;
}

// Model 3: Base GGX substrate under a smooth dielectric clear-coat layer (automotive, lacquer, ceramics).
fn eval_clearcoat(s: SurfaceInput) -> vec3f {
  let base = eval_standard_ggx(s);
  let coat_w = select(max(frame.surface_mode.y, 0.85), clamp(s.lobe_weight, 0.0, 1.0), s.lobe_weight >= 0.0);
  let coat_r = clamp(select(frame.subsurface_color.w, s.lobe_param, s.lobe_param >= 0.0), 0.03, 1.0);
  let ac = coat_r * coat_r;
  let ac2 = ac * ac;

  let cn = normalize(mix(s.geo_n, s.n, 0.25));
  let nv = max(dot(cn, s.v), 1e-4);
  let nl = max(dot(cn, s.l), 0.0);
  let hv = normalize(s.l + s.v);
  let nh = max(dot(cn, hv), 0.0);
  let vh = max(dot(s.v, hv), 0.0);

  let fc = f_schlick(vh, vec3f(0.04)).x * coat_w;
  let dc = d_ggx(nh, ac2);
  let vc = v_smith_ggx(nv, max(nl, 1e-4), ac2);
  let coat_direct = s.key * (dc * vc * fc * nl * s.sh);

  let r_dir = reflect(-s.v, cn);
  let fc_ibl = f_schlick_roughness(nv, vec3f(0.04), coat_r).x * coat_w;
  let coat_env = sky_ibl(r_dir, coat_r, s.hemi) * fc_ibl * s.ao;

  return base * (1.0 - fc_ibl * 0.85) + coat_direct + coat_env + s.lamps * (fc_ibl * 0.35);
}

// Model 4: Estevez & Kulla Charlie sheen BRDF for woven cloth, velvet, silk, and felt-like fabrics.
fn eval_charlie_cloth(s: SurfaceInput) -> vec3f {
  let r = clamp(s.rough, 0.08, 1.0);
  let nv = max(dot(s.n, s.v), 1e-4);
  let nl_raw = dot(s.n, s.l);
  let nl = max(nl_raw, 0.0);
  let hv = normalize(s.l + s.v);
  let nh = max(dot(s.n, hv), 0.0);

  let sheen_w = select(max(frame.surface_mode.z, 0.75), clamp(s.lobe_weight, 0.0, 1.0), s.lobe_weight >= 0.0);
  let sheen_tint = select(0.5, clamp(s.lobe_param, 0.0, 1.0), s.lobe_param >= 0.0);
  let lum = max(luminance(s.albedo), 0.04);
  let tint_col = s.albedo / lum;
  let sheen_col = mix(vec3f(1.0), tint_col, sheen_tint) * sheen_w;

  let dc = d_charlie(nh, r);
  let vc = 1.0 / max(4.0 * (nl + nv - nl * nv), 0.12);
  let cloth_spec = sheen_col * (dc * vc);

  // Soft fabric wrap diffuse + grazing fiber scatter.
  let wrap_nl = clamp((nl_raw + 0.22) / 1.22, 0.0, 1.0);
  let diff = s.albedo * (1.0 / PI) * wrap_nl * mix(0.2, 1.0, s.sh);
  let direct = s.key * (diff + cloth_spec * nl * s.sh);

  let graze = pow(clamp(1.0 - nv, 0.0, 1.0), 3.0);
  let env = (s.albedo * s.hemi + sheen_col * s.hemi * graze * 0.65) * s.ao;
  return direct + env + (s.albedo + sheen_col * 0.25) * s.lamps * (1.0 / PI * 2.2);
}

// Model 5: Dual-lobe specular + RGB wrapped subsurface profile + back-lit translucency (skin, wax, jade).
fn eval_subsurface_skin(s: SurfaceInput) -> vec3f {
  let r = clamp(s.rough, 0.08, 1.0);
  let a1 = r * r;
  let a2_primary = a1 * a1;
  let r_lobe2 = clamp(r * 0.48, 0.05, 1.0);
  let a2_secondary = r_lobe2 * r_lobe2 * r_lobe2 * r_lobe2;

  let nv = max(dot(s.n, s.v), 1e-4);
  let nl_raw = dot(s.n, s.l);
  let nl = max(nl_raw, 0.0);
  let hv = normalize(s.l + s.v);
  let nh = max(dot(s.n, hv), 0.0);
  let vh = max(dot(s.v, hv), 0.0);

  let sss_w = select(max(frame.surface_mode.w, 0.65), clamp(s.lobe_weight, 0.0, 1.0), s.lobe_weight >= 0.0);
  let sss_col = clamp(frame.subsurface_color.rgb, vec3f(0.05), vec3f(1.0));

  // RGB-dependent wrap diffuse: red scatters furthest into the terminator, blue stays crisp.
  let wraps = vec3f(0.52, 0.24, 0.10) * sss_w;
  let nl_rgb = clamp((vec3f(nl_raw) + wraps) / (vec3f(1.0) + wraps), vec3f(0.0), vec3f(1.0));
  let shadow_bleed = mix(vec3f(s.sh), pow(vec3f(max(s.sh, 0.02)), vec3f(1.0) - sss_col * (0.55 * sss_w)), sss_w);
  let diff_light = nl_rgb * shadow_bleed * mix(vec3f(1.0), vec3f(1.0) + sss_col * 0.28, sss_w * (1.0 - nl));

  // Skin IOR ~ 1.38 => F0 = 0.028, dual specular lobe.
  let f = f_schlick(vh, vec3f(0.028));
  let d = mix(d_ggx(nh, a2_primary), d_ggx(nh, a2_secondary), 0.18);
  let vis = v_smith_ggx(nv, max(nl, 1e-4), a2_primary);
  let spec = d * vis * f * nl * s.sh;

  // Back-lit thin translucency.
  let back = pow(clamp(dot(-s.l, s.v), 0.0, 1.0), 2.5) * sss_w * mix(0.25, 1.0, s.sh);
  let trans = s.albedo * sss_col * s.key * (back * 0.45);

  let r_dir = reflect(-s.v, s.n);
  let f_ibl = f_schlick_roughness(nv, vec3f(0.028), r);
  let env_diff = s.albedo * s.hemi * mix(vec3f(s.ao), pow(vec3f(max(s.ao, 0.05)), vec3f(1.0) - sss_col * 0.45), sss_w);
  let env_spec = sky_ibl(r_dir, r, s.hemi) * f_ibl * (1.0 - r * 0.5) * s.ao;

  return (s.albedo * (vec3f(1.0) - f) * (1.0 / PI) * diff_light * s.key)
    + spec * s.key
    + trans
    + env_diff
    + env_spec
    + s.albedo * s.lamps * (1.0 + sss_w * 0.15);
}

fn evaluate_surface(s: SurfaceInput) -> vec3f {
  if (FIXED_SURFACE_MODEL == 0) {
    return eval_stylized_ramp(s);
  } else if (FIXED_SURFACE_MODEL == 1) {
    return eval_stylized_character(s);
  } else if (FIXED_SURFACE_MODEL == 2) {
    return eval_standard_ggx(s);
  } else if (FIXED_SURFACE_MODEL == 3) {
    return eval_clearcoat(s);
  } else if (FIXED_SURFACE_MODEL == 4) {
    return eval_charlie_cloth(s);
  } else if (FIXED_SURFACE_MODEL == 5) {
    return eval_subsurface_skin(s);
  }

  if (ENABLE_PHOTOREAL) {
    if (s.model == 2) {
      return eval_standard_ggx(s);
    }
    if (s.model == 3) {
      return eval_clearcoat(s);
    }
    if (s.model == 4) {
      return eval_charlie_cloth(s);
    }
    if (s.model == 5) {
      return eval_subsurface_skin(s);
    }
  }
  if (ENABLE_STYLIZED) {
    if (s.model == 1) {
      return eval_stylized_character(s);
    }
    return eval_stylized_ramp(s);
  }
  return eval_standard_ggx(s);
}
