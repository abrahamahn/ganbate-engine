// Instanced meadow grass blades and wildflowers placed from the top-down ground map.
#define_import_path scene_grass
#import common

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
  // The grid is side * side cells centred under the camera; the renderer draws the part of it
  // the view can show (`Renderer::visible_grass`), `columns` cells across from (`x0`, `z0`).
  let side = max(u32(frame.water2.y), 1u);
  let half = i32(side / 2u);
  let columns = max(u32(frame.grass_grid.z), 1u);
  let col = i32(frame.grass_grid.x) + i32(ii % columns);
  let row = i32(frame.grass_grid.y) + i32(ii / columns);
  let cx = i32(floor(cam.x / GRASS_SPACING)) + col - half;
  let cz = i32(floor(cam.z / GRASS_SPACING)) + row - half;
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

const FLOWER_CORNERS = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

// Flower heads: small upturned discs on some of the grass sites, above the blades.
@vertex fn vs_flower(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FlowerOut {
  var o: FlowerOut;
  o.clip = vec4f(0.0, 0.0, -1.0, 1.0);
  let site = grass_site(ii);
  if (site.keep <= 0.0 || rand01(site.seed + 11u) > frame.grass.z) {
    return o;
  }
  let seed = site.seed;
  let c = FLOWER_CORNERS[vi];
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
