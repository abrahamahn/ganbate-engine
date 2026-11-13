// Shadow depth, the top-down ground map, the sky backdrop and GPU-generated weather particles
// (rain, petals).
#import common

struct ShadowPush {
  light_vp: mat4x4f,
};
var<push_constant> shadow_push: ShadowPush;

@vertex fn vs_shadow(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return shadow_push.light_vp * vec4f(position, 1.0);
}

// Props cast shadows from their instances' transforms: index.x = first prop instance.
struct PropShadowPush {
  light_vp: mat4x4f,
  params: vec4f,   // as MaskedShadowPush.params (masked props only)
  index: vec4f,
};
var<push_constant> prop_shadow: PropShadowPush;

@vertex fn vs_shadow_prop(@location(0) position: vec3f, @builtin(instance_index) ii: u32, @builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = props[u32(prop_shadow.index.x) + ii];
  let s = skin_vertex(p, vi, position, vec3f(0.0, 1.0, 0.0));
  return prop_shadow.light_vp * (p.model * vec4f(s.position, 1.0));
}

@vertex fn vs_shadow_prop_masked(@location(0) position: vec3f, @location(2) uv: vec2f, @builtin(instance_index) ii: u32, @builtin(vertex_index) vi: u32) -> MaskedOut {
  let p = props[u32(prop_shadow.index.x) + ii];
  let s = skin_vertex(p, vi, position, vec3f(0.0, 1.0, 0.0));
  var o: MaskedOut;
  o.clip = prop_shadow.light_vp * (p.model * vec4f(s.position, 1.0));
  o.uv = uv;
  return o;
}

@fragment fn fs_shadow_prop_masked(in: MaskedOut) {
  var a = prop_shadow.params.y;
  if (prop_shadow.params.z > 0.5) {
    a *= textureSampleLevel(mask_tex, mask_sampler, in.uv, 0.0).a;
  }
  if (a < prop_shadow.params.x) {
    discard;
  }
}

// Alpha-masked casters: params = (alpha cutoff, base alpha, has texture, unused).
struct MaskedShadowPush {
  light_vp: mat4x4f,
  params: vec4f,
};
var<push_constant> masked: MaskedShadowPush;
@group(1) @binding(0) var mask_tex: texture_2d<f32>;
@group(1) @binding(3) var mask_sampler: sampler;

struct MaskedOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
};

@vertex fn vs_shadow_masked(@location(0) position: vec3f, @location(2) uv: vec2f) -> MaskedOut {
  var o: MaskedOut;
  o.clip = masked.light_vp * vec4f(position, 1.0);
  o.uv = uv;
  return o;
}

@fragment fn fs_shadow_masked(in: MaskedOut) {
  var a = masked.params.y;
  if (masked.params.z > 0.5) {
    a *= textureSampleLevel(mask_tex, mask_sampler, in.uv, 0.0).a;
  }
  if (a < masked.params.x) {
    discard;
  }
}

// The ground map (and the world map, drawn the same way): depth of the highest solid surface
// seen from above, and per texel what that surface is: r = grows grass, g = solid. On terrain,
// beach sand and bare rock grow none.
struct GroundPush {
  vp: mat4x4f,
  mask: vec4f,     // grows grass, solid, 1 for terrain, water height
  terrain: vec4f,  // as frame.terrain
};
var<push_constant> ground_push: GroundPush;

struct GroundOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
};

@vertex fn vs_ground(@location(0) position: vec3f, @location(1) normal: vec3f) -> GroundOut {
  var o: GroundOut;
  o.clip = ground_push.vp * vec4f(position, 1.0);
  o.world = position;
  o.normal = normal;
  return o;
}

@fragment fn fs_ground(in: GroundOut) -> @location(0) vec4f {
  let m = ground_push.mask;
  var grass = m.x;
  if (m.z > 0.5 && grass > 0.5) {
    let c = terrain_cover(in.world, normalize(in.normal).y, m.w, ground_push.terrain);
    if (max(c.x, c.y) > 0.5) {
      grass = 0.0;
    }
  }
  return vec4f(grass, m.y, 0.0, 0.0);
}

struct FsOut {
  @location(0) color: vec4f,
  @location(1) aux: vec4f,
};

struct SkyOut {
  @builtin(position) clip: vec4f,
  @location(0) ndc: vec2f,
};

@vertex fn vs_sky(@builtin(vertex_index) i: u32) -> SkyOut {
  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  let ndc = uv * 2.0 - vec2f(1.0);
  var o: SkyOut;
  o.clip = vec4f(ndc, 0.0, 1.0);
  o.ndc = ndc;
  return o;
}

@fragment fn fs_sky(in: SkyOut) -> FsOut {
  let p = frame.inv_view_proj * vec4f(in.ndc, 1.0, 1.0);
  let dir = normalize(p.xyz / p.w - frame.camera.xyz);
  var out: FsOut;
  out.color = vec4f(sky_color(dir), 1.0);
  out.aux = vec4f(0.0);
  return out;
}

// Particles: params = (kind: 0 rain / 1 petals, ground height, box size, amount), tint = rgb, a = alpha,
// centre = the box the particles wrap around (ahead of the camera, so few are wasted behind it).
struct ParticlePush {
  params: vec4f,
  tint: vec4f,
  centre: vec4f,
};
var<push_constant> particle: ParticlePush;
@group(0) @binding(5) var ground_sampler: sampler_comparison;
@group(0) @binding(8) var ground_depth: texture_depth_2d;

// 1 where nothing solid is above `p` (open sky), 0 under a roof (from the ground map).
fn open_sky(p: vec3f) -> f32 {
  let g = ground_texel(p);
  if (g.z <= 0.0) {
    return 1.0;
  }
  return textureSampleCompareLevel(ground_depth, ground_sampler, g.xy / frame.ground.z, g.z - 0.002);
}

struct ParticleOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec4f,
};

const PARTICLE_CORNERS = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

@vertex fn vs_particle(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> ParticleOut {
  let kind = particle.params.x;
  let box = particle.params.z;
  let t = frame.camera.w;
  let seed = ii * 4u;
  let r = vec4f(rand01(seed), rand01(seed + 1u), rand01(seed + 2u), rand01(seed + 3u));
  let cam = frame.camera.xyz;
  let wind = vec3f(frame.wind.x, 0.0, frame.wind.y);
  var pos = r.xyz * box;
  var vel = vec3f(0.0);
  if (kind < 0.5) {
    vel = vec3f(0.0, -(8.0 + 4.0 * r.w), 0.0) + wind;
    pos += vel * t;
  } else {
    let swirl = vec3f(sin(t * 0.8 + r.x * 6.28), 0.0, cos(t * 0.6 + r.y * 6.28)) * 0.7;
    vel = vec3f(0.0, -0.55 - 0.3 * r.w, 0.0) + wind * 0.45;
    pos += vel * t + swirl;
  }
  // Wrap into the box.
  let centre = particle.centre.xyz;
  let rel = (pos - centre) / box + vec3f(0.5);
  let world_c = centre + (rel - floor(rel) - vec3f(0.5)) * box;
  // Thin out near the camera so drops never fill the lens; petals drift only near the ground.
  let d = length(world_c - cam);
  let near_fade = smoothstep(0.6, 2.2, d) * particle.params.w;
  let height_fade = select(1.0, smoothstep(11.0, 5.0, world_c.y - particle.params.y), kind >= 0.5);
  let cw = (frame.view_proj * vec4f(world_c, 1.0)).w;
  if (particle.tint.a <= 0.0 || near_fade <= 0.0 || height_fade <= 0.0 || cw < -1.0) {
    var o: ParticleOut;
    o.clip = vec4f(0.0, 0.0, -1.0, 1.0);
    return o;
  }
  let c = PARTICLE_CORNERS[vi];
  let to_cam = normalize(cam - world_c);
  var world = world_c;
  var color = particle.tint;
  if (kind < 0.5) {
    let along = normalize(vel);
    let side = normalize(cross(along, to_cam) + vec3f(1e-5, 0.0, 0.0));
    world += side * (c.x * 0.006) + along * (c.y * 0.28);
    // Drops stop at roofs.
    let open = open_sky(world_c);
    color = vec4f(particle.tint.rgb * (frame.amb_sky.rgb * 1.5 + frame.key_color.rgb * frame.key_dir.w * 0.3 + vec3f(0.08)), particle.tint.a * open);
  } else {
    let spin = t * (1.5 + r.w * 2.0) + r.x * 6.28;
    let right = normalize(cross(vec3f(0.0, 1.0, 0.0), to_cam) + vec3f(1e-5, 0.0, 0.0));
    let up = cross(to_cam, right);
    let a = vec2f(cos(spin), sin(spin));
    let q = vec2f(c.x * a.x - c.y * a.y, c.x * a.y + c.y * a.x) * vec2f(0.022, 0.015);
    world += right * q.x + up * q.y;
    let lit = frame.amb_sky.rgb * 0.8 + frame.key_color.rgb * (frame.key_dir.w * 0.28) + vec3f(1.0, 0.6, 0.3) * (frame.amb_sky.w * 0.18);
    color = vec4f(particle.tint.rgb * lit * (0.75 + 0.5 * r.y), particle.tint.a);
  }
  color.a *= near_fade;
  if (kind >= 0.5) {
    color.a *= height_fade;
  }
  var o: ParticleOut;
  o.clip = frame.view_proj * vec4f(world, 1.0);
  if (color.a <= 0.0) {
    // Invisible (covered, or at the lens): clip the quad away before it rasterizes.
    o.clip = vec4f(0.0, 0.0, -1.0, 1.0);
  }
  o.uv = c;
  o.color = color;
  return o;
}

@fragment fn fs_particle(in: ParticleOut) -> FsOut {
  var a = in.color.a;
  if (particle.params.x < 0.5) {
    a *= (1.0 - abs(in.uv.x)) * smoothstep(1.0, 0.2, abs(in.uv.y));
  } else {
    let d = length(in.uv);
    a *= smoothstep(1.0, 0.75, d);
  }
  var out: FsOut;
  out.color = vec4f(in.color.rgb, a);
  out.aux = vec4f(0.0);
  return out;
}
