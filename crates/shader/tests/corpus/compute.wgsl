// GPU particles: integrate a read-write particle buffer, then splat the particles into an
// rgba16float storage texture, one 8x8 tile per workgroup.

struct Particle {
  position: vec3f,
  life: f32,
  velocity: vec3f,
  seed: u32,
};

struct ParticleBuffer {
  // x: live count, y: frame, zw: unused
  header: vec4u,
  items: array<Particle>,
};

struct Sim {
  gravity: vec4f,
  emitter: vec4f,
  dt: f32,
  drag: f32,
  radius: f32,
  intensity: f32,
};

@group(0) @binding(0) var<storage, read_write> particles: ParticleBuffer;
@group(0) @binding(1) var<uniform> sim: Sim;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var background: texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> histogram: array<u32>;

const MAX_SPLATS = 256u;

fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn unit(h: u32) -> f32 {
  return f32(h & 0xffffffu) / 16777216.0;
}

fn respawn(seed: u32) -> Particle {
  let a = pcg(seed);
  let b = pcg(a);
  let c = pcg(b);
  let dir = normalize(vec3f(unit(a), unit(b), unit(c)) * 2.0 - 1.0);
  return Particle(sim.emitter.xyz, 1.0 + unit(c) * 2.0, dir * sim.emitter.w, c);
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&particles.items)) {
    return;
  }
  var p = particles.items[i];
  p.velocity += sim.gravity.xyz * sim.dt;
  p.velocity *= 1.0 / (1.0 + sim.drag * sim.dt);
  p.position += p.velocity * sim.dt;
  p.life -= sim.dt;
  if (p.life <= 0.0) {
    p = respawn(p.seed ^ particles.header.y);
  }
  particles.items[i] = p;
  if (i == 0u) {
    particles.header.y += 1u;
  }
}

@compute @workgroup_size(8, 8)
fn splat(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lane: u32,
         @builtin(workgroup_id) group: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let size = textureDimensions(output);
  if (any(gid.xy >= size)) {
    return;
  }
  let pixel = vec2f(gid.xy) + 0.5;
  let uv = pixel / vec2f(size);
  var accum = textureLoad(background, vec2i(gid.xy), 0);
  let count = min(arrayLength(&particles.items), MAX_SPLATS);
  for (var k = 0u; k < count; k++) {
    let p = particles.items[k];
    if (p.life <= 0.0) {
      continue;
    }
    let center = (p.position.xy * 0.5 + 0.5) * vec2f(size);
    let d = distance(center, pixel);
    let w = exp(-d * d / (sim.radius * sim.radius)) * sim.intensity * saturate(p.life);
    accum += vec4f(w * abs(p.velocity), w);
  }
  textureStore(output, vec2i(gid.xy), accum);
  let bucket = min(u32(accum.a * 16.0), arrayLength(&histogram) - 1u);
  if (lane == 0u && group.x + group.y * groups.x < arrayLength(&histogram)) {
    histogram[bucket] = histogram[bucket] + 1u;
  }
  _ = uv;
  storageBarrier();
  workgroupBarrier();
}
