// Matrices, nested uniform structs, control flow and assignment forms: tangent-space
// normal mapping with a hand-built basis plus a few deliberately varied loops.

struct Camera {
  view: mat4x4f,
  proj: mat4x4f,
  position: vec3f,
  exposure: f32,
};

struct Material {
  albedo: vec4f,
  // x: roughness, y: metalness, z: normal strength, w: unused
  params: vec4f,
};

struct Scene {
  camera: Camera,
  material: Material,
  normal_matrix: mat3x3f,
  lights: array<vec4f, 4>,
  counts: vec4u,
};

@group(0) @binding(0) var<uniform> scene: Scene;

var<private> rng_state: u32 = 0x9e3779b9u;
var<private> calls: i32;

fn next_random() -> f32 {
  rng_state ^= rng_state << 13u;
  rng_state ^= rng_state >> 17u;
  rng_state ^= rng_state << 5u;
  calls++;
  return f32(rng_state & 0xffffu) / 65535.0;
}

fn basis(n: vec3f) -> mat3x3f {
  let up = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(n.z) > 0.999);
  let t = normalize(cross(up, n));
  let b = cross(n, t);
  return mat3x3f(t, b, n);
}

struct VsOut {
  @builtin(position) clip: vec4f,
  @location(0) normal: vec3f,
  @location(1) world: vec3f,
};

@vertex
fn vs_main(@location(0) position: vec3f, @location(1) normal: vec3f) -> VsOut {
  var out: VsOut;
  let view_proj = scene.camera.proj * scene.camera.view;
  out.clip = view_proj * vec4f(position, 1.0);
  out.normal = scene.normal_matrix * normal;
  out.world = position;
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
  let tbn = basis(normalize(in.normal));
  let inv = transpose(tbn);
  let det = determinant(tbn);
  let to_eye = scene.camera.position - in.world;
  let local = inv * to_eye;
  let back = local * inv;
  let m2 = mat2x2f(1.0, 0.0, 0.0, 1.0) * 2.0;
  let m4 = mat4x4f(vec4f(1.0, 0.0, 0.0, 0.0), vec4f(0.0, 1.0, 0.0, 0.0), vec4f(0.0, 0.0, 1.0, 0.0), vec4f(back, 1.0));
  let sum_m = tbn + inv - tbn * inv;
  var acc = vec4f(0.0);
  acc.xyz = local;
  acc.w = det;
  acc.zx += vec2f(1.0, 2.0);
  acc.y *= m2[1].y;
  acc = m4 * acc;
  acc.x += sum_m[2][1];

  var i = 0;
  loop {
    acc.y += next_random();
    i++;
    if (i == 2) {
      continue;
    }
    acc.x *= 0.5;
    continuing {
      acc.z -= 0.125;
      break if i >= 4;
    }
  }

  var j = 10u;
  while (j > 0u) {
    j--;
    if (j % 3u == 0u) {
      continue;
    }
    acc.w += f32(j);
  }

  var light = vec3f(0.0);
  for (var k = 0u; k < scene.counts.x; k++) {
    let l = scene.lights[k];
    light += l.rgb * l.a;
  }

  let mask = acc.xyz > vec3f(0.5);
  let picked = select(vec3f(0.0), acc.xyz, mask);
  let flipped = select(-picked, picked, vec3<bool>(true, false, mask.z));

  var bits = 0xF0u;
  bits |= 1u;
  bits &= 0x7Fu;
  bits ^= 0x3u;
  bits <<= 2u;
  bits >>= 1u;
  bits %= 100u;
  bits /= 3u;
  bits += 7u;
  bits -= 1u;
  bits *= 2u;
  var f = 2.0;
  f *= 3.0;
  f /= 2.0;
  f -= 1.0;
  f %= 1.5;
  var n = -7;
  n %= 4;
  n /= 2;
  n--;

  let albedo = scene.material.albedo.rgb * scene.material.params.z;
  let rough = scene.material.params.x * f32(bits) * f + f32(n) + f32(calls);
  let color = albedo * (light + flipped * 0.01) * rough * scene.camera.exposure;
  return vec4f(color, 1.0);
}
