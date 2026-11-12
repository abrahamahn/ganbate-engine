// Every math built-in in scalar and vector form, with float, signed and unsigned overloads.
// The inputs come from varyings and a storage buffer so nothing is constant-folded.

struct Params {
  a: vec4f,
  b: vec4f,
  i: vec4i,
  u: vec4u,
  m: mat3x3f,
  packed: u32,
  bits: u32,
  s: f32,
  k: i32,
};

@group(0) @binding(0) var<storage, read_write> params: Params;

struct FsIn {
  @location(0) x: vec4f,
  @location(1) n: vec3f,
  @location(2) @interpolate(flat) i: vec2i,
  @location(3) @interpolate(flat) u: u32,
};

fn float_scalar(x: f32, y: f32) -> f32 {
  var r = abs(x) + acos(saturate(x)) + acosh(1.0 + abs(y)) + asin(clamp(y, -1.0, 1.0));
  r += asinh(x) + atan(y) + atanh(clamp(x, -0.5, 0.5)) + atan2(y, x);
  r += ceil(x) + cos(x) + cosh(y) + degrees(x) + exp(y) + exp2(x);
  r += floor(y) + fma(x, y, 1.0) + fract(x) + inverseSqrt(abs(x) + 1.0);
  r += log(abs(y) + 1.0) + log2(abs(x) + 2.0) + max(x, y) + min(x, y);
  r += mix(x, y, 0.25) + pow(abs(x), y) + radians(y) + round(x) + sign(y);
  r += sin(x) + sinh(y) + smoothstep(0.0, 1.0, x) + sqrt(abs(y)) + step(0.5, x);
  r += tan(x) + tanh(y) + trunc(y) + clamp(x, 0.0, 1.0) + length(x) + distance(x, y);
  r += ldexp(x, 2) + select(x, y, x < y);
  return r;
}

fn float_vector(x: vec4f, y: vec4f, n: vec3f) -> vec4f {
  var r = abs(x) + ceil(y) + clamp(x, vec4f(0.0), vec4f(1.0)) + cos(x) + exp2(y);
  r += floor(x) + fma(x, y, vec4f(0.5)) + fract(y) + max(x, y) + min(x, y);
  r += mix(x, y, vec4f(0.5)) + mix(x, y, 0.25) + pow(abs(x), y) + round(y) + saturate(x);
  r += sign(y) + smoothstep(vec4f(0.0), vec4f(1.0), x) + sqrt(abs(y)) + step(x, y) + trunc(x);
  r += inverseSqrt(abs(x) + 1.0) + atan2(y, x) + degrees(x) + radians(y) + ldexp(x, vec4i(1, 2, 3, 4));
  let nn = normalize(n);
  let refl = reflect(-nn, vec3f(0.0, 1.0, 0.0));
  let refr = refract(nn, vec3f(0.0, 0.0, 1.0), 1.0 / 1.33);
  let ff = faceForward(nn, refl, refr);
  let c = cross(nn, ff);
  r += vec4f(c * dot(nn, refl) + refr * length(refl) + ff * distance(nn, c), dot(x, y));
  r += select(x, y, x > y) + select(x, y, true);
  return r;
}

fn integer_ops(i: vec2i, u: u32, iv: vec4i, uv: vec4u) -> vec4f {
  let ia = abs(i) + sign(i) + max(i, vec2i(1)) + min(i, vec2i(-1)) + clamp(i, vec2i(-4), vec2i(4));
  let ua = abs(u) + max(u, 2u) + min(u, 9u) + clamp(u, 1u, 7u);
  let uvec = max(uv, vec4u(3u)) + min(uv, uv.wzyx) + clamp(uv, vec4u(0u), vec4u(100u));
  let id = dot(iv, iv) + dot(i, i);
  let ud = dot(uv, uv);
  let bits = countOneBits(u) + countOneBits(uv).x + reverseBits(u) + u32(countOneBits(ia).y);
  let rev = reverseBits(iv);
  let sh = (u << 3u) | (u >> (u & 7u)) | (uv.x >> uv.y);
  let ish = (iv << vec4u(1u, 2u, 3u, 4u)) >> uv;
  let neg = -ia + ~ia;
  let q = vec2i(iv.xy) / max(abs(i), vec2i(1)) + i % vec2i(3);
  let uq = uv / vec4u(7u) + uv % 5u;
  return vec4f(vec2f(ia + neg + q), f32(ua + ud + bits + sh), f32(id)) + vec4f(uvec + uq) + vec4f(rev + ish);
}

fn packing(x: vec4f) -> vec4f {
  let p0 = pack4x8unorm(x);
  let p1 = pack4x8snorm(x * 2.0 - 1.0);
  let p2 = pack2x16unorm(x.xy);
  let p3 = pack2x16snorm(x.zw);
  let p4 = pack2x16float(x.xy * 100.0);
  params.packed = p0 ^ p1 ^ p2 ^ p3 ^ p4;
  return unpack4x8unorm(p0) + unpack4x8snorm(p1) + vec4f(unpack2x16unorm(p2), unpack2x16snorm(p3))
    + vec4f(unpack2x16float(p4), 0.0, 0.0);
}

fn logic(x: vec4f, y: vec4f) -> f32 {
  let lt = x < y;
  let ge = x >= y;
  let eq = x == y;
  let ne = x != y;
  let b = (lt | ge) & !eq;
  var score = 0.0;
  if (all(b) || any(ne)) {
    score += 1.0;
  }
  if (all(lt.xy) && !any(eq.zw)) {
    score += 2.0;
  }
  let flags = vec4<bool>(true, lt.y, ge.z, false);
  return score + f32(all(flags != b)) + f32(any(flags));
}

@fragment
fn fs_builtins(in: FsIn) -> @location(0) vec4f {
  let x = in.x + params.a;
  let y = params.b;
  let d = dpdx(x) + dpdy(y) + fwidth(x);
  let m = params.m;
  let det = determinant(m);
  let t = transpose(mat2x3f(m[0], m[1]));
  var r = float_vector(x, y, in.n + m[2] * det) + d;
  r.x += float_scalar(x.x, y.y) + t[2].y;
  r += integer_ops(in.i + params.i.xy, in.u + params.u.x, params.i, params.u);
  r += packing(saturate(x));
  r.w += logic(x, y);
  return r;
}

@compute @workgroup_size(32)
fn cs_builtins(@builtin(local_invocation_id) lid: vec3u, @builtin(global_invocation_id) gid: vec3u) {
  let x = params.a + vec4f(f32(lid.x));
  let y = params.b;
  var r = float_vector(x, y, x.xyz);
  r.y += float_scalar(params.s, f32(params.k));
  r += integer_ops(vec2i(gid.xy), lid.x, params.i, params.u);
  r += packing(saturate(r));
  r.z += logic(x, y);
  workgroupBarrier();
  params.a = r;
  storageBarrier();
  params.bits = bitcast<u32>(r.x) ^ bitcast<vec2u>(vec2i(1, 2)).y ^ u32(bitcast<i32>(r.y));
  let fb = bitcast<vec4f>(params.u);
  params.b = fb + bitcast<vec4f>(params.i);
}
