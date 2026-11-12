// Language coverage: abstract constants, aliases, inferred constructors, nested structs and
// arrays in every kind of memory, whole-struct copies between buffers and locals, dynamic
// indexing of values (spilled) and constants (private tables), short-circuit evaluation with
// side effects, nested loops, scoping and phony assignments.

alias Color = vec4f;
alias Index = u32;

const N = 8;
const HALF = N / 2;
const SCALE = 1.5;
const MIXED = N * SCALE;
const MASK = (1u << 4u) - 1u;
const ORIGIN = vec3(0.0, 0.5, 1.0);
const HEX_F = 0x1.8p1;
const WEIGHTS = array(0.25, 0.5, 0.25);
const IDENTITY = mat2x2f(1.0, 0.0, 0.0, 1.0);
const_assert N == 8;
const_assert HALF * 2 == N && MIXED == 12.0;

struct Inner {
  a: vec2f,
  b: f32,
};

struct Outer {
  inner: Inner,
  list: array<Inner, 3>,
  flag: u32,
};

struct Instance {
  transform: mat4x3f,
  tint: Color,
  ids: array<Index, 4>,
};

struct Instances {
  count: u32,
  items: array<Instance>,
};

struct Row {
  @size(16) value: f32,
};

struct Uniforms {
  tint: vec4f,
  blocks: array<vec4f, 2>,
  rows: array<Row, 2>,
  offset: vec3f,
  scale: f32,
  @align(16) inner: Inner,
  after: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> instances: Instances;
@group(0) @binding(2) var<storage, read_write> results: array<vec4f>;
@group(0) @binding(3) var<uniform> tint: vec4f;
@group(0) @binding(4) var<storage, read> matrices: array<mat4x4f>;
@group(0) @binding(5) var<storage, read_write> outer_buf: Outer;
@group(0) @binding(6) var<uniform> weights: array<vec4f, 4>;
var<push_constant> pc: vec4u;

var<private> table: array<f32, N> = array<f32, N>(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
var<private> counter: u32;
var<private> origin = ORIGIN;

fn make_inner(x: f32) -> Inner {
  return Inner(vec2f(x, -x), x * 2.0);
}

fn sum3(v: array<f32, 3>, i: i32) -> f32 {
  return v[i] + v[(i + 1) % 3];
}

fn pick_column(m: mat3x3f, c: u32) -> vec3f {
  return m[c];
}

fn bump() -> bool {
  counter++;
  return counter > 3u;
}

fn classify(total: u32) -> i32 {
  if (total < 2u) {
    return 1;
  } else if (total < 4u) {
    return 2;
  } else if (total < 8u) {
    return 3;
  } else {
    return 4;
  }
}

fn spin(limit: u32) -> u32 {
  var x = 0u;
  loop {
    x++;
    if (x >= limit) {
      return x;
    }
  }
}

@compute @workgroup_size(N, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let as_u: u32 = N;
  let as_f: f32 = N;
  let as_i = N;
  var acc = vec4f(f32(as_u) + as_f + f32(as_i), MIXED, HEX_F, f32(MASK));
  acc += vec4f(ORIGIN, 1.0) + vec4f(origin, 0.0);
  for (var k = 0; k < 3; k++) {
    acc.x += WEIGHTS[k];
  }
  let w = WEIGHTS;
  acc.y += sum3(w, i32(i % 3u)) + sum3(array(1.0, 2.0, 3.0), 1);

  var o = Outer();
  o.inner = make_inner(1.0);
  o.list[1].a.y = 3.0;
  o.list[i % 3u] = Inner(vec2f(1.0), 2.0);
  o.list[2].a = o.list[1].a.yx;
  let copy = o;
  acc.z += copy.list[i % 3u].b + copy.inner.a.x;

  outer_buf = o;
  outer_buf.list[0] = copy.inner;
  outer_buf.list[1].a.x += 1.0;
  let loaded = outer_buf;
  o = loaded;
  o.list = loaded.list;
  acc.w += loaded.list[i % 3u].a.y + f32(o.flag);

  let n = arrayLength(&instances.items);
  if (i < n && i < instances.count) {
    let inst = instances.items[i];
    let p = inst.transform * vec4f(1.0, 2.0, 3.0, 1.0);
    acc += vec4f(p, inst.tint.a) * f32(inst.ids[i % 4u]);
    let m = matrices[i % arrayLength(&matrices)];
    acc = m * acc;
    acc += instances.items[i].transform[3].xyzx;
  }

  let basis = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
  acc += vec4f(pick_column(basis, i % 3u), 0.0);
  let col = basis[i % 3u];
  acc.x += col.y + IDENTITY[i % 2u].x + ORIGIN[i % 3u];

  if (i > 100u && bump()) {
    acc.w = 0.0;
  }
  if (i < 2u || bump()) {
    acc.w += 1.0;
  }
  let both = bump() && counter > 1u;
  let either = i == 3u || (i < arrayLength(&results) && results[i].x > 0.0);
  acc.x += f32(both) + f32(u32(either)) + f32(i32(true));
  let bv = vec3<bool>(acc.xyz);
  acc.y += select(0.0, 1.0, all(bv)) + f32(vec3u(bv).y);

  acc += u.tint * tint + u.blocks[i % 2u] + vec4f(u.rows[1].value, u.offset * u.scale) + weights[i % 4u];
  acc.w += f32(pc.x + pc[i % 4u]) + u.inner.b + u.after;

  table[i % 8u] = acc.x;
  acc.x += table[(i + 1u) % 8u];

  var total = 0u;
  for (var a = 0u; a < 4u; a++) {
    for (var b = 0u; b < 4u; b++) {
      if (b == a) {
        continue;
      }
      if (b > a + 1u) {
        break;
      }
      total += a * b;
    }
    if (total > 20u) {
      break;
    }
  }
  total += spin(3u);

  let shadow = 1.0;
  {
    let shadow = 2.0;
    acc.z += shadow;
  }
  acc.z += shadow;

  _ = make_inner(acc.x);
  _ = instances.count;
  _ = weights;
  results[i] = acc + vec4f(f32(classify(total)), f32(total), 0.0, 0.0);
}
