//! End-to-end unit tests: compile snippets, validate the SPIR-V with `spirv-val` (when it is
//! installed) and check typing, layout, diagnostics and reflection.

use crate::ir::{Program, Stmt};
use crate::types::{ConstValue, Scalar, Ty};
use crate::*;

fn tool_available(name: &str) -> bool {
    std::process::Command::new(name)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn temp_spv(words: &[u32]) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!("ganbate-shader-unit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("m{}.spv", N.fetch_add(1, Ordering::Relaxed)));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&path, bytes).expect("write spv");
    path
}

pub(crate) fn validate(words: &[u32]) -> Result<(), String> {
    if !tool_available("spirv-val") {
        return Ok(());
    }
    let path = temp_spv(words);
    let out = std::process::Command::new("spirv-val")
        .args(["--target-env", "vulkan1.0"])
        .arg(&path)
        .output()
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&path);
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}

/// Disassembly, or `None` when `spirv-dis` is not installed.
fn disassemble(words: &[u32]) -> Option<String> {
    if !tool_available("spirv-dis") {
        return None;
    }
    let path = temp_spv(words);
    let out = std::process::Command::new("spirv-dis")
        .arg(&path)
        .output()
        .ok()?;
    let _ = std::fs::remove_file(&path);
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[track_caller]
pub(crate) fn ok(src: &str, stage: Stage, entry: &str) -> Compiled {
    let c = compile(src, stage, entry).unwrap_or_else(|e| panic!("compile error: {e}\n{src}"));
    if let Err(msg) = validate(&c.words) {
        panic!("spirv-val failed for '{entry}':\n{msg}\n{src}");
    }
    c
}

/// Compile and validate every entry point of `src`.
#[track_caller]
fn ok_all(src: &str) -> Vec<Compiled> {
    let eps = entry_points(src).unwrap_or_else(|e| panic!("error: {e}\n{src}"));
    assert!(!eps.is_empty(), "no entry points");
    eps.iter().map(|(n, s)| ok(src, *s, n)).collect()
}

/// The first error reported for `src`, checking every entry point.
#[track_caller]
fn err(src: &str) -> Error {
    match entry_points(src) {
        Err(e) => e,
        Ok(eps) => {
            for (n, s) in eps {
                if let Err(e) = compile(src, s, &n) {
                    return e;
                }
            }
            panic!("expected a compile error for:\n{src}");
        }
    }
}

#[track_caller]
fn expect_err(src: &str, line: u32, column: u32, needle: &str) {
    let e = err(src);
    assert!(
        e.message.contains(needle),
        "expected '{needle}' in error, got {}:{}: {}\n{src}",
        e.line,
        e.column,
        e.message
    );
    assert_eq!(
        (e.line, e.column),
        (line, column),
        "wrong position for '{}'\n{src}",
        e.message
    );
}

fn program(src: &str) -> Program {
    let toks = crate::lexer::tokenize(src).unwrap();
    let m = crate::parser::parse(toks).unwrap();
    crate::check::check(&m).unwrap_or_else(|e| panic!("{e}\n{src}"))
}

/// Types and constant values of the top-level `let`s of function `f`.
fn lets(src: &str) -> Vec<(Ty, Option<ConstValue>)> {
    let p = program(src);
    let f = p.functions.iter().find(|f| f.name == "f").expect("fn f");
    f.body
        .iter()
        .filter_map(|s| match s {
            Stmt::Let(_, e) => Some((e.ty.clone(), e.as_const().cloned())),
            _ => None,
        })
        .collect()
}

fn frag(body: &str) -> String {
    format!("@fragment fn main(@location(0) p: vec4f) -> @location(0) vec4f {{\n{body}\n}}")
}

// ----------------------------------------------------------------------- literal typing

#[test]
fn abstract_literals_adapt_to_context() {
    let src = "const N = 8;
        fn f(x: f32, u: u32) {
            let a: u32 = N;
            let b: f32 = N;
            let c = N;
            let d = 1 + 2.0;
            let e = x * 2;
            let g = u + 1;
            let h = vec3f(1, 2, 3);
            let i = i32(1.9);
            let j = select(1, 2.0, true);
            let k = 0xFFFFFFFFu;
            let l = -2147483648;
            let m = N / 3;
            let n = 7.0 / 2;
            let o = vec2(1, 2) * 3;
            let q = array(1u, 2, 3)[1];
            let r = u32(-1i);
        }";
    let l = lets(src);
    use ConstValue as V;
    let f32v = |v: f32| Some(V::F32(v));
    assert_eq!(l[0], (Ty::Scalar(Scalar::U32), Some(V::U32(8))));
    assert_eq!(l[1], (Ty::Scalar(Scalar::F32), f32v(8.0)));
    assert_eq!(l[2], (Ty::Scalar(Scalar::I32), Some(V::I32(8))));
    assert_eq!(l[3], (Ty::Scalar(Scalar::F32), f32v(3.0)));
    assert_eq!(l[4].0, Ty::Scalar(Scalar::F32));
    assert_eq!(l[5].0, Ty::Scalar(Scalar::U32));
    assert_eq!(
        l[6],
        (
            Ty::Vector(3, Scalar::F32),
            Some(V::Composite(vec![V::F32(1.0), V::F32(2.0), V::F32(3.0)]))
        )
    );
    assert_eq!(l[7], (Ty::Scalar(Scalar::I32), Some(V::I32(1))));
    assert_eq!(l[8], (Ty::Scalar(Scalar::F32), f32v(2.0)));
    assert_eq!(l[9], (Ty::Scalar(Scalar::U32), Some(V::U32(u32::MAX))));
    assert_eq!(l[10], (Ty::Scalar(Scalar::I32), Some(V::I32(i32::MIN))));
    assert_eq!(l[11], (Ty::Scalar(Scalar::I32), Some(V::I32(2))));
    assert_eq!(l[12], (Ty::Scalar(Scalar::F32), f32v(3.5)));
    assert_eq!(
        l[13],
        (
            Ty::Vector(2, Scalar::I32),
            Some(V::Composite(vec![V::I32(3), V::I32(6)]))
        )
    );
    assert_eq!(l[14], (Ty::Scalar(Scalar::U32), Some(V::U32(2))));
    assert_eq!(l[15], (Ty::Scalar(Scalar::U32), Some(V::U32(u32::MAX))));
}

#[test]
fn abstract_constants_in_types_attributes_and_loops() {
    let src = "const N = 4;
        var<private> table: array<f32, N * 2>;
        @compute @workgroup_size(N, N / 2)
        fn main(@builtin(local_invocation_index) i: u32) {
            for (var k = 0u; k < N; k++) { table[k] = f32(N); }
            var f = 0.0;
            for (var k = 0; k < N; k++) { f += table[k]; }
            table[i % 8u] = f;
        }";
    let c = ok(src, Stage::Compute, "main");
    assert_eq!(c.workgroup_size, [4, 2, 1]);
}

#[test]
fn literal_typing_errors() {
    expect_err("fn f() { let x: u32 = -1; }", 1, 23, "u32");
    expect_err("fn f() { let x: i32 = 0xFFFFFFFF; }", 1, 23, "i32");
    expect_err("fn f() { let big = 1 << 40; }", 1, 22, "i32");
    expect_err("fn f() { let x: i32 = 1.5; }", 1, 23, "expected i32");
    expect_err(
        "const X: i32 = 2147483647 + 1;",
        1,
        27,
        "cannot be represented as i32",
    );
    expect_err("const X: i32 = 2147483647i + 1;", 1, 28, "overflow");
    expect_err("const X = 1 / 0;", 1, 13, "division by zero");
    expect_err("const X = 1u << 32u;", 1, 14, "shift");
    expect_err(
        "fn f(u: u32) { let x = u + 1.0; }",
        1,
        26,
        "cannot be applied to u32 and abstract-float",
    );
    expect_err("fn f() { let x = 4294967296u; }", 1, 18, "u32");
    expect_err(
        "fn f() { let v = max(vec3f(1.0), 1.0); }",
        1,
        18,
        "no overload of 'max'",
    );
}

// ----------------------------------------------------------------------- layout

fn struct_layout(src: &str, name: &str) -> (Vec<u32>, u32, u32) {
    let p = program(src);
    let s = p.structs.iter().find(|s| s.name == name).expect("struct");
    let l = s.layout.as_ref().expect("layout");
    (l.offsets.clone(), l.align, l.size)
}

#[test]
fn host_shareable_layout_offsets() {
    let src = "
        struct Frame { view_proj: mat4x4f, inv_view_proj: mat4x4f, camera: vec4f, sun_dir: vec4f,
                       sun_color: vec4f, params: vec4f, viewport: vec4f, tiles: vec4u }
        struct Packed { a: vec3f, b: f32, c: vec2f, d: vec3f, e: f32 }
        struct WithMat { m: mat3x3f, f: f32 }
        struct WithArray { x: f32, arr: array<vec3f, 2>, y: u32 }
        struct Attrs { a: f32, @align(16) b: f32, @size(32) c: vec2f, d: f32 }
        struct Mats { a: mat2x2f, b: mat3x2f, c: mat4x3f, d: mat2x4f }
        struct Nested { s: Packed, v: vec2f, w: Attrs }
        struct Runtime { n: u32, items: array<vec4f> }
    ";
    assert_eq!(
        struct_layout(src, "Frame"),
        (vec![0, 64, 128, 144, 160, 176, 192, 208], 16, 224)
    );
    assert_eq!(
        struct_layout(src, "Packed"),
        (vec![0, 12, 16, 32, 44], 16, 48)
    );
    assert_eq!(struct_layout(src, "WithMat"), (vec![0, 48], 16, 64));
    assert_eq!(struct_layout(src, "WithArray"), (vec![0, 16, 48], 16, 64));
    assert_eq!(struct_layout(src, "Attrs"), (vec![0, 16, 24, 56], 16, 64));
    assert_eq!(struct_layout(src, "Mats"), (vec![0, 16, 48, 112], 16, 144));
    assert_eq!(struct_layout(src, "Nested"), (vec![0, 48, 64], 16, 128));
    assert_eq!(struct_layout(src, "Runtime"), (vec![0, 16], 16, 16));
}

#[test]
fn layout_decorations_and_sizes_validate() {
    let src = "
        struct Inner { a: vec3f, b: f32 }
        struct U { m: mat3x3f, inner: Inner, arr: array<vec4f, 3>, last: f32 }
        struct S { mats: array<mat2x2f, 2>, v: vec3f, rows: array<vec2f> }
        @group(0) @binding(0) var<uniform> u: U;
        @group(0) @binding(1) var<storage, read_write> s: S;
        @compute @workgroup_size(1) fn main() {
            s.rows[0] = u.inner.a.xy + s.mats[1][0] + u.m[2].xy * u.last + u.arr[2].zw;
            s.v = u.inner.a;
            s.mats[0] = mat2x2f(s.rows[1], vec2f(f32(arrayLength(&s.rows))));
        }";
    let c = ok(src, Stage::Compute, "main");
    assert_eq!(c.bindings[0].kind, BindingKind::UniformBuffer { size: 128 });
    if let Some(dis) = disassemble(&c.words) {
        assert!(
            dis.contains("OpMemberDecorate %U_block 0 MatrixStride 16"),
            "{dis}"
        );
        assert!(
            dis.contains("OpMemberDecorate %S_buffer 0 MatrixStride 8"),
            "{dis}"
        );
        assert!(
            dis.contains("OpMemberDecorate %S_buffer 2 Offset 48"),
            "{dis}"
        );
        assert!(dis.contains("ArrayStride 16"));
    }
}

#[test]
fn uniform_layout_rules_are_enforced() {
    let u = |decl: &str| {
        format!(
            "{decl}\n@group(0) @binding(0) var<uniform> u: U;\n@compute @workgroup_size(1) fn main() {{ _ = u; }}"
        )
    };
    let e = err(&u("alias U = array<f32, 4>;"));
    assert!(e.message.contains("stride 4"), "{e}");
    let e = err(&u("struct U { v: array<vec2f, 2> }"));
    assert!(e.message.contains("16-byte element stride"), "{e}");
    let e = err(&u("struct In { v: f32 } struct U { a: f32, inner: In }"));
    assert!(e.message.contains("16-byte aligned") && e.line == 1, "{e}");
    let e = err(&u(
        "struct In { v: f32 } struct U { @align(16) i: In, b: f32 }",
    ));
    assert!(e.message.contains("at least 16 bytes after"), "{e}");
    let e = err(&u("struct U { m: mat2x2f }"));
    assert!(e.message.contains("mat2x2"), "{e}");
    let e = err(&u("struct U { b: bool }"));
    assert!(e.message.contains("host-shareable"), "{e}");
    // The same types are fine in storage buffers (std430 rules).
    let s = |decl: &str| {
        format!(
            "{decl}\n@group(0) @binding(0) var<storage> u: U;\n@compute @workgroup_size(1) fn main() {{ _ = u; }}"
        )
    };
    ok_all(&s("alias U = array<f32, 4>;"));
    ok_all(&s("struct U { a: f32, v: array<vec2f, 2>, m: mat2x2f }"));
    ok_all(&s("struct In { v: f32 } struct U { a: f32, inner: In }"));
    // ...and fixed-up versions validate as uniform buffers.
    ok_all(&u(
        "struct In { v: f32 } struct U { a: f32, @align(16) inner: In, @align(16) b: f32 }",
    ));
    ok_all(&u(
        "struct Row { @size(16) v: f32 } struct U { rows: array<Row, 4>, m: mat3x3f, n: mat4x4<f32> }",
    ));
}

// ----------------------------------------------------------------------- assignments

#[test]
fn swizzle_and_component_stores() {
    let src = frag(
        "var v = p;
         v.zx = vec2f(1.0, 2.0);
         v.xy += 1.0;
         v.w = 3.0;
         v[1] *= 2.0;
         var m = mat2x2f();
         m[1].y = 4.0;
         m[0] = v.xy;
         var a = array<vec4f, 2>();
         a[1].wzyx = v;
         a[0].g -= v.r;
         return v + vec4f(m[1], m[0]) + a[0] + a[1];",
    );
    let c = ok(&src, Stage::Fragment, "main");
    if let Some(dis) = disassemble(&c.words) {
        // v.zx = n: result = (n.y, v.y, n.x, v.w).
        assert!(
            dis.lines()
                .any(|l| l.contains("OpVectorShuffle %v4float") && l.ends_with(" 5 1 4 3")),
            "{dis}"
        );
        // a[1].wzyx = v: result component c takes v[3 - c].
        assert!(
            dis.lines()
                .any(|l| l.contains("OpVectorShuffle %v4float") && l.ends_with(" 7 6 5 4")),
            "{dis}"
        );
    }
    expect_err(
        &frag("var v = p; v.xx = vec2f(1.0); return v;"),
        2,
        12,
        "repeat",
    );
    expect_err(&frag("let v = p; v.x = 1.0; return v;"), 2, 12, "immutable");
    expect_err(
        "@group(0) @binding(0) var<uniform> u: vec4f;\n@fragment fn main() -> @location(0) vec4f { u.x = 1.0; return u; }",
        2,
        45,
        "uniform buffer",
    );
    expect_err(
        "@group(0) @binding(0) var<storage> s: array<f32>;\n@compute @workgroup_size(1) fn main() { s[0] = 1.0; }",
        2,
        41,
        "read-only storage buffer",
    );
}

#[test]
fn increments_and_compound_assignments() {
    let src = "@compute @workgroup_size(1) fn main() {
        var i = 0; var u = 10u; var f = 1.0; var v = vec3f(1.0); var m = mat3x3f();
        i++; i--; u++; u--;
        i += 2; i -= 1; i *= 3; i /= 2; i %= 5; i &= 7; i |= 8; i ^= 3; i <<= 2u; i >>= 1u;
        u += 2u; u >>= 1u;
        f += 1; f *= 2.5; f /= 3.0; f %= 2.0; f -= 0.5;
        v *= 2.0; v += vec3f(1.0, 2.0, 3.0); v /= 4.0; v -= 1.0; v %= 2.0;
        v *= m; m *= 2.0; m += mat3x3f(); m *= mat3x3f(); m -= m;
        _ = i + i32(u) + i32(f + v.x + m[0].y);
    }";
    ok(src, Stage::Compute, "main");
    expect_err("fn f() { var x = 1.0; x++; }", 1, 23, "i32 or u32");
    expect_err(
        "fn f() { var v = vec2f(); v *= mat3x3f(); }",
        1,
        27,
        "cannot be applied",
    );
}

// ----------------------------------------------------------------------- control flow

#[test]
fn loops_with_break_continue_and_continuing() {
    let src = "var<private> calls: u32;
        fn tick() -> bool { calls++; return calls < 8u; }
        @fragment fn main(@location(0) p: vec4f) -> @location(0) vec4f {
            var acc = 0.0;
            for (var i = 0; i < 10; i++) {
                if (i == 2) { continue; }
                if (i == 7) { break; }
                for (var j = 0u; j < 3u; j++) {
                    if (j == 1u) { continue; }
                    acc += f32(j);
                }
            }
            var k = 0;
            loop {
                k++;
                if (k > 100) { break; }
                if ((k & 1) == 0) { continue; }
                acc += 1.0;
                continuing {
                    acc *= 0.5;
                    break if k >= 20 || !tick();
                }
            }
            while (acc < 100.0 && tick()) { acc += 3.0; }
            for (;;) { acc -= 1.0; if (acc < 0.0) { break; } }
            loop { break; }
            loop { if (p.x > 0.0) { break; } else { continue; } }
            loop {
                if (p.y > 0.0) { break; }
                continuing {
                    var n = 0;
                    loop { n++; if (n > 3) { break; } }
                    break if n > 2;
                }
            }
            for (var q = 0; q < 4; q++) {
                if (q > 1) { continue; } else { acc += 0.25; }
            }
            return vec4f(acc);
        }";
    ok(src, Stage::Fragment, "main");
}

#[test]
fn functions_returning_from_every_shape() {
    let src = "
        fn abs_diff(a: i32, b: i32) -> i32 { if (a > b) { return a - b; } else { return b - a; } }
        fn first_even(n: u32) -> u32 {
            for (var i = 0u; ; i++) {
                if (i >= n) { return n; }
                if (i % 2u == 0u && i > 0u) { return i; }
            }
        }
        fn forever(x: f32) -> f32 { loop { if (x > 0.0) { return x; } } }
        fn early(v: vec2f) -> vec2f {
            if (v.x < 0.0) { return vec2f(0.0); }
            return v;
            // Unreachable statements after a return are type-checked but not emitted.
        }
        fn nothing() {}
        fn nothing_returns() { return; }
        fn table() -> array<f32, 3> { return array(1.0, 2.0, 3.0); }
        fn ident() -> mat2x2f { return mat2x2f(1.0, 0.0, 0.0, 1.0); }
        struct Pair { a: f32, b: vec2u }
        fn pair(x: f32) -> Pair { return Pair(x, vec2u(1u, 2u)); }
        @fragment fn main(@location(0) p: vec4f) -> @location(0) vec4f {
            nothing();
            nothing_returns();
            let t = table();
            let pr = pair(p.x);
            let d = f32(abs_diff(i32(p.x), 3) + i32(first_even(5u)));
            return vec4f(d + forever(p.y) + t[1] + pr.a + f32(pr.b.y), early(p.zw) * ident()[1], 1.0);
        }";
    ok(src, Stage::Fragment, "main");
    expect_err(
        "fn f(x: f32) -> f32 { if (x > 0.0) { return 1.0; } }",
        1,
        52,
        "must return a value",
    );
    expect_err(
        "fn f(x: f32) -> f32 { while (true) { return 1.0; } }",
        1,
        52,
        "must return a value",
    );
    expect_err("fn f() -> f32 { return; }", 1, 17, "missing return value");
    expect_err("fn f() { return 1; }", 1, 17, "no return type");
}

#[test]
fn discard_in_helper_and_dead_code() {
    let src = "
        fn clip(a: f32) { if (a < 0.5) { discard; } }
        @fragment fn main(@location(0) p: vec4f) -> @location(0) vec4f {
            clip(p.w);
            if (p.x > 2.0) { discard; } else { return p; }
            return vec4f(0.0);
        }";
    ok(src, Stage::Fragment, "main");
}

#[test]
fn short_circuit_only_when_needed() {
    let src = "
        var<private> n: u32;
        fn bump() -> bool { n++; return n > 2u; }
        @group(0) @binding(0) var<storage, read> data: array<f32>;
        @fragment fn main(@location(0) p: vec4f) -> @location(0) vec4f {
            let i = u32(p.x);
            let simple = p.x > 0.0 && p.y < 1.0;
            let guarded = i < arrayLength(&data) && data[i] > 0.5;
            let effect = p.z > 0.0 || bump();
            let first = arrayLength(&data) > 0u && data[0] > 0.5;
            return vec4f(f32(simple), f32(guarded), f32(effect), f32(first));
        }";
    let c = ok(src, Stage::Fragment, "main");
    if let Some(dis) = disassemble(&c.words) {
        assert_eq!(dis.matches("OpPhi").count(), 3, "{dis}");
        assert_eq!(dis.matches("OpLogicalAnd").count(), 1, "{dis}");
    }
}

// ----------------------------------------------------------------------- values and indexing

#[test]
fn dynamic_indexing_of_values_and_constants() {
    let src = "
        const K = array<vec2f, 4>(vec2f(1.0), vec2f(2.0), vec2f(3.0), vec2f(4.0));
        const M = mat3x3f(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        fn get(a: array<f32, 4>, i: u32) -> f32 { return a[i]; }
        @fragment fn main(@location(0) p: vec4f, @location(1) @interpolate(flat) i: u32) -> @location(0) vec4f {
            let arr = array<f32, 4>(p.x, p.y, p.z, p.w);
            let m = mat2x2f(p.xy, p.zw);
            let v = p.zyx;
            var s = arr[i % 4u] + m[i % 2u].y + v[i % 3u] + K[i % 4u].y + M[i % 3u][i % 3u];
            s += get(arr, i) + K[2].x + M[1][2] + p[i % 4u];
            return vec4f(s);
        }";
    let c = ok(src, Stage::Fragment, "main");
    if let Some(dis) = disassemble(&c.words) {
        assert!(
            dis.contains("%K = OpVariable") && dis.contains("Private %"),
            "{dis}"
        );
        assert!(dis.contains("OpVectorExtractDynamic"), "{dis}");
    }
    expect_err(
        &frag("let a = array<f32, 2>(1.0, 2.0); return vec4f(a[2]);"),
        2,
        49,
        "out of bounds",
    );
    expect_err(&frag("return vec4f(p[4]);"), 2, 16, "out of bounds");
    expect_err(&frag("return p.xq;"), 2, 10, "invalid swizzle");
}

#[test]
fn constructors_conversions_and_bitcasts() {
    let src = frag(
        "let a = vec4f(p.xy, 1, 2);
         let b = vec4<f32>(vec3f(1.0), 0.5);
         let c = vec3(1, 2u, 3);
         let d = vec2i(p.xy) + vec2i(vec2u(3u));
         let e = vec3f(vec3u(d.xyx)) + vec3f(vec3<bool>(true, false, p.x > 0.0));
         let f = mat2x3f(vec3f(1.0), vec3f(2.0));
         let g = mat3x2f(1, 2, 3, 4, 5, 6);
         let h = array<vec2f, 2>(vec2f(), p.zw);
         let i = bitcast<vec4u>(p) ^ vec4u(bitcast<u32>(1.0));
         let j = f32(true) + f32(1u) + f32(-3) + f32(bool(p.x)) + f32(u32(p.y)) + f32(i32(p.z));
         let k = vec2<bool>(d) & vec2<bool>(true);
         let zero = vec4f() + vec4f(mat2x2f()[0], f32(), f32(u32()));
         return a + b + vec4f(vec3f(c), 1.0) + vec4f(e, j) + vec4f(f[1] * g[2].x, h[1].x) + vec4f(i) * 0.0 + vec4f(vec2f(k), 0.0, 0.0) + zero;",
    );
    ok(&src, Stage::Fragment, "main");
    expect_err(
        &frag("return vec4f(1.0, 2.0, 3.0);"),
        2,
        8,
        "needs 4 components, got 3",
    );
    expect_err(
        &frag("return vec4f(bitcast<f32>(vec2u(1u)), 0.0, 0.0);"),
        2,
        14,
        "cannot bitcast",
    );
    expect_err(
        &frag("let m = mat2x2f(1.0, 2.0, 3.0); return p;"),
        2,
        9,
        "mat2x2<f32> needs 2 column vectors or 4 scalars",
    );
    expect_err(
        &frag("let a = array(1.0, true); return p;"),
        2,
        9,
        "different types",
    );
}

// ----------------------------------------------------------------------- built-ins

#[test]
fn builtins_map_to_the_right_instructions() {
    let src = frag(
        "let i = vec2i(p.xy); let u = vec2u(p.zw);
         var r = round(p) + p % 2.0 + vec4f(f32(i.x / i.y), f32(u.x / u.y), f32(i.x % i.y), f32(u.x % u.y));
         r.x += f32(i.x >> 2u) + f32(u.x >> 2u) + f32(i.y << u.y) + select(0.0, 1.0, p.x != p.y);
         r.y += f32(abs(i.x)) + f32(abs(u.x)) + f32(sign(i.y)) + f32(min(u.x, u.y)) + f32(max(i.x, i.y)) + f32(clamp(u.x, 1u, 2u));
         r.z += inverseSqrt(p.x) + fract(p.y) + trunc(p.z) + saturate(p.w);
         return r;",
    );
    let c = ok(&src, Stage::Fragment, "main");
    let Some(dis) = disassemble(&c.words) else {
        return;
    };
    for needle in [
        "RoundEven",
        "OpFRem",
        "OpSDiv",
        "OpUDiv",
        "OpSRem",
        "OpUMod",
        "OpShiftRightArithmetic",
        "OpShiftRightLogical",
        "OpShiftLeftLogical",
        "OpFUnordNotEqual",
        "SAbs",
        "SSign",
        "UMin",
        "SMax",
        "UClamp",
        "InverseSqrt",
        "Fract",
        "Trunc",
        "FClamp",
    ] {
        assert!(dis.contains(needle), "missing {needle}:\n{dis}");
    }
    // abs(u32) is the identity, and the non-constant shift amount is masked.
    assert!(!dis.contains("UAbs"));
    assert!(
        dis.contains("OpBitwiseAnd %uint") && dis.contains("%uint_31"),
        "{dis}"
    );
}

#[test]
fn builtin_argument_errors() {
    expect_err(
        &frag("return vec4f(sin(1.0, 2.0));"),
        2,
        14,
        "expects 1 argument",
    );
    expect_err(
        &frag("return vec4f(sqrt(1u));"),
        2,
        14,
        "no overload of 'sqrt'",
    );
    expect_err(
        &frag("return vec4f(dot(p, p.xyz));"),
        2,
        14,
        "no overload of 'dot'",
    );
    expect_err(
        &frag("return vec4f(mix(p, p, vec2f(0.5)), 1.0, 1.0);"),
        2,
        14,
        "no overload of 'mix'",
    );
    expect_err(
        &frag("return select(p, p, vec3<bool>());"),
        2,
        8,
        "no overload of 'select'",
    );
    expect_err(&frag("sin(1.0); return p;"), 2, 1, "must be used");
    expect_err(
        &frag("return vec4f(frobnicate(1.0));"),
        2,
        14,
        "unknown function 'frobnicate'",
    );
    expect_err(&frag("return vec4f(modf(1.5));"), 2, 14, "not supported");
}

#[test]
fn stage_restricted_builtins() {
    expect_err(
        "@group(0) @binding(0) var t: texture_2d<f32>;\n@group(0) @binding(1) var s: sampler;
fn helper() -> vec4f { return textureSample(t, s, vec2f(0.5)); }
@vertex fn main() -> @builtin(position) vec4f { return helper(); }",
        3,
        31,
        "fragment",
    );
    expect_err(
        "@vertex fn main() -> @builtin(position) vec4f { return vec4f(dpdx(1.0)); }",
        1,
        62,
        "fragment",
    );
    expect_err(
        "@fragment fn main() { workgroupBarrier(); }",
        1,
        23,
        "compute",
    );
    expect_err(
        "@compute @workgroup_size(1) fn main() { discard; }",
        1,
        41,
        "fragment",
    );
    // textureSampleLevel is allowed in every stage.
    ok_all(
        "@group(0) @binding(0) var t: texture_2d<f32>;\n@group(0) @binding(1) var s: sampler;
@vertex fn main() -> @builtin(position) vec4f { return textureSampleLevel(t, s, vec2f(0.5), 0.0); }",
    );
}

// ----------------------------------------------------------------------- interface & reflection

#[test]
fn reflection_lists_only_used_resources() {
    let src = "
        @group(0) @binding(0) var<uniform> a: vec4f;
        @group(0) @binding(1) var<uniform> b: vec4f;
        @group(1) @binding(0) var t: texture_2d<f32>;
        @group(0) @binding(2) var<storage, read_write> c: array<u32>;
        @group(2) @binding(3) var s: sampler;
        var<push_constant> pc: vec4f;
        fn unused() -> vec4f { return b + textureSampleLevel(t, s, vec2f(0.0), 0.0); }
        fn helper() -> vec4f { return a; }
        @fragment fn main() -> @location(0) vec4f { _ = t; return helper() + pc; }
        @compute @workgroup_size(2, 3, 4) fn cs() { c[0] = 1u; }";
    let c = ok(src, Stage::Fragment, "main");
    let got: Vec<(u32, u32)> = c.bindings.iter().map(|b| (b.group, b.binding)).collect();
    assert_eq!(got, vec![(0, 0), (1, 0)]);
    assert_eq!(c.push_constant_size, 16);
    assert_eq!(c.workgroup_size, [1, 1, 1]);
    let c = ok(src, Stage::Compute, "cs");
    assert_eq!(
        c.bindings,
        vec![Binding {
            group: 0,
            binding: 2,
            kind: BindingKind::StorageBuffer { read_only: false }
        }]
    );
    assert_eq!(c.push_constant_size, 0);
    assert_eq!(c.workgroup_size, [2, 3, 4]);
}

#[test]
fn interface_builtins_and_interpolation() {
    let src = "
        struct V {
            @builtin(position) @invariant pos: vec4f,
            @location(0) @interpolate(flat) id: vec2i,
            @location(1) @interpolate(linear) a: vec2f,
            @location(2) @interpolate(perspective, centroid) b: f32,
            @location(3) @interpolate(perspective, sample) c: vec3f,
            @location(4) d: u32,
        }
        @vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32, @location(5) w: vec4i) -> V {
            return V(vec4f(f32(vi), f32(ii), 0.0, 1.0), w.xy, vec2f(0.5), 1.0, vec3f(2.0), vi);
        }
        struct Out { @location(1) n: vec4u, @builtin(frag_depth) depth: f32, @location(0) color: vec4f }
        @fragment fn fs(v: V, @builtin(front_facing) ff: bool, @builtin(sample_index) si: u32) -> Out {
            return Out(vec4u(v.d, si, u32(ff), 1u), v.pos.z, vec4f(v.a, v.b, v.c.x) + vec4f(vec2f(v.id), 0.0, 0.0));
        }";
    let vs = ok(src, Stage::Vertex, "vs");
    assert_eq!(
        vs.inputs,
        vec![Interface {
            location: 5,
            ty: IoType::Vec4I
        }]
    );
    assert_eq!(
        vs.outputs.iter().map(|i| i.location).collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4]
    );
    let fs = ok(src, Stage::Fragment, "fs");
    assert_eq!(
        fs.inputs[0],
        Interface {
            location: 0,
            ty: IoType::Vec2I
        }
    );
    assert_eq!(
        fs.outputs,
        vec![
            Interface {
                location: 0,
                ty: IoType::Vec4F
            },
            Interface {
                location: 1,
                ty: IoType::Vec4U
            }
        ]
    );
    if let Some(dis) = disassemble(&fs.words) {
        for needle in [
            "DepthReplacing",
            "BuiltIn FragDepth",
            "BuiltIn SampleId",
            "SampleRateShading",
            "NoPerspective",
            "Centroid",
            "BuiltIn FrontFacing",
        ] {
            assert!(dis.contains(needle), "missing {needle}:\n{dis}");
        }
        // Integer varyings are flat even without @interpolate(flat).
        assert!(dis.matches(" Flat").count() >= 3, "{dis}");
    }
    if let Some(dis) = disassemble(&vs.words) {
        assert!(
            dis.contains("BuiltIn Position") && dis.contains("Invariant"),
            "{dis}"
        );
        assert!(dis.contains("BuiltIn VertexIndex") && dis.contains("BuiltIn InstanceIndex"));
    }
}

#[test]
fn interface_errors() {
    expect_err(
        "@vertex fn vs() -> @location(0) vec4f { return vec4f(0.0); }",
        1,
        12,
        "must return @builtin(position)",
    );
    expect_err(
        "@vertex fn vs(@builtin(frag_depth) d: f32) -> @builtin(position) vec4f { return vec4f(d); }",
        1,
        15,
        "not a valid vertex input",
    );
    expect_err(
        "@fragment fn fs(@location(0) a: f32, @location(0) b: f32) {}",
        1,
        38,
        "@location(0) is used twice",
    );
    expect_err(
        "@fragment fn fs(a: f32) {}",
        1,
        17,
        "needs @location or @builtin",
    );
    expect_err(
        "@fragment fn fs(@location(0) a: bool) {}",
        1,
        17,
        "f32, i32 or u32",
    );
    expect_err(
        "@fragment fn fs(@builtin(position) p: vec3f) {}",
        1,
        17,
        "must have type vec4<f32>",
    );
    expect_err("@compute fn cs() {}", 1, 1, "@workgroup_size");
    expect_err(
        "fn helper(@location(0) a: f32) {}",
        1,
        11,
        "only allowed on entry-point",
    );
    expect_err(
        "var<push_constant> a: vec4f; var<push_constant> b: vec4f;\n@fragment fn fs() -> @location(0) vec4f { return a + b; }",
        1,
        49,
        "two push constant blocks",
    );
    expect_err(
        "@group(0) @binding(0) var<uniform> a: vec4f; @group(0) @binding(0) var<uniform> b: vec4f;\n@fragment fn fs() -> @location(0) vec4f { return a + b; }",
        1,
        81,
        "both use @group(0) @binding(0)",
    );
    // Unused duplicates do not matter for an entry point that does not use them.
    ok_all(
        "@group(0) @binding(0) var<uniform> a: vec4f; @group(0) @binding(0) var<uniform> b: vec4f;\n@fragment fn fs() -> @location(0) vec4f { return a; }",
    );
}

// ----------------------------------------------------------------------- diagnostics

#[test]
fn diagnostics_point_at_the_offending_token() {
    expect_err("fn f() { let x = y; }", 1, 18, "unknown identifier 'y'");
    expect_err(
        "fn f() {\n  let x: f32 = 1u;\n}",
        2,
        16,
        "expected f32, found u32",
    );
    expect_err("fn f() { let x = 1; x = 2; }", 1, 21, "immutable");
    expect_err("fn f() { break; }", 1, 10, "outside of a loop");
    expect_err(
        "fn f() { loop { continuing { continue; } } }",
        1,
        30,
        "continuing",
    );
    expect_err(
        "fn f() { loop { continuing { return; } } }",
        1,
        30,
        "continuing",
    );
    expect_err("fn f() { if (1) {} }", 1, 14, "must be bool");
    expect_err("fn f() { switch (1) { default: {} } }", 1, 10, "switch");
    expect_err("fn f(p: ptr<function, f32>) {}", 1, 9, "pointer");
    expect_err("fn f() { let x = &y; }", 1, 18, "pointers");
    expect_err("override k: f32 = 1.0;", 1, 1, "override");
    expect_err("enable f16;", 1, 1, "not supported");
    expect_err("fn f() { let h = 1.0h; }", 1, 18, "f16");
    expect_err("fn f() { var h: f16; }", 1, 17, "f16");
    expect_err(
        "@group(0) @binding(0) var<storage, read_write> a: atomic<u32>;",
        1,
        51,
        "atomic",
    );
    expect_err("var<workgroup> w: array<f32, 64>;", 1, 16, "workgroup");
    expect_err("fn f() {}\nfn f() {}", 2, 4, "already declared");
    expect_err("fn a() { b(); }\nfn b() { a(); }", 1, 4, "recursive");
    expect_err(
        "fn f() { let x = 1;\n let x = 2; }",
        2,
        6,
        "already declared in this scope",
    );
    expect_err("let x = 1;", 1, 1, "module scope");
    expect_err("fn f() { let x = 1 }", 1, 20, "expected ';'");
    expect_err("fn f() { let x = (1 + 2; }", 1, 24, "expected ')'");
    expect_err("@frobnicate fn f() {}", 1, 1, "@frobnicate");
    expect_err(
        "struct S { a: f32 }\nfn f(s: S) { let x = s.b; }",
        2,
        24,
        "no member 'b'",
    );
    expect_err("fn f() { let x = vec3<f32>; }", 1, 18, "type");
    expect_err("const A = B; const B = A;", 1, 7, "itself");
    expect_err("struct S { s: S }", 1, 8, "itself");
    expect_err("fn f(t: texture_2d<f32>) {}", 1, 9, "function parameters");
    expect_err(
        "fn f() { let x = textureSample(1, 2, 3); }",
        1,
        32,
        "texture variable",
    );
    expect_err("const_assert 1 > 2;", 1, 16, "const_assert failed");
    expect_err("fn f() { let x = 1 & 2 | 3; }", 1, 24, "parentheses");
    expect_err("fn f() { let x = a < b < c; }", 1, 24, "parentheses");
}

#[test]
fn error_display_has_line_and_column() {
    let e = compile("fn f() {\n  let x = zz;\n}", Stage::Compute, "f").unwrap_err();
    assert_eq!(e.to_string(), "2:11: unknown identifier 'zz'");
}

// ----------------------------------------------------------------------- robustness

/// Deterministic xorshift for the fuzz tests.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn corpus_sources() -> Vec<String> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpus");
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .expect("corpus")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "wgsl"))
        .collect();
    files.sort();
    files
        .iter()
        .map(|p| std::fs::read_to_string(p).unwrap())
        .collect()
}

/// Run every stage/entry combination; any outcome but a panic is fine.
fn exercise(src: &str) {
    let r = std::panic::catch_unwind(|| {
        let _ = entry_points(src);
        for (stage, entry) in [
            (Stage::Vertex, "vs_main"),
            (Stage::Fragment, "fs_main"),
            (Stage::Compute, "main"),
            (Stage::Fragment, "fs_post"),
            (Stage::Compute, "splat"),
        ] {
            let _ = compile(src, stage, entry);
        }
        if let Ok(eps) = entry_points(src) {
            for (n, s) in eps {
                let _ = compile(src, s, &n);
            }
        }
    });
    if r.is_err() {
        panic!("compiler panicked on input:\n{src}");
    }
}

#[test]
fn truncated_sources_return_errors_not_panics() {
    for src in corpus_sources() {
        let chars: Vec<char> = src.chars().collect();
        let step = (chars.len() / 120).max(1);
        let mut cut = 0;
        while cut < chars.len() {
            let s: String = chars[..cut].iter().collect();
            exercise(&s);
            cut += step;
        }
    }
}

#[test]
fn mutated_sources_return_errors_not_panics() {
    let pieces = [
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "<",
        ">",
        ">>",
        "<<",
        ";",
        ",",
        ".",
        ":",
        "=",
        "+",
        "-",
        "*",
        "/",
        "%",
        "&",
        "|",
        "^",
        "!",
        "~",
        "@",
        "->",
        "_",
        "0",
        "1u",
        "2.5",
        "0x",
        "1e",
        "var",
        "let",
        "fn",
        "loop",
        "break",
        "continue",
        "return",
        "if",
        "else",
        "for",
        "vec3f",
        "array",
        "f32",
        "u32",
        "x",
        "struct",
        "&&",
        "||",
        "continuing",
        "break if",
        "discard",
        "textureSample",
        "mat3x3f",
        "<f32>",
        "bitcast",
        "@location(0)",
        "@builtin(position)",
        "var<private>",
        "const",
        "arrayLength(&",
        "\u{1F600}",
        "/*",
        "*/",
        "//",
    ];
    let sources = corpus_sources();
    let seed = std::env::var("GANBATE_SHADER_FUZZ_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x2545_f491_4f6c_dd1d_u64);
    let mut rng = Rng(seed | 1);
    // GANBATE_SHADER_FUZZ_ROUNDS raises the round count for longer local campaigns.
    let rounds = std::env::var("GANBATE_SHADER_FUZZ_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(600usize);
    for round in 0..rounds {
        let base = &sources[round % sources.len()];
        let mut s: Vec<char> = base.chars().collect();
        for _ in 0..1 + rng.below(4) {
            let at = rng.below(s.len());
            match rng.below(4) {
                0 => {
                    let len = 1 + rng.below(12);
                    let end = (at + len).min(s.len());
                    s.drain(at..end);
                }
                1 => {
                    let p = pieces[rng.below(pieces.len())];
                    for (k, c) in p.chars().enumerate() {
                        s.insert((at + k).min(s.len()), c);
                    }
                }
                2 => {
                    if at < s.len() {
                        s[at] = pieces[rng.below(pieces.len())]
                            .chars()
                            .next()
                            .unwrap_or('x');
                    }
                }
                _ => {
                    let len = rng.below(40);
                    let end = (at + len).min(s.len());
                    let chunk: Vec<char> = s[at..end].to_vec();
                    let to = rng.below(s.len());
                    for (k, c) in chunk.into_iter().enumerate() {
                        s.insert((to + k).min(s.len()), c);
                    }
                }
            }
        }
        exercise(&s.into_iter().collect::<String>());
    }
}

#[test]
fn garbage_returns_errors_not_panics() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let alphabet: Vec<char> = "abcxyz019_ \n\t(){}[]<>=!&|^~+-*/%.,;:@'\"\\#$?\u{0}\u{7f}é水"
        .chars()
        .collect();
    for _ in 0..400 {
        let len = rng.below(200);
        let s: String = (0..len)
            .map(|_| alphabet[rng.below(alphabet.len())])
            .collect();
        exercise(&s);
    }
    exercise("");
    exercise("@");
    exercise("fn");
    exercise("fn f(");
    exercise("struct S {");
    exercise("var<");
    exercise("@vertex fn vs_main() -> @builtin(position) vec4f { return vec4f(");
}

#[test]
fn deep_nesting_is_rejected_or_fits_in_a_small_stack() {
    // The deepest accepted input must compile on a 2 MiB thread (debug build), and anything
    // deeper must be a clean error.
    let depth = crate::parser::MAX_DEPTH as usize - 8;
    let chain = crate::parser::MAX_EXPR_DEPTH - 2;
    let mut body = String::new();
    for _ in 0..depth / 2 {
        body.push_str("if (p.x > 0.0) { ");
    }
    body.push_str(&format!("acc += {};", vec!["p.y"; chain].join(" + ")));
    for _ in 0..depth / 2 {
        body.push('}');
    }
    let src = format!(
        "@fragment fn main(@location(0) p: vec4f) -> @location(0) vec4f {{ var acc = 0.0; {body} return vec4f(acc); }}"
    );
    let words = std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(move || compile(&src, Stage::Fragment, "main").map(|c| c.words))
        .unwrap()
        .join()
        .expect("compiler thread")
        .expect("deep but allowed nesting compiles");
    validate(&words).unwrap();

    let long = format!(
        "fn f(a: f32) -> f32 {{ return {}; }}",
        vec!["a"; 400].join(" + ")
    );
    assert!(err(&long).message.contains("too deeply nested"));
    let parens = format!(
        "fn f(a: f32) -> f32 {{ return {}a{}; }}",
        "(".repeat(300),
        ")".repeat(300)
    );
    assert!(err(&parens).message.contains("too deep"));
    let blocks = format!("fn f() {{ {}{} }}", "{".repeat(300), "}".repeat(300));
    assert!(err(&blocks).message.contains("too deep"));
}

#[test]
fn spirv_header_and_entry_point() {
    let c = ok(&frag("return p;"), Stage::Fragment, "main");
    assert_eq!(&c.words[..2], &[0x0723_0203, 0x0001_0000]);
    assert_eq!(c.to_bytes().len(), c.words.len() * 4);
    let bound = c.words[3];
    assert!(bound > 10 && bound < 100, "{bound}");
    if let Some(dis) = disassemble(&c.words) {
        assert!(
            dis.contains("OpEntryPoint Fragment %main_0 \"main\""),
            "{dis}"
        );
        assert!(dis.contains("OpExecutionMode %main_0 OriginUpperLeft"));
        assert!(dis.contains("OpMemoryModel Logical GLSL450"));
        assert!(dis.contains("OpExtInstImport \"GLSL.std.450\""));
        assert!(
            !dis.contains("ImageQuery"),
            "capabilities are only declared when used"
        );
    }
}

#[test]
fn pathological_declarations_are_errors_not_crashes() {
    // A long dependency chain declared in reverse order would recurse once per link.
    let mut rev = String::new();
    for i in (1..300).rev() {
        rev.push_str(&format!("const C{i} = C{} + 1;\n", i - 1));
    }
    rev.push_str("const C0 = 0;\n");
    assert!(err(&rev).message.contains("dependency order"));
    // The same chain in dependency order is fine.
    let mut fwd = String::from("const C0 = 0;\n");
    for i in 1..300 {
        fwd.push_str(&format!("const C{i} = C{} + 1;\n", i - 1));
    }
    fwd.push_str("@compute @workgroup_size(1) fn main() { _ = C299; }");
    let p = program(&fwd);
    assert_eq!(p.functions.len(), 1);
    // Deeply nested types.
    let mut nested = String::from("struct S0 { a: f32 }\n");
    for i in 1..40 {
        nested.push_str(&format!("struct S{i} {{ a: S{} }}\n", i - 1));
    }
    assert!(err(&nested).message.contains("levels deep"));
    // Huge arrays.
    expect_err(
        "var<private> a: array<f32, 4000000000>;",
        1,
        17,
        "too large",
    );
    expect_err(
        "struct S { a: array<vec4f, 60000000>, b: array<vec4f, 60000000> }",
        1,
        8,
        "too large",
    );
    // Large zero values become OpConstantNull instead of giant constants.
    let c = ok(
        &frag(
            "var big = array<vec4f, 100000>(); big[u32(p.x)] = p; return big[7] + array<f32, 70000>()[3];",
        ),
        Stage::Fragment,
        "main",
    );
    assert!(c.words.len() < 1000, "{}", c.words.len());
}

// ------------------------------------------------ #import & specialization v2

#[test]
fn import_diamond_dedup_and_error_reporting() {
    let lib = crate::ShaderLibrary::new()
        .with_module(
            "math",
            "const PI: f32 = 3.14159265;\nfn sq(x: f32) -> f32 { return x * x; }\n",
        )
        .with_module(
            "brdf_a",
            "#import math\nfn lobe_a(x: f32) -> f32 { return sq(x) / PI; }\n",
        )
        .with_module(
            "brdf_b",
            "#import \"math\";\nfn lobe_b(x: f32) -> f32 { return sq(x) * 2.0; }\n",
        );

    let src = r#"
/* #import nonexistent_in_comment */
// #import another_comment
#import brdf_a
#import brdf_b

@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
    return vec4f(lobe_a(uv.x), lobe_b(uv.y), PI, 1.0);
}
"#;
    let compiled = lib.compile_wgsl(src, Stage::Fragment, "main", &[]).unwrap();
    assert!(!compiled.words.is_empty());
    let wgsl = compiled.wgsl.as_ref().unwrap().source.as_str();
    assert!(wgsl.contains("lobe_a") && wgsl.contains("lobe_b"));

    // Unknown module reports 1-based line and column.
    let missing = crate::compose("const X = 1;\n  #import missing_mod\n", &[]).unwrap_err();
    assert_eq!(missing.line, 2);
    assert_eq!(missing.column, 11);
    assert!(missing.message.contains("missing_mod"), "{}", missing);

    // Cyclic imports fail cleanly without recursion overflow.
    let cyc =
        crate::compose("#import a\n", &[("a", "#import b\n"), ("b", "#import a\n")]).unwrap_err();
    assert!(cyc.message.contains("cyclic"), "{}", cyc);
}

#[test]
fn specialize_prunes_dead_brdf_branches_and_unused_bindings_all_backends() {
    let lib = crate::ShaderLibrary::new().with_module(
        "brdf",
        r#"
@group(0) @binding(0) var<uniform> params: vec4f;
@group(0) @binding(1) var lut_tex: texture_2d<f32>;
@group(0) @binding(2) var lut_smp: sampler;

fn eval_stylized(ndotl: f32) -> vec3f {
    return vec3f(smoothstep(0.0, 0.15, ndotl)) * params.xyz;
}

fn eval_photoreal_lut(ndotl: f32, rough: f32) -> vec3f {
    let sample_val = textureSampleLevel(lut_tex, lut_smp, vec2f(ndotl, rough), 0.0).xyz;
    return sample_val * params.xyz;
}
"#,
    );

    let shader = r#"
#import brdf

const SHADING_MODE: u32 = 0u;
const ENABLE_CLEARCOAT: bool = false;
const COAT_STRENGTH: f32 = 0.25;

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    var col = vec3f(0.0);
    if SHADING_MODE == 0u {
        col = eval_stylized(uv.x);
    } else {
        col = eval_photoreal_lut(uv.x, uv.y);
    }
    if ENABLE_CLEARCOAT {
        col += vec3f(COAT_STRENGTH * uv.y);
    }
    return vec4f(col, 1.0);
}
"#;

    // Permutation A: Stylized (SHADING_MODE = 0u) -> `lut_tex`, `lut_smp`, and `eval_photoreal_lut`
    // must be completely stripped from reflection and every backend output!
    let spec_stylized = [
        ("SHADING_MODE", crate::SpecValue::U32(0)),
        ("ENABLE_CLEARCOAT", crate::SpecValue::Bool(false)),
    ];
    let spv_a = lib
        .compile_msl(shader, Stage::Fragment, "fs_main", &spec_stylized)
        .unwrap();
    let wgsl_a = lib
        .compile_wgsl(shader, Stage::Fragment, "fs_main", &spec_stylized)
        .unwrap();
    let glsl_a = lib
        .compile_glsl(shader, Stage::Fragment, "fs_main", &spec_stylized)
        .unwrap();

    assert_eq!(
        spv_a.bindings.len(),
        1,
        "stylized permutation should only reflect uniform binding 0: {:?}",
        spv_a.bindings
    );
    assert_eq!(spv_a.bindings[0].binding, 0);
    let msl_src_a = &spv_a.msl.as_ref().unwrap().source;
    let wgsl_src_a = &wgsl_a.wgsl.as_ref().unwrap().source;
    let glsl_src_a = &glsl_a.glsl.as_ref().unwrap().source;
    assert!(!msl_src_a.contains("eval_photoreal_lut"), "{msl_src_a}");
    assert!(!wgsl_src_a.contains("eval_photoreal_lut"), "{wgsl_src_a}");
    assert!(!glsl_src_a.contains("eval_photoreal_lut"), "{glsl_src_a}");
    assert!(wgsl_src_a.contains("eval_stylized"), "{wgsl_src_a}");

    // Permutation B: Photoreal + ClearCoat -> reflects all 3 bindings, strips `eval_stylized`.
    let spec_photo = [
        ("SHADING_MODE", crate::SpecValue::U32(1)),
        ("ENABLE_CLEARCOAT", crate::SpecValue::Bool(true)),
        ("COAT_STRENGTH", crate::SpecValue::F32(0.75)),
    ];
    let wgsl_b = lib
        .compile_wgsl(shader, Stage::Fragment, "fs_main", &spec_photo)
        .unwrap();
    let msl_b = lib
        .compile_msl(shader, Stage::Fragment, "fs_main", &spec_photo)
        .unwrap();
    let glsl_b = lib
        .compile_glsl(shader, Stage::Fragment, "fs_main", &spec_photo)
        .unwrap();
    assert_eq!(wgsl_b.bindings.len(), 3);
    assert!(
        !wgsl_b
            .wgsl
            .as_ref()
            .unwrap()
            .source
            .contains("eval_stylized")
    );
    assert!(
        wgsl_b
            .wgsl
            .as_ref()
            .unwrap()
            .source
            .contains("eval_photoreal_lut")
    );
    assert!(!msl_b.msl.as_ref().unwrap().source.contains("eval_stylized"));
    assert!(
        !glsl_b
            .glsl
            .as_ref()
            .unwrap()
            .source
            .contains("eval_stylized")
    );
    assert!(spv_a.words != wgsl_b.words);

    // Dead branches are still type-checked so broken inactive permutations cannot hide bugs!
    let broken_dead = r#"
const USE_FAST: bool = true;
@fragment fn fs_main() -> @location(0) vec4f {
    if USE_FAST {
        return vec4f(1.0);
    } else {
        return vec4f(unknown_ident);
    }
}
"#;
    let err = crate::compile(broken_dead, Stage::Fragment, "fs_main").unwrap_err();
    assert!(err.message.contains("unknown_ident"), "{err}");

    // Invalid specialization name or mismatched type produces a clean compile error.
    let bad_name = lib
        .compile(
            shader,
            Stage::Fragment,
            "fs_main",
            &[("NO_SUCH_CONST", crate::SpecValue::Bool(true))],
        )
        .unwrap_err();
    assert!(bad_name.message.contains("NO_SUCH_CONST"), "{bad_name}");

    let bad_type = lib
        .compile(
            shader,
            Stage::Fragment,
            "fs_main",
            &[("ENABLE_CLEARCOAT", crate::SpecValue::U32(3))],
        )
        .unwrap_err();
    assert!(bad_type.message.contains("ENABLE_CLEARCOAT"), "{bad_type}");
}

#[test]
fn define_import_path_deduplicates_preconcatenated_and_library_modules() {
    let common = "#define_import_path common\nfn base_val() -> f32 { return 0.25; }\n";
    let brdf = "#define_import_path brdf\n#import common\nfn shade_val() -> f32 { return base_val() * 2.0; }\n";
    let main = "#import common\n#import brdf\n@fragment fn fs_main() -> @location(0) vec4f { return vec4f(shade_val()); }\n";

    // Pre-concatenated translation unit without ShaderLibrary imports.
    let concat = format!("{common}\n{brdf}\n{main}");
    crate::compile_glsl(&concat, Stage::Fragment, "fs_main").expect("concat glsl");
    crate::compile_msl(&concat, Stage::Fragment, "fs_main").expect("concat msl");

    // Library-composed translation unit.
    let lib = crate::ShaderLibrary::new()
        .with_module("common", common)
        .with_module("brdf", brdf);
    lib.compile(main, Stage::Fragment, "fs_main", &[])
        .expect("library spirv");
}
