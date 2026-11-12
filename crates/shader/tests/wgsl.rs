//! WebGPU WGSL output: every corpus entry point compiles through `compile_wgsl`, with the
//! same reflection as the SPIR-V path, and the lowering of push constants, entry-point IO and
//! assignments has the expected shape.
//!
//! Set `GANBATE_WGSL_DUMP=dir` to write each module to `dir/<file>.<entry>.<stage>.wgsl`;
//! `web/tests/validate-wgsl.mjs` then compiles them with a browser's WebGPU implementation.

use std::path::{Path, PathBuf};

use ganbate_shader::wgsl::{PUSH_CONSTANT_BINDING, PUSH_CONSTANT_GROUP};
use ganbate_shader::{Stage, compile, compile_wgsl, entry_points};

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpus")
}

fn wgsl(src: &str, stage: Stage, entry: &str) -> ganbate_shader::wgsl::Wgsl {
    compile_wgsl(src, stage, entry)
        .unwrap_or_else(|e| panic!("{entry}: {e}\n{src}"))
        .wgsl
        .expect("compile_wgsl fills Compiled::wgsl")
}

#[test]
fn corpus_compiles_to_wgsl_with_the_same_reflection() {
    let dump = std::env::var_os("GANBATE_WGSL_DUMP").map(PathBuf::from);
    if let Some(d) = &dump {
        std::fs::create_dir_all(d).expect("dump directory");
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(corpus_dir())
        .expect("corpus")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "wgsl"))
        .collect();
    files.sort();
    let mut modules = 0;
    for path in &files {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let src = std::fs::read_to_string(path).unwrap();
        for (entry, stage) in entry_points(&src).unwrap() {
            let spirv = compile(&src, stage, &entry).unwrap();
            let both = compile_wgsl(&src, stage, &entry)
                .unwrap_or_else(|e| panic!("{name}:{e} (entry '{entry}')"));
            let w = both.wgsl.clone().expect("wgsl");
            assert_eq!(
                (spirv.bindings, spirv.push_constant_size, spirv.inputs),
                (both.bindings, both.push_constant_size, both.inputs),
                "{name}:{entry}: reflection differs"
            );
            assert_eq!(spirv.words, both.words, "{name}:{entry}: SPIR-V differs");
            assert_eq!(
                w.push_constants.is_some(),
                both.push_constant_size > 0,
                "{name}:{entry}"
            );
            assert!(
                w.source.contains(&format!("fn {}(", w.entry)),
                "{name}:{entry}: entry point missing\n{}",
                w.source
            );
            assert!(!w.source.contains("push_constant>"), "{}", w.source);
            if let Some(d) = &dump {
                let stage = format!("{stage:?}").to_lowercase();
                let file = d.join(format!("{name}.{entry}.{stage}.wgsl"));
                std::fs::write(file, &w.source).unwrap();
            }
            modules += 1;
        }
    }
    assert!(modules >= 10, "only {modules} corpus entry points");
}

#[test]
fn push_constants_become_a_dynamic_uniform() {
    let src = "struct P { m: mat4x4f, tint: vec4f };
        var<push_constant> pc: P;
        @group(0) @binding(0) var<uniform> u: vec4f;
        @vertex fn vs(@location(0) p: vec3f) -> @builtin(position) vec4f {
          return pc.m * vec4f(p, 1.0) + u * pc.tint.x;
        }";
    let w = wgsl(src, Stage::Vertex, "vs");
    assert_eq!(
        w.push_constants,
        Some((PUSH_CONSTANT_GROUP, PUSH_CONSTANT_BINDING))
    );
    assert!(
        w.source
            .contains("@group(3) @binding(0) var<uniform> pc: P;"),
        "{}",
        w.source
    );
    // The clip-space y flip happens in the generated wrapper.
    assert!(
        w.source.contains("vec4<f32>(1.0f, -1.0f, 1.0f, 1.0f)"),
        "{}",
        w.source
    );
    // A shader without push constants reports none and may use group 3.
    let free = wgsl(
        "@group(3) @binding(0) var<uniform> u: vec4f;
         @fragment fn fs() -> @location(0) vec4f { return u; }",
        Stage::Fragment,
        "fs",
    );
    assert_eq!(free.push_constants, None);
    assert!(!free.source.contains("-1.0f"), "{}", free.source);
}

#[test]
fn group_three_is_reserved_when_push_constants_are_used() {
    let src = "var<push_constant> pc: vec4f;
        @group(3) @binding(1) var<uniform> u: vec4f;
        @fragment fn fs() -> @location(0) vec4f { return pc + u; }";
    let err = compile_wgsl(src, Stage::Fragment, "fs").unwrap_err();
    assert!(err.message.contains("@group(3)"), "{err}");
    // The SPIR-V path is unaffected.
    assert!(compile(src, Stage::Fragment, "fs").is_ok());
}

#[test]
fn push_constant_blocks_must_fit_uniform_layout() {
    let src = "struct P { a: f32, b: array<f32, 4> };
        var<push_constant> pc: P;
        @fragment fn fs() -> @location(0) vec4f { return vec4f(pc.b[1]); }";
    assert!(compile(src, Stage::Fragment, "fs").is_ok());
    let err = compile_wgsl(src, Stage::Fragment, "fs").unwrap_err();
    assert!(err.message.contains("uniform"), "{err}");
}

#[test]
fn duplicate_builtin_inputs_and_integer_varyings() {
    let src =
        "struct V { @builtin(position) clip: vec4f, @location(0) id: u32, @location(1) uv: vec2f };
        @fragment fn fs(v: V, @builtin(position) frag: vec4f) -> @location(0) vec4f {
          return vec4f(frag.xy + v.clip.zw, f32(v.id), v.uv.x);
        }";
    let w = wgsl(src, Stage::Fragment, "fs");
    assert_eq!(
        w.source.matches("@builtin(position)").count(),
        1,
        "{}",
        w.source
    );
    assert!(
        w.source.contains("@location(0) @interpolate(flat)"),
        "{}",
        w.source
    );
}

#[test]
fn swizzle_and_compound_assignments_are_lowered() {
    let src = "@group(0) @binding(0) var<storage, read_write> buf: array<vec4f>;
        fn next(i: u32) -> u32 { return i + 1u; }
        @compute @workgroup_size(1) fn cs(@builtin(global_invocation_id) id: vec3u) {
          var v = vec4f(1.0);
          v.zx += vec2f(2.0, 3.0);
          v.yw = v.xz;
          buf[next(id.x)] += v;
          buf[id.x].y *= 2.0;
        }";
    let w = wgsl(src, Stage::Compute, "cs");
    // No multi-component swizzle on the left of an assignment.
    for line in w.source.lines() {
        if let Some((lhs, _)) = line.split_once(" = ")
            && let Some((_, comps)) = lhs.trim().rsplit_once('.')
        {
            assert!(
                comps.len() == 1 || comps.chars().any(|c| !"xyzw".contains(c)),
                "{line}"
            );
        }
    }
    // The dynamic index of the compound assignment is evaluated once.
    assert_eq!(w.source.matches("next(").count(), 2, "{}", w.source);
    assert!(
        w.source.contains("@compute @workgroup_size(1, 1, 1)"),
        "{}",
        w.source
    );
}

#[test]
fn reserved_and_predeclared_names_are_renamed() {
    let src = "struct type { filter: f32, max: f32 };
        @group(0) @binding(0) var<uniform> target: type;
        fn select2(x: f32) -> f32 { return x; }
        @fragment fn fs() -> @location(0) vec4f {
          let match = target.filter;
          return vec4f(select2(max(match, target.max)));
        }";
    let w = wgsl(src, Stage::Fragment, "fs");
    for bad in [
        "struct type ",
        " target:",
        "let match ",
        ".filter;",
        "var<uniform> target:",
    ] {
        assert!(!w.source.contains(bad), "{bad}\n{}", w.source);
    }
    assert!(w.source.contains("max("), "{}", w.source);
}

#[test]
fn discard_keeps_terminate_semantics() {
    let src = "fn f(x: f32) -> f32 {
          if (x < 0.0) { discard; }
          return x;
        }
        @fragment fn fs(@location(0) x: f32) -> @location(0) vec4f { return vec4f(f(x)); }";
    let w = wgsl(src, Stage::Fragment, "fs");
    let lines: Vec<&str> = w.source.lines().map(str::trim).collect();
    let at = lines
        .iter()
        .position(|l| *l == "discard;")
        .expect("discard");
    assert_eq!(lines[at + 1], "return f32();", "{}", w.source);
}
