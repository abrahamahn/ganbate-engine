//! Metal Shading Language output: emits every entry point of every `tests/corpus/*.wgsl` file
//! and checks the binding maps and conventions. When `clang++` is on `PATH` (or named by
//! `GANBATE_MSL_CLANG`), each module is also parsed and type-checked against the stand-in
//! `tests/msl/metal_stdlib`, including the generated struct-layout assertions. This is a
//! compile check with a stock C++ compiler, not Apple's `metal` compiler: it catches syntax,
//! name, arity, type and layout mistakes, and a real Metal compile on a Mac remains the final
//! word.

use std::path::{Path, PathBuf};

#[path = "msl/check.rs"]
mod check;
use check::{clang, clang_check};

use ganbate_shader::{
    MslIndex, MslSlot, PUSH_CONSTANT_BUFFER, SIZES_BUFFER, Stage, VERTEX_BUFFER, compile_msl,
    entry_points,
};

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpus")
}

fn read(name: &str) -> String {
    std::fs::read_to_string(corpus_dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn corpus_emits_msl_that_compiles() {
    let mut files: Vec<PathBuf> = std::fs::read_dir(corpus_dir())
        .expect("corpus directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "wgsl"))
        .collect();
    files.sort();
    let clang = clang();
    if clang.is_none() {
        eprintln!("note: clang++ not found; MSL is generated but not compiled");
    }
    let mut failures = Vec::new();
    let mut modules = 0;
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let src = std::fs::read_to_string(path).unwrap();
        for (entry, stage) in entry_points(&src).unwrap() {
            let tag = format!("{name}:{entry}");
            let c = match compile_msl(&src, stage, &entry) {
                Ok(c) => c,
                Err(e) => {
                    failures.push(format!("{tag}: {e}"));
                    continue;
                }
            };
            let msl = c.msl.expect("compile_msl fills msl");
            modules += 1;
            if let Some(clang) = &clang
                && let Err(e) = clang_check(clang, &tag, &msl.source)
            {
                failures.push(format!("{tag}: clang++ rejected the MSL:\n{e}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "MSL failures:\n{}",
        failures.join("\n")
    );
    eprintln!(
        "{modules} corpus entry points emitted as MSL{}",
        if clang.is_some() {
            " and compiled with clang++ against the stand-in metal_stdlib"
        } else {
            ""
        }
    );
}

#[test]
fn reference_binding_map() {
    let src = read("reference.wgsl");
    let vs = compile_msl(&src, Stage::Vertex, "vs_main").unwrap();
    let m = vs.msl.unwrap();
    assert_eq!(m.entry_point, "ganbate_entry_vs_main");
    assert_eq!(
        m.slots,
        vec![MslSlot {
            group: 0,
            binding: 0,
            index: MslIndex::Buffer(0)
        }]
    );
    assert_eq!(m.push_constant_buffer, Some(PUSH_CONSTANT_BUFFER));
    assert_eq!(m.vertex_buffer, Some(VERTEX_BUFFER));
    assert!(m.source.contains("[[stage_in]]"));
    assert!(m.source.contains("[[attribute(2)]]"));
    assert!(m.source.contains("[[user(locn3)]]"));
    // Vulkan clip space is flipped to Metal's.
    assert!(m.source.contains(".position.y = -"));

    let fs = compile_msl(&src, Stage::Fragment, "fs_main").unwrap();
    let m = fs.msl.unwrap();
    let kinds: Vec<(u32, u32, MslIndex)> = m
        .slots
        .iter()
        .map(|s| (s.group, s.binding, s.index))
        .collect();
    assert_eq!(
        kinds,
        vec![
            (0, 0, MslIndex::Buffer(0)),
            (0, 1, MslIndex::Buffer(1)),
            (0, 2, MslIndex::Buffer(2)),
            (0, 3, MslIndex::Buffer(3)),
            (0, 4, MslIndex::Texture(0)),
            (0, 5, MslIndex::Sampler(0)),
            (1, 0, MslIndex::Texture(1)),
            (1, 2, MslIndex::Sampler(1)),
        ]
    );
    assert_eq!(m.vertex_buffer, None);
    // Integer varyings are flat; comparison sampling uses an explicit level.
    assert!(m.source.contains("[[user(locn3), flat]]"), "{}", m.source);
    assert!(m.source.contains("metal::level(0)"));
    assert!(m.source.contains("metal::discard_fragment();"));
    // The SPIR-V and reflection are unchanged by MSL emission.
    assert_eq!(
        fs.words,
        ganbate_shader::compile(&src, Stage::Fragment, "fs_main")
            .unwrap()
            .words
    );
}

#[test]
fn runtime_arrays_read_their_sizes() {
    let src = read("compute.wgsl");
    let c = compile_msl(&src, Stage::Compute, "splat").unwrap();
    let m = c.msl.unwrap();
    assert_eq!(m.sized_bindings, vec![(0, 0), (0, 4)]);
    assert!(m.source.contains(&format!("[[buffer({SIZES_BUFFER})]]")));
    // ParticleBuffer.items starts at byte 16 and Particle is 32 bytes.
    assert!(m.source.contains("- 16u) / 32u)"), "{}", m.source);
    assert!(m.source.contains("kernel void ganbate_entry_splat("));
    assert!(m.source.contains("[[thread_position_in_grid]]"));
    // Particle { vec3, f32, vec3, u32 } keeps WGSL's 12-byte vec3 slots.
    assert!(
        m.source.contains("metal::packed_float3 position;"),
        "{}",
        m.source
    );
    assert!(m.source.contains("static_assert(sizeof(Particle) == 32"));
}

#[test]
fn uniform_layout_padding_matches_wgsl() {
    let src = read("language.wgsl");
    let c = compile_msl(&src, Stage::Compute, "main").unwrap();
    let m = c.msl.unwrap();
    // `Row { @size(16) value: f32 }` is padded to 16 bytes, `Uniforms.offset` (a vec3
    // followed by a scalar) is packed, and the struct keeps WGSL's 128-byte size.
    assert!(m.source.contains("static_assert(sizeof(Row) == 16"));
    assert!(m.source.contains("metal::packed_float3 offset;"));
    assert!(m.source.contains("static_assert(sizeof(Uniforms) == 128"));
    // `main` is not a valid Metal function name.
    assert!(!m.source.contains(" main("));
    assert_eq!(m.entry_point, "ganbate_entry_main");
}

fn msl_of(src: &str, stage: Stage, entry: &str) -> String {
    let m = compile_msl(src, stage, entry)
        .unwrap_or_else(|e| panic!("{e}"))
        .msl
        .unwrap();
    if let Some(clang) = clang() {
        clang_check(&clang, entry, &m.source).unwrap_or_else(|e| panic!("{e}\n{}", m.source));
    }
    m.source
}

#[test]
fn continuing_reads_body_lets_after_a_continue() {
    // `continue` jumps to `continuing`, which reads a `let` of the body: the MSL loop runs
    // `continuing` at the top of the next iteration, so the `let` lives outside the loop.
    let src = "
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1) fn main() {
  var i = 0u;
  var total = 0u;
  loop {
    let step = i * 2u + 1u;
    if (step > 7u) {
      continue;
    }
    total += step;
    continuing {
      i += step;
      break if i > 20u;
    }
  }
  out[0] = total;
}";
    let m = msl_of(src, Stage::Compute, "main");
    assert!(m.contains("uint step;"), "{m}");
    assert!(m.contains("step = ((i * 2u) + 1u);"), "{m}");
    assert!(m.contains("if (ganbate_continuing) {"), "{m}");
}

#[test]
fn side_effects_keep_wgsl_evaluation_order() {
    // `bump` writes `counter`, so `counter + bump()` must read `counter` first, and the
    // short-circuit right operand with two such calls must only run when needed.
    let src = "
var<private> counter: u32;
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
fn bump() -> u32 {
  counter += 1u;
  return counter;
}
@compute @workgroup_size(1) fn main() {
  let a = counter + bump();
  let b = out[0] > 3u && bump() + bump() > 4u;
  out[1] = a + u32(b);
}";
    let m = msl_of(src, Stage::Compute, "main");
    let first = m.find("const uint ganbate_t = counter;").expect(&m);
    let call = m.find("const uint ganbate_t_1 = bump(counter);").expect(&m);
    assert!(first < call, "{m}");
    assert!(m.contains("bool ganbate_cond = (out[0u] > 3u);"), "{m}");
    assert!(m.contains("if (ganbate_cond) {"), "{m}");
}

#[test]
fn packed_members_are_written_whole() {
    // A component store into a packed vec3 member rewrites the whole vector.
    let src = "
struct P { v: vec3f, w: f32 };
@group(0) @binding(0) var<storage, read_write> ps: array<P>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  ps[id.x].v.y = 2.0;
  ps[id.x].v.xz += vec2f(1.0);
  ps[id.x].w = ps[id.x].v.z;
}";
    let m = msl_of(src, Stage::Compute, "main");
    assert!(m.contains("metal::packed_float3 v;"), "{m}");
    assert!(
        m.contains("device metal::packed_float3& ganbate_ref"),
        "{m}"
    );
    assert!(m.contains("metal::float3(ps[id.x].v).z"), "{m}");
}
