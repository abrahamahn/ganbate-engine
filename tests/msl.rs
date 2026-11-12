//! The stylized renderer's shaders as Metal Shading Language: every entry point of
//! `src/render/shaders` (composed with the renderer's `#import` library) must emit MSL,
//! and, when `clang++` is available, compile against ganbate-shader's stand-in `metal_stdlib`.
//! A compile check on Linux, not a Metal run.

#[path = "../crates/shader/tests/msl/check.rs"]
mod check;

use ganbate_engine::gpu::shader::{MslIndex, compile_msl, entry_points};
use ganbate_engine::render::compose_shader;

const SHADERS: [(&str, &str); 3] = [
    (
        "scene.wgsl",
        include_str!("../src/render/shaders/scene.wgsl"),
    ),
    (
        "world_fx.wgsl",
        include_str!("../src/render/shaders/world_fx.wgsl"),
    ),
    ("post.wgsl", include_str!("../src/render/shaders/post.wgsl")),
];

#[test]
fn renderer_shaders_emit_msl_that_compiles() {
    let clang = check::clang();
    if clang.is_none() {
        eprintln!("note: clang++ not found; MSL is generated but not compiled");
    }
    let mut failures = Vec::new();
    let mut count = 0;
    for (name, body) in SHADERS {
        let source = compose_shader(body).expect("compose shader");
        for (entry, stage) in entry_points(&source).unwrap() {
            let tag = format!("{name}:{entry}");
            let msl = match compile_msl(&source, stage, &entry) {
                Ok(c) => c.msl.expect("compile_msl fills msl"),
                Err(e) => {
                    failures.push(format!("{tag}: {e}"));
                    continue;
                }
            };
            count += 1;
            // Metal binds at most 31 buffers, 128 textures and 16 samplers per stage.
            let samplers = msl
                .slots
                .iter()
                .filter(|s| matches!(s.index, MslIndex::Sampler(_)))
                .count();
            assert!(samplers <= 16, "{tag}: {samplers} samplers");
            if let Some(clang) = &clang
                && let Err(e) = check::clang_check(clang, &tag, &msl.source)
            {
                failures.push(format!("{tag}: clang++ rejected the MSL:\n{e}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(count >= 13, "only {count} entry points");
}
