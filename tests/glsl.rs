//! The renderer's shaders as GLSL ES 3.00 for WebGL 2: every vertex/fragment pair the
//! renderer builds compiles through `compile_glsl`, and each pair is validated and linked with
//! `glslangValidator` when it is installed (skipped with a note otherwise). The browser check
//! (ANGLE compiling and linking them) is `web/test/run.cjs --api webgl2`.

use std::process::Command;

use ganbate_engine::gpu::shader::{Stage, compile_glsl};
use ganbate_engine::render::compose_shader;

const SCENE: &str = include_str!("../src/render/shaders/scene.wgsl");
const WORLD_FX: &str = include_str!("../src/render/shaders/world_fx.wgsl");
const POST: &str = include_str!("../src/render/shaders/post.wgsl");

/// (body, vertex entry, fragment entry or `None` for depth-only pipelines).
const PAIRS: &[(&str, &str, Option<&str>)] = &[
    (SCENE, "vs_main", Some("fs_main")),
    (SCENE, "vs_prop", Some("fs_main")),
    (SCENE, "vs_prop", Some("fs_overlay")),
    (WORLD_FX, "vs_shadow", None),
    (WORLD_FX, "vs_shadow_masked", Some("fs_shadow_masked")),
    (WORLD_FX, "vs_shadow_prop", None),
    (
        WORLD_FX,
        "vs_shadow_prop_masked",
        Some("fs_shadow_prop_masked"),
    ),
    (WORLD_FX, "vs_ground", Some("fs_ground")),
    (SCENE, "vs_grass", Some("fs_grass")),
    (SCENE, "vs_flower", Some("fs_flower")),
    (WORLD_FX, "vs_sky", Some("fs_sky")),
    (WORLD_FX, "vs_particle", Some("fs_particle")),
    (POST, "vs_fullscreen", Some("fs_ssao")),
    (POST, "vs_fullscreen", Some("fs_air")),
    (POST, "vs_fullscreen", Some("fs_bloom_down")),
    (POST, "vs_fullscreen", Some("fs_bloom_up")),
    (POST, "vs_fullscreen", Some("fs_tonemap")),
    (POST, "vs_fullscreen", Some("fs_tonemap_output")),
    (POST, "vs_fullscreen", Some("fs_tonemap_output_plain")),
    (POST, "vs_fullscreen", Some("fs_paint")),
    (POST, "vs_fullscreen", Some("fs_final")),
];

/// The fragment shader the WebGL backend pairs with depth-only pipelines.
const EMPTY_FS: &str = "#version 300 es\nprecision highp float;\nvoid main() {}\n";

fn glslang() -> bool {
    Command::new("glslangValidator")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn link(vs: &str, fs: &str, tag: &str) -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("ganbate-glsl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let v = dir.join(format!("{tag}.vert"));
    let f = dir.join(format!("{tag}.frag"));
    std::fs::write(&v, vs).map_err(|e| e.to_string())?;
    std::fs::write(&f, fs).map_err(|e| e.to_string())?;
    let out = Command::new("glslangValidator")
        .arg("-l")
        .arg(&v)
        .arg(&f)
        .output()
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&v);
    let _ = std::fs::remove_file(&f);
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stdout).into_owned())
    }
}

#[test]
fn renderer_shaders_compile_and_link_as_glsl_es() {
    let check = glslang();
    if !check {
        eprintln!("note: glslangValidator not found on PATH; the GLSL is not validated");
    }
    let dump = std::env::var_os("GANBATE_GLSL_DUMP").map(std::path::PathBuf::from);
    let mut failures = Vec::new();
    for &(body, vs_entry, fs_entry) in PAIRS {
        let source = compose_shader(body).expect("compose shader");
        let vs = compile_glsl(&source, Stage::Vertex, vs_entry)
            .unwrap_or_else(|e| panic!("{vs_entry}: {e}"))
            .glsl
            .expect("glsl");
        let fs = match fs_entry {
            Some(f) => {
                compile_glsl(&source, Stage::Fragment, f)
                    .unwrap_or_else(|e| panic!("{f}: {e}"))
                    .glsl
                    .expect("glsl")
                    .source
            }
            None => EMPTY_FS.to_string(),
        };
        let tag = format!("{vs_entry}-{}", fs_entry.unwrap_or("depth"));
        if let Some(d) = &dump {
            std::fs::create_dir_all(d).unwrap();
            std::fs::write(d.join(format!("{tag}.vert")), &vs.source).unwrap();
            std::fs::write(d.join(format!("{tag}.frag")), &fs).unwrap();
        }
        if check && let Err(e) = link(&vs.source, &fs, &tag) {
            failures.push(format!("{tag}:\n{e}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

const PROP_SHADER_INPUT: &str = include_str!("../src/render/shaders/prop_shader_input.wgsl");
const PROP_SHADER: &str = include_str!("../src/render/shaders/prop_shader.wgsl");

/// A game's prop shader (here a ring whose sweep and colour come from its parameters, over the
/// base texture sampled softer) assembled as the renderer assembles one, compiled to GLSL ES and
/// linked with the prop vertex shader.
#[test]
fn prop_shaders_compile_and_link_as_glsl_es() {
    let user = "fn shade(in: ShadeInput) -> vec4f {
        let d = in.uv - vec2f(0.5);
        let r = length(d) * 2.0;
        let ring = 1.0 - smoothstep(0.0, 0.08, abs(r - 0.8));
        let angle = atan2(d.x, d.y) / 6.2831853 + 0.5;
        let lit = step(angle, in.p0.x);
        let soft = textureSampleBias(base_tex, material_sampler, in.uv, 2.0);
        return vec4f(mix(in.p1.rgb, soft.rgb, in.p4.x), ring * lit * in.tint.a);
    }";
    let raw = format!("{SCENE}\n{PROP_SHADER_INPUT}\n{user}\n{PROP_SHADER}");
    let source = compose_shader(&raw).expect("compose prop shader");
    let vs = compile_glsl(&source, Stage::Vertex, "vs_prop")
        .unwrap_or_else(|e| panic!("vs_prop: {e}"))
        .glsl
        .expect("glsl");
    let fs = compile_glsl(&source, Stage::Fragment, "fs_prop_shader")
        .unwrap_or_else(|e| panic!("fs_prop_shader: {e}"))
        .glsl
        .expect("glsl");
    if glslang() {
        link(&vs.source, &fs.source, "prop-shader").unwrap_or_else(|e| panic!("{e}"));
    } else {
        eprintln!("note: glslangValidator not found on PATH; the GLSL is not validated");
    }
}
