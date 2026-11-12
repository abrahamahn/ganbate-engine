//! Compiles every entry point of every `tests/corpus/*.wgsl` file and validates each module
//! with `spirv-val --target-env vulkan1.0` (skipped with a note when `spirv-val` is missing).
//! Also checks the reflection of the reference shaders.

use std::path::{Path, PathBuf};
use std::process::Command;

use ganbate_shader::{
    Binding, BindingKind, Compiled, Interface, IoType, SampleType, Stage, TexelFormat, TextureView,
    compile, entry_points,
};

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpus")
}

fn spirv_val_available() -> bool {
    Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn validate(words: &[u32], tag: &str) -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("ganbate-shader-corpus-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("{}.spv", tag.replace(['/', ':', ' '], "_")));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
    let out = Command::new("spirv-val")
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

fn read(name: &str) -> String {
    std::fs::read_to_string(corpus_dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn build(name: &str, stage: Stage, entry: &str) -> Compiled {
    compile(&read(name), stage, entry).unwrap_or_else(|e| panic!("{name}:{e}"))
}

#[test]
fn corpus_compiles_and_validates() {
    let mut files: Vec<PathBuf> = std::fs::read_dir(corpus_dir())
        .expect("corpus directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "wgsl"))
        .collect();
    files.sort();
    assert!(files.len() >= 5, "corpus is missing files");
    let check = spirv_val_available();
    if !check {
        eprintln!("note: spirv-val not found on PATH; compiling the corpus without validation");
    }
    let mut failures = Vec::new();
    let mut modules = 0;
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let src = std::fs::read_to_string(path).unwrap();
        let eps = match entry_points(&src) {
            Ok(eps) => eps,
            Err(e) => {
                failures.push(format!("{name}:{e}"));
                continue;
            }
        };
        assert!(!eps.is_empty(), "{name} declares no entry points");
        for (entry, stage) in eps {
            let tag = format!("{name}:{entry}");
            let start = std::time::Instant::now();
            match compile(&src, stage, &entry) {
                Ok(c) => {
                    modules += 1;
                    let ms = start.elapsed().as_secs_f64() * 1e3;
                    if check && let Err(msg) = validate(&c.words, &tag) {
                        failures.push(format!("{tag}: spirv-val failed:\n{msg}"));
                        continue;
                    }
                    eprintln!(
                        "ok   {tag:<40} {stage:?} {:>6} words {ms:>6.2} ms",
                        c.words.len()
                    );
                }
                Err(e) => failures.push(format!("{name}:{e} (entry '{entry}')")),
            }
        }
    }
    assert!(
        failures.is_empty(),
        "corpus failures:\n{}",
        failures.join("\n")
    );
    eprintln!(
        "{modules} modules compiled{}",
        if check {
            " and validated with spirv-val --target-env vulkan1.0"
        } else {
            ""
        }
    );
}

#[test]
fn reference_vertex_reflection() {
    let c = build("reference.wgsl", Stage::Vertex, "vs_main");
    assert_eq!(c.stage, Stage::Vertex);
    assert_eq!(
        c.bindings,
        vec![Binding {
            group: 0,
            binding: 0,
            kind: BindingKind::UniformBuffer { size: 224 }
        }]
    );
    assert_eq!(c.push_constant_size, 112);
    assert_eq!(
        c.inputs,
        vec![
            Interface {
                location: 0,
                ty: IoType::Vec3F
            },
            Interface {
                location: 1,
                ty: IoType::Vec3F
            },
            Interface {
                location: 2,
                ty: IoType::Vec2F
            },
        ]
    );
    assert_eq!(
        c.outputs,
        vec![
            Interface {
                location: 0,
                ty: IoType::Vec3F
            },
            Interface {
                location: 1,
                ty: IoType::Vec3F
            },
            Interface {
                location: 2,
                ty: IoType::Vec2F
            },
            Interface {
                location: 3,
                ty: IoType::U32
            },
        ]
    );
    assert_eq!(c.workgroup_size, [1, 1, 1]);
    assert_eq!(c.words[0], 0x0723_0203);
    assert_eq!(c.words[1], 0x0001_0000);
}

#[test]
fn reference_fragment_reflection() {
    let c = build("reference.wgsl", Stage::Fragment, "fs_main");
    let b = |group, binding, kind| Binding {
        group,
        binding,
        kind,
    };
    // normal_tex (group 1, binding 1) is declared but never used, so it is not reflected.
    assert_eq!(
        c.bindings,
        vec![
            b(0, 0, BindingKind::UniformBuffer { size: 224 }),
            b(0, 1, BindingKind::StorageBuffer { read_only: true }),
            b(0, 2, BindingKind::StorageBuffer { read_only: true }),
            b(0, 3, BindingKind::StorageBuffer { read_only: true }),
            b(
                0,
                4,
                BindingKind::Texture {
                    view: TextureView::D2Array,
                    sample: SampleType::Depth
                }
            ),
            b(0, 5, BindingKind::Sampler { comparison: true }),
            b(
                1,
                0,
                BindingKind::Texture {
                    view: TextureView::D2,
                    sample: SampleType::Float
                }
            ),
            b(1, 2, BindingKind::Sampler { comparison: false }),
        ]
    );
    assert_eq!(c.push_constant_size, 112);
    assert_eq!(c.inputs.len(), 4);
    assert_eq!(
        c.inputs[3],
        Interface {
            location: 3,
            ty: IoType::U32
        }
    );
    assert_eq!(
        c.outputs,
        vec![Interface {
            location: 0,
            ty: IoType::Vec4F
        }]
    );
}

#[test]
fn compute_reflection() {
    let c = build("compute.wgsl", Stage::Compute, "splat");
    assert_eq!(c.workgroup_size, [8, 8, 1]);
    assert_eq!(
        c.bindings
            .iter()
            .map(|b| (b.group, b.binding, b.kind))
            .collect::<Vec<_>>(),
        vec![
            (0, 0, BindingKind::StorageBuffer { read_only: false }),
            (0, 1, BindingKind::UniformBuffer { size: 48 }),
            (
                0,
                2,
                BindingKind::StorageTexture {
                    view: TextureView::D2,
                    format: TexelFormat::Rgba16Float
                }
            ),
            (
                0,
                3,
                BindingKind::Texture {
                    view: TextureView::D2,
                    sample: SampleType::Float
                }
            ),
            (0, 4, BindingKind::StorageBuffer { read_only: false }),
        ]
    );
    assert!(c.inputs.is_empty() && c.outputs.is_empty());
    let c = build("compute.wgsl", Stage::Compute, "integrate");
    assert_eq!(c.workgroup_size, [64, 1, 1]);
    // `integrate` touches only the particles and the simulation constants.
    assert_eq!(c.bindings.len(), 2);
}

#[test]
fn postprocess_and_shadow_reflection() {
    let c = build("postprocess.wgsl", Stage::Vertex, "vs_fullscreen");
    assert!(
        c.bindings.is_empty(),
        "the full-screen vertex shader uses no resources"
    );
    assert!(c.inputs.is_empty());
    assert_eq!(
        c.outputs,
        vec![Interface {
            location: 0,
            ty: IoType::Vec2F
        }]
    );
    let c = build("postprocess.wgsl", Stage::Fragment, "fs_post");
    assert_eq!(c.bindings.len(), 6);
    assert_eq!(c.bindings[0].kind, BindingKind::UniformBuffer { size: 64 });
    let c = build("shadow.wgsl", Stage::Fragment, "fs_shadow");
    assert!(
        c.outputs.is_empty(),
        "frag_depth is a built-in, not a colour target"
    );
    let c = build("shadow.wgsl", Stage::Fragment, "fs_vsm");
    assert_eq!(
        c.outputs,
        vec![Interface {
            location: 0,
            ty: IoType::Vec2F
        }]
    );
}

#[test]
fn matrices_reflection() {
    let c = build("matrices.wgsl", Stage::Fragment, "fs_main");
    assert_eq!(
        c.bindings,
        vec![Binding {
            group: 0,
            binding: 0,
            kind: BindingKind::UniformBuffer { size: 304 }
        }]
    );
}

#[test]
fn stage_mismatch_and_missing_entry_are_errors() {
    let src = read("reference.wgsl");
    let e = compile(&src, Stage::Fragment, "vs_main").unwrap_err();
    assert!(e.message.contains("vertex"), "{e}");
    let e = compile(&src, Stage::Vertex, "nope").unwrap_err();
    assert!(e.message.contains("no entry point named 'nope'"), "{e}");
    let e = compile(&src, Stage::Fragment, "shadow").unwrap_err();
    assert!(e.message.contains("not an entry point"), "{e}");
}

/// A ~500-line module (the reference shader plus generated helpers that are all reachable from
/// one entry point) must compile quickly; release builds take a few milliseconds.
pub fn big_module() -> String {
    let mut src = read("reference.wgsl");
    let mut calls = String::new();
    for i in 0..32 {
        src.push_str(&format!(
            "
fn helper{i}(x: vec3f, k: i32) -> vec3f {{
  var acc = x;
  for (var j = 0; j < k; j++) {{
    acc = normalize(acc * 1.5 + vec3f(f32(j), 0.5, -0.5));
    if (acc.x > 0.5) {{ continue; }}
    acc = mix(acc, cross(acc, vec3f(0.0, 1.0, 0.0)), 0.25);
  }}
  let m = mat3x3f(acc, acc.yzx, acc.zxy);
  let s = textureSampleLevel(albedo_tex, material_sampler, acc.xy, 0.0);
  return transpose(m) * acc + POISSON[k % 4].xyx + s.rgb * frame.params.x;
}}
"
        ));
        calls.push_str(&format!("  c += helper{i}(c, {});\n", i % 5));
    }
    src.push_str(&format!(
        "@fragment fn fs_big(@location(0) p: vec3f) -> @location(0) vec4f {{\n  var c = p;\n{calls}  return vec4f(c, 1.0);\n}}\n"
    ));
    src
}

#[test]
fn compile_time_budget() {
    let src = big_module();
    let lines = src.lines().count();
    assert!(lines >= 450, "{lines}");
    let c = compile(&src, Stage::Fragment, "fs_big").unwrap();
    if spirv_val_available() {
        validate(&c.words, "big_module").unwrap();
    }
    let runs = 5;
    let start = std::time::Instant::now();
    for _ in 0..runs {
        compile(&src, Stage::Fragment, "fs_big").unwrap();
    }
    let per = start.elapsed() / runs;
    eprintln!(
        "{lines}-line module: {per:?} per compile ({} words)",
        c.words.len()
    );
    // Generous bound so unoptimized test builds on a loaded machine pass; release is ~10x faster.
    assert!(per.as_millis() < 400, "compile took {per:?}");
}
