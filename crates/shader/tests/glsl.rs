//! GLSL ES 3.00 output for WebGL 2: every vertex and fragment entry point of the corpus that
//! uses only what WebGL 2 offers compiles through `compile_glsl` with the same reflection as
//! the SPIR-V path; the others fail with a clear message instead of emitting bad code. Each
//! module is validated with `glslangValidator` when it is installed (skipped with a note
//! otherwise), and the lowering of resources, entry-point IO and loops has the expected shape.
//!
//! Set `GANBATE_GLSL_DUMP=dir` to write each module to `dir/<file>.<entry>.<vert|frag>`.

use std::path::{Path, PathBuf};
use std::process::Command;

use ganbate_shader::glsl::{Glsl, HiddenValue};
use ganbate_shader::{SampleType, Stage, TextureView, compile, compile_glsl, entry_points};

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpus")
}

fn glslang() -> bool {
    Command::new("glslangValidator")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Runs `glslangValidator` on the given (source, stage) modules, linked together when there
/// is more than one.
fn validate(modules: &[(&str, Stage)], tag: &str) -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("ganbate-shader-glsl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut paths = Vec::new();
    for (i, (src, stage)) in modules.iter().enumerate() {
        let ext = if *stage == Stage::Vertex {
            "vert"
        } else {
            "frag"
        };
        let path = dir.join(format!("{}-{i}.{ext}", tag.replace(['/', ':', ' '], "_")));
        std::fs::write(&path, src).map_err(|e| e.to_string())?;
        paths.push(path);
    }
    let mut cmd = Command::new("glslangValidator");
    if paths.len() > 1 {
        cmd.arg("-l");
    }
    let out = cmd.args(&paths).output().map_err(|e| e.to_string())?;
    for p in &paths {
        let _ = std::fs::remove_file(p);
    }
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

fn glsl(src: &str, stage: Stage, entry: &str) -> Glsl {
    compile_glsl(src, stage, entry)
        .unwrap_or_else(|e| panic!("{entry}: {e}\n{src}"))
        .glsl
        .expect("compile_glsl fills Compiled::glsl")
}

#[test]
fn corpus_compiles_to_glsl_or_reports_what_webgl_lacks() {
    let dump = std::env::var_os("GANBATE_GLSL_DUMP").map(PathBuf::from);
    if let Some(d) = &dump {
        std::fs::create_dir_all(d).expect("dump directory");
    }
    let check = glslang();
    if !check {
        eprintln!("note: glslangValidator not found on PATH; the GLSL is not validated");
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(corpus_dir())
        .expect("corpus")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "wgsl"))
        .collect();
    files.sort();
    let (mut modules, mut refused) = (0, 0);
    let mut failures = Vec::new();
    for path in &files {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let src = std::fs::read_to_string(path).unwrap();
        for (entry, stage) in entry_points(&src).unwrap() {
            let spirv = compile(&src, stage, &entry).unwrap();
            let both = match compile_glsl(&src, stage, &entry) {
                Ok(c) => c,
                Err(e) => {
                    assert!(
                        e.message.contains("not available on WebGL 2"),
                        "{name}:{entry}: unexpected error {e}"
                    );
                    refused += 1;
                    continue;
                }
            };
            assert_eq!(
                (spirv.bindings, spirv.push_constant_size, spirv.inputs),
                (both.bindings, both.push_constant_size, both.inputs),
                "{name}:{entry}: reflection differs"
            );
            let g = both.glsl.expect("glsl");
            assert!(g.source.starts_with("#version 300 es\n"), "{name}:{entry}");
            assert!(g.source.contains("void main() {"), "{name}:{entry}");
            if let Some(d) = &dump {
                let ext = if stage == Stage::Vertex {
                    "vert"
                } else {
                    "frag"
                };
                std::fs::write(d.join(format!("{name}.{entry}.{ext}")), &g.source).unwrap();
            }
            if check && let Err(e) = validate(&[(&g.source, stage)], &format!("{name}.{entry}")) {
                failures.push(format!("{name}:{entry}:\n{e}\n{}", g.source));
            }
            modules += 1;
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {modules} modules failed validation:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(modules >= 11, "only {modules} entry points compiled");
    assert!(refused >= 1, "the compute corpus should be refused");
}

#[test]
fn resources_become_blocks_units_and_data_textures() {
    let src = r#"
struct Frame { view: mat4x4f, tint: vec4f };
struct Light { pos: vec3f, range: f32, color: vec4f };
struct Push { model: mat4x4f, flags: vec4u };
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> lights: array<Light>;
@group(0) @binding(2) var shadow: texture_depth_2d_array;
@group(0) @binding(3) var shadow_cmp: sampler_comparison;
@group(1) @binding(0) var albedo: texture_2d<f32>;
@group(1) @binding(1) var linear: sampler;
var<push_constant> push: Push;

struct Out { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) @interpolate(flat) id: u32 };

@vertex fn vs(@location(0) p: vec3f, @builtin(instance_index) ii: u32) -> Out {
  var o: Out;
  o.pos = frame.view * push.model * vec4f(p, 1.0);
  o.uv = p.xy;
  o.id = ii + push.flags.x;
  return o;
}

@fragment fn fs(i: Out) -> @location(0) vec4f {
  var c = textureSample(albedo, linear, i.uv) * frame.tint;
  let lit = textureSampleCompareLevel(shadow, shadow_cmp, i.uv, 1, 0.5);
  let n = arrayLength(&lights);
  for (var k = 0u; k < n; k++) {
    c += lights[k].color * lights[k].range * lit;
  }
  let size = textureDimensions(albedo);
  return c + vec4f(f32(size.x), 0.0, 0.0, 0.0) * 0.0;
}
"#;
    let vs = glsl(src, Stage::Vertex, "vs");
    let fs = glsl(src, Stage::Fragment, "fs");
    if glslang() {
        validate(
            &[(&vs.source, Stage::Vertex), (&fs.source, Stage::Fragment)],
            "resources",
        )
        .unwrap_or_else(|e| panic!("{e}\n{}\n{}", vs.source, fs.source));
    }
    // Uniform buffer and push constants are std140 blocks; the vertex shader reads both.
    let blocks: Vec<_> = vs
        .blocks
        .iter()
        .map(|b| (b.name.as_str(), b.binding))
        .collect();
    assert_eq!(
        blocks,
        [("ganbate_ubo_0_0", Some((0, 0))), ("ganbate_push", None)]
    );
    assert!(
        vs.source
            .contains("layout(std140) uniform ganbate_ubo_0_0 {\n  Frame frame;\n};")
    );
    // Depth remap, flat varying by name, base instance.
    assert!(
        vs.source
            .contains("gl_Position.z = 2.0 * gl_Position.z - gl_Position.w;")
    );
    assert!(vs.source.contains("flat out uint ganbate_v1;"));
    assert!(fs.source.contains("flat in uint ganbate_v1;"));
    assert!(
        vs.source
            .contains("uint(gl_InstanceID + ganbate_base_instance)")
    );
    assert!(
        vs.hidden
            .iter()
            .any(|h| h.value == HiddenValue::BaseInstance)
    );
    // One unit per (texture, sampler) pair, plus an unfiltered unit for textureDimensions.
    let units: Vec<_> = fs
        .units
        .iter()
        .map(|u| (u.name.as_str(), u.view, u.sample, u.compare, u.sampler))
        .collect();
    assert!(units.contains(&(
        "ganbate_tex_1_0_1_1",
        TextureView::D2,
        SampleType::Float,
        false,
        Some((1, 1))
    )));
    assert!(units.contains(&(
        "ganbate_tex_0_2_0_3",
        TextureView::D2Array,
        SampleType::Depth,
        true,
        Some((0, 3))
    )));
    assert!(units.contains(&(
        "ganbate_tex_1_0",
        TextureView::D2,
        SampleType::Float,
        false,
        None
    )));
    assert!(
        fs.source
            .contains("uniform highp sampler2DArrayShadow ganbate_tex_0_2_0_3;")
    );
    assert!(fs.source.contains("textureGrad(ganbate_tex_0_2_0_3, vec4("));
    // The storage buffer is a data texture with its offset and size.
    assert_eq!(fs.storage.len(), 1);
    assert_eq!(fs.storage[0].binding, (0, 1));
    assert!(
        fs.hidden
            .iter()
            .any(|h| h.value == HiddenValue::StorageOffset(0, 1))
    );
    assert!(
        fs.hidden
            .iter()
            .any(|h| h.value == HiddenValue::StorageSize(0, 1))
    );
    assert!(
        fs.source.contains("ganbate_ld_s1(usampler2D s, uint off)")
            || fs.source.contains("ganbate_ld_v4f(")
    );
}

#[test]
fn continuing_blocks_see_the_body_lets_they_read() {
    let src = r#"
@fragment fn fs(@location(0) n: f32) -> @location(0) vec4f {
  var total = 0.0;
  var i = 0;
  loop {
    if (i >= 8) { break; }
    let step = f32(i) * n;
    if (step > 3.0) { i += 2; continue; }
    total += step;
    continuing {
      total += step * 0.5;
      i += 1;
      break if total > 100.0;
    }
  }
  return vec4f(total);
}
"#;
    let g = glsl(src, Stage::Fragment, "fs");
    if glslang() {
        validate(&[(&g.source, Stage::Fragment)], "continuing")
            .unwrap_or_else(|e| panic!("{e}\n{}", g.source));
    }
    // `step` is a GLSL built-in, so the let is renamed.
    let hoisted = g.source.find("float step_ = 0.0;").expect("hoisted let");
    let first = g
        .source
        .find("bool ganbate_first = true;")
        .expect("first-pass flag");
    assert!(hoisted < first, "{}", g.source);
    assert!(g.source.contains("step_ = (float(i) * n);"), "{}", g.source);
    assert!(g.source.contains("if (!ganbate_first) {"), "{}", g.source);
}

#[test]
fn webgl_limits_are_reported() {
    for (src, stage, what) in [
        (
            "@group(0) @binding(0) var<storage, read_write> b: array<u32>;\n\
             @fragment fn f() -> @location(0) vec4f { b[0] = 1u; return vec4f(1.0); }",
            Stage::Fragment,
            "writing a storage buffer",
        ),
        (
            "@group(0) @binding(0) var t: texture_storage_2d<rgba8unorm, write>;\n\
             @fragment fn f() -> @location(0) vec4f { textureStore(t, vec2i(0), vec4f(1.0)); return vec4f(1.0); }",
            Stage::Fragment,
            "storage texture",
        ),
        (
            "@compute @workgroup_size(1) fn c() {}",
            Stage::Compute,
            "compute",
        ),
        (
            "@fragment fn f(@builtin(sample_index) s: u32) -> @location(0) vec4f { return vec4f(f32(s)); }",
            Stage::Fragment,
            "sample_index",
        ),
    ] {
        let entry = entry_points(src).unwrap()[0].0.clone();
        let e = compile_glsl(src, stage, &entry).expect_err(what);
        assert!(
            e.message.contains("not available on WebGL 2"),
            "{what}: {e}"
        );
    }
}

#[test]
fn reserved_names_are_renamed_and_helpers_generated() {
    let src = r#"
fn texture(input: vec2f) -> f32 { return input.x; }
fn mix_it(sample: i32, filter: i32) -> i32 { return sample % filter; }
@fragment fn fs(@location(0) gl_thing: vec2f, @location(1) @interpolate(flat) ganbate_x: vec2i) -> @location(0) vec4f {
  let a = texture(gl_thing) % 0.25;
  let b = f32(mix_it(ganbate_x.x, ganbate_x.y)) + f32(countOneBits(u32(ganbate_x.y))) + f32(dot(ganbate_x, ganbate_x));
  let c = fma(a, b, 1.0) + ldexp(1.0, 2);
  let d = select(vec2i(0), ganbate_x, ganbate_x > vec2i(1));
  let e = (a > 0.0) & (b < 1.0);
  return vec4f(c, f32(d.x), f32(e), unpack4x8unorm(pack4x8unorm(vec4f(0.5))).x);
}
"#;
    let g = glsl(src, Stage::Fragment, "fs");
    if glslang() {
        validate(&[(&g.source, Stage::Fragment)], "reserved")
            .unwrap_or_else(|e| panic!("{e}\n{}", g.source));
    }
    for helper in [
        "ganbate_frem(",
        "ganbate_irem(",
        "ganbate_count(",
        "ganbate_dot(",
        "ganbate_select(",
        "ganbate_and(",
        "ganbate_pack4x8unorm(",
        "ganbate_unpack4x8unorm(",
    ] {
        assert!(g.source.contains(helper), "{helper} missing\n{}", g.source);
    }
    assert!(
        g.source.contains("float texture_(vec2 input_)"),
        "{}",
        g.source
    );
}
