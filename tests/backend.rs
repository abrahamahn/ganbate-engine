//! Driver tests of the GPU layer with WGSL shaders compiled by `ganbate-shader`, so the same tests
//! run on every backend: SPIR-V on Vulkan (Linux, Windows) and MSL on Metal (macOS, iOS). A
//! test that passes on Vulkan and fails on Metal points at the Metal backend or the MSL
//! emitter. Run with `cargo test --test backend -- --ignored` (on a Mac with
//! `MTL_DEBUG_LAYER=1`, on Vulkan with `GANBATE_VULKAN_VALIDATION=1`).
#![cfg(any(
    target_os = "linux",
    target_os = "windows",
    target_os = "android",
    target_vendor = "apple"
))]
#[path = "../crates/shader/tests/msl/check.rs"]
mod check;

use ganbate_engine::gpu::shader::{Compiled, Stage, compile, compile_msl, entry_points};
use ganbate_engine::gpu::*;

fn shader(src: &str, stage: Stage, entry: &str) -> Compiled {
    compile(src, stage, entry).unwrap_or_else(|e| panic!("{entry}: {e}"))
}

/// The shaders below as MSL, checked with `clang++` against ganbate-shader's stand-in
/// `metal_stdlib` when it is installed (no GPU needed).
#[test]
fn shaders_emit_msl_that_compiles() {
    let clang = check::clang();
    for (name, source) in [
        ("COMPUTE", COMPUTE),
        ("TRIANGLES", TRIANGLES),
        ("TEXTURES", TEXTURES),
        ("DEPTH", DEPTH),
        ("VERTICES", VERTICES),
    ] {
        for (entry, stage) in entry_points(source).unwrap() {
            let tag = format!("{name}:{entry}");
            let msl = compile_msl(source, stage, &entry)
                .unwrap_or_else(|e| panic!("{tag}: {e}"))
                .msl
                .expect("compile_msl fills msl");
            if let Some(clang) = &clang {
                check::clang_check(clang, &tag, &msl.source)
                    .unwrap_or_else(|e| panic!("{tag}: clang++ rejected the MSL:\n{e}"));
            }
        }
    }
}

fn readback(device: &mut Device, size: u64) -> BufferId {
    device
        .create_buffer(&BufferDesc {
            size,
            usage: BufferUsage::COPY_DST,
            memory: Memory::Readback,
        })
        .expect("readback buffer")
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn check_validation(device: &Device) {
    let errors = device.validation_errors();
    assert!(errors.is_empty(), "validation: {errors:?}");
}

/// One RGBA8 pixel of a tightly packed `width`-wide image.
fn pixel(image: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * width + x) * 4) as usize;
    image[i..i + 4].try_into().unwrap()
}

fn near(a: [u8; 4], b: [u8; 4], tolerance: u8) -> bool {
    a.iter().zip(b).all(|(x, y)| x.abs_diff(y) <= tolerance)
}

const COMPUTE: &str = "
struct Params {
  a: vec3f,
  b: f32,
  c: vec2f,
  d: array<vec4f, 2>,
  m: mat4x4f,
};
struct Push {
  scale: f32,
  offset: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
var<push_constant> push: Push;

@compute @workgroup_size(8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = arrayLength(&input);
  if (i >= arrayLength(&output)) {
    return;
  }
  if (i < n) {
    output[i] = input[i] * push.scale + f32(push.offset);
    return;
  }
  var report = array<f32, 10>(
    params.a.x, params.a.y, params.a.z, params.b, params.c.x, params.c.y,
    params.d[1].w, params.m[3].x, f32(n), f32(arrayLength(&output))
  );
  output[i] = report[i - n];
}
";

/// Uniform layout (vec3 + f32 packing, array and matrix strides), a uniform bound at a
/// non-zero offset, push constants, runtime-sized arrays bound with and without an explicit
/// size, a staged upload, a buffer copy and readback.
#[test]
#[ignore = "requires a GPU; run cargo test --test backend -- --ignored"]
fn compute_reads_uniforms_push_constants_and_runtime_arrays() {
    let mut device = Device::headless().expect("device");
    let pipeline = device
        .create_compute_pipeline(&shader(COMPUTE, Stage::Compute, "main"))
        .expect("compute pipeline");

    let align = device.limits().uniform_offset_alignment.max(16);
    let uniform = device
        .create_buffer(&BufferDesc {
            size: align + 128,
            usage: BufferUsage::UNIFORM,
            memory: Memory::Upload,
        })
        .unwrap();
    let mut params = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0];
    params.extend((7..15).map(|v| v as f32));
    let mut m = [0.0f32; 16];
    for c in 0..4 {
        m[c * 4 + c] = 1.0;
    }
    m[12..15].copy_from_slice(&[15.0, 16.0, 17.0]);
    params.extend(m);
    assert_eq!(params.len() * 4, 128);
    device
        .write_buffer(uniform, align, &f32_bytes(&params))
        .unwrap();

    // 32 floats, of which the binding exposes 16.
    let input = device
        .create_buffer(&BufferDesc {
            size: 128,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
            memory: Memory::Device,
        })
        .unwrap();
    let values: Vec<f32> = (0..32).map(|v| v as f32).collect();
    device.write_buffer(input, 0, &f32_bytes(&values)).unwrap();
    let count = 26u64;
    let output = device
        .create_buffer(&BufferDesc {
            size: count * 4,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: Memory::Device,
        })
        .unwrap();
    let group = device
        .create_bind_group(
            pipeline,
            0,
            &[
                (
                    0,
                    Resource::Buffer {
                        id: uniform,
                        offset: align,
                        size: 128,
                    },
                ),
                (
                    1,
                    Resource::Buffer {
                        id: input,
                        offset: 0,
                        size: 64,
                    },
                ),
                (
                    2,
                    Resource::Buffer {
                        id: output,
                        offset: 0,
                        size: 0,
                    },
                ),
            ],
        )
        .expect("bind group");
    let out = readback(&mut device, count * 4);
    let mut push = f32_bytes(&[2.0]);
    push.extend(u32_bytes(&[3]));
    let mut e = device.encoder();
    e.dispatch(pipeline, &[(0, group)], &push, 4, 1, 1);
    e.copy_buffer_to_buffer(output, 0, out, 0, count * 4);
    device.submit(e).expect("submit");
    let got = floats(&device.read_buffer(out).unwrap());
    let mut want: Vec<f32> = (0..16).map(|v| v as f32 * 2.0 + 3.0).collect();
    want.extend([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 14.0, 15.0, 16.0, 26.0]);
    assert_eq!(got, want);
    check_validation(&device);
}

const TRIANGLES: &str = "
struct VOut {
  @builtin(position) pos: vec4f,
};
@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
  // Clip space has y down. The first triangle (top-left, bottom-left, bottom-right) runs
  // counter-clockwise on screen and covers the bottom-left half; the second (top-left,
  // top-right, bottom-right) runs clockwise and covers the top-right half.
  let p = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  var o: VOut;
  o.pos = vec4f(p[i], 0.5, 1.0);
  return o;
}
@fragment fn fs(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  return vec4f(frag.x / 64.0, frag.y / 64.0, frag.z, 1.0);
}
";

/// Clip-space orientation (y down, as on Vulkan), front-face winding with back-face culling,
/// fragment coordinates (pixel centres from the top-left) and fragment depth.
#[test]
#[ignore = "requires a GPU; run cargo test --test backend -- --ignored"]
fn triangles_are_upright_and_culled_by_winding() {
    let mut device = Device::headless().expect("device");
    let vs = shader(TRIANGLES, Stage::Vertex, "vs");
    let fs = shader(TRIANGLES, Stage::Fragment, "fs");
    let target = device
        .create_texture(&TextureDesc::new(
            64,
            64,
            Format::Rgba8Unorm,
            TextureUsage::COLOR_TARGET | TextureUsage::COPY_SRC,
        ))
        .unwrap();
    let out = readback(&mut device, 64 * 64 * 4);
    for front_ccw in [true, false] {
        let pipeline = device
            .create_render_pipeline(&RenderPipelineDesc {
                vertex: &vs,
                fragment: Some(&fs),
                vertex_layout: None,
                topology: Topology::TriangleList,
                cull: Cull::Back,
                front_ccw,
                depth: None,
                colors: vec![ColorTarget {
                    format: Format::Rgba8Unorm,
                    blend: Blend::Replace,
                    write_mask: 0xF,
                }],
            })
            .expect("render pipeline");
        let mut e = device.encoder();
        e.begin_pass(PassDesc {
            colors: vec![Attachment::clear(target, [0.0; 4])],
            depth: None,
        });
        e.set_pipeline(pipeline);
        e.draw(6, 1, 0, 0);
        e.end_pass();
        e.copy_texture_to_buffer(target, 0, 0, out, 0);
        device.submit(e).expect("submit");
        let image = device.read_buffer(out).unwrap();
        let bottom_left = pixel(&image, 64, 16, 48);
        let top_right = pixel(&image, 64, 48, 16);
        // (x + 0.5) / 64 and (y + 0.5) / 64 in the red and green channels, depth 0.5 in blue.
        let shade = |x: u32, y: u32| {
            let c = |v: u32| ((v as f32 + 0.5) / 64.0 * 255.0).round() as u8;
            [c(x), c(y), 128, 255]
        };
        let (drawn, culled, at) = if front_ccw {
            (bottom_left, top_right, (16, 48))
        } else {
            (top_right, bottom_left, (48, 16))
        };
        assert!(
            near(drawn, shade(at.0, at.1), 2),
            "front_ccw {front_ccw}: the front-facing triangle at {at:?} is {drawn:?}"
        );
        assert_eq!(
            culled, [0; 4],
            "front_ccw {front_ccw}: the back-facing triangle was not culled"
        );
    }
    check_validation(&device);
}

const TEXTURES: &str = "
@group(0) @binding(0) var color: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var depth: texture_depth_2d;
@group(0) @binding(3) var cmp: sampler_comparison;
@group(0) @binding(4) var<storage, read_write> out: array<vec4f>;

@compute @workgroup_size(1)
fn main() {
  out[0] = textureSampleLevel(color, smp, vec2f(0.5, 0.5), 2.0);
  out[1] = textureLoad(color, vec2i(3, 0), 0);
  let size = textureDimensions(color, 1);
  out[2] = vec4f(f32(size.x), f32(size.y), f32(textureNumLevels(color)), 0.0);
  out[3] = vec4f(
    textureSampleCompareLevel(depth, cmp, vec2f(0.25, 0.5), 0.6),
    textureSampleCompareLevel(depth, cmp, vec2f(0.75, 0.5), 0.6),
    textureLoad(depth, vec2i(1, 2), 0),
    textureLoad(depth, vec2i(6, 2), 0)
  );
}
";

const DEPTH: &str = "
struct Out {
  @builtin(position) pos: vec4f,
};
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: Out;
  o.pos = vec4f(uv * 2.0 - 1.0, 0.75, 1.0);
  return o;
}
";

/// Texture upload, mip generation, sampling at an explicit level, texel loads, dimensions of
/// a mip, a depth-only pass with a viewport, and comparison sampling of the depth it wrote.
#[test]
#[ignore = "requires a GPU; run cargo test --test backend -- --ignored"]
fn textures_sample_mips_and_compare_depth() {
    let mut device = Device::headless().expect("device");
    let compute = device
        .create_compute_pipeline(&shader(TEXTURES, Stage::Compute, "main"))
        .expect("compute pipeline");
    let depth_only = device
        .create_render_pipeline(&RenderPipelineDesc {
            vertex: &shader(DEPTH, Stage::Vertex, "vs"),
            fragment: None,
            vertex_layout: None,
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: Some(DepthState {
                format: Format::Depth32Float,
                compare: Compare::Always,
                write: true,
                bias_constant: 0.0,
                bias_slope: 0.0,
                clamp: false,
            }),
            colors: vec![],
        })
        .expect("depth pipeline");

    // 4x4, left half red and right half blue: the 1x1 mip is their average.
    let color = device
        .create_texture(&TextureDesc {
            mips: 3,
            ..TextureDesc::new(
                4,
                4,
                Format::Rgba8Unorm,
                TextureUsage::SAMPLED | TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            )
        })
        .unwrap();
    let mut texels = Vec::new();
    for _ in 0..4 {
        for x in 0..4 {
            texels.extend(if x < 2 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            });
        }
    }
    device.write_texture(color, 0, 0, &texels).unwrap();
    device.generate_mips(color).unwrap();
    let depth = device
        .create_texture(&TextureDesc::new(
            8,
            4,
            Format::Depth32Float,
            TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED,
        ))
        .unwrap();
    let smp = device
        .create_sampler(&SamplerDesc {
            address: Address::Clamp,
            ..SamplerDesc::default()
        })
        .unwrap();
    let cmp = device
        .create_sampler(&SamplerDesc {
            address: Address::Clamp,
            mip_filter: Filter::Nearest,
            compare: Some(Compare::Less),
            ..SamplerDesc::default()
        })
        .unwrap();
    let results = device
        .create_buffer(&BufferDesc {
            size: 64,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: Memory::Device,
        })
        .unwrap();
    let group = device
        .create_bind_group(
            compute,
            0,
            &[
                (0, Resource::Texture(color)),
                (1, Resource::Sampler(smp)),
                (2, Resource::Texture(depth)),
                (3, Resource::Sampler(cmp)),
                (
                    4,
                    Resource::Buffer {
                        id: results,
                        offset: 0,
                        size: 0,
                    },
                ),
            ],
        )
        .expect("bind group");
    let out = readback(&mut device, 64);

    let mut e = device.encoder();
    // Depth 0.75 in the left half (through the viewport), the clear value 0.25 elsewhere.
    e.begin_pass(PassDesc {
        colors: vec![],
        depth: Some(Attachment::clear(depth, [0.25, 0.0, 0.0, 0.0])),
    });
    e.set_pipeline(depth_only);
    e.set_viewport(0.0, 0.0, 4.0, 4.0);
    e.draw(3, 1, 0, 0);
    e.end_pass();
    e.dispatch(compute, &[(0, group)], &[], 1, 1, 1);
    e.copy_buffer_to_buffer(results, 0, out, 0, 64);
    device.submit(e).expect("submit");
    let got = floats(&device.read_buffer(out).unwrap());
    let close = |a: &[f32], b: &[f32], t: f32| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= t);
    assert!(
        close(&got[0..4], &[0.5, 0.0, 0.5, 1.0], 0.02),
        "1x1 mip {:?}",
        &got[0..4]
    );
    assert_eq!(&got[4..8], &[0.0, 0.0, 1.0, 1.0], "texel (3, 0) of mip 0");
    assert_eq!(
        &got[8..12],
        &[2.0, 2.0, 3.0, 0.0],
        "mip 1 size and mip count"
    );
    assert!(
        close(&got[12..16], &[1.0, 0.0, 0.75, 0.25], 1e-6),
        "depth compare (0.6 < stored) left and right, then stored depth left and right: {:?}",
        &got[12..16]
    );
    check_validation(&device);
}

/// A same-size blit (a plain copy) and a scaled blit (drawn), both read back exactly.
#[test]
#[ignore = "requires a GPU; run cargo test --test backend -- --ignored"]
fn blits_copy_and_scale() {
    let mut device = Device::headless().expect("device");
    let usage = TextureUsage::SAMPLED | TextureUsage::COPY_SRC | TextureUsage::COPY_DST;
    let src = device
        .create_texture(&TextureDesc::new(4, 4, Format::Rgba8Unorm, usage))
        .unwrap();
    let mut texels = Vec::new();
    for y in 0..4u8 {
        for x in 0..4u8 {
            texels.extend([x * 60, y * 60, 7, 255]);
        }
    }
    device.write_texture(src, 0, 0, &texels).unwrap();
    let same = device
        .create_texture(&TextureDesc::new(4, 4, Format::Rgba8Unorm, usage))
        .unwrap();
    let big = device
        .create_texture(&TextureDesc::new(8, 8, Format::Rgba8Unorm, usage))
        .unwrap();
    let out = readback(&mut device, 64 + 256);
    let mut e = device.encoder();
    e.blit((src, 0, 0), (same, 0, 0), Filter::Linear);
    e.blit((src, 0, 0), (big, 0, 0), Filter::Nearest);
    e.copy_texture_to_buffer(same, 0, 0, out, 0);
    e.copy_texture_to_buffer(big, 0, 0, out, 64);
    device.submit(e).expect("submit");
    let image = device.read_buffer(out).unwrap();
    assert_eq!(&image[..64], &texels[..], "same-size blit");
    for y in 0..8 {
        for x in 0..8 {
            let (got, want) = (
                pixel(&image[64..], 8, x, y),
                pixel(&texels, 4, x / 2, y / 2),
            );
            assert!(
                near(got, want, 1),
                "scaled blit at ({x}, {y}): {got:?} for {want:?}"
            );
        }
    }
    check_validation(&device);
}

const VERTICES: &str = "
struct VIn {
  @location(0) pos: vec2f,
  @location(1) color: vec4f,
};
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec4f,
};
@vertex fn vs(v: VIn, @builtin(instance_index) inst: u32) -> VOut {
  var o: VOut;
  o.pos = vec4f(v.pos.x - f32(inst), v.pos.y, 0.0, 1.0);
  o.color = v.color;
  if (inst == 1u) {
    o.color = vec4f(0.0, 0.0, 1.0, 1.0);
  }
  return o;
}
@fragment fn fs(v: VOut) -> @location(0) vec4f {
  return v.color;
}
";

/// A vertex buffer with float and normalized-byte attributes, an index buffer read from a
/// first index with a base vertex, and the first instance reaching `instance_index`.
#[test]
#[ignore = "requires a GPU; run cargo test --test backend -- --ignored"]
fn indexed_draws_read_vertex_attributes() {
    let mut device = Device::headless().expect("device");
    let pipeline = device
        .create_render_pipeline(&RenderPipelineDesc {
            vertex: &shader(VERTICES, Stage::Vertex, "vs"),
            fragment: Some(&shader(VERTICES, Stage::Fragment, "fs")),
            vertex_layout: Some(VertexLayout {
                stride: 12,
                attributes: vec![
                    VertexAttribute {
                        location: 0,
                        format: VertexFormat::F32x2,
                        offset: 0,
                    },
                    VertexAttribute {
                        location: 1,
                        format: VertexFormat::Unorm8x4,
                        offset: 8,
                    },
                ],
            }),
            topology: Topology::TriangleList,
            cull: Cull::None,
            front_ccw: true,
            depth: None,
            colors: vec![ColorTarget {
                format: Format::Rgba8Unorm,
                blend: Blend::Replace,
                write_mask: 0xF,
            }],
        })
        .expect("render pipeline");
    // Two unused vertices, then the right half of the screen: red at the top, green at the
    // bottom (clip space has y down).
    let mut vertices = Vec::new();
    for (x, y, rgba) in [
        (9.0f32, 9.0f32, [9u8, 9, 9, 9]),
        (9.0, 9.0, [9, 9, 9, 9]),
        (0.0, -1.0, [255, 0, 0, 255]),
        (1.0, -1.0, [255, 0, 0, 255]),
        (0.0, 1.0, [0, 255, 0, 255]),
        (1.0, 1.0, [0, 255, 0, 255]),
    ] {
        vertices.extend(f32_bytes(&[x, y]));
        vertices.extend(rgba);
    }
    let vb = device
        .create_buffer(&BufferDesc {
            size: vertices.len() as u64,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
            memory: Memory::Device,
        })
        .unwrap();
    device.write_buffer(vb, 0, &vertices).unwrap();
    let indices = u32_bytes(&[77, 77, 0, 1, 2, 2, 1, 3]);
    let ib = device
        .create_buffer(&BufferDesc {
            size: indices.len() as u64,
            usage: BufferUsage::INDEX,
            memory: Memory::Upload,
        })
        .unwrap();
    device.write_buffer(ib, 0, &indices).unwrap();
    let target = device
        .create_texture(&TextureDesc::new(
            64,
            64,
            Format::Rgba8Unorm,
            TextureUsage::COLOR_TARGET | TextureUsage::COPY_SRC,
        ))
        .unwrap();
    let out = readback(&mut device, 64 * 64 * 4);
    let mut e = device.encoder();
    e.begin_pass(PassDesc {
        colors: vec![Attachment::clear(target, [0.0; 4])],
        depth: None,
    });
    e.set_pipeline(pipeline);
    e.set_vertex_buffer(vb, 0);
    e.set_index_buffer(ib, 0);
    // Instance 0 draws the right half; first instance 1 shifts it left and paints it blue.
    e.draw_indexed(6, 1, 2, 2, 0);
    e.draw_indexed(6, 1, 2, 2, 1);
    e.end_pass();
    e.copy_texture_to_buffer(target, 0, 0, out, 0);
    device.submit(e).expect("submit");
    let image = device.read_buffer(out).unwrap();
    let top = pixel(&image, 64, 48, 1);
    let bottom = pixel(&image, 64, 48, 62);
    assert!(
        top[0] > 230 && top[1] < 25,
        "top right should be red: {top:?}"
    );
    assert!(
        bottom[1] > 230 && bottom[0] < 25,
        "bottom right should be green: {bottom:?}"
    );
    for y in [1, 32, 62] {
        let left = pixel(&image, 64, 16, y);
        assert!(
            near(left, [0, 0, 255, 255], 1),
            "left half (instance 1) at y {y} should be blue: {left:?}"
        );
    }
    check_validation(&device);
}
