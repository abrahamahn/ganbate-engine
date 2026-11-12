//! A self-test of the parts of the GPU layer the renderer does not use: on WebGPU a compute
//! dispatch with push constants writing a storage texture and a storage buffer; on WebGL 2 (no
//! compute) a render pass with push constants drawing the same image, read back and checked
//! here. Then texture-to-buffer and buffer-to-buffer copies, mip generation and a blit into the
//! canvas texture.
//!
//! The canvas then shows mip 1 of a 64x64 gradient: red grows to the right, green downwards,
//! blue is 0.5 everywhere. `web/test/run.cjs --selftest` checks those pixels and that the
//! browser reported no error.
use ganbate_engine::gpu::{
    Attachment, Blend, BufferDesc, BufferUsage, ColorTarget, Cull, Device, Filter, Format, Memory,
    PassDesc, RenderPipelineDesc, Resource, TextureDesc, TextureUsage, Topology, WebApi,
    compile_shader, f32_bytes, shader::Stage,
};

const SHADER: &str = r#"
struct Push { scale: vec4f };
var<push_constant> push: Push;
@group(0) @binding(0) var image: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> marks: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn fill(@builtin(global_invocation_id) id: vec3u) {
  let uv = vec2f(id.xy) / 63.0;
  textureStore(image, vec2i(id.xy), vec4f(uv * push.scale.xy, push.scale.z, 1.0));
  if (all(id.xy == vec2u(0u))) {
    marks[0] = 0xabe0u;
  }
}
"#;

/// The same gradient as `SHADER`, drawn by a fragment shader (WebGL 2 has no compute).
const DRAW: &str = r#"
struct Push { scale: vec4f };
var<push_constant> push: Push;
struct Out { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: Out;
  o.pos = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
  o.uv = uv;
  return o;
}

@fragment fn fs(o: Out) -> @location(0) vec4f {
  let uv = floor(o.pos.xy) / 63.0;
  return vec4f(uv * push.scale.xy, push.scale.z, 1.0);
}
"#;

pub fn run() -> Result<String, String> {
    let mut d = Device::for_canvas()?;
    if d.web_api() == WebApi::WebGl2 {
        return run_gl(d);
    }
    let shader = compile_shader(SHADER, Stage::Compute, "fill")?;
    let pipeline = d.create_compute_pipeline(&shader)?;
    let image = d.create_texture(&TextureDesc {
        width: 64,
        height: 64,
        layers: 1,
        mips: 7,
        format: Format::Rgba8Unorm,
        usage: TextureUsage::STORAGE
            | TextureUsage::SAMPLED
            | TextureUsage::COPY_SRC
            | TextureUsage::COPY_DST,
    })?;
    let marks = d.create_buffer(&BufferDesc {
        size: 256,
        usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
        memory: Memory::Device,
    })?;
    let copy = d.create_buffer(&BufferDesc {
        size: 256,
        usage: BufferUsage::COPY_DST,
        memory: Memory::Device,
    })?;
    let readback = d.create_buffer(&BufferDesc {
        size: 64 * 64 * 4,
        usage: BufferUsage::COPY_DST,
        memory: Memory::Readback,
    })?;
    let group = d.create_bind_group(
        pipeline,
        0,
        &[
            (0, Resource::TextureMip(image, 0)),
            (
                1,
                Resource::Buffer {
                    id: marks,
                    offset: 0,
                    size: 0,
                },
            ),
        ],
    )?;
    let mut e = d.encoder();
    e.dispatch(
        pipeline,
        &[(0, group)],
        &f32_bytes(&[1.0, 1.0, 0.5, 0.0]),
        8,
        8,
        1,
    );
    e.copy_texture_to_buffer(image, 0, 0, readback, 0);
    e.copy_buffer_to_buffer(marks, 0, copy, 0, 256);
    d.submit(e)?;
    if d.read_buffer(readback).is_ok() {
        return Err("read_buffer should report that it is unavailable".into());
    }
    d.generate_mips(image)?;
    let canvas = d.acquire()?.ok_or("the canvas has no area")?;
    let mut e = d.encoder();
    e.blit((image, 1, 0), (canvas, 0, 0), Filter::Linear);
    d.present(e)?;
    // Misuse is reported at submission, as on Vulkan.
    let mut bad = d.encoder();
    bad.blit((image, 9, 0), (image, 0, 0), Filter::Nearest);
    let misuse = d
        .submit(bad)
        .err()
        .ok_or("an out-of-range blit was accepted")?;
    for b in [marks, copy, readback] {
        d.destroy_buffer(b);
    }
    d.destroy_texture(image);
    Ok(format!(
        "self-test on {}: compute, copies, mips and a canvas blit submitted; misuse reported: {misuse}",
        d.name()
    ))
}

/// The WebGL 2 variant: a render pass instead of the dispatch, and a synchronous readback.
fn run_gl(mut d: Device<'static>) -> Result<String, String> {
    let vs = compile_shader(DRAW, Stage::Vertex, "vs")?;
    let fs = compile_shader(DRAW, Stage::Fragment, "fs")?;
    let pipeline = d.create_render_pipeline(&RenderPipelineDesc {
        vertex: &vs,
        fragment: Some(&fs),
        vertex_layout: None,
        topology: Topology::TriangleList,
        cull: Cull::None,
        front_ccw: true,
        depth: None,
        colors: vec![ColorTarget {
            format: Format::Rgba8Unorm,
            blend: Blend::Replace,
            write_mask: 0xF,
        }],
    })?;
    let image = d.create_texture(&TextureDesc {
        width: 64,
        height: 64,
        layers: 1,
        mips: 7,
        format: Format::Rgba8Unorm,
        usage: TextureUsage::COLOR_TARGET
            | TextureUsage::SAMPLED
            | TextureUsage::COPY_SRC
            | TextureUsage::COPY_DST,
    })?;
    let marks = d.create_buffer(&BufferDesc {
        size: 256,
        usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        memory: Memory::Device,
    })?;
    let copy = d.create_buffer(&BufferDesc {
        size: 256,
        usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        memory: Memory::Device,
    })?;
    let readback = d.create_buffer(&BufferDesc {
        size: 64 * 64 * 4 + 256,
        usage: BufferUsage::COPY_DST,
        memory: Memory::Readback,
    })?;
    d.write_buffer(marks, 0, &0xabe0u32.to_le_bytes())?;
    let mut e = d.encoder();
    e.begin_pass(PassDesc {
        colors: vec![Attachment::clear(image, [0.0; 4])],
        depth: None,
    });
    e.set_pipeline(pipeline);
    e.push_constants(&f32_bytes(&[1.0, 1.0, 0.5, 0.0]));
    e.draw(3, 1, 0, 0);
    e.end_pass();
    e.copy_texture_to_buffer(image, 0, 0, readback, 0);
    e.copy_buffer_to_buffer(marks, 0, copy, 0, 256);
    e.copy_buffer_to_buffer(copy, 0, readback, 64 * 64 * 4, 256);
    d.submit(e)?;
    let bytes = d.read_buffer(readback)?;
    let texel = |x: usize, y: usize| bytes[(y * 64 + x) * 4..(y * 64 + x) * 4 + 4].to_vec();
    // Row 0 is the top: red grows to the right, green downwards.
    let (top_right, bottom_left) = (texel(63, 0), texel(0, 63));
    if top_right[..3] != [255, 0, 128] || bottom_left[..3] != [0, 255, 128] {
        return Err(format!(
            "the drawn gradient reads back wrong: top right {top_right:?}, bottom left {bottom_left:?}"
        ));
    }
    let mark = u32::from_le_bytes(bytes[64 * 64 * 4..64 * 64 * 4 + 4].try_into().unwrap());
    if mark != 0xabe0 {
        return Err(format!(
            "the copied storage buffer holds {mark:#x}, not 0xabe0"
        ));
    }
    d.generate_mips(image)?;
    let canvas = d.acquire()?.ok_or("the canvas has no area")?;
    let mut e = d.encoder();
    e.blit((image, 1, 0), (canvas, 0, 0), Filter::Linear);
    d.present(e)?;
    let mut bad = d.encoder();
    bad.blit((image, 9, 0), (image, 0, 0), Filter::Nearest);
    let misuse = d
        .submit(bad)
        .err()
        .ok_or("an out-of-range blit was accepted")?;
    for b in [marks, copy, readback] {
        d.destroy_buffer(b);
    }
    d.destroy_texture(image);
    Ok(format!(
        "self-test on {}: a drawn gradient read back, copies, mips and a canvas blit; misuse \
         reported: {misuse}",
        d.name()
    ))
}
