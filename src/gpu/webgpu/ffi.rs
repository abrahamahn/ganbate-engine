//! The browser's WebGPU, reached through imports that the JavaScript host (`web/ganbate.js`)
//! provides in the wasm import module `ganbate`. This is the only `unsafe` code of the backend.
//!
//! Objects: every GPU object lives in a JavaScript table and Rust holds its `u32` index. Index
//! 0 is never a valid object: a create call returns 0 on failure and [`last_error`] explains
//! why. Handles at and above [`SPECIAL`] name objects the host owns (the canvas texture, the
//! shared empty bind group, the push-constant ring's bind groups and layouts).
//!
//! Lifetime: an object stays alive until [`release`]. Releasing a buffer or texture destroys it;
//! WebGPU defers the destruction until submitted work that uses it has finished, so releasing
//! right after a submission is safe. Bind groups, layouts, samplers and pipelines are dropped
//! from the table and collected by the browser.
//!
//! Synchronization: wasm runs on one thread. Each import runs synchronously, reads its pointer
//! arguments only during the call, keeps no reference to wasm memory and never calls back into
//! wasm, so memory cannot grow (detaching the host's view of it) while an import reads it.
//! Queue writes and submissions execute in call order on the GPU queue.
//!
//! Host images: [`copy_source`] copies the host's *current source*, an image the host page set
//! (a canvas, an image bitmap) before calling the runtime export that leads here. The source is
//! a host slot, not a handle: it is valid only during that one export call, and the host clears it
//! when the call returns.

/// First handle value reserved for objects owned by the host.
pub const SPECIAL: u32 = 0xFFFF_FF00;
/// The current canvas texture (`context.getCurrentTexture()` at replay time).
pub const SURFACE: u32 = 0xFFFF_FFFF;
/// The host's bind group with an empty layout, bound at unused group slots.
pub const EMPTY_GROUP: u32 = 0xFFFF_FFFE;
/// The push-constant ring's bind groups (and, in pipeline layouts, their layouts): a dynamic
/// uniform buffer at binding 0, visible to the vertex and fragment stages or to compute.
pub const PUSH_RENDER: u32 = 0xFFFF_FFFD;
pub const PUSH_COMPUTE: u32 = 0xFFFF_FFFC;

#[link(wasm_import_module = "ganbate")]
unsafe extern "C" {
    fn gpu_info(out: *mut u32, cap: u32) -> u32;
    fn gpu_adapter_name(out: *mut u8, cap: u32) -> u32;
    fn gpu_last_error(out: *mut u8, cap: u32) -> u32;
    fn gpu_error_count() -> u32;
    fn gpu_error(index: u32, out: *mut u8, cap: u32) -> u32;
    fn gpu_canvas(out: *mut u32) -> u32;
    fn gpu_create_buffer(size: f64, usage: u32) -> u32;
    fn gpu_write_buffer(buffer: u32, offset: f64, data: *const u8, len: usize);
    fn gpu_create_texture(
        width: u32,
        height: u32,
        layers: u32,
        mips: u32,
        format: u32,
        usage: u32,
    ) -> u32;
    fn gpu_write_texture(
        texture: u32,
        mip: u32,
        layer: u32,
        width: u32,
        height: u32,
        bytes_per_row: u32,
        data: *const u8,
        len: usize,
    );
    fn gpu_copy_source(texture: u32, mip: u32, layer: u32, width: u32, height: u32) -> u32;
    fn gpu_generate_mips(texture: u32);
    fn gpu_create_sampler(desc: *const u32, words: u32) -> u32;
    fn gpu_create_shader(code: *const u8, len: usize) -> u32;
    fn gpu_create_bind_group_layout(entries: *const u32, words: u32) -> u32;
    fn gpu_create_pipeline_layout(layouts: *const u32, count: u32) -> u32;
    fn gpu_create_render_pipeline(json: *const u8, len: usize) -> u32;
    fn gpu_create_compute_pipeline(json: *const u8, len: usize) -> u32;
    fn gpu_create_bind_group(layout: u32, entries: *const u32, words: u32) -> u32;
    fn gpu_submit(commands: *const u32, words: u32, push: *const u8, push_len: usize) -> u32;
    fn gpu_release(handle: u32);
}

fn text(read: impl FnOnce(*mut u8, u32) -> u32) -> String {
    let mut buf = vec![0u8; 4096];
    let n = read(buf.as_mut_ptr(), buf.len() as u32) as usize;
    buf.truncate(n.min(4096));
    String::from_utf8_lossy(&buf).into_owned()
}

/// Why the last create call returned 0.
pub fn last_error() -> String {
    // SAFETY: the host writes at most `cap` bytes into the buffer during the call.
    text(|p, cap| unsafe { gpu_last_error(p, cap) })
}

/// A created object, or the host's error message.
fn made(handle: u32, what: &str) -> Result<u32, String> {
    if handle == 0 {
        Err(format!("{what}: {}", last_error()))
    } else {
        Ok(handle)
    }
}

/// Device limits and features as `[words...]` (see `Gpu::open` for the layout).
pub fn info() -> Vec<u32> {
    let mut out = vec![0u32; 32];
    // SAFETY: the host writes at most `cap` words into the buffer during the call.
    let n = unsafe { gpu_info(out.as_mut_ptr(), out.len() as u32) } as usize;
    out.truncate(n.min(32));
    out
}

pub fn adapter_name() -> String {
    // SAFETY: as in `last_error`.
    text(|p, cap| unsafe { gpu_adapter_name(p, cap) })
}

/// WebGPU validation errors, lost-device reports and shader compilation errors so far.
pub fn errors() -> Vec<String> {
    // SAFETY: a plain query.
    let n = unsafe { gpu_error_count() };
    (0..n)
        .map(|i| {
            // SAFETY: as in `last_error`.
            text(|p, cap| unsafe { gpu_error(i, p, cap) })
        })
        .collect()
}

/// `(width, height, format code)` of the canvas in device pixels, if there is one.
pub fn canvas() -> Option<(u32, u32, u32)> {
    let mut out = [0u32; 3];
    // SAFETY: the host writes exactly three words during the call.
    let ok = unsafe { gpu_canvas(out.as_mut_ptr()) };
    (ok != 0).then_some((out[0], out[1], out[2]))
}

pub fn create_buffer(size: u64, usage: u32) -> Result<u32, String> {
    // SAFETY: plain values.
    made(
        unsafe { gpu_create_buffer(size as f64, usage) },
        "createBuffer",
    )
}

pub fn write_buffer(buffer: u32, offset: u64, data: &[u8]) {
    // SAFETY: the host copies `data` (queue.writeBuffer) during the call.
    unsafe { gpu_write_buffer(buffer, offset as f64, data.as_ptr(), data.len()) }
}

pub fn create_texture(
    size: (u32, u32),
    layers: u32,
    mips: u32,
    format: u32,
    usage: u32,
) -> Result<u32, String> {
    // SAFETY: plain values.
    let h = unsafe { gpu_create_texture(size.0, size.1, layers, mips, format, usage) };
    made(h, "createTexture")
}

pub fn write_texture(
    texture: u32,
    (mip, layer): (u32, u32),
    (width, height): (u32, u32),
    bytes_per_row: u32,
    data: &[u8],
) {
    // SAFETY: the host copies `data` (queue.writeTexture) during the call.
    unsafe {
        gpu_write_texture(
            texture,
            mip,
            layer,
            width,
            height,
            bytes_per_row,
            data.as_ptr(),
            data.len(),
        )
    }
}

/// Copies the host's current source (`width` x `height`, straight alpha, sRGB-encoded) into one
/// whole subresource (`copyExternalImageToTexture`). Returns the host's error if there is no
/// source, its size differs or the browser refused it.
pub fn copy_source(
    texture: u32,
    (mip, layer): (u32, u32),
    (width, height): (u32, u32),
) -> Result<(), String> {
    // SAFETY: plain values; the host reads only its own source slot.
    let ok = unsafe { gpu_copy_source(texture, mip, layer, width, height) };
    if ok == 0 {
        Err(format!("texture from source: {}", last_error()))
    } else {
        Ok(())
    }
}

pub fn generate_mips(texture: u32) {
    // SAFETY: plain value.
    unsafe { gpu_generate_mips(texture) }
}

pub fn create_sampler(desc: &[u32]) -> Result<u32, String> {
    // SAFETY: the host reads `desc` during the call.
    let h = unsafe { gpu_create_sampler(desc.as_ptr(), desc.len() as u32) };
    made(h, "createSampler")
}

pub fn create_shader(code: &str) -> Result<u32, String> {
    // SAFETY: the host reads the UTF-8 text during the call.
    let h = unsafe { gpu_create_shader(code.as_ptr(), code.len()) };
    made(h, "createShaderModule")
}

pub fn create_bind_group_layout(entries: &[u32]) -> Result<u32, String> {
    // SAFETY: the host reads `entries` during the call.
    let h = unsafe { gpu_create_bind_group_layout(entries.as_ptr(), entries.len() as u32) };
    made(h, "createBindGroupLayout")
}

pub fn create_pipeline_layout(layouts: &[u32]) -> Result<u32, String> {
    // SAFETY: the host reads `layouts` during the call.
    let h = unsafe { gpu_create_pipeline_layout(layouts.as_ptr(), layouts.len() as u32) };
    made(h, "createPipelineLayout")
}

pub fn create_render_pipeline(json: &str) -> Result<u32, String> {
    // SAFETY: the host reads the UTF-8 text during the call.
    let h = unsafe { gpu_create_render_pipeline(json.as_ptr(), json.len()) };
    made(h, "createRenderPipeline")
}

pub fn create_compute_pipeline(json: &str) -> Result<u32, String> {
    // SAFETY: the host reads the UTF-8 text during the call.
    let h = unsafe { gpu_create_compute_pipeline(json.as_ptr(), json.len()) };
    made(h, "createComputePipeline")
}

pub fn create_bind_group(layout: u32, entries: &[u32]) -> Result<u32, String> {
    // SAFETY: the host reads `entries` during the call.
    let h = unsafe { gpu_create_bind_group(layout, entries.as_ptr(), entries.len() as u32) };
    made(h, "createBindGroup")
}

/// Writes `push` to the push-constant ring, records `commands` into one command buffer and
/// submits it. Returns the host's error if recording threw.
pub fn submit(commands: &[u32], push: &[u8]) -> Result<(), String> {
    // SAFETY: the host reads both slices during the call.
    let ok = unsafe {
        gpu_submit(
            commands.as_ptr(),
            commands.len() as u32,
            push.as_ptr(),
            push.len(),
        )
    };
    if ok == 0 {
        Err(format!("submit: {}", last_error()))
    } else {
        Ok(())
    }
}

pub fn release(handle: u32) {
    if handle != 0 && handle < SPECIAL {
        // SAFETY: plain value; the host ignores unknown handles.
        unsafe { gpu_release(handle) }
    }
}
