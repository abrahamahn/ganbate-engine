//! The browser's WebGL 2, reached through imports that the JavaScript host (`web/ganbate.js`)
//! provides in the wasm import module `ganbate`. This is the only `unsafe` code of the backend.
//!
//! Objects: every GL object (and every pipeline: a program plus fixed-function state and a
//! vertex layout) lives in a JavaScript table and Rust holds its `u32` index. Index 0 is never
//! valid: a create call returns 0 on failure and [`last_error`] explains why. [`SURFACE`] names
//! the host's canvas texture.
//!
//! Lifetime: an object stays alive until [`release`]. GL defers deleting objects that queued
//! commands still use, so releasing right after a submission is safe.
//!
//! Synchronization: wasm runs on one thread. Each import runs synchronously, reads its pointer
//! arguments only during the call, keeps no reference to wasm memory and never calls back into
//! wasm, so memory cannot grow (detaching the host's view of it) while an import reads it. GL
//! executes calls in order, so writes made after a submission never change what it read.
//!
//! Host images: [`write_texture_source`] uploads the host's *current source*, an image the host
//! page set (a canvas, an image bitmap) before calling the runtime export that leads here. The
//! source is a host slot, not a handle: it is valid only during that one export call, and the host
//! clears it when the call returns.

/// The canvas texture: an RGBA8 texture the host keeps at the canvas size and copies to the
/// canvas (flipped: GL's window origin is the bottom-left) when a frame is presented.
pub const SURFACE: u32 = 0xFFFF_FFFF;

/// Buffer kinds (`gl_create_buffer`).
pub const BUFFER_DATA: u32 = 1;
pub const BUFFER_INDEX: u32 = 2;
/// Also (or only) mirrored into an RGBA32UI data texture that shaders read storage through.
pub const BUFFER_STORAGE: u32 = 4;
/// Host memory only, for readback.
pub const BUFFER_READBACK: u32 = 8;

#[link(wasm_import_module = "ganbate")]
unsafe extern "C" {
    fn gl_info(out: *mut u32, cap: u32) -> u32;
    fn gl_renderer_name(out: *mut u8, cap: u32) -> u32;
    fn gl_last_error(out: *mut u8, cap: u32) -> u32;
    fn gl_error_count() -> u32;
    fn gl_error(index: u32, out: *mut u8, cap: u32) -> u32;
    fn gl_canvas(out: *mut u32) -> u32;
    fn gl_create_buffer(size: f64, kinds: u32) -> u32;
    fn gl_write_buffer(buffer: u32, offset: f64, data: *const u8, len: usize);
    fn gl_read_buffer(buffer: u32, out: *mut u8, len: usize) -> u32;
    fn gl_create_texture(
        width: u32,
        height: u32,
        layers: u32,
        mips: u32,
        format: u32,
        array: u32,
    ) -> u32;
    fn gl_write_texture(
        texture: u32,
        mip: u32,
        layer: u32,
        width: u32,
        height: u32,
        data: *const u8,
        len: usize,
    );
    fn gl_write_texture_source(texture: u32, mip: u32, layer: u32, width: u32, height: u32) -> u32;
    fn gl_generate_mips(texture: u32);
    fn gl_create_sampler(desc: *const u32, words: u32) -> u32;
    fn gl_create_program(
        vs: *const u8,
        vs_len: usize,
        fs: *const u8,
        fs_len: usize,
        meta: *const u8,
        meta_len: usize,
    ) -> u32;
    fn gl_create_pipeline(program: u32, state: *const u32, words: u32) -> u32;
    fn gl_create_vao(pipeline: u32, buffer: u32, offset: u32, index: u32) -> u32;
    fn gl_submit(
        commands: *const u32,
        words: u32,
        push: *const u8,
        push_len: usize,
        present: u32,
    ) -> u32;
    fn gl_release(handle: u32);
}

fn text(read: impl FnOnce(*mut u8, u32) -> u32) -> String {
    let mut buf = vec![0u8; 8192];
    let n = read(buf.as_mut_ptr(), buf.len() as u32) as usize;
    buf.truncate(n.min(8192));
    String::from_utf8_lossy(&buf).into_owned()
}

/// Why the last create call returned 0.
pub fn last_error() -> String {
    // SAFETY: the host writes at most `cap` bytes into the buffer during the call.
    text(|p, cap| unsafe { gl_last_error(p, cap) })
}

fn made(handle: u32, what: &str) -> Result<u32, String> {
    if handle == 0 {
        Err(format!("{what}: {}", last_error()))
    } else {
        Ok(handle)
    }
}

/// Context limits and extensions as `[words...]` (see `Gpu::open` for the layout).
pub fn info() -> Vec<u32> {
    let mut out = vec![0u32; 32];
    // SAFETY: the host writes at most `cap` words into the buffer during the call.
    let n = unsafe { gl_info(out.as_mut_ptr(), out.len() as u32) } as usize;
    out.truncate(n.min(32));
    out
}

pub fn renderer_name() -> String {
    // SAFETY: as in `last_error`.
    text(|p, cap| unsafe { gl_renderer_name(p, cap) })
}

/// GL errors, shader compile/link logs and context loss reported so far.
pub fn errors() -> Vec<String> {
    // SAFETY: a plain query.
    let n = unsafe { gl_error_count() };
    (0..n)
        .map(|i| {
            // SAFETY: as in `last_error`.
            text(|p, cap| unsafe { gl_error(i, p, cap) })
        })
        .collect()
}

/// `(width, height)` of the canvas in device pixels, if there is one; also resizes the
/// surface texture to it.
pub fn canvas() -> Option<(u32, u32)> {
    let mut out = [0u32; 2];
    // SAFETY: the host writes exactly two words during the call.
    let ok = unsafe { gl_canvas(out.as_mut_ptr()) };
    (ok != 0).then_some((out[0], out[1]))
}

pub fn create_buffer(size: u64, kinds: u32) -> Result<u32, String> {
    // SAFETY: plain values.
    made(
        unsafe { gl_create_buffer(size as f64, kinds) },
        "createBuffer",
    )
}

pub fn write_buffer(buffer: u32, offset: u64, data: &[u8]) {
    // SAFETY: the host copies `data` during the call.
    unsafe { gl_write_buffer(buffer, offset as f64, data.as_ptr(), data.len()) }
}

/// The contents of a readback buffer (`len` bytes).
pub fn read_buffer(buffer: u32, len: usize) -> Result<Vec<u8>, String> {
    let mut out = vec![0u8; len];
    // SAFETY: the host writes at most `len` bytes into `out` during the call.
    let ok = unsafe { gl_read_buffer(buffer, out.as_mut_ptr(), len) };
    if ok == 0 {
        Err(format!("read_buffer: {}", last_error()))
    } else {
        Ok(out)
    }
}

pub fn create_texture(
    size: (u32, u32),
    layers: u32,
    mips: u32,
    format: u32,
    array: bool,
) -> Result<u32, String> {
    // SAFETY: plain values.
    let h = unsafe { gl_create_texture(size.0, size.1, layers, mips, format, u32::from(array)) };
    made(h, "createTexture")
}

pub fn write_texture(texture: u32, (mip, layer): (u32, u32), (w, h): (u32, u32), data: &[u8]) {
    // SAFETY: the host copies `data` during the call.
    unsafe { gl_write_texture(texture, mip, layer, w, h, data.as_ptr(), data.len()) }
}

/// Uploads the host's current source (`width` x `height`) into one whole subresource with
/// `texSubImage2D`, unflipped and with straight alpha. Returns the host's error if there is no
/// source, its size differs or the browser refused it.
pub fn write_texture_source(
    texture: u32,
    (mip, layer): (u32, u32),
    (w, h): (u32, u32),
) -> Result<(), String> {
    // SAFETY: plain values; the host reads only its own source slot.
    let ok = unsafe { gl_write_texture_source(texture, mip, layer, w, h) };
    if ok == 0 {
        Err(format!("texture from source: {}", last_error()))
    } else {
        Ok(())
    }
}

pub fn generate_mips(texture: u32) {
    // SAFETY: plain value.
    unsafe { gl_generate_mips(texture) }
}

pub fn create_sampler(desc: &[u32]) -> Result<u32, String> {
    // SAFETY: the host reads `desc` during the call.
    let h = unsafe { gl_create_sampler(desc.as_ptr(), desc.len() as u32) };
    made(h, "createSampler")
}

/// Compiles and links `vs` and `fs`, binds the uniform blocks and texture units listed in
/// `meta` (see `Gpu::program`) and returns the program, or the compile/link log.
pub fn create_program(vs: &str, fs: &str, meta: &str) -> Result<u32, String> {
    // SAFETY: the host reads the three UTF-8 texts during the call.
    let h = unsafe {
        gl_create_program(
            vs.as_ptr(),
            vs.len(),
            fs.as_ptr(),
            fs.len(),
            meta.as_ptr(),
            meta.len(),
        )
    };
    made(h, "linkProgram")
}

pub fn create_pipeline(program: u32, state: &[u32]) -> Result<u32, String> {
    // SAFETY: the host reads `state` during the call.
    let h = unsafe { gl_create_pipeline(program, state.as_ptr(), state.len() as u32) };
    made(h, "createPipeline")
}

/// A vertex array object for `pipeline`'s vertex layout reading `buffer` from `offset`, with
/// `index` (0 for none) as its element buffer.
pub fn create_vao(pipeline: u32, buffer: u32, offset: u32, index: u32) -> Result<u32, String> {
    // SAFETY: plain values.
    made(
        unsafe { gl_create_vao(pipeline, buffer, offset, index) },
        "createVertexArray",
    )
}

/// Uploads `push` to the push-constant ring and executes `commands`; with `present`, then
/// copies the surface texture to the canvas.
pub fn submit(commands: &[u32], push: &[u8], present: bool) -> Result<(), String> {
    // SAFETY: the host reads both slices during the call.
    let ok = unsafe {
        gl_submit(
            commands.as_ptr(),
            commands.len() as u32,
            push.as_ptr(),
            push.len(),
            u32::from(present),
        )
    };
    if ok == 0 {
        Err(format!("submit: {}", last_error()))
    } else {
        Ok(())
    }
}

pub fn release(handle: u32) {
    if handle != 0 && handle != SURFACE {
        // SAFETY: plain value; the host ignores unknown handles.
        unsafe { gl_release(handle) }
    }
}
