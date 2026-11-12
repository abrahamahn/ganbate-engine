//! Ganbate's GPU layer ("RHI v0"): a backend-neutral interface shaped like WebGPU and Metal —
//! pipelines, bind groups, passes and automatic barriers — implemented today on the engine's
//! direct Vulkan 1.0 backend. Metal, WebGPU and D3D12 backends can implement the same API.
//!
//! # Objects
//! A [`Device`] owns every object and hands out small `Copy` handles ([`BufferId`],
//! [`TextureId`], [`SamplerId`], [`PipelineId`], [`BindGroupId`]). Handles are generational:
//! using a destroyed object is reported as an error. Everything is freed when the device drops
//! (after the GPU is idle). Pipelines are built from [`ganbate_shader::Compiled`] modules and their
//! reflection: bind-group layouts, push-constant sizes and vertex inputs come from the shaders.
//! A bind group is created for one `@group` of one pipeline and can be used with any pipeline
//! whose layout for that group is identical (same bindings, kinds and shader stages). Because
//! reflection lists only statically used resources, entries for unused bindings are ignored.
//!
//! # Command recording
//! An [`Encoder`] only records. [`Device::submit`] replays it into a command buffer, submits
//! it and waits for it (offscreen work, uploads, readback). [`Device::present`] replays it for
//! the swapchain texture returned by [`Device::acquire`] and presents without waiting.
//!
//! # Synchronization model
//! Applications never write barriers. During replay the backend tracks, per texture
//! subresource (mip, layer) and per buffer, the current layout and the pipeline stages and
//! accesses of the last write and of the reads since then. This state persists across
//! submissions in queue submission order, so work recorded by `present` for frame N+1 is
//! correctly ordered after frame N even while N is still executing.
//!
//! The commands of an encoder are grouped into *scopes*: each render pass (from `begin_pass`
//! to `end_pass`), each `dispatch`, each copy and each blit. Before a scope begins, the backend
//! collects every resource the scope uses —
//! * colour attachments (colour-attachment layout, read+write),
//! * the depth attachment (depth-attachment layout, read+write),
//! * every texture referenced by a bind group set inside the pass or passed to the dispatch
//!   (shader-read-only layout for sampled textures, *including depth textures*; general layout
//!   for storage textures), in the shader stages declared by the binding,
//! * uniform, storage, vertex and index buffers,
//! * copy/blit sources and destinations (transfer layouts),
//!
//! and emits one `vkCmdPipelineBarrier` that transitions layouts and orders the scope after
//! earlier writes (read-after-write, with memory visibility), earlier reads (write-after-read)
//! and earlier writes (write-after-write). Barriers are only ever placed between scopes, never
//! inside a pass. Read-after-read in the same layout needs no barrier. Contents are discarded
//! (old layout undefined) when a pass clears or ignores an attachment and when a blit or upload
//! overwrites a whole subresource. A subresource used in two different ways in one pass (for
//! example as an attachment and as a sampled texture) is an error. Within one pass, storage
//! writes are not ordered between draws (as in WebGPU); split the work into passes instead.
//!
//! Host interaction: `write_buffer` on `Upload` memory and `read_buffer` first wait for any
//! in-flight submission that uses the buffer, so they are always safe but can stall; index
//! per-frame buffers by [`Device::frame_slot`] to avoid stalls. `write_buffer` on `Device`
//! memory, `write_texture` and `generate_mips` record their own small submission, which is
//! ordered after earlier GPU work on the resource, and wait for it. Readback buffers written
//! by a submission are made visible to the host at its end.
//!
//! Presentation: `acquire` waits on the acquire semaphore in the colour-output and transfer
//! stages, so a frame can render or blit into the swapchain texture (whose previous contents
//! are undefined: clear it, ignore it or blit over it first); `present` transitions it for
//! presentation. At most two frames are in flight: when `present` returns, the GPU has
//! finished the frame that last used the new [`Device::frame_slot`].
//!
//! # Conventions
//! Clip space is Vulkan's: y points down and depth runs 0..1 (the engine camera already
//! produces this), and the viewport has a positive height. Uploads and readbacks use tightly
//! packed rows of `Format::texel_size` bytes.
// Without a backend (no Vulkan on this platform yet) only the descriptions and the encoder exist.
#![cfg_attr(
    not(any(
        target_os = "linux",
        target_os = "windows",
        target_os = "android",
        target_vendor = "apple"
    )),
    allow(dead_code, unused_imports)
)]
mod encoder;
mod types;
pub use encoder::Encoder;
pub(crate) use encoder::{Command, PassView, Span, TransferCmd};
pub use ganbate_shader as shader;
pub use types::*;

#[cfg(target_vendor = "apple")]
use crate::metal::Gpu;
#[cfg(any(target_os = "linux", target_os = "windows", target_os = "android"))]
use crate::vulkan::rhi::Gpu;
pub(crate) mod pool;
#[cfg(target_arch = "wasm32")]
mod webgl;
#[cfg(target_arch = "wasm32")]
mod webgpu;

/// Compiles `entry` of a shader `source` for this platform's backend: SPIR-V and reflection,
/// plus WGSL (with push constants lowered to a dynamically offset uniform) when the backend is
/// WebGPU, or GLSL ES 3.00 when it is WebGL 2. Browser pipelines need shaders compiled this way.
pub fn compile_shader(source: &str, stage: shader::Stage, entry: &str) -> Result<shader::Compiled> {
    compile_shader_specialized(source, stage, entry, &[])
}

/// [`compile_shader`] with module `const`s set (`specialize`): branches on them fold away, with
/// the functions and resources only those branches use.
pub fn compile_shader_specialized(
    source: &str,
    stage: shader::Stage,
    entry: &str,
    specialize: &[(&str, shader::SpecValue)],
) -> Result<shader::Compiled> {
    let options = shader::CompileOptions::new().with_specialize(specialize);
    #[cfg(target_arch = "wasm32")]
    let compiled = match web_api() {
        Some(WebApi::WebGl2) => {
            ganbate_shader::compile_glsl_with_options(source, stage, entry, &options)
        }
        _ => ganbate_shader::compile_wgsl_with_options(source, stage, entry, &options),
    };
    #[cfg(not(target_arch = "wasm32"))]
    let compiled = ganbate_shader::compile_with_options(source, stage, entry, &options);
    compiled.map_err(|e| format!("shader {entry}: {e}"))
}

/// The graphics API a browser host offers.
#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WebApi {
    WebGpu,
    WebGl2,
}

/// The API the host (`web/ganbate.js`) set up before the engine started, if any.
#[cfg(target_arch = "wasm32")]
pub fn web_api() -> Option<WebApi> {
    #[link(wasm_import_module = "ganbate")]
    unsafe extern "C" {
        fn gpu_api() -> u32;
    }
    // SAFETY: a plain query.
    match unsafe { gpu_api() } {
        1 => Some(WebApi::WebGpu),
        2 => Some(WebApi::WebGl2),
        _ => None,
    }
}

/// Validation messages collected while `GANBATE_VULKAN_VALIDATION=1`. The log outlives the device,
/// so it can also report errors raised while the device is torn down.
#[derive(Clone, Debug, Default)]
pub struct ValidationLog(pub(crate) std::sync::Arc<std::sync::Mutex<Vec<String>>>);
impl ValidationLog {
    /// Error messages reported so far.
    pub fn errors(&self) -> Vec<String> {
        self.0.lock().map(|l| l.clone()).unwrap_or_default()
    }
}

/// A GPU device with its queue, object pools and, for [`Device::for_window`], a surface.
///
/// The lifetime is that of the window a presentation device draws into; headless devices are
/// `Device<'static>`. Set `GANBATE_VULKAN_DEVICE` to a name substring to pick an adapter and
/// `GANBATE_VULKAN_VALIDATION=1` to enable the Khronos validation layer with synchronization
/// validation (creation then fails if the layer is missing).
///
/// On macOS and iOS the device is Metal (see the `metal` module): on a Mac,
/// `GANBATE_METAL_DEVICE` picks a GPU by name substring, and Metal's own validation is enabled
/// with `MTL_DEBUG_LAYER=1` (reported on the console, not through [`ValidationLog`]).
#[cfg(any(
    target_os = "linux",
    target_os = "windows",
    target_os = "android",
    target_vendor = "apple"
))]
pub struct Device<'w> {
    gpu: Gpu<'w>,
    encoders: std::sync::Mutex<Vec<Encoder>>,
}

#[cfg(any(
    target_os = "linux",
    target_os = "windows",
    target_os = "android",
    target_vendor = "apple"
))]
impl Device<'static> {
    /// An offscreen device.
    pub fn headless() -> Result<Self> {
        Ok(Self {
            gpu: Gpu::headless()?,
            encoders: std::sync::Mutex::new(Vec::new()),
        })
    }
}

#[cfg(any(
    target_os = "linux",
    target_os = "windows",
    target_os = "android",
    target_vendor = "apple"
))]
impl<'w> Device<'w> {
    /// A device that can present to `window` (X11 on Linux, Win32 on Windows, the window's
    /// `CAMetalLayer` on macOS and iOS). The window must outlive the device.
    #[cfg(any(
        target_os = "linux",
        target_os = "windows",
        target_os = "macos",
        target_os = "ios"
    ))]
    pub fn for_window(window: &'w crate::window::Window) -> Result<Self> {
        Ok(Self {
            gpu: Gpu::for_window(window)?,
            encoders: std::sync::Mutex::new(Vec::new()),
        })
    }
    pub fn name(&self) -> &str {
        self.gpu.name()
    }
    pub fn limits(&self) -> Limits {
        self.gpu.limits()
    }
    /// True when `GANBATE_VULKAN_VALIDATION=1` enabled the validation layer.
    pub fn validation_enabled(&self) -> bool {
        self.gpu.validation_log().is_some()
    }
    /// Validation errors reported so far (always empty without validation).
    pub fn validation_errors(&self) -> Vec<String> {
        self.gpu
            .validation_log()
            .map(|l| l.errors())
            .unwrap_or_default()
    }
    /// How long a submission may run before it is reported as lost (default 20 s). Applies to
    /// every device in the process; raise it for long offline renders on slow devices.
    pub fn set_submit_timeout(&mut self, timeout: std::time::Duration) {
        // Metal waits with waitUntilCompleted and relies on the system's GPU watchdog.
        #[cfg(not(target_vendor = "apple"))]
        crate::vulkan::rhi::set_fence_timeout(timeout);
        #[cfg(target_vendor = "apple")]
        let _ = timeout;
    }
    /// A handle to the validation log that stays readable after the device is dropped.
    pub fn validation_log(&self) -> ValidationLog {
        self.gpu.validation_log().unwrap_or_default()
    }

    pub fn create_buffer(&mut self, d: &BufferDesc) -> Result<BufferId> {
        self.gpu.create_buffer(d)
    }
    /// `Upload`/`Readback` memory: memcpy into the mapping (after waiting for in-flight GPU
    /// use of the buffer). `Device` memory: staged copy, submitted and waited for; requires
    /// `BufferUsage::COPY_DST`.
    pub fn write_buffer(&mut self, b: BufferId, offset: u64, data: &[u8]) -> Result<()> {
        self.gpu.write_buffer(b, offset, data)
    }
    /// The bytes a write to `b` sends at least: a write sends every unit of this many bytes it
    /// touches whole, so writes to one unit cost no more joined. Here only the 4 bytes writes
    /// align to (on WebGL 2 a storage buffer's are a row of its texture).
    pub fn write_granularity(&self, b: BufferId) -> u64 {
        let _ = b;
        4
    }
    /// The whole contents of a `Readback` buffer, after the submissions that wrote it.
    pub fn read_buffer(&self, b: BufferId) -> Result<Vec<u8>> {
        self.gpu.read_buffer(b)
    }
    pub fn create_texture(&mut self, d: &TextureDesc) -> Result<TextureId> {
        self.gpu.create_texture(d)
    }
    /// Replaces one whole subresource with tightly packed rows (requires `COPY_DST`) and leaves
    /// it ready to sample.
    pub fn write_texture(&mut self, t: TextureId, mip: u32, layer: u32, data: &[u8]) -> Result<()> {
        self.gpu.write_texture(t, mip, layer, data)
    }
    /// Fills mips 1.. of every layer by linear blits from mip 0 (requires `COPY_SRC | COPY_DST`
    /// and a filterable colour format) and leaves the texture ready to sample.
    pub fn generate_mips(&mut self, t: TextureId) -> Result<()> {
        self.gpu.generate_mips(t)
    }
    /// The description a texture was created with (swapchain textures included).
    pub fn texture_desc(&self, t: TextureId) -> Result<TextureDesc> {
        self.gpu.texture_desc(t)
    }
    /// Identical descriptions share one sampler.
    pub fn create_sampler(&mut self, d: &SamplerDesc) -> Result<SamplerId> {
        self.gpu.create_sampler(d)
    }
    pub fn create_render_pipeline(&mut self, d: &RenderPipelineDesc) -> Result<PipelineId> {
        self.gpu.create_render_pipeline(d)
    }
    pub fn create_compute_pipeline(
        &mut self,
        shader: &ganbate_shader::Compiled,
    ) -> Result<PipelineId> {
        self.gpu.create_compute_pipeline(shader)
    }
    /// Binds `entries` (`(binding, resource)`) to `@group(group)` of `pipeline`. Every binding
    /// the pipeline uses in that group must be given exactly once, with a matching kind.
    /// Reflection only lists what the entry points statically use, so entries for other
    /// bindings are ignored, and a group the pipeline does not use yields an empty bind group.
    pub fn create_bind_group(
        &mut self,
        pipeline: PipelineId,
        group: u32,
        entries: &[(u32, Resource)],
    ) -> Result<BindGroupId> {
        self.gpu.create_bind_group(pipeline, group, entries)
    }
    /// Safe while frames are in flight: the object is freed once the GPU is done with it.
    pub fn destroy_buffer(&mut self, b: BufferId) {
        self.gpu.destroy_buffer(b)
    }
    /// Safe while frames are in flight. Swapchain textures are owned by the device and ignored.
    pub fn destroy_texture(&mut self, t: TextureId) {
        self.gpu.destroy_texture(t)
    }
    /// Safe while frames are in flight.
    pub fn destroy_bind_group(&mut self, g: BindGroupId) {
        self.gpu.destroy_bind_group(g)
    }
    pub fn encoder(&self) -> Encoder {
        self.encoders
            .lock()
            .ok()
            .and_then(|mut pool| pool.pop())
            .unwrap_or_default()
    }
    fn recycle_encoder(&self, mut e: Encoder) {
        e.clear();
        if let Ok(mut pool) = self.encoders.lock()
            && pool.len() < 4
        {
            pool.push(e);
        }
    }
    /// Executes `e` and waits for completion.
    pub fn submit(&mut self, e: Encoder) -> Result<()> {
        let r = self.gpu.submit(&e);
        self.recycle_encoder(e);
        r
    }
    /// The next swapchain texture, or `None` while the window is hidden, minimized or the
    /// surface is temporarily unavailable. Recreates the swapchain after resizes. Calling it
    /// again before `present` returns the same texture.
    pub fn acquire(&mut self) -> Result<Option<TextureId>> {
        self.gpu.acquire()
    }
    /// Format of the swapchain textures: `Bgra8Srgb` when the surface offers it, otherwise
    /// `Bgra8Unorm`, `Rgba8Srgb` or `Rgba8Unorm`. Headless devices report `Bgra8Srgb`.
    pub fn surface_format(&self) -> Format {
        self.gpu.surface_format()
    }
    /// Size of the current swapchain (or of the window before the first `acquire`); `(0, 0)`
    /// for headless devices.
    pub fn surface_size(&self) -> (u32, u32) {
        self.gpu.surface_size()
    }
    /// Submits `e`, which renders into the texture returned by `acquire`, and presents it.
    pub fn present(&mut self, e: Encoder) -> Result<()> {
        let r = self.gpu.present(&e);
        self.recycle_encoder(e);
        r
    }
    /// 0 or 1: which per-frame resources the next frame may overwrite. When `present` returns,
    /// the GPU has finished the earlier frame that used this slot.
    pub fn frame_slot(&self) -> usize {
        self.gpu.frame_slot()
    }
}

/// A GPU device in a browser: WebGPU when the browser has it, otherwise WebGL 2 (`wasm32`
/// builds hosted by `web/ganbate.js`, which picks the API; see [`web_api`]).
///
/// The host creates the adapter and device (or the WebGL 2 context) and the canvas before the
/// engine starts. The contract above holds with these differences: `submit` and `present`
/// never wait; queue writes and submissions run in call order, so per-frame resources are not
/// required for correctness; the frame is presented when the `requestAnimationFrame` callback
/// returns. On WebGPU `read_buffer` returns an error (browsers map buffers asynchronously),
/// errors arrive asynchronously and [`Device::validation_errors`] returns those reported so
/// far. On WebGL 2 `read_buffer` works, GL errors are collected after every submission, and
/// compute, storage writes, storage textures and texture views are not available (see the
/// `webgl` module). Shaders must come from [`compile_shader`].
#[cfg(target_arch = "wasm32")]
pub struct Device<'w> {
    gpu: Web<'w>,
    encoders: std::sync::Mutex<Vec<Encoder>>,
}

#[cfg(target_arch = "wasm32")]
enum Web<'w> {
    Gpu(webgpu::Gpu<'w>),
    Gl(webgl::Gpu<'w>),
}

/// Calls the same method on whichever browser backend the device uses.
#[cfg(target_arch = "wasm32")]
macro_rules! web {
    ($self:ident, $g:ident => $e:expr) => {
        match &$self.gpu {
            Web::Gpu($g) => $e,
            Web::Gl($g) => $e,
        }
    };
    (mut $self:ident, $g:ident => $e:expr) => {
        match &mut $self.gpu {
            Web::Gpu($g) => $e,
            Web::Gl($g) => $e,
        }
    };
}

#[cfg(target_arch = "wasm32")]
impl Device<'static> {
    fn open(presentable: bool) -> Result<Self> {
        Ok(Self {
            gpu: match web_api() {
                Some(WebApi::WebGl2) => Web::Gl(webgl::Gpu::open(presentable)?),
                _ => Web::Gpu(webgpu::Gpu::open(presentable)?),
            },
            encoders: std::sync::Mutex::new(Vec::new()),
        })
    }
    /// The host's device, without presentation.
    pub fn headless() -> Result<Self> {
        Self::open(false)
    }
    /// The host's device and its canvas: `acquire` returns the canvas texture.
    pub fn for_canvas() -> Result<Self> {
        Self::open(true)
    }
}

#[cfg(target_arch = "wasm32")]
impl Device<'_> {
    /// Which browser API this device draws with.
    pub fn web_api(&self) -> WebApi {
        match self.gpu {
            Web::Gpu(_) => WebApi::WebGpu,
            Web::Gl(_) => WebApi::WebGl2,
        }
    }
    pub fn name(&self) -> &str {
        web!(self, g => g.name())
    }
    pub fn limits(&self) -> Limits {
        web!(self, g => g.limits())
    }
    /// Always true: browsers validate every call.
    pub fn validation_enabled(&self) -> bool {
        true
    }
    /// Errors reported so far (validation, out of memory, lost device, shader compilation).
    /// On WebGPU they arrive asynchronously, after the call that caused them.
    pub fn validation_errors(&self) -> Vec<String> {
        web!(self, g => g.validation_errors())
    }
    pub fn validation_log(&self) -> ValidationLog {
        ValidationLog(std::sync::Arc::new(std::sync::Mutex::new(
            self.validation_errors(),
        )))
    }
    /// No effect: browsers submit without waiting and report a lost device themselves.
    pub fn set_submit_timeout(&mut self, timeout: std::time::Duration) {
        let _ = timeout;
    }
    pub fn create_buffer(&mut self, d: &BufferDesc) -> Result<BufferId> {
        web!(mut self, g => g.create_buffer(d))
    }
    /// Ordered after earlier submissions. On WebGPU, offset and length must be multiples of 4.
    pub fn write_buffer(&mut self, b: BufferId, offset: u64, data: &[u8]) -> Result<()> {
        web!(mut self, g => g.write_buffer(b, offset, data))
    }
    /// The bytes a write to `b` sends at least: a write sends every unit of this many bytes it
    /// touches whole, so writes to one unit cost no more joined. On WebGL 2 a storage buffer's
    /// are a row of the texture the host keeps it in; elsewhere the 4 bytes writes align to.
    pub fn write_granularity(&self, b: BufferId) -> u64 {
        match &self.gpu {
            Web::Gpu(_) => 4,
            Web::Gl(g) => g.write_granularity(b),
        }
    }
    /// Not available on WebGPU; synchronous on WebGL 2.
    pub fn read_buffer(&self, b: BufferId) -> Result<Vec<u8>> {
        web!(self, g => g.read_buffer(b))
    }
    pub fn create_texture(&mut self, d: &TextureDesc) -> Result<TextureId> {
        web!(mut self, g => g.create_texture(d))
    }
    pub fn write_texture(&mut self, t: TextureId, mip: u32, layer: u32, data: &[u8]) -> Result<()> {
        web!(mut self, g => g.write_texture(t, mip, layer, data))
    }
    /// Replaces one whole subresource of an 8-bit RGBA texture (requires `COPY_DST`) with the
    /// image the host page set for the current runtime call (a canvas or an image bitmap), copied
    /// by the browser: unflipped, straight alpha, sRGB-encoded bytes as `write_texture` takes them.
    /// `size` is the image's size and must be the mip's.
    pub fn write_texture_source(
        &mut self,
        t: TextureId,
        mip: u32,
        layer: u32,
        size: (u32, u32),
    ) -> Result<()> {
        web!(mut self, g => g.write_texture_source(t, mip, layer, size))
    }
    pub fn generate_mips(&mut self, t: TextureId) -> Result<()> {
        web!(mut self, g => g.generate_mips(t))
    }
    pub fn texture_desc(&self, t: TextureId) -> Result<TextureDesc> {
        web!(self, g => g.texture_desc(t))
    }
    pub fn create_sampler(&mut self, d: &SamplerDesc) -> Result<SamplerId> {
        web!(mut self, g => g.create_sampler(d))
    }
    pub fn create_render_pipeline(&mut self, d: &RenderPipelineDesc) -> Result<PipelineId> {
        web!(mut self, g => g.create_render_pipeline(d))
    }
    pub fn create_compute_pipeline(
        &mut self,
        shader: &ganbate_shader::Compiled,
    ) -> Result<PipelineId> {
        web!(mut self, g => g.create_compute_pipeline(shader))
    }
    pub fn create_bind_group(
        &mut self,
        pipeline: PipelineId,
        group: u32,
        entries: &[(u32, Resource)],
    ) -> Result<BindGroupId> {
        web!(mut self, g => g.create_bind_group(pipeline, group, entries))
    }
    pub fn destroy_buffer(&mut self, b: BufferId) {
        web!(mut self, g => g.destroy_buffer(b))
    }
    pub fn destroy_texture(&mut self, t: TextureId) {
        web!(mut self, g => g.destroy_texture(t))
    }
    pub fn destroy_bind_group(&mut self, b: BindGroupId) {
        web!(mut self, g => g.destroy_bind_group(b))
    }
    pub fn encoder(&self) -> Encoder {
        self.encoders
            .lock()
            .ok()
            .and_then(|mut pool| pool.pop())
            .unwrap_or_default()
    }
    fn recycle_encoder(&self, mut e: Encoder) {
        e.clear();
        if let Ok(mut pool) = self.encoders.lock()
            && pool.len() < 4
        {
            pool.push(e);
        }
    }
    /// Records `e` and submits it without waiting.
    pub fn submit(&mut self, e: Encoder) -> Result<()> {
        let r = web!(mut self, g => g.submit(&e));
        self.recycle_encoder(e);
        r
    }
    /// The canvas texture at the canvas's current size, or `None` while it has no area.
    pub fn acquire(&mut self) -> Result<Option<TextureId>> {
        web!(mut self, g => g.acquire())
    }
    /// The canvas format (`Bgra8Unorm` or `Rgba8Unorm`; the renderer encodes sRGB itself).
    pub fn surface_format(&self) -> Format {
        web!(self, g => g.surface_format())
    }
    /// The canvas size in device pixels.
    pub fn surface_size(&self) -> (u32, u32) {
        web!(self, g => g.surface_size())
    }
    /// Submits `e`, which renders into the canvas texture; the browser presents it when the
    /// current animation frame callback returns.
    pub fn present(&mut self, e: Encoder) -> Result<()> {
        let r = web!(mut self, g => g.present(&e));
        self.recycle_encoder(e);
        r
    }
    /// 0 or 1, alternating with every `present`.
    pub fn frame_slot(&self) -> usize {
        web!(self, g => g.frame_slot())
    }
}

#[cfg(all(
    test,
    any(target_os = "linux", target_os = "windows", target_vendor = "apple")
))]
mod tests {
    use super::*;
    use ganbate_core::world::TextureCompression;

    #[test]
    fn compressed_bc1_bc7_and_astc4x4_textures_upload_and_sample() {
        let Ok(mut dev) = Device::headless() else {
            return; // Headless GPU unavailable on this host.
        };
        let shader_src = "
            @group(0) @binding(0) var tex: texture_2d<f32>;
            @group(0) @binding(1) var smp: sampler;
            struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
            @vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
                let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
                var o: VOut;
                o.pos = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
                o.uv = uv;
                return o;
            }
            @fragment fn fs(in: VOut) -> @location(0) vec4f {
                return textureSample(tex, smp, in.uv);
            }
        ";
        let vs = compile_shader(shader_src, shader::Stage::Vertex, "vs").unwrap();
        let fs = compile_shader(shader_src, shader::Stage::Fragment, "fs").unwrap();
        let pipe = dev
            .create_render_pipeline(&RenderPipelineDesc {
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
            })
            .unwrap();
        let sampler = dev
            .create_sampler(&SamplerDesc {
                filter: Filter::Nearest,
                mip_filter: Filter::Nearest,
                ..SamplerDesc::default()
            })
            .unwrap();
        let dst = dev
            .create_texture(&TextureDesc::new(
                4,
                4,
                Format::Rgba8Unorm,
                TextureUsage::COLOR_TARGET | TextureUsage::COPY_SRC,
            ))
            .unwrap();
        let rb = dev
            .create_buffer(&BufferDesc {
                size: 4 * 4 * 4,
                usage: BufferUsage::COPY_DST,
                memory: Memory::Readback,
            })
            .unwrap();

        let expected_rgba = [212u8, 84, 168, 255];
        for (comp, fmt) in [
            (TextureCompression::Bc1, Format::Bc1RgbaUnorm),
            (TextureCompression::Bc7, Format::Bc7RgbaUnorm),
            (TextureCompression::Astc4x4, Format::Astc4x4Unorm),
        ] {
            let mut desc =
                TextureDesc::new(8, 8, fmt, TextureUsage::SAMPLED | TextureUsage::COPY_DST);
            desc.mips = 3; // 8x8, 4x4, 2x2 (includes sub-block tail mip)
            let tex = dev.create_texture(&desc).unwrap();
            for mip in 0..3 {
                let (w, h) = desc.mip_size(mip);
                let mut src = Vec::with_capacity((w * h * 4) as usize);
                for _ in 0..(w * h) {
                    src.extend_from_slice(&expected_rgba);
                }
                let blocks = ganbate_assets::bc::encode_rgba8(comp, &src, w, h).unwrap();
                dev.write_texture(tex, mip, 0, &blocks).unwrap();
            }
            let bg = dev
                .create_bind_group(
                    pipe,
                    0,
                    &[(0, Resource::Texture(tex)), (1, Resource::Sampler(sampler))],
                )
                .unwrap();
            let mut enc = dev.encoder();
            enc.begin_pass(PassDesc {
                colors: vec![Attachment::clear(dst, [0.0; 4])],
                depth: None,
            });
            enc.set_pipeline(pipe);
            enc.set_bind_group(0, bg);
            enc.draw(3, 1, 0, 0);
            enc.end_pass();
            enc.copy_texture_to_buffer(dst, 0, 0, rb, 0);
            dev.submit(enc).unwrap();

            let pixels = dev.read_buffer(rb).unwrap();
            for ch in 0..4 {
                assert!(
                    pixels[ch].abs_diff(expected_rgba[ch]) <= 12,
                    "{fmt:?} ch {ch}: got {} vs {}",
                    pixels[ch],
                    expected_rgba[ch]
                );
            }
            dev.destroy_bind_group(bg);
            dev.destroy_texture(tex);
        }
    }
}
