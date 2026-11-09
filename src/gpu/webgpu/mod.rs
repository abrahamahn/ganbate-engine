//! WebGPU implementation of the backend-neutral GPU layer in [`crate::gpu`], for `wasm32`
//! builds running in a browser. The JavaScript host (`web/ganbate.js`) creates the adapter, device
//! and canvas context before the engine starts and provides the imports in [`ffi`].
//!
//! Ownership: [`Gpu`] keeps the engine's generational pools of buffers, textures, samplers,
//! pipelines and bind groups; each entry holds the handle of the browser object in the host's
//! table (see [`ffi`] for its lifetime rules). Layouts and shader modules are cached by content.
//!
//! Commands: an [`Encoder`](crate::gpu::Encoder) is validated and replayed into a compact
//! command stream ([`record`]) that the host turns into one `GPUCommandEncoder` and submits.
//! The browser derives every barrier. Push constants (which WebGPU lacks) are written into a
//! per-submission ring of [`PUSH_SLOT`]-byte uniform blocks and bound with a dynamic offset at
//! [`ganbate_shader::wgsl::PUSH_CONSTANT_GROUP`]; shaders are compiled to WGSL with that lowering
//! by [`ganbate_shader::compile_wgsl`] (see [`crate::gpu::compile_shader`]).
//!
//! Synchronization: `submit` and `present` return without waiting. `write_buffer`,
//! `write_texture` and submissions execute in call order on the queue, so a write made after a
//! submission never changes what that submission reads: per-frame resources indexed by
//! `frame_slot` stay correct, they are just not required. The canvas texture is presented when
//! the `requestAnimationFrame` callback that recorded the frame returns.
//!
//! Not available: `read_buffer` (mapping is asynchronous in browsers), `copy_texture_to_buffer`
//! with rows that are not a multiple of 256 bytes, writes that are not whole 4-byte words,
//! storage textures in `rg16float`/`r16float`, and depth-texture uploads.
mod ffi;
mod record;

use crate::gpu::{
    Address, BindGroupId, Blend, BufferDesc, BufferId, BufferUsage, Compare, Cull, Filter, Format,
    Limits, Memory, PipelineId, RenderPipelineDesc, Resource, SamplerDesc, SamplerId, TextureDesc,
    TextureId, TextureUsage, Topology, VertexFormat,
};
use ganbate_shader::{BindingKind, Compiled, IoType, SampleType, Stage, TexelFormat, TextureView};
use std::collections::HashMap;
use std::fmt::Write as _;

type Result<T> = std::result::Result<T, String>;

/// Bytes reserved per push-constant block in the ring; also the largest push-constant block.
pub(crate) const PUSH_SLOT: u32 = 256;
/// Bind groups every WebGPU device supports; the last one holds push constants.
const MAX_GROUPS: usize = 4;
const PUSH_GROUP: usize = ganbate_shader::wgsl::PUSH_CONSTANT_GROUP as usize;

const VISIBLE_VERTEX: u32 = 1;
const VISIBLE_FRAGMENT: u32 = 2;
const VISIBLE_COMPUTE: u32 = 4;

mod usage {
    pub const MAP_READ: u32 = 1;
    pub const COPY_SRC: u32 = 4;
    pub const COPY_DST: u32 = 8;
    pub const INDEX: u32 = 16;
    pub const VERTEX: u32 = 32;
    pub const UNIFORM: u32 = 64;
    pub const STORAGE: u32 = 128;
    pub const T_COPY_SRC: u32 = 1;
    pub const T_COPY_DST: u32 = 2;
    pub const T_TEXTURE_BINDING: u32 = 4;
    pub const T_STORAGE_BINDING: u32 = 8;
    pub const T_RENDER_ATTACHMENT: u32 = 16;
}

/// Format codes shared with the host (index into its format table).
fn format_code(f: Format) -> u32 {
    match f {
        Format::Rgba8Unorm => 1,
        Format::Rgba8Srgb => 2,
        Format::Bgra8Unorm => 3,
        Format::Bgra8Srgb => 4,
        Format::Rgba16Float => 5,
        Format::Rg16Float => 6,
        Format::R16Float => 7,
        Format::R8Unorm => 8,
        Format::R32Float => 9,
        Format::Rgba32Float => 10,
        Format::Depth32Float => 11,
        Format::Bc1RgbaUnorm => 12,
        Format::Bc1RgbaSrgb => 13,
        Format::Bc7RgbaUnorm => 14,
        Format::Bc7RgbaSrgb => 15,
        Format::Astc4x4Unorm => 16,
        Format::Astc4x4Srgb => 17,
    }
}
fn format_name(f: Format) -> &'static str {
    match f {
        Format::Rgba8Unorm => "rgba8unorm",
        Format::Rgba8Srgb => "rgba8unorm-srgb",
        Format::Bgra8Unorm => "bgra8unorm",
        Format::Bgra8Srgb => "bgra8unorm-srgb",
        Format::Rgba16Float => "rgba16float",
        Format::Rg16Float => "rg16float",
        Format::R16Float => "r16float",
        Format::R8Unorm => "r8unorm",
        Format::R32Float => "r32float",
        Format::Rgba32Float => "rgba32float",
        Format::Depth32Float => "depth32float",
        Format::Bc1RgbaUnorm => "bc1-rgba-unorm",
        Format::Bc1RgbaSrgb => "bc1-rgba-unorm-srgb",
        Format::Bc7RgbaUnorm => "bc7-rgba-unorm",
        Format::Bc7RgbaSrgb => "bc7-rgba-unorm-srgb",
        Format::Astc4x4Unorm => "astc-4x4-unorm",
        Format::Astc4x4Srgb => "astc-4x4-unorm-srgb",
    }
}
fn is_float32(f: Format) -> bool {
    matches!(f, Format::R32Float | Format::Rgba32Float)
}
fn texel_format(f: TexelFormat) -> Format {
    match f {
        TexelFormat::Rgba8Unorm => Format::Rgba8Unorm,
        TexelFormat::Rgba16Float => Format::Rgba16Float,
        TexelFormat::Rgba32Float => Format::Rgba32Float,
        TexelFormat::R32Float => Format::R32Float,
        TexelFormat::Rg16Float => Format::Rg16Float,
        TexelFormat::R16Float => Format::R16Float,
    }
}
fn view_code(v: TextureView) -> u32 {
    match v {
        TextureView::D2 => 0,
        TextureView::D2Array => 1,
        TextureView::Cube => 2,
    }
}
fn compare_name(c: Compare) -> &'static str {
    match c {
        Compare::Never => "never",
        Compare::Less => "less",
        Compare::LessEqual => "less-equal",
        Compare::Equal => "equal",
        Compare::Greater => "greater",
        Compare::GreaterEqual => "greater-equal",
        Compare::NotEqual => "not-equal",
        Compare::Always => "always",
    }
}
fn is_float(ty: IoType) -> bool {
    matches!(
        ty,
        IoType::F32 | IoType::Vec2F | IoType::Vec3F | IoType::Vec4F
    )
}
fn is_uint(ty: IoType) -> bool {
    matches!(
        ty,
        IoType::U32 | IoType::Vec2U | IoType::Vec3U | IoType::Vec4U
    )
}

use crate::gpu::pool::Pool;

struct Buffer {
    js: u32,
    desc: BufferDesc,
}

struct Texture {
    /// [`ffi::SURFACE`] for the canvas texture.
    js: u32,
    desc: TextureDesc,
    sw_decompress: Option<ganbate_core::world::TextureCompression>,
    surface: bool,
}

struct Sampler {
    js: u32,
    comparison: bool,
}

/// One binding of a bind group layout, as reflected from the shaders.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Entry {
    binding: u32,
    kind: BindingKind,
    /// GPUShaderStage bits of the stages that use the binding.
    visibility: u32,
}

#[derive(Clone, Debug, PartialEq)]
struct GroupLayout {
    js: u32,
    entries: Vec<Entry>,
}

/// What a pipeline expects at one bind group index.
#[derive(Clone, Debug, PartialEq)]
enum Slot {
    Empty,
    Group(GroupLayout),
    Push,
}

struct Pipeline {
    js: u32,
    compute: bool,
    slots: Vec<Slot>,
    push_size: u32,
    colors: Vec<Format>,
    depth: Option<Format>,
    vertex_input: bool,
}

/// A resource a bind group references.
#[derive(Clone, Copy, Debug)]
enum Bound {
    Buffer(BufferId),
    Texture {
        id: TextureId,
        mips: (u32, u32),
        layers: (u32, u32),
    },
    Sampler(SamplerId),
}

struct BindGroup {
    js: u32,
    /// `None` for a group the pipeline does not use (bound as the host's empty group).
    layout_js: Option<u32>,
    bound: Vec<Bound>,
}

/// Limits and features of the browser device.
#[derive(Clone, Copy, Debug, Default)]
struct DeviceInfo {
    max_texture_2d: u32,
    max_layers: u32,
    max_bind_groups: u32,
    uniform_align: u32,
    storage_align: u32,
    max_uniform_binding: u32,
    max_storage_binding: u32,
    max_buffer: u32,
    max_color_attachments: u32,
    depth_clip_control: bool,
    float32_filterable: bool,
    float32_blendable: bool,
    bc_compression: bool,
    astc_compression: bool,
    max_workgroups: u32,
    max_vertex_stride: u32,
    max_vertex_attributes: u32,
}

pub(crate) struct Gpu<'w> {
    name: String,
    info: DeviceInfo,
    buffers: Pool<Buffer>,
    textures: Pool<Texture>,
    samplers: Pool<Sampler>,
    pipelines: Pool<Pipeline>,
    groups: Pool<BindGroup>,
    sampler_cache: HashMap<[u32; 6], SamplerId>,
    shader_cache: HashMap<String, u32>,
    group_layouts: HashMap<Vec<u32>, u32>,
    pipeline_layouts: HashMap<Vec<u32>, u32>,
    cmd_buf: Vec<u32>,
    ring: record::Ring,
    /// The canvas texture, once `acquire` has registered it.
    surface: Option<TextureId>,
    presentable: bool,
    frame: usize,
    _window: core::marker::PhantomData<&'w ()>,
}

impl Gpu<'static> {
    /// The device the host created; `presentable` when it also configured a canvas.
    pub fn open(presentable: bool) -> Result<Self> {
        let w = ffi::info();
        if w.len() < 16 {
            return Err(format!(
                "the WebGPU host is not ready ({}); load the module through web/ganbate.js",
                ffi::last_error()
            ));
        }
        let info = DeviceInfo {
            max_texture_2d: w[0],
            max_layers: w[1],
            max_bind_groups: w[2],
            uniform_align: w[3].max(1),
            storage_align: w[4].max(1),
            max_uniform_binding: w[5],
            max_storage_binding: w[6],
            max_buffer: w[7],
            max_color_attachments: w[8],
            depth_clip_control: w[9] & 1 != 0,
            float32_filterable: w[9] & 2 != 0,
            float32_blendable: w[9] & 4 != 0,
            bc_compression: w[9] & 8 != 0,
            astc_compression: w[9] & 16 != 0,
            max_workgroups: w[10],
            max_vertex_stride: w[11],
            max_vertex_attributes: w[12],
        };
        if (info.max_bind_groups as usize) < MAX_GROUPS {
            return Err(format!(
                "the device binds {} groups; WebGPU guarantees {MAX_GROUPS}",
                info.max_bind_groups
            ));
        }
        if presentable && ffi::canvas().is_none() {
            return Err("the host has no canvas to present to".into());
        }
        Ok(Self {
            name: ffi::adapter_name(),
            info,
            buffers: Pool::default(),
            textures: Pool::default(),
            samplers: Pool::default(),
            pipelines: Pool::default(),
            groups: Pool::default(),
            sampler_cache: HashMap::new(),
            shader_cache: HashMap::new(),
            group_layouts: HashMap::new(),
            pipeline_layouts: HashMap::new(),
            cmd_buf: Vec::new(),
            ring: record::Ring::default(),
            surface: None,
            presentable,
            frame: 0,
            _window: core::marker::PhantomData,
        })
    }
}

impl<'w> Gpu<'w> {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn limits(&self) -> Limits {
        Limits {
            max_texture_2d: self.info.max_texture_2d,
            max_push_constants: PUSH_SLOT,
            max_anisotropy: 16.0,
            depth_clamp: self.info.depth_clip_control,
            uniform_offset_alignment: u64::from(self.info.uniform_align),
            storage_offset_alignment: u64::from(self.info.storage_align),
            // WebGPU samples depth32float only through comparison or non-filtering samplers.
            depth_filterable: false,
            texture_compression_bc: self.info.bc_compression,
            texture_compression_astc: self.info.astc_compression,
        }
    }
    pub fn frame_slot(&self) -> usize {
        self.frame
    }
    pub fn validation_errors(&self) -> Vec<String> {
        ffi::errors()
    }

    fn buffer(&self, id: BufferId) -> Result<&Buffer> {
        self.buffers
            .get(id.raw())
            .ok_or_else(|| format!("{id:?} was destroyed or does not belong to this device"))
    }
    fn texture(&self, id: TextureId) -> Result<&Texture> {
        self.textures
            .get(id.raw())
            .ok_or_else(|| format!("{id:?} was destroyed or does not belong to this device"))
    }
    fn pipeline(&self, id: PipelineId) -> Result<&Pipeline> {
        self.pipelines
            .get(id.raw())
            .ok_or_else(|| format!("{id:?} does not belong to this device"))
    }
    fn group(&self, id: BindGroupId) -> Result<&BindGroup> {
        self.groups
            .get(id.raw())
            .ok_or_else(|| format!("{id:?} was destroyed or does not belong to this device"))
    }

    // ------------------------------------------------------------------ buffers

    pub fn create_buffer(&mut self, d: &BufferDesc) -> Result<BufferId> {
        if d.size == 0 {
            return Err("create_buffer: the size must be positive".into());
        }
        let size = d.size.next_multiple_of(4);
        if size > u64::from(self.info.max_buffer) {
            return Err(format!(
                "create_buffer: {size} bytes exceed the device's maxBufferSize {}",
                self.info.max_buffer
            ));
        }
        let mut bits = 0;
        if d.memory == Memory::Readback {
            if d.usage.bits() & !BufferUsage::COPY_DST.bits() != 0 {
                return Err(
                    "create_buffer: WebGPU readback buffers can only be copy destinations".into(),
                );
            }
            bits = usage::MAP_READ | usage::COPY_DST;
        } else {
            for (flag, b) in [
                (BufferUsage::VERTEX, usage::VERTEX),
                (BufferUsage::INDEX, usage::INDEX),
                (BufferUsage::UNIFORM, usage::UNIFORM),
                (BufferUsage::STORAGE, usage::STORAGE),
                (BufferUsage::COPY_SRC, usage::COPY_SRC),
            ] {
                if d.usage.contains(flag) {
                    bits |= b;
                }
            }
            // Every write goes through queue.writeBuffer.
            bits |= usage::COPY_DST;
        }
        let js = ffi::create_buffer(size, bits)?;
        match self.buffers.insert(Buffer { js, desc: *d }) {
            Ok(raw) => Ok(BufferId::from_raw(raw)),
            Err(e) => {
                ffi::release(js);
                Err(e)
            }
        }
    }

    pub fn write_buffer(&mut self, id: BufferId, offset: u64, data: &[u8]) -> Result<()> {
        let b = self.buffer(id)?;
        match b.desc.memory {
            Memory::Readback => {
                return Err(format!("write_buffer: {id:?} is a readback buffer"));
            }
            Memory::Device if !b.desc.usage.contains(BufferUsage::COPY_DST) => {
                return Err(format!(
                    "write_buffer: {id:?} is device memory without COPY_DST usage"
                ));
            }
            _ => {}
        }
        let end = offset.checked_add(data.len() as u64);
        if end.is_none_or(|end| end > b.desc.size) {
            return Err(format!(
                "write_buffer: {} bytes at {offset} do not fit {id:?} ({} bytes)",
                data.len(),
                b.desc.size
            ));
        }
        if offset % 4 != 0 || data.len() % 4 != 0 {
            return Err(format!(
                "write_buffer: WebGPU writes whole 4-byte words ({} bytes at {offset})",
                data.len()
            ));
        }
        if !data.is_empty() {
            ffi::write_buffer(b.js, offset, data);
        }
        Ok(())
    }

    pub fn read_buffer(&self, id: BufferId) -> Result<Vec<u8>> {
        self.buffer(id)?;
        Err("read_buffer is not available on WebGPU: browsers map buffers asynchronously".into())
    }

    pub fn destroy_buffer(&mut self, id: BufferId) {
        if let Some(b) = self.buffers.remove(id.raw()) {
            ffi::release(b.js);
        }
    }

    // ------------------------------------------------------------------ textures

    pub fn create_texture(&mut self, d: &TextureDesc) -> Result<TextureId> {
        let max = self.info.max_texture_2d;
        if d.width == 0 || d.height == 0 || d.width > max || d.height > max {
            return Err(format!(
                "create_texture: {}x{} is outside 1..={max}",
                d.width, d.height
            ));
        }
        if d.layers == 0 || d.layers > self.info.max_layers {
            return Err(format!("create_texture: {} layers", d.layers));
        }
        let full = 32 - d.width.max(d.height).leading_zeros();
        if d.mips == 0 || d.mips > full {
            return Err(format!(
                "create_texture: {} mips for {}x{}",
                d.mips, d.width, d.height
            ));
        }
        let depth = d.format.is_depth();
        if d.format.is_compressed()
            && d.usage.bits()
                & (TextureUsage::COLOR_TARGET
                    | TextureUsage::DEPTH_TARGET
                    | TextureUsage::STORAGE
                    | TextureUsage::COPY_SRC)
                    .bits()
                != 0
        {
            return Err(format!(
                "create_texture: compressed format {:?} supports only SAMPLED | COPY_DST",
                d.format
            ));
        }
        if depth && d.usage.contains(TextureUsage::COLOR_TARGET) {
            return Err("create_texture: a depth format cannot be a colour target".into());
        }
        if !depth && d.usage.contains(TextureUsage::DEPTH_TARGET) {
            return Err(format!(
                "create_texture: {:?} cannot be a depth target",
                d.format
            ));
        }
        let mut sw_decompress = None;
        let actual_format = if let Some(comp) = d.format.compression() {
            let hw = match comp {
                ganbate_core::world::TextureCompression::Bc1
                | ganbate_core::world::TextureCompression::Bc7 => self.info.bc_compression,
                ganbate_core::world::TextureCompression::Astc4x4 => self.info.astc_compression,
            };
            if hw && d.width % 4 == 0 && d.height % 4 == 0 {
                d.format
            } else {
                sw_decompress = Some(comp);
                if d.format.is_srgb() {
                    Format::Rgba8Srgb
                } else {
                    Format::Rgba8Unorm
                }
            }
        } else {
            d.format
        };
        let mut bits = 0;
        if d.usage.contains(TextureUsage::SAMPLED) {
            bits |= usage::T_TEXTURE_BINDING;
        }
        if d.usage.contains(TextureUsage::COLOR_TARGET)
            || d.usage.contains(TextureUsage::DEPTH_TARGET)
        {
            bits |= usage::T_RENDER_ATTACHMENT;
        }
        if d.usage.contains(TextureUsage::STORAGE) {
            if !matches!(
                actual_format,
                Format::Rgba8Unorm | Format::Rgba16Float | Format::Rgba32Float | Format::R32Float
            ) {
                return Err(format!(
                    "create_texture: {:?} is not a WebGPU storage format",
                    d.format
                ));
            }
            bits |= usage::T_STORAGE_BINDING;
        }
        // Blits and mip generation are render passes on WebGPU: sources are sampled and
        // uncompressed colour destinations are rendered to.
        if d.usage.contains(TextureUsage::COPY_SRC) {
            bits |= usage::T_COPY_SRC | usage::T_TEXTURE_BINDING;
        }
        if d.usage.contains(TextureUsage::COPY_DST) {
            bits |= usage::T_COPY_DST;
            if !depth && !actual_format.is_compressed() {
                bits |= usage::T_RENDER_ATTACHMENT;
            }
        }
        let js = ffi::create_texture(
            (d.width, d.height),
            d.layers,
            d.mips,
            format_code(actual_format),
            bits,
        )?;
        match self.textures.insert(Texture {
            js,
            desc: *d,
            sw_decompress,
            surface: false,
        }) {
            Ok(raw) => Ok(TextureId::from_raw(raw)),
            Err(e) => {
                ffi::release(js);
                Err(e)
            }
        }
    }

    pub fn write_texture(
        &mut self,
        id: TextureId,
        mip: u32,
        layer: u32,
        data: &[u8],
    ) -> Result<()> {
        let t = self.texture(id)?;
        if t.surface {
            return Err("write_texture: the canvas texture cannot be written".into());
        }
        if !t.desc.usage.contains(TextureUsage::COPY_DST) {
            return Err(format!("write_texture: {id:?} lacks COPY_DST usage"));
        }
        if t.desc.format.is_depth() {
            return Err("write_texture: WebGPU cannot upload depth textures".into());
        }
        if mip >= t.desc.mips || layer >= t.desc.layers {
            return Err(format!(
                "write_texture: mip {mip} layer {layer} is outside {id:?}"
            ));
        }
        let (w, h) = t.desc.mip_size(mip);
        let expected = t.desc.format.mip_byte_len(w, h);
        if data.len() as u64 != expected {
            return Err(format!(
                "write_texture: expected {expected} bytes for {w}x{h} {:?}, got {}",
                t.desc.format,
                data.len()
            ));
        }
        let (js, sw, fmt) = (t.js, t.sw_decompress, t.desc.format);
        if let Some(comp) = sw {
            let decoded = ganbate_assets::bc::decode_rgba8(comp, data, w, h)?;
            ffi::write_texture(js, (mip, layer), (w, h), w * 4, &decoded);
        } else if fmt.is_compressed() {
            let copy_w = (w + 3) & !3;
            let copy_h = (h + 3) & !3;
            ffi::write_texture(js, (mip, layer), (copy_w, copy_h), fmt.row_bytes(w), data);
        } else {
            ffi::write_texture(js, (mip, layer), (w, h), fmt.row_bytes(w), data);
        }
        Ok(())
    }

    /// Copies the host's current source image (see `ffi::copy_source`) into one whole
    /// subresource; `size` is the source's size, which must be the mip's.
    pub fn write_texture_source(
        &mut self,
        id: TextureId,
        mip: u32,
        layer: u32,
        size: (u32, u32),
    ) -> Result<()> {
        let t = self.texture(id)?;
        if t.surface {
            return Err("write_texture_source: the canvas texture cannot be written".into());
        }
        t.desc.check_source_write(mip, layer, size)?;
        ffi::copy_source(t.js, (mip, layer), size)
    }

    pub fn generate_mips(&mut self, id: TextureId) -> Result<()> {
        let t = self.texture(id)?;
        let need = TextureUsage::COPY_SRC | TextureUsage::COPY_DST;
        if !t.desc.usage.contains(need) {
            return Err(format!("generate_mips: {id:?} needs COPY_SRC | COPY_DST"));
        }
        if t.desc.format.is_depth()
            || t.desc.format.is_compressed()
            || (is_float32(t.desc.format) && !self.info.float32_filterable)
        {
            return Err(format!(
                "generate_mips: {:?} cannot be filtered linearly on this device",
                t.desc.format
            ));
        }
        if t.desc.mips > 1 {
            ffi::generate_mips(t.js);
        }
        Ok(())
    }

    pub fn texture_desc(&self, id: TextureId) -> Result<TextureDesc> {
        Ok(self.texture(id)?.desc)
    }

    pub fn destroy_texture(&mut self, id: TextureId) {
        if self.texture(id).is_ok_and(|t| t.surface) {
            return;
        }
        if let Some(t) = self.textures.remove(id.raw()) {
            ffi::release(t.js);
        }
    }

    pub fn create_sampler(&mut self, d: &SamplerDesc) -> Result<SamplerId> {
        let code = |f: Filter| u32::from(f == Filter::Linear);
        let linear = d.filter == Filter::Linear && d.mip_filter == Filter::Linear;
        let anisotropy = if linear {
            d.anisotropy.clamp(1.0, 16.0) as u32
        } else {
            1
        };
        let key = [
            code(d.filter),
            code(d.mip_filter),
            match d.address {
                Address::Repeat => 0,
                Address::Clamp => 1,
                Address::Mirror => 2,
            },
            anisotropy,
            d.compare.map_or(0, |c| 1 + c as u32),
            d.max_lod.max(0.0).to_bits(),
        ];
        if let Some(&s) = self.sampler_cache.get(&key) {
            return Ok(s);
        }
        let js = ffi::create_sampler(&key)?;
        let id = SamplerId::from_raw(self.samplers.insert(Sampler {
            js,
            comparison: d.compare.is_some(),
        })?);
        self.sampler_cache.insert(key, id);
        Ok(id)
    }

    // ------------------------------------------------------------------ pipelines

    fn shader(&mut self, c: &Compiled) -> Result<(u32, String)> {
        let w = c.wgsl.as_ref().ok_or_else(|| {
            format!(
                "the {:?} shader has no WGSL; compile it with gpu::compile_shader \
                 (ganbate_shader::compile_wgsl) for WebGPU",
                c.stage
            )
        })?;
        if let Some(&m) = self.shader_cache.get(&w.source) {
            return Ok((m, w.entry.clone()));
        }
        let m = ffi::create_shader(&w.source)?;
        self.shader_cache.insert(w.source.clone(), m);
        Ok((m, w.entry.clone()))
    }

    fn group_layout(&mut self, entries: &[Entry]) -> Result<u32> {
        let mut words = Vec::with_capacity(entries.len() * 6);
        for e in entries {
            let (ty, a, b) = match e.kind {
                BindingKind::UniformBuffer { size } => (0, 0, size),
                BindingKind::StorageBuffer { read_only: true } => (1, 0, 0),
                BindingKind::StorageBuffer { read_only: false } => (2, 0, 0),
                BindingKind::Sampler { comparison } => (3, if comparison { 2 } else { 0 }, 0),
                BindingKind::Texture { view, sample } => {
                    let s = match sample {
                        SampleType::Float => 0,
                        SampleType::Depth => 2,
                        SampleType::Sint => 3,
                        SampleType::Uint => 4,
                    };
                    (4, s, view_code(view))
                }
                BindingKind::StorageTexture { view, format } => {
                    (5, format_code(texel_format(format)), view_code(view))
                }
            };
            words.extend([e.binding, e.visibility, ty, a, b, 0]);
        }
        if let Some(&l) = self.group_layouts.get(&words) {
            return Ok(l);
        }
        let l = ffi::create_bind_group_layout(&words)?;
        self.group_layouts.insert(words, l);
        Ok(l)
    }

    /// Slots (group layouts, empty gaps and the push-constant group), the pipeline layout and
    /// the push-constant size, from the reflection of every stage.
    fn layouts(&mut self, stages: &[(&Compiled, u32)]) -> Result<(Vec<Slot>, u32, u32)> {
        let mut groups: Vec<Vec<Entry>> = Vec::new();
        for (shader, bit) in stages {
            for b in &shader.bindings {
                let g = b.group as usize;
                if g >= MAX_GROUPS {
                    return Err(format!(
                        "@group({}) is above the 0..={} WebGPU supports",
                        b.group,
                        MAX_GROUPS - 1
                    ));
                }
                if g >= groups.len() {
                    groups.resize(g + 1, Vec::new());
                }
                match groups[g].iter_mut().find(|e| e.binding == b.binding) {
                    Some(e) => {
                        e.kind = match (e.kind, b.kind) {
                            (
                                BindingKind::UniformBuffer { size: x },
                                BindingKind::UniformBuffer { size: y },
                            ) => BindingKind::UniformBuffer { size: x.max(y) },
                            (
                                BindingKind::StorageBuffer { read_only: x },
                                BindingKind::StorageBuffer { read_only: y },
                            ) => BindingKind::StorageBuffer { read_only: x && y },
                            (x, y) if x == y => x,
                            (x, y) => {
                                return Err(format!(
                                    "@group({}) @binding({}) is declared as {x:?} and {y:?}",
                                    b.group, b.binding
                                ));
                            }
                        };
                        e.visibility |= bit;
                    }
                    None => groups[g].push(Entry {
                        binding: b.binding,
                        kind: b.kind,
                        visibility: *bit,
                    }),
                }
            }
        }
        for g in &mut groups {
            g.sort_by_key(|e| e.binding);
        }
        for e in groups.iter().flatten() {
            let writes = matches!(
                e.kind,
                BindingKind::StorageBuffer { read_only: false }
                    | BindingKind::StorageTexture { .. }
            );
            if writes && e.visibility & VISIBLE_VERTEX != 0 {
                return Err(format!(
                    "binding {} writes storage in a vertex shader, which WebGPU forbids",
                    e.binding
                ));
            }
        }
        let push = stages
            .iter()
            .map(|(s, _)| s.push_constant_size)
            .max()
            .unwrap_or(0)
            .next_multiple_of(4);
        if push > PUSH_SLOT {
            return Err(format!(
                "{push} bytes of push constants exceed the {PUSH_SLOT} bytes of the WebGPU ring"
            ));
        }
        if push > 0 {
            for (s, _) in stages {
                let lowered = s.wgsl.as_ref().and_then(|w| w.push_constants);
                if s.push_constant_size > 0
                    && lowered
                        != Some((
                            PUSH_GROUP as u32,
                            ganbate_shader::wgsl::PUSH_CONSTANT_BINDING,
                        ))
                {
                    return Err(format!(
                        "the {:?} shader's push constants are not at @group({PUSH_GROUP})",
                        s.stage
                    ));
                }
            }
            if groups.len() > PUSH_GROUP {
                return Err(format!(
                    "@group({PUSH_GROUP}) holds push constants on WebGPU; the shaders use it too"
                ));
            }
        }
        let mut slots = Vec::new();
        for entries in groups {
            slots.push(if entries.is_empty() {
                Slot::Empty
            } else {
                Slot::Group(GroupLayout {
                    js: self.group_layout(&entries)?,
                    entries,
                })
            });
        }
        let compute = stages.iter().any(|(s, _)| s.stage == Stage::Compute);
        if push > 0 {
            slots.resize(PUSH_GROUP, Slot::Empty);
            slots.push(Slot::Push);
        }
        let handles: Vec<u32> = slots
            .iter()
            .map(|s| match s {
                Slot::Empty => 0,
                Slot::Group(g) => g.js,
                Slot::Push if compute => ffi::PUSH_COMPUTE,
                Slot::Push => ffi::PUSH_RENDER,
            })
            .collect();
        let layout = match self.pipeline_layouts.get(&handles) {
            Some(&l) => l,
            None => {
                let l = ffi::create_pipeline_layout(&handles)?;
                self.pipeline_layouts.insert(handles, l);
                l
            }
        };
        Ok((slots, layout, push))
    }

    pub fn create_render_pipeline(&mut self, d: &RenderPipelineDesc) -> Result<PipelineId> {
        if d.vertex.stage != Stage::Vertex {
            return Err("create_render_pipeline: `vertex` is not a vertex shader".into());
        }
        if d.fragment.is_some_and(|f| f.stage != Stage::Fragment) {
            return Err("create_render_pipeline: `fragment` is not a fragment shader".into());
        }
        if d.colors.is_empty() && d.depth.is_none() {
            return Err("create_render_pipeline: no colour target and no depth".into());
        }
        if d.colors.len() > self.info.max_color_attachments as usize {
            return Err(format!(
                "{} colour targets exceed the device limit",
                d.colors.len()
            ));
        }
        for (i, c) in d.colors.iter().enumerate() {
            if c.format.is_depth() {
                return Err(format!(
                    "colour target {i}: {:?} is a depth format",
                    c.format
                ));
            }
            if c.blend != Blend::Replace && is_float32(c.format) && !self.info.float32_blendable {
                return Err(format!("colour target {i}: {:?} cannot blend", c.format));
            }
        }
        if d.depth.is_some_and(|z| !z.format.is_depth()) {
            return Err("create_render_pipeline: the depth format is not a depth format".into());
        }
        match d.fragment {
            None if !d.colors.is_empty() => {
                return Err("colour targets need a fragment shader".into());
            }
            Some(f) => {
                if let Some(o) = f.outputs.iter().find(|o| !is_float(o.ty)) {
                    return Err(format!(
                        "fragment @location({}) is {:?}; colour targets are float formats",
                        o.location, o.ty
                    ));
                }
                for (i, c) in d.colors.iter().enumerate() {
                    if c.write_mask & 0xF != 0 && !f.outputs.iter().any(|o| o.location == i as u32)
                    {
                        return Err(format!("colour target {i} has no @location({i}) output"));
                    }
                }
            }
            None => {}
        }
        let buffers = self.vertex_buffers(d)?;
        let mut stages = vec![(d.vertex, VISIBLE_VERTEX)];
        if let Some(f) = d.fragment {
            stages.push((f, VISIBLE_FRAGMENT));
        }
        let (slots, layout, push_size) = self.layouts(&stages)?;
        let (vs, vs_entry) = self.shader(d.vertex)?;
        let mut json = format!(
            "{{\"layout\":{layout},\"vertex\":{{\"module\":{vs},\"entryPoint\":\"{vs_entry}\",\"buffers\":{buffers}}}"
        );
        if let Some(f) = d.fragment {
            let (fs, fs_entry) = self.shader(f)?;
            let mut targets = Vec::new();
            for c in &d.colors {
                let blend = match c.blend {
                    Blend::Replace => String::new(),
                    b => {
                        let (sc, dc, sa, da) = match b {
                            Blend::Alpha => (
                                "src-alpha",
                                "one-minus-src-alpha",
                                "one",
                                "one-minus-src-alpha",
                            ),
                            Blend::Premultiplied => {
                                ("one", "one-minus-src-alpha", "one", "one-minus-src-alpha")
                            }
                            _ => ("one", "one", "one", "one"),
                        };
                        format!(
                            ",\"blend\":{{\"color\":{{\"srcFactor\":\"{sc}\",\"dstFactor\":\"{dc}\",\"operation\":\"add\"}},\"alpha\":{{\"srcFactor\":\"{sa}\",\"dstFactor\":\"{da}\",\"operation\":\"add\"}}}}"
                        )
                    }
                };
                targets.push(format!(
                    "{{\"format\":\"{}\",\"writeMask\":{}{blend}}}",
                    format_name(c.format),
                    c.write_mask & 0xF
                ));
            }
            let _ = write!(
                json,
                ",\"fragment\":{{\"module\":{fs},\"entryPoint\":\"{fs_entry}\",\"targets\":[{}]}}",
                targets.join(",")
            );
        }
        let triangles = d.topology != Topology::LineList;
        let _ = write!(
            json,
            ",\"primitive\":{{\"topology\":\"{}\",\"frontFace\":\"{}\",\"cullMode\":\"{}\"",
            match d.topology {
                Topology::TriangleList => "triangle-list",
                Topology::TriangleStrip => "triangle-strip",
                Topology::LineList => "line-list",
            },
            if d.front_ccw { "ccw" } else { "cw" },
            match d.cull {
                Cull::None => "none",
                Cull::Back => "back",
                Cull::Front => "front",
            }
        );
        if d.topology == Topology::TriangleStrip {
            json.push_str(",\"stripIndexFormat\":\"uint32\"");
        }
        if d.depth.is_some_and(|z| z.clamp) && self.info.depth_clip_control {
            json.push_str(",\"unclippedDepth\":true");
        }
        json.push('}');
        if let Some(z) = d.depth {
            let (bias, slope) = if triangles {
                (z.bias_constant.round() as i32, z.bias_slope)
            } else {
                (0, 0.0)
            };
            let _ = write!(
                json,
                ",\"depthStencil\":{{\"format\":\"{}\",\"depthWriteEnabled\":{},\"depthCompare\":\"{}\",\"depthBias\":{bias},\"depthBiasSlopeScale\":{slope:?},\"depthBiasClamp\":0}}",
                format_name(z.format),
                z.write,
                compare_name(z.compare)
            );
        }
        json.push('}');
        let js = ffi::create_render_pipeline(&json)?;
        self.insert_pipeline(Pipeline {
            js,
            compute: false,
            slots,
            push_size,
            colors: d.colors.iter().map(|c| c.format).collect(),
            depth: d.depth.map(|z| z.format),
            vertex_input: d.vertex_layout.is_some(),
        })
    }

    /// The `buffers` array of the vertex state, after checking the layout against the shader.
    fn vertex_buffers(&self, d: &RenderPipelineDesc) -> Result<String> {
        let shader = d.vertex;
        let Some(layout) = &d.vertex_layout else {
            if let Some(input) = shader.inputs.first() {
                return Err(format!(
                    "the vertex shader reads @location({}) but the pipeline has no vertex layout",
                    input.location
                ));
            }
            return Ok("[]".into());
        };
        if layout.stride == 0
            || layout.stride % 4 != 0
            || layout.stride > self.info.max_vertex_stride
        {
            return Err(format!("invalid vertex stride {}", layout.stride));
        }
        let mut attributes = Vec::new();
        for (i, a) in layout.attributes.iter().enumerate() {
            if layout.attributes[..i]
                .iter()
                .any(|b| b.location == a.location)
            {
                return Err(format!("vertex @location({}) is listed twice", a.location));
            }
            if a.location >= self.info.max_vertex_attributes
                || a.offset % 4 != 0
                || a.offset + a.format.size() > layout.stride
            {
                return Err(format!("vertex attribute {a:?} does not fit the layout"));
            }
            let format = match a.format {
                VertexFormat::F32 => "float32",
                VertexFormat::F32x2 => "float32x2",
                VertexFormat::F32x3 => "float32x3",
                VertexFormat::F32x4 => "float32x4",
                VertexFormat::U32 => "uint32",
                VertexFormat::Unorm8x4 => "unorm8x4",
            };
            attributes.push(format!(
                "{{\"shaderLocation\":{},\"offset\":{},\"format\":\"{format}\"}}",
                a.location, a.offset
            ));
        }
        for input in &shader.inputs {
            let a = layout
                .attributes
                .iter()
                .find(|a| a.location == input.location)
                .ok_or_else(|| {
                    format!(
                        "the vertex shader reads @location({}) but the layout has no such attribute",
                        input.location
                    )
                })?;
            let ok = if a.format == VertexFormat::U32 {
                is_uint(input.ty)
            } else {
                is_float(input.ty)
            };
            if !ok {
                return Err(format!(
                    "vertex @location({}) is {:?} in the shader but {:?} in the layout",
                    input.location, input.ty, a.format
                ));
            }
        }
        Ok(format!(
            "[{{\"arrayStride\":{},\"stepMode\":\"vertex\",\"attributes\":[{}]}}]",
            layout.stride,
            attributes.join(",")
        ))
    }

    pub fn create_compute_pipeline(&mut self, shader: &Compiled) -> Result<PipelineId> {
        if shader.stage != Stage::Compute {
            return Err("create_compute_pipeline: not a compute shader".into());
        }
        let (slots, layout, push_size) = self.layouts(&[(shader, VISIBLE_COMPUTE)])?;
        let (module, entry) = self.shader(shader)?;
        let json = format!(
            "{{\"layout\":{layout},\"compute\":{{\"module\":{module},\"entryPoint\":\"{entry}\"}}}}"
        );
        let js = ffi::create_compute_pipeline(&json)?;
        self.insert_pipeline(Pipeline {
            js,
            compute: true,
            slots,
            push_size,
            colors: Vec::new(),
            depth: None,
            vertex_input: false,
        })
    }

    fn insert_pipeline(&mut self, p: Pipeline) -> Result<PipelineId> {
        let js = p.js;
        match self.pipelines.insert(p) {
            Ok(raw) => Ok(PipelineId::from_raw(raw)),
            Err(e) => {
                ffi::release(js);
                Err(e)
            }
        }
    }

    // ------------------------------------------------------------------ bind groups

    pub fn create_bind_group(
        &mut self,
        pipeline: PipelineId,
        group: u32,
        entries: &[(u32, Resource)],
    ) -> Result<BindGroupId> {
        for (i, (binding, _)) in entries.iter().enumerate() {
            if entries[..i].iter().any(|(b, _)| b == binding) {
                return Err(format!("@binding({binding}) is given twice"));
            }
        }
        let mut inline_entries = [Entry {
            binding: 0,
            kind: BindingKind::Sampler { comparison: false },
            visibility: 0,
        }; 32];
        let mut overflow_entries = Vec::new();
        let (layout_js, entry_count) = match self.pipeline(pipeline)?.slots.get(group as usize) {
            Some(Slot::Group(l)) => {
                if l.entries.len() <= inline_entries.len() {
                    inline_entries[..l.entries.len()].copy_from_slice(&l.entries);
                } else {
                    overflow_entries.extend_from_slice(&l.entries);
                }
                (l.js, l.entries.len())
            }
            Some(Slot::Push) => {
                return Err(format!(
                    "@group({group}) holds the pipeline's push constants on WebGPU"
                ));
            }
            _ => {
                if group as usize >= MAX_GROUPS {
                    return Err(format!(
                        "@group({group}) is above the 0..={} WebGPU supports",
                        MAX_GROUPS - 1
                    ));
                }
                let empty = BindGroup {
                    js: ffi::EMPTY_GROUP,
                    layout_js: None,
                    bound: Vec::new(),
                };
                return Ok(BindGroupId::from_raw(self.groups.insert(empty)?));
            }
        };
        let layout_entries = if entry_count <= inline_entries.len() {
            &inline_entries[..entry_count]
        } else {
            overflow_entries.as_slice()
        };
        let mut words = Vec::with_capacity(entry_count * 8);
        let mut bound = Vec::with_capacity(entry_count);
        for e in layout_entries {
            let (_, res) = *entries
                .iter()
                .find(|(b, _)| *b == e.binding)
                .ok_or_else(|| format!("@group({group}) @binding({}) is missing", e.binding))?;
            let at = || format!("@group({group}) @binding({})", e.binding);
            match (e.kind, res) {
                (
                    BindingKind::UniformBuffer { .. } | BindingKind::StorageBuffer { .. },
                    Resource::Buffer { id, offset, size },
                ) => {
                    let b = self.buffer(id)?;
                    let uniform = matches!(e.kind, BindingKind::UniformBuffer { .. });
                    let (need, align, max) = if uniform {
                        (
                            BufferUsage::UNIFORM,
                            self.info.uniform_align,
                            self.info.max_uniform_binding,
                        )
                    } else {
                        (
                            BufferUsage::STORAGE,
                            self.info.storage_align,
                            self.info.max_storage_binding,
                        )
                    };
                    if !b.desc.usage.contains(need) {
                        return Err(format!("{}: {id:?} lacks {need:?} usage", at()));
                    }
                    if offset >= b.desc.size || offset % u64::from(align) != 0 {
                        return Err(format!(
                            "{}: offset {offset} must be inside the buffer and a multiple of {align}",
                            at()
                        ));
                    }
                    let range = if size == 0 {
                        b.desc.size - offset
                    } else {
                        size
                    };
                    // Buffers are allocated in whole words; storage bindings are word-sized.
                    let physical = b.desc.size.next_multiple_of(4);
                    let range = if uniform {
                        range
                    } else {
                        range.next_multiple_of(4)
                    };
                    if range == 0 || offset + range > physical || range > u64::from(max) {
                        return Err(format!(
                            "{}: range {range} at {offset} is out of bounds",
                            at()
                        ));
                    }
                    if let BindingKind::UniformBuffer { size: need } = e.kind
                        && range < u64::from(need)
                    {
                        return Err(format!(
                            "{}: the shader reads {need} bytes but only {range} are bound",
                            at()
                        ));
                    }
                    words.extend([
                        e.binding,
                        0,
                        b.js,
                        offset as u32,
                        (offset >> 32) as u32,
                        range as u32,
                        (range >> 32) as u32,
                        0,
                    ]);
                    bound.push(Bound::Buffer(id));
                }
                (
                    BindingKind::Texture { view, .. } | BindingKind::StorageTexture { view, .. },
                    Resource::Texture(id)
                    | Resource::TextureLayer(id, _)
                    | Resource::TextureMip(id, _),
                ) => {
                    let t = self.texture(id)?;
                    if t.surface {
                        return Err(format!("{}: the canvas texture cannot be bound", at()));
                    }
                    let desc = t.desc;
                    let storage = matches!(e.kind, BindingKind::StorageTexture { .. });
                    match e.kind {
                        BindingKind::Texture { sample, .. } => {
                            if !desc.usage.contains(TextureUsage::SAMPLED) {
                                return Err(format!("{}: {id:?} lacks SAMPLED usage", at()));
                            }
                            let ok = match sample {
                                SampleType::Depth => desc.format.is_depth(),
                                SampleType::Float => !desc.format.is_depth(),
                                SampleType::Sint | SampleType::Uint => false,
                            };
                            if !ok {
                                return Err(format!(
                                    "{}: a {sample:?} texture binding cannot sample {:?} on WebGPU",
                                    at(),
                                    desc.format
                                ));
                            }
                            if is_float32(desc.format) && !self.info.float32_filterable {
                                return Err(format!(
                                    "{}: {:?} is not filterable on this device",
                                    at(),
                                    desc.format
                                ));
                            }
                        }
                        BindingKind::StorageTexture { format, .. } => {
                            if !desc.usage.contains(TextureUsage::STORAGE) {
                                return Err(format!("{}: {id:?} lacks STORAGE usage", at()));
                            }
                            if texel_format(format) != desc.format {
                                return Err(format!(
                                    "{}: the shader writes {format:?} but {id:?} is {:?}",
                                    at(),
                                    desc.format
                                ));
                            }
                        }
                        _ => unreachable!("matched texture kinds above"),
                    }
                    let (mips, layers) = view_for(view, res, &desc, storage)
                        .map_err(|m| format!("{}: {m}", at()))?;
                    words.extend([
                        e.binding,
                        1,
                        t.js,
                        view_code(view),
                        mips.0,
                        mips.1 - mips.0,
                        layers.0,
                        layers.1 - layers.0,
                    ]);
                    bound.push(Bound::Texture { id, mips, layers });
                }
                (BindingKind::Sampler { comparison }, Resource::Sampler(id)) => {
                    let s = self.samplers.get(id.raw()).ok_or_else(|| {
                        format!("{}: {id:?} does not belong to this device", at())
                    })?;
                    if s.comparison != comparison {
                        return Err(format!(
                            "{}: the shader and {id:?} disagree on comparison sampling",
                            at()
                        ));
                    }
                    words.extend([e.binding, 2, s.js, 0, 0, 0, 0, 0]);
                    bound.push(Bound::Sampler(id));
                }
                (kind, res) => return Err(format!("{} expects {kind:?} but got {res:?}", at())),
            }
        }
        let js = ffi::create_bind_group(layout_js, &words)?;
        let g = BindGroup {
            js,
            layout_js: Some(layout_js),
            bound,
        };
        match self.groups.insert(g) {
            Ok(raw) => Ok(BindGroupId::from_raw(raw)),
            Err(e) => {
                ffi::release(js);
                Err(e)
            }
        }
    }

    pub fn destroy_bind_group(&mut self, id: BindGroupId) {
        if let Some(g) = self.groups.remove(id.raw()) {
            ffi::release(g.js);
        }
    }

    // ------------------------------------------------------------------ submission

    pub fn submit(&mut self, e: &crate::gpu::Encoder) -> Result<()> {
        let mut commands = core::mem::take(&mut self.cmd_buf);
        let mut ring = core::mem::take(&mut self.ring);
        let res = self.record(e, None, &mut commands, &mut ring);
        let out = res.and_then(|()| ffi::submit(&commands, &ring.bytes));
        self.cmd_buf = commands;
        self.ring = ring;
        out
    }

    /// The canvas texture, registered (or resized) to the canvas's current size.
    pub fn acquire(&mut self) -> Result<Option<TextureId>> {
        if !self.presentable {
            return Err("this device has no canvas; create it with Device::for_canvas".into());
        }
        let Some((width, height, code)) = ffi::canvas() else {
            return Err("the canvas is gone".into());
        };
        if width == 0 || height == 0 {
            return Ok(None);
        }
        let desc = TextureDesc::new(
            width,
            height,
            Self::canvas_format(code),
            TextureUsage::COLOR_TARGET | TextureUsage::COPY_SRC,
        );
        if let Some(id) = self.surface
            && let Some(t) = self.textures.get_mut(id.raw())
        {
            t.desc = desc;
            return Ok(Some(id));
        }
        let id = TextureId::from_raw(self.textures.insert(Texture {
            js: ffi::SURFACE,
            desc,
            sw_decompress: None,
            surface: true,
        })?);
        self.surface = Some(id);
        Ok(Some(id))
    }

    fn canvas_format(code: u32) -> Format {
        match code {
            1 => Format::Rgba8Unorm,
            2 => Format::Rgba8Srgb,
            4 => Format::Bgra8Srgb,
            _ => Format::Bgra8Unorm,
        }
    }

    pub fn surface_format(&self) -> Format {
        match ffi::canvas() {
            Some((_, _, code)) if self.presentable => Self::canvas_format(code),
            _ => Format::Bgra8Unorm,
        }
    }

    pub fn surface_size(&self) -> (u32, u32) {
        match ffi::canvas() {
            Some((w, h, _)) if self.presentable => (w, h),
            _ => (0, 0),
        }
    }

    pub fn present(&mut self, e: &crate::gpu::Encoder) -> Result<()> {
        let surface = self
            .surface
            .ok_or("present: call acquire first to get the canvas texture")?;
        let mut commands = core::mem::take(&mut self.cmd_buf);
        let mut ring = core::mem::take(&mut self.ring);
        let res = self.record(e, Some(surface), &mut commands, &mut ring);
        let out = res.and_then(|()| ffi::submit(&commands, &ring.bytes));
        self.cmd_buf = commands;
        self.ring = ring;
        out?;
        self.frame = (self.frame + 1) % 2;
        Ok(())
    }
}

impl Drop for Gpu<'_> {
    /// Hands every remaining object back to the host. WebGPU destroys buffers and textures only
    /// after submitted work that uses them is done, so this is safe while frames are in flight.
    fn drop(&mut self) {
        fn drain<T>(pool: &mut Pool<T>) -> impl Iterator<Item = T> + '_ {
            pool.free.clear();
            pool.slots.drain(..).filter_map(|s| s.1)
        }
        let mut handles: Vec<u32> = Vec::new();
        handles.extend(drain(&mut self.buffers).map(|b| b.js));
        handles.extend(drain(&mut self.textures).map(|t| t.js));
        handles.extend(drain(&mut self.samplers).map(|s| s.js));
        handles.extend(drain(&mut self.pipelines).map(|p| p.js));
        handles.extend(drain(&mut self.groups).map(|g| g.js));
        handles.extend(self.shader_cache.drain().map(|(_, h)| h));
        handles.extend(self.group_layouts.drain().map(|(_, h)| h));
        handles.extend(self.pipeline_layouts.drain().map(|(_, h)| h));
        for h in handles {
            ffi::release(h);
        }
    }
}

/// A half-open range of mips or layers.
type Span = (u32, u32);

/// The mip and layer ranges a binding of dimension `dim` views for `res`.
fn view_for(
    dim: TextureView,
    res: Resource,
    d: &TextureDesc,
    storage: bool,
) -> Result<(Span, Span)> {
    let (mips, layers) = match res {
        Resource::TextureLayer(_, l) if l < d.layers => ((0, d.mips), (l, l + 1)),
        Resource::TextureMip(_, m) if m < d.mips => ((m, m + 1), (0, d.layers)),
        Resource::Texture(_) => ((0, d.mips), (0, d.layers)),
        _ => return Err(format!("{res:?} is outside the texture")),
    };
    let count = layers.1 - layers.0;
    match dim {
        TextureView::D2 if count != 1 => {
            return Err(format!(
                "a 2D binding needs one layer but the texture has {count}; bind \
                 Resource::TextureLayer"
            ));
        }
        TextureView::Cube if count != 6 || d.width != d.height => {
            return Err("a cube binding needs 6 square layers".into());
        }
        _ => {}
    }
    if storage && mips.1 - mips.0 != 1 {
        return Err("storage bindings need a single mip; bind Resource::TextureMip".into());
    }
    Ok((mips, layers))
}
