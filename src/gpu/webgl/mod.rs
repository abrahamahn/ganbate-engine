//! WebGL 2 implementation of the backend-neutral GPU layer in [`crate::gpu`], for `wasm32`
//! builds in browsers without WebGPU. The JavaScript host (`web/ganbate.js`) creates the context
//! before the engine starts and provides the imports in [`ffi`].
//!
//! Shaders come from [`ganbate_shader::compile_glsl`] (see [`crate::gpu::compile_shader`]), whose
//! module documentation describes the lowering this backend follows: a render pipeline is one
//! linked program plus fixed-function state; each of its uniform blocks gets a binding point
//! (push constants at [`PUSH_POINT`]) and each combined sampler and storage buffer a texture
//! unit. Bind groups only record resources; at each draw the recorder resolves the pipeline's
//! blocks and units against the bound groups and emits the GL bindings that changed.
//!
//! Conventions: images lie in GL memory exactly as on Vulkan (row 0 at the top), because
//! shaders keep Vulkan's clip space apart from the depth range. Readbacks therefore need no
//! flip; the host flips the canvas texture when it presents it. Front faces wind clockwise in
//! GL's terms for `front_ccw` pipelines.
//!
//! Differences from the contract in [`crate::gpu`]:
//! * No compute pipelines, storage writes or storage textures (WebGL 2 has none). Storage
//!   buffers are read-only; the host mirrors each into an RGBA32UI texture.
//! * No texture views: a texture with one layer is a `TEXTURE_2D`, one with more layers a
//!   `TEXTURE_2D_ARRAY`; a binding must view it with that dimension (one layer of an array
//!   cannot be bound as a 2D texture, and cubes are not available). Mip ranges are applied
//!   through `TEXTURE_BASE_LEVEL` / `TEXTURE_MAX_LEVEL`, so one draw cannot view a texture's
//!   mips two ways.
//! * Colour targets with a zero write mask are left out with `drawBuffers`; the others share one
//!   blend mode and write mask unless the context has `OES_draw_buffers_indexed`.
//! * Index buffers cannot have other GPU usages, and copies between index and other buffers
//!   are not available.
//! * `read_buffer` works and is synchronous; `submit` and `present` do not wait.
//! * Rendering to `Rgba16Float`, `Rg16Float`, `R16Float`, `R32Float` and `Rgba32Float` needs
//!   `EXT_color_buffer_float`; linear filtering of 32-bit floats `OES_texture_float_linear`
//!   and blending them `EXT_float_blend`.
mod ffi;
mod record;

use crate::gpu::pool::Pool;
use crate::gpu::{
    Address, BindGroupId, Blend, BufferDesc, BufferId, BufferUsage, Compare, Cull, Filter, Format,
    Limits, Memory, PipelineId, RenderPipelineDesc, Resource, SamplerDesc, SamplerId, TextureDesc,
    TextureId, TextureUsage, Topology, VertexFormat,
};
use ganbate_shader::glsl::HiddenValue;
use ganbate_shader::{BindingKind, Compiled, IoType, SampleType, Stage, TextureView};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Write as _;

type Result<T> = std::result::Result<T, String>;

/// Bytes reserved per push-constant block in the ring; also the largest push-constant block.
pub(crate) const PUSH_SLOT: u32 = 256;
/// The uniform block binding point of the push constants.
pub(crate) const PUSH_POINT: u32 = 0;
/// Bind groups a pipeline may use.
const MAX_GROUPS: usize = 8;

/// A fragment shader for depth-only pipelines (GL programs need one).
const EMPTY_FS: &str = "#version 300 es\nprecision highp float;\nvoid main() {}\n";

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
fn is_float32(f: Format) -> bool {
    matches!(f, Format::R32Float | Format::Rgba32Float)
}
fn is_float(f: Format) -> bool {
    matches!(
        f,
        Format::Rgba16Float
            | Format::Rg16Float
            | Format::R16Float
            | Format::R32Float
            | Format::Rgba32Float
    )
}
fn compare_code(c: Compare) -> u32 {
    match c {
        Compare::Never => 1,
        Compare::Less => 2,
        Compare::LessEqual => 3,
        Compare::Equal => 4,
        Compare::Greater => 5,
        Compare::GreaterEqual => 6,
        Compare::NotEqual => 7,
        Compare::Always => 8,
    }
}
fn io_float(ty: IoType) -> bool {
    matches!(
        ty,
        IoType::F32 | IoType::Vec2F | IoType::Vec3F | IoType::Vec4F
    )
}
fn io_uint(ty: IoType) -> bool {
    matches!(
        ty,
        IoType::U32 | IoType::Vec2U | IoType::Vec3U | IoType::Vec4U
    )
}

struct Buffer {
    js: u32,
    desc: BufferDesc,
    /// Bound as `ELEMENT_ARRAY_BUFFER`; WebGL keeps such buffers apart from all others.
    index: bool,
}

struct Texture {
    /// [`ffi::SURFACE`] for the canvas texture.
    js: u32,
    desc: TextureDesc,
    sw_decompress: Option<ganbate_core::world::TextureCompression>,
    surface: bool,
    /// A `TEXTURE_2D_ARRAY` (more than one layer).
    array: bool,
}

struct Sampler {
    js: u32,
    comparison: bool,
}

/// A uniform block of a program.
#[derive(Clone, Debug)]
struct ProgramBlock {
    point: u32,
    /// `None` for the push constants.
    binding: Option<(u32, u32)>,
    size: u32,
}

/// A texture unit of a program: a combined sampler.
#[derive(Clone, Debug)]
struct ProgramUnit {
    unit: u32,
    texture: (u32, u32),
    /// `None`: the host's nearest, non-comparison sampler.
    sampler: Option<(u32, u32)>,
    view: TextureView,
}

/// A storage buffer read through its data texture on a texture unit.
#[derive(Clone, Debug)]
struct ProgramStorage {
    unit: u32,
    binding: (u32, u32),
}

/// A linked program and how its interface maps to bindings.
struct Program {
    js: u32,
    blocks: Vec<ProgramBlock>,
    units: Vec<ProgramUnit>,
    storage: Vec<ProgramStorage>,
    /// Hidden uniforms, by index in the host's table.
    hidden: Vec<HiddenValue>,
}

struct Pipeline {
    js: u32,
    program: usize,
    /// Reflected bindings per group: `(binding, kind)`, sorted by binding.
    groups: Vec<Vec<(u32, BindingKind)>>,
    group_ids: Vec<u32>,
    push_size: u32,
    colors: Vec<Format>,
    depth: Option<Format>,
    vertex_input: bool,
    stride: u32,
    /// The vertex layout's id (0: no vertex input); vertex arrays are shared per layout.
    layout: u32,
}

/// A resource a bind group references.
#[derive(Clone, Copy, Debug)]
enum Bound {
    Buffer {
        id: BufferId,
        offset: u64,
        size: u64,
    },
    Texture {
        id: TextureId,
        mips: (u32, u32),
        layers: (u32, u32),
    },
    Sampler(SamplerId),
}

struct BindGroup {
    /// Interned group layout id (`0` for an empty group).
    layout_id: u32,
    entries: Vec<(u32, Bound)>,
}

/// Limits and extensions of the WebGL 2 context.
#[derive(Clone, Copy, Debug, Default)]
struct DeviceInfo {
    max_texture_2d: u32,
    max_layers: u32,
    max_uniform_block: u32,
    uniform_align: u32,
    max_combined_units: u32,
    max_fragment_units: u32,
    max_vertex_units: u32,
    max_draw_buffers: u32,
    max_vertex_attributes: u32,
    max_uniform_points: u32,
    color_float: bool,
    float_linear: bool,
    float_blend: bool,
    /// `OES_draw_buffers_indexed`: blend state and write masks per colour target.
    indexed_buffers: bool,
    s3tc: bool,
    bptc: bool,
    astc: bool,
    max_anisotropy: f32,
}

/// Key of a cached vertex array: pipeline (vertex layout), buffer, byte offset, index buffer.
type VaoKey = (u32, u32, u32, u32);

pub(crate) struct Gpu<'w> {
    name: String,
    info: DeviceInfo,
    buffers: Pool<Buffer>,
    textures: Pool<Texture>,
    samplers: Pool<Sampler>,
    pipelines: Pool<Pipeline>,
    groups: Pool<BindGroup>,
    programs: Vec<Program>,
    program_cache: HashMap<(String, String), usize>,
    sampler_cache: HashMap<[u32; 6], SamplerId>,
    /// Vertex layouts (stride and attribute words) by id.
    layouts: HashMap<Vec<u32>, u32>,
    layout_ids: HashMap<Vec<(u32, BindingKind)>, u32>,
    cmd_buf: Vec<u32>,
    ring: record::Ring,
    /// Vertex arrays, created while recording.
    vaos: RefCell<HashMap<VaoKey, u32>>,
    /// The canvas texture, once `acquire` has registered it.
    surface: Option<TextureId>,
    presentable: bool,
    frame: usize,
    _window: core::marker::PhantomData<&'w ()>,
}

impl Gpu<'static> {
    /// The context the host created; `presentable` when it belongs to a canvas.
    pub fn open(presentable: bool) -> Result<Self> {
        let w = ffi::info();
        if w.len() < 16 {
            return Err(format!(
                "the WebGL 2 host is not ready ({}); load the module through web/ganbate.js",
                ffi::last_error()
            ));
        }
        let info = DeviceInfo {
            max_texture_2d: w[0],
            max_layers: w[1],
            max_uniform_block: w[2],
            uniform_align: w[3].max(1),
            max_combined_units: w[4],
            max_fragment_units: w[5],
            max_vertex_units: w[6],
            max_draw_buffers: w[7],
            max_vertex_attributes: w[8],
            max_uniform_points: w[9],
            color_float: w[10] & 1 != 0,
            float_linear: w[10] & 2 != 0,
            float_blend: w[10] & 4 != 0,
            indexed_buffers: w[10] & 8 != 0,
            s3tc: w[10] & 16 != 0,
            bptc: w[10] & 32 != 0,
            astc: w[10] & 64 != 0,
            max_anisotropy: f32::from_bits(w[11]).max(1.0),
        };
        if presentable && ffi::canvas().is_none() {
            return Err("the host has no canvas to present to".into());
        }
        Ok(Self {
            name: ffi::renderer_name(),
            info,
            buffers: Pool::default(),
            textures: Pool::default(),
            samplers: Pool::default(),
            pipelines: Pool::default(),
            groups: Pool::default(),
            programs: Vec::new(),
            program_cache: HashMap::new(),
            sampler_cache: HashMap::new(),
            layouts: HashMap::new(),
            layout_ids: HashMap::new(),
            cmd_buf: Vec::new(),
            ring: record::Ring::default(),
            vaos: RefCell::new(HashMap::new()),
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
            max_anisotropy: self.info.max_anisotropy,
            depth_clamp: false,
            uniform_offset_alignment: u64::from(self.info.uniform_align),
            storage_offset_alignment: 16,
            depth_filterable: false,
            texture_compression_bc: self.info.s3tc && self.info.bptc,
            texture_compression_astc: self.info.astc,
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
    fn sampler(&self, id: SamplerId) -> Result<&Sampler> {
        self.samplers
            .get(id.raw())
            .ok_or_else(|| format!("{id:?} does not belong to this device"))
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
        if size > u64::from(u32::MAX) {
            return Err(format!(
                "create_buffer: {size} bytes are too many for WebGL"
            ));
        }
        let index = d.usage.contains(BufferUsage::INDEX);
        let kinds = if d.memory == Memory::Readback {
            if d.usage.bits() & !BufferUsage::COPY_DST.bits() != 0 {
                return Err("create_buffer: readback buffers can only be copy destinations".into());
            }
            ffi::BUFFER_READBACK
        } else {
            let others = BufferUsage::VERTEX.bits()
                | BufferUsage::UNIFORM.bits()
                | BufferUsage::STORAGE.bits();
            if index && d.usage.bits() & others != 0 {
                return Err(
                    "create_buffer: WebGL 2 keeps index buffers apart from vertex, uniform and \
                     storage buffers"
                        .into(),
                );
            }
            let mut kinds = 0;
            // Storage alone lives in the host's data texture; anything else needs a GL buffer.
            let gl = BufferUsage::VERTEX.bits()
                | BufferUsage::UNIFORM.bits()
                | BufferUsage::COPY_SRC.bits();
            if index {
                kinds |= ffi::BUFFER_INDEX;
            } else if d.usage.bits() & gl != 0 || !d.usage.contains(BufferUsage::STORAGE) {
                kinds |= ffi::BUFFER_DATA;
            }
            if d.usage.contains(BufferUsage::STORAGE) {
                kinds |= ffi::BUFFER_STORAGE;
            }
            kinds
        };
        let js = ffi::create_buffer(size, kinds)?;
        match self.buffers.insert(Buffer {
            js,
            desc: *d,
            index,
        }) {
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
        if !data.is_empty() {
            ffi::write_buffer(b.js, offset, data);
        }
        Ok(())
    }

    pub fn read_buffer(&self, id: BufferId) -> Result<Vec<u8>> {
        let b = self.buffer(id)?;
        if b.desc.memory != Memory::Readback {
            return Err(format!("read_buffer: {id:?} is not a readback buffer"));
        }
        ffi::read_buffer(b.js, b.desc.size as usize)
    }

    pub fn destroy_buffer(&mut self, id: BufferId) {
        if let Some(b) = self.buffers.remove(id.raw()) {
            let vaos = self.vaos.get_mut();
            let gone: Vec<VaoKey> = vaos
                .keys()
                .filter(|k| k.1 == b.js || k.3 == b.js)
                .copied()
                .collect();
            for k in gone {
                if let Some(v) = vaos.remove(&k) {
                    ffi::release(v);
                }
            }
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
        if d.usage.contains(TextureUsage::STORAGE) {
            return Err("create_texture: WebGL 2 has no storage textures".into());
        }
        let mut sw_decompress = None;
        let actual_format = if let Some(comp) = d.format.compression() {
            let hw = match comp {
                ganbate_core::world::TextureCompression::Bc1 => self.info.s3tc,
                ganbate_core::world::TextureCompression::Bc7 => self.info.bptc,
                ganbate_core::world::TextureCompression::Astc4x4 => self.info.astc,
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
        let renders = d.usage.contains(TextureUsage::COLOR_TARGET)
            || (d.usage.contains(TextureUsage::COPY_DST)
                && !depth
                && !actual_format.is_compressed())
            || (d.mips > 1 && d.usage.contains(TextureUsage::COPY_SRC));
        if renders && is_float(actual_format) && !self.info.color_float {
            return Err(format!(
                "create_texture: rendering to {:?} needs EXT_color_buffer_float",
                d.format
            ));
        }
        let array = d.layers > 1;
        let js = ffi::create_texture(
            (d.width, d.height),
            d.layers,
            d.mips,
            format_code(actual_format),
            array,
        )?;
        match self.textures.insert(Texture {
            js,
            desc: *d,
            sw_decompress,
            surface: false,
            array,
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
            return Err("write_texture: WebGL 2 cannot upload depth textures".into());
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
        let (js, sw) = (t.js, t.sw_decompress);
        if let Some(comp) = sw {
            let decoded = ganbate_assets::bc::decode_rgba8(comp, data, w, h)?;
            ffi::write_texture(js, (mip, layer), (w, h), &decoded);
        } else {
            ffi::write_texture(js, (mip, layer), (w, h), data);
        }
        Ok(())
    }

    /// Uploads the host's current source image (see `ffi::write_texture_source`) into one whole
    /// subresource; `size` is the source's size, which must be the mip's. A source arrives in
    /// RGBA order, so BGRA textures (stored as RGBA8 here) need no swizzle.
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
        ffi::write_texture_source(t.js, (mip, layer), size)
    }

    pub fn generate_mips(&mut self, id: TextureId) -> Result<()> {
        let t = self.texture(id)?;
        let need = TextureUsage::COPY_SRC | TextureUsage::COPY_DST;
        if !t.desc.usage.contains(need) {
            return Err(format!("generate_mips: {id:?} needs COPY_SRC | COPY_DST"));
        }
        if t.desc.format.is_depth()
            || t.desc.format.is_compressed()
            || (is_float32(t.desc.format) && !self.info.float_linear)
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
            d.anisotropy.clamp(1.0, self.info.max_anisotropy) as u32
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
            d.compare.map_or(0, compare_code),
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

    /// The linked program for a vertex and fragment shader, created once per pair. Uniform
    /// blocks get binding points (push constants [`PUSH_POINT`]), combined samplers and then
    /// storage buffers get texture units, hidden uniforms an index; the host receives that
    /// assignment as lines `b NAME POINT`, `u NAME UNIT`, `h NAME INDEX TYPE`.
    fn program(&mut self, vs: &Compiled, fs: Option<&Compiled>) -> Result<usize> {
        fn glsl(c: &Compiled) -> Result<&ganbate_shader::glsl::Glsl> {
            c.glsl.as_ref().ok_or_else(|| {
                format!(
                    "the {:?} shader has no GLSL; compile it with gpu::compile_shader \
                     (ganbate_shader::compile_glsl) for WebGL 2",
                    c.stage
                )
            })
        }
        let v = glsl(vs)?;
        let f = fs.map(glsl).transpose()?;
        let key = (
            v.source.clone(),
            f.map_or_else(|| EMPTY_FS.to_string(), |f| f.source.clone()),
        );
        if let Some(&p) = self.program_cache.get(&key) {
            return Ok(p);
        }
        let stages: Vec<&ganbate_shader::glsl::Glsl> = std::iter::once(v).chain(f).collect();
        let mut meta = String::new();
        let mut blocks: Vec<(String, ProgramBlock)> = Vec::new();
        let mut units: Vec<(String, ProgramUnit)> = Vec::new();
        let mut storage: Vec<(String, ProgramStorage)> = Vec::new();
        let mut hidden: Vec<(String, HiddenValue)> = Vec::new();
        for g in &stages {
            for b in &g.blocks {
                if !blocks.iter().any(|(n, _)| *n == b.name) {
                    let point = if b.binding.is_none() {
                        PUSH_POINT
                    } else {
                        1 + blocks.iter().filter(|(_, x)| x.binding.is_some()).count() as u32
                    };
                    blocks.push((
                        b.name.clone(),
                        ProgramBlock {
                            point,
                            binding: b.binding,
                            size: b.size,
                        },
                    ));
                }
            }
            for u in &g.units {
                if !units.iter().any(|(n, _)| *n == u.name) {
                    let unit = units.len() as u32;
                    units.push((
                        u.name.clone(),
                        ProgramUnit {
                            unit,
                            texture: u.texture,
                            sampler: u.sampler,
                            view: u.view,
                        },
                    ));
                }
            }
            for h in &g.hidden {
                if !hidden.iter().any(|(n, _)| *n == h.name) {
                    hidden.push((h.name.clone(), h.value));
                }
            }
        }
        for g in &stages {
            for s in &g.storage {
                if !storage.iter().any(|(n, _)| *n == s.name) {
                    let unit = (units.len() + storage.len()) as u32;
                    storage.push((
                        s.name.clone(),
                        ProgramStorage {
                            unit,
                            binding: s.binding,
                        },
                    ));
                }
            }
        }
        let total = (units.len() + storage.len()) as u32;
        if total > self.info.max_combined_units {
            return Err(format!(
                "the pipeline reads {total} textures; this device binds {}",
                self.info.max_combined_units
            ));
        }
        for (g, max, stage) in [
            (Some(v), self.info.max_vertex_units, "vertex"),
            (f, self.info.max_fragment_units, "fragment"),
        ] {
            if let Some(g) = g {
                let n = (g.units.len() + g.storage.len()) as u32;
                if n > max {
                    return Err(format!(
                        "the {stage} shader reads {n} textures; this device binds {max}"
                    ));
                }
            }
        }
        if blocks.len() as u32 + 1 > self.info.max_uniform_points {
            return Err("too many uniform blocks for this device".into());
        }
        for (name, b) in &blocks {
            if b.size > self.info.max_uniform_block {
                return Err(format!(
                    "uniform block {name} has {} bytes; this device allows {}",
                    b.size, self.info.max_uniform_block
                ));
            }
            let _ = writeln!(meta, "b {name} {}", b.point);
        }
        for (name, u) in &units {
            let _ = writeln!(meta, "u {name} {}", u.unit);
        }
        for (name, s) in &storage {
            let _ = writeln!(meta, "u {name} {}", s.unit);
        }
        for (i, (name, h)) in hidden.iter().enumerate() {
            let _ = writeln!(meta, "h {name} {i} {}", h.glsl_type());
        }
        let js = ffi::create_program(&key.0, &key.1, &meta)?;
        self.programs.push(Program {
            js,
            blocks: blocks.into_iter().map(|(_, b)| b).collect(),
            units: units.into_iter().map(|(_, u)| u).collect(),
            storage: storage.into_iter().map(|(_, s)| s).collect(),
            hidden: hidden.into_iter().map(|(_, h)| h).collect(),
        });
        let index = self.programs.len() - 1;
        self.program_cache.insert(key, index);
        Ok(index)
    }

    /// Reflected bindings of every stage, merged per group.
    fn reflect(stages: &[&Compiled]) -> Result<Vec<Vec<(u32, BindingKind)>>> {
        let mut groups: Vec<Vec<(u32, BindingKind)>> = Vec::new();
        for shader in stages {
            for b in &shader.bindings {
                let g = b.group as usize;
                if g >= MAX_GROUPS {
                    return Err(format!(
                        "@group({}) is above the 0..={} the WebGL backend supports",
                        b.group,
                        MAX_GROUPS - 1
                    ));
                }
                if g >= groups.len() {
                    groups.resize(g + 1, Vec::new());
                }
                match groups[g].iter_mut().find(|(n, _)| *n == b.binding) {
                    Some((_, kind)) => {
                        *kind = match (*kind, b.kind) {
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
                        }
                    }
                    None => groups[g].push((b.binding, b.kind)),
                }
            }
        }
        for g in &mut groups {
            g.sort_by_key(|(n, _)| *n);
        }
        Ok(groups)
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
        if d.colors.len() > self.info.max_draw_buffers as usize {
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
            if c.blend != Blend::Replace && is_float32(c.format) && !self.info.float_blend {
                return Err(format!("colour target {i}: {:?} cannot blend", c.format));
            }
            // Targets that are not written are left out with drawBuffers; the others share
            // one blend mode and write mask unless OES_draw_buffers_indexed is there.
            let first = d.colors.iter().find(|x| x.write_mask & 0xF != 0);
            if c.write_mask & 0xF != 0
                && let Some(f) = first
                && (c.blend != f.blend || c.write_mask & 0xF != f.write_mask & 0xF)
                && !self.info.indexed_buffers
            {
                return Err(
                    "this WebGL 2 context applies one blend mode and write mask to every \
                     written colour target (OES_draw_buffers_indexed is missing)"
                        .into(),
                );
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
                if let Some(o) = f.outputs.iter().find(|o| !io_float(o.ty)) {
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
        let (stride, attributes) = self.vertex_layout(d)?;
        let mut stages = vec![d.vertex];
        if let Some(f) = d.fragment {
            stages.push(f);
        }
        let groups = Self::reflect(&stages)?;
        let mut group_ids = Vec::with_capacity(groups.len());
        for g in &groups {
            let gid = if g.is_empty() {
                0
            } else if let Some(&id) = self.layout_ids.get(g) {
                id
            } else {
                let id = self.layout_ids.len() as u32 + 1;
                self.layout_ids.insert(g.clone(), id);
                id
            };
            group_ids.push(gid);
        }
        let push_size = stages
            .iter()
            .map(|s| s.push_constant_size)
            .max()
            .unwrap_or(0)
            .next_multiple_of(4);
        if push_size > PUSH_SLOT {
            return Err(format!(
                "{push_size} bytes of push constants exceed the {PUSH_SLOT} bytes of the ring"
            ));
        }
        let program = self.program(d.vertex, d.fragment)?;
        let (bias, slope) = match d.depth {
            Some(z) if d.topology != Topology::LineList => (z.bias_constant, z.bias_slope),
            _ => (0.0, 0.0),
        };
        let layout = if d.vertex_layout.is_some() {
            let mut key = vec![stride];
            key.extend(&attributes);
            let next = self.layouts.len() as u32 + 1;
            *self.layouts.entry(key).or_insert(next)
        } else {
            0
        };
        let mut state = vec![
            match d.topology {
                Topology::TriangleList => 0,
                Topology::TriangleStrip => 1,
                Topology::LineList => 2,
            },
            match d.cull {
                Cull::None => 0,
                Cull::Back => 1,
                Cull::Front => 2,
            },
            u32::from(d.front_ccw),
            u32::from(d.depth.is_some()),
            d.depth.map_or(8, |z| compare_code(z.compare)),
            u32::from(d.depth.is_some_and(|z| z.write)),
            bias.to_bits(),
            slope.to_bits(),
            d.colors.len() as u32,
        ];
        for c in &d.colors {
            state.push(match c.blend {
                Blend::Replace => 0,
                Blend::Alpha => 1,
                Blend::Premultiplied => 2,
                Blend::Additive => 3,
            });
            state.push(u32::from(c.write_mask & 0xF));
        }
        state.extend([stride, (attributes.len() / 3) as u32]);
        state.extend(attributes);
        let js = ffi::create_pipeline(self.programs[program].js, &state)?;
        let p = Pipeline {
            js,
            program,
            groups,
            group_ids,
            push_size,
            colors: d.colors.iter().map(|c| c.format).collect(),
            depth: d.depth.map(|z| z.format),
            vertex_input: d.vertex_layout.is_some(),
            stride,
            layout,
        };
        match self.pipelines.insert(p) {
            Ok(raw) => Ok(PipelineId::from_raw(raw)),
            Err(e) => {
                ffi::release(js);
                Err(e)
            }
        }
    }

    /// The vertex stride and `(location, format code, offset)` words, after checking the
    /// layout against the shader.
    fn vertex_layout(&self, d: &RenderPipelineDesc) -> Result<(u32, Vec<u32>)> {
        let shader = d.vertex;
        let Some(layout) = &d.vertex_layout else {
            if let Some(input) = shader.inputs.first() {
                return Err(format!(
                    "the vertex shader reads @location({}) but the pipeline has no vertex layout",
                    input.location
                ));
            }
            return Ok((0, Vec::new()));
        };
        if layout.stride == 0 || layout.stride % 4 != 0 || layout.stride > 255 {
            return Err(format!("invalid vertex stride {}", layout.stride));
        }
        let mut words = Vec::new();
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
            // Only the attributes the shader reads are enabled.
            if !shader.inputs.iter().any(|x| x.location == a.location) {
                continue;
            }
            let format = match a.format {
                VertexFormat::F32 => 1,
                VertexFormat::F32x2 => 2,
                VertexFormat::F32x3 => 3,
                VertexFormat::F32x4 => 4,
                VertexFormat::U32 => 5,
                VertexFormat::Unorm8x4 => 6,
            };
            words.extend([a.location, format, a.offset]);
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
                io_uint(input.ty)
            } else {
                io_float(input.ty)
            };
            if !ok {
                return Err(format!(
                    "vertex @location({}) is {:?} in the shader but {:?} in the layout",
                    input.location, input.ty, a.format
                ));
            }
        }
        Ok((layout.stride, words))
    }

    pub fn create_compute_pipeline(&mut self, shader: &Compiled) -> Result<PipelineId> {
        let _ = shader;
        Err("create_compute_pipeline: WebGL 2 has no compute shaders".into())
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
        if group as usize >= MAX_GROUPS {
            return Err(format!(
                "@group({group}) is above the 0..={} the WebGL backend supports",
                MAX_GROUPS - 1
            ));
        }
        let mut inline_layout = [(0u32, BindingKind::Sampler { comparison: false }); 32];
        let mut overflow_layout = Vec::new();
        let (layout_id, entry_count) = {
            let p = self.pipeline(pipeline)?;
            match p
                .groups
                .get(group as usize)
                .filter(|l| !l.is_empty())
                .map(|l| (p.group_ids[group as usize], l))
            {
                Some((id, l)) => {
                    if l.len() <= inline_layout.len() {
                        inline_layout[..l.len()].copy_from_slice(l);
                    } else {
                        overflow_layout.extend_from_slice(l);
                    }
                    (id, l.len())
                }
                None => (0, 0),
            }
        };
        if entry_count == 0 {
            let empty = BindGroup {
                layout_id: 0,
                entries: Vec::new(),
            };
            return Ok(BindGroupId::from_raw(self.groups.insert(empty)?));
        }
        let layout = if entry_count <= inline_layout.len() {
            &inline_layout[..entry_count]
        } else {
            overflow_layout.as_slice()
        };
        let mut bound = Vec::with_capacity(entry_count);
        for &(binding, kind) in layout {
            let (_, res) = *entries
                .iter()
                .find(|(b, _)| *b == binding)
                .ok_or_else(|| format!("@group({group}) @binding({binding}) is missing"))?;
            let at = || format!("@group({group}) @binding({binding})");
            let entry = match (kind, res) {
                (
                    BindingKind::UniformBuffer { .. } | BindingKind::StorageBuffer { .. },
                    Resource::Buffer { id, offset, size },
                ) => {
                    let b = self.buffer(id)?;
                    let uniform = matches!(kind, BindingKind::UniformBuffer { .. });
                    let (need, align) = if uniform {
                        (BufferUsage::UNIFORM, u64::from(self.info.uniform_align))
                    } else {
                        (BufferUsage::STORAGE, 16)
                    };
                    if !b.desc.usage.contains(need) {
                        return Err(format!("{}: {id:?} lacks {need:?} usage", at()));
                    }
                    if offset >= b.desc.size || offset % align != 0 {
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
                    if range == 0 || offset + range > b.desc.size.next_multiple_of(4) {
                        return Err(format!(
                            "{}: range {range} at {offset} is out of bounds",
                            at()
                        ));
                    }
                    if let BindingKind::UniformBuffer { size: need } = kind {
                        if range < u64::from(need) {
                            return Err(format!(
                                "{}: the shader reads {need} bytes but only {range} are bound",
                                at()
                            ));
                        }
                        if range > u64::from(self.info.max_uniform_block) {
                            return Err(format!(
                                "{}: {range} bytes exceed the device's uniform block size {}",
                                at(),
                                self.info.max_uniform_block
                            ));
                        }
                    }
                    Bound::Buffer {
                        id,
                        offset,
                        size: range,
                    }
                }
                (
                    BindingKind::Texture { view, sample },
                    Resource::Texture(id)
                    | Resource::TextureLayer(id, _)
                    | Resource::TextureMip(id, _),
                ) => {
                    let t = self.texture(id)?;
                    if t.surface {
                        return Err(format!("{}: the canvas texture cannot be bound", at()));
                    }
                    let desc = t.desc;
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
                            "{}: a {sample:?} texture binding cannot sample {:?}",
                            at(),
                            desc.format
                        ));
                    }
                    let (mips, layers) = match res {
                        Resource::TextureLayer(_, l) if l < desc.layers => {
                            ((0, desc.mips), (l, l + 1))
                        }
                        Resource::TextureMip(_, m) if m < desc.mips => {
                            ((m, m + 1), (0, desc.layers))
                        }
                        Resource::Texture(_) => ((0, desc.mips), (0, desc.layers)),
                        _ => return Err(format!("{}: {res:?} is outside the texture", at())),
                    };
                    match view {
                        TextureView::D2 if t.array => {
                            return Err(format!(
                                "{}: WebGL 2 cannot view one layer of an array texture as a 2D \
                                 texture (it has no texture views)",
                                at()
                            ));
                        }
                        TextureView::D2Array if !t.array => {
                            return Err(format!(
                                "{}: a 2D array binding needs a texture with more than one layer \
                                 on WebGL 2",
                                at()
                            ));
                        }
                        TextureView::D2Array if layers != (0, desc.layers) => {
                            return Err(format!(
                                "{}: WebGL 2 binds every layer of an array texture",
                                at()
                            ));
                        }
                        TextureView::Cube => {
                            return Err(format!(
                                "{}: WebGL 2 cube textures are not supported",
                                at()
                            ));
                        }
                        _ => {}
                    }
                    Bound::Texture { id, mips, layers }
                }
                (BindingKind::Sampler { comparison }, Resource::Sampler(id)) => {
                    let s = self.sampler(id).map_err(|e| format!("{}: {e}", at()))?;
                    if s.comparison != comparison {
                        return Err(format!(
                            "{}: the shader and {id:?} disagree on comparison sampling",
                            at()
                        ));
                    }
                    Bound::Sampler(id)
                }
                (BindingKind::StorageTexture { .. }, _) => {
                    return Err(format!("{}: WebGL 2 has no storage textures", at()));
                }
                (kind, res) => return Err(format!("{} expects {kind:?} but got {res:?}", at())),
            };
            bound.push((binding, entry));
        }
        let g = BindGroup {
            layout_id,
            entries: bound,
        };
        Ok(BindGroupId::from_raw(self.groups.insert(g)?))
    }

    pub fn destroy_bind_group(&mut self, id: BindGroupId) {
        self.groups.remove(id.raw());
    }

    // ------------------------------------------------------------------ submission

    pub fn submit(&mut self, e: &crate::gpu::Encoder) -> Result<()> {
        let mut commands = core::mem::take(&mut self.cmd_buf);
        let mut ring = core::mem::take(&mut self.ring);
        let res = self.record(e, None, &mut commands, &mut ring);
        let out = res.and_then(|()| ffi::submit(&commands, &ring.bytes, false));
        self.cmd_buf = commands;
        self.ring = ring;
        out
    }

    /// The canvas texture, registered (or resized) to the canvas's current size.
    pub fn acquire(&mut self) -> Result<Option<TextureId>> {
        if !self.presentable {
            return Err("this device has no canvas; create it with Device::for_canvas".into());
        }
        let Some((width, height)) = ffi::canvas() else {
            return Err("the canvas is gone".into());
        };
        if width == 0 || height == 0 {
            return Ok(None);
        }
        let desc = TextureDesc::new(
            width,
            height,
            Format::Rgba8Unorm,
            TextureUsage::COLOR_TARGET | TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
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
            array: false,
        })?);
        self.surface = Some(id);
        Ok(Some(id))
    }

    /// `Rgba8Unorm`: the canvas texture is RGBA8 (the renderer encodes sRGB itself).
    pub fn surface_format(&self) -> Format {
        Format::Rgba8Unorm
    }

    pub fn surface_size(&self) -> (u32, u32) {
        match ffi::canvas() {
            Some(s) if self.presentable => s,
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
        let out = res.and_then(|()| ffi::submit(&commands, &ring.bytes, true));
        self.cmd_buf = commands;
        self.ring = ring;
        out?;
        self.frame = (self.frame + 1) % 2;
        Ok(())
    }
}

impl Drop for Gpu<'_> {
    /// Hands every remaining object back to the host.
    fn drop(&mut self) {
        fn drain<T>(pool: &mut Pool<T>) -> impl Iterator<Item = T> + '_ {
            pool.free.clear();
            pool.slots.drain(..).filter_map(|s| s.1)
        }
        let mut handles: Vec<u32> = Vec::new();
        handles.extend(self.vaos.get_mut().drain().map(|(_, h)| h));
        handles.extend(drain(&mut self.pipelines).map(|p| p.js));
        handles.extend(self.programs.drain(..).map(|p| p.js));
        handles.extend(drain(&mut self.buffers).map(|b| b.js));
        handles.extend(drain(&mut self.textures).map(|t| t.js));
        handles.extend(drain(&mut self.samplers).map(|s| s.js));
        for h in handles {
            ffi::release(h);
        }
    }
}
