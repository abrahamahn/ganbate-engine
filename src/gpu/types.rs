//! Backend-neutral handles and resource descriptions.
use core::ops::{BitOr, BitOrAssign};

/// GPU errors are human-readable strings, like the rest of the engine.
pub type Result<T> = std::result::Result<T, String>;

macro_rules! handle {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        ///
        /// A small `Copy` handle into a pool owned by the [`Device`](super::Device) that created
        /// it. Handles carry a generation: using one after its object was destroyed is reported
        /// as an error instead of touching a recycled object.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(u32);
        impl $name {
            pub(crate) const fn from_raw(raw: u32) -> Self {
                Self(raw)
            }
            pub(crate) const fn raw(self) -> u32 {
                self.0
            }
        }
    };
}
handle!(
    /// A buffer.
    BufferId
);
handle!(
    /// A texture, including the swapchain textures returned by `Device::acquire`.
    TextureId
);
handle!(
    /// An immutable sampler.
    SamplerId
);
handle!(
    /// A render or compute pipeline.
    PipelineId
);
handle!(
    /// A set of resources bound to one `@group` of a pipeline.
    BindGroupId
);

/// Texel formats. Every format is sampled as floating point; `Depth32Float` is the only depth
/// format and is always usable as a sampled depth attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    Rgba8Unorm,
    Rgba8Srgb,
    Bgra8Unorm,
    Bgra8Srgb,
    Rgba16Float,
    Rg16Float,
    R16Float,
    R8Unorm,
    R32Float,
    Rgba32Float,
    Depth32Float,
    /// 4x4 block-compressed BC1 / DXT1 (8 bytes per 4x4 block, 4 bpp, 1-bit punch-through alpha).
    Bc1RgbaUnorm,
    Bc1RgbaSrgb,
    /// 4x4 block-compressed BC7 / BPTC (16 bytes per 4x4 block, 8 bpp, high-precision RGBA).
    Bc7RgbaUnorm,
    Bc7RgbaSrgb,
    /// 4x4 block-compressed ASTC LDR (16 bytes per 4x4 block, 8 bpp, mobile/Apple RGBA).
    Astc4x4Unorm,
    Astc4x4Srgb,
}
impl Format {
    /// Bytes per texel for uncompressed formats (returns block byte size for compressed formats;
    /// prefer [`Self::row_bytes`] and [`Self::mip_byte_len`] for upload math).
    pub const fn texel_size(self) -> u32 {
        match self {
            Self::R8Unorm => 1,
            Self::R16Float => 2,
            Self::Rgba16Float | Self::Bc1RgbaUnorm | Self::Bc1RgbaSrgb => 8,
            Self::Rgba32Float
            | Self::Bc7RgbaUnorm
            | Self::Bc7RgbaSrgb
            | Self::Astc4x4Unorm
            | Self::Astc4x4Srgb => 16,
            _ => 4,
        }
    }
    pub const fn is_depth(self) -> bool {
        matches!(self, Self::Depth32Float)
    }
    pub const fn is_srgb(self) -> bool {
        matches!(
            self,
            Self::Rgba8Srgb
                | Self::Bgra8Srgb
                | Self::Bc1RgbaSrgb
                | Self::Bc7RgbaSrgb
                | Self::Astc4x4Srgb
        )
    }
    pub const fn is_compressed(self) -> bool {
        matches!(
            self,
            Self::Bc1RgbaUnorm
                | Self::Bc1RgbaSrgb
                | Self::Bc7RgbaUnorm
                | Self::Bc7RgbaSrgb
                | Self::Astc4x4Unorm
                | Self::Astc4x4Srgb
        )
    }
    /// Block footprint `(width, height)` in texels (`(4, 4)` for BC/ASTC, `(1, 1)` otherwise).
    pub const fn block_dimensions(self) -> (u32, u32) {
        if self.is_compressed() { (4, 4) } else { (1, 1) }
    }
    /// Bytes per texel block (`8` for BC1, `16` for BC7/ASTC 4x4, `texel_size()` otherwise).
    pub const fn block_bytes(self) -> u32 {
        match self {
            Self::Bc1RgbaUnorm | Self::Bc1RgbaSrgb => 8,
            Self::Bc7RgbaUnorm | Self::Bc7RgbaSrgb | Self::Astc4x4Unorm | Self::Astc4x4Srgb => 16,
            _ => self.texel_size(),
        }
    }
    /// Tightly packed row pitch in bytes (per row of texel blocks for compressed formats).
    pub const fn row_bytes(self, width: u32) -> u32 {
        let (bw, _) = self.block_dimensions();
        let w = if width == 0 { 1 } else { width };
        w.div_ceil(bw) * self.block_bytes()
    }
    /// Total tightly packed byte length of a `width x height` mip subresource.
    pub const fn mip_byte_len(self, width: u32, height: u32) -> u64 {
        let (_, bh) = self.block_dimensions();
        let h = if height == 0 { 1 } else { height };
        (self.row_bytes(width) as u64) * (h.div_ceil(bh) as u64)
    }
    /// Maps a cooked [`ganbate_core::world::TextureCompression`] + sRGB flag to a GPU [`Format`].
    pub const fn from_compression(
        comp: ganbate_core::world::TextureCompression,
        srgb: bool,
    ) -> Self {
        use ganbate_core::world::TextureCompression;
        match (comp, srgb) {
            (TextureCompression::Bc1, false) => Self::Bc1RgbaUnorm,
            (TextureCompression::Bc1, true) => Self::Bc1RgbaSrgb,
            (TextureCompression::Bc7, false) => Self::Bc7RgbaUnorm,
            (TextureCompression::Bc7, true) => Self::Bc7RgbaSrgb,
            (TextureCompression::Astc4x4, false) => Self::Astc4x4Unorm,
            (TextureCompression::Astc4x4, true) => Self::Astc4x4Srgb,
        }
    }
    /// Returns the cooked block compression scheme if `self` is compressed.
    pub const fn compression(self) -> Option<ganbate_core::world::TextureCompression> {
        use ganbate_core::world::TextureCompression;
        match self {
            Self::Bc1RgbaUnorm | Self::Bc1RgbaSrgb => Some(TextureCompression::Bc1),
            Self::Bc7RgbaUnorm | Self::Bc7RgbaSrgb => Some(TextureCompression::Bc7),
            Self::Astc4x4Unorm | Self::Astc4x4Srgb => Some(TextureCompression::Astc4x4),
            _ => None,
        }
    }
}

macro_rules! flags {
    ($(#[$meta:meta])* $name:ident { $($(#[$fmeta:meta])* $flag:ident = $bit:expr),* $(,)? }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub struct $name(u32);
        impl $name {
            $($(#[$fmeta])* pub const $flag: Self = Self($bit);)*
            pub const fn empty() -> Self {
                Self(0)
            }
            pub const fn bits(self) -> u32 {
                self.0
            }
            pub const fn is_empty(self) -> bool {
                self.0 == 0
            }
            /// True when every flag in `other` is set in `self`.
            pub const fn contains(self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }
        }
        impl BitOr for $name {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }
        impl BitOrAssign for $name {
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }
    };
}
flags!(
    /// Every way a texture will be used. Uses that are not declared are rejected.
    TextureUsage {
        /// Bound as `Texture`/`TextureLayer`/`TextureMip` to a texture binding.
        SAMPLED = 1,
        COLOR_TARGET = 2,
        DEPTH_TARGET = 4,
        /// Bound to a write-only storage texture binding.
        STORAGE = 8,
        /// Source of `copy_texture_to_buffer`, `blit` or `generate_mips`.
        COPY_SRC = 16,
        /// Destination of `write_texture`, `blit` or `generate_mips`.
        COPY_DST = 32,
    }
);
flags!(
    /// Every way a buffer will be used. Uses that are not declared are rejected.
    BufferUsage {
        VERTEX = 1,
        /// 32-bit indices.
        INDEX = 2,
        UNIFORM = 4,
        STORAGE = 8,
        COPY_SRC = 16,
        /// Destination of copies, including `write_buffer` on `Memory::Device` buffers.
        COPY_DST = 32,
    }
);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    /// Array layers; 6 layers of a square texture may also be bound as a cube.
    pub layers: u32,
    pub mips: u32,
    pub format: Format,
    pub usage: TextureUsage,
}
impl TextureDesc {
    /// A single-layer, single-mip 2D texture.
    pub const fn new(width: u32, height: u32, format: Format, usage: TextureUsage) -> Self {
        Self {
            width,
            height,
            layers: 1,
            mips: 1,
            format,
            usage,
        }
    }
    /// Extent of one mip level (never smaller than 1x1).
    pub const fn mip_size(&self, mip: u32) -> (u32, u32) {
        let w = self.width >> mip;
        let h = self.height >> mip;
        (if w == 0 { 1 } else { w }, if h == 0 { 1 } else { h })
    }
    /// Whether a browser host may copy a `size` image of its own (a canvas, an image bitmap)
    /// straight into one whole subresource: the texture needs `COPY_DST`, an 8-bit RGBA or BGRA
    /// format and exactly that mip's extent.
    pub fn check_source_write(&self, mip: u32, layer: u32, size: (u32, u32)) -> Result<()> {
        if !self.usage.contains(TextureUsage::COPY_DST) {
            return Err("write_texture_source: the texture lacks COPY_DST usage".into());
        }
        if !matches!(
            self.format,
            Format::Rgba8Unorm | Format::Rgba8Srgb | Format::Bgra8Unorm | Format::Bgra8Srgb
        ) {
            return Err(format!(
                "write_texture_source: {:?} is not an 8-bit RGBA format",
                self.format
            ));
        }
        if mip >= self.mips || layer >= self.layers {
            return Err(format!(
                "write_texture_source: mip {mip} layer {layer} is outside the texture"
            ));
        }
        let want = self.mip_size(mip);
        if size != want {
            return Err(format!(
                "write_texture_source: a {}x{} image for a {}x{} mip",
                size.0, size.1, want.0, want.1
            ));
        }
        Ok(())
    }
}

/// Where a buffer lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Memory {
    /// GPU-local. `write_buffer` stages the data and waits for the copy.
    Device,
    /// Host-visible, coherent and persistently mapped: `write_buffer` is a memcpy.
    Upload,
    /// Host-visible and mapped for `read_buffer` after GPU copies into it.
    Readback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferDesc {
    pub size: u64,
    pub usage: BufferUsage,
    pub memory: Memory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Filter {
    Nearest,
    Linear,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Address {
    Repeat,
    Clamp,
    Mirror,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Compare {
    Never,
    Less,
    LessEqual,
    Equal,
    Greater,
    GreaterEqual,
    NotEqual,
    Always,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerDesc {
    pub filter: Filter,
    pub mip_filter: Filter,
    pub address: Address,
    /// 1 disables anisotropic filtering. Clamped to the device maximum; ignored when the
    /// device does not support it (see [`Limits::max_anisotropy`]).
    pub anisotropy: f32,
    /// `Some` makes a comparison sampler (`sampler_comparison`) using `reference op texel`.
    pub compare: Option<Compare>,
    pub max_lod: f32,
}
impl Default for SamplerDesc {
    /// Trilinear, repeating, every mip.
    fn default() -> Self {
        Self {
            filter: Filter::Linear,
            mip_filter: Filter::Linear,
            address: Address::Repeat,
            anisotropy: 1.0,
            compare: None,
            max_lod: 1000.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VertexFormat {
    F32,
    F32x2,
    F32x3,
    F32x4,
    U32,
    /// Four normalized bytes, read as a float vector in the shader.
    Unorm8x4,
}
impl VertexFormat {
    pub const fn size(self) -> u32 {
        match self {
            Self::F32 | Self::U32 | Self::Unorm8x4 => 4,
            Self::F32x2 => 8,
            Self::F32x3 => 12,
            Self::F32x4 => 16,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexAttribute {
    pub location: u32,
    pub format: VertexFormat,
    pub offset: u32,
}
/// One per-vertex buffer bound at slot 0.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VertexLayout {
    pub stride: u32,
    pub attributes: Vec<VertexAttribute>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Topology {
    TriangleList,
    TriangleStrip,
    LineList,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cull {
    None,
    Back,
    Front,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Blend {
    /// No blending.
    Replace,
    /// `src * a + dst * (1 - a)`; alpha accumulates as `a + dst_a * (1 - a)`.
    Alpha,
    /// `src + dst * (1 - a)` for colour and alpha.
    Premultiplied,
    /// `src + dst` for colour and alpha.
    Additive,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorTarget {
    pub format: Format,
    pub blend: Blend,
    /// Bit 0 = R, 1 = G, 2 = B, 3 = A. `0xF` writes every channel.
    pub write_mask: u8,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthState {
    pub format: Format,
    /// `fragment depth op stored depth`.
    pub compare: Compare,
    pub write: bool,
    pub bias_constant: f32,
    pub bias_slope: f32,
    /// Clamp instead of clipping against the near/far planes. Ignored when unsupported
    /// (see [`Limits::depth_clamp`]).
    pub clamp: bool,
}
/// A render pipeline. Its colour and depth formats must match the attachments of every pass it
/// is used in (a pass with a depth attachment needs a pipeline with `depth: Some(..)`).
#[derive(Clone, Debug)]
pub struct RenderPipelineDesc<'a> {
    pub vertex: &'a ganbate_shader::Compiled,
    /// `None` for depth-only pipelines (requires `colors` to be empty).
    pub fragment: Option<&'a ganbate_shader::Compiled>,
    /// `None` when the vertex shader takes no `@location` inputs (e.g. full-screen passes).
    pub vertex_layout: Option<VertexLayout>,
    pub topology: Topology,
    pub cull: Cull,
    /// Counter-clockwise triangles (in Vulkan framebuffer space: y down) are front-facing.
    pub front_ccw: bool,
    pub depth: Option<DepthState>,
    pub colors: Vec<ColorTarget>,
}

/// What a pass does with an attachment's previous contents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Load {
    /// Clear to this value. Depth attachments use component 0.
    Clear([f32; 4]),
    Load,
    /// Previous contents are undefined.
    DontCare,
}
/// One mip level of one layer of a texture, rendered as a 2D attachment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attachment {
    pub texture: TextureId,
    pub mip: u32,
    pub layer: u32,
    pub load: Load,
    /// `false` leaves the contents undefined after the pass.
    pub store: bool,
}
impl Attachment {
    /// Mip 0, layer 0, cleared and stored.
    pub const fn clear(texture: TextureId, value: [f32; 4]) -> Self {
        Self {
            texture,
            mip: 0,
            layer: 0,
            load: Load::Clear(value),
            store: true,
        }
    }
}
/// Colour attachments are bound to `@location(i)` in order. Every attachment must have the same
/// size (that of the selected mip level).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PassDesc {
    pub colors: Vec<Attachment>,
    pub depth: Option<Attachment>,
}

/// A resource for one binding of a bind group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Resource {
    /// `size` 0 binds the rest of the buffer from `offset`.
    Buffer {
        id: BufferId,
        offset: u64,
        size: u64,
    },
    /// Every mip and layer, viewed with the dimension the shader declares (2D, 2D array, cube).
    Texture(TextureId),
    /// One layer (every mip) viewed as a 2D texture, or as a 1-layer array if the shader
    /// declares an array.
    TextureLayer(TextureId, u32),
    /// One mip level (every layer) viewed as a 2D texture, or as an array/cube when the shader
    /// declares one. Storage bindings of mipmapped textures must use this form.
    TextureMip(TextureId, u32),
    Sampler(SamplerId),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    pub max_texture_2d: u32,
    pub max_push_constants: u32,
    /// 1.0 when anisotropic filtering is unsupported.
    pub max_anisotropy: f32,
    pub depth_clamp: bool,
    /// Required alignment of `Resource::Buffer::offset` for uniform buffer bindings.
    pub uniform_offset_alignment: u64,
    /// Required alignment of `Resource::Buffer::offset` for storage buffer bindings.
    pub storage_offset_alignment: u64,
    /// `Depth32Float` can be sampled through a non-comparison `Filter::Linear` sampler.
    /// (Comparison samplers may always filter linearly.) Use `Filter::Nearest` otherwise.
    pub depth_filterable: bool,
    /// Native hardware sampling of BC1 and BC7 compressed textures (transparently falls back to
    /// software RGBA8 upload when false).
    pub texture_compression_bc: bool,
    /// Native hardware sampling of ASTC 4x4 LDR compressed textures (transparently falls back to
    /// software RGBA8 upload when false).
    pub texture_compression_astc: bool,
}

/// Zero-copy view of `values` as little-endian bytes on little-endian targets.
#[cfg(target_endian = "little")]
#[inline]
pub(crate) fn as_le_bytes_f32(values: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no padding or uninitialized bytes and `u8` has alignment 1.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

/// Zero-copy view of `values` as little-endian bytes on little-endian targets.
#[cfg(target_endian = "little")]
#[inline]
pub(crate) fn as_le_bytes_u32(values: &[u32]) -> &[u8] {
    // SAFETY: `u32` has no padding or uninitialized bytes and `u8` has alignment 1.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

/// Little-endian bytes of `values`, for `write_buffer` and `push_constants`.
#[inline]
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    #[cfg(target_endian = "little")]
    {
        as_le_bytes_f32(values).to_vec()
    }
    #[cfg(not(target_endian = "little"))]
    {
        let mut out = Vec::with_capacity(values.len() * 4);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}
/// Little-endian bytes of `values`, for `write_buffer` and `push_constants`.
#[inline]
pub fn u32_bytes(values: &[u32]) -> Vec<u8> {
    #[cfg(target_endian = "little")]
    {
        as_le_bytes_u32(values).to_vec()
    }
    #[cfg(not(target_endian = "little"))]
    {
        let mut out = Vec::with_capacity(values.len() * 4);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_writes_need_copy_dst_rgba8_and_the_exact_mip_size() {
        let usage = TextureUsage::SAMPLED | TextureUsage::COPY_DST;
        let mut d = TextureDesc::new(512, 717, Format::Rgba8Srgb, usage);
        d.mips = 10;
        assert_eq!(d.check_source_write(0, 0, (512, 717)), Ok(()));
        assert_eq!(d.check_source_write(9, 0, (1, 1)), Ok(()));
        for format in [Format::Rgba8Unorm, Format::Bgra8Unorm, Format::Bgra8Srgb] {
            let t = TextureDesc { format, ..d };
            assert_eq!(t.check_source_write(1, 0, (256, 358)), Ok(()));
        }

        let err =
            |t: &TextureDesc, mip, layer, size| t.check_source_write(mip, layer, size).unwrap_err();
        assert!(err(&d, 0, 0, (717, 512)).contains("a 717x512 image for a 512x717 mip"));
        assert!(err(&d, 1, 0, (512, 717)).contains("for a 256x358 mip"));
        assert!(err(&d, 10, 0, (1, 1)).contains("mip 10 layer 0 is outside"));
        assert!(err(&d, 0, 1, (512, 717)).contains("mip 0 layer 1 is outside"));
        let sampled_only = TextureDesc {
            usage: TextureUsage::SAMPLED,
            ..d
        };
        assert!(err(&sampled_only, 0, 0, (512, 717)).contains("lacks COPY_DST"));
        for format in [
            Format::Rgba16Float,
            Format::R8Unorm,
            Format::Depth32Float,
            Format::Bc7RgbaSrgb,
        ] {
            let t = TextureDesc { format, ..d };
            assert!(err(&t, 0, 0, (512, 717)).contains("not an 8-bit RGBA format"));
        }
    }
}
