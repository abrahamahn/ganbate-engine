//! Run-time glTF 2.0 models from GLB files: a node hierarchy with meshes, materials, skins,
//! morph targets and animation clips, for objects a game places and animates while it runs
//! (the static world cooker is [`crate::glb`]).
//!
//! Geometry may be quantized (`KHR_mesh_quantization`) and accessors sparse; everything is
//! decoded to `f32` (normalized integers per the specification). Images stay encoded (PNG,
//! JPEG, WebP via `EXT_texture_webp`) for the host to decode: browsers do it natively.
//! Draco and meshopt compression are rejected; decode them when preparing assets. Required
//! extensions other than those listed in [`SUPPORTED`] are rejected; optional ones are ignored.
use crate::glb::chunks;
use crate::json::Value;

/// Extensions honoured here. Others that a file *requires* are errors.
pub const SUPPORTED: [&str; 4] = [
    "KHR_mesh_quantization",
    "EXT_texture_webp",
    "KHR_materials_emissive_strength",
    "KHR_materials_unlit",
];
pub const MAX_NODES: usize = 65_536;
pub const MAX_JOINTS: usize = 1024;
/// Vertices across all primitives.
pub const MAX_VERTICES: usize = 4 * 1024 * 1024;
pub const MAX_CLIPS: usize = 4096;
pub const MAX_TARGETS: usize = 64;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Model {
    pub nodes: Vec<Node>,
    /// The nodes of the default scene without a parent (all parentless nodes without scenes).
    pub roots: Vec<usize>,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    pub images: Vec<Image>,
    pub skins: Vec<Skin>,
    pub clips: Vec<Clip>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub name: String,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub translation: [f32; 3],
    /// Unit quaternion `[x, y, z, w]`.
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub mesh: Option<usize>,
    pub skin: Option<usize>,
    /// Morph weights for this node's mesh (`None`: the mesh's defaults).
    pub weights: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub name: String,
    pub primitives: Vec<Primitive>,
    /// Default morph weights, one per target.
    pub weights: Vec<f32>,
    /// `extras.targetNames`, when given.
    pub target_names: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Primitive {
    pub positions: Vec<[f32; 3]>,
    /// Empty, or one per position.
    pub normals: Vec<[f32; 3]>,
    /// `TEXCOORD_0`: empty, or one per position.
    pub uvs: Vec<[f32; 2]>,
    /// Counter-clockwise triangles.
    pub indices: Vec<u32>,
    pub material: Option<usize>,
    /// `JOINTS_0` (indices into the skin's joints) and `WEIGHTS_0`: empty, or one per position.
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub targets: Vec<MorphTarget>,
}

/// Displacements added, scaled by the target's weight.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MorphTarget {
    /// Empty, or one per vertex.
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AlphaMode {
    #[default]
    Opaque,
    Mask,
    Blend,
}

/// A texture used by a material, with the `TEXCOORD_n` set it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureRef {
    pub texture: usize,
    pub tex_coord: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub name: String,
    /// Linear RGBA factor.
    pub base_color: [f32; 4],
    pub base_texture: Option<TextureRef>,
    pub metallic: f32,
    pub roughness: f32,
    pub normal_texture: Option<TextureRef>,
    pub normal_scale: f32,
    /// Linear, times `KHR_materials_emissive_strength`.
    pub emissive: [f32; 3],
    pub emissive_texture: Option<TextureRef>,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    pub unlit: bool,
}

impl Default for Material {
    /// The glTF default material.
    fn default() -> Self {
        Self {
            name: String::new(),
            base_color: [1.0; 4],
            base_texture: None,
            metallic: 1.0,
            roughness: 1.0,
            normal_texture: None,
            normal_scale: 1.0,
            emissive: [0.0; 3],
            emissive_texture: None,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            unlit: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Texture {
    /// Index into [`Model::images`].
    pub image: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub name: String,
    /// `image/png`, `image/jpeg` or `image/webp`.
    pub mime: String,
    pub data: Vec<u8>,
    /// Sampled as colour (base colour, emissive) rather than data (normal maps): decode it as
    /// sRGB.
    pub srgb: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Skin {
    pub name: String,
    pub joints: Vec<usize>,
    /// Column-major, one per joint.
    pub inverse_bind: Vec<[f32; 16]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    Translation,
    Rotation,
    Scale,
    Weights,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Linear,
    Step,
    /// Values hold (in-tangent, value, out-tangent) per key.
    CubicSpline,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    pub node: usize,
    pub path: Path,
    pub interpolation: Interpolation,
    /// Key times in seconds, increasing.
    pub times: Vec<f32>,
    /// `times.len() * width` values (three times that for cubic splines), where `width` is 3 for
    /// translation and scale, 4 for rotation and the node's morph target count for weights.
    pub values: Vec<f32>,
    pub width: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Clip {
    pub name: String,
    pub channels: Vec<Channel>,
    /// The last key time over all channels.
    pub duration: f32,
}

type R<T> = Result<T, String>;

fn arr<'a>(o: &'a Value, k: &str) -> R<&'a [Value]> {
    o.get(k).map_or(Ok(&[][..]), Value::array)
}
fn need<'a>(o: &'a Value, k: &str) -> R<&'a Value> {
    o.get(k).ok_or_else(|| format!("missing glTF field {k}"))
}
fn idx(o: &Value, k: &str, len: usize, what: &str) -> R<usize> {
    let i = need(o, k)?.integer()?;
    if i >= len {
        return Err(format!("{what} {i} outside {len}"));
    }
    Ok(i)
}
fn opt_idx(o: &Value, k: &str, len: usize, what: &str) -> R<Option<usize>> {
    if o.get(k).is_none() {
        return Ok(None);
    }
    idx(o, k, len, what).map(Some)
}
fn num(o: Option<&Value>, k: &str, default: f32) -> R<f32> {
    o.and_then(|o| o.get(k))
        .map_or(Ok(default), |v| v.float().map_err(|e| format!("{k}: {e}")))
}
fn fixed<const N: usize>(v: &Value) -> R<[f32; N]> {
    let a = v.array()?;
    if a.len() != N {
        return Err(format!("expected {N} numbers"));
    }
    let mut out = [0.0; N];
    for (o, x) in out.iter_mut().zip(a) {
        *o = x.float()?;
    }
    Ok(out)
}
fn ext<'a>(o: &'a Value, name: &str) -> Option<&'a Value> {
    o.get("extensions").and_then(|e| e.get(name))
}
fn name(o: &Value) -> String {
    o.get("name")
        .and_then(|n| n.string().ok())
        .unwrap_or("")
        .chars()
        .take(256)
        .collect()
}

/// A decoded accessor: elements of `width` components.
struct Data {
    values: Vec<f32>,
    width: usize,
}

struct Doc<'a> {
    json: &'a Value,
    bin: &'a [u8],
}

impl Doc<'_> {
    fn view(&self, id: usize) -> R<(&[u8], usize)> {
        let views = arr(self.json, "bufferViews")?;
        let v = views
            .get(id)
            .ok_or_else(|| format!("buffer view {id} outside {}", views.len()))?;
        for e in ["EXT_meshopt_compression", "KHR_meshopt_compression"] {
            if ext(v, e).is_some() {
                return Err(format!("decode {e} when preparing the asset"));
            }
        }
        if need(v, "buffer")?.integer()? != 0 {
            return Err("external buffers are unsupported (use a GLB)".into());
        }
        let start = v.get("byteOffset").map_or(Ok(0), Value::integer)?;
        let len = need(v, "byteLength")?.integer()?;
        let stride = v.get("byteStride").map_or(Ok(0), Value::integer)?;
        let bytes = start
            .checked_add(len)
            .and_then(|end| self.bin.get(start..end))
            .ok_or("buffer view outside the binary chunk")?;
        Ok((bytes, stride))
    }

    /// Decodes accessor `id` to floats, applying normalization; integer types keep their value
    /// when not normalized.
    fn floats(&self, id: usize) -> R<Data> {
        let accessors = arr(self.json, "accessors")?;
        let a = accessors
            .get(id)
            .ok_or_else(|| format!("accessor {id} outside {}", accessors.len()))?;
        let count = need(a, "count")?.integer()?;
        if count == 0 || count > MAX_VERTICES * 4 {
            return Err(format!("accessor {id}: count {count} out of range"));
        }
        let width = match need(a, "type")?.string()? {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            "MAT4" => 16,
            t => return Err(format!("accessor {id}: unsupported type {t}")),
        };
        let component = need(a, "componentType")?.integer()?;
        let size = match component {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            5125 | 5126 => 4,
            c => return Err(format!("accessor {id}: unknown component type {c}")),
        };
        let normalized = matches!(a.get("normalized"), Some(Value::Bool(true)));
        let decode = |b: &[u8]| -> f32 {
            let v = match component {
                5120 => f32::from(b[0] as i8),
                5121 => f32::from(b[0]),
                5122 => f32::from(i16::from_le_bytes([b[0], b[1]])),
                5123 => f32::from(u16::from_le_bytes([b[0], b[1]])),
                5125 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
                _ => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            };
            if !normalized {
                return v;
            }
            match component {
                5120 => (v / 127.0).max(-1.0),
                5121 => v / 255.0,
                5122 => (v / 32767.0).max(-1.0),
                5123 => v / 65535.0,
                _ => v,
            }
        };
        // Matrix columns of 1- and 2-byte components are padded to 4 bytes.
        let elem = if width == 16 && size < 4 {
            return Err(format!("accessor {id}: matrices must be floats"));
        } else {
            width * size
        };
        let mut values = vec![0.0f32; count * width];
        if let Some(view) = a.get("bufferView") {
            let (bytes, stride) = self.view(view.integer()?)?;
            let stride = if stride == 0 { elem } else { stride };
            let offset = a.get("byteOffset").map_or(Ok(0), Value::integer)?;
            if stride < elem || offset % size != 0 || stride % size != 0 {
                return Err(format!("accessor {id}: misaligned"));
            }
            let end = (count - 1)
                .checked_mul(stride)
                .and_then(|n| n.checked_add(offset + elem))
                .ok_or("accessor overflow")?;
            if end > bytes.len() {
                return Err(format!("accessor {id}: outside its buffer view"));
            }
            for i in 0..count {
                let at = offset + i * stride;
                for k in 0..width {
                    values[i * width + k] = decode(&bytes[at + k * size..]);
                }
            }
        }
        if let Some(sparse) = a.get("sparse") {
            let n = need(sparse, "count")?.integer()?;
            if n > count {
                return Err(format!("accessor {id}: sparse count {n} exceeds {count}"));
            }
            let ix = need(sparse, "indices")?;
            let (ib, _) = self.view(need(ix, "bufferView")?.integer()?)?;
            let io = ix.get("byteOffset").map_or(Ok(0), Value::integer)?;
            let isize = match need(ix, "componentType")?.integer()? {
                5121 => 1,
                5123 => 2,
                5125 => 4,
                c => return Err(format!("accessor {id}: sparse index type {c}")),
            };
            let vx = need(sparse, "values")?;
            let (vb, _) = self.view(need(vx, "bufferView")?.integer()?)?;
            let vo = vx.get("byteOffset").map_or(Ok(0), Value::integer)?;
            if io + n * isize > ib.len() || vo + n * elem > vb.len() {
                return Err(format!("accessor {id}: sparse data outside its views"));
            }
            for j in 0..n {
                let b = &ib[io + j * isize..];
                let target = match isize {
                    1 => usize::from(b[0]),
                    2 => usize::from(u16::from_le_bytes([b[0], b[1]])),
                    _ => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize,
                };
                if target >= count {
                    return Err(format!("accessor {id}: sparse index {target} out of range"));
                }
                for k in 0..width {
                    values[target * width + k] = decode(&vb[vo + j * elem + k * size..]);
                }
            }
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(format!("accessor {id}: values are not finite"));
        }
        Ok(Data { values, width })
    }

    fn vecs<const N: usize>(&self, id: usize, what: &str) -> R<Vec<[f32; N]>> {
        let accessors = arr(self.json, "accessors")?;
        let a = accessors
            .get(id)
            .ok_or_else(|| format!("accessor {id} outside {}", accessors.len()))?;
        let count = need(a, "count")?.integer()?;
        if count == 0 || count > MAX_VERTICES * 4 {
            return Err(format!("accessor {id}: count {count} out of range"));
        }
        let width = match need(a, "type")?.string()? {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            "MAT4" => 16,
            t => return Err(format!("accessor {id}: unsupported type {t}")),
        };
        if width != N {
            return Err(format!("{what}: expected {N} components, got {width}"));
        }
        let component = need(a, "componentType")?.integer()?;
        let size = match component {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            5125 | 5126 => 4,
            c => return Err(format!("accessor {id}: unknown component type {c}")),
        };
        if width == 16 && size < 4 {
            return Err(format!("accessor {id}: matrices must be floats"));
        }
        let normalized = matches!(a.get("normalized"), Some(Value::Bool(true)));
        let decode = |b: &[u8]| -> f32 {
            let v = match component {
                5120 => f32::from(b[0] as i8),
                5121 => f32::from(b[0]),
                5122 => f32::from(i16::from_le_bytes([b[0], b[1]])),
                5123 => f32::from(u16::from_le_bytes([b[0], b[1]])),
                5125 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
                _ => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            };
            if !normalized {
                return v;
            }
            match component {
                5120 => (v / 127.0).max(-1.0),
                5121 => v / 255.0,
                5122 => (v / 32767.0).max(-1.0),
                5123 => v / 65535.0,
                _ => v,
            }
        };
        let elem = N * size;
        let mut out = vec![[0.0f32; N]; count];
        if let Some(view) = a.get("bufferView") {
            let (bytes, stride) = self.view(view.integer()?)?;
            let stride = if stride == 0 { elem } else { stride };
            let offset = a.get("byteOffset").map_or(Ok(0), Value::integer)?;
            if stride < elem || offset % size != 0 || stride % size != 0 {
                return Err(format!("accessor {id}: misaligned"));
            }
            let end = (count - 1)
                .checked_mul(stride)
                .and_then(|n| n.checked_add(offset + elem))
                .ok_or("accessor overflow")?;
            if end > bytes.len() {
                return Err(format!("accessor {id}: outside its buffer view"));
            }
            if component == 5126 && stride == elem {
                for (dst, chunk) in out
                    .iter_mut()
                    .zip(bytes[offset..offset + count * elem].chunks_exact(elem))
                {
                    for (slot, w) in dst.iter_mut().zip(chunk.chunks_exact(4)) {
                        *slot = f32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                    }
                }
            } else {
                for (i, dst) in out.iter_mut().enumerate() {
                    let at = offset + i * stride;
                    for (k, slot) in dst.iter_mut().enumerate() {
                        *slot = decode(&bytes[at + k * size..]);
                    }
                }
            }
        }
        if let Some(sparse) = a.get("sparse") {
            let n = need(sparse, "count")?.integer()?;
            if n > count {
                return Err(format!("accessor {id}: sparse count {n} exceeds {count}"));
            }
            let ix = need(sparse, "indices")?;
            let (ib, _) = self.view(need(ix, "bufferView")?.integer()?)?;
            let io = ix.get("byteOffset").map_or(Ok(0), Value::integer)?;
            let isize = match need(ix, "componentType")?.integer()? {
                5121 => 1,
                5123 => 2,
                5125 => 4,
                c => return Err(format!("accessor {id}: sparse index type {c}")),
            };
            let vx = need(sparse, "values")?;
            let (vb, _) = self.view(need(vx, "bufferView")?.integer()?)?;
            let vo = vx.get("byteOffset").map_or(Ok(0), Value::integer)?;
            if io + n * isize > ib.len() || vo + n * elem > vb.len() {
                return Err(format!("accessor {id}: sparse data outside its views"));
            }
            for j in 0..n {
                let b = &ib[io + j * isize..];
                let target = match isize {
                    1 => usize::from(b[0]),
                    2 => usize::from(u16::from_le_bytes([b[0], b[1]])),
                    _ => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize,
                };
                if target >= count {
                    return Err(format!("accessor {id}: sparse index {target} out of range"));
                }
                for k in 0..N {
                    out[target][k] = decode(&vb[vo + j * elem + k * size..]);
                }
            }
        }
        if out.iter().flatten().any(|v| !v.is_finite()) {
            return Err(format!("accessor {id}: values are not finite"));
        }
        Ok(out)
    }

    fn indices(&self, id: usize, mi: usize) -> R<Vec<u32>> {
        let accessors = arr(self.json, "accessors")?;
        let a = accessors
            .get(id)
            .ok_or_else(|| format!("accessor {id} outside {}", accessors.len()))?;
        if need(a, "type")?.string()? != "SCALAR" {
            return Err(format!("mesh {mi}: indices must be scalars"));
        }
        let component = need(a, "componentType")?.integer()?;
        let normalized = matches!(a.get("normalized"), Some(Value::Bool(true)));
        if normalized || !matches!(component, 5121 | 5123 | 5125) {
            let d = self.floats(id)?;
            return Ok(d.values.into_iter().map(|v| v as u32).collect());
        }
        let count = need(a, "count")?.integer()?;
        if count == 0 || count > MAX_VERTICES * 4 {
            return Err(format!("accessor {id}: count {count} out of range"));
        }
        let size = match component {
            5121 => 1,
            5123 => 2,
            _ => 4,
        };
        let decode_u32 = |b: &[u8]| -> u32 {
            match size {
                1 => u32::from(b[0]),
                2 => u32::from(u16::from_le_bytes([b[0], b[1]])),
                _ => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            }
        };
        let mut values = vec![0u32; count];
        if let Some(view) = a.get("bufferView") {
            let (bytes, stride) = self.view(view.integer()?)?;
            let stride = if stride == 0 { size } else { stride };
            let offset = a.get("byteOffset").map_or(Ok(0), Value::integer)?;
            if stride < size || offset % size != 0 || stride % size != 0 {
                return Err(format!("accessor {id}: misaligned"));
            }
            let end = (count - 1)
                .checked_mul(stride)
                .and_then(|n| n.checked_add(offset + size))
                .ok_or("accessor overflow")?;
            if end > bytes.len() {
                return Err(format!("accessor {id}: outside its buffer view"));
            }
            let slice = &bytes[offset..end];
            match (size, stride) {
                (1, 1) => {
                    for (dst, &b) in values.iter_mut().zip(&slice[..count]) {
                        *dst = u32::from(b);
                    }
                }
                (2, 2) => {
                    for (dst, c) in values.iter_mut().zip(slice.chunks_exact(2)) {
                        *dst = u32::from(u16::from_le_bytes([c[0], c[1]]));
                    }
                }
                (4, 4) => {
                    for (dst, c) in values.iter_mut().zip(slice.chunks_exact(4)) {
                        *dst = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    }
                }
                _ => {
                    for (i, dst) in values.iter_mut().enumerate() {
                        *dst = decode_u32(&bytes[offset + i * stride..]);
                    }
                }
            }
        }
        if let Some(sparse) = a.get("sparse") {
            let n = need(sparse, "count")?.integer()?;
            if n > count {
                return Err(format!("accessor {id}: sparse count {n} exceeds {count}"));
            }
            let ix = need(sparse, "indices")?;
            let (ib, _) = self.view(need(ix, "bufferView")?.integer()?)?;
            let io = ix.get("byteOffset").map_or(Ok(0), Value::integer)?;
            let isize = match need(ix, "componentType")?.integer()? {
                5121 => 1,
                5123 => 2,
                5125 => 4,
                c => return Err(format!("accessor {id}: sparse index type {c}")),
            };
            let vx = need(sparse, "values")?;
            let (vb, _) = self.view(need(vx, "bufferView")?.integer()?)?;
            let vo = vx.get("byteOffset").map_or(Ok(0), Value::integer)?;
            if io + n * isize > ib.len() || vo + n * size > vb.len() {
                return Err(format!("accessor {id}: sparse data outside its views"));
            }
            for j in 0..n {
                let b = &ib[io + j * isize..];
                let target = match isize {
                    1 => usize::from(b[0]),
                    2 => usize::from(u16::from_le_bytes([b[0], b[1]])),
                    _ => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize,
                };
                if target >= count {
                    return Err(format!("accessor {id}: sparse index {target} out of range"));
                }
                values[target] = decode_u32(&vb[vo + j * size..]);
            }
        }
        Ok(values)
    }

    fn joints(&self, id: usize, what: &str) -> R<Vec<[u16; 4]>> {
        let accessors = arr(self.json, "accessors")?;
        let a = accessors
            .get(id)
            .ok_or_else(|| format!("accessor {id} outside {}", accessors.len()))?;
        let component = need(a, "componentType")?.integer()?;
        let normalized = matches!(a.get("normalized"), Some(Value::Bool(true)));
        if normalized
            || !matches!(component, 5121 | 5123)
            || need(a, "type")?.string()? != "VEC4"
            || a.get("sparse").is_some()
        {
            return Ok(self
                .vecs::<4>(id, what)?
                .into_iter()
                .map(|j| j.map(|v| v as u16))
                .collect());
        }
        let count = need(a, "count")?.integer()?;
        if count == 0 || count > MAX_VERTICES * 4 {
            return Err(format!("accessor {id}: count {count} out of range"));
        }
        let size = if component == 5121 { 1 } else { 2 };
        let elem = 4 * size;
        let mut out = vec![[0u16; 4]; count];
        if let Some(view) = a.get("bufferView") {
            let (bytes, stride) = self.view(view.integer()?)?;
            let stride = if stride == 0 { elem } else { stride };
            let offset = a.get("byteOffset").map_or(Ok(0), Value::integer)?;
            if stride < elem || offset % size != 0 || stride % size != 0 {
                return Err(format!("accessor {id}: misaligned"));
            }
            let end = (count - 1)
                .checked_mul(stride)
                .and_then(|n| n.checked_add(offset + elem))
                .ok_or("accessor overflow")?;
            if end > bytes.len() {
                return Err(format!("accessor {id}: outside its buffer view"));
            }
            if size == 1 {
                for (i, dst) in out.iter_mut().enumerate() {
                    let b = &bytes[offset + i * stride..];
                    *dst = [
                        u16::from(b[0]),
                        u16::from(b[1]),
                        u16::from(b[2]),
                        u16::from(b[3]),
                    ];
                }
            } else {
                for (i, dst) in out.iter_mut().enumerate() {
                    let b = &bytes[offset + i * stride..];
                    *dst = [
                        u16::from_le_bytes([b[0], b[1]]),
                        u16::from_le_bytes([b[2], b[3]]),
                        u16::from_le_bytes([b[4], b[5]]),
                        u16::from_le_bytes([b[6], b[7]]),
                    ];
                }
            }
        }
        Ok(out)
    }
}

/// Decomposes a column-major affine matrix into translation, rotation and scale.
fn decompose(m: &[f32; 16]) -> ([f32; 3], [f32; 4], [f32; 3]) {
    let t = [m[12], m[13], m[14]];
    let col = |c: usize| [m[c * 4], m[c * 4 + 1], m[c * 4 + 2]];
    let len = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let (x, y, z) = (col(0), col(1), col(2));
    let det = x[0] * (y[1] * z[2] - y[2] * z[1]) - x[1] * (y[0] * z[2] - y[2] * z[0])
        + x[2] * (y[0] * z[1] - y[1] * z[0]);
    let mut s = [len(x), len(y), len(z)];
    if det < 0.0 {
        s[0] = -s[0];
    }
    let n = |v: [f32; 3], l: f32| {
        if l.abs() > 1e-12 {
            [v[0] / l, v[1] / l, v[2] / l]
        } else {
            [0.0; 3]
        }
    };
    let (x, y, z) = (n(x, s[0]), n(y, s[1]), n(z, s[2]));
    // Rotation matrix (columns x, y, z) to quaternion.
    let trace = x[0] + y[1] + z[2];
    let q = if trace > 0.0 {
        let r = (1.0 + trace).sqrt() * 2.0;
        [
            (y[2] - z[1]) / r,
            (z[0] - x[2]) / r,
            (x[1] - y[0]) / r,
            0.25 * r,
        ]
    } else if x[0] > y[1] && x[0] > z[2] {
        let r = (1.0 + x[0] - y[1] - z[2]).sqrt() * 2.0;
        [
            0.25 * r,
            (y[0] + x[1]) / r,
            (z[0] + x[2]) / r,
            (y[2] - z[1]) / r,
        ]
    } else if y[1] > z[2] {
        let r = (1.0 + y[1] - x[0] - z[2]).sqrt() * 2.0;
        [
            (y[0] + x[1]) / r,
            0.25 * r,
            (z[1] + y[2]) / r,
            (z[0] - x[2]) / r,
        ]
    } else {
        let r = (1.0 + z[2] - x[0] - y[1]).sqrt() * 2.0;
        [
            (z[0] + x[2]) / r,
            (z[1] + y[2]) / r,
            0.25 * r,
            (x[1] - y[0]) / r,
        ]
    };
    let ql = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let q = if ql > 1e-12 {
        q.map(|v| v / ql)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
    (t, q, s)
}

/// Parses a GLB into a [`Model`].
pub fn load_glb(data: &[u8]) -> R<Model> {
    let (json, bin) = chunks(data)?;
    let doc = Doc { json: &json, bin };
    for e in arr(&json, "extensionsRequired")? {
        let e = e.string()?;
        if !SUPPORTED.contains(&e) {
            return Err(match e {
                "KHR_draco_mesh_compression" => {
                    "Draco-compressed meshes: decode them when preparing the asset".into()
                }
                _ => format!("required extension {e} is unsupported"),
            });
        }
    }
    let mut model = Model::default();

    // Images: kept encoded. Whether each is colour is settled by the materials below.
    for (i, im) in arr(&json, "images")?.iter().enumerate() {
        let mime = im
            .get("mimeType")
            .map(|m| m.string().map(str::to_string))
            .transpose()?
            .unwrap_or_default();
        let view = im
            .get("bufferView")
            .ok_or_else(|| format!("image {i}: images must be embedded (bufferView)"))?
            .integer()?;
        let (bytes, _) = doc.view(view)?;
        let mime = if mime.is_empty() {
            sniff(bytes).ok_or_else(|| format!("image {i}: unknown format"))?
        } else {
            mime
        };
        if !matches!(mime.as_str(), "image/png" | "image/jpeg" | "image/webp") {
            return Err(format!("image {i}: unsupported type {mime}"));
        }
        model.images.push(Image {
            name: name(im),
            mime,
            data: bytes.to_vec(),
            srgb: false,
        });
    }
    let image_count = model.images.len();
    for t in arr(&json, "textures")? {
        let source = ext(t, "EXT_texture_webp")
            .and_then(|w| w.get("source"))
            .or_else(|| t.get("source"))
            .ok_or("a texture without a source")?
            .integer()?;
        if source >= image_count {
            return Err(format!(
                "texture source {source} outside {image_count} images"
            ));
        }
        model.textures.push(Texture { image: source });
    }
    let texture_count = model.textures.len();
    let tex_ref = |o: Option<&Value>| -> R<Option<TextureRef>> {
        let Some(o) = o else { return Ok(None) };
        Ok(Some(TextureRef {
            texture: idx(o, "index", texture_count, "texture")?,
            tex_coord: o.get("texCoord").map_or(Ok(0), Value::integer)?,
        }))
    };

    for m in arr(&json, "materials")? {
        let pbr = m.get("pbrMetallicRoughness");
        let mut mat = Material {
            name: name(m),
            ..Material::default()
        };
        if let Some(c) = pbr.and_then(|p| p.get("baseColorFactor")) {
            mat.base_color = fixed(c).map_err(|e| format!("baseColorFactor: {e}"))?;
        }
        mat.base_texture = tex_ref(pbr.and_then(|p| p.get("baseColorTexture")))?;
        mat.metallic = num(pbr, "metallicFactor", 1.0)?.clamp(0.0, 1.0);
        mat.roughness = num(pbr, "roughnessFactor", 1.0)?.clamp(0.0, 1.0);
        mat.normal_texture = tex_ref(m.get("normalTexture"))?;
        mat.normal_scale = num(m.get("normalTexture"), "scale", 1.0)?;
        if let Some(e) = m.get("emissiveFactor") {
            mat.emissive = fixed(e).map_err(|e| format!("emissiveFactor: {e}"))?;
        }
        let strength = num(
            ext(m, "KHR_materials_emissive_strength"),
            "emissiveStrength",
            1.0,
        )?;
        mat.emissive = mat.emissive.map(|c| c * strength);
        mat.emissive_texture = tex_ref(m.get("emissiveTexture"))?;
        mat.alpha_mode = match m.get("alphaMode").map(Value::string).transpose()? {
            None | Some("OPAQUE") => AlphaMode::Opaque,
            Some("MASK") => AlphaMode::Mask,
            Some("BLEND") => AlphaMode::Blend,
            Some(o) => return Err(format!("alphaMode {o}")),
        };
        mat.alpha_cutoff = num(Some(m), "alphaCutoff", 0.5)?;
        mat.double_sided = matches!(m.get("doubleSided"), Some(Value::Bool(true)));
        mat.unlit = ext(m, "KHR_materials_unlit").is_some();
        for r in [mat.base_texture, mat.emissive_texture]
            .into_iter()
            .flatten()
        {
            let image = model.textures[r.texture].image;
            model.images[image].srgb = true;
        }
        model.materials.push(mat);
    }
    let material_count = model.materials.len();

    let mut vertices = 0usize;
    for (mi, m) in arr(&json, "meshes")?.iter().enumerate() {
        let mut mesh = Mesh {
            name: name(m),
            weights: arr(m, "weights")?
                .iter()
                .map(Value::float)
                .collect::<R<Vec<f32>>>()?,
            target_names: m
                .get("extras")
                .and_then(|e| e.get("targetNames"))
                .map_or(Ok(&[][..]), Value::array)?
                .iter()
                .map(|n| n.string().map(str::to_string))
                .collect::<R<Vec<String>>>()?,
            primitives: vec![],
        };
        for p in arr(m, "primitives")? {
            if ext(p, "KHR_draco_mesh_compression").is_some() {
                return Err(format!(
                    "mesh {mi}: Draco-compressed: decode it when preparing the asset"
                ));
            }
            let mode = p.get("mode").map_or(Ok(4), Value::integer)?;
            if mode != 4 {
                return Err(format!(
                    "mesh {mi}: only triangle lists are supported (mode {mode})"
                ));
            }
            let attrs = need(p, "attributes")?;
            let accessor = |k: &str| attrs.get(k).map(Value::integer).transpose();
            let positions = doc.vecs::<3>(
                accessor("POSITION")?.ok_or("a primitive without POSITION")?,
                "POSITION",
            )?;
            let n = positions.len();
            vertices += n;
            if vertices > MAX_VERTICES {
                return Err(format!("more than {MAX_VERTICES} vertices"));
            }
            let per_vertex = |len: usize, what: &str| -> R<()> {
                if len != n {
                    return Err(format!(
                        "mesh {mi}: {what} has {len} entries for {n} vertices"
                    ));
                }
                Ok(())
            };
            let normals = match accessor("NORMAL")? {
                Some(a) => doc.vecs::<3>(a, "NORMAL")?,
                None => vec![],
            };
            let uvs = match accessor("TEXCOORD_0")? {
                Some(a) => doc.vecs::<2>(a, "TEXCOORD_0")?,
                None => vec![],
            };
            let joints: Vec<[u16; 4]> = match accessor("JOINTS_0")? {
                Some(a) => doc.joints(a, "JOINTS_0")?,
                None => vec![],
            };
            let weights = match accessor("WEIGHTS_0")? {
                Some(a) => doc.vecs::<4>(a, "WEIGHTS_0")?,
                None => vec![],
            };
            for (len, what) in [
                (normals.len(), "NORMAL"),
                (uvs.len(), "TEXCOORD_0"),
                (joints.len(), "JOINTS_0"),
                (weights.len(), "WEIGHTS_0"),
            ] {
                if len != 0 {
                    per_vertex(len, what)?;
                }
            }
            if joints.is_empty() != weights.is_empty() {
                return Err(format!("mesh {mi}: JOINTS_0 and WEIGHTS_0 come together"));
            }
            let indices: Vec<u32> = match p.get("indices") {
                Some(i) => doc.indices(i.integer()?, mi)?,
                None => (0..n as u32).collect(),
            };
            if indices.len() % 3 != 0 || indices.iter().any(|&i| i as usize >= n) {
                return Err(format!(
                    "mesh {mi}: indices out of range or not whole triangles"
                ));
            }
            let mut targets = vec![];
            for t in arr(p, "targets")? {
                if targets.len() >= MAX_TARGETS {
                    return Err(format!("mesh {mi}: more than {MAX_TARGETS} morph targets"));
                }
                let get = |k: &str| t.get(k).map(Value::integer).transpose();
                let positions = match get("POSITION")? {
                    Some(a) => doc.vecs::<3>(a, "target POSITION")?,
                    None => vec![],
                };
                let normals = match get("NORMAL")? {
                    Some(a) => doc.vecs::<3>(a, "target NORMAL")?,
                    None => vec![],
                };
                for (len, what) in [
                    (positions.len(), "target POSITION"),
                    (normals.len(), "target NORMAL"),
                ] {
                    if len != 0 {
                        per_vertex(len, what)?;
                    }
                }
                targets.push(MorphTarget { positions, normals });
            }
            mesh.primitives.push(Primitive {
                positions,
                normals,
                uvs,
                indices,
                material: opt_idx(p, "material", material_count, "material")?,
                joints,
                weights,
                targets,
            });
        }
        let targets = mesh.primitives.first().map_or(0, |p| p.targets.len());
        if mesh.primitives.iter().any(|p| p.targets.len() != targets) {
            return Err(format!("mesh {mi}: primitives disagree on morph targets"));
        }
        if !mesh.weights.is_empty() && mesh.weights.len() != targets {
            return Err(format!(
                "mesh {mi}: {} weights for {targets} targets",
                mesh.weights.len()
            ));
        }
        mesh.weights.resize(targets, 0.0);
        model.meshes.push(mesh);
    }
    let mesh_count = model.meshes.len();

    let nodes = arr(&json, "nodes")?;
    if nodes.len() > MAX_NODES {
        return Err(format!("more than {MAX_NODES} nodes"));
    }
    let skin_count = arr(&json, "skins")?.len();
    for n in nodes {
        let (translation, rotation, scale) = if let Some(m) = n.get("matrix") {
            decompose(&fixed::<16>(m).map_err(|e| format!("node matrix: {e}"))?)
        } else {
            let t = n.get("translation").map(fixed::<3>).transpose()?;
            let r = n.get("rotation").map(fixed::<4>).transpose()?;
            let s = n.get("scale").map(fixed::<3>).transpose()?;
            (
                t.unwrap_or([0.0; 3]),
                r.unwrap_or([0.0, 0.0, 0.0, 1.0]),
                s.unwrap_or([1.0; 3]),
            )
        };
        let weights = n
            .get("weights")
            .map(|w| w.array()?.iter().map(Value::float).collect::<R<Vec<f32>>>())
            .transpose()?;
        model.nodes.push(Node {
            name: name(n),
            parent: None,
            children: arr(n, "children")?
                .iter()
                .map(Value::integer)
                .collect::<R<Vec<usize>>>()?,
            translation,
            rotation,
            scale,
            mesh: opt_idx(n, "mesh", mesh_count, "mesh")?,
            skin: opt_idx(n, "skin", skin_count, "skin")?,
            weights,
        });
    }
    let node_count = model.nodes.len();
    for i in 0..node_count {
        let n_children = model.nodes[i].children.len();
        for k in 0..n_children {
            let c = model.nodes[i].children[k];
            let child = model
                .nodes
                .get_mut(c)
                .ok_or_else(|| format!("node child {c} outside {node_count}"))?;
            if child.parent.is_some() || c == i {
                return Err(format!("node {c} has more than one parent"));
            }
            child.parent = Some(i);
        }
    }
    // No cycles: each node is visited in O(1) amortized time (0: unvisited, 1: active, 2: ok).
    let mut state = vec![0u8; node_count];
    for start in 0..node_count {
        if state[start] == 2 {
            continue;
        }
        let mut at = start;
        while state[at] == 0 {
            state[at] = 1;
            match model.nodes[at].parent {
                Some(p) => at = p,
                None => break,
            }
        }
        if state[at] == 1 && model.nodes[at].parent.is_some() {
            return Err("the node hierarchy has a cycle".into());
        }
        let mut cur = start;
        while state[cur] == 1 {
            state[cur] = 2;
            match model.nodes[cur].parent {
                Some(p) => cur = p,
                None => break,
            }
        }
    }
    let scene = json.get("scene").map_or(Ok(0), Value::integer)?;
    model.roots = match arr(&json, "scenes")?.get(scene) {
        Some(s) => arr(s, "nodes")?
            .iter()
            .map(|n| {
                let n = n.integer()?;
                if n >= node_count {
                    return Err(format!("scene node {n} outside {node_count}"));
                }
                Ok(n)
            })
            .collect::<R<Vec<usize>>>()?,
        None => (0..node_count)
            .filter(|&i| model.nodes[i].parent.is_none())
            .collect(),
    };

    for s in arr(&json, "skins")? {
        let joints = arr(s, "joints")?
            .iter()
            .map(|j| {
                let j = j.integer()?;
                if j >= node_count {
                    return Err(format!("skin joint {j} outside {node_count} nodes"));
                }
                Ok(j)
            })
            .collect::<R<Vec<usize>>>()?;
        if joints.is_empty() || joints.len() > MAX_JOINTS {
            return Err(format!("a skin with {} joints", joints.len()));
        }
        let inverse_bind = match s.get("inverseBindMatrices") {
            Some(a) => {
                let m = doc.vecs::<16>(a.integer()?, "inverseBindMatrices")?;
                if m.len() != joints.len() {
                    return Err("inverseBindMatrices count differs from joints".into());
                }
                m
            }
            None => vec![
                [
                    1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.
                ];
                joints.len()
            ],
        };
        model.skins.push(Skin {
            name: name(s),
            joints,
            inverse_bind,
        });
    }
    // Joint indices stay inside their skin.
    for node in &model.nodes {
        if let (Some(m), Some(s)) = (node.mesh, node.skin) {
            let joints = model.skins[s].joints.len();
            for p in &model.meshes[m].primitives {
                if p.joints.iter().flatten().any(|&j| usize::from(j) >= joints) {
                    return Err(format!("mesh {m}: a joint index outside skin {s}"));
                }
            }
        }
    }

    let clips = arr(&json, "animations")?;
    if clips.len() > MAX_CLIPS {
        return Err(format!("more than {MAX_CLIPS} animations"));
    }
    for a in clips {
        let samplers = arr(a, "samplers")?;
        let mut clip = Clip {
            name: name(a),
            channels: vec![],
            duration: 0.0,
        };
        for c in arr(a, "channels")? {
            let target = need(c, "target")?;
            let Some(node) = target.get("node") else {
                continue;
            };
            let node = node.integer()?;
            if node >= node_count {
                return Err(format!("animation target node {node} outside {node_count}"));
            }
            let path = match need(target, "path")?.string()? {
                "translation" => Path::Translation,
                "rotation" => Path::Rotation,
                "scale" => Path::Scale,
                "weights" => Path::Weights,
                _ => continue,
            };
            let s = samplers
                .get(need(c, "sampler")?.integer()?)
                .ok_or("animation sampler out of range")?;
            let interpolation = match s.get("interpolation").map(Value::string).transpose()? {
                None | Some("LINEAR") => Interpolation::Linear,
                Some("STEP") => Interpolation::Step,
                Some("CUBICSPLINE") => Interpolation::CubicSpline,
                Some(o) => return Err(format!("interpolation {o}")),
            };
            let input = doc.floats(need(s, "input")?.integer()?)?;
            if input.width != 1 {
                return Err("animation input must be scalar times".into());
            }
            let times = input.values;
            if times.windows(2).any(|w| w[1] < w[0]) {
                return Err("animation times must not decrease".into());
            }
            let output = doc.floats(need(s, "output")?.integer()?)?;
            let width = match path {
                Path::Translation | Path::Scale => 3,
                Path::Rotation => 4,
                Path::Weights => {
                    let mesh = model.nodes[node]
                        .mesh
                        .ok_or("weights animated on a node without a mesh")?;
                    model.meshes[mesh].weights.len().max(1)
                }
            };
            let per_key = if interpolation == Interpolation::CubicSpline {
                3
            } else {
                1
            };
            if output.values.len() != times.len() * width * per_key {
                return Err(format!(
                    "animation output has {} values for {} keys of {width}",
                    output.values.len(),
                    times.len()
                ));
            }
            clip.duration = clip.duration.max(times.last().copied().unwrap_or(0.0));
            clip.channels.push(Channel {
                node,
                path,
                interpolation,
                times,
                values: output.values,
                width,
            });
        }
        model.clips.push(clip);
    }
    Ok(model)
}

fn sniff(b: &[u8]) -> Option<String> {
    if b.starts_with(b"\x89PNG") {
        Some("image/png".into())
    } else if b.starts_with(&[0xFF, 0xD8]) {
        Some("image/jpeg".into())
    } else if b.len() > 12 && &b[..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        Some("image/webp".into())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a GLB: binary data with aligned buffer views, and accessors over them.
    #[derive(Default)]
    struct Glb {
        bin: Vec<u8>,
        views: Vec<String>,
        accessors: Vec<String>,
    }
    impl Glb {
        fn view(&mut self, bytes: &[u8]) -> usize {
            while self.bin.len() % 4 != 0 {
                self.bin.push(0);
            }
            self.views.push(format!(
                r#"{{"buffer":0,"byteOffset":{},"byteLength":{}}}"#,
                self.bin.len(),
                bytes.len()
            ));
            self.bin.extend_from_slice(bytes);
            self.views.len() - 1
        }
        fn accessor(
            &mut self,
            bytes: &[u8],
            component: u32,
            kind: &str,
            count: usize,
            extra: &str,
        ) -> usize {
            let v = self.view(bytes);
            self.accessor_on(v, component, kind, count, extra)
        }
        fn accessor_on(
            &mut self,
            v: usize,
            component: u32,
            kind: &str,
            count: usize,
            extra: &str,
        ) -> usize {
            self.accessors.push(format!(
                r#"{{"bufferView":{v},"componentType":{component},"type":"{kind}","count":{count}{extra}}}"#
            ));
            self.accessors.len() - 1
        }
        fn f32s(&mut self, v: &[f32], kind: &str) -> usize {
            let width = match kind {
                "SCALAR" => 1,
                "VEC2" => 2,
                "VEC3" => 3,
                "VEC4" => 4,
                _ => 16,
            };
            let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            self.accessor(&b, 5126, kind, v.len() / width, "")
        }
        fn finish(mut self, rest: &str) -> Vec<u8> {
            let json = format!(
                r#"{{"asset":{{"version":"2.0"}},"buffers":[{{"byteLength":{}}}],"bufferViews":[{}],"accessors":[{}]{rest}}}"#,
                self.bin.len(),
                self.views.join(","),
                self.accessors.join(",")
            );
            let mut j = json.into_bytes();
            while j.len() % 4 != 0 {
                j.push(b' ');
            }
            while self.bin.len() % 4 != 0 {
                self.bin.push(0);
            }
            let total = 12 + 8 + j.len() + 8 + self.bin.len();
            let mut out = Vec::new();
            out.extend_from_slice(b"glTF");
            out.extend_from_slice(&2u32.to_le_bytes());
            out.extend_from_slice(&(total as u32).to_le_bytes());
            out.extend_from_slice(&(j.len() as u32).to_le_bytes());
            out.extend_from_slice(&0x4e4f534au32.to_le_bytes());
            out.extend_from_slice(&j);
            out.extend_from_slice(&(self.bin.len() as u32).to_le_bytes());
            out.extend_from_slice(&0x004e4942u32.to_le_bytes());
            out.extend_from_slice(&self.bin);
            out
        }
    }

    fn i16s(v: &[i16]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// A quantized, skinned, morphed, animated triangle under a matrix node, with a WebP image.
    fn sample(required: &str) -> Vec<u8> {
        let mut g = Glb::default();
        // Positions as normalized i16 (KHR_mesh_quantization), padded to 8-byte elements:
        // 32767 -> 1.0.
        let pv = g.view(&i16s(&[0, 0, 0, 0, 0, 32767, 0, 0, -32767, 0, 0, 0]));
        g.views[pv] = g.views[pv].replace('}', r#","byteStride":8}"#);
        let pos = g.accessor_on(pv, 5122, "VEC3", 3, r#","normalized":true"#);
        let joints = g.accessor(&[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], 5121, "VEC4", 3, "");
        let weights = g.accessor(
            &[255, 0, 0, 0, 255, 0, 0, 0, 128, 127, 0, 0],
            5121,
            "VEC4",
            3,
            r#","normalized":true"#,
        );
        let idx = g.accessor(&[0, 0, 1, 0, 2, 0], 5123, "SCALAR", 3, "");
        // A sparse morph target: only vertex 1 moves, by +z.
        let si = g.view(&[1, 0]);
        let sv = g.view(
            &[0f32, 0.0, 1.0]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        g.accessors.push(format!(
            r#"{{"componentType":5126,"type":"VEC3","count":3,"sparse":{{"count":1,"indices":{{"bufferView":{si},"componentType":5123}},"values":{{"bufferView":{sv}}}}}}}"#
        ));
        let target = g.accessors.len() - 1;
        let ibm = g.f32s(
            &[
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., //
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., -1., 0., 1.,
            ],
            "MAT4",
        );
        let times = g.f32s(&[0.0, 2.0], "SCALAR");
        // Rotations as normalized i16 quaternions: identity, then 90 degrees about z.
        let rot = g.accessor(
            &i16s(&[0, 0, 0, 32767, 0, 0, 23170, 23170]),
            5122,
            "VEC4",
            2,
            r#","normalized":true"#,
        );
        let img = g.view(b"RIFF\x00\x00\x00\x00WEBPVP8 ");
        g.finish(&format!(
            r#"{required},
            "images":[{{"bufferView":{img},"mimeType":"image/webp","name":"skin"}}],
            "textures":[{{"extensions":{{"EXT_texture_webp":{{"source":0}}}}}}],
            "materials":[{{"name":"cloth","pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}},"roughnessFactor":0.4}},
                          "alphaMode":"MASK","extensions":{{"KHR_materials_emissive_strength":{{"emissiveStrength":3}}}},
                          "emissiveFactor":[0.5,0,0]}}],
            "meshes":[{{"name":"tri","weights":[0.25],"extras":{{"targetNames":["lift"]}},
                       "primitives":[{{"attributes":{{"POSITION":{pos},"JOINTS_0":{joints},"WEIGHTS_0":{weights}}},
                                       "indices":{idx},"material":0,"targets":[{{"POSITION":{target}}}]}}]}}],
            "skins":[{{"joints":[0,1],"inverseBindMatrices":{ibm}}}],
            "nodes":[{{"name":"hip","children":[1,2]}},
                     {{"name":"knee","matrix":[0,1,0,0, -1,0,0,0, 0,0,1,0, 0,1,0,1]}},
                     {{"name":"body","mesh":0,"skin":0}}],
            "scenes":[{{"nodes":[0]}}],
            "animations":[{{"name":"bend","channels":[{{"sampler":0,"target":{{"node":1,"path":"rotation"}}}}],
                            "samplers":[{{"input":{times},"output":{rot}}}]}}]"#
        ))
    }

    #[test]
    fn loads_a_quantized_skinned_morphed_animated_model() {
        let m = load_glb(&sample(
            r#","extensionsRequired":["KHR_mesh_quantization","EXT_texture_webp"]"#,
        ))
        .unwrap();
        let p = &m.meshes[0].primitives[0];
        assert_eq!(p.positions[1], [0.0, 1.0, 0.0]);
        assert_eq!(p.positions[2], [-1.0, 0.0, 0.0]);
        assert_eq!(p.joints[1], [1, 0, 0, 0]);
        assert!((p.weights[2][0] - 128.0 / 255.0).abs() < 1e-6);
        assert_eq!(p.indices, [0, 1, 2]);
        assert_eq!(p.targets[0].positions[1], [0.0, 0.0, 1.0]);
        assert_eq!(p.targets[0].positions[0], [0.0; 3]);
        assert_eq!(
            (
                m.meshes[0].weights.clone(),
                m.meshes[0].target_names[0].as_str()
            ),
            (vec![0.25], "lift")
        );
        // The matrix node: 90 degrees about z, 1 m up.
        let knee = &m.nodes[1];
        assert_eq!(knee.parent, Some(0));
        assert!((knee.translation[1] - 1.0).abs() < 1e-6);
        assert!(
            (knee.rotation[2] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5,
            "{:?}",
            knee.rotation
        );
        assert_eq!(m.roots, [0]);
        assert_eq!(m.skins[0].joints, [0, 1]);
        assert_eq!(m.skins[0].inverse_bind[1][13], -1.0);
        let mat = &m.materials[0];
        assert_eq!(
            (mat.roughness, mat.alpha_mode, mat.alpha_cutoff),
            (0.4, AlphaMode::Mask, 0.5)
        );
        assert_eq!(mat.emissive, [1.5, 0.0, 0.0]);
        assert_eq!(
            mat.base_texture,
            Some(TextureRef {
                texture: 0,
                tex_coord: 0
            })
        );
        assert_eq!(
            (m.images[0].mime.as_str(), m.images[0].srgb),
            ("image/webp", true)
        );
        let clip = &m.clips[0];
        assert_eq!((clip.name.as_str(), clip.duration), ("bend", 2.0));
        assert!((clip.channels[0].values[6] - 23170.0 / 32767.0).abs() < 1e-6);
    }

    #[test]
    fn refuses_what_it_cannot_load() {
        let draco = sample(r#","extensionsRequired":["KHR_draco_mesh_compression"]"#);
        assert!(load_glb(&draco).unwrap_err().contains("Draco"));
        let other = sample(r#","extensionsRequired":["KHR_texture_transform"]"#);
        assert!(
            load_glb(&other)
                .unwrap_err()
                .contains("KHR_texture_transform")
        );
        // Corruptions of a valid file are errors, never panics.
        let good = sample("");
        assert!(load_glb(&good).is_ok());
        for cut in [0, 12, 20, good.len() / 2, good.len() - 1] {
            assert!(load_glb(&good[..cut]).is_err());
        }
        // Same-length edits of the JSON chunk keep the GLB's lengths valid.
        let edit = |from: &str, to: &str| {
            let (from, to) = (from.as_bytes(), to.as_bytes());
            assert_eq!(from.len(), to.len());
            let at = good
                .windows(from.len())
                .position(|w| w == from)
                .unwrap_or_else(|| panic!("{}", String::from_utf8_lossy(from)));
            let mut bad = good.clone();
            bad[at..at + to.len()].copy_from_slice(to);
            bad
        };
        for (from, to) in [
            (r#""children":[1,2]"#, r#""children":[1,1]"#),
            (r#""children":[1,2]"#, r#""children":[0,2]"#),
            (r#""indices":3"#, r#""indices":9"#),
            (r#""joints":[0,1]"#, r#""joints":[0,9]"#),
            (r#""mesh":0,"skin":0"#, r#""mesh":4,"skin":0"#),
            (r#""mimeType":"image/webp""#, r#""mimeType":"image/tiff""#),
        ] {
            assert!(load_glb(&edit(from, to)).is_err(), "{to} accepted");
        }
        assert!(
            load_glb(&edit(
                r#""roughnessFactor":0.4"#,
                r#""roughnessFactor":0.5"#
            ))
            .is_ok()
        );
    }
}
