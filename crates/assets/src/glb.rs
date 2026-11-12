//! Static glTF 2.0 GLB import: triangle meshes with normals and UVs, metallic-roughness
//! materials, embedded PNG textures with mip chains, and KHR_lights_punctual point lights.
//! Features the engine cannot represent are rejected with an explicit message.
use crate::{
    cooked::MAX_BYTES,
    json::{self, Value},
    mip::{self, Filter},
    png,
};
use ganbate_core::{
    Rgb, Vec3,
    world::{
        AlphaMode, Emission, Instance, MAX_LIGHTS, MAX_MATERIALS, MAX_NAME_BYTES, MAX_TEXTURES,
        MAX_TRIANGLES, Material, Mesh, PointLight, Shading, Transform, World,
    },
};
use std::collections::HashMap;

/// Extensions implemented here; any other required extension is rejected.
const SUPPORTED: [&str; 3] = [
    "KHR_lights_punctual",
    "KHR_materials_emissive_strength",
    "KHR_materials_unlit",
];
const MAX_MESHES: usize = 65_536;
const MAX_INSTANCES: usize = 250_000;
/// An axis of a node matrix no longer than this fraction of its longest axis is scaled to
/// (almost) nothing, and its direction is rounding noise: `import_nodes` leaves it out when it
/// checks that a matrix does not shear, and a decomposition into translation, rotation and
/// scale builds it from the other axes.
pub const FLAT_AXIS: f32 = 1e-6;
/// Largest cosine between two axes of a node matrix that still counts as a right angle.
const SKEW: f64 = 1e-4;

fn arr<'a>(o: &'a Value, k: &str) -> Result<&'a [Value], String> {
    o.get(k).map_or(Ok(&[][..]), Value::array)
}
fn need<'a>(o: &'a Value, k: &str) -> Result<&'a Value, String> {
    o.get(k).ok_or_else(|| format!("missing glTF field {k}"))
}
fn index<T>(a: &[T], i: usize) -> Result<&T, String> {
    a.get(i)
        .ok_or_else(|| format!("glTF index {i} outside collection of {}", a.len()))
}
fn opt_u(o: &Value, k: &str, default: usize) -> Result<usize, String> {
    o.get(k).map_or(Ok(default), Value::integer)
}
fn opt_f(o: Option<&Value>, k: &str, default: f32) -> Result<f32, String> {
    o.and_then(|o| o.get(k)).map_or(Ok(default), Value::float)
}
fn ext<'a>(o: &'a Value, name: &str) -> Option<&'a Value> {
    o.get("extensions").and_then(|e| e.get(name))
}
fn floats<const N: usize>(v: &Value) -> Result<[f32; N], String> {
    let a = v.array()?;
    if a.len() != N {
        return Err(format!("expected {N} coordinates"));
    }
    let mut out = [0.0; N];
    for (i, n) in a.iter().enumerate() {
        out[i] = n.float()?;
    }
    Ok(out)
}
fn word(b: &[u8], offset: usize) -> Result<u32, String> {
    let bytes = b.get(offset..offset + 4).ok_or("truncated GLB")?;
    Ok(u32::from_le_bytes(bytes.try_into().expect("four bytes")))
}
pub(crate) fn chunks(data: &[u8]) -> Result<(Value, &[u8]), String> {
    if data.len() > MAX_BYTES
        || data.get(..4) != Some(b"glTF")
        || word(data, 4)? != 2
        || word(data, 8)? as usize != data.len()
    {
        return Err("invalid GLB header, length or version".into());
    }
    let mut offset = 12;
    let mut document = None;
    let mut bin = None;
    while offset < data.len() {
        let length = word(data, offset)? as usize;
        let kind = word(data, offset + 4)?;
        offset += 8;
        let end = offset.checked_add(length).ok_or("chunk length overflow")?;
        let chunk = data.get(offset..end).ok_or("truncated GLB chunk")?;
        if length % 4 != 0 {
            return Err("unaligned GLB chunk".into());
        }
        match kind {
            0x4e4f534a if document.is_none() && offset == 20 => {
                document = Some(json::parse(chunk)?)
            }
            0x004e4942 if document.is_some() && bin.is_none() => bin = Some(chunk),
            0x4e4f534a | 0x004e4942 => return Err("duplicate or misplaced GLB chunk".into()),
            _ if document.is_none() => return Err("first GLB chunk must be JSON".into()),
            _ => {}
        }
        offset = end;
    }
    Ok((document.ok_or("missing JSON")?, bin.unwrap_or(&[])))
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Semantic {
    Position,
    Normal,
    TexCoord,
    Index,
}
struct Access<'a> {
    bytes: &'a [u8],
    stride: usize,
    count: usize,
    component: usize,
}
impl Access<'_> {
    #[inline]
    fn at(&self, i: usize) -> &[u8] {
        &self.bytes[i * self.stride..]
    }
    #[inline]
    fn read_f32(b: &[u8], off: usize) -> f32 {
        f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
    }
    fn vec3s(&self) -> Vec<Vec3> {
        if self.stride == 12 {
            self.bytes
                .chunks_exact(12)
                .take(self.count)
                .map(|b| {
                    Vec3(
                        Self::read_f32(b, 0),
                        Self::read_f32(b, 4),
                        Self::read_f32(b, 8),
                    )
                })
                .collect()
        } else {
            (0..self.count)
                .map(|i| {
                    let b = self.at(i);
                    Vec3(
                        Self::read_f32(b, 0),
                        Self::read_f32(b, 4),
                        Self::read_f32(b, 8),
                    )
                })
                .collect()
        }
    }
    fn uvs(&self) -> Vec<[f32; 2]> {
        match self.component {
            5121 => (0..self.count)
                .map(|i| {
                    let b = self.at(i);
                    [f32::from(b[0]) / 255., f32::from(b[1]) / 255.]
                })
                .collect(),
            5123 => (0..self.count)
                .map(|i| {
                    let b = self.at(i);
                    [
                        f32::from(u16::from_le_bytes([b[0], b[1]])) / 65535.,
                        f32::from(u16::from_le_bytes([b[2], b[3]])) / 65535.,
                    ]
                })
                .collect(),
            _ if self.stride == 8 => self
                .bytes
                .chunks_exact(8)
                .take(self.count)
                .map(|b| [Self::read_f32(b, 0), Self::read_f32(b, 4)])
                .collect(),
            _ => (0..self.count)
                .map(|i| {
                    let b = self.at(i);
                    [Self::read_f32(b, 0), Self::read_f32(b, 4)]
                })
                .collect(),
        }
    }
    fn indices(&self) -> Vec<u32> {
        match (self.component, self.stride) {
            (5121, 1) => self.bytes[..self.count]
                .iter()
                .map(|&b| u32::from(b))
                .collect(),
            (5121, _) => (0..self.count).map(|i| u32::from(self.at(i)[0])).collect(),
            (5123, 2) => self
                .bytes
                .chunks_exact(2)
                .take(self.count)
                .map(|b| u32::from(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            (5123, _) => (0..self.count)
                .map(|i| {
                    let b = self.at(i);
                    u32::from(u16::from_le_bytes([b[0], b[1]]))
                })
                .collect(),
            (_, 4) => self
                .bytes
                .chunks_exact(4)
                .take(self.count)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            _ => (0..self.count)
                .map(|i| {
                    let b = self.at(i);
                    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
                })
                .collect(),
        }
    }
}

/// The bytes of a buffer view in the GLB's embedded buffer.
fn view<'a>(doc: &Value, bin: &'a [u8], id: usize) -> Result<&'a [u8], String> {
    let view = index(arr(doc, "bufferViews")?, id)?;
    for name in ["EXT_meshopt_compression", "KHR_meshopt_compression"] {
        if ext(view, name).is_some() {
            return Err(format!("decode {name} in the source asset pipeline"));
        }
    }
    if need(view, "buffer")?.integer()? != 0 {
        return Err("external buffers unsupported".into());
    }
    let start = opt_u(view, "byteOffset", 0)?;
    let length = need(view, "byteLength")?.integer()?;
    let end = start.checked_add(length).ok_or("view range overflow")?;
    bin.get(start..end)
        .ok_or_else(|| "buffer view out of range".into())
}

fn access<'a>(
    doc: &Value,
    bin: &'a [u8],
    id: usize,
    semantic: Semantic,
) -> Result<Access<'a>, String> {
    let a = index(arr(doc, "accessors")?, id)?;
    if a.get("sparse").is_some() {
        return Err(format!("accessor {id}: sparse accessors are unsupported"));
    }
    let normalized = match a.get("normalized") {
        None | Some(Value::Bool(false)) => false,
        Some(Value::Bool(true)) => true,
        _ => return Err(format!("accessor {id}: normalized must be a boolean")),
    };
    let count = need(a, "count")?.integer()?;
    let component = need(a, "componentType")?.integer()?;
    let shape = need(a, "type")?.string()?;
    use Semantic::*;
    let elem = match (semantic, shape, component, normalized) {
        (Position | Normal, "VEC3", 5126, false) => 12,
        (TexCoord, "VEC2", 5126, false) => 8,
        (TexCoord, "VEC2", 5121, true) => 2,
        (TexCoord, "VEC2", 5123, true) => 4,
        (Index, "SCALAR", 5121, false) => 1,
        (Index, "SCALAR", 5123, false) => 2,
        (Index, "SCALAR", 5125, false) => 4,
        _ => {
            return Err(format!(
                "accessor {id}: unsupported {semantic:?} format {shape} of component type {component}{}",
                if normalized { " (normalized)" } else { "" }
            ));
        }
    };
    if count == 0 || count > MAX_TRIANGLES * 3 {
        return Err("accessor element count exceeds limits".into());
    }
    let view_id = need(a, "bufferView")
        .map_err(|_| format!("accessor {id}: accessors without a buffer view are unsupported"))?
        .integer()?;
    let range = view(doc, bin, view_id)?;
    let start = opt_u(index(arr(doc, "bufferViews")?, view_id)?, "byteOffset", 0)?;
    let offset = opt_u(a, "byteOffset", 0)?;
    let stride = opt_u(
        index(arr(doc, "bufferViews")?, view_id)?,
        "byteStride",
        elem,
    )?;
    let align = match component {
        5121 => 1,
        5123 => 2,
        _ => 4,
    };
    if stride < elem
        || stride > 252
        || stride % align != 0
        || start % align != 0
        || offset % align != 0
    {
        return Err("invalid accessor alignment or stride".into());
    }
    let last = (count - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(elem))
        .ok_or("accessor overflow")?;
    let bytes = range
        .get(offset..)
        .and_then(|s| s.get(..last))
        .ok_or("accessor outside buffer view")?;
    Ok(Access {
        bytes,
        stride,
        count,
        component,
    })
}

fn node_transform(n: &Value) -> Result<Transform, String> {
    if let Some(m) = n.get("matrix") {
        if ["translation", "rotation", "scale"]
            .iter()
            .any(|k| n.get(k).is_some())
        {
            return Err("matrix and TRS specified together".into());
        }
        let t = Transform(floats(m)?);
        if !t.valid() {
            return Err("invalid node matrix".into());
        }
        return Ok(t);
    }
    let t = n
        .get("translation")
        .map(floats)
        .transpose()?
        .unwrap_or([0.; 3]);
    let r = n
        .get("rotation")
        .map(floats)
        .transpose()?
        .unwrap_or([0., 0., 0., 1.]);
    let s = n.get("scale").map(floats).transpose()?.unwrap_or([1.; 3]);
    Transform::from_trs(t, r, s)
}

/// Whether a matrix shears: two of its axes (each longer than `FLAT_AXIS` of the longest) are
/// not at right angles. glTF requires node matrices to split into translation, rotation and
/// scale, which a sheared matrix cannot.
fn shears(t: &Transform) -> bool {
    let m = t.0.map(f64::from);
    let axes = [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]];
    let dot = |a: &[f64; 3], b: &[f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let len = axes.map(|a| dot(&a, &a).sqrt());
    let floor = len[0].max(len[1]).max(len[2]) * f64::from(FLAT_AXIS);
    [(0, 1), (0, 2), (1, 2)].into_iter().any(|(i, j)| {
        len[i] > floor && len[j] > floor && dot(&axes[i], &axes[j]).abs() > SKEW * len[i] * len[j]
    })
}

/// Point lights declared by KHR_lights_punctual: colour, intensity and range (0 = unlimited).
fn punctual_lights(doc: &Value) -> Result<Vec<(Rgb, f32, f32)>, String> {
    let Some(e) = ext(doc, "KHR_lights_punctual") else {
        return Ok(vec![]);
    };
    let lights = arr(e, "lights")?;
    if lights.len() > MAX_LIGHTS {
        return Err("too many KHR_lights_punctual lights".into());
    }
    let mut out = Vec::with_capacity(lights.len());
    for (i, l) in lights.iter().enumerate() {
        let kind = need(l, "type")?.string()?;
        if kind != "point" {
            return Err(format!(
                "KHR_lights_punctual light {i} is a {kind} light; only point lights are supported"
            ));
        }
        let color = l.get("color").map(floats).transpose()?.unwrap_or([1.; 3]);
        let range = opt_f(Some(l), "range", 0.)?;
        if l.get("range").is_some() && range <= 0. {
            return Err(format!(
                "KHR_lights_punctual light {i} has a nonpositive range"
            ));
        }
        out.push((color, opt_f(Some(l), "intensity", 1.)?, range));
    }
    Ok(out)
}

struct Importer<'a> {
    doc: &'a Value,
    bin: &'a [u8],
    world: World,
    materials: HashMap<usize, u32>,
    /// Engine texture per (glTF image, sRGB). An image used in both colour spaces is stored twice.
    textures: HashMap<(usize, bool), u32>,
    remaining: usize,
    /// Keep images in their file format (PNG, JPEG, WebP) for the host to decode.
    keep_images: bool,
}
impl Importer<'_> {
    fn spend(&mut self, bytes: u64) -> Result<(), String> {
        self.remaining = usize::try_from(bytes)
            .ok()
            .and_then(|b| self.remaining.checked_sub(b))
            .ok_or("decoded content exceeds memory budget")?;
        Ok(())
    }

    fn texture(&mut self, info: &Value, filter: Filter, role: &str) -> Result<u32, String> {
        let id = need(info, "index")?.integer()?;
        let set = opt_u(info, "texCoord", 0)?;
        if set != 0 {
            return Err(format!(
                "{role} texture uses TEXCOORD_{set}; only TEXCOORD_0 is supported"
            ));
        }
        if ext(info, "KHR_texture_transform").is_some() {
            return Err(format!(
                "{role} texture uses unsupported KHR_texture_transform"
            ));
        }
        let texture = index(arr(self.doc, "textures")?, id)?;
        let webp = ext(texture, "EXT_texture_webp").and_then(|w| w.get("source"));
        let source = if self.keep_images {
            webp.or_else(|| texture.get("source"))
        } else {
            texture.get("source")
        };
        let Some(source) = source else {
            let alternatives = match texture.get("extensions") {
                Some(Value::Object(m)) => m.keys().cloned().collect::<Vec<_>>().join(", "),
                _ => String::new(),
            };
            return Err(format!(
                "texture {id} has no PNG source (extensions: [{alternatives}])"
            ));
        };
        let image_id = source.integer()?;
        let key = (image_id, filter == Filter::Srgb);
        if let Some(&t) = self.textures.get(&key) {
            return Ok(t);
        }
        let image = index(arr(self.doc, "images")?, image_id)?;
        let name = image
            .get("name")
            .and_then(|n| n.string().ok())
            .unwrap_or("");
        if image.get("uri").is_some() {
            return Err(format!(
                "image {image_id} {name:?} is referenced by URI; embed it as image/png"
            ));
        }
        if self.keep_images {
            let bytes = view(self.doc, self.bin, need(image, "bufferView")?.integer()?)?;
            let mime = crate::image::mime(bytes)
                .ok_or_else(|| format!("image {image_id} {name:?} is not PNG, JPEG or WebP"))?;
            let (width, height) = crate::image::size(bytes)
                .filter(|&(w, h)| {
                    w <= ganbate_core::world::MAX_TEXTURE_SIZE
                        && h <= ganbate_core::world::MAX_TEXTURE_SIZE
                })
                .ok_or_else(|| format!("image {image_id} {name:?}: unreadable or too large"))?;
            self.spend(bytes.len() as u64)?;
            if self.world.textures.len() >= MAX_TEXTURES {
                return Err("too many textures".into());
            }
            self.world.textures.push(ganbate_core::world::Texture {
                width,
                height,
                srgb: filter == Filter::Srgb,
                levels: vec![],
                encoded: Some(ganbate_core::world::EncodedImage {
                    mime: mime.to_string(),
                    data: bytes.to_vec(),
                }),
            });
            let k = (self.world.textures.len() - 1) as u32;
            self.textures.insert(key, k);
            return Ok(k);
        }
        let mime = need(image, "mimeType")?.string()?;
        if mime != "image/png" {
            return Err(format!(
                "image {image_id} {name:?} has unsupported type {mime}; convert it to image/png"
            ));
        }
        let bytes = view(self.doc, self.bin, need(image, "bufferView")?.integer()?)?;
        let decoded = png::decode(bytes).map_err(|e| format!("image {image_id} {name:?}: {e}"))?;
        let size = mip::chain_bytes(decoded.width, decoded.height)
            .ok_or_else(|| format!("image {image_id} {name:?} exceeds texture limits"))?;
        self.spend(size)?;
        if self.world.textures.len() >= MAX_TEXTURES {
            return Err("too many textures".into());
        }
        let t = mip::texture(decoded.width, decoded.height, decoded.rgba, filter)?;
        self.world.textures.push(t);
        let k = (self.world.textures.len() - 1) as u32;
        self.textures.insert(key, k);
        Ok(k)
    }

    fn material(&mut self, id: usize) -> Result<u32, String> {
        if let Some(&m) = self.materials.get(&id) {
            return Ok(m);
        }
        let mat = index(arr(self.doc, "materials")?, id)?;
        let mut name = match mat.get("name") {
            Some(n) => n.string()?.to_string(),
            None => format!("material {id}"),
        };
        if name.len() > MAX_NAME_BYTES {
            let mut end = MAX_NAME_BYTES;
            while !name.is_char_boundary(end) {
                end -= 1;
            }
            name.truncate(end);
        }
        let m = self
            .material_fields(mat, name.clone())
            .map_err(|e| format!("material {id} {name:?}: {e}"))?;
        if self.world.materials.len() >= MAX_MATERIALS {
            return Err("too many materials".into());
        }
        self.world.materials.push(m);
        let k = (self.world.materials.len() - 1) as u32;
        self.materials.insert(id, k);
        Ok(k)
    }

    fn material_fields(&mut self, mat: &Value, name: String) -> Result<Material, String> {
        let pbr = mat.get("pbrMetallicRoughness");
        if pbr
            .and_then(|p| p.get("metallicRoughnessTexture"))
            .is_some()
        {
            return Err(
                "metallicRoughnessTexture is unsupported; bake it into metallicFactor and roughnessFactor"
                    .into(),
            );
        }
        if mat.get("occlusionTexture").is_some() {
            return Err("occlusionTexture is unsupported".into());
        }
        let base_color = pbr
            .and_then(|p| p.get("baseColorFactor"))
            .map(floats)
            .transpose()?
            .unwrap_or([1.; 4]);
        let base_color_texture = pbr
            .and_then(|p| p.get("baseColorTexture"))
            .map(|t| self.texture(t, Filter::Srgb, "base colour"))
            .transpose()?;
        let normal = mat.get("normalTexture");
        let normal_texture = normal
            .map(|t| self.texture(t, Filter::Normal, "normal"))
            .transpose()?;
        let strength = opt_f(
            ext(mat, "KHR_materials_emissive_strength"),
            "emissiveStrength",
            1.,
        )?;
        let emissive = mat
            .get("emissiveFactor")
            .map(floats::<3>)
            .transpose()?
            .unwrap_or([0.; 3])
            .map(|v| v * strength);
        let emissive_texture = mat
            .get("emissiveTexture")
            .map(|t| self.texture(t, Filter::Srgb, "emissive"))
            .transpose()?;
        let alpha = match mat.get("alphaMode").map(Value::string).transpose()? {
            None | Some("OPAQUE") => AlphaMode::Opaque,
            Some("MASK") => AlphaMode::Mask(opt_f(Some(mat), "alphaCutoff", 0.5)?),
            Some("BLEND") => AlphaMode::Blend,
            Some(other) => return Err(format!("unknown alphaMode {other:?}")),
        };
        let double_sided = match mat.get("doubleSided") {
            None => false,
            Some(Value::Bool(b)) => *b,
            Some(_) => return Err("doubleSided must be a boolean".into()),
        };
        let ganbate = mat.get("extras").and_then(|e| e.get("ganbate"));
        let shading = match ganbate
            .and_then(|a| a.get("shading"))
            .map(Value::string)
            .transpose()?
        {
            None if ext(mat, "KHR_materials_unlit").is_some() => Shading::Unlit,
            None | Some("standard") => Shading::Standard,
            Some("foliage") => Shading::Foliage,
            Some("water") => Shading::Water,
            Some("unlit") => Shading::Unlit,
            Some(other) => return Err(format!("unknown extras.ganbate.shading {other:?}")),
        };
        let emission = match ganbate
            .and_then(|a| a.get("emission"))
            .map(Value::string)
            .transpose()?
        {
            None | Some("constant") => Emission::Constant,
            Some("lamp") => Emission::Lamp,
            Some(other) => return Err(format!("unknown extras.ganbate.emission {other:?}")),
        };
        Ok(Material {
            name,
            base_color,
            base_color_texture,
            normal_texture,
            normal_scale: opt_f(normal, "scale", 1.)?,
            emissive,
            emissive_texture,
            metallic: opt_f(pbr, "metallicFactor", 1.)?,
            roughness: opt_f(pbr, "roughnessFactor", 1.)?,
            alpha,
            double_sided,
            shading,
            emission,
        })
    }

    fn primitive(&mut self, p: &Value) -> Result<Mesh, String> {
        if opt_u(p, "mode", 4)? != 4 || !arr(p, "targets")?.is_empty() {
            return Err("expected static triangle primitives".into());
        }
        if ext(p, "KHR_draco_mesh_compression").is_some() {
            return Err("decode KHR_draco_mesh_compression in the source asset pipeline".into());
        }
        let attributes = need(p, "attributes")?;
        let (doc, bin) = (self.doc, self.bin);
        let attribute = |name: &str, semantic| {
            attributes
                .get(name)
                .map(|v| access(doc, bin, v.integer()?, semantic))
                .transpose()
        };
        let ap = attribute("POSITION", Semantic::Position)?.ok_or("missing glTF field POSITION")?;
        let an = attribute("NORMAL", Semantic::Normal)?;
        let at = attribute("TEXCOORD_0", Semantic::TexCoord)?;
        let n = ap.count;
        if an.iter().chain(&at).any(|a| a.count != n) {
            return Err("vertex attribute counts differ from POSITION".into());
        }
        let ai = p
            .get("indices")
            .map(|v| access(doc, bin, v.integer()?, Semantic::Index))
            .transpose()?;
        let num = ai.as_ref().map_or(n, |a| a.count);
        if num % 3 != 0 || self.world.meshes.len() >= MAX_MESHES {
            return Err("invalid primitive count".into());
        }
        let per_vertex = 12 + an.as_ref().map_or(0, |_| 12) + at.as_ref().map_or(0, |_| 8);
        self.spend(per_vertex * n as u64 + 4 * num as u64)?;
        let positions = ap.vec3s();
        let normals = an.map_or_else(Vec::new, |a| a.vec3s());
        let uvs: Vec<[f32; 2]> = at.map_or_else(Vec::new, |a| a.uvs());
        let indices = match ai {
            Some(a) => a.indices(),
            None => (0..n as u32).collect(),
        };
        let (mut color, mut material) = ([1.; 3], None);
        if let Some(m) = p.get("material") {
            let k = self.material(m.integer()?)?;
            let mat = &self.world.materials[k as usize];
            let textured = mat.base_color_texture.is_some()
                || mat.normal_texture.is_some()
                || mat.emissive_texture.is_some();
            if textured && uvs.is_empty() {
                return Err(format!(
                    "primitive with textured material {:?} has no TEXCOORD_0",
                    mat.name
                ));
            }
            let [r, g, b, _] = mat.base_color;
            (color, material) = ([r, g, b], Some(k));
        }
        Ok(Mesh {
            positions,
            normals,
            uvs,
            indices,
            color,
            material,
        })
    }
}

struct Scene<'a> {
    nodes: &'a [Value],
    ranges: Vec<std::ops::Range<u32>>,
    lights: Vec<(Rgb, f32, f32)>,
    seen: Vec<bool>,
}
impl Scene<'_> {
    fn visit(
        &mut self,
        w: &mut World,
        id: usize,
        parent: Transform,
        depth: usize,
    ) -> Result<(), String> {
        if depth > 128 || w.instances.len() > MAX_INSTANCES {
            return Err("node hierarchy exceeds limits".into());
        }
        let node = index(self.nodes, id)?;
        if std::mem::replace(&mut self.seen[id], true) {
            return Err("cyclic or multiply-parented glTF node".into());
        }
        if node.get("skin").is_some() || node.get("weights").is_some() {
            return Err("skinned or morphed nodes require the animation importer".into());
        }
        if ext(node, "EXT_mesh_gpu_instancing").is_some() {
            return Err(
                "EXT_mesh_gpu_instancing is unsupported; expand instances into nodes".into(),
            );
        }
        let t = parent.compose(node_transform(node)?);
        if let Some(mesh) = node.get("mesh") {
            for m in index(&self.ranges, mesh.integer()?)?.clone() {
                if w.instances.len() >= MAX_INSTANCES {
                    return Err("too many mesh instances".into());
                }
                w.instances.push(Instance {
                    mesh: m,
                    transform: t,
                });
            }
        }
        if let Some(light) = ext(node, "KHR_lights_punctual") {
            let (color, intensity, range) = *index(&self.lights, need(light, "light")?.integer()?)?;
            if w.lights.len() >= MAX_LIGHTS {
                return Err("too many point lights".into());
            }
            w.lights.push(PointLight {
                position: t.point(Vec3(0., 0., 0.)),
                color,
                intensity,
                range,
            });
        }
        for child in arr(node, "children")? {
            self.visit(w, child.integer()?, t, depth + 1)?;
        }
        Ok(())
    }
}

pub fn import(data: &[u8]) -> Result<World, String> {
    import_with(data, false)
}

/// As [`import`]; with `keep_images` the textures stay in their file format (PNG, JPEG or WebP,
/// `EXT_texture_webp` included) for the host to decode, instead of being decoded to mip chains.
pub fn import_with(data: &[u8], keep_images: bool) -> Result<World, String> {
    let (doc, bin) = chunks(data)?;
    let (mut w, ranges, lights) = load(&doc, bin, keep_images)?;
    let nodes = arr(&doc, "nodes")?;
    let mut scene = Scene {
        nodes,
        ranges,
        lights,
        seen: vec![false; nodes.len()],
    };
    let selected = index(arr(&doc, "scenes")?, opt_u(&doc, "scene", 0)?)?;
    for root in arr(selected, "nodes")? {
        scene.visit(&mut w, root.integer()?, Transform::IDENTITY, 0)?;
    }
    w.validate()?;
    Ok(w)
}

/// A static GLB with its node tree kept, for importing it as a scene of separate objects.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NodeWorld {
    /// Meshes, materials and textures as [`import`] makes them; no instances or lights.
    pub world: World,
    /// The default scene's nodes, parents before children.
    pub nodes: Vec<SourceNode>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SourceNode {
    pub name: String,
    /// Index into [`NodeWorld::nodes`].
    pub parent: Option<usize>,
    /// Relative to the parent.
    pub local: Transform,
    /// The engine meshes this node's glTF mesh became (one per primitive); empty for none.
    pub meshes: std::ops::Range<u32>,
    /// A KHR_lights_punctual point light at the node's origin (`position` is zero).
    pub light: Option<PointLight>,
}

/// Imports a static GLB keeping every node: names, hierarchy and local transforms survive, so
/// a scene built from it can move each object on its own. The same features as [`import`] are
/// accepted and refused, and node matrices that shear are refused too, since a node's local
/// transform is kept as translation, rotation and scale.
pub fn import_nodes(data: &[u8], keep_images: bool) -> Result<NodeWorld, String> {
    let (doc, bin) = chunks(data)?;
    let (world, ranges, lights) = load(&doc, bin, keep_images)?;
    let nodes = arr(&doc, "nodes")?;
    let selected = index(arr(&doc, "scenes")?, opt_u(&doc, "scene", 0)?)?;
    let mut seen = vec![false; nodes.len()];
    let mut out: Vec<SourceNode> = Vec::with_capacity(nodes.len());
    // Depth-first, parents first: (glTF node, parent's index in `out`, depth).
    let mut todo: Vec<(usize, Option<usize>, usize)> = arr(selected, "nodes")?
        .iter()
        .rev()
        .map(|r| r.integer().map(|i| (i, None, 0)))
        .collect::<Result<_, _>>()?;
    while let Some((id, parent, depth)) = todo.pop() {
        if depth > 128 || out.len() >= MAX_INSTANCES {
            return Err("node hierarchy exceeds limits".into());
        }
        let node = index(nodes, id)?;
        if std::mem::replace(index_mut(&mut seen, id)?, true) {
            return Err("cyclic or multiply-parented glTF node".into());
        }
        if node.get("skin").is_some() || node.get("weights").is_some() {
            return Err("skinned or morphed nodes require the animation importer".into());
        }
        if ext(node, "EXT_mesh_gpu_instancing").is_some() {
            return Err(
                "EXT_mesh_gpu_instancing is unsupported; expand instances into nodes".into(),
            );
        }
        let meshes = match node.get("mesh") {
            Some(m) => index(&ranges, m.integer()?)?.clone(),
            None => 0..0,
        };
        let light = match ext(node, "KHR_lights_punctual") {
            Some(l) => {
                let (color, intensity, range) = *index(&lights, need(l, "light")?.integer()?)?;
                Some(PointLight {
                    position: Vec3(0., 0., 0.),
                    color,
                    intensity,
                    range,
                })
            }
            None => None,
        };
        let name = match node.get("name") {
            Some(n) => n.string()?.to_string(),
            None => String::new(),
        };
        if name.len() > MAX_NAME_BYTES {
            return Err("node name too long".into());
        }
        let local = node_transform(node)?;
        if node.get("matrix").is_some() && shears(&local) {
            return Err(format!(
                "node {id} {name:?}: its matrix shears; glTF node matrices must split into translation, rotation and scale"
            ));
        }
        let here = out.len();
        out.push(SourceNode {
            name,
            parent,
            local,
            meshes,
            light,
        });
        for child in arr(node, "children")?.iter().rev() {
            todo.push((child.integer()?, Some(here), depth + 1));
        }
    }
    world.validate()?;
    Ok(NodeWorld { world, nodes: out })
}

fn index_mut<T>(a: &mut [T], i: usize) -> Result<&mut T, String> {
    let n = a.len();
    a.get_mut(i)
        .ok_or_else(|| format!("glTF index {i} outside collection of {n}"))
}

/// Checks the document and imports its meshes, materials, textures and light definitions.
#[allow(clippy::type_complexity)]
fn load(
    doc: &Value,
    bin: &[u8],
    keep_images: bool,
) -> Result<(World, Vec<std::ops::Range<u32>>, Vec<(Rgb, f32, f32)>), String> {
    if need(need(doc, "asset")?, "version")?.string()? != "2.0" {
        return Err("expected glTF 2.0".into());
    }
    for name in arr(doc, "extensionsRequired")? {
        let name = name.string()?;
        let webp = keep_images && name == "EXT_texture_webp";
        if !(SUPPORTED.contains(&name) || webp) {
            return Err(format!("unsupported required glTF extension: {name}"));
        }
    }
    if !arr(doc, "animations")?.is_empty() {
        return Err("animated source requires the animation importer".into());
    }
    let buffers = arr(doc, "buffers")?;
    let declared = match buffers {
        [] => 0,
        [b] if b.get("uri").is_none() => need(b, "byteLength")?.integer()?,
        _ => return Err("expected one embedded GLB buffer".into()),
    };
    let bytes = bin
        .get(..declared)
        .ok_or("declared buffer length exceeds BIN chunk")?;
    if bin.len() - declared > 3 {
        return Err("unexpected excess BIN data".into());
    }
    let mut im = Importer {
        doc,
        bin: bytes,
        world: World::default(),
        materials: HashMap::new(),
        textures: HashMap::new(),
        remaining: MAX_BYTES,
        keep_images,
    };
    let mut ranges = Vec::new();
    for gltf_mesh in arr(doc, "meshes")? {
        let start = im.world.meshes.len() as u32;
        for p in need(gltf_mesh, "primitives")?.array()? {
            let mesh = im.primitive(p)?;
            im.world.meshes.push(mesh);
        }
        ranges.push(start..(im.world.meshes.len() as u32));
    }
    Ok((im.world, ranges, punctual_lights(doc)?))
}
