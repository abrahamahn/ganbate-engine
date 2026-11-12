//! Props: meshes, materials and textures a game creates while it runs (characters, interactive
//! props, architecture), placed by per-instance transforms and drawn with the cooked world in every
//! pass: shadow cascades, the water reflection and the main view, with the same unified
//! shading, tiled lamps and Look rules (a prop material's name is matched like a world
//! material's).
//!
//! Each frame the live, visible instances are batched by (material, mesh); their transforms go
//! to the frame slot's instance buffer in batch order and each batch is one instanced draw that
//! reads `props[first + instance_index]` (see `Prop` in common.wgsl). Meshes keep a CPU copy of
//! their triangles for [`Renderer::pick`].
//!
//! A skinned mesh ([`Renderer::set_prop_mesh_skin`]) keeps its vertices' joints and weights in
//! one buffer shared by every skinned mesh; each instance of it carries its joint matrices
//! ([`Renderer::set_prop_instance_joints`]), which go to the frame slot's joint buffer, and the
//! vertex stages move its vertices on the GPU (`skin_vertex` in common.wgsl).
//!
//! Handles are generational: using one after its object was destroyed is an error, never a
//! different object.
use std::collections::HashMap;

use super::View;
use super::look::SurfaceModel;
use super::math::{self, M4, V3};
use super::renderer::{Fnv, Renderer};
use super::scene::MaterialData;
use crate::gpu::pool::Pool;
use crate::gpu::{
    Attachment, BindGroupId, BufferDesc, BufferId, BufferUsage, Device, Format, Memory, Resource,
    Result, TextureDesc, TextureId, TextureUsage, f32_bytes, u32_bytes,
};
use crate::world::Shading;

handle!(
    /// A prop mesh.
    PropMesh
);
handle!(
    /// A prop material.
    PropMaterial
);
handle!(
    /// A prop texture.
    PropTexture
);
handle!(
    /// A placed prop: one mesh with one material.
    PropInstance
);
handle!(
    /// A game's prop shader: its own colour for overlay props (`MaterialDesc::shader`).
    PropShader
);

/// How many numbers a material hands its prop shader (`ShadeInput::p0`..`p4`).
pub const PROP_SHADER_PARAMS: usize = 20;

/// Vertices in model space and triangles (counter-clockwise front faces, like world meshes).
#[derive(Clone, Copy, Debug)]
pub struct MeshData<'a> {
    pub positions: &'a [V3],
    /// One per position; empty for flat shading from the face normals.
    pub normals: &'a [V3],
    /// One per position; empty for zeros.
    pub uvs: &'a [[f32; 2]],
    pub indices: &'a [u32],
}

/// How a prop's colour combines with what is behind it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlendMode {
    /// Covers what is behind it and writes depth.
    #[default]
    Opaque,
    /// Mixed by its alpha; sorted, and writes no depth.
    Alpha,
    /// Added, scaled by its alpha (glows, sparks); sorted, and writes no depth.
    Add,
}

/// How a prop material looks. Colours are linear.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialDesc {
    /// Look detail rules match this name.
    pub name: String,
    pub base_color: [f32; 4],
    pub emissive: V3,
    /// The emission follows the time of day's lamp level, like the world's lamps.
    pub lamp: bool,
    pub roughness: f32,
    pub metallic: f32,
    pub shading: Shading,
    /// Alpha-tested at this cutoff (`None`: opaque).
    pub alpha_cutoff: Option<f32>,
    pub base_texture: Option<PropTexture>,
    pub normal_texture: Option<PropTexture>,
    pub emissive_texture: Option<PropTexture>,
    pub normal_scale: f32,
    /// Only opaque scene materials cast shadows.
    pub casts_shadow: bool,
    pub blend: BlendMode,
    /// Drawn after grading, over the finished image, in exactly its colour and texture (unlit):
    /// labels, badges, rings. Overlays are never reflected and cast no shadows.
    pub overlay: bool,
    /// Hidden behind nearer scene surfaces (`false`: always drawn on top).
    pub depth_test: bool,
    /// An opaque overlay that tests the scene's depth also writes it: overlays drawn after it hide
    /// behind it where it is nearer (the solid pieces of a close-up drawn wholly in overlays).
    pub depth_write: bool,
    /// Draw order among blended and overlay props: lower first (any finite number, so one can be
    /// slipped between two others); ties draw far to near.
    pub order: f32,
    /// An overlay drawn by a game's prop shader instead of in its colour and texture. The shader
    /// reads all three textures (`ShadeInput::base`, `normal_map`, `emissive_map`): pictures it
    /// combines as it likes.
    pub shader: Option<PropShader>,
    /// What the prop shader reads (`ShadeInput::p0`..`p4`); unused without one.
    pub params: [f32; PROP_SHADER_PARAMS],
    /// Optional per-material surface response override (`None` inherits Look / DetailRule).
    pub surface: Option<SurfaceModel>,
    /// Optional clear-coat lobe weight (`0..1`).
    pub clearcoat: Option<f32>,
    /// Optional clear-coat roughness (`0.02..1`).
    pub clearcoat_roughness: Option<f32>,
    /// Optional fabric sheen weight (`0..1`).
    pub sheen: Option<f32>,
    /// Optional fabric sheen tint (`0..1`, `0` = white, `1` = base colour).
    pub sheen_tint: Option<f32>,
    /// Optional subsurface wrap / scatter weight (`0..1`).
    pub subsurface: Option<f32>,
}

impl Default for MaterialDesc {
    fn default() -> Self {
        Self {
            name: String::new(),
            base_color: [1.0; 4],
            emissive: [0.0; 3],
            lamp: false,
            roughness: 0.8,
            metallic: 0.0,
            shading: Shading::Standard,
            alpha_cutoff: None,
            base_texture: None,
            normal_texture: None,
            emissive_texture: None,
            normal_scale: 1.0,
            casts_shadow: true,
            blend: BlendMode::Opaque,
            overlay: false,
            depth_test: true,
            depth_write: false,
            order: 0.0,
            shader: None,
            params: [0.0; PROP_SHADER_PARAMS],
            surface: None,
            clearcoat: None,
            clearcoat_roughness: None,
            sheen: None,
            sheen_tint: None,
            subsurface: None,
        }
    }
}

impl MaterialDesc {
    /// A description from JSON, for hosts across a language boundary. Every field is optional
    /// and unknown fields are errors:
    ///
    /// ```json
    /// {"name": "stone", "color": [0.1, 0.3, 0.2, 1], "emissive": [0, 0, 0], "lamp": false,
    ///  "roughness": 0.8, "metallic": 0, "shading": "standard", "surface": "standard_ggx",
    ///  "alphaCutoff": 0.5, "baseTexture": 3, "normalTexture": 4, "emissiveTexture": 5,
    ///  "normalScale": 1, "castsShadow": true, "blend": "opaque", "overlay": false,
    ///  "depthTest": true, "depthWrite": false, "order": 0, "shader": 2, "params": [0.5, 1, 0, 0]}
    /// ```
    ///
    /// Colours are linear; `shading` is `standard`, `foliage`, `water` or `unlit`; `blend` is
    /// `opaque`, `alpha` or `add`; textures and the shader are raw handles ([`PropTexture`],
    /// [`PropShader`]); `params` holds up to twenty numbers (the rest are zero).
    pub fn from_json(text: &[u8]) -> Result<Self> {
        use crate::assets::json::{self, Value};
        const KNOWN: [&str; 26] = [
            "name",
            "color",
            "emissive",
            "lamp",
            "roughness",
            "metallic",
            "shading",
            "alphaCutoff",
            "baseTexture",
            "normalTexture",
            "emissiveTexture",
            "normalScale",
            "castsShadow",
            "blend",
            "overlay",
            "depthTest",
            "depthWrite",
            "order",
            "shader",
            "params",
            "surface",
            "clearcoat",
            "clearcoatRoughness",
            "sheen",
            "sheenTint",
            "subsurface",
        ];
        let v = json::parse(text).map_err(|e| format!("material: {e}"))?;
        let Value::Object(map) = &v else {
            return Err("material: expected a JSON object".into());
        };
        if let Some(unknown) = map.keys().find(|k| !KNOWN.contains(&k.as_str())) {
            return Err(format!("material: unknown field {unknown:?}"));
        }
        let field = |key: &str, e: String| format!("material {key}: {e}");
        let num = |key: &str| -> Result<Option<f32>> {
            v.get(key)
                .map(|x| x.float().map_err(|e| field(key, e)))
                .transpose()
        };
        let floats = |key: &str, n: usize| -> Result<Option<Vec<f32>>> {
            let Some(x) = v.get(key) else {
                return Ok(None);
            };
            let a = x.array().map_err(|e| field(key, e))?;
            if a.len() != n {
                return Err(field(key, format!("expected {n} numbers")));
            }
            a.iter()
                .map(|x| x.float().map_err(|e| field(key, e)))
                .collect::<Result<Vec<f32>>>()
                .map(Some)
        };
        let flag = |key: &str| -> Result<Option<bool>> {
            match v.get(key) {
                None => Ok(None),
                Some(Value::Bool(b)) => Ok(Some(*b)),
                Some(_) => Err(field(key, "expected true or false".into())),
            }
        };
        let texture = |key: &str| -> Result<Option<PropTexture>> {
            match v.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(x) => x
                    .unsigned()
                    .ok()
                    .and_then(|n| u32::try_from(n).ok())
                    .filter(|&n| n != 0)
                    .map(|n| Some(PropTexture(n)))
                    .ok_or_else(|| field(key, "expected a texture handle".into())),
            }
        };
        let mut d = MaterialDesc::default();
        if let Some(name) = v.get("name") {
            d.name = name.string().map_err(|e| field("name", e))?.into();
        }
        if let Some(c) = floats("color", 4)? {
            d.base_color = [c[0], c[1], c[2], c[3]];
        }
        if let Some(e) = floats("emissive", 3)? {
            d.emissive = [e[0], e[1], e[2]];
        }
        if let Some(b) = flag("lamp")? {
            d.lamp = b;
        }
        if let Some(x) = num("roughness")? {
            d.roughness = x.clamp(0.0, 1.0);
        }
        if let Some(x) = num("metallic")? {
            d.metallic = x.clamp(0.0, 1.0);
        }
        if let Some(s) = v.get("shading") {
            d.shading = match s.string().map_err(|e| field("shading", e))? {
                "standard" => Shading::Standard,
                "foliage" => Shading::Foliage,
                "water" => Shading::Water,
                "unlit" => Shading::Unlit,
                other => return Err(field("shading", format!("unknown {other:?}"))),
            };
        }
        d.alpha_cutoff = num("alphaCutoff")?;
        d.base_texture = texture("baseTexture")?;
        d.normal_texture = texture("normalTexture")?;
        d.emissive_texture = texture("emissiveTexture")?;
        if let Some(x) = num("normalScale")? {
            d.normal_scale = x;
        }
        if let Some(b) = flag("castsShadow")? {
            d.casts_shadow = b;
        }
        if let Some(b) = v.get("blend") {
            d.blend = match b.string().map_err(|e| field("blend", e))? {
                "opaque" => BlendMode::Opaque,
                "alpha" => BlendMode::Alpha,
                "add" => BlendMode::Add,
                other => return Err(field("blend", format!("unknown {other:?}"))),
            };
        }
        if let Some(b) = flag("overlay")? {
            d.overlay = b;
        }
        if let Some(b) = flag("depthTest")? {
            d.depth_test = b;
        }
        if let Some(b) = flag("depthWrite")? {
            d.depth_write = b;
        }
        if let Some(o) = num("order")? {
            if !o.is_finite() {
                return Err(field("order", "expected a finite number".into()));
            }
            d.order = o;
        }
        match v.get("shader") {
            None | Some(Value::Null) => {}
            Some(x) => {
                let n = x
                    .unsigned()
                    .ok()
                    .and_then(|n| u32::try_from(n).ok())
                    .filter(|&n| n != 0)
                    .ok_or_else(|| field("shader", "expected a shader handle".into()))?;
                d.shader = Some(PropShader(n));
            }
        }
        if let Some(x) = v.get("params") {
            let a = x.array().map_err(|e| field("params", e))?;
            if a.len() > PROP_SHADER_PARAMS {
                return Err(field(
                    "params",
                    format!("expected at most {PROP_SHADER_PARAMS} numbers"),
                ));
            }
            for (out, x) in d.params.iter_mut().zip(a) {
                *out = x.float().map_err(|e| field("params", e))?;
            }
        }
        if let Some(s) = v.get("surface") {
            let name = s.string().map_err(|e| field("surface", e))?;
            d.surface = Some(
                SurfaceModel::parse(name)
                    .ok_or_else(|| field("surface", format!("unknown {name:?}")))?,
            );
        }
        d.clearcoat = num("clearcoat")?.map(|x| x.clamp(0.0, 1.0));
        d.clearcoat_roughness = num("clearcoatRoughness")?.map(|x| x.clamp(0.02, 1.0));
        d.sheen = num("sheen")?.map(|x| x.clamp(0.0, 1.0));
        d.sheen_tint = num("sheenTint")?.map(|x| x.clamp(0.0, 1.0));
        d.subsurface = num("subsurface")?.map(|x| x.clamp(0.0, 1.0));
        Ok(d)
    }
}

/// Bytes per instance in the instance buffer (`Prop` in common.wgsl: two matrices, a tint and
/// its skin).
const PROP_BYTES: u64 = 160;
const MIN_CAPACITY: u32 = 64;
/// Bytes per joint matrix in a slot's joint buffer, and the fewest matrices it holds.
const JOINT_BYTES: u64 = 64;
const MIN_JOINTS: u32 = 64;
/// Bytes per skin record (a vertex has two: its joints, its weights), and the fewest held.
const SKIN_RECORD_BYTES: u64 = 16;
const MIN_SKIN_RECORDS: u32 = 64;

pub(super) struct MeshEntry {
    vertices: BufferId,
    indices: BufferId,
    index_count: u32,
    positions: Vec<V3>,
    uvs: Vec<[f32; 2]>,
    triangles: Vec<u32>,
    min: V3,
    max: V3,
    /// A skinned mesh's records in the shared skin buffer.
    skin: Option<SkinRange>,
    /// Counts changes to its vertices or skin, which its instances' pictures follow.
    revision: u32,
    /// A skinned mesh's per-joint bind-space bounds ([`joint_boxes`]): where its instances'
    /// posed vertices can lie.
    joint_boxes: Vec<Option<(V3, V3)>>,
    /// Counts changes its instances' shadows follow as a whole rather than by how far they
    /// moved ([`shadow_key`]): a new skin, or vertices moved out of their joints' boxes.
    shape: u32,
    /// How far, at most, a vertex moved in model space since the props' shadow maps were last
    /// drawn (each update's largest move, summed).
    drift: f32,
}

/// Where a skinned mesh's vertices' joints and weights lie in the shared skin records.
#[derive(Clone, Copy, Debug)]
struct SkinRange {
    first: u32,
    count: u32,
    /// The highest joint its vertices use: an instance needs more matrices than this.
    max_joint: u16,
}

pub(super) struct MaterialEntry {
    desc: MaterialDesc,
    data: MaterialData,
    /// `@group(1)` of the scene pipelines, or of the overlay pipelines for an overlay.
    group: BindGroupId,
    shadow_group: Option<BindGroupId>,
    detail: [f32; 16],
    /// Counts changes to it, which the pictures of instances using it follow.
    revision: u32,
}

impl MaterialEntry {
    fn layer(&self) -> Layer {
        if self.desc.overlay {
            Layer::Overlay
        } else if self.desc.blend == BlendMode::Opaque {
            Layer::Opaque
        } else {
            Layer::Blended
        }
    }
}

/// Where a prop is drawn: in the scene (opaque, batched, or blended, sorted) or over the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Layer {
    Opaque,
    Blended,
    Overlay,
}

struct InstanceEntry {
    mesh: PropMesh,
    material: PropMaterial,
    model: M4,
    normal: M4,
    tint: [f32; 4],
    visible: bool,
    /// Pick layers (bits); 0 is never picked.
    pick: u32,
    /// A skinned mesh's joint matrices for this instance (empty: drawn as modelled).
    joints: Vec<M4>,
    /// What its picture depends on as of the last frame ([`signature`]), and for how many
    /// frames since then it has stayed the same.
    sig: u64,
    still: u32,
    /// Its world bounds as drawn this frame (none while hidden or incomplete).
    world_bounds: Option<(V3, V3)>,
    /// Where its image and shadows lie in the cached picture, while it is drawn there.
    cached: Option<Rect>,
    shadow: ShadowTrack,
}

/// What follows an instance's shadow in the props' shadow maps, which are drawn again only when
/// a caster moved across them ([`track_cached_props`]).
#[derive(Default)]
struct ShadowTrack {
    /// The corners of the boxes holding its vertices as drawn this frame ([`posed_corners`]),
    /// and the most any matrix placing them stretches a length.
    corners: Vec<V3>,
    stretch: f32,
    /// Its shadow as the maps hold it, while they hold one.
    drawn: Option<DrawnShadow>,
}

/// What a caster's shadow in the props' shadow maps was drawn from ([`commit_prop_shadows`]).
struct DrawnShadow {
    key: u64,
    corners: Vec<V3>,
    bounds: (V3, V3),
}

/// One instanced draw of this frame: instances `first..first + count` of the buffer.
#[derive(Clone, Copy, Debug)]
pub(super) struct Batch {
    mesh: u32,
    material: u32,
    first: u32,
    count: u32,
    layer: Layer,
    /// Its instances are drawn into the cached picture rather than over it each frame.
    settled: bool,
}

/// Which opaque prop batches a draw takes: all of them, those in the cached picture, or those
/// drawn over it each frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PropSet {
    All,
    Settled,
    Live,
}

impl PropSet {
    fn takes(self, b: &Batch) -> bool {
        match self {
            Self::All => true,
            Self::Settled => b.settled,
            Self::Live => !b.settled,
        }
    }
}

/// A region of the picture in pixels: left, top, right, bottom (exclusive).
pub(super) type Rect = [u32; 4];

/// Frames an instance stays the same before it is drawn into the cached picture: what moves now
/// and then (a card dealt, a chip pushed) stays live while it moves and a little after.
const SETTLE_FRAMES: u32 = 8;

/// How far a caster may drift from where the props' shadow maps hold it, in texels of the finest
/// map it lands in, before they are drawn again ([`track_cached_props`]). Any drift shows: a
/// sphere half a texel off left 0.3 to 1 % of a frame more than one 8-bit step off the frame
/// drawn whole (up to 30 steps, along its own shadow's terminator and its shadow's edge), and a
/// fiftieth of a texel up to 0.1 %. So the maps are kept only while every caster stands still.
pub(super) const SHADOW_DRIFT_TEXELS: f32 = 0.0;

/// What an instance's picture depends on besides the frame: where and how it is drawn, its
/// joints, and its mesh, material and textures as they stand.
fn signature(
    inst: &InstanceEntry,
    mesh: &MeshEntry,
    mat: &MaterialEntry,
    edits: &HashMap<u32, u32>,
) -> u64 {
    let mut h = Fnv::new();
    h.write(&[u8::from(inst.visible)]);
    h.write(&f32_bytes(&inst.model));
    h.write(&f32_bytes(&inst.tint));
    for j in &inst.joints {
        h.write(&f32_bytes(j));
    }
    for n in [inst.mesh.0, mesh.revision, inst.material.0, mat.revision] {
        h.write(&n.to_le_bytes());
    }
    let d = &mat.desc;
    for t in [d.base_texture, d.normal_texture, d.emissive_texture]
        .into_iter()
        .flatten()
    {
        h.write(&t.0.to_le_bytes());
        h.write(&edits.get(&t.0).copied().unwrap_or(0).to_le_bytes());
    }
    h.finish()
}

/// Whether an instance is drawn into the cached picture: a shown opaque scene prop that has
/// stayed the same for a few frames (not water, whose surface moves with the clock).
fn settles(inst: &InstanceEntry, mat: &MaterialEntry) -> bool {
    inst.still >= SETTLE_FRAMES
        && inst.world_bounds.is_some()
        && mat.layer() == Layer::Opaque
        && mat.desc.shading != Shading::Water
}

/// Whether an instance is drawn into the props' shadow maps this frame
/// ([`Renderer::draw_prop_shadows`]).
fn casts(inst: &InstanceEntry, mat: &MaterialEntry) -> bool {
    inst.world_bounds.is_some() && mat.layer() == Layer::Opaque && mat.desc.casts_shadow
}

/// Whether an instance's mesh is drawn moved by its joints (as `prepare_props` writes it).
fn skinned(inst: &InstanceEntry, mesh: &MeshEntry) -> bool {
    mesh.skin
        .is_some_and(|s| inst.joints.len() > usize::from(s.max_joint))
}

/// What decides a caster's shadow as a whole rather than by how far it moved: its mesh and the
/// boxes [`posed_corners`] places (its joints' boxes, or for a mesh drawn unskinned its box),
/// and an alpha-tested material's cutoff, alpha and texture.
fn shadow_key(
    inst: &InstanceEntry,
    mesh: &MeshEntry,
    mat: &MaterialEntry,
    edits: &HashMap<u32, u32>,
) -> u64 {
    let mut h = Fnv::new();
    for n in [inst.mesh.0, mesh.shape, u32::from(skinned(inst, mesh))] {
        h.write(&n.to_le_bytes());
    }
    if !skinned(inst, mesh) {
        h.write(&f32_bytes(&[mesh.min, mesh.max].concat()));
    }
    if let Some(cutoff) = mat.desc.alpha_cutoff {
        h.write(&f32_bytes(&[cutoff, mat.desc.base_color[3]]));
        if let Some(t) = mat.desc.base_texture {
            h.write(&t.0.to_le_bytes());
            h.write(&edits.get(&t.0).copied().unwrap_or(0).to_le_bytes());
        }
    }
    h.finish()
}

/// How far (metres) any vertex of an instance can lie from where the props' shadow maps hold it:
/// `None` when it casts into them neither now nor as they were drawn; infinite when it came,
/// went or changed as a whole. Each vertex lies in the hull of one of its boxes' corners (all of
/// its joints' boxes when skinned, weights summing to one), and the matrices placing a box are
/// affine, so none moved farther than those corners did, plus its mesh's morphing stretched by
/// the matrices placing it now.
fn shadow_drift(
    inst: &InstanceEntry,
    mesh: &MeshEntry,
    mat: &MaterialEntry,
    edits: &HashMap<u32, u32>,
) -> Option<f32> {
    let track = &inst.shadow;
    match (&track.drawn, casts(inst, mat)) {
        (None, false) => None,
        (Some(d), true)
            if d.key == shadow_key(inst, mesh, mat, edits)
                && d.corners.len() == track.corners.len() =>
        {
            Some(farthest_move(&d.corners, &track.corners) + track.stretch * mesh.drift)
        }
        _ => Some(f32::INFINITY),
    }
}

/// A shadow map props cast into: world to its clip space, and its texels a side.
pub(super) type ShadowMap = (M4, u32);

/// The most texels a point in `b` moves across a shadow map per metre it moves, or `None` when
/// the box lies outside the map. Its clip x/w moves by `|row_x - x/w row_w| / w` per metre (the
/// rows' xyz), largest where `w` is least and `|x/w|` most, both at a corner.
fn texels_per_metre((m, size): &ShadowMap, b: (V3, V3)) -> Option<f32> {
    let clip = corners(b).map(|p| math::transform(m, [p[0], p[1], p[2], 1.0]));
    let outside = (0..2)
        .any(|k| clip.iter().all(|c| c[k] > c[3]) || clip.iter().all(|c| c[k] < -c[3]))
        || clip.iter().all(|c| c[3] <= 0.0);
    if outside {
        return None;
    }
    let w = clip.iter().map(|c| c[3]).fold(f32::INFINITY, f32::min);
    if w <= 1e-3 {
        return Some(f32::INFINITY);
    }
    let row = |r: usize| [m[r], m[4 + r], m[8 + r]];
    let per_metre = (0..2)
        .map(|k| {
            let n = clip.iter().map(|c| (c[k] / c[3]).abs()).fold(0.0, f32::max);
            let tilt = math::scale(row(3), n);
            math::length(math::add(row(k), tilt)).max(math::length(math::sub(row(k), tilt)))
        })
        .fold(0.0, f32::max);
    Some(per_metre / w * *size as f32 * 0.5)
}

/// The pixels the points cover in a `size` picture seen through `view_proj`, widened by
/// `margin` pixels; all of it when one lies behind the eye.
pub(super) fn screen_rect(
    points: &[V3],
    view_proj: &M4,
    size: (u32, u32),
    margin: f32,
) -> Option<Rect> {
    let (w, h) = (size.0 as f32, size.1 as f32);
    let (mut lo, mut hi) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
    for p in points {
        let c = math::transform(view_proj, [p[0], p[1], p[2], 1.0]);
        if c[3] <= 1e-4 {
            return Some([0, 0, size.0, size.1]);
        }
        let x = (c[0] / c[3] * 0.5 + 0.5) * w;
        let y = (c[1] / c[3] * 0.5 + 0.5) * h;
        lo = [lo[0].min(x), lo[1].min(y)];
        hi = [hi[0].max(x), hi[1].max(y)];
    }
    let r = [
        (lo[0] - margin).floor().clamp(0.0, w) as u32,
        (lo[1] - margin).floor().clamp(0.0, h) as u32,
        (hi[0] + margin).ceil().clamp(0.0, w) as u32,
        (hi[1] + margin).ceil().clamp(0.0, h) as u32,
    ];
    (r[0] < r[2] && r[1] < r[3]).then_some(r)
}

pub(super) fn corners((lo, hi): (V3, V3)) -> [V3; 8] {
    std::array::from_fn(|c| {
        [
            if c & 1 == 0 { lo[0] } else { hi[0] },
            if c & 2 == 0 { lo[1] } else { hi[1] },
            if c & 4 == 0 { lo[2] } else { hi[2] },
        ]
    })
}

/// The smallest region holding both.
pub(super) fn union(a: Rect, b: Rect) -> Rect {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

/// What the cached picture is drawn for: the eye and the picture size.
pub(super) struct CacheView<'a> {
    pub(super) view_proj: &'a M4,
    pub(super) size: (u32, u32),
}

/// What changed in the cached picture this frame ([`track_cached_props`]).
#[derive(Default)]
pub(super) struct CacheChanges {
    /// Regions to empty and draw again: where an instance entered or left the picture.
    pub(super) regions: Vec<Rect>,
    /// Where casters moved (or changed, came or went) since the props' shadow maps were last
    /// drawn: their bounds then and now, overlapping ones joined. When the maps are drawn
    /// again, the picture's pixels their shadows can reach are shaded again
    /// ([`Renderer::reshade`]).
    pub(super) moved: Vec<(V3, V3)>,
    /// The bounds of the props the picture holds.
    pub(super) settled: Vec<(V3, V3)>,
    /// A caster came, went, changed as a whole or moved across a shadow map it lands in (by
    /// `drift_texels` of the finest or more) since the maps were drawn: they are drawn again.
    /// Otherwise the maps and the picture keep it as they were.
    pub(super) casters_moved: bool,
}

/// Boxes that overlap joined into one, so casters moving together are followed as one.
pub(super) fn merge_boxes(mut boxes: Vec<(V3, V3)>) -> Vec<(V3, V3)> {
    let overlap = |a: &(V3, V3), b: &(V3, V3)| (0..3).all(|k| a.0[k] <= b.1[k] && b.0[k] <= a.1[k]);
    let mut merged = true;
    while merged {
        merged = false;
        'scan: for i in 0..boxes.len() {
            for j in i + 1..boxes.len() {
                if overlap(&boxes[i], &boxes[j]) {
                    let b = boxes.swap_remove(j);
                    let a = &mut boxes[i];
                    for k in 0..3 {
                        a.0[k] = a.0[k].min(b.0[k]);
                        a.1[k] = a.1[k].max(b.1[k]);
                    }
                    merged = true;
                    break 'scan;
                }
            }
        }
    }
    boxes
}

/// Follows the props into and out of the cached picture for this frame: the regions of it to
/// draw again where an instance entered or left it, and where casters moved since the props'
/// shadow maps were drawn (their shadows on what it holds change), measured in `maps` against
/// `drift_texels` ([`SHADOW_DRIFT_TEXELS`]). `full` when the whole picture is drawn again: the
/// records are then made afresh and no region or move is returned.
pub(super) fn track_cached_props(
    props: &mut Props,
    full: bool,
    view: &CacheView<'_>,
    maps: &[ShadowMap],
    drift_texels: f32,
) -> CacheChanges {
    let mut regions = std::mem::take(&mut props.vacated);
    let mut moved = std::mem::take(&mut props.vacated_boxes);
    let mut casters_moved = !moved.is_empty();
    let mut settled = vec![];
    let Props {
        instances,
        meshes,
        materials,
        texture_edits,
        ..
    } = props;
    for (_, inst) in instances.iter_mut() {
        let Some(mat) = materials.get(inst.material.0) else {
            regions.extend(inst.cached.take());
            continue;
        };
        if settles(inst, mat) {
            if full || inst.cached.is_none() {
                let image = inst
                    .world_bounds
                    .and_then(|b| screen_rect(&corners(b), view.view_proj, view.size, 2.0));
                regions.extend(image);
                inst.cached = Some(image.unwrap_or_default());
            }
            settled.extend(inst.world_bounds);
        } else {
            regions.extend(inst.cached.take());
        }
        let drift = meshes
            .get(inst.mesh.0)
            .and_then(|mesh| shadow_drift(inst, mesh, mat, texture_edits));
        if let Some(drift) = drift.filter(|d| *d != 0.0) {
            let then = inst.shadow.drawn.as_ref().map(|d| d.bounds);
            let reach = then.into_iter().chain(inst.world_bounds).reduce(|a, b| {
                (
                    std::array::from_fn(|k| a.0[k].min(b.0[k])),
                    std::array::from_fn(|k| a.1[k].max(b.1[k])),
                )
            });
            let finest = reach.map_or(0.0, |r| {
                maps.iter()
                    .filter_map(|m| texels_per_metre(m, r))
                    .fold(0.0, f32::max)
            });
            // A caster that came or went counts in or out of every map (infinite times 0).
            let texels = drift * finest;
            casters_moved |= texels.is_nan() || (texels > 0.0 && texels >= drift_texels);
            moved.extend(then.into_iter().chain(inst.world_bounds));
        }
    }
    if full {
        return CacheChanges {
            settled,
            casters_moved,
            ..CacheChanges::default()
        };
    }
    regions.retain(|r| r[0] < r[2] && r[1] < r[3]);
    CacheChanges {
        regions,
        moved: merge_boxes(moved),
        settled,
        casters_moved,
    }
}

/// Records that the props' shadow maps now hold every caster as this frame draws it: later
/// frames measure from here how far casters moved ([`shadow_drift`]).
pub(super) fn commit_prop_shadows(props: &mut Props) {
    let Props {
        instances,
        meshes,
        materials,
        texture_edits,
        vacated_boxes,
        ..
    } = props;
    vacated_boxes.clear();
    for (_, inst) in instances.iter_mut() {
        let caster = match (
            meshes.get(inst.mesh.0),
            materials.get(inst.material.0),
            inst.world_bounds,
        ) {
            (Some(mesh), Some(mat), Some(bounds)) if casts(inst, mat) => {
                Some((shadow_key(inst, mesh, mat, texture_edits), bounds))
            }
            _ => None,
        };
        let ShadowTrack { corners, drawn, .. } = &mut inst.shadow;
        match (caster, drawn.as_mut()) {
            (None, _) => *drawn = None,
            (Some((key, bounds)), Some(d)) => {
                d.key = key;
                d.corners.clone_from(corners);
                d.bounds = bounds;
            }
            (Some((key, bounds)), None) => {
                *drawn = Some(DrawnShadow {
                    key,
                    corners: corners.clone(),
                    bounds,
                })
            }
        }
    }
    for (_, m) in meshes.iter_mut() {
        m.drift = 0.0;
    }
}

/// Per joint, the bind-space bounds of the vertices it moves (`records`: per vertex its joints
/// and weights, as [`Renderer::set_prop_mesh_skin`] packs them).
fn joint_boxes(positions: &[V3], records: &[[f32; 4]]) -> Vec<Option<(V3, V3)>> {
    let mut boxes: Vec<Option<(V3, V3)>> = vec![];
    for (p, pair) in positions.iter().zip(records.chunks_exact(2)) {
        for (j, w) in pair[0].iter().zip(pair[1]) {
            if w <= 0.0 {
                continue;
            }
            let j = *j as usize;
            if boxes.len() <= j {
                boxes.resize(j + 1, None);
            }
            let b = boxes[j].get_or_insert((*p, *p));
            *b = (
                std::array::from_fn(|k| b.0[k].min(p[k])),
                std::array::from_fn(|k| b.1[k].max(p[k])),
            );
        }
    }
    boxes
}

/// The corners of the boxes holding an instance's vertices as drawn, into `out` (their bounds
/// are its world bounds): its mesh's box placed by its model, or for one drawn skinned each
/// joint's share of the mesh moved by that joint. Returns the most any of those placements
/// stretches a length.
fn posed_corners(inst: &InstanceEntry, mesh: &MeshEntry, out: &mut Vec<V3>) -> f32 {
    let mut stretch = 0.0f32;
    let mut place = |out: &mut Vec<V3>, m: &M4, b: (V3, V3)| {
        out.extend(corners(b).map(|p| math::transform_point(m, p)));
        stretch = stretch.max(max_stretch(m));
    };
    out.clear();
    if skinned(inst, mesh) {
        for (joint, b) in inst.joints.iter().zip(&mesh.joint_boxes) {
            if let Some(b) = b {
                place(out, &math::mul(&inst.model, joint), *b);
            }
        }
    }
    if out.is_empty() {
        place(out, &inst.model, (mesh.min, mesh.max));
    }
    stretch
}

/// The most `m` stretches a length: Gershgorin's bound on the largest eigenvalue of `AᵀA` (`A`
/// its linear part), square-rooted; exact for a rotation with a uniform scale.
fn max_stretch(m: &M4) -> f32 {
    let col = |c: usize| [m[c * 4], m[c * 4 + 1], m[c * 4 + 2]];
    let cols = [col(0), col(1), col(2)];
    cols.iter()
        .map(|a| cols.iter().map(|b| math::dot(*a, *b).abs()).sum::<f32>())
        .fold(0.0, f32::max)
        .sqrt()
}

/// A game's prop shader: its compiled stages and the overlay pipelines made from them so far.
pub(super) struct PropShaderEntry {
    /// Its fragment stage (the vertex stage is every prop's).
    fragment: ganbate_shader::Compiled,
    pipelines: [[Option<crate::gpu::PipelineId>; 3]; 3],
}

/// The overlay pipeline index of a blend mode: opaque, alpha, additive.
fn blend_index(blend: BlendMode) -> usize {
    match blend {
        BlendMode::Opaque => 0,
        BlendMode::Alpha => 1,
        BlendMode::Add => 2,
    }
}

/// The overlay pipeline index of how a material meets the scene's depth: drawn on top, tested,
/// tested and written.
fn depth_index(desc: &MaterialDesc) -> usize {
    match (desc.depth_test, desc.depth_write) {
        (false, _) => 0,
        (true, false) => 1,
        (true, true) => 2,
    }
}

/// A frame slot's instance and joint buffers and the bind groups that read them.
pub(super) struct PropSlot {
    buffer: BufferId,
    joints: BufferId,
    /// Main view and reflection (`@group(0)` of the prop scene pipeline).
    scene: [BindGroupId; 2],
    shadow: BindGroupId,
    /// `@group(0)` of the overlay pipelines.
    overlay: BindGroupId,
}

#[derive(Default)]
pub(super) struct Props {
    meshes: Pool<MeshEntry>,
    materials: Pool<MaterialEntry>,
    /// A game's prop shaders: their stages, and their overlay pipelines, [opaque, alpha,
    /// additive][depth test], each made the first time a prop draws with it.
    shaders: Pool<PropShaderEntry>,
    textures: Pool<TextureId>,
    instances: Pool<InstanceEntry>,
    slots: Vec<Option<PropSlot>>,
    capacity: u32,
    /// Joint matrices each slot's joint buffer holds.
    joint_capacity: u32,
    /// Edits of each prop texture's pixels (by handle), which the pictures of the props using
    /// it follow.
    texture_edits: HashMap<u32, u32>,
    /// Regions of the cached picture held by instances destroyed since the last frame.
    vacated: Vec<Rect>,
    /// Where the shadow maps hold casters destroyed since they were drawn (their shadows are
    /// drawn away).
    vacated_boxes: Vec<(V3, V3)>,
    /// Every skinned mesh's vertices' joints and weights, packed in mesh order of creation.
    skin_records: Vec<[f32; 4]>,
    /// Their GPU copy (and how many records it holds), rewritten when `skin_dirty`.
    skin_buffer: Option<(BufferId, u32)>,
    skin_dirty: bool,
    batches: Vec<Batch>,
    /// World bounds of this frame's visible instances.
    bounds: Option<(V3, V3)>,
}

impl Props {
    /// What each instance's picture depends on now and how long that has held, where its
    /// vertices lie as drawn ([`posed_corners`]) and its world bounds.
    fn pose_instances(&mut self) {
        let Self {
            instances,
            meshes,
            materials,
            texture_edits,
            ..
        } = self;
        for (_, inst) in instances.iter_mut() {
            let (Some(m), Some(mat)) = (meshes.get(inst.mesh.0), materials.get(inst.material.0))
            else {
                inst.world_bounds = None;
                inst.still = 0;
                continue;
            };
            let sig = signature(inst, m, mat, texture_edits);
            inst.still = if sig == inst.sig {
                inst.still.saturating_add(1)
            } else {
                0
            };
            inst.sig = sig;
            inst.world_bounds = None;
            if inst.visible {
                let mut posed = std::mem::take(&mut inst.shadow.corners);
                inst.shadow.stretch = posed_corners(inst, m, &mut posed);
                inst.world_bounds = Some(bounds(&posed));
                inst.shadow.corners = posed;
            }
        }
    }

    /// Removes an instance; what it held in the cached picture is drawn again.
    fn remove_instance(&mut self, id: u32) {
        if let Some(i) = self.instances.remove(id) {
            self.vacated.extend(i.cached);
            self.vacated_boxes.extend(i.shadow.drawn.map(|d| d.bounds));
        }
    }
}

fn normal_matrix(m: &M4) -> M4 {
    // The inverse transpose keeps normals perpendicular under non-uniform scale.
    let inv = math::invert(m).unwrap_or(math::IDENTITY);
    let mut t = [0.0f32; 16];
    for c in 0..4 {
        for r in 0..4 {
            t[c * 4 + r] = inv[r * 4 + c];
        }
    }
    t
}

/// The farthest any point moved from `from` to `to`.
fn farthest_move(from: &[V3], to: &[V3]) -> f32 {
    from.iter()
        .zip(to)
        .map(|(a, b)| math::length(math::sub(*a, *b)))
        .fold(0.0, f32::max)
}

pub(super) fn bounds(points: &[V3]) -> (V3, V3) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in points {
        for k in 0..3 {
            min[k] = min[k].min(p[k]);
            max[k] = max[k].max(p[k]);
        }
    }
    (min, max)
}

/// Interleaved position, normal, uv (the world vertex layout).
fn interleave(positions: &[V3], normals: &[V3], uvs: &[[f32; 2]], indices: &[u32]) -> Vec<f32> {
    let flat;
    let normals = if normals.is_empty() {
        let mut n = vec![[0.0f32; 3]; positions.len()];
        for t in indices.chunks_exact(3) {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            let face = math::cross(
                math::sub(positions[b], positions[a]),
                math::sub(positions[c], positions[a]),
            );
            for v in [a, b, c] {
                n[v] = math::add(n[v], face);
            }
        }
        flat = n.into_iter().map(math::normalize).collect::<Vec<_>>();
        &flat[..]
    } else {
        normals
    };
    let mut out = Vec::with_capacity(positions.len() * 8);
    for (i, p) in positions.iter().enumerate() {
        out.extend_from_slice(p);
        out.extend_from_slice(&normals[i]);
        out.extend_from_slice(uvs.get(i).unwrap_or(&[0.0, 0.0]));
    }
    out
}

/// Ray–triangle distance (Möller–Trumbore), both faces.
pub(super) fn ray_triangle(o: V3, d: V3, a: V3, b: V3, c: V3) -> Option<f32> {
    let e1 = math::sub(b, a);
    let e2 = math::sub(c, a);
    let p = math::cross(d, e2);
    let det = math::dot(e1, p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = math::sub(o, a);
    let u = math::dot(s, p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = math::cross(s, e1);
    let v = math::dot(d, q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = math::dot(e2, q) * inv;
    (t > 0.0).then_some(t)
}

/// Ray–box entry distance (slab test), or `None` when the ray misses.
pub(super) fn ray_box(o: V3, d: V3, min: V3, max: V3) -> Option<f32> {
    let (mut near, mut far) = (0.0f32, f32::INFINITY);
    for k in 0..3 {
        if d[k].abs() < 1e-12 {
            if o[k] < min[k] || o[k] > max[k] {
                return None;
            }
            continue;
        }
        let (a, b) = ((min[k] - o[k]) / d[k], (max[k] - o[k]) / d[k]);
        near = near.max(a.min(b));
        far = far.min(a.max(b));
        if near > far {
            return None;
        }
    }
    Some(near)
}

/// A cooked world instance under a point (`Renderer::pick_world`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldPick {
    /// Index into the cooked world's instances.
    pub instance: usize,
    /// Distance from the eye along the ray, in metres.
    pub distance: f32,
    pub point: V3,
}

/// A pick: the nearest prop under a screen point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pick {
    pub instance: PropInstance,
    /// Distance from the eye along the ray, in metres.
    pub distance: f32,
    pub point: V3,
}

fn normalize_pick_ray(origin: [f32; 3], dir: [f32; 3]) -> Option<(V3, V3)> {
    if !origin.iter().chain(&dir).all(|v| v.is_finite()) {
        return None;
    }
    let len_sq = math::dot(dir, dir);
    if !len_sq.is_finite() || len_sq < 1e-12 {
        return None;
    }
    let inv = 1.0 / len_sq.sqrt();
    let unit_dir = math::scale(dir, inv);
    unit_dir
        .iter()
        .all(|v| v.is_finite())
        .then_some((origin, unit_dir))
}

impl Props {
    fn pick_along_ray(&self, origin: [f32; 3], dir: [f32; 3], layers: u32) -> Option<Pick> {
        let (origin, dir) = normalize_pick_ray(origin, dir)?;
        if layers == 0 {
            return None;
        }
        // (on top, order, distance) of the best hit so far.
        let mut best: Option<(Pick, bool, f32)> = None;
        let better = |top: bool, order: f32, dist: f32, b: &Option<(Pick, bool, f32)>| match b {
            None => true,
            Some((p, btop, border)) => match (top, *btop) {
                (true, false) => true,
                (false, true) => false,
                (true, true) => order > *border || (order == *border && dist < p.distance),
                (false, false) => dist < p.distance,
            },
        };
        for (id, inst) in self.instances.iter() {
            if !inst.visible || inst.pick & layers == 0 {
                continue;
            }
            let (Some(mesh), Some(mat)) = (
                self.meshes.get(inst.mesh.0),
                self.materials.get(inst.material.0),
            ) else {
                continue;
            };
            let top = mat.desc.overlay && !mat.desc.depth_test;
            let order = mat.desc.order;
            // Into model space: the ray's origin and direction through the inverse transform.
            let Some(inv) = math::invert(&inst.model) else {
                continue;
            };
            let o = math::transform_point(&inv, origin);
            let tip = math::transform_point(&inv, math::add(origin, dir));
            let d = math::sub(tip, o);
            // A skinned prop is tested where its joints have moved it, as it is drawn.
            let skinned = self.skinned_positions(mesh, &inst.joints);
            let (positions, (min, max)) = match &skinned {
                Some(p) => (&p[..], bounds(p)),
                None => (&mesh.positions[..], (mesh.min, mesh.max)),
            };
            let Some(enter) = ray_box(o, d, min, max) else {
                continue;
            };
            if !better(top, order, enter, &best) {
                continue;
            }
            let mut hit: Option<f32> = None;
            for t in mesh.triangles.chunks_exact(3) {
                let (a, b, c) = (
                    positions[t[0] as usize],
                    positions[t[1] as usize],
                    positions[t[2] as usize],
                );
                if let Some(s) = ray_triangle(o, d, a, b, c)
                    && hit.is_none_or(|h| s < h)
                {
                    hit = Some(s);
                }
            }
            // `d` is the unit world direction mapped into model space, so `s` is in world metres.
            if let Some(s) = hit
                && better(top, order, s, &best)
            {
                let pick = Pick {
                    instance: PropInstance(id),
                    distance: s,
                    point: math::add(origin, math::scale(dir, s)),
                };
                best = Some((pick, top, order));
            }
        }
        best.map(|b| b.0)
    }

    /// A skinned mesh's vertices moved by `joints` as the vertex stages move them, or `None`
    /// when it is not skinned or `joints` cannot move it.
    fn skinned_positions(&self, mesh: &MeshEntry, joints: &[M4]) -> Option<Vec<V3>> {
        let skin = mesh.skin?;
        if joints.len() <= usize::from(skin.max_joint) {
            return None;
        }
        let records = self
            .skin_records
            .get(skin.first as usize..(skin.first + skin.count) as usize)?;
        let moved = mesh
            .positions
            .iter()
            .zip(records.chunks_exact(2))
            .map(|(&p, r)| {
                let (j, w) = (r[0], r[1]);
                if w.iter().sum::<f32>() < 1e-6 {
                    return p;
                }
                let mut out = [0.0f32; 3];
                for k in 0..4 {
                    if let Some(m) = joints.get(j[k] as usize)
                        && w[k] != 0.0
                    {
                        out = math::add(out, math::scale(math::transform_point(m, p), w[k]));
                    }
                }
                out
            })
            .collect();
        Some(moved)
    }
}

/// The clear value that leaves the bytes a 2D canvas filled with `color` (straight alpha,
/// sRGB-encoded channels from 0 to 1) holds: each channel rounded to 8 bits, and for an sRGB
/// texture, whose stores encode linear light, the colour channels decoded from those bytes.
fn canvas_clear(color: [f32; 4], srgb: bool) -> Result<[f32; 4]> {
    if !color.iter().all(|c| c.is_finite()) {
        return Err(format!("texture clear colour {color:?} is not finite"));
    }
    Ok(std::array::from_fn(|i| {
        let byte = (color[i].clamp(0.0, 1.0) * 255.0).round() as u8;
        if srgb && i < 3 {
            math::srgb_to_linear(byte)
        } else {
            f32::from(byte) / 255.0
        }
    }))
}

impl Renderer {
    fn prop_texture_id(&self, t: Option<PropTexture>, fallback: TextureId) -> Result<TextureId> {
        match t {
            None => Ok(fallback),
            Some(t) => self
                .props
                .textures
                .get(t.0)
                .copied()
                .ok_or_else(|| format!("{t:?} was destroyed")),
        }
    }

    /// An RGBA8 texture (sRGB colour or linear data such as normal maps) with a full mip chain.
    pub fn create_prop_texture(
        &mut self,
        device: &mut Device<'_>,
        size: (u32, u32),
        rgba: &[u8],
        srgb: bool,
    ) -> Result<PropTexture> {
        self.create_prop_texture_with(device, size, srgb, |d, t| d.write_texture(t, 0, 0, rgba))
    }

    /// As [`Self::create_prop_texture`], with mip 0 written by `fill` (an image the browser copies
    /// itself, a clear) instead of from bytes; the other mips are generated from it as there.
    pub fn create_prop_texture_with(
        &mut self,
        device: &mut Device<'_>,
        (width, height): (u32, u32),
        srgb: bool,
        fill: impl FnOnce(&mut Device<'_>, TextureId) -> Result<()>,
    ) -> Result<PropTexture> {
        let mips = 32 - width.max(height).leading_zeros();
        let t = device.create_texture(&TextureDesc {
            width,
            height,
            layers: 1,
            mips,
            format: if srgb {
                Format::Rgba8Srgb
            } else {
                Format::Rgba8Unorm
            },
            usage: TextureUsage::SAMPLED
                | TextureUsage::COPY_DST
                | TextureUsage::COPY_SRC
                | TextureUsage::COLOR_TARGET,
        })?;
        let written = fill(device, t).and_then(|()| device.generate_mips(t));
        if let Err(e) = written {
            device.destroy_texture(t);
            return Err(e);
        }
        Ok(PropTexture(self.props.textures.insert(t)?))
    }

    /// A prop texture whose texels are all `color`, filled on the GPU (mip 0 cleared, then the mip
    /// chain generated) with no pixels from the host: a placeholder until the real picture arrives
    /// through [`Self::update_prop_texture`] at the same size. `color` is straight alpha with
    /// sRGB-encoded channels from 0 to 1, like a canvas colour, and the texels hold the bytes a
    /// canvas filled with it would, so either kind of texture samples what that canvas would.
    pub fn create_prop_texture_clear(
        &mut self,
        device: &mut Device<'_>,
        size: (u32, u32),
        srgb: bool,
        color: [f32; 4],
    ) -> Result<PropTexture> {
        let value = canvas_clear(color, srgb)?;
        self.create_prop_texture_with(device, size, srgb, |d, t| {
            let mut e = d.encoder();
            e.begin_pass_slice(&[Attachment::clear(t, value)], None);
            e.end_pass();
            d.submit(e)
        })
    }

    /// Replaces a prop texture's pixels (same size) and its mips.
    pub fn update_prop_texture(
        &mut self,
        device: &mut Device<'_>,
        texture: PropTexture,
        rgba: &[u8],
    ) -> Result<()> {
        self.update_prop_texture_with(device, texture, |d, t| d.write_texture(t, 0, 0, rgba))
    }

    /// As [`Self::update_prop_texture`], with mip 0 written by `fill`. Nothing changes when it fails.
    pub fn update_prop_texture_with(
        &mut self,
        device: &mut Device<'_>,
        texture: PropTexture,
        fill: impl FnOnce(&mut Device<'_>, TextureId) -> Result<()>,
    ) -> Result<()> {
        let t = self.prop_texture_id(Some(texture), self.dummy)?;
        fill(device, t)?;
        let edits = self.props.texture_edits.entry(texture.0).or_default();
        *edits = edits.wrapping_add(1);
        device.generate_mips(t)
    }

    /// Destroys a texture; materials that use it must be updated or destroyed first.
    pub fn destroy_prop_texture(&mut self, device: &mut Device<'_>, texture: PropTexture) {
        if let Some(t) = self.props.textures.remove(texture.0) {
            device.destroy_texture(t);
        }
        self.props.texture_edits.remove(&texture.0);
    }

    pub fn create_prop_mesh(
        &mut self,
        device: &mut Device<'_>,
        mesh: &MeshData<'_>,
    ) -> Result<PropMesh> {
        let n = mesh.positions.len();
        if n == 0 || mesh.indices.is_empty() || mesh.indices.len() % 3 != 0 {
            return Err("prop mesh: needs vertices and whole triangles".into());
        }
        if (!mesh.normals.is_empty() && mesh.normals.len() != n)
            || (!mesh.uvs.is_empty() && mesh.uvs.len() != n)
        {
            return Err("prop mesh: normals and uvs need one entry per position".into());
        }
        if let Some(&i) = mesh.indices.iter().find(|&&i| i as usize >= n) {
            return Err(format!("prop mesh: index {i} is outside {n} vertices"));
        }
        let data = interleave(mesh.positions, mesh.normals, mesh.uvs, mesh.indices);
        let vertices = device.create_buffer(&BufferDesc {
            size: data.len() as u64 * 4,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
            memory: Memory::Device,
        })?;
        device.write_buffer(vertices, 0, &f32_bytes(&data))?;
        let indices = device.create_buffer(&BufferDesc {
            size: mesh.indices.len() as u64 * 4,
            usage: BufferUsage::INDEX | BufferUsage::COPY_DST,
            memory: Memory::Device,
        })?;
        device.write_buffer(indices, 0, &u32_bytes(mesh.indices))?;
        let (min, max) = bounds(mesh.positions);
        let entry = MeshEntry {
            vertices,
            indices,
            index_count: mesh.indices.len() as u32,
            positions: mesh.positions.to_vec(),
            uvs: if mesh.uvs.is_empty() {
                vec![[0.0; 2]; n]
            } else {
                mesh.uvs.to_vec()
            },
            triangles: mesh.indices.to_vec(),
            min,
            max,
            skin: None,
            revision: 0,
            joint_boxes: vec![],
            shape: 0,
            drift: 0.0,
        };
        Ok(PropMesh(self.props.meshes.insert(entry)?))
    }

    /// Skins a mesh: each vertex follows up to four of an instance's joints (`joints`, indices
    /// into the matrices [`Renderer::set_prop_instance_joints`] gives it) by `weights`, which
    /// are normalized here (a vertex weighing nothing stays as modelled). The GPU moves the
    /// vertices each frame; the mesh keeps its own positions for picking and bounds.
    pub fn set_prop_mesh_skin(
        &mut self,
        mesh: PropMesh,
        joints: &[[u16; 4]],
        weights: &[[f32; 4]],
    ) -> Result<()> {
        let m = self
            .props
            .meshes
            .get(mesh.0)
            .ok_or_else(|| format!("{mesh:?} was destroyed"))?;
        let n = m.positions.len();
        if joints.len() != n || weights.len() != n {
            return Err(format!(
                "set_prop_mesh_skin: {} joints and {} weights for {n} vertices",
                joints.len(),
                weights.len()
            ));
        }
        if weights.iter().flatten().any(|w| !w.is_finite() || *w < 0.0) {
            return Err("set_prop_mesh_skin: weights must be finite and not negative".into());
        }
        let mut records = Vec::with_capacity(n * 2);
        let mut max_joint = 0u16;
        for (js, ws) in joints.iter().zip(weights) {
            let total: f32 = ws.iter().sum();
            let norm = if total > 1e-6 { 1.0 / total } else { 0.0 };
            records.push(js.map(f32::from));
            records.push(ws.map(|w| w * norm));
            for (&j, &w) in js.iter().zip(ws) {
                if w * norm > 0.0 {
                    max_joint = max_joint.max(j);
                }
            }
        }
        let boxes = joint_boxes(&m.positions, &records);
        self.drop_prop_mesh_skin(mesh);
        let first = self.props.skin_records.len() as u32;
        self.props.skin_records.extend(records);
        if let Some(m) = self.props.meshes.get_mut(mesh.0) {
            m.revision = m.revision.wrapping_add(1);
            m.shape = m.shape.wrapping_add(1);
            m.joint_boxes = boxes;
            m.skin = Some(SkinRange {
                first,
                count: (n * 2) as u32,
                max_joint,
            });
        }
        self.props.skin_dirty = true;
        Ok(())
    }

    /// Takes a mesh's records out of the shared skin records, closing the gap.
    fn drop_prop_mesh_skin(&mut self, mesh: PropMesh) {
        let Some(range) = self
            .props
            .meshes
            .get_mut(mesh.0)
            .and_then(|m| m.skin.take())
        else {
            return;
        };
        let (start, end) = (range.first as usize, (range.first + range.count) as usize);
        self.props.skin_records.drain(start..end);
        for (_, m) in self.props.meshes.iter_mut() {
            if let Some(s) = &mut m.skin
                && s.first >= range.first + range.count
            {
                s.first -= range.count;
            }
        }
        self.props.skin_dirty = true;
    }

    /// New vertex positions and normals for a mesh (a skinned or morphed character each frame);
    /// the triangles and uvs stay.
    pub fn update_prop_mesh(
        &mut self,
        device: &mut Device<'_>,
        mesh: PropMesh,
        positions: &[V3],
        normals: &[V3],
    ) -> Result<()> {
        let m = self
            .props
            .meshes
            .get_mut(mesh.0)
            .ok_or_else(|| format!("{mesh:?} was destroyed"))?;
        if positions.len() != m.positions.len()
            || (!normals.is_empty() && normals.len() != positions.len())
        {
            return Err("update_prop_mesh: the vertex count must stay the same".into());
        }
        let data = interleave(positions, normals, &m.uvs, &m.triangles);
        device.write_buffer(m.vertices, 0, &f32_bytes(&data))?;
        m.drift += farthest_move(&m.positions, positions);
        m.positions.copy_from_slice(positions);
        (m.min, m.max) = bounds(positions);
        m.revision = m.revision.wrapping_add(1);
        if let Some(skin) = m.skin {
            let records = &self.props.skin_records;
            let range = skin.first as usize..(skin.first + skin.count) as usize;
            let boxes = joint_boxes(&m.positions, &records[range]);
            if boxes != m.joint_boxes {
                m.shape = m.shape.wrapping_add(1);
                m.joint_boxes = boxes;
            }
        }
        Ok(())
    }

    /// New positions and normals for the vertices `first..first + positions.len()` of a mesh
    /// (a morphed face: only the vertices its targets move), written in place; the rest stay.
    pub fn update_prop_mesh_range(
        &mut self,
        device: &mut Device<'_>,
        mesh: PropMesh,
        first: usize,
        positions: &[V3],
        normals: &[V3],
    ) -> Result<()> {
        let m = self
            .props
            .meshes
            .get_mut(mesh.0)
            .ok_or_else(|| format!("{mesh:?} was destroyed"))?;
        let end = first + positions.len();
        if end > m.positions.len() || normals.len() != positions.len() {
            return Err(
                "update_prop_mesh_range: the range must lie in the mesh, a normal a vertex".into(),
            );
        }
        if positions.is_empty() {
            return Ok(());
        }
        let mut data = Vec::with_capacity(positions.len() * 8);
        for ((p, n), uv) in positions.iter().zip(normals).zip(&m.uvs[first..end]) {
            data.extend_from_slice(p);
            data.extend_from_slice(n);
            data.extend_from_slice(uv);
        }
        device.write_buffer(m.vertices, first as u64 * 32, &f32_bytes(&data))?;
        m.drift += farthest_move(&m.positions[first..end], positions);
        m.positions[first..end].copy_from_slice(positions);
        (m.min, m.max) = bounds(&m.positions);
        m.revision = m.revision.wrapping_add(1);
        // A morph each frame: its joints' bounds only widen to hold the moved vertices (they
        // stay bounds; a full update makes them afresh).
        if let Some(skin) = m.skin {
            let records = &self.props.skin_records;
            let at = skin.first as usize + first * 2;
            let moved = joint_boxes(positions, &records[at..at + positions.len() * 2]);
            for (held, b) in m.joint_boxes.iter_mut().zip(moved) {
                if let Some(b) = b {
                    let was = *held;
                    let h = held.get_or_insert(b);
                    *h = (
                        std::array::from_fn(|k| h.0[k].min(b.0[k])),
                        std::array::from_fn(|k| h.1[k].max(b.1[k])),
                    );
                    if *held != was {
                        m.shape = m.shape.wrapping_add(1);
                    }
                }
            }
        }
        Ok(())
    }

    /// Destroys a mesh and every instance of it.
    pub fn destroy_prop_mesh(&mut self, device: &mut Device<'_>, mesh: PropMesh) {
        self.drop_prop_mesh_skin(mesh);
        if let Some(m) = self.props.meshes.remove(mesh.0) {
            let users: Vec<u32> = self
                .props
                .instances
                .iter()
                .filter(|(_, i)| i.mesh == mesh)
                .map(|(id, _)| id)
                .collect();
            for id in users {
                self.props.remove_instance(id);
            }
            device.destroy_buffer(m.vertices);
            device.destroy_buffer(m.indices);
        }
    }

    fn material_entry(
        &mut self,
        device: &mut Device<'_>,
        desc: &MaterialDesc,
    ) -> Result<MaterialEntry> {
        if desc.depth_write
            && (!desc.overlay || !desc.depth_test || desc.blend != BlendMode::Opaque)
        {
            return Err(
                "material: depthWrite is for an opaque overlay that tests depth (overlay, \
                 depthTest, blend opaque)"
                    .into(),
            );
        }
        if let Some(shader) = desc.shader {
            if !desc.overlay {
                return Err("material: a prop shader draws overlays only (set overlay)".into());
            }
            if self.props.shaders.get(shader.0).is_none() {
                return Err(format!("{shader:?} does not exist"));
            }
        }
        let shading = match desc.shading {
            Shading::Standard => 0.0,
            Shading::Foliage => 1.0,
            Shading::Water => 2.0,
            Shading::Unlit => 3.0,
        };
        let data = MaterialData {
            base_color: desc.base_color,
            emissive: [
                desc.emissive[0],
                desc.emissive[1],
                desc.emissive[2],
                if desc.lamp { 1.0 } else { 0.0 },
            ],
            params: [
                desc.alpha_cutoff.unwrap_or(-1.0),
                desc.roughness,
                desc.metallic,
                shading,
            ],
            flags: [
                f32::from(u8::from(desc.base_texture.is_some())),
                f32::from(u8::from(desc.normal_texture.is_some())),
                f32::from(u8::from(desc.emissive_texture.is_some())),
                desc.normal_scale,
            ],
            base: None,
            normal: None,
            emission: None,
        };
        let base = self.prop_texture_id(desc.base_texture, self.white)?;
        let normal = self.prop_texture_id(desc.normal_texture, self.flat)?;
        let emissive = self.prop_texture_id(desc.emissive_texture, self.white)?;
        let p = &self.pipelines;
        // A prop shader may read all three textures, as a scene prop's material binds them.
        let group = if desc.overlay && desc.shader.is_none() {
            device.create_bind_group(
                p.overlay[0][0],
                1,
                &[
                    (0, Resource::Texture(base)),
                    (3, Resource::Sampler(self.material_sampler)),
                ],
            )?
        } else {
            device.create_bind_group(
                p.scene_prop,
                1,
                &[
                    (0, Resource::Texture(base)),
                    (1, Resource::Texture(normal)),
                    (2, Resource::Texture(emissive)),
                    (3, Resource::Sampler(self.material_sampler)),
                ],
            )?
        };
        let opaque = !desc.overlay && desc.blend == BlendMode::Opaque;
        let shadow_group = if opaque && desc.casts_shadow && desc.alpha_cutoff.is_some() {
            Some(device.create_bind_group(
                p.shadow_prop_masked,
                1,
                &[
                    (0, Resource::Texture(base)),
                    (3, Resource::Sampler(self.material_sampler)),
                ],
            )?)
        } else {
            None
        };
        Ok(MaterialEntry {
            desc: desc.clone(),
            data,
            group,
            shadow_group,
            detail: super::renderer::detail_block(&self.look, &desc.name).0,
            revision: 0,
        })
    }

    /// Compiles a game's prop shader: WGSL that defines `fn shade(in: ShadeInput) -> vec4f`
    /// (render/shaders/prop_shader_input.wgsl) and may call the scene shader's functions. Each of
    /// its overlay pipelines (a blend mode, with the scene's depth test or without) is made the
    /// first time a prop draws with it, and lives as long as the renderer.
    pub fn create_prop_shader(
        &mut self,
        _device: &mut Device<'_>,
        source: &str,
    ) -> Result<PropShader> {
        let fragment = super::renderer::prop_shader_stage(&self.pipelines, source)?;
        Ok(PropShader(self.props.shaders.insert(PropShaderEntry {
            fragment,
            pipelines: [[None; 3]; 3],
        })?))
    }

    /// The overlay pipelines this frame's prop-shader batches need and do not have yet.
    fn make_prop_shader_pipelines(&mut self, device: &mut Device<'_>) -> Result<()> {
        let mut wanted = vec![];
        for b in self
            .props
            .batches
            .iter()
            .filter(|b| b.layer == Layer::Overlay)
        {
            let Some(mat) = self.props.materials.get(b.material) else {
                continue;
            };
            let Some(shader) = mat.desc.shader else {
                continue;
            };
            let (blend, depth) = (blend_index(mat.desc.blend), depth_index(&mat.desc));
            if self
                .props
                .shaders
                .get(shader.0)
                .is_some_and(|s| s.pipelines[blend][depth].is_none())
                && !wanted.contains(&(shader, blend, depth))
            {
                wanted.push((shader, blend, depth));
            }
        }
        for (shader, blend, depth) in wanted {
            let Some(entry) = self.props.shaders.get(shader.0) else {
                continue;
            };
            let made = super::renderer::prop_shader_pipeline(
                device,
                &self.pipelines,
                &entry.fragment,
                blend,
                depth,
            )?;
            if let Some(entry) = self.props.shaders.get_mut(shader.0) {
                entry.pipelines[blend][depth] = Some(made);
            }
        }
        Ok(())
    }

    /// Replaces a material's prop-shader parameters, without rebuilding anything (per frame).
    pub fn set_prop_material_params(
        &mut self,
        material: PropMaterial,
        params: &[f32; PROP_SHADER_PARAMS],
    ) -> Result<()> {
        let m = self
            .props
            .materials
            .get_mut(material.0)
            .ok_or_else(|| format!("{material:?} was destroyed"))?;
        if m.desc.params != *params {
            m.desc.params = *params;
            m.revision = m.revision.wrapping_add(1);
        }
        Ok(())
    }

    pub fn create_prop_material(
        &mut self,
        device: &mut Device<'_>,
        desc: &MaterialDesc,
    ) -> Result<PropMaterial> {
        let entry = self.material_entry(device, desc)?;
        Ok(PropMaterial(self.props.materials.insert(entry)?))
    }

    /// Changes a material in place (every instance that uses it changes with it).
    pub fn update_prop_material(
        &mut self,
        device: &mut Device<'_>,
        material: PropMaterial,
        desc: &MaterialDesc,
    ) -> Result<()> {
        if self.props.materials.get(material.0).is_none() {
            return Err(format!("{material:?} was destroyed"));
        }
        let entry = self.material_entry(device, desc)?;
        if let Some(held) = self.props.materials.get_mut(material.0) {
            let revision = held.revision.wrapping_add(1);
            let old = std::mem::replace(held, entry);
            held.revision = revision;
            device.destroy_bind_group(old.group);
            if let Some(g) = old.shadow_group {
                device.destroy_bind_group(g);
            }
        }
        Ok(())
    }

    /// The description a material was created or last updated with.
    pub fn prop_material(&self, material: PropMaterial) -> Result<&MaterialDesc> {
        self.props
            .materials
            .get(material.0)
            .map(|m| &m.desc)
            .ok_or_else(|| format!("{material:?} was destroyed"))
    }

    /// Destroys a material and every instance that uses it.
    pub fn destroy_prop_material(&mut self, device: &mut Device<'_>, material: PropMaterial) {
        if let Some(m) = self.props.materials.remove(material.0) {
            let users: Vec<u32> = self
                .props
                .instances
                .iter()
                .filter(|(_, i)| i.material == material)
                .map(|(id, _)| id)
                .collect();
            for id in users {
                self.props.remove_instance(id);
            }
            device.destroy_bind_group(m.group);
            if let Some(g) = m.shadow_group {
                device.destroy_bind_group(g);
            }
        }
    }

    /// Places `mesh` with `material` at `model` (column-major, model to world), visible.
    pub fn create_prop_instance(
        &mut self,
        mesh: PropMesh,
        material: PropMaterial,
        model: &M4,
    ) -> Result<PropInstance> {
        if self.props.meshes.get(mesh.0).is_none() {
            return Err(format!("{mesh:?} was destroyed"));
        }
        if self.props.materials.get(material.0).is_none() {
            return Err(format!("{material:?} was destroyed"));
        }
        let entry = InstanceEntry {
            mesh,
            material,
            model: *model,
            normal: normal_matrix(model),
            tint: [1.0; 4],
            visible: true,
            pick: 1,
            joints: vec![],
            sig: 0,
            still: 0,
            world_bounds: None,
            cached: None,
            shadow: ShadowTrack::default(),
        };
        Ok(PropInstance(self.props.instances.insert(entry)?))
    }

    /// An instance's joint matrices (each a joint's world-from-bind transform in the model's
    /// space, its global matrix times its inverse bind matrix): its skinned mesh's vertices
    /// follow them from the next frame. Empty draws the mesh as modelled. There must be more
    /// matrices than the highest joint the mesh's vertices use.
    pub fn set_prop_instance_joints(
        &mut self,
        instance: PropInstance,
        joints: &[M4],
    ) -> Result<()> {
        if joints.iter().flatten().any(|x| !x.is_finite()) {
            return Err("set_prop_instance_joints: values must be finite".into());
        }
        let i = self
            .props
            .instances
            .get_mut(instance.0)
            .ok_or_else(|| format!("{instance:?} was destroyed"))?;
        if !joints.is_empty()
            && let Some(skin) = self.props.meshes.get(i.mesh.0).and_then(|m| m.skin)
            && joints.len() <= usize::from(skin.max_joint)
        {
            return Err(format!(
                "set_prop_instance_joints: {} matrices for joint {}",
                joints.len(),
                skin.max_joint
            ));
        }
        i.joints.clear();
        i.joints.extend_from_slice(joints);
        Ok(())
    }

    /// Moves an instance and shows or hides it.
    pub fn set_prop_instance(
        &mut self,
        instance: PropInstance,
        model: &M4,
        visible: bool,
    ) -> Result<()> {
        let i = self
            .props
            .instances
            .get_mut(instance.0)
            .ok_or_else(|| format!("{instance:?} was destroyed"))?;
        if i.model != *model {
            i.model = *model;
            i.normal = normal_matrix(model);
        }
        i.visible = visible;
        Ok(())
    }

    /// Gives an instance another material.
    pub fn set_prop_instance_material(
        &mut self,
        instance: PropInstance,
        material: PropMaterial,
    ) -> Result<()> {
        if self.props.materials.get(material.0).is_none() {
            return Err(format!("{material:?} was destroyed"));
        }
        let i = self
            .props
            .instances
            .get_mut(instance.0)
            .ok_or_else(|| format!("{instance:?} was destroyed"))?;
        i.material = material;
        Ok(())
    }

    /// Multiplies an instance's material colour and alpha (fades, greying, highlights) without
    /// a material of its own.
    pub fn set_prop_instance_tint(&mut self, instance: PropInstance, tint: [f32; 4]) -> Result<()> {
        if tint.iter().any(|x| !x.is_finite()) {
            return Err("set_prop_instance_tint: values must be finite".into());
        }
        self.props
            .instances
            .get_mut(instance.0)
            .ok_or_else(|| format!("{instance:?} was destroyed"))?
            .tint = tint;
        Ok(())
    }

    /// The pick layers an instance answers to (bits; 0: never picked). New instances are on
    /// layer 1.
    pub fn set_prop_instance_pick(&mut self, instance: PropInstance, layers: u32) -> Result<()> {
        self.props
            .instances
            .get_mut(instance.0)
            .ok_or_else(|| format!("{instance:?} was destroyed"))?
            .pick = layers;
        Ok(())
    }

    pub fn destroy_prop_instance(&mut self, instance: PropInstance) {
        self.props.remove_instance(instance.0);
    }

    /// Live prop instances.
    pub fn prop_instances(&self) -> usize {
        self.props.instances.iter().count()
    }

    /// The visible prop under pixel `(x, y)` of a `size` image of `view` (top-left origin) on
    /// one of the pick `layers`, tested against its triangles. Like the image: an overlay drawn
    /// on top of everything (no depth test) wins, the last drawn of them (highest order, then
    /// nearest); otherwise the nearest hit.
    pub fn pick(
        &self,
        view: &View,
        size: (u32, u32),
        (x, y): (f32, f32),
        layers: u32,
    ) -> Option<Pick> {
        let (origin, dir) = self.pick_ray(view, size, (x, y));
        self.pick_along_ray(origin, dir, layers)
    }

    /// The visible prop along a world-space ray `(origin, dir)` on one of the pick `layers`,
    /// tested against its triangles with the same overlay/order/nearest priority as
    /// [`Renderer::pick`]. Returns `None` if `origin` or `dir` contains non-finite values or
    /// `|dir|^2 < 1e-12`.
    pub fn pick_along_ray(&self, origin: [f32; 3], dir: [f32; 3], layers: u32) -> Option<Pick> {
        self.props.pick_along_ray(origin, dir, layers)
    }

    /// The ray from the eye through pixel (`x`, `y`) of a `size` image of `view`: its origin
    /// and unit direction.
    fn pick_ray(&self, view: &View, size: (u32, u32), (x, y): (f32, f32)) -> (V3, V3) {
        let eye = self.eye(view, [0.0, 0.0], size);
        let ndc = [
            2.0 * x / size.0.max(1) as f32 - 1.0,
            2.0 * y / size.1.max(1) as f32 - 1.0,
        ];
        let unproject = |z: f32| {
            let p = math::transform(&eye.inv, [ndc[0], ndc[1], z, 1.0]);
            [p[0] / p[3], p[1] / p[3], p[2] / p[3]]
        };
        // Reversed depth: 1 is the near plane.
        let near = unproject(1.0);
        let far = unproject(0.5);
        (view.eye, math::normalize(math::sub(far, near)))
    }

    /// The cooked world's instance under pixel (`x`, `y`) of a `size` image of `view`, tested
    /// against its triangles where they are drawn now: the nearest hit. Hidden instances are
    /// skipped. Props are picked with [`Renderer::pick`].
    pub fn pick_world(&self, view: &View, size: (u32, u32), at: (f32, f32)) -> Option<WorldPick> {
        let (origin, dir) = self.pick_ray(view, size, at);
        self.pick_world_along_ray(origin, dir)
    }

    /// The cooked world's instance along a world-space ray `(origin, dir)`, tested against its
    /// triangles where they are drawn now: the nearest hit. Hidden instances are skipped, and
    /// non-finite or near-zero (`|dir|^2 < 1e-12`) rays return `None`.
    pub fn pick_world_along_ray(&self, origin: [f32; 3], dir: [f32; 3]) -> Option<WorldPick> {
        let (origin, unit_dir) = normalize_pick_ray(origin, dir)?;
        self.scene
            .pick(origin, unit_dir)
            .map(|(instance, distance, point)| WorldPick {
                instance,
                distance,
                point,
            })
    }

    /// How many instances the cooked world has (its `World::instances`, in their order).
    pub fn world_instances(&self) -> usize {
        self.scene.edit.instances.len()
    }

    /// Where world instance `index` is drawn now (column-major model-to-world) and whether it
    /// is shown.
    pub fn world_instance(&self, index: usize) -> Option<([f32; 16], bool)> {
        self.scene
            .edit
            .instances
            .get(index)
            .map(|i| (i.transform, i.visible))
    }

    /// Moves world instance `index` to `matrix` and shows or hides it: how a tool or game moves
    /// an object of a cooked scene (its entity table names the instances). Only that instance's
    /// vertices are rewritten; it stays lit, shadowed and mapped like the rest of the world, and
    /// the ground and world maps follow from the next frame.
    pub fn set_world_instance(
        &mut self,
        device: &mut Device<'_>,
        index: usize,
        matrix: [f32; 16],
        visible: bool,
    ) -> Result<()> {
        self.scene.set_instance(device, index, matrix, visible)?;
        self.world_moved = true;
        Ok(())
    }

    /// Moves world lamp `index` (a cooked point light) to `position`.
    pub fn set_world_light(
        &mut self,
        device: &mut Device<'_>,
        index: usize,
        position: V3,
    ) -> Result<()> {
        self.scene.set_light(device, index, position)
    }

    /// Where a world point lands in a `size` image of `view`: pixel x and y (top-left origin)
    /// and the depth in metres from the eye, or `None` behind the eye.
    pub fn project(&self, view: &View, size: (u32, u32), point: V3) -> Option<[f32; 3]> {
        let eye = self.eye(view, [0.0, 0.0], size);
        let c = math::transform(&eye.view_proj, [point[0], point[1], point[2], 1.0]);
        if c[3] <= 1e-6 {
            return None;
        }
        Some([
            (c[0] / c[3] * 0.5 + 0.5) * size.0 as f32,
            (c[1] / c[3] * 0.5 + 0.5) * size.1 as f32,
            c[3],
        ])
    }

    /// World bounds of the visible props, as of the last frame recorded.
    pub(super) fn props_bounds(&self) -> Option<(V3, V3)> {
        self.props.bounds
    }

    /// Recomputes the prop detail blocks after the Look changed.
    pub(super) fn restyle_props(&mut self) {
        let look = &self.look;
        let ids: Vec<u32> = self.props.materials.iter().map(|(id, _)| id).collect();
        for id in ids {
            if let Some(m) = self.props.materials.get_mut(id) {
                m.detail = super::renderer::detail_block(look, &m.desc.name).0;
            }
        }
    }

    /// Forgets the per-slot bind groups (after the frame slots were rebuilt).
    pub(super) fn reset_prop_slots(&mut self, device: &mut Device<'_>) {
        for s in std::mem::take(&mut self.props.slots).into_iter().flatten() {
            for g in s.scene.into_iter().chain([s.shadow, s.overlay]) {
                device.destroy_bind_group(g);
            }
            device.destroy_buffer(s.buffer);
            device.destroy_buffer(s.joints);
        }
    }

    /// Orders this frame's visible instances and writes them to the slot's instance buffer
    /// (growing it, and rebuilding its bind groups, when needed): opaque ones batched by
    /// (material, mesh), then blended ones and then overlays one by one, each sorted by order
    /// and far to near from `eye`.
    pub(super) fn prepare_props(
        &mut self,
        device: &mut Device<'_>,
        slot: usize,
        eye: V3,
    ) -> Result<()> {
        struct Live<'a> {
            layer: Layer,
            settled: bool,
            order: f32,
            /// Negative squared distance: far first.
            far: f32,
            material: u32,
            mesh: u32,
            inst: &'a InstanceEntry,
        }
        self.props.pose_instances();
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        let mut live: Vec<Live> = Vec::new();
        for (_, inst) in self.props.instances.iter() {
            let (Some(mat), Some(m), true) = (
                self.props.materials.get(inst.material.0),
                self.props.meshes.get(inst.mesh.0),
                inst.world_bounds.is_some(),
            ) else {
                continue;
            };
            // The frame's prop bounds (which fit the shadow cascades) as modelled, not posed.
            let (mut blo, mut bhi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
            for p in corners((m.min, m.max)) {
                let w = math::transform_point(&inst.model, p);
                for k in 0..3 {
                    blo[k] = blo[k].min(w[k]);
                    bhi[k] = bhi[k].max(w[k]);
                }
            }
            let layer = mat.layer();
            if layer != Layer::Overlay {
                for k in 0..3 {
                    lo[k] = lo[k].min(blo[k]);
                    hi[k] = hi[k].max(bhi[k]);
                }
            }
            let centre = math::scale(math::add(blo, bhi), 0.5);
            let to = math::sub(centre, eye);
            live.push(Live {
                layer,
                settled: settles(inst, mat),
                order: if layer == Layer::Opaque {
                    0.0
                } else {
                    mat.desc.order
                },
                far: if layer == Layer::Opaque {
                    0.0
                } else {
                    -math::dot(to, to)
                },
                material: inst.material.0,
                mesh: inst.mesh.0,
                inst,
            });
        }
        live.sort_by(|a, b| {
            a.layer
                .cmp(&b.layer)
                .then(a.settled.cmp(&b.settled))
                .then(a.order.total_cmp(&b.order))
                .then(a.far.total_cmp(&b.far))
                .then((a.material, a.mesh).cmp(&(b.material, b.mesh)))
        });
        let mut batches: Vec<Batch> = Vec::new();
        let mut data: Vec<f32> = Vec::with_capacity(live.len() * 40);
        let mut joint_data: Vec<f32> = Vec::new();
        for (n, l) in live.iter().enumerate() {
            match batches.last_mut() {
                Some(b)
                    if l.layer == Layer::Opaque
                        && b.layer == Layer::Opaque
                        && b.settled == l.settled
                        && b.material == l.material
                        && b.mesh == l.mesh =>
                {
                    b.count += 1;
                }
                _ => batches.push(Batch {
                    mesh: l.mesh,
                    material: l.material,
                    first: n as u32,
                    count: 1,
                    layer: l.layer,
                    settled: l.settled,
                }),
            }
            data.extend_from_slice(&l.inst.model);
            data.extend_from_slice(&l.inst.normal);
            data.extend_from_slice(&l.inst.tint);
            let skin = self.props.meshes.get(l.mesh).and_then(|m| m.skin);
            match skin {
                Some(range) if l.inst.joints.len() > usize::from(range.max_joint) => {
                    let first = (joint_data.len() / 16) as f32;
                    for j in &l.inst.joints {
                        joint_data.extend_from_slice(j);
                    }
                    data.extend_from_slice(&[first, range.first as f32, 1.0, 0.0]);
                }
                _ => data.extend_from_slice(&[0.0; 4]),
            }
        }
        let count = live.len() as u32;
        drop(live);
        self.props.batches = batches;
        self.props.bounds = (lo[0] <= hi[0]).then_some((lo, hi));
        self.make_prop_shader_pipelines(device)?;
        // The shared skin records: rewritten when a skinned mesh came or went, and grown (with
        // every slot's bind groups) when they outgrow their buffer.
        if self.props.skin_dirty || self.props.skin_buffer.is_none() {
            let records = (self.props.skin_records.len() as u32).max(MIN_SKIN_RECORDS);
            if self
                .props
                .skin_buffer
                .is_none_or(|(_, held)| records > held)
            {
                self.reset_prop_slots(device);
                if let Some((old, _)) = self.props.skin_buffer.take() {
                    device.destroy_buffer(old);
                }
                let held = records.next_power_of_two();
                let buffer = device.create_buffer(&BufferDesc {
                    size: u64::from(held) * SKIN_RECORD_BYTES,
                    usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
                    memory: Memory::Device,
                })?;
                self.props.skin_buffer = Some((buffer, held));
            }
            if let Some((buffer, _)) = self.props.skin_buffer
                && !self.props.skin_records.is_empty()
            {
                let flat: Vec<f32> = self.props.skin_records.iter().flatten().copied().collect();
                device.write_buffer(buffer, 0, &f32_bytes(&flat))?;
            }
            self.props.skin_dirty = false;
        }
        let needed = count.max(MIN_CAPACITY);
        let joints_needed = ((joint_data.len() / 16) as u32).max(MIN_JOINTS);
        if needed > self.props.capacity || joints_needed > self.props.joint_capacity {
            self.reset_prop_slots(device);
            self.props.capacity = self.props.capacity.max(needed.next_power_of_two());
            self.props.joint_capacity = self
                .props
                .joint_capacity
                .max(joints_needed.next_power_of_two());
        }
        if self.props.slots.len() <= slot {
            self.props.slots.resize_with(slot + 1, || None);
        }
        if self.props.slots[slot].is_none() {
            self.props.slots[slot] = Some(self.create_prop_slot(device, slot)?);
        }
        if let Some(s) = &self.props.slots[slot] {
            if !data.is_empty() {
                device.write_buffer(s.buffer, 0, &f32_bytes(&data))?;
            }
            if !joint_data.is_empty() {
                device.write_buffer(s.joints, 0, &f32_bytes(&joint_data))?;
            }
        }
        Ok(())
    }

    /// Whether this frame has props to draw over the image.
    pub(super) fn has_overlays(&self) -> bool {
        self.props.batches.iter().any(|b| b.layer == Layer::Overlay)
    }

    /// Whether this frame has opaque shadow-casting prop batches.
    pub(super) fn has_prop_shadows(&self) -> bool {
        self.props.batches.iter().any(|b| {
            b.layer == Layer::Opaque
                && self
                    .props
                    .materials
                    .get(b.material)
                    .is_some_and(|m| m.desc.casts_shadow)
        })
    }

    /// Whether this frame has scene prop batches using water shading.
    pub(super) fn has_water_props(&self) -> bool {
        self.props.batches.iter().any(|b| {
            b.layer != Layer::Overlay
                && self
                    .props
                    .materials
                    .get(b.material)
                    .is_some_and(|m| m.desc.shading == Shading::Water)
        })
    }

    fn create_prop_slot(&self, device: &mut Device<'_>, slot: usize) -> Result<PropSlot> {
        let buffer = device.create_buffer(&BufferDesc {
            size: u64::from(self.props.capacity) * PROP_BYTES,
            usage: BufferUsage::STORAGE,
            memory: Memory::Upload,
        })?;
        let joints = device.create_buffer(&BufferDesc {
            size: u64::from(self.props.joint_capacity) * JOINT_BYTES,
            usage: BufferUsage::STORAGE,
            memory: Memory::Upload,
        })?;
        let skins = self
            .props
            .skin_buffer
            .map(|(b, _)| b)
            .ok_or("create_prop_slot: no skin buffer")?;
        let buf = |id| Resource::Buffer {
            id,
            offset: 0,
            size: 0,
        };
        let s = &self.slots[slot];
        let t = &self.targets;
        let p = &self.pipelines;
        let mut scene = Vec::new();
        for v in 0..2 {
            let reflection = if v == 0 { t.refl_hdr } else { self.dummy };
            scene.push(device.create_bind_group(
                p.scene_prop,
                0,
                &[
                    (0, buf(s.frame[v])),
                    (1, buf(self.scene.light_buffer)),
                    (2, buf(s.tiles[v])),
                    (3, buf(s.index[v])),
                    (4, Resource::Texture(self.shadow_map)),
                    (5, Resource::Sampler(self.shadow_sampler)),
                    (6, Resource::Texture(reflection)),
                    (7, Resource::Sampler(self.linear_clamp)),
                    (8, Resource::Texture(self.ground_depth)),
                    (9, Resource::Texture(self.ground_mask)),
                    (10, Resource::Texture(self.world_depth)),
                    (11, buf(buffer)),
                    (12, buf(joints)),
                    (13, buf(skins)),
                    (14, Resource::Texture(self.light_shadow_map)),
                ],
            )?);
        }
        let shadow = device.create_bind_group(
            p.shadow_prop,
            0,
            &[(11, buf(buffer)), (12, buf(joints)), (13, buf(skins))],
        )?;
        let overlay = device.create_bind_group(
            p.overlay[0][0],
            0,
            &[
                (0, buf(s.frame[0])),
                (11, buf(buffer)),
                (12, buf(joints)),
                (13, buf(skins)),
            ],
        )?;
        Ok(PropSlot {
            buffer,
            joints,
            scene: [scene[0], scene[1]],
            shadow,
            overlay,
        })
    }

    /// Draws this frame's prop batches into a shadow cascade (`light_vp`).
    pub(super) fn draw_prop_shadows(
        &self,
        enc: &mut crate::gpu::Encoder,
        slot: usize,
        light_vp: &M4,
    ) {
        let Some(Some(s)) = self.props.slots.get(slot) else {
            return;
        };
        let p = &self.pipelines;
        let mut bound = None;
        for b in &self.props.batches {
            let (Some(mesh), Some(mat)) = (
                self.props.meshes.get(b.mesh),
                self.props.materials.get(b.material),
            ) else {
                continue;
            };
            if b.layer != Layer::Opaque || !mat.desc.casts_shadow {
                continue;
            }
            let masked = mat.shadow_group.is_some();
            if bound != Some(masked) {
                enc.set_pipeline(if masked {
                    p.shadow_prop_masked
                } else {
                    p.shadow_prop
                });
                enc.set_bind_group(0, s.shadow);
                bound = Some(masked);
            }
            if let Some(g) = mat.shadow_group {
                enc.set_bind_group(1, g);
            }
            let mut push = [0.0f32; 24];
            push[..16].copy_from_slice(light_vp);
            push[16..20].copy_from_slice(&[
                mat.data.params[0],
                mat.data.base_color[3],
                mat.data.flags[0],
                0.0,
            ]);
            push[20] = b.first as f32;
            enc.push_constants(&f32_bytes(&push));
            enc.set_vertex_buffer(mesh.vertices, 0);
            enc.set_index_buffer(mesh.indices, 0);
            enc.draw_indexed(mesh.index_count, b.count, 0, 0, 0);
        }
    }

    /// The `Draw` push constants of a prop batch.
    fn prop_push(mat: &MaterialEntry, first: u32, encode_srgb: bool) -> [f32; 32] {
        let mut push = [0.0f32; 32];
        push[..16].copy_from_slice(&mat.detail);
        let mut base = mat.data.base_color;
        if mat.detail[0] == 0.0 && mat.detail[8] > 0.5 {
            base[..3].copy_from_slice(&mat.detail[5..8]);
        }
        push[12] = first as f32;
        push[13] = if encode_srgb { 1.0 } else { 0.0 };
        push[14] = if mat.layer() == Layer::Opaque {
            0.0
        } else {
            1.0
        };
        push[16..20].copy_from_slice(&base);
        push[20..24].copy_from_slice(&mat.data.emissive);
        push[24..28].copy_from_slice(&mat.data.params);
        push[28..32].copy_from_slice(&mat.data.flags);
        if let Some(s) = mat.desc.surface {
            push[10] = (s as u32 + 1) as f32;
        }
        if let Some(w) = mat
            .desc
            .clearcoat
            .or(mat.desc.sheen)
            .or(mat.desc.subsurface)
        {
            push[11] = w;
        }
        if let Some(p) = mat.desc.clearcoat_roughness.or(mat.desc.sheen_tint) {
            push[15] = p;
        }
        if mat.desc.shader.is_some() {
            // A prop shader's parameters: detail0..2, emissive and flags (prop_shader.wgsl).
            push[..12].copy_from_slice(&mat.desc.params[..12]);
            push[20..24].copy_from_slice(&mat.desc.params[12..16]);
            push[28..32].copy_from_slice(&mat.desc.params[16..]);
        }
        push
    }

    /// Draws this frame's opaque prop batches in `set` with the scene shading (`view` 0 main,
    /// 1 reflection).
    pub(super) fn draw_props(
        &self,
        enc: &mut crate::gpu::Encoder,
        slot: usize,
        view: usize,
        set: PropSet,
    ) {
        let Some(Some(s)) = self.props.slots.get(slot) else {
            return;
        };
        let opaque = |b: &&Batch| b.layer == Layer::Opaque && set.takes(b);
        if !self.props.batches.iter().any(|b| opaque(&b)) {
            return;
        }
        // In the main view, materials without an alpha test take the pipeline that never
        // discards (so hidden fragments are dropped before shading); the reflection keeps the
        // clip under the water, so it draws everything with the cutting one.
        // The settled props draw only into the cached picture, with its shadow target.
        let (opaque_pipeline, cut_pipeline) = if set == PropSet::Settled {
            (self.pipelines.cache_prop[1], self.pipelines.cache_prop[0])
        } else {
            (self.pipelines.scene_prop_opaque, self.pipelines.scene_prop)
        };
        let groups: Vec<(crate::gpu::PipelineId, Option<bool>)> = if view == 0 {
            vec![(opaque_pipeline, Some(true)), (cut_pipeline, Some(false))]
        } else {
            vec![(self.pipelines.scene_prop, None)]
        };
        for (pipeline, no_cut) in groups {
            enc.set_pipeline(pipeline);
            enc.set_bind_group(0, s.scene[view]);
            let mut bound = None;
            for b in self.props.batches.iter().filter(opaque) {
                let (Some(mesh), Some(mat)) = (
                    self.props.meshes.get(b.mesh),
                    self.props.materials.get(b.material),
                ) else {
                    continue;
                };
                if no_cut.is_some_and(|o| o != (mat.data.params[0] < 0.0)) {
                    continue;
                }
                if view == 1 && mat.data.params[3] == 2.0 {
                    continue;
                }
                if bound != Some(b.material) {
                    enc.set_bind_group(1, mat.group);
                    bound = Some(b.material);
                }
                enc.push_constants(&f32_bytes(&Self::prop_push(mat, b.first, false)));
                enc.set_vertex_buffer(mesh.vertices, 0);
                enc.set_index_buffer(mesh.indices, 0);
                enc.draw_indexed(mesh.index_count, b.count, 0, 0, 0);
            }
        }
    }

    /// Draws this frame's blended scene props, sorted, after the opaque scene (main view).
    pub(super) fn draw_blended_props(&self, enc: &mut crate::gpu::Encoder, slot: usize) {
        let Some(Some(s)) = self.props.slots.get(slot) else {
            return;
        };
        let mut pipeline = None;
        for b in self
            .props
            .batches
            .iter()
            .filter(|b| b.layer == Layer::Blended)
        {
            let (Some(mesh), Some(mat)) = (
                self.props.meshes.get(b.mesh),
                self.props.materials.get(b.material),
            ) else {
                continue;
            };
            let p = self.pipelines.scene_prop_blend[usize::from(mat.desc.blend == BlendMode::Add)];
            if pipeline != Some(p) {
                enc.set_pipeline(p);
                enc.set_bind_group(0, s.scene[0]);
                pipeline = Some(p);
            }
            enc.set_bind_group(1, mat.group);
            enc.push_constants(&f32_bytes(&Self::prop_push(mat, b.first, false)));
            enc.set_vertex_buffer(mesh.vertices, 0);
            enc.set_index_buffer(mesh.indices, 0);
            enc.draw_indexed(mesh.index_count, b.count, 0, 0, 0);
        }
    }

    /// Draws this frame's overlays into the finished image (a pass over the output target and
    /// the scene depth); `encode_srgb` when the target does not encode sRGB itself.
    pub(super) fn draw_overlays(
        &self,
        enc: &mut crate::gpu::Encoder,
        slot: usize,
        encode_srgb: bool,
    ) {
        let Some(Some(s)) = self.props.slots.get(slot) else {
            return;
        };
        let mut pipeline = None;
        for b in self
            .props
            .batches
            .iter()
            .filter(|b| b.layer == Layer::Overlay)
        {
            let (Some(mesh), Some(mat)) = (
                self.props.meshes.get(b.mesh),
                self.props.materials.get(b.material),
            ) else {
                continue;
            };
            let (blend, depth) = (blend_index(mat.desc.blend), depth_index(&mat.desc));
            let p = match mat.desc.shader {
                // Made in `prepare_props`; a pipeline that could not be made draws nothing.
                Some(h) => match self
                    .props
                    .shaders
                    .get(h.0)
                    .and_then(|s| s.pipelines[blend][depth])
                {
                    Some(p) => p,
                    None => continue,
                },
                None => self.pipelines.overlay[blend][depth],
            };
            if pipeline != Some(p) {
                enc.set_pipeline(p);
                enc.set_bind_group(0, s.overlay);
                pipeline = Some(p);
            }
            enc.set_bind_group(1, mat.group);
            enc.push_constants(&f32_bytes(&Self::prop_push(mat, b.first, encode_srgb)));
            enc.set_vertex_buffer(mesh.vertices, 0);
            enc.set_index_buffer(mesh.indices, 0);
            enc.draw_indexed(mesh.index_count, b.count, 0, 0, 0);
        }
    }

    /// Releases every prop object, models included (when the renderer is dropped with its
    /// device still alive).
    pub fn destroy_props(&mut self, device: &mut Device<'_>) {
        self.drop_models();
        self.reset_prop_slots(device);
        if let Some((buffer, _)) = self.props.skin_buffer.take() {
            device.destroy_buffer(buffer);
        }
        self.props.skin_records.clear();
        for m in self.props.meshes.drain() {
            device.destroy_buffer(m.vertices);
            device.destroy_buffer(m.indices);
        }
        for m in self.props.materials.drain() {
            device.destroy_bind_group(m.group);
            if let Some(g) = m.shadow_group {
                device.destroy_bind_group(g);
            }
        }
        for t in self.props.textures.drain() {
            device.destroy_texture(t);
        }
        self.props.instances.drain();
        self.props.batches.clear();
        self.props.vacated.clear();
        self.props.vacated_boxes.clear();
        self.props.texture_edits.clear();
        self.forget_world_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_from_json() {
        let d = MaterialDesc::from_json(
            br#"{"name": "stone", "color": [0.1, 0.3, 0.2, 1], "roughness": 2,
                 "shading": "unlit", "alphaCutoff": 0.5, "baseTexture": 7,
                 "castsShadow": false, "lamp": true, "emissive": [1, 0.5, 0],
                 "surface": "clearcoat", "clearcoat": 0.8, "clearcoatRoughness": 0.1}"#,
        )
        .unwrap();
        assert_eq!(d.name, "stone");
        assert_eq!(d.surface, Some(SurfaceModel::ClearCoat));
        assert_eq!((d.clearcoat, d.clearcoat_roughness), (Some(0.8), Some(0.1)));
        assert_eq!(d.base_color, [0.1, 0.3, 0.2, 1.0]);
        assert_eq!((d.roughness, d.shading), (1.0, Shading::Unlit));
        assert_eq!(
            (d.alpha_cutoff, d.base_texture),
            (Some(0.5), Some(PropTexture(7)))
        );
        assert!(!d.casts_shadow && d.lamp && d.emissive == [1.0, 0.5, 0.0]);
        assert_eq!(
            MaterialDesc::from_json(b"{}").unwrap(),
            MaterialDesc::default()
        );
        assert_eq!(
            MaterialDesc::from_json(br#"{"order": 44.5}"#)
                .unwrap()
                .order,
            44.5
        );
        let solid = MaterialDesc::from_json(br#"{"overlay": true, "depthWrite": true}"#).unwrap();
        assert!(solid.overlay && solid.depth_test && solid.depth_write);
        assert_eq!(depth_index(&solid), 2);
        for bad in [
            &b"[]"[..],
            br#"{"colour": [1, 1, 1, 1]}"#,
            br#"{"color": [1, 1, 1]}"#,
            br#"{"shading": "toon"}"#,
            br#"{"baseTexture": 0}"#,
            br#"{"castsShadow": 1}"#,
        ] {
            assert!(
                MaterialDesc::from_json(bad).is_err(),
                "{}",
                String::from_utf8_lossy(bad)
            );
        }
    }

    fn dummy_material_entry(desc: MaterialDesc) -> MaterialEntry {
        MaterialEntry {
            desc,
            data: MaterialData {
                base_color: [1.0; 4],
                emissive: [0.0; 4],
                params: [0.0; 4],
                flags: [0.0; 4],
                base: None,
                normal: None,
                emission: None,
            },
            group: BindGroupId::from_raw(1),
            shadow_group: None,
            detail: [0.0; 16],
            revision: 0,
        }
    }

    fn z_quad_mesh_entry(half: f32) -> MeshEntry {
        let positions = vec![
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ];
        let (min, max) = bounds(&positions);
        MeshEntry {
            vertices: BufferId::from_raw(1),
            indices: BufferId::from_raw(2),
            index_count: 6,
            positions,
            uvs: vec![[0.0, 0.0]; 4],
            triangles: vec![0, 1, 2, 0, 2, 3],
            min,
            max,
            skin: None,
            revision: 0,
            joint_boxes: vec![],
            shape: 0,
            drift: 0.0,
        }
    }

    fn translation_matrix(x: f32, y: f32, z: f32) -> M4 {
        let mut m = math::IDENTITY;
        m[12] = x;
        m[13] = y;
        m[14] = z;
        m
    }

    #[test]
    fn posed_corners_follow_the_joints_of_a_skinned_prop() {
        // Two vertices, one on each joint; joint 1 lifts its vertex 3 m.
        let mut mesh = z_quad_mesh_entry(0.5);
        mesh.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        mesh.skin = Some(SkinRange {
            first: 0,
            count: 4,
            max_joint: 1,
        });
        let records = [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ];
        mesh.joint_boxes = joint_boxes(&mesh.positions, &records);
        let model = translation_matrix(10.0, 0.0, 0.0);
        let inst = InstanceEntry {
            mesh: PropMesh::from_raw(1),
            material: PropMaterial::from_raw(1),
            model,
            normal: normal_matrix(&model),
            tint: [1.0; 4],
            visible: true,
            pick: 0,
            joints: vec![math::IDENTITY, translation_matrix(0.0, 3.0, 0.0)],
            sig: 0,
            still: 0,
            world_bounds: None,
            cached: None,
            shadow: ShadowTrack::default(),
        };
        let mut posed = vec![];
        assert_eq!(posed_corners(&inst, &mesh, &mut posed), 1.0);
        assert_eq!(bounds(&posed), ([10.0, 0.0, 0.0], [11.0, 3.0, 0.0]));
    }

    fn instance_entry(mesh: PropMesh, material: PropMaterial, joints: Vec<M4>) -> InstanceEntry {
        InstanceEntry {
            mesh,
            material,
            model: math::IDENTITY,
            normal: math::IDENTITY,
            tint: [1.0; 4],
            visible: true,
            pick: 0,
            joints,
            sig: 0,
            still: 0,
            world_bounds: None,
            cached: None,
            shadow: ShadowTrack::default(),
        }
    }

    #[test]
    fn shadow_maps_are_drawn_again_once_a_caster_drifted_far_enough() {
        // Straight down on the origin, 4 m across in 1024 texels: 256 texels a metre.
        let down = math::look_at([0.0, 5.0, 0.0], [0.0; 3], [0.0, 0.0, -1.0]);
        let map: ShadowMap = (
            math::mul(&math::ortho(-2.0, 2.0, -2.0, 2.0, 0.1, 10.0), &down),
            1024,
        );
        let texel = 1.0 / 256.0;
        let unit = ([-0.5; 3], [0.5; 3]);
        assert!((texels_per_metre(&map, unit).unwrap() - 256.0).abs() < 1e-2);
        assert_eq!(
            texels_per_metre(&map, ([3.0, 0.0, 0.0], [4.0, 1.0, 1.0])),
            None
        );
        // A spot: ahead of it, the nearer the finer; behind it, outside; reaching it, unbounded.
        let spot = (
            math::mul(
                &math::perspective_depth(1.0, 1.0, 0.1, 20.0),
                &math::look_at([0.0, 5.0, 0.0], [0.0; 3], [0.0, 0.0, -1.0]),
            ),
            1024,
        );
        let near = texels_per_metre(&spot, ([-0.1, 3.0, -0.1], [0.1, 3.1, 0.1])).unwrap();
        let far = texels_per_metre(&spot, ([-0.1, 0.0, -0.1], [0.1, 0.1, 0.1])).unwrap();
        assert!(near > far && far > 0.0);
        assert_eq!(
            texels_per_metre(&spot, ([-0.1, 6.0, -0.1], [0.1, 7.0, 0.1])),
            None
        );
        assert_eq!(
            texels_per_metre(&spot, ([-0.1, 4.0, -0.1], [0.1, 6.0, 0.1])),
            Some(f32::INFINITY)
        );

        // A skinned quad (its lower edge on joint 0, its upper on joint 1) casting, and an opaque
        // prop that casts none.
        let mut props = Props::default();
        let mut quad = z_quad_mesh_entry(0.5);
        let on = |j: f32| [[j, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
        let records = [on(0.0), on(0.0), on(1.0), on(1.0)].concat();
        quad.joint_boxes = joint_boxes(&quad.positions, &records);
        quad.skin = Some(SkinRange {
            first: 0,
            count: 8,
            max_joint: 1,
        });
        let quad = PropMesh(props.meshes.insert(quad).unwrap());
        let plain = PropMesh(props.meshes.insert(z_quad_mesh_entry(0.5)).unwrap());
        let caster = PropMaterial(
            props
                .materials
                .insert(dummy_material_entry(MaterialDesc::default()))
                .unwrap(),
        );
        let shadeless = PropMaterial(
            props
                .materials
                .insert(dummy_material_entry(MaterialDesc {
                    casts_shadow: false,
                    ..MaterialDesc::default()
                }))
                .unwrap(),
        );
        let rest = vec![math::IDENTITY; 2];
        let figure = props
            .instances
            .insert(instance_entry(quad, caster, rest.clone()))
            .unwrap();
        let pillar = props
            .instances
            .insert(instance_entry(plain, shadeless, vec![]))
            .unwrap();
        let eye = math::mul(
            &math::perspective(1.0, 1.0, 0.1),
            &math::look_at([0.0, 1.0, 5.0], [0.0; 3], [0.0, 1.0, 0.0]),
        );
        let view = CacheView {
            view_proj: &eye,
            size: (64, 64),
        };
        // Measured against half a texel, so a drift can be told from a move.
        let frame = |props: &mut Props| {
            props.pose_instances();
            track_cached_props(props, false, &view, &[map], 0.5)
        };
        let lift = |props: &mut Props, texels: f32| {
            let joints = &mut props.instances.get_mut(figure).unwrap().joints;
            joints[1] = translation_matrix(0.0, texels * texel, 0.0);
        };

        // A caster appearing counts; the maps then hold it.
        assert!(frame(&mut props).casters_moved);
        commit_prop_shadows(&mut props);
        let still = frame(&mut props);
        assert!(!still.casters_moved && still.moved.is_empty());

        // Its upper edge rises 0.3 texel a frame: measured from where the maps hold it, the
        // second frame's rise counts.
        lift(&mut props, 0.3);
        let small = frame(&mut props);
        assert!(!small.casters_moved && small.moved.len() == 1);
        props.pose_instances();
        let exact = track_cached_props(&mut props, false, &view, &[map], SHADOW_DRIFT_TEXELS);
        assert!(
            exact.casters_moved,
            "any drift counts as the renderer measures it"
        );
        lift(&mut props, 0.6);
        assert!(frame(&mut props).casters_moved);
        commit_prop_shadows(&mut props);
        lift(&mut props, 0.9);
        assert!(!frame(&mut props).casters_moved);
        commit_prop_shadows(&mut props);

        // A prop that casts no shadow moves freely.
        props.instances.get_mut(pillar).unwrap().model = translation_matrix(1.0, 0.0, 0.0);
        let shadeless_move = frame(&mut props);
        assert!(!shadeless_move.casters_moved && shadeless_move.moved.is_empty());

        // Its mesh morphed 0.6 texel since the maps were drawn counts, and only until then.
        props.meshes.get_mut(quad.0).unwrap().drift = 0.6 * texel;
        assert!(frame(&mut props).casters_moved);
        commit_prop_shadows(&mut props);
        assert!(frame(&mut props).moved.is_empty());

        // Hidden, shown and destroyed, it counts each time; while the whole picture is drawn
        // again too.
        props.instances.get_mut(figure).unwrap().visible = false;
        assert!(frame(&mut props).casters_moved);
        commit_prop_shadows(&mut props);
        assert!(!frame(&mut props).casters_moved);
        props.instances.get_mut(figure).unwrap().visible = true;
        props.pose_instances();
        assert!(track_cached_props(&mut props, true, &view, &[map], 0.5).casters_moved);
        commit_prop_shadows(&mut props);
        props.remove_instance(figure);
        let gone = frame(&mut props);
        assert!(gone.casters_moved && gone.moved.len() == 1);
    }

    #[test]
    fn pick_along_ray_normalizes_filters_layers_and_rejects_invalid_rays() {
        let mut props = Props::default();
        let mesh = PropMesh(props.meshes.insert(z_quad_mesh_entry(1.0)).unwrap());
        let opaque_mat = PropMaterial(
            props
                .materials
                .insert(dummy_material_entry(MaterialDesc {
                    name: "panel".into(),
                    ..MaterialDesc::default()
                }))
                .unwrap(),
        );
        let overlay_mat = PropMaterial(
            props
                .materials
                .insert(dummy_material_entry(MaterialDesc {
                    name: "reticle".into(),
                    overlay: true,
                    depth_test: false,
                    order: 2.0,
                    ..MaterialDesc::default()
                }))
                .unwrap(),
        );

        let near_model = translation_matrix(0.0, 0.0, -2.5);
        let near_inst = PropInstance(
            props
                .instances
                .insert(InstanceEntry {
                    mesh,
                    material: opaque_mat,
                    model: near_model,
                    normal: normal_matrix(&near_model),
                    tint: [1.0; 4],
                    visible: true,
                    pick: 0b0001,
                    joints: Vec::new(),
                    sig: 0,
                    still: 0,
                    world_bounds: None,
                    cached: None,
                    shadow: ShadowTrack::default(),
                })
                .unwrap(),
        );

        let far_model = translation_matrix(0.0, 0.0, -6.0);
        let far_inst = PropInstance(
            props
                .instances
                .insert(InstanceEntry {
                    mesh,
                    material: opaque_mat,
                    model: far_model,
                    normal: normal_matrix(&far_model),
                    tint: [1.0; 4],
                    visible: true,
                    pick: 0b0010,
                    joints: Vec::new(),
                    sig: 0,
                    still: 0,
                    world_bounds: None,
                    cached: None,
                    shadow: ShadowTrack::default(),
                })
                .unwrap(),
        );

        // Unnormalized ray direction `[0, 0, -5]` must normalize cleanly and report world-metre distance `5.5`.
        let hit = props
            .pick_along_ray([0.25, -0.5, 3.0], [0.0, 0.0, -5.0], u32::MAX)
            .expect("hit near prop");
        assert_eq!(hit.instance, near_inst);
        assert!(hit.distance.is_finite() && (hit.distance - 5.5).abs() < 1e-5);
        assert!(hit.point.iter().all(|v| v.is_finite()));
        assert!((hit.point[0] - 0.25).abs() < 1e-5);
        assert!((hit.point[1] + 0.5).abs() < 1e-5);
        assert!((hit.point[2] + 2.5).abs() < 1e-5);

        // Layer mask filtering selects only matching layer bits and returns `None` on disjoint/zero masks.
        let layer2_hit = props
            .pick_along_ray([0.0, 0.0, 3.0], [0.0, 0.0, -1.0], 0b0010)
            .expect("hit layer 2 prop");
        assert_eq!(layer2_hit.instance, far_inst);
        assert!((layer2_hit.distance - 9.0).abs() < 1e-5);
        assert!(
            props
                .pick_along_ray([0.0, 0.0, 3.0], [0.0, 0.0, -1.0], 0b0100)
                .is_none()
        );
        assert!(
            props
                .pick_along_ray([0.0, 0.0, 3.0], [0.0, 0.0, -1.0], 0)
                .is_none()
        );

        // Overlay drawn on top (`overlay && !depth_test`) wins even when placed behind an opaque prop.
        let overlay_model = translation_matrix(0.0, 0.0, -8.0);
        let overlay_inst = PropInstance(
            props
                .instances
                .insert(InstanceEntry {
                    mesh,
                    material: overlay_mat,
                    model: overlay_model,
                    normal: normal_matrix(&overlay_model),
                    tint: [1.0; 4],
                    visible: true,
                    pick: 0b0100,
                    joints: Vec::new(),
                    sig: 0,
                    still: 0,
                    world_bounds: None,
                    cached: None,
                    shadow: ShadowTrack::default(),
                })
                .unwrap(),
        );
        let top_hit = props
            .pick_along_ray([0.0, 0.0, 3.0], [0.0, 0.0, -1.0], u32::MAX)
            .expect("overlay wins");
        assert_eq!(top_hit.instance, overlay_inst);
        assert!((top_hit.distance - 11.0).abs() < 1e-5);

        // NaN, Inf, and zero / near-zero (`len_sq < 1e-12`) rays must all return `None`.
        for (bad_origin, bad_dir) in [
            ([f32::NAN, 0.0, 3.0], [0.0, 0.0, -1.0]),
            ([0.0, f32::INFINITY, 3.0], [0.0, 0.0, -1.0]),
            ([0.0, 0.0, f32::NEG_INFINITY], [0.0, 0.0, -1.0]),
            ([0.0, 0.0, 3.0], [f32::NAN, 0.0, -1.0]),
            ([0.0, 0.0, 3.0], [0.0, f32::INFINITY, -1.0]),
            ([0.0, 0.0, 3.0], [0.0, 0.0, 0.0]),
            ([0.0, 0.0, 3.0], [1e-7, -1e-7, 1e-7]),
        ] {
            assert!(
                props
                    .pick_along_ray(bad_origin, bad_dir, u32::MAX)
                    .is_none(),
                "expected None for origin={bad_origin:?}, dir={bad_dir:?}"
            );
        }
    }

    #[test]
    fn renderer_pick_ray_and_world_ray_delegate_cleanly() {
        use crate::render::{Look, Output, Quality};
        use crate::world::{AlphaMode, Instance, Material, Mesh, Transform, World};
        use ganbate_core::Vec3;

        let Ok(mut device) = Device::headless() else {
            return;
        };

        let mut world = World::default();
        world.materials.push(Material {
            name: "wall".into(),
            base_color: [0.6, 0.6, 0.65, 1.0],
            roughness: 0.7,
            metallic: 0.0,
            alpha: AlphaMode::Opaque,
            shading: Shading::Standard,
            ..Material::default()
        });
        world.meshes.push(Mesh {
            positions: vec![
                Vec3(-2.0, -2.0, 0.0),
                Vec3(2.0, -2.0, 0.0),
                Vec3(2.0, 2.0, 0.0),
                Vec3(-2.0, 2.0, 0.0),
            ],
            normals: vec![Vec3(0.0, 0.0, 1.0); 4],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            color: [1.0, 1.0, 1.0],
            indices: vec![0, 1, 2, 0, 2, 3],
            material: Some(0),
        });
        world.instances.push(Instance {
            mesh: 0,
            transform: Transform(translation_matrix(0.0, 0.0, -4.0)),
        });

        let mut renderer = Renderer::new(
            &mut device,
            &world,
            Look::default(),
            Quality::low_cost(),
            (100, 100),
            Output::Still,
        )
        .expect("headless renderer");

        let prop_mesh = renderer
            .create_prop_mesh(
                &mut device,
                &MeshData {
                    positions: &[
                        [-0.5, -0.5, 0.0],
                        [0.5, -0.5, 0.0],
                        [0.5, 0.5, 0.0],
                        [-0.5, 0.5, 0.0],
                    ],
                    normals: &[[0.0, 0.0, 1.0]; 4],
                    uvs: &[[0.0, 0.0]; 4],
                    indices: &[0, 1, 2, 0, 2, 3],
                },
            )
            .expect("prop mesh");
        let prop_mat = renderer
            .create_prop_material(&mut device, &MaterialDesc::default())
            .expect("prop material");
        let prop_inst = renderer
            .create_prop_instance(prop_mesh, prop_mat, &translation_matrix(0.0, 0.0, -1.5))
            .expect("prop instance");
        renderer
            .set_prop_instance_pick(prop_inst, 0b0010)
            .expect("set pick mask");

        // World ray picking with unnormalized ray hits cooked instance 0 at z = -4 (distance 7.0 from z = 3).
        let world_hit = renderer
            .pick_world_along_ray([0.5, -0.25, 3.0], [0.0, 0.0, -3.0])
            .expect("world ray hit");
        assert_eq!(world_hit.instance, 0);
        assert!(world_hit.distance.is_finite() && (world_hit.distance - 7.0).abs() < 1e-5);
        assert!((world_hit.point[0] - 0.5).abs() < 1e-5);
        assert!((world_hit.point[1] + 0.25).abs() < 1e-5);
        assert!((world_hit.point[2] + 4.0).abs() < 1e-5);

        // Prop ray picking respects layer mask and matches centre-screen camera `pick`.
        assert!(
            renderer
                .pick_along_ray([0.0, 0.0, 3.0], [0.0, 0.0, -2.0], 0b0001)
                .is_none()
        );
        let ray_prop = renderer
            .pick_along_ray([0.0, 0.0, 3.0], [0.0, 0.0, -2.0], 0b0010)
            .expect("prop ray hit");
        assert_eq!(ray_prop.instance, prop_inst);
        assert!((ray_prop.distance - 4.5).abs() < 1e-5);

        let view = View {
            name: "centre".into(),
            eye: [0.0, 0.0, 3.0],
            target: [0.0, 0.0, 0.0],
            fov: 50.0,
            ..View::default()
        };
        let screen_prop = renderer
            .pick(&view, (100, 100), (50.0, 50.0), 0b0010)
            .expect("screen prop hit");
        assert_eq!(screen_prop.instance, ray_prop.instance);
        assert!((screen_prop.distance - ray_prop.distance).abs() < 1e-4);

        let screen_world = renderer
            .pick_world(&view, (100, 100), (50.0, 50.0))
            .expect("screen world hit");
        assert_eq!(screen_world.instance, 0);
        assert!((screen_world.distance - 7.0).abs() < 1e-4);

        // Degenerate / non-finite rays return None on both Renderer ray methods.
        for bad_dir in [
            [0.0, 0.0, 0.0],
            [f32::NAN, 0.0, -1.0],
            [0.0, f32::INFINITY, -1.0],
        ] {
            assert!(
                renderer
                    .pick_along_ray([0.0, 0.0, 3.0], bad_dir, u32::MAX)
                    .is_none()
            );
            assert!(
                renderer
                    .pick_world_along_ray([0.0, 0.0, 3.0], bad_dir)
                    .is_none()
            );
        }
        assert!(
            renderer
                .pick_world_along_ray([f32::NEG_INFINITY, 0.0, 3.0], [0.0, 0.0, -1.0])
                .is_none()
        );

        renderer.destroy_props(&mut device);
    }

    #[test]
    fn canvas_clear_rounds_to_canvas_bytes_and_decodes_srgb() {
        assert_eq!(
            canvas_clear([0.2, 0.4, 0.8, 0.6], false),
            Ok([51.0 / 255.0, 102.0 / 255.0, 204.0 / 255.0, 153.0 / 255.0])
        );
        let srgb = canvas_clear([0.2, 0.4, 0.8, 0.6], true).unwrap();
        assert_eq!(srgb[..3], [51, 102, 204].map(math::srgb_to_linear));
        assert_eq!(srgb[3], 153.0 / 255.0, "alpha is never sRGB-encoded");
        assert_eq!(
            canvas_clear([-1.0, 2.0, 0.0, 1.0], false),
            Ok([0.0, 1.0, 0.0, 1.0])
        );
        assert!(canvas_clear([0.5, f32::NAN, 0.5, 1.0], true).is_err());
        assert!(canvas_clear([0.5, 0.5, 0.5, f32::INFINITY], false).is_err());
    }

    #[test]
    fn cleared_prop_textures_hold_the_canvas_bytes_in_every_mip_and_take_updates() {
        use crate::render::{Look, Output, Quality};
        use crate::world::World;

        let Ok(mut device) = Device::headless() else {
            return;
        };
        let mut r = Renderer::new(
            &mut device,
            &World::default(),
            Look::default(),
            Quality::low_cost(),
            (16, 16),
            Output::Still,
        )
        .expect("headless renderer");
        let read = |device: &mut Device<'_>, r: &Renderer, t: PropTexture, mip: u32| {
            let id = r.prop_texture_id(Some(t), r.dummy).expect("texture");
            let (w, h) = device.texture_desc(id).expect("desc").mip_size(mip);
            let rb = device
                .create_buffer(&BufferDesc {
                    size: u64::from(w * h * 4),
                    usage: BufferUsage::COPY_DST,
                    memory: Memory::Readback,
                })
                .expect("readback");
            let mut e = device.encoder();
            e.copy_texture_to_buffer(id, mip, 0, rb, 0);
            device.submit(e).expect("copy");
            let bytes = device.read_buffer(rb).expect("read");
            device.destroy_buffer(rb);
            bytes
        };
        let solid = |px: [u8; 4], n: usize| px.repeat(n);

        // 5x3 has mips 5x3, 2x1 and 1x1; every one holds the bytes a canvas of the colour would.
        let srgb = r
            .create_prop_texture_clear(&mut device, (5, 3), true, [0.2, 0.4, 0.8, 1.0])
            .expect("srgb clear");
        let unorm = r
            .create_prop_texture_clear(&mut device, (5, 3), false, [0.2, 0.4, 0.8, 0.6])
            .expect("unorm clear");
        for (mip, n) in [(0, 15), (1, 2), (2, 1)] {
            assert_eq!(
                read(&mut device, &r, srgb, mip),
                solid([51, 102, 204, 255], n)
            );
            assert_eq!(
                read(&mut device, &r, unorm, mip),
                solid([51, 102, 204, 153], n)
            );
        }

        // A placeholder takes the picture later at its size; a wrong size or a failed fill
        // leaves it and its edit count as they were.
        let edits = |r: &Renderer| r.props.texture_edits.get(&srgb.0).copied().unwrap_or(0);
        assert!(r.update_prop_texture(&mut device, srgb, &[9; 4]).is_err());
        let refused = r.update_prop_texture_with(&mut device, srgb, |_, _| Err("refused".into()));
        assert_eq!(refused, Err("refused".into()));
        assert_eq!(edits(&r), 0);
        assert_eq!(
            read(&mut device, &r, srgb, 0),
            solid([51, 102, 204, 255], 15)
        );
        r.update_prop_texture(&mut device, srgb, &solid([200, 10, 30, 255], 15))
            .expect("update");
        assert_eq!(edits(&r), 1);
        assert_eq!(
            read(&mut device, &r, srgb, 0),
            solid([200, 10, 30, 255], 15)
        );
        assert_eq!(read(&mut device, &r, srgb, 2), solid([200, 10, 30, 255], 1));

        // A refused fill or colour makes no prop texture.
        let count = |r: &Renderer| r.props.textures.iter().count();
        let before = count(&r);
        let made =
            r.create_prop_texture_with(&mut device, (4, 4), true, |_, _| Err("no source".into()));
        assert_eq!(made, Err("no source".into()));
        assert!(
            r.create_prop_texture_clear(&mut device, (4, 4), true, [f32::NAN, 0.0, 0.0, 1.0])
                .is_err()
        );
        assert_eq!(count(&r), before);

        r.destroy_props(&mut device);
    }
}
