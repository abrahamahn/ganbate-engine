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
//! one buffer shared by every skinned mesh; each instance of it follows joint matrices
//! ([`Renderer::set_prop_instance_joints`]; a model instance's skinned parts share their skin's),
//! which go to the frame slot's joint buffer once a frame, and the vertex stages move its
//! vertices on the GPU (`skin_vertex` in common.wgsl).
//!
//! An instance is posed again (its signature, bounds and shadow corners) only when it was set or
//! what it depends on changed, and each frame slot rewrites only the rows of its buffers that
//! differ from what they hold.
//!
//! Handles are generational: using one after its object was destroyed is an error, never a
//! different object.
use std::collections::HashMap;

use super::View;
use super::look::SurfaceModel;
use super::math::{self, Frustum, M4, V3};
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
    /// Water surfaces ([`Shading::Water`]) use procedural wave normals (`normal_texture` is ignored).
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

pub(in crate::render) struct MeshEntry {
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
    /// What the next morph of the range the last one rewrote needs ([`Morph`]).
    morph: Option<Morph>,
}

/// What a mesh keeps of the range its morphs rewrite (`first..end`), frame after frame, so that
/// a morph visits only the range's vertices: the bounds and [`zeros`] of the vertices outside
/// it, and for a skinned mesh the range's vertices each joint moves, as `(joint, vertex)` by
/// joint, each joint's in the order [`joint_boxes`] meets them.
struct Morph {
    first: usize,
    end: usize,
    outside: (V3, V3),
    outside_zeros: [u8; 3],
    moves: Vec<(u32, u32)>,
}

impl Morph {
    /// What morphs of `first..end` need of a mesh with these `positions` (the range's as
    /// morphed) and, when skinned, the range's skin `records`.
    fn new(positions: &[V3], first: usize, end: usize, records: Option<&[[f32; 4]]>) -> Self {
        let (before, after) = (&positions[..first], &positions[end..]);
        let ((blo, bhi), (alo, ahi)) = (bounds(before), bounds(after));
        let mut moves = vec![];
        for (v, pair) in records.unwrap_or(&[]).chunks_exact(2).enumerate() {
            for (j, w) in pair[0].iter().zip(pair[1]) {
                if w <= 0.0 {
                    continue;
                }
                moves.push((*j as u32, v as u32));
            }
        }
        moves.sort_by_key(|&(j, _)| j);
        let (bz, az) = (zeros(before), zeros(after));
        Self {
            first,
            end,
            outside: (
                std::array::from_fn(|k| blo[k].min(alo[k])),
                std::array::from_fn(|k| bhi[k].max(ahi[k])),
            ),
            outside_zeros: std::array::from_fn(|k| bz[k] | az[k]),
            moves,
        }
    }

    /// The bounds of the mesh with the range at `moved`: those of the range and of the rest
    /// joined, the bits [`bounds`] of all `positions` gives (which it makes when a bound of 0
    /// could hold either zero).
    fn bounds(&self, positions: &[V3], moved: &[V3]) -> (V3, V3) {
        let ((olo, ohi), (mlo, mhi)) = (self.outside, bounds(moved));
        let lo: V3 = std::array::from_fn(|k| olo[k].min(mlo[k]));
        let hi: V3 = std::array::from_fn(|k| ohi[k].max(mhi[k]));
        let unsure = (0..3).any(|k| lo[k] == 0.0 || hi[k] == 0.0) && {
            let z = zeros(moved);
            (0..3).any(|k| (self.outside_zeros[k] | z[k]) == 3 && (lo[k] == 0.0 || hi[k] == 0.0))
        };
        if unsure { bounds(positions) } else { (lo, hi) }
    }

    /// Widens `held` (a skinned mesh's joint boxes) to hold the range at `moved`, as joining
    /// the [`joint_boxes`] of `moved` would; returns how many boxes changed.
    fn widen(&self, held: &mut [Option<(V3, V3)>], moved: &[V3]) -> u32 {
        let mut changed = 0;
        for run in self.moves.chunk_by(|a, b| a.0 == b.0) {
            let Some(h) = held.get_mut(run[0].0 as usize) else {
                continue;
            };
            let first = moved[run[0].1 as usize];
            let b = run.iter().fold((first, first), |b, &(_, v)| {
                let p = moved[v as usize];
                (
                    std::array::from_fn(|k| b.0[k].min(p[k])),
                    std::array::from_fn(|k| b.1[k].max(p[k])),
                )
            });
            let was = *h;
            let x = h.get_or_insert(b);
            *x = (
                std::array::from_fn(|k| x.0[k].min(b.0[k])),
                std::array::from_fn(|k| x.1[k].max(b.1[k])),
            );
            if *h != was {
                changed += 1;
            }
        }
        changed
    }
}

/// Where a skinned mesh's vertices' joints and weights lie in the shared skin records.
#[derive(Clone, Copy, Debug)]
struct SkinRange {
    first: u32,
    count: u32,
    /// The highest joint its vertices use: an instance needs more matrices than this.
    max_joint: u16,
}

pub(in crate::render) struct MaterialEntry {
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
    /// The joint matrices its skinned mesh follows (`None`: drawn as modelled), which other
    /// instances can share ([`JointSet`]).
    joints: Option<u32>,
    /// Its place, visibility, tint, joints or material were set since it was last posed.
    dirty: bool,
    /// What it depends on beyond itself as of its last pose ([`stamp`]): it is posed
    /// again when that changed or it is `dirty`, and otherwise stands as it was.
    stamp: [u32; 4],
    /// What its picture depends on as of the last frame ([`signature`]), and for how many
    /// frames since then it has stayed the same.
    sig: u64,
    still: u32,
    /// Its world bounds as drawn this frame (none while hidden or incomplete).
    world_bounds: Option<(V3, V3)>,
    /// Its mesh's box placed by its model, while shown: what fits the shadow cascades.
    modelled: (V3, V3),
    /// Where its image and shadows lie in the cached picture, while it is drawn there.
    cached: Option<Rect>,
    /// Where it lay as last drawn over the cached picture with the depth and normals it writes
    /// (live and opaque), while it was ([`track_cached_props`]).
    live: Option<Rect>,
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
    /// Its [`shadow_key`] as posed.
    key: u64,
    /// Its shadow as the maps hold it, while they hold one.
    drawn: Option<DrawnShadow>,
}

/// What a caster's shadow in the props' shadow maps was drawn from ([`commit_prop_shadows`]).
struct DrawnShadow {
    key: u64,
    corners: Vec<V3>,
    bounds: (V3, V3),
}

/// Joint matrices one or more instances follow: every skinned part of a model instance shares
/// its skin's, so they are set, compared and written to the frame's joint buffer once.
struct JointSet {
    matrices: Vec<M4>,
    /// Counts changes to the matrices; `digest` hashes them as of revision `digested`.
    revision: u32,
    digest: u64,
    digested: Option<u32>,
    /// Instances that follow it (it goes with the last).
    users: u32,
    /// Its first row in the joint rows of frame `placed` ([`Renderer::prepare_props`]).
    first: u32,
    placed: u64,
}

/// The joint matrices an instance follows.
fn joints_of<'a>(sets: &'a Pool<JointSet>, inst: &InstanceEntry) -> &'a [M4] {
    inst.joints
        .and_then(|id| sets.get(id))
        .map_or(&[], |s| &s.matrices[..])
}

/// A hash of joint matrices, a number at a time (bijective in each, so one changed number
/// always changes it).
fn digest_matrices(matrices: &[M4]) -> u64 {
    matrices
        .as_flattened()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325 ^ matrices.len() as u64, |h, x| {
            (h ^ u64::from(x.to_bits())).wrapping_mul(0x0100_0000_01b3)
        })
}

/// One instanced draw of this frame: instances `first..first + count` of the buffer.
#[derive(Clone, Copy, Debug)]
pub(in crate::render) struct Batch {
    mesh: u32,
    material: u32,
    first: u32,
    count: u32,
    layer: Layer,
    /// Its instances are drawn into the cached picture rather than over it each frame.
    settled: bool,
}

mod api;
mod cache;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(in crate::render) use cache::units_sent;
use cache::zeros;
pub(in crate::render) use cache::{
    CacheChanges, CacheView, PropSet, Props, Rect, SHADOW_DRIFT_TEXELS, ShadowMap, bounds,
    box_rect, commit_prop_shadows, corners, merge_boxes, ray_box, ray_triangle, screen_rect,
    track_cached_props, union,
};
pub use cache::{Pick, WorldPick};
