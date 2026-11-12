//! Content-addressed Derived Data Cache (`DDC`) and non-destructive sidecar model fixups.
//!
//! Each `.glb` model's cooked [`NodeWorld`] (post-mesh-optimization, LOD decimation, unit-scale
//! normalization, and pivot alignment) is keyed deterministically by:
//!
//! `Sha256("ganbate-ddc-v1" || TOOL_VERSION || keep_images || source_sha256 || canonical_import_json)`
//!
//! Disk-backed builds store entries atomically under `<project>/.ganbate/cache/ddc/<xx>/<hash>.gbdc`
//! so unchanged models load in `< 1 ms` without reparsing glTF JSON, decoding images, or rerunning
//! vertex-cache optimization. Browser/WASM Studio sessions use [`MemoryDdc`] (and optional `.gbdc`
//! blob export/import) with the exact same key and binary codec.
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use ganbate_assets::cooked::MAX_BYTES;
use ganbate_assets::glb::{self, NodeWorld, SourceNode};
use ganbate_assets::mesh_opt::decimate_mesh;
use ganbate_assets::optimize_mesh;
use ganbate_core::Vec3;
use ganbate_core::digest::{hex, sha256};
use ganbate_core::json::{self, Value};
use ganbate_core::world::{
    AlphaMode, Emission, EncodedImage, IMAGE_TYPES, MAX_LIGHTS, MAX_MATERIALS, MAX_NAME_BYTES,
    MAX_TEXTURE_SIZE, MAX_TEXTURES, Material, Mesh, PointLight, Shading, Texture, Transform, World,
};

const DDC_MAGIC: &[u8; 4] = b"GBDC";
const DDC_FORMAT_VERSION: u32 = 1;
pub const DDC_TOOL_VERSION: &str = "ganbate-ddc-0.2.0-v1";
pub const DEFAULT_LOD_RATIOS: [f32; 4] = [1.0, 0.5, 0.25, 0.125];

/// Non-destructive pivot alignment applied at cook/load time without mutating `.glb` bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PivotPreset {
    #[default]
    Origin,
    BottomCenter,
    Centroid,
}

impl PivotPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Origin => "origin",
            Self::BottomCenter => "bottom_center",
            Self::Centroid => "centroid",
        }
    }

    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "" | "origin" => Ok(Self::Origin),
            "bottom_center" | "bottom" | "ground" => Ok(Self::BottomCenter),
            "centroid" | "center" => Ok(Self::Centroid),
            other => Err(format!("unknown pivot preset {other:?}")),
        }
    }
}

/// Non-destructive model import settings stored in `<model>.glb.meta.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelImportSettings {
    pub pivot: PivotPreset,
    pub offset: [f32; 3],
    pub scale: f32,
    pub target_height: Option<f32>,
    pub active_lod: u8,
    pub lods: Vec<f32>,
}

impl Default for ModelImportSettings {
    fn default() -> Self {
        Self {
            pivot: PivotPreset::Origin,
            offset: [0.0, 0.0, 0.0],
            scale: 1.0,
            target_height: None,
            active_lod: 0,
            lods: DEFAULT_LOD_RATIOS.to_vec(),
        }
    }
}

impl ModelImportSettings {
    /// Whether this configuration leaves imported vertices and root transforms untouched.
    pub fn is_identity(&self) -> bool {
        self.pivot == PivotPreset::Origin
            && self.offset.iter().all(|v| v.abs() < 1e-7)
            && (self.scale - 1.0).abs() < 1e-7
            && self.target_height.is_none()
            && self.active_lod_ratio() >= 0.999
    }

    pub fn active_lod_ratio(&self) -> f32 {
        if self.active_lod == 0 {
            return 1.0;
        }
        self.lods
            .get(self.active_lod as usize)
            .copied()
            .unwrap_or(1.0)
            .clamp(0.01, 1.0)
    }

    pub fn from_sidecar(sidecar: &Value) -> Result<Self, String> {
        let import = sidecar.get("import");
        let lods = match sidecar.get("lods") {
            Some(v) => parse_lod_ratios(v)?,
            None => DEFAULT_LOD_RATIOS.to_vec(),
        };
        let Some(imp) = import else {
            return Ok(Self {
                lods,
                ..Self::default()
            });
        };
        let Value::Object(map) = imp else {
            return Err("sidecar \"import\" must be an object".into());
        };
        let pivot = match map.get("pivot") {
            None | Some(Value::Null) => PivotPreset::Origin,
            Some(v) => PivotPreset::parse(v.string()?)?,
        };
        let offset = match map.get("offset") {
            None | Some(Value::Null) => [0.0, 0.0, 0.0],
            Some(v) => parse_vec3(v, "import.offset")?,
        };
        let scale = match map.get("scale") {
            None | Some(Value::Null) => 1.0,
            Some(v) => {
                let s = v.float()?;
                if !s.is_finite() || !(1e-5..=1e5).contains(&s) {
                    return Err("import.scale must be between 0.00001 and 100000".into());
                }
                s
            }
        };
        let target_height = match map.get("targetHeight").or_else(|| map.get("target_height")) {
            None | Some(Value::Null) => None,
            Some(v) => {
                let h = v.float()?;
                if !h.is_finite() || !(1e-4..=1e5).contains(&h) {
                    return Err("import.target_height must be between 0.0001 and 100000".into());
                }
                Some(h)
            }
        };
        let active_lod = match map.get("activeLod").or_else(|| map.get("active_lod")) {
            None | Some(Value::Null) => 0,
            Some(v) => {
                let l = v.unsigned()?;
                if l > 3 {
                    return Err("import.active_lod must be 0..=3".into());
                }
                l as u8
            }
        };
        Ok(Self {
            pivot,
            offset,
            scale,
            target_height,
            active_lod,
            lods,
        })
    }

    pub fn to_import_value(&self) -> Value {
        let mut m = BTreeMap::new();
        if self.pivot != PivotPreset::Origin {
            m.insert("pivot".into(), Value::String(self.pivot.as_str().into()));
        }
        if self.offset.iter().any(|v| v.abs() >= 1e-7) {
            m.insert("offset".into(), vec_to_value(&self.offset));
        }
        if (self.scale - 1.0).abs() >= 1e-7 {
            m.insert("scale".into(), Value::Number(f64::from(self.scale)));
        }
        if let Some(th) = self.target_height {
            m.insert("target_height".into(), Value::Number(f64::from(th)));
        }
        if self.active_lod > 0 {
            m.insert(
                "active_lod".into(),
                Value::Number(f64::from(self.active_lod)),
            );
        }
        Value::Object(m)
    }

    /// Canonical JSON representation used solely to salt the DDC key. Only properties that alter
    /// cooked geometry (`pivot`, `offset`, `scale`, `target_height`, `active_lod_ratio`) are hashed
    /// so authoring sockets or colliders never invalidates the cooked geometry cache.
    pub fn canonical_cache_descriptor(&self) -> String {
        let mut m = BTreeMap::new();
        m.insert(
            "pivot".to_string(),
            Value::String(self.pivot.as_str().into()),
        );
        m.insert("offset".into(), vec_to_value(&self.offset.map(clean_zero)));
        m.insert(
            "scale".into(),
            Value::Number(f64::from(clean_zero(self.scale))),
        );
        m.insert(
            "target_height".into(),
            self.target_height
                .map_or(Value::Null, |h| Value::Number(f64::from(clean_zero(h)))),
        );
        m.insert(
            "lod_ratio".into(),
            Value::Number(f64::from(clean_zero(self.active_lod_ratio()))),
        );
        json::to_canonical(&Value::Object(m)).unwrap_or_default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColliderShape {
    Box,
    Sphere,
    Capsule,
    ConvexHull,
}

impl ColliderShape {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Box => "box",
            Self::Sphere => "sphere",
            Self::Capsule => "capsule",
            Self::ConvexHull => "convex_hull",
        }
    }

    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "box" | "cuboid" => Ok(Self::Box),
            "sphere" => Ok(Self::Sphere),
            "capsule" => Ok(Self::Capsule),
            "convex_hull" | "hull" | "convex" => Ok(Self::ConvexHull),
            other => Err(format!("unknown collider shape {other:?}")),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ColliderDef {
    pub shape: ColliderShape,
    pub center: [f32; 3],
    pub half_extents: [f32; 3],
    pub radius: f32,
    pub height: f32,
    pub hull_points: Vec<[f32; 3]>,
}

impl ColliderDef {
    pub fn from_value(v: &Value) -> Result<Self, String> {
        let Value::Object(map) = v else {
            return Err("collider must be an object".into());
        };
        let shape =
            ColliderShape::parse(map.get("shape").ok_or("collider needs a shape")?.string()?)?;
        let center = match map.get("center") {
            None | Some(Value::Null) => [0.0, 0.0, 0.0],
            Some(c) => parse_vec3(c, "collider.center")?,
        };
        let half_extents = match map.get("halfExtents").or_else(|| map.get("half_extents")) {
            None | Some(Value::Null) => [0.5, 0.5, 0.5],
            Some(h) => {
                let ext = parse_vec3(h, "collider.half_extents")?;
                if ext.iter().any(|&x| x < 0.0) {
                    return Err("collider.half_extents must be non-negative".into());
                }
                ext
            }
        };
        let radius = match map.get("radius") {
            None | Some(Value::Null) => 0.5,
            Some(r) => {
                let rad = r.float()?;
                if !rad.is_finite() || rad < 0.0 {
                    return Err("collider.radius must be non-negative".into());
                }
                rad
            }
        };
        let height = match map.get("height") {
            None | Some(Value::Null) => 1.0,
            Some(h) => {
                let ht = h.float()?;
                if !ht.is_finite() || ht < 0.0 {
                    return Err("collider.height must be non-negative".into());
                }
                ht
            }
        };
        let mut hull_points = Vec::new();
        if let Some(pts) = map.get("hullPoints").or_else(|| map.get("hull_points"))
            && !matches!(pts, Value::Null)
        {
            let arr = pts.array()?;
            if arr.len() > 128 {
                return Err("collider.hull_points holds at most 128 vertices".into());
            }
            for p in arr {
                hull_points.push(parse_vec3(p, "collider.hull_points")?);
            }
        }
        Ok(Self {
            shape,
            center,
            half_extents,
            radius,
            height,
            hull_points,
        })
    }

    pub fn to_value(&self) -> Value {
        let mut m = BTreeMap::from([
            (
                "shape".to_string(),
                Value::String(self.shape.as_str().into()),
            ),
            ("center".into(), vec_to_value(&self.center.map(clean_zero))),
        ]);
        match self.shape {
            ColliderShape::Box => {
                m.insert("half_extents".into(), vec_to_value(&self.half_extents));
            }
            ColliderShape::Sphere => {
                m.insert("radius".into(), Value::Number(f64::from(self.radius)));
            }
            ColliderShape::Capsule => {
                m.insert("radius".into(), Value::Number(f64::from(self.radius)));
                m.insert("height".into(), Value::Number(f64::from(self.height)));
            }
            ColliderShape::ConvexHull => {
                m.insert("half_extents".into(), vec_to_value(&self.half_extents));
                m.insert(
                    "hull_points".into(),
                    Value::Array(self.hull_points.iter().map(|p| vec_to_value(p)).collect()),
                );
            }
        }
        Value::Object(m)
    }
}

/// Named attachment socket authored on a model (`ground`, `hand_r`, `top_surface`, etc.).
#[derive(Clone, Debug, PartialEq)]
pub struct SocketDef {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
}

impl Default for SocketDef {
    fn default() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

impl SocketDef {
    pub fn from_value(v: &Value) -> Result<Self, String> {
        let Value::Object(map) = v else {
            return Err("socket definition must be an object".into());
        };
        let translation = match map.get("translation") {
            None | Some(Value::Null) => [0.0, 0.0, 0.0],
            Some(t) => parse_vec3(t, "socket.translation")?,
        };
        let rotation = match map.get("rotation") {
            None | Some(Value::Null) => [0.0, 0.0, 0.0, 1.0],
            Some(r) => {
                let q = parse_vec4(r, "socket.rotation")?;
                let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
                if !len.is_finite() || len < 1e-6 {
                    return Err("socket.rotation must be a non-zero quaternion".into());
                }
                let qn = q.map(|c| c / len);
                if qn[3] < 0.0 { qn.map(|c| -c) } else { qn }
            }
        };
        Ok(Self {
            translation,
            rotation,
        })
    }

    pub fn to_value(&self) -> Value {
        let mut m = BTreeMap::from([(
            "translation".to_string(),
            vec_to_value(&self.translation.map(clean_zero)),
        )]);
        let is_id_rot = (self.rotation[0]).abs() < 1e-6
            && (self.rotation[1]).abs() < 1e-6
            && (self.rotation[2]).abs() < 1e-6
            && (self.rotation[3] - 1.0).abs() < 1e-6;
        if !is_id_rot {
            m.insert(
                "rotation".into(),
                vec_to_value(&self.rotation.map(clean_zero)),
            );
        }
        Value::Object(m)
    }
}

/// Validated rich metadata stored in `<asset>.meta.json` alongside `ganbate`, `version`, `id`, `kind`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AssetMeta {
    pub import: ModelImportSettings,
    pub collider: Option<ColliderDef>,
    pub sockets: BTreeMap<String, SocketDef>,
}

impl AssetMeta {
    pub fn from_sidecar(sidecar: &Value) -> Result<Self, String> {
        let import = ModelImportSettings::from_sidecar(sidecar)?;
        let collider = match sidecar.get("collider") {
            None | Some(Value::Null) => None,
            Some(v) => Some(ColliderDef::from_value(v)?),
        };
        let mut sockets = BTreeMap::new();
        if let Some(s_val) = sidecar.get("sockets")
            && !matches!(s_val, Value::Null)
        {
            let Value::Object(map) = s_val else {
                return Err("sidecar \"sockets\" must be an object".into());
            };
            if map.len() > 128 {
                return Err("sidecar \"sockets\" holds at most 128 sockets".into());
            }
            for (name, def) in map {
                validate_socket_name(name)?;
                sockets.insert(name.clone(), SocketDef::from_value(def)?);
            }
        }
        Ok(Self {
            import,
            collider,
            sockets,
        })
    }

    /// Serializes the editable authoring sections (`import`, `lods`, `collider`, `sockets`) for UI
    /// inspection or merging into an existing `.meta.json` sidecar.
    pub fn to_value(&self) -> Value {
        let mut m = BTreeMap::new();
        m.insert("import".to_string(), self.import.to_import_value());
        if self.import.lods != DEFAULT_LOD_RATIOS {
            m.insert("lods".into(), vec_to_value(&self.import.lods));
        }
        if let Some(c) = &self.collider {
            m.insert("collider".into(), c.to_value());
        }
        if !self.sockets.is_empty() {
            let socks = self
                .sockets
                .iter()
                .map(|(k, v)| (k.clone(), v.to_value()))
                .collect();
            m.insert("sockets".into(), Value::Object(socks));
        }
        Value::Object(m)
    }

    /// Applies this validated metadata onto an existing sidecar object while strictly preserving
    /// `ganbate`, `version`, `id`, and `kind`.
    pub fn apply_to_sidecar(&self, existing: &Value) -> Result<Value, String> {
        let Value::Object(mut base) = existing.clone() else {
            return Err("sidecar is not an object".into());
        };
        base.insert("import".into(), self.import.to_import_value());
        if self.import.lods != DEFAULT_LOD_RATIOS {
            base.insert("lods".into(), vec_to_value(&self.import.lods));
        } else {
            base.remove("lods");
        }
        match &self.collider {
            Some(c) => {
                base.insert("collider".into(), c.to_value());
            }
            None => {
                base.remove("collider");
            }
        }
        if self.sockets.is_empty() {
            base.remove("sockets");
        } else {
            let socks = self
                .sockets
                .iter()
                .map(|(k, v)| (k.clone(), v.to_value()))
                .collect();
            base.insert("sockets".into(), Value::Object(socks));
        }
        Ok(Value::Object(base))
    }
}

fn validate_socket_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > 64 {
        return Err("socket name must be 1..=64 characters".into());
    }
    if !name
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
    {
        return Err(format!(
            "socket name {name:?} may contain only ASCII letters, digits, '_', '-' or '.'"
        ));
    }
    Ok(())
}

fn clean_zero(v: f32) -> f32 {
    if v.abs() < 1e-7 { 0.0 } else { v }
}

fn vec_to_value(vals: &[f32]) -> Value {
    Value::Array(vals.iter().map(|&v| Value::Number(f64::from(v))).collect())
}

fn parse_vec3(v: &Value, what: &str) -> Result<[f32; 3], String> {
    let arr = v
        .array()
        .map_err(|_| format!("{what}: expected [x, y, z]"))?;
    if arr.len() != 3 {
        return Err(format!("{what}: expected 3 numbers"));
    }
    let out = [arr[0].float()?, arr[1].float()?, arr[2].float()?];
    if !out.iter().all(|x| x.is_finite()) {
        return Err(format!("{what}: numbers must be finite"));
    }
    Ok(out)
}

fn parse_vec4(v: &Value, what: &str) -> Result<[f32; 4], String> {
    let arr = v
        .array()
        .map_err(|_| format!("{what}: expected [x, y, z, w]"))?;
    if arr.len() != 4 {
        return Err(format!("{what}: expected 4 numbers"));
    }
    let out = [
        arr[0].float()?,
        arr[1].float()?,
        arr[2].float()?,
        arr[3].float()?,
    ];
    if !out.iter().all(|x| x.is_finite()) {
        return Err(format!("{what}: numbers must be finite"));
    }
    Ok(out)
}

fn parse_lod_ratios(v: &Value) -> Result<Vec<f32>, String> {
    let arr = v.array().map_err(|_| "lods: expected a list of ratios")?;
    if arr.is_empty() || arr.len() > 4 {
        return Err("lods: expected 1..=4 ratios".into());
    }
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let r = item.float()?;
        if !r.is_finite() || !(0.01..=1.0).contains(&r) {
            return Err("lods: each ratio must be in 0.01..=1.0".into());
        }
        out.push(r);
    }
    Ok(out)
}

/// Computes model-space bounds `(lo, hi)` across all placed mesh vertices of `nw`.
pub fn model_bounds(nw: &NodeWorld) -> Option<([f32; 3], [f32; 3])> {
    let mut model_space: Vec<Transform> = Vec::with_capacity(nw.nodes.len());
    for n in &nw.nodes {
        let m = match n.parent {
            Some(p) => model_space[p].compose(n.local),
            None => n.local,
        };
        model_space.push(m);
    }
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    let mut any = false;
    for (n, &ms) in nw.nodes.iter().zip(&model_space) {
        for mesh_idx in n.meshes.clone() {
            if let Some(mesh) = nw.world.meshes.get(mesh_idx as usize) {
                for &p in &mesh.positions {
                    let w = ms.point(p);
                    for (k, c) in [w.0, w.1, w.2].into_iter().enumerate() {
                        lo[k] = lo[k].min(c);
                        hi[k] = hi[k].max(c);
                    }
                    any = true;
                }
            }
        }
    }
    if !any {
        for mesh in &nw.world.meshes {
            for &p in &mesh.positions {
                for (k, c) in [p.0, p.1, p.2].into_iter().enumerate() {
                    lo[k] = lo[k].min(c);
                    hi[k] = hi[k].max(c);
                }
                any = true;
            }
        }
    }
    any.then_some((lo, hi))
}

/// Optimizes (or LOD-decimates) all meshes in `nw` and applies non-destructive `settings`
/// (`scale`, `target_height`, `pivot`, `offset`) to vertex positions and root node transforms.
pub fn prepare_node_world(nw: &mut NodeWorld, settings: &ModelImportSettings) {
    let lod_ratio = settings.active_lod_ratio();
    for mesh in &mut nw.world.meshes {
        if lod_ratio < 0.999 {
            *mesh = decimate_mesh(mesh, lod_ratio);
        } else {
            optimize_mesh(mesh);
        }
    }
    if settings.pivot == PivotPreset::Origin
        && settings.offset.iter().all(|v| v.abs() < 1e-7)
        && (settings.scale - 1.0).abs() < 1e-7
        && settings.target_height.is_none()
    {
        return;
    }

    let bounds = model_bounds(nw);
    let raw_h = bounds.map_or(0.0, |(lo, hi)| (hi[1] - lo[1]).max(0.0));
    let eff_scale = match settings.target_height {
        Some(th) if th.is_finite() && th > 0.0 && raw_h > 1e-6 => th / raw_h,
        _ => settings.scale,
    }
    .clamp(1e-5, 1e5);

    let pivot_shift = match bounds {
        Some((lo, hi)) => {
            let cx = (lo[0] + hi[0]) * 0.5 * eff_scale;
            let cy = (lo[1] + hi[1]) * 0.5 * eff_scale;
            let cz = (lo[2] + hi[2]) * 0.5 * eff_scale;
            match settings.pivot {
                PivotPreset::Origin => [0.0, 0.0, 0.0],
                PivotPreset::BottomCenter => [-cx, -lo[1] * eff_scale, -cz],
                PivotPreset::Centroid => [-cx, -cy, -cz],
            }
        }
        None => [0.0, 0.0, 0.0],
    };
    let delta = [
        pivot_shift[0] + settings.offset[0],
        pivot_shift[1] + settings.offset[1],
        pivot_shift[2] + settings.offset[2],
    ];

    if (eff_scale - 1.0).abs() >= 1e-7 {
        for mesh in &mut nw.world.meshes {
            for p in &mut mesh.positions {
                p.0 *= eff_scale;
                p.1 *= eff_scale;
                p.2 *= eff_scale;
            }
        }
        for n in &mut nw.nodes {
            n.local.0[12] *= eff_scale;
            n.local.0[13] *= eff_scale;
            n.local.0[14] *= eff_scale;
            if let Some(l) = &mut n.light
                && l.range > 0.0
            {
                l.range *= eff_scale;
            }
        }
    }

    if delta.iter().any(|v| v.abs() >= 1e-7) {
        for n in &mut nw.nodes {
            if n.parent.is_none() {
                n.local.0[12] += delta[0];
                n.local.0[13] += delta[1];
                n.local.0[14] += delta[2];
            }
        }
    }
}

/// Content-addressed 256-bit key for a cooked [`NodeWorld`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DdcKey(pub [u8; 32]);

impl DdcKey {
    pub fn compute(
        source_sha256: &[u8; 32],
        settings: &ModelImportSettings,
        keep_images: bool,
    ) -> Self {
        let desc = settings.canonical_cache_descriptor();
        let mut preimage = Vec::with_capacity(64 + DDC_TOOL_VERSION.len() + desc.len());
        preimage.extend_from_slice(b"ganbate-ddc-key\0");
        preimage.extend_from_slice(DDC_TOOL_VERSION.as_bytes());
        preimage.push(0);
        preimage.push(u8::from(keep_images));
        preimage.extend_from_slice(source_sha256);
        preimage.extend_from_slice(desc.as_bytes());
        Self(sha256(&preimage))
    }

    pub fn from_bytes(
        source_bytes: &[u8],
        settings: &ModelImportSettings,
        keep_images: bool,
    ) -> Self {
        let src_hash = sha256(source_bytes);
        Self::compute(&src_hash, settings, keep_images)
    }

    pub fn hex(&self) -> String {
        hex(&self.0)
    }
}

/// Pure in-memory Derived Data Cache for browser/WASM Studio sessions and unit tests.
#[derive(Clone, Debug, Default)]
pub struct MemoryDdc {
    entries: HashMap<DdcKey, NodeWorld>,
    pub hits: u64,
    pub misses: u64,
}

impl MemoryDdc {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, key: &DdcKey) -> Option<&NodeWorld> {
        self.entries.get(key)
    }

    pub fn insert(&mut self, key: DdcKey, nw: NodeWorld) {
        self.entries.insert(key, nw);
    }

    pub fn load_or_cook(
        &mut self,
        label: &str,
        bytes: &[u8],
        settings: &ModelImportSettings,
        keep_images: bool,
    ) -> Result<NodeWorld, String> {
        let key = DdcKey::from_bytes(bytes, settings, keep_images);
        if let Some(cached) = self.entries.get(&key) {
            self.hits += 1;
            return Ok(cached.clone());
        }
        self.misses += 1;
        let mut nw = glb::import_nodes(bytes, keep_images).map_err(|e| format!("{label}: {e}"))?;
        prepare_node_world(&mut nw, settings);
        self.entries.insert(key, nw.clone());
        Ok(nw)
    }
}

/// Returns `<project_root>/.ganbate/cache/ddc`.
pub fn ddc_root_dir(project_root: &Path) -> PathBuf {
    project_root.join(".ganbate").join("cache").join("ddc")
}

/// Returns `<project_root>/.ganbate/cache/ddc/<xx>/<hex>.gbdc`.
pub fn ddc_entry_path(project_root: &Path, key: &DdcKey) -> PathBuf {
    let h = key.hex();
    ddc_root_dir(project_root)
        .join(&h[..2])
        .join(format!("{h}.gbdc"))
}

/// Loads a model's cooked [`NodeWorld`] from `<project>/.ganbate/cache/ddc/` on cache hit, or
/// imports, optimizes, transforms, and atomically caches `.gbdc` on cache miss.
pub fn load_or_cook_disk(
    project_root: &Path,
    label: &str,
    bytes: &[u8],
    settings: &ModelImportSettings,
    keep_images: bool,
) -> Result<(NodeWorld, bool), String> {
    let key = DdcKey::from_bytes(bytes, settings, keep_images);
    let cache_path = ddc_entry_path(project_root, &key);
    if let Ok(cached_bytes) = ganbate_assets::read_asset(&cache_path)
        && let Ok(nw) = decode_cached_model(&cached_bytes)
    {
        return Ok((nw, true));
    }

    let mut nw = glb::import_nodes(bytes, keep_images).map_err(|e| format!("{label}: {e}"))?;
    prepare_node_world(&mut nw, settings);

    if let Ok(encoded) = encode_cached_model(&nw) {
        if let Some(parent) = cache_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = ganbate_assets::write_asset(&cache_path, &encoded);
    }
    Ok((nw, false))
}

// ---- Exact Little-Endian Binary Codec (`.gbdc` v1) -------------------------------------------

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_usize(out: &mut Vec<u8>, v: usize) -> Result<(), String> {
    let n = u32::try_from(v).map_err(|_| "DDC count exceeds u32")?;
    put_u32(out, n);
    Ok(())
}

fn put_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_opt_u32(out: &mut Vec<u8>, v: Option<u32>) {
    put_u32(out, v.unwrap_or(u32::MAX));
}

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) -> Result<(), String> {
    put_usize(out, b.len())?;
    out.extend_from_slice(b);
    Ok(())
}

fn put_str(out: &mut Vec<u8>, s: &str) -> Result<(), String> {
    put_bytes(out, s.as_bytes())
}

/// Encodes a post-fixup [`NodeWorld`] into exact IEEE-754 little-endian `.gbdc` bytes (zero lossy
/// quantization and zero zlib overhead so DDC hits are bit-identical and sub-millisecond).
pub fn encode_cached_model(nw: &NodeWorld) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(256);
    out.extend_from_slice(DDC_MAGIC);
    put_u32(&mut out, DDC_FORMAT_VERSION);

    // 1. Meshes
    put_usize(&mut out, nw.world.meshes.len())?;
    for m in &nw.world.meshes {
        put_usize(&mut out, m.positions.len())?;
        for p in &m.positions {
            put_f32(&mut out, p.0);
            put_f32(&mut out, p.1);
            put_f32(&mut out, p.2);
        }
        put_usize(&mut out, m.normals.len())?;
        for n in &m.normals {
            put_f32(&mut out, n.0);
            put_f32(&mut out, n.1);
            put_f32(&mut out, n.2);
        }
        put_usize(&mut out, m.uvs.len())?;
        for uv in &m.uvs {
            put_f32(&mut out, uv[0]);
            put_f32(&mut out, uv[1]);
        }
        put_usize(&mut out, m.indices.len())?;
        for &idx in &m.indices {
            put_u32(&mut out, idx);
        }
        for c in m.color {
            put_f32(&mut out, c);
        }
        put_opt_u32(&mut out, m.material);
    }

    // 2. Materials
    put_usize(&mut out, nw.world.materials.len())?;
    for mat in &nw.world.materials {
        put_str(&mut out, &mat.name)?;
        for c in mat.base_color {
            put_f32(&mut out, c);
        }
        put_opt_u32(&mut out, mat.base_color_texture);
        put_opt_u32(&mut out, mat.normal_texture);
        put_f32(&mut out, mat.normal_scale);
        for c in mat.emissive {
            put_f32(&mut out, c);
        }
        put_opt_u32(&mut out, mat.emissive_texture);
        put_f32(&mut out, mat.metallic);
        put_f32(&mut out, mat.roughness);
        match mat.alpha {
            AlphaMode::Opaque => {
                out.push(0);
                put_f32(&mut out, 0.0);
            }
            AlphaMode::Mask(cutoff) => {
                out.push(1);
                put_f32(&mut out, cutoff);
            }
            AlphaMode::Blend => {
                out.push(2);
                put_f32(&mut out, 0.0);
            }
        }
        out.push(u8::from(mat.double_sided));
        out.push(match mat.shading {
            Shading::Standard => 0,
            Shading::Foliage => 1,
            Shading::Water => 2,
            Shading::Unlit => 3,
        });
        out.push(match mat.emission {
            Emission::Constant => 0,
            Emission::Lamp => 1,
        });
    }

    // 3. Textures
    put_usize(&mut out, nw.world.textures.len())?;
    for tex in &nw.world.textures {
        put_u32(&mut out, tex.width);
        put_u32(&mut out, tex.height);
        out.push(u8::from(tex.srgb));
        put_usize(&mut out, tex.levels.len())?;
        for lvl in &tex.levels {
            put_bytes(&mut out, lvl)?;
        }
        match &tex.encoded {
            None => out.push(0),
            Some(enc) => {
                out.push(1);
                put_str(&mut out, &enc.mime)?;
                put_bytes(&mut out, &enc.data)?;
            }
        }
    }

    // 4. Nodes
    put_usize(&mut out, nw.nodes.len())?;
    for n in &nw.nodes {
        put_str(&mut out, &n.name)?;
        put_opt_u32(&mut out, n.parent.map(|p| p as u32));
        for c in n.local.0 {
            put_f32(&mut out, c);
        }
        put_u32(&mut out, n.meshes.start);
        put_u32(&mut out, n.meshes.end);
        match n.light {
            None => out.push(0),
            Some(l) => {
                out.push(1);
                put_f32(&mut out, l.position.0);
                put_f32(&mut out, l.position.1);
                put_f32(&mut out, l.position.2);
                for c in l.color {
                    put_f32(&mut out, c);
                }
                put_f32(&mut out, l.intensity);
                put_f32(&mut out, l.range);
            }
        }
    }

    if out.len() > MAX_BYTES {
        return Err("DDC entry exceeds size limit".into());
    }
    Ok(out)
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn rem(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).ok_or("DDC length overflow")?;
        let s = self.buf.get(self.pos..end).ok_or("truncated DDC entry")?;
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn f32(&mut self) -> Result<f32, String> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn opt_u32(&mut self) -> Result<Option<u32>, String> {
        let v = self.u32()?;
        Ok((v != u32::MAX).then_some(v))
    }

    fn bytes(&mut self, max_len: usize) -> Result<Vec<u8>, String> {
        let len = self.u32()? as usize;
        if len > max_len || len > self.rem() {
            return Err("DDC byte slice exceeds bounds".into());
        }
        Ok(self.take(len)?.to_vec())
    }

    fn string(&mut self, max_len: usize) -> Result<String, String> {
        let b = self.bytes(max_len)?;
        String::from_utf8(b).map_err(|_| "invalid UTF-8 in DDC".into())
    }
}

pub fn decode_cached_model(bytes: &[u8]) -> Result<NodeWorld, String> {
    if bytes.len() > MAX_BYTES {
        return Err("DDC entry exceeds size limit".into());
    }
    let mut r = Reader { buf: bytes, pos: 0 };
    if r.take(4)? != DDC_MAGIC {
        return Err("invalid DDC magic".into());
    }
    if r.u32()? != DDC_FORMAT_VERSION {
        return Err("unsupported DDC version".into());
    }

    // 1. Meshes
    let mesh_count = r.u32()? as usize;
    if mesh_count > 65_536 || mesh_count.saturating_mul(28) > r.rem() {
        return Err("DDC mesh count out of bounds".into());
    }
    let mut meshes = Vec::with_capacity(mesh_count);
    for _ in 0..mesh_count {
        let p_len = r.u32()? as usize;
        if p_len.saturating_mul(12) > r.rem() {
            return Err("DDC positions out of bounds".into());
        }
        let mut positions = Vec::with_capacity(p_len);
        for _ in 0..p_len {
            positions.push(Vec3(r.f32()?, r.f32()?, r.f32()?));
        }

        let n_len = r.u32()? as usize;
        if n_len.saturating_mul(12) > r.rem() {
            return Err("DDC normals out of bounds".into());
        }
        let mut normals = Vec::with_capacity(n_len);
        for _ in 0..n_len {
            normals.push(Vec3(r.f32()?, r.f32()?, r.f32()?));
        }

        let uv_len = r.u32()? as usize;
        if uv_len.saturating_mul(8) > r.rem() {
            return Err("DDC uvs out of bounds".into());
        }
        let mut uvs = Vec::with_capacity(uv_len);
        for _ in 0..uv_len {
            uvs.push([r.f32()?, r.f32()?]);
        }

        let i_len = r.u32()? as usize;
        if i_len.saturating_mul(4) > r.rem() {
            return Err("DDC indices out of bounds".into());
        }
        let mut indices = Vec::with_capacity(i_len);
        for _ in 0..i_len {
            indices.push(r.u32()?);
        }

        let color = [r.f32()?, r.f32()?, r.f32()?];
        let material = r.opt_u32()?;
        meshes.push(Mesh {
            positions,
            normals,
            uvs,
            indices,
            color,
            material,
        });
    }

    // 2. Materials
    let mat_count = r.u32()? as usize;
    if mat_count > MAX_MATERIALS || mat_count.saturating_mul(56) > r.rem() {
        return Err("DDC material count out of bounds".into());
    }
    let mut materials = Vec::with_capacity(mat_count);
    for _ in 0..mat_count {
        let name = r.string(MAX_NAME_BYTES)?;
        let base_color = [r.f32()?, r.f32()?, r.f32()?, r.f32()?];
        let base_color_texture = r.opt_u32()?;
        let normal_texture = r.opt_u32()?;
        let normal_scale = r.f32()?;
        let emissive = [r.f32()?, r.f32()?, r.f32()?];
        let emissive_texture = r.opt_u32()?;
        let metallic = r.f32()?;
        let roughness = r.f32()?;
        let alpha_tag = r.u8()?;
        let cutoff = r.f32()?;
        let alpha = match alpha_tag {
            0 => AlphaMode::Opaque,
            1 => AlphaMode::Mask(cutoff),
            2 => AlphaMode::Blend,
            _ => return Err("invalid DDC alpha mode".into()),
        };
        let double_sided = r.u8()? != 0;
        let shading = match r.u8()? {
            0 => Shading::Standard,
            1 => Shading::Foliage,
            2 => Shading::Water,
            3 => Shading::Unlit,
            _ => return Err("invalid DDC shading mode".into()),
        };
        let emission = match r.u8()? {
            0 => Emission::Constant,
            1 => Emission::Lamp,
            _ => return Err("invalid DDC emission mode".into()),
        };
        materials.push(Material {
            name,
            base_color,
            base_color_texture,
            normal_texture,
            normal_scale,
            emissive,
            emissive_texture,
            metallic,
            roughness,
            alpha,
            double_sided,
            shading,
            emission,
        });
    }

    // 3. Textures
    let tex_count = r.u32()? as usize;
    if tex_count > MAX_TEXTURES || tex_count.saturating_mul(14) > r.rem() {
        return Err("DDC texture count out of bounds".into());
    }
    let mut textures = Vec::with_capacity(tex_count);
    for _ in 0..tex_count {
        let width = r.u32()?;
        let height = r.u32()?;
        if width == 0 || width > MAX_TEXTURE_SIZE || height == 0 || height > MAX_TEXTURE_SIZE {
            return Err("DDC texture dimensions out of bounds".into());
        }
        let srgb = r.u8()? != 0;
        let lvl_count = r.u32()? as usize;
        if lvl_count > 32 {
            return Err("DDC mip chain too deep".into());
        }
        let mut levels = Vec::with_capacity(lvl_count);
        for _ in 0..lvl_count {
            levels.push(r.bytes(MAX_BYTES)?);
        }
        let encoded = match r.u8()? {
            0 => None,
            1 => {
                let mime = r.string(64)?;
                if !IMAGE_TYPES.contains(&mime.as_str()) {
                    return Err("unsupported DDC image mime".into());
                }
                let data = r.bytes(MAX_BYTES)?;
                Some(EncodedImage { mime, data })
            }
            _ => return Err("invalid DDC texture tag".into()),
        };
        textures.push(Texture {
            width,
            height,
            srgb,
            levels,
            encoded,
        });
    }

    // 4. Nodes
    let node_count = r.u32()? as usize;
    if node_count > 250_000 || node_count.saturating_mul(77) > r.rem() {
        return Err("DDC node count out of bounds".into());
    }
    let mut nodes = Vec::with_capacity(node_count);
    for here in 0..node_count {
        let name = r.string(MAX_NAME_BYTES)?;
        let parent = match r.opt_u32()? {
            None => None,
            Some(p) if (p as usize) < here => Some(p as usize),
            Some(_) => return Err("DDC node parent out of topological order".into()),
        };
        let mut mat = [0.0f32; 16];
        for slot in &mut mat {
            *slot = r.f32()?;
        }
        let local = Transform(mat);
        if !local.valid() {
            return Err("invalid DDC node transform".into());
        }
        let m_start = r.u32()?;
        let m_end = r.u32()?;
        if m_start > m_end || (m_end as usize) > meshes.len() {
            return Err("DDC node mesh range out of bounds".into());
        }
        let light = match r.u8()? {
            0 => None,
            1 => {
                if nodes
                    .iter()
                    .filter(|n: &&SourceNode| n.light.is_some())
                    .count()
                    >= MAX_LIGHTS
                {
                    return Err("DDC light count exceeds limit".into());
                }
                let position = Vec3(r.f32()?, r.f32()?, r.f32()?);
                let color = [r.f32()?, r.f32()?, r.f32()?];
                let intensity = r.f32()?;
                let range = r.f32()?;
                Some(PointLight {
                    position,
                    color,
                    intensity,
                    range,
                })
            }
            _ => return Err("invalid DDC light tag".into()),
        };
        nodes.push(SourceNode {
            name,
            parent,
            local,
            meshes: m_start..m_end,
            light,
        });
    }

    if r.rem() != 0 {
        return Err("trailing bytes in DDC entry".into());
    }

    let world = World {
        meshes,
        instances: Vec::new(),
        materials,
        textures,
        lights: Vec::new(),
    };
    world.validate()?;
    Ok(NodeWorld { world, nodes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cook::{cook_resolved_with_ddc, cook_scene, flatten_checked, tests::sample_glb};
    use crate::project::tests::TempDir;
    use crate::project::{Kind, Manifest, Project, new_sidecar, write_json};
    use ganbate_core::id::Ids;
    use ganbate_scene::schema::engine_registry;

    #[test]
    fn disk_ddc_writes_content_addressed_cache_and_hits_in_under_5ms() {
        let t = TempDir::new("ddc-disk");
        let mut ids = Ids::seeded(41);
        let mut p = Project::create(&t.0, "DdcBench", &mut ids).unwrap();
        let glb = sample_glb();
        std::fs::create_dir_all(t.0.join("content/models")).unwrap();
        std::fs::write(t.0.join("content/models/prop.glb"), &glb).unwrap();
        let model_id = ids.asset();
        write_json(
            &t.0.join("content/models/prop.glb.meta.json"),
            &new_sidecar(&model_id, Kind::Model),
        )
        .unwrap();
        p.scan(None).unwrap();

        let settings = ModelImportSettings::default();
        let (cold_nw, hit0) =
            load_or_cook_disk(&p.root, "prop.glb", &glb, &settings, false).unwrap();
        assert!(!hit0);
        let key = DdcKey::from_bytes(&glb, &settings, false);
        assert!(ddc_entry_path(&p.root, &key).is_file());

        #[cfg(not(target_arch = "wasm32"))]
        let t0 = std::time::Instant::now();
        let (warm_nw, hit1) =
            load_or_cook_disk(&p.root, "prop.glb", &glb, &settings, false).unwrap();
        #[cfg(not(target_arch = "wasm32"))]
        {
            let elapsed = t0.elapsed();
            assert!(
                elapsed.as_millis() < 5,
                "expected warm DDC hit < 5ms, took {elapsed:?}"
            );
        }
        assert!(hit1);
        assert_eq!(cold_nw.world, warm_nw.world);
        assert_eq!(cold_nw.nodes.len(), warm_nw.nodes.len());
    }

    #[test]
    fn sidecar_pivot_and_scale_fixups_transform_cooked_world_without_mutating_glb() {
        let t = TempDir::new("ddc-fixup");
        let mut ids = Ids::seeded(42);
        let mut p = Project::create(&t.0, "Fixup", &mut ids).unwrap();
        let glb = sample_glb();
        std::fs::create_dir_all(t.0.join("content/models")).unwrap();
        let glb_path = t.0.join("content/models/statue.glb");
        std::fs::write(&glb_path, &glb).unwrap();
        let model_id = ids.asset();
        write_json(
            &t.0.join("content/models/statue.glb.meta.json"),
            &new_sidecar(&model_id, Kind::Model),
        )
        .unwrap();
        p.scan(None).unwrap();

        let scene_id = ids.asset();
        let scene = json::parse(
            format!(
                r#"{{"ganbate":"scene","version":1,"id":"{scene_id}","name":"Stage","entities":{{"00000000000000a1":{{"name":"Statue","components":{{"mesh":{{"model":"{model_id}"}}}}}}}}}}"#
            )
            .as_bytes(),
        )
        .unwrap();
        p.write_document("content/stage.scene.json", Kind::Scene, &scene)
            .unwrap();

        // Update sidecar with target_height = 6.0m and BottomCenter pivot + collider + socket.
        let mut sockets = BTreeMap::new();
        sockets.insert(
            "top_surface".into(),
            SocketDef {
                translation: [0.0, 6.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        );
        let meta = AssetMeta {
            import: ModelImportSettings {
                pivot: PivotPreset::BottomCenter,
                offset: [0.0, 0.0, 0.0],
                scale: 0.01,
                target_height: Some(6.0),
                active_lod: 0,
                lods: DEFAULT_LOD_RATIOS.to_vec(),
            },
            collider: Some(ColliderDef {
                shape: ColliderShape::Capsule,
                center: [0.0, 3.0, 0.0],
                half_extents: [1.0, 3.0, 1.0],
                radius: 1.0,
                height: 6.0,
                hull_points: vec![],
            }),
            sockets,
        };
        p.write_sidecar_meta(&model_id, &meta).unwrap();

        // Source GLB bytes remain 100% untouched.
        assert_eq!(std::fs::read(&glb_path).unwrap(), glb);

        let cooked = cook_scene(&p, &engine_registry(), &scene_id, false).unwrap();
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        for inst in &cooked.world.instances {
            for &v in &cooked.world.meshes[inst.mesh as usize].positions {
                let w = inst.transform.point(v);
                min_x = min_x.min(w.0);
                max_x = max_x.max(w.0);
                min_y = min_y.min(w.1);
                max_y = max_y.max(w.1);
            }
        }
        assert!(
            min_y.abs() < 1e-4,
            "expected bottom_center min_y=0, got {min_y}"
        );
        assert!(
            (max_y - 6.0).abs() < 1e-4,
            "expected target_height=6.0, got {max_y}"
        );
        assert!(
            ((min_x + max_x) * 0.5).abs() < 1e-4,
            "expected X centered, got [{min_x}, {max_x}]"
        );

        // Verify round-trip of collider & sockets from sidecar, and that socket edits do NOT
        // change the geometry DDC key!
        let key_before = DdcKey::from_bytes(&glb, &p.model_import_settings(&model_id), false);
        let mut meta2 = AssetMeta::from_sidecar(&p.read_sidecar(&model_id).unwrap()).unwrap();
        assert_eq!(meta2.sockets.len(), 1);
        assert_eq!(
            meta2.collider.as_ref().unwrap().shape,
            ColliderShape::Capsule
        );
        meta2.sockets.insert(
            "hand_r".into(),
            SocketDef {
                translation: [0.5, 1.2, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        );
        p.write_sidecar_meta(&model_id, &meta2).unwrap();
        let key_after = DdcKey::from_bytes(&glb, &p.model_import_settings(&model_id), false);
        assert_eq!(key_before, key_after);
    }

    #[test]
    fn in_memory_wasm_cooker_and_manifest_intake_round_trip() {
        let manifest_val = json::parse(
            br#"{"ganbate":"project","version":1,"id":"0123456789abcdef0123456789abcdef","name":"Web","content":["content"],"schemas":["schemas"],"editor":["editor"],"intake":["../ganbate-art/tripo3d-queue"]}"#,
        )
        .unwrap();
        let manifest = Manifest::parse(&manifest_val).unwrap();
        assert_eq!(manifest.intake, ["../ganbate-art/tripo3d-queue"]);
        let round = Manifest::parse(&manifest.to_value()).unwrap();
        assert_eq!(round.intake, manifest.intake);

        let reg = engine_registry();
        let model_id = "fedcba9876543210fedcba9876543210";
        let scene_doc = json::parse(
            format!(
                r#"{{"ganbate":"scene","version":1,"id":"0123456789abcdef0123456789abcdef","name":"Mem","entities":{{"00000000000000b1":{{"name":"Prop","components":{{"mesh":{{"model":"{model_id}"}}}}}}}}}}"#
            )
            .as_bytes(),
        )
        .unwrap();
        let resolved = flatten_checked(&scene_doc, &reg, &|id| Err(format!("no {id}"))).unwrap();
        let glb = sample_glb();
        let settings = ModelImportSettings {
            pivot: PivotPreset::Centroid,
            scale: 2.0,
            ..ModelImportSettings::default()
        };
        let mut ddc = MemoryDdc::new();
        let c1 = cook_resolved_with_ddc(&reg, &scene_doc, &resolved, true, &mut ddc, &|_| {
            Ok(("mem.glb".into(), glb.clone(), settings.clone()))
        })
        .unwrap();
        let c2 = cook_resolved_with_ddc(&reg, &scene_doc, &resolved, true, &mut ddc, &|_| {
            Ok(("mem.glb".into(), glb.clone(), settings.clone()))
        })
        .unwrap();
        assert_eq!(ddc.misses, 1);
        assert_eq!(ddc.hits, 1);
        assert_eq!(c1.world, c2.world);
    }
}
