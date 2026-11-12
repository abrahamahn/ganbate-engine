//! Import a model as a scene of separate objects, and cook a scene into a world that keeps its
//! entities (GBEW version 4).
//!
//! **Import** (`scene_from_model`): one entity per glTF node, with the node's name, parent and
//! local transform, a `mesh` component naming the model and the node's path, and a point
//! `light` where the node holds one. Nothing is merged, so every object can be selected and
//! moved.
//!
//! **Cook** (`cook_scene`): prefab instances are expanded, world transforms composed, and every
//! entity's static mesh becomes instances of the model's meshes at the entity's transform.
//! Each entity's instances and lights form one run, recorded in the entity table with its id,
//! name, parent and transform; the components the renderer does not draw go to the table's
//! data as canonical JSON. The renderer groups draws by material when it loads the world, so
//! keeping objects separate costs no draw calls.
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};

use ganbate_assets::cooked::MAX_INSTANCES;
use ganbate_assets::glb::{self, NodeWorld, SourceNode};
use ganbate_core::id::{Ids, is_entity_id};
use ganbate_core::json::{self, Value};
use ganbate_core::world::{
    EntityRecord, EntityTable, Instance, MAX_LIGHTS, PointLight, Transform, World,
};
use ganbate_core::{Vec3, digest};
use ganbate_scene::scene::{self, Kind as DocKind, Resolved};
use ganbate_scene::schema::Registry;

use crate::project::{Kind, Project};

/// A node's path in its model: names from the root joined by `/`; a node without a name is
/// `#index`; a path that repeats an earlier one gets `#n` (its n-th repeat, or the next number
/// whose path no other node has, since a name may itself end in `#n`). Every path is unique.
pub fn node_paths(nodes: &[SourceNode]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(nodes.len());
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut taken: HashSet<String> = HashSet::with_capacity(nodes.len());
    for (i, n) in nodes.iter().enumerate() {
        let own = if n.name.is_empty() {
            format!("#{i}")
        } else {
            n.name.replace('/', "_")
        };
        let base = match n.parent {
            Some(p) => format!("{}/{own}", out[p]),
            None => own,
        };
        let k = seen.entry(base.clone()).or_insert(0);
        let mut path = if *k == 0 {
            base.clone()
        } else {
            format!("{base}#{k}")
        };
        while taken.contains(&path) {
            *k += 1;
            path = format!("{base}#{k}");
        }
        *k += 1;
        taken.insert(path.clone());
        out.push(path);
    }
    out
}

fn num(n: f32) -> Value {
    Value::Number(f64::from(n))
}
fn nums(v: &[f32]) -> Value {
    Value::Array(v.iter().map(|x| num(*x)).collect())
}
fn s(v: &str) -> Value {
    Value::String(v.into())
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn unit(a: [f32; 3]) -> [f32; 3] {
    let n = dot(a, a).sqrt();
    a.map(|v| v / n)
}

/// The columns of a rotation whose axes follow `axes` (of lengths `len`) where they have a
/// direction: the longest axis, then the part of the next longest at right angles to it. The
/// shortest axis completes a right-handed frame, and an axis without a direction of its own
/// (scaled to nothing, or along another) gets one at right angles to the others.
fn frame(axes: &[[f32; 3]; 3], len: &[f32; 3]) -> [[f32; 3]; 3] {
    let mut order = [0, 1, 2];
    order.sort_by(|&i, &j| len[j].total_cmp(&len[i]).then(i.cmp(&j)));
    let [a, b, k] = order;
    let mut r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    // No axis has a length (or the matrix is not finite): no direction to follow.
    if len[a].is_nan() || len[a] <= 0.0 {
        return r;
    }
    r[a] = axes[a].map(|v| v / len[a]);
    // At right angles to r[a]: the rest of axis b, or the world axis least along r[a].
    let off = |v: [f32; 3]| {
        let along = dot(v, r[a]);
        [0, 1, 2].map(|i| v[i] - along * r[a][i])
    };
    let rest = if len[b] > 0.0 {
        off(axes[b].map(|v| v / len[b]))
    } else {
        [0.0; 3]
    };
    r[b] = if dot(rest, rest) > 1e-6 {
        unit(off(unit(rest)))
    } else {
        let m = (0..3)
            .min_by(|&i, &j| r[a][i].abs().total_cmp(&r[a][j].abs()))
            .unwrap_or(0);
        let mut e = [0.0; 3];
        e[m] = 1.0;
        unit(off(e))
    };
    // Each axis of a right-handed frame is the cross product of the next two.
    r[k] = cross(r[(k + 1) % 3], r[(k + 2) % 3]);
    r
}

/// Translation, rotation and scale of an affine matrix without shear, such that
/// `Transform::from_trs` gives the matrix back. A mirrored matrix keeps its handedness in the
/// scale's x. An axis scaled to (almost) nothing (`glb::FLAT_AXIS` of the longest) has no
/// direction of its own: the rotation builds it from the other axes, and its scale is what the
/// matrix holds along it.
pub fn decompose(m: &Transform) -> ([f32; 3], [f32; 4], [f32; 3]) {
    let c = m.0;
    let axes = [[c[0], c[1], c[2]], [c[4], c[5], c[6]], [c[8], c[9], c[10]]];
    let len = |x: f32, y: f32, z: f32| (x * x + y * y + z * z).sqrt();
    let mut sc = [
        len(c[0], c[1], c[2]),
        len(c[4], c[5], c[6]),
        len(c[8], c[9], c[10]),
    ];
    let longest = sc[0].max(sc[1]).max(sc[2]);
    let r = if longest > 0.0 && sc.iter().all(|&s| s > longest * glb::FLAT_AXIS) {
        // A mirrored matrix keeps its handedness in the scale's x.
        let det = c[0] * (c[5] * c[10] - c[9] * c[6]) - c[4] * (c[1] * c[10] - c[9] * c[2])
            + c[8] * (c[1] * c[6] - c[5] * c[2]);
        if det < 0.0 {
            sc[0] = -sc[0];
        }
        [0, 1, 2].map(|k| axes[k].map(|v| v / sc[k]))
    } else {
        let mut r = frame(&axes, &sc);
        sc = [0, 1, 2].map(|k| dot(axes[k], r[k]));
        // An axis pointing against its column is a mirror; keep it in x (turning both axes
        // half round leaves the rotation proper and the matrix the same).
        if let Some(k) = (1..3).find(|&k| sc[k] < 0.0) {
            for i in [0, k] {
                sc[i] = -sc[i];
                r[i] = r[i].map(|v| -v);
            }
        }
        r
    };
    // Rotation matrix (columns r[0..3]) to quaternion.
    let (m00, m11, m22) = (r[0][0], r[1][1], r[2][2]);
    let trace = m00 + m11 + m22;
    let q = if trace > 0.0 {
        let k = 0.5 / (trace + 1.0).sqrt();
        [
            (r[1][2] - r[2][1]) * k,
            (r[2][0] - r[0][2]) * k,
            (r[0][1] - r[1][0]) * k,
            0.25 / k,
        ]
    } else if m00 > m11 && m00 > m22 {
        let k = 2.0 * (1.0 + m00 - m11 - m22).sqrt();
        [
            0.25 * k,
            (r[1][0] + r[0][1]) / k,
            (r[2][0] + r[0][2]) / k,
            (r[1][2] - r[2][1]) / k,
        ]
    } else if m11 > m22 {
        let k = 2.0 * (1.0 + m11 - m00 - m22).sqrt();
        [
            (r[1][0] + r[0][1]) / k,
            0.25 * k,
            (r[2][1] + r[1][2]) / k,
            (r[2][0] - r[0][2]) / k,
        ]
    } else {
        let k = 2.0 * (1.0 + m22 - m00 - m11).sqrt();
        [
            (r[2][0] + r[0][2]) / k,
            (r[2][1] + r[1][2]) / k,
            0.25 * k,
            (r[0][1] - r[1][0]) / k,
        ]
    };
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let q = if n > 0.0 && n.is_finite() {
        q.map(|v| v / n)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
    // One sign for each rotation, so files do not flip between equivalent spellings.
    let q = if q[3] < 0.0 { q.map(|v| -v) } else { q };
    ([c[12], c[13], c[14]], q, sc)
}

/// A sparse `transform` component: only what differs from the defaults.
pub fn transform_component(m: &Transform) -> Value {
    let (t, q, sc) = decompose(m);
    let round = |v: f32| if v.abs() < 1e-7 { 0.0 } else { v };
    let (t, q, sc) = (t.map(round), q.map(round), sc.map(round));
    let mut c = BTreeMap::new();
    if t != [0.0; 3] {
        c.insert("translation".to_string(), nums(&t));
    }
    if q != [0.0, 0.0, 0.0, 1.0] {
        c.insert("rotation".into(), nums(&q));
    }
    if sc != [1.0; 3] {
        c.insert("scale".into(), nums(&sc));
    }
    Value::Object(c)
}

/// A scene with one entity per node of a static model (see the module notes).
pub fn scene_from_model(
    model: &NodeWorld,
    model_asset: &str,
    scene_id: &str,
    name: &str,
    ids: &mut Ids,
) -> Value {
    let mut doc = scene::new_document(DocKind::Scene, scene_id, name);
    let paths = node_paths(&model.nodes);
    let entity_ids: Vec<String> = model.nodes.iter().map(|_| ids.entity()).collect();
    // Order keys spread evenly among each node's children (and among the roots).
    let mut by_parent: BTreeMap<Option<usize>, Vec<usize>> = BTreeMap::new();
    for (i, n) in model.nodes.iter().enumerate() {
        by_parent.entry(n.parent).or_default().push(i);
    }
    let mut orders = vec![String::new(); model.nodes.len()];
    for kids in by_parent.values() {
        let keys = ganbate_scene::order::spread(None, None, kids.len()).expect("keys");
        for (k, i) in keys.into_iter().zip(kids) {
            orders[*i] = k;
        }
    }
    let mut entities = BTreeMap::new();
    for (i, n) in model.nodes.iter().enumerate() {
        let mut comps = BTreeMap::new();
        let t = transform_component(&n.local);
        if !matches!(&t, Value::Object(m) if m.is_empty()) {
            comps.insert("transform".to_string(), t);
        }
        if !n.meshes.is_empty() {
            comps.insert(
                "mesh".into(),
                Value::Object(BTreeMap::from([
                    ("model".to_string(), s(model_asset)),
                    ("node".into(), s(&paths[i])),
                ])),
            );
        }
        if let Some(l) = &n.light {
            let mut light = BTreeMap::from([
                ("color".to_string(), nums(&l.color)),
                ("intensity".into(), num(l.intensity)),
            ]);
            light.insert("range".into(), num(l.range));
            // A model's lights are lamps: lit as the evening lights the world's lamps.
            light.insert("follows_lamps".into(), Value::Bool(true));
            comps.insert("light".into(), Value::Object(light));
        }
        let mut e = BTreeMap::from([
            (
                "name".to_string(),
                s(if n.name.is_empty() {
                    &paths[i]
                } else {
                    &n.name
                }),
            ),
            ("order".into(), s(&orders[i])),
        ]);
        if let Some(p) = n.parent {
            e.insert("parent".into(), s(&entity_ids[p]));
        }
        if !comps.is_empty() {
            e.insert("components".into(), Value::Object(comps));
        }
        entities.insert(entity_ids[i].clone(), Value::Object(e));
    }
    if let Value::Object(m) = &mut doc {
        m.insert("entities".into(), Value::Object(entities));
    }
    doc
}

/// A cooked scene: its world, its entity table and anything worth a warning.
#[derive(Debug)]
pub struct Cooked {
    pub world: World,
    pub table: EntityTable,
    pub warnings: Vec<String>,
}

/// Cached model metadata after its textures, materials and meshes have been moved into the
/// cooked [`World`].
struct CachedModel {
    nodes: Vec<SourceNode>,
    path_to_idx: HashMap<String, usize>,
    model_space: Vec<Transform>,
    mesh0: u32,
}

/// Models already imported, with their meshes appended to the world being built.
struct Library<'l> {
    import_model: &'l mut dyn FnMut(&str) -> Result<NodeWorld, String>,
    models: HashMap<String, CachedModel>,
}

impl Library<'_> {
    /// The model's nodes, path index and model-space transforms, importing it (and moving its
    /// optimized meshes, materials and textures into `w`) the first time.
    fn model(&mut self, w: &mut World, id: &str) -> Result<&CachedModel, String> {
        if !self.models.contains_key(id) {
            let nw = (self.import_model)(id)?;
            let (mesh0, mat0, tex0) = (
                w.meshes.len() as u32,
                w.materials.len() as u32,
                w.textures.len() as u32,
            );
            w.textures.extend(nw.world.textures);
            w.materials.reserve(nw.world.materials.len());
            for mut m in nw.world.materials {
                for t in [
                    &mut m.base_color_texture,
                    &mut m.normal_texture,
                    &mut m.emissive_texture,
                ] {
                    *t = t.map(|i| i + tex0);
                }
                w.materials.push(m);
            }
            w.meshes.reserve(nw.world.meshes.len());
            for mut mesh in nw.world.meshes {
                mesh.material = mesh.material.map(|i| i + mat0);
                w.meshes.push(mesh);
            }
            let paths = node_paths(&nw.nodes);
            let mut path_to_idx = HashMap::with_capacity(paths.len());
            for (i, p) in paths.into_iter().enumerate() {
                path_to_idx.insert(p, i);
            }
            let mut model_space: Vec<Transform> = Vec::with_capacity(nw.nodes.len());
            for n in &nw.nodes {
                let m = match n.parent {
                    Some(p) => model_space[p].compose(n.local),
                    None => n.local,
                };
                model_space.push(m);
            }
            self.models.insert(
                id.to_string(),
                CachedModel {
                    nodes: nw.nodes,
                    path_to_idx,
                    model_space,
                    mesh0,
                },
            );
        }
        Ok(&self.models[id])
    }
}

fn field<'a>(c: &'a Value, k: &str) -> Option<&'a Value> {
    c.get(k)
}
fn floats3(v: Option<&Value>, default: [f32; 3]) -> [f32; 3] {
    match v {
        Some(Value::Array(a)) if a.len() == 3 => {
            let mut out = default;
            for (o, x) in out.iter_mut().zip(a) {
                if let Value::Number(n) = x {
                    *o = *n as f32;
                }
            }
            out
        }
        _ => default,
    }
}
fn float(v: Option<&Value>, default: f32) -> f32 {
    match v {
        Some(Value::Number(n)) => *n as f32,
        _ => default,
    }
}

/// Cooks scene `scene_id` of `project` (see the module notes). `keep_images` keeps textures in
/// their file format for browsers.
pub fn cook_scene(
    project: &Project,
    registry: &Registry,
    scene_id: &str,
    keep_images: bool,
) -> Result<Cooked, String> {
    let doc = project.read_document(scene_id)?;
    let issues = scene::check(&doc, registry);
    if let Some(i) = issues.first() {
        return Err(format!(
            "{} problem(s) in the scene; first: {}: {}",
            issues.len(),
            i.path,
            i.message
        ));
    }
    let load = |id: &str| match project.entry(id) {
        Some(e) if e.kind == Kind::Prefab => project.read_document(id),
        Some(e) => Err(format!(
            "{id} ({}) is a {}, not a prefab",
            e.path,
            e.kind.name()
        )),
        None => Err(format!("missing prefab {id}")),
    };
    let resolved = flatten_checked(&doc, registry, &load)?;
    cook_resolved(project, registry, &doc, &resolved, keep_images)
}

/// Expands `doc`'s prefab instances as `scene::flatten` does, reading prefabs with `load` and
/// refusing any prefab in which `scene::check` finds a problem (a parent cycle among its
/// entities, a second root, a component that breaks its schema), with an error naming it.
/// Each prefab is read and checked once. Editors that cook open documents pass their own
/// `load`.
pub fn flatten_checked(
    doc: &Value,
    registry: &Registry,
    load: &dyn Fn(&str) -> Result<Value, String>,
) -> Result<Vec<Resolved>, String> {
    let checked: RefCell<BTreeMap<String, Value>> = RefCell::default();
    let refused: RefCell<Option<String>> = RefCell::default();
    let guarded = |id: &str| -> Option<Value> {
        if let Some(prefab) = checked.borrow().get(id) {
            return Some(prefab.clone());
        }
        let prefab = load(id).and_then(|prefab| {
            let issues = scene::check(&prefab, registry);
            match issues.first() {
                None => Ok(prefab),
                Some(i) => Err(format!(
                    "prefab {id} {:?}: {} problem(s); first: {}: {}",
                    prefab
                        .get("name")
                        .and_then(|n| n.string().ok())
                        .unwrap_or(""),
                    issues.len(),
                    i.path,
                    i.message
                )),
            }
        });
        match prefab {
            Ok(prefab) => {
                checked.borrow_mut().insert(id.to_string(), prefab.clone());
                Some(prefab)
            }
            Err(e) => {
                refused.borrow_mut().get_or_insert(e);
                None
            }
        }
    };
    let resolved = scene::flatten(doc, &guarded);
    // A refused prefab fails the expansion, however `flatten` treats the prefab it was not given.
    match refused.into_inner() {
        Some(e) => Err(e),
        None => resolved,
    }
}

/// Cooks already-resolved entities (the editor cooks open, unsaved documents this way), backed by
/// `<project>/.ganbate/cache/ddc/` and each model's `.meta.json` import settings.
pub fn cook_resolved(
    project: &Project,
    registry: &Registry,
    doc: &Value,
    resolved: &[Resolved],
    keep_images: bool,
) -> Result<Cooked, String> {
    let mut import_fn = |id: &str| {
        let path = project
            .path_of(id)
            .ok_or_else(|| format!("missing model {id}"))?;
        let bytes = ganbate_assets::read_asset(&path)?;
        let settings = project.model_import_settings(id);
        let label = path.display().to_string();
        let (nw, _hit) =
            crate::ddc::load_or_cook_disk(&project.root, &label, &bytes, &settings, keep_images)?;
        Ok(nw)
    };
    cook_resolved_internal(registry, doc, resolved, &mut import_fn)
}

/// Callback fetching `(label, glb_bytes)` for a model asset id.
pub type LoadModelFn<'a> = &'a dyn Fn(&str) -> Result<(String, Vec<u8>), String>;

/// Callback fetching `(label, glb_bytes, import_settings)` for a model asset id.
pub type LoadModelWithSettingsFn<'a> =
    &'a dyn Fn(&str) -> Result<(String, Vec<u8>, crate::ddc::ModelImportSettings), String>;

/// Cooks already-resolved entities using `load_model` to fetch each referenced model's `(label, bytes)`.
/// Used by both disk-backed hosts and the pure-web Wasm Studio host.
pub fn cook_resolved_with(
    registry: &Registry,
    doc: &Value,
    resolved: &[Resolved],
    keep_images: bool,
    load_model: LoadModelFn<'_>,
) -> Result<Cooked, String> {
    let default_settings = crate::ddc::ModelImportSettings::default();
    let mut import_fn = |id: &str| {
        let (label, bytes) = (load_model)(id)?;
        let mut nw = glb::import_nodes(&bytes, keep_images).map_err(|e| format!("{label}: {e}"))?;
        crate::ddc::prepare_node_world(&mut nw, &default_settings);
        Ok(nw)
    };
    cook_resolved_internal(registry, doc, resolved, &mut import_fn)
}

/// Pure in-memory / WASM cooker backed by [`crate::ddc::MemoryDdc`] and per-model
/// [`crate::ddc::ModelImportSettings`] so browser Studio sessions cook unchanged models in `< 1 ms`
/// without `std::fs`.
pub fn cook_resolved_with_ddc(
    registry: &Registry,
    doc: &Value,
    resolved: &[Resolved],
    keep_images: bool,
    ddc: &mut crate::ddc::MemoryDdc,
    load_model: LoadModelWithSettingsFn<'_>,
) -> Result<Cooked, String> {
    let mut import_fn = |id: &str| {
        let (label, bytes, settings) = (load_model)(id)?;
        ddc.load_or_cook(&label, &bytes, &settings, keep_images)
    };
    cook_resolved_internal(registry, doc, resolved, &mut import_fn)
}

fn cook_resolved_internal(
    registry: &Registry,
    doc: &Value,
    resolved: &[Resolved],
    import_model: &mut dyn FnMut(&str) -> Result<NodeWorld, String>,
) -> Result<Cooked, String> {
    let transforms = scene::world_transforms(resolved);
    let mut w = World::default();
    let mut lib = Library {
        import_model,
        models: HashMap::new(),
    };
    let mut warnings = vec![];
    let mut index_of: HashMap<&str, u32> = HashMap::with_capacity(resolved.len());
    let mut records = Vec::with_capacity(resolved.len());
    let mut data = BTreeMap::new();
    for (r, &world_m) in resolved.iter().zip(&transforms) {
        if !is_entity_id(&r.id) {
            return Err(format!("{}: invalid entity id", r.id));
        }
        let first = w.instances.len() as u32;
        let first_light = w.lights.len() as u32;
        let mut rest = BTreeMap::new();
        for (kind, c) in &r.components {
            match kind.as_str() {
                "transform" => {}
                "mesh" => {
                    let is_static = registry
                        .field("mesh", c, "static")
                        .is_none_or(|v| *v == Value::Bool(true));
                    let model = field(c, "model").and_then(|v| v.string().ok());
                    let Some(model) = model.filter(|_| is_static) else {
                        // Dynamic meshes and empty slots are the runtime's to place.
                        rest.insert(kind.clone(), c.clone());
                        continue;
                    };
                    let node = field(c, "node").and_then(|v| v.string().ok()).unwrap_or("");
                    let cached = lib.model(&mut w, model)?;
                    let offset = cached.mesh0;
                    if node.is_empty() {
                        // The whole model under the entity, its node tree intact.
                        for (n, &ms) in cached.nodes.iter().zip(&cached.model_space) {
                            // The world's limits, checked as it grows: a large model placed many
                            // times fails here instead of exhausting memory.
                            if w.instances.len() + n.meshes.len() > MAX_INSTANCES {
                                return Err(format!(
                                    "{} ({}): the scene places more than {MAX_INSTANCES} mesh instances",
                                    r.name, r.id
                                ));
                            }
                            if n.light.is_some() && w.lights.len() >= MAX_LIGHTS {
                                return Err(format!(
                                    "{} ({}): the scene has more than {MAX_LIGHTS} lamps",
                                    r.name, r.id
                                ));
                            }
                            if n.meshes.is_empty() && n.light.is_none() {
                                continue;
                            }
                            let m = world_m.compose(ms);
                            for mesh in n.meshes.clone() {
                                w.instances.push(Instance {
                                    mesh: mesh + offset,
                                    transform: m,
                                });
                            }
                            if let Some(l) = n.light {
                                w.lights.push(PointLight {
                                    position: m.point(Vec3(0.0, 0.0, 0.0)),
                                    ..l
                                });
                            }
                        }
                    } else if let Some(&i) = cached.path_to_idx.get(node) {
                        let n = &cached.nodes[i];
                        if w.instances.len() + n.meshes.len() > MAX_INSTANCES {
                            return Err(format!(
                                "{} ({}): the scene places more than {MAX_INSTANCES} mesh instances",
                                r.name, r.id
                            ));
                        }
                        for mesh in n.meshes.clone() {
                            w.instances.push(Instance {
                                mesh: mesh + offset,
                                transform: world_m,
                            });
                        }
                    } else {
                        warnings.push(format!(
                            "{} ({}): model {model} has no node {node:?}",
                            r.name, r.id
                        ));
                    }
                }
                "light"
                    if registry
                        .field("light", c, "kind")
                        .is_none_or(|k| k.string().ok() == Some("point")) =>
                {
                    let follows = registry
                        .field("light", c, "follows_lamps")
                        .is_some_and(|v| *v == Value::Bool(true));
                    if !follows {
                        // A cooked lamp follows the hour; a light that always shines is a game
                        // light, which the runtime hangs from the table's data.
                        rest.insert(kind.clone(), c.clone());
                        continue;
                    }
                    if w.lights.len() >= MAX_LIGHTS {
                        return Err(format!(
                            "{} ({}): the scene has more than {MAX_LIGHTS} lamps",
                            r.name, r.id
                        ));
                    }
                    w.lights.push(PointLight {
                        position: world_m.point(Vec3(0.0, 0.0, 0.0)),
                        color: floats3(field(c, "color"), [1.0; 3]),
                        intensity: float(field(c, "intensity"), 1.0),
                        range: float(field(c, "range"), 6.0),
                    });
                }
                _ => {
                    rest.insert(kind.clone(), c.clone());
                }
            }
        }
        let parent = r.parent.as_deref().and_then(|p| index_of.get(p).copied());
        index_of.insert(&r.id, records.len() as u32);
        records.push(EntityRecord {
            id: u64::from_str_radix(&r.id, 16).map_err(|e| e.to_string())?,
            name: r.name.clone(),
            parent,
            transform: world_m,
            instances: first..w.instances.len() as u32,
            lights: first_light..w.lights.len() as u32,
        });
        if !rest.is_empty() {
            data.insert(
                r.id.clone(),
                Value::Object(BTreeMap::from([(
                    "components".to_string(),
                    Value::Object(rest),
                )])),
            );
        }
    }
    let mut root = BTreeMap::from([
        ("entities".to_string(), Value::Object(data)),
        (
            "scene".into(),
            doc.get("id").cloned().unwrap_or(Value::Null),
        ),
    ]);
    if let Some(n) = doc.get("name") {
        root.insert("name".into(), n.clone());
    }
    let table = EntityTable {
        entities: records,
        data: json::to_canonical(&Value::Object(root))?,
    };
    w.validate()?;
    table.validate(&w)?;
    Ok(Cooked {
        world: w,
        table,
        warnings,
    })
}

/// A file name that changes whenever the bytes do: `<stem>.<first 16 hex of sha256>.<ext>`.
pub fn content_name(stem: &str, ext: &str, bytes: &[u8]) -> String {
    let h = digest::hex(&digest::sha256(bytes));
    format!("{stem}.{}.{ext}", &h[..16])
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::project::tests::TempDir;
    use crate::project::{Kind, new_sidecar, write_json};
    use ganbate_scene::doc::{Document, History};
    use ganbate_scene::schema::engine_registry;

    /// The start of a glTF document whose buffer holds one triangle (accessors 0 and 1).
    const TRIANGLE: &str = r#"{"asset":{"version":"2.0"},"buffers":[{"byteLength":54}],"bufferViews":[{"buffer":0,"byteLength":48,"byteStride":16},{"buffer":0,"byteOffset":48,"byteLength":6}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],"#;
    /// A mesh of that triangle.
    const MESH: &str = r#"{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}"#;

    /// A GLB of glTF document `doc` with the triangle's buffer.
    fn glb_of(doc: &str) -> Vec<u8> {
        let mut j = doc.as_bytes().to_vec();
        while j.len() % 4 != 0 {
            j.push(b' ');
        }
        let mut b = Vec::new();
        for v in [[0f32, 0., 0., 9.], [1., 0., 0., 9.], [0., 1., 0., 9.]] {
            for f in v {
                b.extend_from_slice(&f.to_le_bytes());
            }
        }
        for i in [0u16, 1, 2] {
            b.extend_from_slice(&i.to_le_bytes());
        }
        while b.len() % 4 != 0 {
            b.push(0);
        }
        let mut out = b"glTF".to_vec();
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(28 + j.len() as u32 + b.len() as u32).to_le_bytes());
        for (id, chunk) in [(0x4e4f534au32, j), (0x004e4942, b)] {
            out.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&chunk);
        }
        out
    }

    /// A GLB with two nodes using one triangle mesh, a child, and a lamp node.
    pub(crate) fn sample_glb() -> Vec<u8> {
        glb_of(
            r#"{"asset":{"version":"2.0"},"buffers":[{"byteLength":54}],"bufferViews":[{"buffer":0,"byteLength":48,"byteStride":16},{"buffer":0,"byteOffset":48,"byteLength":6}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],"materials":[{"name":"stone","pbrMetallicRoughness":{"baseColorFactor":[0.5,0.5,0.5,1]}}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1,"material":0}]}],"extensions":{"KHR_lights_punctual":{"lights":[{"type":"point","intensity":3,"range":5}]}},"nodes":[{"name":"Tree","mesh":0,"translation":[1,0,0],"children":[1]},{"name":"Crown","mesh":0,"translation":[0,2,0]},{"name":"Tree","mesh":0,"translation":[5,0,0]},{"name":"Lamp","translation":[0,3,0],"extensions":{"KHR_lights_punctual":{"light":0}}}],"scenes":[{"nodes":[0,2,3]}],"scene":0}"#,
        )
    }

    #[test]
    fn paths_name_every_node_once() {
        let nw = glb::import_nodes(&sample_glb(), false).unwrap();
        assert_eq!(
            node_paths(&nw.nodes),
            ["Tree", "Tree/Crown", "Tree#1", "Lamp"]
        );
    }

    #[test]
    fn paths_stay_unique_when_names_look_like_repeats() {
        let node = |name: &str, parent: Option<usize>| SourceNode {
            name: name.into(),
            parent,
            local: Transform::IDENTITY,
            meshes: 0..0,
            light: None,
        };
        let nodes = [
            node("A", None),
            node("A", None),
            node("A#1", None),
            node("B#1", None),
            node("B", None),
            node("B", None),
            node("", None),
            node("#6", None),
            node("C", Some(0)),
            node("C", Some(1)),
            node("C", Some(2)),
        ];
        let paths = node_paths(&nodes);
        assert_eq!(
            paths,
            [
                "A", "A#1", "A#1#1", "B#1", "B", "B#2", "#6", "#6#1", "A/C", "A#1/C", "A#1#1/C"
            ]
        );
    }

    #[test]
    fn nodes_named_like_repeats_cook_their_own_meshes() {
        // The third node's own name is what the second node's repeat is called.
        let glb = glb_of(&format!(
            r#"{TRIANGLE}"meshes":[{MESH},{MESH},{MESH}],"nodes":[{{"name":"A","mesh":0}},{{"name":"A","mesh":1,"translation":[5,0,0]}},{{"name":"A#1","mesh":2,"translation":[9,0,0]}}],"scenes":[{{"nodes":[0,1,2]}}],"scene":0}}"#
        ));
        let t = TempDir::new("repeats");
        let (mut p, model, mut ids) = project_with(&t, &glb);
        let nw = glb::import_nodes(&glb, false).unwrap();
        let scene_id = ids.asset();
        let doc = scene_from_model(&nw, &model, &scene_id, "Repeats", &mut ids);
        p.write_document("content/repeats.scene.json", Kind::Scene, &doc)
            .unwrap();
        let cooked = cook_scene(&p, &engine_registry(), &scene_id, false).unwrap();
        assert!(cooked.warnings.is_empty(), "{:?}", cooked.warnings);
        // (mesh, x) of each instance, as the flattening import places them.
        let placed = |w: &World| {
            let mut v: Vec<(u32, i64)> = w
                .instances
                .iter()
                .map(|i| (i.mesh, (i.transform.0[12] * 1000.0).round() as i64))
                .collect();
            v.sort();
            v
        };
        assert_eq!(
            placed(&glb::import(&glb).unwrap()),
            [(0, 0), (1, 5000), (2, 9000)]
        );
        assert_eq!(placed(&cooked.world), [(0, 0), (1, 5000), (2, 9000)]);
    }

    #[test]
    fn decomposition_recovers_trs() {
        let q = {
            let v = [0.1f32, 0.7, -0.2, 0.6];
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.map(|x| x / n)
        };
        let m = Transform::from_trs([1.0, 2.0, 3.0], q, [2.0, 3.0, 0.5]).unwrap();
        let (t, q2, sc) = decompose(&m);
        assert_eq!(t, [1.0, 2.0, 3.0]);
        for (a, b) in q.iter().zip(q2) {
            assert!((a - b).abs() < 1e-5, "{q:?} {q2:?}");
        }
        for (a, b) in [2.0f32, 3.0, 0.5].iter().zip(sc) {
            assert!((a - b).abs() < 1e-5);
        }
        assert_eq!(
            json::to_compact(&transform_component(&Transform::IDENTITY)).unwrap(),
            "{}"
        );
    }

    #[test]
    fn decomposition_rebuilds_axes_scaled_to_nothing() {
        let unit = |v: [f32; 4]| {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.map(|x| x / n)
        };
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let rotations = [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, h, 0.0, h],
            unit([0.1, 0.7, -0.2, 0.6]),
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let scales = [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, -2.0, 0.0],
            [0.0, -1.0, 1.0],
            [-2.0, 0.0, 0.0],
            [1.0, 1.0, 1e-9],
            [1.0, -1.0, 1e-9],
            [1e7, 1.0, 1.0],
            [2.0, 3.0, 0.5],
            [-1.0, 1.0, 1.0],
        ];
        for q in rotations {
            for s in scales {
                let m = Transform::from_trs([1.0, -2.0, 3.0], q, s).unwrap();
                let (t, q2, s2) = decompose(&m);
                let back = Transform::from_trs(t, q2, s2).unwrap();
                // Each axis comes back within rounding of its own length.
                for k in 0..4 {
                    let (a, b) = (&m.0[4 * k..4 * k + 3], &back.0[4 * k..4 * k + 3]);
                    let size = a.iter().map(|v| v * v).sum::<f32>().sqrt().max(1.0);
                    let err = a
                        .iter()
                        .zip(b)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0, f32::max);
                    assert!(
                        err <= 1e-5 * size,
                        "rotation {q:?} scale {s:?}: axis {k} {a:?} came back as {b:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_node_scaled_flat_cooks_where_the_flat_import_puts_it() {
        // A quarter turn about y, flattened along z (glTF TRS), and the same as a matrix.
        for node in [
            r#""rotation":[0,0.7071068,0,0.7071068],"scale":[1,1,0]"#,
            r#""matrix":[0,0,-1,0,0,1,0,0,0,0,0,0,4,0,0,1]"#,
        ] {
            let glb = glb_of(&format!(
                r#"{TRIANGLE}"meshes":[{MESH}],"nodes":[{{"name":"Flat","mesh":0,{node}}}],"scenes":[{{"nodes":[0]}}],"scene":0}}"#
            ));
            let t = TempDir::new("flat");
            let (mut p, model, mut ids) = project_with(&t, &glb);
            let nw = glb::import_nodes(&glb, false).unwrap();
            let scene_id = ids.asset();
            let doc = scene_from_model(&nw, &model, &scene_id, "Flat", &mut ids);
            p.write_document("content/flat.scene.json", Kind::Scene, &doc)
                .unwrap();
            let cooked = cook_scene(&p, &engine_registry(), &scene_id, false).unwrap();
            let flat = glb::import(&glb).unwrap();
            let (a, b) = (
                cooked.world.instances[0].transform.0,
                flat.instances[0].transform.0,
            );
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-6, "{node}: cooked {a:?}, imported {b:?}");
            }
        }
    }

    fn project_with_model(t: &TempDir) -> (Project, String, Ids) {
        project_with(t, &sample_glb())
    }

    /// A new project in `t` holding model `glb`; returns the project, the model's id and ids.
    fn project_with(t: &TempDir, glb: &[u8]) -> (Project, String, Ids) {
        let mut ids = Ids::seeded(11);
        let mut p = Project::create(&t.0, "Cook", &mut ids).unwrap();
        std::fs::create_dir_all(t.0.join("content/models")).unwrap();
        std::fs::write(t.0.join("content/models/grove.glb"), glb).unwrap();
        let model = ids.asset();
        write_json(
            &t.0.join("content/models/grove.glb.meta.json"),
            &new_sidecar(&model, Kind::Model),
        )
        .unwrap();
        p.scan(None).unwrap();
        (p, model, ids)
    }

    #[test]
    fn an_imported_model_cooks_to_the_same_world_with_entities() {
        let t = TempDir::new("cook");
        let (mut p, model, mut ids) = project_with_model(&t);
        let reg = engine_registry();
        let nw = glb::import_nodes(&sample_glb(), false).unwrap();
        let scene_id = ids.asset();
        let doc = scene_from_model(&nw, &model, &scene_id, "Grove", &mut ids);
        assert_eq!(scene::check(&doc, &reg), vec![]);
        p.write_document("content/grove.scene.json", Kind::Scene, &doc)
            .unwrap();
        let cooked = cook_scene(&p, &reg, &scene_id, false).unwrap();
        assert!(cooked.warnings.is_empty(), "{:?}", cooked.warnings);
        // The same instances and lamp as cooking the GLB directly.
        let flat = glb::import(&sample_glb()).unwrap();
        let place = |w: &World| {
            let mut v: Vec<[i64; 3]> = w
                .instances
                .iter()
                .map(|i| [12, 13, 14].map(|k| (i.transform.0[k] * 1000.0).round() as i64))
                .collect();
            v.sort();
            v
        };
        assert_eq!(place(&cooked.world), place(&flat));
        assert_eq!(cooked.world.lights, flat.lights);
        // One record per node, each with its own run.
        let names: Vec<&str> = cooked
            .table
            .entities
            .iter()
            .map(|e| e.name.as_str())
            .collect();
        assert_eq!(names.len(), 4);
        let crown = cooked
            .table
            .entities
            .iter()
            .find(|e| e.name == "Crown")
            .unwrap();
        let parent = &cooked.table.entities[crown.parent.unwrap() as usize];
        assert_eq!(parent.name, "Tree");
        assert_eq!(crown.instances.len(), 1);
        assert_eq!(crown.transform.0[12..15], [1.0, 2.0, 0.0]);
        // The cooked bytes are deterministic and decode to the same table.
        let a = ganbate_assets::cooked::encode_scene(&cooked.world, &cooked.table, false).unwrap();
        let again = cook_scene(&p, &reg, &scene_id, false).unwrap();
        let b = ganbate_assets::cooked::encode_scene(&again.world, &again.table, false).unwrap();
        assert_eq!(a, b);
        let (_, table) = ganbate_assets::cooked::decode_scene(&a).unwrap();
        assert_eq!(table, cooked.table);
    }

    #[test]
    fn moving_an_entity_moves_its_children_in_the_cook() {
        let t = TempDir::new("move");
        let (mut p, model, mut ids) = project_with_model(&t);
        let reg = engine_registry();
        let nw = glb::import_nodes(&sample_glb(), false).unwrap();
        let scene_id = ids.asset();
        let mut doc =
            Document::new(scene_from_model(&nw, &model, &scene_id, "G", &mut ids)).unwrap();
        let tree = scene::entities(doc.root())
            .iter()
            .find(|(_, e)| {
                e.get("name") == Some(&Value::String("Tree".into())) && e.get("parent").is_none()
            })
            .map(|(id, _)| id.clone())
            .unwrap();
        let mut h = History::new(8);
        h.edit(
            &mut doc,
            "Move",
            None,
            vec![scene::set_field(
                &tree,
                "transform",
                &["translation"],
                nums(&[10.0, 0.0, 0.0]),
            )],
        )
        .unwrap();
        p.write_document("content/g.scene.json", Kind::Scene, doc.root())
            .unwrap();
        let cooked = cook_scene(&p, &reg, &scene_id, false).unwrap();
        let crown = cooked
            .table
            .entities
            .iter()
            .find(|e| e.name == "Crown")
            .unwrap();
        let inst = &cooked.world.instances[crown.instances.start as usize];
        assert_eq!(inst.transform.0[12..15], [10.0, 2.0, 0.0]);
    }

    #[test]
    fn game_components_and_dynamic_meshes_go_to_the_data() {
        let t = TempDir::new("data");
        let (mut p, model, mut ids) = project_with_model(&t);
        let mut reg = engine_registry();
        reg.add(
            &json::parse(br#"{"ganbate":"schema","version":1,"components":{"arena.spawn":{"fields":{"team":{"type":"enum","values":["red","blue"]}}}}}"#).unwrap(),
            false,
        )
        .unwrap();
        let scene_id = ids.asset();
        let mut doc = scene::new_document(DocKind::Scene, &scene_id, "Data");
        let e1 = ids.entity();
        let e2 = ids.entity();
        let mut d = Document::new(doc.clone()).unwrap();
        d.apply_all(&scene::create(
            &e1,
            "Spawn",
            None,
            None,
            BTreeMap::from([(
                "arena.spawn".to_string(),
                json::parse(br#"{"team":"blue"}"#).unwrap(),
            )]),
        ))
        .unwrap();
        d.apply_all(&scene::create(
            &e2,
            "Moving tree",
            None,
            None,
            BTreeMap::from([(
                "mesh".to_string(),
                json::parse(format!(r#"{{"model":"{model}","static":false}}"#).as_bytes()).unwrap(),
            )]),
        ))
        .unwrap();
        doc = d.root().clone();
        p.write_document("content/d.scene.json", Kind::Scene, &doc)
            .unwrap();
        let cooked = cook_scene(&p, &reg, &scene_id, false).unwrap();
        assert!(cooked.world.instances.is_empty());
        let data = json::parse(cooked.table.data.as_bytes()).unwrap();
        let ents = data.get("entities").unwrap();
        assert!(
            ents.get(&e1)
                .unwrap()
                .get("components")
                .unwrap()
                .get("arena.spawn")
                .is_some()
        );
        assert!(
            ents.get(&e2)
                .unwrap()
                .get("components")
                .unwrap()
                .get("mesh")
                .is_some()
        );
    }

    /// A scene with one instance of `prefab` (its entities as JSON), written to the project
    /// with the prefab; returns the scene's id and the prefab's.
    fn scene_with_prefab(p: &mut Project, ids: &mut Ids, prefab: &str) -> (String, String) {
        let (scene_id, prefab_id) = (ids.asset(), ids.asset());
        let prefab = json::parse(
            format!(r#"{{"ganbate":"prefab","version":1,"id":"{prefab_id}","name":"Loop","entities":{prefab}}}"#)
                .as_bytes(),
        )
        .unwrap();
        p.write_document("content/loop.prefab.json", Kind::Prefab, &prefab)
            .unwrap();
        let scene = json::parse(
            format!(r#"{{"ganbate":"scene","version":1,"id":"{scene_id}","name":"S","entities":{{"00000000000000b1":{{"name":"Instance","prefab":{{"asset":"{prefab_id}"}}}}}}}}"#)
                .as_bytes(),
        )
        .unwrap();
        assert_eq!(scene::check(&scene, &engine_registry()), vec![]);
        p.write_document("content/s.scene.json", Kind::Scene, &scene)
            .unwrap();
        (scene_id, prefab_id)
    }

    #[test]
    fn a_prefab_with_a_parent_cycle_fails_the_cook_naming_it() {
        let t = TempDir::new("cycle");
        let (mut p, _, mut ids) = project_with_model(&t);
        // Two entities under each other, neither reachable from the root.
        let (scene_id, prefab_id) = scene_with_prefab(
            &mut p,
            &mut ids,
            r#"{"00000000000000a0":{"name":"R"},"00000000000000a1":{"name":"A","parent":"00000000000000a2"},"00000000000000a2":{"name":"B","parent":"00000000000000a1"}}"#,
        );
        let err = cook_scene(&p, &engine_registry(), &scene_id, false).unwrap_err();
        assert!(
            err.contains(&prefab_id) && err.contains("Loop") && err.contains("parent cycle"),
            "{err}"
        );
    }

    #[test]
    fn checked_expansion_reads_each_prefab_once() {
        let pid = "fedcba9876543210fedcba9876543210";
        let prefab = json::parse(
            format!(r#"{{"ganbate":"prefab","version":1,"id":"{pid}","name":"Post","entities":{{"00000000000000a0":{{"name":"Post"}},"00000000000000a1":{{"name":"Lamp","parent":"00000000000000a0"}}}}}}"#)
                .as_bytes(),
        )
        .unwrap();
        let instance =
            |e: &str| format!(r#""{e}":{{"name":"Post {e}","prefab":{{"asset":"{pid}"}}}}"#);
        let scene = json::parse(
            format!(
                r#"{{"ganbate":"scene","version":1,"id":"0123456789abcdef0123456789abcdef","name":"S","entities":{{{},{},{}}}}}"#,
                instance("00000000000000b1"),
                instance("00000000000000b2"),
                instance("00000000000000b3")
            )
            .as_bytes(),
        )
        .unwrap();
        let reads = std::cell::Cell::new(0);
        let load = |id: &str| {
            reads.set(reads.get() + 1);
            (id == pid)
                .then(|| prefab.clone())
                .ok_or_else(|| format!("no {id}"))
        };
        let flat = flatten_checked(&scene, &engine_registry(), &load).unwrap();
        assert_eq!(flat.len(), 6);
        assert_eq!(reads.get(), 1);
    }

    #[test]
    fn prefabs_are_checked_like_scenes_before_they_cook() {
        let t = TempDir::new("prefab-check");
        let (mut p, _, mut ids) = project_with_model(&t);
        let reg = engine_registry();
        let (scene_id, prefab_id) = scene_with_prefab(
            &mut p,
            &mut ids,
            r#"{"00000000000000a0":{"name":"R","components":{"transform":{"scale":[1,1]}}}}"#,
        );
        let err = cook_scene(&p, &reg, &scene_id, false).unwrap_err();
        assert!(
            err.contains(&prefab_id) && err.contains("expected 3 numbers"),
            "{err}"
        );
        // A missing prefab is named as such.
        std::fs::remove_file(t.0.join("content/loop.prefab.json")).unwrap();
        p.scan(None).unwrap();
        let err = cook_scene(&p, &reg, &scene_id, false).unwrap_err();
        assert!(
            err.contains(&format!("missing prefab {prefab_id}")),
            "{err}"
        );
    }

    #[test]
    fn a_scene_placing_more_instances_than_a_world_holds_fails_the_cook() {
        // A model of 1000 parts, placed whole by 251 entities: 251,000 instances.
        let nodes: Vec<String> = (0..1000)
            .map(|i| format!(r#"{{"mesh":0,"translation":[{i},0,0]}}"#))
            .collect();
        let roots: Vec<String> = (0..1000).map(|i| i.to_string()).collect();
        let glb = glb_of(&format!(
            r#"{TRIANGLE}"meshes":[{MESH}],"nodes":[{}],"scenes":[{{"nodes":[{}]}}],"scene":0}}"#,
            nodes.join(","),
            roots.join(",")
        ));
        let t = TempDir::new("instances");
        let (mut p, model, mut ids) = project_with(&t, &glb);
        let scene_id = ids.asset();
        let entities: Vec<String> = (1..=251)
            .map(|i| {
                format!(
                    r#""{i:016x}":{{"name":"Copy {i}","components":{{"mesh":{{"model":"{model}"}}}}}}"#
                )
            })
            .collect();
        let scene = json::parse(
            format!(
                r#"{{"ganbate":"scene","version":1,"id":"{scene_id}","name":"Many","entities":{{{}}}}}"#,
                entities.join(",")
            )
            .as_bytes(),
        )
        .unwrap();
        p.write_document("content/many.scene.json", Kind::Scene, &scene)
            .unwrap();
        let err = cook_scene(&p, &engine_registry(), &scene_id, false).unwrap_err();
        assert!(err.contains("250000 mesh instances"), "{err}");
    }

    #[test]
    fn content_names_follow_the_bytes() {
        let a = content_name("world", "gbew", b"one");
        assert_eq!(a, content_name("world", "gbew", b"one"));
        assert_ne!(a, content_name("world", "gbew", b"two"));
        assert!(
            a.starts_with("world.") && a.ends_with(".gbew") && a.len() == "world..gbew".len() + 16
        );
    }
}
