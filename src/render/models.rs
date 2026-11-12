//! Models: run-time glTF models ([`crate::assets::model`]) drawn as props. A model's
//! materials and static primitives become shared prop materials and meshes; each placed
//! instance has a pose ([`crate::anim::Pose`]) and one prop instance per primitive of every mesh
//! node, and skinned or morphed primitives get meshes of their own. A skinned primitive follows
//! its joints on the GPU ([`Renderer::set_prop_mesh_skin`]): [`Renderer::update_model_instance`]
//! hands it the pose's joint matrices (one copy per skin, which all its primitives share), and
//! morphs a primitive's vertices on the CPU only when the weight of a target that moves them
//! changed. A morphed primitive's moved vertices are gathered at the front of its vertices as it
//! is uploaded ([`anim::gather_morphed`]), so a morph rewrites one short run of them.
//!
//! Textures are the host's: it decodes the model's images (browsers decode PNG, JPEG and WebP
//! natively) into prop textures and passes them in, one per image.
use super::math::{self, M4};
use super::props::{
    BlendMode, MaterialDesc, MeshData, PropInstance, PropMaterial, PropMesh, PropTexture,
};
use super::renderer::Renderer;
use crate::anim::{self, Pose};
use crate::assets::model::{AlphaMode, Model};
use crate::gpu::pool::Pool;
use crate::gpu::{Device, Result};
use crate::world::Shading;

handle!(
    /// A loaded model.
    ModelId
);
handle!(
    /// A placed, posed model.
    ModelInstance
);

pub(super) struct ModelEntry {
    model: Model,
    /// One per glTF material, then the default material for primitives without one.
    materials: Vec<PropMaterial>,
    /// Per mesh and primitive: the shared mesh of a static primitive (`None` when deformable).
    meshes: Vec<Vec<Option<PropMesh>>>,
}

struct Part {
    node: usize,
    mesh: usize,
    primitive: usize,
    instance: PropInstance,
    /// This instance's own mesh for a skinned or morphed primitive.
    own: Option<PropMesh>,
    /// The morph weights its own mesh's vertices were last morphed with.
    morphed: Vec<f32>,
    /// The vertices its morph targets move (ascending): only they are morphed again.
    moved: Vec<u32>,
    /// Per morph target: whether it moves any of them.
    moving: Vec<bool>,
}

pub(super) struct InstanceEntry {
    model: ModelId,
    transform: M4,
    visible: bool,
    pose: Pose,
    parts: Vec<Part>,
    hidden_nodes: Vec<bool>,
    globals: Vec<M4>,
}

#[derive(Default)]
pub(super) struct Models {
    models: Pool<ModelEntry>,
    instances: Pool<InstanceEntry>,
}

/// Whether a primitive of `mesh` under `node` is deformed per instance.
fn deformable(model: &Model, node: usize, mesh: usize, primitive: usize) -> bool {
    let p = &model.meshes[mesh].primitives[primitive];
    !p.targets.is_empty() || (!p.joints.is_empty() && model.nodes[node].skin.is_some())
}

/// The nodes drawn: every node with a mesh reachable from the scene's roots.
fn drawn_nodes(model: &Model) -> Vec<usize> {
    let mut out = vec![];
    let mut stack: Vec<usize> = model.roots.iter().rev().copied().collect();
    while let Some(n) = stack.pop() {
        if model.nodes[n].mesh.is_some() {
            out.push(n);
        }
        stack.extend(model.nodes[n].children.iter().rev());
    }
    out
}

impl Renderer {
    /// Uploads a model: its materials (with `textures`, one prop texture per image of the
    /// model, `None` where the host could not decode it) and its static primitives. A morphed
    /// primitive keeps its vertices with those its targets move first ([`anim::gather_morphed`]),
    /// as [`Self::model`] then gives it.
    pub fn create_model(
        &mut self,
        device: &mut Device<'_>,
        mut model: Model,
        textures: &[Option<PropTexture>],
    ) -> Result<ModelId> {
        if textures.len() != model.images.len() {
            return Err(format!(
                "create_model: {} textures for {} images",
                textures.len(),
                model.images.len()
            ));
        }
        let tex = |r: Option<crate::assets::model::TextureRef>| {
            r.and_then(|r| textures[model.textures[r.texture].image])
        };
        let mut materials = Vec::with_capacity(model.materials.len() + 1);
        let mut made_meshes: Vec<PropMesh> = vec![];
        let cleanup = |r: &mut Renderer,
                       device: &mut Device<'_>,
                       mats: &[PropMaterial],
                       meshes: &[PropMesh]| {
            for &m in mats {
                r.destroy_prop_material(device, m);
            }
            for &m in meshes {
                r.destroy_prop_mesh(device, m);
            }
        };
        for m in &model.materials {
            let desc = MaterialDesc {
                name: m.name.clone(),
                base_color: m.base_color,
                emissive: m.emissive,
                roughness: m.roughness,
                metallic: m.metallic,
                shading: if m.unlit {
                    Shading::Unlit
                } else {
                    Shading::Standard
                },
                alpha_cutoff: (m.alpha_mode == AlphaMode::Mask).then_some(m.alpha_cutoff),
                blend: if m.alpha_mode == AlphaMode::Blend {
                    BlendMode::Alpha
                } else {
                    BlendMode::Opaque
                },
                base_texture: tex(m.base_texture),
                normal_texture: tex(m.normal_texture),
                emissive_texture: tex(m.emissive_texture),
                normal_scale: m.normal_scale,
                ..MaterialDesc::default()
            };
            match self.create_prop_material(device, &desc) {
                Ok(id) => materials.push(id),
                Err(e) => {
                    cleanup(self, device, &materials, &[]);
                    return Err(e);
                }
            }
        }
        match self.create_prop_material(device, &MaterialDesc::default()) {
            Ok(id) => materials.push(id),
            Err(e) => {
                cleanup(self, device, &materials, &[]);
                return Err(e);
            }
        }
        for p in model.meshes.iter_mut().flat_map(|m| &mut m.primitives) {
            anim::gather_morphed(p);
        }
        // Static primitives are shared; which are static depends on the nodes that use them.
        let mut needs_shared: Vec<Vec<bool>> = model
            .meshes
            .iter()
            .map(|m| vec![false; m.primitives.len()])
            .collect();
        for n in drawn_nodes(&model) {
            let mesh = model.nodes[n].mesh.unwrap_or(0);
            for (p, shared) in needs_shared[mesh].iter_mut().enumerate() {
                if !deformable(&model, n, mesh, p) {
                    *shared = true;
                }
            }
        }
        let mut meshes = Vec::with_capacity(model.meshes.len());
        for (mi, m) in model.meshes.iter().enumerate() {
            let mut prims = Vec::with_capacity(m.primitives.len());
            for (pi, p) in m.primitives.iter().enumerate() {
                if !needs_shared[mi][pi] {
                    prims.push(None);
                    continue;
                }
                let made = self.create_prop_mesh(
                    device,
                    &MeshData {
                        positions: &p.positions,
                        normals: &p.normals,
                        uvs: &p.uvs,
                        indices: &p.indices,
                    },
                );
                match made {
                    Ok(id) => {
                        made_meshes.push(id);
                        prims.push(Some(id));
                    }
                    Err(e) => {
                        cleanup(self, device, &materials, &made_meshes);
                        return Err(format!("model mesh {mi}: {e}"));
                    }
                }
            }
            meshes.push(prims);
        }
        Ok(ModelId(self.models.models.insert(ModelEntry {
            model,
            materials,
            meshes,
        })?))
    }

    /// The model's data (nodes, meshes, clips) for hosts that look up names.
    pub fn model(&self, id: ModelId) -> Result<&Model> {
        self.models
            .models
            .get(id.0)
            .map(|m| &m.model)
            .ok_or_else(|| format!("{id:?} was destroyed"))
    }

    /// The prop material made for glTF material `index` (to restyle it).
    pub fn model_material(&self, id: ModelId, index: usize) -> Result<PropMaterial> {
        let m = self
            .models
            .models
            .get(id.0)
            .ok_or_else(|| format!("{id:?} was destroyed"))?;
        m.materials
            .get(index)
            .copied()
            .filter(|_| index < m.model.materials.len())
            .ok_or_else(|| format!("{id:?} has no material {index}"))
    }

    /// Places a model at `transform` (column-major, model to world) in its rest pose.
    pub fn create_model_instance(
        &mut self,
        device: &mut Device<'_>,
        id: ModelId,
        transform: &M4,
    ) -> Result<ModelInstance> {
        let entry = self
            .models
            .models
            .get(id.0)
            .ok_or_else(|| format!("{id:?} was destroyed"))?;
        let model = &entry.model;
        let pose = Pose::rest(model);
        let globals = pose.globals(model);
        let default_material = *entry.materials.last().expect("a default material");
        // What to create, decided while the model is borrowed.
        struct Plan {
            node: usize,
            mesh: usize,
            primitive: usize,
            shared: Option<PropMesh>,
            material: PropMaterial,
            /// Positions and normals of a primitive deformed per instance: morphed, not
            /// skinned (the GPU skins it).
            deformed: Option<Deformed>,
            /// A skinned primitive's vertices' joints and weights.
            skin: Option<Skin>,
            morphed: Vec<f32>,
            moved: Vec<u32>,
            moving: Vec<bool>,
        }
        type Deformed = (Vec<[f32; 3]>, Vec<[f32; 3]>);
        type Skin = (Vec<[u16; 4]>, Vec<[f32; 4]>);
        let mut plans = vec![];
        for n in drawn_nodes(model) {
            let mesh = model.nodes[n].mesh.unwrap_or(0);
            for (pi, p) in model.meshes[mesh].primitives.iter().enumerate() {
                let material = p.material.map_or(default_material, |m| entry.materials[m]);
                let deformed =
                    deformable(model, n, mesh, pi).then(|| anim::deform(p, &pose.weights[n], None));
                let skin = (model.nodes[n].skin.is_some() && !p.joints.is_empty())
                    .then(|| (p.joints.clone(), p.weights.clone()));
                plans.push(Plan {
                    node: n,
                    mesh,
                    primitive: pi,
                    shared: entry.meshes[mesh][pi],
                    material,
                    deformed,
                    skin,
                    morphed: pose.weights[n].clone(),
                    moved: anim::morphed_vertices(p),
                    moving: anim::moving_targets(p),
                });
            }
        }
        let uvs_of = |r: &Renderer, mesh: usize, primitive: usize| {
            r.models.models.get(id.0).map(|e| {
                let p = &e.model.meshes[mesh].primitives[primitive];
                (p.uvs.clone(), p.indices.clone())
            })
        };
        let mut parts: Vec<Part> = vec![];
        for plan in plans {
            let (mesh_id, own) = match (plan.shared, plan.deformed) {
                (Some(m), None) => (m, None),
                (_, Some((positions, normals))) => {
                    let (uvs, indices) =
                        uvs_of(self, plan.mesh, plan.primitive).ok_or("model destroyed")?;
                    let made = self.create_prop_mesh(
                        device,
                        &MeshData {
                            positions: &positions,
                            normals: &normals,
                            uvs: &uvs,
                            indices: &indices,
                        },
                    );
                    let skinned = made.and_then(|m| match &plan.skin {
                        Some((joints, weights)) => {
                            match self.set_prop_mesh_skin(m, joints, weights) {
                                Ok(()) => Ok(m),
                                Err(e) => {
                                    self.destroy_prop_mesh(device, m);
                                    Err(e)
                                }
                            }
                        }
                        None => Ok(m),
                    });
                    match skinned {
                        Ok(m) => (m, Some(m)),
                        Err(e) => {
                            self.release_parts(device, parts);
                            return Err(e);
                        }
                    }
                }
                (None, None) => continue,
            };
            match self.create_prop_instance(mesh_id, plan.material, &math::IDENTITY) {
                Ok(instance) => parts.push(Part {
                    node: plan.node,
                    mesh: plan.mesh,
                    primitive: plan.primitive,
                    instance,
                    own,
                    morphed: plan.morphed,
                    moved: plan.moved,
                    moving: plan.moving,
                }),
                Err(e) => {
                    if let Some(m) = own {
                        self.destroy_prop_mesh(device, m);
                    }
                    self.release_parts(device, parts);
                    return Err(e);
                }
            }
        }
        let nodes = self
            .models
            .models
            .get(id.0)
            .map_or(0, |e| e.model.nodes.len());
        let instance = ModelInstance(self.models.instances.insert(InstanceEntry {
            model: id,
            transform: *transform,
            visible: true,
            pose,
            parts,
            hidden_nodes: vec![false; nodes],
            globals,
        })?);
        self.update_model_instance(device, instance)?;
        Ok(instance)
    }

    fn release_parts(&mut self, device: &mut Device<'_>, parts: Vec<Part>) {
        for p in parts {
            self.destroy_prop_instance(p.instance);
            if let Some(m) = p.own {
                self.destroy_prop_mesh(device, m);
            }
        }
    }

    fn model_instance_entry(&mut self, i: ModelInstance) -> Result<&mut InstanceEntry> {
        self.models
            .instances
            .get_mut(i.0)
            .ok_or_else(|| format!("{i:?} was destroyed"))
    }

    /// The instance's pose, to change before [`Renderer::update_model_instance`].
    pub fn model_pose_mut(&mut self, i: ModelInstance) -> Result<&mut Pose> {
        Ok(&mut self.model_instance_entry(i)?.pose)
    }

    /// The instance's pose and its model, for sampling clips into it.
    pub fn model_pose_and_model(&mut self, i: ModelInstance) -> Result<(&mut Pose, &Model)> {
        let entry = self
            .models
            .instances
            .get_mut(i.0)
            .ok_or_else(|| format!("{i:?} was destroyed"))?;
        let model = self
            .models
            .models
            .get(entry.model.0)
            .map(|m| &m.model)
            .ok_or("the instance's model was destroyed")?;
        Ok((&mut entry.pose, model))
    }

    /// Moves the instance and shows or hides it (its pose is kept).
    pub fn set_model_instance(
        &mut self,
        i: ModelInstance,
        transform: &M4,
        visible: bool,
    ) -> Result<()> {
        if transform.iter().any(|x| !x.is_finite()) {
            return Err("set_model_instance: the matrix is not finite".into());
        }
        let e = self.model_instance_entry(i)?;
        e.transform = *transform;
        e.visible = visible;
        self.place_model_parts(i)
    }

    /// Shows or hides the meshes of one node (a sub-assembly or equipment attachment).
    pub fn set_model_node_visible(
        &mut self,
        i: ModelInstance,
        node: usize,
        visible: bool,
    ) -> Result<()> {
        let e = self.model_instance_entry(i)?;
        let hidden = e
            .hidden_nodes
            .get_mut(node)
            .ok_or_else(|| format!("no node {node}"))?;
        *hidden = !visible;
        self.place_model_parts(i)
    }

    /// Tints every part of the instance.
    pub fn set_model_instance_tint(&mut self, i: ModelInstance, tint: [f32; 4]) -> Result<()> {
        let parts: Vec<PropInstance> = self
            .model_instance_entry(i)?
            .parts
            .iter()
            .map(|p| p.instance)
            .collect();
        for p in parts {
            self.set_prop_instance_tint(p, tint)?;
        }
        Ok(())
    }

    /// Applies the pose: node transforms, the joints of skinned primitives (skinned on the
    /// GPU) and morphing (on the CPU, for a primitive where a target that moves its vertices
    /// changed weight).
    pub fn update_model_instance(
        &mut self,
        device: &mut Device<'_>,
        i: ModelInstance,
    ) -> Result<()> {
        let e = self
            .models
            .instances
            .get(i.0)
            .ok_or_else(|| format!("{i:?} was destroyed"))?;
        let model = &self
            .models
            .models
            .get(e.model.0)
            .ok_or("the instance's model was destroyed")?
            .model;
        let globals = e.pose.globals(model);
        // Each skin's joint matrices once, shared by every primitive that follows them.
        let mut skins: Vec<(usize, Vec<M4>, Vec<PropInstance>)> = vec![];
        let mut morphs = vec![];
        for (k, part) in e.parts.iter().enumerate() {
            let Some(own) = part.own else { continue };
            let p = &model.meshes[part.mesh].primitives[part.primitive];
            if let Some(s) = model.nodes[part.node].skin.filter(|_| !p.joints.is_empty()) {
                match skins.iter_mut().find(|(skin, ..)| *skin == s) {
                    Some((.., parts)) => parts.push(part.instance),
                    None => skins.push((
                        s,
                        anim::joint_matrices(model, s, &globals),
                        vec![part.instance],
                    )),
                }
            }
            let weights = &e.pose.weights[part.node];
            if !p.targets.is_empty() && part.morphed != *weights {
                match (part.moved.first(), part.moved.last()) {
                    // Only targets that move none of its vertices changed: nothing to rewrite.
                    _ if !anim::morph_changes(&part.morphed, weights, &part.moving) => {
                        morphs.push((k, own, Some(0), vec![], vec![], weights.clone()));
                    }
                    // Without normals of its own the whole mesh is shaded again.
                    _ if p.normals.is_empty() => {
                        let (pos, nrm) = anim::deform(p, weights, None);
                        morphs.push((k, own, None, pos, nrm, weights.clone()));
                    }
                    // Only the span of the vertices its targets move is morphed and rewritten.
                    (Some(&lo), Some(&hi)) => {
                        let (first, count) = (lo as usize, (hi - lo) as usize + 1);
                        let (pos, nrm) = anim::morph_range(p, weights, &part.moved, first, count);
                        morphs.push((k, own, Some(first), pos, nrm, weights.clone()));
                    }
                    _ => morphs.push((k, own, Some(0), vec![], vec![], weights.clone())),
                }
            }
        }
        for (k, mesh, first, pos, nrm, weights) in morphs {
            match first {
                Some(first) => self.update_prop_mesh_range(device, mesh, first, &pos, &nrm)?,
                None => self.update_prop_mesh(device, mesh, &pos, &nrm)?,
            }
            if let Some(part) = self.model_instance_entry(i)?.parts.get_mut(k) {
                part.morphed = weights;
            }
        }
        for (_, joints, parts) in &skins {
            self.share_prop_instance_joints(parts, joints)?;
        }
        self.model_instance_entry(i)?.globals = globals;
        self.place_model_parts(i)
    }

    /// Sets every part's prop instance from the instance transform and node globals.
    fn place_model_parts(&mut self, i: ModelInstance) -> Result<()> {
        let e = self
            .models
            .instances
            .get(i.0)
            .ok_or_else(|| format!("{i:?} was destroyed"))?;
        let model = &self
            .models
            .models
            .get(e.model.0)
            .ok_or("the instance's model was destroyed")?
            .model;
        let mut placements = Vec::with_capacity(e.parts.len());
        for part in &e.parts {
            let p = &model.meshes[part.mesh].primitives[part.primitive];
            let skinned = !p.joints.is_empty() && model.nodes[part.node].skin.is_some();
            // Skinned vertices are already in the model's space.
            let m = if skinned {
                e.transform
            } else {
                math::mul(&e.transform, &e.globals[part.node])
            };
            placements.push((part.instance, m, e.visible && !e.hidden_nodes[part.node]));
        }
        for (instance, m, visible) in placements {
            self.set_prop_instance(instance, &m, visible)?;
        }
        Ok(())
    }

    /// Node `node`'s world transform as of the last update (a socket an attachment rides on).
    pub fn model_node_world(&self, i: ModelInstance, node: usize) -> Result<M4> {
        let e = self
            .models
            .instances
            .get(i.0)
            .ok_or_else(|| format!("{i:?} was destroyed"))?;
        let g = e
            .globals
            .get(node)
            .ok_or_else(|| format!("no node {node}"))?;
        Ok(math::mul(&e.transform, g))
    }

    /// The model instance a prop instance belongs to, with the node it draws.
    pub fn model_part_owner(&self, prop: PropInstance) -> Option<(ModelInstance, usize)> {
        self.models.instances.iter().find_map(|(id, e)| {
            e.parts
                .iter()
                .find(|p| p.instance == prop)
                .map(|p| (ModelInstance(id), p.node))
        })
    }

    pub fn destroy_model_instance(&mut self, device: &mut Device<'_>, i: ModelInstance) {
        if let Some(e) = self.models.instances.remove(i.0) {
            self.release_parts(device, e.parts);
        }
    }

    /// Destroys a model, its instances, meshes and materials (the host's textures stay).
    pub fn destroy_model(&mut self, device: &mut Device<'_>, id: ModelId) {
        let users: Vec<u32> = self
            .models
            .instances
            .iter()
            .filter(|(_, e)| e.model == id)
            .map(|(i, _)| i)
            .collect();
        for u in users {
            self.destroy_model_instance(device, ModelInstance(u));
        }
        if let Some(m) = self.models.models.remove(id.0) {
            for mat in m.materials {
                self.destroy_prop_material(device, mat);
            }
            for mesh in m.meshes.into_iter().flatten().flatten() {
                self.destroy_prop_mesh(device, mesh);
            }
        }
    }

    /// Forgets every model (their props go with `destroy_props`).
    pub(super) fn drop_models(&mut self) {
        self.models = Models::default();
    }
}
