use super::cache::*;
use super::*;

impl Renderer {
    pub(super) fn prop_texture_id(
        &self,
        t: Option<PropTexture>,
        fallback: TextureId,
    ) -> Result<TextureId> {
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
        self.props.texture_epoch = self.props.texture_epoch.wrapping_add(1);
        device.generate_mips(t)
    }

    /// Destroys a texture; materials that use it must be updated or destroyed first.
    pub fn destroy_prop_texture(&mut self, device: &mut Device<'_>, texture: PropTexture) {
        if let Some(t) = self.props.textures.remove(texture.0) {
            device.destroy_texture(t);
        }
        self.props.texture_edits.remove(&texture.0);
        self.props.texture_epoch = self.props.texture_epoch.wrapping_add(1);
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
            morph: None,
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
            m.morph = None;
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
    pub(super) fn drop_prop_mesh_skin(&mut self, mesh: PropMesh) {
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
        m.morph = None;
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
        let mut data = vec![0.0f32; positions.len() * 8];
        let vertices = positions.iter().zip(normals).zip(&m.uvs[first..end]);
        for (d, ((p, n), uv)) in data.chunks_exact_mut(8).zip(vertices) {
            d[..3].copy_from_slice(p);
            d[3..6].copy_from_slice(n);
            d[6..].copy_from_slice(uv);
        }
        device.write_buffer(m.vertices, first as u64 * 32, &f32_bytes(&data))?;
        m.drift += farthest_move(&m.positions[first..end], positions);
        m.positions[first..end].copy_from_slice(positions);
        m.revision = m.revision.wrapping_add(1);
        // What morphs of this range need, found once while it is morphed frame after frame.
        let morph = match m.morph.take() {
            Some(morph) if (morph.first, morph.end) == (first, end) => morph,
            _ => {
                let records = m.skin.map(|skin| {
                    let at = skin.first as usize + first * 2;
                    &self.props.skin_records[at..at + positions.len() * 2]
                });
                Morph::new(&m.positions, first, end, records)
            }
        };
        (m.min, m.max) = morph.bounds(&m.positions, positions);
        // Its joints' bounds only widen to hold the moved vertices (they stay bounds; a full
        // update makes them afresh).
        if m.skin.is_some() {
            let changed = morph.widen(&mut m.joint_boxes, positions);
            m.shape = m.shape.wrapping_add(changed);
        }
        m.morph = Some(morph);
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

    pub(super) fn material_entry(
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
            detail: crate::render::renderer::detail_block(&self.look, &desc.name).0,
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
        let fragment = crate::render::renderer::prop_shader_stage(&self.pipelines, source)?;
        Ok(PropShader(self.props.shaders.insert(PropShaderEntry {
            fragment,
            pipelines: [[None; 3]; 3],
        })?))
    }

    /// The overlay pipelines this frame's prop-shader batches need and do not have yet.
    pub(super) fn make_prop_shader_pipelines(&mut self, device: &mut Device<'_>) -> Result<()> {
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
            let made = crate::render::renderer::prop_shader_pipeline(
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
            joints: None,
            dirty: true,
            stamp: [0; 4],
            sig: 0,
            still: 0,
            world_bounds: None,
            modelled: ([0.0; 3], [0.0; 3]),
            cached: None,
            live: None,
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
        self.share_prop_instance_joints(&[instance], joints)
    }

    /// [`Self::set_prop_instance_joints`] for several distinct instances at once, which then
    /// share one copy of the matrices (every skinned part of a model instance follows its
    /// skin's): set, compared and written to the frame's joint buffer once.
    pub(in crate::render) fn share_prop_instance_joints(
        &mut self,
        instances: &[PropInstance],
        joints: &[M4],
    ) -> Result<()> {
        if joints.iter().flatten().any(|x| !x.is_finite()) {
            return Err("set_prop_instance_joints: values must be finite".into());
        }
        let ids: Vec<u32> = instances.iter().map(|i| i.0).collect();
        for &id in &ids {
            self.props.check_joints(id, joints.len())?;
        }
        self.props.share_joints(&ids, joints)
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
            i.dirty = true;
        }
        if i.visible != visible {
            i.visible = visible;
            i.dirty = true;
        }
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
        if i.material != material {
            i.material = material;
            i.dirty = true;
        }
        Ok(())
    }

    /// Multiplies an instance's material colour and alpha (fades, greying, highlights) without
    /// a material of its own.
    pub fn set_prop_instance_tint(&mut self, instance: PropInstance, tint: [f32; 4]) -> Result<()> {
        if tint.iter().any(|x| !x.is_finite()) {
            return Err("set_prop_instance_tint: values must be finite".into());
        }
        let i = self
            .props
            .instances
            .get_mut(instance.0)
            .ok_or_else(|| format!("{instance:?} was destroyed"))?;
        if !same_bits(&i.tint, &tint) {
            i.tint = tint;
            i.dirty = true;
        }
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
    pub(super) fn pick_ray(&self, view: &View, size: (u32, u32), (x, y): (f32, f32)) -> (V3, V3) {
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
    pub(in crate::render) fn props_bounds(&self) -> Option<(V3, V3)> {
        self.props.bounds
    }

    /// Recomputes the prop detail blocks after the Look changed.
    pub(in crate::render) fn restyle_props(&mut self) {
        let look = &self.look;
        let ids: Vec<u32> = self.props.materials.iter().map(|(id, _)| id).collect();
        for id in ids {
            if let Some(m) = self.props.materials.get_mut(id) {
                m.detail = crate::render::renderer::detail_block(look, &m.desc.name).0;
            }
        }
    }

    /// Forgets the per-slot bind groups (after the frame slots were rebuilt).
    pub(in crate::render) fn reset_prop_slots(&mut self, device: &mut Device<'_>) {
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
    pub(in crate::render) fn prepare_props(
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
        let mut live: Vec<Live> = Vec::with_capacity(self.props.rows.capacity() / 40);
        for (_, inst) in self.props.instances.iter() {
            let (Some(mat), Some(_), true) = (
                self.props.materials.get(inst.material.0),
                self.props.meshes.get(inst.mesh.0),
                inst.world_bounds.is_some(),
            ) else {
                continue;
            };
            // The frame's prop bounds (which fit the shadow cascades) as modelled, not posed.
            let (blo, bhi) = inst.modelled;
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
        let mut batches = std::mem::take(&mut self.props.batches);
        let mut placed = std::mem::take(&mut self.props.placed);
        let mut data = std::mem::take(&mut self.props.rows);
        let mut joint_data = std::mem::take(&mut self.props.joint_rows);
        batches.clear();
        placed.clear();
        placed.extend(live.iter().map(|l| l.inst.world_bounds.unwrap_or_default()));
        data.clear();
        joint_data.clear();
        self.props.frames += 1;
        let frame = self.props.frames;
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
            let set = l
                .inst
                .joints
                .and_then(|id| self.props.joint_sets.get_mut(id));
            match (skin, set) {
                (Some(range), Some(set)) if set.matrices.len() > usize::from(range.max_joint) => {
                    // A set's matrices go to the joint rows once a frame, for all who follow it.
                    if set.placed != frame {
                        set.placed = frame;
                        set.first = (joint_data.len() / 16) as u32;
                        joint_data.extend_from_slice(set.matrices.as_flattened());
                    }
                    data.extend_from_slice(&[set.first as f32, range.first as f32, 1.0, 0.0]);
                }
                _ => data.extend_from_slice(&[0.0; 4]),
            }
        }
        let count = live.len() as u32;
        drop(live);
        self.props.batches = batches;
        self.props.placed = placed;
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
        if let Some(s) = &mut self.props.slots[slot] {
            write_changed(device, s.buffer, &mut s.rows, &mut data, 40)?;
            write_changed(device, s.joints, &mut s.joint_rows, &mut joint_data, 16)?;
        }
        self.props.rows = data;
        self.props.joint_rows = joint_data;
        Ok(())
    }

    /// Whether this frame has props to draw over the image.
    pub(in crate::render) fn has_overlays(&self) -> bool {
        self.props.batches.iter().any(|b| b.layer == Layer::Overlay)
    }

    /// Whether this frame has opaque shadow-casting prop batches.
    pub(in crate::render) fn has_prop_shadows(&self) -> bool {
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
    pub(in crate::render) fn has_water_props(&self) -> bool {
        self.props.batches.iter().any(|b| {
            b.layer != Layer::Overlay
                && self
                    .props
                    .materials
                    .get(b.material)
                    .is_some_and(|m| m.desc.shading == Shading::Water)
        })
    }

    pub(super) fn create_prop_slot(
        &self,
        device: &mut Device<'_>,
        slot: usize,
    ) -> Result<PropSlot> {
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
            rows: vec![],
            joint_rows: vec![],
            scene: [scene[0], scene[1]],
            shadow,
            overlay,
        })
    }

    /// Draws this frame's prop batches into a shadow cascade (`light_vp`).
    pub(in crate::render) fn draw_prop_shadows(
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
    pub(super) fn prop_push(mat: &MaterialEntry, first: u32, encode_srgb: bool) -> [f32; 32] {
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

    /// The runs (first, count) of the `count` instances from `first` (their world bounds in
    /// `placed`) whose bounds `within` sees, and how many there are: all of them as one without
    /// it, and as one too where those it sees lie in more than [`MAX_INSTANCE_RUNS`] runs (those
    /// it does not see draw no pixel inside it).
    pub(super) fn instance_runs(
        placed: &[(V3, V3)],
        first: u32,
        count: u32,
        within: Option<&Frustum>,
    ) -> ([(u32, u32); MAX_INSTANCE_RUNS], usize) {
        let mut runs = [(first, count); MAX_INSTANCE_RUNS];
        let Some(f) = within else {
            return (runs, 1);
        };
        let mut n = 0;
        for i in first..first + count {
            let (lo, hi) = placed[i as usize];
            if !f.intersects_aabb(lo, hi) {
                continue;
            }
            if n > 0 && runs[n - 1].0 + runs[n - 1].1 == i {
                runs[n - 1].1 += 1;
            } else if n == MAX_INSTANCE_RUNS {
                return ([(first, count); MAX_INSTANCE_RUNS], 1);
            } else {
                runs[n] = (i, 1);
                n += 1;
            }
        }
        (runs, n)
    }

    /// Draws this frame's opaque prop batches in `set` with the scene shading (`view` 0 main,
    /// 1 reflection): of their instances, those `within` sees (all without it).
    pub(in crate::render) fn draw_props(
        &self,
        enc: &mut crate::gpu::Encoder,
        slot: usize,
        view: usize,
        set: PropSet,
        within: Option<&Frustum>,
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
        let has_water = view == 0
            && set != PropSet::Settled
            && self.props.batches.iter().filter(opaque).any(|b| {
                self.props
                    .materials
                    .get(b.material)
                    .is_some_and(|m| m.data.params[3] == 2.0)
            });
        let mut groups: Vec<(crate::gpu::PipelineId, Option<bool>, bool)> = if view == 0 {
            vec![
                (opaque_pipeline, Some(true), false),
                (cut_pipeline, Some(false), false),
            ]
        } else {
            vec![(self.pipelines.scene_prop, None, false)]
        };
        if has_water {
            groups.push((self.pipelines.water_prop_opaque, Some(true), true));
            groups.push((self.pipelines.water_prop, Some(false), true));
        }
        for (pipeline, no_cut, water_only) in groups {
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
                if (mat.data.params[3] == 2.0) != water_only {
                    continue;
                }
                let (runs, n) = Self::instance_runs(&self.props.placed, b.first, b.count, within);
                if n == 0 {
                    continue;
                }
                if bound != Some(b.material) {
                    enc.set_bind_group(1, mat.group);
                    bound = Some(b.material);
                }
                enc.set_vertex_buffer(mesh.vertices, 0);
                enc.set_index_buffer(mesh.indices, 0);
                for &(first, count) in &runs[..n] {
                    enc.push_constants(&f32_bytes(&Self::prop_push(mat, first, false)));
                    enc.draw_indexed(mesh.index_count, count, 0, 0, 0);
                }
            }
        }
    }

    /// Draws this frame's blended scene props, sorted, after the opaque scene (main view).
    pub(in crate::render) fn draw_blended_props(&self, enc: &mut crate::gpu::Encoder, slot: usize) {
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
            let idx = usize::from(mat.desc.blend == BlendMode::Add);
            let p = if mat.desc.shading == Shading::Water {
                self.pipelines.water_prop_blend[idx]
            } else {
                self.pipelines.scene_prop_blend[idx]
            };
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
    pub(in crate::render) fn draw_overlays(
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
        self.props.joint_sets.drain();
        self.forget_prop_shadows();
        self.props.batches.clear();
        self.props.placed.clear();
        self.props.vacated.clear();
        self.props.vacated_boxes.clear();
        self.props.vacated_live.clear();
        self.props.texture_edits.clear();
        self.forget_world_cache();
    }
}
