use super::*;

impl Renderer {
    pub fn new(
        device: &mut Device<'_>,
        world: &crate::world::World,
        look: Look,
        quality: Quality,
        size: (u32, u32),
        output: Output,
    ) -> Result<Self> {
        let out_format = match output {
            Output::Still => HDR,
            Output::Surface(f) => f,
        };
        let pipelines = Pipelines::new(device, out_format)?;
        let scene = GpuScene::upload(device, world)?;
        let sampled = TextureUsage::SAMPLED | TextureUsage::COPY_DST;
        let solid = |device: &mut Device<'_>, format, texel: [u8; 4]| -> Result<TextureId> {
            let t = device.create_texture(&TextureDesc::new(1, 1, format, sampled))?;
            device.write_texture(t, 0, 0, &texel)?;
            Ok(t)
        };
        let white = solid(device, Format::Rgba8Srgb, [255; 4])?;
        let flat = solid(device, Format::Rgba8Unorm, [128, 128, 255, 255])?;
        let dummy = solid(device, Format::Rgba8Unorm, [0; 4])?;
        let material_sampler = device.create_sampler(&SamplerDesc {
            anisotropy: 8.0,
            ..SamplerDesc::default()
        })?;
        let linear_clamp = device.create_sampler(&SamplerDesc {
            address: Address::Clamp,
            ..SamplerDesc::default()
        })?;
        let shadow_sampler = device.create_sampler(&SamplerDesc {
            address: Address::Clamp,
            compare: Some(Compare::LessEqual),
            mip_filter: Filter::Nearest,
            ..SamplerDesc::default()
        })?;
        let mut materials = vec![];
        let mut shadow_materials = vec![];
        for m in &scene.materials {
            shadow_materials.push(if m.params[0] >= 0.0 {
                Some(device.create_bind_group(
                    pipelines.shadow_masked,
                    1,
                    &[
                        (
                            0,
                            Resource::Texture(m.base.map_or(white, |i| scene.textures[i])),
                        ),
                        (3, Resource::Sampler(material_sampler)),
                    ],
                )?)
            } else {
                None
            });
            let tex = |i: Option<usize>, fallback| i.map_or(fallback, |i| scene.textures[i]);
            materials.push(device.create_bind_group(
                pipelines.scene,
                1,
                &[
                    (0, Resource::Texture(tex(m.base, white))),
                    (1, Resource::Texture(tex(m.normal, flat))),
                    (2, Resource::Texture(tex(m.emission, white))),
                    (3, Resource::Sampler(material_sampler)),
                ],
            )?);
        }
        let shadow_size = quality.shadow_size.max(16);
        let shadow_map = device.create_texture(&TextureDesc {
            width: shadow_size,
            height: shadow_size,
            layers: 4,
            mips: 1,
            format: Format::Depth32Float,
            usage: TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED,
        })?;
        let light_shadow_map = device.create_texture(&TextureDesc {
            width: quality.light_shadow_size.max(16),
            height: quality.light_shadow_size.max(16),
            layers: crate::render::lights::MAX_SHADOWED_LIGHTS as u32,
            mips: 1,
            format: Format::Depth32Float,
            usage: TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED,
        })?;
        let ground_size = quality.ground_size.max(64);
        let ground_depth = device.create_texture(&TextureDesc {
            width: ground_size,
            height: ground_size,
            layers: 1,
            mips: 1,
            format: Format::Depth32Float,
            usage: TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED,
        })?;
        let ground_mask = device.create_texture(&TextureDesc {
            width: ground_size,
            height: ground_size,
            layers: 1,
            mips: 1,
            format: GROUND_FORMAT,
            usage: TextureUsage::COLOR_TARGET | TextureUsage::SAMPLED,
        })?;
        let world_size = quality.world_size.max(64);
        let world_depth = device.create_texture(&TextureDesc {
            width: world_size,
            height: world_size,
            layers: 1,
            mips: 1,
            format: Format::Depth32Float,
            usage: TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED,
        })?;
        let world_mask = device.create_texture(&TextureDesc {
            width: world_size,
            height: world_size,
            layers: 1,
            mips: 1,
            format: GROUND_FORMAT,
            usage: TextureUsage::COLOR_TARGET | TextureUsage::SAMPLED,
        })?;
        let world = world_view(&scene, world_size);
        let (details, grassy) = material_details(&look, &scene);
        let lut = device.create_texture(&TextureDesc::new(
            LUT_SIZE * LUT_SIZE,
            LUT_SIZE,
            Format::Rgba8Unorm,
            sampled,
        ))?;
        device.write_texture(lut, 0, 0, &bake_lut(&look))?;
        let targets = Self::create_targets(device, &quality, output, size, false)?;
        let mut r = Self {
            scene,
            pipelines,
            quality,
            output,
            look,
            targets,
            slots: vec![],
            materials,
            shadow_materials,
            shadow_map,
            shadow_layers_ready: false,
            static_shadows_vp: None,
            prop_shadows_for: None,
            light_shadow_map,
            light_shadows_ready: false,
            ground_depth,
            ground_mask,
            ground_cached_vp: None,
            world_depth,
            world_mask,
            world,
            world_drawn: false,
            world_cache: None,
            grass_grid: [0; 4],
            receivers: None,
            water_statics: None,
            cache_ao_revision: 0,
            cache_age: 0,
            split_passes: false,
            world_moved: false,
            ao_quiet_frames: AO_SETTLE_FRAMES,
            details,
            grassy,
            lut,
            lut_dirty: false,
            dummy,
            linear_clamp,
            shadow_sampler,
            white,
            flat,
            material_sampler,
            props: crate::render::props::Props::default(),
            game_lights: crate::render::lights::GameLights::default(),
            models: crate::render::models::Models::default(),
            stats: FrameStats::default(),
        };
        let slots = match output {
            Output::Still => 1,
            Output::Surface(_) => 2,
        };
        for _ in 0..slots {
            let s = r.create_slot(device)?;
            r.slots.push(s);
        }
        Ok(r)
    }

    pub fn look(&self) -> &Look {
        &self.look
    }
    /// Draws the cached picture afresh next frame.
    pub(in crate::render) fn forget_world_cache(&mut self) {
        self.world_cache = None;
        self.forget_ao();
    }
    pub fn set_look(&mut self, look: Look) {
        (self.details, self.grassy) = material_details(&look, &self.scene);
        self.look = look;
        self.world_drawn = false;
        self.world_cache = None;
        self.forget_ao();
        self.ground_cached_vp = None;
        self.lut_dirty = true;
        self.restyle_props();
    }
    /// Current per-frame rendering quality budget.
    pub fn quality(&self) -> Quality {
        self.quality
    }
    /// Visibility, cache, and pass-skip telemetry recorded during the most recent [`Renderer::encode`].
    pub fn last_frame_stats(&self) -> FrameStats {
        self.stats
    }
    /// Advances the baked vertex AO by up to `samples` vertices' worth of work outside a frame
    /// (reading or building occluders first, then sampling nearest `eye`), so a host can bake in
    /// idle time with its own clock. Returns whether there was work to do: false once converged,
    /// and while world edits have not yet paused.
    pub fn step_ao(&mut self, device: &mut Device<'_>, samples: usize, eye: V3) -> Result<bool> {
        if self.ao_quiet_frames < AO_SETTLE_FRAMES || self.scene.ao.is_idle() {
            return Ok(false);
        }
        self.scene.step_ao(device, samples, Some(eye))?;
        Ok(true)
    }

    /// Takes the world's baked vertex AO from a sidecar baked for it (`ganbate-build bake-ao`),
    /// so frames bake only around what moves. Fails, changing nothing, when the sidecar was
    /// baked for another world.
    pub fn import_ao(&mut self, device: &mut Device<'_>, data: &[u8]) -> Result<()> {
        self.scene.import_ao(device, data)?;
        self.world_cache = None;
        self.forget_ao();
        Ok(())
    }

    /// Completes the baked vertex AO now (a capture that must not depend on frame count).
    pub fn finish_ao(&mut self, device: &mut Device<'_>) -> Result<()> {
        self.scene.finish_ao(device).map(|_| ())
    }
    /// Converged fraction of the baked vertex AO in `[0, 1]` (it refines a slice per live
    /// frame after start and after world edits).
    /// The props' shadow maps no longer hold the props there are: the next frame draws them again.
    pub(in crate::render) fn forget_prop_shadows(&mut self) {
        self.prop_shadows_for = None;
    }
    pub fn ao_progress(&self) -> f32 {
        self.scene.ao.progress()
    }
    /// Draws the main pass as several, one for each thing it draws over the cached picture (the
    /// picture, the live props, the water and the masked water, the grass, the sky, the weather,
    /// the blended props), so
    /// a device's pass timing shows what each costs; the picture is the same.
    pub fn set_split_passes(&mut self, on: bool) {
        self.split_passes = on;
    }
    pub fn size(&self) -> (u32, u32) {
        (self.targets.width, self.targets.height)
    }
    pub fn triangles(&self) -> usize {
        self.scene.triangles
    }
    pub fn lights(&self) -> usize {
        self.scene.lights.len()
    }
    pub fn water_level(&self) -> Option<f32> {
        self.scene.water_level
    }
    /// World bounds of everything drawn.
    pub fn bounds(&self) -> (V3, V3) {
        (self.scene.min, self.scene.max)
    }

    /// The size-dependent targets, with `fs_grade`'s where frames grade once (`grade_once`).
    pub(super) fn create_targets(
        device: &mut Device<'_>,
        q: &Quality,
        output: Output,
        (width, height): (u32, u32),
        grade_once: bool,
    ) -> Result<Targets> {
        let rt = TextureUsage::COLOR_TARGET | TextureUsage::SAMPLED;
        let tex = |d: &mut Device<'_>, w: u32, h: u32, f: Format, u| {
            d.create_texture(&TextureDesc::new(w.max(1), h.max(1), f, u))
        };
        let scaled = |s: f32| {
            (
                ((width as f32 * s).round() as u32).max(1),
                ((height as f32 * s).round() as u32).max(1),
            )
        };
        let refl = scaled(q.reflection_scale);
        let ao = scaled(q.ao_scale);
        let ((gw, gh), (cw, ch)) = ao_grass_sizes((width, height));
        let bloom_size = ((width / 2).max(1), (height / 2).max(1));
        let max_levels = 32 - bloom_size.0.min(bloom_size.1).leading_zeros();
        let bloom_levels = q.bloom_levels.clamp(1, max_levels.max(1));
        let depth_usage = TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED;
        let mut t = Targets {
            width,
            height,
            hdr: tex(device, width, height, HDR, rt)?,
            aux: tex(device, width, height, HDR, rt)?,
            // The frame's depth starts as a copy of the cached picture's (`encode`).
            depth: tex(
                device,
                width,
                height,
                Format::Depth32Float,
                depth_usage | TextureUsage::COPY_DST,
            )?,
            cache_hdr: tex(device, width, height, HDR, rt)?,
            cache_aux: tex(device, width, height, HDR, rt)?,
            cache_depth: tex(
                device,
                width,
                height,
                Format::Depth32Float,
                depth_usage | TextureUsage::COPY_SRC,
            )?,
            reshade_depth: tex(device, width, height, Format::Depth32Float, depth_usage)?,
            cache_shadow: tex(device, width, height, HDR, rt)?,
            cache_water: (output != Output::Still && device.limits().float32_readable)
                .then(|| tex(device, width, height, WATER_STATICS_FORMAT, rt))
                .transpose()?,
            refl,
            refl_hdr: tex(device, refl.0, refl.1, HDR, rt)?,
            refl_aux: tex(device, refl.0, refl.1, HDR, rt)?,
            refl_depth: tex(device, refl.0, refl.1, Format::Depth32Float, depth_usage)?,
            refl_cleared: false,
            ao,
            ao_tex: tex(device, ao.0, ao.1, Format::R16Float, rt)?,
            ao_for: None,
            ao_cleared: false,
            ao_grass: [
                tex(device, gw, gh, Format::R8Unorm, rt)?,
                tex(device, gw, gh, Format::R8Unorm, rt)?,
            ],
            ao_grass_cur: 0,
            ao_grass_valid: [false; 2],
            ao_grass_time: 0,
            ao_grass_sway: 0,
            ao_grass_union: tex(device, gw, gh, Format::R8Unorm, rt)?,
            ao_grass_coarse: tex(device, cw, ch, Format::R8Unorm, rt)?,
            lit: tex(device, width, height, HDR, rt)?,
            bloom: device.create_texture(&TextureDesc {
                width: bloom_size.0,
                height: bloom_size.1,
                layers: 1,
                mips: bloom_levels,
                format: HDR,
                usage: rt,
            })?,
            bloom_size,
            bloom_levels,
            ldr: tex(device, width, height, HDR, rt)?,
            paint: tex(device, width, height, HDR, rt)?,
            graded: grade_once
                .then(|| tex(device, width, height, GRADED_FORMAT, rt))
                .transpose()?,
            still: None,
            readback: None,
            tiles: [
                (width.div_ceil(q.tile_size), height.div_ceil(q.tile_size)),
                (refl.0.div_ceil(q.tile_size), refl.1.div_ceil(q.tile_size)),
            ],
        };
        if output == Output::Still {
            t.still = Some(tex(
                device,
                width,
                height,
                HDR,
                rt | TextureUsage::COPY_SRC,
            )?);
            t.readback = Some(device.create_buffer(&BufferDesc {
                size: u64::from(width) * u64::from(height) * 8,
                usage: BufferUsage::COPY_DST,
                memory: Memory::Readback,
            })?);
        }
        Ok(t)
    }

    pub(super) fn create_slot(&self, device: &mut Device<'_>) -> Result<Slot> {
        let t = &self.targets;
        let p = &self.pipelines;
        let upload = |d: &mut Device<'_>, size: u64, usage| {
            d.create_buffer(&BufferDesc {
                size: size.max(16),
                usage,
                memory: Memory::Upload,
            })
        };
        let mut frame = vec![];
        let mut tiles = vec![];
        let mut index = vec![];
        for v in 0..2 {
            let (tx, ty) = t.tiles[v];
            let n = u64::from(tx) * u64::from(ty);
            frame.push(upload(device, FRAME_BYTES, BufferUsage::UNIFORM)?);
            tiles.push(upload(
                device,
                n * u64::from(CLUSTER_SLICES) * 8,
                BufferUsage::STORAGE,
            )?);
            index.push(upload(
                device,
                n * CLUSTER_ENTRIES_PER_TILE * 4,
                BufferUsage::STORAGE,
            )?);
        }
        let buf = |id| Resource::Buffer {
            id,
            offset: 0,
            size: 0,
        };
        let mut scene = vec![];
        let mut sky = vec![];
        let mut blades = vec![];
        let mut water = None;
        for v in 0..2 {
            let reflection = if v == 0 { t.refl_hdr } else { self.dummy };
            let entries = [
                (0, buf(frame[v])),
                (1, buf(self.scene.light_buffer)),
                (2, buf(tiles[v])),
                (3, buf(index[v])),
                (4, Resource::Texture(self.shadow_map)),
                (5, Resource::Sampler(self.shadow_sampler)),
                (6, Resource::Texture(reflection)),
                (7, Resource::Sampler(self.linear_clamp)),
                (8, Resource::Texture(self.ground_depth)),
                (9, Resource::Texture(self.ground_mask)),
                (10, Resource::Texture(self.world_depth)),
                (14, Resource::Texture(self.light_shadow_map)),
            ];
            scene.push(device.create_bind_group(p.scene, 0, &entries)?);
            sky.push(device.create_bind_group(p.sky, 0, &entries[..1])?);
            if v == 0 {
                blades.push(device.create_bind_group(p.grass, 0, &entries)?);
                blades.push(device.create_bind_group(p.flower, 0, &entries)?);
                blades.push(device.create_bind_group(p.particle, 0, &entries)?);
                if let Some(kept) = t.cache_water {
                    let moved = upload(device, RESHADE_BYTES, BufferUsage::UNIFORM)?;
                    let more = [(17, buf(moved)), (19, Resource::Texture(kept))];
                    let group =
                        device.create_bind_group(p.water, 0, &[&entries[..], &more].concat())?;
                    water = Some((group, moved));
                }
            }
        }
        fn pair<T: Copy>(v: Vec<T>) -> [T; 2] {
            [v[0], v[1]]
        }
        let (frame, tiles, index) = (pair(frame), pair(tiles), pair(index));
        let (scene, sky) = (pair(scene), pair(sky));
        let (grass, flower, particle) = (blades[0], blades[1], blades[2]);
        let ao_changed = upload(device, AO_CHANGED_BYTES, BufferUsage::UNIFORM)?;
        device.write_buffer(ao_changed, 0, &[0; 32])?;
        let post = |d: &mut Device<'_>, pipe, src: Resource, extra: Resource| {
            d.create_bind_group(
                pipe,
                0,
                &[
                    (0, buf(frame[0])),
                    (1, Resource::Texture(t.depth)),
                    (2, Resource::Texture(t.aux)),
                    (3, src),
                    (4, extra),
                    (5, Resource::Sampler(self.linear_clamp)),
                    (7, buf(ao_changed)),
                ],
            )
        };
        let none = Resource::Texture(self.dummy);
        let restore = device.create_bind_group(
            p.restore,
            0,
            &[
                (0, buf(frame[0])),
                (1, Resource::Texture(t.cache_depth)),
                (2, Resource::Texture(t.cache_aux)),
                (3, Resource::Texture(t.cache_hdr)),
                (4, none),
                (5, Resource::Sampler(self.linear_clamp)),
            ],
        )?;
        let reshade = upload(device, RESHADE_BYTES, BufferUsage::UNIFORM)?;
        let mask = device.create_bind_group(
            p.reshade_mask,
            0,
            &[
                (0, buf(frame[0])),
                (1, buf(self.scene.light_buffer)),
                (4, Resource::Texture(self.shadow_map)),
                (5, Resource::Sampler(self.shadow_sampler)),
                (14, Resource::Texture(self.light_shadow_map)),
                (15, Resource::Texture(t.cache_depth)),
                (16, Resource::Texture(t.cache_aux)),
                (17, buf(reshade)),
                (18, Resource::Texture(t.cache_shadow)),
            ],
        )?;
        let ao_grass_tile = post(device, p.ao_grass_tile, none, none)?;
        let ao_grass_union = post(
            device,
            p.ao_grass_union,
            Resource::Texture(t.ao_grass[0]),
            Resource::Texture(t.ao_grass[1]),
        )?;
        let ao_grass_coarse = post(
            device,
            p.ao_grass_coarse,
            Resource::Texture(t.ao_grass_union),
            none,
        )?;
        let ssao = post(
            device,
            p.ssao,
            Resource::Texture(t.ao_grass_coarse),
            Resource::Texture(t.ao_grass_union),
        )?;
        let air = post(
            device,
            p.air,
            Resource::Texture(t.hdr),
            Resource::Texture(t.ao_tex),
        )?;
        let mut down = vec![];
        let mut up = vec![];
        for level in 0..t.bloom_levels {
            let src = if level == 0 {
                Resource::Texture(t.lit)
            } else {
                Resource::TextureMip(t.bloom, level - 1)
            };
            down.push(post(device, p.bloom_down, src, none)?);
            if level + 1 < t.bloom_levels {
                up.push(post(
                    device,
                    p.bloom_up,
                    Resource::TextureMip(t.bloom, level + 1),
                    none,
                )?);
            }
        }
        let tonemap = post(
            device,
            p.tonemap,
            Resource::Texture(t.lit),
            Resource::TextureMip(t.bloom, 0),
        )?;
        // The fused output's groups, or `fs_grade`'s: the lit picture and its bloom.
        let lit = |d: &mut Device<'_>, pipes: [PipelineId; 2]| -> Result<[BindGroupId; 2]> {
            let [graded, plain] = pipes.map(|pipe| {
                let (lit, bloom) = (Resource::Texture(t.lit), Resource::TextureMip(t.bloom, 0));
                post(d, pipe, lit, bloom)
            });
            Ok([graded?, plain?])
        };
        let tonemap_output = lit(device, p.tonemap_output)?;
        let grade_once = match (t.graded, p.grade_once) {
            (Some(graded), Some((grade, read))) => Some((
                lit(device, grade)?,
                post(device, read, Resource::Texture(graded), none)?,
            )),
            _ => None,
        };
        let tonemap_final = device.create_bind_group(
            p.tonemap_final,
            0,
            &[
                (0, buf(frame[0])),
                (1, Resource::Texture(t.depth)),
                (2, Resource::Texture(t.aux)),
                (3, Resource::Texture(t.lit)),
                (4, Resource::TextureMip(t.bloom, 0)),
                (5, Resource::Sampler(self.linear_clamp)),
                (6, Resource::Texture(self.lut)),
            ],
        )?;
        let paint = post(device, p.paint, Resource::Texture(t.ldr), none)?;
        let final_src = if self.quality.painterly {
            t.paint
        } else {
            t.ldr
        };
        let output = post(device, p.output, Resource::Texture(final_src), none)?;
        Ok(Slot {
            frame,
            tiles,
            index,
            tiles_for: [None; 2],
            scene,
            sky,
            water,
            grass,
            flower,
            particle,
            restore,
            mask,
            reshade,
            ao_grass_tile,
            ao_grass_union,
            ao_grass_coarse,
            ssao,
            ao_changed,
            air,
            down,
            up,
            tonemap,
            tonemap_final,
            tonemap_output,
            grade_once,
            paint,
            output,
        })
    }

    pub(super) fn destroy_slots(device: &mut Device<'_>, slots: Vec<Slot>) {
        for s in slots {
            for g in s
                .scene
                .into_iter()
                .chain(s.sky)
                .chain(s.water.map(|w| w.0))
                .chain([
                    s.grass,
                    s.flower,
                    s.particle,
                    s.restore,
                    s.mask,
                    s.ao_grass_tile,
                    s.ao_grass_union,
                    s.ao_grass_coarse,
                    s.ssao,
                    s.air,
                ])
                .chain(s.tonemap_output)
                .chain(
                    s.grade_once
                        .into_iter()
                        .flat_map(|(grade, read)| grade.into_iter().chain([read])),
                )
                .chain([s.tonemap, s.tonemap_final, s.paint, s.output])
            {
                device.destroy_bind_group(g);
            }
            for g in s.down.into_iter().chain(s.up) {
                device.destroy_bind_group(g);
            }
            for b in s
                .frame
                .into_iter()
                .chain(s.tiles)
                .chain(s.index)
                .chain([s.reshade, s.ao_changed])
                .chain(s.water.map(|w| w.1))
            {
                device.destroy_buffer(b);
            }
        }
    }

    pub(super) fn destroy_targets(device: &mut Device<'_>, old: Targets) {
        for t in [
            old.hdr,
            old.aux,
            old.depth,
            old.cache_hdr,
            old.cache_aux,
            old.cache_depth,
            old.reshade_depth,
            old.cache_shadow,
            old.refl_hdr,
            old.refl_aux,
            old.refl_depth,
            old.ao_tex,
            old.ao_grass[0],
            old.ao_grass[1],
            old.ao_grass_union,
            old.ao_grass_coarse,
            old.lit,
            old.bloom,
            old.ldr,
            old.paint,
        ]
        .into_iter()
        .chain(old.graded)
        .chain(old.still)
        .chain(old.cache_water)
        {
            device.destroy_texture(t);
        }
        if let Some(b) = old.readback {
            device.destroy_buffer(b);
        }
    }

    /// Recreate size-dependent targets (e.g. after a window resize).
    pub fn resize(&mut self, device: &mut Device<'_>, size: (u32, u32)) -> Result<()> {
        if size == (self.targets.width, self.targets.height) {
            return Ok(());
        }
        self.rebuild_targets(device, size, self.grade_once())
    }

    /// Draws the fused tone-map output (no painterly filter between the tone-map and the output)
    /// as two passes: `fs_grade` grades each pixel once into a 32-bit float target, which holds
    /// the grade exactly, then `fs_tonemap_output` sharpens, smooths edges and encodes from
    /// there, where the fused pass grades the pixel and its four neighbours itself. The picture
    /// is the same (byte for byte on lavapipe); the GPU grades one pixel a pixel instead of five
    /// and moves 32 bytes a pixel more (the target written and read back): for a GPU whose
    /// arithmetic the five grades bind (Arm Mali). A device that cannot read 32-bit float
    /// textures (WebGPU without `float32-filterable`) keeps the fused pass. Returns whether
    /// frames now grade once.
    pub fn set_grade_once(&mut self, device: &mut Device<'_>, on: bool) -> Result<bool> {
        let on = on && device.limits().float32_readable;
        if on != self.grade_once() {
            if on && self.pipelines.grade_once.is_none() {
                let output = self.pipelines.output_format;
                self.pipelines.grade_once = Some(Pipelines::grade_once(device, output)?);
            }
            self.rebuild_targets(device, self.size(), on)?;
        }
        Ok(on)
    }

    /// Frames grade each pixel once ([`Renderer::set_grade_once`]).
    pub fn grade_once(&self) -> bool {
        self.targets.graded.is_some()
    }

    /// Makes the size-dependent targets and the frame slots binding them again.
    pub(super) fn rebuild_targets(
        &mut self,
        device: &mut Device<'_>,
        size: (u32, u32),
        grade_once: bool,
    ) -> Result<()> {
        let new_targets =
            Self::create_targets(device, &self.quality, self.output, size, grade_once)?;
        self.world_cache = None;
        self.reset_prop_slots(device);
        Self::destroy_slots(device, std::mem::take(&mut self.slots));
        let old = std::mem::replace(&mut self.targets, new_targets);
        Self::destroy_targets(device, old);
        let slots = match self.output {
            Output::Still => 1,
            Output::Surface(_) => 2,
        };
        for _ in 0..slots {
            let s = self.create_slot(device)?;
            self.slots.push(s);
        }
        Ok(())
    }

    /// Reconfigure quality-dependent targets, shadow maps, and ground/world caches at runtime
    /// without losing uploaded world geometry, props, skinned models, lights, or [`Look`] state.
    pub fn set_quality(&mut self, device: &mut Device<'_>, quality: Quality) -> Result<()> {
        if quality == self.quality {
            return Ok(());
        }
        let size = (self.targets.width, self.targets.height);
        let new_targets =
            Self::create_targets(device, &quality, self.output, size, self.grade_once())?;
        self.world_cache = None;

        let old_shadow_size = self.quality.shadow_size.max(16);
        let new_shadow_size = quality.shadow_size.max(16);
        let new_shadow_map = if new_shadow_size != old_shadow_size {
            Some(device.create_texture(&TextureDesc {
                width: new_shadow_size,
                height: new_shadow_size,
                layers: 4,
                mips: 1,
                format: Format::Depth32Float,
                usage: TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED,
            })?)
        } else {
            None
        };

        let old_light_shadow_size = self.quality.light_shadow_size.max(16);
        let new_light_shadow_size = quality.light_shadow_size.max(16);
        let new_light_shadow_map = if new_light_shadow_size != old_light_shadow_size {
            Some(device.create_texture(&TextureDesc {
                width: new_light_shadow_size,
                height: new_light_shadow_size,
                layers: crate::render::lights::MAX_SHADOWED_LIGHTS as u32,
                mips: 1,
                format: Format::Depth32Float,
                usage: TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED,
            })?)
        } else {
            None
        };

        let old_ground_size = self.quality.ground_size.max(64);
        let new_ground_size = quality.ground_size.max(64);
        let new_ground = if new_ground_size != old_ground_size {
            let depth = device.create_texture(&TextureDesc {
                width: new_ground_size,
                height: new_ground_size,
                layers: 1,
                mips: 1,
                format: Format::Depth32Float,
                usage: TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED,
            })?;
            let mask = device.create_texture(&TextureDesc {
                width: new_ground_size,
                height: new_ground_size,
                layers: 1,
                mips: 1,
                format: GROUND_FORMAT,
                usage: TextureUsage::COLOR_TARGET | TextureUsage::SAMPLED,
            })?;
            Some((depth, mask))
        } else {
            None
        };

        let old_world_size = self.quality.world_size.max(64);
        let new_world_size = quality.world_size.max(64);
        let new_world = if new_world_size != old_world_size {
            let depth = device.create_texture(&TextureDesc {
                width: new_world_size,
                height: new_world_size,
                layers: 1,
                mips: 1,
                format: Format::Depth32Float,
                usage: TextureUsage::DEPTH_TARGET | TextureUsage::SAMPLED,
            })?;
            let mask = device.create_texture(&TextureDesc {
                width: new_world_size,
                height: new_world_size,
                layers: 1,
                mips: 1,
                format: GROUND_FORMAT,
                usage: TextureUsage::COLOR_TARGET | TextureUsage::SAMPLED,
            })?;
            Some((depth, mask))
        } else {
            None
        };

        self.reset_prop_slots(device);
        Self::destroy_slots(device, std::mem::take(&mut self.slots));
        let old_targets = std::mem::replace(&mut self.targets, new_targets);
        Self::destroy_targets(device, old_targets);

        if let Some(shadow_map) = new_shadow_map {
            let old = std::mem::replace(&mut self.shadow_map, shadow_map);
            device.destroy_texture(old);
            self.shadow_layers_ready = false;
            self.static_shadows_vp = None;
        }
        if let Some(light_shadow_map) = new_light_shadow_map {
            let old = std::mem::replace(&mut self.light_shadow_map, light_shadow_map);
            device.destroy_texture(old);
            self.light_shadows_ready = false;
        }
        if let Some((depth, mask)) = new_ground {
            let old_depth = std::mem::replace(&mut self.ground_depth, depth);
            let old_mask = std::mem::replace(&mut self.ground_mask, mask);
            device.destroy_texture(old_depth);
            device.destroy_texture(old_mask);
            self.ground_cached_vp = None;
        }
        if let Some((depth, mask)) = new_world {
            let old_depth = std::mem::replace(&mut self.world_depth, depth);
            let old_mask = std::mem::replace(&mut self.world_mask, mask);
            device.destroy_texture(old_depth);
            device.destroy_texture(old_mask);
            self.world = world_view(&self.scene, new_world_size);
            self.world_drawn = false;
        }

        self.quality = quality;
        let slots = match self.output {
            Output::Still => 1,
            Output::Surface(_) => 2,
        };
        for _ in 0..slots {
            let s = self.create_slot(device)?;
            self.slots.push(s);
        }
        Ok(())
    }

    pub(in crate::render) fn eye(&self, view: &View, jitter: [f32; 2], (w, h): (u32, u32)) -> Eye {
        let v = look_at(view.eye, view.target, camera_up(view));
        let mut p = perspective(view.fov.to_radians(), w as f32 / h as f32, near(view));
        // Column 2 skews the frustum: x_ndc gains -p[8] and y_ndc (pointing down) -p[9].
        p[8] += 2.0 * jitter[0] / w as f32 - view.shift[0];
        p[9] += 2.0 * jitter[1] / h as f32 + view.shift[1];
        let view_proj = mul(&p, &v);
        Eye {
            view_proj,
            inv: invert(&view_proj).unwrap_or(math::IDENTITY),
            pos: view.eye,
            near: near(view),
        }
    }

    /// The ground map's top-down orthographic projection: centred on the view target (at most
    /// 40 m ahead of the eye, so the camera is always inside the map), depth 0 above the
    /// highest point of the world and 1 below the lowest. The centre snaps to whole texels so
    /// grass, rain cover and foam do not shimmer as the camera moves.
    pub(super) fn ground_view(&self, view: &View) -> Ground {
        let to = sub(view.target, view.eye);
        let d = math::length(to);
        let centre = if d <= 40.0 {
            view.target
        } else {
            add(view.eye, scale(to, 40.0 / d))
        };
        let size = self.quality.ground_size.max(64);
        let texel = 2.0 * GROUND_REACH / size as f32;
        let (x, z) = (
            (centre[0] / texel).round() * texel,
            (centre[2] / texel).round() * texel,
        );
        let top = self.scene.max[1] + 1.0;
        let range = (top - self.scene.min[1] + 1.0).max(2.0);
        let v = look_at([x, top, z], [x, top - 1.0, z], [0.0, 0.0, -1.0]);
        let r = GROUND_REACH;
        Ground {
            vp: mul(&ortho(-r, r, -r, r, 0.0, range), &v),
            top,
            range,
            size,
        }
    }

    pub(super) fn cascades(&self, view: &View, env: &Environment) -> Cascades {
        let (w, h) = (self.targets.width as f32, self.targets.height as f32);
        let focus = math::length(sub(view.target, view.eye)).max(1.0);
        let split = (focus * 1.5).clamp(18.0, 140.0);
        let far = (focus * 4.5).clamp(90.0, 420.0);
        let l = normalize(env.key_dir);
        let size = self.quality.shadow_size as f32;
        let fwd = normalize(sub(view.target, view.eye));
        let right = normalize(cross(fwd, camera_up(view)));
        let up = cross(right, fwd);
        let ty = (0.5 * view.fov.to_radians()).tan();
        let tx = ty * w / h;
        let [shift_x, shift_y] = view.shift;
        let mut out = Cascades {
            mats: [math::IDENTITY; 2],
            split,
            offset: [0.0; 2],
            radius: [0.0; 2],
            bias: [0.0; 2],
        };
        // Water casts no shadow, so a wide sea does not stretch the depth range.
        let (mut smin, mut smax) = (self.scene.land_min, self.scene.land_max);
        if let Some((lo, hi)) = self.props_bounds() {
            for k in 0..3 {
                smin[k] = smin[k].min(lo[k]);
                smax[k] = smax[k].max(hi[k]);
            }
        }
        for (c, (d0, d1)) in [(near(view), split), (split * 0.85, far)]
            .into_iter()
            .enumerate()
        {
            let mut pts = vec![];
            for d in [d0, d1] {
                for sx in [-1.0, 1.0] {
                    for sy in [-1.0, 1.0] {
                        pts.push(add(
                            add(view.eye, scale(fwd, d)),
                            add(
                                scale(right, (sx - shift_x) * d * tx),
                                scale(up, (sy - shift_y) * d * ty),
                            ),
                        ));
                    }
                }
            }
            let centre = scale(pts.iter().fold([0.0; 3], |a, p| add(a, *p)), 1.0 / 8.0);
            let radius = pts
                .iter()
                .map(|p| math::length(sub(*p, centre)))
                .fold(0.0f32, f32::max)
                .max(1.0);
            let up_hint = if l[1].abs() > 0.98 {
                [0.0, 0.0, 1.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let back = 1000.0;
            // The light's view is anchored at the world origin, so it only changes when the
            // light moves. The cascade centre is then snapped to whole texels in that frame,
            // and the radius to a fixed step, so shadows don't crawl as the camera moves.
            let lv = look_at(scale(l, back), [0.0; 3], up_hint);
            let radius = (radius * 2.0).ceil() * 0.5;
            let texel = 2.0 * radius / size;
            let lc = transform(&lv, [centre[0], centre[1], centre[2], 1.0]);
            let snap = |x: f32| (x / texel).round() * texel - x;
            let (dx, dy) = (snap(lc[0]), snap(lc[1]));
            // Depth range from the scene bounds along the light.
            let (mut zmin, mut zmax) = (f32::INFINITY, f32::NEG_INFINITY);
            for i in 0..8 {
                let p = [
                    if i & 1 == 0 { smin[0] } else { smax[0] },
                    if i & 2 == 0 { smin[1] } else { smax[1] },
                    if i & 4 == 0 { smin[2] } else { smax[2] },
                ];
                let z = -transform(&lv, [p[0], p[1], p[2], 1.0])[2];
                zmin = zmin.min(z);
                zmax = zmax.max(z);
            }
            let near = (zmin - 5.0).max(1.0);
            let farz = (zmax + 5.0).max(near + 1.0);
            let proj = ortho(
                lc[0] + dx - radius,
                lc[0] + dx + radius,
                lc[1] + dy - radius,
                lc[1] + dy + radius,
                near,
                farz,
            );
            out.mats[c] = mul(&proj, &lv);
            out.offset[c] = texel * 1.5;
            let filter_texels = if c == 0 { 1.6 } else { 1.1 };
            out.radius[c] = filter_texels * env.shadow_softness / size;
            out.bias[c] = (texel * 0.6) / (farz - near);
        }
        out
    }

    /// Light lists per screen tile for one camera. Lights are sorted nearest first so a full
    /// tile keeps the lights that matter most.
    /// The shadow maps props cast into, as `fs_reshade_mask` numbers them (0, 1 the key light's
    /// cascades; 2, 3 a game's lights), with their view-projection and margins around a
    /// footprint (clip units: the filter and a few texels) and around a receiver (metres: the
    /// key light's receivers sample from a point off their surface, `LOOKUP_SLACK`).
    pub(super) fn prop_shadow_maps(&self, cas: &Cascades) -> [(M4, f32, f32); 4] {
        let texel = 2.0 / self.quality.shadow_size.max(16) as f32;
        let light_texel = 2.0 / self.quality.light_shadow_size.max(16) as f32;
        let lights = self.game_lights.shadow_views();
        std::array::from_fn(|v| match v {
            0 | 1 => (
                cas.mats[v],
                2.0 * cas.radius[v] + 4.0 * texel,
                2.5 * LOOKUP_SLACK * cas.offset[v],
            ),
            _ => (
                lights.get(v - 2).copied().unwrap_or(math::IDENTITY),
                6.0 * light_texel,
                0.0,
            ),
        })
    }

    /// Where shadows of `moved` (boxes where props were and are) fall in each shadow map props
    /// cast into ([`Renderer::prop_shadow_maps`]) that holds them: the key light's cascades when
    /// `key` (it shines and they hold props), a game's shadowed lights'. Boxes too many for the
    /// re-shade block are joined into one.
    pub(super) fn shadow_footprints(
        &self,
        moved: &[(V3, V3)],
        cas: &Cascades,
        key: bool,
    ) -> Vec<Footprint> {
        let maps = self.prop_shadow_maps(cas);
        let lights = self.game_lights.shadow_views().len().min(2);
        let active: Vec<usize> = (0..4)
            .filter(|&v| if v < 2 { key } else { v - 2 < lights })
            .collect();
        let boxes = if moved.len() * active.len() > MAX_RESHADE_ENTRIES {
            let all = moved.iter().copied().reduce(|a, b| {
                (
                    std::array::from_fn(|k| a.0[k].min(b.0[k])),
                    std::array::from_fn(|k| a.1[k].max(b.1[k])),
                )
            });
            all.into_iter().collect()
        } else {
            moved.to_vec()
        };
        let mut footprints = vec![];
        for b in &boxes {
            for &v in &active {
                let (m, margin, _) = &maps[v];
                let (r, near, _) = map_bounds(*b, m);
                let r = [r[0] - margin, r[1] - margin, r[2] + margin, r[3] + margin];
                footprints.push((v, r, near));
            }
        }
        footprints
    }

    /// How far a lookup in each shadow map props cast into ([`Renderer::prop_shadow_maps`]) can
    /// lie from the point it shades, in the map's clip units across and in depth: the key light's
    /// within its receivers' margin, through its cascade's scale; a game light's footprints are
    /// tested at the point itself.
    pub(super) fn lookup_margins(maps: &[(M4, f32, f32); 4]) -> [[f32; 2]; 4] {
        std::array::from_fn(|v| {
            let (m, _, grow) = &maps[v];
            let scale = |r: usize| grow * math::length([m[r], m[4 + r], m[8 + r]]);
            [scale(0).max(scale(1)), scale(2)]
        })
    }

    /// What the GPU needs to shade again the cached picture's pixels that moved props' shadows
    /// (`footprints`, which `fs_reshade_mask` tests per pixel) can reach: the world's draws a
    /// footprint can reach, and the region of the picture holding where they and the settled
    /// props can see one ([`shadow_volume`], for `budget` pairs of a receiver and a footprint at
    /// most, [`MAX_RESHADE_PAIRS`]). None when nothing the picture holds is in reach.
    pub(super) fn reshade_plan(
        &self,
        footprints: &[Footprint],
        settled: &[(V3, V3)],
        eye: &Eye,
        size: (u32, u32),
        cas: &Cascades,
        budget: u32,
    ) -> Option<ReshadePlan> {
        let receivers = self.receivers.as_ref().filter(|_| !footprints.is_empty())?;
        let maps = self.prop_shadow_maps(cas);
        let (m0, _, grow0) = &maps[0];
        // The region is bounded with each map's footprints as one (the rect holding theirs, from
        // the nearest of their depths), which every point that sees one of them sees: a receiver
        // costs a pair a map however many props moved, and the region holds a little more where
        // they lie apart.
        let bounding: Vec<Footprint> = (0..4)
            .filter_map(|v| {
                let of_map = footprints.iter().filter(|f| f.0 == v).copied();
                of_map.reduce(|(_, a, near), (_, b, other)| (v, union_rect(a, b), near.min(other)))
            })
            .collect();
        let mut region: Option<Rect> = None;
        let (mut pairs, mut points) = (0, Vec::with_capacity(48));
        // Grows `region` by the pixels of a receiver in box `b` (lying within `screen`; `bounds`
        // in the maps) where it can see a footprint: the projection of the points that can,
        // rounded out by half a pixel (a pixel is tested at its centre) and half a pixel more for
        // rounding. Past the budget, by all of `screen`.
        let mut add = |region: &mut Option<Rect>, b, bounds: &[([f32; 4], f32); 4], screen| {
            for f in &bounding {
                // The region already holds all of the receiver's pixels.
                if region.is_some_and(|a| union(a, screen) == a) {
                    return;
                }
                if !reached(std::slice::from_ref(f), bounds) {
                    continue;
                }
                let r = if pairs == budget {
                    Some(screen)
                } else {
                    pairs += 1;
                    shadow_volume(b, f, &maps, &mut points);
                    // Where the key light's far cascade is not looked up, its footprints change
                    // nothing.
                    if f.0 == 1 && near_cascade_only(&points, eye.pos, cas.split, m0, *grow0) {
                        continue;
                    }
                    screen_rect(&points, &eye.view_proj, size, 1.0)
                        .and_then(|r| intersect(r, screen))
                };
                if let Some(r) = r {
                    *region = Some(region.map_or(r, |a| union(a, r)));
                }
            }
        };
        // Every draw in reach is drawn again, as the region holds where they can see a
        // footprint: the mask may mark a pixel of one where nothing changed (as where only the
        // far cascade's footprint holds its lookups), which must find what it held.
        let draws: Vec<bool> = self
            .scene
            .draws
            .iter()
            .zip(&receivers.maps)
            .zip(&receivers.screen)
            .map(|((d, m), screen)| {
                let hit = reached(footprints, m);
                if let (true, Some(s)) = (hit, *screen) {
                    add(&mut region, d.bounds, m, s);
                }
                hit
            })
            .collect();
        // The settled props' bounds in the maps the footprints lie in, and their pictures where
        // those reach them, until the region is all of the picture.
        let used: [bool; 4] = std::array::from_fn(|v| footprints.iter().any(|f| f.0 == v));
        for b in settled {
            if region == Some([0, 0, size.0, size.1]) {
                break;
            }
            let bounds = std::array::from_fn(|v| {
                let (m, _, grow) = &maps[v];
                if !used[v] {
                    return OUT_OF_REACH;
                }
                let (r, _, far) = map_bounds((b.0.map(|x| x - grow), b.1.map(|x| x + grow)), m);
                (r, far)
            });
            if !reached(footprints, &bounds) {
                continue;
            }
            if let Some(screen) = screen_rect(&corners(*b), &eye.view_proj, size, 2.0) {
                add(&mut region, *b, &bounds, screen);
            }
        }
        let region = region?;
        // The light buffer index (+ 1) of a game's light casting into each shadow layer.
        let lamps = self.scene.lights.len() as u32;
        let mut shadowed = [0u32; 2];
        for (k, r) in self.game_lights.records().iter().enumerate() {
            if let Some(slot) = shadowed.get_mut((r[14] as usize).wrapping_sub(1)) {
                *slot = lamps + k as u32 + 1;
            }
        }
        Some(ReshadePlan {
            block: reshade_block(footprints, shadowed, Self::lookup_margins(&maps)),
            draws,
            region,
        })
    }
}
