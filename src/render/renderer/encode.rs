use super::*;

impl Renderer {
    /// What a view's light tiles ([`Renderer::tile_lights`]) depend on: the eye, the picture and
    /// tile sizes, whether the lamps are lit, and where the world's lamps and the game's lights
    /// stand.
    pub(super) fn tile_key(&self, eye: &Eye, size: (u32, u32), tiles: (u32, u32), on: bool) -> u64 {
        let mut key = Fnv::new();
        key.write(&f32_bytes(&eye.view_proj));
        key.write(&f32_bytes(&[eye.pos[0], eye.pos[1], eye.pos[2], eye.near]));
        for n in [size.0, size.1, tiles.0, tiles.1, u32::from(on)] {
            key.write(&n.to_le_bytes());
        }
        key.write(&self.quality.tile_size.to_le_bytes());
        key.write(&self.scene.revision.to_le_bytes());
        for record in self.game_lights.records() {
            key.write(&f32_bytes(record));
        }
        key.finish()
    }

    pub(super) fn tile_lights(
        &self,
        eye: &Eye,
        (w, h): (u32, u32),
        (tx, ty): (u32, u32),
        on: bool,
    ) -> (Vec<u32>, Vec<u32>) {
        let ts = self.quality.tile_size as f32;
        let mut lists: Vec<Vec<u32>> = vec![vec![]; (tx * ty * CLUSTER_SLICES) as usize];
        // The lamps while they are lit, and a game's lights (after them in the light buffer):
        // those that follow the lamps while they are lit, the rest always.
        let lamps = self.scene.lights.len();
        let game = self.game_lights.records();
        let mut order: Vec<usize> = (0..lamps)
            .filter(|_| on)
            .chain(
                game.iter()
                    .enumerate()
                    .filter(|(_, l)| on || l[13] > 0.5)
                    .map(|(k, _)| lamps + k),
            )
            .collect();
        let light = |i: usize| -> &[f32; 16] {
            if i < lamps {
                &self.scene.lights[i]
            } else {
                &game[i - lamps]
            }
        };
        if !order.is_empty() {
            let dist = |i: usize| {
                let l = light(i);
                math::length(sub([l[0], l[1], l[2]], eye.pos))
            };
            order.sort_by(|a, b| dist(*a).total_cmp(&dist(*b)));
            for i in order {
                let l = light(i);
                let r = l[3];
                let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
                let mut behind = 0;
                let mut straddle = false;
                for k in 0..8 {
                    let p = [
                        l[0] + if k & 1 == 0 { -r } else { r },
                        l[1] + if k & 2 == 0 { -r } else { r },
                        l[2] + if k & 4 == 0 { -r } else { r },
                    ];
                    let c = transform(&eye.view_proj, [p[0], p[1], p[2], 1.0]);
                    if c[3] <= eye.near * 0.5 {
                        behind += 1;
                        straddle = true;
                        continue;
                    }
                    let (sx, sy) = (c[0] / c[3], c[1] / c[3]);
                    x0 = x0.min(sx);
                    x1 = x1.max(sx);
                    y0 = y0.min(sy);
                    y1 = y1.max(sy);
                }
                if behind == 8 {
                    continue;
                }
                if straddle {
                    (x0, y0, x1, y1) = (-1.0, -1.0, 1.0, 1.0);
                }
                if x1 < -1.0 || y1 < -1.0 || x0 > 1.0 || y0 > 1.0 {
                    continue;
                }
                let px = |v: f32, size: u32| ((v * 0.5 + 0.5) * size as f32 / ts).floor();
                let ax = px(x0.max(-1.0), w).max(0.0) as u32;
                let bx = (px(x1.min(1.0), w) as u32).min(tx - 1);
                let ay = px(y0.max(-1.0), h).max(0.0) as u32;
                let by = (px(y1.min(1.0), h) as u32).min(ty - 1);
                // The slices of view depth the sphere spans (clip w is the view depth).
                let centre = transform(&eye.view_proj, [l[0], l[1], l[2], 1.0])[3];
                let s0 = if straddle {
                    0
                } else {
                    cluster_slice(centre - r)
                };
                let s1 = cluster_slice(centre + r);
                for s in s0..=s1 {
                    for y in ay..=by {
                        for x in ax..=bx {
                            let list = &mut lists[((s * ty + y) * tx + x) as usize];
                            if list.len() < MAX_LIGHTS_PER_CLUSTER {
                                list.push(i as u32);
                            }
                        }
                    }
                }
            }
        }
        let mut tiles = Vec::with_capacity(lists.len() * 2);
        let mut index = vec![];
        // The index buffer holds `CLUSTER_ENTRIES_PER_TILE` a tile. When the lists hold more,
        // every cluster keeps the same number of its nearest lamps (each list was filled nearest
        // first), as many as fit, rather than the last clusters losing all of theirs.
        let room = (u64::from(tx) * u64::from(ty) * CLUSTER_ENTRIES_PER_TILE) as usize;
        let cap = cluster_cap(lists.iter().map(Vec::len), room);
        for list in lists {
            let kept = list.len().min(cap);
            tiles.push(index.len() as u32);
            tiles.push(kept as u32);
            index.extend_from_slice(&list[..kept]);
        }
        if index.is_empty() {
            index.push(0);
        }
        (tiles, index)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn frame_block(
        &self,
        eye: &Eye,
        env: &Environment,
        cas: &Cascades,
        ground: &Ground,
        (w, h): (u32, u32),
        tiles: (u32, u32),
        reflection: bool,
        has_prop_shadows: bool,
        has_reflection: bool,
        seed: u32,
    ) -> Vec<u8> {
        let l = &self.look;
        let mut b = Block(Vec::with_capacity(248));
        b.m(&eye.view_proj);
        b.m(&eye.inv);
        b.m(&cas.mats[0]);
        b.m(&cas.mats[1]);
        b.m(&ground.vp);
        b.v(eye.pos, env.seconds);
        b.v(normalize(env.key_dir), env.key_intensity);
        b.v(env.key_color, 0.0);
        b.v(scale(env.ambient_sky, env.ambient), env.lamp_level);
        // Ground ambient plus the key light bounced up off sunlit ground.
        let bounce = scale(env.key_color, env.key_intensity * l.bounce);
        b.v(
            add(scale(env.ambient_ground, env.ambient), bounce),
            env.wetness,
        );
        b.v(l.shadow_tint, l.shadow_strength);
        b.f(
            l.ramp_center,
            l.ramp_softness.max(0.002),
            l.rim_strength,
            l.rim_power,
        );
        b.f(
            l.specular,
            l.specular_size,
            l.foliage_wrap,
            l.foliage_translucency,
        );
        b.f(
            l.texture_softness,
            l.normal_map_strength,
            l.lamp_light,
            l.lamp_glow,
        );
        b.f(
            self.scene.water_level.unwrap_or(-1e6),
            l.water_reflection,
            l.water_ripples,
            env.rain,
        );
        b.v(l.water_color, if reflection { 1.0 } else { 0.0 });
        b.v(env.zenith, env.stars);
        b.v(env.horizon, env.clouds);
        b.v(env.glow, env.sun_disc);
        b.v(normalize(env.glow_dir), env.moon_disc);
        b.v(normalize(env.sun_dir), l.cloud_scale);
        b.v(normalize(env.moon_dir), l.moon_size);
        b.v(env.cloud_lit, l.star_density);
        b.v(env.cloud_shadow, env.petals);
        b.v(env.fog_color, env.fog_density);
        let fog_base = l.fog_height + self.scene.water_level.unwrap_or(0.0);
        b.f(fog_base, l.fog_falloff, env.mist, l.fog);
        b.f(env.wind[0], env.wind[1], 1.0, seed as f32);
        b.f(w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32);
        b.f(cas.split, cas.offset[0], cas.offset[1], eye.near);
        b.f(cas.radius[0], cas.radius[1], cas.bias[0], cas.bias[1]);
        b.f(
            ground.top,
            ground.range,
            ground.size as f32,
            l.grass.density,
        );
        b.f(l.brush, l.brush_scale.max(0.01), l.sun_shafts, 0.0);
        b.v(l.water_shallow, l.foam);
        b.f(
            l.sparkle,
            self.grass_side() as f32,
            if has_prop_shadows { 1.0 } else { 0.0 },
            if has_reflection { 1.0 } else { 0.0 },
        );
        let g = &l.grass;
        b.f(g.height, g.width, g.flowers, g.sway);
        b.v(g.base, 0.0);
        b.v(g.tip, 0.0);
        for c in g.flower_colors {
            b.v(c, 0.0);
        }
        let w = &self.world;
        b.m(&w.vp);
        b.f(
            w.top,
            w.range,
            w.size as f32,
            if self.world_drawn { 1.0 } else { 0.0 },
        );
        let t = &l.terrain;
        b.f(t.beach, t.rock_slope[0], t.rock_slope[1], t.patch_size);
        b.v(t.sand, t.patches);
        b.v(t.wet_sand, 0.0);
        b.v(t.rock, 0.0);
        b.v(t.meadow, 0.0);
        b.f(
            l.lamp_foliage,
            l.ambient_occlusion,
            CLUSTER_NEAR,
            cluster_scale(),
        );
        let tide = if self.scene.water_level.is_some() {
            l.tide
        } else {
            0.0
        };
        let [r, g, bl] = l.tide_color;
        b.f(tide, r, g, bl);
        let mut bytes = f32_bytes(&b.0);
        for u in [
            tiles.0,
            tiles.1,
            self.quality.tile_size,
            (self.scene.lights.len() + self.game_lights.records().len()) as u32,
        ] {
            bytes.extend_from_slice(&u.to_le_bytes());
        }
        // The game lights' shadow maps and unified surface model parameters.
        let mut b = Block(Vec::with_capacity(44));
        let views = self.game_lights.shadow_views();
        for layer in 0..crate::render::lights::MAX_SHADOWED_LIGHTS {
            b.m(views.get(layer).unwrap_or(&math::IDENTITY));
        }
        b.f(
            1.0 / self.quality.light_shadow_size.max(16) as f32,
            0.0,
            0.0,
            0.0,
        );
        b.f(
            l.surface_model as u32 as f32,
            l.clearcoat,
            l.sheen,
            l.subsurface,
        );
        b.v(l.subsurface_color, l.clearcoat_roughness);
        let [x, z, w, _] = self.grass_grid;
        b.f(x as f32, z as f32, w as f32, 0.0);
        bytes.extend_from_slice(&f32_bytes(&b.0));
        debug_assert!(bytes.len() as u64 <= FRAME_BYTES);
        bytes
    }

    /// Grass instances to draw: the part of the instance grid (scaled by quality and the Look's
    /// density) the view can show, or 0 when no material grows grass.
    pub(super) fn grass_instances(&self) -> u32 {
        self.grass_grid[2] * self.grass_grid[3]
    }

    /// The part of the grass grid `vs_grass` can show: its cells (offset from the grid's corner,
    /// then its size) whose blades can pass the view test in `grass_site`. The grid's box, from the
    /// lowest ground to the tallest blade, clipped to the view widened as that test widens it; the
    /// blades outside it are dropped by the test anyway, so the picture is the same, without
    /// running the vertex stage for hundreds of thousands of blades the view cannot show.
    pub(super) fn visible_grass(&self, eye: &Eye, ground: &Ground) -> [u32; 4] {
        let side = self.grass_side();
        if side == 0 {
            return [0; 4];
        }
        let s = f64::from(GRASS_SPACING);
        let half = i64::from(side / 2);
        let corner = [
            (f64::from(eye.pos[0]) / s).floor() as i64 - half,
            (f64::from(eye.pos[2]) / s).floor() as i64 - half,
        ];
        let lo = [
            corner[0] as f64 * s,
            f64::from(ground.top - ground.range) - 1.0,
            corner[1] as f64 * s,
        ];
        let hi = [
            (corner[0] + i64::from(side)) as f64 * s,
            f64::from(ground.top) + 1.5 * f64::from(self.look.grass.height) + 1.0,
            (corner[1] + i64::from(side)) as f64 * s,
        ];
        // Half-spaces n . p + d <= 0: the box, then the test (|x|, |y| <= 1.15 w + 0.5, w > 0).
        let m = &eye.view_proj;
        let row = |k: usize| [m[k], m[4 + k], m[8 + k], m[12 + k]].map(f64::from);
        let (r0, r1, r3) = (row(0), row(1), row(3));
        let mut planes: Vec<([f64; 3], f64)> = vec![];
        for k in 0..3 {
            let mut n = [0.0; 3];
            n[k] = -1.0;
            planes.push((n, lo[k]));
            n[k] = 1.0;
            planes.push((n, -hi[k]));
        }
        for (r, sign) in [(r0, 1.0), (r0, -1.0), (r1, 1.0), (r1, -1.0)] {
            let a: [f64; 4] = std::array::from_fn(|i| sign * r[i] - 1.15 * r3[i]);
            planes.push(([a[0], a[1], a[2]], a[3] - 0.5));
        }
        planes.push(([-r3[0], -r3[1], -r3[2]], -r3[3]));
        // The clipped box's corners: where three planes meet inside all the others.
        let (mut xz_lo, mut xz_hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for i in 0..planes.len() {
            for j in i + 1..planes.len() {
                for k in j + 1..planes.len() {
                    let (a, b, c) = (planes[i], planes[j], planes[k]);
                    let det = |x: [f64; 3], y: [f64; 3], z: [f64; 3]| {
                        x[0] * (y[1] * z[2] - y[2] * z[1]) - x[1] * (y[0] * z[2] - y[2] * z[0])
                            + x[2] * (y[0] * z[1] - y[1] * z[0])
                    };
                    let d = det(a.0, b.0, c.0);
                    if d.abs() < 1e-12 {
                        continue;
                    }
                    let rhs = [-a.1, -b.1, -c.1];
                    let col = |q: usize| {
                        let pick = |p: ([f64; 3], f64), r: f64| {
                            let mut n = p.0;
                            n[q] = r;
                            n
                        };
                        det(pick(a, rhs[0]), pick(b, rhs[1]), pick(c, rhs[2])) / d
                    };
                    let p = [col(0), col(1), col(2)];
                    let inside = planes.iter().all(|(n, e)| {
                        n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + e
                            <= 1e-6 * (1.0 + n.iter().map(|x| x.abs()).sum::<f64>() * 100.0)
                    });
                    if inside {
                        xz_lo = [xz_lo[0].min(p[0]), xz_lo[1].min(p[2])];
                        xz_hi = [xz_hi[0].max(p[0]), xz_hi[1].max(p[2])];
                    }
                }
            }
        }
        if xz_lo[0] > xz_hi[0] {
            return [0; 4];
        }
        let cells = |v: f64, axis: usize| {
            ((v / s).floor() as i64 - corner[axis]).clamp(0, i64::from(side)) as u32
        };
        let (x0, z0) = (cells(xz_lo[0], 0), cells(xz_lo[1], 1));
        let (x1, z1) = (cells(xz_hi[0], 0) + 1, cells(xz_hi[1], 1) + 1);
        let (x1, z1) = (x1.min(side), z1.min(side));
        [x0, z0, x1.saturating_sub(x0), z1.saturating_sub(z0)]
    }

    /// Rows of the grass instance grid this frame (0 when nothing grows grass).
    pub(super) fn grass_side(&self) -> u32 {
        let g = &self.look.grass;
        if g.density <= 0.0 || self.quality.grass <= 0.0 || !self.grassy.iter().any(|&x| x) {
            return 0;
        }
        ((GRASS_SIDE as f32 * self.quality.grass.min(1.0).sqrt()).round() as u32 / 2 * 2).max(2)
    }

    /// A ground (or world) map: the highest solid surfaces seen from above through `vp`, and
    /// which of them grow grass. Roofs and solid things count; foliage, alpha-masked cut-outs
    /// and water do not (rain falls through trees, grass grows under them). On terrain, beach
    /// sand and bare rock grow no grass (see `fs_ground`).
    pub(super) fn draw_ground(
        &self,
        enc: &mut Encoder,
        depth: TextureId,
        mask: TextureId,
        vp: &M4,
    ) {
        enc.begin_pass(PassDesc {
            colors: vec![Attachment::clear(mask, [0.0; 4])],
            depth: Some(Attachment::clear(depth, [1.0; 4])),
        });
        enc.set_pipeline(self.pipelines.ground);
        enc.set_vertex_buffer(self.scene.vertices, 0);
        enc.set_index_buffer(self.scene.indices, 0);
        let t = &self.look.terrain;
        let mut gp = [0.0f32; 24];
        gp[..16].copy_from_slice(vp);
        gp[17] = 1.0;
        gp[19] = self.scene.water_level.unwrap_or(-1.0e6);
        gp[20..].copy_from_slice(&[t.beach, t.rock_slope[0], t.rock_slope[1], t.patch_size]);
        let terrain = DetailPreset::Terrain as u32 as f32;
        let frustum = Frustum::from_view_proj(vp);
        for d in self.scene.draws.iter() {
            let foliage = self.scene.materials[d.material].params[3] == 1.0;
            if d.water || foliage || self.shadow_materials[d.material].is_some() {
                continue;
            }
            if !frustum.intersects_aabb_clamped(d.bounds.0, d.bounds.1) {
                continue;
            }
            gp[16] = if self.grassy[d.material] { 1.0 } else { 0.0 };
            gp[18] = if self.details[d.material][0] == terrain {
                1.0
            } else {
                0.0
            };
            enc.push_constants(&f32_bytes(&gp));
            enc.draw_indexed(d.count, 1, d.first_index, 0, 0);
        }
        enc.end_pass();
    }

    /// Draws visible static world chunks against `eye`, returning `(total, submitted)` chunk counts.
    /// Draws the visible world draws of `part`: all of them in scene order, or (for the main view)
    /// the opaque ones nearest first and then the rest. The caller binds the pipeline and frame
    /// group for the part.
    pub(super) fn draw_world(
        &self,
        enc: &mut Encoder,
        model_push: &mut [f32; 32],
        eye: &Eye,
        water: bool,
        part: WorldPart,
        only: Option<&[bool]>,
    ) -> (u32, u32) {
        enc.set_vertex_buffer(self.scene.vertices, 0);
        enc.set_index_buffer(self.scene.indices, 0);
        let frustum = Frustum::from_view_proj(&eye.view_proj);
        let hybrid = self.quality.ao_samples == 0 && !self.quality.painterly;
        let (mut total, mut submitted) = (0u32, 0u32);
        let mut nearest: Vec<(f32, usize)> = vec![];
        let mut bound = usize::MAX;
        for (i, d) in self.scene.draws.iter().enumerate() {
            if (d.water && !water) || only.is_some_and(|o| !o[i]) {
                continue;
            }
            // Without an alpha test and off the water, a draw is opaque.
            let solid = self.scene.materials[d.material].params[0] < 0.0;
            let opaque = !d.water && solid;
            match part {
                WorldPart::Opaque if !opaque => continue,
                WorldPart::Masked if opaque || d.water => continue,
                WorldPart::Water if !d.water || !solid => continue,
                WorldPart::MaskedWater if !d.water || solid => continue,
                _ => {}
            }
            total += 1;
            if !frustum.intersects_aabb(d.bounds.0, d.bounds.1) {
                continue;
            }
            if hybrid && !d.water && d.max_extent < 160.0 {
                let dist = aabb_distance(eye.pos, d.bounds.0, d.bounds.1);
                if (d.max_extent < 2.0 && dist > 140.0) || (d.max_extent < 12.0 && dist > 260.0) {
                    continue;
                }
            }
            submitted += 1;
            if part == WorldPart::Opaque {
                nearest.push((aabb_distance(eye.pos, d.bounds.0, d.bounds.1), i));
            } else {
                self.draw_item(enc, model_push, d, &mut bound);
            }
        }
        // Nearest first: what stands behind then fails the depth test before it is shaded.
        nearest.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (_, i) in nearest {
            self.draw_item(enc, model_push, &self.scene.draws[i], &mut bound);
        }
        (total, submitted)
    }

    pub(super) fn draw_item(
        &self,
        enc: &mut Encoder,
        model_push: &mut [f32; 32],
        d: &DrawItem,
        bound: &mut usize,
    ) {
        if *bound != d.material {
            enc.set_bind_group(1, self.materials[d.material]);
            *bound = d.material;
        }
        let m = &self.scene.materials[d.material];
        let detail = &self.details[d.material];
        model_push[..16].copy_from_slice(detail);
        // A colour-only rule (preset `none`) recolours the material whatever its shading,
        // foliage included; patterned presets take the colour in their own way.
        let mut base = m.base_color;
        if detail[0] == 0.0 && detail[8] > 0.5 {
            base[..3].copy_from_slice(&detail[5..8]);
        }
        model_push[16..20].copy_from_slice(&base);
        model_push[20..24].copy_from_slice(&m.emissive);
        model_push[24..28].copy_from_slice(&m.params);
        model_push[28..32].copy_from_slice(&m.flags);
        enc.push_constants(&f32_bytes(model_push));
        enc.draw_indexed(d.count, 1, d.first_index, 0, 0);
    }

    pub(super) fn post_pass(
        enc: &mut Encoder,
        target: TextureId,
        mip: u32,
        load: Load,
        pipeline: PipelineId,
        group: BindGroupId,
        push: [f32; 16],
    ) {
        enc.begin_pass(PassDesc {
            colors: vec![Attachment {
                texture: target,
                mip,
                layer: 0,
                load,
                store: true,
            }],
            depth: None,
        });
        enc.set_pipeline(pipeline);
        enc.set_bind_group(0, group);
        enc.push_constants(&f32_bytes(&push));
        enc.draw(3, 1, 0, 0);
        enc.end_pass();
    }

    /// Record one frame. `target` is the swapchain texture for [`Output::Surface`] renderers
    /// and ignored for stills (which write their readback target). `jitter` is a sub-pixel
    /// offset in pixels; `seed` varies per accumulated sample.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        device: &mut Device<'_>,
        enc: &mut Encoder,
        view: &View,
        env: &Environment,
        jitter: [f32; 2],
        seed: u32,
        target: Option<TextureId>,
    ) -> Result<()> {
        let slot_index = match self.output {
            Output::Still => 0,
            Output::Surface(_) => device.frame_slot() % self.slots.len(),
        };
        // The world map is drawn in this frame's first passes, before anything reads it. After
        // a move it is drawn again, over the land's new bounds (as a cook of the moved world).
        if std::mem::take(&mut self.world_moved) {
            self.scene.refresh_world();
            self.world = world_view(&self.scene, self.world.size);
            self.world_drawn = false;
            self.static_shadows_vp = None;
            self.ground_cached_vp = None;
            self.ao_quiet_frames = 0;
        } else {
            self.ao_quiet_frames = self.ao_quiet_frames.saturating_add(1);
        }
        // Baked AO: a still completes it before drawing; a live view refines it a slice per
        // frame, nearest the camera first, once edits have paused (a drag stays smooth).
        match self.output {
            Output::Still => {
                self.scene.finish_ao(device)?;
            }
            Output::Surface(_) if self.ao_quiet_frames >= AO_SETTLE_FRAMES => {
                self.scene
                    .step_ao(device, AO_FRAME_SAMPLES, Some(view.eye))?;
            }
            Output::Surface(_) => {}
        }
        if std::mem::take(&mut self.lut_dirty) {
            device.write_texture(self.lut, 0, 0, &bake_lut(&self.look))?;
        }
        let draw_world = !self.world_drawn;
        self.world_drawn = true;
        let first_shadow_layers = !self.shadow_layers_ready;
        self.shadow_layers_ready = true;
        // Every layer of the light shadow map holds depth before any is read.
        let first_light_shadows = !self.light_shadows_ready;
        self.light_shadows_ready = true;
        // Props first: their bounds widen the shadow cascades' depth range.
        self.prepare_props(device, slot_index, view.eye)?;
        let has_prop_shadows = self.has_prop_shadows();
        // A game's lights after the lamps in the light buffer, written when they changed.
        if self.game_lights.dirty {
            let data: Vec<f32> = self
                .game_lights
                .records()
                .iter()
                .flatten()
                .copied()
                .collect();
            if !data.is_empty() {
                let at = (self.scene.lights.len() * 64) as u64;
                device.write_buffer(self.scene.light_buffer, at, &crate::gpu::f32_bytes(&data))?;
            }
            self.game_lights.dirty = false;
        }
        let size = (self.targets.width, self.targets.height);
        let eye = self.eye(view, jitter, size);
        let main_frustum = Frustum::from_view_proj(&eye.view_proj);
        let cas = self.cascades(view, env);
        let ground = self.ground_view(view);
        self.grass_grid = self.visible_grass(&eye, &ground);
        let lamps_on = env.lamp_level > 0.001;
        let water_visible = self.quality.reflection_scale > 0.0
            && self.look.water_reflection > 0.001
            && (self.has_water_props()
                || self
                    .scene
                    .draws
                    .iter()
                    .any(|d| d.water && main_frustum.intersects_aabb(d.bounds.0, d.bounds.1)));
        // Mirror camera for the water reflection (skipped when no water enters the view frustum).
        let refl_eye = self.scene.water_level.filter(|_| water_visible).map(|h| {
            let r = reflect_y(h);
            let base = self.eye(view, [0.0, 0.0], self.targets.refl);
            let view_proj = mul(&base.view_proj, &r);
            Eye {
                view_proj,
                inv: invert(&view_proj).unwrap_or(math::IDENTITY),
                pos: [view.eye[0], 2.0 * h - view.eye[1], view.eye[2]],
                near: base.near,
            }
        });
        let has_reflection = refl_eye.is_some();
        let world_key;
        {
            let tiles_dim = self.targets.tiles;
            let refl_size = self.targets.refl;
            let slot = &self.slots[slot_index];
            let (frame_buffers, tile_buffers, index_buffers) = (slot.frame, slot.tiles, slot.index);
            // The lamps binned to screen tiles (617 of them in the Harbor): only when the view
            // or the lights changed, so a held camera does not bin them again every frame.
            let tiles_key = self.tile_key(&eye, size, tiles_dim[0], lamps_on);
            if self.slots[slot_index].tiles_for[0] != Some(tiles_key) {
                let (tiles, index) = self.tile_lights(&eye, size, tiles_dim[0], lamps_on);
                device.write_buffer(tile_buffers[0], 0, &crate::gpu::u32_bytes(&tiles))?;
                device.write_buffer(index_buffers[0], 0, &crate::gpu::u32_bytes(&index))?;
                self.slots[slot_index].tiles_for[0] = Some(tiles_key);
            }
            let block = self.frame_block(
                &eye,
                env,
                &cas,
                &ground,
                size,
                tiles_dim[0],
                false,
                has_prop_shadows,
                has_reflection,
                seed,
            );
            device.write_buffer(frame_buffers[0], 0, &block)?;
            // What the world's picture depends on: this view's frame inputs (but the clock,
            // which moves only what the cached world leaves out), the world and the lights.
            let mut key = Fnv::new();
            key.write(&block[..CAMERA_SECONDS]);
            key.write(&block[CAMERA_SECONDS + 4..]);
            key.write(&self.scene.revision.to_le_bytes());
            for record in self.game_lights.records() {
                key.write(&f32_bytes(record));
            }
            world_key = key.finish();
            if let Some(re) = &refl_eye {
                let tiles_key = self.tile_key(re, refl_size, tiles_dim[1], lamps_on);
                if self.slots[slot_index].tiles_for[1] != Some(tiles_key) {
                    let (tiles, index) = self.tile_lights(re, refl_size, tiles_dim[1], lamps_on);
                    device.write_buffer(tile_buffers[1], 0, &crate::gpu::u32_bytes(&tiles))?;
                    device.write_buffer(index_buffers[1], 0, &crate::gpu::u32_bytes(&index))?;
                    self.slots[slot_index].tiles_for[1] = Some(tiles_key);
                }
                let block = self.frame_block(
                    re,
                    env,
                    &cas,
                    &ground,
                    refl_size,
                    tiles_dim[1],
                    true,
                    has_prop_shadows,
                    has_reflection,
                    seed,
                );
                device.write_buffer(frame_buffers[1], 0, &block)?;
            }
        }
        // A presented view draws the world's part of the main pass, and the water's reflection,
        // only when what they show changed; otherwise the frame starts from those pictures. A
        // still draws everything.
        let cached = self.output != Output::Still;
        // While the baked AO converges (a minute or so after start or an edit), the picture
        // follows it every few frames rather than being drawn again every frame.
        let ao_due = self.scene.ao_revision != self.cache_ao_revision
            && (self.cache_age >= AO_CACHE_REFRESH_FRAMES || self.scene.ao.is_idle());
        let rebuild = cached && (self.world_cache != Some(world_key) || ao_due);
        if rebuild {
            self.world_cache = Some(world_key);
            self.cache_ao_revision = self.scene.ao_revision;
            self.cache_age = 0;
        } else {
            self.cache_age = self.cache_age.saturating_add(1);
        }
        if !cached {
            // A still draws everything itself; the picture no longer follows the props.
            self.world_cache = None;
        }
        // The views the props' shadow maps are drawn from this frame, and those maps as casters'
        // moves are measured in.
        let prop_views = PropShadowViews {
            cascades: (env.key_intensity > 0.0 && has_prop_shadows).then_some(cas.mats),
            lights: self.game_lights.shadow_views().to_vec(),
        };
        let cascade_maps = prop_views.cascades.iter().flatten();
        let light_maps = prop_views.lights.iter();
        let shadow_maps: Vec<ShadowMap> = cascade_maps
            .map(|m| (*m, self.quality.shadow_size.max(16)))
            .chain(light_maps.map(|m| (*m, self.quality.light_shadow_size.max(16))))
            .collect();
        // The props the cached picture holds, and the regions of it to draw again: where props
        // settled or left, and where the shadows of casters moved over it.
        let CacheChanges {
            regions,
            moved,
            settled,
            casters_moved,
        } = if cached {
            let view = CacheView {
                view_proj: &eye.view_proj,
                size,
            };
            track_cached_props(
                &mut self.props,
                rebuild,
                &view,
                &shadow_maps,
                SHADOW_DRIFT_TEXELS,
            )
        } else {
            CacheChanges::default()
        };
        let regions = merge_regions(regions);
        let key = env.key_intensity > 0.0 && has_prop_shadows;
        // The props' shadow maps are kept while their views hold and no caster moved across
        // them (props that cast none move freely): a frame draws its live props' shadows from
        // them as they are, and the cached picture was shaded from them. A still draws them
        // every frame.
        let redraw_prop_shadows = !cached
            || casters_moved
            || first_shadow_layers
            || first_light_shadows
            || self.prop_shadows_for.as_ref() != Some(&prop_views);
        // The pixels of the picture moved casters' shadows can reach, shaded again below, and
        // the moved casters whose shadows can reach the water.
        let (reshade, water_moved) =
            if cached && !rebuild && redraw_prop_shadows && !moved.is_empty() {
                if self.receivers.as_ref().is_none_or(|r| r.key != world_key) {
                    let maps = self.prop_shadow_maps(&cas);
                    self.receivers = Some(ReceiverBounds::new(
                        world_key,
                        &self.scene.draws,
                        &maps,
                        &eye.view_proj,
                        size,
                    ));
                }
                let footprints = self.shadow_footprints(&moved, &cas, key);
                let water = self.receivers.as_ref().map_or(vec![], |r| {
                    let reaches = |b: &(V3, V3)| {
                        let f = self.shadow_footprints(&[*b], &cas, key);
                        r.water.iter().any(|m| reached(&f, m))
                    };
                    moved.iter().copied().filter(reaches).collect()
                });
                let budget = MAX_RESHADE_PAIRS;
                let plan = self.reshade_plan(&footprints, &settled, &eye, size, &cas, budget);
                (plan, water)
            } else {
                (None, vec![])
            };
        // The water's lamp light and shadow, kept (`cache_water`) while the picture holds. A
        // frame drawing the picture whole (as while the camera moves) computes them, as a still
        // does; the first frame keeping it draws them whole. Later frames draw them again in the
        // regions of the picture drawn again, and follow the props moved since whose shadows can
        // reach the water: within those shadows' footprints the water computes them (`fs_water`,
        // as `fs_reshade_mask` tests the picture's pixels), until too many have them drawn whole.
        let keep_water = cached && !rebuild && self.targets.cache_water.is_some();
        if !keep_water {
            self.water_statics = None;
        }
        let water_whole = keep_water
            && self.water_statics.as_mut().is_none_or(|boxes| {
                boxes.extend(water_moved);
                *boxes = merge_boxes(std::mem::take(boxes));
                boxes.len() > MAX_WATER_MOVED
            });
        if water_whole {
            self.water_statics = Some(vec![]);
        }
        // The water tests its footprints at the lookups themselves (`water_shadows_moved`).
        let water_block = self.water_statics.as_ref().map(|boxes| {
            let footprints = self.shadow_footprints(boxes, &cas, key);
            reshade_block(&footprints, [0, 0], [[0.0; 2]; 4])
        });
        let mut stats = FrameStats::default();

        // Shadow cascades: layers 0..2 cache static world depth; layers 2..4 hold dynamic props.
        if env.key_intensity > 0.0 {
            if self.static_shadows_vp == Some(cas.mats) {
                stats.shadow_static_cached = true;
            } else {
                for (layer, m) in cas.mats.iter().enumerate() {
                    let shadow_frustum = Frustum::from_view_proj(m);
                    enc.begin_pass(PassDesc {
                        colors: vec![],
                        depth: Some(Attachment {
                            texture: self.shadow_map,
                            mip: 0,
                            layer: layer as u32,
                            load: Load::Clear([1.0; 4]),
                            store: true,
                        }),
                    });
                    enc.set_pipeline(self.pipelines.shadow);
                    enc.set_vertex_buffer(self.scene.vertices, 0);
                    enc.set_index_buffer(self.scene.indices, 0);
                    enc.push_constants(&f32_bytes(m));
                    let masked = |d: &&DrawItem| self.shadow_materials[d.material].is_some();
                    for d in self.scene.draws.iter().filter(|d| {
                        !d.water
                            && !masked(d)
                            && shadow_frustum.intersects_aabb_clamped(d.bounds.0, d.bounds.1)
                    }) {
                        stats.shadow_world_draws += 1;
                        enc.draw_indexed(d.count, 1, d.first_index, 0, 0);
                    }
                    let mut bound = false;
                    for d in self.scene.draws.iter().filter(|d| {
                        !d.water
                            && masked(d)
                            && shadow_frustum.intersects_aabb_clamped(d.bounds.0, d.bounds.1)
                    }) {
                        if !bound {
                            enc.set_pipeline(self.pipelines.shadow_masked);
                            bound = true;
                        }
                        let mat = &self.scene.materials[d.material];
                        if let Some(g) = self.shadow_materials[d.material] {
                            enc.set_bind_group(1, g);
                        }
                        let mut push = [0.0f32; 20];
                        push[..16].copy_from_slice(m);
                        push[16..].copy_from_slice(&[
                            mat.params[0],
                            mat.base_color[3],
                            mat.flags[0],
                            0.0,
                        ]);
                        enc.push_constants(&f32_bytes(&push));
                        stats.shadow_world_draws += 1;
                        enc.draw_indexed(d.count, 1, d.first_index, 0, 0);
                    }
                    enc.end_pass();
                }
                self.static_shadows_vp = Some(cas.mats);
            }
            if redraw_prop_shadows && (has_prop_shadows || first_shadow_layers) {
                for (layer, m) in cas.mats.iter().enumerate() {
                    enc.begin_pass(PassDesc {
                        colors: vec![],
                        depth: Some(Attachment {
                            texture: self.shadow_map,
                            mip: 0,
                            layer: (layer + 2) as u32,
                            load: Load::Clear([1.0; 4]),
                            store: true,
                        }),
                    });
                    if has_prop_shadows {
                        self.draw_prop_shadows(enc, slot_index, m);
                    }
                    enc.end_pass();
                }
            }
        } else {
            self.static_shadows_vp = None;
            let clear_layers = if first_shadow_layers { 4 } else { 2 };
            for layer in 0..clear_layers {
                enc.begin_pass(PassDesc {
                    colors: vec![],
                    depth: Some(Attachment {
                        texture: self.shadow_map,
                        mip: 0,
                        layer,
                        load: Load::Clear([1.0; 4]),
                        store: true,
                    }),
                });
                enc.end_pass();
            }
        }

        // A game's shadowed lights: the props seen from each, on a layer apiece.
        for layer in (0..crate::render::lights::MAX_SHADOWED_LIGHTS).filter(|_| redraw_prop_shadows)
        {
            let light_vp = prop_views.lights.get(layer);
            if light_vp.is_none() && !first_light_shadows {
                continue;
            }
            enc.begin_pass(PassDesc {
                colors: vec![],
                depth: Some(Attachment {
                    texture: self.light_shadow_map,
                    mip: 0,
                    layer: layer as u32,
                    load: Load::Clear([1.0; 4]),
                    store: true,
                }),
            });
            if let Some(m) = light_vp {
                self.draw_prop_shadows(enc, slot_index, m);
            }
            enc.end_pass();
        }
        if redraw_prop_shadows {
            self.prop_shadows_for = Some(prop_views);
            commit_prop_shadows(&mut self.props);
        } else {
            stats.prop_shadows_kept = true;
        }

        // Rain and petals wrap around a box just ahead of the camera.
        let fwd = normalize(sub(view.target, view.eye));
        let boxes = [RAIN_BOX, PETAL_BOX].map(|b| add(view.eye, scale(fwd, b * 0.5 - 2.0)));
        // The ground map around the view (cached while snapped texel grid stays fixed), and the world map.
        if self.ground_cached_vp == Some(ground.vp) {
            stats.ground_cached = true;
        } else {
            self.draw_ground(enc, self.ground_depth, self.ground_mask, &ground.vp);
            self.ground_cached_vp = Some(ground.vp);
        }
        if draw_world {
            self.draw_ground(enc, self.world_depth, self.world_mask, &self.world.vp);
        }

        let mut push = [0.0f32; 32];
        let clear = [0.0, 0.0, 0.0, 0.0];
        // Water reflection (kept from the frame that last drew it while the world's picture holds;
        // the props it mirrors stand far from the water).
        if let Some(re) = refl_eye.as_ref().filter(|_| !cached || rebuild) {
            stats.reflection_drawn = true;
            self.targets.refl_cleared = true;
            let (refl_hdr, refl_aux, refl_depth) = (
                self.targets.refl_hdr,
                self.targets.refl_aux,
                self.targets.refl_depth,
            );
            enc.begin_pass(PassDesc {
                colors: vec![
                    Attachment::clear(refl_hdr, clear),
                    Attachment::clear(refl_aux, clear),
                ],
                depth: Some(Attachment::clear(refl_depth, clear)),
            });
            enc.set_pipeline(self.pipelines.scene);
            enc.set_bind_group(0, self.slots[slot_index].scene[1]);
            self.draw_world(enc, &mut push, re, false, WorldPart::All, None);
            self.draw_props(enc, slot_index, 1, PropSet::All, None);
            enc.set_pipeline(self.pipelines.sky);
            enc.set_bind_group(0, self.slots[slot_index].sky[1]);
            enc.draw(3, 1, 0, 0);
            enc.end_pass();
        } else if !self.targets.refl_cleared {
            self.targets.refl_cleared = true;
            enc.begin_pass(PassDesc {
                colors: vec![Attachment::clear(self.targets.refl_hdr, clear)],
                depth: None,
            });
            enc.end_pass();
        }

        let need_ao_clear = self.quality.ao_samples == 0 && !self.targets.ao_cleared;
        if need_ao_clear {
            self.targets.ao_cleared = true;
        }

        let slot = &self.slots[slot_index];
        let t = &self.targets;
        let p = &self.pipelines;
        let (mut total_draws, mut submitted_draws) = (0, 0);
        let mut count = |(total, submitted): (u32, u32)| {
            total_draws += total;
            submitted_draws += submitted;
        };
        let whole = [[0, 0, size.0, size.1]];
        // The cached picture: the world and the settled props, drawn whole or in regions (each
        // emptied, then drawn again within it).
        if rebuild || !regions.is_empty() {
            let target = |texture| Attachment {
                load: if rebuild {
                    Load::Clear(clear)
                } else {
                    Load::Load
                },
                ..Attachment::clear(texture, clear)
            };
            let shadow_target = Attachment {
                load: if rebuild {
                    Load::Clear([1.0, 1.0, 1.0, 0.0])
                } else {
                    Load::Load
                },
                ..Attachment::clear(t.cache_shadow, clear)
            };
            enc.begin_pass(PassDesc {
                colors: vec![target(t.cache_hdr), target(t.cache_aux), shadow_target],
                depth: Some(target(t.cache_depth)),
            });
            for r in if rebuild { &whole[..] } else { &regions[..] } {
                if !rebuild {
                    enc.set_scissor(r[0], r[1], r[2] - r[0], r[3] - r[1]);
                    enc.set_pipeline(p.cache_clear);
                    enc.draw(3, 1, 0, 0);
                }
                let within = eye.within(r, size);
                // The settled props first: they stand nearest and cover most of the picture.
                self.draw_props(enc, slot_index, 0, PropSet::Settled, None);
                enc.set_pipeline(p.cache_scene[1]);
                enc.set_bind_group(0, slot.scene[0]);
                count(self.draw_world(enc, &mut push, &within, true, WorldPart::Opaque, None));
                enc.set_pipeline(p.cache_scene[0]);
                enc.set_bind_group(0, slot.scene[0]);
                count(self.draw_world(enc, &mut push, &within, true, WorldPart::Masked, None));
            }
            enc.end_pass();
        }
        stats.cache_regions = if rebuild { 0 } else { regions.len() as u32 };
        if let Some(plan) = &reshade {
            // Mark the pixels to shade again (depth 0; elsewhere 1, which nothing passes), then
            // draw what the picture holds over them alone, as the picture was drawn.
            device.write_buffer(slot.reshade, 0, &crate::gpu::u32_bytes(&plan.block))?;
            let [x0, y0, x1, y1] = plan.region;
            let keep = |texture| Attachment {
                load: Load::Load,
                ..Attachment::clear(texture, clear)
            };
            enc.begin_pass(PassDesc {
                colors: vec![],
                depth: Some(Attachment::clear(t.reshade_depth, [1.0; 4])),
            });
            enc.set_scissor(x0, y0, x1 - x0, y1 - y0);
            enc.set_pipeline(p.reshade_mask);
            enc.set_bind_group(0, slot.mask);
            enc.draw(3, 1, 0, 0);
            enc.end_pass();
            enc.begin_pass(PassDesc {
                colors: vec![keep(t.cache_hdr), keep(t.cache_aux), keep(t.cache_shadow)],
                depth: Some(keep(t.reshade_depth)),
            });
            enc.set_scissor(x0, y0, x1 - x0, y1 - y0);
            let within = eye.within(&plan.region, size);
            // The settled props with a pixel in the region (the rest draw none there).
            let seen = Frustum::from_view_proj(&within.view_proj);
            self.draw_props(enc, slot_index, 0, PropSet::Settled, Some(&seen));
            enc.set_pipeline(p.cache_scene[1]);
            enc.set_bind_group(0, slot.scene[0]);
            let only = Some(&plan.draws[..]);
            count(self.draw_world(enc, &mut push, &within, true, WorldPart::Opaque, only));
            enc.set_pipeline(p.cache_scene[0]);
            enc.set_bind_group(0, slot.scene[0]);
            count(self.draw_world(enc, &mut push, &within, true, WorldPart::Masked, only));
            enc.end_pass();
            stats.reshaded = true;
            stats.reshade_pixels = (x1 - x0) * (y1 - y0);
        }
        // The water's kept lamp light and shadow, drawn whole or in the regions drawn again where
        // the picture's depth (a copy) shows the water, the nearer of two waters keeping them; the
        // main pass's water reads them where it is drawn, which lies within that.
        let water_regions = match t.cache_water {
            Some(kept) if water_whole => Some((kept, &whole[..])),
            Some(kept) if keep_water && !regions.is_empty() => Some((kept, &regions[..])),
            _ => None,
        };
        if let Some((kept, rects)) = water_regions {
            enc.blit((t.cache_depth, 0, 0), (t.depth, 0, 0), Filter::Nearest);
            enc.begin_pass(PassDesc {
                colors: vec![Attachment {
                    load: if water_whole {
                        Load::Clear(clear)
                    } else {
                        Load::Load
                    },
                    ..Attachment::clear(kept, clear)
                }],
                depth: Some(Attachment {
                    load: Load::Load,
                    store: false,
                    ..Attachment::clear(t.depth, clear)
                }),
            });
            enc.set_pipeline(p.water_static);
            enc.set_bind_group(0, slot.scene[0]);
            for r in rects {
                enc.set_scissor(r[0], r[1], r[2] - r[0], r[3] - r[1]);
                let within = eye.within(r, size);
                self.draw_world(enc, &mut push, &within, true, WorldPart::Water, None);
            }
            enc.end_pass();
            stats.water_statics_whole = water_whole;
            stats.water_statics_regions = if water_whole { 0 } else { rects.len() as u32 };
        }
        let water_kept = match (slot.water, &water_block) {
            (Some((group, moved)), Some(block)) => {
                device.write_buffer(moved, 0, &crate::gpu::u32_bytes(block))?;
                stats.water_kept = true;
                stats.water_footprints = block[0];
                Some(group)
            }
            _ => None,
        };

        // Main pass: world, sky, weather. A cached frame starts from the picture's depth, copied:
        // drawn from a shader instead, it would cost the GPU its early depth test for the rest of
        // the pass, and the water under the deck (4.4 of 10.4 ms on a 780M) would be shaded
        // whole behind it.
        if cached {
            enc.blit((t.cache_depth, 0, 0), (t.depth, 0, 0), Filter::Nearest);
        }
        enc.begin_pass(PassDesc {
            colors: vec![
                Attachment::clear(t.hdr, clear),
                Attachment::clear(t.aux, clear),
            ],
            depth: Some(Attachment {
                load: if cached {
                    Load::Load
                } else {
                    Load::Clear(clear)
                },
                ..Attachment::clear(t.depth, clear)
            }),
        });
        // Timing each part (`split_passes`): the pass ends and goes on as another.
        let splits = self.split_passes && cached;
        let split = |enc: &mut crate::gpu::Encoder| {
            if splits {
                let keep = |texture| Attachment {
                    load: Load::Load,
                    ..Attachment::clear(texture, clear)
                };
                enc.end_pass();
                enc.begin_pass(PassDesc {
                    colors: vec![keep(t.hdr), keep(t.aux)],
                    depth: Some(keep(t.depth)),
                });
            }
        };
        if cached {
            // The cached picture (colour, aux and depth), then the live props over it.
            enc.set_pipeline(p.restore);
            enc.set_bind_group(0, slot.restore);
            enc.draw(3, 1, 0, 0);
            split(enc);
            self.draw_props(enc, slot_index, 0, PropSet::Live, None);
        } else {
            // The game's props first: the table and its players stand nearest and cover most
            // of the screen, so the world behind them fails the depth test before it is shaded.
            self.draw_props(enc, slot_index, 0, PropSet::All, None);
            enc.set_pipeline(p.scene_opaque);
            enc.set_bind_group(0, slot.scene[0]);
            count(self.draw_world(enc, &mut push, &eye, true, WorldPart::Opaque, None));
            enc.set_pipeline(p.scene);
            enc.set_bind_group(0, slot.scene[0]);
            count(self.draw_world(enc, &mut push, &eye, true, WorldPart::Masked, None));
        }
        // The water moves with the clock, so it is drawn every frame. Without an alpha test it
        // never discards, so the floor and the table in front of it hide it before it is
        // shaded: behind the room it spans the whole view. It takes its lamp light and shadow
        // from `cache_water` while they are kept.
        split(enc);
        if let Some(group) = water_kept {
            enc.set_pipeline(p.water);
            enc.set_bind_group(0, group);
        } else {
            enc.set_pipeline(p.scene_opaque);
            enc.set_bind_group(0, slot.scene[0]);
        }
        count(self.draw_world(enc, &mut push, &eye, true, WorldPart::Water, None));
        split(enc);
        enc.set_pipeline(p.scene);
        enc.set_bind_group(0, slot.scene[0]);
        count(self.draw_world(enc, &mut push, &eye, true, WorldPart::MaskedWater, None));

        stats.world_cached = cached && !rebuild;
        stats.world_draws_total = total_draws;
        stats.world_draws_submitted = submitted_draws;
        stats.world_draws_culled = total_draws.saturating_sub(submitted_draws);
        let blades = self.grass_instances();
        split(enc);
        if blades > 0 {
            enc.set_pipeline(p.grass);
            enc.set_bind_group(0, slot.grass);
            enc.draw(15, blades, 0, 0);
            if self.look.grass.flowers > 0.0 {
                enc.set_pipeline(p.flower);
                enc.set_bind_group(0, slot.flower);
                enc.draw(6, blades, 0, 0);
            }
        }
        split(enc);
        enc.set_pipeline(p.sky);
        enc.set_bind_group(0, slot.sky[0]);
        enc.draw(3, 1, 0, 0);
        split(enc);
        if self.quality.particles {
            enc.set_pipeline(p.particle);
            enc.set_bind_group(0, slot.particle);
            let ground = self.scene.water_level.unwrap_or(self.scene.min[1]);
            let particle_push = |params: [f32; 4], tint: [f32; 4], centre: V3| {
                let mut pp = [0.0f32; 12];
                pp[..4].copy_from_slice(&params);
                pp[4..8].copy_from_slice(&tint);
                pp[8..11].copy_from_slice(&centre);
                pp
            };
            let rain = (env.rain * 26000.0) as u32;
            if rain > 0 {
                let pp = particle_push(
                    [0.0, ground, RAIN_BOX, 1.0],
                    [0.7, 0.75, 0.85, 0.3],
                    boxes[0],
                );
                enc.push_constants(&f32_bytes(&pp));
                enc.draw(6, rain, 0, 0);
            }
            let petals = (env.petals * 3500.0) as u32;
            if petals > 0 {
                let pp = particle_push(
                    [1.0, ground, PETAL_BOX, 1.0],
                    [0.95, 0.5, 0.62, 0.9],
                    boxes[1],
                );
                enc.push_constants(&f32_bytes(&pp));
                enc.draw(6, petals, 0, 0);
            }
        }
        split(enc);
        self.draw_blended_props(enc, slot_index);
        enc.end_pass();

        let l = &self.look;
        let px_scale = t.height as f32 / 1080.0;
        // Ambient occlusion (skipped when `ao_samples == 0`, relying on baked vertex AO).
        if self.quality.ao_samples > 0 {
            stats.ssao_drawn = true;
            let mut pc = [0.0f32; 16];
            pc[..4].copy_from_slice(&[
                l.ao_radius,
                self.quality.ao_samples as f32,
                t.ao.0 as f32,
                t.ao.1 as f32,
            ]);
            Self::post_pass(enc, t.ao_tex, 0, Load::DontCare, p.ssao, slot.ssao, pc);
        } else if need_ao_clear {
            enc.begin_pass(PassDesc {
                colors: vec![Attachment::clear(t.ao_tex, [1.0; 4])],
                depth: None,
            });
            enc.end_pass();
        }
        // Air.
        let mut pc = [0.0f32; 16];
        pc[..4].copy_from_slice(&[
            l.ambient_occlusion,
            l.outline,
            (1.4 * px_scale).max(1.0),
            0.0,
        ]);
        pc[4..7].copy_from_slice(&l.ao_tint);
        pc[8..11].copy_from_slice(&l.outline_color);
        Self::post_pass(enc, t.lit, 0, Load::DontCare, p.air, slot.air, pc);
        // Bloom down and up.
        for level in 0..t.bloom_levels {
            let (sw, sh) = if level == 0 {
                (t.width, t.height)
            } else {
                (
                    (t.bloom_size.0 >> (level - 1)).max(1),
                    (t.bloom_size.1 >> (level - 1)).max(1),
                )
            };
            let mut pc = [0.0f32; 16];
            pc[..4].copy_from_slice(&[
                1.0 / sw as f32,
                1.0 / sh as f32,
                l.bloom_threshold,
                if level == 0 { 1.0 } else { 0.0 },
            ]);
            Self::post_pass(
                enc,
                t.bloom,
                level,
                Load::DontCare,
                p.bloom_down,
                slot.down[level as usize],
                pc,
            );
        }
        for level in (0..t.bloom_levels.saturating_sub(1)).rev() {
            let (sw, sh) = (
                (t.bloom_size.0 >> (level + 1)).max(1),
                (t.bloom_size.1 >> (level + 1)).max(1),
            );
            let mut pc = [0.0f32; 16];
            pc[..4].copy_from_slice(&[1.0 / sw as f32, 1.0 / sh as f32, 1.0, 0.0]);
            Self::post_pass(
                enc,
                t.bloom,
                level,
                Load::Load,
                p.bloom_up,
                slot.up[level as usize],
                pc,
            );
        }
        let (dest, encode_srgb) = match self.output {
            Output::Still => (t.still.ok_or("still target missing")?, false),
            Output::Surface(f) => (target.ok_or("no surface texture")?, !f.is_srgb()),
        };
        let exposure = l.effective_exposure() * env.exposure;
        if !self.quality.painterly {
            stats.fused_tonemap = true;
            let mut pc = [0.0f32; 16];
            pc[..4].copy_from_slice(&[exposure, l.bloom, l.vignette, l.sharpen.max(0.0)]);
            pc[4] = if encode_srgb { 1.0 } else { 0.0 };
            pc[5] = if self.quality.fxaa { 1.0 } else { 0.0 };
            Self::post_pass(
                enc,
                dest,
                0,
                Load::DontCare,
                p.tonemap_final,
                slot.tonemap_final,
                pc,
            );
        } else {
            // Tone mapping and grading.
            let mut grade = [0.0f32; 16];
            grade[..4].copy_from_slice(&[exposure, l.bloom, l.saturation, l.contrast]);
            grade[4..7].copy_from_slice(&l.lift);
            grade[7] = l.vignette;
            grade[8..11].copy_from_slice(&l.gain);
            grade[11] = l.tonemap as u32 as f32;
            let mut pc = [0.0f32; 16];
            pc[..4].copy_from_slice(&[
                (l.painterly * px_scale * 0.6).clamp(0.0, 6.0),
                (l.painterly * l.painterly_far.max(0.0) * px_scale).clamp(0.0, 6.0),
                0.85,
                0.0,
            ]);
            // Depth of field, drawn with the painterly filter.
            if let Some(f) = view.focus.filter(Focus::blurs) {
                pc[4..8].copy_from_slice(&[
                    f.distance.max(0.0),
                    f.depth,
                    (f.blur * t.height as f32).min(MAX_FOCUS_BLUR_PX),
                    1.0,
                ]);
            }
            // At radii under a pixel and without depth of field the filter copies each pixel
            // through. Then one pass tone-maps, grades, sharpens and smooths edges as the three
            // passes would, two full-screen passes fewer.
            let painted = pc[0].round() >= 1.0 || pc[1].round() >= 1.0 || pc[7] > 0.5;
            stats.painted = painted;
            if painted {
                Self::post_pass(
                    enc,
                    t.ldr,
                    0,
                    Load::DontCare,
                    p.tonemap,
                    slot.tonemap,
                    grade,
                );
                Self::post_pass(enc, t.paint, 0, Load::DontCare, p.paint, slot.paint, pc);
                let mut pc = [0.0f32; 16];
                pc[0] = if self.quality.fxaa { 1.0 } else { 0.0 };
                pc[1] = if encode_srgb { 1.0 } else { 0.0 };
                pc[2] = l.sharpen.max(0.0);
                Self::post_pass(enc, dest, 0, Load::DontCare, p.output, slot.output, pc);
            } else {
                grade[12] = l.sharpen.max(0.0);
                grade[13] = if self.quality.fxaa { 1.0 } else { 0.0 };
                grade[14] = if encode_srgb { 1.0 } else { 0.0 };
                let plain = usize::from(l.grade_is_identity());
                let (pipe, group) = match (t.graded, p.grade_once, slot.grade_once) {
                    (Some(graded), Some((pipes, read)), Some((groups, read_group))) => {
                        stats.graded_once = true;
                        let (pipe, group) = (pipes[plain], groups[plain]);
                        Self::post_pass(enc, graded, 0, Load::DontCare, pipe, group, grade);
                        (read, read_group)
                    }
                    _ => (p.tonemap_output[plain], slot.tonemap_output[plain]),
                };
                Self::post_pass(enc, dest, 0, Load::DontCare, pipe, group, grade);
            }
        }
        if self.has_overlays() {
            enc.begin_pass(PassDesc {
                colors: vec![Attachment {
                    texture: dest,
                    mip: 0,
                    layer: 0,
                    load: Load::Load,
                    store: true,
                }],
                depth: Some(Attachment {
                    texture: t.depth,
                    mip: 0,
                    layer: 0,
                    load: Load::Load,
                    store: false,
                }),
            });
            self.draw_overlays(enc, slot_index, encode_srgb);
            enc.end_pass();
        }
        if let (Output::Still, Some(still), Some(rb)) = (self.output, t.still, t.readback) {
            enc.copy_texture_to_buffer(still, 0, 0, rb, 0);
        }
        self.stats = stats;
        Ok(())
    }

    /// Render a still: `samples` jittered frames averaged in linear display space. Returns
    /// RGBA8 sRGB pixels, top row first.
    pub fn render_still(
        &mut self,
        device: &mut Device<'_>,
        view: &View,
        env: &Environment,
        samples: u32,
    ) -> Result<Vec<u8>> {
        if self.output != Output::Still {
            return Err("render_still needs a renderer created with Output::Still".into());
        }
        let (w, h) = self.size();
        let n = (w * h * 4) as usize;
        let mut acc = vec![0.0f32; n];
        let samples = samples.max(1);
        for s in 0..samples {
            let jitter = if samples == 1 {
                [0.0, 0.0]
            } else {
                [halton(s + 1, 2) - 0.5, halton(s + 1, 3) - 0.5]
            };
            let mut enc = device.encoder();
            self.encode(device, &mut enc, view, env, jitter, s, None)?;
            device.submit(enc)?;
            let bytes = device.read_buffer(self.targets.readback.ok_or("no readback")?)?;
            for (i, a) in acc.iter_mut().enumerate() {
                let h = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                *a += f16_to_f32(h);
            }
        }
        let inv = 1.0 / samples as f32;
        let mut out = vec![0u8; n];
        for (i, (o, a)) in out.iter_mut().zip(&acc).enumerate() {
            let v = a * inv;
            if i % 4 == 3 {
                *o = 255;
            } else {
                // Ordered dither before quantizing to avoid banding in skies.
                let px = (i / 4) as u32;
                let d = (halton(px % 4096 + 1, 5) - 0.5) / 255.0;
                *o = ((math::linear_to_srgb(v) + d) * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
        Ok(out)
    }
}
