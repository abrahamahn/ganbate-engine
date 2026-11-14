use super::*;

/// Regions whose depth or normals changed that a frame keeping the AO (`AoChanged` in post.wgsl)
/// computes it again around, at most; more are joined (`AoChanged.rects` holds 16).
pub(super) const MAX_AO_REGIONS: usize = 16;
const _: () = assert!(MAX_AO_REGIONS <= 16);

/// Bytes of the AO's changed regions block: its head, view_proj's row lengths, active grass
/// coarse-tile bounds, then the regions.
pub(super) const AO_CHANGED_BYTES: u64 = 48 + MAX_AO_REGIONS as u64 * 16;

/// Pixels the AO's changed regions are widened by, against the rounding of a kernel point's pixel
/// apart where `fs_ssao` tests it and where it reads it.
pub(super) const AO_REGION_MARGIN: u32 = 2;

/// The farthest a kernel point of `fs_ssao` lies from its surface point, in radii (0.951: the
/// fifth, lifted 0.08 off the surface), rounded up.
pub(super) const AO_KERNEL_REACH: f32 = 0.96;

/// Half (`1/2`), fine (`1/8`) and coarse (`1/64`) dimensions for the GPU visible-grass masks.
pub(super) fn ao_grass_sizes((width, height): (u32, u32)) -> ((u32, u32), (u32, u32), (u32, u32)) {
    let hw = width.max(1).div_ceil(2);
    let hh = height.max(1).div_ceil(2);
    let gw = width.max(1).div_ceil(8);
    let gh = height.max(1).div_ceil(8);
    let cw = gw.div_ceil(8).max(1);
    let ch = gh.div_ceil(8).max(1);
    ((hw, hh), (gw, gh), (cw, ch))
}

/// The regions of a `size` picture whose depth or normals changed, widened by
/// [`AO_REGION_MARGIN`] and joined to at most [`MAX_AO_REGIONS`] ([`join_regions`]).
pub(super) fn ao_regions(rects: impl IntoIterator<Item = Rect>, size: (u32, u32)) -> Vec<Rect> {
    let m = AO_REGION_MARGIN;
    let widened = rects.into_iter().map(|r| {
        [
            r[0].saturating_sub(m),
            r[1].saturating_sub(m),
            r[2].saturating_add(m).min(size.0),
            r[3].saturating_add(m).min(size.1),
        ]
    });
    join_regions(widened.collect(), size)
}

/// Regions of a `size` picture joined to at most [`MAX_AO_REGIONS`] holding every pixel they
/// hold: many are first cut along an 8 x 8 grid and joined within each cell; then, each time,
/// the two whose union holds the fewest pixels they do not are joined.
pub(super) fn join_regions(rects: Vec<Rect>, size: (u32, u32)) -> Vec<Rect> {
    const CELLS: u32 = 8;
    let mut out: Vec<Rect> = rects
        .into_iter()
        .filter(|r| r[0] < r[2] && r[1] < r[3])
        .collect();
    if out.len() > CELLS as usize * CELLS as usize {
        let (cw, ch) = (size.0.div_ceil(CELLS).max(1), size.1.div_ceil(CELLS).max(1));
        let mut cells: Vec<Option<Rect>> = vec![None; (CELLS * CELLS) as usize];
        for r in &out {
            for cy in r[1] / ch..=((r[3] - 1) / ch).min(CELLS - 1) {
                for cx in r[0] / cw..=((r[2] - 1) / cw).min(CELLS - 1) {
                    let cell = [cx * cw, cy * ch, (cx + 1) * cw, (cy + 1) * ch];
                    if let Some(piece) = intersect(*r, cell) {
                        let joined = &mut cells[(cy * CELLS + cx) as usize];
                        *joined = Some(joined.map_or(piece, |j| union(j, piece)));
                    }
                }
            }
        }
        out = cells.into_iter().flatten().collect();
    }
    let area = |r: Rect| u64::from(r[2] - r[0]) * u64::from(r[3] - r[1]);
    while out.len() > MAX_AO_REGIONS {
        let mut best = (u64::MAX, 0, 0);
        for i in 0..out.len() {
            for j in i + 1..out.len() {
                let grown = area(union(out[i], out[j])).saturating_sub(area(out[i]) + area(out[j]));
                best = best.min((grown, i, j));
            }
        }
        let b = out.swap_remove(best.2);
        out[best.1] = union(out[best.1], b);
    }
    out
}

/// The pixels the regions hold together.
pub(super) fn union_area(regions: &[Rect]) -> u32 {
    let mut xs: Vec<u32> = regions.iter().flat_map(|r| [r[0], r[2]]).collect();
    xs.sort_unstable();
    xs.dedup();
    xs.windows(2)
        .map(|x| {
            let mut spans: Vec<(u32, u32)> = regions
                .iter()
                .filter(|r| r[0] <= x[0] && x[1] <= r[2])
                .map(|r| (r[1], r[3]))
                .collect();
            spans.sort_unstable();
            let (mut held, mut end) = (0, 0);
            for (y0, y1) in spans {
                held += y1.saturating_sub(y0.max(end));
                end = end.max(y1);
            }
            (x[1] - x[0]) * held
        })
        .sum()
}

/// The AO's changed regions block (`AoChanged` in post.wgsl) for `regions` and active coarse
/// grass bounds `grass_coarse` of a picture seen through `view_proj`.
pub(super) fn ao_changed_block(
    regions: &[Rect],
    grass_coarse: [i32; 4],
    view_proj: &M4,
) -> Vec<u8> {
    let row = |k: usize| math::length([view_proj[k], view_proj[4 + k], view_proj[8 + k]]);
    let mut block = vec![regions.len() as f32, AO_KERNEL_REACH, 0.0, 0.0];
    block.extend([row(0), row(1), row(3), 0.0]);
    block.extend(grass_coarse.map(|v| v as f32));
    block.extend(regions.iter().flat_map(|r| r.map(|v| v as f32)));
    f32_bytes(&block)
}

fn scissored_post_pass(
    enc: &mut Encoder,
    (target, target_size): (TextureId, (u32, u32)),
    scissor: Rect,
    load_partial: Load,
    (pipeline, group): (PipelineId, BindGroupId),
    push: [f32; 16],
) {
    if scissor[0] >= scissor[2] || scissor[1] >= scissor[3] {
        return;
    }
    let full = scissor == [0, 0, target_size.0, target_size.1];
    let load = if full { Load::DontCare } else { load_partial };
    enc.begin_pass(PassDesc {
        colors: vec![Attachment {
            texture: target,
            mip: 0,
            layer: 0,
            load,
            store: true,
        }],
        depth: None,
    });
    if !full {
        enc.set_scissor(
            scissor[0],
            scissor[1],
            scissor[2] - scissor[0],
            scissor[3] - scissor[1],
        );
    }
    enc.set_pipeline(pipeline);
    enc.set_bind_group(0, group);
    enc.push_constants(&f32_bytes(&push));
    enc.draw(3, 1, 0, 0);
    enc.end_pass();
}

#[derive(Clone, Copy)]
pub(super) struct AoPrevState {
    ao_for: Option<u64>,
    grass_cur: usize,
    grass_valid: [bool; 2],
    grass_rect: [Option<Rect>; 2],
    slot_has_water: [bool; 2],
    water_stale: [Option<Rect>; 2],
    union_has_water: bool,
    grass_time: u32,
    grass_sway: u32,
}

#[derive(Clone, Copy, Default)]
pub(super) struct AoNextState {
    ao_for: Option<u64>,
    ao_cleared: bool,
    grass_cur: usize,
    grass_valid: [bool; 2],
    grass_rect: [Option<Rect>; 2],
    slot_has_water: [bool; 2],
    water_stale: [Option<Rect>; 2],
    union_has_water: bool,
    grass_time: u32,
    grass_sway: u32,
}

pub(super) struct AoEncodeInput<'a> {
    pub(super) prev: AoPrevState,
    pub(super) cached: bool,
    pub(super) rebuild: bool,
    pub(super) world_key: u64,
    pub(super) eye: &'a Eye,
    pub(super) size: (u32, u32),
    pub(super) seed: u32,
    pub(super) seconds: f32,
    pub(super) cache_regions: &'a [Rect],
    pub(super) live_regions: &'a [Rect],
    pub(super) water_rect: Option<Rect>,
    pub(super) has_water_props: bool,
    pub(super) grass_rect: Option<Rect>,
}

impl Renderer {
    /// Invalidates the kept AO target and its visible-grass/water coverage masks.
    pub(in crate::render) fn forget_ao(&mut self) {
        self.targets.ao_for = None;
        self.targets.ao_grass_valid = [false; 2];
        self.targets.ao_grass_rect = [None; 2];
        self.targets.ao_grass_water = [false; 2];
        self.targets.ao_water_stale = [None; 2];
        self.targets.ao_union_water = false;
    }

    /// Takes the kept AO state before `track_cached_props` consumes prop rects, leaving `ao_for`
    /// and the mask validity flags cleared if `encode` returns early on an error.
    pub(super) fn take_ao_state(&mut self) -> AoPrevState {
        let prev = AoPrevState {
            ao_for: self.targets.ao_for.take(),
            grass_cur: self.targets.ao_grass_cur,
            grass_valid: self.targets.ao_grass_valid,
            grass_rect: self.targets.ao_grass_rect,
            slot_has_water: self.targets.ao_grass_water,
            water_stale: self.targets.ao_water_stale,
            union_has_water: self.targets.ao_union_water,
            grass_time: self.targets.ao_grass_time,
            grass_sway: self.targets.ao_grass_sway,
        };
        self.targets.ao_grass_valid = [false; 2];
        self.targets.ao_grass_rect = [None; 2];
        self.targets.ao_grass_water = [false; 2];
        self.targets.ao_water_stale = [None; 2];
        self.targets.ao_union_water = false;
        prev
    }

    /// Commits the AO state at the end of `encode` after all fallible steps have succeeded.
    pub(super) fn commit_ao_state(&mut self, next: AoNextState) {
        self.targets.ao_for = next.ao_for;
        self.targets.ao_cleared = next.ao_cleared;
        self.targets.ao_grass_cur = next.grass_cur;
        self.targets.ao_grass_valid = next.grass_valid;
        self.targets.ao_grass_rect = next.grass_rect;
        self.targets.ao_grass_water = next.slot_has_water;
        self.targets.ao_water_stale = next.water_stale;
        self.targets.ao_union_water = next.union_has_water;
        self.targets.ao_grass_time = next.grass_time;
        self.targets.ao_grass_sway = next.grass_sway;
    }

    /// Fingerprint of everything `fs_ssao` depends on besides the frame's depth and aux targets.
    pub(super) fn ao_key(&self, world_key: u64, eye: &Eye, size: (u32, u32), seed: u32) -> u64 {
        let mut key = Fnv::new();
        key.write(&world_key.to_le_bytes());
        key.write(&f32_bytes(&eye.view_proj));
        for n in [
            size.0,
            size.1,
            self.targets.ao.0,
            self.targets.ao.1,
            seed,
            self.look.ao_radius.to_bits(),
            self.quality.ao_samples,
            self.quality.ao_scale.to_bits(),
        ] {
            key.write(&n.to_le_bytes());
        }
        key.write(&self.scene.revision.to_le_bytes());
        key.finish()
    }

    /// Conservative screen rectangle covering all grass blades and flowers `self.grass_grid` can
    /// place (`None` when `blades == 0` or completely outside the viewport).
    pub(super) fn grass_rect(&self, eye: &Eye, ground: &Ground, size: (u32, u32)) -> Option<Rect> {
        let [x0, z0, cols, rows] = self.grass_grid;
        if cols == 0 || rows == 0 {
            return None;
        }
        let side = self.grass_side();
        if side == 0 {
            return None;
        }
        let s = GRASS_SPACING;
        let half = (side / 2) as f32;
        let cx = (eye.pos[0] / s).floor() - half;
        let cz = (eye.pos[2] / s).floor() - half;
        let g = &self.look.grass;
        let (h, w, sway) = (g.height.abs(), g.width.abs(), g.sway.abs());
        let pad_xz = s + 3.9 * w + (0.25 + 0.75 * sway) * 1.45 * h + 0.4;
        let lo = [
            (cx + x0 as f32) * s - pad_xz,
            ground.top - ground.range - 1.5 * h - 1.0,
            (cz + z0 as f32) * s - pad_xz,
        ];
        let hi = [
            (cx + (x0 + cols) as f32) * s + pad_xz,
            ground.top + 1.5 * h + 1.0,
            (cz + (z0 + rows) as f32) * s + pad_xz,
        ];
        box_rect((lo, hi), &eye.view_proj, size, 2.0)
    }

    /// Conservative screen rectangle covering all visible world water draws and water-shaded scene
    /// props (`None` when `!water_drawn` or completely outside the viewport).
    pub(super) fn water_rect(
        &self,
        eye: &Eye,
        main_frustum: &Frustum,
        size: (u32, u32),
        water_drawn: bool,
    ) -> Option<Rect> {
        if !water_drawn {
            return None;
        }
        let mut out = self.water_props_rect(&eye.view_proj, size);
        for d in &self.scene.draws {
            if d.water
                && main_frustum.intersects_aabb(d.bounds.0, d.bounds.1)
                && let Some(r) = box_rect(d.bounds, &eye.view_proj, size, 2.0)
            {
                out = Some(out.map_or(r, |acc| union(acc, r)));
            }
        }
        out
    }

    /// Encodes the GPU visible-grass masks (when needed) and the whole, kept, or skipped SSAO pass.
    pub(super) fn encode_ao(
        &self,
        device: &mut Device<'_>,
        enc: &mut Encoder,
        slot_index: usize,
        inp: AoEncodeInput<'_>,
        stats: &mut FrameStats,
    ) -> Result<AoNextState> {
        let t = &self.targets;
        let p = &self.pipelines;
        let slot = &self.slots[slot_index];
        if self.quality.ao_samples == 0 {
            if !t.ao_cleared {
                enc.begin_pass(PassDesc {
                    colors: vec![Attachment::clear(t.ao_tex, [1.0; 4])],
                    depth: None,
                });
                enc.end_pass();
            }
            return Ok(AoNextState {
                ao_cleared: true,
                ..AoNextState::default()
            });
        }

        let ao_key = self.ao_key(inp.world_key, inp.eye, inp.size, inp.seed);
        let full_pixels = inp.size.0 * inp.size.1;
        let can_keep = inp.cached && !inp.rebuild && inp.prev.ao_for == Some(ao_key);
        let (keep_ao, regions, changed_pixels, fallback) = if can_keep {
            let joined = ao_regions(
                inp.cache_regions
                    .iter()
                    .copied()
                    .chain(inp.live_regions.iter().copied()),
                inp.size,
            );
            let changed = union_area(&joined);
            if u64::from(changed) * 2 >= u64::from(full_pixels) {
                (false, Vec::new(), full_pixels, true)
            } else {
                (true, joined, changed, false)
            }
        } else {
            (false, Vec::new(), 0, false)
        };

        let grass_drawn = inp.grass_rect.is_some();
        let water_drawn = inp.water_rect.is_some();
        let sway_bits = self.look.grass.sway.to_bits();
        let time_bits = inp.seconds.to_bits();
        let prev_idx = inp.prev.grass_cur;
        let prev_has_grass = inp.prev.grass_valid[prev_idx];
        let grass_changed = if !keep_ao || (!grass_drawn && !prev_has_grass) {
            false
        } else {
            grass_drawn != prev_has_grass
                || sway_bits != inp.prev.grass_sway
                || (self.look.grass.sway != 0.0 && time_bits != inp.prev.grass_time)
        };

        let union_opt = |a: Option<Rect>, b: Option<Rect>| match (a, b) {
            (Some(a), Some(b)) => Some(union(a, b)),
            (a, b) => a.or(b),
        };
        let ((hw, hh), (gw, gh), (cw, ch)) = ao_grass_sizes(inp.size);
        let regions_bbox = regions.iter().copied().reduce(union);
        let next = if !inp.cached {
            AoNextState::default()
        } else if keep_ao && !water_drawn && regions.is_empty() && !grass_changed {
            AoNextState {
                ao_for: Some(ao_key),
                ao_cleared: false,
                grass_cur: prev_idx,
                grass_valid: inp.prev.grass_valid,
                grass_rect: inp.prev.grass_rect,
                slot_has_water: [false; 2],
                water_stale: [None; 2],
                union_has_water: false,
                grass_time: inp.prev.grass_time,
                grass_sway: inp.prev.grass_sway,
            }
        } else {
            let capture_slot = !keep_ao
                || grass_changed
                || (grass_drawn && self.look.grass.sway != 0.0 && !regions.is_empty())
                || (water_drawn
                    && (!inp.prev.slot_has_water[prev_idx]
                        || !regions.is_empty()
                        || inp.has_water_props));
            let (
                active_idx,
                valid,
                grass_rect,
                slot_has_water,
                water_stale,
                grass_time,
                grass_sway,
            ) = if capture_slot {
                let cur_idx = 1 - prev_idx;
                let mut valid = [false; 2];
                valid[prev_idx] = keep_ao && prev_has_grass;
                valid[cur_idx] = grass_drawn;
                let mut slot_grass = [None; 2];
                slot_grass[prev_idx] = if keep_ao {
                    inp.prev.grass_rect[prev_idx]
                } else {
                    None
                };
                slot_grass[cur_idx] = inp.grass_rect;
                let static_water = water_drawn && !inp.has_water_props;
                let incremental = keep_ao && static_water && inp.prev.slot_has_water[cur_idx];
                let mask_rect = if incremental {
                    [
                        inp.prev.grass_rect[cur_idx],
                        inp.grass_rect,
                        inp.prev.water_stale[cur_idx],
                        regions_bbox,
                    ]
                    .into_iter()
                    .flatten()
                    .reduce(union)
                } else {
                    union_opt(inp.grass_rect, inp.water_rect)
                };
                if let Some([mx0, my0, mx1, my1]) = mask_rect {
                    let g_rect = [
                        mx0 / 8,
                        my0 / 8,
                        mx1.div_ceil(8).min(gw),
                        my1.div_ceil(8).min(gh),
                    ];
                    let h_rect = [
                        g_rect[0] * 4,
                        g_rect[1] * 4,
                        (g_rect[2] * 4).min(hw),
                        (g_rect[3] * 4).min(hh),
                    ];
                    scissored_post_pass(
                        enc,
                        (t.ao_grass_half, (hw, hh)),
                        h_rect,
                        Load::DontCare,
                        (p.ao_grass_half, slot.ao_grass_half),
                        [0.0; 16],
                    );
                    let mut tpc = [0.0f32; 16];
                    tpc[0] = 1.0 / hw as f32;
                    tpc[1] = 1.0 / hh as f32;
                    let tile_load = if incremental {
                        Load::Load
                    } else {
                        Load::Clear([0.0; 4])
                    };
                    scissored_post_pass(
                        enc,
                        (t.ao_grass[cur_idx], (gw, gh)),
                        g_rect,
                        tile_load,
                        (p.ao_grass_tile, slot.ao_grass_tile),
                        tpc,
                    );
                }
                let mut slot_water = [false; 2];
                let mut water_stale = [None; 2];
                slot_water[cur_idx] = static_water;
                if keep_ao && static_water && inp.prev.slot_has_water[prev_idx] {
                    slot_water[prev_idx] = true;
                    water_stale[prev_idx] = union_opt(inp.prev.water_stale[prev_idx], regions_bbox);
                }
                (
                    cur_idx,
                    valid,
                    slot_grass,
                    slot_water,
                    water_stale,
                    time_bits,
                    sway_bits,
                )
            } else {
                let (slot_water, water_stale) = if water_drawn && !inp.has_water_props {
                    (inp.prev.slot_has_water, inp.prev.water_stale)
                } else {
                    ([false; 2], [None; 2])
                };
                (
                    prev_idx,
                    inp.prev.grass_valid,
                    inp.prev.grass_rect,
                    slot_water,
                    water_stale,
                    inp.prev.grass_time,
                    inp.prev.grass_sway,
                )
            };
            let run_union_coarse = keep_ao
                && (grass_changed || (water_drawn && (capture_slot || !inp.prev.union_has_water)));
            if run_union_coarse {
                let mut gpc = [0.0f32; 16];
                gpc[0] = f32::from(u8::from(grass_changed && valid[0]));
                gpc[1] = f32::from(u8::from(grass_changed && valid[1]));
                gpc[2] = f32::from(u8::from(water_drawn && active_idx == 0));
                gpc[3] = f32::from(u8::from(water_drawn && active_idx == 1));
                Self::post_pass(
                    enc,
                    t.ao_grass_union,
                    0,
                    Load::DontCare,
                    p.ao_grass_union,
                    slot.ao_grass_union,
                    gpc,
                );
                let mut cpc = [0.0f32; 16];
                cpc[0] = 1.0 / gw as f32;
                cpc[1] = 1.0 / gh as f32;
                Self::post_pass(
                    enc,
                    t.ao_grass_coarse,
                    0,
                    Load::DontCare,
                    p.ao_grass_coarse,
                    slot.ao_grass_coarse,
                    cpc,
                );
            }
            let union_has_water = water_drawn
                && (run_union_coarse || (keep_ao && !capture_slot && inp.prev.union_has_water));
            AoNextState {
                ao_for: Some(ao_key),
                ao_cleared: false,
                grass_cur: active_idx,
                grass_valid: valid,
                grass_rect,
                slot_has_water,
                water_stale,
                union_has_water,
                grass_time,
                grass_sway,
            }
        };

        let mut pc = [0.0f32; 16];
        pc[..4].copy_from_slice(&[
            self.look.ao_radius,
            self.quality.ao_samples as f32,
            t.ao.0 as f32,
            t.ao.1 as f32,
        ]);
        if keep_ao {
            stats.ao_kept = true;
            stats.ao_changed_pixels = changed_pixels;
            if water_drawn || !regions.is_empty() || grass_changed {
                let grass_union = if grass_changed {
                    (0..2)
                        .filter(|&s| next.grass_valid[s])
                        .filter_map(|s| next.grass_rect[s])
                        .reduce(union)
                } else {
                    None
                };
                let grass_coarse = match grass_union {
                    Some([gx0, gy0, gx1, gy1]) if gx0 < gx1 && gy0 < gy1 => [
                        (gx0 / 64) as i32,
                        (gy0 / 64) as i32,
                        gx1.div_ceil(64).min(cw) as i32 - 1,
                        gy1.div_ceil(64).min(ch) as i32 - 1,
                    ],
                    _ => [0, 0, -1, -1],
                };
                let block = ao_changed_block(&regions, grass_coarse, &inp.eye.view_proj);
                device.write_buffer(slot.ao_changed, 0, &block)?;
                pc[4] = 1.0;
                pc[5] = f32::from(u8::from(grass_changed));
                pc[6] = f32::from(u8::from(water_drawn));
                stats.ssao_drawn = true;
                Self::post_pass(enc, t.ao_tex, 0, Load::Load, p.ssao, slot.ssao, pc);
            }
        } else {
            if fallback {
                stats.ao_changed_pixels = full_pixels;
            }
            stats.ssao_drawn = true;
            Self::post_pass(enc, t.ao_tex, 0, Load::DontCare, p.ssao, slot.ssao, pc);
        }
        Ok(next)
    }
}
