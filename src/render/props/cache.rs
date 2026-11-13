use super::*;
/// Runs of a batch's instances drawn apart at most ([`Renderer::instance_runs`]); a batch whose
/// instances in view lie in more is drawn whole.
pub(super) const MAX_INSTANCE_RUNS: usize = 4;

/// Which opaque prop batches a draw takes: all of them, those in the cached picture, or those
/// drawn over it each frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::render) enum PropSet {
    All,
    Settled,
    Live,
}

impl PropSet {
    pub(super) fn takes(self, b: &Batch) -> bool {
        match self {
            Self::All => true,
            Self::Settled => b.settled,
            Self::Live => !b.settled,
        }
    }
}

/// A region of the picture in pixels: left, top, right, bottom (exclusive).
pub(in crate::render) type Rect = [u32; 4];

/// Frames an instance stays the same before it is drawn into the cached picture: what moves now
/// and then (a card dealt, a chip pushed) stays live while it moves and a little after.
pub(super) const SETTLE_FRAMES: u32 = 8;

/// How far a caster may drift from where the props' shadow maps hold it, in texels of the finest
/// map it lands in, before they are drawn again ([`track_cached_props`]). Any drift shows: a
/// sphere half a texel off left 0.3 to 1 % of a frame more than one 8-bit step off the frame
/// drawn whole (up to 30 steps, along its own shadow's terminator and its shadow's edge), and a
/// fiftieth of a texel up to 0.1 %. So the maps are kept only while every caster stands still.
pub(in crate::render) const SHADOW_DRIFT_TEXELS: f32 = 0.0;

/// Hashes the bytes of `values` (as [`f32_bytes`] gives them, without making them).
pub(super) fn hash_f32s(h: &mut Fnv, values: &[f32]) {
    for v in values {
        h.write(&v.to_le_bytes());
    }
}

/// What an instance's picture depends on besides the frame: where and how it is drawn, its
/// influencing joints (`joints`, their [`mesh_joint_digest`]; 0 without), and its mesh, material
/// and textures as they stand.
pub(super) fn signature(
    inst: &InstanceEntry,
    joints: u64,
    mesh: &MeshEntry,
    mat: &MaterialEntry,
    edits: &HashMap<u32, u32>,
) -> u64 {
    let mut h = Fnv::new();
    h.write(&[u8::from(inst.visible)]);
    hash_f32s(&mut h, &inst.model);
    hash_f32s(&mut h, &inst.tint);
    h.write(&joints.to_le_bytes());
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
pub(super) fn settles(inst: &InstanceEntry, mat: &MaterialEntry) -> bool {
    inst.still >= SETTLE_FRAMES
        && inst.world_bounds.is_some()
        && mat.layer() == Layer::Opaque
        && mat.desc.shading != Shading::Water
}

/// Whether an instance is drawn into the props' shadow maps this frame
/// ([`Renderer::draw_prop_shadows`]).
pub(super) fn casts(inst: &InstanceEntry, mat: &MaterialEntry) -> bool {
    inst.world_bounds.is_some() && mat.layer() == Layer::Opaque && mat.desc.casts_shadow
}

/// Whether a mesh is drawn moved by an instance's `joints` (as `prepare_props` writes it).
pub(super) fn skinned(joints: &[M4], mesh: &MeshEntry) -> bool {
    mesh.skin
        .is_some_and(|s| joints.len() > usize::from(s.max_joint))
}

/// Hash of the joint matrices in `joints` that `m`'s vertices actually use (`0` when unskinned).
pub(super) fn mesh_joint_digest(joints: &[M4], m: &MeshEntry) -> u64 {
    if !skinned(joints, m) {
        return 0;
    }
    let mut h = Fnv::new();
    for (j, (joint, b)) in joints.iter().zip(&m.joint_boxes).enumerate() {
        if b.is_some() {
            h.write(&(j as u32).to_le_bytes());
            hash_f32s(&mut h, joint);
        }
    }
    h.finish()
}

/// What decides a caster's shadow as a whole rather than by how far it moved: its mesh and the
/// boxes [`posed_corners`] places (its joints' boxes, or for a mesh drawn unskinned its box),
/// and an alpha-tested material's cutoff, alpha and texture.
pub(super) fn shadow_key(
    inst: &InstanceEntry,
    joints: &[M4],
    mesh: &MeshEntry,
    mat: &MaterialEntry,
    edits: &HashMap<u32, u32>,
) -> u64 {
    let skinned = skinned(joints, mesh);
    let mut h = Fnv::new();
    for n in [inst.mesh.0, mesh.shape, u32::from(skinned)] {
        h.write(&n.to_le_bytes());
    }
    if !skinned {
        hash_f32s(&mut h, &mesh.min);
        hash_f32s(&mut h, &mesh.max);
    }
    if let Some(cutoff) = mat.desc.alpha_cutoff {
        hash_f32s(&mut h, &[cutoff, mat.desc.base_color[3]]);
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
pub(super) fn shadow_drift(
    inst: &InstanceEntry,
    mesh: &MeshEntry,
    mat: &MaterialEntry,
) -> Option<f32> {
    let track = &inst.shadow;
    match (&track.drawn, casts(inst, mat)) {
        (None, false) => None,
        (Some(d), true) if d.key == track.key && d.corners.len() == track.corners.len() => {
            Some(farthest_move(&d.corners, &track.corners) + track.stretch * mesh.drift)
        }
        _ => Some(f32::INFINITY),
    }
}

/// A shadow map props cast into: world to its clip space, and its texels a side.
pub(in crate::render) type ShadowMap = (M4, u32);

/// The most texels a point in `b` moves across a shadow map per metre it moves, or `None` when
/// the box lies outside the map. Its clip x/w moves by `|row_x - x/w row_w| / w` per metre (the
/// rows' xyz), largest where `w` is least and `|x/w|` most, both at a corner.
pub(super) fn texels_per_metre((m, size): &ShadowMap, b: (V3, V3)) -> Option<f32> {
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
pub(in crate::render) fn screen_rect(
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

/// The pixels a box can cover in a `size` picture seen through `view_proj`, widened by `margin`
/// pixels: its corners in front of the eye and where its edges cross the plane just before it,
/// projected (nothing behind it is drawn). `None` when that is no pixel.
pub(in crate::render) fn box_rect(
    (lo, hi): (V3, V3),
    view_proj: &M4,
    size: (u32, u32),
    margin: f32,
) -> Option<Rect> {
    const FRONT: f32 = 1e-4;
    let clip = corners((lo, hi)).map(|p| math::transform(view_proj, [p[0], p[1], p[2], 1.0]));
    let mut points: Vec<[f32; 4]> = clip.iter().copied().filter(|c| c[3] >= FRONT).collect();
    for i in 0..8 {
        for j in [1, 2, 4].map(|bit| i | bit).into_iter().filter(|&j| j != i) {
            let (a, b) = (clip[i], clip[j]);
            if (a[3] < FRONT) != (b[3] < FRONT) {
                let f = (FRONT - a[3]) / (b[3] - a[3]);
                points.push(std::array::from_fn(|k| a[k] + (b[k] - a[k]) * f));
            }
        }
    }
    let (w, h) = (size.0 as f32, size.1 as f32);
    let (mut lo, mut hi) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
    for c in &points {
        let p = [(c[0] / c[3] * 0.5 + 0.5) * w, (c[1] / c[3] * 0.5 + 0.5) * h];
        lo = [lo[0].min(p[0]), lo[1].min(p[1])];
        hi = [hi[0].max(p[0]), hi[1].max(p[1])];
    }
    let r = [
        (lo[0] - margin).floor().clamp(0.0, w) as u32,
        (lo[1] - margin).floor().clamp(0.0, h) as u32,
        (hi[0] + margin).ceil().clamp(0.0, w) as u32,
        (hi[1] + margin).ceil().clamp(0.0, h) as u32,
    ];
    (r[0] < r[2] && r[1] < r[3]).then_some(r)
}

pub(in crate::render) fn corners((lo, hi): (V3, V3)) -> [V3; 8] {
    std::array::from_fn(|c| {
        [
            if c & 1 == 0 { lo[0] } else { hi[0] },
            if c & 2 == 0 { lo[1] } else { hi[1] },
            if c & 4 == 0 { lo[2] } else { hi[2] },
        ]
    })
}

/// The smallest region holding both.
pub(in crate::render) fn union(a: Rect, b: Rect) -> Rect {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

/// What the cached picture is drawn for: the eye and the picture size.
pub(in crate::render) struct CacheView<'a> {
    pub(in crate::render) view_proj: &'a M4,
    pub(in crate::render) size: (u32, u32),
}

/// What changed in the cached picture this frame ([`track_cached_props`]).
#[derive(Default)]
pub(in crate::render) struct CacheChanges {
    /// Regions to empty and draw again: where an instance entered or left the picture.
    pub(in crate::render) regions: Vec<Rect>,
    /// Where casters moved (or changed, came or went) since the props' shadow maps were last
    /// drawn: their bounds then and now, overlapping ones joined. When the maps are drawn
    /// again, the picture's pixels their shadows can reach are shaded again
    /// ([`Renderer::reshade`]).
    pub(in crate::render) moved: Vec<(V3, V3)>,
    /// The bounds of the props the picture holds.
    pub(in crate::render) settled: Vec<(V3, V3)>,
    /// Where the props drawn over the picture changed the frame's depth and normals since the
    /// last frame: the pixels each one that changed (moved, came or went, settled or woke)
    /// covered then and covers now.
    pub(in crate::render) live: Vec<Rect>,
    /// A caster came, went, changed as a whole or moved across a shadow map it lands in (by
    /// `drift_texels` of the finest or more) since the maps were drawn: they are drawn again.
    /// Otherwise the maps and the picture keep it as they were.
    pub(in crate::render) casters_moved: bool,
}

/// Boxes that overlap joined into one, so casters moving together are followed as one.
pub(in crate::render) fn merge_boxes(mut boxes: Vec<(V3, V3)>) -> Vec<(V3, V3)> {
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
pub(in crate::render) fn track_cached_props(
    props: &mut Props,
    full: bool,
    view: &CacheView<'_>,
    maps: &[ShadowMap],
    drift_texels: f32,
) -> CacheChanges {
    let mut regions = std::mem::take(&mut props.vacated);
    let mut moved = std::mem::take(&mut props.vacated_boxes);
    let mut live = std::mem::take(&mut props.vacated_live);
    let mut casters_moved = !moved.is_empty();
    let mut settled = vec![];
    let Props {
        instances,
        meshes,
        materials,
        ..
    } = props;
    for (_, inst) in instances.iter_mut() {
        let Some(mat) = materials.get(inst.material.0) else {
            regions.extend(inst.cached.take());
            live.extend(inst.live.take());
            continue;
        };
        // Drawn over the picture, it writes the frame's depth and normals where it stands; they
        // change where it changed (its signature, `still` 0) or came, went, settled or woke.
        let drawn_live = inst
            .world_bounds
            .filter(|_| mat.layer() == Layer::Opaque && !settles(inst, mat))
            .and_then(|b| box_rect(b, view.view_proj, view.size, 2.0));
        if !full && (inst.still == 0 || drawn_live != inst.live) {
            live.extend(inst.live);
            live.extend(drawn_live);
        }
        inst.live = drawn_live;
        if settles(inst, mat) {
            if full || inst.cached.is_none() {
                let image = inst
                    .world_bounds
                    .and_then(|b| box_rect(b, view.view_proj, view.size, 2.0));
                regions.extend(image);
                inst.cached = Some(image.unwrap_or_default());
            }
            settled.extend(inst.world_bounds);
        } else {
            regions.extend(inst.cached.take());
        }
        let drift = meshes
            .get(inst.mesh.0)
            .and_then(|mesh| shadow_drift(inst, mesh, mat));
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
    live.retain(|r| r[0] < r[2] && r[1] < r[3]);
    CacheChanges {
        regions,
        moved: merge_boxes(moved),
        settled,
        live,
        casters_moved,
    }
}

/// Records that the props' shadow maps now hold every caster as this frame draws it: later
/// frames measure from here how far casters moved ([`shadow_drift`]).
pub(in crate::render) fn commit_prop_shadows(props: &mut Props) {
    let Props {
        instances,
        meshes,
        materials,
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
            (Some(_), Some(mat), Some(bounds)) if casts(inst, mat) => {
                Some((inst.shadow.key, bounds))
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
pub(super) fn joint_boxes(positions: &[V3], records: &[[f32; 4]]) -> Vec<Option<(V3, V3)>> {
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
pub(super) fn posed_corners(
    inst: &InstanceEntry,
    joints: &[M4],
    mesh: &MeshEntry,
    out: &mut Vec<V3>,
) -> f32 {
    let mut stretch = 0.0f32;
    let mut place = |out: &mut Vec<V3>, m: &M4, b: (V3, V3)| {
        out.extend(corners(b).map(|p| math::transform_point(m, p)));
        stretch = stretch.max(max_stretch(m));
    };
    out.clear();
    if skinned(joints, mesh) {
        for (joint, b) in joints.iter().zip(&mesh.joint_boxes) {
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
pub(super) fn max_stretch(m: &M4) -> f32 {
    let col = |c: usize| [m[c * 4], m[c * 4 + 1], m[c * 4 + 2]];
    let cols = [col(0), col(1), col(2)];
    cols.iter()
        .map(|a| cols.iter().map(|b| math::dot(*a, *b).abs()).sum::<f32>())
        .fold(0.0, f32::max)
        .sqrt()
}

/// Writes to `buffer` the rows of `data` (`row` numbers each) that differ from `held`, what it
/// holds, in the runs [`changed_runs`] makes for its write granularity (the unchanged rows a run
/// spans are written with the bits it holds). `held` then holds `data`, and `data` what `held`
/// held.
pub(super) fn write_changed(
    device: &mut Device<'_>,
    buffer: BufferId,
    held: &mut Vec<f32>,
    data: &mut Vec<f32>,
    row: usize,
) -> Result<()> {
    let at = |r: usize| r * row..(r + 1) * row;
    let changed = |r: usize| {
        held.get(at(r))
            .is_none_or(|old| !same_bits(old, &data[at(r)]))
    };
    let unit = device.write_granularity(buffer);
    for run in changed_runs(data.len() / row, row * 4, unit, changed) {
        let bytes = f32_bytes(&data[run.start * row..run.end * row]);
        device.write_buffer(buffer, (run.start * row * 4) as u64, &bytes)?;
    }
    std::mem::swap(held, data);
    Ok(())
}

/// The runs of rows to write of `rows` rows of `row_bytes` bytes, where `changed` says which
/// differ: each changed row in one, joined across a few unchanged rows between them (fewer,
/// larger writes), and across any number while they start in the `unit` (the buffer's write
/// granularity) the run ends in, so that no unit is sent twice.
pub(super) fn changed_runs(
    rows: usize,
    row_bytes: usize,
    unit: u64,
    changed: impl Fn(usize) -> bool,
) -> impl Iterator<Item = std::ops::Range<usize>> {
    const GAP: usize = 4;
    let unit = usize::try_from(unit).unwrap_or(usize::MAX).max(1);
    let mut r = 0;
    std::iter::from_fn(move || {
        while r < rows && !changed(r) {
            r += 1;
        }
        if r == rows {
            return None;
        }
        let (start, mut end) = (r, r + 1);
        r = end;
        while r < rows && (r - end <= GAP || r * row_bytes / unit <= (end * row_bytes - 1) / unit) {
            if changed(r) {
                end = r + 1;
            }
            r += 1;
        }
        Some(start..end)
    })
}

/// How many units of `unit` bytes writing `data` over `held` (rows of `row` numbers) sends when
/// its runs are made for write granularity `granularity`: for a benchmark of WebGL 2's storage
/// uploads, which send their texture's rows whole.
#[cfg(test)]
pub(in crate::render) fn units_sent(
    held: &[f32],
    data: &[f32],
    row: usize,
    granularity: u64,
    unit: usize,
) -> usize {
    let at = |r: usize| r * row..(r + 1) * row;
    let changed = |r: usize| {
        held.get(at(r))
            .is_none_or(|old| !same_bits(old, &data[at(r)]))
    };
    changed_runs(data.len() / row, row * 4, granularity, changed)
        .map(|run| (run.end * row * 4 - 1) / unit - run.start * row * 4 / unit + 1)
        .sum()
}

/// A game's prop shader: its compiled stages and the overlay pipelines made from them so far.
pub(in crate::render) struct PropShaderEntry {
    /// Its fragment stage (the vertex stage is every prop's).
    pub(super) fragment: ganbate_shader::Compiled,
    pub(super) pipelines: [[Option<crate::gpu::PipelineId>; 3]; 3],
}

/// The overlay pipeline index of a blend mode: opaque, alpha, additive.
pub(super) fn blend_index(blend: BlendMode) -> usize {
    match blend {
        BlendMode::Opaque => 0,
        BlendMode::Alpha => 1,
        BlendMode::Add => 2,
    }
}

/// The overlay pipeline index of how a material meets the scene's depth: drawn on top, tested,
/// tested and written.
pub(super) fn depth_index(desc: &MaterialDesc) -> usize {
    match (desc.depth_test, desc.depth_write) {
        (false, _) => 0,
        (true, false) => 1,
        (true, true) => 2,
    }
}

/// A frame slot's instance and joint buffers and the bind groups that read them.
pub(in crate::render) struct PropSlot {
    pub(super) buffer: BufferId,
    pub(super) joints: BufferId,
    /// What the buffers hold ([`write_changed`]).
    pub(super) rows: Vec<f32>,
    pub(super) joint_rows: Vec<f32>,
    /// Main view and reflection (`@group(0)` of the prop scene pipeline).
    pub(super) scene: [BindGroupId; 2],
    pub(super) shadow: BindGroupId,
    /// `@group(0)` of the overlay pipelines.
    pub(super) overlay: BindGroupId,
}

#[derive(Default)]
pub(in crate::render) struct Props {
    pub(super) meshes: Pool<MeshEntry>,
    pub(super) materials: Pool<MaterialEntry>,
    /// A game's prop shaders: their stages, and their overlay pipelines, [opaque, alpha,
    /// additive][depth test], each made the first time a prop draws with it.
    pub(super) shaders: Pool<PropShaderEntry>,
    pub(super) textures: Pool<TextureId>,
    pub(super) instances: Pool<InstanceEntry>,
    pub(super) joint_sets: Pool<JointSet>,
    /// Frames prepared so far (which places each joint set's rows once a frame).
    pub(super) frames: u64,
    pub(super) slots: Vec<Option<PropSlot>>,
    pub(super) capacity: u32,
    /// Joint matrices each slot's joint buffer holds.
    pub(super) joint_capacity: u32,
    /// Edits of each prop texture's pixels (by handle), which the pictures of the props using
    /// it follow, and how many times any texture was edited or destroyed ([`stamp`]).
    pub(super) texture_edits: HashMap<u32, u32>,
    pub(super) texture_epoch: u32,
    /// Regions of the cached picture held by instances destroyed since the last frame.
    pub(super) vacated: Vec<Rect>,
    /// Where the shadow maps hold casters destroyed since they were drawn (their shadows are
    /// drawn away).
    pub(super) vacated_boxes: Vec<(V3, V3)>,
    /// Where instances destroyed since the last frame were drawn over the cached picture.
    pub(super) vacated_live: Vec<Rect>,
    /// Every skinned mesh's vertices' joints and weights, packed in mesh order of creation.
    pub(super) skin_records: Vec<[f32; 4]>,
    /// Their GPU copy (and how many records it holds), rewritten when `skin_dirty`.
    pub(super) skin_buffer: Option<(BufferId, u32)>,
    pub(super) skin_dirty: bool,
    pub(super) batches: Vec<Batch>,
    /// Each of this frame's instances' world bounds, in the order the instance buffer holds them.
    pub(super) placed: Vec<(V3, V3)>,
    /// This frame's instance and joint rows as `prepare_props` makes them (kept to reuse).
    pub(super) rows: Vec<f32>,
    pub(super) joint_rows: Vec<f32>,
    /// World bounds of this frame's visible instances.
    pub(super) bounds: Option<(V3, V3)>,
}

/// What an instance depends on beyond itself: its mesh's, material's and joints' revisions
/// and, when the material has textures, how many times prop textures were edited.
pub(super) fn stamp(
    mesh: &MeshEntry,
    mat: &MaterialEntry,
    joints: Option<&JointSet>,
    texture_epoch: u32,
) -> [u32; 4] {
    let d = &mat.desc;
    let textured = [d.base_texture, d.normal_texture, d.emissive_texture]
        .iter()
        .any(Option::is_some);
    [
        mesh.revision,
        mat.revision,
        joints.map_or(0, |j| j.revision),
        if textured { texture_epoch } else { 0 },
    ]
}

/// Whether two runs of numbers hold the same bits.
pub(super) fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// A mesh's box placed by `model`.
pub(super) fn modelled_bounds(model: &M4, mesh: &MeshEntry) -> (V3, V3) {
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for p in corners((mesh.min, mesh.max)) {
        let w = math::transform_point(model, p);
        for k in 0..3 {
            lo[k] = lo[k].min(w[k]);
            hi[k] = hi[k].max(w[k]);
        }
    }
    (lo, hi)
}

impl Props {
    /// What each instance's picture depends on now and how long that has held, where its
    /// vertices lie as drawn ([`posed_corners`]) and its world bounds. Only instances that were
    /// set since, or whose mesh, material or textures changed, are posed again: the rest stand
    /// as they were, one more frame the same.
    pub(super) fn pose_instances(&mut self) {
        let Self {
            instances,
            meshes,
            materials,
            joint_sets,
            texture_edits,
            texture_epoch,
            ..
        } = self;
        for (_, inst) in instances.iter_mut() {
            let (Some(m), Some(mat)) = (meshes.get(inst.mesh.0), materials.get(inst.material.0))
            else {
                inst.world_bounds = None;
                inst.still = 0;
                inst.dirty = true;
                continue;
            };
            let set = inst.joints.and_then(|id| joint_sets.get(id));
            let stamp = stamp(m, mat, set, *texture_epoch);
            if !inst.dirty && stamp == inst.stamp {
                inst.still = inst.still.saturating_add(1);
                continue;
            }
            inst.dirty = false;
            inst.stamp = stamp;
            let joints = set.map_or(&[][..], |s| &s.matrices[..]);
            let sig = signature(inst, mesh_joint_digest(joints, m), m, mat, texture_edits);
            inst.still = if sig == inst.sig {
                inst.still.saturating_add(1)
            } else {
                0
            };
            inst.sig = sig;
            inst.shadow.key = shadow_key(inst, joints, m, mat, texture_edits);
            inst.world_bounds = None;
            if inst.visible {
                let mut posed = std::mem::take(&mut inst.shadow.corners);
                inst.shadow.stretch = posed_corners(inst, joints, m, &mut posed);
                inst.world_bounds = Some(bounds(&posed));
                inst.shadow.corners = posed;
                inst.modelled = modelled_bounds(&inst.model, m);
            }
        }
    }

    /// This frame's batches and prop bounds as a hash, for a benchmark to show two builds
    /// decide alike.
    #[cfg(test)]
    pub(in crate::render) fn digest(&self) -> u64 {
        let mut h = Fnv::new();
        h.write(format!("{:?} {:?}", self.batches, self.bounds).as_bytes());
        h.finish()
    }

    /// What frame slot `slot`'s instance and joint buffers hold.
    #[cfg(test)]
    pub(in crate::render) fn slot_rows(&self, slot: usize) -> Option<(&[f32], &[f32])> {
        let s = self.slots.get(slot)?.as_ref()?;
        Some((&s.rows, &s.joint_rows))
    }

    /// Removes an instance; what it held in the cached picture is drawn again.
    pub(super) fn remove_instance(&mut self, id: u32) {
        if let Some(i) = self.instances.remove(id) {
            self.vacated.extend(i.cached);
            self.vacated_boxes.extend(i.shadow.drawn.map(|d| d.bounds));
            self.vacated_live.extend(i.live);
            self.release_joints(i.joints);
        }
    }

    /// An error unless instance `id` lives and `count` joint matrices can skin it: none (drawn
    /// as modelled), or more than the highest joint its skinned mesh's vertices use.
    pub(super) fn check_joints(&self, id: u32, count: usize) -> Result<()> {
        let i = self
            .instances
            .get(id)
            .ok_or_else(|| format!("{:?} was destroyed", PropInstance(id)))?;
        if count > 0
            && let Some(skin) = self.meshes.get(i.mesh.0).and_then(|m| m.skin)
            && count <= usize::from(skin.max_joint)
        {
            return Err(format!(
                "set_prop_instance_joints: {count} matrices for joint {}",
                skin.max_joint
            ));
        }
        Ok(())
    }

    /// One user fewer for a joint set, which goes with its last.
    pub(super) fn release_joints(&mut self, set: Option<u32>) {
        if let Some(id) = set
            && let Some(s) = self.joint_sets.get_mut(id)
        {
            s.users -= 1;
            if s.users == 0 {
                self.joint_sets.remove(id);
            }
        }
    }

    /// Writes `joints` into a set, counting a change when they differ from what it holds.
    pub(super) fn write_joints(&mut self, id: u32, joints: &[M4]) {
        if let Some(s) = self.joint_sets.get_mut(id)
            && !same_bits(s.matrices.as_flattened(), joints.as_flattened())
        {
            s.matrices.clear();
            s.matrices.extend_from_slice(joints);
            s.revision = s.revision.wrapping_add(1);
        }
    }

    /// Makes `ids` (live instances) follow one joint set holding `joints` (`[]`: none, drawn as
    /// modelled): the one they follow already when it is theirs alone, or a new one.
    pub(super) fn share_joints(&mut self, ids: &[u32], joints: &[M4]) -> Result<()> {
        if let Some(&id) = ids.iter().find(|&&id| self.instances.get(id).is_none()) {
            return Err(format!("{:?} was destroyed", PropInstance(id)));
        }
        if ids.is_empty() {
            return Ok(());
        }
        // The set is theirs alone when its users are `ids`: as many as they, each following it,
        // none named twice (else another user could hide behind the count).
        let distinct = ids.iter().enumerate().all(|(k, id)| !ids[..k].contains(id));
        let theirs = ids
            .first()
            .and_then(|&id| self.instances.get(id)?.joints)
            .filter(|&set| {
                distinct
                    && self
                        .joint_sets
                        .get(set)
                        .is_some_and(|s| s.users as usize == ids.len())
                    && ids.iter().all(|&id| {
                        self.instances
                            .get(id)
                            .is_some_and(|i| i.joints == Some(set))
                    })
            });
        if let Some(set) = theirs.filter(|_| !joints.is_empty()) {
            self.write_joints(set, joints);
            return Ok(());
        }
        let set = if joints.is_empty() {
            None
        } else {
            Some(self.joint_sets.insert(JointSet {
                matrices: joints.to_vec(),
                revision: 0,
                users: 0,
                first: 0,
                placed: 0,
            })?)
        };
        for &id in ids {
            let Some(i) = self.instances.get_mut(id).filter(|i| i.joints != set) else {
                continue;
            };
            let old = std::mem::replace(&mut i.joints, set);
            i.dirty = true;
            if let Some(s) = set.and_then(|s| self.joint_sets.get_mut(s)) {
                s.users += 1;
            }
            self.release_joints(old);
        }
        Ok(())
    }
}

pub(super) fn normal_matrix(m: &M4) -> M4 {
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

/// The farthest any point moved from `from` to `to` (the square root of the largest squared
/// move: the largest of the moves' lengths, with one root).
pub(super) fn farthest_move(from: &[V3], to: &[V3]) -> f32 {
    from.iter()
        .zip(to)
        .map(|(a, b)| {
            let d = math::sub(*a, *b);
            math::dot(d, d)
        })
        .fold(0.0, f32::max)
        .sqrt()
}

pub(in crate::render) fn bounds(points: &[V3]) -> (V3, V3) {
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

/// Per axis, the zeros among `points` (1: +0, 2: -0). Bounds joined from parts are the bits
/// [`bounds`] of all of them gives, in any order, but where a bound is 0 and both zeros are
/// there: which one it holds then depends on the order they are met in.
pub(super) fn zeros(points: &[V3]) -> [u8; 3] {
    let mut zeros = [0u8; 3];
    for p in points {
        for k in 0..3 {
            if p[k] == 0.0 {
                zeros[k] |= 1 << (p[k].to_bits() >> 31);
            }
        }
    }
    zeros
}

/// Interleaved position, normal, uv (the world vertex layout).
pub(super) fn interleave(
    positions: &[V3],
    normals: &[V3],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<f32> {
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
pub(in crate::render) fn ray_triangle(o: V3, d: V3, a: V3, b: V3, c: V3) -> Option<f32> {
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
pub(in crate::render) fn ray_box(o: V3, d: V3, min: V3, max: V3) -> Option<f32> {
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

pub(super) fn normalize_pick_ray(origin: [f32; 3], dir: [f32; 3]) -> Option<(V3, V3)> {
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
    pub(super) fn pick_along_ray(
        &self,
        origin: [f32; 3],
        dir: [f32; 3],
        layers: u32,
    ) -> Option<Pick> {
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
            let skinned = self.skinned_positions(mesh, joints_of(&self.joint_sets, inst));
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
    pub(super) fn skinned_positions(&self, mesh: &MeshEntry, joints: &[M4]) -> Option<Vec<V3>> {
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
pub(super) fn canvas_clear(color: [f32; 4], srgb: bool) -> Result<[f32; 4]> {
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
