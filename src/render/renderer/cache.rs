use super::*;

/// Footprints of moved props (a shadow map apiece) `fs_reshade_mask` tests at most; more boxes
/// are joined into one.
pub(super) const MAX_RESHADE_ENTRIES: usize = 32;
/// Bytes of the re-shade block (`Reshade` in scene.wgsl): a count, the key light's near cascade's
/// margins, then two vec4s an entry.
pub(super) const RESHADE_BYTES: u64 = 32 + MAX_RESHADE_ENTRIES as u64 * 32;
/// Boxes of props moved since the water's lamp light and shadow were kept that the water follows
/// (`fs_water` computes them within their shadows' footprints), at most: a footprint in each of
/// the four maps fits the re-shade block. More have them drawn whole again.
pub(super) const MAX_WATER_MOVED: usize = MAX_RESHADE_ENTRIES / 4;

/// Where each world draw lies, for re-shading the cached picture ([`Renderer::reshade_plan`]):
/// in each shadow map props cast into (the key light's cascades 0, 1, grown by the offset its
/// receivers sample from; a game's lights 0, 1) its clip-space rect and farthest depth, and in
/// the picture. Made when first needed after the picture is drawn whole (the maps and the view
/// hold while it holds).
pub(super) struct ReceiverBounds {
    pub(super) key: u64,
    pub(super) maps: Vec<[([f32; 4], f32); 4]>,
    pub(super) screen: Vec<Option<Rect>>,
    /// Where the water draws in view lie in the shadow maps: the water's kept lamp light and
    /// shadow (`cache_water`) follow the moved props whose shadows can reach one.
    pub(super) water: Vec<[([f32; 4], f32); 4]>,
}

/// Bounds in a shadow map no footprint reaches ([`reached`]).
pub(super) const OUT_OF_REACH: ([f32; 4], f32) = ([1.0, 1.0, -1.0, -1.0], f32::NEG_INFINITY);

/// Pairs of a receiver and a footprint whose shadow volume [`Renderer::reshade_plan`] bounds in a
/// frame, at most (a microsecond or so apiece natively): past them a receiver in reach adds all
/// of its picture, as each did before the volumes bounded it.
pub(super) const MAX_RESHADE_PAIRS: u32 = 128;

/// A moved prop's footprint in one shadow map props cast into ([`Renderer::shadow_footprints`]):
/// the map (0, 1 the key light's cascades; 2, 3 a game's lights), its clip-space rect and its
/// nearest depth.
pub(super) type Footprint = (usize, [f32; 4], f32);

/// Whether a receiver with these bounds in the shadow maps can sample one of `footprints`: its
/// bounds in that map overlap the footprint and reach past its nearest depth.
pub(super) fn reached(footprints: &[Footprint], maps: &[([f32; 4], f32); 4]) -> bool {
    footprints.iter().any(|(v, fr, near)| {
        let (r, far) = &maps[*v];
        r[0] <= fr[2] && fr[0] <= r[2] && r[1] <= fr[3] && fr[1] <= r[3] && *far >= *near
    })
}

/// The `Reshade` block (scene.wgsl) for `footprints`: their count and the light buffer index + 1
/// of a game's light casting into each shadow layer (`shadowed`, 0: none), the key light's near
/// cascade's `margins`, then per footprint its rect, nearest depth, map and that map's `margins`
/// (clip x and y, depth: how far a lookup can lie from the point it shades,
/// [`Renderer::lookup_margins`]).
pub(super) fn reshade_block(
    footprints: &[Footprint],
    shadowed: [u32; 2],
    margins: [[f32; 2]; 4],
) -> Vec<u32> {
    let mut block = vec![footprints.len() as u32, shadowed[0], shadowed[1], 0];
    let [xy, depth] = margins[0];
    block.extend([xy, depth, 0.0, 0.0].map(f32::to_bits));
    for (v, r, near) in footprints {
        block.extend(r.map(f32::to_bits));
        let [xy, depth] = margins[*v];
        block.extend([*near, *v as f32, xy, depth].map(f32::to_bits));
    }
    block
}

/// How much farther than its cascade's offset, at the most a grazing light moves it (2.5 times),
/// a key light's lookup may lie from the point it shades ([`Renderer::prop_shadow_maps`]): the
/// normal it is moved along is read back from a half-float target, a hair longer than a unit.
pub(super) const LOOKUP_SLACK: f32 = 1.01;

/// A convex volume: its six bounding planes (`h · [p, 1] >= 0` inside) and its corners, corner
/// `c` on planes `c & 1`, `2 + (c >> 1 & 1)` and `4 + (c >> 2)` (low or high x, y and z), as
/// [`corners`] numbers a box's.
pub(super) struct Hull {
    pub(super) planes: [[f32; 4]; 6],
    pub(super) corners: [V3; 8],
}

pub(super) fn box_hull((lo, hi): (V3, V3)) -> Hull {
    Hull {
        planes: std::array::from_fn(|i| {
            let k = i / 2;
            let mut h = [0.0; 4];
            (h[k], h[3]) = if i % 2 == 0 {
                (1.0, -lo[k])
            } else {
                (-1.0, hi[k])
            };
            h
        }),
        corners: corners((lo, hi)),
    }
}

/// The world points within `grow` metres of one whose lookup in shadow map `m` falls in the box
/// `lo..hi` (clip x, y and depth) ahead of the map's eye. None when its corners do not lie ahead
/// of that eye.
pub(super) fn map_hull(m: &M4, lo: V3, hi: V3, grow: f32) -> Option<Hull> {
    let row = |r: usize| [m[r], m[4 + r], m[8 + r], m[12 + r]];
    let planes: [[f32; 4]; 6] = std::array::from_fn(|i| {
        let k = i / 2;
        let (s, bound) = if i % 2 == 0 {
            (1.0, lo[k])
        } else {
            (-1.0, hi[k])
        };
        // x / w >= lo is x - lo w >= 0 ahead of the eye (w > 0); the plane moved out by `grow`
        // metres holds the points within `grow` of its side.
        let h: [f32; 4] = std::array::from_fn(|j| s * (row(k)[j] - bound * row(3)[j]));
        [
            h[0],
            h[1],
            h[2],
            h[3] + grow * math::length([h[0], h[1], h[2]]),
        ]
    });
    let normal = |h: [f32; 4]| [h[0], h[1], h[2]];
    let corners: [V3; 8] = std::array::from_fn(|c| {
        let [a, b, d] = [
            planes[c & 1],
            planes[2 + (c >> 1 & 1)],
            planes[4 + (c >> 2)],
        ];
        let (bd, da, ab) = (
            cross(normal(b), normal(d)),
            cross(normal(d), normal(a)),
            cross(normal(a), normal(b)),
        );
        let p = add(add(scale(bd, a[3]), scale(da, b[3])), scale(ab, d[3]));
        scale(p, -1.0 / math::dot(normal(a), bd))
    });
    let ahead =
        |p: &V3| p.iter().all(|x| x.is_finite()) && transform(m, [p[0], p[1], p[2], 1.0])[3] > 0.0;
    corners
        .iter()
        .all(ahead)
        .then_some(Hull { planes, corners })
}

/// The ends of the parts of `a`'s edges that lie in `b`, into `out`.
pub(super) fn clip_edges(a: &Hull, b: &Hull, out: &mut Vec<V3>) {
    'edges: for (c, bit) in (0..8).flat_map(|c| [1, 2, 4].map(|bit| (c, bit))) {
        if c & bit != 0 {
            continue;
        }
        let (p, q) = (a.corners[c], a.corners[c | bit]);
        let (mut t0, mut t1) = (0.0f32, 1.0f32);
        for h in &b.planes {
            let side = |x: V3| h[0] * x[0] + h[1] * x[1] + h[2] * x[2] + h[3];
            let (dp, dq) = (side(p), side(q));
            if dp < 0.0 && dq < 0.0 {
                continue 'edges;
            }
            if dp < 0.0 {
                t0 = t0.max(dp / (dp - dq));
            } else if dq < 0.0 {
                t1 = t1.min(dp / (dp - dq));
            }
        }
        if t0 <= t1 {
            out.extend([math::lerp3(p, q, t0), math::lerp3(p, q, t1)]);
        }
    }
}

/// Into `out` (emptied first), points whose convex hull holds every point of a surface in box `b`
/// that can see footprint `f` as `fs_reshade_mask` tests it (none when no point can): the points
/// of the box whose lookups in the footprint's map (`maps`; the key light's lie off the surface
/// by up to its receivers' margin) fall in the footprint behind its nearest depth. They lie in
/// the box and in the footprint's box in the map, its depth reaching past the box (the footprint
/// swept along the light) and grown by that margin; every corner of the volume both hold is where
/// an edge of one crosses into the other. The shadow filter adds nothing here: its taps (within
/// the unit disc, `TAPS`) and the comparison's bilinear texels sample where the footprint is grown
/// to. The box and the margin grow by a hundred-thousandth of the coordinates, far more than
/// rounding moves the planes and the points the picture's depth gives back.
pub(super) fn shadow_volume(
    b: (V3, V3),
    &(v, r, near): &Footprint,
    maps: &[(M4, f32, f32); 4],
    out: &mut Vec<V3>,
) {
    out.clear();
    let (m, _, grow) = &maps[v];
    let eps = 1e-5
        * corners(b)
            .iter()
            .flatten()
            .fold(1.0f32, |a, x| a.max(x.abs()));
    let b = (b.0.map(|x| x - eps), b.1.map(|x| x + eps));
    // A moved box across a spot light's eye has a footprint over all of its map.
    if r.iter().chain([&near]).any(|x| x.abs() > 1e6) {
        return out.extend(corners(b));
    }
    // As deep as the box reaches in the map (ahead of a spot light, where the deepest of its
    // points is a corner: depth grows with the distance along the light), and no deeper than a
    // game light's range (depth 1, its map's far plane): past it `game_light_share` is 1 whatever
    // moved.
    let far = corners(b)
        .iter()
        .map(|p| transform(m, [p[0], p[1], p[2], 1.0]))
        .filter(|c| c[3] > 1e-6)
        .map(|c| c[2] / c[3])
        .fold(f32::NEG_INFINITY, f32::max);
    let far = if v >= 2 { far.min(1.0) } else { far };
    if far == f32::NEG_INFINITY || v >= 2 && near > far {
        return;
    }
    let (lo, hi) = ([r[0], r[1], near], [r[2], r[3], far.max(near)]);
    let Some(volume) = map_hull(m, lo, hi, grow + eps) else {
        return out.extend(corners(b));
    };
    let receiver = box_hull(b);
    clip_edges(&receiver, &volume, out);
    clip_edges(&volume, &receiver, out);
}

/// Whether the key light's shadow at every point in the convex hull of `points` takes nothing
/// from its far cascade (`shadow_taps`): they lie nearer `eye` than where the near cascade starts
/// to blend into it, and their lookups (off the surface by up to `grow` metres) fall in the near
/// cascade's map `m`, short of its border by a hair more than rounding moves them.
pub(super) fn near_cascade_only(points: &[V3], eye: V3, split: f32, m: &M4, grow: f32) -> bool {
    let margin = |r: usize| grow * math::length([m[r], m[4 + r], m[8 + r]]) + 1e-4;
    points.iter().all(|p| {
        let c = transform(m, [p[0], p[1], p[2], 1.0]);
        math::length(sub(*p, eye)) < 0.85 * split * (1.0 - 1e-4)
            && c[0].abs() < 0.996 - margin(0)
            && c[1].abs() < 0.996 - margin(1)
            && margin(2) < c[2]
            && c[2] < 1.0 - margin(2)
    })
}

/// The smallest clip-space rect holding both.
pub(super) fn union_rect(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

/// The pixels both regions hold.
pub(super) fn intersect(a: Rect, b: Rect) -> Option<Rect> {
    let r = [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ];
    (r[0] < r[2] && r[1] < r[3]).then_some(r)
}

/// Slices of view depth per doubling of it (`CLUSTER_SLICES` from `CLUSTER_NEAR` to `CLUSTER_FAR`).
pub(super) fn cluster_scale() -> f32 {
    CLUSTER_SLICES as f32 / (CLUSTER_FAR / CLUSTER_NEAR).log2()
}

/// The most lamps each cluster keeps so the clusters' `lengths` together fit in `room`.
pub(super) fn cluster_cap(lengths: impl Iterator<Item = usize> + Clone, room: usize) -> usize {
    let total = |cap: usize| lengths.clone().map(|n| n.min(cap)).sum::<usize>();
    let (mut lo, mut hi) = (0, MAX_LIGHTS_PER_CLUSTER);
    if total(hi) <= room {
        return hi;
    }
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if total(mid) <= room {
            lo = mid
        } else {
            hi = mid - 1
        }
    }
    lo
}

/// The cluster slice a view depth `z` falls in, as `lamp_light` finds it.
pub(super) fn cluster_slice(z: f32) -> u32 {
    if z <= CLUSTER_NEAR {
        return 0;
    }
    ((z / CLUSTER_NEAR).log2() * cluster_scale())
        .floor()
        .min((CLUSTER_SLICES - 1) as f32) as u32
}

impl ReceiverBounds {
    /// Where each of `draws` lies in `maps` ([`Renderer::prop_shadow_maps`]) and in a `size`
    /// picture seen through `view_proj`. The water and the draws outside the view (the test
    /// `draw_world` culls with) get none, which no footprint reaches: the picture holds no
    /// pixel of them. Indices stay those of `draws`. The water in view is kept apart (`water`).
    pub(super) fn new(
        key: u64,
        draws: &[DrawItem],
        maps: &[(M4, f32, f32); 4],
        view_proj: &M4,
        size: (u32, u32),
    ) -> Self {
        let view = Frustum::from_view_proj(view_proj);
        let seen = |d: &&DrawItem| view.intersects_aabb(d.bounds.0, d.bounds.1);
        let in_maps = |d: &DrawItem| {
            std::array::from_fn(|v| {
                let (m, margin, grow) = &maps[v];
                let grown = (d.bounds.0.map(|x| x - grow), d.bounds.1.map(|x| x + grow));
                receiver_map_bounds(grown, m, (v >= 2).then_some(*margin))
            })
        };
        let (bounds, screen) = draws
            .iter()
            .map(|d| {
                if d.water || !seen(&d) {
                    return ([OUT_OF_REACH; 4], None);
                }
                (in_maps(d), box_rect(d.bounds, view_proj, size, 2.0))
            })
            .unzip();
        Self {
            key,
            maps: bounds,
            screen,
            water: draws
                .iter()
                .filter(|d| d.water)
                .filter(seen)
                .map(in_maps)
                .collect(),
        }
    }
}

/// A box in a shadow map seen through `m`: its clip-space rect, nearest and farthest depth, or
/// all of it when part lies behind a spot light.
pub(super) fn map_bounds(b: (V3, V3), m: &M4) -> ([f32; 4], f32, f32) {
    let mut r = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    let (mut near, mut far) = (f32::INFINITY, f32::NEG_INFINITY);
    for p in corners(b) {
        let c = transform(m, [p[0], p[1], p[2], 1.0]);
        if c[3] <= 1e-6 {
            return ([-1e9, -1e9, 1e9, 1e9], -1e9, 1e9);
        }
        let q = [c[0] / c[3], c[1] / c[3], c[2] / c[3]];
        r = [
            r[0].min(q[0]),
            r[1].min(q[1]),
            r[2].max(q[0]),
            r[3].max(q[1]),
        ];
        (near, far) = (near.min(q[2]), far.max(q[2]));
    }
    (r, near, far)
}

/// Receiver box `b` in shadow map `m` (`spot_margin`: `Some(margin)` for a perspective spotlight
/// map, where points behind the light `c.w <= 1e-6`, beyond range `q.z > 1 + 1e-3`, or outside
/// `[-1 - margin, 1 + margin]^2` have `game_light_share == 1.0` identically). Expands `b` by
/// world-space depth-reconstruction slack `eps` and clips against `c.w >= 1e-6` so tall walls,
/// roofs, and water extending above a spotlight do not expand to `[-1e9, 1e9]`.
pub(super) fn receiver_map_bounds(
    b: (V3, V3),
    m: &M4,
    spot_margin: Option<f32>,
) -> ([f32; 4], f32) {
    let perspective = m[3] != 0.0 || m[7] != 0.0 || m[11] != 0.0;
    let Some(margin) = spot_margin.filter(|_| perspective) else {
        let (r, _, far) = map_bounds(b, m);
        return (r, far);
    };
    const FRONT: f32 = 1e-6;
    let eps = 1e-5
        * corners(b)
            .iter()
            .flatten()
            .fold(1.0f32, |a, x| a.max(x.abs()));
    let b = (b.0.map(|x| x - eps), b.1.map(|x| x + eps));
    let clip = corners(b).map(|p| transform(m, [p[0], p[1], p[2], 1.0]));
    let mut r = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    let (mut near, mut far, mut any) = (f32::INFINITY, f32::NEG_INFINITY, false);
    let mut push = |c: [f32; 4]| {
        let q = [c[0] / c[3], c[1] / c[3], c[2] / c[3]];
        r = [
            r[0].min(q[0]),
            r[1].min(q[1]),
            r[2].max(q[0]),
            r[3].max(q[1]),
        ];
        (near, far, any) = (near.min(q[2]), far.max(q[2]), true);
    };
    for c in clip {
        if c[3] >= FRONT {
            push(c);
        }
    }
    for i in 0..8 {
        for bit in [1, 2, 4] {
            if i & bit == 0 {
                let (a, b_c) = (clip[i], clip[i | bit]);
                if (a[3] < FRONT) != (b_c[3] < FRONT) {
                    let f = (FRONT - a[3]) / (b_c[3] - a[3]);
                    let mut c = std::array::from_fn(|k| a[k] + (b_c[k] - a[k]) * f);
                    c[3] = FRONT;
                    push(c);
                }
            }
        }
    }
    let xy_lim = 1.0 + margin;
    let z_lim = 1.0 + 1e-3;
    if !any || near > z_lim || r[0] > xy_lim || r[2] < -xy_lim || r[1] > xy_lim || r[3] < -xy_lim {
        return OUT_OF_REACH;
    }
    (r, far.min(z_lim))
}

/// Screen rectangle of the shadow prism `shadow_volume(b, f, maps)` (`points`) clipped to the
/// camera frustum when any prism vertex crosses behind/near the camera or outside the viewport.
pub(super) fn volume_screen_rect(
    b: (V3, V3),
    f: &Footprint,
    maps: &[(M4, f32, f32); 4],
    view_proj: &M4,
    near: f32,
    size: (u32, u32),
    points: &[V3],
) -> Option<Rect> {
    if points.is_empty() {
        return None;
    }
    let front_w = (near * 0.99).max(1e-4);
    let all_inside = points.iter().all(|p| {
        let c = transform(view_proj, [p[0], p[1], p[2], 1.0]);
        c[3] >= front_w && c[0].abs() <= c[3] && c[1].abs() <= c[3]
    });
    if all_inside {
        return screen_rect(points, view_proj, size, 1.0);
    }
    let &(v, fr, f_near) = f;
    let (m, _, grow) = &maps[v];
    let scale_coord = corners(b)
        .iter()
        .flatten()
        .fold(1.0f32, |a, x| a.max(x.abs()));
    let eps = 1e-5 * scale_coord;
    let b_eps = (b.0.map(|x| x - eps), b.1.map(|x| x + eps));
    let mut planes: Vec<[f64; 4]> = Vec::with_capacity(18);
    let mut push_plane = |h: [f64; 4]| {
        let len2 = h[0] * h[0] + h[1] * h[1] + h[2] * h[2];
        if len2 >= 1e-24 {
            let inv = 1.0 / len2.sqrt();
            planes.push([h[0] * inv, h[1] * inv, h[2] * inv, h[3] * inv]);
        }
    };
    for h in box_hull(b_eps).planes {
        push_plane(h.map(f64::from));
    }
    if !fr.iter().chain([&f_near]).any(|x| x.abs() > 1e6) {
        let far = corners(b_eps)
            .iter()
            .map(|p| transform(m, [p[0], p[1], p[2], 1.0]))
            .filter(|c| c[3] > 1e-6)
            .map(|c| c[2] / c[3])
            .fold(f32::NEG_INFINITY, f32::max);
        let far = if v >= 2 { far.min(1.0 + 1e-3) } else { far };
        if far == f32::NEG_INFINITY || v >= 2 && f_near > far {
            return None;
        }
        let (lo, hi) = ([fr[0], fr[1], f_near], [fr[2], fr[3], far.max(f_near)]);
        if let Some(vol) = map_hull(m, lo, hi, grow + eps) {
            for h in vol.planes {
                push_plane(h.map(f64::from));
            }
        }
    }
    let row = |r: usize| {
        [
            view_proj[r],
            view_proj[4 + r],
            view_proj[8 + r],
            view_proj[12 + r],
        ]
        .map(f64::from)
    };
    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
    for h in [
        std::array::from_fn(|j| 1.001 * r3[j] + r0[j]),
        std::array::from_fn(|j| 1.001 * r3[j] - r0[j]),
        std::array::from_fn(|j| 1.001 * r3[j] + r1[j]),
        std::array::from_fn(|j| 1.001 * r3[j] - r1[j]),
        std::array::from_fn(|j| 1.001 * r3[j] - r2[j]),
        [r3[0], r3[1], r3[2], r3[3] - 1e-4],
    ] {
        push_plane(h);
    }
    let tol = -1e-4 * f64::from(scale_coord);
    let n = planes.len();
    let (w_px, h_px) = (size.0 as f32, size.1 as f32);
    let (mut lo, mut hi, mut any) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2], false);
    let cross3 = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    for i in 0..n {
        let pi = planes[i];
        let ni = [pi[0], pi[1], pi[2]];
        for j in (i + 1)..n {
            if i < 6 && j == (i ^ 1) {
                continue;
            }
            let pj = planes[j];
            let nj = [pj[0], pj[1], pj[2]];
            let ab = cross3(ni, nj);
            if ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2] <= 1e-18 {
                continue;
            }
            for k in (j + 1)..n {
                if j < 6 && k == (j ^ 1) {
                    continue;
                }
                let pk = planes[k];
                let nk = [pk[0], pk[1], pk[2]];
                let det = ni[0] * (nj[1] * nk[2] - nj[2] * nk[1])
                    + ni[1] * (nj[2] * nk[0] - nj[0] * nk[2])
                    + ni[2] * (nj[0] * nk[1] - nj[1] * nk[0]);
                if det.abs() <= 1e-10 {
                    continue;
                }
                let bd = cross3(nj, nk);
                let da = cross3(nk, ni);
                let inv = -1.0 / det;
                let x = [
                    (bd[0] * pi[3] + da[0] * pj[3] + ab[0] * pk[3]) * inv,
                    (bd[1] * pi[3] + da[1] * pj[3] + ab[1] * pk[3]) * inv,
                    (bd[2] * pi[3] + da[2] * pj[3] + ab[2] * pk[3]) * inv,
                ];
                if !x.iter().all(|v| v.is_finite()) {
                    continue;
                }
                if planes
                    .iter()
                    .all(|h| h[0] * x[0] + h[1] * x[1] + h[2] * x[2] + h[3] >= tol)
                {
                    let c = transform(view_proj, [x[0] as f32, x[1] as f32, x[2] as f32, 1.0]);
                    let cw = c[3].max(5e-5);
                    let sx = (c[0] / cw * 0.5 + 0.5) * w_px;
                    let sy = (c[1] / cw * 0.5 + 0.5) * h_px;
                    lo = [lo[0].min(sx), lo[1].min(sy)];
                    hi = [hi[0].max(sx), hi[1].max(sy)];
                    any = true;
                }
            }
        }
    }
    if !any {
        return None;
    }
    let r = [
        (lo[0] - 1.0).floor().clamp(0.0, w_px) as u32,
        (lo[1] - 1.0).floor().clamp(0.0, h_px) as u32,
        (hi[0] + 1.0).ceil().clamp(0.0, w_px) as u32,
        (hi[1] + 1.0).ceil().clamp(0.0, h_px) as u32,
    ];
    (r[0] < r[2] && r[1] < r[3]).then_some(r)
}

/// The picture's pixels a frame shades again because moved props' shadows can reach them
/// ([`Renderer::reshade_plan`]).
pub(super) struct ReshadePlan {
    /// The `Reshade` block: per entry the footprint (clip-space rect), the nearest depth of
    /// what moved and the shadow map (0, 1 the key light's cascades, 2, 3 a game's lights).
    pub(super) block: Vec<u32>,
    /// The world's draws a footprint can reach (the rest cannot be in a marked pixel).
    pub(super) draws: Vec<bool>,
    /// The region of the picture holding them.
    pub(super) region: Rect,
}
/// Regions of the cached picture drawn again in one frame, at most; more are drawn as the one
/// region holding them all.
pub(super) const MAX_CACHE_REGIONS: usize = 6;

/// The regions to draw again, overlapping ones joined, at most [`MAX_CACHE_REGIONS`].
pub(super) fn merge_regions(mut rects: Vec<Rect>) -> Vec<Rect> {
    let overlap = |a: &Rect, b: &Rect| a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3];
    let mut merged = true;
    while merged {
        merged = false;
        'scan: for i in 0..rects.len() {
            for j in i + 1..rects.len() {
                if overlap(&rects[i], &rects[j]) {
                    let b = rects.swap_remove(j);
                    rects[i] = union(rects[i], b);
                    merged = true;
                    break 'scan;
                }
            }
        }
    }
    if rects.len() > MAX_CACHE_REGIONS {
        rects = rects.into_iter().reduce(union).into_iter().collect();
    }
    rects
}
