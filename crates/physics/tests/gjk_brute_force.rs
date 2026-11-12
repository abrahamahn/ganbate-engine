//! GJK distances and casts against brute force: every pair of convex kinds in random poses,
//! compared with an `f64` search over the cores' features (vertices, edges, face triangles).
//!
//! For two disjoint convex cores the closest pair is a vertex and a face triangle, or two edges
//! (Ericson, Real-Time Collision Detection 5.1.10); they intersect exactly when a vertex of one is
//! inside the other's solid or an edge of one crosses a face triangle of the other.

use std::ops::{Add, Mul, Sub};

use ganbate_physics::gjk::{MAX_GJK_ITERATIONS, cast, distance};
use ganbate_physics::hull::ConvexHull;
use ganbate_physics::math::{Iso, Quat, Vec3};
use ganbate_physics::shape::{Convex, ConvexShape};

#[derive(Clone, Copy, Debug, PartialEq)]
struct D3 {
    x: f64,
    y: f64,
    z: f64,
}

impl D3 {
    const ZERO: D3 = D3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    fn new(x: f64, y: f64, z: f64) -> D3 {
        D3 { x, y, z }
    }
    fn of(v: Vec3) -> D3 {
        D3::new(v.x as f64, v.y as f64, v.z as f64)
    }
    fn dot(self, o: D3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    fn cross(self, o: D3) -> D3 {
        D3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    fn length(self) -> f64 {
        self.dot(self).sqrt()
    }
}

impl Add for D3 {
    type Output = D3;
    fn add(self, o: D3) -> D3 {
        D3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl Sub for D3 {
    type Output = D3;
    fn sub(self, o: D3) -> D3 {
        D3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Mul<f64> for D3 {
    type Output = D3;
    fn mul(self, s: f64) -> D3 {
        D3::new(self.x * s, self.y * s, self.z * s)
    }
}

/// `q` applied to `v` in `f64`.
fn rotate(q: Quat, v: D3) -> D3 {
    let u = D3::new(q.x as f64, q.y as f64, q.z as f64);
    let t = u.cross(v) * 2.0;
    v + t * q.w as f64 + u.cross(t)
}

fn inverse_rotate(q: Quat, v: D3) -> D3 {
    rotate(Quat::new(-q.x, -q.y, -q.z, q.w), v)
}

/// A core's features in world space, and its solid's local planes if it has an inside.
struct Core {
    points: Vec<D3>,
    segments: Vec<(D3, D3)>,
    triangles: Vec<[D3; 3]>,
    iso: Iso,
    planes: Vec<(D3, f64)>,
}

impl Core {
    fn of(c: &Convex) -> Core {
        let world = |p: Vec3| rotate(c.iso.rotation, D3::of(p)) + D3::of(c.iso.position);
        let mut core = Core {
            points: vec![],
            segments: vec![],
            triangles: vec![],
            iso: c.iso,
            planes: vec![],
        };
        let polyhedron = |h: &ConvexHull, core: &mut Core| {
            core.points = h.vertices.iter().map(|&v| world(v)).collect();
            for e in h.edges.iter().step_by(2) {
                let twin = &h.edges[e.twin as usize];
                core.segments.push((
                    world(h.vertices[e.origin as usize]),
                    world(h.vertices[twin.origin as usize]),
                ));
            }
            for (f, face) in h.faces.iter().enumerate() {
                let v = h.face_vertices(f);
                for k in 1..v.len() - 1 {
                    core.triangles
                        .push([world(v[0]), world(v[k]), world(v[k + 1])]);
                }
                core.planes.push((D3::of(face.normal), face.offset as f64));
            }
        };
        match c.shape {
            ConvexShape::Sphere(_) => core.points.push(world(Vec3::ZERO)),
            ConvexShape::Capsule { half_height, .. } => {
                let (a, b) = (
                    world(Vec3::new(0.0, -half_height, 0.0)),
                    world(Vec3::new(0.0, half_height, 0.0)),
                );
                core.points = vec![a, b];
                core.segments.push((a, b));
            }
            ConvexShape::Segment { a, b, .. } => {
                let (a, b) = (world(a), world(b));
                core.points = vec![a, b];
                core.segments.push((a, b));
            }
            ConvexShape::Triangle(t) => {
                let t = t.map(world);
                core.points = t.to_vec();
                core.segments = vec![(t[0], t[1]), (t[1], t[2]), (t[2], t[0])];
                core.triangles.push(t);
            }
            ConvexShape::Box(h) => polyhedron(&ConvexHull::cuboid(h), &mut core),
            ConvexShape::Hull(h) => polyhedron(h, &mut core),
        }
        core
    }

    /// The same core with polyhedron corners placed in `f32` (what GJK's support function
    /// returns), to tell rounding of the input from rounding in GJK.
    fn of_rounded(c: &Convex) -> Core {
        let mut core = Core::of(c);
        let hull = match c.shape {
            ConvexShape::Box(h) => ConvexHull::cuboid(h),
            ConvexShape::Hull(h) => h.clone(),
            _ => return core,
        };
        let world = |p: Vec3| D3::of(c.iso.point(p));
        core.points = hull.vertices.iter().map(|&v| world(v)).collect();
        core.segments.clear();
        for e in hull.edges.iter().step_by(2) {
            let twin = &hull.edges[e.twin as usize];
            core.segments.push((
                world(hull.vertices[e.origin as usize]),
                world(hull.vertices[twin.origin as usize]),
            ));
        }
        core.triangles.clear();
        for f in 0..hull.faces.len() {
            let v = hull.face_vertices(f);
            for k in 1..v.len() - 1 {
                core.triangles
                    .push([world(v[0]), world(v[k]), world(v[k + 1])]);
            }
        }
        core
    }

    /// How deep a world point is inside the solid (positive inside), or `None` for flat cores.
    fn depth(&self, p: D3) -> Option<f64> {
        if self.planes.is_empty() {
            return None;
        }
        let local = inverse_rotate(self.iso.rotation, p - D3::of(self.iso.position));
        Some(
            self.planes
                .iter()
                .map(|&(n, d)| d - n.dot(local))
                .fold(f64::INFINITY, f64::min),
        )
    }

    /// The distance from a world point to the core (0 inside a solid).
    fn distance_to(&self, p: D3) -> f64 {
        if self.depth(p).is_some_and(|d| d >= 0.0) {
            return 0.0;
        }
        let mut best = f64::INFINITY;
        for &q in &self.points {
            best = best.min((p - q).length());
        }
        for &(a, b) in &self.segments {
            best = best.min((p - closest_on_segment(p, a, b)).length());
        }
        for t in &self.triangles {
            best = best.min((p - closest_on_triangle(p, t)).length());
        }
        best
    }
}

fn closest_on_segment(p: D3, a: D3, b: D3) -> D3 {
    let ab = b - a;
    let ll = ab.dot(ab);
    let t = if ll > 0.0 {
        ((p - a).dot(ab) / ll).clamp(0.0, 1.0)
    } else {
        0.0
    };
    a + ab * t
}

/// Ericson 5.1.5, with degenerate triangles handled by their edges.
fn closest_on_triangle(p: D3, t: &[D3; 3]) -> D3 {
    let [a, b, c] = *t;
    let (ab, ac, ap) = (b - a, c - a, p - a);
    if ab.cross(ac).length() <= 1e-12 * ab.length() * ac.length() {
        let mut best = closest_on_segment(p, a, b);
        for q in [closest_on_segment(p, b, c), closest_on_segment(p, c, a)] {
            if (p - q).length() < (p - best).length() {
                best = q;
            }
        }
        return best;
    }
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}

/// Ericson 5.1.9: the closest points of two segments.
fn closest_between_segments(p1: D3, q1: D3, p2: D3, q2: D3) -> (D3, D3) {
    let (d1, d2, r) = (q1 - p1, q2 - p2, p1 - p2);
    let (a, e, f) = (d1.dot(d1), d2.dot(d2), d2.dot(r));
    let eps = 1e-18;
    let (s, t) = if a <= eps && e <= eps {
        (0.0, 0.0)
    } else if a <= eps {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = d1.dot(r);
        if e <= eps {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            let mut s = if denom > 0.0 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
            (s, t)
        }
    };
    (p1 + d1 * s, p2 + d2 * t)
}

/// Whether the segment `p q` passes through triangle `t` (not in its plane).
fn segment_crosses_triangle(p: D3, q: D3, t: &[D3; 3]) -> bool {
    let (e1, e2, d) = (t[1] - t[0], t[2] - t[0], q - p);
    let h = d.cross(e2);
    let det = e1.dot(h);
    if det.abs() < 1e-14 {
        return false;
    }
    let s = p - t[0];
    let u = s.dot(h) / det;
    let qv = s.cross(e1);
    let v = d.dot(qv) / det;
    let along = e2.dot(qv) / det;
    (0.0..=1.0).contains(&u) && v >= 0.0 && u + v <= 1.0 && (0.0..=1.0).contains(&along)
}

/// The exact distance between two cores (0 when they intersect) and a closest pair.
fn brute_force(a: &Core, b: &Core) -> (f64, D3, D3) {
    let deep = |x: &Core, y: &Core| {
        x.points
            .iter()
            .any(|&p| y.depth(p).is_some_and(|d| d >= 0.0))
    };
    let crosses = |x: &Core, y: &Core| {
        x.segments.iter().any(|&(p, q)| {
            y.triangles
                .iter()
                .any(|t| segment_crosses_triangle(p, q, t))
        })
    };
    let mut best = (f64::INFINITY, D3::ZERO, D3::ZERO);
    let mut consider = |pa: D3, pb: D3| {
        let d = (pb - pa).length();
        if d < best.0 {
            best = (d, pa, pb);
        }
    };
    for &p in &a.points {
        for &q in &b.points {
            consider(p, q);
        }
        for &(s, e) in &b.segments {
            consider(p, closest_on_segment(p, s, e));
        }
        for t in &b.triangles {
            consider(p, closest_on_triangle(p, t));
        }
    }
    for &q in &b.points {
        for &(s, e) in &a.segments {
            consider(closest_on_segment(q, s, e), q);
        }
        for t in &a.triangles {
            consider(closest_on_triangle(q, t), q);
        }
    }
    for &(p1, q1) in &a.segments {
        for &(p2, q2) in &b.segments {
            let (ca, cb) = closest_between_segments(p1, q1, p2, q2);
            consider(ca, cb);
        }
    }
    if deep(a, b) || deep(b, a) || crosses(a, b) || crosses(b, a) {
        best.0 = 0.0;
    }
    best
}

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    fn vec(&mut self, r: f32) -> Vec3 {
        Vec3::new(self.range(-r, r), self.range(-r, r), self.range(-r, r))
    }
    fn rotation(&mut self) -> Quat {
        loop {
            let q = Quat::new(
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            );
            if q.length() > 0.2 {
                return q.normalize();
            }
        }
    }
}

/// The cuboid's vertex loops (see `ConvexHull::cuboid`).
const CUBOID_LOOPS: [&[u32]; 6] = [
    &[0, 4, 6, 2],
    &[1, 3, 7, 5],
    &[0, 1, 5, 4],
    &[2, 6, 7, 3],
    &[0, 2, 3, 1],
    &[4, 5, 7, 6],
];

/// Hulls to test with (`from_points` lives elsewhere): a sheared box, an octahedron, a
/// tetrahedron, all off-centre.
fn hulls() -> Vec<ConvexHull> {
    let sheared = ConvexHull::cuboid(Vec3::new(0.7, 0.4, 0.5))
        .vertices
        .iter()
        .map(|v| Vec3::new(v.x + 0.5 * v.y, v.y + 0.3 * v.z, v.z - 0.2 * v.x) + Vec3::Y * 0.3)
        .collect();
    let octahedron = vec![
        Vec3::new(0.8, 0.1, 0.0),
        Vec3::new(-0.6, 0.1, 0.0),
        Vec3::new(0.0, 0.9, 0.1),
        Vec3::new(0.0, -0.5, 0.1),
        Vec3::new(0.1, 0.0, 0.7),
        Vec3::new(0.1, 0.0, -0.4),
    ];
    let octahedron_faces: [&[u32]; 8] = [
        &[0, 2, 4],
        &[1, 4, 2],
        &[0, 4, 3],
        &[1, 3, 4],
        &[0, 5, 2],
        &[1, 2, 5],
        &[0, 3, 5],
        &[1, 5, 3],
    ];
    let tetrahedron = vec![
        Vec3::new(-0.5, -0.4, -0.3),
        Vec3::new(0.7, -0.4, -0.2),
        Vec3::new(-0.1, 0.6, -0.4),
        Vec3::new(0.0, 0.0, 0.8),
    ];
    let tetrahedron_faces: [&[u32]; 4] = [&[0, 2, 1], &[0, 1, 3], &[1, 2, 3], &[2, 0, 3]];
    vec![
        ConvexHull::from_faces(sheared, &CUBOID_LOOPS).unwrap(),
        ConvexHull::from_faces(octahedron, &octahedron_faces).unwrap(),
        ConvexHull::from_faces(tetrahedron, &tetrahedron_faces).unwrap(),
    ]
}

const KINDS: [&str; 6] = ["sphere", "capsule", "box", "hull", "triangle", "segment"];

fn random_shape<'h>(kind: usize, rng: &mut Rng, hulls: &'h [ConvexHull]) -> ConvexShape<'h> {
    match kind {
        0 => ConvexShape::Sphere(rng.range(0.1, 0.6)),
        1 => ConvexShape::Capsule {
            half_height: rng.range(0.0, 0.8),
            radius: rng.range(0.05, 0.4),
        },
        2 => ConvexShape::Box(Vec3::new(
            rng.range(0.05, 1.0),
            rng.range(0.05, 1.0),
            rng.range(0.05, 1.0),
        )),
        3 => ConvexShape::Hull(&hulls[(rng.unit() * hulls.len() as f32) as usize % hulls.len()]),
        4 => ConvexShape::Triangle([rng.vec(1.0), rng.vec(1.0), rng.vec(1.0)]),
        _ => ConvexShape::Segment {
            a: rng.vec(1.0),
            b: rng.vec(1.0),
            radius: rng.range(0.0, 0.3),
        },
    }
}

/// Checks one GJK answer against brute force; returns whether the cores were apart.
fn check_pair(a: &Convex, b: &Convex, label: &str) -> bool {
    let r = distance(a, b);
    let (ca, cb) = (Core::of(a), Core::of(b));
    let (want, ref_a, ref_b) = brute_force(&ca, &cb);
    let (ra, rb) = (a.radius(), b.radius());
    assert!(r.iterations <= MAX_GJK_ITERATIONS, "{label}: {r:?}");
    assert!(
        r.distance.is_finite() && r.point_a.is_finite() && r.point_b.is_finite(),
        "{label}: {r:?}"
    );
    if r.overlap {
        // Overlap means the cores meet (or are within rounding of meeting).
        assert!(
            want < 1e-4,
            "{label}: GJK overlaps, brute force {want}: {r:?}"
        );
        assert_eq!((r.distance, r.normal), (0.0, Vec3::ZERO), "{label}");
        let p = D3::of(r.point_a);
        assert!(
            ca.distance_to(p) < 1e-4 && cb.distance_to(p) < 1e-4,
            "{label}: {r:?}"
        );
        return false;
    }
    let core = (r.distance + ra + rb) as f64;
    assert!(
        (core - want).abs() < 2e-5,
        "{label}: GJK {core}, brute force {want}: {r:?}"
    );
    assert!((r.normal.length() - 1.0).abs() < 1e-5, "{label}: {r:?}");
    // The core points under the reported surface points lie on the cores and span the distance.
    let (pa, pb) = (r.point_a - r.normal * ra, r.point_b + r.normal * rb);
    assert!(ca.distance_to(D3::of(pa)) < 2e-5, "{label}: {r:?}");
    assert!(cb.distance_to(D3::of(pb)) < 2e-5, "{label}: {r:?}");
    assert!(
        ((D3::of(pb) - D3::of(pa)).length() - want).abs() < 2e-5,
        "{label}"
    );
    // The direction between the closest points of two convex sets is unique.
    if want > 1e-3 {
        let n = (ref_b - ref_a) * (1.0 / want);
        let err = (D3::of(r.normal) - n).length();
        assert!(
            err < 1e-3 + 1e-5 / want,
            "{label}: normal {:?} vs {n:?}",
            r.normal
        );
    }
    true
}

#[test]
fn every_pair_of_convex_kinds_matches_brute_force() {
    let hulls = hulls();
    let mut rng = Rng(0x5eed);
    let (mut apart, mut overlapping) = (0, 0);
    for (ka, kind_a) in KINDS.iter().enumerate() {
        for (kb, kind_b) in KINDS.iter().enumerate() {
            for case in 0..60 {
                let sa = random_shape(ka, &mut rng, &hulls);
                let sb = random_shape(kb, &mut rng, &hulls);
                let a = Convex::new(sa, Iso::new(rng.vec(0.5), rng.rotation()));
                // Alternately close (mostly overlapping) and spread out (mostly apart).
                let spread = if case % 2 == 0 { 0.6 } else { 2.0 };
                let b = Convex::new(sb, Iso::new(rng.vec(spread), rng.rotation()));
                let label = format!("{} vs {} #{case}", kind_a, kind_b);
                if check_pair(&a, &b, &label) {
                    apart += 1;
                } else {
                    overlapping += 1;
                }
            }
        }
    }
    assert!(
        apart > 1500 && overlapping > 250,
        "{apart} apart, {overlapping} overlapping"
    );
}

#[test]
fn nearly_touching_pairs_match_brute_force() {
    let hulls = hulls();
    let mut rng = Rng(0x7007);
    for (ka, kind_a) in KINDS.iter().enumerate() {
        for (kb, kind_b) in KINDS.iter().enumerate() {
            for case in 0..15 {
                let sa = random_shape(ka, &mut rng, &hulls);
                let sb = random_shape(kb, &mut rng, &hulls);
                let a = Convex::new(sa, Iso::new(rng.vec(0.5), rng.rotation()));
                // Slide B along a random line until the cores are 1e-3 apart.
                let dir = rng.vec(1.0).normalize_or_zero();
                let rot = rng.rotation();
                let place = |s: f32| Convex::new(sb, Iso::new(a.iso.position + dir * s, rot));
                let gap = |s: f32| brute_force(&Core::of(&a), &Core::of(&place(s))).0;
                let (mut lo, mut hi) = (0.0f32, 6.0f32);
                if gap(hi) < 1e-3 {
                    continue;
                }
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    if gap(mid) < 1e-3 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let label = format!("{} touching {} #{case}", kind_a, kind_b);
                assert!(check_pair(&a, &place(hi), &label), "{label}");
            }
        }
    }
}

#[test]
fn swapped_pairs_agree() {
    let hulls = hulls();
    let mut rng = Rng(0xab);
    for (ka, kind_a) in KINDS.iter().enumerate() {
        for (kb, kind_b) in KINDS.iter().enumerate() {
            for case in 0..20 {
                let a = Convex::new(
                    random_shape(ka, &mut rng, &hulls),
                    Iso::new(rng.vec(0.5), rng.rotation()),
                );
                let b = Convex::new(
                    random_shape(kb, &mut rng, &hulls),
                    Iso::new(rng.vec(2.5), rng.rotation()),
                );
                let label = format!("{kind_a} and {kind_b} #{case}");
                let (ab, ba) = (distance(&a, &b), distance(&b, &a));
                assert_eq!(ab.overlap, ba.overlap, "{label}: {ab:?} {ba:?}");
                if ab.overlap {
                    continue;
                }
                assert!(
                    (ab.distance - ba.distance).abs() < 2e-5,
                    "{label}: {ab:?} {ba:?}"
                );
                if ab.distance > 1e-2 {
                    assert!(
                        (ab.normal + ba.normal).length() < 1e-3,
                        "{label}: {ab:?} {ba:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn pairs_far_from_the_origin_match_brute_force() {
    let hulls = hulls();
    let mut rng = Rng(0xfa7);
    let far = Vec3::new(9_000.0, -6_000.0, 11_000.0);
    for (ka, kind_a) in KINDS.iter().enumerate() {
        for (kb, kind_b) in KINDS.iter().enumerate() {
            for case in 0..10 {
                let a = Convex::new(
                    random_shape(ka, &mut rng, &hulls),
                    Iso::new(far + rng.vec(0.5), rng.rotation()),
                );
                let b = Convex::new(
                    random_shape(kb, &mut rng, &hulls),
                    Iso::new(far + rng.vec(2.5), rng.rotation()),
                );
                // The same 2e-5 m as near the origin: the cores are measured relative to each
                // other, not to the world origin.
                let label = format!("{kind_a} and {kind_b} far away #{case}");
                let r = distance(&a, &b);
                let (want, _, _) = brute_force(&Core::of(&a), &Core::of(&b));
                if r.overlap {
                    assert!(want < 1e-4, "{label}: {r:?} vs {want}");
                } else {
                    let core = (r.distance + a.radius() + b.radius()) as f64;
                    assert!((core - want).abs() < 2e-5, "{label}: {core} vs {want}");
                }
            }
        }
    }
}

/// The first fraction at which `a` moved along `translation` comes within `target` of `b`, by
/// sampling the distance along the way and bisecting.
fn sampled_time_of_impact(a: &Convex, translation: Vec3, b: &Convex, target: f32) -> Option<f32> {
    let at = |f: f32| {
        let moved = Convex::new(
            a.shape,
            Iso::new(a.iso.position + translation * f, a.iso.rotation),
        );
        let r = distance(&moved, b);
        if r.overlap {
            f32::NEG_INFINITY
        } else {
            r.distance
        }
    };
    let n = 400;
    let first = (0..=n).find(|&i| at(i as f32 / n as f32) <= target)?;
    if first == 0 {
        return Some(0.0);
    }
    let (mut lo, mut hi) = ((first - 1) as f32 / n as f32, first as f32 / n as f32);
    for _ in 0..30 {
        let mid = 0.5 * (lo + hi);
        if at(mid) <= target {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(hi)
}

#[test]
fn casts_never_pass_the_sampled_time_of_impact() {
    let hulls = hulls();
    let mut rng = Rng(0xca57);
    let (mut hits, mut misses) = (0, 0);
    for (ka, kind_a) in KINDS.iter().enumerate() {
        for (kb, kind_b) in KINDS.iter().enumerate() {
            for case in 0..6 {
                let a = Convex::new(
                    random_shape(ka, &mut rng, &hulls),
                    Iso::new(Vec3::new(-4.0, 0.0, 0.0) + rng.vec(1.0), rng.rotation()),
                );
                let b = Convex::new(
                    random_shape(kb, &mut rng, &hulls),
                    Iso::new(rng.vec(0.5), rng.rotation()),
                );
                let translation = Vec3::new(8.0, 0.0, 0.0) + rng.vec(1.5);
                let target = if case % 2 == 0 { 0.0 } else { 0.1 };
                let label = format!("{} cast at {} #{case}", kind_a, kind_b);
                let start = distance(&a, &b);
                if start.overlap || start.distance <= target + 1e-3 {
                    continue;
                }
                let hit = cast(&a, translation, &b, target);
                match (hit, sampled_time_of_impact(&a, translation, &b, target)) {
                    (Some(hit), Some(toi)) => {
                        hits += 1;
                        // Never past the impact, and short of it by at most the tolerance band
                        // (0.1 mm) over the closing speed.
                        assert!(hit.fraction <= toi + 1e-5, "{label}: {hit:?} vs {toi}");
                        assert!(hit.fraction >= toi - 2e-4, "{label}: {hit:?} vs {toi}");
                        assert!(!hit.started_overlapping, "{label}");
                        assert!((hit.normal.length() - 1.0).abs() < 1e-5, "{label}");
                        // At the reported fraction the shapes are just outside the target.
                        let moved = Convex::new(
                            a.shape,
                            Iso::new(a.iso.position + translation * hit.fraction, a.iso.rotation),
                        );
                        let there = distance(&moved, &b);
                        assert!(!there.overlap, "{label}");
                        assert!(there.distance >= target - 1e-5, "{label}: {there:?}");
                        assert!(there.distance <= target + 2e-4, "{label}: {there:?}");
                        assert!((there.point_b - hit.point).length() < 1e-4, "{label}");
                        // B's normal points towards A.
                        assert!((there.normal + hit.normal).length() < 1e-4, "{label}");
                    }
                    (None, None) => misses += 1,
                    (None, Some(toi)) => {
                        // Only a graze within the tolerance band may be missed.
                        let moved = Convex::new(
                            a.shape,
                            Iso::new(a.iso.position + translation * toi, a.iso.rotation),
                        );
                        let there = distance(&moved, &b);
                        assert!(there.distance > target - 2e-4, "{label}: missed {toi}");
                    }
                    (Some(hit), None) => {
                        // Only a graze within the tolerance band may be reported.
                        let moved = Convex::new(
                            a.shape,
                            Iso::new(a.iso.position + translation * hit.fraction, a.iso.rotation),
                        );
                        let there = distance(&moved, &b);
                        assert!(
                            there.distance <= target + 2e-4,
                            "{label}: {hit:?} {there:?}"
                        );
                    }
                }
            }
        }
    }
    assert!(hits > 40 && misses > 10, "{hits} hits, {misses} misses");
}

/// Quarter turns about the axes: faces end up exactly parallel or perpendicular.
fn snapped_rotation(rng: &mut Rng) -> Quat {
    let k = (rng.unit() * 24.0) as u32;
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let quarter = std::f32::consts::FRAC_PI_2;
    let a = Quat::from_axis_angle(axes[(k % 3) as usize], quarter * (k / 3 % 4) as f32);
    let b = Quat::from_axis_angle(axes[(k / 12 % 3) as usize], quarter * (k % 2) as f32);
    (a * b).normalize()
}

/// Cores that break naive solvers: collinear and repeated triangle corners, a zero-length
/// segment, 0.2 mm plates and rods, a capsule with no height; or an ordinary random kind.
fn degenerate_shape<'h>(kind: usize, rng: &mut Rng, hulls: &'h [ConvexHull]) -> ConvexShape<'h> {
    match kind {
        0 => {
            let (a, b) = (rng.vec(1.0), rng.vec(1.0));
            ConvexShape::Triangle([a, b, a + (b - a) * rng.range(-0.5, 1.5)])
        }
        1 => {
            let a = rng.vec(1.0);
            ConvexShape::Triangle([a, a, rng.vec(1.0)])
        }
        2 => ConvexShape::Segment {
            a: Vec3::new(0.1, 0.2, 0.3),
            b: Vec3::new(0.1, 0.2, 0.3),
            radius: 0.0,
        },
        3 => ConvexShape::Box(Vec3::new(rng.range(0.1, 1.0), 1e-4, rng.range(0.1, 1.0))),
        4 => ConvexShape::Box(Vec3::new(1e-4, 1e-4, rng.range(0.1, 1.0))),
        5 => ConvexShape::Capsule {
            half_height: 0.0,
            radius: rng.range(0.1, 0.5),
        },
        _ => random_shape((rng.unit() * 6.0) as usize % 6, rng, hulls),
    }
}

#[test]
fn degenerate_and_axis_aligned_cores_match_brute_force() {
    let hulls = hulls();
    let mut rng = Rng(0x57e55);
    for case in 0..1500 {
        let ka = (rng.unit() * 8.0) as usize % 8;
        let kb = (rng.unit() * 8.0) as usize % 8;
        let (sa, sb) = (
            degenerate_shape(ka, &mut rng, &hulls),
            degenerate_shape(kb, &mut rng, &hulls),
        );
        // Two in three are snapped to quarter turns and a quarter-metre grid: coplanar faces,
        // parallel edges and exact touching.
        let snap = case % 3 != 0;
        let mut place = |r: f32| {
            let v = rng.vec(r);
            let rotation = if snap {
                snapped_rotation(&mut rng)
            } else {
                rng.rotation()
            };
            let grid = |x: f32| if snap { (x * 4.0).round() / 4.0 } else { x };
            Iso::new(Vec3::new(grid(v.x), grid(v.y), grid(v.z)), rotation)
        };
        let a = Convex::new(sa, place(0.5));
        let b = Convex::new(sb, place(2.0));
        let r = distance(&a, &b);
        assert!(r.point_a.is_finite() && r.point_b.is_finite() && r.normal.is_finite());
        let (want, _, _) = brute_force(&Core::of(&a), &Core::of(&b));
        if r.overlap {
            assert!(
                want < 1e-4,
                "#{case}: overlap, brute force {want}: {a:?} {b:?}"
            );
        } else {
            let core = (r.distance + a.radius() + b.radius()) as f64;
            assert!(
                (core - want).abs() < 1e-5,
                "#{case}: {core} vs {want}: {a:?} {b:?}"
            );
            assert!((r.normal.length() - 1.0).abs() < 1e-5);
        }
    }
}

#[test]
fn sizes_from_a_tenth_of_a_millimetre_to_a_hundred_kilometres_keep_their_relative_accuracy() {
    let mut rng = Rng(0x5ca1e);
    for exp in -4i32..=5 {
        let s = 10f32.powi(exp);
        let mut worst = 0.0f64;
        for case in 0..150 {
            let shape = |rng: &mut Rng, k: usize| match k % 3 {
                0 => ConvexShape::Box(
                    Vec3::new(
                        rng.range(0.1, 1.0),
                        rng.range(0.1, 1.0),
                        rng.range(0.1, 1.0),
                    ) * s,
                ),
                1 => ConvexShape::Capsule {
                    half_height: rng.range(0.0, 1.0) * s,
                    radius: rng.range(0.1, 0.5) * s,
                },
                _ => ConvexShape::Triangle([rng.vec(s), rng.vec(s), rng.vec(s)]),
            };
            let (sa, sb) = (shape(&mut rng, case), shape(&mut rng, case / 3));
            let a = Convex::new(sa, Iso::new(rng.vec(0.5 * s), rng.rotation()));
            let b = Convex::new(sb, Iso::new(rng.vec(3.0 * s), rng.rotation()));
            let r = distance(&a, &b);
            let (want, _, _) = brute_force(&Core::of(&a), &Core::of(&b));
            if r.overlap {
                assert!(want < 1e-5 * s as f64, "1e{exp} #{case}: overlap vs {want}");
            } else {
                let core = (r.distance + a.radius() + b.radius()) as f64;
                worst = worst.max((core - want).abs() / s as f64);
            }
        }
        assert!(
            worst < 3e-6,
            "size 1e{exp}: worst error {worst:e} of the size"
        );
    }
}

#[test]
fn nearly_parallel_edges_keep_the_gap_and_the_normal() {
    // Two long boxes on edge whose ridges cross at a tiny angle, a small gap apart, the whole
    // pair turned at random: the Minkowski difference has needle-thin faces there.
    let mut rng = Rng(0x9eed);
    let ridge = 0.05 * std::f32::consts::SQRT_2;
    let turn = Quat::from_axis_angle(Vec3::X, std::f32::consts::FRAC_PI_4);
    // Normal tolerance against the f32-placed corners, by the crossing angle: below 1e-4 rad
    // the rounding of the corners themselves turns the common perpendicular by this much.
    for (angle_exp, normal_tol) in [(-6.0f32, 6e-2), (-4.0, 1e-2), (-3.0, 4e-3), (-2.0, 1e-4)] {
        for case in 0..150 {
            let angle = 10f32.powf(angle_exp) * rng.range(0.5, 2.0);
            // Mostly gaps under a millimetre, where the faces are thinnest next to the gap.
            let gap = 10f32.powf(rng.range(-4.0, -1.5));
            let whole = rng.rotation();
            let half = Vec3::new(rng.range(0.5, 2.0), 0.05, 0.05);
            let pa = rng.vec(0.3);
            let pb = pa + whole.rotate(Vec3::new(rng.range(-0.3, 0.3), 2.0 * ridge + gap, 0.0));
            let a = Convex::new(ConvexShape::Box(half), Iso::new(pa, whole * turn));
            let b = Convex::new(
                ConvexShape::Box(half),
                Iso::new(pb, whole * Quat::from_axis_angle(Vec3::Y, angle) * turn),
            );
            let r = distance(&a, &b);
            let (want, ra, rb) = brute_force(&Core::of_rounded(&a), &Core::of_rounded(&b));
            let label = format!("angle {angle:e}, gap {gap:e}, #{case}");
            assert!(!r.overlap, "{label}");
            assert!(
                (r.distance as f64 - want).abs() < 1e-6,
                "{label}: {r:?} vs {want}"
            );
            let n = (rb - ra) * (1.0 / want);
            let err = (D3::of(r.normal) - n).length();
            assert!(err < normal_tol, "{label}: normal {:?} vs {n:?}", r.normal);
        }
    }
}

#[test]
fn a_fast_small_ball_never_tunnels_through_a_thin_plate() {
    let mut rng = Rng(0xb011);
    for case in 0..200 {
        let plate = Convex::new(
            ConvexShape::Box(Vec3::new(1.0, 1e-4, 1.0)),
            Iso::new(rng.vec(0.1), rng.rotation()),
        );
        let up = plate.iso.vector(Vec3::Y);
        let across = plate
            .iso
            .vector(Vec3::new(rng.range(-0.8, 0.8), 0.0, rng.range(-0.8, 0.8)));
        let start = plate.iso.position + across + up * rng.range(1.0, 50.0);
        let travel = -up * rng.range(60.0, 200.0) + rng.vec(0.2);
        let ball = Convex::new(
            ConvexShape::Sphere(rng.range(1e-3, 1e-2)),
            Iso::from_position(start),
        );
        let hit = cast(&ball, travel, &plate, 0.0).unwrap_or_else(|| panic!("#{case} tunnelled"));
        let there = distance(
            &Convex::new(
                ball.shape,
                Iso::from_position(start + travel * hit.fraction),
            ),
            &plate,
        );
        assert!(
            !there.overlap && there.distance >= -1e-5,
            "#{case}: {there:?}"
        );
        assert!(there.distance < 2e-4, "#{case}: stopped short at {there:?}");
        assert!((hit.normal - up).length() < 1e-3, "#{case}: {hit:?}");
    }
}

#[test]
fn casts_at_every_size_never_pass_the_sampled_time_of_impact() {
    let mut rng = Rng(0xca5e);
    for exp in [-3i32, 0, 3] {
        let s = 10f32.powi(exp);
        let mut hits = 0;
        for case in 0..60 {
            let mut shape = |k: usize| match k % 4 {
                0 => ConvexShape::Sphere(rng.range(0.1, 0.6) * s),
                1 => ConvexShape::Capsule {
                    half_height: rng.range(0.0, 0.8) * s,
                    radius: rng.range(0.05, 0.4) * s,
                },
                2 => ConvexShape::Box(
                    Vec3::new(
                        rng.range(0.05, 1.0),
                        rng.range(0.05, 1.0),
                        rng.range(0.05, 1.0),
                    ) * s,
                ),
                _ => ConvexShape::Triangle([rng.vec(s), rng.vec(s), rng.vec(s)]),
            };
            let (sa, sb) = (shape(case), shape(case / 4));
            let a = Convex::new(
                sa,
                Iso::new(
                    (Vec3::new(-4.0, 0.0, 0.0) + rng.vec(1.0)) * s,
                    rng.rotation(),
                ),
            );
            let b = Convex::new(sb, Iso::new(rng.vec(0.6) * s, rng.rotation()));
            let translation = (Vec3::new(8.0, 0.0, 0.0) + rng.vec(1.2)) * s;
            let target = [0.0, 0.01 * s, 0.1 * s][case % 3];
            let start = distance(&a, &b);
            if start.overlap || start.distance <= target + 1e-3 * s {
                continue;
            }
            let label = format!("size 1e{exp} #{case}");
            if let Some(hit) = cast(&a, translation, &b, target) {
                hits += 1;
                let toi = sampled_time_of_impact(&a, translation, &b, target);
                if let Some(toi) = toi {
                    assert!(hit.fraction <= toi + 1e-5, "{label}: {hit:?} vs {toi}");
                }
                let moved = Convex::new(
                    a.shape,
                    Iso::new(a.iso.position + translation * hit.fraction, a.iso.rotation),
                );
                let there = distance(&moved, &b);
                assert!(
                    !there.overlap && there.distance >= target - 1e-5 * s,
                    "{label}"
                );
                assert!(
                    there.distance <= target + 2e-4 * s.max(1.0),
                    "{label}: {there:?}"
                );
            } else if let Some(toi) = sampled_time_of_impact(&a, translation, &b, target) {
                let moved = Convex::new(
                    a.shape,
                    Iso::new(a.iso.position + translation * toi, a.iso.rotation),
                );
                let there = distance(&moved, &b);
                assert!(
                    there.distance > target - 2e-4 * s.max(1.0),
                    "{label}: missed {toi}"
                );
            }
        }
        assert!(hits > 15, "size 1e{exp}: {hits} hits");
    }
}

#[test]
fn a_point_inside_a_thin_plate_overlaps_it() {
    // The tetrahedra around such a point are thin (0.2 mm against a metre) but real: their
    // barycentric signs must be trusted, or the point reads as just outside.
    let mut rng = Rng(0x91a7e);
    for case in 0..300 {
        let half = Vec3::new(rng.range(0.1, 1.0), 1e-4, rng.range(0.1, 1.0));
        let rotation = if case % 2 == 0 {
            snapped_rotation(&mut rng)
        } else {
            rng.rotation()
        };
        let plate = Convex::new(ConvexShape::Box(half), Iso::new(rng.vec(0.5), rotation));
        let local = Vec3::new(
            rng.range(-0.9, 0.9) * half.x,
            rng.range(-0.5, 0.5) * half.y,
            rng.range(-0.9, 0.9) * half.z,
        );
        let ball = Convex::new(
            ConvexShape::Sphere(rng.range(0.05, 0.3)),
            Iso::from_position(plate.iso.point(local)),
        );
        for r in [distance(&ball, &plate), distance(&plate, &ball)] {
            assert!(r.overlap, "#{case}: {r:?}");
        }
    }
}
