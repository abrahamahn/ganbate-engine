//! Distance between convex shapes (GJK) and casting one convex against another (conservative
//! advancement on that distance).
//!
//! A [`Convex`] is a core (point, segment, box, hull or triangle) plus a rounding radius; these
//! routines work on the cores and account for the radii, so spheres and capsules are exact.
//!
//! [`distance`] runs GJK on the Minkowski difference `A − B` of the two cores, looking for its
//! point closest to the origin. The simplex (a point, segment, triangle or tetrahedron of support
//! points) is reduced exactly to the sub-simplex that holds its closest point, by the signs of the
//! barycentric coordinates of the origin's projection (its Voronoi regions); a triangle or
//! tetrahedron too flat to have an inside is searched by its edges or faces instead. The search
//! stops when the lower bound on the distance that the newest support point gives comes within a
//! tolerance of the upper bound `|v|` (relative to the distance, with a floor at the rounding of
//! the bound itself), when a support point repeats one of the simplex, when the simplex stops
//! getting closer, or after [`MAX_GJK_ITERATIONS`].
//!
//! Precision: support points are taken relative to A's centre instead of the world origin (the
//! same world axes, shifted), so two bodies 10 km out keep a centimetre gap exact to a few
//! micrometres; only the returned points carry world-sized rounding. The simplex is solved in
//! `f64`, where its vertices (the `f32` support points) are exact: beside a kilometre-wide face
//! the closest point can be under a metre from the origin while its weight on a far corner is
//! about 1e-8, which `f32` loses next to 1 (the search then stalled centimetres short). Near a
//! needle-thin face of the Minkowski difference (nearly parallel edges close together) a
//! segment's closest point is taken across the segment rather than along it, a triangle's plane
//! comes from its longest edge and the apex's height over that edge, and a step that comes out
//! level within rounding is kept: together they hold the normal to what the rounding of the
//! input itself allows. Cores closer than about a millionth of the problem's size (the largest
//! support point's distance from A's centre) count as overlapping: `f32` cannot tell the
//! direction between them.
//!
//! [`cast`] moves A by the distance left over the closing speed along the separating normal, a
//! step that can never pass the time of impact (the normal is a separating axis, and A cannot
//! close the gap along it faster than that), until the shapes are within `target` plus a small
//! tolerance. Within that band a hit needs the shapes still closing, and a probe one full step
//! ahead (which cannot pass the impact either) that confirms the approach: one that does not is
//! a graze, and the cast goes on from the probe.
//!
//! Only `+ − × ÷` and `sqrt` on `f32` and `f64` (and conversions between them, which round to
//! nearest), fixed-size arrays and no allocation: deterministic.

use crate::math::{Iso, Vec3};
use crate::shape::{Convex, ConvexShape};

/// Most GJK iterations before giving up with the best answer so far.
pub const MAX_GJK_ITERATIONS: u32 = 64;

/// GJK's accuracy relative to the size of the problem (about eight `f32` ulps).
const RELATIVE_TOLERANCE: f32 = 1e-6;
/// The rounding of the support points relative to the problem's size (a few `f32` ulps): the
/// convergence test's floor. Near a thin face of the Minkowski difference a looser test would
/// stop with the distance right but the direction off.
const ROUNDING: f32 = 2e-7;
/// Cores closer than this (metres) overlap whatever their size.
const ABSOLUTE_TOLERANCE: f32 = 1e-10;
/// A triangle whose apex is within `√this` (1e-6) of its longest edge's line, relative to the
/// edge from there to the apex, is flat: its plane is lost in the rounding of its corners.
const FLAT_TRIANGLE: f64 = 1e-12;
/// A tetrahedron whose `e1 · (e2 × e3)` is at most this times `|e1| |e2| |e3|` is flat: the signs
/// of its barycentric coordinates are lost in the rounding of its corners. (Thin but real ones,
/// such as the difference of a 0.2 mm plate and a 1 m box, are well above it.)
const FLAT_TETRAHEDRON: f64 = 1e-6;

/// Most conservative-advancement steps of one cast.
const MAX_CAST_ITERATIONS: u32 = 32;
/// The most a cast may stop short of `target` (metres); smaller shapes get a smaller tolerance.
const CAST_TOLERANCE: f32 = 1e-4;
/// The least tolerance of a cast (metres), so that point-like cores still converge.
const MIN_CAST_TOLERANCE: f32 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceResult {
    /// Distance between the two surfaces (core distance minus both radii); negative down to
    /// minus the radii sum when only the rounding overlaps; 0 when the cores themselves overlap.
    pub distance: f32,
    /// Closest point on A's surface (world). When the cores overlap: a point inside both.
    pub point_a: Vec3,
    /// Closest point on B's surface (world).
    pub point_b: Vec3,
    /// Unit direction from A towards B (world); zero when the cores overlap.
    pub normal: Vec3,
    /// The cores intersect (`distance`, `normal` are then not meaningful).
    pub overlap: bool,
    pub iterations: u32,
}

/// Minimal interface over a local-frame core + rounding radius so stack-backed hull vertices can
/// run through GJK and conservative advancement with zero heap allocation.
trait CoreShape: Copy {
    fn radius(self) -> f32;
    fn center(self) -> Vec3;
    fn support_core(self, d: Vec3) -> Vec3;
    fn reach(self) -> f32;
}

impl CoreShape for &ConvexShape<'_> {
    #[inline]
    fn radius(self) -> f32 {
        ConvexShape::radius(self)
    }
    #[inline]
    fn center(self) -> Vec3 {
        ConvexShape::center(self)
    }
    #[inline]
    fn support_core(self, d: Vec3) -> Vec3 {
        ConvexShape::support_core(self, d)
    }
    #[inline]
    fn reach(self) -> f32 {
        reach(self)
    }
}

#[derive(Clone, Copy)]
struct HullVerts<'a> {
    vertices: &'a [Vec3],
    centroid: Vec3,
}

impl CoreShape for HullVerts<'_> {
    #[inline]
    fn radius(self) -> f32 {
        0.0
    }
    #[inline]
    fn center(self) -> Vec3 {
        self.centroid
    }
    #[inline]
    fn support_core(self, d: Vec3) -> Vec3 {
        let mut best = 0;
        let mut best_d = f32::NEG_INFINITY;
        for (i, v) in self.vertices.iter().enumerate() {
            let p = v.dot(d);
            if p > best_d {
                best_d = p;
                best = i;
            }
        }
        self.vertices[best]
    }
    #[inline]
    fn reach(self) -> f32 {
        let c = self.centroid;
        self.vertices
            .iter()
            .fold(0.0f32, |m, p| m.max((*p - c).length_squared()))
            .sqrt()
    }
}

/// The distance and closest points between two convex shapes.
pub fn distance(a: &Convex, b: &Convex) -> DistanceResult {
    distance_impl(&a.shape, &a.iso, b)
}

fn distance_impl<A: CoreShape>(a_shape: A, a_iso: &Iso, b: &Convex) -> DistanceResult {
    // Work relative to A's centre: the cores' coordinates stay as small as the shapes and the gap
    // between them, wherever they are in the world.
    let origin = a_iso.point(a_shape.center());
    let core = closest_cores(a_shape, a_iso, b, origin);
    if core.overlap {
        let p = (core.a + core.b) * 0.5 + origin;
        return DistanceResult {
            distance: 0.0,
            point_a: p,
            point_b: p,
            normal: Vec3::ZERO,
            overlap: true,
            iterations: core.iterations,
        };
    }
    // Not overlapping means |v| is above the tolerance, so the normal is a unit vector.
    let (normal, length) = (-core.v).normalize_and_length();
    let (ra, rb) = (a_shape.radius(), b.radius());
    DistanceResult {
        distance: length - ra - rb,
        point_a: (core.a + normal * ra) + origin,
        point_b: (core.b - normal * rb) + origin,
        normal,
        overlap: false,
        iterations: core.iterations,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CastHit {
    /// Fraction of `translation` travelled when the shapes come within `target` of each other,
    /// in `[0, 1]`.
    pub fraction: f32,
    /// The contact point on B's surface at that moment (world).
    pub point: Vec3,
    /// B's surface normal there (unit, world), pointing towards A.
    pub normal: Vec3,
    /// A already overlapped B (closer than `target`) at the start; `fraction` is 0.
    pub started_overlapping: bool,
}

/// Moves `a` along `translation` (B stays still) and reports when they first come within
/// `target` (≥ 0) of each other, if they do within the translation. Conservative advancement:
/// never reports a fraction past the true one.
///
/// A hit is reported once the shapes are within `target` plus a small tolerance (at most 0.1 mm,
/// less for shapes under a centimetre, more where `f32` cannot resolve that at the shapes' size)
/// and still closing, so the fraction may fall short of the exact one by that tolerance over the
/// closing speed. Within that band but moving apart or along the surface, A never comes closer:
/// no hit (a shape left in the band by an earlier cast can be cast on, away or sideways); and a
/// path that closes in only to graze past (onto a coplanar neighbour, say) goes on past it.
pub fn cast(a: &Convex, translation: Vec3, b: &Convex, target: f32) -> Option<CastHit> {
    cast_impl(&a.shape, a.iso, translation, b, target)
}

/// Casts a convex hull represented directly by its local `vertices` and `centroid` without
/// allocating a `ConvexHull` on the heap.
pub(crate) fn cast_hull_vertices(
    vertices: &[Vec3],
    centroid: Vec3,
    iso: Iso,
    translation: Vec3,
    b: &Convex,
    target: f32,
) -> Option<CastHit> {
    cast_impl(
        HullVerts { vertices, centroid },
        iso,
        translation,
        b,
        target,
    )
}

fn cast_impl<A: CoreShape>(
    a_shape: A,
    a_iso: Iso,
    translation: Vec3,
    b: &Convex,
    target: f32,
) -> Option<CastHit> {
    let target = if target > 0.0 { target } else { 0.0 };
    let start = distance_impl(a_shape, &a_iso, b);
    if start.overlap || start.distance <= target {
        let normal = if start.overlap {
            // No separating direction: report the one A came from.
            (-translation)
                .try_normalize()
                .or_else(|| (a_iso.point(a_shape.center()) - b.center()).try_normalize())
                .unwrap_or(Vec3::Y)
        } else {
            -start.normal
        };
        return Some(CastHit {
            fraction: 0.0,
            point: start.point_b,
            normal,
            started_overlapping: true,
        });
    }
    let tolerance = cast_tolerance_reaches(a_shape.reach(), reach(&b.shape), target);
    let hit = |fraction: f32, r: &DistanceResult| CastHit {
        fraction,
        point: r.point_b,
        normal: -r.normal,
        started_overlapping: false,
    };
    let mut fraction = 0.0f32;
    let mut current = start;
    for _ in 0..MAX_CAST_ITERATIONS {
        if current.distance <= target {
            return Some(hit(fraction, &current));
        }
        // The normal separates the shapes; A closes the gap along it at this rate, never faster.
        // Moving apart or along it, A never comes closer than it is (the distance along a
        // translation is convex), in the tolerance band or not.
        let closing = translation.dot(current.normal);
        if closing <= 0.0 {
            return None;
        }
        let gap = current.distance - target;
        if gap <= tolerance {
            // In the band and closing. The step that would close the whole gap at this rate
            // cannot pass the impact either: a real approach comes most of the way there, while
            // a graze (past an edge, such as the seam onto a coplanar neighbour, or tangent to a
            // curve) does not, and goes on from there.
            let probe = (fraction + gap / closing).min(1.0);
            let moved_iso = Iso::new(a_iso.position + translation * probe, a_iso.rotation);
            let r = distance_impl(a_shape, &moved_iso, b);
            if r.overlap || r.distance <= target + 0.5 * gap {
                return Some(hit(fraction, &current));
            }
            if probe >= 1.0 {
                return None;
            }
            fraction = probe;
            current = r;
            continue;
        }
        if fraction + (gap - tolerance) / closing > 1.0 {
            return None;
        }
        // Aim into the middle of the tolerance band, so an approach face first does not land
        // exactly on contact.
        let next = (fraction + (gap - 0.5 * tolerance) / closing).min(1.0);
        let moved_iso = Iso::new(a_iso.position + translation * next, a_iso.rotation);
        let r = distance_impl(a_shape, &moved_iso, b);
        if r.overlap {
            // Rounding carried A into B: the last separated position is the conservative answer.
            return Some(hit(fraction, &current));
        }
        fraction = next;
        current = r;
    }
    // Out of steps (only on a rounding stalemate): where A got to is still short of the impact.
    Some(hit(fraction, &current))
}

/// The band within which a cast counts as having reached `target`.
#[inline]
fn cast_tolerance_reaches(ra: f32, rb: f32, target: f32) -> f32 {
    CAST_TOLERANCE
        .min(0.01 * ra.min(rb))
        // Stay well above the rounding of `distance` itself (see the module doc).
        .max(4.0 * RELATIVE_TOLERANCE * (ra + rb + target))
        .max(MIN_CAST_TOLERANCE)
}

/// How far the shape reaches from its centre: the core's farthest point plus the radius.
fn reach(shape: &ConvexShape) -> f32 {
    let core = match *shape {
        ConvexShape::Sphere(_) => 0.0,
        ConvexShape::Capsule { half_height, .. } => half_height.abs(),
        ConvexShape::Box(h) => h.length(),
        ConvexShape::Hull(h) => h.reach,
        ConvexShape::Triangle(t) => {
            let c = shape.center();
            t.iter()
                .fold(0.0f32, |m, p| m.max((*p - c).length_squared()))
                .sqrt()
        }
        ConvexShape::Segment { a, b, .. } => (b - a).length() * 0.5,
    };
    core + shape.radius().abs()
}

/// What GJK found, relative to the frame origin it worked in.
#[derive(Clone, Copy, Debug)]
struct CoreDistance {
    /// The closest point of A's core (when overlapping: a point inside both cores).
    a: Vec3,
    /// The closest point of B's core.
    b: Vec3,
    /// The point of `A − B` closest to the origin, `a − b`, computed directly for accuracy.
    v: Vec3,
    overlap: bool,
    iterations: u32,
}

/// A vector in `f64`, what the simplex is solved in (see the module doc).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
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

    fn of(v: Vec3) -> D3 {
        D3 {
            x: v.x as f64,
            y: v.y as f64,
            z: v.z as f64,
        }
    }
    /// The nearest `f32` vector.
    fn to_vec3(self) -> Vec3 {
        Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }
    fn dot(self, o: D3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    fn cross(self, o: D3) -> D3 {
        D3 {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }
    fn length_squared(self) -> f64 {
        self.dot(self)
    }
    fn length(self) -> f64 {
        self.length_squared().sqrt()
    }
}

impl std::ops::Add for D3 {
    type Output = D3;
    fn add(self, o: D3) -> D3 {
        D3 {
            x: self.x + o.x,
            y: self.y + o.y,
            z: self.z + o.z,
        }
    }
}

impl std::ops::Sub for D3 {
    type Output = D3;
    fn sub(self, o: D3) -> D3 {
        D3 {
            x: self.x - o.x,
            y: self.y - o.y,
            z: self.z - o.z,
        }
    }
}

impl std::ops::Neg for D3 {
    type Output = D3;
    fn neg(self) -> D3 {
        D3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl std::ops::Mul<f64> for D3 {
    type Output = D3;
    fn mul(self, s: f64) -> D3 {
        D3 {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

/// One support point of the Minkowski difference and the core points it came from.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SupportPoint {
    a: Vec3,
    b: Vec3,
    /// `a − b`, in `f64`.
    w: D3,
}

/// The farthest point of a core along world direction `d`, relative to a frame origin from which
/// the convex's position is `offset`.
#[inline]
fn support_shape<S: CoreShape>(shape: S, iso: &Iso, offset: Vec3, d: Vec3) -> Vec3 {
    iso.rotation
        .rotate(shape.support_core(iso.inverse_vector(d)))
        + offset
}

/// GJK on the cores of `a` and `b`, in world axes shifted to `origin`.
fn closest_cores<A: CoreShape>(a_shape: A, a_iso: &Iso, b: &Convex, origin: Vec3) -> CoreDistance {
    let (offset_a, offset_b) = (a_iso.position - origin, b.iso.position - origin);
    let support_point = |d: Vec3| {
        let (pa, pb) = (
            support_shape(a_shape, a_iso, offset_a, d),
            support_shape(&b.shape, &b.iso, offset_b, -d),
        );
        SupportPoint {
            a: pa,
            b: pb,
            w: D3::of(pa) - D3::of(pb),
        }
    };
    // Start with A's point farthest towards B and B's farthest towards A.
    let center_a = a_iso.rotation.rotate(a_shape.center()) + offset_a;
    let center_b = b.iso.rotation.rotate(b.shape.center()) + offset_b;
    let toward_b = center_b - center_a;
    let first = support_point(if toward_b.length_squared() > 0.0 {
        toward_b
    } else {
        Vec3::X
    });
    let magnitude = |p: &SupportPoint| p.a.abs().max_element().max(p.b.abs().max_element()) as f64;
    let (relative, rounding) = (RELATIVE_TOLERANCE as f64, ROUNDING as f64);
    let mut scale = magnitude(&first);
    let mut simplex = Simplex::new(first);
    let mut v = first.w;
    let mut iterations = 0;
    // Steps whose answer came out no closer, within rounding, that were kept anyway.
    let mut level_steps = 0;
    let overlap = loop {
        let vv = v.length_squared();
        let touching = relative * scale + ABSOLUTE_TOLERANCE as f64;
        if vv <= touching * touching {
            break true;
        }
        if iterations >= MAX_GJK_ITERATIONS {
            break false;
        }
        iterations += 1;
        let p = support_point((-v).to_vec3());
        scale = scale.max(magnitude(&p));
        if simplex.contains(&p) {
            break false;
        }
        // |v| bounds the distance from above and v·w / |v| from below: stop once they agree to
        // the tolerance, or to the rounding of the input (the support points are `f32`).
        let length = vv.sqrt();
        if vv - v.dot(p.w) <= length * (relative * length + rounding * scale) {
            break false;
        }
        let previous = simplex;
        simplex.push(p);
        let next = simplex.solve();
        if simplex.count == 4 {
            // The origin is inside the tetrahedron.
            v = D3::ZERO;
            break true;
        }
        if next.length_squared() >= vv {
            // No closer. Near a thin face of the Minkowski difference the true gain can be below
            // the rounding of v (a few ulps of the far vertices) while the bigger simplex holds
            // the right direction: keep it, at most twice, if it is no farther than rounding.
            let slack = length + rounding * scale;
            if level_steps >= 2 || next.length_squared() > slack * slack {
                simplex = previous;
                break false;
            }
            level_steps += 1;
        }
        v = next;
    };
    let (pa, pb) = simplex.witnesses();
    CoreDistance {
        a: pa.to_vec3(),
        b: pb.to_vec3(),
        v: v.to_vec3(),
        overlap,
        iterations,
    }
}

/// Up to four support points and the barycentric weights of the simplex's point closest to the
/// origin.
#[derive(Clone, Copy, Debug)]
struct Simplex {
    points: [SupportPoint; 4],
    weights: [f64; 4],
    count: usize,
}

impl Simplex {
    fn new(p: SupportPoint) -> Self {
        Simplex {
            points: [p; 4],
            weights: [1.0, 0.0, 0.0, 0.0],
            count: 1,
        }
    }

    fn contains(&self, p: &SupportPoint) -> bool {
        self.points[..self.count]
            .iter()
            .any(|q| q.a == p.a && q.b == p.b)
    }

    fn push(&mut self, p: SupportPoint) {
        // Never full here: a full simplex holds the origin and ends the search.
        self.points[self.count.min(3)] = p;
        self.count = (self.count + 1).min(4);
    }

    /// Finds the point closest to the origin, keeps only the points it needs (in order) with
    /// their weights, and returns it.
    fn solve(&mut self) -> D3 {
        let w = self.points.map(|p| p.w);
        let closest = match self.count {
            0 | 1 => Closest::vertex(&w, 0),
            2 => closest_on_segment(&w, 0, 1),
            3 => closest_on_triangle(&w, 0, 1, 2),
            _ => closest_on_tetrahedron(&w),
        };
        let mut kept = 0;
        for i in 0..self.count {
            if closest.mask & (1 << i) != 0 {
                self.points[kept] = self.points[i];
                self.weights[kept] = closest.weights[i];
                kept += 1;
            }
        }
        self.count = kept;
        closest.v
    }

    /// The closest points of A's and B's cores: the same weights applied to each side.
    fn witnesses(&self) -> (D3, D3) {
        let mut a = D3::ZERO;
        let mut b = D3::ZERO;
        for (p, &l) in self.points[..self.count].iter().zip(&self.weights) {
            a = a + D3::of(p.a) * l;
            b = b + D3::of(p.b) * l;
        }
        (a, b)
    }
}

/// The point of a sub-simplex closest to the origin: its position, its barycentric weights by
/// slot of the full simplex, and which slots it needs.
#[derive(Clone, Copy, Debug)]
struct Closest {
    v: D3,
    weights: [f64; 4],
    mask: u8,
}

impl Closest {
    fn vertex(w: &[D3; 4], i: usize) -> Closest {
        let mut weights = [0.0; 4];
        weights[i] = 1.0;
        Closest {
            v: w[i],
            weights,
            mask: 1 << i,
        }
    }

    fn closer_than(&self, other: &Closest) -> bool {
        self.v.length_squared() < other.v.length_squared()
    }
}

/// The point of segment `w[i] w[j]` closest to the origin.
fn closest_on_segment(w: &[D3; 4], i: usize, j: usize) -> Closest {
    let (a, b) = (w[i], w[j]);
    let e = b - a;
    // The origin's projection is at a + e·t/|e|²; a zero-length segment ends up at a.
    let t = -a.dot(e);
    if t <= 0.0 {
        return Closest::vertex(w, i);
    }
    let ee = e.dot(e);
    if t >= ee {
        return Closest::vertex(w, j);
    }
    let s = t / ee;
    let mut weights = [0.0; 4];
    weights[i] = 1.0 - s;
    weights[j] = s;
    // The part of an end across the segment, e × (p × e) / |e|², rather than a + e·s: on a long
    // segment close to the origin a + e·s cancels, and its rounding along the segment would tilt
    // the search direction enough to miss the support point that improves on it.
    let p = if a.length_squared() <= b.length_squared() {
        a
    } else {
        b
    };
    Closest {
        v: e.cross(p.cross(e)) * (1.0 / ee),
        weights,
        mask: (1 << i) | (1 << j),
    }
}

/// The point of triangle `w[i] w[j] w[k]` closest to the origin.
fn closest_on_triangle(w: &[D3; 4], i: usize, j: usize, k: usize) -> Closest {
    // Relabel (keeping the winding) so that the longest edge runs from p0 to p1.
    let (ij, jk, ki) = (
        (w[j] - w[i]).length_squared(),
        (w[k] - w[j]).length_squared(),
        (w[i] - w[k]).length_squared(),
    );
    let [i0, i1, i2] = if ij >= jk && ij >= ki {
        [i, j, k]
    } else if jk >= ki {
        [j, k, i]
    } else {
        [k, i, j]
    };
    let (p0, p1, p2) = (w[i0], w[i1], w[i2]);
    let e = p1 - p0;
    let ee = e.dot(e);
    // The apex's height over the longest edge, h, taken from the nearer end. The normal e × h
    // of two perpendicular vectors stays accurate for needle-thin triangles, where the cross
    // product of two nearly parallel edges would tilt the plane.
    let (f, from_p1) = if (p2 - p1).length_squared() < (p2 - p0).length_squared() {
        (p2 - p1, true)
    } else {
        (p2 - p0, false)
    };
    if ee > 0.0 {
        let along = f.dot(e) / ee;
        let h = f - e * along;
        // A second pass removes the rounding left along e, which a far vertex would magnify.
        let h = h - e * (h.dot(e) / ee);
        let hh = h.dot(h);
        if hh > FLAT_TRIANGLE * f.length_squared() {
            // The origin's projection is p0 + α e + β h, and the apex is p0 + γ e + h.
            let alpha = -p0.dot(e) / ee;
            let beta = -p0.dot(h) / hh;
            let gamma = if from_p1 { along + 1.0 } else { along };
            let (w1, w2) = (alpha - beta * gamma, beta);
            let w0 = 1.0 - w1 - w2;
            if w0 > 0.0 && w1 > 0.0 && w2 > 0.0 {
                let n = e.cross(h);
                // The projection itself, along the normal from the vertex nearest the origin.
                let mut near = p0;
                for p in [p1, p2] {
                    if p.length_squared() < near.length_squared() {
                        near = p;
                    }
                }
                let mut weights = [0.0; 4];
                weights[i0] = w0;
                weights[i1] = w1;
                weights[i2] = w2;
                return Closest {
                    v: n * (n.dot(near) / n.dot(n)),
                    weights,
                    mask: (1 << i) | (1 << j) | (1 << k),
                };
            }
        }
    }
    // The projection is outside (or the triangle has no usable plane): the closest point is on
    // the boundary.
    let mut best = closest_on_segment(w, i, j);
    for (p, q) in [(j, k), (i, k)] {
        let c = closest_on_segment(w, p, q);
        if c.closer_than(&best) {
            best = c;
        }
    }
    best
}

/// The point of tetrahedron `w[0..4]` closest to the origin: the origin itself (all four slots)
/// when inside.
fn closest_on_tetrahedron(w: &[D3; 4]) -> Closest {
    /// The face opposite each vertex.
    const FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];
    let a = w[0];
    let (e1, e2, e3) = (w[1] - a, w[2] - a, w[3] - a);
    let (n1, n2, n3) = (e2.cross(e3), e3.cross(e1), e1.cross(e2));
    let det = e1.dot(n1);
    let mut search = [true; 4];
    if det.abs() > FLAT_TETRAHEDRON * e1.length() * e2.length() * e3.length() {
        // Barycentric coordinates of the origin (Cramer's rule on a + Σ lᵢ eᵢ = 0).
        let (l1, l2, l3) = (-a.dot(n1) / det, -a.dot(n2) / det, -a.dot(n3) / det);
        let weights = [1.0 - l1 - l2 - l3, l1, l2, l3];
        if weights.iter().all(|&l| l >= 0.0) {
            return Closest {
                v: D3::ZERO,
                weights,
                mask: 0b1111,
            };
        }
        // The closest point is on a face whose plane has the origin outside.
        search = weights.map(|l| l < 0.0);
    }
    let mut best: Option<Closest> = None;
    for (face, _) in FACES.iter().zip(search).filter(|(_, s)| *s) {
        let c = closest_on_triangle(w, face[0], face[1], face[2]);
        if best.is_none_or(|b| c.closer_than(&b)) {
            best = Some(c);
        }
    }
    best.unwrap_or_else(|| closest_on_triangle(w, 0, 1, 2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hull::ConvexHull;
    use crate::math::Quat;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, SQRT_2};

    fn at(shape: ConvexShape, position: Vec3) -> Convex {
        Convex::new(shape, Iso::from_position(position))
    }

    fn turned(shape: ConvexShape, position: Vec3, axis: Vec3, angle: f32) -> Convex {
        Convex::new(
            shape,
            Iso::new(position, Quat::from_axis_angle(axis, angle)),
        )
    }

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn close_v(a: Vec3, b: Vec3, tol: f32) -> bool {
        (a - b).length() <= tol
    }

    /// Checks the parts of the contract every separated result must meet.
    fn assert_separated(r: &DistanceResult, distance: f32, normal: Vec3, tol: f32) {
        assert!(!r.overlap, "{r:?}");
        assert!(
            close(r.distance, distance, tol),
            "distance {r:?}, want {distance}"
        );
        assert!(
            close_v(r.normal, normal, 1e-3),
            "normal {r:?}, want {normal:?}"
        );
        assert!(close(r.normal.length(), 1.0, 1e-5));
        assert!(
            close_v(r.point_b - r.point_a, r.normal * r.distance, tol),
            "points do not span the distance: {r:?}"
        );
        assert!(r.iterations <= MAX_GJK_ITERATIONS);
    }

    #[test]
    fn separated_spheres_measure_between_centres_less_both_radii() {
        let a = at(ConvexShape::Sphere(0.5), Vec3::new(1.0, 2.0, 3.0));
        let b = at(ConvexShape::Sphere(0.25), Vec3::new(4.0, 6.0, 3.0));
        let r = distance(&a, &b);
        let n = Vec3::new(0.6, 0.8, 0.0);
        assert_separated(&r, 4.25, n, 1e-5);
        assert!(close_v(r.point_a, Vec3::new(1.0, 2.0, 3.0) + n * 0.5, 1e-5));
        assert!(close_v(
            r.point_b,
            Vec3::new(4.0, 6.0, 3.0) - n * 0.25,
            1e-5
        ));
    }

    #[test]
    fn spheres_whose_rounding_overlaps_have_a_negative_distance() {
        let a = at(ConvexShape::Sphere(0.5), Vec3::new(1.0, 2.0, 3.0));
        let b = at(ConvexShape::Sphere(0.25), Vec3::new(1.5, 2.0, 3.0));
        let r = distance(&a, &b);
        assert_separated(&r, -0.25, Vec3::X, 1e-5);
        assert!(close_v(r.point_a, Vec3::new(1.5, 2.0, 3.0), 1e-5));
        assert!(close_v(r.point_b, Vec3::new(1.25, 2.0, 3.0), 1e-5));
    }

    #[test]
    fn spheres_with_one_centre_overlap_at_that_centre() {
        let c = Vec3::new(1.0, 2.0, 3.0);
        let r = distance(
            &at(ConvexShape::Sphere(0.5), c),
            &at(ConvexShape::Sphere(0.25), c),
        );
        assert!(r.overlap);
        assert_eq!((r.distance, r.normal), (0.0, Vec3::ZERO));
        assert!(close_v(r.point_a, c, 1e-6) && close_v(r.point_b, c, 1e-6));
    }

    #[test]
    fn a_sphere_near_a_box_face_edge_or_corner_matches_the_clamped_point() {
        let half = Vec3::new(1.0, 2.0, 0.5);
        let iso = Iso::new(
            Vec3::new(-3.0, 1.0, 2.0),
            Quat::from_axis_angle(Vec3::new(1.0, 2.0, -1.0), 0.8),
        );
        let boxed = Convex::new(ConvexShape::Box(half), iso);
        // Local sphere centres in a face region, two edge regions and a corner region.
        for local in [
            Vec3::new(0.3, 3.0, 0.1),
            Vec3::new(1.5, 2.5, 0.2),
            Vec3::new(-0.2, -2.6, -1.1),
            Vec3::new(1.3, -2.4, 0.9),
        ] {
            let on_box = local.max(-half).min(half);
            let want = (local - on_box).length() - 0.2;
            let sphere = at(ConvexShape::Sphere(0.2), iso.point(local));
            let r = distance(&sphere, &boxed);
            let n = iso.vector(on_box - local).normalize_or_zero();
            assert_separated(&r, want, n, 2e-5);
            assert!(
                close_v(r.point_b, iso.point(on_box), 2e-5),
                "{local:?}: {r:?}"
            );
        }
    }

    #[test]
    fn parallel_capsules_measure_between_their_axes() {
        let a = at(
            ConvexShape::Capsule {
                half_height: 1.0,
                radius: 0.3,
            },
            Vec3::ZERO,
        );
        let b = at(
            ConvexShape::Capsule {
                half_height: 1.0,
                radius: 0.2,
            },
            Vec3::new(2.0, 0.5, 0.0),
        );
        let r = distance(&a, &b);
        assert_separated(&r, 1.5, Vec3::X, 1e-5);
        // Any pair across the shared span [-0.5, 1] of heights is closest.
        assert!(close(r.point_a.y, r.point_b.y, 1e-5));
        assert!((-0.5 - 1e-5..=1.0 + 1e-5).contains(&r.point_a.y), "{r:?}");
        assert!(close(r.point_a.x, 0.3, 1e-5) && close(r.point_b.x, 1.8, 1e-5));
        // End to end along one line.
        let c = at(
            ConvexShape::Capsule {
                half_height: 1.0,
                radius: 0.2,
            },
            Vec3::new(0.0, 3.0, 0.0),
        );
        let r = distance(&a, &c);
        assert_separated(&r, 0.5, Vec3::Y, 1e-5);
        assert!(close_v(r.point_a, Vec3::new(0.0, 1.3, 0.0), 1e-5));
    }

    #[test]
    fn skew_capsules_measure_along_their_common_perpendicular() {
        let a = at(
            ConvexShape::Capsule {
                half_height: 1.0,
                radius: 0.3,
            },
            Vec3::ZERO,
        );
        // B's axis lies in the plane x = 1.5 at 30° from Z: the common perpendicular is X.
        let b = turned(
            ConvexShape::Capsule {
                half_height: 1.0,
                radius: 0.2,
            },
            Vec3::new(1.5, 0.3, 0.0),
            Vec3::X,
            FRAC_PI_2 + 0.5,
        );
        let r = distance(&a, &b);
        assert_separated(&r, 1.0, Vec3::X, 1e-5);
        assert!(close_v(r.point_a, Vec3::new(0.3, 0.3, 0.0), 1e-5), "{r:?}");
        assert!(close_v(r.point_b, Vec3::new(1.3, 0.3, 0.0), 1e-5), "{r:?}");
    }

    #[test]
    fn boxes_face_to_face_and_edge_to_edge_match_the_analytic_gap() {
        let a = at(ConvexShape::Box(Vec3::ONE), Vec3::ZERO);
        // Turning B about X keeps its ±X faces facing A's.
        let b = turned(
            ConvexShape::Box(Vec3::new(0.5, 2.0, 0.5)),
            Vec3::new(1.75, 0.3, -0.2),
            Vec3::X,
            0.7,
        );
        let r = distance(&a, &b);
        assert_separated(&r, 0.25, Vec3::X, 1e-5);
        assert!(close(r.point_a.x, 1.0, 1e-5) && close(r.point_b.x, 1.25, 1e-5));
        // A turned 45° about Z has an edge along Z at x = √2; B turned 45° about Y has an edge
        // along Y at x = c − √2. The edges cross with a gap of 0.3.
        let a = turned(ConvexShape::Box(Vec3::ONE), Vec3::ZERO, Vec3::Z, FRAC_PI_4);
        let c = 2.0 * SQRT_2 + 0.3;
        let b = turned(
            ConvexShape::Box(Vec3::ONE),
            Vec3::new(c, 0.0, 0.0),
            Vec3::Y,
            FRAC_PI_4,
        );
        let r = distance(&a, &b);
        assert_separated(&r, 0.3, Vec3::X, 2e-5);
        assert!(
            close_v(r.point_a, Vec3::new(SQRT_2, 0.0, 0.0), 2e-5),
            "{r:?}"
        );
        assert!(close_v(r.point_b, Vec3::new(c - SQRT_2, 0.0, 0.0), 2e-5));
    }

    /// An octahedron with vertices at ±1 on each axis.
    fn octahedron() -> ConvexHull {
        let v = vec![Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
        let faces: [&[u32]; 8] = [
            &[0, 2, 4],
            &[1, 4, 2],
            &[0, 4, 3],
            &[1, 3, 4],
            &[0, 5, 2],
            &[1, 2, 5],
            &[0, 3, 5],
            &[1, 5, 3],
        ];
        ConvexHull::from_faces(v, &faces).unwrap()
    }

    #[test]
    fn a_hull_above_a_triangle_measures_from_its_lowest_vertex() {
        let hull = octahedron();
        let tri = ConvexShape::Triangle([
            Vec3::new(-2.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(2.0, 0.0, -1.0),
        ]);
        let a = at(ConvexShape::Hull(&hull), Vec3::new(0.1, 1.5, 0.2));
        let b = at(tri, Vec3::ZERO);
        let r = distance(&a, &b);
        assert_separated(&r, 0.5, -Vec3::Y, 1e-5);
        assert!(close_v(r.point_a, Vec3::new(0.1, 0.5, 0.2), 1e-5));
        assert!(close_v(r.point_b, Vec3::new(0.1, 0.0, 0.2), 1e-5));
        // Beside the triangle, an edge of the hull faces the triangle's edge from (−2, 0, −1) to
        // (2, 0, −1): the hull's vertex −Z is closest.
        let a = at(ConvexShape::Hull(&hull), Vec3::new(0.5, 0.0, -2.5));
        let r = distance(&a, &b);
        assert_separated(&r, 0.5, Vec3::Z, 1e-5);
        assert!(close_v(r.point_b, Vec3::new(0.5, 0.0, -1.0), 1e-5), "{r:?}");
    }

    #[test]
    fn swapping_the_shapes_swaps_the_points_and_flips_the_normal() {
        let hull = octahedron();
        let shapes = [
            ConvexShape::Sphere(0.4),
            ConvexShape::Capsule {
                half_height: 0.7,
                radius: 0.2,
            },
            ConvexShape::Box(Vec3::new(0.5, 0.3, 0.8)),
            ConvexShape::Hull(&hull),
            ConvexShape::Triangle([Vec3::ZERO, Vec3::X, Vec3::new(0.2, 0.9, 0.3)]),
            ConvexShape::Segment {
                a: Vec3::new(-0.5, 0.1, 0.0),
                b: Vec3::new(0.4, -0.2, 0.3),
                radius: 0.1,
            },
        ];
        for (i, sa) in shapes.iter().enumerate() {
            for (j, sb) in shapes.iter().enumerate() {
                let a = turned(
                    *sa,
                    Vec3::new(0.1, -0.2, 0.3),
                    Vec3::new(1.0, 1.0, 0.0),
                    0.4,
                );
                let b = turned(
                    *sb,
                    Vec3::new(2.6, 0.9, -0.4),
                    Vec3::new(0.0, 1.0, 1.0),
                    1.1,
                );
                let ab = distance(&a, &b);
                let ba = distance(&b, &a);
                assert!(!ab.overlap && !ba.overlap, "{i} {j}");
                assert!(
                    close(ab.distance, ba.distance, 1e-5),
                    "{i} {j}: {ab:?} {ba:?}"
                );
                assert!(
                    close_v(ab.normal, -ba.normal, 1e-4),
                    "{i} {j}: {ab:?} {ba:?}"
                );
                // Closest points are unique here except along parallel features; their gaps
                // must agree either way.
                assert!(close_v(
                    ab.point_b - ab.point_a,
                    ba.point_a - ba.point_b,
                    1e-4
                ));
            }
        }
    }

    #[test]
    fn overlapping_cores_report_a_point_inside_both() {
        let a_iso = Iso::new(
            Vec3::new(0.2, 0.1, 0.0),
            Quat::from_axis_angle(Vec3::Y, 0.3),
        );
        let b_iso = Iso::new(
            Vec3::new(1.6, 0.5, 0.2),
            Quat::from_axis_angle(Vec3::Z, 0.5),
        );
        let (ha, hb) = (Vec3::ONE, Vec3::new(0.8, 0.8, 0.8));
        let a = Convex::new(ConvexShape::Box(ha), a_iso);
        let b = Convex::new(ConvexShape::Box(hb), b_iso);
        let r = distance(&a, &b);
        assert!(r.overlap, "{r:?}");
        assert_eq!((r.distance, r.normal), (0.0, Vec3::ZERO));
        let inside = |iso: &Iso, h: Vec3, p: Vec3| {
            let l = iso.inverse_point(p).abs();
            l.x <= h.x + 1e-5 && l.y <= h.y + 1e-5 && l.z <= h.z + 1e-5
        };
        assert!(inside(&a_iso, ha, r.point_a) && inside(&b_iso, hb, r.point_a));
        assert_eq!(r.point_a, r.point_b);
        // A sphere's centre inside the box, a capsule's core through it, a triangle piercing it.
        for shape in [
            ConvexShape::Sphere(0.1),
            ConvexShape::Capsule {
                half_height: 3.0,
                radius: 0.1,
            },
            ConvexShape::Triangle([
                Vec3::new(-3.0, 0.0, -3.0),
                Vec3::new(3.0, 0.0, -3.0),
                Vec3::new(0.0, 0.0, 3.0),
            ]),
        ] {
            let s = at(shape, Vec3::new(0.5, 0.3, 0.2));
            for r in [distance(&s, &a), distance(&a, &s)] {
                assert!(r.overlap, "{shape:?}: {r:?}");
                assert!(inside(&a_iso, ha, r.point_a), "{shape:?}: {r:?}");
            }
        }
    }

    #[test]
    fn degenerate_cores_give_exact_distances_or_overlap() {
        // A zero-length segment is a point.
        let tri = at(
            ConvexShape::Triangle([
                Vec3::new(-1.0, 0.0, -1.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, -1.0),
            ]),
            Vec3::ZERO,
        );
        let point = at(
            ConvexShape::Segment {
                a: Vec3::ZERO,
                b: Vec3::ZERO,
                radius: 0.0,
            },
            Vec3::new(0.2, 0.7, 0.1),
        );
        assert_separated(&distance(&point, &tri), 0.7, -Vec3::Y, 1e-6);
        // Collinear segments that share a stretch overlap; end to end they do not.
        let seg = |a: Vec3, b: Vec3| {
            at(
                ConvexShape::Segment { a, b, radius: 0.0 },
                Vec3::new(5.0, 1.0, 0.0),
            )
        };
        let r = distance(
            &seg(Vec3::ZERO, Vec3::X * 2.0),
            &seg(Vec3::X, Vec3::X * 3.0),
        );
        assert!(r.overlap && (1.0 + 5.0 - 1e-5..=2.0 + 5.0 + 1e-5).contains(&r.point_a.x));
        let r = distance(
            &seg(Vec3::ZERO, Vec3::X),
            &seg(Vec3::X * 1.5, Vec3::X * 3.0),
        );
        assert_separated(&r, 0.5, Vec3::X, 1e-6);
        // Coplanar triangles: apart, overlapping, and stacked parallel.
        let t1 = [Vec3::ZERO, Vec3::Z, Vec3::X];
        let t2 = [
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(1.5, 0.0, 1.0),
            Vec3::new(2.5, 0.0, 0.0),
        ];
        let r = distance(
            &at(ConvexShape::Triangle(t1), Vec3::ZERO),
            &at(ConvexShape::Triangle(t2), Vec3::ZERO),
        );
        assert_separated(&r, 0.5, Vec3::X, 1e-6);
        assert!(close_v(r.point_a, Vec3::X, 1e-6) && close_v(r.point_b, t2[0], 1e-6));
        let r = distance(
            &at(ConvexShape::Triangle(t1), Vec3::ZERO),
            &at(ConvexShape::Triangle(t1), Vec3::new(0.3, 0.0, 0.2)),
        );
        assert!(r.overlap, "{r:?}");
        let r = distance(
            &at(ConvexShape::Triangle(t1), Vec3::ZERO),
            &at(ConvexShape::Triangle(t1), Vec3::new(0.1, 0.3, 0.1)),
        );
        assert_separated(&r, 0.3, Vec3::Y, 1e-6);
        // Boxes side by side with coplanar faces, then touching (the cores meet: overlap).
        let unit = ConvexShape::Box(Vec3::ONE);
        let r = distance(&at(unit, Vec3::ZERO), &at(unit, Vec3::new(2.1, 0.0, 0.0)));
        assert_separated(&r, 0.1, Vec3::X, 1e-6);
        for offset in [Vec3::new(2.0, 0.0, 0.0), Vec3::new(2.0, 0.5, -0.3)] {
            assert!(distance(&at(unit, Vec3::ZERO), &at(unit, offset)).overlap);
        }
        // A sphere centre exactly on a face: the cores touch.
        let r = distance(
            &at(ConvexShape::Sphere(0.5), Vec3::new(0.3, 1.0, 0.2)),
            &at(unit, Vec3::ZERO),
        );
        assert!(r.overlap && close_v(r.point_a, Vec3::new(0.3, 1.0, 0.2), 1e-6));
        // The smallest box beside the largest.
        let tiny = ConvexShape::Box(Vec3::splat(crate::shape::MIN_EXTENT));
        let huge = ConvexShape::Box(Vec3::splat(crate::shape::MAX_EXTENT));
        let r = distance(
            &at(tiny, Vec3::new(3.0, 1e5 + 0.5, -7.0)),
            &at(huge, Vec3::ZERO),
        );
        assert_separated(&r, 0.5 - 1e-4, -Vec3::Y, 2e-2);
    }

    #[test]
    fn far_from_the_origin_centimetre_gaps_stay_exact() {
        let far = Vec3::new(10_000.0, -7_000.0, 12_000.0);
        let unit = ConvexShape::Box(Vec3::ONE);
        // The f32 position nearest 1 cm beyond touching, and the gap it really leaves.
        let bx = far.x + 2.01;
        let gap = (bx as f64 - far.x as f64 - 2.0) as f32;
        let a = at(unit, far);
        let b = turned(unit, Vec3::new(bx, far.y + 0.3, far.z), Vec3::X, 0.6);
        let r = distance(&a, &b);
        assert!(close(r.distance, gap, 1e-5), "{r:?}, want {gap}");
        assert!(close_v(r.normal, Vec3::X, 1e-4));
        // Any pair placed far away measures what the same pair measures at the origin (positions
        // on a 1/256 grid are exact at both places).
        let hull = octahedron();
        let shapes = [
            ConvexShape::Sphere(0.25),
            ConvexShape::Capsule {
                half_height: 0.5,
                radius: 0.125,
            },
            ConvexShape::Box(Vec3::new(0.5, 0.25, 0.75)),
            ConvexShape::Hull(&hull),
            ConvexShape::Triangle([Vec3::ZERO, Vec3::X, Vec3::new(0.25, 0.75, 0.5)]),
        ];
        let rotation =
            |k: usize| Quat::from_axis_angle(Vec3::new(1.0, k as f32, 2.0), 0.3 * k as f32);
        for (i, sa) in shapes.iter().enumerate() {
            for (j, sb) in shapes.iter().enumerate() {
                let pa = Vec3::new(0.25, -0.5, 0.125);
                let pb = pa + Vec3::new(1.5 + 0.25 * j as f32, 0.375, -0.25 * i as f32);
                let near = distance(
                    &Convex::new(*sa, Iso::new(pa, rotation(i))),
                    &Convex::new(*sb, Iso::new(pb, rotation(j + 3))),
                );
                let away = distance(
                    &Convex::new(*sa, Iso::new(pa + far, rotation(i))),
                    &Convex::new(*sb, Iso::new(pb + far, rotation(j + 3))),
                );
                assert!(!near.overlap && !away.overlap, "{i} {j}");
                assert!(
                    close(near.distance, away.distance, 2e-5),
                    "{i} {j}: {near:?} {away:?}"
                );
                assert!(close_v(near.normal, away.normal, 1e-3), "{i} {j}");
                // The points carry only the rounding of world coordinates (under a millimetre).
                assert!(close_v(near.point_a + far, away.point_a, 2e-3), "{i} {j}");
            }
        }
    }

    /// The smallest distance from the origin to a triangle, by sampling it finely.
    fn sampled_triangle_distance(a: Vec3, b: Vec3, c: Vec3) -> f32 {
        let n = 300;
        let mut best = f32::INFINITY;
        for i in 0..=n {
            for j in 0..=(n - i) {
                let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
                best = best.min((a + (b - a) * u + (c - a) * v).length());
            }
        }
        best
    }

    /// A small deterministic generator (splitmix64) for the tests.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.next()
        }
        fn vec(&mut self, r: f32) -> Vec3 {
            Vec3::new(self.range(-r, r), self.range(-r, r), self.range(-r, r))
        }
    }

    #[test]
    fn the_triangle_solver_finds_the_sampled_nearest_point() {
        let mut rng = Rng(7);
        for _ in 0..200 {
            let w = [rng.vec(2.0), rng.vec(2.0), rng.vec(2.0), Vec3::ZERO];
            let c = closest_on_triangle(&w.map(D3::of), 0, 1, 2);
            let v = c.v.to_vec3();
            let sampled = sampled_triangle_distance(w[0], w[1], w[2]);
            // Sampling only finds points at least as far as the true nearest one.
            let found = v.length();
            assert!(found <= sampled + 1e-5, "{w:?}: {found} > {sampled}");
            assert!(found >= sampled - 0.02, "{w:?}: {found} < {sampled}");
            // The weights rebuild the point from the vertices it keeps.
            let rebuilt = (0..3).fold(Vec3::ZERO, |s, i| s + w[i] * c.weights[i] as f32);
            assert!(close_v(rebuilt, v, 1e-5));
            assert!((c.weights.iter().sum::<f64>() - 1.0).abs() <= 1e-9);
            for (i, &l) in c.weights.iter().enumerate() {
                assert!(l >= 0.0 && (l == 0.0 || c.mask & (1 << i) != 0));
            }
        }
        // A collinear triangle is its longest edge.
        let w = [
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::new(3.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::ZERO,
        ];
        let v = closest_on_triangle(&w.map(D3::of), 0, 1, 2).v.to_vec3();
        assert!(close_v(v, Vec3::Y, 1e-6) && v.is_finite());
    }

    #[test]
    fn the_tetrahedron_solver_contains_the_origin_or_finds_the_nearest_face() {
        let mut rng = Rng(11);
        let mut inside = 0;
        for _ in 0..200 {
            let w = [rng.vec(2.0), rng.vec(2.0), rng.vec(2.0), rng.vec(2.0)];
            let c = closest_on_tetrahedron(&w.map(D3::of));
            let v = c.v.to_vec3();
            let faces = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];
            let nearest_face = faces
                .iter()
                .map(|f| sampled_triangle_distance(w[f[0]], w[f[1]], w[f[2]]))
                .fold(f32::INFINITY, f32::min);
            let rebuilt = (0..4).fold(Vec3::ZERO, |s, i| s + w[i] * c.weights[i] as f32);
            assert!(close_v(rebuilt, v, 1e-5));
            if c.mask == 0b1111 {
                inside += 1;
                assert_eq!(v, Vec3::ZERO);
                assert!(c.weights.iter().all(|&l| l >= 0.0));
            } else {
                let found = v.length();
                assert!(
                    found <= nearest_face + 1e-5,
                    "{w:?}: {found} > {nearest_face}"
                );
                assert!(
                    found >= nearest_face - 0.02,
                    "{w:?}: {found} < {nearest_face}"
                );
            }
        }
        assert!(inside > 10, "only {inside} tetrahedra held the origin");
        // A flat tetrahedron is searched by its faces: here it lies in the plane y = 1.
        let w = [
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(0.1, 1.0, 0.1),
        ];
        let c = closest_on_tetrahedron(&w.map(D3::of));
        assert!(close_v(c.v.to_vec3(), Vec3::Y, 1e-6), "{c:?}");
    }

    #[test]
    fn a_sphere_cast_into_a_box_face_stops_at_the_face() {
        let boxed = at(ConvexShape::Box(Vec3::ONE), Vec3::ZERO);
        let sphere = at(ConvexShape::Sphere(0.5), Vec3::new(5.0, 0.2, 0.1));
        let hit = cast(&sphere, Vec3::new(-8.0, 0.0, 0.0), &boxed, 0.0).unwrap();
        // The centre reaches x = 1.5 after 3.5 of the 8 m.
        assert!(!hit.started_overlapping);
        assert!(
            hit.fraction <= 0.4375 + 1e-6 && hit.fraction > 0.4375 - 2e-5,
            "{hit:?}"
        );
        assert!(close_v(hit.normal, Vec3::X, 1e-5));
        assert!(
            close_v(hit.point, Vec3::new(1.0, 0.2, 0.1), 1e-4),
            "{hit:?}"
        );
        // Obliquely onto the top of a turned box: the centre's local height falls from 4 to
        // 1 + 0.5 halfway along.
        let half = Vec3::new(2.0, 1.0, 1.5);
        let iso = Iso::new(
            Vec3::new(3.0, -1.0, 2.0),
            Quat::from_axis_angle(Vec3::new(1.0, 0.5, -0.3), 0.9),
        );
        let boxed = Convex::new(ConvexShape::Box(half), iso);
        let sphere = at(
            ConvexShape::Sphere(0.5),
            iso.point(Vec3::new(0.3, 4.0, -0.2)),
        );
        let along = iso.vector(Vec3::new(0.5, -5.0, 0.25));
        let hit = cast(&sphere, along, &boxed, 0.0).unwrap();
        assert!(
            hit.fraction <= 0.5 + 1e-6 && hit.fraction > 0.5 - 2e-5,
            "{hit:?}"
        );
        assert!(close_v(hit.normal, iso.vector(Vec3::Y), 1e-4));
        assert!(close_v(
            hit.point,
            iso.point(Vec3::new(0.55, 1.0, -0.075)),
            2e-4
        ));
    }

    #[test]
    fn a_box_cast_onto_a_box_edge_stops_at_first_contact() {
        // Turned 45° about Z, the lower box has its top edge along Z at y = √2.
        let edge = turned(ConvexShape::Box(Vec3::ONE), Vec3::ZERO, Vec3::Z, FRAC_PI_4);
        let falling = at(ConvexShape::Box(Vec3::splat(0.5)), Vec3::new(0.3, 3.0, 0.0));
        let hit = cast(&falling, Vec3::new(0.0, -3.0, 0.0), &edge, 0.0).unwrap();
        let want = (2.5 - SQRT_2) / 3.0;
        assert!(
            hit.fraction <= want + 1e-6 && hit.fraction > want - 2e-5,
            "{hit:?}"
        );
        assert!(close_v(hit.normal, Vec3::Y, 1e-4), "{hit:?}");
        assert!(close(hit.point.x, 0.0, 1e-4) && close(hit.point.y, SQRT_2, 1e-4));
        assert!(hit.point.z.abs() <= 0.5 + 1e-4);
        // Edge first onto the other's edge, crossed: the same gap closes along X.
        let a = turned(
            ConvexShape::Box(Vec3::ONE),
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::Z,
            FRAC_PI_4,
        );
        let b = turned(
            ConvexShape::Box(Vec3::ONE),
            Vec3::new(2.0, 0.1, 0.2),
            Vec3::Y,
            FRAC_PI_4,
        );
        let hit = cast(&a, Vec3::new(2.0, 0.0, 0.0), &b, 0.0).unwrap();
        // Closing at 2 m per unit of fraction, the 0.1 mm tolerance is 5e-5 of fraction.
        let want = (4.0 - 2.0 * SQRT_2) / 2.0;
        assert!(
            hit.fraction <= want + 1e-6 && hit.fraction > want - 6e-5,
            "{hit:?}"
        );
        assert!(close_v(hit.normal, -Vec3::X, 1e-3), "{hit:?}");
    }

    #[test]
    fn a_capsule_sliding_past_a_corner_misses_unless_the_target_reaches_it() {
        let boxed = at(ConvexShape::Box(Vec3::ONE), Vec3::ZERO);
        let capsule = ConvexShape::Capsule {
            half_height: 0.5,
            radius: 0.2,
        };
        // Along the line x + z = c, which passes 0.3 from the vertical edge at x = z = 1: the
        // surfaces come no closer than 0.1.
        let c = 2.0 + 0.3 * SQRT_2;
        let start = at(capsule, Vec3::new(-2.0, 0.0, c + 2.0));
        let path = Vec3::new(c + 4.0, 0.0, -(c + 4.0));
        assert_eq!(cast(&start, path, &boxed, 0.0), None);
        assert_eq!(cast(&start, path, &boxed, 0.09), None);
        // With a target of 0.15 the core must come within 0.35 of the edge: where
        // |(x − 1, z − 1)| = 0.35 on the line, first at x = 1 + 0.15√2 − √0.01625.
        let hit = cast(&start, path, &boxed, 0.15).unwrap();
        let x = 1.0 + 0.15 * SQRT_2 - 0.01625f32.sqrt();
        let want = (x + 2.0) / (c + 4.0);
        assert!(
            hit.fraction <= want + 1e-6 && hit.fraction > want - 1e-4,
            "{hit:?}"
        );
        // The normal runs from the edge to the core there.
        let n = Vec3::new(x - 1.0, 0.0, c - x - 1.0) * (1.0 / 0.35);
        assert!(close_v(hit.normal, n, 1e-3), "{hit:?}, want {n:?}");
        assert!(
            close_v(hit.point, Vec3::new(1.0, hit.point.y, 1.0), 1e-4),
            "{hit:?}"
        );
    }

    #[test]
    fn a_cast_that_starts_within_the_target_reports_fraction_zero() {
        let boxed = at(ConvexShape::Box(Vec3::ONE), Vec3::ZERO);
        let push = Vec3::new(0.0, -2.0, 0.0);
        // Only the rounding overlaps: the normal comes from the closest features.
        let hit = cast(
            &at(ConvexShape::Sphere(0.5), Vec3::new(1.3, 0.0, 0.0)),
            push,
            &boxed,
            0.0,
        )
        .unwrap();
        assert!(hit.started_overlapping && hit.fraction == 0.0);
        assert!(close_v(hit.normal, Vec3::X, 1e-5) && close_v(hit.point, Vec3::X, 1e-5));
        // Apart, but already within the target.
        let hit = cast(
            &at(ConvexShape::Sphere(0.5), Vec3::new(1.6, 0.0, 0.0)),
            push,
            &boxed,
            0.2,
        )
        .unwrap();
        assert!(hit.started_overlapping && close_v(hit.normal, Vec3::X, 1e-5));
        // The cores overlap: the normal points back along the motion.
        let inside = at(ConvexShape::Sphere(0.5), Vec3::new(0.5, 0.0, 0.0));
        let hit = cast(&inside, push, &boxed, 0.0).unwrap();
        assert!(hit.started_overlapping && close_v(hit.normal, Vec3::Y, 1e-6));
        let hit = cast(&inside, Vec3::ZERO, &boxed, 0.0).unwrap();
        assert!(hit.started_overlapping && close(hit.normal.length(), 1.0, 1e-5));
    }

    #[test]
    fn a_cast_with_a_target_stops_that_far_away() {
        let boxed = at(ConvexShape::Box(Vec3::ONE), Vec3::ZERO);
        let sphere = at(ConvexShape::Sphere(0.5), Vec3::new(5.0, 0.2, 0.1));
        let hit = cast(&sphere, Vec3::new(-8.0, 0.0, 0.0), &boxed, 0.25).unwrap();
        // The surfaces are 0.25 apart when the centre is at x = 1.75.
        let want = (5.0 - 1.75) / 8.0;
        assert!(
            hit.fraction <= want + 1e-6 && hit.fraction > want - 2e-5,
            "{hit:?}"
        );
        assert!(close_v(hit.point, Vec3::new(1.0, 0.2, 0.1), 1e-4));
    }

    #[test]
    fn casts_that_move_away_fall_short_or_pass_by_miss() {
        let boxed = at(ConvexShape::Box(Vec3::ONE), Vec3::ZERO);
        let sphere = at(ConvexShape::Sphere(0.5), Vec3::new(5.0, 0.2, 0.1));
        assert_eq!(cast(&sphere, Vec3::new(8.0, 0.0, 0.0), &boxed, 0.0), None);
        assert_eq!(cast(&sphere, Vec3::new(-3.4, 0.0, 0.0), &boxed, 0.0), None);
        assert_eq!(cast(&sphere, Vec3::new(0.0, 9.0, 0.0), &boxed, 0.0), None);
        assert_eq!(cast(&sphere, Vec3::ZERO, &boxed, 0.0), None);
        // Exactly long enough: a hit at the very end.
        let hit = cast(&sphere, Vec3::new(-3.5, 0.0, 0.0), &boxed, 0.0).unwrap();
        assert!(hit.fraction <= 1.0 && hit.fraction > 1.0 - 1e-4, "{hit:?}");
    }
}
