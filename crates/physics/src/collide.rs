//! Narrow phase: contact manifolds between two placed shapes.
//!
//! Contract (the solver depends on every line of it):
//! - [`collide`] appends manifolds to `out`; it never clears `out`.
//! - A manifold's `normal` is a unit world vector pointing from shape A towards shape B.
//! - Each point's `point` is the world-space midpoint between the two surfaces at that contact,
//!   and `separation` is the signed distance between the surfaces along `normal` (negative when
//!   they overlap). Only points with `separation <= speculative` are produced: speculative points
//!   let the solver stop approaching bodies before they touch.
//! - At most [`MAX_MANIFOLD_POINTS`] points per manifold, chosen to keep the deepest point (to
//!   within a thousandth of the patch's width, at most 2.5 mm) and the largest area (so a box
//!   resting on a face keeps 4 corners), and to stay the same while the bodies rest.
//! - Each point has a feature `id` that stays the same from step to step while the same features
//!   touch (for example the reference face and the incident vertex or clipped edge), so the
//!   solver can carry impulses over; 0 means "no feature", matched by position instead.
//! - Two convex shapes give at most one manifold. A convex shape against a mesh or heightfield
//!   gives one manifold per touching triangle group (merged when normals agree within
//!   [`MERGE_NORMAL_COS`]), at most [`MAX_MESH_MANIFOLDS`], deepest kept; ids carry the triangle id
//!   in their upper bits so they stay unique across manifolds.
//! - Meshes and heightfields never collide with each other (both are static).
//! - The result depends only on the inputs (no global state), so it is deterministic.
//!
//! How (after Dirk Gregorius, "The Separating Axis Test between Convex Polyhedra", GDC 2013, and
//! "Robust Contact Creation for Physics Simulations", GDC 2015):
//! - Spheres and capsules are a point or a segment (their core) plus a radius. Their contacts come
//!   from the exact closest points between the core and the other shape; a core inside a
//!   polytope uses the axis of least penetration instead.
//! - Boxes and hulls are polytopes (a box borrows the cuboid's half-edge tables, with its corners
//!   on the stack). Two polytopes run the separating-axis test over both shapes' face normals and
//!   the cross products of edge pairs that build a face of the Minkowski difference (their arcs
//!   cross on the Gauss map). An axis whose bound, from points known to lie in the other shape,
//!   shows it cannot change the answer is skipped, so large hulls cost about linear time. A face
//!   axis clips the other shape's face most opposed to it among those round its deepest vertex
//!   against the reference face's side planes and keeps up to four points; an edge axis gives one
//!   point between the two edges. Face axes win unless an edge axis is clearly better, so a resting
//!   contact keeps its features from step to step. Overlapping, the best axis is the minimum
//!   translation; apart, its separation is a lower bound of the distance (exact when a face and
//!   a vertex, or two edges, are closest), so speculative contacts err on the early side.
//! - Triangles of meshes and heightfields are one-sided flat polytopes. A convex whose centre is
//!   behind a triangle, or whose contact normal points behind it, does not touch it. A normal
//!   that comes from a triangle's inactive edge or vertex is replaced by the face normal, so
//!   bodies slide over the seams of flat floors; beside the triangle that contact is never
//!   deeper than the true distance (across a crease the plane can pass above the surface). A
//!   ridge's normals lie between its two faces' normals, and a vertex's never lean out over a
//!   seam there: a normal past them is a neighbour's, so a gentle crease or a straight border is
//!   no wall. Such borrowed contacts, and those of triangles the centre is just behind (by up to
//!   the speculative distance), count only when no triangle touches from the front with a normal
//!   of its own: then the convex sank deep into a corner, or a thin body went through the
//!   surface on a hard landing, and they push it back out. Manifolds of neighbouring triangles
//!   with the same normal merge, keeping the corners of the patch they share.
//! - Every computation is `+ - * /` and `sqrt` on `f32`, with fixed-size buffers on the stack.

use crate::hull::{ConvexHull, HalfEdge, HullFace, MAX_HULL_VERTICES};
use crate::math::{Iso, Mat3, Vec3};
use crate::mesh::{Heightfield, TriangleMesh, TriangleRef};
use crate::shape::Shape;

pub const MAX_MANIFOLD_POINTS: usize = 4;
pub const MAX_MESH_MANIFOLDS: usize = 8;
/// Manifolds against neighbouring triangles whose normals' dot product exceeds this merge.
pub const MERGE_NORMAL_COS: f32 = 0.9998;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ManifoldPoint {
    pub point: Vec3,
    pub separation: f32,
    pub id: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Manifold {
    pub normal: Vec3,
    pub points: [ManifoldPoint; MAX_MANIFOLD_POINTS],
    pub count: u8,
}

const _: () = assert!(std::mem::size_of::<Manifold>() == 96);

impl Manifold {
    #[inline]
    pub fn points(&self) -> &[ManifoldPoint] {
        &self.points[..usize::from(self.count)]
    }
    /// The smallest separation of its points (`f32::INFINITY` when empty).
    #[inline]
    pub fn min_separation(&self) -> f32 {
        self.points()
            .iter()
            .fold(f32::INFINITY, |m, p| m.min(p.separation))
    }
}

/// Appends the contact manifolds between shape `a` at `iso_a` and shape `b` at `iso_b` to `out`
/// (see the module doc for the contract).
pub fn collide(
    a: &Shape,
    iso_a: &Iso,
    b: &Shape,
    iso_b: &Iso,
    speculative: f32,
    out: &mut Vec<Manifold>,
) {
    let mut buf: [Manifold; MAX_MESH_MANIFOLDS] = Default::default();
    let len = collide_stack(a, iso_a, b, iso_b, speculative, &mut buf);
    out.extend_from_slice(&buf[..len]);
}

/// Writes the contact manifolds between `a` and `b` into a fixed stack buffer (at most
/// [`MAX_MESH_MANIFOLDS`]) and returns how many were written, with zero heap allocation.
pub(crate) fn collide_stack(
    a: &Shape,
    iso_a: &Iso,
    b: &Shape,
    iso_b: &Iso,
    speculative: f32,
    out: &mut [Manifold; MAX_MESH_MANIFOLDS],
) -> usize {
    let spec = if speculative.is_nan() {
        0.0
    } else {
        speculative
    };
    match (StaticGeometry::of(a), StaticGeometry::of(b)) {
        (Some(_), Some(_)) => 0,
        (Some(g), None) => collide_static(g, iso_a, b, iso_b, spec, false, out),
        (None, Some(g)) => collide_static(g, iso_b, a, iso_a, spec, true, out),
        (None, None) => {
            // Each pair is written once, lower-ranked shape first; the other order swaps the
            // shapes and flips the normal, so both orders give the same points.
            let m = if rank(a) <= rank(b) {
                convex_pair(a, iso_a, b, iso_b, spec)
            } else {
                convex_pair(b, iso_b, a, iso_a, spec).map(|mut m| {
                    m.normal = -m.normal;
                    m
                })
            };
            if let Some(mut m) = m {
                unique_ids(&mut m);
                if let Some(m) = sanitize_finite(m) {
                    out[0] = m;
                    return 1;
                }
            }
            0
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Tolerances.

/// Contact normals within this cosine of a face normal count as that face's (a capsule lying on
/// a face, a box resting on a triangle).
const FACE_COS: f32 = 0.9998;
/// Segments closer to parallel than this sine get two contacts, at the ends of their overlap.
const PARALLEL_SIN: f32 = 0.02;
/// Edge pairs closer to parallel than this sine are skipped by the separating-axis test: their
/// cross product has no reliable direction, and the face axes cover them.
const EDGE_SKIP_SIN: f32 = 0.005;
/// An edge pair's axis must lie on both edges' arcs of normals within this angle (radians).
const ARC_SLACK: f32 = 1e-3;
/// An edge axis replaces the best face axis only when its separation is larger by this fraction
/// of the face separation's magnitude plus the absolute tolerance; the second shape's face
/// replaces the first's likewise with [`FACE_REL_TOL`]. Keeps resting contacts on the same
/// features from step to step.
const EDGE_REL_TOL: f32 = 0.1;
const FACE_REL_TOL: f32 = 0.05;
/// The absolute part of those tolerances (metres), scaled down for small shapes.
const ABS_TOL: f32 = 0.0025;
/// Normals pointing behind a triangle by more than this cosine are on its back side: ignored.
const BACK_COS: f32 = -0.01;
/// How close (as a sine) a normal must be to an edge's arc to count as that edge's rather than
/// a vertex's, when deciding which triangle feature a normal belongs to.
const EDGE_FEATURE_SIN: f32 = 1e-3;
/// How far (as a sine) a normal must turn past a ridge's far face normal to count as that face's
/// rather than the ridge's.
const RIDGE_PAST: f32 = 1e-6;
/// Low bits of a mesh contact id that name its features; the triangle id fills the rest
/// (25 bits cover every mesh and heightfield triangle id).
const LOCAL_ID_BITS: u32 = 7;

/// `x` beats `y` by more than the relative and absolute tolerances.
fn clearly_greater(x: f32, y: f32, rel: f32, abs: f32) -> bool {
    x > y + rel * y.abs() + abs
}

/// The absolute tolerance for preferring face axes, for a shape of thickness about `size`.
fn abs_tolerance(size: f32) -> f32 {
    (0.02 * size).min(ABS_TOL)
}

// ---------------------------------------------------------------------------------------------
// Dispatch.

/// A shape's place in the pair table: each convex pair is handled lower rank first.
fn rank(s: &Shape) -> u8 {
    match s {
        Shape::Sphere { .. } => 0,
        Shape::Capsule { .. } => 1,
        Shape::Box { .. } => 2,
        Shape::Hull(_) => 3,
        Shape::Mesh(_) | Shape::Heightfield(_) => 4,
    }
}

/// The manifold of two convex shapes with `rank(a) <= rank(b)`, normal from A to B.
fn convex_pair(a: &Shape, iso_a: &Iso, b: &Shape, iso_b: &Iso, spec: f32) -> Option<Manifold> {
    match (a, b) {
        (Shape::Sphere { radius: ra }, Shape::Sphere { radius: rb }) => {
            sphere_sphere(iso_a.position, *ra, iso_b.position, *rb, spec)
        }
        (
            Shape::Sphere { radius: ra },
            Shape::Capsule {
                half_height,
                radius: rb,
            },
        ) => {
            let (b0, b1) = capsule_ends(iso_b, *half_height);
            sphere_capsule(iso_a.position, *ra, b0, b1, *rb, spec)
        }
        (
            Shape::Capsule {
                half_height: ha,
                radius: ra,
            },
            Shape::Capsule {
                half_height: hb,
                radius: rb,
            },
        ) => {
            let (a0, a1) = capsule_ends(iso_a, *ha);
            let (b0, b1) = capsule_ends(iso_b, *hb);
            capsule_capsule(a0, a1, *ra, b0, b1, *rb, spec)
        }
        (Shape::Sphere { radius }, _) => {
            let poly = PolyShape::of(b)?;
            sphere_poly(iso_a.position, *radius, &poly.view(), iso_b, spec)
        }
        (
            Shape::Capsule {
                half_height,
                radius,
            },
            _,
        ) => {
            let poly = PolyShape::of(b)?;
            capsule_poly(*half_height, *radius, iso_a, &poly.view(), iso_b, spec)
        }
        _ => {
            let (pa, pb) = (PolyShape::of(a)?, PolyShape::of(b)?);
            let rel = Frame::new(&iso_a.inverse_mul(iso_b));
            let mut m = poly_poly(&pa.view(), &pb.view(), &rel, spec)?;
            to_world(&mut m, iso_a);
            Some(m)
        }
    }
}

/// The world ends of a capsule's core.
fn capsule_ends(iso: &Iso, half_height: f32) -> (Vec3, Vec3) {
    let axis = iso.vector(Vec3::new(0.0, half_height, 0.0));
    (iso.position - axis, iso.position + axis)
}

/// Moves a manifold computed in the frame `iso` places into the world.
fn to_world(m: &mut Manifold, iso: &Iso) {
    m.normal = iso.vector(m.normal);
    for p in &mut m.points[..usize::from(m.count)] {
        p.point = iso.point(p.point);
    }
}

/// Normalizes `m.normal` and retains only finite points; returns `None` when nothing usable is left.
fn sanitize_finite(mut m: Manifold) -> Option<Manifold> {
    let n = m.normal.try_normalize()?;
    if !n.is_finite() {
        return None;
    }
    m.normal = n;
    let mut k = 0usize;
    for i in 0..usize::from(m.count).min(MAX_MANIFOLD_POINTS) {
        let p = m.points[i];
        if p.point.is_finite() && p.separation.is_finite() {
            m.points[k] = p;
            k += 1;
        }
    }
    for p in &mut m.points[k..] {
        *p = ManifoldPoint::default();
    }
    m.count = k as u8;
    (k > 0).then_some(m)
}

// ---------------------------------------------------------------------------------------------
// Frames and feature ids.

/// A rigid transform as a matrix: cheaper than a quaternion when many points move.
#[derive(Clone, Copy, Debug)]
struct Frame {
    rot: Mat3,
    pos: Vec3,
}

impl Frame {
    const IDENTITY: Frame = Frame {
        rot: Mat3::IDENTITY,
        pos: Vec3::ZERO,
    };

    fn new(iso: &Iso) -> Frame {
        Frame {
            rot: iso.rotation.to_mat3(),
            pos: iso.position,
        }
    }
    fn point(&self, p: Vec3) -> Vec3 {
        self.rot.mul_vec(p) + self.pos
    }
    fn vector(&self, v: Vec3) -> Vec3 {
        self.rot.mul_vec(v)
    }
    fn inverse_vector(&self, v: Vec3) -> Vec3 {
        let c = &self.rot.cols;
        Vec3::new(c[0].dot(v), c[1].dot(v), c[2].dot(v))
    }
}

/// The finalizer of MurmurHash3: spreads every input bit over the output.
fn fmix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

/// Combines a feature id with one more feature number (order matters).
fn mix(h: u32, v: u32) -> u32 {
    fmix(h.rotate_left(11) ^ v.wrapping_mul(0x9e37_79b9).wrapping_add(0x7f4a_7c15))
}

// Kinds of contact, the first number of every id.
const KIND_SPHERE: u32 = 1;
const KIND_SPHERE_CAPSULE: u32 = 2;
const KIND_CAPSULES: u32 = 3;
const KIND_SPHERE_POLY: u32 = 4;
const KIND_CAPSULE_FACE: u32 = 5;
const KIND_CAPSULE_EDGE: u32 = 6;
const KIND_CAPSULE_POINT: u32 = 7;
const KIND_FACE_A: u32 = 8;
const KIND_FACE_B: u32 = 9;
const KIND_EDGES: u32 = 10;
const KIND_TRIANGLE_FACE: u32 = 11;
const KIND_TRIANGLE_POLY_FACE: u32 = 12;
// Feature numbers inside a kind.
const KEY_VERTEX: u32 = 0x1000_0000;
const KEY_DEEPEST: u32 = 0x2000_0000;
const TAG_INCIDENT_EDGE: u32 = 0x4000_0000;
const TAG_REFERENCE_SIDE: u32 = 0x8000_0000;

/// Makes the ids of a convex pair's manifold nonzero and distinct (a hash collision is
/// vanishingly rare; this keeps the contract even then).
fn unique_ids(m: &mut Manifold) {
    for i in 0..usize::from(m.count) {
        let mut id = m.points[i].id;
        while id == 0 || m.points[..i].iter().any(|p| p.id == id) {
            id = mix(id, 0x5bd1_e995);
        }
        m.points[i].id = id;
    }
}

/// Turns a triangle manifold's feature hashes into final ids: the triangle id in the upper bits,
/// a nonzero local number below, distinct within the manifold. Colliding local numbers probe in
/// the order of their full hashes, so the result depends only on which features touch.
fn assign_triangle_ids(m: &mut Manifold, triangle: u32) {
    let local_count = (1u32 << LOCAL_ID_BITS) - 1;
    let mut order = [0usize, 1, 2, 3];
    let n = usize::from(m.count).min(MAX_MANIFOLD_POINTS);
    // Insertion sort of at most four entries by full hash (ties keep their order).
    for i in 1..n {
        let mut j = i;
        while j > 0 && m.points[order[j - 1]].id > m.points[order[j]].id {
            order.swap(j - 1, j);
            j -= 1;
        }
    }
    let mut taken = [0u32; MAX_MANIFOLD_POINTS];
    let high = triangle.wrapping_shl(LOCAL_ID_BITS);
    for (k, &i) in order[..n].iter().enumerate() {
        let mut local = 1 + m.points[i].id % local_count;
        while taken[..k].contains(&local) {
            local = local % local_count + 1;
        }
        taken[k] = local;
        m.points[i].id = high | local;
    }
}

/// A feature of a polytope (indices into its faces, half-edges or vertices) or of a triangle
/// (face 0; edge `k` from vertex `k` to `k + 1`; vertex `k`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Feature {
    Face(u32),
    Edge(u32),
    Vertex(u32),
}

impl Feature {
    fn code(self) -> u32 {
        match self {
            Feature::Face(i) => (i << 2) | 1,
            Feature::Edge(i) => (i << 2) | 2,
            Feature::Vertex(i) => (i << 2) | 3,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Polytopes: hulls, boxes (on the stack) and triangles behind one half-edge view.

/// A polytope in its own frame: a hull's tables, or a box's or triangle's stack copies.
#[derive(Clone, Copy, Debug)]
struct Poly<'a> {
    vertices: &'a [Vec3],
    edges: &'a [HalfEdge],
    faces: &'a [HullFace],
    vertex_edge_offsets: &'a [u16],
    vertex_edges: &'a [u16],
    /// A point inside (on the plane, for a triangle).
    centroid: Vec3,
    /// About half the smallest thickness (0 for a triangle): scales tolerances.
    size: f32,
    /// Largest `|coord|` over all vertices (used for SAT rounding slack).
    max_abs_coord: f32,
}

impl<'a> Poly<'a> {
    #[inline]
    fn of_hull(h: &'a ConvexHull) -> Poly<'a> {
        Poly {
            vertices: &h.vertices,
            edges: &h.edges,
            faces: &h.faces,
            vertex_edge_offsets: &h.vertex_edge_offsets,
            vertex_edges: &h.vertex_edges,
            centroid: h.centroid,
            size: h.inner_radius,
            max_abs_coord: h.max_abs_coord,
        }
    }

    /// Outgoing half-edge indices from vertex `v`, in ascending half-edge index order.
    #[inline]
    fn vertex_outgoing(&self, v: usize) -> &'a [u16] {
        if v + 1 < self.vertex_edge_offsets.len() {
            let start = self.vertex_edge_offsets[v] as usize;
            let end = self.vertex_edge_offsets[v + 1] as usize;
            &self.vertex_edges[start..end]
        } else {
            &[]
        }
    }

    /// The index of the vertex farthest along `d` (the first of equals).
    #[inline]
    fn support(&self, d: Vec3) -> usize {
        let mut best = 0;
        let mut best_d = f32::NEG_INFINITY;
        for (i, v) in self.vertices.iter().enumerate() {
            let p = v.dot(d);
            if p > best_d {
                best_d = p;
                best = i;
            }
        }
        best
    }

    /// The half-edges round face `f`, counter-clockwise seen from outside.
    #[inline]
    fn face_loop(&self, f: usize) -> FaceLoop<'a> {
        let start = self.faces[f].edge;
        FaceLoop {
            edges: self.edges,
            start,
            next: start,
            left: self.edges.len(),
        }
    }

    /// The vertex at the end of half-edge `e`.
    #[inline]
    fn head(&self, e: usize) -> Vec3 {
        self.vertices[self.edges[self.edges[e].next as usize].origin as usize]
    }

    /// Whether `q` (on or near the plane of face `f`) lies within the face's side planes.
    fn face_contains(&self, f: usize, q: Vec3) -> bool {
        let n = self.faces[f].normal;
        self.face_loop(f).all(|e| {
            let p0 = self.vertices[self.edges[e].origin as usize];
            (self.head(e) - p0).cross(n).dot(q - p0) <= 0.0
        })
    }
}

/// Walks a face's half-edges, stopping after as many steps as there are half-edges.
struct FaceLoop<'a> {
    edges: &'a [HalfEdge],
    start: u32,
    next: u32,
    left: usize,
}

impl Iterator for FaceLoop<'_> {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.left == 0 {
            return None;
        }
        let e = self.next;
        self.next = u32::from(self.edges[e as usize].next);
        self.left = if self.next == self.start {
            0
        } else {
            self.left - 1
        };
        Some(e as usize)
    }
}

const fn half_edge(origin: u16, twin: u16, next: u16, face: u16) -> HalfEdge {
    HalfEdge {
        origin,
        twin,
        next,
        face,
    }
}

/// The half-edges of [`ConvexHull::cuboid`] (checked against it by a test), so a box collides
/// as that hull without building one.
const CUBOID_EDGES: [HalfEdge; 24] = [
    half_edge(0, 1, 2, 0),
    half_edge(4, 0, 16, 2),
    half_edge(4, 3, 4, 0),
    half_edge(6, 2, 19, 5),
    half_edge(6, 5, 6, 0),
    half_edge(2, 4, 20, 3),
    half_edge(2, 7, 0, 0),
    half_edge(0, 6, 23, 4),
    half_edge(1, 9, 10, 1),
    half_edge(3, 8, 17, 4),
    half_edge(3, 11, 12, 1),
    half_edge(7, 10, 22, 3),
    half_edge(7, 13, 14, 1),
    half_edge(5, 12, 21, 5),
    half_edge(5, 15, 8, 1),
    half_edge(1, 14, 18, 2),
    half_edge(0, 17, 15, 2),
    half_edge(1, 16, 7, 4),
    half_edge(5, 19, 1, 2),
    half_edge(4, 18, 13, 5),
    half_edge(6, 21, 11, 3),
    half_edge(7, 20, 3, 5),
    half_edge(3, 23, 5, 3),
    half_edge(2, 22, 9, 4),
];
/// The first half-edge of each cuboid face (−X, +X, −Y, +Y, −Z, +Z).
const CUBOID_FACE_EDGES: [u32; 6] = [0, 8, 16, 5, 7, 19];
const CUBOID_VERTEX_OFFSETS: [u16; 9] = [0, 3, 6, 9, 12, 15, 18, 21, 24];
const CUBOID_VERTEX_EDGES: [u16; 24] = [
    0, 7, 16, // v0
    8, 15, 17, // v1
    5, 6, 23, // v2
    9, 10, 22, // v3
    1, 2, 19, // v4
    13, 14, 18, // v5
    3, 4, 20, // v6
    11, 12, 21, // v7
];

/// A box as a polytope, built on the stack.
#[derive(Clone, Copy, Debug)]
struct BoxPoly {
    vertices: [Vec3; 8],
    faces: [HullFace; 6],
    size: f32,
    max_abs_coord: f32,
}

impl BoxPoly {
    fn new(h: Vec3) -> BoxPoly {
        let vertices = std::array::from_fn(|i| {
            Vec3::new(
                if i & 1 != 0 { h.x } else { -h.x },
                if i & 2 != 0 { h.y } else { -h.y },
                if i & 4 != 0 { h.z } else { -h.z },
            )
        });
        let normals = [-Vec3::X, Vec3::X, -Vec3::Y, Vec3::Y, -Vec3::Z, Vec3::Z];
        let offsets = [h.x, h.x, h.y, h.y, h.z, h.z];
        let faces = std::array::from_fn(|f| HullFace {
            normal: normals[f],
            offset: offsets[f],
            edge: CUBOID_FACE_EDGES[f],
        });
        BoxPoly {
            vertices,
            faces,
            size: h.min_element().max(0.0),
            max_abs_coord: h.abs().max_element(),
        }
    }
    #[inline]
    fn view(&self) -> Poly<'_> {
        Poly {
            vertices: &self.vertices,
            edges: &CUBOID_EDGES,
            faces: &self.faces,
            vertex_edge_offsets: &CUBOID_VERTEX_OFFSETS,
            vertex_edges: &CUBOID_VERTEX_EDGES,
            centroid: Vec3::ZERO,
            size: self.size,
            max_abs_coord: self.max_abs_coord,
        }
    }
}

/// A box or a hull, ready to be viewed as a polytope. The box variant is large on purpose: it
/// keeps the box's corners and faces on the stack instead of allocating.
#[allow(clippy::large_enum_variant)]
enum PolyShape<'a> {
    Box(BoxPoly),
    Hull(&'a ConvexHull),
}

impl PolyShape<'_> {
    fn of(s: &Shape) -> Option<PolyShape<'_>> {
        match s {
            Shape::Box { half_extents } => Some(PolyShape::Box(BoxPoly::new(*half_extents))),
            Shape::Hull(h) if !h.vertices.is_empty() && !h.faces.is_empty() => {
                Some(PolyShape::Hull(h))
            }
            _ => None,
        }
    }
    #[inline]
    fn view(&self) -> Poly<'_> {
        match self {
            PolyShape::Box(b) => b.view(),
            PolyShape::Hull(h) => Poly::of_hull(h),
        }
    }
}

/// A triangle as a flat polytope: the front loop is half-edges 0, 2, 4 (edge `k` is half-edge
/// `2k`, from vertex `k` to `k + 1`), the back loop their twins.
const TRIANGLE_EDGES: [HalfEdge; 6] = [
    half_edge(0, 1, 2, 0),
    half_edge(1, 0, 5, 1),
    half_edge(1, 3, 4, 0),
    half_edge(2, 2, 1, 1),
    half_edge(2, 5, 0, 0),
    half_edge(0, 4, 3, 1),
];
const TRIANGLE_VERTEX_OFFSETS: [u16; 4] = [0, 2, 4, 6];
const TRIANGLE_VERTEX_EDGES: [u16; 6] = [0, 5, 1, 2, 3, 4];

/// A mesh triangle in the colliding convex's frame.
#[derive(Clone, Copy, Debug)]
struct TriPoly {
    vertices: [Vec3; 3],
    /// Front (counter-clockwise) and back faces.
    faces: [HullFace; 2],
    normal: Vec3,
    active: u8,
    /// Per edge, the cosine between this face's normal and the neighbour's across a ridge (-1
    /// when no neighbour bounds the edge's normals; see [`TriangleRef::ridge_cos`]).
    ridge_cos: [f32; 3],
}

impl TriPoly {
    /// `None` for a triangle with no area.
    fn new(vertices: [Vec3; 3], active: u8, ridge_cos: [f32; 3]) -> Option<TriPoly> {
        let [a, b, c] = vertices;
        let normal = (b - a).cross(c - a).try_normalize()?;
        if !normal.is_finite() || !a.is_finite() || !b.is_finite() || !c.is_finite() {
            return None;
        }
        let offset = normal.dot(a);
        Some(TriPoly {
            vertices,
            faces: [
                HullFace {
                    normal,
                    offset,
                    edge: 0,
                },
                HullFace {
                    normal: -normal,
                    offset: -offset,
                    edge: 1,
                },
            ],
            normal,
            active,
            ridge_cos,
        })
    }

    fn view(&self) -> Poly<'_> {
        let [a, b, c] = self.vertices;
        Poly {
            vertices: &self.vertices,
            edges: &TRIANGLE_EDGES,
            faces: &self.faces,
            vertex_edge_offsets: &TRIANGLE_VERTEX_OFFSETS,
            vertex_edges: &TRIANGLE_VERTEX_EDGES,
            centroid: (a + b + c) * (1.0 / 3.0),
            size: 0.0,
            max_abs_coord: a.abs().max(b.abs()).max(c.abs()).max_element(),
        }
    }

    /// Signed distance of `p` above the front face.
    fn height(&self, p: Vec3) -> f32 {
        self.normal.dot(p) - self.faces[0].offset
    }

    /// Whether a normal from this feature may be used (see the mesh module's active edges); a
    /// vertex may when either of its edges may.
    fn allows(&self, f: Feature) -> bool {
        match f {
            Feature::Face(_) => true,
            Feature::Edge(k) => self.active & (1 << k) != 0,
            Feature::Vertex(k) => self.active & ((1 << k) | (1 << ((k + 2) % 3))) != 0,
        }
    }

    /// The direction out of the triangle across edge `k`, in its plane (not unit length).
    fn outward(&self, k: usize) -> Vec3 {
        let v = &self.vertices;
        (v[(k + 1) % 3] - v[k]).cross(self.normal)
    }

    /// Whether unit normal `d` leans across ridge `k` past the neighbour's normal. A ridge's
    /// normals turn about the edge from this face's normal towards the outward direction, as far
    /// as the neighbour's normal; one that turns further belongs to the neighbour's face.
    fn leans_past_ridge(&self, k: usize, d: Vec3) -> bool {
        let c = self.ridge_cos[k];
        if self.active & (1 << k) == 0 || c.is_nan() || c <= -1.0 {
            return false;
        }
        let out = self.outward(k).normalize_or_zero();
        let s = (1.0 - c * c).max(0.0).sqrt();
        // `out · c − normal · s` runs from the edge into the neighbour, in its plane.
        c * d.dot(out) - s * d.dot(self.normal) > RIDGE_PAST
    }

    /// Whether normal `d` leans out over inactive edge `k`, towards the coplanar or valley
    /// neighbour across it.
    fn leans_over_seam(&self, k: usize, d: Vec3) -> bool {
        self.active & (1 << k) == 0 && d.dot(self.outward(k)) > 0.0
    }

    /// Whether a normal `d` from feature `f` (allowed) is a neighbour's rather than this
    /// triangle's: past a ridge at `f`, or, from a vertex, out over a seam there (a vertex on a
    /// straight border or a flat run is no corner). The triangles across answer for it.
    fn belongs_to_neighbour(&self, f: Feature, d: Vec3) -> bool {
        match f {
            Feature::Face(_) => false,
            Feature::Edge(k) => self.leans_past_ridge(k as usize, d),
            Feature::Vertex(k) => [k as usize, (k as usize + 2) % 3]
                .into_iter()
                .any(|e| self.leans_past_ridge(e, d) || self.leans_over_seam(e, d)),
        }
    }

    /// The triangle's feature farthest along `d` (unit): the face when `d` is its normal, an
    /// edge when `d` is (within a small angle) perpendicular to it, a vertex otherwise.
    fn feature_along(&self, d: Vec3) -> Feature {
        if d.dot(self.normal) >= FACE_COS {
            return Feature::Face(0);
        }
        let v = &self.vertices;
        let s = [v[0].dot(d), v[1].dot(d), v[2].dot(d)];
        let i = if s[0] >= s[1] && s[0] >= s[2] {
            0
        } else if s[1] >= s[2] {
            1
        } else {
            2
        };
        let (j, h) = ((i + 1) % 3, (i + 2) % 3);
        let w = (d - self.normal * d.dot(self.normal)).length();
        if s[i] - s[j] <= EDGE_FEATURE_SIN * w * (v[i] - v[j]).length() {
            Feature::Edge(i as u32)
        } else if s[i] - s[h] <= EDGE_FEATURE_SIN * w * (v[i] - v[h]).length() {
            Feature::Edge(h as u32)
        } else {
            Feature::Vertex(i as u32)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Closest points.

/// Below this squared length a segment is a point.
const POINT_LEN2: f32 = 1e-30;

/// The point of segment `[a, b]` nearest to `p`, and its parameter.
fn closest_on_segment(p: Vec3, a: Vec3, b: Vec3) -> (f32, Vec3) {
    let ab = b - a;
    let l2 = ab.length_squared();
    if l2 <= POINT_LEN2 {
        return (0.0, a);
    }
    let t = ((p - a).dot(ab) / l2).clamp(0.0, 1.0);
    (t, a + ab * t)
}

/// Closest points of segments `[p1, q1]` and `[p2, q2]` (Ericson, "Real-Time Collision
/// Detection" 5.1.9): their parameters `s`, `t` and the points. Parallel segments give one of
/// their closest pairs.
fn closest_segments(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (f32, f32, Vec3, Vec3) {
    let (d1, d2, r) = (q1 - p1, q2 - p2, p1 - p2);
    let (a, e, f) = (d1.length_squared(), d2.length_squared(), d2.dot(r));
    let (s, t) = if a <= POINT_LEN2 && e <= POINT_LEN2 {
        (0.0, 0.0)
    } else if a <= POINT_LEN2 {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = d1.dot(r);
        if e <= POINT_LEN2 {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            let s = if denom > 1e-6 * a * e {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let t = (b * s + f) / e;
            if t < 0.0 {
                ((-c / a).clamp(0.0, 1.0), 0.0)
            } else if t > 1.0 {
                (((b - c) / a).clamp(0.0, 1.0), 1.0)
            } else {
                (s, t)
            }
        }
    };
    (s, t, p1 + d1 * s, p2 + d2 * t)
}

/// The point of triangle `v` nearest to `p` and the feature it lies on (Ericson 5.1.5).
fn closest_on_triangle(p: Vec3, v: &[Vec3; 3]) -> (Vec3, Feature) {
    let [a, b, c] = *v;
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, Feature::Vertex(0));
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (b, Feature::Vertex(1));
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let den = d1 - d3;
        let t = if den > 0.0 { d1 / den } else { 0.0 };
        return (a + ab * t, Feature::Edge(0));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (c, Feature::Vertex(2));
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let den = d2 - d6;
        let t = if den > 0.0 { d2 / den } else { 0.0 };
        return (a + ac * t, Feature::Edge(2));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let den = (d4 - d3) + (d5 - d6);
        let t = if den > 0.0 { (d4 - d3) / den } else { 0.0 };
        return (b + (c - b) * t, Feature::Edge(1));
    }
    let den = va + vb + vc;
    if den <= 0.0 || !den.is_finite() {
        return (a, Feature::Vertex(0));
    }
    (a + ab * (vb / den) + ac * (vc / den), Feature::Face(0))
}

/// Where a point is relative to a polytope.
#[derive(Clone, Copy, Debug)]
struct PolyPoint {
    /// The nearest surface point (inside: the point's projection on the face of least
    /// penetration).
    q: Vec3,
    /// Signed distance: negative inside.
    dist: f32,
    /// Unit normal from the polytope towards the point.
    normal: Vec3,
    feature: Feature,
}

/// The point of `poly` nearest to `c`, or for `c` inside, the face it is least deep behind;
/// `None` when `c` is certainly farther than `cutoff`. Exact: a point outside is nearest to the
/// face whose plane it is farthest above when its projection falls inside that face, and
/// otherwise to an edge next to a face it is above.
fn point_poly(poly: &Poly, c: Vec3, cutoff: f32) -> Option<PolyPoint> {
    let mut best = f32::NEG_INFINITY;
    let mut bf = 0;
    for (f, face) in poly.faces.iter().enumerate() {
        let s = face.normal.dot(c) - face.offset;
        if s > best {
            best = s;
            bf = f;
        }
    }
    if best > cutoff {
        return None;
    }
    let face = poly.faces[bf];
    let projected = c - face.normal * best;
    let on_face = PolyPoint {
        q: projected,
        dist: best,
        normal: face.normal,
        feature: Feature::Face(bf as u32),
    };
    if best <= 0.0 || poly.face_contains(bf, projected) {
        return Some(on_face);
    }
    let mut d2 = f32::INFINITY;
    let mut q = projected;
    let mut feature = on_face.feature;
    for e in (0..poly.edges.len()).step_by(2) {
        let (h, t) = (poly.edges[e], poly.edges[e + 1]);
        let (f1, f2) = (&poly.faces[h.face as usize], &poly.faces[t.face as usize]);
        if f1.normal.dot(c) <= f1.offset && f2.normal.dot(c) <= f2.offset {
            continue;
        }
        let (p0, p1) = (
            poly.vertices[h.origin as usize],
            poly.vertices[t.origin as usize],
        );
        let (s, x) = closest_on_segment(c, p0, p1);
        let dd = (c - x).length_squared();
        if dd < d2 {
            d2 = dd;
            q = x;
            feature = if s <= 0.0 {
                Feature::Vertex(u32::from(h.origin))
            } else if s >= 1.0 {
                Feature::Vertex(u32::from(t.origin))
            } else {
                Feature::Edge(e as u32)
            };
        }
    }
    if !d2.is_finite() {
        return Some(on_face);
    }
    let dist = d2.sqrt().max(best);
    let normal = (c - q).try_normalize().unwrap_or(face.normal);
    Some(PolyPoint {
        q,
        dist,
        normal,
        feature,
    })
}

/// Clips the segment `a + t (b - a)` to face `f`'s side planes: the parameter range left and,
/// for each end, 0 when it is the segment's own end or 1 + the half-edge whose side plane cut it.
fn clip_segment_to_face(poly: &Poly, f: usize, a: Vec3, b: Vec3) -> Option<(f32, u32, f32, u32)> {
    let n = poly.faces[f].normal;
    let (mut t0, mut t1, mut c0, mut c1) = (0.0f32, 1.0f32, 0u32, 0u32);
    for e in poly.face_loop(f) {
        let p0 = poly.vertices[poly.edges[e].origin as usize];
        let m = (poly.head(e) - p0).cross(n);
        let (da, db) = (m.dot(a - p0), m.dot(b - p0));
        if da > 0.0 && db > 0.0 {
            return None;
        }
        if da > 0.0 {
            let t = da / (da - db);
            if t > t0 {
                t0 = t;
                c0 = e as u32 + 1;
            }
        } else if db > 0.0 {
            let t = da / (da - db);
            if t < t1 {
                t1 = t;
                c1 = e as u32 + 1;
            }
        }
        if t0 > t1 {
            return None;
        }
    }
    Some((t0, c0, t1, c1))
}

/// The closest points of a segment and a triangle the segment does not cross.
#[derive(Clone, Copy, Debug)]
struct SegmentTriangle {
    /// On the segment.
    p: Vec3,
    /// On the triangle.
    q: Vec3,
    d2: f32,
    feature: Feature,
    /// Which part of the segment: 1 its start, 2 its end, 0 between.
    end: u32,
}

fn segment_triangle_closest(a: Vec3, b: Vec3, v: &[Vec3; 3]) -> SegmentTriangle {
    let (qa, fa) = closest_on_triangle(a, v);
    let mut best = SegmentTriangle {
        p: a,
        q: qa,
        d2: (a - qa).length_squared(),
        feature: fa,
        end: 1,
    };
    let (qb, fb) = closest_on_triangle(b, v);
    let d2 = (b - qb).length_squared();
    if d2 < best.d2 {
        best = SegmentTriangle {
            p: b,
            q: qb,
            d2,
            feature: fb,
            end: 2,
        };
    }
    for k in 0..3 {
        let (s, t, c1, c2) = closest_segments(a, b, v[k], v[(k + 1) % 3]);
        let d2 = (c1 - c2).length_squared();
        if d2 < best.d2 {
            best = SegmentTriangle {
                p: c1,
                q: c2,
                d2,
                feature: if t <= 0.0 {
                    Feature::Vertex(k as u32)
                } else if t >= 1.0 {
                    Feature::Vertex(((k + 1) % 3) as u32)
                } else {
                    Feature::Edge(k as u32)
                },
                end: segment_end(s),
            };
        }
    }
    best
}

/// 1 for a segment parameter at its start, 2 at its end, 0 between.
fn segment_end(s: f32) -> u32 {
    if s <= 0.0 {
        1
    } else if s >= 1.0 {
        2
    } else {
        0
    }
}

/// Two contacts for nearly parallel segments `[a0, a1]` (radius `ra`) and `[b0, b1]` (radius
/// `rb`) that overlap along their length: the ends of the overlap, separated along `n` (unit,
/// from A to B). Each is (midpoint, separation, which segment end bounds it).
fn parallel_points(
    a0: Vec3,
    a1: Vec3,
    ra: f32,
    b0: Vec3,
    b1: Vec3,
    rb: f32,
    n: Vec3,
) -> Option<[(Vec3, f32, u32); 2]> {
    let (ua, ub) = (a1 - a0, b1 - b0);
    let (la2, lb2) = (ua.length_squared(), ub.length_squared());
    if la2 <= POINT_LEN2 || lb2 <= POINT_LEN2 {
        return None;
    }
    if ua.cross(ub).length_squared() > PARALLEL_SIN * PARALLEL_SIN * la2 * lb2 {
        return None;
    }
    let (tb0, tb1) = ((b0 - a0).dot(ua) / la2, (b1 - a0).dot(ua) / la2);
    let (lo_b, hi_b, lo_code, hi_code) = if tb0 <= tb1 {
        (tb0, tb1, 3, 4)
    } else {
        (tb1, tb0, 4, 3)
    };
    let (lo, lo_code) = if lo_b > 0.0 {
        (lo_b, lo_code)
    } else {
        (0.0, 1)
    };
    let (hi, hi_code) = if hi_b < 1.0 {
        (hi_b, hi_code)
    } else {
        (1.0, 2)
    };
    // A tiny overlap is one contact, not two.
    if hi - lo <= 1e-4 {
        return None;
    }
    let point = |t: f32, code: u32| {
        let pa = a0 + ua * t;
        let (_, pb) = closest_on_segment(pa, b0, b1);
        let gap = n.dot(pb - pa);
        let sep = gap - ra - rb;
        (pa + n * (ra + 0.5 * sep), sep, code)
    };
    Some([point(lo, lo_code), point(hi, hi_code)])
}

// ---------------------------------------------------------------------------------------------
// Clipping, point reduction and manifold assembly.

/// A polygon vertex during clipping, then a contact candidate.
#[derive(Clone, Copy, Debug, Default)]
struct ClipVertex {
    /// The incident point while clipping; the contact midpoint afterwards.
    p: Vec3,
    sep: f32,
    /// The features that made this vertex (hashed).
    key: u32,
    /// The feature the polygon edge from this vertex to the next lies on.
    edge: u32,
}

/// Sutherland–Hodgman: clips the convex polygon `src` to the half-space `normal · x <= offset`
/// into `dst` (room for one more vertex than `src` keeps it whole) and returns the new length.
/// Points the plane cuts are named by the polygon edge they lie on and the plane's `side` tag.
fn clip_against(
    src: &[ClipVertex],
    dst: &mut [ClipVertex],
    normal: Vec3,
    offset: f32,
    side: u32,
) -> usize {
    let mut k = 0;
    let mut put = |v: ClipVertex, k: &mut usize| {
        if *k < dst.len() {
            dst[*k] = v;
            *k += 1;
        }
    };
    let Some(&last) = src.last() else {
        return 0;
    };
    let mut prev = last;
    let mut dp = normal.dot(prev.p) - offset;
    for &cur in src {
        let dc = normal.dot(cur.p) - offset;
        if (dp <= 0.0) != (dc <= 0.0) {
            let t = dp / (dp - dc);
            let p = prev.p + (cur.p - prev.p) * t;
            let key = mix(prev.edge, side);
            // Leaving: the polygon continues along the plane; entering: along the cut edge.
            let edge = if dp <= 0.0 { side } else { prev.edge };
            put(
                ClipVertex {
                    p,
                    sep: 0.0,
                    key,
                    edge,
                },
                &mut k,
            );
        }
        if dc <= 0.0 {
            put(cur, &mut k);
        }
        prev = cur;
        dp = dc;
    }
    k
}

/// [`clip_against`], except that a point cut within `reach` of the vertex the plane cuts off
/// takes that vertex's key, or else within `reach` of the vertex it keeps, that one's. A corner
/// lying on a side plane (a box on an aligned box) is then one feature whichever side of the
/// plane rounding or a micron of drift puts it, instead of a vertex one step and two cut points
/// the next.
fn clip_against_snapping(
    src: &[ClipVertex],
    dst: &mut [ClipVertex],
    normal: Vec3,
    offset: f32,
    side: u32,
    reach: f32,
) -> usize {
    let mut k = 0;
    let mut put = |v: ClipVertex, k: &mut usize| {
        if *k < dst.len() {
            dst[*k] = v;
            *k += 1;
        }
    };
    let Some(&last) = src.last() else {
        return 0;
    };
    let mut prev = last;
    let mut dp = normal.dot(prev.p) - offset;
    for &cur in src {
        let dc = normal.dot(cur.p) - offset;
        if (dp <= 0.0) != (dc <= 0.0) {
            let t = dp / (dp - dc);
            let p = prev.p + (cur.p - prev.p) * t;
            // Leaving: the polygon continues along the plane; entering: along the cut edge.
            let (edge, outside, inside) = if dp <= 0.0 {
                (side, cur, prev)
            } else {
                (prev.edge, prev, cur)
            };
            let key = if (p - outside.p).length_squared() <= reach * reach {
                outside.key
            } else if (p - inside.p).length_squared() <= reach * reach {
                inside.key
            } else {
                mix(prev.edge, side)
            };
            put(
                ClipVertex {
                    p,
                    sep: 0.0,
                    key,
                    edge,
                },
                &mut k,
            );
        }
        if dc <= 0.0 {
            put(cur, &mut k);
        }
        prev = cur;
        dp = dc;
    }
    k
}

/// Scores within this fraction of the candidates' extent of the best tie when a manifold is
/// reduced (see [`reduce`]).
const REDUCE_TIE: f32 = 1e-3;

/// Chooses at most four of `c` to keep (after Gregorius 2015): a first point, the one farthest
/// from it, the one spanning the largest triangle with those two, and the one outside that
/// triangle that adds the most area. `n` orients the areas. The first point is the candidate
/// farthest from the patch's centre, so the four span the patch, unless they would leave out a
/// point deeper than all four by more than [`REDUCE_TIE`] of the patch's width (at most
/// [`ABS_TOL`]); then it is the deepest. Repeated points add nothing and are dropped; of
/// collinear points both ends stay (and the deepest, when it is deeper than both). Returns
/// indices into `c` and how many, listed as [`reduce_strict`] lists them: the deepest first (the
/// solver takes them in order, and starting at the deepest steadies it); with at most four
/// candidates that is exactly its answer.
///
/// The choice is stable: a score within [`REDUCE_TIE`] of the patch's width of the best counts
/// as tied, and ties go to the smallest feature key, which stays the same while the same features
/// touch. As depth only counts past that window, neither a resting body's micron of rocking nor
/// a leaning stack's steady tilt swaps the points of a polygon of more than four corners (a box
/// turned on a box, a prism standing on a face) from step to step, which would lose every
/// carried impulse.
fn reduce(c: &[ClipVertex], n: Vec3) -> ([usize; MAX_MANIFOLD_POINTS], usize) {
    let (chosen, count) = reduce_stable(c, n, REDUCE_TIE);
    let mut sorted = chosen;
    sorted[..count].sort_unstable();
    let mut sub = [ClipVertex::default(); MAX_MANIFOLD_POINTS];
    for (s, &i) in sub.iter_mut().zip(&sorted[..count]) {
        *s = c[i];
    }
    let (order, listed) = reduce_strict(&sub[..count], n);
    if listed != count {
        return (chosen, count);
    }
    let mut out = [0; MAX_MANIFOLD_POINTS];
    for (o, &k) in out.iter_mut().zip(&order[..count]) {
        *o = sorted[k];
    }
    (out, count)
}

/// The points [`reduce`] keeps: the widest four, from the candidate farthest from the patch's
/// centre, unless that leaves out a point deeper than all four by more than the tie window;
/// then the widest four from the deepest point.
fn reduce_stable(
    c: &[ClipVertex],
    n: Vec3,
    geometric_tie: f32,
) -> ([usize; MAX_MANIFOLD_POINTS], usize) {
    if c.is_empty() {
        return ([0; MAX_MANIFOLD_POINTS], 0);
    }
    // The patch's centre and size from a long chord (two farthest-point steps): unlike a box
    // round the points, they do not depend on the frame (the shapes may come in either order),
    // and unlike the mean, they do not move when a corner is cut into two points.
    let farthest = |p: Vec3| {
        let mut best = 0;
        for (i, v) in c.iter().enumerate() {
            if (v.p - p).length_squared() > (c[best].p - p).length_squared() {
                best = i;
            }
        }
        c[best].p
    };
    let end = farthest(c[0].p);
    let other = farthest(end);
    let width = (other - end).length();
    let tol = geometric_tie * width;
    let window = (REDUCE_TIE * width).min(ABS_TOL);
    let centre = (end + other) * 0.5;
    let (start, _) = pick_stable(c, tol, false, |v| (v.p - centre).length());
    let (picks, count) = spread_from(c, n, tol, window, start);
    let deepest = c.iter().fold(f32::INFINITY, |m, v| m.min(v.sep));
    let kept = picks[..count]
        .iter()
        .fold(f32::INFINITY, |m, &i| m.min(c[i].sep));
    if kept <= deepest + window || kept.is_nan() {
        return (picks, count);
    }
    let (start, _) = pick_stable(c, window, false, |v| -v.sep);
    spread_from(c, n, tol, window, start)
}

/// The points of `c` kept with `i1`: the one farthest from it, the one spanning the largest
/// triangle with those two, and the one outside that triangle that adds the most area; of
/// collinear points both ends, and the deepest when it is clearly deeper than both.
fn spread_from(
    c: &[ClipVertex],
    n: Vec3,
    tol: f32,
    window: f32,
    i1: usize,
) -> ([usize; MAX_MANIFOLD_POINTS], usize) {
    let p1 = c[i1].p;
    let (i2, far) = pick_stable(c, tol, true, |v| (v.p - p1).length());
    if far <= 0.0 || far.is_nan() {
        return ([i1, 0, 0, 0], 1);
    }
    let p2 = c[i2].p;
    let e = p2 - p1;
    // Distances from the line through p1 and p2, across `n`.
    let (i3, height) = pick_stable(c, tol, true, |v| e.cross(v.p - p1).dot(n).abs() / far);
    if height <= 1e-6 * far || height.is_nan() {
        let (j, _) = pick_stable(c, tol, true, |v| (v.p - p2).length());
        let (d, _) = pick_stable(c, window, false, |v| -v.sep);
        let mut picks = [i2, j, 0, 0];
        let ends = c[i2].sep.min(c[j].sep);
        if d != i2 && d != j && c[d].sep < ends - 1e-6 * far {
            picks[2] = d;
            return (picks, 3);
        }
        return (picks, if j == i2 { 1 } else { 2 });
    }
    let p3 = c[i3].p;
    let sign = if e.cross(p3 - p1).dot(n) < 0.0 {
        -1.0
    } else {
        1.0
    };
    // How far outside the triangle (twice the area added, over |p2 - p1|).
    let (i4, added) = pick_stable(c, tol, true, |v| {
        let q = v.p;
        let a12 = (p2 - p1).cross(q - p1).dot(n) * sign;
        let a23 = (p3 - p2).cross(q - p2).dot(n) * sign;
        let a31 = (p1 - p3).cross(q - p3).dot(n) * sign;
        (-a12).max(-a23).max(-a31) / far
    });
    if added <= 1e-6 * far || added.is_nan() {
        return ([i1, i2, i3, 0], 3);
    }
    ([i1, i2, i3, i4], 4)
}

/// [`reduce`]'s choice by the strictly best scores, the first of equals: the strictly deepest
/// point first, then the farthest from it, and so on.
fn reduce_strict(c: &[ClipVertex], n: Vec3) -> ([usize; MAX_MANIFOLD_POINTS], usize) {
    if c.is_empty() {
        return ([0; MAX_MANIFOLD_POINTS], 0);
    }
    let mut i1 = 0;
    for (i, v) in c.iter().enumerate() {
        if v.sep < c[i1].sep {
            i1 = i;
        }
    }
    let farthest_from = |p: Vec3| {
        let (mut best, mut far) = (0, -1.0f32);
        for (i, v) in c.iter().enumerate() {
            let d = (v.p - p).length_squared();
            if d > far {
                far = d;
                best = i;
            }
        }
        (best, far)
    };
    let p1 = c[i1].p;
    let (i2, far) = farthest_from(p1);
    if far <= 0.0 {
        return ([i1, 0, 0, 0], 1);
    }
    let p2 = c[i2].p;
    let e = p2 - p1;
    let (mut i3, mut area, mut sign) = (i1, 0.0f32, 1.0f32);
    for (i, v) in c.iter().enumerate() {
        let a = e.cross(v.p - p1).dot(n);
        if a.abs() > area {
            area = a.abs();
            i3 = i;
            sign = if a < 0.0 { -1.0 } else { 1.0 };
        }
    }
    if area <= 1e-6 * far {
        // Collinear: both ends, and the deepest point when it is clearly deeper than both.
        let (j, _) = farthest_from(p2);
        let mut picks = [i2, j, 0, 0];
        let ends = c[i2].sep.min(c[j].sep);
        if i1 != i2 && i1 != j && c[i1].sep < ends - 1e-6 * far.sqrt() {
            picks[2] = i1;
            return (picks, 3);
        }
        return (picks, if j == i2 { 1 } else { 2 });
    }
    let p3 = c[i3].p;
    let (mut i4, mut added) = (i1, 0.0f32);
    for (i, v) in c.iter().enumerate() {
        let q = v.p;
        let a12 = (p2 - p1).cross(q - p1).dot(n) * sign;
        let a23 = (p3 - p2).cross(q - p2).dot(n) * sign;
        let a31 = (p1 - p3).cross(q - p3).dot(n) * sign;
        let out = (-a12).max(-a23).max(-a31);
        if out > added {
            added = out;
            i4 = i;
        }
    }
    if added <= 1e-6 * far {
        return ([i1, i2, i3, 0], 3);
    }
    ([i1, i2, i3, i4], 4)
}

/// The candidate of `c` with the smallest key among those whose score is within `tol` of the
/// best, and the best score; index 0 when no score is a number. With `apart`, the window also
/// stays under half the best, so a point already chosen (which scores 0) is not chosen again.
fn pick_stable(
    c: &[ClipVertex],
    tol: f32,
    apart: bool,
    score: impl Fn(&ClipVertex) -> f32,
) -> (usize, f32) {
    let best = c.iter().fold(f32::NEG_INFINITY, |m, v| m.max(score(v)));
    let window = if apart { tol.min(0.5 * best) } else { tol };
    let mut k = usize::MAX;
    for (i, v) in c.iter().enumerate() {
        if score(v) >= best - window && (k == usize::MAX || v.key < c[k].key) {
            k = i;
        }
    }
    (if k == usize::MAX { 0 } else { k }, best)
}

/// A manifold of the chosen candidates, each id the hash of `base` and the candidate's key.
fn build(normal: Vec3, c: &[ClipVertex], base: u32) -> Option<Manifold> {
    if c.is_empty() {
        return None;
    }
    let (idx, count) = reduce(c, normal);
    let mut m = Manifold {
        normal,
        ..Default::default()
    };
    for (k, &i) in idx[..count].iter().enumerate() {
        m.points[k] = ManifoldPoint {
            point: c[i].p,
            separation: c[i].sep,
            id: mix(base, c[i].key),
        };
    }
    m.count = count as u8;
    Some(m)
}

/// Clip buffers: small ones cover boxes and triangles; large ones any hull face (a face has at
/// most [`MAX_HULL_VERTICES`] vertices, and clipping adds at most one per side plane).
const SMALL_CLIP: usize = 16;
const LARGE_CLIP: usize = 2 * MAX_HULL_VERTICES + 2;

/// A face contact to compute: reference face `face` of `r` (placed by `rf` in the working
/// frame), incident face `fi` of `inc` (placed by `incf`).
struct FaceClip<'p, 'a> {
    r: &'p Poly<'a>,
    rf: &'p Frame,
    face: usize,
    inc: &'p Poly<'a>,
    incf: &'p Frame,
    fi: usize,
    /// The reference plane in the working frame: `n · x = off`, `n` out of `r`.
    n: Vec3,
    off: f32,
    spec: f32,
    base: u32,
    /// Only fall back to the deepest incident vertex when it lies over the reference face (for
    /// triangles, whose neighbours cover the rest; needs `rf` to be the identity).
    strict: bool,
    /// Name a point cut next to an incident vertex after that vertex (two polytopes).
    snap: bool,
}

/// Clips a face of `inc` against the side planes of reference face `face` of `r` and keeps up to
/// four points within `spec`; their separations are along the reference normal, which is the
/// manifold's normal (pointing out of `r`). The incident face is, of the faces round the vertex of
/// `inc` deepest along the normal, the one most opposed to it, so that vertex is always clipped.
/// When nothing survives clipping, the deepest incident vertex stands in. `kind` starts the ids.
#[allow(clippy::too_many_arguments)]
fn face_contact(
    r: &Poly,
    rf: &Frame,
    face: usize,
    inc: &Poly,
    incf: &Frame,
    spec: f32,
    kind: u32,
    strict: bool,
) -> Option<Manifold> {
    let rface = r.faces[face];
    let n = rf.vector(rface.normal);
    let off = rface.offset + n.dot(rf.pos);
    let ni = incf.inverse_vector(n);
    let fi = incident_face(inc, ni);
    let clip = FaceClip {
        r,
        rf,
        face,
        inc,
        incf,
        fi,
        n,
        off,
        spec,
        base: mix(mix(kind, face as u32), fi as u32),
        strict,
        snap: kind == KIND_FACE_A || kind == KIND_FACE_B,
    };
    let need = inc.face_loop(fi).count() + r.face_loop(face).count() + 1;
    if need <= SMALL_CLIP {
        let mut buf = [ClipVertex::default(); 2 * SMALL_CLIP];
        clip.run(&mut buf)
    } else {
        face_contact_large(&clip)
    }
}

/// The face of `inc` to clip against a reference face whose normal is `ni` in `inc`'s frame: of
/// the faces round the vertex deepest along `-ni`, the one most opposed to `ni` (the lowest face
/// index of equals). The most opposed face of all can miss that vertex: on an irregular hull a
/// vertex's region of normals may be thin, and a corner of a neighbouring region lie nearer to
/// `-ni`; the manifold would then report a gap where the shapes overlap.
fn incident_face(inc: &Poly, ni: Vec3) -> usize {
    let deepest = inc.support(-ni);
    let (mut fi, mut lowest) = (usize::MAX, f32::INFINITY);
    for &e in inc.vertex_outgoing(deepest) {
        let g = inc.edges[e as usize].face as usize;
        let d = inc.faces[g].normal.dot(ni);
        if d < lowest || (d == lowest && g < fi) {
            lowest = d;
            fi = g;
        }
    }
    if fi == usize::MAX {
        // No numbers, or no half-edge leaves the vertex: the most opposed face of all.
        fi = 0;
        for (g, f) in inc.faces.iter().enumerate() {
            let d = f.normal.dot(ni);
            if d < lowest {
                lowest = d;
                fi = g;
            }
        }
    }
    fi
}

/// The large-buffer path, apart so the common path's stack frame stays small.
#[inline(never)]
fn face_contact_large(clip: &FaceClip) -> Option<Manifold> {
    let mut buf = [ClipVertex::default(); 2 * LARGE_CLIP];
    clip.run(&mut buf)
}

impl FaceClip<'_, '_> {
    fn run(&self, buf: &mut [ClipVertex]) -> Option<Manifold> {
        let half = buf.len() / 2;
        let (mut a, mut b) = buf.split_at_mut(half);
        let mut len = 0;
        for e in self.inc.face_loop(self.fi) {
            if len == a.len() {
                break;
            }
            let o = self.inc.edges[e].origin;
            a[len] = ClipVertex {
                p: self.incf.point(self.inc.vertices[o as usize]),
                sep: 0.0,
                key: mix(KEY_VERTEX, u32::from(o)),
                edge: TAG_INCIDENT_EDGE | e as u32,
            };
            len += 1;
        }
        for s in self.r.face_loop(self.face) {
            let p0 = self
                .rf
                .point(self.r.vertices[self.r.edges[s].origin as usize]);
            let p1 = self.rf.point(self.r.head(s));
            let m = (p1 - p0).cross(self.n);
            let side = TAG_REFERENCE_SIDE | s as u32;
            len = if self.snap {
                let reach = (REDUCE_TIE * (p1 - p0).length()).min(ABS_TOL);
                clip_against_snapping(&a[..len], b, m, m.dot(p0), side, reach)
            } else {
                clip_against(&a[..len], b, m, m.dot(p0), side)
            };
            std::mem::swap(&mut a, &mut b);
            if len == 0 {
                break;
            }
        }
        let mut k = 0;
        for i in 0..len {
            let v = a[i];
            let sep = self.n.dot(v.p) - self.off;
            if sep <= self.spec {
                a[k] = ClipVertex {
                    p: v.p - self.n * (0.5 * sep),
                    sep,
                    key: v.key,
                    edge: 0,
                };
                k += 1;
            }
        }
        if k == 0 {
            let vi = self.inc.support(self.incf.inverse_vector(-self.n));
            let p = self.incf.point(self.inc.vertices[vi]);
            let sep = self.n.dot(p) - self.off;
            if sep > self.spec {
                return None;
            }
            if self.strict && !self.r.face_contains(self.face, p) {
                return None;
            }
            a[0] = ClipVertex {
                p: p - self.n * (0.5 * sep),
                sep,
                key: mix(KEY_DEEPEST, vi as u32),
                edge: 0,
            };
            k = 1;
        }
        build(self.n, &a[..k], self.base)
    }
}

// ---------------------------------------------------------------------------------------------
// Spheres and capsules: a point or a segment plus a radius.

/// One contact between a core point of A at `pa` (radius `ra`) and a surface point or core
/// point of B at `pb` (radius `rb`), `n` the unit normal from A to B.
fn single(n: Vec3, pa: Vec3, ra: f32, pb: Vec3, rb: f32, id: u32) -> Manifold {
    let sep = n.dot(pb - pa) - ra - rb;
    let mut m = Manifold {
        normal: n,
        ..Default::default()
    };
    m.points[0] = ManifoldPoint {
        point: (pa + n * ra + pb - n * rb) * 0.5,
        separation: sep,
        id,
    };
    m.count = 1;
    m
}

fn sphere_sphere(ca: Vec3, ra: f32, cb: Vec3, rb: f32, spec: f32) -> Option<Manifold> {
    let (n, d) = (cb - ca).normalize_and_length();
    if d - ra - rb > spec {
        return None;
    }
    // Coincident centres: any direction separates them equally well.
    let n = if d > 0.0 { n } else { Vec3::Y };
    Some(single(n, ca, ra, cb, rb, KIND_SPHERE))
}

fn sphere_capsule(c: Vec3, ra: f32, b0: Vec3, b1: Vec3, rb: f32, spec: f32) -> Option<Manifold> {
    let (s, x) = closest_on_segment(c, b0, b1);
    let (n, d) = (x - c).normalize_and_length();
    if d - ra - rb > spec {
        return None;
    }
    let n = if d > 0.0 {
        n
    } else {
        // The centre is on the core: push out sideways.
        (b1 - b0)
            .any_perpendicular()
            .try_normalize()
            .unwrap_or(Vec3::Y)
    };
    Some(single(
        n,
        c,
        ra,
        x,
        rb,
        mix(KIND_SPHERE_CAPSULE, segment_end(s)),
    ))
}

fn capsule_capsule(
    a0: Vec3,
    a1: Vec3,
    ra: f32,
    b0: Vec3,
    b1: Vec3,
    rb: f32,
    spec: f32,
) -> Option<Manifold> {
    let (s, t, c1, c2) = closest_segments(a0, a1, b0, b1);
    let (n, d) = (c2 - c1).normalize_and_length();
    if d - ra - rb > spec {
        return None;
    }
    let n = if d > 0.0 {
        n
    } else {
        // The cores cross: separate along their common perpendicular, towards B's centre.
        let towards = (b0 + b1 - a0 - a1) * 0.5;
        let (ua, ub) = (a1 - a0, b1 - b0);
        let across = ua.cross(ub).try_normalize().or_else(|| {
            let w = towards - ua * (towards.dot(ua) / ua.length_squared().max(POINT_LEN2));
            w.try_normalize()
        });
        let n = across
            .or_else(|| ua.any_perpendicular().try_normalize())
            .unwrap_or(Vec3::Y);
        if n.dot(towards) < 0.0 { -n } else { n }
    };
    if let Some(pair) = parallel_points(a0, a1, ra, b0, b1, rb, n) {
        let mut m = Manifold {
            normal: n,
            ..Default::default()
        };
        for (point, separation, code) in pair {
            if separation <= spec {
                m.points[usize::from(m.count)] = ManifoldPoint {
                    point,
                    separation,
                    id: mix(KIND_CAPSULES, 8 + code),
                };
                m.count += 1;
            }
        }
        if m.count > 0 {
            return Some(m);
        }
    }
    Some(single(
        n,
        c1,
        ra,
        c2,
        rb,
        mix(mix(KIND_CAPSULES, segment_end(s)), segment_end(t)),
    ))
}

/// Sphere A (centre `c`, world) against polytope B placed at `iso_p`.
fn sphere_poly(c: Vec3, r: f32, poly: &Poly, iso_p: &Iso, spec: f32) -> Option<Manifold> {
    let cl = iso_p.inverse_point(c);
    let hit = point_poly(poly, cl, r + spec)?;
    let sep = hit.dist - r;
    if sep > spec {
        return None;
    }
    // Local normal from the polytope to the sphere; the manifold's runs from the sphere.
    let n = hit.normal;
    let mut m = Manifold {
        normal: -n,
        ..Default::default()
    };
    m.points[0] = ManifoldPoint {
        point: (hit.q + cl - n * r) * 0.5,
        separation: sep,
        id: mix(KIND_SPHERE_POLY, hit.feature.code()),
    };
    m.count = 1;
    to_world(&mut m, iso_p);
    Some(m)
}

/// Capsule A (at `iso_c`) against polytope B placed at `iso_p`.
fn capsule_poly(
    half_height: f32,
    r: f32,
    iso_c: &Iso,
    poly: &Poly,
    iso_p: &Iso,
    spec: f32,
) -> Option<Manifold> {
    let rel = iso_p.inverse_mul(iso_c);
    let a = rel.point(Vec3::new(0.0, -half_height, 0.0));
    let b = rel.point(Vec3::new(0.0, half_height, 0.0));
    let mut m = segment_poly(a, b, r, poly, spec)?;
    m.normal = -m.normal;
    to_world(&mut m, iso_p);
    Some(m)
}

/// A segment `[a, b]` with radius `r` against a polytope, in the polytope's frame; the normal
/// points from the polytope to the segment.
fn segment_poly(a: Vec3, b: Vec3, r: f32, poly: &Poly, spec: f32) -> Option<Manifold> {
    // Reject on any face plane both ends are far above, and clip the segment to every face
    // plane at once (Cyrus–Beck) to see whether the core enters the polytope.
    let (mut t0, mut t1, mut inside) = (0.0f32, 1.0f32, true);
    for face in poly.faces {
        let (da, db) = (
            face.normal.dot(a) - face.offset,
            face.normal.dot(b) - face.offset,
        );
        if da.min(db) > r + spec {
            return None;
        }
        if da > 0.0 && db > 0.0 {
            inside = false;
        } else if da > 0.0 {
            t0 = t0.max(da / (da - db));
        } else if db > 0.0 {
            t1 = t1.min(da / (da - db));
        }
    }
    if !(inside && t0 <= t1) {
        if let Some(m) = segment_poly_outside(a, b, r, poly, spec) {
            return m;
        }
    }
    segment_poly_inside(a, b, r, poly, spec)
}

/// The core is outside: contacts from the exact closest points. `None` when the core turns out
/// to touch the polytope after all (then the caller treats it as inside).
fn segment_poly_outside(
    a: Vec3,
    b: Vec3,
    r: f32,
    poly: &Poly,
    spec: f32,
) -> Option<Option<Manifold>> {
    // Closest points: either end against the polytope, or the segment against an edge.
    let mut best: Option<(Vec3, Vec3, f32, Feature, u32)> = None;
    for (p, end) in [(a, 1), (b, 2)] {
        let h = point_poly(poly, p, f32::INFINITY)?;
        if h.dist <= 0.0 {
            return None;
        }
        if best.is_none_or(|bb| h.dist < bb.2) {
            best = Some((p, h.q, h.dist, h.feature, end));
        }
    }
    for e in (0..poly.edges.len()).step_by(2) {
        let (h, t) = (poly.edges[e], poly.edges[e + 1]);
        let (p0, p1) = (
            poly.vertices[h.origin as usize],
            poly.vertices[t.origin as usize],
        );
        let (s, u, c1, c2) = closest_segments(a, b, p0, p1);
        let d = (c1 - c2).length();
        if best.is_none_or(|bb| d < bb.2) {
            let feature = if u <= 0.0 {
                Feature::Vertex(u32::from(h.origin))
            } else if u >= 1.0 {
                Feature::Vertex(u32::from(t.origin))
            } else {
                Feature::Edge(e as u32)
            };
            best = Some((c1, c2, d, feature, segment_end(s)));
        }
    }
    let (p, q, d, feature, end) = best?;
    if d - r > spec {
        return Some(None);
    }
    let n = (p - q).try_normalize()?;
    // A face whose normal is (almost) the contact normal makes a face contact: clip the core to
    // that face so a capsule lying on it gets two points.
    let mut face = None;
    let mut consider = |f: usize| {
        let c = poly.faces[f].normal.dot(n);
        if c >= FACE_COS && face.is_none_or(|(_, bc)| c > bc) {
            face = Some((f, c));
        }
    };
    match feature {
        Feature::Face(f) => consider(f as usize),
        Feature::Edge(e) => {
            consider(poly.edges[e as usize].face as usize);
            consider(poly.edges[e as usize + 1].face as usize);
        }
        Feature::Vertex(v) => {
            for &e in poly.vertex_outgoing(v as usize) {
                consider(poly.edges[e as usize].face as usize);
            }
        }
    }
    if let Some((f, _)) = face
        && let Some(m) = segment_face_points(a, b, r, poly, f, spec)
    {
        return Some(Some(m));
    }
    if let Feature::Edge(e) = feature {
        let h = poly.edges[e as usize];
        let (p0, p1) = (
            poly.vertices[h.origin as usize],
            poly.vertices[poly.edges[e as usize + 1].origin as usize],
        );
        if let Some(m) = parallel_manifold(p0, p1, a, b, r, n, mix(KIND_CAPSULE_EDGE, e), spec) {
            return Some(Some(m));
        }
    }
    let m = single(
        n,
        q,
        0.0,
        p,
        r,
        mix(mix(KIND_CAPSULE_POINT, feature.code()), end),
    );
    Some((m.points[0].separation <= spec).then_some(m))
}

/// Up to two contacts of nearly parallel segments `[a0, a1]` (radius 0, on a polytope or
/// triangle) and `[b0, b1]` (radius `r`), normal `n` from A to B.
#[allow(clippy::too_many_arguments)]
fn parallel_manifold(
    a0: Vec3,
    a1: Vec3,
    b0: Vec3,
    b1: Vec3,
    r: f32,
    n: Vec3,
    base: u32,
    spec: f32,
) -> Option<Manifold> {
    let pair = parallel_points(a0, a1, 0.0, b0, b1, r, n)?;
    let mut m = Manifold {
        normal: n,
        ..Default::default()
    };
    for (point, separation, code) in pair {
        if separation <= spec {
            m.points[usize::from(m.count)] = ManifoldPoint {
                point,
                separation,
                id: mix(base, code),
            };
            m.count += 1;
        }
    }
    (m.count > 0).then_some(m)
}

/// Contacts of a segment (radius `r`) with face `f` of `poly`: the segment clipped to the face's
/// side planes, its ends separated along the face normal (the manifold's normal).
fn segment_face_points(
    a: Vec3,
    b: Vec3,
    r: f32,
    poly: &Poly,
    f: usize,
    spec: f32,
) -> Option<Manifold> {
    let face = poly.faces[f];
    let (t0, c0, t1, c1) = clip_segment_to_face(poly, f, a, b)?;
    let n = face.normal;
    let mut m = Manifold {
        normal: n,
        ..Default::default()
    };
    let ends: &[(f32, u32, u32)] = if t1 > t0 {
        &[(t0, c0, 0), (t1, c1, 1)]
    } else {
        &[(t0, c0, 0)]
    };
    for &(t, code, which) in ends {
        let p = a + (b - a) * t;
        let height = n.dot(p) - face.offset;
        let sep = height - r;
        if sep <= spec {
            m.points[usize::from(m.count)] = ManifoldPoint {
                point: p - n * (0.5 * (height + r)),
                separation: sep,
                id: mix(mix(mix(KIND_CAPSULE_FACE, f as u32), which), code),
            };
            m.count += 1;
        }
    }
    (m.count > 0).then_some(m)
}

/// The core reaches into the polytope: the axis of least penetration among the face normals and
/// the cross products of the segment with edges whose arcs it crosses on the Gauss map.
fn segment_poly_inside(a: Vec3, b: Vec3, r: f32, poly: &Poly, spec: f32) -> Option<Manifold> {
    let (mut s_face, mut f_best) = (f32::NEG_INFINITY, 0);
    for (f, face) in poly.faces.iter().enumerate() {
        let s = face.normal.dot(a).min(face.normal.dot(b)) - face.offset;
        if s > s_face {
            s_face = s;
            f_best = f;
        }
    }
    let u = b - a;
    let mut edge: Option<(f32, usize, Vec3)> = None;
    for e in (0..poly.edges.len()).step_by(2) {
        let (h, t) = (poly.edges[e], poly.edges[e + 1]);
        let (n1, n2) = (
            poly.faces[h.face as usize].normal,
            poly.faces[t.face as usize].normal,
        );
        // The segment's Gauss map is the great circle perpendicular to it; the edge's arc must
        // cross it.
        if n1.dot(u) * n2.dot(u) >= 0.0 {
            continue;
        }
        let p0 = poly.vertices[h.origin as usize];
        let ev = poly.vertices[t.origin as usize] - p0;
        let axis = u.cross(ev);
        let l2 = axis.length_squared();
        if l2 <= EDGE_SKIP_SIN * EDGE_SKIP_SIN * u.length_squared() * ev.length_squared() {
            continue;
        }
        let mut n = axis * (1.0 / l2.sqrt());
        if n.dot(p0 - poly.centroid) < 0.0 {
            n = -n;
        }
        let s = n.dot(a - p0);
        if edge.is_none_or(|(bs, _, _)| s > bs) {
            edge = Some((s, e, n));
        }
    }
    if let Some((s, e, n)) = edge
        && clearly_greater(s, s_face, EDGE_REL_TOL, abs_tolerance(poly.size))
    {
        if s - r > spec {
            return None;
        }
        let h = poly.edges[e];
        let (p0, p1) = (
            poly.vertices[h.origin as usize],
            poly.vertices[poly.edges[e + 1].origin as usize],
        );
        let (_, _, cs, ce) = closest_segments(a, b, p0, p1);
        // Along `n` the edge sits `s` below the segment wherever they are.
        let mut m = single(n, ce, 0.0, ce + n * s, r, mix(KIND_CAPSULE_EDGE, e as u32));
        m.points[0].point = (cs - n * r + ce) * 0.5;
        return Some(m);
    }
    if s_face - r > spec {
        return None;
    }
    if let Some(m) = segment_face_points(a, b, r, poly, f_best, spec) {
        return Some(m);
    }
    // The clipped core misses the face: use its deeper end alone.
    let face = poly.faces[f_best];
    let (p, end) = if face.normal.dot(a) <= face.normal.dot(b) {
        (a, 1)
    } else {
        (b, 2)
    };
    let height = face.normal.dot(p) - face.offset;
    let mut m = single(
        face.normal,
        p - face.normal * height,
        0.0,
        p,
        r,
        mix(mix(KIND_CAPSULE_FACE, f_best as u32), 4 + end),
    );
    m.points[0].separation = height - r;
    Some(m)
}

// ---------------------------------------------------------------------------------------------
// Polytope against polytope: the separating-axis test.

/// Whether arcs `a`–`b` and `c`–`d` (each shorter than a half circle) cross on the unit sphere:
/// the edges with those face normals then build a face of the Minkowski difference (Gregorius
/// 2013; pass the second polytope's normals negated). `b_x_a` and `d_x_c` are the cross
/// products, which only need the right direction.
fn minkowski_face(a: Vec3, b: Vec3, b_x_a: Vec3, c: Vec3, d: Vec3, d_x_c: Vec3) -> bool {
    let cba = c.dot(b_x_a);
    let dba = d.dot(b_x_a);
    let adc = a.dot(d_x_c);
    let bdc = b.dot(d_x_c);
    cba * dba < 0.0 && adc * bdc < 0.0 && cba * bdc > 0.0
}

/// An upper bound of `n · w` over the unit vectors `n` on the shorter arc between unit vectors
/// `a` and `b`: each such `n` is a point of the chord from `a` to `b` scaled by 1 to `2 / |a + b|`.
fn arc_max(a: Vec3, b: Vec3, w: Vec3) -> f32 {
    let m = a.dot(w).max(b.dot(w));
    if m <= 0.0 {
        return m;
    }
    let h = (a + b).length();
    if h > 1e-3 { m * (2.0 / h) } else { w.length() }
}

/// Whether unit `n` lies on the shorter arc between unit vectors `a` and `b`, within
/// [`ARC_SLACK`]: `(a × n) · (a × b)` and `(n × b) · (a × b)` are the sines of the angles from `a`
/// to `n` and from `n` to `b`, times `|a × b|`.
fn on_arc(n: Vec3, a: Vec3, b: Vec3) -> bool {
    let (ab, na, nb) = (a.dot(b), n.dot(a), n.dot(b));
    let slack = ARC_SLACK * (1.0 - ab * ab).max(0.0).sqrt();
    nb - ab * na >= -slack && na - ab * nb >= -slack
}

/// Polytopes A (the working frame) and B (placed in it by `rel`): the manifold in A's frame,
/// normal from A to B, or `None` when an axis separates them by more than `spec`.
///
/// The axes are tested in the order of an exhaustive test and give its answer, but an axis is
/// skipped when a bound of its separation shows it cannot change that answer. B reaches no
/// deeper below a plane than any point of B does, so B's deepest vertex below a likely face of A
/// and the bottom of B's inscribed ball bound the separation of every face of A (skipped below
/// the best face found) and, over the arc of an edge's normals, of every edge pair the edge is in
/// (skipped below what an edge needs to replace the best face); points of A bound B's faces and
/// edges likewise. A resting pair of 256-vertex hulls tests a few dozen axes instead of half a
/// million edge pairs.
#[derive(Clone, Copy)]
struct KeptEdgeA {
    edge: u16,
    a: Vec3,
    b: Vec3,
    b_x_a: Vec3,
    p0: Vec3,
    ea: Vec3,
    la2: f32,
}

impl KeptEdgeA {
    const ZERO: Self = Self {
        edge: 0,
        a: Vec3::ZERO,
        b: Vec3::ZERO,
        b_x_a: Vec3::ZERO,
        p0: Vec3::ZERO,
        ea: Vec3::ZERO,
        la2: 0.0,
    };
}

fn poly_poly(pa: &Poly, pb: &Poly, rel: &Frame, spec: f32) -> Option<Manifold> {
    // The bounds and the separations they bound round differently: skip only below this much.
    let slack = 1e-5 * (pa.max_abs_coord + pb.max_abs_coord + rel.pos.abs().max_element());
    // A's face normals: B's deepest vertex below each face. First the face B's centre lies
    // farthest in front of: its separation is a floor for the best.
    let cb = rel.point(pb.centroid);
    let (mut g, mut ahead) = (0, f32::NEG_INFINITY);
    for (f, face) in pa.faces.iter().enumerate() {
        let d = face.normal.dot(cb) - face.offset;
        if d > ahead {
            ahead = d;
            g = f;
        }
    }
    let deep_b = |f: usize| {
        let normal = pa.faces[f].normal;
        rel.point(pb.vertices[pb.support(rel.inverse_vector(-normal))])
    };
    let vb = deep_b(g);
    let floor = pa.faces[g].normal.dot(vb) - pa.faces[g].offset - slack;
    let (mut s_a, mut f_a) = (f32::NEG_INFINITY, 0);
    for (f, face) in pa.faces.iter().enumerate() {
        let bound = face.normal.dot(vb).min(face.normal.dot(cb) - pb.size) - face.offset;
        if bound < floor {
            continue;
        }
        let s = face.normal.dot(deep_b(f)) - face.offset;
        if s > spec {
            return None;
        }
        if s > s_a {
            s_a = s;
            f_a = f;
        }
    }
    // B's face normals, moved into A's frame; bounded in B's frame by A's centre and the vertex
    // of A's best face nearest B's deepest vertex. A face of B below A's best changes nothing.
    let mut ua = pa.vertices[pa.edges[pa.faces[f_a].edge as usize].origin as usize];
    for e in pa.face_loop(f_a) {
        let p = pa.vertices[pa.edges[e].origin as usize];
        if (p - vb).length_squared() < (ua - vb).length_squared() {
            ua = p;
        }
    }
    let ua = rel.inverse_vector(ua - rel.pos);
    let ca = rel.inverse_vector(pa.centroid - rel.pos);
    let floor = s_a - slack;
    let (mut s_b, mut f_b) = (f32::NEG_INFINITY, 0);
    for (f, face) in pb.faces.iter().enumerate() {
        let bound = face.normal.dot(ua).min(face.normal.dot(ca) - pa.size) - face.offset;
        if bound < floor {
            continue;
        }
        let n = rel.vector(face.normal);
        let v = pa.vertices[pa.support(-n)];
        let s = n.dot(v - rel.pos) - face.offset;
        if s > spec {
            return None;
        }
        if s > s_b {
            s_b = s;
            f_b = f;
        }
    }
    // Edge pairs whose arcs cross on the Gauss map. A pair matters only when it separates the
    // shapes or clearly beats the best face; each edge bounds every pair it is in, A's by points
    // of B, B's (in B's frame) by points of A.
    let tol = abs_tolerance(pa.size.min(pb.size));
    let best = s_a.max(s_b);
    let need = (best + EDGE_REL_TOL * best.abs() + tol).min(spec) - slack;
    const FAST_KEPT_A: usize = 32;
    let mut kept_fast = [KeptEdgeA::ZERO; FAST_KEPT_A];
    let mut kept = [0u16; 3 * MAX_HULL_VERTICES];
    let every_a = pa.edges.len() / 2 > kept.len();
    let mut count_a = 0;
    if every_a {
        count_a = pa.edges.len() / 2;
    } else {
        for i in (0..pa.edges.len()).step_by(2) {
            let (ha, ta) = (pa.edges[i], pa.edges[i + 1]);
            let a = pa.faces[ha.face as usize].normal;
            let b = pa.faces[ta.face as usize].normal;
            let p0 = pa.vertices[ha.origin as usize];
            let bound = arc_max(a, b, vb - p0).min(arc_max(a, b, cb - p0) - pb.size);
            if bound >= need || bound.is_nan() {
                if count_a < FAST_KEPT_A {
                    let ea = pa.vertices[ta.origin as usize] - p0;
                    kept_fast[count_a] = KeptEdgeA {
                        edge: i as u16,
                        a,
                        b,
                        b_x_a: b.cross(a),
                        p0,
                        ea,
                        la2: ea.length_squared(),
                    };
                }
                kept[count_a] = i as u16;
                count_a += 1;
            }
        }
    }
    let use_fast_a = !every_a && count_a <= FAST_KEPT_A;
    let mut edge: Option<(f32, usize, usize, Vec3)> = None;
    if count_a > 0 {
        for j in (0..pb.edges.len()).step_by(2) {
            let (hb, tb) = (pb.edges[j], pb.edges[j + 1]);
            let (cl, dl) = (
                pb.faces[hb.face as usize].normal,
                pb.faces[tb.face as usize].normal,
            );
            let q0l = pb.vertices[hb.origin as usize];
            let bound = arc_max(cl, dl, ua - q0l).min(arc_max(cl, dl, ca - q0l) - pa.size);
            if bound < need {
                continue;
            }
            let q0 = rel.point(q0l);
            let eb = rel.point(pb.vertices[tb.origin as usize]) - q0;
            let c = -rel.vector(cl);
            let d = -rel.vector(dl);
            let d_x_c = d.cross(c);
            let lb2 = eb.length_squared();
            if use_fast_a {
                for ka in &kept_fast[..count_a] {
                    if !minkowski_face(ka.a, ka.b, ka.b_x_a, c, d, d_x_c) {
                        continue;
                    }
                    let axis = ka.ea.cross(eb);
                    let l2 = axis.length_squared();
                    if l2 <= EDGE_SKIP_SIN * EDGE_SKIP_SIN * ka.la2 * lb2 {
                        continue;
                    }
                    let mut n = axis * (1.0 / l2.sqrt());
                    if n.dot(ka.p0 - pa.centroid) < 0.0 {
                        n = -n;
                    }
                    if !on_arc(n, ka.a, ka.b) || !on_arc(n, c, d) {
                        continue;
                    }
                    let s = n.dot(q0 - ka.p0);
                    if s > spec {
                        return None;
                    }
                    if edge.is_none_or(|(bs, ..)| s > bs) {
                        edge = Some((s, usize::from(ka.edge), j, n));
                    }
                }
            } else {
                let pairs =
                    (0..count_a).map(|k| if every_a { 2 * k } else { usize::from(kept[k]) });
                for i in pairs {
                    let (ha, ta) = (pa.edges[i], pa.edges[i + 1]);
                    let a = pa.faces[ha.face as usize].normal;
                    let b = pa.faces[ta.face as usize].normal;
                    if !minkowski_face(a, b, b.cross(a), c, d, d_x_c) {
                        continue;
                    }
                    let p0 = pa.vertices[ha.origin as usize];
                    let ea = pa.vertices[ta.origin as usize] - p0;
                    let axis = ea.cross(eb);
                    let l2 = axis.length_squared();
                    if l2 <= EDGE_SKIP_SIN * EDGE_SKIP_SIN * ea.length_squared() * lb2 {
                        continue;
                    }
                    let mut n = axis * (1.0 / l2.sqrt());
                    if n.dot(p0 - pa.centroid) < 0.0 {
                        n = -n;
                    }
                    // Rounding passes arcs that only touch (a face of each shape nearly parallel): the
                    // axis then lies off them and is no face of the Minkowski difference, and its
                    // separation means nothing. The face axes cover that direction.
                    if !on_arc(n, a, b) || !on_arc(n, c, d) {
                        continue;
                    }
                    let s = n.dot(q0 - p0);
                    if s > spec {
                        return None;
                    }
                    if edge.is_none_or(|(bs, ..)| s > bs) {
                        edge = Some((s, i, j, n));
                    }
                }
            }
        }
    }
    if let Some((s, i, j, n)) = edge
        && clearly_greater(s, best, EDGE_REL_TOL, tol)
    {
        let p0 = pa.vertices[pa.edges[i].origin as usize];
        let p1 = pa.vertices[pa.edges[i + 1].origin as usize];
        let q0 = rel.point(pb.vertices[pb.edges[j].origin as usize]);
        let q1 = rel.point(pb.vertices[pb.edges[j + 1].origin as usize]);
        let (_, _, c1, c2) = closest_segments(p0, p1, q0, q1);
        let mut m = Manifold {
            normal: n,
            ..Default::default()
        };
        m.points[0] = ManifoldPoint {
            point: (c1 + c2) * 0.5,
            separation: s,
            id: mix(mix(KIND_EDGES, i as u32), j as u32),
        };
        m.count = 1;
        return Some(m);
    }
    if clearly_greater(s_b, s_a, FACE_REL_TOL, tol) {
        let mut m = face_contact(pb, rel, f_b, pa, &Frame::IDENTITY, spec, KIND_FACE_B, false)?;
        m.normal = -m.normal;
        Some(m)
    } else {
        face_contact(pa, &Frame::IDENTITY, f_a, pb, rel, spec, KIND_FACE_A, false)
    }
}

// ---------------------------------------------------------------------------------------------
// Convex shapes against triangles (one-sided, with active edges), in the convex's frame. Every
// normal here points from the triangle to the convex.

/// A convex shape in its own frame.
enum Core<'a> {
    Sphere(f32),
    Capsule(f32, f32),
    Poly(Poly<'a>),
}

/// The contact of a convex with one triangle, and whether it is *borrowed*: its normal is a
/// neighbour's (it leans past a ridge, or out over a seam, at the triangle's feature), so it
/// counts only when no triangle touches with a normal of its own (see `collide_static`). The
/// convex's centre may be up to `behind` behind the triangle (which then also counts only
/// then).
fn convex_triangle(core: &Core, tri: &TriPoly, spec: f32, behind: f32) -> Option<(Manifold, bool)> {
    match core {
        Core::Sphere(r) => sphere_triangle(*r, tri, spec, behind),
        Core::Capsule(h, r) => capsule_triangle(
            Vec3::new(0.0, -h, 0.0),
            Vec3::new(0.0, *h, 0.0),
            *r,
            tri,
            spec,
            behind,
        ),
        Core::Poly(p) => poly_triangle(p, tri, spec, behind),
    }
}

/// A sphere of radius `r` at the origin.
fn sphere_triangle(r: f32, tri: &TriPoly, spec: f32, behind: f32) -> Option<(Manifold, bool)> {
    let height = tri.height(Vec3::ZERO);
    if height < -behind || height - r > spec {
        return None;
    }
    let (q, mut feature) = closest_on_triangle(Vec3::ZERO, &tri.vertices);
    let (dir, d) = (-q).normalize_and_length();
    if d - r > spec {
        return None;
    }
    let beside = feature != Feature::Face(0);
    // A centre behind the plane takes the face's normal: an edge's would point behind it.
    let along_face = d <= 0.0 || height < 0.0 || dir.dot(tri.normal) >= FACE_COS;
    let borrowed = !along_face && tri.allows(feature) && tri.belongs_to_neighbour(feature, dir);
    if along_face || !tri.allows(feature) {
        // Beside the triangle its plane can pass above the surface (across a convex crease, at
        // a ridge vertex whose edges here are seams): a sphere the triangle does not reach gets
        // nothing from it, and one it does is never deeper than its true distance.
        if beside && height - r < 0.0 && d - r > 0.0 {
            return None;
        }
        feature = Feature::Face(0);
    }
    let (normal, q, sep) = if feature == Feature::Face(0) {
        // The face, or an inactive edge or vertex: along the face normal.
        let sep = if beside {
            (height - r).max(d - r)
        } else {
            height - r
        };
        (tri.normal, -tri.normal * height, sep)
    } else {
        (dir, q, d - r)
    };
    if sep > spec {
        return None;
    }
    let mut m = single(
        normal,
        q,
        0.0,
        Vec3::ZERO,
        r,
        mix(KIND_SPHERE_POLY, feature.code()),
    );
    m.points[0].separation = sep;
    Some((m, borrowed))
}

/// A capsule core `[a, b]` with radius `r`.
fn capsule_triangle(
    a: Vec3,
    b: Vec3,
    r: f32,
    tri: &TriPoly,
    spec: f32,
    behind: f32,
) -> Option<(Manifold, bool)> {
    let own = |m: Manifold| (m, false);
    let (ha, hb) = (tri.height(a), tri.height(b));
    if ha + hb < -2.0 * behind || ha.min(hb) - r > spec {
        return None;
    }
    let view = tri.view();
    let n = tri.normal;
    // A core that passes through the triangle is pushed out along the face normal.
    if let Some((t0, _, t1, _)) = clip_segment_to_face(&view, 0, a, b) {
        let (h0, h1) = (ha + (hb - ha) * t0, ha + (hb - ha) * t1);
        if h0.min(h1) <= 0.0 {
            return segment_face_points(a, b, r, &view, 0, spec).map(own);
        }
    }
    let cl = segment_triangle_closest(a, b, &tri.vertices);
    let d = cl.d2.sqrt();
    if d - r > spec {
        return None;
    }
    let mut feature = cl.feature;
    let mut borrowed = false;
    let normal = match (cl.p - cl.q).try_normalize() {
        Some(dir) if dir.dot(n) < FACE_COS && tri.allows(feature) => {
            if tri.belongs_to_neighbour(feature, dir) {
                // A neighbour's normal: the core touches this triangle only where it passes over
                // its face; elsewhere the contact is borrowed.
                if let Some(m) = segment_face_points(a, b, r, &view, 0, spec) {
                    return Some(own(m));
                }
                borrowed = true;
            }
            dir
        }
        _ => {
            feature = Feature::Face(0);
            n
        }
    };
    if feature == Feature::Face(0) {
        if let Some(m) = segment_face_points(a, b, r, &view, 0, spec) {
            return Some(own(m));
        }
        // The core passes beside the triangle: its nearest point, separated along the normal,
        // never deeper than its true distance (see `sphere_triangle`).
        let height = tri.height(cl.p);
        if height - r > spec || (height - r < 0.0 && d - r > 0.0) {
            return None;
        }
        let id = mix(mix(KIND_CAPSULE_POINT, feature.code()), cl.end);
        let mut m = single(n, cl.p - n * height, 0.0, cl.p, r, id);
        m.points[0].separation = (height - r).max(d - r);
        return Some(own(m));
    }
    if normal.dot(n) < BACK_COS {
        return None;
    }
    if let Feature::Edge(k) = feature {
        let k = k as usize;
        let (p0, p1) = (tri.vertices[k], tri.vertices[(k + 1) % 3]);
        let base = mix(KIND_CAPSULE_EDGE, k as u32);
        if let Some(m) = parallel_manifold(p0, p1, a, b, r, normal, base, spec) {
            return Some((m, borrowed));
        }
    }
    let id = mix(mix(KIND_CAPSULE_POINT, feature.code()), cl.end);
    let m = single(normal, cl.q, 0.0, cl.p, r, id);
    (m.points[0].separation <= spec).then_some((m, borrowed))
}

/// The axis a polytope-triangle contact is built on.
#[derive(Clone, Copy, Debug)]
enum TriangleAxis {
    Triangle,
    PolyFace(usize),
    /// Separation, triangle edge, polytope half-edge, normal.
    Edges(f32, usize, usize, Vec3),
}

fn poly_triangle(poly: &Poly, tri: &TriPoly, spec: f32, behind: f32) -> Option<(Manifold, bool)> {
    let (n, v) = (tri.normal, tri.vertices);
    if tri.height(poly.centroid) < -behind {
        return None;
    }
    // The triangle's front face (its back face never separates a convex in front of it).
    let s_tri = tri.height(poly.vertices[poly.support(-n)]);
    if s_tri > spec {
        return None;
    }
    // The polytope's faces.
    let (mut s_face, mut f_best) = (f32::NEG_INFINITY, 0);
    for (f, face) in poly.faces.iter().enumerate() {
        let lo = face
            .normal
            .dot(v[0])
            .min(face.normal.dot(v[1]))
            .min(face.normal.dot(v[2]));
        let s = lo - face.offset;
        if s > spec {
            return None;
        }
        if s > s_face {
            s_face = s;
            f_best = f;
        }
    }
    // Edge pairs. A triangle edge's arc is a half circle from the front normal through the
    // edge's outward direction to the back normal; the polytope edge's (negated) arc must cross
    // its plane on the outward side.
    let mut edge: Option<(f32, usize, usize, Vec3)> = None;
    for k in 0..3 {
        let p = v[k];
        let t = v[(k + 1) % 3] - p;
        let outward = t.cross(n);
        for j in (0..poly.edges.len()).step_by(2) {
            let (h, tw) = (poly.edges[j], poly.edges[j + 1]);
            let c = -poly.faces[h.face as usize].normal;
            let d = -poly.faces[tw.face as usize].normal;
            let (cu, du) = (c.dot(t), d.dot(t));
            if cu * du >= 0.0 {
                continue;
            }
            let x = c * du.abs() + d * cu.abs();
            if x.dot(outward) <= 0.0 {
                continue;
            }
            let q0 = poly.vertices[h.origin as usize];
            let e = poly.vertices[tw.origin as usize] - q0;
            let axis = t.cross(e);
            let l2 = axis.length_squared();
            if l2 <= EDGE_SKIP_SIN * EDGE_SKIP_SIN * t.length_squared() * e.length_squared() {
                continue;
            }
            let mut nn = axis * (1.0 / l2.sqrt());
            if nn.dot(x) < 0.0 {
                nn = -nn;
            }
            let s = nn.dot(q0 - p);
            if s > spec {
                return None;
            }
            if edge.is_none_or(|(bs, ..)| s > bs) {
                edge = Some((s, k, j, nn));
            }
        }
    }
    // The triangle's face first (resting contact), then the polytope's, then edges.
    let tol = abs_tolerance(poly.size);
    let (mut axis, mut best) = (TriangleAxis::Triangle, s_tri);
    if clearly_greater(s_face, s_tri, FACE_REL_TOL, tol) {
        axis = TriangleAxis::PolyFace(f_best);
        best = s_face;
    }
    if let Some((s, k, j, nn)) = edge
        && clearly_greater(s, best, EDGE_REL_TOL, tol)
    {
        axis = TriangleAxis::Edges(s, k, j, nn);
    }
    let (normal, feature) = match axis {
        TriangleAxis::Triangle => (n, Feature::Face(0)),
        TriangleAxis::PolyFace(f) => {
            let d = -poly.faces[f].normal;
            (d, tri.feature_along(d))
        }
        TriangleAxis::Edges(_, k, _, nn) if nn.dot(n) < FACE_COS => (nn, Feature::Edge(k as u32)),
        TriangleAxis::Edges(..) => (n, Feature::Face(0)),
    };
    let edge_along_face = matches!(axis, TriangleAxis::Edges(..)) && feature == Feature::Face(0);
    // A normal past a ridge or out over a seam is the neighbour's: the contact is only borrowed.
    // (Its face instead would measure depths below this plane at points beside the triangle,
    // which can be far deeper than the polytope really is.)
    let borrowed = tri.allows(feature) && tri.belongs_to_neighbour(feature, normal);
    if edge_along_face || !tri.allows(feature) {
        // An inactive edge or vertex, or an edge axis that is the face normal anyway: the face
        // normal, with separations recomputed along it.
        axis = TriangleAxis::Triangle;
    } else if normal.dot(n) < BACK_COS {
        return None;
    }
    let m = match axis {
        TriangleAxis::Triangle => {
            if s_tri > spec {
                return None;
            }
            let view = tri.view();
            face_contact(
                &view,
                &Frame::IDENTITY,
                0,
                poly,
                &Frame::IDENTITY,
                spec,
                KIND_TRIANGLE_FACE,
                true,
            )
        }
        TriangleAxis::PolyFace(f) => {
            let view = tri.view();
            let mut m = face_contact(
                poly,
                &Frame::IDENTITY,
                f,
                &view,
                &Frame::IDENTITY,
                spec,
                KIND_TRIANGLE_POLY_FACE,
                false,
            )?;
            m.normal = -m.normal;
            Some(m)
        }
        TriangleAxis::Edges(s, k, j, nn) => {
            let q0 = poly.vertices[poly.edges[j].origin as usize];
            let q1 = poly.vertices[poly.edges[j + 1].origin as usize];
            let (_, _, c1, c2) = closest_segments(v[k], v[(k + 1) % 3], q0, q1);
            let mut m = Manifold {
                normal: nn,
                ..Default::default()
            };
            m.points[0] = ManifoldPoint {
                point: (c1 + c2) * 0.5,
                separation: s,
                id: mix(mix(KIND_EDGES, k as u32), j as u32),
            };
            m.count = 1;
            Some(m)
        }
    };
    m.map(|m| (m, borrowed))
}

// ---------------------------------------------------------------------------------------------
// Meshes and heightfields.

#[derive(Clone, Copy)]
enum StaticGeometry<'a> {
    Mesh(&'a TriangleMesh),
    Heightfield(&'a Heightfield),
}

impl StaticGeometry<'_> {
    fn of(s: &Shape) -> Option<StaticGeometry<'_>> {
        match s {
            Shape::Mesh(m) => Some(StaticGeometry::Mesh(m)),
            Shape::Heightfield(h) => Some(StaticGeometry::Heightfield(h)),
            _ => None,
        }
    }
}

/// Visits each nearby triangle in the convex shape's frame, until the visitor returns false.
fn visit_triangles(
    g: StaticGeometry,
    iso_g: &Iso,
    convex: &Shape,
    iso_c: &Iso,
    spec: f32,
    mut visit: impl FnMut(&Core, Vec3, &TriangleRef, [Vec3; 3]) -> bool,
) {
    let poly_shape;
    let core = match convex {
        Shape::Sphere { radius } => Core::Sphere(*radius),
        Shape::Capsule {
            half_height,
            radius,
        } => Core::Capsule(*half_height, *radius),
        _ => {
            let Some(p) = PolyShape::of(convex) else {
                return;
            };
            poly_shape = p;
            Core::Poly(poly_shape.view())
        }
    };
    // The convex's bounds in the geometry's frame find the triangles; each triangle then moves
    // into the convex's frame (three points instead of the whole convex).
    let bounds = convex.aabb(&iso_g.inverse_mul(iso_c)).expand(spec.max(0.0));
    let to_convex = Frame::new(&iso_c.inverse_mul(iso_g));
    let centre = match &core {
        Core::Poly(p) => p.centroid,
        _ => Vec3::ZERO,
    };
    let mut triangle =
        |t: &TriangleRef| visit(&core, centre, t, t.vertices.map(|p| to_convex.point(p)));
    match g {
        StaticGeometry::Mesh(m) => m.query_aabb(&bounds, &mut triangle),
        StaticGeometry::Heightfield(h) => h.query_aabb(&bounds, &mut triangle),
    }
}

/// Whether a convex shape crosses the other shape's surface. Mesh triangles face the query
/// shape from either side and a mere touch has zero separation.
pub(crate) fn overlaps(convex: &Shape, iso: &Iso, other: &Shape, at: &Iso) -> bool {
    let Some(g) = StaticGeometry::of(other) else {
        let mut buf: [Manifold; MAX_MESH_MANIFOLDS] = Default::default();
        let len = collide_stack(convex, iso, other, at, 0.0, &mut buf);
        return buf[..len]
            .iter()
            .any(|m| m.count > 0 && m.min_separation() < 0.0);
    };
    let mut hit = false;
    visit_triangles(g, at, convex, iso, 0.0, |core, centre, _, mut v| {
        if (v[1] - v[0]).cross(v[2] - v[0]).dot(centre - v[0]) < 0.0 {
            v.swap(1, 2);
        }
        // Each triangle is an independent surface for an overlap, including all its edges.
        if let Some(tri) = TriPoly::new(v, 0b111, [-1.0; 3])
            && let Some((m, _)) = convex_triangle(core, &tri, 0.0, 0.0)
        {
            hit = m.count > 0 && m.min_separation() < 0.0;
        }
        !hit
    });
    hit
}

/// Convex shape `convex` at `iso_c` against the triangles of `g` at `iso_g` near it. Normals run
/// from the geometry to the convex, flipped when the geometry is shape B.
fn collide_static(
    g: StaticGeometry,
    iso_g: &Iso,
    convex: &Shape,
    iso_c: &Iso,
    spec: f32,
    geometry_is_b: bool,
    out: &mut [Manifold; MAX_MESH_MANIFOLDS],
) -> usize {
    // Contacts of triangles the centre is in front of, with normals of their own; and the rest,
    // which count only when there are none of those (see below).
    let mut contacts = MeshContacts::default();
    let mut fallback = MeshContacts::default();
    let behind = spec.max(0.0);
    visit_triangles(g, iso_g, convex, iso_c, spec, |core, centre, t, v| {
        if let Some(tri) = TriPoly::new(v, t.active_edges, t.ridge_cos)
            && let Some((mut m, borrowed)) = convex_triangle(core, &tri, spec, behind)
        {
            assign_triangle_ids(&mut m, t.id);
            if borrowed || tri.height(centre) < 0.0 {
                fallback.add(m);
            } else {
                contacts.add(m);
            }
        }
        true
    });
    // Triangles are one-sided, and a triangle whose normal is a neighbour's leaves the contact
    // to that neighbour, which touches with a normal of its own. When no triangle does, the
    // convex either went through the surface (a thin body landing hard: its centre is just
    // behind it) or sank deep into a corner (where every triangle's own test points past its
    // neighbours); then the others push it out. While any triangle touches from the front they
    // do not count, so the underside of a thin closed mesh never pulls down what rests on top.
    if contacts.len == 0 {
        contacts = fallback;
    }
    contacts.finish(iso_c, geometry_is_b, out)
}

/// The manifolds of one convex against many triangles, merged by normal.
#[derive(Default)]
struct MeshContacts {
    list: [MeshGroup; MAX_MESH_MANIFOLDS],
    len: usize,
}

impl MeshContacts {
    fn add(&mut self, m: Manifold) {
        if m.count == 0 {
            return;
        }
        for g in &mut self.list[..self.len] {
            if g.first.normal.dot(m.normal) > MERGE_NORMAL_COS {
                g.merge(&m);
                return;
            }
        }
        if self.len < MAX_MESH_MANIFOLDS {
            self.list[self.len] = MeshGroup::new(m);
            self.len += 1;
            return;
        }
        // Full: the shallowest gives way to a deeper one.
        let mut worst = 0;
        for i in 1..self.len {
            if self.list[i].min_separation() >= self.list[worst].min_separation() {
                worst = i;
            }
        }
        if m.min_separation() < self.list[worst].min_separation() {
            self.list[worst] = MeshGroup::new(m);
        }
    }

    /// Deepest first, into the world.
    fn finish(self, iso: &Iso, flip: bool, out: &mut [Manifold; MAX_MESH_MANIFOLDS]) -> usize {
        let mut list: [Manifold; MAX_MESH_MANIFOLDS] = Default::default();
        for (m, g) in list.iter_mut().zip(&self.list[..self.len]) {
            *m = g.manifold();
        }
        let list = &mut list[..self.len];
        for i in 1..list.len() {
            let mut j = i;
            while j > 0 && list[j].min_separation() < list[j - 1].min_separation() {
                list.swap(j - 1, j);
                j -= 1;
            }
        }
        let mut written = 0;
        for m in list.iter_mut() {
            let mut m = std::mem::take(m);
            to_world(&mut m, iso);
            if flip {
                m.normal = -m.normal;
            }
            if let Some(m) = sanitize_finite(m) {
                out[written] = m;
                written += 1;
            }
        }
        written
    }
}

/// Positions and separations of merged mesh contacts that differ by less than this, relative to
/// one plus their size, count as equal: two triangles' copies of one point, or equally deep
/// points.
const PATCH_TOL: f32 = 1e-5;
/// A merged group's candidates: the deepest, then the farthest along each of eight directions.
const GROUP_SLOTS: usize = 9;

/// The contacts of one normal, merged from neighbouring triangles. Until a second triangle's
/// manifold merges, the first one as it came; from then on, the candidates the final four come
/// from: the deepest point, and the point farthest along each of eight directions in the
/// plane. The farthest point of a union along a direction is the farther of its parts', so no
/// order of arriving triangles loses a corner of the patch, as reducing to four at every merge
/// would (each corner of a box's face is the farthest along at least one of the directions,
/// which are 45° apart).
#[derive(Clone, Debug, Default)]
struct MeshGroup {
    first: Manifold,
    merged: bool,
    /// In the plane, unit: u, v, (u + v) / √2, (u − v) / √2.
    dirs: [Vec3; 4],
    /// The deepest, then the farthest along +dirs[0], −dirs[0], +dirs[1], …
    slots: [ManifoldPoint; GROUP_SLOTS],
}

impl MeshGroup {
    fn new(first: Manifold) -> MeshGroup {
        MeshGroup {
            first,
            ..Default::default()
        }
    }

    fn min_separation(&self) -> f32 {
        if self.merged {
            self.slots[0].separation
        } else {
            self.first.min_separation()
        }
    }

    fn merge(&mut self, m: &Manifold) {
        if !self.merged {
            self.merged = true;
            let n = self.first.normal;
            let u = n.any_perpendicular();
            let v = n.cross(u);
            let h = std::f32::consts::FRAC_1_SQRT_2;
            self.dirs = [u, v, (u + v) * h, (u - v) * h];
            self.slots = [self.first.points[0]; GROUP_SLOTS];
            for i in 1..usize::from(self.first.count) {
                self.offer(self.first.points[i]);
            }
        }
        for &p in m.points() {
            self.offer(p);
        }
    }

    /// Puts `p` in each slot it fills better than its holder by more than rounding, and in each
    /// slot held by a copy of the same point that is shallower by more than rounding (two
    /// triangles' copies of one point: every slot then holds the same one, and neither trades
    /// places with the other from step to step).
    fn offer(&mut self, p: ManifoldPoint) {
        let tol = PATCH_TOL * (1.0 + p.point.abs().max_element());
        let deeper_copy = |q: &ManifoldPoint| {
            (q.point - p.point).length_squared() <= tol * tol && p.separation < q.separation - tol
        };
        if p.separation < self.slots[0].separation - tol {
            self.slots[0] = p;
        }
        for (i, &d) in self.dirs.iter().enumerate() {
            let s = p.point.dot(d);
            let (most, least) = (1 + 2 * i, 2 + 2 * i);
            if s > self.slots[most].point.dot(d) + tol || deeper_copy(&self.slots[most]) {
                self.slots[most] = p;
            }
            if s < self.slots[least].point.dot(d) - tol || deeper_copy(&self.slots[least]) {
                self.slots[least] = p;
            }
        }
    }

    /// The group as one manifold of at most four points.
    fn manifold(&self) -> Manifold {
        if !self.merged {
            return self.first.clone();
        }
        // The distinct candidates (a point may hold several slots, and a seam's crossing may
        // come from both triangles): the deeper of a pair by more than rounding, else the first.
        let mut c = [ClipVertex::default(); GROUP_SLOTS];
        let mut k = 0;
        let scale = self
            .slots
            .iter()
            .fold(0.0f32, |s, p| s.max(p.point.abs().max_element()));
        let tol = PATCH_TOL * (1.0 + scale);
        for p in &self.slots {
            if let Some(j) = (0..k).find(|&j| (c[j].p - p.point).length_squared() <= tol * tol) {
                if p.separation < c[j].sep - tol {
                    c[j].p = p.point;
                    c[j].sep = p.separation;
                    c[j].key = p.id;
                }
                continue;
            }
            c[k] = ClipVertex {
                p: p.point,
                sep: p.separation,
                key: p.id,
                edge: 0,
            };
            k += 1;
        }
        // Keep the depth tolerance of a single convex face, but resolve geometry more
        // finely: a nearby grid seam crossing must not displace a true outer corner.
        let (mut idx, count) = reduce_stable(&c[..k], self.first.normal, PATCH_TOL);
        // Deepest first, as `reduce` lists its points (the solver takes them in order).
        for i in 1..count {
            let mut j = i;
            while j > 0 && c[idx[j]].sep < c[idx[j - 1]].sep {
                idx.swap(j - 1, j);
                j -= 1;
            }
        }
        let mut m = Manifold {
            normal: self.first.normal,
            ..Default::default()
        };
        for (slot, &i) in m.points.iter_mut().zip(&idx[..count]) {
            *slot = ManifoldPoint {
                point: c[i].p,
                separation: c[i].sep,
                id: c[i].key,
            };
        }
        m.count = count as u8;
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32) / (1u64 << 24) as f32
        }
        fn vec(&mut self, r: f32) -> Vec3 {
            Vec3::new(
                (self.next() * 2.0 - 1.0) * r,
                (self.next() * 2.0 - 1.0) * r,
                (self.next() * 2.0 - 1.0) * r,
            )
        }
    }

    #[test]
    fn the_box_tables_are_the_cuboid_hulls() {
        let h = Vec3::new(1.0, 2.0, 3.0);
        let hull = ConvexHull::cuboid(h);
        let b = BoxPoly::new(h);
        assert_eq!(hull.edges.as_slice(), &CUBOID_EDGES[..]);
        assert_eq!(
            hull.vertex_edge_offsets.as_slice(),
            &CUBOID_VERTEX_OFFSETS[..]
        );
        assert_eq!(hull.vertex_edges.as_slice(), &CUBOID_VERTEX_EDGES[..]);
        assert_eq!(hull.vertices.as_slice(), &b.vertices[..]);
        for (f, face) in hull.faces.iter().enumerate() {
            assert_eq!(face.edge, b.faces[f].edge);
            assert!((face.normal - b.faces[f].normal).length() < 1e-6);
            assert!((face.offset - b.faces[f].offset).abs() < 1e-6);
        }
        assert_eq!(b.view().size, 1.0);
        assert!((Poly::of_hull(&hull).size - 1.0).abs() < 1e-6);
    }

    #[test]
    fn the_triangle_tables_are_a_closed_two_sided_polygon() {
        for (i, e) in TRIANGLE_EDGES.iter().enumerate() {
            assert_eq!(e.twin as usize, i ^ 1);
            let twin = TRIANGLE_EDGES[i ^ 1];
            assert_eq!(TRIANGLE_EDGES[e.next as usize].origin, twin.origin);
            assert_eq!(TRIANGLE_EDGES[e.next as usize].face, e.face);
            assert_ne!(e.face, twin.face);
        }
        // Counter-clockwise seen from +Y.
        let t = TriPoly::new([Vec3::ZERO, Vec3::Z, Vec3::X], 0b101, [-1.0; 3]).unwrap();
        assert!((t.normal - Vec3::Y).length() < 1e-6);
        let view = t.view();
        assert_eq!(view.face_loop(0).collect::<Vec<_>>(), vec![0, 2, 4]);
        assert_eq!(view.face_loop(1).collect::<Vec<_>>(), vec![1, 5, 3]);
        assert!(view.face_contains(0, Vec3::new(0.2, 0.0, 0.2)));
        assert!(!view.face_contains(0, Vec3::new(0.6, 0.0, 0.6)));
        assert!(!view.face_contains(0, Vec3::new(-0.1, 0.0, 0.2)));
        assert!(TriPoly::new([Vec3::ZERO, Vec3::X, Vec3::X * 2.0], 0, [-1.0; 3]).is_none());
    }

    #[test]
    fn closest_points_of_segments_match_brute_force() {
        let mut rng = Rng(1);
        for i in 0..300 {
            let (p1, mut q1, p2, mut q2) = (rng.vec(1.0), rng.vec(1.0), rng.vec(1.0), rng.vec(1.0));
            match i % 5 {
                // Parallel, a point, both points.
                0 => q2 = p2 + (q1 - p1) * 0.7,
                1 => q1 = p1,
                2 => (q1, q2) = (p1, p2),
                _ => {}
            }
            let (s, t, c1, c2) = closest_segments(p1, q1, p2, q2);
            assert!((0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&t));
            assert!((c1 - p1.lerp(q1, s)).length() < 1e-5);
            assert!((c2 - p2.lerp(q2, t)).length() < 1e-5);
            let mut best = f32::INFINITY;
            for a in 0..=100 {
                for b in 0..=100 {
                    let d = p1.lerp(q1, a as f32 / 100.0) - p2.lerp(q2, b as f32 / 100.0);
                    best = best.min(d.length());
                }
            }
            let d = (c1 - c2).length();
            assert!(d <= best + 1e-5 && d >= best - 0.03, "{i}: {d} vs {best}");
        }
    }

    #[test]
    fn closest_points_of_triangles_match_brute_force_and_name_their_feature() {
        let mut rng = Rng(2);
        for i in 0..300 {
            let v = [rng.vec(1.0), rng.vec(1.0), rng.vec(1.0)];
            let p = rng.vec(2.0);
            let (q, f) = closest_on_triangle(p, &v);
            let mut best = f32::INFINITY;
            let n = 120;
            for a in 0..=n {
                for b in 0..=(n - a) {
                    let (u, w) = (a as f32 / n as f32, b as f32 / n as f32);
                    let x = v[0] + (v[1] - v[0]) * u + (v[2] - v[0]) * w;
                    best = best.min((x - p).length());
                }
            }
            let d = (q - p).length();
            assert!(d <= best + 1e-5 && d >= best - 0.03, "{i}: {d} vs {best}");
            // The point lies on the feature named.
            let on_segment = |a: Vec3, b: Vec3| (closest_on_segment(q, a, b).1 - q).length() < 1e-4;
            match f {
                Feature::Vertex(k) => assert!((q - v[k as usize]).length() < 1e-5),
                Feature::Edge(k) => assert!(on_segment(v[k as usize], v[(k as usize + 1) % 3])),
                Feature::Face(_) => {}
            }
        }
    }

    /// The distance from `p` to a triangulated polytope surface, through the triangle routine.
    fn surface_distance(poly: &Poly, p: Vec3) -> f32 {
        let mut best = f32::INFINITY;
        for f in 0..poly.faces.len() {
            let ring: Vec<Vec3> = poly
                .face_loop(f)
                .map(|e| poly.vertices[poly.edges[e].origin as usize])
                .collect();
            for k in 1..ring.len() - 1 {
                let (q, _) = closest_on_triangle(p, &[ring[0], ring[k], ring[k + 1]]);
                best = best.min((q - p).length());
            }
        }
        best
    }

    #[test]
    fn point_to_polytope_distances_match_the_surface() {
        let hull = ConvexHull::cuboid(Vec3::new(0.5, 1.0, 0.25));
        let b = BoxPoly::new(Vec3::new(1.0, 0.3, 0.6));
        let mut rng = Rng(3);
        for (k, poly) in [Poly::of_hull(&hull), b.view()].iter().enumerate() {
            for i in 0..400 {
                let p = rng.vec(2.0);
                let hit = point_poly(poly, p, f32::INFINITY).unwrap();
                let d = surface_distance(poly, p);
                if hit.dist > 0.0 {
                    assert!((hit.dist - d).abs() < 1e-5, "{k} {i}: {hit:?} vs {d}");
                    assert!((hit.q + hit.normal * hit.dist - p).length() < 1e-4);
                } else {
                    // Inside: the depth behind the nearest face.
                    assert!((-hit.dist - d).abs() < 1e-5, "{k} {i}: {hit:?} vs {d}");
                }
                // A cutoff below the distance rejects; above it keeps.
                if hit.dist > 0.1 {
                    assert!(point_poly(poly, p, hit.dist * 0.5).is_none() || hit.dist < 0.2);
                    assert!(point_poly(poly, p, hit.dist + 0.01).is_some());
                }
            }
        }
    }

    fn cand(x: f32, z: f32, sep: f32) -> ClipVertex {
        ClipVertex {
            p: Vec3::new(x, 0.0, z),
            sep,
            key: 0,
            edge: 0,
        }
    }

    #[test]
    fn reduction_keeps_the_deepest_point_and_the_largest_area() {
        // An octagon with one corner deepest: that corner, then the widest quad.
        let pts: Vec<ClipVertex> = (0..8)
            .map(|i| {
                let a = i as f32 * std::f32::consts::FRAC_PI_4;
                cand(a.cos(), a.sin(), if i == 3 { -0.2 } else { -0.1 })
            })
            .collect();
        let (idx, n) = reduce(&pts, Vec3::Y);
        assert_eq!(n, 4);
        assert_eq!(idx[0], 3);
        assert_eq!(idx[1], 7, "the opposite corner is farthest");
        // The quad spans the octagon: its area is the largest of any four corners (2).
        let q: Vec<Vec3> = idx.iter().map(|&i| pts[i].p).collect();
        let area = 0.5
            * ((q[0] - q[2]).cross(q[1] - q[3]).length())
                .max((q[0] - q[1]).cross(q[2] - q[3]).length())
                .max((q[0] - q[3]).cross(q[1] - q[2]).length());
        assert!((area - 2.0).abs() < 1e-4, "{area}");
        // A square with a repeated corner keeps four distinct corners.
        let sq = [
            cand(0.0, 0.0, 0.0),
            cand(1.0, 0.0, 0.0),
            cand(1.0, 0.0, 0.0),
            cand(1.0, 1.0, 0.0),
            cand(0.0, 1.0, 0.0),
        ];
        let (idx, n) = reduce(&sq, Vec3::Y);
        assert_eq!(n, 4);
        let mut picked: Vec<usize> = idx[..n].to_vec();
        picked.sort_unstable();
        assert!(
            picked == vec![0, 1, 3, 4] || picked == vec![0, 2, 3, 4],
            "{picked:?}"
        );
        // Collinear points keep their ends, and a deeper middle point.
        let line = [
            cand(0.5, 0.0, 0.0),
            cand(0.0, 0.0, 0.0),
            cand(1.0, 0.0, 0.0),
            cand(0.2, 0.0, 0.0),
        ];
        let (idx, n) = reduce(&line, Vec3::Y);
        let mut picked: Vec<usize> = idx[..n].to_vec();
        picked.sort_unstable();
        assert_eq!(picked, vec![1, 2]);
        let line = [
            cand(0.5, 0.0, -0.1),
            cand(0.0, 0.0, 0.0),
            cand(1.0, 0.0, 0.0),
        ];
        let (_, n) = reduce(&line, Vec3::Y);
        assert_eq!(n, 3);
        // One point, or the same point twice.
        assert_eq!(reduce(&[cand(0.0, 0.0, 0.0); 2], Vec3::Y).1, 1);
        assert_eq!(reduce(&[], Vec3::Y).1, 0);
    }

    /// A candidate with a feature key.
    fn keyed(x: f32, z: f32, sep: f32, key: u32) -> ClipVertex {
        ClipVertex {
            p: Vec3::new(x, 0.0, z),
            sep,
            key,
            edge: 0,
        }
    }

    /// The picks as a sorted set of keys.
    fn picked_keys(c: &[ClipVertex]) -> Vec<u32> {
        let (idx, n) = reduce(c, Vec3::Y);
        let mut keys: Vec<u32> = idx[..n].iter().map(|&i| c[i].key).collect();
        keys.sort_unstable();
        keys
    }

    #[test]
    fn reduction_keeps_the_same_points_while_the_depths_wobble() {
        // A regular 16-gon 2 mm deep (a prism standing on a face), its depths wobbling by up to
        // 2 microns, in any order: always the same four corners, the widest square.
        let mut rng = Rng(12);
        let corner = |i: usize, wobble: f32| {
            let a = i as f32 * std::f32::consts::TAU / 16.0;
            keyed(
                0.5 * a.cos(),
                0.5 * a.sin(),
                -0.002 + wobble,
                mix(KEY_VERTEX, i as u32),
            )
        };
        let still: Vec<ClipVertex> = (0..16).map(|i| corner(i, 0.0)).collect();
        let keys = picked_keys(&still);
        assert_eq!(keys.len(), 4);
        for round in 0..200 {
            let mut c: Vec<ClipVertex> = (0..16)
                .map(|i| corner(i, (rng.next() - 0.5) * 4e-6))
                .collect();
            c.rotate_left(round % 16);
            assert_eq!(picked_keys(&c), keys, "round {round}");
        }
        let q: Vec<Vec3> = still
            .iter()
            .filter(|v| keys.contains(&v.key))
            .map(|v| v.p)
            .collect();
        let area = 0.5 * (q[0] - q[2]).cross(q[1] - q[3]).length();
        assert!(
            (area - 0.5).abs() < 1e-4,
            "a square of the circumradius: {area}"
        );
        // A tilt of a millimetre across the patch: the deepest corner leads.
        let tilted: Vec<ClipVertex> = (0..16)
            .map(|i| corner(i, 0.002 * (i as f32 * std::f32::consts::TAU / 16.0).cos()))
            .collect();
        let (idx, n) = reduce(&tilted, Vec3::Y);
        assert_eq!((n, idx[0]), (4, 8), "{idx:?}");
        // With at most four candidates, the strict rule's answer exactly.
        for _ in 0..200 {
            let c: Vec<ClipVertex> = (0..4)
                .map(|i| {
                    let a = (i as f32 + 0.8 * rng.next()) * std::f32::consts::FRAC_PI_2;
                    keyed(a.cos(), a.sin(), -0.01 * rng.next(), rng.0 as u32)
                })
                .collect();
            assert_eq!(reduce(&c, Vec3::Y), reduce_strict(&c, Vec3::Y));
        }
    }

    #[test]
    fn reduction_never_picks_a_point_twice() {
        // A near-triangle (the fourth point 0.35 mm outside the hypotenuse of a 1 m triangle)
        // and a 2 m x 1 mm sliver: the near-tie windows must not bring a chosen point back.
        let near = [
            keyed(0.0, 0.0, -0.001, 1),
            keyed(1.0, 0.0, -0.001, 2),
            keyed(0.0, 1.0, -0.001, 3),
            keyed(0.5, 0.5005, -0.001, 4),
        ];
        let sliver = [
            keyed(-1.0, 0.0, -0.001, 1),
            keyed(1.0, 0.0, -0.001, 2),
            keyed(1.0, 0.001, -0.001, 3),
            keyed(-1.0, 0.001, -0.001, 4),
        ];
        for c in [&near[..], &sliver[..]] {
            let (idx, n) = reduce(c, Vec3::Y);
            for a in 0..n {
                for b in a + 1..n {
                    assert_ne!(idx[a], idx[b], "{:?}", &idx[..n]);
                }
            }
        }
    }

    /// The separating-axis test over every face and every edge pair whose axis lies on both arcs:
    /// what [`poly_poly`] must answer, bit for bit.
    fn poly_poly_exhaustive(pa: &Poly, pb: &Poly, rel: &Frame, spec: f32) -> Option<Manifold> {
        let (mut s_a, mut f_a) = (f32::NEG_INFINITY, 0);
        for (f, face) in pa.faces.iter().enumerate() {
            let v = rel.point(pb.vertices[pb.support(rel.inverse_vector(-face.normal))]);
            let s = face.normal.dot(v) - face.offset;
            if s > spec {
                return None;
            }
            if s > s_a {
                s_a = s;
                f_a = f;
            }
        }
        let (mut s_b, mut f_b) = (f32::NEG_INFINITY, 0);
        for (f, face) in pb.faces.iter().enumerate() {
            let n = rel.vector(face.normal);
            let v = pa.vertices[pa.support(-n)];
            let s = n.dot(v - rel.pos) - face.offset;
            if s > spec {
                return None;
            }
            if s > s_b {
                s_b = s;
                f_b = f;
            }
        }
        let mut edge: Option<(f32, usize, usize, Vec3)> = None;
        for j in (0..pb.edges.len()).step_by(2) {
            let (hb, tb) = (pb.edges[j], pb.edges[j + 1]);
            let q0 = rel.point(pb.vertices[hb.origin as usize]);
            let eb = rel.point(pb.vertices[tb.origin as usize]) - q0;
            let c = -rel.vector(pb.faces[hb.face as usize].normal);
            let d = -rel.vector(pb.faces[tb.face as usize].normal);
            let d_x_c = d.cross(c);
            let lb2 = eb.length_squared();
            for i in (0..pa.edges.len()).step_by(2) {
                let (ha, ta) = (pa.edges[i], pa.edges[i + 1]);
                let a = pa.faces[ha.face as usize].normal;
                let b = pa.faces[ta.face as usize].normal;
                if !minkowski_face(a, b, b.cross(a), c, d, d_x_c) {
                    continue;
                }
                let p0 = pa.vertices[ha.origin as usize];
                let ea = pa.vertices[ta.origin as usize] - p0;
                let axis = ea.cross(eb);
                let l2 = axis.length_squared();
                if l2 <= EDGE_SKIP_SIN * EDGE_SKIP_SIN * ea.length_squared() * lb2 {
                    continue;
                }
                let mut n = axis * (1.0 / l2.sqrt());
                if n.dot(p0 - pa.centroid) < 0.0 {
                    n = -n;
                }
                if !on_arc(n, a, b) || !on_arc(n, c, d) {
                    continue;
                }
                let s = n.dot(q0 - p0);
                if s > spec {
                    return None;
                }
                if edge.is_none_or(|(bs, ..)| s > bs) {
                    edge = Some((s, i, j, n));
                }
            }
        }
        let tol = abs_tolerance(pa.size.min(pb.size));
        if let Some((s, i, j, n)) = edge
            && clearly_greater(s, s_a.max(s_b), EDGE_REL_TOL, tol)
        {
            let p0 = pa.vertices[pa.edges[i].origin as usize];
            let p1 = pa.vertices[pa.edges[i + 1].origin as usize];
            let q0 = rel.point(pb.vertices[pb.edges[j].origin as usize]);
            let q1 = rel.point(pb.vertices[pb.edges[j + 1].origin as usize]);
            let (_, _, c1, c2) = closest_segments(p0, p1, q0, q1);
            let mut m = Manifold {
                normal: n,
                ..Default::default()
            };
            m.points[0] = ManifoldPoint {
                point: (c1 + c2) * 0.5,
                separation: s,
                id: mix(mix(KIND_EDGES, i as u32), j as u32),
            };
            m.count = 1;
            return Some(m);
        }
        if clearly_greater(s_b, s_a, FACE_REL_TOL, tol) {
            let mut m = face_contact(pb, rel, f_b, pa, &Frame::IDENTITY, spec, KIND_FACE_B, false)?;
            m.normal = -m.normal;
            Some(m)
        } else {
            face_contact(pa, &Frame::IDENTITY, f_a, pb, rel, spec, KIND_FACE_A, false)
        }
    }

    #[test]
    fn the_bounded_separating_axis_test_answers_as_the_exhaustive_one() {
        let mut rng = Rng(31);
        let unit = |rng: &mut Rng| loop {
            let p = rng.vec(1.0);
            let l = p.length();
            if l > 0.1 && l <= 1.0 {
                return p * (1.0 / l);
            }
        };
        let turn = |rng: &mut Rng| {
            let axis = unit(rng);
            crate::math::Quat::from_axis_angle(
                axis,
                (rng.next() * 2.0 - 1.0) * std::f32::consts::PI,
            )
        };
        let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
        let ball = |n: usize, r: f32| -> Vec<Vec3> {
            (0..n)
                .map(|i| {
                    let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
                    let ring = (1.0 - y * y).sqrt();
                    let a = golden * i as f32;
                    Vec3::new(ring * a.cos(), y, ring * a.sin()) * r
                })
                .collect()
        };
        let prism = |n: usize, r: f32, h: f32| -> Vec<Vec3> {
            (0..2 * n)
                .map(|i| {
                    let a = (i / 2) as f32 * std::f32::consts::TAU / n as f32;
                    let y = if i % 2 == 0 { -h } else { h };
                    // A tenth of a millimetre of noise keeps the side faces from merging.
                    Vec3::new(r * a.cos(), y, r * a.sin()) + Vec3::splat(1e-4 * ((i % 3) as f32))
                })
                .collect()
        };
        let mut clouds: Vec<Vec<Vec3>> = vec![
            ball(32, 0.5),
            ball(64, 0.6),
            prism(6, 0.5, 0.4),
            prism(24, 0.5, 0.01),
            prism(40, 0.3, 0.8),
        ];
        for (n, stretch) in [
            (12, Vec3::ONE),
            (40, Vec3::new(1.5, 0.4, 0.8)),
            (60, Vec3::ONE),
        ] {
            let points = (0..n)
                .map(|_| unit(&mut rng).mul_elem(stretch) * (0.3 + 0.2 * rng.next()))
                .collect();
            clouds.push(points);
        }
        // A rock far from its own origin, a tiny one and a big one.
        let rock = clouds[clouds.len() - 1].clone();
        clouds.push(
            rock.iter()
                .map(|&p| p + Vec3::new(3.0, -2.0, 1.0))
                .collect(),
        );
        clouds.push(rock.iter().map(|&p| p * 0.02).collect());
        clouds.push(rock.iter().map(|&p| p * 8.0).collect());
        let hulls: Vec<ConvexHull> = clouds
            .iter()
            .map(|c| ConvexHull::from_points(c).unwrap())
            .collect();
        let boxes = [
            BoxPoly::new(Vec3::new(0.5, 0.5, 0.5)),
            BoxPoly::new(Vec3::new(2.0, 0.1, 1.0)),
        ];
        let polys: Vec<Poly> = hulls
            .iter()
            .map(Poly::of_hull)
            .chain(boxes.iter().map(|b| b.view()))
            .collect();
        let radius = |p: &Poly| {
            p.vertices
                .iter()
                .fold(0.0f32, |m, v| m.max((*v - p.centroid).length()))
        };
        let (mut none, mut edges, mut faces) = (0, 0, 0);
        for case in 0..3000 {
            let (ia, ib) = (
                case % polys.len(),
                (case / polys.len() + case) % polys.len(),
            );
            let (pa, pb) = (&polys[ia], &polys[ib]);
            let (ra, rb) = (radius(pa), radius(pb));
            let rot = turn(&mut rng);
            let gap = 0.01 * (ra + rb);
            let centre = if case % 2 == 0 {
                // Anywhere from deep inside to just apart.
                pa.centroid + unit(&mut rng) * ((ra + rb) * (0.3 + 0.75 * rng.next()))
            } else {
                // Resting on a face of A, turned to lie on a face of its own, nearly flat.
                let f = (rng.next() * pa.faces.len() as f32) as usize % pa.faces.len();
                let g = (rng.next() * pb.faces.len() as f32) as usize % pb.faces.len();
                let n = pa.faces[f].normal;
                let tilt = crate::math::Quat::from_axis_angle(unit(&mut rng), 1e-3 * rng.next());
                let lie = tilt * crate::math::Quat::from_rotation_arc(pb.faces[g].normal, -n);
                let low = pb.vertices.iter().fold(f32::INFINITY, |m, &v| {
                    m.min(n.dot(lie.rotate(v - pb.centroid)))
                });
                let face_point = pa.vertices[pa.edges[pa.faces[f].edge as usize].origin as usize];
                let along = face_point + (pa.centroid - face_point) * (0.5 * rng.next());
                let lateral = along - n * n.dot(along) + n * pa.faces[f].offset;
                let depth = gap * (2.0 * rng.next() - 1.0);
                // Place B so that its lowest point is `depth` beyond A's face plane.
                let c = lateral - n * (low - depth);
                let iso = Iso::new(c - lie.rotate(pb.centroid), lie);
                let rel = Frame::new(&iso);
                let spec = [0.0, gap, 5.0 * gap][case % 3];
                let fast = poly_poly(pa, pb, &rel, spec);
                assert_eq!(
                    fast,
                    poly_poly_exhaustive(pa, pb, &rel, spec),
                    "case {case}"
                );
                match &fast {
                    None => none += 1,
                    Some(m) if m.count == 1 => edges += 1,
                    Some(_) => faces += 1,
                }
                continue;
            };
            let iso = Iso::new(centre - rot.rotate(pb.centroid), rot);
            let rel = Frame::new(&iso);
            let spec = [0.0, gap, 5.0 * gap][case % 3];
            let fast = poly_poly(pa, pb, &rel, spec);
            assert_eq!(
                fast,
                poly_poly_exhaustive(pa, pb, &rel, spec),
                "case {case}"
            );
            match &fast {
                None => none += 1,
                Some(m) if m.count == 1 => edges += 1,
                Some(_) => faces += 1,
            }
        }
        assert!(
            none > 200 && edges > 100 && faces > 1000,
            "{none} {edges} {faces}"
        );
    }

    #[test]
    fn an_edge_axis_off_its_arcs_is_not_an_axis() {
        // A tiny rock lying on a big rock's face, its own face nearly parallel to it and two of
        // their edges nearly parallel (found by the test above): the arcs only touch where the
        // two faces' normals nearly meet, but rounding passes them as crossing. The cross product
        // of the edges lies 0.013 rad beyond both arcs, and its separation (-0.0237, less than
        // the true overlap of 0.0285) would have won as an edge axis.
        let v = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
        let a = v(0.12743019, -0.9876474, 0.09118202);
        let b = v(0.7237776, -0.68335044, -0.095802575);
        let c = v(0.12748928, -0.9876389, 0.09119189);
        let d = v(0.24174431, -0.968614, 0.05784881);
        let n = v(0.114761986, -0.9888876, 0.09450334);
        assert!(minkowski_face(a, b, b.cross(a), c, d, d.cross(c)));
        assert!(!on_arc(n, a, b) && !on_arc(n, c, d));
        // On the arcs, at their ends and just past them within the slack, it is.
        let mid = (a + b).normalize_or_zero();
        assert!(on_arc(mid, a, b) && on_arc(a, a, b) && on_arc(b, a, b));
        let past = (a * 1.0 + (a - mid) * 0.0004).normalize_or_zero();
        assert!(on_arc(past, a, b));
        let beyond = (a + (a - mid) * 0.01).normalize_or_zero();
        assert!(!on_arc(beyond, a, b));
    }

    #[test]
    fn triangle_ids_are_nonzero_distinct_and_carry_the_triangle() {
        let mut m = Manifold {
            count: 4,
            ..Default::default()
        };
        // Hashes that all fall on the same local number.
        for (i, p) in m.points.iter_mut().enumerate() {
            p.id = 5 + 127 * i as u32;
        }
        assign_triangle_ids(&mut m, 33_554_431);
        let mut locals = vec![];
        for p in m.points() {
            assert_eq!(p.id >> LOCAL_ID_BITS, 33_554_431);
            let local = p.id & 0x7f;
            assert!(local != 0 && !locals.contains(&local));
            locals.push(local);
        }
        // The same hashes in another order get the same ids.
        let mut r = Manifold {
            count: 4,
            ..Default::default()
        };
        for (i, p) in r.points.iter_mut().enumerate() {
            p.id = 5 + 127 * (3 - i) as u32;
        }
        assign_triangle_ids(&mut r, 33_554_431);
        for i in 0..4 {
            assert_eq!(m.points[i].id, r.points[3 - i].id);
        }
        // Triangle 0 still gives nonzero ids.
        let mut z = Manifold {
            count: 1,
            ..Default::default()
        };
        assign_triangle_ids(&mut z, 0);
        assert_ne!(z.points[0].id, 0);
    }

    #[test]
    fn triangle_features_follow_the_normal_and_the_active_bits() {
        // Counter-clockwise from +Y: edge 0 runs along +Z at x = 0, edge 2 along X at z = 0.
        let t = TriPoly::new([Vec3::ZERO, Vec3::Z, Vec3::X], 0b001, [-1.0; 3]).unwrap();
        assert_eq!(t.feature_along(Vec3::Y), Feature::Face(0));
        let out0 = Vec3::new(-1.0, 1.0, 0.0).normalize_or_zero();
        assert_eq!(t.feature_along(out0), Feature::Edge(0));
        let out2 = Vec3::new(0.0, 1.0, -1.0).normalize_or_zero();
        assert_eq!(t.feature_along(out2), Feature::Edge(2));
        let corner = Vec3::new(-1.0, 1.0, -1.0).normalize_or_zero();
        assert_eq!(t.feature_along(corner), Feature::Vertex(0));
        assert!(t.allows(Feature::Face(0)));
        assert!(t.allows(Feature::Edge(0)) && !t.allows(Feature::Edge(1)));
        assert!(!t.allows(Feature::Edge(2)));
        // A vertex is allowed when either of its edges is.
        assert!(t.allows(Feature::Vertex(0)) && t.allows(Feature::Vertex(1)));
        assert!(!t.allows(Feature::Vertex(2)));
    }

    #[test]
    fn a_ridge_bounds_its_normals_and_a_vertex_by_a_seam_is_no_corner() {
        // Edge 0 (along +Z at x = 0) is a ridge whose neighbour is turned 10° towards -X; edge 2
        // (along X at z = 0) is a seam; edge 1 a border.
        let c = 10f32.to_radians().cos();
        let t = TriPoly::new([Vec3::ZERO, Vec3::Z, Vec3::X], 0b011, [c, -1.0, -1.0]).unwrap();
        let towards_x = |deg: f32| {
            let a = deg.to_radians();
            Vec3::new(-a.sin(), a.cos(), 0.0)
        };
        let towards_z = |deg: f32| {
            let a = deg.to_radians();
            Vec3::new(0.0, a.cos(), -a.sin())
        };
        // The ridge's normals run from this face's (0°) to the neighbour's (10°).
        for (deg, past) in [
            (5.0, false),
            (10.0, false),
            (11.0, true),
            (45.0, true),
            (90.0, true),
        ] {
            assert_eq!(t.leans_past_ridge(0, towards_x(deg)), past, "{deg}°");
            assert_eq!(
                t.belongs_to_neighbour(Feature::Edge(0), towards_x(deg)),
                past
            );
        }
        // A border's (bit set, no neighbour) reach down to the face's plane.
        assert!(!t.leans_past_ridge(1, Vec3::new(1.0, 0.0, 1.0).normalize_or_zero()));
        // At vertex 0, between the ridge and the seam: leaning over the seam is the neighbour's.
        assert!(!t.belongs_to_neighbour(Feature::Vertex(0), towards_x(5.0)));
        assert!(t.belongs_to_neighbour(Feature::Vertex(0), towards_x(20.0)));
        assert!(t.belongs_to_neighbour(Feature::Vertex(0), towards_z(5.0)));
        // Leaning back over the triangle itself is no neighbour's.
        assert!(!t.belongs_to_neighbour(Feature::Vertex(0), towards_z(-5.0)));
        assert!(!t.belongs_to_neighbour(Feature::Face(0), towards_x(45.0)));
    }

    /// A manifold of points `(x, z, separation, id)` with normal +Y.
    fn patch(points: &[(f32, f32, f32, u32)]) -> Manifold {
        let mut m = Manifold {
            normal: Vec3::Y,
            ..Default::default()
        };
        for (slot, &(x, z, separation, id)) in m.points.iter_mut().zip(points) {
            *slot = ManifoldPoint {
                point: Vec3::new(x, 0.0, z),
                separation,
                id,
            };
        }
        m.count = points.len() as u8;
        m
    }

    #[test]
    fn merged_patches_keep_their_corners_whatever_the_order_and_rounding() {
        // A unit square lying flat, seen by four triangles: each reports one corner, the seam
        // crossings beside it and the mesh vertex in the middle, all as deep up to rounding.
        let quarters = [
            [
                (0.0, 0.0, 10),
                (0.5, 0.0, 11),
                (0.0, 0.5, 12),
                (0.5, 0.5, 13),
            ],
            [
                (1.0, 0.0, 20),
                (0.5, 0.0, 21),
                (1.0, 0.5, 22),
                (0.5, 0.5, 23),
            ],
            [
                (1.0, 1.0, 30),
                (1.0, 0.5, 31),
                (0.5, 1.0, 32),
                (0.5, 0.5, 33),
            ],
            [
                (0.0, 1.0, 40),
                (0.0, 0.5, 41),
                (0.5, 1.0, 42),
                (0.5, 0.5, 43),
            ],
        ];
        let mut rng = Rng(9);
        for order in 0..24usize {
            // Every order of the four (Lehmer code), with fresh rounding on every depth.
            let mut left: Vec<usize> = (0..4).collect();
            let mut code = order;
            let mut contacts = MeshContacts::default();
            for k in (1..=4).rev() {
                let q = quarters[left.remove(code % k)];
                code /= k;
                let points: Vec<(f32, f32, f32, u32)> = q
                    .iter()
                    .map(|&(x, z, id)| (x, z, -1e-4 + (rng.next() - 0.5) * 1e-7, id))
                    .collect();
                contacts.add(patch(&points));
            }
            let mut out: [Manifold; MAX_MESH_MANIFOLDS] = Default::default();
            let out_len = contacts.finish(&Iso::IDENTITY, false, &mut out);
            assert_eq!(out_len, 1);
            let mut ids: Vec<u32> = out[0].points().iter().map(|p| p.id).collect();
            ids.sort_unstable();
            assert_eq!(ids, vec![10, 20, 30, 40], "order {order}: {:?}", out[0]);
            // Deepest first.
            let seps: Vec<f32> = out[0].points().iter().map(|p| p.separation).collect();
            assert!(seps.windows(2).all(|w| w[0] <= w[1]), "{seps:?}");
        }
        // A point clearly deeper than the rest stays, whichever triangle reports it.
        let mut contacts = MeshContacts::default();
        contacts.add(patch(&[(0.0, 0.0, -1e-4, 1), (1.0, 0.0, -1e-4, 2)]));
        contacts.add(patch(&[(1.0, 1.0, -1e-4, 3), (0.5, 0.4, -3e-3, 4)]));
        contacts.add(patch(&[(0.0, 1.0, -1e-4, 5), (0.5, 0.0, -1e-4, 6)]));
        let mut out: [Manifold; MAX_MESH_MANIFOLDS] = Default::default();
        contacts.finish(&Iso::IDENTITY, false, &mut out);
        assert_eq!(out[0].points[0].id, 4, "{:?}", out[0]);
        assert_eq!(out[0].count, 4);
        // Two copies of one point: the deeper one's id, in every slot it holds.
        let mut contacts = MeshContacts::default();
        contacts.add(patch(&[(0.0, 0.0, -1e-4, 1), (1.0, 0.0, -1e-4, 2)]));
        contacts.add(patch(&[(1.0, 0.0, -2e-4, 7), (1.0, 1.0, -1e-4, 3)]));
        let mut out: [Manifold; MAX_MESH_MANIFOLDS] = Default::default();
        contacts.finish(&Iso::IDENTITY, false, &mut out);
        let ids: Vec<u32> = out[0].points().iter().map(|p| p.id).collect();
        assert!(ids.contains(&7) && !ids.contains(&2), "{:?}", out[0]);
    }

    #[test]
    fn a_patch_reduction_breaks_ties_by_id() {
        // A regular hexagon lying flat: every corner as far from the centre and as deep; the
        // same four are kept however rounding moves them.
        let mut rng = Rng(5);
        let mut kept = None;
        for _ in 0..50 {
            let c: Vec<ClipVertex> = (0..6)
                .map(|i| {
                    let a = i as f32 * std::f32::consts::FRAC_PI_3;
                    ClipVertex {
                        p: Vec3::new(a.cos(), 0.0, a.sin()) + rng.vec(1e-7),
                        sep: -1e-4 + (rng.next() - 0.5) * 1e-7,
                        key: 100 - i,
                        edge: 0,
                    }
                })
                .collect();
            let (idx, count) = reduce_stable(&c, Vec3::Y, PATCH_TOL);
            assert_eq!(count, 4);
            let mut keys: Vec<u32> = idx[..count].iter().map(|&i| c[i].key).collect();
            keys.sort_unstable();
            assert!(
                kept.as_ref().is_none_or(|k| *k == keys),
                "{keys:?} after {kept:?}"
            );
            kept = Some(keys);
        }
        // Collinear: both ends, and a clearly deeper middle.
        let line = [
            cand(0.5, 0.0, -0.1),
            cand(0.0, 0.0, 0.0),
            cand(1.0, 0.0, 0.0),
        ];
        let (idx, n) = reduce_stable(&line, Vec3::Y, PATCH_TOL);
        let mut picked: Vec<usize> = idx[..n].to_vec();
        picked.sort_unstable();
        assert_eq!(picked, vec![0, 1, 2]);
        assert_eq!(reduce_stable(&[], Vec3::Y, PATCH_TOL).1, 0);
        assert_eq!(
            reduce_stable(&[cand(1.0, 2.0, 0.0); 2], Vec3::Y, PATCH_TOL).1,
            1
        );
    }

    #[test]
    fn clipping_names_points_by_the_features_that_made_them() {
        let square: Vec<ClipVertex> = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            .iter()
            .enumerate()
            .map(|(i, &(x, z))| ClipVertex {
                p: Vec3::new(x, 0.0, z),
                sep: 0.0,
                key: mix(KEY_VERTEX, i as u32),
                edge: TAG_INCIDENT_EDGE | i as u32,
            })
            .collect();
        let mut out = [ClipVertex::default(); 5];
        // Keep x <= 0.5.
        let n = clip_against(&square, &mut out, Vec3::X, 0.5, TAG_REFERENCE_SIDE | 7);
        assert_eq!(n, 4);
        let xs: Vec<f32> = out[..n].iter().map(|v| v.p.x).collect();
        assert!(xs.iter().all(|&x| x <= 0.5 + 1e-6));
        // The cut points carry the cut edge and the plane; kept corners keep their keys.
        let cut_bottom = mix(TAG_INCIDENT_EDGE, TAG_REFERENCE_SIDE | 7);
        let cut_top = mix(TAG_INCIDENT_EDGE | 2, TAG_REFERENCE_SIDE | 7);
        let keys: Vec<u32> = out[..n].iter().map(|v| v.key).collect();
        assert!(
            keys.contains(&cut_bottom) && keys.contains(&cut_top),
            "{keys:?}"
        );
        assert!(keys.contains(&mix(KEY_VERTEX, 0)) && keys.contains(&mix(KEY_VERTEX, 3)));
        // Everything outside: nothing; everything inside: unchanged.
        assert_eq!(clip_against(&square, &mut out, Vec3::X, -1.0, 0), 0);
        assert_eq!(clip_against(&square, &mut out, Vec3::X, 2.0, 0), 4);
        assert_eq!(out[0].key, square[0].key);
    }

    #[test]
    fn the_minkowski_face_test_matches_arc_intersection() {
        // Point `x` on the shorter arc from `a` to `b`.
        let on_arc = |x: Vec3, a: Vec3, b: Vec3| {
            let n = a.cross(b);
            a.cross(x).dot(n) >= 0.0 && x.cross(b).dot(n) >= 0.0 && x.dot(a + b) > 0.0
        };
        let mut rng = Rng(4);
        let mut agree = 0;
        for _ in 0..3000 {
            let (a, b, c, d) = (
                rng.vec(1.0).normalize_or_zero(),
                rng.vec(1.0).normalize_or_zero(),
                rng.vec(1.0).normalize_or_zero(),
                rng.vec(1.0).normalize_or_zero(),
            );
            let Some(x) = a.cross(b).cross(c.cross(d)).try_normalize() else {
                continue;
            };
            let crosses =
                (on_arc(x, a, b) && on_arc(x, c, d)) || (on_arc(-x, a, b) && on_arc(-x, c, d));
            assert_eq!(minkowski_face(a, b, b.cross(a), c, d, d.cross(c)), crosses);
            agree += 1;
        }
        assert!(agree > 2900);
    }
}
