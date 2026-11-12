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

mod primitives;
mod static_mesh;
#[cfg(test)]
mod tests;

use primitives::*;
pub(crate) use static_mesh::overlaps;
use static_mesh::*;
