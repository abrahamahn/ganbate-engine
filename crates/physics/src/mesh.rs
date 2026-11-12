//! Static triangle geometry: triangle meshes (level geometry) and heightfields (terrain).
//!
//! Both answer the same two questions for the rest of the engine: which triangles touch a box
//! ([`TriangleMesh::query_aabb`], [`Heightfield::query_aabb`]) and where a ray first hits
//! ([`TriangleMesh::raycast`], [`Heightfield::raycast`]). Triangles are one-sided for collision:
//! their front face is the side their counter-clockwise winding faces.
//!
//! Each triangle carries *active edge* bits: bit `i` is set when the edge from vertex `i` to
//! vertex `(i + 1) % 3` may produce contacts whose normal is not the triangle's face normal. An
//! edge is inactive when the neighbouring triangle across it is coplanar or makes a concave
//! (valley) crease; collision against such an edge uses the face normal instead, so a body sliding
//! over a flat floor made of many triangles never catches on the seams between them. An active
//! edge between two triangles (a ridge) also keeps the cosine between their normals: its contact
//! normals lie between the two faces' normals, so a gentle ridge is never a wall.
//!
//! A mesh finds its triangles through a bounding-volume hierarchy built once, top-down, by binned
//! surface-area splits; a heightfield needs none, as its grid gives the cells under a box or along
//! a ray directly. The ray–triangle test ([`ray_triangle`]) is watertight: a ray through an edge
//! or a vertex that triangles share hits at least one of them, so no ray slips through a seam of
//! a closed surface. A ray reports the nearest hit and, between hits at the same distance, the
//! lower triangle id, so the answer never depends on the walk order. A ray exactly through a
//! shared edge may hit both triangles there, at distances a rounding apart; it reports the
//! nearer (the lower id on a tie).

use crate::math::{Aabb, Vec3, inverse_direction};
use crate::tree::WalkStack;

/// One triangle handed to a query callback, in the mesh's local frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleRef {
    /// Stable within its mesh or heightfield: a triangle index, or `2 · cell + half`.
    pub id: u32,
    pub vertices: [Vec3; 3],
    /// Active edge bits (see the module doc).
    pub active_edges: u8,
    /// Per edge: on a ridge (an active edge shared with one neighbour), the cosine between this
    /// triangle's normal and the neighbour's, which bounds the normals the edge may produce;
    /// -1 on every other edge (a border, a non-manifold or inconsistently wound edge, an
    /// inactive edge).
    pub ridge_cos: [f32; 3],
    normal: Vec3,
}

impl TriangleRef {
    /// The unit front-face normal (zero for a degenerate triangle).
    #[inline]
    pub fn normal(&self) -> Vec3 {
        self.normal
    }
}

/// Where a ray hit a triangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshHit {
    /// Distance along the (unit) ray direction.
    pub t: f32,
    pub triangle: u32,
    /// The hit triangle's front-face normal, local frame.
    pub normal: Vec3,
}

/// Most triangles one mesh may hold.
pub const MAX_MESH_TRIANGLES: usize = 4_000_000;

/// Most triangles in one leaf of a mesh's hierarchy.
const LEAF_TRIANGLES: usize = 4;
/// Candidate split planes per node, minus one: the binned surface-area build sorts triangle
/// centres into this many slices of the node's longest axis.
const SAH_BINS: usize = 16;
/// The sine of 0.1°: neighbours whose normals differ by less are coplanar.
const COPLANAR_SIN: f32 = 1.745_328_3e-3;

/// A node of a mesh's hierarchy.
#[derive(Clone, Copy, Debug, PartialEq)]
struct BvhNode {
    /// Bounds of every triangle below.
    aabb: Aabb,
    /// A leaf's first slot in `TriangleMesh::order`; an internal node's first child (the second
    /// child follows it).
    first: u32,
    /// A leaf's number of triangles (at least one); 0 for an internal node.
    count: u32,
}

#[derive(Clone, Debug)]
pub struct TriangleMesh {
    pub vertices: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    /// One byte per triangle (bits 0..=2).
    pub active_edges: Vec<u8>,
    /// One per triangle: [`TriangleRef::ridge_cos`].
    pub ridge_cos: Vec<[f32; 3]>,
    normals: Vec<Vec3>,
    aabb: Aabb,
    /// The hierarchy, root first; built from `vertices` and `triangles` by [`TriangleMesh::new`].
    nodes: Vec<BvhNode>,
    /// Triangle indices in leaf order: a leaf holds `order[first..first + count]`.
    order: Vec<u32>,
}

impl TriangleMesh {
    /// Validates indices and values, drops degenerate (zero-area) triangles, computes active
    /// edges and builds the acceleration structure.
    ///
    /// Two triangles are neighbours when they share an edge, with vertices matched by position
    /// (bit-identical coordinates, `-0` and `+0` alike) as well as by index, so a mesh exported
    /// with split vertices still finds its neighbours. Triangle indices (the ids queries report)
    /// follow `triangles` once degenerate ones are dropped.
    pub fn new(vertices: Vec<Vec3>, triangles: Vec<[u32; 3]>) -> Result<TriangleMesh, String> {
        if vertices.iter().any(|v| !v.is_finite()) {
            return Err("mesh vertex is not finite".into());
        }
        if triangles.len() > MAX_MESH_TRIANGLES {
            return Err(format!(
                "{} triangles exceeds {MAX_MESH_TRIANGLES}",
                triangles.len()
            ));
        }
        for t in &triangles {
            if t.iter().any(|&i| i as usize >= vertices.len()) {
                return Err(format!(
                    "triangle {t:?} indexes past {} vertices",
                    vertices.len()
                ));
            }
        }
        let triangles: Vec<[u32; 3]> = triangles
            .into_iter()
            .filter(|t| {
                let [a, b, c] = t.map(|i| vertices[i as usize]);
                (b - a).cross(c - a).length_squared() > 1e-20
            })
            .collect();
        if triangles.is_empty() {
            return Err("mesh has no triangles with area".into());
        }
        let aabb = triangles
            .iter()
            .flatten()
            .fold(Aabb::EMPTY, |b, &i| b.include(vertices[i as usize]));
        let normals: Vec<Vec3> = triangles
            .iter()
            .map(|t| face_normal(&t.map(|i| vertices[i as usize])))
            .collect();
        let (active_edges, ridge_cos) = mesh_active_edges(&vertices, &triangles, &normals);
        let (nodes, order) = build_hierarchy(&vertices, &triangles);
        Ok(TriangleMesh {
            vertices,
            triangles,
            active_edges,
            ridge_cos,
            normals,
            aabb,
            nodes,
            order,
        })
    }

    #[inline]
    pub fn local_aabb(&self) -> Aabb {
        self.aabb
    }

    pub fn triangle(&self, i: u32) -> TriangleRef {
        TriangleRef {
            id: i,
            vertices: self.corners(i),
            active_edges: self.active_edges[i as usize],
            ridge_cos: self.ridge_cos[i as usize],
            normal: self.normals[i as usize],
        }
    }

    #[inline]
    fn corners(&self, i: u32) -> [Vec3; 3] {
        self.triangles[i as usize].map(|v| self.vertices[v as usize])
    }

    #[inline]
    fn leaf(&self, node: &BvhNode) -> &[u32] {
        &self.order[node.first as usize..(node.first + node.count) as usize]
    }

    /// Calls `f` for every triangle whose bounds overlap `aabb` (local frame), in a deterministic
    /// order, until `f` returns `false`.
    pub fn query_aabb(&self, aabb: &Aabb, f: &mut dyn FnMut(&TriangleRef) -> bool) {
        if !self.nodes[0].aabb.overlaps(aabb) {
            return;
        }
        let mut stack = WalkStack::<u32, 32>::new(0);
        stack.push(0);
        while let Some(i) = stack.pop() {
            let node = &self.nodes[i as usize];
            if !node.aabb.overlaps(aabb) {
                continue;
            }
            if node.count == 0 {
                stack.push(node.first + 1);
                stack.push(node.first);
                continue;
            }
            for &t in self.leaf(node) {
                let vertices = self.corners(t);
                if Aabb::from_points(&vertices).overlaps(aabb) {
                    let tri = TriangleRef {
                        id: t,
                        vertices,
                        active_edges: self.active_edges[t as usize],
                        ridge_cos: self.ridge_cos[t as usize],
                        normal: self.normals[t as usize],
                    };
                    if !f(&tri) {
                        return;
                    }
                }
            }
        }
    }

    /// The first triangle a ray `origin + t·dir` (`dir` unit length, local frame) hits for `t` in
    /// `[0, max_t]`. Only front faces are hit unless `both_sides`.
    pub fn raycast(
        &self,
        origin: Vec3,
        dir: Vec3,
        max_t: f32,
        both_sides: bool,
    ) -> Option<MeshHit> {
        if max_t.is_nan() || max_t < 0.0 || !origin.is_finite() || !dir.is_finite() {
            return None;
        }
        let inv = inverse_direction(dir);
        let pad = cull_pad(&self.aabb, origin);
        let (t, _) = slab(&self.nodes[0].aabb, origin, inv, max_t, pad)?;
        let mut best: Option<MeshHit> = None;
        let mut limit = max_t;
        // Each entry carries where the ray enters the node's box, so a node beyond the nearest
        // hit so far is dropped without another slab test.
        let mut stack = WalkStack::<(u32, f32), 32>::new((0, 0.0));
        stack.push((0, t));
        while let Some((i, entry)) = stack.pop() {
            if entry > limit {
                continue;
            }
            let node = &self.nodes[i as usize];
            if node.count > 0 {
                for &id in self.leaf(node) {
                    let v = self.corners(id);
                    if let Some(t) = ray_triangle(origin, dir, &v, limit, both_sides)
                        && is_nearer(t, id, best)
                    {
                        best = Some(MeshHit {
                            t,
                            triangle: id,
                            normal: self.normals[id as usize],
                        });
                        limit = t;
                    }
                }
                continue;
            }
            let (a, b) = (node.first, node.first + 1);
            let ta = slab(&self.nodes[a as usize].aabb, origin, inv, limit, pad);
            let tb = slab(&self.nodes[b as usize].aabb, origin, inv, limit, pad);
            match (ta, tb) {
                (Some((ta, _)), Some((tb, _))) => {
                    // The nearer child goes on top so it is walked first.
                    if tb < ta {
                        stack.push((a, ta));
                        stack.push((b, tb));
                    } else {
                        stack.push((b, tb));
                        stack.push((a, ta));
                    }
                }
                (Some((ta, _)), None) => stack.push((a, ta)),
                (None, Some((tb, _))) => stack.push((b, tb)),
                (None, None) => {}
            }
        }
        best
    }
}

/// Builds a mesh's hierarchy top-down: each node splits at the cheapest of the planes between
/// [`SAH_BINS`] slices of its triangles' centres along their longest axis (surface-area
/// heuristic), until a leaf holds at most [`LEAF_TRIANGLES`]. Returns the nodes (root first, the
/// two children of a node side by side) and the triangle order the leaves index into.
fn build_hierarchy(vertices: &[Vec3], triangles: &[[u32; 3]]) -> (Vec<BvhNode>, Vec<u32>) {
    let boxes: Vec<Aabb> = triangles
        .iter()
        .map(|t| Aabb::from_points(&t.map(|i| vertices[i as usize])))
        .collect();
    let n = triangles.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    let blank = BvhNode {
        aabb: Aabb::EMPTY,
        first: 0,
        count: 0,
    };
    let mut nodes = Vec::with_capacity(n);
    nodes.push(blank);
    // (node, first item, end of items), built depth first with an explicit stack.
    let mut work = vec![(0usize, 0usize, n)];
    while let Some((node, start, end)) = work.pop() {
        let items = &mut order[start..end];
        let aabb = items
            .iter()
            .fold(Aabb::EMPTY, |b, &t| b.union(boxes[t as usize]));
        if items.len() <= LEAF_TRIANGLES {
            items.sort_unstable();
            nodes[node] = BvhNode {
                aabb,
                first: start as u32,
                count: items.len() as u32,
            };
            continue;
        }
        let mid = start + sah_split(items, &boxes);
        let first = nodes.len();
        nodes.push(blank);
        nodes.push(blank);
        nodes[node] = BvhNode {
            aabb,
            first: first as u32,
            count: 0,
        };
        work.push((first + 1, mid, end));
        work.push((first, start, mid));
    }
    nodes.shrink_to_fit();
    (nodes, order)
}

/// Reorders `items` (more than one) so the triangles on the near side of the cheapest split plane
/// come first, and returns how many they are (at least one, and not all).
fn sah_split(items: &mut [u32], boxes: &[Aabb]) -> usize {
    let center = |t: u32| boxes[t as usize].center();
    let half = items.len() / 2;
    let span = items.iter().fold(Aabb::EMPTY, |b, &t| b.include(center(t)));
    let extent = span.max - span.min;
    let axis = extent.max_axis();
    let (lo, width) = (span.min[axis], extent[axis]);
    if width.is_nan() || width <= 0.0 {
        // Every centre in one place: any split serves.
        return half;
    }
    let scale = SAH_BINS as f32 / width;
    // NaN casts to 0 and huge values saturate, so every triangle lands in some bin.
    let bin = |t: u32| (((center(t)[axis] - lo) * scale) as usize).min(SAH_BINS - 1);
    let mut counts = [0usize; SAH_BINS];
    let mut bounds = [Aabb::EMPTY; SAH_BINS];
    for &t in items.iter() {
        let k = bin(t);
        counts[k] += 1;
        bounds[k] = bounds[k].union(boxes[t as usize]);
    }
    // The cost of everything right of plane `k` (between bins `k - 1` and `k`).
    let mut right = [0.0f32; SAH_BINS];
    let (mut b, mut c) = (Aabb::EMPTY, 0usize);
    for k in (1..SAH_BINS).rev() {
        b = b.union(bounds[k]);
        c += counts[k];
        if c > 0 {
            right[k] = c as f32 * b.half_area();
        }
    }
    let (mut best_cost, mut best_plane) = (f32::INFINITY, 0);
    let (mut b, mut c) = (Aabb::EMPTY, 0usize);
    for k in 1..SAH_BINS {
        b = b.union(bounds[k - 1]);
        c += counts[k - 1];
        if c == 0 || c == items.len() {
            continue;
        }
        let cost = c as f32 * b.half_area() + right[k];
        if cost < best_cost {
            best_cost = cost;
            best_plane = k;
        }
    }
    if best_plane == 0 {
        // No plane separates anything (only with sizes beyond f32).
        return half;
    }
    let (mut i, mut j) = (0, items.len());
    while i < j {
        if bin(items[i]) < best_plane {
            i += 1;
        } else {
            j -= 1;
            items.swap(i, j);
        }
    }
    if i == 0 || i == items.len() { half } else { i }
}

/// The active-edge bits and ridge cosines of every triangle of a mesh. Shared edges are found by
/// sorting all edges on their welded vertex pairs (never by hashing), so the result is
/// deterministic.
fn mesh_active_edges(
    vertices: &[Vec3],
    triangles: &[[u32; 3]],
    normals: &[Vec3],
) -> (Vec<u8>, Vec<[f32; 3]>) {
    let weld = weld(vertices);
    // [lower welded vertex, higher welded vertex, 3 · triangle + edge]
    let mut edges: Vec<[u32; 3]> = Vec::with_capacity(triangles.len() * 3);
    for (t, tri) in triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (weld[tri[k] as usize], weld[tri[(k + 1) % 3] as usize]);
            edges.push([a.min(b), a.max(b), (3 * t + k) as u32]);
        }
    }
    edges.sort_unstable();
    let mut bits = vec![0u8; triangles.len()];
    let mut ridge_cos = vec![[-1.0f32; 3]; triangles.len()];
    for run in edges.chunk_by(|p, q| p[..2] == q[..2]) {
        let active = match run {
            [p, q] => {
                let (t1, k1) = ((p[2] / 3) as usize, (p[2] % 3) as usize);
                let (t2, k2) = ((q[2] / 3) as usize, (q[2] % 3) as usize);
                let (s1, s2) = (triangles[t1], triangles[t2]);
                if weld[s1[k1] as usize] == weld[s2[k2] as usize] {
                    // Both run the edge the same way: one of them is wound backwards, so
                    // neither side can be called convex or concave.
                    true
                } else {
                    let v1_a = vertices[s1[k1] as usize];
                    let v1_b = vertices[s1[(k1 + 1) % 3] as usize];
                    let (n1, n2) = (normals[t1], normals[t2]);
                    let ridge = crease_is_active(n1, n2, v1_a, v1_b);
                    if ridge {
                        let c = n1.dot(n2);
                        ridge_cos[t1][k1] = c;
                        ridge_cos[t2][k2] = c;
                    }
                    ridge
                }
            }
            // A border (one triangle) or a non-manifold edge (three or more).
            _ => true,
        };
        if active {
            for e in run {
                bits[(e[2] / 3) as usize] |= 1 << (e[2] % 3);
            }
        }
    }
    (bits, ridge_cos)
}

/// For every vertex, the lowest index of a vertex at exactly the same position (`-0` and `+0`
/// alike).
fn weld(vertices: &[Vec3]) -> Vec<u32> {
    // `+ 0.0` turns -0 into +0, so equal coordinates have equal bits.
    let bits = |v: f32| (v + 0.0).to_bits();
    let mut keys: Vec<[u32; 4]> = vertices
        .iter()
        .enumerate()
        .map(|(i, v)| [bits(v.x), bits(v.y), bits(v.z), i as u32])
        .collect();
    keys.sort_unstable();
    let mut canonical = vec![0u32; vertices.len()];
    for run in keys.chunk_by(|p, q| p[..3] == q[..3]) {
        for k in run {
            canonical[k[3] as usize] = run[0][3];
        }
    }
    canonical
}

/// Whether the edge from `a` to `b` of a triangle with unit normal `n1` is active, given the unit
/// normal `n2` of the triangle across it (which runs the edge from `b` to `a`). Inactive when the
/// two are coplanar (within 0.1°) or fold into a valley; active on a ridge, and when folded flat
/// back onto each other (a knife edge, where neither reading is safe). Swapping the two triangles
/// gives the same answer bit for bit, so both sides of an edge always agree.
fn crease_is_active(n1: Vec3, n2: Vec3, a: Vec3, b: Vec3) -> bool {
    let axis = n1.cross(n2);
    if n1.dot(n2) > 0.0 && axis.length_squared() <= COPLANAR_SIN * COPLANAR_SIN {
        return false;
    }
    // The turn from `n1` to `n2` about the edge: positive on a ridge, negative in a valley.
    let turn = axis.dot(b - a);
    turn >= 0.0 || turn.is_nan()
}

/// The unit normal of the side triangle `v` winds counter-clockwise about (zero when degenerate).
fn face_normal(v: &[Vec3; 3]) -> Vec3 {
    (v[1] - v[0]).cross(v[2] - v[0]).normalize_or_zero()
}

/// Whether a hit at `t` on triangle `id` beats `best`: nearer, or as near with a lower id.
fn is_nearer(t: f32, id: u32, best: Option<MeshHit>) -> bool {
    best.is_none_or(|b| t < b.t || (t == b.t && id < b.triangle))
}

/// How far to grow boxes when culling a ray against them: well past the rounding of the slab and
/// triangle tests at these coordinates, so culling never drops a triangle the triangle test
/// would hit.
fn cull_pad(bounds: &Aabb, origin: Vec3) -> f32 {
    let reach = bounds
        .min
        .abs()
        .max(bounds.max.abs())
        .max(origin.abs())
        .max_element();
    reach * 1e-5
}

/// The part `[t0, t1]` of `[0, max_t]` in which the ray `origin + t·dir` (`inv` from
/// [`inverse_direction`]) is inside `b` grown by `pad` on every side.
fn slab(b: &Aabb, origin: Vec3, inv: Vec3, max_t: f32, pad: f32) -> Option<(f32, f32)> {
    let mut t0 = 0.0f32;
    let mut t1 = max_t;
    for i in 0..3 {
        let (lo, hi) = (b.min[i] - pad, b.max[i] + pad);
        let (o, inv) = (origin[i], inv[i]);
        if inv.is_infinite() {
            // Parallel to this slab: inside it or never.
            if o < lo || o > hi {
                return None;
            }
            continue;
        }
        let (mut near, mut far) = ((lo - o) * inv, (hi - o) * inv);
        if near > far {
            std::mem::swap(&mut near, &mut far);
        }
        t0 = t0.max(near);
        t1 = t1.min(far);
        if t0 > t1 {
            return None;
        }
    }
    Some((t0, t1))
}

/// A regular grid of heights in the local XZ plane: sample `(i, j)` (column `i` along +X, row
/// `j` along +Z) sits at `(i · scale.x, heights[j · columns + i] · scale.y, j · scale.z)`.
/// Cell `(i, j)` (`i < columns − 1`, `j < rows − 1`) splits into two triangles along the diagonal
/// from sample `(i, j)` to `(i + 1, j + 1)`; triangle ids are `2 · (j · (columns − 1) + i) + half`.
#[derive(Clone, Debug)]
pub struct Heightfield {
    pub columns: u32,
    pub rows: u32,
    pub heights: Vec<f32>,
    pub scale: Vec3,
    aabb: Aabb,
}

/// Most samples one heightfield may hold.
pub const MAX_HEIGHTFIELD_SAMPLES: usize = 16_777_216;

impl Heightfield {
    pub fn new(
        columns: u32,
        rows: u32,
        heights: Vec<f32>,
        scale: Vec3,
    ) -> Result<Heightfield, String> {
        if columns < 2 || rows < 2 {
            return Err(format!(
                "a heightfield needs at least 2×2 samples, not {columns}×{rows}"
            ));
        }
        let n = columns as usize * rows as usize;
        if n > MAX_HEIGHTFIELD_SAMPLES || heights.len() != n {
            return Err(format!(
                "{columns}×{rows} samples, {} heights",
                heights.len()
            ));
        }
        if heights.iter().any(|h| !h.is_finite())
            || !scale.is_finite()
            || scale.x <= 0.0
            || scale.z <= 0.0
        {
            return Err("heightfield values or scale are not usable".into());
        }
        let (lo, hi) = heights
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| {
                (l.min(v), h.max(v))
            });
        let (lo, hi) = if scale.y >= 0.0 {
            (lo * scale.y, hi * scale.y)
        } else {
            (hi * scale.y, lo * scale.y)
        };
        let aabb = Aabb::new(
            Vec3::new(0.0, lo, 0.0),
            Vec3::new(
                (columns - 1) as f32 * scale.x,
                hi,
                (rows - 1) as f32 * scale.z,
            ),
        );
        Ok(Heightfield {
            columns,
            rows,
            heights,
            scale,
            aabb,
        })
    }

    #[inline]
    pub fn local_aabb(&self) -> Aabb {
        self.aabb
    }

    /// The position of sample `(i, j)`.
    #[inline]
    pub fn sample(&self, i: u32, j: u32) -> Vec3 {
        Vec3::new(
            i as f32 * self.scale.x,
            self.heights[(j * self.columns + i) as usize] * self.scale.y,
            j as f32 * self.scale.z,
        )
    }

    /// Triangle `id` (see the type doc), wound so its front face points up (+Y). Its active edge
    /// bits come from its neighbours: an edge on the grid's border is active, and an inner one
    /// (the cell's own diagonal included) is active only on a ridge, inactive where the
    /// triangles across it are coplanar or form a valley.
    pub fn triangle(&self, id: u32) -> TriangleRef {
        let vertices = self.corners(id);
        let normal = face_normal(&vertices);
        let (active_edges, ridge_cos) = self.edge_bits_with_normal(id, &vertices, normal, None);
        TriangleRef {
            id,
            vertices,
            active_edges,
            ridge_cos,
            normal,
        }
    }

    /// The vertices of triangle `id`.
    fn corners(&self, id: u32) -> [Vec3; 3] {
        let cell = id / 2;
        let (i, j) = (cell % (self.columns - 1), cell / (self.columns - 1));
        let (p00, p10, p01, p11) = (
            self.sample(i, j),
            self.sample(i + 1, j),
            self.sample(i, j + 1),
            self.sample(i + 1, j + 1),
        );
        // Seen from above (+Y) with +X right and +Z down the page, counter-clockwise about +Y is
        // p00 → p01 → p11 and p00 → p11 → p10.
        if id % 2 == 0 {
            [p00, p01, p11]
        } else {
            [p00, p11, p10]
        }
    }

    /// Triangle `id`'s active-edge bits and ridge cosines (see [`TriangleRef`]).
    fn edge_bits_with_normal(
        &self,
        id: u32,
        v: &[Vec3; 3],
        n: Vec3,
        sibling_normal: Option<Vec3>,
    ) -> (u8, [f32; 3]) {
        let mut bits = 0;
        let mut ridge_cos = [-1.0f32; 3];
        for k in 0..3 {
            let active = match self.neighbour(id, k) {
                None => true,
                Some(other) => {
                    let m = if other == (id ^ 1) {
                        sibling_normal.unwrap_or_else(|| face_normal(&self.corners(other)))
                    } else {
                        face_normal(&self.corners(other))
                    };
                    let ridge = crease_is_active(n, m, v[k], v[(k + 1) % 3]);
                    if ridge {
                        ridge_cos[k] = n.dot(m);
                    }
                    ridge
                }
            };
            bits |= u8::from(active) << k;
        }
        (bits, ridge_cos)
    }

    /// The triangle across edge `edge` of triangle `id`, unless the edge is on the grid's border.
    #[inline]
    fn neighbour(&self, id: u32, edge: usize) -> Option<u32> {
        let (cw, ch) = (self.columns - 1, self.rows - 1);
        let cell = id / 2;
        let (i, j) = (cell % cw, cell / cw);
        let id_of = |i: u32, j: u32, half: u32| 2 * (j * cw + i) + half;
        match (id % 2, edge) {
            // Half 0 is p00 → p01 → p11: the cell's −X side, its +Z side, then the diagonal.
            (0, 0) => (i > 0).then(|| id_of(i - 1, j, 1)),
            (0, 1) => (j + 1 < ch).then(|| id_of(i, j + 1, 1)),
            (0, _) => Some(id + 1),
            // Half 1 is p00 → p11 → p10: the diagonal, the cell's +X side, then its −Z side.
            (_, 0) => Some(id - 1),
            (_, 1) => (i + 1 < cw).then(|| id_of(i + 1, j, 0)),
            _ => (j > 0).then(|| id_of(i, j - 1, 0)),
        }
    }

    /// The lowest and highest sample heights of cell `(i, j)`.
    #[inline]
    fn cell_heights(&self, i: u32, j: u32) -> (f32, f32) {
        let y = |i: u32, j: u32| self.heights[(j * self.columns + i) as usize] * self.scale.y;
        let (a, b, c, d) = (y(i, j), y(i + 1, j), y(i, j + 1), y(i + 1, j + 1));
        (a.min(b).min(c).min(d), a.max(b).max(c).max(d))
    }

    /// Calls `f` for every triangle whose bounds overlap `aabb` (local frame), in a deterministic
    /// order, until `f` returns `false`. Only the cells whose column and row ranges overlap the
    /// box are visited, row by row; the order is that of the triangle ids.
    pub fn query_aabb(&self, aabb: &Aabb, f: &mut dyn FnMut(&TriangleRef) -> bool) {
        // Every triangle's heights lie within the field's, so a box wholly above or below them
        // meets nothing.
        if !(aabb.min.y <= self.aabb.max.y && aabb.max.y >= self.aabb.min.y) {
            return;
        }
        let (cw, ch) = (self.columns - 1, self.rows - 1);
        let Some((i0, i1)) = cell_span(aabb.min.x, aabb.max.x, self.scale.x, cw) else {
            return;
        };
        let Some((j0, j1)) = cell_span(aabb.min.z, aabb.max.z, self.scale.z, ch) else {
            return;
        };
        for j in j0..=j1 {
            for i in i0..=i1 {
                let (p00, p10, p01, p11) = (
                    self.sample(i, j),
                    self.sample(i + 1, j),
                    self.sample(i, j + 1),
                    self.sample(i + 1, j + 1),
                );
                let y_lo = p00.y.min(p10.y).min(p01.y).min(p11.y);
                let y_hi = p00.y.max(p10.y).max(p01.y).max(p11.y);
                if !(aabb.min.y <= y_hi && aabb.max.y >= y_lo) {
                    continue;
                }
                let cell = j * cw + i;
                let v0 = [p00, p01, p11];
                let v1 = [p00, p11, p10];
                let hit0 = Aabb::from_points(&v0).overlaps(aabb);
                let hit1 = Aabb::from_points(&v1).overlaps(aabb);
                if !hit0 && !hit1 {
                    continue;
                }
                let n0 = face_normal(&v0);
                let n1 = face_normal(&v1);
                if hit0 {
                    let id = 2 * cell;
                    let (active_edges, ridge_cos) =
                        self.edge_bits_with_normal(id, &v0, n0, Some(n1));
                    let tri = TriangleRef {
                        id,
                        vertices: v0,
                        active_edges,
                        ridge_cos,
                        normal: n0,
                    };
                    if !f(&tri) {
                        return;
                    }
                }
                if hit1 {
                    let id = 2 * cell + 1;
                    let (active_edges, ridge_cos) =
                        self.edge_bits_with_normal(id, &v1, n1, Some(n0));
                    let tri = TriangleRef {
                        id,
                        vertices: v1,
                        active_edges,
                        ridge_cos,
                        normal: n1,
                    };
                    if !f(&tri) {
                        return;
                    }
                }
            }
        }
    }

    /// The first triangle a ray (`dir` unit length, local frame) hits for `t` in `[0, max_t]`,
    /// front faces only (terrain is seen from above).
    ///
    /// Walks the cells under the ray's path across the grid in order (a 2D DDA, a row at a time,
    /// each row's cells along the ray), skipping cells the ray passes wholly above or below, and
    /// stops at the first row or cell the ray reaches only beyond the nearest hit so far. Cells
    /// are taken with a small tolerance, so a ray along a grid line, through a grid point or
    /// straight down also tries the triangles on the other side of the line, which the triangle
    /// test may hit just as well.
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_t: f32) -> Option<MeshHit> {
        if max_t.is_nan()
            || max_t < 0.0
            || !origin.is_finite()
            || !dir.is_finite()
            || dir == Vec3::ZERO
        {
            return None;
        }
        let (sx, sz) = (self.scale.x, self.scale.z);
        let (cw, ch) = (self.columns - 1, self.rows - 1);
        let tol = 1e-4 * sx.max(sz) + cull_pad(&self.aabb, origin);
        let inv = inverse_direction(dir);
        let (t0, t1) = slab(&self.aabb, origin, inv, max_t, tol)?;
        // Where the ray is along one axis at `t` (where it does not move along the axis, the
        // walk has already bounded `t` by the others).
        let at = |o: f32, d: f32, inv: f32, t: f32| if inv.is_infinite() { o } else { o + t * d };
        let (za, zb) = (
            at(origin.z, dir.z, inv.z, t0),
            at(origin.z, dir.z, inv.z, t1),
        );
        let (j0, j1) = cell_span(za.min(zb) - tol, za.max(zb) + tol, sz, ch)?;
        let mut best: Option<MeshHit> = None;
        let mut limit = max_t;
        for row in 0..=j1 - j0 {
            let j = if dir.z < 0.0 { j1 - row } else { j0 + row };
            let (ra, rb) = band(origin.z, inv.z, j, sz, tol, t0, t1);
            if ra > rb {
                continue;
            }
            if ra > limit {
                break;
            }
            let (xa, xb) = (
                at(origin.x, dir.x, inv.x, ra),
                at(origin.x, dir.x, inv.x, rb),
            );
            let Some((i0, i1)) = cell_span(xa.min(xb) - tol, xa.max(xb) + tol, sx, cw) else {
                continue;
            };
            for column in 0..=i1 - i0 {
                let i = if dir.x < 0.0 {
                    i1 - column
                } else {
                    i0 + column
                };
                let (ca, cb) = band(origin.x, inv.x, i, sx, tol, ra, rb);
                if ca > cb {
                    continue;
                }
                if ca > limit {
                    break;
                }
                let (lo, hi) = self.cell_heights(i, j);
                let (ya, yb) = (
                    at(origin.y, dir.y, inv.y, ca),
                    at(origin.y, dir.y, inv.y, cb),
                );
                if ya.min(yb) > hi + tol || ya.max(yb) < lo - tol {
                    continue;
                }
                let cell = j * cw + i;
                for id in [2 * cell, 2 * cell + 1] {
                    let v = self.corners(id);
                    if let Some(t) = ray_triangle(origin, dir, &v, limit, false)
                        && is_nearer(t, id, best)
                    {
                        best = Some(MeshHit {
                            t,
                            triangle: id,
                            normal: face_normal(&v),
                        });
                        limit = t;
                    }
                }
            }
        }
        best
    }
}

/// The cells `k` in `0..n` whose span `[k·s, (k + 1)·s]` (computed as the samples' coordinates
/// are) meets `[lo, hi]`, that is `k·s <= hi` and `(k + 1)·s >= lo`: the same test as
/// [`Aabb::overlaps`] on the cells' bounds. `None` when there are none (or a bound is NaN).
fn cell_span(lo: f32, hi: f32, s: f32, n: u32) -> Option<(u32, u32)> {
    let edge = |k: u32| k as f32 * s;
    // A first guess from the division, corrected by the exact tests below (a step or two).
    let guess = |x: f32| {
        let q = x / s;
        if q >= (n - 1) as f32 {
            n - 1
        } else if q > 0.0 {
            q as u32
        } else {
            0
        }
    };
    // The first cell whose far side reaches `lo`.
    let mut k0 = guess(lo);
    while k0 > 0 && edge(k0) >= lo {
        k0 -= 1;
    }
    while k0 < n - 1 && edge(k0 + 1) < lo {
        k0 += 1;
    }
    // The last cell whose near side is not past `hi`.
    let mut k1 = guess(hi);
    while k1 > 0 && edge(k1) > hi {
        k1 -= 1;
    }
    while k1 < n - 1 && edge(k1 + 1) <= hi {
        k1 += 1;
    }
    (k0 <= k1 && edge(k0 + 1) >= lo && edge(k1) <= hi).then_some((k0, k1))
}

/// The part of `[ta, tb]` in which `o + t·d` (`inv = 1/d`) is within cell `k`'s span along one
/// axis (cell size `s`), grown by `tol`. All of it when the ray does not move along the axis: the
/// cell was picked from the ray's coordinate.
fn band(o: f32, inv: f32, k: u32, s: f32, tol: f32, ta: f32, tb: f32) -> (f32, f32) {
    if inv.is_infinite() {
        return (ta, tb);
    }
    let a = (k as f32 * s - tol - o) * inv;
    let b = ((k + 1) as f32 * s + tol - o) * inv;
    (ta.max(a.min(b)), tb.min(a.max(b)))
}

/// The distance along `dir` at which the ray `origin + t·dir` meets triangle `v`, if within
/// `[0, max_t]`: front faces (the ray against the counter-clockwise normal) only, unless
/// `both_sides`.
///
/// Watertight (Woop, Benthin and Wald, "Watertight Ray/Triangle Intersection", JCGT 2013): the
/// corners, taken relative to the origin, are sheared so the ray runs along an axis, and the
/// test is three 2D edge functions there, one per edge, each from that edge's two corners alone.
/// Two triangles that share an edge (with bit-identical corners) get exactly opposite values for
/// it, so they never both turn away a ray that passes between them: a ray through a shared edge
/// or vertex hits at least one of the triangles there (both, when the value is exactly zero),
/// and never slips through a closed surface. Rounding moves each sheared corner by a few ulps of
/// its coordinates, the same for every triangle that has it, and the sign of each edge function
/// is then exact or zero for the moved corners (rounding never reverses the order of two
/// products).
pub fn ray_triangle(
    origin: Vec3,
    dir: Vec3,
    v: &[Vec3; 3],
    max_t: f32,
    both_sides: bool,
) -> Option<f32> {
    // The axis the ray runs along most becomes the depth; swapping the other two when the ray
    // runs down it keeps the corners' winding as the ray sees them.
    let kz = dir.abs().max_axis();
    let (mut kx, mut ky) = ((kz + 1) % 3, (kz + 2) % 3);
    if dir[kz] < 0.0 {
        std::mem::swap(&mut kx, &mut ky);
    }
    let (sx, sy, sz) = (dir[kx] / dir[kz], dir[ky] / dir[kz], 1.0 / dir[kz]);
    let (a, b, c) = (v[0] - origin, v[1] - origin, v[2] - origin);
    // Each corner slid along the ray onto the plane through the origin, where the ray is the
    // point (0, 0).
    let (ax, ay) = (a[kx] - sx * a[kz], a[ky] - sy * a[kz]);
    let (bx, by) = (b[kx] - sx * b[kz], b[ky] - sy * b[kz]);
    let (cx, cy) = (c[kx] - sx * c[kz], c[ky] - sy * c[kz]);
    // The edge functions of the edges facing `a`, `b` and `c`: for the edge from p to q,
    // `qx·py − qy·px`. Products commute and `x − y` rounds to exactly `−(y − x)`, so the
    // neighbour that runs the edge from q to p gets exactly the negated value.
    let eu = cx * by - cy * bx;
    let ev = ax * cy - ay * cx;
    let ew = bx * ay - by * ax;
    // All three at least zero: the ray passes inside a front face; at most zero: a back face.
    // An exact zero is on the edge, and the triangles on both sides of it take it.
    let front = eu >= 0.0 && ev >= 0.0 && ew >= 0.0;
    let back = both_sides && eu <= 0.0 && ev <= 0.0 && ew <= 0.0;
    // Twice the triangle's area as the ray sees it: zero when the ray runs along the triangle's
    // plane. Under the smallest normal float (corners some 1e-19 m apart, far smaller than any
    // triangle a mesh keeps), the weighted depths that place the hit underflow; past the largest,
    // the sum itself overflowed. Neither places a hit, so neither is one.
    let det = eu + ev + ew;
    if !(front || back) || !det.is_normal() {
        return None;
    }
    // The corners' depths along the ray, weighted by the edge functions (the barycentric
    // coordinates times `det`): the hit's depth, within the corners' own up to rounding.
    let mut t = (eu * a[kz] + ev * b[kz] + ew * c[kz]) * sz / det;
    // A triangle square to an axis (a level floor, an upright wall): its plane's distance, exact
    // to a rounding, where interpolating the corners' depths loses digits to corners far beyond
    // the hit and puts a ray that starts on the floor a rounding behind it. Kept within the
    // corners' depths, as the interpolated distance is.
    if let Some(k) = (0..3).find(|&k| v[0][k] == v[1][k] && v[1][k] == v[2][k]) {
        let (za, zb, zc) = (a[kz] * sz, b[kz] * sz, c[kz] * sz);
        let plane = a[k] / dir[k];
        if plane >= za.min(zb).min(zc) && plane <= za.max(zb).max(zc) {
            t = plane;
        }
    }
    (t >= 0.0 && t <= max_t).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> TriangleMesh {
        // A 2 m square floor at y = 0, front face up.
        let v = vec![
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, 1.0),
        ];
        TriangleMesh::new(v, vec![[0, 2, 1], [0, 3, 2]]).unwrap()
    }

    #[test]
    fn a_floor_is_hit_from_above_only() {
        let m = quad();
        assert!((m.triangle(0).normal() - Vec3::Y).length() < 1e-6);
        let hit = m
            .raycast(Vec3::new(0.2, 3.0, 0.1), -Vec3::Y, 10.0, false)
            .unwrap();
        assert!((hit.t - 3.0).abs() < 1e-5 && (hit.normal - Vec3::Y).length() < 1e-6);
        assert!(
            m.raycast(Vec3::new(0.2, -3.0, 0.1), Vec3::Y, 10.0, false)
                .is_none()
        );
        assert!(
            m.raycast(Vec3::new(0.2, -3.0, 0.1), Vec3::Y, 10.0, true)
                .is_some()
        );
        let mut seen = vec![];
        m.query_aabb(
            &Aabb::new(Vec3::new(0.5, -0.1, -0.9), Vec3::new(0.9, 0.1, -0.8)),
            &mut |t| {
                seen.push(t.id);
                true
            },
        );
        // Both triangles' bounds cover the whole square.
        assert_eq!(seen, vec![0, 1]);
        let mut above = 0;
        m.query_aabb(
            &Aabb::new(Vec3::new(-1.0, 1.0, -1.0), Vec3::new(1.0, 2.0, 1.0)),
            &mut |_| {
                above += 1;
                true
            },
        );
        assert_eq!(above, 0);
    }

    #[test]
    fn bad_meshes_are_refused_and_degenerate_triangles_dropped() {
        assert!(TriangleMesh::new(vec![Vec3::ZERO], vec![[0, 0, 1]]).is_err());
        let v = vec![Vec3::ZERO, Vec3::X, Vec3::Z, Vec3::X * 2.0];
        let m = TriangleMesh::new(v, vec![[0, 2, 1], [0, 1, 3]]).unwrap();
        assert_eq!(m.triangles.len(), 1, "the collinear triangle is dropped");
    }

    #[test]
    fn heightfield_triangles_face_up_and_tile_the_grid() {
        let h = Heightfield::new(
            3,
            2,
            vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            Vec3::new(2.0, 1.0, 2.0),
        )
        .unwrap();
        assert_eq!(
            h.local_aabb(),
            Aabb::new(Vec3::ZERO, Vec3::new(4.0, 1.0, 2.0))
        );
        for id in 0..4 {
            assert!(h.triangle(id).normal().y > 0.0, "triangle {id} faces up");
        }
        let hit = h.raycast(Vec3::new(1.0, 5.0, 1.0), -Vec3::Y, 10.0).unwrap();
        assert!((hit.t - 4.5).abs() < 1e-5, "{hit:?}");
    }

    /// SplitMix64, for repeatable test data.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        /// Uniform in `[0, 1)`.
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 * (1.0 / 16_777_216.0)
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
        fn below(&mut self, n: u32) -> u32 {
            (self.next() % u64::from(n)) as u32
        }
        fn point(&mut self, lo: f32, hi: f32) -> Vec3 {
            Vec3::new(self.range(lo, hi), self.range(lo, hi), self.range(lo, hi))
        }
        fn direction(&mut self) -> Vec3 {
            loop {
                let v = self.point(-1.0, 1.0);
                if (1e-4..=1.0).contains(&v.length_squared()) {
                    return v.normalize_or_zero();
                }
            }
        }
        /// A query box around `center`: usually small, sometimes large, flat, a point or
        /// inside out.
        fn query_box(&mut self, center: Vec3, size: f32) -> Aabb {
            let half = Vec3::new(
                self.range(0.0, size),
                self.range(0.0, size),
                self.range(0.0, size),
            );
            match self.below(8) {
                0 => Aabb::new(center, center),
                1 => Aabb::from_center_half(center, Vec3::new(half.x, 0.0, half.z)),
                2 => Aabb::from_center_half(center, half * 6.0),
                // Inside out along X: what `Aabb::overlaps` makes of it must still hold.
                3 => Aabb::new(
                    Vec3::new(center.x + half.x, center.y - half.y, center.z - half.z),
                    Vec3::new(center.x - half.x, center.y + half.y, center.z + half.z),
                ),
                _ => Aabb::from_center_half(center, half),
            }
        }
    }

    /// The nearest hit by brute force (smallest distance, then lowest id) as `(t bits, id)`.
    fn brute_ray(
        triangles: impl IntoIterator<Item = (u32, [Vec3; 3])>,
        origin: Vec3,
        dir: Vec3,
        max_t: f32,
        both_sides: bool,
    ) -> Option<(u32, u32)> {
        let mut best: Option<(f32, u32)> = None;
        for (id, v) in triangles {
            if let Some(t) = ray_triangle(origin, dir, &v, max_t, both_sides)
                && best.is_none_or(|(bt, bid)| t < bt || (t == bt && id < bid))
            {
                best = Some((t, id));
            }
        }
        best.map(|(t, id)| (t.to_bits(), id))
    }

    fn mesh_triangles(m: &TriangleMesh) -> impl Iterator<Item = (u32, [Vec3; 3])> + '_ {
        (0..m.triangles.len() as u32).map(move |i| (i, m.corners(i)))
    }

    fn hit_key(hit: Option<MeshHit>) -> Option<(u32, u32)> {
        hit.map(|h| (h.t.to_bits(), h.triangle))
    }

    fn mesh_query(m: &TriangleMesh, b: &Aabb) -> Vec<u32> {
        let mut seen = vec![];
        m.query_aabb(b, &mut |t| {
            assert_eq!(*t, m.triangle(t.id), "the reference matches the mesh");
            seen.push(t.id);
            true
        });
        seen
    }

    fn mesh_brute_query(m: &TriangleMesh, b: &Aabb) -> Vec<u32> {
        (0..m.triangles.len() as u32)
            .filter(|&i| Aabb::from_points(&m.corners(i)).overlaps(b))
            .collect()
    }

    /// Checks the hierarchy's invariants: each triangle in exactly one leaf, leaves of 1 to
    /// [`LEAF_TRIANGLES`] in ascending order, and every box holding what is below it.
    fn check_hierarchy(m: &TriangleMesh) -> usize {
        let mut seen = vec![0u32; m.triangles.len()];
        let mut stack = vec![(0usize, 0usize)];
        let (mut visited, mut depth) = (0, 0);
        while let Some((i, d)) = stack.pop() {
            visited += 1;
            depth = depth.max(d);
            let node = m.nodes[i];
            if node.count > 0 {
                assert!(node.count as usize <= LEAF_TRIANGLES);
                let leaf = m.leaf(&node);
                assert!(leaf.windows(2).all(|w| w[0] < w[1]), "leaf {i} is sorted");
                for &t in leaf {
                    seen[t as usize] += 1;
                    assert!(node.aabb.contains(&Aabb::from_points(&m.corners(t))));
                }
            } else {
                for c in [node.first as usize, node.first as usize + 1] {
                    assert!(c > i && c < m.nodes.len(), "child {c} of {i}");
                    assert!(node.aabb.contains(&m.nodes[c].aabb));
                    stack.push((c, d + 1));
                }
            }
        }
        assert_eq!(visited, m.nodes.len(), "every node is reached once");
        assert!(
            seen.iter().all(|&n| n == 1),
            "every triangle is in one leaf"
        );
        assert_eq!(m.nodes[0].aabb, m.local_aabb());
        depth
    }

    /// A random triangle soup: 3,000 triangles of mixed sizes in a 100 m cube.
    fn soup(rng: &mut Rng) -> TriangleMesh {
        let (mut vertices, mut triangles) = (vec![], vec![]);
        for k in 0..3000u32 {
            let c = rng.point(-50.0, 50.0);
            let size = if rng.below(10) == 0 { 12.0 } else { 2.5 };
            for _ in 0..3 {
                vertices.push(c + rng.point(-size, size));
            }
            triangles.push([3 * k, 3 * k + 1, 3 * k + 2]);
        }
        TriangleMesh::new(vertices, triangles).unwrap()
    }

    #[test]
    fn a_mesh_hierarchy_answers_box_queries_exactly_like_brute_force() {
        let mut rng = Rng(11);
        let m = soup(&mut rng);
        assert_eq!(m.triangles.len(), 3000);
        let depth = check_hierarchy(&m);
        assert!(depth < 40, "depth {depth}");
        let mut total = 0;
        for k in 0..400 {
            let center = if k % 2 == 0 {
                rng.point(-60.0, 60.0)
            } else {
                // Around a triangle, so most queries find something.
                m.corners(rng.below(3000))[0]
            };
            let b = rng.query_box(center, 4.0);
            let mut got = mesh_query(&m, &b);
            let want = mesh_brute_query(&m, &b);
            got.sort_unstable();
            assert_eq!(got, want, "box {b:?}");
            total += want.len();
        }
        assert!(total > 2000, "the queries find triangles ({total})");
        // The whole mesh, nothing, and the same order every time.
        assert_eq!(mesh_query(&m, &m.local_aabb()).len(), 3000);
        let far = Aabb::new(Vec3::splat(100.0), Vec3::splat(101.0));
        assert!(mesh_query(&m, &far).is_empty());
        let b = Aabb::from_center_half(Vec3::ZERO, Vec3::splat(20.0));
        assert_eq!(mesh_query(&m, &b), mesh_query(&soup(&mut Rng(11)), &b));
        // A query stops when its callback says so.
        let mut calls = 0;
        m.query_aabb(&m.local_aabb(), &mut |_| {
            calls += 1;
            calls < 7
        });
        assert_eq!(calls, 7);
    }

    #[test]
    fn a_mesh_hierarchy_casts_rays_exactly_like_brute_force() {
        let mut rng = Rng(12);
        let m = soup(&mut rng);
        let mut hits = 0;
        for k in 0..1500 {
            let origin = rng.point(-80.0, 80.0);
            let dir = if k % 2 == 0 {
                let [a, b, c] = m.corners(rng.below(3000));
                (a + b + c - origin * 3.0).normalize_or_zero()
            } else {
                rng.direction()
            };
            let max_t = match rng.below(4) {
                0 => f32::INFINITY,
                1 => rng.range(0.0, 30.0),
                _ => rng.range(0.0, 250.0),
            };
            let both_sides = rng.below(2) == 0;
            let got = hit_key(m.raycast(origin, dir, max_t, both_sides));
            let want = brute_ray(mesh_triangles(&m), origin, dir, max_t, both_sides);
            assert_eq!(got, want, "ray {origin:?} {dir:?} {max_t} {both_sides}");
            if let Some(hit) = m.raycast(origin, dir, max_t, both_sides) {
                assert_eq!(hit.normal, m.triangle(hit.triangle).normal());
                hits += 1;
            }
        }
        assert!(hits > 400, "the rays hit something ({hits})");
        // Rays that cannot hit: zero length, NaN or negative reach, a zero direction.
        let o = Vec3::new(0.0, 100.0, 0.0);
        assert!(m.raycast(o, -Vec3::Y, -1.0, true).is_none());
        assert!(m.raycast(o, -Vec3::Y, f32::NAN, true).is_none());
        assert!(m.raycast(o, Vec3::ZERO, 1e6, true).is_none());
        assert!(m.raycast(o, -Vec3::Y, 0.0, true).is_none());
    }

    /// A `cells × cells` grid mesh of random heights with shared vertices, laid out like a
    /// heightfield of unit cells (so its triangle `2 · cell + half` matches one).
    fn terrain_mesh(rng: &mut Rng, cells: u32) -> TriangleMesh {
        let n = cells + 1;
        let vertices = (0..n * n)
            .map(|k| Vec3::new((k % n) as f32, rng.range(-2.0, 2.0), (k / n) as f32))
            .collect();
        let mut triangles = vec![];
        for j in 0..cells {
            for i in 0..cells {
                let (p00, p10, p01, p11) = (
                    j * n + i,
                    j * n + i + 1,
                    (j + 1) * n + i,
                    (j + 1) * n + i + 1,
                );
                triangles.push([p00, p01, p11]);
                triangles.push([p00, p11, p10]);
            }
        }
        TriangleMesh::new(vertices, triangles).unwrap()
    }

    #[test]
    fn rays_through_shared_edges_and_vertices_pick_the_same_triangle_as_brute_force() {
        let mut rng = Rng(13);
        let m = terrain_mesh(&mut rng, 24);
        check_hierarchy(&m);
        let mut rays = vec![];
        for _ in 0..300 {
            let (i, j) = (rng.below(25) as f32, rng.below(25) as f32);
            // Straight down onto a vertex, an edge's midpoint and the middle of a diagonal.
            rays.push((Vec3::new(i, 10.0, j), -Vec3::Y));
            rays.push((Vec3::new(i + 0.5, 10.0, j), -Vec3::Y));
            rays.push((Vec3::new(i + 0.5, 10.0, j + 0.5), -Vec3::Y));
            // Along a grid line, slanting down; along a diagonal through vertices.
            let slope = Vec3::new(0.0, -rng.range(0.2, 2.0), rng.range(-1.0, 1.0));
            rays.push((Vec3::new(i, 6.0, j), slope.normalize_or_zero()));
            rays.push((
                Vec3::new(j, 6.0, i),
                Vec3::new(slope.z, slope.y, 0.0).normalize_or_zero(),
            ));
            rays.push((
                Vec3::new(i, 8.0, j),
                Vec3::new(1.0, -rng.range(0.1, 1.0), 1.0).normalize_or_zero(),
            ));
        }
        let mut hits = 0;
        for (origin, dir) in rays {
            for both_sides in [false, true] {
                let got = hit_key(m.raycast(origin, dir, f32::INFINITY, both_sides));
                let want = brute_ray(mesh_triangles(&m), origin, dir, f32::INFINITY, both_sides);
                assert_eq!(got, want, "ray {origin:?} {dir:?}");
                hits += usize::from(got.is_some());
            }
        }
        assert!(hits > 2000, "{hits} hits");
    }

    /// The bits a triangle's edges should have: those for which `active(a, b)` holds.
    fn expected_bits(v: [Vec3; 3], active: impl Fn(Vec3, Vec3) -> bool) -> u8 {
        (0..3).fold(0, |bits, k| {
            bits | (u8::from(active(v[k], v[(k + 1) % 3])) << k)
        })
    }

    /// Whether the segment from `a` to `b` runs along the border of the square `[0, n]²` (XZ).
    fn on_border(a: Vec3, b: Vec3, n: f32) -> bool {
        (a.x == b.x && (a.x == 0.0 || a.x == n)) || (a.z == b.z && (a.z == 0.0 || a.z == n))
    }

    #[test]
    fn a_flat_grid_has_inactive_seams_and_active_borders_even_with_split_vertices() {
        let mut rng = Rng(14);
        let shared = terrain_mesh(&mut rng, 8);
        // Flatten it, then split every triangle's vertices apart (with some zeros negated, as
        // exporters do), which must find the same neighbours by position.
        let flat: Vec<Vec3> = shared
            .vertices
            .iter()
            .map(|v| Vec3::new(v.x, 0.0, v.z))
            .collect();
        let shared = TriangleMesh::new(flat.clone(), shared.triangles.clone()).unwrap();
        let mut split_vertices = vec![];
        let mut split_triangles = vec![];
        for (k, t) in shared.triangles.iter().enumerate() {
            let base = split_vertices.len() as u32;
            for &i in t {
                let v = flat[i as usize];
                split_vertices.push(if k % 2 == 1 {
                    Vec3::new(v.x, -0.0, v.z)
                } else {
                    v
                });
            }
            split_triangles.push([base, base + 1, base + 2]);
        }
        let split = TriangleMesh::new(split_vertices, split_triangles).unwrap();
        for m in [&shared, &split] {
            assert_eq!(m.triangles.len(), 128);
            for i in 0..128 {
                let t = m.triangle(i);
                let want = expected_bits(t.vertices, |a, b| on_border(a, b, 8.0));
                assert_eq!(t.active_edges, want, "triangle {i} {:?}", t.vertices);
            }
        }
        // Every cell's diagonal and inner side is a seam; 32 border edges stay active.
        let active: u32 = split.active_edges.iter().map(|b| b.count_ones()).sum();
        assert_eq!(active, 32);
    }

    /// Two triangles sharing the edge from (0, 0, 0) to (0, 0, 1): one flat on the −X side, one
    /// on the +X side tilted down by `degrees` (a ridge) or up (a valley, negative degrees).
    fn fold(degrees: f32) -> TriangleMesh {
        let drop = degrees.to_radians().tan();
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, -drop, 0.5),
            Vec3::new(-1.0, 0.0, 0.5),
        ];
        TriangleMesh::new(v, vec![[0, 1, 2], [1, 0, 3]]).unwrap()
    }

    #[test]
    fn ridges_are_active_while_valleys_and_near_flat_folds_are_not() {
        for (degrees, active) in [
            (40.0, true),
            (1.0, true),
            (0.2, true),
            (0.05, false),
            (0.0, false),
            (-0.05, false),
            (-0.2, false),
            (-40.0, false),
            (-89.0, false),
        ] {
            let m = fold(degrees);
            assert!(m.triangle(0).normal().y > 0.0 && m.triangle(1).normal().y > 0.0);
            // Edge 0 of each triangle is the shared one; the other edges are borders.
            for t in 0..2 {
                assert_eq!(
                    m.active_edges[t],
                    0b110 | u8::from(active),
                    "{degrees}° fold, triangle {t}"
                );
                // A ridge keeps the cosine between the two normals; nothing else bounds its
                // edges' normals.
                let ridge = if active {
                    let (n0, n1) = (m.triangle(0).normal(), m.triangle(1).normal());
                    assert!((n0.dot(n1) - degrees.to_radians().cos()).abs() < 1e-6);
                    n0.dot(n1)
                } else {
                    -1.0
                };
                assert_eq!(m.ridge_cos[t], [ridge, -1.0, -1.0], "{degrees}° fold");
                assert_eq!(m.triangle(t as u32).ridge_cos, m.ridge_cos[t]);
            }
        }
        // A roof: both slopes fall away from the ridge.
        let roof = TriangleMesh::new(
            vec![
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 1.0),
                Vec3::new(1.0, 0.0, 0.5),
                Vec3::new(-1.0, 0.0, 0.5),
            ],
            vec![[0, 1, 2], [1, 0, 3]],
        )
        .unwrap();
        assert_eq!(roof.active_edges, vec![0b111, 0b111]);
        // A gutter: both slopes rise from the valley.
        let gutter = TriangleMesh::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, 1.0, 0.5),
                Vec3::new(-1.0, 1.0, 0.5),
            ],
            vec![[0, 1, 2], [1, 0, 3]],
        )
        .unwrap();
        assert_eq!(gutter.active_edges, vec![0b110, 0b110]);
    }

    #[test]
    fn non_manifold_and_inconsistently_wound_edges_stay_active() {
        // Three triangles fanned around the edge (0, 0, 0)–(0, 0, 1).
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.5),
            Vec3::new(-1.0, 0.0, 0.5),
            Vec3::new(0.0, 1.0, 0.5),
        ];
        let m = TriangleMesh::new(v.clone(), vec![[0, 1, 2], [1, 0, 3], [0, 4, 1]]).unwrap();
        assert_eq!(m.active_edges, vec![0b111; 3]);
        // Two coplanar triangles, one wound backwards: the shared edge runs the same way twice.
        let m = TriangleMesh::new(v, vec![[0, 1, 2], [0, 1, 3]]).unwrap();
        assert_eq!(m.active_edges, vec![0b111, 0b111]);
    }

    /// A `columns × rows` field of random heights (some cells flat, some steep) with uneven
    /// scales, so axis mix-ups show.
    fn random_field(rng: &mut Rng, columns: u32, rows: u32) -> Heightfield {
        let heights = (0..columns * rows)
            .map(|k| match rng.below(6) {
                0 => 0.0,
                1 => rng.range(-6.0, 6.0),
                _ => rng.range(-1.5, 1.5) + ((k % columns) as f32 * 0.1),
            })
            .collect();
        Heightfield::new(columns, rows, heights, Vec3::new(0.75, 1.5, 1.25)).unwrap()
    }

    fn field_query(h: &Heightfield, b: &Aabb) -> Vec<u32> {
        let mut seen = vec![];
        h.query_aabb(b, &mut |t| {
            assert_eq!(*t, h.triangle(t.id), "the reference matches the field");
            seen.push(t.id);
            true
        });
        seen
    }

    fn field_triangles(h: &Heightfield) -> u32 {
        2 * (h.columns - 1) * (h.rows - 1)
    }

    #[test]
    fn heightfield_box_queries_visit_exactly_the_overlapping_triangles_in_id_order() {
        let mut rng = Rng(15);
        let h = random_field(&mut rng, 64, 64);
        let (sx, sz) = (h.scale.x, h.scale.z);
        let brute = |b: &Aabb| -> Vec<u32> {
            (0..field_triangles(&h))
                .filter(|&id| Aabb::from_points(&h.corners(id)).overlaps(b))
                .collect()
        };
        let mut boxes = vec![
            h.local_aabb(),
            h.local_aabb().expand(5.0),
            Aabb::new(Vec3::splat(-10.0), Vec3::splat(-1.0)),
            Aabb::new(Vec3::new(0.0, 50.0, 0.0), Vec3::new(10.0, 60.0, 10.0)),
            // Exactly on grid lines: touching a cell's side counts as overlapping it.
            Aabb::new(
                Vec3::new(3.0 * sx, -100.0, 5.0 * sz),
                Vec3::new(3.0 * sx, 100.0, 5.0 * sz),
            ),
            Aabb::new(
                Vec3::new(sx * 63.0, -100.0, 0.0),
                Vec3::new(sx * 70.0, 100.0, 1.0),
            ),
        ];
        for _ in 0..400 {
            let c = Vec3::new(
                rng.range(-5.0, 52.0),
                rng.range(-8.0, 8.0),
                rng.range(-5.0, 85.0),
            );
            boxes.push(rng.query_box(c, 3.0));
        }
        let mut total = 0;
        for b in &boxes {
            let want = brute(b);
            assert_eq!(field_query(&h, b), want, "box {b:?}");
            total += want.len();
        }
        assert!(total > 20_000, "the queries find triangles ({total})");
        let mut calls = 0;
        h.query_aabb(&h.local_aabb(), &mut |_| {
            calls += 1;
            calls < 5
        });
        assert_eq!(calls, 5, "a query stops when its callback says so");
    }

    #[test]
    fn heightfield_rays_match_brute_force_along_grid_lines_straight_down_and_from_below() {
        let mut rng = Rng(16);
        let h = random_field(&mut rng, 64, 64);
        let (sx, sz) = (h.scale.x, h.scale.z);
        let size = h.local_aabb().max;
        let all: Vec<(u32, [Vec3; 3])> = (0..field_triangles(&h))
            .map(|id| (id, h.corners(id)))
            .collect();
        // The surface's height at (x, z), by brute force.
        let ground = |x: f32, z: f32| {
            let top = Vec3::new(x, 100.0, z);
            let (t, _) = brute_ray(all.iter().copied(), top, -Vec3::Y, f32::INFINITY, false)
                .expect("the point is over the field");
            100.0 - f32::from_bits(t)
        };
        let mut rays: Vec<(Vec3, Vec3, f32)> = vec![];
        let reach = |rng: &mut Rng| match rng.below(4) {
            0 => rng.range(0.0, 15.0),
            _ => f32::INFINITY,
        };
        for _ in 0..150 {
            let (i, j) = (rng.below(64) as f32, rng.below(64) as f32);
            let (x, z) = (rng.range(0.0, size.x), rng.range(0.0, size.z));
            // From above towards a random point of the field.
            let o = Vec3::new(
                rng.range(-10.0, 58.0),
                rng.range(2.0, 20.0),
                rng.range(-10.0, 90.0),
            );
            let target = Vec3::new(x, rng.range(-3.0, 3.0), z);
            rays.push((o, (target - o).normalize_or_zero(), reach(&mut rng)));
            // Along a column line and along a row line, slanting down.
            let down = -rng.range(0.1, 1.0);
            let along = rng.range(-1.0, 1.0);
            rays.push((
                Vec3::new(i * sx, 10.0, z),
                Vec3::new(0.0, down, along).normalize_or_zero(),
                reach(&mut rng),
            ));
            rays.push((
                Vec3::new(x, 10.0, j * sz),
                Vec3::new(along, down, 0.0).normalize_or_zero(),
                reach(&mut rng),
            ));
            // Straight down: anywhere, onto a grid line, onto a grid point.
            rays.push((Vec3::new(x, 20.0, z), -Vec3::Y, reach(&mut rng)));
            rays.push((Vec3::new(i * sx, 20.0, z), -Vec3::Y, f32::INFINITY));
            rays.push((Vec3::new(i * sx, 20.0, j * sz), -Vec3::Y, f32::INFINITY));
            // Through grid points along the cells' diagonals.
            rays.push((
                Vec3::new(i * sx, 9.0, j * sz),
                Vec3::new(sx, -rng.range(0.05, 1.0), sz).normalize_or_zero(),
                f32::INFINITY,
            ));
            // From below the surface: up, sideways (out through a back face, perhaps into a
            // hill) and down.
            let below = Vec3::new(x, ground(x, z) - rng.range(0.01, 1.0), z);
            rays.push((below, Vec3::Y, f32::INFINITY));
            rays.push((
                below,
                Vec3::new(
                    rng.range(-1.0, 1.0),
                    rng.range(-0.1, 0.3),
                    rng.range(-1.0, 1.0),
                )
                .normalize_or_zero(),
                f32::INFINITY,
            ));
            rays.push((below, -Vec3::Y, f32::INFINITY));
            // Skimming, nearly level.
            rays.push((
                Vec3::new(-2.0, rng.range(-1.0, 3.0), z),
                Vec3::new(1.0, rng.range(-0.05, 0.05), rng.range(-0.2, 0.2)).normalize_or_zero(),
                f32::INFINITY,
            ));
        }
        let mut hits = 0;
        for (origin, dir, max_t) in rays {
            let got = hit_key(h.raycast(origin, dir, max_t));
            let want = brute_ray(all.iter().copied(), origin, dir, max_t, false);
            assert_eq!(got, want, "ray {origin:?} {dir:?} {max_t}");
            hits += usize::from(got.is_some());
        }
        assert!(hits > 900, "the rays hit the field ({hits})");
        let o = Vec3::new(10.0, 50.0, 10.0);
        assert!(h.raycast(o, -Vec3::Y, f32::NAN).is_none());
        assert!(h.raycast(o, Vec3::ZERO, 1e9).is_none());
        assert!(h.raycast(o, Vec3::Y, 1e9).is_none(), "the field is below");
    }

    #[test]
    fn rays_match_brute_force_at_tiny_and_huge_scales_and_from_far_away() {
        let mut rng = Rng(18);
        for unit in [1e-3f32, 1.0, 1e3] {
            // A terrain mesh far from the origin and a heightfield, both `unit`-sized cells.
            let base = terrain_mesh(&mut rng, 12);
            let offset = Vec3::new(4e3, -1e3, 2e3) * unit;
            let vertices = base.vertices.iter().map(|&v| v * unit + offset).collect();
            let m = TriangleMesh::new(vertices, base.triangles.clone()).unwrap();
            let heights = (0..32 * 32).map(|_| rng.range(-2.0, 2.0)).collect();
            let h = Heightfield::new(32, 32, heights, Vec3::splat(unit)).unwrap();
            let field: Vec<(u32, [Vec3; 3])> = (0..field_triangles(&h))
                .map(|id| (id, h.corners(id)))
                .collect();
            for k in 0..200 {
                let (i, j) = (rng.below(13) as f32, rng.below(13) as f32);
                let (origin, dir) = match k % 4 {
                    // Straight down through a vertex, and slanting along a grid line.
                    0 => (Vec3::new(i, 9.0, j), -Vec3::Y),
                    1 => (
                        Vec3::new(i, 9.0, j),
                        Vec3::new(0.0, -1.0, rng.range(-1.0, 1.0)).normalize_or_zero(),
                    ),
                    // From far away (a thousand times the size), towards the middle.
                    2 => {
                        let o = rng.direction() * 1e4;
                        (o, (Vec3::new(6.0, 0.0, 6.0) - o).normalize_or_zero())
                    }
                    _ => (rng.point(-2.0, 14.0), rng.direction()),
                };
                let mesh_origin = origin * unit + offset;
                let got = hit_key(m.raycast(mesh_origin, dir, f32::INFINITY, true));
                let want = brute_ray(mesh_triangles(&m), mesh_origin, dir, f32::INFINITY, true);
                assert_eq!(got, want, "mesh at {unit}: {mesh_origin:?} {dir:?}");
                let field_origin = origin * unit;
                let got = hit_key(h.raycast(field_origin, dir, f32::INFINITY));
                let want = brute_ray(
                    field.iter().copied(),
                    field_origin,
                    dir,
                    f32::INFINITY,
                    false,
                );
                assert_eq!(got, want, "field at {unit}: {field_origin:?} {dir:?}");
            }
        }
    }

    #[test]
    fn heightfield_edges_match_a_mesh_of_the_same_triangles() {
        let mut rng = Rng(17);
        for (columns, rows) in [(2, 2), (2, 7), (9, 2), (33, 21)] {
            let h = random_field(&mut rng, columns, rows);
            let vertices = (0..columns * rows)
                .map(|k| h.sample(k % columns, k / columns))
                .collect();
            let mut triangles = vec![];
            for j in 0..rows - 1 {
                for i in 0..columns - 1 {
                    let s = |i: u32, j: u32| j * columns + i;
                    let (p00, p10, p01, p11) = (s(i, j), s(i + 1, j), s(i, j + 1), s(i + 1, j + 1));
                    triangles.push([p00, p01, p11]);
                    triangles.push([p00, p11, p10]);
                }
            }
            let m = TriangleMesh::new(vertices, triangles).unwrap();
            assert_eq!(m.triangles.len() as u32, field_triangles(&h));
            for id in 0..field_triangles(&h) {
                assert_eq!(
                    h.triangle(id),
                    m.triangle(id),
                    "triangle {id} of {columns}×{rows}"
                );
            }
        }
    }

    #[test]
    fn heightfield_seams_are_inactive_on_flat_ground_and_creases_follow_the_terrain() {
        let field = |f: &dyn Fn(u32, u32) -> f32| {
            let heights = (0..81).map(|k| f(k % 9, k / 9)).collect();
            Heightfield::new(9, 9, heights, Vec3::ONE).unwrap()
        };
        let all_bits = |h: &Heightfield, active: &dyn Fn(Vec3, Vec3) -> bool| {
            for id in 0..field_triangles(h) {
                let t = h.triangle(id);
                assert_eq!(
                    t.active_edges,
                    expected_bits(t.vertices, active),
                    "triangle {id}"
                );
            }
        };
        // Flat: only the border.
        let flat = field(&|_, _| 2.0);
        all_bits(&flat, &|a, b| on_border(a, b, 8.0));
        // A ridge along the column line x = 4: active there; flat slopes elsewhere. Its slopes
        // fall at 45°, so its normals meet at 90° (cosine 0); no other edge is bounded.
        let ridge = field(&|i, _| -(i as f32 - 4.0).abs());
        all_bits(&ridge, &|a, b| {
            on_border(a, b, 8.0) || (a.x == 4.0 && b.x == 4.0)
        });
        for id in 0..field_triangles(&ridge) {
            let t = ridge.triangle(id);
            for k in 0..3 {
                let (a, b) = (t.vertices[k], t.vertices[(k + 1) % 3]);
                let want = if a.x == 4.0 && b.x == 4.0 { 0.0 } else { -1.0 };
                assert!(
                    (t.ridge_cos[k] - want).abs() < 1e-6,
                    "triangle {id} edge {k}"
                );
            }
        }
        // A valley along the same line: a seam like any other.
        let valley = field(&|i, _| (i as f32 - 4.0).abs());
        all_bits(&valley, &|a, b| on_border(a, b, 8.0));
        // One cell's diagonal: a ridge when its ends are raised, a valley when the other two
        // corners are.
        let diagonal = |raise: [f32; 4]| Heightfield::new(2, 2, raise.to_vec(), Vec3::ONE).unwrap();
        let up = diagonal([1.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            (up.triangle(0).active_edges, up.triangle(1).active_edges),
            (0b111, 0b111)
        );
        let down = diagonal([0.0, 1.0, 1.0, 0.0]);
        assert_eq!(
            (down.triangle(0).active_edges, down.triangle(1).active_edges),
            (0b011, 0b110)
        );
        let level = diagonal([0.5; 4]);
        assert_eq!(
            (
                level.triangle(0).active_edges,
                level.triangle(1).active_edges
            ),
            (0b011, 0b110)
        );
    }
}
