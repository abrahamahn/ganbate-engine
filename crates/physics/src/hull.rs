//! Convex polyhedra as half-edge meshes: what boxes and hulls collide as.
//!
//! Layout rules every module relies on:
//! - `edges[2k]` and `edges[2k + 1]` are twins (the same edge in opposite directions), so
//!   iterating `edges.iter().step_by(2)` visits each undirected edge once.
//! - Walking `next` from `faces[f].edge` visits the face's boundary counter-clockwise seen from
//!   outside (along the face normal), and returns to the start.
//! - `faces[f].normal` is unit length and points outward; `faces[f].offset == normal · v` for
//!   every vertex `v` of the face (within rounding).
//! - Neighbouring faces are never coplanar ([`ConvexHull::from_points`] merges them,
//!   [`ConvexHull::from_faces`] refuses them), so every edge is a real crease, which the
//!   separating-axis test and the contact clipping need.

use crate::math::{Mat3, Vec3};
use crate::shape::MassProperties;

/// Most vertices a hull may have. Points beyond this are refused by [`ConvexHull::from_points`].
pub const MAX_HULL_VERTICES: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfEdge {
    /// The vertex this half-edge starts at.
    pub origin: u16,
    /// The opposite half-edge (`index ^ 1`).
    pub twin: u16,
    /// The next half-edge round the same face.
    pub next: u16,
    /// The face on this half-edge's left (seen from outside).
    pub face: u16,
}

const _: () = assert!(std::mem::size_of::<HalfEdge>() == 8);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HullFace {
    pub normal: Vec3,
    pub offset: f32,
    /// One half-edge of the face.
    pub edge: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConvexHull {
    pub vertices: Vec<Vec3>,
    pub edges: Vec<HalfEdge>,
    pub faces: Vec<HullFace>,
    /// A point strictly inside (the mean of the vertices).
    pub centroid: Vec3,
    pub(crate) inner_radius: f32,
    pub(crate) outer_face_dist: f32,
    pub(crate) reach: f32,
    pub(crate) max_abs_coord: f32,
    pub(crate) vertex_edge_offsets: Vec<u16>,
    pub(crate) vertex_edges: Vec<u16>,
}

impl ConvexHull {
    /// The box `[-half, half]` as a hull: 8 vertices, 12 edges, 6 faces. Vertex `i` has
    /// coordinates `(±x, ±y, ±z)` with bit 0, 1 and 2 of `i` choosing `+`. Faces are, in order,
    /// −X, +X, −Y, +Y, −Z, +Z.
    pub fn cuboid(half: Vec3) -> ConvexHull {
        Self::parallelepiped(&Mat3::from_diagonal(half))
            .expect("the cuboid's faces are a closed polyhedron")
    }

    /// A box transformed by its three half edges. Its topology is known even when the box is
    /// thinner than the welding tolerance for an unstructured point cloud.
    pub(crate) fn parallelepiped(edges: &Mat3) -> Result<Self, String> {
        let mirrored = edges.determinant() < 0.0;
        let vertices = (0..8)
            .map(|i| {
                let sign = |bit| if i & bit != 0 { 1.0 } else { -1.0 };
                let x = if mirrored { -sign(1) } else { sign(1) };
                edges.mul_vec(Vec3::new(x, sign(2), sign(4)))
            })
            .collect();
        let loops: [&[u32]; 6] = [
            &[0, 4, 6, 2],
            &[1, 3, 7, 5],
            &[0, 1, 5, 4],
            &[2, 6, 7, 3],
            &[0, 2, 3, 1],
            &[4, 5, 7, 6],
        ];
        ConvexHull::from_faces(vertices, &loops)
    }

    /// Builds the half-edge structure from vertex loops, one per face, each counter-clockwise seen
    /// from outside. Every edge must be shared by exactly two faces in opposite directions, and
    /// neighbouring faces must not lie in one plane: a flat side is one loop, not the triangles
    /// of a mesh export ([`ConvexHull::from_points`] merges those itself). Face planes come from
    /// Newell's method over each loop.
    pub fn from_faces(vertices: Vec<Vec3>, loops: &[&[u32]]) -> Result<ConvexHull, String> {
        let hull = ConvexHull::assemble(vertices, loops)?;
        hull.validate()?;
        Ok(hull)
    }

    /// Refuses neighbouring faces that lie in one plane: the module's rule that every edge is a
    /// real crease. A box resting across such a seam gets contact points on one of the two
    /// halves only, and tips over.
    ///
    /// Neighbours share an edge, so lying in one plane is facing the same way with each within
    /// rounding of the other's plane. Both ways: a face
    /// thinner than the rounding across the shared edge (the side of a thin plate) is within it
    /// of any plane through that edge, its big perpendicular neighbour's included.
    fn check_creases(&self) -> Result<(), String> {
        let mut extent = Vec3::ZERO;
        for v in &self.vertices {
            extent = extent.max(v.abs());
        }
        // Every vertex of face `g` within `flat` of face `f`'s plane.
        let on = |f: usize, g: usize| {
            let plane = &self.faces[f];
            // Only errors along this normal affect its plane distance. A long plate's span
            // along X cannot erase its resolved thickness along Y.
            let flat = 3.0 * f32::EPSILON * plane.normal.abs().dot(extent).max(1e-6);
            self.face_vertices(g)
                .iter()
                .all(|v| (plane.normal.dot(*v) - plane.offset).abs() <= flat)
        };
        for e in self.edges.iter().step_by(2) {
            let (f, g) = (e.face as usize, self.edges[e.twin as usize].face as usize);
            if f == g {
                return Err(format!("face {f} meets itself along an edge"));
            }
            if !(on(f, g) && on(g, f)) {
                continue;
            }
            let facing = self.faces[f].normal.dot(self.faces[g].normal);
            if facing > 0.0 {
                return Err(format!(
                    "faces {f} and {g} are coplanar neighbours: give them as one face"
                ));
            }
            if facing < -0.5 {
                return Err(format!(
                    "faces {f} and {g} lie back to back in one plane: the hull is flat"
                ));
            }
        }
        Ok(())
    }

    /// [`ConvexHull::from_faces`] without the final [`ConvexHull::validate`].
    fn assemble(vertices: Vec<Vec3>, loops: &[&[u32]]) -> Result<ConvexHull, String> {
        if vertices.len() < 4 || vertices.len() > MAX_HULL_VERTICES {
            return Err(format!(
                "a hull needs 4 to {MAX_HULL_VERTICES} vertices, not {}",
                vertices.len()
            ));
        }
        if vertices.iter().any(|v| !v.is_finite()) {
            return Err("hull vertex is not finite".into());
        }
        if loops.len() >= u16::MAX as usize {
            return Err("too many faces".into());
        }
        let mut edges: Vec<HalfEdge> = Vec::new();
        // Directed edge (from, to) → half-edge index, kept sorted for a deterministic build.
        let mut directed: std::collections::BTreeMap<(u32, u32), u16> = Default::default();
        let mut faces = Vec::with_capacity(loops.len());
        for (f, lp) in loops.iter().enumerate() {
            if lp.len() < 3 {
                return Err(format!("face {f} has fewer than 3 vertices"));
            }
            let mut first = u16::MAX;
            let mut prev = u16::MAX;
            for j in 0..lp.len() {
                let (a, b) = (lp[j], lp[(j + 1) % lp.len()]);
                if a as usize >= vertices.len() || b as usize >= vertices.len() || a == b {
                    return Err(format!("face {f} has a bad edge {a}→{b}"));
                }
                if directed.contains_key(&(a, b)) {
                    return Err(format!(
                        "edge {a}→{b} is used by two faces in the same direction"
                    ));
                }
                let e = if let Some(&twin) = directed.get(&(b, a)) {
                    twin ^ 1
                } else {
                    if edges.len() + 2 >= u16::MAX as usize {
                        return Err("too many half-edges".into());
                    }
                    let k = edges.len() as u16;
                    let blank = HalfEdge {
                        origin: 0,
                        twin: 0,
                        next: u16::MAX,
                        face: u16::MAX,
                    };
                    edges.push(HalfEdge {
                        origin: a as u16,
                        twin: k + 1,
                        ..blank
                    });
                    edges.push(HalfEdge {
                        origin: b as u16,
                        twin: k,
                        ..blank
                    });
                    k
                };
                directed.insert((a, b), e);
                edges[e as usize].face = f as u16;
                if prev != u16::MAX {
                    edges[prev as usize].next = e;
                } else {
                    first = e;
                }
                prev = e;
            }
            edges[prev as usize].next = first;
            // Sum the face's signed triangle areas relative to its first vertex. The
            // f64 cross products preserve a small crease even far from the local origin.
            let origin = D3::from_vec3(vertices[lp[0] as usize]);
            let mut area = D3::ZERO;
            let mut sum = D3::ZERO;
            for j in 0..lp.len() {
                let p = D3::from_vec3(vertices[lp[j] as usize]);
                let q = D3::from_vec3(vertices[lp[(j + 1) % lp.len()] as usize]);
                area = area + (p - origin).cross(q - origin);
                sum = sum + p;
            }
            let normal = area.unit_or_zero();
            if normal == D3::ZERO {
                return Err(format!("face {f} has no area"));
            }
            faces.push(HullFace {
                normal: normal.to_vec3(),
                offset: (normal.dot(sum) / lp.len() as f64) as f32,
                edge: u32::from(first),
            });
        }
        if edges
            .iter()
            .any(|e| e.face == u16::MAX || e.next == u16::MAX)
        {
            return Err("the faces do not close: some edge has only one face".into());
        }
        let centroid =
            vertices.iter().fold(Vec3::ZERO, |s, &v| s + v) * (1.0 / vertices.len() as f32);
        let mut hull = ConvexHull {
            vertices,
            edges,
            faces,
            centroid,
            inner_radius: 0.0,
            outer_face_dist: 0.0,
            reach: 0.0,
            max_abs_coord: 0.0,
            vertex_edge_offsets: Vec::new(),
            vertex_edges: Vec::new(),
        };
        hull.recompute_invariants();
        Ok(hull)
    }

    fn recompute_invariants(&mut self) {
        let r = self.faces.iter().fold(f32::INFINITY, |r, f| {
            r.min(f.offset - f.normal.dot(self.centroid))
        });
        self.inner_radius = if r.is_finite() { r.max(0.0) } else { 0.0 };
        self.outer_face_dist = self
            .faces
            .iter()
            .fold(0.0f32, |m, f| m.max(f.offset - f.normal.dot(self.centroid)));
        self.reach = self
            .vertices
            .iter()
            .fold(0.0f32, |m, p| m.max((*p - self.centroid).length_squared()))
            .sqrt();
        self.max_abs_coord = self
            .vertices
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs().max_element()));

        let nv = self.vertices.len();
        let mut offsets = vec![0u16; nv + 1];
        for e in &self.edges {
            let o = e.origin as usize;
            if o < nv {
                offsets[o + 1] += 1;
            }
        }
        for v in 0..nv {
            offsets[v + 1] += offsets[v];
        }
        let mut cursor = offsets.clone();
        let mut vertex_edges = vec![0u16; self.edges.len()];
        for (idx, e) in self.edges.iter().enumerate() {
            let o = e.origin as usize;
            if o < nv {
                let pos = cursor[o] as usize;
                vertex_edges[pos] = idx as u16;
                cursor[o] += 1;
            }
        }
        self.vertex_edge_offsets = offsets;
        self.vertex_edges = vertex_edges;
    }

    /// The index of the vertex farthest along `d` (the first of equals).
    #[inline]
    pub fn support_index(&self, d: Vec3) -> usize {
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

    /// Checks the layout rules in the module doc and convexity (every vertex on or behind every
    /// face plane, within a tolerance scaled to the hull's size).
    pub fn validate(&self) -> Result<(), String> {
        if !(4..=MAX_HULL_VERTICES).contains(&self.vertices.len()) || self.edges.len() % 2 != 0 {
            return Err("invalid vertex or half-edge count".into());
        }
        if !self.centroid.is_finite() || self.vertices.iter().any(|v| !v.is_finite()) {
            return Err("hull contains a non-finite vertex or centroid".into());
        }
        let scale = self
            .vertices
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs().max_element()))
            .max(1e-3);
        let tol = 1e-4 * scale;
        for (i, e) in self.edges.iter().enumerate() {
            if e.twin as usize != (i ^ 1)
                || e.origin as usize >= self.vertices.len()
                || e.next as usize >= self.edges.len()
                || e.face as usize >= self.faces.len()
            {
                return Err(format!("half-edge {i} is malformed"));
            }
            let twin = &self.edges[i ^ 1];
            if self.edges[e.next as usize].origin != twin.origin {
                return Err(format!("half-edge {i}'s next does not start where it ends"));
            }
            if self.edges[e.next as usize].face != e.face {
                return Err(format!("half-edge {i}'s next is on another face"));
            }
        }
        for (f, face) in self.faces.iter().enumerate() {
            if face.edge as usize >= self.edges.len() || !face.offset.is_finite() {
                return Err(format!(
                    "face {f} has an invalid first edge or plane offset"
                ));
            }
            if !face.normal.is_finite() || (face.normal.length() - 1.0).abs() > 1e-3 {
                return Err(format!("face {f}'s normal is not unit length"));
            }
            for (i, v) in self.vertices.iter().enumerate() {
                if face.normal.dot(*v) - face.offset > tol {
                    return Err(format!("vertex {i} is in front of face {f}: not convex"));
                }
            }
            // Walk the loop: it must close within the number of half-edges.
            let mut e = face.edge;
            let mut n = 0;
            loop {
                if self.edges[e as usize].face as usize != f {
                    return Err(format!("face {f}'s loop leaves the face"));
                }
                let v = self.vertices[self.edges[e as usize].origin as usize];
                if (face.normal.dot(v) - face.offset).abs() > tol * 10.0 {
                    return Err(format!("face {f} is not planar"));
                }
                e = u32::from(self.edges[e as usize].next);
                n += 1;
                if e == face.edge {
                    break;
                }
                if n > self.edges.len() {
                    return Err(format!("face {f}'s loop does not close"));
                }
            }
        }
        // Euler's formula for a closed convex polyhedron: V − E + F = 2.
        let (v, e, f) = (
            self.vertices.len() as i64,
            (self.edges.len() / 2) as i64,
            self.faces.len() as i64,
        );
        if v - e + f != 2 {
            return Err(format!("V − E + F = {} (should be 2)", v - e + f));
        }
        self.check_creases()
    }

    /// The vertices of face `f`, in loop order.
    pub fn face_vertices(&self, f: usize) -> Vec<Vec3> {
        let mut out = Vec::new();
        let start = self.faces[f].edge;
        let mut e = start;
        loop {
            out.push(self.vertices[self.edges[e as usize].origin as usize]);
            e = u32::from(self.edges[e as usize].next);
            if e == start {
                break;
            }
        }
        out
    }

    /// The convex hull of `points` (quickhull), with coplanar faces merged so that every edge is a
    /// real crease. Refuses fewer than 4 non-coplanar points, non-finite points, and results with
    /// more than [`MAX_HULL_VERTICES`] vertices. Points closer than a tolerance scaled to the input
    /// are welded.
    ///
    /// The tolerance is `3 ε (max |x| + max |y| + max |z|)` (`ε = f32::EPSILON`): points within it
    /// of a face are on the face, faces whose vertices all lie within it of one plane are merged,
    /// and vertices left on only two faces (on the straight edge between them) are dropped. Every
    /// vertex of the result is one of `points`, and every point is inside it or within a few
    /// tolerances outside. The result is the same on every platform: the build runs in `f64`
    /// with only `+ − × ÷ √`, decides which faces a point sees exactly, and breaks ties by index.
    pub fn from_points(points: &[Vec3]) -> Result<ConvexHull, String> {
        if points.len() < 4 {
            return Err(format!(
                "convex hull: needs at least 4 points, got {}",
                points.len()
            ));
        }
        if let Some(i) = points.iter().position(|p| !p.is_finite()) {
            return Err(format!("convex hull: point {i} is not finite"));
        }
        // The tolerance: a few units of rounding at the input's largest coordinates.
        let mut extent = [0.0f64; 3];
        for p in points {
            for (a, e) in extent.iter_mut().enumerate() {
                *e = e.max((p[a] as f64).abs());
            }
        }
        let tol = 3.0 * f32::EPSILON as f64 * (extent[0] + extent[1] + extent[2]).max(1e-6);
        // Weld: keep a point only if no kept point is within `tol` (the first in index order
        // wins), finding neighbours through a grid of `tol`-sized cells.
        let mut cells: std::collections::BTreeMap<[i64; 3], Vec<u32>> = Default::default();
        let mut kept: Vec<usize> = Vec::new();
        let mut pts: Vec<D3> = Vec::new();
        for (i, &p) in points.iter().enumerate() {
            let p = D3::from_vec3(p);
            let cell = [p.x, p.y, p.z].map(|c| (c / tol).floor() as i64);
            let mut near = false;
            'search: for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let key = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                        if let Some(list) = cells.get(&key) {
                            if list
                                .iter()
                                .any(|&k| (pts[k as usize] - p).length_squared() <= tol * tol)
                            {
                                near = true;
                                break 'search;
                            }
                        }
                    }
                }
            }
            if !near {
                cells.entry(cell).or_default().push(pts.len() as u32);
                kept.push(i);
                pts.push(p);
            }
        }
        if pts.len() < 4 {
            return Err(format!(
                "convex hull: needs at least 4 distinct points, got {} after welding",
                pts.len()
            ));
        }
        let simplex = initial_tetrahedron(&pts, tol)?;
        let mut qh = Quickhull {
            pts: &pts,
            tol,
            tris: Vec::new(),
            owner: Default::default(),
        };
        let [i0, i1, i2, i3] = simplex;
        for v in [[i0, i2, i1], [i0, i1, i3], [i1, i2, i3], [i2, i0, i3]] {
            qh.add_tri(v);
        }
        for p in 0..pts.len() as u32 {
            if !simplex.contains(&p) {
                qh.assign(p, [0, 1, 2, 3]);
            }
        }
        qh.run(4 * MAX_HULL_VERTICES)?;
        let mut loops = merge_coplanar(&qh)?;
        // A vertex left on only two faces lies on the straight edge between them: drop it.
        let mut faces_at = vec![0u32; pts.len()];
        for (lp, _) in &loops {
            for &v in lp {
                faces_at[v as usize] += 1;
            }
        }
        for (lp, _) in &mut loops {
            lp.retain(|&v| faces_at[v as usize] >= 3);
            if lp.len() < 3 {
                return Err("convex hull: the points are too degenerate (a face collapsed)".into());
            }
        }
        let mut remap = vec![u32::MAX; pts.len()];
        let mut vertices = Vec::new();
        for (v, &n) in faces_at.iter().enumerate() {
            if n >= 3 {
                remap[v] = vertices.len() as u32;
                vertices.push(points[kept[v]]);
            }
        }
        if vertices.len() > MAX_HULL_VERTICES {
            return Err(format!(
                "convex hull: the hull of these points has {} vertices, more than the \
                 {MAX_HULL_VERTICES} allowed; reduce or simplify the points first",
                vertices.len()
            ));
        }
        // Face planes in double precision (more exact than `from_faces`'s Newell planes).
        let planes: Vec<(Vec3, f32)> = loops
            .iter()
            .map(|(lp, n)| {
                let offset =
                    lp.iter().map(|&v| n.dot(pts[v as usize])).sum::<f64>() / lp.len() as f64;
                (n.to_vec3(), offset as f32)
            })
            .collect();
        let loops: Vec<Vec<u32>> = loops
            .iter()
            .map(|(lp, _)| lp.iter().map(|&v| remap[v as usize]).collect())
            .collect();
        let refs: Vec<&[u32]> = loops.iter().map(|lp| lp.as_slice()).collect();
        let mut hull = ConvexHull::assemble(vertices, &refs)?;
        for (face, (normal, offset)) in hull.faces.iter_mut().zip(planes) {
            face.normal = normal;
            face.offset = offset;
        }
        hull.recompute_invariants();
        hull.validate()
            .map_err(|e| format!("convex hull: the result is not valid: {e}"))?;
        Ok(hull)
    }

    /// Mass, centre of mass and inertia about it for a uniform `density`, by splitting the hull
    /// into tetrahedra from an interior point (exact for any closed polyhedron).
    pub fn mass_properties(&self, density: f32) -> MassProperties {
        // Tetrahedra from the vertex mean `r` to each triangle of each face's fan, summed in
        // double precision about `r`: volume, first moment and the second moment ∫ x xᵀ dV
        // (xx, yy, zz, xy, yz, zx), which for the tetrahedron (0, a, b, c) is
        // det/120 · (a aᵀ + b bᵀ + c cᵀ + s sᵀ) with det = a · (b × c) and s = a + b + c.
        let r = D3::from_vec3(self.centroid);
        let at = |e: u32| D3::from_vec3(self.vertices[self.edges[e as usize].origin as usize]) - r;
        let mut volume = 0.0f64;
        let mut first = D3::ZERO;
        let mut second = [0.0f64; 6];
        for face in &self.faces {
            let a = at(face.edge);
            let mut e = u32::from(self.edges[face.edge as usize].next);
            let mut b = at(e);
            e = u32::from(self.edges[e as usize].next);
            while e != face.edge {
                let c = at(e);
                let det = a.dot(b.cross(c));
                let s = a + b + c;
                volume += det / 6.0;
                first = first + s * (det / 24.0);
                let k = det / 120.0;
                for u in [a, b, c, s] {
                    second[0] += k * u.x * u.x;
                    second[1] += k * u.y * u.y;
                    second[2] += k * u.z * u.z;
                    second[3] += k * u.x * u.y;
                    second[4] += k * u.y * u.z;
                    second[5] += k * u.z * u.x;
                }
                b = c;
                e = u32::from(self.edges[e as usize].next);
            }
        }
        if volume.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return MassProperties {
                mass: 0.0,
                center: self.centroid,
                inertia: Mat3::ZERO,
            };
        }
        // Move the second moment to the centre of mass, then I = ρ (tr C · 1 − C).
        let m = first * (1.0 / volume);
        let rho = density as f64;
        let c = [
            second[0] - volume * m.x * m.x,
            second[1] - volume * m.y * m.y,
            second[2] - volume * m.z * m.z,
            second[3] - volume * m.x * m.y,
            second[4] - volume * m.y * m.z,
            second[5] - volume * m.z * m.x,
        ];
        let (ixx, iyy, izz) = (
            rho * (c[1] + c[2]),
            rho * (c[0] + c[2]),
            rho * (c[0] + c[1]),
        );
        let (ixy, iyz, izx) = (-rho * c[3], -rho * c[4], -rho * c[5]);
        let col = |x: f64, y: f64, z: f64| Vec3::new(x as f32, y as f32, z as f32);
        MassProperties {
            mass: (rho * volume) as f32,
            center: (r + m).to_vec3(),
            inertia: Mat3::from_cols(col(ixx, ixy, izx), col(ixy, iyy, iyz), col(izx, iyz, izz)),
        }
    }
}

/// A vector in `f64`: hull building runs in double precision so that its rounding stays far
/// below the welding and coplanarity tolerance, which is set by the `f32` input.
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

    fn from_vec3(v: Vec3) -> D3 {
        D3 {
            x: v.x as f64,
            y: v.y as f64,
            z: v.z as f64,
        }
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
    /// This vector scaled to unit length, or zero when it has none.
    fn unit_or_zero(self) -> D3 {
        let l = self.length_squared().sqrt();
        if l > 0.0 {
            D3 {
                x: self.x / l,
                y: self.y / l,
                z: self.z / l,
            }
        } else {
            D3::ZERO
        }
    }
    fn to_vec3(self) -> Vec3 {
        Vec3::new(self.x as f32, self.y as f32, self.z as f32)
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

/// `a + b` exactly, as the rounded sum and its rounding error (Knuth's two-sum).
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bv = s - a;
    (s, (a - (s - bv)) + (b - bv))
}

/// `a × b` exactly, as the rounded product and its rounding error (Dekker's product with
/// Veltkamp's split, no fused multiply-add).
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let split = |x: f64| {
        let c = 134_217_729.0 * x;
        let hi = c - (c - x);
        (hi, x - hi)
    };
    let p = a * b;
    let ((ah, al), (bh, bl)) = (split(a), split(b));
    (p, al * bl - (((p - ah * bh) - al * bh) - ah * bl))
}

/// The sign of `((b − a) × (c − a)) · (d − a)`, exactly, for points with `f32` coordinates:
/// positive when `d` is on the side the counter-clockwise triangle `(a, b, c)` faces. It is
/// `[b c d] − [a c d] + [a b d] − [a b c]` (`[u v w] = u · (v × w)`): 24 products of three
/// coordinates, each exact as two doubles (two `f32` multiply exactly in `f64`). A rounded sum
/// decides when it is clear of its error bound; otherwise the terms are summed exactly as a
/// Shewchuk expansion.
fn orient(a: D3, b: D3, c: D3, d: D3) -> f64 {
    // (±x·y, z) for each product x·y·z; x·y is exact.
    let mut terms = [(0.0f64, 0.0f64); 24];
    let mut k = 0;
    for (s, u, v, w) in [
        (1.0, b, c, d),
        (-1.0, a, c, d),
        (1.0, a, b, d),
        (-1.0, a, b, c),
    ] {
        for (t, x, y, z) in [
            (s, u.x, v.y, w.z),
            (-s, u.x, v.z, w.y),
            (s, u.y, v.z, w.x),
            (-s, u.y, v.x, w.z),
            (s, u.z, v.x, w.y),
            (-s, u.z, v.y, w.x),
        ] {
            terms[k] = (t * (x * y), z);
            k += 1;
        }
    }
    // Each rounded product is off by at most one unit (2⁻⁵³ relative) and the sum by 23 more.
    let (sum, size) = terms.iter().fold((0.0, 0.0), |(s, m), &(xy, z)| {
        let t: f64 = xy * z;
        (s + t, m + t.abs())
    });
    if sum.abs() > 16.0 * f64::EPSILON * size {
        return sum;
    }
    // Exactly: a nonoverlapping expansion, smallest component first; its sign is the last one's.
    let mut e: Vec<f64> = Vec::with_capacity(48);
    for &(xy, z) in &terms {
        let (hi, lo) = two_product(xy, z);
        for part in [lo, hi] {
            let mut q = part;
            let mut grown = Vec::with_capacity(e.len() + 1);
            for &x in &e {
                let (s, err) = two_sum(q, x);
                if err != 0.0 {
                    grown.push(err);
                }
                q = s;
            }
            if q != 0.0 {
                grown.push(q);
            }
            e = grown;
        }
    }
    e.last().copied().unwrap_or(0.0)
}

/// A directed edge `(from, to, tag)`; on a horizon the tag is the triangle beyond the edge.
type LoopEdge = (u32, u32, u32);

/// Orders directed edges into one closed loop starting at the smallest `from`. `None` when they
/// do not form exactly one simple loop (a vertex left twice, or several loops).
fn order_loop(edges: &[LoopEdge]) -> Option<Vec<LoopEdge>> {
    let mut sorted = edges.to_vec();
    sorted.sort_unstable();
    if sorted.is_empty() || sorted.windows(2).any(|w| w[0].0 == w[1].0) {
        return None;
    }
    let mut out = Vec::with_capacity(sorted.len());
    let mut i = 0;
    loop {
        out.push(sorted[i]);
        i = sorted.binary_search_by_key(&sorted[i].1, |e| e.0).ok()?;
        if i == 0 {
            break;
        }
        if out.len() >= sorted.len() {
            return None;
        }
    }
    (out.len() == sorted.len()).then_some(out)
}

/// Four points spanning a tetrahedron, as large as cheaply found: the extremes along the axis
/// of largest extent, the point farthest from their line, and the point farthest from the plane
/// of those three (the first of equals each time). Fails when the points are (nearly) collinear
/// or coplanar.
fn initial_tetrahedron(pts: &[D3], tol: f64) -> Result<[u32; 4], String> {
    let coord = |p: D3, a: usize| [p.x, p.y, p.z][a];
    let (mut lo, mut hi) = ([0usize; 3], [0usize; 3]);
    for (i, &p) in pts.iter().enumerate() {
        for a in 0..3 {
            if coord(p, a) < coord(pts[lo[a]], a) {
                lo[a] = i;
            }
            if coord(p, a) > coord(pts[hi[a]], a) {
                hi[a] = i;
            }
        }
    }
    let size = |a: usize| coord(pts[hi[a]], a) - coord(pts[lo[a]], a);
    let mut axis = 0;
    for a in 1..3 {
        if size(a) > size(axis) {
            axis = a;
        }
    }
    let (i0, i1) = (lo[axis], hi[axis]);
    let (p0, p1) = (pts[i0], pts[i1]);
    let dir = p1 - p0;
    let farthest = |f: &dyn Fn(D3) -> f64| {
        let (mut best, mut best_d) = (0, f64::NEG_INFINITY);
        for (i, &p) in pts.iter().enumerate() {
            let d = f(p);
            if d > best_d {
                best = i;
                best_d = d;
            }
        }
        (best, best_d)
    };
    let (i2, d2) = farthest(&|p| (p - p0).cross(dir).length_squared());
    if d2 <= tol * tol * dir.length_squared() {
        return Err("convex hull: the points are (nearly) collinear".into());
    }
    let n = dir.cross(pts[i2] - p0).unit_or_zero();
    let (i3, d3) = farthest(&|p| n.dot(p - p0).abs());
    if d3 <= tol {
        return Err("convex hull: the points are (nearly) coplanar".into());
    }
    // Wind the base so that the fourth point is behind it.
    let (i1, i2) = if n.dot(pts[i3] - p0) > 0.0 {
        (i1, i2)
    } else {
        (i2, i1)
    };
    Ok([i0 as u32, i1 as u32, i2 as u32, i3 as u32])
}

/// A triangle of the hull while quickhull builds it, counter-clockwise seen from outside.
struct Tri {
    v: [u32; 3],
    /// `(v1 − v0) × (v2 − v0)`: the outward normal times twice the area.
    area: D3,
    normal: D3,
    offset: f64,
    alive: bool,
    /// Points more than the tolerance in front of this triangle and not yet on the hull.
    outside: Vec<u32>,
    /// The farthest of `outside` (the first of equals) and its distance; `u32::MAX` when empty.
    far: u32,
    far_dist: f64,
}

/// Quickhull over welded points: a triangulated hull, exactly convex since which triangles a
/// point sees is decided exactly. Points wait in the outside set of the triangle they are
/// farthest in front of; only those more than `tol` out are added.
struct Quickhull<'a> {
    pts: &'a [D3],
    tol: f64,
    tris: Vec<Tri>,
    /// Directed edge `(from, to)` → the live triangle that has it.
    owner: std::collections::BTreeMap<(u32, u32), u32>,
}

impl Quickhull<'_> {
    fn add_tri(&mut self, v: [u32; 3]) -> u32 {
        let [a, b, c] = v.map(|i| self.pts[i as usize]);
        let area = (b - a).cross(c - a);
        let normal = area.unit_or_zero();
        let offset = (normal.dot(a) + normal.dot(b) + normal.dot(c)) * (1.0 / 3.0);
        let t = self.tris.len() as u32;
        for k in 0..3 {
            self.owner.insert((v[k], v[(k + 1) % 3]), t);
        }
        self.tris.push(Tri {
            v,
            area,
            normal,
            offset,
            alive: true,
            outside: Vec::new(),
            far: u32::MAX,
            far_dist: 0.0,
        });
        t
    }

    fn kill_tri(&mut self, t: u32) -> Vec<u32> {
        let tri = &mut self.tris[t as usize];
        tri.alive = false;
        let v = tri.v;
        for k in 0..3 {
            self.owner.remove(&(v[k], v[(k + 1) % 3]));
        }
        std::mem::take(&mut self.tris[t as usize].outside)
    }

    fn dist(&self, t: u32, p: u32) -> f64 {
        let tri = &self.tris[t as usize];
        tri.normal.dot(self.pts[p as usize]) - tri.offset
    }

    /// The triangle across edge `k` (from `v[k]` to `v[k + 1]`) of triangle `t`.
    fn neighbour(&self, t: u32, k: usize) -> u32 {
        let v = self.tris[t as usize].v;
        self.owner[&(v[(k + 1) % 3], v[k])]
    }

    /// Puts point `p` in the outside set of the one of `tris` it is farthest in front of (the
    /// first of equals). Returns false, and drops the point, when it is in front of none: it is
    /// inside the hull, and stays inside as the hull grows.
    fn assign(&mut self, p: u32, tris: impl IntoIterator<Item = u32>) -> bool {
        let (mut best, mut best_d) = (u32::MAX, 0.0);
        for t in tris {
            let d = self.dist(t, p);
            if d > best_d {
                best = t;
                best_d = d;
            }
        }
        if best == u32::MAX {
            return false;
        }
        let tri = &mut self.tris[best as usize];
        tri.outside.push(p);
        if tri.far == u32::MAX || best_d > tri.far_dist || (best_d == tri.far_dist && p < tri.far) {
            tri.far = p;
            tri.far_dist = best_d;
        }
        true
    }

    /// Re-files every waiting point under the live triangle it is farthest in front of. A point
    /// no more than `tol` in front of its own triangle may be further in front of one built
    /// since; returns whether any point is now more than `tol` out.
    fn refile(&mut self) -> bool {
        let live: Vec<u32> = (0..self.tris.len() as u32)
            .filter(|&t| self.tris[t as usize].alive)
            .collect();
        let mut waiting = Vec::new();
        for &t in &live {
            let tri = &mut self.tris[t as usize];
            waiting.append(&mut tri.outside);
            tri.far = u32::MAX;
            tri.far_dist = 0.0;
        }
        waiting.sort_unstable();
        for p in waiting {
            self.assign(p, live.iter().copied());
        }
        live.iter()
            .any(|&t| self.tris[t as usize].far_dist > self.tol)
    }

    /// Whether point `p` is strictly in front of triangle `t`, decided exactly.
    fn sees(&self, t: u32, p: u32) -> bool {
        let [a, b, c] = self.tris[t as usize].v.map(|i| self.pts[i as usize]);
        orient(a, b, c, self.pts[p as usize]) > 0.0
    }

    /// The triangles `eye` sees, found from `start` (which it sees), and the horizon round them
    /// as directed edges `(from, to, triangle beyond)` in loop order. With visibility decided
    /// exactly, the visible triangles form a disc and no new triangle is degenerate: a point
    /// in a triangle's plane does not see it, and a point on an edge's line sees neither side.
    fn horizon(&self, start: u32, eye: u32) -> Result<(Vec<bool>, Vec<LoopEdge>), String> {
        let n = self.tris.len();
        let mut visible = vec![false; n];
        let mut seen = vec![false; n];
        visible[start as usize] = true;
        seen[start as usize] = true;
        let mut stack = vec![start];
        let mut edges = Vec::new();
        while let Some(t) = stack.pop() {
            for k in 0..3 {
                let g = self.neighbour(t, k);
                if !seen[g as usize] {
                    seen[g as usize] = true;
                    if self.sees(g, eye) {
                        visible[g as usize] = true;
                        stack.push(g);
                    }
                }
                if !visible[g as usize] {
                    let v = self.tris[t as usize].v;
                    edges.push((v[k], v[(k + 1) % 3], g));
                }
            }
        }
        let cycle =
            order_loop(&edges).ok_or("convex hull: internal error: the horizon is not one loop")?;
        Ok((visible, cycle))
    }

    /// Adds the farthest waiting point (more than `tol` out) until none is left. Fails when more
    /// than `limit` points have been added.
    fn run(&mut self, limit: usize) -> Result<(), String> {
        let mut added = 4;
        loop {
            let mut start = u32::MAX;
            let mut best = self.tol;
            for (t, tri) in self.tris.iter().enumerate() {
                if tri.alive && tri.far != u32::MAX && tri.far_dist > best {
                    best = tri.far_dist;
                    start = t as u32;
                }
            }
            if start == u32::MAX {
                if self.refile() {
                    continue;
                }
                return Ok(());
            }
            let eye = self.tris[start as usize].far;
            if !self.sees(start, eye) {
                // Only a sliver's rounded plane can put a point it does not face more than the
                // tolerance in front of it: such a point is not outside; drop it.
                let tri = &mut self.tris[start as usize];
                tri.outside.retain(|&p| p != eye);
                let (mut far, mut far_dist) = (u32::MAX, 0.0);
                for &p in &tri.outside {
                    let d = tri.normal.dot(self.pts[p as usize]) - tri.offset;
                    if far == u32::MAX || d > far_dist || (d == far_dist && p < far) {
                        far = p;
                        far_dist = d;
                    }
                }
                tri.far = far;
                tri.far_dist = far_dist;
                continue;
            }
            added += 1;
            if added > limit {
                return Err(format!(
                    "convex hull: the points have far more than {MAX_HULL_VERTICES} extreme \
                     points; reduce or simplify them first"
                ));
            }
            let (visible, cycle) = self.horizon(start, eye)?;
            let mut orphans = Vec::new();
            for (t, &vis) in visible.iter().enumerate() {
                if vis {
                    orphans.extend(self.kill_tri(t as u32));
                }
            }
            let new: Vec<u32> = cycle
                .iter()
                .map(|&(a, b, _)| self.add_tri([a, b, eye]))
                .collect();
            for p in orphans {
                if p != eye {
                    self.assign(p, new.iter().copied());
                }
            }
        }
    }
}

/// Merges the triangles of a finished quickhull into faces: neighbouring groups are merged,
/// flattest pairs first, while every vertex of the union stays within `tol` of one plane (their
/// area-weighted plane, or either group's own, which keeps a sliver's rounded normal from
/// tilting a large face). Returns each face's vertex loop, counter-clockwise from outside, and
/// the unit normal of the plane its vertices are within `tol` of.
fn merge_coplanar(qh: &Quickhull) -> Result<Vec<(Vec<u32>, D3)>, String> {
    let live: Vec<u32> = (0..qh.tris.len() as u32)
        .filter(|&t| qh.tris[t as usize].alive)
        .collect();
    let mut index = vec![usize::MAX; qh.tris.len()];
    for (i, &t) in live.iter().enumerate() {
        index[t as usize] = i;
    }
    let tri = |i: usize| &qh.tris[live[i] as usize];
    let across = |i: usize, k: usize| index[qh.neighbour(live[i], k) as usize];
    let m = live.len();
    // Each triangle's group is named by its smallest member, so groups merge deterministically.
    let mut group: Vec<usize> = (0..m).collect();
    let mut members: Vec<Vec<usize>> = (0..m).map(|i| vec![i]).collect();
    let mut verts: Vec<Vec<u32>> = (0..m)
        .map(|i| {
            let mut v = tri(i).v.to_vec();
            v.sort_unstable();
            v
        })
        .collect();
    let mut area: Vec<D3> = (0..m).map(|i| tri(i).area).collect();
    let mut normal: Vec<D3> = (0..m).map(|i| tri(i).normal).collect();
    let mut hull_verts: Vec<u32> = verts.iter().flatten().copied().collect();
    hull_verts.sort_unstable();
    hull_verts.dedup();
    let mut pairs = Vec::new();
    for i in 0..m {
        for k in 0..3 {
            let j = across(i, k);
            if i < j {
                pairs.push((tri(i).normal.dot(tri(j).normal), i, j));
            }
        }
    }
    pairs.sort_by(|p, q| q.0.total_cmp(&p.0).then((p.1, p.2).cmp(&(q.1, q.2))));
    // Repeat until nothing merges: a pair refused while a group was small (and its plane
    // uncertain) may be accepted once the group has grown.
    let mut merged = true;
    while merged {
        merged = false;
        for &(_, i, j) in &pairs {
            let (a, b) = (group[i].min(group[j]), group[i].max(group[j]));
            if a == b {
                continue;
            }
            // Flat: the union within `tol` of the plane, which must still support the hull (no
            // vertex more than `tol` in front), or a slight tilt could cut off a far corner.
            let flat = |n: D3| {
                let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
                for &v in verts[a].iter().chain(&verts[b]) {
                    let s = n.dot(qh.pts[v as usize]);
                    lo = lo.min(s);
                    hi = hi.max(s);
                }
                let top = (hi + lo) * 0.5 + qh.tol;
                hi - lo <= 2.0 * qh.tol
                    && hull_verts.iter().all(|&v| n.dot(qh.pts[v as usize]) <= top)
            };
            let candidates = [(area[a] + area[b]).unit_or_zero(), normal[a], normal[b]];
            let Some(&n) = candidates.iter().find(|&&n| flat(n)) else {
                continue;
            };
            merged = true;
            normal[a] = n;
            let moved = std::mem::take(&mut members[b]);
            for &t in &moved {
                group[t] = a;
            }
            members[a].extend(moved);
            let moved = std::mem::take(&mut verts[b]);
            verts[a].extend(moved);
            verts[a].sort_unstable();
            verts[a].dedup();
            area[a] = area[a] + area[b];
        }
    }
    let mut loops = Vec::new();
    for (a, group_members) in members.iter().enumerate() {
        let mut edges = Vec::new();
        for &i in group_members {
            let v = tri(i).v;
            for k in 0..3 {
                if group[across(i, k)] != a {
                    edges.push((v[k], v[(k + 1) % 3], 0));
                }
            }
        }
        if group_members.is_empty() {
            continue;
        }
        let cycle = order_loop(&edges)
            .ok_or("convex hull: merging coplanar triangles did not give a simple face")?;
        loops.push((cycle.iter().map(|e| e.0).collect(), normal[a]));
    }
    Ok(loops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cuboid_is_a_valid_closed_hull() {
        let h = ConvexHull::cuboid(Vec3::new(1.0, 2.0, 3.0));
        h.validate().unwrap();
        assert_eq!((h.vertices.len(), h.edges.len(), h.faces.len()), (8, 24, 6));
        let expect = [-Vec3::X, Vec3::X, -Vec3::Y, Vec3::Y, -Vec3::Z, Vec3::Z];
        for (f, n) in expect.iter().enumerate() {
            assert!((h.faces[f].normal - *n).length() < 1e-6, "face {f}");
        }
        assert!((h.faces[5].offset - 3.0).abs() < 1e-6);
        assert_eq!(
            h.vertices[h.support_index(Vec3::new(1.0, -1.0, 1.0))],
            Vec3::new(1.0, -2.0, 3.0)
        );
        assert_eq!(h.face_vertices(1).len(), 4);
    }

    #[test]
    fn a_hand_built_hull_must_have_real_creases_before_registering_its_collider() {
        let vertices = ConvexHull::cuboid(Vec3::ONE).vertices;
        let split = ConvexHull::assemble(
            vertices,
            &[
                &[0, 4, 6, 2],
                &[1, 3, 7, 5],
                &[0, 1, 5, 4],
                &[2, 6, 7],
                &[2, 7, 3],
                &[0, 2, 3, 1],
                &[4, 5, 7, 6],
            ],
        )
        .unwrap();
        assert!(split.validate().unwrap_err().contains("coplanar"));
        let mut world = crate::PhysicsWorld::default();
        let body = world.create_body(&crate::BodyDesc::default()).unwrap();
        let collider = crate::ColliderDesc::new(crate::Shape::Hull(std::sync::Arc::new(split)));
        assert!(
            world
                .add_collider(body, &collider)
                .unwrap_err()
                .contains("coplanar")
        );
    }

    #[test]
    fn validating_invalid_half_edge_indices_returns_an_error() {
        let hull = ConvexHull::cuboid(Vec3::ONE);
        let mut broken = hull.clone();
        broken.edges.pop();
        assert!(broken.validate().is_err());
        let mut broken = hull.clone();
        broken.edges[0].origin = u16::MAX;
        assert!(broken.validate().is_err());
        let mut broken = hull;
        broken.faces[0].edge = u32::MAX;
        assert!(broken.validate().is_err());
    }

    #[test]
    fn orientation_is_exact_for_f32_points() {
        // Integer points near 2²³ a short way apart, many exactly coplanar: the rounded sum of
        // the 24 products cancels catastrophically, so the exact expansion decides.
        let mut s = 12345u64;
        let mut next = |range: i64| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as i64 % (2 * range + 1) - range
        };
        let mut exact_zeros = 0;
        for _ in 0..3000 {
            let a = [next(1 << 23), next(1 << 23), next(1 << 23)];
            let e1 = [next(1 << 9), next(1 << 9), next(1 << 9)];
            let e2 = [next(1 << 9), next(1 << 9), next(1 << 9)];
            let (k1, k2) = (next(3), next(3));
            let b: [i64; 3] = std::array::from_fn(|i| a[i] + e1[i]);
            let c: [i64; 3] = std::array::from_fn(|i| a[i] + e2[i]);
            let mut d: [i64; 3] = std::array::from_fn(|i| a[i] + k1 * e1[i] + k2 * e2[i]);
            d[(next(1) + 1) as usize] += next(1);
            let i128v = |p: [i64; 3], q: [i64; 3]| -> [i128; 3] {
                std::array::from_fn(|i| (p[i] - q[i]) as i128)
            };
            let (u, v, w) = (i128v(b, a), i128v(c, a), i128v(d, a));
            let det = (u[1] * v[2] - u[2] * v[1]) * w[0]
                + (u[2] * v[0] - u[0] * v[2]) * w[1]
                + (u[0] * v[1] - u[1] * v[0]) * w[2];
            let pt = |p: [i64; 3]| D3::from_vec3(Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32));
            let got = orient(pt(a), pt(b), pt(c), pt(d));
            let sign = |x: f64| (x > 0.0) as i32 - (x < 0.0) as i32;
            assert_eq!(sign(got), det.signum() as i32, "{a:?} {b:?} {c:?} {d:?}");
            exact_zeros += (det == 0) as i32;
        }
        assert!(exact_zeros > 300, "{exact_zeros}");
    }

    #[test]
    fn a_box_from_its_corners_and_inner_points_is_the_cuboid() {
        let half = Vec3::new(1.0, 2.0, 0.5);
        let cuboid = ConvexHull::cuboid(half);
        let mut points = vec![Vec3::ZERO, half * 0.5, Vec3::new(1.0, 0.0, 0.0)];
        points.extend(cuboid.vertices.iter().rev());
        let hull = ConvexHull::from_points(&points).unwrap();
        assert_eq!((hull.vertices.len(), hull.faces.len()), (8, 6));
        let mut got = hull.vertices.clone();
        let mut want = cuboid.vertices.clone();
        let key = |v: &Vec3| (v.x.to_bits(), v.y.to_bits(), v.z.to_bits());
        got.sort_by_key(key);
        want.sort_by_key(key);
        assert_eq!(got, want);
        for f in &hull.faces {
            assert!(cuboid.faces.iter().any(
                |g| (g.normal - f.normal).length() < 1e-6 && (g.offset - f.offset).abs() < 1e-6
            ));
        }
        let (a, b) = (hull.mass_properties(3.0), cuboid.mass_properties(3.0));
        assert!((a.mass - 3.0 * 8.0).abs() < 1e-4 && (a.mass - b.mass).abs() < 1e-4);
        assert!(a.center.length() < 1e-6);
        let expect = Vec3::new(4.0 + 0.25, 1.0 + 0.25, 1.0 + 4.0) * (a.mass / 3.0);
        assert!((a.inertia.diagonal() - expect).length() < 1e-4 * expect.length());
        assert!((a.inertia.cols[0].y.abs() + a.inertia.cols[1].z.abs()) < 1e-5);
    }

    #[test]
    fn coplanar_and_collinear_points_are_refused() {
        let square = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(1.0, 1.0, 0.0)];
        let e = ConvexHull::from_points(&square).unwrap_err();
        assert!(e.contains("coplanar"), "{e}");
        let line: Vec<Vec3> = (0..5).map(|i| Vec3::ONE * i as f32).collect();
        let e = ConvexHull::from_points(&line).unwrap_err();
        assert!(e.contains("collinear"), "{e}");
    }

    #[test]
    fn open_or_inside_out_faces_are_refused() {
        let v = ConvexHull::cuboid(Vec3::ONE).vertices;
        // Five faces of a box do not close.
        let loops: [&[u32]; 5] = [
            &[0, 4, 6, 2],
            &[1, 3, 7, 5],
            &[0, 1, 5, 4],
            &[2, 6, 7, 3],
            &[0, 2, 3, 1],
        ];
        assert!(ConvexHull::from_faces(v.clone(), &loops).is_err());
        // A face wound the wrong way faces inward.
        let loops: [&[u32]; 6] = [
            &[0, 2, 6, 4],
            &[1, 3, 7, 5],
            &[0, 1, 5, 4],
            &[2, 6, 7, 3],
            &[0, 2, 3, 1],
            &[4, 5, 7, 6],
        ];
        assert!(ConvexHull::from_faces(v, &loops).is_err());
    }
}
