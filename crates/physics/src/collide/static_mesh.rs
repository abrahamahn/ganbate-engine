use super::*;
// ---------------------------------------------------------------------------------------------
// Convex shapes against triangles (one-sided, with active edges), in the convex's frame. Every
// normal here points from the triangle to the convex.

/// A convex shape in its own frame.
pub(super) enum Core<'a> {
    Sphere(f32),
    Capsule(f32, f32),
    Poly(Poly<'a>),
}

/// The contact of a convex with one triangle, and whether it is *borrowed*: its normal is a
/// neighbour's (it leans past a ridge, or out over a seam, at the triangle's feature), so it
/// counts only when no triangle touches with a normal of its own (see `collide_static`). The
/// convex's centre may be up to `behind` behind the triangle (which then also counts only
/// then).
pub(super) fn convex_triangle(
    core: &Core,
    tri: &TriPoly,
    spec: f32,
    behind: f32,
) -> Option<(Manifold, bool)> {
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
pub(super) fn sphere_triangle(
    r: f32,
    tri: &TriPoly,
    spec: f32,
    behind: f32,
) -> Option<(Manifold, bool)> {
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
pub(super) fn capsule_triangle(
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

pub(super) fn poly_triangle(
    poly: &Poly,
    tri: &TriPoly,
    spec: f32,
    behind: f32,
) -> Option<(Manifold, bool)> {
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
pub(super) enum StaticGeometry<'a> {
    Mesh(&'a TriangleMesh),
    Heightfield(&'a Heightfield),
}

impl StaticGeometry<'_> {
    pub(super) fn of(s: &Shape) -> Option<StaticGeometry<'_>> {
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
pub(super) fn collide_static(
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
pub(super) struct MeshContacts {
    list: [MeshGroup; MAX_MESH_MANIFOLDS],
    len: usize,
}

impl MeshContacts {
    pub(super) fn add(&mut self, m: Manifold) {
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
    pub(super) fn finish(
        self,
        iso: &Iso,
        flip: bool,
        out: &mut [Manifold; MAX_MESH_MANIFOLDS],
    ) -> usize {
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
pub(super) const PATCH_TOL: f32 = 1e-5;
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
