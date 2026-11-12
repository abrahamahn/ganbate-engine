use super::*;
// ---------------------------------------------------------------------------------------------
// Spheres and capsules: a point or a segment plus a radius.

/// One contact between a core point of A at `pa` (radius `ra`) and a surface point or core
/// point of B at `pb` (radius `rb`), `n` the unit normal from A to B.
pub(super) fn single(n: Vec3, pa: Vec3, ra: f32, pb: Vec3, rb: f32, id: u32) -> Manifold {
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

pub(super) fn sphere_sphere(ca: Vec3, ra: f32, cb: Vec3, rb: f32, spec: f32) -> Option<Manifold> {
    let (n, d) = (cb - ca).normalize_and_length();
    if d - ra - rb > spec {
        return None;
    }
    // Coincident centres: any direction separates them equally well.
    let n = if d > 0.0 { n } else { Vec3::Y };
    Some(single(n, ca, ra, cb, rb, KIND_SPHERE))
}

pub(super) fn sphere_capsule(
    c: Vec3,
    ra: f32,
    b0: Vec3,
    b1: Vec3,
    rb: f32,
    spec: f32,
) -> Option<Manifold> {
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

pub(super) fn capsule_capsule(
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
pub(super) fn sphere_poly(
    c: Vec3,
    r: f32,
    poly: &Poly,
    iso_p: &Iso,
    spec: f32,
) -> Option<Manifold> {
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
pub(super) fn capsule_poly(
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
pub(super) fn parallel_manifold(
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
pub(super) fn segment_face_points(
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
pub(super) fn minkowski_face(a: Vec3, b: Vec3, b_x_a: Vec3, c: Vec3, d: Vec3, d_x_c: Vec3) -> bool {
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
pub(super) fn on_arc(n: Vec3, a: Vec3, b: Vec3) -> bool {
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

pub(super) fn poly_poly(pa: &Poly, pb: &Poly, rel: &Frame, spec: f32) -> Option<Manifold> {
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
