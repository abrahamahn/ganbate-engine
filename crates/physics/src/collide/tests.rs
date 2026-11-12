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
        crate::math::Quat::from_axis_angle(axis, (rng.next() * 2.0 - 1.0) * std::f32::consts::PI)
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
