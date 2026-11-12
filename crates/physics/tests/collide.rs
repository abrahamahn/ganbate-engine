//! Contact manifolds between every pair of shapes, checked against analytic answers and brute
//! force.

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};
use std::sync::Arc;

use ganbate_physics::collide::{MAX_MANIFOLD_POINTS, MAX_MESH_MANIFOLDS, Manifold, collide};
use ganbate_physics::hull::ConvexHull;
use ganbate_physics::math::{Iso, Quat, Vec3};
use ganbate_physics::mesh::{Heightfield, TriangleMesh};
use ganbate_physics::shape::Shape;

// ---------------------------------------------------------------------------------------------
// Helpers.

fn run(a: &Shape, ia: &Iso, b: &Shape, ib: &Iso, spec: f32) -> Vec<Manifold> {
    let mut out = Vec::new();
    collide(a, ia, b, ib, spec, &mut out);
    check_contract(&out, spec);
    out
}

/// The single manifold two convex shapes must make.
fn one(a: &Shape, ia: &Iso, b: &Shape, ib: &Iso, spec: f32) -> Manifold {
    let out = run(a, ia, b, ib, spec);
    assert_eq!(out.len(), 1, "expected one manifold: {out:?}");
    out[0].clone()
}

fn none(a: &Shape, ia: &Iso, b: &Shape, ib: &Iso, spec: f32) {
    let out = run(a, ia, b, ib, spec);
    assert!(out.is_empty(), "expected no contact: {out:?}");
}

/// What every result must satisfy (module contract).
fn check_contract(out: &[Manifold], spec: f32) {
    let mut all_ids = Vec::new();
    for m in out {
        assert!(
            m.count >= 1 && usize::from(m.count) <= MAX_MANIFOLD_POINTS,
            "{m:?}"
        );
        assert!(m.normal.is_finite(), "{m:?}");
        assert!(
            (m.normal.length() - 1.0).abs() < 1e-5,
            "normal not unit: {m:?}"
        );
        for (i, p) in m.points().iter().enumerate() {
            assert!(p.point.is_finite() && p.separation.is_finite(), "{m:?}");
            assert!(
                p.separation <= spec + 1e-6,
                "point beyond the speculative distance: {m:?}"
            );
            assert_ne!(p.id, 0, "{m:?}");
            assert!(
                m.points()[..i].iter().all(|q| q.id != p.id),
                "duplicate id: {m:?}"
            );
            all_ids.push(p.id);
        }
    }
    let n = all_ids.len();
    all_ids.sort_unstable();
    all_ids.dedup();
    assert_eq!(all_ids.len(), n, "ids repeat across manifolds: {out:?}");
}

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}
fn at(x: f32, y: f32, z: f32) -> Iso {
    Iso::from_position(v(x, y, z))
}
fn turned(p: Vec3, axis: Vec3, angle: f32) -> Iso {
    Iso::new(p, Quat::from_axis_angle(axis, angle))
}
fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}
fn close_v(a: Vec3, b: Vec3, tol: f32) -> bool {
    (a - b).length() <= tol
}

fn sphere(radius: f32) -> Shape {
    Shape::Sphere { radius }
}
fn capsule(half_height: f32, radius: f32) -> Shape {
    Shape::Capsule {
        half_height,
        radius,
    }
}
fn cuboid(x: f32, y: f32, z: f32) -> Shape {
    Shape::Box {
        half_extents: v(x, y, z),
    }
}
fn hull_cuboid(x: f32, y: f32, z: f32) -> Shape {
    Shape::Hull(Arc::new(ConvexHull::cuboid(v(x, y, z))))
}

/// A hull from triangles of its surface (wound either way: fixed to face outward). Coplanar
/// neighbours are not merged, so only use shapes whose faces are triangles.
fn hull_from_triangles(vertices: Vec<Vec3>, triangles: &[[u32; 3]]) -> Shape {
    let c = vertices.iter().fold(Vec3::ZERO, |s, &p| s + p) * (1.0 / vertices.len() as f32);
    let loops: Vec<[u32; 3]> = triangles
        .iter()
        .map(|&[a, b, d]| {
            let (pa, pb, pd) = (
                vertices[a as usize],
                vertices[b as usize],
                vertices[d as usize],
            );
            if (pb - pa).cross(pd - pa).dot(pa - c) < 0.0 {
                [a, d, b]
            } else {
                [a, b, d]
            }
        })
        .collect();
    let refs: Vec<&[u32]> = loops.iter().map(|l| &l[..]).collect();
    Shape::Hull(Arc::new(ConvexHull::from_faces(vertices, &refs).unwrap()))
}

/// The regular octahedron `|x| + |y| + |z| <= s`.
fn octahedron(s: f32) -> Shape {
    let vertices = vec![
        v(s, 0.0, 0.0),
        v(-s, 0.0, 0.0),
        v(0.0, s, 0.0),
        v(0.0, -s, 0.0),
        v(0.0, 0.0, s),
        v(0.0, 0.0, -s),
    ];
    let mut tris = Vec::new();
    for x in [0u32, 1] {
        for y in [2u32, 3] {
            for z in [4u32, 5] {
                tris.push([x, y, z]);
            }
        }
    }
    hull_from_triangles(vertices, &tris)
}

/// A tetrahedron with corners at alternate corners of the cube `[-s, s]³`.
fn tetrahedron(s: f32) -> Shape {
    let vertices = vec![v(s, s, s), v(s, -s, -s), v(-s, s, -s), v(-s, -s, s)];
    hull_from_triangles(vertices, &[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
}

/// A small deterministic generator for the randomized tests.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
    fn vec(&mut self, r: f32) -> Vec3 {
        v(self.range(-r, r), self.range(-r, r), self.range(-r, r))
    }
    fn rotation(&mut self) -> Quat {
        let axis = self.vec(1.0).try_normalize().unwrap_or(Vec3::Y);
        Quat::from_axis_angle(axis, self.range(-3.2, 3.2))
    }
}

/// The distance from `p` to the box `[-h, h]` (0 inside).
fn box_distance(p: Vec3, h: Vec3) -> f32 {
    let q = p.abs() - h;
    q.max(Vec3::ZERO).length()
}

// ---------------------------------------------------------------------------------------------
// Spheres and capsules.

#[test]
fn spheres_touch_at_the_exact_distance_between_their_surfaces() {
    let (a, b) = (sphere(1.0), sphere(0.5));
    let m = one(&a, &at(0.0, 0.0, 0.0), &b, &at(0.0, 0.0, 1.6), 0.2);
    assert!(close_v(m.normal, Vec3::Z, 1e-6));
    assert_eq!(m.count, 1);
    assert!(close(m.points[0].separation, 0.1, 1e-6));
    // Midway between the surfaces at z = 1 and z = 1.1.
    assert!(close_v(m.points[0].point, v(0.0, 0.0, 1.05), 1e-6));
    let m = one(&a, &at(1.0, 2.0, 3.0), &b, &at(1.0, 2.0 - 1.2, 3.0), 0.0);
    assert!(close_v(m.normal, -Vec3::Y, 1e-6));
    assert!(close(m.points[0].separation, -0.3, 1e-6));
    none(&a, &at(0.0, 0.0, 0.0), &b, &at(0.0, 0.0, 1.6), 0.05);
}

#[test]
fn a_sphere_meets_a_capsule_at_the_nearest_point_of_its_core() {
    let (s, c) = (sphere(0.5), capsule(1.0, 0.25));
    // Beside the core.
    let m = one(&s, &at(1.0, 0.3, 0.0), &c, &Iso::IDENTITY, 0.5);
    assert!(close_v(m.normal, -Vec3::X, 1e-6));
    assert!(close(m.points[0].separation, 0.25, 1e-6));
    assert!(close_v(m.points[0].point, v(0.375, 0.3, 0.0), 1e-6));
    // Beyond the top of the core: the round end.
    let m = one(&s, &at(0.0, 1.0 + 0.6, 0.8), &c, &Iso::IDENTITY, 0.5);
    let d = (0.6f32 * 0.6 + 0.8 * 0.8).sqrt();
    assert!(close(m.points[0].separation, d - 0.75, 1e-6));
    assert!(close_v(m.normal, v(0.0, -0.6, -0.8) * (1.0 / d), 1e-6));
    // A lying capsule seen from the sphere's side of the argument list and the other.
    let lying = turned(v(0.0, 0.0, 0.0), Vec3::Z, FRAC_PI_2);
    let m = one(&c, &lying, &s, &at(0.5, -0.7, 0.0), 0.0);
    assert!(close_v(m.normal, -Vec3::Y, 1e-5));
    assert!(close(m.points[0].separation, 0.7 - 0.75, 1e-5));
}

#[test]
fn crossed_capsules_give_one_point_and_parallel_capsules_two() {
    let c = capsule(1.0, 0.25);
    let along_x = turned(v(0.0, 0.0, 0.0), Vec3::Z, FRAC_PI_2);
    // Crossed: one point at the common perpendicular.
    let m = one(
        &c,
        &Iso::IDENTITY,
        &c,
        &Iso::new(v(0.0, 0.3, 0.45), along_x.rotation),
        0.1,
    );
    assert_eq!(m.count, 1);
    assert!(close_v(m.normal, Vec3::Z, 1e-5));
    assert!(close(m.points[0].separation, -0.05, 1e-5));
    assert!(close_v(m.points[0].point, v(0.0, 0.3, 0.225), 1e-5));
    // Parallel, side by side and overlapping by half their length: the overlap's two ends.
    let m = one(&c, &Iso::IDENTITY, &c, &at(0.45, 1.0, 0.0), 0.1);
    assert_eq!(m.count, 2, "{m:?}");
    assert!(close_v(m.normal, Vec3::X, 1e-5));
    let mut ys: Vec<f32> = m.points().iter().map(|p| p.point.y).collect();
    ys.sort_by(f32::total_cmp);
    assert!(close(ys[0], 0.0, 1e-5) && close(ys[1], 1.0, 1e-5), "{ys:?}");
    for p in m.points() {
        assert!(close(p.separation, -0.05, 1e-5));
        assert!(close(p.point.x, 0.225, 1e-5));
    }
    // End to end along the same line: one point between the caps.
    let m = one(&c, &Iso::IDENTITY, &c, &at(0.0, 2.45, 0.0), 0.1);
    assert_eq!(m.count, 1);
    assert!(close_v(m.normal, Vec3::Y, 1e-5));
    assert!(close(m.points[0].separation, -0.05, 1e-5));
}

#[test]
fn a_sphere_against_a_box_matches_the_distance_to_the_box_everywhere() {
    let h = v(1.0, 0.5, 0.75);
    let b = cuboid(h.x, h.y, h.z);
    let s = sphere(0.3);
    let rot = Quat::from_axis_angle(v(1.0, 2.0, 0.5).normalize_or_zero(), 0.8);
    let box_iso = Iso::new(v(2.0, -1.0, 0.5), rot);
    let mut rng = Rng(7);
    let mut seen = 0;
    for _ in 0..2000 {
        let local = rng.vec(2.0);
        let d = box_distance(local, h);
        if d == 0.0 {
            continue;
        }
        let c = box_iso.point(local);
        let out = run(&s, &at(c.x, c.y, c.z), &b, &box_iso, 0.25);
        if d - 0.3 > 0.25 {
            assert!(out.is_empty(), "{d} {out:?}");
            continue;
        }
        seen += 1;
        assert_eq!(out.len(), 1);
        let m = &out[0];
        assert!(close(m.points[0].separation, d - 0.3, 2e-5), "{d} {m:?}");
        // The normal runs from the sphere's centre towards the nearest box point.
        let nearest = box_iso.point(local.max(-h).min(h));
        assert!(close_v(m.normal, (nearest - c) * (1.0 / d), 1e-4), "{m:?}");
        // The point is midway between the sphere's surface and the box's.
        let mid = (c + m.normal * 0.3 + nearest) * 0.5;
        assert!(close_v(m.points[0].point, mid, 1e-4));
    }
    assert!(seen > 300, "{seen}");
}

#[test]
fn a_sphere_inside_a_box_leaves_through_the_nearest_face() {
    let b = cuboid(1.0, 2.0, 3.0);
    let s = sphere(0.5);
    let m = one(&s, &at(0.7, 0.2, -0.4), &b, &Iso::IDENTITY, 0.0);
    // 0.3 from the +X face: pushed out along +X (the normal runs from the sphere into the box).
    assert!(close_v(m.normal, -Vec3::X, 1e-6));
    assert!(close(m.points[0].separation, -0.3 - 0.5, 1e-6));
    // The centre of the box: any face is as good, but the result is finite and deterministic.
    let m1 = one(
        &s,
        &Iso::IDENTITY,
        &cuboid(1.0, 1.0, 1.0),
        &Iso::IDENTITY,
        0.0,
    );
    let m2 = one(
        &s,
        &Iso::IDENTITY,
        &cuboid(1.0, 1.0, 1.0),
        &Iso::IDENTITY,
        0.0,
    );
    assert_eq!(m1, m2);
    assert!(close(m1.points[0].separation, -1.5, 1e-6));
}

#[test]
fn a_capsule_against_a_box_matches_the_distance_between_core_and_box() {
    let h = v(1.0, 0.5, 0.75);
    let b = cuboid(h.x, h.y, h.z);
    let c = capsule(0.6, 0.2);
    let mut rng = Rng(11);
    let mut seen = 0;
    for _ in 0..1500 {
        let p = rng.vec(2.2);
        let rot = rng.rotation();
        let iso = Iso::new(p, rot);
        let (e0, e1) = (iso.point(v(0.0, -0.6, 0.0)), iso.point(v(0.0, 0.6, 0.0)));
        // Brute force: the distance from the box to points along the core.
        let mut d = f32::INFINITY;
        for i in 0..=2000 {
            d = d.min(box_distance(e0.lerp(e1, i as f32 / 2000.0), h));
        }
        if d < 1e-3 {
            continue;
        }
        let out = run(&c, &iso, &b, &Iso::IDENTITY, 0.3);
        if d - 0.2 > 0.3 + 1e-4 {
            assert!(out.is_empty(), "{d} {out:?}");
            continue;
        }
        if d - 0.2 > 0.3 - 1e-4 {
            continue;
        }
        seen += 1;
        assert_eq!(out.len(), 1, "{d} {out:?}");
        let s = out[0].min_separation();
        assert!(close(s, d - 0.2, 2e-4), "{d} {s} {:?}", out[0]);
    }
    assert!(seen > 200, "{seen}");
}

#[test]
fn a_capsule_lying_on_a_box_gets_two_points_at_the_ends_of_its_core() {
    let b = cuboid(2.0, 0.5, 2.0);
    let c = capsule(0.8, 0.25);
    let lying = turned(v(0.3, 0.5 + 0.25 + 0.01, -0.2), Vec3::Z, FRAC_PI_2);
    let m = one(&b, &Iso::IDENTITY, &c, &lying, 0.05);
    assert_eq!(m.count, 2, "{m:?}");
    assert!(close_v(m.normal, Vec3::Y, 1e-6));
    let mut xs: Vec<f32> = m.points().iter().map(|p| p.point.x).collect();
    xs.sort_by(f32::total_cmp);
    assert!(
        close(xs[0], 0.3 - 0.8, 1e-5) && close(xs[1], 0.3 + 0.8, 1e-5),
        "{xs:?}"
    );
    for p in m.points() {
        assert!(close(p.separation, 0.01, 1e-5));
        assert!(close(p.point.y, 0.505, 1e-5));
    }
    // Hanging over the box's edge: the core is clipped to the face.
    let over = turned(v(1.5, 0.5 + 0.25 - 0.02, 0.0), Vec3::Z, FRAC_PI_2);
    let m = one(&b, &Iso::IDENTITY, &c, &over, 0.05);
    assert_eq!(m.count, 2, "{m:?}");
    let mut xs: Vec<f32> = m.points().iter().map(|p| p.point.x).collect();
    xs.sort_by(f32::total_cmp);
    assert!(close(xs[0], 0.7, 1e-5) && close(xs[1], 2.0, 1e-5), "{xs:?}");
    // Slightly tilted, the far end still within the speculative distance keeps its point.
    let tilted = turned(v(0.0, 0.5 + 0.25 + 0.01, 0.0), Vec3::Z, FRAC_PI_2 - 0.02);
    let m = one(&b, &Iso::IDENTITY, &c, &tilted, 0.1);
    assert_eq!(m.count, 2, "{m:?}");
    assert!(close_v(m.normal, Vec3::Y, 1e-5));
    // Standing up: one point under its lower end.
    let m = one(
        &b,
        &Iso::IDENTITY,
        &c,
        &at(0.0, 0.5 + 0.8 + 0.25 - 0.03, 0.0),
        0.1,
    );
    assert_eq!(m.count, 1);
    assert!(close(m.points[0].separation, -0.03, 1e-5));
}

#[test]
fn a_capsule_poking_into_a_box_is_pushed_out_the_shortest_way() {
    let b = cuboid(1.0, 1.0, 1.0);
    let c = capsule(1.0, 0.1);
    // Upright with its lower end 0.3 deep under the top face.
    let m = one(&b, &Iso::IDENTITY, &c, &at(0.2, 1.0 + 1.0 - 0.3, 0.1), 0.0);
    assert!(close_v(m.normal, Vec3::Y, 1e-6));
    assert_eq!(m.count, 1);
    assert!(close(m.points[0].separation, -0.3 - 0.1, 1e-5));
    // Lying inside near the +Z face: out through +Z.
    let lying = turned(v(0.0, 0.0, 0.8), Vec3::Z, FRAC_PI_2);
    let m = one(&b, &Iso::IDENTITY, &capsule(0.5, 0.1), &lying, 0.0);
    assert!(close_v(m.normal, Vec3::Z, 1e-5), "{m:?}");
    assert_eq!(m.count, 2);
    for p in m.points() {
        assert!(close(p.separation, -0.2 - 0.1, 1e-5));
    }
    // Lying along a box edge just inside it: a face is the shorter way out (0.03 against
    // 0.03·√2 across the edge).
    let along_edge = turned(v(0.97, 0.97, 0.0), Vec3::X, FRAC_PI_2);
    let m = one(&b, &Iso::IDENTITY, &capsule(0.5, 0.05), &along_edge, 0.0);
    assert!(
        close_v(m.normal, Vec3::X, 1e-5) || close_v(m.normal, Vec3::Y, 1e-5),
        "{m:?}"
    );
    assert!(close(m.min_separation(), -0.08, 1e-5), "{m:?}");
    // Across a box edge at right angles, cutting its corner 0.03 deep: out across the edge.
    let diagonal = v(1.0, 1.0, 0.0).normalize_or_zero();
    let across = turned(v(1.0, 1.0, 0.2) - diagonal * 0.03, Vec3::Z, FRAC_PI_4);
    let m = one(&b, &Iso::IDENTITY, &capsule(0.3, 0.05), &across, 0.0);
    assert!(close_v(m.normal, diagonal, 1e-4), "{m:?}");
    assert_eq!(m.count, 1);
    assert!(close(m.points[0].separation, -0.08, 1e-4), "{m:?}");
    assert!(
        close_v(m.points[0].point, v(1.0, 1.0, 0.2) - diagonal * 0.04, 1e-4),
        "{m:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// Boxes and hulls.

#[test]
fn a_box_resting_flat_on_a_box_gets_four_points_with_zero_separation() {
    let ground = cuboid(5.0, 0.5, 5.0);
    let b = cuboid(0.5, 0.5, 0.5);
    let iso = turned(v(0.3, 1.0, -0.2), Vec3::Y, 0.4);
    let m = one(&ground, &at(0.0, 0.0, 0.0), &b, &iso, 0.02);
    assert_eq!(m.count, 4, "{m:?}");
    assert!(close_v(m.normal, Vec3::Y, 1e-6));
    for p in m.points() {
        assert!(close(p.separation, 0.0, 1e-5), "{m:?}");
        assert!(close(p.point.y, 0.5, 1e-5));
    }
    // The four points are the upper box's bottom corners.
    for sx in [-0.5f32, 0.5] {
        for sz in [-0.5f32, 0.5] {
            let corner = iso.point(v(sx, -0.5, sz));
            assert!(
                m.points().iter().any(|p| close_v(p.point, corner, 1e-5)),
                "{corner:?} {m:?}"
            );
        }
    }
    // The other order: the normal flips, the points stay.
    let r = one(&b, &iso, &ground, &at(0.0, 0.0, 0.0), 0.02);
    assert_eq!(r.count, 4);
    assert!(close_v(r.normal, -Vec3::Y, 1e-6));
    for p in r.points() {
        assert!(m.points().iter().any(|q| close_v(p.point, q.point, 1e-5)));
    }
    // The same box made a hull gives the same contact.
    let h = one(
        &hull_cuboid(5.0, 0.5, 5.0),
        &at(0.0, 0.0, 0.0),
        &b,
        &iso,
        0.02,
    );
    assert_eq!(h.count, 4);
    for p in h.points() {
        assert!(m.points().iter().any(|q| close_v(p.point, q.point, 1e-5)));
    }
}

#[test]
fn a_box_tilted_onto_an_edge_gets_two_points_and_on_a_corner_one() {
    let ground = cuboid(5.0, 0.5, 5.0);
    let b = cuboid(0.5, 0.5, 0.5);
    // Rolled 30 degrees about Z, resting on the edge along Z.
    let a = 30f32.to_radians();
    let lowest = 0.5 * (a.cos() + a.sin());
    let iso = turned(v(0.0, 0.5 + lowest - 0.001, 0.0), Vec3::Z, a);
    let m = one(&ground, &Iso::IDENTITY, &b, &iso, 0.02);
    assert_eq!(m.count, 2, "{m:?}");
    assert!(close_v(m.normal, Vec3::Y, 1e-6));
    let mut zs: Vec<f32> = m.points().iter().map(|p| p.point.z).collect();
    zs.sort_by(f32::total_cmp);
    assert!(
        close(zs[0], -0.5, 1e-4) && close(zs[1], 0.5, 1e-4),
        "{zs:?}"
    );
    for p in m.points() {
        assert!(close(p.separation, -0.001, 1e-4));
    }
    // A corner straight down (the cube's diagonal vertical): one point.
    let q = Quat::from_rotation_arc(v(1.0, 1.0, 1.0).normalize_or_zero(), -Vec3::Y);
    let half_diagonal = 0.5 * 3f32.sqrt();
    let iso = Iso::new(v(0.2, 0.5 + half_diagonal - 0.002, 0.1), q);
    let m = one(&ground, &Iso::IDENTITY, &b, &iso, 0.02);
    assert_eq!(m.count, 1, "{m:?}");
    assert!(close_v(m.normal, Vec3::Y, 1e-5));
    assert!(close(m.points[0].separation, -0.002, 1e-4));
    assert!(
        close_v(m.points[0].point, v(0.2, 0.499, 0.1), 1e-4),
        "{m:?}"
    );
}

#[test]
fn crossed_boxes_touching_edge_to_edge_get_one_edge_contact() {
    // Two long bars turned 45 degrees about their long axes, crossed at right angles, so the
    // lower one's top edge (along X) meets the upper one's bottom edge (along Z).
    let bar = cuboid(2.0, 0.25, 0.25);
    let lower = turned(Vec3::ZERO, Vec3::X, FRAC_PI_4);
    let upper_rot =
        Quat::from_axis_angle(Vec3::Y, FRAC_PI_2) * Quat::from_axis_angle(Vec3::X, FRAC_PI_4);
    let reach = 0.25 * 2f32.sqrt();
    for gap in [0.01f32, -0.02] {
        let upper = Iso::new(v(0.3, 2.0 * reach + gap, -0.2), upper_rot);
        let m = one(&bar, &lower, &bar, &upper, 0.05);
        assert_eq!(m.count, 1, "{m:?}");
        assert!(close_v(m.normal, Vec3::Y, 1e-4), "{m:?}");
        assert!(close(m.points[0].separation, gap, 1e-4), "{m:?}");
        // Where the upper edge (x = 0.3) crosses the lower one (z = 0).
        assert!(
            close_v(m.points[0].point, v(0.3, reach + 0.5 * gap, 0.0), 1e-4),
            "{m:?}"
        );
    }
}

#[test]
fn deeply_overlapping_boxes_separate_along_the_minimum_translation() {
    let b = cuboid(1.0, 1.0, 1.0);
    for (offset, normal, depth) in [
        (v(1.5, 0.2, -0.1), Vec3::X, 0.5),
        (v(0.1, -1.2, 0.3), -Vec3::Y, 0.8),
        (v(0.2, 0.1, 1.9), Vec3::Z, 0.1),
    ] {
        let m = one(&b, &Iso::IDENTITY, &b, &Iso::from_position(offset), 0.0);
        assert!(close_v(m.normal, normal, 1e-6), "{offset:?} {m:?}");
        assert!(close(m.min_separation(), -depth, 1e-5), "{m:?}");
        assert_eq!(m.count, 4);
    }
    // A small box sunk deep into a big one near its top face.
    let small = cuboid(0.2, 0.2, 0.2);
    let m = one(
        &cuboid(3.0, 3.0, 3.0),
        &Iso::IDENTITY,
        &small,
        &at(0.5, 2.5, -1.0),
        0.0,
    );
    assert!(close_v(m.normal, Vec3::Y, 1e-6));
    assert!(close(m.min_separation(), -0.7, 1e-5));
}

/// The vertices and face normals of a box or hull placed at `iso`, for brute force.
fn polytope(shape: &Shape, iso: &Iso) -> (Vec<Vec3>, Vec<Vec3>, Vec<Vec3>) {
    let hull = match shape {
        Shape::Box { half_extents } => ConvexHull::cuboid(*half_extents),
        Shape::Hull(h) => (**h).clone(),
        _ => unreachable!(),
    };
    let verts = hull.vertices.iter().map(|&p| iso.point(p)).collect();
    let normals = hull.faces.iter().map(|f| iso.vector(f.normal)).collect();
    let edges = hull
        .edges
        .iter()
        .step_by(2)
        .enumerate()
        .map(|(k, e)| {
            let to = hull.edges[2 * k + 1].origin as usize;
            iso.vector(hull.vertices[to] - hull.vertices[e.origin as usize])
        })
        .collect();
    (verts, normals, edges)
}

/// How far B lies beyond A along `n` (negative: overlap), by projecting every vertex.
fn projected_separation(va: &[Vec3], vb: &[Vec3], n: Vec3) -> f32 {
    let hi_a = va
        .iter()
        .map(|p| p.dot(n))
        .fold(f32::NEG_INFINITY, f32::max);
    let lo_b = vb.iter().map(|p| p.dot(n)).fold(f32::INFINITY, f32::min);
    lo_b - hi_a
}

/// The largest separation over all face normals and edge cross products (the minimum
/// translation's depth, negated, when they overlap).
fn brute_force_sat(a: &Shape, ia: &Iso, b: &Shape, ib: &Iso) -> f32 {
    let (va, na, ea) = polytope(a, ia);
    let (vb, nb, eb) = polytope(b, ib);
    let mut axes: Vec<Vec3> = na.iter().chain(nb.iter()).copied().collect();
    for x in &ea {
        for y in &eb {
            if let Some(n) = x.cross(*y).try_normalize()
                && x.cross(*y).length() > 1e-3 * x.length() * y.length()
            {
                axes.push(n);
            }
        }
    }
    axes.iter()
        .map(|&n| projected_separation(&va, &vb, n).max(projected_separation(&va, &vb, -n)))
        .fold(f32::NEG_INFINITY, f32::max)
}

/// Checks a polytope pair's manifold against brute force. Overlapping, the separating-axis test
/// is exact: the normal is a candidate axis whose separation is the best one within the
/// face-preference tolerances. Separated, its best axis is a lower bound of the distance (exact
/// when a face and a vertex or two edges are closest), so a contact must exist whenever brute
/// force finds one within `spec`, and what is reported must agree with the projections.
/// Returns whether the shapes overlap.
fn check_against_brute_force(a: &Shape, ia: &Iso, b: &Shape, ib: &Iso, spec: f32) -> bool {
    let best = brute_force_sat(a, ia, b, ib);
    let out = run(a, ia, b, ib, spec);
    if best <= spec - 1e-4 {
        assert_eq!(out.len(), 1, "best {best}");
    }
    let Some(m) = out.first() else {
        return false;
    };
    let (va, _, _) = polytope(a, ia);
    let (vb, _, _) = polytope(b, ib);
    let along = projected_separation(&va, &vb, m.normal);
    assert!(along <= best + 1e-4, "{along} > {best}: {m:?}");
    assert!(m.min_separation() >= along - 1e-4, "{along}: {m:?}");
    if best > 0.0 {
        return false;
    }
    // Face axes win unless an edge axis is clearly better (10%), and A's face unless B's is
    // (5%), plus a small absolute tolerance each: the chosen axis may be that much shallower.
    assert!(
        along >= best - 0.17 * best.abs() - 0.006,
        "{along} vs {best}: {m:?}"
    );
    true
}

#[test]
fn random_boxes_match_a_brute_force_separating_axis_test() {
    let mut rng = Rng(3);
    let mut checked = 0;
    for _ in 0..1500 {
        let a = cuboid(
            rng.range(0.2, 1.5),
            rng.range(0.2, 1.5),
            rng.range(0.2, 1.5),
        );
        let b = cuboid(
            rng.range(0.2, 1.5),
            rng.range(0.2, 1.5),
            rng.range(0.2, 1.5),
        );
        let ia = Iso::new(rng.vec(0.5), rng.rotation());
        let ib = Iso::new(rng.vec(2.5), rng.rotation());
        if check_against_brute_force(&a, &ia, &b, &ib, 0.1) {
            checked += 1;
        }
    }
    assert!(checked > 400, "{checked}");
}

#[test]
fn random_hulls_match_a_brute_force_separating_axis_test() {
    let shapes = [
        octahedron(1.0),
        tetrahedron(0.7),
        hull_cuboid(0.5, 0.8, 0.3),
        cuboid(0.6, 0.4, 0.9),
    ];
    let mut rng = Rng(5);
    let mut checked = 0;
    for i in 0..2000 {
        let a = &shapes[i % shapes.len()];
        let b = &shapes[(i / shapes.len()) % shapes.len()];
        let ia = Iso::new(rng.vec(0.5), rng.rotation());
        let ib = Iso::new(rng.vec(2.0), rng.rotation());
        if check_against_brute_force(a, &ia, b, &ib, 0.1) {
            checked += 1;
        }
    }
    assert!(checked > 300, "{checked}");
}

#[test]
fn hulls_rest_face_to_face_and_meet_edge_to_edge() {
    // An octahedron balanced on its tip above a box: one point.
    let o = octahedron(1.0);
    let ground = cuboid(4.0, 0.5, 4.0);
    let m = one(
        &ground,
        &Iso::IDENTITY,
        &o,
        &at(0.3, 0.5 + 1.0 + 0.01, 0.0),
        0.05,
    );
    assert_eq!(m.count, 1);
    assert!(close_v(m.normal, Vec3::Y, 1e-6));
    assert!(close(m.points[0].separation, 0.01, 1e-5));
    assert!(close_v(m.points[0].point, v(0.3, 0.505, 0.0), 1e-5));
    // Two tetrahedra face to face: a triangle's three corners.
    let t = tetrahedron(0.5);
    // Face (1, 2, 3) of `tetrahedron` points along (-1, -1, -1); turn it to face down.
    let down = Quat::from_rotation_arc(v(-1.0, -1.0, -1.0).normalize_or_zero(), -Vec3::Y);
    let up = Quat::from_rotation_arc(v(-1.0, -1.0, -1.0).normalize_or_zero(), Vec3::Y);
    let inradius = 0.5 / 3f32.sqrt();
    let lower = Iso::new(v(0.0, 0.0, 0.0), up);
    let upper = Iso::new(
        v(0.0, 2.0 * inradius - 0.004, 0.0),
        down * Quat::from_axis_angle(Vec3::Y, 0.0),
    );
    let m = one(&t, &lower, &t, &upper, 0.01);
    assert!(close_v(m.normal, Vec3::Y, 1e-4), "{m:?}");
    assert!(m.count >= 3, "{m:?}");
    for p in m.points() {
        assert!(close(p.separation, -0.004, 1e-4), "{m:?}");
    }
    // Two octahedra turned so that the lower one's top is an edge along X and the upper one's
    // bottom an edge along Z: one point where the edges cross.
    let lower = turned(Vec3::ZERO, Vec3::Z, FRAC_PI_4);
    let reach = FRAC_PI_4.cos();
    for gap in [0.02f32, -0.03] {
        let upper = turned(v(0.1, 2.0 * reach + gap, -0.1), Vec3::X, FRAC_PI_4);
        let m = one(&o, &lower, &o, &upper, 0.05);
        assert_eq!(m.count, 1, "{m:?}");
        assert!(close_v(m.normal, Vec3::Y, 1e-4), "{m:?}");
        assert!(close(m.points[0].separation, gap, 1e-4), "{m:?}");
        assert!(
            close_v(m.points[0].point, v(0.1, reach + 0.5 * gap, 0.0), 1e-4),
            "{m:?}"
        );
    }
}

#[test]
fn swapping_the_shapes_flips_the_normal_and_keeps_the_points() {
    let shapes = [
        sphere(0.5),
        capsule(0.6, 0.3),
        cuboid(0.5, 0.4, 0.7),
        octahedron(0.8),
        tetrahedron(0.6),
    ];
    let mut rng = Rng(9);
    let mut compared = 0;
    for i in 0..1500 {
        let a = &shapes[i % shapes.len()];
        let b = &shapes[(i / shapes.len()) % shapes.len()];
        let ia = Iso::new(rng.vec(0.3), rng.rotation());
        let ib = Iso::new(rng.vec(1.6), rng.rotation());
        let ab = run(a, &ia, b, &ib, 0.05);
        let ba = run(b, &ib, a, &ia, 0.05);
        assert_eq!(ab.len(), ba.len());
        let (Some(m), Some(r)) = (ab.first(), ba.first()) else {
            continue;
        };
        compared += 1;
        // Two polytopes prefer A's face as reference unless B's is clearly better (so a resting
        // contact keeps its features); with near-equal axes the two orders may then choose
        // differently, and both choices must be valid. Everything else must match exactly.
        let round = |s: &Shape| matches!(s, Shape::Sphere { .. } | Shape::Capsule { .. });
        if !round(a) && !round(b) && !close_v(m.normal, -r.normal, 1e-4) {
            check_against_brute_force(a, &ia, b, &ib, 0.05);
            check_against_brute_force(b, &ib, a, &ia, 0.05);
            continue;
        }
        assert!(close_v(m.normal, -r.normal, 1e-4), "{m:?} {r:?}");
        assert_eq!(m.count, r.count);
        for p in m.points() {
            assert!(
                r.points()
                    .iter()
                    .any(|q| close_v(p.point, q.point, 1e-3)
                        && close(p.separation, q.separation, 1e-4)),
                "{m:?} {r:?}"
            );
        }
    }
    assert!(compared > 300, "{compared}");
}

#[test]
fn speculative_points_are_kept_within_the_distance_and_dropped_beyond_it() {
    let ground = cuboid(3.0, 0.5, 3.0);
    let b = cuboid(0.5, 0.5, 0.5);
    let above = at(0.0, 1.0 + 0.05, 0.0);
    // 5 cm apart: kept with a positive separation when the speculative distance reaches.
    let m = one(&ground, &Iso::IDENTITY, &b, &above, 0.1);
    assert_eq!(m.count, 4);
    for p in m.points() {
        assert!(close(p.separation, 0.05, 1e-5));
    }
    none(&ground, &Iso::IDENTITY, &b, &above, 0.04);
    // Tilted: only the corners within the distance stay.
    let tilted = turned(v(0.0, 1.0 + 0.02, 0.0), Vec3::Z, 0.05);
    let m = one(&ground, &Iso::IDENTITY, &b, &tilted, 0.03);
    assert_eq!(m.count, 2, "{m:?}");
    for p in m.points() {
        assert!(p.separation <= 0.03);
    }
    // Every shape type honours it.
    for (s, reach) in [
        (sphere(0.5), 0.5),
        (capsule(0.5, 0.5), 1.0),
        (octahedron(0.5), 0.5),
    ] {
        let iso = at(0.0, 0.5 + reach + 0.07, 0.0);
        let m = one(&ground, &Iso::IDENTITY, &s, &iso, 0.08);
        assert!(close(m.min_separation(), 0.07, 1e-4), "{m:?}");
        none(&ground, &Iso::IDENTITY, &s, &iso, 0.06);
    }
}

#[test]
fn ids_stay_the_same_while_a_resting_box_moves_slightly() {
    let ground = cuboid(5.0, 0.5, 5.0);
    let b = cuboid(0.5, 0.25, 0.75);
    let base = one(
        &ground,
        &Iso::IDENTITY,
        &b,
        &turned(v(0.0, 0.749, 0.0), Vec3::Y, 0.3),
        0.02,
    );
    assert_eq!(base.count, 4);
    let mut rng = Rng(21);
    for _ in 0..50 {
        let iso = Iso::new(
            v(
                rng.range(-0.05, 0.05),
                0.749 + rng.range(-0.002, 0.002),
                rng.range(-0.05, 0.05),
            ),
            Quat::from_axis_angle(Vec3::Y, 0.3 + rng.range(-0.02, 0.02))
                * Quat::from_axis_angle(Vec3::X, rng.range(-0.002, 0.002)),
        );
        let m = one(&ground, &Iso::IDENTITY, &b, &iso, 0.02);
        assert_eq!(m.count, 4);
        let mut a: Vec<u32> = base.points().iter().map(|p| p.id).collect();
        let mut c: Vec<u32> = m.points().iter().map(|p| p.id).collect();
        a.sort_unstable();
        c.sort_unstable();
        assert_eq!(a, c, "{base:?} {m:?}");
    }
    // The same corner keeps its id: match by position.
    let moved = one(
        &ground,
        &Iso::IDENTITY,
        &b,
        &turned(v(0.01, 0.749, 0.0), Vec3::Y, 0.3),
        0.02,
    );
    for p in base.points() {
        let q = moved
            .points()
            .iter()
            .min_by(|x, y| {
                (x.point - p.point)
                    .length()
                    .total_cmp(&(y.point - p.point).length())
            })
            .unwrap();
        assert_eq!(p.id, q.id);
    }
}

#[test]
fn collide_appends_without_clearing() {
    let mut out = vec![Manifold::default()];
    collide(
        &sphere(1.0),
        &Iso::IDENTITY,
        &sphere(1.0),
        &at(1.5, 0.0, 0.0),
        0.0,
        &mut out,
    );
    assert_eq!(out.len(), 2);
    assert_eq!(out[0], Manifold::default());
}

// ---------------------------------------------------------------------------------------------
// Meshes and heightfields.

/// Clears the active bit of every edge two triangles share in opposite directions: what a mesh
/// marks for flat seams (these tests build meshes whose shared edges are all flat or concave).
fn mark_shared_edges_inactive(mesh: &mut TriangleMesh) {
    let tris = mesh.triangles.clone();
    for (t, tri) in tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            let shared = tris
                .iter()
                .any(|o| (0..3).any(|j| o[j] == b && o[(j + 1) % 3] == a));
            if shared {
                mesh.active_edges[t] &= !(1u8 << k);
            }
        }
    }
}

/// A flat floor at y = 0 of `cells` × `cells` square cells of side `size` centred on the origin,
/// each split into two triangles; shared edges inactive, the outer ones active.
fn grid_floor(cells: u32, size: f32) -> TriangleMesh {
    let n = cells + 1;
    let half = cells as f32 * size * 0.5;
    let mut vertices = vec![];
    for j in 0..n {
        for i in 0..n {
            vertices.push(v(i as f32 * size - half, 0.0, j as f32 * size - half));
        }
    }
    let mut tris = vec![];
    for j in 0..cells {
        for i in 0..cells {
            let (a, c) = (j * n + i, (j + 1) * n + i);
            // Counter-clockwise seen from above.
            tris.push([a, c, c + 1]);
            tris.push([a, c + 1, a + 1]);
        }
    }
    let mut mesh = TriangleMesh::new(vertices, tris).unwrap();
    mark_shared_edges_inactive(&mut mesh);
    mesh
}

#[test]
fn a_box_sliding_over_a_triangulated_floor_only_sees_up_normals() {
    let floor = Shape::Mesh(Arc::new(grid_floor(4, 2.0)));
    let b = cuboid(0.5, 0.3, 0.4);
    let mut rng = Rng(31);
    for step in 0..400 {
        let t = step as f32 / 399.0;
        // Across many seams and the cells' shared corners, turning, with a small tilt.
        let pos = v(-2.6 + 5.2 * t, 0.3 - 0.004, -1.9 + 3.8 * t * t);
        let tilt = rng.range(-0.003, 0.003);
        let rot = Quat::from_axis_angle(Vec3::Y, 2.0 * t + rng.range(-0.1, 0.1))
            * Quat::from_axis_angle(Vec3::X, tilt);
        let iso = Iso::new(pos, rot);
        let out = run(&floor, &Iso::IDENTITY, &b, &iso, 0.02);
        assert_eq!(out.len(), 1, "step {step}: {out:?}");
        let m = &out[0];
        assert!(close_v(m.normal, Vec3::Y, 1e-5), "step {step}: {m:?}");
        assert_eq!(m.count, 4, "step {step}: {m:?}");
        for p in m.points() {
            let corner_depth = -0.004 - 0.4 * tilt.abs() - 1e-4;
            assert!(
                p.separation >= corner_depth && p.separation <= 0.02,
                "step {step}: {m:?}"
            );
        }
        // The other order flips the normal.
        let r = run(&b, &iso, &floor, &Iso::IDENTITY, 0.02);
        assert!(r.len() == 1 && close_v(r[0].normal, -Vec3::Y, 1e-5));
    }
}

/// Two 2 m square cells side by side (x in [-2, 0] and [0, 2]), each split along a diagonal.
fn two_cells() -> TriangleMesh {
    let vertices = vec![
        v(-2.0, 0.0, -2.0),
        v(0.0, 0.0, -2.0),
        v(2.0, 0.0, -2.0),
        v(-2.0, 0.0, 2.0),
        v(0.0, 0.0, 2.0),
        v(2.0, 0.0, 2.0),
    ];
    TriangleMesh::new(vertices, vec![[0, 3, 4], [0, 4, 1], [1, 4, 5], [1, 5, 2]]).unwrap()
}

#[test]
fn a_seam_catches_a_box_only_while_it_is_active() {
    let b = cuboid(0.5, 0.5, 0.5);
    // Reaching 0.5 mm over the seam at x = 0, sunk 4 mm.
    let iso = at(-0.5 + 0.0005, 0.5 - 0.004, 0.0);
    // TriangleMesh::new already makes the flat seam inactive; mark every edge active to see
    // what an active seam does.
    let mut all_active = two_cells();
    all_active.active_edges.iter_mut().for_each(|e| *e = 0b111);
    let active = Shape::Mesh(Arc::new(all_active));
    let out = run(&active, &Iso::IDENTITY, &b, &iso, 0.01);
    assert!(
        out.iter().any(|m| m.normal.y.abs() < 0.1),
        "an active seam gives a side normal: {out:?}"
    );
    let mut mesh = two_cells();
    mark_shared_edges_inactive(&mut mesh);
    let inactive = Shape::Mesh(Arc::new(mesh));
    let out = run(&inactive, &Iso::IDENTITY, &b, &iso, 0.01);
    assert_eq!(out.len(), 1, "{out:?}");
    assert!(close_v(out[0].normal, Vec3::Y, 1e-6));
    assert_eq!(out[0].count, 4);
    for p in out[0].points() {
        assert!(close(p.separation, -0.004, 1e-5));
    }
}

/// A step: a top face at y = 0 for x in [-2, 0] and a side face at x = 0 going down to y = -2,
/// facing +X, meeting at a convex edge along Z (active unless `crease_active` is false).
fn step_mesh(crease_active: bool) -> TriangleMesh {
    let vertices = vec![
        v(-2.0, 0.0, -2.0),
        v(0.0, 0.0, -2.0),
        v(-2.0, 0.0, 2.0),
        v(0.0, 0.0, 2.0),
        v(0.0, -2.0, -2.0),
        v(0.0, -2.0, 2.0),
    ];
    // Top (facing +Y), then side (facing +X).
    let tris = vec![[0, 2, 3], [0, 3, 1], [1, 3, 5], [1, 5, 4]];
    let mut mesh = TriangleMesh::new(vertices, tris).unwrap();
    for t in 0..2 {
        assert!(mesh.triangle(t).normal().y > 0.99);
        assert!(mesh.triangle(t + 2).normal().x > 0.99);
    }
    mesh.active_edges = vec![0b111; 4];
    mark_shared_edges_inactive(&mut mesh);
    if crease_active {
        // The crease (vertices 1 and 3) is convex: active. The diagonals stay inactive.
        mesh.active_edges[1] |= 0b010; // 3 → 1 is edge 1 of [0, 3, 1]
        mesh.active_edges[2] |= 0b001; // 1 → 3 is edge 0 of [1, 3, 5]
    }
    mesh
}

#[test]
fn a_sphere_at_a_convex_mesh_corner_gets_the_edge_normal_where_the_edge_is_active() {
    let step = Shape::Mesh(Arc::new(step_mesh(true)));
    let s = sphere(0.5);
    // Rolling from over the top face, round the corner, down the side.
    let mut previous_y = 1.1f32;
    for i in 0..=40 {
        let angle = FRAC_PI_2 * i as f32 / 40.0;
        let dir = v(angle.sin(), angle.cos(), 0.0);
        let centre = dir * 0.45 + v(0.0, 0.0, 0.3);
        let out = run(
            &step,
            &Iso::IDENTITY,
            &s,
            &at(centre.x, centre.y, centre.z),
            0.0,
        );
        assert_eq!(out.len(), 1, "{i}: {out:?}");
        let m = &out[0];
        assert!(close_v(m.normal, dir, 1e-4), "{i}: {m:?}");
        assert!(close(m.points[0].separation, -0.05, 1e-4), "{i}: {m:?}");
        assert!(m.normal.y < previous_y + 1e-6);
        previous_y = m.normal.y;
    }
    // Off the corner over the top face: the face normal.
    let m = one(&step, &Iso::IDENTITY, &s, &at(-0.6, 0.45, 0.0), 0.0);
    assert!(close_v(m.normal, Vec3::Y, 1e-6));
    // An inactive crease turns the edge normal into each face's normal. Beside each face its
    // plane passes above the corner, so the separation is the true distance (5 cm deep here),
    // not the depth below the plane (14 and 23 cm).
    let smooth = Shape::Mesh(Arc::new(step_mesh(false)));
    let out = run(&smooth, &Iso::IDENTITY, &s, &at(0.27, 0.36, 0.3), 0.0);
    assert_eq!(out.len(), 2, "{out:?}");
    let up = out
        .iter()
        .find(|m| close_v(m.normal, Vec3::Y, 1e-5))
        .expect("top face");
    let side = out
        .iter()
        .find(|m| close_v(m.normal, Vec3::X, 1e-5))
        .expect("side face");
    assert!(close(up.points[0].separation, -0.05, 1e-5), "{up:?}");
    assert!(close(side.points[0].separation, -0.05, 1e-5), "{side:?}");
    // Deepest first.
    assert!(out[0].min_separation() <= out[1].min_separation());
    // A sphere 5 cm from the corner is below both planes but touches neither face.
    none(&smooth, &Iso::IDENTITY, &s, &at(0.33, 0.44, 0.3), 0.1);
}

#[test]
fn triangles_ignore_shapes_behind_them() {
    let floor = Shape::Mesh(Arc::new(grid_floor(2, 2.0)));
    for (s, depth) in [
        (sphere(0.5), 0.3),
        (capsule(0.4, 0.3), 0.6),
        (cuboid(0.5, 0.5, 0.5), 0.4),
        (octahedron(0.5), 0.4),
    ] {
        // Centre below the floor, poking up through it: nothing (one-sided).
        none(&floor, &Iso::IDENTITY, &s, &at(0.1, -depth, 0.2), 0.05);
        // Centre above, sunk as deep: pushed up.
        let out = run(&floor, &Iso::IDENTITY, &s, &at(0.1, depth, 0.2), 0.05);
        assert_eq!(out.len(), 1, "{out:?}");
        assert!(close_v(out[0].normal, Vec3::Y, 1e-5), "{out:?}");
    }
}

#[test]
fn shapes_resting_on_a_mesh_get_exact_separations() {
    let floor = Shape::Mesh(Arc::new(grid_floor(4, 1.0)));
    // A capsule lying across a seam: two points, one per end of its core.
    let lying = turned(v(0.1, 0.3 - 0.01, 0.37), Vec3::Z, FRAC_PI_2);
    let m = one(&floor, &Iso::IDENTITY, &capsule(0.8, 0.3), &lying, 0.02);
    assert!(close_v(m.normal, Vec3::Y, 1e-6));
    assert_eq!(m.count, 2, "{m:?}");
    for p in m.points() {
        assert!(close(p.separation, -0.01, 1e-5));
    }
    // A sphere right over a vertex shared by six triangles: one point.
    let m = one(
        &floor,
        &Iso::IDENTITY,
        &sphere(0.25),
        &at(0.0, 0.24, 0.0),
        0.0,
    );
    assert!(close_v(m.normal, Vec3::Y, 1e-6));
    assert_eq!(m.count, 1, "{m:?}");
    assert!(close(m.points[0].separation, -0.01, 1e-6));
    assert!(close_v(m.points[0].point, v(0.0, -0.005, 0.0), 1e-6));
    // An octahedron on its tip: one point.
    let m = one(
        &floor,
        &Iso::IDENTITY,
        &octahedron(0.5),
        &at(0.3, 0.5 + 0.01, -0.2),
        0.02,
    );
    assert_eq!(m.count, 1);
    assert!(close(m.points[0].separation, 0.01, 1e-5));
    // A box turned to rest on its edge.
    let iso = turned(v(0.2, 0.5 * 2f32.sqrt() - 0.002, 0.1), Vec3::Z, FRAC_PI_4);
    let m = one(&floor, &Iso::IDENTITY, &cuboid(0.5, 0.5, 0.5), &iso, 0.01);
    assert!(close_v(m.normal, Vec3::Y, 1e-5));
    // The edge crosses seams; its ends are what is kept.
    assert_eq!(m.count, 2, "{m:?}");
    let mut zs: Vec<f32> = m.points().iter().map(|p| p.point.z).collect();
    zs.sort_by(f32::total_cmp);
    assert!(close(zs[0], -0.4, 1e-4) && close(zs[1], 0.6, 1e-4), "{m:?}");
}

#[test]
fn mesh_manifolds_merge_by_normal_and_keep_the_deepest_eight() {
    // A sphere inside an icosahedral cage of inward-facing triangles: 20 normals.
    let t = (1.0 + 5f32.sqrt()) / 2.0;
    let vertices = vec![
        v(-1.0, t, 0.0),
        v(1.0, t, 0.0),
        v(-1.0, -t, 0.0),
        v(1.0, -t, 0.0),
        v(0.0, -1.0, t),
        v(0.0, 1.0, t),
        v(0.0, -1.0, -t),
        v(0.0, 1.0, -t),
        v(t, 0.0, -1.0),
        v(t, 0.0, 1.0),
        v(-t, 0.0, -1.0),
        v(-t, 0.0, 1.0),
    ];
    let faces: [[u32; 3]; 20] = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    // Wind each face to look inward.
    let faces: Vec<[u32; 3]> = faces
        .iter()
        .map(|&[a, b, c]| {
            let n = (vertices[b as usize] - vertices[a as usize])
                .cross(vertices[c as usize] - vertices[a as usize]);
            if n.dot(vertices[a as usize]) > 0.0 {
                [a, c, b]
            } else {
                [a, b, c]
            }
        })
        .collect();
    let cage = TriangleMesh::new(vertices.clone(), faces.clone()).unwrap();
    let centre = v(0.03, 0.02, -0.01);
    let r = 1.56;
    let s = sphere(r);
    let out = run(
        &Shape::Mesh(Arc::new(cage)),
        &Iso::IDENTITY,
        &s,
        &at(centre.x, centre.y, centre.z),
        0.0,
    );
    // Expected: each face's plane distance minus the radius; the eight deepest.
    let mut expect: Vec<(f32, Vec3)> = faces
        .iter()
        .map(|&[a, b, c]| {
            let (pa, pb, pc) = (
                vertices[a as usize],
                vertices[b as usize],
                vertices[c as usize],
            );
            let n = (pb - pa).cross(pc - pa).normalize_or_zero();
            (n.dot(centre - pa) - r, n)
        })
        .collect();
    expect.sort_by(|x, y| x.0.total_cmp(&y.0));
    assert_eq!(out.len(), MAX_MESH_MANIFOLDS, "{out:?}");
    for (m, (sep, n)) in out.iter().zip(&expect) {
        assert!(close(m.min_separation(), *sep, 1e-4), "{m:?} {sep}");
        assert!(close_v(m.normal, *n, 1e-4));
    }
    // Deepest first.
    for w in out.windows(2) {
        assert!(w[0].min_separation() <= w[1].min_separation());
    }
}

#[test]
fn mesh_point_ids_carry_the_triangle_id_in_their_upper_bits() {
    let mesh = grid_floor(4, 1.0);
    let floor = Shape::Mesh(Arc::new(mesh.clone()));
    let iso = turned(v(0.3, 0.25 - 0.003, -0.2), Vec3::Y, 0.35);
    let m = one(&floor, &Iso::IDENTITY, &cuboid(0.6, 0.25, 0.45), &iso, 0.01);
    assert_eq!(m.count, 4);
    for p in m.points() {
        let tri = mesh.triangle(p.id >> 7);
        // The point lies over (or on the edge of) its triangle.
        let [a, b, c] = tri.vertices;
        let inside = [(a, b), (b, c), (c, a)]
            .iter()
            .all(|&(p0, p1)| (p1 - p0).cross(Vec3::Y).dot(p.point - p0) <= 1e-4);
        assert!(inside, "{p:?} {tri:?}");
        assert_ne!(p.id & 0x7f, 0);
    }
}

#[test]
fn a_heightfield_collides_like_the_mesh_of_its_triangles() {
    let heights: Vec<f32> = (0..36).map(|i| ((i * 7) % 5) as f32 * 0.1).collect();
    let hf = Heightfield::new(6, 6, heights, v(1.0, 1.0, 1.0)).unwrap();
    let count = 2 * 5 * 5;
    let mut vertices = vec![];
    let mut tris = vec![];
    let mut active = vec![];
    for id in 0..count {
        let t = hf.triangle(id);
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&t.vertices);
        tris.push([base, base + 1, base + 2]);
        active.push(t.active_edges);
    }
    let mut mesh = TriangleMesh::new(vertices, tris).unwrap();
    assert_eq!(mesh.triangles.len(), count as usize);
    mesh.active_edges = active;
    let (hf, mesh) = (
        Shape::Heightfield(Arc::new(hf)),
        Shape::Mesh(Arc::new(mesh)),
    );
    let mut rng = Rng(41);
    let shapes = [
        sphere(0.4),
        capsule(0.5, 0.3),
        cuboid(0.4, 0.3, 0.5),
        octahedron(0.5),
    ];
    let mut touching = 0;
    for i in 0..200 {
        let s = &shapes[i % shapes.len()];
        let iso = Iso::new(
            v(
                rng.range(0.5, 4.5),
                rng.range(0.2, 0.9),
                rng.range(0.5, 4.5),
            ),
            rng.rotation(),
        );
        let a = run(&hf, &Iso::IDENTITY, s, &iso, 0.05);
        let b = run(&mesh, &Iso::IDENTITY, s, &iso, 0.05);
        // The same contacts; a triangle's normal may differ in its last bits between the two
        // (the heightfield builds it from its samples, the mesh from its stored vertices).
        assert_eq!(a.len(), b.len(), "{a:?} {b:?}");
        for (ma, mb) in a.iter().zip(&b) {
            assert!((ma.normal - mb.normal).length() < 1e-5, "{ma:?} {mb:?}");
            assert_eq!(ma.count, mb.count, "{ma:?} {mb:?}");
            for (pa, pb) in ma.points().iter().zip(mb.points()) {
                assert_eq!(pa.id, pb.id);
                assert!((pa.point - pb.point).length() < 1e-4, "{pa:?} {pb:?}");
                assert!(
                    (pa.separation - pb.separation).abs() < 1e-5,
                    "{pa:?} {pb:?}"
                );
            }
        }
        touching += usize::from(!a.is_empty());
    }
    assert!(touching > 50, "{touching}");
}

#[test]
fn static_geometry_never_collides_with_static_geometry() {
    let mesh = Shape::Mesh(Arc::new(grid_floor(2, 1.0)));
    let hf = Shape::Heightfield(Arc::new(
        Heightfield::new(2, 2, vec![0.0; 4], v(1.0, 1.0, 1.0)).unwrap(),
    ));
    for (a, b) in [(&mesh, &mesh), (&mesh, &hf), (&hf, &mesh), (&hf, &hf)] {
        none(a, &Iso::IDENTITY, b, &Iso::IDENTITY, 1.0);
    }
}

#[test]
fn a_moved_and_turned_mesh_collides_in_its_own_frame() {
    let floor = Shape::Mesh(Arc::new(grid_floor(4, 1.0)));
    // The floor stood up as a wall facing +X at x = 3.
    let wall = Iso::new(
        v(3.0, 1.0, -2.0),
        Quat::from_axis_angle(Vec3::Z, -FRAC_PI_2),
    );
    let m = one(
        &cuboid(0.5, 0.5, 0.5),
        &at(3.49, 1.2, -1.9),
        &floor,
        &wall,
        0.02,
    );
    assert!(close_v(m.normal, -Vec3::X, 1e-5), "{m:?}");
    assert_eq!(m.count, 4);
    for p in m.points() {
        assert!(close(p.separation, -0.01, 1e-4), "{m:?}");
        assert!(close(p.point.x, 2.995, 1e-4));
    }
    // Behind the wall (it is one-sided): nothing.
    none(
        &cuboid(0.5, 0.5, 0.5),
        &at(2.51, 1.2, -1.9),
        &floor,
        &wall,
        0.02,
    );
}

// ---------------------------------------------------------------------------------------------
// Exact distances to hulls, capsules on meshes, robustness.

/// The distance from `p` to triangle `t` (an independent formulation: inside the prism the plane
/// distance, else the nearest of the three sides).
fn point_triangle_distance(p: Vec3, t: [Vec3; 3]) -> f32 {
    let n = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
    let inside = (0..3).all(|k| (t[(k + 1) % 3] - t[k]).cross(n).dot(p - t[k]) <= 0.0);
    if inside {
        return n.dot(p - t[0]).abs();
    }
    (0..3)
        .map(|k| {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let s = ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
            (a + (b - a) * s - p).length()
        })
        .fold(f32::INFINITY, f32::min)
}

/// The distance from `p` to a hull placed at `iso` (0 inside).
fn point_hull_distance(p: Vec3, hull: &ConvexHull, iso: &Iso) -> f32 {
    let q = iso.inverse_point(p);
    if hull.faces.iter().all(|f| f.normal.dot(q) <= f.offset) {
        return 0.0;
    }
    let mut best = f32::INFINITY;
    for f in 0..hull.faces.len() {
        let ring = hull.face_vertices(f);
        for k in 1..ring.len() - 1 {
            best = best.min(point_triangle_distance(q, [ring[0], ring[k], ring[k + 1]]));
        }
    }
    best
}

fn hull_of(s: &Shape) -> &ConvexHull {
    match s {
        Shape::Hull(h) => h,
        _ => unreachable!(),
    }
}

#[test]
fn a_sphere_against_hulls_matches_the_exact_distance() {
    let mut rng = Rng(51);
    let mut seen = 0;
    for shape in [octahedron(1.0), tetrahedron(0.8)] {
        let hull = hull_of(&shape);
        let iso = Iso::new(v(0.5, -0.2, 0.1), rng.rotation());
        for _ in 0..800 {
            let c = iso.position + rng.vec(2.0);
            let d = point_hull_distance(c, hull, &iso);
            if d < 1e-3 {
                continue;
            }
            let out = run(&sphere(0.25), &at(c.x, c.y, c.z), &shape, &iso, 0.3);
            if d - 0.25 > 0.3 {
                assert!(out.is_empty());
                continue;
            }
            seen += 1;
            assert_eq!(out.len(), 1);
            assert!(
                close(out[0].points[0].separation, d - 0.25, 1e-4),
                "{d} {out:?}"
            );
        }
    }
    assert!(seen > 150, "{seen}");
}

#[test]
fn a_capsule_against_hulls_matches_the_exact_distance() {
    let mut rng = Rng(52);
    let mut seen = 0;
    for shape in [
        octahedron(1.0),
        tetrahedron(0.8),
        hull_cuboid(0.4, 0.7, 0.5),
    ] {
        let hull = hull_of(&shape);
        let iso = Iso::new(v(-0.3, 0.2, 0.1), rng.rotation());
        for _ in 0..300 {
            let ci = Iso::new(iso.position + rng.vec(2.2), rng.rotation());
            let (e0, e1) = (ci.point(v(0.0, -0.5, 0.0)), ci.point(v(0.0, 0.5, 0.0)));
            let mut d = f32::INFINITY;
            for i in 0..=400 {
                d = d.min(point_hull_distance(
                    e0.lerp(e1, i as f32 / 400.0),
                    hull,
                    &iso,
                ));
            }
            if d < 1e-3 {
                continue;
            }
            let out = run(&capsule(0.5, 0.2), &ci, &shape, &iso, 0.3);
            if d - 0.2 > 0.3 + 1e-3 {
                assert!(out.is_empty(), "{d} {out:?}");
                continue;
            }
            if d - 0.2 > 0.3 - 1e-3 {
                continue;
            }
            seen += 1;
            assert_eq!(out.len(), 1);
            // Sampling the core overestimates the distance slightly.
            let s = out[0].min_separation();
            assert!(
                s <= d - 0.2 + 1e-4 && s >= d - 0.2 - 2e-3,
                "{d} {s} {out:?}"
            );
        }
    }
    assert!(seen > 80, "{seen}");
}

#[test]
fn a_capsule_on_a_mesh_edge_gets_the_edge_normal_only_where_it_is_active() {
    let s = capsule(0.5, 0.25);
    // Lying along the step's crease (Z), above and outside it.
    let iso = Iso::new(
        v(0.12, 0.16, 0.0),
        Quat::from_axis_angle(Vec3::X, FRAC_PI_2),
    );
    let step = Shape::Mesh(Arc::new(step_mesh(true)));
    let out = run(&step, &Iso::IDENTITY, &s, &iso, 0.0);
    assert_eq!(out.len(), 1, "{out:?}");
    let m = &out[0];
    assert!(close_v(m.normal, v(0.6, 0.8, 0.0), 1e-4), "{m:?}");
    // Parallel to the edge: both ends of the core.
    assert_eq!(m.count, 2, "{m:?}");
    for p in m.points() {
        assert!(close(p.separation, -0.05, 1e-4), "{m:?}");
    }
    let smooth = Shape::Mesh(Arc::new(step_mesh(false)));
    let out = run(&smooth, &Iso::IDENTITY, &s, &iso, 0.0);
    assert!(!out.is_empty());
    for m in &out {
        assert!(
            close_v(m.normal, Vec3::Y, 1e-5) || close_v(m.normal, Vec3::X, 1e-5),
            "{m:?}"
        );
    }
}

#[test]
fn a_capsule_through_a_triangle_is_pushed_out_to_its_front() {
    let floor = Shape::Mesh(Arc::new(grid_floor(2, 2.0)));
    // Upright, poking 0.2 below the floor (the centre stays above).
    let m = one(
        &floor,
        &Iso::IDENTITY,
        &capsule(0.5, 0.1),
        &at(0.3, 0.3, 0.2),
        0.0,
    );
    assert!(close_v(m.normal, Vec3::Y, 1e-6));
    assert!(close(m.min_separation(), -0.2 - 0.1, 1e-5), "{m:?}");
    // Lying through it at a shallow angle: two points, the lower one deeper.
    let iso = Iso::new(
        v(0.3, 0.05, 0.2),
        Quat::from_axis_angle(Vec3::Z, FRAC_PI_2 - 0.1),
    );
    let m = one(&floor, &Iso::IDENTITY, &capsule(0.5, 0.1), &iso, 0.2);
    assert!(close_v(m.normal, Vec3::Y, 1e-6));
    assert_eq!(m.count, 2, "{m:?}");
}

#[test]
fn coincident_and_degenerate_placements_give_finite_contacts() {
    let shapes = [
        sphere(0.5),
        capsule(0.5, 0.3),
        capsule(0.0, 0.3),
        cuboid(0.5, 0.2, 0.7),
        octahedron(0.6),
        tetrahedron(0.5),
        hull_cuboid(0.3, 0.3, 0.3),
    ];
    let placements = [
        Iso::IDENTITY,
        turned(Vec3::ZERO, Vec3::Y, FRAC_PI_2),
        turned(Vec3::ZERO, v(1.0, 1.0, 1.0), 1.0),
    ];
    for a in &shapes {
        for b in &shapes {
            for pb in &placements {
                // Same centre: deep overlap, still one finite manifold.
                let m = one(a, &Iso::IDENTITY, b, pb, 0.0);
                assert!(m.min_separation() < 0.0, "{m:?}");
            }
            // Exactly touching along +Y: separation 0 (or a contact within the tolerance).
            let out = run(a, &Iso::IDENTITY, b, &at(0.0, 5.0, 0.0), 0.0);
            assert!(out.is_empty());
        }
    }
    // Exactly touching faces, edges and corners of boxes.
    let b = cuboid(0.5, 0.5, 0.5);
    for offset in [
        v(1.0, 0.0, 0.0),
        v(1.0, 1.0, 0.0),
        v(1.0, 1.0, 1.0),
        v(1.0, 0.3, 0.2),
    ] {
        let m = one(&b, &Iso::IDENTITY, &b, &Iso::from_position(offset), 0.0);
        assert!(close(m.min_separation(), 0.0, 1e-6), "{offset:?} {m:?}");
    }
    // A sphere centred exactly on a box corner, edge and face.
    for c in [v(0.5, 0.5, 0.5), v(0.5, 0.5, 0.0), v(0.5, 0.0, 0.0)] {
        let m = one(
            &sphere(0.1),
            &Iso::from_position(c),
            &b,
            &Iso::IDENTITY,
            0.0,
        );
        assert!(close(m.points[0].separation, -0.1, 1e-6), "{m:?}");
    }
    // A capsule's core exactly along a box edge.
    let edge = Iso::new(v(0.5, 0.5, 0.0), Quat::from_axis_angle(Vec3::X, FRAC_PI_2));
    let m = one(&capsule(0.3, 0.1), &edge, &b, &Iso::IDENTITY, 0.0);
    assert!(close(m.min_separation(), -0.1, 1e-5), "{m:?}");
    // Everything on a mesh vertex and in a mesh's plane.
    let floor = Shape::Mesh(Arc::new(grid_floor(2, 1.0)));
    for s in &shapes {
        let out = run(&floor, &Iso::IDENTITY, s, &Iso::IDENTITY, 0.0);
        for m in &out {
            assert!(close_v(m.normal, Vec3::Y, 1e-5), "{m:?}");
        }
    }
}

#[test]
fn tiny_and_huge_shapes_collide_sensibly() {
    for scale in [1e-4f32, 1e-2, 1.0, 1e3, 1e5] {
        let b = cuboid(scale, scale, scale);
        let gap = -0.01 * scale;
        let m = one(
            &b,
            &Iso::IDENTITY,
            &b,
            &at(0.3 * scale, 2.0 * scale + gap, 0.0),
            0.0,
        );
        assert!(close_v(m.normal, Vec3::Y, 1e-5));
        assert_eq!(m.count, 4, "{scale}: {m:?}");
        for p in m.points() {
            assert!(close(p.separation, gap, 1e-4 * scale), "{scale}: {m:?}");
        }
        let s = one(
            &sphere(scale),
            &Iso::IDENTITY,
            &b,
            &at(0.0, 2.0 * scale + gap, 0.0),
            0.0,
        );
        assert!(
            close(s.points[0].separation, gap, 1e-4 * scale),
            "{scale}: {s:?}"
        );
        let c = one(
            &capsule(scale, scale),
            &Iso::IDENTITY,
            &b,
            &at(0.0, 3.0 * scale + gap, 0.0),
            0.0,
        );
        assert!(
            close(c.min_separation(), gap, 1e-4 * scale),
            "{scale}: {c:?}"
        );
        // A small box on a big floor.
        let floor = Shape::Mesh(Arc::new(grid_floor(2, 1e3)));
        let m = one(&floor, &Iso::IDENTITY, &b, &at(0.0, scale + gap, 0.0), 0.0);
        assert!(close_v(m.normal, Vec3::Y, 1e-5));
        for p in m.points() {
            assert!(
                close(p.separation, gap, 1e-4 * scale.max(1.0)),
                "{scale}: {m:?}"
            );
        }
    }
}

#[test]
fn random_placements_of_every_pair_keep_the_contract() {
    let mesh = Shape::Mesh(Arc::new(grid_floor(4, 0.7)));
    let heights: Vec<f32> = (0..49).map(|i| ((i * 13) % 7) as f32 * 0.15).collect();
    let terrain = Shape::Heightfield(Arc::new(
        Heightfield::new(7, 7, heights, v(0.5, 1.0, 0.5)).unwrap(),
    ));
    let shapes = [
        sphere(0.4),
        capsule(0.6, 0.25),
        capsule(0.0, 0.2),
        cuboid(0.5, 0.3, 0.4),
        cuboid(1e-3, 0.5, 0.5),
        octahedron(0.7),
        tetrahedron(0.5),
        hull_cuboid(0.4, 0.2, 0.6),
        mesh,
        terrain,
    ];
    let mut rng = Rng(77);
    let mut contacts = 0;
    for i in 0..6000 {
        let a = &shapes[i % shapes.len()];
        let b = &shapes[(i / shapes.len()) % shapes.len()];
        let spread = if i % 3 == 0 { 0.3 } else { 1.5 };
        let ia = Iso::new(rng.vec(spread) + v(1.2, 0.3, 1.2), rng.rotation());
        let ib = Iso::new(rng.vec(spread) + v(1.2, 0.3, 1.2), rng.rotation());
        let spec = rng.range(0.0, 0.2);
        let out = run(a, &ia, b, &ib, spec);
        contacts += out.len();
        // Deterministic: the same call gives the same bits.
        let again = run(a, &ia, b, &ib, spec);
        assert_eq!(out, again);
        if a.is_convex() && b.is_convex() {
            assert!(out.len() <= 1);
        } else {
            assert!(out.len() <= MAX_MESH_MANIFOLDS);
            for w in out.windows(2) {
                assert!(w[0].min_separation() <= w[1].min_separation());
            }
        }
    }
    assert!(contacts > 1000, "{contacts}");
}

/// The manifold with normal `n` among a mesh contact's; any other must be shallower (the
/// heightfield marks every edge active, so flat seams may add speculative edge contacts).
fn manifold_along(out: &[Manifold], n: Vec3) -> Manifold {
    let m = out
        .iter()
        .find(|m| close_v(m.normal, n, 1e-5))
        .unwrap_or_else(|| panic!("no manifold along {n:?}: {out:?}"))
        .clone();
    for o in out {
        assert!(o.min_separation() >= m.min_separation() - 1e-6, "{out:?}");
    }
    m
}

#[test]
fn shapes_on_a_sloped_heightfield_are_separated_along_its_normal() {
    // A plane rising along +X with slope 0.5: y = 0.5 x.
    let heights: Vec<f32> = (0..16).map(|i| 0.5 * (i % 4) as f32).collect();
    let hf = Shape::Heightfield(Arc::new(
        Heightfield::new(4, 4, heights, v(1.0, 1.0, 1.0)).unwrap(),
    ));
    let n = v(-0.5, 1.0, 0.0).normalize_or_zero();
    let ground = |x: f32, z: f32| v(x, 0.5 * x, z);
    // A sphere 0.02 above the slope, over a seam between cells.
    let c = ground(1.5, 1.0) + n * (0.3 + 0.02);
    let m = manifold_along(
        &run(&hf, &Iso::IDENTITY, &sphere(0.3), &at(c.x, c.y, c.z), 0.05),
        n,
    );
    assert!(close(m.points[0].separation, 0.02, 1e-5), "{m:?}");
    // A capsule lying on the slope across cells, sunk 0.01: two points along the normal.
    let along = Quat::from_rotation_arc(Vec3::Y, v(1.0, 0.5, 0.0).normalize_or_zero());
    let centre = ground(1.4, 1.2) + n * (0.2 - 0.01);
    let out = run(
        &hf,
        &Iso::IDENTITY,
        &capsule(0.6, 0.2),
        &Iso::new(centre, along),
        0.05,
    );
    let m = manifold_along(&out, n);
    assert_eq!(m.count, 2, "{m:?}");
    for p in m.points() {
        assert!(close(p.separation, -0.01, 1e-5), "{m:?}");
    }
    // A box resting flat on it: four corners.
    let tilt = Quat::from_rotation_arc(Vec3::Y, n);
    let centre = ground(1.5, 1.5) + n * (0.25 - 0.003);
    let out = run(
        &hf,
        &Iso::IDENTITY,
        &cuboid(0.3, 0.25, 0.3),
        &Iso::new(centre, tilt),
        0.05,
    );
    let m = manifold_along(&out, n);
    assert_eq!(m.count, 4, "{m:?}");
    for p in m.points() {
        assert!(close(p.separation, -0.003, 1e-5), "{m:?}");
    }
}

#[test]
fn clipped_points_keep_their_ids_while_a_box_hangs_over_a_ledge() {
    // Half the box hangs over the +X edge of a step: two of its corners and two points where
    // the step's edge cuts the box's bottom face.
    let step = cuboid(1.0, 0.5, 1.0);
    let b = cuboid(0.3, 0.3, 0.3);
    let place = |dz: f32| at(1.0, 0.5 + 0.3 - 0.002, dz);
    let base = one(&step, &Iso::IDENTITY, &b, &place(0.0), 0.01);
    assert_eq!(base.count, 4, "{base:?}");
    let on_edge = base
        .points()
        .iter()
        .filter(|p| close(p.point.x, 1.0, 1e-5))
        .count();
    assert_eq!(on_edge, 2, "{base:?}");
    for dz in [-0.05f32, -0.01, 0.02, 0.06] {
        let m = one(&step, &Iso::IDENTITY, &b, &place(dz), 0.01);
        assert_eq!(m.count, 4);
        for p in base.points() {
            let moved = p.point + v(0.0, 0.0, dz);
            let q = m
                .points()
                .iter()
                .find(|q| close_v(q.point, moved, 1e-4))
                .expect("same point");
            assert_eq!(p.id, q.id, "{base:?} {m:?}");
        }
    }
}

#[test]
fn non_finite_placements_neither_panic_nor_return_nan() {
    let floor = Shape::Mesh(Arc::new(grid_floor(2, 1.0)));
    let shapes = [
        sphere(0.5),
        capsule(0.5, 0.2),
        cuboid(0.5, 0.5, 0.5),
        octahedron(0.5),
        floor,
    ];
    let bad = [
        Iso::from_position(v(f32::NAN, 0.0, 0.0)),
        Iso::from_position(v(f32::INFINITY, 0.0, 0.0)),
        Iso::new(Vec3::ZERO, Quat::new(f32::NAN, 0.0, 0.0, 1.0)),
        Iso::new(Vec3::ZERO, Quat::new(0.0, 0.0, 0.0, 0.0)),
    ];
    for a in &shapes {
        for b in &shapes {
            for ib in &bad {
                for spec in [0.0, f32::NAN, f32::INFINITY, -1.0] {
                    let mut out = Vec::new();
                    collide(a, &Iso::IDENTITY, b, ib, spec, &mut out);
                    for m in &out {
                        assert!(m.normal.is_finite());
                        for p in m.points() {
                            assert!(p.point.is_finite() && p.separation.is_finite());
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn round_pairs_match_brute_force_distances_between_their_cores() {
    let mut rng = Rng(91);
    let mut seen = 0;
    for i in 0..1500 {
        let (ha, hb) = (rng.range(0.0, 0.8), rng.range(0.0, 0.8));
        let (ra, rb) = (rng.range(0.05, 0.4), rng.range(0.05, 0.4));
        let ia = Iso::new(rng.vec(0.5), rng.rotation());
        let ib = Iso::new(rng.vec(1.5), rng.rotation());
        // Sphere-sphere, sphere-capsule and capsule-capsule by turns.
        let (a, ha) = if i % 3 < 2 {
            (sphere(ra), 0.0)
        } else {
            (capsule(ha, ra), ha)
        };
        let (b, hb) = if i % 3 == 0 {
            (sphere(rb), 0.0)
        } else {
            (capsule(hb, rb), hb)
        };
        let (a0, a1) = (ia.point(v(0.0, -ha, 0.0)), ia.point(v(0.0, ha, 0.0)));
        let (b0, b1) = (ib.point(v(0.0, -hb, 0.0)), ib.point(v(0.0, hb, 0.0)));
        let mut d = f32::INFINITY;
        for x in 0..=120 {
            for y in 0..=120 {
                let pa = a0.lerp(a1, x as f32 / 120.0);
                d = d.min((pa - b0.lerp(b1, y as f32 / 120.0)).length());
            }
        }
        if d < 1e-3 {
            continue;
        }
        let out = run(&a, &ia, &b, &ib, 0.2);
        if d - ra - rb > 0.2 + 1e-2 {
            assert!(out.is_empty(), "{out:?}");
            continue;
        }
        if d - ra - rb > 0.2 - 1e-2 {
            continue;
        }
        seen += 1;
        let s = out[0].min_separation();
        assert!(
            s <= d - ra - rb + 1e-4 && s >= d - ra - rb - 1e-2,
            "{d} {s} {out:?}"
        );
    }
    assert!(seen > 120, "{seen}");
}

#[test]
fn a_box_turned_on_an_equal_box_keeps_four_of_the_eight_overlap_corners() {
    let b = cuboid(0.5, 0.5, 0.5);
    let m = one(
        &b,
        &Iso::IDENTITY,
        &b,
        &turned(v(0.0, 0.999, 0.0), Vec3::Y, FRAC_PI_4),
        0.01,
    );
    assert_eq!(m.count, 4);
    // The overlap is a regular octagon of circumradius r; the best four corners make a square
    // of area 2 r².
    let r = 0.5 / (std::f32::consts::PI / 8.0).cos();
    let q: Vec<Vec3> = m.points().iter().map(|p| p.point).collect();
    let area = 0.5
        * ((q[0] - q[2]).cross(q[1] - q[3]).length())
            .max((q[0] - q[1]).cross(q[2] - q[3]).length())
            .max((q[0] - q[3]).cross(q[1] - q[2]).length());
    assert!(close(area, 2.0 * r * r, 1e-3), "{area} {m:?}");
    for p in m.points() {
        assert!(close(p.separation, -0.001, 1e-5));
        assert!(
            close((p.point - v(0.0, p.point.y, 0.0)).length(), r, 1e-4),
            "{m:?}"
        );
    }
}
