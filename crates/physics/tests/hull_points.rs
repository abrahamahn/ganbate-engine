//! `ConvexHull::from_points` against point clouds with known hulls, degenerate inputs and
//! extreme scales, and `ConvexHull::mass_properties` against closed forms.

use ganbate_physics::hull::{ConvexHull, MAX_HULL_VERTICES};
use ganbate_physics::{MassProperties, Mat3, Vec3};

/// A seeded linear congruential generator (no `rand` crate).
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform in `[-1, 1)`.
    fn signed(&mut self) -> f32 {
        (self.next_u32() >> 7) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    }
    fn in_box(&mut self) -> Vec3 {
        Vec3::new(self.signed(), self.signed(), self.signed())
    }
    fn in_ball(&mut self) -> Vec3 {
        loop {
            let v = self.in_box();
            if v.length_squared() <= 1.0 {
                return v;
            }
        }
    }
    fn on_sphere(&mut self) -> Vec3 {
        loop {
            let v = self.in_ball();
            if v.length_squared() > 0.01 {
                return v * (1.0 / v.length());
            }
        }
    }
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.next_u32() as usize % (i + 1));
        }
    }
}

/// What holds for the hull of any points: it validates, its vertices are distinct input points,
/// every input point is inside or on it, and a second run gives the same bits.
fn check(points: &[Vec3]) -> ConvexHull {
    let hull = ConvexHull::from_points(points).unwrap_or_else(|e| panic!("{e}"));
    hull.validate().unwrap();
    for (i, v) in hull.vertices.iter().enumerate() {
        assert!(
            points.contains(v),
            "hull vertex {v:?} is not an input point"
        );
        assert!(
            !hull.vertices[..i].contains(v),
            "hull vertex {v:?} is repeated"
        );
    }
    let scale = points
        .iter()
        .fold(0.0f32, |m, p| m.max(p.abs().max_element()))
        .max(1e-6);
    let tol = 1e-5 * scale;
    for p in points {
        for (f, face) in hull.faces.iter().enumerate() {
            let d = face.normal.dot(*p) - face.offset;
            assert!(
                d <= tol,
                "point {p:?} is {d} in front of face {f} (tol {tol})"
            );
        }
    }
    assert_eq!(
        ConvexHull::from_points(points).unwrap(),
        hull,
        "two runs differ"
    );
    hull
}

#[test]
fn random_clouds_in_balls_and_boxes_with_duplicates() {
    let mut rng = Lcg(1);
    for (round, n) in [4usize, 5, 8, 20, 100, 500, 2000].into_iter().enumerate() {
        for ball in [true, false] {
            let mut points: Vec<Vec3> = (0..n)
                .map(|_| if ball { rng.in_ball() } else { rng.in_box() })
                .collect();
            check(&points);
            // Exact and near duplicates (a few units of rounding away) change nothing.
            let copies: Vec<Vec3> = points.iter().step_by(3).copied().collect();
            points.extend(copies.iter().map(|p| *p * (1.0 + 2.0 * f32::EPSILON)));
            points.extend(copies);
            rng.shuffle(&mut points);
            let h = check(&points);
            assert!(h.vertices.len() >= 4, "round {round}");
        }
    }
}

#[test]
fn the_surface_of_a_grid_gives_the_eight_corners_of_its_cube() {
    let at = |i: usize| -1.0 + 2.0 * i as f32 / 9.0;
    let mut points = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            for k in 0..10 {
                if [i, j, k].iter().any(|&c| c == 0 || c == 9) {
                    points.push(Vec3::new(at(i), at(j), at(k)));
                }
            }
        }
    }
    assert_eq!(points.len(), 488);
    let mut rng = Lcg(7);
    for round in 0..4 {
        if round > 0 {
            rng.shuffle(&mut points);
        }
        let h = check(&points);
        assert_eq!((h.vertices.len(), h.faces.len()), (8, 6), "round {round}");
        for v in &h.vertices {
            assert_eq!(v.abs(), Vec3::ONE);
        }
        for f in 0..6 {
            assert_eq!(h.face_vertices(f).len(), 4);
        }
    }
}

#[test]
fn points_on_a_sphere_are_all_vertices() {
    let mut rng = Lcg(3);
    for n in [10, 50, 200] {
        let points: Vec<Vec3> = (0..n).map(|_| rng.on_sphere() * 3.0).collect();
        let h = check(&points);
        assert_eq!(h.vertices.len(), n);
    }
}

#[test]
fn simple_solids_and_their_inner_points() {
    let tet = [
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
    ];
    let h = check(&tet);
    assert_eq!((h.vertices.len(), h.faces.len()), (4, 4));
    // Edge midpoints, face centres and the centre are not vertices.
    let mut points = tet.to_vec();
    for i in 0..4 {
        for j in i + 1..4 {
            points.push((tet[i] + tet[j]) * 0.5);
        }
    }
    points.push(Vec3::ZERO);
    points.push((tet[0] + tet[1] + tet[2]) * (1.0 / 3.0));
    let h = check(&points);
    assert_eq!((h.vertices.len(), h.faces.len()), (4, 4));

    let mut octa = vec![Vec3::ZERO];
    for a in [Vec3::X, Vec3::Y, Vec3::Z] {
        octa.push(a * 2.0);
        octa.push(a * -2.0);
    }
    let h = check(&octa);
    assert_eq!(
        (h.vertices.len(), h.edges.len() / 2, h.faces.len()),
        (6, 12, 8)
    );

    // A box's corners, edge midpoints and face centres.
    let mut points = Vec::new();
    for x in [-1.0, 0.0, 1.0] {
        for y in [-2.0, 0.0, 2.0] {
            for z in [-0.5, 0.0, 0.5] {
                points.push(Vec3::new(x, y, z));
            }
        }
    }
    let h = check(&points);
    assert_eq!((h.vertices.len(), h.faces.len()), (8, 6));
}

#[test]
fn degenerate_inputs_are_refused() {
    let p = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
    assert!(ConvexHull::from_points(&[]).is_err());
    assert!(ConvexHull::from_points(&p[..3]).is_err());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut q = p.to_vec();
        q[2].y = bad;
        let e = ConvexHull::from_points(&q).unwrap_err();
        assert!(e.contains("finite"), "{e}");
    }
    // The same point many times, and four points of which two coincide.
    assert!(ConvexHull::from_points(&[Vec3::ONE; 10]).is_err());
    assert!(ConvexHull::from_points(&[Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Y]).is_err());
    let mut rng = Lcg(5);
    // Coplanar, nearly coplanar (off the plane by far less than the welding tolerance) and
    // tilted coplanar.
    let flat: Vec<Vec3> = (0..200)
        .map(|_| Vec3::new(rng.signed(), rng.signed(), 0.0))
        .collect();
    assert!(ConvexHull::from_points(&flat).is_err());
    let nearly: Vec<Vec3> = flat
        .iter()
        .map(|p| Vec3::new(p.x, p.y, rng.signed() * 1e-8))
        .collect();
    let e = ConvexHull::from_points(&nearly).unwrap_err();
    assert!(e.contains("coplanar"), "{e}");
    let tilted: Vec<Vec3> = flat
        .iter()
        .map(|p| Vec3::new(p.x, p.y, 0.3 * p.x - 0.7 * p.y + 2.0))
        .collect();
    let e = ConvexHull::from_points(&tilted).unwrap_err();
    assert!(e.contains("coplanar"), "{e}");
    // Collinear, exactly and nearly.
    let line: Vec<Vec3> = (0..50)
        .map(|_| {
            let t = rng.signed();
            Vec3::new(t, 2.0 * t + 1.0, -t) + Vec3::new(rng.signed(), rng.signed(), 0.0) * 1e-8
        })
        .collect();
    let e = ConvexHull::from_points(&line).unwrap_err();
    assert!(e.contains("collinear"), "{e}");
}

#[test]
fn huge_tiny_and_far_away_clouds() {
    let mut rng = Lcg(11);
    let cloud: Vec<Vec3> = (0..300).map(|_| rng.in_ball()).collect();
    let base = check(&cloud);
    for scale in [1e5f32, 1e-3] {
        let points: Vec<Vec3> = cloud.iter().map(|p| *p * scale).collect();
        let h = check(&points);
        // Scaling by a power of ten rounds, but the hull stays about the same size.
        let n = h.vertices.len() as f32 / base.vertices.len() as f32;
        assert!((0.8..1.25).contains(&n), "{scale}: {n}");
    }
    let offset = Vec3::new(1e4, -1e4, 1e4);
    let points: Vec<Vec3> = cloud.iter().map(|p| *p + offset).collect();
    check(&points);
    let points: Vec<Vec3> = cloud.iter().map(|p| *p * 100.0 + offset).collect();
    check(&points);
}

#[test]
fn too_many_hull_vertices_are_refused() {
    let mut rng = Lcg(13);
    let points: Vec<Vec3> = (0..MAX_HULL_VERTICES + 50)
        .map(|_| rng.on_sphere())
        .collect();
    let e = ConvexHull::from_points(&points).unwrap_err();
    assert!(e.contains(&MAX_HULL_VERTICES.to_string()), "{e}");
    // Ten thousand points on a sphere are refused without building them all.
    let points: Vec<Vec3> = (0..10_000).map(|_| rng.on_sphere()).collect();
    assert!(ConvexHull::from_points(&points).is_err());
    // Exactly the limit is fine.
    let points: Vec<Vec3> = (0..MAX_HULL_VERTICES).map(|_| rng.on_sphere()).collect();
    assert_eq!(check(&points).vertices.len(), MAX_HULL_VERTICES);
}

fn assert_mat_near(a: &Mat3, b: &Mat3, tol: f32) {
    for c in 0..3 {
        for r in 0..3 {
            let (x, y) = (a.cols[c][r], b.cols[c][r]);
            assert!((x - y).abs() <= tol, "[{r}][{c}]: {x} vs {y}\n{a:?}\n{b:?}");
        }
    }
}

fn assert_mass_near(a: &MassProperties, b: &MassProperties, rel: f32) {
    assert!(
        (a.mass - b.mass).abs() <= rel * b.mass,
        "{} vs {}",
        a.mass,
        b.mass
    );
    let size = b.inertia.diagonal().max_element();
    assert!(
        (a.center - b.center).length() <= rel * (size / b.mass).sqrt(),
        "{:?} vs {:?}",
        a.center,
        b.center
    );
    assert_mat_near(&a.inertia, &b.inertia, rel * size);
}

#[test]
fn a_box_hull_has_the_mass_of_a_box() {
    let (h, rho) = (Vec3::new(0.5, 2.0, 1.5), 2.5);
    let mass = rho * 8.0 * h.x * h.y * h.z;
    let expect = MassProperties {
        mass,
        center: Vec3::ZERO,
        inertia: Mat3::from_diagonal(Vec3::new(
            h.y * h.y + h.z * h.z,
            h.x * h.x + h.z * h.z,
            h.x * h.x + h.y * h.y,
        ))
        .scale(mass / 3.0),
    };
    let cuboid = ConvexHull::cuboid(h);
    assert_mass_near(&cuboid.mass_properties(rho), &expect, 1e-5);
    let corners = cuboid.vertices.clone();
    let from_points = check(&corners);
    assert_mass_near(&from_points.mass_properties(rho), &expect, 1e-5);
}

#[test]
fn a_regular_tetrahedron_has_isotropic_inertia() {
    let tet = [
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
    ];
    // Edge s = 2√2, volume s³ / (6√2) = 8/3, inertia m s² / 20 about every axis.
    let rho = 3.0;
    let mass = rho * 8.0 / 3.0;
    let expect = MassProperties {
        mass,
        center: Vec3::ZERO,
        inertia: Mat3::from_diagonal(Vec3::splat(mass * 8.0 / 20.0)),
    };
    let props = check(&tet).mass_properties(rho);
    assert_mass_near(&props, &expect, 1e-5);
    // Moved away from the origin, only the centre moves.
    let offset = Vec3::new(3.0, -2.0, 5.0);
    let moved: Vec<Vec3> = tet.iter().map(|p| *p + offset).collect();
    let props = check(&moved).mass_properties(rho);
    assert_mass_near(
        &props,
        &MassProperties {
            center: offset,
            ..expect
        },
        1e-5,
    );
}

#[test]
fn inertia_about_the_centre_does_not_depend_on_where_the_hull_is() {
    let mut rng = Lcg(17);
    let cloud: Vec<Vec3> = (0..200)
        .map(|_| rng.in_box().mul_elem(Vec3::new(1.0, 0.5, 2.0)))
        .collect();
    let base = check(&cloud).mass_properties(1.5);
    for offset in [Vec3::new(16.0, -8.0, 4.0), Vec3::new(-100.0, 50.0, 25.0)] {
        let moved: Vec<Vec3> = cloud.iter().map(|p| *p + offset).collect();
        let props = check(&moved).mass_properties(1.5);
        let expect = MassProperties {
            center: base.center + offset,
            ..base
        };
        assert_mass_near(&props, &expect, 1e-4);
    }
}

#[test]
fn the_parts_of_a_split_box_add_up_to_the_box() {
    let corners = |lo: Vec3, hi: Vec3| -> Vec<Vec3> {
        (0..8)
            .map(|i| {
                Vec3::new(
                    if i & 1 != 0 { hi.x } else { lo.x },
                    if i & 2 != 0 { hi.y } else { lo.y },
                    if i & 4 != 0 { hi.z } else { lo.z },
                )
            })
            .collect()
    };
    let (lo, hi) = (Vec3::new(-1.0, -2.0, -0.5), Vec3::new(1.0, 2.0, 0.5));
    let whole = check(&corners(lo, hi)).mass_properties(2.0);
    // Split along a slanted plane: two wedges.
    let left = [
        corners(lo, hi)[0],
        corners(lo, hi)[2],
        corners(lo, hi)[4],
        corners(lo, hi)[6],
        Vec3::new(0.25, -2.0, -0.5),
        Vec3::new(-0.5, 2.0, -0.5),
        Vec3::new(0.25, -2.0, 0.5),
        Vec3::new(-0.5, 2.0, 0.5),
    ];
    let right = [
        corners(lo, hi)[1],
        corners(lo, hi)[3],
        corners(lo, hi)[5],
        corners(lo, hi)[7],
        left[4],
        left[5],
        left[6],
        left[7],
    ];
    let parts = [
        check(&left).mass_properties(2.0),
        check(&right).mass_properties(2.0),
    ];
    assert_mass_near(&MassProperties::combine(&parts), &whole, 1e-5);
}
