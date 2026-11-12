//! `gjk::cast` from a start inside its tolerance band (review finding GJK-CAST-BAND).
//!
//! A cast reports a hit once the shapes are within `target` plus a small tolerance, and aims each
//! step into the middle of that band, so a shape placed where a cast stopped starts inside it.
//! Before, the band test came first: from there any cast was a hit at fraction 0 (not
//! `started_overlapping`), even straight away from the surface or along it. And a slide over a
//! surface closing in on a coplanar neighbour's edge (a seam between floor tiles) was a hit at
//! the seam. Now a hit inside the band needs the shapes still closing, and a probe one full step
//! ahead must confirm the approach. Every scene here is synthetic.

use std::sync::Arc;

use ganbate_physics::gjk::{cast, distance};
use ganbate_physics::hull::ConvexHull;
use ganbate_physics::mesh::Heightfield;
use ganbate_physics::{
    BodyDesc, BodyKind, Character, CharacterSettings, ColliderDesc, Convex, ConvexShape, Iso,
    PhysicsWorld, Quat, QueryFilter, Shape, Vec3,
};

/// `shape` moved by a first cast along `path` onto `onto`, to where that cast stopped (inside
/// the band), checked to be apart by more than `target`.
fn placed_by_a_cast<'a>(
    shape: ConvexShape<'a>,
    from: Vec3,
    path: Vec3,
    onto: &Convex,
    target: f32,
) -> Convex<'a> {
    let start = Convex::new(shape, Iso::from_position(from));
    let hit = cast(&start, path, onto, target).expect("the first cast lands");
    assert!(!hit.started_overlapping && hit.fraction > 0.0, "{hit:?}");
    let placed = Convex::new(shape, Iso::from_position(from + path * hit.fraction));
    let gap = distance(&placed, onto);
    assert!(!gap.overlap && gap.distance > target, "{gap:?}");
    placed
}

#[test]
fn casts_from_inside_the_band_that_move_away_or_along_miss() {
    let boxed = Convex::new(ConvexShape::Box(Vec3::ONE), Iso::IDENTITY);
    let turned = Convex::new(
        ConvexShape::Box(Vec3::new(2.0, 0.5, 1.5)),
        Iso::new(
            Vec3::new(0.3, -0.2, 0.1),
            Quat::from_axis_angle(Vec3::new(0.2, 1.0, -0.3), 0.7),
        ),
    );
    let capsule = ConvexShape::Capsule {
        half_height: 0.4,
        radius: 0.2,
    };
    for (k, (shape, onto, target)) in [
        (ConvexShape::Sphere(0.5), &boxed, 0.0),
        (ConvexShape::Sphere(0.5), &boxed, 0.05),
        (ConvexShape::Box(Vec3::splat(0.25)), &boxed, 0.0),
        (capsule, &turned, 0.0),
        (capsule, &turned, 0.02),
    ]
    .into_iter()
    .enumerate()
    {
        let placed = placed_by_a_cast(
            shape,
            Vec3::new(0.2, 4.0, 0.1),
            Vec3::new(0.0, -6.0, 0.0),
            onto,
            target,
        );
        let n = distance(&placed, onto).normal;
        // Directions straight away, along the surface (both ways, and diagonally), away and
        // along at once, and no motion at all.
        let side = n.cross(Vec3::Z).normalize_or_zero();
        let other = n.cross(side);
        for t in [
            -n * 2.0,
            side * 3.0,
            -side * 3.0,
            other * 2.5,
            (side + other) * 1.5,
            (side - n) * 1.2,
            Vec3::ZERO,
        ] {
            let got = cast(&placed, t, onto, target);
            assert_eq!(got, None, "case {k}: cast by {t:?} from the band");
        }
        // Into the surface it is still a hit, at once and not as a start inside the target.
        let into = cast(&placed, n * 1.0 + side * 0.3, onto, target).expect("into the surface");
        assert!(
            into.fraction == 0.0 && !into.started_overlapping,
            "case {k}: {into:?}"
        );
        assert!((into.normal + n).length() < 1e-3, "case {k}: {into:?}");
    }
}

#[test]
fn a_cast_sliding_onto_a_coplanar_neighbour_passes_the_seam() {
    // Two boxes side by side, their tops level at y = 0, meeting at x = 0; and a third whose
    // top is a centimetre higher (a real step) beyond them.
    let tile = |x: f32, top: f32| {
        Convex::new(
            ConvexShape::Box(Vec3::new(2.0, 0.5, 2.0)),
            Iso::from_position(Vec3::new(x, top - 0.5, 0.0)),
        )
    };
    let (first, second, step) = (tile(-2.0, 0.0), tile(2.0, 0.0), tile(6.0, 0.01));
    for shape in [
        ConvexShape::Box(Vec3::new(0.25, 0.25, 0.25)),
        ConvexShape::Box(Vec3::new(0.4, 0.05, 0.3)),
        ConvexShape::Sphere(0.3),
        ConvexShape::Capsule {
            half_height: 0.3,
            radius: 0.2,
        },
    ] {
        let placed = placed_by_a_cast(
            shape,
            Vec3::new(-1.0, 3.0, 0.2),
            Vec3::new(0.0, -5.0, 0.0),
            &first,
            0.0,
        );
        // Along the floor towards and over the seam: the next tile is never closer than the
        // floor already is.
        for along in [Vec3::new(4.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.7)] {
            assert_eq!(cast(&placed, along, &second, 0.0), None, "{shape:?}");
            assert_eq!(cast(&placed, along, &first, 0.0), None, "{shape:?}");
        }
        // A step a centimetre up is in the way: hit on its near face or edge at x = 4, against
        // the motion (a box's flat front on the face, a rounded one on the edge).
        let hit = cast(&placed, Vec3::new(8.0, 0.0, 0.0), &step, 0.0).expect("the step");
        assert!(
            !hit.started_overlapping
                && (hit.point.x - 4.0).abs() < 1e-4
                && hit.point.y <= 0.01 + 1e-4
                && hit.normal.x < -0.2,
            "{shape:?}: {hit:?}"
        );
    }
}

fn static_box(w: &mut PhysicsWorld, center: Vec3, half: Vec3) {
    let g = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: center,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(g, &ColliderDesc::new(Shape::Box { half_extents: half }))
        .unwrap();
}

#[test]
fn shape_casts_slide_across_a_tiled_floor() {
    let mut w = PhysicsWorld::default();
    // Eight 1 m tiles in a row, tops at y = 0, and a 2 cm kerb past them at x = 4.
    for i in 0..8 {
        static_box(
            &mut w,
            Vec3::new(i as f32 - 3.5, -0.5, 0.0),
            Vec3::new(0.5, 0.5, 3.0),
        );
    }
    static_box(&mut w, Vec3::new(4.5, -0.49, 0.0), Vec3::new(0.5, 0.5, 3.0));
    let all = QueryFilter::default();
    for (k, shape) in [
        Shape::Box {
            half_extents: Vec3::splat(0.25),
        },
        Shape::Box {
            half_extents: Vec3::new(0.4, 0.1, 0.3),
        },
        Shape::Sphere { radius: 0.3 },
        Shape::Capsule {
            half_height: 0.3,
            radius: 0.2,
        },
    ]
    .iter()
    .enumerate()
    {
        let start = Iso::from_position(Vec3::new(-3.3 + 0.07 * k as f32, 3.0, 0.4));
        let down = Vec3::new(0.0, -5.0, 0.0);
        let drop = w.shape_cast(shape, &start, down, &all).unwrap();
        let placed = Iso::from_position(start.position + down * drop.fraction);
        // Over seven seams: nothing in the way short of the kerb.
        let slide = w.shape_cast(shape, &placed, Vec3::new(6.0, 0.0, 0.0), &all);
        assert_eq!(slide, None, "{shape:?} slid into a seam");
        let lift = w.shape_cast(shape, &placed, Vec3::new(0.0, 1.0, 0.0), &all);
        assert_eq!(lift, None, "{shape:?}");
        // On to the kerb: its near face or edge stops the shape.
        let hit = w
            .shape_cast(shape, &placed, Vec3::new(9.0, 0.0, 0.0), &all)
            .expect("the kerb");
        assert!(
            !hit.started_overlapping && (hit.point.x - 4.0).abs() < 1e-4 && hit.normal.x < -0.2,
            "{shape:?}: {hit:?}"
        );
    }
}

/// The review's CHAR-SEAM-ITERATIONS case: the character stands inside the band of its ground
/// casts, so on a finely tessellated floor every seam it walked onto was a touch at fraction 0
/// that used up one of its slide iterations, and a step's motion ran out of them.
#[test]
fn a_character_walks_the_whole_way_over_fine_ground() {
    const DT: f32 = 1.0 / 60.0;
    let (cell, n) = (0.05f32, 280u32);
    let mut w = PhysicsWorld::default();
    let g = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(-(n as f32) * cell * 0.5, 0.0, -(n as f32) * cell * 0.5),
            ..Default::default()
        })
        .unwrap();
    let field = Heightfield::new(
        n,
        n,
        vec![0.0; (n * n) as usize],
        Vec3::new(cell, 1.0, cell),
    );
    w.add_collider(
        g,
        &ColliderDesc::new(Shape::Heightfield(Arc::new(field.unwrap()))),
    )
    .unwrap();
    let s = CharacterSettings::default();
    let stand = s.half_height + s.radius + s.skin;
    let mut c = Character::new(Vec3::new(-3.0, stand + 0.05, -3.0), s);
    for _ in 0..30 {
        c.move_and_slide(&mut w, Vec3::Y * -0.01, DT);
        w.step(DT);
    }
    let v = Vec3::new(1.0, 0.0, 0.83).normalize_or_zero() * 10.0;
    let start = c.position;
    for _ in 0..60 {
        c.move_and_slide(&mut w, v * DT - Vec3::Y * (0.1 * DT), DT);
        w.step(DT);
    }
    let walked = Vec3::new(c.position.x - start.x, 0.0, c.position.z - start.z).length();
    assert!(walked > 9.99, "walked {walked} m of 10 in 1 s at 10 m/s");
}

/// A small deterministic generator (xorshift64).
struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    fn vec(&mut self, r: f32) -> Vec3 {
        Vec3::new(self.range(-r, r), self.range(-r, r), self.range(-r, r))
    }
    fn dir(&mut self) -> Vec3 {
        loop {
            let v = self.vec(1.0);
            let l = v.length();
            if l > 0.1 && l <= 1.0 {
                return v * (1.0 / l);
            }
        }
    }
    fn rotation(&mut self) -> Quat {
        let axis = self.dir();
        Quat::from_axis_angle(axis, self.range(0.0, std::f32::consts::TAU))
    }
}

/// From starts inside the band (each placed by a first cast), casts in random directions, some
/// nearly along the contact plane, against the distance sampled along the path: every path that
/// comes within the target is a hit and never past the sampled impact, and no path that never
/// comes closer than its start is.
#[test]
fn from_the_band_real_hits_stay_and_receding_paths_miss() {
    let hull = ConvexHull::from_points(&[
        Vec3::new(0.4, 0.0, 0.0),
        Vec3::new(-0.3, 0.2, 0.1),
        Vec3::new(0.0, -0.35, 0.2),
        Vec3::new(0.1, 0.1, -0.4),
        Vec3::new(-0.2, -0.2, -0.2),
        Vec3::new(0.2, 0.3, 0.3),
    ])
    .unwrap();
    let mut rng = Rng(0x5eed_1234_abcd);
    let (mut starts, mut real, mut receding) = (0, 0, 0);
    for case in 0..900 {
        let mut shape = |k: usize| match k % 6 {
            0 => ConvexShape::Sphere(rng.range(0.1, 0.6)),
            1 => ConvexShape::Capsule {
                half_height: rng.range(0.0, 0.8),
                radius: rng.range(0.05, 0.4),
            },
            2 => ConvexShape::Box(Vec3::new(
                rng.range(0.05, 1.0),
                rng.range(0.05, 1.0),
                rng.range(0.05, 1.0),
            )),
            3 => ConvexShape::Triangle([rng.vec(1.0), rng.vec(1.0), rng.vec(1.0)]),
            4 => ConvexShape::Hull(&hull),
            _ => ConvexShape::Box(Vec3::new(rng.range(1.0, 30.0), 0.5, rng.range(1.0, 30.0))),
        };
        let (sa, sb) = (shape(case), shape(case / 6 + 1));
        let a0 = Convex::new(
            sa,
            Iso::new(Vec3::new(-4.0, 0.0, 0.0) + rng.vec(1.0), rng.rotation()),
        );
        let b = Convex::new(sb, Iso::new(rng.vec(0.5), rng.rotation()));
        let target = [0.0f32, 0.0, 0.02, 0.1][case % 4];
        let t0 = Vec3::new(8.0, 0.0, 0.0) + rng.vec(1.5);
        let d0 = distance(&a0, &b);
        if d0.overlap || d0.distance <= target + 1e-3 {
            continue;
        }
        let Some(first) = cast(&a0, t0, &b, target) else {
            continue;
        };
        let a = Convex::new(
            sa,
            Iso::new(a0.iso.position + t0 * first.fraction, a0.iso.rotation),
        );
        let start = distance(&a, &b);
        if start.overlap || start.distance <= target {
            continue;
        }
        starts += 1;
        let mut dir = rng.dir();
        if case % 3 == 0 {
            let n = start.normal;
            dir = (dir - n * dir.dot(n)).normalize_or_zero();
            if dir == Vec3::ZERO {
                continue;
            }
            dir = (dir + n * rng.range(-0.02, 0.02)).normalize_or_zero();
        }
        let translation = dir * rng.range(0.3, 3.0);
        let at = |f: f32| {
            let moved = Convex::new(
                sa,
                Iso::new(a.iso.position + translation * f, a.iso.rotation),
            );
            let r = distance(&moved, &b);
            if r.overlap { -1.0 } else { r.distance }
        };
        let samples = 1500;
        let gaps: Vec<f32> = (0..=samples)
            .map(|i| at(i as f32 / samples as f32))
            .collect();
        let least = gaps.iter().copied().fold(f32::INFINITY, f32::min);
        let result = cast(&a, translation, &b, target);
        if least < target - 5e-5 {
            real += 1;
            let hit = result.unwrap_or_else(|| {
                panic!("case {case}: a path reaching {least} (target {target}) was missed")
            });
            let impact = gaps.iter().position(|&d| d <= target).unwrap() as f32 / samples as f32;
            assert!(
                hit.fraction <= impact + 1e-3,
                "case {case}: {hit:?} past the impact at {impact}"
            );
        } else if gaps.iter().all(|&d| d >= start.distance - 2e-6) {
            receding += 1;
            assert_eq!(
                result, None,
                "case {case}: from {} (target {target}) along {translation:?}",
                start.distance
            );
        }
    }
    assert!(
        starts > 350 && real > 120 && receding > 120,
        "{starts} starts, {real} real hits, {receding} receding paths"
    );
}
