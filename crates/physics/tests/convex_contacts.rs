//! Contacts between convex shapes that must hold still and be complete: a resting manifold keeps
//! its points (and their ids) while the bodies rock by a micron, an irregular hull's manifold holds
//! its deepest vertex, and large hulls collide in about linear time. Every scene is synthetic.

use std::sync::Arc;
use std::time::Instant;

use ganbate_physics::collide::{Manifold, collide};
use ganbate_physics::hull::ConvexHull;
use ganbate_physics::math::{Iso, Quat, Vec3};
use ganbate_physics::mesh::TriangleMesh;
use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, ColliderId, PhysicsWorld, Settings, Shape,
};

const DT: f32 = 1.0 / 60.0;

// ---------------------------------------------------------------------------------------------
// Helpers.

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

fn cube(half: f32) -> Shape {
    Shape::Box {
        half_extents: Vec3::splat(half),
    }
}

fn yaw(angle: f32) -> Quat {
    Quat::from_axis_angle(Vec3::Y, angle)
}

fn hull(points: &[Vec3]) -> Shape {
    Shape::Hull(Arc::new(ConvexHull::from_points(points).unwrap()))
}

/// An upright prism with `n` sides, circumradius `r` and half height `h`.
fn prism(n: usize, r: f32, h: f32) -> Shape {
    let mut points = Vec::new();
    for i in 0..n {
        let a = i as f32 * std::f32::consts::TAU / n as f32;
        // Exact corners for the square, so it is a box.
        let (s, c) = if n == 4 {
            [(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)][i]
        } else {
            (a.sin(), a.cos())
        };
        points.push(v(r * c, -h, r * s));
        points.push(v(r * c, h, r * s));
    }
    hull(&points)
}

/// `n` points spread evenly over a sphere of radius `r`: every one is a hull vertex.
fn ball(n: usize, r: f32) -> Shape {
    let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
    let points: Vec<Vec3> = (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let ring = (1.0 - y * y).sqrt();
            let a = golden * i as f32;
            v(ring * a.cos(), y, ring * a.sin()) * r
        })
        .collect();
    hull(&points)
}

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
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
    fn unit(&mut self) -> Vec3 {
        loop {
            let p = v(
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            );
            let l = p.length();
            if l > 0.1 && l <= 1.0 {
                return p * (1.0 / l);
            }
        }
    }
    /// A uniformly random rotation.
    fn rotation(&mut self) -> Quat {
        loop {
            let q = Quat::new(
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            );
            let l = q.length();
            if l > 0.1 && l <= 1.0 {
                return q.normalize();
            }
        }
    }
}

/// An irregular rock: `n` random points at 0.3 to 0.5 m from its centre.
fn random_rock(rng: &mut Rng, n: usize) -> Shape {
    let points: Vec<Vec3> = (0..n).map(|_| rng.unit() * rng.range(0.3, 0.5)).collect();
    hull(&points)
}

fn sorted_ids(m: &Manifold) -> Vec<u32> {
    let mut ids: Vec<u32> = m.points().iter().map(|p| p.id).collect();
    ids.sort_unstable();
    ids
}

fn world(sleep: bool) -> PhysicsWorld {
    PhysicsWorld::new(Settings {
        enable_sleep: sleep,
        ..Default::default()
    })
}

fn add(
    w: &mut PhysicsWorld,
    kind: BodyKind,
    shape: Shape,
    p: Vec3,
    q: Quat,
) -> (BodyId, ColliderId) {
    let b = w
        .create_body(&BodyDesc {
            kind,
            position: p,
            rotation: q,
            ..Default::default()
        })
        .unwrap();
    let c = w.add_collider(b, &ColliderDesc::new(shape)).unwrap();
    (b, c)
}

/// A static box whose top face is the plane y = 0.
fn ground(w: &mut PhysicsWorld) -> (BodyId, ColliderId) {
    add(
        w,
        BodyKind::Static,
        Shape::Box {
            half_extents: v(50.0, 0.5, 50.0),
        },
        v(0.0, -0.5, 0.0),
        Quat::IDENTITY,
    )
}

/// The sorted point ids of the manifolds between two colliders.
fn ids_between(w: &PhysicsWorld, a: ColliderId, b: ColliderId) -> Vec<u32> {
    let mut ids = Vec::new();
    for (x, y, m) in w.manifolds() {
        if (x == a && y == b) || (x == b && y == a) {
            ids.extend(m.points().iter().map(|p| p.id));
        }
    }
    ids.sort_unstable();
    ids
}

// ---------------------------------------------------------------------------------------------
// A resting manifold keeps its points.

#[test]
fn a_resting_polygon_keeps_its_four_points_however_the_body_rocks() {
    // Contact polygons with more than four corners: n-gon prisms standing 3 mm deep on a slab,
    // and a cube turned 45° on an equal cube (an octagon). Tilted by a micron or a tenth of a
    // millimetre across the face, towards any side, the same four corners must be kept.
    let slab = Shape::Box {
        half_extents: v(5.0, 0.5, 5.0),
    };
    let cases = [
        (
            "16-gon prism",
            &slab,
            prism(16, 0.5, 0.5),
            Quat::IDENTITY,
            32,
        ),
        ("8-gon prism", &slab, prism(8, 0.5, 0.5), Quat::IDENTITY, 16),
        ("6-gon prism", &slab, prism(6, 0.5, 0.5), Quat::IDENTITY, 12),
        (
            "cube turned 45°",
            &cube(0.5),
            cube(0.5),
            yaw(std::f32::consts::FRAC_PI_4),
            16,
        ),
    ];
    for (name, floor, shape, turn, directions) in &cases {
        let mut sets: Vec<Vec<u32>> = Vec::new();
        for tilt in [1e-6f32, 1e-5, 1e-4] {
            for i in 0..*directions {
                let a = i as f32 * std::f32::consts::TAU / *directions as f32;
                let axis = v(a.cos(), 0.0, a.sin());
                let iso = Iso::new(
                    v(0.0, 0.997, 0.0),
                    Quat::from_axis_angle(axis, tilt) * *turn,
                );
                let mut out = Vec::new();
                collide(floor, &Iso::IDENTITY, shape, &iso, 0.02, &mut out);
                assert_eq!(out.len(), 1, "{name}");
                assert_eq!(out[0].count, 4, "{name}: {:?}", out[0]);
                let ids = sorted_ids(&out[0]);
                if !sets.contains(&ids) {
                    sets.push(ids);
                }
            }
        }
        assert_eq!(sets.len(), 1, "{name}: {} different point sets", sets.len());
    }
}

#[test]
fn an_aligned_body_keeps_its_points_whichever_side_of_an_edge_its_corners_fall() {
    // A box on an equal box, aligned to a few microns: its corners lie on the lower face's
    // edges, inside or outside by rounding. A corner just outside is cut into two points next to
    // it; it must stay the same feature, with the same id. Likewise square and 16-gon prisms.
    let cases = [
        ("cube", cube(0.5)),
        ("square prism", prism(4, 0.5, 0.5)),
        ("16-gon prism", prism(16, 0.5, 0.5)),
    ];
    let shift = [-4e-6f32, 0.0, 4e-6];
    for (name, shape) in &cases {
        let mut sets: Vec<Vec<u32>> = Vec::new();
        for dx in shift {
            for dz in shift {
                for turn in shift {
                    let iso = Iso::new(v(dx, 0.997, dz), yaw(turn));
                    let mut out = Vec::new();
                    collide(shape, &Iso::IDENTITY, shape, &iso, 0.02, &mut out);
                    assert_eq!(out.len(), 1, "{name}");
                    assert_eq!(out[0].count, 4, "{name}: {:?}", out[0]);
                    let ids = sorted_ids(&out[0]);
                    if !sets.contains(&ids) {
                        sets.push(ids);
                    }
                }
            }
        }
        assert_eq!(sets.len(), 1, "{name}: {} different point sets", sets.len());
    }
}

#[test]
fn resting_bodies_keep_their_contact_ids_from_step_to_step() {
    let q45 = yaw(std::f32::consts::FRAC_PI_4);
    // (name, lower shape on the ground or the ground itself, upper shape, its turn).
    let cases = [
        ("cube turned 45° on a cube", Some(cube(0.5)), cube(0.5), q45),
        (
            "16-gon prism on the ground",
            None,
            prism(16, 0.5, 0.5),
            Quat::IDENTITY,
        ),
        (
            "8-gon prism on a cube",
            Some(cube(0.5)),
            prism(8, 0.5, 0.5),
            Quat::IDENTITY,
        ),
        (
            "cube on a 6-gon prism",
            Some(prism(6, 0.6, 0.5)),
            cube(0.5),
            yaw(0.1),
        ),
        (
            "cube on an aligned cube",
            Some(cube(0.5)),
            cube(0.5),
            Quat::IDENTITY,
        ),
        (
            "4-gon prism on an aligned one",
            Some(prism(4, 0.5, 0.5)),
            prism(4, 0.5, 0.5),
            Quat::IDENTITY,
        ),
        (
            "16-gon prism on an aligned one",
            Some(prism(16, 0.5, 0.5)),
            prism(16, 0.5, 0.5),
            Quat::IDENTITY,
        ),
    ];
    for (name, lower, upper, turn) in cases {
        let mut w = world(false);
        let (_, g) = ground(&mut w);
        let (a, top_y) = match lower {
            Some(s) => (
                add(
                    &mut w,
                    BodyKind::Dynamic,
                    s,
                    v(0.0, 0.5, 0.0),
                    Quat::IDENTITY,
                )
                .1,
                1.505,
            ),
            None => (g, 0.505),
        };
        let (_, b) = add(&mut w, BodyKind::Dynamic, upper, v(0.0, top_y, 0.0), turn);
        let mut prev = Vec::new();
        let mut changes = 0;
        for step in 0..360 {
            w.step(DT);
            let ids = ids_between(&w, a, b);
            if step >= 60 && ids != prev {
                changes += 1;
            }
            prev = ids;
        }
        assert_eq!(prev.len(), 4, "{name}");
        assert_eq!(
            changes, 0,
            "{name}: the point set changed in {changes} of 300 steps"
        );
    }
}

/// When a tower of `shapes` (stacked 1.005 m apart) is first all asleep, and its fastest body
/// between 3 and 6 s with sleep off.
fn settle(shapes: &[(Shape, Quat)]) -> (Option<f32>, f32) {
    let build = |sleep: bool| {
        let mut w = world(sleep);
        ground(&mut w);
        let bodies: Vec<BodyId> = shapes
            .iter()
            .enumerate()
            .map(|(i, (s, q))| {
                let p = v(0.0, 0.5 + i as f32 * 1.005, 0.0);
                add(&mut w, BodyKind::Dynamic, s.clone(), p, *q).0
            })
            .collect();
        (w, bodies)
    };
    let (mut w, bodies) = build(true);
    let mut asleep = None;
    for step in 0..240 {
        w.step(DT);
        if bodies.iter().all(|&b| !w.is_awake(b)) {
            asleep = Some((step + 1) as f32 * DT);
            break;
        }
    }
    let (mut w, bodies) = build(false);
    let mut fastest = 0.0f32;
    for step in 0..360 {
        w.step(DT);
        if step >= 180 {
            for &b in &bodies {
                fastest = fastest.max(w.linear_velocity(b).unwrap().length());
            }
        }
    }
    (asleep, fastest)
}

#[test]
fn towers_of_boxes_and_prisms_settle_and_sleep() {
    // Aligned boxes, whose corners sit on the edges below, and towers whose contact polygons
    // have more than four corners.
    let q45 = yaw(std::f32::consts::FRAC_PI_4);
    let towers = [
        (
            "aligned boxes",
            (0..10)
                .map(|_| (cube(0.5), Quat::IDENTITY))
                .collect::<Vec<_>>(),
        ),
        (
            "boxes turned 0°/45°",
            (0..10)
                .map(|i| (cube(0.5), if i % 2 == 1 { q45 } else { Quat::IDENTITY }))
                .collect(),
        ),
        (
            "boxes turned 0.37 rad each",
            (0..10).map(|i| (cube(0.5), yaw(0.37 * i as f32))).collect(),
        ),
        (
            "16-gon prisms",
            (0..10)
                .map(|_| (prism(16, 0.5, 0.5), Quat::IDENTITY))
                .collect(),
        ),
    ];
    for (name, shapes) in &towers {
        let (asleep, fastest) = settle(shapes);
        let t = asleep.unwrap_or(f32::INFINITY);
        assert!(t <= 1.5, "{name}: asleep at {asleep:?} s");
        assert!(fastest < 0.01, "{name}: still moving at {fastest} m/s");
    }
}

// ---------------------------------------------------------------------------------------------
// An irregular hull's manifold holds its deepest vertex.

/// The lowest height of a hull's vertices placed at `iso`.
fn lowest(shape: &Shape, iso: &Iso) -> f32 {
    let Shape::Hull(h) = shape else {
        unreachable!()
    };
    h.vertices
        .iter()
        .map(|&p| iso.point(p).y)
        .fold(f32::INFINITY, f32::min)
}

/// A rock whose deepest vertex's Gauss-map region is thin: in the pose of
/// [`ROCK_TURN`], the face most opposed to the floor normal is not one round that vertex.
fn rock_with_a_thin_corner() -> Shape {
    hull(&[
        v(-0.07152515, 0.36162308, -0.32141668),
        v(-0.2524875, -0.16207838, -0.39029223),
        v(-0.092764236, -0.30562332, -0.33297196),
        v(-0.005008262, 0.47196582, -0.010177106),
        v(-0.29700527, -0.023150947, 0.2018781),
        v(-0.32623526, 0.20684536, -0.17378394),
        v(-0.20847633, -0.3104047, 0.107808635),
        v(0.28949133, -0.07750952, -0.22569941),
        v(-0.4243747, 0.030323328, -0.18738551),
        v(-0.28727093, 0.10731741, 0.2850182),
        v(0.14148356, -0.2988835, -0.3731504),
        v(0.28234464, 0.23641856, 0.26343128),
        v(0.21015246, -0.38285896, 0.21268345),
        v(0.38313198, -0.20128514, 0.17821461),
        v(0.15712892, 0.1943064, 0.43206674),
        v(0.2830342, -0.29728156, -0.19793268),
        v(-0.26433915, -0.33656242, -0.2445534),
        v(0.20691639, -0.21430452, 0.3229596),
        v(0.25008506, -0.010904501, -0.27648184),
        v(0.37192968, -0.026078578, 0.20661542),
        v(0.33744475, 0.08482884, -0.070443444),
        v(0.39428917, -0.09767738, -0.07395374),
        v(-0.04395615, -0.33113262, 0.35496065),
        v(0.26263997, 0.19352141, -0.15934844),
        v(0.048919417, -0.2645091, 0.38525265),
        v(-0.29690892, 0.3482832, 0.0025186709),
        v(-0.25841466, 0.18850535, -0.34104148),
        v(-0.2531295, 0.41823676, 0.035349354),
    ])
}
const ROCK_TURN: Quat = Quat::new(-0.37286812, -0.1903194, 0.8599497, 0.2919496);

/// A big box whose top face is y = 0, and a big floor triangle in the plane y = 0.
fn floors() -> [(Shape, Iso); 2] {
    let slab = (
        Shape::Box {
            half_extents: v(10.0, 0.5, 10.0),
        },
        Iso::from_position(v(0.0, -0.5, 0.0)),
    );
    let triangle = TriangleMesh::new(
        vec![v(-20.0, 0.0, -20.0), v(0.0, 0.0, 30.0), v(20.0, 0.0, -20.0)],
        vec![[0, 1, 2]],
    )
    .unwrap();
    [slab, (Shape::Mesh(Arc::new(triangle)), Iso::IDENTITY)]
}

#[test]
fn an_irregular_hull_reports_its_deepest_vertex() {
    let thin = rock_with_a_thin_corner();
    let low = lowest(&thin, &Iso::new(Vec3::ZERO, ROCK_TURN));
    // The lowest vertex 1 cm below the floor.
    let iso = Iso::new(v(0.0, -low - 0.01, 0.0), ROCK_TURN);
    for (floor, floor_iso) in floors() {
        for swap in [false, true] {
            let mut out = Vec::new();
            if swap {
                collide(&thin, &iso, &floor, &floor_iso, 0.02, &mut out);
            } else {
                collide(&floor, &floor_iso, &thin, &iso, 0.02, &mut out);
            }
            assert_eq!(out.len(), 1, "{out:?}");
            let s = out[0].min_separation();
            assert!(
                (s + 0.01).abs() <= 1e-4,
                "swapped {swap}: {s} instead of -0.01: {out:?}"
            );
        }
    }
    // Random rocks in random poses, from just touching to 1 cm deep.
    let mut rng = Rng(7);
    let mut checked = 0;
    let mut misses = Vec::new();
    for r in 0..8 {
        let rock = random_rock(&mut rng, 40);
        for _ in 0..200 {
            let turn = rng.rotation();
            let low = lowest(&rock, &Iso::new(Vec3::ZERO, turn));
            for depth in [0.0f32, 0.005, 0.01] {
                let iso = Iso::new(
                    v(rng.range(-1.0, 1.0), -low - depth, rng.range(-1.0, 1.0)),
                    turn,
                );
                for (floor, floor_iso) in floors() {
                    let mut out = Vec::new();
                    collide(&floor, &floor_iso, &rock, &iso, 0.02, &mut out);
                    let s = out
                        .iter()
                        .map(Manifold::min_separation)
                        .fold(f32::INFINITY, f32::min);
                    checked += 1;
                    if (s + depth).abs() > 1e-3 {
                        misses.push((r, depth, s));
                    }
                }
            }
        }
    }
    assert_eq!(checked, 8 * 200 * 3 * 2);
    assert!(
        misses.is_empty(),
        "{} of {checked} miss the deepest vertex: {:?}",
        misses.len(),
        &misses[..misses.len().min(10)]
    );
}

#[test]
fn an_irregular_hull_spawned_in_the_floor_is_pushed_out() {
    let rock = rock_with_a_thin_corner();
    let low = lowest(&rock, &Iso::new(Vec3::ZERO, ROCK_TURN));
    let mut w = world(false);
    ground(&mut w);
    // Its turn locked, so the pose that hides the deepest vertex stays.
    let b = w
        .create_body(&BodyDesc {
            position: v(0.0, -low - 0.01, 0.0),
            rotation: ROCK_TURN,
            lock_rotation: [true; 3],
            ..Default::default()
        })
        .unwrap();
    w.add_collider(b, &ColliderDesc::new(rock.clone())).unwrap();
    for _ in 0..180 {
        w.step(DT);
    }
    let rest = lowest(&rock, &w.body_pose(b).unwrap());
    assert!(rest > -0.003, "the rock rests {rest} deep");
    let reported = w
        .manifolds()
        .map(|(_, _, m)| m.min_separation())
        .fold(f32::INFINITY, f32::min);
    assert!((reported - rest).abs() < 1e-3, "{reported} vs {rest}");
}

// ---------------------------------------------------------------------------------------------
// Large hulls collide in about linear time.

/// The time `runs` collisions of two touching copies of `shape` take.
fn collide_time(shape: &Shape, runs: usize) -> f64 {
    let iso = Iso::new(
        v(0.1, 0.9, 0.05),
        Quat::from_axis_angle(v(1.0, 2.0, 3.0).normalize_or_zero(), 0.7),
    );
    let mut out = Vec::new();
    let t = Instant::now();
    for _ in 0..runs {
        out.clear();
        collide(shape, &Iso::IDENTITY, shape, &iso, 0.02, &mut out);
    }
    assert_eq!(out.len(), 1, "the hulls touch");
    t.elapsed().as_secs_f64() / runs as f64
}

#[test]
fn the_cost_of_colliding_hulls_grows_about_linearly_with_their_size() {
    // 8 times the vertices: about 64 times the cost for a test of every edge pair, about 8 times
    // for one that skips the pairs that cannot matter. The fastest of several interleaved runs,
    // so a busy machine slows both alike.
    let (small, large) = (ball(32, 0.5), ball(256, 0.5));
    let (mut t_small, mut t_large) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..7 {
        t_small = t_small.min(collide_time(&small, 40));
        t_large = t_large.min(collide_time(&large, 5));
    }
    let (large, ratio) = (t_large, t_large / t_small);
    assert!(
        ratio < 25.0,
        "256 vertices cost {ratio:.0} times as much as 32 ({:.3} ms)",
        large * 1e3
    );
}
