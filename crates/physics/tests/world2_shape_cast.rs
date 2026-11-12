//! `PhysicsWorld::shape_cast` from a resting start: a surface the shape already touches (within
//! the cast's tolerance) stops the cast only if the motion heads into it. Before, any such
//! surface was reported at fraction 0, whatever the direction, so a shape resting on the ground
//! could not be cast sideways or up, and the first real obstacle was hidden behind the ground
//! (review note from the character fix: shape_cast starting in contact). Every scene here is
//! synthetic.

use std::sync::Arc;

use ganbate_physics::mesh::{Heightfield, TriangleMesh};
use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, Iso, PhysicsWorld, QueryFilter, Shape, ShapeCastHit,
    Vec3,
};

fn static_body(w: &mut PhysicsWorld, position: Vec3, shape: Shape) -> BodyId {
    let b = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(b, &ColliderDesc::new(shape)).unwrap();
    b
}

/// The ground's top at y = 0 and a wall whose face is at x = 3.
fn ground_and_wall() -> (PhysicsWorld, BodyId, BodyId) {
    let mut w = PhysicsWorld::default();
    let ground = static_body(
        &mut w,
        Vec3::new(0.0, -0.5, 0.0),
        Shape::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        },
    );
    let wall = static_body(
        &mut w,
        Vec3::new(3.5, 2.0, 0.0),
        Shape::Box {
            half_extents: Vec3::new(0.5, 2.0, 5.0),
        },
    );
    (w, ground, wall)
}

fn cast(w: &PhysicsWorld, shape: &Shape, at: Vec3, d: Vec3) -> Option<ShapeCastHit> {
    w.shape_cast(shape, &Iso::from_position(at), d, &QueryFilter::default())
}

#[test]
fn a_shape_resting_on_the_ground_casts_along_it_and_away_from_it() {
    let (w, ground, wall) = ground_and_wall();
    for (name, shape, y) in [
        (
            "box",
            Shape::Box {
                half_extents: Vec3::splat(0.5),
            },
            0.5,
        ),
        ("ball", Shape::Sphere { radius: 0.5 }, 0.5),
        // 0.05 mm above the ground: inside the cast's tolerance.
        ("hovering ball", Shape::Sphere { radius: 0.5 }, 0.50005),
        (
            "capsule",
            Shape::Capsule {
                half_height: 0.5,
                radius: 0.25,
            },
            0.75,
        ),
    ] {
        let at = Vec3::new(0.0, y, 0.0);
        // Along the ground, into the wall 2.5 m away: the wall, not the ground.
        let hit = cast(&w, &shape, at, Vec3::new(5.0, 0.0, 0.0))
            .unwrap_or_else(|| panic!("{name}: the wall was missed"));
        let expected = if name == "box" || name == "ball" || name == "hovering ball" {
            2.5 / 5.0
        } else {
            2.75 / 5.0
        };
        assert_eq!(hit.body, wall, "{name}: {hit:?}");
        assert!((hit.fraction - expected).abs() < 1e-3, "{name}: {hit:?}");
        // Along the ground away from anything, and up: nothing.
        let free = cast(&w, &shape, at, Vec3::new(-5.0, 0.0, 3.0));
        assert!(free.is_none(), "{name}: {free:?}");
        let up = cast(&w, &shape, at, Vec3::new(0.0, 3.0, 0.0));
        assert!(up.is_none(), "{name}: {up:?}");
        // Into the ground: at once.
        let down = cast(&w, &shape, at, Vec3::new(0.3, -1.0, 0.0)).unwrap();
        assert!(
            down.body == ground && down.fraction == 0.0,
            "{name}: {down:?}"
        );
    }
}

/// Against the wall: sliding along it or leaving it is free, heading into it stops at once.
#[test]
fn a_shape_touching_a_wall_slides_along_it_and_leaves_it() {
    let (w, _, wall) = ground_and_wall();
    let ball = Shape::Sphere { radius: 0.25 };
    let at = Vec3::new(2.75, 1.0, 0.0);
    assert!(cast(&w, &ball, at, Vec3::new(0.0, 0.0, 2.0)).is_none());
    assert!(cast(&w, &ball, at, Vec3::new(-2.0, 0.5, 0.0)).is_none());
    let into = cast(&w, &ball, at, Vec3::new(1.0, 0.0, 1.0)).unwrap();
    assert!(into.body == wall && into.fraction == 0.0, "{into:?}");
}

/// A shape that starts inside a collider (cores overlapping) reports it whatever the motion.
#[test]
fn a_shape_starting_inside_a_collider_always_reports_it() {
    let (w, ground, _) = ground_and_wall();
    let hit = cast(
        &w,
        &Shape::Sphere { radius: 0.2 },
        Vec3::new(0.0, -0.3, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
    )
    .unwrap();
    assert!(hit.body == ground && hit.fraction == 0.0 && hit.started_overlapping);
    assert!(hit.normal.dot(Vec3::Y) < 0.0, "against the motion: {hit:?}");
    // A cast that does not move reports what it touches.
    let still = cast(
        &w,
        &Shape::Sphere { radius: 0.5 },
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::ZERO,
    );
    assert!(still.is_some_and(|h| h.body == ground), "{still:?}");
}

/// A floor of `cell` m triangles, front faces up, at y = 0 over x, z in [-5, 5].
fn tessellated(cell: f32) -> Shape {
    let n = (10.0 / cell).round() as u32;
    let mut v = Vec::new();
    for i in 0..=n {
        for k in 0..=n {
            v.push(Vec3::new(i as f32 * cell - 5.0, 0.0, k as f32 * cell - 5.0));
        }
    }
    let mut t = Vec::new();
    let at = |i: u32, k: u32| i * (n + 1) + k;
    for i in 0..n {
        for k in 0..n {
            let (a, b, c, d) = (at(i, k), at(i + 1, k), at(i, k + 1), at(i + 1, k + 1));
            t.push([a, c, b]);
            t.push([b, c, d]);
        }
    }
    Shape::Mesh(Arc::new(TriangleMesh::new(v, t).unwrap()))
}

/// On a tessellated floor, a resting ball is within the cast's tolerance of the neighbouring
/// triangles' edges and vertices, whose normals tilt against a motion across them; the seams
/// are inactive (coplanar neighbours), so they answer with the face normal, as in the narrow
/// phase, and the ball glides across.
#[test]
fn a_ball_resting_on_a_tessellated_floor_casts_across_its_seams() {
    let mut mesh = PhysicsWorld::default();
    static_body(&mut mesh, Vec3::ZERO, tessellated(0.1));
    let mut field = PhysicsWorld::default();
    static_body(
        &mut field,
        Vec3::new(-5.0, 0.0, -5.0),
        Shape::Heightfield(Arc::new(
            Heightfield::new(101, 101, vec![0.0; 101 * 101], Vec3::new(0.1, 1.0, 0.1)).unwrap(),
        )),
    );
    let ball = Shape::Sphere { radius: 0.5 };
    for (name, w) in [("mesh", &mesh), ("heightfield", &field)] {
        for (x, z) in [(0.003, 0.0), (0.0496, 0.0021), (0.01, 0.01), (0.0, 0.0)] {
            for d in [
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(-1.0, 0.0, 1.7),
                Vec3::new(0.3, 0.0, -2.0),
            ] {
                let hit = cast(w, &ball, Vec3::new(x, 0.5, z), d);
                assert!(hit.is_none(), "{name} from ({x}, {z}) along {d:?}: {hit:?}");
            }
            let down = cast(w, &ball, Vec3::new(x, 0.5, z), Vec3::new(1.0, -0.5, 0.0));
            assert!(down.is_some_and(|h| h.fraction == 0.0), "{name}: {down:?}");
        }
    }
}

/// A box set exactly on a tessellated floor (its core touching the triangles closer than GJK
/// can resolve a direction) slides along it; a ramp rising ahead (a concave crease, inactive)
/// still stops a ball, at its face.
#[test]
fn casts_over_a_mesh_floor_find_the_ramp_ahead() {
    let mut w = PhysicsWorld::default();
    static_body(&mut w, Vec3::ZERO, tessellated(0.25));
    let crate_ = Shape::Box {
        half_extents: Vec3::splat(0.25),
    };
    let at = Vec3::new(0.125, 0.25, 0.0);
    for d in [Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -2.0)] {
        let hit = cast(&w, &crate_, at, d);
        assert!(hit.is_none(), "the crate along {d:?}: {hit:?}");
    }
    let down = cast(&w, &crate_, at, Vec3::new(0.0, -1.0, 0.0)).unwrap();
    assert!(
        down.fraction == 0.0 && (down.normal - Vec3::Y).length() < 1e-5,
        "{down:?}"
    );

    // A floor up to x = 1, then a 30° ramp: one mesh, the crease between them inactive.
    let h = 3.0 * (30f32.to_radians().tan());
    let v = vec![
        Vec3::new(-5.0, 0.0, -2.0),
        Vec3::new(1.0, 0.0, -2.0),
        Vec3::new(-5.0, 0.0, 2.0),
        Vec3::new(1.0, 0.0, 2.0),
        Vec3::new(4.0, h, -2.0),
        Vec3::new(4.0, h, 2.0),
    ];
    let t = vec![[0, 2, 1], [1, 2, 3], [1, 3, 4], [4, 3, 5]];
    let mut w = PhysicsWorld::default();
    let ramp = static_body(
        &mut w,
        Vec3::ZERO,
        Shape::Mesh(Arc::new(TriangleMesh::new(v, t).unwrap())),
    );
    let ball = Shape::Sphere { radius: 0.5 };
    let hit = cast(
        &w,
        &ball,
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::new(3.0, 0.0, 0.0),
    )
    .unwrap();
    // The ball meets the ramp's face (above the crease) when its centre, 0.5 up, is 0.5 from
    // the ramp's plane: x = 1 − (0.5 − 0.5·cos 30°) / sin 30°.
    let x = 1.0 - (0.5 - 0.5 * 30f32.to_radians().cos()) / 30f32.to_radians().sin();
    assert_eq!(hit.body, ramp);
    assert!(
        (hit.fraction - x / 3.0).abs() < 1e-3,
        "{hit:?}, expected x = {x}"
    );
    let n = Vec3::new(-0.5, 30f32.to_radians().cos(), 0.0);
    assert!((hit.normal - n).length() < 1e-3, "{hit:?}");
}

/// A box a hair above the ground, on it, or sunk into it by less than `linear_slop` (5 mm)
/// slides and lifts freely, turned or not; GJK reads cores this close as overlapping, so the
/// narrow phase gives the normal. Sunk deeper, the ground is reported at once.
#[test]
fn a_box_within_a_hair_of_the_ground_casts_along_it() {
    let (w, ground, _) = ground_and_wall();
    let crate_ = Shape::Box {
        half_extents: Vec3::splat(0.5),
    };
    for dy in [1e-4f32, 1e-5, 1e-6, 0.0, -1e-5, -1e-3, -4e-3] {
        for turn in [0.0f32, 0.3] {
            let at = Iso::new(
                Vec3::new(0.0, 0.5 + dy, 0.0),
                ganbate_physics::Quat::from_axis_angle(Vec3::Y, turn),
            );
            for d in [Vec3::new(-3.0, 0.0, 0.0), Vec3::new(0.0, 3.0, 0.0)] {
                let hit = w.shape_cast(&crate_, &at, d, &QueryFilter::default());
                assert!(
                    hit.is_none(),
                    "{dy} up, turned {turn}, along {d:?}: {hit:?}"
                );
            }
        }
    }
    let deep = Iso::from_position(Vec3::new(0.0, 0.5 - 6e-3, 0.0));
    let hit = w
        .shape_cast(
            &crate_,
            &deep,
            Vec3::new(-3.0, 0.0, 0.0),
            &QueryFilter::default(),
        )
        .unwrap();
    assert!(hit.body == ground && hit.fraction == 0.0 && hit.started_overlapping);
}
