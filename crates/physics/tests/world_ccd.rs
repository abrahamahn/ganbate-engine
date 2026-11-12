//! Continuous collision: fast bodies stop at the first time of impact with static and kinematic
//! geometry, but never at the seams of a flat floor, never for good, and never at geometry they
//! may pass through. Every scene here is synthetic.

use std::sync::Arc;

use ganbate_physics::mesh::{Heightfield, TriangleMesh};
use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, DistanceOptions, Iso, JointDesc, JointKind,
    PhysicsWorld, Settings, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;

fn run(w: &mut PhysicsWorld, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        w.step(DT);
    }
}

fn x(w: &PhysicsWorld, b: BodyId) -> f32 {
    w.body_pose(b).unwrap().position.x
}

fn static_body(w: &mut PhysicsWorld, position: Vec3, shape: Shape, friction: f32) -> BodyId {
    let b = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position,
            ..Default::default()
        })
        .unwrap();
    let mut d = ColliderDesc::new(shape);
    d.friction = friction;
    w.add_collider(b, &d).unwrap();
    b
}

/// A floor of 1 m quads along X at y = 0 (x in [-10, 10], z in [-1, 1]), front faces up.
fn mesh_floor() -> Shape {
    let mut v = Vec::new();
    for i in 0..=20 {
        let x = i as f32 - 10.0;
        v.push(Vec3::new(x, 0.0, -1.0));
        v.push(Vec3::new(x, 0.0, 1.0));
    }
    let mut t = Vec::new();
    for i in 0..20u32 {
        let (a, b, c, d) = (2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3);
        t.push([a, b, c]);
        t.push([c, b, d]);
    }
    Shape::Mesh(Arc::new(TriangleMesh::new(v, t).unwrap()))
}

/// The same floor as a flat heightfield (placed at (-10, 0, -1)).
fn field_floor() -> Shape {
    Shape::Heightfield(Arc::new(
        Heightfield::new(21, 3, vec![0.0; 63], Vec3::ONE).unwrap(),
    ))
}

fn floors() -> [(&'static str, Vec3, Shape); 2] {
    [
        ("mesh", Vec3::ZERO, mesh_floor()),
        ("heightfield", Vec3::new(-10.0, 0.0, -1.0), field_floor()),
    ]
}

fn slider(w: &mut PhysicsWorld, x: f32, speed: f32, bullet: bool) -> BodyId {
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(x, 0.1, 0.3),
            linear_velocity: Vec3::new(speed, 0.0, 0.0),
            bullet,
            can_sleep: false,
            ..Default::default()
        })
        .unwrap();
    let mut d = ColliderDesc::new(Shape::Sphere { radius: 0.1 });
    d.friction = 0.0;
    w.add_collider(b, &d).unwrap();
    b
}

/// A ball gliding on a flat triangle floor touches the next triangle across every seam. The cast
/// against it started inside its tolerance and reported a time of impact of zero, which stopped
/// the ball for good at the first seam (review finding CCD-FREEZE). Without gravity the ball
/// needs no contacts to glide.
#[test]
fn a_fast_ball_glides_across_the_seams_of_a_flat_triangle_floor() {
    for (name, at, floor) in floors() {
        for (speed, bullet) in [(6.0f32, false), (1.0, true)] {
            let mut w = PhysicsWorld::new(Settings {
                gravity: Vec3::ZERO,
                ..Default::default()
            });
            static_body(&mut w, at, floor.clone(), 0.0);
            let b = slider(&mut w, -5.55, speed, bullet);
            run(&mut w, 1.5);
            let expected = -5.55 + 1.5 * speed;
            assert!(
                (x(&w, b) - expected).abs() < 0.01,
                "{name}, {speed} m/s, bullet {bullet}: the ball stopped at x = {} (expected {expected})",
                x(&w, b)
            );
            assert!((w.linear_velocity(b).unwrap().x - speed).abs() < 1e-4);
        }
    }
}

/// The same under gravity, resting on the floor: as far as on a flat box (the reviewer's
/// scenario; the ball needs the floor's contacts to stay up).
#[test]
fn a_fast_ball_slides_as_far_on_a_triangle_floor_as_on_a_box() {
    let mut w = PhysicsWorld::default();
    static_body(
        &mut w,
        Vec3::new(0.0, -0.5, 0.0),
        Shape::Box {
            half_extents: Vec3::new(10.0, 0.5, 1.0),
        },
        0.0,
    );
    let b = slider(&mut w, -5.55, 6.0, false);
    run(&mut w, 1.5);
    let on_box = x(&w, b);
    assert!((on_box - 3.45).abs() < 0.05, "on a box: {on_box}");
    for (name, at, floor) in floors() {
        for (speed, bullet, expected) in [(6.0f32, false, on_box), (1.0, true, -4.05)] {
            let mut w = PhysicsWorld::default();
            static_body(&mut w, at, floor.clone(), 0.0);
            let b = slider(&mut w, -5.55, speed, bullet);
            run(&mut w, 1.5);
            assert!(
                (x(&w, b) - expected).abs() < 0.05,
                "{name}, {speed} m/s, bullet {bullet}: x = {} (expected {expected})",
                x(&w, b)
            );
        }
    }
}

/// Skipping the floor's seams must not let a fast ball through the floor itself: the sweep that
/// crosses the triangles' plane still stops at it.
#[test]
fn a_fast_ball_falling_onto_a_triangle_floor_stops_on_it() {
    for (name, at, floor) in floors() {
        let mut w = PhysicsWorld::default();
        static_body(&mut w, at, floor, 0.6);
        let b = w
            .create_body(&BodyDesc {
                position: Vec3::new(0.3, 1.0, 0.2),
                linear_velocity: Vec3::new(0.0, -120.0, 0.0),
                ..Default::default()
            })
            .unwrap();
        w.add_collider(b, &ColliderDesc::new(Shape::Sphere { radius: 0.1 }))
            .unwrap();
        // One step would carry it 2 m down, through the floor at y = 0.
        w.step(DT);
        let y = w.body_pose(b).unwrap().position.y;
        assert!(
            y > 0.1 && y < 0.11,
            "{name}: the ball is at y = {y} after the step"
        );
    }
}

/// A ball already touching a thin wall and shot at it has a time of impact of zero, which is not
/// taken (it would stop the ball for good, also sliding along the wall); the ball's core still
/// stops before it can pass through.
#[test]
fn a_ball_touching_a_thin_wall_does_not_pass_through_when_shot_at_it() {
    let mut w = PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    static_body(
        &mut w,
        Vec3::new(10.0, 0.0, 0.0),
        Shape::Box {
            half_extents: Vec3::new(0.02, 5.0, 5.0),
        },
        0.6,
    );
    // 2 mm from the wall's face at x = 9.98, then 300 m/s: 5 m in one step.
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(9.928, 0.0, 0.0),
            linear_velocity: Vec3::new(300.0, 0.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(b, &ColliderDesc::new(Shape::Sphere { radius: 0.05 }))
        .unwrap();
    w.step(DT);
    assert!(
        x(&w, b) < 9.98,
        "the ball's centre went past the wall's face: x = {}",
        x(&w, b)
    );
}

/// Continuous collision must follow the same filter as contacts: a joint whose bodies do not
/// collide lets one pass through the other (review finding CCD-COLLIDE-CONNECTED).
#[test]
fn a_fast_body_passes_geometry_its_joint_says_it_does_not_collide_with() {
    let mut w = PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let wall = static_body(
        &mut w,
        Vec3::new(5.0, 0.0, 0.0),
        Shape::Box {
            half_extents: Vec3::new(0.05, 2.0, 2.0),
        },
        0.6,
    );
    let ball = w
        .create_body(&BodyDesc {
            linear_velocity: Vec3::new(10.0, 0.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(ball, &ColliderDesc::new(Shape::Sphere { radius: 0.1 }))
        .unwrap();
    // A slack rope from the wall to the ball whose two ends do not collide.
    w.create_joint(&JointDesc {
        body_a: wall,
        body_b: ball,
        frame_a: Iso::IDENTITY,
        frame_b: Iso::IDENTITY,
        kind: JointKind::Distance(DistanceOptions {
            length: 10.0,
            min: 0.0,
            max: 10.0,
            spring: None,
        }),
        collide_connected: false,
        user_data: 0,
    })
    .unwrap();
    run(&mut w, 1.0);
    assert!(
        x(&w, ball) > 9.9,
        "stopped in front of a wall it may pass through: x = {}",
        x(&w, ball)
    );
    // The same sweep without the joint stops at the wall: 1 m in one step from x = 4.7 would
    // end past it.
    let other = w
        .create_body(&BodyDesc {
            position: Vec3::new(4.7, 1.0, 0.0),
            linear_velocity: Vec3::new(60.0, 0.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(other, &ColliderDesc::new(Shape::Sphere { radius: 0.1 }))
        .unwrap();
    w.step(DT);
    assert!(
        x(&w, other) > 4.8 && x(&w, other) < 4.85,
        "the ball without the joint stops at the wall: x = {}",
        x(&w, other)
    );
}
