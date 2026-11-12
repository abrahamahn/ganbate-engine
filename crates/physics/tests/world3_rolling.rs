//! Rolling and spinning rounded shapes (review finding SOLVER-ROLLING-ANCHOR). Each substep the
//! solver re-estimates a contact's separation from how far the bodies have moved and turned. It
//! turned the whole contact anchor with the body, but on a sphere or capsule the touching point
//! does not turn with it: it stays under the core, a radius along the normal. A ball that turned
//! by `θ` within the step read `r (1 − cos θ)` farther from the ground than it was, so its point
//! counted as speculative and let it sink: rolling balls rode millimetres deep, and a fast ball
//! hitting a wall at a slant went into it (through a one-sided wall with continuous collision
//! off). Every scene here is synthetic.

use std::sync::Arc;

use ganbate_physics::mesh::{Heightfield, TriangleMesh};
use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, PhysicsWorld, Quat, Settings, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;

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

/// A 200 m box whose top is at y = 0.
fn ground(w: &mut PhysicsWorld) {
    static_body(
        w,
        Vec3::new(0.0, -0.5, 0.0),
        Shape::Box {
            half_extents: Vec3::new(100.0, 0.5, 100.0),
        },
    );
}

/// The lowest centre height of a ball of `radius` set rolling (or only spinning) on the ground,
/// over steps 60 to 240.
fn lowest_centre(linear: Vec3, angular: Vec3, friction: f32, radius: f32) -> f32 {
    let mut w = PhysicsWorld::new(Settings {
        enable_continuous: false,
        ..Default::default()
    });
    ground(&mut w);
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(-50.0, radius, 0.0),
            linear_velocity: linear,
            angular_velocity: angular,
            can_sleep: false,
            angular_damping: 0.0,
            ..Default::default()
        })
        .unwrap();
    let mut d = ColliderDesc::new(Shape::Sphere { radius });
    d.friction = friction;
    w.add_collider(b, &d).unwrap();
    let mut lowest = f32::INFINITY;
    for step in 0..240 {
        w.step(DT);
        if step >= 60 {
            lowest = lowest.min(w.body_pose(b).unwrap().position.y);
        }
    }
    lowest
}

/// The reviewer's case: a 10 cm ball rolling at 2, 6 and 12 m/s sank 2.4, 8 and 12.5 mm into the
/// ground; sliding without spin it stays at 0.0999.
#[test]
fn a_rolling_ball_rides_on_the_ground_not_in_it() {
    let r = 0.1;
    for speed in [2.0f32, 6.0, 12.0] {
        let lowest = lowest_centre(
            Vec3::new(speed, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -speed / r),
            0.6,
            r,
        );
        assert!(
            lowest > r - 3e-4,
            "rolling at {speed} m/s the ball's centre sank to {lowest} (radius {r})"
        );
    }
}

/// The control: a frictionless ball spinning in place about a horizontal axis sank 7.9 mm and
/// jittered; about the vertical axis (its anchor on the spin axis) it did not.
#[test]
fn a_ball_spinning_in_place_stays_at_its_height() {
    let r = 0.1;
    for spin in [
        Vec3::new(0.0, 0.0, 60.0),
        Vec3::new(60.0, 0.0, 0.0),
        Vec3::new(0.0, 60.0, 0.0),
    ] {
        let lowest = lowest_centre(Vec3::ZERO, spin, 0.0, r);
        assert!(
            lowest > r - 2e-4,
            "spinning at {spin:?} rad/s the ball's centre sank to {lowest}"
        );
    }
}

/// A capsule rolling on a flat heightfield (axis across the motion) sank 2.6 to 9.1 mm.
#[test]
fn a_capsule_rolling_on_a_heightfield_stays_on_it() {
    for speed in [2.0f32, 4.0, 8.0] {
        let mut w = PhysicsWorld::default();
        static_body(
            &mut w,
            Vec3::new(-50.0, 0.0, -5.0),
            Shape::Heightfield(Arc::new(
                Heightfield::new(101, 11, vec![0.0; 101 * 11], Vec3::ONE).unwrap(),
            )),
        );
        let r = 0.1;
        let b = w
            .create_body(&BodyDesc {
                position: Vec3::new(-45.0, r, 0.0),
                rotation: Quat::from_axis_angle(Vec3::X, std::f32::consts::FRAC_PI_2),
                linear_velocity: Vec3::new(speed, 0.0, 0.0),
                angular_velocity: Vec3::new(0.0, 0.0, -speed / r),
                can_sleep: false,
                ..Default::default()
            })
            .unwrap();
        w.add_collider(
            b,
            &ColliderDesc::new(Shape::Capsule {
                half_height: 0.3,
                radius: r,
            }),
        )
        .unwrap();
        let mut lowest = f32::INFINITY;
        for step in 0..240 {
            w.step(DT);
            if step >= 60 {
                lowest = lowest.min(w.body_pose(b).unwrap().position.y);
            }
        }
        assert!(
            lowest > r - 5e-4,
            "rolling at {speed} m/s the capsule's axis sank to {lowest} (radius {r})"
        );
    }
}

/// A one-sided wall in the plane x = 0 facing -X (y and z in [-10, 10]).
fn mesh_wall() -> Shape {
    let v = vec![
        Vec3::new(0.0, -10.0, -10.0),
        Vec3::new(0.0, -10.0, 10.0),
        Vec3::new(0.0, 10.0, -10.0),
        Vec3::new(0.0, 10.0, 10.0),
    ];
    Shape::Mesh(Arc::new(
        TriangleMesh::new(v, vec![[0, 1, 2], [2, 1, 3]]).unwrap(),
    ))
}

/// How far past the wall's plane (x = 0) the surface of a 5 cm ball shot from 5 mm in front of
/// it gets within six steps, and where its centre ends (no gravity, continuous collision off:
/// the solver alone).
fn oblique_hit(mesh: bool, v: Vec3) -> (f32, f32) {
    let mut w = PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        enable_continuous: false,
        ..Default::default()
    });
    if mesh {
        static_body(&mut w, Vec3::ZERO, mesh_wall());
    } else {
        static_body(
            &mut w,
            Vec3::new(0.5, 0.0, 0.0),
            Shape::Box {
                half_extents: Vec3::new(0.5, 10.0, 10.0),
            },
        );
    }
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(-0.055, 1.0, 0.0),
            linear_velocity: v,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(b, &ColliderDesc::new(Shape::Sphere { radius: 0.05 }))
        .unwrap();
    let mut deepest = f32::NEG_INFINITY;
    for _ in 0..6 {
        w.step(DT);
        deepest = deepest.max(w.body_pose(b).unwrap().position.x + 0.05);
    }
    (deepest, w.body_pose(b).unwrap().position.x)
}

/// With friction the ball starts turning as it meets the wall, so its point read as apart: at
/// (20, 0, 12) m/s it got 3.4 cm past the wall's plane, and at (200, 0, 120) m/s it went through
/// the one-sided wall and 10 cm into the box wall.
#[test]
fn a_ball_hitting_a_wall_at_a_slant_stops_at_it() {
    for mesh in [true, false] {
        for v in [Vec3::new(20.0, 0.0, 12.0), Vec3::new(200.0, 0.0, 120.0)] {
            let (deepest, end) = oblique_hit(mesh, v);
            assert!(
                deepest < 0.0075 && end < 0.0,
                "{} wall, ball at {v:?}: its surface got {deepest} m past the wall, its centre ends at x = {end}",
                if mesh { "one-sided" } else { "box" }
            );
        }
    }
}
