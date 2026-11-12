//! Joints on bodies whose inertia is near the largest `f32` (the world2 fix's note: joint
//! impulses were not saturated).
//!
//! A body's mass is kept within `MAX_BODY_MASS`, but colliders thousands of kilometres apart
//! still give it an inertia near `f32::MAX`, and a joint row's impulse is its effective mass (the
//! inertia) times a velocity error: 1e38 kg m² turning at 5 rad/s asks for 5e38 N m s, past
//! `f32::MAX`. The row's impulse became infinite, the velocities it gave were not numbers, and
//! the joint dropped out for the step: every step, so a fixed joint let such a body spin on
//! freely. Accumulated joint impulses now saturate (within half the largest `f32`, NaN giving
//! zero, as contacts' normal impulses do), so each pass applies the most it can and the joint
//! holds. Every scene here is synthetic.

use ganbate_physics::world::MAX_BODY_MASS;
use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, HingeOptions, Iso, JointDesc, JointKind, Motor,
    PhysicsWorld, Quat, Settings, Shape, SliderOptions, Vec3,
};

const DT: f32 = 1.0 / 60.0;

fn all_finite(w: &PhysicsWorld) -> bool {
    w.bodies().all(|b| {
        let p = w.body_pose(b).unwrap();
        p.position.is_finite()
            && p.rotation.is_finite()
            && w.linear_velocity(b).unwrap().is_finite()
            && w.angular_velocity(b).unwrap().is_finite()
    })
}

/// A static post at the origin and a body there with six 1 m balls 4000 km out along the axes,
/// dense enough that its mass is held at `MAX_BODY_MASS`: an inertia of about 1e38 kg m².
fn post_and_giant(desc: BodyDesc) -> (PhysicsWorld, BodyId, BodyId) {
    let mut w = PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let post = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(post, &ColliderDesc::new(Shape::Sphere { radius: 0.5 }))
        .unwrap();
    let giant = w.create_body(&desc).unwrap();
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        for side in [-4e6f32, 4e6] {
            let mut d = ColliderDesc::new(Shape::Sphere { radius: 1.0 });
            d.local = Iso::from_position(axis * side);
            d.density = 1e25;
            w.add_collider(giant, &d).unwrap();
        }
    }
    assert_eq!(w.mass(giant), Some(MAX_BODY_MASS));
    (w, post, giant)
}

fn joint(w: &mut PhysicsWorld, a: BodyId, b: BodyId, kind: JointKind, at: Vec3) {
    w.create_joint(&JointDesc {
        body_a: a,
        body_b: b,
        frame_a: Iso::from_position(at),
        frame_b: Iso::from_position(at),
        kind,
        collide_connected: false,
        user_data: 0,
    })
    .unwrap();
}

/// Spins the giant at `spin` and steps: the joint must stop what it locks within a second, and
/// keep the giant's pose where it was.
fn holds(kind: JointKind, at: Vec3, spin: Vec3, locked: Vec3, desc: BodyDesc) {
    let (mut w, post, giant) = post_and_giant(desc);
    joint(&mut w, post, giant, kind, at);
    w.set_angular_velocity(giant, spin);
    for step in 0..60 {
        w.step(DT);
        assert!(all_finite(&w), "{kind:?}: step {step}");
    }
    let left = w.angular_velocity(giant).unwrap().dot(locked);
    assert!(left.abs() < 1e-3, "{kind:?}: still turning at {left} rad/s");
    let pose = w.body_pose(giant).unwrap();
    let turned = pose.rotation.rotate(Vec3::X).cross(Vec3::X).length()
        + pose.rotation.rotate(Vec3::Y).cross(Vec3::Y).length();
    assert!(
        pose.position.length() < 1e-2 && turned < 0.2,
        "{kind:?}: drifted to {pose:?}"
    );
}

#[test]
fn a_fixed_joint_stops_a_giant_spinning_body() {
    holds(
        JointKind::Fixed,
        Vec3::ZERO,
        Vec3::new(0.0, 5.0, 0.0),
        Vec3::Y,
        BodyDesc::default(),
    );
}

#[test]
fn a_hinge_stops_a_giant_turning_off_its_axis() {
    holds(
        JointKind::Hinge(HingeOptions::default()),
        Vec3::ZERO,
        Vec3::new(0.0, 4.0, 3.0),
        Vec3::new(0.0, 0.8, 0.6),
        BodyDesc::default(),
    );
}

#[test]
fn a_ball_joint_off_the_centre_holds_a_giant_that_only_turns() {
    // Translation locked, so the point rows' effective mass is the inertia over the lever arm
    // squared alone: the anchor 1 m out moves at 5 m/s.
    holds(
        JointKind::Ball,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 5.0, 0.0),
        Vec3::Y,
        BodyDesc {
            lock_translation: [true; 3],
            ..Default::default()
        },
    );
}

#[test]
fn a_slider_holds_a_giant_turning_and_sliding() {
    // Every rotation locked, and the origin kept on the slider's axis (X).
    holds(
        JointKind::Slider(SliderOptions::default()),
        Vec3::ZERO,
        Vec3::new(3.0, 4.0, 0.0),
        Vec3::new(0.6, 0.8, 0.0),
        BodyDesc::default(),
    );
}

#[test]
fn a_hinge_motor_turns_a_giant_and_its_limit_stops_it() {
    // A hinge about world Y (the frame's X turned onto Y) with the strongest motor there is:
    // f32::MAX N m on about 1e38 kg m² turns the giant up at about 3 rad/s², to the motor's
    // speed, and then the upper limit holds it.
    let (mut w, post, giant) = post_and_giant(BodyDesc::default());
    let frame = Iso::new(Vec3::ZERO, Quat::from_rotation_arc(Vec3::X, Vec3::Y));
    w.create_joint(&JointDesc {
        body_a: post,
        body_b: giant,
        frame_a: frame,
        frame_b: frame,
        kind: JointKind::Hinge(HingeOptions {
            limit: Some((-0.5, 0.5)),
            motor: Some(Motor {
                speed: 1.0,
                max_force: f32::MAX,
            }),
            spring: None,
        }),
        collide_connected: false,
        user_data: 0,
    })
    .unwrap();
    w.set_angular_velocity(giant, Vec3::new(4.0, 0.0, -3.0));
    let mut fastest = 0.0f32;
    for step in 0..120 {
        w.step(DT);
        assert!(all_finite(&w), "step {step}");
        let spin = w.angular_velocity(giant).unwrap();
        fastest = fastest.max(spin.y);
        if step >= 3 {
            // Only the hinge's own axis turns.
            assert!(spin.x.abs() + spin.z.abs() < 1e-3, "step {step}: {spin:?}");
        }
        if step == 25 {
            assert!((spin.y - 1.0).abs() < 1e-2, "{spin:?}");
        }
    }
    let angle = w.body_pose(giant).unwrap().rotation.twist_angle(Vec3::Y);
    let spin = w.angular_velocity(giant).unwrap();
    assert!(fastest < 1.01, "the motor overshot to {fastest} rad/s");
    assert!(
        (angle - 0.5).abs() < 0.02 && spin.length() < 1e-2,
        "angle {angle}, spin {spin:?}"
    );
}
