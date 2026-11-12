//! Thin bodies on joints (the joints fixer's note for world.rs and solver.rs). The solver pushed
//! every body through its world inverse inertia of the start of the step, while the joints turn
//! their anchors with the bodies each substep. A very thin rod (2 m by 2 mm: its inertia about
//! its own axis is 2e-6 of that across it) that had swung a little was pushed as if its axis had
//! not moved, turned about that old axis almost freely, and fell off a hinge across it (the end
//! 1.3 m from the pivot, the axis flipped). The inverse inertia now turns with the body each
//! substep for the joints; the contacts, whose anchors stay those of the start of the step, keep
//! its inertia of the start of the step too. Every scene here is synthetic.

use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, HingeOptions, Iso, JointDesc, JointKind,
    PhysicsWorld, Quat, Settings, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;

/// A static post at (0, 10, 0) and a 2 m rod `2·t` thick lying along +X from it, joined at the
/// post by `kind` about world `axis`; returns the world, the rod and the joint's frame on it.
fn rod_on_post(
    t: f32,
    kind: JointKind,
    axis: Vec3,
    lock_rotation: [bool; 3],
    push: Vec3,
) -> (PhysicsWorld, BodyId, Iso) {
    let mut w = PhysicsWorld::new(Settings {
        enable_sleep: false,
        ..Default::default()
    });
    let pivot = Vec3::new(0.0, 10.0, 0.0);
    let post = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: pivot,
            ..Default::default()
        })
        .unwrap();
    let rod = w
        .create_body(&BodyDesc {
            position: pivot + Vec3::X,
            linear_velocity: push,
            angular_damping: 0.0,
            lock_rotation,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        rod,
        &ColliderDesc {
            mask: 0,
            ..ColliderDesc::new(Shape::Box {
                half_extents: Vec3::new(1.0, t, t),
            })
        },
    )
    .unwrap();
    let frame = Iso::new(pivot, Quat::from_rotation_arc(Vec3::X, axis));
    let frame_a = w.body_pose(post).unwrap().inverse_mul(&frame);
    let frame_b = w.body_pose(rod).unwrap().inverse_mul(&frame);
    w.create_joint(&JointDesc {
        body_a: post,
        body_b: rod,
        frame_a,
        frame_b,
        kind,
        collide_connected: false,
        user_data: 0,
    })
    .unwrap();
    (w, rod, frame_b)
}

/// Rods 4, 2, 1 and 0.5 mm thick, hinged at one end about an axis across them and let fall from
/// level: each swings on its hinge as a 2 cm rod does. At 4a2333a the 2 mm and thinner ones fell
/// off (their ends 1.3 to 12 m from the pivot, the hinge axis flipped, spinning at up to 2000
/// rad/s).
#[test]
fn thin_rods_hinged_across_themselves_swing_on_their_hinges() {
    let hinge = JointKind::Hinge(HingeOptions::default());
    let swing = |t: f32| {
        let (mut w, rod, frame_b) = rod_on_post(t, hinge, Vec3::Z, [false; 3], Vec3::ZERO);
        let (mut gap, mut tilt) = (0.0f32, 0.0f32);
        for _ in 0..120 {
            w.step(DT);
            let pose = w.body_pose(rod).unwrap();
            gap = gap.max((pose.point(-Vec3::X) - Vec3::new(0.0, 10.0, 0.0)).length());
            let axis = pose.rotation.rotate(frame_b.rotation.rotate(Vec3::X));
            tilt = tilt.max(axis.cross(Vec3::Z).length());
        }
        let far = w.body_pose(rod).unwrap().point(Vec3::X);
        (gap, tilt, far)
    };
    let (_, _, reference) = swing(0.01);
    for t in [0.002f32, 0.001, 0.0005, 0.00025] {
        let (gap, tilt, far) = swing(t);
        assert!(
            gap < 2e-3 && tilt < 1e-4 && (far - reference).length() < 1e-3,
            "a 2 m x {} mm rod: its end {gap} m from the pivot, its axis {tilt} rad off the \
             hinge's, its far end at {far:?} (a 2 cm rod's at {reference:?})",
            2000.0 * t
        );
    }
}

/// The same rods on a ball joint, let fall from level with a push sideways: a rod's swing does
/// not depend on its thickness (its inertia across it barely does), so each follows the 2 cm
/// rod. At 4a2333a the thin rods' far ends were 11 to 14 cm from the thick one's after two
/// seconds, the thick rod itself pushed off its own path.
#[test]
fn thin_rods_on_a_ball_joint_swing_as_a_thick_one_does() {
    let far_end = |t: f32| {
        let (mut w, rod, _) = rod_on_post(t, JointKind::Ball, Vec3::Z, [false; 3], Vec3::Z);
        let mut gap = 0.0f32;
        for _ in 0..120 {
            w.step(DT);
            let pose = w.body_pose(rod).unwrap();
            gap = gap.max((pose.point(-Vec3::X) - Vec3::new(0.0, 10.0, 0.0)).length());
        }
        (gap, w.body_pose(rod).unwrap().point(Vec3::X))
    };
    let (_, reference) = far_end(0.01);
    for t in [0.002f32, 0.001] {
        let (gap, far) = far_end(t);
        assert!(
            gap < 2e-3 && (far - reference).length() < 0.01,
            "a 2 m x {} mm rod: its end {gap} m from the pivot, its far end at {far:?} (a 2 cm \
             rod's at {reference:?})",
            2000.0 * t
        );
    }
}

/// A rod whose rotation is locked about some world axes keeps its angular velocity about them
/// exactly zero while its inverse inertia turns with it (one axis locked: the free block is
/// inverted afresh each substep), and stays on a joint its locks leave it free to follow.
#[test]
fn a_thin_rod_with_locked_axes_never_turns_about_them() {
    let hinge = JointKind::Hinge(HingeOptions::default());
    for (kind, lock, push) in [
        (hinge, [false, true, false], Vec3::ZERO),
        (hinge, [true, true, false], Vec3::ZERO),
        (JointKind::Ball, [false, true, false], Vec3::Z),
        (JointKind::Ball, [true, false, false], Vec3::ZERO),
    ] {
        for t in [0.01f32, 0.001] {
            let (mut w, rod, _) = rod_on_post(t, kind, Vec3::Z, lock, push);
            for step in 0..120 {
                w.step(DT);
                let spin = w.angular_velocity(rod).unwrap();
                let gap = (w.body_pose(rod).unwrap().point(-Vec3::X) - Vec3::new(0.0, 10.0, 0.0))
                    .length();
                for k in 0..3 {
                    assert!(
                        !lock[k] || spin[k] == 0.0,
                        "{kind:?} locked {lock:?}, {t}: step {step} spin {spin:?}"
                    );
                }
                assert!(
                    gap < 2e-3,
                    "{kind:?} locked {lock:?}, {t}: step {step} gap {gap}"
                );
            }
        }
    }
}

/// The contacts keep the inertia of the start of the step with their anchors of the start of the
/// step: thin bodies tumbling on the ground stay near it (an inertia turned under the contacts'
/// unturned anchors threw a 2 m x 2 mm rod hundreds of metres into the air at 10⁵ rad/s).
#[test]
fn thin_bodies_tumbling_on_the_ground_are_not_thrown_off_it() {
    let shapes = [
        Shape::Box {
            half_extents: Vec3::new(1.0, 0.001, 0.001),
        },
        Shape::Capsule {
            half_height: 0.5,
            radius: 0.001,
        },
    ];
    for shape in shapes {
        let mut w = PhysicsWorld::default();
        let ground = w
            .create_body(&BodyDesc {
                kind: BodyKind::Static,
                position: Vec3::new(0.0, -0.5, 0.0),
                ..Default::default()
            })
            .unwrap();
        w.add_collider(
            ground,
            &ColliderDesc::new(Shape::Box {
                half_extents: Vec3::new(50.0, 0.5, 50.0),
            }),
        )
        .unwrap();
        let b = w
            .create_body(&BodyDesc {
                position: Vec3::new(0.0, 1.0, 0.0),
                rotation: Quat::from_axis_angle(Vec3::new(0.3, 0.2, 1.0).normalize_or_zero(), 0.7),
                linear_velocity: Vec3::new(1.0, 0.0, 0.5),
                angular_velocity: Vec3::new(2.0, 5.0, 3.0),
                ..Default::default()
            })
            .unwrap();
        w.add_collider(b, &ColliderDesc::new(shape.clone()))
            .unwrap();
        let (mut highest, mut fastest) = (f32::NEG_INFINITY, 0.0f32);
        for step in 0..600 {
            w.step(DT);
            fastest = fastest.max(w.linear_velocity(b).unwrap().length());
            if step > 120 {
                highest = highest.max(w.body_pose(b).unwrap().position.y);
            }
        }
        assert!(
            highest < 0.5 && fastest < 10.0,
            "{shape:?}: as high as {highest} m after two seconds, as fast as {fastest} m/s"
        );
    }
}
