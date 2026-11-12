//! Bodies with two locked rotation axes stay on the orientations they can reach (the joints
//! verifier's new defect: axis-locked turntables freeze or get kicked after 9 to 23 simulated
//! minutes). Such a body may only turn about its one free world axis, so every orientation it can
//! reach is its rest orientation (as created, or as last set with `set_body_pose`) turned about
//! that axis. Each step's `dq · rotation0`, normalized, was off that set by a rounding, and the
//! roundings added up: about 1e-5 rad a minute for some orientations. Joints tolerate 1e-4 rad of
//! that before the rows the locks leave as pure rounding come back as real ones, carrying the
//! anchor error built up while they were ignored: a turntable spinning at 3 rad/s was thrown to
//! -21 rad/s, then froze. Every scene here is synthetic.

use std::f32::consts::FRAC_PI_2;

use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, HingeOptions, Iso, JointDesc, JointKind,
    PhysicsWorld, Quat, Settings, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;

fn world() -> PhysicsWorld {
    PhysicsWorld::new(Settings {
        enable_sleep: false,
        ..Default::default()
    })
}

/// How far orientation `q` is from `rest` turned about `axis`: the angle of the part of
/// `q · rest⁻¹` that is not a turn about the axis.
fn tilt(q: Quat, rest: Quat, axis: Vec3) -> f32 {
    let v = (q * rest.conjugate()).vector();
    2.0 * (v - axis * v.dot(axis)).length()
}

/// The verifier's orientations, which drift linearly (the first 1.05e-5 rad a minute) or random
/// walk (the last).
fn orientations() -> [Quat; 3] {
    [
        Quat::from_axis_angle(Vec3::new(-0.7, 0.1, 0.5).normalize_or_zero(), 2.5),
        Quat::from_axis_angle(Vec3::X, FRAC_PI_2)
            * Quat::from_axis_angle(Vec3::new(0.3, 0.9, -0.2).normalize_or_zero(), 1.234),
        Quat::from_axis_angle(Vec3::Y, 0.7) * Quat::from_axis_angle(Vec3::X, FRAC_PI_2),
    ]
}

/// A translation-locked table at (0, 1, 0) authored at `rotation`, turning about world `axis`
/// only, spun at 3 rad/s.
fn table(w: &mut PhysicsWorld, rotation: Quat, axis: usize) -> BodyId {
    let mut lock_rotation = [true; 3];
    lock_rotation[axis] = false;
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, 1.0, 0.0),
            rotation,
            angular_damping: 0.0,
            lock_translation: [true; 3],
            lock_rotation,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        b,
        &ColliderDesc {
            mask: 0,
            ..ColliderDesc::new(Shape::Box {
                half_extents: Vec3::new(1.0, 1.0, 0.05),
            })
        },
    )
    .unwrap();
    let mut spin = Vec3::ZERO;
    spin[axis] = 3.0;
    w.set_angular_velocity(b, spin);
    b
}

/// A spinning table, alone, about each world axis: after a simulated minute its orientation is
/// still its rest orientation turned about the free axis, to a rounding. It was up to 1e-5 rad
/// off, and growing.
#[test]
fn a_body_free_to_turn_about_one_axis_stays_on_its_reachable_orientations() {
    for (i, rotation) in orientations().into_iter().enumerate() {
        for axis in 0..3 {
            let mut w = world();
            let b = table(&mut w, rotation, axis);
            let rest = w.body_pose(b).unwrap().rotation;
            let mut worst = 0.0f32;
            for _ in 0..3600 {
                w.step(DT);
                worst = worst.max(tilt(w.body_pose(b).unwrap().rotation, rest, unit(axis)));
            }
            let spin = w.angular_velocity(b).unwrap()[axis];
            assert!(
                worst < 1e-6 && (spin - 3.0).abs() < 1e-4,
                "orientation {i}, free axis {axis}: {worst} rad off its reachable orientations, \
                 spinning at {spin}"
            );
        }
    }
}

fn unit(axis: usize) -> Vec3 {
    let mut u = Vec3::ZERO;
    u[axis] = 1.0;
    u
}

/// The rest orientation moves with `set_body_pose`: a table teleported to a new orientation turns
/// about its free axis from there.
#[test]
fn a_teleported_body_turns_from_its_new_orientation() {
    let mut w = world();
    let [first, second, _] = orientations();
    let b = table(&mut w, first, 1);
    for _ in 0..30 {
        w.step(DT);
    }
    w.set_body_pose(b, Iso::new(Vec3::new(0.0, 1.0, 0.0), second));
    w.set_angular_velocity(b, Vec3::new(0.0, 3.0, 0.0));
    let mut worst = 0.0f32;
    for _ in 0..600 {
        w.step(DT);
        worst = worst.max(tilt(w.body_pose(b).unwrap().rotation, second, Vec3::Y));
    }
    let turned =
        w.body_pose(b).unwrap().rotation.twist_angle(Vec3::Y) - second.twist_angle(Vec3::Y);
    assert!(worst < 1e-6, "{worst} rad off the new orientation's turns");
    assert!(turned.abs() > 0.1, "it turns: {turned}");
}

/// The verifier's repro: a turntable on a hinge 0.1 m below its centre, about world Y, spun at
/// 3 rad/s, keeps turning at 3 rad/s for ten simulated minutes, for each orientation it may be
/// authored at. At 4a2333a the first was thrown from 3 to -21 rad/s at t = 566 s, then froze.
#[test]
fn an_axis_locked_turntable_keeps_turning_for_ten_minutes() {
    for (i, authored) in orientations().into_iter().enumerate() {
        let mut w = world();
        let ground = w
            .create_body(&BodyDesc {
                kind: BodyKind::Static,
                ..Default::default()
            })
            .unwrap();
        let t = table(&mut w, authored, 1);
        let hinge = Iso::new(
            Vec3::new(0.0, 0.9, 0.0),
            Quat::from_rotation_arc(Vec3::X, Vec3::Y),
        );
        let frame_a = w.body_pose(ground).unwrap().inverse_mul(&hinge);
        let frame_b = w.body_pose(t).unwrap().inverse_mul(&hinge);
        w.create_joint(&JointDesc {
            body_a: ground,
            body_b: t,
            frame_a,
            frame_b,
            kind: JointKind::Hinge(HingeOptions::default()),
            collide_connected: false,
            user_data: 0,
        })
        .unwrap();
        for step in 0..36_000 {
            w.step(DT);
            let spin = w.angular_velocity(t).unwrap().y;
            assert!(
                (spin - 3.0).abs() < 1e-3,
                "orientation {i}, t = {:.2} s: spin {spin} rad/s",
                (step + 1) as f32 * DT
            );
        }
    }
}

/// Upright bodies (rotation locked about X and Z, as a character's capsule) still turn about Y
/// when pushed off-centre, and settle as before.
#[test]
fn an_upright_body_pushed_off_centre_turns_about_its_free_axis() {
    let mut w = world();
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
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        }),
    )
    .unwrap();
    let rest = Quat::from_axis_angle(Vec3::Y, 0.3);
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, 0.5, 0.0),
            rotation: rest,
            lock_rotation: [true, false, true],
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        b,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(0.3, 0.5, 0.2),
        }),
    )
    .unwrap();
    w.apply_impulse_at(b, Vec3::new(0.0, 0.0, 120.0), Vec3::new(0.3, 0.5, 0.0));
    let mut worst = 0.0f32;
    for _ in 0..240 {
        w.step(DT);
        worst = worst.max(tilt(w.body_pose(b).unwrap().rotation, rest, Vec3::Y));
    }
    let turned = w.body_pose(b).unwrap().rotation.twist_angle(Vec3::Y) - 0.3;
    assert!(worst < 1e-6, "{worst} rad off upright");
    assert!(turned < -0.05, "the push turned it about -Y: {turned}");
}
