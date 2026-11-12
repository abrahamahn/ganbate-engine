//! Single bodies: their inertia, axis locks, kinematic targets and the inputs a world accepts.
//! Every scene here is synthetic and needs no contacts.

use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, PhysicsWorld, Quat, Settings, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;

fn run(w: &mut PhysicsWorld, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        w.step(DT);
    }
}

fn no_gravity() -> PhysicsWorld {
    PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    })
}

fn body_with(w: &mut PhysicsWorld, desc: BodyDesc, shape: Shape) -> BodyId {
    let b = w.create_body(&desc).unwrap();
    w.add_collider(b, &ColliderDesc::new(shape)).unwrap();
    b
}

/// The inertia of a millimetre ball is about 1e-11 kg·m² and its determinant underflowed the
/// fixed threshold of the inverse, so it could never turn (review finding
/// MAT3-INVERSE-THRESHOLD).
#[test]
fn tiny_balls_turn_like_big_ones() {
    for radius in [1.0f32, 0.05, 0.002, 1e-4] {
        let mut w = no_gravity();
        let b = body_with(&mut w, BodyDesc::default(), Shape::Sphere { radius });
        let mass = w.mass(b).unwrap();
        let inertia = 0.4 * mass * radius * radius;
        let impulse = Vec3::new(0.0, inertia, 0.0);
        w.apply_angular_impulse(b, impulse);
        let spin = w.angular_velocity(b).unwrap();
        assert!(
            (spin - Vec3::Y).length() < 1e-3,
            "a ball of radius {radius} given one radian per second turns at {spin:?}"
        );
    }
}

/// A locked axis carries no velocity, whether it was given at creation or set later (review
/// finding LOCK-VELOCITY).
#[test]
fn locked_axes_never_move_whatever_velocity_they_are_given() {
    let mut w = PhysicsWorld::new(Settings {
        enable_sleep: false,
        ..Default::default()
    });
    let up = body_with(
        &mut w,
        BodyDesc {
            position: Vec3::new(0.0, 10.0, 0.0),
            linear_velocity: Vec3::new(1.0, 5.0, 0.0),
            lock_translation: [false, true, false],
            ..Default::default()
        },
        Shape::Sphere { radius: 0.5 },
    );
    assert_eq!(w.linear_velocity(up), Some(Vec3::new(1.0, 0.0, 0.0)));
    let spin = body_with(
        &mut w,
        BodyDesc {
            position: Vec3::new(5.0, 10.0, 0.0),
            lock_rotation: [true, true, true],
            ..Default::default()
        },
        Shape::Box {
            half_extents: Vec3::splat(0.5),
        },
    );
    w.set_angular_velocity(spin, Vec3::new(0.0, 3.0, 0.0));
    assert_eq!(w.angular_velocity(spin), Some(Vec3::ZERO));
    // Only yaw free: the other two components are dropped.
    let yaw = body_with(
        &mut w,
        BodyDesc {
            position: Vec3::new(-5.0, 10.0, 0.0),
            angular_velocity: Vec3::new(2.0, 1.0, -2.0),
            lock_rotation: [true, false, true],
            ..Default::default()
        },
        Shape::Box {
            half_extents: Vec3::splat(0.5),
        },
    );
    assert_eq!(w.angular_velocity(yaw), Some(Vec3::new(0.0, 1.0, 0.0)));
    w.set_linear_velocity(up, Vec3::new(0.0, -7.0, 2.0));
    assert_eq!(w.linear_velocity(up), Some(Vec3::new(0.0, 0.0, 2.0)));
    run(&mut w, 1.0);
    let p = w.body_pose(up).unwrap().position;
    assert!(
        (p.y - 10.0).abs() < 1e-4 && (p.z - 2.0).abs() < 1e-3,
        "moved along a locked axis: {p:?}"
    );
    let q = w.body_pose(spin).unwrap().rotation;
    assert!(
        q.dot(Quat::IDENTITY).abs() > 0.9999,
        "turned about locked axes: {q:?}"
    );
    let q = w.body_pose(yaw).unwrap().rotation;
    let tilt = q.rotate(Vec3::Y);
    assert!(
        (tilt - Vec3::Y).length() < 1e-4,
        "tipped about locked axes: {q:?}"
    );
    assert!(
        q.twist_angle(Vec3::Y) > 0.9,
        "but turned about the free one: {q:?}"
    );
}

/// With some rotation axes locked, a body turns about the free ones as the inverse of the free
/// block of its world inertia says, not as the masked inverse of the whole tensor (review finding
/// LOCK-INERTIA).
#[test]
fn partly_locked_rotation_uses_the_inertia_of_the_free_axes() {
    // A plank (2 × 0.2 × 0.2 m, 80 kg) tilted 45° about Z: its world inertia couples X and Y.
    let tilt = Quat::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_4);
    let plank = Shape::Box {
        half_extents: Vec3::new(1.0, 0.1, 0.1),
    };
    let mass = 80.0f32;
    let (ix, iy) = (mass / 12.0 * 0.08, mass / 12.0 * 4.04);
    // World inertia: xx = yy = (ix + iy) / 2, xy = (ix - iy) / 2, zz = iy.
    let (diag, zz) = (0.5 * (ix + iy), iy);
    for (locks, impulse, expected) in [
        // Only yaw free: 1 / I_yy.
        (
            [true, false, true],
            Vec3::Y,
            Vec3::new(0.0, 1.0 / diag, 0.0),
        ),
        // Only Y locked: X and Z are uncoupled in the free block, so 1 / I_xx and 1 / I_zz.
        (
            [false, true, false],
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0 / diag, 0.0, 1.0 / zz),
        ),
        // Only Z locked: the X-Y block is coupled.
        ([false, false, true], Vec3::X, {
            let xy = 0.5 * (ix - iy);
            let det = diag * diag - xy * xy;
            Vec3::new(diag / det, -xy / det, 0.0)
        }),
    ] {
        let mut w = no_gravity();
        let b = body_with(
            &mut w,
            BodyDesc {
                rotation: tilt,
                lock_rotation: locks,
                ..Default::default()
            },
            plank.clone(),
        );
        assert!((w.mass(b).unwrap() - mass).abs() < 1e-3);
        w.apply_angular_impulse(b, impulse);
        let got = w.angular_velocity(b).unwrap();
        assert!(
            (got - expected).length() <= 1e-3 * expected.length(),
            "locks {locks:?}: an angular impulse {impulse:?} gives {got:?}, expected {expected:?}"
        );
    }
}

/// Re-targeting a rotated kinematic body to where it is must give it no velocity: normalizing the
/// target's rotation changed its last bits, which read as a turn of about 1e-6 rad/s, so the body
/// never slept and slowly drifted (review finding KIN-TARGET-NOISE).
#[test]
fn a_kinematic_body_targeted_to_its_own_pose_stays_still_and_sleeps() {
    let mut w = PhysicsWorld::default();
    let k = body_with(
        &mut w,
        BodyDesc {
            kind: BodyKind::Kinematic,
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::from_axis_angle(Vec3::new(0.3, 1.0, 0.2), 0.7),
            ..Default::default()
        },
        Shape::Box {
            half_extents: Vec3::new(2.0, 0.2, 1.0),
        },
    );
    let start = w.body_pose(k).unwrap();
    // The game keeps its own copy of the pose, or re-reads it every frame.
    let mut noisy = 0;
    for i in 0..120 {
        let target = if i % 2 == 0 {
            start
        } else {
            w.body_pose(k).unwrap()
        };
        w.set_kinematic_target(k, target, DT);
        if w.angular_velocity(k) != Some(Vec3::ZERO) || w.linear_velocity(k) != Some(Vec3::ZERO) {
            noisy += 1;
        }
        w.step(DT);
    }
    assert_eq!(noisy, 0, "steps with a velocity out of 120");
    let end = w.body_pose(k).unwrap();
    assert!(
        end.rotation.dot(start.rotation).abs() > 1.0 - 1e-6,
        "{end:?}"
    );
    assert_eq!(end.position, start.position);
    assert!(
        !w.is_awake(k),
        "a kinematic body that does not move falls asleep"
    );
    // A real target still moves and turns it, and wakes it.
    let turned = ganbate_physics::Iso::new(
        start.position + Vec3::new(0.1, 0.0, 0.0),
        Quat::from_axis_angle(Vec3::Y, 0.01) * start.rotation,
    );
    w.set_kinematic_target(k, turned, DT);
    assert!(w.is_awake(k));
    w.step(DT);
    let p = w.body_pose(k).unwrap();
    assert!((p.position - turned.position).length() < 1e-5, "{p:?}");
    assert!(p.rotation.dot(turned.rotation).abs() > 1.0 - 1e-6, "{p:?}");
}

/// A door turning at a constant world angular velocity must reach its requested orientation
/// for the frame, including on a world that uses only one solver substep.
#[test]
fn a_kinematic_rotation_reaches_its_target_for_any_substep_count() {
    for substeps in [1, 4, 8] {
        let mut w = PhysicsWorld::new(Settings {
            substeps,
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let start = Quat::from_axis_angle(Vec3::X, 0.6);
        let body = body_with(
            &mut w,
            BodyDesc {
                kind: BodyKind::Kinematic,
                rotation: start,
                ..Default::default()
            },
            Shape::Box {
                half_extents: Vec3::new(1.0, 0.5, 0.05),
            },
        );
        let turn = Quat::from_axis_angle(Vec3::new(0.2, 1.0, -0.4), 2.7);
        let target = ganbate_physics::Iso::new(Vec3::new(0.1, 0.2, 0.0), turn * start);
        w.set_kinematic_target(body, target, DT);
        for expected in [target.rotation, turn * target.rotation] {
            w.step(DT);
            let actual = w.body_pose(body).unwrap().rotation;
            for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                let error = (actual.rotate(axis) - expected.rotate(axis)).length();
                assert!(error < 1e-4, "{substeps} substeps, {axis:?}: {error}");
            }
        }
    }
}

fn all_finite(w: &PhysicsWorld) -> bool {
    w.bodies().all(|b| {
        let p = w.body_pose(b).unwrap();
        p.position.is_finite()
            && p.rotation.is_finite()
            && w.linear_velocity(b).unwrap().is_finite()
            && w.angular_velocity(b).unwrap().is_finite()
    })
}

/// Inputs that are not finite, or whose mass overflows, are refused instead of turning the
/// world's poses into NaN (review finding NAN-INPUTS).
#[test]
fn unusable_inputs_are_refused_and_the_world_stays_finite() {
    // A kinematic platform with a lamp hanging from it.
    let mut w = PhysicsWorld::default();
    let k = body_with(
        &mut w,
        BodyDesc {
            kind: BodyKind::Kinematic,
            position: Vec3::new(0.0, 5.0, 0.0),
            ..Default::default()
        },
        Shape::Box {
            half_extents: Vec3::new(1.0, 0.1, 1.0),
        },
    );
    let lamp = body_with(
        &mut w,
        BodyDesc {
            position: Vec3::new(0.0, 4.0, 0.0),
            ..Default::default()
        },
        Shape::Sphere { radius: 0.2 },
    );
    w.create_joint(&ganbate_physics::JointDesc {
        body_a: k,
        body_b: lamp,
        frame_a: ganbate_physics::Iso::from_position(Vec3::new(0.0, -1.0, 0.0)),
        frame_b: ganbate_physics::Iso::IDENTITY,
        kind: ganbate_physics::JointKind::Ball,
        collide_connected: false,
        user_data: 0,
    })
    .unwrap();
    let here = w.body_pose(k).unwrap();
    let moved = ganbate_physics::Iso::from_position(here.position + Vec3::new(0.1, 0.0, 0.0));
    let nan_position = ganbate_physics::Iso::from_position(Vec3::new(f32::NAN, 5.0, 0.0));
    let nan_rotation = ganbate_physics::Iso::new(here.position, Quat::new(f32::NAN, 0.0, 0.0, 1.0));
    let far = ganbate_physics::Iso::from_position(Vec3::new(1e30, 5.0, 0.0));
    for (target, dt) in [
        (nan_position, DT),
        (nan_rotation, DT),
        (moved, f32::NAN),
        (moved, f32::INFINITY),
        (moved, -DT),
        // Finite, but the velocity it asks for is not.
        (far, 1e-20),
    ] {
        w.set_kinematic_target(k, target, dt);
        assert_eq!(w.linear_velocity(k), Some(Vec3::ZERO), "{target:?} in {dt}");
        assert_eq!(
            w.angular_velocity(k),
            Some(Vec3::ZERO),
            "{target:?} in {dt}"
        );
        run(&mut w, 0.1);
        assert!(all_finite(&w), "{target:?} in {dt}");
    }
    // A gravity scale that is not a number.
    for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let r = w.create_body(&BodyDesc {
            gravity_scale: scale,
            ..Default::default()
        });
        assert!(r.is_err(), "gravity scale {scale} accepted");
    }
    // Masses that overflow: a dense box, a capsule too long.
    let b = w.create_body(&BodyDesc::default()).unwrap();
    let mass = w.mass(b).unwrap();
    let mut dense = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::ONE,
    });
    dense.density = 1e38;
    assert!(w.add_collider(b, &dense).is_err(), "density 1e38 accepted");
    let long = Shape::Capsule {
        half_height: 3e38,
        radius: 1.0,
    };
    assert!(long.validate().is_err());
    assert!(w.add_collider(b, &ColliderDesc::new(long)).is_err());
    assert_eq!(w.mass(b), Some(mass), "a refused collider changed the mass");
    // Two colliders whose masses overflow only together.
    let mut heavy = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::ONE,
    });
    heavy.density = 3e37;
    w.add_collider(b, &heavy).unwrap();
    assert!(
        w.add_collider(b, &heavy).is_err(),
        "the total mass overflows"
    );
    run(&mut w, 0.1);
    assert!(all_finite(&w));
}

/// A velocity so large that its squared length overflows is still clamped to the top speed,
/// along its direction.
#[test]
fn huge_velocities_are_clamped_along_their_direction() {
    let mut w = no_gravity();
    let b = body_with(&mut w, BodyDesc::default(), Shape::Sphere { radius: 0.5 });
    w.set_linear_velocity(b, Vec3::new(3e30, 4e30, 0.0));
    w.step(DT);
    let v = w.linear_velocity(b).unwrap();
    let top = w.settings.max_linear_speed;
    assert!(
        (v - Vec3::new(0.6, 0.8, 0.0) * top).length() < 1e-2,
        "{v:?}"
    );
    assert!(all_finite(&w));
}

/// Settings are public; a value out of its range is replaced by its default at the next step
/// instead of misbehaving: with `time_to_sleep = 0` a falling body was put to sleep in mid-air
/// after its first step, a negative top speed reversed fast bodies every substep, a negative
/// margin made broad-phase boxes smaller than their shapes (review finding
/// SETTINGS-UNVALIDATED).
#[test]
fn settings_out_of_range_do_not_misbehave() {
    // Zero time to sleep: still bodies sleep at once, moving ones never.
    let mut w = PhysicsWorld::new(Settings {
        time_to_sleep: 0.0,
        ..Default::default()
    });
    let falling = body_with(
        &mut w,
        BodyDesc {
            position: Vec3::new(0.0, 10.0, 0.0),
            ..Default::default()
        },
        Shape::Sphere { radius: 0.5 },
    );
    let floating = body_with(
        &mut w,
        BodyDesc {
            position: Vec3::new(5.0, 10.0, 0.0),
            gravity_scale: 0.0,
            ..Default::default()
        },
        Shape::Sphere { radius: 0.5 },
    );
    w.step(DT);
    assert!(
        !w.is_awake(floating),
        "a still body sleeps after one still step"
    );
    run(&mut w, 1.0);
    assert!(
        w.body_pose(falling).unwrap().position.y < 6.0,
        "a falling body stopped in mid-air: {:?}",
        w.body_pose(falling)
    );
    // Out of range: a negative top speed, margin and speculative distance, gravity and a
    // damping ratio that are not numbers.
    let mut w = PhysicsWorld::new(Settings {
        max_linear_speed: -1.0,
        aabb_margin: -0.5,
        speculative_distance: -0.1,
        contact_damping_ratio: f32::NAN,
        gravity: Vec3::new(0.0, f32::NAN, 0.0),
        ..Default::default()
    });
    let b = body_with(
        &mut w,
        BodyDesc {
            linear_velocity: Vec3::new(5.0, 0.0, 0.0),
            ..Default::default()
        },
        Shape::Sphere { radius: 0.5 },
    );
    for step in 0..30 {
        w.step(DT);
        let p = w.body_pose(b).unwrap().position;
        assert!(p.is_finite(), "step {step}: {p:?}");
        // The ball is where the queries look for it.
        let hit = w.raycast(
            Vec3::new(p.x + 0.45, p.y + 5.0, p.z),
            -Vec3::Y,
            10.0,
            &ganbate_physics::QueryFilter::default(),
        );
        assert!(
            hit.is_some_and(|h| h.body == b),
            "step {step}: missed at {p:?}"
        );
    }
    let v = w.linear_velocity(b).unwrap();
    assert!(v.x > 4.9, "reversed or stopped: {v:?}");
    let defaults = Settings::default();
    assert_eq!(w.settings.max_linear_speed, defaults.max_linear_speed);
    assert_eq!(w.settings.aabb_margin, defaults.aabb_margin);
    assert_eq!(
        w.settings.speculative_distance,
        defaults.speculative_distance
    );
    assert_eq!(
        w.settings.contact_damping_ratio,
        defaults.contact_damping_ratio
    );
    assert_eq!(w.settings.gravity, defaults.gravity);
    // Valid values are kept as given.
    let custom = Settings {
        gravity: Vec3::new(0.0, -1.62, 0.0),
        substeps: 8,
        time_to_sleep: 0.0,
        max_linear_speed: f32::INFINITY,
        ..Default::default()
    };
    let mut w = PhysicsWorld::new(custom.clone());
    w.step(DT);
    assert_eq!(w.settings, custom);
}
