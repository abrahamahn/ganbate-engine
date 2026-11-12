//! Regression tests for the joints' review findings: joints on bodies with locked axes (J1),
//! hinge limits at ±π (J2), and the numerical damping of springs (J9). As in `joints.rs`, the
//! colliders collide with nothing, so the results depend on the joints alone.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

use ganbate_physics::math::atan2;
use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, DistanceOptions, HingeOptions, Iso, JointDesc,
    JointKind, Motor, PhysicsWorld, Quat, Settings, Shape, SliderOptions, Spring, Vec3,
};

const DT: f32 = 1.0 / 60.0;
const G: f32 = 9.81;
const DEGREE: f32 = PI / 180.0;

fn world_with(substeps: u32) -> PhysicsWorld {
    PhysicsWorld::new(Settings {
        enable_sleep: false,
        substeps,
        ..Default::default()
    })
}

/// Default settings without sleeping.
fn world() -> PhysicsWorld {
    world_with(Settings::default().substeps)
}

fn fixed_body(w: &mut PhysicsWorld, position: Vec3) -> BodyId {
    w.create_body(&BodyDesc {
        kind: BodyKind::Static,
        position,
        ..Default::default()
    })
    .unwrap()
}

/// A dynamic body with one collider (density 1000) that collides with nothing.
fn dynamic_with(w: &mut PhysicsWorld, shape: Shape, desc: BodyDesc) -> BodyId {
    let b = w.create_body(&desc).unwrap();
    w.add_collider(
        b,
        &ColliderDesc {
            mask: 0,
            ..ColliderDesc::new(shape)
        },
    )
    .unwrap();
    b
}

fn join(
    w: &mut PhysicsWorld,
    a: BodyId,
    b: BodyId,
    frame_a: Iso,
    frame_b: Iso,
    kind: JointKind,
) -> JointDesc {
    let desc = JointDesc {
        body_a: a,
        body_b: b,
        frame_a,
        frame_b,
        kind,
        collide_connected: false,
        user_data: 0,
    };
    w.create_joint(&desc).unwrap();
    desc
}

/// The joint frame at `at` in world space, seen from `body`.
fn frame_at(w: &PhysicsWorld, body: BodyId, at: Iso) -> Iso {
    w.body_pose(body).unwrap().inverse_mul(&at)
}

/// A frame whose X axis (the hinge or slider axis) points along `axis`.
fn axis_frame(position: Vec3, axis: Vec3) -> Iso {
    Iso::new(position, Quat::from_rotation_arc(Vec3::X, axis))
}

fn steps(seconds: f32) -> usize {
    (seconds / DT).round() as usize
}

/// The angle of a body's X axis about world Z, from world X.
fn angle_about_z(w: &PhysicsWorld, body: BodyId) -> f32 {
    let x = w.body_pose(body).unwrap().rotation.rotate(Vec3::X);
    atan2(x.y, x.x)
}

/// `a` within `(-π, π]`.
fn wrapped(a: f32) -> f32 {
    if a > PI {
        a - TAU
    } else if a <= -PI {
        a + TAU
    } else {
        a
    }
}

const FREE: [bool; 3] = [false; 3];
const LOCKED: [bool; 3] = [true; 3];

// ---------------------------------------------------------------- J1: locked axes

/// 2.5D: a 1 m bar that may only move in the XY plane and turn about Z, hinged at one end about
/// Z to a static post, swings down from level. The hinge frame built with `from_rotation_arc`
/// leaves about 1e-7 of its Z axis along world Z, so the hinge's two axis rows (which the locks
/// make directions the bar cannot turn in) carried only rounding, which the solve inverted: their
/// impulses cancelled the free spin and the bar hung level. The locks only keep what the hinge
/// keeps anyway, so the bar must swing exactly as it does unlocked.
#[test]
fn a_planar_bar_hinged_about_its_free_axis_swings_as_it_does_unlocked() {
    let swing = |locked: bool| {
        let mut w = world();
        let pivot = Vec3::new(0.0, 5.0, 0.0);
        let post = fixed_body(&mut w, pivot);
        let rod = dynamic_with(
            &mut w,
            Shape::Box {
                half_extents: Vec3::new(0.5, 0.05, 0.05),
            },
            BodyDesc {
                position: pivot + Vec3::new(0.5, 0.0, 0.0),
                lock_rotation: if locked { [true, true, false] } else { FREE },
                lock_translation: if locked { [false, false, true] } else { FREE },
                ..Default::default()
            },
        );
        join(
            &mut w,
            post,
            rod,
            axis_frame(Vec3::ZERO, Vec3::Z),
            axis_frame(Vec3::new(-0.5, 0.0, 0.0), Vec3::Z),
            JointKind::Hinge(HingeOptions::default()),
        );
        for _ in 0..steps(0.5) {
            w.step(DT);
            if locked {
                assert_eq!(w.center_of_mass(rod).unwrap().z, 0.0, "in its plane");
            }
        }
        (angle_about_z(&w, rod), w.angular_velocity(rod).unwrap().z)
    };
    let (free, locked) = (swing(false), swing(true));
    assert!(free.0 < -1.5, "the bar swings down: {free:?}");
    assert!(
        (locked.0 - free.0).abs() < 1e-3 && (locked.1 - free.1).abs() < 1e-3,
        "locked {locked:?}, free {free:?} (angle, spin)"
    );
}

/// A block that may only move along world Y (no rotation, no other translation) on a vertical
/// slider. The slider's two rows across its axis are directions the block cannot move in, with
/// rounding of the frame along Y in them: it must fall along the rail as it does unlocked.
#[test]
fn a_slider_along_its_bodys_only_free_axis_slides() {
    let fall = |locked: bool| {
        let mut w = world();
        let top = Vec3::new(0.0, 5.0, 0.0);
        let rail = fixed_body(&mut w, top);
        let block = dynamic_with(
            &mut w,
            Shape::Box {
                half_extents: Vec3::splat(0.2),
            },
            BodyDesc {
                position: top,
                lock_rotation: if locked { LOCKED } else { FREE },
                lock_translation: if locked { [true, false, true] } else { FREE },
                ..Default::default()
            },
        );
        let frame = axis_frame(Vec3::ZERO, -Vec3::Y);
        join(
            &mut w,
            rail,
            block,
            frame,
            frame,
            JointKind::Slider(SliderOptions::default()),
        );
        for _ in 0..steps(0.5) {
            w.step(DT);
        }
        top.y - w.center_of_mass(block).unwrap().y
    };
    let (free, locked) = (fall(false), fall(true));
    assert!(free > 1.2, "it falls freely along the rail: {free} m");
    assert!(
        (locked - free).abs() < 1e-4,
        "locked {locked} m, free {free} m"
    );
}

/// A 2 m square plate that may only turn about world Y, driven by a hinge motor about Y that is
/// too weak for its speed: it must spin up at torque / inertia, as it does unlocked.
#[test]
fn a_motor_turns_a_body_that_may_only_turn_about_the_motors_axis() {
    let spin = |locked: bool| {
        let mut w = world();
        let at = Vec3::new(0.0, 2.0, 0.0);
        let mount = fixed_body(&mut w, at);
        let plate = dynamic_with(
            &mut w,
            Shape::Box {
                half_extents: Vec3::new(1.0, 0.01, 1.0),
            },
            BodyDesc {
                position: at,
                angular_damping: 0.0,
                lock_rotation: if locked { [true, false, true] } else { FREE },
                lock_translation: if locked { LOCKED } else { FREE },
                ..Default::default()
            },
        );
        let axle = axis_frame(Vec3::ZERO, Vec3::Y);
        join(
            &mut w,
            mount,
            plate,
            axle,
            axle,
            JointKind::Hinge(HingeOptions {
                motor: Some(Motor {
                    speed: 100.0,
                    max_force: 1.0,
                }),
                ..Default::default()
            }),
        );
        for _ in 0..steps(1.0) {
            w.step(DT);
        }
        let inertia = w.mass(plate).unwrap() * (1.0 + 1.0) / 3.0;
        (w.angular_velocity(plate).unwrap().y, 1.0 * 1.0 / inertia)
    };
    for locked in [false, true] {
        let (spin, expected) = spin(locked);
        assert!(
            (spin - expected).abs() < 1e-3 * expected,
            "locked {locked}: {spin} rad/s, expected {expected}"
        );
    }
}

/// A turntable that cannot move and may only turn about world Y, built rotated (a model authored
/// Z-up: turned 90° about X, then about Y), on a ball joint 0.1 m below its centre, on the axis.
/// The anchor's offset off the axis is only rounding (7.45e-9 in the table's frame), so the point
/// rows are directions the table cannot move the anchor in; inverted, they stopped it dead. Driven
/// by a torque, it must spin up at torque / inertia.
#[test]
fn a_turntable_on_a_ball_joint_on_its_axis_turns() {
    for start in [0.0f32, 0.7, 2.1] {
        let mut w = world();
        let at = Vec3::new(0.0, 1.0, 0.0);
        let ground = fixed_body(&mut w, Vec3::ZERO);
        let table = dynamic_with(
            &mut w,
            Shape::Box {
                half_extents: Vec3::new(1.0, 1.0, 0.05),
            },
            BodyDesc {
                position: at,
                rotation: Quat::from_axis_angle(Vec3::Y, start)
                    * Quat::from_axis_angle(Vec3::X, FRAC_PI_2),
                angular_damping: 0.0,
                lock_translation: LOCKED,
                lock_rotation: [true, false, true],
                ..Default::default()
            },
        );
        let pin = Iso::from_position(at - Vec3::new(0.0, 0.1, 0.0));
        let (fa, fb) = (frame_at(&w, ground, pin), frame_at(&w, table, pin));
        join(&mut w, ground, table, fa, fb, JointKind::Ball);
        let torque = 50.0;
        for _ in 0..steps(1.0) {
            w.apply_torque(table, Vec3::new(0.0, torque, 0.0));
            w.step(DT);
        }
        // About Y the table is a 2 m square plate.
        let expected = torque * 1.0 / (w.mass(table).unwrap() * (1.0 + 1.0) / 3.0);
        let spin = w.angular_velocity(table).unwrap();
        assert!(
            (spin.y - expected).abs() < 1e-3 * expected,
            "turned to {start}: {spin:?}, expected {expected} rad/s about Y"
        );
    }
}

/// The turntable again, its orientation 5e-5 rad off its free axis, as rounding leaves a body
/// whose rotation is locked about two axes after a long run (or as a rotation may be authored):
/// its anchor, 0.1 m below the centre, is then 5e-6 m off the axis, so turning moves the anchor
/// that little. The solve takes rows so nearly frozen for rounding (up to 1e-4 rad off the
/// axis), so the table must turn.
#[test]
fn a_turntable_slightly_off_its_axis_on_a_ball_joint_turns() {
    let mut w = world();
    let at = Vec3::new(0.0, 1.0, 0.0);
    let ground = fixed_body(&mut w, Vec3::ZERO);
    let table = dynamic_with(
        &mut w,
        Shape::Box {
            half_extents: Vec3::new(1.0, 0.05, 1.0),
        },
        BodyDesc {
            position: at,
            rotation: Quat::from_axis_angle(Vec3::X, 5e-5),
            angular_damping: 0.0,
            lock_translation: LOCKED,
            lock_rotation: [true, false, true],
            ..Default::default()
        },
    );
    let below = Vec3::new(0.0, -0.1, 0.0);
    join(
        &mut w,
        ground,
        table,
        Iso::from_position(at + below),
        Iso::from_position(below),
        JointKind::Ball,
    );
    let torque = 50.0;
    for _ in 0..steps(1.0) {
        w.apply_torque(table, Vec3::new(0.0, torque, 0.0));
        w.step(DT);
    }
    let expected = torque * 1.0 / (w.mass(table).unwrap() * (1.0 + 1.0) / 3.0);
    let spin = w.angular_velocity(table).unwrap();
    assert!(
        (spin.y - expected).abs() < 1e-3 * expected,
        "{spin:?}, expected {expected} rad/s about Y"
    );
}

// ---------------------------------------------------------------- J2: hinge limits at ±π

/// A 1 m bar (10 kg) hinged at one end about world Z on a static post, starting at `start` rad
/// and turning at `spin` rad/s. Returns the lowest and highest angle it reaches over `seconds`
/// (unwrapped: counted on round the turn), and the most its energy rose above the start.
fn swing_through_pi(limit: (f32, f32), start: f32, spin: f32, seconds: f32) -> (f32, f32, f32) {
    let mut w = world();
    let pivot = Vec3::new(0.0, 5.0, 0.0);
    let post = fixed_body(&mut w, pivot);
    let turn = Quat::from_axis_angle(Vec3::Z, start);
    let arm = turn.rotate(Vec3::new(0.5, 0.0, 0.0));
    let rod = dynamic_with(
        &mut w,
        Shape::Box {
            half_extents: Vec3::new(0.5, 0.05, 0.05),
        },
        BodyDesc {
            position: pivot + arm,
            rotation: turn,
            linear_velocity: Vec3::new(0.0, 0.0, spin).cross(arm),
            angular_velocity: Vec3::new(0.0, 0.0, spin),
            angular_damping: 0.0,
            ..Default::default()
        },
    );
    join(
        &mut w,
        post,
        rod,
        axis_frame(Vec3::ZERO, Vec3::Z),
        axis_frame(Vec3::new(-0.5, 0.0, 0.0), Vec3::Z),
        JointKind::Hinge(HingeOptions {
            limit: Some(limit),
            ..Default::default()
        }),
    );
    let mass = w.mass(rod).unwrap();
    let inertia = mass * (1.0 + 0.01) / 12.0 + mass * 0.25;
    let energy = |w: &PhysicsWorld| {
        let s = w.angular_velocity(rod).unwrap().z;
        0.5 * inertia * s * s + mass * G * w.center_of_mass(rod).unwrap().y
    };
    let e0 = energy(&w);
    let mut prev = angle_about_z(&w, rod);
    let mut angle = prev;
    let (mut lowest, mut highest, mut rise) = (angle, angle, f32::NEG_INFINITY);
    for _ in 0..steps(seconds) {
        w.step(DT);
        let now = angle_about_z(&w, rod);
        angle += wrapped(now - prev);
        prev = now;
        lowest = lowest.min(angle);
        highest = highest.max(angle);
        rise = rise.max(energy(&w) - e0);
    }
    (lowest, highest, rise)
}

/// A hinge's angle reads within (-π, π]. Limits at π, beyond it (taken as π) or missing on one
/// side all end at π, and the full turn keeps a 0.1 rad gap there. Once a joint was pushed past
/// its end at ±π, its angle read as far past the other limit, whose soft correction threw it on
/// round the turn, faster each time (some 20 turns and 3.5 kJ in 3 s). Each must now stop within
/// a degree of its end, and the stop may not feed the bar energy.
#[test]
fn hinge_limits_at_pi_stop_a_swinging_bar_there() {
    let nan = f32::NAN;
    let gap_end = PI - 0.05;
    // (limit, start, spin, where it stops): every case turns towards its stop.
    let cases = [
        ((-1.0, PI), 2.0, 4.0, PI),
        ((-1.0, 4.0), 2.0, 4.0, PI),
        ((-1.0, f32::INFINITY), 2.0, 4.0, PI),
        ((-1.0, nan), 2.0, 4.0, PI),
        ((0.0, nan), 2.0, 12.0, PI),
        ((-1.0, 3.13), 2.0, 12.0, 3.13),
        ((nan, 1.0), -2.0, -12.0, -PI),
        ((-PI, PI), 2.0, 4.0, gap_end),
        ((5.0, -7.0), 2.0, 12.0, gap_end),
        ((-PI, PI), -2.0, -12.0, -gap_end),
    ];
    for (limit, start, spin, end) in cases {
        let (lowest, highest, rise) = swing_through_pi(limit, start, spin, 3.0);
        let (reached, past) = if spin > 0.0 {
            (highest, highest - end)
        } else {
            (lowest, end - lowest)
        };
        assert!(
            past < DEGREE && past > -0.01,
            "{limit:?} turning at {spin} rad/s: reached {reached}, the stop is at {end}"
        );
        assert!(
            rise < 0.1,
            "{limit:?} turning at {spin} rad/s: the energy rose by {rise} J"
        );
    }
}

/// A motor driving a plate into an upper limit at π must be held there, as below π.
#[test]
fn a_hinge_motor_driving_into_a_limit_at_pi_is_held_there() {
    for upper in [3.0f32, PI] {
        let mut w = world();
        let hub = Vec3::new(0.0, 2.0, 0.0);
        let mount = fixed_body(&mut w, hub);
        let plate = dynamic_with(
            &mut w,
            Shape::Box {
                half_extents: Vec3::new(0.4, 0.4, 0.05),
            },
            BodyDesc {
                position: hub,
                angular_damping: 0.0,
                ..Default::default()
            },
        );
        let axle = axis_frame(Vec3::ZERO, Vec3::Z);
        join(
            &mut w,
            mount,
            plate,
            axle,
            axle,
            JointKind::Hinge(HingeOptions {
                limit: Some((-1.0, upper)),
                motor: Some(Motor {
                    speed: 5.0,
                    max_force: 100.0,
                }),
                ..Default::default()
            }),
        );
        let mut prev = angle_about_z(&w, plate);
        let mut angle = prev;
        let mut highest = angle;
        for _ in 0..steps(3.0) {
            w.step(DT);
            let now = angle_about_z(&w, plate);
            angle += wrapped(now - prev);
            prev = now;
            highest = highest.max(angle);
        }
        assert!(highest < upper + 0.01, "{upper}: reached {highest}");
        assert!((angle - upper).abs() < 0.01, "{upper}: held at {angle}");
        assert!(w.angular_velocity(plate).unwrap().length() < 1e-3);
    }
}

// ---------------------------------------------------------------- J9: spring damping

/// How fast an oscillation decays, as a damping ratio: from its envelope `√(x² + (v/ω)²)` at the
/// start and after four periods.
fn damping_ratio(
    w: &mut PhysicsWorld,
    hertz: f32,
    state: impl Fn(&PhysicsWorld) -> (f32, f32),
) -> f32 {
    let (omega, seconds) = (TAU * hertz, 4.0 / hertz);
    let envelope = |w: &PhysicsWorld| {
        let (x, v) = state(w);
        (x * x + (v / omega) * (v / omega)).sqrt()
    };
    let start = envelope(w);
    for _ in 0..steps(seconds) {
        w.step(DT);
    }
    -(envelope(w) / start).ln() / (omega * seconds)
}

/// Undamped hinge, slider and distance springs of `hertz` in worlds of `substeps`: their decay
/// as damping ratios.
fn undamped_springs(hertz: f32, substeps: u32) -> [f32; 3] {
    let spring = Some(Spring {
        hertz,
        damping_ratio: 0.0,
    });
    // A 0.8 m square plate centred on a hinge about world Z (gravity does not turn it), spun.
    let mut w = world_with(substeps);
    let hub = Vec3::new(0.0, 2.0, 0.0);
    let mount = fixed_body(&mut w, hub);
    let plate = dynamic_with(
        &mut w,
        Shape::Box {
            half_extents: Vec3::new(0.4, 0.4, 0.05),
        },
        BodyDesc {
            position: hub,
            angular_velocity: Vec3::new(0.0, 0.0, 1.0),
            angular_damping: 0.0,
            ..Default::default()
        },
    );
    let axle = axis_frame(Vec3::ZERO, Vec3::Z);
    join(
        &mut w,
        mount,
        plate,
        axle,
        axle,
        JointKind::Hinge(HingeOptions {
            spring,
            ..Default::default()
        }),
    );
    let hinge = damping_ratio(&mut w, hertz, |w| {
        (
            angle_about_z(w, plate),
            w.angular_velocity(plate).unwrap().z,
        )
    });
    // A cube on a level rail along world X (gravity pulls across it), pushed.
    let mut w = world_with(substeps);
    let start = Vec3::new(0.0, 2.0, 0.0);
    let rail = fixed_body(&mut w, start);
    let block = dynamic_with(
        &mut w,
        Shape::Box {
            half_extents: Vec3::splat(0.2),
        },
        BodyDesc {
            position: start,
            linear_velocity: Vec3::new(1.0, 0.0, 0.0),
            linear_damping: 0.0,
            ..Default::default()
        },
    );
    join(
        &mut w,
        rail,
        block,
        Iso::IDENTITY,
        Iso::IDENTITY,
        JointKind::Slider(SliderOptions {
            spring,
            ..Default::default()
        }),
    );
    let slider = damping_ratio(&mut w, hertz, |w| {
        (
            w.center_of_mass(block).unwrap().x,
            w.linear_velocity(block).unwrap().x,
        )
    });
    // A ball hanging from a spring of rest length 1 m, released at that length: it bobs about
    // the point where the spring carries its weight, g / ω² lower.
    let mut w = world_with(substeps);
    let hook = fixed_body(&mut w, Vec3::new(0.0, 5.0, 0.0));
    let ball = dynamic_with(
        &mut w,
        Shape::Sphere { radius: 0.1 },
        BodyDesc {
            position: Vec3::new(0.0, 4.0, 0.0),
            linear_damping: 0.0,
            ..Default::default()
        },
    );
    join(
        &mut w,
        hook,
        ball,
        Iso::IDENTITY,
        Iso::IDENTITY,
        JointKind::Distance(DistanceOptions {
            length: 1.0,
            min: 0.0,
            max: 10.0,
            spring,
        }),
    );
    let omega = TAU * hertz;
    let rest = 4.0 - G / (omega * omega);
    let distance = damping_ratio(&mut w, hertz, |w| {
        (
            w.center_of_mass(ball).unwrap().y - rest,
            w.linear_velocity(ball).unwrap().y,
        )
    });
    [hinge, slider, distance]
}

/// Springs act in both passes of every substep, as in Box2D v3 (the relax pass keeps stiff
/// springs stable), and each pass is implicit, which damps a spring as if its damping ratio were
/// higher by about `h·ω` (exactly `ln(1 + (h·ω)²) / (h·ω)` for an undamped one). `Spring`
/// documents this; here it is measured, for every kind of spring, with the default 4 substeps
/// and with 16, which must damp four times less.
#[test]
fn undamped_springs_lose_their_swing_at_the_documented_numerical_damping() {
    for hertz in [0.5f32, 2.0, 8.0] {
        for substeps in [4u32, 16] {
            let hw = TAU * hertz * DT / substeps as f32;
            let documented = (1.0 + hw * hw).ln() / hw;
            for (kind, measured) in ["hinge", "slider", "distance"]
                .into_iter()
                .zip(undamped_springs(hertz, substeps))
            {
                assert!(
                    (measured - documented).abs() < 0.05 * documented,
                    "{kind}, {hertz} Hz, {substeps} substeps: damping ratio {measured}, \
                     documented {documented} (h·ω = {hw})"
                );
            }
        }
    }
}

/// A needle (2 m by 0.2 mm, the thinnest box a shape accepts, and the same as a capsule) hanging
/// from a ball joint at its end stays on it: the rounding floor of the joint blocks must not drop
/// a needle's tiny but real pivots (found by the pivot fix's verifier).
#[test]
fn a_needle_hanging_from_a_ball_joint_stays_on_it() {
    for shape in [
        Shape::Box {
            half_extents: Vec3::new(1e-4, 1.0, 1e-4),
        },
        Shape::Capsule {
            half_height: 1.0 - 1e-4,
            radius: 1e-4,
        },
    ] {
        let mut w = PhysicsWorld::new(Settings {
            enable_sleep: false,
            ..Default::default()
        });
        let top = Vec3::new(0.0, 100.0, 0.0);
        let post = w
            .create_body(&BodyDesc {
                kind: BodyKind::Static,
                position: top,
                ..Default::default()
            })
            .unwrap();
        let needle = w
            .create_body(&BodyDesc {
                position: top - Vec3::Y,
                ..Default::default()
            })
            .unwrap();
        w.add_collider(
            needle,
            &ColliderDesc {
                mask: 0,
                ..ColliderDesc::new(shape.clone())
            },
        )
        .unwrap();
        w.create_joint(&JointDesc {
            body_a: post,
            body_b: needle,
            frame_a: Iso::IDENTITY,
            frame_b: Iso::from_position(Vec3::Y),
            kind: JointKind::Ball,
            collide_connected: false,
            user_data: 0,
        })
        .unwrap();
        for _ in 0..120 {
            w.step(DT);
        }
        let gap = (w.body_pose(needle).unwrap().point(Vec3::Y) - top).length();
        assert!(
            gap < 1e-3,
            "{shape:?}: the needle's end is {gap} m off the joint"
        );
    }
}
