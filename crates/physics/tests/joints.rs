//! Joints in a whole world: pendulums, chains, limits, motors, springs, ropes, momentum, locked
//! axes, determinism and degenerate input. Every scene here is synthetic and has no ground; all
//! but the momentum test have gravity; colliders collide with nothing (empty masks), so the
//! results depend on the joints alone, whatever the narrow phase does.

use std::f32::consts::{FRAC_PI_4, PI, TAU};

use ganbate_physics::math::atan2;
use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, DistanceOptions, HingeOptions, Iso, JointDesc,
    JointKind, Motor, PhysicsWorld, Quat, Settings, Shape, SliderOptions, Spring, Vec3,
};

const DT: f32 = 1.0 / 60.0;
const G: f32 = 9.81;
const DEGREE: f32 = PI / 180.0;

/// Default settings without sleeping: a joint must hold on its own, not because its bodies fell
/// asleep.
fn world() -> PhysicsWorld {
    PhysicsWorld::new(Settings {
        enable_sleep: false,
        ..Default::default()
    })
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

fn dynamic(w: &mut PhysicsWorld, shape: Shape, position: Vec3) -> BodyId {
    dynamic_with(
        w,
        shape,
        BodyDesc {
            position,
            ..Default::default()
        },
    )
}

fn bar(half_length: f32) -> Shape {
    Shape::Box {
        half_extents: Vec3::new(half_length, 0.05, 0.05),
    }
}

fn join(
    w: &mut PhysicsWorld,
    a: BodyId,
    b: BodyId,
    frame_a: Iso,
    frame_b: Iso,
    kind: JointKind,
) -> JointDesc {
    join_as(w, a, b, frame_a, frame_b, kind, false)
}

/// [`join`], with bodies A and B (and their frames) swapped when `swap`.
fn join_as(
    w: &mut PhysicsWorld,
    a: BodyId,
    b: BodyId,
    frame_a: Iso,
    frame_b: Iso,
    kind: JointKind,
    swap: bool,
) -> JointDesc {
    let (a, b, frame_a, frame_b) = if swap {
        (b, a, frame_b, frame_a)
    } else {
        (a, b, frame_a, frame_b)
    };
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

/// Where a joint frame's origin is now.
fn anchor(w: &PhysicsWorld, body: BodyId, frame: &Iso) -> Vec3 {
    w.body_pose(body).unwrap().point(frame.position)
}

/// How far apart a joint's two frame origins are now.
fn gap(w: &PhysicsWorld, j: &JointDesc) -> f32 {
    anchor(w, j.body_a, &j.frame_a).distance(anchor(w, j.body_b, &j.frame_b))
}

/// The angle of a rotation, in `[0, π]`.
fn angle_of(q: Quat) -> f32 {
    2.0 * atan2(q.vector().length(), q.w.abs())
}

/// A frame whose X axis (the hinge or slider axis) points along `axis`.
fn axis_frame(position: Vec3, axis: Vec3) -> Iso {
    Iso::new(position, Quat::from_rotation_arc(Vec3::X, axis))
}

fn steps(seconds: f32) -> usize {
    (seconds / DT).round() as usize
}

#[test]
fn a_small_pendulum_on_a_ball_joint_swings_with_the_period_of_a_simple_pendulum() {
    let mut w = world();
    let pivot = Vec3::new(0.0, 3.0, 0.0);
    let top = fixed_body(&mut w, pivot);
    let (length, amplitude) = (1.0f32, 0.05f32);
    // A small ball held 1 m below the pivot, released 0.05 rad from the vertical. Its own inertia
    // lengthens the period by 0.05%, the amplitude by 0.02%.
    let at = pivot + Vec3::new(length * amplitude.sin(), -length * amplitude.cos(), 0.0);
    let bob = dynamic(&mut w, Shape::Sphere { radius: 0.05 }, at);
    let j = join(
        &mut w,
        top,
        bob,
        Iso::IDENTITY,
        Iso::from_position(pivot - at),
        JointKind::Ball,
    );
    // Times the bob passes the bottom going towards -x, interpolated within the step.
    let mut passes = Vec::new();
    let mut prev = w.center_of_mass(bob).unwrap().x;
    let mut worst_gap = 0.0f32;
    for step in 1..=steps(12.0) {
        w.step(DT);
        let x = w.center_of_mass(bob).unwrap().x;
        if prev > 0.0 && x <= 0.0 {
            passes.push((step as f32 - 1.0 + prev / (prev - x)) * DT);
        }
        prev = x;
        worst_gap = worst_gap.max(gap(&w, &j));
    }
    let n = passes.len() - 1;
    assert!(n >= 4, "it swings: {passes:?}");
    let period = (passes[n] - passes[0]) / n as f32;
    let expected = TAU * (length / G).sqrt();
    assert!(
        (period - expected).abs() < 0.02 * expected,
        "period {period} s, expected {expected} s"
    );
    assert!(worst_gap < 1e-3, "the pivot holds: {worst_gap} m");
    // The swing keeps its size: the joint neither feeds nor drains much energy.
    let x = w.center_of_mass(bob).unwrap().x.abs();
    assert!(x < 1.05 * length * amplitude.sin());
}

#[test]
fn a_chain_of_ten_links_hanging_from_a_static_body_stretches_less_than_one_percent() {
    let mut w = world();
    let top = Vec3::new(0.0, 10.0, 0.0);
    let ceiling = fixed_body(&mut w, top);
    // Ten 0.5 m links laid out horizontally from the ceiling, so the chain falls and swings.
    let half = 0.25;
    let mut joints = Vec::new();
    let (mut prev, mut prev_frame) = (ceiling, Iso::IDENTITY);
    for i in 0..10 {
        let at = top + Vec3::new((2 * i + 1) as f32 * half, 0.0, 0.0);
        let link = dynamic(&mut w, bar(half), at);
        let frame = Iso::from_position(Vec3::new(-half, 0.0, 0.0));
        joints.push(join(&mut w, prev, link, prev_frame, frame, JointKind::Ball));
        (prev, prev_frame) = (link, Iso::from_position(Vec3::new(half, 0.0, 0.0)));
    }
    let length = 10.0 * 2.0 * half;
    let stretch = |w: &PhysicsWorld| joints.iter().map(|j| gap(w, j)).sum::<f32>();
    let mut worst = 0.0f32;
    let mut lowest = f32::INFINITY;
    for _ in 0..steps(5.0) {
        w.step(DT);
        worst = worst.max(stretch(&w));
        lowest = lowest.min(anchor(&w, prev, &prev_frame).y);
    }
    let now = stretch(&w);
    assert!(
        now < 0.01 * length,
        "stretched {now} m of {length} m after 5 s (worst {worst} m)"
    );
    assert!(worst < 0.02 * length, "worst stretch {worst} m");
    assert!(lowest < top.y - 0.9 * length, "it swung down: {lowest}");
}

#[test]
fn a_hinge_limit_stops_a_swinging_bar_at_the_limit_angle_and_holds_it() {
    // With the static post as body A, and as body B (the angle of B relative to A then has the
    // other sign, and the upper limit stops the bar).
    for swap in [false, true] {
        hinge_limit(swap);
    }
}

fn hinge_limit(swap: bool) {
    let mut w = world();
    let pivot = Vec3::new(0.0, 5.0, 0.0);
    let post = fixed_body(&mut w, pivot);
    // A 1 m bar hinged at one end about the world Z axis, starting level: it swings down until
    // the lower limit, 45° below level, stops it.
    let rod = dynamic(&mut w, bar(0.5), pivot + Vec3::new(0.5, 0.0, 0.0));
    let limit = -FRAC_PI_4;
    let j = join_as(
        &mut w,
        post,
        rod,
        axis_frame(Vec3::ZERO, Vec3::Z),
        axis_frame(Vec3::new(-0.5, 0.0, 0.0), Vec3::Z),
        JointKind::Hinge(HingeOptions {
            limit: Some((limit, FRAC_PI_4)),
            ..Default::default()
        }),
        swap,
    );
    let angle = |w: &PhysicsWorld| {
        let x = w.body_pose(rod).unwrap().rotation.rotate(Vec3::X);
        atan2(x.y, x.x)
    };
    let mut lowest = 0.0f32;
    let mut reached = None;
    for step in 0..steps(3.0) {
        w.step(DT);
        lowest = lowest.min(angle(&w));
        if reached.is_none() && angle(&w) < limit + DEGREE {
            reached = Some(step);
        }
    }
    assert!(
        reached.is_some_and(|s| s < 60),
        "it reaches the limit: {reached:?}"
    );
    assert!(
        lowest > limit - DEGREE,
        "it never goes more than a degree past the limit: {} degrees",
        lowest / DEGREE
    );
    let now = angle(&w);
    assert!(
        (now - limit).abs() < DEGREE,
        "it rests at the limit: {} degrees",
        now / DEGREE
    );
    let spin = w.angular_velocity(rod).unwrap().length();
    assert!(spin < 0.01, "and stays there: {spin} rad/s");
    assert!(gap(&w, &j) < 1e-3, "the pivot holds: {} m", gap(&w, &j));
    // Rotation stays about the hinge axis.
    let axis = w.body_pose(rod).unwrap().rotation.rotate(Vec3::Z);
    assert!(axis.dot(Vec3::Z) > 0.99999, "{axis:?}");
}

#[test]
fn a_hinge_motor_spins_a_free_wheel_to_its_speed_and_saturates_when_too_weak() {
    let target = 8.0;
    for (max_torque, strong) in [(500.0f32, true), (20.0, false)] {
        let mut w = world();
        let hub = Vec3::new(0.0, 2.0, 0.0);
        let mount = fixed_body(&mut w, hub);
        // A 0.8 m square plate turning about the world Z axis through its centre.
        let (a, t) = (0.4f32, 0.05f32);
        let wheel = dynamic_with(
            &mut w,
            Shape::Box {
                half_extents: Vec3::new(a, a, t),
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
            wheel,
            axle,
            axle,
            JointKind::Hinge(HingeOptions {
                motor: Some(Motor {
                    speed: target,
                    max_force: max_torque,
                }),
                ..Default::default()
            }),
        );
        let mass = w.mass(wheel).unwrap();
        let inertia = mass * (a * a + a * a) / 3.0;
        for _ in 0..steps(1.0) {
            w.step(DT);
        }
        let spin = w.angular_velocity(wheel).unwrap();
        if strong {
            assert!(
                (spin.z - target).abs() < 0.001 * target,
                "a strong motor reaches its speed: {spin:?}"
            );
        } else {
            // The full torque for the whole second, and no more.
            let expected = max_torque * 1.0 / inertia;
            assert!(expected < 0.5 * target);
            assert!(
                (spin.z - expected).abs() < 0.001 * expected,
                "a weak motor saturates: {} rad/s, expected {expected}",
                spin.z
            );
        }
        assert!(spin.x.abs() + spin.y.abs() < 1e-4, "{spin:?}");
        assert!(w.center_of_mass(wheel).unwrap().distance(hub) < 1e-3);
    }
}

#[test]
fn a_slider_limit_stops_a_falling_body() {
    let mut w = world();
    let top = Vec3::new(0.0, 5.0, 0.0);
    let rail = fixed_body(&mut w, top);
    // A vertical rail (the slider axis points down); the block falls 1.5 m to the limit.
    let block = dynamic(
        &mut w,
        Shape::Box {
            half_extents: Vec3::splat(0.2),
        },
        top,
    );
    let frame = axis_frame(Vec3::ZERO, -Vec3::Y);
    join(
        &mut w,
        rail,
        block,
        frame,
        frame,
        JointKind::Slider(SliderOptions {
            limit: Some((-0.5, 1.5)),
            ..Default::default()
        }),
    );
    let stop = top.y - 1.5;
    let mut lowest = f32::INFINITY;
    for _ in 0..steps(2.0) {
        w.step(DT);
        lowest = lowest.min(w.center_of_mass(block).unwrap().y);
    }
    assert!(
        lowest > stop - 0.005,
        "it never goes 5 mm past the stop: {lowest}"
    );
    let at = w.center_of_mass(block).unwrap();
    assert!((at.y - stop).abs() < 0.002, "it rests at the stop: {at:?}");
    assert!(
        at.x.abs() < 1e-4 && at.z.abs() < 1e-4,
        "on the rail: {at:?}"
    );
    assert!(w.linear_velocity(block).unwrap().length() < 0.01);
    let turned = angle_of(w.body_pose(block).unwrap().rotation);
    assert!(turned < 0.01 * DEGREE, "it does not turn: {turned}");
}

#[test]
fn a_fixed_joint_holds_a_lever_under_gravity() {
    for swap in [false, true] {
        fixed_lever(swap);
    }
}

fn fixed_lever(swap: bool) {
    let mut w = world();
    let wall = fixed_body(&mut w, Vec3::new(0.0, 2.0, 0.0));
    let lever = dynamic(&mut w, bar(0.5), Vec3::new(0.5, 2.0, 0.0));
    let j = join_as(
        &mut w,
        wall,
        lever,
        Iso::IDENTITY,
        Iso::from_position(Vec3::new(-0.5, 0.0, 0.0)),
        JointKind::Fixed,
        swap,
    );
    let (mut worst_gap, mut worst_angle) = (0.0f32, 0.0f32);
    for _ in 0..steps(3.0) {
        w.step(DT);
        worst_gap = worst_gap.max(gap(&w, &j));
        worst_angle = worst_angle.max(angle_of(w.body_pose(lever).unwrap().rotation));
    }
    assert!(worst_gap < 1e-3, "{worst_gap} m");
    assert!(
        worst_angle < 0.5 * DEGREE,
        "{} degrees",
        worst_angle / DEGREE
    );
    assert!(w.linear_velocity(lever).unwrap().length() < 1e-3);
}

#[test]
fn a_distance_spring_oscillates_near_its_frequency() {
    let mut w = world();
    let hook = fixed_body(&mut w, Vec3::new(0.0, 5.0, 0.0));
    // A ball on a 2 Hz spring of rest length 1 m, released at the rest length: it bobs about the
    // point where the spring carries its weight, g / ω² lower.
    let ball = dynamic(
        &mut w,
        Shape::Sphere { radius: 0.1 },
        Vec3::new(0.0, 4.0, 0.0),
    );
    let hertz = 2.0;
    join(
        &mut w,
        hook,
        ball,
        Iso::IDENTITY,
        Iso::IDENTITY,
        JointKind::Distance(DistanceOptions {
            length: 1.0,
            min: 0.0,
            max: 2.0,
            spring: Some(Spring {
                hertz,
                damping_ratio: 0.0,
            }),
        }),
    );
    let omega = TAU * hertz;
    let rest = 4.0 - G / (omega * omega);
    let mut ups = Vec::new();
    let mut prev = w.center_of_mass(ball).unwrap().y - rest;
    let mut deepest = f32::INFINITY;
    for step in 1..=steps(2.0) {
        w.step(DT);
        let y = w.center_of_mass(ball).unwrap().y - rest;
        if prev < 0.0 && y >= 0.0 {
            ups.push((step as f32 - 1.0 - prev / (y - prev)) * DT);
        }
        prev = y;
        deepest = deepest.min(y);
    }
    let n = ups.len() - 1;
    assert!(n >= 3, "it oscillates: {ups:?}");
    let frequency = n as f32 / (ups[n] - ups[0]);
    assert!(
        (frequency - hertz).abs() < 0.02 * hertz,
        "{frequency} Hz for a {hertz} Hz spring"
    );
    // The first dip goes about as far below the rest point as the release was above it.
    let drop = G / (omega * omega);
    assert!(
        deepest < -0.8 * drop && deepest > -1.05 * drop,
        "{deepest} vs {drop}"
    );
}

/// Times `signal` rises through zero over `seconds` (interpolated within the step), and its
/// lowest value.
fn rises(
    w: &mut PhysicsWorld,
    seconds: f32,
    mut signal: impl FnMut(&PhysicsWorld) -> f32,
) -> (Vec<f32>, f32) {
    let mut times = Vec::new();
    let mut prev = signal(w);
    let mut lowest = prev;
    for step in 1..=steps(seconds) {
        w.step(DT);
        let now = signal(w);
        if prev < 0.0 && now >= 0.0 {
            times.push((step as f32 - 1.0 - prev / (now - prev)) * DT);
        }
        prev = now;
        lowest = lowest.min(now);
    }
    (times, lowest)
}

fn frequency(times: &[f32]) -> f32 {
    let n = times.len() - 1;
    n as f32 / (times[n] - times[0])
}

/// A 0.8 m square plate centred on the hinge (about world Z, through its centre of mass), so
/// gravity does not turn it; and its moment of inertia about the hinge.
fn wheel(w: &mut PhysicsWorld, options: HingeOptions, spin: f32) -> (BodyId, f32) {
    let hub = Vec3::new(0.0, 2.0, 0.0);
    let mount = fixed_body(w, hub);
    let a = 0.4f32;
    let plate = dynamic_with(
        w,
        Shape::Box {
            half_extents: Vec3::new(a, a, 0.05),
        },
        BodyDesc {
            position: hub,
            angular_velocity: Vec3::new(0.0, 0.0, spin),
            angular_damping: 0.0,
            ..Default::default()
        },
    );
    let axle = axis_frame(Vec3::ZERO, Vec3::Z);
    join(w, mount, plate, axle, axle, JointKind::Hinge(options));
    let inertia = w.mass(plate).unwrap() * (a * a + a * a) / 3.0;
    (plate, inertia)
}

/// A 0.4 m cube on a level rail along world X (gravity pulls across it).
fn carriage(w: &mut PhysicsWorld, options: SliderOptions, speed: f32) -> BodyId {
    let start = Vec3::new(0.0, 2.0, 0.0);
    let rail = fixed_body(w, start);
    let block = dynamic_with(
        w,
        Shape::Box {
            half_extents: Vec3::splat(0.2),
        },
        BodyDesc {
            position: start,
            linear_velocity: Vec3::new(speed, 0.0, 0.0),
            ..Default::default()
        },
    );
    join(
        w,
        rail,
        block,
        Iso::IDENTITY,
        Iso::IDENTITY,
        JointKind::Slider(options),
    );
    block
}

fn plate_angle(w: &PhysicsWorld, plate: BodyId) -> f32 {
    let x = w.body_pose(plate).unwrap().rotation.rotate(Vec3::X);
    atan2(x.y, x.x)
}

#[test]
fn hinge_and_slider_springs_oscillate_near_their_frequency() {
    let spring = |hertz| {
        Some(Spring {
            hertz,
            damping_ratio: 0.0,
        })
    };
    let mut w = world();
    let (plate, _) = wheel(
        &mut w,
        HingeOptions {
            spring: spring(1.0),
            ..Default::default()
        },
        1.0,
    );
    let (times, _) = rises(&mut w, 4.0, |w| plate_angle(w, plate));
    let f = frequency(&times);
    assert!((f - 1.0).abs() < 0.02, "hinge spring: {f} Hz for 1 Hz");
    let spin = w.angular_velocity(plate).unwrap();
    assert!(spin.x.abs() + spin.y.abs() < 1e-4, "{spin:?}");

    let mut w = world();
    let block = carriage(
        &mut w,
        SliderOptions {
            spring: spring(1.5),
            ..Default::default()
        },
        1.0,
    );
    let (times, _) = rises(&mut w, 3.0, |w| w.center_of_mass(block).unwrap().x);
    let f = frequency(&times);
    assert!((f - 1.5).abs() < 0.03, "slider spring: {f} Hz for 1.5 Hz");
    let at = w.center_of_mass(block).unwrap();
    assert!(
        (at.y - 2.0).abs() < 1e-3 && at.z.abs() < 1e-4,
        "on the rail: {at:?}"
    );
}

#[test]
fn a_slider_motor_drives_its_body_to_speed_and_saturates_when_too_weak() {
    let target = 1.5;
    for (max_force, strong) in [(2000.0f32, true), (10.0, false)] {
        let mut w = world();
        let block = carriage(
            &mut w,
            SliderOptions {
                motor: Some(Motor {
                    speed: target,
                    max_force,
                }),
                ..Default::default()
            },
            0.0,
        );
        for _ in 0..steps(1.0) {
            w.step(DT);
        }
        let v = w.linear_velocity(block).unwrap();
        let expected = if strong {
            target
        } else {
            max_force * 1.0 / w.mass(block).unwrap()
        };
        assert!(expected <= target);
        assert!(
            (v.x - expected).abs() < 0.001 * expected,
            "{max_force} N: {v:?}, expected {expected} m/s"
        );
        assert!(v.y.abs() < 1e-3 && v.z.abs() < 1e-4, "{v:?}");
    }
}

#[test]
fn a_hinge_motor_driving_into_a_limit_is_held_there() {
    let mut w = world();
    let (plate, _) = wheel(
        &mut w,
        HingeOptions {
            limit: Some((-0.5, 0.5)),
            motor: Some(Motor {
                speed: 5.0,
                max_force: 100.0,
            }),
            ..Default::default()
        },
        0.0,
    );
    let mut highest = f32::NEG_INFINITY;
    for _ in 0..steps(2.0) {
        w.step(DT);
        highest = highest.max(plate_angle(&w, plate));
    }
    let angle = plate_angle(&w, plate);
    assert!(highest < 0.5 + DEGREE, "{highest}");
    assert!((angle - 0.5).abs() < 0.1 * DEGREE, "{angle}");
    assert!(w.angular_velocity(plate).unwrap().length() < 1e-3);
}

#[test]
fn a_rope_is_slack_until_taut_and_a_rod_keeps_its_length_both_ways() {
    let mut w = world();
    let hook = Vec3::new(0.0, 5.0, 0.0);
    let ceiling = fixed_body(&mut w, hook);
    // A 1 m rope (no spring: free between 0 and 1 m) holding a ball released 0.5 m below the hook.
    let start = hook - Vec3::new(0.0, 0.5, 0.0);
    let ball = dynamic(&mut w, Shape::Sphere { radius: 0.1 }, start);
    join(
        &mut w,
        ceiling,
        ball,
        Iso::IDENTITY,
        Iso::IDENTITY,
        JointKind::Distance(DistanceOptions {
            length: 1.0,
            min: 0.0,
            max: 1.0,
            spring: None,
        }),
    );
    // A 1 m rigid rod holding a ball above its pivot, nudged so that it falls over.
    let pivot = Vec3::new(3.0, 5.0, 0.0);
    let post = fixed_body(&mut w, pivot);
    let top = dynamic_with(
        &mut w,
        Shape::Sphere { radius: 0.1 },
        BodyDesc {
            position: pivot + Vec3::Y,
            linear_velocity: Vec3::new(0.2, 0.0, 0.0),
            ..Default::default()
        },
    );
    let rod = join(
        &mut w,
        post,
        top,
        Iso::IDENTITY,
        Iso::IDENTITY,
        JointKind::Distance(DistanceOptions {
            length: 1.0,
            min: 1.0,
            max: 1.0,
            spring: None,
        }),
    );
    let mut longest_rope = 0.0f32;
    let mut rod_error = 0.0f32;
    let mut rod_lowest = f32::INFINITY;
    for step in 1..=steps(3.0) {
        w.step(DT);
        let t = step as f32 * DT;
        let y = w.center_of_mass(ball).unwrap().y;
        if t < 0.3 {
            // Exactly the world's free fall (semi-implicit Euler over 4 substeps of h):
            // y = y0 - g·t·(t + h)/2.
            let free = start.y - 0.5 * G * t * (t + DT / 4.0);
            assert!(
                (y - free).abs() < 1e-4,
                "slack, it falls freely: {y} vs {free} at {t}"
            );
        }
        longest_rope = longest_rope.max(hook.y - y);
        let length = gap(&w, &rod);
        rod_error = rod_error.max((length - 1.0).abs());
        rod_lowest = rod_lowest.min(w.center_of_mass(top).unwrap().y);
    }
    assert!(
        longest_rope < 1.0 + 2e-3,
        "taut, it stops at the rope's length: {longest_rope}"
    );
    let hang = hook.y - w.center_of_mass(ball).unwrap().y;
    assert!((hang - 1.0).abs() < 1e-3, "and hangs there: {hang}");
    assert!(rod_error < 2e-3, "the rod keeps its length: {rod_error}");
    assert!(
        rod_lowest < pivot.y - 0.9,
        "pushing and pulling as it swings under: {rod_lowest}"
    );
}

/// Linear and angular momentum (about the origin) of spheres of the given radii.
fn momentum(w: &PhysicsWorld, bodies: &[(BodyId, f32)]) -> (Vec3, Vec3) {
    let (mut p, mut l) = (Vec3::ZERO, Vec3::ZERO);
    for &(b, radius) in bodies {
        let m = w.mass(b).unwrap();
        let v = w.linear_velocity(b).unwrap();
        p += v * m;
        l += w.center_of_mass(b).unwrap().cross(v * m)
            + w.angular_velocity(b).unwrap() * (0.4 * m * radius * radius);
    }
    (p, l)
}

#[test]
fn joints_between_two_dynamic_bodies_conserve_the_pairs_momentum() {
    let spring = Some(Spring {
        hertz: 3.0,
        damping_ratio: 0.3,
    });
    let motor = Some(Motor {
        speed: 4.0,
        max_force: 5.0,
    });
    let kinds = [
        JointKind::Ball,
        JointKind::Hinge(HingeOptions {
            limit: Some((-0.3, 0.6)),
            motor,
            spring,
        }),
        JointKind::Slider(SliderOptions {
            limit: Some((-0.1, 0.2)),
            motor,
            spring,
        }),
        JointKind::Fixed,
        JointKind::Distance(DistanceOptions {
            length: 1.0,
            min: 0.8,
            max: 1.2,
            spring,
        }),
    ];
    for kind in kinds {
        let mut w = PhysicsWorld::new(Settings {
            gravity: Vec3::ZERO,
            enable_sleep: false,
            ..Default::default()
        });
        let (pa, pb, ra, rb) = (
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.6, 0.1, 0.0),
            0.3,
            0.2,
        );
        let a = dynamic_with(
            &mut w,
            Shape::Sphere { radius: ra },
            BodyDesc {
                position: pa,
                linear_velocity: Vec3::new(0.3, 1.0, -0.2),
                angular_velocity: Vec3::new(0.5, -1.0, 2.0),
                angular_damping: 0.0,
                ..Default::default()
            },
        );
        let b = dynamic_with(
            &mut w,
            Shape::Sphere { radius: rb },
            BodyDesc {
                position: pb,
                linear_velocity: Vec3::new(-0.5, -0.5, 0.4),
                angular_velocity: Vec3::new(-1.0, 0.3, 0.8),
                angular_damping: 0.0,
                ..Default::default()
            },
        );
        // The frames meet at a point between the balls, axes tilted off the world's; a distance
        // joint holds the centres instead.
        let (frame_a, frame_b) = if let JointKind::Distance(_) = kind {
            (Iso::IDENTITY, Iso::IDENTITY)
        } else {
            let meet = Vec3::new(0.05, 0.02, 0.01);
            let q = Quat::from_rotation_arc(Vec3::X, Vec3::new(1.0, 2.0, -0.5).normalize_or_zero());
            (Iso::new(meet - pa, q), Iso::new(meet - pb, q))
        };
        join(&mut w, a, b, frame_a, frame_b, kind);
        let pair = [(a, ra), (b, rb)];
        let (p0, l0) = momentum(&w, &pair);
        for _ in 0..steps(2.0) {
            w.step(DT);
        }
        let (p1, l1) = momentum(&w, &pair);
        assert!(
            (p1 - p0).length() < 1e-4 * p0.length(),
            "{kind:?}: linear momentum {p0:?} became {p1:?}"
        );
        assert!(
            (l1 - l0).length() < 1e-3 * l0.length(),
            "{kind:?}: angular momentum {l0:?} became {l1:?}"
        );
        // And the joint really worked: the bodies did not just fly apart.
        let apart = w
            .center_of_mass(a)
            .unwrap()
            .distance(w.center_of_mass(b).unwrap());
        assert!(apart < 1.5, "{kind:?}: {apart}");
    }
}

#[test]
fn joints_respect_locked_axes() {
    let mut w = world();
    // A planar pendulum: the bob may only move in the XY plane and turn about Z. Its anchor is
    // 0.2 m off the pivot along Z, which the joint cannot close and must leave alone.
    let pivot = Vec3::new(0.0, 3.0, 0.2);
    let top = fixed_body(&mut w, pivot);
    let bob_at = Vec3::new(0.6, 2.2, 0.0);
    let bob = dynamic_with(
        &mut w,
        Shape::Sphere { radius: 0.1 },
        BodyDesc {
            position: bob_at,
            lock_translation: [false, false, true],
            lock_rotation: [true, true, false],
            ..Default::default()
        },
    );
    let swing = join(
        &mut w,
        top,
        bob,
        Iso::IDENTITY,
        Iso::from_position(Vec3::new(-0.6, 0.8, 0.0)),
        JointKind::Ball,
    );
    // A crank: a wheel that may turn but not move, with a weight hung off its rim by a hinge; the
    // weight turns the wheel as it falls.
    let hub = Vec3::new(5.0, 3.0, 0.0);
    let wheel = dynamic_with(
        &mut w,
        Shape::Sphere { radius: 0.3 },
        BodyDesc {
            position: hub,
            lock_translation: [true; 3],
            ..Default::default()
        },
    );
    let weight = dynamic(
        &mut w,
        Shape::Sphere { radius: 0.1 },
        hub + Vec3::new(0.5, 0.0, 0.0),
    );
    let rim = join(
        &mut w,
        wheel,
        weight,
        axis_frame(Vec3::new(0.5, 0.0, 0.0), Vec3::Z),
        axis_frame(Vec3::ZERO, Vec3::Z),
        JointKind::Hinge(HingeOptions::default()),
    );
    let mut swept = (f32::INFINITY, f32::NEG_INFINITY);
    for _ in 0..steps(3.0) {
        w.step(DT);
        let at = w.center_of_mass(bob).unwrap();
        assert_eq!(at.z, 0.0, "the bob stays in its plane");
        let q = w.body_pose(bob).unwrap().rotation;
        assert_eq!((q.x, q.y), (0.0, 0.0), "and turns about Z only");
        swept = (swept.0.min(at.x), swept.1.max(at.x));
        assert_eq!(
            w.center_of_mass(wheel).unwrap(),
            hub,
            "the wheel does not move"
        );
    }
    assert!(swept.0 < -0.5 && swept.1 > 0.5, "the bob swings: {swept:?}");
    let off_plane = anchor(&w, top, &swing.frame_a) - anchor(&w, bob, &swing.frame_b);
    let in_plane = Vec3::new(off_plane.x, off_plane.y, 0.0).length();
    assert!(
        in_plane < 1e-3,
        "the joint holds in the plane: {off_plane:?}"
    );
    assert!(
        (off_plane.z - 0.2).abs() < 1e-6,
        "and leaves Z alone: {off_plane:?}"
    );
    assert!(
        gap(&w, &rim) < 1e-3,
        "the rim joint holds: {}",
        gap(&w, &rim)
    );
    let turned = w.angular_velocity(wheel).unwrap();
    assert!(
        turned.length() > 0.1,
        "the weight turns the wheel: {turned:?}"
    );
}

/// A scene with every kind of joint and every option, already moving.
fn every_joint() -> PhysicsWorld {
    let mut w = world();
    let top = fixed_body(&mut w, Vec3::new(0.0, 10.0, 0.0));
    let mut prev = top;
    for i in 0..3 {
        let link = dynamic(
            &mut w,
            bar(0.25),
            Vec3::new(0.25 + 0.5 * i as f32, 10.0, 0.0),
        );
        let fa = if i == 0 {
            Iso::IDENTITY
        } else {
            Iso::from_position(Vec3::new(0.25, 0.0, 0.0))
        };
        join(
            &mut w,
            prev,
            link,
            fa,
            Iso::from_position(Vec3::new(-0.25, 0.0, 0.0)),
            JointKind::Ball,
        );
        prev = link;
    }
    let door = dynamic(&mut w, bar(0.5), Vec3::new(3.5, 10.0, 0.0));
    join(
        &mut w,
        top,
        door,
        axis_frame(Vec3::new(3.0, 0.0, 0.0), Vec3::new(0.0, 0.6, 0.8)),
        axis_frame(Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.0, 0.6, 0.8)),
        JointKind::Hinge(HingeOptions {
            limit: Some((-1.0, 0.5)),
            motor: Some(Motor {
                speed: -2.0,
                max_force: 30.0,
            }),
            spring: Some(Spring {
                hertz: 1.5,
                damping_ratio: 0.2,
            }),
        }),
    );
    let drawer = dynamic(&mut w, bar(0.3), Vec3::new(-3.0, 10.0, 0.0));
    join(
        &mut w,
        top,
        drawer,
        axis_frame(Vec3::new(-3.0, 0.0, 0.0), Vec3::new(1.0, -1.0, 0.3)),
        axis_frame(Vec3::ZERO, Vec3::new(1.0, -1.0, 0.3)),
        JointKind::Slider(SliderOptions {
            limit: Some((-0.4, 0.7)),
            motor: Some(Motor {
                speed: 0.5,
                max_force: 40.0,
            }),
            spring: Some(Spring {
                hertz: 2.0,
                damping_ratio: 0.5,
            }),
        }),
    );
    let lever = dynamic(&mut w, bar(0.4), Vec3::new(6.4, 10.0, 0.0));
    join(
        &mut w,
        top,
        lever,
        Iso::from_position(Vec3::new(6.0, 0.0, 0.0)),
        Iso::from_position(Vec3::new(-0.4, 0.0, 0.0)),
        JointKind::Fixed,
    );
    let weight = dynamic(
        &mut w,
        Shape::Sphere { radius: 0.2 },
        Vec3::new(-6.0, 9.0, 0.5),
    );
    join(
        &mut w,
        top,
        weight,
        Iso::from_position(Vec3::new(-6.0, 0.0, 0.0)),
        Iso::IDENTITY,
        JointKind::Distance(DistanceOptions {
            length: 1.2,
            min: 0.9,
            max: 1.3,
            spring: Some(Spring {
                hertz: 2.5,
                damping_ratio: 0.1,
            }),
        }),
    );
    // Two dynamic bodies joined to each other, spinning.
    let a = dynamic(
        &mut w,
        Shape::Sphere { radius: 0.2 },
        Vec3::new(9.0, 10.0, 0.0),
    );
    let b = dynamic(
        &mut w,
        Shape::Sphere { radius: 0.2 },
        Vec3::new(9.6, 10.0, 0.0),
    );
    join(
        &mut w,
        top,
        a,
        Iso::from_position(Vec3::new(9.0, 0.5, 0.0)),
        Iso::from_position(Vec3::new(0.0, 0.5, 0.0)),
        JointKind::Ball,
    );
    join(
        &mut w,
        a,
        b,
        Iso::from_position(Vec3::new(0.3, 0.0, 0.0)),
        Iso::from_position(Vec3::new(-0.3, 0.0, 0.0)),
        JointKind::Fixed,
    );
    w.set_angular_velocity(a, Vec3::new(0.3, 1.0, -0.4));
    w
}

#[test]
fn identical_runs_give_identical_state_hashes() {
    let (mut a, mut b) = (every_joint(), every_joint());
    let start = a.state_hash();
    for _ in 0..steps(4.0) {
        a.step(DT);
        b.step(DT);
        assert_eq!(a.state_hash(), b.state_hash());
    }
    assert_ne!(a.state_hash(), start, "the scene moved");
    for id in a.bodies() {
        let pose = a.body_pose(id).unwrap();
        assert!(pose.position.is_finite() && pose.rotation.is_finite());
    }
}

#[test]
fn degenerate_joints_neither_panic_nor_produce_non_finite_states() {
    let mut w = world();
    let ground = fixed_body(&mut w, Vec3::ZERO);
    let other_static = fixed_body(&mut w, Vec3::new(1.0, 0.0, 0.0));
    let kinematic = w
        .create_body(&BodyDesc {
            kind: BodyKind::Kinematic,
            position: Vec3::new(0.0, 2.0, 0.0),
            linear_velocity: Vec3::new(0.5, 0.0, 0.0),
            angular_velocity: Vec3::new(0.0, 1.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    let ball = |w: &mut PhysicsWorld, at: Vec3| dynamic(w, Shape::Sphere { radius: 0.1 }, at);
    let (p, q) = (
        ball(&mut w, Vec3::new(0.0, 1.0, 0.0)),
        ball(&mut w, Vec3::new(0.0, 1.0, 0.0)),
    );
    let r = ball(&mut w, Vec3::new(0.0, 1.0, 0.0));
    let locked = dynamic_with(
        &mut w,
        Shape::Sphere { radius: 0.1 },
        BodyDesc {
            position: Vec3::new(0.0, 1.0, 0.0),
            lock_translation: [true; 3],
            lock_rotation: [true; 3],
            ..Default::default()
        },
    );
    let far = ball(&mut w, Vec3::new(2.0, 1.0, 0.0));
    let zero = Quat::new(0.0, 0.0, 0.0, 0.0);
    let odd = Iso::new(Vec3::new(0.0, 0.0, 0.0), Quat::new(0.0, 3.0, 0.0, 4.0));
    let cases: Vec<(BodyId, BodyId, Iso, Iso, JointKind)> = vec![
        // Two coincident bodies on coincident anchors.
        (p, q, Iso::IDENTITY, Iso::IDENTITY, JointKind::Ball),
        (p, q, Iso::IDENTITY, Iso::IDENTITY, JointKind::Fixed),
        // Zero-length distances and ranges, upside-down ranges, negative lengths.
        (
            p,
            r,
            Iso::IDENTITY,
            Iso::IDENTITY,
            JointKind::Distance(DistanceOptions {
                length: 0.0,
                min: 0.0,
                max: 0.0,
                spring: None,
            }),
        ),
        (
            q,
            r,
            Iso::IDENTITY,
            Iso::IDENTITY,
            JointKind::Distance(DistanceOptions {
                length: -1.0,
                min: 2.0,
                max: 1.0,
                spring: Some(Spring {
                    hertz: 0.0,
                    damping_ratio: -1.0,
                }),
            }),
        ),
        // A frame without a rotation, limits past ±π and upside down, a negative torque limit,
        // a negative spring, an unnormalized frame.
        (
            ground,
            p,
            Iso::new(Vec3::new(0.0, 1.0, 0.0), zero),
            Iso::new(Vec3::ZERO, zero),
            JointKind::Hinge(HingeOptions {
                limit: Some((5.0, -7.0)),
                motor: Some(Motor {
                    speed: 3.0,
                    max_force: -1.0,
                }),
                spring: Some(Spring {
                    hertz: -2.0,
                    damping_ratio: 1.0,
                }),
            }),
        ),
        (
            kinematic,
            q,
            odd,
            odd,
            JointKind::Slider(SliderOptions {
                limit: Some((0.0, 0.0)),
                motor: Some(Motor {
                    speed: 1.0,
                    max_force: f32::MAX,
                }),
                spring: Some(Spring {
                    hertz: 1e30,
                    damping_ratio: 1e30,
                }),
            }),
        ),
        // Frames absurdly far from their bodies (their squares still finite).
        (
            ground,
            far,
            Iso::from_position(Vec3::new(1e18, -3e17, 0.0)),
            Iso::from_position(Vec3::new(0.0, 0.0, -1e18)),
            JointKind::Ball,
        ),
        (
            far,
            r,
            Iso::from_position(Vec3::splat(4e17)),
            Iso::IDENTITY,
            JointKind::Slider(SliderOptions {
                limit: Some((-1.0, 1.0)),
                ..Default::default()
            }),
        ),
        // A body that cannot move at all, and joints between bodies that never move.
        (
            ground,
            locked,
            Iso::IDENTITY,
            Iso::IDENTITY,
            JointKind::Fixed,
        ),
        (
            locked,
            r,
            Iso::IDENTITY,
            Iso::IDENTITY,
            JointKind::Hinge(HingeOptions::default()),
        ),
        (
            ground,
            other_static,
            Iso::IDENTITY,
            Iso::IDENTITY,
            JointKind::Fixed,
        ),
        (
            ground,
            kinematic,
            Iso::IDENTITY,
            Iso::IDENTITY,
            JointKind::Ball,
        ),
    ];
    for (a, b, fa, fb, kind) in cases {
        join(&mut w, a, b, fa, fb, kind);
    }
    for _ in 0..steps(2.0) {
        w.step(DT);
        for id in w.bodies() {
            let pose = w.body_pose(id).unwrap();
            let (v, s) = (
                w.linear_velocity(id).unwrap(),
                w.angular_velocity(id).unwrap(),
            );
            assert!(
                pose.position.is_finite()
                    && pose.rotation.is_finite()
                    && v.is_finite()
                    && s.is_finite(),
                "{id:?}: {pose:?} {v:?} {s:?}"
            );
        }
    }
    assert_eq!(w.center_of_mass(locked).unwrap(), Vec3::new(0.0, 1.0, 0.0));
}
