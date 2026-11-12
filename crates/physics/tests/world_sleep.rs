//! Sleeping and waking: what keeps a body awake, what wakes it, and what must not. Every scene
//! here is synthetic. Scenes that rest bodies on others need the narrow phase's contacts; joints
//! do not.

use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, Iso, JointDesc, JointKind, PhysicsWorld, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;

fn run(w: &mut PhysicsWorld, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        w.step(DT);
    }
}

fn pos(w: &PhysicsWorld, b: BodyId) -> Vec3 {
    w.body_pose(b).unwrap().position
}

fn cube(w: &mut PhysicsWorld, position: Vec3, half: f32) -> BodyId {
    let b = w
        .create_body(&BodyDesc {
            position,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        b,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::splat(half),
        }),
    )
    .unwrap();
    b
}

/// A 6 × 0.5 × 6 m platform with its top at y = 0 and a 0.5 m box resting on it.
fn platform_scene(kind: BodyKind, can_sleep: bool) -> (PhysicsWorld, BodyId, BodyId) {
    let mut w = PhysicsWorld::default();
    let k = w
        .create_body(&BodyDesc {
            kind,
            position: Vec3::new(0.0, -0.25, 0.0),
            can_sleep,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        k,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(3.0, 0.25, 3.0),
        }),
    )
    .unwrap();
    let d = cube(&mut w, Vec3::new(0.0, 0.25, 0.0), 0.25);
    (w, k, d)
}

/// A kinematic body that stands still must not wake what rests on it, whether it never sleeps
/// (an elevator the game drives) or the game re-targets it to where it is every frame (review
/// finding KIN-STILL-WAKES).
#[test]
fn a_box_on_a_still_kinematic_platform_stays_asleep() {
    let awake_steps =
        |w: &mut PhysicsWorld, d: BodyId, mut each: Box<dyn FnMut(&mut PhysicsWorld)>| {
            run(w, 1.0);
            (0..300)
                .filter(|_| {
                    each(w);
                    w.step(DT);
                    w.is_awake(d)
                })
                .count()
        };
    let (mut w, _, d) = platform_scene(BodyKind::Static, true);
    let on_static = awake_steps(&mut w, d, Box::new(|_| {}));
    let (mut w, _, d) = platform_scene(BodyKind::Kinematic, false);
    let on_never_sleeping = awake_steps(&mut w, d, Box::new(|_| {}));
    let (mut w, k, d) = platform_scene(BodyKind::Kinematic, true);
    let here = w.body_pose(k).unwrap();
    let on_retargeted = awake_steps(
        &mut w,
        d,
        Box::new(move |w| w.set_kinematic_target(k, here, DT)),
    );
    assert_eq!(on_static, 0);
    assert!(
        on_never_sleeping < 30 && on_retargeted < 30,
        "steps awake out of 300: never-sleeping platform {on_never_sleeping}, re-targeted {on_retargeted}"
    );
    // And a platform that starts moving still carries the box along (friction takes about
    // 0.17 s to bring the box up to the platform's 1 m/s).
    let (mut w, k, d) = platform_scene(BodyKind::Kinematic, false);
    run(&mut w, 2.0);
    assert!(!w.is_awake(d));
    w.set_linear_velocity(k, Vec3::new(1.0, 0.0, 0.0));
    run(&mut w, 1.0);
    let v = w.linear_velocity(d).unwrap();
    assert!(
        pos(&w, d).x > 0.85 && (v.x - 1.0).abs() < 0.02,
        "the box rides along: at {:?}, moving at {v:?}",
        pos(&w, d)
    );
}

/// A kinematic arm and a lamp hanging 1 m below it from a ball joint.
fn hanging(speed: f32) -> (PhysicsWorld, BodyId, BodyId) {
    let mut w = PhysicsWorld::default();
    let k = w
        .create_body(&BodyDesc {
            kind: BodyKind::Kinematic,
            position: Vec3::new(0.0, 5.0, 0.0),
            linear_velocity: Vec3::new(speed, 0.0, 0.0),
            can_sleep: false,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        k,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::splat(0.1),
        }),
    )
    .unwrap();
    let d = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, 4.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(d, &ColliderDesc::new(Shape::Sphere { radius: 0.2 }))
        .unwrap();
    w.create_joint(&JointDesc {
        body_a: k,
        body_b: d,
        frame_a: Iso::from_position(Vec3::new(0.0, -1.0, 0.0)),
        frame_b: Iso::IDENTITY,
        kind: JointKind::Ball,
        collide_connected: false,
        user_data: 0,
    })
    .unwrap();
    (w, k, d)
}

fn joint_gap(w: &PhysicsWorld, k: BodyId, d: BodyId) -> f32 {
    (pos(w, k) + Vec3::new(0.0, -1.0, 0.0) - pos(w, d)).length()
}

/// A moving kinematic body wakes, and keeps awake, what hangs from it by a joint, as it does
/// what rests on it; a still one lets it sleep (review finding KIN-JOINT-SLEEP).
#[test]
fn a_lamp_hanging_from_a_kinematic_arm_follows_the_arm() {
    // The arm stands still: the lamp settles and sleeps. Then the arm moves.
    let (mut w, k, d) = hanging(0.0);
    run(&mut w, 2.0);
    assert!(!w.is_awake(d), "the lamp sleeps while the arm stands still");
    w.set_linear_velocity(k, Vec3::new(1.0, 0.0, 0.0));
    w.step(DT);
    assert!(w.is_awake(d), "the arm's first moving step wakes the lamp");
    run(&mut w, 2.0);
    let gap = joint_gap(&w, k, d);
    assert!(gap < 0.05, "the ball joint came apart by {gap} m");
    // An arm creeping slower than the sleep threshold keeps the lamp awake and with it.
    let (mut w, k, d) = hanging(0.03);
    run(&mut w, 20.0);
    let gap = joint_gap(&w, k, d);
    assert!(
        w.is_awake(d) && gap < 0.05,
        "creeping arm: lamp awake {}, joint stretched by {gap} m",
        w.is_awake(d)
    );
}

/// The same with gravity off: nothing pulls the lamp, only the arm's motion can wake it.
#[test]
fn a_body_jointed_to_an_awake_kinematic_body_wakes_when_it_moves() {
    let (mut w, k, d) = hanging(0.0);
    w.settings.gravity = Vec3::ZERO;
    run(&mut w, 1.0);
    assert!(!w.is_awake(d));
    w.set_linear_velocity(k, Vec3::new(2.0, 0.0, 0.0));
    run(&mut w, 1.0);
    assert!(
        w.is_awake(d),
        "the jointed body sleeps on while the arm moves away"
    );
    assert!(
        joint_gap(&w, k, d) < 0.05,
        "and follows it: {}",
        joint_gap(&w, k, d)
    );
}

/// Waking a body wakes the bodies it touches and theirs, but a kinematic body passes nothing on:
/// it is never pushed, so what rests on it elsewhere stays asleep (review finding
/// WAKE-THROUGH-KINEMATIC).
#[test]
fn waking_one_box_on_a_kinematic_platform_leaves_the_others_asleep() {
    let mut w = PhysicsWorld::default();
    let k = w
        .create_body(&BodyDesc {
            kind: BodyKind::Kinematic,
            position: Vec3::new(0.0, -0.25, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        k,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(10.0, 0.25, 3.0),
        }),
    )
    .unwrap();
    let a = cube(&mut w, Vec3::new(-8.0, 0.25, 0.0), 0.25);
    let a_top = cube(&mut w, Vec3::new(-8.0, 0.75, 0.0), 0.25);
    let b = cube(&mut w, Vec3::new(8.0, 0.25, 0.0), 0.25);
    run(&mut w, 2.0);
    assert!(!w.is_awake(a) && !w.is_awake(a_top) && !w.is_awake(b) && !w.is_awake(k));
    w.wake(a);
    assert!(
        w.is_awake(a) && w.is_awake(a_top),
        "a and the box on it wake"
    );
    assert!(
        !w.is_awake(b) && !w.is_awake(k),
        "a box 16 m away (awake {}) or the platform (awake {}) woke through the platform",
        w.is_awake(b),
        w.is_awake(k)
    );
}

/// Teleporting a static body must wake what slept on it: its pairs with sleeping bodies are never
/// updated, so the box floated where the platform had been, its contact still listed (review
/// finding STATIC-TELEPORT).
#[test]
fn moving_a_static_platform_drops_what_slept_on_it() {
    let (mut w, s, d) = platform_scene(BodyKind::Static, true);
    run(&mut w, 2.0);
    assert!(!w.is_awake(d), "precondition: the box sleeps");
    w.set_body_pose(s, Iso::from_position(Vec3::new(100.0, -0.25, 0.0)));
    run(&mut w, 2.0);
    assert!(
        pos(&w, d).y < -1.0 && w.contact_count() == 0,
        "the box at {:?}, contacts {}",
        pos(&w, d),
        w.contact_count()
    );
}

/// The same through a joint: a lamp asleep under a static hook follows the hook when it is
/// teleported.
#[test]
fn moving_a_static_hook_wakes_the_lamp_hanging_from_it() {
    let mut w = PhysicsWorld::default();
    let hook = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(0.0, 5.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    let lamp = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, 4.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(lamp, &ColliderDesc::new(Shape::Sphere { radius: 0.2 }))
        .unwrap();
    w.create_joint(&JointDesc {
        body_a: hook,
        body_b: lamp,
        frame_a: Iso::from_position(Vec3::new(0.0, -1.0, 0.0)),
        frame_b: Iso::IDENTITY,
        kind: JointKind::Ball,
        collide_connected: false,
        user_data: 0,
    })
    .unwrap();
    run(&mut w, 2.0);
    assert!(!w.is_awake(lamp), "precondition: the lamp sleeps");
    w.set_body_pose(hook, Iso::from_position(Vec3::new(0.0, 6.0, 0.0)));
    assert!(
        w.is_awake(lamp),
        "the lamp sleeps on under a hook that moved away"
    );
    run(&mut w, 2.0);
    let gap = (pos(&w, hook) + Vec3::new(0.0, -1.0, 0.0) - pos(&w, lamp)).length();
    assert!(gap < 0.05, "the joint is stretched by {gap} m");
}

/// A static body moved or grown into a sleeping body wakes it: it never wakes anything through
/// its contacts, which are not even updated while the other body sleeps. One moved or grown
/// elsewhere wakes nothing.
#[test]
fn static_geometry_moved_or_added_into_a_sleeping_body_wakes_it() {
    let mut w = PhysicsWorld::new(ganbate_physics::Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let d = cube(&mut w, Vec3::new(0.0, 1.0, 0.0), 0.5);
    let s = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(10.0, 0.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    let wall = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::new(0.1, 2.0, 2.0),
    });
    w.add_collider(s, &wall).unwrap();
    run(&mut w, 1.0);
    assert!(!w.is_awake(d), "precondition: the box sleeps");
    // Elsewhere: nothing wakes.
    w.set_body_pose(s, Iso::from_position(Vec3::new(-10.0, 0.0, 0.0)));
    let mut far = wall.clone();
    far.local = Iso::from_position(Vec3::new(0.0, 0.0, 5.0));
    w.add_collider(s, &far).unwrap();
    assert!(!w.is_awake(d), "geometry 10 m away woke the box");
    // Moved into the box.
    w.set_body_pose(s, Iso::from_position(Vec3::new(0.3, 1.0, 0.0)));
    assert!(
        w.is_awake(d),
        "a wall teleported into the box left it asleep"
    );
    // Grown into the box.
    w.set_body_pose(s, Iso::from_position(Vec3::new(-10.0, 0.0, 0.0)));
    run(&mut w, 1.0);
    assert!(!w.is_awake(d), "precondition: the box sleeps again");
    let mut into = wall;
    into.local = Iso::from_position(Vec3::new(10.3, 1.0, 0.0));
    w.add_collider(s, &into).unwrap();
    assert!(w.is_awake(d), "a wall added into the box left it asleep");
}

/// A sleeping body that loses a collider must wake: its mass and shape changed, and only
/// static geometry may be touching it, which wakes nothing (review finding
/// REMOVE-COLLIDER-SLEEP).
#[test]
fn a_sleeping_body_that_loses_a_collider_wakes() {
    let mut w = PhysicsWorld::new(ganbate_physics::Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let b = w.create_body(&BodyDesc::default()).unwrap();
    let mut leg = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::new(0.2, 0.25, 0.2),
    });
    leg.local = Iso::from_position(Vec3::new(0.0, -0.5, 0.0));
    let leg = w.add_collider(b, &leg).unwrap();
    let mut top = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::new(0.5, 0.25, 0.5),
    });
    top.local = Iso::from_position(Vec3::new(0.0, 0.5, 0.0));
    w.add_collider(b, &top).unwrap();
    run(&mut w, 1.0);
    assert!(!w.is_awake(b), "precondition: the table sleeps");
    w.remove_collider(leg);
    assert!(w.is_awake(b), "the table sleeps on without its leg");
}

/// The reviewer's table on the ground: with its leg gone it drops.
#[test]
fn a_sleeping_table_without_its_leg_drops_to_the_ground() {
    let mut w = PhysicsWorld::default();
    let g = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(0.0, -0.5, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        g,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(50.0, 0.5, 50.0),
        }),
    )
    .unwrap();
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, 0.75, 0.0),
            ..Default::default()
        })
        .unwrap();
    let mut leg = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::new(0.2, 0.25, 0.2),
    });
    leg.local = Iso::from_position(Vec3::new(0.0, -0.5, 0.0));
    let leg = w.add_collider(b, &leg).unwrap();
    let mut top = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::new(0.5, 0.25, 0.5),
    });
    top.local = Iso::from_position(Vec3::new(0.0, 0.5, 0.0));
    w.add_collider(b, &top).unwrap();
    run(&mut w, 3.0);
    assert!(!w.is_awake(b), "precondition: the table sleeps");
    w.remove_collider(leg);
    run(&mut w, 2.0);
    assert!(pos(&w, b).y < 0.5, "the table floats at {:?}", pos(&w, b));
}

/// A body created asleep stays asleep when it gets its colliders (a level's pre-settled stack),
/// until something wakes it (review finding AWAKE-FLAG).
#[test]
fn a_body_created_asleep_stays_asleep_until_woken() {
    let mut w = PhysicsWorld::default();
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, 10.0, 0.0),
            awake: false,
            ..Default::default()
        })
        .unwrap();
    assert!(!w.is_awake(b));
    w.add_collider(b, &ColliderDesc::new(Shape::Sphere { radius: 0.5 }))
        .unwrap();
    let mut second = ColliderDesc::new(Shape::Sphere { radius: 0.25 });
    second.local = Iso::from_position(Vec3::new(0.0, 0.6, 0.0));
    w.add_collider(b, &second).unwrap();
    assert!(
        !w.is_awake(b),
        "a body created asleep woke when it got a collider"
    );
    run(&mut w, 1.0);
    assert!(
        !w.is_awake(b) && pos(&w, b).y == 10.0,
        "it fell asleep: {:?}",
        pos(&w, b)
    );
    w.wake(b);
    run(&mut w, 1.0);
    assert!(pos(&w, b).y < 6.0, "woken, it falls: {:?}", pos(&w, b));
}

/// A body counts as still when its farthest point moves slower than the sleep threshold; that
/// distance is measured from its centre of mass and over its solid colliders. Measured from the
/// body's origin, a 0.5 m ball placed 10 m from it reached 10.5 m, and turning at 0.05 rad/s
/// (its surface at 2.5 cm/s) it never slept (review finding MAX-EXTENT-ORIGIN).
#[test]
fn a_slowly_turning_body_sleeps_however_far_its_shape_is_from_its_origin() {
    let ball_at = |w: &mut PhysicsWorld, b: BodyId, at: Vec3, sensor: bool| {
        let mut d = ColliderDesc::new(Shape::Sphere { radius: 0.5 });
        d.local = Iso::from_position(at);
        d.sensor = sensor;
        w.add_collider(b, &d).unwrap();
    };
    let mut w = PhysicsWorld::new(ganbate_physics::Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let spin = Vec3::new(0.0, 0.05, 0.0);
    // A ball 10 m from its body's origin.
    let offset = w
        .create_body(&BodyDesc {
            angular_velocity: spin,
            ..Default::default()
        })
        .unwrap();
    ball_at(&mut w, offset, Vec3::new(10.0, 0.0, 0.0), false);
    // A ball at its body's origin with a sensor 10 m away.
    let sensing = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, 0.0, 20.0),
            angular_velocity: spin,
            ..Default::default()
        })
        .unwrap();
    ball_at(&mut w, sensing, Vec3::ZERO, false);
    ball_at(&mut w, sensing, Vec3::new(10.0, 0.0, 0.0), true);
    // Control: a 10 m ball turning as fast moves its surface at 0.5 m/s and stays awake.
    let big = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, 0.0, -40.0),
            angular_velocity: spin,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(big, &ColliderDesc::new(Shape::Sphere { radius: 10.0 }))
        .unwrap();
    run(&mut w, 1.5);
    assert!(!w.is_awake(offset), "the offset ball never sleeps");
    assert!(!w.is_awake(sensing), "the sensor keeps its body awake");
    assert!(w.is_awake(big));
}
