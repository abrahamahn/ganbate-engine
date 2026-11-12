//! Continuous collision between a translating kinematic obstacle and a dynamic body.
//! Every scene is synthetic and uses the normal world step.
use ganbate_physics::{
    BodyDesc, BodyKind, ColliderDesc, ContactEvent, JointDesc, JointKind, PhysicsWorld, Quat,
    Settings, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;
const BALL: Shape = Shape::Sphere { radius: 0.15 };

fn wall() -> Shape {
    Shape::Box {
        half_extents: Vec3::new(0.04, 1.0, 1.0),
    }
}
fn world(continuous: bool) -> PhysicsWorld {
    PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        enable_continuous: continuous,
        ..Default::default()
    })
}
fn body(
    w: &mut PhysicsWorld,
    kind: BodyKind,
    x: f32,
    speed: f32,
    col: ColliderDesc,
    awake: bool,
) -> ganbate_physics::BodyId {
    let id = w
        .create_body(&BodyDesc {
            kind,
            position: Vec3::X * x,
            linear_velocity: Vec3::X * speed,
            bullet: kind == BodyKind::Dynamic && speed != 0.0,
            awake,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(id, &col).unwrap();
    id
}

#[test]
fn a_bullet_and_wall_moving_together_do_not_hit_the_walls_end_pose() {
    let mut w = world(true);
    body(
        &mut w,
        BodyKind::Kinematic,
        -1.0,
        120.0,
        ColliderDesc::new(wall()),
        true,
    );
    let ball = body(
        &mut w,
        BodyKind::Dynamic,
        0.0,
        120.0,
        ColliderDesc::new(BALL),
        true,
    );
    w.step(DT);
    assert!((w.body_pose(ball).unwrap().position.x - 2.0).abs() < 1e-5);
    assert!(
        !w.events()
            .contacts
            .iter()
            .any(|e| matches!(e, ContactEvent::Hit { .. }))
    );
}

#[test]
fn a_head_on_bullet_stays_in_front_of_the_translating_wall() {
    let mut w = world(true);
    let wall = body(
        &mut w,
        BodyKind::Kinematic,
        3.0,
        -240.0,
        ColliderDesc::new(wall()),
        true,
    );
    let ball = body(
        &mut w,
        BodyKind::Dynamic,
        0.0,
        120.0,
        ColliderDesc::new(BALL),
        true,
    );
    w.step(DT);
    let x = w.body_pose(ball).unwrap().position.x;
    assert!(
        (x - (w.body_pose(wall).unwrap().position.x - 0.19)).abs() < 0.01,
        "bullet {x}"
    );
    assert!((w.linear_velocity(ball).unwrap().x + 240.0).abs() < 0.01);
}

#[test]
fn a_sleeping_ball_is_woken_and_pushed_only_along_the_face_normal() {
    let mut w = world(true);
    let wall = body(
        &mut w,
        BodyKind::Kinematic,
        -1.0,
        120.0,
        ColliderDesc::new(wall()),
        true,
    );
    w.set_linear_velocity(wall, Vec3::new(120.0, 12.0, 0.0));
    let ball = body(
        &mut w,
        BodyKind::Dynamic,
        0.0,
        0.0,
        ColliderDesc::new(BALL),
        false,
    );
    w.step(DT);
    let p = w.body_pose(ball).unwrap().position;
    assert!(w.is_awake(ball));
    assert!((p.x - 1.19).abs() < 0.01 && p.y.abs() < 1e-5, "{p:?}");
    let v = w.linear_velocity(ball).unwrap();
    assert!((v.x - 120.0).abs() < 0.01 && v.y.abs() < 1e-4, "{v:?}");
}

#[test]
fn filters_and_disabling_ccd_leave_an_unhit_sleeper_as_it_was() {
    for (continuous, sensor, mask) in [
        (false, false, u32::MAX),
        (true, true, u32::MAX),
        (true, false, 0),
    ] {
        let mut w = world(continuous);
        let mut wc = ColliderDesc::new(wall());
        wc.sensor = sensor;
        wc.mask = mask;
        body(&mut w, BodyKind::Kinematic, -1.0, 120.0, wc, true);
        let ball = body(
            &mut w,
            BodyKind::Dynamic,
            0.0,
            0.0,
            ColliderDesc::new(BALL),
            false,
        );
        w.step(DT);
        assert!(!w.is_awake(ball));
        assert_eq!(w.body_pose(ball).unwrap().position, Vec3::ZERO);
        assert!(
            !w.events()
                .contacts
                .iter()
                .any(|e| matches!(e, ContactEvent::Hit { .. }))
        );
    }
}

#[test]
fn a_moving_wall_cannot_push_a_ball_through_fixed_scenery() {
    let mut w = world(true);
    body(
        &mut w,
        BodyKind::Static,
        0.65,
        0.0,
        ColliderDesc::new(wall()),
        true,
    );
    body(
        &mut w,
        BodyKind::Kinematic,
        -1.0,
        120.0,
        ColliderDesc::new(wall()),
        true,
    );
    let ball = body(
        &mut w,
        BodyKind::Dynamic,
        0.0,
        0.0,
        ColliderDesc::new(BALL),
        false,
    );
    w.step(DT);
    let x = w.body_pose(ball).unwrap().position.x;
    assert!((0.44..=0.49).contains(&x), "ball {x}");
}

#[test]
fn a_joint_that_disables_collision_also_excludes_the_translating_wall() {
    let mut w = world(true);
    let wall = body(
        &mut w,
        BodyKind::Kinematic,
        -1.0,
        120.0,
        ColliderDesc::new(wall()),
        true,
    );
    let ball = body(
        &mut w,
        BodyKind::Dynamic,
        0.0,
        0.0,
        ColliderDesc::new(BALL),
        true,
    );
    w.create_joint(&JointDesc {
        body_a: wall,
        body_b: ball,
        frame_a: ganbate_physics::Iso::IDENTITY,
        frame_b: ganbate_physics::Iso::IDENTITY,
        kind: JointKind::Distance(ganbate_physics::joint::DistanceOptions {
            length: 1.0,
            min: 0.0,
            max: 10.0,
            spring: None,
        }),
        collide_connected: false,
        user_data: 0,
    })
    .unwrap();
    w.step(DT);
    assert!(w.body_pose(ball).unwrap().position.x.abs() < 0.01);
}

#[test]
fn two_translating_faces_in_a_corner_can_each_push_the_ball() {
    let mut w = world(true);
    body(
        &mut w,
        BodyKind::Kinematic,
        -1.0,
        120.0,
        ColliderDesc::new(wall()),
        true,
    );
    let from_below = w
        .create_body(&BodyDesc {
            kind: BodyKind::Kinematic,
            position: -Vec3::Y,
            linear_velocity: Vec3::Y * 120.0,
            ..Default::default()
        })
        .unwrap();
    // Wide enough to cover the ball's whole path after the other face pushes it sideways.
    w.add_collider(
        from_below,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(2.0, 0.04, 2.0),
        }),
    )
    .unwrap();
    let ball = body(
        &mut w,
        BodyKind::Dynamic,
        0.0,
        0.0,
        ColliderDesc::new(BALL),
        false,
    );
    w.step(DT);
    let p = w.body_pose(ball).unwrap().position;
    assert!(p.x > 1.13 && p.y > 1.13, "{p:?}");
}

#[test]
fn a_sweep_cannot_move_a_body_along_a_locked_axis() {
    let mut w = world(true);
    body(
        &mut w,
        BodyKind::Kinematic,
        -1.0,
        120.0,
        ColliderDesc::new(wall()),
        true,
    );
    let ball = w
        .create_body(&BodyDesc {
            lock_translation: [true, false, false],
            awake: false,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(ball, &ColliderDesc::new(BALL)).unwrap();
    w.step(DT);
    assert_eq!(w.body_pose(ball).unwrap().position, Vec3::ZERO);
    assert_eq!(w.linear_velocity(ball).unwrap(), Vec3::ZERO);
    assert!(!w.is_awake(ball));
}

fn wide_moving_face(w: &mut PhysicsWorld, rotation: Quat, speed: f32) {
    let normal = rotation.rotate(Vec3::X);
    let face = w
        .create_body(&BodyDesc {
            kind: BodyKind::Kinematic,
            position: -normal,
            rotation,
            linear_velocity: normal * speed,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        face,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(0.04, 100.0, 100.0),
        }),
    )
    .unwrap();
}

#[test]
fn a_wall_respects_the_speed_cap_without_losing_its_normal_push() {
    let mut w = PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        max_linear_speed: 120.0,
        ..Default::default()
    });
    wide_moving_face(&mut w, Quat::IDENTITY, 100.0);
    let ball = w
        .create_body(&BodyDesc {
            linear_velocity: Vec3::Y * 99.0,
            bullet: true,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(ball, &ColliderDesc::new(BALL)).unwrap();
    w.step(DT);
    let p = w.body_pose(ball).unwrap().position;
    let v = w.linear_velocity(ball).unwrap();
    assert!(p.x >= 100.0 * DT - 1.0 + 0.19);
    assert!(v.x >= 99.99 && v.y > 0.0);
    assert!(p.length() <= 120.0 * DT + 1e-4 && v.length() <= 120.0001);
}

#[test]
fn a_nearly_blocked_sweep_cannot_launch_a_slider_along_its_unlocked_axis() {
    let mut w = PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        max_linear_speed: 120.0,
        ..Default::default()
    });
    wide_moving_face(&mut w, Quat::from_axis_angle(Vec3::Z, 0.001), 100.0);
    let ball = w
        .create_body(&BodyDesc {
            awake: false,
            lock_translation: [true, false, true],
            ..Default::default()
        })
        .unwrap();
    w.add_collider(ball, &ColliderDesc::new(BALL)).unwrap();
    w.step(DT);
    let p = w.body_pose(ball).unwrap().position;
    let v = w.linear_velocity(ball).unwrap();
    assert_eq!((p.x, p.z, v.x, v.z), (0.0, 0.0, 0.0, 0.0));
    assert!(p.length() <= 120.0 * DT + 1e-4 && v.length() <= 120.0001);
}

#[test]
fn simultaneous_oblique_planes_carry_a_body_through_a_feasible_corner() {
    for angle in [120.0_f32, 135.0] {
        for order in [[0.0, angle], [angle, 0.0]] {
            let mut w = world(true);
            for degrees in order {
                wide_moving_face(
                    &mut w,
                    Quat::from_axis_angle(Vec3::Z, degrees.to_radians()),
                    120.0,
                );
            }
            let ball = body(
                &mut w,
                BodyKind::Dynamic,
                0.0,
                0.0,
                ColliderDesc::new(BALL),
                false,
            );
            w.step(DT);
            let p = w.body_pose(ball).unwrap().position;
            for degrees in order {
                let n = Quat::from_axis_angle(Vec3::Z, degrees.to_radians()).rotate(Vec3::X);
                assert!(p.dot(n) - 1.0 >= 0.18, "angle {angle}, pose {p:?}");
            }
        }
    }
}

#[test]
fn a_body_can_follow_three_oblique_translating_faces_at_once() {
    let mut w = world(true);
    let planes = [
        Quat::IDENTITY,
        Quat::from_axis_angle(Vec3::Z, 1.9),
        Quat::from_axis_angle(Vec3::Y, -1.8),
    ];
    for q in planes {
        wide_moving_face(&mut w, q, 100.0);
    }
    let ball = body(
        &mut w,
        BodyKind::Dynamic,
        0.0,
        0.0,
        ColliderDesc::new(BALL),
        false,
    );
    w.step(DT);
    let p = w.body_pose(ball).unwrap().position;
    for q in planes {
        let n = q.rotate(Vec3::X);
        assert!(p.dot(n) >= 100.0 * DT - 1.0 + 0.18, "pose {p:?}");
    }
}

#[test]
fn a_swept_hit_reports_its_world_witness_and_ordered_pair_once() {
    for awake in [false, true] {
        for wall_first in [false, true] {
            let mut w = world(true);
            let ball = |w: &mut PhysicsWorld| {
                let b = w
                    .create_body(&BodyDesc {
                        awake,
                        ..Default::default()
                    })
                    .unwrap();
                w.add_collider(b, &ColliderDesc::new(BALL)).unwrap()
            };
            let platform = |w: &mut PhysicsWorld| {
                let b = w
                    .create_body(&BodyDesc {
                        kind: BodyKind::Kinematic,
                        position: -Vec3::X,
                        linear_velocity: Vec3::X * 120.0,
                        ..Default::default()
                    })
                    .unwrap();
                w.add_collider(b, &ColliderDesc::new(wall())).unwrap()
            };
            let pair = if wall_first {
                (platform(&mut w), ball(&mut w))
            } else {
                (ball(&mut w), platform(&mut w))
            };
            w.step(DT);
            let hits: Vec<_> = w
                .events()
                .contacts
                .iter()
                .filter_map(|e| match *e {
                    ContactEvent::Hit {
                        a,
                        b,
                        normal,
                        point,
                        speed,
                    } => Some((a, b, normal, point, speed)),
                    _ => None,
                })
                .collect();
            assert_eq!(hits.len(), 1);
            let (a, b, normal, point, speed) = hits[0];
            assert_eq!((a, b), pair);
            let expected = if wall_first { Vec3::X } else { -Vec3::X };
            assert!(normal.dot(expected) > 0.99);
            assert!(
                (point.x + 0.155).abs() < 0.001 && point.y.abs() < 0.001 && point.z.abs() < 0.001,
                "witness {point:?}"
            );
            assert!((speed - 120.0).abs() < 0.001, "approach {speed}");
            w.step(DT);
            assert!(
                !w.events()
                    .contacts
                    .iter()
                    .any(|e| matches!(e, ContactEvent::Hit { .. }))
            );
        }
    }
}

#[test]
fn a_moving_obstacle_respects_the_hit_threshold_and_does_not_duplicate_a_discrete_hit() {
    for (threshold, start, expected) in [
        (f32::INFINITY, -1.0, 0),
        (120.0, -1.0, 0),
        (119.0, -1.0, 1),
        (1.0, -0.18, 1),
    ] {
        let mut w = PhysicsWorld::new(Settings {
            gravity: Vec3::ZERO,
            hit_event_threshold: threshold,
            ..Default::default()
        });
        body(
            &mut w,
            BodyKind::Kinematic,
            start,
            120.0,
            ColliderDesc::new(wall()),
            true,
        );
        body(
            &mut w,
            BodyKind::Dynamic,
            0.0,
            0.0,
            ColliderDesc::new(BALL),
            true,
        );
        w.step(DT);
        let hits = w
            .events()
            .contacts
            .iter()
            .filter(|e| matches!(e, ContactEvent::Hit { .. }))
            .count();
        assert_eq!(hits, expected, "threshold {threshold}, start {start}");
        if start == -0.18 {
            assert!(
                w.events()
                    .contacts
                    .iter()
                    .any(|e| matches!(e, ContactEvent::Begin { .. }))
            );
        }
    }
}
