//! A joint whose bodies do not collide turns off their solid contacts, never a sensor's, both
//! at every step and when the joint is created; and the filter finds the joint whichever body
//! has the longer joint list. Every scene here is synthetic.

use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, HingeOptions, Iso, JointDesc, JointKind,
    PhysicsWorld, Quat, SensorEvent, Settings, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;

/// A static frame carrying a sensor ball, and a door (a dynamic slab) inside it.
fn frame_and_door() -> (PhysicsWorld, BodyId, BodyId) {
    let mut w = PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let frame = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            ..Default::default()
        })
        .unwrap();
    let mut zone = ColliderDesc::new(Shape::Sphere { radius: 0.6 });
    zone.local = Iso::from_position(Vec3::new(0.5, 1.0, 0.0));
    zone.sensor = true;
    w.add_collider(frame, &zone).unwrap();
    let door = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.5, 1.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        door,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(0.5, 1.0, 0.05),
        }),
    )
    .unwrap();
    (w, frame, door)
}

/// A vertical hinge at the door's edge (x = 0), its bodies not colliding.
fn hinge(w: &PhysicsWorld, frame: BodyId, door: BodyId) -> JointDesc {
    let frame_b = Iso::new(
        Vec3::new(-0.5, 0.0, 0.0),
        Quat::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_2),
    );
    JointDesc {
        body_a: frame,
        body_b: door,
        frame_a: w.body_pose(door).unwrap().mul(&frame_b),
        frame_b,
        kind: JointKind::Hinge(HingeOptions::default()),
        collide_connected: false,
        user_data: 0,
    }
}

/// `should_collide` keeps sensor pairs between jointed bodies, but `create_joint` dropped them:
/// the sensor reported an End while the door was still inside, and nothing after it (review
/// finding J8).
#[test]
fn a_joint_created_while_a_sensor_sees_its_body_keeps_the_sensor_pair() {
    let (mut w, frame, door) = frame_and_door();
    w.step(DT);
    let begins = |w: &PhysicsWorld| {
        w.events()
            .sensors
            .iter()
            .filter(|e| matches!(e, SensorEvent::Begin { .. }))
            .count()
    };
    assert_eq!(begins(&w), 1, "the sensor sees the door");
    let joint = hinge(&w, frame, door);
    w.create_joint(&joint).unwrap();
    // Two seconds of the door swinging slowly, well inside the sensor.
    w.set_angular_velocity(door, Vec3::new(0.0, 0.05, 0.0));
    w.set_linear_velocity(door, Vec3::new(0.0, 0.0, -0.025));
    let mut events = Vec::new();
    for _ in 0..120 {
        w.step(DT);
        events.extend(w.events().sensors.iter().copied());
    }
    assert!(
        events.is_empty(),
        "the door never left the sensor, which reported {events:?}"
    );
}

/// The joint still drops solid contacts at once: a door sunk into its frame's post stops
/// touching it when hinged to the frame.
#[test]
fn a_joint_created_between_touching_bodies_drops_their_solid_contacts() {
    let (mut w, frame, door) = frame_and_door();
    let mut post = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::new(0.05, 1.0, 0.05),
    });
    post.local = Iso::from_position(Vec3::new(-0.04, 1.0, 0.0));
    w.add_collider(frame, &post).unwrap();
    w.step(DT);
    assert_eq!(w.contact_count(), 1, "the door touches the post");
    w.create_joint(&hinge(&w, frame, door)).unwrap();
    assert_eq!(
        w.contact_count(),
        0,
        "the hinged door still touches the post"
    );
    for _ in 0..60 {
        w.step(DT);
    }
    assert_eq!(w.contact_count(), 0);
}

/// The filter searches the shorter of the two bodies' joint lists (J5); either way it finds
/// the joint that turns the pair off, whichever body was created first and whichever has the
/// longer list.
#[test]
fn the_joint_filter_finds_the_joint_from_either_body() {
    for level_first in [true, false] {
        for (level_extra, crate_extra) in [(0, 0), (30, 0), (0, 30), (30, 50), (50, 30)] {
            let mut w = PhysicsWorld::default();
            let make_level = |w: &mut PhysicsWorld| {
                let level = w
                    .create_body(&BodyDesc {
                        kind: BodyKind::Static,
                        position: Vec3::new(0.0, -0.5, 0.0),
                        ..Default::default()
                    })
                    .unwrap();
                w.add_collider(
                    level,
                    &ColliderDesc::new(Shape::Box {
                        half_extents: Vec3::new(20.0, 0.5, 20.0),
                    }),
                )
                .unwrap();
                level
            };
            let make_crate = |w: &mut PhysicsWorld, x: f32| {
                let b = w
                    .create_body(&BodyDesc {
                        position: Vec3::new(x, 0.25, 0.0),
                        ..Default::default()
                    })
                    .unwrap();
                w.add_collider(
                    b,
                    &ColliderDesc::new(Shape::Box {
                        half_extents: Vec3::splat(0.25),
                    }),
                )
                .unwrap();
                b
            };
            let (level, held) = if level_first {
                let l = make_level(&mut w);
                (l, make_crate(&mut w, 0.0))
            } else {
                let c = make_crate(&mut w, 0.0);
                (make_level(&mut w), c)
            };
            let free = make_crate(&mut w, 3.0);
            let anchor = w
                .create_body(&BodyDesc {
                    kind: BodyKind::Static,
                    position: Vec3::new(0.0, 10.0, 0.0),
                    ..Default::default()
                })
                .unwrap();
            let rope = |w: &mut PhysicsWorld, a: BodyId, b: BodyId, collide: bool| {
                w.create_joint(&JointDesc {
                    body_a: a,
                    body_b: b,
                    frame_a: Iso::IDENTITY,
                    frame_b: Iso::IDENTITY,
                    kind: JointKind::Distance(ganbate_physics::DistanceOptions {
                        length: 100.0,
                        min: 0.0,
                        max: 100.0,
                        spring: None,
                    }),
                    collide_connected: collide,
                    user_data: 0,
                })
                .unwrap();
            };
            for _ in 0..level_extra {
                let lamp = w.create_body(&BodyDesc::default()).unwrap();
                rope(&mut w, level, lamp, true);
            }
            for _ in 0..crate_extra {
                rope(&mut w, held, anchor, true);
            }
            // The crate held to the level does not collide with it: it sinks through.
            rope(&mut w, level, held, false);
            for _ in 0..60 {
                w.step(DT);
            }
            let (held_y, free_y) = (
                w.body_pose(held).unwrap().position.y,
                w.body_pose(free).unwrap().position.y,
            );
            let case =
                format!("level first {level_first}, extra joints {level_extra}/{crate_extra}");
            assert!(held_y < -1.0, "{case}: the held crate rests at {held_y}");
            assert!(
                (free_y - 0.25).abs() < 0.01,
                "{case}: the free crate is at {free_y}"
            );
        }
    }
}
