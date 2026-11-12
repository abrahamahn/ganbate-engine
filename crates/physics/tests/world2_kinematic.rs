//! A kinematic body that has stopped and fallen asleep is like the ground: given a new collider
//! through a sleeping body, it must wake that body so the overlap is resolved (review finding:
//! a collider added to a sleeping kinematic body left a crate embedded in it for good). Every
//! scene here is synthetic.

use ganbate_physics::{BodyDesc, BodyKind, ColliderDesc, Iso, PhysicsWorld, Shape, Vec3};

const DT: f32 = 1.0 / 60.0;

#[test]
fn a_collider_added_to_a_sleeping_kinematic_body_pushes_out_what_it_overlaps() {
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
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        }),
    )
    .unwrap();
    let platform = w
        .create_body(&BodyDesc {
            kind: BodyKind::Kinematic,
            position: Vec3::new(3.0, 1.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        platform,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::splat(0.2),
        }),
    )
    .unwrap();
    let crate_ = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, 0.25, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        crate_,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::splat(0.25),
        }),
    )
    .unwrap();
    for _ in 0..120 {
        w.step(DT);
    }
    assert!(!w.is_awake(platform) && !w.is_awake(crate_), "both sleep");
    // A plank through the crate's lower half (y from 0.1 to 0.3).
    let mut plank = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::new(1.5, 0.1, 0.5),
    });
    plank.local = Iso::from_position(Vec3::new(-3.0, -0.8, 0.0));
    w.add_collider(platform, &plank).unwrap();
    for _ in 0..60 {
        w.step(DT);
    }
    let y = w.body_pose(crate_).unwrap().position.y;
    assert!(
        (y - 0.55).abs() < 0.01,
        "the crate is at y = {y}: it should rest on the plank (0.55)"
    );
    // The platform itself stays where it is, asleep or not: it is never pushed.
    let p = w.body_pose(platform).unwrap().position;
    assert_eq!(p, Vec3::new(3.0, 1.0, 0.0));
}

/// Adding a collider to an awake kinematic body near nothing wakes nothing and keeps it as it
/// was; adding one to a sleeping dynamic body still leaves it asleep (a level's pre-settled
/// stack is built that way).
#[test]
fn adding_colliders_wakes_only_what_must_react() {
    let mut w = PhysicsWorld::default();
    let sleeper = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, 5.0, 0.0),
            awake: false,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(sleeper, &ColliderDesc::new(Shape::Sphere { radius: 0.5 }))
        .unwrap();
    let mover = w
        .create_body(&BodyDesc {
            kind: BodyKind::Kinematic,
            position: Vec3::new(10.0, 5.0, 0.0),
            linear_velocity: Vec3::new(1.0, 0.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(mover, &ColliderDesc::new(Shape::Sphere { radius: 0.5 }))
        .unwrap();
    w.step(DT);
    assert!(!w.is_awake(sleeper) && w.is_awake(mover));
}
