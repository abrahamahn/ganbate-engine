//! Events reach a game that reads them after each step, including the End events of contacts
//! that its own calls between steps removed. Every scene here is synthetic; touching needs the
//! narrow phase.

use ganbate_physics::{
    BodyDesc, BodyKind, ColliderDesc, ContactEvent, PhysicsWorld, SensorEvent, Settings, Shape,
    Vec3,
};

const DT: f32 = 1.0 / 60.0;

/// A pickup inside a trigger zone is collected (destroyed) by the game after it read a step's
/// events: the zone must report that it left, in the next step's events (review finding
/// EVENTS-LOST).
#[test]
fn a_trigger_reports_a_visitor_destroyed_between_steps() {
    let mut w = PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let zone = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            ..Default::default()
        })
        .unwrap();
    let mut d = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::splat(2.0),
    });
    d.sensor = true;
    let sensor = w.add_collider(zone, &d).unwrap();
    let pickup = w.create_body(&BodyDesc::default()).unwrap();
    let visitor = w
        .add_collider(pickup, &ColliderDesc::new(Shape::Sphere { radius: 0.25 }))
        .unwrap();
    let mut seen = Vec::new();
    for i in 0..10 {
        w.step(DT);
        seen.extend(w.events().sensors.iter().copied());
        if i == 4 {
            w.destroy_body(pickup);
            // The last step's events are what they were.
            assert!(w.events().sensors.is_empty(), "{:?}", w.events().sensors);
        }
    }
    assert_eq!(
        seen,
        vec![
            SensorEvent::Begin { sensor, visitor },
            SensorEvent::End { sensor, visitor }
        ]
    );
}

/// The same for a contact: a box resting on the ground loses its collider between steps.
#[test]
fn a_contact_reports_its_end_when_a_collider_is_removed_between_steps() {
    let mut w = PhysicsWorld::default();
    let ground = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(0.0, -0.5, 0.0),
            ..Default::default()
        })
        .unwrap();
    let g = w
        .add_collider(
            ground,
            &ColliderDesc::new(Shape::Box {
                half_extents: Vec3::new(10.0, 0.5, 10.0),
            }),
        )
        .unwrap();
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, 0.5, 0.0),
            ..Default::default()
        })
        .unwrap();
    let c = w
        .add_collider(
            b,
            &ColliderDesc::new(Shape::Box {
                half_extents: Vec3::splat(0.5),
            }),
        )
        .unwrap();
    let mut began = false;
    for _ in 0..30 {
        w.step(DT);
        began |= w
            .events()
            .contacts
            .contains(&ContactEvent::Begin { a: g, b: c });
    }
    assert!(began);
    w.remove_collider(c);
    w.step(DT);
    assert!(
        w.events()
            .contacts
            .contains(&ContactEvent::End { a: g, b: c }),
        "{:?}",
        w.events().contacts
    );
    w.step(DT);
    assert!(w.events().contacts.is_empty(), "reported once");
}
