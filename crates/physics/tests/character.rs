//! The character controller on synthetic ground: walking, walls, slopes, steps, snapping down,
//! pushing a box and getting out of an overlap.

use ganbate_physics::{
    BodyDesc, BodyKind, Character, CharacterSettings, ColliderDesc, PhysicsWorld, Quat, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;

fn static_box(w: &mut PhysicsWorld, center: Vec3, half: Vec3, rotation: Quat) {
    let b = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: center,
            rotation,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(b, &ColliderDesc::new(Shape::Box { half_extents: half }))
        .unwrap();
}

fn flat() -> PhysicsWorld {
    let mut w = PhysicsWorld::default();
    static_box(
        &mut w,
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(50.0, 0.5, 50.0),
        Quat::IDENTITY,
    );
    w
}

/// Standing height of the capsule's centre on flat ground at y = 0.
fn stand() -> f32 {
    let s = CharacterSettings::default();
    s.half_height + s.radius + s.skin
}

/// Walks at `velocity` (horizontal m/s) for `seconds`, with gravity integrated by the caller.
fn walk(w: &mut PhysicsWorld, c: &mut Character, velocity: Vec3, seconds: f32) {
    let mut fall = 0.0f32;
    for _ in 0..(seconds / DT).round() as usize {
        fall = if c.grounded { -0.1 } else { fall - 9.81 * DT };
        let r = c.move_and_slide(w, velocity * DT + Vec3::Y * (fall * DT), DT);
        assert!(c.position.is_finite(), "{r:?}");
        w.step(DT);
    }
}

#[test]
fn it_walks_on_flat_ground_and_stays_grounded() {
    let mut w = flat();
    let mut c = Character::new(
        Vec3::new(0.0, stand() + 0.5, 0.0),
        CharacterSettings::default(),
    );
    walk(&mut w, &mut c, Vec3::ZERO, 1.0);
    assert!(c.grounded, "it lands");
    assert!(
        (c.position.y - stand()).abs() < 0.03,
        "at standing height: {}",
        c.position.y
    );
    walk(&mut w, &mut c, Vec3::new(2.0, 0.0, 0.0), 1.0);
    assert!(
        (c.position.x - 2.0).abs() < 0.1,
        "walked 2 m: {}",
        c.position.x
    );
    assert!(c.grounded && (c.position.y - stand()).abs() < 0.03);
}

#[test]
fn it_slides_along_a_wall() {
    let mut w = flat();
    static_box(
        &mut w,
        Vec3::new(2.0, 1.0, 0.0),
        Vec3::new(0.5, 1.0, 10.0),
        Quat::IDENTITY,
    );
    let mut c = Character::new(Vec3::new(0.0, stand(), 0.0), CharacterSettings::default());
    walk(&mut w, &mut c, Vec3::new(0.0, 0.0, 0.0), 0.2);
    walk(&mut w, &mut c, Vec3::new(3.0, 0.0, 3.0), 1.0);
    assert!(
        c.position.x < 1.5 - 0.3 + 0.05,
        "stopped by the wall: {:?}",
        c.position
    );
    assert!(c.position.z > 2.0, "slid along it: {:?}", c.position);
}

#[test]
fn it_climbs_a_gentle_slope_but_not_a_steep_one() {
    for (degrees, climbs) in [(20.0f32, true), (60.0, false)] {
        let mut w = flat();
        let a = degrees.to_radians();
        // A ramp rising towards +X from x = 1.
        let half = Vec3::new(5.0, 0.25, 5.0);
        let q = Quat::from_axis_angle(Vec3::Z, a);
        let center = Vec3::new(
            1.0 + half.x * a.cos(),
            half.x * a.sin() - half.y * a.cos(),
            0.0,
        );
        static_box(&mut w, center, half, q);
        let mut c = Character::new(Vec3::new(0.0, stand(), 0.0), CharacterSettings::default());
        walk(&mut w, &mut c, Vec3::ZERO, 0.2);
        walk(&mut w, &mut c, Vec3::new(2.0, 0.0, 0.0), 2.0);
        if climbs {
            assert!(
                c.position.y > stand() + 0.5,
                "{degrees}°: climbed to {:?}",
                c.position
            );
        } else {
            assert!(
                c.position.y < stand() + 0.35,
                "{degrees}°: held back at {:?}",
                c.position
            );
        }
    }
}

#[test]
fn it_climbs_a_low_step_but_not_a_high_one() {
    for (height, climbs) in [(0.25f32, true), (0.5, false)] {
        let mut w = flat();
        static_box(
            &mut w,
            Vec3::new(3.0, height * 0.5, 0.0),
            Vec3::new(2.0, height * 0.5, 5.0),
            Quat::IDENTITY,
        );
        let mut c = Character::new(Vec3::new(0.0, stand(), 0.0), CharacterSettings::default());
        walk(&mut w, &mut c, Vec3::ZERO, 0.2);
        walk(&mut w, &mut c, Vec3::new(2.0, 0.0, 0.0), 1.5);
        if climbs {
            assert!(
                c.position.x > 1.5 && (c.position.y - (stand() + height)).abs() < 0.05,
                "{height} m: on top at {:?}",
                c.position
            );
        } else {
            assert!(
                c.position.x < 1.0 - 0.3 + 0.05,
                "{height} m: blocked at {:?}",
                c.position
            );
        }
    }
}

#[test]
fn it_stays_on_the_ground_walking_down_a_slope() {
    let mut w = flat();
    let a = 15.0f32.to_radians();
    let half = Vec3::new(6.0, 0.25, 5.0);
    let q = Quat::from_axis_angle(Vec3::Z, -a);
    // A ramp falling towards +X from (0, 2).
    let top = Vec3::new(0.0, 2.0, 0.0);
    let center = top + Vec3::new(half.x * a.cos(), -half.x * a.sin() - half.y * a.cos(), 0.0);
    static_box(&mut w, center, half, q);
    let mut c = Character::new(
        Vec3::new(0.3, 2.0 + stand(), 0.0),
        CharacterSettings::default(),
    );
    walk(&mut w, &mut c, Vec3::ZERO, 0.3);
    let mut airborne = 0;
    for _ in 0..90 {
        let r = c.move_and_slide(&mut w, Vec3::new(3.0, -0.1, 0.0) * DT, DT);
        if !r.grounded {
            airborne += 1;
        }
        w.step(DT);
    }
    assert!(
        airborne <= 2,
        "it hugs the slope: airborne {airborne} of 90 steps"
    );
}

#[test]
// The box rests on the floor through box-box contacts.
fn it_pushes_a_box() {
    let mut w = flat();
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(1.5, 0.25, 0.0),
            ..Default::default()
        })
        .unwrap();
    let mut d = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::splat(0.25),
    });
    d.density = 200.0;
    w.add_collider(b, &d).unwrap();
    let mut c = Character::new(Vec3::new(0.0, stand(), 0.0), CharacterSettings::default());
    walk(&mut w, &mut c, Vec3::ZERO, 0.2);
    walk(&mut w, &mut c, Vec3::new(1.5, 0.0, 0.0), 2.0);
    assert!(
        w.body_pose(b).unwrap().position.x > 2.0,
        "the box moved: {:?}",
        w.body_pose(b)
    );
}

#[test]
// Getting out of a box the capsule's core is inside takes the capsule-box contact's normal.
fn it_gets_out_of_a_box_it_starts_inside() {
    let mut w = flat();
    static_box(
        &mut w,
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::splat(0.5),
        Quat::IDENTITY,
    );
    let mut c = Character::new(Vec3::new(0.35, stand(), 0.0), CharacterSettings::default());
    c.move_and_slide(&mut w, Vec3::ZERO, DT);
    // Out sideways (+X is the shortest way out), clear of the box.
    assert!(
        c.position.x >= 0.5 + 0.3 - 0.01,
        "pushed out: {:?}",
        c.position
    );
}
