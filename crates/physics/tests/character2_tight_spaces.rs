//! Getting out of what moved into the character, or what it was put down in, where there is no
//! room for a full skin all round: corridors between twice its radius and twice its radius plus
//! skin wide, and doors closing on it. Surfaces facing each other closer than two skins apart
//! share what room there is, and nothing ends closer to the capsule than it was.

use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, Character, CharacterSettings, ColliderDesc, Iso, PhysicsWorld,
    Quat, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;

fn static_box(w: &mut PhysicsWorld, center: Vec3, half: Vec3, rotation: Quat) -> BodyId {
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
    b
}

/// A 100 m floor whose top is at y = 0.
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

fn settings() -> CharacterSettings {
    CharacterSettings::default()
}

fn stand() -> f32 {
    let s = settings();
    s.half_height + s.radius + s.skin
}

/// A corridor along Z, `width` wide between its walls' faces (at x = ±width / 2), from z = -5 to
/// z = 5 + `length`.
fn corridor(w: &mut PhysicsWorld, width: f32, z0: f32, length: f32) {
    for side in [-1.0f32, 1.0] {
        static_box(
            w,
            Vec3::new(side * (0.5 * width + 0.5), 1.0, z0 + 0.5 * length),
            Vec3::new(0.5, 1.0, 0.5 * length),
            Quat::IDENTITY,
        );
    }
}

/// How far the capsule is inside the nearer of two walls at x = ±width / 2 (negative: clear).
fn into_walls(c: &Character, width: f32) -> f32 {
    c.position.x.abs() + c.settings.radius - 0.5 * width
}

/// Spawned in the middle of a corridor 1 cm wider than the capsule (5 mm each side, under a skin),
/// standing still: it stays in the middle. (Each move used to push it 1 cm into one wall: the two
/// walls' pushes to a skin each could not both be met, and the last one won.)
#[test]
fn standing_in_a_corridor_narrower_than_the_skins_keeps_the_character_centred() {
    let width = 0.61;
    let mut w = flat();
    corridor(&mut w, width, -5.0, 10.0);
    let mut c = Character::new(Vec3::new(0.0, stand(), 0.0), settings());
    c.grounded = true;
    for i in 0..8 {
        c.move_and_slide(&mut w, Vec3::new(0.0, -0.1 * DT, 0.0), DT);
        assert!(
            c.position.x.abs() < 1e-4,
            "move {i}: pushed off the middle to {:?}",
            c.position
        );
    }
}

/// Put down off-centre in a corridor 5 mm wider than the capsule, closer than the narrow phase's
/// slop to one wall: it is put in the middle, and never deeper into either wall than it was.
#[test]
fn put_down_off_centre_in_a_tight_corridor_the_character_is_centred() {
    for (width, x) in [(0.605f32, 0.0021f32), (0.605, -0.0024), (0.602, 0.0009)] {
        let mut w = flat();
        corridor(&mut w, width, -5.0, 10.0);
        let mut c = Character::new(Vec3::new(x, stand(), 0.0), settings());
        c.grounded = true;
        let start = into_walls(&c, width);
        for i in 0..8 {
            c.move_and_slide(&mut w, Vec3::new(0.0, -0.1 * DT, 0.0), DT);
            assert!(
                into_walls(&c, width) <= start.max(-0.5 * (width - 0.6)) + 1e-5,
                "{width} m, from x = {x}, move {i}: into a wall at {:?}",
                c.position
            );
        }
        assert!(
            c.position.x.abs() < 1e-4,
            "{width} m, from x = {x}: centred, not at {:?}",
            c.position
        );
    }
}

/// Two kinematic doors slide shut on the character to 0.62 m apart: it is never inside either.
#[test]
fn doors_closing_on_the_character_never_push_it_into_each_other() {
    let mut w = flat();
    let door = |w: &mut PhysicsWorld, x: f32| {
        let d = w
            .create_body(&BodyDesc {
                kind: BodyKind::Kinematic,
                position: Vec3::new(x, 1.0, 0.0),
                can_sleep: false,
                ..Default::default()
            })
            .unwrap();
        w.add_collider(
            d,
            &ColliderDesc::new(Shape::Box {
                half_extents: Vec3::new(0.5, 1.0, 1.0),
            }),
        )
        .unwrap();
        d
    };
    let (l, r) = (door(&mut w, -1.5), door(&mut w, 1.5));
    for offset in [0.0f32, 0.05] {
        let mut c = Character::new(Vec3::new(offset, stand(), 0.0), settings());
        let mut vy = 0.0f32;
        for i in 0..120 {
            // Faces from ±1 m to ±0.31 m at 0.5 m/s, then still.
            let face = (1.0 - 0.5 * DT * (i + 1) as f32).max(0.31);
            w.set_kinematic_target(l, Iso::from_position(Vec3::new(-face - 0.5, 1.0, 0.0)), DT);
            w.set_kinematic_target(r, Iso::from_position(Vec3::new(face + 0.5, 1.0, 0.0)), DT);
            w.step(DT);
            vy = if c.grounded { -0.1 } else { vy - 9.81 * DT };
            c.move_and_slide(&mut w, Vec3::Y * (vy * DT), DT);
            let lf = -(w.body_pose(l).unwrap().position.x + 0.5);
            let rf = w.body_pose(r).unwrap().position.x - 0.5;
            let into = (c.position.x + 0.3 - rf).max(-lf - (c.position.x - 0.3));
            assert!(
                into <= 1e-4,
                "from {offset}, move {i}: {into} m into a door at {:?}",
                c.position
            );
        }
        // Reopen for the next run.
        w.set_body_pose(l, Iso::from_position(Vec3::new(-1.5, 1.0, 0.0)));
        w.set_body_pose(r, Iso::from_position(Vec3::new(1.5, 1.0, 0.0)));
        w.set_linear_velocity(l, Vec3::ZERO);
        w.set_linear_velocity(r, Vec3::ZERO);
    }
}

/// A lift carries the character up into a ceiling: it is never pushed into the ceiling (the
/// level geometry), whatever the lift does to it. (Centred between the two, it was carried 0.7 m
/// up into the ceiling slab.)
#[test]
fn a_lift_rising_into_a_ceiling_never_pushes_the_character_into_it() {
    for after in [true, false] {
        let mut w = PhysicsWorld::default();
        let lift = w
            .create_body(&BodyDesc {
                kind: BodyKind::Kinematic,
                position: Vec3::new(0.0, -0.25, 0.0),
                can_sleep: false,
                ..Default::default()
            })
            .unwrap();
        w.add_collider(
            lift,
            &ColliderDesc::new(Shape::Box {
                half_extents: Vec3::new(5.0, 0.25, 5.0),
            }),
        )
        .unwrap();
        // A ceiling slab whose underside is at y = 2.5.
        let ceiling = w
            .create_body(&BodyDesc {
                kind: BodyKind::Static,
                position: Vec3::new(0.0, 3.0, 0.0),
                ..Default::default()
            })
            .unwrap();
        w.add_collider(
            ceiling,
            &ColliderDesc::new(Shape::Box {
                half_extents: Vec3::new(10.0, 0.5, 10.0),
            }),
        )
        .unwrap();
        let mut c = Character::new(Vec3::new(0.0, stand(), 0.0), settings());
        let mut vy = 0.0f32;
        for i in 0..190 {
            if i == 10 {
                w.set_linear_velocity(lift, Vec3::new(0.0, 1.0, 0.0));
            }
            vy = if c.grounded { -0.1 } else { vy - 9.81 * DT };
            if after {
                w.step(DT);
            }
            c.move_and_slide(&mut w, Vec3::Y * (vy * DT), DT);
            if !after {
                w.step(DT);
            }
            let head = c.position.y + c.settings.half_height + c.settings.radius;
            assert!(
                c.position.is_finite() && head < 2.5 + 1e-3,
                "moved {} the step, move {i}: head at {head}",
                if after { "after" } else { "before" }
            );
        }
    }
}
