//! The character wedged where no one surface holds it up: in a V of two slopes too steep to stand
//! on, or between such a slope and a wall. The surfaces hold it up together, so it is grounded
//! there instead of hanging in the air forever while the game's fall speed grows.

use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, Character, CharacterSettings, ColliderDesc, PhysicsWorld, Quat,
    Shape, Vec3,
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

/// A slab 6 m long whose top face rises at `degrees` from the line x = 0, y = 0.5, towards +X
/// (`side` 1) or -X (`side` -1).
fn slope(w: &mut PhysicsWorld, degrees: f32, side: f32) {
    let a = degrees.to_radians();
    let half = Vec3::new(3.0, 0.25, 5.0);
    let cx = half.x * a.cos() + half.y * a.sin();
    let cy = half.x * a.sin() - half.y * a.cos() + 0.5;
    static_box(
        w,
        Vec3::new(side * cx, cy, 0.0),
        half,
        Quat::from_axis_angle(Vec3::Z, side * a),
    );
}

/// Drops the character from `start` for five seconds; returns the move it was first grounded on
/// (`usize::MAX` if never). Once grounded it must stay grounded on walkable ground.
fn drop_in(w: &mut PhysicsWorld, start: Vec3, label: &str) -> usize {
    let mut c = Character::new(start, settings());
    let mut vy = 0.0f32;
    let mut landed = None;
    for i in 0..300 {
        vy = if c.grounded { -0.1 } else { vy - 9.81 * DT };
        let r = c.move_and_slide(w, Vec3::Y * (vy * DT), DT);
        w.step(DT);
        if r.grounded && landed.is_none() {
            landed = Some(i);
        }
        if landed.is_some() {
            assert!(
                r.grounded && r.ground_normal.y >= c.settings.max_slope_cos,
                "{label}, move {i}: not standing: {r:?}"
            );
        }
    }
    landed.unwrap_or(usize::MAX)
}

/// Dropped into a V of two slopes too steep to stand on (55° to 75°), in the middle or off it: it
/// comes to rest in the V and is grounded there within a second, standing on the two slopes
/// together. (No single slope holds it up, so it used to hang in the V forever, never grounded,
/// while the game's fall speed grew to 49 m/s.)
#[test]
fn a_v_of_two_slopes_too_steep_to_stand_on_holds_the_character_up() {
    for degrees in [55.0f32, 60.0, 75.0, 80.0] {
        for x in [0.0f32, 0.1, 0.2] {
            let mut w = flat();
            slope(&mut w, degrees, 1.0);
            slope(&mut w, degrees, -1.0);
            let label = format!("{degrees}° from x = {x}");
            let landed = drop_in(&mut w, Vec3::new(x, 4.0, 0.0), &label);
            assert!(landed < 90, "{label}: grounded only at move {landed}");
        }
    }
}

/// Dropped against a wall onto a slope too steep to stand on that runs down into the wall: it
/// slides down into the corner and is grounded there, held by the slope and the wall together
/// (the wall is no obstacle to a straight drop, so only a look down the slope finds it).
#[test]
fn a_slope_too_steep_to_stand_on_against_a_wall_holds_the_character_up() {
    for degrees in [55.0f32, 70.0] {
        let mut w = flat();
        slope(&mut w, degrees, 1.0);
        // A wall whose face is the plane x = 0.
        static_box(
            &mut w,
            Vec3::new(-0.5, 2.0, 0.0),
            Vec3::new(0.5, 2.0, 5.0),
            Quat::IDENTITY,
        );
        let label = format!("{degrees}° against a wall");
        let landed = drop_in(&mut w, Vec3::new(0.35, 4.0, 0.0), &label);
        assert!(landed < 90, "{label}: grounded only at move {landed}");
    }
}

/// Two walls 0.7 m apart hold nothing up: the character falls down between them to the floor.
#[test]
fn a_chimney_of_two_walls_does_not_hold_the_character_up() {
    let mut w = flat();
    for side in [-1.0f32, 1.0] {
        static_box(
            &mut w,
            Vec3::new(side * 0.85, 5.0, 0.0),
            Vec3::new(0.5, 4.0, 5.0),
            Quat::IDENTITY,
        );
    }
    let mut c = Character::new(Vec3::new(0.02, 8.0, 0.0), settings());
    let mut vy = 0.0f32;
    for i in 0..120 {
        vy = if c.grounded { -0.1 } else { vy - 9.81 * DT };
        let r = c.move_and_slide(&mut w, Vec3::new(0.3, vy, 0.0) * DT, DT);
        w.step(DT);
        if c.position.y > 1.0 {
            assert!(
                !r.grounded,
                "move {i}: grounded between the walls at {:?}",
                c.position
            );
        }
    }
    let s = settings();
    assert!(
        c.grounded && (c.feet().y - s.skin).abs() < 5e-4,
        "on the floor: {:?}",
        c.position
    );
}
