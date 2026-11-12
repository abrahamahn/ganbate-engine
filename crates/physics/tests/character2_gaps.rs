//! Walking through openings with no room for a full skin all round: doorways and corridor mouths
//! between twice the capsule's radius and twice its radius plus skin wide, which the capsule fits
//! through when its skin gives way; and the narrow end of a wedge, where giving way must not make
//! it creep in and out.

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

/// Walking through a doorway in a wall: any doorway wider than the capsule by more than a few
/// millimetres lets it through, as it did before the skin was kept from every surface (it then
/// needed 0.645 m for this 0.6 m capsule); a doorway as wide as the capsule does not.
#[test]
fn doorways_wider_than_the_capsule_let_it_through() {
    for (width, passes) in [
        (0.6f32, false),
        (0.61, true),
        (0.62, true),
        (0.63, true),
        (0.64, true),
        (0.7, true),
    ] {
        for (x, heading) in [(0.0f32, 0.0f32), (0.002, 0.0), (-0.04, 1.0)] {
            let mut w = flat();
            // A wall 0.2 m thick from z = 2 to z = 2.2 with the doorway in its middle.
            for side in [-1.0f32, 1.0] {
                static_box(
                    &mut w,
                    Vec3::new(side * (0.5 * width + 2.0), 1.0, 2.1),
                    Vec3::new(2.0, 1.0, 0.1),
                    Quat::IDENTITY,
                );
            }
            let mut c = Character::new(Vec3::new(x, stand(), 0.0), settings());
            let a = heading.to_radians();
            let velocity = Vec3::new(a.sin(), 0.0, a.cos()) * 2.0;
            let mut vy = 0.0f32;
            for i in 0..120 {
                vy = if c.grounded { -0.1 } else { vy - 9.81 * DT };
                c.move_and_slide(&mut w, velocity * DT + Vec3::Y * (vy * DT), DT);
                w.step(DT);
                if (2.0..=2.2).contains(&c.position.z) {
                    assert!(
                        into_walls(&c, width) < 0.0,
                        "{width} m from x = {x} heading {heading}°, move {i}: in a jamb at {:?}",
                        c.position
                    );
                }
            }
            assert_eq!(
                c.position.z > 3.0,
                passes,
                "{width} m from x = {x} heading {heading}°: ended at {:?}",
                c.position
            );
        }
    }
}

/// Walking into a corridor 0.61 m to 0.64 m wide from outside, it goes in and along it.
#[test]
fn it_walks_into_a_corridor_barely_wider_than_itself() {
    for width in [0.61f32, 0.62, 0.64] {
        let mut w = flat();
        corridor(&mut w, width, 2.0, 6.0);
        let mut c = Character::new(Vec3::new(0.0, stand(), 0.0), settings());
        let mut vy = 0.0f32;
        for _ in 0..150 {
            vy = if c.grounded { -0.1 } else { vy - 9.81 * DT };
            c.move_and_slide(&mut w, Vec3::new(0.0, vy, 2.0) * DT, DT);
            w.step(DT);
        }
        assert!(
            c.position.z > 4.5 && into_walls(&c, width) < 0.0 && c.grounded,
            "{width} m: walked in to {:?}",
            c.position
        );
    }
}

/// Walking into the narrow end of a wedge between two walls 10° to 30° apart, straight in or off
/// its middle: it stops a skin from both walls and stays still there, move after move. (Giving
/// the skin way between surfaces facing each other must not let it creep deeper into a corner it
/// cannot pass, nor in and out every move.)
#[test]
fn walking_into_a_narrow_wedge_stops_a_skin_from_both_walls() {
    for (degrees, x) in [(15.0f32, 0.0f32), (15.0, 0.1), (5.0, 0.0), (5.0, 0.05)] {
        let mut w = flat();
        // Two walls meeting at x = 0, z = 4, opening towards -Z at `degrees` each side.
        let a = degrees.to_radians();
        let mut faces = Vec::new();
        for side in [-1.0f32, 1.0] {
            // Along the wall's inner face from the apex, and out of the wedge through the wall.
            let along = Vec3::new(side * a.sin(), 0.0, -a.cos());
            let out = Vec3::new(side * a.cos(), 0.0, a.sin());
            let center = Vec3::new(0.0, 1.0, 4.0) + along * 3.0 + out * 0.5;
            static_box(
                &mut w,
                center,
                Vec3::new(0.5, 1.0, 3.0),
                Quat::from_axis_angle(Vec3::Y, -side * a),
            );
            faces.push(out);
        }
        let s = settings();
        let mut c = Character::new(Vec3::new(x, stand(), 0.0), s.clone());
        let mut vy = 0.0f32;
        let mut trail = Vec::new();
        for i in 0..240 {
            vy = if c.grounded { -0.1 } else { vy - 9.81 * DT };
            c.move_and_slide(&mut w, Vec3::new(0.0, vy, 2.0) * DT, DT);
            w.step(DT);
            trail.push(c.position);
            for &out in &faces {
                // How far the capsule is from this wall's face (through the apex).
                let gap = (Vec3::new(0.0, 0.0, 4.0) - Vec3::new(c.position.x, 0.0, c.position.z))
                    .dot(out)
                    - s.radius;
                assert!(
                    gap > s.skin - 5e-4,
                    "{degrees}° from x = {x}, move {i}: {gap} m from a wall at {:?}",
                    c.position
                );
            }
        }
        // Where both walls are a skin away, on the wedge's middle line.
        let stop = 4.0 - (s.radius + s.skin) / a.sin();
        let last = trail[trail.len() - 1];
        assert!(
            (last.z - stop).abs() < 0.01 && last.x.abs() < 0.01,
            "{degrees}° from x = {x}: walked in to {last:?}, not to z = {stop}"
        );
        for p in &trail[trail.len() - 60..] {
            assert!(
                (*p - last).length() < 1e-4,
                "{degrees}° from x = {x}: still moving in the wedge: {p:?} then {last:?}"
            );
        }
    }
}

/// A floor slab at y = 2 with a square hole `side` metres wide in its middle, over the floor.
fn holed_floor(side: f32) -> PhysicsWorld {
    let mut w = flat();
    let h = 0.5 * side;
    for (center, half) in [
        (Vec3::new(h + 2.0, 1.9, 0.0), Vec3::new(2.0, 0.1, 5.0)),
        (Vec3::new(-h - 2.0, 1.9, 0.0), Vec3::new(2.0, 0.1, 5.0)),
        (Vec3::new(0.0, 1.9, h + 2.0), Vec3::new(h, 0.1, 2.0)),
        (Vec3::new(0.0, 1.9, -h - 2.0), Vec3::new(h, 0.1, 2.0)),
    ] {
        static_box(&mut w, center, half, Quat::IDENTITY);
    }
    w
}

/// Drops the character from 2 m above a holed floor for three seconds; the last move's result.
fn drop_over_hole(w: &mut PhysicsWorld, c: &mut Character) -> ganbate_physics::MoveResult {
    let mut vy = 0.0f32;
    let mut last = None;
    for i in 0..180 {
        vy = if c.grounded { -0.1 } else { vy - 9.81 * DT };
        let r = c.move_and_slide(w, Vec3::Y * (vy * DT), DT);
        w.step(DT);
        assert!(
            r.grounded || r.moved.y < -1e-4,
            "move {i}: hanging at {:?}",
            c.position
        );
        last = Some(r);
    }
    last.unwrap()
}

/// Dropped straight into a hole in a floor 1 to 4 cm wider than itself, it falls through: the
/// hole's rim is beyond its radius all round, so only its skin would rest on it. (It used to stand
/// on the rim, held up by its skin, in any hole up to 0.64 m wide.)
#[test]
fn falling_into_a_hole_it_fits_through_drops_through() {
    for side in [0.61f32, 0.62, 0.64] {
        let mut w = holed_floor(side);
        let mut c = Character::new(Vec3::new(0.0, 4.0, 0.0), settings());
        let r = drop_over_hole(&mut w, &mut c);
        assert!(
            r.grounded && (c.position.y - stand()).abs() < 5e-4,
            "{side} m: fell through to the floor below, not at {:?}",
            c.position
        );
    }
}

/// A hole too narrow for even the squeezed gap all round holds the character up on its rim (and
/// never leaves it hanging there, neither grounded nor falling).
#[test]
fn a_hole_too_narrow_to_squeeze_through_holds_the_character_up() {
    for side in [0.59f32, 0.602] {
        let mut w = holed_floor(side);
        let mut c = Character::new(Vec3::new(0.0, 4.0, 0.0), settings());
        let r = drop_over_hole(&mut w, &mut c);
        assert!(
            r.grounded && c.position.y > 2.0,
            "{side} m: on the rim, not at {:?}",
            c.position
        );
    }
}
