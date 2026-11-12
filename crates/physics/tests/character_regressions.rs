//! Regression tests for the character controller's review findings: the skin it keeps from every
//! surface, falling onto slopes too steep to stand on, pushing bodies, and riding platforms that
//! rise or turn. None of them needs contact manifolds: everything here goes through the capsule's
//! sweeps (GJK casts) and rays, so they run with any narrow phase.

use std::sync::Arc;

use ganbate_physics::mesh::{Heightfield, TriangleMesh};
use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, Character, CharacterSettings, ColliderDesc, PhysicsWorld, Quat,
    Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;
/// How far from the skin a resting gap may be: the casts' tolerance band (a few tenths of a
/// millimetre against the 100 m floor) and rounding.
const GAP: f32 = 5e-4;

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

/// A wall whose face is the plane x = 1.5 (the wall of `character.rs`'s sliding test).
fn wall(w: &mut PhysicsWorld) {
    static_box(
        w,
        Vec3::new(2.0, 1.0, 0.0),
        Vec3::new(0.5, 1.0, 10.0),
        Quat::IDENTITY,
    );
}

/// A box ramp rising towards +X from x = 1 at `degrees`.
fn ramp(w: &mut PhysicsWorld, degrees: f32) {
    let a = degrees.to_radians();
    let half = Vec3::new(5.0, 0.25, 5.0);
    let center = Vec3::new(
        1.0 + half.x * a.cos(),
        half.x * a.sin() - half.y * a.cos(),
        0.0,
    );
    static_box(w, center, half, Quat::from_axis_angle(Vec3::Z, a));
}

fn settings() -> CharacterSettings {
    CharacterSettings::default()
}

/// Height of the capsule's centre standing a skin above ground at y = 0.
fn stand() -> f32 {
    let s = settings();
    s.half_height + s.radius + s.skin
}

/// The game's vertical speed: a small push down while grounded, gravity otherwise.
fn fall(c: &Character, vy: &mut f32) -> f32 {
    *vy = if c.grounded { -0.1 } else { *vy - 9.81 * DT };
    *vy
}

/// A kinematic box platform (still until given a velocity) that never sleeps.
fn platform(w: &mut PhysicsWorld, center: Vec3, half: Vec3) -> BodyId {
    let p = w
        .create_body(&BodyDesc {
            kind: BodyKind::Kinematic,
            position: center,
            can_sleep: false,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(p, &ColliderDesc::new(Shape::Box { half_extents: half }))
        .unwrap();
    p
}

/// char-skin-lost: standing still and then sliding along a wall, the capsule stays a skin above
/// the floor and a skin off the wall. (It used to creep down to 0.6 mm above the floor within ten
/// steps, and to stop only 14 mm off a wall it walked into at 45°.)
#[test]
fn the_skin_is_kept_from_the_floor_and_from_a_wall() {
    let mut w = flat();
    wall(&mut w);
    let mut c = Character::new(Vec3::new(0.0, stand(), 0.0), settings());
    let skin = c.settings.skin;
    let mut vy = 0.0;
    for i in 0..120 {
        let walk = if i < 60 {
            Vec3::ZERO
        } else {
            Vec3::new(3.0, 0.0, 3.0)
        };
        let dy = fall(&c, &mut vy);
        c.move_and_slide(&mut w, walk * DT + Vec3::Y * (dy * DT), DT);
        w.step(DT);
        let floor = c.feet().y;
        let off_wall = 1.5 - (c.position.x + c.settings.radius);
        assert!(
            (floor - skin).abs() < GAP,
            "step {i}: {floor} m above the floor"
        );
        assert!(off_wall > skin - GAP, "step {i}: {off_wall} m off the wall");
        assert!(c.grounded, "step {i}: not grounded");
    }
    let off_wall = 1.5 - (c.position.x + c.settings.radius);
    assert!(
        (off_wall - skin).abs() < GAP,
        "ends a skin off the wall: {off_wall}"
    );
    assert!(c.position.z > 2.0, "slid along the wall: {:?}", c.position);
}

/// char-skin-lost: a capsule closer to the floor and a wall than the casts' tolerance (where every
/// cast of the old controller stopped at once, whatever its direction) walks along the wall and
/// away from it, and is put back a skin above the floor.
#[test]
fn touching_the_floor_and_a_wall_does_not_hold_the_character() {
    let mut w = flat();
    wall(&mut w);
    let s = settings();
    let touching = 5e-5;
    let mut c = Character::new(
        Vec3::new(
            1.5 - s.radius - touching,
            s.half_height + s.radius + touching,
            0.0,
        ),
        s,
    );
    c.grounded = true;
    for _ in 0..30 {
        c.move_and_slide(&mut w, Vec3::new(0.0, -0.1, 2.0) * DT, DT);
        w.step(DT);
    }
    assert!(
        c.position.z > 0.95,
        "walked 1 m along the wall: {:?}",
        c.position
    );
    assert!(
        (c.feet().y - c.settings.skin).abs() < GAP,
        "back a skin above the floor: {:?}",
        c.position
    );
    let x = c.position.x;
    for _ in 0..30 {
        c.move_and_slide(&mut w, Vec3::new(-2.0, -0.1, 0.0) * DT, DT);
        w.step(DT);
    }
    assert!(
        c.position.x < x - 0.95,
        "walked 1 m away from the wall: {:?}",
        c.position
    );
}

/// char-skin-lost: the sliding sweeps keep their skin on a triangle mesh whose floor and walls are
/// one collider (each triangle is its own surface: the floor under the capsule must not stop it
/// walking into, and along, the wall).
#[test]
fn it_slides_along_the_wall_of_a_mesh_room() {
    let mut w = PhysicsWorld::default();
    // A floor (y = 0, facing up) and a wall at x = 1.5 facing -X, in one mesh.
    let vertices = vec![
        Vec3::new(-10.0, 0.0, -10.0),
        Vec3::new(10.0, 0.0, -10.0),
        Vec3::new(10.0, 0.0, 10.0),
        Vec3::new(-10.0, 0.0, 10.0),
        Vec3::new(1.5, 0.0, -10.0),
        Vec3::new(1.5, 3.0, -10.0),
        Vec3::new(1.5, 3.0, 10.0),
        Vec3::new(1.5, 0.0, 10.0),
    ];
    let triangles = vec![[0, 2, 1], [0, 3, 2], [4, 6, 5], [4, 7, 6]];
    let room = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        room,
        &ColliderDesc::new(Shape::Mesh(Arc::new(
            TriangleMesh::new(vertices, triangles).unwrap(),
        ))),
    )
    .unwrap();
    let mut c = Character::new(Vec3::new(0.0, stand(), -5.0), settings());
    let skin = c.settings.skin;
    let mut vy = 0.0;
    for i in 0..90 {
        let dy = fall(&c, &mut vy);
        c.move_and_slide(&mut w, Vec3::new(3.0, dy, 3.0) * DT, DT);
        w.step(DT);
        assert!(
            (c.feet().y - skin).abs() < GAP && c.grounded,
            "step {i}: {:?} grounded {}",
            c.position,
            c.grounded
        );
        assert!(
            1.5 - (c.position.x + c.settings.radius) > skin - GAP,
            "step {i}: into the wall at {:?}",
            c.position
        );
    }
    assert!(
        (1.5 - (c.position.x + c.settings.radius) - skin).abs() < GAP,
        "a skin off the wall: {:?}",
        c.position
    );
    assert!(c.position.z > -1.0, "slid along it: {:?}", c.position);
}

/// char-skin-lost: walking onto a gently sloped heightfield keeps it on the ground every step.
#[test]
fn it_walks_up_and_across_a_heightfield() {
    let mut w = flat();
    // 3 cm up per metre along +X, from x = 2.
    let heights: Vec<f32> = (0..16 * 16).map(|k| ((k % 16) as f32) * 0.03).collect();
    let terrain = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(2.0, 0.0, -8.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        terrain,
        &ColliderDesc::new(Shape::Heightfield(Arc::new(
            Heightfield::new(16, 16, heights, Vec3::new(1.0, 1.0, 1.0)).unwrap(),
        ))),
    )
    .unwrap();
    let mut c = Character::new(Vec3::new(0.0, stand(), 0.0), settings());
    let mut vy = 0.0;
    for i in 0..300 {
        let dy = fall(&c, &mut vy);
        c.move_and_slide(&mut w, Vec3::new(2.0, dy, 0.0) * DT, DT);
        w.step(DT);
        if i > 0 {
            assert!(c.grounded, "step {i}: in the air at {:?}", c.position);
        }
    }
    assert!(c.position.x > 9.5, "walked across: {:?}", c.position);
    // 3 cm per metre at x - 2 metres in, standing a skin above it (measured along the
    // slope's normal, so a hair more straight up).
    let ground = 0.03 * (c.position.x - 2.0);
    assert!(
        (c.feet().y - ground - c.settings.skin).abs() < 2e-3,
        "on the terrain: {:?}",
        c.position
    );
}

/// char-skin-lost: the step test at walking speeds from a stroll to a run (the old controller
/// needed more than a skin of motion left over to even try a step, and its landing on the step's
/// nosing was always too steep for a rounded capsule).
#[test]
fn it_climbs_a_step_at_any_walking_speed_but_never_a_step_too_high() {
    for speed in [0.5f32, 1.0, 2.0, 4.0] {
        for (height, climbs) in [(0.25f32, true), (0.3, true), (0.32, false), (0.5, false)] {
            let mut w = flat();
            // A step from x = 1 to x = 9.
            static_box(
                &mut w,
                Vec3::new(5.0, height * 0.5, 0.0),
                Vec3::new(4.0, height * 0.5, 5.0),
                Quat::IDENTITY,
            );
            let mut c = Character::new(Vec3::new(0.0, stand(), 0.0), settings());
            let mut vy = 0.0;
            // Long enough to walk 3 m on the flat.
            let steps = (3.0 / speed / DT) as usize;
            for _ in 0..steps {
                let dy = fall(&c, &mut vy);
                c.move_and_slide(&mut w, Vec3::new(speed, dy, 0.0) * DT, DT);
                w.step(DT);
            }
            if climbs {
                assert!(
                    c.position.x > 1.4 && (c.position.y - (stand() + height)).abs() < GAP,
                    "{speed} m/s, {height} m: on top at {:?}",
                    c.position
                );
            } else {
                assert!(
                    c.position.x < 1.0 - c.settings.radius && (c.position.y - stand()).abs() < GAP,
                    "{speed} m/s, {height} m: blocked at {:?}",
                    c.position
                );
            }
        }
    }
}

/// char-steep-slope-hover: dropped onto a 60° slope (steeper than the 50° limit), it slides down
/// the slope and lands on the floor below; while in the air it never stops moving. (It used to
/// hang at its first touch on the slope forever, never grounded.)
#[test]
fn falling_onto_a_slope_too_steep_to_stand_on_slides_off_it() {
    let mut w = flat();
    ramp(&mut w, 60.0);
    let mut c = Character::new(Vec3::new(3.0, 6.0, 0.0), settings());
    let mut vy = 0.0;
    let mut touched_slope = false;
    for i in 0..240 {
        let dy = fall(&c, &mut vy);
        let r = c.move_and_slide(&mut w, Vec3::Y * (dy * DT), DT);
        w.step(DT);
        touched_slope |= r.hits.iter().any(|h| h.1.y < 0.6);
        if !c.grounded {
            assert!(
                r.moved.y < -1e-4,
                "step {i}: hanging in the air at {:?}: {r:?}",
                c.position
            );
        }
    }
    assert!(touched_slope, "it met the slope");
    assert!(
        c.grounded && c.position.x < 1.0 && (c.position.y - stand()).abs() < GAP,
        "on the floor at the foot of the slope: {:?}",
        c.position
    );
}

/// char-steep-slope-hover: dropped onto a ledge's edge at every offset, it either stands on the
/// ledge or slides off it to the floor, and never hangs in the air.
#[test]
fn falling_onto_a_ledge_edge_never_hangs() {
    for offset in [-0.1f32, 0.0, 0.1, 0.2, 0.25, 0.29, 0.31] {
        let mut w = flat();
        // A ledge 1 m high whose edge is the line x = 0, y = 1.
        static_box(
            &mut w,
            Vec3::new(-1.0, 0.5, 0.0),
            Vec3::new(1.0, 0.5, 5.0),
            Quat::IDENTITY,
        );
        // Capsule axis `offset` beyond the edge.
        let mut c = Character::new(Vec3::new(offset, 3.0, 0.0), settings());
        let mut vy = 0.0;
        for i in 0..120 {
            let dy = fall(&c, &mut vy);
            let r = c.move_and_slide(&mut w, Vec3::Y * (dy * DT), DT);
            w.step(DT);
            if !c.grounded {
                assert!(
                    r.moved.y < -1e-4,
                    "offset {offset}, step {i}: hanging at {:?}",
                    c.position
                );
            }
        }
        assert!(c.grounded, "offset {offset}: landed at {:?}", c.position);
    }
}

/// A light box floating at the character's height (no gravity, no damping): only pushes move it.
fn floating_box(w: &mut PhysicsWorld, half: Vec3, mass: f32) -> BodyId {
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(1.0, 0.9, 0.0),
            gravity_scale: 0.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            can_sleep: false,
            ..Default::default()
        })
        .unwrap();
    let mut d = ColliderDesc::new(Shape::Box { half_extents: half });
    d.density = mass / (8.0 * half.x * half.y * half.z);
    w.add_collider(b, &d).unwrap();
    b
}

/// char-push-overspeed: walking into a 1 kg box never makes it faster than the walk. (It reached
/// 2.67 m/s at a 1.5 m/s walk: a push for every hit in one move, each ignoring the box's speed.)
#[test]
fn a_pushed_light_box_never_outruns_the_walker() {
    let mut w = flat();
    let b = floating_box(&mut w, Vec3::splat(0.25), 1.0);
    let mut c = Character::new(Vec3::new(0.0, stand(), 0.0), settings());
    let walk = 1.5f32;
    let mut vy = 0.0;
    let mut fastest = 0.0f32;
    for _ in 0..90 {
        let dy = fall(&c, &mut vy);
        c.move_and_slide(&mut w, Vec3::new(walk, dy, 0.0) * DT, DT);
        w.step(DT);
        fastest = fastest.max(w.linear_velocity(b).unwrap().length());
    }
    assert!(fastest > 1.4, "it was pushed: {fastest} m/s");
    assert!(
        fastest <= walk + 1e-3,
        "no faster than the walk: {fastest} m/s"
    );
}

/// char-push-overspeed: a step attempt that is undone (the box is too tall to climb) pushes
/// nothing, and each move pushes a body once. (The old controller pushed this 4 kg box twice in
/// one move: 3.81 m/s at a 2 m/s walk.)
#[test]
fn an_undone_step_attempt_pushes_nothing() {
    let mut w = flat();
    let b = floating_box(&mut w, Vec3::new(0.25, 0.8, 0.25), 4.0);
    let m = w.mass(b).unwrap();
    let mut c = Character::new(Vec3::new(0.0, stand(), 0.0), settings());
    let walk = 2.0f32;
    let one_push = walk * 80.0 / (80.0 + m);
    let mut vy = 0.0;
    let mut fastest = 0.0f32;
    for i in 0..60 {
        let dy = fall(&c, &mut vy);
        let before = w.linear_velocity(b).unwrap().x;
        c.move_and_slide(&mut w, Vec3::new(walk, dy, 0.0) * DT, DT);
        let after = w.linear_velocity(b).unwrap().x;
        assert!(
            after - before <= one_push + 1e-4,
            "step {i}: one move gave {} pushes' worth",
            (after - before) / one_push
        );
        w.step(DT);
        fastest = fastest.max(w.linear_velocity(b).unwrap().length());
    }
    assert!(fastest > 1.8, "it was pushed: {fastest} m/s");
    assert!(
        fastest <= walk + 1e-3,
        "no faster than the walk: {fastest} m/s"
    );
}

/// char-push-overspeed: a dynamic body the character stands still on can fall asleep (the old
/// controller pushed it down every step with the game's grounded fall speed, which woke it).
#[test]
fn a_body_the_character_stands_on_falls_asleep() {
    let mut w = PhysicsWorld::default();
    // A dynamic slab that cannot move at all: left alone it falls asleep in half a second.
    let slab = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, -0.25, 0.0),
            gravity_scale: 0.0,
            lock_translation: [true; 3],
            lock_rotation: [true; 3],
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        slab,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(3.0, 0.25, 3.0),
        }),
    )
    .unwrap();
    let mut c = Character::new(Vec3::new(0.0, stand() + 0.3, 0.0), settings());
    let mut vy = 0.0;
    for _ in 0..180 {
        let dy = fall(&c, &mut vy);
        c.move_and_slide(&mut w, Vec3::Y * (dy * DT), DT);
        w.step(DT);
    }
    assert!(c.grounded && c.ground_body == Some(slab), "standing on it");
    assert!(
        !w.is_awake(slab),
        "the slab fell asleep under the character"
    );
}

/// Runs a character standing on platform `p` for `steps`, moving it after the world steps
/// (`after`) or before; returns, for each move, whether it was grounded and the gap between its
/// feet and the platform's top (`top` above the platform's origin) after the move and after the
/// step.
fn ride(
    w: &mut PhysicsWorld,
    c: &mut Character,
    p: BodyId,
    top: f32,
    steps: usize,
    after: bool,
) -> Vec<(bool, f32, f32)> {
    let gap =
        |w: &PhysicsWorld, c: &Character| c.feet().y - (w.body_pose(p).unwrap().position.y + top);
    let mut vy = 0.0;
    let mut out = Vec::new();
    for _ in 0..steps {
        let dy = fall(c, &mut vy);
        if after {
            w.step(DT);
        }
        let r = c.move_and_slide(w, Vec3::Y * (dy * DT), DT);
        let moved = gap(w, c);
        if !after {
            w.step(DT);
        }
        out.push((r.grounded, moved, gap(w, c)));
    }
    out
}

/// char-rising-platform: riding a platform rising (or sinking) at 2 m/s, the character stays
/// grounded every move and a skin above it, moved after the step or before it. (It used to flick
/// between grounded and not, or with the real casts freeze inside the rising platform.) Moved
/// before the step it follows a step late: the platform rises one step into its skin and no
/// further.
#[test]
fn riding_a_rising_or_sinking_platform_stays_grounded_a_skin_above_it() {
    for speed in [2.0f32, -2.0] {
        for after in [true, false] {
            let mut w = PhysicsWorld::default();
            let p = platform(
                &mut w,
                Vec3::new(0.0, -0.25, 0.0),
                Vec3::new(3.0, 0.25, 3.0),
            );
            let mut c = Character::new(Vec3::new(0.0, stand() + 0.2, 0.0), settings());
            let skin = c.settings.skin;
            ride(&mut w, &mut c, p, 0.25, 30, after);
            w.set_linear_velocity(p, Vec3::new(0.0, speed, 0.0));
            for (i, (grounded, moved, stepped)) in ride(&mut w, &mut c, p, 0.25, 60, after)
                .into_iter()
                .enumerate()
            {
                let label = format!(
                    "{speed} m/s, moved {} the step, move {i}",
                    if after { "after" } else { "before" }
                );
                assert!(grounded, "{label}: not grounded");
                assert!(
                    (moved - skin).abs() < GAP,
                    "{label}: {moved} m above the platform"
                );
                assert!(
                    stepped > skin - (speed * DT).max(0.0) - GAP,
                    "{label}: the platform rose {} m into the capsule",
                    -stepped
                );
            }
        }
    }
}

/// char-turntable-drift: riding a platform turning at 1 rad/s for 20 s, the character stays over
/// the point of the platform it stood on: 2 m from the axis. (It drifted out to 2.36 m, each step a
/// straight line along the tangent.)
#[test]
fn riding_a_turntable_keeps_the_character_over_the_same_spot() {
    for after in [true, false] {
        let mut w = PhysicsWorld::default();
        let p = platform(
            &mut w,
            Vec3::new(0.0, -0.25, 0.0),
            Vec3::new(6.0, 0.25, 6.0),
        );
        let mut c = Character::new(Vec3::new(2.0, stand(), 0.0), settings());
        ride(&mut w, &mut c, p, 0.25, 10, after);
        let spot = w.body_pose(p).unwrap().inverse_point(c.position);
        w.set_angular_velocity(p, Vec3::new(0.0, 1.0, 0.0));
        let moves = ride(&mut w, &mut c, p, 0.25, 1200, after);
        let axis = Vec3::new(c.position.x, 0.0, c.position.z).length();
        assert!(
            (axis - 2.0).abs() < 1e-3,
            "moved after the step {after}: {axis} m from the axis"
        );
        if after {
            let now = w.body_pose(p).unwrap().point(spot);
            assert!(
                (now - c.position).length() < 2e-3,
                "over the same spot: {:?} vs {:?}",
                c.position,
                now
            );
        }
        for (i, (grounded, moved, _)) in moves.into_iter().enumerate() {
            assert!(grounded, "move {i}: not grounded");
            assert!(
                (moved - c.settings.skin).abs() < GAP,
                "move {i}: {moved} m above it"
            );
        }
    }
}

/// Riding is swept (past the platform itself): a platform passing under a wall leaves the
/// character at the wall, a skin from it, instead of carrying it into the wall.
#[test]
fn a_platform_carrying_the_character_into_a_wall_leaves_it_at_the_wall() {
    let mut w = flat();
    // A wall from 0.4 m up (the platform passes under it), its face at x = 3.
    static_box(
        &mut w,
        Vec3::new(3.5, 2.0, 0.0),
        Vec3::new(0.5, 1.6, 5.0),
        Quat::IDENTITY,
    );
    // A slab 0.2 m thick on the floor, its top at y = 0.2, sliding towards +X at 1 m/s.
    let p = platform(&mut w, Vec3::new(0.0, 0.1, 0.0), Vec3::new(1.5, 0.1, 1.5));
    let mut c = Character::new(Vec3::new(0.0, stand() + 0.2, 0.0), settings());
    ride(&mut w, &mut c, p, 0.1, 10, true);
    w.set_linear_velocity(p, Vec3::new(1.0, 0.0, 0.0));
    let mut vy = 0.0;
    for i in 0..240 {
        let dy = fall(&c, &mut vy);
        w.step(DT);
        c.move_and_slide(&mut w, Vec3::Y * (dy * DT), DT);
        let off_wall = 3.0 - (c.position.x + c.settings.radius);
        assert!(
            off_wall > c.settings.skin - GAP,
            "step {i}: {off_wall} m off the wall"
        );
    }
    let off_wall = 3.0 - (c.position.x + c.settings.radius);
    assert!(
        (off_wall - c.settings.skin).abs() < GAP,
        "left at the wall: {:?}",
        c.position
    );
}

/// No step length or displacement makes the character's position non-finite, riding or not.
#[test]
fn unusable_steps_and_displacements_leave_the_character_finite() {
    let mut w = PhysicsWorld::default();
    let p = platform(
        &mut w,
        Vec3::new(0.0, -0.25, 0.0),
        Vec3::new(3.0, 0.25, 3.0),
    );
    let mut c = Character::new(Vec3::new(0.0, stand(), 0.0), settings());
    ride(&mut w, &mut c, p, 0.25, 5, true);
    w.set_linear_velocity(p, Vec3::new(1.0, 0.0, 0.0));
    for dt in [f32::INFINITY, f32::NAN, 0.0, -DT, DT] {
        for d in [
            Vec3::ZERO,
            Vec3::new(f32::NAN, 0.0, 0.0),
            Vec3::new(f32::INFINITY, -1.0, 0.0),
            Vec3::new(0.01, -0.001, 0.0),
        ] {
            let before = c.position;
            let r = c.move_and_slide(&mut w, d, dt);
            assert!(
                c.position.is_finite() && r.moved.is_finite(),
                "dt {dt}, displacement {d:?}: {:?}",
                c.position
            );
            if !d.is_finite() {
                assert_eq!(
                    c.position, before,
                    "dt {dt}: a non-finite displacement moved it"
                );
            }
            w.step(DT);
        }
    }
}

/// The same moves on the same world give the same bits.
#[test]
fn identical_runs_give_identical_characters_and_worlds() {
    let run = || {
        let mut w = flat();
        wall(&mut w);
        static_box(
            &mut w,
            Vec3::new(-3.0, 0.125, 0.0),
            Vec3::new(1.0, 0.125, 5.0),
            Quat::IDENTITY,
        );
        let b = floating_box(&mut w, Vec3::splat(0.25), 3.0);
        let p = platform(&mut w, Vec3::new(0.0, -0.2, 6.0), Vec3::new(1.5, 0.2, 1.5));
        w.set_angular_velocity(p, Vec3::new(0.0, 0.7, 0.0));
        let mut c = Character::new(Vec3::new(0.0, stand() + 0.5, 0.0), settings());
        let mut vy = 0.0;
        let mut trace = Vec::new();
        for i in 0..300 {
            let a = i as f32 * 0.05;
            let walk = Vec3::new(2.0 - a * 0.2, 0.0, 1.0 - a * 0.1);
            let dy = fall(&c, &mut vy);
            w.step(DT);
            let r = c.move_and_slide(&mut w, walk * DT + Vec3::Y * (dy * DT), DT);
            trace.push((c.position, r.grounded, r.stepped, r.hits.len()));
        }
        (trace, w.state_hash(), w.body_pose(b))
    };
    assert!(run() == run());
}
