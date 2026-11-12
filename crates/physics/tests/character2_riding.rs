//! Riding what the character stands on: it follows the body's motion, and only that. A body that
//! is teleported (`set_body_pose`) does not take the character along its teleport, and nothing in
//! the way is pushed by it.

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

/// A kinematic slab 0.2 m thick lying on the floor at `x` (its top at y = 0.2), never asleep.
fn slab(w: &mut PhysicsWorld, x: f32) -> BodyId {
    let p = w
        .create_body(&BodyDesc {
            kind: BodyKind::Kinematic,
            position: Vec3::new(x, 0.1, 0.0),
            can_sleep: false,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        p,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(1.0, 0.1, 1.0),
        }),
    )
    .unwrap();
    p
}

fn stand() -> f32 {
    let s = CharacterSettings::default();
    s.half_height + s.radius + s.skin
}

/// One game step: the world steps, then the character moves (standing still, or falling).
fn tick(w: &mut PhysicsWorld, c: &mut Character, vy: &mut f32) -> ganbate_physics::MoveResult {
    *vy = if c.grounded { -0.1 } else { *vy - 9.81 * DT };
    w.step(DT);
    c.move_and_slide(w, Vec3::Y * (*vy * DT), DT)
}

/// A character standing on a slab at the origin, `x` and `z` from its middle, settled.
fn on_slab(w: &mut PhysicsWorld, x: f32, z: f32) -> (Character, BodyId, f32) {
    let p = slab(w, 0.0);
    let mut c = Character::new(Vec3::new(x, 0.2 + stand(), z), CharacterSettings::default());
    let mut vy = 0.0;
    for _ in 0..20 {
        tick(w, &mut c, &mut vy);
    }
    assert!(
        c.grounded && c.ground_body == Some(p),
        "standing on the slab"
    );
    (c, p, vy)
}

/// The slab it stands on is teleported 20 m away, across a 20° ramp: the character stays where it
/// was and drops onto the floor below. (It used to be swept along the teleport, up the ramp, and
/// left 5.8 m up in the air.)
#[test]
fn a_teleported_platform_leaves_the_character_behind() {
    let mut w = flat();
    // A 20° ramp rising towards +X from x = 5.
    let a = 20.0f32.to_radians();
    let half = Vec3::new(2.5, 0.25, 3.0);
    let center = Vec3::new(
        5.0 + half.x * a.cos() + half.y * a.sin(),
        half.x * a.sin() - half.y * a.cos(),
        0.0,
    );
    static_box(&mut w, center, half, Quat::from_axis_angle(Vec3::Z, a));
    let (mut c, p, mut vy) = on_slab(&mut w, 0.0, 0.0);
    let before = c.position;
    w.set_body_pose(p, Iso::from_position(Vec3::new(20.0, 0.1, 0.0)));
    let r = tick(&mut w, &mut c, &mut vy);
    assert!(
        Vec3::new(r.moved.x, 0.0, r.moved.z).length() < 1e-4,
        "not carried along the teleport: moved {:?}",
        r.moved
    );
    assert!(
        c.position.y <= before.y + 1e-4,
        "not lifted: {:?}",
        c.position
    );
    for _ in 0..30 {
        tick(&mut w, &mut c, &mut vy);
    }
    assert!(
        c.grounded && (c.position.y - stand()).abs() < 5e-4 && c.position.x.abs() < 1e-4,
        "on the floor where it stood: {:?}",
        c.position
    );
}

/// The slab it stands on is teleported 20 m away through a 10 kg crate: the crate is not hit.
/// (The character was dragged through it and threw it at over a thousand metres a second.)
#[test]
fn a_teleported_platform_throws_nothing_in_its_way() {
    let mut w = flat();
    let crate_ = w
        .create_body(&BodyDesc {
            position: Vec3::new(8.0, 0.5, 0.0),
            ..Default::default()
        })
        .unwrap();
    let mut d = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::splat(0.5),
    });
    d.density = 10.0;
    w.add_collider(crate_, &d).unwrap();
    let (mut c, p, mut vy) = on_slab(&mut w, 0.0, 0.0);
    for _ in 0..30 {
        tick(&mut w, &mut c, &mut vy);
    }
    w.set_body_pose(p, Iso::from_position(Vec3::new(20.0, 0.1, 0.0)));
    let r = tick(&mut w, &mut c, &mut vy);
    let v = w.linear_velocity(crate_).unwrap();
    assert!(v.length() < 0.05, "the crate was thrown at {v:?}: {r:?}");
    assert!(
        Vec3::new(r.moved.x, 0.0, r.moved.z).length() < 1e-4,
        "not carried along the teleport: moved {:?}",
        r.moved
    );
}

/// An explicit teleport can be smaller than the skin, or comparable to one frame's travel.
#[test]
fn small_teleports_leave_the_rider_at_its_world_position() {
    for (jump, speed) in [(0.005, 0.0), (0.05, 6.0), (0.0, 6.0)] {
        let mut w = flat();
        let (mut c, p, mut vy) = on_slab(&mut w, 0.0, 0.0);
        w.set_linear_velocity(p, Vec3::X * speed);
        tick(&mut w, &mut c, &mut vy);
        let before = c.position;
        // The platform has moved since the last observation, so the old velocity-based
        // heuristic explained even a 5 cm jump. Setting the current pose also breaks footing.
        w.step(DT);
        let mut pose = w.body_pose(p).unwrap();
        pose.position.x += jump;
        w.set_body_pose(p, pose);
        // Observe the teleport before any further platform motion.
        let r = c.move_and_slide(&mut w, Vec3::ZERO, DT);
        assert!(r.grounded, "the slab is still underneath");
        assert!(
            (c.position.x - before.x).abs() < 1e-5,
            "{jump} m teleport carried the rider: {r:?}"
        );
        // From its new footing, the rider follows normal motion again.
        tick(&mut w, &mut c, &mut vy);
        assert!((c.position.x - before.x - speed * DT).abs() < 1e-4);
    }
}

/// The slab it stands on is turned a quarter round at once, or lifted 3 m at once: neither carries
/// the character.
#[test]
fn a_platform_turned_or_lifted_at_once_does_not_carry_the_character() {
    for pose in [
        Iso::new(
            Vec3::new(0.0, 0.1, 0.0),
            Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2),
        ),
        Iso::from_position(Vec3::new(0.0, 3.1, 0.0)),
    ] {
        let mut w = flat();
        // Off the slab's middle, so turning it would carry the character round.
        let (mut c, p, mut vy) = on_slab(&mut w, 0.6, 0.3);
        let before = c.position;
        w.set_body_pose(p, pose);
        let r = tick(&mut w, &mut c, &mut vy);
        assert!(
            Vec3::new(r.moved.x, 0.0, r.moved.z).length() < 1e-4 && c.position.y <= before.y,
            "{pose:?}: moved {:?} to {:?}",
            r.moved,
            c.position
        );
    }
}

/// A platform moved by its velocity is still ridden exactly, also when the game moves the
/// character only every other step and when the platform changes speed: only a pose change its
/// motion cannot explain is a teleport.
#[test]
fn a_moving_platform_is_ridden_at_any_move_rate() {
    for every in [1usize, 2, 3] {
        let mut w = flat();
        let (mut c, p, mut vy) = on_slab(&mut w, 0.5, 0.3);
        let spot = w.body_pose(p).unwrap().inverse_point(c.position);
        for i in 0..120 {
            // Speeding up from 0 to 6 m/s along +Z and turning.
            let speed = 0.05 * i as f32;
            w.set_linear_velocity(p, Vec3::new(0.0, 0.0, speed));
            w.set_angular_velocity(p, Vec3::new(0.0, 0.5, 0.0));
            w.step(DT);
            if i % every == every - 1 {
                vy = if c.grounded { -0.1 } else { vy - 9.81 * DT };
                let r = c.move_and_slide(&mut w, Vec3::Y * (vy * DT), DT);
                assert!(r.grounded, "every {every}, step {i}: not grounded");
                let now = w.body_pose(p).unwrap().point(spot);
                assert!(
                    (now - c.position).length() < 2e-3,
                    "every {every}, step {i}: {:?} is not over its spot {:?}",
                    c.position,
                    now
                );
            }
        }
    }
}
