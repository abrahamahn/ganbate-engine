//! Pushing bodies the character walks into, caught off their centre of mass: the push is an
//! inelastic collision at the point touched, with the body's effective mass there, so the touched
//! point never leaves faster than the character walks along the push, however the body turns, and
//! the body's centre of mass never does either.

use ganbate_physics::gjk;
use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, Character, CharacterSettings, ColliderDesc, Convex, ConvexShape,
    HingeOptions, Iso, JointDesc, JointKind, PhysicsWorld, Quat, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;
const WALK: f32 = 1.5;

fn static_box(w: &mut PhysicsWorld, center: Vec3, half: Vec3) -> BodyId {
    let b = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: center,
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
    );
    w
}

fn stand() -> f32 {
    let s = CharacterSettings::default();
    s.half_height + s.radius + s.skin
}

/// A box body of `mass` kg.
fn box_body(w: &mut PhysicsWorld, desc: &BodyDesc, half: Vec3, mass: f32) -> BodyId {
    let b = w.create_body(desc).unwrap();
    let mut d = ColliderDesc::new(Shape::Box { half_extents: half });
    d.density = mass / (8.0 * half.x * half.y * half.z);
    w.add_collider(b, &d).unwrap();
    b
}

/// How much faster than the character (moving at `velocity`) the point of box `b` (half extents
/// `half`) nearest the capsule moves away from it, along the line between them: the point a push
/// just touched, when the capsule stopped against the box.
fn outrun(w: &PhysicsWorld, c: &Character, b: BodyId, half: Vec3, velocity: Vec3) -> f32 {
    let s = &c.settings;
    let capsule = Convex::new(
        ConvexShape::Capsule {
            half_height: s.half_height,
            radius: s.radius,
        },
        Iso::from_position(c.position),
    );
    let body = Convex::new(ConvexShape::Box(half), w.body_pose(b).unwrap());
    let near = gjk::distance(&capsule, &body);
    let v = w.point_velocity(b, near.point_b).unwrap();
    (v - velocity).dot(near.normal)
}

/// One move along +X or +Z at the walking speed with the game's vertical speed; returns the
/// character's velocity.
fn step_walk(w: &mut PhysicsWorld, c: &mut Character, direction: Vec3, vy: &mut f32) -> Vec3 {
    *vy = if c.grounded { -0.1 } else { *vy - 9.81 * DT };
    let velocity = direction * WALK + Vec3::Y * *vy;
    c.move_and_slide(w, velocity * DT, DT);
    velocity
}

/// A 1 kg box resting on the floor, walked into at 1.5 m/s: the push lands above its centre of
/// mass; the point it touched never leaves faster than the walk, and the box never moves faster
/// than the walk once the floor has had its say in the step, its spin turned to speed by
/// friction. (It reached 1.63 m/s after the step, spinning at 7 rad/s.)
#[test]
fn a_box_on_the_floor_pushed_above_its_centre_never_outruns_the_walker() {
    let mut w = flat();
    let half = Vec3::splat(0.25);
    let b = box_body(
        &mut w,
        &BodyDesc {
            position: Vec3::new(1.2, 0.25, 0.0),
            ..Default::default()
        },
        half,
        1.0,
    );
    for _ in 0..30 {
        w.step(DT);
    }
    let mut c = Character::new(Vec3::new(0.0, stand(), 0.0), CharacterSettings::default());
    let mut fastest = 0.0f32;
    let mut vy = 0.0f32;
    for i in 0..180 {
        let before = w.linear_velocity(b).unwrap();
        let velocity = step_walk(&mut w, &mut c, Vec3::X, &mut vy);
        if w.linear_velocity(b).unwrap() != before {
            let ahead = outrun(&w, &c, b, half, velocity);
            assert!(
                ahead <= 1e-3,
                "move {i}: the touched point leaves {ahead} m/s faster than the walk"
            );
        }
        w.step(DT);
        let v = w.linear_velocity(b).unwrap();
        let speed = Vec3::new(v.x, 0.0, v.z).length();
        fastest = fastest.max(speed);
        assert!(
            speed <= WALK + 1e-2,
            "move {i}: the box moves at {speed} m/s after the step"
        );
    }
    assert!(fastest > 0.5, "it was pushed: {fastest} m/s");
}

/// A floating 2 kg box whose lower edge the character's head catches, 0.25 m below the box's
/// centre: each push leaves the point it touched no faster than the walk along the push, and the
/// box's centre no faster than the walk. (The first push alone gave the box 8.3 rad/s, and the
/// touched edge moved at 3.5 m/s; now 3.6 rad/s and 1.54 m/s. Walking on into the turning box
/// keeps hitting it, and sets it turning faster, as a frictionless push would.)
#[test]
fn a_box_caught_below_its_centre_is_not_flung_by_its_edge() {
    let mut w = flat();
    let half = Vec3::splat(0.25);
    let b = box_body(
        &mut w,
        &BodyDesc {
            position: Vec3::new(0.6, 1.72, 0.0),
            gravity_scale: 0.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            can_sleep: false,
            ..Default::default()
        },
        half,
        2.0,
    );
    let mut c = Character::new(Vec3::new(0.0, stand(), 0.0), CharacterSettings::default());
    let mut pushes = 0;
    let mut vy = 0.0f32;
    for i in 0..60 {
        let before = w.linear_velocity(b).unwrap();
        let velocity = step_walk(&mut w, &mut c, Vec3::X, &mut vy);
        let v = w.linear_velocity(b).unwrap();
        if v != before {
            pushes += 1;
            let ahead = outrun(&w, &c, b, half, velocity);
            assert!(
                ahead <= 1e-3 && v.x <= WALK + 1e-3,
                "move {i}: the touched point leaves {ahead} m/s faster than the walk; the box \
                 moves at {v:?}"
            );
            if pushes == 1 {
                let spin = w.angular_velocity(b).unwrap().length();
                assert!(spin < 4.0, "the first push spun it at {spin} rad/s");
            }
        }
        w.step(DT);
    }
    assert!(pushes > 0, "it was pushed");
}

/// A door on a vertical hinge, walked into near its free edge: each push leaves the door's point
/// it touched no faster than the walk along the push, and once the hinge has had its say in the
/// step, the door's point at the character swings no faster than the walk (to within the 5% a
/// soft joint allows). (A 20 kg door's point there swung at 1.97 m/s, a 5 kg one's at 2.31 m/s,
/// at a 1.5 m/s walk.)
#[test]
fn a_hinged_door_is_not_swung_faster_than_the_walk() {
    for mass in [20.0f32, 5.0] {
        let mut w = flat();
        let post = w
            .create_body(&BodyDesc {
                kind: BodyKind::Static,
                position: Vec3::new(0.0, 1.0, 1.0),
                ..Default::default()
            })
            .unwrap();
        // A slab 1 m wide from the hinge at x = 0 to x = 1, its faces at z = 0.95 and 1.05.
        let half = Vec3::new(0.5, 1.0, 0.05);
        let door = box_body(
            &mut w,
            &BodyDesc {
                position: Vec3::new(0.5, 1.05, 1.0),
                gravity_scale: 0.0,
                linear_damping: 0.0,
                angular_damping: 0.0,
                can_sleep: false,
                ..Default::default()
            },
            half,
            mass,
        );
        let up = Quat::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_2);
        w.create_joint(&JointDesc {
            body_a: post,
            body_b: door,
            frame_a: Iso::new(Vec3::new(0.0, 0.05, 0.0), up),
            frame_b: Iso::new(Vec3::new(-0.5, 0.0, 0.0), up),
            kind: JointKind::Hinge(HingeOptions::default()),
            collide_connected: false,
            user_data: 0,
        })
        .unwrap();
        let mut c = Character::new(Vec3::new(0.75, stand(), 0.0), CharacterSettings::default());
        let mut swung = 0.0f32;
        let mut vy = 0.0f32;
        for i in 0..90 {
            let before = w.angular_velocity(door).unwrap();
            let velocity = step_walk(&mut w, &mut c, Vec3::Z, &mut vy);
            if w.angular_velocity(door).unwrap() != before {
                let ahead = outrun(&w, &c, door, half, velocity);
                assert!(
                    ahead <= 1e-3,
                    "{mass} kg, move {i}: the touched point leaves {ahead} m/s faster than the \
                     walk"
                );
            }
            w.step(DT);
            // The door's point nearest the character's axis, at its middle's height.
            let pose = w.body_pose(door).unwrap();
            let local = pose.inverse_point(Vec3::new(c.position.x, 1.05, c.position.z));
            let point = pose.point(Vec3::new(local.x.clamp(-0.5, 0.5), 0.0, 0.0));
            let speed = w.point_velocity(door, point).unwrap().length();
            swung = swung.max(speed);
            assert!(
                speed <= WALK * 1.05,
                "{mass} kg, move {i}: the door's point at the character swings at {speed} m/s"
            );
        }
        assert!(
            swung > 0.5,
            "{mass} kg: the door was pushed open: {swung} m/s"
        );
    }
}
