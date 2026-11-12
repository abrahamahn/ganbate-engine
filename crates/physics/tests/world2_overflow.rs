//! Finite inputs that overflowed: an impulse, force or torque whose effect is not a number, and
//! masses so large that momentum overflows. Each turned the world's poses into NaN within a step
//! (review finding: finite inputs still overflow to inf/NaN, a gap left by NAN-INPUTS). Every
//! scene here is synthetic.

use ganbate_physics::world::{MAX_BODY_MASS, MIN_BODY_MASS};
use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, Iso, PhysicsWorld, Settings, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;

fn all_finite(w: &PhysicsWorld) -> bool {
    w.bodies().all(|b| {
        let p = w.body_pose(b).unwrap();
        p.position.is_finite()
            && p.rotation.is_finite()
            && w.linear_velocity(b).unwrap().is_finite()
            && w.angular_velocity(b).unwrap().is_finite()
    })
}

fn no_gravity() -> PhysicsWorld {
    PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    })
}

/// A ball of radius 1 cm and density 1: 4.2e-6 kg.
fn speck(w: &mut PhysicsWorld) -> BodyId {
    let b = w.create_body(&BodyDesc::default()).unwrap();
    let mut d = ColliderDesc::new(Shape::Sphere { radius: 0.01 });
    d.density = 1.0;
    w.add_collider(b, &d).unwrap();
    b
}

#[test]
fn an_impulse_too_large_for_a_light_body_is_refused() {
    let mut w = no_gravity();
    let b = speck(&mut w);
    let c = w.center_of_mass(b).unwrap();
    w.apply_impulse_at(b, Vec3::new(1e38, 0.0, 0.0), c);
    assert_eq!(w.linear_velocity(b), Some(Vec3::ZERO), "refused whole");
    // Off the centre, the turn overflows too.
    w.apply_impulse_at(b, Vec3::new(0.0, 1e30, 0.0), c + Vec3::new(1e10, 0.0, 0.0));
    w.apply_angular_impulse(b, Vec3::new(3e38, -3e38, 1e38));
    assert_eq!(w.angular_velocity(b), Some(Vec3::ZERO));
    // A large one that is still a number is applied; the step clamps the speed.
    w.apply_impulse_at(b, Vec3::new(1e3, 0.0, 0.0), c);
    assert!(w.linear_velocity(b).unwrap().x > 1e8);
    w.step(DT);
    assert!(all_finite(&w));
    let v = w.linear_velocity(b).unwrap();
    assert!((v.x - w.settings.max_linear_speed).abs() < 1e-3, "{v:?}");
}

#[test]
fn forces_and_torques_that_add_up_past_the_largest_number_are_refused() {
    let mut w = no_gravity();
    let b = speck(&mut w);
    let c = w.center_of_mass(b).unwrap();
    w.apply_force(b, Vec3::new(3e38, 0.0, 0.0));
    w.apply_force(b, Vec3::new(3e38, 0.0, 0.0));
    w.apply_torque(b, Vec3::new(0.0, 3e38, 0.0));
    w.apply_torque(b, Vec3::new(0.0, 3e38, 0.0));
    w.apply_force_at(b, Vec3::new(0.0, 0.0, 1e30), c + Vec3::new(1e10, 0.0, 0.0));
    w.step(DT);
    assert!(all_finite(&w));
    // The accepted force (3e38 N on 4.2e-6 kg) is far more than the top speed allows: the
    // body leaves along +X at the top speed, and turns at the top angular speed about +Y.
    let v = w.linear_velocity(b).unwrap();
    assert!(
        (v - Vec3::new(w.settings.max_linear_speed, 0.0, 0.0)).length() < 1e-2,
        "{v:?}"
    );
    let spin = w.angular_velocity(b).unwrap();
    assert!(
        spin.y > 100.0 && spin.x.abs() < 1e-3 && spin.z.abs() < 1e-3,
        "{spin:?}"
    );
}

/// The torque of a large force at a far point, turned by an inverse inertia with off-diagonal
/// terms, overflowed as ∞ − ∞: NaN. It turns the body as fast as the step allows instead.
#[test]
fn a_torque_whose_turn_overflows_turns_the_body_at_the_top_angular_speed() {
    let mut w = no_gravity();
    let b = w
        .create_body(&BodyDesc {
            rotation: ganbate_physics::Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0), 0.7),
            ..Default::default()
        })
        .unwrap();
    let mut d = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::new(0.01, 0.02, 0.05),
    });
    d.density = 1.0;
    w.add_collider(b, &d).unwrap();
    w.apply_torque(b, Vec3::new(3e38, 3e38, -3e38));
    w.step(DT);
    assert!(all_finite(&w));
    assert!(w.angular_velocity(b).unwrap().length() > 100.0);
}

/// A dense cube (2.4e38 kg) sunk into the ground and a kinematic platform: the contact
/// impulse overflowed, and the platform's zero inverse mass times infinity was NaN.
#[test]
fn a_body_heavier_than_the_largest_mass_weighs_the_largest_mass() {
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
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        }),
    )
    .unwrap();
    let platform = w
        .create_body(&BodyDesc {
            kind: BodyKind::Kinematic,
            position: Vec3::new(0.0, 2.0, 0.0),
            linear_velocity: Vec3::new(0.0, -1.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        platform,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(2.0, 0.2, 2.0),
        }),
    )
    .unwrap();
    let cube = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, 0.9, 0.0),
            ..Default::default()
        })
        .unwrap();
    let mut dense = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::ONE,
    });
    dense.density = 3e37;
    w.add_collider(cube, &dense).unwrap();
    for step in 0..30 {
        w.step(DT);
        assert!(all_finite(&w), "step {step}");
    }
    // Pressed between the ground and the platform; the platform is never pushed.
    assert_eq!(w.linear_velocity(platform), Some(Vec3::new(0.0, -1.0, 0.0)));
    assert_eq!(w.mass(cube), Some(MAX_BODY_MASS));
}

/// A body lighter than about 3e-39 kg (a tiny density) was refused as if its mass overflowed:
/// its centre of mass came out 0 × ∞ = NaN. It now weighs the smallest mass, same shape.
#[test]
fn a_body_lighter_than_the_smallest_mass_weighs_the_smallest_mass() {
    let mut w = PhysicsWorld::default();
    let b = w.create_body(&BodyDesc::default()).unwrap();
    let mut feather = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::new(0.5, 0.1, 0.25),
    });
    feather.density = 1e-40;
    feather.local = Iso::from_position(Vec3::new(0.3, 0.0, 0.0));
    w.add_collider(b, &feather).unwrap();
    let m = w.mass(b).unwrap();
    assert!((m - MIN_BODY_MASS).abs() < 1e-3 * MIN_BODY_MASS, "{m}");
    // The same shape at the smallest mass: the centre of mass stays in the box.
    let c = w.center_of_mass(b).unwrap();
    assert!((c - Vec3::new(0.3, 0.0, 0.0)).length() < 1e-5, "{c:?}");
    w.apply_impulse_at(b, Vec3::new(0.0, 1e-21, 0.0), c + Vec3::new(0.4, 0.0, 0.0));
    for _ in 0..60 {
        w.step(DT);
        assert!(all_finite(&w));
    }
    // It falls like anything else and turns from the off-centre push.
    assert!(w.body_pose(b).unwrap().position.y < -4.0);
    assert!(w.angular_velocity(b).unwrap().length() > 0.01);
}

/// Masses in the range are kept exactly.
#[test]
fn masses_in_range_are_kept() {
    let mut w = PhysicsWorld::default();
    let b = w.create_body(&BodyDesc::default()).unwrap();
    let mut d = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::ONE,
    });
    d.density = 1e20;
    w.add_collider(b, &d).unwrap();
    assert_eq!(w.mass(b), Some(8e20));
}
