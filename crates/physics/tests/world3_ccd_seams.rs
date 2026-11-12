//! Continuous collision over tiled floors and against off-centre hits (review finding
//! CCD-TILE-SEAMS). A body resting on one of two flush static boxes has sunk a fraction of a
//! millimetre into it, so its front face meets the side of the next box at the seam. Continuous
//! collision took that time of impact and stopped the body 5 mm short of every seam, where the
//! contacts then held it: planks and cubes gliding over a tiled floor stopped dead. Only a target
//! the body's inner shape (the body less a skin of the linear slop, its front kept) reaches is
//! taken now; the off-centre hits that must still stop a body (a wall under one end of a rod, the
//! edge of a plate in a ball's path) stay stopped. Every scene here is synthetic.

use std::sync::Arc;

use ganbate_physics::hull::ConvexHull;
use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, PhysicsWorld, Quat, Settings, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;

fn add_static(w: &mut PhysicsWorld, at: Vec3, shape: Shape, friction: f32) -> BodyId {
    let b = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: at,
            ..Default::default()
        })
        .unwrap();
    let mut d = ColliderDesc::new(shape);
    d.friction = friction;
    w.add_collider(b, &d).unwrap();
    b
}

/// Sixteen frictionless boxes 1 m wide along X with their tops at y = 0: seams at whole x.
fn tiles(w: &mut PhysicsWorld) {
    for i in 0..16 {
        add_static(
            w,
            Vec3::new(i as f32 - 7.5, -0.5, 0.0),
            Shape::Box {
                half_extents: Vec3::new(0.5, 0.5, 20.0),
            },
            0.0,
        );
    }
}

/// How far a frictionless body set down on the tiles at `x` (centre `rest` above them) travels in
/// a second once given `speed` along X, and its speed then.
fn glide(shape: &Shape, rest: f32, speed: f32, x: f32, continuous: bool) -> (f32, f32) {
    let mut w = PhysicsWorld::new(Settings {
        enable_continuous: continuous,
        ..Default::default()
    });
    tiles(&mut w);
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(x, rest, 0.0),
            can_sleep: false,
            ..Default::default()
        })
        .unwrap();
    let mut d = ColliderDesc::new(shape.clone());
    d.friction = 0.0;
    w.add_collider(b, &d).unwrap();
    for _ in 0..30 {
        w.step(DT);
    }
    let x0 = w.body_pose(b).unwrap().position.x;
    w.set_linear_velocity(b, Vec3::new(speed, 0.0, 0.0));
    for _ in 0..60 {
        w.step(DT);
    }
    (
        w.body_pose(b).unwrap().position.x - x0,
        w.linear_velocity(b).unwrap().x,
    )
}

/// The gliding bodies: name, shape, resting height of the centre and speed.
fn gliders() -> Vec<(&'static str, Shape, f32, f32)> {
    let slab = ConvexHull::from_points(&[
        Vec3::new(-0.3, -0.03, -0.2),
        Vec3::new(0.3, -0.03, -0.2),
        Vec3::new(-0.3, 0.03, -0.2),
        Vec3::new(0.3, 0.03, -0.2),
        Vec3::new(-0.3, -0.03, 0.2),
        Vec3::new(0.3, -0.03, 0.2),
        Vec3::new(-0.3, 0.03, 0.2),
        Vec3::new(0.3, 0.03, 0.2),
    ])
    .unwrap();
    vec![
        (
            "plank 0.6 x 0.04 x 0.6",
            Shape::Box {
                half_extents: Vec3::new(0.3, 0.02, 0.3),
            },
            0.02,
            4.0,
        ),
        (
            "cube 0.2",
            Shape::Box {
                half_extents: Vec3::splat(0.1),
            },
            0.1,
            4.0,
        ),
        (
            "cube 0.2",
            Shape::Box {
                half_extents: Vec3::splat(0.1),
            },
            0.1,
            8.0,
        ),
        ("ball r 0.1", Shape::Sphere { radius: 0.1 }, 0.1, 6.0),
        (
            "upright capsule r 0.1",
            Shape::Capsule {
                half_height: 0.2,
                radius: 0.1,
            },
            0.3,
            6.0,
        ),
        (
            "hull slab 0.6 x 0.06 x 0.4",
            Shape::Hull(Arc::new(slab)),
            0.03,
            4.0,
        ),
    ]
}

/// The reviewer's runs from x = -6: with continuous collision on, the plank travelled 0.709 m of
/// 4 and ended still, the cubes 1.893 of 4 and 3.287 of 8 m, the ball 5.255 of 6 m. Without it,
/// each glides the whole way.
#[test]
fn bodies_glide_over_a_tiled_floor_with_continuous_collision_on() {
    for (name, shape, rest, speed) in gliders() {
        let (off, _) = glide(&shape, rest, speed, -6.0, false);
        let (on, v) = glide(&shape, rest, speed, -6.0, true);
        assert!(
            (off - speed).abs() < 0.01 && (on - off).abs() < 1e-3 && (v - speed).abs() < 0.01,
            "{name} at {speed} m/s: travelled {on} m (ends at {v} m/s) with continuous collision, \
             {off} m without"
        );
    }
}

/// Whatever the phase of the seams against the steps, wherever its contacts let a glider cross
/// every seam (they sometimes catch it at one on their own: a ghost contact between separate
/// boxes, the narrow phase's, which can throw it into a tumbling hop), continuous collision lets
/// it cross too. At 4a2333a it stopped every one of them at the first seam.
#[test]
fn continuous_collision_adds_no_stops_at_seams() {
    let mut worse = Vec::new();
    let mut free = 0;
    for (name, shape, rest, speed) in gliders() {
        for k in 0..10 {
            let x = -6.0 + 0.05 * k as f32;
            let glides = |(travelled, v): (f32, f32)| {
                (travelled - speed).abs() < 0.01 && (v - speed).abs() < 0.01
            };
            if !glides(glide(&shape, rest, speed, x, false)) {
                continue;
            }
            free += 1;
            let (on, v) = glide(&shape, rest, speed, x, true);
            if !glides((on, v)) {
                worse.push(format!(
                    "{name} at {speed} m/s from x = {x}: {on} m (ends at {v} m/s) with continuous \
                     collision, all the way without"
                ));
            }
        }
    }
    assert!(
        free >= 30,
        "only {free} of 60 runs glide freely without continuous collision"
    );
    assert!(worse.is_empty(), "{}", worse.join("\n"));
}

/// Off-centre hits on thin static boxes, each from 20 phases of the step (no gravity): the body's
/// side runs into the box while its centre passes beside it. None may pass through it untouched.
/// (The verifier of the first fix proposed, a precondition on the core ball alone, found 13 to
/// 18 of 20 passing through.)
#[test]
fn off_centre_hits_on_thin_walls_are_still_stopped() {
    struct Case {
        label: &'static str,
        shape: Shape,
        rotation: Quat,
        /// How far the body reaches behind its centre along +Z.
        back: f32,
        speed: f32,
        wall_at: Vec3,
        wall_half: Vec3,
    }
    let cases = [
        Case {
            label: "rod 1 m x 4 cm broadside, a wall under its end",
            shape: Shape::Box {
                half_extents: Vec3::new(0.5, 0.02, 0.02),
            },
            rotation: Quat::IDENTITY,
            back: 0.02,
            speed: 30.0,
            wall_at: Vec3::new(0.5, 0.0, 1.0),
            wall_half: Vec3::new(0.3, 1.0, 0.01),
        },
        Case {
            label: "plank flat, a 10 cm beam under its end",
            shape: Shape::Box {
                half_extents: Vec3::new(0.3, 0.3, 0.02),
            },
            rotation: Quat::IDENTITY,
            back: 0.02,
            speed: 12.0,
            wall_at: Vec3::new(0.2, 0.0, 1.0),
            wall_half: Vec3::new(0.05, 1.0, 0.01),
        },
        Case {
            label: "ball r 0.1, a plate's edge 5 cm into its path",
            shape: Shape::Sphere { radius: 0.1 },
            rotation: Quat::IDENTITY,
            back: 0.0866,
            speed: 30.0,
            wall_at: Vec3::new(1.05, 0.0, 1.0),
            wall_half: Vec3::new(1.0, 1.0, 0.01),
        },
        Case {
            label: "cube 0.2, a wall's edge 6 cm into its path",
            shape: Shape::Box {
                half_extents: Vec3::splat(0.1),
            },
            rotation: Quat::IDENTITY,
            back: 0.1,
            speed: 30.0,
            wall_at: Vec3::new(1.04, 0.0, 1.0),
            wall_half: Vec3::new(1.0, 1.0, 0.01),
        },
        Case {
            label: "capsule lying along X broadside, a wall under its end",
            shape: Shape::Capsule {
                half_height: 0.4,
                radius: 0.05,
            },
            rotation: Quat::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_2),
            back: 0.05,
            speed: 30.0,
            wall_at: Vec3::new(0.6, 0.0, 1.0),
            wall_half: Vec3::new(0.3, 1.0, 0.01),
        },
        Case {
            label: "hull rod 1 m x 4 cm broadside, a wall under its last 10 cm",
            shape: Shape::Hull(Arc::new(ConvexHull::cuboid(Vec3::new(0.5, 0.02, 0.02)))),
            rotation: Quat::IDENTITY,
            back: 0.02,
            speed: 30.0,
            wall_at: Vec3::new(0.6, 0.0, 1.0),
            wall_half: Vec3::new(0.2, 1.0, 0.01),
        },
    ];
    let mut through = Vec::new();
    for c in &cases {
        for k in 0..20 {
            let mut w = PhysicsWorld::new(Settings {
                gravity: Vec3::ZERO,
                ..Default::default()
            });
            add_static(
                &mut w,
                c.wall_at,
                Shape::Box {
                    half_extents: c.wall_half,
                },
                0.5,
            );
            let b = w
                .create_body(&BodyDesc {
                    position: Vec3::new(0.0, 0.0, -(k as f32) * c.speed * DT / 20.0),
                    rotation: c.rotation,
                    linear_velocity: Vec3::new(0.0, 0.0, c.speed),
                    ..Default::default()
                })
                .unwrap();
            w.add_collider(b, &ColliderDesc::new(c.shape.clone()))
                .unwrap();
            // Past the wall's far face with its velocity and line unchanged: through it untouched
            // (a body that hit the wall is turned aside and may slide on past its end).
            let far = c.wall_at.z + c.wall_half.z;
            for _ in 0..12 {
                w.step(DT);
                let p = w.body_pose(b).unwrap().position;
                if p.z - c.back > far {
                    let v = w.linear_velocity(b).unwrap();
                    if (v - Vec3::new(0.0, 0.0, c.speed)).length() < 1e-3 && p.x.abs() < 1e-4 {
                        through.push(format!("{} (phase {k}): at {p:?}", c.label));
                    }
                    break;
                }
            }
        }
    }
    assert!(through.is_empty(), "through:\n{}", through.join("\n"));
}

/// A 1 m x 4 cm rod shot tip-first into a 4 cm wall, from 12 phases of the step at each speed:
/// its tip never gets past the wall's near face.
#[test]
fn a_rod_shot_tip_first_stops_at_the_wall() {
    for speed in [20.0f32, 40.0, 60.0, 120.0] {
        for k in 0..12 {
            let mut w = PhysicsWorld::new(Settings {
                gravity: Vec3::ZERO,
                ..Default::default()
            });
            add_static(
                &mut w,
                Vec3::new(5.02, 0.0, 0.0),
                Shape::Box {
                    half_extents: Vec3::new(0.02, 2.0, 2.0),
                },
                0.5,
            );
            let b = w
                .create_body(&BodyDesc {
                    position: Vec3::new(3.0 - k as f32 * speed * DT / 12.0, 0.0, 0.0),
                    linear_velocity: Vec3::new(speed, 0.0, 0.0),
                    ..Default::default()
                })
                .unwrap();
            w.add_collider(
                b,
                &ColliderDesc::new(Shape::Box {
                    half_extents: Vec3::new(0.5, 0.02, 0.02),
                }),
            )
            .unwrap();
            for step in 0..30 {
                w.step(DT);
                let tip = w.body_pose(b).unwrap().position.x + 0.5;
                assert!(
                    tip < 5.0 + 1e-3,
                    "{speed} m/s, phase {k}, step {step}: the tip is {} m past the wall's face",
                    tip - 5.0
                );
            }
        }
    }
}
