//! Continuous collision of elongated shapes already touching a wall: the contact at the near end
//! turns the approach into spin, the shape's own sweep starts touching (so it does not count),
//! and the core ball stops only the centre, with the far end deep in the wall. With the
//! velocity into the wall kept, the next step's core sweep started within its target (so it
//! did not count either) and the centre crossed: capsules and planks went through thin walls
//! and out of closed mesh rooms (review finding: the CCD doc claimed a body touching a thin
//! wall cannot pass through it). Every scene here is synthetic.

use std::sync::Arc;

use ganbate_physics::mesh::TriangleMesh;
use ganbate_physics::{
    BodyDesc, BodyKind, ColliderDesc, Iso, PhysicsWorld, Quat, Settings, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;

fn no_gravity() -> PhysicsWorld {
    PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    })
}

/// A wall whose front face is at x = 10, facing -x: a 2 cm static box, or a mesh quad.
fn wall(w: &mut PhysicsWorld, mesh: bool) {
    let b = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: if mesh {
                Vec3::ZERO
            } else {
                Vec3::new(10.01, 0.0, 0.0)
            },
            ..Default::default()
        })
        .unwrap();
    let shape = if mesh {
        let v = vec![
            Vec3::new(10.0, -5.0, -5.0),
            Vec3::new(10.0, 5.0, -5.0),
            Vec3::new(10.0, -5.0, 5.0),
            Vec3::new(10.0, 5.0, 5.0),
        ];
        Shape::Mesh(Arc::new(
            TriangleMesh::new(v, vec![[0, 2, 1], [1, 2, 3]]).unwrap(),
        ))
    } else {
        Shape::Box {
            half_extents: Vec3::new(0.01, 5.0, 5.0),
        }
    };
    w.add_collider(b, &ColliderDesc::new(shape)).unwrap();
}

/// Shoots `shape`, tilted about Z, along +X from 3 mm in front of the wall; returns where its
/// centre is after a second, and the farthest it got.
fn shoot(shape: &Shape, tilt: f32, speed: f32, mesh: bool) -> (f32, f32) {
    let mut w = no_gravity();
    wall(&mut w, mesh);
    let q = Quat::from_axis_angle(Vec3::Z, tilt.to_radians());
    let x = 10.0 - 0.003 - shape.aabb(&Iso::new(Vec3::ZERO, q)).max.x;
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(x, 0.0, 0.0),
            rotation: q,
            linear_velocity: Vec3::new(speed, 0.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(b, &ColliderDesc::new(shape.clone()))
        .unwrap();
    let mut farthest = f32::NEG_INFINITY;
    for _ in 0..60 {
        w.step(DT);
        farthest = farthest.max(w.body_pose(b).unwrap().position.x);
    }
    (w.body_pose(b).unwrap().position.x, farthest)
}

/// The reviewer's plank: half extents (0.05, 0.3, 0.05), tilted 10°, shot at 20 m/s. Its
/// centre ended at x = 12.69, behind the wall.
#[test]
fn a_tilted_plank_touching_a_thin_wall_does_not_pass_through() {
    let plank = Shape::Box {
        half_extents: Vec3::new(0.05, 0.3, 0.05),
    };
    for tilt in [-10.0, 10.0] {
        let (end, farthest) = shoot(&plank, tilt, 20.0, false);
        assert!(
            farthest < 10.0,
            "tilt {tilt}: the centre reached x = {farthest} and ended at {end}"
        );
    }
}

/// Capsules and planks at five tilts and three speeds, 3 mm from a thin box wall or a mesh
/// wall: 8 of the 60 went through (the reviewer counted 11 of 60 on their tree).
#[test]
fn elongated_shapes_touching_a_wall_never_pass_through_it() {
    let shapes = [
        (
            "capsule",
            Shape::Capsule {
                half_height: 0.2,
                radius: 0.06,
            },
        ),
        (
            "plank",
            Shape::Box {
                half_extents: Vec3::new(0.05, 0.3, 0.05),
            },
        ),
    ];
    let mut through = Vec::new();
    for mesh in [false, true] {
        for (name, shape) in &shapes {
            for tilt in [-30.0, -10.0, 0.0, 10.0, 30.0] {
                for speed in [10.0, 20.0, 50.0] {
                    let (end, farthest) = shoot(shape, tilt, speed, mesh);
                    if farthest >= 10.0 {
                        through.push(format!(
                            "{} {name} tilt {tilt} at {speed} m/s: x {farthest} (ends {end})",
                            if mesh { "mesh" } else { "box" }
                        ));
                    }
                }
            }
        }
    }
    assert!(
        through.is_empty(),
        "through the wall:\n{}",
        through.join("\n")
    );
}

/// The reviewer's room: a closed 4 m box of inward-facing triangles and a capsule bouncing in
/// it at 50 m/s with restitution 0.9 (12 of these 24 throws escaped).
#[test]
fn a_capsule_bouncing_fast_in_a_closed_mesh_room_stays_in() {
    let h = 2.0f32;
    let corner = |x: f32, y: f32, z: f32| Vec3::new(x * h, y * h, z * h);
    let v = vec![
        corner(-1., -1., -1.),
        corner(1., -1., -1.),
        corner(1., 1., -1.),
        corner(-1., 1., -1.),
        corner(-1., -1., 1.),
        corner(1., -1., 1.),
        corner(1., 1., 1.),
        corner(-1., 1., 1.),
    ];
    // A cube's faces wound outwards, reversed so that they face in.
    let outward: [[u32; 3]; 12] = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [1, 2, 6],
        [1, 6, 5],
        [0, 4, 7],
        [0, 7, 3],
    ];
    let room = Shape::Mesh(Arc::new(
        TriangleMesh::new(v, outward.iter().map(|f| [f[0], f[2], f[1]]).collect()).unwrap(),
    ));
    let mut escaped = Vec::new();
    for k in 0..24 {
        let mut w = no_gravity();
        let walls = w
            .create_body(&BodyDesc {
                kind: BodyKind::Static,
                ..Default::default()
            })
            .unwrap();
        w.add_collider(walls, &ColliderDesc::new(room.clone()))
            .unwrap();
        let a = k as f32 * 0.7;
        let (s, c) = ganbate_physics::math::sin_cos(a);
        let dir =
            Vec3::new(c, 0.6 * ganbate_physics::math::sin_cos(1.3 * a).0, s).normalize_or_zero();
        let b = w
            .create_body(&BodyDesc {
                rotation: Quat::from_axis_angle(Vec3::new(1.0, 0.3, 0.2).normalize_or_zero(), a),
                linear_velocity: dir * 50.0,
                ..Default::default()
            })
            .unwrap();
        let mut d = ColliderDesc::new(Shape::Capsule {
            half_height: 0.2,
            radius: 0.06,
        });
        d.restitution = 0.9;
        w.add_collider(b, &d).unwrap();
        for step in 0..300 {
            w.step(DT);
            let p = w.body_pose(b).unwrap().position;
            if p.x.abs() > h || p.y.abs() > h || p.z.abs() > h {
                escaped.push(format!("throw {k}: out at step {step}, {p:?}"));
                break;
            }
        }
    }
    assert!(escaped.is_empty(), "{}", escaped.join("\n"));
}

/// Only a core's stop removes velocity: a bouncy ball meeting a wall from afar is stopped by
/// its own sweep and still bounces.
#[test]
fn a_fast_ball_from_afar_still_bounces_off_a_wall() {
    let mut w = no_gravity();
    wall(&mut w, false);
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(5.0, 0.0, 0.0),
            linear_velocity: Vec3::new(120.0, 0.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    let mut d = ColliderDesc::new(Shape::Sphere { radius: 0.1 });
    d.restitution = 1.0;
    w.add_collider(b, &d).unwrap();
    for _ in 0..10 {
        w.step(DT);
    }
    let v = w.linear_velocity(b).unwrap();
    assert!(v.x < -100.0, "it should bounce back: {v:?}");
    assert!(w.body_pose(b).unwrap().position.x < 9.9);
}
