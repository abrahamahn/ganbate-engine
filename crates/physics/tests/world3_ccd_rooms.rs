//! Fast planks and capsules bouncing in a closed room of one-sided triangles escaped through it
//! about once in fifty throws. Wedged in a corner, a body's core sat at the sweep's target from a
//! wall (where the sweep of the step before had stopped it), so the next sweep's time of impact
//! was zero, which does not count; pushed on by friction, its centre went through the wall, and
//! the narrow phase drops a triangle whose plane the centre is behind. A centre never passes
//! through a one-sided triangle from its front now. Every scene here is synthetic.

use std::sync::Arc;

use ganbate_physics::mesh::TriangleMesh;
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

/// A closed 4 m cube of inward-facing triangles.
fn room() -> Shape {
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
    Shape::Mesh(Arc::new(
        TriangleMesh::new(v, outward.iter().map(|f| [f[0], f[2], f[1]]).collect()).unwrap(),
    ))
}

/// Planks thrown at 80 m/s (restitution 0.9, no gravity) in the closed mesh room, 96 throws: at
/// 4a2333a five of them escaped, after 5 to 17 steps. Every one now stays in.
#[test]
fn fast_planks_bouncing_in_a_closed_mesh_room_stay_in() {
    let mut escaped = Vec::new();
    for i in 0..96 {
        let mut w = PhysicsWorld::new(Settings {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        add_static(&mut w, Vec3::ZERO, room(), 0.6);
        let a = i as f32 * 0.7 + (i / 24) as f32 * 0.013;
        let (s, c) = ganbate_physics::math::sin_cos(a);
        let dir =
            Vec3::new(c, 0.6 * ganbate_physics::math::sin_cos(1.3 * a).0, s).normalize_or_zero();
        let b = w
            .create_body(&BodyDesc {
                rotation: Quat::from_axis_angle(Vec3::new(1.0, 0.3, 0.2).normalize_or_zero(), a),
                linear_velocity: dir * 80.0,
                ..Default::default()
            })
            .unwrap();
        let mut d = ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(0.05, 0.3, 0.05),
        });
        d.restitution = 0.9;
        w.add_collider(b, &d).unwrap();
        for step in 0..300 {
            w.step(DT);
            let p = w.body_pose(b).unwrap().position;
            if p.x.abs() > 2.0 || p.y.abs() > 2.0 || p.z.abs() > 2.0 {
                escaped.push(format!("throw {i}: out at step {step}, {p:?}"));
                break;
            }
        }
    }
    assert!(escaped.is_empty(), "{}", escaped.join("\n"));
}
