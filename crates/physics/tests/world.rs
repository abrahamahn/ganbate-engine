//! Whole-world behaviour: falling, resting, stacking, friction, bouncing, events, queries,
//! filters, sleeping, continuous collision and determinism. Every scene here is synthetic.

use std::sync::Arc;

use ganbate_physics::hull::ConvexHull;
use ganbate_physics::mesh::{Heightfield, TriangleMesh};
use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, ContactEvent, Iso, PhysicsWorld, Quat, QueryFilter,
    SensorEvent, Settings, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;

fn ground(world: &mut PhysicsWorld) -> BodyId {
    let g = world
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(0.0, -0.5, 0.0),
            ..Default::default()
        })
        .unwrap();
    world
        .add_collider(
            g,
            &ColliderDesc::new(Shape::Box {
                half_extents: Vec3::new(50.0, 0.5, 50.0),
            }),
        )
        .unwrap();
    g
}

fn dynamic(world: &mut PhysicsWorld, shape: Shape, position: Vec3) -> BodyId {
    let b = world
        .create_body(&BodyDesc {
            position,
            ..Default::default()
        })
        .unwrap();
    world.add_collider(b, &ColliderDesc::new(shape)).unwrap();
    b
}

fn cube(half: f32) -> Shape {
    Shape::Box {
        half_extents: Vec3::splat(half),
    }
}

fn run(world: &mut PhysicsWorld, seconds: f32) {
    let steps = (seconds / DT).round() as usize;
    for _ in 0..steps {
        world.step(DT);
    }
}

fn y(world: &PhysicsWorld, b: BodyId) -> f32 {
    world.body_pose(b).unwrap().position.y
}

#[test]
fn a_falling_body_follows_the_parabola() {
    let mut w = PhysicsWorld::default();
    let b = dynamic(
        &mut w,
        Shape::Sphere { radius: 0.5 },
        Vec3::new(0.0, 100.0, 0.0),
    );
    run(&mut w, 1.0);
    let expect = 100.0 - 0.5 * 9.81;
    assert!(
        (y(&w, b) - expect).abs() < 0.05,
        "y {} expected {expect}",
        y(&w, b)
    );
    assert!((w.linear_velocity(b).unwrap().y + 9.81).abs() < 0.01);
}

#[test]
fn a_box_comes_to_rest_on_the_ground_and_falls_asleep() {
    let mut w = PhysicsWorld::default();
    ground(&mut w);
    let b = dynamic(&mut w, cube(0.5), Vec3::new(0.0, 2.0, 0.0));
    run(&mut w, 3.0);
    let p = w.body_pose(b).unwrap();
    assert!(
        (p.position.y - 0.5).abs() < 0.01,
        "rests on its face: {:?}",
        p.position
    );
    assert!(
        p.position.x.abs() < 1e-3 && p.position.z.abs() < 1e-3,
        "does not wander: {:?}",
        p.position
    );
    assert!(w.linear_velocity(b).unwrap().length() < 0.01);
    assert!(!w.is_awake(b), "a still box falls asleep");
}

#[test]
fn a_tall_stack_of_boxes_stands() {
    let mut w = PhysicsWorld::default();
    ground(&mut w);
    let boxes: Vec<BodyId> = (0..10)
        .map(|i| dynamic(&mut w, cube(0.5), Vec3::new(0.0, 0.5 + i as f32 * 1.0, 0.0)))
        .collect();
    run(&mut w, 5.0);
    for (i, &b) in boxes.iter().enumerate() {
        let p = w.body_pose(b).unwrap().position;
        assert!(
            p.x.abs() < 0.02 && p.z.abs() < 0.02,
            "box {i} drifted to {p:?}"
        );
        assert!(
            (p.y - (0.5 + i as f32)).abs() < 0.06,
            "box {i} at height {}",
            p.y
        );
    }
}

#[test]
fn a_pyramid_of_boxes_stands() {
    let mut w = PhysicsWorld::default();
    ground(&mut w);
    let base = 6;
    let mut all = Vec::new();
    for row in 0..base {
        for i in 0..(base - row) {
            let x = (i as f32 - (base - row - 1) as f32 * 0.5) * 1.05;
            all.push((
                dynamic(&mut w, cube(0.5), Vec3::new(x, 0.5 + row as f32, 0.0)),
                x,
                row,
            ));
        }
    }
    run(&mut w, 5.0);
    for (b, x, row) in all {
        let p = w.body_pose(b).unwrap().position;
        assert!(
            (p.x - x).abs() < 0.05 && (p.y - (0.5 + row as f32)).abs() < 0.08,
            "box at row {row} moved to {p:?}"
        );
    }
}

#[test]
fn friction_holds_a_box_on_a_gentle_slope_and_lets_it_slide_on_a_steep_one() {
    for (degrees, friction, should_slide) in [(20.0f32, 0.6f32, false), (40.0, 0.2, true)] {
        let mut w = PhysicsWorld::default();
        let angle = degrees.to_radians();
        let slope = w
            .create_body(&BodyDesc {
                kind: BodyKind::Static,
                rotation: Quat::from_axis_angle(Vec3::Z, angle),
                ..Default::default()
            })
            .unwrap();
        let mut d = ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(20.0, 0.5, 5.0),
        });
        d.friction = friction;
        w.add_collider(slope, &d).unwrap();
        let up = Quat::from_axis_angle(Vec3::Z, angle).rotate(Vec3::Y);
        let b = w
            .create_body(&BodyDesc {
                position: up * 1.01,
                rotation: Quat::from_axis_angle(Vec3::Z, angle),
                ..Default::default()
            })
            .unwrap();
        let mut d = ColliderDesc::new(cube(0.5));
        d.friction = friction;
        w.add_collider(b, &d).unwrap();
        run(&mut w, 0.5);
        let start = w.body_pose(b).unwrap().position;
        run(&mut w, 1.0);
        let moved = w.body_pose(b).unwrap().position - start;
        if should_slide {
            // a = g (sin θ − μ cos θ); over 1 s from (nearly) the speed reached at 0.5 s.
            let a = 9.81 * (angle.sin() - friction * angle.cos());
            assert!(moved.length() > 0.5 * a, "slides: moved {moved:?}, a = {a}");
        } else {
            assert!(moved.length() < 0.01, "holds: moved {moved:?}");
        }
    }
}

#[test]
fn a_bouncy_ball_bounces_to_restitution_squared_of_its_height() {
    let mut w = PhysicsWorld::default();
    let g = ground(&mut w);
    let _ = g;
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, 5.5, 0.0),
            ..Default::default()
        })
        .unwrap();
    let mut d = ColliderDesc::new(Shape::Sphere { radius: 0.5 });
    d.restitution = 0.8;
    w.add_collider(b, &d).unwrap();
    // Track the peak after the first bounce.
    let mut bounced = false;
    let mut peak: f32 = 0.0;
    for _ in 0..240 {
        let before = w.linear_velocity(b).unwrap().y;
        w.step(DT);
        let after = w.linear_velocity(b).unwrap().y;
        if before < 0.0 && after > 0.0 {
            bounced = true;
        }
        if bounced {
            peak = peak.max(y(&w, b));
        }
    }
    let expect = 0.5 + 5.0 * 0.64;
    assert!(
        bounced && (peak - expect).abs() < 0.35,
        "peak {peak}, expected about {expect}"
    );
}

#[test]
fn identical_runs_give_identical_bits() {
    let build = || {
        let mut w = PhysicsWorld::default();
        ground(&mut w);
        for i in 0..20 {
            let x = (i % 5) as f32 * 0.3 - 0.6;
            let shape = if i % 3 == 0 {
                Shape::Sphere { radius: 0.3 }
            } else if i % 3 == 1 {
                Shape::Capsule {
                    half_height: 0.3,
                    radius: 0.2,
                }
            } else {
                cube(0.3)
            };
            let b = dynamic(
                &mut w,
                shape,
                Vec3::new(x, 1.0 + i as f32 * 0.7, (i % 2) as f32 * 0.2),
            );
            w.set_angular_velocity(b, Vec3::new(0.1 * i as f32, 0.3, -0.2));
        }
        w
    };
    let (mut a, mut b) = (build(), build());
    for _ in 0..300 {
        a.step(DT);
        b.step(DT);
    }
    assert_eq!(a.state_hash(), b.state_hash());
}

#[test]
fn a_fast_bullet_does_not_tunnel_through_a_thin_wall() {
    let mut w = PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let wall = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(10.0, 0.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        wall,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(0.02, 5.0, 5.0),
        }),
    )
    .unwrap();
    let b = w
        .create_body(&BodyDesc {
            linear_velocity: Vec3::new(300.0, 0.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(b, &ColliderDesc::new(Shape::Sphere { radius: 0.05 }))
        .unwrap();
    run(&mut w, 0.5);
    let x = w.body_pose(b).unwrap().position.x;
    assert!(x < 10.0, "the ball stays on its side of the wall: x = {x}");
}

#[test]
fn landing_reports_begin_and_hit_and_lifting_reports_end() {
    let mut w = PhysicsWorld::default();
    ground(&mut w);
    let b = dynamic(&mut w, cube(0.5), Vec3::new(0.0, 3.0, 0.0));
    let (mut began, mut hit) = (false, 0.0f32);
    for _ in 0..120 {
        w.step(DT);
        for e in &w.events().contacts {
            match *e {
                ContactEvent::Begin { .. } => began = true,
                ContactEvent::Hit { speed, .. } => hit = hit.max(speed),
                ContactEvent::End { .. } => {}
            }
        }
    }
    assert!(began, "begin event");
    assert!(hit > 5.0, "hit at about sqrt(2 g h) = 6.9 m/s, got {hit}");
    w.set_body_pose(b, Iso::from_position(Vec3::new(0.0, 10.0, 0.0)));
    w.step(DT);
    assert!(
        w.events()
            .contacts
            .iter()
            .any(|e| matches!(e, ContactEvent::End { .. })),
        "end event"
    );
}

#[test]
fn a_sensor_reports_a_ball_passing_through_it() {
    let mut w = PhysicsWorld::default();
    let gate = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(0.0, 5.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    let mut d = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::new(2.0, 0.5, 2.0),
    });
    d.sensor = true;
    w.add_collider(gate, &d).unwrap();
    dynamic(
        &mut w,
        Shape::Sphere { radius: 0.25 },
        Vec3::new(0.0, 8.0, 0.0),
    );
    let (mut begin, mut end) = (false, false);
    for _ in 0..120 {
        w.step(DT);
        for e in &w.events().sensors {
            match e {
                SensorEvent::Begin { .. } => begin = true,
                SensorEvent::End { .. } => end = begin,
            }
        }
    }
    assert!(begin && end, "begin {begin}, end {end}");
}

#[test]
fn a_kinematic_platform_carries_a_box() {
    let mut w = PhysicsWorld::default();
    let platform = w
        .create_body(&BodyDesc {
            kind: BodyKind::Kinematic,
            linear_velocity: Vec3::new(1.0, 0.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        platform,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(3.0, 0.25, 3.0),
        }),
    )
    .unwrap();
    // The box starts moving with the platform, so it never slips.
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, 0.5, 0.0),
            linear_velocity: Vec3::new(1.0, 0.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(b, &ColliderDesc::new(cube(0.25))).unwrap();
    run(&mut w, 2.0);
    let px = w.body_pose(platform).unwrap().position.x;
    let bx = w.body_pose(b).unwrap().position.x;
    assert!((px - 2.0).abs() < 0.01, "the platform moved 2 m: {px}");
    assert!(
        (bx - px).abs() < 0.05,
        "the box rode along: box {bx}, platform {px}"
    );
    assert!(y(&w, b) > 0.4, "and stayed on top");
}

#[test]
fn rays_casts_and_overlaps_find_the_nearest_things() {
    let mut w = PhysicsWorld::default();
    ground(&mut w);
    let near = dynamic(&mut w, cube(0.5), Vec3::new(0.0, 0.5, 3.0));
    let far = dynamic(&mut w, cube(0.5), Vec3::new(0.0, 0.5, 6.0));
    let all = QueryFilter::default();
    let hit = w
        .raycast(Vec3::new(0.0, 0.5, 0.0), Vec3::Z, 100.0, &all)
        .unwrap();
    assert_eq!(hit.body, near);
    assert!(
        (hit.distance - 2.5).abs() < 1e-4 && (hit.normal - -Vec3::Z).length() < 1e-4,
        "{hit:?}"
    );
    let skip_near = QueryFilter {
        exclude_body: Some(near),
        ..all
    };
    assert_eq!(
        w.raycast(Vec3::new(0.0, 0.5, 0.0), Vec3::Z, 100.0, &skip_near)
            .unwrap()
            .body,
        far
    );
    let down = w
        .raycast(Vec3::new(10.0, 5.0, 10.0), -Vec3::Y, 100.0, &all)
        .unwrap();
    assert!((down.distance - 5.0).abs() < 1e-4);
    let cast = w
        .shape_cast(
            &Shape::Sphere { radius: 0.5 },
            &Iso::from_position(Vec3::new(0.0, 0.6, 0.0)),
            Vec3::new(0.0, 0.0, 10.0),
            &all,
        )
        .unwrap();
    assert_eq!(cast.body, near);
    assert!(
        (cast.fraction - 0.2).abs() < 0.01,
        "touches after 2 m of 10: {}",
        cast.fraction
    );
    let mut seen = Vec::new();
    w.overlap(
        &Shape::Box {
            half_extents: Vec3::new(1.0, 1.0, 1.0),
        },
        &Iso::from_position(Vec3::new(0.0, 0.5, 3.0)),
        &all,
        &mut |c| {
            seen.push(w.collider_body(c).unwrap());
            true
        },
    );
    assert!(seen.contains(&near) && !seen.contains(&far), "{seen:?}");
}

#[test]
fn layers_and_masks_let_bodies_pass_through_each_other() {
    let mut w = PhysicsWorld::default();
    let g = ground(&mut w);
    let _ = g;
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, 2.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    let mut d = ColliderDesc::new(cube(0.5));
    d.layers = 2;
    d.mask = !1; // ignores the ground's layer 1
    w.add_collider(b, &d).unwrap();
    run(&mut w, 1.0);
    assert!(
        y(&w, b) < -1.0,
        "fell through the ground it ignores: {}",
        y(&w, b)
    );
}

#[test]
fn a_body_with_locked_rotation_does_not_tip_over() {
    let mut w = PhysicsWorld::default();
    ground(&mut w);
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, 1.0, 0.0),
            lock_rotation: [true, true, true],
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        b,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(0.2, 1.0, 0.2),
        }),
    )
    .unwrap();
    w.apply_impulse_at(b, Vec3::new(50.0, 0.0, 0.0), Vec3::new(0.0, 1.8, 0.0));
    run(&mut w, 2.0);
    let q = w.body_pose(b).unwrap().rotation;
    assert!(q.dot(Quat::IDENTITY).abs() > 0.9999, "still upright: {q:?}");
}

#[test]
fn bodies_rest_on_triangle_meshes_and_heightfields() {
    let mut w = PhysicsWorld::default();
    // A 10 m floor of 8×8 quads at y = 0.
    let n = 8;
    let mut vertices = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            vertices.push(Vec3::new(
                i as f32 * 10.0 / n as f32 - 5.0,
                0.0,
                j as f32 * 10.0 / n as f32 - 5.0,
            ));
        }
    }
    let mut triangles = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let a = (j * (n + 1) + i) as u32;
            let (b, c, d) = (a + 1, a + (n + 1) as u32, a + (n + 2) as u32);
            triangles.push([a, c, d]);
            triangles.push([a, d, b]);
        }
    }
    let floor = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        floor,
        &ColliderDesc::new(Shape::Mesh(Arc::new(
            TriangleMesh::new(vertices, triangles).unwrap(),
        ))),
    )
    .unwrap();
    // A gently sloped heightfield next to it, 2 m up.
    let heights: Vec<f32> = (0..16 * 16).map(|k| ((k % 16) as f32) * 0.05).collect();
    let terrain = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(20.0, 2.0, 0.0),
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
    let on_mesh = dynamic(&mut w, cube(0.5), Vec3::new(0.3, 1.5, 0.2));
    // A box holds on the 3° slope (a ball would roll off the edge).
    let on_field = dynamic(&mut w, cube(0.5), Vec3::new(27.0, 4.0, 7.0));
    run(&mut w, 3.0);
    assert!(
        (y(&w, on_mesh) - 0.5).abs() < 0.02,
        "box on the mesh at {}",
        y(&w, on_mesh)
    );
    let p = w.body_pose(on_field).unwrap().position;
    assert!(
        p.y > 2.0 && p.y < 4.0 && (p.x - 27.0).abs() < 0.5,
        "box on the terrain at {p:?}"
    );
}

#[test]
fn a_sleeping_stack_wakes_when_hit() {
    let mut w = PhysicsWorld::default();
    ground(&mut w);
    let bottom = dynamic(&mut w, cube(0.5), Vec3::new(0.0, 0.5, 0.0));
    let top = dynamic(&mut w, cube(0.5), Vec3::new(0.0, 1.5, 0.0));
    run(&mut w, 3.0);
    assert!(!w.is_awake(bottom) && !w.is_awake(top), "the stack sleeps");
    let ball = dynamic(
        &mut w,
        Shape::Sphere { radius: 0.25 },
        Vec3::new(-3.0, 1.5, 0.0),
    );
    w.set_linear_velocity(ball, Vec3::new(10.0, 0.0, 0.0));
    run(&mut w, 0.5);
    assert!(w.is_awake(top), "the hit box woke");
    // The 65 kg ball and the 1000 kg box share the ball's momentum (restitution 0), so the box
    // starts at about 0.61 m/s and friction on the box below (0.6 g, carried by both) stops it
    // after about 3.4 cm.
    let moved = w.body_pose(top).unwrap().position.x;
    assert!(moved > 0.02 && moved < 0.08, "and moved: {moved}");
}

#[test]
fn hulls_roll_and_settle() {
    let mut w = PhysicsWorld::default();
    ground(&mut w);
    // A square pyramid built from its faces.
    let v = vec![
        Vec3::new(-0.5, 0.0, -0.5),
        Vec3::new(0.5, 0.0, -0.5),
        Vec3::new(0.5, 0.0, 0.5),
        Vec3::new(-0.5, 0.0, 0.5),
        Vec3::new(0.0, 0.8, 0.0),
    ];
    let loops: [&[u32]; 5] = [
        &[0, 1, 2, 3],
        &[0, 4, 1],
        &[1, 4, 2],
        &[2, 4, 3],
        &[3, 4, 0],
    ];
    let hull = ConvexHull::from_faces(v, &loops).unwrap();
    let b = dynamic(
        &mut w,
        Shape::Hull(Arc::new(hull)),
        Vec3::new(0.0, 2.0, 0.0),
    );
    w.set_angular_velocity(b, Vec3::new(2.0, 0.0, 1.0));
    run(&mut w, 4.0);
    let p = w.body_pose(b).unwrap().position;
    assert!(p.y > -0.01 && p.y < 1.0, "rests on the ground: {p:?}");
    assert!(w.linear_velocity(b).unwrap().length() < 0.05);
}
