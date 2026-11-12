//! Warm starting with the real narrow phase: in stacks of equal, aligned boxes the incident
//! vertices lie on the reference face's side planes, so their feature ids change from step to
//! step. Points whose id found no match must still take last step's impulse by position, or the
//! stack restarts from nothing at those points every step and settles several times slower
//! (review finding WARM-MATCH regression). Every scene here is synthetic.

use ganbate_physics::{BodyDesc, BodyId, BodyKind, ColliderDesc, PhysicsWorld, Shape, Vec3};

const DT: f32 = 1.0 / 60.0;

fn ground(w: &mut PhysicsWorld) {
    let g = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(0.0, -0.5, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        g,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(50.0, 0.5, 50.0),
        }),
    )
    .unwrap();
}

fn cube(w: &mut PhysicsWorld, position: Vec3, half: f32) -> BodyId {
    let b = w
        .create_body(&BodyDesc {
            position,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(
        b,
        &ColliderDesc::new(Shape::Box {
            half_extents: Vec3::splat(half),
        }),
    )
    .unwrap();
    b
}

/// Steps until every body sleeps; returns the step count (or `limit`).
fn steps_to_sleep(w: &mut PhysicsWorld, limit: usize) -> usize {
    for step in 1..=limit {
        w.step(DT);
        if w.awake_body_count() == 0 {
            return step;
        }
    }
    limit
}

/// A column of `n` boxes (half extent 0.25) standing on the ground.
fn column(n: usize) -> (PhysicsWorld, Vec<BodyId>) {
    let mut w = PhysicsWorld::default();
    ground(&mut w);
    let boxes = (0..n)
        .map(|i| cube(&mut w, Vec3::new(0.0, 0.25 + 0.5 * i as f32, 0.0), 0.25))
        .collect();
    (w, boxes)
}

/// With position matching the columns sleep after 41 and 57 steps and their tops stay within a
/// millimetre of the axis; with ids only they took 98 and 163 steps and drifted 2 and 6.5 cm.
#[test]
fn aligned_columns_settle_as_fast_as_with_position_matching() {
    for (n, most) in [(10usize, 50usize), (20, 70)] {
        let (mut w, boxes) = column(n);
        let steps = steps_to_sleep(&mut w, 400);
        let top = w.body_pose(*boxes.last().unwrap()).unwrap().position;
        assert!(
            steps <= most,
            "{n} boxes: asleep after {steps} steps (at most {most}); top at {top:?}"
        );
        let drift = (top - Vec3::new(0.0, top.y, 0.0)).length();
        assert!(drift < 0.01, "{n} boxes: the top drifted {drift} m");
    }
}

/// Half a second in, the top of a 10-box column is nearly still (0.011 m/s with position
/// matching, 0.043 m/s with ids only).
#[test]
fn an_aligned_column_is_nearly_still_after_half_a_second() {
    let (mut w, boxes) = column(10);
    for _ in 0..30 {
        w.step(DT);
    }
    let top = *boxes.last().unwrap();
    let v = w.linear_velocity(top).unwrap().length();
    assert!(v < 0.03, "the top still moves at {v} m/s");
}

/// A 3 × 3 × 4 block of aligned boxes sleeps after 30 steps (a guard: ids alone did as well
/// here, the columns above are what they slowed down).
#[test]
fn an_aligned_block_settles_quickly() {
    let mut w = PhysicsWorld::default();
    ground(&mut w);
    for layer in 0..4 {
        for i in 0..3 {
            for k in 0..3 {
                let p = Vec3::new(
                    0.5 * i as f32 - 0.5,
                    0.25 + 0.5 * layer as f32,
                    0.5 * k as f32 - 0.5,
                );
                cube(&mut w, p, 0.25);
            }
        }
    }
    let steps = steps_to_sleep(&mut w, 400);
    assert!(steps <= 33, "asleep after {steps} steps");
}
