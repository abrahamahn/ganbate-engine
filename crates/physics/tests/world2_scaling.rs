//! Calls that concern one body or one joint must cost what that body or joint has, not what the
//! world has: teleporting a body, waking one, destroying one, destroying a joint, and the
//! joint filter of every pair. Each test times the same work in a small world and in one with
//! 16 times as many contacts (or colliders, or joints) and asserts the cost did not grow with
//! it; at 7842e05 each of them scanned every contact, pair or level collider, so the larger
//! world cost about 16 times as much (review findings: set_body_pose scanning every contact,
//! J5, J6, J7, F3). Every scene here is synthetic.
//!
//! Timings are the fastest of several runs and the tests run one at a time, so the ratios are
//! steady; the bound (4x) is far from both the fixed cost (about 1x) and the old one (16x).

use std::sync::Mutex;
use std::time::{Duration, Instant};

use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, DistanceOptions, Iso, JointDesc, JointKind,
    PhysicsWorld, Settings, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;
const SMALL: usize = 250;
const LARGE: usize = 16 * SMALL;
/// How much more the large world may cost.
const MOST: f64 = 4.0;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

/// The fastest of `runs` runs of `f` on fresh copies of `w` (copying is not timed).
fn fastest(w: &PhysicsWorld, runs: usize, f: &mut dyn FnMut(&mut PhysicsWorld)) -> Duration {
    (0..runs)
        .map(|_| {
            let mut w = w.clone();
            let t = Instant::now();
            f(&mut w);
            t.elapsed()
        })
        .min()
        .unwrap()
}

fn assert_flat(what: &str, small: Duration, large: Duration) {
    let ratio = large.as_secs_f64() / small.as_secs_f64().max(1e-9);
    assert!(
        ratio < MOST,
        "{what}: {small:?} with {SMALL}, {large:?} with {LARGE} (x{ratio:.1}): the cost grows with the world"
    );
}

fn no_gravity() -> PhysicsWorld {
    PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    })
}

fn body(w: &mut PhysicsWorld, desc: BodyDesc, shape: Option<Shape>) -> BodyId {
    let b = w.create_body(&desc).unwrap();
    if let Some(shape) = shape {
        w.add_collider(b, &ColliderDesc::new(shape)).unwrap();
    }
    b
}

fn ground(w: &mut PhysicsWorld) -> BodyId {
    body(
        w,
        BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::new(0.0, -0.5, 0.0),
            ..Default::default()
        },
        Some(Shape::Box {
            half_extents: Vec3::new(200.0, 0.5, 200.0),
        }),
    )
}

/// `n` balls resting on the ground, a metre apart: one contact each.
fn props(w: &mut PhysicsWorld, n: usize, awake: bool) {
    for i in 0..n {
        let p = Vec3::new((i % 64) as f32 - 32.0, 0.25, (i / 64) as f32 - 32.0);
        body(
            w,
            BodyDesc {
                position: p,
                awake,
                can_sleep: false,
                ..Default::default()
            },
            Some(Shape::Sphere { radius: 0.25 }),
        );
    }
}

/// `n` balls far from everything, with no contacts.
fn loners(w: &mut PhysicsWorld, n: usize, awake: bool) -> Vec<BodyId> {
    (0..n)
        .map(|i| {
            body(
                w,
                BodyDesc {
                    position: Vec3::new(1000.0 + 3.0 * i as f32, 50.0, 1000.0),
                    awake,
                    gravity_scale: 0.0,
                    ..Default::default()
                },
                Some(Shape::Sphere { radius: 0.5 }),
            )
        })
        .collect()
}

/// Teleporting a body walked every contact in the world, also for an awake body with none.
#[test]
fn teleporting_a_body_costs_the_same_in_a_larger_world() {
    let _one = serial();
    let world = |n: usize| {
        let mut w = no_gravity();
        ground(&mut w);
        props(&mut w, n, true);
        let k = body(
            &mut w,
            BodyDesc {
                kind: BodyKind::Kinematic,
                position: Vec3::new(-1000.0, 5.0, 0.0),
                ..Default::default()
            },
            Some(Shape::Box {
                half_extents: Vec3::splat(0.5),
            }),
        );
        w.step(DT);
        assert!(w.contact_count() >= n);
        (w, k)
    };
    let time = |(w, k): &(PhysicsWorld, BodyId)| {
        let k = *k;
        fastest(w, 5, &mut |w| {
            for i in 0..200 {
                let p = Vec3::new(-1000.0 + 0.01 * i as f32, 5.0, 0.0);
                w.set_body_pose(k, Iso::from_position(p));
            }
        })
    };
    let (small, large) = (world(SMALL), world(LARGE));
    assert_flat("200 teleports", time(&small), time(&large));
}

/// Waking a sleeping body walked every contact in the world (J6): an explosion waking many
/// separate sleepers one by one cost their count times the world's contacts.
#[test]
fn waking_a_body_costs_the_same_in_a_larger_world() {
    let _one = serial();
    let world = |n: usize| {
        let mut w = no_gravity();
        ground(&mut w);
        props(&mut w, n, false);
        let sleepers = loners(&mut w, 100, false);
        w.step(DT);
        assert_eq!(w.awake_body_count(), 0);
        (w, sleepers)
    };
    let time = |(w, sleepers): &(PhysicsWorld, Vec<BodyId>)| {
        fastest(w, 5, &mut |w| {
            for &b in sleepers {
                w.wake(b);
            }
        })
    };
    let (small, large) = (world(SMALL), world(LARGE));
    assert_flat("100 wakes", time(&small), time(&large));
    let (mut w, sleepers) = large;
    for &b in &sleepers {
        w.wake(b);
    }
    assert_eq!(w.awake_body_count(), sleepers.len(), "only the 100 woke");
}

/// Destroying a body scanned every pair to find its own, and every contact to find what
/// touched it (F3), even for a body touching nothing.
#[test]
fn destroying_a_body_costs_the_same_in_a_larger_world() {
    let _one = serial();
    let world = |n: usize| {
        let mut w = no_gravity();
        ground(&mut w);
        props(&mut w, n, true);
        let doomed = loners(&mut w, 100, true);
        w.step(DT);
        (w, doomed)
    };
    let time = |(w, doomed): &(PhysicsWorld, Vec<BodyId>)| {
        fastest(w, 5, &mut |w| {
            for &b in doomed {
                w.destroy_body(b);
            }
        })
    };
    let (small, large) = (world(SMALL), world(LARGE));
    assert_flat("100 destroys", time(&small), time(&large));
}

/// Bodies created and destroyed before any step: each destroy also searched the whole move
/// buffer (F3), so clearing n new bodies cost n².
#[test]
fn destroying_new_bodies_before_a_step_costs_the_same_per_body() {
    let _one = serial();
    let time = |n: usize| {
        let mut w = no_gravity();
        let bodies = loners(&mut w, n, true);
        fastest(&w, 3, &mut |w| {
            for &b in &bodies {
                w.destroy_body(b);
            }
        })
        .div_f64(n as f64)
    };
    assert_flat("a destroy before the first step", time(SMALL), time(LARGE));
}

/// A rope from a lamp to a level with `tiles` colliders, for every lamp.
fn roped_level(tiles: usize, lamps: usize) -> (PhysicsWorld, Vec<ganbate_physics::JointId>) {
    let mut w = no_gravity();
    let level = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            ..Default::default()
        })
        .unwrap();
    for i in 0..tiles {
        let mut d = ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(0.5, 0.1, 0.5),
        });
        d.local = Iso::from_position(Vec3::new((i % 64) as f32, 0.0, (i / 64) as f32));
        w.add_collider(level, &d).unwrap();
    }
    let ropes = (0..lamps)
        .map(|i| {
            let lamp = body(
                &mut w,
                BodyDesc {
                    position: Vec3::new(i as f32, 20.0, -10.0),
                    ..Default::default()
                },
                None,
            );
            w.create_joint(&JointDesc {
                body_a: level,
                body_b: lamp,
                frame_a: Iso::from_position(Vec3::new(i as f32, 21.0, -10.0)),
                frame_b: Iso::IDENTITY,
                kind: JointKind::Distance(DistanceOptions {
                    length: 1.0,
                    min: 0.0,
                    max: 1.0,
                    spring: None,
                }),
                collide_connected: false,
                user_data: 0,
            })
            .unwrap()
        })
        .collect();
    w.step(DT);
    (w, ropes)
}

/// Destroying a joint queued every collider of both bodies for pair finding, a static level's
/// thousands included, and the next step queried the moving tree once for each (J7).
#[test]
fn destroying_a_joint_to_a_large_level_costs_the_same_as_to_a_small_one() {
    let _one = serial();
    let time = |(w, ropes): &(PhysicsWorld, Vec<ganbate_physics::JointId>)| {
        fastest(w, 3, &mut |w| {
            for &j in ropes {
                w.destroy_joint(j);
            }
            w.step(DT);
        })
    };
    let (small, large) = (roped_level(SMALL, 100), roped_level(LARGE, 100));
    assert_flat(
        "destroying 100 ropes and stepping",
        time(&small),
        time(&large),
    );
}

/// Every pair's joint filter searched the joint list of the body owning the lower collider
/// index: with lamps roped to the level, every ball resting on the level paid for every rope
/// at every step (J5). The same scene with the ropes on a separate anchor is the reference.
#[test]
fn joints_on_a_level_do_not_slow_the_pairs_touching_it() {
    let _one = serial();
    let world = |on_level: bool| {
        let mut w = no_gravity();
        let level = ground(&mut w);
        props(&mut w, 400, true);
        let anchor = if on_level {
            level
        } else {
            body(
                &mut w,
                BodyDesc {
                    kind: BodyKind::Static,
                    ..Default::default()
                },
                None,
            )
        };
        for i in 0..LARGE {
            let lamp = body(
                &mut w,
                BodyDesc {
                    position: Vec3::new(i as f32, 30.0, 0.0),
                    ..Default::default()
                },
                None,
            );
            w.create_joint(&JointDesc {
                body_a: anchor,
                body_b: lamp,
                frame_a: Iso::from_position(Vec3::new(i as f32, 31.0, 0.0)),
                frame_b: Iso::IDENTITY,
                kind: JointKind::Ball,
                collide_connected: false,
                user_data: 0,
            })
            .unwrap();
        }
        // The lamps hang still and fall asleep; the balls stay awake on the level.
        for _ in 0..40 {
            w.step(DT);
        }
        assert_eq!(w.awake_body_count(), 400);
        w
    };
    // Pair the measurements while the same other work is running on the machine. Alternate
    // the order so the second world does not always inherit a warmer CPU or a busier one.
    let (mut apart, mut on_level) = (world(false), world(true));
    let time = |w: &mut PhysicsWorld| {
        let start = Instant::now();
        for _ in 0..3 {
            w.step(DT);
        }
        start.elapsed()
    };
    let mut ratios = [0.0; 7];
    for (i, ratio) in ratios.iter_mut().enumerate() {
        let (reference, with_ropes) = if i % 2 == 0 {
            (time(&mut apart), time(&mut on_level))
        } else {
            let with_ropes = time(&mut on_level);
            (time(&mut apart), with_ropes)
        };
        *ratio = with_ropes.as_secs_f64() / reference.as_secs_f64();
    }
    ratios.sort_unstable_by(f64::total_cmp);
    let median = ratios[ratios.len() / 2];
    assert!(
        median < 1.5,
        "steps with {LARGE} ropes on the level / on their own anchor: {ratios:?} (median x{median:.2})"
    );
}
