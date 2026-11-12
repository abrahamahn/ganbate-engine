//! `ScenePhysics::advance`, the step a host such as the web library's `physicsStep(dt, steps)`
//! runs once a frame: any `steps` must return promptly. Before it, the web library ran
//! `for _ in 0..steps` with whatever count the page passed, so `physicsStep(1/60, 1e7)` (a frame
//! counter or milliseconds passed as steps) simulated ten million steps on the page's only thread
//! before returning.

use std::time::{Duration, Instant};

use ganbate_core::world::{EntityRecord, EntityTable, Transform, World};
use ganbate_physics::Settings;
use ganbate_physics::scene::{self, MAX_STEP, MAX_STEPS_PER_CALL, ScenePhysics};

/// A ground and a few crates that never sleep, so every step costs a full solve.
fn crates() -> ScenePhysics {
    let at = |id: u64, x: f32, y: f32| EntityRecord {
        id,
        name: format!("entity {id}"),
        parent: None,
        transform: Transform::from_trs([x, y, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0; 3]).unwrap(),
        instances: 0..0,
        lights: 0..0,
    };
    let mut entities = vec![at(1, 0.0, -0.5)];
    let mut data = vec![
        r#""0000000000000001":{"components":{"collider":{"half_extents":[20,0.5,20]}}}"#
            .to_string(),
    ];
    for i in 0..6u64 {
        entities.push(at(
            2 + i,
            (i % 3) as f32 * 1.2 - 1.2,
            0.5 + (i / 3) as f32 * 1.1,
        ));
        data.push(format!(
            r#""{:016x}":{{"components":{{"collider":{{}},"rigid_body":{{"never_sleeps":true}}}}}}"#,
            2 + i
        ));
    }
    scene::build(
        &World::default(),
        &EntityTable {
            entities,
            data: format!(r#"{{"entities":{{{}}}}}"#, data.join(",")),
        },
        Settings::default(),
    )
    .unwrap()
}

#[test]
fn one_call_runs_at_most_64_steps_however_many_are_asked_for() {
    assert_eq!(MAX_STEPS_PER_CALL, 64);
    for asked in [65, 1000, 10_000_000, u32::MAX] {
        let mut s = crates();
        let start = Instant::now();
        let ran = s.advance(1.0 / 60.0, asked);
        let took = start.elapsed();
        assert_eq!(ran, 64, "asked for {asked}");
        assert_eq!(s.world.step_count(), 64, "asked for {asked}");
        // 64 steps of seven bodies take milliseconds even unoptimized; ten million took minutes.
        assert!(
            took < Duration::from_secs(20),
            "asked for {asked}: {took:?}"
        );
    }
}

#[test]
fn a_call_computes_what_the_same_steps_one_by_one_compute() {
    let dt = (1.0f64 / 60.0) as f32;
    for steps in [1, 4, 64] {
        let mut at_once = crates();
        assert_eq!(at_once.advance(dt, steps), steps);
        let mut one_by_one = crates();
        for _ in 0..steps {
            one_by_one.world.step(dt);
        }
        assert_eq!(
            at_once.world.state_hash(),
            one_by_one.world.state_hash(),
            "{steps} steps"
        );
    }
    // Asking for more than the limit computes exactly the limit.
    let mut capped = crates();
    capped.advance(dt, 1000);
    let mut limit = crates();
    limit.advance(dt, 64);
    assert_eq!(capped.world.state_hash(), limit.world.state_hash());
}

#[test]
fn long_steps_are_shortened_and_no_steps_are_taken_without_time() {
    // A step longer than 1/15 s takes 1/15 s.
    let mut long = crates();
    assert_eq!(long.advance(1.0, 3), 3);
    let mut longest = crates();
    longest.advance(MAX_STEP, 3);
    assert_eq!(long.world.state_hash(), longest.world.state_hash());
    let mut endless = crates();
    endless.advance(f32::INFINITY, 3);
    assert_eq!(endless.world.state_hash(), longest.world.state_hash());
    // No time, negative time or NaN: nothing runs.
    let still = crates().world.state_hash();
    for dt in [0.0, -1.0 / 60.0, f32::NAN, f32::NEG_INFINITY] {
        let mut s = crates();
        assert_eq!(s.advance(dt, 10), 0, "dt {dt}");
        assert_eq!(s.world.step_count(), 0, "dt {dt}");
        assert_eq!(s.world.state_hash(), still, "dt {dt}");
    }
    // And no steps, no time passes.
    let mut none = crates();
    assert_eq!(none.advance(1.0 / 60.0, 0), 0);
    assert_eq!(none.world.state_hash(), still);
}
