//! Settings that would freeze the world take their defaults: a top speed of zero held every body
//! where it was, and an infinite sleep threshold counted every body as still, so a falling ball
//! fell asleep in mid-air (review finding: Settings::sanitized kept both). Every scene here is
//! synthetic.

use ganbate_physics::{BodyDesc, BodyId, ColliderDesc, PhysicsWorld, Settings, Shape, Vec3};

const DT: f32 = 1.0 / 60.0;

fn dropped_ball(settings: Settings) -> (PhysicsWorld, BodyId) {
    let mut w = PhysicsWorld::new(settings);
    let b = w
        .create_body(&BodyDesc {
            position: Vec3::new(0.0, 10.0, 0.0),
            ..Default::default()
        })
        .unwrap();
    w.add_collider(b, &ColliderDesc::new(Shape::Sphere { radius: 0.5 }))
        .unwrap();
    (w, b)
}

#[test]
fn a_zero_top_speed_or_an_infinite_sleep_threshold_takes_its_default() {
    let defaults = Settings::default();
    for (name, settings) in [
        (
            "max_linear_speed = 0",
            Settings {
                max_linear_speed: 0.0,
                ..Default::default()
            },
        ),
        (
            "max_linear_speed = -0",
            Settings {
                max_linear_speed: -0.0,
                ..Default::default()
            },
        ),
        (
            "sleep_threshold = inf",
            Settings {
                sleep_threshold: f32::INFINITY,
                ..Default::default()
            },
        ),
    ] {
        let (mut w, b) = dropped_ball(settings);
        for _ in 0..120 {
            w.step(DT);
        }
        let y = w.body_pose(b).unwrap().position.y;
        // Two seconds of free fall from 10 m: about 19.6 m down.
        assert!(y < -5.0, "{name}: the ball is at y = {y}");
        assert!(w.is_awake(b), "{name}: the falling ball fell asleep");
        assert_eq!(w.settings.max_linear_speed, defaults.max_linear_speed);
        assert_eq!(w.settings.sleep_threshold, defaults.sleep_threshold);
    }
}

/// The values that remain allowed keep their meaning: an infinite top speed, and a sleep
/// threshold of zero (nothing ever counts as still, so nothing sleeps).
#[test]
fn an_infinite_top_speed_and_a_zero_sleep_threshold_are_kept() {
    let settings = Settings {
        max_linear_speed: f32::INFINITY,
        sleep_threshold: 0.0,
        gravity: Vec3::ZERO,
        ..Default::default()
    };
    let (mut w, b) = dropped_ball(settings.clone());
    for _ in 0..120 {
        w.step(DT);
    }
    assert_eq!(w.settings, settings);
    assert!(w.is_awake(b), "a still ball slept with a zero threshold");
}

/// Squaring a very large speed limit and velocity overflows both, even though each is finite.
#[test]
fn a_large_finite_speed_limit_clamps_only_faster_bodies() {
    let limit = 1e20;
    for (speed, expected) in [(2e20, limit), (5e19, 5e19), (3.0, 3.0)] {
        let (mut w, b) = dropped_ball(Settings {
            gravity: Vec3::ZERO,
            max_linear_speed: limit,
            ..Default::default()
        });
        let direction = Vec3::new(0.6, 0.8, 0.0);
        w.set_linear_velocity(b, direction * speed);
        w.step(DT);
        let scaled = w.linear_velocity(b).unwrap() * (1.0 / expected);
        assert!((scaled - direction).length() < 1e-5, "{speed}: {scaled:?}");
    }
}
