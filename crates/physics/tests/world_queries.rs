//! Queries between steps: they must see every collider where it is now, honour the generation of
//! an excluded body's handle and refuse directions that have none. Every scene here is synthetic.

use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, Iso, PhysicsWorld, QueryFilter, Settings, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;

fn ball(w: &mut PhysicsWorld, position: Vec3, velocity: Vec3, radius: f32) -> BodyId {
    let b = w
        .create_body(&BodyDesc {
            position,
            linear_velocity: velocity,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(b, &ColliderDesc::new(Shape::Sphere { radius }))
        .unwrap();
    b
}

/// A falling ball leaves the broad-phase box it had at the start of the step; a ray through it
/// between steps must still find it (review finding STALE-BOXES).
#[test]
fn a_ray_between_steps_finds_a_falling_ball_where_it_is_now() {
    let mut w = PhysicsWorld::default();
    let b = ball(&mut w, Vec3::new(0.0, 50.0, 0.0), Vec3::ZERO, 0.25);
    // About 10 m/s after a second of free fall: 16 cm per step, more than the boxes' slack.
    for step in 0..60 {
        w.step(DT);
        let p = w.body_pose(b).unwrap().position;
        // Horizontal, straight through the ball's axis, 3 cm above its bottom.
        let hit = w.raycast(
            Vec3::new(-5.0, p.y - 0.22, 0.0),
            Vec3::X,
            10.0,
            &QueryFilter::default(),
        );
        assert!(
            hit.is_some_and(|h| h.body == b),
            "step {step}: the ray misses the ball at y = {}",
            p.y
        );
    }
}

/// The same for a fast ball and every query that walks the trees (review finding
/// query-stale-proxies): a ray and a sphere cast through its centre after each step.
#[test]
fn rays_and_casts_between_steps_find_a_fast_ball() {
    let mut w = PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let b = ball(&mut w, Vec3::ZERO, Vec3::new(20.0, 0.0, 0.0), 0.1);
    let all = QueryFilter::default();
    for step in 0..3 {
        w.step(DT);
        let x = w.body_pose(b).unwrap().position.x;
        let ray = w.raycast(Vec3::new(x, 5.0, 0.0), -Vec3::Y, 10.0, &all);
        assert!(
            ray.is_some_and(|h| h.body == b && (h.distance - 4.9).abs() < 1e-3),
            "step {step}: ray at x = {x}: {ray:?}"
        );
        let cast = w.shape_cast(
            &Shape::Sphere { radius: 0.05 },
            &Iso::from_position(Vec3::new(x, 5.0, 0.0)),
            Vec3::new(0.0, -10.0, 0.0),
            &all,
        );
        assert!(
            cast.is_some_and(|h| h.body == b),
            "step {step}: cast at x = {x}: {cast:?}"
        );
    }
}

/// Overlap tests run the narrow phase, so this one needs it.
#[test]
fn overlaps_between_steps_find_a_fast_ball() {
    let mut w = PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let b = ball(&mut w, Vec3::ZERO, Vec3::new(20.0, 0.0, 0.0), 0.1);
    for step in 0..3 {
        w.step(DT);
        let x = w.body_pose(b).unwrap().position.x;
        let mut seen = Vec::new();
        w.overlap(
            &Shape::Sphere { radius: 0.05 },
            &Iso::from_position(Vec3::new(x, 0.0, 0.0)),
            &QueryFilter::default(),
            &mut |c| {
                seen.push(w.collider_body(c).unwrap());
                true
            },
        );
        assert_eq!(seen, vec![b], "step {step}: overlap at x = {x}");
    }
}

/// Excluding a destroyed body must not hide whatever body reuses its slot (review finding
/// query-exclude-generation).
#[test]
fn excluding_a_destroyed_body_does_not_hide_the_body_that_reuses_its_slot() {
    let mut w = PhysicsWorld::default();
    let old = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            ..Default::default()
        })
        .unwrap();
    w.destroy_body(old);
    let new = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(new, &ColliderDesc::new(Shape::Sphere { radius: 1.0 }))
        .unwrap();
    assert_eq!(new.index, old.index, "the slot is reused");
    assert!(!w.contains_body(old));
    let filter = QueryFilter {
        exclude_body: Some(old),
        ..Default::default()
    };
    let hit = w.raycast(Vec3::new(0.0, 5.0, 0.0), -Vec3::Y, 10.0, &filter);
    assert!(
        hit.is_some_and(|h| h.body == new),
        "the new body is hidden by a stale handle: {hit:?}"
    );
    // Excluding the live handle still works.
    let filter = QueryFilter {
        exclude_body: Some(new),
        ..Default::default()
    };
    assert!(
        w.raycast(Vec3::new(0.0, 5.0, 0.0), -Vec3::Y, 10.0, &filter)
            .is_none()
    );
}

/// A direction that is infinite or not a number has none; a finite one whose squared length
/// overflows still has one (review finding query-ray-direction).
#[test]
fn rays_refuse_non_finite_directions_and_keep_huge_ones() {
    let mut w = PhysicsWorld::default();
    let s = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(s, &ColliderDesc::new(Shape::Sphere { radius: 1.0 }))
        .unwrap();
    let all = QueryFilter::default();
    for bad in [
        Vec3::new(f32::INFINITY, 0.0, 0.0),
        Vec3::new(f32::NEG_INFINITY, 1.0, 0.0),
        Vec3::new(f32::NAN, 0.0, 0.0),
        Vec3::ZERO,
    ] {
        let hit = w.raycast(Vec3::ZERO, bad, 10.0, &all);
        assert!(hit.is_none(), "direction {bad:?} gave {hit:?}");
    }
    // From inside the ball and from outside, along a diagonal given with huge components.
    let huge = Vec3::new(1e20, 0.0, 1e20);
    let unit = Vec3::new(1.0, 0.0, 1.0) * (1.0 / 2.0f32.sqrt());
    for origin in [Vec3::ZERO, -unit * 5.0] {
        let hit = w.raycast(origin, huge, 10.0, &all).expect("a hit");
        assert!(
            (hit.normal.length() - 1.0).abs() < 1e-5,
            "from {origin:?}: the normal is not a unit vector: {hit:?}"
        );
        let expected = (hit.point - origin).length();
        assert!(
            (hit.distance - expected).abs() < 1e-4,
            "from {origin:?}: {hit:?}"
        );
    }
}
