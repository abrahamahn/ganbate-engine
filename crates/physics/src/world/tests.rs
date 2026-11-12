//! White-box tests of the world's bookkeeping that the narrow phase would otherwise have to
//! set up: contacts are marked touching by hand, as a settled scene's would be.

use super::*;

const DT: f32 = 1.0 / 60.0;

fn cuboid(w: &mut PhysicsWorld, kind: BodyKind, position: Vec3, half: Vec3) -> BodyId {
    let b = w
        .create_body(&BodyDesc {
            kind,
            position,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(b, &ColliderDesc::new(Shape::Box { half_extents: half }))
        .unwrap();
    b
}

/// Steps once so the broad phase pairs every collider, then marks touching exactly the
/// contacts between the given bodies and puts every body to sleep: a settled scene.
fn settle(w: &mut PhysicsWorld, touching: &[(BodyId, BodyId)]) {
    w.step(DT);
    let bodies: Vec<(u32, u32)> = w
        .contacts
        .iter()
        .map(|c| {
            (
                w.colliders[c.a as usize].body,
                w.colliders[c.b as usize].body,
            )
        })
        .collect();
    for (c, (p, q)) in w.contacts.iter_mut().zip(bodies) {
        c.touching = touching
            .iter()
            .any(|(x, y)| (x.index, y.index) == (p, q) || (y.index, x.index) == (p, q));
    }
    for b in &mut w.bodies {
        b.awake = false;
        b.sleep_time = 0.0;
        b.linear_velocity = Vec3::ZERO;
        b.angular_velocity = Vec3::ZERO;
    }
    w.awake_words.fill(0);
}

/// End events of contacts removed between steps (here: a visitor destroyed inside a sensor)
/// reach the next step's events, which a game reads after each step; the last step's events
/// stay as they were until then (review finding EVENTS-LOST).
#[test]
fn end_events_from_calls_between_steps_reach_the_next_steps_events() {
    let mut w = PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let zone = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            ..Default::default()
        })
        .unwrap();
    let mut d = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::splat(2.0),
    });
    d.sensor = true;
    let sensor = w.add_collider(zone, &d).unwrap();
    // A box resting on the ground (top at y = -0.5), inside the zone.
    let ground = cuboid(
        &mut w,
        BodyKind::Static,
        Vec3::new(0.0, -1.5, 0.0),
        Vec3::splat(1.0),
    );
    let ball = cuboid(&mut w, BodyKind::Dynamic, Vec3::ZERO, Vec3::splat(0.5));
    settle(&mut w, &[(ground, ball)]);
    assert_eq!(
        w.contacts.len(),
        2,
        "the ground and the zone pair with the box"
    );
    // The visitor inside the zone (sensor contacts are marked as the narrow phase would).
    let (z, g) = (zone.index, ground.index);
    for c in &mut w.contacts {
        let bodies = (
            w.colliders[c.a as usize].body,
            w.colliders[c.b as usize].body,
        );
        if bodies.0 == z || bodies.1 == z {
            c.touching = true;
        }
        assert!(bodies.0 == g || bodies.1 == g || c.touching);
    }
    let visitor = w.bodies[ball.index as usize].colliders[0];
    let visitor = w.collider_id(visitor);
    let ground_collider = w.collider_id(w.bodies[g as usize].colliders[0]);
    w.step(DT);
    assert!(w.events().sensors.is_empty() && w.events().contacts.is_empty());
    w.destroy_body(ball);
    assert!(w.events().sensors.is_empty() && w.events().contacts.is_empty());
    // A step that is not one (no time passes) changes nothing.
    w.step(0.0);
    w.step(f32::NAN);
    w.step(DT);
    assert_eq!(
        w.events().sensors,
        vec![SensorEvent::End { sensor, visitor }]
    );
    assert_eq!(
        w.events().contacts,
        vec![ContactEvent::End {
            a: ground_collider,
            b: visitor
        }]
    );
    w.step(DT);
    assert!(w.events().sensors.is_empty() && w.events().contacts.is_empty());
}

/// A body's inverse inertia, turned with it within a step: the body's own world inverse
/// inertia at the turned orientation; locked rows and columns stay exact zeros; a ball's or a
/// cube's is the same up to a rounding; and a body that may turn about one axis only keeps
/// the one it has.
#[test]
fn inverse_inertia_turns_with_the_body_and_keeps_locked_rows_zero() {
    let mut w = PhysicsWorld::default();
    let dq = Quat::from_axis_angle(Vec3::new(0.3, -0.8, 0.5).normalize_or_zero(), 0.9);
    let start = Quat::from_axis_angle(Vec3::new(1.0, 0.2, -0.4).normalize_or_zero(), 0.6);
    let close = |a: &Mat3, b: &Mat3, tol: f32| {
        (0..3).all(|i| (a.cols[i] - b.cols[i]).length() <= tol * b.cols[i].length().max(1.0))
    };
    for lock in [
        [false; 3],
        [false, true, false],
        [true, false, false],
        [false, false, true],
        [true, false, true],
        [true; 3],
    ] {
        let b = w
            .create_body(&BodyDesc {
                rotation: start,
                lock_rotation: lock,
                ..Default::default()
            })
            .unwrap();
        w.add_collider(
            b,
            &ColliderDesc::new(Shape::Box {
                half_extents: Vec3::new(1.0, 0.05, 0.3),
            }),
        )
        .unwrap();
        // A body free about one axis only can turn about that axis only.
        let free = mask(lock);
        let dq = if free.x + free.y + free.z == 1.0 {
            Quat::from_axis_angle(free, 0.9)
        } else {
            dq
        };
        let body = w.bodies[b.index as usize].clone();
        let turned = Body {
            rotation: dq * start,
            ..body.clone()
        };
        let expected = turned.world_inv_inertia();
        let got = body.turning().after(dq).unwrap_or(body.world_inv_inertia());
        assert!(
            close(&got, &expected, 1e-4),
            "{lock:?}: {got:?} vs {expected:?}"
        );
        for k in 0..3 {
            if lock[k] {
                assert!(
                    got.cols[k] == Vec3::ZERO && got.row(k) == Vec3::ZERO,
                    "{lock:?}: {got:?}"
                );
            }
        }
    }
    // A ball and a cube: turned, the same to a rounding.
    for shape in [
        Shape::Sphere { radius: 0.4 },
        Shape::Box {
            half_extents: Vec3::splat(0.3),
        },
    ] {
        let b = w
            .create_body(&BodyDesc {
                rotation: start,
                ..Default::default()
            })
            .unwrap();
        w.add_collider(b, &ColliderDesc::new(shape)).unwrap();
        let body = &w.bodies[b.index as usize];
        let own = body.world_inv_inertia();
        let turned = body.turning().after(dq).unwrap();
        assert!(close(&turned, &own, 1e-6), "{turned:?} vs {own:?}");
    }
}

/// A body's reach (for the sleep test) is measured from its centre of mass, over its solid
/// colliders, to the farthest point of each shape; its smallest size (for continuous
/// collision) is that of its thinnest solid collider (review finding MAX-EXTENT-ORIGIN).
#[test]
fn extents_are_measured_from_the_centre_of_mass_over_solid_colliders() {
    let mut w = PhysicsWorld::default();
    let b = w.create_body(&BodyDesc::default()).unwrap();
    let mut ball = ColliderDesc::new(Shape::Sphere { radius: 0.5 });
    ball.local = Iso::from_position(Vec3::new(10.0, 0.0, 0.0));
    w.add_collider(b, &ball).unwrap();
    let mut sensor = ColliderDesc::new(Shape::Box {
        half_extents: Vec3::splat(0.01),
    });
    sensor.local = Iso::from_position(Vec3::new(-30.0, 0.0, 0.0));
    sensor.sensor = true;
    w.add_collider(b, &sensor).unwrap();
    let body = &w.bodies[b.index as usize];
    assert_eq!(body.local_center, Vec3::new(10.0, 0.0, 0.0));
    assert!((body.max_extent - 0.5).abs() < 1e-5, "{}", body.max_extent);
    assert!((body.min_extent - 0.5).abs() < 1e-5, "{}", body.min_extent);
    // Two boxes either side of the centre: the farthest corner, and the thinner box.
    let b = w.create_body(&BodyDesc::default()).unwrap();
    for (x, half) in [
        (-1.0f32, Vec3::new(0.5, 0.1, 0.2)),
        (1.0, Vec3::new(0.5, 0.3, 0.2)),
    ] {
        let mut d = ColliderDesc::new(Shape::Box { half_extents: half });
        d.local = Iso::from_position(Vec3::new(x, 0.0, 0.0));
        w.add_collider(b, &d).unwrap();
    }
    let body = &w.bodies[b.index as usize];
    // The heavier box pulls the centre to x = 0.5, so the lighter box's far corner is the
    // farthest point.
    let c = body.local_center;
    assert!((c - Vec3::new(0.5, 0.0, 0.0)).length() < 1e-5, "{c:?}");
    let far = (Vec3::new(-1.5, 0.1, 0.2) - c).length();
    assert!(
        (body.max_extent - far).abs() < 1e-5,
        "{} vs {far}",
        body.max_extent
    );
    assert!((body.min_extent - 0.1).abs() < 1e-6, "{}", body.min_extent);
}

/// Carried-over impulses are per substep: a step after the substep count changed starts
/// from them rescaled to the new count (part of review finding SETTINGS-UNVALIDATED).
#[test]
fn changing_the_substep_count_rescales_the_carried_impulses() {
    let mut w = PhysicsWorld::new(Settings {
        enable_sleep: false,
        ..Default::default()
    });
    let hook = cuboid(
        &mut w,
        BodyKind::Static,
        Vec3::new(0.0, 5.0, 0.0),
        Vec3::splat(0.1),
    );
    let lamp = cuboid(
        &mut w,
        BodyKind::Dynamic,
        Vec3::new(0.0, 4.0, 0.0),
        Vec3::splat(0.1),
    );
    w.create_joint(&JointDesc {
        body_a: hook,
        body_b: lamp,
        frame_a: Iso::from_position(Vec3::new(0.0, -1.0, 0.0)),
        frame_b: Iso::IDENTITY,
        kind: joint::JointKind::Ball,
        collide_connected: false,
        user_data: 0,
    })
    .unwrap();
    for _ in 0..60 {
        w.step(DT);
    }
    assert_eq!(w.warm_substeps, 4);
    // A contact's carried impulses, as the narrow phase would have left them.
    let mut cm = ContactManifold::default();
    cm.manifold.count = 1;
    cm.cache[0].normal_impulse = 2.0;
    cm.cache[0].friction_impulse = Vec3::new(0.5, 0.0, -1.0);
    w.contacts.push(Contact {
        a: 0,
        b: 1,
        manifolds: vec![cm],
        touching: false,
        sensor: false,
        slots: [0, 0],
    });
    let held = w.joints[0].impulses.linear;
    let weight = 1000.0 * 0.008 * 9.81 / 60.0 / 4.0;
    assert!((held.y.abs() - weight).abs() < 0.05 * weight, "{held:?}");
    w.settings.substeps = 2;
    w.rescale_warm_starts(2);
    assert_eq!(w.joints[0].impulses.linear, held * 2.0);
    let p = w.contacts.last().unwrap().manifolds[0].cache[0];
    assert_eq!(p.normal_impulse, 4.0);
    assert_eq!(p.friction_impulse, Vec3::new(1.0, 0.0, -2.0));
    w.contacts.pop();
    // The step itself does it: back to four substeps, half the impulse again.
    w.settings.substeps = 4;
    w.step(DT);
    assert_eq!(w.warm_substeps, 4);
    let after = w.joints[0].impulses.linear;
    assert!(
        (after - held).length() < 0.05 * held.length(),
        "{after:?} vs {held:?}"
    );
}

/// A manifold with the given normal and points (position, feature id), and the impulses
/// last step left on them.
fn manifold(normal: Vec3, points: &[(Vec3, u32, f32)]) -> ContactManifold {
    let mut cm = ContactManifold::default();
    cm.manifold.normal = normal;
    for (i, &(p, id, impulse)) in points.iter().enumerate() {
        cm.manifold.points[i] = collide::ManifoldPoint {
            point: p,
            separation: 0.0,
            id,
        };
        cm.cache[i] = CachedPoint {
            id,
            local_a: p,
            normal_impulse: impulse,
            friction_impulse: Vec3::new(impulse, 0.0, 0.0),
        };
    }
    cm.manifold.count = points.len() as u8;
    cm
}

/// The impulses each new point starts with.
fn carried(old: &[ContactManifold], fresh: &[ContactManifold]) -> Vec<Vec<f32>> {
    let fresh: Vec<Manifold> = fresh.iter().map(|m| m.manifold.clone()).collect();
    let mut out = Vec::new();
    carry_impulses_into(old, &fresh, &Iso::IDENTITY, 0.045, &mut out);
    out.iter()
        .map(|m| {
            m.cache[..usize::from(m.manifold.count)]
                .iter()
                .map(|c| {
                    assert_eq!(c.friction_impulse.x, c.normal_impulse);
                    c.normal_impulse
                })
                .collect()
        })
        .collect()
}

/// Warm starting hands each old point's impulse to at most one new point: a box held by one
/// point that settles onto four 2 cm apart started with four times the impulse that held it
/// (review finding WARM-MATCH).
#[test]
fn carried_impulses_go_to_one_new_point_each() {
    let p = |x: f32, z: f32| Vec3::new(x, 0.0, z);
    let one = [manifold(Vec3::Y, &[(p(0.0, 0.0), 0, 5.0)])];
    let four = [manifold(
        Vec3::Y,
        &[
            (p(-0.01, -0.01), 0, 0.0),
            (p(0.01, -0.01), 0, 0.0),
            (p(0.01, 0.011), 0, 0.0),
            (p(-0.01, 0.01), 0, 0.0),
        ],
    )];
    let got = carried(&one, &four);
    assert_eq!(got.concat().iter().sum::<f32>(), 5.0, "{got:?}");
    assert_eq!(
        got,
        vec![vec![5.0, 0.0, 0.0, 0.0]],
        "the nearest (first on ties) takes it"
    );
    // The nearest pairs claim first: the second new point is nearer to the only old point.
    let near = [manifold(
        Vec3::Y,
        &[(p(0.03, 0.0), 0, 0.0), (p(0.001, 0.0), 0, 0.0)],
    )];
    assert_eq!(carried(&one, &near), vec![vec![0.0, 5.0]]);
    // Two old points, two new ones that both are nearest to the first old one.
    let two = [manifold(
        Vec3::Y,
        &[(p(0.0, 0.0), 0, 5.0), (p(0.04, 0.0), 0, 3.0)],
    )];
    let both = [manifold(
        Vec3::Y,
        &[(p(0.005, 0.0), 0, 0.0), (p(0.012, 0.0), 0, 0.0)],
    )];
    assert_eq!(carried(&two, &both), vec![vec![5.0, 3.0]]);
    // A feature id takes the old point with the same id first, once, wherever it is; the
    // points left over then match by position whatever their ids: the second id-7 point
    // takes the id-9 point where it was, and the id-8 point is too far from anything left.
    let ids = [manifold(
        Vec3::Y,
        &[(p(0.0, 0.0), 7, 5.0), (p(0.3, 0.0), 9, 2.0)],
    )];
    let renumbered = [manifold(
        Vec3::Y,
        &[
            (p(0.3, 0.0), 7, 0.0),
            (p(0.0, 0.0), 8, 0.0),
            (p(0.3, 0.0), 7, 0.0),
        ],
    )];
    assert_eq!(carried(&ids, &renumbered), vec![vec![5.0, 0.0, 2.0]]);
    // Ids that changed between steps (the incident vertices of a box resting on an equal,
    // aligned box lie on the side planes, so their clip ids flicker): matched by position.
    let before = [manifold(
        Vec3::Y,
        &[(p(-0.25, -0.25), 3, 1.0), (p(0.25, -0.25), 5, 2.0)],
    )];
    let after = [manifold(
        Vec3::Y,
        &[(p(0.2501, -0.25), 6, 0.0), (p(-0.2499, -0.25), 4, 0.0)],
    )];
    assert_eq!(carried(&before, &after), vec![vec![2.0, 1.0]]);
    // A point does not take one from a manifold facing another way (a neighbouring triangle
    // across a crease), but takes one with another id from a manifold facing its own way.
    let crease = [
        manifold(Vec3::X, &[(p(0.0, 0.0), 0, 5.0)]),
        manifold(Vec3::Y, &[(p(0.01, 0.0), 4, 2.0)]),
    ];
    let flat = [manifold(Vec3::Y, &[(p(0.0, 0.0), 0, 0.0)])];
    assert_eq!(carried(&crease, &flat), vec![vec![2.0]]);
    let crease = [manifold(Vec3::X, &[(p(0.0, 0.0), 0, 5.0)])];
    assert_eq!(carried(&crease, &flat), vec![vec![0.0]]);
    // Rounded shapes have the same feature id as their contact normal turns. Carrying an
    // impulse from the floor onto a wall would turn that support into a sideways push.
    let old = [manifold(Vec3::Y, &[(Vec3::ZERO, 7, 5.0)])];
    let sideways = [manifold(Vec3::X, &[(Vec3::ZERO, 7, 0.0)])];
    assert_eq!(carried(&old, &sideways), vec![vec![0.0]]);
}

/// Removing one collider of a static body (a piece of a level) wakes what rested on that
/// collider, not everything resting on the body's other colliders.
#[test]
fn removing_a_piece_of_static_geometry_wakes_only_what_rested_on_it() {
    let mut w = PhysicsWorld::default();
    let level = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            ..Default::default()
        })
        .unwrap();
    let mut pieces = Vec::new();
    for x in [-5.0f32, 5.0] {
        let mut d = ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(1.0, 0.5, 1.0),
        });
        d.local = Iso::from_position(Vec3::new(x, -0.5, 0.0));
        pieces.push(w.add_collider(level, &d).unwrap());
    }
    let half = Vec3::splat(0.25);
    let left = cuboid(&mut w, BodyKind::Dynamic, Vec3::new(-5.0, 0.25, 0.0), half);
    let right = cuboid(&mut w, BodyKind::Dynamic, Vec3::new(5.0, 0.25, 0.0), half);
    settle(&mut w, &[(level, left), (level, right)]);
    assert_eq!(w.contact_count(), 2);
    w.remove_collider(pieces[0]);
    assert!(w.is_awake(left), "the box on the removed piece sleeps on");
    assert!(!w.is_awake(right), "the box on the other piece woke");
}

/// Waking a box wakes what it touches, transitively, but not through a kinematic platform:
/// the box at the platform's other end stays asleep, and so does the platform (review finding
/// WAKE-THROUGH-KINEMATIC).
#[test]
fn waking_spreads_through_dynamic_bodies_only() {
    let mut w = PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let k = cuboid(
        &mut w,
        BodyKind::Kinematic,
        Vec3::new(0.0, -0.25, 0.0),
        Vec3::new(10.0, 0.25, 3.0),
    );
    let half = Vec3::splat(0.25);
    let a = cuboid(&mut w, BodyKind::Dynamic, Vec3::new(-8.0, 0.25, 0.0), half);
    let stack: Vec<BodyId> = (1..6)
        .map(|i| {
            let y = 0.25 + 0.5 * i as f32;
            cuboid(&mut w, BodyKind::Dynamic, Vec3::new(-8.0, y, 0.0), half)
        })
        .collect();
    let b = cuboid(&mut w, BodyKind::Dynamic, Vec3::new(8.0, 0.25, 0.0), half);
    let mut touching = vec![(k, a), (k, b), (a, stack[0])];
    touching.extend(stack.windows(2).map(|p| (p[0], p[1])));
    settle(&mut w, &touching);
    assert_eq!(w.contact_count(), touching.len());
    w.wake(a);
    assert!(w.is_awake(a));
    for (i, &s) in stack.iter().enumerate() {
        assert!(w.is_awake(s), "box {i} of the stack on a");
    }
    assert!(!w.is_awake(k), "the platform woke");
    assert!(!w.is_awake(b), "the box at the platform's other end woke");
    // Waking the platform itself (the game moves it) wakes only the platform; its moving
    // contacts wake what rests on it in the next step.
    w.wake(b);
    let b_on = w.is_awake(b);
    settle(&mut w, &touching);
    w.wake(k);
    assert!(b_on && w.is_awake(k) && !w.is_awake(a) && !w.is_awake(b));
}

/// Every contact sits in its two colliders' lists at the places it records and in the pair
/// map under its index; the lists hold nothing else; a destroyed collider's list is empty;
/// the move buffer holds each queued collider once, and only those.
fn check_bookkeeping(w: &PhysicsWorld) {
    for (k, c) in w.contacts.iter().enumerate() {
        assert!(c.a < c.b, "contact {k}: {} {}", c.a, c.b);
        for (end, col) in [c.a, c.b].into_iter().enumerate() {
            let list = &w.colliders[col as usize].contacts;
            assert_eq!(
                list.get(c.slots[end] as usize),
                Some(&(k as u32)),
                "contact {k}"
            );
        }
        assert_eq!(w.pairs.get(&(c.a, c.b)), Some(&(k as u32)));
    }
    assert_eq!(w.pairs.len(), w.contacts.len());
    let mut entries = 0;
    for (i, col) in w.colliders.iter().enumerate() {
        let i = i as u32;
        assert!(col.alive || col.contacts.is_empty(), "dead collider {i}");
        for &k in &col.contacts {
            let c = &w.contacts[k as usize];
            assert!(c.a == i || c.b == i, "collider {i} lists contact {k}");
        }
        entries += col.contacts.len();
        let queued = w.move_buffer.iter().filter(|&&x| x == i).count();
        assert_eq!(
            queued,
            usize::from(col.queued),
            "collider {i} in the move buffer"
        );
    }
    assert_eq!(entries, 2 * w.contacts.len());
    let expected_kinematic = w
        .bodies
        .iter()
        .filter(|b| b.alive && b.kind == BodyKind::Kinematic)
        .count();
    assert_eq!(w.kinematic_count, expected_kinematic, "kinematic_count");
    for (i, b) in w.bodies.iter().enumerate() {
        let expected = b.alive && b.awake && b.moves();
        let got = w
            .awake_words
            .get(i >> 6)
            .is_some_and(|&word| (word & (1u64 << (i & 63))) != 0);
        assert_eq!(got, expected, "awake_words bit for body {i}");
    }
}

/// A small deterministic random sequence.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % n as u64) as u32
    }
    fn unit(&mut self) -> f32 {
        self.next(1 << 20) as f32 / (1 << 20) as f32
    }
}

/// Contacts are found through each collider's own list (review findings: set_body_pose
/// and wake_body scanning every contact, J6, F3). Random creation, destruction, joints,
/// teleports, sensors and steps in a crowded space keep the lists exact.
#[test]
fn contact_lists_stay_exact_through_random_churn() {
    let mut w = PhysicsWorld::default();
    let ground = cuboid(
        &mut w,
        BodyKind::Static,
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(4.0, 0.5, 4.0),
    );
    let mut rng = Lcg(7);
    let mut bodies: Vec<BodyId> = Vec::new();
    let mut colliders: Vec<ColliderId> = Vec::new();
    let mut joints: Vec<JointId> = Vec::new();
    let (mut most_contacts, mut destroyed) = (0, 0);
    for round in 0..600 {
        let at =
            |r: &mut Lcg| Vec3::new(4.0 * r.unit() - 2.0, 2.0 * r.unit(), 4.0 * r.unit() - 2.0);
        match rng.next(12) {
            0..=2 => {
                let kind = [
                    BodyKind::Dynamic,
                    BodyKind::Dynamic,
                    BodyKind::Kinematic,
                    BodyKind::Static,
                ][rng.next(4) as usize];
                let b = w
                    .create_body(&BodyDesc {
                        kind,
                        position: at(&mut rng),
                        awake: rng.next(3) != 0,
                        ..Default::default()
                    })
                    .unwrap();
                for _ in 0..1 + rng.next(3) {
                    let mut d = ColliderDesc::new(Shape::Box {
                        half_extents: Vec3::splat(0.2 + 0.3 * rng.unit()),
                    });
                    d.local = Iso::from_position(Vec3::new(0.3 * rng.unit(), 0.0, 0.0));
                    d.sensor = rng.next(6) == 0;
                    colliders.push(w.add_collider(b, &d).unwrap());
                }
                bodies.push(b);
            }
            3 if !colliders.is_empty() => {
                let c = colliders.swap_remove(rng.next(colliders.len() as u32) as usize);
                w.remove_collider(c);
            }
            4 if !bodies.is_empty() => {
                let b = bodies.swap_remove(rng.next(bodies.len() as u32) as usize);
                w.destroy_body(b);
                destroyed += 1;
            }
            5 if bodies.len() >= 2 => {
                let a = bodies[rng.next(bodies.len() as u32) as usize];
                let b = bodies[rng.next(bodies.len() as u32) as usize];
                let other = if rng.next(4) == 0 { ground } else { b };
                if let Ok(j) = w.create_joint(&JointDesc {
                    body_a: a,
                    body_b: other,
                    frame_a: Iso::IDENTITY,
                    frame_b: Iso::IDENTITY,
                    kind: joint::JointKind::Ball,
                    collide_connected: rng.next(2) == 0,
                    user_data: 0,
                }) {
                    joints.push(j);
                }
            }
            6 if !joints.is_empty() => {
                let j = joints.swap_remove(rng.next(joints.len() as u32) as usize);
                w.destroy_joint(j);
            }
            7 if !bodies.is_empty() => {
                let b = bodies[rng.next(bodies.len() as u32) as usize];
                w.set_body_pose(b, Iso::from_position(at(&mut rng)));
            }
            8 if !bodies.is_empty() => {
                let b = bodies[rng.next(bodies.len() as u32) as usize];
                w.apply_impulse_at(b, Vec3::new(0.0, 1.0, 0.0), at(&mut rng));
            }
            _ => w.step(DT),
        }
        colliders.retain(|&c| w.collider_index(c).is_some());
        joints.retain(|&j| w.joint_index(j).is_some());
        check_bookkeeping(&w);
        most_contacts = most_contacts.max(w.contacts.len());
        assert!(w.state_hash() != 0, "round {round}");
    }
    assert!(
        most_contacts > 40 && destroyed > 20,
        "{most_contacts} contacts, {destroyed} bodies destroyed"
    );
}

/// Destroying a joint between a static level and a lamp queues the lamp's colliders for
/// pair finding, not the level's thousands (review finding J7); the pairs the joint kept
/// apart are still found again.
#[test]
fn destroying_a_joint_queues_only_the_moving_bodys_colliders() {
    let mut w = PhysicsWorld::new(Settings {
        gravity: Vec3::ZERO,
        ..Default::default()
    });
    let level = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            ..Default::default()
        })
        .unwrap();
    for i in 0..50 {
        let mut d = ColliderDesc::new(Shape::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        });
        d.local = Iso::from_position(Vec3::new(i as f32, 0.0, 0.0));
        w.add_collider(level, &d).unwrap();
    }
    // A crate sunk 5 mm into the first tile, fixed to the level without colliding.
    let lamp = cuboid(
        &mut w,
        BodyKind::Dynamic,
        Vec3::new(0.0, 0.795, 0.0),
        Vec3::splat(0.3),
    );
    let j = w
        .create_joint(&JointDesc {
            body_a: level,
            body_b: lamp,
            frame_a: Iso::from_position(Vec3::new(0.0, 0.795, 0.0)),
            frame_b: Iso::IDENTITY,
            kind: joint::JointKind::Fixed,
            collide_connected: false,
            user_data: 0,
        })
        .unwrap();
    w.step(DT);
    assert!(
        w.contacts.is_empty(),
        "the joint keeps the crate and the level apart"
    );
    let crate_collider = w.bodies[lamp.index as usize].colliders[0];
    w.destroy_joint(j);
    assert_eq!(w.move_buffer, vec![crate_collider]);
    check_bookkeeping(&w);
    w.step(DT);
    assert_eq!(w.contact_count(), 1, "the crate and the tile collide again");
    // A joint that never kept them apart queues nothing.
    let j = w
        .create_joint(&JointDesc {
            body_a: level,
            body_b: lamp,
            frame_a: Iso::IDENTITY,
            frame_b: Iso::IDENTITY,
            kind: joint::JointKind::Ball,
            collide_connected: true,
            user_data: 0,
        })
        .unwrap();
    w.step(DT);
    let before = w.move_buffer.clone();
    w.destroy_joint(j);
    assert_eq!(w.move_buffer, before);
}

/// The move buffer stays bounded however colliders are churned between steps: a slot reused
/// while its index is still queued is not queued again.
#[test]
fn churning_colliders_between_steps_keeps_the_move_buffer_bounded() {
    let mut w = PhysicsWorld::default();
    let b = w.create_body(&BodyDesc::default()).unwrap();
    let ball = ColliderDesc::new(Shape::Sphere { radius: 0.5 });
    for _ in 0..1000 {
        let c = w.add_collider(b, &ball).unwrap();
        w.remove_collider(c);
    }
    assert_eq!(w.move_buffer.len(), 1);
    check_bookkeeping(&w);
    w.add_collider(b, &ball).unwrap();
    w.step(DT);
    assert!(w.move_buffer.is_empty());
    check_bookkeeping(&w);
}
