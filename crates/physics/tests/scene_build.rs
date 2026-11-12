//! `ganbate_physics::scene::build` against cooked scenes whose entities are stretched, turned,
//! mirrored, sheared or nested: the colliders must sit where the entity is drawn, and the
//! instances must stay exactly where they were cooked until their body moves.

use std::f32::consts::FRAC_PI_2;
use std::ops::Range;

use ganbate_core::world::{EntityRecord, EntityTable, Instance, Mesh, Transform, World};
use ganbate_physics::scene::{self, ScenePhysics};
use ganbate_physics::{Aabb, BodyKind, ColliderId, Iso, Quat, QueryFilter, Settings, Shape, Vec3};

const DT: f32 = 1.0 / 60.0;

fn trs(t: [f32; 3], q: Quat, s: [f32; 3]) -> Transform {
    Transform::from_trs(t, q.to_array(), s).unwrap()
}

fn record(id: u64, parent: Option<u32>, t: Transform, instances: Range<u32>) -> EntityRecord {
    EntityRecord {
        id,
        name: format!("entity {id}"),
        parent,
        transform: t,
        instances,
        lights: 0..0,
    }
}

/// The entity table's data: each entity's components, given as the inside of a JSON object.
fn data(entities: &[(u64, &str)]) -> String {
    let list: Vec<String> = entities
        .iter()
        .map(|(id, c)| format!(r#""{id:016x}":{{"components":{{{c}}}}}"#))
        .collect();
    format!(r#"{{"entities":{{{}}}}}"#, list.join(","))
}

fn table(entities: Vec<EntityRecord>, data: String) -> EntityTable {
    EntityTable { entities, data }
}

fn json_quat(q: Quat) -> String {
    format!("[{},{},{},{}]", q.x, q.y, q.z, q.w)
}

/// Every collider of a freshly built scene (their slots are used once, in order).
fn colliders(s: &ScenePhysics) -> Vec<ColliderId> {
    (0..)
        .map(|index| ColliderId {
            index,
            generation: 0,
        })
        .take_while(|&c| s.world.collider_body(c).is_some())
        .collect()
}

fn world_aabb(s: &ScenePhysics, c: ColliderId) -> Aabb {
    s.world
        .collider_shape(c)
        .unwrap()
        .aabb(&s.world.collider_pose(c).unwrap())
}

fn point(m: &Transform, p: Vec3) -> Vec3 {
    let ganbate_core::Vec3(x, y, z) = m.point(ganbate_core::Vec3(p.x, p.y, p.z));
    Vec3::new(x, y, z)
}

/// The world bounds of a box collider as authored: its corners in the entity's frame, placed by
/// the entity's cooked transform.
fn authored_box(m: &Transform, center: Vec3, rotation: Quat, half: Vec3) -> Aabb {
    let corners: Vec<Vec3> = (0..8)
        .map(|i| {
            let c = Vec3::new(
                if i & 1 != 0 { half.x } else { -half.x },
                if i & 2 != 0 { half.y } else { -half.y },
                if i & 4 != 0 { half.z } else { -half.z },
            );
            point(m, center + rotation.rotate(c))
        })
        .collect();
    Aabb::from_points(&corners)
}

fn assert_aabb_close(got: Aabb, want: Aabb, tol: f32, what: &str) {
    assert!(
        (got.min - want.min).abs().max_element() <= tol
            && (got.max - want.max).abs().max_element() <= tol,
        "{what}: bounds {got:?}, drawn {want:?}"
    );
}

fn max_diff(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn linear_det(m: &[f32; 16]) -> f32 {
    let c = |i: usize| Vec3::new(m[i * 4], m[i * 4 + 1], m[i * 4 + 2]);
    c(0).dot(c(1).cross(c(2)))
}

fn cube() -> Mesh {
    Mesh {
        positions: vec![
            ganbate_core::Vec3(-0.5, -0.5, -0.5),
            ganbate_core::Vec3(0.5, 0.5, 0.5),
        ],
        indices: vec![],
        ..Default::default()
    }
}

// ---------------------------------------------------------------- colliders on scaled entities

#[test]
fn a_turned_capsule_on_a_stretched_entity_stretches_along_its_length() {
    // Turned a quarter about Z, the capsule lies along the entity's X; the entity is stretched
    // 4× along X, so the capsule is 4× longer and just as thick.
    let t = trs([0.0, 1.0, 0.0], Quat::IDENTITY, [4.0, 1.0, 1.0]);
    let turn = Quat::from_axis_angle(Vec3::Z, FRAC_PI_2);
    let c = format!(
        r#""collider":{{"shape":"capsule","half_height":0.5,"radius":0.5,"rotation":{}}},"rigid_body":{{"kind":"kinematic"}}"#,
        json_quat(turn)
    );
    let s = scene::build(
        &World::default(),
        &table(vec![record(1, None, t, 0..0)], data(&[(1, &c)])),
        Settings::default(),
    )
    .unwrap();
    let [id] = colliders(&s)[..] else {
        panic!("one collider")
    };
    match s.world.collider_shape(id) {
        Some(&Shape::Capsule {
            half_height,
            radius,
        }) => assert!(
            (half_height - 2.0).abs() < 1e-5 && (radius - 0.5).abs() < 1e-5,
            "capsule half height {half_height}, radius {radius}: should be 2 and 0.5"
        ),
        other => panic!("not a capsule: {other:?}"),
    }
    assert_aabb_close(
        world_aabb(&s, id),
        Aabb::new(Vec3::new(-2.5, 0.5, -0.5), Vec3::new(2.5, 1.5, 0.5)),
        1e-5,
        "the capsule",
    );
}

#[test]
fn boxes_turned_by_quarter_turns_on_stretched_turned_entities_fit_what_is_drawn() {
    let entity_turns = [
        Quat::IDENTITY,
        Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.7),
        Quat::from_axis_angle(Vec3::new(-2.0, 0.5, 1.0), -1.3),
    ];
    let box_turns = [
        Quat::from_axis_angle(Vec3::Y, FRAC_PI_2),
        Quat::from_axis_angle(Vec3::X, -FRAC_PI_2),
        Quat::from_axis_angle(Vec3::Z, FRAC_PI_2) * Quat::from_axis_angle(Vec3::X, FRAC_PI_2),
    ];
    let center = Vec3::new(0.2, -0.1, 0.3);
    let half = Vec3::new(0.5, 0.25, 0.1);
    for (i, &et) in entity_turns.iter().enumerate() {
        for (j, &bt) in box_turns.iter().enumerate() {
            let t = trs([1.0, 2.0, -3.0], et, [1.0, 2.0, 3.0]);
            let c = format!(
                r#""collider":{{"shape":"box","center":[0.2,-0.1,0.3],"half_extents":[0.5,0.25,0.1],"rotation":{}}}"#,
                json_quat(bt)
            );
            let s = scene::build(
                &World::default(),
                &table(vec![record(1, None, t, 0..0)], data(&[(1, &c)])),
                Settings::default(),
            )
            .unwrap();
            let [id] = colliders(&s)[..] else {
                panic!("one collider")
            };
            assert_aabb_close(
                world_aabb(&s, id),
                authored_box(&t, center, bt, half),
                1e-4,
                &format!("entity turn {i}, box turn {j}"),
            );
        }
    }
}

#[test]
fn any_turn_is_exact_on_an_evenly_scaled_entity_mirrored_or_not() {
    let et = Quat::from_axis_angle(Vec3::new(0.3, -1.0, 0.2), 0.9);
    let turns = [
        Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0), 0.6),
        Quat::from_axis_angle(Vec3::Z, 30f32.to_radians()),
    ];
    let scales = [
        [2.5, 2.5, 2.5],
        [-2.5, 2.5, 2.5],
        [2.5, -2.5, 2.5],
        [2.5, 2.5, -2.5],
        [-2.5, -2.5, -2.5],
    ];
    for bt in turns {
        for scale in scales {
            let t = trs([0.0, 1.0, 0.0], et, scale);
            let c = format!(
                r#""collider":{{"shape":"box","center":[0.2,-0.1,0.3],"half_extents":[0.5,0.25,0.1],"rotation":{}}},"rigid_body":{{}}"#,
                json_quat(bt)
            );
            let s = scene::build(
                &World::default(),
                &table(vec![record(1, None, t, 0..0)], data(&[(1, &c)])),
                Settings::default(),
            )
            .unwrap();
            let [id] = colliders(&s)[..] else {
                panic!("one collider")
            };
            assert_aabb_close(
                world_aabb(&s, id),
                authored_box(&t, Vec3::new(0.2, -0.1, 0.3), bt, Vec3::new(0.5, 0.25, 0.1)),
                1e-4,
                &format!("the box turned {bt:?} on scale {scale:?}"),
            );
        }
    }
}

#[test]
fn sheared_colliders_take_the_nearest_shape_of_their_kind() {
    // Turned 30° about Z on an entity stretched 2× along X, a box would be a parallelogram and a
    // capsule a slanted one. The box keeps its turn with each edge as long as drawn; the capsule
    // keeps its drawn axis, end to end.
    let et = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.4);
    let t = trs([1.0, 2.0, 3.0], et, [2.0, 1.0, 1.0]);
    let turn = Quat::from_axis_angle(Vec3::Z, 30f32.to_radians());
    let (half, hh) = (Vec3::new(0.5, 0.3, 0.2), 0.6);
    let edge = |e: Vec3| turn.rotate(e).mul_elem(Vec3::new(2.0, 1.0, 1.0)).length();
    for shape in ["box", "capsule"] {
        let c = format!(
            r#""collider":{{"shape":"{shape}","center":[0.1,0.2,0],"half_extents":[0.5,0.3,0.2],"half_height":0.6,"radius":0.25,"rotation":{}}},"rigid_body":{{}}"#,
            json_quat(turn)
        );
        let s = scene::build(
            &World::default(),
            &table(vec![record(1, None, t, 0..0)], data(&[(1, &c)])),
            Settings::default(),
        )
        .unwrap();
        let [id] = colliders(&s)[..] else {
            panic!("one collider")
        };
        let pose = s.world.collider_pose(id).unwrap();
        match s.world.collider_shape(id) {
            Some(&Shape::Box { half_extents }) => {
                let want = half.mul_elem(Vec3::new(edge(Vec3::X), edge(Vec3::Y), edge(Vec3::Z)));
                assert!(
                    (half_extents - want).length() < 1e-5,
                    "half extents {half_extents:?}, edges drawn {want:?}"
                );
                assert!(
                    pose.rotation.dot(et * turn).abs() > 0.99999,
                    "the box is turned {:?}, drawn {:?}",
                    pose.rotation,
                    et * turn
                );
            }
            Some(&Shape::Capsule {
                half_height,
                radius,
            }) => {
                for end in [-1.0, 1.0] {
                    let got = pose.point(Vec3::new(0.0, end * half_height, 0.0));
                    let want = point(
                        &t,
                        Vec3::new(0.1, 0.2, 0.0) + turn.rotate(Vec3::new(0.0, end * hh, 0.0)),
                    );
                    assert!((got - want).length() < 1e-5, "end {got:?}, drawn {want:?}");
                }
                let across = edge(Vec3::X).max(edge(Vec3::Z));
                assert!((radius - 0.25 * across).abs() < 1e-5, "radius {radius}");
            }
            other => panic!("{shape}: {other:?}"),
        }
    }
}

// ---------------------------------------------------------------- mirrored and sheared entities

#[test]
fn a_mirrored_entity_keeps_its_turn_and_its_collider_sits_where_it_is_drawn() {
    // Turned a quarter about Y and mirrored along its own X (scale −1): what a mirrored prop is.
    let r = Quat::from_axis_angle(Vec3::Y, FRAC_PI_2);
    let t = trs([1.0, 2.0, 3.0], r, [-1.0, 1.0, 1.0]);
    let c = r#""collider":{"center":[0.5,0.25,0],"half_extents":[0.5,0.25,0.25]},"rigid_body":{}"#;
    let s = scene::build(
        &World::default(),
        &table(vec![record(1, None, t, 0..0)], data(&[(1, c)])),
        Settings::default(),
    )
    .unwrap();
    let rotation = s.world.body_pose(s.bodies[0].body).unwrap().rotation;
    assert!(
        rotation.dot(r).abs() > 0.9999,
        "the body's rotation {rotation:?} is not the entity's turn {r:?}"
    );
    let [id] = colliders(&s)[..] else {
        panic!("one collider")
    };
    assert_aabb_close(
        world_aabb(&s, id),
        authored_box(
            &t,
            Vec3::new(0.5, 0.25, 0.0),
            Quat::IDENTITY,
            Vec3::new(0.5, 0.25, 0.25),
        ),
        1e-5,
        "the mirrored box",
    );
}

#[test]
fn mirrored_entities_get_proper_rotations_that_with_the_scale_give_their_transform() {
    let cases = [
        (Quat::from_axis_angle(Vec3::Y, FRAC_PI_2), [-1.0, 1.0, 1.0]),
        (Quat::from_axis_angle(Vec3::Y, FRAC_PI_2), [1.0, 1.0, -1.0]),
        (Quat::IDENTITY, [1.0, -2.0, 1.0]),
        (
            Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.7),
            [-1.5, 1.0, 2.0],
        ),
        (
            Quat::from_axis_angle(Vec3::new(-2.0, 0.5, 1.0), -1.3),
            [-1.0, -1.0, -1.0],
        ),
    ];
    let bt = Quat::from_axis_angle(Vec3::X, FRAC_PI_2);
    let (center, half) = (Vec3::new(0.3, 0.2, -0.1), Vec3::new(0.4, 0.3, 0.2));
    for (k, (q, scale)) in cases.into_iter().enumerate() {
        let t = trs([1.0, -2.0, 0.5], q, scale);
        let c = format!(
            r#""collider":{{"center":[0.3,0.2,-0.1],"half_extents":[0.4,0.3,0.2],"rotation":{}}},"rigid_body":{{}}"#,
            json_quat(bt)
        );
        let s = scene::build(
            &World::default(),
            &table(vec![record(1, None, t, 0..0)], data(&[(1, &c)])),
            Settings::default(),
        )
        .unwrap();
        let b = &s.bodies[0];
        let pose = s.world.body_pose(b.body).unwrap();
        // A proper rotation that, with the body's (signed) scale, gives the cooked transform.
        for (axis, col) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
            let want = Vec3::new(t.0[axis * 4], t.0[axis * 4 + 1], t.0[axis * 4 + 2]);
            let got = pose.rotation.rotate(col) * b.scale[axis];
            assert!(
                (got - want).length() < 1e-5,
                "case {k}: axis {axis} is {got:?} from the body, {want:?} cooked"
            );
        }
        let [id] = colliders(&s)[..] else {
            panic!("one collider")
        };
        assert_aabb_close(
            world_aabb(&s, id),
            authored_box(&t, center, bt, half),
            1e-4,
            &format!("case {k}"),
        );
    }
}

/// A kinematic entity with a box collider and one instance of a unit cube where it stands.
fn one_moving_entity(t: Transform) -> (World, ScenePhysics) {
    let world = World {
        meshes: vec![cube()],
        instances: vec![Instance {
            mesh: 0,
            transform: t,
        }],
        ..Default::default()
    };
    let s = scene::build(
        &world,
        &table(
            vec![record(1, None, t, 0..1)],
            data(&[(
                1,
                r#""collider":{"shape":"box"},"rigid_body":{"kind":"kinematic"}"#,
            )]),
        ),
        Settings::default(),
    )
    .unwrap();
    (world, s)
}

/// Checks that the instance of a kinematic entity cooked at `t` stays exactly where it was cooked
/// before any step, then moves rigidly with its body.
fn stays_put_then_moves_rigidly(what: &str, t: Transform) {
    let (world, mut s) = one_moving_entity(t);
    let cooked = world.instances[0].transform.0;
    let at_rest = s.instance_matrix(0, &cooked).unwrap();
    assert!(
        max_diff(&at_rest, &cooked) <= 1e-6,
        "{what}: before any step the instance is at {at_rest:?}, cooked at {cooked:?}"
    );
    // Moving it moves the instance rigidly: still mirrored or sheared, just carried along.
    let body = s.bodies[0].body;
    s.world.set_linear_velocity(body, Vec3::new(1.0, 0.0, 0.0));
    for _ in 0..30 {
        s.world.step(DT);
    }
    let moved = s.world.body_pose(body).unwrap().position - Vec3::new(t.0[12], t.0[13], t.0[14]);
    assert!(moved.x > 0.4, "{what}: the body moved {moved:?}");
    let mut want = cooked;
    for (i, d) in [moved.x, moved.y, moved.z].into_iter().enumerate() {
        want[12 + i] += d;
    }
    let now = s.instance_matrix(0, &cooked).unwrap();
    assert!(
        max_diff(&now, &want) <= 1e-5,
        "{what}: after moving the instance is at {now:?}, should be {want:?}"
    );
    assert_eq!(linear_det(&now) < 0.0, linear_det(&cooked) < 0.0, "{what}");
}

#[test]
fn a_mirrored_entity_stays_where_it_was_cooked_until_it_moves() {
    // Mirrored along X and turned 30° about Y.
    let t = trs(
        [1.0, 2.0, 3.0],
        Quat::from_axis_angle(Vec3::Y, 30f32.to_radians()),
        [-1.0, 1.0, 1.0],
    );
    stays_put_then_moves_rigidly("mirrored", t);
}

#[test]
fn a_sheared_entity_stays_where_it_was_cooked_until_it_moves() {
    // A child turned 45° about Z under a parent stretched 2× along X: sheared.
    let t = trs([0.0, 0.0, 0.0], Quat::IDENTITY, [2.0, 1.0, 1.0]).compose(trs(
        [0.0, 1.0, 0.0],
        Quat::from_axis_angle(Vec3::Z, 45f32.to_radians()),
        [1.0, 1.0, 1.0],
    ));
    stays_put_then_moves_rigidly("sheared", t);
}

#[test]
fn identity_poses_and_plain_colliders_come_out_exact() {
    // The common case: no turn, no mirror. The collider's placement is exactly the authored one.
    let t = trs([1.0, 2.0, 3.0], Quat::IDENTITY, [2.0, 0.5, 4.0]);
    let s = scene::build(
        &World::default(),
        &table(
            vec![record(1, None, t, 0..0)],
            data(&[(
                1,
                r#""collider":{"center":[0.25,-0.5,1],"half_extents":[1,2,0.5]},"rigid_body":{}"#,
            )]),
        ),
        Settings::default(),
    )
    .unwrap();
    let pose = s.world.body_pose(s.bodies[0].body).unwrap();
    assert_eq!(pose, Iso::new(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY));
    let [id] = colliders(&s)[..] else {
        panic!("one collider")
    };
    assert_eq!(
        s.world.collider_pose(id).unwrap(),
        Iso::new(Vec3::new(1.5, 1.75, 7.0), Quat::IDENTITY)
    );
    match s.world.collider_shape(id) {
        Some(&Shape::Box { half_extents }) => {
            assert_eq!(half_extents, Vec3::new(2.0, 1.0, 2.0))
        }
        other => panic!("not a box: {other:?}"),
    }
}

// ---------------------------------------------------------------- mirrored meshes

/// A floor quad two metres wide with its front face up (+Y) in its mesh's frame.
fn floor() -> Mesh {
    Mesh {
        positions: vec![
            ganbate_core::Vec3(-1.0, 0.0, -1.0),
            ganbate_core::Vec3(1.0, 0.0, -1.0),
            ganbate_core::Vec3(1.0, 0.0, 1.0),
            ganbate_core::Vec3(-1.0, 0.0, 1.0),
        ],
        indices: vec![0, 2, 1, 0, 3, 2],
        ..Default::default()
    }
}

/// A static mesh-collider floor: an entity at `entity` with one instance of [`floor`] at
/// `instance`.
fn mesh_floor(entity: Transform, instance: Transform) -> ScenePhysics {
    let world = World {
        meshes: vec![floor()],
        instances: vec![Instance {
            mesh: 0,
            transform: instance,
        }],
        ..Default::default()
    };
    scene::build(
        &world,
        &table(
            vec![record(1, None, entity, 0..1)],
            data(&[(1, r#""collider":{"shape":"mesh"}"#)]),
        ),
        Settings::default(),
    )
    .unwrap()
}

#[test]
fn a_mirrored_mesh_keeps_its_front_faces() {
    let mirror_x = trs([0.0, 0.0, 0.0], Quat::IDENTITY, [-1.0, 1.0, 1.0]);
    let mirror_z = trs([0.0, 0.0, 0.0], Quat::IDENTITY, [1.0, 1.0, -1.0]);
    for (what, s) in [
        (
            "plain",
            mesh_floor(Transform::IDENTITY, Transform::IDENTITY),
        ),
        ("a mirrored entity", mesh_floor(mirror_x, mirror_x)),
        (
            "a mirrored instance of a plain entity",
            mesh_floor(Transform::IDENTITY, mirror_z),
        ),
    ] {
        let down = s.world.raycast(
            Vec3::new(0.2, 3.0, 0.1),
            -Vec3::Y,
            10.0,
            &QueryFilter::default(),
        );
        let hit = down.unwrap_or_else(|| panic!("{what}: a ray from above misses the floor"));
        assert!(
            hit.normal.y > 0.999 && hit.distance > 2.99 && hit.distance < 3.01,
            "{what}: {hit:?}"
        );
        let up = s.world.raycast(
            Vec3::new(0.2, -3.0, 0.1),
            Vec3::Y,
            10.0,
            &QueryFilter::default(),
        );
        assert!(
            up.is_none(),
            "{what}: a ray from below hits the floor's back: {up:?}"
        );
    }
}

#[test]
fn a_ball_rests_on_a_mirrored_mesh_floor() {
    let mirror_x = trs([0.0, 0.0, 0.0], Quat::IDENTITY, [-1.0, 1.0, 1.0]);
    let floor_world = World {
        meshes: vec![floor()],
        instances: vec![Instance {
            mesh: 0,
            transform: mirror_x,
        }],
        ..Default::default()
    };
    let ball = trs([0.1, 1.0, 0.2], Quat::IDENTITY, [1.0, 1.0, 1.0]);
    let mut s = scene::build(
        &floor_world,
        &table(
            vec![record(1, None, mirror_x, 0..1), record(2, None, ball, 0..0)],
            data(&[
                (1, r#""collider":{"shape":"mesh"}"#),
                (
                    2,
                    r#""collider":{"shape":"sphere","radius":0.25},"rigid_body":{}"#,
                ),
            ]),
        ),
        Settings::default(),
    )
    .unwrap();
    for _ in 0..120 {
        s.world.step(DT);
    }
    let y = s.world.body_pose(s.bodies[1].body).unwrap().position.y;
    assert!(
        (y - 0.25).abs() < 0.02,
        "the ball should rest on the floor at 0.25, is at {y}"
    );
}

// ---------------------------------------------------------------- children of moving bodies

/// A cart (a dynamic box at y = 3) with a wheel: a child entity with only a sphere collider.
fn cart_and_wheel() -> ScenePhysics {
    let cart = trs([0.0, 3.0, 0.0], Quat::IDENTITY, [1.0, 1.0, 1.0]);
    let wheel = trs([0.6, 2.6, 0.0], Quat::IDENTITY, [1.0, 1.0, 1.0]);
    scene::build(
        &World::default(),
        &table(
            vec![record(1, None, cart, 0..0), record(2, Some(0), wheel, 0..0)],
            data(&[
                (1, r#""collider":{"shape":"box"},"rigid_body":{}"#),
                (2, r#""collider":{"shape":"sphere","radius":0.3}"#),
            ]),
        ),
        Settings::default(),
    )
    .unwrap()
}

#[test]
fn a_collider_on_a_child_of_a_moving_body_is_part_of_that_body() {
    let mut s = cart_and_wheel();
    let cart = s.bodies[0].body;
    assert_eq!(
        s.bodies.len(),
        1,
        "the wheel has no body of its own: {:?}",
        s.bodies
            .iter()
            .map(|b| (b.entity, s.world.body_kind(b.body)))
            .collect::<Vec<_>>()
    );
    let ids = colliders(&s);
    assert_eq!(ids.len(), 2);
    for &c in &ids {
        assert_eq!(s.world.collider_body(c), Some(cart));
    }
    assert_eq!(s.world.collider_user_data(ids[1]), Some(2));
    // A cubic metre of water and a ball of radius 0.3 m.
    let mass = 1000.0 + 1000.0 * 4.0 / 3.0 * std::f32::consts::PI * 0.027;
    let got = s.world.mass(cart).unwrap();
    assert!((got - mass).abs() < 0.1, "mass {got}, should be {mass}");
    for _ in 0..30 {
        s.world.step(DT);
    }
    let pose = s.world.body_pose(cart).unwrap();
    assert!(pose.position.y < 2.0, "the cart falls: {pose:?}");
    let wheel = s.world.collider_pose(ids[1]).unwrap().position;
    let want = pose.point(Vec3::new(0.6, -0.4, 0.0));
    assert!(
        (wheel - want).length() < 1e-4,
        "the wheel is at {wheel:?}, it should ride on the cart at {want:?}"
    );
}

#[test]
fn a_falling_cart_is_not_pushed_by_its_own_wheel() {
    // Meaningful with a narrow phase: a wheel built as static scenery inside the cart would push
    // it sideways (with no contacts at all it trivially stays on line).
    let mut s = cart_and_wheel();
    for _ in 0..30 {
        s.world.step(DT);
    }
    let p = s.world.body_pose(s.bodies[0].body).unwrap().position;
    assert!(
        p.x.abs() < 1e-4 && p.z.abs() < 1e-4,
        "the cart left its line: {p:?}"
    );
}

#[test]
fn a_mesh_collider_on_a_child_of_a_moving_body_is_refused() {
    let world = World {
        meshes: vec![floor()],
        instances: vec![Instance {
            mesh: 0,
            transform: Transform::IDENTITY,
        }],
        ..Default::default()
    };
    let err = scene::build(
        &world,
        &table(
            vec![
                record(1, None, Transform::IDENTITY, 0..0),
                record(2, Some(0), Transform::IDENTITY, 0..1),
            ],
            data(&[
                (1, r#""rigid_body":{"kind":"kinematic"}"#),
                (2, r#""collider":{"shape":"mesh"}"#),
            ]),
        ),
        Settings::default(),
    )
    .err()
    .expect("a mesh cannot move");
    assert!(
        err.contains("entity 2") && err.contains("entity 1") && err.contains("hull"),
        "{err}"
    );
}

#[test]
fn children_with_bodies_of_their_own_and_children_of_static_bodies() {
    let at = |y: f32| trs([0.0, y, 0.0], Quat::IDENTITY, [1.0, 1.0, 1.0]);
    let s = scene::build(
        &World::default(),
        &table(
            vec![
                record(1, None, at(3.0), 0..0),
                record(2, Some(0), at(5.0), 0..0),
                record(3, None, at(0.0), 0..0),
                record(4, Some(2), at(1.0), 0..0),
            ],
            data(&[
                (1, r#""collider":{},"rigid_body":{}"#),
                (2, r#""collider":{},"rigid_body":{"kind":"kinematic"}"#),
                (3, r#""collider":{},"rigid_body":{"kind":"static"}"#),
                (4, r#""collider":{"shape":"sphere"}"#),
            ]),
        ),
        Settings::default(),
    )
    .unwrap();
    let kinds: Vec<_> = s
        .bodies
        .iter()
        .map(|b| (b.entity, s.world.body_kind(b.body)))
        .collect();
    assert_eq!(
        &kinds[..3],
        &[
            (1, Some(BodyKind::Dynamic)),
            (2, Some(BodyKind::Kinematic)),
            (3, Some(BodyKind::Static)),
        ]
    );
    // The child of the static body is static too, where it was drawn.
    let ids = colliders(&s);
    let c = ids[3];
    assert_eq!(s.world.collider_user_data(c), Some(4));
    let body = s.world.collider_body(c).unwrap();
    assert_eq!(s.world.body_kind(body), Some(BodyKind::Static));
    let p = s.world.collider_pose(c).unwrap().position;
    assert!((p - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6, "{p:?}");
}

#[test]
fn a_child_collider_under_a_stretched_turned_parent_sits_where_it_is_drawn() {
    // A parent body stretched unevenly and turned; its child is turned a quarter about X and
    // scaled again, with an off-centre box: the box rides on the parent's body, where drawn.
    let parent = trs(
        [3.0, 1.0, -1.0],
        Quat::from_axis_angle(Vec3::new(0.2, 1.0, -0.4), 0.8),
        [1.0, 2.0, 1.0],
    );
    let child_turn = Quat::from_axis_angle(Vec3::X, FRAC_PI_2);
    let child = parent.compose(trs([0.0, 0.75, 0.0], child_turn, [0.5, 0.25, 0.5]));
    let s = scene::build(
        &World::default(),
        &table(
            vec![
                record(1, None, parent, 0..0),
                record(2, Some(0), child, 0..0),
            ],
            data(&[
                (1, r#""rigid_body":{"kind":"kinematic"}"#),
                (
                    2,
                    r#""collider":{"center":[0.1,0.2,0.3],"half_extents":[0.5,0.4,0.3]}"#,
                ),
            ]),
        ),
        Settings::default(),
    )
    .unwrap();
    let [id] = colliders(&s)[..] else {
        panic!("one collider")
    };
    assert_eq!(s.world.collider_body(id), Some(s.bodies[0].body));
    assert_aabb_close(
        world_aabb(&s, id),
        authored_box(
            &child,
            Vec3::new(0.1, 0.2, 0.3),
            Quat::IDENTITY,
            Vec3::new(0.5, 0.4, 0.3),
        ),
        1e-4,
        "the child's box",
    );
}

// ---------------------------------------------------------------- what bodies carry

#[test]
fn bodies_carry_the_entities_below_them_that_have_no_body_of_their_own() {
    // 0 cart (dynamic) ─ 1 axle (a mesh only) ─ 2 wheel (a collider)
    //                  └ 3 trailer (its own dynamic body) ─ 4 hitch (a mesh only)
    // 5 wall (static scenery) ─ 6 poster (a mesh only)
    let at = |x: f32, y: f32| trs([x, y, 0.0], Quat::IDENTITY, [1.0, 1.0, 1.0]);
    let instances: Vec<Instance> = [at(0.0, 3.0), at(0.0, 2.5), at(0.6, 2.4), at(-2.0, 3.0)]
        .into_iter()
        .chain([at(-1.2, 3.0), at(5.0, 1.0), at(5.0, 1.5)])
        .map(|transform| Instance { mesh: 0, transform })
        .collect();
    let world = World {
        meshes: vec![cube()],
        instances: instances.clone(),
        ..Default::default()
    };
    let mut s = scene::build(
        &world,
        &table(
            vec![
                record(1, None, at(0.0, 3.0), 0..1),
                record(2, Some(0), at(0.0, 2.5), 1..2),
                record(3, Some(1), at(0.6, 2.4), 2..3),
                record(4, Some(0), at(-2.0, 3.0), 3..4),
                record(5, Some(3), at(-1.2, 3.0), 4..5),
                record(6, None, at(5.0, 1.0), 5..6),
                record(7, Some(5), at(5.0, 1.5), 6..7),
            ],
            data(&[
                (1, r#""collider":{},"rigid_body":{}"#),
                (3, r#""collider":{"shape":"sphere","radius":0.3}"#),
                (4, r#""collider":{},"rigid_body":{"gravity_scale":0}"#),
                (6, r#""collider":{"half_extents":[0.5,1,2]}"#),
            ]),
        ),
        Settings::default(),
    )
    .unwrap();
    let carries: Vec<(u64, Vec<u32>)> = s
        .bodies
        .iter()
        .map(|b| (b.entity, b.carries.clone()))
        .collect();
    assert_eq!(
        carries,
        vec![(1, vec![0, 1, 2]), (4, vec![3, 4]), (6, vec![5])]
    );
    for _ in 0..30 {
        s.world.step(DT);
    }
    // The axle's and the wheel's instances fall with the cart; the trailer's stay with it.
    let cart = s.world.body_pose(s.bodies[0].body).unwrap().position;
    let dropped = 3.0 - cart.y;
    assert!(dropped > 1.0, "the cart fell {dropped}");
    for (body, inst, drop) in [(0, 1, dropped), (0, 2, dropped), (1, 4, 0.0), (2, 6, 0.0)] {
        let cooked = instances[inst].transform.0;
        let now = s.instance_matrix(body, &cooked).unwrap();
        let mut want = cooked;
        want[13] -= drop;
        assert!(
            max_diff(&now, &want) < 1e-4,
            "instance {inst} on body {body}: {now:?}, should be {want:?}"
        );
    }
}

// ---------------------------------------------------------------- hulls of detailed meshes

#[test]
fn a_hull_of_a_detailed_mesh_keeps_within_the_vertex_limit() {
    // A ball of 3280 vertices: its hull is made from at most MAX_HULL_VERTICES extreme points.
    let mut positions = Vec::new();
    for i in 0..=40 {
        let (st, ct) = ganbate_physics::math::sin_cos(std::f32::consts::PI * i as f32 / 40.0);
        for j in 0..80 {
            let (sp, cp) = ganbate_physics::math::sin_cos(std::f32::consts::PI * j as f32 / 40.0);
            positions.push(ganbate_core::Vec3(0.5 * st * cp, 0.5 * ct, 0.5 * st * sp));
        }
    }
    let t = trs([0.0, 2.0, 0.0], Quat::IDENTITY, [1.0, 1.0, 1.0]);
    let world = World {
        meshes: vec![Mesh {
            positions,
            ..Default::default()
        }],
        instances: vec![Instance {
            mesh: 0,
            transform: t,
        }],
        ..Default::default()
    };
    let s = scene::build(
        &world,
        &table(
            vec![record(1, None, t, 0..1)],
            data(&[(1, r#""collider":{"shape":"hull"},"rigid_body":{}"#)]),
        ),
        Settings::default(),
    )
    .unwrap();
    let [id] = colliders(&s)[..] else {
        panic!("one collider")
    };
    match s.world.collider_shape(id) {
        Some(Shape::Hull(h)) => {
            assert!(h.vertices.len() <= ganbate_physics::hull::MAX_HULL_VERTICES);
            let b = Aabb::from_points(&h.vertices);
            assert_aabb_close(
                b,
                Aabb::new(Vec3::splat(-0.5), Vec3::splat(0.5)),
                1e-5,
                "the hull keeps the ball's bounds",
            );
        }
        other => panic!("not a hull: {other:?}"),
    }
}

// ---------------------------------------------------------------- what moved

#[test]
fn only_bodies_that_moved_are_named_after_a_step() {
    let at = |x: f32| trs([x, 1.0, 0.0], Quat::IDENTITY, [1.0, 1.0, 1.0]);
    let mut s = scene::build(
        &World::default(),
        &table(
            vec![
                record(1, None, at(0.0), 0..0),
                record(2, None, at(3.0), 0..0),
                record(3, None, at(6.0), 0..0),
                record(4, None, at(9.0), 0..0),
                record(5, None, at(12.0), 0..0),
            ],
            data(&[
                // Static scenery, a kinematic body standing still, one moving, a dynamic body
                // floating without gravity (it falls asleep), and one falling.
                (1, r#""collider":{}"#),
                (2, r#""collider":{},"rigid_body":{"kind":"kinematic"}"#),
                (
                    3,
                    r#""collider":{},"rigid_body":{"kind":"kinematic","velocity":[0,0,1]}"#,
                ),
                (4, r#""collider":{},"rigid_body":{"gravity_scale":0}"#),
                (5, r#""collider":{},"rigid_body":{}"#),
            ]),
        ),
        Settings::default(),
    )
    .unwrap();
    let mut shown = s.poses();
    assert_eq!(s.moved_since(&mut shown), Vec::<usize>::new());
    s.world.step(DT);
    assert_eq!(s.moved_since(&mut shown), vec![2, 4]);
    assert_eq!(
        s.moved_since(&mut shown),
        Vec::<usize>::new(),
        "nothing new"
    );
    for _ in 0..60 {
        s.world.step(DT);
    }
    assert!(
        !s.world.is_awake(s.bodies[3].body),
        "the floating body sleeps"
    );
    s.world.step(DT);
    assert_eq!(s.moved_since(&mut shown), vec![2, 4]);
    // Every body that did not move is drawn exactly where it was cooked.
    for i in [0, 1, 3] {
        assert_eq!(s.motion(i), Some(Iso::IDENTITY), "body {i}");
        assert_eq!(s.entity_matrix(i), Some(at(3.0 * i as f32).0), "body {i}");
    }
}
