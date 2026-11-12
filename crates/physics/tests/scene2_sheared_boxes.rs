//! Box colliders whose drawn shape is sheared: a box turned against an uneven scale, or any box on
//! an entity that its parents shear (a turned child of a stretched parent). What collides must be
//! what is drawn, not a box many times its size: keeping the box's turn with each edge as long as
//! drawn made colliders up to 45 times the drawn volume, turned so that they missed most of it.

use std::f32::consts::FRAC_PI_4;
use std::ops::Range;

use ganbate_core::world::{EntityRecord, EntityTable, Transform, World};
use ganbate_physics::scene::{self, ScenePhysics};
use ganbate_physics::{ColliderId, Mat3, Quat, QueryFilter, Settings, Shape, Vec3};

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

fn build(entities: Vec<EntityRecord>, components: &[(u64, &str)]) -> ScenePhysics {
    scene::build(
        &World::default(),
        &EntityTable {
            entities,
            data: data(components),
        },
        Settings::default(),
    )
    .unwrap()
}

fn json_vec(v: Vec3) -> String {
    format!("[{},{},{}]", v.x, v.y, v.z)
}

fn json_quat(q: Quat) -> String {
    format!("[{},{},{},{}]", q.x, q.y, q.z, q.w)
}

/// A dynamic entity cooked at `t` whose only collider is the box `half` turned by `rotation`
/// about `center` (of density 1000).
fn box_entity(t: Transform, center: Vec3, rotation: Quat, half: Vec3) -> ScenePhysics {
    let c = format!(
        r#""collider":{{"shape":"box","center":{},"half_extents":{},"rotation":{}}},"rigid_body":{{}}"#,
        json_vec(center),
        json_vec(half),
        json_quat(rotation)
    );
    build(vec![record(1, None, t, 0..0)], &[(1, &c)])
}

fn the_collider(s: &ScenePhysics) -> ColliderId {
    let id = ColliderId {
        index: 0,
        generation: 0,
    };
    assert!(s.world.collider_body(id).is_some());
    id
}

fn point(t: &Transform, p: Vec3) -> Vec3 {
    let ganbate_core::Vec3(x, y, z) = t.point(ganbate_core::Vec3(p.x, p.y, p.z));
    Vec3::new(x, y, z)
}

fn linear(t: &Transform) -> Mat3 {
    let m = &t.0;
    Mat3::from_cols(
        Vec3::new(m[0], m[1], m[2]),
        Vec3::new(m[4], m[5], m[6]),
        Vec3::new(m[8], m[9], m[10]),
    )
}

/// A box as it is drawn: its centre and its three half edges (columns) in the world.
struct Drawn {
    center: Vec3,
    edges: Mat3,
}

impl Drawn {
    fn of(t: &Transform, center: Vec3, rotation: Quat, half: Vec3) -> Drawn {
        let l = linear(t);
        Drawn {
            center: point(t, center),
            edges: Mat3::from_cols(
                l.mul_vec(rotation.rotate(Vec3::X)) * half.x,
                l.mul_vec(rotation.rotate(Vec3::Y)) * half.y,
                l.mul_vec(rotation.rotate(Vec3::Z)) * half.z,
            ),
        }
    }

    fn volume(&self) -> f32 {
        8.0 * self.edges.determinant().abs()
    }

    /// The length of the longest half edge.
    fn size(&self) -> f32 {
        self.edges
            .cols
            .iter()
            .fold(0.0f32, |m, c| m.max(c.length()))
    }

    /// `p` in the drawn box's own coordinates, where the box is the cube [-1, 1]³.
    fn local(&self, p: Vec3) -> Vec3 {
        self.edges.inverse().unwrap().mul_vec(p - self.center)
    }

    fn corner(&self, i: usize) -> Vec3 {
        let s = |bit: usize| if i & bit != 0 { 1.0 } else { -1.0 };
        self.center + self.edges.mul_vec(Vec3::new(s(1), s(2), s(4)))
    }

    /// How far along the unit `dir` a ray from `origin` (outside) enters the drawn box.
    fn ray(&self, origin: Vec3, dir: Vec3) -> Option<f32> {
        let to = self.edges.inverse().unwrap();
        let (o, d) = (to.mul_vec(origin - self.center), to.mul_vec(dir));
        let (mut enter, mut exit) = (f32::NEG_INFINITY, f32::INFINITY);
        for k in 0..3 {
            if d[k].abs() < 1e-12 {
                if o[k].abs() > 1.0 {
                    return None;
                }
                continue;
            }
            let (a, b) = ((-1.0 - o[k]) / d[k], (1.0 - o[k]) / d[k]);
            enter = enter.max(a.min(b));
            exit = exit.min(a.max(b));
        }
        (enter <= exit && enter >= 0.0).then_some(enter)
    }
}

/// A small xorshift generator, so the cases are the same on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * ((self.next() >> 40) as f32 / (1u64 << 24) as f32)
    }
    fn v3(&mut self, a: f32, b: f32) -> Vec3 {
        Vec3::new(self.range(a, b), self.range(a, b), self.range(a, b))
    }
    fn unit(&mut self) -> Vec3 {
        loop {
            let v = self.v3(-1.0, 1.0);
            let l = v.length();
            if l > 0.1 && l <= 1.0 {
                return v / l;
            }
        }
    }
    fn quat(&mut self) -> Quat {
        let axis = self.unit();
        Quat::from_axis_angle(axis, self.range(-3.1, 3.1))
    }
    /// A turned entity stretched 0.3 to 3 times along each axis, mirrored along one or three axes
    /// in some cases.
    fn entity(&mut self, case: usize) -> Transform {
        let turn = self.quat();
        let mut s = [
            self.range(0.3, 3.0),
            self.range(0.3, 3.0),
            self.range(0.3, 3.0),
        ];
        if case % 2 == 1 {
            let k = (self.next() % 3) as usize;
            s[k] = -s[k];
        }
        if case % 7 == 0 {
            s = s.map(|v| -v.abs());
        }
        trs(self.v3(-5.0, 5.0).to_array(), turn, s)
    }
}

/// Checks that the only collider of `s`, made for the box drawn as `drawn`, is exactly that box:
/// rays from all round hit it where they hit the drawn box, and its body has the drawn box's
/// mass (density 1000) and centre of mass.
fn assert_exactly_drawn(s: &ScenePhysics, drawn: &Drawn, what: &str) {
    let size = drawn.size();
    let body = s.bodies[0].body;
    let mass = s.world.mass(body).unwrap();
    let want = 1000.0 * drawn.volume();
    assert!(
        (mass - want).abs() <= 1e-3 * want,
        "{what}: mass {mass}, the drawn box has {want} (collider {:?})",
        s.world.collider_shape(the_collider(s))
    );
    let com = s.world.center_of_mass(body).unwrap();
    assert!(
        (com - drawn.center).length() <= 1e-4 * size,
        "{what}: centre of mass {com:?}, drawn centre {:?}",
        drawn.center
    );
    let mut rng = Rng(0x5eed_0000_0000_0001);
    for k in 0..48 {
        // From a sphere well outside, towards a point inside the drawn box.
        let target = drawn.center + drawn.edges.mul_vec(rng.v3(-0.95, 0.95));
        let origin = drawn.center + rng.unit() * (4.0 * size);
        let dir = (target - origin).normalize_or_zero();
        let want = drawn
            .ray(origin, dir)
            .expect("the ray aims inside the drawn box");
        let got = s
            .world
            .raycast(origin, dir, 10.0 * size, &QueryFilter::default())
            .map(|h| h.distance);
        assert!(
            got.is_some_and(|d| (d - want).abs() <= 1e-3 * size),
            "{what}: ray {k} from {origin:?} along {dir:?} hits the collider at {got:?}, the drawn \
             box at {want}"
        );
    }
}

/// How much of the drawn box the only collider of `s` covers, and how much of the collider lies
/// in the drawn box, by sampling both on a grid: their intersection over union.
fn overlap(s: &ScenePhysics, drawn: &Drawn) -> f32 {
    let id = the_collider(s);
    let pose = s.world.collider_pose(id).unwrap();
    let Some(&Shape::Box { half_extents: h }) = s.world.collider_shape(id) else {
        panic!("not a box: {:?}", s.world.collider_shape(id));
    };
    let n = 12;
    let at = |i: usize| -1.0 + (2 * i + 1) as f32 / n as f32;
    let (mut drawn_in_box, mut box_in_drawn) = (0, 0);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let u = Vec3::new(at(i), at(j), at(k));
                let p = drawn.center + drawn.edges.mul_vec(u);
                let q = pose.inverse_point(p).abs();
                if q.x <= h.x && q.y <= h.y && q.z <= h.z {
                    drawn_in_box += 1;
                }
                let b = drawn.local(pose.point(u.mul_elem(h))).abs();
                if b.max_element() <= 1.0 {
                    box_in_drawn += 1;
                }
            }
        }
    }
    let total = (n * n * n) as f32;
    let (vd, vb) = (drawn.volume(), 8.0 * h.x * h.y * h.z);
    let both = 0.5 * (drawn_in_box as f32 / total * vd + box_in_drawn as f32 / total * vb);
    both / (vd + vb - both)
}

/// A box collider the verifier found: its entity's cooked world matrix (a child of a stretched,
/// turned parent), and the box's turn (x, y, z, w), centre and half extents.
struct Found {
    matrix: [f32; 16],
    turn: [f32; 4],
    center: [f32; 3],
    half: [f32; 3],
}

impl Found {
    fn turn(&self) -> Quat {
        let [x, y, z, w] = self.turn;
        Quat::new(x, y, z, w)
    }
}

/// The verifier's worst three: the drawn edges meet at 7 to 15 degrees, and the boxes that kept
/// their turn were 14, 21 and 45 times the drawn volume.
#[rustfmt::skip]
const FOUND: [Found; 3] = [
    Found {
        matrix: [
            1.2797432, -1.161298, -2.1249619, 0.0,
            -0.07204284, -0.008680254, 0.35254464, 0.0,
            -0.048368663, -0.8694838, -1.9846839, 0.0,
            -3.7677753, -0.9683483, 2.5602865, 1.0,
        ],
        turn: [0.107113294, -0.10233876, -0.2133764, 0.96567285],
        center: [0.0015660524, -0.26384717, 0.11618549],
        half: [0.6001219, 0.4358039, 0.79317343],
    },
    Found {
        matrix: [
            -0.23524663, -0.36574808, 0.17741539, 0.0,
            -3.0210423, -1.1063524, 0.05908221, 0.0,
            -1.095313, -0.22557092, 0.30883121, 0.0,
            -3.318437, -3.616696, 1.899169, 1.0,
        ],
        turn: [-0.09473255, 0.16314963, 0.39405748, 0.8995147],
        center: [0.14549136, -0.48496097, -0.097032964],
        half: [0.76658326, 0.27958328, 0.3150981],
    },
    Found {
        matrix: [
            5.1137476, -0.29334533, 0.21661772, 0.0,
            0.16966183, 0.17322771, -0.15874827, 0.0,
            -0.18406504, -0.045143764, -0.41335872, 0.0,
            -4.8498, 2.4868789, -2.8848872, 1.0,
        ],
        turn: [-0.3846591, -0.82047313, -0.24706155, 0.34325162],
        center: [-0.085892975, 0.44209307, 0.12945044],
        half: [0.5322867, 0.48946363, 0.85633343],
    },
];

#[test]
fn the_worst_boxes_the_verifier_found_are_exactly_what_is_drawn() {
    for (k, f) in FOUND.iter().enumerate() {
        let t = Transform(f.matrix);
        let (q, c, h) = (
            f.turn(),
            Vec3::from_array(f.center),
            Vec3::from_array(f.half),
        );
        let s = box_entity(t, c, q, h);
        assert_exactly_drawn(&s, &Drawn::of(&t, c, q, h), &format!("case {k}"));
    }
}

#[test]
fn a_cube_turned_on_a_flattened_entity_is_not_blown_up() {
    // A plank-like entity, 10 m long and 10 cm thick, with its cube collider turned 45 degrees
    // about Z: drawn, the cube is a flat rhombus of a cubic metre. Kept turned with its edges as
    // drawn (7.07 m), it was a box of 50 cubic metres reaching 5 m above and below the plank.
    let t = trs([0.0, 2.0, 0.0], Quat::IDENTITY, [10.0, 0.1, 1.0]);
    let turn = Quat::from_axis_angle(Vec3::Z, FRAC_PI_4);
    let half = Vec3::splat(0.5);
    let s = box_entity(t, Vec3::ZERO, turn, half);
    let drawn = Drawn::of(&t, Vec3::ZERO, turn, half);
    assert!((drawn.volume() - 1.0).abs() < 1e-4);
    assert_exactly_drawn(&s, &drawn, "the turned cube");
    let b = s
        .world
        .collider_shape(the_collider(&s))
        .unwrap()
        .aabb(&s.world.collider_pose(the_collider(&s)).unwrap());
    assert!(
        b.max.y < 2.08 && b.min.y > 1.92,
        "the collider reaches from {} to {} in height; the plank from 1.93 to 2.07",
        b.min.y,
        b.max.y
    );
}

#[test]
fn a_long_sheared_plate_keeps_its_drawn_thickness() {
    // Every half thickness is above MIN_EXTENT, but the point-cloud hull's
    // welding tolerance at the ends is larger than the thickness of the plate.
    let t = trs([0.0; 3], Quat::IDENTITY, [60_000.0, 0.001, 1.0]);
    let turn = Quat::from_axis_angle(Vec3::Z, FRAC_PI_4);
    let half = Vec3::splat(0.5);
    let s = box_entity(t, Vec3::ZERO, turn, half);
    let collider = the_collider(&s);
    let bounds = s
        .world
        .collider_shape(collider)
        .unwrap()
        .aabb(&s.world.collider_pose(collider).unwrap());
    let drawn = Drawn::of(&t, Vec3::ZERO, turn, half);
    let highest = (0..8)
        .map(|i| drawn.corner(i).y)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (bounds.max.y - highest).abs() < 1e-6,
        "height {} versus {highest}",
        bounds.max.y
    );
}

#[test]
fn any_box_on_an_entity_its_parents_shear_is_exactly_what_is_drawn() {
    // The verifier's 200 random children of stretched, turned (and some mirrored) parents, each
    // turned and stretched again, with a randomly turned box.
    let mut rng = Rng(0x3141_5926_5358_9793);
    for case in 0..200 {
        let parent = rng.entity(case);
        let child = parent.compose(trs(
            rng.v3(-1.0, 1.0).to_array(),
            rng.quat(),
            rng.v3(0.4, 2.0).to_array(),
        ));
        let (turn, center, half) = (rng.quat(), rng.v3(-0.5, 0.5), rng.v3(0.2, 1.0));
        let s = box_entity(child, center, turn, half);
        assert_exactly_drawn(
            &s,
            &Drawn::of(&child, center, turn, half),
            &format!("case {case}"),
        );
    }
}

#[test]
fn a_box_drawn_square_under_a_shearing_parent_is_an_exact_box() {
    // A child turned 30 degrees under a parent stretched 3 : 0.5 : 1.5 is sheared, but a box
    // turned back by the same 30 degrees lies along the parent's axes: drawn, it is a box.
    let parent = trs(
        [1.0, 2.0, -1.0],
        Quat::from_axis_angle(Vec3::new(0.2, 1.0, -0.3), 0.9),
        [3.0, 0.5, 1.5],
    );
    let child_turn = Quat::from_axis_angle(Vec3::new(1.0, 0.5, 0.2), 30f32.to_radians());
    let child = parent.compose(trs([0.5, 0.2, 0.0], child_turn, [1.0, 1.0, 1.0]));
    let (center, turn, half) = (
        Vec3::new(0.1, -0.2, 0.3),
        child_turn.conjugate(),
        Vec3::new(0.5, 0.4, 0.3),
    );
    let s = box_entity(child, center, turn, half);
    let drawn = Drawn::of(&child, center, turn, half);
    match s.world.collider_shape(the_collider(&s)) {
        Some(&Shape::Box { half_extents }) => {
            let want = Vec3::new(1.5, 0.2, 0.45);
            assert!(
                (half_extents - want).abs().max_element() < 1e-4,
                "half extents {half_extents:?}, drawn {want:?}"
            );
        }
        other => panic!("drawn as a box, it should be one: {other:?}"),
    }
    assert_exactly_drawn(&s, &drawn, "the box along the parent's axes");
    // Its corners are the drawn corners.
    let pose = s.world.collider_pose(the_collider(&s)).unwrap();
    let Some(&Shape::Box { half_extents: h }) = s.world.collider_shape(the_collider(&s)) else {
        unreachable!()
    };
    for i in 0..8 {
        let c = drawn.corner(i);
        let local = pose.inverse_point(c).abs();
        assert!(
            (local - h).abs().max_element() < 1e-4,
            "drawn corner {i} at {c:?} is not a corner of the collider ({local:?} of {h:?})"
        );
    }
}

#[test]
fn a_box_turned_against_an_uneven_scale_keeps_its_turn_only_while_close() {
    // Turned boxes on stretched (not sheared) entities: each collider is either the box keeping
    // its turn with each edge as long as drawn, at most a quarter larger than drawn and
    // overlapping it well, or exactly the drawn shape. None is blown up.
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let (mut kept, mut exact, mut worst) = (0, 0, 1.0f32);
    for case in 0..400 {
        let t = rng.entity(case);
        let (turn, center, half) = (rng.quat(), rng.v3(-0.5, 0.5), rng.v3(0.2, 1.0));
        let s = box_entity(t, center, turn, half);
        let drawn = Drawn::of(&t, center, turn, half);
        let shape = s.world.collider_shape(the_collider(&s)).unwrap().clone();
        let volume = s.world.mass(s.bodies[0].body).unwrap() / 1000.0;
        assert!(
            volume <= 1.2501 * drawn.volume(),
            "case {case}: the collider {shape:?} holds {volume} m³, the drawn box {}",
            drawn.volume()
        );
        if let Shape::Box { .. } = shape {
            let iou = overlap(&s, &drawn);
            assert!(
                iou >= 0.5,
                "case {case}: the kept box overlaps the drawn one by {iou} only"
            );
            worst = worst.min(iou);
            kept += 1;
        } else {
            assert_exactly_drawn(&s, &drawn, &format!("case {case}"));
            exact += 1;
        }
    }
    println!("{kept} boxes kept their turn (overlap at worst {worst}), {exact} exactly drawn");
    assert!(kept > 50 && exact > 50, "{kept} kept, {exact} exact");
}

#[test]
fn a_sheared_box_falls_and_rests_on_its_drawn_corners() {
    // The verifier's 45-times case, moved to 3 m above the ground and dropped: kept turned, the
    // blown-up box held the drawn shape up in the air. Now the drawn shape itself lands.
    let found = &FOUND[2];
    let mut m = found.matrix;
    m[12..15].copy_from_slice(&[0.0, 3.0, 0.0]);
    let t = Transform(m);
    let (turn, center, half) = (
        found.turn(),
        Vec3::from_array(found.center),
        Vec3::from_array(found.half),
    );
    let c = format!(
        r#""collider":{{"shape":"box","center":{},"half_extents":{},"rotation":{}}},"rigid_body":{{}}"#,
        json_vec(center),
        json_vec(half),
        json_quat(turn)
    );
    let ground = trs([0.0, -0.5, 0.0], Quat::IDENTITY, [1.0, 1.0, 1.0]);
    let mut s = build(
        vec![record(1, None, t, 0..0), record(2, None, ground, 0..0)],
        &[(1, &c), (2, r#""collider":{"half_extents":[50,0.5,50]}"#)],
    );
    let drawn = Drawn::of(&t, center, turn, half);
    for _ in 0..240 {
        s.world.step(DT);
    }
    // Where the drawn corners are now: carried by the body's motion, as the instances are.
    let motion = s.motion(0).unwrap();
    let lowest = (0..8)
        .map(|i| motion.point(drawn.corner(i)).y)
        .fold(f32::INFINITY, f32::min);
    assert!(
        lowest.abs() < 0.02,
        "the drawn shape's lowest corner rests at {lowest}, the ground is at 0"
    );
}
