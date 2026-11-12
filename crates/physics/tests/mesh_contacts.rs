//! Convex shapes against triangle meshes and heightfields: which normals a triangle may give at
//! creases, seams, borders and vertices, which points a merged manifold keeps, and how bodies
//! move over such geometry in a world.

use std::f32::consts::{FRAC_PI_2, TAU};
use std::sync::Arc;

use ganbate_physics::collide::{Manifold, collide};
use ganbate_physics::mesh::{Heightfield, TriangleMesh};
use ganbate_physics::{
    BodyDesc, BodyId, BodyKind, ColliderDesc, Iso, PhysicsWorld, Quat, Settings, Shape, Vec3,
};

const DT: f32 = 1.0 / 60.0;

// ---------------------------------------------------------------------------------------------
// Helpers.

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

fn contacts(floor: &Shape, shape: &Shape, pose: &Iso, spec: f32) -> Vec<Manifold> {
    let mut out = vec![];
    collide(floor, &Iso::IDENTITY, shape, pose, spec, &mut out);
    out
}

/// The smallest separation of all manifolds (infinity for none).
fn deepest(out: &[Manifold]) -> f32 {
    out.iter()
        .map(|m| m.min_separation())
        .fold(f32::INFINITY, f32::min)
}

/// How far the steepest normal leans from +Y, in degrees.
fn steepest(out: &[Manifold]) -> f32 {
    out.iter()
        .map(|m| m.normal.y.clamp(-1.0, 1.0).acos().to_degrees())
        .fold(0.0, f32::max)
}

fn sphere(radius: f32) -> Shape {
    Shape::Sphere { radius }
}
fn capsule(half_height: f32, radius: f32) -> Shape {
    Shape::Capsule {
        half_height,
        radius,
    }
}
fn cuboid(h: Vec3) -> Shape {
    Shape::Box { half_extents: h }
}

/// `n` × `n` samples `cell` apart with a straight ridge along Z at x = `ridge` · `cell`; each side
/// falls away at `deg` degrees (a crease of twice that).
fn ridge_field(deg: f32, ridge: u32, n: u32, cell: f32) -> Heightfield {
    let t = deg.to_radians().tan();
    let heights = (0..n * n)
        .map(|k| -((k % n) as f32 - ridge as f32).abs() * cell * t)
        .collect();
    Heightfield::new(n, n, heights, v(cell, 1.0, cell)).unwrap()
}

/// The triangles of a heightfield as a mesh (shared vertices welded by position).
fn mesh_of(h: &Heightfield) -> TriangleMesh {
    let count = 2 * (h.columns - 1) * (h.rows - 1);
    let mut vertices = vec![];
    let mut tris = vec![];
    for id in 0..count {
        let t = h.triangle(id);
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&t.vertices);
        tris.push([base, base + 1, base + 2]);
    }
    TriangleMesh::new(vertices, tris).unwrap()
}

/// A flat floor at y = 0 of `cells`² square cells of side `size` centred on the origin, each
/// split along its diagonal.
fn grid_mesh(cells: u32, size: f32) -> TriangleMesh {
    let n = cells + 1;
    let half = cells as f32 * size * 0.5;
    let mut vertices = vec![];
    for j in 0..n {
        for i in 0..n {
            vertices.push(v(i as f32 * size - half, 0.0, j as f32 * size - half));
        }
    }
    let mut tris = vec![];
    for j in 0..cells {
        for i in 0..cells {
            let (a, c) = (j * n + i, (j + 1) * n + i);
            tris.push([a, c, c + 1]);
            tris.push([a, c + 1, a + 1]);
        }
    }
    TriangleMesh::new(vertices, tris).unwrap()
}

/// A mesh of triangles each wound to face up (+Y).
fn up_mesh(vertices: Vec<Vec3>, tris: &[[u32; 3]]) -> TriangleMesh {
    let tris: Vec<[u32; 3]> = tris
        .iter()
        .map(|&[a, b, c]| {
            let n = (vertices[b as usize] - vertices[a as usize])
                .cross(vertices[c as usize] - vertices[a as usize]);
            if n.y < 0.0 { [a, c, b] } else { [a, b, c] }
        })
        .collect();
    TriangleMesh::new(vertices, tris).unwrap()
}

/// A flat floor for z in [0, 2], x in [-2, 2] whose straight border z = 0 has a vertex V at the
/// origin: right triangles (45° at V) or a fan (30° triangles next to the border at V).
fn border_floor(fan: bool) -> TriangleMesh {
    if fan {
        let t30 = 2.0 * 30f32.to_radians().tan();
        let vertices = vec![
            v(0.0, 0.0, 0.0),
            v(2.0, 0.0, 0.0),
            v(2.0, 0.0, t30),
            v(2.0, 0.0, 2.0),
            v(-2.0, 0.0, 2.0),
            v(-2.0, 0.0, t30),
            v(-2.0, 0.0, 0.0),
        ];
        up_mesh(
            vertices,
            &[[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 6]],
        )
    } else {
        let vertices = vec![
            v(0.0, 0.0, 0.0),
            v(2.0, 0.0, 0.0),
            v(2.0, 0.0, 2.0),
            v(0.0, 0.0, 2.0),
            v(-2.0, 0.0, 2.0),
            v(-2.0, 0.0, 0.0),
        ];
        up_mesh(vertices, &[[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]])
    }
}

/// A pyramid of `sides` triangles round an apex above the origin, base radius 2, its faces
/// rising at `slope` degrees.
fn pyramid(sides: u32, slope: f32) -> (TriangleMesh, f32) {
    let height = 2.0 * slope.to_radians().tan();
    let mut vertices = vec![v(0.0, height, 0.0)];
    for i in 0..sides {
        let a = TAU * i as f32 / sides as f32;
        vertices.push(v(2.0 * a.cos(), 0.0, 2.0 * a.sin()));
    }
    let tris: Vec<[u32; 3]> = (0..sides)
        .map(|i| [0, 1 + (i + 1) % sides, 1 + i])
        .collect();
    (up_mesh(vertices, &tris), height)
}

/// The closed surface of the cube `[-1, 1]³`, faces outward, two triangles per face.
fn cube_mesh() -> TriangleMesh {
    let vertices: Vec<Vec3> = (0..8)
        .map(|i| {
            v(
                if i & 1 != 0 { 1.0 } else { -1.0 },
                if i & 2 != 0 { 1.0 } else { -1.0 },
                if i & 4 != 0 { 1.0 } else { -1.0 },
            )
        })
        .collect();
    let quads = [
        [0, 4, 6, 2],
        [1, 3, 7, 5],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 2, 3, 1],
        [4, 5, 7, 6],
    ];
    let mut tris = vec![];
    for [a, b, c, d] in quads {
        // Wind each triangle to face away from the centre.
        for t in [[a, b, c], [a, c, d]] {
            let [p, q, r] = t.map(|i: u32| vertices[i as usize]);
            let outward = (q - p).cross(r - p).dot(p) > 0.0;
            tris.push(if outward { t } else { [t[0], t[2], t[1]] });
        }
    }
    TriangleMesh::new(vertices, tris).unwrap()
}

/// The distance from `p` to triangle `t` (brute force over the plane and the three edges).
fn point_triangle_distance(p: Vec3, t: [Vec3; 3]) -> f32 {
    let n = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
    let q = p - n * n.dot(p - t[0]);
    let inside = (0..3).all(|k| {
        let (a, b) = (t[k], t[(k + 1) % 3]);
        (b - a).cross(q - a).dot(n) >= 0.0
    });
    if inside {
        return (p - q).length();
    }
    (0..3)
        .map(|k| {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let ab = b - a;
            let s = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
            (p - (a + ab * s)).length()
        })
        .fold(f32::INFINITY, f32::min)
}

/// The distance from `p` to the nearest triangle of `mesh`.
fn mesh_distance(mesh: &TriangleMesh, p: Vec3) -> f32 {
    (0..mesh.triangles.len() as u32)
        .map(|i| point_triangle_distance(p, mesh.triangle(i).vertices))
        .fold(f32::INFINITY, f32::min)
}

/// A small deterministic generator.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
}

fn no_ccd() -> Settings {
    Settings {
        enable_continuous: false,
        ..Default::default()
    }
}

fn world(settings: Settings, floor: Shape, friction: f32) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(settings);
    let ground = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            ..Default::default()
        })
        .unwrap();
    let mut d = ColliderDesc::new(floor);
    d.friction = friction;
    w.add_collider(ground, &d).unwrap();
    w
}

fn add(w: &mut PhysicsWorld, shape: Shape, pose: Iso, velocity: Vec3, friction: f32) -> BodyId {
    let b = w
        .create_body(&BodyDesc {
            position: pose.position,
            rotation: pose.rotation,
            linear_velocity: velocity,
            can_sleep: false,
            ..Default::default()
        })
        .unwrap();
    let mut d = ColliderDesc::new(shape);
    d.friction = friction;
    w.add_collider(b, &d).unwrap();
    b
}

// ---------------------------------------------------------------------------------------------
// Ridges: an active edge's normals lie between its two faces' normals.

/// A frictionless box sliding at 2 m/s up to and over a ridge of `crease` degrees: its lowest
/// speed along X before it is well past the ridge, and the speed energy allows there.
fn box_over_ridge(floor: Shape, crease: f32) -> (f32, f32) {
    let half = crease * 0.5;
    let t = half.to_radians().tan();
    let h = v(0.3, 0.15, 0.25);
    let mut w = world(no_ccd(), floor, 0.0);
    let tilt = Quat::from_axis_angle(Vec3::Z, half.to_radians());
    let pose = Iso::new(v(3.5, -1.5 * t + h.y + 0.0005, 5.0), tilt);
    let b = add(&mut w, cuboid(h), pose, v(2.0, 0.0, 0.0), 0.0);
    let mut slowest = f32::INFINITY;
    for _ in 0..90 {
        w.step(DT);
        if w.body_pose(b).unwrap().position.x < 7.5 {
            slowest = slowest.min(w.linear_velocity(b).unwrap().x);
        }
    }
    (slowest, (4.0 - 2.0 * 9.81 * 1.5 * t).sqrt())
}

#[test]
fn a_gentle_convex_crease_is_no_wall_for_a_sliding_box() {
    // A 0.2° ridge is active (it is over 0.1°), but its edge's normals are within 0.2° of up.
    let field = ridge_field(0.1, 10, 21, 0.5);
    let t = 0.1f32.to_radians().tan();
    let h = v(0.3, 0.15, 0.25);
    let tilt = Quat::from_axis_angle(Vec3::Z, 0.1f32.to_radians());
    // Resting on the left slope, its leading face 5 mm short of the ridge.
    let pose = Iso::new(
        v(5.0 - h.x - 0.005, -(h.x + 0.005) * t + h.y - 0.0002, 5.0),
        tilt,
    );
    let mesh = mesh_of(&field);
    for floor in [
        Shape::Heightfield(Arc::new(field.clone())),
        Shape::Mesh(Arc::new(mesh.clone())),
    ] {
        let out = contacts(&floor, &cuboid(h), &pose, 0.02);
        assert!(!out.is_empty() && steepest(&out) < 0.25, "{out:?}");
        // Sliding over it loses no more speed than the climb takes.
        let (slowest, bound) = box_over_ridge(floor.clone(), 0.2);
        assert!(
            slowest > bound - 0.03,
            "{slowest} m/s, energy allows {bound}"
        );
    }
    for crease in [0.12, 0.4, 1.0] {
        let floor = Shape::Heightfield(Arc::new(ridge_field(crease * 0.5, 10, 21, 0.5)));
        let (slowest, bound) = box_over_ridge(floor, crease);
        assert!(
            slowest > bound - 0.05,
            "{crease}°: {slowest} m/s of {bound}"
        );
    }
    // Gently rolling terrain: 2 cm undulations, slopes under 1°.
    let n = 60u32;
    let heights = (0..n * n)
        .map(|k| {
            let (i, j) = ((k % n) as f32, (k / n) as f32);
            0.02 * (0.3 * i).sin() + 0.01 * (0.23 * j).cos()
        })
        .collect();
    let hills = Heightfield::new(n, n, heights, v(0.5, 1.0, 0.5)).unwrap();
    let mut w = world(no_ccd(), Shape::Heightfield(Arc::new(hills)), 0.0);
    let b = add(
        &mut w,
        cuboid(h),
        Iso::from_position(v(2.0, 0.2, 15.0)),
        v(2.5, 0.0, 0.0),
        0.0,
    );
    let mut slowest = f32::INFINITY;
    for step in 0..240 {
        w.step(DT);
        if step > 30 {
            slowest = slowest.min(w.linear_velocity(b).unwrap().x);
        }
    }
    // The heights vary by 6 cm: energy allows at least 2.25 m/s.
    assert!(slowest > 2.2, "caught on gentle terrain: {slowest} m/s");
}

/// A sphere, or a capsule lying with its core along the ridge, sliding or rolling at 3 m/s over a
/// ridge of `crease` degrees: its highest upward speed and how far it leaves the ground.
fn round_over_ridge(crease: f32, shape: &Shape, friction: f32) -> (f32, f32) {
    let half = crease * 0.5;
    let (t, c) = (half.to_radians().tan(), half.to_radians().cos());
    let (r, rotation) = match shape {
        Shape::Capsule { radius, .. } => (*radius, Quat::from_axis_angle(Vec3::X, FRAC_PI_2)),
        Shape::Sphere { radius } => (*radius, Quat::IDENTITY),
        _ => unreachable!(),
    };
    let field = ridge_field(half, 10, 21, 0.5);
    let mut w = world(
        Settings::default(),
        Shape::Heightfield(Arc::new(field)),
        friction,
    );
    let pose = Iso::new(v(3.5, -1.5 * t + r / c + 0.0005, 5.0), rotation);
    let b = add(&mut w, shape.clone(), pose, v(3.0, 0.0, 0.0), friction);
    if friction > 0.0 {
        w.set_angular_velocity(b, v(0.0, 0.0, -3.0 / r));
    }
    let (mut rise, mut hop) = (f32::NEG_INFINITY, 0.0f32);
    for _ in 0..90 {
        w.step(DT);
        let p = w.body_pose(b).unwrap().position;
        if p.x < 8.0 {
            rise = rise.max(w.linear_velocity(b).unwrap().y);
            hop = hop.max(p.y - (-(p.x - 5.0).abs() * t + r / c));
        }
    }
    (rise, hop)
}

#[test]
fn gentle_creases_do_not_launch_spheres_or_capsules() {
    let field = ridge_field(0.1, 10, 21, 0.5);
    let floor = Shape::Heightfield(Arc::new(field.clone()));
    let t = 0.1f32.to_radians().tan();
    // Resting on the left slope 5 cm short of the ridge: only the slope's normal.
    let ball = contacts(
        &floor,
        &sphere(0.25),
        &Iso::from_position(v(4.95, -0.05 * t + 0.25, 5.0)),
        0.02,
    );
    assert!(!ball.is_empty() && steepest(&ball) < 0.25, "{ball:?}");
    let lying = Quat::from_axis_angle(Vec3::X, FRAC_PI_2);
    let pill = contacts(
        &floor,
        &capsule(0.3, 0.15),
        &Iso::new(v(4.95, -0.05 * t + 0.15, 5.0), lying),
        0.02,
    );
    assert!(!pill.is_empty() && steepest(&pill) < 0.25, "{pill:?}");
    // Sliding over creases of 0.2°, 2° and 10°: no faster upward than the slope itself lifts.
    for crease in [0.2f32, 2.0, 10.0] {
        let slope = 3.0 * (crease * 0.5).to_radians().sin();
        for shape in [sphere(0.25), capsule(0.3, 0.15)] {
            let (rise, _) = round_over_ridge(crease, &shape, 0.0);
            assert!(rise < slope + 0.03, "{crease}° {shape:?}: {rise} m/s");
        }
    }
    // Rolling over a 0.2° crease: they stay on the ground.
    for shape in [sphere(0.25), capsule(0.3, 0.15)] {
        let (_, hop) = round_over_ridge(0.2, &shape, 0.6);
        assert!(hop < 0.001, "{shape:?} hops {hop} m");
    }
}

#[test]
fn apexes_and_peaks_keep_their_upward_normal_and_exact_distance() {
    // The cone of normals at a convex vertex is wider than any one edge's crease.
    for (sides, slope) in [(4, 30.0), (6, 30.0), (12, 30.0), (12, 10.0), (24, 5.0)] {
        let (mesh, top) = pyramid(sides, slope);
        let floor = Shape::Mesh(Arc::new(mesh));
        for gap in [-0.002f32, 0.0, 0.005] {
            let pose = Iso::from_position(v(0.0, top + 0.25 + gap, 0.0));
            let out = contacts(&floor, &sphere(0.25), &pose, 0.02);
            assert!(
                (deepest(&out) - gap).abs() < 1e-4,
                "{sides} {slope}: {out:?}"
            );
            assert!(
                out.iter().all(|m| m.normal.y > 0.9999),
                "{sides} {slope}: {out:?}"
            );
        }
    }
    for raise in [0.05f32, 0.15, 0.3] {
        let n = 9u32;
        let heights = (0..n * n)
            .map(|k| if k == 4 * n + 4 { raise } else { 0.0 })
            .collect();
        let field = Heightfield::new(n, n, heights, v(0.5, 1.0, 0.5)).unwrap();
        let floor = Shape::Heightfield(Arc::new(field));
        let pose = Iso::from_position(v(2.0, raise + 0.25 - 0.001, 2.0));
        let out = contacts(&floor, &sphere(0.25), &pose, 0.02);
        assert!((deepest(&out) + 0.001).abs() < 1e-4, "{raise}: {out:?}");
        assert!(out.iter().all(|m| m.normal.y > 0.9999), "{raise}: {out:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Ridge vertices: beside a triangle, its plane is no measure of depth.

#[test]
fn a_sphere_beside_a_ridge_vertex_gets_no_phantom_penetration() {
    // A 15° ridge along Z at x = 5; z = 5 is a grid vertex, where some triangles touch the
    // ridge only at that vertex.
    let field = ridge_field(15.0, 10, 21, 0.5);
    let a = 15f32.to_radians();
    let (left, up) = (v(-a.sin(), a.cos(), 0.0), Vec3::Y);
    let vertex = v(5.0, 0.0, 5.0);
    let r = 0.25;
    for floor in [
        Shape::Heightfield(Arc::new(field.clone())),
        Shape::Mesh(Arc::new(mesh_of(&field))),
    ] {
        // (centre, true separation): resting on the slope 5 cm and 2 cm short of the ridge;
        // touching the ridge vertex 0.5 mm deep along the slope's normal and from straight
        // above.
        for (centre, truth) in [
            (v(4.95, -0.05 * a.tan(), 5.0) + left * r, 0.0),
            (v(4.98, -0.02 * a.tan(), 5.0) + left * r, 0.0),
            (vertex + left * (r - 0.0005), -0.0005),
            (vertex + up * (r - 0.0005), -0.0005),
        ] {
            let out = contacts(&floor, &sphere(r), &Iso::from_position(centre), 0.02);
            assert!(
                (deepest(&out) - truth).abs() < 1e-4,
                "{centre:?}: {truth} vs {out:?}"
            );
        }
        // A capsule lying along the ridge's direction, resting on the slope 5 cm short of it.
        let lying = Quat::from_axis_angle(Vec3::X, FRAC_PI_2);
        let centre = v(4.95, -0.05 * a.tan(), 5.0) + left * 0.15;
        let out = contacts(&floor, &capsule(0.1, 0.15), &Iso::new(centre, lying), 0.02);
        assert!(deepest(&out).abs() < 1e-4, "{out:?}");
    }
    // In a world, a ball let go on the slope by the vertex, or balanced on it, is not thrown.
    let floor = Shape::Heightfield(Arc::new(field));
    // Where a ball's centre rests `dx` from the ridge line: on the ridge line, or on a slope.
    let rest = |dx: f32| {
        if dx.abs() < r * a.sin() {
            (r * r - dx * dx).sqrt()
        } else {
            -dx.abs() * a.tan() + r / a.cos()
        }
    };
    for centre in [
        v(4.96, -0.04 * a.tan(), 5.0) + left * r,
        vertex + up * (r + 0.0005),
    ] {
        let mut w = world(no_ccd(), floor.clone(), 0.8);
        let b = add(
            &mut w,
            sphere(r),
            Iso::from_position(centre),
            Vec3::ZERO,
            0.8,
        );
        let mut lift = 0.0f32;
        for _ in 0..30 {
            w.step(DT);
            let p = w.body_pose(b).unwrap().position;
            lift = lift.max(p.y - rest(p.x - 5.0));
        }
        assert!(lift < 0.002, "{centre:?}: thrown up {lift} m");
    }
}

// ---------------------------------------------------------------------------------------------
// Straight borders: a vertex between a border edge and a flat seam is no corner.

#[test]
fn a_vertex_on_a_straight_border_is_no_wall() {
    let h = v(0.3, 0.2, 0.3);
    for fan in [false, true] {
        let floor = Shape::Mesh(Arc::new(border_floor(fan)));
        // A box 3 mm inside the border, its leading face 1 cm short of the border vertex.
        let pose = Iso::from_position(v(-0.31, h.y - 0.001, 0.003 + h.z));
        let out = contacts(&floor, &cuboid(h), &pose, 0.02);
        assert!(
            !out.is_empty() && steepest(&out) < 0.01,
            "fan {fan}: {out:?}"
        );
        // Overhanging the border by 5 cm.
        let pose = Iso::from_position(v(-0.31, h.y - 0.001, h.z - 0.05));
        let out = contacts(&floor, &cuboid(h), &pose, 0.02);
        assert!(
            !out.is_empty() && steepest(&out) < 0.01,
            "fan {fan}: {out:?}"
        );
        // A ball resting 3 cm inside the border, 5 cm short of the vertex.
        let pose = Iso::from_position(v(-0.05, 0.25, 0.03));
        let out = contacts(&floor, &sphere(0.25), &pose, 0.02);
        assert!(
            !out.is_empty() && steepest(&out) < 0.01,
            "fan {fan}: {out:?}"
        );
    }
    // A frictionless box sliding along the border keeps its speed and does not turn.
    for (fan, inset, speed) in [
        (false, 0.001f32, 0.3f32),
        (true, 0.006, 0.6),
        (false, 0.003, 1.0),
        (false, -0.05, 1.0),
    ] {
        let floor = Shape::Mesh(Arc::new(border_floor(fan)));
        let mut w = world(Settings::default(), floor, 0.0);
        let pose = Iso::from_position(v(-0.6, h.y, inset + h.z));
        let b = add(&mut w, cuboid(h), pose, v(speed, 0.0, 0.0), 0.0);
        let mut slowest = f32::INFINITY;
        for _ in 0..60 {
            w.step(DT);
            slowest = slowest.min(w.linear_velocity(b).unwrap().x);
        }
        let spin = w.angular_velocity(b).unwrap().length();
        assert!(
            slowest > speed - 1e-3,
            "fan {fan} inset {inset}: {slowest} m/s"
        );
        assert!(
            spin < 1e-3,
            "fan {fan} inset {inset}: turning at {spin} rad/s"
        );
    }
}

#[test]
fn spheres_by_a_cube_corner_get_their_exact_distance() {
    // Every convex edge of a closed mesh is active and its face diagonals are seams: the
    // corner's contact must come from some triangle, and none may be deeper than the truth.
    let cube = cube_mesh();
    let floor = Shape::Mesh(Arc::new(cube.clone()));
    let mut rng = Rng(7);
    let mut touching = 0;
    for _ in 0..3000 {
        let r = rng.range(0.05, 0.4);
        let c = v(
            rng.range(0.6, 1.4),
            rng.range(0.6, 1.4),
            rng.range(0.6, 1.4),
        );
        if c.x < 1.0 && c.y < 1.0 && c.z < 1.0 {
            continue;
        }
        let truth = mesh_distance(&cube, c) - r;
        let out = contacts(&floor, &sphere(r), &Iso::from_position(c), 0.02);
        if truth <= 0.019 {
            touching += 1;
            assert!(
                (deepest(&out) - truth).abs() < 1e-4,
                "{c:?} r {r}: {truth} vs {out:?}"
            );
        } else {
            assert!(
                deepest(&out) > truth - 1e-4,
                "{c:?} r {r}: {truth} vs {out:?}"
            );
        }
    }
    assert!(touching > 500, "{touching}");
}

// ---------------------------------------------------------------------------------------------
// Merged manifolds keep the patch's corners.

/// The corners of the bottom face of the box `h` at `pose`.
fn bottom_corners(h: Vec3, pose: &Iso) -> [Vec3; 4] {
    [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        .map(|(sx, sz)| pose.point(v(sx * h.x, -h.y, sz * h.z)))
}

/// Whether one of `m`'s points lies over `c` (seen from above).
fn has_point_over(m: &Manifold, c: Vec3) -> bool {
    m.points().iter().any(|p| {
        let d = p.point - c;
        v(d.x, 0.0, d.z).length() < 1e-4
    })
}

/// Whether the triangle of `mesh` under corner `c` reports `c` in its own contact with the box
/// (alone, so nothing merges): which points one triangle's clipped polygon keeps is not the
/// merge's doing.
fn own_triangle_reports(mesh: &TriangleMesh, h: Vec3, pose: &Iso, c: Vec3) -> bool {
    (0..mesh.triangles.len() as u32).any(|i| {
        let t = mesh.triangle(i).vertices;
        let under = (0..3).all(|k| {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            (b - a).cross(c - a).y >= 0.0
        });
        if !under {
            return false;
        }
        let alone = Shape::Mesh(Arc::new(
            TriangleMesh::new(t.to_vec(), vec![[0, 1, 2]]).unwrap(),
        ));
        let out = contacts(&alone, &cuboid(h), pose, 0.02);
        out.len() == 1 && has_point_over(&out[0], c)
    })
}

#[test]
fn a_box_lying_flat_on_a_mesh_keeps_its_four_corners() {
    let h = v(0.4, 0.2, 0.3);
    let floor = Shape::Mesh(Arc::new(grid_mesh(8, 0.5)));
    // Unturned, its corners at x in {-0.27, 0.53}, z in {-0.23, 0.37}: mesh vertices and seam
    // crossings lie under it too, all as deep.
    let pose = Iso::from_position(v(0.13, h.y - 1e-4, 0.07));
    let out = contacts(&floor, &cuboid(h), &pose, 0.02);
    assert_eq!(out.len(), 1, "{out:?}");
    for c in bottom_corners(h, &pose) {
        assert!(has_point_over(&out[0], c), "{c:?} missing: {out:?}");
    }
    // Anywhere on grids of three sizes, turned about Y, tilted by rounding-sized amounts:
    // merging the triangles' manifolds keeps every corner a triangle reports.
    let mut rng = Rng(11);
    for (cells, size) in [(8u32, 0.5f32), (16, 0.25), (32, 0.125)] {
        let mesh = grid_mesh(cells, size);
        let floor = Shape::Mesh(Arc::new(mesh.clone()));
        let (mut lost, mut kept) = (vec![], 0);
        for i in 0..200 {
            let yaw = rng.range(0.0, TAU);
            let tilt = rng.range(-6e-6, 6e-6);
            let rotation =
                Quat::from_axis_angle(Vec3::Y, yaw) * Quat::from_axis_angle(Vec3::X, tilt);
            let at = v(rng.range(-0.8, 0.8), h.y - 1e-4, rng.range(-0.8, 0.8));
            let pose = Iso::new(at, rotation);
            let out = contacts(&floor, &cuboid(h), &pose, 0.02);
            assert_eq!(out.len(), 1, "{cells}x{size} pose {i}: {out:?}");
            for c in bottom_corners(h, &pose) {
                if has_point_over(&out[0], c) {
                    kept += 1;
                } else if own_triangle_reports(&mesh, h, &pose, c) {
                    lost.push(i);
                }
            }
        }
        assert!(
            lost.is_empty(),
            "{cells}x{size}: poses {lost:?} lose a corner"
        );
        assert!(kept > 760, "{cells}x{size}: {kept} of 800 corners kept");
    }
    // Resting in a world, its contact points stay the same features step after step.
    let mut w = world(Settings::default(), floor, 0.6);
    let b = add(
        &mut w,
        cuboid(h),
        Iso::from_position(v(0.13, h.y + 0.05, 0.07)),
        Vec3::ZERO,
        0.6,
    );
    let (mut previous, mut changes, mut fastest) = (vec![], 0, 0.0f32);
    for step in 0..240 {
        w.step(DT);
        let mut ids: Vec<u32> = w
            .manifolds()
            .flat_map(|(_, _, m)| m.points().iter().map(|p| p.id).collect::<Vec<_>>())
            .collect();
        ids.sort_unstable();
        if step > 60 {
            changes += usize::from(ids != previous);
            fastest = fastest.max(w.linear_velocity(b).unwrap().length());
        }
        previous = ids;
    }
    assert_eq!(
        changes, 0,
        "the resting box's points changed {changes} times"
    );
    assert!(fastest < 1e-5, "the resting box moves at {fastest} m/s");
}

#[test]
fn a_resting_hexagonal_prism_keeps_four_contacts_across_mesh_seams() {
    let vertices: Vec<_> = (0..6)
        .flat_map(|i| {
            let (sin, cos) = (TAU * i as f32 / 6.0).sin_cos();
            [-0.5, 0.5].map(|y| v(0.5 * cos, y, 0.5 * sin))
        })
        .collect();
    let shape = Shape::Hull(Arc::new(
        ganbate_physics::hull::ConvexHull::from_points(&vertices).unwrap(),
    ));
    for (cell, x, z, yaw) in [
        (0.5, 0.13, 0.07, 0.0),
        (0.25, 0.0, 0.0, 0.0),
        (0.125, -0.07, 0.21, 0.35),
    ] {
        let floor = Shape::Mesh(Arc::new(grid_mesh(16, cell)));
        let mut w = world(Settings::default(), floor, 0.6);
        let pose = Iso::new(v(x, 0.505, z), Quat::from_axis_angle(Vec3::Y, yaw));
        let body = add(&mut w, shape.clone(), pose, Vec3::ZERO, 0.6);
        let mut previous = Vec::new();
        let mut changes = 0;
        let mut fastest = 0.0f32;
        for step in 0..360 {
            w.step(DT);
            let mut ids = Vec::new();
            for (_, _, m) in w.manifolds() {
                ids.extend(m.points().iter().map(|p| p.id));
            }
            ids.sort_unstable();
            if step >= 60 {
                assert_eq!(ids.len(), 4);
                changes += usize::from(ids != previous);
                fastest = fastest.max(w.linear_velocity(body).unwrap().length());
            }
            previous = ids;
        }
        assert_eq!(
            changes, 0,
            "mesh {cell} m, yaw {yaw}: resting contacts changed"
        );
        assert!(
            fastest < 1e-4,
            "mesh {cell} m, yaw {yaw}: speed {fastest} m/s"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// One-sided triangles and bodies that went through them.

/// Where a box `h` falling flat ends on `floor` (over (x, z)), let go `height` up at `speed`.
fn landing(floor: &Shape, h: Vec3, at: Vec3, speed: f32) -> f32 {
    let mut w = world(Settings::default(), floor.clone(), 0.6);
    let b = add(
        &mut w,
        cuboid(h),
        Iso::from_position(at),
        v(0.0, -speed, 0.0),
        0.6,
    );
    let fall = ((speed * speed + 2.0 * 9.81 * at.y).sqrt() - speed) / 9.81;
    for _ in 0..(fall / DT) as usize + 120 {
        w.step(DT);
    }
    w.body_pose(b).unwrap().position.y
}

#[test]
fn a_thin_plate_landing_hard_stays_on_a_mesh() {
    // Landing flat at 28 m/s or so, the solve can leave a plate tilted with its centre a few
    // millimetres under the floor: every triangle then has it behind, yet a box floor would
    // still push it back up.
    let field = Heightfield::new(9, 9, vec![0.0; 81], v(1.0, 1.0, 1.0)).unwrap();
    let floors = [
        (Shape::Mesh(Arc::new(grid_mesh(8, 1.0))), v(0.13, 0.0, 0.21)),
        (Shape::Heightfield(Arc::new(field)), v(4.13, 0.0, 4.21)),
    ];
    for (floor, at) in &floors {
        for (h, height, speed) in [
            (v(0.15, 0.005, 0.15), 40.0f32, 0.0f32),
            (v(0.15, 0.01, 0.15), 80.0, 0.0),
            (v(0.15, 0.005, 0.15), 0.6, 20.0),
        ] {
            let y = landing(floor, h, *at + v(0.0, height, 0.0), speed);
            assert!(
                (y - h.y).abs() < 0.002,
                "{h:?} from {height} m at {speed} m/s ends at y {y}"
            );
        }
    }
}

/// A closed box mesh from (-1, -t, -1) to (1, 0, 1), faces outward: a shelf or a pane.
fn slab(t: f32) -> TriangleMesh {
    let vertices: Vec<Vec3> = (0..8)
        .map(|i| {
            v(
                if i & 1 != 0 { 1.0 } else { -1.0 },
                if i & 2 != 0 { 0.0 } else { -t },
                if i & 4 != 0 { 1.0 } else { -1.0 },
            )
        })
        .collect();
    let quads = [
        [0, 4, 6, 2],
        [1, 3, 7, 5],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 2, 3, 1],
        [4, 5, 7, 6],
    ];
    let centre = v(0.0, -0.5 * t, 0.0);
    let mut tris = vec![];
    for [a, b, c, d] in quads {
        for t in [[a, b, c], [a, c, d]] {
            let [p, q, r] = t.map(|i: u32| vertices[i as usize]);
            let outward = (q - p).cross(r - p).dot(p + q + r - centre * 3.0) > 0.0;
            tris.push(if outward { t } else { [t[0], t[2], t[1]] });
        }
    }
    TriangleMesh::new(vertices, tris).unwrap()
}

#[test]
fn a_thin_closed_mesh_holds_what_rests_on_it() {
    // A body resting on a 1 cm shelf has its centre less than the speculative distance behind
    // the underside: the underside must not pull it down while the top face holds it.
    let shelf = Shape::Mesh(Arc::new(slab(0.01)));
    let lying = Quat::from_axis_angle(Vec3::Z, FRAC_PI_2);
    for (shape, rotation, rest) in [
        (sphere(0.005), Quat::IDENTITY, 0.005),
        (capsule(0.15, 0.005), lying, 0.005),
        (cuboid(v(0.15, 0.005, 0.15)), Quat::IDENTITY, 0.005),
    ] {
        let pose = Iso::new(v(0.13, rest, 0.21), rotation);
        let out = contacts(&shelf, &shape, &pose, 0.02);
        assert!(
            !out.is_empty() && out.iter().all(|m| m.normal.y > 0.9999),
            "{shape:?}: {out:?}"
        );
        let mut w = world(Settings::default(), shelf.clone(), 0.6);
        let at = Iso::new(v(0.13, rest + 0.001, 0.21), rotation);
        let b = add(&mut w, shape.clone(), at, Vec3::ZERO, 0.6);
        for _ in 0..120 {
            w.step(DT);
        }
        let y = w.body_pose(b).unwrap().position.y;
        assert!((y - rest).abs() < 0.002, "{shape:?} ends at y {y}");
    }
}
