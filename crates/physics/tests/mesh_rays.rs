//! Rays never slip between the triangles of a mesh or heightfield that share an edge or a vertex.
//!
//! A ray aimed at a seam of a surface (a cell's diagonal, the side two cells share, a grid point
//! where six triangles meet) must hit one of the triangles there. Rays from eye height to tile
//! centres, side midpoints and grid points are the ones a grid-snapped level casts for picking,
//! placement and line of sight, and the Möller–Trumbore test the crate used first let about one
//! in fifteen of them through a level floor: each of the two triangles at a seam computed the
//! seam's barycentric coordinate from its own corners, and both could round it to a tiny
//! negative value and turn the ray away. The watertight test (Woop, Benthin and Wald) gives the
//! two triangles exactly opposite values for the edge they share, so one of them always takes
//! the ray.

use std::sync::Arc;

use ganbate_physics::mesh::{Heightfield, MeshHit, TriangleMesh, ray_triangle};
use ganbate_physics::{
    BodyDesc, BodyKind, ColliderDesc, PhysicsWorld, Quat, QueryFilter, Settings, Shape, Vec3,
};

/// SplitMix64, for repeatable test data.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 * (1.0 / 16_777_216.0)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next() % u64::from(n)) as u32
    }
    fn direction(&mut self) -> Vec3 {
        loop {
            let v = Vec3::new(
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            );
            if (1e-4..=1.0).contains(&v.length_squared()) {
                return v.normalize_or_zero();
            }
        }
    }
}

/// A `columns × rows` grid of samples (sample `(i, j)` at `place(i, j)`) split into triangles
/// the way a heightfield splits its cells, so triangle `2 · cell + half` is the heightfield's.
fn grid_mesh(columns: u32, rows: u32, place: impl Fn(u32, u32) -> Vec3) -> TriangleMesh {
    let vertices = (0..columns * rows)
        .map(|k| place(k % columns, k / columns))
        .collect();
    let mut triangles = vec![];
    for j in 0..rows - 1 {
        for i in 0..columns - 1 {
            let s = |i: u32, j: u32| j * columns + i;
            let (p00, p10, p01, p11) = (s(i, j), s(i + 1, j), s(i, j + 1), s(i + 1, j + 1));
            triangles.push([p00, p01, p11]);
            triangles.push([p00, p11, p10]);
        }
    }
    TriangleMesh::new(vertices, triangles).unwrap()
}

/// Where a grid's sample `(i, j)` goes.
type Place = dyn Fn(u32, u32) -> Vec3;

/// A seam of a grid: two samples `(i, j)` and the fraction of the way from the first to the
/// second (both the same sample for a grid point).
type Seam = ([u32; 2], [u32; 2], f32);

/// Where the triangles of a `columns × rows` grid meet: the middle (the tile's centre) and quarter
/// points of every cell's diagonal, the middle of every side two cells share, and every grid
/// point inside the border (where six triangles meet).
fn seams(columns: u32, rows: u32) -> Vec<Seam> {
    let mut out = vec![];
    for j in 0..rows - 1 {
        for i in 0..columns - 1 {
            for s in [0.25, 0.5, 0.75] {
                out.push(([i, j], [i + 1, j + 1], s));
            }
            if j > 0 {
                out.push(([i, j], [i + 1, j], 0.5));
            }
            if i > 0 {
                out.push(([i, j], [i, j + 1], 0.5));
            }
            if i > 0 && j > 0 {
                out.push(([i, j], [i, j], 0.0));
            }
        }
    }
    out
}

/// The point of `seam` on a grid whose samples are at `place(i, j)`.
fn seam_point(seam: &Seam, place: &impl Fn(u32, u32) -> Vec3) -> Vec3 {
    let (a, b, s) = seam;
    let (pa, pb) = (place(a[0], a[1]), place(b[0], b[1]));
    pa + (pb - pa) * *s
}

/// Fails with the number of rays that went wrong and the first few.
fn assert_none_failed(what: &str, rays: usize, failures: &[String]) {
    assert!(
        failures.is_empty(),
        "{what}: {} of {rays} rays failed; first: {:#?}",
        failures.len(),
        &failures[..failures.len().min(4)]
    );
}

/// The ray the review found: from eye height to a point of a level floor's cell diagonal, where
/// both triangles of the cell computed the diagonal's barycentric coordinate as −7.2e-7 and
/// rejected it, so it went through the floor of a mesh and of a heightfield alike.
#[test]
fn the_review_ray_through_a_cell_diagonal_hits_the_floor() {
    let (p00, p10, p01, p11) = (
        Vec3::new(29.0, 0.0, 16.0),
        Vec3::new(30.0, 0.0, 16.0),
        Vec3::new(29.0, 0.0, 17.0),
        Vec3::new(30.0, 0.0, 17.0),
    );
    let eye = Vec3::new(24.362_268, 1.7, 25.048_065);
    let dir = (Vec3::new(29.25, 0.0, 16.25) - eye).normalize_or_zero();
    // In f64 the ray meets y = 0 at (29.250000117, 0, 16.249999916), t = 10.2071483: inside the
    // cell, 2e-7 off the diagonal.
    let want = 10.207_148;
    let halves = [[p00, p01, p11], [p00, p11, p10]];
    let ts = halves.map(|v| ray_triangle(eye, dir, &v, f32::INFINITY, false));
    assert!(
        ts.iter().flatten().any(|t| (t - want).abs() < 1e-4),
        "neither half of the cell takes the ray: {ts:?}"
    );
    let mesh = TriangleMesh::new(vec![p00, p10, p01, p11], vec![[0, 2, 3], [0, 3, 1]]).unwrap();
    let hit = mesh.raycast(eye, dir, f32::INFINITY, false);
    assert!(
        hit.is_some_and(|h| (h.t - want).abs() < 1e-4),
        "the ray goes through the mesh floor: {hit:?}"
    );
    // The same cell as a one-cell heightfield, the ray moved with it (exactly).
    let field = Heightfield::new(2, 2, vec![0.0; 4], Vec3::ONE).unwrap();
    let local = eye - p00;
    assert_eq!(local + p00, eye);
    let hit = field.raycast(local, dir, f32::INFINITY);
    assert!(
        hit.is_some_and(|h| (h.t - want).abs() < 1e-4),
        "the ray goes through the heightfield floor: {hit:?}"
    );
}

/// The triangle test alone, with no hierarchy or grid walk in the way: a ray through any point
/// of an edge two triangles share, or through a grid point six share, hits at least one of them,
/// from the front, and from the back when back faces count. On a level floor at the origin, the
/// same floor tilted and far away, and a floor of 0.37 m cells turned about +Y.
#[test]
fn a_ray_through_a_shared_edge_or_vertex_hits_one_of_its_triangles() {
    let n = 9;
    let tilt = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.3).normalize_or_zero(), 0.35);
    let turn = Quat::from_axis_angle(Vec3::Y, 0.7);
    let floors: [(&str, Box<Place>); 3] = [
        ("level", Box::new(|i, j| Vec3::new(i as f32, 0.0, j as f32))),
        (
            "tilted",
            Box::new(move |i, j| {
                tilt.rotate(Vec3::new(i as f32, 0.0, j as f32)) + Vec3::new(300.0, 20.0, -150.0)
            }),
        ),
        (
            "turned",
            Box::new(move |i, j| turn.rotate(Vec3::new(i as f32, 0.0, j as f32) * 0.37)),
        ),
    ];
    let mut rng = Rng(0x5EA3);
    for (name, place) in &floors {
        let mesh = grid_mesh(n, n, place);
        let triangles: Vec<[Vec3; 3]> = (0..mesh.triangles.len() as u32)
            .map(|id| mesh.triangle(id).vertices)
            .collect();
        let up = mesh.triangle(0).normal();
        let size = (place(1, 0) - place(0, 0)).length();
        let (mut rays, mut failures) = (0, vec![]);
        for (a, b, s) in seams(n, n) {
            for _ in 0..12 {
                // Any point of the edge, not only the ones `seams` names.
                let s = if a == b { s } else { rng.range(0.0, 1.0) };
                let target = seam_point(&(a, b, s), place);
                // From the front or the back, at any slant down to about 3 degrees.
                let offset = Vec3::new(rng.range(-8.0, 8.0), 0.0, rng.range(-8.0, 8.0)) * size;
                let height = rng.range(0.5, 4.0) * size;
                for (side, both_sides) in [(1.0, false), (-1.0, true)] {
                    let eye = target + (offset - up * offset.dot(up)) + up * (height * side);
                    let dir = (target - eye).normalize_or_zero();
                    rays += 1;
                    if !triangles
                        .iter()
                        .any(|v| ray_triangle(eye, dir, v, f32::INFINITY, both_sides).is_some())
                    {
                        failures.push(format!("{eye:?} → {target:?} (both sides: {both_sides})"));
                    }
                }
            }
        }
        assert_none_failed(&format!("{name} floor"), rays, &failures);
    }
}

/// Whether `hit` is the level floor at y = 0, `distance` along the ray (where it aims).
fn hits_level_floor(hit: Option<MeshHit>, distance: f32) -> bool {
    hit.is_some_and(|h| {
        (h.t - distance).abs() <= 1e-4 * distance && (h.normal - Vec3::Y).length() < 1e-6
    })
}

/// Rays from eye height (and from farther off, at a shallow slant) to every seam of a level floor
/// of 16 × 16 one-metre cells, as a mesh and as a heightfield: every one hits the floor where it
/// aims.
#[test]
fn eye_rays_to_every_seam_of_a_level_floor_hit_it() {
    let n = 17;
    let mesh = grid_mesh(n, n, |i, j| Vec3::new(i as f32, 0.0, j as f32));
    let field = Heightfield::new(n, n, vec![0.0; (n * n) as usize], Vec3::ONE).unwrap();
    let place = |i: u32, j: u32| field.sample(i, j);
    let mut rng = Rng(0xF2);
    let (mut rays, mut failures) = (0, vec![]);
    for seam in seams(n, n) {
        let target = seam_point(&seam, &place);
        for k in 0..4 {
            let eye = if k < 3 {
                Vec3::new(rng.range(-2.0, 18.0), 1.7, rng.range(-2.0, 18.0))
            } else {
                Vec3::new(
                    rng.range(-40.0, 56.0),
                    rng.range(0.5, 6.0),
                    rng.range(-40.0, 56.0),
                )
            };
            let (dir, distance) = (target - eye).normalize_and_length();
            let from_mesh = mesh.raycast(eye, dir, f32::INFINITY, false);
            let from_field = field.raycast(eye, dir, f32::INFINITY);
            for (what, hit) in [("mesh", from_mesh), ("heightfield", from_field)] {
                rays += 1;
                if !hits_level_floor(hit, distance) {
                    failures.push(format!("{what}: {eye:?} → {target:?}: {hit:?}"));
                }
            }
        }
    }
    assert_none_failed("level floor", rays, &failures);
}

/// Whether `hit` is no farther along `dir` than `distance`, give or take the rounding of
/// coordinates as large as `reach` over the slant at which the ray meets the triangle.
fn no_farther(hit: Option<MeshHit>, dir: Vec3, distance: f32, reach: f32) -> bool {
    hit.is_some_and(|h| {
        let slant = h.normal.dot(dir).abs().max(1e-3);
        h.t <= distance + 8.0 * f32::EPSILON * reach / slant
    })
}

/// The triangles of a grid mesh `columns` samples wide (or of the heightfield it copies, whose
/// ids are the same) that meet at `seam`: the two along an edge, the six around a grid point.
fn seam_triangles(mesh: &TriangleMesh, columns: u32, seam: &Seam) -> Vec<u32> {
    let (a, b, _) = seam;
    let (ia, ib) = (a[1] * columns + a[0], b[1] * columns + b[0]);
    (0..mesh.triangles.len() as u32)
        .filter(|&t| mesh.triangles[t as usize].contains(&ia))
        .filter(|&t| mesh.triangles[t as usize].contains(&ib))
        .collect()
}

/// Rays from above to every seam of hilly terrain with uneven cells (0.37 × 0.61 m), as a
/// heightfield and as a mesh of the same triangles 2 km from its origin: every one that crosses
/// the surface there hits the terrain, no farther than the seam (a hill may come first).
///
/// A ray crosses the surface at a seam when it runs into the front of every triangle that meets
/// there. One aimed at a ridge from the side may instead touch it, and pass a rounding above it
/// or below it; those are left out.
#[test]
fn rays_to_every_seam_of_hilly_terrain_hit_it_no_farther_than_the_seam() {
    let (columns, rows) = (14, 11);
    let mut rng = Rng(0x7177);
    let heights = (0..columns * rows).map(|_| rng.range(-2.0, 2.0)).collect();
    let field = Heightfield::new(columns, rows, heights, Vec3::new(0.37, 1.0, 0.61)).unwrap();
    let away = Vec3::new(1500.0, -40.0, -2000.0);
    let mesh = grid_mesh(columns, rows, |i, j| field.sample(i, j) + away);
    let field_place = |i: u32, j: u32| field.sample(i, j);
    let mesh_place = |i: u32, j: u32| mesh.vertices[(j * columns + i) as usize];
    let crosses = |normals: &[Vec3], dir: Vec3| normals.iter().all(|n| n.dot(dir) < -0.02);
    let size = field.local_aabb().max;
    let (mut rays, mut touching, mut failures) = (0, 0, vec![]);
    for seam in seams(columns, rows) {
        let ids = seam_triangles(&mesh, columns, &seam);
        assert!(ids.len() == 2 || ids.len() == 6, "{seam:?}: {ids:?}");
        let field_normals: Vec<Vec3> = ids.iter().map(|&t| field.triangle(t).normal()).collect();
        let mesh_normals: Vec<Vec3> = ids.iter().map(|&t| mesh.triangle(t).normal()).collect();
        let target = seam_point(&seam, &field_place);
        for _ in 0..6 {
            // Above every hill.
            let eye = Vec3::new(
                rng.range(0.0, size.x),
                rng.range(2.5, 8.0),
                rng.range(0.0, size.z),
            );
            let (dir, distance) = (target - eye).normalize_and_length();
            if crosses(&field_normals, dir) {
                rays += 1;
                let hit = field.raycast(eye, dir, f32::INFINITY);
                if !no_farther(hit, dir, distance, 8.0) {
                    failures.push(format!("heightfield: {eye:?} → {target:?}: {hit:?}"));
                }
            } else {
                touching += 1;
            }
            // The mesh's copy of the ray, rounded where its coordinates are.
            let (eye, target) = (eye + away, seam_point(&seam, &mesh_place));
            let (dir, distance) = (target - eye).normalize_and_length();
            if crosses(&mesh_normals, dir) {
                rays += 1;
                let hit = mesh.raycast(eye, dir, f32::INFINITY, false);
                if !no_farther(hit, dir, distance, 2000.0) {
                    failures.push(format!("mesh: {eye:?} → {target:?}: {hit:?}"));
                }
            } else {
                touching += 1;
            }
        }
    }
    assert!(
        rays > 4000,
        "{rays} rays cross the surface, {touching} may only touch it"
    );
    assert_none_failed("hilly terrain", rays, &failures);
}

/// Through the world, as games cast them: a 32 × 32 m floor of one-metre cells as a mesh collider
/// and as a heightfield collider, at the origin and 1.4 km away. Every ray from eye height to a
/// tile centre, a side midpoint or a grid point hits the floor where it aims.
#[test]
fn world_raycasts_from_eye_height_never_fall_through_a_floor() {
    let n = 33;
    let filter = QueryFilter::default();
    for position in [Vec3::ZERO, Vec3::new(-640.0, 3.0, 1280.0)] {
        for heightfield in [false, true] {
            let shape = if heightfield {
                let field = Heightfield::new(n, n, vec![0.0; (n * n) as usize], Vec3::ONE);
                Shape::Heightfield(Arc::new(field.unwrap()))
            } else {
                let mesh = grid_mesh(n, n, |i, j| Vec3::new(i as f32, 0.0, j as f32));
                Shape::Mesh(Arc::new(mesh))
            };
            let mut world = PhysicsWorld::new(Settings::default());
            let floor = world
                .create_body(&BodyDesc {
                    kind: BodyKind::Static,
                    position,
                    ..Default::default()
                })
                .unwrap();
            world
                .add_collider(floor, &ColliderDesc::new(shape))
                .unwrap();
            let mut rng = Rng(0x3D + u64::from(heightfield));
            let (mut rays, mut failures) = (0, vec![]);
            for _ in 0..3000 {
                let (i, j) = (rng.below(32) as f32, rng.below(32) as f32);
                let local = match rng.below(4) {
                    0 | 1 => Vec3::new(i + 0.5, 0.0, j + 0.5),
                    2 => Vec3::new(i + 0.5, 0.0, j.max(1.0)),
                    _ => Vec3::new(i.max(1.0), 0.0, j.max(1.0)),
                };
                let target = position + local;
                let eye = position + Vec3::new(rng.range(0.0, 32.0), 1.7, rng.range(0.0, 32.0));
                let distance = (target - eye).length();
                rays += 1;
                let hit = world.raycast(eye, target - eye, 100.0, &filter);
                if !hit.is_some_and(|h| h.body == floor && (h.distance - distance).abs() < 1e-3) {
                    failures.push(format!("{eye:?} → {target:?}: {hit:?}"));
                }
            }
            let what = if heightfield { "heightfield" } else { "mesh" };
            assert_none_failed(&format!("{what} floor at {position:?}"), rays, &failures);
        }
    }
}

type F64x3 = [f64; 3];

fn f64x3(p: Vec3) -> F64x3 {
    [f64::from(p.x), f64::from(p.y), f64::from(p.z)]
}

fn sub(p: F64x3, q: F64x3) -> F64x3 {
    [p[0] - q[0], p[1] - q[1], p[2] - q[2]]
}

fn dot(p: F64x3, q: F64x3) -> f64 {
    p[0] * q[0] + p[1] * q[1] + p[2] * q[2]
}

fn cross(p: F64x3, q: F64x3) -> F64x3 {
    [
        p[1] * q[2] - p[2] * q[1],
        p[2] * q[0] - p[0] * q[2],
        p[0] * q[1] - p[1] * q[0],
    ]
}

/// The exact answer, near enough: Möller–Trumbore in `f64` on the same inputs. How far inside
/// the triangle the ray passes (its smallest barycentric coordinate, negative outside), whether
/// it meets the front face (against the counter-clockwise normal), the distance to the plane and
/// the cosine between the ray and the normal.
fn reference(origin: Vec3, dir: Vec3, v: &[Vec3; 3]) -> (f64, bool, f64, f64) {
    let (o, d) = (f64x3(origin), f64x3(dir));
    let (e1, e2) = (sub(f64x3(v[1]), f64x3(v[0])), sub(f64x3(v[2]), f64x3(v[0])));
    let p = cross(d, e2);
    let det = dot(e1, p);
    let s = sub(o, f64x3(v[0]));
    let u = dot(s, p) / det;
    let q = cross(s, e1);
    let w = dot(d, q) / det;
    let t = dot(e2, q) / det;
    let n = cross(e1, e2);
    let cos = dot(n, d) / (dot(n, n) * dot(d, d)).sqrt();
    (u.min(w).min(1.0 - u - w), det > 0.0, t, cos)
}

/// Away from the cases rounding decides (a ray within 0.1% of an edge, grazing the plane, or
/// starting on it), the watertight test answers as exact arithmetic does: hit or miss, front
/// faces only unless both sides count, and the distance to within a few ulps of the coordinates
/// over the slant. Triangles of 1 cm, 1 m and 30 m in every orientation, seen along every axis
/// both ways.
#[test]
fn ray_triangle_agrees_with_an_f64_reference_wherever_the_answer_is_clear() {
    let mut rng = Rng(0x7E57);
    let (mut clear, mut hits, mut back_hits) = (0, 0, 0);
    for k in 0..40_000 {
        let size = [0.01f32, 1.0, 30.0][k % 3];
        let centre = Vec3::new(
            rng.range(-100.0, 100.0),
            rng.range(-100.0, 100.0),
            rng.range(-100.0, 100.0),
        );
        let v = [0; 3].map(|_| centre + rng.direction() * size);
        let origin = centre + rng.direction() * (size * rng.range(1.5, 40.0));
        let (a, b) = (rng.range(-0.3, 1.3), rng.range(-0.3, 1.3));
        let aim = v[0] + (v[1] - v[0]) * a + (v[2] - v[0]) * b;
        let dir = if k % 5 == 0 {
            rng.direction()
        } else {
            (aim - origin).normalize_or_zero()
        };
        let both_sides = rng.below(2) == 0;
        let (inside, front, t, cos) = reference(origin, dir, &v);
        let clear_case =
            inside.abs() >= 1e-3 && cos.abs() >= 0.05 && t.abs() >= 1e-3 * f64::from(size);
        // A sliver's barycentric coordinates say little about distances: shapely ones only.
        let area = Vec3::length((v[1] - v[0]).cross(v[2] - v[0]));
        if !clear_case || area < 0.1 * size * size {
            continue;
        }
        clear += 1;
        let want = inside > 0.0 && t > 0.0 && (front || both_sides);
        let got = ray_triangle(origin, dir, &v, f32::INFINITY, both_sides);
        let case = format!("{origin:?} {dir:?} {v:?} both sides {both_sides}: f64 t {t}");
        assert_eq!(got.is_some(), want, "{case}: {got:?}");
        let Some(got) = got else { continue };
        hits += 1;
        back_hits += usize::from(!front);
        // The corners, relative to the ray's start, are rounded to f32 where they are (`reach`
        // from the world's origin); the hit point moves with them, the distance by that over the
        // slant.
        let reach = v.iter().fold(origin.abs().max_element(), |r, p| {
            r.max(p.abs().max_element())
        });
        let tolerance = 4.0 * f64::from(f32::EPSILON) * (f64::from(reach) / cos.abs() + t);
        let error = (f64::from(got) - t).abs();
        assert!(error <= tolerance, "{case}: {got} is off by {error:e}");
        // `max_t` bounds it.
        assert!(ray_triangle(origin, dir, &v, got * 0.999, both_sides).is_none());
        assert_eq!(ray_triangle(origin, dir, &v, got, both_sides), Some(got));
    }
    assert!(
        clear > 25_000 && hits > 4_000 && back_hits > 1_000,
        "{clear} clear cases, {hits} hits, {back_hits} from behind"
    );
}

/// The distance is the corners' distances along the ray weighted by the edge functions. A
/// triangle whose area as the ray sees it is not a normal `f32` cannot be weighed: one a few
/// 1e-23 m across gives weights whose products with those distances underflow (the distance comes
/// out as 0, a hit at the ray's start), one some 1e19 m across a sum that overflows (0 again).
/// Both are missed rather than hit at the wrong distance (anything a mesh keeps lies far between
/// the two sizes); the same triangle at an ordinary size is hit where it is.
#[test]
fn a_triangle_too_small_or_too_large_to_weigh_is_missed_rather_than_hit_at_the_wrong_distance() {
    let shape = [
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, -1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    for (scale, depth) in [(4e-23f32, 0.2f32), (1.1e19, 0.5), (1.0, 0.2), (1.0, 0.5)] {
        let v = shape.map(|p| Vec3::new(p.x * scale, p.y * scale, -depth));
        for both_sides in [false, true] {
            let t = ray_triangle(Vec3::ZERO, -Vec3::Z, &v, f32::INFINITY, both_sides);
            if scale == 1.0 {
                assert!(
                    t.is_some_and(|t| (t - depth).abs() < 1e-6),
                    "a {scale} m triangle {depth} m away is hit at {t:?}"
                );
            } else {
                assert!(
                    t.is_none_or(|t| (t - depth).abs() < 1e-6),
                    "a {scale} m triangle {depth} m away is hit at {t:?}"
                );
            }
        }
    }
}

/// Shallow rays that start exactly on a level floor and head down into it hit it: the watertight
/// test interpolates the corners' depths along the ray, which loses digits to corners far beyond
/// the hit, so a triangle square to an axis takes its plane's exact distance (found by the fix's
/// verifier: about a third of such rays missed).
#[test]
fn shallow_rays_starting_on_a_level_floor_hit_it() {
    let n = 65u32;
    let field = Heightfield::new(n, n, vec![0.0; (n * n) as usize], Vec3::ONE).unwrap();
    let mut verts = vec![];
    for j in 0..n {
        for i in 0..n {
            verts.push(Vec3::new(i as f32, 0.0, j as f32));
        }
    }
    let at = |i: u32, j: u32| j * n + i;
    let mut tris = vec![];
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            tris.push([at(i, j), at(i, j + 1), at(i + 1, j + 1)]);
            tris.push([at(i, j), at(i + 1, j + 1), at(i + 1, j)]);
        }
    }
    let mesh = TriangleMesh::new(verts, tris).unwrap();
    let mut w = PhysicsWorld::new(Settings::default());
    let floor = w
        .create_body(&BodyDesc {
            kind: BodyKind::Static,
            ..Default::default()
        })
        .unwrap();
    w.add_collider(floor, &ColliderDesc::new(Shape::Mesh(Arc::new(mesh))))
        .unwrap();
    let mut state = 31337u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    };
    let (mut field_misses, mut world_misses) = (0, 0);
    for _ in 0..5000 {
        let o = Vec3::new(4.0 + 56.0 * next(), 0.0, 4.0 + 56.0 * next());
        let a = std::f32::consts::TAU * next();
        let dy = -(0.01 + 0.59 * next());
        let flat = (1.0 - dy * dy).sqrt();
        let d = Vec3::new(flat * a.cos(), dy, flat * a.sin());
        field_misses += usize::from(field.raycast(o, d, 10.0).is_none());
        world_misses += usize::from(w.raycast(o, d, 10.0, &QueryFilter::default()).is_none());
    }
    assert_eq!(
        (field_misses, world_misses),
        (0, 0),
        "shallow rays from the floor that miss it (heightfield, mesh through the world)"
    );
}
