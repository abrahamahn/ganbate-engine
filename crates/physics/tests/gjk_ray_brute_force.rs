//! Rays against every convex kind compared with brute force: march along the ray testing whether
//! each point is inside the shape, then bisect the first step that enters. Triangles, which have
//! no inside, are compared with an `f64` plane intersection instead.

use ganbate_physics::hull::ConvexHull;
use ganbate_physics::math::{Quat, Vec3};
use ganbate_physics::ray::raycast_convex;
use ganbate_physics::shape::ConvexShape;

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    fn vec(&mut self, r: f32) -> Vec3 {
        Vec3::new(self.range(-r, r), self.range(-r, r), self.range(-r, r))
    }
}

fn distance_to_segment(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let ll = ab.length_squared();
    let s = if ll > 0.0 {
        ((p - a).dot(ab) / ll).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p - (a + ab * s)).length()
}

/// How far a point is outside the shape (negative inside), for shapes with an inside.
fn outside(shape: &ConvexShape, p: Vec3) -> f32 {
    match *shape {
        ConvexShape::Sphere(r) => p.length() - r,
        ConvexShape::Capsule {
            half_height,
            radius,
        } => {
            let (a, b) = (
                Vec3::new(0.0, -half_height, 0.0),
                Vec3::new(0.0, half_height, 0.0),
            );
            distance_to_segment(p, a, b) - radius
        }
        ConvexShape::Segment { a, b, radius } => distance_to_segment(p, a, b) - radius,
        ConvexShape::Box(h) => (p.abs() - h).max_element(),
        ConvexShape::Hull(hull) => hull
            .faces
            .iter()
            .map(|f| f.normal.dot(p) - f.offset)
            .fold(f32::NEG_INFINITY, f32::max),
        ConvexShape::Triangle(_) => unreachable!("a triangle has no inside"),
    }
}

/// The shape's farthest extent along unit direction `n`.
fn support(shape: &ConvexShape, n: Vec3) -> f32 {
    match *shape {
        ConvexShape::Sphere(r) => r,
        ConvexShape::Capsule {
            half_height,
            radius,
        } => (n.y * half_height).abs() + radius,
        ConvexShape::Segment { a, b, radius } => n.dot(a).max(n.dot(b)) + radius,
        ConvexShape::Box(h) => n.abs().dot(h),
        ConvexShape::Hull(hull) => hull
            .vertices
            .iter()
            .map(|v| n.dot(*v))
            .fold(f32::NEG_INFINITY, f32::max),
        ConvexShape::Triangle(t) => t
            .iter()
            .map(|v| n.dot(*v))
            .fold(f32::NEG_INFINITY, f32::max),
    }
}

/// The first `t` at which the ray is inside, by marching `steps` steps and bisecting.
fn sampled_entry(shape: &ConvexShape, origin: Vec3, dir: Vec3, max_t: f32) -> Option<f32> {
    let steps = 4000;
    let inside = |t: f32| outside(shape, origin + dir * t) <= 0.0;
    let first = (0..=steps).find(|&i| inside(max_t * i as f32 / steps as f32))?;
    if first == 0 {
        return Some(0.0);
    }
    let (mut lo, mut hi) = (
        max_t * (first - 1) as f32 / steps as f32,
        max_t * first as f32 / steps as f32,
    );
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if inside(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(hi)
}

/// A sheared box as a hull (the cuboid's loops keep their winding under a map with a positive
/// determinant).
fn sheared_hull(scale: f32) -> ConvexHull {
    const LOOPS: [&[u32]; 6] = [
        &[0, 4, 6, 2],
        &[1, 3, 7, 5],
        &[0, 1, 5, 4],
        &[2, 6, 7, 3],
        &[0, 2, 3, 1],
        &[4, 5, 7, 6],
    ];
    let q = Quat::from_axis_angle(Vec3::new(0.3, 1.0, -0.2), 0.7);
    let vertices = ConvexHull::cuboid(Vec3::new(0.9, 0.5, 0.7))
        .vertices
        .iter()
        .map(|v| q.rotate(Vec3::new(v.x + 0.4 * v.y, v.y + 0.3 * v.z, v.z)) * scale)
        .collect();
    ConvexHull::from_faces(vertices, &LOOPS).unwrap()
}

#[test]
fn rays_against_every_convex_kind_match_sampling_at_every_scale() {
    let mut rng = Rng(0x7a7);
    for scale in [1e-3f32, 1.0, 1e3] {
        let hull = sheared_hull(scale);
        let shapes = [
            ConvexShape::Sphere(0.8 * scale),
            ConvexShape::Capsule {
                half_height: 0.7 * scale,
                radius: 0.4 * scale,
            },
            ConvexShape::Box(Vec3::new(0.9, 0.4, 0.6) * scale),
            ConvexShape::Hull(&hull),
            ConvexShape::Segment {
                a: Vec3::new(-0.6, 0.2, 0.1) * scale,
                b: Vec3::new(0.5, -0.4, 0.3) * scale,
                radius: 0.3 * scale,
            },
        ];
        for shape in &shapes {
            let (mut hits, mut misses) = (0, 0);
            for case in 0..120 {
                // From outside, aimed near the shape; one in four starts inside.
                let origin = if case % 4 == 0 {
                    rng.vec(0.2 * scale)
                } else {
                    rng.vec(1.0).normalize_or_zero() * (3.0 * scale)
                };
                let aim = rng.vec(1.2 * scale);
                let dir = (aim - origin).normalize_or_zero();
                let max_t = 6.0 * scale;
                let hit = raycast_convex(shape, origin, dir, max_t);
                let tol = 2e-4 * scale;
                match (hit, sampled_entry(shape, origin, dir, max_t)) {
                    (Some(hit), Some(t)) => {
                        hits += 1;
                        assert!(
                            (hit.t - t).abs() <= tol,
                            "{shape:?} #{case}: {hit:?} vs {t}"
                        );
                        assert!((hit.normal.length() - 1.0).abs() < 1e-4, "{hit:?}");
                        assert!(hit.normal.dot(dir) <= 0.0, "faces the ray: {hit:?}");
                        if t == 0.0 {
                            assert_eq!((hit.t, hit.normal), (0.0, -dir));
                        } else {
                            // On the surface, and the normal is the surface's: the plane through
                            // the hit across it holds the whole shape behind it.
                            let p = origin + dir * hit.t;
                            assert!(outside(shape, p).abs() <= tol, "{shape:?} #{case}: {hit:?}");
                            let reach = support(shape, hit.normal) - hit.normal.dot(p);
                            assert!(reach.abs() <= tol, "{shape:?} #{case}: {hit:?}, {reach}");
                        }
                    }
                    (None, None) => misses += 1,
                    (Some(hit), None) => {
                        // A graze too thin for the samples: the hit must still be on the surface.
                        let p = origin + dir * hit.t;
                        assert!(outside(shape, p).abs() <= tol, "{shape:?} #{case}: {hit:?}");
                    }
                    (None, Some(t)) => panic!("{shape:?} #{case}: missed, sampling entered at {t}"),
                }
            }
            assert!(
                hits > 20 && misses > 10,
                "{shape:?}: {hits} hits, {misses} misses"
            );
        }
    }
}

#[test]
fn rays_against_triangles_match_a_plane_intersection_from_either_side() {
    let mut rng = Rng(0x7e1);
    let (mut hits, mut misses) = (0, 0);
    for case in 0..400 {
        let tri = [rng.vec(1.0), rng.vec(1.0), rng.vec(1.0)];
        let origin = rng.vec(1.0).normalize_or_zero() * 3.0;
        // Half the rays aim at a point of the triangle, half anywhere near it.
        let aim = if case % 2 == 0 {
            let (u, v) = (rng.unit(), rng.unit());
            let (u, v) = if u + v > 1.0 {
                (1.0 - u, 1.0 - v)
            } else {
                (u, v)
            };
            tri[0] + (tri[1] - tri[0]) * u + (tri[2] - tri[0]) * v
        } else {
            rng.vec(1.0)
        };
        let dir = (aim - origin).normalize_or_zero();
        let hit = raycast_convex(&ConvexShape::Triangle(tri), origin, dir, 10.0);
        // In f64: where the ray meets the plane, and whether that point is inside all three edges.
        let d = |v: Vec3| [v.x as f64, v.y as f64, v.z as f64];
        let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let [a, b, c] = tri.map(d);
        let n = cross(sub(b, a), sub(c, a));
        let (o, r) = (d(origin), d(dir));
        let t = dot(n, sub(a, o)) / dot(n, r);
        let p = [o[0] + r[0] * t, o[1] + r[1] * t, o[2] + r[2] * t];
        let inside = [(a, b), (b, c), (c, a)]
            .iter()
            .all(|&(u, v)| dot(n, cross(sub(v, u), sub(p, u))) >= 0.0);
        let want = ((0.0..=10.0).contains(&t) && inside).then_some(t as f32);
        match (hit, want) {
            (Some(hit), Some(t)) => {
                hits += 1;
                assert!((hit.t - t).abs() < 1e-4, "#{case}: {hit:?} vs {t}");
                let unit = n.map(|x| x / dot(n, n).sqrt());
                let along = dot(
                    unit,
                    [
                        hit.normal.x as f64,
                        hit.normal.y as f64,
                        hit.normal.z as f64,
                    ],
                );
                assert!((along.abs() - 1.0).abs() < 1e-5, "#{case}: {hit:?}");
                assert!(hit.normal.dot(dir) < 0.0, "#{case}: faces the ray");
            }
            (None, None) => misses += 1,
            // Rays through an edge or a vertex may fall either way.
            (hit, want) => {
                let edge = [(a, b), (b, c), (c, a)].iter().any(|&(u, v)| {
                    let e = sub(v, u);
                    let w = sub(p, u);
                    let s = (dot(w, e) / dot(e, e)).clamp(0.0, 1.0);
                    let q = sub(w, e.map(|x| x * s));
                    dot(q, q).sqrt() < 1e-4
                });
                assert!(edge, "#{case}: {hit:?} vs {want:?}");
            }
        }
    }
    assert!(hits > 150 && misses > 100, "{hits} hits, {misses} misses");
}
