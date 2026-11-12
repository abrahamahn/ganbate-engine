//! GJK next to kilometre-sized faces (review finding GJK-STALL-HUGE-FACES).
//!
//! Near a corner of a huge box the support point in `−v` is a far corner, kilometres away, while
//! the closest point is under a metre from the origin: its barycentric weight on that corner is
//! about 1e-8, which an `f32` solve of the simplex (a difference of two numbers near 1) loses.
//! The solve then fell back to the old edge, the search stopped as "no closer", and the distance
//! came out centimetres long (0.8249 instead of 0.7986 next to a 1 km box; 0.709 instead of
//! 0.480 next to a 10 km one), past the documented accuracy of about a millionth of the size.
//! The simplex is now solved in `f64`. The reference here is exact: an `f64` search over the
//! same `f32` support points GJK sees. Every shape is synthetic.

use std::ops::{Add, Mul, Sub};

use ganbate_physics::gjk::distance;
use ganbate_physics::{Convex, ConvexShape, Iso, Quat, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
struct D3 {
    x: f64,
    y: f64,
    z: f64,
}

impl D3 {
    fn of(v: Vec3) -> D3 {
        D3 {
            x: v.x as f64,
            y: v.y as f64,
            z: v.z as f64,
        }
    }
    fn dot(self, o: D3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    fn length(self) -> f64 {
        self.dot(self).sqrt()
    }
}

impl Add for D3 {
    type Output = D3;
    fn add(self, o: D3) -> D3 {
        D3 {
            x: self.x + o.x,
            y: self.y + o.y,
            z: self.z + o.z,
        }
    }
}

impl Sub for D3 {
    type Output = D3;
    fn sub(self, o: D3) -> D3 {
        D3 {
            x: self.x - o.x,
            y: self.y - o.y,
            z: self.z - o.z,
        }
    }
}

impl Mul<f64> for D3 {
    type Output = D3;
    fn mul(self, s: f64) -> D3 {
        D3 {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

/// A core's points, edges and face triangles, placed as GJK's support function places them
/// (`f32`, relative to `origin`), in `f64`.
struct Core {
    points: Vec<D3>,
    edges: Vec<(D3, D3)>,
    triangles: Vec<[D3; 3]>,
}

fn core(c: &Convex, origin: Vec3) -> Core {
    let offset = c.iso.position - origin;
    let place = |p: Vec3| D3::of(c.iso.rotation.rotate(p) + offset);
    match c.shape {
        ConvexShape::Sphere(_) => Core {
            points: vec![place(Vec3::ZERO)],
            edges: vec![],
            triangles: vec![],
        },
        ConvexShape::Capsule { half_height, .. } => {
            let (a, b) = (
                place(Vec3::new(0.0, -half_height, 0.0)),
                place(Vec3::new(0.0, half_height, 0.0)),
            );
            Core {
                points: vec![a, b],
                edges: vec![(a, b)],
                triangles: vec![],
            }
        }
        ConvexShape::Box(h) => {
            // Corner i has +x, +y, +z where bits 0, 1, 2 are set.
            let corners: Vec<D3> = (0..8)
                .map(|i| {
                    let sign = |bit: usize, v: f32| if i & (1 << bit) != 0 { v } else { -v };
                    place(Vec3::new(sign(0, h.x), sign(1, h.y), sign(2, h.z)))
                })
                .collect();
            let mut edges = Vec::new();
            for i in 0..8usize {
                for bit in 0..3 {
                    if i & (1 << bit) == 0 {
                        edges.push((corners[i], corners[i | (1 << bit)]));
                    }
                }
            }
            let quads = [
                [0, 4, 6, 2],
                [1, 3, 7, 5],
                [0, 1, 5, 4],
                [2, 6, 7, 3],
                [0, 2, 3, 1],
                [4, 5, 7, 6],
            ];
            let mut triangles = Vec::new();
            for q in quads {
                triangles.push([corners[q[0]], corners[q[1]], corners[q[2]]]);
                triangles.push([corners[q[0]], corners[q[2]], corners[q[3]]]);
            }
            Core {
                points: corners,
                edges,
                triangles,
            }
        }
        _ => unreachable!("not used here"),
    }
}

fn closest_on_segment(p: D3, a: D3, b: D3) -> D3 {
    let ab = b - a;
    let ll = ab.dot(ab);
    let t = if ll > 0.0 {
        ((p - a).dot(ab) / ll).clamp(0.0, 1.0)
    } else {
        0.0
    };
    a + ab * t
}

/// Ericson, Real-Time Collision Detection 5.1.5.
fn closest_on_triangle(p: D3, t: &[D3; 3]) -> D3 {
    let [a, b, c] = *t;
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}

/// Ericson 5.1.9: the closest points of two segments.
fn closest_between_segments(p1: D3, q1: D3, p2: D3, q2: D3) -> (D3, D3) {
    let (d1, d2, r) = (q1 - p1, q2 - p2, p1 - p2);
    let (a, e, f) = (d1.dot(d1), d2.dot(d2), d2.dot(r));
    let c = d1.dot(r);
    let b = d1.dot(d2);
    let denom = a * e - b * b;
    let mut s = if denom > 0.0 {
        ((b * f - c * e) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = (b * s + f) / e;
    if t < 0.0 {
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - c) / a).clamp(0.0, 1.0);
    }
    (p1 + d1 * s, p2 + d2 * t)
}

/// The exact distance between two separated cores: for convex polytopes and segments the
/// closest pair is a point and a face, a point and an edge, or two edges.
fn reference(a: &Core, b: &Core) -> f64 {
    let mut best = f64::INFINITY;
    for (x, y) in [(a, b), (b, a)] {
        for &p in &x.points {
            for &q in &y.points {
                best = best.min((p - q).length());
            }
            for &(s, e) in &y.edges {
                best = best.min((p - closest_on_segment(p, s, e)).length());
            }
            for t in &y.triangles {
                best = best.min((p - closest_on_triangle(p, t)).length());
            }
        }
    }
    for &(p1, q1) in &a.edges {
        for &(p2, q2) in &b.edges {
            let (ca, cb) = closest_between_segments(p1, q1, p2, q2);
            best = best.min((cb - ca).length());
        }
    }
    best
}

/// GJK's core distance (the surfaces' plus both radii) and the exact one between the same
/// `f32` support points.
fn measure(a: &Convex, b: &Convex) -> (f64, f64) {
    let r = distance(a, b);
    assert!(!r.overlap, "{r:?}");
    let got = (r.distance + a.radius() + b.radius()) as f64;
    let origin = a.center();
    (got, reference(&core(a, origin), &core(b, origin)))
}

#[test]
fn next_to_a_kilometre_box_gjk_finds_the_distance() {
    // The review's case: a 7.7 cm box 0.8 m off a corner region of a 0.9 x 0.4 x 2 km box.
    let b = Convex::new(
        ConvexShape::Box(Vec3::new(431.78397, 212.93297, 997.4582)),
        Iso::new(
            Vec3::new(0.18068469, 0.56211805, -0.64751863),
            Quat::new(-0.2603255, 0.93430746, 0.24014165, -0.040402073),
        ),
    );
    let a = Convex::new(
        ConvexShape::Box(Vec3::splat(0.038309462)),
        Iso::new(
            Vec3::new(-271.58792, -484.22668, 957.332),
            Quat::new(0.6759044, 0.4149782, 0.27906513, 0.5413584),
        ),
    );
    for (x, y) in [(&a, &b), (&b, &a)] {
        let (got, want) = measure(x, y);
        // The review's f64 brute force over the world-placed corners: 0.798607647756489.
        assert!((want - 0.7986076).abs() < 1e-4, "reference {want}");
        assert!((got - want).abs() < 1e-4, "GJK {got}, exactly {want}");
    }
}

#[test]
fn next_to_a_ten_kilometre_box_gjk_finds_the_distance() {
    let b = Convex::new(
        ConvexShape::Box(Vec3::new(3895.7446, 7884.9688, 8761.46)),
        Iso::new(
            Vec3::new(0.6555114, -0.29064476, -0.0071083307),
            Quat::new(-0.84272176, -0.31348744, 0.35615847, 0.254356),
        ),
    );
    let a = Convex::new(
        ConvexShape::Capsule {
            half_height: 0.19802107,
            radius: 0.06633686,
        },
        Iso::new(
            Vec3::new(6061.539, 6279.761, 8828.4375),
            Quat::new(-0.66340536, -0.27377102, -0.44290718, -0.5373788),
        ),
    );
    for (x, y) in [(&a, &b), (&b, &a)] {
        let (got, want) = measure(x, y);
        // About a millionth of the size is all `f32` placement allows at 10 km.
        assert!((got - want).abs() < 2e-3, "GJK {got}, exactly {want}");
    }
}

/// splitmix64, seeded as in the review's sweep so that its cases come up again.
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
    fn rotation(&mut self) -> Quat {
        loop {
            let q = Quat::new(
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            );
            if q.length() > 0.2 {
                return q.normalize();
            }
        }
    }
}

/// Small spheres, boxes, points and capsules 0.3 to 1.5 m off a face, an edge or a corner of
/// boxes up to 300 m, 1 km, 3 km and 10 km: GJK within a millionth of the size of the exact
/// distance.
#[test]
fn small_shapes_beside_huge_boxes_are_measured_to_a_millionth_of_the_size() {
    for big in [300.0f32, 1000.0, 3000.0, 10000.0] {
        let mut rng = Rng(0xb17);
        let mut worst = (0.0f64, String::new());
        let mut checked = 0;
        for case in 0..600 {
            let half = Vec3::new(
                rng.range(0.2, 1.0),
                rng.range(0.2, 1.0),
                rng.range(0.2, 1.0),
            ) * big;
            let biso = Iso::new(rng.vec(1.0), rng.rotation());
            let b = Convex::new(ConvexShape::Box(half), biso);
            let mut local = Vec3::new(
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            )
            .mul_elem(half);
            // 0: on a face, 1: on an edge, 2: at a corner.
            let kind = case % 3;
            for k in 0..=kind {
                let axis = (case + k) % 3;
                local[axis] = if rng.unit() < 0.5 {
                    -half[axis]
                } else {
                    half[axis]
                };
            }
            let out = biso.inverse_vector(rng.vec(1.0)).normalize_or_zero();
            let shape = match case % 4 {
                0 => ConvexShape::Sphere(rng.range(0.01, 0.2)),
                1 => ConvexShape::Box(Vec3::splat(rng.range(0.01, 0.2))),
                2 => ConvexShape::Sphere(0.1),
                _ => ConvexShape::Capsule {
                    half_height: rng.range(0.01, 0.3),
                    radius: rng.range(0.01, 0.1),
                },
            };
            let a_local = local + out * rng.range(0.3, 1.5);
            let a = Convex::new(shape, Iso::new(biso.point(a_local), rng.rotation()));
            // The direction off the surface is random: keep the cases whose core is clearly
            // outside the box (the reference measures separated cores only).
            let outside = |i: usize| (a_local[i].abs() - half[i]).max(0.0) as f64;
            let clear = (outside(0).powi(2) + outside(1).powi(2) + outside(2).powi(2)).sqrt();
            let reach = match shape {
                ConvexShape::Box(h) => h.length(),
                ConvexShape::Capsule { half_height, .. } => half_height,
                _ => 0.0,
            };
            if clear <= reach as f64 + 0.01 {
                continue;
            }
            checked += 1;
            let (got, want) = measure(&a, &b);
            let err = (got - want).abs();
            if err > worst.0 {
                worst = (err, format!("case {case}: GJK {got}, exactly {want}"));
            }
        }
        assert!(checked > 300, "{checked} cases");
        assert!(
            worst.0 <= 1e-6 * big as f64,
            "boxes up to {big} m: worst error {:e} m, {}",
            worst.0,
            worst.1
        );
    }
}
