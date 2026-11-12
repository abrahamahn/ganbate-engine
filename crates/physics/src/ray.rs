//! Rays against single shapes, in the shape's local frame.
//!
//! Every convex shape is a closed set: a ray that starts inside it or on its surface hits at
//! `t = 0` with the normal against its direction. Spheres and capsules are solved analytically
//! (the capsule as its cylinder and two cap spheres), boxes with slabs, hulls by clipping the ray
//! against every face plane, triangles from either side with the watertight test of Woop,
//! Benthin and Wald ([`ray_triangle`]), which no ray slips through between two triangles that
//! share an edge. Quadratics use the cancellation-free forms (the discriminant from the ray's
//! distance to the centre or axis, the near root as `c / q`), so a ray from far away still hits
//! small shapes accurately.
//!
//! Only `+ − × ÷` and `sqrt` on `f32`, no allocation: deterministic.

use crate::hull::ConvexHull;
use crate::math::Vec3;
use crate::mesh::ray_triangle;
use crate::shape::{ConvexShape, Shape};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    /// Distance along the unit ray direction.
    pub t: f32,
    /// Unit surface normal at the hit, local frame, facing the ray (against its direction).
    pub normal: Vec3,
    /// The triangle id for meshes and heightfields, 0 otherwise.
    pub feature: u32,
}

/// The first hit of the ray `origin + t·dir` (`dir` unit length, local frame) with `shape` for
/// `t` in `[0, max_t]`. A ray that starts inside a convex shape hits at `t = 0` with the normal
/// pointing against `dir`. Meshes are hit on front faces, heightfields from above.
pub fn raycast(shape: &Shape, origin: Vec3, dir: Vec3, max_t: f32) -> Option<RayHit> {
    let from_mesh = |h: crate::mesh::MeshHit| RayHit {
        t: h.t,
        normal: h.normal,
        feature: h.triangle,
    };
    match shape {
        Shape::Mesh(mesh) => mesh.raycast(origin, dir, max_t, false).map(from_mesh),
        Shape::Heightfield(field) => field.raycast(origin, dir, max_t).map(from_mesh),
        _ => raycast_convex(&shape.as_convex()?, origin, dir, max_t),
    }
}

/// The same for a convex shape (including triangles, hit on either side).
pub fn raycast_convex(shape: &ConvexShape, origin: Vec3, dir: Vec3, max_t: f32) -> Option<RayHit> {
    if max_t.is_nan() || max_t < 0.0 || !origin.is_finite() || !dir.is_finite() {
        return None;
    }
    // A unit direction (to rounding) is its own normal from inside, bit for bit.
    let dd = dir.length_squared();
    let back = if (dd - 1.0).abs() <= 4.0 * f32::EPSILON {
        -dir
    } else {
        -dir.try_normalize()?
    };
    let ray = Ray {
        origin,
        dir,
        max_t,
        back,
    };
    let (t, normal) = match *shape {
        ConvexShape::Sphere(radius) => ray_sphere(&ray, Vec3::ZERO, radius),
        ConvexShape::Capsule {
            half_height,
            radius,
        } => ray_capsule(
            &ray,
            Vec3::new(0.0, -half_height, 0.0),
            Vec3::new(0.0, half_height, 0.0),
            radius,
        ),
        ConvexShape::Box(half) => ray_box(&ray, half),
        ConvexShape::Hull(hull) => ray_hull(&ray, hull),
        ConvexShape::Triangle(v) => ray_two_sided_triangle(&ray, &v),
        ConvexShape::Segment { a, b, radius } => ray_capsule(&ray, a, b, radius),
    }?;
    Some(RayHit {
        t,
        normal,
        feature: 0,
    })
}

#[derive(Clone, Copy, Debug)]
struct Ray {
    origin: Vec3,
    dir: Vec3,
    max_t: f32,
    /// The normal of a hit from inside: against the direction, unit length.
    back: Vec3,
}

/// A ball of `radius` about `center`.
fn ray_sphere(ray: &Ray, center: Vec3, radius: f32) -> Option<(f32, Vec3)> {
    let m = ray.origin - center;
    let ml = m.length();
    if ml <= radius {
        return Some((0.0, ray.back));
    }
    sphere_entry(ray, m, ml, radius)
}

/// Where a ray from outside a ball enters it: `m` is the ray's origin relative to the centre and
/// `ml` its length.
fn sphere_entry(ray: &Ray, m: Vec3, ml: f32, radius: f32) -> Option<(f32, Vec3)> {
    let d = ray.dir;
    let b = m.dot(d);
    if b >= 0.0 {
        // Heading away from the centre (or across it at a right angle).
        return None;
    }
    // b² − |d|²(|m|² − r²) written as |d|² r² − |m × d|²: no cancellation for a far origin.
    let cross = m.cross(d);
    let disc = d.dot(d) * radius * radius - cross.dot(cross);
    if disc < 0.0 {
        return None;
    }
    // The near root c / q with q = −b + √disc > 0, c = |m|² − r² as a product.
    let t = ((ml - radius) * (ml + radius) / (disc.sqrt() - b)).max(0.0);
    if t > ray.max_t {
        return None;
    }
    let normal = (m + d * t).try_normalize().unwrap_or(ray.back);
    Some((t, normal))
}

/// The points within `radius` of the segment `a b`: a capsule between arbitrary points.
fn ray_capsule(ray: &Ray, a: Vec3, b: Vec3, radius: f32) -> Option<(f32, Vec3)> {
    let (o, d) = (ray.origin, ray.dir);
    let ab = b - a;
    let ao = o - a;
    let ll = ab.dot(ab);
    let s = if ll > 0.0 {
        (ao.dot(ab) / ll).clamp(0.0, 1.0)
    } else {
        0.0
    };
    if (ao - ab * s).length_squared() <= radius * radius {
        return Some((0.0, ray.back));
    }
    let mut best: Option<(f32, Vec3)> = None;
    if ll > 0.0 {
        // The cylinder's side, in the plane across the axis.
        let length = ll.sqrt();
        let u = ab * (1.0 / length);
        let (ou, du) = (ao.dot(u), d.dot(u));
        let across_o = ao - u * ou;
        let across_d = d - u * du;
        let aa = across_d.dot(across_d);
        let bb = across_o.dot(across_d);
        let distance = across_o.length();
        // Only a ray from outside the infinite cylinder, heading towards the axis, enters it
        // through the side.
        if aa > 0.0 && bb < 0.0 && distance > radius {
            let cross = ao.cross(d).dot(u);
            let disc = aa * radius * radius - cross * cross;
            if disc >= 0.0 {
                let t = (distance - radius) * (distance + radius) / (disc.sqrt() - bb);
                let along = ou + du * t;
                if t <= ray.max_t && (0.0..=length).contains(&along) {
                    let normal = (across_o + across_d * t)
                        .try_normalize()
                        .unwrap_or(ray.back);
                    best = Some((t, normal));
                }
            }
        }
    }
    // The cap spheres (the first entry into the union is the first entry into any part).
    for end in [a, b] {
        let m = o - end;
        if let Some(hit) = sphere_entry(ray, m, m.length(), radius)
            && best.is_none_or(|(t, _)| hit.0 < t)
        {
            best = Some(hit);
        }
    }
    best
}

/// The box `[-half, half]`, by slabs; the normal is the entered face's.
fn ray_box(ray: &Ray, half: Vec3) -> Option<(f32, Vec3)> {
    let (o, d) = (ray.origin, ray.dir);
    if o.x.abs() <= half.x && o.y.abs() <= half.y && o.z.abs() <= half.z {
        return Some((0.0, ray.back));
    }
    let mut t_in = 0.0f32;
    let mut t_out = ray.max_t;
    let mut entered: Option<(usize, f32)> = None;
    for i in 0..3 {
        if d[i].abs() < 1e-30 {
            // Parallel to this slab: inside it or never.
            if o[i].abs() > half[i] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / d[i];
        // The face whose outward normal opposes the direction is the near one.
        let (near, far, sign) = if d[i] > 0.0 {
            (-half[i], half[i], -1.0)
        } else {
            (half[i], -half[i], 1.0)
        };
        let t0 = (near - o[i]) * inv;
        let t1 = (far - o[i]) * inv;
        if entered.is_none() || t0 > t_in {
            t_in = t_in.max(t0);
            entered = Some((i, sign));
        }
        t_out = t_out.min(t1);
        if t_in > t_out {
            return None;
        }
    }
    let (axis, sign) = entered?;
    let mut normal = Vec3::ZERO;
    normal[axis] = sign;
    Some((t_in, normal))
}

/// A convex hull, by clipping the ray against every face plane; the normal is the entered
/// face's.
fn ray_hull(ray: &Ray, hull: &ConvexHull) -> Option<(f32, Vec3)> {
    let (o, d) = (ray.origin, ray.dir);
    let mut t_in = 0.0f32;
    let mut t_out = ray.max_t;
    let mut entered: Option<Vec3> = None;
    let mut inside = true;
    for face in &hull.faces {
        let height = face.normal.dot(o) - face.offset;
        let rate = face.normal.dot(d);
        if height > 0.0 {
            // In front of this face: the ray must come in through it.
            inside = false;
            if rate >= 0.0 {
                return None;
            }
            let t = height / -rate;
            if entered.is_none() || t > t_in {
                t_in = t_in.max(t);
                entered = Some(face.normal);
            }
        } else if rate > 0.0 {
            // Behind it, heading out through it.
            t_out = t_out.min(-height / rate);
        }
        if t_in > t_out {
            return None;
        }
    }
    if inside {
        return Some((0.0, ray.back));
    }
    entered.map(|n| (t_in, n))
}

/// A triangle seen from either side; the normal faces the ray.
fn ray_two_sided_triangle(ray: &Ray, v: &[Vec3; 3]) -> Option<(f32, Vec3)> {
    let t = ray_triangle(ray.origin, ray.dir, v, ray.max_t, true)?;
    let n = (v[1] - v[0]).cross(v[2] - v[0]).try_normalize()?;
    Some((t, if n.dot(ray.dir) > 0.0 { -n } else { n }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Iso, Quat};
    use crate::mesh::{Heightfield, TriangleMesh};
    use std::sync::Arc;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn close_v(a: Vec3, b: Vec3, tol: f32) -> bool {
        (a - b).length() <= tol
    }

    fn ray(shape: &ConvexShape, origin: Vec3, dir: Vec3) -> Option<RayHit> {
        raycast_convex(shape, origin, dir.normalize_or_zero(), 100.0)
    }

    /// Checks a hit's distance and normal, and that the normal is unit and faces the ray.
    fn assert_hit(hit: Option<RayHit>, dir: Vec3, t: f32, normal: Vec3, tol: f32) {
        let hit = hit.unwrap_or_else(|| panic!("no hit, want t = {t}"));
        assert!(close(hit.t, t, tol), "{hit:?}, want t = {t}");
        assert!(
            close_v(hit.normal, normal, tol.max(1e-5)),
            "{hit:?}, want {normal:?}"
        );
        assert!(close(hit.normal.length(), 1.0, 1e-5));
        assert!(
            hit.normal.dot(dir) <= 0.0,
            "the normal faces the ray: {hit:?}"
        );
        assert_eq!(hit.feature, 0);
    }

    #[test]
    fn rays_hit_spheres_at_the_analytic_distance() {
        let s = ConvexShape::Sphere(2.0);
        let root3 = 3.0f32.sqrt();
        assert_hit(
            ray(&s, Vec3::new(-5.0, 1.0, 0.0), Vec3::X),
            Vec3::X,
            5.0 - root3,
            Vec3::new(-root3, 1.0, 0.0) * 0.5,
            1e-5,
        );
        assert!(ray(&s, Vec3::new(-5.0, 2.01, 0.0), Vec3::X).is_none());
        assert!(ray(&s, Vec3::new(-5.0, 1.0, 0.0), -Vec3::X).is_none());
        assert!(raycast_convex(&s, Vec3::new(-5.0, 1.0, 0.0), Vec3::X, 3.2).is_none());
        assert!(raycast_convex(&s, Vec3::new(-5.0, 1.0, 0.0), Vec3::X, 3.3).is_some());
        // Nearly grazing, and from 10 km away.
        let hit = ray(&s, Vec3::new(-5.0, 1.9999, 0.0), Vec3::X).unwrap();
        assert!(
            close(hit.t, 5.0 - (4.0f32 - 1.9999 * 1.9999).sqrt(), 1e-4),
            "{hit:?}"
        );
        let hit = raycast_convex(&s, Vec3::new(-10_000.0, 1.0, 0.0), Vec3::X, 1e5).unwrap();
        assert!(close(hit.t, 10_000.0 - root3, 2e-3), "{hit:?}");
        assert!(close_v(hit.normal, Vec3::new(-root3, 1.0, 0.0) * 0.5, 1e-3));
    }

    #[test]
    fn rays_hit_capsules_on_the_side_and_on_both_caps() {
        let c = ConvexShape::Capsule {
            half_height: 1.0,
            radius: 0.5,
        };
        // The side, the top cap from above, the bottom cap from below, a cap from the side.
        assert_hit(
            ray(&c, Vec3::new(-3.0, 0.4, 0.0), Vec3::X),
            Vec3::X,
            2.5,
            -Vec3::X,
            1e-5,
        );
        assert_hit(
            ray(&c, Vec3::new(0.3, 5.0, 0.0), -Vec3::Y),
            -Vec3::Y,
            3.6,
            Vec3::new(0.6, 0.8, 0.0),
            1e-5,
        );
        let y = 1.0 + 0.21f32.sqrt();
        assert_hit(
            ray(&c, Vec3::new(0.0, -5.0, 0.2), Vec3::Y),
            Vec3::Y,
            5.0 - y,
            Vec3::new(0.0, -(y - 1.0), 0.2) * 2.0,
            1e-5,
        );
        assert_hit(
            ray(&c, Vec3::new(-3.0, 1.3, 0.0), Vec3::X),
            Vec3::X,
            2.6,
            Vec3::new(-0.8, 0.6, 0.0),
            1e-5,
        );
        assert!(ray(&c, Vec3::new(-3.0, 1.6, 0.0), Vec3::X).is_none());
        assert!(ray(&c, Vec3::new(-3.0, 0.0, 0.6), Vec3::X).is_none());
        // Obliquely onto the side: the hit lies on the cylinder within the straight part.
        // Seen from above this ray crosses the axis, at height 0.3.
        let dir = Vec3::new(1.0, -0.3, 0.2).normalize_or_zero();
        let origin = Vec3::new(-4.0, 1.5, -0.8);
        let hit = ray(&c, origin, dir).unwrap();
        let p = origin + dir * hit.t;
        assert!(
            close((p.x * p.x + p.z * p.z).sqrt(), 0.5, 1e-5) && p.y.abs() <= 1.0,
            "{p:?}"
        );
        assert!(close_v(hit.normal, Vec3::new(p.x, 0.0, p.z) * 2.0, 1e-4));
        // With no height it is a sphere.
        let ball = ConvexShape::Capsule {
            half_height: 0.0,
            radius: 0.5,
        };
        assert_hit(
            ray(&ball, Vec3::new(0.0, 0.0, -3.0), Vec3::Z),
            Vec3::Z,
            2.5,
            -Vec3::Z,
            1e-5,
        );
    }

    #[test]
    fn rays_hit_boxes_on_the_face_they_enter() {
        let b = ConvexShape::Box(Vec3::new(1.0, 2.0, 3.0));
        assert_hit(
            ray(&b, Vec3::new(-5.0, 0.5, 0.5), Vec3::X),
            Vec3::X,
            4.0,
            -Vec3::X,
            1e-6,
        );
        assert_hit(
            ray(&b, Vec3::new(0.5, 10.0, -1.0), -Vec3::Y),
            -Vec3::Y,
            8.0,
            Vec3::Y,
            1e-6,
        );
        assert_hit(
            ray(&b, Vec3::new(0.5, 1.0, -9.0), Vec3::Z),
            Vec3::Z,
            6.0,
            -Vec3::Z,
            1e-6,
        );
        // Diagonally: the x slab is entered last, at x = 1 (where y = 1 is within the box).
        let dir = Vec3::new(-1.0, -1.0, 0.0).normalize_or_zero();
        assert_hit(
            ray(&b, Vec3::new(5.0, 5.0, 0.0), dir),
            dir,
            4.0 * 2f32.sqrt(),
            Vec3::X,
            1e-5,
        );
        assert!(ray(&b, Vec3::new(5.0, 2.5, 0.0), -Vec3::X).is_none());
        // Starting higher, it leaves the x slab (s/√2 in [4, 6]) before entering the y slab
        // (s/√2 in [6.5, 10.5]).
        assert!(ray(&b, Vec3::new(5.0, 8.5, 0.0), dir).is_none());
        assert!(raycast_convex(&b, Vec3::new(-5.0, 0.5, 0.5), Vec3::X, 3.9).is_none());
    }

    /// The cuboid's vertex loops (see `ConvexHull::cuboid`), for hulls bent out of a box.
    const CUBOID_LOOPS: [&[u32]; 6] = [
        &[0, 4, 6, 2],
        &[1, 3, 7, 5],
        &[0, 1, 5, 4],
        &[2, 6, 7, 3],
        &[0, 2, 3, 1],
        &[4, 5, 7, 6],
    ];

    /// A sheared, turned and shifted box: a hull with no two faces at right angles.
    fn slanted_hull() -> ConvexHull {
        let q = Quat::from_axis_angle(Vec3::new(0.3, 1.0, -0.2), 0.7);
        let vertices = ConvexHull::cuboid(Vec3::new(1.0, 0.6, 0.8))
            .vertices
            .iter()
            .map(|v| q.rotate(Vec3::new(v.x + 0.4 * v.y, v.y + 0.3 * v.z, v.z)) + Vec3::X * 0.2)
            .collect();
        ConvexHull::from_faces(vertices, &CUBOID_LOOPS).unwrap()
    }

    fn inside_hull(h: &ConvexHull, p: Vec3) -> bool {
        h.faces.iter().all(|f| f.normal.dot(p) <= f.offset)
    }

    /// The entry of a ray into a hull by marching along it, refined by bisection.
    fn sampled_entry(h: &ConvexHull, origin: Vec3, dir: Vec3, max_t: f32) -> Option<f32> {
        let step = 1e-3;
        let n = (max_t / step) as usize;
        let first = (0..=n).find(|&i| inside_hull(h, origin + dir * (i as f32 * step)))?;
        if first == 0 {
            return Some(0.0);
        }
        let (mut lo, mut hi) = ((first - 1) as f32 * step, first as f32 * step);
        for _ in 0..30 {
            let mid = 0.5 * (lo + hi);
            if inside_hull(h, origin + dir * mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(hi)
    }

    #[test]
    fn rays_hit_hulls_where_sampling_along_the_ray_finds_them() {
        let hull = slanted_hull();
        let shape = ConvexShape::Hull(&hull);
        let mut state = 0x1234_5678u32;
        let mut next = move || {
            // xorshift32, mapped to [-1, 1).
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        };
        let (mut hits, mut misses) = (0, 0);
        for _ in 0..150 {
            let origin = Vec3::new(next(), next(), next()) * 4.0;
            let aim = Vec3::new(next(), next(), next()) * 1.5;
            let dir = (aim - origin).normalize_or_zero();
            let hit = raycast_convex(&shape, origin, dir, 10.0);
            match (hit, sampled_entry(&hull, origin, dir, 10.0)) {
                (Some(hit), Some(t)) => {
                    hits += 1;
                    assert!(close(hit.t, t, 1e-4), "{origin:?} {dir:?}: {hit:?} vs {t}");
                    if hit.t > 0.0 {
                        // The normal is the face the hit lies on, facing the ray.
                        let p = origin + dir * hit.t;
                        let face = hull
                            .faces
                            .iter()
                            .find(|f| close_v(f.normal, hit.normal, 1e-6));
                        let face = face.expect("the normal is a face normal");
                        assert!(close(face.normal.dot(p), face.offset, 1e-4));
                        assert!(hit.normal.dot(dir) < 0.0);
                    } else {
                        assert_eq!(hit.normal, -dir);
                    }
                }
                (None, None) => misses += 1,
                (hit, t) => panic!("{origin:?} {dir:?}: {hit:?} vs sampled {t:?}"),
            }
        }
        assert!(hits > 30 && misses > 30, "{hits} hits, {misses} misses");
        // The same through the general entry point.
        let arc = Shape::Hull(Arc::new(slanted_hull()));
        let o = Vec3::new(-5.0, 0.1, 0.2);
        assert_eq!(
            raycast(&arc, o, Vec3::X, 10.0),
            raycast_convex(&shape, o, Vec3::X, 10.0)
        );
    }

    #[test]
    fn rays_hit_triangles_from_either_side_with_the_normal_facing_them() {
        let t = ConvexShape::Triangle([
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, -1.0),
        ]);
        assert_hit(
            ray(&t, Vec3::new(0.1, 3.0, 0.0), -Vec3::Y),
            -Vec3::Y,
            3.0,
            Vec3::Y,
            1e-6,
        );
        assert_hit(
            ray(&t, Vec3::new(0.1, -2.0, 0.0), Vec3::Y),
            Vec3::Y,
            2.0,
            -Vec3::Y,
            1e-6,
        );
        let dir = Vec3::new(0.3, -1.0, 0.2).normalize_or_zero();
        let hit = ray(&t, Vec3::new(-0.3, 2.0, -0.4), dir).unwrap();
        assert!(close(
            (Vec3::new(-0.3, 2.0, -0.4) + dir * hit.t).y,
            0.0,
            1e-6
        ));
        assert!(ray(&t, Vec3::new(1.1, 3.0, 0.0), -Vec3::Y).is_none());
        // Edge-on, in its plane: a flat triangle has nothing to hit.
        assert!(ray(&t, Vec3::new(-5.0, 0.0, 0.0), Vec3::X).is_none());
    }

    #[test]
    fn rays_hit_a_segment_with_radius_as_the_capsule_between_its_ends() {
        let (a, b, radius) = (Vec3::new(1.0, -0.5, 2.0), Vec3::new(-1.0, 1.5, 1.0), 0.4);
        let segment = ConvexShape::Segment { a, b, radius };
        // The same capsule, upright in its own frame.
        let half_height = 0.5 * (b - a).length();
        let capsule = ConvexShape::Capsule {
            half_height,
            radius,
        };
        let iso = Iso::new(
            (a + b) * 0.5,
            Quat::from_rotation_arc(Vec3::Y, (b - a).normalize_or_zero()),
        );
        let mut state = 0x2545_f491u32;
        let mut next = move || {
            // xorshift32, mapped to [-1, 1).
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        };
        let mut hits = 0;
        for _ in 0..200 {
            let origin = Vec3::new(3.0 * next(), 3.0 * next(), 2.0 + 2.5 * next());
            let aim = (a + b) * 0.5 + Vec3::new(next(), next(), next()) * 1.2;
            let dir = (aim - origin).normalize_or_zero();
            let world = raycast_convex(&segment, origin, dir, 20.0);
            let local = raycast_convex(
                &capsule,
                iso.inverse_point(origin),
                iso.inverse_vector(dir),
                20.0,
            );
            match (world, local) {
                (Some(w), Some(l)) => {
                    hits += 1;
                    assert!(close(w.t, l.t, 1e-4), "{w:?} {l:?}");
                    assert!(close_v(w.normal, iso.vector(l.normal), 1e-3), "{w:?} {l:?}");
                    let p = origin + dir * w.t;
                    let s = ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
                    assert!(w.t == 0.0 || close((p - (a + (b - a) * s)).length(), radius, 1e-4));
                }
                (None, None) => {}
                (w, l) => panic!("{w:?} vs {l:?}"),
            }
        }
        assert!(hits > 50, "{hits}");
    }

    #[test]
    fn rays_starting_inside_hit_at_zero_against_their_direction() {
        let hull = slanted_hull();
        let dir = Vec3::new(0.48, 0.6, 0.64);
        for shape in [
            ConvexShape::Sphere(1.0),
            ConvexShape::Capsule {
                half_height: 1.0,
                radius: 0.3,
            },
            ConvexShape::Box(Vec3::new(0.5, 1.0, 0.5)),
            ConvexShape::Hull(&hull),
            ConvexShape::Segment {
                a: Vec3::new(-1.0, 0.0, 0.0),
                b: Vec3::new(1.0, 0.2, 0.0),
                radius: 0.25,
            },
        ] {
            let hit = raycast_convex(
                &shape,
                shape.center() + Vec3::new(0.01, 0.02, -0.01),
                dir,
                5.0,
            );
            assert_eq!(
                hit,
                Some(RayHit {
                    t: 0.0,
                    normal: -dir,
                    feature: 0
                }),
                "{shape:?}"
            );
        }
        // On the surface counts as inside, whichever way the ray points.
        let s = ConvexShape::Sphere(1.0);
        assert_eq!(raycast_convex(&s, Vec3::X, Vec3::X, 5.0).unwrap().t, 0.0);
        let b = ConvexShape::Box(Vec3::ONE);
        assert_eq!(
            raycast_convex(&b, Vec3::X, -Vec3::X, 5.0).unwrap().normal,
            Vec3::X
        );
    }

    #[test]
    fn rays_against_meshes_and_heightfields_report_the_triangle() {
        let floor = TriangleMesh::new(
            vec![
                Vec3::new(-1.0, 0.0, -1.0),
                Vec3::new(1.0, 0.0, -1.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(-1.0, 0.0, 1.0),
            ],
            vec![[0, 2, 1], [0, 3, 2]],
        )
        .unwrap();
        let mesh = Shape::Mesh(Arc::new(floor));
        let hit = raycast(&mesh, Vec3::new(0.5, 2.0, -0.5), -Vec3::Y, 10.0).unwrap();
        assert!(close(hit.t, 2.0, 1e-6) && close_v(hit.normal, Vec3::Y, 1e-6));
        let other = raycast(&mesh, Vec3::new(-0.5, 2.0, 0.5), -Vec3::Y, 10.0).unwrap();
        assert_eq!((hit.feature, other.feature), (0, 1));
        // One-sided: from below the floor there is nothing.
        assert!(raycast(&mesh, Vec3::new(0.5, -2.0, -0.5), Vec3::Y, 10.0).is_none());
        assert!(raycast(&mesh, Vec3::new(0.5, 2.0, -0.5), -Vec3::Y, 1.5).is_none());
        let field = Heightfield::new(
            3,
            2,
            vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            Vec3::new(2.0, 1.0, 2.0),
        )
        .unwrap();
        let terrain = Shape::Heightfield(Arc::new(field.clone()));
        let hit = raycast(&terrain, Vec3::new(1.0, 5.0, 1.0), -Vec3::Y, 10.0).unwrap();
        assert!(close(hit.t, 4.5, 1e-5), "{hit:?}");
        assert!(close_v(
            hit.normal,
            field.triangle(hit.feature).normal(),
            1e-6
        ));
        let tri = field.triangle(hit.feature).vertices;
        assert!(
            crate::mesh::ray_triangle(Vec3::new(1.0, 5.0, 1.0), -Vec3::Y, &tri, 10.0, false)
                .is_some()
        );
        assert!(raycast(&terrain, Vec3::new(1.0, -5.0, 1.0), Vec3::Y, 10.0).is_none());
    }

    #[test]
    fn unusable_rays_miss_and_nothing_returns_nan() {
        let hull = slanted_hull();
        let shapes = [
            ConvexShape::Sphere(1.0),
            ConvexShape::Capsule {
                half_height: 1.0,
                radius: 0.3,
            },
            ConvexShape::Box(Vec3::ONE),
            ConvexShape::Hull(&hull),
            ConvexShape::Triangle([Vec3::ZERO, Vec3::X, Vec3::Z]),
            ConvexShape::Segment {
                a: Vec3::ZERO,
                b: Vec3::ZERO,
                radius: 0.5,
            },
        ];
        let far = Vec3::new(-5.0, 0.1, 0.2);
        for s in &shapes {
            assert!(raycast_convex(s, far, Vec3::X, -1.0).is_none());
            assert!(raycast_convex(s, far, Vec3::X, f32::NAN).is_none());
            assert!(raycast_convex(s, far, Vec3::ZERO, 10.0).is_none());
            assert!(raycast_convex(s, Vec3::splat(f32::NAN), Vec3::X, 10.0).is_none());
            assert!(raycast_convex(s, far, Vec3::new(f32::NAN, 0.0, 0.0), 10.0).is_none());
            // Every axis-aligned and diagonal direction from points around the shape.
            for i in 0..27 {
                let d = Vec3::new(
                    (i % 3) as f32 - 1.0,
                    (i / 3 % 3) as f32 - 1.0,
                    (i / 9) as f32 - 1.0,
                );
                for o in [
                    far,
                    Vec3::new(0.3, 3.0, -0.2),
                    Vec3::new(1.0, 1.0, 1.0),
                    Vec3::ZERO,
                ] {
                    if let Some(h) = raycast_convex(s, o, d.normalize_or_zero(), f32::INFINITY) {
                        assert!(
                            h.t.is_finite() && h.t >= 0.0 && h.normal.is_finite(),
                            "{s:?} {o:?} {d:?}: {h:?}"
                        );
                        assert!(
                            close(h.normal.length(), 1.0, 1e-5),
                            "{s:?} {o:?} {d:?}: {h:?}"
                        );
                    }
                }
            }
        }
    }
}
