//! Collision shapes, their bounds and their mass.
//!
//! Every shape is defined in its own local frame; a collider places it on a body with a local
//! [`Iso`]. Rounded shapes (sphere, capsule) are a *core* (a point, a segment) plus a radius, which
//! is how the distance and collision code treats them.

use std::sync::Arc;

use crate::hull::ConvexHull;
use crate::math::{Aabb, Iso, Mat3, Vec3};
use crate::mesh::{Heightfield, TriangleMesh};

/// A collision shape.
#[derive(Clone, Debug)]
pub enum Shape {
    /// A ball of `radius` about the origin.
    Sphere { radius: f32 },
    /// The points within `radius` of the segment from `(0, -half_height, 0)` to
    /// `(0, half_height, 0)`.
    Capsule { half_height: f32, radius: f32 },
    /// The box `[-half_extents, half_extents]`.
    Box { half_extents: Vec3 },
    /// A convex polyhedron.
    Hull(Arc<ConvexHull>),
    /// Triangles, for static level geometry: one-sided (front faces wind counter-clockwise) and
    /// never on a dynamic body.
    Mesh(Arc<TriangleMesh>),
    /// A grid of heights, for terrain: static only.
    Heightfield(Arc<Heightfield>),
}

/// Limits that keep a shape usable by the solver.
pub const MIN_EXTENT: f32 = 1e-4;
pub const MAX_EXTENT: f32 = 1e5;

impl Shape {
    /// Checks the shape's numbers (finite, positive, within [`MIN_EXTENT`]..[`MAX_EXTENT`]).
    pub fn validate(&self) -> Result<(), String> {
        let ok = |v: f32| v.is_finite() && (MIN_EXTENT..=MAX_EXTENT).contains(&v);
        match self {
            Shape::Sphere { radius } if !ok(*radius) => Err(format!("sphere radius {radius}")),
            Shape::Capsule {
                half_height,
                radius,
            } if !ok(*radius) || !(0.0..=MAX_EXTENT).contains(half_height) => Err(format!(
                "capsule half height {half_height}, radius {radius}"
            )),
            Shape::Box { half_extents }
                if !(ok(half_extents.x) && ok(half_extents.y) && ok(half_extents.z)) =>
            {
                Err(format!("box half extents {half_extents:?}"))
            }
            Shape::Hull(h) => h.validate(),
            _ => Ok(()),
        }
    }

    /// Whether the shape can be on a dynamic body.
    pub fn is_convex(&self) -> bool {
        !matches!(self, Shape::Mesh(_) | Shape::Heightfield(_))
    }

    /// The shape as a convex for the distance and cast code, or `None` for meshes and
    /// heightfields (whose triangles are convexes of their own: [`ConvexShape::Triangle`]).
    pub fn as_convex(&self) -> Option<ConvexShape<'_>> {
        Some(match self {
            Shape::Sphere { radius } => ConvexShape::Sphere(*radius),
            Shape::Capsule {
                half_height,
                radius,
            } => ConvexShape::Capsule {
                half_height: *half_height,
                radius: *radius,
            },
            Shape::Box { half_extents } => ConvexShape::Box(*half_extents),
            Shape::Hull(h) => ConvexShape::Hull(h),
            Shape::Mesh(_) | Shape::Heightfield(_) => return None,
        })
    }

    /// The world-space bounds of this shape placed at `iso`.
    pub fn aabb(&self, iso: &Iso) -> Aabb {
        match self {
            Shape::Sphere { radius } => Aabb::from_center_half(iso.position, Vec3::splat(*radius)),
            Shape::Capsule {
                half_height,
                radius,
            } => {
                let axis = iso.vector(Vec3::new(0.0, *half_height, 0.0));
                let (a, b) = (iso.position + axis, iso.position - axis);
                Aabb::new(a.min(b), a.max(b)).expand(*radius)
            }
            Shape::Box { half_extents } => {
                let m = iso.rotation.to_mat3();
                let h = Vec3::new(
                    m.row(0).abs().dot(*half_extents),
                    m.row(1).abs().dot(*half_extents),
                    m.row(2).abs().dot(*half_extents),
                );
                Aabb::from_center_half(iso.position, h)
            }
            Shape::Hull(h) => h
                .vertices
                .iter()
                .fold(Aabb::EMPTY, |b, &v| b.include(iso.point(v))),
            Shape::Mesh(m) => transform_aabb(&m.local_aabb(), iso),
            Shape::Heightfield(h) => transform_aabb(&h.local_aabb(), iso),
        }
    }

    /// Mass, centre of mass and inertia (about the centre of mass, in the shape's frame) for a
    /// uniform `density` in kg/m³. Meshes and heightfields have none: they are static only.
    pub fn mass_properties(&self, density: f32) -> MassProperties {
        use std::f32::consts::PI;
        match self {
            Shape::Sphere { radius: r } => {
                let mass = density * (4.0 / 3.0) * PI * r * r * r;
                MassProperties {
                    mass,
                    center: Vec3::ZERO,
                    inertia: Mat3::from_diagonal(Vec3::splat(0.4 * mass * r * r)),
                }
            }
            Shape::Capsule {
                half_height: h,
                radius: r,
            } => {
                // A cylinder of height 2h plus two hemispheres, each hemisphere's inertia moved to
                // the capsule's centre by the parallel-axis theorem.
                let (h, r) = (*h, *r);
                let cylinder = density * PI * r * r * 2.0 * h;
                let caps = density * (4.0 / 3.0) * PI * r * r * r;
                let mass = cylinder + caps;
                let axial = cylinder * r * r * 0.5 + caps * 0.4 * r * r;
                let cyl_side = cylinder * (3.0 * r * r + 4.0 * h * h) / 12.0;
                // Each hemisphere: 83/320 m r² about its own centre of mass, which sits 3r/8
                // from the flat face; the face is h from the capsule's centre.
                let hemi = caps * 0.5;
                let d = h + 0.375 * r;
                let cap_side = 2.0 * (hemi * (83.0 / 320.0) * r * r + hemi * d * d);
                let side = cyl_side + cap_side;
                MassProperties {
                    mass,
                    center: Vec3::ZERO,
                    inertia: Mat3::from_diagonal(Vec3::new(side, axial, side)),
                }
            }
            Shape::Box { half_extents: e } => {
                let mass = density * 8.0 * e.x * e.y * e.z;
                let (x2, y2, z2) = (4.0 * e.x * e.x, 4.0 * e.y * e.y, 4.0 * e.z * e.z);
                MassProperties {
                    mass,
                    center: Vec3::ZERO,
                    inertia: Mat3::from_diagonal(
                        Vec3::new(y2 + z2, x2 + z2, x2 + y2) * (mass / 12.0),
                    ),
                }
            }
            Shape::Hull(h) => h.mass_properties(density),
            Shape::Mesh(_) | Shape::Heightfield(_) => MassProperties::default(),
        }
    }

    /// The rounding radius (spheres and capsules), 0 for sharp shapes.
    pub fn radius(&self) -> f32 {
        match self {
            Shape::Sphere { radius } | Shape::Capsule { radius, .. } => *radius,
            _ => 0.0,
        }
    }

    /// How far the farthest point of this shape placed at `iso` is from `point`.
    pub fn farthest_from(&self, iso: &Iso, point: Vec3) -> f32 {
        let far = |corners: &mut dyn Iterator<Item = Vec3>| {
            corners
                .fold(0.0f32, |m, p| {
                    m.max((iso.point(p) - point).length_squared())
                })
                .sqrt()
        };
        let box_corners = |h: Vec3| {
            (0..8).map(move |i| {
                Vec3::new(
                    if i & 1 != 0 { h.x } else { -h.x },
                    if i & 2 != 0 { h.y } else { -h.y },
                    if i & 4 != 0 { h.z } else { -h.z },
                )
            })
        };
        let aabb_corners = |b: Aabb| box_corners(b.half_extents()).map(move |c| c + b.center());
        match self {
            Shape::Sphere { radius } => (iso.position - point).length() + radius,
            Shape::Capsule {
                half_height,
                radius,
            } => {
                let ends = [
                    Vec3::new(0.0, *half_height, 0.0),
                    Vec3::new(0.0, -half_height, 0.0),
                ];
                far(&mut ends.into_iter()) + radius
            }
            Shape::Box { half_extents } => far(&mut box_corners(*half_extents)),
            Shape::Hull(h) => far(&mut h.vertices.iter().copied()),
            Shape::Mesh(m) => far(&mut aabb_corners(m.local_aabb())),
            Shape::Heightfield(h) => far(&mut aabb_corners(h.local_aabb())),
        }
    }

    /// The radius of the largest ball about the centre of the shape's core
    /// ([`ConvexShape::center`]) that lies inside the shape; 0 for meshes and heightfields.
    #[inline]
    pub fn inner_radius(&self) -> f32 {
        match self {
            Shape::Sphere { radius } | Shape::Capsule { radius, .. } => *radius,
            Shape::Box { half_extents } => half_extents.min_element(),
            Shape::Hull(h) => h.inner_radius,
            Shape::Mesh(_) | Shape::Heightfield(_) => 0.0,
        }
    }
}

/// The bounds of a local box after a rigid transform.
pub fn transform_aabb(local: &Aabb, iso: &Iso) -> Aabb {
    if !local.is_valid() {
        return Aabb::EMPTY;
    }
    let m = iso.rotation.to_mat3();
    let c = iso.point(local.center());
    let e = local.half_extents();
    let h = Vec3::new(
        m.row(0).abs().dot(e),
        m.row(1).abs().dot(e),
        m.row(2).abs().dot(e),
    );
    Aabb::from_center_half(c, h)
}

/// A body's or a shape's mass distribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MassProperties {
    pub mass: f32,
    /// Centre of mass in the shape's (or body's) frame.
    pub center: Vec3,
    /// Inertia tensor about the centre of mass, in the same frame's axes.
    pub inertia: Mat3,
}

impl Default for MassProperties {
    fn default() -> Self {
        Self {
            mass: 0.0,
            center: Vec3::ZERO,
            inertia: Mat3::ZERO,
        }
    }
}

impl MassProperties {
    /// These properties seen from a frame in which the shape sits at `iso`.
    pub fn transformed(&self, iso: &Iso) -> MassProperties {
        MassProperties {
            mass: self.mass,
            center: iso.point(self.center),
            inertia: self.inertia.rotated(iso.rotation),
        }
    }

    /// The combination of several parts (each already in the common frame): masses add, the
    /// centre is the mass-weighted mean and each part's inertia moves to it by the parallel-axis
    /// theorem.
    pub fn combine(parts: &[MassProperties]) -> MassProperties {
        let mass: f32 = parts.iter().map(|p| p.mass).sum();
        if mass <= 0.0 {
            return MassProperties::default();
        }
        let center = parts.iter().fold(Vec3::ZERO, |c, p| c + p.center * p.mass) * (1.0 / mass);
        let inertia = parts.iter().fold(Mat3::ZERO, |acc, p| {
            let d = p.center - center;
            // I + m (|d|² 1 − d dᵀ)
            let shift = Mat3::IDENTITY.scale(d.length_squared()) - Mat3::outer(d, d);
            acc + p.inertia + shift.scale(p.mass)
        });
        MassProperties {
            mass,
            center,
            inertia,
        }
    }
}

/// A convex shape for the distance, cast and collision routines, in its own local frame.
#[derive(Clone, Copy, Debug)]
pub enum ConvexShape<'a> {
    Sphere(f32),
    Capsule {
        half_height: f32,
        radius: f32,
    },
    Box(Vec3),
    Hull(&'a ConvexHull),
    /// A triangle of a mesh or heightfield, vertices in the mesh's frame.
    Triangle([Vec3; 3]),
    /// A segment with a rounding radius (what a capsule's core is), for casts and queries.
    Segment {
        a: Vec3,
        b: Vec3,
        radius: f32,
    },
}

impl ConvexShape<'_> {
    /// The rounding radius: the shape is its core grown by this much.
    #[inline]
    pub fn radius(&self) -> f32 {
        match *self {
            ConvexShape::Sphere(r) => r,
            ConvexShape::Capsule { radius, .. } | ConvexShape::Segment { radius, .. } => radius,
            _ => 0.0,
        }
    }

    /// The farthest point of the *core* along local direction `d` (need not be unit length).
    #[inline]
    pub fn support_core(&self, d: Vec3) -> Vec3 {
        match *self {
            ConvexShape::Sphere(_) => Vec3::ZERO,
            ConvexShape::Capsule { half_height, .. } => Vec3::new(
                0.0,
                if d.y >= 0.0 {
                    half_height
                } else {
                    -half_height
                },
                0.0,
            ),
            ConvexShape::Box(h) => Vec3::new(
                if d.x >= 0.0 { h.x } else { -h.x },
                if d.y >= 0.0 { h.y } else { -h.y },
                if d.z >= 0.0 { h.z } else { -h.z },
            ),
            ConvexShape::Hull(hull) => hull.vertices[hull.support_index(d)],
            ConvexShape::Triangle(t) => {
                let (a, b, c) = (t[0].dot(d), t[1].dot(d), t[2].dot(d));
                if a >= b && a >= c {
                    t[0]
                } else if b >= c {
                    t[1]
                } else {
                    t[2]
                }
            }
            ConvexShape::Segment { a, b, .. } => {
                if a.dot(d) >= b.dot(d) {
                    a
                } else {
                    b
                }
            }
        }
    }

    /// A point inside the core (its centre), to start iterative searches from.
    #[inline]
    pub fn center(&self) -> Vec3 {
        match *self {
            ConvexShape::Hull(h) => h.centroid,
            ConvexShape::Triangle(t) => (t[0] + t[1] + t[2]) * (1.0 / 3.0),
            ConvexShape::Segment { a, b, .. } => (a + b) * 0.5,
            _ => Vec3::ZERO,
        }
    }
}

/// A convex shape placed in the world.
#[derive(Clone, Copy, Debug)]
pub struct Convex<'a> {
    pub shape: ConvexShape<'a>,
    pub iso: Iso,
}

impl<'a> Convex<'a> {
    #[inline]
    pub fn new(shape: ConvexShape<'a>, iso: Iso) -> Self {
        Self { shape, iso }
    }
    /// The core's farthest point along world direction `d`, in world space.
    #[inline]
    pub fn support_core(&self, d: Vec3) -> Vec3 {
        self.iso
            .point(self.shape.support_core(self.iso.inverse_vector(d)))
    }
    #[inline]
    pub fn radius(&self) -> f32 {
        self.shape.radius()
    }
    #[inline]
    pub fn center(&self) -> Vec3 {
        self.iso.point(self.shape.center())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Quat;

    #[test]
    fn primitive_masses_match_their_formulas() {
        let s = Shape::Sphere { radius: 2.0 }.mass_properties(10.0);
        let m = 10.0 * 4.0 / 3.0 * std::f32::consts::PI * 8.0;
        assert!((s.mass - m).abs() < 1e-2);
        assert!((s.inertia.cols[0].x - 0.4 * m * 4.0).abs() < 1e-1);
        let b = Shape::Box {
            half_extents: Vec3::new(1.0, 2.0, 3.0),
        }
        .mass_properties(1.0);
        assert!((b.mass - 48.0).abs() < 1e-4);
        assert!((b.inertia.cols[0].x - 48.0 * (16.0 + 36.0) / 12.0).abs() < 1e-3);
        // A capsule with no height is a sphere.
        let c = Shape::Capsule {
            half_height: 0.0,
            radius: 2.0,
        }
        .mass_properties(10.0);
        assert!((c.mass - m).abs() < 1e-2);
        assert!(
            (c.inertia.cols[0].x - 0.4 * m * 4.0).abs() < 1e-1,
            "{:?}",
            c.inertia
        );
        assert!((c.inertia.cols[1].y - 0.4 * m * 4.0).abs() < 1e-1);
    }

    #[test]
    fn combining_two_spheres_uses_the_parallel_axis_theorem() {
        let one = Shape::Sphere { radius: 1.0 }.mass_properties(1.0);
        let parts = [
            one.transformed(&Iso::from_position(Vec3::new(-2.0, 0.0, 0.0))),
            one.transformed(&Iso::from_position(Vec3::new(2.0, 0.0, 0.0))),
        ];
        let c = MassProperties::combine(&parts);
        assert!((c.mass - 2.0 * one.mass).abs() < 1e-5);
        assert!(c.center.length() < 1e-6);
        let i0 = one.inertia.cols[0].x;
        assert!((c.inertia.cols[0].x - 2.0 * i0).abs() < 1e-4);
        assert!((c.inertia.cols[1].y - 2.0 * (i0 + one.mass * 4.0)).abs() < 1e-3);
    }

    #[test]
    fn bounds_follow_rotation() {
        let b = Shape::Box {
            half_extents: Vec3::new(2.0, 1.0, 1.0),
        };
        let iso = Iso::new(
            Vec3::new(5.0, 0.0, 0.0),
            Quat::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_2),
        );
        let a = b.aabb(&iso);
        assert!((a.half_extents() - Vec3::new(1.0, 2.0, 1.0)).length() < 1e-5);
        assert!((a.center() - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5);
        let cap = Shape::Capsule {
            half_height: 1.0,
            radius: 0.5,
        };
        let a = cap.aabb(&Iso::IDENTITY);
        assert_eq!(
            a,
            Aabb::new(Vec3::new(-0.5, -1.5, -0.5), Vec3::new(0.5, 1.5, 0.5))
        );
    }

    #[test]
    fn the_farthest_point_of_a_placed_shape_is_a_corner_or_a_rounded_end() {
        let at = Iso::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_2),
        );
        let p = Vec3::new(1.0, 0.0, 3.0);
        let sphere = Shape::Sphere { radius: 0.5 };
        assert!((sphere.farthest_from(&at, p) - 2.5).abs() < 1e-5);
        // Laid along X by the quarter turn: its ends are at x = 0 and x = 2, y = 2.
        let capsule = Shape::Capsule {
            half_height: 1.0,
            radius: 0.25,
        };
        let want = (Vec3::new(1.0, 2.0, 0.0)).length() + 0.25;
        assert!((capsule.farthest_from(&at, p) - want).abs() < 1e-5);
        let b = Shape::Box {
            half_extents: Vec3::new(1.0, 0.5, 0.25),
        };
        // Turned, the box spans x 0.5..1.5, y 1..3, z 2.75..3.25 about (1, 2, 3).
        let want = Vec3::new(0.5, 3.0, 0.25).length();
        assert!((b.farthest_from(&at, p) - want).abs() < 1e-5);
        let hull = Shape::Hull(Arc::new(ConvexHull::cuboid(Vec3::new(1.0, 0.5, 0.25))));
        assert!((hull.farthest_from(&at, p) - want).abs() < 1e-5);
    }

    #[test]
    fn inner_radii_are_the_largest_balls_about_the_core_centre() {
        assert_eq!(Shape::Sphere { radius: 0.3 }.inner_radius(), 0.3);
        let capsule = Shape::Capsule {
            half_height: 2.0,
            radius: 0.25,
        };
        assert_eq!(capsule.inner_radius(), 0.25);
        let b = Shape::Box {
            half_extents: Vec3::new(1.0, 0.2, 3.0),
        };
        assert_eq!(b.inner_radius(), 0.2);
        let hull = Shape::Hull(Arc::new(ConvexHull::cuboid(Vec3::new(0.5, 2.0, 1.0))));
        assert!((hull.inner_radius() - 0.5).abs() < 1e-6);
        // A pyramid: its centroid (mean of the vertices) is 0.16 above the base, the nearest face
        // (the slanted ones are about 0.34 away).
        let v = vec![
            Vec3::new(-0.5, 0.0, -0.5),
            Vec3::new(0.5, 0.0, -0.5),
            Vec3::new(0.5, 0.0, 0.5),
            Vec3::new(-0.5, 0.0, 0.5),
            Vec3::new(0.0, 0.8, 0.0),
        ];
        let loops: [&[u32]; 5] = [
            &[0, 1, 2, 3],
            &[0, 4, 1],
            &[1, 4, 2],
            &[2, 4, 3],
            &[3, 4, 0],
        ];
        let pyramid = ConvexHull::from_faces(v, &loops).unwrap();
        let c = pyramid.centroid;
        let r = Shape::Hull(Arc::new(pyramid.clone())).inner_radius();
        assert!((r - c.y).abs() < 1e-6 && (c.y - 0.16).abs() < 1e-6, "{r}");
        for f in &pyramid.faces {
            assert!(f.offset - f.normal.dot(c) >= r - 1e-6);
        }
    }

    #[test]
    fn convex_supports_pick_the_farthest_core_point() {
        let b = ConvexShape::Box(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(
            b.support_core(Vec3::new(1.0, -1.0, 1.0)),
            Vec3::new(1.0, -2.0, 3.0)
        );
        let c = Convex::new(
            ConvexShape::Capsule {
                half_height: 1.0,
                radius: 0.2,
            },
            Iso::new(
                Vec3::new(0.0, 5.0, 0.0),
                Quat::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_2),
            ),
        );
        // A quarter turn about Z lays the capsule along X; its core ends are at x = ±1.
        assert!((c.support_core(Vec3::X) - Vec3::new(1.0, 5.0, 0.0)).length() < 1e-5);
        assert!((c.support_core(-Vec3::X) - Vec3::new(-1.0, 5.0, 0.0)).length() < 1e-5);
    }
}
