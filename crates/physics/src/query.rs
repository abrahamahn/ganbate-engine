//! Scene queries on a [`PhysicsWorld`]: rays, swept shapes and overlaps.
//!
//! All results are in world space and come back in a deterministic order (nearest first for
//! rays and casts, collider index order for overlaps).

use crate::collide::{self, Manifold};
use crate::gjk;
use crate::math::{Aabb, Iso, Vec3};
use crate::mesh::TriangleRef;
use crate::ray;
use crate::shape::{Convex, ConvexShape, Shape, transform_aabb};
use crate::small_buf::SmallBuf;
use crate::world::{BodyId, ColliderId, PhysicsWorld, QueryFilter};

/// A surface stops a cast only when the motion heads into it by more than this (the cosine of
/// the angle between the motion and the surface): closer to parallel is sliding along it,
/// within the rounding of a contact normal.
const ALONG: f32 = 1e-5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaycastHit {
    pub collider: ColliderId,
    pub body: BodyId,
    pub point: Vec3,
    /// Unit surface normal at the hit, facing the ray.
    pub normal: Vec3,
    /// Distance from the ray's origin.
    pub distance: f32,
    /// The triangle id for meshes and heightfields, 0 otherwise.
    pub feature: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeCastHit {
    pub collider: ColliderId,
    pub body: BodyId,
    /// Fraction of the translation travelled before touching, in `[0, 1]`.
    pub fraction: f32,
    /// Contact point on the hit collider's surface.
    pub point: Vec3,
    /// The hit surface's normal, facing the cast shape.
    pub normal: Vec3,
    /// The shape already overlapped this collider at its start.
    pub started_overlapping: bool,
}

impl PhysicsWorld {
    pub(crate) fn passes(&self, c: u32, filter: &QueryFilter) -> bool {
        let col = &self.colliders[c as usize];
        if !col.alive || (col.sensor && !filter.include_sensors) {
            return false;
        }
        if col.layers & filter.mask == 0 || filter.layers & col.mask == 0 {
            return false;
        }
        // A handle to a destroyed body excludes nothing, not whatever reuses its slot.
        !filter.exclude_body.is_some_and(|b| {
            b.index == col.body && b.generation == self.bodies[col.body as usize].generation
        })
    }

    /// The first collider hit by a ray from `origin` along `direction` (normalized here) within
    /// `max_distance`. A direction that is not finite, or too short to have one, hits nothing.
    pub fn raycast(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        filter: &QueryFilter,
    ) -> Option<RaycastHit> {
        if !direction.is_finite() {
            return None;
        }
        let dir = direction.try_normalize()?;
        if !origin.is_finite() || max_distance.is_nan() || max_distance < 0.0 {
            return None;
        }
        let mut best: Option<RaycastHit> = None;
        let mut limit = max_distance;
        for tree in [&self.static_tree, &self.moving_tree] {
            tree.raycast(origin, dir, limit, &mut |_, c, max_t| {
                if !self.passes(c, filter) {
                    return max_t;
                }
                let iso = self.collider_iso(c);
                let (o, d) = (iso.inverse_point(origin), iso.inverse_vector(dir));
                match ray::raycast(&self.colliders[c as usize].shape, o, d, max_t) {
                    Some(hit)
                        if best.is_none_or(|b| {
                            hit.t < b.distance || (hit.t == b.distance && c < b.collider.index)
                        }) =>
                    {
                        best = Some(RaycastHit {
                            collider: self.collider_id(c),
                            body: self.body_id(self.colliders[c as usize].body),
                            point: origin + dir * hit.t,
                            normal: iso.vector(hit.normal),
                            distance: hit.t,
                            feature: hit.feature,
                        });
                        limit = hit.t;
                        hit.t
                    }
                    _ => max_t,
                }
            });
        }
        best
    }

    /// Sweeps a convex `shape` placed at `iso` along `translation` and reports the first collider
    /// it would touch.
    ///
    /// A collider the shape already touches at the start (within the cast's tolerance of about
    /// 0.1 mm, or overlapping it by at most the world's `linear_slop`) is in the way only when
    /// the motion heads into it, against its surface normal there: one the shape slides along
    /// or moves away from is skipped (for a convex piece the gap can then only grow), so a shape
    /// resting on the ground can be cast along it or up, and finds what lies ahead. A shape
    /// deeper inside a collider reports it at fraction 0 whatever the motion
    /// (`started_overlapping`); a cast that does not move reports what it touches. On meshes and
    /// heightfields a hit on a seam the mesh marks inactive (a flat or concave crease, see
    /// [`crate::mesh`]) takes the face normal, as in the narrow phase, so a shape gliding over a
    /// tessellated floor is not stopped at its seams.
    ///
    /// Mesh and heightfield triangles are one-sided here as everywhere else (contacts, continuous
    /// collision, rays): a triangle whose plane the shape's centre starts behind is not in the
    /// way, so a cast up through a floor from below passes, as a ray and a body do.
    pub fn shape_cast(
        &self,
        shape: &Shape,
        iso: &Iso,
        translation: Vec3,
        filter: &QueryFilter,
    ) -> Option<ShapeCastHit> {
        let convex = shape.as_convex()?;
        let moving = Convex::new(convex, *iso);
        let swept = shape.aabb(iso).sweep(translation);
        let dir = translation.try_normalize();
        let touching = self.linear_slop();
        let mut best: Option<ShapeCastHit> = None;
        // A hit of `other` (collider `c`, or one of its triangles): whether it is in the way,
        // and the nearest so far.
        let consider = |c: u32,
                        other: &Convex,
                        tri: Option<(&TriangleRef, &Iso)>,
                        best: &mut Option<ShapeCastHit>| {
            let Some(mut hit) = gjk::cast(&moving, translation, other, 0.0) else {
                return;
            };
            // A triangle's face normal, and whether the hit is on an inactive seam, which answers
            // with the face normal; met later in the motion, and not against it, such a seam is
            // one the shape glides over.
            let face = tri.and_then(|(t, ciso)| {
                let n = ciso.vector(t.normal());
                (n != Vec3::ZERO).then_some((n, !seam_active(t, ciso.inverse_point(hit.point))))
            });
            if let Some((n, true)) = face {
                hit.normal = n;
                if hit.fraction > 0.0 && dir.is_some_and(|d| n.dot(d) > -ALONG) {
                    return;
                }
            }
            if hit.fraction == 0.0
                && let Some(d) = dir
            {
                let start = gjk::distance(&moving, other);
                // The surface normal there and how far apart they are (negative: overlapping).
                // Cores closer than GJK can resolve (a box set exactly on the ground) have no
                // direction: the narrow phase's contact gives it, or a triangle's face.
                let contact = if !start.overlap {
                    let n = match face {
                        Some((n, true)) => n,
                        _ => -start.normal,
                    };
                    Some((n, start.distance))
                } else if let Some((n, _)) = face {
                    // How far the shape's lowest point is above the triangle's plane.
                    let low = moving.support_core(-n).dot(n) - moving.radius();
                    Some((n, low - other.support_core(n).dot(n)))
                } else {
                    self.contact_normal(shape, iso, c, touching)
                };
                match contact {
                    Some((n, gap)) if gap >= -touching => {
                        if n.dot(d) > -ALONG {
                            return;
                        }
                        hit.normal = n;
                    }
                    _ => {}
                }
            }
            let better = best.is_none_or(|b| {
                hit.fraction < b.fraction || (hit.fraction == b.fraction && c < b.collider.index)
            });
            if better {
                *best = Some(ShapeCastHit {
                    collider: self.collider_id(c),
                    body: self.body_id(self.colliders[c as usize].body),
                    fraction: hit.fraction,
                    point: hit.point,
                    normal: hit.normal,
                    started_overlapping: hit.started_overlapping,
                });
            }
        };
        let mut candidates = SmallBuf::<u32, 64>::new(0);
        self.fill_candidates(&swept, filter, &mut candidates);
        for &c in candidates.as_slice() {
            let col = &self.colliders[c as usize];
            let ciso = self.collider_iso(c);
            let centre = ciso.inverse_point(moving.center());
            let mut triangle = |t: &TriangleRef| {
                if !behind(t, centre) {
                    let tri = Convex::new(ConvexShape::Triangle(t.vertices), ciso);
                    consider(c, &tri, Some((t, &ciso)), &mut best);
                }
                true
            };
            match &col.shape {
                Shape::Mesh(m) => {
                    m.query_aabb(&transform_aabb(&swept, &ciso.inverse()), &mut triangle)
                }
                Shape::Heightfield(h) => {
                    h.query_aabb(&transform_aabb(&swept, &ciso.inverse()), &mut triangle)
                }
                other => {
                    if let Some(os) = other.as_convex() {
                        consider(c, &Convex::new(os, ciso), None, &mut best);
                    }
                }
            }
        }
        best
    }

    /// The surface normal of convex collider `c` towards `shape` at `iso` and their separation
    /// (negative: overlap), from the narrow phase: the deepest manifold, with points up to
    /// `speculative` apart (GJK reads cores a hair apart as touching).
    fn contact_normal(
        &self,
        shape: &Shape,
        iso: &Iso,
        c: u32,
        speculative: f32,
    ) -> Option<(Vec3, f32)> {
        let mut manifolds: [collide::Manifold; collide::MAX_MESH_MANIFOLDS] = Default::default();
        let col = &self.colliders[c as usize];
        let len = collide::collide_stack(
            shape,
            iso,
            &col.shape,
            &self.collider_iso(c),
            speculative,
            &mut manifolds,
        );
        manifolds[..len]
            .iter()
            .filter(|m| m.count > 0)
            .map(|m| (-m.normal, m.min_separation()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// Calls `f` with every collider a convex `shape` at `iso` overlaps, in collider index order,
    /// until `f` returns false.
    ///
    /// Mesh and heightfield triangles are tested from either side. Touching a face, edge or
    /// vertex without penetrating it does not count as overlap.
    pub fn overlap(
        &self,
        shape: &Shape,
        iso: &Iso,
        filter: &QueryFilter,
        f: &mut dyn FnMut(ColliderId) -> bool,
    ) {
        if !shape.is_convex() {
            return;
        }
        let mut candidates = SmallBuf::<u32, 64>::new(0);
        self.fill_candidates(&shape.aabb(iso), filter, &mut candidates);
        for &c in candidates.as_slice() {
            let col = &self.colliders[c as usize];
            if collide::overlaps(shape, iso, &col.shape, &self.collider_iso(c))
                && !f(self.collider_id(c))
            {
                return;
            }
        }
    }

    fn fill_candidates<const N: usize>(
        &self,
        bounds: &Aabb,
        filter: &QueryFilter,
        candidates: &mut SmallBuf<u32, N>,
    ) {
        for tree in [&self.static_tree, &self.moving_tree] {
            tree.query(bounds, &mut |_, c| {
                if self.passes(c, filter) {
                    candidates.push(c);
                }
                true
            });
        }
        candidates.sort_dedup();
    }

    /// The contact manifolds a convex `shape` at `iso` would have with the world's colliders
    /// (normals point from the shape to the collider), points within `speculative`. This is the
    /// narrow phase, including its one-sided mesh and heightfield contacts. It can recover a
    /// centre up to `speculative` behind a triangle if there is no contact from the front.
    pub fn contacts_with(
        &self,
        shape: &Shape,
        iso: &Iso,
        speculative: f32,
        filter: &QueryFilter,
        f: &mut dyn FnMut(ColliderId, &Manifold),
    ) {
        if !shape.is_convex() {
            return;
        }
        let bounds: Aabb = shape.aabb(iso).expand(speculative);
        let mut candidates = SmallBuf::<u32, 64>::new(0);
        self.fill_candidates(&bounds, filter, &mut candidates);
        let mut manifolds: [Manifold; collide::MAX_MESH_MANIFOLDS] = Default::default();
        for &c in candidates.as_slice() {
            let ciso = self.collider_iso(c);
            let len = collide::collide_stack(
                shape,
                iso,
                &self.colliders[c as usize].shape,
                &ciso,
                speculative,
                &mut manifolds,
            );
            for m in &manifolds[..len] {
                if m.count > 0 {
                    f(self.collider_id(c), m);
                }
            }
        }
    }
}

/// Whether point `p` (in the mesh's frame) is behind triangle `t`: one-sided triangles stop no
/// shape whose centre is behind them (the narrow phase's rule, and the rays').
fn behind(t: &TriangleRef, p: Vec3) -> bool {
    t.normal().dot(p - t.vertices[0]) < 0.0
}

/// Whether point `p` (in the mesh's frame) of triangle `t` lies on an active edge or vertex,
/// where a contact may have a normal of its own: the nearest edge decides, and a vertex is
/// active when either of its edges is (the narrow phase's rule). Elsewhere a cast's normal
/// should be the face's.
pub(crate) fn seam_active(t: &TriangleRef, p: Vec3) -> bool {
    let v = t.vertices;
    let mut best = (f32::INFINITY, 0usize, 0.5f32);
    for i in 0..3 {
        let (a, b) = (v[i], v[(i + 1) % 3]);
        let ab = b - a;
        let l = ab.length_squared();
        if l <= 0.0 {
            continue;
        }
        let s = ((p - a).dot(ab) / l).clamp(0.0, 1.0);
        let d = (a + ab * s - p).length_squared();
        if d < best.0 {
            best = (d, i, s);
        }
    }
    let (_, i, s) = best;
    let bit = |k: usize| t.active_edges & (1 << k) != 0;
    if s <= 1e-4 {
        bit(i) || bit((i + 2) % 3)
    } else if s >= 1.0 - 1e-4 {
        bit(i) || bit((i + 1) % 3)
    } else {
        bit(i)
    }
}
