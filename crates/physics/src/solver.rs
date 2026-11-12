//! The constraint solver: soft step (Erin Catto, "Solver2D", 2024; Box2D v3) in 3D.
//!
//! A step of `dt` runs `n` substeps of `h = dt / n`. Each substep integrates velocities (gravity,
//! forces, damping), warm-starts every constraint with its accumulated impulse, solves the
//! constraints with soft position correction (a spring of `contact_hertz` with damping ratio
//! `contact_damping_ratio`, so stacks settle without jitter or explosive pushes), integrates
//! positions, then relaxes (solves again without the position bias, removing the velocity the
//! correction added). Restitution is applied once after the substeps. Contact points are found
//! once per step; their separation is updated each substep from how far the bodies have moved
//! (`dp`, `dq`), which is what makes substepping cheap and stable.
//!
//! Hot linear/angular velocities live in `SolverVel` (24 bytes per body) separate from cold
//! pose/inertia state in `SolverBody`, cutting Gauss-Seidel writeback cache traffic by 5.6x
//! while keeping identical arithmetic order.

use crate::math::{Mat3, Quat, Vec3};

/// Hot per-body linear and angular velocity mutated by every Gauss-Seidel constraint iteration.
/// Index 0 is the shared static slot (always zero).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct SolverVel {
    pub v: Vec3,
    pub w: Vec3,
}

impl SolverVel {
    pub const ZERO: SolverVel = SolverVel {
        v: Vec3::ZERO,
        w: Vec3::ZERO,
    };

    /// Velocity of the material point at offset `r` from the centre of mass.
    #[inline]
    pub fn point_velocity(&self, r: Vec3) -> Vec3 {
        self.v + self.w.cross(r)
    }
}

/// Cold per-body pose delta and inverse mass/inertia during a step (read-only during inner
/// constraint solves; mutated only once per substep in [`integrate_positions`]). Index 0 is the
/// shared static body (zero inverse mass, never moves): every static body and every body asleep
/// this step maps to it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SolverBody {
    /// How far the centre of mass has moved this step.
    pub dp: Vec3,
    /// How far the body has turned this step (world frame): `rotation = dq * rotation0`.
    pub dq: Quat,
    /// Inverse mass per world axis (zero on locked axes and for non-dynamic bodies).
    pub inv_mass: Vec3,
    /// World inverse inertia as the body is turned now (locked axes zeroed): the body's own at
    /// the start of the step, turned with it after each substep's positions. The joints, whose
    /// anchors turn with the bodies each substep, push with it, and torques turn the body by it:
    /// a very thin rod (2 m by 2 mm) hinged across itself, pushed through the inertia of the
    /// start of the step once it had swung a little, turned about its own axis as if that axis
    /// had not moved, and fell off its hinge.
    pub inv_inertia: Mat3,
    /// World inverse inertia at the start of the step (locked axes zeroed): the contacts'. Their
    /// anchors are those of the start of the step all through it (only the separation follows
    /// the turn), and an inertia turned under unturned lever arms coupled their pushes into a
    /// thin body's tiny inertia about its own axis: a 2 m by 2 mm rod tumbling on the ground was
    /// thrown into the air spinning at 10⁵ rad/s.
    pub contact_inv_inertia: Mat3,
}

impl SolverBody {
    pub const STATIC: SolverBody = SolverBody {
        dp: Vec3::ZERO,
        dq: Quat::IDENTITY,
        inv_mass: Vec3::ZERO,
        inv_inertia: Mat3::ZERO,
        contact_inv_inertia: Mat3::ZERO,
    };

    /// Applies impulse `p` at offset `r` from the centre of mass (as the body is turned now).
    #[inline]
    pub fn apply(&self, vel: &mut SolverVel, p: Vec3, r: Vec3) {
        vel.v += self.inv_mass.mul_elem(p);
        vel.w += self.inv_inertia.mul_vec(r.cross(p));
    }

    /// Applies a contact's impulse `p` at its anchor `r` (both of the start of the step).
    #[inline]
    fn apply_contact(&self, vel: &mut SolverVel, p: Vec3, r: Vec3) {
        vel.v += self.inv_mass.mul_elem(p);
        vel.w += self.contact_inv_inertia.mul_vec(r.cross(p));
    }

    /// `1 / (effective mass)` contribution of this body for a contact's impulse along `d` at
    /// `r`: `d · M⁻¹ d + (r × d) · I⁻¹ (r × d)`, the inertia of the start of the step.
    #[inline]
    pub fn inverse_mass_along(&self, r: Vec3, d: Vec3) -> f32 {
        let rd = r.cross(d);
        d.dot(self.inv_mass.mul_elem(d)) + rd.dot(self.contact_inv_inertia.mul_vec(rd))
    }
}

/// A body seen by constraint preparation: its pose at the start of the step and its solver slot.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // the joints read every field
pub(crate) struct BodyFrame {
    /// World centre of mass at the start of the step.
    pub center: Vec3,
    pub rotation: Quat,
    /// Centre of mass in the body's own frame.
    pub local_center: Vec3,
    /// Index into the solver bodies (0 for static or sleeping).
    pub index: usize,
}

/// Soft constraint coefficients for a substep (Catto's soft step).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Softness {
    pub bias_rate: f32,
    pub mass_scale: f32,
    pub impulse_scale: f32,
}

impl Softness {
    /// A rigid constraint: no bias, full mass.
    pub const RIGID: Softness = Softness {
        bias_rate: 0.0,
        mass_scale: 1.0,
        impulse_scale: 0.0,
    };

    /// A spring of `hertz` with damping ratio `zeta`, solved implicitly over a substep of `h`.
    pub fn new(hertz: f32, zeta: f32, h: f32) -> Softness {
        if hertz <= 0.0 {
            return Softness::RIGID;
        }
        let omega = 2.0 * std::f32::consts::PI * hertz;
        let a1 = 2.0 * zeta + h * omega;
        let a2 = h * omega * a1;
        let a3 = 1.0 / (1.0 + a2);
        Softness {
            bias_rate: omega / a1,
            mass_scale: a2 * a3,
            impulse_scale: a3,
        }
    }
}

/// What every constraint needs to know about the current step.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // the joints read every field
pub(crate) struct StepContext {
    pub dt: f32,
    pub h: f32,
    pub inv_h: f32,
    pub contact_softness: Softness,
    /// Contacts against a static (or kinematic) body are stiffer: nothing gives on the other side.
    pub static_softness: Softness,
    pub joint_softness: Softness,
    pub max_push_velocity: f32,
    pub restitution_threshold: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ConstraintPoint {
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    /// Separation at the start of the step minus `n · (anchor_b − anchor_a)`, so that the
    /// current separation is `n · (dp_b − dp_a + dq_b·anchor_b − dq_a·anchor_a) + base`.
    pub base_separation: f32,
    pub normal_impulse: f32,
    pub tangent_impulse: [f32; 2],
    pub max_normal_impulse: f32,
    pub normal_mass: f32,
    pub tangent_mass: [f32; 2],
    /// Normal relative velocity at the start of the step (negative when approaching).
    pub relative_velocity: f32,
}

/// One manifold as a solver constraint.
#[derive(Clone, Debug)]
pub(crate) struct ContactConstraint {
    pub a: usize,
    pub b: usize,
    pub normal: Vec3,
    pub tangents: [Vec3; 2],
    pub friction: f32,
    pub restitution: f32,
    pub softness: Softness,
    pub points: [ConstraintPoint; 4],
    pub count: usize,
    /// Where to write the impulses back: (contact index, manifold index).
    pub source: (u32, u32),
    /// The rounding radius of each collider (a sphere's or capsule's radius, else 0): the part
    /// of an anchor that does not turn with its body (see [`solve_contacts`]).
    pub rounding: (f32, f32),
}

/// Constant inputs for one moving solver body over a step.
pub(crate) struct BodyInput {
    pub force: Vec3,
    pub torque: Vec3,
    pub gravity_scale: f32,
    /// The damping multipliers 1/(1 + h·damping).
    pub linear_damping: f32,
    pub angular_damping: f32,
    /// A kinematic body's prescribed turn per substep. Dynamic bodies integrate their solved ω.
    pub kinematic_turn: Option<Quat>,
}

pub(crate) struct IntegrationInputs<'a> {
    pub gravity: Vec3,
    /// Inputs for solver body slots 1 onward; slot 0 is static.
    pub per_body: &'a [BodyInput],
    pub max_linear_speed: f32,
    pub max_angular_speed: f32,
}

pub(crate) fn integrate_velocities(
    bodies: &[SolverBody],
    vels: &mut [SolverVel],
    inputs: &IntegrationInputs,
    h: f32,
) {
    for ((b, vel), input) in bodies
        .iter()
        .skip(1)
        .zip(vels.iter_mut().skip(1))
        .zip(inputs.per_body)
    {
        if input.kinematic_turn.is_some() {
            continue;
        }
        let accel = b.inv_mass.mul_elem(input.force)
            + gravity_masked(inputs.gravity * input.gravity_scale, b.inv_mass);
        let v = (vel.v + accel * h) * input.linear_damping;
        let w = (vel.w + turn(&b.inv_inertia, input.torque) * h) * input.angular_damping;
        vel.v = clamp_length(v, inputs.max_linear_speed);
        vel.w = clamp_length(w, inputs.max_angular_speed);
    }
}

/// Shortens a velocity that exceeds the speed limit. Very large or small finite values need a
/// wider length comparison when squaring them in f32 would overflow or underflow.
#[inline]
fn clamp_length(v: Vec3, max: f32) -> Vec3 {
    let max = max.min(f32::MAX);
    let l2 = v.length_squared();
    let max2 = max * max;
    if l2.is_finite() && max2 >= f32::MIN_POSITIVE {
        return if l2 <= max2 { v } else { v * (max / l2.sqrt()) };
    }
    if v.is_finite() {
        let (x, y, z) = (f64::from(v.x), f64::from(v.y), f64::from(v.z));
        let length = (x * x + y * y + z * z).sqrt();
        if length <= f64::from(max) {
            return v;
        }
        let k = f64::from(max) / length;
        return Vec3::new((x * k) as f32, (y * k) as f32, (z * k) as f32);
    }
    // Infinite components outweigh the finite ones.
    let along = |x: f32| {
        if x == f32::INFINITY {
            1.0
        } else if x == f32::NEG_INFINITY {
            -1.0
        } else {
            0.0
        }
    };
    let u = Vec3::new(along(v.x), along(v.y), along(v.z));
    let length = u.length();
    if length > 0.0 {
        u * (max / length)
    } else {
        Vec3::ZERO
    }
}

/// `m · t` (an inverse inertia turning a torque), never NaN: where the sum of products
/// overflows (∞ − ∞), the torque is first scaled down to a largest component of 1, so the
/// result is at worst infinite along the right direction.
#[inline]
fn turn(m: &Mat3, t: Vec3) -> Vec3 {
    let r = m.mul_vec(t);
    if r.is_finite() {
        return r;
    }
    let s = t.abs().max_element();
    m.mul_vec(t * (1.0 / s)) * s
}

/// Gravity acts only on the axes a body may move along.
#[inline]
fn gravity_masked(g: Vec3, inv_mass: Vec3) -> Vec3 {
    Vec3::new(
        if inv_mass.x > 0.0 { g.x } else { 0.0 },
        if inv_mass.y > 0.0 { g.y } else { 0.0 },
        if inv_mass.z > 0.0 { g.z } else { 0.0 },
    )
}

/// Moves every non-static body by its velocity over one substep.
pub(crate) fn integrate_positions(
    bodies: &mut [SolverBody],
    vels: &[SolverVel],
    inputs: &IntegrationInputs,
    h: f32,
) {
    for ((b, vel), input) in bodies
        .iter_mut()
        .skip(1)
        .zip(vels.iter().skip(1))
        .zip(inputs.per_body)
    {
        b.dp += vel.v * h;
        b.dq = match input.kinematic_turn {
            Some(turn) => (turn * b.dq).normalize(),
            None => b.dq.integrate(vel.w, h),
        };
    }
}

/// Fills a constraint point's anchors, masses and restitution input.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_point(
    c: &ContactConstraint,
    bodies: &[SolverBody],
    vels: &[SolverVel],
    fa: &BodyFrame,
    fb: &BodyFrame,
    point: Vec3,
    separation: f32,
    normal_impulse: f32,
    friction_impulse: Vec3,
) -> ConstraintPoint {
    let (ba, bb) = (&bodies[c.a], &bodies[c.b]);
    let (va, vb) = (&vels[c.a], &vels[c.b]);
    let anchor_a = point - fa.center;
    let anchor_b = point - fb.center;
    let n = c.normal;
    let k = ba.inverse_mass_along(anchor_a, n) + bb.inverse_mass_along(anchor_b, n);
    let mass = |k: f32| if k > 0.0 { 1.0 / k } else { 0.0 };
    let tangent_mass = c
        .tangents
        .map(|t| mass(ba.inverse_mass_along(anchor_a, t) + bb.inverse_mass_along(anchor_b, t)));
    let dv = vb.point_velocity(anchor_b) - va.point_velocity(anchor_a);
    ConstraintPoint {
        anchor_a,
        anchor_b,
        base_separation: separation - n.dot(anchor_b - anchor_a),
        normal_impulse,
        tangent_impulse: [
            friction_impulse.dot(c.tangents[0]),
            friction_impulse.dot(c.tangents[1]),
        ],
        max_normal_impulse: 0.0,
        normal_mass: mass(k),
        tangent_mass,
        relative_velocity: n.dot(dv),
    }
}

pub(crate) fn warm_start_contacts(
    constraints: &[ContactConstraint],
    bodies: &[SolverBody],
    vels: &mut [SolverVel],
) {
    for c in constraints {
        let (ba, bb) = (&bodies[c.a], &bodies[c.b]);
        let (mut va, mut vb) = (vels[c.a], vels[c.b]);
        for p in &c.points[..c.count] {
            let impulse = c.normal * p.normal_impulse
                + c.tangents[0] * p.tangent_impulse[0]
                + c.tangents[1] * p.tangent_impulse[1];
            ba.apply_contact(&mut va, -impulse, p.anchor_a);
            bb.apply_contact(&mut vb, impulse, p.anchor_b);
        }
        write_back(vels, c.a, va, c.b, vb);
    }
}

/// Writes two solver velocities back, never touching the static slot.
#[inline]
fn write_back(vels: &mut [SolverVel], a: usize, va: SolverVel, b: usize, vb: SolverVel) {
    if a != 0 {
        vels[a] = va;
    }
    if b != 0 {
        vels[b] = vb;
    }
}

/// One pass over the contacts: non-penetration (with the soft position bias when `use_bias`),
/// then friction inside the cone `|f| ≤ μ·λn`.
///
/// Each point's separation is measured afresh from how far the bodies have moved and turned
/// this step. Only the core part of an anchor turns with its body: a rounded shape's touching
/// point stays a radius from its core along the normal, however the shape turns (turned along,
/// a ball's anchor read `r (1 − cos θ)` too far apart after a turn of `θ`, so rolling balls sank
/// and a ball meeting a wall at a slant went into it).
pub(crate) fn solve_contacts(
    constraints: &mut [ContactConstraint],
    bodies: &[SolverBody],
    vels: &mut [SolverVel],
    ctx: &StepContext,
    use_bias: bool,
    reverse: bool,
) {
    for c in constraints.iter_mut() {
        let (ba, bb) = (&bodies[c.a], &bodies[c.b]);
        let (mut va, mut vb) = (vels[c.a], vels[c.b]);
        let n = c.normal;
        let count = c.count;
        // A's surface is a radius along the normal from its core, B's a radius against it.
        let (round_a, round_b) = (n * c.rounding.0, n * c.rounding.1);
        for k in 0..2 * count {
            let i = if (k < count) != reverse {
                k % count
            } else {
                count - 1 - k % count
            };
            let p = &mut c.points[i];
            // Current separation from how far the bodies have moved this step.
            let ra = ba.dq.rotate(p.anchor_a - round_a) + round_a;
            let rb = bb.dq.rotate(p.anchor_b + round_b) - round_b;
            let d = (bb.dp - ba.dp) + (rb - ra);
            let s = n.dot(d) + p.base_separation;
            let (bias, mass_scale, impulse_scale) = if s > 0.0 {
                // Speculative: let them approach exactly until they touch.
                (s * ctx.inv_h, 1.0, 0.0)
            } else if use_bias {
                (
                    (c.softness.bias_rate * s).max(-ctx.max_push_velocity),
                    c.softness.mass_scale,
                    c.softness.impulse_scale,
                )
            } else {
                (0.0, 1.0, 0.0)
            };
            let vn = n.dot(vb.point_velocity(p.anchor_b) - va.point_velocity(p.anchor_a));
            let impulse =
                -p.normal_mass * mass_scale * (vn + bias) - impulse_scale * p.normal_impulse;
            let total = saturate(p.normal_impulse + impulse);
            let delta = total - p.normal_impulse;
            p.normal_impulse = total;
            p.max_normal_impulse = p.max_normal_impulse.max(delta);
            let impulse = n * delta;
            ba.apply_contact(&mut va, -impulse, p.anchor_a);
            bb.apply_contact(&mut vb, impulse, p.anchor_b);
        }
        for k in 0..count {
            let p = &mut c.points[if reverse { count - 1 - k } else { k }];
            let dv = vb.point_velocity(p.anchor_b) - va.point_velocity(p.anchor_a);
            let old = p.tangent_impulse;
            let mut new = [
                old[0] - p.tangent_mass[0] * dv.dot(c.tangents[0]),
                old[1] - p.tangent_mass[1] * dv.dot(c.tangents[1]),
            ];
            if !(new[0].is_finite() && new[1].is_finite()) {
                // Momentum past f32's range (only at absurd speeds): this pass leaves friction.
                new = old;
            }
            let limit = c.friction * p.normal_impulse;
            let len2 = new[0] * new[0] + new[1] * new[1];
            if len2 > limit * limit {
                let s = limit / len2.sqrt();
                new = [new[0] * s, new[1] * s];
            }
            p.tangent_impulse = new;
            let impulse = c.tangents[0] * (new[0] - old[0]) + c.tangents[1] * (new[1] - old[1]);
            ba.apply_contact(&mut va, -impulse, p.anchor_a);
            bb.apply_contact(&mut vb, impulse, p.anchor_b);
        }
        write_back(vels, c.a, va, c.b, vb);
    }
}

/// An accumulated normal impulse: never negative (contacts only push), never past the largest
/// number (an impulse that overflowed, or came out NaN, would turn into NaN velocities, also in
/// a kinematic body's slot as 0 × ∞). Exact for every impulse in range.
#[inline]
#[allow(clippy::manual_clamp)] // `clamp` would keep a NaN; `max` drops it.
fn saturate(total: f32) -> f32 {
    total.max(0.0).min(f32::MAX)
}

/// Bounces: where a point approached faster than the threshold and was pushed apart this step,
/// target a separating speed of `restitution × approach speed`.
pub(crate) fn apply_restitution(
    constraints: &mut [ContactConstraint],
    bodies: &[SolverBody],
    vels: &mut [SolverVel],
    ctx: &StepContext,
) {
    for c in constraints.iter_mut() {
        if c.restitution == 0.0 {
            continue;
        }
        let (ba, bb) = (&bodies[c.a], &bodies[c.b]);
        let (mut va, mut vb) = (vels[c.a], vels[c.b]);
        for p in c.points[..c.count].iter_mut() {
            if p.relative_velocity > -ctx.restitution_threshold || p.max_normal_impulse == 0.0 {
                continue;
            }
            let vn = c
                .normal
                .dot(vb.point_velocity(p.anchor_b) - va.point_velocity(p.anchor_a));
            let impulse = -p.normal_mass * (vn + c.restitution * p.relative_velocity);
            let total = saturate(p.normal_impulse + impulse);
            let delta = total - p.normal_impulse;
            p.normal_impulse = total;
            p.max_normal_impulse = p.max_normal_impulse.max(delta);
            let impulse = c.normal * delta;
            ba.apply_contact(&mut va, -impulse, p.anchor_a);
            bb.apply_contact(&mut vb, impulse, p.anchor_b);
        }
        write_back(vels, c.a, va, c.b, vb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A velocity with infinite components (a force too large for a light body) is clamped
    /// along them, never to NaN (review finding: clamp_length computed inf × (1/inf) = NaN).
    #[test]
    fn clamping_keeps_the_direction_of_infinite_and_huge_vectors() {
        let clamp = clamp_length;
        let inf = f32::INFINITY;
        assert_eq!(
            clamp(Vec3::new(inf, 5.0, 0.0), 400.0),
            Vec3::new(400.0, 0.0, 0.0)
        );
        let d = clamp(Vec3::new(-inf, 0.0, inf), 10.0);
        assert!((d - Vec3::new(-1.0, 0.0, 1.0) * (10.0 / 2f32.sqrt())).length() < 1e-5);
        let huge = clamp(Vec3::new(3e30, 4e30, 0.0), 10.0);
        assert!(
            (huge - Vec3::new(6.0, 8.0, 0.0)).length() < 1e-5,
            "{huge:?}"
        );
        // No top speed: the largest finite length.
        let top = clamp(Vec3::new(0.0, -inf, 0.0), inf);
        assert_eq!(top, Vec3::new(0.0, -f32::MAX, 0.0));
        assert_eq!(clamp(Vec3::new(f32::NAN, 0.0, 0.0), 10.0), Vec3::ZERO);
    }

    /// A huge torque turned by an inverse inertia with off-diagonal terms overflowed as ∞ − ∞.
    #[test]
    fn turning_a_huge_torque_never_gives_nan() {
        let m = Mat3::from_cols(
            Vec3::new(2.0, -1.0, 0.0),
            Vec3::new(-1.0, 2.0, 0.5),
            Vec3::new(0.0, 0.5, 3.0),
        );
        let t = Vec3::new(3e38, 3e38, -3e38);
        assert!(m.mul_vec(t).x.is_nan() || !m.mul_vec(t).is_finite());
        let r = turn(&m, t);
        assert!(!r.x.is_nan() && !r.y.is_nan() && !r.z.is_nan(), "{r:?}");
        // The direction is that of m · t.
        let expected = m.mul_vec(t * (1.0 / 3e38));
        assert_eq!(r.x.signum(), expected.x.signum());
        assert_eq!(r.z.signum(), expected.z.signum());
        // In range, exactly the product.
        let small = Vec3::new(1.0, -2.0, 0.5);
        assert_eq!(turn(&m, small), m.mul_vec(small));
    }

    #[test]
    fn accumulated_normal_impulses_stay_numbers() {
        assert_eq!(saturate(f32::INFINITY), f32::MAX);
        assert_eq!(saturate(f32::NAN), 0.0);
        assert_eq!(saturate(-1.0), 0.0);
        assert_eq!(saturate(2.5), 2.5);
    }

    #[test]
    fn softness_goes_rigid_at_zero_hertz_and_stays_bounded() {
        assert_eq!(Softness::new(0.0, 1.0, 1.0 / 240.0), Softness::RIGID);
        let s = Softness::new(30.0, 10.0, 1.0 / 240.0);
        assert!(s.bias_rate > 0.0 && s.mass_scale > 0.0 && s.mass_scale < 1.0);
        assert!((s.mass_scale + s.impulse_scale - 1.0).abs() < 1e-6, "{s:?}");
    }

    /// There is no allowed overlap: a point overlapping by 1 mm (less than the world's default
    /// `linear_slop` of 5 mm) is pushed apart by the soft pass, and the relax pass takes the push
    /// back out of the velocity (Settings::linear_slop documents this; review finding
    /// LINEAR-SLOP-UNUSED).
    #[test]
    fn any_overlap_is_pushed_apart_softly() {
        let h = 1.0 / 240.0;
        let ctx = StepContext {
            dt: 4.0 * h,
            h,
            inv_h: 1.0 / h,
            contact_softness: Softness::new(30.0, 10.0, h),
            static_softness: Softness::new(60.0, 10.0, h),
            joint_softness: Softness::RIGID,
            max_push_velocity: 3.0,
            restitution_threshold: 1.0,
        };
        let bodies = vec![
            SolverBody::STATIC,
            SolverBody {
                inv_mass: Vec3::ONE,
                inv_inertia: Mat3::IDENTITY,
                contact_inv_inertia: Mat3::IDENTITY,
                ..SolverBody::STATIC
            },
        ];
        let mut vels = vec![SolverVel::ZERO, SolverVel::ZERO];
        let (t1, t2) = Vec3::Y.orthonormal_basis();
        let mut c = vec![ContactConstraint {
            a: 0,
            b: 1,
            normal: Vec3::Y,
            tangents: [t1, t2],
            friction: 0.5,
            restitution: 0.0,
            softness: ctx.static_softness,
            points: [ConstraintPoint::default(); 4],
            count: 1,
            source: (0, 0),
            rounding: (0.0, 0.0),
        }];
        let fa = BodyFrame {
            center: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            local_center: Vec3::ZERO,
            index: 0,
        };
        let fb = BodyFrame {
            center: Vec3::new(0.0, 0.499, 0.0),
            ..fa
        };
        c[0].points[0] = prepare_point(
            &c[0],
            &bodies,
            &vels,
            &fa,
            &fb,
            Vec3::ZERO,
            -0.001,
            0.0,
            Vec3::ZERO,
        );
        solve_contacts(&mut c, &bodies, &mut vels, &ctx, true, false);
        let push = vels[1].v.y;
        let expected = ctx.static_softness.bias_rate * 0.001 * ctx.static_softness.mass_scale;
        assert!(
            push > 0.0 && (push - expected).abs() < 1e-3 * expected,
            "pushed apart at {push} m/s, expected {expected}"
        );
        solve_contacts(&mut c, &bodies, &mut vels, &ctx, false, true);
        assert!(
            vels[1].v.y.abs() < 1e-6,
            "the relax pass leaves no push velocity: {:?}",
            vels[1].v
        );
    }

    #[test]
    fn a_resting_point_stops_a_falling_body_without_bouncing() {
        // One dynamic unit mass resting on the static slot through a single contact point.
        let bodies = vec![
            SolverBody::STATIC,
            SolverBody {
                inv_mass: Vec3::ONE,
                inv_inertia: Mat3::IDENTITY,
                contact_inv_inertia: Mat3::IDENTITY,
                ..SolverBody::STATIC
            },
        ];
        let mut vels = vec![
            SolverVel::ZERO,
            SolverVel {
                v: Vec3::new(0.0, -2.0, 0.0),
                w: Vec3::ZERO,
            },
        ];
        let ctx = StepContext {
            dt: 1.0 / 60.0,
            h: 1.0 / 240.0,
            inv_h: 240.0,
            contact_softness: Softness::new(30.0, 10.0, 1.0 / 240.0),
            static_softness: Softness::new(60.0, 10.0, 1.0 / 240.0),
            joint_softness: Softness::RIGID,
            max_push_velocity: 3.0,
            restitution_threshold: 1.0,
        };
        let (t1, t2) = Vec3::Y.orthonormal_basis();
        let mut c = vec![ContactConstraint {
            a: 0,
            b: 1,
            normal: Vec3::Y,
            tangents: [t1, t2],
            friction: 0.5,
            restitution: 0.0,
            softness: ctx.static_softness,
            points: [ConstraintPoint::default(); 4],
            count: 1,
            source: (0, 0),
            rounding: (0.0, 0.0),
        }];
        let fa = BodyFrame {
            center: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            local_center: Vec3::ZERO,
            index: 0,
        };
        let fb = BodyFrame {
            center: Vec3::new(0.0, 0.5, 0.0),
            ..fa
        };
        c[0].points[0] = prepare_point(
            &c[0],
            &bodies,
            &vels,
            &fa,
            &fb,
            Vec3::ZERO,
            0.0,
            0.0,
            Vec3::ZERO,
        );
        // The soft pass takes out most of the approach; the relax pass the rest.
        solve_contacts(&mut c, &bodies, &mut vels, &ctx, true, false);
        assert!(vels[1].v.y < 0.0 && vels[1].v.y > -0.2, "{:?}", vels[1].v);
        solve_contacts(&mut c, &bodies, &mut vels, &ctx, false, true);
        assert!(
            vels[1].v.y.abs() < 1e-4,
            "the approach is cancelled: {:?}",
            vels[1].v
        );
        assert!((c[0].points[0].normal_impulse - 2.0).abs() < 1e-3);
        assert_eq!(vels[0].v, Vec3::ZERO, "the static slot never moves");
    }
}
