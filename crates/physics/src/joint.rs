//! Joints: constraints between two bodies, solved with the same soft step as contacts.
//!
//! Each body gets a joint frame (`frame_a`, `frame_b`), given in the body's own frame (relative
//! to the body origin, not its centre of mass). The joint keeps the two frames together as its
//! kind says. The joint axis of hinges and sliders is the frame's local X axis; hinge angles are
//! the rotation of frame B relative to frame A about that axis, slider positions the offset of
//! frame B's origin from frame A's along it.
//!
//! # Solving
//!
//! Joints are Box2D v3's (Erin Catto's soft step) in three dimensions. `prepare` takes the anchors
//! (the frames' origins) relative to the centres of mass and the frames' rotations, in world
//! space, at the start of the step. Every substep turns them by how far each body has turned since
//! (`SolverBody::dq`) and adds how far the centres have moved (`SolverBody::dp`), so the position
//! error is measured afresh without touching the bodies' poses, as contacts do. `warm_start`
//! applies the impulses of the substep before; `solve` then runs once with a soft position
//! correction (`StepContext::joint_softness`) and once rigidly without it (relax). The rows of
//! each kind, in solving order:
//!
//! - Ball: the point, three linear rows solved together through the 3×3 effective mass.
//! - Hinge: spring, motor, lower and upper limit about frame A's X axis; two angular rows keeping
//!   frame B's X axis on frame A's; the point.
//! - Slider: spring, motor, lower and upper limit along frame A's X axis; three angular rows
//!   locking the rotation; two linear rows keeping frame B's origin on frame A's X axis.
//! - Fixed: three angular rows; the point.
//! - Distance: one row along the line between the anchors, rigid when `min == max`; otherwise the
//!   spring (if any) and one-sided rows at `min` and `max`.
//!
//! Springs and motors act in both passes, as in Box2D v3. A spring is a real force, so the relax
//! pass applies it again at the integrated position; that is what keeps stiff springs stable (the
//! soft pass of a stiff spring is a full position correction, which the relax pass takes back as
//! it does for the rigid rows). The price is numerical damping: two implicit passes a substep
//! damp a spring as if its damping ratio were higher by about `h·ω` (0.05 for a 2 Hz spring at the
//! default 4 substeps of a 60 Hz step; see [`Spring`]).
//!
//! Limits are one-sided rows: while the bodies are short of a limit they may approach it exactly
//! until they meet it (speculative, as contacts), once past it they are pushed back softly, and
//! while relaxing rigidly. A hinge's angle reads within `(-π, π]`, so its limits read it unwound
//! around the middle of their range, and a limited hinge also stops at `±π` (see
//! [`HingeOptions::limit`]).
//!
//! Locked axes need nothing of their own: the solver bodies' inverse masses and inertias are zero
//! there, so no impulse moves a body along them. A direction neither body can move in at all is
//! left out of the solve instead of answered with a huge impulse: a single row whose effective
//! mass is at or below a millionth of a bound of the magnitudes it was summed from, or a pivot of
//! a block of rows at or below a millionth of the block's trace or a hundred-millionth of the
//! magnitudes the block was summed from. The second bound matters where every row of a block is
//! such a direction (a 2.5D body's hinge axis rows): the block and its trace are then only
//! rounding. Everything uses only `+ - * /`, `sqrt`, comparisons and [`crate::math::atan2`]
//! (through [`Quat::twist_angle`]), in a fixed order, so joints are as deterministic as the rest.
//!
//! Numbers `f32` cannot carry are refused rather than solved: a joint whose frames lie farther
//! than [`MAX_EXTENT`] from their bodies (or are not finite), or a distance joint whose length is
//! beyond it, does nothing; a limit beyond it is no limit; and a joint whose pass would leave a
//! body with a velocity that is not a number, or so large that its square is not (an error
//! measured across 1e30 m, say), writes nothing and drops out for the rest of the step, its
//! impulses back to zero. Accumulated impulses saturate at half the largest `f32`, as contacts'
//! normal impulses do: a body whose inertia is near the largest `f32` (colliders thousands of
//! kilometres apart) asks a row for more than `f32` holds whenever it turns a few radians a
//! second, and then gets the most the row gives each pass instead of a joint that drops out.

use std::f32::consts::{PI, TAU};

use crate::math::{Iso, Mat3, Quat, Vec3};
use crate::shape::MAX_EXTENT;
use crate::solver::{BodyFrame, Softness, SolverBody, SolverVel, StepContext};
use crate::world::BodyId;

/// Drives a hinge (angular speed, rad/s, with a torque limit, N·m) or a slider (speed, m/s, with
/// a force limit, N).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Motor {
    pub speed: f32,
    pub max_force: f32,
}

/// A spring towards a rest value: frequency in hertz and damping ratio (1 = critical).
///
/// The step itself damps springs a little more than the ratio says. A spring acts in both of a
/// substep's passes (see the module notes), and each pass is implicit, so a spring of frequency
/// `ω = 2π·hertz` loses energy as if its damping ratio were higher by about `h·ω`, where `h` is
/// the substep (`dt / substeps`); exactly, an undamped one decays at a ratio of
/// `ln(1 + (h·ω)²) / (h·ω)`. With the default 4 substeps of a 60 Hz step that is about 0.013 at
/// 0.5 Hz, 0.05 at 2 Hz and 0.2 at 8 Hz: an undamped 2 Hz spring keeps half its swing for about
/// a second. Lightly damped springs that must ring longer need more substeps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spring {
    pub hertz: f32,
    pub damping_ratio: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HingeOptions {
    /// Lower and upper angle in radians, within `[-π, π]` (angles beyond are taken as `±π`; a NaN
    /// side is `±π` too). The angle reads within `(-π, π]`, so a limited hinge cannot turn past
    /// `±π` either: past `π` would read as coming up from `-π`. A range spanning (nearly) the
    /// whole turn, such as `(-π, π)`, is narrowed to leave a gap of 0.1 rad at `±π`, where the
    /// hinge then stops; a hinge that must turn freely has no limit.
    pub limit: Option<(f32, f32)>,
    pub motor: Option<Motor>,
    /// A spring towards angle 0.
    pub spring: Option<Spring>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SliderOptions {
    /// Lower and upper position in metres.
    pub limit: Option<(f32, f32)>,
    pub motor: Option<Motor>,
    /// A spring towards position 0.
    pub spring: Option<Spring>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceOptions {
    /// Rest length in metres.
    pub length: f32,
    /// Allowed range; `min == max == length` makes a rigid rod. Without a spring the length is
    /// free within the range (a rope when `min` is 0).
    pub min: f32,
    pub max: f32,
    /// Makes the rest length soft (a spring) instead of rigid; `min`/`max` stay hard.
    pub spring: Option<Spring>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JointKind {
    /// The frames' origins stay together; any rotation is free.
    Ball,
    /// Origins together, X axes aligned; rotation about X only.
    Hinge(HingeOptions),
    /// Orientations locked, origins on frame A's X axis; translation along X only.
    Slider(SliderOptions),
    /// The frames stay together completely.
    Fixed,
    /// The frames' origins keep a distance.
    Distance(DistanceOptions),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointDesc {
    pub body_a: BodyId,
    pub body_b: BodyId,
    pub frame_a: Iso,
    pub frame_b: Iso,
    pub kind: JointKind,
    /// Whether the two bodies still collide with each other.
    pub collide_connected: bool,
    pub user_data: u64,
}

/// A single row's effective mass at or below this fraction of its scale (a bound of what rounding
/// of the rotated inertias can produce), or a pivot of a block of rows at or below this fraction
/// of the block's trace (what cancellation in the factorization leaves), is a direction neither
/// body can move in: locked axes, static bodies, or `f32` rounding of one of those. Its row is
/// left out of the solve. See also [`ROUNDING_EPS`].
pub(crate) const PIVOT_EPS: f32 = 1e-6;

/// A pivot of a block of rows at or below this fraction of the magnitudes the block was summed
/// from (the bodies' inverse masses, and `|r|²` times the trace of their inverse inertias) is a
/// direction neither body can move in, whatever the block's own trace: when every row of the
/// block is such a direction (a 2.5D body's hinge axis rows, say), the entries and the trace are
/// both rounding, and [`PIVOT_EPS`] of that trace would let the rounding through. Rounding of a
/// row a hair off a locked axis leaves about 1e-14 of those magnitudes; a body whose orientation
/// is off its one free axis by `t` radians leaves about `t²`, so this tolerates up to 1e-4 rad of
/// that. Real directions keep far more: the thinnest of a 2 m × 1 mm rod keeps 5e-7.
const ROUNDING_EPS: f32 = 1e-8;

/// The least gap, in radians, that a hinge's limits leave between them at `±π` (see
/// [`Axial::new`]). A joint that overshoots one end by more than half the gap (where the reading
/// wraps) reads as past the other end and goes round once more; at the default step that takes
/// more than about 15 rad/s.
const ANGLE_LIMIT_GAP: f32 = 0.1;

/// Hinge and slider springs stiffer than this act as rigid as the step allows anyway; the bound
/// keeps the soft coefficients finite.
const MAX_SPRING_HERTZ: f32 = 1.0e4;
const MAX_DAMPING_RATIO: f32 = 1.0e6;

/// Accumulated impulses kept from step to step for warm starting. Vectors are world-frame
/// impulses on body B; body A receives the opposite.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct JointImpulses {
    /// The point rows (ball, hinge, fixed), or the slider's two rows across its axis.
    pub linear: Vec3,
    /// The angular rows: the hinge's two keeping the axes together, or the three locking the
    /// rotation (slider, fixed).
    pub angular: Vec3,
    /// The spring (hinge, slider, distance), or a rigid distance's row.
    pub axial: f32,
    pub lower: f32,
    pub upper: f32,
    pub motor: f32,
}

/// A joint prepared for one step.
#[derive(Clone, Debug)]
pub(crate) struct JointConstraint {
    pub a: usize,
    pub b: usize,
    pub impulses: JointImpulses,
    rows: Rows,
    /// The anchors (frame origins) relative to each body's centre of mass, world frame, at the
    /// start of the step.
    anchor_a: Vec3,
    anchor_b: Vec3,
    /// Centre of mass of B minus that of A at the start of the step.
    delta_center: Vec3,
    /// The joint frames' world rotations at the start of the step.
    frame_a: Quat,
    frame_b: Quat,
    /// Accumulated impulses of the two rows across the axis (the hinge's angular ones or the
    /// slider's linear ones), along frame A's Y and Z axes, which turn with body A.
    cross: [f32; 2],
}

/// What a joint solves, with its options checked once per step.
#[derive(Clone, Copy, Debug)]
enum Rows {
    /// Nothing: neither body can be moved by it, or its numbers are unusable.
    Inactive,
    Ball,
    Hinge(Axial),
    Slider(Axial),
    Fixed,
    Distance(Distance),
}

/// The optional rows along a hinge's or slider's axis.
#[derive(Clone, Copy, Debug)]
struct Axial {
    lower: Option<f32>,
    upper: Option<f32>,
    motor: Option<Motor>,
    spring: Option<Softness>,
}

#[derive(Clone, Copy, Debug)]
struct Distance {
    length: f32,
    min: f32,
    max: f32,
    /// One bilateral row at `length` when `min == max`; otherwise the spring (when it has a
    /// stiffness) with one-sided rows at `min` (when above zero) and `max` (when finite).
    rigid: bool,
    spring: Option<Softness>,
}

impl Rows {
    fn new(desc: &JointDesc, h: f32) -> Rows {
        let usable = |f: &Iso| {
            within_extent(f.position.x)
                && within_extent(f.position.y)
                && within_extent(f.position.z)
                && f.rotation.is_finite()
        };
        if !(usable(&desc.frame_a) && usable(&desc.frame_b)) {
            return Rows::Inactive;
        }
        match desc.kind {
            JointKind::Ball => Rows::Ball,
            JointKind::Fixed => Rows::Fixed,
            JointKind::Hinge(o) => Rows::Hinge(Axial::new(o.limit, o.motor, o.spring, h, true)),
            JointKind::Slider(o) => Rows::Slider(Axial::new(o.limit, o.motor, o.spring, h, false)),
            JointKind::Distance(o) => Distance::new(&o, h).map_or(Rows::Inactive, Rows::Distance),
        }
    }
}

/// A spring's soft coefficients for substeps of `h`; none without a usable stiffness.
fn spring_softness(spring: Option<Spring>, h: f32) -> Option<Softness> {
    let s = spring?;
    if !(s.hertz > 0.0 && s.hertz.is_finite()) {
        return None;
    }
    let zeta = clamp(s.damping_ratio, 0.0, MAX_DAMPING_RATIO);
    Some(Softness::new(
        clamp(s.hertz, 0.0, MAX_SPRING_HERTZ),
        zeta,
        h,
    ))
}

impl Axial {
    fn new(
        limit: Option<(f32, f32)>,
        motor: Option<Motor>,
        spring: Option<Spring>,
        h: f32,
        angular: bool,
    ) -> Axial {
        // A NaN side is no limit (for an angle, `±π`: below); the sides are put in order; an
        // angle is kept within [-π, π] and a position beyond `MAX_EXTENT` (or infinite) means no
        // limit on that side.
        let (mut lower, mut upper) = (None, None);
        if let Some((l, u)) = limit {
            let side = |v: f32| (!v.is_nan()).then_some(v);
            (lower, upper) = match (side(l), side(u)) {
                (Some(l), Some(u)) if l > u => (Some(u), Some(l)),
                sides => sides,
            };
            let keep = |v: f32| {
                if angular {
                    Some(clamp(v, -PI, PI))
                } else {
                    within_extent(v).then_some(v)
                }
            };
            (lower, upper) = (lower.and_then(keep), upper.and_then(keep));
        }
        // An angle reads within (-π, π], so turning past π reads as coming up from -π: a limited
        // hinge stops at ±π whatever its limits say, and a missing side is that end. The limits
        // read the angle unwound around the middle of their range (`solve_hinge`), which tells an
        // overshoot of one end from a reading past the other only while the ends leave a gap
        // between them at ±π: a range spanning (nearly) the whole turn is narrowed to leave
        // `ANGLE_LIMIT_GAP`.
        if angular && (lower.is_some() || upper.is_some()) {
            let (mut l, mut u) = (lower.unwrap_or(-PI), upper.unwrap_or(PI));
            let gap = (l + TAU) - u;
            if gap < ANGLE_LIMIT_GAP {
                let pull = 0.5 * (ANGLE_LIMIT_GAP - gap);
                l += pull;
                u -= pull;
            }
            (lower, upper) = (Some(l), Some(u));
        }
        let motor = motor.filter(|m| m.speed.is_finite()).map(|m| Motor {
            speed: m.speed,
            max_force: clamp(m.max_force, 0.0, f32::INFINITY),
        });
        Axial {
            lower,
            upper,
            motor,
            spring: spring_softness(spring, h),
        }
    }

    fn any(&self) -> bool {
        self.lower.is_some()
            || self.upper.is_some()
            || self.motor.is_some()
            || self.spring.is_some()
    }

    /// Keeps the warm-start impulses of the rows this axis has (a motor's within its current
    /// limit for substeps of `h`).
    fn carry(&self, warm: &JointImpulses, out: &mut JointImpulses, h: f32) {
        if self.spring.is_some() {
            out.axial = saturate(warm.axial);
        }
        if let Some(m) = self.motor {
            let max = saturate(m.max_force * h);
            out.motor = clamp(saturate(warm.motor), -max, max);
        }
        if self.lower.is_some() {
            out.lower = saturate(warm.lower);
        }
        if self.upper.is_some() {
            out.upper = saturate(warm.upper);
        }
    }
}

impl Distance {
    fn new(o: &DistanceOptions, h: f32) -> Option<Distance> {
        // Written so that NaN falls to the safe side of every comparison. A `max` beyond
        // `MAX_EXTENT` is no maximum; a `min` or `length` beyond it is unusable.
        let min = if o.min > 0.0 { o.min } else { 0.0 };
        let max = if o.max >= min { o.max } else { min };
        let max = if max <= MAX_EXTENT {
            max
        } else {
            f32::INFINITY
        };
        let length = if o.length > max {
            max
        } else if o.length > min {
            o.length
        } else {
            min
        };
        if !(min <= MAX_EXTENT && length <= MAX_EXTENT) {
            return None;
        }
        let rigid = min >= max;
        Some(Distance {
            length,
            min,
            max,
            rigid,
            spring: if rigid {
                None
            } else {
                spring_softness(o.spring, h)
            },
        })
    }
}

/// The most a joint row accumulates either way: half the largest `f32`, so that the change from
/// one accumulated impulse to another is a number too. Only a body whose inertia is near the
/// largest `f32` (colliders thousands of kilometres apart) asks a row for more; there the row
/// applies this much each pass, instead of an impulse that is not a number.
const MAX_IMPULSE: f32 = 0.5 * f32::MAX;

/// An accumulated impulse kept a number: within `±MAX_IMPULSE`, NaN giving zero. The same bits
/// for every impulse in range.
#[inline]
fn saturate(x: f32) -> f32 {
    if x > MAX_IMPULSE {
        MAX_IMPULSE
    } else if x < -MAX_IMPULSE {
        -MAX_IMPULSE
    } else if x.is_nan() {
        0.0
    } else {
        x
    }
}

#[inline]
fn saturate_vec(v: Vec3) -> Vec3 {
    Vec3::new(saturate(v.x), saturate(v.y), saturate(v.z))
}

/// Adds `impulse` to the accumulated `acc`, saturated, and returns the impulse to apply:
/// `impulse` itself while the sum is in range (the same bits as an unbounded sum), otherwise the
/// change in `acc`. A NaN impulse (from a velocity error that is not a number) is applied as it
/// is: the pass's velocities are then not numbers either, and the joint drops out for the step
/// (see [`write_back`]), its impulses back to zero.
#[inline]
fn accumulate(acc: &mut f32, impulse: f32) -> f32 {
    let sum = *acc + impulse;
    let total = saturate(sum);
    let applied = if total == sum || sum.is_nan() {
        impulse
    } else {
        total - *acc
    };
    *acc = total;
    applied
}

#[inline]
fn accumulate_vec(acc: &mut Vec3, impulse: Vec3) -> Vec3 {
    Vec3::new(
        accumulate(&mut acc.x, impulse.x),
        accumulate(&mut acc.y, impulse.y),
        accumulate(&mut acc.z, impulse.z),
    )
}

/// Whether a coordinate or length is one the solver can work with: within `±MAX_EXTENT` (so not
/// NaN or infinite either).
#[inline]
fn within_extent(x: f32) -> bool {
    x.abs() <= MAX_EXTENT
}

/// `x` within `[lo, hi]` (`lo <= hi`) by comparisons alone, NaN going to `lo`. Unlike `f32::max`
/// and `f32::min`, which may return either zero of `+0` and `-0` depending on the platform, this
/// always gives the same bits.
#[inline]
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x > hi {
        hi
    } else if x > lo {
        x
    } else {
        lo
    }
}

/// Where a joint's frames are during a substep, in world space.
struct Geometry {
    /// The anchors relative to the centres of mass.
    ra: Vec3,
    rb: Vec3,
    /// Anchor B minus anchor A.
    d: Vec3,
    /// The joint frames' rotations.
    qa: Quat,
    qb: Quat,
}

impl JointConstraint {
    fn geometry(&self, ba: &SolverBody, bb: &SolverBody) -> Geometry {
        let ra = ba.dq.rotate(self.anchor_a);
        let rb = bb.dq.rotate(self.anchor_b);
        Geometry {
            ra,
            rb,
            d: (bb.dp - ba.dp) + self.delta_center + (rb - ra),
            qa: ba.dq * self.frame_a,
            qb: bb.dq * self.frame_b,
        }
    }

    /// The warm-start impulses of last step that this joint's rows use; the rest are dropped (a
    /// joint whose kind or options changed starts those rows from zero).
    fn carry(&mut self, warm: &JointImpulses, h: f32) -> JointImpulses {
        let mut out = JointImpulses::default();
        let (_, y, z) = axes(self.frame_a);
        match self.rows {
            Rows::Inactive => {}
            Rows::Ball => out.linear = saturate_vec(warm.linear),
            Rows::Fixed => {
                out.linear = saturate_vec(warm.linear);
                out.angular = saturate_vec(warm.angular);
            }
            Rows::Hinge(axial) => {
                out.linear = saturate_vec(warm.linear);
                let w = saturate_vec(warm.angular);
                self.cross = [saturate(w.dot(y)), saturate(w.dot(z))];
                out.angular = y * self.cross[0] + z * self.cross[1];
                axial.carry(warm, &mut out, h);
            }
            Rows::Slider(axial) => {
                out.angular = saturate_vec(warm.angular);
                let w = saturate_vec(warm.linear);
                self.cross = [saturate(w.dot(y)), saturate(w.dot(z))];
                out.linear = y * self.cross[0] + z * self.cross[1];
                axial.carry(warm, &mut out, h);
            }
            Rows::Distance(dist) => {
                if dist.rigid || dist.spring.is_some() {
                    out.axial = saturate(warm.axial);
                }
                if !dist.rigid && dist.min > 0.0 {
                    out.lower = saturate(warm.lower);
                }
                if !dist.rigid && dist.max.is_finite() {
                    out.upper = saturate(warm.upper);
                }
            }
        }
        out
    }
}

/// Builds the constraint for this step from the description, the two bodies' frames at the start
/// of the step and last step's impulses.
pub(crate) fn prepare(
    desc: &JointDesc,
    fa: &BodyFrame,
    fb: &BodyFrame,
    bodies: &[SolverBody],
    ctx: &StepContext,
    warm: JointImpulses,
) -> JointConstraint {
    let (a, b) = (fa.index, fb.index);
    let movable = |i: usize| {
        bodies
            .get(i)
            .is_some_and(|s| s.inv_mass != Vec3::ZERO || s.inv_inertia != Mat3::ZERO)
    };
    let usable = a != b && a < bodies.len() && b < bodies.len() && (movable(a) || movable(b));
    let mut c = JointConstraint {
        a,
        b,
        impulses: JointImpulses::default(),
        rows: if usable {
            Rows::new(desc, ctx.h)
        } else {
            Rows::Inactive
        },
        anchor_a: fa.rotation.rotate(desc.frame_a.position - fa.local_center),
        anchor_b: fb.rotation.rotate(desc.frame_b.position - fb.local_center),
        delta_center: fb.center - fa.center,
        frame_a: (fa.rotation * desc.frame_a.rotation.normalize()).normalize(),
        frame_b: (fb.rotation * desc.frame_b.rotation.normalize()).normalize(),
        cross: [0.0; 2],
    };
    c.impulses = c.carry(&warm, ctx.h);
    c
}

/// Applies the accumulated impulses (each substep, before solving).
pub(crate) fn warm_start(c: &JointConstraint, bodies: &[SolverBody], vels: &mut [SolverVel]) {
    if let Rows::Inactive = c.rows {
        return;
    }
    let (ba, bb) = (&bodies[c.a], &bodies[c.b]);
    let (mut va, mut vb) = (vels[c.a], vels[c.b]);
    let g = c.geometry(ba, bb);
    let i = &c.impulses;
    // Each accumulated impulse is a number; their sums are kept numbers too.
    let axial = saturate(i.axial + i.motor + i.lower - i.upper);
    match c.rows {
        Rows::Inactive => {}
        Rows::Ball => push(ba, bb, &mut va, &mut vb, i.linear, g.ra, g.rb),
        Rows::Fixed => {
            push(ba, bb, &mut va, &mut vb, i.linear, g.ra, g.rb);
            turn(ba, bb, &mut va, &mut vb, i.angular);
        }
        Rows::Hinge(_) => {
            let (x, y, z) = axes(g.qa);
            push(ba, bb, &mut va, &mut vb, i.linear, g.ra, g.rb);
            turn(
                ba,
                bb,
                &mut va,
                &mut vb,
                saturate_vec(y * c.cross[0] + z * c.cross[1] + x * axial),
            );
        }
        Rows::Slider(_) => {
            let (x, y, z) = axes(g.qa);
            let p = saturate_vec(y * c.cross[0] + z * c.cross[1] + x * axial);
            push(ba, bb, &mut va, &mut vb, p, g.ra + g.d, g.rb);
            turn(ba, bb, &mut va, &mut vb, i.angular);
        }
        Rows::Distance(_) => {
            let (n, _) = g.d.normalize_and_length();
            push(ba, bb, &mut va, &mut vb, n * axial, g.ra, g.rb);
        }
    }
    // Impulses that would leave a velocity not finite are not applied; the solve then drops the
    // joint for the step.
    write_back(vels, c.a, va, c.b, vb);
}

/// One pass over the joint: with the soft position bias when `use_bias`, without it to relax.
pub(crate) fn solve(
    c: &mut JointConstraint,
    desc: &JointDesc,
    bodies: &[SolverBody],
    vels: &mut [SolverVel],
    ctx: &StepContext,
    use_bias: bool,
) {
    // Everything the rows need was taken from the description in `prepare`.
    let _ = desc;
    if let Rows::Inactive = c.rows {
        return;
    }
    let (ba, bb) = (&bodies[c.a], &bodies[c.b]);
    let (mut va, mut vb) = (vels[c.a], vels[c.b]);
    let g = c.geometry(ba, bb);
    let (va_, vb_) = (&mut va, &mut vb);
    match c.rows {
        Rows::Inactive => {}
        Rows::Ball => solve_point(&mut c.impulses.linear, &g, ba, bb, va_, vb_, ctx, use_bias),
        Rows::Hinge(axial) => solve_hinge(c, &axial, &g, ba, bb, va_, vb_, ctx, use_bias),
        Rows::Slider(axial) => solve_slider(c, &axial, &g, ba, bb, va_, vb_, ctx, use_bias),
        Rows::Fixed => {
            solve_rotation(&mut c.impulses.angular, &g, ba, bb, va_, vb_, ctx, use_bias);
            solve_point(&mut c.impulses.linear, &g, ba, bb, va_, vb_, ctx, use_bias);
        }
        Rows::Distance(dist) => solve_distance(c, &dist, &g, ba, bb, va_, vb_, ctx, use_bias),
    }
    if !write_back(vels, c.a, va, c.b, vb) {
        // Numbers past what `f32` can carry: this joint does nothing more this step.
        c.rows = Rows::Inactive;
        c.impulses = JointImpulses::default();
        c.cross = [0.0; 2];
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_hinge(
    c: &mut JointConstraint,
    axial: &Axial,
    g: &Geometry,
    ba: &SolverBody,
    bb: &SolverBody,
    va: &mut SolverVel,
    vb: &mut SolverVel,
    ctx: &StepContext,
    use_bias: bool,
) {
    let (x, y, z) = axes(g.qa);
    let inv_i = ba.inv_inertia + bb.inv_inertia;
    let scale = trace(&inv_i);
    if axial.any() {
        let mass = inverse(x.dot(inv_i.mul_vec(x)), scale);
        let angle = if axial.spring.is_some() || axial.lower.is_some() || axial.upper.is_some() {
            (g.qa.conjugate() * g.qb).twist_angle(Vec3::X)
        } else {
            0.0
        };
        // The limits read the angle unwound around the middle of their range, into
        // `(mid - π, mid + π]`: a joint pushed just past one end (whose reading wrapped round to
        // the far side of ±π) reads as just past that end, not as far past the other one, which
        // would throw it on round the turn.
        let limit_angle = match (axial.lower, axial.upper) {
            (Some(lower), Some(upper)) => {
                let mid = 0.5 * (lower + upper);
                let mut d = angle - mid;
                if d > PI {
                    d -= TAU;
                } else if d <= -PI {
                    d += TAU;
                }
                mid + d
            }
            _ => angle,
        };
        let speed = |va: &SolverVel, vb: &SolverVel| x.dot(vb.w - va.w);
        let i = &mut c.impulses;
        if let Some(s) = axial.spring {
            let impulse = -mass * s.mass_scale * (speed(va, vb) + s.bias_rate * angle)
                - s.impulse_scale * i.axial;
            let applied = accumulate(&mut i.axial, impulse);
            turn(ba, bb, va, vb, x * applied);
        }
        if let Some(m) = axial.motor {
            let impulse = -mass * (speed(va, vb) - m.speed);
            let max = saturate(m.max_force * ctx.h);
            let old = i.motor;
            i.motor = clamp(saturate(old + impulse), -max, max);
            turn(ba, bb, va, vb, x * (i.motor - old));
        }
        if let Some(lower) = axial.lower {
            let delta = one_sided(
                &mut i.lower,
                mass,
                speed(va, vb),
                limit_angle - lower,
                ctx,
                use_bias,
            );
            turn(ba, bb, va, vb, x * delta);
        }
        if let Some(upper) = axial.upper {
            let delta = one_sided(
                &mut i.upper,
                mass,
                -speed(va, vb),
                upper - limit_angle,
                ctx,
                use_bias,
            );
            turn(ba, bb, va, vb, x * -delta);
        }
    }

    // Frame B's X axis on frame A's: two angular rows about frame A's Y and Z axes, whose error is
    // the swing between the axes.
    let s = hard(ctx, use_bias);
    let (iy, iz) = (inv_i.mul_vec(y), inv_i.mul_vec(z));
    let w = vb.w - va.w;
    let mut rhs = (y.dot(w), z.dot(w));
    if use_bias {
        let e = swing(x, g.qb.rotate(Vec3::X), y);
        rhs.0 += s.bias_rate * e.dot(y);
        rhs.1 += s.bias_rate * e.dot(z);
    }
    // The rounding floor only where some world axis is one neither body can turn about: elsewhere
    // a needle's tiny pivots are real rows, not rounding.
    let free = inv_i.cols[0].x > 0.0 && inv_i.cols[1].y > 0.0 && inv_i.cols[2].z > 0.0;
    let k = solve2(
        y.dot(iy),
        y.dot(iz),
        z.dot(iz),
        rhs,
        if free { 0.0 } else { scale },
    );
    let impulse = [
        -s.mass_scale * k.0 - s.impulse_scale * c.cross[0],
        -s.mass_scale * k.1 - s.impulse_scale * c.cross[1],
    ];
    let applied = [
        accumulate(&mut c.cross[0], impulse[0]),
        accumulate(&mut c.cross[1], impulse[1]),
    ];
    turn(ba, bb, va, vb, y * applied[0] + z * applied[1]);

    solve_point(&mut c.impulses.linear, g, ba, bb, va, vb, ctx, use_bias);
    c.impulses.angular = y * c.cross[0] + z * c.cross[1];
}

#[allow(clippy::too_many_arguments)]
fn solve_slider(
    c: &mut JointConstraint,
    axial: &Axial,
    g: &Geometry,
    ba: &SolverBody,
    bb: &SolverBody,
    va: &mut SolverVel,
    vb: &mut SolverVel,
    ctx: &StepContext,
    use_bias: bool,
) {
    let (x, y, z) = axes(g.qa);
    // Rows along and across the axis act on A at B's anchor, so both bodies are pushed at the
    // same point (no torque on the pair).
    let (ra, rb) = (g.ra + g.d, g.rb);
    let m = ba.inv_mass + bb.inv_mass;
    let relative = |va: &SolverVel, vb: &SolverVel| vb.point_velocity(rb) - va.point_velocity(ra);
    if axial.any() {
        let mass = linear_row_mass(ba, bb, ra, rb, x);
        let position = x.dot(g.d);
        let speed = |va: &SolverVel, vb: &SolverVel| x.dot(relative(va, vb));
        let i = &mut c.impulses;
        if let Some(s) = axial.spring {
            let impulse = -mass * s.mass_scale * (speed(va, vb) + s.bias_rate * position)
                - s.impulse_scale * i.axial;
            let applied = accumulate(&mut i.axial, impulse);
            push(ba, bb, va, vb, x * applied, ra, rb);
        }
        if let Some(motor) = axial.motor {
            let impulse = -mass * (speed(va, vb) - motor.speed);
            let max = saturate(motor.max_force * ctx.h);
            let old = i.motor;
            i.motor = clamp(saturate(old + impulse), -max, max);
            push(ba, bb, va, vb, x * (i.motor - old), ra, rb);
        }
        if let Some(lower) = axial.lower {
            let delta = one_sided(
                &mut i.lower,
                mass,
                speed(va, vb),
                position - lower,
                ctx,
                use_bias,
            );
            push(ba, bb, va, vb, x * delta, ra, rb);
        }
        if let Some(upper) = axial.upper {
            let delta = one_sided(
                &mut i.upper,
                mass,
                -speed(va, vb),
                upper - position,
                ctx,
                use_bias,
            );
            push(ba, bb, va, vb, x * -delta, ra, rb);
        }
    }

    solve_rotation(&mut c.impulses.angular, g, ba, bb, va, vb, ctx, use_bias);

    // Frame B's origin on frame A's X axis: two linear rows along frame A's Y and Z axes.
    let s = hard(ctx, use_bias);
    let (say, saz, sby, sbz) = (ra.cross(y), ra.cross(z), rb.cross(y), rb.cross(z));
    let (iay, iaz) = (ba.inv_inertia.mul_vec(say), ba.inv_inertia.mul_vec(saz));
    let (iby, ibz) = (bb.inv_inertia.mul_vec(sby), bb.inv_inertia.mul_vec(sbz));
    let (my, mz) = (m.mul_elem(y), m.mul_elem(z));
    let v = relative(va, vb);
    let mut rhs = (y.dot(v), z.dot(v));
    if use_bias {
        rhs.0 += s.bias_rate * y.dot(g.d);
        rhs.1 += s.bias_rate * z.dot(g.d);
    }
    let k = solve2(
        y.dot(my) + say.dot(iay) + sby.dot(iby),
        y.dot(mz) + say.dot(iaz) + sby.dot(ibz),
        z.dot(mz) + saz.dot(iaz) + sbz.dot(ibz),
        rhs,
        linear_scale(ba, bb, ra, rb),
    );
    let impulse = [
        -s.mass_scale * k.0 - s.impulse_scale * c.cross[0],
        -s.mass_scale * k.1 - s.impulse_scale * c.cross[1],
    ];
    let applied = [
        accumulate(&mut c.cross[0], impulse[0]),
        accumulate(&mut c.cross[1], impulse[1]),
    ];
    push(ba, bb, va, vb, y * applied[0] + z * applied[1], ra, rb);
    c.impulses.linear = y * c.cross[0] + z * c.cross[1];
}

#[allow(clippy::too_many_arguments)]
fn solve_distance(
    c: &mut JointConstraint,
    dist: &Distance,
    g: &Geometry,
    ba: &SolverBody,
    bb: &SolverBody,
    va: &mut SolverVel,
    vb: &mut SolverVel,
    ctx: &StepContext,
    use_bias: bool,
) {
    // Coincident anchors have no line between them: the axis is zero and so is every impulse.
    let (n, length) = g.d.normalize_and_length();
    let (ra, rb) = (g.ra, g.rb);
    let mass = linear_row_mass(ba, bb, ra, rb, n);
    let speed =
        |va: &SolverVel, vb: &SolverVel| n.dot(vb.point_velocity(rb) - va.point_velocity(ra));
    let i = &mut c.impulses;
    if dist.rigid {
        let s = hard(ctx, use_bias);
        let bias = if use_bias {
            s.bias_rate * (length - dist.length)
        } else {
            0.0
        };
        let impulse = -mass * s.mass_scale * (speed(va, vb) + bias) - s.impulse_scale * i.axial;
        let applied = accumulate(&mut i.axial, impulse);
        push(ba, bb, va, vb, n * applied, ra, rb);
        return;
    }
    if let Some(s) = dist.spring {
        let impulse = -mass * s.mass_scale * (speed(va, vb) + s.bias_rate * (length - dist.length))
            - s.impulse_scale * i.axial;
        let applied = accumulate(&mut i.axial, impulse);
        push(ba, bb, va, vb, n * applied, ra, rb);
    }
    if dist.min > 0.0 {
        let delta = one_sided(
            &mut i.lower,
            mass,
            speed(va, vb),
            length - dist.min,
            ctx,
            use_bias,
        );
        push(ba, bb, va, vb, n * delta, ra, rb);
    }
    if dist.max.is_finite() {
        let delta = one_sided(
            &mut i.upper,
            mass,
            -speed(va, vb),
            dist.max - length,
            ctx,
            use_bias,
        );
        push(ba, bb, va, vb, n * -delta, ra, rb);
    }
}

/// The point rows: anchor B on anchor A, three linear rows solved together through the 3×3
/// effective mass `K = (M_a⁻¹ + M_b⁻¹) − [r_a]× I_a⁻¹ [r_a]× − [r_b]× I_b⁻¹ [r_b]×`.
#[allow(clippy::too_many_arguments)]
fn solve_point(
    acc: &mut Vec3,
    g: &Geometry,
    ba: &SolverBody,
    bb: &SolverBody,
    va: &mut SolverVel,
    vb: &mut SolverVel,
    ctx: &StepContext,
    use_bias: bool,
) {
    let s = hard(ctx, use_bias);
    let (sa, sb) = (Mat3::skew(g.ra), Mat3::skew(g.rb));
    let k = Mat3::from_diagonal(ba.inv_mass + bb.inv_mass)
        - sa * ba.inv_inertia * sa
        - sb * bb.inv_inertia * sb;
    let cdot = vb.point_velocity(g.rb) - va.point_velocity(g.ra);
    let bias = if use_bias {
        g.d * s.bias_rate
    } else {
        Vec3::ZERO
    };
    let x = solve3(&k, cdot + bias, linear_scale(ba, bb, g.ra, g.rb));
    let impulse = -(x * s.mass_scale) - *acc * s.impulse_scale;
    let applied = accumulate_vec(acc, impulse);
    push(ba, bb, va, vb, applied, g.ra, g.rb);
}

/// Three angular rows locking frame B's rotation to frame A's (slider, fixed), through the
/// effective mass `I_a⁻¹ + I_b⁻¹`; the error is the rotation vector between the frames.
#[allow(clippy::too_many_arguments)]
fn solve_rotation(
    acc: &mut Vec3,
    g: &Geometry,
    ba: &SolverBody,
    bb: &SolverBody,
    va: &mut SolverVel,
    vb: &mut SolverVel,
    ctx: &StepContext,
    use_bias: bool,
) {
    let s = hard(ctx, use_bias);
    let k = ba.inv_inertia + bb.inv_inertia;
    let cdot = vb.w - va.w;
    let bias = if use_bias {
        rotation_error(g.qa, g.qb) * s.bias_rate
    } else {
        Vec3::ZERO
    };
    // The block is the inverse inertias themselves, whose locked rows and columns are exact
    // zeros: its own trace is its scale.
    let x = solve3(&k, cdot + bias, trace(&k));
    let impulse = -(x * s.mass_scale) - *acc * s.impulse_scale;
    let applied = accumulate_vec(acc, impulse);
    turn(ba, bb, va, vb, applied);
}

/// One side of a limit. `c` is how far the joint is inside it (negative once past it) and `cdot`
/// how fast that grows. Short of the limit the row only stops an approach that would pass it
/// within the substep (speculative); past it, it pushes back softly while correcting position and
/// rigidly while relaxing. The accumulated impulse stays non-negative; returns its change.
fn one_sided(
    acc: &mut f32,
    mass: f32,
    cdot: f32,
    c: f32,
    ctx: &StepContext,
    use_bias: bool,
) -> f32 {
    let (bias, mass_scale, impulse_scale) = if c > 0.0 {
        (c * ctx.inv_h, 1.0, 0.0)
    } else if use_bias {
        let s = ctx.joint_softness;
        (s.bias_rate * c, s.mass_scale, s.impulse_scale)
    } else {
        (0.0, 1.0, 0.0)
    };
    let impulse = -mass * mass_scale * (cdot + bias) - impulse_scale * *acc;
    let total = clamp(*acc + impulse, 0.0, MAX_IMPULSE);
    let delta = total - *acc;
    *acc = total;
    delta
}

/// The soft coefficients of a hard row: the joint softness while correcting position, rigid while
/// relaxing.
#[inline]
fn hard(ctx: &StepContext, use_bias: bool) -> Softness {
    if use_bias {
        ctx.joint_softness
    } else {
        Softness::RIGID
    }
}

/// Linear impulse `p` on B at `rb` and `-p` on A at `ra`.
#[inline]
fn push(
    ba: &SolverBody,
    bb: &SolverBody,
    va: &mut SolverVel,
    vb: &mut SolverVel,
    p: Vec3,
    ra: Vec3,
    rb: Vec3,
) {
    ba.apply(va, -p, ra);
    bb.apply(vb, p, rb);
}

/// Angular impulse `l` on B and `-l` on A.
#[inline]
fn turn(ba: &SolverBody, bb: &SolverBody, va: &mut SolverVel, vb: &mut SolverVel, l: Vec3) {
    va.w -= ba.inv_inertia.mul_vec(l);
    vb.w += bb.inv_inertia.mul_vec(l);
}

/// Writes two solver velocities back, never touching the static slot. Writes nothing and returns
/// false when a velocity is not a number, or so large that its square is not (beyond 1.8e19:
/// only numbers past what `f32` carries, such as a saturated impulse on a light body, give one).
#[inline]
fn write_back(vels: &mut [SolverVel], a: usize, va: SolverVel, b: usize, vb: SolverVel) -> bool {
    let usable = |v: Vec3| v.length_squared().is_finite();
    if !(usable(va.v) && usable(va.w) && usable(vb.v) && usable(vb.w)) {
        return false;
    }
    if a != 0 {
        vels[a] = va;
    }
    if b != 0 {
        vels[b] = vb;
    }
    true
}

/// The rotation's X, Y and Z axes.
#[inline]
fn axes(q: Quat) -> (Vec3, Vec3, Vec3) {
    let [x, y, z] = q.to_mat3().cols;
    (x, y, z)
}

#[inline]
fn trace(m: &Mat3) -> f32 {
    m.cols[0].x + m.cols[1].y + m.cols[2].z
}

/// `1 / k`, or zero when `k` is no more than rounding of `scale` (a direction that cannot move);
/// saturated, so an inertia near the largest `f32` gives a number.
#[inline]
fn inverse(k: f32, scale: f32) -> f32 {
    if k > scale * PIVOT_EPS && k > 0.0 {
        saturate(1.0 / k)
    } else {
        0.0
    }
}

/// A bound of every entry of a block of linear rows acting at `ra` on A and `rb` on B, from the
/// magnitudes they are summed from: the scale for [`solve2`] and [`solve3`].
fn linear_scale(ba: &SolverBody, bb: &SolverBody, ra: Vec3, rb: Vec3) -> f32 {
    let m = ba.inv_mass + bb.inv_mass;
    // Free along every axis, the block is at least diag(m): never rounding.
    if m.x > 0.0 && m.y > 0.0 && m.z > 0.0 {
        return 0.0;
    }
    m.x + m.y
        + m.z
        + ra.length_squared() * trace(&ba.inv_inertia)
        + rb.length_squared() * trace(&bb.inv_inertia)
}

/// The effective mass of one linear row along unit `n` acting at `ra` on A and `rb` on B. Its
/// scale for [`inverse`] bounds what rounding of the rotated inertias can produce.
fn linear_row_mass(ba: &SolverBody, bb: &SolverBody, ra: Vec3, rb: Vec3, n: Vec3) -> f32 {
    let (sa, sb) = (ra.cross(n), rb.cross(n));
    let linear = n.dot((ba.inv_mass + bb.inv_mass).mul_elem(n));
    let k = linear + sa.dot(ba.inv_inertia.mul_vec(sa)) + sb.dot(bb.inv_inertia.mul_vec(sb));
    let scale = linear
        + sa.length_squared() * trace(&ba.inv_inertia)
        + sb.length_squared() * trace(&bb.inv_inertia);
    inverse(k, scale)
}

/// The rotation vector turning unit vector `from` onto unit vector `to` the short way: along
/// `from × to`, of length `2·sin(θ/2)` (θ for small angles). Opposite vectors turn about
/// `fallback`.
fn swing(from: Vec3, to: Vec3, fallback: Vec3) -> Vec3 {
    let axis = from.cross(to);
    let cos_half_squared = 0.5 * (1.0 + from.dot(to));
    if cos_half_squared > 1e-6 {
        axis * (1.0 / cos_half_squared.sqrt())
    } else {
        fallback * 2.0
    }
}

/// The rotation vector (along the axis, of length `2·sin(θ/2)`) turning orientation `qa` onto
/// `qb`, in world space.
fn rotation_error(qa: Quat, qb: Quat) -> Vec3 {
    let q = qb * qa.conjugate();
    let v = q.vector() * 2.0;
    if q.w < 0.0 { -v } else { v }
}

/// The pivot at or below which a block's direction is one neither body can move in:
/// [`PIVOT_EPS`] of the block's own `trace` (what cancellation in the factorization leaves), but
/// never below [`ROUNDING_EPS`] of the `scale` the block was summed from (what rounding leaves of
/// a block whose every row is such a direction).
#[inline]
fn pivot_tolerance(trace: f32, scale: f32) -> f32 {
    let (own, rounding) = (PIVOT_EPS * trace, ROUNDING_EPS * scale);
    if rounding > own { rounding } else { own }
}

/// `2⁻⁶⁴`: the part of a velocity error a block too small for `f32` is solved for (see
/// [`solve3`]).
const PART: f32 = 1.0 / 18_446_744_073_709_551_616.0;

/// Solves `k x = b` for a symmetric positive semi-definite `k` (LDLᵀ in index order). A pivot at
/// or below [`pivot_tolerance`] of the trace and of `scale` (a bound of the entries from the
/// magnitudes they were summed from) is a direction neither body can move in: its unknown stays
/// zero and the others are solved without it, so locked axes and static bodies never divide by
/// zero, nor by rounding.
///
/// A `b` that is not a number gives an answer that is not either (the joint then drops out for
/// the step). A block so small that the answer overflows (an inertia near the largest `f32`) is
/// solved for [`PART`] of `b`, and that answer scaled back and saturated: `±∞ · 0` would turn the
/// overflow into NaN.
fn solve3(k: &Mat3, b: Vec3, scale: f32) -> Vec3 {
    if !b.is_finite() {
        return Vec3::splat(f32::NAN);
    }
    let x = ldl3(k, b, scale);
    if x.is_finite() {
        return x;
    }
    saturate_vec(ldl3(k, b * PART, scale) * (1.0 / PART))
}

/// [`solve3`] without the care for numbers past `f32`.
fn ldl3(k: &Mat3, b: Vec3, scale: f32) -> Vec3 {
    let [c0, c1, c2] = k.cols;
    let tol = pivot_tolerance(c0.x + c1.y + c2.z, scale);
    let (mut d0, mut l10, mut l20) = (c0.x, 0.0, 0.0);
    if d0 > tol {
        l10 = c0.y / d0;
        l20 = c0.z / d0;
    } else {
        d0 = 0.0;
    }
    let (mut d1, mut l21) = (c1.y - l10 * l10 * d0, 0.0);
    if d1 > tol {
        l21 = (c1.z - l20 * l10 * d0) / d1;
    } else {
        d1 = 0.0;
    }
    let d2 = c2.z - l20 * l20 * d0 - l21 * l21 * d1;
    let y0 = b.x;
    let y1 = b.y - l10 * y0;
    let y2 = b.z - l20 * y0 - l21 * y1;
    let z0 = if d0 > 0.0 { y0 / d0 } else { 0.0 };
    let z1 = if d1 > 0.0 { y1 / d1 } else { 0.0 };
    let x2 = if d2 > tol { y2 / d2 } else { 0.0 };
    let x1 = z1 - l21 * x2;
    let x0 = z0 - l10 * x1 - l20 * x2;
    Vec3::new(x0, x1, x2)
}

/// [`solve3`] for the symmetric 2×2 `[[k11, k12], [k12, k22]]`.
fn solve2(k11: f32, k12: f32, k22: f32, b: (f32, f32), scale: f32) -> (f32, f32) {
    if !(b.0.is_finite() && b.1.is_finite()) {
        return (f32::NAN, f32::NAN);
    }
    let x = ldl2(k11, k12, k22, b, scale);
    if x.0.is_finite() && x.1.is_finite() {
        return x;
    }
    let part = ldl2(k11, k12, k22, (b.0 * PART, b.1 * PART), scale);
    (
        saturate(part.0 * (1.0 / PART)),
        saturate(part.1 * (1.0 / PART)),
    )
}

/// [`solve2`] without the care for numbers past `f32`.
fn ldl2(k11: f32, k12: f32, k22: f32, b: (f32, f32), scale: f32) -> (f32, f32) {
    let tol = pivot_tolerance(k11 + k22, scale);
    let (mut d0, mut l) = (k11, 0.0);
    if d0 > tol {
        l = k12 / d0;
    } else {
        d0 = 0.0;
    }
    let d1 = k22 - l * l * d0;
    let y1 = b.1 - l * b.0;
    let x1 = if d1 > tol { y1 / d1 } else { 0.0 };
    let z0 = if d0 > 0.0 { b.0 / d0 } else { 0.0 };
    (z0 - l * x1, x1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3, tol: f32) -> bool {
        (a - b).length() <= tol
    }

    #[test]
    fn small_solves_match_the_inverse_and_leave_immovable_directions_alone() {
        let k = Mat3::from_cols(
            Vec3::new(4.0, 1.0, 0.5),
            Vec3::new(1.0, 3.0, 0.2),
            Vec3::new(0.5, 0.2, 2.0),
        );
        let b = Vec3::new(1.0, -2.0, 0.5);
        let x = solve3(&k, b, 9.0);
        assert!(close(x, k.inverse().unwrap().mul_vec(b), 1e-5));
        // A direction nothing can move in (a zero row and column): its unknown stays zero and the
        // others are solved exactly.
        let k = Mat3::from_cols(
            Vec3::new(2.0, 0.0, 0.5),
            Vec3::ZERO,
            Vec3::new(0.5, 0.0, 1.0),
        );
        let x = solve3(&k, Vec3::new(1.0, 7.0, 2.0), 3.0);
        assert_eq!(x.y, 0.0);
        let r = k.mul_vec(x);
        assert!(
            (r.x - 1.0).abs() < 1e-5 && (r.z - 2.0).abs() < 1e-5,
            "{r:?}"
        );
        // Rank one (rounding leaves a tiny pivot that must not be divided by), and nothing.
        let n = Vec3::new(1.0, 2.0, 2.0) / 3.0;
        let k = Mat3::outer(n, n);
        let x = solve3(&k, n * 3.0, 1.0);
        assert!(x.is_finite() && close(k.mul_vec(x), n * 3.0, 1e-5), "{x:?}");
        assert_eq!(solve3(&Mat3::ZERO, Vec3::ONE, 0.0), Vec3::ZERO);
        let (x0, x1) = solve2(2.0, 0.5, 1.0, (1.0, 1.0), 3.0);
        assert!((2.0 * x0 + 0.5 * x1 - 1.0).abs() < 1e-6 && (0.5 * x0 + x1 - 1.0).abs() < 1e-6);
        assert_eq!(solve2(0.0, 0.0, 3.0, (5.0, 6.0), 3.0), (0.0, 2.0));
        assert_eq!(solve2(0.0, 0.0, 0.0, (5.0, 6.0), 0.0), (0.0, 0.0));
    }

    /// The hinge axis rows of a 2.5D body, which may only turn about world Z, hinged about Z:
    /// both rows are directions it cannot turn in. A frame built with `from_rotation_arc` leaves
    /// about 1.2e-7 of its Z axis along world Z (as here), so the block is pure rounding (about
    /// 1e-14 of the inverse inertia) and its own trace is rounding too. The solve must leave it
    /// alone instead of cancelling the free spin with an impulse of 1e7.
    #[test]
    fn a_block_that_is_only_rounding_of_its_scale_is_left_out_of_the_solve() {
        let (y, z) = (Vec3::Y, Vec3::new(-1.0, 0.0, 1.2e-7));
        let inv_i = Mat3::from_diagonal(Vec3::new(0.0, 0.0, 1.2));
        let (iy, iz) = (inv_i.mul_vec(y), inv_i.mul_vec(z));
        let (k11, k12, k22) = (y.dot(iy), y.dot(iz), z.dot(iz));
        assert!(k11 + k22 > 0.0, "the frame carries rounding: {k11} {k22}");
        assert!(k11 + k22 < 1e-12, "and nothing but rounding: {k11} {k22}");
        let spin = Vec3::new(0.0, 0.0, -2.0);
        let rhs = (y.dot(spin), z.dot(spin));
        assert_eq!(solve2(k11, k12, k22, rhs, trace(&inv_i)), (0.0, 0.0));
        // The same rows against a body that can turn about them are solved exactly.
        let free = Mat3::from_diagonal(Vec3::new(0.5, 0.7, 1.2));
        let (iy, iz) = (free.mul_vec(y), free.mul_vec(z));
        let x = solve2(y.dot(iy), y.dot(iz), z.dot(iz), rhs, trace(&free));
        let w = spin - free.mul_vec(y * x.0 + z * x.1);
        assert!(y.dot(w).abs() < 1e-5 && z.dot(w).abs() < 1e-5, "{w:?}");
        // The point rows of a turntable, which cannot move and may only turn about world Y, with
        // its anchor on that axis 0.1 m below its centre: no row can move the anchor, but the
        // anchor's offset off the axis carries rounding (7.45e-9), so turning moves it along X
        // by that much. That row must be left alone too, or it stops the table.
        let rb = Vec3::new(0.0, -0.1, 7.45e-9);
        let table = SolverBody {
            inv_inertia: Mat3::from_diagonal(Vec3::new(0.0, 1.2, 0.0)),
            ..SolverBody::STATIC
        };
        let table_vel = SolverVel {
            v: Vec3::ZERO,
            w: Vec3::new(0.0, 2.0, 0.0),
        };
        let sb = Mat3::skew(rb);
        let k = Mat3::ZERO - sb * table.inv_inertia * sb;
        assert!(k.cols[0].x > 0.0 && k.cols[0].x < 1e-15, "{k:?}");
        let cdot = table_vel.point_velocity(rb);
        let scale = linear_scale(&SolverBody::STATIC, &table, Vec3::ZERO, rb);
        assert_eq!(solve3(&k, cdot, scale), Vec3::ZERO, "{k:?} {cdot:?}");
    }

    #[test]
    fn the_error_measures_turn_the_short_way() {
        let s = swing(Vec3::X, Vec3::Y, Vec3::Z);
        assert!(close(s, Vec3::Z * std::f32::consts::SQRT_2, 1e-6), "{s:?}");
        assert_eq!(swing(Vec3::X, -Vec3::X, Vec3::Z), Vec3::Z * 2.0);
        assert_eq!(swing(Vec3::X, Vec3::X, Vec3::Z), Vec3::ZERO);
        let axis = Vec3::new(1.0, 2.0, 2.0) / 3.0;
        let q = Quat::from_axis_angle(axis, 0.2);
        let e = rotation_error(Quat::IDENTITY, q);
        assert!(close(e, axis * 0.2, 1e-3), "{e:?}");
        // -q is the same rotation.
        let minus = Quat::new(-q.x, -q.y, -q.z, -q.w);
        assert!(close(rotation_error(Quat::IDENTITY, minus), e, 1e-6));
        // Measured in world space, whatever frame A's orientation.
        let r = Quat::from_axis_angle(Vec3::Y, 1.0);
        assert!(close(rotation_error(r, q * r), e, 1e-5));
    }

    #[test]
    fn options_are_put_in_order_and_unusable_numbers_dropped() {
        let h = 1.0 / 240.0;
        let a = Axial::new(
            Some((2.0, -5.0)),
            Some(Motor {
                speed: f32::NAN,
                max_force: 1.0,
            }),
            Some(Spring {
                hertz: -1.0,
                damping_ratio: 1.0,
            }),
            h,
            true,
        );
        assert_eq!((a.lower, a.upper), (Some(-PI), Some(2.0)));
        assert!(a.motor.is_none() && a.spring.is_none());
        let a = Axial::new(
            Some((f32::NAN, f32::INFINITY)),
            Some(Motor {
                speed: 1.0,
                max_force: -3.0,
            }),
            None,
            h,
            false,
        );
        assert_eq!((a.lower, a.upper), (None, None));
        assert_eq!(a.motor.map(|m| m.max_force), Some(0.0));
        // Without a spring the length is free in the range (a rope); `min == max` is rigid.
        let rope = DistanceOptions {
            length: 2.0,
            min: 0.0,
            max: 1.5,
            spring: None,
        };
        let rope = Distance::new(&rope, h).unwrap();
        assert!(!rope.rigid && rope.length == 1.5 && rope.spring.is_none());
        let rod = DistanceOptions {
            length: 0.3,
            min: 1.0,
            max: 0.5,
            spring: Some(Spring {
                hertz: 4.0,
                damping_ratio: 0.5,
            }),
        };
        let rod = Distance::new(&rod, h).unwrap();
        assert!(rod.rigid && rod.length == 1.0 && rod.max == 1.0 && rod.spring.is_none());
        let endless = DistanceOptions {
            length: 1.0,
            min: f32::INFINITY,
            max: f32::INFINITY,
            spring: None,
        };
        assert!(Distance::new(&endless, h).is_none());
    }

    #[test]
    fn hinge_limits_end_at_plus_or_minus_pi_and_leave_a_gap_there() {
        let h = 1.0 / 240.0;
        let limits = |limit: (f32, f32), angular: bool| {
            let a = Axial::new(Some(limit), None, None, h, angular);
            (a.lower, a.upper)
        };
        let nan = f32::NAN;
        // A missing side of an angle is the end at ±π; ranges short of the whole turn stay.
        assert_eq!(limits((-1.0, nan), true), (Some(-1.0), Some(PI)));
        assert_eq!(limits((nan, 1.0), true), (Some(-PI), Some(1.0)));
        assert_eq!(limits((-1.0, f32::INFINITY), true), (Some(-1.0), Some(PI)));
        assert_eq!(limits((-1.0, 3.138), true), (Some(-1.0), Some(3.138)));
        assert_eq!(limits((3.0, 3.1), true), (Some(3.0), Some(3.1)));
        assert_eq!(limits((nan, nan), true), (None, None));
        // The whole turn, however given, keeps a gap at ±π; nearly the whole turn, the rest of
        // that gap.
        let gap = |(l, u): (Option<f32>, Option<f32>)| (l.unwrap() + TAU) - u.unwrap();
        for whole in [(-PI, PI), (5.0, -7.0), (-PI, nan), (nan, PI), (-4.0, 4.0)] {
            let (l, u) = limits(whole, true);
            assert!(
                (l.unwrap() + PI - 0.05).abs() < 1e-6 && (u.unwrap() - PI + 0.05).abs() < 1e-6,
                "{whole:?}: {l:?} {u:?}"
            );
        }
        let nearly = limits((-PI, 3.1), true);
        assert!((gap(nearly) - ANGLE_LIMIT_GAP).abs() < 1e-6, "{nearly:?}");
        let pulled = (nearly.0.unwrap() + PI, 3.1 - nearly.1.unwrap());
        assert!(
            pulled.0 > 0.0 && (pulled.0 - pulled.1).abs() < 1e-6,
            "{nearly:?}"
        );
        // A slider's sides stay as given.
        assert_eq!(limits((nan, 1.0), false), (None, Some(1.0)));
        assert_eq!(limits((-PI, PI), false), (Some(-PI), Some(PI)));
    }

    /// A body on a hinge about X, turned 0.01 rad past its upper limit and turning on at 2 rad/s:
    /// its spin about X after one soft pass and after a relax pass, and the lower and upper
    /// impulses.
    fn just_past_the_upper_limit(lower: f32, upper: f32) -> (f32, f32, f32, f32) {
        let bodies = vec![
            SolverBody::STATIC,
            SolverBody {
                inv_mass: Vec3::ONE,
                inv_inertia: Mat3::IDENTITY,
                ..SolverBody::STATIC
            },
        ];
        let mut vels = vec![
            SolverVel::ZERO,
            SolverVel {
                v: Vec3::ZERO,
                w: Vec3::new(2.0, 0.0, 0.0),
            },
        ];
        let id = BodyId {
            index: 0,
            generation: 0,
        };
        let desc = JointDesc {
            body_a: id,
            body_b: id,
            frame_a: Iso::IDENTITY,
            frame_b: Iso::IDENTITY,
            kind: JointKind::Hinge(HingeOptions {
                limit: Some((lower, upper)),
                ..Default::default()
            }),
            collide_connected: false,
            user_data: 0,
        };
        let end = if upper < PI { upper } else { PI };
        let fb = BodyFrame {
            rotation: Quat::from_axis_angle(Vec3::X, end + 0.01),
            ..frame(Vec3::ZERO, 1)
        };
        let ctx = context();
        let fa = frame(Vec3::ZERO, 0);
        let mut c = prepare(&desc, &fa, &fb, &bodies, &ctx, JointImpulses::default());
        solve(&mut c, &desc, &bodies, &mut vels, &ctx, true);
        let soft = vels[1].w.x;
        solve(&mut c, &desc, &bodies, &mut vels, &ctx, false);
        (soft, vels[1].w.x, c.impulses.lower, c.impulses.upper)
    }

    #[test]
    fn a_hinge_just_past_an_upper_limit_at_pi_is_stopped_by_that_limit() {
        // Past π the angle reads near -π: far past the lower limit, whose soft pass used to push
        // the joint on round the turn at some 130 rad/s while the upper one did nothing.
        for (lower, upper) in [
            (-1.0f32, 3.0f32),
            (-1.0, 3.138),
            (-1.0, PI),
            (-1.0, 4.0),
            (-1.0, f32::INFINITY),
            (-1.0, f32::NAN),
        ] {
            let (soft, spin, lower_impulse, upper_impulse) =
                just_past_the_upper_limit(lower, upper);
            assert!(
                soft < 0.0 && spin.abs() < 1e-3 && lower_impulse == 0.0 && upper_impulse > 1.9,
                "({lower}, {upper}): spin 2 -> {soft} -> {spin}, impulses {lower_impulse} \
                 {upper_impulse}"
            );
        }
    }

    fn context() -> StepContext {
        let h = 1.0 / 240.0;
        StepContext {
            dt: 4.0 * h,
            h,
            inv_h: 1.0 / h,
            contact_softness: Softness::new(30.0, 10.0, h),
            static_softness: Softness::new(60.0, 10.0, h),
            joint_softness: Softness::new(60.0, 2.0, h),
            max_push_velocity: 3.0,
            restitution_threshold: 1.0,
        }
    }

    fn frame(center: Vec3, index: usize) -> BodyFrame {
        BodyFrame {
            center,
            rotation: Quat::IDENTITY,
            local_center: Vec3::ZERO,
            index,
        }
    }

    #[test]
    fn a_rigid_pass_stops_the_anchor_without_touching_locked_axes_or_the_static_slot() {
        // A body hanging 1 m below a fixed pivot, moving away from it. It cannot move along Z
        // nor turn about X or Y.
        let bodies = vec![
            SolverBody::STATIC,
            SolverBody {
                inv_mass: Vec3::new(1.0, 1.0, 0.0),
                inv_inertia: Mat3::from_diagonal(Vec3::new(0.0, 0.0, 2.0)),
                ..SolverBody::STATIC
            },
        ];
        let mut vels = vec![
            SolverVel::ZERO,
            SolverVel {
                v: Vec3::new(1.0, -2.0, 0.0),
                w: Vec3::new(0.0, 0.0, 0.5),
            },
        ];
        let id = BodyId {
            index: 0,
            generation: 0,
        };
        let anchor = Vec3::new(0.3, 1.0, 0.0);
        let desc = JointDesc {
            body_a: id,
            body_b: id,
            frame_a: Iso::from_position(Vec3::new(0.3, 0.0, 0.0)),
            frame_b: Iso::from_position(anchor),
            kind: JointKind::Ball,
            collide_connected: false,
            user_data: 0,
        };
        let ctx = context();
        let (fa, fb) = (frame(Vec3::ZERO, 0), frame(Vec3::new(0.0, -1.0, 0.0), 1));
        let mut c = prepare(&desc, &fa, &fb, &bodies, &ctx, JointImpulses::default());
        warm_start(&c, &bodies, &mut vels);
        solve(&mut c, &desc, &bodies, &mut vels, &ctx, false);
        let v = vels[1].point_velocity(anchor);
        assert!(
            v.x.abs() < 1e-5 && v.y.abs() < 1e-5,
            "the anchor stops: {v:?}"
        );
        assert_eq!(vels[1].v.z, 0.0);
        assert_eq!((vels[1].w.x, vels[1].w.y), (0.0, 0.0));
        assert_eq!(
            c.impulses.linear.z, 0.0,
            "nothing pushes along the locked axis"
        );
        assert_eq!(vels[0].v, Vec3::ZERO);
        assert_eq!(vels[0].w, Vec3::ZERO);
        // The impulse is kept for the next substep, and warm starting with it alone stops the
        // same motion again.
        let kept = c.impulses;
        let mut again_vels = vec![
            SolverVel::ZERO,
            SolverVel {
                v: Vec3::new(1.0, -2.0, 0.0),
                w: Vec3::new(0.0, 0.0, 0.5),
            },
        ];
        let c = prepare(&desc, &fa, &fb, &bodies, &ctx, kept);
        warm_start(&c, &bodies, &mut again_vels);
        assert!(close(again_vels[1].v, vels[1].v, 1e-5) && close(again_vels[1].w, vels[1].w, 1e-5));
    }

    #[test]
    fn a_pass_that_would_leave_a_velocity_not_finite_writes_nothing_and_drops_the_joint() {
        // Body B's centre is so far from A's that the soft correction overflows.
        let body = SolverBody {
            inv_mass: Vec3::ONE,
            inv_inertia: Mat3::IDENTITY,
            ..SolverBody::STATIC
        };
        let start_vel = SolverVel {
            v: Vec3::new(1.0, 0.0, 0.0),
            w: Vec3::ZERO,
        };
        let bodies = vec![SolverBody::STATIC, body];
        let mut vels = vec![SolverVel::ZERO, start_vel];
        let id = BodyId {
            index: 0,
            generation: 0,
        };
        let desc = JointDesc {
            body_a: id,
            body_b: id,
            frame_a: Iso::IDENTITY,
            frame_b: Iso::IDENTITY,
            kind: JointKind::Ball,
            collide_connected: false,
            user_data: 0,
        };
        let ctx = context();
        let (fa, fb) = (frame(Vec3::ZERO, 0), frame(Vec3::new(3e38, 0.0, 0.0), 1));
        let mut c = prepare(&desc, &fa, &fb, &bodies, &ctx, JointImpulses::default());
        solve(&mut c, &desc, &bodies, &mut vels, &ctx, true);
        assert_eq!(vels[1].v, start_vel.v, "nothing written");
        assert_eq!(c.impulses, JointImpulses::default());
        warm_start(&c, &bodies, &mut vels);
        solve(&mut c, &desc, &bodies, &mut vels, &ctx, false);
        assert_eq!(vels[1].v, start_vel.v, "and nothing more this step");
    }

    #[test]
    fn a_joint_between_bodies_that_cannot_move_does_nothing() {
        let kinematic = SolverBody::STATIC;
        let bodies = vec![SolverBody::STATIC, kinematic];
        let mut vels = vec![
            SolverVel::ZERO,
            SolverVel {
                v: Vec3::new(1.0, 0.0, 0.0),
                w: Vec3::ZERO,
            },
        ];
        let id = BodyId {
            index: 0,
            generation: 0,
        };
        let desc = JointDesc {
            body_a: id,
            body_b: id,
            frame_a: Iso::IDENTITY,
            frame_b: Iso::IDENTITY,
            kind: JointKind::Fixed,
            collide_connected: false,
            user_data: 0,
        };
        let ctx = context();
        let warm = JointImpulses {
            linear: Vec3::ONE,
            ..Default::default()
        };
        let (fa, fb) = (frame(Vec3::ZERO, 0), frame(Vec3::Y, 1));
        let mut c = prepare(&desc, &fa, &fb, &bodies, &ctx, warm);
        warm_start(&c, &bodies, &mut vels);
        solve(&mut c, &desc, &bodies, &mut vels, &ctx, true);
        assert_eq!(vels[1].v, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(c.impulses, JointImpulses::default());
    }
    #[test]
    fn accumulated_impulses_saturate_and_a_nan_one_is_applied_as_it_is() {
        assert_eq!(saturate(f32::INFINITY), MAX_IMPULSE);
        assert_eq!(saturate(f32::NEG_INFINITY), -MAX_IMPULSE);
        assert_eq!(saturate(f32::MAX), MAX_IMPULSE);
        assert_eq!(saturate(f32::NAN), 0.0);
        assert_eq!(saturate(2.5), 2.5);
        assert_eq!(saturate(-0.0).to_bits(), (-0.0f32).to_bits());
        // In range: the impulse itself, and the plain sum.
        let mut acc = 1.0;
        assert_eq!(accumulate(&mut acc, 1e-8), 1e-8);
        assert_eq!(acc, 1.0 + 1e-8);
        // Past the bound: what brings the sum to it.
        let start = 0.75 * MAX_IMPULSE;
        let mut acc = start;
        assert_eq!(accumulate(&mut acc, f32::MAX), MAX_IMPULSE - start);
        assert_eq!(acc, MAX_IMPULSE);
        assert_eq!(accumulate(&mut acc, f32::NEG_INFINITY), -2.0 * MAX_IMPULSE);
        assert_eq!(acc, -MAX_IMPULSE);
        // NaN goes on to the velocities (the joint drops out); the sum keeps a number.
        let mut acc = 3.0;
        assert!(accumulate(&mut acc, f32::NAN).is_nan());
        assert_eq!(acc, 0.0);
    }

    #[test]
    fn a_block_too_small_for_f32_gives_saturated_impulses_not_nan() {
        // The inverse inertia of a body whose inertia is 1e38 kg m², turning at 5 rad/s about Y:
        // the impulse that stops it, 5e38, overflows, and the plain solve's back substitution
        // turns the overflow into NaN (0 × ∞).
        let k = Mat3::from_diagonal(Vec3::splat(1e-38));
        let b = Vec3::new(0.0, 5.0, 0.0);
        assert!(!ldl3(&k, b, trace(&k)).is_finite());
        assert_eq!(solve3(&k, b, trace(&k)), Vec3::new(0.0, MAX_IMPULSE, 0.0));
        let x = solve2(1e-38, 0.0, 1e-38, (0.0, -5.0), 2e-38);
        assert_eq!(x, (0.0, -MAX_IMPULSE));
        // In range nothing changes; an error that is not a number is not solved.
        let k = Mat3::from_diagonal(Vec3::new(2.0, 4.0, 8.0));
        assert_eq!(solve3(&k, b, trace(&k)), ldl3(&k, b, trace(&k)));
        assert!(
            solve3(&k, Vec3::new(f32::INFINITY, 0.0, 0.0), 14.0)
                .x
                .is_nan()
        );
        assert!(solve2(2.0, 0.0, 4.0, (f32::NAN, 1.0), 6.0).0.is_nan());
        // A single row's effective mass is a number too.
        assert_eq!(inverse(1e-45, 1e-45), MAX_IMPULSE);
    }

    #[test]
    fn a_pass_that_would_throw_a_light_body_beyond_any_speed_drops_the_joint() {
        // A ball joint whose anchors are 1e30 m apart: its soft correction asks a 1 kg body for
        // 1e31 m/s. That is a number, but no speed a body can have: the pass writes nothing and
        // the joint drops out for the step, as for one that is not a number.
        let body = SolverBody {
            inv_mass: Vec3::ONE,
            inv_inertia: Mat3::IDENTITY,
            ..SolverBody::STATIC
        };
        let start_vel = SolverVel {
            v: Vec3::new(1.0, 0.0, 0.0),
            w: Vec3::ZERO,
        };
        let bodies = vec![SolverBody::STATIC, body];
        let mut vels = vec![SolverVel::ZERO, start_vel];
        let id = BodyId {
            index: 0,
            generation: 0,
        };
        let desc = JointDesc {
            body_a: id,
            body_b: id,
            frame_a: Iso::IDENTITY,
            frame_b: Iso::IDENTITY,
            kind: JointKind::Ball,
            collide_connected: false,
            user_data: 0,
        };
        let ctx = context();
        let (fa, fb) = (frame(Vec3::ZERO, 0), frame(Vec3::new(1e30, 0.0, 0.0), 1));
        let mut c = prepare(&desc, &fa, &fb, &bodies, &ctx, JointImpulses::default());
        solve(&mut c, &desc, &bodies, &mut vels, &ctx, true);
        assert_eq!(vels[1].v, start_vel.v, "nothing written");
        assert_eq!(c.impulses, JointImpulses::default());
        warm_start(&c, &bodies, &mut vels);
        solve(&mut c, &desc, &bodies, &mut vels, &ctx, false);
        assert_eq!(vels[1].v, start_vel.v, "and nothing more this step");
    }
}
