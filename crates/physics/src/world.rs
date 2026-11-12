//! The physics world: bodies, colliders, joints and contacts, and the step that moves them.
//!
//! A step runs, in order: the broad phase (moved colliders find new overlapping pairs in two
//! bounding-volume trees, one for static colliders and one for everything that moves), the narrow
//! phase (every live pair's contact manifolds, events), the solver (soft step with substeps, see
//! [`crate::solver`]), sleeping (islands of touching bodies that have been still long enough stop
//! being simulated), the broad-phase boxes of the bodies that moved (so queries between steps
//! see every collider where it is), and continuous collision (fast bodies are swept against
//! static and kinematic geometry so they cannot tunnel through it). Its events are those it
//! produced and those of calls since the last step. Everything runs in index order with
//! deterministic tie breaks, so the same calls give the same results.

use std::collections::{BTreeMap, BTreeSet};

use crate::collide::{self, MAX_MANIFOLD_POINTS, Manifold};
use crate::gjk;
use crate::joint::{self, JointConstraint, JointDesc, JointImpulses};
use crate::math::{Aabb, Iso, Mat3, Quat, Vec3};
use crate::mesh::TriangleRef;
use crate::shape::{Convex, ConvexShape, MassProperties, Shape};
use crate::small_buf::SmallBuf;
use crate::solver::{
    self, BodyFrame, BodyInput, ConstraintPoint, ContactConstraint, IntegrationInputs, Softness,
    SolverBody, SolverVel, StepContext,
};
use crate::tree::DynamicTree;

macro_rules! handle {
    ($name:ident) => {
        /// A handle: an index and the generation it was created in, so a handle to something that
        /// was destroyed is never mistaken for whatever reuses its slot.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name {
            pub index: u32,
            pub generation: u32,
        }
    };
}
handle!(BodyId);
handle!(ColliderId);
handle!(JointId);

/// The lightest a dynamic body may be (kg). Lighter colliders (a tiny density) make a body of
/// this mass with the same shape, so its inverse mass stays a number: below about 3e-39 kg it
/// overflowed, and the first step turned the body into NaN.
pub const MIN_BODY_MASS: f32 = 1e-20;

/// The heaviest a dynamic body may be (kg). Heavier colliders make a body of this mass with the
/// same shape: next to anything else in a world it is immovable either way (the densest real
/// material in the largest box a shape may have weighs about 2e20 kg), and momentum (mass ×
/// speed, what the solver's impulses are made of) stays a number up to 1e13 m/s; a cube of
/// 2.4e38 kg overflowed its first contact, and turned what it touched into NaN.
pub const MAX_BODY_MASS: f32 = 1e25;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyKind {
    /// Never moves (level geometry). Infinite mass.
    Static,
    /// Moves only as its velocity says; pushes dynamic bodies, is never pushed. Infinite mass.
    Kinematic,
    /// Moved by forces, contacts and joints.
    Dynamic,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BodyDesc {
    pub kind: BodyKind,
    /// Pose of the body's origin (colliders are placed relative to it).
    pub position: Vec3,
    pub rotation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    /// Fraction of velocity lost per second, roughly (`v /= 1 + h·damping` each substep).
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub gravity_scale: f32,
    pub can_sleep: bool,
    pub awake: bool,
    /// Swept against static and kinematic geometry every step, however slowly it moves.
    pub bullet: bool,
    /// World axes a dynamic body may not turn about (for example `[true, false, true]` keeps a
    /// character upright): its angular velocity about them is always zero, whatever it is given.
    /// Kinematic bodies move as they are told.
    pub lock_rotation: [bool; 3],
    /// World axes a dynamic body may not move along (its velocity along them is always zero).
    pub lock_translation: [bool; 3],
    pub user_data: u64,
}

impl Default for BodyDesc {
    fn default() -> Self {
        Self {
            kind: BodyKind::Dynamic,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            linear_damping: 0.0,
            angular_damping: 0.05,
            gravity_scale: 1.0,
            can_sleep: true,
            awake: true,
            bullet: false,
            lock_rotation: [false; 3],
            lock_translation: [false; 3],
            user_data: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ColliderDesc {
    pub shape: Shape,
    /// Placement on the body, relative to the body origin.
    pub local: Iso,
    /// kg/m³ (water is 1000).
    pub density: f32,
    pub friction: f32,
    pub restitution: f32,
    /// Detects overlaps (sensor events) without colliding.
    pub sensor: bool,
    /// The layers this collider is on.
    pub layers: u32,
    /// The layers it collides with. Two colliders interact when each one's layers meet the other's
    /// mask.
    pub mask: u32,
    pub user_data: u64,
}

impl ColliderDesc {
    pub fn new(shape: Shape) -> Self {
        Self {
            shape,
            local: Iso::IDENTITY,
            density: 1000.0,
            friction: 0.6,
            restitution: 0.0,
            sensor: false,
            layers: 1,
            mask: u32::MAX,
            user_data: 0,
        }
    }
}

/// World-wide tuning. The defaults suit metre-scale objects at 60 steps per second.
///
/// The fields are public and can change between steps. A value outside its range (finite
/// unless noted, never negative, at least one substep, a top speed above zero) takes its
/// default at the start of the next step ([`Settings::sanitized`]), so a typo cannot freeze,
/// reverse or poison the world.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub gravity: Vec3,
    /// Substeps per step. Changing it between steps rescales the impulses carried over from
    /// the last step, which are per substep.
    pub substeps: u32,
    /// Stiffness of contact position correction; clamped to a quarter of the substep rate.
    pub contact_hertz: f32,
    pub contact_damping_ratio: f32,
    pub joint_hertz: f32,
    pub joint_damping_ratio: f32,
    /// Fastest speed at which overlapping bodies are pushed apart (m/s; may be infinite).
    pub contact_push_max_velocity: f32,
    /// Approach speed below which contacts do not bounce (m/s; may be infinite).
    pub restitution_threshold: f32,
    /// Approach speed above which a contact reports a hit event (m/s; may be infinite).
    pub hit_event_threshold: f32,
    /// A distance too small to matter (m): the gap continuous collision leaves when it stops a
    /// fast body (its time-of-impact target), and, added to twice the speculative distance, how
    /// far a contact point may move between steps and keep its impulse. It is not an allowed
    /// overlap: as in Box2D v3's soft step, contacts push overlapping bodies apart however
    /// little they overlap (softly, so stacks rest without jitter or sinking).
    pub linear_slop: f32,
    /// How far apart contact points are still produced, so approaching bodies stop exactly when
    /// they touch (m).
    pub speculative_distance: f32,
    /// How far a collider may move before its broad-phase box is rebuilt (m).
    pub aabb_margin: f32,
    /// The fastest a body may move (m/s; above zero, may be infinite).
    pub max_linear_speed: f32,
    pub enable_sleep: bool,
    /// How long a body (with everything it touches) must stay still before it sleeps (s; may be
    /// infinite). Zero puts bodies to sleep after one still step.
    pub time_to_sleep: f32,
    /// Speed below which a body counts as still: its centre's speed, and the speed of its
    /// farthest point due to turning (m/s). Zero keeps every body awake.
    pub sleep_threshold: f32,
    pub enable_continuous: bool,
}

impl Settings {
    /// These settings with every value outside its range replaced by its default.
    pub fn sanitized(&self) -> Settings {
        let d = Settings::default();
        // Finite and not negative; not negative (infinity allowed, NaN not).
        let size = |v: f32, default: f32| {
            if v.is_finite() && v >= 0.0 {
                v
            } else {
                default
            }
        };
        let limit = |v: f32, default: f32| if v >= 0.0 { v } else { default };
        // Above zero (infinity allowed): a top speed of zero holds every body where it is.
        let positive = |v: f32, default: f32| if v > 0.0 { v } else { default };
        Settings {
            gravity: if self.gravity.is_finite() {
                self.gravity
            } else {
                d.gravity
            },
            substeps: if self.substeps >= 1 {
                self.substeps
            } else {
                d.substeps
            },
            contact_hertz: size(self.contact_hertz, d.contact_hertz),
            contact_damping_ratio: size(self.contact_damping_ratio, d.contact_damping_ratio),
            joint_hertz: size(self.joint_hertz, d.joint_hertz),
            joint_damping_ratio: size(self.joint_damping_ratio, d.joint_damping_ratio),
            contact_push_max_velocity: limit(
                self.contact_push_max_velocity,
                d.contact_push_max_velocity,
            ),
            restitution_threshold: limit(self.restitution_threshold, d.restitution_threshold),
            hit_event_threshold: limit(self.hit_event_threshold, d.hit_event_threshold),
            linear_slop: size(self.linear_slop, d.linear_slop),
            speculative_distance: size(self.speculative_distance, d.speculative_distance),
            aabb_margin: size(self.aabb_margin, d.aabb_margin),
            max_linear_speed: positive(self.max_linear_speed, d.max_linear_speed),
            enable_sleep: self.enable_sleep,
            time_to_sleep: limit(self.time_to_sleep, d.time_to_sleep),
            // Finite: at infinity every body counts as still, and a falling one sleeps in
            // mid-air (zero is fine: nothing is ever still enough to sleep).
            sleep_threshold: size(self.sleep_threshold, d.sleep_threshold),
            enable_continuous: self.enable_continuous,
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            substeps: 4,
            contact_hertz: 30.0,
            contact_damping_ratio: 10.0,
            joint_hertz: 60.0,
            joint_damping_ratio: 2.0,
            contact_push_max_velocity: 3.0,
            restitution_threshold: 1.0,
            hit_event_threshold: 1.0,
            linear_slop: 0.005,
            speculative_distance: 0.02,
            aabb_margin: 0.05,
            max_linear_speed: 400.0,
            enable_sleep: true,
            time_to_sleep: 0.5,
            sleep_threshold: 0.05,
            enable_continuous: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContactEvent {
    /// Two colliders started touching (their manifold gained points).
    Begin { a: ColliderId, b: ColliderId },
    /// They stopped touching, or one was removed.
    End { a: ColliderId, b: ColliderId },
    /// They met faster than `hit_event_threshold`: where, along which normal (from `a` to `b`)
    /// and how fast they approached.
    Hit {
        a: ColliderId,
        b: ColliderId,
        point: Vec3,
        normal: Vec3,
        speed: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SensorEvent {
    Begin {
        sensor: ColliderId,
        visitor: ColliderId,
    },
    End {
        sensor: ColliderId,
        visitor: ColliderId,
    },
}

#[derive(Clone, Debug, Default)]
pub struct Events {
    pub contacts: Vec<ContactEvent>,
    pub sensors: Vec<SensorEvent>,
}

/// Which colliders a query sees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryFilter {
    /// The query's own layers: a collider is seen when its mask meets these.
    pub layers: u32,
    /// The layers the query sees: a collider is seen when its layers meet this.
    pub mask: u32,
    pub exclude_body: Option<BodyId>,
    pub include_sensors: bool,
}

impl Default for QueryFilter {
    fn default() -> Self {
        Self {
            layers: u32::MAX,
            mask: u32::MAX,
            exclude_body: None,
            include_sensors: false,
        }
    }
}

#[derive(Clone)]
pub(crate) struct Body {
    pub generation: u32,
    pub alive: bool,
    pub kind: BodyKind,
    pub position: Vec3,
    pub rotation: Quat,
    pub local_center: Vec3,
    pub center: Vec3,
    /// Pose at the start of the last step (for continuous collision).
    pub center0: Vec3,
    pub rotation0: Quat,
    /// The orientation the body was created or last teleported at: a dynamic body that may turn
    /// about one world axis only can reach no orientation but this one turned about that axis
    /// (see [`Body::reachable`]).
    pub rest_rotation: Quat,
    /// Changes on each explicit teleport so riders can distinguish it from integrated motion.
    pub teleport_revision: u64,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub force: Vec3,
    pub torque: Vec3,
    pub mass: f32,
    pub inv_mass: f32,
    pub local_inertia: Mat3,
    pub inv_local_inertia: Mat3,
    /// Farthest any solid collider reaches from the centre of mass (for the sleep test).
    pub max_extent: f32,
    /// The thinnest solid collider's inner radius (for continuous collision).
    pub min_extent: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub gravity_scale: f32,
    pub can_sleep: bool,
    pub awake: bool,
    pub sleep_time: f32,
    pub bullet: bool,
    pub lock_rotation: [bool; 3],
    pub lock_translation: [bool; 3],
    pub user_data: u64,
    pub colliders: Vec<u32>,
    pub joints: Vec<u32>,
}

impl Body {
    pub fn iso(&self) -> Iso {
        Iso::new(self.position, self.rotation)
    }
    fn is_dynamic(&self) -> bool {
        self.kind == BodyKind::Dynamic
    }
    fn moves(&self) -> bool {
        self.kind != BodyKind::Static
    }
    /// A kinematic body with a velocity: it carries what touches it or hangs from it along.
    fn is_kinematic_moving(&self) -> bool {
        self.kind == BodyKind::Kinematic
            && (self.linear_velocity != Vec3::ZERO || self.angular_velocity != Vec3::ZERO)
    }
    /// Whether the body, awake, can disturb what it touches: a dynamic body, or a moving
    /// kinematic one.
    fn pushes(&self) -> bool {
        self.is_dynamic() || self.is_kinematic_moving()
    }
    /// Orientation `q` of this body brought back onto those it can reach, when it is a dynamic
    /// body that may turn about one world axis only: its rest orientation turned about that axis,
    /// by the angle of `q`'s turn from it about the axis. Each step's turn is exactly about the
    /// axis, but applying it to the orientation rounds, and the roundings added up (some 1e-5 rad
    /// a minute; joints read a body that far off its axis as able to turn about the locked ones).
    /// Any other body's `q` is its own.
    fn reachable(&self, q: Quat) -> Quat {
        let axis = match self.lock_rotation {
            [false, true, true] => Vec3::X,
            [true, false, true] => Vec3::Y,
            [true, true, false] => Vec3::Z,
            _ => return q,
        };
        if self.kind != BodyKind::Dynamic {
            return q;
        }
        // The turn from the rest orientation, less any part about another axis.
        let turn = q * self.rest_rotation.conjugate();
        let s = turn.vector().dot(axis);
        let twist = Quat::new(axis.x * s, axis.y * s, axis.z * s, turn.w).normalize();
        (twist * self.rest_rotation).normalize()
    }
    /// 1 on the world axes a dynamic body may turn about, 0 on the locked ones.
    fn rotation_mask(&self) -> Vec3 {
        mask(self.lock_rotation)
    }
    /// 1 on the world axes a dynamic body may move along, 0 on the locked ones.
    fn translation_mask(&self) -> Vec3 {
        mask(self.lock_translation)
    }
    /// The velocities a body can have: a dynamic body's are zero on its locked axes.
    fn allowed_velocities(&self, v: Vec3, w: Vec3) -> (Vec3, Vec3) {
        if self.is_dynamic() {
            (
                v.mul_elem(self.translation_mask()),
                w.mul_elem(self.rotation_mask()),
            )
        } else {
            (v, w)
        }
    }
    /// The world inverse inertia: how an angular impulse turns the body. With some rotation axes
    /// locked, the body turns only about the free ones, so the free axes' response is the inverse
    /// of the free block of the world inertia (not that block of the full inverse, which is
    /// larger wherever the locked axes are coupled to the free ones).
    fn world_inv_inertia(&self) -> Mat3 {
        if self.kind != BodyKind::Dynamic {
            return Mat3::ZERO;
        }
        match self.lock_rotation {
            [false, false, false] => self.inv_local_inertia.rotated(self.rotation),
            [true, true, true] => Mat3::ZERO,
            _ => free_block_inverse(
                &self.local_inertia.rotated(self.rotation),
                self.rotation_mask(),
            ),
        }
    }
    /// How this body's world inverse inertia follows it as it turns within a step.
    fn turning(&self) -> Turning {
        if self.kind != BodyKind::Dynamic {
            return Turning::Fixed;
        }
        match self.lock_rotation {
            [false, false, false] => Turning::Free(self.inv_local_inertia.rotated(self.rotation)),
            // Turning only about one world axis leaves that axis's inertia as it is.
            [true, true, true]
            | [false, true, true]
            | [true, false, true]
            | [true, true, false] => Turning::Fixed,
            _ => Turning::Locked(
                self.local_inertia.rotated(self.rotation),
                self.rotation_mask(),
            ),
        }
    }
    fn inv_mass_vector(&self) -> Vec3 {
        if self.kind != BodyKind::Dynamic {
            return Vec3::ZERO;
        }
        Vec3::new(
            if self.lock_translation[0] {
                0.0
            } else {
                self.inv_mass
            },
            if self.lock_translation[1] {
                0.0
            } else {
                self.inv_mass
            },
            if self.lock_translation[2] {
                0.0
            } else {
                self.inv_mass
            },
        )
    }
}

/// Paths of translating kinematic colliders through this step. The ordinary broad phase only
/// needs their final extents for overlap queries and the next step's discrete contacts.
#[derive(Default)]
struct KinematicSweeps {
    tree: DynamicTree<u32>,
    travel: BTreeMap<u32, Vec3>,
    sleepers: BTreeSet<u32>,
}

type HitEventIndex = Option<BTreeMap<(u32, u32), usize>>;

struct SweepInterval {
    center: Vec3,
    travel: Vec3,
    remaining: f32,
    after_hit: bool,
}

#[derive(Clone, Copy)]
struct CcdHit {
    impact: Impact,
    moving: u32,
    collider: u32,
    target_travel: Vec3,
}

/// Cancel only the part closing on a surface, using the dynamic body's free translation axes.
/// A moving obstacle can push a body outward without towing it along the face.
fn remove_closing(motion: &mut Vec3, target: Vec3, normal: Vec3, mask: Vec3) {
    let free = normal.mul_elem(mask);
    let weight = free.length_squared();
    let closing = (target - *motion).dot(normal);
    if closing > 0.0 && weight >= 1e-12 {
        *motion += free * (closing / weight);
    }
}

/// What continuous collision takes from one sweep (see [`PhysicsWorld::continuous`]).
#[derive(Clone, Copy, Debug)]
struct Impact {
    fraction: f32,
    normal: Vec3,
    point: Vec3,
    /// A shape already within tolerance was swept using its inner core.
    core: bool,
}

impl Impact {
    fn cast(hit: gjk::CastHit, core: bool) -> Self {
        Self {
            fraction: hit.fraction,
            normal: hit.normal,
            point: hit.point,
            core,
        }
    }
}

/// How `moving` (with its `core`) meets `other` when moved by `d`, when that is a time of impact
/// continuous collision should take. An initial touch gives way to the same cast of the core;
/// inside a step, contacts reached simultaneously may close at zero in the remaining interval.
fn time_of_impact(
    moving: &Convex,
    core: &Convex,
    d: Vec3,
    other: &Convex,
    target: f32,
    after_hit: bool,
) -> Option<Impact> {
    let hit = gjk::cast(moving, d, other, target)?;
    // A second face reached at the same time as the first can already touch at the
    // start of the remaining interval. Keep it if still closing; waiting for the core
    // would let the body sink part-way through the second face.
    if hit.fraction > 0.0 || (after_hit && d.dot(hit.normal) < -1e-5 * d.length()) {
        return Some(Impact::cast(hit, false));
    }
    let hit = gjk::cast(core, d, other, target)?;
    (hit.fraction > 0.0).then(|| Impact::cast(hit, true))
}

/// Where a centre moving from `c0` by `d` passes through one-sided triangle `t` (placed at `at`)
/// from its front, as an impact that keeps the centre `keep` in front of the plane (or where it
/// is, when it starts closer). The narrow phase drops a triangle whose plane a shape's centre is
/// behind, so a centre through it is a body through it. The sweeps miss this when the core
/// starts within their target of the triangle (a body wedged against a surface, whose core a
/// sweep stopped there the step before, pushed on by friction): their time of impact is zero,
/// which does not count. Watertight ([`crate::mesh::ray_triangle`]), so a centre crossing on a
/// seam passes through one of the triangles sharing it.
fn centre_through(t: &TriangleRef, at: &Iso, c0: Vec3, d: Vec3, keep: f32) -> Option<Impact> {
    let (o, dir) = (at.inverse_point(c0), at.inverse_vector(d));
    let cross = crate::mesh::ray_triangle(o, dir, &t.vertices, 1.0, false)?;
    let n = t.normal();
    // How much nearer the plane the whole motion brings the centre.
    let closing = -n.dot(dir);
    (closing > 0.0).then(|| Impact {
        fraction: (cross - keep / closing).max(0.0),
        normal: at.vector(n),
        point: c0 + d * cross,
        core: true,
    })
}

/// The mass properties of colliders (as a function of a scale on every density) with the mass
/// brought into [`MIN_BODY_MASS`]..[`MAX_BODY_MASS`]: out of range, the same shapes are weighed
/// again with every density scaled alike, so the centre and the shape of the inertia stay (and
/// a mass so small that its inertia underflowed gets a usable one). `None` when the mass
/// overflows `f32`, or the centre or inertia is not a number.
fn bounded_mass(parts: impl Fn(f32) -> Vec<MassProperties>) -> Option<MassProperties> {
    let mut total = MassProperties::combine(&parts(1.0));
    if !total.mass.is_finite() {
        return None;
    }
    if total.mass > 0.0 && !(MIN_BODY_MASS..=MAX_BODY_MASS).contains(&total.mass) {
        let k = total.mass.clamp(MIN_BODY_MASS, MAX_BODY_MASS) / total.mass;
        total = MassProperties::combine(&parts(k));
    }
    (total.center.is_finite() && total.inertia.is_finite()).then_some(total)
}

/// Manifolds whose points may hand impulses to each other have normals within about 25°.
const CARRY_NORMAL_COS: f32 = 0.9;

/// This step's manifolds with the impulses carried over from last step's points, each old point
/// to at most one new point (several new points sharing one old impulse would warm-start with a
/// multiple of what held the bodies). A point with a feature id first takes the old point with
/// the same id and a compatible normal. Each remaining point then takes the nearest unclaimed
/// old point within `match_distance` in collider A's frame whose normal agrees with its own.
/// The closest pairs claim first, with ties in index order.
fn carry_impulses_into(
    old: &[ContactManifold],
    fresh: &[Manifold],
    iso_a: &Iso,
    match_distance: f32,
    out: &mut Vec<ContactManifold>,
) {
    out.clear();
    out.reserve(fresh.len());
    for m in fresh {
        let mut cm = ContactManifold {
            manifold: m.clone(),
            cache: Default::default(),
        };
        for (i, p) in m.points().iter().enumerate() {
            cm.cache[i] = CachedPoint {
                id: p.id,
                local_a: iso_a.inverse_point(p.point),
                normal_impulse: 0.0,
                friction_impulse: Vec3::ZERO,
            };
        }
        out.push(cm);
    }
    if old.is_empty() {
        return;
    }
    // Last step's points, flattened: (manifold, point).
    let mut olds = SmallBuf::<(usize, usize), 32>::new((0, 0));
    let mut claimed = SmallBuf::<bool, 32>::new(false);
    for (om, m) in old.iter().enumerate() {
        for op in 0..usize::from(m.manifold.count) {
            olds.push((om, op));
            claimed.push(false);
        }
    }
    let mut filled = SmallBuf::<[bool; MAX_MANIFOLD_POINTS], 8>::new([false; MAX_MANIFOLD_POINTS]);
    for _ in 0..out.len() {
        filled.push([false; MAX_MANIFOLD_POINTS]);
    }
    let give = |cm: &mut ContactManifold, i: usize, from: &CachedPoint| {
        cm.cache[i].normal_impulse = from.normal_impulse;
        cm.cache[i].friction_impulse = from.friction_impulse;
    };
    let olds_slice = olds.as_slice();
    let claimed_slice = claimed.as_mut_slice();
    let filled_slice = filled.as_mut_slice();
    // Feature ids: the same feature with a compatible surface normal, once.
    for (cm, done) in out.iter_mut().zip(filled_slice.iter_mut()) {
        for (i, done) in done
            .iter_mut()
            .enumerate()
            .take(usize::from(cm.manifold.count))
        {
            let id = cm.cache[i].id;
            if id == 0 {
                continue;
            }
            let found = olds_slice.iter().enumerate().find(|&(o, &(om, op))| {
                !claimed_slice[o]
                    && old[om].cache[op].id == id
                    && old[om].manifold.normal.dot(cm.manifold.normal) >= CARRY_NORMAL_COS
            });
            if let Some((o, &(om, op))) = found {
                claimed_slice[o] = true;
                *done = true;
                give(cm, i, &old[om].cache[op]);
            }
        }
    }
    // Every point still without an impulse, by position, nearest pairs first: points without a
    // feature id, and points whose id changed (in a stack of equal aligned boxes the incident
    // vertices lie on the reference face's side planes, so their clip ids flicker from step to
    // step; matched by id only, those points restarted from nothing every step and the stack
    // took two to three times as long to settle).
    let mut pairs = SmallBuf::<(f32, usize, usize, usize), 32>::new((0.0, 0, 0, 0));
    for (nm, (cm, done)) in out.iter().zip(&*filled_slice).enumerate() {
        for (i, &done) in done.iter().enumerate().take(usize::from(cm.manifold.count)) {
            if done {
                continue;
            }
            for (o, &(om, op)) in olds_slice.iter().enumerate() {
                let from = &old[om].cache[op];
                if claimed_slice[o]
                    || old[om].manifold.normal.dot(cm.manifold.normal) < CARRY_NORMAL_COS
                {
                    continue;
                }
                let d = (from.local_a - cm.cache[i].local_a).length();
                if d <= match_distance {
                    pairs.push((d, nm, i, o));
                }
            }
        }
    }
    pairs.as_mut_slice().sort_by(|x, y| {
        x.0.total_cmp(&y.0)
            .then(x.1.cmp(&y.1))
            .then(x.2.cmp(&y.2))
            .then(x.3.cmp(&y.3))
    });
    for &(_, nm, i, o) in pairs.as_slice() {
        if claimed_slice[o] || filled_slice[nm][i] {
            continue;
        }
        claimed_slice[o] = true;
        filled_slice[nm][i] = true;
        let (om, op) = olds_slice[o];
        give(&mut out[nm], i, &old[om].cache[op]);
    }
}

/// The inverse inertia of a body free to turn only about the world axes where `free` is 1, with
/// world inertia `iw`: the inverse of the free block of `iw` (not that block of the full inverse,
/// which is larger wherever the locked axes are coupled to the free ones), zero on the locked
/// rows and columns.
fn free_block_inverse(iw: &Mat3, free: Vec3) -> Mat3 {
    let mask = Mat3::from_diagonal(free);
    // Locked axes get a diagonal of the free block's size, so the padded matrix keeps the scale
    // of the inertia (block diagonal: its inverse holds the free block's inverse, and the padding
    // is masked away).
    let size = iw.diagonal().mul_elem(free).max_element();
    let pad = Mat3::from_diagonal((Vec3::ONE - free) * size);
    mask * (mask * *iw * mask + pad).inverse_or_zero() * mask
}

/// How a solver body's world inverse inertia follows the body as it turns within a step, so that
/// what the joints (whose anchors turn with the bodies) push it with is its inertia as it is now:
/// with the inertia of the start of the step, a very thin rod (2 m by 2 mm) hinged across itself
/// turned far too easily about its own axis once it had swung a little, and fell off its hinge.
#[derive(Clone, Copy, Debug)]
enum Turning {
    /// It does not change: a body that is not dynamic, may not turn, or may turn about one world
    /// axis only (about which its inertia stays the same).
    Fixed,
    /// A body free to turn every way: its world inverse inertia at the start of the step, turned
    /// by how far the body has turned since.
    Free(Mat3),
    /// A body with one rotation axis locked: its world inertia at the start of the step and its
    /// rotation mask, the free block inverted afresh once turned (the locked rows stay zero).
    Locked(Mat3, Vec3),
}

impl Turning {
    /// The world inverse inertia after turning by `dq`, or `None` when it stays as it was.
    fn after(&self, dq: Quat) -> Option<Mat3> {
        match self {
            Turning::Fixed => None,
            Turning::Free(m) => Some(m.rotated(dq)),
            Turning::Locked(iw, free) => Some(free_block_inverse(&iw.rotated(dq), *free)),
        }
    }
}

/// 1 where an axis is free, 0 where it is locked.
fn mask(locks: [bool; 3]) -> Vec3 {
    Vec3::new(
        if locks[0] { 0.0 } else { 1.0 },
        if locks[1] { 0.0 } else { 1.0 },
        if locks[2] { 0.0 } else { 1.0 },
    )
}

#[derive(Clone)]
pub(crate) struct Collider {
    pub generation: u32,
    pub alive: bool,
    pub body: u32,
    pub shape: Shape,
    pub local: Iso,
    pub density: f32,
    pub friction: f32,
    pub restitution: f32,
    pub sensor: bool,
    pub layers: u32,
    pub mask: u32,
    pub user_data: u64,
    pub proxy: u32,
    /// In the static tree (true) or the moving tree.
    pub in_static_tree: bool,
    /// The box in the broad phase: the collider's bounds, grown by the margin.
    pub fat: Aabb,
    /// The contacts this collider is in (indices into the world's contacts, in no set order);
    /// each contact knows its place in its two colliders' lists, so it leaves them in O(1).
    /// What touches a collider or a body is found here, never by scanning every contact.
    pub contacts: Vec<u32>,
    /// Its index is in the move buffer (kept by a destroyed collider until the next broad phase,
    /// so a collider reusing the slot is not queued twice).
    pub queued: bool,
}

/// Impulses remembered for one manifold point between steps.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CachedPoint {
    pub id: u32,
    /// Where the point was, in collider A's frame (for matching points without ids).
    pub local_a: Vec3,
    pub normal_impulse: f32,
    pub friction_impulse: Vec3,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ContactManifold {
    pub manifold: Manifold,
    pub cache: [CachedPoint; MAX_MANIFOLD_POINTS],
}

#[derive(Clone)]
pub(crate) struct Contact {
    /// Collider indices, `a < b`.
    pub a: u32,
    pub b: u32,
    pub manifolds: Vec<ContactManifold>,
    pub touching: bool,
    pub sensor: bool,
    /// Where this contact is in collider `a`'s and collider `b`'s contact lists.
    pub slots: [u32; 2],
}

impl Contact {
    /// The collider at the other end from collider `c` (one of the two).
    fn other(&self, c: u32) -> u32 {
        if self.a == c { self.b } else { self.a }
    }
}

#[derive(Clone)]
pub(crate) struct JointSlot {
    pub generation: u32,
    pub alive: bool,
    pub desc: JointDesc,
    pub impulses: JointImpulses,
}

/// Reusable scratch buffers across steps so steady-state simulation performs zero per-step heap
/// allocations; cloning a [`PhysicsWorld`] starts the clone with empty scratch buffers.
#[derive(Default)]
pub(crate) struct StepScratch {
    broad_found: Vec<(u32, u32)>,
    narrow_manifolds: Vec<Manifold>,
    old_manifolds: Vec<ContactManifold>,
    wake: Vec<u32>,
    active_indices: Vec<u32>,
    slot: Vec<usize>,
    solver_bodies: Vec<SolverBody>,
    solver_vels: Vec<SolverVel>,
    per_body: Vec<BodyInput>,
    turning: Vec<Turning>,
    members: Vec<u32>,
    constraints: Vec<ContactConstraint>,
    joint_constraints: Vec<(u32, JointConstraint)>,
    hits: Vec<(u32, f32, Vec3, Vec3)>,
    sleep_kept: Vec<u32>,
    sleep_visited: Vec<u32>,
    sleep_stamp: u32,
    sleep_stack: Vec<u32>,
    sleep_island: Vec<u32>,
}

impl Clone for StepScratch {
    fn clone(&self) -> Self {
        Self::default()
    }
}

/// A rigid-body simulation. Cloning gives an independent copy (to restart from a saved state).
#[derive(Clone)]
pub struct PhysicsWorld {
    pub settings: Settings,
    pub(crate) bodies: Vec<Body>,
    free_bodies: Vec<u32>,
    /// Bitset of body indices where `alive && awake && moves()` holds (64 bodies per `u64` word).
    awake_words: Vec<u64>,
    /// Count of live `BodyKind::Kinematic` bodies.
    kinematic_count: usize,
    pub(crate) colliders: Vec<Collider>,
    free_colliders: Vec<u32>,
    pub(crate) joints: Vec<JointSlot>,
    free_joints: Vec<u32>,
    pub(crate) contacts: Vec<Contact>,
    pairs: BTreeMap<(u32, u32), u32>,
    pub(crate) static_tree: DynamicTree<u32>,
    pub(crate) moving_tree: DynamicTree<u32>,
    move_buffer: Vec<u32>,
    /// The events of the last step.
    events: Events,
    /// Events for the next step's list: the step in progress adds to them, and so do calls
    /// between steps that end contacts (removing a body or collider, a joint that turns
    /// collisions off), so a game reading the events after each step sees those too.
    pending: Events,
    /// The substep count the carried-over impulses are for (0 before the first step).
    warm_substeps: u32,
    steps: u64,
    scratch: StepScratch,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new(Settings::default())
    }
}

impl PhysicsWorld {
    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            bodies: Vec::new(),
            free_bodies: Vec::new(),
            awake_words: Vec::new(),
            kinematic_count: 0,
            colliders: Vec::new(),
            free_colliders: Vec::new(),
            joints: Vec::new(),
            free_joints: Vec::new(),
            contacts: Vec::new(),
            pairs: BTreeMap::new(),
            static_tree: DynamicTree::new(),
            moving_tree: DynamicTree::new(),
            move_buffer: Vec::new(),
            events: Events::default(),
            pending: Events::default(),
            warm_substeps: 0,
            steps: 0,
            scratch: StepScratch::default(),
        }
    }

    #[inline]
    fn set_awake_bit(&mut self, index: u32, awake_moving: bool) {
        let i = index as usize;
        let word = i >> 6;
        let mask = 1u64 << (i & 63);
        if awake_moving {
            if word >= self.awake_words.len() {
                self.awake_words.resize(word + 1, 0);
            }
            self.awake_words[word] |= mask;
        } else if let Some(w) = self.awake_words.get_mut(word) {
            *w &= !mask;
        }
    }

    #[inline]
    fn has_awake_moving_bodies(&self) -> bool {
        self.awake_words.iter().any(|&w| w != 0)
    }

    /// Appends every live, awake, moving body index to `out` in ascending body index order.
    fn collect_awake_moving(&self, out: &mut Vec<u32>) {
        out.clear();
        for (wi, &word) in self.awake_words.iter().enumerate() {
            let mut w = word;
            let base = (wi << 6) as u32;
            while w != 0 {
                let bit = w.trailing_zeros();
                out.push(base + bit);
                w &= w - 1;
            }
        }
    }

    // ---------------------------------------------------------------- bodies

    pub fn create_body(&mut self, desc: &BodyDesc) -> Result<BodyId, String> {
        if !desc.position.is_finite()
            || !desc.rotation.is_finite()
            || !desc.linear_velocity.is_finite()
            || !desc.angular_velocity.is_finite()
        {
            return Err("body pose or velocity is not finite".into());
        }
        if !desc.gravity_scale.is_finite() {
            return Err(format!(
                "gravity scale {} is not finite",
                desc.gravity_scale
            ));
        }
        let rotation = desc.rotation.normalize();
        let dynamic_or_kinematic = desc.kind != BodyKind::Static;
        let mut body = Body {
            generation: 0,
            alive: true,
            kind: desc.kind,
            position: desc.position,
            rotation,
            local_center: Vec3::ZERO,
            center: desc.position,
            center0: desc.position,
            rotation0: rotation,
            rest_rotation: rotation,
            teleport_revision: 0,
            linear_velocity: if dynamic_or_kinematic {
                desc.linear_velocity
            } else {
                Vec3::ZERO
            },
            angular_velocity: if dynamic_or_kinematic {
                desc.angular_velocity
            } else {
                Vec3::ZERO
            },
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
            mass: 0.0,
            inv_mass: 0.0,
            local_inertia: Mat3::ZERO,
            inv_local_inertia: Mat3::ZERO,
            max_extent: 0.0,
            min_extent: 0.0,
            linear_damping: desc.linear_damping.max(0.0),
            angular_damping: desc.angular_damping.max(0.0),
            gravity_scale: desc.gravity_scale,
            can_sleep: desc.can_sleep,
            awake: desc.awake && desc.kind != BodyKind::Static,
            sleep_time: 0.0,
            bullet: desc.bullet,
            lock_rotation: desc.lock_rotation,
            lock_translation: desc.lock_translation,
            user_data: desc.user_data,
            colliders: Vec::new(),
            joints: Vec::new(),
        };
        (body.linear_velocity, body.angular_velocity) =
            body.allowed_velocities(body.linear_velocity, body.angular_velocity);
        let index = if let Some(i) = self.free_bodies.pop() {
            let generation = self.bodies[i as usize].generation + 1;
            self.bodies[i as usize] = Body { generation, ..body };
            i
        } else {
            self.bodies.push(body);
            (self.bodies.len() - 1) as u32
        };
        let id = BodyId {
            index,
            generation: self.bodies[index as usize].generation,
        };
        if desc.kind == BodyKind::Dynamic {
            // A dynamic body without colliders still needs mass to move.
            self.update_mass(index);
        } else if desc.kind == BodyKind::Kinematic {
            self.kinematic_count += 1;
        }
        let awake_moving = self.bodies[index as usize].awake && self.bodies[index as usize].moves();
        self.set_awake_bit(index, awake_moving);
        Ok(id)
    }

    fn body_index(&self, id: BodyId) -> Option<usize> {
        let b = self.bodies.get(id.index as usize)?;
        (b.alive && b.generation == id.generation).then_some(id.index as usize)
    }

    pub fn contains_body(&self, id: BodyId) -> bool {
        self.body_index(id).is_some()
    }

    /// Removes a body with its colliders and joints; bodies touching it wake up.
    pub fn destroy_body(&mut self, id: BodyId) {
        let Some(i) = self.body_index(id) else { return };
        self.wake_neighbours(i as u32);
        for j in self.bodies[i].joints.clone() {
            self.destroy_joint_index(j);
        }
        for c in self.bodies[i].colliders.clone() {
            self.destroy_collider_index(c);
        }
        if self.bodies[i].kind == BodyKind::Kinematic {
            self.kinematic_count = self.kinematic_count.saturating_sub(1);
        }
        self.bodies[i].alive = false;
        self.set_awake_bit(i as u32, false);
        self.free_bodies.push(i as u32);
    }

    /// Live bodies in index order.
    pub fn bodies(&self) -> impl Iterator<Item = BodyId> + '_ {
        self.bodies
            .iter()
            .enumerate()
            .filter(|(_, b)| b.alive)
            .map(|(i, b)| BodyId {
                index: i as u32,
                generation: b.generation,
            })
    }

    pub fn body_count(&self) -> usize {
        self.bodies.iter().filter(|b| b.alive).count()
    }

    pub fn awake_body_count(&self) -> usize {
        self.awake_words
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum()
    }

    pub(crate) fn body(&self, id: BodyId) -> Option<&Body> {
        self.body_index(id).map(|i| &self.bodies[i])
    }

    /// The pose of the body's origin.
    pub fn body_pose(&self, id: BodyId) -> Option<Iso> {
        self.body(id).map(Body::iso)
    }

    /// Teleports a body (and wakes it). Contacts are found again from the new place. What
    /// touched it or hung from it where it was wakes up, and so does what a static body now
    /// overlaps (a moving body, awake, wakes what it touches by itself).
    pub fn set_body_pose(&mut self, id: BodyId, pose: Iso) {
        let Some(i) = self.body_index(id) else { return };
        if !pose.position.is_finite() || !pose.rotation.is_finite() {
            return;
        }
        self.wake_neighbours(i as u32);
        let b = &mut self.bodies[i];
        b.position = pose.position;
        b.rotation = pose.rotation.normalize();
        b.center = b.position + b.rotation.rotate(b.local_center);
        b.center0 = b.center;
        b.rotation0 = b.rotation;
        b.rest_rotation = b.rotation;
        b.teleport_revision = b.teleport_revision.wrapping_add(1);
        self.refresh_body_proxies(i as u32, true);
        self.wake_body(i as u32);
        if self.bodies[i].kind == BodyKind::Static {
            let colliders = self.bodies[i].colliders.clone();
            self.wake_overlapping(&colliders);
        }
    }

    pub fn body_kind(&self, id: BodyId) -> Option<BodyKind> {
        self.body(id).map(|b| b.kind)
    }

    pub fn body_user_data(&self, id: BodyId) -> Option<u64> {
        self.body(id).map(|b| b.user_data)
    }

    pub fn linear_velocity(&self, id: BodyId) -> Option<Vec3> {
        self.body(id).map(|b| b.linear_velocity)
    }

    pub fn angular_velocity(&self, id: BodyId) -> Option<Vec3> {
        self.body(id).map(|b| b.angular_velocity)
    }

    /// The velocity of the body's material point at world position `p`.
    pub fn point_velocity(&self, id: BodyId, p: Vec3) -> Option<Vec3> {
        self.body(id)
            .map(|b| b.linear_velocity + b.angular_velocity.cross(p - b.center))
    }

    /// Sets the velocity of the centre of mass (a dynamic body's locked axes keep zero).
    pub fn set_linear_velocity(&mut self, id: BodyId, v: Vec3) {
        let Some(i) = self.body_index(id) else { return };
        if self.bodies[i].kind == BodyKind::Static || !v.is_finite() {
            return;
        }
        let b = &mut self.bodies[i];
        let v = b.allowed_velocities(v, Vec3::ZERO).0;
        b.linear_velocity = v;
        if v != Vec3::ZERO {
            self.wake_body(i as u32);
        }
    }

    /// Sets the angular velocity (a dynamic body's locked axes keep zero).
    pub fn set_angular_velocity(&mut self, id: BodyId, w: Vec3) {
        let Some(i) = self.body_index(id) else { return };
        if self.bodies[i].kind == BodyKind::Static || !w.is_finite() {
            return;
        }
        let b = &mut self.bodies[i];
        let w = b.allowed_velocities(Vec3::ZERO, w).1;
        b.angular_velocity = w;
        if w != Vec3::ZERO {
            self.wake_body(i as u32);
        }
    }

    /// Sets a kinematic body's velocities so that it reaches `target` at the end of a step of
    /// `dt` (what moving platforms and doors use). A target or step that is not finite, or that
    /// would need a velocity that is not, is ignored.
    pub fn set_kinematic_target(&mut self, id: BodyId, target: Iso, dt: f32) {
        let Some(i) = self.body_index(id) else { return };
        let b = &self.bodies[i];
        if b.kind != BodyKind::Kinematic
            || dt <= 0.0
            || !dt.is_finite()
            || !target.position.is_finite()
            || !target.rotation.is_finite()
        {
            return;
        }
        let q = target.rotation.normalize();
        let target_center = target.position + q.rotate(b.local_center);
        let linear_velocity = (target_center - b.center) * (1.0 / dt);
        // The rotation taking the current orientation to the target, as an angular velocity:
        // for a unit quaternion (v, w) of angle θ, v = sin(θ/2)·axis; ω ≈ 2·v/dt for small θ,
        // corrected by θ/sin θ via the deterministic atan2. The body's own rotation is no turn,
        // also after normalizing changed its last bits, and so is anything within the rounding
        // of a unit quaternion (|v| up to a few 1e-7): read as a turn, that noise kept a still
        // body awake and slowly turning.
        let mut d = if target.rotation == b.rotation {
            Quat::IDENTITY
        } else {
            q * b.rotation.conjugate()
        };
        if d.w < 0.0 {
            d = Quat::new(-d.x, -d.y, -d.z, -d.w);
        }
        let v = d.vector();
        let s = v.length();
        let angular_velocity = if s > 1e-6 {
            let angle = 2.0 * crate::math::atan2(s, d.w);
            v * (angle / (s * dt))
        } else {
            Vec3::ZERO
        };
        if !linear_velocity.is_finite() || !angular_velocity.is_finite() {
            return;
        }
        let b = &mut self.bodies[i];
        b.linear_velocity = linear_velocity;
        b.angular_velocity = angular_velocity;
        // A target where the body already is leaves it as it is (asleep, or falling asleep).
        if b.linear_velocity != Vec3::ZERO || b.angular_velocity != Vec3::ZERO {
            self.wake_body(i as u32);
        }
    }

    /// Adds a force at the centre of mass until the end of the next step. Ignored (as are the
    /// calls below) on a body that is not dynamic, for an input that is not finite, and when
    /// the total it would make (or the velocity it would give) is not finite either.
    pub fn apply_force(&mut self, id: BodyId, f: Vec3) {
        self.add_force_and_torque(id, f, Vec3::ZERO);
    }

    /// Adds a force at world point `p` until the end of the next step.
    pub fn apply_force_at(&mut self, id: BodyId, f: Vec3, p: Vec3) {
        let Some(i) = self.body_index(id) else { return };
        if p.is_finite() {
            let r = p - self.bodies[i].center;
            self.add_force_and_torque(id, f, r.cross(f));
        }
    }

    pub fn apply_torque(&mut self, id: BodyId, t: Vec3) {
        self.add_force_and_torque(id, Vec3::ZERO, t);
    }

    fn add_force_and_torque(&mut self, id: BodyId, f: Vec3, t: Vec3) {
        let Some(i) = self.body_index(id) else { return };
        let b = &mut self.bodies[i];
        let (force, torque) = (b.force + f, b.torque + t);
        if b.kind == BodyKind::Dynamic && f.is_finite() && force.is_finite() && torque.is_finite() {
            b.force = force;
            b.torque = torque;
            self.wake_body(i as u32);
        }
    }

    /// Changes velocity at once: impulse `j` (N·s) at world point `p`.
    pub fn apply_impulse_at(&mut self, id: BodyId, j: Vec3, p: Vec3) {
        let Some(i) = self.body_index(id) else { return };
        let b = &self.bodies[i];
        if !j.is_finite() || !p.is_finite() {
            return;
        }
        let w = b.world_inv_inertia().mul_vec((p - b.center).cross(j));
        self.add_velocity(i, b.inv_mass_vector().mul_elem(j), w);
    }

    pub fn apply_angular_impulse(&mut self, id: BodyId, j: Vec3) {
        let Some(i) = self.body_index(id) else { return };
        if j.is_finite() {
            let w = self.bodies[i].world_inv_inertia().mul_vec(j);
            self.add_velocity(i, Vec3::ZERO, w);
        }
    }

    /// Adds to a dynamic body's velocities, unless either result is not finite (an impulse too
    /// large for a light body, or one far from it whose turn overflows).
    fn add_velocity(&mut self, i: usize, dv: Vec3, dw: Vec3) {
        let b = &mut self.bodies[i];
        let (v, w) = (b.linear_velocity + dv, b.angular_velocity + dw);
        if b.kind == BodyKind::Dynamic && v.is_finite() && w.is_finite() {
            b.linear_velocity = v;
            b.angular_velocity = w;
            self.wake_body(i as u32);
        }
    }

    pub fn mass(&self, id: BodyId) -> Option<f32> {
        self.body(id).map(|b| b.mass)
    }

    /// World centre of mass.
    pub fn center_of_mass(&self, id: BodyId) -> Option<Vec3> {
        self.body(id).map(|b| b.center)
    }

    pub fn is_awake(&self, id: BodyId) -> bool {
        self.body(id).is_some_and(|b| b.awake)
    }

    pub fn wake(&mut self, id: BodyId) {
        if let Some(i) = self.body_index(id) {
            self.wake_body(i as u32);
        }
    }

    // ---------------------------------------------------------------- colliders

    /// Attaches a collider to a body (meshes and heightfields only to static bodies) and updates
    /// the body's mass.
    pub fn add_collider(
        &mut self,
        body: BodyId,
        desc: &ColliderDesc,
    ) -> Result<ColliderId, String> {
        let bi = self.body_index(body).ok_or("no such body")?;
        desc.shape.validate()?;
        if !desc.shape.is_convex() && self.bodies[bi].kind != BodyKind::Static {
            return Err("meshes and heightfields can only be on static bodies".into());
        }
        if !desc.local.position.is_finite() || !desc.local.rotation.is_finite() {
            return Err("collider placement is not finite".into());
        }
        if [desc.density, desc.friction, desc.restitution]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err("collider density, friction or restitution is not usable".into());
        }
        if !desc.sensor && self.bodies[bi].kind == BodyKind::Dynamic {
            // The body's mass with this collider must still be a number (a dense or huge shape
            // overflows f32), and so must its centre and inertia once the mass is brought into
            // range.
            let local = Iso::new(desc.local.position, desc.local.rotation.normalize());
            let with = |scale: f32| {
                let mut parts = self.mass_parts(bi as u32, scale);
                parts.push(
                    desc.shape
                        .mass_properties(desc.density * scale)
                        .transformed(&local),
                );
                parts
            };
            if bounded_mass(with).is_none() {
                return Err("the body's mass with this collider is not finite".into());
            }
        }
        let in_static_tree = self.bodies[bi].kind == BodyKind::Static;
        let collider = Collider {
            generation: 0,
            alive: true,
            body: bi as u32,
            shape: desc.shape.clone(),
            local: Iso::new(desc.local.position, desc.local.rotation.normalize()),
            density: desc.density,
            friction: desc.friction,
            restitution: desc.restitution,
            sensor: desc.sensor,
            layers: desc.layers,
            mask: desc.mask,
            user_data: desc.user_data,
            proxy: 0,
            in_static_tree,
            fat: Aabb::EMPTY,
            contacts: Vec::new(),
            queued: false,
        };
        let index = if let Some(i) = self.free_colliders.pop() {
            let old = &self.colliders[i as usize];
            let (generation, queued) = (old.generation + 1, old.queued);
            self.colliders[i as usize] = Collider {
                generation,
                queued,
                ..collider
            };
            i
        } else {
            self.colliders.push(collider);
            (self.colliders.len() - 1) as u32
        };
        let fat = self.collider_box(index).expand(self.aabb_margin());
        let c = &mut self.colliders[index as usize];
        c.fat = fat;
        c.proxy = if in_static_tree {
            self.static_tree.insert(fat, index)
        } else {
            self.moving_tree.insert(fat, index)
        };
        self.queue(index);
        self.bodies[bi].colliders.push(index);
        self.update_mass(bi as u32);
        // A dynamic body created asleep stays asleep with its new shape (a level's pre-settled
        // stack); pair finding needs only the move buffer. A static or kinematic body grown into
        // a sleeping body wakes it: neither one ever pushes it through a contact while both
        // rest (a pair with no awake moving body is not even updated), and a kinematic body
        // that has stopped sleeps like the ground.
        if self.bodies[bi].awake {
            self.bodies[bi].sleep_time = 0.0;
        }
        if self.bodies[bi].kind != BodyKind::Dynamic {
            self.wake_overlapping(&[index]);
        }
        Ok(ColliderId {
            index,
            generation: self.colliders[index as usize].generation,
        })
    }

    fn collider_index(&self, id: ColliderId) -> Option<usize> {
        let c = self.colliders.get(id.index as usize)?;
        (c.alive && c.generation == id.generation).then_some(id.index as usize)
    }

    /// Removes a collider. What touched it wakes, and so does its body, whose mass and shape
    /// change (with its island: what touches it or hangs from it).
    pub fn remove_collider(&mut self, id: ColliderId) {
        if let Some(i) = self.collider_index(id) {
            let body = self.colliders[i].body;
            self.wake_touching_collider(i as u32);
            self.destroy_collider_index(i as u32);
            self.update_mass(body);
            self.wake_body(body);
        }
    }

    /// Removes a collider with its contacts (its own list: the cost does not grow with the
    /// world). If it is in the move buffer, its index stays there: the broad phase skips a dead
    /// collider, and a collider reusing the slot is already queued.
    fn destroy_collider_index(&mut self, c: u32) {
        let mut contacts = self.colliders[c as usize].contacts.clone();
        // Highest index first, so swap-removal never moves a contact still to be removed.
        contacts.sort_unstable_by(|a, b| b.cmp(a));
        for k in contacts {
            self.destroy_contact(k as usize, true);
        }
        let col = &mut self.colliders[c as usize];
        if col.in_static_tree {
            self.static_tree.remove(col.proxy);
        } else {
            self.moving_tree.remove(col.proxy);
        }
        col.alive = false;
        let body = col.body as usize;
        self.bodies[body].colliders.retain(|&x| x != c);
        self.free_colliders.push(c);
    }

    /// Queues a collider for pair finding at the next step (once).
    fn queue(&mut self, c: u32) {
        let col = &mut self.colliders[c as usize];
        if !col.queued {
            col.queued = true;
            self.move_buffer.push(c);
        }
    }

    pub fn collider_body(&self, id: ColliderId) -> Option<BodyId> {
        let c = &self.colliders[self.collider_index(id)?];
        Some(BodyId {
            index: c.body,
            generation: self.bodies[c.body as usize].generation,
        })
    }

    pub fn collider_shape(&self, id: ColliderId) -> Option<&Shape> {
        self.collider_index(id).map(|i| &self.colliders[i].shape)
    }

    pub fn collider_user_data(&self, id: ColliderId) -> Option<u64> {
        self.collider_index(id).map(|i| self.colliders[i].user_data)
    }

    /// The collider's world pose.
    pub fn collider_pose(&self, id: ColliderId) -> Option<Iso> {
        self.collider_index(id).map(|i| self.collider_iso(i as u32))
    }

    pub(crate) fn collider_iso(&self, c: u32) -> Iso {
        let col = &self.colliders[c as usize];
        self.bodies[col.body as usize].iso().mul(&col.local)
    }

    pub(crate) fn collider_id(&self, c: u32) -> ColliderId {
        ColliderId {
            index: c,
            generation: self.colliders[c as usize].generation,
        }
    }

    pub(crate) fn body_id(&self, b: u32) -> BodyId {
        BodyId {
            index: b,
            generation: self.bodies[b as usize].generation,
        }
    }

    /// The broad-phase margin, as the next step will use it (calls between steps may meet a
    /// value out of range that the step has not replaced yet).
    fn aabb_margin(&self) -> f32 {
        let m = self.settings.aabb_margin;
        if m.is_finite() && m >= 0.0 {
            m
        } else {
            Settings::default().aabb_margin
        }
    }

    /// The speculative distance, as the next step will use it.
    fn speculative_distance(&self) -> f32 {
        let d = self.settings.speculative_distance;
        if d.is_finite() && d >= 0.0 {
            d
        } else {
            Settings::default().speculative_distance
        }
    }

    /// The linear slop, as the next step will use it.
    pub(crate) fn linear_slop(&self) -> f32 {
        let d = self.settings.linear_slop;
        if d.is_finite() && d >= 0.0 {
            d
        } else {
            Settings::default().linear_slop
        }
    }

    /// The collider's bounds grown by the speculative distance.
    fn collider_box(&self, c: u32) -> Aabb {
        let iso = self.collider_iso(c);
        self.colliders[c as usize]
            .shape
            .aabb(&iso)
            .expand(self.speculative_distance())
    }

    /// The mass of each of a body's colliders, in the body's frame (none for sensors), with
    /// every density multiplied by `scale`.
    fn mass_parts(&self, b: u32, scale: f32) -> Vec<MassProperties> {
        self.bodies[b as usize]
            .colliders
            .iter()
            .map(|&c| {
                let col = &self.colliders[c as usize];
                if col.sensor {
                    MassProperties::default()
                } else {
                    col.shape
                        .mass_properties(col.density * scale)
                        .transformed(&col.local)
                }
            })
            .collect()
    }

    /// Recomputes a body's mass, centre and inertia from its colliders.
    fn update_mass(&mut self, b: u32) {
        // add_collider refuses colliders that would leave no usable mass.
        let total = bounded_mass(|scale| self.mass_parts(b, scale)).unwrap_or_default();
        let body = &mut self.bodies[b as usize];
        if body.kind != BodyKind::Dynamic {
            body.mass = 0.0;
            body.inv_mass = 0.0;
            body.local_inertia = Mat3::ZERO;
            body.inv_local_inertia = Mat3::ZERO;
            body.local_center = Vec3::ZERO;
            body.center = body.position;
        } else {
            let (mass, center, inertia) = if total.mass > 0.0 {
                (total.mass, total.center, total.inertia)
            } else {
                // No massive colliders: a unit point mass, so it still falls.
                (1.0, Vec3::ZERO, Mat3::from_diagonal(Vec3::splat(0.4)))
            };
            body.mass = mass;
            body.inv_mass = 1.0 / mass;
            body.local_inertia = inertia;
            body.inv_local_inertia = inertia.inverse_or_zero();
            body.local_center = center;
            body.center = body.position + body.rotation.rotate(center);
            body.center0 = body.center;
        }
        // The sleep test's reach (how fast the farthest point moves) is measured from the centre
        // of mass, which the body turns about; continuous collision's smallest size is that of
        // the thinnest collider. Sensors neither move nor stop anything.
        let center = self.bodies[b as usize].local_center;
        let (mut max_extent, mut min_extent) = (0.0f32, f32::INFINITY);
        for &c in &self.bodies[b as usize].colliders {
            let col = &self.colliders[c as usize];
            if col.sensor {
                continue;
            }
            max_extent = max_extent.max(col.shape.farthest_from(&col.local, center));
            min_extent = min_extent.min(col.shape.inner_radius());
        }
        let body = &mut self.bodies[b as usize];
        body.max_extent = max_extent;
        body.min_extent = if min_extent.is_finite() {
            min_extent
        } else {
            0.0
        };
    }

    // ---------------------------------------------------------------- joints

    pub fn create_joint(&mut self, desc: &JointDesc) -> Result<JointId, String> {
        let a = self
            .body_index(desc.body_a)
            .ok_or("joint body A does not exist")?;
        let b = self
            .body_index(desc.body_b)
            .ok_or("joint body B does not exist")?;
        if a == b {
            return Err("a joint needs two different bodies".into());
        }
        let slot = JointSlot {
            generation: 0,
            alive: true,
            desc: *desc,
            impulses: JointImpulses::default(),
        };
        let index = if let Some(i) = self.free_joints.pop() {
            let generation = self.joints[i as usize].generation + 1;
            self.joints[i as usize] = JointSlot { generation, ..slot };
            i
        } else {
            self.joints.push(slot);
            (self.joints.len() - 1) as u32
        };
        self.bodies[a].joints.push(index);
        self.bodies[b].joints.push(index);
        if !desc.collide_connected {
            // Drop any contact between the two bodies now, as the step would (found in the
            // contact lists of whichever body has fewer). Sensor pairs stay: a joint never
            // turns a sensor off (see `should_collide`), and a pair dropped here would not be
            // found again until a collider left its broad-phase box.
            let count = |w: &PhysicsWorld, x: usize| -> usize {
                w.bodies[x]
                    .colliders
                    .iter()
                    .map(|&c| w.colliders[c as usize].contacts.len())
                    .sum()
            };
            let (from, to) = if count(self, a) <= count(self, b) {
                (a, b)
            } else {
                (b, a)
            };
            let mut doomed: Vec<u32> = Vec::new();
            for &c in &self.bodies[from].colliders {
                for &k in &self.colliders[c as usize].contacts {
                    let ct = &self.contacts[k as usize];
                    if !ct.sensor && self.colliders[ct.other(c) as usize].body as usize == to {
                        doomed.push(k);
                    }
                }
            }
            // Highest index first, so swap-removal never moves a contact still to be removed.
            doomed.sort_unstable_by(|x, y| y.cmp(x));
            for k in doomed {
                self.destroy_contact(k as usize, true);
            }
        }
        self.wake_body(a as u32);
        self.wake_body(b as u32);
        Ok(JointId {
            index,
            generation: self.joints[index as usize].generation,
        })
    }

    fn joint_index(&self, id: JointId) -> Option<usize> {
        let j = self.joints.get(id.index as usize)?;
        (j.alive && j.generation == id.generation).then_some(id.index as usize)
    }

    pub fn destroy_joint(&mut self, id: JointId) {
        if let Some(i) = self.joint_index(id) {
            self.destroy_joint_index(i as u32);
        }
    }

    fn destroy_joint_index(&mut self, j: u32) {
        let desc = self.joints[j as usize].desc;
        let ends = [desc.body_a, desc.body_b].map(|id| self.body_index(id));
        for b in ends.into_iter().flatten() {
            self.bodies[b].joints.retain(|&x| x != j);
            self.wake_body(b as u32);
        }
        // The two bodies may collide again: their overlapping colliders must be paired at the
        // next step. Queueing the colliders of one moving body is enough, as each of them looks
        // for pairs in both trees (the one with fewer colliders; never a static level's, whose
        // colliders would each query the moving tree for nothing). Nothing is queued when the
        // joint never kept them apart, or when neither body is dynamic (they never collide;
        // sensors ignore joints).
        if let [Some(a), Some(b)] = ends
            && !desc.collide_connected
            && (self.bodies[a].is_dynamic() || self.bodies[b].is_dynamic())
        {
            let moving = [a, b].into_iter().filter(|&x| self.bodies[x].moves());
            let fewest = moving.min_by_key(|&x| (self.bodies[x].colliders.len(), x));
            if let Some(x) = fewest {
                for c in self.bodies[x].colliders.clone() {
                    self.queue(c);
                }
            }
        }
        self.joints[j as usize].alive = false;
        self.free_joints.push(j);
    }

    pub fn joint_desc(&self, id: JointId) -> Option<&JointDesc> {
        self.joint_index(id).map(|i| &self.joints[i].desc)
    }

    /// Replaces a joint's options (limits, motor, spring); its bodies and frames stay.
    pub fn set_joint_kind(&mut self, id: JointId, kind: joint::JointKind) {
        if let Some(i) = self.joint_index(id) {
            self.joints[i].desc.kind = kind;
            let d = self.joints[i].desc;
            for b in [d.body_a, d.body_b] {
                if let Some(b) = self.body_index(b) {
                    self.wake_body(b as u32);
                }
            }
        }
    }

    // ---------------------------------------------------------------- events

    /// Events of the last step: what began, ended and hit during it, and the ends of contacts
    /// that calls between it and the step before removed (bodies or colliders destroyed, joints
    /// that turn collisions off). Calls between steps leave this list as it is.
    pub fn events(&self) -> &Events {
        &self.events
    }

    /// Steps taken so far.
    pub fn step_count(&self) -> u64 {
        self.steps
    }

    pub fn contact_count(&self) -> usize {
        self.contacts
            .iter()
            .filter(|c| c.touching && !c.sensor)
            .count()
    }

    /// Every touching manifold, for drawing and inspection: (collider A, collider B, manifold).
    pub fn manifolds(&self) -> impl Iterator<Item = (ColliderId, ColliderId, &Manifold)> + '_ {
        self.contacts
            .iter()
            .filter(|c| c.touching && !c.sensor)
            .flat_map(move |c| {
                c.manifolds
                    .iter()
                    .map(move |m| (self.collider_id(c.a), self.collider_id(c.b), &m.manifold))
            })
    }

    // ---------------------------------------------------------------- the step

    /// Advances the simulation by `dt` seconds (use a fixed step, for example 1/60).
    pub fn step(&mut self, dt: f32) {
        if !(dt > 0.0 && dt.is_finite()) {
            return;
        }
        self.settings = self.settings.sanitized();
        self.steps += 1;
        if self.move_buffer.is_empty() && !self.has_awake_moving_bodies() {
            self.rescale_warm_starts(self.settings.substeps.max(1));
            std::mem::swap(&mut self.events, &mut self.pending);
            self.pending.contacts.clear();
            self.pending.sensors.clear();
            return;
        }
        self.update_broad_phase();
        self.update_contacts();
        self.solve(dt);
        // The broad-phase boxes follow the bodies at once, so that continuous collision below and
        // every query between steps (rays, casts, overlaps, the character's sweeps) see each
        // collider where it is now, not where the step started. Colliders that left their box
        // are queued for pair finding at the next step.
        let moved = std::mem::take(&mut self.scratch.members);
        for &b in &moved {
            self.refresh_body_proxies(b, false);
        }
        if self.settings.enable_continuous && !moved.is_empty() {
            self.continuous(&moved, dt);
        }
        self.scratch.members = moved;
        std::mem::swap(&mut self.events, &mut self.pending);
        self.pending.contacts.clear();
        self.pending.sensors.clear();
    }

    /// Finds new pairs for the colliders that moved out of their boxes (or were added or
    /// teleported). Boxes themselves are refreshed where bodies move: at the end of each step and
    /// in [`Self::set_body_pose`].
    fn update_broad_phase(&mut self) {
        if self.move_buffer.is_empty() {
            return;
        }
        let mut moved = std::mem::take(&mut self.move_buffer);
        moved.sort_unstable();
        moved.dedup();
        for &c in &moved {
            self.colliders[c as usize].queued = false;
        }
        let mut found = std::mem::take(&mut self.scratch.broad_found);
        found.clear();
        for &c in &moved {
            let col = &self.colliders[c as usize];
            if !col.alive {
                continue;
            }
            let fat = col.fat;
            let mut visit = |o: u32| {
                if o != c {
                    let key = (c.min(o), c.max(o));
                    if !self.pairs.contains_key(&key) && self.should_collide(key.0, key.1) {
                        found.push(key);
                    }
                }
            };
            self.moving_tree.query(&fat, &mut |_, o| {
                visit(o);
                true
            });
            if !col.in_static_tree {
                self.static_tree.query(&fat, &mut |_, o| {
                    visit(o);
                    true
                });
            }
        }
        found.sort_unstable();
        found.dedup();
        for &(a, b) in &found {
            let sensor = self.colliders[a as usize].sensor || self.colliders[b as usize].sensor;
            let k = self.contacts.len() as u32;
            let slots = [a, b].map(|c| {
                let list = &mut self.colliders[c as usize].contacts;
                list.push(k);
                (list.len() - 1) as u32
            });
            self.contacts.push(Contact {
                a,
                b,
                manifolds: Vec::new(),
                touching: false,
                sensor,
                slots,
            });
            self.pairs.insert((a, b), k);
        }
        found.clear();
        self.scratch.broad_found = found;
        moved.clear();
        self.move_buffer = moved;
    }

    /// Updates a body's collider boxes; queues those that left their fat box (or all, when
    /// `force`) for pair finding.
    fn refresh_body_proxies(&mut self, b: u32, force: bool) {
        let margin = self.aabb_margin();
        let len = self.bodies[b as usize].colliders.len();
        for ci in 0..len {
            let c = self.bodies[b as usize].colliders[ci];
            let tight = self.collider_box(c);
            let col = &self.colliders[c as usize];
            if force || !col.fat.contains(&tight) {
                let fat = tight.expand(margin);
                let (proxy, static_tree) = (col.proxy, col.in_static_tree);
                if static_tree {
                    self.static_tree.update(proxy, fat);
                } else {
                    self.moving_tree.update(proxy, fat);
                }
                self.colliders[c as usize].fat = fat;
                self.queue(c);
            }
        }
    }

    fn should_collide(&self, a: u32, b: u32) -> bool {
        let (ca, cb) = (&self.colliders[a as usize], &self.colliders[b as usize]);
        if !ca.alive || !cb.alive || ca.body == cb.body {
            return false;
        }
        if ca.layers & cb.mask == 0 || cb.layers & ca.mask == 0 {
            return false;
        }
        let (ba, bb) = (
            &self.bodies[ca.body as usize],
            &self.bodies[cb.body as usize],
        );
        if ca.sensor || cb.sensor {
            // Sensors see moving bodies, never static ones or other sensors.
            if ca.sensor && cb.sensor {
                return false;
            }
            let other = if ca.sensor { bb } else { ba };
            return other.moves();
        }
        if !ba.is_dynamic() && !bb.is_dynamic() {
            return false;
        }
        if !ca.shape.is_convex() && !cb.shape.is_convex() {
            return false;
        }
        // A joint between the two bodies may turn their collisions off: it is in both bodies'
        // lists, so the shorter one is searched (a level body with many joints touches many
        // bodies with none).
        let joints = if ba.joints.len() <= bb.joints.len() {
            &ba.joints
        } else {
            &bb.joints
        };
        for &j in joints {
            let d = &self.joints[j as usize].desc;
            let (x, y) = (d.body_a.index, d.body_b.index);
            if !d.collide_connected
                && ((x == ca.body && y == cb.body) || (x == cb.body && y == ca.body))
            {
                return false;
            }
        }
        true
    }

    fn destroy_contact(&mut self, k: usize, report: bool) {
        let c = &self.contacts[k];
        let key = (c.a, c.b);
        if report && c.touching {
            let (a, b) = (self.collider_id(c.a), self.collider_id(c.b));
            if c.sensor {
                let (sensor, visitor) = if self.colliders[c.a as usize].sensor {
                    (a, b)
                } else {
                    (b, a)
                };
                self.pending
                    .sensors
                    .push(SensorEvent::End { sensor, visitor });
            } else {
                self.pending.contacts.push(ContactEvent::End { a, b });
            }
        }
        // Out of its colliders' lists: the last entry of each list takes its place there.
        let slots = self.contacts[k].slots;
        for (end, c) in [key.0, key.1].into_iter().enumerate() {
            let s = slots[end] as usize;
            let list = &mut self.colliders[c as usize].contacts;
            list.swap_remove(s);
            if let Some(&m) = list.get(s) {
                let mc = &mut self.contacts[m as usize];
                mc.slots[usize::from(mc.a != c)] = s as u32;
            }
        }
        self.pairs.remove(&key);
        self.contacts.swap_remove(k);
        // The last contact now has index `k`: in the pair map and in its colliders' lists.
        if let Some(m) = self.contacts.get(k) {
            let (moved, slots) = ((m.a, m.b), m.slots);
            self.pairs.insert(moved, k as u32);
            self.colliders[moved.0 as usize].contacts[slots[0] as usize] = k as u32;
            self.colliders[moved.1 as usize].contacts[slots[1] as usize] = k as u32;
        }
    }

    /// The narrow phase: new manifolds for every pair with an awake moving body, carried
    /// impulses, begin and end events; pairs whose boxes no longer overlap are dropped.
    fn update_contacts(&mut self) {
        let awake_count = self.awake_body_count();
        if awake_count == 0 {
            return;
        }
        let mut narrow_manifolds = std::mem::take(&mut self.scratch.narrow_manifolds);
        let mut old_manifolds = std::mem::take(&mut self.scratch.old_manifolds);
        let mut wake = std::mem::take(&mut self.scratch.wake);
        let mut active_indices = std::mem::take(&mut self.scratch.active_indices);
        wake.clear();

        if awake_count.saturating_mul(8) < self.contacts.len() {
            active_indices.clear();
            for (wi, &word) in self.awake_words.iter().enumerate() {
                let mut w = word;
                let base = (wi << 6) as u32;
                while w != 0 {
                    let b = (base + w.trailing_zeros()) as usize;
                    for &c in &self.bodies[b].colliders {
                        active_indices.extend_from_slice(&self.colliders[c as usize].contacts);
                    }
                    w &= w - 1;
                }
            }
            active_indices.sort_unstable();
            active_indices.dedup();

            let mut idx = 0;
            while idx < active_indices.len() {
                let k = active_indices[idx] as usize;
                if !self.update_single_contact(
                    k,
                    &mut narrow_manifolds,
                    &mut old_manifolds,
                    &mut wake,
                ) {
                    let last = self.contacts.len();
                    if k < last && active_indices.last() == Some(&(last as u32)) {
                        active_indices.pop();
                    } else {
                        idx += 1;
                    }
                    continue;
                }
                idx += 1;
            }
        } else {
            let mut k = 0;
            while k < self.contacts.len() {
                let (a, b) = (self.contacts[k].a, self.contacts[k].b);
                let (ca, cb) = (&self.colliders[a as usize], &self.colliders[b as usize]);
                let (ba, bb) = (
                    &self.bodies[ca.body as usize],
                    &self.bodies[cb.body as usize],
                );
                let active = (ba.awake && ba.moves()) || (bb.awake && bb.moves());
                if !active {
                    k += 1;
                    continue;
                }
                if !self.update_single_contact(
                    k,
                    &mut narrow_manifolds,
                    &mut old_manifolds,
                    &mut wake,
                ) {
                    continue;
                }
                k += 1;
            }
        }

        // A moving kinematic body wakes what hangs from it by a joint (islands leave kinematic
        // bodies out, and so does waking along joints).
        if self.kinematic_count > 0 {
            for j in &self.joints {
                if !j.alive {
                    continue;
                }
                let (x, y) = (j.desc.body_a.index, j.desc.body_b.index);
                for (p, q) in [(x, y), (y, x)] {
                    let (bp, bq) = (&self.bodies[p as usize], &self.bodies[q as usize]);
                    if bp.awake && bp.is_kinematic_moving() && !bq.awake && bq.is_dynamic() {
                        wake.push(q);
                    }
                }
            }
        }
        self.wake_islands(&wake);

        self.scratch.narrow_manifolds = narrow_manifolds;
        self.scratch.old_manifolds = old_manifolds;
        self.scratch.wake = wake;
        self.scratch.active_indices = active_indices;
    }

    /// Updates active contact `k`; returns `false` if the contact was destroyed.
    fn update_single_contact(
        &mut self,
        k: usize,
        narrow_manifolds: &mut Vec<Manifold>,
        old_manifolds: &mut Vec<ContactManifold>,
        wake: &mut Vec<u32>,
    ) -> bool {
        let (a, b) = (self.contacts[k].a, self.contacts[k].b);
        let (ca, cb) = (&self.colliders[a as usize], &self.colliders[b as usize]);
        if !ca.fat.overlaps(&cb.fat) || !self.should_collide(a, b) {
            self.destroy_contact(k, true);
            return false;
        }
        let (iso_a, iso_b) = (self.collider_iso(a), self.collider_iso(b));
        narrow_manifolds.clear();
        let sensor = self.contacts[k].sensor;
        let speculative = if sensor {
            0.0
        } else {
            self.settings.speculative_distance
        };
        collide::collide(
            &self.colliders[a as usize].shape,
            &iso_a,
            &self.colliders[b as usize].shape,
            &iso_b,
            speculative,
            narrow_manifolds,
        );
        let was_touching = self.contacts[k].touching;
        let touching = if sensor {
            narrow_manifolds.iter().any(|m| m.min_separation() < 0.0)
        } else {
            narrow_manifolds.iter().any(|m| m.count > 0)
        };
        if !sensor {
            let mut manifolds = std::mem::take(&mut self.contacts[k].manifolds);
            std::mem::swap(&mut manifolds, old_manifolds);
            let match_distance =
                2.0 * self.settings.speculative_distance + self.settings.linear_slop;
            carry_impulses_into(
                old_manifolds,
                narrow_manifolds,
                &iso_a,
                match_distance,
                &mut manifolds,
            );
            self.contacts[k].manifolds = manifolds;
        } else {
            self.contacts[k].manifolds.clear();
        }
        self.contacts[k].touching = touching;
        if touching != was_touching {
            let (ia, ib) = (self.collider_id(a), self.collider_id(b));
            if sensor {
                let (sensor_id, visitor) = if self.colliders[a as usize].sensor {
                    (ia, ib)
                } else {
                    (ib, ia)
                };
                self.pending.sensors.push(if touching {
                    SensorEvent::Begin {
                        sensor: sensor_id,
                        visitor,
                    }
                } else {
                    SensorEvent::End {
                        sensor: sensor_id,
                        visitor,
                    }
                });
            } else {
                self.pending.contacts.push(if touching {
                    ContactEvent::Begin { a: ia, b: ib }
                } else {
                    ContactEvent::End { a: ia, b: ib }
                });
            }
        }
        if touching && !sensor {
            // An awake body that can push wakes a sleeping one it touches: a dynamic body, or
            // a kinematic one that moves. A kinematic body standing still does not (it would
            // wake its load the step after the load fell asleep, every time), unless the
            // contact has just begun (it was put there, or was given a new collider there).
            let (pa, pb) = (
                self.colliders[a as usize].body,
                self.colliders[b as usize].body,
            );
            for (p, q) in [(pa, pb), (pb, pa)] {
                let (bp, bq) = (&self.bodies[p as usize], &self.bodies[q as usize]);
                if bp.awake && (bp.pushes() || !was_touching) && !bq.awake && bq.is_dynamic() {
                    wake.push(q);
                }
            }
        }
        true
    }

    /// The soft step over every awake moving body; stores those bodies in `self.scratch.members`
    /// in ascending index order.
    fn solve(&mut self, dt: f32) {
        self.rescale_warm_starts(self.settings.substeps.max(1));
        let mut scratch = std::mem::take(&mut self.scratch);
        self.collect_awake_moving(&mut scratch.members);
        if scratch.members.is_empty() {
            self.scratch = scratch;
            return;
        }

        let s = &self.settings;
        let substeps = s.substeps.max(1);
        let h = dt / substeps as f32;
        let contact_hertz = s.contact_hertz.min(0.25 / h);
        let joint_hertz = s.joint_hertz.min(0.25 / h);
        let ctx = StepContext {
            dt,
            h,
            inv_h: 1.0 / h,
            contact_softness: Softness::new(contact_hertz, s.contact_damping_ratio, h),
            static_softness: Softness::new(2.0 * contact_hertz, s.contact_damping_ratio, h),
            joint_softness: Softness::new(joint_hertz, s.joint_damping_ratio, h),
            max_push_velocity: s.contact_push_max_velocity,
            restitution_threshold: s.restitution_threshold,
        };

        // Solver bodies: slot 0 is static; then every awake moving body in index order.
        if scratch.slot.len() < self.bodies.len() {
            scratch.slot.resize(self.bodies.len(), 0);
        }
        scratch.solver_bodies.clear();
        scratch.solver_bodies.push(SolverBody::STATIC);
        scratch.solver_vels.clear();
        scratch.solver_vels.push(SolverVel::ZERO);
        scratch.per_body.clear();
        scratch.turning.clear();
        scratch.turning.push(Turning::Fixed);

        for &bi in &scratch.members {
            let i = bi as usize;
            let b = &mut self.bodies[i];
            b.center0 = b.center;
            b.rotation0 = b.rotation;
            scratch.slot[i] = scratch.solver_bodies.len();
            // Every setter keeps locked axes at zero already; this makes sure of it, since
            // nothing in the solve could take such a velocity away again.
            let (v, w) = b.allowed_velocities(b.linear_velocity, b.angular_velocity);
            let inv_inertia = b.world_inv_inertia();
            scratch.solver_vels.push(SolverVel { v, w });
            scratch.solver_bodies.push(SolverBody {
                dp: Vec3::ZERO,
                dq: Quat::IDENTITY,
                inv_mass: b.inv_mass_vector(),
                inv_inertia,
                contact_inv_inertia: inv_inertia,
            });
            scratch.turning.push(b.turning());
            scratch.per_body.push(BodyInput {
                force: b.force,
                torque: b.torque,
                gravity_scale: b.gravity_scale,
                linear_damping: 1.0 / (1.0 + h * b.linear_damping),
                angular_damping: 1.0 / (1.0 + h * b.angular_damping),
                kinematic_turn: (b.kind == BodyKind::Kinematic)
                    .then(|| Quat::from_angular_velocity(w, h)),
            });
        }

        // Contact constraints.
        scratch.constraints.clear();
        let build_contact_constraint =
            |w: &PhysicsWorld,
             k: usize,
             slot: &[usize],
             solver_bodies: &[SolverBody],
             solver_vels: &[SolverVel],
             constraints: &mut Vec<ContactConstraint>| {
                let c = &w.contacts[k];
                if !c.touching || c.sensor {
                    return;
                }
                let (ca, cb) = (&w.colliders[c.a as usize], &w.colliders[c.b as usize]);
                let ia = slot[ca.body as usize];
                let ib = slot[cb.body as usize];
                if ia == 0 && ib == 0 {
                    return;
                }
                let ba = &w.bodies[ca.body as usize];
                let bb = &w.bodies[cb.body as usize];
                let fa = BodyFrame {
                    center: ba.center,
                    rotation: ba.rotation,
                    local_center: ba.local_center,
                    index: ia,
                };
                let fb = BodyFrame {
                    center: bb.center,
                    rotation: bb.rotation,
                    local_center: bb.local_center,
                    index: ib,
                };
                let friction = (ca.friction * cb.friction).sqrt();
                let restitution = ca.restitution.max(cb.restitution);
                let either_static = solver_bodies[ia].inv_mass == Vec3::ZERO
                    || solver_bodies[ib].inv_mass == Vec3::ZERO;
                for (mi, m) in c.manifolds.iter().enumerate() {
                    if m.manifold.count == 0 {
                        continue;
                    }
                    let n = m.manifold.normal;
                    let (t1, t2) = n.orthonormal_basis();
                    let mut cc = ContactConstraint {
                        a: ia,
                        b: ib,
                        normal: n,
                        tangents: [t1, t2],
                        friction,
                        restitution,
                        softness: if either_static {
                            ctx.static_softness
                        } else {
                            ctx.contact_softness
                        },
                        points: [ConstraintPoint::default(); 4],
                        count: usize::from(m.manifold.count),
                        source: (k as u32, mi as u32),
                        rounding: (ca.shape.radius(), cb.shape.radius()),
                    };
                    for i in 0..usize::from(m.manifold.count) {
                        let p = m.manifold.points[i];
                        let cache = m.cache[i];
                        cc.points[i] = solver::prepare_point(
                            &cc,
                            solver_bodies,
                            solver_vels,
                            &fa,
                            &fb,
                            p.point,
                            p.separation,
                            cache.normal_impulse,
                            cache.friction_impulse,
                        );
                    }
                    constraints.push(cc);
                }
            };

        if scratch.members.len().saturating_mul(8) < self.contacts.len() {
            scratch.active_indices.clear();
            for &b in &scratch.members {
                for &c in &self.bodies[b as usize].colliders {
                    scratch
                        .active_indices
                        .extend_from_slice(&self.colliders[c as usize].contacts);
                }
            }
            scratch.active_indices.sort_unstable();
            scratch.active_indices.dedup();
            for &k in &scratch.active_indices {
                build_contact_constraint(
                    self,
                    k as usize,
                    &scratch.slot,
                    &scratch.solver_bodies,
                    &scratch.solver_vels,
                    &mut scratch.constraints,
                );
            }
        } else {
            for k in 0..self.contacts.len() {
                build_contact_constraint(
                    self,
                    k,
                    &scratch.slot,
                    &scratch.solver_bodies,
                    &scratch.solver_vels,
                    &mut scratch.constraints,
                );
            }
        }

        // Joint constraints.
        scratch.joint_constraints.clear();
        let build_joint_constraint =
            |w: &PhysicsWorld,
             j: usize,
             slot: &[usize],
             solver_bodies: &[SolverBody],
             joint_constraints: &mut Vec<(u32, JointConstraint)>| {
                let slot_j = &w.joints[j];
                if !slot_j.alive {
                    return;
                }
                let (Some(a), Some(b)) = (
                    w.body_index(slot_j.desc.body_a),
                    w.body_index(slot_j.desc.body_b),
                ) else {
                    return;
                };
                let ia = slot[a];
                let ib = slot[b];
                if ia == 0 && ib == 0 {
                    return;
                }
                let ba = &w.bodies[a];
                let bb = &w.bodies[b];
                let fa = BodyFrame {
                    center: ba.center,
                    rotation: ba.rotation,
                    local_center: ba.local_center,
                    index: ia,
                };
                let fb = BodyFrame {
                    center: bb.center,
                    rotation: bb.rotation,
                    local_center: bb.local_center,
                    index: ib,
                };
                let jc =
                    joint::prepare(&slot_j.desc, &fa, &fb, solver_bodies, &ctx, slot_j.impulses);
                joint_constraints.push((j as u32, jc));
            };

        if scratch.members.len().saturating_mul(8) < self.joints.len() {
            scratch.active_indices.clear();
            for &b in &scratch.members {
                scratch
                    .active_indices
                    .extend_from_slice(&self.bodies[b as usize].joints);
            }
            scratch.active_indices.sort_unstable();
            scratch.active_indices.dedup();
            for &j in &scratch.active_indices {
                build_joint_constraint(
                    self,
                    j as usize,
                    &scratch.slot,
                    &scratch.solver_bodies,
                    &mut scratch.joint_constraints,
                );
            }
        } else {
            for j in 0..self.joints.len() {
                build_joint_constraint(
                    self,
                    j,
                    &scratch.slot,
                    &scratch.solver_bodies,
                    &mut scratch.joint_constraints,
                );
            }
        }

        let inputs = IntegrationInputs {
            gravity: self.settings.gravity,
            per_body: &scratch.per_body,
            max_linear_speed: self.settings.max_linear_speed,
            max_angular_speed: 0.25 * std::f32::consts::PI * ctx.inv_h,
        };
        for sub in 0..substeps {
            solver::integrate_velocities(
                &scratch.solver_bodies,
                &mut scratch.solver_vels,
                &inputs,
                h,
            );
            for (_, jc) in &scratch.joint_constraints {
                joint::warm_start(jc, &scratch.solver_bodies, &mut scratch.solver_vels);
            }
            solver::warm_start_contacts(
                &scratch.constraints,
                &scratch.solver_bodies,
                &mut scratch.solver_vels,
            );
            for (j, jc) in scratch.joint_constraints.iter_mut() {
                let desc = self.joints[*j as usize].desc;
                joint::solve(
                    jc,
                    &desc,
                    &scratch.solver_bodies,
                    &mut scratch.solver_vels,
                    &ctx,
                    true,
                );
            }
            solver::solve_contacts(
                &mut scratch.constraints,
                &scratch.solver_bodies,
                &mut scratch.solver_vels,
                &ctx,
                true,
                sub % 2 == 1,
            );
            solver::integrate_positions(
                &mut scratch.solver_bodies,
                &scratch.solver_vels,
                &inputs,
                h,
            );
            for (sb, t) in scratch.solver_bodies.iter_mut().zip(&scratch.turning) {
                if let Some(m) = t.after(sb.dq) {
                    sb.inv_inertia = m;
                }
            }
            for (j, jc) in scratch.joint_constraints.iter_mut() {
                let desc = self.joints[*j as usize].desc;
                joint::solve(
                    jc,
                    &desc,
                    &scratch.solver_bodies,
                    &mut scratch.solver_vels,
                    &ctx,
                    false,
                );
            }
            solver::solve_contacts(
                &mut scratch.constraints,
                &scratch.solver_bodies,
                &mut scratch.solver_vels,
                &ctx,
                false,
                sub % 2 == 0,
            );
        }
        solver::apply_restitution(
            &mut scratch.constraints,
            &scratch.solver_bodies,
            &mut scratch.solver_vels,
            &ctx,
        );

        // Store impulses for warm starting, and report hits.
        let hit_threshold = self.settings.hit_event_threshold;
        scratch.hits.clear();
        for cc in &scratch.constraints {
            let (k, mi) = (cc.source.0 as usize, cc.source.1 as usize);
            let cm = &mut self.contacts[k].manifolds[mi];
            for i in 0..cc.count {
                let p = &cc.points[i];
                cm.cache[i].normal_impulse = p.normal_impulse;
                cm.cache[i].friction_impulse =
                    cc.tangents[0] * p.tangent_impulse[0] + cc.tangents[1] * p.tangent_impulse[1];
                let approach = -p.relative_velocity;
                if approach > hit_threshold && p.max_normal_impulse > 0.0 {
                    let point = cm.manifold.points[i].point;
                    let ku = k as u32;
                    if let Some(last) = scratch.hits.last_mut().filter(|last| last.0 == ku) {
                        if approach > last.1 {
                            *last = (ku, approach, point, cc.normal);
                        }
                    } else {
                        scratch.hits.push((ku, approach, point, cc.normal));
                    }
                }
            }
        }
        for &(k, speed, point, normal) in &scratch.hits {
            let c = &self.contacts[k as usize];
            self.pending.contacts.push(ContactEvent::Hit {
                a: self.collider_id(c.a),
                b: self.collider_id(c.b),
                point,
                normal,
                speed,
            });
        }
        for (j, jc) in &scratch.joint_constraints {
            self.joints[*j as usize].impulses = jc.impulses;
        }

        // Write the bodies back and clear their temporary slot assignments.
        for &b in &scratch.members {
            let sidx = scratch.slot[b as usize];
            scratch.slot[b as usize] = 0;
            let sb = &scratch.solver_bodies[sidx];
            let sv = scratch.solver_vels[sidx];
            let body = &mut self.bodies[b as usize];
            body.linear_velocity = sv.v;
            body.angular_velocity = sv.w;
            body.center = body.center0 + sb.dp;
            body.rotation = body.reachable((sb.dq * body.rotation0).normalize());
            body.position = body.center - body.rotation.rotate(body.local_center);
            body.force = Vec3::ZERO;
            body.torque = Vec3::ZERO;
        }

        self.scratch = scratch;
        if self.settings.enable_sleep {
            self.update_sleep(dt);
        }
    }

    /// The impulses carried over from the last step warm-start each substep, so they are per
    /// substep: when the substep count changes, they are rescaled to the new one.
    fn rescale_warm_starts(&mut self, substeps: u32) {
        let before = std::mem::replace(&mut self.warm_substeps, substeps);
        if before == 0 || before == substeps {
            return;
        }
        let scale = before as f32 / substeps as f32;
        for c in &mut self.contacts {
            for m in &mut c.manifolds {
                for p in &mut m.cache {
                    p.normal_impulse *= scale;
                    p.friction_impulse *= scale;
                }
            }
        }
        for j in &mut self.joints {
            let i = &mut j.impulses;
            i.linear *= scale;
            i.angular *= scale;
            i.axial *= scale;
            i.lower *= scale;
            i.upper *= scale;
            i.motor *= scale;
        }
    }

    /// Sleep timers, then islands (bodies linked by touching contacts and joints) that have all
    /// been still for `time_to_sleep` go to sleep together. Runs in `O(awake)` using adjacency
    /// traversal instead of scanning all sleeping bodies and contacts in the world.
    fn update_sleep(&mut self, dt: f32) {
        let mut scratch = std::mem::take(&mut self.scratch);
        let threshold = self.settings.sleep_threshold;
        for &b in &scratch.members {
            let body = &mut self.bodies[b as usize];
            let still = body.linear_velocity.length() < threshold
                && body.angular_velocity.length() * body.max_extent < threshold;
            if body.can_sleep && still {
                body.sleep_time += dt;
            } else {
                body.sleep_time = 0.0;
            }
        }
        // A moving kinematic body keeps what it touches, and what hangs from it by a joint,
        // awake, however slowly it moves (islands leave kinematic bodies out).
        if self.kinematic_count > 0 {
            scratch.sleep_kept.clear();
            for (p, bp) in self.bodies.iter().enumerate() {
                if bp.alive && bp.is_kinematic_moving() {
                    self.push_neighbours(p as u32, &mut scratch.sleep_kept);
                }
            }
            for &q in &scratch.sleep_kept {
                self.bodies[q as usize].sleep_time = 0.0;
            }
        }

        scratch.sleep_stamp = scratch.sleep_stamp.wrapping_add(1);
        if scratch.sleep_stamp == 0 {
            scratch.sleep_visited.fill(0);
            scratch.sleep_stamp = 1;
        }
        if scratch.sleep_visited.len() < self.bodies.len() {
            scratch.sleep_visited.resize(self.bodies.len(), 0);
        }
        let stamp = scratch.sleep_stamp;
        let time_to_sleep = self.settings.time_to_sleep;

        for mi in 0..scratch.members.len() {
            let b = scratch.members[mi];
            let body = &self.bodies[b as usize];
            if !body.is_dynamic() {
                // A kinematic body sleeps on its own once it has stopped.
                if body.sleep_time > 0.0
                    && body.sleep_time >= time_to_sleep
                    && body.linear_velocity == Vec3::ZERO
                    && body.angular_velocity == Vec3::ZERO
                {
                    self.bodies[b as usize].awake = false;
                    self.set_awake_bit(b, false);
                }
                continue;
            }
            if scratch.sleep_visited[b as usize] == stamp {
                continue;
            }

            scratch.sleep_visited[b as usize] = stamp;
            scratch.sleep_stack.clear();
            scratch.sleep_island.clear();
            scratch.sleep_stack.push(b);
            let mut min_sleep = f32::INFINITY;

            while let Some(u) = scratch.sleep_stack.pop() {
                scratch.sleep_island.push(u);
                let bu = &self.bodies[u as usize];
                min_sleep = min_sleep.min(if bu.can_sleep { bu.sleep_time } else { 0.0 });

                for &c in &bu.colliders {
                    for &k in &self.colliders[c as usize].contacts {
                        let ct = &self.contacts[k as usize];
                        if !ct.touching || ct.sensor {
                            continue;
                        }
                        let v = self.colliders[ct.other(c) as usize].body;
                        let bv = &self.bodies[v as usize];
                        if !bv.alive || !bv.is_dynamic() {
                            continue;
                        }
                        if !bv.awake {
                            // Touching an already-asleep dynamic body prevents sleeping here;
                            // waking has priority.
                            min_sleep = 0.0;
                        } else if scratch.sleep_visited[v as usize] != stamp {
                            scratch.sleep_visited[v as usize] = stamp;
                            scratch.sleep_stack.push(v);
                        }
                    }
                }
                for &j in &bu.joints {
                    let d = &self.joints[j as usize].desc;
                    for v in [d.body_a.index, d.body_b.index] {
                        if v == u {
                            continue;
                        }
                        let bv = &self.bodies[v as usize];
                        if !bv.alive || !bv.is_dynamic() {
                            continue;
                        }
                        if !bv.awake {
                            min_sleep = 0.0;
                        } else if scratch.sleep_visited[v as usize] != stamp {
                            scratch.sleep_visited[v as usize] = stamp;
                            scratch.sleep_stack.push(v);
                        }
                    }
                }
            }

            // Still for at least one step, whatever time_to_sleep is (zero included).
            if min_sleep > 0.0 && min_sleep >= time_to_sleep {
                for &u in &scratch.sleep_island {
                    let body = &mut self.bodies[u as usize];
                    body.awake = false;
                    body.linear_velocity = Vec3::ZERO;
                    body.angular_velocity = Vec3::ZERO;
                    self.set_awake_bit(u, false);
                }
            }
        }
        self.scratch = scratch;
    }

    /// Wakes a body: a dynamic one with its island (see [`Self::wake_islands`]), a kinematic one
    /// alone; an awake one restarts its sleep timer.
    fn wake_body(&mut self, b: u32) {
        self.wake_islands(&[b]);
    }

    /// Wakes `roots` and, for each dynamic one that was asleep, every sleeping dynamic body linked
    /// to it by touching contacts and joints, transitively: its island. Kinematic bodies wake
    /// alone and pass nothing on, as islands leave them out: they are never pushed, so bodies
    /// resting on the same platform are not linked through it. Awake roots restart their sleep
    /// timers. Each woken body's own contact lists and joints give its links, so the cost is
    /// that of the island, not of the world.
    fn wake_islands(&mut self, roots: &[u32]) {
        let mut stack: Vec<u32> = Vec::new();
        for &b in roots {
            let body = &mut self.bodies[b as usize];
            if !body.alive || body.kind == BodyKind::Static {
                continue;
            }
            body.sleep_time = 0.0;
            if body.awake {
                continue;
            }
            if body.is_dynamic() {
                stack.push(b);
            } else {
                body.awake = true;
                self.set_awake_bit(b, true);
            }
        }
        let asleep = |b: &Body| b.alive && !b.awake && b.is_dynamic();
        let mut near: Vec<u32> = Vec::new();
        while let Some(x) = stack.pop() {
            let body = &mut self.bodies[x as usize];
            if body.awake {
                continue;
            }
            body.awake = true;
            body.sleep_time = 0.0;
            self.set_awake_bit(x, true);
            near.clear();
            self.push_neighbours(x, &mut near);
            stack.extend(
                near.iter()
                    .copied()
                    .filter(|&o| asleep(&self.bodies[o as usize])),
            );
        }
    }

    /// Appends the bodies touching collider `c` through touching solid contacts (repeats
    /// included).
    fn push_touching(&self, c: u32, out: &mut Vec<u32>) {
        for &k in &self.colliders[c as usize].contacts {
            let ct = &self.contacts[k as usize];
            if ct.touching && !ct.sensor {
                out.push(self.colliders[ct.other(c) as usize].body);
            }
        }
    }

    /// Appends the bodies touching body `b` or jointed to it (repeats included).
    fn push_neighbours(&self, b: u32, out: &mut Vec<u32>) {
        for &c in &self.bodies[b as usize].colliders {
            self.push_touching(c, out);
        }
        for &j in &self.bodies[b as usize].joints {
            let d = &self.joints[j as usize].desc;
            for o in [d.body_a.index, d.body_b.index] {
                if o != b {
                    out.push(o);
                }
            }
        }
    }

    /// Wakes the dynamic bodies touching `b` or jointed to it (before `b` goes away, moves or
    /// changes shape), with their islands.
    fn wake_neighbours(&mut self, b: u32) {
        let mut others = Vec::new();
        self.push_neighbours(b, &mut others);
        others.retain(|&o| self.bodies[o as usize].is_dynamic());
        self.wake_islands(&others);
    }

    /// Wakes the dynamic bodies touching collider `c` (before it goes away), with their islands;
    /// what touches the body's other colliders is not disturbed (a level losing one of its
    /// pieces wakes only what rested on that piece).
    fn wake_touching_collider(&mut self, c: u32) {
        let mut others = Vec::new();
        self.push_touching(c, &mut others);
        others.retain(|&o| self.bodies[o as usize].is_dynamic());
        self.wake_islands(&others);
    }

    /// Wakes the sleeping dynamic bodies (with their islands) whose colliders' broad-phase boxes
    /// overlap those of `colliders` and that would collide with them: what static geometry moved
    /// or grown into them must react, as static bodies never wake anything through contacts
    /// (whose pairs with sleeping bodies are not even updated).
    fn wake_overlapping(&mut self, colliders: &[u32]) {
        let mut found: Vec<u32> = Vec::new();
        for &c in colliders {
            let col = &self.colliders[c as usize];
            self.moving_tree.query(&col.fat, &mut |_, o| {
                let ob = self.colliders[o as usize].body;
                let other = &self.bodies[ob as usize];
                if ob != col.body && other.is_dynamic() && !other.awake && self.should_collide(c, o)
                {
                    found.push(ob);
                }
                true
            });
        }
        self.wake_islands(&found);
    }

    /// Continuous collision: a dynamic body that moved farther than half its smallest size this
    /// step (or any bullet) is swept from its start position, in its orientation at the end of
    /// the step, against static and kinematic geometry it collides with (the same filter as
    /// contacts, joints included). A translating kinematic collider can strike even a sleeping
    /// body. Its sweep uses relative translation and pushes the body along the surface normal for
    /// the remaining time; fixed geometry still limits that motion.
    ///
    /// Three kinds of hit are not taken, because the body's contacts already handle them and
    /// taking them would stop it for good (also sliding along): a time of impact of zero (the
    /// shape starts touching, or within the cast's tolerance of touching); against one-sided
    /// triangles, a sweep that neither carries the shape's centre across the triangle's plane nor
    /// ends close to it (a body gliding over a flat floor touches the next triangle at every
    /// seam); and against convex geometry, a target that only the shape's outer skin (the linear
    /// slop) beside its path reaches (a body gliding over a floor of flush boxes, sunk into one by
    /// a fraction of a millimetre, meets the next one's side at every seam). For the first kind,
    /// a small ball about the shape's centre (a quarter of the largest that fits inside it, as in
    /// Box2D) is swept instead, so a body already touching a thin wall still cannot pass through
    /// it. When that ball stops the body, the shape is left deep in the surface (an elongated one
    /// whose near end, touching, turned the approach into spin, by up to three quarters of its
    /// inner radius), so its velocity into the surface goes as well and the next step's contacts
    /// only push it out. Whatever the sweeps take, a centre never passes through a one-sided
    /// triangle from its front (see [`centre_through`]): the narrow phase would lose the body
    /// behind it.
    ///
    /// What the sweep does not see: a turn within the step. A thin body spinning fast enough to
    /// swing through a wall and back out of it within one step is not stopped (Box2D sweeps the
    /// rotation as well).
    fn continuous(&mut self, moved: &[u32], dt: f32) {
        let kinematics = self.kinematic_sweeps(moved);
        // Keep the same stable body order for solver members and newly reachable sleepers.
        let mut members = moved.iter().copied().peekable();
        let mut sleepers = kinematics.sleepers.iter().copied().peekable();
        let candidates = std::iter::from_fn(|| {
            if sleepers
                .peek()
                .is_some_and(|s| members.peek().is_none_or(|m| s < m))
            {
                sleepers.next().map(|b| (b, false))
            } else {
                members.next().map(|b| (b, true))
            }
        });
        let mut hit_events = None;
        for (b, integrated) in candidates {
            self.sweep_body(b, integrated, dt, &kinematics, &mut hit_events);
        }
    }

    fn kinematic_sweeps(&self, moved: &[u32]) -> KinematicSweeps {
        // The normal broad phase indexes the end poses. A translating platform can cross a
        // sleeping body and finish nowhere near it, so index the path of each moving platform.
        let mut kinematics = KinematicSweeps::default();
        let mut search_sleepers = false;
        for &b in moved {
            let body = &self.bodies[b as usize];
            let travel = body.center - body.center0;
            if body.kind != BodyKind::Kinematic || travel == Vec3::ZERO {
                continue;
            }
            kinematics.travel.insert(b, travel);
            for &c in &body.colliders {
                if self.colliders[c as usize].sensor {
                    continue;
                }
                if kinematics.tree.is_empty() {
                    // A scene containing only moving scenery needs no dynamic-response sweeps.
                    let mut dynamics = self
                        .bodies
                        .iter()
                        .filter(|b| b.alive && b.is_dynamic())
                        .peekable();
                    if dynamics.peek().is_none() {
                        return kinematics;
                    }
                    search_sleepers = dynamics.any(|b| !b.awake);
                }
                let swept = self.collider_box(c).sweep(-travel);
                kinematics.tree.insert(swept, c);
                if !search_sleepers {
                    continue;
                }
                // Collect unintegrated bodies at rest in the platform's path. Looking up the
                // sleepers here avoids testing every sleeping collider against a distant lift.
                self.moving_tree
                    .query(&swept.expand(self.settings.linear_slop), &mut |_, o| {
                        let other = &self.colliders[o as usize];
                        let owner = &self.bodies[other.body as usize];
                        if !other.sensor
                            && owner.is_dynamic()
                            && !owner.awake
                            && moved.binary_search(&other.body).is_err()
                        {
                            kinematics.sleepers.insert(other.body);
                        }
                        true
                    });
            }
        }

        kinematics
    }

    fn sweep_body(
        &mut self,
        b: u32,
        integrated: bool,
        dt: f32,
        kinematics: &KinematicSweeps,
        hit_events: &mut HitEventIndex,
    ) {
        let body = &self.bodies[b as usize];
        if !body.is_dynamic() || (!body.awake && kinematics.tree.is_empty()) {
            return;
        }
        // A contact can wake a whole island. Only a body integrated in this step refreshed
        // its starting center; the other sleepers have not moved.
        let start = if integrated {
            body.center0
        } else {
            body.center
        };
        let mut path = SweepInterval {
            center: start,
            travel: body.center - start,
            remaining: 1.0,
            after_hit: false,
        };
        let threshold = 0.5 * body.min_extent.max(self.settings.linear_slop);
        let mut check_fixed = body.bullet || path.travel.length() > threshold;
        if !check_fixed && kinematics.tree.is_empty() {
            return;
        }
        let mut velocity = body.linear_velocity;
        let mask = body.translation_mask();
        let mut pushed = false;
        let mut travel_planes = Vec::new();
        let mut speed_planes = Vec::new();
        // Move chronologically. After a moving obstacle pushes the body, sweep its new
        // path as well: the obstacle must not shove it through a second, fixed wall.
        // If crowded between incompatible obstacles, keep the last safe position.
        for _ in 0..8 {
            let Some(hit) = self.first_ccd_hit(b, &path, check_fixed, kinematics) else {
                path.center += path.travel;
                break;
            };
            let impact = hit.impact;
            let fixed = hit.target_travel == Vec3::ZERO;
            if !fixed && path.remaining > 0.0 {
                let approach = (hit.target_travel * path.remaining - path.travel)
                    .dot(impact.normal)
                    / (dt * path.remaining);
                self.report_swept_hit(hit_events, hit, approach);
            }
            path.after_hit = true;
            path.center += path.travel * impact.fraction;
            path.remaining *= 1.0 - impact.fraction;
            if fixed && !pushed {
                // Preserve the fixed-geometry stop. A core hit also cancels the inward
                // velocity so the next step's solver only has to push out the penetration.
                if impact.core {
                    remove_closing(&mut velocity, Vec3::ZERO, impact.normal, mask);
                }
                path.travel = Vec3::ZERO;
                if kinematics.tree.is_empty() {
                    break;
                }
            } else {
                path.travel *= 1.0 - impact.fraction;
            }
            let target = self.colliders[hit.collider as usize].body;
            if let Some(p) = crate::ccd::Plane::new(impact.normal, hit.target_travel, mask) {
                travel_planes.push(p);
            }
            if let Some(p) = crate::ccd::Plane::new(
                impact.normal,
                self.bodies[target as usize].linear_velocity,
                mask,
            ) {
                speed_planes.push(p);
            }
            pushed |= !fixed;
            if pushed {
                check_fixed = true;
                let cap = self.settings.max_linear_speed;
                let travel = if path.remaining > 0.0 {
                    crate::ccd::permitted(path.travel / path.remaining, &travel_planes, cap * dt)
                        .map(|v| v * path.remaining)
                } else {
                    Some(Vec3::ZERO)
                };
                let speed = crate::ccd::permitted(velocity, &speed_planes, cap);
                let (Some(travel), Some(speed)) = (travel, speed) else {
                    // A prescribed platform can overrun a body pinned by scenery, axis
                    // locks or its speed cap. Keep the last safe pose in that case.
                    velocity = Vec3::ZERO;
                    break;
                };
                path.travel = travel;
                velocity = speed;
            }
        }
        if path.after_hit {
            if pushed {
                self.wake_body(b);
            }
            let body = &mut self.bodies[b as usize];
            body.center = path.center;
            body.position = body.center - body.rotation.rotate(body.local_center);
            body.linear_velocity = velocity;
            self.refresh_body_proxies(b, false);
        }
    }

    /// CCD resolves a moving wall's approach after the discrete solver. Keep the strongest
    /// impact per ordered pair from either source; Begin/End still follow the manifold.
    fn report_swept_hit(&mut self, reported: &mut HitEventIndex, hit: CcdHit, speed: f32) {
        if !speed.is_finite() || speed <= self.settings.hit_event_threshold {
            return;
        }
        let index = reported.get_or_insert_with(|| {
            self.pending
                .contacts
                .iter()
                .enumerate()
                .filter_map(|(i, event)| match event {
                    ContactEvent::Hit { a, b, .. } => Some(((a.index, b.index), i)),
                    _ => None,
                })
                .collect()
        });
        let (a, b, normal) = if hit.collider < hit.moving {
            (hit.collider, hit.moving, hit.impact.normal)
        } else {
            (hit.moving, hit.collider, -hit.impact.normal)
        };
        let event = ContactEvent::Hit {
            a: self.collider_id(a),
            b: self.collider_id(b),
            point: hit.impact.point,
            normal,
            speed,
        };
        if let Some(&i) = index.get(&(a, b)) {
            if matches!(self.pending.contacts[i], ContactEvent::Hit {speed: old, ..} if old < speed)
            {
                self.pending.contacts[i] = event;
            }
        } else {
            index.insert((a, b), self.pending.contacts.len());
            self.pending.contacts.push(event);
        }
    }

    /// Earliest impact along the remaining interval, measured in the target's frame. We hold
    /// both colliders at their end orientations as in the original fixed-geometry sweep.
    fn first_ccd_hit(
        &self,
        b: u32,
        path: &SweepInterval,
        check_fixed: bool,
        kinematics: &KinematicSweeps,
    ) -> Option<CcdHit> {
        let body = &self.bodies[b as usize];
        let slop = self.settings.linear_slop;
        let start = Iso::new(
            path.center - body.rotation.rotate(body.local_center),
            body.rotation,
        );
        let mut best: Option<CcdHit> = None;
        let mut candidates = SmallBuf::<(u32, Vec3), 32>::new((0, Vec3::ZERO));
        for &c in &body.colliders {
            let col = &self.colliders[c as usize];
            if col.sensor {
                continue;
            }
            let Some(shape) = col.shape.as_convex() else {
                continue;
            };
            let from = start.mul(&col.local);
            let swept = col.shape.aabb(&from).sweep(path.travel).expand(slop);
            candidates.clear();
            if check_fixed && path.travel != Vec3::ZERO {
                let mut fixed = |_: u32, o: u32| {
                    let owner = self.colliders[o as usize].body;
                    if !self.bodies[owner as usize].is_dynamic()
                        && !kinematics.travel.contains_key(&owner)
                    {
                        candidates.push((o, Vec3::ZERO));
                    }
                    true
                };
                self.static_tree.query(&swept, &mut fixed);
                self.moving_tree.query(&swept, &mut fixed);
            }
            kinematics.tree.query(&swept, &mut |_, o| {
                let owner = self.colliders[o as usize].body;
                candidates.push((o, kinematics.travel[&owner]));
                true
            });
            if candidates.is_empty() {
                continue;
            }
            let moving = Convex::new(shape, from);
            let center0 = moving.center();
            let core_radius = 0.25 * col.shape.inner_radius();
            let core = Convex::new(
                ConvexShape::Sphere(core_radius),
                Iso::from_position(center0),
            );
            // Shave the skin beside the path so a seam between flush boxes is not a wall.
            let skin = slop.min(0.75 * col.shape.inner_radius());
            let mut shrunk_verts = [Vec3::ZERO; crate::hull::MAX_HULL_VERTICES];
            let shrunk_hull_len = if let ConvexShape::Hull(h) = shape {
                let farthest = h.outer_face_dist;
                let k = if farthest > skin {
                    1.0 - skin / farthest
                } else {
                    1.0
                };
                let n = h.vertices.len();
                for (dst, &v) in shrunk_verts[..n].iter_mut().zip(&h.vertices) {
                    *dst = h.centroid + (v - h.centroid) * k;
                }
                Some((n, h.centroid))
            } else {
                None
            };
            let inner_shape = match shape {
                ConvexShape::Sphere(r) => ConvexShape::Sphere(r - skin),
                ConvexShape::Capsule {
                    half_height,
                    radius,
                } => ConvexShape::Capsule {
                    half_height,
                    radius: radius - skin,
                },
                ConvexShape::Box(h) => ConvexShape::Box(h - Vec3::splat(skin)),
                other => other,
            };
            for &(other, target_travel) in candidates.as_slice() {
                let oc = &self.colliders[other as usize];
                if oc.sensor || !self.should_collide(c, other) {
                    continue;
                }
                let remaining_target = target_travel * path.remaining;
                let relative = path.travel - remaining_target;
                let (direction, distance) = relative.normalize_and_length();
                if distance == 0.0
                    || (!body.bullet && !check_fixed && distance <= 0.5 * body.min_extent.max(slop))
                {
                    continue;
                }
                let mut at = self.collider_iso(other);
                at.position -= remaining_target;
                let mut take = |impact: Option<Impact>| {
                    let Some(impact) = impact else { return };
                    if impact.fraction >= 1.0
                        || (target_travel != Vec3::ZERO
                            && impact
                                .normal
                                .mul_elem(body.translation_mask())
                                .length_squared()
                                < 1e-12)
                    {
                        return;
                    }
                    if best.is_none_or(|old| {
                        impact.fraction < old.impact.fraction
                            || (impact.fraction == old.impact.fraction && other < old.collider)
                    }) {
                        best = Some(CcdHit {
                            impact: Impact {
                                // The relative cast fixed B at its interval-start pose.
                                // Bring its surface witness forward to the instant of contact.
                                point: impact.point + remaining_target * impact.fraction,
                                ..impact
                            },
                            moving: c,
                            collider: other,
                            target_travel,
                        });
                    }
                };
                let mut triangle = |t: &TriangleRef| {
                    let (n, v0) = (at.vector(t.normal()), at.point(t.vertices[0]));
                    let (s0, s1) = (n.dot(center0 - v0), n.dot((center0 + relative) - v0));
                    if s0 >= 0.0 && (s0 - s1 >= core_radius || s1 <= core_radius) {
                        let target = Convex::new(ConvexShape::Triangle(t.vertices), at);
                        take(time_of_impact(
                            &moving,
                            &core,
                            relative,
                            &target,
                            slop,
                            path.after_hit,
                        ));
                        take(centre_through(
                            t,
                            &at,
                            center0,
                            relative,
                            core_radius + slop,
                        ));
                    }
                    true
                };
                match &oc.shape {
                    Shape::Mesh(m) => {
                        let local = crate::shape::transform_aabb(&swept, &at.inverse());
                        m.query_aabb(&local, &mut triangle);
                    }
                    Shape::Heightfield(h) => {
                        let local = crate::shape::transform_aabb(&swept, &at.inverse());
                        h.query_aabb(&local, &mut triangle);
                    }
                    other_shape => {
                        if let Some(os) = other_shape.as_convex() {
                            let target = Convex::new(os, at);
                            let front =
                                |s: &Convex| s.support_core(direction).dot(direction) + s.radius();
                            let inner_hit = if let Some((n, centroid)) = shrunk_hull_len {
                                let verts = &shrunk_verts[..n];
                                let local_d = from.inverse_vector(direction);
                                let mut best_i = 0;
                                let mut best_d = f32::NEG_INFINITY;
                                for (i, v) in verts.iter().enumerate() {
                                    let p = v.dot(local_d);
                                    if p > best_d {
                                        best_d = p;
                                        best_i = i;
                                    }
                                }
                                let inner_front = from.point(verts[best_i]).dot(direction);
                                let ahead = direction * (front(&moving) - inner_front);
                                gjk::cast_hull_vertices(
                                    verts,
                                    centroid,
                                    Iso::new(from.position + ahead, from.rotation),
                                    relative,
                                    &target,
                                    0.0,
                                )
                                .is_some()
                            } else {
                                let ahead = direction
                                    * (front(&moving) - front(&Convex::new(inner_shape, from)));
                                let inner = Convex::new(
                                    inner_shape,
                                    Iso::new(from.position + ahead, from.rotation),
                                );
                                gjk::cast(&inner, relative, &target, 0.0).is_some()
                            };
                            if inner_hit {
                                take(time_of_impact(
                                    &moving,
                                    &core,
                                    relative,
                                    &target,
                                    slop,
                                    path.after_hit,
                                ));
                            }
                        }
                    }
                }
            }
        }
        best
    }

    /// A hash of every body's pose and velocity (bit patterns, index order): equal hashes after
    /// equal inputs show the simulation is deterministic.
    pub fn state_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |x: f32| {
            for byte in x.to_bits().to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for b in self.bodies.iter().filter(|b| b.alive) {
            for v in [b.position, b.linear_velocity, b.angular_velocity] {
                eat(v.x);
                eat(v.y);
                eat(v.z);
            }
            for q in b.rotation.to_array() {
                eat(q);
            }
        }
        h
    }
}

#[cfg(test)]
mod tests {
    //! White-box tests of the world's bookkeeping that the narrow phase would otherwise have to
    //! set up: contacts are marked touching by hand, as a settled scene's would be.

    use super::*;

    const DT: f32 = 1.0 / 60.0;

    fn cuboid(w: &mut PhysicsWorld, kind: BodyKind, position: Vec3, half: Vec3) -> BodyId {
        let b = w
            .create_body(&BodyDesc {
                kind,
                position,
                ..Default::default()
            })
            .unwrap();
        w.add_collider(b, &ColliderDesc::new(Shape::Box { half_extents: half }))
            .unwrap();
        b
    }

    /// Steps once so the broad phase pairs every collider, then marks touching exactly the
    /// contacts between the given bodies and puts every body to sleep: a settled scene.
    fn settle(w: &mut PhysicsWorld, touching: &[(BodyId, BodyId)]) {
        w.step(DT);
        let bodies: Vec<(u32, u32)> = w
            .contacts
            .iter()
            .map(|c| {
                (
                    w.colliders[c.a as usize].body,
                    w.colliders[c.b as usize].body,
                )
            })
            .collect();
        for (c, (p, q)) in w.contacts.iter_mut().zip(bodies) {
            c.touching = touching
                .iter()
                .any(|(x, y)| (x.index, y.index) == (p, q) || (y.index, x.index) == (p, q));
        }
        for b in &mut w.bodies {
            b.awake = false;
            b.sleep_time = 0.0;
            b.linear_velocity = Vec3::ZERO;
            b.angular_velocity = Vec3::ZERO;
        }
        w.awake_words.fill(0);
    }

    /// End events of contacts removed between steps (here: a visitor destroyed inside a sensor)
    /// reach the next step's events, which a game reads after each step; the last step's events
    /// stay as they were until then (review finding EVENTS-LOST).
    #[test]
    fn end_events_from_calls_between_steps_reach_the_next_steps_events() {
        let mut w = PhysicsWorld::new(Settings {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let zone = w
            .create_body(&BodyDesc {
                kind: BodyKind::Static,
                ..Default::default()
            })
            .unwrap();
        let mut d = ColliderDesc::new(Shape::Box {
            half_extents: Vec3::splat(2.0),
        });
        d.sensor = true;
        let sensor = w.add_collider(zone, &d).unwrap();
        // A box resting on the ground (top at y = -0.5), inside the zone.
        let ground = cuboid(
            &mut w,
            BodyKind::Static,
            Vec3::new(0.0, -1.5, 0.0),
            Vec3::splat(1.0),
        );
        let ball = cuboid(&mut w, BodyKind::Dynamic, Vec3::ZERO, Vec3::splat(0.5));
        settle(&mut w, &[(ground, ball)]);
        assert_eq!(
            w.contacts.len(),
            2,
            "the ground and the zone pair with the box"
        );
        // The visitor inside the zone (sensor contacts are marked as the narrow phase would).
        let (z, g) = (zone.index, ground.index);
        for c in &mut w.contacts {
            let bodies = (
                w.colliders[c.a as usize].body,
                w.colliders[c.b as usize].body,
            );
            if bodies.0 == z || bodies.1 == z {
                c.touching = true;
            }
            assert!(bodies.0 == g || bodies.1 == g || c.touching);
        }
        let visitor = w.bodies[ball.index as usize].colliders[0];
        let visitor = w.collider_id(visitor);
        let ground_collider = w.collider_id(w.bodies[g as usize].colliders[0]);
        w.step(DT);
        assert!(w.events().sensors.is_empty() && w.events().contacts.is_empty());
        w.destroy_body(ball);
        assert!(w.events().sensors.is_empty() && w.events().contacts.is_empty());
        // A step that is not one (no time passes) changes nothing.
        w.step(0.0);
        w.step(f32::NAN);
        w.step(DT);
        assert_eq!(
            w.events().sensors,
            vec![SensorEvent::End { sensor, visitor }]
        );
        assert_eq!(
            w.events().contacts,
            vec![ContactEvent::End {
                a: ground_collider,
                b: visitor
            }]
        );
        w.step(DT);
        assert!(w.events().sensors.is_empty() && w.events().contacts.is_empty());
    }

    /// A body's inverse inertia, turned with it within a step: the body's own world inverse
    /// inertia at the turned orientation; locked rows and columns stay exact zeros; a ball's or a
    /// cube's is the same up to a rounding; and a body that may turn about one axis only keeps
    /// the one it has.
    #[test]
    fn inverse_inertia_turns_with_the_body_and_keeps_locked_rows_zero() {
        let mut w = PhysicsWorld::default();
        let dq = Quat::from_axis_angle(Vec3::new(0.3, -0.8, 0.5).normalize_or_zero(), 0.9);
        let start = Quat::from_axis_angle(Vec3::new(1.0, 0.2, -0.4).normalize_or_zero(), 0.6);
        let close = |a: &Mat3, b: &Mat3, tol: f32| {
            (0..3).all(|i| (a.cols[i] - b.cols[i]).length() <= tol * b.cols[i].length().max(1.0))
        };
        for lock in [
            [false; 3],
            [false, true, false],
            [true, false, false],
            [false, false, true],
            [true, false, true],
            [true; 3],
        ] {
            let b = w
                .create_body(&BodyDesc {
                    rotation: start,
                    lock_rotation: lock,
                    ..Default::default()
                })
                .unwrap();
            w.add_collider(
                b,
                &ColliderDesc::new(Shape::Box {
                    half_extents: Vec3::new(1.0, 0.05, 0.3),
                }),
            )
            .unwrap();
            // A body free about one axis only can turn about that axis only.
            let free = mask(lock);
            let dq = if free.x + free.y + free.z == 1.0 {
                Quat::from_axis_angle(free, 0.9)
            } else {
                dq
            };
            let body = w.bodies[b.index as usize].clone();
            let turned = Body {
                rotation: dq * start,
                ..body.clone()
            };
            let expected = turned.world_inv_inertia();
            let got = body.turning().after(dq).unwrap_or(body.world_inv_inertia());
            assert!(
                close(&got, &expected, 1e-4),
                "{lock:?}: {got:?} vs {expected:?}"
            );
            for k in 0..3 {
                if lock[k] {
                    assert!(
                        got.cols[k] == Vec3::ZERO && got.row(k) == Vec3::ZERO,
                        "{lock:?}: {got:?}"
                    );
                }
            }
        }
        // A ball and a cube: turned, the same to a rounding.
        for shape in [
            Shape::Sphere { radius: 0.4 },
            Shape::Box {
                half_extents: Vec3::splat(0.3),
            },
        ] {
            let b = w
                .create_body(&BodyDesc {
                    rotation: start,
                    ..Default::default()
                })
                .unwrap();
            w.add_collider(b, &ColliderDesc::new(shape)).unwrap();
            let body = &w.bodies[b.index as usize];
            let own = body.world_inv_inertia();
            let turned = body.turning().after(dq).unwrap();
            assert!(close(&turned, &own, 1e-6), "{turned:?} vs {own:?}");
        }
    }

    /// A body's reach (for the sleep test) is measured from its centre of mass, over its solid
    /// colliders, to the farthest point of each shape; its smallest size (for continuous
    /// collision) is that of its thinnest solid collider (review finding MAX-EXTENT-ORIGIN).
    #[test]
    fn extents_are_measured_from_the_centre_of_mass_over_solid_colliders() {
        let mut w = PhysicsWorld::default();
        let b = w.create_body(&BodyDesc::default()).unwrap();
        let mut ball = ColliderDesc::new(Shape::Sphere { radius: 0.5 });
        ball.local = Iso::from_position(Vec3::new(10.0, 0.0, 0.0));
        w.add_collider(b, &ball).unwrap();
        let mut sensor = ColliderDesc::new(Shape::Box {
            half_extents: Vec3::splat(0.01),
        });
        sensor.local = Iso::from_position(Vec3::new(-30.0, 0.0, 0.0));
        sensor.sensor = true;
        w.add_collider(b, &sensor).unwrap();
        let body = &w.bodies[b.index as usize];
        assert_eq!(body.local_center, Vec3::new(10.0, 0.0, 0.0));
        assert!((body.max_extent - 0.5).abs() < 1e-5, "{}", body.max_extent);
        assert!((body.min_extent - 0.5).abs() < 1e-5, "{}", body.min_extent);
        // Two boxes either side of the centre: the farthest corner, and the thinner box.
        let b = w.create_body(&BodyDesc::default()).unwrap();
        for (x, half) in [
            (-1.0f32, Vec3::new(0.5, 0.1, 0.2)),
            (1.0, Vec3::new(0.5, 0.3, 0.2)),
        ] {
            let mut d = ColliderDesc::new(Shape::Box { half_extents: half });
            d.local = Iso::from_position(Vec3::new(x, 0.0, 0.0));
            w.add_collider(b, &d).unwrap();
        }
        let body = &w.bodies[b.index as usize];
        // The heavier box pulls the centre to x = 0.5, so the lighter box's far corner is the
        // farthest point.
        let c = body.local_center;
        assert!((c - Vec3::new(0.5, 0.0, 0.0)).length() < 1e-5, "{c:?}");
        let far = (Vec3::new(-1.5, 0.1, 0.2) - c).length();
        assert!(
            (body.max_extent - far).abs() < 1e-5,
            "{} vs {far}",
            body.max_extent
        );
        assert!((body.min_extent - 0.1).abs() < 1e-6, "{}", body.min_extent);
    }

    /// Carried-over impulses are per substep: a step after the substep count changed starts
    /// from them rescaled to the new count (part of review finding SETTINGS-UNVALIDATED).
    #[test]
    fn changing_the_substep_count_rescales_the_carried_impulses() {
        let mut w = PhysicsWorld::new(Settings {
            enable_sleep: false,
            ..Default::default()
        });
        let hook = cuboid(
            &mut w,
            BodyKind::Static,
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::splat(0.1),
        );
        let lamp = cuboid(
            &mut w,
            BodyKind::Dynamic,
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::splat(0.1),
        );
        w.create_joint(&JointDesc {
            body_a: hook,
            body_b: lamp,
            frame_a: Iso::from_position(Vec3::new(0.0, -1.0, 0.0)),
            frame_b: Iso::IDENTITY,
            kind: joint::JointKind::Ball,
            collide_connected: false,
            user_data: 0,
        })
        .unwrap();
        for _ in 0..60 {
            w.step(DT);
        }
        assert_eq!(w.warm_substeps, 4);
        // A contact's carried impulses, as the narrow phase would have left them.
        let mut cm = ContactManifold::default();
        cm.manifold.count = 1;
        cm.cache[0].normal_impulse = 2.0;
        cm.cache[0].friction_impulse = Vec3::new(0.5, 0.0, -1.0);
        w.contacts.push(Contact {
            a: 0,
            b: 1,
            manifolds: vec![cm],
            touching: false,
            sensor: false,
            slots: [0, 0],
        });
        let held = w.joints[0].impulses.linear;
        let weight = 1000.0 * 0.008 * 9.81 / 60.0 / 4.0;
        assert!((held.y.abs() - weight).abs() < 0.05 * weight, "{held:?}");
        w.settings.substeps = 2;
        w.rescale_warm_starts(2);
        assert_eq!(w.joints[0].impulses.linear, held * 2.0);
        let p = w.contacts.last().unwrap().manifolds[0].cache[0];
        assert_eq!(p.normal_impulse, 4.0);
        assert_eq!(p.friction_impulse, Vec3::new(1.0, 0.0, -2.0));
        w.contacts.pop();
        // The step itself does it: back to four substeps, half the impulse again.
        w.settings.substeps = 4;
        w.step(DT);
        assert_eq!(w.warm_substeps, 4);
        let after = w.joints[0].impulses.linear;
        assert!(
            (after - held).length() < 0.05 * held.length(),
            "{after:?} vs {held:?}"
        );
    }

    /// A manifold with the given normal and points (position, feature id), and the impulses
    /// last step left on them.
    fn manifold(normal: Vec3, points: &[(Vec3, u32, f32)]) -> ContactManifold {
        let mut cm = ContactManifold::default();
        cm.manifold.normal = normal;
        for (i, &(p, id, impulse)) in points.iter().enumerate() {
            cm.manifold.points[i] = collide::ManifoldPoint {
                point: p,
                separation: 0.0,
                id,
            };
            cm.cache[i] = CachedPoint {
                id,
                local_a: p,
                normal_impulse: impulse,
                friction_impulse: Vec3::new(impulse, 0.0, 0.0),
            };
        }
        cm.manifold.count = points.len() as u8;
        cm
    }

    /// The impulses each new point starts with.
    fn carried(old: &[ContactManifold], fresh: &[ContactManifold]) -> Vec<Vec<f32>> {
        let fresh: Vec<Manifold> = fresh.iter().map(|m| m.manifold.clone()).collect();
        let mut out = Vec::new();
        carry_impulses_into(old, &fresh, &Iso::IDENTITY, 0.045, &mut out);
        out.iter()
            .map(|m| {
                m.cache[..usize::from(m.manifold.count)]
                    .iter()
                    .map(|c| {
                        assert_eq!(c.friction_impulse.x, c.normal_impulse);
                        c.normal_impulse
                    })
                    .collect()
            })
            .collect()
    }

    /// Warm starting hands each old point's impulse to at most one new point: a box held by one
    /// point that settles onto four 2 cm apart started with four times the impulse that held it
    /// (review finding WARM-MATCH).
    #[test]
    fn carried_impulses_go_to_one_new_point_each() {
        let p = |x: f32, z: f32| Vec3::new(x, 0.0, z);
        let one = [manifold(Vec3::Y, &[(p(0.0, 0.0), 0, 5.0)])];
        let four = [manifold(
            Vec3::Y,
            &[
                (p(-0.01, -0.01), 0, 0.0),
                (p(0.01, -0.01), 0, 0.0),
                (p(0.01, 0.011), 0, 0.0),
                (p(-0.01, 0.01), 0, 0.0),
            ],
        )];
        let got = carried(&one, &four);
        assert_eq!(got.concat().iter().sum::<f32>(), 5.0, "{got:?}");
        assert_eq!(
            got,
            vec![vec![5.0, 0.0, 0.0, 0.0]],
            "the nearest (first on ties) takes it"
        );
        // The nearest pairs claim first: the second new point is nearer to the only old point.
        let near = [manifold(
            Vec3::Y,
            &[(p(0.03, 0.0), 0, 0.0), (p(0.001, 0.0), 0, 0.0)],
        )];
        assert_eq!(carried(&one, &near), vec![vec![0.0, 5.0]]);
        // Two old points, two new ones that both are nearest to the first old one.
        let two = [manifold(
            Vec3::Y,
            &[(p(0.0, 0.0), 0, 5.0), (p(0.04, 0.0), 0, 3.0)],
        )];
        let both = [manifold(
            Vec3::Y,
            &[(p(0.005, 0.0), 0, 0.0), (p(0.012, 0.0), 0, 0.0)],
        )];
        assert_eq!(carried(&two, &both), vec![vec![5.0, 3.0]]);
        // A feature id takes the old point with the same id first, once, wherever it is; the
        // points left over then match by position whatever their ids: the second id-7 point
        // takes the id-9 point where it was, and the id-8 point is too far from anything left.
        let ids = [manifold(
            Vec3::Y,
            &[(p(0.0, 0.0), 7, 5.0), (p(0.3, 0.0), 9, 2.0)],
        )];
        let renumbered = [manifold(
            Vec3::Y,
            &[
                (p(0.3, 0.0), 7, 0.0),
                (p(0.0, 0.0), 8, 0.0),
                (p(0.3, 0.0), 7, 0.0),
            ],
        )];
        assert_eq!(carried(&ids, &renumbered), vec![vec![5.0, 0.0, 2.0]]);
        // Ids that changed between steps (the incident vertices of a box resting on an equal,
        // aligned box lie on the side planes, so their clip ids flicker): matched by position.
        let before = [manifold(
            Vec3::Y,
            &[(p(-0.25, -0.25), 3, 1.0), (p(0.25, -0.25), 5, 2.0)],
        )];
        let after = [manifold(
            Vec3::Y,
            &[(p(0.2501, -0.25), 6, 0.0), (p(-0.2499, -0.25), 4, 0.0)],
        )];
        assert_eq!(carried(&before, &after), vec![vec![2.0, 1.0]]);
        // A point does not take one from a manifold facing another way (a neighbouring triangle
        // across a crease), but takes one with another id from a manifold facing its own way.
        let crease = [
            manifold(Vec3::X, &[(p(0.0, 0.0), 0, 5.0)]),
            manifold(Vec3::Y, &[(p(0.01, 0.0), 4, 2.0)]),
        ];
        let flat = [manifold(Vec3::Y, &[(p(0.0, 0.0), 0, 0.0)])];
        assert_eq!(carried(&crease, &flat), vec![vec![2.0]]);
        let crease = [manifold(Vec3::X, &[(p(0.0, 0.0), 0, 5.0)])];
        assert_eq!(carried(&crease, &flat), vec![vec![0.0]]);
        // Rounded shapes have the same feature id as their contact normal turns. Carrying an
        // impulse from the floor onto a wall would turn that support into a sideways push.
        let old = [manifold(Vec3::Y, &[(Vec3::ZERO, 7, 5.0)])];
        let sideways = [manifold(Vec3::X, &[(Vec3::ZERO, 7, 0.0)])];
        assert_eq!(carried(&old, &sideways), vec![vec![0.0]]);
    }

    /// Removing one collider of a static body (a piece of a level) wakes what rested on that
    /// collider, not everything resting on the body's other colliders.
    #[test]
    fn removing_a_piece_of_static_geometry_wakes_only_what_rested_on_it() {
        let mut w = PhysicsWorld::default();
        let level = w
            .create_body(&BodyDesc {
                kind: BodyKind::Static,
                ..Default::default()
            })
            .unwrap();
        let mut pieces = Vec::new();
        for x in [-5.0f32, 5.0] {
            let mut d = ColliderDesc::new(Shape::Box {
                half_extents: Vec3::new(1.0, 0.5, 1.0),
            });
            d.local = Iso::from_position(Vec3::new(x, -0.5, 0.0));
            pieces.push(w.add_collider(level, &d).unwrap());
        }
        let half = Vec3::splat(0.25);
        let left = cuboid(&mut w, BodyKind::Dynamic, Vec3::new(-5.0, 0.25, 0.0), half);
        let right = cuboid(&mut w, BodyKind::Dynamic, Vec3::new(5.0, 0.25, 0.0), half);
        settle(&mut w, &[(level, left), (level, right)]);
        assert_eq!(w.contact_count(), 2);
        w.remove_collider(pieces[0]);
        assert!(w.is_awake(left), "the box on the removed piece sleeps on");
        assert!(!w.is_awake(right), "the box on the other piece woke");
    }

    /// Waking a box wakes what it touches, transitively, but not through a kinematic platform:
    /// the box at the platform's other end stays asleep, and so does the platform (review finding
    /// WAKE-THROUGH-KINEMATIC).
    #[test]
    fn waking_spreads_through_dynamic_bodies_only() {
        let mut w = PhysicsWorld::new(Settings {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let k = cuboid(
            &mut w,
            BodyKind::Kinematic,
            Vec3::new(0.0, -0.25, 0.0),
            Vec3::new(10.0, 0.25, 3.0),
        );
        let half = Vec3::splat(0.25);
        let a = cuboid(&mut w, BodyKind::Dynamic, Vec3::new(-8.0, 0.25, 0.0), half);
        let stack: Vec<BodyId> = (1..6)
            .map(|i| {
                let y = 0.25 + 0.5 * i as f32;
                cuboid(&mut w, BodyKind::Dynamic, Vec3::new(-8.0, y, 0.0), half)
            })
            .collect();
        let b = cuboid(&mut w, BodyKind::Dynamic, Vec3::new(8.0, 0.25, 0.0), half);
        let mut touching = vec![(k, a), (k, b), (a, stack[0])];
        touching.extend(stack.windows(2).map(|p| (p[0], p[1])));
        settle(&mut w, &touching);
        assert_eq!(w.contact_count(), touching.len());
        w.wake(a);
        assert!(w.is_awake(a));
        for (i, &s) in stack.iter().enumerate() {
            assert!(w.is_awake(s), "box {i} of the stack on a");
        }
        assert!(!w.is_awake(k), "the platform woke");
        assert!(!w.is_awake(b), "the box at the platform's other end woke");
        // Waking the platform itself (the game moves it) wakes only the platform; its moving
        // contacts wake what rests on it in the next step.
        w.wake(b);
        let b_on = w.is_awake(b);
        settle(&mut w, &touching);
        w.wake(k);
        assert!(b_on && w.is_awake(k) && !w.is_awake(a) && !w.is_awake(b));
    }

    /// Every contact sits in its two colliders' lists at the places it records and in the pair
    /// map under its index; the lists hold nothing else; a destroyed collider's list is empty;
    /// the move buffer holds each queued collider once, and only those.
    fn check_bookkeeping(w: &PhysicsWorld) {
        for (k, c) in w.contacts.iter().enumerate() {
            assert!(c.a < c.b, "contact {k}: {} {}", c.a, c.b);
            for (end, col) in [c.a, c.b].into_iter().enumerate() {
                let list = &w.colliders[col as usize].contacts;
                assert_eq!(
                    list.get(c.slots[end] as usize),
                    Some(&(k as u32)),
                    "contact {k}"
                );
            }
            assert_eq!(w.pairs.get(&(c.a, c.b)), Some(&(k as u32)));
        }
        assert_eq!(w.pairs.len(), w.contacts.len());
        let mut entries = 0;
        for (i, col) in w.colliders.iter().enumerate() {
            let i = i as u32;
            assert!(col.alive || col.contacts.is_empty(), "dead collider {i}");
            for &k in &col.contacts {
                let c = &w.contacts[k as usize];
                assert!(c.a == i || c.b == i, "collider {i} lists contact {k}");
            }
            entries += col.contacts.len();
            let queued = w.move_buffer.iter().filter(|&&x| x == i).count();
            assert_eq!(
                queued,
                usize::from(col.queued),
                "collider {i} in the move buffer"
            );
        }
        assert_eq!(entries, 2 * w.contacts.len());
        let expected_kinematic = w
            .bodies
            .iter()
            .filter(|b| b.alive && b.kind == BodyKind::Kinematic)
            .count();
        assert_eq!(w.kinematic_count, expected_kinematic, "kinematic_count");
        for (i, b) in w.bodies.iter().enumerate() {
            let expected = b.alive && b.awake && b.moves();
            let got = w
                .awake_words
                .get(i >> 6)
                .is_some_and(|&word| (word & (1u64 << (i & 63))) != 0);
            assert_eq!(got, expected, "awake_words bit for body {i}");
        }
    }

    /// A small deterministic random sequence.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, n: u32) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 33) % n as u64) as u32
        }
        fn unit(&mut self) -> f32 {
            self.next(1 << 20) as f32 / (1 << 20) as f32
        }
    }

    /// Contacts are found through each collider's own list (review findings: set_body_pose
    /// and wake_body scanning every contact, J6, F3). Random creation, destruction, joints,
    /// teleports, sensors and steps in a crowded space keep the lists exact.
    #[test]
    fn contact_lists_stay_exact_through_random_churn() {
        let mut w = PhysicsWorld::default();
        let ground = cuboid(
            &mut w,
            BodyKind::Static,
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(4.0, 0.5, 4.0),
        );
        let mut rng = Lcg(7);
        let mut bodies: Vec<BodyId> = Vec::new();
        let mut colliders: Vec<ColliderId> = Vec::new();
        let mut joints: Vec<JointId> = Vec::new();
        let (mut most_contacts, mut destroyed) = (0, 0);
        for round in 0..600 {
            let at =
                |r: &mut Lcg| Vec3::new(4.0 * r.unit() - 2.0, 2.0 * r.unit(), 4.0 * r.unit() - 2.0);
            match rng.next(12) {
                0..=2 => {
                    let kind = [
                        BodyKind::Dynamic,
                        BodyKind::Dynamic,
                        BodyKind::Kinematic,
                        BodyKind::Static,
                    ][rng.next(4) as usize];
                    let b = w
                        .create_body(&BodyDesc {
                            kind,
                            position: at(&mut rng),
                            awake: rng.next(3) != 0,
                            ..Default::default()
                        })
                        .unwrap();
                    for _ in 0..1 + rng.next(3) {
                        let mut d = ColliderDesc::new(Shape::Box {
                            half_extents: Vec3::splat(0.2 + 0.3 * rng.unit()),
                        });
                        d.local = Iso::from_position(Vec3::new(0.3 * rng.unit(), 0.0, 0.0));
                        d.sensor = rng.next(6) == 0;
                        colliders.push(w.add_collider(b, &d).unwrap());
                    }
                    bodies.push(b);
                }
                3 if !colliders.is_empty() => {
                    let c = colliders.swap_remove(rng.next(colliders.len() as u32) as usize);
                    w.remove_collider(c);
                }
                4 if !bodies.is_empty() => {
                    let b = bodies.swap_remove(rng.next(bodies.len() as u32) as usize);
                    w.destroy_body(b);
                    destroyed += 1;
                }
                5 if bodies.len() >= 2 => {
                    let a = bodies[rng.next(bodies.len() as u32) as usize];
                    let b = bodies[rng.next(bodies.len() as u32) as usize];
                    let other = if rng.next(4) == 0 { ground } else { b };
                    if let Ok(j) = w.create_joint(&JointDesc {
                        body_a: a,
                        body_b: other,
                        frame_a: Iso::IDENTITY,
                        frame_b: Iso::IDENTITY,
                        kind: joint::JointKind::Ball,
                        collide_connected: rng.next(2) == 0,
                        user_data: 0,
                    }) {
                        joints.push(j);
                    }
                }
                6 if !joints.is_empty() => {
                    let j = joints.swap_remove(rng.next(joints.len() as u32) as usize);
                    w.destroy_joint(j);
                }
                7 if !bodies.is_empty() => {
                    let b = bodies[rng.next(bodies.len() as u32) as usize];
                    w.set_body_pose(b, Iso::from_position(at(&mut rng)));
                }
                8 if !bodies.is_empty() => {
                    let b = bodies[rng.next(bodies.len() as u32) as usize];
                    w.apply_impulse_at(b, Vec3::new(0.0, 1.0, 0.0), at(&mut rng));
                }
                _ => w.step(DT),
            }
            colliders.retain(|&c| w.collider_index(c).is_some());
            joints.retain(|&j| w.joint_index(j).is_some());
            check_bookkeeping(&w);
            most_contacts = most_contacts.max(w.contacts.len());
            assert!(w.state_hash() != 0, "round {round}");
        }
        assert!(
            most_contacts > 40 && destroyed > 20,
            "{most_contacts} contacts, {destroyed} bodies destroyed"
        );
    }

    /// Destroying a joint between a static level and a lamp queues the lamp's colliders for
    /// pair finding, not the level's thousands (review finding J7); the pairs the joint kept
    /// apart are still found again.
    #[test]
    fn destroying_a_joint_queues_only_the_moving_bodys_colliders() {
        let mut w = PhysicsWorld::new(Settings {
            gravity: Vec3::ZERO,
            ..Default::default()
        });
        let level = w
            .create_body(&BodyDesc {
                kind: BodyKind::Static,
                ..Default::default()
            })
            .unwrap();
        for i in 0..50 {
            let mut d = ColliderDesc::new(Shape::Box {
                half_extents: Vec3::new(0.5, 0.5, 0.5),
            });
            d.local = Iso::from_position(Vec3::new(i as f32, 0.0, 0.0));
            w.add_collider(level, &d).unwrap();
        }
        // A crate sunk 5 mm into the first tile, fixed to the level without colliding.
        let lamp = cuboid(
            &mut w,
            BodyKind::Dynamic,
            Vec3::new(0.0, 0.795, 0.0),
            Vec3::splat(0.3),
        );
        let j = w
            .create_joint(&JointDesc {
                body_a: level,
                body_b: lamp,
                frame_a: Iso::from_position(Vec3::new(0.0, 0.795, 0.0)),
                frame_b: Iso::IDENTITY,
                kind: joint::JointKind::Fixed,
                collide_connected: false,
                user_data: 0,
            })
            .unwrap();
        w.step(DT);
        assert!(
            w.contacts.is_empty(),
            "the joint keeps the crate and the level apart"
        );
        let crate_collider = w.bodies[lamp.index as usize].colliders[0];
        w.destroy_joint(j);
        assert_eq!(w.move_buffer, vec![crate_collider]);
        check_bookkeeping(&w);
        w.step(DT);
        assert_eq!(w.contact_count(), 1, "the crate and the tile collide again");
        // A joint that never kept them apart queues nothing.
        let j = w
            .create_joint(&JointDesc {
                body_a: level,
                body_b: lamp,
                frame_a: Iso::IDENTITY,
                frame_b: Iso::IDENTITY,
                kind: joint::JointKind::Ball,
                collide_connected: true,
                user_data: 0,
            })
            .unwrap();
        w.step(DT);
        let before = w.move_buffer.clone();
        w.destroy_joint(j);
        assert_eq!(w.move_buffer, before);
    }

    /// The move buffer stays bounded however colliders are churned between steps: a slot reused
    /// while its index is still queued is not queued again.
    #[test]
    fn churning_colliders_between_steps_keeps_the_move_buffer_bounded() {
        let mut w = PhysicsWorld::default();
        let b = w.create_body(&BodyDesc::default()).unwrap();
        let ball = ColliderDesc::new(Shape::Sphere { radius: 0.5 });
        for _ in 0..1000 {
            let c = w.add_collider(b, &ball).unwrap();
            w.remove_collider(c);
        }
        assert_eq!(w.move_buffer.len(), 1);
        check_bookkeeping(&w);
        w.add_collider(b, &ball).unwrap();
        w.step(DT);
        assert!(w.move_buffer.is_empty());
        check_bookkeeping(&w);
    }
}
