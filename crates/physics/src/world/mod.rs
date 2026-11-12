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
}

mod step;
mod sweep;
#[cfg(test)]
mod tests;
