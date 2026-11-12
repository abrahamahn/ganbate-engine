//! A kinematic character controller: an upright capsule that walks, slides along walls, climbs
//! steps up to a height, keeps off slopes it cannot walk, snaps to the ground when walking down,
//! rides moving platforms and pushes dynamic bodies.
//!
//! The character is not a body: the game moves it with [`Character::move_and_slide`] once per
//! step, after [`PhysicsWorld::step`], giving the displacement it wants this step (walking input
//! times speed times `dt`, plus gravity it integrates itself). A move first carries the character
//! with what it stood on, by that body's change of pose since the last move, so a platform's
//! motion counts exactly once and a turning one carries it round its axis. (Moved before the step
//! instead, it follows a step late: a platform rising more than a skin per step overlaps the
//! capsule's bottom by the difference until the next move.) A platform moves its riders by its
//! velocity (`set_kinematic_target` or `set_linear_velocity`). Calling `set_body_pose` teleports
//! the platform and the character stays where it was.
//!
//! The controller sweeps the capsule through the world and stops it `skin` away from every
//! surface, measured straight out from the surface whatever direction the capsule came from, so
//! the next sweep starts clear. A surface it already touches stops only motion into it: walking
//! along a wall or the floor, or away from them, is never held up by them. Nor does a surface the
//! motion would only graze within the casts' tolerance of the skin, such as the ridges of gently
//! rolling terrain it passes over. Each triangle of a mesh or heightfield is swept on its own, and
//! its seam with a coplanar neighbour, or in a valley, is the face itself (as the narrow phase
//! reads such inactive edges), so a floor of small triangles is as smooth as a single one.
//!
//! Where two surfaces facing each other leave no room for a full skin to each (a doorway, a
//! corridor's mouth or a hole barely wider than the capsule), the skin gives way, down to an eighth
//! of itself: the capsule fits through any opening a few millimetres wider than itself (5 mm with
//! the default settings; straight on, or off its middle by centimetres; walking in at a sharp
//! angle takes a centimetre or two more). It keeps that much off both, and is only pushed back
//! out of what comes closer than half of it. Walls that close in on it (the corner of a wedge)
//! are no opening: there it stops a skin from both.
//!
//! Ground is a walkable surface, or the walkable top of an edge under the capsule's rounded bottom
//! (a step's nosing, a ledge), or surfaces that hold the capsule up together though none does
//! alone (a V of slopes too steep to stand on, such a slope against a wall). The rim of a hole the
//! capsule fits through is not ground once the rim is beyond its radius all round: it drops
//! through. (Come down off the hole's middle, it lands on the nearest edge of the rim with its
//! skin, as it lands on a step's nosing; walking over the middle drops it through.) Something
//! that moved into the capsule pushes it back out by the contacts the narrow phase reports.

use crate::gjk;
use crate::math::{Iso, Vec3};
use crate::mesh::TriangleRef;
use crate::shape::{Convex, ConvexShape, Shape, transform_aabb};
use crate::small_buf::SmallBuf;
use crate::world::{BodyId, BodyKind, ColliderId, PhysicsWorld, QueryFilter};

/// A touching surface stops a motion only when the motion heads into it by more than this (the
/// cosine of its angle with the surface): closer to parallel is sliding along it, within the
/// rounding of the projection that made the motion parallel.
const ALONG: f32 = 1e-5;
/// The flattest a contact on the capsule's rounded bottom may face (the up part of its normal)
/// and still hold the character up on the walkable top of an edge.
const LOWEST_SUPPORT: f32 = 0.05;
/// A surface stops a sweep only if the motion would take the capsule this much (metres) inside
/// the skin: twice the casts' tolerance band against triangles and ordinary shapes.
const GRAZE: f32 = 2e-4;
/// Touches whose normals are closer than this (the cosine of about 1°) are one surface when they
/// hold the character up together.
const SAME_SURFACE_COS: f32 = 0.9998;
/// Two surfaces face each other when their normals are more than 120° apart (the cosine of the
/// angle between them below minus this): there may be no room between them for a skin to each.
const OPPOSED: f32 = 0.5;
/// The fraction of the skin the capsule keeps from two surfaces facing each other that leave no
/// room for a full skin each when it walks (or falls, or rides) between them: 2.5 mm by default,
/// so it fits through any opening 5 mm wider than itself.
const SQUEEZE: f32 = 0.125;
/// Something closer than this fraction of the skin crowds the capsule: depenetration takes the
/// capsule out of it. Anything farther is left be, so the capsule may rest closer than a skin,
/// down to [`SQUEEZE`] of it, to surfaces facing each other with no room for more.
const CROWDED: f32 = 0.5 * SQUEEZE;

#[derive(Clone, Debug, PartialEq)]
pub struct CharacterSettings {
    /// Capsule radius (m).
    pub radius: f32,
    /// Half the capsule's straight part (m); the capsule is `2·(half_height + radius)` tall.
    pub half_height: f32,
    /// Gap kept between the capsule and every surface (m); between two surfaces facing each other
    /// with no room for that much each, an eighth of it.
    pub skin: f32,
    /// Steepest walkable slope, as the cosine of its angle from horizontal ground.
    pub max_slope_cos: f32,
    /// Highest step the character climbs without jumping (m).
    pub step_height: f32,
    /// How far it follows the ground down while walking (m), so it does not bounce down slopes and
    /// stairs.
    pub snap_distance: f32,
    /// Most sweeps for each part of a move (riding, walking, stepping, rising or falling); each
    /// surface met costs one.
    pub max_iterations: u32,
    /// The character's mass when it pushes dynamic bodies (kg).
    pub mass: f32,
    pub filter: QueryFilter,
}

impl Default for CharacterSettings {
    fn default() -> Self {
        Self {
            radius: 0.3,
            half_height: 0.6,
            skin: 0.02,
            // cos 50°
            max_slope_cos: 0.642_787_6,
            step_height: 0.3,
            snap_distance: 0.3,
            max_iterations: 4,
            mass: 80.0,
            filter: QueryFilter::default(),
        }
    }
}

/// What one move did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MoveResult {
    /// How far the capsule actually moved (riding included).
    pub moved: Vec3,
    pub grounded: bool,
    pub ground_normal: Vec3,
    /// It climbed a step this move.
    pub stepped: bool,
    /// Colliders it touched while sliding, with their surface normals (facing the character).
    pub hits: Vec<(ColliderId, Vec3)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Character {
    /// The capsule's centre.
    pub position: Vec3,
    pub settings: CharacterSettings,
    pub grounded: bool,
    pub ground_normal: Vec3,
    pub ground_body: Option<BodyId>,
    /// `ground_body` as it was when the last move ended; the next move follows its change.
    footing: Option<Footing>,
}

/// A body the character stands on, as it was when a move ended.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Footing {
    pose: Iso,
    teleport_revision: u64,
}

impl Footing {
    fn of(world: &PhysicsWorld, id: BodyId) -> Option<Footing> {
        let body = world.body(id)?;
        Some(Footing {
            pose: body.iso(),
            teleport_revision: body.teleport_revision,
        })
    }
}

/// A surface a sweep came within the skin of.
#[derive(Clone, Copy, Debug)]
struct Touch {
    /// Fraction of the motion travelled before it came within the skin.
    fraction: f32,
    /// The surface's point closest to the capsule then.
    point: Vec3,
    /// Unit direction from that point towards the capsule: the surface's normal on a face,
    /// rounded over an edge or corner.
    normal: Vec3,
    collider: ColliderId,
    body: BodyId,
}

/// How a slide treats the surfaces it meets.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    /// Walking: a slope too steep to walk is a wall, never climbed.
    Walk,
    /// Moving down: ground that holds the character up ends the motion (it lands).
    Fall,
    /// Rising or riding: slides along whatever it meets.
    Free,
}

/// What a part of a move touched: hits for the result, and dynamic bodies to push (the body, the
/// point and the direction). Kept, or dropped with a step attempt that is undone.
#[derive(Default)]
struct Log {
    hits: Vec<(ColliderId, Vec3)>,
    pushes: Vec<(BodyId, Vec3, Vec3)>,
}

impl Log {
    fn touched(&mut self, world: &PhysicsWorld, t: &Touch) {
        self.hits.push((t.collider, t.normal));
        if world.body_kind(t.body) == Some(BodyKind::Dynamic) {
            self.pushes.push((t.body, t.point, -t.normal));
        }
    }

    fn keep(&mut self, other: Log) {
        self.hits.extend(other.hits);
        self.pushes.extend(other.pushes);
    }
}

impl Character {
    pub fn new(position: Vec3, settings: CharacterSettings) -> Self {
        Self {
            position,
            settings,
            grounded: false,
            ground_normal: Vec3::Y,
            ground_body: None,
            footing: None,
        }
    }

    /// The capsule.
    pub fn shape(&self) -> Shape {
        Shape::Capsule {
            half_height: self.settings.half_height,
            radius: self.settings.radius,
        }
    }

    /// Where the capsule's bottom is (its lowest point).
    pub fn feet(&self) -> Vec3 {
        self.position - Vec3::Y * (self.settings.half_height + self.settings.radius)
    }

    fn walkable(&self, n: Vec3) -> bool {
        n.y >= self.settings.max_slope_cos
    }

    /// How far point `p` is from the capsule's surface (negative inside).
    fn clearance(&self, p: Vec3) -> f32 {
        let h = self.settings.half_height.max(0.0);
        let y = p.y.max(self.position.y - h).min(self.position.y + h);
        (p - Vec3::new(self.position.x, y, self.position.z)).length() - self.settings.radius
    }

    fn skin(&self) -> f32 {
        self.settings.skin.max(0.0)
    }

    /// Moves the capsule by `displacement` (world metres for this step), sliding along what it
    /// hits, climbing steps and snapping to the ground, after following whatever it stands on;
    /// then pushes the dynamic bodies it walked into. `dt` is the step's length (for pushing).
    pub fn move_and_slide(
        &mut self,
        world: &mut PhysicsWorld,
        displacement: Vec3,
        dt: f32,
    ) -> MoveResult {
        let mut result = MoveResult::default();
        if !displacement.is_finite() || !self.position.is_finite() {
            return result;
        }
        let start = self.position;
        let was_grounded = self.grounded;
        // The body it stands on carries it and is not pushed by it: the game's push down to stay
        // on the ground is not momentum, and pushing every step would keep the body from sleeping.
        let standing_on = self.ground_body.filter(|_| was_grounded);
        let mut log = Log::default();

        // Follow what it stands on (before anything else, so that body's motion is not mistaken
        // for an overlap), then get out of anything else that moved into it.
        let riding = self.ride(world, &mut log);
        self.depenetrate(world);

        // Walk horizontally (climbing a step when a wall stops it), then move vertically.
        let horizontal = Vec3::new(displacement.x, 0.0, displacement.z);
        let vertical = Vec3::new(0.0, displacement.y, 0.0);
        let before = self.position;
        let blocked = self.slide(world, horizontal, None, Mode::Walk, &mut log);
        if blocked && was_grounded {
            let walked = self.position - before;
            let left = horizontal - Vec3::new(walked.x, 0.0, walked.z);
            if self.try_step(world, left, &mut log) {
                result.stepped = true;
            }
        }
        let mode = if vertical.y < 0.0 {
            Mode::Fall
        } else {
            Mode::Free
        };
        self.slide(world, vertical, None, mode, &mut log);

        // Ground: snap down when walking along or down (the game's own motion decides, not the
        // platform's), and find out what is underfoot.
        self.find_ground(world, was_grounded && displacement.y <= 0.0);
        self.footing = match self.ground_body {
            Some(b) if self.grounded && world.body_kind(b) != Some(BodyKind::Static) => {
                Footing::of(world, b)
            }
            _ => None,
        };

        if dt > 0.0 && dt.is_finite() {
            let velocity = displacement * (1.0 / dt) + riding;
            self.push(world, &log.pushes, velocity, standing_on);
        }
        result.hits = log.hits;
        result.grounded = self.grounded;
        result.ground_normal = self.ground_normal;
        result.moved = self.position - start;
        result
    }

    /// Carries the character by the change of pose of the body it stood on since the last move
    /// (exact, so a turning platform carries it round, not outward), swept past that body so a
    /// wall still stops it. An explicit teleport breaks the footing even if a platform jumps
    /// less than a skin or was already moving in that direction.
    fn ride(&mut self, world: &PhysicsWorld, log: &mut Log) -> Vec3 {
        let (true, Some(id), Some(then)) = (self.grounded, self.ground_body, self.footing) else {
            return Vec3::ZERO;
        };
        let Some(body) = world.body(id) else {
            return Vec3::ZERO;
        };
        if body.teleport_revision != then.teleport_revision {
            return Vec3::ZERO;
        }
        // Where the body's change of pose takes the character's centre. A rigid motion keeps its
        // distance from the body's origin; turning by quaternions that are unit only to rounding
        // does not quite, which would spiral it in or out over minutes on a turntable.
        let from_origin = self.position - then.pose.position;
        let turned = (body.rotation * then.pose.rotation.conjugate()).rotate(from_origin);
        let turned = turned.normalize_or_zero() * from_origin.length();
        let carried = body.position + turned - self.position;
        if !carried.is_finite() {
            return Vec3::ZERO;
        }
        let velocity =
            body.linear_velocity + body.angular_velocity.cross(self.position - body.center);
        self.slide(world, carried, Some(id), Mode::Free, log);
        if velocity.is_finite() {
            velocity
        } else {
            Vec3::ZERO
        }
    }

    /// Sweeps the capsule from `from` along `motion` and returns every surface it would come
    /// within `gap` of (the skin, unless squeezing between two surfaces), nearest first (then by
    /// collider index): bodies `except` and surfaces it already touches but moves along or away
    /// from are left out, as are those it is already deep inside (the narrow phase gets it out of
    /// those, see [`Self::depenetrate`]).
    fn sweep(
        &self,
        world: &PhysicsWorld,
        from: Vec3,
        motion: Vec3,
        except: Option<BodyId>,
        gap: f32,
    ) -> Vec<Touch> {
        let mut found: Vec<Touch> = Vec::new();
        let (dir, len) = motion.normalize_and_length();
        if len == 0.0 || !motion.is_finite() {
            return found;
        }
        let skin = self.skin();
        let gap = gap.max(0.0).min(skin);
        // How far inside the gap a motion must reach to be stopped (see `test`): past the casts'
        // tolerance band of 0.1 mm (more against very large shapes, whose grazes then stop it).
        let graze = (gap - GRAZE).max(0.5 * gap);
        let radius = self.settings.radius.max(0.0);
        let shape = self.shape();
        let iso = Iso::from_position(from);
        let Some(core) = shape.as_convex() else {
            return found;
        };
        let moving = Convex::new(core, iso);
        // Grown past the skin by the casts' own tolerance.
        let bounds = shape.aabb(&iso).sweep(motion).expand(1.25 * skin + 1e-3);
        let mut candidates = SmallBuf::<u32, 64>::new(0);
        for tree in [&world.static_tree, &world.moving_tree] {
            tree.query(&bounds, &mut |_, c| {
                let skip = except.is_some_and(|b| b.index == world.colliders[c as usize].body);
                if !skip && world.passes(c, &self.settings.filter) {
                    candidates.push(c);
                }
                true
            });
        }
        candidates.sort_dedup();
        for &c in candidates.as_slice() {
            let col = &world.colliders[c as usize];
            let ciso = world.collider_iso(c);
            let mut test = |other: &Convex, triangle: Option<&TriangleRef>| {
                let Some(mut hit) = gjk::cast(&moving, motion, other, gap) else {
                    return;
                };
                let face = triangle
                    .and_then(|t| self.triangle_face(t, &ciso, &hit, from + motion * hit.fraction));
                if let Some((normal, point)) = face {
                    // The seam of a triangle with a coplanar or concave neighbour is the face
                    // itself, as in the narrow phase: on a flat floor of many triangles, the
                    // seams around the one underfoot come within the skin with normals tilted
                    // against the motion, and each would stop a slide.
                    (hit.normal, hit.point) = (normal, point);
                    if hit.normal.dot(dir) > -ALONG {
                        return;
                    }
                }
                if hit.fraction == 0.0 {
                    // Already within the skin: in the way only if moving into it.
                    if hit.normal.dot(dir) > -ALONG {
                        return;
                    }
                    if hit.started_overlapping && gjk::distance(&moving, other).overlap {
                        return;
                    }
                }
                // In the way only if the motion would take the capsule inside the skin, past the
                // cast's tolerance band: what it only grazes within the band (the ridge between
                // two triangles of a gentle hill, which it passes over a skin above) would stop
                // it for nothing, and several such at once used up a slide's iterations.
                if graze < gap && !closes_in(face.is_some(), &hit, dir, len, radius) {
                    let real = gjk::cast(&moving, motion, other, graze).is_some();
                    if !real {
                        return;
                    }
                }
                found.push(Touch {
                    fraction: hit.fraction,
                    point: hit.point,
                    normal: hit.normal,
                    collider: world.collider_id(c),
                    body: world.body_id(col.body),
                });
            };
            match &col.shape {
                Shape::Mesh(m) => {
                    m.query_aabb(&transform_aabb(&bounds, &ciso.inverse()), &mut |t| {
                        test(
                            &Convex::new(ConvexShape::Triangle(t.vertices), ciso),
                            Some(t),
                        );
                        true
                    })
                }
                Shape::Heightfield(h) => {
                    h.query_aabb(&transform_aabb(&bounds, &ciso.inverse()), &mut |t| {
                        test(
                            &Convex::new(ConvexShape::Triangle(t.vertices), ciso),
                            Some(t),
                        );
                        true
                    })
                }
                other => {
                    if let Some(os) = other.as_convex() {
                        test(&Convex::new(os, ciso), None);
                    }
                }
            }
        }
        // Stable: triangles of one mesh at the same fraction stay in the mesh's order.
        found.sort_by(|a, b| {
            a.fraction
                .total_cmp(&b.fraction)
                .then(a.collider.index.cmp(&b.collider.index))
        });
        found
    }

    /// For a cast `hit` on triangle `t` of a mesh or heightfield at `ciso`, by the capsule centred
    /// at `centre`: when the hit is on the triangle's front and on its face, or on an edge or
    /// vertex the mesh marks inactive, the face's normal and the point of its plane nearest the
    /// capsule. `None` where the cast's own normal stands: an active edge or vertex (a ridge or a
    /// border), the back of the triangle, a triangle with no area.
    fn triangle_face(
        &self,
        t: &TriangleRef,
        ciso: &Iso,
        hit: &gjk::CastHit,
        centre: Vec3,
    ) -> Option<(Vec3, Vec3)> {
        let face = ciso.rotation.rotate(t.normal());
        if face == Vec3::ZERO || hit.normal.dot(face) <= 0.0 {
            return None;
        }
        if hit.normal != face && seam_is_active(t, ciso.inverse_point(hit.point)) {
            return None;
        }
        // The capsule's point deepest towards the plane (its bottom over a floor, its top under a
        // ceiling), dropped onto the plane: what the face itself would have touched.
        let h = self.settings.half_height.max(0.0);
        let lowest = if face.y > 0.0 {
            centre - Vec3::Y * h
        } else if face.y < 0.0 {
            centre + Vec3::Y * h
        } else {
            centre
        };
        Some((face, lowest - face * face.dot(lowest - hit.point)))
    }

    /// The first touch of `touches` (from a sweep straight down, or along a slide, of length
    /// `len`) that holds the character up, with its ground normal, among those met first (within a
    /// millimetre). When none does on its own, those met first may together: wedged in a V of two
    /// slopes too steep to stand on (a trench, a cleft between rocks), or between such a slope and
    /// a wall, the capsule rests on both, as it would with no friction at all. They are ground
    /// when the sum of their distinct normals (walls and anything facing up; each surface once,
    /// however many triangles of it touch) is walkable, and the most upward of them stands for
    /// that ground.
    ///
    /// The edges of a hole or slot the capsule fits through are no ground: when every touch that
    /// would hold it up is an edge beyond its radius by more than the squeezed gap (which only its
    /// skin rests on, as on a step's nosing it has not yet reached) and two of them face each other
    /// across it, it squeezes through between them instead (see [`Self::slide`]).
    fn first_support(
        &self,
        world: &PhysicsWorld,
        touches: &[Touch],
        len: f32,
    ) -> Option<(Touch, Vec3)> {
        let first = touches.first()?.fraction;
        let met = || {
            touches
                .iter()
                .take_while(move |t| (t.fraction - first) * len <= 1e-3)
        };
        let supports: Vec<(Touch, Vec3)> = met()
            .filter_map(|t| self.support(world, t).map(|n| (*t, n)))
            .collect();
        if let Some(&found) = supports.first() {
            // Beyond the capsule's radius by more than the squeezed gap: it fits past.
            let past = self.settings.radius + SQUEEZE * self.skin() + GRAZE;
            let beyond = |t: &Touch| {
                let off = t.point - self.position;
                Vec3::new(off.x, 0.0, off.z).length() > past
            };
            let rim = supports.iter().all(|(t, _)| beyond(t))
                && supports.iter().any(|(a, _)| {
                    supports
                        .iter()
                        .any(|(b, _)| a.normal.dot(b.normal) < -OPPOSED)
                });
            return (!rim).then_some(found);
        }
        let mut normals: Vec<Vec3> = Vec::new();
        let mut upmost: Option<Touch> = None;
        for t in met().filter(|t| t.normal.y >= 0.0) {
            add_surface(&mut normals, t.normal);
            if upmost.is_none_or(|u| t.normal.y > u.normal.y) {
                upmost = Some(*t);
            }
        }
        let normal = self.held_by(&normals)?;
        Some((upmost?, normal))
    }

    /// Ground the capsule is wedged on that no downward sweep shows: a slope too steep to stand on
    /// that it touches (among `touches`, from a sweep down from where it is) and what stops it
    /// sliding down that slope, also touching it (a wall, an overhang). Held up as in
    /// [`Self::first_support`]; the slope stands for the ground.
    fn wedged(&self, world: &PhysicsWorld, touches: &[Touch]) -> Option<(Touch, Vec3)> {
        let mut tried: Vec<Vec3> = Vec::new();
        for slope in touches.iter().take_while(|t| t.fraction == 0.0) {
            let n = slope.normal;
            // A face too steep to stand on; not an edge with ground beyond it (a step's nosing,
            // the rim of a hole), which `first_support` has already judged.
            if n.y <= 0.0
                || self.walkable(n)
                || self.support(world, slope).is_some()
                || !add_surface(&mut tried, n)
            {
                continue;
            }
            // Straight down, less the part into the slope: the way it would slide.
            let downhill = (n * n.y - Vec3::Y).normalize_or_zero();
            let skin = self.skin();
            let probe = self.sweep(world, self.position, downhill * skin.max(1e-3), None, skin);
            let mut normals = vec![n];
            for t in probe.iter().take_while(|t| t.fraction == 0.0) {
                add_surface(&mut normals, t.normal);
            }
            if let Some(normal) = self.held_by(&normals) {
                return Some((*slope, normal));
            }
        }
        None
    }

    /// The ground several surfaces make together (`normals`, one each), none of which holds the
    /// character up on its own: the direction of their sum, when it is walkable.
    fn held_by(&self, normals: &[Vec3]) -> Option<Vec3> {
        let sum = normals.iter().fold(Vec3::ZERO, |sum, &n| sum + n);
        let normal = sum.normalize_or_zero();
        (normals.len() > 1 && self.walkable(normal)).then_some(normal)
    }

    /// The normal of the ground a touch holds the character up on: the touch's own when it is
    /// walkable; for an edge or corner under the capsule's rounded bottom (a step's nosing, a
    /// ledge), the normal of the walkable top beyond it. `None` for walls, slopes too steep to
    /// walk and the capsule's side.
    fn support(&self, world: &PhysicsWorld, touch: &Touch) -> Option<Vec3> {
        let n = touch.normal;
        if self.walkable(n) {
            return Some(n);
        }
        if n.y < LOWEST_SUPPORT {
            return None;
        }
        // Look straight down just past the contact, away from the capsule: on a step that is its
        // top; on a steep face it is the face itself (or inside it, which a ray reports at 0).
        let skin = self.skin().max(1e-3);
        let inward = -Vec3::new(n.x, 0.0, n.z).normalize_or_zero();
        let from = touch.point + inward * (0.5 * skin) + Vec3::Y * skin;
        let hit = world.raycast(from, -Vec3::Y, 2.0 * skin, &self.settings.filter)?;
        (hit.distance > 0.0 && self.walkable(hit.normal)).then_some(hit.normal)
    }

    /// Moves along `motion`, sliding along what it meets (see [`Mode`]), and logs what it touched.
    /// Returns whether a surface too steep to walk stopped part of a walk.
    ///
    /// Stopped in a crease between two surfaces facing each other (a doorway's jambs, a
    /// corridor's mouth, the walls of a hole) while still clear of both, it is only their skins
    /// that are in the way: the skin gives way, down to [`SQUEEZE`] of it, and what is left of the
    /// motion goes on straight from there, so the capsule passes through any opening it fits
    /// through with a few millimetres to spare. Caught between surfaces facing each other again
    /// after that, they close in on it (a wedge): the squeeze is undone, and it stays where its
    /// skin stopped it. Falling between two surfaces that hold it up together (a V) it lands
    /// instead.
    fn slide(
        &mut self,
        world: &PhysicsWorld,
        motion: Vec3,
        except: Option<BodyId>,
        mode: Mode,
        log: &mut Log,
    ) -> bool {
        let skin = self.skin();
        let start = self.position;
        let mut gap = skin;
        // Where it squeezed from, and how much it had logged then.
        let mut squeezed: Option<(Vec3, usize, usize)> = None;
        let mut remaining = motion;
        let mut previous: Option<Touch> = None;
        let mut blocked = false;
        for _ in 0..self.settings.max_iterations {
            let len = remaining.length();
            if !len.is_finite() || len <= 1e-6 {
                break;
            }
            let touches = self.sweep(world, self.position, remaining, except, gap);
            let Some(&touch) = touches.first() else {
                self.position += remaining;
                break;
            };
            self.position += remaining * touch.fraction;
            if mode == Mode::Fall
                && let Some((ground, _)) = self.first_support(world, &touches, len)
            {
                // Landed: ground stops a fall, it does not slide down it.
                log.touched(world, &ground);
                break;
            }
            log.touched(world, &touch);
            let mut left = remaining * (1.0 - touch.fraction);
            let mut n = touch.normal;
            if mode == Mode::Walk && !self.walkable(n) {
                blocked = true;
                if n.y > 0.0 && (left - n * left.dot(n)).y > 0.0 {
                    // Sliding along it would climb it: treat it as a wall instead.
                    n = Vec3::new(n.x, 0.0, n.z).normalize_or_zero();
                    if n == Vec3::ZERO {
                        break;
                    }
                }
            }
            // What is left, minus the part into the surface.
            left -= n * left.dot(n).min(0.0);
            if let Some(p) = previous
                && left.dot(p.normal) < 0.0
            {
                if mode == Mode::Fall
                    && self.held_by(&[p.normal, n]).is_some()
                    && self.support(world, &p).is_none()
                    && self.support(world, &touch).is_none()
                {
                    // Landed in a V, or against a wall: the two hold it up together (see
                    // `first_support`). (Edges with ground beyond them, such as the rim of a hole
                    // it fits through, it squeezes past instead.)
                    break;
                }
                let opposed = p.normal.dot(n) < -OPPOSED;
                if opposed && let Some((at, hits, pushes)) = squeezed {
                    // Caught between surfaces facing each other again, nearer: they close in on
                    // it (a wedge), and squeezing only took it deeper. Back to its skin's stop.
                    self.position = at;
                    log.hits.truncate(hits);
                    log.pushes.truncate(pushes);
                    break;
                }
                // Farther than the squeezed gap, past the casts' tolerance band.
                let clear = SQUEEZE * skin + GRAZE;
                let (from_p, from_n) = (self.clearance(p.point), self.clearance(touch.point));
                let at = self.position;
                if gap == skin
                    && opposed
                    && from_p > clear
                    && from_n > clear
                    && self.squeeze(world, (p.normal, from_p), (n, from_n), except)
                {
                    // Squeezing between them: the rest of the motion straight on, nearer.
                    squeezed = Some((at, log.hits.len(), log.pushes.len()));
                    gap = SQUEEZE * skin;
                    let (dir, total) = motion.normalize_and_length();
                    remaining = dir * (total - (self.position - start).dot(dir)).max(0.0);
                    previous = None;
                    continue;
                }
                // In a crease between two surfaces: move along the crease only.
                let crease = p.normal.cross(n).normalize_or_zero();
                left = crease * left.dot(crease);
            }
            previous = Some(Touch { normal: n, ..touch });
            remaining = left;
        }
        blocked
    }

    /// Gets ready to squeeze between two surfaces facing each other (their normals and how far
    /// the capsule is from each), both farther than the squeezed gap: moves to the middle between
    /// them, where the opening has the most room each side, and looks along the way that closes
    /// on both alike (into a doorway, down a hole, into a wedge's corner) as far as two flat walls
    /// like them would take to pinch it to the squeezed gap. Pinched there by either, they close
    /// in on it (a wedge, not an opening): it goes back where it was, and `false`.
    fn squeeze(
        &mut self,
        world: &PhysicsWorld,
        (a, from_a): (Vec3, f32),
        (b, from_b): (Vec3, f32),
        except: Option<BodyId>,
    ) -> bool {
        let skin = self.skin();
        let gap = SQUEEZE * skin;
        let saved = self.position;
        let across = a - b;
        let centre = across * ((from_b - from_a) / across.length_squared());
        if centre.is_finite() {
            let fraction = self
                .sweep(world, self.position, centre, except, gap)
                .first()
                .map_or(1.0, |t| t.fraction);
            self.position += centre * fraction;
        }
        // Walls facing each other square on (a corridor) never close in.
        let (inward, width) = (-(a + b)).normalize_and_length();
        if width < 1e-3 {
            return true;
        }
        // How fast moving `inward` closes on each: half the width of `a + b`.
        let ahead = (1.25 * (from_a.max(from_b) - gap) / (0.5 * width)).min(0.5);
        let same =
            |t: &Touch| t.normal.dot(a) >= SAME_SURFACE_COS || t.normal.dot(b) >= SAME_SURFACE_COS;
        let pinched = self
            .sweep(world, self.position, inward * ahead, except, gap)
            .first()
            .is_some_and(same);
        if pinched {
            self.position = saved;
        }
        !pinched
    }

    /// Up by the step height, across what is left, then down onto ground (a top, or the nosing of
    /// one under the capsule's rounded bottom) at most the step height above the ground it stood
    /// on; undone when it lands on nothing of the kind or gets less than half way across.
    fn try_step(&mut self, world: &PhysicsWorld, across: Vec3, log: &mut Log) -> bool {
        let height = self.settings.step_height;
        let (dir, len) = across.normalize_and_length();
        if !height.is_finite() || height <= 0.0 || len < 1e-5 {
            return false;
        }
        let saved = self.position;
        // The highest ground it may land on: a step's height above the ground it stands on.
        let highest = self.feet().y - self.skin() + height + 1e-3;
        let rise = match self
            .sweep(world, saved, Vec3::Y * height, None, self.skin())
            .first()
        {
            Some(t) => t.fraction * height,
            None => height,
        };
        if rise.is_nan() || rise <= 1e-4 {
            return false;
        }
        self.position += Vec3::Y * rise;
        let mut scratch = Log::default();
        self.slide(world, across, None, Mode::Walk, &mut scratch);
        let progress = (self.position - saved).dot(dir);
        let drop = rise + self.skin();
        let touches = self.sweep(world, self.position, Vec3::Y * -drop, None, self.skin());
        if let Some((ground, _)) = self.first_support(world, &touches, drop)
            && ground.point.y <= highest
            && progress > 0.5 * len
        {
            self.position -= Vec3::Y * (ground.fraction * drop);
            scratch.touched(world, &ground);
            log.keep(scratch);
            return true;
        }
        self.position = saved;
        false
    }

    /// Finds the ground just below (within the skin past the capsule's rest distance, or the snap
    /// distance when `snap`), and when `snap`, puts the capsule a skin above it.
    fn find_ground(&mut self, world: &PhysicsWorld, snap: bool) {
        let skin = self.skin();
        let reach = if snap {
            self.settings.snap_distance.max(skin)
        } else {
            skin
        };
        let touches = if reach > 0.0 {
            self.sweep(world, self.position, Vec3::Y * -reach, None, skin)
        } else {
            Vec::new()
        };
        let ground = self
            .first_support(world, &touches, reach)
            .or_else(|| self.wedged(world, &touches));
        match ground {
            Some((ground, normal)) => {
                if snap {
                    self.position -= Vec3::Y * (ground.fraction * reach);
                    // Already inside the skin (placed there, or rounding): back up to it. Not
                    // off surfaces that hold it up only together (a V): straight up restores one's
                    // skin only by leaving the other's.
                    let short = skin - self.clearance(ground.point);
                    if ground.fraction == 0.0
                        && short > 1e-5
                        && ground.normal.y > 0.0
                        && self.support(world, &ground).is_some()
                    {
                        let lift = (short / ground.normal.y).min(2.0 * skin);
                        let up = self.sweep(world, self.position, Vec3::Y * lift, None, skin);
                        self.position += Vec3::Y * (lift * up.first().map_or(1.0, |t| t.fraction));
                    }
                }
                self.grounded = true;
                self.ground_normal = normal;
                self.ground_body = Some(ground.body);
            }
            None => {
                self.grounded = false;
                self.ground_normal = Vec3::Y;
                self.ground_body = None;
            }
        }
    }

    /// Gets the capsule out of anything closer than [`CROWDED`] of the skin (spawned inside a box,
    /// a body moved into it), by the contacts the narrow phase reports, until every surface
    /// within the skin is a skin away. Where two face each other closer than two skins apart (a
    /// corridor or doorway barely wider than the capsule, doors closing on it) there is no room
    /// for that: the capsule goes to the middle between them, however deep in it is; but between
    /// level geometry (a static body) and a body that moves (a platform rising into a ceiling, a
    /// door closing against a wall), it keeps off the level geometry and the moving body takes the
    /// shortfall, as it would pass through the capsule anyway. No surface ends closer than it
    /// started or than its target; when the targets cannot all be met, the capsule is only taken
    /// out of what it is inside, and when not even that, left alone.
    fn depenetrate(&mut self, world: &PhysicsWorld) {
        let skin = self.skin();
        let shape = self.shape();
        let mut planes = SmallBuf::<(Vec3, f32), 16>::new((Vec3::ZERO, 0.0));
        let mut fixed = SmallBuf::<bool, 16>::new(false);
        let mut targets = SmallBuf::<f32, 16>::new(0.0);
        let mut inside = SmallBuf::<f32, 16>::new(0.0);
        for _ in 0..4 {
            let iso = Iso::from_position(self.position);
            // (normal from the capsule to the surface, separation), and whether it is static.
            planes.clear();
            fixed.clear();
            world.contacts_with(&shape, &iso, skin, &self.settings.filter, &mut |c, m| {
                planes.push((m.normal, m.min_separation()));
                let body = world.collider_body(c);
                fixed.push(body.and_then(|b| world.body_kind(b)) == Some(BodyKind::Static));
            });
            let ps = planes.as_slice();
            let fs = fixed.as_slice();
            if !ps.iter().any(|&(_, s)| s < CROWDED * skin) {
                break;
            }
            // Where each surface should end: a skin away, or its share of the room between it
            // and one facing it closer than that.
            targets.clear();
            for (&(n, s), &level) in ps.iter().zip(fs) {
                let t = ps
                    .iter()
                    .zip(fs)
                    .filter(|&(&(m, _), _)| n.dot(m) < -OPPOSED)
                    .fold(skin, |t, (&(_, r), &other)| {
                        t.min(match (level, other) {
                            (true, false) => skin,
                            (false, true) => s + r - skin,
                            _ => 0.5 * (s + r),
                        })
                    });
                targets.push(t);
            }
            let ts = targets.as_slice();
            let push = least_push(ps, ts).or_else(|| {
                // Only out of what it is inside, coming no closer to anything else.
                inside.clear();
                for (&(_, s), &t) in ps.iter().zip(ts) {
                    inside.push(t.min(s.max(0.0)));
                }
                least_push(ps, inside.as_slice())
            });
            match push {
                Some(push) if push.length_squared() >= 1e-12 => self.position += push,
                _ => break,
            }
        }
    }

    /// Gives each dynamic body it ran into this move one push, along the hit that met it most
    /// head-on: an inelastic collision between the character (`mass` at `velocity`) and the
    /// body's point it touched, so that point ends no faster than the character moves along the
    /// push, and one already moving away gets only what closes the gap. The body it stands on is
    /// not pushed.
    ///
    /// The collision is with the body's effective mass at that point along the push (its mass,
    /// and its inertia about the point unless its rotation is locked): caught off its centre of
    /// mass a body turns as well as moves, so the same push moves the touched point faster than
    /// it would the centre, and a push measured at the centre flung the edge of a box it caught,
    /// or the far side of a door, faster than the walk. A joint holding the body only makes the
    /// point slower. The centre of mass, too, never ends faster than the character: a body
    /// already turning so that the point lags its centre gets only what brings the centre level.
    fn push(
        &self,
        world: &mut PhysicsWorld,
        pushes: &[(BodyId, Vec3, Vec3)],
        velocity: Vec3,
        standing_on: Option<BodyId>,
    ) {
        let m = self.settings.mass;
        if !(m.is_finite() && m > 0.0 && velocity.is_finite()) {
            return;
        }
        let mut done: Vec<BodyId> = Vec::new();
        for (i, &(body, _, _)) in pushes.iter().enumerate() {
            if done.contains(&body) || standing_on == Some(body) {
                continue;
            }
            done.push(body);
            let (Some(v), Some(mb)) = (world.linear_velocity(body), world.mass(body)) else {
                continue;
            };
            if !(mb.is_finite() && mb > 0.0) {
                continue;
            }
            let mut best: Option<(f32, Vec3, Vec3, Vec3)> = None;
            for &(b, point, direction) in &pushes[i..] {
                let Some(at) = world.point_velocity(b, point).filter(|_| b == body) else {
                    continue;
                };
                let approach = (velocity - at).dot(direction);
                if best.is_none_or(|(a, ..)| approach > a) {
                    best = Some((approach, point, direction, at));
                }
            }
            let Some((approach, point, direction, before)) = best else {
                continue;
            };
            // The push that leaves the centre of mass no faster than the character: the most
            // it gets, and what a body touched at its centre gets.
            let at_centre = (velocity - v).dot(direction) * m * mb / (m + mb);
            if !(approach > 0.0 && at_centre > 0.0) {
                continue;
            }
            // How fast an impulse there moves the point that way (the effective inverse mass),
            // by the world's own response to a first push, locked axes and all: the push at the
            // centre (at least a millimetre a second's worth, to measure it), then the difference.
            let probe = at_centre.max(1e-3 * mb);
            world.apply_impulse_at(body, direction * probe, point);
            let Some(after) = world.point_velocity(body, point) else {
                continue;
            };
            let inverse_mass = ((after - before).dot(direction) / probe).max(0.0);
            let impulse = (approach / (1.0 / m + inverse_mass)).min(at_centre);
            if impulse.is_finite() {
                world.apply_impulse_at(body, direction * (impulse - probe), point);
            }
        }
    }
}

/// The push that leaves each surface of `planes` (the normal from the capsule to it, and its
/// separation) at least its target away, found by projecting onto each in turn; `None` unless it
/// leaves every surface no closer than it was or than its target, whichever is closer.
fn least_push(planes: &[(Vec3, f32)], targets: &[f32]) -> Option<Vec3> {
    let mut push = Vec3::ZERO;
    for _ in 0..16 {
        for (&(n, s), &t) in planes.iter().zip(targets) {
            let short = t - (s - push.dot(n));
            if short > 0.0 {
                push -= n * short;
            }
        }
    }
    let kept = planes
        .iter()
        .zip(targets)
        .all(|(&(n, s), &t)| s - push.dot(n) >= s.min(t) - 1e-4);
    (push.is_finite() && kept).then_some(push)
}

/// Whether the rest of a sweep (along unit `dir`, `len` long) after `hit` surely takes the
/// capsule (rounded by `radius`) more than twice [`GRAZE`] nearer the surface hit, so that it is
/// no graze and needs no second cast. Head on, the gap to a face closes as fast as the motion
/// heads into it, `k` (the cosine with the normal), along the whole way; to an edge or corner
/// it closes by at least `k s - s² / 2r` over `s` (the most curved case: a corner against the
/// capsule's rounding).
fn closes_in(face: bool, hit: &gjk::CastHit, dir: Vec3, len: f32, radius: f32) -> bool {
    let k = -hit.normal.dot(dir);
    let rest = len * (1.0 - hit.fraction);
    let closing = if face {
        k * rest
    } else if radius > 0.0 {
        let s = rest.min(k * radius);
        k * s - s * s / (2.0 * radius)
    } else {
        0.0
    };
    closing > 2.0 * GRAZE
}

/// Adds `normal` to `normals` unless it is the same surface as one there (within about 1°, so the
/// triangles of one slope count once); whether it was added.
fn add_surface(normals: &mut Vec<Vec3>, normal: Vec3) -> bool {
    let new = !normals.iter().any(|n| n.dot(normal) >= SAME_SURFACE_COS);
    if new {
        normals.push(normal);
    }
    new
}

/// Whether the feature of triangle `t` nearest `p` (a point on it, in its frame) may give a contact
/// a normal of its own (see the mesh module's active edges): the edge `p` lies on, or, within
/// rounding of a corner, that vertex, which may when either of its edges may (as in the narrow
/// phase).
fn seam_is_active(t: &TriangleRef, p: Vec3) -> bool {
    let v = t.vertices;
    // Rounding of the casts' closest point at these coordinates, well under any triangle's size.
    let tolerance = 1e-5 + 1e-6 * p.abs().max_element();
    let mut nearest = (f32::INFINITY, 0usize, 0.0f32, 0.0f32);
    for k in 0..3 {
        let (a, b) = (v[k], v[(k + 1) % 3]);
        let (along, len) = (b - a).normalize_and_length();
        let s = (p - a).dot(along).max(0.0).min(len);
        let d = (a + along * s - p).length_squared();
        if d < nearest.0 {
            nearest = (d, k, s, len);
        }
    }
    let (_, k, s, len) = nearest;
    let bit = |k: usize| t.active_edges & (1 << k) != 0;
    if s <= tolerance {
        // Vertex k, shared with the edge before it.
        bit(k) || bit((k + 2) % 3)
    } else if s >= len - tolerance {
        // Vertex k + 1, shared with the edge after it.
        bit(k) || bit((k + 1) % 3)
    } else {
        bit(k)
    }
}
