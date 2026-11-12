//! The solver behind every commit. It grounds each node a patch placed by a relation, then
//! keeps every node the patch moved to three rules:
//!
//! 1. support: a node stands on the highest node under its centre whose top is not above it, or
//!    on the floor (`stage.floorY`, default 0), and inside that node's footprint;
//! 2. stage: a floor node's footprint stays inside `stage.bounds` (`{"min": [x, z], "max": [x, z]}`);
//! 3. clearance: floor nodes' footprints stay at least [`MIN_GAP`] apart.
//!
//! With [`Mode::Project`] a broken rule is repaired by the smallest move that satisfies it and
//! reported; with [`Mode::Reject`] the patch is refused and the graph keeps its state. A node that
//! lost what it stood on settles by gravity without that being a broken rule. Footprints are
//! rotated rectangles (`aabb` bounds) or discs (`cylinder`), compared by their separating axes.
//!
//! Afterwards every relation describes where the nodes stand: a relation that still holds is kept
//! (its clearance updated), one that no longer does is derived again, so the same layout is the
//! same graph whichever participant solved it. Nodes that face another turn toward it.
use crate::patch::{self, Applied, Patch};
use crate::{Bounds, Edge, Graph, Node, Relation};
use ganbate_core::json::Value;

/// Smallest gap between two floor nodes' footprints, metres.
pub const MIN_GAP: f32 = 0.02;
/// Gap a placing relation leaves when it names none.
pub const DEFAULT_CLEARANCE: f32 = 0.3;
const SUPPORT_TOLERANCE: f32 = 0.03;
/// Positions are kept to the millimetre, so comparisons allow for that rounding.
const EPS: f32 = 1e-3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Project,
    Reject,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Commit {
    pub graph: Graph,
    /// What the solver moved to keep the rules, in plain words (empty when nothing was).
    pub repairs: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
struct Stage {
    floor_y: f32,
    bounds: Option<([f32; 2], [f32; 2])>,
}

impl Stage {
    fn of(graph: &Graph) -> Result<Self, String> {
        let Some(stage) = graph.extra.get("stage") else {
            return Ok(Self {
                floor_y: 0.0,
                bounds: None,
            });
        };
        let floor_y = stage
            .get("floorY")
            .map(|y| crate::number(y, "stage.floorY"))
            .transpose()?;
        let xz = |v: Option<&Value>, what: &str| -> Result<[f32; 2], String> {
            match v.map(Value::array) {
                Some(Ok([x, z])) => Ok([crate::number(x, what)?, crate::number(z, what)?]),
                _ => Err(format!("{what} must be [x, z]")),
            }
        };
        let bounds = match stage.get("bounds") {
            None => None,
            Some(b) => {
                let min = xz(b.get("min"), "stage.bounds.min")?;
                let max = xz(b.get("max"), "stage.bounds.max")?;
                if min[0] >= max[0] || min[1] >= max[1] {
                    return Err("stage.bounds.min must be below stage.bounds.max".into());
                }
                Some((min, max))
            }
        };
        Ok(Self {
            floor_y: floor_y.unwrap_or(0.0),
            bounds,
        })
    }
}

fn dot(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

fn sub(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn scale(a: [f32; 2], k: f32) -> [f32; 2] {
    [a[0] * k, a[1] * k]
}

fn len(a: [f32; 2]) -> f32 {
    a[0].hypot(a[1])
}

fn xz(n: &Node) -> [f32; 2] {
    [n.position[0], n.position[2]]
}

fn half(n: &Node) -> [f32; 2] {
    n.bounds.map_or([0.0, 0.0], |b| b.half_xz())
}

/// Size for choosing what a node relates to: the radius of the circle around its footprint.
fn size(n: &Node) -> f32 {
    n.bounds.map_or(0.0, |b| b.radius())
}

fn top(n: &Node) -> f32 {
    n.position[1] + n.bounds.map_or(0.0, |b| b.top())
}

/// `n`'s own X (its left-to-right, negated) and Z (its back-to-front) axes, in world XZ.
fn axes(n: &Node) -> [[f32; 2]; 2] {
    let (s, c) = n.yaw_deg.to_radians().sin_cos();
    [[c, -s], [s, c]]
}

/// A point given in `n`'s own frame (heading 0 faces +Z), in world XZ.
fn to_world(n: &Node, local: [f32; 2]) -> [f32; 2] {
    let [ax, az] = axes(n);
    [
        n.position[0] + local[0] * ax[0] + local[1] * az[0],
        n.position[2] + local[0] * ax[1] + local[1] * az[1],
    ]
}

fn to_local(n: &Node, world: [f32; 2]) -> [f32; 2] {
    let [ax, az] = axes(n);
    let d = sub(world, xz(n));
    [dot(d, ax), dot(d, az)]
}

/// A node's footprint on the floor plane: a rectangle in its own frame, or a disc.
#[derive(Clone, Copy, Debug)]
struct Foot {
    c: [f32; 2],
    half: [f32; 2],
    axes: [[f32; 2]; 2],
    disc: bool,
}

fn foot_at(n: &Node, c: [f32; 2]) -> Foot {
    Foot {
        c,
        half: half(n),
        axes: axes(n),
        disc: !matches!(n.bounds, Some(Bounds::Box { .. })),
    }
}

fn foot(n: &Node) -> Foot {
    foot_at(n, xz(n))
}

impl Foot {
    /// Half the footprint's extent along the unit vector `u`.
    fn reach(&self, u: [f32; 2]) -> f32 {
        if self.disc {
            self.half[0]
        } else {
            self.half[0] * dot(self.axes[0], u).abs() + self.half[1] * dot(self.axes[1], u).abs()
        }
    }

    fn local(&self, p: [f32; 2]) -> [f32; 2] {
        let d = sub(p, self.c);
        [dot(d, self.axes[0]), dot(d, self.axes[1])]
    }

    /// The point of this rectangle nearest to `p`.
    fn nearest(&self, p: [f32; 2]) -> [f32; 2] {
        let l = self.local(p);
        let (x, z) = (
            l[0].clamp(-self.half[0], self.half[0]),
            l[1].clamp(-self.half[1], self.half[1]),
        );
        [
            self.c[0] + x * self.axes[0][0] + z * self.axes[1][0],
            self.c[1] + x * self.axes[0][1] + z * self.axes[1][1],
        ]
    }

    fn contains(&self, p: [f32; 2]) -> bool {
        let l = self.local(p);
        if self.disc {
            len(l) <= self.half[0]
        } else {
            l[0].abs() <= self.half[0] && l[1].abs() <= self.half[1]
        }
    }
}

/// Gap between a rectangle and a disc of radius `r` at `p`; negative where they overlap.
fn rect_disc_gap(rect: Foot, p: [f32; 2], r: f32) -> f32 {
    let l = rect.local(p);
    let [hx, hz] = rect.half;
    if l[0].abs() < hx && l[1].abs() < hz {
        -(hx - l[0].abs()).min(hz - l[1].abs()) - r
    } else {
        len(sub(p, rect.nearest(p))) - r
    }
}

/// Gap between two footprints; negative where they overlap. Between two rectangles it is the
/// widest separation along their axes, which never exceeds the true distance.
fn gap(a: Foot, b: Foot) -> f32 {
    match (a.disc, b.disc) {
        (true, true) => len(sub(a.c, b.c)) - a.half[0] - b.half[0],
        (false, true) => rect_disc_gap(a, b.c, b.half[0]),
        (true, false) => rect_disc_gap(b, a.c, a.half[0]),
        (false, false) => a
            .axes
            .iter()
            .chain(&b.axes)
            .map(|u| dot(sub(a.c, b.c), *u).abs() - a.reach(*u) - b.reach(*u))
            .fold(f32::MIN, f32::max),
    }
}

/// Moves of `a` that leave it `MIN_GAP` clear of `b`, shortest first: along each rectangle's
/// axes and the line between them, both ways.
fn separations(a: Foot, b: Foot) -> Vec<[f32; 2]> {
    let d = sub(a.c, b.c);
    let mut directions: Vec<[f32; 2]> = Vec::new();
    for f in [a, b] {
        if !f.disc {
            directions.extend(f.axes);
        }
    }
    let toward = match (a.disc, b.disc) {
        (true, true) => d,
        (true, false) => sub(a.c, b.nearest(a.c)),
        (false, true) => sub(a.nearest(b.c), b.c),
        (false, false) => [0.0, 0.0],
    };
    if len(toward) > 1e-6 {
        directions.push(scale(toward, 1.0 / len(toward)));
    } else if a.disc && b.disc {
        directions.push([0.0, 1.0]);
    }
    let mut moves: Vec<(f32, [f32; 2])> = directions
        .into_iter()
        .flat_map(|u| [u, scale(u, -1.0)])
        .map(|u| {
            let need = (a.reach(u) + b.reach(u) + MIN_GAP - dot(d, u)).max(0.0);
            (need, scale(u, need))
        })
        .collect();
    moves.sort_by(|x, y| x.0.total_cmp(&y.0));
    moves.into_iter().map(|(_, m)| m).collect()
}

fn support_of<'g>(graph: &'g Graph, id: &str) -> Option<&'g Node> {
    graph
        .placement(id)
        .filter(|e| e.rel == Relation::OnTopOf)
        .and_then(|e| graph.node(&e.to))
}

/// What `id` would stand on by gravity: the highest node under its centre whose top is not above
/// it (and that does not stand on it); none for the floor.
fn support_below<'g>(graph: &'g Graph, id: &str) -> Option<&'g Node> {
    let node = graph.node(id)?;
    graph
        .nodes
        .iter()
        .filter(|t| t.id != id && t.bounds.is_some() && !graph.rests_on(&t.id, id))
        .filter(|t| top(t) <= node.position[1] + SUPPORT_TOLERANCE && foot(t).contains(xz(node)))
        .max_by(|a, b| top(a).total_cmp(&top(b)))
}

/// Nodes standing on the floor, other than `id` and `except`.
fn floor_nodes<'g>(
    graph: &'g Graph,
    id: &'g str,
    except: &'g str,
) -> impl Iterator<Item = &'g Node> {
    graph
        .nodes
        .iter()
        .filter(move |n| n.id != id && n.id != except && support_of(graph, &n.id).is_none())
}

fn clear(a: Foot, others: &[Foot]) -> bool {
    others.iter().all(|o| gap(a, *o) >= MIN_GAP - EPS)
}

/// Where `a`'s centre must stay for its footprint to keep inside the stage.
fn inside_stage(stage: &Stage, a: Foot) -> Option<[f32; 2]> {
    let (min, max) = stage.bounds?;
    let (rx, rz) = (a.reach([1.0, 0.0]), a.reach([0.0, 1.0]));
    let clamp = |v: f32, lo: f32, hi: f32| {
        if lo > hi {
            (lo + hi) / 2.0
        } else {
            v.clamp(lo, hi)
        }
    };
    let c = [
        clamp(a.c[0], min[0] + rx, max[0] - rx),
        clamp(a.c[1], min[1] + rz, max[1] - rz),
    ];
    (len(sub(c, a.c)) > EPS).then_some(c)
}

/// Offsets 0, +1, -1, +2, -2, ... steps along one axis: where to try next to a crowded spot.
fn spread(step: f32) -> impl Iterator<Item = f32> {
    (0..9).map(move |i| {
        let k = (i + 1) / 2;
        if i % 2 == 1 {
            k as f32 * step
        } else {
            -(k as f32) * step
        }
    })
}

fn round(x: f32, per_unit: f32) -> f32 {
    // Dividing keeps a decimal like 2.7 exact in `f32`.
    let r = (x * per_unit).round() / per_unit;
    if r == 0.0 { 0.0 } else { r }
}

/// Moves `id`, and what stands on it, to `to` (to the millimetre).
fn shift(graph: &mut Graph, id: &str, to: [f32; 3]) {
    let Some(from) = graph.node(id).map(|n| n.position) else {
        return;
    };
    let to = to.map(|v| round(v, 1000.0));
    let delta = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
    patch::carry(graph, id, delta, &mut |_| {});
    if let Some(n) = graph.node_mut(id) {
        n.position = to;
    }
}

/// Stands `id` where its placing relation says, beside what is already there.
/// `pending`: nodes this commit places later, which are not anywhere yet.
fn ground(graph: &mut Graph, id: &str, stage: &Stage, pending: &[String]) -> Result<(), String> {
    let (Some(edge), Some(node)) = (graph.placement(id).cloned(), graph.node(id).cloned()) else {
        return Ok(());
    };
    let target = graph
        .node(&edge.to)
        .cloned()
        .ok_or(format!("no node \"{}\"", edge.to))?;
    let on_top = edge.rel == Relation::OnTopOf;
    let [hx, hz] = half(&target);
    let [tx, tz] = axes(&target);
    let me = foot(&node);
    let (rx, rz) = (me.reach(tx), me.reach(tz));
    let limit = [(hx - rx).max(0.0), (hz - rz).max(0.0)];
    let c = edge.clearance.unwrap_or(DEFAULT_CLEARANCE);
    // Where to stand in the target's frame, and along which of its axes to spread when crowded.
    let (base, along_x) = match edge.rel {
        Relation::OnTopOf => match edge.anchor.as_deref() {
            Some("left") => ([-0.5 * limit[0], 0.0], false),
            Some("right") => ([0.5 * limit[0], 0.0], false),
            Some("front") => ([0.0, 0.5 * limit[1]], true),
            Some("back") => ([0.0, -0.5 * limit[1]], true),
            _ => ([0.0, 0.0], true),
        },
        Relation::LeftOf => ([-(hx + rx + c), 0.0], false),
        Relation::RightOf => ([hx + rx + c, 0.0], false),
        Relation::Behind => ([0.0, -(hz + rz + c)], true),
        _ => ([0.0, hz + rz + c], true),
    };
    let neighbours: Vec<Foot> = if on_top {
        graph
            .nodes
            .iter()
            .filter(|n| n.id != id && !pending.contains(&n.id))
            .filter(|n| support_of(graph, &n.id).is_some_and(|s| s.id == target.id))
            .map(foot)
            .collect()
    } else {
        floor_nodes(graph, id, &target.id)
            .filter(|n| !pending.contains(&n.id))
            .map(foot)
            .collect()
    };
    let step = 2.0 * if along_x { rx } else { rz } + MIN_GAP;
    let fits = |l: [f32; 2]| {
        let a = foot_at(&node, to_world(&target, l));
        let placed = if on_top {
            l[0].abs() <= limit[0] && l[1].abs() <= limit[1]
        } else {
            inside_stage(stage, a).is_none()
        };
        placed && clear(a, &neighbours)
    };
    let local = spread(step)
        .map(|d| {
            if along_x {
                [base[0] + d, base[1]]
            } else {
                [base[0], base[1] + d]
            }
        })
        .find(|l| fits(*l))
        .unwrap_or(base);
    let [x, z] = to_world(&target, local);
    let y = if on_top { top(&target) } else { stage.floor_y };
    shift(graph, id, [x, y, z]);
    Ok(())
}

fn violation(mode: Mode, repairs: &mut Vec<String>, what: String) -> Result<(), String> {
    match mode {
        Mode::Project => {
            repairs.push(what);
            Ok(())
        }
        Mode::Reject => Err(what),
    }
}

/// Keeps `id` supported, inside the stage and clear of the floor nodes around it. A `falling`
/// node lost what it stood on: its support is a consequence, not a broken rule.
fn settle(
    graph: &mut Graph,
    id: &str,
    stage: &Stage,
    mode: Mode,
    falling: bool,
    pending: &[String],
    repairs: &mut Vec<String>,
) -> Result<(), String> {
    let Some(node) = graph.node(id).cloned() else {
        return Ok(());
    };
    let support = support_of(graph, id).cloned();
    let want_y = support.as_ref().map_or(stage.floor_y, top);
    if (node.position[1] - want_y).abs() > SUPPORT_TOLERANCE && !falling {
        let on = support
            .as_ref()
            .map_or("the floor".to_owned(), |s| format!("\"{}\"", s.id));
        violation(
            mode,
            repairs,
            format!(
                "\"{id}\" stood at height {:.2}, set on {on} at {want_y:.2}",
                node.position[1]
            ),
        )?;
    }
    let mut p = xz(&node);
    if let Some(s) = &support {
        let ([hx, hz], me, [sx, sz]) = (half(s), foot(&node), axes(s));
        let (lx, lz) = ((hx - me.reach(sx)).max(0.0), (hz - me.reach(sz)).max(0.0));
        let l = to_local(s, p);
        let c = [l[0].clamp(-lx, lx), l[1].clamp(-lz, lz)];
        if len(sub(c, l)) > EPS {
            if !falling {
                let what = format!("\"{id}\" hung over the edge of \"{}\", moved onto it", s.id);
                violation(mode, repairs, what)?;
            }
            p = to_world(s, c);
        }
    } else {
        let named: Vec<(String, Foot)> = floor_nodes(graph, id, "")
            .filter(|n| !pending.contains(&n.id))
            .map(|n| (n.id.clone(), foot(n)))
            .collect();
        let others: Vec<Foot> = named.iter().map(|(_, f)| *f).collect();
        let settled = |c: [f32; 2]| {
            let a = foot_at(&node, c);
            inside_stage(stage, a).is_none() && clear(a, &others)
        };
        for _ in 0..8 {
            let a = foot_at(&node, p);
            if let Some(c) = inside_stage(stage, a) {
                violation(
                    mode,
                    repairs,
                    format!("\"{id}\" left the stage, moved back inside"),
                )?;
                p = c;
            } else if let Some((name, o)) = named.iter().find(|(_, o)| gap(a, *o) < MIN_GAP - EPS) {
                let what = format!("\"{id}\" overlapped \"{name}\", moved clear of it");
                violation(mode, repairs, what)?;
                // The shortest way out that also keeps inside the stage and clear of the rest.
                let moves = separations(a, *o);
                let to = |m: &[f32; 2]| [p[0] + m[0], p[1] + m[1]];
                p = moves
                    .iter()
                    .map(to)
                    .find(|c| settled(*c))
                    .unwrap_or_else(|| moves.first().map_or(p, to));
            } else {
                break;
            }
        }
        if !settled(p) {
            return Err(format!("no clear place for \"{id}\" there"));
        }
    }
    shift(graph, id, [p[0], want_y, p[1]]);
    Ok(())
}

/// Whether `edge` (a placing relation of the node at `p`) still describes where it stands.
fn holds(graph: &Graph, edge: &Edge, p: [f32; 2], resting_on: Option<&Node>) -> bool {
    let Some(t) = graph.node(&edge.to) else {
        return false;
    };
    let ([hx, hz], l) = (half(t), to_local(t, p));
    match edge.rel {
        Relation::OnTopOf => {
            resting_on.is_some_and(|r| r.id == t.id)
                && match edge.anchor.as_deref() {
                    Some("left") => l[0] < 0.0,
                    Some("right") => l[0] > 0.0,
                    Some("front") => l[1] > 0.0,
                    Some("back") => l[1] < 0.0,
                    _ => true,
                }
        }
        _ if resting_on.is_some() || support_of(graph, &t.id).is_some() => false,
        Relation::LeftOf => l[0] <= -hx,
        Relation::RightOf => l[0] >= hx,
        Relation::Behind => l[1] <= -hz,
        Relation::InFrontOf => l[1] >= hz,
        Relation::Facing => false,
    }
}

/// Makes `id`'s placing relation describe where it stands: kept while it still holds (with its
/// clearance updated), otherwise derived again: on what it rests on, or beside the nearest floor
/// node at least its size; none when there is neither.
fn lift(graph: &mut Graph, id: &str) {
    let Some(node) = graph.node(id).cloned() else {
        return;
    };
    let p = xz(&node);
    // By gravity: settling then stands it on that node's top.
    let resting = support_below(graph, id).cloned();
    let me = foot(&node);
    let kept = graph
        .placement(id)
        .filter(|e| holds(graph, e, p, resting.as_ref()))
        .cloned();
    let edge = if let Some(mut e) = kept {
        if e.rel != Relation::OnTopOf {
            let t = graph.node(&e.to).map(foot);
            e.clearance = t.map(|t| round(gap(me, t).max(0.0), 100.0));
        }
        Some(e)
    } else if let Some(t) = resting {
        let ([hx, hz], l) = (half(&t), to_local(&t, p));
        let (u, w) = (l[0] / hx, l[1] / hz);
        let anchor = if u.abs() < 0.33 && w.abs() < 0.33 {
            "center"
        } else if u.abs() >= w.abs() {
            if u < 0.0 { "left" } else { "right" }
        } else if w > 0.0 {
            "front"
        } else {
            "back"
        };
        Some(Edge {
            from: id.into(),
            rel: Relation::OnTopOf,
            to: t.id,
            anchor: Some(anchor.into()),
            clearance: None,
        })
    } else {
        floor_nodes(graph, id, "")
            .filter(|t| t.bounds.is_some() && size(t) >= size(&node))
            .min_by(|a, b| gap(me, foot(a)).total_cmp(&gap(me, foot(b))))
            .map(|t| {
                let ([hx, hz], l) = (half(t), to_local(t, p));
                let rel = if l[0].abs() - hx >= l[1].abs() - hz {
                    if l[0] < 0.0 {
                        Relation::LeftOf
                    } else {
                        Relation::RightOf
                    }
                } else if l[1] < 0.0 {
                    Relation::Behind
                } else {
                    Relation::InFrontOf
                };
                Edge {
                    from: id.into(),
                    rel,
                    to: t.id.clone(),
                    anchor: None,
                    clearance: Some(round(gap(me, foot(t)).max(0.0), 100.0)),
                }
            })
    };
    graph.edges.retain(|e| !(e.from == id && e.rel.places()));
    graph.edges.extend(edge);
}

/// Turns every node that faces another toward it; returns the nodes it turned.
fn face(graph: &mut Graph) -> Vec<String> {
    let turns: Vec<(String, f32)> = graph
        .edges
        .iter()
        .filter(|e| e.rel == Relation::Facing)
        .filter_map(|e| {
            let (from, to) = (graph.node(&e.from)?, graph.node(&e.to)?);
            let d = sub(xz(to), xz(from));
            let yaw = round(d[0].atan2(d[1]).to_degrees(), 10.0);
            (len(d) > 1e-3 && yaw != from.yaw_deg).then(|| (e.from.clone(), yaw))
        })
        .collect();
    for (id, yaw) in &turns {
        if let Some(n) = graph.node_mut(id) {
            n.yaw_deg = *yaw;
        }
    }
    turns.into_iter().map(|(id, _)| id).collect()
}

/// Grounds the placed nodes, keeps every moved and dropped node to the rules, and makes every
/// relation describe the result.
fn solve(mut g: Graph, applied: &Applied, mode: Mode) -> Result<Commit, String> {
    let stage = Stage::of(&g)?;
    let mut repairs = Vec::new();
    let placed = |id: &&String| applied.placed.contains(id);
    // A node moved by its position, or left without support, stands on what is now under it,
    // not on what it stood on before.
    let free: Vec<&String> = applied
        .moved
        .iter()
        .chain(&applied.dropped)
        .filter(|id| !placed(id))
        .collect();
    for id in &free {
        lift(&mut g, id);
    }
    let pending = &applied.placed;
    for id in applied.moved.iter().filter(|id| !placed(id)) {
        settle(&mut g, id, &stage, mode, false, pending, &mut repairs)?;
    }
    for id in applied
        .dropped
        .iter()
        .filter(|id| !placed(id) && !applied.moved.contains(id))
    {
        settle(&mut g, id, &stage, mode, true, pending, &mut repairs)?;
    }
    // Placed nodes stand by their targets where those ended up, in the patch's order.
    for (i, id) in pending.iter().enumerate() {
        ground(&mut g, id, &stage, &pending[i + 1..])?;
        settle(
            &mut g,
            id,
            &stage,
            mode,
            false,
            &pending[i + 1..],
            &mut repairs,
        )?;
    }
    // A turned box sweeps new ground: keep it clear too.
    for id in face(&mut g) {
        if matches!(g.node(&id).and_then(|n| n.bounds), Some(Bounds::Box { .. })) {
            settle(&mut g, &id, &stage, mode, false, &[], &mut repairs)?;
        }
    }
    let ids: Vec<String> = g.nodes.iter().map(|n| n.id.clone()).collect();
    for id in &ids {
        lift(&mut g, id);
    }
    face(&mut g);
    g.check()?;
    Ok(Commit { graph: g, repairs })
}

/// Applies `patch` to a copy of `graph` and solves it; `graph` itself never changes, so a refused
/// patch leaves nothing to roll back.
pub fn commit(graph: &Graph, patch: &Patch, mode: Mode) -> Result<Commit, String> {
    let mut g = graph.clone();
    let applied = patch::apply(&mut g, patch)?;
    solve(g, &applied, mode)
}

/// Checks a whole graph, as written in a text file or received from another participant. Every
/// node is kept to the rules; a node whose placing relation differs from `previous` is placed by
/// it, the others relate anew from where they stand. Without `previous` (opening a graph) the
/// positions are authoritative.
pub fn check(previous: Option<&Graph>, next: &Graph, mode: Mode) -> Result<Commit, String> {
    next.check()?;
    let placed = match previous {
        None => Vec::new(),
        Some(p) => next
            .edges
            .iter()
            .filter(|e| e.rel.places() && p.placement(&e.from) != Some(e))
            .map(|e| e.from.clone())
            .collect(),
    };
    let moved = next.nodes.iter().map(|n| n.id.clone()).collect();
    let applied = Applied {
        moved,
        placed,
        dropped: Vec::new(),
    };
    solve(next.clone(), &applied, mode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::History;

    const ROOM: &str = r#"{
      "schema": "ganbate.semantic.graph.v2",
      "stage": {"floorY": 0, "bounds": {"min": [-3, -3], "max": [3, 3]}},
      "environment": {"preset": "dusk"},
      "nodes": [
        {"id": "desk", "type": "desk", "position": [0, 0, 0], "surface": "walnut",
         "bounds": {"shape": "aabb", "halfExtents": [1.2, 0.38, 0.6], "surfaceY": 0.76}},
        {"id": "person", "type": "figure", "position": [0, 0, -1.2],
         "bounds": {"shape": "cylinder", "radius": 0.3, "height": 1.7}}
      ],
      "edges": [
        {"from": "person", "rel": "behind", "to": "desk", "clearance": 0.3},
        {"from": "person", "rel": "facing", "to": "desk"}
      ]
    }"#;

    fn room() -> Graph {
        Graph::parse(ROOM).unwrap()
    }

    fn run(g: &Graph, patch: &str, mode: Mode) -> Result<Commit, String> {
        commit(g, &Patch::parse(patch).unwrap(), mode)
    }

    fn put(id: &str, bounds: &str) -> String {
        format!(
            r#"{{"op": "put", "node": {{"id": "{id}", "type": "{id}", "position": [0, 0, 0], "bounds": {bounds}}}}}"#
        )
    }

    fn cup(id: &str) -> String {
        put(
            id,
            r#"{"shape": "cylinder", "radius": 0.1, "height": 0.15}"#,
        )
    }

    fn on_desk(ids: &[&str], anchor: &str) -> String {
        let ops: Vec<String> = ids
            .iter()
            .map(|id| format!(r#"{}, {{"op": "place", "id": "{id}", "rel": "on_top_of", "to": "desk"{anchor}}}"#, cup(id)))
            .collect();
        format!("[{}]", ops.join(", "))
    }

    fn at(g: &Graph, id: &str) -> [f32; 3] {
        g.node(id).unwrap().position
    }

    fn gap_between(g: &Graph, a: &str, b: &str) -> f32 {
        gap(foot(g.node(a).unwrap()), foot(g.node(b).unwrap()))
    }

    #[test]
    fn round_trips_its_own_text_and_keeps_unknown_fields() {
        let g = room();
        let again = Graph::parse(&g.canonical().unwrap()).unwrap();
        assert_eq!(again.digest().unwrap(), g.digest().unwrap());
        assert_eq!(
            again.node("desk").unwrap().extra["surface"],
            Value::String("walnut".into())
        );
        assert!(again.extra.contains_key("environment"));
        for bad in [
            ROOM.replace("v2", "v1"),
            ROOM.replace(r#""clearance": 0.3"#, r#""spin": 1"#),
            ROOM.replace(r#""clearance": 0.3"#, r#""clearance": -1"#),
            ROOM.replace(r#""surfaceY": 0.76"#, r#""surfaceY": 0.76, "offset": 1"#),
            ROOM.replace(r#""surfaceY": 0.76"#, r#""surfaceY": 0"#),
        ] {
            assert!(Graph::parse(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn items_placed_on_a_surface_stand_on_it_side_by_side_and_keep_their_anchor() {
        let g = run(
            &room(),
            &on_desk(&["a", "b", "c"], r#", "anchor": "left""#),
            Mode::Reject,
        )
        .unwrap()
        .graph;
        for (x, y) in [("a", "b"), ("a", "c"), ("b", "c")] {
            assert!(gap_between(&g, x, y) >= MIN_GAP - EPS);
        }
        for id in ["a", "b", "c"] {
            assert_eq!(at(&g, id)[1], 0.76);
            assert!(at(&g, id)[0] < 0.0);
            assert_eq!(
                g.placement(id).unwrap().anchor.as_deref(),
                Some("left"),
                "{id}"
            );
        }
    }

    #[test]
    fn in_front_follows_the_targets_heading() {
        let patch = format!(
            r#"[{{"op": "set", "id": "person", "key": "position", "value": [-2.5, 0, 0]}},
                {{"op": "set", "id": "desk", "key": "yawDeg", "value": 90}}, {},
                {{"op": "place", "id": "chair", "rel": "in_front_of", "to": "desk"}}]"#,
            put(
                "chair",
                r#"{"shape": "cylinder", "radius": 0.28, "height": 0.9}"#
            )
        );
        let g = run(&room(), &patch, Mode::Reject).unwrap().graph;
        let local = to_local(g.node("desk").unwrap(), xz(g.node("chair").unwrap()));
        assert!(
            (local[0]).abs() < EPS && (local[1] - (0.6 + 0.28 + DEFAULT_CLEARANCE)).abs() < EPS,
            "{local:?}"
        );
        assert_eq!(g.placement("chair").unwrap().rel, Relation::InFrontOf);
    }

    #[test]
    fn a_move_into_another_node_is_projected_out_or_refused() {
        let into = r#"[{"op": "set", "id": "person", "key": "position", "value": [0.2, 0, 0.1]}]"#;
        assert!(run(&room(), into, Mode::Reject).is_err());
        let done = run(&room(), into, Mode::Project).unwrap();
        assert!(gap_between(&done.graph, "person", "desk") >= MIN_GAP - EPS);
        assert_eq!(done.repairs.len(), 1, "{:?}", done.repairs);
    }

    #[test]
    fn relations_follow_where_a_node_was_moved_and_faces_turn() {
        let to_left = r#"[{"op": "set", "id": "person", "key": "position", "value": [-2, 0, 0]}]"#;
        let g = run(&room(), to_left, Mode::Reject).unwrap().graph;
        let placed = g.placement("person").unwrap();
        assert_eq!((placed.rel, placed.to.as_str()), (Relation::LeftOf, "desk"));
        assert_eq!(placed.clearance, Some(0.5));
        assert_eq!(g.node("person").unwrap().yaw_deg, 90.0);
    }

    #[test]
    fn what_stands_on_a_node_moves_with_it_and_settles_when_it_goes() {
        let g = run(&room(), &on_desk(&["a"], ""), Mode::Reject)
            .unwrap()
            .graph;
        let x = at(&g, "a")[0];
        let moved = r#"[{"op": "set", "id": "desk", "key": "position", "value": [1, 0, 0.5]}]"#;
        let moved = run(&g, moved, Mode::Reject).unwrap().graph;
        assert_eq!(at(&moved, "a")[0], x + 1.0);
        // Placing the desk by a relation carries what stands on it too.
        let placed = r#"[{"op": "place", "id": "desk", "rel": "in_front_of", "to": "person"}]"#;
        let placed = run(&g, placed, Mode::Reject).unwrap().graph;
        assert_eq!(at(&placed, "a")[1], 0.76);
        assert_eq!(support_of(&placed, "a").unwrap().id, "desk");
        // Taking the desk away is not a broken rule: the cup settles on the floor.
        let gone = run(&moved, r#"[{"op": "remove", "id": "desk"}]"#, Mode::Reject).unwrap();
        assert_eq!(at(&gone.graph, "a")[1], 0.0);
        assert!(gone.graph.edges.iter().all(|e| e.to != "desk"));
    }

    #[test]
    fn a_node_left_without_support_falls_onto_what_is_below() {
        let shelf = put(
            "shelf",
            r#"{"shape": "aabb", "halfExtents": [0.3, 0.1, 0.2]}"#,
        );
        let patch = format!(
            r#"[{shelf}, {{"op": "place", "id": "shelf", "rel": "on_top_of", "to": "desk"}}, {},
                {{"op": "place", "id": "a", "rel": "on_top_of", "to": "shelf"}}]"#,
            cup("a")
        );
        let g = run(&room(), &patch, Mode::Reject).unwrap().graph;
        assert_eq!(at(&g, "a")[1], 0.96);
        let g = run(&g, r#"[{"op": "remove", "id": "shelf"}]"#, Mode::Reject)
            .unwrap()
            .graph;
        assert_eq!(at(&g, "a")[1], 0.76);
        assert_eq!(g.placement("a").unwrap().to, "desk");
    }

    #[test]
    fn nothing_can_stand_on_what_stands_on_it() {
        let g = run(&room(), &on_desk(&["a"], ""), Mode::Reject)
            .unwrap()
            .graph;
        let cycle = r#"[{"op": "place", "id": "desk", "rel": "on_top_of", "to": "a"}]"#;
        for mode in [Mode::Project, Mode::Reject] {
            assert!(run(&g, cycle, mode).is_err());
        }
        let mut text = g.clone();
        text.edges.push(Edge {
            from: "desk".into(),
            rel: Relation::OnTopOf,
            to: "a".into(),
            anchor: None,
            clearance: None,
        });
        assert!(text.check().is_err());
    }

    #[test]
    fn turning_a_box_into_a_neighbour_is_caught() {
        let turn = r#"[{"op": "set", "id": "desk", "key": "yawDeg", "value": 90}]"#;
        assert!(run(&room(), turn, Mode::Reject).is_err());
        let g = run(&room(), turn, Mode::Project).unwrap().graph;
        assert!(gap_between(&g, "desk", "person") >= MIN_GAP - EPS);
    }

    #[test]
    fn the_stage_holds_its_floor_nodes_even_while_clearing_another() {
        let out = r#"[{"op": "set", "id": "person", "key": "position", "value": [5, 0, 0]}]"#;
        assert_eq!(
            at(&run(&room(), out, Mode::Project).unwrap().graph, "person")[0],
            2.7
        );
        let crate_by_wall = format!(
            r#"[{}, {{"op": "set", "id": "box", "key": "position", "value": [2.5, 0, 0]}},
                {{"op": "set", "id": "person", "key": "position", "value": [2.8, 0, 0]}}]"#,
            put(
                "box",
                r#"{"shape": "aabb", "halfExtents": [0.4, 0.5, 1.0]}"#
            )
        );
        let g = run(&room(), &crate_by_wall, Mode::Project).unwrap().graph;
        let person = foot(g.node("person").unwrap());
        assert!(inside_stage(&Stage::of(&g).unwrap(), person).is_none());
        assert!(gap_between(&g, "person", "box") >= MIN_GAP - EPS);
    }

    #[test]
    fn the_solver_accepts_what_it_produced() {
        let ball = |id: &str, x: f32, z: f32| {
            format!(
                r#"{{"op": "put", "node": {{"id": "{id}", "type": "ball", "position": [{x}, 0, {z}],
                    "bounds": {{"shape": "cylinder", "radius": 0.3, "height": 0.6}}}}}}"#
            )
        };
        let patch = format!("[{}, {}]", ball("p", -2.0, 2.0), ball("q", -1.8, 2.2));
        let g = run(&room(), &patch, Mode::Project).unwrap().graph;
        let q = at(&g, "q");
        let again = format!(
            r#"[{{"op": "set", "id": "q", "key": "position", "value": [{}, {}, {}]}}]"#,
            q[0], q[1], q[2]
        );
        assert!(run(&g, &again, Mode::Reject).is_ok());
        // Untouched nodes keep their exact positions.
        let unrounded = r#"[{"op": "graph", "key": "environment", "value": {"preset": "night"}}]"#;
        let mut fine = room();
        fine.nodes[1].position = [0.0, 0.0, -1.2004];
        assert_eq!(
            at(
                &run(&fine, unrounded, Mode::Reject).unwrap().graph,
                "person"
            )[2],
            -1.2004
        );
    }

    #[test]
    fn a_later_position_wins_over_an_earlier_placement() {
        let patch = format!(
            r#"[{}, {{"op": "place", "id": "a", "rel": "on_top_of", "to": "desk"}},
                {{"op": "set", "id": "a", "key": "position", "value": [2, 0, 2]}}]"#,
            cup("a")
        );
        assert_eq!(
            at(&run(&room(), &patch, Mode::Reject).unwrap().graph, "a"),
            [2.0, 0.0, 2.0]
        );
    }

    #[test]
    fn an_item_moved_off_its_surface_stays_where_it_was_put() {
        let g = run(&room(), &on_desk(&["a"], ""), Mode::Reject)
            .unwrap()
            .graph;
        let down = r#"[{"op": "set", "id": "a", "key": "position", "value": [2, 0, 1.5]}]"#;
        let g = run(&g, down, Mode::Reject).unwrap().graph;
        assert_eq!(at(&g, "a"), [2.0, 0.0, 1.5]);
        assert_ne!(g.placement("a").map(|e| e.rel), Some(Relation::OnTopOf));
    }

    #[test]
    fn undo_and_redo_restore_exact_states() {
        let mut history = History::new(8);
        let before = room();
        let night = r#"[{"op": "graph", "key": "environment", "value": {"preset": "night"}}]"#;
        let after = run(&before, night, Mode::Reject).unwrap().graph;
        history.record(before.clone());
        let undone = history.undo(&after).unwrap();
        assert_eq!(undone.digest().unwrap(), before.digest().unwrap());
        assert_eq!(
            history.redo(&undone).unwrap().digest().unwrap(),
            after.digest().unwrap()
        );
    }

    #[test]
    fn misspelt_or_unsound_operations_are_refused() {
        for bad in [
            r#"[{"op": "set", "id": "desk", "key": "position", "valeu": [0, 0, 0]}]"#,
            r#"[{"op": "place", "id": "person", "rel": "facing", "to": "desk"}]"#,
            r#"[{"op": "set", "id": "desk", "key": "id", "value": "table"}]"#,
            r#"[{"op": "place", "id": "person", "rel": "behind", "to": "person"}]"#,
            r#"[{"op": "place", "id": "person", "rel": "behind", "to": "desk", "clearance": 0}]"#,
            r#"[{"op": "graph", "key": "nodes", "value": []}]"#,
        ] {
            let refused = Patch::parse(bad).and_then(|p| commit(&room(), &p, Mode::Project));
            assert!(refused.is_err(), "{bad}");
        }
    }

    #[test]
    fn an_opened_graph_keeps_its_positions_and_a_solved_graph_is_unchanged_by_checking() {
        let opened = check(None, &room(), Mode::Reject).unwrap().graph;
        assert_eq!(at(&opened, "person"), [0.0, 0.0, -1.2]);
        let behind = opened.placement("person").unwrap();
        assert_eq!(
            (behind.rel, behind.clearance),
            (Relation::Behind, Some(0.3))
        );
        let patch = format!(
            r#"[{}, {{"op": "place", "id": "a", "rel": "on_top_of", "to": "desk", "anchor": "left"}},
                {}, {{"op": "place", "id": "b", "rel": "in_front_of", "to": "desk"}}]"#,
            cup("a"),
            cup("b")
        );
        let solved = run(&opened, &patch, Mode::Reject).unwrap().graph;
        let again = check(None, &solved, Mode::Reject).unwrap().graph;
        assert_eq!(again.digest().unwrap(), solved.digest().unwrap());
    }

    #[test]
    fn a_relation_rewritten_as_text_places_the_node_and_a_floating_one_is_caught() {
        let mut text = room();
        text.edges[0].rel = Relation::LeftOf;
        let placed = check(Some(&room()), &text, Mode::Reject).unwrap().graph;
        assert_eq!(at(&placed, "person"), [-1.8, 0.0, 0.0]);
        let mut floating = room();
        floating.nodes[1].position[1] = 0.5;
        assert!(check(Some(&room()), &floating, Mode::Reject).is_err());
        let landed = check(Some(&room()), &floating, Mode::Project).unwrap();
        assert_eq!(at(&landed.graph, "person")[1], 0.0);
        assert_eq!(landed.repairs.len(), 1);
    }
}
