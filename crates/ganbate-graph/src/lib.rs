//! Semantic scene graph (`ganbate.semantic.graph.v2`): a compact, diffable description of a stage
//! as nodes (things with a position, a heading and a footprint) and edges (spatial relations
//! between them). A person in the editor, a programmer in a text file and an agent through a tool
//! call all edit the same graph with the same patches ([`patch`]); the solver ([`spatial`]) places
//! nodes from their relations, keeps them supported, inside the stage and clear of each other, and
//! repairs or refuses what breaks that; [`History`] restores earlier states.
//!
//! Units are metres and degrees; +Y is up, a heading of 0° faces +Z, and a node's position is the
//! centre of its base. A node's front is its +Z side and its left its -X side, as seen by someone
//! standing in front of it and looking at it (a viewer's left). Fields the graph does not interpret (a colour, a material, an action) are
//! kept as they were read, on the node or on the graph, so a game's own data travels with it.
use std::collections::BTreeMap;

use ganbate_core::digest::{hex, sha256};
use ganbate_core::json::{self, Value};

pub mod patch;
pub mod spatial;

pub const SCHEMA: &str = "ganbate.semantic.graph.v2";

/// A node's footprint and height, centred on its position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bounds {
    /// A box in the node's own frame; `surface_y` is the height of its top surface when that is
    /// lower than the box (a table's top under a raised rim), otherwise twice `half_extents[1]`.
    Box {
        half_extents: [f32; 3],
        surface_y: Option<f32>,
    },
    Cylinder {
        radius: f32,
        height: f32,
    },
}

impl Bounds {
    /// Height above the node's base of what stands on it.
    pub fn top(&self) -> f32 {
        match *self {
            Self::Box {
                half_extents,
                surface_y,
            } => surface_y.unwrap_or(2.0 * half_extents[1]),
            Self::Cylinder { height, .. } => height,
        }
    }

    /// Half the footprint along the node's own X and Z axes.
    pub fn half_xz(&self) -> [f32; 2] {
        match *self {
            Self::Box { half_extents, .. } => [half_extents[0], half_extents[2]],
            Self::Cylinder { radius, .. } => [radius, radius],
        }
    }

    /// Radius of a circle that holds the footprint.
    pub fn radius(&self) -> f32 {
        match *self {
            Self::Box { half_extents, .. } => half_extents[0].hypot(half_extents[2]),
            Self::Cylinder { radius, .. } => radius,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relation {
    OnTopOf,
    LeftOf,
    RightOf,
    Behind,
    InFrontOf,
    Facing,
}

impl Relation {
    pub const ALL: [Self; 6] = [
        Self::OnTopOf,
        Self::LeftOf,
        Self::RightOf,
        Self::Behind,
        Self::InFrontOf,
        Self::Facing,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::OnTopOf => "on_top_of",
            Self::LeftOf => "left_of",
            Self::RightOf => "right_of",
            Self::Behind => "behind",
            Self::InFrontOf => "in_front_of",
            Self::Facing => "facing",
        }
    }

    pub fn parse(name: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|r| r.name() == name)
            .ok_or_else(|| format!("unknown relation \"{name}\""))
    }

    /// Whether the relation fixes where the node stands (every relation but `facing`).
    pub const fn places(self) -> bool {
        !matches!(self, Self::Facing)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub id: String,
    pub kind: String,
    pub position: [f32; 3],
    pub yaw_deg: f32,
    pub bounds: Option<Bounds>,
    /// Every other field of the node, as read.
    pub extra: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub from: String,
    pub rel: Relation,
    pub to: String,
    /// Where on the target: `center`, `left`, `right`, `front` or `back` (`on_top_of` only).
    pub anchor: Option<String>,
    /// Gap between the two footprints, metres.
    pub clearance: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    /// Every other field of the graph (`stage`, `environment`, a game's own sections), as read.
    pub extra: BTreeMap<String, Value>,
}

pub const ANCHORS: [&str; 5] = ["center", "left", "right", "front", "back"];

fn object(v: &Value, what: &str) -> Result<BTreeMap<String, Value>, String> {
    match v {
        Value::Object(m) => Ok(m.clone()),
        _ => Err(format!("{what} must be an object")),
    }
}

pub(crate) fn number(v: &Value, what: &str) -> Result<f32, String> {
    v.float()
        .map_err(|_| format!("{what} must be a finite number"))
}

pub(crate) fn vec3(v: &Value, what: &str) -> Result<[f32; 3], String> {
    match v.array() {
        Ok([x, y, z]) => Ok([number(x, what)?, number(y, what)?, number(z, what)?]),
        _ => Err(format!("{what} must be [x, y, z]")),
    }
}

fn num(x: f32) -> Value {
    Value::Number(f64::from(x))
}

fn text(s: &str) -> Value {
    Value::String(s.to_owned())
}

impl Bounds {
    pub(crate) fn from_json(v: &Value, id: &str) -> Result<Self, String> {
        let what = format!("node \"{id}\".bounds");
        let shape = v.get("shape").map(Value::string);
        let known: &[&str] = match shape {
            Some(Ok("aabb")) => &["shape", "halfExtents", "surfaceY"],
            _ => &["shape", "radius", "height"],
        };
        if let Some(key) = object(v, &what)?
            .keys()
            .find(|k| !known.contains(&k.as_str()))
        {
            return Err(format!("{what} has an unknown field \"{key}\""));
        }
        let positive = |key: &str| -> Result<f32, String> {
            let n = number(v.get(key).ok_or(format!("{what}.{key} is missing"))?, &what)?;
            if n > 0.0 {
                Ok(n)
            } else {
                Err(format!("{what}.{key} must be positive"))
            }
        };
        match shape {
            Some(Ok("aabb")) => {
                let half_extents = vec3(v.get("halfExtents").unwrap_or(&Value::Null), &what)?;
                if half_extents.iter().any(|h| *h <= 0.0) {
                    return Err(format!("{what}.halfExtents must be positive"));
                }
                let surface_y = v.get("surfaceY").map(|s| number(s, &what)).transpose()?;
                if surface_y.is_some_and(|s| s <= 0.0 || s > 2.0 * half_extents[1]) {
                    return Err(format!(
                        "{what}.surfaceY must be above 0 and within the box"
                    ));
                }
                Ok(Self::Box {
                    half_extents,
                    surface_y,
                })
            }
            Some(Ok("cylinder")) => Ok(Self::Cylinder {
                radius: positive("radius")?,
                height: positive("height")?,
            }),
            _ => Err(format!("{what}.shape must be \"aabb\" or \"cylinder\"")),
        }
    }

    fn to_json(self) -> Value {
        let mut m = BTreeMap::new();
        match self {
            Self::Box {
                half_extents,
                surface_y,
            } => {
                m.insert("shape".into(), text("aabb"));
                m.insert(
                    "halfExtents".into(),
                    Value::Array(half_extents.map(num).to_vec()),
                );
                if let Some(s) = surface_y {
                    m.insert("surfaceY".into(), num(s));
                }
            }
            Self::Cylinder { radius, height } => {
                m.insert("shape".into(), text("cylinder"));
                m.insert("radius".into(), num(radius));
                m.insert("height".into(), num(height));
            }
        }
        Value::Object(m)
    }
}

impl Node {
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let mut extra = object(v, "a node")?;
        let id = match extra.remove("id") {
            Some(Value::String(s)) if !s.is_empty() => s,
            _ => return Err("a node needs a nonempty \"id\"".into()),
        };
        let kind = match extra.remove("type") {
            Some(Value::String(s)) if !s.is_empty() => s,
            _ => return Err(format!("node \"{id}\" needs a nonempty \"type\"")),
        };
        let position = vec3(
            &extra.remove("position").unwrap_or(Value::Null),
            &format!("node \"{id}\".position"),
        )?;
        let yaw_deg = extra
            .remove("yawDeg")
            .map(|y| number(&y, &format!("node \"{id}\".yawDeg")))
            .transpose()?
            .unwrap_or(0.0);
        let bounds = extra
            .remove("bounds")
            .map(|b| Bounds::from_json(&b, &id))
            .transpose()?;
        Ok(Self {
            id,
            kind,
            position,
            yaw_deg,
            bounds,
            extra,
        })
    }

    pub fn to_json(&self) -> Value {
        let mut m = self.extra.clone();
        m.insert("id".into(), text(&self.id));
        m.insert("type".into(), text(&self.kind));
        m.insert(
            "position".into(),
            Value::Array(self.position.map(num).to_vec()),
        );
        m.insert("yawDeg".into(), num(self.yaw_deg));
        if let Some(b) = self.bounds {
            m.insert("bounds".into(), b.to_json());
        }
        Value::Object(m)
    }
}

impl Edge {
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let m = object(v, "an edge")?;
        if let Some(key) = m
            .keys()
            .find(|k| !matches!(k.as_str(), "from" | "rel" | "to" | "anchor" | "clearance"))
        {
            return Err(format!("an edge has an unknown field \"{key}\""));
        }
        let field = |key: &str| -> Result<String, String> {
            match m.get(key) {
                Some(Value::String(s)) if !s.is_empty() => Ok(s.clone()),
                _ => Err(format!("an edge needs a nonempty \"{key}\"")),
            }
        };
        let rel = Relation::parse(&field("rel")?)?;
        let anchor = m
            .get("anchor")
            .map(|a| a.string().map(str::to_owned))
            .transpose()?;
        if let Some(a) = &anchor {
            if rel != Relation::OnTopOf || !ANCHORS.contains(&a.as_str()) {
                return Err(format!(
                    "anchor \"{a}\" needs on_top_of and one of {ANCHORS:?}"
                ));
            }
        }
        let clearance = m
            .get("clearance")
            .map(|c| number(c, "an edge's clearance"))
            .transpose()?;
        if clearance.is_some_and(|c| c < 0.0) {
            return Err("an edge's clearance must not be negative".into());
        }
        Ok(Self {
            from: field("from")?,
            rel,
            to: field("to")?,
            anchor,
            clearance,
        })
    }

    pub fn to_json(&self) -> Value {
        let mut m = BTreeMap::new();
        m.insert("from".into(), text(&self.from));
        m.insert("rel".into(), text(self.rel.name()));
        m.insert("to".into(), text(&self.to));
        if let Some(a) = &self.anchor {
            m.insert("anchor".into(), text(a));
        }
        if let Some(c) = self.clearance {
            m.insert("clearance".into(), num(c));
        }
        Value::Object(m)
    }
}

impl Graph {
    pub fn parse(text: &str) -> Result<Self, String> {
        Self::from_json(&json::parse(text.as_bytes())?)
    }

    pub fn from_json(v: &Value) -> Result<Self, String> {
        let mut extra = object(v, "the graph")?;
        match extra.remove("schema") {
            Some(Value::String(s)) if s == SCHEMA => {}
            _ => return Err(format!("the graph's \"schema\" must be \"{SCHEMA}\"")),
        }
        let list = |key: &str, extra: &mut BTreeMap<String, Value>| match extra.remove(key) {
            None => Ok(Vec::new()),
            Some(Value::Array(a)) => Ok(a),
            Some(_) => Err(format!("\"{key}\" must be an array")),
        };
        let nodes = list("nodes", &mut extra)?
            .iter()
            .map(Node::from_json)
            .collect::<Result<_, _>>()?;
        let edges = list("edges", &mut extra)?
            .iter()
            .map(Edge::from_json)
            .collect::<Result<_, _>>()?;
        let graph = Self {
            nodes,
            edges,
            extra,
        };
        graph.check()?;
        Ok(graph)
    }

    /// Node ids are unique; every edge joins two different existing nodes; a node has at most
    /// one placing relation and one `facing`.
    pub fn check(&self) -> Result<(), String> {
        for (i, n) in self.nodes.iter().enumerate() {
            if self.nodes[..i].iter().any(|m| m.id == n.id) {
                return Err(format!("two nodes have the id \"{}\"", n.id));
            }
        }
        for (i, e) in self.edges.iter().enumerate() {
            for end in [&e.from, &e.to] {
                if self.node(end).is_none() {
                    return Err(format!("an edge names the missing node \"{end}\""));
                }
            }
            if e.from == e.to {
                return Err(format!("node \"{}\" relates to itself", e.from));
            }
            if self.edges[..i]
                .iter()
                .any(|f| f.from == e.from && f.rel.places() == e.rel.places())
            {
                return Err(format!(
                    "node \"{}\" has two {} relations",
                    e.from,
                    e.rel.name()
                ));
            }
        }
        if let Some(n) = self
            .nodes
            .iter()
            .find(|n| self.rests_on_through(&n.id, &n.id))
        {
            return Err(format!("\"{}\" rests on itself through other nodes", n.id));
        }
        Ok(())
    }

    /// Whether `upper` stands, directly or on other nodes, on `lower`.
    pub fn rests_on(&self, upper: &str, lower: &str) -> bool {
        upper != lower && self.rests_on_through(upper, lower)
    }

    fn rests_on_through(&self, upper: &str, lower: &str) -> bool {
        let mut at = upper;
        for _ in 0..self.nodes.len() {
            match self
                .edges
                .iter()
                .find(|e| e.from == at && e.rel == Relation::OnTopOf)
            {
                Some(e) if e.to == lower => return true,
                Some(e) => at = &e.to,
                None => return false,
            }
        }
        false
    }

    pub fn to_json(&self) -> Value {
        let mut m = self.extra.clone();
        m.insert("schema".into(), text(SCHEMA));
        m.insert(
            "nodes".into(),
            Value::Array(self.nodes.iter().map(Node::to_json).collect()),
        );
        m.insert(
            "edges".into(),
            Value::Array(self.edges.iter().map(Edge::to_json).collect()),
        );
        Value::Object(m)
    }

    /// One spelling per graph (see `ganbate_core::json`), so equal graphs have equal text.
    pub fn canonical(&self) -> Result<String, String> {
        json::to_canonical(&self.to_json())
    }

    /// sha256 of the canonical text, in hex: the graph's identity across editor, code and agents.
    pub fn digest(&self) -> Result<String, String> {
        Ok(hex(&sha256(self.canonical()?.as_bytes())))
    }

    pub fn node(&self, id: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn node_mut(&mut self, id: &str) -> Option<&mut Node> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    /// The relation that fixes where `id` stands, if any.
    pub fn placement(&self, id: &str) -> Option<&Edge> {
        self.edges.iter().find(|e| e.from == id && e.rel.places())
    }
}

/// Earlier and later states of a graph, for undo and redo.
#[derive(Clone, Debug)]
pub struct History {
    past: Vec<Graph>,
    future: Vec<Graph>,
    limit: usize,
}

impl History {
    pub fn new(limit: usize) -> Self {
        Self {
            past: Vec::new(),
            future: Vec::new(),
            limit: limit.max(1),
        }
    }

    /// Keeps `before`, the state a change replaced; a new change forgets what was undone.
    pub fn record(&mut self, before: Graph) {
        self.past.push(before);
        if self.past.len() > self.limit {
            self.past.remove(0);
        }
        self.future.clear();
    }

    pub fn undo(&mut self, current: &Graph) -> Option<Graph> {
        let previous = self.past.pop()?;
        self.future.push(current.clone());
        Some(previous)
    }

    pub fn redo(&mut self, current: &Graph) -> Option<Graph> {
        let next = self.future.pop()?;
        self.past.push(current.clone());
        Some(next)
    }
}
