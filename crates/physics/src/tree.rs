//! A dynamic bounding-volume tree: the broad phase's index of colliders, and the query index of
//! anything else that needs "which boxes overlap this box or this ray" quickly.
//!
//! Proxies are leaves holding a box and a small `Copy` payload. Boxes are stored exactly as given;
//! callers fatten them (the world adds a margin so a slowly moving collider does not reinsert
//! every step). The same sequence of calls always builds the same tree, so queries visit
//! proxies in a deterministic order.
//!
//! How it works:
//! - Nodes live in one pool with a free list. A proxy id is the index of its leaf in the pool, so
//!   it stays the same while the proxy lives (an update re-places the same leaf) and is handed out
//!   again after the proxy is removed (the most recently freed id first).
//! - Insertion pairs the new leaf with the node that adds the least surface area to the tree,
//!   found by a greedy descent with lower bounds (Box2D's `b2DynamicTree`). On the way back up,
//!   each ancestor is refitted and takes the swap of a child with a grandchild (a tree rotation)
//!   that shrinks its boxes most. A removal frees the leaf's parent and refits the ancestors.
//! - Surface area alone can build a tree as deep as it has leaves (boxes nested inside each other,
//!   or many identical ones), so the height is budgeted: insertion only pairs with nodes that keep
//!   the tree at most `2 · bit_length(n) − 1` high (about `2 · log2(n)`), steered by each node's
//!   shallowest leaf, and rotations never make a node taller. A removal that leaves the tree two
//!   levels over that rebuilds it top-down, by surface area with balanced splits; that needs about
//!   half the proxies removed since the tree was last that tall, so it is rare.
//! - Queries and rays walk the tree with a small explicit stack, rays nearer child first. Nothing
//!   allocates per query.

use std::num::NonZeroU32;

use crate::math::{Aabb, Vec3};

pub const NULL: u32 = u32::MAX;

/// The height that marks a node on the free list.
const FREE: u16 = u16::MAX;

const FREE_CHILDREN: [NonZeroU32; 2] = [NonZeroU32::MAX; 2];

#[inline]
fn encode_child(c: u32) -> NonZeroU32 {
    NonZeroU32::new(c + 1).expect("child index < NULL")
}

#[inline]
fn decode_child(c: NonZeroU32) -> u32 {
    c.get() - 1
}

#[derive(Clone, Copy, Debug)]
enum NodePayload<T: Copy> {
    Internal([NonZeroU32; 2]),
    Leaf(T),
}

#[derive(Clone, Copy, Debug)]
struct Node<T: Copy> {
    aabb: Aabb,
    /// The parent (`NULL` at the root), or the next free node while on the free list.
    parent: u32,
    /// The depth of the deepest leaf below: 0 for a leaf, [`FREE`] when free.
    height: u16,
    /// The depth of the shallowest leaf below (0 for a leaf).
    short: u16,
    /// Internal children or leaf payload (overlapping in 8 bytes for 32-bit payloads).
    payload: NodePayload<T>,
}

const _: () = assert!(std::mem::size_of::<Node<u32>>() == 40);

impl<T: Copy> Node<T> {
    #[inline]
    fn children(&self) -> [u32; 2] {
        match self.payload {
            NodePayload::Internal([a, b]) => [decode_child(a), decode_child(b)],
            NodePayload::Leaf(_) => [NULL; 2],
        }
    }

    #[inline]
    fn set_child(&mut self, slot: usize, child: u32) {
        if let NodePayload::Internal(c) = &mut self.payload {
            c[slot] = encode_child(child);
        } else {
            unreachable!("set_child on leaf node");
        }
    }

    #[inline]
    fn leaf_data(&self) -> Option<T> {
        if self.height == 0
            && let NodePayload::Leaf(d) = self.payload
        {
            Some(d)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct DynamicTree<T: Copy> {
    nodes: Vec<Node<T>>,
    root: u32,
    /// The head of the free list.
    free: u32,
    /// The number of live proxies.
    leaves: usize,
}

impl<T: Copy> Default for DynamicTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// The number of bits `n` needs: `floor(log2(n)) + 1` for `n > 0`.
fn bit_length(n: usize) -> u32 {
    usize::BITS - n.leading_zeros()
}

/// How high insertions may make a tree of `n ≥ 1` leaves: `2 · bit_length(n) − 1`.
fn insertion_budget(n: usize) -> u32 {
    (2 * bit_length(n)).saturating_sub(1)
}

/// The most a tree of `n` leaves is ever high (removals rebuild a taller one): two levels over
/// the insertion budget, at most `2 · log2(n) + 3`, which is below `3 · log2(n)` for `n ≥ 2`.
pub fn height_limit(n: usize) -> u32 {
    insertion_budget(n) + 2
}

impl<T: Copy> DynamicTree<T> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root: NULL,
            free: NULL,
            leaves: 0,
        }
    }

    /// Adds a leaf and returns its proxy id (stable until removed; ids are reused afterwards).
    pub fn insert(&mut self, aabb: Aabb, data: T) -> u32 {
        let leaf = self.allocate();
        self.nodes[leaf as usize] = Node {
            aabb,
            parent: NULL,
            height: 0,
            short: 0,
            payload: NodePayload::Leaf(data),
        };
        self.leaves += 1;
        self.insert_leaf(leaf);
        leaf
    }

    pub fn remove(&mut self, proxy: u32) {
        self.expect_live(proxy);
        self.remove_leaf(proxy);
        // Freed after its parent, so the next insertion gets this id back.
        self.release(proxy);
        self.leaves -= 1;
        if self.leaves >= 2 && self.height() > height_limit(self.leaves) {
            self.rebuild();
        }
    }

    /// Gives a leaf a new box (the tree re-places it).
    pub fn update(&mut self, proxy: u32, aabb: Aabb) {
        self.expect_live(proxy);
        self.remove_leaf(proxy);
        self.nodes[proxy as usize].aabb = aabb;
        self.insert_leaf(proxy);
    }

    pub fn aabb(&self, proxy: u32) -> Aabb {
        self.expect_live(proxy);
        self.nodes[proxy as usize].aabb
    }

    pub fn data(&self, proxy: u32) -> T {
        self.expect_live(proxy);
        self.nodes[proxy as usize].leaf_data().expect("live proxy")
    }

    /// Number of live proxies.
    pub fn len(&self) -> usize {
        self.leaves
    }

    pub fn is_empty(&self) -> bool {
        self.leaves == 0
    }

    /// The number of edges on the longest path from the root to a leaf (0 when empty or with one
    /// proxy); never more than [`height_limit`] of [`len`](Self::len).
    pub fn height(&self) -> u32 {
        if self.root == NULL {
            0
        } else {
            u32::from(self.nodes[self.root as usize].height)
        }
    }

    /// Calls `f(proxy, data)` for every leaf whose box overlaps `aabb`, until `f` returns false.
    pub fn query(&self, aabb: &Aabb, f: &mut dyn FnMut(u32, T) -> bool) {
        if self.root == NULL {
            return;
        }
        let mut stack = WalkStack::<u32, 32>::new(NULL);
        stack.push(self.root);
        while let Some(i) = stack.pop() {
            let node = &self.nodes[i as usize];
            if !node.aabb.overlaps(aabb) {
                continue;
            }
            if node.height == 0 {
                if let NodePayload::Leaf(data) = node.payload
                    && !f(i, data)
                {
                    return;
                }
            } else {
                let [c0, c1] = node.children();
                stack.push(c1);
                stack.push(c0);
            }
        }
    }

    /// Visits leaves whose boxes the ray `origin + t·dir` (`dir` unit) enters for `t` in
    /// `[0, max_t]`, nearest boxes first where cheap. `f(proxy, data, max_t)` returns the new
    /// `max_t` (smaller after a hit, which clips the rest of the walk; return the given value to
    /// keep going, or a negative value to stop). A larger value than the one given keeps the
    /// given one: the walk only ever shortens.
    pub fn raycast(
        &self,
        origin: Vec3,
        dir: Vec3,
        max_t: f32,
        f: &mut dyn FnMut(u32, T, f32) -> f32,
    ) {
        if self.root == NULL || max_t.is_nan() || max_t < 0.0 {
            return;
        }
        let inv = crate::math::inverse_direction(dir);
        let mut max_t = max_t;
        let Some(t) = self.nodes[self.root as usize]
            .aabb
            .ray_entry(origin, inv, max_t)
        else {
            return;
        };
        // Each entry carries where the ray enters the node's box, so a node that a later hit
        // has moved out of reach is dropped without another slab test.
        let mut stack = WalkStack::<(u32, f32), 32>::new((NULL, 0.0));
        stack.push((self.root, t));
        while let Some((i, entry)) = stack.pop() {
            if entry > max_t {
                continue;
            }
            let node = &self.nodes[i as usize];
            if node.height == 0 {
                if let NodePayload::Leaf(data) = node.payload {
                    let m = f(i, data, max_t);
                    if m < 0.0 {
                        return;
                    }
                    // `min` keeps `max_t` when the callback returns NaN.
                    max_t = max_t.min(m);
                }
                continue;
            }
            let [a, b] = node.children();
            let ta = self.nodes[a as usize].aabb.ray_entry(origin, inv, max_t);
            let tb = self.nodes[b as usize].aabb.ray_entry(origin, inv, max_t);
            match (ta, tb) {
                (Some(ta), Some(tb)) => {
                    // The nearer child goes on top so it is walked first.
                    if tb < ta {
                        stack.push((a, ta));
                        stack.push((b, tb));
                    } else {
                        stack.push((b, tb));
                        stack.push((a, ta));
                    }
                }
                (Some(ta), None) => stack.push((a, ta)),
                (None, Some(tb)) => stack.push((b, tb)),
                (None, None) => {}
            }
        }
    }

    /// Checks the tree's structure: parent and child links agree; every internal node's box is
    /// the union of its children's, its height one more than its taller child's and its shortest
    /// path one more than its shorter child's; the height is within [`height_limit`]; leaves hold
    /// data and internal nodes none; the leaf count matches [`len`](Self::len); and every node is
    /// either in the tree or free.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.nodes.len();
        let mut free = 0usize;
        let mut i = self.free;
        while i != NULL {
            let node = self
                .nodes
                .get(i as usize)
                .ok_or_else(|| format!("free node {i} is out of range"))?;
            if node.height != FREE {
                return Err(format!("node {i} is on the free list but in use"));
            }
            free += 1;
            if free > n {
                return Err("the free list loops".into());
            }
            i = node.parent;
        }
        let (mut reached, mut leaves) = (0usize, 0usize);
        if self.root != NULL {
            let root = self
                .nodes
                .get(self.root as usize)
                .ok_or_else(|| format!("root {} is out of range", self.root))?;
            if root.parent != NULL {
                return Err(format!("root {} has parent {}", self.root, root.parent));
            }
            let mut stack = vec![self.root];
            while let Some(i) = stack.pop() {
                reached += 1;
                if reached > n {
                    return Err("the tree has a cycle".into());
                }
                let node = &self.nodes[i as usize];
                match node.height {
                    FREE => return Err(format!("node {i} is in the tree but free")),
                    0 => {
                        if node.short != 0 {
                            return Err(format!("leaf {i} has shortest path {}", node.short));
                        }
                        if !matches!(node.payload, NodePayload::Leaf(_)) {
                            return Err(format!("leaf {i} has no data"));
                        }
                        leaves += 1;
                    }
                    height => {
                        if !matches!(node.payload, NodePayload::Internal(_)) {
                            return Err(format!("internal node {i} holds data"));
                        }
                        let [a, b] = node.children();
                        for c in [a, b] {
                            let child = self
                                .nodes
                                .get(c as usize)
                                .ok_or_else(|| format!("node {i}'s child {c} is out of range"))?;
                            if child.parent != i {
                                return Err(format!(
                                    "node {c}'s parent is {}, not {i}",
                                    child.parent
                                ));
                            }
                            if child.height == FREE {
                                return Err(format!("node {i}'s child {c} is free"));
                            }
                        }
                        let (na, nb) = (&self.nodes[a as usize], &self.nodes[b as usize]);
                        if height != 1 + na.height.max(nb.height)
                            || node.short != 1 + na.short.min(nb.short)
                        {
                            return Err(format!(
                                "node {i} has height {height} and shortest path {}, its \
                                 children {}/{} and {}/{}",
                                node.short, na.height, na.short, nb.height, nb.short
                            ));
                        }
                        if node.aabb != na.aabb.union(nb.aabb) {
                            return Err(format!(
                                "node {i}'s box {:?} is not the union of its children's",
                                node.aabb
                            ));
                        }
                        stack.push(b);
                        stack.push(a);
                    }
                }
            }
        }
        if leaves != self.leaves {
            return Err(format!(
                "{leaves} leaves in the tree, {} proxies counted",
                self.leaves
            ));
        }
        if reached + free != n {
            return Err(format!(
                "{} of {n} nodes are neither in the tree nor free",
                n - reached - free
            ));
        }
        if leaves >= 2 && self.height() > height_limit(leaves) {
            return Err(format!(
                "height {} exceeds {} for {leaves} leaves",
                self.height(),
                height_limit(leaves)
            ));
        }
        Ok(())
    }

    fn expect_live(&self, proxy: u32) {
        let live = self
            .nodes
            .get(proxy as usize)
            .is_some_and(|n| n.height == 0 && matches!(n.payload, NodePayload::Leaf(_)));
        assert!(live, "proxy {proxy} is not live");
    }

    fn allocate(&mut self) -> u32 {
        if self.free != NULL {
            let i = self.free;
            self.free = self.nodes[i as usize].parent;
            return i;
        }
        let i = self.nodes.len();
        assert!(i < NULL as usize, "the tree is full");
        self.nodes.push(Node {
            aabb: Aabb::EMPTY,
            parent: NULL,
            height: FREE,
            short: 0,
            payload: NodePayload::Internal(FREE_CHILDREN),
        });
        i as u32
    }

    fn release(&mut self, i: u32) {
        self.nodes[i as usize] = Node {
            aabb: Aabb::EMPTY,
            parent: self.free,
            height: FREE,
            short: 0,
            payload: NodePayload::Internal(FREE_CHILDREN),
        };
        self.free = i;
    }

    /// Links a leaf (counted in `self.leaves` already) into the tree.
    fn insert_leaf(&mut self, leaf: u32) {
        if self.root == NULL {
            self.root = leaf;
            self.nodes[leaf as usize].parent = NULL;
            return;
        }
        let aabb = self.nodes[leaf as usize].aabb;
        let sibling = self.find_best_sibling(&aabb, insertion_budget(self.leaves));
        let old_parent = self.nodes[sibling as usize].parent;
        let parent = self.allocate();
        let s = self.nodes[sibling as usize];
        self.nodes[parent as usize] = Node {
            aabb: s.aabb.union(aabb),
            parent: old_parent,
            height: s.height + 1,
            short: 1,
            payload: NodePayload::Internal([encode_child(sibling), encode_child(leaf)]),
        };
        self.nodes[sibling as usize].parent = parent;
        self.nodes[leaf as usize].parent = parent;
        if old_parent == NULL {
            self.root = parent;
        } else {
            self.replace_child(old_parent, sibling, parent);
        }
        self.refit_from(parent);
    }

    /// Unlinks a leaf (not freeing it) and frees its parent, whose place its sibling takes.
    fn remove_leaf(&mut self, leaf: u32) {
        if leaf == self.root {
            self.root = NULL;
            return;
        }
        let parent = self.nodes[leaf as usize].parent;
        let grandparent = self.nodes[parent as usize].parent;
        let [a, b] = self.nodes[parent as usize].children();
        let sibling = if a == leaf { b } else { a };
        if grandparent == NULL {
            self.root = sibling;
        } else {
            self.replace_child(grandparent, parent, sibling);
        }
        self.nodes[sibling as usize].parent = grandparent;
        self.nodes[leaf as usize].parent = NULL;
        self.release(parent);
        // Only refit: rotating here too costs time and gains nothing measurable.
        let mut i = grandparent;
        while i != NULL {
            self.refit(i);
            i = self.nodes[i as usize].parent;
        }
    }

    fn replace_child(&mut self, parent: u32, old: u32, new: u32) {
        let p = &mut self.nodes[parent as usize];
        let [c0, _] = p.children();
        if c0 == old {
            p.set_child(0, new);
        } else {
            p.set_child(1, new);
        }
    }

    /// The node to pair a new leaf with: the one whose choice adds the least surface area, among
    /// those that keep the tree within `budget` (a node at depth `d` of height `h` qualifies when
    /// `d + h + 1 <= budget`). Descends towards the child with the lower bound on its cost, into
    /// children that still hold a leaf shallow enough to qualify, until neither child can beat
    /// the best found so far.
    fn find_best_sibling(&self, leaf: &Aabb, budget: u32) -> u32 {
        let leaf_cost = cost(leaf);
        let leaf_center = leaf.center();
        let (mut index, mut depth) = (self.root, 0u32);
        // The area the ancestors of `index` gain when the leaf goes below them.
        let mut inherited = 0.0f32;
        let mut best = NULL;
        let mut best_cost = f32::INFINITY;
        loop {
            let node = &self.nodes[index as usize];
            let direct = cost(&node.aabb.union(*leaf));
            // `best == NULL` takes a first candidate even when costs overflow to NaN.
            if depth + u32::from(node.height) < budget
                && (best == NULL || inherited + direct < best_cost)
            {
                best = index;
                best_cost = inherited + direct;
            }
            if node.height == 0 {
                // The descent only enters subtrees with a qualifying leaf, and the root always
                // has one (a tree whose leaves are all at depth `s` or more has `2^s` leaves), so
                // `best` is set; the fallback only guards against a broken tree.
                return if best == NULL { index } else { best };
            }
            inherited += direct - cost(&node.aabb);
            let [a, b] = node.children();
            let open = |c: u32| depth + 1 + u32::from(self.nodes[c as usize].short) < budget;
            let (open_a, open_b) = (open(a), open(b));
            let lower_a = if open_a {
                self.lower_bound(a, leaf, leaf_cost, inherited)
            } else {
                f32::INFINITY
            };
            let lower_b = if open_b {
                self.lower_bound(b, leaf, leaf_cost, inherited)
            } else {
                f32::INFINITY
            };
            if best != NULL && best_cost <= lower_a && best_cost <= lower_b {
                return best;
            }
            index = if !open_b {
                a
            } else if !open_a {
                b
            } else if lower_a == lower_b {
                // No clear choice (both children hold the new box, say): go towards the nearer.
                let da = (self.nodes[a as usize].aabb.center() - leaf_center).length_squared();
                let db = (self.nodes[b as usize].aabb.center() - leaf_center).length_squared();
                if db < da { b } else { a }
            } else if lower_b < lower_a {
                b
            } else {
                a
            };
            depth += 1;
        }
    }

    /// A lower bound on the cost of pairing the new leaf with `child` or a node below it, where
    /// `inherited` is what `child`'s ancestors gain.
    fn lower_bound(&self, child: u32, leaf: &Aabb, leaf_cost: f32, inherited: f32) -> f32 {
        let node = &self.nodes[child as usize];
        let direct = cost(&node.aabb.union(*leaf));
        if node.height == 0 {
            inherited + direct
        } else {
            // The child itself costs `direct`; below it, the child grows to `direct` and the new
            // parent covers at least the leaf.
            inherited + direct + (leaf_cost - cost(&node.aabb)).min(0.0)
        }
    }

    /// Refits and rotates `index` and each of its ancestors (after an insertion).
    fn refit_from(&mut self, mut index: u32) {
        while index != NULL {
            self.refit(index);
            self.rotate(index);
            index = self.nodes[index as usize].parent;
        }
    }

    fn refit(&mut self, i: u32) {
        let [a, b] = self.nodes[i as usize].children();
        let (na, nb) = (self.nodes[a as usize], self.nodes[b as usize]);
        let node = &mut self.nodes[i as usize];
        node.aabb = na.aabb.union(nb.aabb);
        node.height = 1 + na.height.max(nb.height);
        node.short = 1 + na.short.min(nb.short);
    }

    /// Takes the swap of a child of `a` with a grandchild under its other child (a tree rotation)
    /// that shrinks the boxes most, among those that do not make `a` taller.
    fn rotate(&mut self, a: u32) {
        let node = self.nodes[a as usize];
        if node.height < 2 {
            return;
        }
        let node_children = node.children();
        let mut best: Option<(usize, usize)> = None;
        let mut best_gain = 0.0f32;
        for slot in 0..2 {
            let x = self.nodes[node_children[slot] as usize];
            let other = self.nodes[node_children[1 - slot] as usize];
            if other.height == 0 {
                continue;
            }
            let other_children = other.children();
            let other_cost = cost(&other.aabb);
            for inner in 0..2 {
                let y = self.nodes[other_children[inner] as usize];
                let z = self.nodes[other_children[1 - inner] as usize];
                // After the swap `other` holds (x, z) and `a` holds (y, other).
                let other_height = 1 + x.height.max(z.height);
                if 1 + y.height.max(other_height) > node.height {
                    continue;
                }
                let gain = cost(&x.aabb.union(z.aabb)) - other_cost;
                if gain < best_gain {
                    best_gain = gain;
                    best = Some((slot, inner));
                }
            }
        }
        if let Some((slot, inner)) = best {
            self.swap(a, slot, inner);
        }
    }

    /// Swaps `a`'s child in `slot` with the grandchild in `inner` under `a`'s other child, and
    /// refits the other child and `a` (whose box covers the same leaves as before).
    fn swap(&mut self, a: u32, slot: usize, inner: usize) {
        let a_children = self.nodes[a as usize].children();
        let x = a_children[slot];
        let c = a_children[1 - slot];
        let y = self.nodes[c as usize].children()[inner];
        self.nodes[a as usize].set_child(slot, y);
        self.nodes[c as usize].set_child(inner, x);
        self.nodes[x as usize].parent = c;
        self.nodes[y as usize].parent = a;
        self.refit(c);
        self.refit(a);
    }

    /// Rebuilds the tree top-down from its leaves (which keep their ids), splitting each node at
    /// the cheapest surface-area split along its longest axis that leaves at least a third of
    /// the leaves on each side, so the result is at most about `1.71 · log2(n)` high.
    fn rebuild(&mut self) {
        let mut leaves = Vec::with_capacity(self.leaves);
        let mut internal = Vec::with_capacity(self.leaves);
        let mut stack = vec![self.root];
        while let Some(i) = stack.pop() {
            let node = &self.nodes[i as usize];
            if node.height == 0 {
                leaves.push(i);
            } else {
                internal.push(i);
                let [c0, c1] = node.children();
                stack.push(c1);
                stack.push(c0);
            }
        }
        for &i in internal.iter().rev() {
            self.release(i);
        }
        let mut scratch = Vec::with_capacity(leaves.len());
        self.root = self.build(&mut leaves, &mut scratch);
        self.nodes[self.root as usize].parent = NULL;
    }

    /// Builds a subtree over `leaves` and returns its root. Recursion is as deep as the result,
    /// which the balanced splits keep logarithmic.
    fn build(&mut self, leaves: &mut [u32], scratch: &mut Vec<f32>) -> u32 {
        let m = leaves.len();
        if m == 1 {
            return leaves[0];
        }
        let center = |nodes: &[Node<T>], i: u32| nodes[i as usize].aabb.center();
        let span = leaves
            .iter()
            .fold(Aabb::EMPTY, |b, &i| b.include(center(&self.nodes, i)));
        let axis = (span.max - span.min).max_axis();
        let nodes = &self.nodes;
        leaves.sort_unstable_by(|&p, &q| {
            center(nodes, p)[axis]
                .total_cmp(&center(nodes, q)[axis])
                .then(p.cmp(&q))
        });
        // scratch[k]: the cost of the leaves from k on.
        scratch.clear();
        scratch.resize(m, 0.0);
        let mut b = Aabb::EMPTY;
        for k in (1..m).rev() {
            b = b.union(nodes[leaves[k] as usize].aabb);
            scratch[k] = cost(&b) * (m - k) as f32;
        }
        let low = m.div_ceil(3);
        let mut split = low;
        let mut best = f32::INFINITY;
        let mut b = Aabb::EMPTY;
        for k in 1..=m - low {
            b = b.union(nodes[leaves[k - 1] as usize].aabb);
            if k >= low {
                let c = cost(&b) * k as f32 + scratch[k];
                if c < best {
                    best = c;
                    split = k;
                }
            }
        }
        let (left, right) = leaves.split_at_mut(split);
        let a = self.build(left, scratch);
        let b = self.build(right, scratch);
        let parent = self.allocate();
        self.nodes[parent as usize] = Node {
            aabb: Aabb::EMPTY,
            parent: NULL,
            height: 0,
            short: 0,
            payload: NodePayload::Internal([encode_child(a), encode_child(b)]),
        };
        self.nodes[a as usize].parent = parent;
        self.nodes[b as usize].parent = parent;
        self.refit(parent);
        parent
    }
}

/// The surface-area cost of a box: half its area, plus a little of its squared perimeter so that
/// flat, thin and point-like boxes (which have no area) still group by distance.
#[inline]
fn cost(b: &Aabb) -> f32 {
    let d = b.max - b.min;
    let p = d.x + d.y + d.z;
    d.x * d.y + d.y * d.z + d.z * d.x + p * p * (1.0 / 32.0)
}

/// A last-in-first-out stack for tree walks that lives on the call stack up to `N` entries and
/// spills to the heap beyond that (which balanced trees never need), so a walk allocates nothing
/// and still cannot overflow.
pub(crate) struct WalkStack<E: Copy, const N: usize> {
    fixed: [E; N],
    len: usize,
    spill: Vec<E>,
}

impl<E: Copy, const N: usize> WalkStack<E, N> {
    /// An empty stack; `fill` only initializes the unused slots.
    #[inline]
    pub(crate) fn new(fill: E) -> Self {
        Self {
            fixed: [fill; N],
            len: 0,
            spill: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn push(&mut self, e: E) {
        if self.len < N {
            self.fixed[self.len] = e;
            self.len += 1;
        } else {
            self.spill.push(e);
        }
    }

    #[inline]
    pub(crate) fn pop(&mut self) -> Option<E> {
        // Entries spill only while the fixed part is full, so the spilled ones are the newest.
        if let Some(e) = self.spill.pop() {
            return Some(e);
        }
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(self.fixed[self.len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube(center: Vec3, half: f32) -> Aabb {
        Aabb::from_center_half(center, Vec3::splat(half))
    }

    fn found(tree: &DynamicTree<u32>, b: &Aabb) -> Vec<u32> {
        let mut seen = vec![];
        tree.query(b, &mut |p, _| {
            seen.push(p);
            true
        });
        seen
    }

    #[test]
    fn an_empty_tree_finds_nothing_and_is_valid() {
        let tree = DynamicTree::<u32>::default();
        tree.validate().unwrap();
        assert!(tree.is_empty());
        assert_eq!((tree.len(), tree.height()), (0, 0));
        let mut calls = 0;
        tree.query(&cube(Vec3::ZERO, 1e9), &mut |_, _| {
            calls += 1;
            true
        });
        tree.raycast(Vec3::ZERO, Vec3::X, f32::INFINITY, &mut |_, _, m| {
            calls += 1;
            m
        });
        assert_eq!(calls, 0);
    }

    #[test]
    fn proxy_ids_stay_put_while_live_and_come_back_after_removal() {
        let mut tree = DynamicTree::new();
        let a = tree.insert(cube(Vec3::ZERO, 1.0), 10u32);
        let b = tree.insert(cube(Vec3::X * 5.0, 1.0), 11);
        let c = tree.insert(cube(Vec3::Z * 5.0, 1.0), 12);
        assert_eq!((tree.len(), tree.height()), (3, 2));
        // An update moves the box and keeps the id and the data.
        tree.update(b, cube(Vec3::Y * 9.0, 0.5));
        assert_eq!((tree.aabb(b), tree.data(b)), (cube(Vec3::Y * 9.0, 0.5), 11));
        assert_eq!(found(&tree, &cube(Vec3::Y * 9.0, 0.1)), vec![b]);
        assert!(found(&tree, &cube(Vec3::X * 5.0, 0.1)).is_empty());
        // The next insertion gets a removed id back; the others keep theirs.
        tree.remove(a);
        assert_eq!(tree.len(), 2);
        let d = tree.insert(cube(Vec3::splat(-4.0), 1.0), 13);
        assert_eq!(d, a);
        assert_eq!((tree.data(b), tree.data(c), tree.data(d)), (11, 12, 13));
        assert_eq!(found(&tree, &cube(Vec3::splat(-4.0), 0.1)), vec![d]);
        tree.validate().unwrap();
        // Removing everything leaves a valid empty tree whose pool is reused, not grown.
        for p in [b, c, d] {
            tree.remove(p);
            tree.validate().unwrap();
        }
        assert!(tree.is_empty());
        let pool = tree.nodes.len();
        for k in 0..3 {
            tree.insert(cube(Vec3::X * k as f32, 1.0), k);
        }
        assert_eq!(tree.nodes.len(), pool);
        tree.validate().unwrap();
    }

    #[test]
    #[should_panic(expected = "is not live")]
    fn a_removed_proxy_cannot_be_read() {
        let mut tree = DynamicTree::new();
        let a = tree.insert(cube(Vec3::ZERO, 1.0), 1u32);
        tree.insert(cube(Vec3::X, 1.0), 2);
        tree.remove(a);
        tree.data(a);
    }

    #[test]
    #[should_panic(expected = "is not live")]
    fn an_internal_node_is_not_a_proxy() {
        let mut tree = DynamicTree::new();
        tree.insert(cube(Vec3::ZERO, 1.0), 1u32);
        tree.insert(cube(Vec3::X, 1.0), 2);
        // Ids 0 and 1 are the leaves; 2 is their parent.
        tree.remove(2);
    }

    #[test]
    fn queries_stop_when_the_callback_says_so() {
        let mut tree = DynamicTree::new();
        for k in 0..50u32 {
            tree.insert(cube(Vec3::X * k as f32, 0.75), k);
        }
        let mut seen = vec![];
        tree.query(&cube(Vec3::X * 25.0, 100.0), &mut |p, _| {
            seen.push(p);
            seen.len() < 3
        });
        assert_eq!(seen.len(), 3);
        // Boxes that only touch overlap, as with `Aabb::overlaps`.
        let mut touching = vec![];
        let corner = Aabb::new(Vec3::new(10.75, 0.75, 0.75), Vec3::splat(11.0));
        tree.query(&corner, &mut |_, d| {
            touching.push(d);
            true
        });
        touching.sort_unstable();
        assert_eq!(touching, vec![10, 11]);
    }

    #[test]
    fn rays_walk_nearer_boxes_first_and_clip_or_stop_as_told() {
        // A row of boxes along +X centred at x = 2, 4, ..., 20, inserted out of order.
        let mut tree = DynamicTree::new();
        let mut ids = [0u32; 10];
        for k in [7u32, 2, 9, 0, 5, 3, 8, 1, 6, 4] {
            ids[k as usize] = tree.insert(cube(Vec3::X * (2.0 * k as f32 + 2.0), 0.5), k);
        }
        let walk = |origin: Vec3, dir: Vec3, max_t: f32, clip: bool| {
            let inv = crate::math::inverse_direction(dir);
            let mut seen = vec![];
            tree.raycast(origin, dir, max_t, &mut |p, d, m| {
                assert_eq!(p, ids[d as usize]);
                seen.push(d);
                if clip {
                    tree.aabb(p).ray_entry(origin, inv, m).unwrap()
                } else {
                    m
                }
            });
            seen
        };
        assert_eq!(
            walk(Vec3::ZERO, Vec3::X, 100.0, false),
            (0..10).collect::<Vec<_>>()
        );
        assert_eq!(
            walk(Vec3::X * 30.0, -Vec3::X, 100.0, false),
            (0..10).rev().collect::<Vec<_>>()
        );
        // Clipping at the first box's entry leaves nothing else in reach.
        assert_eq!(walk(Vec3::ZERO, Vec3::X, 100.0, true), vec![0]);
        // The reach ends inside the fourth box.
        assert_eq!(walk(Vec3::ZERO, Vec3::X, 7.6, false), vec![0, 1, 2, 3]);
        // Starting inside a box enters it at once; passing beside the row enters none.
        assert_eq!(walk(Vec3::X * 8.0, Vec3::X, 2.0, false), vec![3, 4]);
        assert!(walk(Vec3::new(0.0, 0.6, 0.0), Vec3::X, 100.0, false).is_empty());
        // A negative return stops the walk at once.
        let mut calls = 0;
        tree.raycast(Vec3::ZERO, Vec3::X, 100.0, &mut |_, _, _| {
            calls += 1;
            -1.0
        });
        assert_eq!(calls, 1);
        // A larger or NaN return does not lengthen the walk.
        let mut seen = 0;
        tree.raycast(Vec3::ZERO, Vec3::X, 3.0, &mut |_, _, m| {
            seen += 1;
            assert_eq!(m, 3.0);
            if seen == 1 { f32::NAN } else { 1e9 }
        });
        assert_eq!(seen, 1);
        let mut none = 0;
        tree.raycast(Vec3::ZERO, Vec3::X, f32::NAN, &mut |_, _, m| {
            none += 1;
            m
        });
        tree.raycast(Vec3::ZERO, Vec3::X, -1.0, &mut |_, _, m| {
            none += 1;
            m
        });
        assert_eq!(none, 0, "no reach, no walk");
    }

    #[test]
    fn degenerate_and_adversarial_boxes_keep_the_tree_within_budget_and_exact() {
        let point = |p: Vec3| Aabb::new(p, p);
        let cases: Vec<(&str, Vec<Aabb>)> = vec![
            (
                "points on a line",
                (0..300).map(|k| point(Vec3::X * k as f32)).collect(),
            ),
            ("one point", vec![point(Vec3::ONE); 300]),
            ("identical boxes", vec![cube(Vec3::ONE, 1.0); 300]),
            (
                "nested, growing",
                (1..300).map(|k| cube(Vec3::ZERO, k as f32)).collect(),
            ),
            (
                "nested, shrinking",
                (1..300).rev().map(|k| cube(Vec3::ZERO, k as f32)).collect(),
            ),
            (
                "a sorted row",
                (0..300)
                    .map(|k| cube(Vec3::X * (1.5 * k as f32), 1.0))
                    .collect(),
            ),
            (
                "flat tiles",
                (0..300)
                    .map(|k| {
                        let c = Vec3::new((k % 20) as f32, 0.0, (k / 20) as f32);
                        Aabb::from_center_half(c, Vec3::new(0.5, 0.0, 0.5))
                    })
                    .collect(),
            ),
            (
                "tiny and huge",
                (0..300)
                    .map(|k| {
                        let half = if k % 2 == 0 { 1e-4 } else { 1e5 };
                        cube(Vec3::new(k as f32, -(k as f32), 0.0), half)
                    })
                    .collect(),
            ),
            (
                "inside out",
                (0..300)
                    .map(|k| {
                        let c = Vec3::X * k as f32;
                        Aabb::new(c + Vec3::ONE, c - Vec3::ONE)
                    })
                    .collect(),
            ),
        ];
        for (name, boxes) in cases {
            let mut tree = DynamicTree::new();
            let mut ids = vec![];
            for (k, b) in boxes.iter().enumerate() {
                ids.push(tree.insert(*b, k as u32));
                tree.validate()
                    .unwrap_or_else(|e| panic!("{name}, {k} boxes: {e}"));
            }
            // Nested and identical boxes would chain to 300 levels by surface area alone.
            assert!(
                tree.height() <= insertion_budget(boxes.len()),
                "{name}: height {}",
                tree.height()
            );
            // Move every other box onto the next one, drop every third.
            for k in (0..boxes.len()).step_by(2) {
                tree.update(ids[k], boxes[(k + 1) % boxes.len()]);
            }
            for k in (0..boxes.len()).step_by(3) {
                tree.remove(ids[k]);
            }
            tree.validate()
                .unwrap_or_else(|e| panic!("{name}, churned: {e}"));
            assert!(
                tree.height() <= height_limit(tree.len()),
                "{name}: height {}",
                tree.height()
            );
            for q in [
                cube(Vec3::ZERO, 0.5),
                cube(Vec3::new(100.0, 0.0, 3.0), 2.0),
                cube(Vec3::ONE, 0.0),
                cube(Vec3::ZERO, 1e6),
            ] {
                let mut got = found(&tree, &q);
                got.sort_unstable();
                let mut want: Vec<u32> = (0..boxes.len())
                    .filter(|k| k % 3 != 0)
                    .filter(|&k| tree.aabb(ids[k]).overlaps(&q))
                    .map(|k| ids[k])
                    .collect();
                want.sort_unstable();
                assert_eq!(got, want, "{name}: query {q:?}");
            }
        }
    }

    #[test]
    fn far_apart_clusters_meet_only_near_the_root() {
        // Two clusters of 128 small boxes 1 km apart, inserted alternately: the surface-area
        // choices keep them in separate subtrees, so few nodes have boxes spanning the gap.
        let mut tree = DynamicTree::new();
        let mut seed = 1u32;
        let mut jitter = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / 16_777_216.0 * 20.0
        };
        for k in 0..256u32 {
            let base = if k % 2 == 0 { 0.0 } else { 1000.0 };
            let c = Vec3::new(base + jitter(), jitter(), jitter());
            tree.insert(cube(c, 0.5), k);
        }
        tree.validate().unwrap();
        let spanning = tree
            .nodes
            .iter()
            .filter(|n| n.height != FREE && n.height > 0)
            .filter(|n| n.aabb.min.x < 500.0 && n.aabb.max.x > 500.0)
            .count();
        assert!(spanning <= 3, "{spanning} nodes span both clusters");
        assert!(tree.height() <= insertion_budget(256));
    }

    #[test]
    fn the_height_limit_stays_below_three_log2_n_and_above_a_balanced_rebuild() {
        for n in 2..100_000usize {
            let log2 = (n as f64).log2();
            assert!(
                f64::from(height_limit(n)).min((n - 1) as f64) < 3.0 * log2,
                "{n}"
            );
            assert!(
                insertion_budget(n) as usize >= bit_length(n) as usize,
                "{n}"
            );
        }
        // The worst height of a rebuild (each split leaves at most two thirds on a side) fits
        // in the insertion budget, so a rebuilt tree has room to grow.
        let mut worst = vec![0u32; 100_000];
        for m in 2..worst.len() {
            worst[m] = 1 + worst[m - m.div_ceil(3)];
            assert!(worst[m] < insertion_budget(m), "{m}: {}", worst[m]);
        }
    }

    #[test]
    fn removals_that_leave_the_tree_too_tall_rebuild_it_and_keep_every_proxy() {
        // Nested boxes fill the insertion budget. Keeping the deepest path (one leaf under each
        // of its side branches) while removing every other proxy leaves that path standing over
        // fewer and fewer leaves, until a removal finds the tree over its limit and rebuilds it.
        let mut tree = DynamicTree::new();
        let ids: Vec<u32> = (1..=1024u32)
            .map(|k| tree.insert(cube(Vec3::ZERO, k as f32), k))
            .collect();
        tree.validate().unwrap();
        // Far taller than a balanced tree's 10 levels, yet within the budget.
        let tall = tree.height();
        assert!(
            (18..=insertion_budget(1024)).contains(&tall),
            "height {tall}"
        );
        let first_leaf = |tree: &DynamicTree<u32>, mut i: u32| {
            while tree.nodes[i as usize].height > 0 {
                i = tree.nodes[i as usize].children()[0];
            }
            i
        };
        let mut keep = vec![];
        let mut i = tree.root;
        while tree.nodes[i as usize].height > 0 {
            let [a, b] = tree.nodes[i as usize].children();
            let (deep, side) = if tree.nodes[a as usize].height >= tree.nodes[b as usize].height {
                (a, b)
            } else {
                (b, a)
            };
            keep.push(first_leaf(&tree, side));
            i = deep;
        }
        keep.push(i);
        let mut rebuilt = false;
        for &p in ids.iter().filter(|p| !keep.contains(p)) {
            let before = tree.height();
            tree.remove(p);
            tree.validate().unwrap();
            // Without a rebuild a removal lowers the height by one at most.
            rebuilt |= tree.height() + 1 < before;
        }
        assert!(rebuilt, "some removal rebuilt the tree");
        assert_eq!(tree.len(), keep.len());
        for &p in &keep {
            let k = tree.data(p);
            assert_eq!(tree.aabb(p), cube(Vec3::ZERO, k as f32));
        }
        let mut seen = found(&tree, &cube(Vec3::ZERO, 0.1));
        seen.sort_unstable();
        keep.sort_unstable();
        assert_eq!(seen, keep);
    }

    #[test]
    fn the_walk_stack_spills_to_the_heap_and_stays_last_in_first_out() {
        let mut stack = WalkStack::<u32, 4>::new(0);
        for k in 0..10 {
            stack.push(k);
        }
        assert_eq!(stack.pop(), Some(9));
        stack.push(42);
        let mut rest = vec![];
        while let Some(k) = stack.pop() {
            rest.push(k);
        }
        assert_eq!(rest, vec![42, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
        stack.push(1);
        assert_eq!((stack.pop(), stack.pop()), (Some(1), None));
    }
}
