//! The dynamic tree against brute force: 2,000 proxies inserted, removed, updated and churned at
//! random (from a fixed seed), with every phase's box queries and rays checked against a plain
//! list of the live proxies, and the same calls always giving the same tree and visit order.

use std::collections::{BTreeMap, BTreeSet};

use ganbate_physics::math::{Aabb, Vec3, inverse_direction};
use ganbate_physics::tree::DynamicTree;

/// SplitMix64, for repeatable test data.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 * (1.0 / 16_777_216.0)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn point(&mut self, lo: f32, hi: f32) -> Vec3 {
        Vec3::new(self.range(lo, hi), self.range(lo, hi), self.range(lo, hi))
    }
    fn direction(&mut self) -> Vec3 {
        if self.below(4) == 0 {
            // Along an axis, where the slab test meets infinite inverse components.
            let axes = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
            return axes[self.below(6)];
        }
        loop {
            let v = self.point(-1.0, 1.0);
            if (1e-4..=1.0).contains(&v.length_squared()) {
                return v.normalize_or_zero();
            }
        }
    }
    /// A collider-like box in a 200 m cube: mostly small, sometimes a point, flat or large.
    fn aabb(&mut self) -> Aabb {
        let c = self.point(-100.0, 100.0);
        let half = match self.below(20) {
            0 => Vec3::ZERO,
            1 => Vec3::new(self.range(0.0, 3.0), 0.0, self.range(0.0, 3.0)),
            2 => Vec3::splat(self.range(10.0, 40.0)),
            _ => Vec3::new(
                self.range(0.05, 3.0),
                self.range(0.05, 3.0),
                self.range(0.05, 3.0),
            ),
        };
        Aabb::from_center_half(c, half)
    }
}

/// What the world would remember of each live proxy: its box and data.
type Live = BTreeMap<u32, (Aabb, u32)>;

/// Validates the tree, bounds its height and compares 150 box queries and 150 rays with brute
/// force over `live`.
fn check(tree: &DynamicTree<u32>, live: &Live, rng: &mut Rng, phase: &str) {
    tree.validate().unwrap_or_else(|e| panic!("{phase}: {e}"));
    assert_eq!(tree.len(), live.len(), "{phase}");
    let n = live.len().max(2) as f32;
    assert!(
        (tree.height() as f32) < 3.0 * n.log2(),
        "{phase}: height {} for {} proxies",
        tree.height(),
        live.len()
    );
    for (&p, &(b, d)) in live {
        assert_eq!((tree.aabb(p), tree.data(p)), (b, d), "{phase}: proxy {p}");
    }
    for _ in 0..150 {
        let mut q = rng.aabb();
        if rng.below(3) == 0 {
            q = q.expand(rng.range(0.0, 15.0));
        }
        let mut got = vec![];
        tree.query(&q, &mut |p, d| {
            got.push((p, d));
            true
        });
        got.sort_unstable();
        let want: Vec<(u32, u32)> = live
            .iter()
            .filter(|(_, (b, _))| b.overlaps(&q))
            .map(|(&p, &(_, d))| (p, d))
            .collect();
        assert_eq!(got, want, "{phase}: query {q:?}");
    }
    for _ in 0..150 {
        let origin = rng.point(-130.0, 130.0);
        let dir = rng.direction();
        let max_t = match rng.below(4) {
            0 => f32::INFINITY,
            _ => rng.range(0.0, 300.0),
        };
        let inv = inverse_direction(dir);
        let entered: Vec<(f32, u32)> = live
            .iter()
            .filter_map(|(&p, (b, _))| b.ray_entry(origin, inv, max_t).map(|t| (t, p)))
            .collect();
        // Without clipping, the walk visits exactly the boxes the ray enters within reach.
        let mut got = vec![];
        tree.raycast(origin, dir, max_t, &mut |p, d, m| {
            assert_eq!((live[&p].1, m), (d, max_t));
            got.push(p);
            m
        });
        got.sort_unstable();
        let want: Vec<u32> = entered.iter().map(|&(_, p)| p).collect();
        assert_eq!(got, want, "{phase}: ray {origin:?} {dir:?} {max_t}");
        // Clipping at each box's entry finds the nearest box.
        let mut nearest: Option<f32> = None;
        tree.raycast(origin, dir, max_t, &mut |p, _, m| {
            let t = live[&p]
                .0
                .ray_entry(origin, inv, m)
                .expect("a visited box is entered within the current reach");
            nearest = Some(nearest.map_or(t, |n| n.min(t)));
            t
        });
        let want = entered.iter().map(|&(t, _)| t).reduce(f32::min);
        assert_eq!(nearest, want, "{phase}: nearest along {origin:?} {dir:?}");
    }
}

#[test]
fn two_thousand_random_proxies_match_brute_force_through_inserts_removes_and_updates() {
    let mut rng = Rng(2024);
    let mut tree = DynamicTree::new();
    let mut live = Live::new();
    let mut next = 0u32;
    let mut insert = |tree: &mut DynamicTree<u32>, live: &mut Live, b: Aabb| {
        let p = tree.insert(b, next);
        assert!(
            live.insert(p, (b, next)).is_none(),
            "id {p} is live already"
        );
        next += 1;
        p
    };

    for _ in 0..2000 {
        let b = rng.aabb();
        insert(&mut tree, &mut live, b);
    }
    check(&tree, &live, &mut rng, "inserted");
    let pool = 2 * 2000 - 1;
    assert!(live.keys().all(|&p| p < pool));

    let mut removed = BTreeSet::new();
    while removed.len() < 1000 {
        let &p = live.keys().nth(rng.below(live.len())).unwrap();
        tree.remove(p);
        live.remove(&p);
        removed.insert(p);
    }
    check(&tree, &live, &mut rng, "removed");

    // Most survivors move a little, as bodies do between steps; some jump anywhere.
    let ids: Vec<u32> = live.keys().copied().collect();
    for p in ids {
        let b = if rng.below(4) == 0 {
            rng.aabb()
        } else {
            let d = rng.point(-1.0, 1.0);
            let old = live[&p].0;
            Aabb::new(old.min + d, old.max + d)
        };
        tree.update(p, b);
        live.get_mut(&p).unwrap().0 = b;
    }
    check(&tree, &live, &mut rng, "updated");

    // Back to 2,000: removed ids come back into use and the node pool does not grow.
    let mut reused = 0;
    for _ in 0..1000 {
        let b = rng.aabb();
        let p = insert(&mut tree, &mut live, b);
        assert!(p < pool, "id {p} past the pool of {pool}");
        reused += usize::from(removed.contains(&p));
    }
    assert!(reused > 0);
    check(&tree, &live, &mut rng, "refilled");

    for _ in 0..4000 {
        match rng.below(3) {
            0 if live.len() < 2500 => {
                let b = rng.aabb();
                insert(&mut tree, &mut live, b);
            }
            1 if !live.is_empty() => {
                let &p = live.keys().nth(rng.below(live.len())).unwrap();
                tree.remove(p);
                live.remove(&p);
            }
            _ if !live.is_empty() => {
                let &p = live.keys().nth(rng.below(live.len())).unwrap();
                let b = rng.aabb();
                tree.update(p, b);
                live.get_mut(&p).unwrap().0 = b;
            }
            _ => {}
        }
    }
    check(&tree, &live, &mut rng, "churned");

    let ids: Vec<u32> = live.keys().copied().collect();
    for p in ids {
        tree.remove(p);
        live.remove(&p);
    }
    check(&tree, &live, &mut rng, "emptied");
    assert!(tree.is_empty() && tree.height() == 0);
}

/// 1,500 inserts, removes and updates from a fixed seed.
fn scripted_tree() -> DynamicTree<u32> {
    let mut rng = Rng(77);
    let mut tree = DynamicTree::new();
    let mut ids = vec![];
    for k in 0..1500u32 {
        match rng.below(5) {
            0 if !ids.is_empty() => {
                let p = ids.swap_remove(rng.below(ids.len()));
                tree.remove(p);
            }
            1 if !ids.is_empty() => {
                let p = ids[rng.below(ids.len())];
                tree.update(p, rng.aabb());
            }
            _ => ids.push(tree.insert(rng.aabb(), k)),
        }
    }
    tree
}

/// The order in which a box query and a ray visit their proxies.
fn visits(tree: &DynamicTree<u32>, q: &Aabb, origin: Vec3, dir: Vec3) -> (Vec<u32>, Vec<u32>) {
    let mut boxes = vec![];
    tree.query(q, &mut |p, _| {
        boxes.push(p);
        true
    });
    let mut rays = vec![];
    tree.raycast(origin, dir, f32::INFINITY, &mut |p, _, m| {
        rays.push(p);
        m
    });
    (boxes, rays)
}

#[test]
fn identical_call_sequences_build_identical_trees_that_visit_in_identical_orders() {
    let (a, b) = (scripted_tree(), scripted_tree());
    a.validate().unwrap();
    assert_eq!(
        format!("{a:?}"),
        format!("{b:?}"),
        "the same nodes, links and boxes"
    );
    let mut rng = Rng(5);
    let mut visited = 0;
    for _ in 0..200 {
        let q = rng.aabb().expand(30.0);
        let (origin, dir) = (rng.point(-120.0, 120.0), rng.direction());
        let va = visits(&a, &q, origin, dir);
        assert_eq!(va, visits(&b, &q, origin, dir));
        assert_eq!(va, visits(&a.clone(), &q, origin, dir));
        visited += va.0.len() + va.1.len();
    }
    assert!(visited > 1000, "the walks visit proxies ({visited})");
}
