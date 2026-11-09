//! Deterministic CPU hemisphere ambient occlusion baking over static world triangle soups.
//!
//! Uses a flat binned-SAH bounding volume hierarchy (BVH) and nearest-hit distance queries so
//! baked vertex occlusion is fast and strictly independent of triangle or instance ordering.
//!
//! A world of a million triangles takes tens of seconds to bake, so the renderer never bakes it
//! in one call: [`AoBake`] samples a bounded number of vertices per frame, nearest the camera
//! first, and after an edit re-samples only the vertices within reach of what moved.

use std::collections::{HashMap, VecDeque};

use crate::world::{AlphaMode, Shading, World};

/// A cooked world's baked AO as a file (`ganbate-build bake-ao`): "GBAO", the version, the
/// [`fingerprint`] of the world it was baked for, the instance count, then per instance its
/// vertex count (0: none baked) and each vertex's AO in 65535ths, all little-endian.
const SIDECAR_MAGIC: &[u8; 4] = b"GBAO";
const SIDECAR_VERSION: u32 = 1;

/// What a sidecar must have been baked for: each instance's mesh with its vertex count and
/// bounds, where the instance stands, and whether it is shown and drawn (FNV-1a).
fn fingerprint(edit: &WorldEdit) -> u64 {
    fn put(h: &mut u64, bytes: &[u8]) {
        for b in bytes {
            *h = (*h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    let mut h = 0xcbf2_9ce4_8422_2325;
    put(&mut h, &(edit.instances.len() as u32).to_le_bytes());
    for inst in &edit.instances {
        let mesh = &edit.meshes[inst.mesh as usize];
        put(&mut h, &inst.mesh.to_le_bytes());
        put(&mut h, &(mesh.positions.len() as u32).to_le_bytes());
        let (lo, hi) = mesh.bounds;
        for v in lo.iter().chain(&hi).chain(&inst.transform) {
            put(&mut h, &v.to_le_bytes());
        }
        put(
            &mut h,
            &[
                u8::from(inst.visible),
                u8::from(inst.first_vertex.is_some()),
            ],
        );
    }
    h
}

/// Reads a sidecar front to back.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.0.len() < n {
            return Err("baked AO: the file ends early".into());
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().unwrap_or_default(),
        ))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().unwrap_or_default(),
        ))
    }
}

/// Bakes `world`'s vertex AO on the CPU to the end, as a live view converges to it, and returns
/// it as a sidecar ([`AoBake::export`]) a runtime loads instead of baking it again.
pub fn bake_world(world: &World) -> Result<Vec<u8>, String> {
    let edit = super::scene::cpu_edit(world)?;
    let mut bake = AoBake::new(&edit, AoConfig::default());
    while !bake.is_idle() {
        bake.step(&edit, usize::MAX / 128, None);
    }
    bake.export(&edit)
        .ok_or_else(|| "baked AO: the bake did not finish".into())
}

use super::math::{V3, normalize, transform_point};
use super::scene::WorldEdit;

/// `f32::min`/`max` honour IEEE `minNum` semantics, which wasm32 lowers to `fminf`/`fmaxf` calls
/// rather than its `f32.min` instruction; in the traversal loops those calls cost more than the
/// arithmetic. No operand here is NaN, so plain comparisons are exact.
#[inline(always)]
fn fmin(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

#[inline(always)]
fn fmax(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// SAH bins per axis, and leaf sizes: a node of at most `LEAF_TRIS` triangles is always a leaf,
/// one of more than `MAX_LEAF_TRIS` is always split.
const SAH_BINS: usize = 16;
const LEAF_TRIS: usize = 2;
const MAX_LEAF_TRIS: usize = 12;
/// Below this depth splits follow the surface area heuristic; deeper ones split at the median so
/// the tree (and the traversal stack) stays bounded however the triangles cluster.
const SAH_MAX_DEPTH: u32 = 40;
const STACK: usize = 128;

/// Parameters controlling hemisphere ambient occlusion baking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AoConfig {
    /// Number of cosine-weighted hemisphere rays cast per unique `(position, normal)` sample.
    pub rays: u32,
    /// Maximum occlusion reach in metres.
    pub max_distance: f32,
    /// Surface normal offset in metres preventing self-intersection on convex geometry.
    pub normal_bias: f32,
}

impl Default for AoConfig {
    fn default() -> Self {
        Self {
            rays: 16,
            max_distance: 3.5,
            normal_bias: 0.02,
        }
    }
}

#[derive(Clone, Copy)]
struct Tri {
    p0: [f32; 3],
    e1: [f32; 3],
    e2: [f32; 3],
    min: [f32; 3],
    max: [f32; 3],
    centroid: [f32; 3],
}

#[derive(Clone, Copy, Default)]
struct Node {
    min: [f32; 3],
    max: [f32; 3],
    /// Internal node (`count == 0`): left child index; right child is stored in `right_or_first`.
    /// Leaf node (`count > 0`): `right_or_first` is the start index in `tris`.
    left: u32,
    right_or_first: u32,
    count: u32,
}

/// Acceleration structure over world-space occluding triangles.
pub struct AoBvh {
    tris: Vec<Tri>,
    nodes: Vec<Node>,
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(a: [f32; 3]) -> [f32; 3] {
    let l = dot3(a, a).sqrt();
    if l > 1e-12 {
        scale3(a, 1.0 / l)
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn transform_normal(m: &[f32; 16], n: [f32; 3]) -> [f32; 3] {
    normalize3([
        m[0] * n[0] + m[4] * n[1] + m[8] * n[2],
        m[1] * n[0] + m[5] * n[1] + m[9] * n[2],
        m[2] * n[0] + m[6] * n[1] + m[10] * n[2],
    ])
}

fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

fn spatial_spin(pos: [f32; 3], normal: [f32; 3]) -> f32 {
    let qx = (pos[0] * 1024.0).round() as i32 as u32;
    let qy = (pos[1] * 1024.0).round() as i32 as u32;
    let qz = (pos[2] * 1024.0).round() as i32 as u32;
    let nx = (normal[0] * 512.0).round() as i32 as u32;
    let ny = (normal[1] * 512.0).round() as i32 as u32;
    let nz = (normal[2] * 512.0).round() as i32 as u32;
    let h = hash_u32(qx.wrapping_mul(73_856_093) ^ nx)
        ^ hash_u32(qy.wrapping_mul(19_349_663) ^ ny.wrapping_add(1))
        ^ hash_u32(qz.wrapping_mul(83_492_791) ^ nz.wrapping_add(7));
    (h >> 8) as f32 * (std::f32::consts::TAU / 16_777_216.0)
}

/// A world-space triangle as the BVH stores it, or `None` for a degenerate one.
fn make_tri(a: V3, b: V3, c: V3) -> Option<Tri> {
    let e1 = sub3(b, a);
    let e2 = sub3(c, a);
    let n = cross3(e1, e2);
    if dot3(n, n) <= 1e-16 {
        return None;
    }
    Some(Tri {
        p0: a,
        e1,
        e2,
        min: [0, 1, 2].map(|k| fmin(fmin(a[k], b[k]), c[k])),
        max: [0, 1, 2].map(|k| fmax(fmax(a[k], b[k]), c[k])),
        centroid: [0, 1, 2].map(|k| (a[k] + b[k] + c[k]) * (1.0 / 3.0)),
    })
}

/// A node still to split: triangles `start..end` of the builder's array, written to `node`.
struct BuildTask {
    node: u32,
    start: usize,
    end: usize,
    depth: u32,
}

/// A BVH under construction that splits a bounded number of triangles per [`BvhBuilder::step`],
/// so a million-triangle world never blocks a frame (or a page's input) while it is built.
pub(super) struct BvhBuilder {
    tris: Vec<Tri>,
    nodes: Vec<Node>,
    tasks: Vec<BuildTask>,
}

impl BvhBuilder {
    fn new(tris: Vec<Tri>) -> Self {
        let mut nodes = Vec::with_capacity(tris.len().saturating_mul(2).max(1));
        let mut tasks = Vec::new();
        if !tris.is_empty() {
            nodes.push(Node::default());
            tasks.push(BuildTask {
                node: 0,
                start: 0,
                end: tris.len(),
                depth: 0,
            });
        }
        Self { tris, nodes, tasks }
    }

    /// Splits nodes until about `budget` triangles have been binned; true once the tree is
    /// complete. Each node costs its triangle count, so the whole build costs ~n log n.
    fn step(&mut self, budget: usize) -> bool {
        let mut spent = 0usize;
        while spent < budget {
            let Some(task) = self.tasks.pop() else {
                return true;
            };
            spent += (task.end - task.start).max(1);
            self.split(task);
        }
        self.tasks.is_empty()
    }

    fn finish(self) -> AoBvh {
        AoBvh {
            tris: self.tris,
            nodes: self.nodes,
        }
    }

    fn split(
        &mut self,
        BuildTask {
            node,
            start,
            end,
            depth,
        }: BuildTask,
    ) {
        let tris = &mut self.tris;
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        let mut cmin = [f32::INFINITY; 3];
        let mut cmax = [f32::NEG_INFINITY; 3];
        for t in &tris[start..end] {
            for k in 0..3 {
                min[k] = fmin(min[k], t.min[k]);
                max[k] = fmax(max[k], t.max[k]);
                cmin[k] = fmin(cmin[k], t.centroid[k]);
                cmax[k] = fmax(cmax[k], t.centroid[k]);
            }
        }
        let count = end - start;
        let leaf = Node {
            min,
            max,
            left: 0,
            right_or_first: start as u32,
            count: count as u32,
        };
        let extent = [cmax[0] - cmin[0], cmax[1] - cmin[1], cmax[2] - cmin[2]];
        let widest = if extent[0] >= extent[1] && extent[0] >= extent[2] {
            0
        } else if extent[1] >= extent[2] {
            1
        } else {
            2
        };
        if count <= LEAF_TRIS || extent[widest] <= 1e-5 {
            self.nodes[node as usize] = leaf;
            return;
        }

        let split = if depth < SAH_MAX_DEPTH {
            sah_split(&tris[start..end], cmin, extent, area(min, max))
        } else {
            None
        };
        let mid = match split {
            Some(SahSplit::Leaf) if count <= MAX_LEAF_TRIS => {
                self.nodes[node as usize] = leaf;
                return;
            }
            Some(SahSplit::At { axis, bin }) => {
                let scale = SAH_BINS as f32 / extent[axis];
                let lo = cmin[axis];
                let slice = &mut tris[start..end];
                let (mut i, mut j) = (0usize, slice.len());
                while i < j {
                    if bin_of(slice[i].centroid[axis], lo, scale) < bin {
                        i += 1;
                    } else {
                        j -= 1;
                        slice.swap(i, j);
                    }
                }
                start + i
            }
            _ => start,
        };
        let mid = if mid == start || mid == end {
            // No useful SAH split: the median of the widest axis, in a total order so the set on
            // each side does not depend on the input order.
            tris[start..end].select_nth_unstable_by(count / 2, |a, b| {
                a.centroid[widest]
                    .total_cmp(&b.centroid[widest])
                    .then_with(|| a.centroid[0].total_cmp(&b.centroid[0]))
                    .then_with(|| a.centroid[1].total_cmp(&b.centroid[1]))
                    .then_with(|| a.centroid[2].total_cmp(&b.centroid[2]))
            });
            start + count / 2
        } else {
            mid
        };
        let left = self.nodes.len() as u32;
        self.nodes.push(Node::default());
        self.nodes.push(Node::default());
        self.nodes[node as usize] = Node {
            min,
            max,
            left,
            right_or_first: left + 1,
            count: 0,
        };
        self.tasks.push(BuildTask {
            node: left + 1,
            start: mid,
            end,
            depth: depth + 1,
        });
        self.tasks.push(BuildTask {
            node: left,
            start,
            end: mid,
            depth: depth + 1,
        });
    }
}

impl AoBvh {
    /// Builds a BVH over world-space triangle vertex triples.
    pub fn build(triangles: &[([f32; 3], [f32; 3], [f32; 3])]) -> Self {
        let tris = triangles
            .iter()
            .filter_map(|&(a, b, c)| make_tri(a, b, c))
            .collect();
        let mut builder = BvhBuilder::new(tris);
        while !builder.step(usize::MAX) {}
        builder.finish()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn nearest_hit(&self, origin: [f32; 3], dir: [f32; 3], t_min: f32, t_max: f32) -> Option<f32> {
        let root = self.nodes.first()?;
        let inv_dir = dir.map(|d| {
            if d.abs() > 1e-12 {
                1.0 / d
            } else if d >= 0.0 {
                1e12
            } else {
                -1e12
            }
        });
        let mut best = t_max;
        let mut hit = false;
        // Each entry carries the distance at which the ray enters the node, tested when it was
        // pushed, so a popped node is only re-checked against the nearest hit found since.
        let mut stack = [(0u32, 0.0f32); STACK];
        stack[0] = (0, ray_aabb(origin, inv_dir, root.min, root.max, best)?);
        let mut sp = 1usize;
        while sp > 0 {
            sp -= 1;
            let (index, enter) = stack[sp];
            if enter >= best {
                continue;
            }
            let node = &self.nodes[index as usize];
            if node.count > 0 {
                let first = node.right_or_first as usize;
                let end = first + node.count as usize;
                for tri in &self.tris[first..end] {
                    if let Some(t) = ray_tri(origin, dir, tri, t_min, best) {
                        best = t;
                        hit = true;
                    }
                }
                continue;
            }
            let (l, r) = (node.left, node.right_or_first);
            let (ln, rn) = (&self.nodes[l as usize], &self.nodes[r as usize]);
            let dl = ray_aabb(origin, inv_dir, ln.min, ln.max, best);
            let dr = ray_aabb(origin, inv_dir, rn.min, rn.max, best);
            // Far child first, so the near one is popped next.
            let mut push = |entry: (u32, f32)| {
                if sp < STACK {
                    stack[sp] = entry;
                    sp += 1;
                }
            };
            match (dl, dr) {
                (Some(tl), Some(tr)) if tl <= tr => {
                    push((r, tr));
                    push((l, tl));
                }
                (Some(tl), Some(tr)) => {
                    push((l, tl));
                    push((r, tr));
                }
                (Some(tl), None) => push((l, tl)),
                (None, Some(tr)) => push((r, tr)),
                (None, None) => {}
            }
        }
        hit.then_some(best)
    }

    /// Evaluates hemisphere ambient occlusion in `[0.08, 1.0]` at a world-space surface sample.
    pub fn sample_ao(&self, pos: [f32; 3], normal: [f32; 3], config: &AoConfig) -> f32 {
        let rays = config.rays.max(1);
        let max_d = config.max_distance;
        if self.nodes.is_empty() || max_d <= 1e-4 {
            return 1.0;
        }
        let n = normalize3(normal);
        let up = if n[1].abs() < 0.9 {
            [0.0, 1.0, 0.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        let t0 = normalize3(cross3(up, n));
        let b0 = cross3(n, t0);
        let spin = spatial_spin(pos, n);
        let bias = config.normal_bias.max(1e-4);
        let pullback = (bias * 0.1).min(0.003);
        let t_eps = pullback * 0.25;
        let base_origin = add3(pos, scale3(n, bias));
        const GOLDEN_ANGLE: f32 = 2.399_963_1;

        let mut occlusion = 0.0f32;
        for i in 0..rays {
            let u = (i as f32 + 0.5) / rays as f32;
            let z = (0.18 + 0.82 * u).sqrt();
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = i as f32 * GOLDEN_ANGLE + spin;
            let (s, c) = phi.sin_cos();
            let dir = add3(add3(scale3(t0, r * c), scale3(b0, r * s)), scale3(n, z));
            let origin = sub3(base_origin, scale3(dir, pullback));
            if let Some(hit_t) = self.nearest_hit(origin, dir, t_eps, max_d + pullback) {
                let dist = (hit_t - pullback).max(0.0);
                if dist < max_d {
                    let falloff = 1.0 - dist / max_d;
                    occlusion += falloff * falloff;
                }
            }
        }
        (1.0 - occlusion / rays as f32).clamp(0.08, 1.0)
    }

    /// Evaluates hemisphere AO for a batch of `(position, normal)` samples, deduplicating shared
    /// seam vertices.
    pub fn sample_batch(
        &self,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        config: &AoConfig,
    ) -> Vec<f32> {
        if self.nodes.is_empty() {
            return vec![1.0; positions.len()];
        }
        let mut cache: HashMap<([i32; 3], [i16; 3]), f32> = HashMap::with_capacity(positions.len());
        positions
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let n = normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);
                let n = normalize3(n);
                let key = (
                    [
                        (p[0] * 2048.0).round() as i32,
                        (p[1] * 2048.0).round() as i32,
                        (p[2] * 2048.0).round() as i32,
                    ],
                    [
                        (n[0] * 1024.0).round() as i16,
                        (n[1] * 1024.0).round() as i16,
                        (n[2] * 1024.0).round() as i16,
                    ],
                );
                if let Some(&ao) = cache.get(&key) {
                    ao
                } else {
                    let ao = self.sample_ao(p, n, config);
                    cache.insert(key, ao);
                    ao
                }
            })
            .collect()
    }
}

enum SahSplit {
    /// Splitting costs more than intersecting every triangle of the node.
    Leaf,
    /// Triangles whose centroid falls in a bin below `bin` on `axis` go left.
    At { axis: usize, bin: usize },
}

fn area(min: [f32; 3], max: [f32; 3]) -> f32 {
    let d = [
        fmax(max[0] - min[0], 0.0),
        fmax(max[1] - min[1], 0.0),
        fmax(max[2] - min[2], 0.0),
    ];
    2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
}

#[inline(always)]
fn bin_of(c: f32, lo: f32, scale: f32) -> usize {
    (((c - lo) * scale) as usize).min(SAH_BINS - 1)
}

/// The cheapest binned surface-area-heuristic split of `tris` over all three axes, with
/// traversal and intersection weighted equally; `None` when every centroid shares one bin.
fn sah_split(tris: &[Tri], cmin: [f32; 3], extent: [f32; 3], node_area: f32) -> Option<SahSplit> {
    #[derive(Clone, Copy)]
    struct Bin {
        count: u32,
        min: [f32; 3],
        max: [f32; 3],
    }
    const EMPTY: Bin = Bin {
        count: 0,
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };
    let grow = |b: &mut Bin, min: [f32; 3], max: [f32; 3], count: u32| {
        b.count += count;
        for k in 0..3 {
            b.min[k] = fmin(b.min[k], min[k]);
            b.max[k] = fmax(b.max[k], max[k]);
        }
    };
    let mut best: Option<(f32, usize, usize)> = None;
    for axis in 0..3 {
        if extent[axis] <= 1e-6 {
            continue;
        }
        let scale = SAH_BINS as f32 / extent[axis];
        let mut bins = [EMPTY; SAH_BINS];
        for t in tris {
            grow(
                &mut bins[bin_of(t.centroid[axis], cmin[axis], scale)],
                t.min,
                t.max,
                1,
            );
        }
        // right[i]: everything in bins i.. ; the split before bin i costs left(..i) + right(i..).
        let mut right = [(0u32, 0.0f32); SAH_BINS];
        let mut acc = EMPTY;
        for i in (1..SAH_BINS).rev() {
            let b = bins[i];
            grow(&mut acc, b.min, b.max, b.count);
            right[i] = (acc.count, area(acc.min, acc.max));
        }
        let mut acc = EMPTY;
        for i in 1..SAH_BINS {
            let b = bins[i - 1];
            grow(&mut acc, b.min, b.max, b.count);
            let (rc, ra) = right[i];
            if acc.count == 0 || rc == 0 {
                continue;
            }
            let cost = acc.count as f32 * area(acc.min, acc.max) + rc as f32 * ra;
            if best.is_none_or(|(c, _, _)| cost < c) {
                best = Some((cost, axis, i));
            }
        }
    }
    let (cost, axis, bin) = best?;
    Some(if node_area + cost >= tris.len() as f32 * node_area {
        SahSplit::Leaf
    } else {
        SahSplit::At { axis, bin }
    })
}

#[inline(always)]
fn ray_aabb(
    origin: [f32; 3],
    inv_dir: [f32; 3],
    min: [f32; 3],
    max: [f32; 3],
    t_max: f32,
) -> Option<f32> {
    let mut t0 = 0.0f32;
    let mut t1 = t_max;
    for k in 0..3 {
        let a = (min[k] - origin[k]) * inv_dir[k];
        let b = (max[k] - origin[k]) * inv_dir[k];
        t0 = fmax(t0, fmin(a, b));
        t1 = fmin(t1, fmax(a, b));
        if t1 < t0 {
            return None;
        }
    }
    Some(t0)
}

fn ray_tri(origin: [f32; 3], dir: [f32; 3], tri: &Tri, t_min: f32, t_max: f32) -> Option<f32> {
    let h = cross3(dir, tri.e2);
    let a = dot3(tri.e1, h);
    if a.abs() < 1e-8 {
        return None;
    }
    let f = 1.0 / a;
    let s = sub3(origin, tri.p0);
    let u = f * dot3(s, h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross3(s, tri.e1);
    let v = f * dot3(dir, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * dot3(tri.e2, q);
    (t > t_min && t < t_max).then_some(t)
}

/// Bakes hemisphere ambient occlusion for `query_positions` / `query_normals` against an indexed
/// occluder triangle mesh.
pub fn bake_vertex_ao(
    occluder_positions: &[[f32; 3]],
    occluder_indices: &[u32],
    query_positions: &[[f32; 3]],
    query_normals: &[[f32; 3]],
    config: &AoConfig,
) -> Vec<f32> {
    let mut tris = Vec::with_capacity(occluder_indices.len() / 3);
    for tri in occluder_indices.chunks_exact(3) {
        let (Some(&a), Some(&b), Some(&c)) = (
            occluder_positions.get(tri[0] as usize),
            occluder_positions.get(tri[1] as usize),
            occluder_positions.get(tri[2] as usize),
        ) else {
            continue;
        };
        tris.push((a, b, c));
    }
    let bvh = AoBvh::build(&tris);
    bvh.sample_batch(query_positions, query_normals, config)
}

/// Bakes per-instance vertex AO across a cooked [`World`]. Water, foliage, unlit, and
/// non-opaque surfaces neither cast nor receive baked solid-surface occlusion.
pub fn bake_world_ao(world: &World, config: &AoConfig) -> Vec<Vec<f32>> {
    let is_occluder_mesh = |mesh_idx: usize| -> bool {
        let Some(mesh) = world.meshes.get(mesh_idx) else {
            return false;
        };
        match mesh.material.and_then(|m| world.materials.get(m as usize)) {
            Some(mat) => mat.shading == Shading::Standard && mat.alpha == AlphaMode::Opaque,
            None => true,
        }
    };
    let is_receiver_mesh = |mesh_idx: usize| -> bool {
        let Some(mesh) = world.meshes.get(mesh_idx) else {
            return false;
        };
        match mesh.material.and_then(|m| world.materials.get(m as usize)) {
            Some(mat) => mat.shading == Shading::Standard,
            None => true,
        }
    };

    let mut tris = Vec::new();
    for inst in &world.instances {
        let mesh_idx = inst.mesh as usize;
        if !is_occluder_mesh(mesh_idx) {
            continue;
        }
        let mesh = &world.meshes[mesh_idx];
        let m = &inst.transform.0;
        for t in mesh.indices.chunks_exact(3) {
            let a = mesh.positions[t[0] as usize];
            let b = mesh.positions[t[1] as usize];
            let c = mesh.positions[t[2] as usize];
            tris.push((
                transform_point(m, [a.0, a.1, a.2]),
                transform_point(m, [b.0, b.1, b.2]),
                transform_point(m, [c.0, c.1, c.2]),
            ));
        }
    }
    let bvh = AoBvh::build(&tris);
    let mut cache: HashMap<([i32; 3], [i16; 3]), f32> = HashMap::new();
    let mut out = Vec::with_capacity(world.instances.len());
    for inst in &world.instances {
        let mesh_idx = inst.mesh as usize;
        let Some(mesh) = world.meshes.get(mesh_idx) else {
            out.push(Vec::new());
            continue;
        };
        if mesh.indices.is_empty() || !is_receiver_mesh(mesh_idx) || bvh.is_empty() {
            out.push(vec![1.0; mesh.positions.len()]);
            continue;
        }
        let m = &inst.transform.0;
        let mut inst_ao = Vec::with_capacity(mesh.positions.len());
        for (vi, p) in mesh.positions.iter().enumerate() {
            let wp = transform_point(m, [p.0, p.1, p.2]);
            let wn = mesh
                .normals
                .get(vi)
                .map_or([0.0, 1.0, 0.0], |n| transform_normal(m, [n.0, n.1, n.2]));
            let key = (
                [
                    (wp[0] * 2048.0).round() as i32,
                    (wp[1] * 2048.0).round() as i32,
                    (wp[2] * 2048.0).round() as i32,
                ],
                [
                    (wn[0] * 1024.0).round() as i16,
                    (wn[1] * 1024.0).round() as i16,
                    (wn[2] * 1024.0).round() as i16,
                ],
            );
            let ao = if let Some(&cached) = cache.get(&key) {
                cached
            } else {
                let v = bvh.sample_ao(wp, wn, config);
                cache.insert(key, v);
                v
            };
            inst_ao.push(ao);
        }
        out.push(inst_ao);
    }
    out
}

type SampleKey = ([i32; 3], [i16; 3]);

/// Seam vertices sharing a position (to 1/2048 m) and normal (to 1/1024) share one sample.
fn sample_key(p: V3, n: V3) -> SampleKey {
    (
        [
            (p[0] * 2048.0).round() as i32,
            (p[1] * 2048.0).round() as i32,
            (p[2] * 2048.0).round() as i32,
        ],
        [
            (n[0] * 1024.0).round() as i16,
            (n[1] * 1024.0).round() as i16,
            (n[2] * 1024.0).round() as i16,
        ],
    )
}

fn world_normal(m: &[f32; 16], n: V3) -> V3 {
    normalize([
        m[0] * n[0] + m[4] * n[1] + m[8] * n[2],
        m[1] * n[0] + m[5] * n[1] + m[9] * n[2],
        m[2] * n[0] + m[6] * n[1] + m[10] * n[2],
    ])
}

/// Axis-aligned world bounds of local bounds `b` under `m` (the box of its eight corners).
pub(super) fn transformed_bounds(b: (V3, V3), m: &[f32; 16]) -> (V3, V3) {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for corner in 0..8 {
        let p = [
            if corner & 1 == 0 { b.0[0] } else { b.1[0] },
            if corner & 2 == 0 { b.0[1] } else { b.1[1] },
            if corner & 4 == 0 { b.0[2] } else { b.1[2] },
        ];
        let w = transform_point(m, p);
        for k in 0..3 {
            lo[k] = fmin(lo[k], w[k]);
            hi[k] = fmax(hi[k], w[k]);
        }
    }
    (lo, hi)
}

fn grown(b: (V3, V3), by: f32) -> (V3, V3) {
    (b.0.map(|v| v - by), b.1.map(|v| v + by))
}

fn merged(a: (V3, V3), b: (V3, V3)) -> (V3, V3) {
    (
        [
            fmin(a.0[0], b.0[0]),
            fmin(a.0[1], b.0[1]),
            fmin(a.0[2], b.0[2]),
        ],
        [
            fmax(a.1[0], b.1[0]),
            fmax(a.1[1], b.1[1]),
            fmax(a.1[2], b.1[2]),
        ],
    )
}

fn overlaps(a: (V3, V3), b: (V3, V3)) -> bool {
    (0..3).all(|k| a.0[k] <= b.1[k] && b.0[k] <= a.1[k])
}

fn contains(b: (V3, V3), p: V3) -> bool {
    (0..3).all(|k| b.0[k] <= p[k] && p[k] <= b.1[k])
}

/// Occluder work one AO sample's budget buys: a sample casts 16 rays, about the cost of reading
/// or binning a hundred-odd triangles.
const BUILD_WORK_PER_SAMPLE: usize = 128;

/// The occluders a bake samples against, read and built a slice at a time.
enum Occluders {
    /// Reading triangles: the next instance and triangle, and those read so far.
    Collecting {
        instance: usize,
        triangle: usize,
        tris: Vec<Tri>,
    },
    Building(BvhBuilder),
    Ready(AoBvh),
}

/// A receiver instance waiting to be (re)sampled: every vertex, or only those inside `region`.
#[derive(Clone, Copy)]
struct Pending {
    instance: u32,
    region: Option<(V3, V3)>,
}

/// Incremental vertex-AO bake over a [`WorldEdit`]. [`AoBake::step`] samples a bounded number
/// of vertices, so a renderer can spread a full-world bake over frames and stay interactive;
/// [`AoBake::invalidate_region`] re-queues only the vertices an edit can have changed.
pub(super) struct AoBake {
    config: AoConfig,
    /// Per instance: the AO its vertices carry (empty until first baked: 1.0 everywhere).
    values: Vec<Vec<f32>>,
    queue: VecDeque<Pending>,
    /// Vertices of the front entry already scanned.
    cursor: usize,
    /// Occluders the BVH holds: all of them (`None`), or those reaching into this box.
    scope: Option<(V3, V3)>,
    occluders: Option<Occluders>,
    cache: HashMap<SampleKey, f32>,
    /// The queue was refilled since it was last sorted by distance to the camera.
    unordered: bool,
    queued_vertices: usize,
    cycle_vertices: usize,
}

impl AoBake {
    /// A bake with every receiver of `edit` queued.
    pub fn new(edit: &WorldEdit, config: AoConfig) -> Self {
        let mut bake = Self {
            config,
            values: vec![Vec::new(); edit.instances.len()],
            queue: VecDeque::new(),
            cursor: 0,
            scope: None,
            occluders: None,
            cache: HashMap::new(),
            unordered: false,
            queued_vertices: 0,
            cycle_vertices: 0,
        };
        for index in 0..edit.instances.len() {
            bake.enqueue(edit, index, None);
        }
        bake
    }

    /// The AO instance `index`'s vertices carry now (`None`: unbaked, 1.0 everywhere).
    pub fn values(&self, index: usize) -> Option<&[f32]> {
        self.values
            .get(index)
            .filter(|v| !v.is_empty())
            .map(Vec::as_slice)
    }

    pub fn is_idle(&self) -> bool {
        self.queue.is_empty()
    }

    /// The finished bake as a sidecar for this world (see [`SIDECAR_MAGIC`]), or `None` while
    /// it is under way.
    pub fn export(&self, edit: &WorldEdit) -> Option<Vec<u8>> {
        if !self.is_idle() {
            return None;
        }
        let mut out = Vec::new();
        out.extend_from_slice(SIDECAR_MAGIC);
        out.extend_from_slice(&SIDECAR_VERSION.to_le_bytes());
        out.extend_from_slice(&fingerprint(edit).to_le_bytes());
        out.extend_from_slice(&(edit.instances.len() as u32).to_le_bytes());
        for index in 0..edit.instances.len() {
            let values = self.values(index).unwrap_or_default();
            out.extend_from_slice(&(values.len() as u32).to_le_bytes());
            for v in values {
                let q = (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
                out.extend_from_slice(&q.to_le_bytes());
            }
        }
        Some(out)
    }

    /// Takes a finished bake from a sidecar [`AoBake::export`] made for this world: nothing is
    /// left to bake (later edits re-queue what they reach, as usual). Fails, changing nothing,
    /// when it was baked for another world or this one has changed since it loaded.
    pub fn import(&mut self, edit: &WorldEdit, data: &[u8]) -> Result<(), String> {
        let mut r = Reader(data);
        if r.take(4)? != SIDECAR_MAGIC {
            return Err("baked AO: not a baked-AO file".into());
        }
        let version = r.u32()?;
        if version != SIDECAR_VERSION {
            return Err(format!(
                "baked AO: version {version}, not {SIDECAR_VERSION}"
            ));
        }
        if r.u64()? != fingerprint(edit) || r.u32()? as usize != edit.instances.len() {
            return Err("baked AO: baked for another world".into());
        }
        let mut values = Vec::with_capacity(edit.instances.len());
        for inst in &edit.instances {
            let n = r.u32()? as usize;
            if n != 0 && n != edit.meshes[inst.mesh as usize].positions.len() {
                return Err("baked AO: baked for another world".into());
            }
            let v: Vec<f32> = r
                .take(n * 2)?
                .chunks_exact(2)
                .map(|b| f32::from(u16::from_le_bytes([b[0], b[1]])) / 65535.0)
                .collect();
            values.push(v);
        }
        if !r.0.is_empty() {
            return Err("baked AO: bytes after the last instance".into());
        }
        self.values = values;
        self.queue.clear();
        self.cursor = 0;
        self.scope = None;
        self.occluders = None;
        self.cache = HashMap::new();
        self.unordered = false;
        self.queued_vertices = 0;
        self.cycle_vertices = 0;
        Ok(())
    }

    /// Fraction of the current bake done, in `[0, 1]` (1 when idle).
    pub fn progress(&self) -> f32 {
        if self.queue.is_empty() || self.cycle_vertices == 0 {
            return 1.0;
        }
        let left = self.queued_vertices.saturating_sub(self.cursor);
        (1.0 - left as f32 / self.cycle_vertices as f32).clamp(0.0, 1.0)
    }

    fn enqueue(&mut self, edit: &WorldEdit, index: usize, region: Option<(V3, V3)>) {
        let Some(inst) = edit.instances.get(index) else {
            return;
        };
        let receiver = edit
            .receiver_meshes
            .get(inst.mesh as usize)
            .copied()
            .unwrap_or(false);
        if !receiver || inst.first_vertex.is_none() {
            return;
        }
        if let Some(p) = self.queue.iter_mut().find(|p| p.instance as usize == index) {
            p.region = match (p.region, region) {
                (Some(a), Some(b)) => Some(merged(a, b)),
                _ => None,
            };
            return;
        }
        if self.queue.is_empty() {
            self.cycle_vertices = 0;
        }
        let n = edit.meshes[inst.mesh as usize].positions.len();
        self.queue.push_back(Pending {
            instance: index as u32,
            region,
        });
        self.queued_vertices += n;
        self.cycle_vertices += n;
        self.unordered = true;
    }

    /// Re-queues the vertices whose occlusion can change because geometry inside `changed`
    /// (world bounds of what moved, before and after) moved, appeared or disappeared.
    pub fn invalidate_region(&mut self, edit: &WorldEdit, changed: (V3, V3)) {
        let reach = self.config.max_distance + self.config.normal_bias + 0.01;
        let region = grown(changed, reach);
        self.scope = match (self.queue.is_empty(), self.scope) {
            (true, _) => Some(grown(region, reach)),
            (false, Some(s)) => Some(merged(s, grown(region, reach))),
            (false, None) => None,
        };
        self.occluders = None;
        self.cache.clear();
        let front = self.queue.front().map(|p| p.instance as usize);
        for (index, inst) in edit.instances.iter().enumerate() {
            let mesh = &edit.meshes[inst.mesh as usize];
            if overlaps(transformed_bounds(mesh.bounds, &inst.transform), region) {
                // The entry being sampled used the old occluders for the vertices behind its
                // cursor: restart it.
                if front == Some(index) {
                    self.cursor = 0;
                }
                self.enqueue(edit, index, Some(region));
            }
        }
    }

    /// Reads occluder triangles (all, or those reaching into `scope`) from instance `*instance`,
    /// triangle `*triangle` on, until `*work` runs out; true once every instance is read.
    fn collect(
        edit: &WorldEdit,
        scope: Option<(V3, V3)>,
        (instance, triangle): (&mut usize, &mut usize),
        tris: &mut Vec<Tri>,
        work: &mut usize,
    ) -> bool {
        while let Some(inst) = edit.instances.get(*instance) {
            let mesh_idx = inst.mesh as usize;
            let mesh = &edit.meshes[mesh_idx];
            let m = &inst.transform;
            let skip = !inst.visible
                || inst.first_vertex.is_none()
                || !edit.occluder_meshes.get(mesh_idx).copied().unwrap_or(false)
                || scope.is_some_and(|s| !overlaps(transformed_bounds(mesh.bounds, m), s));
            let count = if skip { 0 } else { mesh.indices.len() / 3 };
            while *triangle < count {
                if *work == 0 {
                    return false;
                }
                *work -= 1;
                let t = &mesh.indices[*triangle * 3..*triangle * 3 + 3];
                *triangle += 1;
                let a = transform_point(m, mesh.positions[t[0] as usize]);
                let b = transform_point(m, mesh.positions[t[1] as usize]);
                let c = transform_point(m, mesh.positions[t[2] as usize]);
                if let Some(s) = scope {
                    let lo = [0, 1, 2].map(|k| fmin(fmin(a[k], b[k]), c[k]));
                    let hi = [0, 1, 2].map(|k| fmax(fmax(a[k], b[k]), c[k]));
                    if !overlaps((lo, hi), s) {
                        continue;
                    }
                }
                tris.extend(make_tri(a, b, c));
            }
            *instance += 1;
            *triangle = 0;
        }
        true
    }

    /// Advances the occluders by `work` units (a triangle read or binned); true once ready.
    fn prepare_occluders(&mut self, edit: &WorldEdit, mut work: usize) -> bool {
        let state = self.occluders.take().unwrap_or(Occluders::Collecting {
            instance: 0,
            triangle: 0,
            tris: Vec::new(),
        });
        self.occluders = Some(match state {
            Occluders::Collecting {
                mut instance,
                mut triangle,
                mut tris,
            } => {
                let done = Self::collect(
                    edit,
                    self.scope,
                    (&mut instance, &mut triangle),
                    &mut tris,
                    &mut work,
                );
                if done {
                    Occluders::Building(BvhBuilder::new(tris))
                } else {
                    Occluders::Collecting {
                        instance,
                        triangle,
                        tris,
                    }
                }
            }
            Occluders::Building(mut builder) => {
                if builder.step(work) {
                    Occluders::Ready(builder.finish())
                } else {
                    Occluders::Building(builder)
                }
            }
            ready @ Occluders::Ready(_) => ready,
        });
        matches!(self.occluders, Some(Occluders::Ready(_)))
    }

    /// Samples up to `budget` vertices (scanning past unchanged ones costs nothing against it),
    /// nearest `eye` first when given. Returns the vertex ranges whose AO changed, per instance,
    /// so they can be rewritten at once rather than when a large instance finishes.
    pub fn step(
        &mut self,
        edit: &WorldEdit,
        budget: usize,
        eye: Option<V3>,
    ) -> Vec<(usize, std::ops::Range<usize>)> {
        let mut updated = Vec::new();
        if self.queue.is_empty() {
            return updated;
        }
        if let Some(eye) = eye
            && std::mem::take(&mut self.unordered)
        {
            let distance = |p: &Pending| {
                let inst = &edit.instances[p.instance as usize];
                let (lo, hi) =
                    transformed_bounds(edit.meshes[inst.mesh as usize].bounds, &inst.transform);
                (0..3)
                    .map(|k| fmax(fmax(lo[k] - eye[k], eye[k] - hi[k]), 0.0).powi(2))
                    .sum::<f32>()
            };
            // An entry already under way keeps its place.
            let keep = usize::from(self.cursor > 0);
            if let Some(rest) = self.queue.make_contiguous().get_mut(keep..) {
                rest.sort_by_cached_key(|p| distance(p).to_bits());
            }
        }
        // The occluders come first, read and built a slice at a time like the samples.
        if !self.prepare_occluders(edit, budget.saturating_mul(BUILD_WORK_PER_SAMPLE)) {
            return updated;
        }
        let Some(Occluders::Ready(bvh)) = self.occluders.as_ref() else {
            return updated;
        };
        let mut cost = 0usize;
        let scan_limit = budget.saturating_mul(64);
        let mut scanned = 0usize;
        while cost < budget && scanned < scan_limit {
            let Some(front) = self.queue.front().copied() else {
                break;
            };
            let index = front.instance as usize;
            let inst = &edit.instances[index];
            let mesh = &edit.meshes[inst.mesh as usize];
            let n = mesh.positions.len();
            let values = &mut self.values[index];
            if values.len() != n {
                *values = vec![1.0; n];
            }
            // The span of vertices this pass gave new values.
            let mut changed: Option<(usize, usize)> = None;
            if bvh.is_empty() {
                values.fill(1.0);
                changed = Some((0, n));
                self.cursor = n;
            } else if !inst.visible {
                // Hidden: its vertices are collapsed; it is re-queued when shown.
                self.cursor = n;
            }
            let m = &inst.transform;
            while self.cursor < n && cost < budget && scanned < scan_limit {
                let vi = self.cursor;
                self.cursor += 1;
                scanned += 1;
                let wp = transform_point(m, mesh.positions[vi]);
                if front.region.is_some_and(|r| !contains(r, wp)) {
                    continue;
                }
                changed =
                    Some(changed.map_or((vi, vi + 1), |(lo, hi)| (lo.min(vi), hi.max(vi + 1))));
                let wn = mesh
                    .normals
                    .get(vi)
                    .map_or([0.0, 1.0, 0.0], |&nrm| world_normal(m, nrm));
                let key = sample_key(wp, wn);
                values[vi] = match self.cache.get(&key) {
                    Some(&v) => v,
                    None => {
                        cost += 1;
                        let v = bvh.sample_ao(wp, wn, &self.config);
                        self.cache.insert(key, v);
                        v
                    }
                };
            }
            if let Some((lo, hi)) = changed {
                updated.push((index, lo..hi));
            }
            if self.cursor >= n {
                self.queue.pop_front();
                self.queued_vertices = self.queued_vertices.saturating_sub(n);
                self.cursor = 0;
            }
        }
        if self.queue.is_empty() {
            // Idle: free the occluders and samples until the next edit.
            self.occluders = None;
            self.cache = HashMap::new();
            self.scope = None;
        }
        updated
    }
}

#[cfg(test)]
mod tests {
    use super::super::scene::{EditInstance, EditMesh};
    use super::*;

    /// A 12 m floor quad (instance 0) and a 4 m wall quad standing on it at x = `wall_x`
    /// (instance 1), both opaque receivers and occluders.
    fn floor_and_wall(wall_x: f32) -> WorldEdit {
        let quad = |positions: Vec<V3>, normal: V3| {
            let bounds = super::super::props::bounds(&positions);
            EditMesh {
                normals: vec![normal; positions.len()],
                uvs: vec![[0.0, 0.0]; positions.len()],
                indices: vec![0, 2, 1, 0, 3, 2],
                positions,
                bounds,
            }
        };
        let mut floor: Vec<V3> = Vec::new();
        // A grid of floor vertices so AO varies across it.
        for i in 0..=12 {
            floor.push([-6.0 + i as f32, 0.0, 0.0]);
        }
        let mut floor_mesh = quad(
            vec![
                [-6.0, 0.0, -6.0],
                [6.0, 0.0, -6.0],
                [6.0, 0.0, 6.0],
                [-6.0, 0.0, 6.0],
            ],
            [0.0, 1.0, 0.0],
        );
        floor_mesh.positions.extend(floor);
        floor_mesh.normals = vec![[0.0, 1.0, 0.0]; floor_mesh.positions.len()];
        floor_mesh.uvs = vec![[0.0, 0.0]; floor_mesh.positions.len()];
        let wall = quad(
            vec![
                [0.0, 0.0, -6.0],
                [0.0, 0.0, 6.0],
                [0.0, 4.0, 6.0],
                [0.0, 4.0, -6.0],
            ],
            [1.0, 0.0, 0.0],
        );
        let translate = |x: f32| {
            let mut m = [0.0; 16];
            m[0] = 1.0;
            m[5] = 1.0;
            m[10] = 1.0;
            m[15] = 1.0;
            m[12] = x;
            m
        };
        WorldEdit {
            water_meshes: vec![false, false],
            occluder_meshes: vec![true, true],
            receiver_meshes: vec![true, true],
            instances: vec![
                EditInstance {
                    mesh: 0,
                    transform: translate(0.0),
                    first_vertex: Some(0),
                    visible: true,
                },
                EditInstance {
                    mesh: 1,
                    transform: translate(wall_x),
                    first_vertex: Some(17),
                    visible: true,
                },
            ],
            vertex_count: floor_mesh.positions.len() + 4,
            meshes: vec![floor_mesh, wall],
            lights: Vec::new(),
            draw_sources: Vec::new(),
        }
    }

    #[test]
    fn a_sidecar_carries_a_finished_bake_to_the_same_world_only() {
        let edit = floor_and_wall(0.0);
        let baked = bake_all(&edit, usize::MAX / 128);
        let file = baked.export(&edit).expect("a finished bake exports");
        assert!(
            AoBake::new(&edit, AoConfig::default())
                .export(&edit)
                .is_none()
        );

        let mut loaded = AoBake::new(&edit, AoConfig::default());
        loaded
            .import(&edit, &file)
            .expect("the same world loads it");
        assert!(loaded.is_idle());
        for i in 0..edit.instances.len() {
            let (a, b) = (baked.values(i).unwrap(), loaded.values(i).unwrap());
            assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1.0 / 65535.0));
        }

        // The wall moved: the file no longer fits, and nothing changes.
        let moved = floor_and_wall(1.0);
        let mut other = AoBake::new(&moved, AoConfig::default());
        assert!(other.import(&moved, &file).is_err());
        assert!(!other.is_idle() && other.values(0).is_none());
        assert!(other.import(&moved, &file[..file.len() - 1]).is_err());
    }

    fn bake_all(edit: &WorldEdit, budget: usize) -> AoBake {
        let mut bake = AoBake::new(edit, AoConfig::default());
        while !bake.is_idle() {
            bake.step(edit, budget, None);
        }
        bake
    }

    #[test]
    fn sliced_bake_matches_a_single_pass_and_reports_progress() {
        let edit = floor_and_wall(0.0);
        let whole = bake_all(&edit, usize::MAX / 128);
        let mut sliced = AoBake::new(&edit, AoConfig::default());
        assert_eq!(sliced.progress(), 0.0);
        let mut steps = 0;
        while !sliced.is_idle() {
            sliced.step(&edit, 1, Some([5.0, 1.0, 0.0]));
            steps += 1;
            assert!((0.0..=1.0).contains(&sliced.progress()));
        }
        assert!(steps > 2, "a one-sample budget must take several steps");
        assert_eq!(sliced.progress(), 1.0);
        for i in 0..edit.instances.len() {
            assert_eq!(whole.values(i), sliced.values(i), "instance {i}");
        }
        let floor = whole.values(0).expect("floor baked");
        // Floor vertex at x = 0.0 touches the wall; x = -6.0 is 6 m away (beyond reach).
        assert!(
            floor[4 + 6] < 0.8,
            "floor at the wall darkens: {}",
            floor[10]
        );
        assert!(
            floor[4] > 0.98,
            "floor far from the wall stays open: {}",
            floor[4]
        );
    }

    #[test]
    fn moving_an_occluder_resamples_only_vertices_within_reach() {
        let mut edit = floor_and_wall(0.0);
        let mut bake = bake_all(&edit, usize::MAX / 128);
        let before: Vec<f32> = bake.values(0).expect("floor baked").to_vec();

        // Slide the wall 5 m along +x: floor near x = 0 opens up, floor near x = 5 darkens.
        let old = transformed_bounds(edit.meshes[1].bounds, &edit.instances[1].transform);
        edit.instances[1].transform[12] = 5.0;
        let new = transformed_bounds(edit.meshes[1].bounds, &edit.instances[1].transform);
        bake.invalidate_region(&edit, (old.0, new.1));
        assert!(!bake.is_idle());
        while !bake.is_idle() {
            bake.step(&edit, 2, None);
        }
        let after = bake.values(0).expect("floor baked");
        let fresh = bake_all(&edit, usize::MAX / 128);
        assert_eq!(
            Some(after),
            fresh.values(0),
            "incremental equals a full re-bake"
        );
        assert!(
            after[4 + 6] > before[4 + 6],
            "floor at the old wall opens up"
        );
        assert!(
            after[4 + 11] < before[4 + 11],
            "floor at the new wall darkens"
        );
    }

    #[test]
    fn open_plane_and_convex_box_stay_unoccluded_while_concave_corner_darkens() {
        let positions = vec![
            [-6.0, 0.0, -6.0],
            [6.0, 0.0, -6.0],
            [6.0, 0.0, 6.0],
            [-6.0, 0.0, 6.0],
            [0.0, 0.0, -6.0],
            [0.0, 0.0, 6.0],
            [0.0, 4.0, 6.0],
            [0.0, 4.0, -6.0],
        ];
        let indices = vec![0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7];
        let queries = vec![[5.5, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let normals = vec![[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let ao = bake_vertex_ao(
            &positions,
            &indices,
            &queries,
            &normals,
            &AoConfig::default(),
        );
        assert!(ao[0] > 0.98, "open floor should stay unoccluded: {}", ao[0]);
        assert!(ao[1] < 0.65, "floor beside wall should darken: {}", ao[1]);
        assert!(ao[2] < 0.70, "wall base contact should darken: {}", ao[2]);

        let mut rev_indices = Vec::new();
        for tri in indices.chunks_exact(3).rev() {
            rev_indices.extend_from_slice(tri);
        }
        let ao_rev = bake_vertex_ao(
            &positions,
            &rev_indices,
            &queries,
            &normals,
            &AoConfig::default(),
        );
        assert_eq!(ao, ao_rev);
    }

    #[test]
    fn floating_convex_cube_has_no_false_self_occlusion() {
        let mut pos = Vec::new();
        let mut nrm = Vec::new();
        let mut idx = Vec::new();
        let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
            (
                [0.0, 1.0, 0.0],
                [
                    [-1.0, 1.0, -1.0],
                    [-1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, -1.0],
                ],
            ),
            (
                [0.0, -1.0, 0.0],
                [
                    [-1.0, -1.0, -1.0],
                    [1.0, -1.0, -1.0],
                    [1.0, -1.0, 1.0],
                    [-1.0, -1.0, 1.0],
                ],
            ),
            (
                [1.0, 0.0, 0.0],
                [
                    [1.0, -1.0, -1.0],
                    [1.0, 1.0, -1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, -1.0, 1.0],
                ],
            ),
            (
                [-1.0, 0.0, 0.0],
                [
                    [-1.0, -1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, -1.0],
                ],
            ),
            (
                [0.0, 0.0, 1.0],
                [
                    [-1.0, -1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                ],
            ),
            (
                [0.0, 0.0, -1.0],
                [
                    [1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                    [1.0, 1.0, -1.0],
                ],
            ),
        ];
        for (n, corners) in faces {
            let base = pos.len() as u32;
            pos.extend_from_slice(&corners);
            nrm.extend_from_slice(&[n; 4]);
            idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        let ao = bake_vertex_ao(&pos, &idx, &pos, &nrm, &AoConfig::default());
        for (i, &v) in ao.iter().enumerate() {
            assert_eq!(v, 1.0, "convex vertex {i} self-occluded: {v}");
        }
    }
}
