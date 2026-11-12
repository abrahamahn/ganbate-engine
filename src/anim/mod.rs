//! Animation for run-time models ([`crate::assets::model::Model`]): poses (every node's local
//! translation, rotation and scale, plus morph weights), clip sampling and blending, node
//! world transforms and CPU skinning and morphing of primitives.
//!
//! A game drives a pose each frame: start from [`Pose::rest`] (or keep the last pose), apply one
//! or more clips with weights ([`Pose::apply`]: weight 1 replaces what the clip animates, a
//! smaller weight mixes toward it, which is how cross-fades and two-clip blends are built),
//! adjust nodes directly (a head turned toward a target, or a two-bone limb effector), or
//! evaluate the entire stack in one call via [`pose_model_clip`], then hand it to the renderer.
use crate::assets::model::{Channel, Interpolation, Model, Path, Primitive};
use crate::render::math::{self, M4, V3};

pub type Quat = [f32; 4];

/// Local transforms of every node of a model and the morph weights of its mesh nodes.
#[derive(Clone, Debug, PartialEq)]
pub struct Pose {
    pub translation: Vec<V3>,
    /// Unit quaternions `[x, y, z, w]`.
    pub rotation: Vec<Quat>,
    pub scale: Vec<V3>,
    /// Per node: one weight per morph target of its mesh (empty without targets).
    pub weights: Vec<Vec<f32>>,
}

impl Pose {
    /// The model's own pose: node transforms and default morph weights as authored.
    pub fn rest(model: &Model) -> Self {
        Self {
            translation: model.nodes.iter().map(|n| n.translation).collect(),
            rotation: model
                .nodes
                .iter()
                .map(|n| normalize_q(n.rotation))
                .collect(),
            scale: model.nodes.iter().map(|n| n.scale).collect(),
            weights: model
                .nodes
                .iter()
                .map(|n| match (n.mesh, &n.weights) {
                    (Some(m), Some(w)) if w.len() == model.meshes[m].weights.len() => w.clone(),
                    (Some(m), _) => model.meshes[m].weights.clone(),
                    (None, _) => vec![],
                })
                .collect(),
        }
    }

    /// Mixes clip `clip` sampled at `time` seconds (clamped to the clip) into the pose:
    /// `weight` 1 replaces every channel the clip animates, 0.5 goes halfway (rotations by
    /// spherical interpolation). Channels the clip does not animate are left alone.
    pub fn apply(&mut self, model: &Model, clip: usize, time: f32, weight: f32) {
        let Some(clip) = model.clips.get(clip) else {
            return;
        };
        let w = if weight.is_finite() {
            weight.clamp(0.0, 1.0)
        } else {
            return;
        };
        if w <= 0.0 {
            return;
        }
        let time = if time.is_finite() {
            time.clamp(0.0, clip.duration)
        } else {
            0.0
        };
        let mut v = [0.0f32; 4];
        let mut many = Vec::new();
        for ch in &clip.channels {
            let n = ch.node;
            match ch.path {
                Path::Translation => {
                    sample(ch, time, &mut v[..3]);
                    self.translation[n] = math::lerp3(self.translation[n], [v[0], v[1], v[2]], w);
                }
                Path::Scale => {
                    sample(ch, time, &mut v[..3]);
                    self.scale[n] = math::lerp3(self.scale[n], [v[0], v[1], v[2]], w);
                }
                Path::Rotation => {
                    sample_rotation(ch, time, &mut v);
                    self.rotation[n] = slerp(self.rotation[n], v, w);
                }
                Path::Weights => {
                    many.resize(ch.width, 0.0);
                    sample(ch, time, &mut many);
                    let out = &mut self.weights[n];
                    for (o, s) in out.iter_mut().zip(&many) {
                        *o += (*s - *o) * w;
                    }
                }
            }
        }
    }

    /// Node `node`'s local transform as a matrix.
    pub fn local(&self, node: usize) -> M4 {
        compose(
            self.translation[node],
            self.rotation[node],
            self.scale[node],
        )
    }

    /// Every node's transform in the model's space (parents first; roots relative to the
    /// model's origin).
    pub fn globals(&self, model: &Model) -> Vec<M4> {
        let mut out = vec![math::IDENTITY; model.nodes.len()];
        let mut done = vec![false; model.nodes.len()];
        // Parents before children: walk up from each node to its first resolved ancestor.
        let mut chain = Vec::new();
        for i in 0..model.nodes.len() {
            let mut at = i;
            chain.clear();
            while !done[at] {
                chain.push(at);
                match model.nodes[at].parent {
                    Some(p) => at = p,
                    None => break,
                }
            }
            for &n in chain.iter().rev() {
                let local = self.local(n);
                out[n] = match model.nodes[n].parent {
                    Some(p) => math::mul(&out[p], &local),
                    None => local,
                };
                done[n] = true;
            }
        }
        out
    }
}

/// The joint matrices of `skin` for a mesh skinned by it: `global(joint) * inverse_bind`, in
/// the model's space (glTF ignores the skinned mesh node's own transform).
pub fn joint_matrices(model: &Model, skin: usize, globals: &[M4]) -> Vec<M4> {
    let s = &model.skins[skin];
    s.joints
        .iter()
        .zip(&s.inverse_bind)
        .map(|(&j, ib)| math::mul(&globals[j], ib))
        .collect()
}

/// A primitive's positions and normals after morph targets (`weights`) and skinning (`joints`,
/// from [`joint_matrices`]; `None` keeps them in the mesh's space). Normals come back unit
/// length, or empty when the primitive has none.
pub fn deform(p: &Primitive, weights: &[f32], joints: Option<&[M4]>) -> (Vec<V3>, Vec<V3>) {
    let mut pos = p.positions.clone();
    let mut nrm = p.normals.clone();
    for (t, &w) in p.targets.iter().zip(weights) {
        if w == 0.0 {
            continue;
        }
        for (v, d) in pos.iter_mut().zip(&t.positions) {
            *v = math::add(*v, math::scale(*d, w));
        }
        for (v, d) in nrm.iter_mut().zip(&t.normals) {
            *v = math::add(*v, math::scale(*d, w));
        }
    }
    if let Some(mats) = joints
        && !p.joints.is_empty()
    {
        for (i, (js, ws)) in p.joints.iter().zip(&p.weights).enumerate() {
            let (js, ws) = (*js, *ws);
            let total: f32 = ws.iter().sum();
            let norm = if total > 1e-6 { 1.0 / total } else { 0.0 };
            let mut m = [0.0f32; 16];
            for (&joint, &weight) in js.iter().zip(&ws) {
                let w = weight * norm;
                if w == 0.0 {
                    continue;
                }
                let Some(j) = mats.get(usize::from(joint)) else {
                    continue;
                };
                for (o, x) in m.iter_mut().zip(j) {
                    *o += w * x;
                }
            }
            if norm == 0.0 {
                m = math::IDENTITY;
            }
            pos[i] = math::transform_point(&m, pos[i]);
            if let Some(n) = nrm.get_mut(i) {
                // Joint matrices are rigid or uniformly scaled in practice: the upper 3x3 serves.
                let t = math::transform(&m, [n[0], n[1], n[2], 0.0]);
                *n = [t[0], t[1], t[2]];
            }
        }
    }
    for n in &mut nrm {
        *n = math::normalize(*n);
    }
    (pos, nrm)
}

/// The vertices of `p` that some morph target moves (by position or normal), ascending: a face's
/// targets move a few hundred vertices of a whole figure.
pub fn morphed_vertices(p: &Primitive) -> Vec<u32> {
    let mut moved = vec![false; p.positions.len()];
    for t in &p.targets {
        for deltas in [&t.positions, &t.normals] {
            for (m, d) in moved.iter_mut().zip(deltas.iter()) {
                if d.iter().any(|x| *x != 0.0) {
                    *m = true;
                }
            }
        }
    }
    moved
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| m.then_some(i as u32))
        .collect()
}

/// Per morph target of `p`: whether it moves any vertex (by position or normal). A figure's
/// face targets move nothing on its body.
pub fn moving_targets(p: &Primitive) -> Vec<bool> {
    p.targets
        .iter()
        .map(|t| {
            t.positions
                .iter()
                .chain(&t.normals)
                .any(|d| d.iter().any(|x| *x != 0.0))
        })
        .collect()
}

/// Whether morphing with `weights` rather than `from` moves a vertex: some target that moves one
/// (`moving`, from [`moving_targets`]) changed weight.
pub fn morph_changes(from: &[f32], weights: &[f32], moving: &[bool]) -> bool {
    from.len() != weights.len()
        || from
            .iter()
            .zip(weights)
            .zip(moving)
            .any(|((a, b), &moves)| moves && a != b)
}

/// Reorders `p`'s vertices so that those its morph targets move ([`morphed_vertices`]) come
/// first, in their order, and the rest follow in theirs: a morph then rewrites one short run of
/// the vertex buffer, however the modeller ordered them. Every per-vertex attribute moves with
/// its vertex and the triangles are renumbered, so the primitive draws and deforms as before.
pub fn gather_morphed(p: &mut Primitive) {
    let moved = morphed_vertices(p);
    // Nothing moves, or what moves is first already.
    if moved
        .last()
        .is_none_or(|&last| last as usize + 1 == moved.len())
    {
        return;
    }
    let count = p.positions.len();
    let mut is_moved = vec![false; count];
    for &v in &moved {
        is_moved[v as usize] = true;
    }
    let order: Vec<usize> = moved
        .iter()
        .map(|&v| v as usize)
        .chain((0..count).filter(|&v| !is_moved[v]))
        .collect();
    let mut renumber = vec![0u32; count];
    for (new, &old) in order.iter().enumerate() {
        renumber[old] = new as u32;
    }
    fn reorder<T: Copy>(values: &mut Vec<T>, order: &[usize]) {
        if values.len() == order.len() {
            *values = order.iter().map(|&i| values[i]).collect();
        }
    }
    // Every field named, so a new per-vertex attribute cannot be left out of order.
    let Primitive {
        positions,
        normals,
        uvs,
        indices,
        material: _,
        joints,
        weights,
        targets,
    } = p;
    reorder(positions, &order);
    reorder(normals, &order);
    reorder(uvs, &order);
    reorder(joints, &order);
    reorder(weights, &order);
    for t in targets {
        reorder(&mut t.positions, &order);
        reorder(&mut t.normals, &order);
    }
    for i in indices.iter_mut() {
        if let Some(&new) = renumber.get(*i as usize) {
            *i = new;
        }
    }
}

/// Positions and normals of `p`'s vertices `first..first + count` under morph `weights` (not
/// skinned), where only `moved` ([`morphed_vertices`]) differ from the mesh: what
/// [`deform`] gives for that range, without visiting the rest.
pub fn morph_range(
    p: &Primitive,
    weights: &[f32],
    moved: &[u32],
    first: usize,
    count: usize,
) -> (Vec<V3>, Vec<V3>) {
    let end = (first + count).min(p.positions.len());
    let mut pos = p.positions[first..end].to_vec();
    let mut nrm = if p.normals.is_empty() {
        vec![]
    } else {
        p.normals[first..end].to_vec()
    };
    for &v in moved {
        let v = v as usize;
        if v < first || v >= end {
            continue;
        }
        for (t, &w) in p.targets.iter().zip(weights) {
            if w == 0.0 {
                continue;
            }
            if let Some(d) = t.positions.get(v) {
                pos[v - first] = math::add(pos[v - first], math::scale(*d, w));
            }
            if let (Some(n), Some(d)) = (nrm.get_mut(v - first), t.normals.get(v)) {
                *n = math::add(*n, math::scale(*d, w));
            }
        }
    }
    for n in &mut nrm {
        *n = math::normalize(*n);
    }
    (pos, nrm)
}

/// Translation, rotation, scale to a column-major matrix.
pub fn compose(t: V3, q: Quat, s: V3) -> M4 {
    let [x, y, z, w] = q;
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, xz, yz) = (x * y, x * z, y * z);
    let (wx, wy, wz) = (w * x, w * y, w * z);
    [
        (1.0 - 2.0 * (yy + zz)) * s[0],
        2.0 * (xy + wz) * s[0],
        2.0 * (xz - wy) * s[0],
        0.0,
        2.0 * (xy - wz) * s[1],
        (1.0 - 2.0 * (xx + zz)) * s[1],
        2.0 * (yz + wx) * s[1],
        0.0,
        2.0 * (xz + wy) * s[2],
        2.0 * (yz - wx) * s[2],
        (1.0 - 2.0 * (xx + yy)) * s[2],
        0.0,
        t[0],
        t[1],
        t[2],
        1.0,
    ]
}

pub fn normalize_q(q: Quat) -> Quat {
    let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if l > 1e-12 && l.is_finite() {
        q.map(|v| v / l)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// `a * b`: rotate by `b`, then by `a`.
pub fn mul_q(a: Quat, b: Quat) -> Quat {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

/// Spherical interpolation along the shorter arc.
pub fn slerp(a: Quat, b: Quat, t: f32) -> Quat {
    let mut d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let b = if d < 0.0 {
        d = -d;
        b.map(|v| -v)
    } else {
        b
    };
    if d > 0.9995 {
        return normalize_q(std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t));
    }
    let theta = d.clamp(-1.0, 1.0).acos();
    let s = theta.sin();
    let (wa, wb) = (((1.0 - t) * theta).sin() / s, (t * theta).sin() / s);
    normalize_q(std::array::from_fn(|i| a[i] * wa + b[i] * wb))
}

/// Inverse of a unit quaternion (`[-x, -y, -z, w]`).
pub fn conjugate_q(q: Quat) -> Quat {
    [-q[0], -q[1], -q[2], q[3]]
}

/// Rotates `v` by unit quaternion `q`.
pub fn rotate_vec(q: Quat, v: V3) -> V3 {
    let qv = [q[0], q[1], q[2]];
    let t = math::scale(math::cross(qv, v), 2.0);
    math::add(v, math::add(math::scale(t, q[3]), math::cross(qv, t)))
}

/// Unit quaternion rotating `angle` radians counter-clockwise about `axis`.
pub fn quat_from_axis_angle(axis: V3, angle: f32) -> Quat {
    if !angle.is_finite() || axis.iter().any(|x| !x.is_finite()) {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let len = math::length(axis);
    if len <= 1e-8 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let u = math::scale(axis, 1.0 / len);
    let half = 0.5 * angle;
    let (s, c) = half.sin_cos();
    normalize_q([u[0] * s, u[1] * s, u[2] * s, c])
}

/// Euler angles (`pitch` about +X, `yaw` about +Y, `roll` about +Z) in radians: roll first, then
/// pitch, then yaw (`R_y(yaw) * R_x(pitch) * R_z(roll)`).
pub fn quat_from_euler(pitch: f32, yaw: f32, roll: f32) -> Quat {
    let qx = quat_from_axis_angle([1.0, 0.0, 0.0], pitch);
    let qy = quat_from_axis_angle([0.0, 1.0, 0.0], yaw);
    let qz = quat_from_axis_angle([0.0, 0.0, 1.0], roll);
    normalize_q(mul_q(mul_q(qy, qx), qz))
}

/// Shortest-arc unit quaternion rotating unit direction `from` onto unit direction `to`.
pub fn quat_from_vectors(from: V3, to: V3) -> Quat {
    let lf = math::length(from);
    let lt = math::length(to);
    if lf <= 1e-8 || lt <= 1e-8 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let a = math::scale(from, 1.0 / lf);
    let b = math::scale(to, 1.0 / lt);
    let d = math::dot(a, b);
    if d >= 1.0 - 1e-6 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    if d <= -1.0 + 1e-6 {
        let mut ortho = math::cross([1.0, 0.0, 0.0], a);
        if math::length(ortho) < 1e-4 {
            ortho = math::cross([0.0, 1.0, 0.0], a);
        }
        return quat_from_axis_angle(ortho, std::f32::consts::PI);
    }
    let c = math::cross(a, b);
    normalize_q([c[0], c[1], c[2], 1.0 + d])
}

/// The key interval holding `time`: (index of the key at or before it, fraction to the next).
fn locate(times: &[f32], time: f32) -> (usize, f32) {
    let n = times.len();
    if n == 0 || time <= times[0] {
        return (0, 0.0);
    }
    if time >= times[n - 1] {
        return (n - 1, 0.0);
    }
    let i = times.partition_point(|&t| t <= time) - 1;
    let span = times[i + 1] - times[i];
    let f = if span > 0.0 {
        (time - times[i]) / span
    } else {
        0.0
    };
    (i, f)
}

/// Samples a translation, scale or weights channel into `out` (`ch.width` values).
fn sample(ch: &Channel, time: f32, out: &mut [f32]) {
    let w = ch.width;
    let (i, f) = locate(&ch.times, time);
    let last = ch.times.len() - 1;
    match ch.interpolation {
        Interpolation::Step => out.copy_from_slice(&ch.values[i * w..i * w + w]),
        Interpolation::Linear => {
            let j = (i + 1).min(last);
            let (a, b) = (&ch.values[i * w..i * w + w], &ch.values[j * w..j * w + w]);
            for ((o, a), b) in out.iter_mut().zip(a).zip(b) {
                *o = a + (b - a) * f;
            }
        }
        Interpolation::CubicSpline => {
            // Keys are (in-tangent, value, out-tangent), each `w` wide.
            let key = |n: usize, part: usize, k: usize| ch.values[(n * 3 + part) * w + k];
            if i == last || f == 0.0 {
                for (k, o) in out.iter_mut().enumerate() {
                    *o = key(i, 1, k);
                }
                return;
            }
            let dt = ch.times[i + 1] - ch.times[i];
            let (t2, t3) = (f * f, f * f * f);
            for (k, o) in out.iter_mut().enumerate() {
                let p0 = key(i, 1, k);
                let m0 = key(i, 2, k) * dt;
                let p1 = key(i + 1, 1, k);
                let m1 = key(i + 1, 0, k) * dt;
                *o = (2.0 * t3 - 3.0 * t2 + 1.0) * p0
                    + (t3 - 2.0 * t2 + f) * m0
                    + (-2.0 * t3 + 3.0 * t2) * p1
                    + (t3 - t2) * m1;
            }
        }
    }
}

fn sample_rotation(ch: &Channel, time: f32, out: &mut [f32; 4]) {
    match ch.interpolation {
        Interpolation::Linear => {
            let (i, f) = locate(&ch.times, time);
            let j = (i + 1).min(ch.times.len() - 1);
            let q = |n: usize| -> Quat { std::array::from_fn(|k| ch.values[n * 4 + k]) };
            *out = slerp(normalize_q(q(i)), normalize_q(q(j)), f);
        }
        _ => {
            sample(ch, time, out);
            *out = normalize_q(*out);
        }
    }
}

mod ik;
#[cfg(test)]
mod tests;

pub use ik::{
    BlendSample1D, BlendSample2D, BoneOffset, ClipLayer, LookAtChain, MorphOverride, PoseSpec,
    TwoBoneIk, blend_space_1d_layers, blend_space_2d_layers, hermite_fade, pose_model_clip,
    weights_to_clip_layers,
};
