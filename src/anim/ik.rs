//! Blend spaces, Hermite cross-fades, additive bone offsets, Look-At aim chains, analytical
//! Two-Bone IK, and single-call pose stack evaluation.

use super::*;

/// C1-continuous Hermite cross-fade weight (`3t^2 - 2t^3`) over `duration` seconds, clamped to
/// `0.0..=1.0` (`1.0` once `elapsed >= duration` or when `duration <= 0.0`).
pub fn hermite_fade(elapsed: f32, duration: f32) -> f32 {
    if !elapsed.is_finite() || !duration.is_finite() || duration <= 0.0 || elapsed >= duration {
        return 1.0;
    }
    if elapsed <= 0.0 {
        return 0.0;
    }
    let t = (elapsed / duration).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// One animation clip layer sampled at `time` seconds with blend `weight` (`0.0..=1.0`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipLayer {
    pub clip: usize,
    pub time: f32,
    pub weight: f32,
}

/// Sample point in a 1D parametric blend space (for example speed or stance).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendSample1D {
    pub clip: usize,
    pub position: f32,
    pub rate: f32,
}

/// Sample point in a 2D parametric blend space (for example `[strafe, forward]` or `[yaw, pitch]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendSample2D {
    pub clip: usize,
    pub position: [f32; 2],
    pub rate: f32,
}

/// Converts normalized non-negative sample entries `(clip, sample_time, weight)` into sequential
/// [`ClipLayer`] cascade weights (`alpha_m = w_m / sum_{i<=m} w_i`) so [`pose_model_clip`]
/// reproduces the exact convex combination across any number of active samples.
pub fn weights_to_clip_layers(samples: &[(usize, f32, f32)]) -> Vec<ClipLayer> {
    let mut running = 0.0f32;
    let mut out = Vec::new();
    for &(clip, time, w) in samples {
        if !w.is_finite() || w <= 1e-6 {
            continue;
        }
        running += w;
        let alpha = if out.is_empty() {
            1.0
        } else {
            (w / running).clamp(0.0, 1.0)
        };
        out.push(ClipLayer {
            clip,
            time,
            weight: alpha,
        });
    }
    out
}

/// Evaluates a 1D blend space at `param` and phase `normalized_time` (or seconds scaled by `rate`)
/// into sequential [`ClipLayer`]s.
pub fn blend_space_1d_layers(
    samples: &[BlendSample1D],
    param: f32,
    base_time: f32,
) -> Vec<ClipLayer> {
    if samples.is_empty() || !param.is_finite() {
        return Vec::new();
    }
    let mut sorted: Vec<BlendSample1D> = samples
        .iter()
        .copied()
        .filter(|s| s.position.is_finite())
        .collect();
    if sorted.is_empty() {
        return Vec::new();
    }
    sorted.sort_by(|a, b| a.position.total_cmp(&b.position));
    if sorted.len() == 1 || param <= sorted[0].position {
        let s = sorted[0];
        return vec![ClipLayer {
            clip: s.clip,
            time: base_time * s.rate,
            weight: 1.0,
        }];
    }
    let last = sorted[sorted.len() - 1];
    if param >= last.position {
        return vec![ClipLayer {
            clip: last.clip,
            time: base_time * last.rate,
            weight: 1.0,
        }];
    }
    for pair in sorted.windows(2) {
        let a = pair[0];
        let b = pair[1];
        if param <= b.position {
            let span = b.position - a.position;
            let t = if span > 1e-6 {
                ((param - a.position) / span).clamp(0.0, 1.0)
            } else {
                0.0
            };
            return weights_to_clip_layers(&[
                (a.clip, base_time * a.rate, 1.0 - t),
                (b.clip, base_time * b.rate, t),
            ]);
        }
    }
    Vec::new()
}

/// Evaluates a scattered 2D blend space using Gradient Band Interpolation (freeform Cartesian)
/// into sequential [`ClipLayer`]s. Hits `1.0` at every authored sample point and blends smoothly
/// across arbitrary 2D layouts without triangulation artifacts.
pub fn blend_space_2d_layers(
    samples: &[BlendSample2D],
    param: [f32; 2],
    base_time: f32,
) -> Vec<ClipLayer> {
    if samples.is_empty() || !param[0].is_finite() || !param[1].is_finite() {
        return Vec::new();
    }
    if samples.len() == 1 {
        let s = samples[0];
        return vec![ClipLayer {
            clip: s.clip,
            time: base_time * s.rate,
            weight: 1.0,
        }];
    }
    let mut raw_weights = vec![0.0f32; samples.len()];
    let mut sum = 0.0f32;
    for (i, si) in samples.iter().enumerate() {
        let mut influence = 1.0f32;
        for (j, sj) in samples.iter().enumerate() {
            if i == j {
                continue;
            }
            let vx = sj.position[0] - si.position[0];
            let vy = sj.position[1] - si.position[1];
            let denom = vx * vx + vy * vy;
            if denom <= 1e-8 {
                continue;
            }
            let px = param[0] - si.position[0];
            let py = param[1] - si.position[1];
            let hij = (1.0 - (px * vx + py * vy) / denom).clamp(0.0, 1.0);
            if hij < influence {
                influence = hij;
            }
        }
        raw_weights[i] = influence;
        sum += influence;
    }
    if sum <= 1e-6 {
        // Fallback to nearest sample if degenerate.
        let (best_idx, _) = samples
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let da = (param[0] - a.position[0]).powi(2) + (param[1] - a.position[1]).powi(2);
                let db = (param[0] - b.position[0]).powi(2) + (param[1] - b.position[1]).powi(2);
                da.total_cmp(&db)
            })
            .unwrap();
        let s = samples[best_idx];
        return vec![ClipLayer {
            clip: s.clip,
            time: base_time * s.rate,
            weight: 1.0,
        }];
    }
    let weighted: Vec<(usize, f32, f32)> = samples
        .iter()
        .zip(raw_weights)
        .map(|(s, w)| (s.clip, base_time * s.rate, w / sum))
        .collect();
    weights_to_clip_layers(&weighted)
}

/// Additive local transform adjustment applied on top of sampled clips (`translation + dt`,
/// `mul_q(rotation, dq)`, component-wise `scale`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoneOffset {
    pub node: usize,
    pub translation: V3,
    pub rotation: Quat,
    pub scale: V3,
}

impl BoneOffset {
    pub fn euler(node: usize, pitch: f32, yaw: f32, roll: f32) -> Self {
        Self {
            node,
            translation: [0.0; 3],
            rotation: quat_from_euler(pitch, yaw, roll),
            scale: [1.0; 3],
        }
    }
}

/// Explicit morph target weight override applied after clip evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MorphOverride {
    pub node: usize,
    pub target: usize,
    pub weight: f32,
}

/// Procedural multi-joint Look-At aim chain (for example neck + head, or eyes) turning `forward`
/// toward `target_model` (in the model's space) within `[-max_yaw, max_yaw]` and
/// `[-max_pitch, max_pitch]` radians.
#[derive(Clone, Debug, PartialEq)]
pub struct LookAtChain {
    /// `(node_index, share)` pairs ordered parent-to-child; shares typically sum to `1.0`.
    pub nodes: Vec<(usize, f32)>,
    pub target_model: V3,
    pub forward: V3,
    pub up: V3,
    pub max_yaw: f32,
    pub max_pitch: f32,
    pub weight: f32,
}

impl Default for LookAtChain {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            target_model: [0.0, 1.5, 2.0],
            forward: [0.0, 0.0, 1.0],
            up: [0.0, 1.0, 0.0],
            max_yaw: 1.05,
            max_pitch: 0.70,
            weight: 1.0,
        }
    }
}

/// Analytical Two-Bone Inverse Kinematics chain (`root -> mid -> end`) bending in the plane of
/// `pole_model` to reach `target_model` (both in the model's space).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoBoneIk {
    pub root: usize,
    pub mid: usize,
    pub end: usize,
    pub target_model: V3,
    pub pole_model: V3,
    pub weight: f32,
}

/// Complete single-call skeletal evaluation specification for [`pose_model_clip`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PoseSpec<'a> {
    pub reset_to_rest: bool,
    pub clips: &'a [ClipLayer],
    /// `(elapsed_seconds, duration_seconds, optional_node_mask)` for Hermite cross-fade from a
    /// previously snapshotted pose.
    pub crossfade: Option<(f32, f32, Option<&'a [usize]>)>,
    pub look_at: &'a [LookAtChain],
    pub two_bone_ik: &'a [TwoBoneIk],
    pub bone_offsets: &'a [BoneOffset],
    pub morph_overrides: &'a [MorphOverride],
}

impl Pose {
    /// Propagates unit rotations from roots to leaves (free of non-uniform scale shear).
    pub fn world_rotations(&self, model: &Model) -> Vec<Quat> {
        let count = model.nodes.len();
        let mut out = vec![[0.0, 0.0, 0.0, 1.0]; count];
        let mut done = vec![false; count];
        let mut chain = Vec::new();
        for i in 0..count {
            let mut at = i;
            chain.clear();
            while !done[at] {
                chain.push(at);
                match model.nodes[at].parent {
                    Some(p) if p < count => at = p,
                    _ => break,
                }
            }
            for &n in chain.iter().rev() {
                let local = self.rotation[n];
                out[n] = match model.nodes[n].parent {
                    Some(p) if p < count => normalize_q(mul_q(out[p], local)),
                    _ => normalize_q(local),
                };
                done[n] = true;
            }
        }
        out
    }

    /// Blends from `from` toward `self` by `fade` (`0.0` keeps `from`, `1.0` keeps `self`),
    /// restricted to `mask` when provided.
    pub fn blend_from(&mut self, from: &Pose, fade: f32, mask: Option<&[usize]>) {
        if !fade.is_finite() || fade >= 1.0 {
            return;
        }
        let t = fade.clamp(0.0, 1.0);
        let n_max = self.translation.len().min(from.translation.len());
        let mut apply_node = |n: usize| {
            if n >= n_max {
                return;
            }
            self.translation[n] = math::lerp3(from.translation[n], self.translation[n], t);
            self.rotation[n] = slerp(from.rotation[n], self.rotation[n], t);
            self.scale[n] = math::lerp3(from.scale[n], self.scale[n], t);
        };
        match mask {
            Some(nodes) => {
                for &n in nodes {
                    apply_node(n);
                }
            }
            None => {
                for n in 0..n_max {
                    apply_node(n);
                }
            }
        }
    }

    /// Applies an additive local [`BoneOffset`] to one node.
    pub fn apply_offset(&mut self, offset: &BoneOffset) {
        let n = offset.node;
        if n >= self.translation.len() {
            return;
        }
        if offset.translation.iter().all(|x| x.is_finite()) {
            self.translation[n] = math::add(self.translation[n], offset.translation);
        }
        if offset.rotation.iter().all(|x| x.is_finite()) {
            self.rotation[n] = normalize_q(mul_q(self.rotation[n], normalize_q(offset.rotation)));
        }
        if offset.scale.iter().all(|x| x.is_finite()) {
            self.scale[n] = [
                self.scale[n][0] * offset.scale[0],
                self.scale[n][1] * offset.scale[1],
                self.scale[n][2] * offset.scale[2],
            ];
        }
    }

    /// Turns a chain of joints (such as neck and head) toward `chain.target_model` in model space.
    pub fn apply_look_at(&mut self, model: &Model, chain: &LookAtChain) {
        if !chain.weight.is_finite()
            || chain.weight <= 0.0
            || chain.target_model.iter().any(|x| !x.is_finite())
        {
            return;
        }
        let w = chain.weight.clamp(0.0, 1.0);
        let count = model.nodes.len();
        let Some(&(aim_node, _)) = chain.nodes.iter().rev().find(|&&(n, _)| n < count) else {
            return;
        };
        let globals = self.globals(model);
        let world_rot = self.world_rotations(model);
        let aim_pos = [
            globals[aim_node][12],
            globals[aim_node][13],
            globals[aim_node][14],
        ];
        let to_target = math::sub(chain.target_model, aim_pos);
        let dist = math::length(to_target);
        if dist <= 1e-5 {
            return;
        }
        let local_dir = rotate_vec(
            conjugate_q(world_rot[aim_node]),
            math::scale(to_target, 1.0 / dist),
        );
        let fwd = if math::length(chain.forward) > 1e-5 {
            math::normalize(chain.forward)
        } else {
            [0.0, 0.0, 1.0]
        };
        let up_hint = if math::length(chain.up) > 1e-5 {
            math::normalize(chain.up)
        } else {
            [0.0, 1.0, 0.0]
        };
        let mut right = math::cross(up_hint, fwd);
        if math::length(right) < 1e-5 {
            right = math::cross([1.0, 0.0, 0.0], fwd);
        }
        let right = math::normalize(right);
        let up = math::normalize(math::cross(fwd, right));

        let df = math::dot(local_dir, fwd);
        let dr = math::dot(local_dir, right);
        let du = math::dot(local_dir, up);
        let max_yaw = chain.max_yaw.abs();
        let max_pitch = chain.max_pitch.abs();
        let yaw = dr.atan2(df).clamp(-max_yaw, max_yaw) * w;
        let horiz = (df * df + dr * dr).sqrt();
        let elev = du.atan2(horiz).clamp(-max_pitch, max_pitch) * w;

        for &(node, share) in &chain.nodes {
            if node >= count || !share.is_finite() || share == 0.0 {
                continue;
            }
            let q_yaw = quat_from_axis_angle(up, yaw * share);
            let q_pitch = quat_from_axis_angle(right, -elev * share);
            let delta = mul_q(q_yaw, q_pitch);
            self.rotation[node] = normalize_q(mul_q(self.rotation[node], delta));
        }
    }

    /// Analytical Two-Bone IK solver (`root -> mid -> end`) bending toward `ik.pole_model`.
    pub fn solve_two_bone_ik(&mut self, model: &Model, ik: &TwoBoneIk) {
        let count = model.nodes.len();
        if ik.root >= count
            || ik.mid >= count
            || ik.end >= count
            || ik.root == ik.mid
            || ik.mid == ik.end
            || !ik.weight.is_finite()
            || ik.weight <= 0.0
            || ik.target_model.iter().any(|x| !x.is_finite())
            || ik.pole_model.iter().any(|x| !x.is_finite())
        {
            return;
        }
        let w = ik.weight.clamp(0.0, 1.0);
        let g0 = self.globals(model);
        let wr0 = self.world_rotations(model);
        let pos_of = |g: &[M4], n: usize| -> V3 { [g[n][12], g[n][13], g[n][14]] };
        let a = pos_of(&g0, ik.root);
        let b = pos_of(&g0, ik.mid);
        let c = pos_of(&g0, ik.end);
        let ab = math::sub(b, a);
        let bc = math::sub(c, b);
        let lab = math::length(ab);
        let lbc = math::length(bc);
        if lab <= 1e-5 || lbc <= 1e-5 {
            return;
        }

        let at = math::sub(ik.target_model, a);
        let raw_dist = math::length(at);
        let u_at = if raw_dist > 1e-5 {
            math::scale(at, 1.0 / raw_dist)
        } else {
            let ac = math::sub(c, a);
            if math::length(ac) > 1e-5 {
                math::normalize(ac)
            } else {
                [0.0, 1.0, 0.0]
            }
        };
        let min_reach = (lab - lbc).abs() + 1e-4;
        let max_reach = (lab + lbc - 1e-4).max(min_reach);
        let lat = raw_dist.clamp(min_reach, max_reach);

        let ap = math::sub(ik.pole_model, a);
        let mut plane_n = math::cross(u_at, ap);
        if math::length(plane_n) < 1e-5 {
            plane_n = math::cross(u_at, ab);
        }
        if math::length(plane_n) < 1e-5 {
            plane_n = math::cross(u_at, [0.0, 0.0, 1.0]);
        }
        if math::length(plane_n) < 1e-5 {
            plane_n = math::cross(u_at, [1.0, 0.0, 0.0]);
        }
        let plane_n = math::normalize(plane_n);
        let u_pole = math::normalize(math::cross(plane_n, u_at));

        let cos_a = ((lab * lab + lat * lat - lbc * lbc) / (2.0 * lab * lat)).clamp(-1.0, 1.0);
        let sin_a = (1.0 - cos_a * cos_a).max(0.0).sqrt();
        let b_des = math::add(
            a,
            math::add(
                math::scale(u_at, lab * cos_a),
                math::scale(u_pole, lab * sin_a),
            ),
        );
        let c_des = math::add(a, math::scale(u_at, lat));

        let dq_root = quat_from_vectors(ab, math::sub(b_des, a));
        let w_root_des = normalize_q(mul_q(dq_root, wr0[ik.root]));
        let parent_root = model.nodes[ik.root]
            .parent
            .filter(|&p| p < count)
            .map_or([0.0, 0.0, 0.0, 1.0], |p| wr0[p]);
        let local_root_des = normalize_q(mul_q(conjugate_q(parent_root), w_root_des));
        let orig_root = self.rotation[ik.root];
        let orig_mid = self.rotation[ik.mid];
        self.rotation[ik.root] = local_root_des;

        let g1 = self.globals(model);
        let wr1 = self.world_rotations(model);
        let b1 = pos_of(&g1, ik.mid);
        let c1 = pos_of(&g1, ik.end);
        let dq_mid = quat_from_vectors(math::sub(c1, b1), math::sub(c_des, b1));
        let w_mid_des = normalize_q(mul_q(dq_mid, wr1[ik.mid]));
        let parent_mid = model.nodes[ik.mid]
            .parent
            .filter(|&p| p < count)
            .map_or([0.0, 0.0, 0.0, 1.0], |p| wr1[p]);
        let local_mid_des = normalize_q(mul_q(conjugate_q(parent_mid), w_mid_des));

        if w < 1.0 {
            self.rotation[ik.root] = slerp(orig_root, local_root_des, w);
            self.rotation[ik.mid] = slerp(orig_mid, local_mid_des, w);
        } else {
            self.rotation[ik.mid] = local_mid_des;
        }
    }
}

/// Evaluates an entire skeletal pose stack in a single pass: optional rest reset, clip layer
/// sampling/blending, Hermite cross-fade from `from_pose`, Look-At chains, Two-Bone IK chains,
/// additive bone offsets, and morph weight overrides.
pub fn pose_model_clip(pose: &mut Pose, from_pose: Option<&Pose>, model: &Model, spec: &PoseSpec) {
    if spec.reset_to_rest {
        *pose = Pose::rest(model);
    }
    for layer in spec.clips {
        pose.apply(model, layer.clip, layer.time, layer.weight);
    }
    if let (Some(from), Some((elapsed, duration, mask))) = (from_pose, spec.crossfade) {
        let fade = hermite_fade(elapsed, duration);
        pose.blend_from(from, fade, mask);
    }
    for chain in spec.look_at {
        pose.apply_look_at(model, chain);
    }
    for ik in spec.two_bone_ik {
        pose.solve_two_bone_ik(model, ik);
    }
    for offset in spec.bone_offsets {
        pose.apply_offset(offset);
    }
    for ov in spec.morph_overrides {
        if ov.weight.is_finite()
            && let Some(slot) = pose
                .weights
                .get_mut(ov.node)
                .and_then(|w| w.get_mut(ov.target))
        {
            *slot = ov.weight;
        }
    }
}
