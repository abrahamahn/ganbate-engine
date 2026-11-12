//! Run-time glTF/GLB models, clip evaluation, Hermite cross-fade, Look-At, Two-Bone IK, and
//! morph target overrides.

use super::{Runtime, bytes, floats, matrix, with, words, write_floats};
use crate::{give, say};
use ganbate_engine::anim::{
    BoneOffset, ClipLayer, LookAtChain, MorphOverride, PoseSpec, TwoBoneIk, normalize_q,
    pose_model_clip,
};
use ganbate_engine::assets::model::{self, Model};
use ganbate_engine::render::{ModelId, ModelInstance, PropInstance, PropTexture};

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A model's contents as JSON, for the host: images to decode (type, colour or data, size),
/// nodes (name, parent, mesh), meshes (name, morph target names), clips (name, duration and the
/// nodes they animate) and material names. Indices are positions in these lists.
fn describe(m: &Model) -> String {
    let list = |items: Vec<String>| format!("[{}]", items.join(","));
    let opt = |o: Option<usize>| o.map_or("null".to_string(), |v| v.to_string());
    let images = m
        .images
        .iter()
        .map(|i| {
            format!(
                r#"{{"name":{},"mime":{},"srgb":{},"size":{}}}"#,
                json_string(&i.name),
                json_string(&i.mime),
                i.srgb,
                i.data.len()
            )
        })
        .collect();
    let nodes = m
        .nodes
        .iter()
        .map(|n| {
            format!(
                r#"{{"name":{},"parent":{},"mesh":{},"skin":{}}}"#,
                json_string(&n.name),
                opt(n.parent),
                opt(n.mesh),
                opt(n.skin)
            )
        })
        .collect();
    let meshes = m
        .meshes
        .iter()
        .map(|me| {
            format!(
                r#"{{"name":{},"targets":{},"targetNames":{}}}"#,
                json_string(&me.name),
                me.weights.len(),
                list(me.target_names.iter().map(|t| json_string(t)).collect())
            )
        })
        .collect();
    let clips = m
        .clips
        .iter()
        .map(|c| {
            // Each node a channel moves, once, in the order first met.
            let mut nodes: Vec<usize> = Vec::new();
            for channel in &c.channels {
                if !nodes.contains(&channel.node) {
                    nodes.push(channel.node);
                }
            }
            format!(
                r#"{{"name":{},"duration":{},"nodes":{}}}"#,
                json_string(&c.name),
                c.duration,
                list(nodes.iter().map(usize::to_string).collect())
            )
        })
        .collect();
    let materials = m.materials.iter().map(|x| json_string(&x.name)).collect();
    format!(
        r#"{{"images":{},"nodes":{},"meshes":{},"clips":{},"materials":{}}}"#,
        list(images),
        list(nodes),
        list(meshes),
        list(clips),
        list(materials)
    )
}

fn pending(rt: &mut Runtime, id: u32) -> Result<&mut Model, String> {
    rt.pending
        .get_mut((id as usize).wrapping_sub(1))
        .and_then(Option::as_mut)
        .ok_or_else(|| format!("no parsed model {id}"))
}

/// Parses a GLB model (`len` bytes at `ptr`, read during the call). Returns a parsed-model
/// handle for [`ganbate_rt_model_describe`], [`ganbate_rt_model_image`] and [`ganbate_rt_model_create`].
///
/// # Safety
/// `ptr` must address `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_parse(ptr: *const u8, len: usize) -> u32 {
    // SAFETY: the caller passes a readable range.
    let data = unsafe { bytes(ptr, len) };
    with(0, |rt| {
        let m = model::load_glb(data).map_err(|e| format!("model: {e}"))?;
        let slot = rt.pending.iter().position(Option::is_none);
        let i = match slot {
            Some(i) => {
                rt.pending[i] = Some(m);
                i
            }
            None => {
                rt.pending.push(Some(m));
                rt.pending.len() - 1
            }
        };
        Ok(i as u32 + 1)
    })
}

/// Writes a parsed model's description (JSON, see `describe`) to `out` if it fits in `cap`
/// bytes; returns its length.
///
/// # Safety
/// `out` must address `cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_describe(parsed: u32, out: *mut u8, cap: usize) -> u32 {
    with(0, |rt| {
        let text = describe(pending(rt, parsed)?);
        // SAFETY: forwarded.
        Ok(unsafe { give(text.as_bytes(), out, cap) })
    })
}

/// Copies image `index` of a parsed model (encoded PNG, JPEG or WebP) to `out` if it fits in
/// `cap` bytes; returns its length.
///
/// # Safety
/// `out` must address `cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_image(
    parsed: u32,
    index: u32,
    out: *mut u8,
    cap: usize,
) -> u32 {
    with(0, |rt| {
        let m = pending(rt, parsed)?;
        let image = m
            .images
            .get(index as usize)
            .ok_or_else(|| format!("no image {index}"))?;
        // SAFETY: forwarded.
        Ok(unsafe { give(&image.data, out, cap) })
    })
}

/// Uploads a parsed model with its textures (`count` texture handles at `textures`, one per
/// image, 0 where an image could not be decoded). Returns the model handle; the parsed handle
/// is spent either way.
///
/// # Safety
/// `textures` must address `count` words.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_create(
    parsed: u32,
    textures: *const u8,
    count: usize,
) -> u32 {
    // SAFETY: the caller passes `count` words.
    let tex: Vec<Option<PropTexture>> = unsafe { words(textures, count) }
        .into_iter()
        .map(|t| (t != 0).then(|| PropTexture::from_raw(t)))
        .collect();
    with(0, |rt| {
        let slot = (parsed as usize).wrapping_sub(1);
        let m = rt
            .pending
            .get_mut(slot)
            .and_then(Option::take)
            .ok_or_else(|| format!("no parsed model {parsed}"))?;
        Ok(rt.renderer.create_model(&mut rt.device, m, &tex)?.raw())
    })
}

/// The prop material made for the model's material `index` (to restyle it).
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_material(model: u32, index: u32) -> u32 {
    with(0, |rt| {
        Ok(rt
            .renderer
            .model_material(ModelId::from_raw(model), index as usize)?
            .raw())
    })
}

/// Destroys a model and its instances (textures stay the host's).
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_destroy(model: u32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .destroy_model(&mut rt.device, ModelId::from_raw(model));
        Ok(1)
    })
}

/// Places a model at the column-major matrix (16 floats) at `matrix`, in its rest pose.
///
/// # Safety
/// `matrix` must address 64 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_instance(model: u32, matrix: *const u8) -> u32 {
    // SAFETY: the caller passes 16 floats.
    let m = unsafe { floats(matrix, 16) };
    with(0, |rt| {
        let i = rt.renderer.create_model_instance(
            &mut rt.device,
            ModelId::from_raw(model),
            &self::matrix(&m)?,
        )?;
        Ok(i.raw())
    })
}

/// Moves a model instance and shows (`visible` 1) or hides it.
///
/// # Safety
/// `matrix` must address 64 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_instance_set(
    instance: u32,
    matrix: *const u8,
    visible: u32,
) -> u32 {
    // SAFETY: the caller passes 16 floats.
    let m = unsafe { floats(matrix, 16) };
    with(0, |rt| {
        rt.renderer.set_model_instance(
            ModelInstance::from_raw(instance),
            &self::matrix(&m)?,
            visible != 0,
        )?;
        Ok(1)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_instance_destroy(instance: u32) -> u32 {
    with(0, |rt| {
        rt.crossfade_from.remove(&instance);
        rt.renderer
            .destroy_model_instance(&mut rt.device, ModelInstance::from_raw(instance));
        Ok(1)
    })
}

/// Snapshots `instance`'s current pose inside Rust WASM memory so subsequent calls to
/// [`ganbate_rt_pose_model_clip`] can Hermite cross-fade from it without JS joint copies.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_snapshot_pose(instance: u32) -> u32 {
    with(0, |rt| {
        let (pose, _) = rt
            .renderer
            .model_pose_and_model(ModelInstance::from_raw(instance))?;
        let snap = pose.clone();
        rt.crossfade_from.insert(instance, snap);
        Ok(1)
    })
}

/// Evaluates skeletal clip layers, Hermite cross-fade from [`ganbate_rt_model_snapshot_pose`],
/// Look-At chains, analytical Two-Bone IK, additive bone offsets, morph overrides, and optional
/// GPU skinning/morph commit in a single WASM call.
///
/// Packet layout (`float_count` little-endian `f32` values at `cmd_ptr`):
/// - Header (`9` floats): `[flags, clip_count, fade_elapsed_s, fade_duration_s, fade_mask_count,
///   look_at_count, ik_count, offset_count, morph_count]`
///   where `flags` bits are `1` (`reset_to_rest`), `2` (`commit_update`), `4` (`use_crossfade`).
/// - `clip_count * 3`: `[clip_idx, time_s, weight]`
/// - `fade_mask_count`: `[node_idx]`
/// - `look_at_count`: `[joint_count, tx, ty, tz, fx, fy, fz, ux, uy, uz, max_yaw, max_pitch,
///   weight, (node_idx, share) * joint_count]`
/// - `ik_count * 10`: `[root, mid, end, tx, ty, tz, px, py, pz, weight]`
/// - `offset_count * 11`: `[node, tx, ty, tz, qx, qy, qz, qw, sx, sy, sz]`
/// - `morph_count * 3`: `[node, target, weight]`
///
/// # Safety
/// `cmd_ptr` must address `float_count * 4` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_pose_model_clip(
    instance: u32,
    cmd_ptr: *const u8,
    float_count: usize,
) -> u32 {
    if float_count < 9 {
        say(2, "pose_model_clip: packet needs at least 9 header floats");
        return 0;
    }
    // SAFETY: forwarded from caller.
    let f = unsafe { floats(cmd_ptr, float_count) };
    with(0, |rt| {
        if f.iter().any(|x| !x.is_finite()) {
            return Err("pose_model_clip: non-finite packet value".into());
        }
        let flags = f[0] as u32;
        let clip_count = f[1] as usize;
        let fade_elapsed = f[2];
        let fade_duration = f[3];
        let mask_count = f[4] as usize;
        let look_at_count = f[5] as usize;
        let ik_count = f[6] as usize;
        let offset_count = f[7] as usize;
        let morph_count = f[8] as usize;

        let mut cursor = 9usize;
        let take = |cur: &mut usize, n: usize| -> Result<&[f32], String> {
            let end = cur
                .checked_add(n)
                .filter(|&e| e <= f.len())
                .ok_or_else(|| "pose_model_clip: truncated command packet".to_string())?;
            let slice = &f[*cur..end];
            *cur = end;
            Ok(slice)
        };

        let raw_clips = take(&mut cursor, clip_count * 3)?;
        let clips: Vec<ClipLayer> = raw_clips
            .chunks_exact(3)
            .map(|c| ClipLayer {
                clip: c[0] as usize,
                time: c[1],
                weight: c[2],
            })
            .collect();

        let raw_mask = take(&mut cursor, mask_count)?;
        let mask: Vec<usize> = raw_mask.iter().map(|&n| n as usize).collect();

        let mut look_at = Vec::with_capacity(look_at_count);
        for _ in 0..look_at_count {
            let hdr = take(&mut cursor, 13)?;
            let joints_n = hdr[0] as usize;
            let raw_joints = take(&mut cursor, joints_n * 2)?;
            let nodes = raw_joints
                .chunks_exact(2)
                .map(|j| (j[0] as usize, j[1]))
                .collect();
            look_at.push(LookAtChain {
                nodes,
                target_model: [hdr[1], hdr[2], hdr[3]],
                forward: [hdr[4], hdr[5], hdr[6]],
                up: [hdr[7], hdr[8], hdr[9]],
                max_yaw: hdr[10],
                max_pitch: hdr[11],
                weight: hdr[12],
            });
        }

        let raw_ik = take(&mut cursor, ik_count * 10)?;
        let two_bone_ik: Vec<TwoBoneIk> = raw_ik
            .chunks_exact(10)
            .map(|k| TwoBoneIk {
                root: k[0] as usize,
                mid: k[1] as usize,
                end: k[2] as usize,
                target_model: [k[3], k[4], k[5]],
                pole_model: [k[6], k[7], k[8]],
                weight: k[9],
            })
            .collect();

        let raw_offsets = take(&mut cursor, offset_count * 11)?;
        let bone_offsets: Vec<BoneOffset> = raw_offsets
            .chunks_exact(11)
            .map(|o| BoneOffset {
                node: o[0] as usize,
                translation: [o[1], o[2], o[3]],
                rotation: [o[4], o[5], o[6], o[7]],
                scale: [o[8], o[9], o[10]],
            })
            .collect();

        let raw_morphs = take(&mut cursor, morph_count * 3)?;
        let morph_overrides: Vec<MorphOverride> = raw_morphs
            .chunks_exact(3)
            .map(|m| MorphOverride {
                node: m[0] as usize,
                target: m[1] as usize,
                weight: m[2],
            })
            .collect();

        let snap = if (flags & 4) != 0 {
            rt.crossfade_from.get(&instance).cloned()
        } else {
            None
        };
        let crossfade = ((flags & 4) != 0).then_some((
            fade_elapsed,
            fade_duration,
            (!mask.is_empty()).then_some(mask.as_slice()),
        ));

        let handle = ModelInstance::from_raw(instance);
        {
            let (pose, model) = rt.renderer.model_pose_and_model(handle)?;
            pose_model_clip(
                pose,
                snap.as_ref(),
                model,
                &PoseSpec {
                    reset_to_rest: (flags & 1) != 0,
                    clips: &clips,
                    crossfade,
                    look_at: &look_at,
                    two_bone_ik: &two_bone_ik,
                    bone_offsets: &bone_offsets,
                    morph_overrides: &morph_overrides,
                },
            );
        }
        if (flags & 2) != 0 {
            rt.renderer.update_model_instance(&mut rt.device, handle)?;
        }
        Ok(1)
    })
}

/// Resets an instance's pose to the model's rest pose.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_rest(instance: u32) -> u32 {
    with(0, |rt| {
        let (pose, model) = rt
            .renderer
            .model_pose_and_model(ModelInstance::from_raw(instance))?;
        *pose = ganbate_engine::anim::Pose::rest(model);
        Ok(1)
    })
}

/// Mixes clip `clip` at `time` seconds into the instance's pose with `weight` (1 replaces what
/// the clip animates).
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_clip(instance: u32, clip: u32, time: f32, weight: f32) -> u32 {
    with(0, |rt| {
        let (pose, model) = rt
            .renderer
            .model_pose_and_model(ModelInstance::from_raw(instance))?;
        if clip as usize >= model.clips.len() {
            return Err(format!("no clip {clip}"));
        }
        pose.apply(model, clip as usize, time, weight);
        Ok(1)
    })
}

/// Writes node `node`'s local transform as 10 floats at `out`: translation (3), rotation
/// quaternion x, y, z, w (4), scale (3).
///
/// # Safety
/// `out` must address 40 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_node_get(instance: u32, node: u32, out: *mut u8) -> u32 {
    with(0, |rt| {
        let (pose, _) = rt
            .renderer
            .model_pose_and_model(ModelInstance::from_raw(instance))?;
        let n = node as usize;
        if n >= pose.translation.len() {
            return Err(format!("no node {node}"));
        }
        let mut v = [0.0f32; 10];
        v[..3].copy_from_slice(&pose.translation[n]);
        v[3..7].copy_from_slice(&pose.rotation[n]);
        v[7..].copy_from_slice(&pose.scale[n]);
        // SAFETY: the caller passes 40 writable bytes.
        unsafe { write_floats(out, &v) };
        Ok(1)
    })
}

/// Sets node `node`'s local transform from 10 floats at `ptr` (as `ganbate_rt_model_node_get`
/// writes them; the rotation is normalized).
///
/// # Safety
/// `ptr` must address 40 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_node_set(
    instance: u32,
    node: u32,
    ptr: *const u8,
) -> u32 {
    // SAFETY: the caller passes 10 floats.
    let v = unsafe { floats(ptr, 10) };
    with(0, |rt| {
        if v.iter().any(|x| !x.is_finite()) {
            return Err("model_node_set: values must be finite".into());
        }
        let (pose, _) = rt
            .renderer
            .model_pose_and_model(ModelInstance::from_raw(instance))?;
        let n = node as usize;
        if n >= pose.translation.len() {
            return Err(format!("no node {node}"));
        }
        pose.translation[n] = [v[0], v[1], v[2]];
        pose.rotation[n] = normalize_q([v[3], v[4], v[5], v[6]]);
        pose.scale[n] = [v[7], v[8], v[9]];
        Ok(1)
    })
}

/// Sets morph target `target` of node `node`'s mesh to `weight`.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_morph(
    instance: u32,
    node: u32,
    target: u32,
    weight: f32,
) -> u32 {
    with(0, |rt| {
        let (pose, _) = rt
            .renderer
            .model_pose_and_model(ModelInstance::from_raw(instance))?;
        let w = pose
            .weights
            .get_mut(node as usize)
            .and_then(|w| w.get_mut(target as usize))
            .ok_or_else(|| format!("node {node} has no morph target {target}"))?;
        if !weight.is_finite() {
            return Err("model_morph: the weight must be finite".into());
        }
        *w = weight;
        Ok(1)
    })
}

/// Applies the instance's pose: node transforms, skinning and morphing.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_update(instance: u32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .update_model_instance(&mut rt.device, ModelInstance::from_raw(instance))?;
        Ok(1)
    })
}

/// Writes node `node`'s world matrix (16 floats, as of the last update) at `out`.
///
/// # Safety
/// `out` must address 64 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_node_world(
    instance: u32,
    node: u32,
    out: *mut u8,
) -> u32 {
    with(0, |rt| {
        let m = rt
            .renderer
            .model_node_world(ModelInstance::from_raw(instance), node as usize)?;
        // SAFETY: the caller passes 64 writable bytes.
        unsafe { write_floats(out, &m) };
        Ok(1)
    })
}

/// Shows (`visible` 1) or hides the meshes of node `node`.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_node_visible(instance: u32, node: u32, visible: u32) -> u32 {
    with(0, |rt| {
        rt.renderer.set_model_node_visible(
            ModelInstance::from_raw(instance),
            node as usize,
            visible != 0,
        )?;
        Ok(1)
    })
}

/// Tints every part of a model instance.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_model_tint(instance: u32, r: f32, g: f32, b: f32, a: f32) -> u32 {
    with(0, |rt| {
        rt.renderer
            .set_model_instance_tint(ModelInstance::from_raw(instance), [r, g, b, a])?;
        Ok(1)
    })
}

/// The model instance a picked prop instance belongs to (0 for none); writes the node it draws
/// as a word at `out`.
///
/// # Safety
/// `out` must address 4 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_model_owner(prop: u32, out: *mut u8) -> u32 {
    with(0, |rt| {
        let Some((i, node)) = rt.renderer.model_part_owner(PropInstance::from_raw(prop)) else {
            return Ok(0);
        };
        let b = (node as u32).to_le_bytes();
        // SAFETY: the caller passes 4 writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), out, 4) };
        Ok(i.raw())
    })
}
