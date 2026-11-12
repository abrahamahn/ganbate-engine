/**
 * Character morph binding, single-call pose packet encoding, and procedural rig evaluation for
 * Ganbate model instances (`ganbate_rt_pose_model_clip`).
 */

/** Bind named glTF morphs to one Ganbate model instance. Sharing a model does not share its pose.
 * Pass only the channels this editor owns. Omitted weights leave the current pose alone.
 * Call apply with force after another producer (for example an animation clip) writes the pose.
 */
export function bindCharacterShapes(engine, instance, info, ranges) {
  const bindings = new Map();
  for (const [node, description] of info.nodes.entries()) {
    if (description.mesh == null) continue;
    const mesh = info.meshes[description.mesh];
    if (!mesh) continue;
    for (const [target, name] of mesh.targetNames.entries()) {
      if (!Object.hasOwn(ranges, name) || target >= mesh.targets) continue;
      const [min, max] = ranges[name];
      if (!Number.isFinite(min) || !Number.isFinite(max) || min > max) {
        throw new RangeError('Invalid morph range for ' + name);
      }
      let group = bindings.get(name);
      if (!group) bindings.set(name, group = { min, max, slots: [] });
      group.slots.push([node, target]);
    }
  }
  const applied = new Map();
  return {
    available: Object.freeze([...bindings.keys()]),
    apply(weights, force = false) {
      const pending = [];
      for (const [name, raw] of Object.entries(weights)) {
        const group = bindings.get(name);
        if (!group) continue;
        if (!Number.isFinite(raw)) throw new TypeError('Invalid morph weight for ' + name);
        const value = Math.min(group.max, Math.max(group.min, raw));
        if (force || value !== applied.get(name)) pending.push({ name, group, value });
      }
      if (!pending.length) return false;
      for (const { group, value } of pending) {
        for (const [node, target] of group.slots) engine.modelMorph(instance, node, target, value);
      }
      engine.updateModel(instance);
      for (const { name, value } of pending) applied.set(name, value);
      return true;
    },
  };
}

/** C1-continuous Hermite cross-fade weight (`3t^2 - 2t^3`), clamped to `0..1`. */
export function hermiteFade(elapsedSeconds, durationSeconds) {
  if (!Number.isFinite(elapsedSeconds) || !Number.isFinite(durationSeconds) ||
      durationSeconds <= 0 || elapsedSeconds >= durationSeconds) {
    return 1;
  }
  if (elapsedSeconds <= 0) return 0;
  const t = Math.min(1, Math.max(0, elapsedSeconds / durationSeconds));
  return t * t * (3 - 2 * t);
}

/**
 * Converts weighted `(clip, time, weight)` entries into sequential `ClipLayer` cascade weights
 * (`alpha_m = w_m / sum_{i<=m} w_i`) so `poseModelClip` reproduces exact multi-way convex blends.
 */
export function weightsToClipLayers(entries) {
  let running = 0;
  const layers = [];
  for (const entry of entries ?? []) {
    const w = entry.weight ?? 0;
    if (!Number.isFinite(w) || w <= 1e-6) continue;
    running += w;
    layers.push({
      clip: entry.clip,
      time: entry.time ?? 0,
      weight: layers.length === 0 ? 1 : Math.min(1, Math.max(0, w / running)),
    });
  }
  return layers;
}

/** Evaluates a 1D parametric blend space (`[{ clip, position, rate? }]`) into `ClipLayer`s. */
export function sampleBlendSpace1D(samples, param, baseTime = 0) {
  if (!Array.isArray(samples) || !samples.length || !Number.isFinite(param)) return [];
  const sorted = samples
    .filter((s) => Number.isFinite(s?.position))
    .slice()
    .sort((a, b) => a.position - b.position);
  if (!sorted.length) return [];
  if (sorted.length === 1 || param <= sorted[0].position) {
    const s = sorted[0];
    return [{ clip: s.clip, time: baseTime * (s.rate ?? 1), weight: 1 }];
  }
  const last = sorted[sorted.length - 1];
  if (param >= last.position) {
    return [{ clip: last.clip, time: baseTime * (last.rate ?? 1), weight: 1 }];
  }
  for (let i = 0; i < sorted.length - 1; i++) {
    const a = sorted[i];
    const b = sorted[i + 1];
    if (param <= b.position) {
      const span = b.position - a.position;
      const t = span > 1e-6 ? Math.min(1, Math.max(0, (param - a.position) / span)) : 0;
      return weightsToClipLayers([
        { clip: a.clip, time: baseTime * (a.rate ?? 1), weight: 1 - t },
        { clip: b.clip, time: baseTime * (b.rate ?? 1), weight: t },
      ]);
    }
  }
  return [];
}

/**
 * Evaluates a 2D freeform Cartesian blend space (`[{ clip, position: [x, y], rate? }]`) via
 * Gradient Band Interpolation into sequential `ClipLayer`s.
 */
export function sampleBlendSpace2D(samples, param, baseTime = 0) {
  if (!Array.isArray(samples) || !samples.length || !Array.isArray(param) ||
      !Number.isFinite(param[0]) || !Number.isFinite(param[1])) {
    return [];
  }
  if (samples.length === 1) {
    const s = samples[0];
    return [{ clip: s.clip, time: baseTime * (s.rate ?? 1), weight: 1 }];
  }
  const raw = new Array(samples.length).fill(0);
  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    const si = samples[i];
    let influence = 1;
    for (let j = 0; j < samples.length; j++) {
      if (i === j) continue;
      const sj = samples[j];
      const vx = sj.position[0] - si.position[0];
      const vy = sj.position[1] - si.position[1];
      const denom = vx * vx + vy * vy;
      if (denom <= 1e-8) continue;
      const px = param[0] - si.position[0];
      const py = param[1] - si.position[1];
      const hij = Math.min(1, Math.max(0, 1 - (px * vx + py * vy) / denom));
      if (hij < influence) influence = hij;
    }
    raw[i] = influence;
    sum += influence;
  }
  if (sum <= 1e-6) {
    let best = samples[0];
    let bestDist = Infinity;
    for (const s of samples) {
      const d = (param[0] - s.position[0]) ** 2 + (param[1] - s.position[1]) ** 2;
      if (d < bestDist) {
        bestDist = d;
        best = s;
      }
    }
    return [{ clip: best.clip, time: baseTime * (best.rate ?? 1), weight: 1 }];
  }
  return weightsToClipLayers(
    samples.map((s, idx) => ({
      clip: s.clip,
      time: baseTime * (s.rate ?? 1),
      weight: raw[idx] / sum,
    })),
  );
}

export function normalizeQuat(q) {
  const len = Math.hypot(q[0], q[1], q[2], q[3]);
  return len > 1e-12 && Number.isFinite(len)
    ? [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
    : [0, 0, 0, 1];
}

export function mulQuat(a, b) {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}

export function slerpQuat(a, b, t) {
  let bx = b[0], by = b[1], bz = b[2], bw = b[3];
  let dot = a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw;
  if (dot < 0) {
    dot = -dot;
    bx = -bx; by = -by; bz = -bz; bw = -bw;
  }
  let wa = 1 - t;
  let wb = t;
  if (dot < 0.9995) {
    const theta = Math.acos(Math.min(1, Math.max(-1, dot)));
    const sin = Math.sin(theta);
    wa = Math.sin((1 - t) * theta) / sin;
    wb = Math.sin(t * theta) / sin;
  }
  return normalizeQuat([
    a[0] * wa + bx * wb,
    a[1] * wa + by * wb,
    a[2] * wa + bz * wb,
    a[3] * wa + bw * wb,
  ]);
}

export function quatFromAxisAngle(axis, angle) {
  const len = Math.hypot(axis[0], axis[1], axis[2]);
  if (len <= 1e-8 || !Number.isFinite(angle)) return [0, 0, 0, 1];
  const s = Math.sin(angle * 0.5) / len;
  return normalizeQuat([axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle * 0.5)]);
}

/** Euler `(pitch, yaw, roll)` in radians: roll about Z first, then pitch about X, then yaw about Y. */
export function quatFromEuler(pitch, yaw, roll) {
  return normalizeQuat(mulQuat(
    mulQuat(quatFromAxisAngle([0, 1, 0], yaw), quatFromAxisAngle([1, 0, 0], pitch)),
    quatFromAxisAngle([0, 0, 1], roll),
  ));
}

export function rotateVec3(q, v) {
  const tx = 2 * (q[1] * v[2] - q[2] * v[1]);
  const ty = 2 * (q[2] * v[0] - q[0] * v[2]);
  const tz = 2 * (q[0] * v[1] - q[1] * v[0]);
  return [
    v[0] + q[3] * tx + (q[1] * tz - q[2] * ty),
    v[1] + q[3] * ty + (q[2] * tx - q[0] * tz),
    v[2] + q[3] * tz + (q[0] * ty - q[1] * tx),
  ];
}

/**
 * Encodes a `PoseSpec` into the flat `Float32Array` command buffer consumed by
 * `ganbate_rt_pose_model_clip`.
 */
export function encodePoseCommandPacket(spec = {}) {
  const resetToRest = spec.resetToRest !== false;
  const commitUpdate = spec.commitUpdate !== false;
  const crossfade = spec.crossfade ?? null;
  const clips = spec.clips ?? [];
  const mask = crossfade?.mask ?? [];
  const lookAt = spec.lookAt ?? [];
  const twoBoneIk = spec.twoBoneIk ?? [];
  const boneOffsets = spec.boneOffsets ?? [];
  const morphOverrides = spec.morphOverrides ?? [];

  let flags = 0;
  if (resetToRest) flags |= 1;
  if (commitUpdate) flags |= 2;
  if (crossfade && Number.isFinite(crossfade.duration) && crossfade.duration > 0) flags |= 4;

  let lookAtFloats = 0;
  for (const chain of lookAt) {
    lookAtFloats += 13 + (chain.nodes?.length ?? 0) * 2;
  }
  const total = 9
    + clips.length * 3
    + mask.length
    + lookAtFloats
    + twoBoneIk.length * 10
    + boneOffsets.length * 11
    + morphOverrides.length * 3;

  const out = new Float32Array(total);
  out[0] = flags;
  out[1] = clips.length;
  out[2] = crossfade?.elapsed ?? 0;
  out[3] = crossfade?.duration ?? 0;
  out[4] = mask.length;
  out[5] = lookAt.length;
  out[6] = twoBoneIk.length;
  out[7] = boneOffsets.length;
  out[8] = morphOverrides.length;

  let at = 9;
  for (const c of clips) {
    out[at++] = c.clip;
    out[at++] = c.time;
    out[at++] = c.weight ?? 1;
  }
  for (const node of mask) {
    out[at++] = node;
  }
  for (const chain of lookAt) {
    const nodes = chain.nodes ?? [];
    const target = chain.targetModel ?? [0, 1.5, 2];
    const forward = chain.forward ?? [0, 0, 1];
    const up = chain.up ?? [0, 1, 0];
    out[at++] = nodes.length;
    out[at++] = target[0]; out[at++] = target[1]; out[at++] = target[2];
    out[at++] = forward[0]; out[at++] = forward[1]; out[at++] = forward[2];
    out[at++] = up[0]; out[at++] = up[1]; out[at++] = up[2];
    out[at++] = chain.maxYaw ?? 1.05;
    out[at++] = chain.maxPitch ?? 0.70;
    out[at++] = chain.weight ?? 1;
    for (const [node, share] of nodes) {
      out[at++] = node;
      out[at++] = share;
    }
  }
  for (const ik of twoBoneIk) {
    const target = ik.targetModel ?? [0, 0, 0];
    const pole = ik.poleModel ?? [0, 1, 1];
    out[at++] = ik.root;
    out[at++] = ik.mid;
    out[at++] = ik.end;
    out[at++] = target[0]; out[at++] = target[1]; out[at++] = target[2];
    out[at++] = pole[0]; out[at++] = pole[1]; out[at++] = pole[2];
    out[at++] = ik.weight ?? 1;
  }
  for (const off of boneOffsets) {
    const t = off.translation ?? [0, 0, 0];
    const r = off.rotation
      ?? (off.euler ? quatFromEuler(off.euler[0], off.euler[1], off.euler[2]) : [0, 0, 0, 1]);
    const s = off.scale ?? [1, 1, 1];
    out[at++] = off.node;
    out[at++] = t[0]; out[at++] = t[1]; out[at++] = t[2];
    out[at++] = r[0]; out[at++] = r[1]; out[at++] = r[2]; out[at++] = r[3];
    out[at++] = s[0]; out[at++] = s[1]; out[at++] = s[2];
  }
  for (const m of morphOverrides) {
    out[at++] = m.node;
    out[at++] = m.target;
    out[at++] = m.weight;
  }
  return out;
}

function lerpVec3(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

/**
 * Evaluates a complete skeletal pose stack on `instance`. Uses the 1-call WASM path when
 * `engine.poseModelClip` is present, and falls back deterministically to the standard model API.
 */
export function poseModelClip(engine, instance, spec = {}, options = {}) {
  const packet = encodePoseCommandPacket(spec);
  if (typeof engine.poseModelClip === 'function') {
    engine.poseModelClip(instance, packet);
    return packet;
  }

  if (spec.resetToRest !== false && typeof engine.modelRest === 'function') {
    engine.modelRest(instance);
  }
  for (const layer of spec.clips ?? []) {
    engine.modelClip(instance, layer.clip, layer.time, layer.weight ?? 1);
  }
  const cf = spec.crossfade;
  const fromPose = options.fromPose ?? null;
  if (cf && fromPose && cf.duration > 0) {
    const fade = hermiteFade(cf.elapsed ?? 0, cf.duration);
    if (fade < 1) {
      const nodes = cf.mask?.length ? cf.mask : [...fromPose.keys()];
      for (const node of nodes) {
        const prev = fromPose.get(node);
        if (!prev) continue;
        const cur = engine.modelNode(instance, node);
        engine.setModelNode(instance, node, {
          translation: lerpVec3(prev.translation, cur.translation, fade),
          rotation: slerpQuat(prev.rotation, cur.rotation, fade),
          scale: lerpVec3(prev.scale, cur.scale, fade),
        });
      }
    }
  }
  for (const chain of spec.lookAt ?? []) {
    const weight = Math.min(1, Math.max(0, chain.weight ?? 1));
    if (weight <= 0 || !chain.nodes?.length) continue;
    const target = chain.targetModel ?? [0, 1.5, 2];
    const origin = chain.originModel ?? [0, 1.5, 0];
    const dx = target[0] - origin[0];
    const dy = target[1] - origin[1];
    const dz = target[2] - origin[2];
    const dist = Math.hypot(dx, dy, dz);
    if (dist <= 1e-5) continue;
    const maxYaw = Math.abs(chain.maxYaw ?? 1.05);
    const maxPitch = Math.abs(chain.maxPitch ?? 0.70);
    const yaw = Math.min(maxYaw, Math.max(-maxYaw, Math.atan2(dx, dz))) * weight;
    const elev = Math.min(maxPitch, Math.max(-maxPitch, Math.atan2(dy, Math.hypot(dx, dz)))) * weight;
    for (const [node, share] of chain.nodes) {
      const cur = engine.modelNode(instance, node);
      const dq = quatFromEuler(-elev * share, yaw * share, 0);
      engine.setModelNode(instance, node, {
        ...cur,
        rotation: normalizeQuat(mulQuat(cur.rotation, dq)),
      });
    }
  }
  for (const off of spec.boneOffsets ?? []) {
    const cur = engine.modelNode(instance, off.node);
    const t = off.translation ?? [0, 0, 0];
    const r = off.rotation
      ?? (off.euler ? quatFromEuler(off.euler[0], off.euler[1], off.euler[2]) : [0, 0, 0, 1]);
    const s = off.scale ?? [1, 1, 1];
    engine.setModelNode(instance, off.node, {
      translation: [cur.translation[0] + t[0], cur.translation[1] + t[1], cur.translation[2] + t[2]],
      rotation: normalizeQuat(mulQuat(cur.rotation, r)),
      scale: [cur.scale[0] * s[0], cur.scale[1] * s[1], cur.scale[2] * s[2]],
    });
  }
  for (const m of spec.morphOverrides ?? []) {
    engine.modelMorph(instance, m.node, m.target, m.weight);
  }
  if (spec.commitUpdate !== false && typeof engine.updateModel === 'function') {
    engine.updateModel(instance);
  }
  return packet;
}

/**
 * Stateful rig controller managing clip cuts, Hermite cross-fades, Look-At tracking, bone offsets,
 * and named morph channels for one model instance.
 */
export function createCharacterRigController(engine, instance, info, config = {}) {
  const clipByName = new Map(info.clips.map((c, index) => [c.name, { ...c, index }]));
  const nodeByName = new Map(info.nodes.map((n, index) => [n.name, index]));
  const morphSlots = new Map();
  for (const [nodeIndex, node] of info.nodes.entries()) {
    if (node.mesh == null) continue;
    const mesh = info.meshes[node.mesh];
    if (!mesh) continue;
    for (const [targetIndex, name] of mesh.targetNames.entries()) {
      let slots = morphSlots.get(name);
      if (!slots) morphSlots.set(name, slots = []);
      slots.push([nodeIndex, targetIndex]);
    }
  }

  const animatedJoints = new Set();
  for (const clip of info.clips) {
    for (const n of clip.nodes ?? []) animatedJoints.add(n);
  }
  const jointMask = [...animatedJoints];

  let activeKey = null;
  let fadeStartedAt = null;
  let fadeDuration = config.defaultCrossfadeSeconds ?? 0.1;
  let fallbackSnapshot = null;

  const snapshotCurrent = () => {
    if (typeof engine.snapshotModelPose === 'function') {
      engine.snapshotModelPose(instance);
      fallbackSnapshot = null;
      return;
    }
    if (typeof engine.modelNode === 'function') {
      fallbackSnapshot = new Map();
      for (const node of jointMask) {
        fallbackSnapshot.set(node, engine.modelNode(instance, node));
      }
    }
  };

  return {
    clipByName,
    nodeByName,
    morphNames: Object.freeze([...morphSlots.keys()]),
    seek() {
      activeKey = null;
      fadeStartedAt = null;
      fallbackSnapshot = null;
    },
    evaluate(frame) {
      const nowSeconds = frame.clockSeconds ?? 0;
      const key = frame.stateKey ?? frame.clip ?? '';
      if (activeKey !== null && key !== activeKey && (frame.crossfadeSeconds ?? fadeDuration) > 0) {
        snapshotCurrent();
        fadeStartedAt = nowSeconds;
        fadeDuration = frame.crossfadeSeconds ?? config.defaultCrossfadeSeconds ?? 0.1;
      }
      activeKey = key;

      const clips = [];
      if (frame.secondaryClip) {
        const sec = typeof frame.secondaryClip === 'number'
          ? { index: frame.secondaryClip }
          : clipByName.get(frame.secondaryClip);
        if (sec) clips.push({ clip: sec.index, time: frame.secondaryTime ?? frame.time ?? 0, weight: 1 });
      }
      if (frame.clip != null) {
        const pri = typeof frame.clip === 'number' ? { index: frame.clip } : clipByName.get(frame.clip);
        if (pri) {
          clips.push({
            clip: pri.index,
            time: frame.time ?? 0,
            weight: frame.secondaryClip ? (frame.blendWeight ?? 0.5) : (frame.weight ?? 1),
          });
        }
      }

      let crossfade = null;
      if (fadeStartedAt !== null) {
        const elapsed = nowSeconds - fadeStartedAt;
        if (elapsed >= 0 && elapsed < fadeDuration) {
          crossfade = { elapsed, duration: fadeDuration, mask: jointMask };
        } else {
          fadeStartedAt = null;
          fallbackSnapshot = null;
        }
      }

      const lookAt = [];
      if (frame.lookAt) {
        const resolvedNodes = (frame.lookAt.joints ?? []).flatMap(([nameOrIndex, share]) => {
          const idx = typeof nameOrIndex === 'number' ? nameOrIndex : nodeByName.get(nameOrIndex);
          return idx == null ? [] : [[idx, share]];
        });
        if (resolvedNodes.length) {
          lookAt.push({ ...frame.lookAt, nodes: resolvedNodes });
        }
      }

      const boneOffsets = [];
      for (const [nameOrIndex, offset] of Object.entries(frame.bonePoses ?? {})) {
        const idx = /^\d+$/.test(nameOrIndex) ? Number(nameOrIndex) : nodeByName.get(nameOrIndex);
        if (idx == null) continue;
        if (Array.isArray(offset)) {
          boneOffsets.push({ node: idx, euler: offset });
        } else {
          boneOffsets.push({ node: idx, ...offset });
        }
      }

      const morphOverrides = [];
      for (const [name, weight] of Object.entries(frame.morphWeights ?? {})) {
        const slots = morphSlots.get(name);
        if (!slots || !Number.isFinite(weight)) continue;
        for (const [node, target] of slots) {
          morphOverrides.push({ node, target, weight });
        }
      }

      return poseModelClip(
        engine,
        instance,
        {
          resetToRest: frame.resetToRest !== false,
          commitUpdate: frame.commitUpdate !== false,
          clips,
          crossfade,
          lookAt,
          twoBoneIk: frame.twoBoneIk ?? [],
          boneOffsets,
          morphOverrides,
        },
        { fromPose: fallbackSnapshot },
      );
    },
  };
}
