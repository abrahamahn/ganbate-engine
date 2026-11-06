/**
 * Character Studio workbench (`*.character.json`, schema `ganbate.character/1`).
 *
 * Genre-neutral authoring surface for humanoid & creature skeletal rigs, upgraded onto the
 * Tri-Modal Articulated Semantic Graph Engine:
 * - Hierarchy Bone Tree with search + local Euler `(pitch, yaw, roll)` offset sliders
 * - Animation Clip scrubber & Hermite (`3t^2 - 2t^3`) state cross-fade preview
 * - Procedural Look-At 2D/3D target pad + joint distribution weights + Gaze Cone projection (`C_cone`)
 * - Analytical Two-Bone IK chain inspector + Reachability Manifold & Torso Exclusion projection (`C_IK`)
 * - 5-Layer Parametric Character DNA (`character-dna.js`) + Direct 3D Stage Region Drag (`L_dna: R^3 -> G_dna`)
 * - Circumstance-Driven Animation State Machine (`anim-state-machine.js`) with Live Condition Simulator
 * - Bidirectional Semantic Subgraph (`nodes[] + edges[] + constraints[]`), Inverse Semantic Lifting
 *   (`L_char: R^3 -> G_char`), Minimal-Perturbation Constraint Projection (`x* = argmin ||x - x_hat||^2`),
 *   and 2-Agent Concurrent Character Conflict Resolution.
 */

import {
  hermiteFade,
  poseModelClip,
  quatFromEuler,
  sampleBlendSpace1D,
  sampleBlendSpace2D,
} from '../web/character-customization.mjs';
import {
  ANIM_STATE_MACHINE_SCHEMA,
  asmToSemanticSubgraph,
  createAnimStateMachineController,
  createDefaultAnimStateMachine,
  liftContinuousPoseToAsmParams,
  normalizeAnimStateMachine,
  summarizeAsmForAi,
} from './anim-state-machine.js';
import { drawGlbPreviewCanvas, parseGlbPreview, parseGlbPreviewAsync } from './asset-previews.js';
import {
  BODY_OCCLUSION_REGIONS,
  CHARACTER_DNA_SCHEMA,
  DNA_SLIDER_CATEGORIES,
  STAGE_DRAG_REGIONS,
  WARDROBE_SLOT_CATALOG,
  compileDnaToRigOverrides,
  createAvatarDnaController,
  createDefaultCharacterDna,
  generateDnaFromTraits,
  hitTestStageDragRegion,
  liftStageRegionDragToDna,
  normalizeCharacterDna,
  solveWardrobeOcclusionConstraints,
  summarizeCharacterDnaForAi,
} from './character-dna.js';

export {
  ANIM_STATE_MACHINE_SCHEMA,
  BODY_OCCLUSION_REGIONS,
  CHARACTER_DNA_SCHEMA,
  DNA_SLIDER_CATEGORIES,
  STAGE_DRAG_REGIONS,
  WARDROBE_SLOT_CATALOG,
  compileDnaToRigOverrides,
  createAnimStateMachineController,
  createAvatarDnaController,
  createDefaultAnimStateMachine,
  createDefaultCharacterDna,
  generateDnaFromTraits,
  liftStageRegionDragToDna,
  solveWardrobeOcclusionConstraints,
};

export const CHARACTER_SCHEMA = 'ganbate.character/1';
export const CHARACTER_SEMANTIC_GRAPH_SCHEMA = 'ganbate.semantic.graph.v2';

export const TURNTABLE_LIGHTING_PRESETS = Object.freeze({
  studio_neutral: {
    id: 'studio_neutral',
    label: 'Studio Neutral',
    sunDirection: [-0.38, -0.78, -0.49],
    sunColor: [1.0, 0.98, 0.95],
    ambientSky: [0.34, 0.37, 0.44],
    ambientGround: [0.16, 0.15, 0.17],
  },
  warm_rim: {
    id: 'warm_rim',
    label: 'Warm Rim',
    sunDirection: [0.52, -0.62, -0.58],
    sunColor: [1.0, 0.86, 0.68],
    ambientSky: [0.30, 0.26, 0.34],
    ambientGround: [0.18, 0.13, 0.11],
  },
  cool_moonlight: {
    id: 'cool_moonlight',
    label: 'Cool Moonlight',
    sunDirection: [-0.45, -0.68, 0.57],
    sunColor: [0.76, 0.86, 1.0],
    ambientSky: [0.18, 0.23, 0.36],
    ambientGround: [0.08, 0.10, 0.15],
  },
});

export const CHARACTER_ACADEMIC_SCENARIOS = Object.freeze({
  embodied_scholar_reach: {
    id: 'embodied_scholar_reach',
    label: '1. Embodied Scholar Reach & Gaze',
    prompt: 'Reach for [0.36, 1.18, 0.42] with right hand, look at [0.22, 1.52, 1.8], tall scholar coat',
  },
  unreachable_ik_projection: {
    id: 'unreachable_ik_projection',
    label: '2. Unreachable IK & Neck Cone Projection (x̂ → x*)',
    prompt: 'Reach for [1.45, 2.10, 1.35] with right hand and look at [2.80, 2.40, -0.50]',
  },
  two_agent_posture_conflict: {
    id: 'two_agent_posture_conflict',
    label: '3. 2-Agent Posture + IK Torso Collision',
    isTwoAgentConflict: true,
  },
  wardrobe_layer_occlusion: {
    id: 'wardrobe_layer_occlusion',
    label: '4. 5-Layer Wardrobe Occlusion & Clearance',
    prompt: 'Tall athletic scholar wearing pavilion coat with heterochromia, focus on table',
  },
});

const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;
const clamp = (val, min, max) => Math.min(max, Math.max(min, Number.isFinite(val) ? val : 0));

export function createDefaultCharacterDocument(id = 'humanoid_actor', name = 'Humanoid Actor') {
  return {
    schema: CHARACTER_SCHEMA,
    id,
    name,
    skeleton: {
      scale: 1.0,
      scalePivot: [0, 0, 0],
      placement: { position: [0, 0, 0], yawRad: 0 },
      joints: [
        { index: 0, name: 'hips', parent: -1 },
        { index: 1, name: 'spine_01', parent: 0 },
        { index: 2, name: 'neck_01', parent: 1 },
        { index: 3, name: 'head', parent: 2 },
        { index: 4, name: 'upperarm_r', parent: 1 },
        { index: 5, name: 'lowerarm_r', parent: 4 },
        { index: 6, name: 'hand_r', parent: 5 },
      ],
    },
    activeOutfitId: 'standard_attire',
    outfits: [
      {
        id: 'standard_attire',
        label: 'Standard Attire',
        modelPath: 'content/characters/humanoid_standard.glb',
        scale: 1.0,
        hiddenMaterials: [],
      },
    ],
    sockets: [
      {
        id: 'prop_hand_r',
        parentJoint: 'hand_r',
        translation: [0.02, 0.04, 0.0],
        eulerDeg: [0, 0, 0],
        attachedAsset: '',
      },
    ],
    clips: [
      { id: 'idle_calm', index: 0, label: 'Idle Calm', durationS: 2.4, looping: true },
      { id: 'gesture_greet', index: 1, label: 'Gesture Greet', durationS: 1.6, looping: false },
    ],
    stateChains: [
      { fromClip: 'gesture_greet', toClip: 'idle_calm', crossfadeMs: 220 },
    ],
    blendSpaces: [
      {
        id: 'stance_blend_1d',
        dimensions: 1,
        samples: [
          { clipId: 'idle_calm', position: 0, rate: 1 },
          { clipId: 'gesture_greet', position: 1, rate: 1 },
        ],
      },
    ],
    lookAt: {
      enabled: true,
      origin: [0, 1.55, 0],
      target: [0, 1.55, 2.0],
      maxYawDeg: 55,
      maxPitchDeg: 35,
      chain: [
        { joint: 'neck_01', weight: 0.35 },
        { joint: 'head', weight: 0.65 },
      ],
    },
    ikChains: [
      {
        id: 'reach_hand_r',
        enabled: false,
        rootJoint: 'upperarm_r',
        midJoint: 'lowerarm_r',
        endJoint: 'hand_r',
        target: [0.32, 1.15, 0.38],
        poleVector: [0.25, 1.05, -0.35],
        weight: 1.0,
      },
    ],
    morphTargets: [
      { id: 'blink', index: 0, node: 3, label: 'Blink', group: 'expression', min: 0, max: 1, defaultWeight: 0 },
      { id: 'smile', index: 1, node: 3, label: 'Smile', group: 'expression', min: 0, max: 1, defaultWeight: 0 },
      { id: 'jaw_width', index: 2, node: 3, label: 'Jaw Width', group: 'face', min: -1, max: 1, defaultWeight: 0 },
      { id: 'shoulder_breadth', index: 3, node: 1, label: 'Shoulder Breadth', group: 'body', min: -1, max: 1, defaultWeight: 0 },
      { id: 'cloth_settle', index: 4, node: 1, label: 'Cloth Settle', group: 'secondary', min: 0, max: 1, defaultWeight: 0 },
    ],
    expressionIntensity: 1.0,
    morphWeights: {
      blink: 0,
      smile: 0.15,
      jaw_width: 0,
      shoulder_breadth: 0,
      cloth_settle: 0,
    },
    expressionPresets: {
      neutral: { blink: 0, smile: 0 },
      warm_smile: { blink: 0.08, smile: 0.72 },
      focused: { blink: 0.18, smile: 0.05 },
    },
    boneOffsets: {},
    dna: createDefaultCharacterDna(),
    animStateMachine: createDefaultAnimStateMachine(),
  };
}

export function normalizeCharacterDocument(raw) {
  const fallback = createDefaultCharacterDocument();
  if (!raw || typeof raw !== 'object') return fallback;
  const joints = Array.isArray(raw.skeleton?.joints) && raw.skeleton.joints.length
    ? raw.skeleton.joints.map((j, idx) => ({
        index: typeof j.index === 'number' ? j.index : idx,
        name: String(j.name ?? `joint_${idx}`),
        parent: typeof j.parent === 'number' ? j.parent : -1,
      }))
    : fallback.skeleton.joints;
  const outfits = Array.isArray(raw.outfits) && raw.outfits.length ? raw.outfits : fallback.outfits;
  return {
    schema: CHARACTER_SCHEMA,
    id: String(raw.id ?? fallback.id),
    name: String(raw.name ?? raw.id ?? fallback.name),
    skeleton: {
      scale: Number.isFinite(raw.skeleton?.scale) ? raw.skeleton.scale : 1.0,
      scalePivot: Array.isArray(raw.skeleton?.scalePivot) ? [...raw.skeleton.scalePivot] : [0, 0, 0],
      placement: raw.skeleton?.placement
        ? {
            position: [...(raw.skeleton.placement.position ?? [0, 0, 0])],
            yawRad: Number(raw.skeleton.placement.yawRad ?? 0),
          }
        : { position: [0, 0, 0], yawRad: 0 },
      joints,
    },
    activeOutfitId: String(raw.activeOutfitId ?? outfits[0].id),
    outfits: outfits.map((o) => ({
      id: String(o.id),
      label: String(o.label ?? o.id),
      modelPath: String(o.modelPath ?? ''),
      scale: Number.isFinite(o.scale) ? o.scale : 1.0,
      hiddenMaterials: Array.isArray(o.hiddenMaterials) ? [...o.hiddenMaterials] : [],
    })),
    sockets: Array.isArray(raw.sockets)
      ? raw.sockets.map((s) => ({
          id: String(s.id),
          parentJoint: String(s.parentJoint ?? ''),
          translation: Array.isArray(s.translation) ? [...s.translation] : [0, 0, 0],
          eulerDeg: Array.isArray(s.eulerDeg) ? [...s.eulerDeg] : [0, 0, 0],
          attachedAsset: String(s.attachedAsset ?? ''),
        }))
      : [],
    clips: Array.isArray(raw.clips) && raw.clips.length
      ? raw.clips.map((c, idx) => ({
          id: String(c.id ?? `clip_${idx}`),
          index: typeof c.index === 'number' ? c.index : idx,
          label: String(c.label ?? c.id ?? `Clip ${idx}`),
          durationS: Number.isFinite(c.durationS) && c.durationS > 0 ? c.durationS : 1.0,
          looping: c.looping !== false,
        }))
      : fallback.clips,
    stateChains: Array.isArray(raw.stateChains) ? raw.stateChains.map((ch) => ({ ...ch })) : [],
    blendSpaces: Array.isArray(raw.blendSpaces) ? raw.blendSpaces.map((bs) => ({ ...bs })) : fallback.blendSpaces,
    lookAt: {
      enabled: raw.lookAt?.enabled !== false,
      origin: Array.isArray(raw.lookAt?.origin) ? [...raw.lookAt.origin] : [0, 1.55, 0],
      target: Array.isArray(raw.lookAt?.target) ? [...raw.lookAt.target] : [0, 1.55, 2.0],
      maxYawDeg: Number.isFinite(raw.lookAt?.maxYawDeg) ? raw.lookAt.maxYawDeg : 55,
      maxPitchDeg: Number.isFinite(raw.lookAt?.maxPitchDeg) ? raw.lookAt.maxPitchDeg : 35,
      chain: Array.isArray(raw.lookAt?.chain)
        ? raw.lookAt.chain.map((entry) => ({
            joint: String(entry.joint),
            weight: Number(entry.weight ?? 0.5),
          }))
        : fallback.lookAt.chain,
    },
    ikChains: Array.isArray(raw.ikChains)
      ? raw.ikChains.map((ik, idx) => ({
          id: String(ik.id ?? `ik_${idx}`),
          enabled: Boolean(ik.enabled),
          rootJoint: String(ik.rootJoint ?? ''),
          midJoint: String(ik.midJoint ?? ''),
          endJoint: String(ik.endJoint ?? ''),
          target: Array.isArray(ik.target) ? [...ik.target] : [0, 1, 0.3],
          poleVector: Array.isArray(ik.poleVector) ? [...ik.poleVector] : [0, 1, -0.4],
          weight: Number.isFinite(ik.weight) ? ik.weight : 1.0,
        }))
      : [],
    morphTargets: Array.isArray(raw.morphTargets)
      ? raw.morphTargets.map((m, idx) => ({
          id: String(m.id ?? `morph_${idx}`),
          index: typeof m.index === 'number' ? m.index : idx,
          node: typeof m.node === 'number' ? m.node : 0,
          label: String(m.label ?? m.id ?? `Morph ${idx}`),
          group: ['expression', 'face', 'body', 'secondary'].includes(m.group) ? m.group : 'expression',
          min: Number.isFinite(m.min) ? m.min : 0,
          max: Number.isFinite(m.max) ? m.max : 1,
          defaultWeight: Number.isFinite(m.defaultWeight) ? m.defaultWeight : 0,
        }))
      : fallback.morphTargets,
    expressionIntensity: Number.isFinite(raw.expressionIntensity) ? raw.expressionIntensity : 1.0,
    morphWeights: raw.morphWeights && typeof raw.morphWeights === 'object' ? { ...raw.morphWeights } : {},
    expressionPresets:
      raw.expressionPresets && typeof raw.expressionPresets === 'object'
        ? { ...raw.expressionPresets }
        : fallback.expressionPresets,
    boneOffsets: raw.boneOffsets && typeof raw.boneOffsets === 'object' ? { ...raw.boneOffsets } : {},
    dna: normalizeCharacterDna(raw.dna),
    animStateMachine: normalizeAnimStateMachine(raw.animStateMachine),
  };
}

export function filterBoneTree(joints, query = '') {
  const depthByIndex = new Map();
  const byIndex = new Map(joints.map((j) => [j.index, j]));
  const depthOf = (j, seen = new Set()) => {
    if (depthByIndex.has(j.index)) return depthByIndex.get(j.index);
    if (j.parent < 0 || !byIndex.has(j.parent) || seen.has(j.index)) {
      depthByIndex.set(j.index, 0);
      return 0;
    }
    seen.add(j.index);
    const d = depthOf(byIndex.get(j.parent), seen) + 1;
    depthByIndex.set(j.index, d);
    return d;
  };
  const needle = String(query ?? '').trim().toLowerCase();
  return joints
    .map((j) => ({ ...j, depth: depthOf(j) }))
    .filter((j) => !needle || j.name.toLowerCase().includes(needle));
}

export function groupMorphTargets(morphTargets) {
  const groups = {
    expression: [],
    face: [],
    body: [],
    secondary: [],
  };
  for (const m of morphTargets ?? []) {
    const key = Object.hasOwn(groups, m.group) ? m.group : 'expression';
    groups[key].push(m);
  }
  return groups;
}

export function buildBoneOffsetOps(jointName, eulerDeg, translation = [0, 0, 0]) {
  const isZero = eulerDeg.every((v) => Math.abs(v) < 1e-6) && translation.every((v) => Math.abs(v) < 1e-6);
  if (isZero) {
    return [{ remove: ['boneOffsets', jointName] }];
  }
  return [
    {
      set: ['boneOffsets', jointName],
      value: {
        eulerDeg: eulerDeg.map((n) => Number(n)),
        translation: translation.map((n) => Number(n)),
      },
    },
  ];
}

export function buildMorphWeightOps(morphId, weight) {
  return [{ set: ['morphWeights', morphId], value: Number(weight) }];
}

export function buildPresetApplyOps(doc, presetName) {
  const preset = doc.expressionPresets?.[presetName];
  if (!preset || typeof preset !== 'object') return [];
  return Object.entries(preset).map(([morphId, weight]) => ({
    set: ['morphWeights', morphId],
    value: Number(weight),
  }));
}

export function buildLookAtTargetOps(targetVec3, enabled = true) {
  return [
    { set: ['lookAt', 'enabled'], value: Boolean(enabled) },
    { set: ['lookAt', 'target'], value: targetVec3.map((n) => Number(n)) },
  ];
}

export function buildIkChainOps(chainIndex, patch) {
  return Object.entries(patch).map(([key, value]) => ({
    set: ['ikChains', String(chainIndex), key],
    value: Array.isArray(value) ? value.map((n) => Number(n)) : value,
  }));
}

export function buildActiveOutfitOps(outfitId) {
  return [{ set: ['activeOutfitId'], value: String(outfitId) }];
}

export function buildSocketTransformOps(socketIndex, patch) {
  return Object.entries(patch).map(([key, value]) => ({
    set: ['sockets', String(socketIndex), key],
    value: Array.isArray(value) ? value.map((n) => Number(n)) : value,
  }));
}

// ============================================================================
// ACADEMIC CORE ENGINE:
//   1. Minimal-Perturbation Articulated Constraint Projection (`x* = argmin ||x - x_hat||^2`)
//   2. Inverse Semantic Lifting (`L_char: R^3 -> G_char`)
//   3. Bidirectional Semantic Subgraph (`nodes[] + edges[] + constraints[]`)
//   4. 2-Agent Concurrent Character Conflict Resolution
// ============================================================================

/**
 * Minimal-Perturbation Articulated Constraint Projection (`x* = argmin_{x in C} ||x - x_hat||^2`).
 *
 * Enforces:
 *   - `C_IK`: Two-Bone IK spherical shell `[minReachM, maxReachM]` around shoulder/hip anchor
 *   - `C_torso`: Cylindrical torso self-collision exclusion (`x^2 + z^2 >= torsoRadiusM^2`)
 *   - `C_floor`: Support plane lower bound (`y >= minFloorY`)
 *   - `C_cone`: Physiological neck+head Look-At angular cone (`[-maxYawDeg, +maxYawDeg] x [-maxPitchDeg, +maxPitchDeg]`)
 *   - `C_occ`: Wardrobe layer clearance and `bodyOcclusionMask`
 */
export function projectCharacterIkConstraints(candidateDocOrPose, rigConstraints = {}) {
  const doc = normalizeCharacterDocument(candidateDocOrPose);
  const dnaOverrides = compileDnaToRigOverrides(doc.dna);
  const placementPos = doc.skeleton?.placement?.position ?? [0, 0, 0];
  const baseShoulder = rigConstraints.shoulderOrigin ?? [
    placementPos[0] + 0.18,
    placementPos[1] + 1.38 * dnaOverrides.skeletonScale,
    placementPos[2] + 0.0,
  ];
  const maxReachM = Number(rigConstraints.maxReachM ?? dnaOverrides.effectiveArmReachM ?? 0.56);
  const minReachM = Number(rigConstraints.minReachM ?? 0.14);
  const torsoRadiusM = Number(rigConstraints.torsoRadiusM ?? 0.22);
  const minFloorY = Number(rigConstraints.minFloorY ?? 0.05);

  const violations = [];
  const hallucinationGhosts = [];
  const projectedChains = [];

  doc.ikChains = doc.ikChains.map((chain) => {
    const proposed = (chain.target ?? [0.32, 1.15, 0.38]).map((n) => Number(n) || 0);
    let [x, y, z] = proposed;
    const reasons = [];

    // 1. Floor support plane projection (`y >= minFloorY`)
    if (y < minFloorY) {
      y = minFloorY;
      reasons.push('floor_penetration');
    }

    // 2. Torso cylinder self-collision exclusion (`C_torso`)
    const relTorsoX = x - placementPos[0];
    const relTorsoZ = z - placementPos[2];
    const horizDist = Math.hypot(relTorsoX, relTorsoZ);
    if (y >= placementPos[1] + 0.65 && y <= placementPos[1] + 1.52 && horizDist < torsoRadiusM) {
      const pushDirX = horizDist > 1e-5 ? relTorsoX / horizDist : 0.7071;
      const pushDirZ = horizDist > 1e-5 ? relTorsoZ / horizDist : 0.7071;
      x = placementPos[0] + pushDirX * torsoRadiusM;
      z = placementPos[2] + pushDirZ * torsoRadiusM;
      reasons.push('torso_self_collision');
    }

    // 3. Spherical Two-Bone IK Reachability Manifold (`C_IK`)
    const dx = x - baseShoulder[0];
    const dy = y - baseShoulder[1];
    const dz = z - baseShoulder[2];
    const dist = Math.hypot(dx, dy, dz);
    if (dist > maxReachM && dist > 1e-6) {
      const scale = maxReachM / dist;
      x = baseShoulder[0] + dx * scale;
      y = baseShoulder[1] + dy * scale;
      z = baseShoulder[2] + dz * scale;
      reasons.push('exceeds_arm_reach_manifold');
    } else if (dist < minReachM) {
      const ux = dist > 1e-5 ? dx / dist : 0;
      const uy = dist > 1e-5 ? dy / dist : 0;
      const uz = dist > 1e-5 ? dz / dist : 1;
      x = baseShoulder[0] + ux * minReachM;
      y = baseShoulder[1] + uy * minReachM;
      z = baseShoulder[2] + uz * minReachM;
      reasons.push('inside_min_elbow_fold');
    }

    const projected = [Number(x.toFixed(3)), Number(y.toFixed(3)), Number(z.toFixed(3))];
    const perturbationM = Number(
      Math.hypot(projected[0] - proposed[0], projected[1] - proposed[1], projected[2] - proposed[2]).toFixed(4),
    );

    if (perturbationM > 1e-3) {
      const record = {
        kind: 'ik_target',
        chainId: chain.id,
        endJoint: chain.endJoint,
        proposed,
        projected,
        perturbationM,
        reasons,
      };
      violations.push(record);
      hallucinationGhosts.push(record);
    }

    projectedChains.push({
      chainId: chain.id,
      proposed,
      projected,
      perturbationM,
      reasons,
    });

    return {
      ...chain,
      target: projected,
    };
  });

  // 4. Physiological Look-At Gaze Cone Projection (`C_cone`)
  let lookAtProjection = null;
  if (doc.lookAt) {
    const origin = (doc.lookAt.origin ?? [0, 1.55, 0]).map((n) => Number(n) || 0);
    const proposedGaze = (doc.lookAt.target ?? [0, 1.55, 2.0]).map((n) => Number(n) || 0);
    let gx = proposedGaze[0] - origin[0];
    let gy = proposedGaze[1] - origin[1];
    let gz = proposedGaze[2] - origin[2];
    const gazeReasons = [];

    if (gz < 0.25) {
      gz = 0.25;
      gazeReasons.push('behind_head_plane');
    }
    const maxYawRad = (Number(doc.lookAt.maxYawDeg) || 55) * DEG_TO_RAD;
    const maxPitchRad = (Number(doc.lookAt.maxPitchDeg) || 35) * DEG_TO_RAD;
    const curYaw = Math.atan2(gx, gz);
    if (Math.abs(curYaw) > maxYawRad) {
      const clampedYaw = Math.sign(curYaw) * maxYawRad;
      const horizMag = Math.hypot(gx, gz);
      gx = Math.sin(clampedYaw) * horizMag;
      gz = Math.cos(clampedYaw) * horizMag;
      gazeReasons.push('exceeds_neck_yaw_cone');
    }
    const horizDist = Math.max(0.25, Math.hypot(gx, gz));
    const curPitch = Math.atan2(gy, horizDist);
    if (Math.abs(curPitch) > maxPitchRad) {
      const clampedPitch = Math.sign(curPitch) * maxPitchRad;
      gy = Math.tan(clampedPitch) * horizDist;
      gazeReasons.push('exceeds_neck_pitch_cone');
    }

    const projectedGaze = [
      Number((origin[0] + gx).toFixed(3)),
      Number((origin[1] + gy).toFixed(3)),
      Number((origin[2] + gz).toFixed(3)),
    ];
    const gazePerturbationM = Number(
      Math.hypot(
        projectedGaze[0] - proposedGaze[0],
        projectedGaze[1] - proposedGaze[1],
        projectedGaze[2] - proposedGaze[2],
      ).toFixed(4),
    );

    if (gazePerturbationM > 1e-3) {
      const rec = {
        kind: 'look_at',
        chainId: 'look_at_head',
        proposed: proposedGaze,
        projected: projectedGaze,
        perturbationM: gazePerturbationM,
        reasons: gazeReasons,
      };
      violations.push(rec);
      hallucinationGhosts.push(rec);
    }
    lookAtProjection = {
      proposed: proposedGaze,
      projected: projectedGaze,
      perturbationM: gazePerturbationM,
      reasons: gazeReasons,
    };
    doc.lookAt.target = projectedGaze;
  }

  // 5. Wardrobe Anti-Clipping Constraint Projection (`C_occ`)
  const wardrobeOcclusion = solveWardrobeOcclusionConstraints(doc.dna);
  doc.dna = wardrobeOcclusion.dna;
  for (const wp of wardrobeOcclusion.projections) {
    violations.push({
      kind: 'wardrobe_clearance',
      slot: wp.slot,
      proposed: wp.proposedScaleOffsetMm,
      projected: wp.projectedScaleOffsetMm,
      perturbationM: Number((wp.perturbationMm / 1000).toFixed(4)),
      reasons: [wp.reason],
    });
  }

  return {
    document: doc,
    projectedChains,
    lookAtProjection,
    wardrobeOcclusion,
    effectiveArmReachM: maxReachM,
    violations,
    hallucinationGhosts,
    verified: violations.length === 0,
  };
}

function inferReachSymbolicRelation(shoulderOrigin, targetPos, enabled, maxReachM) {
  if (!enabled) {
    return { rel: 'stowed_beside', to: 'torso_rest_anchor', zone: 'stowed', distanceM: 0 };
  }
  const dx = targetPos[0] - shoulderOrigin[0];
  const dy = targetPos[1] - shoulderOrigin[1];
  const dz = targetPos[2] - shoulderOrigin[2];
  const distanceM = Number(Math.hypot(dx, dy, dz).toFixed(3));
  let zone = 'mid_interact';
  if (distanceM <= 0.34) zone = 'close_grasp';
  else if (distanceM >= maxReachM * 0.85) zone = 'extended_reach';

  let directionTag = 'front_ workspace';
  if (dx > 0.28) directionTag = 'right_workspace';
  else if (dx < -0.28) directionTag = 'left_workspace';
  else if (dy > 0.18) directionTag = 'elevated_shelf';
  else if (dy < -0.25) directionTag = 'low_surface';

  return {
    rel: 'reaches_for',
    to: `anchor:${directionTag.replace(/\s+/g, '')}`,
    zone,
    directionTag: directionTag.replace(/\s+/g, ''),
    distanceM,
  };
}

function inferGazeSymbolicRelation(origin, target) {
  const dx = (target[0] ?? 0) - (origin[0] ?? 0);
  const dy = (target[1] ?? 1.55) - (origin[1] ?? 1.55);
  const dz = Math.max(0.25, (target[2] ?? 2.0) - (origin[2] ?? 0));
  const yawDeg = Number((Math.atan2(dx, dz) * RAD_TO_DEG).toFixed(1));
  const pitchDeg = Number((Math.atan2(dy, Math.hypot(dx, dz)) * RAD_TO_DEG).toFixed(1));

  let focusAnchor = 'anchor:eye_level_front';
  if (yawDeg <= -15) focusAnchor = 'anchor:left_subject';
  else if (yawDeg >= 15) focusAnchor = 'anchor:right_subject';
  else if (pitchDeg <= -12) focusAnchor = 'anchor:work_surface_below';
  else if (pitchDeg >= 12) focusAnchor = 'anchor:upper_gallery';

  return {
    rel: 'looks_at',
    to: focusAnchor,
    yawDeg,
    pitchDeg,
  };
}

/**
 * Inverse Semantic Lifting (`L_char: R^3 -> G_char`).
 * Converts continuous 3D viewport drags (IK effector pin, Look-At gaze reticle, or 3D DNA sculpt region)
 * into verified character state + discrete symbolic relational edges (`edges[]`).
 */
export function liftCharacterStageDragToSemanticGraph(rawDoc, dragEvent = {}, options = {}) {
  let doc = normalizeCharacterDocument(rawDoc);
  const solverMode = options.solverMode ?? 'project'; // 'project' | 'reject'
  const beforeDoc = structuredClone(doc);

  if (dragEvent.kind === 'ik_target') {
    const chainIdx = Number(dragEvent.chainIndex ?? 0);
    if (doc.ikChains[chainIdx] && Array.isArray(dragEvent.target)) {
      doc.ikChains[chainIdx].enabled = dragEvent.enabled !== false;
      doc.ikChains[chainIdx].target = dragEvent.target.map((n) => Number(n));
    }
  } else if (dragEvent.kind === 'look_at') {
    if (Array.isArray(dragEvent.target)) {
      doc.lookAt.enabled = dragEvent.enabled !== false;
      doc.lookAt.target = dragEvent.target.map((n) => Number(n));
    }
  } else if (dragEvent.kind === 'dna_region') {
    const liftedDna = liftStageRegionDragToDna(
      doc.dna,
      dragEvent.regionId ?? 'nose',
      Number(dragEvent.dxNorm ?? 0),
      Number(dragEvent.dyNorm ?? 0),
    );
    doc.dna = liftedDna.dna;
    const rigOverrides = compileDnaToRigOverrides(doc.dna);
    doc.skeleton.scale = rigOverrides.skeletonScale;
    Object.assign(doc.morphWeights, rigOverrides.morphWeights);
    Object.assign(doc.boneOffsets, rigOverrides.boneOffsets);
  }

  const projection = projectCharacterIkConstraints(doc, options.rigConstraints);
  if (solverMode === 'reject' && !projection.verified) {
    const fallbackSubgraph = getCharacterSemanticSubgraph(beforeDoc);
    return {
      rejected: true,
      document: beforeDoc,
      projection,
      subgraph: fallbackSubgraph,
      liftedEdges: fallbackSubgraph.edges,
    };
  }

  doc = projection.document;

  // Lift continuous IK effector state into discrete ASM parameters (`reachActive`, `interactionDistance`)
  const primaryIk = doc.ikChains[0];
  if (primaryIk) {
    doc.animStateMachine.parameters = liftContinuousPoseToAsmParams(doc.animStateMachine, {
      ikTarget: primaryIk.target,
      ikEnabled: primaryIk.enabled,
    });
  }

  const subgraph = getCharacterSemanticSubgraph(doc);
  return {
    rejected: false,
    document: doc,
    projection,
    subgraph,
    liftedEdges: subgraph.edges,
  };
}

/**
 * Projects a `.character.json` document into the canonical `ganbate.semantic.graph.v2`
 * Articulated Semantic Subgraph (`nodes[] + edges[] + constraints[]`).
 */
export function getCharacterSemanticSubgraph(rawDoc, options = {}) {
  const projection = projectCharacterIkConstraints(rawDoc, options.rigConstraints);
  const doc = projection.document;
  const dnaOverrides = compileDnaToRigOverrides(doc.dna);
  const placementPos = doc.skeleton?.placement?.position ?? [0, 0, 0];
  const shoulderOrigin = [
    placementPos[0] + 0.18,
    placementPos[1] + 1.38 * dnaOverrides.skeletonScale,
    placementPos[2],
  ];

  const nodes = [
    {
      id: `char:${doc.id}`,
      type: 'articulated_humanoid',
      name: doc.name,
      position: placementPos,
      yawDeg: Number(((doc.skeleton?.placement?.yawRad ?? 0) * RAD_TO_DEG).toFixed(1)),
      statureScale: dnaOverrides.skeletonScale,
      activeOutfitId: doc.activeOutfitId,
      bodyOcclusionMask: dnaOverrides.bodyOcclusionMask,
      hiddenBodyRegions: dnaOverrides.hiddenBodyRegions,
    },
  ];

  const edges = [];

  // 1. Two-Bone IK Effector Nodes & Symbolic `reaches_for` Edges
  for (const chain of doc.ikChains) {
    const effId = `effector:${doc.id}:${chain.endJoint}`;
    const lifted = inferReachSymbolicRelation(
      shoulderOrigin,
      chain.target,
      chain.enabled,
      projection.effectiveArmReachM,
    );
    nodes.push({
      id: effId,
      type: 'ik_effector',
      chainId: chain.id,
      endJoint: chain.endJoint,
      enabled: chain.enabled,
      target: chain.target,
      weight: chain.weight,
    });
    edges.push({
      from: effId,
      rel: lifted.rel,
      to: lifted.to,
      zone: lifted.zone,
      distanceM: lifted.distanceM,
    });
  }

  // 2. Look-At Gaze Node & Symbolic `looks_at` Edge
  if (doc.lookAt) {
    const gazeNodeId = `gaze:${doc.id}:head`;
    const liftedGaze = inferGazeSymbolicRelation(doc.lookAt.origin, doc.lookAt.target);
    nodes.push({
      id: gazeNodeId,
      type: 'gaze_target',
      enabled: doc.lookAt.enabled,
      origin: doc.lookAt.origin,
      target: doc.lookAt.target,
      maxYawDeg: doc.lookAt.maxYawDeg,
      maxPitchDeg: doc.lookAt.maxPitchDeg,
    });
    edges.push({
      from: `char:${doc.id}:head`,
      rel: liftedGaze.rel,
      to: liftedGaze.to,
      yawDeg: liftedGaze.yawDeg,
      pitchDeg: liftedGaze.pitchDeg,
    });
  }

  // 3. Socket Attachment Edges (`attached_to_socket`)
  for (const sock of doc.sockets) {
    const sockNodeId = `socket:${doc.id}:${sock.id}`;
    nodes.push({
      id: sockNodeId,
      type: 'rig_socket',
      parentJoint: sock.parentJoint,
      translation: sock.translation,
      attachedAsset: sock.attachedAsset || null,
    });
    if (sock.attachedAsset) {
      edges.push({
        from: `asset:${sock.attachedAsset}`,
        rel: 'attached_to_socket',
        to: sockNodeId,
        parentJoint: sock.parentJoint,
      });
    }
  }

  // 4. Wardrobe Occlusion Constraint Edge
  edges.push({
    from: `wardrobe:${doc.activeOutfitId}`,
    rel: 'occludes_body_mask',
    to: `char:${doc.id}`,
    maskHex: `0x${dnaOverrides.bodyOcclusionMask.toString(16)}`,
    hiddenRegions: dnaOverrides.hiddenBodyRegions,
  });

  // 5. Merge ASM behavioral state nodes & transition edges
  const asmSub = asmToSemanticSubgraph(doc.animStateMachine, options.asmRuntime);
  nodes.push(...asmSub.nodes);
  edges.push(...asmSub.edges);

  return {
    schema: CHARACTER_SEMANTIC_GRAPH_SCHEMA,
    characterId: doc.id,
    nodes,
    edges,
    constraints: {
      maxArmReachM: projection.effectiveArmReachM,
      torsoRadiusM: 0.22,
      maxYawDeg: doc.lookAt.maxYawDeg,
      maxPitchDeg: doc.lookAt.maxPitchDeg,
      bodyOcclusionMask: dnaOverrides.bodyOcclusionMask,
    },
    verification: {
      verified: projection.verified,
      violationCount: projection.violations.length,
      hallucinationGhosts: projection.hallucinationGhosts,
    },
    aiDigest: summarizeCharacterForAi(doc, options.asmRuntime),
  };
}

/**
 * Resolves a 2-Agent Concurrent Character Edit Conflict:
 *   - Agent A (Scene / Posture / Placement Agent) shifts the character's root placement or target anchor.
 *   - Agent B (Behavioral / IK / Gaze / Wardrobe Agent) commands an IK reach (`reaches_for`)
 *     and gaze focus (`looks_at`) using a target coordinate or symbolic anchor.
 *
 * Naive merge leaves Agent B's world-space IK effector either inside Agent A's shifted torso
 * or outside the shifted shoulder's reachable sphere `C_IK`. Our relational solver preserves
 * Agent B's symbolic `reaches_for` / `looks_at` intent relative to Agent A's new posture and
 * projects onto `C_IK \cap C_torso \cap C_cone`.
 */
export function resolveTwoAgentCharacterConflict(rawDoc, agentAPatch = {}, agentBPatch = {}) {
  const baseDoc = normalizeCharacterDocument(rawDoc);

  // Agent A shifts character placement or posture
  const shiftPos = Array.isArray(agentAPatch.placementPosition)
    ? agentAPatch.placementPosition.map((n) => Number(n))
    : [0.38, 0.0, 0.28];
  const naiveMerged = structuredClone(baseDoc);
  naiveMerged.skeleton.placement.position = shiftPos;
  if (Array.isArray(agentAPatch.lookAtOrigin)) {
    naiveMerged.lookAt.origin = agentAPatch.lookAtOrigin.map((n) => Number(n));
  } else {
    naiveMerged.lookAt.origin = [shiftPos[0], shiftPos[1] + 1.55, shiftPos[2]];
  }

  // Agent B commands IK reach & gaze at a coordinate that collides with Agent A's new torso or exceeds reach
  const proposedIkTarget = Array.isArray(agentBPatch.ikTarget)
    ? agentBPatch.ikTarget.map((n) => Number(n))
    : [shiftPos[0] + 0.04, shiftPos[1] + 1.18, shiftPos[2] + 0.03]; // Inside shifted torso cylinder!
  const proposedGazeTarget = Array.isArray(agentBPatch.lookAtTarget)
    ? agentBPatch.lookAtTarget.map((n) => Number(n))
    : [shiftPos[0] + 2.4, shiftPos[1] + 1.55, shiftPos[2] + 0.3]; // Exceeds ±55° neck yaw cone!

  if (naiveMerged.ikChains[0]) {
    naiveMerged.ikChains[0].enabled = true;
    naiveMerged.ikChains[0].target = proposedIkTarget;
  }
  naiveMerged.lookAt.enabled = true;
  naiveMerged.lookAt.target = proposedGazeTarget;

  if (agentBPatch.wardrobeInnerId) {
    naiveMerged.dna.wardrobeSlots.inner.itemId = agentBPatch.wardrobeInnerId;
  }

  const projection = projectCharacterIkConstraints(naiveMerged);
  const resolvedDoc = projection.document;
  resolvedDoc.animStateMachine.parameters = liftContinuousPoseToAsmParams(resolvedDoc.animStateMachine, {
    shoulderOrigin: [shiftPos[0] + 0.18, shiftPos[1] + 1.38, shiftPos[2]],
    ikTarget: resolvedDoc.ikChains[0]?.target,
    ikEnabled: true,
  });

  return {
    agentA: { placementPosition: shiftPos },
    agentB: { proposedIkTarget, proposedGazeTarget },
    naiveConflictCount: projection.violations.length,
    hallucinationGhosts: projection.hallucinationGhosts,
    resolvedDocument: resolvedDoc,
    subgraph: getCharacterSemanticSubgraph(resolvedDoc),
  };
}

/**
 * Applies either a natural-language prompt string or a structured JSON semantic patch
 * to a Character document, running Minimal-Perturbation Constraint Projection (`x_hat -> x*`)
 * and Inverse Semantic Lifting (`R^3 -> G_char`).
 */
export function applyCharacterSemanticPatch(rawDoc, patchOrPrompt, options = {}) {
  let doc = normalizeCharacterDocument(rawDoc);
  const beforeDoc = structuredClone(doc);
  const solverMode = options.solverMode ?? 'project';

  if (typeof patchOrPrompt === 'string') {
    const prompt = patchOrPrompt.trim();
    const lower = prompt.toLowerCase();

    // Parse coordinate triples `[x, y, z]` if present
    const vecMatches = [...prompt.matchAll(/\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]/g)].map(
      (m) => [Number(m[1]), Number(m[2]), Number(m[3])],
    );

    if (/\b(reach|grasp|touch|pick|hand)\b/i.test(lower)) {
      if (doc.ikChains[0]) {
        doc.ikChains[0].enabled = !/\b(stop|disable|rest|release)\b/i.test(lower);
        if (vecMatches[0]) {
          doc.ikChains[0].target = vecMatches[0];
        } else if (/\bleft\b/i.test(lower)) {
          doc.ikChains[0].target = [-0.26, 1.18, 0.36];
        } else if (/\bright\b/i.test(lower)) {
          doc.ikChains[0].target = [0.38, 1.18, 0.36];
        } else if (/\bhigh|up|shelf\b/i.test(lower)) {
          doc.ikChains[0].target = [0.28, 1.56, 0.34];
        } else {
          doc.ikChains[0].target = [0.32, 1.16, 0.42];
        }
      }
    }

    if (/\b(look|gaze|face|watch)\b/i.test(lower)) {
      doc.lookAt.enabled = true;
      const gazeVec = vecMatches.length > 1 ? vecMatches[1] : (!/\breach\b/i.test(lower) && vecMatches[0] ? vecMatches[0] : null);
      if (gazeVec) {
        doc.lookAt.target = gazeVec;
      } else if (/\bleft\b/i.test(lower)) {
        doc.lookAt.target = [-0.85, 1.55, 1.6];
      } else if (/\bright\b/i.test(lower)) {
        doc.lookAt.target = [0.85, 1.55, 1.6];
      } else if (/\bdown|table|book|surface\b/i.test(lower)) {
        doc.lookAt.target = [0.0, 1.05, 1.4];
      } else {
        doc.lookAt.target = [0.0, 1.55, 2.0];
      }
    }

    if (/\b(tall|compact|petite|athletic|broad|slender|scholar|sharp|sculpted|heterochromia|coat|pavilion|haori|vest)\b/i.test(lower)) {
      doc.dna = generateDnaFromTraits(lower);
      const rigOverrides = compileDnaToRigOverrides(doc.dna);
      doc.skeleton.scale = rigOverrides.skeletonScale;
      Object.assign(doc.morphWeights, rigOverrides.morphWeights);
      Object.assign(doc.boneOffsets, rigOverrides.boneOffsets);
    }

    if (/\b(smile|warm)\b/i.test(lower)) {
      Object.assign(doc.morphWeights, doc.expressionPresets.warm_smile ?? { smile: 0.72 });
    } else if (/\b(focus|alert|serious)\b/i.test(lower)) {
      Object.assign(doc.morphWeights, doc.expressionPresets.focused ?? { blink: 0.18, smile: 0.05 });
      doc.animStateMachine.parameters.circumstance = 'focused';
    }
  } else if (patchOrPrompt && typeof patchOrPrompt === 'object') {
    if (patchOrPrompt.dna) {
      doc.dna = normalizeCharacterDna(patchOrPrompt.dna);
      const rigOverrides = compileDnaToRigOverrides(doc.dna);
      doc.skeleton.scale = rigOverrides.skeletonScale;
      Object.assign(doc.morphWeights, rigOverrides.morphWeights);
    }
    if (patchOrPrompt.lookAt) {
      doc.lookAt = { ...doc.lookAt, ...patchOrPrompt.lookAt };
    }
    if (Array.isArray(patchOrPrompt.ikChains)) {
      doc.ikChains = patchOrPrompt.ikChains.map((c, i) => ({ ...(doc.ikChains[i] ?? {}), ...c }));
    }
    if (Array.isArray(patchOrPrompt.nodes)) {
      for (const n of patchOrPrompt.nodes) {
        if (n.type === 'ik_effector' && Array.isArray(n.target)) {
          const idx = doc.ikChains.findIndex((c) => c.id === n.chainId || c.endJoint === n.endJoint);
          const targetChain = doc.ikChains[idx >= 0 ? idx : 0];
          if (targetChain) {
            targetChain.enabled = n.enabled !== false;
            targetChain.target = n.target.map((v) => Number(v));
          }
        } else if (n.type === 'gaze_target' && Array.isArray(n.target)) {
          doc.lookAt.enabled = n.enabled !== false;
          doc.lookAt.target = n.target.map((v) => Number(v));
        } else if (n.type === 'articulated_humanoid') {
          if (Array.isArray(n.position)) {
            doc.skeleton.placement.position = n.position.map((v) => Number(v));
          }
          if (typeof n.activeOutfitId === 'string' && n.activeOutfitId) {
            doc.activeOutfitId = n.activeOutfitId;
          }
        }
      }
    }
    if (Array.isArray(patchOrPrompt.edges)) {
      for (const e of patchOrPrompt.edges) {
        if (!e || typeof e !== 'object') continue;
        if ((e.rel === 'reaches_for' || e.rel === 'stowed_beside') && doc.ikChains[0]) {
          if (e.rel === 'stowed_beside' || e.to === 'torso_rest_anchor') {
            doc.ikChains[0].enabled = false;
          } else {
            doc.ikChains[0].enabled = true;
            const toAnchor = String(e.to ?? '');
            if (Array.isArray(e.target) && e.target.length === 3) {
              doc.ikChains[0].target = e.target.map((v) => Number(v));
            } else if (toAnchor.includes('left_workspace')) {
              doc.ikChains[0].target = [-0.32, 1.18, 0.36];
            } else if (toAnchor.includes('right_workspace')) {
              doc.ikChains[0].target = [0.42, 1.18, 0.36];
            } else if (toAnchor.includes('elevated_shelf')) {
              doc.ikChains[0].target = [0.28, 1.56, 0.34];
            } else if (toAnchor.includes('low_surface')) {
              doc.ikChains[0].target = [0.24, 0.88, 0.34];
            } else if (toAnchor.includes('front_workspace')) {
              doc.ikChains[0].target = [0.28, 1.16, 0.42];
            }
          }
        } else if (e.rel === 'looks_at') {
          doc.lookAt.enabled = true;
          const toFocus = String(e.to ?? '');
          if (Array.isArray(e.target) && e.target.length === 3) {
            doc.lookAt.target = e.target.map((v) => Number(v));
          } else if (toFocus.includes('left_subject')) {
            doc.lookAt.target = [-0.85, 1.55, 1.6];
          } else if (toFocus.includes('right_subject')) {
            doc.lookAt.target = [0.85, 1.55, 1.6];
          } else if (toFocus.includes('work_surface_below')) {
            doc.lookAt.target = [0.0, 1.05, 1.4];
          } else if (toFocus.includes('upper_gallery')) {
            doc.lookAt.target = [0.0, 2.05, 1.6];
          } else if (toFocus.includes('eye_level_front')) {
            doc.lookAt.target = [0.0, 1.55, 2.0];
          }
        } else if (e.rel === 'occludes_body_mask' && typeof e.from === 'string' && e.from.startsWith('wardrobe:')) {
          const outfitId = e.from.slice('wardrobe:'.length);
          if (outfitId) doc.activeOutfitId = outfitId;
        }
      }
    }
  }

  const projection = projectCharacterIkConstraints(doc, options.rigConstraints);
  if (solverMode === 'reject' && !projection.verified) {
    return {
      rejected: true,
      document: beforeDoc,
      projection,
      subgraph: getCharacterSemanticSubgraph(beforeDoc),
    };
  }

  doc = projection.document;
  if (doc.ikChains[0]) {
    doc.animStateMachine.parameters = liftContinuousPoseToAsmParams(doc.animStateMachine, {
      ikTarget: doc.ikChains[0].target,
      ikEnabled: doc.ikChains[0].enabled,
    });
  }

  return {
    rejected: false,
    document: doc,
    projection,
    subgraph: getCharacterSemanticSubgraph(doc),
  };
}

/**
 * Produces a `<120-token` compact AI summary of the Character's rig, IK/Gaze relations,
 * 5-layer Parametric DNA, and Circumstance Animation State Machine.
 */
export function summarizeCharacterForAi(rawDoc, asmRuntime = {}) {
  const doc = normalizeCharacterDocument(rawDoc);
  const ik0 = doc.ikChains[0];
  const ikSummary = ik0
    ? `${ik0.endJoint}:${ik0.enabled ? `[${ik0.target.map((n) => n.toFixed(2)).join(',')}]` : 'off'}`
    : 'none';
  const gazeSummary = doc.lookAt?.enabled
    ? `[${doc.lookAt.target.map((n) => n.toFixed(2)).join(',')}](±${doc.lookAt.maxYawDeg}°/${doc.lookAt.maxPitchDeg}°)`
    : 'off';
  return [
    `CHAR[${doc.id} joints=${doc.skeleton.joints.length} outfit=${doc.activeOutfitId}`,
    `ik=${ikSummary}`,
    `gaze=${gazeSummary}]`,
    summarizeCharacterDnaForAi(doc.dna),
    summarizeAsmForAi(doc.animStateMachine, asmRuntime),
  ].join(' | ');
}

/**
 * Compiles a `.character.json` document + live playback state into a single-call `PoseSpec`
 * suitable for `poseModelClip` / `ganbate_rt_pose_model_clip`.
 */
export function evaluateCharacterPoseSpec(rawDoc, state = {}) {
  const doc = normalizeCharacterDocument(rawDoc);
  const jointByName = new Map(doc.skeleton.joints.map((j) => [j.name, j.index]));
  const clipById = new Map(doc.clips.map((c) => [c.id, c]));

  const primaryClip = clipById.get(state.clipId) ?? doc.clips[0];
  const clips = [];
  let hermiteWeight = 1;

  if (state.blendSpaceId) {
    const bs = doc.blendSpaces.find((b) => b.id === state.blendSpaceId);
    if (bs) {
      const resolvedSamples = (bs.samples ?? [])
        .map((s) => {
          const c = typeof s.clip === 'number' ? { index: s.clip } : clipById.get(s.clipId);
          return c ? { clip: c.index, position: s.position, rate: s.rate ?? 1 } : null;
        })
        .filter(Boolean);
      const layers = bs.dimensions === 2
        ? sampleBlendSpace2D(resolvedSamples, Array.isArray(state.blendParam) ? state.blendParam : [0, 0], Number(state.timeS ?? 0))
        : sampleBlendSpace1D(resolvedSamples, Number(state.blendParam ?? 0), Number(state.timeS ?? 0));
      clips.push(...layers);
    }
  } else {
    if (state.fromClipId && state.crossfadeDurationS > 0) {
      const prevClip = clipById.get(state.fromClipId);
      hermiteWeight = hermiteFade(state.crossfadeElapsedS ?? 0, state.crossfadeDurationS);
      if (prevClip && hermiteWeight < 1) {
        clips.push({
          clip: prevClip.index,
          time: state.fromClipTimeS ?? 0,
          weight: 1,
        });
      }
    }
    if (primaryClip) {
      const rawTime = Number(state.timeS ?? 0);
      const clipTime = primaryClip.looping && primaryClip.durationS > 0
        ? ((rawTime % primaryClip.durationS) + primaryClip.durationS) % primaryClip.durationS
        : Math.min(primaryClip.durationS, Math.max(0, rawTime));
      clips.push({
        clip: primaryClip.index,
        time: clipTime,
        weight: clips.length ? hermiteWeight : 1,
      });
    }
  }

  const lookAt = [];
  if (doc.lookAt.enabled) {
    const nodes = doc.lookAt.chain
      .map((entry) => [jointByName.get(entry.joint), entry.weight])
      .filter(([idx]) => idx != null);
    if (nodes.length) {
      lookAt.push({
        targetModel: doc.lookAt.target,
        originModel: doc.lookAt.origin,
        forwardLocal: [0, 0, 1],
        upLocal: [0, 1, 0],
        maxYaw: doc.lookAt.maxYawDeg * DEG_TO_RAD,
        maxPitch: doc.lookAt.maxPitchDeg * DEG_TO_RAD,
        weight: 1,
        nodes,
      });
    }
  }

  const twoBoneIk = [];
  for (const ik of doc.ikChains) {
    if (!ik.enabled || ik.weight <= 0) continue;
    const root = jointByName.get(ik.rootJoint);
    const mid = jointByName.get(ik.midJoint);
    const end = jointByName.get(ik.endJoint);
    if (root == null || mid == null || end == null) continue;
    twoBoneIk.push({
      root,
      mid,
      end,
      targetModel: ik.target,
      poleModel: ik.poleVector,
      weight: Math.min(1, Math.max(0, ik.weight)),
    });
  }

  const boneOffsets = [];
  for (const [jointName, off] of Object.entries(doc.boneOffsets)) {
    const node = jointByName.get(jointName);
    if (node == null || !off) continue;
    const deg = off.eulerDeg ?? [0, 0, 0];
    const rotation = quatFromEuler(deg[0] * DEG_TO_RAD, deg[1] * DEG_TO_RAD, deg[2] * DEG_TO_RAD);
    boneOffsets.push({
      node,
      translation: off.translation ?? [0, 0, 0],
      rotation,
      scale: [1, 1, 1],
    });
  }

  const intensity = Number.isFinite(doc.expressionIntensity) ? doc.expressionIntensity : 1.0;
  const morphOverrides = [];
  for (const target of doc.morphTargets) {
    const rawWeight = doc.morphWeights[target.id] ?? target.defaultWeight ?? 0;
    const scaled = target.group === 'expression' ? rawWeight * intensity : rawWeight;
    const clamped = Math.min(target.max, Math.max(target.min, scaled));
    morphOverrides.push({
      node: target.node,
      target: target.index,
      weight: clamped,
    });
  }

  return {
    resetToRest: true,
    commitUpdate: true,
    clips,
    crossfade: null,
    lookAt,
    twoBoneIk,
    boneOffsets,
    morphOverrides,
    hermiteWeight,
  };
}

function applyOpLocally(root, op) {
  if (op.set && op.set.length) {
    let cur = root;
    for (let i = 0; i < op.set.length - 1; i++) {
      const k = op.set[i];
      if (cur[k] == null || typeof cur[k] !== 'object') cur[k] = {};
      cur = cur[k];
    }
    cur[op.set[op.set.length - 1]] = op.value;
  } else if (op.remove && op.remove.length) {
    let cur = root;
    for (let i = 0; i < op.remove.length - 1; i++) {
      cur = cur?.[op.remove[i]];
      if (!cur) return;
    }
    delete cur[op.remove[op.remove.length - 1]];
  }
}

export function mountCharacterStudio(container, studioApi = {}) {
  const dom = container.ownerDocument || globalThis.document;
  let activeDocId = null;
  let docData = createDefaultCharacterDocument();
  let selectedJoint = docData.skeleton.joints[0]?.name ?? 'hips';
  let boneQuery = '';
  let lightingPreset = 'studio_neutral';
  let clipId = docData.clips[0]?.id ?? '';
  let targetClipId = docData.clips[1]?.id ?? docData.clips[0]?.id ?? '';
  let fromClipId = null;
  let crossfadeMs = 220;
  let crossfadeElapsedS = 0;
  let timeS = 0;
  let playing = false;
  let active = true;
  let destroyed = false;

  // Tri-Modal Articulated Graph, DNA & ASM state
  let stageInteractionMode = 'orbit'; // 'orbit' | 'sculpt_dna' | 'drag_ik'
  let activeDnaCategory = 'faceStructure';
  let solverMode = 'project'; // 'project' (Auto-Project x_hat -> x*) | 'reject' (Strict Rollback)
  let lastHallucinationGhosts = [];
  let lastSolverBanner = 'Verified: 0 constraint violations · C_IK ∩ C_cone ∩ C_occ active';
  let lastLiftedRegionLabel = '';
  const asmController = createAnimStateMachineController(docData.animStateMachine);

  // Persistent 3D Character Viewport Stage state
  let stageYaw = 0.24;
  let stagePitch = 0.14;
  let stageZoom = 1.05;
  let stagePanX = 0;
  let stagePanY = 0;
  let stageFocus = 'body';
  let stageShading = 'textured';
  let stageTurntable = false;
  let stageRaf = 0;
  let activeModelUrl = '';
  let activeParsedGlb = null;
  let stageStatusText = '';
  const glbByUrl = new Map();

  const unsubscribers = [];

  const el = (tag, className, text) => {
    const node = dom.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  };

  const resolveCharacterModelUrl = (rawPath) => {
    const clean = String(rawPath || '').trim();
    if (!clean) return '';
    if (clean.startsWith('/') || clean.startsWith('http://') || clean.startsWith('https://')) {
      return clean;
    }
    if (typeof studioApi.resolveAssetUrl === 'function') {
      return studioApi.resolveAssetUrl(clean);
    }
    return `/project/${clean.split('/').map(encodeURIComponent).join('/')}`;
  };

  const currentOutfit = () =>
    docData.outfits.find((o) => o.id === docData.activeOutfitId) ?? docData.outfits[0] ?? null;

  // Create persistent canvas once so dragging sliders never recreates or flickers the 3D stage
  const stageCanvas = el('canvas', 'gb-char-viewport-canvas');
  stageCanvas.dataset.role = 'character-stage-canvas';
  const hasRealCanvas = typeof stageCanvas.getContext === 'function';

  const repaintStage = () => {
    if (destroyed || !hasRealCanvas) return;
    const cssW = Math.max(260, stageCanvas.clientWidth || 480);
    const cssH = Math.max(220, stageCanvas.clientHeight || 360);
    drawGlbPreviewCanvas(stageCanvas, activeParsedGlb, {
      width: cssW,
      height: cssH,
      yaw: stageYaw + (Number(docData.skeleton?.placement?.yawRad) || 0),
      pitch: stagePitch,
      zoom: stageZoom,
      panX: stagePanX,
      panY: stagePanY,
      shading: stageShading,
      focus: stageFocus,
      onTextureLoaded: repaintStage,
    });

    const ctx = stageCanvas.getContext('2d');
    if (!ctx) return;
    const dpr = typeof window !== 'undefined' ? Math.min(2, window.devicePixelRatio || 1) : 1;
    ctx.save?.();
    ctx.setTransform?.(dpr, 0, 0, dpr, 0, 0);

    // Top-left model / outfit badge
    const outfit = currentOutfit();
    const topLabel = activeParsedGlb
      ? `${outfit?.label || docData.name} · ${activeParsedGlb.triangles.toLocaleString()} tris`
      : stageStatusText || 'Loading 3D character mesh…';
    ctx.fillStyle = 'rgba(14, 17, 24, 0.76)';
    ctx.fillRect?.(10, 10, Math.min(cssW - 20, Math.max(150, topLabel.length * 6.4 + 18)), 24);
    ctx.fillStyle = '#e8ecf4';
    ctx.font = '600 11px ui-sans-serif, system-ui, sans-serif';
    ctx.fillText?.(topLabel, 18, 25);

    // Live Look-At reticle indicator when enabled
    if (docData.lookAt?.enabled) {
      const tx = Math.min(1, Math.max(-1, (docData.lookAt.target[0] ?? 0) / 1.5));
      const ty = Math.min(
        1,
        Math.max(-1, ((docData.lookAt.target[1] ?? 1.55) - (docData.lookAt.origin[1] ?? 1.55)) / 1.0),
      );
      const gx = cssW * 0.5 + tx * (cssW * 0.28);
      const gy = cssH * (stageFocus === 'bust' ? 0.42 : 0.26) - ty * (cssH * 0.2);
      ctx.strokeStyle = 'rgba(232, 180, 74, 0.85)';
      ctx.lineWidth = 1.4;
      ctx.beginPath?.();
      ctx.arc?.(gx, gy, 6, 0, Math.PI * 2);
      ctx.stroke?.();
    }

    // Hallucination Ghost (`x_hat` red dashed circle -> `x*` projected emerald target)
    if (lastHallucinationGhosts.length > 0) {
      for (const ghost of lastHallucinationGhosts) {
        const hx = cssW * 0.5 + clamp((ghost.proposed[0] ?? 0) / 1.2, -0.44, 0.44) * cssW;
        const hy = cssH * 0.55 - clamp(((ghost.proposed[1] ?? 1.15) - 1.1) / 1.2, -0.4, 0.4) * cssH;
        const px = cssW * 0.5 + clamp((ghost.projected[0] ?? 0) / 1.2, -0.44, 0.44) * cssW;
        const py = cssH * 0.55 - clamp(((ghost.projected[1] ?? 1.15) - 1.1) / 1.2, -0.4, 0.4) * cssH;

        ctx.strokeStyle = 'rgba(248, 113, 113, 0.92)';
        ctx.lineWidth = 1.6;
        ctx.beginPath?.();
        ctx.moveTo?.(hx, hy);
        ctx.lineTo?.(px, py);
        ctx.stroke?.();

        ctx.fillStyle = 'rgba(248, 113, 113, 0.35)';
        ctx.beginPath?.();
        ctx.arc?.(hx, hy, 8, 0, Math.PI * 2);
        ctx.fill?.();

        ctx.fillStyle = 'rgba(52, 211, 153, 0.9)';
        ctx.beginPath?.();
        ctx.arc?.(px, py, 5, 0, Math.PI * 2);
        ctx.fill?.();
      }
    }

    // Bottom status bar showing active pose / morph / DNA / ASM summary
    const activeMorphs = Object.entries(docData.morphWeights || {})
      .filter(([, w]) => Math.abs(Number(w) || 0) > 0.02)
      .slice(0, 3)
      .map(([k, w]) => `${k}:${Number(w).toFixed(2)}`);
    const boneOffCount = Object.keys(docData.boneOffsets || {}).length;
    const asmSnap = asmController.getState();
    const footerParts = [
      `ASM: ${asmSnap.currentState}`,
      `Clip: ${clipId || 'rest'} (${timeS.toFixed(2)}s)`,
      `Mode: ${stageInteractionMode}`,
    ];
    if (lastLiftedRegionLabel) footerParts.push(`Lifted: ${lastLiftedRegionLabel}`);
    if (boneOffCount > 0) footerParts.push(`${boneOffCount} bone offset${boneOffCount > 1 ? 's' : ''}`);
    if (activeMorphs.length) footerParts.push(activeMorphs.join(' '));
    const footerText = footerParts.join(' · ');
    ctx.fillStyle = 'rgba(14, 17, 24, 0.72)';
    ctx.fillRect?.(10, cssH - 30, Math.min(cssW - 20, Math.max(160, footerText.length * 6.0 + 18)), 20);
    ctx.fillStyle = '#aeb6c6';
    ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
    ctx.fillText?.(footerText, 18, cssH - 16);
    ctx.restore?.();
  };

  const ensureOutfitModelLoaded = async () => {
    if (destroyed || !hasRealCanvas || typeof globalThis.fetch !== 'function') return;
    const outfit = currentOutfit();
    const candidates = [];
    if (outfit?.modelPath) candidates.push(resolveCharacterModelUrl(outfit.modelPath));
    if (typeof studioApi.listAssets === 'function') {
      const firstCharAsset = studioApi
        .listAssets({ kind: 'model' })
        .find((a) => String(a.path || '').toLowerCase().includes('char') || String(a.path || '').endsWith('.glb'));
      if (firstCharAsset) candidates.push(resolveCharacterModelUrl(firstCharAsset.path));
    }
    const targetUrl = candidates[0] || '';
    if (!targetUrl) {
      activeParsedGlb = null;
      stageStatusText = 'No 3D .glb assigned to active outfit';
      repaintStage();
      return;
    }
    if (glbByUrl.has(targetUrl)) {
      activeModelUrl = targetUrl;
      activeParsedGlb = glbByUrl.get(targetUrl);
      stageStatusText = '';
      repaintStage();
      return;
    }
    activeModelUrl = targetUrl;
    activeParsedGlb = null;
    stageStatusText = `Loading ${outfit?.label || 'character'}…`;
    repaintStage();

    for (const url of candidates) {
      if (!url) continue;
      if (glbByUrl.has(url)) {
        if (destroyed) return;
        activeModelUrl = url;
        activeParsedGlb = glbByUrl.get(url);
        stageStatusText = '';
        repaintStage();
        return;
      }
      try {
        const res = await globalThis.fetch(url);
        if (!res?.ok) continue;
        const buf = await res.arrayBuffer();
        const parsed = (await parseGlbPreviewAsync(buf, { maxFullTris: 140000 })) || parseGlbPreview(buf);
        if (parsed) {
          glbByUrl.set(url, parsed);
          if (!destroyed && activeModelUrl === targetUrl) {
            activeModelUrl = url;
            activeParsedGlb = parsed;
            stageStatusText = '';
            repaintStage();
          }
          return;
        }
      } catch {
        // try next candidate
      }
    }
    if (!destroyed && activeModelUrl === targetUrl) {
      stageStatusText = `Model not found: ${outfit?.modelPath || targetUrl}`;
      repaintStage();
    }
  };

  if (hasRealCanvas) {
    let dragging = false;
    let panMode = false;
    let activeSculptRegion = 'nose';
    let lx = 0;
    let ly = 0;
    stageCanvas.addEventListener('contextmenu', (ev) => ev.preventDefault?.());
    stageCanvas.addEventListener('pointerdown', (ev) => {
      dragging = true;
      panMode = ev.button === 2 || ev.button === 1 || Boolean(ev.shiftKey);
      lx = ev.clientX ?? 0;
      ly = ev.clientY ?? 0;
      const rect = stageCanvas.getBoundingClientRect?.() ?? { left: 0, top: 0, width: 480, height: 360 };
      const nx = rect.width > 0 ? (lx - rect.left) / rect.width : 0.5;
      const ny = rect.height > 0 ? (ly - rect.top) / rect.height : 0.5;
      activeSculptRegion = hitTestStageDragRegion(nx, ny, stageFocus);
      stageCanvas.setPointerCapture?.(ev.pointerId);
    });
    stageCanvas.addEventListener('pointermove', (ev) => {
      if (!dragging) return;
      const dx = (ev.clientX ?? 0) - lx;
      const dy = (ev.clientY ?? 0) - ly;
      lx = ev.clientX ?? 0;
      ly = ev.clientY ?? 0;

      if (stageInteractionMode === 'sculpt_dna' && !panMode) {
        const lifted = liftCharacterStageDragToSemanticGraph(docData, {
          kind: 'dna_region',
          regionId: activeSculptRegion,
          dxNorm: dx / 240,
          dyNorm: dy / 240,
        });
        docData = lifted.document;
        lastLiftedRegionLabel = STAGE_DRAG_REGIONS[activeSculptRegion]?.label ?? activeSculptRegion;
        commitOps(
          [
            { set: ['dna'], value: docData.dna },
            { set: ['morphWeights'], value: docData.morphWeights },
          ],
          `3D Sculpt ${lastLiftedRegionLabel}`,
          true,
          { skipRebuild: true },
        );
        return;
      }

      if (stageInteractionMode === 'drag_ik' && !panMode) {
        const curTarget = docData.ikChains[0]?.target ?? [0.32, 1.15, 0.38];
        const proposedTarget = [
          Number((curTarget[0] + dx * 0.004).toFixed(3)),
          Number((curTarget[1] - dy * 0.004).toFixed(3)),
          curTarget[2],
        ];
        const lifted = liftCharacterStageDragToSemanticGraph(
          docData,
          { kind: 'ik_target', chainIndex: 0, target: proposedTarget, enabled: true },
          { solverMode },
        );
        docData = lifted.document;
        lastHallucinationGhosts = lifted.projection.hallucinationGhosts;
        lastSolverBanner = lifted.projection.verified
          ? 'Verified: Inverse Lifting R³ → G_char (0 violations)'
          : `Projected x̂ → x* (${lifted.projection.violations.length} constraint projection)`;
        commitOps(
          buildIkChainOps(0, { enabled: true, target: docData.ikChains[0].target }),
          '3D Drag IK Effector',
          true,
          { skipRebuild: true },
        );
        return;
      }

      if (panMode) {
        stagePanX += dx;
        stagePanY += dy;
      } else {
        stageYaw -= dx * 0.01;
        stagePitch = Math.max(-1.15, Math.min(1.15, stagePitch + dy * 0.01));
      }
      repaintStage();
    });
    const endDrag = (ev) => {
      if (!dragging) return;
      dragging = false;
      try {
        stageCanvas.releasePointerCapture?.(ev.pointerId);
      } catch {
        /* ignore */
      }
      if (stageInteractionMode !== 'orbit') {
        render();
      }
    };
    stageCanvas.addEventListener('pointerup', endDrag);
    stageCanvas.addEventListener('pointercancel', endDrag);
    stageCanvas.addEventListener('wheel', (ev) => {
      ev.preventDefault?.();
      const factor = (ev.deltaY ?? 0) < 0 ? 1.12 : 1 / 1.12;
      stageZoom = Math.max(0.35, Math.min(6.0, stageZoom * factor));
      repaintStage();
    });
  }

  const stepStageAnimation = () => {
    if (destroyed || !active || (!stageTurntable && !playing)) {
      stageRaf = 0;
      return;
    }
    if (stageTurntable) stageYaw += 0.012;
    if (playing) {
      const asmSnap = asmController.step(1 / 60);
      if (docData.clips.some((c) => c.id === asmSnap.clipId)) {
        clipId = asmSnap.clipId;
      }
      const clipObj = docData.clips.find((c) => c.id === clipId) ?? docData.clips[0];
      const dur = Math.max(0.2, clipObj?.durationS ?? 2.0);
      timeS = clipObj?.looping !== false ? (timeS + 1 / 60) % dur : Math.min(dur, timeS + 1 / 60);
      pushPoseToEngine();
    }
    repaintStage();
    if (typeof requestAnimationFrame === 'function') {
      stageRaf = requestAnimationFrame(stepStageAnimation);
    }
  };

  const ensureAnimLoop = () => {
    if ((stageTurntable || playing) && !stageRaf && typeof requestAnimationFrame === 'function') {
      stageRaf = requestAnimationFrame(stepStageAnimation);
    }
  };

  const syncFromStudio = () => {
    const activeDoc = studioApi.getActiveDocument?.();
    if (
      activeDoc &&
      (String(activeDoc.path ?? activeDoc.id ?? '').endsWith('.character.json') ||
        activeDoc.root?.schema === CHARACTER_SCHEMA)
    ) {
      activeDocId = activeDoc.id ?? activeDoc.path;
      docData = normalizeCharacterDocument(activeDoc.root ?? activeDoc.data ?? activeDoc);
      asmController.setDefinition(docData.animStateMachine);
      if (!docData.skeleton.joints.some((j) => j.name === selectedJoint)) {
        selectedJoint = docData.skeleton.joints[0]?.name ?? '';
      }
      if (!docData.clips.some((c) => c.id === clipId)) {
        clipId = docData.clips[0]?.id ?? '';
      }
    }
  };

  const commitOps = (ops, label, merge = false, { skipRebuild = false } = {}) => {
    if (!ops.length) return;
    for (const op of ops) applyOpLocally(docData, op);
    docData = normalizeCharacterDocument(docData);
    if (activeDocId && typeof studioApi.edit === 'function') {
      studioApi.edit(activeDocId, ops, label, merge);
    }
    pushPoseToEngine();
    if (skipRebuild) {
      repaintStage();
    } else {
      render();
      void ensureOutfitModelLoaded();
    }
  };

  const pushPoseToEngine = () => {
    const spec = evaluateCharacterPoseSpec(docData, {
      clipId,
      fromClipId,
      crossfadeElapsedS,
      crossfadeDurationS: crossfadeMs / 1000,
      timeS,
    });
    const engine = studioApi.getEngine?.();
    const viewport = studioApi.getViewport?.();
    const instance = viewport?.characterInstance ?? engine?.activeCharacterInstance;
    if (engine && instance != null) {
      poseModelClip(engine, instance, spec);
    }
    if (viewport?.setLightingPreset && TURNTABLE_LIGHTING_PRESETS[lightingPreset]) {
      viewport.setLightingPreset(TURNTABLE_LIGHTING_PRESETS[lightingPreset]);
    }
    return spec;
  };

  const root = dom.createElement('div');
  root.className = 'gb-char-studio';
  container.innerHTML = '';
  container.appendChild(root);

  const render = () => {
    if (destroyed) return;
    root.innerHTML = '';

    // Toolbar
    const toolbar = el('div', 'gb-char-toolbar');
    toolbar.appendChild(el('span', 'gb-char-title', 'Character Studio'));
    toolbar.appendChild(el('span', 'gb-char-meta', `${docData.name} (${docData.id})`));

    // Project Character Document Switcher (when multiple `.character.json` files exist)
    const charDocs =
      typeof studioApi.listDocuments === 'function' ? studioApi.listDocuments({ suffix: '.character.json' }) : [];
    if (Array.isArray(charDocs) && charDocs.length > 0) {
      const docSelect = el('select', 'gb-char-select');
      docSelect.dataset.role = 'character-doc-select';
      for (const cd of charDocs) {
        const shortName = String(cd.path || cd.id)
          .split('/')
          .pop()
          .replace(/\.character\.json$/i, '');
        const opt = el('option', '', shortName);
        opt.value = cd.id;
        if (cd.id === activeDocId || cd.path === activeDocId) opt.selected = true;
        docSelect.appendChild(opt);
      }
      docSelect.addEventListener('change', async (ev) => {
        const nextId = ev.target.value;
        if (nextId && typeof studioApi.openDocument === 'function') {
          const opened = await studioApi.openDocument(nextId);
          if (opened) {
            activeDocId = opened.id ?? nextId;
            docData = normalizeCharacterDocument(opened.root ?? opened);
            asmController.setDefinition(docData.animStateMachine);
            selectedJoint = docData.skeleton.joints[0]?.name ?? '';
            clipId = docData.clips[0]?.id ?? '';
            render();
            pushPoseToEngine();
            void ensureOutfitModelLoaded();
          }
        }
      });
      toolbar.appendChild(docSelect);
    }

    toolbar.appendChild(el('div', 'gb-char-spacer'));

    const lightSelect = el('select', 'gb-char-select');
    lightSelect.dataset.role = 'lighting-select';
    for (const preset of Object.values(TURNTABLE_LIGHTING_PRESETS)) {
      const opt = el('option', '', preset.label);
      opt.value = preset.id;
      if (preset.id === lightingPreset) opt.selected = true;
      lightSelect.appendChild(opt);
    }
    lightSelect.addEventListener('change', (ev) => {
      lightingPreset = ev.target.value;
      pushPoseToEngine();
    });
    toolbar.appendChild(lightSelect);

    const resetBtn = el('button', 'gb-char-btn', 'Reset Pose');
    resetBtn.dataset.role = 'reset-pose';
    resetBtn.addEventListener('click', () => {
      lastHallucinationGhosts = [];
      commitOps([{ set: ['boneOffsets'], value: {} }], 'Reset bone offsets');
    });
    toolbar.appendChild(resetBtn);

    const saveBtn = el('button', 'gb-char-btn', 'Save');
    saveBtn.dataset.role = 'save-doc';
    saveBtn.addEventListener('click', () => {
      if (activeDocId) studioApi.save?.(activeDocId);
    });
    toolbar.appendChild(saveBtn);
    root.appendChild(toolbar);

    const grid = el('div', 'gb-char-grid');

    // Left column: Bone tree + Outfits & Sockets + 5-Layer Parametric Character DNA
    const leftCol = el('div', 'gb-char-col');

    const boneCard = el('div', 'gb-char-card');
    boneCard.appendChild(el('div', 'gb-char-card-header', `Skeleton (${docData.skeleton.joints.length} joints)`));
    const searchInput = el('input', 'gb-char-input');
    searchInput.type = 'search';
    searchInput.placeholder = 'Filter joints...';
    searchInput.value = boneQuery;
    searchInput.dataset.role = 'bone-search';
    searchInput.addEventListener('input', (ev) => {
      boneQuery = ev.target.value;
      render();
    });
    boneCard.appendChild(searchInput);

    const boneList = el('div', 'gb-char-bone-list');
    for (const joint of filterBoneTree(docData.skeleton.joints, boneQuery)) {
      const btn = el(
        'button',
        `gb-char-bone-item${joint.name === selectedJoint ? ' is-selected' : ''}`,
        `${'  '.repeat(joint.depth)}${joint.name}`,
      );
      btn.dataset.joint = joint.name;
      btn.addEventListener('click', () => {
        selectedJoint = joint.name;
        render();
      });
      boneList.appendChild(btn);
    }
    boneCard.appendChild(boneList);

    if (selectedJoint) {
      const curEuler = docData.boneOffsets[selectedJoint]?.eulerDeg ?? [0, 0, 0];
      const axes = ['Pitch', 'Yaw', 'Roll'];
      axes.forEach((axisLabel, axisIdx) => {
        const row = el('div', 'gb-char-slider-row');
        row.appendChild(el('span', '', `${selectedJoint} ${axisLabel}`));
        const slider = el('input');
        slider.type = 'range';
        slider.min = '-90';
        slider.max = '90';
        slider.step = '1';
        slider.value = String(curEuler[axisIdx] ?? 0);
        slider.dataset.role = `bone-euler-${axisIdx}`;
        const valSpan = el('span', 'gb-char-val', `${curEuler[axisIdx] ?? 0}°`);
        slider.addEventListener('input', (ev) => {
          const cur = docData.boneOffsets[selectedJoint]?.eulerDeg ?? [0, 0, 0];
          const next = [...cur];
          next[axisIdx] = Number(ev.target.value);
          valSpan.textContent = `${next[axisIdx]}°`;
          commitOps(
            buildBoneOffsetOps(selectedJoint, next, docData.boneOffsets[selectedJoint]?.translation ?? [0, 0, 0]),
            `Adjust ${selectedJoint} ${axisLabel}`,
            true,
            { skipRebuild: true },
          );
        });
        row.appendChild(slider);
        row.appendChild(valSpan);
        boneCard.appendChild(row);
      });
    }
    leftCol.appendChild(boneCard);

    // Outfit & Sockets card
    const outfitCard = el('div', 'gb-char-card');
    outfitCard.appendChild(el('div', 'gb-char-card-header', 'Outfits & Sockets'));
    const outfitSelect = el('select', 'gb-char-select');
    outfitSelect.dataset.role = 'outfit-select';
    for (const outfit of docData.outfits) {
      const opt = el('option', '', outfit.label);
      opt.value = outfit.id;
      if (outfit.id === docData.activeOutfitId) opt.selected = true;
      outfitSelect.appendChild(opt);
    }
    outfitSelect.addEventListener('change', (ev) => {
      commitOps(buildActiveOutfitOps(ev.target.value), 'Switch outfit variant');
    });
    outfitCard.appendChild(outfitSelect);

    docData.sockets.forEach((sock, sIdx) => {
      const sockRow = el(
        'div',
        'gb-char-meta',
        `Socket ${sock.id} → ${sock.parentJoint} [${sock.translation.map((n) => n.toFixed(2)).join(', ')}]`,
      );
      sockRow.dataset.socket = sock.id;
      outfitCard.appendChild(sockRow);
      const offsetSlider = el('input');
      offsetSlider.type = 'range';
      offsetSlider.min = '-0.5';
      offsetSlider.max = '0.5';
      offsetSlider.step = '0.01';
      offsetSlider.value = String(sock.translation[1] ?? 0);
      offsetSlider.dataset.role = `socket-y-${sIdx}`;
      offsetSlider.addEventListener('input', (ev) => {
        const curSock = docData.sockets[sIdx] ?? sock;
        const nextT = [...curSock.translation];
        nextT[1] = Number(ev.target.value);
        sockRow.textContent = `Socket ${curSock.id} → ${curSock.parentJoint} [${nextT.map((n) => n.toFixed(2)).join(', ')}]`;
        commitOps(buildSocketTransformOps(sIdx, { translation: nextT }), `Move socket ${sock.id}`, true, {
          skipRebuild: true,
        });
      });
      outfitCard.appendChild(offsetSlider);
    });
    leftCol.appendChild(outfitCard);

    // 5-Layer Parametric Character DNA & Wardrobe Anti-Clipping Card
    const dnaCard = el('div', 'gb-char-card');
    const dnaOcclusion = solveWardrobeOcclusionConstraints(docData.dna);
    dnaCard.appendChild(
      el(
        'div',
        'gb-char-card-header',
        `5-Layer Parametric DNA (Occ 0x${dnaOcclusion.bodyOcclusionMask.toString(16)})`,
      ),
    );

    const catSelect = el('select', 'gb-char-select');
    catSelect.dataset.role = 'dna-category-select';
    for (const cat of Object.values(DNA_SLIDER_CATEGORIES)) {
      const opt = el('option', '', cat.label);
      opt.value = cat.id;
      if (cat.id === activeDnaCategory) opt.selected = true;
      catSelect.appendChild(opt);
    }
    catSelect.addEventListener('change', (ev) => {
      activeDnaCategory = ev.target.value;
      render();
    });
    dnaCard.appendChild(catSelect);

    const activeCatMeta = DNA_SLIDER_CATEGORIES[activeDnaCategory] ?? DNA_SLIDER_CATEGORIES.faceStructure;
    for (const param of activeCatMeta.parameters) {
      const curVal =
        activeDnaCategory === 'bodyBuild'
          ? (docData.dna.bodyBuild[param.id] ?? param.defaultValue)
          : (docData.dna.craniofacial[activeDnaCategory]?.[param.id] ?? param.defaultValue);
      const row = el('div', 'gb-char-slider-row');
      row.appendChild(el('span', '', param.label));
      const pSlider = el('input');
      pSlider.type = 'range';
      pSlider.min = String(param.min);
      pSlider.max = String(param.max);
      pSlider.step = '0.02';
      pSlider.value = String(curVal);
      pSlider.dataset.role = `dna-slider-${param.id}`;
      const pVal = el('span', 'gb-char-val', Number(curVal).toFixed(2));
      pSlider.addEventListener('input', (ev) => {
        const nextV = Number(ev.target.value);
        pVal.textContent = nextV.toFixed(2);
        const path =
          activeDnaCategory === 'bodyBuild'
            ? ['dna', 'bodyBuild', param.id]
            : ['dna', 'craniofacial', activeDnaCategory, param.id];
        applyOpLocally(docData, { set: path, value: nextV });
        const compiled = compileDnaToRigOverrides(docData.dna);
        const ops = [
          { set: path, value: nextV },
          { set: ['skeleton', 'scale'], value: compiled.skeletonScale },
        ];
        if (param.morphId && compiled.morphWeights[param.morphId] !== undefined) {
          ops.push({ set: ['morphWeights', param.morphId], value: compiled.morphWeights[param.morphId] });
        }
        commitOps(ops, `DNA ${param.label}`, true, { skipRebuild: true });
      });
      row.appendChild(pSlider);
      row.appendChild(pVal);
      dnaCard.appendChild(row);
    }

    // Layer 2: Skin Melanin & Iris Heterochromia
    const melaninRow = el('div', 'gb-char-slider-row');
    melaninRow.appendChild(el('span', '', 'Skin Melanin'));
    const melaninSlider = el('input');
    melaninSlider.type = 'range';
    melaninSlider.min = '0';
    melaninSlider.max = '1';
    melaninSlider.step = '0.02';
    melaninSlider.value = String(docData.dna.skinEyesTeeth?.skinMelanin ?? 0.32);
    melaninSlider.dataset.role = 'dna-skin-melanin';
    const melaninVal = el('span', 'gb-char-val', Number(docData.dna.skinEyesTeeth?.skinMelanin ?? 0.32).toFixed(2));
    melaninSlider.addEventListener('input', (ev) => {
      const nextV = Number(ev.target.value);
      melaninVal.textContent = nextV.toFixed(2);
      commitOps([{ set: ['dna', 'skinEyesTeeth', 'skinMelanin'], value: nextV }], 'Skin Melanin', true, {
        skipRebuild: true,
      });
    });
    melaninRow.appendChild(melaninSlider);
    melaninRow.appendChild(melaninVal);
    dnaCard.appendChild(melaninRow);

    const heteroRow = el('div', 'gb-char-slider-row');
    heteroRow.appendChild(el('span', '', 'Heterochromia'));
    const heteroSlider = el('input');
    heteroSlider.type = 'range';
    heteroSlider.min = '0';
    heteroSlider.max = '1';
    heteroSlider.step = '0.05';
    heteroSlider.value = String(docData.dna.skinEyesTeeth?.irisHeterochromia ?? 0);
    heteroSlider.dataset.role = 'dna-iris-heterochromia';
    const heteroVal = el('span', 'gb-char-val', Number(docData.dna.skinEyesTeeth?.irisHeterochromia ?? 0).toFixed(2));
    heteroSlider.addEventListener('input', (ev) => {
      const nextV = Number(ev.target.value);
      heteroVal.textContent = nextV.toFixed(2);
      commitOps([{ set: ['dna', 'skinEyesTeeth', 'irisHeterochromia'], value: nextV }], 'Iris Heterochromia', true, {
        skipRebuild: true,
      });
    });
    heteroRow.appendChild(heteroSlider);
    heteroRow.appendChild(heteroVal);
    dnaCard.appendChild(heteroRow);

    // Layer 3: Physics Groom Spring-Bone Stiffness & Damping
    const stiffRow = el('div', 'gb-char-slider-row');
    stiffRow.appendChild(el('span', '', 'Groom Stiffness'));
    const stiffSlider = el('input');
    stiffSlider.type = 'range';
    stiffSlider.min = '0.05';
    stiffSlider.max = '1';
    stiffSlider.step = '0.05';
    stiffSlider.value = String(docData.dna.groom?.springStiffness ?? 0.62);
    stiffSlider.dataset.role = 'dna-groom-stiffness';
    const stiffVal = el('span', 'gb-char-val', Number(docData.dna.groom?.springStiffness ?? 0.62).toFixed(2));
    stiffSlider.addEventListener('input', (ev) => {
      const nextV = Number(ev.target.value);
      stiffVal.textContent = nextV.toFixed(2);
      commitOps([{ set: ['dna', 'groom', 'springStiffness'], value: nextV }], 'Groom Spring Stiffness', true, {
        skipRebuild: true,
      });
    });
    stiffRow.appendChild(stiffSlider);
    stiffRow.appendChild(stiffVal);
    dnaCard.appendChild(stiffRow);

    const dampRow = el('div', 'gb-char-slider-row');
    dampRow.appendChild(el('span', '', 'Groom Damping'));
    const dampSlider = el('input');
    dampSlider.type = 'range';
    dampSlider.min = '0.05';
    dampSlider.max = '1';
    dampSlider.step = '0.05';
    dampSlider.value = String(docData.dna.groom?.springDamping ?? 0.28);
    dampSlider.dataset.role = 'dna-groom-damping';
    const dampVal = el('span', 'gb-char-val', Number(docData.dna.groom?.springDamping ?? 0.28).toFixed(2));
    dampSlider.addEventListener('input', (ev) => {
      const nextV = Number(ev.target.value);
      dampVal.textContent = nextV.toFixed(2);
      commitOps([{ set: ['dna', 'groom', 'springDamping'], value: nextV }], 'Groom Spring Damping', true, {
        skipRebuild: true,
      });
    });
    dampRow.appendChild(dampSlider);
    dampRow.appendChild(dampVal);
    dnaCard.appendChild(dampRow);

    // Layer 4: Modular Wardrobe Slots (`inner`, `top`, `outer`) with Live Occlusion Mask
    for (const slotKey of ['inner', 'top', 'outer']) {
      const catalog = WARDROBE_SLOT_CATALOG[slotKey] ?? [];
      if (!catalog.length) continue;
      const wRow = el('div', 'gb-char-slider-row');
      wRow.appendChild(el('span', '', `Slot: ${slotKey}`));
      const wSel = el('select', 'gb-char-select');
      wSel.dataset.role = `dna-wardrobe-${slotKey}`;
      const curItemId = docData.dna.wardrobeSlots?.[slotKey]?.itemId ?? catalog[0].id;
      for (const item of catalog) {
        const opt = el('option', '', item.label);
        opt.value = item.id;
        if (item.id === curItemId) opt.selected = true;
        wSel.appendChild(opt);
      }
      wSel.addEventListener('change', (ev) => {
        const nextItemId = ev.target.value;
        applyOpLocally(docData, { set: ['dna', 'wardrobeSlots', slotKey, 'itemId'], value: nextItemId });
        const solved = solveWardrobeOcclusionConstraints(docData.dna);
        docData.dna = solved.dna;
        commitOps([{ set: ['dna'], value: docData.dna }], `Wardrobe ${slotKey} -> ${nextItemId}`);
      });
      wRow.appendChild(wSel);
      dnaCard.appendChild(wRow);
    }

    const occMeta = el(
      'div',
      'gb-char-meta',
      `Hidden skin regions (${dnaOcclusion.hiddenBodyRegions.length}): ${dnaOcclusion.hiddenBodyRegions.join(', ') || 'none'}`,
    );
    occMeta.dataset.role = 'dna-occlusion-summary';
    dnaCard.appendChild(occMeta);
    leftCol.appendChild(dnaCard);
    grid.appendChild(leftCol);

    // Center column: Live 3D Character Stage + Tri-Modal Semantic Graph Bar + Animation Clips + Look-At + Two-Bone IK
    const centerCol = el('div', 'gb-char-col');

    const stageCard = el('div', 'gb-char-stage-card');
    const stageHdr = el('div', 'gb-char-card-header');
    stageHdr.appendChild(el('span', '', '3D Character Turntable'));

    const stageControls = el('div', 'gb-char-stage-controls');
    const interactionModes = [
      ['orbit', 'Orbit'],
      ['sculpt_dna', '3D Sculpt DNA'],
      ['drag_ik', 'Drag IK (x̂→x*)'],
    ];
    for (const [mid, mlabel] of interactionModes) {
      const mBtn = el('button', `gb-char-btn${stageInteractionMode === mid ? ' is-active' : ''}`, mlabel);
      mBtn.dataset.role = `stage-mode-${mid}`;
      mBtn.addEventListener('click', () => {
        stageInteractionMode = mid;
        render();
        repaintStage();
      });
      stageControls.appendChild(mBtn);
    }

    const focusModes = [
      ['body', 'Full Body'],
      ['bust', 'Face / Bust'],
    ];
    for (const [fid, flabel] of focusModes) {
      const fBtn = el('button', `gb-char-btn${stageFocus === fid ? ' is-active' : ''}`, flabel);
      fBtn.dataset.role = `stage-focus-${fid}`;
      fBtn.addEventListener('click', () => {
        stageFocus = fid;
        stagePanX = 0;
        stagePanY = 0;
        render();
        repaintStage();
      });
      stageControls.appendChild(fBtn);
    }

    const shadingModes = [
      ['textured', 'Textured'],
      ['clay', 'Clay'],
      ['wireframe', 'Wire'],
      ['normals', 'Normals'],
    ];
    for (const [sid, slabel] of shadingModes) {
      const sBtn = el('button', `gb-char-btn${stageShading === sid ? ' is-active' : ''}`, slabel);
      sBtn.dataset.role = `stage-shade-${sid}`;
      sBtn.addEventListener('click', () => {
        stageShading = sid;
        render();
        repaintStage();
      });
      stageControls.appendChild(sBtn);
    }

    const turnBtn = el('button', `gb-char-btn${stageTurntable ? ' is-active' : ''}`, 'Turn');
    turnBtn.dataset.role = 'stage-turntable';
    turnBtn.addEventListener('click', () => {
      stageTurntable = !stageTurntable;
      render();
      ensureAnimLoop();
      repaintStage();
    });
    stageControls.appendChild(turnBtn);

    const resetCamBtn = el('button', 'gb-char-btn', 'Reset Cam');
    resetCamBtn.dataset.role = 'stage-reset-cam';
    resetCamBtn.addEventListener('click', () => {
      stageYaw = 0.24;
      stagePitch = 0.14;
      stageZoom = 1.05;
      stagePanX = 0;
      stagePanY = 0;
      repaintStage();
    });
    stageControls.appendChild(resetCamBtn);
    stageHdr.appendChild(stageControls);
    stageCard.appendChild(stageHdr);

    const stageWrap = el('div', 'gb-char-stage-wrap');
    stageWrap.appendChild(stageCanvas);
    stageCard.appendChild(stageWrap);

    const stageHint = el('div', 'gb-char-stage-hint');
    stageHint.appendChild(el('span', '', 'Left-drag: Orbit/Sculpt/IK · Right/Shift-drag: Pan · Wheel: Zoom'));
    const activeOutfitObj = currentOutfit();
    stageHint.appendChild(el('span', '', activeOutfitObj?.modelPath || ''));
    stageCard.appendChild(stageHint);
    centerCol.appendChild(stageCard);

    // Tri-Modal Articulated Semantic Graph & Constraint Projection Card
    const triCard = el('div', 'gb-char-card');
    triCard.appendChild(
      el('div', 'gb-char-card-header', 'Tri-Modal Articulated Semantic Graph (L: R³ → G_char & x̂ → x*)'),
    );

    const scenarioRow = el('div', 'gb-char-preset-strip');
    const scenarioSelect = el('select', 'gb-char-select');
    scenarioSelect.dataset.role = 'char-academic-preset';
    for (const sc of Object.values(CHARACTER_ACADEMIC_SCENARIOS)) {
      const opt = el('option', '', sc.label);
      opt.value = sc.id;
      scenarioSelect.appendChild(opt);
    }
    scenarioRow.appendChild(scenarioSelect);

    const solverModeSelect = el('select', 'gb-char-select');
    solverModeSelect.dataset.role = 'char-solver-mode';
    for (const [mId, mLabel] of [
      ['project', 'Solver: Auto-Project (x̂ → x*)'],
      ['reject', 'Solver: Strict Reject'],
    ]) {
      const opt = el('option', '', mLabel);
      opt.value = mId;
      if (mId === solverMode) opt.selected = true;
      solverModeSelect.appendChild(opt);
    }
    solverModeSelect.addEventListener('change', (ev) => {
      solverMode = ev.target.value;
    });
    scenarioRow.appendChild(solverModeSelect);

    const runScenarioBtn = el('button', 'gb-char-btn', '▶ Run Academic Preset');
    runScenarioBtn.dataset.role = 'char-run-scenario';
    runScenarioBtn.addEventListener('click', () => {
      const sc = CHARACTER_ACADEMIC_SCENARIOS[scenarioSelect.value] ?? CHARACTER_ACADEMIC_SCENARIOS.embodied_scholar_reach;
      if (sc.isTwoAgentConflict) {
        const resolved = resolveTwoAgentCharacterConflict(docData);
        docData = resolved.resolvedDocument;
        lastHallucinationGhosts = resolved.hallucinationGhosts;
        lastSolverBanner = `2-Agent Conflict Resolved: ${resolved.naiveConflictCount} collisions projected (x̂ → x*)`;
        commitOps(
          [
            { set: ['skeleton', 'placement'], value: docData.skeleton.placement },
            { set: ['ikChains'], value: docData.ikChains },
            { set: ['lookAt'], value: docData.lookAt },
          ],
          `Academic Preset: ${sc.label}`,
        );
      } else {
        const res = applyCharacterSemanticPatch(docData, sc.prompt, { solverMode });
        docData = res.document;
        lastHallucinationGhosts = res.projection.hallucinationGhosts;
        lastSolverBanner = res.rejected
          ? `Rejected (Rollback): ${res.projection.violations.length} constraint violations`
          : res.projection.verified
            ? `Preset Verified (${sc.id}): 0 violations`
            : `Preset Auto-Projected x̂ → x* (${res.projection.violations.length} constraints repaired)`;
        commitOps(
          [
            { set: ['ikChains'], value: docData.ikChains },
            { set: ['lookAt'], value: docData.lookAt },
            { set: ['dna'], value: docData.dna },
            { set: ['morphWeights'], value: docData.morphWeights },
          ],
          `Academic Preset: ${sc.label}`,
        );
      }
    });
    scenarioRow.appendChild(runScenarioBtn);
    triCard.appendChild(scenarioRow);

    const promptRow = el('div', 'gb-char-preset-strip');
    const promptInput = el('input', 'gb-char-input');
    promptInput.type = 'text';
    promptInput.placeholder = 'e.g. Reach for [0.95, 1.25, 0.85] with right hand, look at [0.4, 1.55, 1.8], tall scholar';
    promptInput.dataset.role = 'char-semantic-prompt';
    promptRow.appendChild(promptInput);

    const applyPromptBtn = el('button', 'gb-char-btn is-active', 'Apply Prompt');
    applyPromptBtn.dataset.role = 'char-apply-prompt';
    applyPromptBtn.addEventListener('click', () => {
      const res = applyCharacterSemanticPatch(docData, promptInput.value, { solverMode });
      docData = res.document;
      lastHallucinationGhosts = res.projection.hallucinationGhosts;
      lastSolverBanner = res.rejected
        ? `Rejected (Rollback): ${res.projection.violations.length} constraint violations`
        : res.projection.verified
          ? 'Verified: 0 constraint violations'
          : `Auto-Projected x̂ → x* (${res.projection.violations.length} articulated constraints repaired)`;
      commitOps(
        [
          { set: ['ikChains'], value: docData.ikChains },
          { set: ['lookAt'], value: docData.lookAt },
          { set: ['dna'], value: docData.dna },
          { set: ['morphWeights'], value: docData.morphWeights },
        ],
        'Apply Tri-Modal Character Prompt',
      );
    });
    promptRow.appendChild(applyPromptBtn);

    const simConflictBtn = el('button', 'gb-char-btn', '⚡ Simulate 2-Agent Conflict');
    simConflictBtn.dataset.role = 'char-simulate-conflict';
    simConflictBtn.addEventListener('click', () => {
      const resolved = resolveTwoAgentCharacterConflict(docData);
      docData = resolved.resolvedDocument;
      lastHallucinationGhosts = resolved.hallucinationGhosts;
      lastSolverBanner = `2-Agent Conflict Resolved: ${resolved.naiveConflictCount} collisions projected (x̂ → x*)`;
      commitOps(
        [
          { set: ['skeleton', 'placement'], value: docData.skeleton.placement },
          { set: ['ikChains'], value: docData.ikChains },
          { set: ['lookAt'], value: docData.lookAt },
        ],
        'Resolve 2-Agent Character Conflict',
      );
    });
    promptRow.appendChild(simConflictBtn);
    triCard.appendChild(promptRow);

    const solverStatus = el('div', 'gb-char-meta', lastSolverBanner);
    solverStatus.dataset.role = 'char-solver-status';
    triCard.appendChild(solverStatus);

    const currentSubgraph = getCharacterSemanticSubgraph(docData, { asmRuntime: asmController.getState() });
    const graphTextarea = el('textarea', 'gb-char-input gb-char-graph-editor');
    graphTextarea.dataset.role = 'char-semantic-graph-editor';
    graphTextarea.rows = 5;
    graphTextarea.value = JSON.stringify(
      { nodes: currentSubgraph.nodes.slice(0, 4), edges: currentSubgraph.edges.slice(0, 4) },
      null,
      2,
    );
    triCard.appendChild(graphTextarea);

    const applyGraphBtn = el('button', 'gb-char-btn', 'Apply Graph JSON (G_char → R³)');
    applyGraphBtn.dataset.role = 'char-apply-graph-json';
    applyGraphBtn.addEventListener('click', () => {
      try {
        const parsedPatch = JSON.parse(graphTextarea.value);
        const res = applyCharacterSemanticPatch(docData, parsedPatch, { solverMode });
        docData = res.document;
        lastHallucinationGhosts = res.projection.hallucinationGhosts;
        lastSolverBanner = res.rejected
          ? `Rejected Graph Patch: ${res.projection.violations.length} violations`
          : res.projection.verified
            ? 'Verified Graph Grounding G_char → R³ (0 violations)'
            : `Grounded & Projected G_char → R³ (${res.projection.violations.length} constraints repaired)`;
        commitOps(
          [
            { set: ['ikChains'], value: docData.ikChains },
            { set: ['lookAt'], value: docData.lookAt },
            { set: ['activeOutfitId'], value: docData.activeOutfitId },
          ],
          'Apply Character Semantic Graph JSON',
        );
      } catch (err) {
        lastSolverBanner = `Invalid JSON in Semantic Graph editor: ${err?.message ?? err}`;
        solverStatus.textContent = lastSolverBanner;
      }
    });
    triCard.appendChild(applyGraphBtn);
    centerCol.appendChild(triCard);

    const animCard = el('div', 'gb-char-card');
    animCard.appendChild(el('div', 'gb-char-card-header', 'Animation & Hermite Crossfade'));
    const clipSelect = el('select', 'gb-char-select');
    clipSelect.dataset.role = 'clip-select';
    for (const clip of docData.clips) {
      const opt = el('option', '', `${clip.label} (${clip.durationS.toFixed(2)}s)`);
      opt.value = clip.id;
      if (clip.id === clipId) opt.selected = true;
      clipSelect.appendChild(opt);
    }
    clipSelect.addEventListener('change', (ev) => {
      clipId = ev.target.value;
      timeS = 0;
      pushPoseToEngine();
      render();
      repaintStage();
    });
    animCard.appendChild(clipSelect);

    const activeClipObj = docData.clips.find((c) => c.id === clipId) ?? docData.clips[0];
    const scrubRow = el('div', 'gb-char-slider-row');
    const playBtn = el('button', `gb-char-btn${playing ? ' is-active' : ''}`, playing ? 'Pause' : 'Play');
    playBtn.dataset.role = 'clip-play';
    playBtn.addEventListener('click', () => {
      playing = !playing;
      render();
      ensureAnimLoop();
      repaintStage();
    });
    scrubRow.appendChild(playBtn);
    const scrubber = el('input');
    scrubber.type = 'range';
    scrubber.min = '0';
    scrubber.max = String(activeClipObj?.durationS ?? 2);
    scrubber.step = '0.01';
    scrubber.value = String(timeS);
    scrubber.dataset.role = 'clip-scrubber';
    const scrubVal = el('span', 'gb-char-val', `${timeS.toFixed(2)}s`);
    scrubber.addEventListener('input', (ev) => {
      timeS = Number(ev.target.value);
      scrubVal.textContent = `${timeS.toFixed(2)}s`;
      pushPoseToEngine();
      repaintStage();
    });
    scrubRow.appendChild(scrubber);
    scrubRow.appendChild(scrubVal);
    animCard.appendChild(scrubRow);

    const fadeRow = el('div', 'gb-char-slider-row');
    const triggerFadeBtn = el('button', 'gb-char-btn', 'Crossfade');
    triggerFadeBtn.dataset.role = 'trigger-crossfade';
    triggerFadeBtn.addEventListener('click', () => {
      fromClipId = clipId;
      clipId = targetClipId;
      crossfadeElapsedS = crossfadeMs / 2000;
      pushPoseToEngine();
      render();
      repaintStage();
    });
    fadeRow.appendChild(triggerFadeBtn);
    const fadeSlider = el('input');
    fadeSlider.type = 'range';
    fadeSlider.min = '40';
    fadeSlider.max = '800';
    fadeSlider.step = '10';
    fadeSlider.value = String(crossfadeMs);
    fadeSlider.dataset.role = 'crossfade-ms';
    const hVal = hermiteFade(crossfadeElapsedS, crossfadeMs / 1000);
    const fadeVal = el('span', 'gb-char-val', `${crossfadeMs}ms (${hVal.toFixed(2)})`);
    fadeSlider.addEventListener('input', (ev) => {
      crossfadeMs = Number(ev.target.value);
      const hv = hermiteFade(crossfadeElapsedS, crossfadeMs / 1000);
      fadeVal.textContent = `${crossfadeMs}ms (${hv.toFixed(2)})`;
    });
    fadeRow.appendChild(fadeSlider);
    fadeRow.appendChild(fadeVal);
    animCard.appendChild(fadeRow);
    centerCol.appendChild(animCard);

    // Look-At Pad
    const lookCard = el('div', 'gb-char-card');
    lookCard.appendChild(el('div', 'gb-char-card-header', 'Procedural Look-At Target'));
    const pad = el('div', 'gb-char-lookpad');
    pad.dataset.role = 'look-at-pad';
    const dot = el('div', 'gb-char-lookpad-dot');
    const tx = Math.min(1, Math.max(-1, (docData.lookAt.target[0] ?? 0) / 1.5));
    const ty = Math.min(
      1,
      Math.max(-1, ((docData.lookAt.target[1] ?? 1.55) - (docData.lookAt.origin[1] ?? 1.55)) / 1.0),
    );
    dot.style.left = `${(tx * 0.5 + 0.5) * 100}%`;
    dot.style.top = `${(0.5 - ty * 0.5) * 100}%`;
    pad.appendChild(dot);
    pad.addEventListener('click', (ev) => {
      const rect = pad.getBoundingClientRect?.() ?? { left: 0, top: 0, width: 200, height: 110 };
      const nx = rect.width > 0 ? ((ev.clientX ?? 100) - rect.left) / rect.width : 0.5;
      const ny = rect.height > 0 ? ((ev.clientY ?? 55) - rect.top) / rect.height : 0.5;
      const worldX = Number(((nx - 0.5) * 3.0).toFixed(3));
      const worldY = Number(((docData.lookAt.origin[1] ?? 1.55) + (0.5 - ny) * 2.0).toFixed(3));
      const worldZ = docData.lookAt.target[2] ?? 2.0;
      const lifted = liftCharacterStageDragToSemanticGraph(docData, {
        kind: 'look_at',
        target: [worldX, worldY, worldZ],
        enabled: true,
      });
      lastHallucinationGhosts = lifted.projection.hallucinationGhosts;
      commitOps(buildLookAtTargetOps(lifted.document.lookAt.target, true), 'Move Look-At target', true);
    });
    lookCard.appendChild(pad);
    lookCard.appendChild(
      el(
        'div',
        'gb-char-meta',
        `Chain: ${docData.lookAt.chain.map((c) => `${c.joint} (${Math.round(c.weight * 100)}%)`).join(' → ')}`,
      ),
    );
    centerCol.appendChild(lookCard);

    // Two-Bone IK Chains
    const ikCard = el('div', 'gb-char-card');
    ikCard.appendChild(el('div', 'gb-char-card-header', 'Analytical Two-Bone IK'));
    docData.ikChains.forEach((ik, idx) => {
      const toggle = el(
        'button',
        `gb-char-btn${ik.enabled ? ' is-active' : ''}`,
        `${ik.id} (${ik.enabled ? 'ON' : 'OFF'})`,
      );
      toggle.dataset.role = `ik-toggle-${idx}`;
      toggle.addEventListener('click', () => {
        commitOps(buildIkChainOps(idx, { enabled: !ik.enabled }), `Toggle IK ${ik.id}`);
      });
      ikCard.appendChild(toggle);
      const wRow = el('div', 'gb-char-slider-row');
      wRow.appendChild(el('span', '', 'IK Weight'));
      const wSlider = el('input');
      wSlider.type = 'range';
      wSlider.min = '0';
      wSlider.max = '1';
      wSlider.step = '0.05';
      wSlider.value = String(ik.weight);
      wSlider.dataset.role = `ik-weight-${idx}`;
      const wVal = el('span', 'gb-char-val', ik.weight.toFixed(2));
      wSlider.addEventListener('input', (ev) => {
        const nextW = Number(ev.target.value);
        wVal.textContent = nextW.toFixed(2);
        commitOps(buildIkChainOps(idx, { weight: nextW }), `Adjust IK ${ik.id}`, true, { skipRebuild: true });
      });
      wRow.appendChild(wSlider);
      wRow.appendChild(wVal);
      ikCard.appendChild(wRow);
    });
    centerCol.appendChild(ikCard);
    grid.appendChild(centerCol);

    // Right column: Circumstance ASM Live Simulator + Morphs & Expression Presets
    const rightCol = el('div', 'gb-char-col');

    const asmCard = el('div', 'gb-char-card');
    const asmSnap = asmController.getState();
    asmCard.appendChild(
      el('div', 'gb-char-card-header', `Circumstance ASM (${asmSnap.currentState})`),
    );
    const stateStrip = el('div', 'gb-char-preset-strip');
    for (const st of docData.animStateMachine.states) {
      const sBtn = el(
        'button',
        `gb-char-btn${st.id === asmSnap.currentState ? ' is-active' : ''}`,
        st.id,
      );
      sBtn.dataset.role = `asm-state-${st.id}`;
      sBtn.addEventListener('click', () => {
        const nextSnap = asmController.forceState(st.id, 200);
        if (docData.clips.some((c) => c.id === nextSnap.clipId)) {
          clipId = nextSnap.clipId;
        }
        render();
        pushPoseToEngine();
        repaintStage();
      });
      stateStrip.appendChild(sBtn);
    }
    asmCard.appendChild(stateStrip);

    const trigStrip = el('div', 'gb-char-preset-strip');
    for (const trigName of ['execute_action', 'applaud']) {
      const tBtn = el('button', 'gb-char-btn', `⚡ ${trigName}`);
      tBtn.dataset.role = `asm-trigger-${trigName}`;
      tBtn.addEventListener('click', () => {
        const nextSnap = asmController.fireTrigger(trigName);
        if (docData.clips.some((c) => c.id === nextSnap.clipId)) {
          clipId = nextSnap.clipId;
        }
        render();
        pushPoseToEngine();
        repaintStage();
      });
      trigStrip.appendChild(tBtn);
    }
    asmCard.appendChild(trigStrip);

    // Live Condition Parameters for Circumstance ASM (`locomotionSpeed`, `interactionDistance`, `reachActive`, `circumstance`)
    const asmParams = asmSnap.parameters ?? {};

    const speedRow = el('div', 'gb-char-slider-row');
    speedRow.appendChild(el('span', '', 'Locomotion Speed'));
    const speedSlider = el('input');
    speedSlider.type = 'range';
    speedSlider.min = '0';
    speedSlider.max = '4';
    speedSlider.step = '0.05';
    speedSlider.value = String(asmParams.locomotionSpeed ?? 0);
    speedSlider.dataset.role = 'asm-param-locomotionSpeed';
    const speedVal = el('span', 'gb-char-val', Number(asmParams.locomotionSpeed ?? 0).toFixed(2));
    speedSlider.addEventListener('input', (ev) => {
      const nextV = Number(ev.target.value);
      speedVal.textContent = nextV.toFixed(2);
      const nextSnap = asmController.setParameters({ locomotionSpeed: nextV });
      docData.animStateMachine.parameters = nextSnap.parameters;
      if (docData.clips.some((c) => c.id === nextSnap.clipId)) clipId = nextSnap.clipId;
      pushPoseToEngine();
      render();
      repaintStage();
    });
    speedRow.appendChild(speedSlider);
    speedRow.appendChild(speedVal);
    asmCard.appendChild(speedRow);

    const distRow = el('div', 'gb-char-slider-row');
    distRow.appendChild(el('span', '', 'Interact Dist (m)'));
    const distSlider = el('input');
    distSlider.type = 'range';
    distSlider.min = '0.1';
    distSlider.max = '2.5';
    distSlider.step = '0.05';
    distSlider.value = String(asmParams.interactionDistance ?? 1.8);
    distSlider.dataset.role = 'asm-param-interactionDistance';
    const distVal = el('span', 'gb-char-val', Number(asmParams.interactionDistance ?? 1.8).toFixed(2));
    distSlider.addEventListener('input', (ev) => {
      const nextV = Number(ev.target.value);
      distVal.textContent = nextV.toFixed(2);
      const nextSnap = asmController.setParameters({ interactionDistance: nextV });
      docData.animStateMachine.parameters = nextSnap.parameters;
      if (docData.clips.some((c) => c.id === nextSnap.clipId)) clipId = nextSnap.clipId;
      pushPoseToEngine();
      render();
      repaintStage();
    });
    distRow.appendChild(distSlider);
    distRow.appendChild(distVal);
    asmCard.appendChild(distRow);

    const reachToggleBtn = el(
      'button',
      `gb-char-btn${asmParams.reachActive ? ' is-active' : ''}`,
      `reachActive: ${asmParams.reachActive ? 'true' : 'false'}`,
    );
    reachToggleBtn.dataset.role = 'asm-param-reachActive';
    reachToggleBtn.addEventListener('click', () => {
      const nextReach = !asmParams.reachActive;
      const nextSnap = asmController.setParameters({ reachActive: nextReach });
      docData.animStateMachine.parameters = nextSnap.parameters;
      if (docData.clips.some((c) => c.id === nextSnap.clipId)) clipId = nextSnap.clipId;
      pushPoseToEngine();
      render();
      repaintStage();
    });
    asmCard.appendChild(reachToggleBtn);

    const circSelect = el('select', 'gb-char-select');
    circSelect.dataset.role = 'asm-param-circumstance';
    for (const cOpt of ['calm', 'focused', 'celebratory', 'alert']) {
      const opt = el('option', '', `Circumstance: ${cOpt}`);
      opt.value = cOpt;
      if (cOpt === (asmParams.circumstance ?? 'calm')) opt.selected = true;
      circSelect.appendChild(opt);
    }
    circSelect.addEventListener('change', (ev) => {
      const nextSnap = asmController.setParameters({ circumstance: ev.target.value });
      docData.animStateMachine.parameters = nextSnap.parameters;
      if (docData.clips.some((c) => c.id === nextSnap.clipId)) clipId = nextSnap.clipId;
      pushPoseToEngine();
      render();
      repaintStage();
    });
    asmCard.appendChild(circSelect);
    rightCol.appendChild(asmCard);

    const presetCard = el('div', 'gb-char-card');
    presetCard.appendChild(el('div', 'gb-char-card-header', 'Expression Presets'));
    const strip = el('div', 'gb-char-preset-strip');
    for (const presetName of Object.keys(docData.expressionPresets)) {
      const pBtn = el('button', 'gb-char-btn', presetName);
      pBtn.dataset.preset = presetName;
      pBtn.addEventListener('click', () => {
        commitOps(buildPresetApplyOps(docData, presetName), `Apply preset ${presetName}`);
      });
      strip.appendChild(pBtn);
    }
    presetCard.appendChild(strip);

    const intensityRow = el('div', 'gb-char-slider-row');
    intensityRow.appendChild(el('span', '', 'Intensity'));
    const intSlider = el('input');
    intSlider.type = 'range';
    intSlider.min = '0';
    intSlider.max = '2';
    intSlider.step = '0.05';
    intSlider.value = String(docData.expressionIntensity);
    intSlider.dataset.role = 'expression-intensity';
    const intVal = el('span', 'gb-char-val', `${docData.expressionIntensity.toFixed(2)}x`);
    intSlider.addEventListener('input', (ev) => {
      const v = Number(ev.target.value);
      intVal.textContent = `${v.toFixed(2)}x`;
      commitOps([{ set: ['expressionIntensity'], value: v }], 'Adjust expression intensity', true, {
        skipRebuild: true,
      });
    });
    intensityRow.appendChild(intSlider);
    intensityRow.appendChild(intVal);
    presetCard.appendChild(intensityRow);
    rightCol.appendChild(presetCard);

    const groups = groupMorphTargets(docData.morphTargets);
    for (const [groupName, targets] of Object.entries(groups)) {
      if (!targets.length) continue;
      const gCard = el('div', 'gb-char-card');
      gCard.appendChild(el('div', 'gb-char-card-header', `${groupName} (${targets.length})`));
      for (const target of targets) {
        const curVal = docData.morphWeights[target.id] ?? target.defaultWeight ?? 0;
        const row = el('div', 'gb-char-slider-row');
        row.appendChild(el('span', '', target.label));
        const mSlider = el('input');
        mSlider.type = 'range';
        mSlider.min = String(target.min);
        mSlider.max = String(target.max);
        mSlider.step = '0.01';
        mSlider.value = String(curVal);
        mSlider.dataset.morph = target.id;
        const mVal = el('span', 'gb-char-val', curVal.toFixed(2));
        mSlider.addEventListener('input', (ev) => {
          const nextV = Number(ev.target.value);
          mVal.textContent = nextV.toFixed(2);
          commitOps(buildMorphWeightOps(target.id, nextV), `Morph ${target.id}`, true, { skipRebuild: true });
        });
        row.appendChild(mSlider);
        row.appendChild(mVal);
        gCard.appendChild(row);
      }
      rightCol.appendChild(gCard);
    }

    grid.appendChild(rightCol);
    root.appendChild(grid);
    if (typeof requestAnimationFrame === 'function') {
      requestAnimationFrame(() => repaintStage());
    } else {
      repaintStage();
    }
  };

  syncFromStudio();
  render();
  pushPoseToEngine();
  void ensureOutfitModelLoaded();

  if (typeof studioApi.subscribe === 'function') {
    const unsub = studioApi.subscribe(() => {
      if (!active || destroyed) return;
      const focusedTag = dom.activeElement?.tagName;
      if (focusedTag === 'INPUT' || focusedTag === 'SELECT' || focusedTag === 'TEXTAREA') return;
      const prevId = activeDocId;
      const prevOutfit = docData.activeOutfitId;
      syncFromStudio();
      if (activeDocId !== prevId || docData.activeOutfitId !== prevOutfit) {
        render();
        void ensureOutfitModelLoaded();
      }
    });
    if (typeof unsub === 'function') unsubscribers.push(unsub);
  }

  const destroy = () => {
    if (destroyed) return;
    destroyed = true;
    playing = false;
    stageTurntable = false;
    if (stageRaf && typeof cancelAnimationFrame === 'function') {
      cancelAnimationFrame(stageRaf);
      stageRaf = 0;
    }
    while (unsubscribers.length) {
      const fn = unsubscribers.pop();
      try {
        fn();
      } catch {
        // Ignore teardown errors from external listeners.
      }
    }
    container.innerHTML = '';
  };

  const controller = {
    destroy,
    unmount: destroy,
    refresh() {
      syncFromStudio();
      render();
      pushPoseToEngine();
      void ensureOutfitModelLoaded();
    },
    setDocument(pathOrId, rawDoc) {
      activeDocId = pathOrId;
      docData = normalizeCharacterDocument(rawDoc?.root ?? rawDoc);
      asmController.setDefinition(docData.animStateMachine);
      selectedJoint = docData.skeleton.joints[0]?.name ?? '';
      clipId = docData.clips[0]?.id ?? '';
      render();
      pushPoseToEngine();
      void ensureOutfitModelLoaded();
    },
    getCharacterSemanticSubgraph: () =>
      getCharacterSemanticSubgraph(docData, { asmRuntime: asmController.getState() }),
    applyCharacterSemanticPatch(patchOrPrompt, opts = {}) {
      const res = applyCharacterSemanticPatch(docData, patchOrPrompt, { solverMode, ...opts });
      docData = res.document;
      lastHallucinationGhosts = res.projection.hallucinationGhosts;
      render();
      pushPoseToEngine();
      return res;
    },
    projectCharacterIkConstraints(rigConstraints = {}) {
      const res = projectCharacterIkConstraints(docData, rigConstraints);
      docData = res.document;
      lastHallucinationGhosts = res.hallucinationGhosts;
      render();
      pushPoseToEngine();
      return res;
    },
    resolveTwoAgentConflict(agentAPatch, agentBPatch) {
      const res = resolveTwoAgentCharacterConflict(docData, agentAPatch, agentBPatch);
      docData = res.resolvedDocument;
      lastHallucinationGhosts = res.hallucinationGhosts;
      render();
      pushPoseToEngine();
      return res;
    },
    summarizeForAi: () => summarizeCharacterForAi(docData, asmController.getState()),
    onActivate() {
      active = true;
      syncFromStudio();
      render();
      pushPoseToEngine();
      void ensureOutfitModelLoaded();
    },
    onDeactivate() {
      active = false;
      playing = false;
      stageTurntable = false;
      if (stageRaf && typeof cancelAnimationFrame === 'function') {
        cancelAnimationFrame(stageRaf);
        stageRaf = 0;
      }
    },
  };

  if (typeof globalThis.window !== 'undefined') {
    globalThis.window.__characterStudio = controller;
  }

  return controller;
}
