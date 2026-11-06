/**
 * Parametric Character DNA System (`ganbate.character.dna/1`).
 *
 * Genre-agnostic 5-layer character customization, direct 3D stage region inverse lifting,
 * and wardrobe anti-clipping constraint projection:
 *   1. Craniofacial & Body Proportion DNA (`faceStructure`, `eyesBrows`, `nose`, `mouthJaw`, `bodyBuild`)
 *   2. Multi-Layer Skin, Iris & Decal Stack (`skin`, `iris` with heterochromia, `decals`)
 *   3. Physics Groom (`frontBangs`, `crown`, `backHair` + Spring-Bone `stiffness`/`damping`)
 *   4. Modular Wardrobe Slots (`inner`, `top`, `outer`, `bottom`, `footwear`, `accessory`)
 *      with `bodyOcclusionMask` bitmasks and layer-clearance constraint projection
 *   5. Direct 3D Stage Region Drag Inverse Lifting (`L_dna: R^3 -> G_dna`)
 *
 * Reusable headlessly in both Ganbate Studio and in-game Player Avatar Creation.
 */

export const CHARACTER_DNA_SCHEMA = 'ganbate.character.dna/1';

export const BODY_OCCLUSION_REGIONS = Object.freeze({
  NONE: 0,
  TORSO_UPPER: 1 << 0, // 1
  TORSO_LOWER: 1 << 1, // 2
  ARMS_UPPER: 1 << 2,  // 4
  ARMS_FOREARM: 1 << 3, // 8
  LEGS_UPPER: 1 << 4,  // 16
  LEGS_LOWER: 1 << 5,  // 32
  FEET: 1 << 6,        // 64
  NECK_COLLAR: 1 << 7, // 128
});

export const BODY_OCCLUSION_LABELS = Object.freeze({
  [BODY_OCCLUSION_REGIONS.TORSO_UPPER]: 'torso_upper',
  [BODY_OCCLUSION_REGIONS.TORSO_LOWER]: 'torso_lower',
  [BODY_OCCLUSION_REGIONS.ARMS_UPPER]: 'arms_upper',
  [BODY_OCCLUSION_REGIONS.ARMS_FOREARM]: 'arms_forearm',
  [BODY_OCCLUSION_REGIONS.LEGS_UPPER]: 'legs_upper',
  [BODY_OCCLUSION_REGIONS.LEGS_LOWER]: 'legs_lower',
  [BODY_OCCLUSION_REGIONS.FEET]: 'feet',
  [BODY_OCCLUSION_REGIONS.NECK_COLLAR]: 'neck_collar',
});

export const DNA_SLIDER_CATEGORIES = Object.freeze({
  faceStructure: {
    id: 'faceStructure',
    label: 'Face Structure',
    parameters: [
      { id: 'cranioWidth', label: 'Cranial Width', min: -1, max: 1, defaultValue: 0, morphId: 'head_width', bone: 'head', axis: 'scaleX' },
      { id: 'templeDepth', label: 'Temple Depth', min: -1, max: 1, defaultValue: 0, morphId: 'temple_depth' },
      { id: 'cheekboneHeight', label: 'Cheekbone Height', min: -1, max: 1, defaultValue: 0, morphId: 'cheek_height' },
      { id: 'cheekboneProminence', label: 'Cheekbone Prominence', min: -1, max: 1, defaultValue: 0, morphId: 'cheek_prominence' },
      { id: 'foreheadSlope', label: 'Forehead Slope', min: -1, max: 1, defaultValue: 0, morphId: 'forehead_slope' },
    ],
  },
  eyesBrows: {
    id: 'eyesBrows',
    label: 'Eyes & Brows',
    parameters: [
      { id: 'eyeSize', label: 'Eye Aperture', min: -1, max: 1, defaultValue: 0, morphId: 'eye_aperture' },
      { id: 'eyeSpacing', label: 'Interpupillary Spacing', min: -1, max: 1, defaultValue: 0, morphId: 'eye_spacing' },
      { id: 'eyeTilt', label: 'Canthal Tilt', min: -1, max: 1, defaultValue: 0, morphId: 'eye_tilt' },
      { id: 'browRidgeHeight', label: 'Brow Ridge Height', min: -1, max: 1, defaultValue: 0, morphId: 'brow_height' },
      { id: 'browArch', label: 'Brow Arch', min: -1, max: 1, defaultValue: 0, morphId: 'brow_arch' },
    ],
  },
  nose: {
    id: 'nose',
    label: 'Nose',
    parameters: [
      { id: 'bridgeHeight', label: 'Bridge Height', min: -1, max: 1, defaultValue: 0, morphId: 'nose_bridge' },
      { id: 'noseLength', label: 'Nose Length', min: -1, max: 1, defaultValue: 0, morphId: 'nose_length' },
      { id: 'nostrilWidth', label: 'Alar Width', min: -1, max: 1, defaultValue: 0, morphId: 'nostril_width' },
      { id: 'tipTilt', label: 'Tip Elevation', min: -1, max: 1, defaultValue: 0, morphId: 'nose_tip_tilt' },
    ],
  },
  mouthJaw: {
    id: 'mouthJaw',
    label: 'Mouth & Jaw',
    parameters: [
      { id: 'mouthWidth', label: 'Mouth Width', min: -1, max: 1, defaultValue: 0, morphId: 'mouth_width' },
      { id: 'lipFullness', label: 'Lip Fullness', min: -1, max: 1, defaultValue: 0, morphId: 'lip_fullness' },
      { id: 'jawWidth', label: 'Jaw Width', min: -1, max: 1, defaultValue: 0, morphId: 'jaw_width' },
      { id: 'chinProjection', label: 'Chin Projection', min: -1, max: 1, defaultValue: 0, morphId: 'chin_projection' },
      { id: 'jawAngle', label: 'Gonial Angle', min: -1, max: 1, defaultValue: 0, morphId: 'jaw_angle' },
    ],
  },
  bodyBuild: {
    id: 'bodyBuild',
    label: 'Body Build & Height',
    parameters: [
      { id: 'statureScale', label: 'Stature Scale', min: 0.82, max: 1.18, defaultValue: 1.0, bone: 'hips', axis: 'uniformScale' },
      { id: 'shoulderBreadth', label: 'Shoulder Breadth', min: -1, max: 1, defaultValue: 0, morphId: 'shoulder_breadth', bone: 'spine_01', axis: 'scaleX' },
      { id: 'torsoLength', label: 'Torso Proportion', min: -1, max: 1, defaultValue: 0, bone: 'spine_01', axis: 'scaleY' },
      { id: 'limbLength', label: 'Limb Reach Scale', min: -1, max: 1, defaultValue: 0, bone: 'upperarm_r', axis: 'scaleY' },
      { id: 'musculature', label: 'Musculature Tone', min: -1, max: 1, defaultValue: 0, morphId: 'muscle_tone' },
      { id: 'posturePoise', label: 'Posture Poise', min: -1, max: 1, defaultValue: 0, bone: 'spine_01', axis: 'pitchDeg' },
    ],
  },
});

export const STAGE_DRAG_REGIONS = Object.freeze({
  brow: {
    id: 'brow',
    label: 'Brow & Forehead',
    focusArea: 'bust',
    normalizedBox: { xMin: 0.36, xMax: 0.64, yMin: 0.14, yMax: 0.25 },
    mapDelta: (dx, dy) => ({
      'eyesBrows.browArch': dx * 1.6,
      'eyesBrows.browRidgeHeight': -dy * 1.8,
    }),
  },
  eyes: {
    id: 'eyes',
    label: 'Eyes & Canthal Tilt',
    focusArea: 'bust',
    normalizedBox: { xMin: 0.34, xMax: 0.66, yMin: 0.25, yMax: 0.34 },
    mapDelta: (dx, dy) => ({
      'eyesBrows.eyeSpacing': dx * 1.5,
      'eyesBrows.eyeTilt': -dy * 1.5,
    }),
  },
  nose: {
    id: 'nose',
    label: 'Nose Bridge & Tip',
    focusArea: 'bust',
    normalizedBox: { xMin: 0.43, xMax: 0.57, yMin: 0.34, yMax: 0.45 },
    mapDelta: (dx, dy) => ({
      'nose.nostrilWidth': dx * 1.7,
      'nose.bridgeHeight': -dy * 1.8,
      'nose.tipTilt': -dy * 1.2,
    }),
  },
  jaw: {
    id: 'jaw',
    label: 'Jawline & Chin',
    focusArea: 'bust',
    normalizedBox: { xMin: 0.33, xMax: 0.67, yMin: 0.45, yMax: 0.58 },
    mapDelta: (dx, dy) => ({
      'mouthJaw.jawWidth': dx * 1.8,
      'mouthJaw.chinProjection': -dy * 1.6,
    }),
  },
  shoulders: {
    id: 'shoulders',
    label: 'Shoulder Frame & Stature',
    focusArea: 'body',
    normalizedBox: { xMin: 0.22, xMax: 0.78, yMin: 0.24, yMax: 0.48 },
    mapDelta: (dx, dy) => ({
      'bodyBuild.shoulderBreadth': dx * 1.8,
      'bodyBuild.statureScale': -dy * 0.22,
    }),
  },
  torso: {
    id: 'torso',
    label: 'Torso Build & Poise',
    focusArea: 'body',
    normalizedBox: { xMin: 0.28, xMax: 0.72, yMin: 0.48, yMax: 0.76 },
    mapDelta: (dx, dy) => ({
      'bodyBuild.musculature': dx * 1.6,
      'bodyBuild.torsoLength': -dy * 1.4,
    }),
  },
});

export const WARDROBE_SLOT_CATALOG = Object.freeze({
  inner: [
    {
      id: 'inner_silk_chemise',
      slot: 'inner',
      label: 'Fitted Silk Base Layer',
      layerThicknessMm: 1.5,
      bodyOcclusionMask: BODY_OCCLUSION_REGIONS.TORSO_UPPER,
    },
    {
      id: 'inner_high_collar_knit',
      slot: 'inner',
      label: 'High-Collar Thermal Knit',
      layerThicknessMm: 3.2,
      bodyOcclusionMask:
        BODY_OCCLUSION_REGIONS.TORSO_UPPER |
        BODY_OCCLUSION_REGIONS.ARMS_UPPER |
        BODY_OCCLUSION_REGIONS.NECK_COLLAR,
    },
  ],
  top: [
    {
      id: 'top_tailored_vest',
      slot: 'top',
      label: 'Tailored Studio Vest',
      layerThicknessMm: 4.5,
      minInnerClearanceMm: 2.0,
      bodyOcclusionMask:
        BODY_OCCLUSION_REGIONS.TORSO_UPPER | BODY_OCCLUSION_REGIONS.TORSO_LOWER,
    },
    {
      id: 'top_pleated_tunic',
      slot: 'top',
      label: 'Architectural Pleated Tunic',
      layerThicknessMm: 5.0,
      minInnerClearanceMm: 3.5,
      bodyOcclusionMask:
        BODY_OCCLUSION_REGIONS.TORSO_UPPER |
        BODY_OCCLUSION_REGIONS.TORSO_LOWER |
        BODY_OCCLUSION_REGIONS.ARMS_UPPER,
    },
  ],
  outer: [
    {
      id: 'outer_none',
      slot: 'outer',
      label: 'No Outer Layer',
      layerThicknessMm: 0,
      minTopClearanceMm: 100,
      bodyOcclusionMask: BODY_OCCLUSION_REGIONS.NONE,
    },
    {
      id: 'outer_structured_blazer',
      slot: 'outer',
      label: 'Structured Academic Blazer',
      layerThicknessMm: 8.0,
      minTopClearanceMm: 4.8,
      bodyOcclusionMask:
        BODY_OCCLUSION_REGIONS.TORSO_UPPER |
        BODY_OCCLUSION_REGIONS.TORSO_LOWER |
        BODY_OCCLUSION_REGIONS.ARMS_UPPER |
        BODY_OCCLUSION_REGIONS.ARMS_FOREARM,
    },
    {
      id: 'outer_haori_coat',
      slot: 'outer',
      label: 'Draped Pavilion Overcoat',
      layerThicknessMm: 9.5,
      minTopClearanceMm: 6.5,
      bodyOcclusionMask:
        BODY_OCCLUSION_REGIONS.TORSO_UPPER |
        BODY_OCCLUSION_REGIONS.TORSO_LOWER |
        BODY_OCCLUSION_REGIONS.ARMS_UPPER |
        BODY_OCCLUSION_REGIONS.ARMS_FOREARM,
    },
  ],
  bottom: [
    {
      id: 'bottom_tailored_trousers',
      slot: 'bottom',
      label: 'Tailored Pleated Trousers',
      layerThicknessMm: 4.0,
      bodyOcclusionMask:
        BODY_OCCLUSION_REGIONS.LEGS_UPPER | BODY_OCCLUSION_REGIONS.LEGS_LOWER,
    },
    {
      id: 'bottom_midi_drape',
      slot: 'bottom',
      label: 'Structured Midi Drape',
      layerThicknessMm: 4.2,
      bodyOcclusionMask: BODY_OCCLUSION_REGIONS.LEGS_UPPER,
    },
  ],
  footwear: [
    {
      id: 'footwear_oxford_loafers',
      slot: 'footwear',
      label: 'Polished Studio Loafers',
      layerThicknessMm: 5.0,
      bodyOcclusionMask: BODY_OCCLUSION_REGIONS.FEET,
    },
    {
      id: 'footwear_ankle_boots',
      slot: 'footwear',
      label: 'Sculpted Ankle Boots',
      layerThicknessMm: 6.0,
      bodyOcclusionMask: BODY_OCCLUSION_REGIONS.FEET | BODY_OCCLUSION_REGIONS.LEGS_LOWER,
    },
  ],
  accessory: [
    {
      id: 'accessory_none',
      slot: 'accessory',
      label: 'None',
      layerThicknessMm: 0,
      bodyOcclusionMask: BODY_OCCLUSION_REGIONS.NONE,
    },
    {
      id: 'accessory_chronograph',
      slot: 'accessory',
      label: 'Precision Wrist Chronograph',
      layerThicknessMm: 2.0,
      bodyOcclusionMask: BODY_OCCLUSION_REGIONS.NONE,
    },
    {
      id: 'accessory_scholar_frames',
      slot: 'accessory',
      label: 'Titanium Optical Frames',
      layerThicknessMm: 1.0,
      bodyOcclusionMask: BODY_OCCLUSION_REGIONS.NONE,
    },
  ],
});

const clamp = (val, min, max) => Math.min(max, Math.max(min, Number.isFinite(val) ? val : 0));

export function createDefaultCharacterDna() {
  return {
    schema: CHARACTER_DNA_SCHEMA,
    craniofacial: {
      faceStructure: {
        cranioWidth: 0,
        templeDepth: 0,
        cheekboneHeight: 0.1,
        cheekboneProminence: 0.08,
        foreheadSlope: 0,
      },
      eyesBrows: {
        eyeSize: 0.05,
        eyeSpacing: 0,
        eyeTilt: 0.06,
        browRidgeHeight: 0,
        browArch: 0.08,
      },
      nose: {
        bridgeHeight: 0.05,
        noseLength: 0,
        nostrilWidth: -0.04,
        tipTilt: 0.04,
      },
      mouthJaw: {
        mouthWidth: 0,
        lipFullness: 0.06,
        jawWidth: -0.05,
        chinProjection: 0.04,
        jawAngle: 0,
      },
    },
    bodyBuild: {
      statureScale: 1.0,
      shoulderBreadth: 0,
      torsoLength: 0,
      limbLength: 0,
      musculature: 0.08,
      posturePoise: 0.1,
    },
    skinIrisDecals: {
      skin: {
        melanin: 0.32,
        undertone: 'warm',
        subsurfaceWarmth: 0.64,
        poreRoughness: 0.38,
        sheen: 0.28,
      },
      iris: {
        leftOuterColor: '#3b4d5e',
        leftInnerRingColor: '#c89b54',
        rightOuterColor: '#3b4d5e',
        rightInnerRingColor: '#c89b54',
        heterochromia: false,
        pupilDilation: 0.42,
        scleraBrightness: 0.94,
      },
      decals: [
        {
          id: 'subtle_freckle_dust',
          kind: 'freckles',
          region: 'face',
          opacity: 0.18,
          tint: '#8b5e3c',
          uvScale: 1.0,
          enabled: true,
        },
      ],
    },
    physicsGroom: {
      frontBangs: { styleId: 'swept_curtain', length: 0.52 },
      crown: { styleId: 'layered_crown', volume: 0.48 },
      backHair: { styleId: 'tailored_taper', length: 0.58 },
      rootColor: '#1b1d24',
      tipColor: '#2f3544',
      highlightColor: '#66738f',
      anisotropy: 0.72,
      springBone: {
        stiffness: 165,
        damping: 0.34,
        gravityScale: 0.85,
        collisionRadius: 0.06,
      },
    },
    wardrobeSlots: {
      inner: {
        itemId: 'inner_silk_chemise',
        tint: '#f3efe6',
        roughness: 0.34,
        scaleOffsetMm: 0,
      },
      top: {
        itemId: 'top_tailored_vest',
        tint: '#1e293b',
        roughness: 0.48,
        scaleOffsetMm: 0,
      },
      outer: {
        itemId: 'outer_structured_blazer',
        tint: '#172033',
        roughness: 0.44,
        scaleOffsetMm: 0,
      },
      bottom: {
        itemId: 'bottom_tailored_trousers',
        tint: '#192130',
        roughness: 0.5,
        scaleOffsetMm: 0,
      },
      footwear: {
        itemId: 'footwear_oxford_loafers',
        tint: '#241c18',
        roughness: 0.3,
        scaleOffsetMm: 0,
      },
      accessory: {
        itemId: 'accessory_chronograph',
        tint: '#d4af37',
        roughness: 0.22,
        scaleOffsetMm: 0,
      },
    },
  };
}

export function normalizeCharacterDna(raw) {
  const base = createDefaultCharacterDna();
  if (!raw || typeof raw !== 'object') return base;

  for (const catKey of ['faceStructure', 'eyesBrows', 'nose', 'mouthJaw']) {
    const src = raw.craniofacial?.[catKey];
    if (src && typeof src === 'object') {
      for (const param of DNA_SLIDER_CATEGORIES[catKey].parameters) {
        if (src[param.id] !== undefined) {
          base.craniofacial[catKey][param.id] = clamp(Number(src[param.id]), param.min, param.max);
        }
      }
    }
  }

  if (raw.bodyBuild && typeof raw.bodyBuild === 'object') {
    for (const param of DNA_SLIDER_CATEGORIES.bodyBuild.parameters) {
      if (raw.bodyBuild[param.id] !== undefined) {
        base.bodyBuild[param.id] = clamp(Number(raw.bodyBuild[param.id]), param.min, param.max);
      }
    }
  }

  if (raw.skinIrisDecals && typeof raw.skinIrisDecals === 'object') {
    const sk = raw.skinIrisDecals.skin;
    if (sk && typeof sk === 'object') {
      base.skinIrisDecals.skin.melanin = clamp(Number(sk.melanin ?? base.skinIrisDecals.skin.melanin), 0, 1);
      if (['warm', 'neutral', 'cool', 'olive'].includes(sk.undertone)) {
        base.skinIrisDecals.skin.undertone = sk.undertone;
      }
      base.skinIrisDecals.skin.subsurfaceWarmth = clamp(
        Number(sk.subsurfaceWarmth ?? base.skinIrisDecals.skin.subsurfaceWarmth),
        0,
        1,
      );
      base.skinIrisDecals.skin.poreRoughness = clamp(
        Number(sk.poreRoughness ?? base.skinIrisDecals.skin.poreRoughness),
        0.1,
        0.9,
      );
      base.skinIrisDecals.skin.sheen = clamp(Number(sk.sheen ?? base.skinIrisDecals.skin.sheen), 0, 1);
    }
    const ir = raw.skinIrisDecals.iris;
    if (ir && typeof ir === 'object') {
      base.skinIrisDecals.iris.leftOuterColor = String(ir.leftOuterColor ?? base.skinIrisDecals.iris.leftOuterColor);
      base.skinIrisDecals.iris.leftInnerRingColor = String(
        ir.leftInnerRingColor ?? base.skinIrisDecals.iris.leftInnerRingColor,
      );
      base.skinIrisDecals.iris.heterochromia = Boolean(ir.heterochromia);
      base.skinIrisDecals.iris.rightOuterColor = base.skinIrisDecals.iris.heterochromia
        ? String(ir.rightOuterColor ?? '#5b7a52')
        : base.skinIrisDecals.iris.leftOuterColor;
      base.skinIrisDecals.iris.rightInnerRingColor = base.skinIrisDecals.iris.heterochromia
        ? String(ir.rightInnerRingColor ?? '#d6b265')
        : base.skinIrisDecals.iris.leftInnerRingColor;
      base.skinIrisDecals.iris.pupilDilation = clamp(
        Number(ir.pupilDilation ?? base.skinIrisDecals.iris.pupilDilation),
        0.2,
        0.8,
      );
      base.skinIrisDecals.iris.scleraBrightness = clamp(
        Number(ir.scleraBrightness ?? base.skinIrisDecals.iris.scleraBrightness),
        0.7,
        1.0,
      );
    }
    if (Array.isArray(raw.skinIrisDecals.decals)) {
      base.skinIrisDecals.decals = raw.skinIrisDecals.decals.map((d, idx) => ({
        id: String(d.id ?? `decal_${idx}`),
        kind: ['freckles', 'makeup', 'scar', 'tattoo'].includes(d.kind) ? d.kind : 'freckles',
        region: ['face', 'torso', 'arms', 'back'].includes(d.region) ? d.region : 'face',
        opacity: clamp(Number(d.opacity ?? 0.5), 0, 1),
        tint: String(d.tint ?? '#6e473b'),
        uvScale: clamp(Number(d.uvScale ?? 1.0), 0.2, 4.0),
        enabled: d.enabled !== false,
      }));
    }
  }

  if (raw.physicsGroom && typeof raw.physicsGroom === 'object') {
    const pg = raw.physicsGroom;
    if (pg.frontBangs) {
      base.physicsGroom.frontBangs = {
        styleId: String(pg.frontBangs.styleId ?? base.physicsGroom.frontBangs.styleId),
        length: clamp(Number(pg.frontBangs.length ?? base.physicsGroom.frontBangs.length), 0, 1),
      };
    }
    if (pg.crown) {
      base.physicsGroom.crown = {
        styleId: String(pg.crown.styleId ?? base.physicsGroom.crown.styleId),
        volume: clamp(Number(pg.crown.volume ?? base.physicsGroom.crown.volume), 0, 1),
      };
    }
    if (pg.backHair) {
      base.physicsGroom.backHair = {
        styleId: String(pg.backHair.styleId ?? base.physicsGroom.backHair.styleId),
        length: clamp(Number(pg.backHair.length ?? base.physicsGroom.backHair.length), 0, 1),
      };
    }
    if (pg.rootColor) base.physicsGroom.rootColor = String(pg.rootColor);
    if (pg.tipColor) base.physicsGroom.tipColor = String(pg.tipColor);
    if (pg.highlightColor) base.physicsGroom.highlightColor = String(pg.highlightColor);
    if (pg.anisotropy !== undefined) {
      base.physicsGroom.anisotropy = clamp(Number(pg.anisotropy), 0, 1);
    }
    if (pg.springBone && typeof pg.springBone === 'object') {
      base.physicsGroom.springBone = {
        stiffness: clamp(Number(pg.springBone.stiffness ?? 165), 10, 400),
        damping: clamp(Number(pg.springBone.damping ?? 0.34), 0.05, 0.95),
        gravityScale: clamp(Number(pg.springBone.gravityScale ?? 0.85), 0, 2),
        collisionRadius: clamp(Number(pg.springBone.collisionRadius ?? 0.06), 0.02, 0.18),
      };
    }
  }

  if (raw.wardrobeSlots && typeof raw.wardrobeSlots === 'object') {
    for (const slotKey of Object.keys(base.wardrobeSlots)) {
      const s = raw.wardrobeSlots[slotKey];
      if (s && typeof s === 'object') {
        base.wardrobeSlots[slotKey] = {
          itemId: String(s.itemId ?? base.wardrobeSlots[slotKey].itemId),
          tint: String(s.tint ?? base.wardrobeSlots[slotKey].tint),
          roughness: clamp(Number(s.roughness ?? base.wardrobeSlots[slotKey].roughness), 0.05, 0.98),
          scaleOffsetMm: clamp(Number(s.scaleOffsetMm ?? 0), -10, 25),
        };
      }
    }
  }

  return base;
}

/**
 * Solves Modular Wardrobe anti-clipping constraints (`C_occ`):
 * 1. Computes union `bodyOcclusionMask` across all active clothing layers so covered
 *    anatomical skin submeshes are culled before skinning/rendering.
 * 2. Performs Minimal-Perturbation Layer Clearance Projection (`s* = argmin ||s - s_hat||^2`)
 *    when an inner/top garment thickness exceeds the outer garment's clearance envelope.
 */
export function solveWardrobeOcclusionConstraints(rawDna) {
  const dna = normalizeCharacterDna(rawDna);
  let bodyOcclusionMask = BODY_OCCLUSION_REGIONS.NONE;
  const resolvedSlots = {};
  const projections = [];

  for (const [slotKey, slotVal] of Object.entries(dna.wardrobeSlots)) {
    const catalogList = WARDROBE_SLOT_CATALOG[slotKey] ?? [];
    const meta = catalogList.find((item) => item.id === slotVal.itemId) ?? catalogList[0] ?? {
      id: slotVal.itemId,
      layerThicknessMm: 2.0,
      bodyOcclusionMask: BODY_OCCLUSION_REGIONS.NONE,
    };
    bodyOcclusionMask |= meta.bodyOcclusionMask || 0;
    resolvedSlots[slotKey] = {
      ...slotVal,
      layerThicknessMm: meta.layerThicknessMm ?? 2.0,
      minInnerClearanceMm: meta.minInnerClearanceMm,
      minTopClearanceMm: meta.minTopClearanceMm,
      bodyOcclusionMask: meta.bodyOcclusionMask ?? 0,
    };
  }

  // Layer clearance constraint 1: inner vs top
  const inner = resolvedSlots.inner;
  const top = resolvedSlots.top;
  if (inner && top && Number.isFinite(top.minInnerClearanceMm)) {
    const effectiveInnerMm = inner.layerThicknessMm + inner.scaleOffsetMm;
    const effectiveTopClearanceMm = top.minInnerClearanceMm + top.scaleOffsetMm;
    if (effectiveInnerMm > effectiveTopClearanceMm) {
      const neededDeltaMm = Number((effectiveInnerMm - effectiveTopClearanceMm + 0.4).toFixed(2));
      const proposedOffset = top.scaleOffsetMm;
      top.scaleOffsetMm = Number((top.scaleOffsetMm + neededDeltaMm).toFixed(2));
      dna.wardrobeSlots.top.scaleOffsetMm = top.scaleOffsetMm;
      projections.push({
        slot: 'top',
        reason: 'inner_layer_penetration',
        proposedScaleOffsetMm: proposedOffset,
        projectedScaleOffsetMm: top.scaleOffsetMm,
        perturbationMm: neededDeltaMm,
      });
    }
  }

  // Layer clearance constraint 2: top vs outer
  const outer = resolvedSlots.outer;
  if (top && outer && outer.layerThicknessMm > 0 && Number.isFinite(outer.minTopClearanceMm)) {
    const effectiveTopMm = top.layerThicknessMm + top.scaleOffsetMm;
    const effectiveOuterClearanceMm = outer.minTopClearanceMm + outer.scaleOffsetMm;
    if (effectiveTopMm > effectiveOuterClearanceMm) {
      const neededDeltaMm = Number((effectiveTopMm - effectiveOuterClearanceMm + 0.5).toFixed(2));
      const proposedOffset = outer.scaleOffsetMm;
      outer.scaleOffsetMm = Number((outer.scaleOffsetMm + neededDeltaMm).toFixed(2));
      dna.wardrobeSlots.outer.scaleOffsetMm = outer.scaleOffsetMm;
      projections.push({
        slot: 'outer',
        reason: 'top_layer_penetration',
        proposedScaleOffsetMm: proposedOffset,
        projectedScaleOffsetMm: outer.scaleOffsetMm,
        perturbationMm: neededDeltaMm,
      });
    }
  }

  const hiddenBodyRegions = [];
  for (const [bitStr, regionName] of Object.entries(BODY_OCCLUSION_LABELS)) {
    const bit = Number(bitStr);
    if ((bodyOcclusionMask & bit) !== 0) {
      hiddenBodyRegions.push(regionName);
    }
  }

  return {
    dna,
    bodyOcclusionMask,
    hiddenBodyRegions,
    hiddenSubmeshNames: hiddenBodyRegions.map((r) => `body_${r}`),
    projections,
    verified: projections.length === 0,
  };
}

/**
 * Picks which 3D anatomical region (`brow`, `eyes`, `nose`, `jaw`, `shoulders`, `torso`)
 * corresponds to normalized viewport coordinates `(nx, ny) in [0, 1]^2`.
 */
export function hitTestStageDragRegion(nx, ny, focusMode = 'bust') {
  const x = clamp(nx, 0, 1);
  const y = clamp(ny, 0, 1);
  const candidates = Object.values(STAGE_DRAG_REGIONS).filter(
    (r) => !focusMode || r.focusArea === focusMode || focusMode === 'all',
  );
  for (const region of candidates) {
    const b = region.normalizedBox;
    if (x >= b.xMin && x <= b.xMax && y >= b.yMin && y <= b.yMax) {
      return region.id;
    }
  }
  return focusMode === 'body' ? 'shoulders' : 'nose';
}

/**
 * Inverse Semantic Lifting (`L_dna: R^3 / R^2 -> G_dna`) for Direct 3D Stage Region Dragging.
 * Maps continuous viewport drag deltas `(dxNorm, dyNorm)` on an anatomical region into
 * clamped Parametric Character DNA updates and semantic trait descriptors.
 */
export function liftStageRegionDragToDna(rawDna, regionId, dxNorm, dyNorm) {
  const dna = normalizeCharacterDna(rawDna);
  const region = STAGE_DRAG_REGIONS[regionId] ?? STAGE_DRAG_REGIONS.nose;
  const deltaMap = region.mapDelta(Number(dxNorm) || 0, Number(dyNorm) || 0);
  const updatedTraits = [];

  for (const [dotPath, deltaVal] of Object.entries(deltaMap)) {
    const [groupKey, paramKey] = dotPath.split('.');
    const cat = DNA_SLIDER_CATEGORIES[groupKey];
    if (!cat) continue;
    const paramMeta = cat.parameters.find((p) => p.id === paramKey);
    if (!paramMeta) continue;

    if (groupKey === 'bodyBuild') {
      const prev = dna.bodyBuild[paramKey] ?? paramMeta.defaultValue;
      const next = Number(clamp(prev + deltaVal, paramMeta.min, paramMeta.max).toFixed(3));
      dna.bodyBuild[paramKey] = next;
      updatedTraits.push({ path: ['dna', 'bodyBuild', paramKey], paramId: paramKey, previous: prev, value: next });
    } else if (dna.craniofacial[groupKey]) {
      const prev = dna.craniofacial[groupKey][paramKey] ?? paramMeta.defaultValue;
      const next = Number(clamp(prev + deltaVal, paramMeta.min, paramMeta.max).toFixed(3));
      dna.craniofacial[groupKey][paramKey] = next;
      updatedTraits.push({
        path: ['dna', 'craniofacial', groupKey, paramKey],
        paramId: paramKey,
        previous: prev,
        value: next,
      });
    }
  }

  return {
    dna,
    regionId: region.id,
    regionLabel: region.label,
    updatedTraits,
  };
}

/**
 * Compiles 5-Layer Parametric DNA into deterministic skeleton scale, bone offsets,
 * morph target weights, and wardrobe occlusion masks for 1-call WASM evaluation.
 */
export function compileDnaToRigOverrides(rawDna) {
  const occlusion = solveWardrobeOcclusionConstraints(rawDna);
  const dna = occlusion.dna;
  const morphWeights = {};

  for (const catKey of ['faceStructure', 'eyesBrows', 'nose', 'mouthJaw']) {
    for (const param of DNA_SLIDER_CATEGORIES[catKey].parameters) {
      if (param.morphId) {
        morphWeights[param.morphId] = Number((dna.craniofacial[catKey][param.id] ?? 0).toFixed(3));
      }
    }
  }
  for (const param of DNA_SLIDER_CATEGORIES.bodyBuild.parameters) {
    if (param.morphId) {
      morphWeights[param.morphId] = Number((dna.bodyBuild[param.id] ?? 0).toFixed(3));
    }
  }

  const boneOffsets = {};
  const poisePitchDeg = Number(((dna.bodyBuild.posturePoise ?? 0) * -6.0).toFixed(2));
  if (Math.abs(poisePitchDeg) > 0.05) {
    boneOffsets.spine_01 = {
      eulerDeg: [poisePitchDeg, 0, 0],
      translation: [0, (dna.bodyBuild.torsoLength ?? 0) * 0.04, 0],
    };
  }
  const cranioOffset = dna.craniofacial.faceStructure.cranioWidth ?? 0;
  if (Math.abs(cranioOffset) > 0.05) {
    boneOffsets.head = {
      eulerDeg: [0, 0, 0],
      translation: [0, cranioOffset * 0.015, 0],
    };
  }

  const limbReachFactor = 1 + (dna.bodyBuild.limbLength ?? 0) * 0.12;
  const statureScale = dna.bodyBuild.statureScale ?? 1.0;

  return {
    skeletonScale: Number(statureScale.toFixed(3)),
    effectiveArmReachM: Number((0.56 * statureScale * limbReachFactor).toFixed(3)),
    morphWeights,
    boneOffsets,
    bodyOcclusionMask: occlusion.bodyOcclusionMask,
    hiddenBodyRegions: occlusion.hiddenBodyRegions,
    hiddenSubmeshNames: occlusion.hiddenSubmeshNames,
    wardrobeProjections: occlusion.projections,
    springBone: { ...dna.physicsGroom.springBone },
  };
}

/**
 * Generates a Parametric Character DNA profile from semantic natural-language traits
 * or archetype descriptors (e.g. `{ archetype: 'scholar', height: 'tall', iris: 'heterochromia' }`).
 */
export function generateDnaFromTraits(traits = {}) {
  const dna = createDefaultCharacterDna();
  const text = typeof traits === 'string' ? traits.toLowerCase() : JSON.stringify(traits).toLowerCase();

  if (text.includes('tall') || traits.height === 'tall') {
    dna.bodyBuild.statureScale = 1.08;
    dna.bodyBuild.limbLength = 0.25;
  } else if (text.includes('compact') || text.includes('petite') || traits.height === 'compact') {
    dna.bodyBuild.statureScale = 0.92;
    dna.bodyBuild.limbLength = -0.18;
  }

  if (text.includes('athletic') || text.includes('broad') || traits.build === 'athletic') {
    dna.bodyBuild.shoulderBreadth = 0.42;
    dna.bodyBuild.musculature = 0.48;
  } else if (text.includes('slender') || text.includes('scholar') || traits.build === 'slender') {
    dna.bodyBuild.shoulderBreadth = -0.16;
    dna.bodyBuild.posturePoise = 0.35;
  }

  if (text.includes('sharp') || text.includes('sculpted')) {
    dna.craniofacial.mouthJaw.jawWidth = -0.24;
    dna.craniofacial.mouthJaw.chinProjection = 0.28;
    dna.craniofacial.faceStructure.cheekboneProminence = 0.36;
    dna.craniofacial.nose.bridgeHeight = 0.26;
  }

  if (text.includes('heterochromia') || traits.heterochromia === true) {
    dna.skinIrisDecals.iris.heterochromia = true;
    dna.skinIrisDecals.iris.leftOuterColor = '#325d7a';
    dna.skinIrisDecals.iris.rightOuterColor = '#7a5c32';
  }

  if (text.includes('coat') || text.includes('pavilion') || text.includes('haori')) {
    dna.wardrobeSlots.outer.itemId = 'outer_haori_coat';
    dna.wardrobeSlots.top.itemId = 'top_pleated_tunic';
  } else if (text.includes('vest') || text.includes('no outer')) {
    dna.wardrobeSlots.outer.itemId = 'outer_none';
    dna.wardrobeSlots.top.itemId = 'top_tailored_vest';
  }

  return solveWardrobeOcclusionConstraints(dna).dna;
}

/**
 * Produces a token-compact (`<70 tokens`) semantic summary of the 5-Layer DNA state for LLM context.
 */
export function summarizeCharacterDnaForAi(rawDna) {
  const solved = solveWardrobeOcclusionConstraints(rawDna);
  const d = solved.dna;
  const irisTag = d.skinIrisDecals.iris.heterochromia
    ? `heterochromia(${d.skinIrisDecals.iris.leftOuterColor}/${d.skinIrisDecals.iris.rightOuterColor})`
    : d.skinIrisDecals.iris.leftOuterColor;
  return [
    `DNA[stature=${d.bodyBuild.statureScale.toFixed(2)}`,
    `shoulder=${d.bodyBuild.shoulderBreadth.toFixed(2)}`,
    `jaw=${d.craniofacial.mouthJaw.jawWidth.toFixed(2)}`,
    `noseBridge=${d.craniofacial.nose.bridgeHeight.toFixed(2)}`,
    `skin=${d.skinIrisDecals.skin.undertone}:${d.skinIrisDecals.skin.melanin.toFixed(2)}`,
    `iris=${irisTag}`,
    `groom=${d.physicsGroom.frontBangs.styleId}/${d.physicsGroom.backHair.styleId}(k=${d.physicsGroom.springBone.stiffness})`,
    `wardrobe=${d.wardrobeSlots.top.itemId}+${d.wardrobeSlots.outer.itemId}`,
    `occMask=0x${solved.bodyOcclusionMask.toString(16)}]`,
  ].join(' ');
}

/**
 * Headless controller for in-game Player Avatar Creation or Studio DNA manipulation.
 */
export function createAvatarDnaController(initialDna) {
  let dna = normalizeCharacterDna(initialDna);
  const listeners = new Set();

  const notify = () => {
    const compiled = compileDnaToRigOverrides(dna);
    for (const fn of listeners) fn({ dna, compiled });
  };

  return {
    getDna: () => structuredClone(dna),
    setDna(nextDna) {
      dna = solveWardrobeOcclusionConstraints(nextDna).dna;
      notify();
      return dna;
    },
    setTrait(categoryKey, paramId, value) {
      if (categoryKey === 'bodyBuild') {
        dna.bodyBuild[paramId] = Number(value);
      } else if (dna.craniofacial[categoryKey]) {
        dna.craniofacial[categoryKey][paramId] = Number(value);
      }
      dna = normalizeCharacterDna(dna);
      notify();
      return dna;
    },
    dragStageRegion(regionId, dxNorm, dyNorm) {
      const lifted = liftStageRegionDragToDna(dna, regionId, dxNorm, dyNorm);
      dna = lifted.dna;
      notify();
      return lifted;
    },
    setWardrobeSlot(slotKey, patch) {
      if (dna.wardrobeSlots[slotKey]) {
        dna.wardrobeSlots[slotKey] = { ...dna.wardrobeSlots[slotKey], ...patch };
      }
      const solved = solveWardrobeOcclusionConstraints(dna);
      dna = solved.dna;
      notify();
      return solved;
    },
    applySemanticTraits(traits) {
      dna = generateDnaFromTraits(traits);
      notify();
      return dna;
    },
    compileRigOverrides: () => compileDnaToRigOverrides(dna),
    summarizeForAi: () => summarizeCharacterDnaForAi(dna),
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}
