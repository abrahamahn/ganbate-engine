/** Minimal Ganbate model interface. The native engine owns the model, its render data and its pose. */
export type Vec3 = readonly [number, number, number];
export type Quat = readonly [number, number, number, number];

export interface CharacterModelInfo {
  readonly nodes: readonly {
    readonly name: string;
    readonly parent?: number | null;
    readonly mesh: number | null;
    readonly skin?: number | null;
  }[];
  readonly meshes: readonly {
    readonly name?: string;
    readonly targetNames: readonly string[];
    readonly targets: number;
  }[];
  readonly clips: readonly {
    readonly name: string;
    readonly duration: number;
    readonly nodes?: readonly number[];
  }[];
}

export interface ClipLayerSpec {
  readonly clip: number;
  readonly time: number;
  readonly weight?: number;
}

export interface CrossfadeSpec {
  readonly elapsed: number;
  readonly duration: number;
  readonly mask?: readonly number[];
}

export interface LookAtChainSpec {
  readonly nodes: readonly (readonly [number, number])[];
  readonly targetModel: Vec3;
  readonly originModel?: Vec3;
  readonly forward?: Vec3;
  readonly up?: Vec3;
  readonly maxYaw?: number;
  readonly maxPitch?: number;
  readonly weight?: number;
}

export interface TwoBoneIkSpec {
  readonly root: number;
  readonly mid: number;
  readonly end: number;
  readonly targetModel: Vec3;
  readonly poleModel: Vec3;
  readonly weight?: number;
}

export interface BoneOffsetSpec {
  readonly node: number;
  readonly translation?: Vec3;
  readonly rotation?: Quat;
  readonly euler?: Vec3;
  readonly scale?: Vec3;
}

export interface MorphOverrideSpec {
  readonly node: number;
  readonly target: number;
  readonly weight: number;
}

export interface PoseModelClipSpec {
  readonly resetToRest?: boolean;
  readonly commitUpdate?: boolean;
  readonly clips?: readonly ClipLayerSpec[];
  readonly crossfade?: CrossfadeSpec | null;
  readonly lookAt?: readonly LookAtChainSpec[];
  readonly twoBoneIk?: readonly TwoBoneIkSpec[];
  readonly boneOffsets?: readonly BoneOffsetSpec[];
  readonly morphOverrides?: readonly MorphOverrideSpec[];
}

export interface GanbateCharacterEngine {
  modelRest?(instance: number): void;
  modelClip(instance: number, clip: number, time: number, weight?: number): void;
  modelNode(instance: number, node: number): { translation: number[]; rotation: number[]; scale: number[] };
  setModelNode(
    instance: number,
    node: number,
    transform: { translation: readonly number[]; rotation: readonly number[]; scale: readonly number[] },
  ): void;
  modelMorph(instance: number, node: number, target: number, weight: number): void;
  updateModel(instance: number): void;
  snapshotModelPose?(instance: number): void;
  poseModelClip?(instance: number, packet: Float32Array): void;
}

export function bindCharacterShapes<K extends string>(
  engine: Pick<GanbateCharacterEngine, 'modelMorph' | 'updateModel'>,
  instance: number,
  info: {
    readonly nodes: readonly { readonly mesh: number | null }[];
    readonly meshes: readonly { readonly targetNames: readonly string[]; readonly targets: number }[];
  },
  ranges: Readonly<Record<K, readonly [number, number]>>,
): {
  readonly available: readonly K[];
  /** Returns true if new weights were committed. Force reapplies weights after external animation. */
  apply(weights: Readonly<Partial<Record<K, number>>>, force?: boolean): boolean;
};

export interface BlendSample1D {
  readonly clip: number;
  readonly position: number;
  readonly rate?: number;
}

export interface BlendSample2D {
  readonly clip: number;
  readonly position: readonly [number, number];
  readonly rate?: number;
}

export function hermiteFade(elapsedSeconds: number, durationSeconds: number): number;
export function weightsToClipLayers(
  entries: readonly { readonly clip: number; readonly time?: number; readonly weight: number }[],
): ClipLayerSpec[];
export function sampleBlendSpace1D(
  samples: readonly BlendSample1D[],
  param: number,
  baseTime?: number,
): ClipLayerSpec[];
export function sampleBlendSpace2D(
  samples: readonly BlendSample2D[],
  param: readonly [number, number],
  baseTime?: number,
): ClipLayerSpec[];
export function normalizeQuat(q: Quat): [number, number, number, number];
export function mulQuat(a: Quat, b: Quat): [number, number, number, number];
export function slerpQuat(a: Quat, b: Quat, t: number): [number, number, number, number];
export function quatFromAxisAngle(axis: Vec3, angle: number): [number, number, number, number];
export function quatFromEuler(pitch: number, yaw: number, roll: number): [number, number, number, number];
export function rotateVec3(q: Quat, v: Vec3): [number, number, number];
export function encodePoseCommandPacket(spec?: PoseModelClipSpec): Float32Array;
export function poseModelClip(
  engine: GanbateCharacterEngine,
  instance: number,
  spec?: PoseModelClipSpec,
  options?: {
    readonly fromPose?: ReadonlyMap<
      number,
      { readonly translation: Vec3; readonly rotation: Quat; readonly scale: Vec3 }
    > | null;
  },
): Float32Array;

export interface CharacterRigFrame {
  readonly clockSeconds?: number;
  readonly stateKey?: string;
  readonly clip?: string | number | null;
  readonly time?: number;
  readonly weight?: number;
  readonly secondaryClip?: string | number | null;
  readonly secondaryTime?: number;
  readonly blendWeight?: number;
  readonly crossfadeSeconds?: number;
  readonly resetToRest?: boolean;
  readonly commitUpdate?: boolean;
  readonly lookAt?: Omit<LookAtChainSpec, 'nodes'> & {
    readonly joints: readonly (readonly [string | number, number])[];
  };
  readonly twoBoneIk?: readonly TwoBoneIkSpec[];
  readonly bonePoses?: Readonly<Record<string, Vec3 | Omit<BoneOffsetSpec, 'node'>>>;
  readonly morphWeights?: Readonly<Record<string, number>>;
}

export function createCharacterRigController(
  engine: GanbateCharacterEngine,
  instance: number,
  info: CharacterModelInfo,
  config?: { readonly defaultCrossfadeSeconds?: number },
): {
  readonly clipByName: ReadonlyMap<string, CharacterModelInfo['clips'][number] & { readonly index: number }>;
  readonly nodeByName: ReadonlyMap<string, number>;
  readonly morphNames: readonly string[];
  seek(): void;
  evaluate(frame: CharacterRigFrame): Float32Array;
};
