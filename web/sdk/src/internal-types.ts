import type { GanbateCapturedFrame, GanbatePassTiming, GanbateXrTarget } from './types.ts';
import type { WebGpuAdapter, WebGpuCanvasContext, WebGpuDevice, WebGpuTexture } from './webgpu-globals.ts';

export interface PassTimingMark {
  label: string;
  sent: number;
  done: number;
  sync?: WebGLSync | null;
}

export interface PassTimingBatch {
  readonly marks: PassTimingMark[];
  finish(): void;
}

export interface PassTimingTracker {
  on(): boolean;
  set(value: boolean): void;
  busy(): boolean;
  mark(label: string): PassTimingMark;
  take(): PassTimingBatch | null;
  report(): GanbatePassTiming;
}

export interface WebGpuState {
  readonly api: 'webgpu';
  readonly adapter: WebGpuAdapter;
  readonly device: WebGpuDevice;
  readonly adapterName: string;
  readonly features: readonly string[];
  readonly canvas: HTMLCanvasElement | undefined;
  readonly context: WebGpuCanvasContext | null;
  readonly format: string | null;
}

export interface WebGlExtensions {
  readonly colorFloat: unknown;
  readonly floatLinear: unknown;
  readonly floatBlend: unknown;
  readonly aniso: { readonly MAX_TEXTURE_MAX_ANISOTROPY_EXT: number; readonly TEXTURE_MAX_ANISOTROPY_EXT: number } | null;
  readonly provoking: {
    readonly FIRST_VERTEX_CONVENTION_WEBGL: number;
    provokingVertexWEBGL(mode: number): void;
  } | null;
  readonly indexed: {
    enableiOES(target: number, index: number): void;
    disableiOES(target: number, index: number): void;
    blendFuncSeparateiOES(
      buf: number,
      srcRGB: number,
      dstRGB: number,
      srcAlpha: number,
      dstAlpha: number,
    ): void;
    colorMaskiOES(buf: number, r: boolean, g: boolean, b: boolean, a: boolean): void;
  } | null;
  readonly s3tc: unknown;
  readonly s3tcSrgb: unknown;
  readonly bptc: unknown;
  readonly astc: unknown;
}

export interface WebGlState {
  readonly api: 'webgl2';
  readonly gl: WebGL2RenderingContext;
  readonly ext: WebGlExtensions;
  readonly canvas: HTMLCanvasElement | undefined;
  readonly renderer: string;
  readonly adapterName: string;
  readonly features: readonly string[];
  readonly format: 'rgba8unorm';
}

export type AnyGpuState = WebGpuState | WebGlState;

export interface GanbateHostRuntime {
  readonly imports: WebAssembly.Imports;
  readonly errors: string[];
  readonly timing: PassTimingTracker;
  attach(instance: WebAssembly.Instance): void;
  setSource(image: unknown): void;
  nextFrame(): void;
  frameTexture(): unknown;
  present?(frame: WebGpuTexture): void;
  release?(): void;
  bundleStats(): { cached: number; candidates: number; replays: number };
  endFrame(): void;
  setXrTarget(target: GanbateXrTarget | null): boolean;
  readSurface?(): GanbateCapturedFrame | null;
}

export interface GanbateWasmExports {
  readonly memory: WebAssembly.Memory;
  ganbate_alloc(len: number): number;
  ganbate_free(ptr: number, len: number): void;
  ganbate_selftest(): number;
  ganbate_load_world(ptr: number, len: number): number;
  ganbate_world_images(ptr: number, cap: number): number;
  ganbate_world_image(tex: number, ptr: number, cap: number): number;
  ganbate_world_set_image(tex: number, width: number, height: number, ptr: number, len: number): void;
  ganbate_world_entities(ptr: number, cap: number): number;
  ganbate_start(lookPtr: number, lookLen: number, queryPtr: number, queryLen: number): number;
  ganbate_key(code: number, down: number): void;
  ganbate_blur(): void;
  ganbate_look(dx: number, dy: number): void;
  ganbate_move(fwd: number, right: number): void;
  ganbate_frame(dt: number): number;
  ganbate_frames(): number;

  ganbate_graph_commit(ap: number, al: number, bp: number, bl: number, mode: number): number;
  ganbate_graph_check(ap: number, al: number, bp: number, bl: number, mode: number): number;
  ganbate_graph_result(out: number, len: number): void;

  ganbate_rt_start(a: number, b: number, lookPtr: number, lookLen: number, quality: number): number;
  ganbate_rt_stop(): void;
  ganbate_rt_set_grade_once?(on: number): number;
  ganbate_rt_ao_import(ptr: number, len: number): number;
  ganbate_rt_size(out: number): number;
  ganbate_rt_set_look(ptr: number, len: number): number;
  ganbate_rt_set_view(ptr: number): number;
  ganbate_rt_set_quality?(code: number): number;
  ganbate_rt_frame_stats?(out: number): number;
  ganbate_rt_cache_stats?(out: number): number;
  ganbate_rt_set_stereo_eye?(code: number, ipd: number): number;
  ganbate_rt_stereo_views?(out: number): number;
  ganbate_rt_governor_record?(dtMs: number, out: number): number;
  ganbate_rt_governor_query?(out: number): number;
  ganbate_rt_governor_reset?(): number;
  ganbate_rt_set_clock(hours: number, seconds: number): number;
  ganbate_rt_set_lamp_boost(boost: number): number;
  ganbate_rt_set_weather(ptr: number, len: number): number;
  ganbate_rt_render(): number;
  ganbate_rt_frames(): number;
  ganbate_rt_step_ao?(samples: number): number;
  ganbate_rt_finish_ao?(): number;
  ganbate_rt_ao_progress?(): number;
  ganbate_rt_set_split_passes?(on: number): void;

  ganbate_rt_texture_create_clear(
    width: number,
    height: number,
    srgb: number,
    r: number,
    g: number,
    b: number,
    a: number,
  ): number;
  ganbate_rt_texture_create_source?(width: number, height: number, srgb: number): number;
  ganbate_rt_texture_create(width: number, height: number, ptr: number, len: number, srgb: number): number;
  ganbate_rt_texture_update_source?(tex: number, width: number, height: number): number;
  ganbate_rt_texture_update(tex: number, ptr: number, len: number): number;
  ganbate_rt_texture_destroy(tex: number): void;
  ganbate_rt_texture_png(ptr: number, len: number, srgb: number): number;

  ganbate_rt_mesh_create(
    posPtr: number,
    verts: number,
    normPtr: number,
    uvPtr: number,
    idxPtr: number,
    idxLen: number,
  ): number;
  ganbate_rt_mesh_update(mesh: number, posPtr: number, normPtr: number, verts: number): number;
  ganbate_rt_mesh_destroy(mesh: number): void;

  ganbate_rt_material_create(ptr: number, len: number): number;
  ganbate_rt_material_update(mat: number, ptr: number, len: number): number;
  ganbate_rt_material_destroy(mat: number): void;
  ganbate_rt_shader_create(ptr: number, len: number): number;
  ganbate_rt_material_params(mat: number, ptr: number): number;

  ganbate_rt_instance_create(mesh: number, mat: number, matPtr: number): number;
  ganbate_rt_instance_set(inst: number, matPtr: number, visible: number): number;
  ganbate_rt_instances_set(ptr: number, count: number): number;
  ganbate_rt_instance_material(inst: number, mat: number): number;
  ganbate_rt_instance_tint(inst: number, r: number, g: number, b: number, a: number): number;
  ganbate_rt_instance_pick(inst: number, layers: number): number;
  ganbate_rt_instance_destroy(inst: number): void;

  ganbate_rt_light_create(ptr: number): number;
  ganbate_rt_light_set(id: number, ptr: number): number;
  ganbate_rt_light_destroy(id: number): void;

  ganbate_rt_project(x: number, y: number, z: number, out: number): number;
  ganbate_rt_pick(x: number, y: number, layers: number, out: number): number;
  ganbate_rt_pick_ray?(
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    layers: number,
    out: number,
  ): number;

  ganbate_rt_world_instances(): number;
  ganbate_rt_world_instance(index: number, out: number): number;
  ganbate_rt_set_world_instance(index: number, matPtr: number, visible: number): number;
  ganbate_rt_set_world_light(index: number, x: number, y: number, z: number): number;

  ganbate_rt_physics_start(): number;
  ganbate_rt_physics_step(dt: number, steps: number): number;
  ganbate_rt_physics_stop(): void;
  ganbate_rt_physics_error(ptr: number, cap: number): number;
  ganbate_rt_physics_hash(out: number): number;

  ganbate_rt_pick_world(x: number, y: number, out: number): number;
  ganbate_rt_pick_world_ray?(
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    out: number,
  ): number;

  ganbate_rt_model_parse(ptr: number, len: number): number;
  ganbate_rt_model_describe(parsed: number, out: number, cap: number): number;
  ganbate_rt_model_image(parsed: number, index: number, out: number, cap: number): number;
  ganbate_rt_model_create(parsed: number, texPtr: number, texLen: number): number;
  ganbate_rt_model_destroy(model: number): void;
  ganbate_rt_model_material(model: number, index: number): number;
  ganbate_rt_model_instance(model: number, matPtr: number): number;
  ganbate_rt_model_instance_set(inst: number, matPtr: number, visible: number): number;
  ganbate_rt_model_instance_destroy(inst: number): void;
  ganbate_rt_model_rest(inst: number): number;
  ganbate_rt_model_snapshot_pose(inst: number): number;
  ganbate_rt_pose_model_clip(inst: number, ptr: number, len: number): number;
  ganbate_rt_model_clip(inst: number, clip: number, time: number, weight: number): number;
  ganbate_rt_model_node_get(inst: number, node: number, out: number): number;
  ganbate_rt_model_node_set(inst: number, node: number, ptr: number): number;
  ganbate_rt_model_morph(inst: number, node: number, target: number, weight: number): number;
  ganbate_rt_model_update(inst: number): number;
  ganbate_rt_model_node_world(inst: number, node: number, out: number): number;
  ganbate_rt_model_node_visible(inst: number, node: number, visible: number): number;
  ganbate_rt_model_tint(inst: number, r: number, g: number, b: number, a: number): number;
  ganbate_rt_model_owner(propInst: number, out: number): number;
}
