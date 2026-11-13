import { graphCalls, scratchFor } from './graph-solver.ts';
import { createFrameGate } from './pass-timing.ts';
import {
  copiesDirectly,
  pixelsOf,
  qualityCodeFromMode,
  resolveGradeOnce,
  resolveQualityCode,
  resolveStereoEyeCode,
  sourceSize,
} from './texture-source.ts';
import type {
  CreateEngineOptions,
  GanbateClearTexture,
  GanbateEngine,
  GanbateEntityTable,
  GanbateGovernorSnapshot,
  GanbateImageSource,
  GanbateLight,
  GanbateMaterialDesc,
  GanbateMeshData,
  GanbateModelInfo,
  GanbateNodeTransform,
  GanbateQualityMode,
  GanbateStereoEye,
  GanbateStereoEyeName,
  GanbateView,
  GanbateXrTarget,
  Matrix4,
  Rgba,
  Vec3,
} from './types.ts';
import { boot, decodeWorldImages, fetchInto, readPixels } from './wasm-boot.ts';

function isClearTexture(
  source: GanbateImageSource | GanbateClearTexture,
): source is GanbateClearTexture {
  return (
    Boolean(source) &&
    typeof source === 'object' &&
    !('data' in source && source.data) &&
    'clear' in source &&
    Boolean((source as GanbateClearTexture).clear)
  );
}

/**
 * Library mode: the engine as a renderer that a host page drives. Resolves with a `GanbateEngine`
 * instance; its methods throw an `Error` carrying the engine's message when a call fails.
 */
export async function createEngine(options: CreateEngineOptions): Promise<GanbateEngine> {
  const { canvas } = options;
  let lastError = '';
  const onLog = options.onLog || (() => {});
  let trying = false;
  const { gpu, host, x, errors, settled } = await boot({
    ...options,
    onLog: (level, text) => {
      if (trying && level >= 2 && !text.startsWith('panic')) {
        onLog(1, `${text}; sending the pixels instead`);
        return;
      }
      if (level >= 2) lastError = text;
      onLog(level, text);
    },
    onError: (text) => {
      lastError = text;
      onLog(2, text);
    },
  });
  const fail = (what: string) => new Error(`${what}: ${lastError || 'the engine refused'}`);
  const ok = <T extends number>(what: string, r: T): T => {
    if (!r) throw fail(what);
    return r;
  };

  const { reserve, place, free: freeScratch } = scratchFor(x);
  const encoder = new TextEncoder();
  const fromSource = (source: GanbateImageSource, call: (width: number, height: number) => number): number => {
    if (
      typeof x.ganbate_rt_texture_create_source !== 'function' ||
      !copiesDirectly(source, gpu.api !== 'webgl2')
    ) {
      return 0;
    }
    const [width, height] = sourceSize(source);
    if (!width || !height) return 0;
    host.setSource(source);
    trying = true;
    try {
      return call(width, height);
    } catch (e) {
      if (e instanceof WebAssembly.RuntimeError) throw e;
      onLog(1, `texture from source: ${e instanceof Error ? e.message : String(e)}; sending the pixels instead`);
      return 0;
    } finally {
      trying = false;
      host.setSource(null);
    }
  };
  const text = (s: string) => encoder.encode(s);
  const f32 = (a: ArrayLike<number>) => (a instanceof Float32Array ? a : new Float32Array(a));
  const lightFloats = (light: GanbateLight) => {
    const v = new Float32Array(15);
    v.set(light.position, 0);
    v[3] = light.range;
    v.set(light.color, 4);
    v[7] = light.intensity;
    if (light.direction !== undefined) {
      v.set(light.direction, 8);
      v[11] = light.inner ?? 0;
      v[12] = light.outer ?? light.inner ?? 0;
    }
    v[13] = light.followsLamps === true ? 1 : 0;
    v[14] = light.shadow ?? 0;
    return v;
  };
  const out = (n: number) => new Float32Array(x.memory.buffer, reserve(n * 4), n);
  const readBudget = (p: number): GanbateGovernorSnapshot => {
    const v = new Float32Array(x.memory.buffer, p, 6);
    return {
      resolutionScale: v[0],
      maxDynamicLights: v[1] >>> 0,
      ambientParticles: v[2] > 0.5,
      shadowTier: v[3] >>> 0,
      tier: v[4] >>> 0,
      rollingMeanMs: v[5],
    };
  };

  const bakedAo: Promise<ArrayBuffer | Uint8Array | null> =
    typeof options.ao === 'string'
      ? fetch(options.ao).then(
          (r) => (r.ok ? r.arrayBuffer() : null),
          () => null,
        )
      : Promise.resolve(options.ao ?? null);

  let world: { ptr: number; len: number } | null = null;
  if (typeof options.world === 'string') {
    world = await fetchInto(options.world, x, (p, bytes) =>
      (options.onPhase || (() => {}))(`Downloading the world (${(bytes / 1048576).toFixed(1)} MB)`, p),
    );
  } else if (options.world) {
    const bytes = options.world instanceof Uint8Array ? options.world : new Uint8Array(options.world);
    const ptr = x.ganbate_alloc(bytes.length) >>> 0;
    new Uint8Array(x.memory.buffer, ptr, bytes.length).set(bytes);
    world = { ptr, len: bytes.length };
  }
  if (world) {
    (options.onPhase || (() => {}))('Decoding the world', 1);
    if (!x.ganbate_load_world(world.ptr, world.len)) throw fail('world');
    await decodeWorldImages(x, onLog);
  }
  const lookText =
    options.look == null
      ? ''
      : typeof options.look === 'string'
        ? options.look
        : JSON.stringify(options.look);
  const lookBytes = text(lookText);
  const [lookPtr] = place(lookBytes);
  (options.onPhase || (() => {}))('Compiling shaders and uploading', 1);
  ok(
    'start',
    x.ganbate_rt_start(0, 0, lookPtr, lookBytes.length, resolveQualityCode(options.quality, options.still)),
  );
  const gradeOnce =
    resolveGradeOnce(options.gradeOnce, gpu) &&
    typeof x.ganbate_rt_set_grade_once === 'function' &&
    x.ganbate_rt_set_grade_once(1) === 1;
  const ao = world ? await bakedAo : null;
  if (ao) {
    const bytes = ao instanceof Uint8Array ? ao : new Uint8Array(ao);
    const [p] = place(bytes);
    if (!x.ganbate_rt_ao_import(p, bytes.length)) {
      onLog(1, 'The baked AO does not fit this world; baking it live');
    }
  }

  const gate = createFrameGate(gpu, host, settled);
  let stopped = false;
  let drawn: { width: number; height: number } | null = null;
  const drawnSize = () => {
    if (!drawn) {
      const p = reserve(8);
      ok('size', x.ganbate_rt_size(p));
      const s = new Uint32Array(x.memory.buffer, p, 2);
      drawn = { width: s[0], height: s[1] };
    }
    return drawn;
  };
  const size = () => ({ ...drawnSize() });

  let cssBox: { width: number; height: number } | null = null;
  let observer: ResizeObserver | null = null;
  const cssSize = () => cssBox || { width: canvas.clientWidth, height: canvas.clientHeight };
  const observedBox = (entry: ResizeObserverEntry): [number, number] => {
    const rawBox = entry.contentBoxSize as unknown as
      | readonly ResizeObserverSize[]
      | ResizeObserverSize
      | undefined;
    const b = Array.isArray(rawBox) ? rawBox[0] : (rawBox as ResizeObserverSize | undefined);
    if (!b) return [entry.contentRect.width, entry.contentRect.height];
    const vertical = /^(vertical|sideways)/.test(getComputedStyle(canvas).writingMode);
    return vertical ? [b.blockSize, b.inlineSize] : [b.inlineSize, b.blockSize];
  };
  if (typeof ResizeObserver === 'function') {
    try {
      observer = new ResizeObserver((entries) => {
        const [w, h] = observedBox(entries[entries.length - 1]);
        cssBox = { width: Math.round(w), height: Math.round(h) };
      });
      observer.observe(canvas);
      cssBox = { width: canvas.clientWidth, height: canvas.clientHeight };
    } catch {
      observer = null;
    }
  }
  const cssScale = () => {
    const s = drawnSize();
    const css = cssSize();
    return { sx: (css.width || s.width) / s.width, sy: (css.height || s.height) / s.height };
  };

  const engine: GanbateEngine = {
    api: gpu.api,
    present: settled ? 'settled' : 'direct',
    adapter: gpu.adapterName,
    gradeOnce,
    errors: () => [...errors, ...host.errors.filter((e) => !errors.includes(e))],
    fitCanvas(
      dpr = (typeof window !== 'undefined' ? window.devicePixelRatio : 1) || 1,
      box: { readonly width: number; readonly height: number } | null = null,
    ) {
      const w = Math.max(1, Math.round((box ? box.width : canvas.clientWidth) * dpr));
      const h = Math.max(1, Math.round((box ? box.height : canvas.clientHeight) * dpr));
      if (canvas.width === w && canvas.height === h) return;
      canvas.width = w;
      canvas.height = h;
      gate.reshow();
    },
    size,
    setLook(look: object | string) {
      const t = text(typeof look === 'string' ? look : JSON.stringify(look));
      const [p] = place(t);
      ok('setLook', x.ganbate_rt_set_look(p, t.length));
    },
    setView(v: GanbateView) {
      const up = v.up || [0, 1, 0];
      const shift = v.shift || [0, 0];
      const f = v.focus;
      const p = reserve(68);
      const dst = new Float32Array(x.memory.buffer, p, 17);
      dst[0] = v.eye[0];
      dst[1] = v.eye[1];
      dst[2] = v.eye[2];
      dst[3] = v.target[0];
      dst[4] = v.target[1];
      dst[5] = v.target[2];
      dst[6] = up[0];
      dst[7] = up[1];
      dst[8] = up[2];
      dst[9] = v.fov ?? 50;
      dst[10] = shift[0];
      dst[11] = shift[1];
      dst[12] = v.near ?? 0.08;
      dst[13] = f ? f.distance : 0;
      dst[14] = f ? (f.depth ?? 4) : 0;
      dst[15] = f ? (f.blur ?? 0.006) : 0;
      dst[16] = f ? 1 : 0;
      ok('setView', x.ganbate_rt_set_view(p));
    },
    setQuality(mode: GanbateQualityMode) {
      if (typeof x.ganbate_rt_set_quality !== 'function') return false;
      const code = qualityCodeFromMode(mode);
      if (code === null) return false;
      return Boolean(x.ganbate_rt_set_quality(code));
    },
    frameStats() {
      if (typeof x.ganbate_rt_frame_stats !== 'function') return null;
      const p = reserve(36);
      if (!x.ganbate_rt_frame_stats(p)) return null;
      const w = new Uint32Array(x.memory.buffer, p, 9);
      return {
        worldDrawsTotal: w[0],
        worldDrawsSubmitted: w[1],
        worldDrawsCulled: w[2],
        shadowStaticCached: w[3] !== 0,
        shadowWorldDraws: w[4],
        groundCached: w[5] !== 0,
        reflectionDrawn: w[6] !== 0,
        ssaoDrawn: w[7] !== 0,
        fusedTonemap: w[8] !== 0,
      };
    },
    cacheStats() {
      if (typeof x.ganbate_rt_cache_stats !== 'function') return null;
      const p = reserve(40);
      if (!x.ganbate_rt_cache_stats(p)) return null;
      const w = new Uint32Array(x.memory.buffer, p, 10);
      return {
        worldCached: w[0] !== 0,
        regions: w[1],
        reshaded: w[2] !== 0,
        waterKept: w[3] !== 0,
        waterStaticsWhole: w[4] !== 0,
        waterStaticsRegions: w[5],
        waterFootprints: w[6],
        reshadePixels: w[7],
        aoKept: w[8] !== 0,
        aoChangedPixels: w[9],
      };
    },
    setStereoEye(eye: GanbateStereoEyeName | GanbateStereoEye = 'mono', ipdMetres = 0.063) {
      if (typeof x.ganbate_rt_set_stereo_eye !== 'function') return false;
      const code = resolveStereoEyeCode(eye);
      if (code === null || !Number.isFinite(ipdMetres)) return false;
      return Boolean(x.ganbate_rt_set_stereo_eye(code, ipdMetres));
    },
    stereoViews() {
      if (typeof x.ganbate_rt_stereo_views !== 'function') return null;
      const p = out(16).byteOffset;
      if (!x.ganbate_rt_stereo_views(p)) return null;
      const v = Array.from(new Float32Array(x.memory.buffer, p, 16));
      return {
        left: { eye: [v[0], v[1], v[2]], target: [v[3], v[4], v[5]] },
        right: { eye: [v[6], v[7], v[8]], target: [v[9], v[10], v[11]] },
        rightAxis: [v[12], v[13], v[14]],
        ipdMetres: v[15],
      };
    },
    glContext() {
      return gpu.api === 'webgl2' ? gpu.gl : null;
    },
    setXrTarget(target: GanbateXrTarget | null = null) {
      return Boolean(host.setXrTarget?.(target));
    },
    governorRecord(dtMs: number) {
      if (typeof x.ganbate_rt_governor_record !== 'function') return null;
      const p = out(6).byteOffset;
      if (!x.ganbate_rt_governor_record(dtMs, p)) return null;
      return readBudget(p);
    },
    governorQuery() {
      if (typeof x.ganbate_rt_governor_query !== 'function') return null;
      const p = out(6).byteOffset;
      if (!x.ganbate_rt_governor_query(p)) return null;
      return readBudget(p);
    },
    governorReset() {
      if (typeof x.ganbate_rt_governor_reset !== 'function') return false;
      return Boolean(x.ganbate_rt_governor_reset());
    },
    recordFrameBudget(dtMs: number) {
      return engine.governorRecord(dtMs);
    },
    frameBudget() {
      return engine.governorQuery();
    },
    resetFrameGovernor() {
      return engine.governorReset();
    },
    setClock(hours: number, seconds = 0) {
      ok('setClock', x.ganbate_rt_set_clock(hours, seconds));
    },
    setLampBoost(boost: number) {
      ok('setLampBoost', x.ganbate_rt_set_lamp_boost(boost));
    },
    setWeather(preset: string) {
      const t = text(preset);
      const [p] = place(t);
      ok('setWeather', x.ganbate_rt_set_weather(p, t.length));
    },
    render() {
      if (stopped) return false;
      gate.show();
      if (host.timing.busy() || gate.full()) return false;
      host.nextFrame();
      drawn = null;
      const r = x.ganbate_rt_render();
      if (!r) throw fail('render');
      if (r !== 1) return false;
      host.endFrame();
      gate.submitted();
      return true;
    },
    frames: () => x.ganbate_rt_frames(),
    stepAo: (samples: number) =>
      typeof x.ganbate_rt_step_ao === 'function' ? x.ganbate_rt_step_ao(Math.max(0, samples | 0)) === 1 : false,
    finishAo: () => {
      if (typeof x.ganbate_rt_finish_ao === 'function') ok('finishAo', x.ganbate_rt_finish_ao());
    },
    aoProgress: () =>
      typeof x.ganbate_rt_ao_progress === 'function' ? x.ganbate_rt_ao_progress() / 1000 : 1,
    bundleStats: () => (host.bundleStats ? host.bundleStats() : { cached: 0, candidates: 0, replays: 0 }),
    async capture() {
      const texture = host.frameTexture();
      if (!texture) throw new Error('capture: no frame was rendered');
      return readPixels(gpu, texture, host);
    },
    async idle() {
      if (gpu.api === 'webgpu') await gpu.device.queue.onSubmittedWorkDone();
    },
    setPassTiming(on: boolean) {
      host.timing.set(on);
      if (typeof x.ganbate_rt_set_split_passes === 'function') x.ganbate_rt_set_split_passes(on ? 1 : 0);
    },
    passTiming: () => host.timing.report(),

    createTexture(source: GanbateImageSource | GanbateClearTexture, { srgb = true }: { readonly srgb?: boolean } = {}) {
      const s = srgb ? 1 : 0;
      if (isClearTexture(source)) {
        const { width, height, clear } = source;
        if (
          !(Number.isInteger(width) && Number.isInteger(height) && width > 0 && height > 0) ||
          clear.length !== 4 ||
          !Array.prototype.every.call(clear, Number.isFinite)
        ) {
          throw new Error(
            'createTexture: a cleared texture takes a width and height in whole pixels and clear: [r, g, b, a]',
          );
        }
        return ok(
          'createTexture',
          x.ganbate_rt_texture_create_clear(width, height, s, clear[0], clear[1], clear[2], clear[3]),
        );
      }
      const direct = fromSource(source, (w, h) => x.ganbate_rt_texture_create_source!(w, h, s));
      if (direct) return direct;
      const { width, height, data } = pixelsOf(source);
      const [p] = place(data);
      return ok('createTexture', x.ganbate_rt_texture_create(width, height, p, data.byteLength, s));
    },
    updateTexture(texture: number, source: GanbateImageSource) {
      if (
        typeof x.ganbate_rt_texture_update_source === 'function' &&
        fromSource(source, (w, h) => x.ganbate_rt_texture_update_source!(texture, w, h))
      ) {
        return;
      }
      const { data } = pixelsOf(source);
      const [p] = place(data);
      ok('updateTexture', x.ganbate_rt_texture_update(texture, p, data.byteLength));
    },
    destroyTexture(texture: number) {
      x.ganbate_rt_texture_destroy(texture);
    },
    createMesh(m: GanbateMeshData) {
      const positions = f32(m.positions);
      const normals = m.normals && m.normals.length ? f32(m.normals) : null;
      const uvs = m.uvs && m.uvs.length ? f32(m.uvs) : null;
      const indices = m.indices instanceof Uint32Array ? m.indices : new Uint32Array(m.indices);
      const [p, n, u, i] = place(positions, normals, uvs, indices);
      return ok('createMesh', x.ganbate_rt_mesh_create(p, positions.length / 3, n, u, i, indices.length));
    },
    updateMesh(mesh: number, m: { readonly positions: ArrayLike<number>; readonly normals?: ArrayLike<number> }) {
      const positions = f32(m.positions);
      const normals = m.normals && m.normals.length ? f32(m.normals) : null;
      const [p, n] = place(positions, normals);
      ok('updateMesh', x.ganbate_rt_mesh_update(mesh, p, n, positions.length / 3));
    },
    destroyMesh(mesh: number) {
      x.ganbate_rt_mesh_destroy(mesh);
    },
    createMaterial(desc: GanbateMaterialDesc) {
      const t = text(JSON.stringify(desc));
      const [p] = place(t);
      return ok('createMaterial', x.ganbate_rt_material_create(p, t.length));
    },
    updateMaterial(material: number, desc: GanbateMaterialDesc) {
      const t = text(JSON.stringify(desc));
      const [p] = place(t);
      ok('updateMaterial', x.ganbate_rt_material_update(material, p, t.length));
    },
    destroyMaterial(material: number) {
      x.ganbate_rt_material_destroy(material);
    },
    createShader(source: string) {
      const t = text(source);
      const [p] = place(t);
      return ok('createShader', x.ganbate_rt_shader_create(p, t.length));
    },
    setMaterialParams(material: number, params: readonly number[]) {
      const v = new Float32Array(20);
      v.set(params.slice(0, 20));
      const [p] = place(v);
      ok('setMaterialParams', x.ganbate_rt_material_params(material, p));
    },
    createInstance(mesh: number, material: number, matrix: Matrix4) {
      const [p] = place(f32(matrix));
      return ok('createInstance', x.ganbate_rt_instance_create(mesh, material, p));
    },
    setInstance(instance: number, matrix: Matrix4, visible = true) {
      const [p] = place(f32(matrix));
      ok('setInstance', x.ganbate_rt_instance_set(instance, p, visible ? 1 : 0));
    },
    setInstances(list: readonly { readonly instance: number; readonly matrix: Matrix4; readonly visible?: boolean }[]) {
      const n = list.length;
      if (!n) return;
      const p = reserve(n * 72);
      const u = new Uint32Array(x.memory.buffer, p, n * 18);
      const f = new Float32Array(x.memory.buffer, p, n * 18);
      for (let k = 0; k < n; k++) {
        const r = list[k];
        const base = k * 18;
        u[base] = r.instance;
        u[base + 1] = r.visible === false ? 0 : 1;
        f.set(r.matrix, base + 2);
      }
      ok('setInstances', x.ganbate_rt_instances_set(p, n));
    },
    setInstanceMaterial(instance: number, material: number) {
      ok('setInstanceMaterial', x.ganbate_rt_instance_material(instance, material));
    },
    setInstanceTint(instance: number, tint: Rgba) {
      ok('setInstanceTint', x.ganbate_rt_instance_tint(instance, tint[0], tint[1], tint[2], tint[3] ?? 1));
    },
    setInstancePick(instance: number, layers: number) {
      ok('setInstancePick', x.ganbate_rt_instance_pick(instance, layers >>> 0));
    },
    destroyInstance(instance: number) {
      x.ganbate_rt_instance_destroy(instance);
    },
    createLight(light: GanbateLight) {
      const [p] = place(lightFloats(light));
      return ok('createLight', x.ganbate_rt_light_create(p));
    },
    setLight(id: number, light: GanbateLight) {
      const [p] = place(lightFloats(light));
      ok('setLight', x.ganbate_rt_light_set(id, p));
    },
    destroyLight(id: number) {
      x.ganbate_rt_light_destroy(id);
    },
    project(point: Vec3) {
      const { sx, sy } = cssScale();
      const p = out(3).byteOffset;
      if (!x.ganbate_rt_project(point[0], point[1], point[2], p)) return null;
      const [px, py, depth] = new Float32Array(x.memory.buffer, p, 3);
      return { x: px * sx, y: py * sy, depth };
    },
    pick(cssX: number, cssY: number, layers = 0xffffffff) {
      const { sx, sy } = cssScale();
      const p = out(4).byteOffset;
      const instance = x.ganbate_rt_pick(cssX / sx, cssY / sy, layers >>> 0, p);
      if (!instance) return null;
      const [distance, px, py, pz] = new Float32Array(x.memory.buffer, p, 4);
      return { instance, distance, point: [px, py, pz] };
    },
    pickRay(origin: Vec3, direction: Vec3, layers = 0xffffffff) {
      if (typeof x.ganbate_rt_pick_ray !== 'function') return null;
      if (!origin || !direction || origin.length < 3 || direction.length < 3) return null;
      const p = out(4).byteOffset;
      const instance = x.ganbate_rt_pick_ray(
        origin[0],
        origin[1],
        origin[2],
        direction[0],
        direction[1],
        direction[2],
        layers >>> 0,
        p,
      );
      if (!instance) return null;
      const [distance, px, py, pz] = new Float32Array(x.memory.buffer, p, 4);
      return { instance, distance, point: [px, py, pz] };
    },
    worldInstances() {
      return x.ganbate_rt_world_instances();
    },
    worldInstance(index: number) {
      const p = out(16).byteOffset;
      const r = x.ganbate_rt_world_instance(index >>> 0, p);
      if (!r) throw fail('worldInstance');
      return { matrix: Array.from(new Float32Array(x.memory.buffer, p, 16)), visible: r === 1 };
    },
    setWorldInstance(index: number, matrix: Matrix4, visible = true) {
      const [p] = place(f32(matrix));
      ok('setWorldInstance', x.ganbate_rt_set_world_instance(index >>> 0, p, visible ? 1 : 0));
    },
    setWorldLight(index: number, position: Vec3) {
      ok('setWorldLight', x.ganbate_rt_set_world_light(index >>> 0, position[0], position[1], position[2]));
    },
    physicsStart() {
      return x.ganbate_rt_physics_start() !== 0;
    },
    physicsStep(dt: number, steps = 1) {
      const n = Math.floor(steps);
      return x.ganbate_rt_physics_step(dt, n > 0 ? Math.min(n, 0xffffffff) : 0) !== 0;
    },
    physicsStop() {
      x.ganbate_rt_physics_stop();
    },
    physicsError() {
      const len = x.ganbate_rt_physics_error(0, 0);
      if (!len) return null;
      const p = reserve(len);
      x.ganbate_rt_physics_error(p, len);
      return new TextDecoder().decode(new Uint8Array(x.memory.buffer, p, len));
    },
    physicsHash() {
      const p = out(2).byteOffset;
      if (!x.ganbate_rt_physics_hash(p)) return null;
      const b = new Uint8Array(x.memory.buffer, p, 8);
      return Array.from(b)
        .reverse()
        .map((v) => v.toString(16).padStart(2, '0'))
        .join('');
    },
    pickWorld(cssX: number, cssY: number) {
      const { sx, sy } = cssScale();
      const p = out(4).byteOffset;
      const hit = x.ganbate_rt_pick_world(cssX / sx, cssY / sy, p);
      if (!hit) return null;
      const [distance, px, py, pz] = new Float32Array(x.memory.buffer, p, 4);
      return { instance: hit - 1, distance, point: [px, py, pz] };
    },
    pickWorldRay(origin: Vec3, direction: Vec3) {
      if (typeof x.ganbate_rt_pick_world_ray !== 'function') return null;
      if (!origin || !direction || origin.length < 3 || direction.length < 3) return null;
      const p = out(4).byteOffset;
      const hit = x.ganbate_rt_pick_world_ray(
        origin[0],
        origin[1],
        origin[2],
        direction[0],
        direction[1],
        direction[2],
        p,
      );
      if (!hit) return null;
      const [distance, px, py, pz] = new Float32Array(x.memory.buffer, p, 4);
      return { instance: hit - 1, distance, point: [px, py, pz] };
    },
    worldEntities(): GanbateEntityTable | null {
      const len = x.ganbate_world_entities(0, 0);
      if (!len) return null;
      const p = reserve(len);
      x.ganbate_world_entities(p, len);
      return JSON.parse(new TextDecoder().decode(new Uint8Array(x.memory.buffer, p, len)));
    },
    ...graphCalls(x, { reserve, place }),
    async loadModel(source: string | ArrayBuffer | Uint8Array) {
      let bytes: Uint8Array;
      if (typeof source === 'string') {
        const r = await fetch(source);
        if (!r.ok) throw new Error(`${source}: HTTP ${r.status} ${r.statusText}`);
        bytes = new Uint8Array(await r.arrayBuffer());
      } else {
        bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
      }
      const [p] = place(bytes);
      const parsed = ok('loadModel', x.ganbate_rt_model_parse(p, bytes.length));
      const fetchOut = (call: (at: number, cap: number) => number) => {
        const len = call(0, 0);
        const at = reserve(len);
        call(at, len);
        return new Uint8Array(x.memory.buffer, at, len).slice();
      };
      const info = JSON.parse(
        new TextDecoder().decode(fetchOut((at, cap) => x.ganbate_rt_model_describe(parsed, at, cap))),
      ) as GanbateModelInfo;
      const encoded = info.images.map((im, i) => ({
        ...im,
        bytes: fetchOut((at, cap) => x.ganbate_rt_model_image(parsed, i, at, cap)),
      }));
      const textures = await Promise.all(
        encoded.map(async (im) => {
          if (im.mime === 'image/png') {
            const [imgPtr] = place(im.bytes);
            const t = x.ganbate_rt_texture_png(imgPtr, im.bytes.length, im.srgb ? 1 : 0);
            if (!t) onLog(1, `model image ${im.name || ''} not decoded: ${lastError}`);
            return t;
          }
          try {
            const bitmap = await createImageBitmap(
              new Blob([im.bytes as unknown as BlobPart], { type: im.mime }),
              {
                premultiplyAlpha: 'none',
                colorSpaceConversion: 'none',
              },
            );
            const t = engine.createTexture(bitmap, { srgb: im.srgb });
            if (bitmap.close) bitmap.close();
            return t;
          } catch (e) {
            onLog(
              1,
              `model image ${im.name || ''} (${im.mime}) not decoded: ${e instanceof Error ? e.message : String(e)}`,
            );
            return 0;
          }
        }),
      );
      const [tp] = place(new Uint32Array(textures));
      const model = ok('loadModel', x.ganbate_rt_model_create(parsed, tp, textures.length));
      return { model, info, textures };
    },
    destroyModel(model: number) {
      x.ganbate_rt_model_destroy(model);
    },
    modelMaterial(model: number, index: number) {
      return ok('modelMaterial', x.ganbate_rt_model_material(model, index));
    },
    createModelInstance(model: number, matrix: Matrix4) {
      const [p] = place(f32(matrix));
      return ok('createModelInstance', x.ganbate_rt_model_instance(model, p));
    },
    setModelInstance(instance: number, matrix: Matrix4, visible = true) {
      const [p] = place(f32(matrix));
      ok('setModelInstance', x.ganbate_rt_model_instance_set(instance, p, visible ? 1 : 0));
    },
    destroyModelInstance(instance: number) {
      x.ganbate_rt_model_instance_destroy(instance);
    },
    modelRest(instance: number) {
      ok('modelRest', x.ganbate_rt_model_rest(instance));
    },
    snapshotModelPose(instance: number) {
      ok('snapshotModelPose', x.ganbate_rt_model_snapshot_pose(instance));
    },
    poseModelClip(instance: number, packet: Float32Array | readonly number[]) {
      const a = f32(packet);
      const [p] = place(a);
      ok('poseModelClip', x.ganbate_rt_pose_model_clip(instance, p, a.length));
    },
    modelClip(instance: number, clip: number, time: number, weight = 1) {
      ok('modelClip', x.ganbate_rt_model_clip(instance, clip, time, weight));
    },
    modelNode(instance: number, node: number): GanbateNodeTransform {
      const at = out(10).byteOffset;
      ok('modelNode', x.ganbate_rt_model_node_get(instance, node, at));
      const v = Array.from(new Float32Array(x.memory.buffer, at, 10));
      return {
        translation: [v[0], v[1], v[2]],
        rotation: [v[3], v[4], v[5], v[6]],
        scale: [v[7], v[8], v[9]],
      };
    },
    setModelNode(instance: number, node: number, { translation, rotation, scale }: GanbateNodeTransform) {
      const [p] = place(f32([...translation, ...rotation, ...scale]));
      ok('setModelNode', x.ganbate_rt_model_node_set(instance, node, p));
    },
    modelMorph(instance: number, node: number, target: number, weight: number) {
      ok('modelMorph', x.ganbate_rt_model_morph(instance, node, target, weight));
    },
    updateModel(instance: number) {
      ok('updateModel', x.ganbate_rt_model_update(instance));
    },
    modelNodeWorld(instance: number, node: number) {
      const at = out(16).byteOffset;
      ok('modelNodeWorld', x.ganbate_rt_model_node_world(instance, node, at));
      return new Float32Array(x.memory.buffer, at, 16).slice();
    },
    setModelNodeVisible(instance: number, node: number, visible: boolean) {
      ok('setModelNodeVisible', x.ganbate_rt_model_node_visible(instance, node, visible ? 1 : 0));
    },
    setModelTint(instance: number, tint: Rgba) {
      ok('setModelTint', x.ganbate_rt_model_tint(instance, tint[0], tint[1], tint[2], tint[3] ?? 1));
    },
    modelOwner(propInstance: number) {
      const at = reserve(4);
      const instance = x.ganbate_rt_model_owner(propInstance, at);
      if (!instance) return null;
      return { instance, node: new Uint32Array(x.memory.buffer, at, 1)[0] };
    },
    destroy() {
      if (stopped) return;
      stopped = true;
      gate.stop();
      x.ganbate_rt_stop();
      if (settled) host.release?.();
      observer?.disconnect();
      drawn = null;
      freeScratch();
    },
  };
  return engine;
}
