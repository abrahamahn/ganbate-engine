import { mapBounded, preparationBoundary } from './preparation.ts';
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
  options.signal?.throwIfAborted();
  let lastError = '';
  const onLog = options.onLog || (() => {});
  let trying = false;
  const { gpu, host, x, errors, settled } = await boot({
    ...options,
    stagedPipelines: true,
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
  let ioBuf: ArrayBufferLike | null = null;
  let ioBase = -1;
  let ioF32!: Float32Array;
  let ioU32!: Uint32Array;
  const syncIo = (base: number) => {
    const buf = x.memory.buffer;
    if (buf !== ioBuf || base !== ioBase) {
      ioBuf = buf;
      ioBase = base;
      ioF32 = new Float32Array(buf, base, 20);
      ioU32 = new Uint32Array(buf, base, 20);
    }
  };
  const acquireIo = () => {
    const base = reserve(80);
    syncIo(base);
    return base;
  };
  let batchBuf: ArrayBufferLike | null = null;
  let batchPtr = -1;
  let batchWords = 0;
  let batchU32!: Uint32Array;
  let batchF32!: Float32Array;
  const writeLightIo = (light: GanbateLight): number => {
    const base = acquireIo();
    const v = ioF32;
    v[0] = light.position[0];
    v[1] = light.position[1];
    v[2] = light.position[2];
    v[3] = light.range;
    v[4] = light.color[0];
    v[5] = light.color[1];
    v[6] = light.color[2];
    v[7] = light.intensity;
    if (light.direction !== undefined) {
      v[8] = light.direction[0];
      v[9] = light.direction[1];
      v[10] = light.direction[2];
      v[11] = light.inner ?? 0;
      v[12] = light.outer ?? light.inner ?? 0;
    } else {
      v[8] = 0;
      v[9] = 0;
      v[10] = 0;
      v[11] = 0;
      v[12] = 0;
    }
    v[13] = light.followsLamps === true ? 1 : 0;
    v[14] = light.shadow ?? 0;
    return base;
  };
  const readBudget = (p: number): GanbateGovernorSnapshot => {
    syncIo(p);
    const v = ioF32;
    return {
      resolutionScale: v[0],
      maxDynamicLights: v[1] >>> 0,
      ambientParticles: v[2] > 0.5,
      shadowTier: v[3] >>> 0,
      tier: v[4] >>> 0,
      rollingMeanMs: v[5],
    };
  };

  let gradeOnce = false;
  try {
    const bakedAo: Promise<ArrayBuffer | Uint8Array | null> =
      typeof options.ao === 'string'
        ? fetch(options.ao, { signal: options.signal }).then(
            (r) => (r.ok ? r.arrayBuffer() : null),
            () => null,
          )
        : Promise.resolve(options.ao ?? null);

    let world: { ptr: number; len: number } | null = null;
    if (typeof options.world === 'string') {
      world = await fetchInto(options.world, x, (p, bytes) =>
        (options.onPhase || (() => {}))(`Downloading the world (${(bytes / 1048576).toFixed(1)} MB)`, p),
        options.signal,
      );
    } else if (options.world) {
      const bytes = options.world instanceof Uint8Array ? options.world : new Uint8Array(options.world);
      const ptr = x.ganbate_alloc(bytes.length) >>> 0;
      new Uint8Array(x.memory.buffer, ptr, bytes.length).set(bytes);
      world = { ptr, len: bytes.length };
    }
    options.signal?.throwIfAborted();
    if (world) {
      (options.onPhase || (() => {}))('Decoding the world', 1);
      if (!x.ganbate_load_world(world.ptr, world.len)) throw fail('world');
      await decodeWorldImages(x, onLog, options.signal);
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
    try {
      await host.prepare({ signal: options.signal });
    } finally {
      host.setStagedPipelines(false);
    }
    gradeOnce =
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
    options.signal?.throwIfAborted();
  } catch (error) {
    x.ganbate_rt_stop();
    host.release?.();
    if (gpu.api === 'webgpu') gpu.device.destroy?.();
    freeScratch();
    throw error;
  }

  const gate = createFrameGate(gpu, host, settled);
  let stopped = false;
  const modelLifetime = new AbortController();
  void host.lost.then((reason) => modelLifetime.abort(new Error(reason)));
  let drawn: { width: number; height: number } | null = null;
  const drawnSize = () => {
    if (!drawn) {
      const p = acquireIo();
      ok('size', x.ganbate_rt_size(p));
      syncIo(p);
      drawn = { width: ioU32[0], height: ioU32[1] };
    }
    return drawn;
  };
  const size = () => ({ ...drawnSize() });

  let cssBox: { width: number; height: number } | null = null;
  let observer: ResizeObserver | null = null;
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
  const scaleOut = { sx: 1, sy: 1 };
  const cssScale = () => {
    const s = drawnSize();
    const cssW = cssBox ? cssBox.width : canvas.clientWidth;
    const cssH = cssBox ? cssBox.height : canvas.clientHeight;
    scaleOut.sx = (cssW || s.width) / s.width;
    scaleOut.sy = (cssH || s.height) / s.height;
    return scaleOut;
  };

  const engine: GanbateEngine = {
    api: gpu.api,
    present: settled ? 'settled' : 'direct',
    adapter: gpu.adapterName,
    gradeOnce,
    errors: () => [...errors, ...host.errors.filter((e) => !errors.includes(e))],
    lost: host.lost,
    async prepare({ signal } = {}) {
      if (stopped) throw new Error('Cannot prepare a destroyed engine');
      await host.prepare({ signal });
      if (stopped) throw new Error('Engine destroyed during preparation');
    },
    preparationStats: () => host.preparationStats(),
    resourceStats: () => host.resourceStats(),
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
      const up = v.up;
      const shift = v.shift;
      const f = v.focus;
      const p = acquireIo();
      const dst = ioF32;
      dst[0] = v.eye[0];
      dst[1] = v.eye[1];
      dst[2] = v.eye[2];
      dst[3] = v.target[0];
      dst[4] = v.target[1];
      dst[5] = v.target[2];
      dst[6] = up ? up[0] : 0;
      dst[7] = up ? up[1] : 1;
      dst[8] = up ? up[2] : 0;
      dst[9] = v.fov ?? 50;
      dst[10] = shift ? shift[0] : 0;
      dst[11] = shift ? shift[1] : 0;
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
      const p = acquireIo();
      if (!x.ganbate_rt_frame_stats(p)) return null;
      syncIo(p);
      const w = ioU32;
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
      const p = acquireIo();
      if (!x.ganbate_rt_cache_stats(p)) return null;
      syncIo(p);
      const w = ioU32;
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
      const p = acquireIo();
      if (!x.ganbate_rt_stereo_views(p)) return null;
      syncIo(p);
      const v = ioF32;
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
      const p = acquireIo();
      if (!x.ganbate_rt_governor_record(dtMs, p)) return null;
      return readBudget(p);
    },
    governorQuery() {
      if (typeof x.ganbate_rt_governor_query !== 'function') return null;
      const p = acquireIo();
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
    setPetals(amount: number) {
      if (typeof x.ganbate_rt_set_petals === 'function') {
        ok('setPetals', x.ganbate_rt_set_petals(amount));
      }
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
      const p = acquireIo();
      const v = ioF32;
      for (let k = 0; k < 20; k++) v[k] = k < params.length ? params[k] : 0;
      ok('setMaterialParams', x.ganbate_rt_material_params(material, p));
    },
    createInstance(mesh: number, material: number, matrix: Matrix4) {
      const p = acquireIo();
      const v = ioF32;
      for (let k = 0; k < 16; k++) v[k] = k < matrix.length ? matrix[k] : Number.NaN;
      return ok('createInstance', x.ganbate_rt_instance_create(mesh, material, p));
    },
    setInstance(instance: number, matrix: Matrix4, visible = true) {
      const p = acquireIo();
      const v = ioF32;
      for (let k = 0; k < 16; k++) v[k] = k < matrix.length ? matrix[k] : Number.NaN;
      ok('setInstance', x.ganbate_rt_instance_set(instance, p, visible ? 1 : 0));
    },
    setInstances(list: readonly { readonly instance: number; readonly matrix: Matrix4; readonly visible?: boolean }[]) {
      const n = list.length;
      if (!n) return;
      const words = n * 18;
      const p = reserve(n * 72);
      const buf = x.memory.buffer;
      if (buf !== batchBuf || p !== batchPtr || words > batchWords) {
        batchBuf = buf;
        batchPtr = p;
        batchWords = words;
        batchU32 = new Uint32Array(buf, p, words);
        batchF32 = new Float32Array(buf, p, words);
      }
      const u = batchU32;
      const f = batchF32;
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
      return ok('createLight', x.ganbate_rt_light_create(writeLightIo(light)));
    },
    setLight(id: number, light: GanbateLight) {
      ok('setLight', x.ganbate_rt_light_set(id, writeLightIo(light)));
    },
    destroyLight(id: number) {
      x.ganbate_rt_light_destroy(id);
    },
    project(point: Vec3) {
      const s = cssScale();
      const sx = s.sx;
      const sy = s.sy;
      const p = acquireIo();
      if (!x.ganbate_rt_project(point[0], point[1], point[2], p)) return null;
      syncIo(p);
      return { x: ioF32[0] * sx, y: ioF32[1] * sy, depth: ioF32[2] };
    },
    pick(cssX: number, cssY: number, layers = 0xffffffff) {
      const s = cssScale();
      const p = acquireIo();
      const instance = x.ganbate_rt_pick(cssX / s.sx, cssY / s.sy, layers >>> 0, p);
      if (!instance) return null;
      syncIo(p);
      return { instance, distance: ioF32[0], point: [ioF32[1], ioF32[2], ioF32[3]] };
    },
    pickRay(origin: Vec3, direction: Vec3, layers = 0xffffffff) {
      if (typeof x.ganbate_rt_pick_ray !== 'function') return null;
      if (!origin || !direction || origin.length < 3 || direction.length < 3) return null;
      const p = acquireIo();
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
      syncIo(p);
      return { instance, distance: ioF32[0], point: [ioF32[1], ioF32[2], ioF32[3]] };
    },
    worldInstances() {
      return x.ganbate_rt_world_instances();
    },
    worldInstance(index: number) {
      const p = acquireIo();
      const r = x.ganbate_rt_world_instance(index >>> 0, p);
      if (!r) throw fail('worldInstance');
      syncIo(p);
      const matrix = new Array<number>(16);
      for (let k = 0; k < 16; k++) matrix[k] = ioF32[k];
      return { matrix, visible: r === 1 };
    },
    setWorldInstance(index: number, matrix: Matrix4, visible = true) {
      const p = acquireIo();
      const v = ioF32;
      for (let k = 0; k < 16; k++) v[k] = k < matrix.length ? matrix[k] : Number.NaN;
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
      const p = acquireIo();
      if (!x.ganbate_rt_physics_hash(p)) return null;
      const b = new Uint8Array(x.memory.buffer, p, 8);
      return Array.from(b)
        .reverse()
        .map((v) => v.toString(16).padStart(2, '0'))
        .join('');
    },
    pickWorld(cssX: number, cssY: number) {
      const s = cssScale();
      const p = acquireIo();
      const hit = x.ganbate_rt_pick_world(cssX / s.sx, cssY / s.sy, p);
      if (!hit) return null;
      syncIo(p);
      return { instance: hit - 1, distance: ioF32[0], point: [ioF32[1], ioF32[2], ioF32[3]] };
    },
    pickWorldRay(origin: Vec3, direction: Vec3) {
      if (typeof x.ganbate_rt_pick_world_ray !== 'function') return null;
      if (!origin || !direction || origin.length < 3 || direction.length < 3) return null;
      const p = acquireIo();
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
      syncIo(p);
      return { instance: hit - 1, distance: ioF32[0], point: [ioF32[1], ioF32[2], ioF32[3]] };
    },
    worldEntities(): GanbateEntityTable | null {
      const len = x.ganbate_world_entities(0, 0);
      if (!len) return null;
      const p = reserve(len);
      x.ganbate_world_entities(p, len);
      return JSON.parse(new TextDecoder().decode(new Uint8Array(x.memory.buffer, p, len)));
    },
    ...graphCalls(x, { reserve, place }),
    async loadModel(source: string | ArrayBuffer | Uint8Array, { signal }: { readonly signal?: AbortSignal } = {}) {
      const controller = new AbortController();
      const cancel = () => controller.abort(signal?.reason);
      const stop = () => controller.abort(modelLifetime.signal.reason);
      const check = () => {
        controller.signal.throwIfAborted();
        if (stopped) throw new Error('Cannot load a model into a destroyed engine');
      };
      if (signal?.aborted) cancel();
      if (modelLifetime.signal.aborted) stop();
      signal?.addEventListener('abort', cancel, { once: true });
      modelLifetime.signal.addEventListener('abort', stop, { once: true });
      let parsed = 0, model = 0;
      let transferred = false;
      const ownedTextures = new Set<number>();
      const own = (texture: number) => {
        if (texture) ownedTextures.add(texture);
        return texture;
      };
      try {
        check();
        if (typeof x.ganbate_rt_model_discard !== 'function') {
          throw new Error('loadModel requires matching SDK and WASM builds with parsed-model cleanup');
        }
        let bytes: Uint8Array;
        if (typeof source === 'string') {
          const r = await preparationBoundary(fetch(source, { signal: controller.signal }), controller.signal);
          if (!r.ok) throw new Error(`${source}: HTTP ${r.status} ${r.statusText}`);
          bytes = new Uint8Array(await preparationBoundary(r.arrayBuffer(), controller.signal));
        } else {
          bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
        }
        check();
        const [p] = place(bytes);
        parsed = ok('loadModel', x.ganbate_rt_model_parse(p, bytes.length));
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
        const loadImage = async (im: (typeof encoded)[number]) => {
          check();
          if (im.mime === 'image/png') {
            const [imgPtr] = place(im.bytes);
            const t = own(x.ganbate_rt_texture_png(imgPtr, im.bytes.length, im.srgb ? 1 : 0));
            if (!t) onLog(1, `model image ${im.name || ''} not decoded: ${lastError}`);
            check();
            return t;
          }
          let bitmap: ImageBitmap | undefined;
          try {
            const decoding = createImageBitmap(
              new Blob([im.bytes as unknown as BlobPart], { type: im.mime }),
              { premultiplyAlpha: 'none', colorSpaceConversion: 'none' },
            ).then((image) => {
              // Browser decoding cannot be interrupted. A cancelled waiter settles promptly,
              // while a late decoded image closes here without ever touching the engine.
              try { check(); bitmap = image; return image; }
              catch (error) { image.close(); throw error; }
            });
            await preparationBoundary(decoding, controller.signal);
            check();
            const t = own(engine.createTexture(bitmap!, { srgb: im.srgb }));
            check();
            return t;
          } catch (e) {
            check(); // cancellation/destruction is never an optional missing-image fallback
            onLog(
              1,
              `model image ${im.name || ''} (${im.mime}) not decoded: ${e instanceof Error ? e.message : String(e)}`,
            );
            return 0;
          } finally { bitmap?.close(); }
        };
        const textures = await mapBounded(encoded, (im) => loadImage(im).catch((error) => {
          // Stop sibling decoders promptly on a fatal upload/runtime failure. Their late
          // browser results still close through the ownership path above.
          controller.abort(error);
          throw error;
        }));
        check();
        const [tp] = place(new Uint32Array(textures));
        const consumed = parsed;
        parsed = 0; // model_create spends this handle on both success and failure
        model = ok('loadModel', x.ganbate_rt_model_create(consumed, tp, textures.length));
        check();
        transferred = true;
        return { model, info, textures };
      } finally {
        // Invalidate any browser decoder that could resolve after the transaction ends.
        controller.abort();
        signal?.removeEventListener('abort', cancel);
        modelLifetime.signal.removeEventListener('abort', stop);
        if (!transferred && !stopped) {
          if (model) x.ganbate_rt_model_destroy(model);
          if (parsed) x.ganbate_rt_model_discard(parsed);
          for (const texture of ownedTextures) x.ganbate_rt_texture_destroy(texture);
        }
      }
    },
    destroyModel(model: number) {
      x.ganbate_rt_model_destroy(model);
    },
    modelMaterial(model: number, index: number) {
      return ok('modelMaterial', x.ganbate_rt_model_material(model, index));
    },
    createModelInstance(model: number, matrix: Matrix4) {
      const p = acquireIo();
      const v = ioF32;
      for (let k = 0; k < 16; k++) v[k] = k < matrix.length ? matrix[k] : Number.NaN;
      return ok('createModelInstance', x.ganbate_rt_model_instance(model, p));
    },
    setModelInstance(instance: number, matrix: Matrix4, visible = true) {
      const p = acquireIo();
      const v = ioF32;
      for (let k = 0; k < 16; k++) v[k] = k < matrix.length ? matrix[k] : Number.NaN;
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
      const at = acquireIo();
      ok('modelNode', x.ganbate_rt_model_node_get(instance, node, at));
      syncIo(at);
      const v = ioF32;
      return {
        translation: [v[0], v[1], v[2]],
        rotation: [v[3], v[4], v[5], v[6]],
        scale: [v[7], v[8], v[9]],
      };
    },
    setModelNode(instance: number, node: number, { translation, rotation, scale }: GanbateNodeTransform) {
      const p = acquireIo();
      const v = ioF32;
      v[0] = translation[0];
      v[1] = translation[1];
      v[2] = translation[2];
      v[3] = rotation[0];
      v[4] = rotation[1];
      v[5] = rotation[2];
      v[6] = rotation[3];
      v[7] = scale[0];
      v[8] = scale[1];
      v[9] = scale[2];
      ok('setModelNode', x.ganbate_rt_model_node_set(instance, node, p));
    },
    modelMorph(instance: number, node: number, target: number, weight: number) {
      ok('modelMorph', x.ganbate_rt_model_morph(instance, node, target, weight));
    },
    updateModel(instance: number) {
      ok('updateModel', x.ganbate_rt_model_update(instance));
    },
    modelNodeWorld(instance: number, node: number) {
      const at = acquireIo();
      ok('modelNodeWorld', x.ganbate_rt_model_node_world(instance, node, at));
      syncIo(at);
      return ioF32.slice(0, 16);
    },
    setModelNodeVisible(instance: number, node: number, visible: boolean) {
      ok('setModelNodeVisible', x.ganbate_rt_model_node_visible(instance, node, visible ? 1 : 0));
    },
    setModelTint(instance: number, tint: Rgba) {
      ok('setModelTint', x.ganbate_rt_model_tint(instance, tint[0], tint[1], tint[2], tint[3] ?? 1));
    },
    modelOwner(propInstance: number) {
      const at = acquireIo();
      const instance = x.ganbate_rt_model_owner(propInstance, at);
      if (!instance) return null;
      syncIo(at);
      return { instance, node: ioU32[0] };
    },
    destroy() {
      if (stopped) return;
      stopped = true;
      modelLifetime.abort(new Error('Engine destroyed during model loading'));
      gate.stop();
      x.ganbate_rt_stop();
      host.release?.();
      if (gpu.api === 'webgpu') gpu.device.destroy?.();
      observer?.disconnect();
      drawn = null;
      freeScratch();
    },
  };
  return engine;
}
