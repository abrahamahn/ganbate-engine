import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  createEngine,
  createGlHost,
  createHost,
} from './ganbate.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');

function makeRecordingWebGl2Canvas(width = 640, height = 360) {
  const recorded = {
    drawFbos: [],
    blits: [],
    disabledCaps: [],
  };
  const constants = {
    RENDERER: 0x1F01,
    MAX_COMBINED_TEXTURE_IMAGE_UNITS: 0x8B4D,
    NO_ERROR: 0,
    DRAW_FRAMEBUFFER: 0x8CA9,
    READ_FRAMEBUFFER: 0x8CA8,
    FRAMEBUFFER: 0x8D40,
    FRAMEBUFFER_COMPLETE: 0x8CD5,
    COLOR_ATTACHMENT0: 0x8CE0,
    COLOR_BUFFER_BIT: 0x00004000,
    NEAREST: 0x2600,
    LINEAR: 0x2601,
    SCISSOR_TEST: 0x0C11,
    TEXTURE_2D: 0x0DE1,
    RGBA8: 0x8058,
    RGBA: 0x1908,
    UNSIGNED_BYTE: 0x1401,
    COMPILE_STATUS: 0x8B81,
    LINK_STATUS: 0x8B82,
  };
  let nextId = 1;
  const base = {
    ...constants,
    getExtension() { return null; },
    getParameter(param) {
      if (param === constants.RENDERER) return 'MockXRWebGL2';
      if (param === constants.MAX_COMBINED_TEXTURE_IMAGE_UNITS) return 16;
      return 0;
    },
    getShaderParameter() { return true; },
    getProgramParameter() { return true; },
    checkFramebufferStatus() { return constants.FRAMEBUFFER_COMPLETE; },
    createTexture() { return { kind: 'tex', id: nextId++ }; },
    createFramebuffer() { return { kind: 'fbo', id: nextId++ }; },
    createSampler() { return { kind: 'smp', id: nextId++ }; },
    createShader() { return { kind: 'shd', id: nextId++ }; },
    createProgram() { return { kind: 'prg', id: nextId++ }; },
    createVertexArray() { return { kind: 'vao', id: nextId++ }; },
    createBuffer() { return { kind: 'buf', id: nextId++ }; },
    bindFramebuffer(target, fbo) {
      if (target === constants.DRAW_FRAMEBUFFER) {
        recorded.drawFbos.push(fbo);
      }
    },
    disable(cap) {
      recorded.disabledCaps.push(cap);
    },
    blitFramebuffer(sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, mask, filter) {
      recorded.blits.push({ sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, mask, filter });
    },
    getError() { return 0; },
    async makeXRCompatible() { return true; },
  };
  const gl = new Proxy(base, {
    get(target, prop) {
      if (prop in target) return target[prop];
      if (typeof prop === 'string' && /^[A-Z0-9_]+$/.test(prop)) return 0;
      return () => ({});
    },
  });
  const canvas = {
    width,
    height,
    clientWidth: width,
    clientHeight: height,
    addEventListener() {},
    getContext(kind) {
      return kind === 'webgl2' ? gl : null;
    },
  };
  return { canvas, gl, recorded, constants };
}

function makeMockWasm(overrides = {}) {
  const memory = new WebAssembly.Memory({ initial: 4 });
  let bump = 1024;
  const calls = {
    pickRay: [],
    pickWorldRay: [],
  };

  const exports = {
    memory,
    ganbate_alloc(len) {
      const ptr = (bump + 7) & ~7;
      bump = ptr + Math.max(len >>> 0, 8);
      return ptr;
    },
    ganbate_free() {},
    ganbate_rt_start() { return 1; },
    ganbate_rt_stop() { return 1; },
    ganbate_rt_size(outPtr) {
      const dst = new Uint32Array(memory.buffer, outPtr >>> 0, 2);
      dst[0] = 640;
      dst[1] = 360;
      return 1;
    },
    ganbate_rt_pick_ray(ox, oy, oz, dx, dy, dz, layers, outPtr) {
      calls.pickRay.push({ origin: [ox, oy, oz], direction: [dx, dy, dz], layers: layers >>> 0 });
      if ((layers & 0b0010) === 0 || dz >= 0) return 0;
      const dst = new Float32Array(memory.buffer, outPtr >>> 0, 4);
      dst.set([2.75, ox + dx * 2.75, oy + dy * 2.75, oz + dz * 2.75]);
      return 42;
    },
    ganbate_rt_pick_world_ray(ox, oy, oz, dx, dy, dz, outPtr) {
      calls.pickWorldRay.push({ origin: [ox, oy, oz], direction: [dx, dy, dz] });
      if (dz >= 0) return 0;
      const dst = new Float32Array(memory.buffer, outPtr >>> 0, 4);
      dst.set([6.5, ox, oy, oz - 6.5]);
      return 5; // world instance index 4 encoded as index + 1
    },
    ...overrides,
  };

  for (const [key, val] of Object.entries(overrides)) {
    if (val === undefined) delete exports[key];
  }

  return { instance: { exports }, calls, memory };
}

test('pickRay and pickWorldRay marshal 3D rays, layer masks, and world indices cleanly', async () => {
  const { canvas } = makeRecordingWebGl2Canvas();
  const { instance, calls } = makeMockWasm();
  const engine = await createEngine({
    canvas,
    wasm: instance,
    api: 'webgl2',
  });

  const propHit = engine.pickRay([0.25, 1.5, 3.0], [0.0, 0.0, -1.0], 0b0010);
  assert.ok(propHit);
  assert.equal(propHit.instance, 42);
  assert.ok(Math.abs(propHit.distance - 2.75) < 1e-6);
  assert.ok(Math.abs(propHit.point[0] - 0.25) < 1e-6);
  assert.ok(Math.abs(propHit.point[1] - 1.5) < 1e-6);
  assert.ok(Math.abs(propHit.point[2] - 0.25) < 1e-6);

  assert.equal(engine.pickRay([0.25, 1.5, 3.0], [0.0, 0.0, -1.0], 0b0001), null);
  assert.equal(engine.pickRay([0.25, 1.5, 3.0], [0.0, 0.0, 1.0]), null);
  assert.equal(engine.pickRay(null, [0, 0, -1]), null);
  assert.equal(engine.pickRay([0, 0], [0, 0, -1]), null);

  assert.equal(calls.pickRay.length, 3);
  assert.equal(calls.pickRay[2].layers, 0xFFFFFFFF);

  const worldHit = engine.pickWorldRay([1.0, 1.5, 4.0], [0.0, 0.0, -1.0]);
  assert.ok(worldHit);
  assert.equal(worldHit.instance, 4);
  assert.ok(Math.abs(worldHit.distance - 6.5) < 1e-6);
  assert.deepEqual(worldHit.point, [1.0, 1.5, -2.5]);

  assert.equal(engine.pickWorldRay([1.0, 1.5, 4.0], [0.0, 0.0, 1.0]), null);
  assert.equal(engine.pickWorldRay([1.0, 1.5], [0.0, 0.0, -1.0]), null);
});

test('pickRay and pickWorldRay degrade gracefully to null when WASM omits ray exports', async () => {
  const { canvas } = makeRecordingWebGl2Canvas();
  const { instance } = makeMockWasm({
    ganbate_rt_pick_ray: undefined,
    ganbate_rt_pick_world_ray: undefined,
  });
  const engine = await createEngine({
    canvas,
    wasm: instance,
    api: 'webgl2',
  });

  assert.equal(engine.pickRay([0, 1.6, 3], [0, 0, -1]), null);
  assert.equal(engine.pickWorldRay([0, 1.6, 3], [0, 0, -1]), null);
});

/** A canvas whose CSS box counts its reads, and a mock engine that projects to (100, 50). */
async function makeLayoutProbe() {
  const { canvas } = makeRecordingWebGl2Canvas(640, 360);
  const probe = { reads: 0, box: [320, 180], sizeCalls: 0, picks: [] };
  Object.defineProperty(canvas, 'clientWidth', { get() { probe.reads++; return probe.box[0]; } });
  Object.defineProperty(canvas, 'clientHeight', { get() { probe.reads++; return probe.box[1]; } });
  const mock = makeMockWasm({
    ganbate_rt_size(outPtr) {
      probe.sizeCalls++;
      new Uint32Array(mock.memory.buffer, outPtr >>> 0, 2).set([640, 360]);
      return 1;
    },
    ganbate_rt_project(_x, _y, _z, outPtr) {
      new Float32Array(mock.memory.buffer, outPtr >>> 0, 3).set([100, 50, 2]);
      return 1;
    },
    ganbate_rt_pick(x, y) {
      probe.picks.push([x, y]);
      return 0;
    },
    ganbate_rt_pick_world(x, y) {
      probe.picks.push([x, y]);
      return 0;
    },
    ganbate_rt_render() { return 2; },
  });
  const engine = await createEngine({ canvas, wasm: mock.instance, api: 'webgl2' });
  probe.reads = 0;
  return { canvas, engine, probe };
}

test('project, pick and fitCanvas use the observed CSS box and read the drawn size once a frame', async () => {
  const saved = { ResizeObserver: globalThis.ResizeObserver, getComputedStyle: globalThis.getComputedStyle };
  const observers = [];
  globalThis.ResizeObserver = class {
    constructor(callback) { this.callback = callback; observers.push(this); }
    observe(target) { this.target = target; }
    disconnect() { this.target = null; }
  };
  let writingMode = 'horizontal-tb';
  globalThis.getComputedStyle = () => ({ writingMode });
  try {
    const { canvas, engine, probe } = await makeLayoutProbe();
    assert.equal(observers.length, 1);
    assert.equal(observers[0].target, canvas);

    // The box read at start: 320x180 CSS pixels over 640x360 drawn, so half a CSS pixel each.
    assert.deepEqual(engine.project([0, 0, 0]), { x: 50, y: 25, depth: 2 });
    engine.pick(10, 20);
    engine.pickWorld(10, 20);
    engine.project([1, 2, 3]);
    assert.deepEqual(probe.picks, [[20, 40], [20, 40]]);
    engine.fitCanvas(3);
    assert.deepEqual([canvas.width, canvas.height], [960, 540]);
    assert.equal(probe.reads, 0, 'no layout reads after start');
    assert.equal(probe.sizeCalls, 1, 'the drawn size is read once until the next frame');

    // Resizes arrive through the observer, rounded to whole CSS pixels like clientWidth.
    probe.box = [1, 1];
    const resize = (entry) => observers[0].callback([entry]);
    resize({ contentBoxSize: [{ inlineSize: 480.4, blockSize: 270 }], contentRect: { width: 480.4, height: 270 } });
    assert.deepEqual(engine.project([0, 0, 0]), { x: 75, y: 37.5, depth: 2 });
    writingMode = 'vertical-rl';
    resize({ contentBoxSize: [{ inlineSize: 180, blockSize: 320 }], contentRect: { width: 320, height: 180 } });
    assert.deepEqual(engine.project([0, 0, 0]), { x: 50, y: 25, depth: 2 });
    writingMode = 'horizontal-tb';
    resize({ contentBoxSize: { inlineSize: 640, blockSize: 360 } });
    assert.deepEqual(engine.project([0, 0, 0]), { x: 100, y: 50, depth: 2 });
    resize({ contentRect: { width: 160, height: 90 } });
    engine.fitCanvas(2);
    assert.deepEqual([canvas.width, canvas.height], [320, 180]);
    assert.equal(probe.reads, 0);

    // A frame may resize the drawing, so the next call reads the size again.
    assert.equal(probe.sizeCalls, 1);
    engine.render();
    engine.project([0, 0, 0]);
    engine.project([0, 0, 0]);
    assert.equal(probe.sizeCalls, 2);
    const s = engine.size();
    s.width = 1;
    assert.deepEqual(engine.size(), { width: 640, height: 360 });

    engine.destroy();
    assert.equal(observers[0].target, null);
  } finally {
    for (const [k, v] of Object.entries(saved)) {
      if (v === undefined) delete globalThis[k];
      else globalThis[k] = v;
    }
  }
});

test('without ResizeObserver, project and fitCanvas read the CSS box on every call', async () => {
  assert.equal(typeof globalThis.ResizeObserver, 'undefined');
  const { canvas, engine, probe } = await makeLayoutProbe();
  assert.deepEqual(engine.project([0, 0, 0]), { x: 50, y: 25, depth: 2 });
  probe.box = [640, 360];
  assert.deepEqual(engine.project([0, 0, 0]), { x: 100, y: 50, depth: 2 });
  engine.fitCanvas(1);
  assert.deepEqual([canvas.width, canvas.height], [640, 360]);
  assert.equal(probe.reads, 6);
});

test('glContext and setXrTarget support WebXR stereo eye framebuffer blits on WebGL 2 and restore canvas blit', async () => {
  const { canvas, gl, recorded, constants } = makeRecordingWebGl2Canvas(640, 360);
  let glSubmit = null;
  const memory = new WebAssembly.Memory({ initial: 2 });
  const gpu = {
    api: 'webgl2',
    gl,
    ext: {},
    canvas,
    adapterName: 'WebGL 2 MockXRWebGL2',
    features: [],
    format: 'rgba8unorm',
  };
  const host = createGlHost(gpu);
  host.attach({ exports: { memory } });
  glSubmit = host.imports.ganbate.gl_submit;

  // Default canvas presentation blits (0, 0, w, h) -> (0, h, w, 0) to DRAW_FRAMEBUFFER = null.
  assert.equal(glSubmit(0, 0, 0, 0, 1), 1);
  assert.equal(recorded.blits.length, 1);
  assert.deepEqual(recorded.blits[0], {
    sx0: 0,
    sy0: 0,
    sx1: 640,
    sy1: 360,
    dx0: 0,
    dy0: 360,
    dx1: 640,
    dy1: 0,
    mask: constants.COLOR_BUFFER_BIT,
    filter: constants.NEAREST,
  });

  // Left and right WebXR eye viewports bind XRWebGLLayer framebuffer and flip Y inside each eye sub-rect.
  const xrLayerFbo = { kind: 'xr-layer-fbo' };
  assert.equal(host.setXrTarget({ framebuffer: xrLayerFbo, x: 0, y: 0, width: 960, height: 1080 }), true);
  assert.equal(glSubmit(0, 0, 0, 0, 1), 1);
  assert.ok(recorded.drawFbos.includes(xrLayerFbo));
  assert.ok(recorded.disabledCaps.includes(constants.SCISSOR_TEST));
  assert.deepEqual(recorded.blits[1], {
    sx0: 0,
    sy0: 0,
    sx1: 640,
    sy1: 360,
    dx0: 0,
    dy0: 1080,
    dx1: 960,
    dy1: 0,
    mask: constants.COLOR_BUFFER_BIT,
    filter: constants.LINEAR,
  });

  assert.equal(host.setXrTarget({ framebuffer: xrLayerFbo, x: 960, y: 20, width: 640, height: 360 }), true);
  assert.equal(glSubmit(0, 0, 0, 0, 1), 1);
  assert.deepEqual(recorded.blits[2], {
    sx0: 0,
    sy0: 0,
    sx1: 640,
    sy1: 360,
    dx0: 960,
    dy0: 380,
    dx1: 1600,
    dy1: 20,
    mask: constants.COLOR_BUFFER_BIT,
    filter: constants.NEAREST,
  });

  // Invalid XR targets are rejected with false; null restores default canvas presentation.
  assert.equal(host.setXrTarget({ framebuffer: xrLayerFbo, x: 0, y: 0, width: 0, height: 360 }), false);
  assert.equal(host.setXrTarget({ framebuffer: xrLayerFbo, x: Number.NaN, y: 0, width: 640, height: 360 }), false);
  assert.equal(host.setXrTarget(null), true);
  assert.equal(glSubmit(0, 0, 0, 0, 1), 1);
  assert.deepEqual(recorded.blits[3], recorded.blits[0]);

  // Verify createEngine exposes glContext() and setXrTarget() on WebGL2 and returns null/false on WebGPU.
  const { instance } = makeMockWasm();
  const engine = await createEngine({
    canvas,
    wasm: instance,
    api: 'webgl2',
  });
  assert.equal(engine.glContext(), gl);
  assert.equal(await engine.glContext().makeXRCompatible(), true);
  assert.equal(engine.setXrTarget({ framebuffer: xrLayerFbo, x: 0, y: 0, width: 640, height: 360 }), true);
  assert.equal(engine.setXrTarget(null), true);

  const makeMockGpuDevice = () => new Proxy(
    { lost: new Promise(() => {}) },
    {
      get(target, prop) {
        if (prop === 'then') return undefined;
        if (prop in target) return target[prop];
        return () => ({ destroy() {} });
      },
    },
  );
  const prevNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  const prevGpuTextureUsage = globalThis.GPUTextureUsage;
  const prevGpuShaderStage = globalThis.GPUShaderStage;
  const prevGpuBufferUsage = globalThis.GPUBufferUsage;
  try {
    globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 0x10, COPY_SRC: 0x01 };
    globalThis.GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
    globalThis.GPUBufferUsage = { UNIFORM: 0x40, COPY_DST: 0x08 };
    const webgpuHost = createHost({ api: 'webgpu', device: makeMockGpuDevice(), canvas: null, context: null });
    assert.equal(webgpuHost.setXrTarget({ framebuffer: null, x: 0, y: 0, width: 640, height: 360 }), false);

    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: {
        gpu: {
          getPreferredCanvasFormat: () => 'bgra8unorm',
          async requestAdapter() {
            return {
              features: new Set(),
              limits: {
                maxBufferSize: 65536,
                maxStorageBufferBindingSize: 65536,
                maxTextureDimension2D: 2048,
              },
              info: { vendor: 'MockWebGPU' },
              async requestDevice() {
                return makeMockGpuDevice();
              },
            };
          },
        },
      },
    });
    const gpuCanvas = {
      width: 640,
      height: 360,
      clientWidth: 640,
      clientHeight: 360,
      getContext(kind) {
        return kind === 'webgpu' ? { configure() {} } : null;
      },
    };
    const { instance: gpuWasm } = makeMockWasm();
    const webgpuEngine = await createEngine({
      canvas: gpuCanvas,
      wasm: gpuWasm,
      api: 'webgpu',
    });
    assert.equal(webgpuEngine.api, 'webgpu');
    assert.equal(webgpuEngine.glContext(), null);
    assert.equal(webgpuEngine.setXrTarget({ framebuffer: xrLayerFbo, x: 0, y: 0, width: 640, height: 360 }), false);
    assert.equal(webgpuEngine.setXrTarget(null), false);
  } finally {
    if (prevNavigator) Object.defineProperty(globalThis, 'navigator', prevNavigator);
    else delete globalThis.navigator;
    if (prevGpuTextureUsage !== undefined) globalThis.GPUTextureUsage = prevGpuTextureUsage;
    else delete globalThis.GPUTextureUsage;
    if (prevGpuShaderStage !== undefined) globalThis.GPUShaderStage = prevGpuShaderStage;
    else delete globalThis.GPUShaderStage;
    if (prevGpuBufferUsage !== undefined) globalThis.GPUBufferUsage = prevGpuBufferUsage;
    else delete globalThis.GPUBufferUsage;
  }
});

test('web/ganbate.d.ts and compiled ganbate_web.wasm expose Wave 8 ray-picking and WebXR contracts', () => {
  const dts = fs.readFileSync(path.join(__dirname, 'ganbate.d.ts'), 'utf8');
  for (const token of [
    'export interface GanbatePickHit',
    'export interface GanbateWorldPickHit',
    'export interface GanbateXrTarget',
    'pickRay(origin: Vec3, direction: Vec3, layers?: number): GanbatePickHit | null;',
    'pickWorldRay(origin: Vec3, direction: Vec3): GanbateWorldPickHit | null;',
    'glContext(): WebGL2RenderingContext | null;',
    'setXrTarget(target: GanbateXrTarget | null): boolean;',
  ]) {
    assert.ok(dts.includes(token), `missing TypeScript declaration: ${token}`);
  }

  const wasmPath = path.join(
    REPO_ROOT,
    'target',
    'wasm32-unknown-unknown',
    'release',
    'ganbate_web.wasm',
  );
  if (!fs.existsSync(wasmPath)) return;
  const mod = new WebAssembly.Module(fs.readFileSync(wasmPath));
  const exportedFns = new Set(
    WebAssembly.Module.exports(mod)
      .filter((e) => e.kind === 'function')
      .map((e) => e.name),
  );
  assert.ok(exportedFns.has('ganbate_rt_pick_ray'), 'missing WASM export ganbate_rt_pick_ray');
  assert.ok(exportedFns.has('ganbate_rt_pick_world_ray'), 'missing WASM export ganbate_rt_pick_world_ray');
});
