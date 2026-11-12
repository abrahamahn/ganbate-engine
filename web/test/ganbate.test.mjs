import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  createEngine,
  createGlHost,
  createGraphSolver,
  createHost,
  gradeOnceFor,
  resolveGradeOnce,
} from '../sdk/ganbate.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..');

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
    assert.equal(probe.reads, 0, 'project and pick read no layout after start');
    // fitCanvas reads the box itself, once a frame: a page's own resize handler may run first.
    engine.fitCanvas(3);
    assert.deepEqual([canvas.width, canvas.height], [960, 540]);
    assert.equal(probe.reads, 2);
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
    assert.deepEqual(engine.project([0, 0, 0]), { x: 25, y: 12.5, depth: 2 });
    assert.equal(probe.reads, 2);
    probe.box = [160, 90];
    engine.fitCanvas(2);
    assert.deepEqual([canvas.width, canvas.height], [320, 180]);
    assert.equal(probe.reads, 4);

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

test('cacheStats reads the eight words of the last frame\'s cached picture, the re-shaded pixels last', async () => {
  const { canvas } = makeRecordingWebGl2Canvas();
  const mock = makeMockWasm({
    ganbate_rt_cache_stats(outPtr) {
      new Uint32Array(mock.memory.buffer, outPtr >>> 0, 8).set([1, 2, 1, 0, 1, 3, 4, 450892]);
      return 1;
    },
  });
  const engine = await createEngine({ canvas, wasm: mock.instance, api: 'webgl2' });
  assert.deepEqual(engine.cacheStats(), {
    worldCached: true,
    regions: 2,
    reshaded: true,
    waterKept: false,
    waterStaticsWhole: true,
    waterStaticsRegions: 3,
    waterFootprints: 4,
    reshadePixels: 450892,
  });
  const dts = fs.readFileSync(path.join(__dirname, '..', 'sdk', 'ganbate.d.ts'), 'utf8');
  assert.ok(dts.includes('readonly reshadePixels: number;'));
  const without = makeMockWasm({ ganbate_rt_cache_stats: undefined });
  const bare = await createEngine({ canvas, wasm: without.instance, api: 'webgl2' });
  assert.equal(bare.cacheStats(), null);
});

test('gradeOnce \'auto\' grades once on Arm Mali: the WebGPU adapter\'s vendor, or a WebGL 2 renderer that renders floats', () => {
  const webgpu = (info) => ({ api: 'webgpu', adapter: { info } });
  assert.equal(gradeOnceFor(webgpu({ vendor: 'arm', architecture: 'valhall' })), true);
  assert.equal(gradeOnceFor(webgpu({ vendor: 'ARM' })), true);
  assert.equal(gradeOnceFor(webgpu({ vendor: 'qualcomm' })), false);
  assert.equal(gradeOnceFor(webgpu({})), false);
  assert.equal(gradeOnceFor({ api: 'webgpu', adapter: {} }), false);
  const webgl = (renderer, colorFloat = {}) => ({ api: 'webgl2', renderer, ext: { colorFloat } });
  assert.equal(gradeOnceFor(webgl('ANGLE (ARM, Mali-G57 MC2, OpenGL ES 3.2)')), true);
  assert.equal(gradeOnceFor(webgl('Mali-G615')), true);
  // Without EXT_color_buffer_float no 32-bit float target: the fused pass.
  assert.equal(gradeOnceFor(webgl('Mali-G615', null)), false);
  assert.equal(gradeOnceFor(webgl('Adreno (TM) 740')), false);
  assert.equal(gradeOnceFor(webgl(undefined)), false);
  const mali = webgl('Mali-G57');
  assert.deepEqual([undefined, false, true, 'auto', 'yes'].map((o) => resolveGradeOnce(o, mali)), [false, false, true, true, false]);
  assert.deepEqual([true, 'auto'].map((o) => resolveGradeOnce(o, webgpu({ vendor: 'apple' }))), [true, false]);
  const dts = fs.readFileSync(path.join(__dirname, '..', 'sdk', 'ganbate.d.ts'), 'utf8');
  for (const token of ["readonly gradeOnce?: boolean | 'auto';", 'readonly gradeOnce: boolean;', 'export function gradeOnceFor(']) {
    assert.ok(dts.includes(token), `missing TypeScript declaration: ${token}`);
  }
});

test('createEngine grades once where gradeOnce resolves true and the engine can, and says whether it does', async () => {
  const start = async (gradeOnce, renderer, overrides = {}) => {
    const { canvas, gl, constants } = makeRecordingWebGl2Canvas();
    const getParameter = gl.getParameter;
    gl.getParameter = (p) => (p === constants.RENDERER ? renderer : getParameter(p));
    gl.getExtension = (name) => (name === 'EXT_color_buffer_float' ? {} : null);
    const asked = [];
    const mock = makeMockWasm({
      ganbate_rt_set_grade_once(on) {
        asked.push(on);
        return on;
      },
      ...overrides,
    });
    const engine = await createEngine({ canvas, wasm: mock.instance, api: 'webgl2', gradeOnce });
    return [engine.gradeOnce, asked];
  };
  assert.deepEqual(await start('auto', 'Mali-G57 MC2'), [true, [1]]);
  assert.deepEqual(await start('auto', 'Adreno (TM) 740'), [false, []]);
  assert.deepEqual(await start(true, 'Adreno (TM) 740'), [true, [1]]);
  assert.deepEqual(await start(undefined, 'Mali-G57 MC2'), [false, []]);
  // An engine that cannot (no 32-bit float textures), or a module without the export.
  assert.deepEqual(await start(true, 'Mali-G57 MC2', { ganbate_rt_set_grade_once: () => 0 }), [false, []]);
  assert.deepEqual(await start(true, 'Mali-G57 MC2', { ganbate_rt_set_grade_once: undefined }), [false, []]);
});

/** Sets globals for one test and puts the previous values back afterwards. */
async function withGlobals(values, run) {
  const saved = Object.keys(values).map((k) => [k, Object.getOwnPropertyDescriptor(globalThis, k)]);
  for (const [k, v] of Object.entries(values)) {
    Object.defineProperty(globalThis, k, { configurable: true, writable: true, value: v });
  }
  try {
    return await run();
  } finally {
    for (const [k, d] of saved) {
      if (d) Object.defineProperty(globalThis, k, d);
      else delete globalThis[k];
    }
  }
}

/** pixelsOf's 2D canvas: counts the readbacks and returns pixels of 7. */
function makeReadbackCanvas(readbacks) {
  return class {
    constructor(width, height) { Object.assign(this, { width, height }); readbacks.push([width, height]); }
    getContext() {
      return {
        clearRect() {},
        drawImage() {},
        getImageData: (_x, _y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4).fill(7) }),
      };
    }
  };
}

/**
 * An engine on `api` over a mock module whose texture exports act like the runtime: during the
 * call they create the texture through the host's imports and fill it from the host's source,
 * and a refusal is logged at level 2 and returns 0. `uploads` records what reached the GPU API.
 */
async function makeTextureEngine(api, exportOverrides = {}) {
  const t = { calls: [], uploads: [], logs: [], readbacks: [], imports: null, unpack: null, refuse: false };
  const sizes = new Map();
  const mock = makeMockWasm({
    ganbate_rt_texture_create_source(w, h, srgb) {
      t.calls.push(['create_source', w, h, srgb]);
      const g = t.imports;
      const tex = api === 'webgl2'
        ? g.gl_create_texture(w, h, 1, 1, srgb ? 2 : 1, 0)
        : g.gpu_create_texture(w, h, 1, 1, srgb ? 2 : 1, 0);
      sizes.set(tex, [w, h]);
      if (write(tex, w, h)) return tex;
      say('texture: the browser refused the source');
      return 0;
    },
    ganbate_rt_texture_update_source(tex, w, h) {
      t.calls.push(['update_source', tex, w, h]);
      const [tw, th] = sizes.get(tex);
      if (tw === w && th === h && write(tex, w, h)) return 1;
      say(`write_texture_source: a ${w}x${h} image for a ${tw}x${th} mip`);
      return 0;
    },
    ganbate_rt_texture_create(w, h, _p, len, srgb) {
      t.calls.push(['create', w, h, len, srgb]);
      return 99;
    },
    ganbate_rt_texture_update(tex, _p, len) {
      t.calls.push(['update', tex, len]);
      const [tw, th] = sizes.get(tex) || [0, 0];
      if (len === tw * th * 4) return 1;
      say(`write_texture: expected ${tw * th * 4} bytes, got ${len}`);
      return 0;
    },
    ganbate_rt_texture_create_clear(...args) {
      t.calls.push(['clear', ...args]);
      return 77;
    },
    ...exportOverrides,
  });
  const write = (tex, w, h) => (api === 'webgl2'
    ? t.imports.gl_write_texture_source(tex, 0, 0, w, h)
    : t.imports.gpu_copy_source(tex, 0, 0, w, h)) === 1;
  const say = (text) => {
    const bytes = new TextEncoder().encode(text);
    new Uint8Array(mock.memory.buffer, 64, bytes.length).set(bytes);
    t.imports.log(2, 64, bytes.length);
  };

  let canvas;
  const globals = { OffscreenCanvas: makeReadbackCanvas(t.readbacks) };
  if (api === 'webgl2') {
    const made = makeRecordingWebGl2Canvas(16, 16);
    canvas = made.canvas;
    const { gl } = made;
    // The page left its own unpack state, which an upload must restore.
    const unpack = { 0x9240: true, 0x9241: true, 0x9243: 0 };
    t.unpack = unpack;
    Object.assign(gl, {
      UNPACK_FLIP_Y_WEBGL: 0x9240,
      UNPACK_PREMULTIPLY_ALPHA_WEBGL: 0x9241,
      UNPACK_COLORSPACE_CONVERSION_WEBGL: 0x9243,
      BROWSER_DEFAULT_WEBGL: 0x9244,
      TEXTURE_2D_ARRAY: 0x8C1A,
    });
    const baseGetParameter = gl.getParameter;
    gl.getParameter = (p) => (p in unpack ? unpack[p] : baseGetParameter(p));
    gl.pixelStorei = (p, v) => { if (p in unpack) unpack[p] = v; };
    gl.texSubImage2D = (...args) => t.uploads.push({ args, unpack: { ...unpack } });
  } else {
    const device = new Proxy({
      lost: new Promise(() => {}),
      queue: {
        copyExternalImageToTexture(src, dst, size) {
          if (t.refuse) throw new Error('unsupported source');
          t.uploads.push({ src, dst, size });
        },
        submit() {},
      },
      createTexture: (desc) => ({ desc, destroy() {} }),
    }, {
      get(target, prop) {
        if (prop === 'then') return undefined;
        if (prop in target) return target[prop];
        return () => ({ destroy() {} });
      },
    });
    canvas = { width: 16, height: 16, clientWidth: 16, clientHeight: 16, getContext: () => ({ configure() {} }) };
    Object.assign(globals, {
      GPUTextureUsage: { RENDER_ATTACHMENT: 0x10, COPY_SRC: 0x01 },
      GPUShaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 },
      GPUBufferUsage: { UNIFORM: 0x40, COPY_DST: 0x08 },
      navigator: {
        gpu: {
          getPreferredCanvasFormat: () => 'bgra8unorm',
          requestAdapter: async () => ({
            features: new Set(),
            limits: { maxBufferSize: 65536, maxStorageBufferBindingSize: 65536, maxTextureDimension2D: 2048 },
            info: { vendor: 'MockWebGPU' },
            requestDevice: async () => device,
          }),
        },
      },
    });
  }
  // The module is instantiated with the host's imports, so the mock exports can call them.
  const instantiate = WebAssembly.instantiate;
  WebAssembly.instantiate = async (_bytes, imports) => {
    t.imports = imports.ganbate;
    return { instance: mock.instance };
  };
  try {
    t.engine = await withGlobals(globals, () => createEngine({
      canvas, wasm: new Uint8Array(8), api, onLog: (level, text) => t.logs.push([level, text]),
    }));
  } finally {
    WebAssembly.instantiate = instantiate;
  }
  t.withGlobals = (run) => withGlobals(globals, run);
  return t;
}

const canvasSource = (width, height) => ({ width, height, getContext() { return null; } });
const imageSource = (width, height) => ({ naturalWidth: width, naturalHeight: height, width, height });

test('WebGL 2: canvases upload from the source itself with straight alpha, unflipped, and the page\'s unpack state comes back', async () => {
  const t = await makeTextureEngine('webgl2');
  await t.withGlobals(() => {
    const canvas = canvasSource(4, 2);
    const tex = t.engine.createTexture(canvas, { srgb: false });
    assert.deepEqual(t.calls, [['create_source', 4, 2, 0]]);
    assert.equal(t.readbacks.length, 0, 'no pixels are read back');
    assert.equal(t.uploads.length, 1);
    const [target, mip, x, y, w, h, , , image] = t.uploads[0].args;
    assert.deepEqual([target, mip, x, y, w, h, image], [0x0DE1, 0, 0, 0, 4, 2, canvas]);
    assert.deepEqual(t.uploads[0].unpack, { 0x9240: false, 0x9241: false, 0x9243: 0x9244 });
    assert.deepEqual(t.unpack, { 0x9240: true, 0x9241: true, 0x9243: 0 });

    const redrawn = canvasSource(4, 2);
    t.engine.updateTexture(tex, redrawn);
    assert.deepEqual(t.calls[1], ['update_source', tex, 4, 2]);
    assert.equal(t.uploads[1].args[8], redrawn);
    assert.equal(t.readbacks.length, 0);
    // The host holds the source for that one call only.
    assert.equal(t.imports.gl_write_texture_source(tex, 0, 0, 4, 2), 0);
    assert.equal(t.uploads.length, 2);
  });
});

test('WebGL 2 keeps images and {data} on the pixel path; WebGPU copies images too', async () => {
  const gl = await makeTextureEngine('webgl2');
  await gl.withGlobals(() => {
    gl.engine.createTexture(imageSource(4, 2));
    gl.engine.createTexture({ width: 1, height: 1, data: new Uint8Array(4) });
    assert.deepEqual(gl.calls, [['create', 4, 2, 32, 1], ['create', 1, 1, 4, 1]]);
    assert.deepEqual(gl.readbacks, [[4, 2]], 'only the image is read back');
    // An image bitmap's alpha cannot be un-premultiplied by a WebGL upload, so it is read back too.
    const saved = globalThis.ImageBitmap;
    globalThis.ImageBitmap = class { constructor(w, h) { this.width = w; this.height = h; } close() {} };
    try {
      gl.engine.createTexture(new globalThis.ImageBitmap(2, 2));
    } finally {
      if (saved === undefined) delete globalThis.ImageBitmap;
      else globalThis.ImageBitmap = saved;
    }
    assert.deepEqual(gl.calls.at(-1), ['create', 2, 2, 16, 1]);
    assert.deepEqual(gl.readbacks.at(-1), [2, 2]);
  });

  const gpu = await makeTextureEngine('webgpu');
  await gpu.withGlobals(() => {
    const image = imageSource(4, 2);
    gpu.engine.createTexture(image);
    assert.deepEqual(gpu.calls, [['create_source', 4, 2, 1]]);
    assert.equal(gpu.readbacks.length, 0);
    assert.deepEqual(gpu.uploads, [{
      src: { source: image, flipY: false },
      dst: { texture: gpu.uploads[0].dst.texture, mipLevel: 0, origin: [0, 0, 0], colorSpace: 'srgb', premultipliedAlpha: false },
      size: [4, 2],
    }]);
    assert.equal(gpu.uploads[0].dst.texture.desc.format, 'rgba8unorm-srgb');
  });
});

test('a refused, failing or missing source upload falls back to the pixels with a warning', async () => {
  // The browser refuses the source: the runtime logs at level 2, which the engine turns into a warning.
  const refused = await makeTextureEngine('webgpu');
  refused.refuse = true;
  await refused.withGlobals(() => {
    assert.equal(refused.engine.createTexture(canvasSource(4, 2)), 99);
    assert.deepEqual(refused.calls.map((c) => c[0]), ['create_source', 'create']);
    assert.deepEqual(refused.readbacks, [[4, 2]]);
    assert.ok(refused.logs.some(([l, s]) => l === 1 && s.includes('refused') && s.includes('sending the pixels instead')));
    assert.ok(!refused.logs.some(([l]) => l >= 2), 'no error is reported');
  });

  // The export traps: the module is dead, so the trap surfaces instead of a fallback into it.
  const trapped = await makeTextureEngine('webgl2', {
    ganbate_rt_texture_create_source() { throw new WebAssembly.RuntimeError('unreachable'); },
  });
  await trapped.withGlobals(() => {
    assert.throws(() => trapped.engine.createTexture(canvasSource(2, 2)), WebAssembly.RuntimeError);
    assert.deepEqual(trapped.readbacks, []);
  });

  // An older module without the exports reads the pixels, as before.
  const old = await makeTextureEngine('webgl2', {
    ganbate_rt_texture_create_source: undefined,
    ganbate_rt_texture_update_source: undefined,
  });
  await old.withGlobals(() => {
    assert.equal(old.engine.createTexture(canvasSource(2, 2)), 99);
    assert.deepEqual(old.calls, [['create', 2, 2, 16, 1]]);
  });
});

test('updateTexture with a source of another size fails after trying the pixels', async () => {
  const t = await makeTextureEngine('webgl2');
  await t.withGlobals(() => {
    const tex = t.engine.createTexture(canvasSource(4, 2));
    assert.throws(() => t.engine.updateTexture(tex, canvasSource(8, 8)),
      /updateTexture: write_texture: expected 32 bytes, got 256/);
    assert.deepEqual(t.calls.slice(1), [['update_source', tex, 8, 8], ['update', tex, 256]]);
    assert.ok(t.logs.some(([l, s]) => l === 1 && s.includes('a 8x8 image for a 4x2 mip')));
    assert.equal(t.uploads.length, 1, 'the mismatched source never reaches the texture');
  });
});

test('createTexture({width, height, clear}) fills on the GPU and checks its arguments', async () => {
  const t = await makeTextureEngine('webgl2');
  await t.withGlobals(() => {
    assert.equal(t.engine.createTexture({ width: 512, height: 717, clear: [0.9, 0.88, 0.8, 1] }), 77);
    assert.equal(t.engine.createTexture({ width: 2, height: 3, clear: new Float32Array([0, 0.5, 1, 0.25]) }, { srgb: false }), 77);
    assert.deepEqual(t.calls, [['clear', 512, 717, 1, 0.9, 0.88, 0.8, 1], ['clear', 2, 3, 0, 0, 0.5, 1, 0.25]]);
    for (const bad of [
      { width: 0, height: 4, clear: [0, 0, 0, 1] },
      { width: 4.5, height: 4, clear: [0, 0, 0, 1] },
      { width: 4, height: 4, clear: [0, 0, 1] },
      { width: 4, height: 4, clear: [0, 0, Number.NaN, 1] },
      { width: 4, height: 4, clear: 'red!' },
    ]) {
      assert.throws(() => t.engine.createTexture(bad), /a cleared texture takes/, JSON.stringify(bad));
    }
    assert.equal(t.calls.length, 2);
    assert.equal(t.readbacks.length, 0);
  });
});

test('the WebGPU source copy checks the slot and the size before copying', async () => {
  const copies = [];
  const device = new Proxy({
    lost: new Promise(() => {}),
    queue: { copyExternalImageToTexture: (...args) => copies.push(args) },
    createTexture: (desc) => ({ desc, destroy() {} }),
  }, { get: (target, prop) => (prop in target ? target[prop] : () => ({})) });
  const host = await withGlobals({
    GPUShaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 },
    GPUBufferUsage: { UNIFORM: 0x40, COPY_DST: 0x08 },
  }, () => createHost({ api: 'webgpu', device, canvas: null, context: null }));
  const memory = new WebAssembly.Memory({ initial: 1 });
  host.attach({ exports: { memory } });
  const g = host.imports.ganbate;
  const lastError = () => new TextDecoder().decode(new Uint8Array(memory.buffer, 0, g.gpu_last_error(0, 256)));
  const tex = g.gpu_create_texture(4, 2, 1, 3, 1, 0);
  assert.equal(g.gpu_copy_source(tex, 0, 0, 4, 2), 0);
  assert.match(lastError(), /copyExternalImageToTexture: no source image is set/);
  host.setSource(canvasSource(4, 2));
  assert.equal(g.gpu_copy_source(tex, 1, 0, 2, 1), 0);
  assert.match(lastError(), /the source image is 4x2, not 2x1/);
  assert.equal(copies.length, 0);
  assert.equal(g.gpu_copy_source(tex, 0, 0, 4, 2), 1);
  assert.equal(copies.length, 1);
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

const GRAPH = {
  schema: 'ganbate.semantic.graph.v2',
  nodes: [
    { id: 'desk', type: 'desk', position: [0, 0, 0], bounds: { shape: 'aabb', halfExtents: [1.2, 0.38, 0.6], surfaceY: 0.76 } },
    { id: 'cup', type: 'cup', position: [2, 0, 2], bounds: { shape: 'cylinder', radius: 0.1, height: 0.15 } },
  ],
  edges: [],
};

test('graphCommit hands graph and patch to the engine as JSON and returns or throws its outcome', async () => {
  const { canvas } = makeRecordingWebGl2Canvas();
  const seen = [];
  let outcome = '';
  const text = (memory, p, len) => new TextDecoder().decode(new Uint8Array(memory.buffer, p >>> 0, len >>> 0));
  const mock = makeMockWasm({
    ganbate_graph_commit(gp, gl, pp, pl, reject) {
      seen.push({ graph: JSON.parse(text(mock.memory, gp, gl)), patch: JSON.parse(text(mock.memory, pp, pl)), reject });
      outcome = JSON.stringify(reject ? { error: 'refused' } : { graph: GRAPH, digest: 'ab', repairs: ['moved'] });
      return new TextEncoder().encode(outcome).byteLength;
    },
    ganbate_graph_result(out, cap) {
      const bytes = new TextEncoder().encode(outcome);
      if (bytes.byteLength <= cap) new Uint8Array(mock.memory.buffer, out >>> 0, bytes.byteLength).set(bytes);
      return bytes.byteLength;
    },
  });
  const engine = await createEngine({ canvas, wasm: mock.instance, api: 'webgl2' });
  const patch = [{ op: 'place', id: 'cup', rel: 'on_top_of', to: 'desk' }];
  assert.deepEqual(engine.graphCommit(GRAPH, patch), { graph: GRAPH, digest: 'ab', repairs: ['moved'] });
  assert.deepEqual(seen[0], { graph: GRAPH, patch, reject: 0 });
  assert.throws(() => engine.graphCommit(JSON.stringify(GRAPH), '[]', 'reject'), /graphCommit: refused/);
  assert.equal(seen[1].reject, 1);
});

test('the compiled ganbate_web.wasm solves a graph patch', () => {
  const wasmPath = path.join(REPO_ROOT, 'target', 'wasm32-unknown-unknown', 'release', 'ganbate_web.wasm');
  if (!fs.existsSync(wasmPath)) return;
  const mod = new WebAssembly.Module(fs.readFileSync(wasmPath));
  const imports = {};
  for (const i of WebAssembly.Module.imports(mod)) (imports[i.module] ??= {})[i.name] = () => 0;
  const x = new WebAssembly.Instance(mod, imports).exports;
  // Calls an export taking two JSON texts and the mode, and reads back its outcome.
  const call = (name, first, second, reject) => {
    const [a, b] = [first, second].map((v) => new TextEncoder().encode(v === null ? '' : JSON.stringify(v)));
    const ap = x.ganbate_alloc(Math.max(a.byteLength, 1));
    const bp = x.ganbate_alloc(Math.max(b.byteLength, 1));
    new Uint8Array(x.memory.buffer, ap, a.byteLength).set(a);
    new Uint8Array(x.memory.buffer, bp, b.byteLength).set(b);
    const len = x[name](ap, a.byteLength, bp, b.byteLength, reject);
    const out = x.ganbate_alloc(len);
    x.ganbate_graph_result(out, len);
    return JSON.parse(new TextDecoder().decode(new Uint8Array(x.memory.buffer, out, len)));
  };
  const commit = (patch, reject) => call('ganbate_graph_commit', GRAPH, patch, reject);
  const placed = commit([{ op: 'place', id: 'cup', rel: 'on_top_of', to: 'desk' }], 0);
  assert.deepEqual(placed.graph.nodes[1].position, [0, 0.76, 0]);
  assert.match(placed.digest, /^[0-9a-f]{64}$/);
  const into = [{ op: 'set', id: 'cup', key: 'position', value: [0.5, 0, 0.7] }];
  assert.match(commit(into, 1).error, /overlapped "desk"/);
  assert.equal(commit(into, 0).repairs.length, 1);
  // A whole graph: opened as written, and checked again it is the same graph.
  const opened = call('ganbate_graph_check', null, placed.graph, 1);
  assert.equal(opened.digest, placed.digest);
  const floating = structuredClone(placed.graph);
  floating.nodes[1].position = [3, 1, 3];
  assert.match(call('ganbate_graph_check', placed.graph, floating, 1).error, /stood at height 1.00/);
});

test('createGraphSolver solves a graph without a canvas', async () => {
  const wasmPath = path.join(REPO_ROOT, 'target', 'wasm32-unknown-unknown', 'release', 'ganbate_web.wasm');
  if (!fs.existsSync(wasmPath)) return;
  const solver = await createGraphSolver({ wasm: fs.readFileSync(wasmPath) });
  const placed = solver.graphCommit(GRAPH, [{ op: 'place', id: 'cup', rel: 'on_top_of', to: 'desk' }]);
  assert.deepEqual(placed.graph.nodes[1].position, [0, 0.76, 0]);
  assert.equal(solver.graphCheck(placed.graph).digest, placed.digest);
  const floating = structuredClone(placed.graph);
  floating.nodes[1].position = [3, 1, 3];
  assert.throws(() => solver.graphCheck(floating, placed.graph, 'reject'), /^Error: graphCheck: .*stood at height 1.00/);
});

test('web/ganbate.d.ts and compiled ganbate_web.wasm expose Wave 8 ray-picking and WebXR contracts', () => {
  const dts = fs.readFileSync(path.join(__dirname, '..', 'sdk', 'ganbate.d.ts'), 'utf8');
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

test('both hosts provide every import of the compiled ganbate_web.wasm, which has the texture source and grade-once exports', async () => {
  const wasmPath = path.join(REPO_ROOT, 'target', 'wasm32-unknown-unknown', 'release', 'ganbate_web.wasm');
  if (!fs.existsSync(wasmPath)) return;
  const mod = new WebAssembly.Module(fs.readFileSync(wasmPath));
  const exported = new Set(WebAssembly.Module.exports(mod).map((e) => e.name));
  for (const name of ['ganbate_rt_texture_create_source', 'ganbate_rt_texture_update_source', 'ganbate_rt_texture_create_clear', 'ganbate_rt_set_grade_once']) {
    assert.ok(exported.has(name), `missing WASM export ${name}`);
  }
  const { gl, canvas } = makeRecordingWebGl2Canvas();
  const glHost = createGlHost({ api: 'webgl2', gl, ext: {}, canvas, adapterName: 'mock', features: [], format: 'rgba8unorm' });
  const gpuHost = await withGlobals({
    GPUShaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 },
    GPUBufferUsage: { UNIFORM: 0x40, COPY_DST: 0x08 },
  }, () => createHost({ api: 'webgpu', device: new Proxy({ lost: new Promise(() => {}) }, { get: (t, p) => (p in t ? t[p] : () => ({})) }), canvas: null, context: null }));
  const needed = WebAssembly.Module.imports(mod).filter((i) => i.module === 'ganbate').map((i) => i.name);
  assert.ok(needed.includes('gpu_copy_source') && needed.includes('gl_write_texture_source'));
  for (const host of [glHost, gpuHost]) {
    for (const name of needed) assert.equal(typeof host.imports.ganbate[name], 'function', `host lacks ${name}`);
  }
});

/** createEngine on a mock WebGPU device whose frames draw into the canvas, with GPU completion by hand. */
async function makePresentEngine(present) {
  const t = { log: [], done: [], rafs: [], canvasTextures: 0 };
  const anything = () => new Proxy(function () {}, {
    get: (_, prop) => (prop === 'then' ? undefined : anything()),
    apply: () => anything(),
  });
  const texture = (width, height) => ({ width, height, createView: anything, destroy() {} });
  const device = new Proxy({
    lost: new Promise(() => {}),
    queue: {
      submit: () => t.log.push(`submit (${t.done.length} unfinished)`),
      writeBuffer() {},
      onSubmittedWorkDone: () => new Promise((resolve) => t.done.push(resolve)),
    },
    createTexture: ({ size }) => texture(size[0], size[1]),
  }, { get: (target, prop) => (prop === 'then' ? undefined : prop in target ? target[prop] : anything()) });
  const canvas = {
    width: 16,
    height: 16,
    clientWidth: 16,
    clientHeight: 16,
    getContext: () => ({
      configure() {},
      getCurrentTexture: () => {
        t.canvasTextures++;
        t.log.push(`canvas texture (${t.done.length} unfinished)`);
        return texture(canvas.width, canvas.height);
      },
    }),
  };
  const globals = {
    GPUTextureUsage: { RENDER_ATTACHMENT: 0x10, COPY_SRC: 0x01, TEXTURE_BINDING: 0x04 },
    GPUShaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 },
    GPUBufferUsage: { UNIFORM: 0x40, COPY_DST: 0x08 },
    requestAnimationFrame: (cb) => t.rafs.push(cb),
    navigator: {
      gpu: {
        getPreferredCanvasFormat: () => 'bgra8unorm',
        requestAdapter: async () => ({
          features: new Set(),
          limits: { maxBufferSize: 65536, maxStorageBufferBindingSize: 65536, maxTextureDimension2D: 2048 },
          info: {},
          requestDevice: async () => device,
        }),
      },
    },
  };
  let imports = null;
  // A frame: one pass clearing the canvas (BEGIN_RENDER, the surface, clear and store; END_RENDER).
  const mock = makeMockWasm({
    ganbate_rt_render() {
      const at = 4096;
      new Uint32Array(mock.memory.buffer, at, 12).set([1, 1, 0, 0xFFFFFFFF, 0, 0, 1, 1, 0, 0, 0, 0, 2].slice(0, 12));
      new Uint32Array(mock.memory.buffer, at + 48, 1)[0] = 2;
      imports.gpu_submit(at, 13, 0, 0);
      return 1;
    },
  });
  const instantiate = WebAssembly.instantiate;
  WebAssembly.instantiate = async (_bytes, given) => {
    imports = given.ganbate;
    return { instance: mock.instance };
  };
  try {
    t.engine = await withGlobals(globals, () => createEngine({ canvas, wasm: new Uint8Array(8), api: 'webgpu', present }));
  } finally {
    WebAssembly.instantiate = instantiate;
  }
  t.canvas = canvas;
  t.run = (steps) => withGlobals(globals, steps);
  // Finishes the oldest frame on the GPU, as the queue reports it.
  t.finish = async () => {
    t.done.shift()();
    await new Promise((resolve) => setTimeout(resolve, 0));
  };
  return t;
}

test('settled WebGPU shows only finished frames, copied ahead of the next; direct keeps two in flight', async () => {
  const s = await makePresentEngine('settled');
  assert.equal(s.engine.present, 'settled');
  await s.run(async () => {
    assert.equal(s.engine.render(), true);
    // Drawn offscreen: the canvas is not touched while the frame is on the GPU, and no second
    // frame joins it.
    assert.equal(s.canvasTextures, 0);
    assert.equal(s.engine.render(), false);
    await s.finish();
    // A table at rest still shows it, at the next animation frame.
    assert.equal(s.rafs.length, 1);
    assert.equal(s.engine.render(), true);
    s.rafs.shift()();
    assert.deepEqual(s.log.slice(-3), [
      'canvas texture (0 unfinished)',
      'submit (0 unfinished)',
      'submit (0 unfinished)',
    ], 'the finished frame reaches the canvas before the next frame goes to the GPU');
    assert.equal(s.canvasTextures, 1);
    await s.finish();
    s.rafs.shift()();
    assert.equal(s.canvasTextures, 2);
    // A resize shows the last frame again in the new size.
    s.canvas.clientWidth = 20;
    s.engine.fitCanvas(1);
    assert.equal(s.canvasTextures, 3);
  });

  const d = await makePresentEngine('direct');
  assert.equal(d.engine.present, 'direct');
  await d.run(async () => {
    assert.equal(d.engine.render(), true);
    assert.equal(d.engine.render(), true);
    assert.equal(d.engine.render(), false);
    assert.equal(d.canvasTextures, 2, 'direct frames draw straight into the canvas');
    await d.finish();
    assert.equal(d.rafs.length, 0);
    assert.equal(d.engine.render(), true);
  });
});
