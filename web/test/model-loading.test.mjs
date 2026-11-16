import test from 'node:test';
import assert from 'node:assert/strict';
import { createEngine } from '../sdk/src/engine.ts';

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  return { promise, resolve, reject };
};
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

async function fixture(t, { images = ['image/webp'], badDescription = false, failCreate = false, failUpload = false } = {}) {
  const saved = Object.getOwnPropertyDescriptor(globalThis, 'createImageBitmap');
  t.after(() => saved ? Object.defineProperty(globalThis, 'createImageBitmap', saved) : delete globalThis.createImageBitmap);
  const decodes = [], bitmaps = [], warnings = [];
  globalThis.createImageBitmap = () => {
    const decode = deferred();
    decodes.push(decode);
    return decode.promise;
  };
  const bitmap = () => {
    const image = { width: 1, height: 1, data: new Uint8Array([1, 2, 3, 255]), closed: 0, close() { this.closed++; } };
    bitmaps.push(image);
    return image;
  };
  const gl = new Proxy({
    getExtension: () => null, getParameter: () => 16, getShaderParameter: () => true,
    getProgramParameter: () => true, fenceSync: () => ({}), clientWaitSync: () => 1,
    ALREADY_SIGNALED: 1, isContextLost: () => false,
  }, { get(target, key) {
    if (key in target) return target[key];
    if (/^[A-Z_0-9]+$/.test(String(key))) return 0;
    return () => ({});
  } });
  const canvas = Object.assign(new EventTarget(), { width: 16, height: 16, clientWidth: 16, clientHeight: 16, getContext: () => gl });
  const memory = new WebAssembly.Memory({ initial: 4 });
  let bump = 1024, nextTexture = 1, nextModel = 1, parseCalls = 0, uploads = 0, stops = 0;
  const pending = new Set(), textures = new Set(), models = new Set(), discards = [], destroyedTextures = [], creates = [];
  const give = (bytes, at, cap) => {
    if (at && cap >= bytes.length) new Uint8Array(memory.buffer, at, bytes.length).set(bytes);
    return bytes.length;
  };
  const createTexture = () => {
    uploads++;
    if (failUpload) throw new Error('upload failed');
    const id = nextTexture++;
    textures.add(id);
    return id;
  };
  const x = {
    memory,
    ganbate_alloc(len) { const at = bump; bump += (len + 7) & ~7; return at; },
    ganbate_free() {}, ganbate_rt_start: () => 1,
    ganbate_rt_stop() { stops++; pending.clear(); textures.clear(); models.clear(); },
    ganbate_rt_model_parse() {
      parseCalls++;
      let id = 1; while (pending.has(id)) id++;
      pending.add(id); return id;
    },
    ganbate_rt_model_describe(id, at, cap) {
      assert.ok(pending.has(id));
      return give(new TextEncoder().encode(badDescription ? '{' : JSON.stringify({ images: images.map((mime) => ({ mime, srgb: true })) })), at, cap);
    },
    ganbate_rt_model_image(id, index, at, cap) { assert.ok(pending.has(id)); return give(new Uint8Array([index]), at, cap); },
    ganbate_rt_model_discard(id) { discards.push(id); pending.delete(id); },
    ganbate_rt_model_create(id, at, count) {
      assert.ok(pending.delete(id), 'parsed handle is consumed exactly once');
      creates.push([...new Uint32Array(memory.buffer, at, count)]);
      if (failCreate) return 0;
      const model = nextModel++; models.add(model); return model;
    },
    ganbate_rt_model_destroy(id) { models.delete(id); },
    ganbate_rt_texture_create: createTexture,
    ganbate_rt_texture_png: createTexture,
    ganbate_rt_texture_destroy(id) { assert.ok(textures.delete(id), 'texture released once'); destroyedTextures.push(id); },
  };
  const engine = await createEngine({ canvas, wasm: { exports: x }, api: 'webgl2', onLog: (_level, text) => warnings.push(text) });
  t.after(() => engine.destroy());
  return { engine, x, canvas, bitmap, bitmaps, decodes, warnings, pending, textures, models, discards, destroyedTextures, creates,
    counts: () => ({ parseCalls, uploads, stops }) };
}

test('cancelling a model closes late decodes and discards only its own parsed handle', { timeout: 1000 }, async (t) => {
  const f = await fixture(t);
  const controller = new AbortController();
  const first = f.engine.loadModel(new Uint8Array([1]), { signal: controller.signal });
  const rejected = assert.rejects(first, { name: 'AbortError' });
  const retained = f.engine.loadModel(new Uint8Array([2]));
  controller.abort();
  await rejected;
  assert.deepEqual([...f.pending], [2]);
  assert.deepEqual(f.discards, [1]);
  f.decodes[0].resolve(f.bitmap());
  f.decodes[1].resolve(f.bitmap());
  const result = await retained;
  assert.deepEqual(result.textures, [1]);
  assert.deepEqual([...f.textures], [1]);
  assert.ok(f.bitmaps.every((image) => image.closed === 1));
  assert.equal(f.pending.size, 0);
  assert.equal(f.models.size, 1);
});

test('destroy rejects pending decode promptly and late completion cannot upload or discard reused runtime slots', { timeout: 1000 }, async (t) => {
  const f = await fixture(t);
  const pending = f.engine.loadModel(new Uint8Array([1]));
  const rejected = assert.rejects(pending, /destroyed/);
  f.engine.destroy();
  await rejected;
  // Simulate a later session reusing the underlying WASM instance's numeric slots.
  f.pending.add(1);
  f.decodes[0].resolve(f.bitmap());
  await tick();
  assert.equal(f.counts().uploads, 0);
  assert.deepEqual(f.discards, []);
  assert.deepEqual([...f.pending], [1]);
  assert.equal(f.bitmaps[0].closed, 1);
  await assert.rejects(f.engine.loadModel(new Uint8Array([1])), /destroyed/);
  assert.equal(f.counts().parseCalls, 1);
});

test('context loss cancels model decode without uploading and releases pending CPU model ownership', { timeout: 1000 }, async (t) => {
  const f = await fixture(t);
  const pending = f.engine.loadModel(new Uint8Array([1]));
  const rejected = assert.rejects(pending, /context lost/);
  f.canvas.dispatchEvent(new Event('webglcontextlost', { cancelable: true }));
  await rejected;
  assert.equal(f.pending.size, 0);
  f.decodes[0].resolve(f.bitmap());
  await tick();
  assert.equal(f.counts().uploads, 0);
  assert.equal(f.bitmaps[0].closed, 1);
});

test('failed model creation releases uploaded textures without discarding its spent parse slot', async (t) => {
  const f = await fixture(t, { images: ['image/png', 'image/webp'], failCreate: true });
  const pending = f.engine.loadModel(new Uint8Array([1]));
  f.decodes[0].resolve(f.bitmap());
  await assert.rejects(pending, /loadModel/);
  assert.equal(f.pending.size, 0);
  assert.equal(f.textures.size, 0);
  assert.equal(f.models.size, 0);
  assert.deepEqual(f.destroyedTextures.sort(), [1, 2]);
  assert.deepEqual(f.discards, []);
  assert.equal(f.bitmaps[0].closed, 1);
});

test('image upload and decode failures retain optional-image fallback and always close decoded bitmaps', async (t) => {
  const f = await fixture(t, { failUpload: true, images: ['image/webp', 'image/jpeg'] });
  const pending = f.engine.loadModel(new Uint8Array([1]));
  f.decodes[0].resolve(f.bitmap());
  f.decodes[1].reject(new Error('decoder failed'));
  const result = await pending;
  assert.deepEqual(result.textures, [0, 0]);
  assert.equal(f.bitmaps[0].closed, 1);
  assert.equal(f.textures.size, 0);
  assert.ok(f.warnings.some((text) => text.includes('upload failed')));
  assert.ok(f.warnings.some((text) => text.includes('decoder failed')));
});

test('invalid model metadata discards its pending parse and a retry can reuse the slot', async (t) => {
  const f = await fixture(t, { badDescription: true });
  for (let attempt = 0; attempt < 2; attempt++) {
    await assert.rejects(f.engine.loadModel(new Uint8Array([1])), SyntaxError);
    assert.equal(f.pending.size, 0);
    assert.equal(f.textures.size, 0);
  }
  assert.deepEqual(f.discards, [1, 1]);
});

test('pre-aborted model load performs no fetch, parse or GPU work', async (t) => {
  const f = await fixture(t);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(f.engine.loadModel('/unused.glb', { signal: controller.signal }), { name: 'AbortError' });
  assert.equal(f.counts().parseCalls, 0);
  assert.equal(f.counts().uploads, 0);
});

test('cancelling after one upload releases it while other browser decodes finish late', { timeout: 1000 }, async (t) => {
  const f = await fixture(t, { images: ['image/png', 'image/webp'] });
  const controller = new AbortController();
  const pending = f.engine.loadModel(new Uint8Array([1]), { signal: controller.signal });
  const rejected = assert.rejects(pending, { name: 'AbortError' });
  assert.equal(f.textures.size, 1);
  controller.abort(); await rejected;
  assert.equal(f.textures.size, 0);
  assert.equal(f.pending.size, 0);
  f.decodes[0].resolve(f.bitmap()); await tick();
  assert.equal(f.counts().uploads, 1);
  assert.equal(f.bitmaps[0].closed, 1);
});

test('cancellation between browser decode fulfillment and its awaiting continuation still closes the bitmap', async (t) => {
  const f = await fixture(t);
  const controller = new AbortController();
  const pending = f.engine.loadModel(new Uint8Array([1]), { signal: controller.signal });
  const rejected = assert.rejects(pending, { name: 'AbortError' });
  f.decodes[0].resolve(f.bitmap());
  queueMicrotask(() => controller.abort());
  await rejected;
  assert.equal(f.bitmaps[0].closed, 1);
  assert.equal(f.counts().uploads, 0);
  assert.equal(f.pending.size, 0);
});

test('fetch cancellation settles promptly even if the response ignores abort', { timeout: 1000 }, async (t) => {
  const f = await fixture(t);
  const saved = globalThis.fetch;
  const response = deferred();
  let requestedSignal;
  globalThis.fetch = (_url, { signal }) => { requestedSignal = signal; return response.promise; };
  t.after(() => { globalThis.fetch = saved; });
  const controller = new AbortController();
  const pending = f.engine.loadModel('/delayed.glb', { signal: controller.signal });
  const rejected = assert.rejects(pending, { name: 'AbortError' });
  controller.abort(); await rejected;
  assert.equal(requestedSignal.aborted, true);
  response.resolve(new Response(new Uint8Array([1]))); await tick();
  assert.equal(f.counts().parseCalls, 0);
});

test('fatal native image upload aborts sibling decoders without waiting for browser completion', { timeout: 1000 }, async (t) => {
  const f = await fixture(t, { images: ['image/png', 'image/webp'], failUpload: true });
  await assert.rejects(f.engine.loadModel(new Uint8Array([1])), /upload failed/);
  assert.equal(f.pending.size, 0);
  assert.equal(f.textures.size, 0);
  f.decodes[0].resolve(f.bitmap()); await tick();
  assert.equal(f.bitmaps[0].closed, 1);
  assert.equal(f.counts().uploads, 1);
});

test('an unmatched WASM build fails before acquiring a parsed model', async (t) => {
  const f = await fixture(t);
  delete f.x.ganbate_rt_model_discard;
  await assert.rejects(f.engine.loadModel(new Uint8Array([1])), /matching SDK and WASM/);
  assert.equal(f.counts().parseCalls, 0);
});
