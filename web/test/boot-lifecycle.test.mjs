import test from 'node:test';
import assert from 'node:assert/strict';
import { boot } from '../sdk/src/wasm-boot.ts';
import { createGpu } from '../sdk/src/webgpu-host.ts';

// Synthetic platform ownership checks: these do not exercise a physical GPU driver.
function fixture(t, { configureError, delayedDevice = false } = {}) {
  const globals = ['navigator', 'GPUShaderStage', 'GPUBufferUsage', 'GPUTextureUsage', 'fetch'];
  const saved = new Map(globals.map((name) => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
  t.after(() => {
    for (const [name, descriptor] of saved) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
  });
  const live = new Set();
  const events = new Set();
  let deviceRequests = 0;
  let destroyedDevices = 0;
  let resolveDevice;
  const resource = () => {
    const value = { destroy() { assert.ok(live.delete(value), 'resource destroyed exactly once'); } };
    live.add(value);
    return value;
  };
  const device = {
    lost: new Promise(() => {}),
    addEventListener(_name, listener) { events.add(listener); },
    removeEventListener(_name, listener) { events.delete(listener); },
    createBindGroupLayout: () => ({}),
    createBindGroup: () => ({}),
    createPipelineLayout: () => ({}),
    createSampler: () => ({}),
    createShaderModule: () => ({}),
    createBuffer: resource,
    destroy() { destroyedDevices++; },
  };
  const adapter = {
    features: new Set(), limits: {},
    requestDevice() {
      deviceRequests++;
      return delayedDevice ? new Promise((resolve) => { resolveDevice = resolve; }) : Promise.resolve(device);
    },
  };
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: {
    gpu: { requestAdapter: async () => adapter, getPreferredCanvasFormat: () => 'rgba8unorm' },
  } });
  globalThis.GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
  globalThis.GPUBufferUsage = { UNIFORM: 64, COPY_DST: 8 };
  globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 16, COPY_SRC: 1 };
  const canvas = { getContext: () => ({ configure() { if (configureError) throw configureError; } }) };
  return {
    canvas, live, events,
    requests: () => deviceRequests,
    destroyed: () => destroyedDevices,
    finishDevice: () => resolveDevice(device),
  };
}

test('failed WASM downloads release WebGPU hosts and devices on every retry', async (t) => {
  const f = fixture(t);
  globalThis.fetch = async () => new Response('', { status: 503 });
  for (let attempt = 1; attempt <= 3; attempt++) {
    await assert.rejects(boot({ canvas: f.canvas, wasm: '/engine.wasm', api: 'webgpu' }), /HTTP 503/);
    assert.equal(f.live.size, 0);
    assert.equal(f.events.size, 0);
    assert.equal(f.destroyed(), attempt);
  }
});

test('WebGPU canvas configuration failure destroys the newly acquired device', async (t) => {
  const failure = new Error('invalid canvas configuration');
  const f = fixture(t, { configureError: failure });
  await assert.rejects(createGpu(f.canvas), (error) => error === failure);
  assert.equal(f.destroyed(), 1);
});

test('cancellation during device creation releases the device before WASM startup', async (t) => {
  const f = fixture(t, { delayedDevice: true });
  const controller = new AbortController();
  const pending = boot({ canvas: f.canvas, wasm: '/engine.wasm', api: 'webgpu', signal: controller.signal });
  const cancelled = assert.rejects(pending, { name: 'AbortError' });
  await new Promise((resolve) => setTimeout(resolve, 0));
  controller.abort();
  f.finishDevice();
  await cancelled;
  assert.equal(f.destroyed(), 1);
  assert.equal(f.live.size, 0);
});

test('pre-aborted boot does not request a device', async (t) => {
  const f = fixture(t);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(boot({ canvas: f.canvas, wasm: '/engine.wasm', api: 'webgpu', signal: controller.signal }), { name: 'AbortError' });
  assert.equal(f.requests(), 0);
});

test('cancellation during WASM instantiation releases the partial host', async (t) => {
  const f = fixture(t);
  const original = WebAssembly.instantiate;
  let finish;
  WebAssembly.instantiate = () => new Promise((resolve) => { finish = resolve; });
  t.after(() => { WebAssembly.instantiate = original; });
  const controller = new AbortController();
  const pending = boot({ canvas: f.canvas, wasm: new ArrayBuffer(0), api: 'webgpu', signal: controller.signal });
  const cancelled = assert.rejects(pending, { name: 'AbortError' });
  await new Promise((resolve) => setTimeout(resolve, 0));
  controller.abort();
  finish({ instance: { exports: { memory: new WebAssembly.Memory({ initial: 1 }) } } });
  await cancelled;
  assert.equal(f.live.size, 0);
  assert.equal(f.events.size, 0);
  assert.equal(f.destroyed(), 1);
});

test('failed WASM download releases WebGL helpers and context listeners', async (t) => {
  fixture(t);
  globalThis.fetch = async () => new Response('', { status: 503 });
  const live = new Set();
  const events = new Set();
  const gl = new Proxy({
    getExtension: () => null,
    getParameter: () => 16,
    getShaderParameter: () => true,
    getProgramParameter: () => true,
  }, { get(target, key) {
    if (key in target) return target[key];
    if (/^[A-Z_0-9]+$/.test(key)) return 0;
    if (key.startsWith('create')) return () => { const object = {}; live.add(object); return object; };
    if (key.startsWith('delete')) return (object) => live.delete(object);
    return () => {};
  } });
  const canvas = {
    getContext: () => gl,
    addEventListener(_name, listener) { events.add(listener); },
    removeEventListener(_name, listener) { events.delete(listener); },
  };
  await assert.rejects(boot({ canvas, wasm: '/engine.wasm', api: 'webgl2' }), /HTTP 503/);
  assert.equal(live.size, 0);
  assert.equal(events.size, 0);
});
