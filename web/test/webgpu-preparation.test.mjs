import test from 'node:test';
import assert from 'node:assert/strict';
import { createHost } from '../sdk/src/webgpu-host.ts';

const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };

function fixture() {
  globalThis.GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
  globalThis.GPUBufferUsage = { UNIFORM: 64, COPY_DST: 8 };
  const compilation = deferred();
  const completion = deferred();
  const deviceLoss = deferred();
  const calls = [];
  const destroyed = [];
  const errors = [];
  let completionWaits = 0;
  const device = {
    lost: deviceLoss.promise,
    addEventListener() {},
    createBindGroupLayout: () => ({}),
    createBindGroup: () => ({}),
    createPipelineLayout: () => ({}),
    createSampler: () => ({}),
    createBuffer: () => { const buffer = { destroy() { destroyed.push(buffer); } }; return buffer; },
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createRenderPipeline() { throw new Error('Unexpected synchronous pipeline'); },
    createRenderPipelineAsync(descriptor) { calls.push(descriptor); return compilation.promise; },
    queue: { onSubmittedWorkDone: () => { completionWaits++; return completion.promise; } },
  };
  const host = createHost({ api: 'webgpu', device, canvas: null, context: null }, { stagedPipelines: true, onError: (message) => errors.push(message) });
  const memory = new WebAssembly.Memory({ initial: 1 });
  host.attach({ exports: { memory } });
  const g = host.imports.ganbate;
  const shader = g.gpu_create_shader(0, 0);
  const layout = g.gpu_create_pipeline_layout(0, 0);
  const descriptor = { layout, vertex: { module: shader, entryPoint: 'vs' }, primitive: { topology: 'triangle-list' } };
  const create = (value = descriptor) => {
    const bytes = new TextEncoder().encode(JSON.stringify(value));
    new Uint8Array(memory.buffer, 128, bytes.length).set(bytes);
    return g.gpu_create_render_pipeline(128, bytes.length);
  };
  return { host, g, compilation, completion, deviceLoss, calls, descriptor, create, destroyed, errors, completionWaits: () => completionWaits };
}

test('WebGPU keeps numeric pending handles, deduplicates descriptors and waits GPU completion', async () => {
  const f = fixture();
  const first = f.create();
  assert.ok(Number.isInteger(first) && first > 0);
  const second = f.create();
  assert.notEqual(first, second, 'independently releasable resource handles');
  assert.equal(f.calls.length, 1);
  assert.equal(f.host.preparationStats().pipelinesPending, 1);
  assert.equal(f.g.gpu_submit(0, 0, 0, 0), 0, 'pending handles cannot submit');
  let ready = false;
  const preparation = f.host.prepare().then(() => { ready = true; });
  f.compilation.resolve({ label: 'ready' });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(ready, false, 'pipeline completion alone is not upload/draw completion');
  f.completion.resolve();
  await preparation;
  assert.equal(f.host.preparationStats().pipelinesReady, 1);
  f.g.gpu_release(first);
  assert.ok(f.create());
  assert.equal(f.calls.length, 1, 'cache stays available to another shared consumer');
});

test('WebGPU distinguishes descriptor variants', async () => {
  const f = fixture();
  f.create();
  f.create({ ...f.descriptor, primitive: { topology: 'line-list' } });
  assert.equal(f.calls.length, 2);
  f.compilation.resolve({}); f.completion.resolve();
  await f.host.prepare();
});

test('WebGPU rejects failed pipeline preparation and device loss', async () => {
  const f = fixture();
  f.create();
  const preparation = f.host.prepare();
  f.compilation.reject(new Error('invalid entry point'));
  await assert.rejects(preparation, /invalid entry point/);
  const lost = fixture();
  const work = lost.host.prepare();
  lost.deviceLoss.resolve({ reason: 'unknown', message: 'removed' });
  await assert.rejects(work, /device lost/);
  assert.match(await lost.host.lost, /removed/);
});

test('cancelling a preparation waiter preserves another consumer', async () => {
  const f = fixture(); f.create();
  const controller = new AbortController();
  const cancelled = f.host.prepare({ signal: controller.signal });
  const retained = f.host.prepare();
  controller.abort();
  await assert.rejects(cancelled, { name: 'AbortError' });
  f.compilation.resolve({}); f.completion.resolve();
  await retained;
});

test('releasing a WebGPU host rejects a pending barrier before the driver settles', { timeout: 1000 }, async () => {
  const f = fixture();
  f.create();
  const pending = assert.rejects(f.host.prepare(), /host was released/);
  f.host.release();
  await pending;
  assert.equal(f.host.preparationStats().pipelinesPending, 0);
  const stats = f.host.preparationStats();
  f.compilation.resolve({ label: 'late' });
  f.completion.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(f.host.preparationStats(), stats, 'late pipelines cannot restore released readiness');
  assert.equal(f.create(), 0, 'released imports cannot recreate a device resource');
});

test('WebGPU release is idempotent and suppresses late compilation errors', async () => {
  const f = fixture();
  f.create();
  f.host.release();
  const released = f.destroyed.length;
  assert.ok(released > 0, 'host push buffer released');
  f.host.release();
  assert.equal(f.destroyed.length, released);
  f.compilation.reject(new Error('driver rejected after teardown'));
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(f.errors, []);
});

test('a pre-aborted preparation does not start a GPU completion barrier', async () => {
  const f = fixture();
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(f.host.prepare({ signal: controller.signal }), { name: 'AbortError' });
  assert.equal(f.completionWaits(), 0);
  f.host.release();
});
