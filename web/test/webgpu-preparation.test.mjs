import test from 'node:test';
import assert from 'node:assert/strict';
import { createHost } from '../sdk/src/webgpu-host.ts';

const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };

function fixture({ independent = false } = {}) {
  globalThis.GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
  globalThis.GPUBufferUsage = { UNIFORM: 64, COPY_DST: 8 };
  const compilation = deferred();
  const completion = deferred();
  const deviceLoss = deferred();
  const calls = [];
  const destroyed = [];
  const errors = [];
  const compilations = [], completions = [];
  const shaderChecks = [];
  let delayShaderChecks = false;
  let completionWaits = 0;
  const device = {
    lost: deviceLoss.promise,
    addEventListener() {},
    createBindGroupLayout: () => ({}),
    createBindGroup: () => ({}),
    createPipelineLayout: () => ({}),
    createSampler: () => ({}),
    createBuffer: () => { const buffer = { destroy() { destroyed.push(buffer); } }; return buffer; },
    createShaderModule: () => ({ getCompilationInfo: () => {
      if (!delayShaderChecks) return Promise.resolve({ messages: [] });
      const check = deferred(); shaderChecks.push(check); return check.promise;
    } }),
    createRenderPipeline() { throw new Error('Unexpected synchronous pipeline'); },
    createRenderPipelineAsync(descriptor) {
      calls.push(descriptor);
      const next = independent ? deferred() : compilation;
      compilations.push(next);
      return next.promise;
    },
    queue: { onSubmittedWorkDone: () => {
      completionWaits++;
      const next = independent ? deferred() : completion;
      completions.push(next);
      return next.promise;
    } },
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
  return { host, g, compilation, completion, compilations, completions, shaderChecks, delayShaders: () => { delayShaderChecks = true; }, deviceLoss, calls, descriptor, create, destroyed, errors, completionWaits: () => completionWaits };
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

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

test('preparation includes a new pipeline created while an earlier GPU completion is pending', async (t) => {
  const f = fixture({ independent: true });
  t.after(() => f.host.release());
  f.create();
  let ready = false;
  const work = f.host.prepare().then(() => { ready = true; });
  work.catch(() => {});
  f.compilations[0].resolve({}); await tick();
  assert.equal(f.completions.length, 1);
  f.create({ ...f.descriptor, primitive: { topology: 'line-list' } });
  f.completions[0].resolve(); await tick();
  assert.equal(ready, false, 'the first fence cannot declare a new pending pipeline ready');
  assert.equal(f.host.preparationStats().pipelinesPending, 1);
  f.compilations[1].resolve({}); await tick();
  assert.equal(f.completions.length, 2, 'new pipeline generation needs a subsequent completion boundary');
  f.completions[1].resolve(); await work;
  assert.equal(f.host.preparationStats().pipelinesPending, 0);
});

test('preparation refences when new pipeline work already settled during the old queue wait', async (t) => {
  const f = fixture({ independent: true });
  t.after(() => f.host.release());
  f.create();
  let ready = false;
  const work = f.host.prepare().then(() => { ready = true; }); work.catch(() => {});
  f.compilations[0].resolve({}); await tick();
  f.create({ ...f.descriptor, primitive: { topology: 'line-list' } });
  f.compilations[1].resolve({}); await tick();
  assert.equal(f.host.preparationStats().pipelinesPending, 0);
  f.completions[0].resolve(); await tick();
  assert.equal(ready, false, 'zero pending jobs does not prove the old fence covers their generation');
  assert.equal(f.completions.length, 2);
  f.completions[1].resolve(); await work;
});

test('a cached pipeline handle during completion does not restart preparation', async (t) => {
  const f = fixture({ independent: true });
  t.after(() => f.host.release());
  f.create();
  const work = f.host.prepare();
  f.compilations[0].resolve({}); await tick();
  f.create();
  f.completions[0].resolve(); await work;
  assert.equal(f.calls.length, 1);
  assert.equal(f.completions.length, 1);
});

test('a failed pipeline rejects preparation promptly while another driver compilation hangs', async (t) => {
  const f = fixture({ independent: true });
  t.after(() => f.host.release());
  f.create(); f.create({ ...f.descriptor, primitive: { topology: 'line-list' } });
  let outcome;
  const work = f.host.prepare().then(() => { outcome = 'ready'; }, (error) => { outcome = error.message; });
  f.compilations[0].reject(new Error('broken shader')); await tick();
  assert.match(outcome ?? '', /broken shader/);
  assert.equal(f.completions.length, 0);
  await work;
});

test('cancelled preparation does not issue an orphan queue wait when shared compilation completes', async (t) => {
  const f = fixture({ independent: true });
  t.after(() => f.host.release());
  f.create();
  const controller = new AbortController();
  const cancelled = assert.rejects(f.host.prepare({ signal: controller.signal }), { name: 'AbortError' });
  const retained = f.host.prepare(); retained.catch(() => {});
  controller.abort(); await cancelled;
  f.compilations[0].resolve({}); await tick();
  assert.equal(f.completions.length, 1, 'only the retained consumer starts a completion boundary');
  f.completions[0].resolve(); await retained;
});

test('device loss invalidates pending preparation and late pipeline results cannot revive statistics', async (t) => {
  const f = fixture({ independent: true });
  t.after(() => f.host.release());
  f.create();
  f.deviceLoss.resolve({ reason: 'unknown', message: 'removed during compilation' });
  await f.host.lost;
  const lost = f.host.preparationStats();
  assert.equal(lost.pipelinesPending, 0);
  f.compilations[0].resolve({}); await tick();
  assert.deepEqual(f.host.preparationStats(), lost);
  assert.equal(f.create({ ...f.descriptor, primitive: { topology: 'line-list' } }), 0);
  assert.equal(f.calls.length, 1);
});

test('shader validation added during queue completion also advances the preparation generation', async (t) => {
  const f = fixture({ independent: true });
  t.after(() => f.host.release());
  let ready = false;
  const work = f.host.prepare().then(() => { ready = true; }); work.catch(() => {});
  await tick();
  f.delayShaders(); f.g.gpu_create_shader(0, 0);
  f.shaderChecks[0].resolve({ messages: [] }); await tick();
  f.completions[0].resolve(); await tick();
  assert.equal(ready, false);
  assert.equal(f.completions.length, 2);
  f.completions[1].resolve(); await work;
});

test('shader failure wakes preparation even if a pipeline driver job never completes', async (t) => {
  const f = fixture({ independent: true });
  t.after(() => f.host.release());
  f.create(); f.delayShaders(); f.g.gpu_create_shader(0, 0);
  let outcome;
  const work = f.host.prepare().catch((error) => { outcome = error.message; });
  f.shaderChecks[0].resolve({ messages: [{ type: 'error', message: 'invalid shader' }] }); await tick();
  assert.match(outcome ?? '', /invalid shader/);
  await work;
  assert.equal(f.completions.length, 0);
});
