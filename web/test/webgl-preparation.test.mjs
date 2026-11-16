import test from 'node:test';
import assert from 'node:assert/strict';
import { createGlHost } from '../sdk/src/webgl-host.ts';

// Driver-independent ABI tests. Actual KHR/fence behavior still requires browser driver runs.
function setup({ parallel = true, linked = true } = {}) {
  let nextId = 0;
  let completed = false;
  let signaled = false;
  let lost = false;
  const calls = [];
  const fences = [];
  const deletedFences = new Map();
  const constants = {
    MAX_COMBINED_TEXTURE_IMAGE_UNITS: 1,
    COMPILE_STATUS: 2,
    LINK_STATUS: 3,
    INVALID_INDEX: 0xffffffff,
    COMPLETION_STATUS_KHR: 4,
    TIMEOUT_EXPIRED: 5,
    WAIT_FAILED: 6,
    CONDITION_SATISFIED: 7,
  };
  const target = {
    ...constants,
    getExtension: (name) => name === 'KHR_parallel_shader_compile' && parallel
      ? { COMPLETION_STATUS_KHR: constants.COMPLETION_STATUS_KHR } : null,
    getParameter: () => 16,
    getShaderParameter: () => { calls.push('compile-status'); return true; },
    getProgramParameter: (_program, parameter) => {
      if (parameter === constants.COMPLETION_STATUS_KHR) { calls.push('completion-status'); return completed; }
      calls.push('link-status');
      return linked;
    },
    getProgramInfoLog: () => 'synthetic invalid program',
    getUniformBlockIndex: () => constants.INVALID_INDEX,
    getUniformLocation: () => { calls.push('uniform-query'); return {}; },
    useProgram: () => calls.push('use-program'),
    createShader: () => ({ id: nextId++ }),
    createProgram: () => ({ id: nextId++ }),
    fenceSync: () => {
      calls.push('fence');
      const fence = { signaled: false };
      fences.push(fence);
      return fence;
    },
    clientWaitSync: (fence) => signaled || fence.signaled
      ? constants.CONDITION_SATISFIED : constants.TIMEOUT_EXPIRED,
    deleteSync: (fence) => {
      calls.push('delete-fence');
      deletedFences.set(fence, (deletedFences.get(fence) ?? 0) + 1);
    },
    deleteProgram: () => calls.push('delete-program'),
    deleteSampler: () => calls.push('delete-sampler'),
    deleteVertexArray: () => calls.push('delete-vao'),
    deleteBuffer: () => calls.push('delete-buffer'),
    deleteFramebuffer: () => calls.push('delete-fbo'),
    isContextLost: () => lost,
  };
  const gl = new Proxy(target, {
    get(object, key) {
      if (key in object) return object[key];
      if (/^[A-Z_0-9]+$/.test(key)) return 0;
      return () => ({});
    },
  });
  // Blit programs are compiled before the staged world path and must succeed.
  const linkedAtStart = linked;
  linked = true;
  const host = createGlHost({ gl, ext: {}, canvas: undefined }, { stagedPipelines: true });
  linked = linkedAtStart;
  const memory = new WebAssembly.Memory({ initial: 1 });
  host.attach({ exports: { memory } });
  calls.length = 0;
  const source = new TextEncoder().encode('void main() {}');
  new Uint8Array(memory.buffer, 16, source.length).set(source);
  const metadata = new TextEncoder().encode('u texture 0');
  new Uint8Array(memory.buffer, 64, metadata.length).set(metadata);
  const create = () => host.imports.ganbate.gl_create_program(16, source.length, 16, source.length, 64, metadata.length);
  return {
    host, calls, create, fences, deletedFences,
    complete() { completed = true; },
    signal() { signaled = true; },
    signalFence(index) { fences[index].signaled = true; },
    lose() { lost = true; },
  };
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 10));
const poll = async (t) => {
  t.mock.timers.tick(4);
  await Promise.resolve();
  await Promise.resolve();
};

test('staged program returns a numeric pending handle without blocking queries or use', async () => {
  const h = setup();
  const handle = h.create();
  assert.ok(handle > 0);
  assert.deepEqual(h.calls, []);
  assert.equal(h.host.preparationStats().pipelinesPending, 1);
  assert.equal(h.host.imports.ganbate.gl_submit(0, 0, 0, 0, 0), 0);
  let ready = false;
  const prepared = h.host.prepare().then(() => { ready = true; });
  await tick();
  assert.deepEqual(new Set(h.calls), new Set(['completion-status']));
  h.complete();
  await tick();
  assert.equal(h.host.preparationStats().pipelinesPending, 0);
  assert.equal(h.host.preparationStats().pipelinesReady, 1);
  assert.equal(ready, false, 'link completion alone is not GPU completion');
  h.signal();
  await prepared;
  assert.ok(h.calls.includes('uniform-query'));
  assert.ok(h.calls.includes('delete-fence'));
});

test('program failure rejects preparation and never initializes uniforms', async () => {
  const h = setup({ linked: false });
  h.create();
  h.complete();
  await assert.rejects(h.host.prepare(), /synthetic invalid program/);
  assert.ok(!h.calls.includes('uniform-query'));
  assert.ok(!h.calls.includes('fence'));
});

test('preparation cancellation leaves shared pending work available for another consumer', async () => {
  const h = setup();
  h.create();
  const controller = new AbortController();
  const first = h.host.prepare({ signal: controller.signal });
  controller.abort();
  await assert.rejects(first, { name: 'AbortError' });
  assert.equal(h.host.preparationStats().pipelinesPending, 1);
  h.complete();
  h.signal();
  await h.host.prepare();
});

test('context loss rejects the readiness barrier', async () => {
  const h = setup();
  h.create();
  h.complete();
  const pending = h.host.prepare();
  h.lose();
  await assert.rejects(pending, /context lost/);
  assert.equal(await h.host.lost, 'WebGL 2 context lost');
  assert.ok(h.calls.includes('delete-fence'));
});

test('extension absence uses the supported synchronous compile path and fences completion', async () => {
  const h = setup({ parallel: false });
  assert.ok(h.create() > 0);
  assert.ok(h.calls.includes('compile-status'));
  assert.ok(h.calls.includes('link-status'));
  assert.equal(h.host.preparationStats().pipelinesPending, 0);
  h.signal();
  await h.host.prepare();
  assert.ok(h.calls.includes('fence'));
});

test('released pending program cannot be initialized by a late completion', async () => {
  const h = setup();
  const handle = h.create();
  h.host.imports.ganbate.gl_release(handle);
  h.complete();
  h.signal();
  await h.host.prepare();
  assert.equal(h.host.preparationStats().pipelinesReady, 0);
  assert.ok(!h.calls.includes('uniform-query'));
});


test('release deletes pending programs and host helpers once after failed startup', () => {
  const h = setup();
  h.create();
  h.host.release();
  assert.equal(h.calls.filter((call) => call === 'delete-program').length, 3);
  assert.equal(h.calls.filter((call) => call === 'delete-sampler').length, 2);
  assert.equal(h.calls.filter((call) => call === 'delete-vao').length, 1);
  assert.equal(h.calls.filter((call) => call === 'delete-buffer').length, 1);
  assert.equal(h.calls.filter((call) => call === 'delete-fbo').length, 1);
  const releasedCalls = h.calls.length;
  h.host.release();
  assert.equal(h.calls.length, releasedCalls);
  assert.equal(h.host.preparationStats().pipelinesPending, 0);
});

test('preparation includes a program created while its first fence is pending', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const h = setup();
  let ready = false;
  const prepared = h.host.prepare().then(() => { ready = true; });
  try {
    assert.equal(h.fences.length, 1);
    h.create();
    h.signalFence(0);
    await poll(t);
    assert.equal(ready, false, 'a preceding fence cannot complete a newly created program');
    assert.equal(h.host.preparationStats().pipelinesPending, 1);
    h.complete();
    await poll(t);
    assert.equal(h.host.preparationStats().pipelinesPending, 0);
    assert.equal(h.fences.length, 2, 'new initialization needs its own completion fence');
    assert.equal(ready, false);
    h.signalFence(1);
    await poll(t);
    await prepared;
    assert.deepEqual([...h.deletedFences.values()], [1, 1]);
  } finally {
    h.complete();
    h.signal();
    await poll(t);
    await prepared;
    h.host.release();
  }
});

test('a concurrent consumer cannot hide a new program from an older preparation fence', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const h = setup();
  let firstReady = false;
  const first = h.host.prepare().then(() => { firstReady = true; });
  h.create();
  h.complete();
  const second = h.host.prepare();
  try {
    assert.equal(h.host.preparationStats().pipelinesPending, 0, 'second consumer drained the program');
    assert.equal(h.fences.length, 2);
    h.signalFence(1);
    await poll(t);
    await second;
    assert.equal(firstReady, false);
    h.signalFence(0);
    await poll(t);
    assert.equal(firstReady, false, 'the generation changed even though no programs remain pending');
    assert.equal(h.fences.length, 3);
    h.signalFence(2);
    await poll(t);
    await first;
    assert.deepEqual([...h.deletedFences.values()], [1, 1, 1]);
  } finally {
    h.signal();
    await poll(t);
    await Promise.all([first, second]);
    h.host.release();
  }
});

test('synchronous fallback programs also invalidate an older preparation fence', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const h = setup({ parallel: false });
  let ready = false;
  const prepared = h.host.prepare().then(() => { ready = true; });
  try {
    h.create();
    assert.equal(h.host.preparationStats().pipelinesPending, 0);
    h.signalFence(0);
    await poll(t);
    assert.equal(ready, false, 'synchronous linking still happened after the original fence');
    assert.equal(h.fences.length, 2);
    h.signalFence(1);
    await poll(t);
    await prepared;
    assert.deepEqual([...h.deletedFences.values()], [1, 1]);
  } finally {
    h.signal();
    await poll(t);
    await prepared;
    h.host.release();
  }
});

test('cancelling a fence waiter leaves another consumer and its fence intact', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const h = setup();
  const controller = new AbortController();
  const first = h.host.prepare({ signal: controller.signal });
  const cancelled = assert.rejects(first, { name: 'AbortError' });
  h.create();
  h.complete();
  let secondReady = false;
  const second = h.host.prepare().then(() => { secondReady = true; });
  try {
    controller.abort();
    await poll(t);
    await cancelled;
    assert.equal(h.deletedFences.get(h.fences[0]), 1);
    assert.equal(h.deletedFences.has(h.fences[1]), false);
    assert.equal(secondReady, false);
    h.signalFence(1);
    await poll(t);
    await second;
    assert.deepEqual([...h.deletedFences.values()], [1, 1]);
    assert.equal(h.host.preparationStats().pipelinesReady, 1);
  } finally {
    controller.abort();
    h.signal();
    await poll(t);
    await Promise.all([cancelled, second]);
    h.host.release();
  }
});

test('ordinary frame submissions do not extend an unchanged preparation boundary', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const h = setup();
  const prepared = h.host.prepare();
  try {
    assert.equal(h.host.imports.ganbate.gl_submit(0, 0, 0, 0, 0), 1);
    h.signalFence(0);
    await poll(t);
    await prepared;
    assert.equal(h.fences.length, 1);
    assert.deepEqual([...h.deletedFences.values()], [1]);
  } finally {
    h.signal();
    await poll(t);
    await prepared;
    h.host.release();
  }
});
