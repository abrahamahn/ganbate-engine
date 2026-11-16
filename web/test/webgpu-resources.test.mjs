import test from 'node:test';
import assert from 'node:assert/strict';
import { createHost } from '../sdk/src/webgpu-host.ts';
import { readPixels } from '../sdk/src/wasm-boot.ts';

const constants = {
  GPUShaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 },
  GPUBufferUsage: { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, QUERY_RESOLVE: 512 },
  GPUTextureUsage: { COPY_SRC: 1, TEXTURE_BINDING: 4, RENDER_ATTACHMENT: 16 },
  GPUMapMode: { READ: 1 },
};
const saved = new Map(Object.keys(constants).map((name) => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
Object.assign(globalThis, constants);
test.after(() => {
  for (const [name, descriptor] of saved) {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete globalThis[name];
  }
});

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

function fixture({ settled = false, timestamps = false } = {}) {
  const loss = deferred();
  const mapping = deferred();
  const buffers = [], textures = [], queries = [];
  let bufferAttempts = 0, failBuffer = 0, failTexture = false, failGroup = false;
  const canvas = { width: 8, height: 4 };
  let views = 0, externalTextures = 0;
  const texture = (descriptor) => ({
    width: descriptor.size[0], height: descriptor.size[1],
    depthOrArrayLayers: descriptor.size[2] ?? 1,
    mipLevelCount: descriptor.mipLevelCount ?? 1,
    destroyed: 0,
    createView() { views++; return {}; },
    destroy() { this.destroyed++; },
  });
  const pass = () => ({ end() {}, setPipeline() {}, setBindGroup() {}, draw() {} });
  const device = {
    lost: loss.promise,
    features: new Set(timestamps ? ['timestamp-query'] : []),
    addEventListener() {}, removeEventListener() {},
    createBindGroupLayout: () => ({}),
    createBindGroup() { if (failGroup) { failGroup = false; throw new Error('group failed'); } return {}; },
    createPipelineLayout: () => ({}),
    createSampler: () => ({}),
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createRenderPipeline: () => ({}),
    createBuffer(descriptor) {
      if (++bufferAttempts === failBuffer) throw new Error('buffer failed');
      const buffer = {
        size: descriptor.size, destroyed: 0, maps: 0, reads: 0, unmapped: 0,
        destroy() { this.destroyed++; },
        mapAsync() { this.maps++; return mapping.promise; },
        getMappedRange(_offset, size = descriptor.size) { this.reads++; return new ArrayBuffer(size); },
        unmap() { this.unmapped++; },
      };
      buffers.push(buffer);
      return buffer;
    },
    createTexture(descriptor) {
      if (failTexture) { failTexture = false; throw new Error('texture failed'); }
      const result = texture(descriptor);
      textures.push(result);
      return result;
    },
    createQuerySet() {
      const query = { destroyed: 0, destroy() { this.destroyed++; } };
      queries.push(query);
      return query;
    },
    createCommandEncoder: () => ({
      beginRenderPass: pass,
      resolveQuerySet() {}, copyBufferToBuffer() {}, copyTextureToBuffer() {},
      finish: () => ({}),
    }),
    queue: { writeBuffer() {}, submit() {}, onSubmittedWorkDone: async () => {} },
  };
  const context = { getCurrentTexture() { externalTextures++; return texture({ size: [canvas.width, canvas.height] }); } };
  const gpu = { api: 'webgpu', device, canvas, context, format: 'rgba8unorm' };
  const host = createHost(gpu, { settled });
  const memory = new WebAssembly.Memory({ initial: 4 });
  host.attach({ exports: { memory } });
  const g = host.imports.ganbate;
  const render = (target = 0xffffffff) => {
    const words = [1, 1, 0, target, 0, 0, 1, 1, 0, 0, 0, 0, 2];
    new Uint32Array(memory.buffer, 1024, words.length).set(words);
    return g.gpu_submit(1024, words.length, 0, 0);
  };
  return {
    host, gpu, g, render, memory, canvas, buffers, textures, queries, loss, mapping,
    views: () => views, externalTextures: () => externalTextures,
    failBufferAfter: (count) => { failBuffer = bufferAttempts + count; },
    failNextTexture: () => { failTexture = true; },
    failNextGroup: () => { failGroup = true; },
  };
}

test('WebGPU counts ABI allocations once and repeated handle reuse reaches a stable plateau', () => {
  const f = fixture();
  assert.equal(f.host.resourceStats().buffers.estimatedBytes, 65536, 'push ring is host owned');
  const first = f.g.gpu_create_buffer(128, 8);
  assert.equal(f.host.resourceStats().buffers.liveCount, 2);
  assert.equal(f.host.resourceStats().buffers.estimatedBytes, 65664);
  f.g.gpu_release(first);
  f.g.gpu_release(first);
  assert.equal(f.buffers[1].destroyed, 1);
  for (let index = 0; index < 50; index++) {
    const handle = f.g.gpu_create_buffer(256, 8);
    assert.equal(handle, first, 'ABI handle is recycled, underlying resource is new');
    f.g.gpu_release(handle);
    assert.equal(f.host.resourceStats().buffers.estimatedBytes, 65536);
  }
  const before = f.host.resourceStats();
  assert.equal(before.buffers.liveCount, 1);
  assert.equal(before.buffers.allocations, 52);
  assert.equal(before.buffers.peakCount, 2);
  f.host.release();
  const after = f.host.resourceStats();
  assert.equal(after.total.liveCount, 0);
  assert.equal(after.total.estimatedBytes, 0);
  assert.equal(after.total.allocations, before.total.allocations);
  assert.equal(after.total.peakEstimatedBytes, before.total.peakEstimatedBytes);
  f.host.release();
  assert.ok(f.buffers.every((buffer) => buffer.destroyed === 1));
});

test('WebGPU estimates array mip and compressed storage without counting cached views or groups', () => {
  const f = fixture();
  const rgba = f.g.gpu_create_texture(8, 4, 3, 3, 1, 16);
  const compressed = f.g.gpu_create_texture(8, 8, 1, 4, 12, 16);
  assert.equal(f.host.resourceStats().textures.estimatedBytes, 504 + 56);
  assert.equal(f.render(rgba), 1);
  assert.equal(f.render(rgba), 1);
  assert.equal(f.views(), 1, 'attachment view is cached');
  const layout = f.g.gpu_create_bind_group_layout(0, 0);
  const words = [0, 1, rgba, 0, 0, 1, 0, 1];
  new Uint32Array(f.memory.buffer, 128, words.length).set(words);
  const group = f.g.gpu_create_bind_group(layout, 128, words.length);
  assert.ok(group);
  assert.equal(f.host.resourceStats().textures.liveCount, 2);
  assert.equal(f.host.resourceStats().textures.allocations, 2);
  f.g.gpu_release(group);
  assert.equal(f.host.resourceStats().textures.estimatedBytes, 560);
  f.g.gpu_release(rgba);
  assert.equal(f.host.resourceStats().textures.estimatedBytes, 56);
  f.g.gpu_release(compressed);
  assert.equal(f.host.resourceStats().textures.estimatedBytes, 0);
  f.host.release();
});

test('WebGPU ring replacement records transient overlap and preserves the old ring on failure', () => {
  const f = fixture();
  f.failNextGroup();
  assert.equal(f.g.gpu_submit(0, 0, 0, 70000), 0);
  assert.equal(f.host.resourceStats().buffers.liveCount, 1);
  assert.equal(f.host.resourceStats().buffers.estimatedBytes, 65536);
  assert.equal(f.buffers[0].destroyed, 0, 'working ring survives failed replacement');
  assert.equal(f.buffers[1].destroyed, 1, 'unusable new ring is destroyed');
  assert.equal(f.g.gpu_submit(0, 0, 0, 70000), 1);
  const stats = f.host.resourceStats().buffers;
  assert.equal(stats.liveCount, 1);
  assert.equal(stats.estimatedBytes, 131072);
  assert.equal(stats.peakCount, 2);
  assert.equal(stats.peakEstimatedBytes, 65536 + 131072);
  assert.equal(stats.allocations, 3);
  f.host.release();
});

test('WebGPU thrown buffer and texture creation does not claim an allocation', () => {
  const f = fixture();
  const before = f.host.resourceStats();
  f.failBufferAfter(1);
  assert.equal(f.g.gpu_create_buffer(2048, 8), 0);
  f.failNextTexture();
  assert.equal(f.g.gpu_create_texture(8, 4, 1, 1, 1, 16), 0);
  assert.deepEqual(f.host.resourceStats(), before);
  f.host.release();
});

test('WebGPU retains resized settled textures until presentation and excludes swapchain images', () => {
  const f = fixture({ settled: true });
  assert.equal(f.render(), 1);
  const first = f.host.frameTexture();
  assert.equal(f.host.resourceStats().textures.estimatedBytes, 128);
  f.canvas.width = 16;
  f.host.nextFrame();
  assert.equal(f.render(), 1);
  const second = f.host.frameTexture();
  assert.equal(f.host.resourceStats().textures.liveCount, 2);
  assert.equal(f.host.resourceStats().textures.estimatedBytes, 384);
  assert.equal(first.destroyed, 0);
  f.host.present(first);
  assert.equal(f.host.resourceStats().textures.liveCount, 2, 'presenting an older frame retains newer state');
  f.host.present(second);
  assert.equal(f.host.resourceStats().textures.liveCount, 1);
  assert.equal(f.host.resourceStats().textures.estimatedBytes, 256);
  assert.equal(first.destroyed, 1);
  assert.equal(f.externalTextures(), 2);
  f.host.release();
  assert.equal(second.destroyed, 1);
  const direct = fixture();
  assert.equal(direct.render(), 1);
  assert.equal(direct.host.resourceStats().textures.allocations, 0, 'external canvas image is not owned');
  direct.host.release();
});

test('WebGPU profiling helpers remain resident until release and late readback cannot restore counters', async () => {
  const f = fixture({ timestamps: true });
  f.host.timing.set(true);
  assert.equal(f.render(), 1);
  assert.equal(f.host.resourceStats().buffers.liveCount, 3);
  assert.equal(f.host.resourceStats().buffers.estimatedBytes, 65536 + 4096);
  f.host.endFrame();
  assert.equal(f.buffers[2].maps, 1);
  f.host.release();
  const released = f.host.resourceStats();
  assert.equal(released.total.liveCount, 0);
  assert.equal(f.queries[0].destroyed, 1);
  assert.ok(f.buffers.every((buffer) => buffer.destroyed === 1));
  f.mapping.resolve();
  await Promise.resolve();
  assert.deepEqual(f.host.resourceStats(), released);
  assert.equal(f.host.timing.busy(), false);
});

test('WebGPU partial profiling allocation failure destroys its successful helper', () => {
  const f = fixture({ timestamps: true });
  f.host.timing.set(true);
  f.failBufferAfter(2);
  assert.equal(f.render(), 0);
  assert.equal(f.host.resourceStats().buffers.liveCount, 1);
  assert.equal(f.host.resourceStats().buffers.estimatedBytes, 65536);
  assert.equal(f.host.resourceStats().buffers.allocations, 2);
  assert.equal(f.buffers[1].destroyed, 1);
  assert.equal(f.queries[0].destroyed, 1);
  f.host.release();
});

test('WebGPU late timestamp mapping does not read an invalidated device buffer', async () => {
  const f = fixture({ timestamps: true });
  f.host.timing.set(true);
  assert.equal(f.render(), 1);
  f.host.endFrame();
  f.loss.resolve({ reason: 'unknown', message: 'removed during timestamp mapping' });
  await f.host.lost;
  f.mapping.resolve();
  await Promise.resolve();
  assert.equal(f.buffers[2].reads, 0);
  assert.equal(f.host.resourceStats().total.liveCount, 0);
  f.host.release();
});

test('WebGPU capture readback contributes aligned temporary bytes to the same host snapshot', async () => {
  const f = fixture();
  assert.equal(f.render(), 1);
  const capture = readPixels(f.gpu, f.host.frameTexture(), f.host);
  assert.equal(f.host.resourceStats().buffers.liveCount, 2);
  assert.equal(f.host.resourceStats().buffers.estimatedBytes, 65536 + 256 * 4);
  assert.equal(f.host.resourceStats().textures.liveCount, 0, 'capture still does not own the swapchain');
  f.mapping.resolve();
  const pixels = await capture;
  assert.equal(pixels.rgba.length, 8 * 4 * 4);
  assert.equal(f.buffers[1].destroyed, 1);
  assert.equal(f.host.resourceStats().buffers.liveCount, 1);
  assert.equal(f.host.resourceStats().buffers.estimatedBytes, 65536);
  assert.equal(f.host.resourceStats().buffers.peakEstimatedBytes, 65536 + 1024);
  f.host.release();
});

test('WebGPU device loss clears live estimates and keeps history without late resurrection', async () => {
  const f = fixture();
  const buffer = f.g.gpu_create_buffer(4096, 8);
  f.g.gpu_create_texture(8, 4, 1, 1, 1, 16);
  const before = f.host.resourceStats();
  f.loss.resolve({ reason: 'unknown', message: 'removed' });
  await f.host.lost;
  const lost = f.host.resourceStats();
  assert.equal(lost.total.liveCount, 0);
  assert.equal(lost.total.estimatedBytes, 0);
  assert.equal(lost.total.allocations, before.total.allocations);
  assert.equal(lost.total.peakEstimatedBytes, before.total.peakEstimatedBytes);
  f.g.gpu_release(buffer);
  f.g.gpu_create_buffer(1024, 8);
  assert.deepEqual(f.host.resourceStats(), lost);
  f.host.release();
  assert.deepEqual(f.host.resourceStats(), lost);
});
