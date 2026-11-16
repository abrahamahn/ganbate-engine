import test from 'node:test';
import assert from 'node:assert/strict';
import { createResourceTracker, estimateTextureBytes, registerResourceTracker } from '../sdk/src/resource-stats.ts';
import { readPixels } from '../sdk/src/wasm-boot.ts';

test('resource snapshots count allocations once, remain independent, and measure concurrent peaks', () => {
  const tracker = createResourceTracker();
  const buffer = {}, texture = {};
  tracker.allocate('buffers', buffer, 128);
  tracker.allocate('buffers', buffer, 128); // another handle/view, not another allocation
  tracker.allocate('textures', null, 256); // failed platform allocation
  const before = tracker.snapshot();
  tracker.release(buffer);
  tracker.release(buffer);
  tracker.allocate('textures', texture, 256);
  const stats = tracker.snapshot();
  assert.equal(before.buffers.liveCount, 1);
  assert.equal(stats.buffers.liveCount, 0);
  assert.equal(stats.buffers.allocations, 1);
  assert.equal(stats.total.allocations, 2);
  assert.equal(stats.total.liveCount, 1);
  assert.equal(stats.total.peakCount, 1);
  assert.equal(stats.total.estimatedBytes, 256);
  assert.equal(stats.total.peakEstimatedBytes, 256, 'peaks are concurrent, not 128+256');
  assert.equal(stats.total.allocatedEstimatedBytes, 384);
  stats.total.estimatedBytes = -1;
  assert.equal(tracker.snapshot().total.estimatedBytes, 256);
});

test('resource estimates distinguish unknown sizes and preserve history after loss or teardown', () => {
  const tracker = createResourceTracker();
  const unknown = {};
  tracker.allocate('textures', unknown, null);
  tracker.allocate('buffers', {}, 64);
  tracker.allocate('buffers', {}, Number.NaN);
  const before = tracker.snapshot();
  assert.equal(before.total.liveCount, 3);
  assert.equal(before.total.unknownSizeCount, 2);
  assert.equal(before.total.unknownSizeAllocations, 2);
  assert.equal(before.total.estimatedBytes, 64);
  tracker.release(unknown);
  tracker.close();
  tracker.close();
  tracker.release(unknown);
  tracker.allocate('textures', {}, 4096); // late async completion cannot restore residency
  const after = tracker.snapshot();
  assert.equal(after.total.liveCount, 0);
  assert.equal(after.total.estimatedBytes, 0);
  assert.equal(after.total.unknownSizeCount, 0);
  assert.equal(after.total.allocations, 3);
  assert.equal(after.total.peakCount, 3);
  assert.equal(after.total.peakEstimatedBytes, 64);
  assert.equal(after.total.unknownSizeAllocations, 2);
});

test('texture estimates include all mips, array layers, samples, and depth formats', () => {
  assert.equal(estimateTextureBytes({ format: 'rgba8unorm-srgb', width: 8, height: 4, mipLevelCount: 4, depthOrArrayLayers: 6 }), (32 + 8 + 2 + 1) * 4 * 6);
  assert.equal(estimateTextureBytes({ format: 'rgba16float', width: 8, height: 4, sampleCount: 4 }), 8 * 4 * 8 * 4);
  assert.equal(estimateTextureBytes({ format: 'depth32float', width: 8, height: 4 }), 128);
  assert.equal(estimateTextureBytes({ format: 'depth16unorm', width: 8, height: 4 }), 64);
  assert.equal(estimateTextureBytes({ format: 'r8unorm', width: 8, height: 4, depthOrArrayLayers: 4, dimension: '3d', mipLevelCount: 4 }), 128 + 16 + 2 + 1);
});

test('compressed estimates round each mip to full blocks without shrinking array layers', () => {
  assert.equal(estimateTextureBytes({ format: 'bc1-rgba-unorm-srgb', width: 7, height: 5, mipLevelCount: 3, depthOrArrayLayers: 2 }), (4 + 1 + 1) * 8 * 2);
  assert.equal(estimateTextureBytes({ format: 'bc7-rgba-unorm', width: 4, height: 4 }), 16);
  assert.equal(estimateTextureBytes({ format: 'astc-5x4-unorm', width: 6, height: 7, mipLevelCount: 3 }), (4 + 1 + 1) * 16);
  assert.equal(estimateTextureBytes({ format: 'etc2-rgb8a1unorm', width: 5, height: 5 }), 32);
  assert.equal(estimateTextureBytes({ format: 'eac-rg11snorm', width: 5, height: 5 }), 64);
});

test('unknown/opaque formats and invalid or overflowing dimensions report unknown', () => {
  for (const format of ['future-format', 'bc4-r-unorm-srgb', 'eac-r11unorm-srgb', 'depth24plus', 'depth24plus-stencil8', 'depth32float-stencil8']) {
    assert.equal(estimateTextureBytes({ format, width: 8, height: 8 }), null);
  }
  for (const width of [0, -1, 0.5, NaN, Infinity, Number.MAX_SAFE_INTEGER]) {
    assert.equal(estimateTextureBytes({ format: 'rgba32float', width, height: 8 }), null);
  }
  assert.equal(estimateTextureBytes({ format: 'r8unorm', width: 4, mipLevelCount: 20 }), null);
});

test('capture readback accounts aligned storage and releases it after success or mapping failure', async (t) => {
  const previousUsage = globalThis.GPUBufferUsage, previousMode = globalThis.GPUMapMode;
  t.after(() => { globalThis.GPUBufferUsage = previousUsage; globalThis.GPUMapMode = previousMode; });
  globalThis.GPUBufferUsage = { COPY_DST: 8, MAP_READ: 1 };
  globalThis.GPUMapMode = { READ: 1 };
  for (const fail of [false, true]) {
    const resources = createResourceTracker();
    const host = registerResourceTracker({}, resources);
    let resolve, reject, destroyed = 0;
    const pending = new Promise((a, b) => { resolve = a; reject = b; });
    const buffer = { mapAsync: () => pending, getMappedRange: () => new ArrayBuffer(512), unmap() {}, destroy() { destroyed++; } };
    const gpu = { api: 'webgpu', format: 'rgba8unorm', device: {
      createBuffer(descriptor) { assert.equal(descriptor.size, 512); return buffer; },
      createCommandEncoder: () => ({ copyTextureToBuffer() {}, finish() {} }), queue: { submit() {} },
    } };
    const capture = readPixels(gpu, { width: 3, height: 2 }, host);
    assert.equal(resources.snapshot().buffers.estimatedBytes, 512);
    if (fail) {
      const failure = assert.rejects(capture, /mapping failed/);
      reject(new Error('mapping failed'));
      await failure;
    } else {
      resolve();
      assert.equal((await capture).rgba.length, 24);
    }
    assert.equal(destroyed, 1);
    assert.equal(resources.snapshot().buffers.liveCount, 0);
    assert.equal(resources.snapshot().buffers.allocatedEstimatedBytes, 512);
  }
});
