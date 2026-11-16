import test from 'node:test';
import assert from 'node:assert/strict';
import { createGlHost } from '../sdk/src/webgl-host.ts';

// Allocation accounting on synthetic GL objects; no claim of physical driver residency.
function fixture() {
  let nextId = 0;
  const constants = new Map();
  const deletedBuffers = new Set();
  const deletedTextures = new Set();
  const storage = [];
  const fail = { buffer: false, texture: false, bufferStorage: false, textureStorage: false };
  const canvas = Object.assign(new EventTarget(), { width: 32, height: 16 });
  const object = (kind) => ({ kind, id: nextId++ });
  const gl = new Proxy({
    getExtension: () => null,
    getParameter: () => 16,
    getShaderParameter: () => true,
    getProgramParameter: () => true,
    getUniformLocation: () => ({}),
    createShader: () => object('shader'),
    createProgram: () => object('program'),
    createBuffer: () => fail.buffer ? null : object('buffer'),
    createTexture: () => fail.texture ? null : object('texture'),
    bufferData: () => {
      if (fail.bufferStorage) throw new Error('buffer storage failed');
    },
    texStorage2D: (...args) => {
      if (fail.textureStorage) throw new Error('texture storage failed');
      storage.push(args);
    },
    texStorage3D: (...args) => {
      if (fail.textureStorage) throw new Error('texture storage failed');
      storage.push(args);
    },
    deleteBuffer: (value) => { if (value) deletedBuffers.add(value); },
    deleteTexture: (value) => { if (value) deletedTextures.add(value); },
    isContextLost: () => false,
    fenceSync: () => ({}),
    clientWaitSync: () => gl.ALREADY_SIGNALED,
  }, {
    get(target, key) {
      if (key in target) return target[key];
      if (/^[A-Z_0-9]+$/.test(String(key))) {
        if (!constants.has(key)) constants.set(key, constants.size + 1);
        return constants.get(key);
      }
      return () => ({});
    },
  });
  const host = createGlHost({ gl, ext: {}, canvas });
  host.attach({ exports: { memory: new WebAssembly.Memory({ initial: 4 }) } });
  return { host, g: host.imports.ganbate, gl, canvas, fail, storage, deletedBuffers, deletedTextures };
}

test('counts actual native buffers and padded storage textures, excluding CPU readback arrays', () => {
  const f = fixture();
  const initial = f.host.resourceStats();
  assert.equal(initial.buffers.liveCount, 1, 'the host uniform ring exists without storage yet');
  assert.equal(initial.total.estimatedBytes, 0);
  const vertex = f.g.gl_create_buffer(64, 1 | 2);
  const storage = f.g.gl_create_buffer(16385, 4);
  const dual = f.g.gl_create_buffer(64, 1 | 4 | 8);
  const readback = f.g.gl_create_buffer(1024, 8);
  const allocated = f.host.resourceStats();
  assert.equal(allocated.buffers.liveCount, 3);
  assert.equal(allocated.buffers.estimatedBytes, 128);
  assert.equal(allocated.textures.liveCount, 2);
  assert.equal(allocated.textures.estimatedBytes, 32768 + 16384);
  assert.equal(allocated.total.estimatedBytes, 128 + 32768 + 16384);
  assert.equal(allocated.total.unknownSizeCount, 0);
  for (const handle of [vertex, storage, dual, readback, vertex]) f.g.gl_release(handle);
  const released = f.host.resourceStats();
  assert.equal(released.total.liveCount, 1);
  assert.equal(released.total.estimatedBytes, 0);
  assert.equal(released.total.allocations, allocated.total.allocations);
  assert.equal(f.deletedBuffers.size, 2);
  assert.equal(f.deletedTextures.size, 2);
  f.host.release();
  assert.equal(f.host.resourceStats().total.liveCount, 0);
});

test('estimates array mip chains and uses BC1 sRGB storage rather than DXT5', () => {
  const f = fixture();
  const floatArray = f.g.gl_create_texture(8, 4, 3, 4, 5, 1);
  assert.equal(f.host.resourceStats().textures.estimatedBytes, 1032);
  const compressed = f.g.gl_create_texture(8, 8, 2, 4, 13, 1);
  assert.equal(f.storage.at(-1)[2], 0x8c4d, 'COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT');
  assert.equal(f.host.resourceStats().textures.estimatedBytes, 1032 + 112);
  f.g.gl_release(floatArray);
  assert.equal(f.host.resourceStats().textures.estimatedBytes, 112);
  f.g.gl_release(compressed);
  assert.equal(f.host.resourceStats().textures.estimatedBytes, 0);
  f.host.release();
});

test('tracks uniform ring storage growth without counting updates as allocations', () => {
  const f = fixture();
  assert.equal(f.g.gl_submit(0, 0, 0, 64, 0), 1);
  assert.equal(f.host.resourceStats().buffers.estimatedBytes, 65536);
  assert.equal(f.host.resourceStats().buffers.allocations, 2);
  assert.equal(f.g.gl_submit(0, 0, 0, 128, 0), 1);
  assert.equal(f.host.resourceStats().buffers.allocations, 2);
  assert.equal(f.g.gl_submit(0, 0, 0, 131072, 0), 1);
  const grown = f.host.resourceStats().buffers;
  assert.equal(grown.liveCount, 1);
  assert.equal(grown.peakCount, 1);
  assert.equal(grown.estimatedBytes, 131072);
  assert.equal(grown.peakEstimatedBytes, 131072);
  assert.equal(grown.allocatedEstimatedBytes, 65536 + 131072);
  f.fail.bufferStorage = true;
  assert.equal(f.g.gl_submit(0, 0, 0, 196608, 0), 0);
  assert.deepEqual(f.host.resourceStats().buffers, grown, 'a failed replacement preserves the old store');
  f.host.release();
});

test('surface resize replaces owned storage; aliases and external XR targets add no resources', () => {
  const f = fixture();
  assert.equal(f.g.gl_canvas(0), 1);
  assert.equal(f.host.resourceStats().textures.estimatedBytes, 32 * 16 * 4);
  const alias = f.host.frameTexture();
  f.g.gl_canvas(0);
  assert.equal(f.host.frameTexture(), alias);
  assert.equal(f.host.resourceStats().textures.allocations, 1);
  assert.equal(f.host.setXrTarget({ framebuffer: {}, x: 0, y: 0, width: 512, height: 512 }), true);
  assert.equal(f.host.resourceStats().textures.allocations, 1);
  f.canvas.width = 64;
  f.g.gl_canvas(0);
  assert.equal(f.deletedTextures.size, 1);
  assert.equal(f.host.resourceStats().textures.liveCount, 1);
  assert.equal(f.host.resourceStats().textures.estimatedBytes, 64 * 16 * 4);
  assert.equal(f.host.resourceStats().textures.allocations, 2);
  assert.equal(f.host.resourceStats().textures.peakCount, 1);
  f.host.release();
  const closed = f.host.resourceStats();
  assert.equal(closed.total.liveCount, 0);
  assert.equal(closed.total.estimatedBytes, 0);
  f.host.release();
  assert.deepEqual(f.host.resourceStats(), closed);
});

test('null and thrown allocations do not enter the ledger and partial objects are deleted', () => {
  const f = fixture();
  const baseline = f.host.resourceStats();
  f.fail.buffer = true;
  assert.equal(f.g.gl_create_buffer(64, 1), 0);
  f.fail.buffer = false;
  f.fail.bufferStorage = true;
  assert.equal(f.g.gl_create_buffer(64, 1), 0);
  f.fail.bufferStorage = false;
  f.fail.texture = true;
  assert.equal(f.g.gl_create_texture(8, 8, 1, 1, 1, 0), 0);
  assert.equal(f.g.gl_create_buffer(64, 4), 0);
  f.fail.texture = false;
  f.fail.textureStorage = true;
  assert.equal(f.g.gl_create_texture(8, 8, 1, 1, 1, 0), 0);
  assert.equal(f.g.gl_create_buffer(64, 1 | 4), 0);
  assert.deepEqual(f.host.resourceStats(), baseline);
  assert.equal(f.deletedBuffers.size, 2);
  assert.equal(f.deletedTextures.size, 2);
  f.host.release();
});

test('context loss clears live accounting, keeps peaks, and ignores late allocations', async () => {
  const f = fixture();
  f.g.gl_create_buffer(128, 1 | 4);
  f.g.gl_create_texture(32, 32, 1, 1, 1, 0);
  const before = f.host.resourceStats();
  before.total.liveCount = -1;
  assert.ok(f.host.resourceStats().total.liveCount > 0, 'snapshots are independent');
  f.canvas.dispatchEvent(new Event('webglcontextlost', { cancelable: true }));
  assert.equal(await f.host.lost, 'WebGL 2 context lost');
  const lost = f.host.resourceStats();
  assert.equal(lost.total.liveCount, 0);
  assert.equal(lost.total.estimatedBytes, 0);
  assert.ok(lost.total.peakEstimatedBytes > 0);
  f.g.gl_create_buffer(64, 1);
  f.g.gl_create_texture(4, 4, 1, 1, 1, 0);
  assert.deepEqual(f.host.resourceStats(), lost);
  f.host.release();
  assert.deepEqual(f.host.resourceStats(), lost);
});
