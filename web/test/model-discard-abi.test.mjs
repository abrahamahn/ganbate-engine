import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const wasmPath = new URL('../../target/wasm32-unknown-unknown/release/ganbate_web.wasm', import.meta.url);

// Exercise the actual Rust runtime/GLB parser. Numeric GPU stubs accept resources but do
// not compile shaders or render; this is an ABI/ownership test, not driver validation.
function runtime(module) {
  let x, handle = 0;
  let bufferAttempts = 0, failBufferAt = -1;
  const liveBuffers = new Set();
  const releasedBuffers = [];
  const logs = [];
  const imports = {};
  for (const entry of WebAssembly.Module.imports(module)) {
    (imports[entry.module] ??= {})[entry.name] = () => 0;
  }
  const host = imports.ganbate;
  for (const name of Object.keys(host)) {
    if (name.startsWith('gpu_create_')) host[name] = () => ++handle;
  }
  host.gpu_create_buffer = () => {
    if (++bufferAttempts === failBufferAt) return 0;
    liveBuffers.add(++handle);
    return handle;
  };
  host.gpu_release = (id) => {
    if (liveBuffers.delete(id)) releasedBuffers.push(id);
  };
  host.gpu_api = () => 1;
  host.gpu_submit = () => 1;
  host.gpu_info = (ptr, cap) => {
    const limits = [8192, 256, 4, 256, 256, 65536, 134217728, 268435456, 8, 7, 65535, 2048, 16, 0, 0, 0];
    const count = Math.min(cap, limits.length);
    new Uint32Array(x.memory.buffer, ptr, count).set(limits.slice(0, count));
    return count;
  };
  host.gpu_canvas = (ptr) => {
    new Uint32Array(x.memory.buffer, ptr, 3).set([128, 72, 1]);
    return 1;
  };
  host.log = (level, ptr, len) => {
    logs.push({ level, text: new TextDecoder().decode(new Uint8Array(x.memory.buffer, ptr, len)) });
  };
  x = new WebAssembly.Instance(module, imports).exports;
  const parse = (name, triangle = false) => {
    const document = {
      asset: { version: '2.0' }, nodes: [{ name }], scenes: [{ nodes: [0] }], scene: 0,
    };
    const bin = new Uint8Array(triangle ? 44 : 0);
    if (triangle) {
      new Float32Array(bin.buffer, 0, 9).set([0, 0, 0, 1, 0, 0, 0, 1, 0]);
      new Uint16Array(bin.buffer, 36, 3).set([0, 1, 2]);
      document.nodes[0].mesh = 0;
      document.buffers = [{ byteLength: 42 }];
      document.bufferViews = [{ buffer: 0, byteOffset: 0, byteLength: 36 }, { buffer: 0, byteOffset: 36, byteLength: 6 }];
      document.accessors = [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5123, count: 3, type: 'SCALAR' },
      ];
      document.meshes = [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }];
    }
    const json = new TextEncoder().encode(JSON.stringify(document));
    const length = (json.length + 3) & ~3;
    const bytes = new Uint8Array(20 + length + (triangle ? 8 + bin.length : 0));
    const header = new DataView(bytes.buffer);
    [0x46546c67, 2, bytes.length, length, 0x4e4f534a].forEach((value, i) => header.setUint32(i * 4, value, true));
    bytes.fill(32, 20, 20 + length);
    bytes.set(json, 20);
    if (triangle) {
      header.setUint32(20 + length, bin.length, true);
      header.setUint32(24 + length, 0x004e4942, true);
      bytes.set(bin, 28 + length);
    }
    const ptr = x.ganbate_alloc(bytes.length);
    try {
      new Uint8Array(x.memory.buffer, ptr, bytes.length).set(bytes);
      return x.ganbate_rt_model_parse(ptr, bytes.length);
    } finally {
      x.ganbate_free(ptr, bytes.length);
    }
  };
  const describe = (parsed) => {
    const length = x.ganbate_rt_model_describe(parsed, 0, 0);
    assert.ok(length > 0);
    const ptr = x.ganbate_alloc(length);
    try {
      x.ganbate_rt_model_describe(parsed, ptr, length);
      return JSON.parse(new TextDecoder().decode(new Uint8Array(x.memory.buffer, ptr, length)));
    } finally {
      x.ganbate_free(ptr, length);
    }
  };
  return { x, logs, parse, describe, liveBuffers, releasedBuffers,
    failBufferAfter: (count) => { failBufferAt = bufferAttempts + count; },
  };
}

test('compiled model discard spends only its pending slot and is quiet on redundant cleanup', {
  skip: !fs.existsSync(wasmPath) && 'Build ganbate-web for wasm32 to exercise the compiled ABI',
}, () => {
  const module = new WebAssembly.Module(fs.readFileSync(wasmPath));
  const { x, logs, parse, describe } = runtime(module);
  assert.equal(typeof x.ganbate_rt_model_discard, 'function', 'SDK and WASM must provide matching cleanup ABI');
  x.ganbate_rt_model_discard(0);
  x.ganbate_rt_model_discard(0xffffffff);
  assert.deepEqual(logs, [], 'cleanup before start is silent');
  assert.equal(x.ganbate_rt_start(0, 0, 0, 0, 3), 1, JSON.stringify(logs));
  const first = parse('discarded');
  const second = parse('retained');
  assert.ok(first > 0 && second > first);
  const logCount = logs.length;
  for (const id of [first, first, 0, 0xffffffff]) x.ganbate_rt_model_discard(id);
  assert.equal(logs.length, logCount, 'missing and spent handles do not log cleanup errors');
  assert.equal(describe(second).nodes[0].name, 'retained');
  const replacement = parse('replacement');
  assert.equal(replacement, first, 'discard returns the slot to the pending-model pool');
  assert.equal(describe(replacement).nodes[0].name, 'replacement');

  const model = x.ganbate_rt_model_create(replacement, 0, 0);
  assert.ok(model > 0, JSON.stringify(logs));
  x.ganbate_rt_model_discard(replacement);
  assert.equal(logs.length, logCount, 'create already consumed the parsed handle');
  const transform = x.ganbate_alloc(64);
  try {
    new Float32Array(x.memory.buffer, transform, 16).set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    assert.ok(x.ganbate_rt_model_instance(model, transform) > 0, 'discard does not destroy the uploaded model');
  } finally {
    x.ganbate_free(transform, 64);
  }
  const failing = parse('failed upload');
  const textures = x.ganbate_alloc(4);
  try {
    new Uint32Array(x.memory.buffer, textures, 1)[0] = 0;
    assert.equal(x.ganbate_rt_model_create(failing, textures, 1), 0, 'mismatched texture count rejects upload');
  } finally {
    x.ganbate_free(textures, 4);
  }
  const afterFailure = logs.length;
  x.ganbate_rt_model_discard(failing);
  assert.equal(logs.length, afterFailure, 'failed create also consumed the pending handle');
  const after = parse('after failure');
  assert.equal(after, failing);
  x.ganbate_rt_model_discard(after);
  x.ganbate_rt_model_discard(second);
  x.ganbate_rt_stop();
  x.ganbate_rt_model_discard(second);
  assert.equal(logs.length, afterFailure, 'cleanup after stop is silent');
});

test('compiled model upload rolls back a vertex buffer when index allocation fails', {
  skip: !fs.existsSync(wasmPath) && 'Build ganbate-web for wasm32 to exercise the compiled ABI',
}, () => {
  const module = new WebAssembly.Module(fs.readFileSync(wasmPath));
  const f = runtime(module);
  const { x, logs } = f;
  assert.equal(x.ganbate_rt_start(0, 0, 0, 0, 3), 1, JSON.stringify(logs));
  const baseline = f.liveBuffers.size;
  try {
    for (const failedAllocation of [1, 2]) {
      const parsed = f.parse('synthetic triangle', true);
      assert.ok(parsed > 0, JSON.stringify(logs));
      const released = f.releasedBuffers.length;
      f.failBufferAfter(failedAllocation);
      assert.equal(x.ganbate_rt_model_create(parsed, 0, 0), 0, 'injected native allocation failure rejects model');
      assert.equal(f.liveBuffers.size, baseline, 'failed model retains no partial vertex/index buffer');
      assert.equal(f.releasedBuffers.length - released, failedAllocation - 1);
    }
    const retry = f.parse('retry triangle', true);
    const model = x.ganbate_rt_model_create(retry, 0, 0);
    assert.ok(model > 0, JSON.stringify(logs));
    assert.equal(f.liveBuffers.size, baseline + 2, 'successful retry owns a vertex and index buffer');
    x.ganbate_rt_model_destroy(model);
    assert.equal(f.liveBuffers.size, baseline, 'destroy returns to the same resource plateau');
  } finally {
    x.ganbate_rt_stop();
  }
});
