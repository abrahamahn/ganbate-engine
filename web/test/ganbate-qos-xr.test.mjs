import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  createEngine,
  qualityCodeFromMode,
  resolveQualityCode,
} from '../dist/ganbate.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..');

function makeMockCanvas(width = 128, height = 72) {
  return {
    width,
    height,
    clientWidth: width,
    clientHeight: height,
    addEventListener() {},
    getContext(kind) {
      if (kind !== 'webgl2') return null;
      const base = {
        RENDERER: 0x1F01,
        MAX_COMBINED_TEXTURE_IMAGE_UNITS: 0x8B4D,
        NO_ERROR: 0,
        getExtension() { return null; },
        getParameter(param) {
          if (param === 0x1F01) return 'MockWebGL2';
          if (param === 0x8B4D) return 16;
          return 0;
        },
        getError() { return 0; },
      };
      return new Proxy(base, {
        get(target, prop) {
          if (prop in target) return target[prop];
          if (typeof prop === 'string' && /^[A-Z0-9_]+$/.test(prop)) return 0;
          return () => ({});
        },
      });
    },
  };
}

function makeMockWasmInstance(overrides = {}) {
  const memory = new WebAssembly.Memory({ initial: 4 });
  let bump = 1024;
  const calls = {
    startedQuality: null,
    setQualityCodes: [],
    stereoCalls: [],
    governorRecorded: [],
    governorQueries: 0,
    governorResets: 0,
  };
  let tier = 3;
  let rollingMeanMs = 4.2;
  let ipd = 0.063;

  const exports = {
    memory,
    ganbate_alloc(len) {
      const ptr = (bump + 7) & ~7;
      bump = ptr + Math.max(len >>> 0, 8);
      return ptr;
    },
    ganbate_free() {},
    ganbate_rt_start(_worldPtr, _worldLen, _lookPtr, _lookLen, quality) {
      calls.startedQuality = quality >>> 0;
      return 1;
    },
    ganbate_rt_stop() { return 1; },
    ganbate_rt_size(outPtr) {
      const dst = new Uint32Array(memory.buffer, outPtr >>> 0, 2);
      dst[0] = 128;
      dst[1] = 72;
      return 1;
    },
    ganbate_rt_set_quality(code) {
      calls.setQualityCodes.push(code >>> 0);
      return 1;
    },
    ganbate_rt_frame_stats(outPtr) {
      const dst = new Uint32Array(memory.buffer, outPtr >>> 0, 9);
      dst.set([48, 31, 17, 1, 9, 1, 0, 0, 1]);
      return 1;
    },
    ganbate_rt_set_stereo_eye(eye, ipdMetres) {
      calls.stereoCalls.push({ eye, ipdMetres });
      ipd = Math.min(0.078, Math.max(0.050, ipdMetres));
      return 1;
    },
    ganbate_rt_stereo_views(outPtr) {
      const half = ipd * 0.5;
      const dst = new Float32Array(memory.buffer, outPtr >>> 0, 16);
      dst.set([
        -half, 1.65, 3.0, -half, 1.4, 0.0,
        half, 1.65, 3.0, half, 1.4, 0.0,
        1.0, 0.0, 0.0,
        ipd,
      ]);
      return 1;
    },
    ganbate_rt_governor_record(dtMs, outPtr) {
      calls.governorRecorded.push(dtMs);
      rollingMeanMs = dtMs;
      if (dtMs > 8.6) tier = Math.max(0, tier - 1);
      const dst = new Float32Array(memory.buffer, outPtr >>> 0, 6);
      dst.set([tier === 3 ? 1.0 : 0.85, tier === 3 ? 8 : 4, tier >= 2 ? 1 : 0, tier, tier, rollingMeanMs]);
      return 1;
    },
    ganbate_rt_governor_query(outPtr) {
      calls.governorQueries += 1;
      const dst = new Float32Array(memory.buffer, outPtr >>> 0, 6);
      dst.set([tier === 3 ? 1.0 : 0.85, tier === 3 ? 8 : 4, tier >= 2 ? 1 : 0, tier, tier, rollingMeanMs]);
      return 1;
    },
    ganbate_rt_governor_reset() {
      calls.governorResets += 1;
      tier = 3;
      rollingMeanMs = 0.0;
      return 1;
    },
    ...overrides,
  };

  for (const [k, v] of Object.entries(overrides)) {
    if (v === undefined) delete exports[k];
  }

  return { instance: { exports }, calls };
}

test('qualityCodeFromMode and resolveQualityCode map all presets and legacy still option', () => {
  assert.equal(qualityCodeFromMode('realtime'), 0);
  assert.equal(qualityCodeFromMode('still'), 1);
  assert.equal(qualityCodeFromMode('hybrid'), 2);
  assert.equal(qualityCodeFromMode('low-cost'), 3);
  assert.equal(qualityCodeFromMode(false), 0);
  assert.equal(qualityCodeFromMode(true), 1);
  assert.equal(qualityCodeFromMode('unknown'), null);

  assert.equal(resolveQualityCode(undefined, undefined), 0);
  assert.equal(resolveQualityCode(undefined, false), 0);
  assert.equal(resolveQualityCode(undefined, true), 1);
  assert.equal(resolveQualityCode('low-cost', true), 3);
  assert.equal(resolveQualityCode('hybrid', false), 2);
  assert.equal(resolveQualityCode(true, false), 1);
  assert.equal(resolveQualityCode(false, true), 0);
});

test('createEngine forwards startup quality and legacy still options to ganbate_rt_start', async () => {
  const cases = [
    [{}, 0],
    [{ quality: 'realtime' }, 0],
    [{ quality: 'still' }, 1],
    [{ quality: 'hybrid' }, 2],
    [{ quality: 'low-cost' }, 3],
    [{ quality: true }, 1],
    [{ quality: false, still: true }, 0],
    [{ still: true }, 1],
  ];

  for (const [opts, expectedCode] of cases) {
    const { instance, calls } = makeMockWasmInstance();
    await createEngine({
      canvas: makeMockCanvas(),
      wasm: instance,
      api: 'webgl2',
      ...opts,
    });
    assert.equal(calls.startedQuality, expectedCode, `options ${JSON.stringify(opts)}`);
  }
});

test('setQuality and frameStats wrap WASM exports cleanly', async () => {
  const { instance, calls } = makeMockWasmInstance();
  const engine = await createEngine({
    canvas: makeMockCanvas(),
    wasm: instance,
    api: 'webgl2',
    quality: 'hybrid',
  });

  assert.equal(calls.startedQuality, 2);
  assert.equal(engine.setQuality('low-cost'), true);
  assert.equal(engine.setQuality('realtime'), true);
  assert.equal(engine.setQuality('still'), true);
  assert.equal(engine.setQuality('hybrid'), true);
  assert.equal(engine.setQuality('bogus-preset'), false);
  assert.deepEqual(calls.setQualityCodes, [3, 0, 1, 2]);

  const stats = engine.frameStats();
  assert.deepEqual(stats, {
    worldDrawsTotal: 48,
    worldDrawsSubmitted: 31,
    worldDrawsCulled: 17,
    shadowStaticCached: true,
    shadowWorldDraws: 9,
    groundCached: true,
    reflectionDrawn: false,
    ssaoDrawn: false,
    fusedTonemap: true,
  });
});

test('setStereoEye, stereoViews, and governorRecord/Query/Reset work end-to-end', async () => {
  const { instance, calls } = makeMockWasmInstance();
  const engine = await createEngine({
    canvas: makeMockCanvas(),
    wasm: instance,
    api: 'webgl2',
  });

  assert.equal(engine.setStereoEye('left', 0.064), true);
  assert.equal(engine.setStereoEye('right', 0.066), true);
  assert.equal(engine.setStereoEye('mono'), true);
  assert.equal(engine.setStereoEye(-1, 0.062), true);
  assert.equal(engine.setStereoEye('invalid-eye', 0.063), false);
  assert.equal(engine.setStereoEye('left', Number.NaN), false);
  assert.deepEqual(calls.stereoCalls, [
    { eye: -1, ipdMetres: 0.064 },
    { eye: 1, ipdMetres: 0.066 },
    { eye: 0, ipdMetres: 0.063 },
    { eye: -1, ipdMetres: 0.062 },
  ]);

  const views = engine.stereoViews();
  assert.ok(views);
  assert.ok(Math.abs(views.ipdMetres - 0.062) < 1e-6);
  assert.ok(views.left.eye[0] < 0 && views.right.eye[0] > 0);
  assert.deepEqual(views.rightAxis, [1, 0, 0]);

  const snap0 = engine.governorQuery();
  assert.ok(snap0);
  assert.equal(snap0.tier, 3);
  assert.equal(snap0.ambientParticles, true);

  const snap1 = engine.governorRecord(11.4);
  assert.ok(snap1);
  assert.equal(snap1.tier, 2);
  assert.ok(Math.abs(snap1.rollingMeanMs - 11.4) < 1e-4);

  // Legacy aliases stay wired to the exact same governor exports.
  const aliasSnap = engine.frameBudget();
  assert.deepEqual(aliasSnap, snap1);
  assert.equal(engine.governorReset(), true);
  const resetSnap = engine.governorQuery();
  assert.equal(resetSnap.tier, 3);
  assert.equal(resetSnap.rollingMeanMs, 0);
  assert.equal(engine.resetFrameGovernor(), true);
  assert.equal(calls.governorResets, 2);
});

test('QoS and XR methods degrade gracefully when older mock WASM omits exports', async () => {
  const { instance } = makeMockWasmInstance({
    ganbate_rt_set_quality: undefined,
    ganbate_rt_frame_stats: undefined,
    ganbate_rt_set_stereo_eye: undefined,
    ganbate_rt_stereo_views: undefined,
    ganbate_rt_governor_record: undefined,
    ganbate_rt_governor_query: undefined,
    ganbate_rt_governor_reset: undefined,
  });

  const engine = await createEngine({
    canvas: makeMockCanvas(),
    wasm: instance,
    api: 'webgl2',
  });

  assert.equal(engine.setQuality('low-cost'), false);
  assert.equal(engine.frameStats(), null);
  assert.equal(engine.setStereoEye('left', 0.063), false);
  assert.equal(engine.stereoViews(), null);
  assert.equal(engine.governorRecord(8.0), null);
  assert.equal(engine.governorQuery(), null);
  assert.equal(engine.governorReset(), false);
});

test('compiled ganbate_web.wasm exports all Wave 6 QoS and XR C-ABI symbols when present', () => {
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
  for (const sym of [
    'ganbate_rt_start',
    'ganbate_rt_set_quality',
    'ganbate_rt_frame_stats',
    'ganbate_rt_set_stereo_eye',
    'ganbate_rt_stereo_views',
    'ganbate_rt_governor_record',
    'ganbate_rt_governor_query',
    'ganbate_rt_governor_reset',
  ]) {
    assert.ok(exportedFns.has(sym), `missing WASM export ${sym}`);
  }
});
