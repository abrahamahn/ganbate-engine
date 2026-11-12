// Ganbate web host: creates the WebGPU device (or, without WebGPU, a WebGL 2 context) and the
// canvas context, provides the imports of the wasm module (crates/ganbate-web, which links
// ganbate-engine's WebGPU and WebGL 2 backends), executes the command streams it submits, forwards
// input and drives the frame loop. `api: 'webgl2'` (the page's ?api=webgl2) forces WebGL 2.
//
// Imports (wasm module "ganbate"): gpu_* calls of src/gpu/webgpu/ffi.rs, gl_* calls of
// src/gpu/webgl/ffi.rs, gpu_api (1 WebGPU, 2 WebGL 2), plus log and status for the player. GPU objects live in `objects`; Rust holds their indices (see ffi.rs for the
// lifetime and synchronization rules). Everything here runs on the page's main thread.

const FORMATS = [null, 'rgba8unorm', 'rgba8unorm-srgb', 'bgra8unorm', 'bgra8unorm-srgb',
  'rgba16float', 'rg16float', 'r16float', 'r8unorm', 'r32float', 'rgba32float', 'depth32float',
  'bc1-rgba-unorm', 'bc1-rgba-unorm-srgb', 'bc7-rgba-unorm', 'bc7-rgba-unorm-srgb',
  'astc-4x4-unorm', 'astc-4x4-unorm-srgb'];
const SURFACE = 0xFFFFFFFF;
const EMPTY_GROUP = 0xFFFFFFFE;
const PUSH_RENDER = 0xFFFFFFFD;
const PUSH_COMPUTE = 0xFFFFFFFC;
const PUSH_SLOT = 256;
const VIEW_DIMENSIONS = ['2d', '2d-array', 'cube'];
const SAMPLE_TYPES = ['float', 'unfilterable-float', 'depth', 'sint', 'uint'];
const COMPARES = [undefined, 'never', 'less', 'less-equal', 'equal', 'greater', 'greater-equal',
  'not-equal', 'always'];
const ADDRESS = ['repeat', 'clamp-to-edge', 'mirror-repeat'];

const BLIT_WGSL = `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
struct Out { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: Out;
  o.pos = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
  o.uv = uv;
  return o;
}
@fragment fn fs(i: Out) -> @location(0) vec4f { return textureSampleLevel(src, smp, i.uv, 0.0); }
`;

/**
 * Each pass's GPU time on a device no profiler reaches (an iPhone; 2026-11-12). While it is on, the
 * engine draws one frame at a time (`busy`), so every frame starts on an idle GPU, and a host marks
 * each pass (`mark`) with when it started (`sent`) and finished (`done`): WebGL 2 sends the passes
 * one at a time, each followed by a fence; WebGPU writes timestamps where the GPU has them, and
 * otherwise marks the whole frame. A pass took from the later of its start and the mark before it
 * to its finish, so the time the GPU waited for the page is left out.
 */
function createPassTiming() {
  let on = false;
  let generation = 0;
  let marks = [];
  let pending = 0;
  let frames = 0;
  let busyMs = 0;
  let wallMs = 0;
  const sums = new Map();
  return {
    on: () => on,
    set(value) {
      on = !!value;
      generation++;
      marks = [];
      pending = 0;
      frames = 0;
      busyMs = 0;
      wallMs = 0;
      sums.clear();
    },
    busy: () => pending > 0,
    /** A marker after the work sent so far; the host sets its `done` when the GPU finished it. */
    mark(label) {
      const m = { label, sent: performance.now(), done: 0 };
      marks.push(m);
      return m;
    },
    /** The frame's markers and `finish`, which books them once every `done` is set. */
    take() {
      const taken = marks;
      marks = [];
      if (!taken.length) return null;
      pending++;
      const g = generation;
      return {
        marks: taken,
        finish() {
          if (g !== generation) return;
          pending--;
          let prev = taken[0].sent;
          for (const m of taken) {
            const ms = Math.max(0, m.done - Math.max(prev, m.sent));
            sums.set(m.label, (sums.get(m.label) || 0) + ms);
            busyMs += ms;
            prev = Math.max(prev, m.done);
          }
          wallMs += prev - taken[0].sent;
          frames++;
        },
      };
    },
    report() {
      const n = Math.max(1, frames);
      return {
        frames,
        gpuMs: busyMs / n,
        wallMs: wallMs / n,
        passes: [...sums].map(([label, ms]) => ({ label, ms: ms / n })),
      };
    },
  };
}

/** Apple's WebKit: every browser on an iPhone or iPad (iPadOS asks as a Mac with touch), and Safari on a Mac. */
function appleWebKit() {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent || '';
  const ios = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
  const macSafari = /Macintosh/.test(ua) && /Version\/[\d.]+.*Safari\//.test(ua)
    && !/Chrome|Chromium|Edg|Firefox|OPR/.test(ua);
  return ios || macSafari;
}

/**
 * How many frames are on the GPU at once and, on a settled host, when a finished frame is shown.
 * Direct (the default), the engine draws into the canvas with up to two frames on the GPU. Settled
 * (`present: 'settled'`), it draws offscreen with one frame on the GPU and copies a frame to the
 * canvas only once the GPU has finished it, ahead of the next: the canvas then only ever receives a
 * finished picture, and a frame the GPU refused shows the last one again rather than whatever an
 * undrawn canvas buffer held (Safari's WebGPU canvas shows a three-frame-old picture there). On an
 * iPhone it costs throughput: Safari reports a 9 ms frame done some 35 ms after it was sent.
 */
function createFrameGate(gpu, host, settled) {
  let inFlight = 0;
  let ready = null;
  let shown = null;
  let wake = 0;
  let stopped = false;
  const show = () => {
    if (!ready || stopped) return;
    shown = ready;
    ready = null;
    host.present(shown);
  };
  return {
    full: () => inFlight >= (settled ? 1 : 2),
    /** Counts a submitted frame until the GPU has finished it; settled, it is then shown. */
    submitted() {
      if (!gpu.device) return;
      inFlight++;
      const frame = settled ? host.frameTexture() : null;
      const done = () => {
        inFlight--;
        if (!frame || stopped) return;
        ready = frame;
        // A table at rest draws no next frame to show this one before: an animation frame does.
        if (!wake) wake = requestAnimationFrame(() => { wake = 0; show(); });
      };
      gpu.device.queue.onSubmittedWorkDone().then(done, done);
    },
    /** Shows the finished frame (settled), before the next is drawn into its texture. */
    show,
    /** A resized canvas starts empty: the last finished frame, stretched, until the next. */
    reshow() {
      if (!settled || inFlight || stopped) return;
      if (ready) show();
      else if (shown) host.present(shown);
    },
    /** Shows nothing more (the engine is destroyed). */
    stop() { stopped = true; },
  };
}

/** Creates the WebGPU device and, if `canvas` is given, configures its context. */
export async function createGpu(canvas) {
  if (!navigator.gpu) {
    throw new Error('This browser has no WebGPU. Use Chrome or Edge 113+ or Safari 26+, ' +
      'served over https or from localhost.');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU is available but no adapter was found.');
  const wanted = ['depth-clip-control', 'float32-filterable', 'float32-blendable',
    'texture-compression-bc', 'texture-compression-astc', 'timestamp-query'];
  const requiredFeatures = wanted.filter((f) => adapter.features.has(f));
  const A = adapter.limits;
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxBufferSize: A.maxBufferSize,
      maxStorageBufferBindingSize: A.maxStorageBufferBindingSize,
      maxTextureDimension2D: A.maxTextureDimension2D,
    },
  });
  const info = adapter.info || {};
  const adapterName = ['WebGPU', info.vendor, info.architecture, info.device, info.description]
    .filter(Boolean).join(' ');
  let context = null;
  let format = null;
  if (canvas) {
    context = canvas.getContext('webgpu');
    format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format,
      alphaMode: 'opaque',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
  }
  return { api: 'webgpu', adapter, device, adapterName, features: requiredFeatures, canvas, context, format };
}

/** The import object for the wasm module and the host state behind it. */
export function createHost(gpu, { onLog = () => {}, onStatus = () => {}, onError = () => {}, settled = false, bundles = true } = {}) {
  const { device, canvas, context } = gpu;
  const objects = [null];
  const freeList = [];
  const errors = [];
  let lastError = '';
  let memory = null;
  let surfaceTexture = null;
  let surfaceView = null;
  /** The image `gpu_copy_source` copies: set by the engine around one runtime call. */
  let source = null;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  const report = (message) => {
    errors.push(message);
    onError(message);
  };
  device.addEventListener('uncapturederror', (e) => report(`WebGPU: ${e.error.message}`));
  device.lost.then((i) => report(`WebGPU device lost (${i.reason}): ${i.message}`));

  const put = (obj) => {
    const h = freeList.length ? freeList.pop() : objects.length;
    objects[h] = obj;
    return h;
  };
  const u8 = (ptr, len) => new Uint8Array(memory.buffer, ptr >>> 0, len >>> 0);
  const u32 = (ptr, n) => new Uint32Array(memory.buffer, ptr >>> 0, n >>> 0);
  const str = (ptr, len) => decoder.decode(u8(ptr, len));
  const writeStr = (s, out, cap) => {
    const bytes = encoder.encode(s);
    const n = Math.min(bytes.length, cap >>> 0);
    u8(out, n).set(bytes.subarray(0, n));
    return n;
  };
  const guard = (what, f) => (...args) => {
    try {
      return f(...args);
    } catch (e) {
      lastError = `${what}: ${e.message}`;
      return 0;
    }
  };

  // Settled (`createFrameGate`), frames are drawn into a texture of the canvas's size and copied
  // to the canvas once finished (`present`); one replaced by a resize goes once a newer is shown.
  let stage = null;
  const retired = [];
  const presentGroups = new WeakMap();
  const currentTexture = () => {
    if (surfaceTexture) return surfaceTexture;
    if (!settled) {
      surfaceTexture = context.getCurrentTexture();
    } else {
      if (!stage || stage.width !== canvas.width || stage.height !== canvas.height) {
        if (stage) retired.push(stage);
        stage = device.createTexture({
          size: [canvas.width, canvas.height],
          format: gpu.format,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
      }
      surfaceTexture = stage;
    }
    return surfaceTexture;
  };
  const texture = (h) => (h >>> 0 === SURFACE
    ? { texture: currentTexture(), format: gpu.format, views: new Map() }
    : objects[h >>> 0]);
  const viewOf = (t, dimension, baseMipLevel, mipLevelCount, baseArrayLayer, arrayLayerCount) => {
    const dim = dimension === '2d' ? 0 : dimension === '2d-array' ? 1 : 2;
    const key = dim + baseMipLevel * 4 + mipLevelCount * 128
      + baseArrayLayer * 4096 + arrayLayerCount * 16777216;
    let v = t.views.get(key);
    if (!v) {
      v = t.texture.createView({ dimension, baseMipLevel, mipLevelCount, baseArrayLayer, arrayLayerCount });
      t.views.set(key, v);
    }
    return v;
  };
  const attachment = (h, mip, layer) => {
    if (h >>> 0 === SURFACE) {
      if (!surfaceView) surfaceView = currentTexture().createView();
      return surfaceView;
    }
    return viewOf(objects[h], '2d', mip, 1, layer, 1);
  };

  // Shared layouts and groups: the empty group and the push-constant ring.
  const emptyLayout = device.createBindGroupLayout({ entries: [] });
  const emptyGroup = device.createBindGroup({ layout: emptyLayout, entries: [] });
  const pushLayout = (visibility) => device.createBindGroupLayout({
    entries: [{ binding: 0, visibility, buffer: { type: 'uniform', hasDynamicOffset: true } }],
  });
  const pushRenderLayout = pushLayout(GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT);
  const pushComputeLayout = pushLayout(GPUShaderStage.COMPUTE);
  // Static pass cache: promotes repeated identical render pass command streams into GPURenderBundles.
  const bundleBuckets = new Map();
  let bundleEntries = 0;
  let bundleCompiled = 0;
  let bundleReplays = 0;
  const clearBundles = () => {
    bundleBuckets.clear();
    bundleEntries = 0;
    bundleCompiled = 0;
  };
  let ring = null;
  const ensureRing = (len) => {
    if (ring && ring.size >= len) return;
    let size = 65536;
    while (size < len) size *= 2;
    if (ring) {
      ring.buffer.destroy();
      clearBundles();
    }
    const buffer = device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const group = (layout) => device.createBindGroup({
      layout, entries: [{ binding: 0, resource: { buffer, offset: 0, size: PUSH_SLOT } }],
    });
    ring = { buffer, size, render: group(pushRenderLayout), compute: group(pushComputeLayout) };
  };
  ensureRing(1);
  const group = (h) => {
    switch (h >>> 0) {
      case EMPTY_GROUP: return emptyGroup;
      case PUSH_RENDER: return ring.render;
      case PUSH_COMPUTE: return ring.compute;
      default: return objects[h >>> 0];
    }
  };

  // Blits and mip generation are render passes.
  const blitModule = device.createShaderModule({ code: BLIT_WGSL });
  const blitPipelines = new Map();
  let nextBlitId = 0;
  const blitSamplers = [device.createSampler({}),
    device.createSampler({ magFilter: 'linear', minFilter: 'linear' })];
  const blitPipeline = (format, linear) => {
    const key = `${format}/${linear}`;
    let p = blitPipelines.get(key);
    if (!p) {
      const layout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT,
            texture: { sampleType: linear ? 'float' : 'unfilterable-float' } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT,
            sampler: { type: linear ? 'filtering' : 'non-filtering' } },
        ],
      });
      const pipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: { module: blitModule, entryPoint: 'vs' },
        fragment: { module: blitModule, entryPoint: 'fs', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
      });
      p = { id: ++nextBlitId, pipeline, layout };
      blitPipelines.set(key, p);
    }
    return p;
  };
  const blit = (enc, src, srcMip, srcLayer, dst, dstMip, dstLayer, linear) => {
    const s = objects[src];
    const dstFormat = dst >>> 0 === SURFACE ? gpu.format : objects[dst].format;
    const p = blitPipeline(dstFormat, linear);
    const bgKey = p.id + srcMip * 256 + srcLayer * 8192;
    let bindGroup = s.blitGroups && s.blitGroups.get(bgKey);
    if (!bindGroup) {
      bindGroup = device.createBindGroup({
        layout: p.layout,
        entries: [
          { binding: 0, resource: viewOf(s, '2d', srcMip, 1, srcLayer, 1) },
          { binding: 1, resource: blitSamplers[linear ? 1 : 0] },
        ],
      });
      if (s.blitGroups) s.blitGroups.set(bgKey, bindGroup);
    }
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: attachment(dst, dstMip, dstLayer), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
    });
    pass.setPipeline(p.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  };

  const canBundle = bundles && typeof device.createRenderBundleEncoder === 'function';
  // Settled, the copy to the canvas is ready before the first frame needs it.
  if (settled) blitPipeline(gpu.format, false);
  // While passes are timed (`timing`): each pass's own begin and end where the GPU has timestamps,
  // otherwise the frame's submission to its completion.
  const timing = createPassTiming();
  let stampable = null;
  const STAMPED_PASSES = 128;
  let stamps = null;
  let stampLabels = [];
  let frameSent = 0;
  const stampWrites = (k) => {
    if (!stamps) {
      const size = STAMPED_PASSES * 16;
      stamps = {
        set: device.createQuerySet({ type: 'timestamp', count: STAMPED_PASSES * 2 }),
        resolve: device.createBuffer({ size, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
        read: device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      };
    }
    return { querySet: stamps.set, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 };
  };
  const wordsEqual = (saved, c, start, len) => {
    for (let k = 0; k < len; k++) {
      if (saved[k] !== c[start + k]) return false;
    }
    return true;
  };
  const recordBundleBody = (rbe, c, start, end) => {
    const u64 = (idx) => c[idx] + c[idx + 1] * 4294967296;
    let p = start;
    while (p < end) {
      const bop = c[p++];
      switch (bop) {
        case 3: rbe.setPipeline(objects[c[p++]]); break;
        case 4: {
          const index = c[p]; const g = c[p + 1]; const n = c[p + 2];
          if (n === 0) rbe.setBindGroup(index, group(g));
          else rbe.setBindGroup(index, group(g), c, p + 3, n);
          p += 3 + n;
          break;
        }
        case 5: rbe.setVertexBuffer(c[p], objects[c[p + 1]], u64(p + 2)); p += 4; break;
        case 6: rbe.setIndexBuffer(objects[c[p]], 'uint32', u64(p + 1)); p += 3; break;
        case 9: rbe.draw(c[p], c[p + 1], c[p + 2], c[p + 3]); p += 4; break;
        case 10: rbe.drawIndexed(c[p], c[p + 1], c[p + 2], c[p + 3] | 0, c[p + 4]); p += 5; break;
        default: throw new Error(`unsupported bundle op ${bop}`);
      }
    }
  };

  const execute = (cmdPtr, words, pushPtr, pushLen) => {
    const c = u32(cmdPtr, words);
    const f = new Float32Array(memory.buffer, cmdPtr >>> 0, words >>> 0);
    pushLen >>>= 0;
    if (pushLen) {
      ensureRing(pushLen);
      device.queue.writeBuffer(ring.buffer, 0, memory.buffer, pushPtr >>> 0, pushLen);
    }
    const enc = device.createCommandEncoder();
    let pass = null;
    const u64 = (idx) => c[idx] + c[idx + 1] * 4294967296;
    const timed = timing.on();
    if (timed && stampable === null) stampable = device.features.has('timestamp-query');
    let passLabel = '';
    let passSize = '';
    let stamp = -1;
    let i = 0;
    while (i < c.length) {
      const op = c[i++];
      switch (op) {
        case 1: { // BEGIN_RENDER
          const colors = c[i++];
          const hasDepth = c[i++];
          if (timed) {
            const t = texture(c[i]).texture;
            passLabel = '';
            passSize = `${Math.max(1, t.width >> c[i + 1])}x${Math.max(1, t.height >> c[i + 1])}`;
            stamp = stampable && stampLabels.length < STAMPED_PASSES ? stampLabels.push('') - 1 : -1;
          }
          const colorAttachments = [];
          const colorFormats = canBundle ? [] : null;
          let h = (2166136261 ^ colors) >>> 0;
          for (let k = 0; k < colors; k++) {
            const th = c[i];
            const fmt = th >>> 0 === SURFACE ? gpu.format : objects[th].format;
            if (colorFormats) colorFormats.push(fmt);
            colorAttachments.push({
              view: attachment(th, c[i + 1], c[i + 2]),
              loadOp: c[i + 3] ? 'clear' : 'load',
              storeOp: c[i + 4] ? 'store' : 'discard',
              clearValue: [f[i + 5], f[i + 6], f[i + 7], f[i + 8]],
            });
            i += 9;
          }
          let depthStencilAttachment;
          let depthStencilFormat;
          if (hasDepth) {
            depthStencilFormat = objects[c[i]].format;
            depthStencilAttachment = {
              view: attachment(c[i], c[i + 1], c[i + 2]),
              depthLoadOp: c[i + 3] ? 'clear' : 'load',
              depthStoreOp: c[i + 4] ? 'store' : 'discard',
              depthClearValue: f[i + 5],
            };
            i += 6;
          }
          if (canBundle && !timed) {
            const bodyStart = i;
            let scan = i;
            let eligible = true;
            let draws = 0;
            while (scan < c.length) {
              const bop = c[scan];
              if (bop === 2) break;
              switch (bop) {
                case 3: scan += 2; break;
                case 4: scan += 4 + c[scan + 3]; break;
                case 5: scan += 5; break;
                case 6: scan += 4; break;
                case 9: draws++; scan += 5; break;
                case 10: draws++; scan += 6; break;
                case 7:
                case 8:
                  eligible = false; scan += 5; break;
                default:
                  eligible = false; scan = c.length; break;
              }
            }
            if (eligible && draws > 0 && scan < c.length && c[scan] === 2) {
              const bodyLen = scan - bodyStart;
              const fmtKey = `${colorFormats.join(',')}|${depthStencilFormat || ''}`;
              for (let k = 0; k < fmtKey.length; k++) {
                h = Math.imul(h ^ fmtKey.charCodeAt(k), 16777619) >>> 0;
              }
              for (let p = bodyStart; p < scan; p++) {
                h = Math.imul(h ^ c[p], 16777619) >>> 0;
              }
              let bucket = bundleBuckets.get(h);
              let entry = null;
              if (bucket) {
                for (let b = 0; b < bucket.length; b++) {
                  const cand = bucket[b];
                  if (cand.fmtKey === fmtKey && cand.words.length === bodyLen
                    && wordsEqual(cand.words, c, bodyStart, bodyLen)) {
                    entry = cand;
                    break;
                  }
                }
              }
              if (entry) {
                if (!entry.bundle) {
                  const rbe = device.createRenderBundleEncoder({
                    colorFormats,
                    depthStencilFormat,
                  });
                  recordBundleBody(rbe, c, bodyStart, scan);
                  entry.bundle = rbe.finish();
                  bundleCompiled++;
                }
                const rp = enc.beginRenderPass({ colorAttachments, depthStencilAttachment });
                rp.executeBundles([entry.bundle]);
                rp.end();
                bundleReplays++;
                i = scan + 1;
                break;
              }
              if (bundleEntries >= 128) clearBundles();
              if (!bucket) {
                bucket = [];
                bundleBuckets.set(h, bucket);
              }
              bucket.push({ fmtKey, words: c.slice(bodyStart, scan), bundle: null });
              bundleEntries++;
            }
          }
          pass = enc.beginRenderPass({
            colorAttachments,
            depthStencilAttachment,
            ...(stamp >= 0 ? { timestampWrites: stampWrites(stamp) } : {}),
          });
          break;
        }
        case 2:
          pass.end();
          pass = null;
          if (stamp >= 0) stampLabels[stamp] = `${passLabel || 'clear'} ${passSize}`;
          stamp = -1;
          break;
        case 3: {
          const p = objects[c[i++]];
          pass.setPipeline(p);
          if (timed && !passLabel) passLabel = p.label;
          break;
        }
        case 4: {
          const index = c[i]; const g = c[i + 1]; const n = c[i + 2];
          if (n === 0) pass.setBindGroup(index, group(g));
          else pass.setBindGroup(index, group(g), c, i + 3, n);
          i += 3 + n;
          break;
        }
        case 5: pass.setVertexBuffer(c[i], objects[c[i + 1]], u64(i + 2)); i += 4; break;
        case 6: pass.setIndexBuffer(objects[c[i]], 'uint32', u64(i + 1)); i += 3; break;
        case 7: pass.setViewport(f[i], f[i + 1], f[i + 2], f[i + 3], 0, 1); i += 4; break;
        case 8: pass.setScissorRect(c[i], c[i + 1], c[i + 2], c[i + 3]); i += 4; break;
        case 9: pass.draw(c[i], c[i + 1], c[i + 2], c[i + 3]); i += 4; break;
        case 10: pass.drawIndexed(c[i], c[i + 1], c[i + 2], c[i + 3] | 0, c[i + 4]); i += 5; break;
        case 11: { // DISPATCH
          const cp = enc.beginComputePass();
          cp.setPipeline(objects[c[i]]);
          const n = c[i + 1];
          i += 2;
          for (let k = 0; k < n; k++) {
            const index = c[i]; const g = c[i + 1]; const no = c[i + 2];
            if (no === 0) cp.setBindGroup(index, group(g));
            else cp.setBindGroup(index, group(g), c, i + 3, no);
            i += 3 + no;
          }
          cp.dispatchWorkgroups(c[i], c[i + 1], c[i + 2]);
          i += 3;
          cp.end();
          break;
        }
        case 12: // COPY_BUFFER
          enc.copyBufferToBuffer(objects[c[i]], u64(i + 1), objects[c[i + 3]], u64(i + 4), u64(i + 6));
          i += 8;
          break;
        case 13: { // COPY_TEXTURE
          const t = texture(c[i]);
          enc.copyTextureToBuffer(
            { texture: t.texture, mipLevel: c[i + 1], origin: [0, 0, c[i + 2]] },
            { buffer: objects[c[i + 5]], offset: u64(i + 6), bytesPerRow: c[i + 8], rowsPerImage: c[i + 4] },
            [c[i + 3], c[i + 4], 1],
          );
          i += 9;
          break;
        }
        case 14: // BLIT
          blit(enc, c[i], c[i + 1], c[i + 2], c[i + 3], c[i + 4], c[i + 5], c[i + 6]);
          i += 7;
          break;
        case 15: // COPY_DEPTH
          enc.copyTextureToTexture(
            { texture: objects[c[i]].texture, mipLevel: c[i + 1], origin: [0, 0, c[i + 2]] },
            { texture: objects[c[i + 3]].texture, mipLevel: c[i + 4], origin: [0, 0, c[i + 5]] },
            [c[i + 6], c[i + 7], 1],
          );
          i += 8;
          break;
        default:
          throw new Error(`unknown command ${op} at word ${i - 1}`);
      }
    }
    if (timed && !stampable && !frameSent) frameSent = performance.now();
    device.queue.submit([enc.finish()]);
  };

  const imports = {
    gpu_info(out, cap) {
      const L = device.limits;
      const features = (device.features.has('depth-clip-control') ? 1 : 0)
        | (device.features.has('float32-filterable') ? 2 : 0)
        | (device.features.has('float32-blendable') ? 4 : 0)
        | (device.features.has('texture-compression-bc') ? 8 : 0)
        | (device.features.has('texture-compression-astc') ? 16 : 0);
      const v = [L.maxTextureDimension2D, L.maxTextureArrayLayers, L.maxBindGroups,
        L.minUniformBufferOffsetAlignment, L.minStorageBufferOffsetAlignment,
        L.maxUniformBufferBindingSize, Math.min(L.maxStorageBufferBindingSize, 0xFFFFFFFF),
        Math.min(L.maxBufferSize, 0xFFFFFFFF), L.maxColorAttachments, features,
        L.maxComputeWorkgroupsPerDimension, L.maxVertexBufferArrayStride, L.maxVertexAttributes,
        0, 0, 0];
      const n = Math.min(v.length, cap >>> 0);
      u32(out, n).set(v.slice(0, n));
      return n;
    },
    gpu_adapter_name: (out, cap) => writeStr(gpu.adapterName, out, cap),
    gpu_last_error: (out, cap) => writeStr(lastError, out, cap),
    gpu_error_count: () => errors.length,
    gpu_error: (i, out, cap) => writeStr(errors[i >>> 0] || '', out, cap),
    gpu_canvas(out) {
      if (!canvas) return 0;
      const code = Math.max(1, FORMATS.indexOf(gpu.format));
      u32(out, 3).set([canvas.width, canvas.height, code]);
      return 1;
    },
    gpu_create_buffer: guard('createBuffer', (size, usage) => put(device.createBuffer({ size, usage }))),
    gpu_write_buffer(h, offset, ptr, len) {
      device.queue.writeBuffer(objects[h >>> 0], offset, memory.buffer, ptr >>> 0, len >>> 0);
    },
    gpu_create_texture: guard('createTexture', (width, height, layers, mips, format, usage) => put({
      texture: device.createTexture({
        size: [width, height, layers], mipLevelCount: mips, format: FORMATS[format], usage,
      }),
      format: FORMATS[format],
      compressed: format >= 12,
      views: new Map(),
      blitGroups: new Map(),
    })),
    gpu_write_texture(h, mip, layer, width, height, bytesPerRow, ptr, len) {
      const t = objects[h >>> 0];
      const rowsPerImage = t.compressed ? Math.max(1, height >>> 2) : height;
      device.queue.writeTexture(
        { texture: t.texture, mipLevel: mip, origin: [0, 0, layer] },
        memory.buffer,
        { offset: ptr >>> 0, bytesPerRow, rowsPerImage },
        [width, height, 1],
      );
    },
    gpu_copy_source: guard('copyExternalImageToTexture', (h, mip, layer, width, height) => {
      // Straight alpha and sRGB-encoded values, the image's top row first: the bytes pixelsOf
      // reads back (an -srgb texture stores the same bytes as its unorm twin).
      device.queue.copyExternalImageToTexture(
        { source: sourceOfSize(source, width, height), flipY: false },
        { texture: objects[h >>> 0].texture, mipLevel: mip, origin: [0, 0, layer], colorSpace: 'srgb', premultipliedAlpha: false },
        [width, height],
      );
      return 1;
    }),
    gpu_generate_mips(h) {
      const t = objects[h >>> 0];
      const enc = device.createCommandEncoder();
      for (let layer = 0; layer < t.texture.depthOrArrayLayers; layer++) {
        for (let mip = 1; mip < t.texture.mipLevelCount; mip++) {
          blit(enc, h, mip - 1, layer, h, mip, layer, 1);
        }
      }
      device.queue.submit([enc.finish()]);
    },
    gpu_create_sampler: guard('createSampler', (ptr, words) => {
      const d = u32(ptr, words);
      const lod = new Float32Array(memory.buffer, (ptr >>> 0) + 20, 1)[0];
      const desc = {
        magFilter: d[0] ? 'linear' : 'nearest',
        minFilter: d[0] ? 'linear' : 'nearest',
        mipmapFilter: d[1] ? 'linear' : 'nearest',
        addressModeU: ADDRESS[d[2]],
        addressModeV: ADDRESS[d[2]],
        addressModeW: ADDRESS[d[2]],
        maxAnisotropy: d[3],
        lodMaxClamp: lod,
      };
      if (d[4]) desc.compare = COMPARES[d[4]];
      return put(device.createSampler(desc));
    }),
    gpu_create_shader: guard('createShaderModule', (ptr, len) => {
      const code = str(ptr, len);
      // The entry point it was generated from names the passes it draws in (`timing`).
      const module = device.createShaderModule({ code, label: /from '([^']+)'/.exec(code)?.[1] });
      module.getCompilationInfo().then((info) => {
        const lines = code.split('\n');
        for (const m of info.messages) {
          const where = m.lineNum ? `${m.lineNum}:${m.linePos} ${lines[m.lineNum - 1].trim()}` : '';
          const text = `WGSL ${m.type}: ${m.message} ${where}`;
          if (m.type === 'error') report(text);
          else onLog(1, text);
        }
      });
      return put(module);
    }),
    gpu_create_bind_group_layout: guard('createBindGroupLayout', (ptr, words) => {
      const w = u32(ptr, words);
      const entries = [];
      for (let i = 0; i < w.length; i += 6) {
        const [binding, visibility, type, a, b] = w.subarray(i, i + 5);
        const e = { binding, visibility };
        switch (type) {
          case 0: e.buffer = { type: 'uniform', minBindingSize: b }; break;
          case 1: e.buffer = { type: 'read-only-storage' }; break;
          case 2: e.buffer = { type: 'storage' }; break;
          case 3: e.sampler = { type: a === 2 ? 'comparison' : a === 1 ? 'non-filtering' : 'filtering' }; break;
          case 4: e.texture = { sampleType: SAMPLE_TYPES[a], viewDimension: VIEW_DIMENSIONS[b] }; break;
          case 5: e.storageTexture = { access: 'write-only', format: FORMATS[a], viewDimension: VIEW_DIMENSIONS[b] }; break;
          default: throw new Error(`binding type ${type}`);
        }
        entries.push(e);
      }
      return put(device.createBindGroupLayout({ entries }));
    }),
    gpu_create_pipeline_layout: guard('createPipelineLayout', (ptr, count) => {
      const bindGroupLayouts = Array.from(u32(ptr, count), (h) => {
        switch (h) {
          case 0: return emptyLayout;
          case PUSH_RENDER: return pushRenderLayout;
          case PUSH_COMPUTE: return pushComputeLayout;
          default: return objects[h];
        }
      });
      return put(device.createPipelineLayout({ bindGroupLayouts }));
    }),
    gpu_create_render_pipeline: guard('createRenderPipeline', (ptr, len) => {
      const d = JSON.parse(str(ptr, len));
      d.layout = objects[d.layout];
      d.vertex.module = objects[d.vertex.module];
      if (d.fragment) d.fragment.module = objects[d.fragment.module];
      // Its fragment (or vertex) shader names the passes it draws in (`timing`).
      d.label = (d.fragment || d.vertex).module.label;
      return put(device.createRenderPipeline(d));
    }),
    gpu_create_compute_pipeline: guard('createComputePipeline', (ptr, len) => {
      const d = JSON.parse(str(ptr, len));
      d.layout = objects[d.layout];
      d.compute.module = objects[d.compute.module];
      return put(device.createComputePipeline(d));
    }),
    gpu_create_bind_group: guard('createBindGroup', (layout, ptr, words) => {
      const w = u32(ptr, words);
      const entries = [];
      for (let i = 0; i < w.length; i += 8) {
        const [binding, kind, h, a, b, c, d, e] = w.subarray(i, i + 8);
        let resource;
        if (kind === 0) {
          resource = { buffer: objects[h], offset: a + b * 4294967296, size: c + d * 4294967296 };
        } else if (kind === 1) {
          resource = viewOf(objects[h], VIEW_DIMENSIONS[a], b, c, d, e);
        } else {
          resource = objects[h];
        }
        entries.push({ binding, resource });
      }
      return put(device.createBindGroup({ layout: objects[layout >>> 0], entries }));
    }),
    gpu_submit: guard('submit', (cmdPtr, words, pushPtr, pushLen) => {
      execute(cmdPtr, words, pushPtr, pushLen);
      return 1;
    }),
    gpu_release(h) {
      h >>>= 0;
      const o = objects[h];
      if (!o) return;
      clearBundles();
      if (o.texture && o.texture.destroy) o.texture.destroy();
      else if (o instanceof GPUBuffer) o.destroy();
      objects[h] = null;
      freeList.push(h);
    },
    log: (level, ptr, len) => onLog(level, str(ptr, len)),
    status: (ptr, len) => onStatus(str(ptr, len)),
    gpu_api: () => 1,
  };
  // The WebGL 2 imports exist in the module too; on this host they are never reached.
  const unavailable = () => { throw new Error('WebGL 2 call on the WebGPU host'); };
  for (const name of GL_IMPORTS) imports[name] = unavailable;

  return {
    imports: { ganbate: imports },
    errors,
    attach(instance) { memory = instance.exports.memory; },
    /** Sets (or with null clears) the image the next runtime call may copy into a texture. */
    setSource(image) { source = image; },
    /** Forgets the canvas texture of the last frame (call once per animation frame). */
    nextFrame() { surfaceTexture = null; surfaceView = null; },
    /** The canvas texture of the current frame (settled, its offscreen texture), if one was rendered. */
    frameTexture() { return surfaceTexture; },
    /** Copies a finished frame onto the canvas (settled), stretched if the canvas was resized. */
    present(frame) {
      const target = context.getCurrentTexture();
      const linear = frame.width !== target.width || frame.height !== target.height;
      const p = blitPipeline(gpu.format, linear);
      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      });
      pass.setPipeline(p.pipeline);
      let groups = presentGroups.get(frame);
      if (!groups) presentGroups.set(frame, groups = new Map());
      let group = groups.get(p.id);
      if (!group) {
        group = device.createBindGroup({
          layout: p.layout,
          entries: [
            { binding: 0, resource: frame.createView() },
            { binding: 1, resource: blitSamplers[linear ? 1 : 0] },
          ],
        });
        groups.set(p.id, group);
      }
      pass.setBindGroup(0, group);
      pass.draw(3);
      pass.end();
      device.queue.submit([enc.finish()]);
      if (frame === stage) while (retired.length) retired.pop().destroy();
    },
    /** Releases the settled host's frame textures. */
    release() {
      while (retired.length) retired.pop().destroy();
      stage?.destroy();
      stage = null;
    },
    /** Telemetry for cached GPURenderBundle passes. */
    bundleStats: () => ({ cached: bundleCompiled, candidates: bundleEntries, replays: bundleReplays }),
    timing,
    /** Closes a drawn frame's timing: booked once its timestamps are read, or the GPU finished it. */
    endFrame() {
      if (!timing.on()) return;
      if (!stampLabels.length) {
        const m = timing.mark('frame');
        m.sent = frameSent || m.sent;
        frameSent = 0;
        const frame = timing.take();
        const done = () => { m.done = performance.now(); frame.finish(); };
        device.queue.onSubmittedWorkDone().then(done, done);
        return;
      }
      const labels = stampLabels;
      stampLabels = [];
      for (const label of labels) timing.mark(label);
      const frame = timing.take();
      const bytes = labels.length * 16;
      const enc = device.createCommandEncoder();
      enc.resolveQuerySet(stamps.set, 0, labels.length * 2, stamps.resolve, 0);
      enc.copyBufferToBuffer(stamps.resolve, 0, stamps.read, 0, bytes);
      device.queue.submit([enc.finish()]);
      stamps.read.mapAsync(GPUMapMode.READ, 0, bytes).then(() => {
        const t = new BigInt64Array(stamps.read.getMappedRange(0, bytes));
        frame.marks.forEach((m, k) => {
          m.sent = Number(t[2 * k] - t[0]) / 1e6;
          m.done = Number(t[2 * k + 1] - t[0]) / 1e6;
        });
        stamps.read.unmap();
        frame.finish();
      }, frame.finish);
    },
    setXrTarget: () => false,
  };
}


// ---------------------------------------------------------------------------------- WebGL 2
//
// The fallback host for browsers without WebGPU: provides the gl_* imports of
// src/gpu/webgl/ffi.rs and executes the command streams of src/gpu/webgl/record.rs. Shaders
// arrive as GLSL ES 3.00 from ganbate-shader's GLSL emitter, which keeps Vulkan's clip space apart
// from the depth range: images lie in GL memory with row 0 at the top, so the canvas texture
// is flipped when it is copied to the canvas.

const GL_SURFACE = 0xFFFFFFFF;
const ROW_TEXELS = 1024; // ganbate_shader::glsl::STORAGE_ROW_TEXELS
const ROW_BYTES = ROW_TEXELS * 16;
const BUF_DATA = 1;
const BUF_INDEX = 2;
const BUF_STORAGE = 4;
const BUF_READBACK = 8;

const BLIT_VS = `#version 300 es
out vec2 uv;
void main() {
  uv = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}`;
const blitFs = (array) => `#version 300 es
precision highp float;
precision highp sampler2DArray;
uniform highp ${array ? 'sampler2DArray' : 'sampler2D'} src;
uniform float layer;
in vec2 uv;
out vec4 color;
void main() { color = ${array ? 'texture(src, vec3(uv, layer))' : 'texture(src, uv)'}; }`;

/** Creates a WebGL 2 context on `canvas` (or an offscreen one) with the extensions Ganbate uses. */
export function createGl(canvas) {
  const target = canvas || (typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(1, 1) : document.createElement('canvas'));
  const gl = target.getContext('webgl2', {
    alpha: false, antialias: false, depth: false, stencil: false, premultipliedAlpha: false,
    preserveDrawingBuffer: false, powerPreference: 'high-performance',
  });
  if (!gl) {
    throw new Error('This browser has neither WebGPU nor WebGL 2.');
  }
  const ext = {
    colorFloat: gl.getExtension('EXT_color_buffer_float'),
    floatLinear: gl.getExtension('OES_texture_float_linear'),
    floatBlend: gl.getExtension('EXT_float_blend'),
    aniso: gl.getExtension('EXT_texture_filter_anisotropic'),
    provoking: gl.getExtension('WEBGL_provoking_vertex'),
    indexed: gl.getExtension('OES_draw_buffers_indexed'),
    s3tc: gl.getExtension('WEBGL_compressed_texture_s3tc') || gl.getExtension('EXT_texture_compression_s3tc'),
    s3tcSrgb: gl.getExtension('WEBGL_compressed_texture_s3tc_srgb') || gl.getExtension('EXT_texture_compression_s3tc_srgb'),
    bptc: gl.getExtension('EXT_texture_compression_bptc'),
    astc: gl.getExtension('WEBGL_compressed_texture_astc'),
  };
  // Flat varyings come from the first vertex, as on Vulkan and WebGPU.
  if (ext.provoking) ext.provoking.provokingVertexWEBGL(ext.provoking.FIRST_VERTEX_CONVENTION_WEBGL);
  const debug = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  const features = Object.entries(ext).filter(([, v]) => v).map(([k]) => k);
  return { api: 'webgl2', gl, ext, canvas, adapterName: `WebGL 2 ${renderer}`, features, format: 'rgba8unorm' };
}

/** The import object for the wasm module on WebGL 2 and the host state behind it. */
export function createGlHost(gpu, { onLog = () => {}, onStatus = () => {}, onError = () => {}, checkErrors = false } = {}) {
  const { gl, ext, canvas } = gpu;
  const objects = [null];
  const freeList = [];
  const errors = [];
  let lastError = '';
  let memory = null;
  /** The image `gl_write_texture_source` uploads: set by the engine around one runtime call. */
  let source = null;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const report = (message) => {
    errors.push(message);
    onError(message);
  };
  if (canvas && canvas.addEventListener) {
    canvas.addEventListener('webglcontextlost', () => report('WebGL 2 context lost'));
  }
  const put = (obj) => {
    const h = freeList.length ? freeList.pop() : objects.length;
    objects[h] = obj;
    return h;
  };
  const u8 = (ptr, len) => new Uint8Array(memory.buffer, ptr >>> 0, len >>> 0);
  const u32 = (ptr, n) => new Uint32Array(memory.buffer, ptr >>> 0, n >>> 0);
  const str = (ptr, len) => decoder.decode(u8(ptr, len));
  const writeStr = (s, out, cap) => {
    const bytes = encoder.encode(s);
    const n = Math.min(bytes.length, cap >>> 0);
    u8(out, n).set(bytes.subarray(0, n));
    return n;
  };
  const guard = (what, f) => (...args) => {
    try {
      return f(...args);
    } catch (e) {
      lastError = `${what}: ${e.message}`;
      return 0;
    }
  };
  const glErrors = (what) => {
    if (!checkErrors) return;
    for (let e = gl.getError(), n = 0; e !== gl.NO_ERROR && n < 8; e = gl.getError(), n++) {
      report(`WebGL error 0x${e.toString(16)} in ${what}`);
    }
  };

  const FORMATS = [null,
    { internal: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE, size: 4 },
    { internal: gl.SRGB8_ALPHA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE, size: 4 },
    { internal: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE, size: 4, bgra: true },
    { internal: gl.SRGB8_ALPHA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE, size: 4, bgra: true },
    { internal: gl.RGBA16F, format: gl.RGBA, type: gl.HALF_FLOAT, size: 8, channels: 4, half: true },
    { internal: gl.RG16F, format: gl.RG, type: gl.HALF_FLOAT, size: 4, channels: 2, half: true },
    { internal: gl.R16F, format: gl.RED, type: gl.HALF_FLOAT, size: 2, channels: 1, half: true },
    { internal: gl.R8, format: gl.RED, type: gl.UNSIGNED_BYTE, size: 1, channels: 1, unorm: true },
    { internal: gl.R32F, format: gl.RED, type: gl.FLOAT, size: 4, channels: 1, float: true },
    { internal: gl.RGBA32F, format: gl.RGBA, type: gl.FLOAT, size: 16, channels: 4, float: true },
    { internal: gl.DEPTH_COMPONENT32F, format: gl.DEPTH_COMPONENT, type: gl.FLOAT, size: 4, depth: true },
    { internal: 0x83F1, compressed: true }, // COMPRESSED_RGBA_S3TC_DXT1_EXT
    { internal: 0x8C4F, compressed: true }, // COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT
    { internal: 0x8E8C, compressed: true }, // COMPRESSED_RGBA_BPTC_UNORM_EXT
    { internal: 0x8E8D, compressed: true }, // COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT
    { internal: 0x93B0, compressed: true }, // COMPRESSED_RGBA_ASTC_4x4_KHR
    { internal: 0x93D0, compressed: true }, // COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR
  ];
  const COMPARES = [0, gl.NEVER, gl.LESS, gl.LEQUAL, gl.EQUAL, gl.GREATER, gl.GEQUAL, gl.NOTEQUAL, gl.ALWAYS];
  const WRAP = [gl.REPEAT, gl.CLAMP_TO_EDGE, gl.MIRRORED_REPEAT];
  const MODES = [gl.TRIANGLES, gl.TRIANGLE_STRIP, gl.LINES];

  // A texture unit the stream never uses, for uploads and mip generation.
  const scratchUnit = gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS) - 1;
  let uid = 0;
  const textures = new Set();
  const bindScratch = (t) => {
    gl.activeTexture(gl.TEXTURE0 + scratchUnit);
    gl.bindTexture(t.target, t.tex);
  };
  const setLevels = (t, base, count) => {
    if (t.base === base && t.count === count) return;
    gl.texParameteri(t.target, gl.TEXTURE_BASE_LEVEL, base);
    gl.texParameteri(t.target, gl.TEXTURE_MAX_LEVEL, base + count - 1);
    t.base = base;
    t.count = count;
  };
  const makeTexture = (w, h, layers, mips, f, array) => {
    const target = array ? gl.TEXTURE_2D_ARRAY : gl.TEXTURE_2D;
    const t = { tex: gl.createTexture(), target, f, w, h, layers, mips, base: 0, count: mips, uid: ++uid };
    bindScratch(t);
    if (array) gl.texStorage3D(target, mips, f.internal, w, h, layers);
    else gl.texStorage2D(target, mips, f.internal, w, h);
    textures.add(t);
    return t;
  };

  // The canvas texture (RGBA8) and its copy to the canvas.
  let surface = null;
  const ensureSurface = () => {
    if (!canvas) return null;
    const w = canvas.width; const h = canvas.height;
    if (w === 0 || h === 0) return null;
    if (!surface || surface.w !== w || surface.h !== h) {
      if (surface) dropTexture(surface);
      surface = makeTexture(w, h, 1, 1, FORMATS[1], false);
    }
    return surface;
  };
  const texture = (h) => (h >>> 0 === GL_SURFACE ? ensureSurface() : objects[h >>> 0]);

  // Framebuffers, cached by attachments.
  const fbos = new Map();
  const attach = (point, t, mip, layer) => {
    if (t.target === gl.TEXTURE_2D_ARRAY) gl.framebufferTextureLayer(gl.FRAMEBUFFER, point, t.tex, mip, layer);
    else gl.framebufferTexture2D(gl.FRAMEBUFFER, point, gl.TEXTURE_2D, t.tex, mip);
  };
  const framebuffer = (colors, depth) => {
    const key = [...colors, depth].map((a) => (a ? `${a.t.uid}.${a.mip}.${a.layer}` : '-')).join('|');
    let f = fbos.get(key);
    if (!f) {
      f = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      colors.forEach((a, i) => attach(gl.COLOR_ATTACHMENT0 + i, a.t, a.mip, a.layer));
      if (depth) attach(gl.DEPTH_ATTACHMENT, depth.t, depth.mip, depth.layer);
      gl.drawBuffers(colors.length ? colors.map((_, i) => gl.COLOR_ATTACHMENT0 + i) : [gl.NONE]);
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        gl.deleteFramebuffer(f);
        throw new Error(`framebuffer incomplete (0x${status.toString(16)}) for ${key}`);
      }
      f.uids = [...colors, depth].filter(Boolean).map((a) => a.t.uid);
      fbos.set(key, f);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    }
    return f;
  };
  const dropTexture = (t) => {
    for (const [k, f] of fbos) {
      if (f.uids.includes(t.uid)) { gl.deleteFramebuffer(f); fbos.delete(k); }
    }
    gl.deleteTexture(t.tex);
    textures.delete(t);
  };
  const readFbo = gl.createFramebuffer();

  // Fixed-function state, applied by difference with a stable hidden class (no property deletion).
  const fixed = {
    [gl.CULL_FACE]: undefined,
    [gl.DEPTH_TEST]: undefined,
    [gl.POLYGON_OFFSET_FILL]: undefined,
    [gl.BLEND]: undefined,
    [gl.SCISSOR_TEST]: undefined,
    cullFace: undefined,
    frontFace: undefined,
    depthFunc: undefined,
    depthMask: undefined,
    bias: undefined,
    slope: undefined,
    drawBuffers: undefined,
    blendMode: undefined,
    colorMask: undefined,
  };
  const setFlag = (cap, on) => {
    if (fixed[cap] !== on) {
      if (on) gl.enable(cap); else gl.disable(cap);
      fixed[cap] = on;
    }
  };
  const resetFixed = () => {
    fixed[gl.CULL_FACE] = undefined;
    fixed[gl.DEPTH_TEST] = undefined;
    fixed[gl.POLYGON_OFFSET_FILL] = undefined;
    fixed[gl.BLEND] = undefined;
    fixed[gl.SCISSOR_TEST] = undefined;
    fixed.cullFace = undefined;
    fixed.frontFace = undefined;
    fixed.depthFunc = undefined;
    fixed.depthMask = undefined;
    fixed.bias = undefined;
    fixed.slope = undefined;
    fixed.drawBuffers = undefined;
    fixed.blendMode = undefined;
    fixed.colorMask = undefined;
  };
  let current = null; // pipeline
  let currentProgram = null;
  const applyPipeline = (p) => {
    if (currentProgram !== p.program.program) {
      gl.useProgram(p.program.program);
      currentProgram = p.program.program;
    }
    current = p;
    setFlag(gl.CULL_FACE, p.cull !== 0);
    if (p.cull !== 0 && fixed.cullFace !== p.cull) { gl.cullFace(p.cull === 1 ? gl.BACK : gl.FRONT); fixed.cullFace = p.cull; }
    // Vulkan counter-clockwise is clockwise in GL window terms (see the module comment).
    const front = p.frontCcw ? gl.CW : gl.CCW;
    if (fixed.frontFace !== front) { gl.frontFace(front); fixed.frontFace = front; }
    setFlag(gl.DEPTH_TEST, p.depthTest);
    if (p.depthTest) {
      if (fixed.depthFunc !== p.depthFunc) { gl.depthFunc(p.depthFunc); fixed.depthFunc = p.depthFunc; }
      if (fixed.depthMask !== p.depthWrite) { gl.depthMask(p.depthWrite); fixed.depthMask = p.depthWrite; }
    }
    const offset = p.bias !== 0 || p.slope !== 0;
    setFlag(gl.POLYGON_OFFSET_FILL, offset);
    if (offset && (fixed.bias !== p.bias || fixed.slope !== p.slope)) {
      gl.polygonOffset(p.slope, p.bias);
      fixed.bias = p.bias; fixed.slope = p.slope;
    }
    // Targets the pipeline does not write are left out of the draw buffers.
    if (p.targets.length && fixed.drawBuffers !== p.drawBuffers) {
      gl.drawBuffers(p.targets.map((t, i) => (t.mask ? gl.COLOR_ATTACHMENT0 + i : gl.NONE)));
      fixed.drawBuffers = p.drawBuffers;
    }
    if (p.indexed) {
      // Per-target blend state and masks (OES_draw_buffers_indexed).
      const x = ext.indexed;
      p.targets.forEach((t, i) => {
        if (t.blend) x.enableiOES(gl.BLEND, i); else x.disableiOES(gl.BLEND, i);
        if (t.blend) blendFunc((...a) => x.blendFuncSeparateiOES(i, ...a), t.blend);
        x.colorMaskiOES(i, !!(t.mask & 1), !!(t.mask & 2), !!(t.mask & 4), !!(t.mask & 8));
      });
      fixed[gl.BLEND] = undefined;
      fixed.blendMode = undefined;
      fixed.colorMask = undefined;
      return;
    }
    setFlag(gl.BLEND, p.blend !== 0);
    if (p.blend !== 0 && fixed.blendMode !== p.blend) {
      blendFunc((...a) => gl.blendFuncSeparate(...a), p.blend);
      gl.blendEquation(gl.FUNC_ADD);
      fixed.blendMode = p.blend;
    }
    if (fixed.colorMask !== p.mask) {
      gl.colorMask(!!(p.mask & 1), !!(p.mask & 2), !!(p.mask & 4), !!(p.mask & 8));
      fixed.colorMask = p.mask;
    }
  };
  const blendFunc = (set, mode) => {
    if (mode === 1) set(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    else if (mode === 2) set(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    else set(gl.ONE, gl.ONE, gl.ONE, gl.ONE);
  };

  // Samplers: the host's nearest sampler for units without one (texelFetch, storage).
  const nearest = gl.createSampler();
  gl.samplerParameteri(nearest, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.samplerParameteri(nearest, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.samplerParameteri(nearest, w, gl.CLAMP_TO_EDGE);
  const linear = gl.createSampler();
  gl.samplerParameteri(linear, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.samplerParameteri(linear, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.samplerParameteri(linear, w, gl.CLAMP_TO_EDGE);

  // Programs.
  const compile = (type, source) => {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, source);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh) || '';
      gl.deleteShader(sh);
      const lines = source.split('\n');
      const where = [...log.matchAll(/ERROR: \d+:(\d+)/g)].slice(0, 3)
        .map((m) => `  ${m[1]}: ${(lines[Number(m[1]) - 1] || '').trim()}`).join('\n');
      throw new Error(`${type === gl.VERTEX_SHADER ? 'vertex' : 'fragment'} shader: ${log}${where ? `\n${where}` : ''}`);
    }
    return sh;
  };
  const link = (vs, fs) => {
    const p = gl.createProgram();
    const v = compile(gl.VERTEX_SHADER, vs);
    const f = compile(gl.FRAGMENT_SHADER, fs);
    gl.attachShader(p, v);
    gl.attachShader(p, f);
    gl.linkProgram(p);
    gl.deleteShader(v);
    gl.deleteShader(f);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(p);
      gl.deleteProgram(p);
      throw new Error(`link: ${log}`);
    }
    return p;
  };
  const blitPrograms = [false, true].map((array) => {
    const program = link(BLIT_VS, blitFs(array));
    return { program, src: gl.getUniformLocation(program, 'src'), layer: gl.getUniformLocation(program, 'layer') };
  });
  const emptyVao = gl.createVertexArray();

  // The push-constant ring.
  const ring = gl.createBuffer();
  let ringSize = 0;

  // Buffers.
  const bufferTarget = (b) => (b.index ? gl.ELEMENT_ARRAY_BUFFER : gl.COPY_WRITE_BUFFER);
  const bindForWrite = (b) => {
    if (b.index) {
      gl.bindVertexArray(null);
      currentVao = null;
    }
    gl.bindBuffer(bufferTarget(b), b.gl);
  };
  const uploadRows = (b, offset, len) => {
    if (!len) return;
    const first = Math.floor(offset / ROW_BYTES);
    const last = Math.floor((offset + len - 1) / ROW_BYTES);
    gl.activeTexture(gl.TEXTURE0 + scratchUnit);
    gl.bindTexture(gl.TEXTURE_2D, b.tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, first, ROW_TEXELS, last - first + 1, gl.RGBA_INTEGER, gl.UNSIGNED_INT,
      new Uint32Array(b.shadow.buffer, first * ROW_BYTES, (last - first + 1) * ROW_TEXELS * 4));
  };
  /** The bytes of a buffer range (read back from GL when the buffer has no copy here). */
  const bytesOf = (b, offset, size) => {
    if (b.shadow) return b.shadow.subarray(offset, offset + size);
    if (b.host) return b.host.subarray(offset, offset + size);
    const out = new Uint8Array(size);
    gl.bindBuffer(gl.COPY_READ_BUFFER, b.gl);
    gl.getBufferSubData(gl.COPY_READ_BUFFER, offset, out);
    return out;
  };
  const writeBytes = (b, offset, data) => {
    if (b.gl) {
      bindForWrite(b);
      gl.bufferSubData(bufferTarget(b), offset, data);
    }
    if (b.shadow) {
      b.shadow.set(data, offset);
      uploadRows(b, offset, data.length);
    }
    if (b.host) b.host.set(data, offset);
  };

  // Half floats, for reading float textures back in their own format.
  const f32 = new Float32Array(1);
  const i32 = new Uint32Array(f32.buffer);
  const toHalf = (v) => {
    f32[0] = v;
    const x = i32[0];
    const sign = (x >>> 16) & 0x8000;
    const e = (x >>> 23) & 0xff;
    const m = x & 0x7fffff;
    if (e === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
    const he = e - 127 + 15;
    if (he >= 0x1f) return sign | 0x7c00;
    if (he <= 0) {
      if (he < -10) return sign;
      const mm = (m | 0x800000) >> (1 - he);
      return sign | ((mm + 0x1000) >> 13);
    }
    return sign | (he << 10) | ((m + 0x1000) >> 13);
  };

  let currentVao = null;

  // While passes are timed (`timing`): a fence after each, flushed so the GPU starts on it, and
  // polled from later tasks (WebGL signals a fence only between tasks).
  const timing = createPassTiming();
  let channel = null;
  const waiting = [];
  let passLabel = '';
  let passSize = '';
  // A pass that draws with the pipeline the one before it left applied sets none of its own.
  let lastPipeline = '';
  let passDrew = false;
  const glMark = (label) => {
    const m = timing.mark(label);
    m.sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    m.sent = performance.now();
  };
  const poll = () => {
    while (waiting.length) {
      const frame = waiting[0];
      const m = frame.marks.find((x) => !x.done);
      if (!m) {
        waiting.shift();
        frame.finish();
        continue;
      }
      if (gl.clientWaitSync(m.sync, 0, 0) === gl.TIMEOUT_EXPIRED) {
        channel.port2.postMessage(0);
        return;
      }
      m.done = performance.now();
      gl.deleteSync(m.sync);
    }
  };

  const blit = (src, srcMip, srcLayer, dst, dstMip, dstLayer, filterLinear) => {
    const s = texture(src);
    const d = texture(dst);
    const w = Math.max(1, d.w >> dstMip); const h = Math.max(1, d.h >> dstMip);
    framebuffer([{ t: d, mip: dstMip, layer: dstLayer }], null);
    const bp = blitPrograms[s.target === gl.TEXTURE_2D_ARRAY ? 1 : 0];
    gl.useProgram(bp.program);
    currentProgram = bp.program;
    gl.uniform1i(bp.src, 0);
    gl.uniform1f(bp.layer, srcLayer);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(s.target, s.tex);
    setLevels(s, srcMip, 1);
    gl.bindSampler(0, filterLinear ? linear : nearest);
    gl.bindVertexArray(emptyVao);
    currentVao = emptyVao;
    resetFixed();
    for (const cap of [gl.DEPTH_TEST, gl.BLEND, gl.CULL_FACE, gl.SCISSOR_TEST, gl.POLYGON_OFFSET_FILL]) gl.disable(cap);
    gl.colorMask(true, true, true, true);
    gl.viewport(0, 0, w, h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    current = null;
  };

  const execute = (cmdPtr, words, pushPtr, pushLen) => {
    const c = u32(cmdPtr, words);
    const f = new Float32Array(memory.buffer, cmdPtr >>> 0, words >>> 0);
    pushLen >>>= 0;
    if (pushLen) {
      gl.bindBuffer(gl.UNIFORM_BUFFER, ring);
      if (pushLen > ringSize) {
        ringSize = Math.max(pushLen, 65536);
        gl.bufferData(gl.UNIFORM_BUFFER, ringSize, gl.DYNAMIC_DRAW);
      }
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, u8(pushPtr, pushLen));
    }
    let discard = [];
    let i = 0;
    while (i < c.length) {
      const op = c[i++];
      switch (op) {
        case 1: { // BEGIN_PASS
          const colors = c[i++];
          const hasDepth = c[i++];
          const cs = [];
          for (let k = 0; k < colors; k++) {
            cs.push({ t: texture(c[i]), mip: c[i + 1], layer: c[i + 2], load: c[i + 3], store: c[i + 4],
              clear: [f[i + 5], f[i + 6], f[i + 7], f[i + 8]] });
            i += 9;
          }
          let depth = null;
          if (hasDepth) {
            depth = { t: texture(c[i]), mip: c[i + 1], layer: c[i + 2], load: c[i + 3], store: c[i + 4], clear: f[i + 5] };
            i += 6;
          }
          framebuffer(cs, depth);
          gl.drawBuffers(cs.length ? cs.map((_, k) => gl.COLOR_ATTACHMENT0 + k) : [gl.NONE]);
          fixed.drawBuffers = cs.map(() => 1).join('');
          const a = cs[0] || depth;
          gl.viewport(0, 0, Math.max(1, a.t.w >> a.mip), Math.max(1, a.t.h >> a.mip));
          passLabel = '';
          passDrew = false;
          passSize = `${Math.max(1, a.t.w >> a.mip)}x${Math.max(1, a.t.h >> a.mip)}`;
          setFlag(gl.SCISSOR_TEST, false);
          // Clears obey the write masks.
          const dontCare = [];
          cs.forEach((x, k) => {
            if (x.load === 1) {
              if (fixed.colorMask !== 15) { gl.colorMask(true, true, true, true); fixed.colorMask = 15; }
              gl.clearBufferfv(gl.COLOR, k, x.clear);
            } else if (x.load === 2) dontCare.push(gl.COLOR_ATTACHMENT0 + k);
          });
          if (depth) {
            if (depth.load === 1) {
              if (fixed.depthMask !== true) { gl.depthMask(true); fixed.depthMask = true; }
              gl.clearBufferfv(gl.DEPTH, 0, [depth.clear]);
            } else if (depth.load === 2) dontCare.push(gl.DEPTH_ATTACHMENT);
          }
          if (dontCare.length) gl.invalidateFramebuffer(gl.FRAMEBUFFER, dontCare);
          discard = cs.map((x, k) => (x.store ? null : gl.COLOR_ATTACHMENT0 + k)).filter((x) => x !== null);
          if (depth && !depth.store) discard.push(gl.DEPTH_ATTACHMENT);
          break;
        }
        case 2: // END_PASS
          if (discard.length) gl.invalidateFramebuffer(gl.FRAMEBUFFER, discard);
          discard = [];
          if (timing.on()) glMark(`${passLabel || (passDrew ? lastPipeline : 'clear')} ${passSize}`);
          break;
        case 3: {
          const p = objects[c[i++]];
          applyPipeline(p);
          lastPipeline = p.program.name;
          if (!passLabel) passLabel = lastPipeline;
          break;
        }
        case 4: gl.viewport(Math.round(f[i]), Math.round(f[i + 1]), Math.round(f[i + 2]), Math.round(f[i + 3])); i += 4; break;
        case 5:
          setFlag(gl.SCISSOR_TEST, true);
          gl.scissor(c[i], c[i + 1], c[i + 2], c[i + 3]);
          i += 4;
          break;
        case 6: gl.bindBufferRange(gl.UNIFORM_BUFFER, c[i], objects[c[i + 1]].gl, c[i + 2], c[i + 3]); i += 4; break;
        case 7: gl.bindBufferRange(gl.UNIFORM_BUFFER, 0, ring, c[i], c[i + 1]); i += 2; break;
        case 8: { // BIND_UNIT
          const t = texture(c[i + 1]);
          gl.activeTexture(gl.TEXTURE0 + c[i]);
          gl.bindTexture(t.target, t.tex);
          setLevels(t, c[i + 3], c[i + 4]);
          gl.bindSampler(c[i], c[i + 2] ? objects[c[i + 2]] : nearest);
          i += 5;
          break;
        }
        case 9: { // BIND_STORAGE
          gl.activeTexture(gl.TEXTURE0 + c[i]);
          gl.bindTexture(gl.TEXTURE_2D, objects[c[i + 1]].tex);
          gl.bindSampler(c[i], nearest);
          i += 2;
          break;
        }
        case 10: { // SET_HIDDEN
          const h = current.program.hidden[c[i]];
          if (h && h.loc) {
            if (h.uint) gl.uniform1ui(h.loc, c[i + 1]);
            else gl.uniform1i(h.loc, c[i + 1] | 0);
          }
          i += 2;
          break;
        }
        case 11: {
          const v = objects[c[i++]].vao;
          if (currentVao !== v) { gl.bindVertexArray(v); currentVao = v; }
          break;
        }
        case 12: gl.drawArraysInstanced(current.mode, c[i + 2], c[i], c[i + 1]); i += 3; passDrew = true; break;
        case 13: gl.drawElementsInstanced(current.mode, c[i], gl.UNSIGNED_INT, c[i + 2] * 4, c[i + 1]); i += 3; passDrew = true; break;
        case 14: { // COPY_BUFFER
          const s = objects[c[i]]; const d = objects[c[i + 2]];
          const so = c[i + 1]; const dO = c[i + 3]; const size = c[i + 4];
          if (s.gl && d.gl && !d.shadow && !d.host) {
            gl.bindBuffer(gl.COPY_READ_BUFFER, s.gl);
            gl.bindBuffer(gl.COPY_WRITE_BUFFER, d.gl);
            gl.copyBufferSubData(gl.COPY_READ_BUFFER, gl.COPY_WRITE_BUFFER, so, dO, size);
          } else {
            writeBytes(d, dO, bytesOf(s, so, size).slice());
          }
          i += 5;
          break;
        }
        case 15: { // COPY_TEXTURE
          const t = texture(c[i]); const mip = c[i + 1]; const layer = c[i + 2];
          const w = c[i + 3]; const h = c[i + 4]; const b = objects[c[i + 5]]; const off = c[i + 6];
          gl.bindFramebuffer(gl.READ_FRAMEBUFFER, readFbo);
          if (t.target === gl.TEXTURE_2D_ARRAY) gl.framebufferTextureLayer(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, t.tex, mip, layer);
          else gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.tex, mip);
          gl.readBuffer(gl.COLOR_ATTACHMENT0);
          gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
          const fm = t.f;
          let out;
          if (fm.type === gl.UNSIGNED_BYTE && !fm.unorm) {
            out = new Uint8Array(w * h * 4);
            gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, out);
            if (fm.bgra) for (let k = 0; k < out.length; k += 4) { const r = out[k]; out[k] = out[k + 2]; out[k + 2] = r; }
          } else if (fm.unorm) {
            const rgba = new Uint8Array(w * h * 4);
            gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
            out = new Uint8Array(w * h);
            for (let k = 0; k < w * h; k++) out[k] = rgba[k * 4];
          } else {
            const rgba = new Float32Array(w * h * 4);
            gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, rgba);
            const n = fm.channels;
            if (fm.half) {
              const halfs = new Uint16Array(w * h * n);
              for (let k = 0; k < w * h; k++) for (let ch = 0; ch < n; ch++) halfs[k * n + ch] = toHalf(rgba[k * 4 + ch]);
              out = new Uint8Array(halfs.buffer);
            } else {
              const fl = new Float32Array(w * h * n);
              for (let k = 0; k < w * h; k++) for (let ch = 0; ch < n; ch++) fl[k * n + ch] = rgba[k * 4 + ch];
              out = new Uint8Array(fl.buffer);
            }
          }
          b.host.set(out, off);
          gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
          i += 7;
          break;
        }
        case 16: // BLIT
          blit(c[i], c[i + 1], c[i + 2], c[i + 3], c[i + 4], c[i + 5], c[i + 6]);
          i += 7;
          if (timing.on()) glMark('blit');
          break;
        case 17: { // COPY_DEPTH
          const from = framebuffer([], { t: texture(c[i]), mip: c[i + 1], layer: c[i + 2] });
          framebuffer([], { t: texture(c[i + 3]), mip: c[i + 4], layer: c[i + 5] });
          gl.bindFramebuffer(gl.READ_FRAMEBUFFER, from);
          setFlag(gl.SCISSOR_TEST, false);
          gl.blitFramebuffer(0, 0, c[i + 6], c[i + 7], 0, 0, c[i + 6], c[i + 7], gl.DEPTH_BUFFER_BIT, gl.NEAREST);
          i += 8;
          if (timing.on()) glMark('copy depth');
          break;
        }
        default:
          throw new Error(`unknown command ${op} at word ${i - 1}`);
      }
    }
  };

  let xrTarget = null;
  const setXrTarget = (target) => {
    if (target == null) {
      xrTarget = null;
      return true;
    }
    if (
      typeof target !== 'object'
      || !Number.isFinite(target.x)
      || !Number.isFinite(target.y)
      || !Number.isFinite(target.width)
      || !Number.isFinite(target.height)
      || target.width <= 0
      || target.height <= 0
    ) {
      return false;
    }
    xrTarget = {
      framebuffer: target.framebuffer ?? null,
      x: Math.round(target.x),
      y: Math.round(target.y),
      width: Math.round(target.width),
      height: Math.round(target.height),
    };
    return true;
  };

  const present = () => {
    const s = ensureSurface();
    if (!s) return;
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, readFbo);
    gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, s.tex, 0);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    setFlag(gl.SCISSOR_TEST, false);
    gl.disable(gl.SCISSOR_TEST);
    if (xrTarget) {
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, xrTarget.framebuffer);
      const x0 = xrTarget.x;
      const y0 = xrTarget.y;
      const x1 = x0 + xrTarget.width;
      const y1 = y0 + xrTarget.height;
      const filter = (s.w === xrTarget.width && s.h === xrTarget.height) ? gl.NEAREST : gl.LINEAR;
      gl.blitFramebuffer(0, 0, s.w, s.h, x0, y1, x1, y0, gl.COLOR_BUFFER_BIT, filter);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    } else {
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
      // Row 0 of the surface is the top of the image; GL's window origin is the bottom-left.
      gl.blitFramebuffer(0, 0, s.w, s.h, 0, s.h, s.w, 0, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    }
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    if (timing.on()) glMark(`present ${s.w}x${s.h}`);
  };

  const imports = {
    gpu_api: () => 2,
    gl_info(out, cap) {
      const P = (p) => gl.getParameter(p);
      const features = (ext.colorFloat ? 1 : 0) | (ext.floatLinear ? 2 : 0) | (ext.floatBlend ? 4 : 0)
        | (ext.indexed ? 8 : 0) | (ext.s3tc && ext.s3tcSrgb ? 16 : 0)
        | (ext.bptc ? 32 : 0) | (ext.astc ? 64 : 0);
      const aniso = ext.aniso ? P(ext.aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT) : 1;
      f32[0] = aniso;
      const v = [P(gl.MAX_TEXTURE_SIZE), P(gl.MAX_ARRAY_TEXTURE_LAYERS), P(gl.MAX_UNIFORM_BLOCK_SIZE),
        P(gl.UNIFORM_BUFFER_OFFSET_ALIGNMENT), scratchUnit, P(gl.MAX_TEXTURE_IMAGE_UNITS),
        P(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS), P(gl.MAX_DRAW_BUFFERS), P(gl.MAX_VERTEX_ATTRIBS),
        P(gl.MAX_UNIFORM_BUFFER_BINDINGS), features, i32[0], 0, 0, 0, 0];
      const n = Math.min(v.length, cap >>> 0);
      u32(out, n).set(v.slice(0, n));
      return n;
    },
    gl_renderer_name: (out, cap) => writeStr(gpu.adapterName, out, cap),
    gl_last_error: (out, cap) => writeStr(lastError, out, cap),
    gl_error_count: () => errors.length,
    gl_error: (i, out, cap) => writeStr(errors[i >>> 0] || '', out, cap),
    gl_canvas(out) {
      if (!canvas) return 0;
      ensureSurface();
      u32(out, 2).set([canvas.width, canvas.height]);
      return 1;
    },
    gl_create_buffer: guard('createBuffer', (size, kinds) => {
      const b = { size, index: !!(kinds & BUF_INDEX), gl: null, shadow: null, tex: null, host: null };
      if (kinds & (BUF_DATA | BUF_INDEX)) {
        b.gl = gl.createBuffer();
        bindForWrite(b);
        gl.bufferData(bufferTarget(b), size, gl.DYNAMIC_DRAW);
      }
      if (kinds & BUF_STORAGE) {
        const rows = Math.max(1, Math.ceil(size / ROW_BYTES));
        b.shadow = new Uint8Array(rows * ROW_BYTES);
        b.tex = gl.createTexture();
        gl.activeTexture(gl.TEXTURE0 + scratchUnit);
        gl.bindTexture(gl.TEXTURE_2D, b.tex);
        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32UI, ROW_TEXELS, rows);
        uploadRows(b, 0, rows * ROW_BYTES);
      }
      if (kinds & BUF_READBACK) b.host = new Uint8Array(size);
      return put(b);
    }),
    gl_write_buffer(h, offset, ptr, len) {
      writeBytes(objects[h >>> 0], offset, u8(ptr, len));
    },
    gl_read_buffer: guard('readBuffer', (h, out, len) => {
      const b = objects[h >>> 0];
      u8(out, len).set(b.host.subarray(0, len));
      return 1;
    }),
    gl_create_texture: guard('createTexture', (w, h, layers, mips, format, array) =>
      put(makeTexture(w, h, layers, mips, FORMATS[format], !!array))),
    gl_write_texture(h, mip, layer, w, hgt, ptr, len) {
      const t = objects[h >>> 0];
      const fm = t.f;
      bindScratch(t);
      if (fm.compressed) {
        const view = u8(ptr, len);
        if (t.target === gl.TEXTURE_2D_ARRAY) {
          gl.compressedTexSubImage3D(t.target, mip, 0, 0, layer, w, hgt, 1, fm.internal, view);
        } else {
          gl.compressedTexSubImage2D(t.target, mip, 0, 0, w, hgt, fm.internal, view);
        }
        return;
      }
      let data = fm.bgra || fm.type !== gl.UNSIGNED_BYTE ? u8(ptr, len).slice() : u8(ptr, len);
      if (fm.bgra) for (let k = 0; k < data.length; k += 4) { const r = data[k]; data[k] = data[k + 2]; data[k + 2] = r; }
      if (fm.type === gl.HALF_FLOAT) data = new Uint16Array(data.buffer);
      else if (fm.type === gl.FLOAT) data = new Float32Array(data.buffer);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      if (t.target === gl.TEXTURE_2D_ARRAY) gl.texSubImage3D(t.target, mip, 0, 0, layer, w, hgt, 1, fm.format, fm.type, data);
      else gl.texSubImage2D(t.target, mip, 0, 0, w, hgt, fm.format, fm.type, data);
    },
    gl_write_texture_source: guard('texSubImage2D', (h, mip, layer, w, hgt) => {
      const image = sourceOfSize(source, w, hgt);
      const t = objects[h >>> 0];
      bindScratch(t);
      // Unflipped (top row first) and straight alpha, as pixelsOf reads it back. Colour
      // conversion stays the browser's default, as drawImage applies it. The page's own
      // unpack state comes back afterwards.
      const unpack = [[gl.UNPACK_FLIP_Y_WEBGL, false], [gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false],
        [gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.BROWSER_DEFAULT_WEBGL]];
      const saved = unpack.map(([p]) => gl.getParameter(p));
      for (const [p, v] of unpack) gl.pixelStorei(p, v);
      try {
        if (t.target === gl.TEXTURE_2D_ARRAY) gl.texSubImage3D(t.target, mip, 0, 0, layer, w, hgt, 1, t.f.format, t.f.type, image);
        else gl.texSubImage2D(t.target, mip, 0, 0, w, hgt, t.f.format, t.f.type, image);
      } finally {
        unpack.forEach(([p], i) => gl.pixelStorei(p, saved[i]));
      }
      return 1;
    }),
    gl_generate_mips(h) {
      const t = objects[h >>> 0];
      bindScratch(t);
      setLevels(t, 0, t.mips);
      gl.generateMipmap(t.target);
    },
    gl_create_sampler: guard('createSampler', (ptr, words) => {
      const d = u32(ptr, words);
      const lod = new Float32Array(memory.buffer, (ptr >>> 0) + 20, 1)[0];
      const s = gl.createSampler();
      const lin = d[0] !== 0;
      gl.samplerParameteri(s, gl.TEXTURE_MAG_FILTER, lin ? gl.LINEAR : gl.NEAREST);
      gl.samplerParameteri(s, gl.TEXTURE_MIN_FILTER, lin
        ? (d[1] ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR_MIPMAP_NEAREST)
        : (d[1] ? gl.NEAREST_MIPMAP_LINEAR : gl.NEAREST_MIPMAP_NEAREST));
      for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.samplerParameteri(s, w, WRAP[d[2]]);
      gl.samplerParameterf(s, gl.TEXTURE_MAX_LOD, lod);
      if (d[4]) {
        gl.samplerParameteri(s, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
        gl.samplerParameteri(s, gl.TEXTURE_COMPARE_FUNC, COMPARES[d[4]]);
      }
      if (ext.aniso && d[3] > 1) gl.samplerParameterf(s, ext.aniso.TEXTURE_MAX_ANISOTROPY_EXT, d[3]);
      return put(s);
    }),
    gl_create_program: guard('createProgram', (vp, vl, fp, fl, mp, ml) => {
      const program = link(str(vp, vl), str(fp, fl));
      // The entry point it was generated from names the passes it draws in (`timing`).
      const entry = (source) => /from '([^']+)'/.exec(source)?.[1];
      const name = entry(str(fp, fl)) || entry(str(vp, vl)) || 'program';
      const hidden = [];
      gl.useProgram(program);
      for (const line of str(mp, ml).split('\n')) {
        const [kind, name, a, b] = line.split(' ');
        if (kind === 'b') {
          const index = gl.getUniformBlockIndex(program, name);
          if (index !== gl.INVALID_INDEX) gl.uniformBlockBinding(program, index, Number(a));
        } else if (kind === 'u') {
          const loc = gl.getUniformLocation(program, name);
          if (loc) gl.uniform1i(loc, Number(a));
        } else if (kind === 'h') {
          hidden[Number(a)] = { loc: gl.getUniformLocation(program, name), uint: b === 'uint' };
        }
      }
      gl.useProgram(currentProgram);
      return put({ program, hidden, name });
    }),
    gl_create_pipeline: guard('createPipeline', (program, ptr, words) => {
      const w = u32(ptr, words);
      const fl = new Float32Array(w.slice().buffer);
      const targets = [];
      let k = 9;
      for (let t = 0; t < w[8]; t++, k += 2) targets.push({ blend: w[k], mask: w[k + 1] });
      const stride = w[k]; const count = w[k + 1];
      const attrs = [];
      for (let a = 0; a < count; a++) attrs.push([w[k + 2 + a * 3], w[k + 3 + a * 3], w[k + 4 + a * 3]]);
      const written = targets.filter((t) => t.mask !== 0);
      const uniform = written.every((t) => t.blend === written[0].blend && t.mask === written[0].mask);
      return put({
        program: objects[program], mode: MODES[w[0]], cull: w[1], frontCcw: w[2] !== 0,
        depthTest: w[3] !== 0, depthFunc: COMPARES[w[4]], depthWrite: w[5] !== 0, bias: fl[6], slope: fl[7],
        targets, drawBuffers: targets.map((t) => (t.mask ? 1 : 0)).join(''),
        blend: written.length ? written[0].blend : 0, mask: written.length ? written[0].mask : 15,
        indexed: !uniform, stride, attrs,
      });
    }),
    gl_create_vao: guard('createVertexArray', (pipeline, buffer, offset, index) => {
      const p = objects[pipeline];
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      if (buffer) {
        gl.bindBuffer(gl.ARRAY_BUFFER, objects[buffer].gl);
        for (const [loc, fmt, off] of p.attrs) {
          gl.enableVertexAttribArray(loc);
          if (fmt <= 4) gl.vertexAttribPointer(loc, fmt, gl.FLOAT, false, p.stride, offset + off);
          else if (fmt === 5) gl.vertexAttribIPointer(loc, 1, gl.UNSIGNED_INT, p.stride, offset + off);
          else gl.vertexAttribPointer(loc, 4, gl.UNSIGNED_BYTE, true, p.stride, offset + off);
        }
      }
      if (index) gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, objects[index].gl);
      gl.bindVertexArray(null);
      currentVao = null;
      return put({ vao });
    }),
    gl_submit: guard('submit', (cmdPtr, words, pushPtr, pushLen, isPresent) => {
      execute(cmdPtr, words, pushPtr, pushLen);
      if (isPresent) present();
      glErrors(isPresent ? 'present' : 'submit');
      return 1;
    }),
    gl_release(h) {
      h >>>= 0;
      const o = objects[h];
      if (!o) return;
      if (o.tex && o.target) dropTexture(o);
      else if (o.vao) gl.deleteVertexArray(o.vao);
      else if (o.program && o.hidden) gl.deleteProgram(o.program);
      else if (o.mode !== undefined) { /* a pipeline: its program is released on its own */ } else if ('shadow' in o) {
        if (o.gl) gl.deleteBuffer(o.gl);
        if (o.tex) gl.deleteTexture(o.tex);
      } else if (o instanceof WebGLSampler) gl.deleteSampler(o);
      objects[h] = null;
      freeList.push(h);
    },
    log: (level, ptr, len) => onLog(level, str(ptr, len)),
    status: (ptr, len) => onStatus(str(ptr, len)),
  };
  // The WebGPU imports exist in the module too; on this host they are never reached.
  const unavailable = () => { throw new Error('WebGPU call on the WebGL 2 host'); };
  for (const name of GPU_IMPORTS) imports[name] = unavailable;

  return {
    imports: { ganbate: imports },
    errors,
    attach(instance) { memory = instance.exports.memory; },
    /** Sets (or with null clears) the image the next runtime call may upload into a texture. */
    setSource(image) { source = image; },
    nextFrame() {},
    timing,
    /** Closes a drawn frame's timed passes: booked once its last fence signalled. */
    endFrame() {
      const frame = timing.take();
      if (!frame) return;
      if (!channel && typeof MessageChannel === 'function') {
        channel = new MessageChannel();
        channel.port1.onmessage = poll;
      }
      if (!channel) {
        for (const m of frame.marks) gl.deleteSync(m.sync);
        timing.set(false);
        return;
      }
      waiting.push(frame);
      if (waiting.length === 1) channel.port2.postMessage(0);
    },
    frameTexture() { return surface; },
    bundleStats: () => ({ cached: 0, candidates: 0, replays: 0 }),
    setXrTarget,
    /** The canvas texture's pixels as RGBA rows, top row first. */
    readSurface() {
      const s = surface;
      if (!s) return null;
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, readFbo);
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, s.tex, 0);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
      const rgba = new Uint8ClampedArray(s.w * s.h * 4);
      gl.readPixels(0, 0, s.w, s.h, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
      for (let k = 3; k < rgba.length; k += 4) rgba[k] = 255;
      return { width: s.w, height: s.h, rgba };
    },
  };
}

const GPU_IMPORTS = ['gpu_info', 'gpu_adapter_name', 'gpu_last_error', 'gpu_error_count', 'gpu_error',
  'gpu_canvas', 'gpu_create_buffer', 'gpu_write_buffer', 'gpu_create_texture', 'gpu_write_texture',
  'gpu_copy_source', 'gpu_generate_mips', 'gpu_create_sampler', 'gpu_create_shader', 'gpu_create_bind_group_layout',
  'gpu_create_pipeline_layout', 'gpu_create_render_pipeline', 'gpu_create_compute_pipeline',
  'gpu_create_bind_group', 'gpu_submit', 'gpu_release'];
const GL_IMPORTS = ['gl_info', 'gl_renderer_name', 'gl_last_error', 'gl_error_count', 'gl_error',
  'gl_canvas', 'gl_create_buffer', 'gl_write_buffer', 'gl_read_buffer', 'gl_create_texture',
  'gl_write_texture', 'gl_write_texture_source', 'gl_generate_mips', 'gl_create_sampler', 'gl_create_program',
  'gl_create_pipeline', 'gl_create_vao', 'gl_submit', 'gl_release'];

const KEY_CODES = { ArrowLeft: 1, ArrowRight: 2, ArrowUp: 3, ArrowDown: 4, ShiftLeft: 5, ShiftRight: 5, Space: 32 };
function keyCode(e) {
  if (e.code in KEY_CODES) return KEY_CODES[e.code];
  let m = /^Key([A-Z])$/.exec(e.code);
  if (m) return m[1].toLowerCase().charCodeAt(0);
  m = /^(?:Digit|Numpad)([0-9])$/.exec(e.code);
  if (m) return m[1].charCodeAt(0);
  return 0;
}

/**
 * Streams `url` into wasm memory from `ganbate_alloc`, reporting progress (0..1) and bytes. With
 * an uncompressed Content-Length the bytes go straight into wasm memory; otherwise (unknown
 * length, or a compressed response that turns out longer) they are buffered and copied once.
 */
async function fetchInto(url, exports, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status} ${response.statusText}`);
  const encoding = response.headers.get('content-encoding');
  const announced = Number(response.headers.get('content-length')) || 0;
  const reader = response.body.getReader();
  let ptr = 0;
  let cap = 0;
  let offset = 0;
  let chunks = null;
  if (announced > 0 && (!encoding || encoding === 'identity')) {
    cap = announced;
    ptr = exports.ganbate_alloc(cap) >>> 0;
  } else {
    chunks = [];
  }
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!chunks && offset + value.length > cap) {
      // More than announced: keep what arrived and buffer the rest.
      chunks = [new Uint8Array(exports.memory.buffer, ptr, offset).slice()];
      exports.ganbate_free(ptr, cap);
      ptr = 0;
    }
    if (chunks) chunks.push(value);
    else new Uint8Array(exports.memory.buffer, ptr + offset, value.length).set(value);
    offset += value.length;
    onProgress(announced ? Math.min(offset / announced, 1) : 0, offset);
  }
  if (!chunks) {
    if (offset !== cap) {
      exports.ganbate_free(ptr, cap);
      throw new Error(`${url}: received ${offset} of ${cap} bytes`);
    }
    return { ptr, len: cap };
  }
  ptr = exports.ganbate_alloc(offset) >>> 0;
  let at = 0;
  for (const chunk of chunks) {
    new Uint8Array(exports.memory.buffer, ptr + at, chunk.length).set(chunk);
    at += chunk.length;
  }
  return { ptr, len: offset };
}

const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

/**
 * Decodes the loaded world's JPEG and WebP images with the browser's decoders and hands the
 * pixels to the engine. PNGs stay for the engine's own decoder, which keeps the colour of
 * transparent pixels (a 2D canvas premultiplies them away, darkening filtered edges).
 */
async function decodeWorldImages(x, onLog) {
  const read = (call) => {
    const len = call(0, 0);
    if (!len) return new Uint8Array(0);
    const p = x.ganbate_alloc(len) >>> 0;
    call(p, len);
    const out = new Uint8Array(x.memory.buffer, p, len).slice();
    x.ganbate_free(p, len);
    return out;
  };
  const list = JSON.parse(new TextDecoder().decode(read((p, cap) => x.ganbate_world_images(p, cap))))
    .filter((im) => im.mime !== 'image/png');
  const decoded = await Promise.all(list.map(async (im) => {
    const bytes = read((p, cap) => x.ganbate_world_image(im.texture, p, cap));
    try {
      const bitmap = await createImageBitmap(new Blob([bytes], { type: im.mime }),
        { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
      const pixels = pixelsOf(bitmap);
      if (bitmap.close) bitmap.close();
      return { im, pixels };
    } catch (e) {
      onLog(1, `world image ${im.texture} (${im.mime}) not decoded: ${e.message}`);
      return null;
    }
  }));
  for (const d of decoded) {
    if (!d) continue;
    const { width, height, data } = d.pixels;
    const p = x.ganbate_alloc(data.length) >>> 0;
    new Uint8Array(x.memory.buffer, p, data.length).set(data);
    x.ganbate_world_set_image(d.im.texture, width, height, p, data.length);
    x.ganbate_free(p, data.length);
  }
  return list.length;
}

/** WebGPU, the host and the wasm module; shared by the player and the self-test. */
async function boot(options) {
  const { canvas, onPhase = () => {}, onLog = () => {} } = options;
  const errors = [];
  const fail = (text) => {
    errors.push(text);
    (options.onError || (() => {}))(text);
  };
  // WebGPU when the browser has it (unless `api` is 'webgl2'), otherwise WebGL 2.
  const api = options.api || 'auto';
  let gpu = null;
  if (api !== 'webgl2' && typeof navigator !== 'undefined' && navigator.gpu) {
    onPhase('Starting WebGPU', 0);
    try {
      gpu = await createGpu(canvas);
    } catch (e) {
      if (api === 'webgpu') throw e;
      onLog(1, `WebGPU unavailable (${e.message}); using WebGL 2`);
    }
  } else if (api === 'webgpu') {
    throw new Error('This browser has no WebGPU. Use Chrome or Edge 113+ or Safari 26+, ' +
      'served over https or from localhost.');
  }
  if (!gpu) {
    onPhase('Starting WebGL 2', 0);
    gpu = createGl(canvas);
  }
  onLog(0, `${gpu.adapterName}; features: ${gpu.features.join(', ') || 'none'}; canvas ${gpu.format}`);
  // Frames go straight into the canvas unless `present: 'settled'` asks otherwise (`createFrameGate`).
  const settled = gpu.api === 'webgpu' && options.present === 'settled';
  const hostOptions = {
    onLog,
    onStatus: options.onStatus || (() => {}),
    onError: fail,
    checkErrors: !!options.checkErrors,
    settled,
    // Safari refuses some of the engine's render bundles as it records them ("setPipeline:
    // validation failed"); every frame replaying one is then refused whole, and the table freezes
    // between the few frames drawn after the bundle cache empties. Apple's WebKit encodes every pass.
    bundles: options.bundles ?? !appleWebKit(),
  };
  const host = gpu.api === 'webgl2' ? createGlHost(gpu, hostOptions) : createHost(gpu, hostOptions);
  onPhase('Loading the engine', 0);
  let instance;
  if (options.wasm && typeof options.wasm === 'object' && options.wasm.exports) {
    instance = options.wasm;
  } else if (options.wasm instanceof ArrayBuffer || ArrayBuffer.isView(options.wasm)) {
    ({ instance } = await WebAssembly.instantiate(options.wasm, host.imports));
  } else {
    const wasmResponse = await fetch(options.wasm);
    if (!wasmResponse.ok) throw new Error(`${options.wasm}: HTTP ${wasmResponse.status}`);
    try {
      ({ instance } = await WebAssembly.instantiateStreaming(wasmResponse.clone(), host.imports));
    } catch (_) {
      ({ instance } = await WebAssembly.instantiate(await wasmResponse.arrayBuffer(), host.imports));
    }
  }
  host.attach(instance);
  return { gpu, host, x: instance.exports, errors, fail, settled };
}

/** Reads `texture` (the canvas texture of the current frame) back as RGBA8 pixels. */
async function readPixels(gpu, texture, host) {
  if (gpu.api === 'webgl2') return host.readSurface();
  const width = texture.width; const height = texture.height;
  const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
  const buffer = gpu.device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = gpu.device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, [width, height, 1]);
  gpu.device.queue.submit([enc.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(buffer.getMappedRange());
  const rgba = new Uint8ClampedArray(width * height * 4);
  const bgra = gpu.format.startsWith('bgra');
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const s = y * bytesPerRow + x * 4; const d = (y * width + x) * 4;
      rgba[d] = src[s + (bgra ? 2 : 0)]; rgba[d + 1] = src[s + 1];
      rgba[d + 2] = src[s + (bgra ? 0 : 2)]; rgba[d + 3] = 255;
    }
  }
  buffer.unmap();
  buffer.destroy();
  return { width, height, rgba };
}

/**
 * Runs the GPU-layer self-test of the wasm module (compute, copies, mips, a blit into the
 * canvas) and resolves with the canvas pixels and any errors.
 */
export async function startSelfTest(options) {
  const { gpu, host, x, errors, fail } = await boot({ ...options, checkErrors: true, present: 'direct' });
  if (options.sizeCanvas) options.sizeCanvas();
  const pixels = await new Promise((resolve, reject) => requestAnimationFrame(() => {
    host.nextFrame();
    let ok = 0;
    try {
      ok = x.ganbate_selftest();
    } catch (e) {
      fail(`the self-test stopped: ${e.message}`);
    }
    const texture = host.frameTexture();
    if (!ok || !texture) {
      reject(new Error(errors[errors.length - 1] || 'the self-test presented nothing'));
      return;
    }
    readPixels(gpu, texture, host).then(resolve, reject);
  }));
  // Let asynchronous WebGPU errors arrive.
  if (gpu.device) await gpu.device.queue.onSubmittedWorkDone();
  await nextFrame();
  return {
    adapter: gpu.adapterName,
    format: gpu.format,
    pixels,
    errors: () => [...errors, ...host.errors.filter((e) => !errors.includes(e))],
  };
}

/**
 * Loads the player: WebGPU, the wasm module, the world and the Look file; then runs frames.
 * `options`: canvas, wasm (URL), scene (URL), look (URL or ''), query (string),
 * onPhase(text, progress), onLog(level, text), onStatus(text), onError(text), sizeCanvas().
 * Returns a controller for input and tests.
 */
export async function startPlayer(options) {
  const { canvas, onPhase = () => {} } = options;
  const { gpu, host, x, errors, fail, settled } = await boot(options);
  const call = (what, f) => {
    try {
      return f();
    } catch (e) {
      throw new Error(`${what} stopped: ${e.message}${errors.length ? ` (${errors[errors.length - 1]})` : ''}`);
    }
  };

  const world = await fetchInto(options.scene, x, (p, bytes) =>
    onPhase(`Downloading the world (${(bytes / 1048576).toFixed(1)} MB)`, p));
  onPhase('Decoding the world', 1);
  await nextFrame();
  if (!call('decoding', () => x.ganbate_load_world(world.ptr, world.len))) {
    throw new Error(errors[errors.length - 1] || 'the world could not be decoded');
  }
  onPhase('Decoding images', 1);
  await decodeWorldImages(x, options.onLog || (() => {}));
  let lookText = '';
  if (options.look) {
    onPhase('Loading the look', 1);
    const r = await fetch(options.look);
    if (!r.ok) throw new Error(`${options.look}: HTTP ${r.status} ${r.statusText}`);
    lookText = await r.text();
  }
  let views = [];
  try {
    views = (JSON.parse(lookText || '{}').views || []).map((v) => String(v.name));
  } catch (_) {
    // The engine reports Look file errors.
  }
  const toWasm = (text) => {
    const bytes = new TextEncoder().encode(text);
    const ptr = x.ganbate_alloc(bytes.length) >>> 0;
    new Uint8Array(x.memory.buffer, ptr, bytes.length).set(bytes);
    return { ptr, len: bytes.length };
  };
  const look = toWasm(lookText);
  const query = toWasm(options.query || '');
  onPhase('Compiling shaders and uploading', 1);
  await nextFrame();
  if (options.sizeCanvas) options.sizeCanvas();
  const started = call('start', () => x.ganbate_start(look.ptr, look.len, query.ptr, query.len));
  x.ganbate_free(look.ptr, look.len);
  x.ganbate_free(query.ptr, query.len);
  if (!started) throw new Error(errors[errors.length - 1] || 'the player could not start');
  onPhase('', 1);

  // Input.
  const keys = new Set();
  const key = (code, down) => {
    if (!code) return;
    if (down) keys.add(code); else keys.delete(code);
    x.ganbate_key(code, down ? 1 : 0);
  };
  window.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const code = keyCode(e);
    if (!code) return;
    if (code <= 4 || code === 32) e.preventDefault();
    if (!e.repeat) key(code, true);
  });
  window.addEventListener('keyup', (e) => key(keyCode(e), false));
  window.addEventListener('blur', () => { keys.clear(); x.ganbate_blur(); });
  const pointers = new Map();
  let pinch = null;
  const centroid = () => {
    let sx = 0; let sy = 0;
    for (const p of pointers.values()) { sx += p.x; sy += p.y; }
    return { x: sx / pointers.size, y: sy / pointers.size };
  };
  canvas.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch (_) {
      // Not an active pointer (for example a synthetic event); drags still work while over the canvas.
    }
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2) pinch = centroid();
    e.preventDefault();
  });
  canvas.addEventListener('pointermove', (e) => {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x; const dy = e.clientY - p.y;
    p.x = e.clientX; p.y = e.clientY;
    if (pointers.size === 1) {
      x.ganbate_look(dx * (e.pointerType === 'touch' ? 1.4 : 1), dy * (e.pointerType === 'touch' ? 1.4 : 1));
    } else if (pointers.size === 2 && pinch) {
      const c = centroid();
      const scale = Math.min(canvas.clientWidth, canvas.clientHeight) * 0.25;
      x.ganbate_move((pinch.y - c.y) / scale, (c.x - pinch.x) / scale);
    }
  });
  const lift = (e) => {
    pointers.delete(e.pointerId);
    if (pointers.size < 2 && pinch) { pinch = null; x.ganbate_move(0, 0); }
  };
  canvas.addEventListener('pointerup', lift);
  canvas.addEventListener('pointercancel', lift);

  // Frames.
  let last = performance.now();
  let running = true;
  let captures = [];
  const capture = (texture) => readPixels(gpu, texture, host);
  // Frames on the GPU are bounded (`createFrameGate`): without vsync (or on a slow GPU) they would
  // otherwise queue up without bound and input, readbacks and resizes would wait behind them.
  const gate = createFrameGate(gpu, host, settled);
  const tick = (now) => {
    if (!running) return;
    if (options.sizeCanvas) options.sizeCanvas();
    gate.show();
    if (gate.full()) {
      requestAnimationFrame(tick);
      return;
    }
    const dt = Math.min(Math.max((now - last) / 1000, 0), 0.25);
    last = now;
    host.nextFrame();
    let ok = 0;
    try {
      ok = x.ganbate_frame(dt);
    } catch (e) {
      fail(`the engine stopped: ${e.message}`);
    }
    if (!ok) {
      running = false;
      if (!errors.length) fail('the engine stopped');
      return;
    }
    gate.submitted();
    const texture = host.frameTexture();
    if (captures.length && texture) {
      const pending = captures;
      captures = [];
      capture(texture).then((img) => pending.forEach((r) => r.resolve(img)),
        (e) => pending.forEach((r) => r.reject(e)));
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);

  return {
    adapter: gpu.adapterName,
    format: gpu.format,
    views,
    exports: x,
    errors: () => [...errors, ...host.errors.filter((e) => !errors.includes(e))],
    frames: () => x.ganbate_frames(),
    key,
    stop() { running = false; gate.stop(); },
    /** The next rendered frame as RGBA pixels. */
    capture: () => new Promise((resolve, reject) => captures.push({ resolve, reject })),
  };
}

// ------------------------------------------------------------------------------ Library mode

/** An image source's size in pixels: an image's natural size, a video's frame size. */
function sourceSize(source) {
  return [source.naturalWidth || source.videoWidth || source.width,
    source.naturalHeight || source.videoHeight || source.height];
}

/** `source` when it is set and `width` x `height`; throws otherwise (a host's upload check). */
function sourceOfSize(source, width, height) {
  if (!source) throw new Error('no source image is set for this call');
  const [w, h] = sourceSize(source);
  if (w !== width || h !== height) throw new Error(`the source image is ${w}x${h}, not ${width}x${height}`);
  return source;
}

/**
 * Whether the browser can copy `source` into a texture itself: canvases and videos, and on WebGPU
 * image bitmaps and images too. `{width, height, data}` stays pixels. On WebGL 2 so do images,
 * where an upload without premultiplication makes Chrome decode them again, and image bitmaps,
 * whose alpha the upload cannot un-premultiply (it ignores UNPACK_PREMULTIPLY_ALPHA_WEBGL).
 */
function copiesDirectly(source, webgpu) {
  if (!source || typeof source !== 'object' || source.data) return false;
  if (webgpu) return true;
  const bitmap = typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap;
  return !bitmap && !('naturalWidth' in source);
}

/** RGBA8 pixels of an image source: `{width, height, data}` passes through; canvases, image
 * bitmaps and images are drawn into a 2D canvas and read back (straight alpha). */
function pixelsOf(source) {
  if (source && source.data && source.width && source.height) return source;
  const [width, height] = sourceSize(source);
  if (!width || !height) throw new Error('createTexture: the source has no size');
  const c = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(width, height)
    : Object.assign(document.createElement('canvas'), { width, height });
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(source, 0, 0, width, height);
  return { width, height, data: ctx.getImageData(0, 0, width, height).data };
}

/**
 * Maps a `quality` mode (`'realtime' | 'still' | 'hybrid' | 'low-cost' | boolean`) to the C-ABI
 * preset code (`0` = realtime, `1` = still, `2` = hybrid stage, `3` = low-cost fast-path), or
 * `null` when unrecognized.
 */
export function qualityCodeFromMode(mode) {
  if (mode === true || mode === 'still') return 1;
  if (mode === false || mode === 'realtime') return 0;
  if (mode === 'hybrid' || mode === 'hybrid-stage' || mode === 'hybrid_stage') return 2;
  if (mode === 'low-cost' || mode === 'low_cost' || mode === 'lowcost') return 3;
  return null;
}

/**
 * Resolves startup `quality` / legacy `still` options for `createEngine` into a C-ABI preset code
 * (`0` = realtime, `1` = still, `2` = hybrid stage, `3` = low-cost fast-path).
 */
export function resolveQualityCode(quality, still) {
  const explicit = qualityCodeFromMode(quality);
  if (explicit !== null) return explicit;
  return still === true ? 1 : 0;
}

function resolveStereoEyeCode(eye) {
  if (eye === undefined || eye === 'mono' || eye === 0) return 0;
  if (eye === 'left' || eye === -1) return -1;
  if (eye === 'right' || eye === 1) return 1;
  return null;
}

/**
 * Library mode: the engine as a renderer that a host page (a game with its own interface)
 * drives. Resolves with an engine object; its methods throw with the engine's message when a
 * call fails. Handles (textures, meshes, materials, instances) are nonzero numbers.
 *
 * `options`: canvas; wasm (URL of ganbate_web.wasm); api ('auto', 'webgpu' or 'webgl2');
 * world (URL, ArrayBuffer or Uint8Array of a cooked `.gbew`; omit for a scene of props only);
 * look (Look file as an object or JSON text; optional);
 * quality ('realtime' | 'still' | 'hybrid' | 'low-cost' | boolean; legacy `still?: boolean` also supported);
 * onLog(level, text); onPhase(text, progress).
 *
 * The page sizes the canvas (canvas.width/height in device pixels; `fitCanvas(dpr)` helps) and
 * calls `render()` when it wants a frame. Positions are metres; matrices are 16 numbers,
 * column-major, model to world.
 */
/**
 * One scratch allocation in the module's memory, reused: the module reads call arguments during
 * the call only. Its start is rounded up to 4 bytes so typed-array views of it are aligned.
 * `reserve(len)` returns room for `len` bytes; `place(...arrays)` copies byte arrays there, each at
 * a 4-byte boundary, and returns their pointers; `free()` gives the memory back.
 */
function scratchFor(x) {
  let scratch = 0;
  let scratchCap = 0;
  const reserve = (len) => {
    if (len + 4 > scratchCap) {
      if (scratch) x.ganbate_free(scratch, scratchCap);
      scratchCap = Math.max(len + 4, scratchCap * 2, 4096);
      scratch = x.ganbate_alloc(scratchCap) >>> 0;
    }
    return (scratch + 3) & ~3;
  };
  const place = (...arrays) => {
    let total = 0;
    const offsets = arrays.map((a) => {
      const at = total;
      total += a ? (a.byteLength + 3) & ~3 : 0;
      return at;
    });
    const base = reserve(Math.max(total, 4));
    const mem = new Uint8Array(x.memory.buffer);
    return arrays.map((a, i) => {
      if (!a) return 0;
      mem.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), base + offsets[i]);
      return base + offsets[i];
    });
  };
  const free = () => {
    if (scratch) x.ganbate_free(scratch, scratchCap);
    scratch = 0;
    scratchCap = 0;
  };
  return { reserve, place, free };
}

/**
 * The graph solver's two calls on module exports `x` (see `crates/ganbate-graph`): `graphCommit`
 * applies a patch, `graphCheck` takes a whole graph. Each returns `{graph, digest, repairs}` or
 * throws with the reason a change was refused.
 */
function graphCalls(x, { reserve, place }) {
  const encoder = new TextEncoder();
  const json = (v) => (v === null ? new Uint8Array(0) : encoder.encode(typeof v === 'string' ? v : JSON.stringify(v)));
  const call = (name, label, first, second, mode) => {
    const [a, b] = [json(first), json(second)];
    const [ap, bp] = place(a, b);
    const len = x[name](ap, a.byteLength, bp, b.byteLength, mode === 'reject' ? 1 : 0);
    const out = reserve(len);
    x.ganbate_graph_result(out, len);
    const result = JSON.parse(new TextDecoder().decode(new Uint8Array(x.memory.buffer, out, len)));
    if (result.error) throw new Error(`${label}: ${result.error}`);
    return result;
  };
  return {
    /**
     * Commits `patch` (graph operations) to `graph` (`ganbate.semantic.graph.v2`), each an object
     * or its JSON text: with `mode` 'project' what breaks support, stage or clearance is moved to
     * the nearest valid place and reported in `repairs`; with 'reject' the patch is refused.
     * `graph` itself never changes.
     */
    graphCommit: (graph, patch, mode = 'project') => call('ganbate_graph_commit', 'graphCommit', graph, patch, mode),
    /**
     * Checks a whole graph, as written by hand or received from another participant, against
     * `previous` (null when opening it): a node whose relation changed is placed by it, the
     * others relate anew from where they stand, and every node is kept to the rules.
     */
    graphCheck: (graph, previous = null, mode = 'project') => call('ganbate_graph_check', 'graphCheck', previous, graph, mode),
  };
}

/**
 * The graph solver without a canvas, for pages and tools that edit a semantic graph without
 * drawing it (an editor panel, a demo's controls): the module is instantiated with inert host
 * imports, which the graph functions never call. `wasm`: the module's URL, bytes, or a compiled
 * `WebAssembly.Module`.
 */
export async function createGraphSolver({ wasm }) {
  const module =
    wasm instanceof WebAssembly.Module
      ? wasm
      : await WebAssembly.compile(typeof wasm === 'string' ? await (await fetch(wasm)).arrayBuffer() : wasm);
  const imports = {};
  for (const i of WebAssembly.Module.imports(module)) (imports[i.module] ??= {})[i.name] = () => 0;
  const { exports: x } = await WebAssembly.instantiate(module, imports);
  return graphCalls(x, scratchFor(x));
}

export async function createEngine(options) {
  const { canvas } = options;
  let lastError = '';
  const onLog = options.onLog || (() => {});
  // While a texture is copied from its source: a refusal falls back to the pixels, so it warns.
  let trying = false;
  const { gpu, host, x, errors, settled } = await boot({
    ...options,
    onLog: (level, text) => {
      // A panic is never a refusal: the module is dead, and the error must stay one.
      if (trying && level >= 2 && !text.startsWith('panic')) {
        onLog(1, `${text}; sending the pixels instead`);
        return;
      }
      if (level >= 2) lastError = text;
      onLog(level, text);
    },
    onError: (text) => {
      lastError = text;
      onLog(2, text);
    },
  });
  const fail = (what) => new Error(`${what}: ${lastError || 'the engine refused'}`);
  const ok = (what, r) => {
    if (!r) throw fail(what);
    return r;
  };

  const { reserve, place, free: freeScratch } = scratchFor(x);
  const encoder = new TextEncoder();
  /**
   * Has the browser copy `source` straight into a texture (no readback, no copy through wasm
   * memory): `call(width, height)` is the runtime export, and the host holds `source` for that
   * call only. 0 when the module, the browser or the source cannot; the caller then sends pixels.
   */
  const fromSource = (source, call) => {
    if (typeof x.ganbate_rt_texture_create_source !== 'function' || !copiesDirectly(source, gpu.api !== 'webgl2')) return 0;
    const [width, height] = sourceSize(source);
    if (!width || !height) return 0;
    host.setSource(source);
    trying = true;
    try {
      return call(width, height);
    } catch (e) {
      // A trap left the module unusable; falling back would hide why.
      if (e instanceof WebAssembly.RuntimeError) throw e;
      onLog(1, `texture from source: ${e.message}; sending the pixels instead`);
      return 0;
    } finally {
      trying = false;
      host.setSource(null);
    }
  };
  const text = (s) => encoder.encode(s);
  const f32 = (a) => (a instanceof Float32Array ? a : new Float32Array(a));
  /** A light as the 14 floats the runtime reads (`light_desc` in runtime.rs). */
  const lightFloats = (light) => {
    const v = new Float32Array(15);
    v.set(light.position, 0);
    v[3] = light.range;
    v.set(light.color, 4);
    v[7] = light.intensity;
    if (light.direction !== undefined) {
      v.set(light.direction, 8);
      v[11] = light.inner ?? 0;
      v[12] = light.outer ?? light.inner ?? 0;
    }
    v[13] = light.followsLamps === true ? 1 : 0;
    v[14] = light.shadow ?? 0;
    return v;
  };
  const out = (n) => new Float32Array(x.memory.buffer, reserve(n * 4), n);
  const readBudget = (p) => {
    const v = new Float32Array(x.memory.buffer, p, 6);
    return {
      resolutionScale: v[0],
      maxDynamicLights: v[1] >>> 0,
      ambientParticles: v[2] > 0.5,
      shadowTier: v[3] >>> 0,
      tier: v[4] >>> 0,
      rollingMeanMs: v[5],
    };
  };

  // The world's baked AO (a `ganbate-build bake-ao` sidecar), fetched while the world loads.
  const bakedAo = typeof options.ao === 'string'
    ? fetch(options.ao).then((r) => (r.ok ? r.arrayBuffer() : null), () => null)
    : Promise.resolve(options.ao ?? null);
  // The world (decoded, its images decoded by the browser) and the Look.
  let world = null;
  if (typeof options.world === 'string') {
    world = await fetchInto(options.world, x, (p, bytes) =>
      (options.onPhase || (() => {}))(`Downloading the world (${(bytes / 1048576).toFixed(1)} MB)`, p));
  } else if (options.world) {
    const bytes = options.world instanceof Uint8Array ? options.world : new Uint8Array(options.world);
    const ptr = x.ganbate_alloc(bytes.length) >>> 0;
    new Uint8Array(x.memory.buffer, ptr, bytes.length).set(bytes);
    world = { ptr, len: bytes.length };
  }
  if (world) {
    (options.onPhase || (() => {}))('Decoding the world', 1);
    if (!x.ganbate_load_world(world.ptr, world.len)) throw fail('world');
    await decodeWorldImages(x, onLog);
  }
  const lookText = options.look == null ? ''
    : (typeof options.look === 'string' ? options.look : JSON.stringify(options.look));
  const lookBytes = text(lookText);
  const [lookPtr] = place(lookBytes);
  (options.onPhase || (() => {}))('Compiling shaders and uploading', 1);
  ok('start', x.ganbate_rt_start(0, 0, lookPtr, lookBytes.length, resolveQualityCode(options.quality, options.still)));
  const ao = world ? await bakedAo : null;
  if (ao) {
    const bytes = ao instanceof Uint8Array ? ao : new Uint8Array(ao);
    const [p] = place(bytes);
    if (!x.ganbate_rt_ao_import(p, bytes.length)) onLog(1, 'The baked AO does not fit this world; baking it live');
  }

  const gate = createFrameGate(gpu, host, settled);
  let stopped = false;
  // The size frames are drawn at changes only inside ganbate_rt_render (a resize), so it is
  // read once after each render() call, not on every project or pick.
  let drawn = null;
  const drawnSize = () => {
    if (!drawn) {
      const p = reserve(8);
      ok('size', x.ganbate_rt_size(p));
      const s = new Uint32Array(x.memory.buffer, p, 2);
      drawn = { width: s[0], height: s[1] };
    }
    return drawn;
  };
  const size = () => ({ ...drawnSize() });
  // The canvas's CSS box, kept by a ResizeObserver: reading clientWidth or clientHeight forces
  // a layout whenever styles changed, and fitCanvas, project and pick run every frame or many
  // times per action. The observed box follows a CSS resize one frame late. Without
  // ResizeObserver every call reads the box, as before.
  let cssBox = null;
  let observer = null;
  const cssSize = () => cssBox || { width: canvas.clientWidth, height: canvas.clientHeight };
  /** The observed content box as [width, height]: contentBoxSize is logical (inline, block). */
  const observedBox = (entry) => {
    const b = entry.contentBoxSize?.[0] ?? entry.contentBoxSize;
    if (!b) return [entry.contentRect.width, entry.contentRect.height];
    const vertical = /^(vertical|sideways)/.test(getComputedStyle(canvas).writingMode);
    return vertical ? [b.blockSize, b.inlineSize] : [b.inlineSize, b.blockSize];
  };
  if (typeof ResizeObserver === 'function') {
    try {
      observer = new ResizeObserver((entries) => {
        // Whole CSS pixels, as clientWidth and clientHeight give them.
        const [w, h] = observedBox(entries[entries.length - 1]);
        cssBox = { width: Math.round(w), height: Math.round(h) };
      });
      observer.observe(canvas);
      cssBox = { width: canvas.clientWidth, height: canvas.clientHeight };
    } catch (_) {
      observer = null; // Not an element (an OffscreenCanvas): read the box per call.
    }
  }
  /** CSS pixels per rendered pixel along x and y. */
  const cssScale = () => {
    const s = drawnSize();
    const css = cssSize();
    return { sx: (css.width || s.width) / s.width, sy: (css.height || s.height) / s.height };
  };

  const engine = {
    api: gpu.api,
    /** How frames reach the canvas: 'settled' (finished first, `createFrameGate`) or 'direct'. */
    present: settled ? 'settled' : 'direct',
    adapter: gpu.adapterName,
    errors: () => [...errors, ...host.errors.filter((e) => !errors.includes(e))],
    /**
     * Sizes the canvas to its CSS box times `dpr` (default: the device pixel ratio). It reads the
     * box itself, once a frame at most: a page may call it from its own resize handler, which can
     * run before the engine's observer has seen the new size. A page calling it every frame passes
     * the box it observed (`box`, CSS pixels) instead: reading it there forces a layout whenever
     * the page changed since the last.
     */
    fitCanvas(dpr = window.devicePixelRatio || 1, box = null) {
      const w = Math.max(1, Math.round((box ? box.width : canvas.clientWidth) * dpr));
      const h = Math.max(1, Math.round((box ? box.height : canvas.clientHeight) * dpr));
      if (canvas.width === w && canvas.height === h) return;
      canvas.width = w;
      canvas.height = h;
      gate.reshow();
    },
    /** The size the last frame was drawn at, in device pixels. */
    size,
    setLook(look) {
      const t = text(typeof look === 'string' ? look : JSON.stringify(look));
      const [p] = place(t);
      ok('setLook', x.ganbate_rt_set_look(p, t.length));
    },
    /**
     * `{eye, target, up = [0, 1, 0], fov = 50, shift = [0, 0], near = 0.08, focus}`, where
     * `focus` is `{distance, depth = 4, blur = 0.006}` or absent for everything sharp.
     */
    setView(v) {
      const up = v.up || [0, 1, 0];
      const shift = v.shift || [0, 0];
      const f = v.focus;
      const p = reserve(68);
      const dst = new Float32Array(x.memory.buffer, p, 17);
      dst[0] = v.eye[0]; dst[1] = v.eye[1]; dst[2] = v.eye[2];
      dst[3] = v.target[0]; dst[4] = v.target[1]; dst[5] = v.target[2];
      dst[6] = up[0]; dst[7] = up[1]; dst[8] = up[2];
      dst[9] = v.fov ?? 50;
      dst[10] = shift[0]; dst[11] = shift[1];
      dst[12] = v.near ?? 0.08;
      dst[13] = f ? f.distance : 0;
      dst[14] = f ? f.depth ?? 4 : 0;
      dst[15] = f ? f.blur ?? 0.006 : 0;
      dst[16] = f ? 1 : 0;
      ok('setView', x.ganbate_rt_set_view(p));
    },
    /** Switches the renderer quality preset dynamically at runtime (`'realtime' | 'still' | 'hybrid' | 'low-cost'`). */
    setQuality(mode) {
      if (typeof x.ganbate_rt_set_quality !== 'function') return false;
      const code = qualityCodeFromMode(mode);
      if (code === null) return false;
      return Boolean(x.ganbate_rt_set_quality(code));
    },
    /** Latest per-frame draw, cull, cache, and pass-skip telemetry, or `null` if unavailable. */
    frameStats() {
      if (typeof x.ganbate_rt_frame_stats !== 'function') return null;
      const p = reserve(36);
      if (!x.ganbate_rt_frame_stats(p)) return null;
      const w = new Uint32Array(x.memory.buffer, p, 9);
      return {
        worldDrawsTotal: w[0],
        worldDrawsSubmitted: w[1],
        worldDrawsCulled: w[2],
        shadowStaticCached: w[3] !== 0,
        shadowWorldDraws: w[4],
        groundCached: w[5] !== 0,
        reflectionDrawn: w[6] !== 0,
        ssaoDrawn: w[7] !== 0,
        fusedTonemap: w[8] !== 0,
      };
    },
    /** What the last frame did with the room's cached picture, or `null` if unavailable. */
    cacheStats() {
      if (typeof x.ganbate_rt_cache_stats !== 'function') return null;
      const p = reserve(28);
      if (!x.ganbate_rt_cache_stats(p)) return null;
      const w = new Uint32Array(x.memory.buffer, p, 7);
      return {
        worldCached: w[0] !== 0,
        regions: w[1],
        reshaded: w[2] !== 0,
        waterKept: w[3] !== 0,
        waterStaticsWhole: w[4] !== 0,
        waterStaticsRegions: w[5],
        waterFootprints: w[6],
      };
    },
    /** Selects stereo XR eye (`'mono' | 'left' | 'right'` or `-1 | 0 | 1`) with IPD in metres (`0.050..=0.078`, default `0.063`). */
    setStereoEye(eye = 'mono', ipdMetres = 0.063) {
      if (typeof x.ganbate_rt_set_stereo_eye !== 'function') return false;
      const code = resolveStereoEyeCode(eye);
      if (code === null || !Number.isFinite(ipdMetres)) return false;
      return Boolean(x.ganbate_rt_set_stereo_eye(code, ipdMetres));
    },
    /** Parallel-axis left/right stereo view pair derived from the current camera and IPD, or `null` if unavailable. */
    stereoViews() {
      if (typeof x.ganbate_rt_stereo_views !== 'function') return null;
      const p = out(16).byteOffset;
      if (!x.ganbate_rt_stereo_views(p)) return null;
      const v = Array.from(new Float32Array(x.memory.buffer, p, 16));
      return {
        left: { eye: v.slice(0, 3), target: v.slice(3, 6) },
        right: { eye: v.slice(6, 9), target: v.slice(9, 12) },
        rightAxis: v.slice(12, 15),
        ipdMetres: v[15],
      };
    },
    /** Underlying WebGL 2 context on `'webgl2'` (for `gl.makeXRCompatible()` / `XRWebGLLayer`), or `null` on `'webgpu'`. */
    glContext() {
      return gpu.api === 'webgl2' ? gpu.gl : null;
    },
    /** Sets the active WebXR eye framebuffer and viewport on `'webgl2'` (`null` restores canvas blit); returns `false` on `'webgpu'`. */
    setXrTarget(target = null) {
      return Boolean(host.setXrTarget?.(target));
    },
    /** Records a completed frame duration (`dtMs`, ms) into the 120Hz (`8.333ms`) governor and returns its snapshot, or `null` if unavailable. */
    governorRecord(dtMs) {
      if (typeof x.ganbate_rt_governor_record !== 'function') return null;
      const p = out(6).byteOffset;
      if (!x.ganbate_rt_governor_record(dtMs, p)) return null;
      return readBudget(p);
    },
    /** Queries the current 120Hz (`8.333ms`) governor snapshot without recording a frame sample, or `null` if unavailable. */
    governorQuery() {
      if (typeof x.ganbate_rt_governor_query !== 'function') return null;
      const p = out(6).byteOffset;
      if (!x.ganbate_rt_governor_query(p)) return null;
      return readBudget(p);
    },
    /** Resets the 120Hz governor rolling window and restores top quality tier (`3`). */
    governorReset() {
      if (typeof x.ganbate_rt_governor_reset !== 'function') return false;
      return Boolean(x.ganbate_rt_governor_reset());
    },
    /** Legacy alias for `governorRecord(dtMs)`. */
    recordFrameBudget(dtMs) {
      return engine.governorRecord(dtMs);
    },
    /** Legacy alias for `governorQuery()`. */
    frameBudget() {
      return engine.governorQuery();
    },
    /** Legacy alias for `governorReset()`. */
    resetFrameGovernor() {
      return engine.governorReset();
    },
    /** Time of day in hours and the animation clock in seconds. */
    setClock(hours, seconds = 0) { ok('setClock', x.ganbate_rt_set_clock(hours, seconds)); },
    /** The world's lamps this many times as bright as the hour lights them (1: as it does). */
    setLampBoost(boost) { ok('setLampBoost', x.ganbate_rt_set_lamp_boost(boost)); },
    setWeather(preset) {
      const t = text(preset);
      const [p] = place(t);
      ok('setWeather', x.ganbate_rt_set_weather(p, t.length));
    },
    /** Draws and presents a frame. Returns false when it skipped (GPU busy, nothing to draw). */
    render() {
      if (stopped) return false;
      gate.show();
      if (host.timing.busy() || gate.full()) return false;
      host.nextFrame();
      drawn = null;
      const r = x.ganbate_rt_render();
      if (!r) throw fail('render');
      if (r !== 1) return false;
      host.endFrame();
      gate.submitted();
      return true;
    },
    frames: () => x.ganbate_rt_frames(),
    /**
     * Does up to `samples` vertices' worth of the ambient-occlusion bake between frames; true when
     * there was work (false once converged or while edits settle). Call it from idle time.
     */
    stepAo: (samples) => (typeof x.ganbate_rt_step_ao === 'function' ? x.ganbate_rt_step_ao(Math.max(0, samples | 0)) === 1 : false),
    /** Completes the baked ambient occlusion now, for captures that must not depend on frame count. */
    finishAo: () => {
      if (typeof x.ganbate_rt_finish_ao === 'function') ok('finishAo', x.ganbate_rt_finish_ao());
    },
    /** Baked ambient-occlusion convergence in [0, 1]; it refines over frames after start and edits. */
    aoProgress: () => (typeof x.ganbate_rt_ao_progress === 'function' ? x.ganbate_rt_ao_progress() / 1000 : 1),
    bundleStats: () => (host.bundleStats ? host.bundleStats() : { cached: 0, candidates: 0, replays: 0 }),
    /** The last frame's pixels (RGBA rows, top first); for tests. */
    async capture() {
      const texture = host.frameTexture();
      if (!texture) throw new Error('capture: no frame was rendered');
      return readPixels(gpu, texture, host);
    },
    /** Waits until the GPU has finished submitted work (WebGPU; immediate on WebGL 2). */
    async idle() { if (gpu.device) await gpu.device.queue.onSubmittedWorkDone(); },
    /**
     * Times each pass of the frames drawn from now on (true), drawing one frame at a time and the
     * main pass as its parts, or stops (false); each call starts the means afresh.
     */
    setPassTiming(on) {
      host.timing.set(on);
      // The main pass splits into its parts while timed, so each shows on its own.
      if (typeof x.ganbate_rt_set_split_passes === 'function') x.ganbate_rt_set_split_passes(on ? 1 : 0);
    },
    /** The timed frames' mean GPU time a pass, in drawing order, and a frame's (ms). */
    passTiming: () => host.timing.report(),

    /**
     * `source`: {width, height, data (RGBA8)}; a canvas, ImageBitmap, video or loaded image, which
     * the browser copies to the GPU itself where it can (otherwise its pixels are read back); or
     * {width, height, clear: [r, g, b, a]}: one colour (straight alpha, sRGB-encoded channels
     * from 0 to 1, like a canvas colour) filled on the GPU, sampling as a canvas of it would.
     */
    createTexture(source, { srgb = true } = {}) {
      const s = srgb ? 1 : 0;
      if (source && !source.data && source.clear) {
        const { width, height, clear } = source;
        if (!(Number.isInteger(width) && Number.isInteger(height) && width > 0 && height > 0)
          || clear.length !== 4 || !Array.prototype.every.call(clear, Number.isFinite)) {
          throw new Error('createTexture: a cleared texture takes a width and height in whole pixels and clear: [r, g, b, a]');
        }
        return ok('createTexture', x.ganbate_rt_texture_create_clear(width, height, s, ...clear));
      }
      const direct = fromSource(source, (w, h) => x.ganbate_rt_texture_create_source(w, h, s));
      if (direct) return direct;
      const { width, height, data } = pixelsOf(source);
      const [p] = place(data);
      return ok('createTexture', x.ganbate_rt_texture_create(width, height, p, data.byteLength, s));
    },
    /** New pixels for a texture, from a source of its size (as createTexture's, without clear). */
    updateTexture(texture, source) {
      if (fromSource(source, (w, h) => x.ganbate_rt_texture_update_source(texture, w, h))) return;
      const { data } = pixelsOf(source);
      const [p] = place(data);
      ok('updateTexture', x.ganbate_rt_texture_update(texture, p, data.byteLength));
    },
    destroyTexture(texture) { x.ganbate_rt_texture_destroy(texture); },
    /** `{positions, normals?, uvs?, indices}` as flat arrays (3, 3 and 2 numbers per vertex). */
    createMesh(m) {
      const positions = f32(m.positions);
      const normals = m.normals && m.normals.length ? f32(m.normals) : null;
      const uvs = m.uvs && m.uvs.length ? f32(m.uvs) : null;
      const indices = m.indices instanceof Uint32Array ? m.indices : new Uint32Array(m.indices);
      const [p, n, u, i] = place(positions, normals, uvs, indices);
      return ok('createMesh', x.ganbate_rt_mesh_create(p, positions.length / 3, n, u, i, indices.length));
    },
    /** New `{positions, normals?}` for a mesh (same vertex count). */
    updateMesh(mesh, m) {
      const positions = f32(m.positions);
      const normals = m.normals && m.normals.length ? f32(m.normals) : null;
      const [p, n] = place(positions, normals);
      ok('updateMesh', x.ganbate_rt_mesh_update(mesh, p, n, positions.length / 3));
    },
    destroyMesh(mesh) { x.ganbate_rt_mesh_destroy(mesh); },
    /** See `MaterialDesc::from_json` (name, color, emissive, lamp, roughness, metallic,
     * shading, alphaCutoff, baseTexture, normalTexture, emissiveTexture, normalScale,
     * castsShadow, blend: 'opaque' | 'alpha' | 'add', overlay, depthTest, depthWrite, order, shader,
     * params). */
    createMaterial(desc) {
      const t = text(JSON.stringify(desc));
      const [p] = place(t);
      return ok('createMaterial', x.ganbate_rt_material_create(p, t.length));
    },
    updateMaterial(material, desc) {
      const t = text(JSON.stringify(desc));
      const [p] = place(t);
      ok('updateMaterial', x.ganbate_rt_material_update(material, p, t.length));
    },
    destroyMaterial(material) { x.ganbate_rt_material_destroy(material); },
    /** A game's prop shader: WGSL defining `fn shade(in: ShadeInput) -> vec4f` (see
     * src/render/shaders/prop_shader_input.wgsl). Overlay materials draw with it through their
     * `shader` and read their `params` (twenty numbers). It lasts as long as the engine. */
    createShader(source) {
      const t = text(source);
      const [p] = place(t);
      return ok('createShader', x.ganbate_rt_shader_create(p, t.length));
    },
    /** New `params` for a material drawn by a prop shader, cheaply (every frame if need be). */
    setMaterialParams(material, params) {
      const v = new Float32Array(20);
      v.set(params.slice(0, 20));
      const [p] = place(v);
      ok('setMaterialParams', x.ganbate_rt_material_params(material, p));
    },
    createInstance(mesh, material, matrix) {
      const [p] = place(f32(matrix));
      return ok('createInstance', x.ganbate_rt_instance_create(mesh, material, p));
    },
    setInstance(instance, matrix, visible = true) {
      const [p] = place(f32(matrix));
      ok('setInstance', x.ganbate_rt_instance_set(instance, p, visible ? 1 : 0));
    },
    /** Many at once: `[{instance, matrix, visible = true}]`. */
    setInstances(list) {
      const n = list.length;
      if (!n) return;
      const p = reserve(n * 72);
      const u = new Uint32Array(x.memory.buffer, p, n * 18);
      const f = new Float32Array(x.memory.buffer, p, n * 18);
      for (let k = 0; k < n; k++) {
        const r = list[k];
        const base = k * 18;
        u[base] = r.instance;
        u[base + 1] = r.visible === false ? 0 : 1;
        f.set(r.matrix, base + 2);
      }
      ok('setInstances', x.ganbate_rt_instances_set(p, n));
    },
    setInstanceMaterial(instance, material) {
      ok('setInstanceMaterial', x.ganbate_rt_instance_material(instance, material));
    },
    /** Multiplies an instance's material colour and alpha: `[r, g, b, a]`. */
    setInstanceTint(instance, tint) {
      ok('setInstanceTint', x.ganbate_rt_instance_tint(instance, tint[0], tint[1], tint[2], tint[3] ?? 1));
    },
    /** The pick layers (bit mask) an instance answers to; 0: never picked. New ones: 1. */
    setInstancePick(instance, layers) {
      ok('setInstancePick', x.ganbate_rt_instance_pick(instance, layers >>> 0));
    },
    destroyInstance(instance) { x.ganbate_rt_instance_destroy(instance); },
    /** A light the game hangs: a point light, or a spot with a `direction` and its `inner` and
     * `outer` half-angles (radians); it shines at every hour unless it `followsLamps`. */
    createLight(light) {
      const [p] = place(lightFloats(light));
      return ok('createLight', x.ganbate_rt_light_create(p));
    },
    setLight(id, light) {
      const [p] = place(lightFloats(light));
      ok('setLight', x.ganbate_rt_light_set(id, p));
    },
    destroyLight(id) { x.ganbate_rt_light_destroy(id); },
    /** Where a world point lands, in CSS pixels of the canvas (and depth in metres), or null
     * behind the camera. */
    project(point) {
      const { sx, sy } = cssScale();
      const p = out(3).byteOffset;
      if (!x.ganbate_rt_project(point[0], point[1], point[2], p)) return null;
      const [px, py, depth] = new Float32Array(x.memory.buffer, p, 3);
      return { x: px * sx, y: py * sy, depth };
    },
    /** The visible prop under a CSS pixel of the canvas on one of the pick `layers`:
     * `{instance, distance, point}` or null. Overlays drawn on top win (the last drawn), then
     * the nearest hit. */
    pick(cssX, cssY, layers = 0xFFFFFFFF) {
      const { sx, sy } = cssScale();
      const p = out(4).byteOffset;
      const instance = x.ganbate_rt_pick(cssX / sx, cssY / sy, layers >>> 0, p);
      if (!instance) return null;
      const [distance, px, py, pz] = new Float32Array(x.memory.buffer, p, 4);
      return { instance, distance, point: [px, py, pz] };
    },
    /** The visible prop along a 3D world-space ray `(origin, direction)` on one of `layers`:
     * `{instance, distance, point}` or `null`. */
    pickRay(origin, direction, layers = 0xFFFFFFFF) {
      if (typeof x.ganbate_rt_pick_ray !== 'function') return null;
      if (!origin || !direction || origin.length < 3 || direction.length < 3) return null;
      const p = out(4).byteOffset;
      const instance = x.ganbate_rt_pick_ray(
        origin[0], origin[1], origin[2],
        direction[0], direction[1], direction[2],
        layers >>> 0,
        p,
      );
      if (!instance) return null;
      const [distance, px, py, pz] = new Float32Array(x.memory.buffer, p, 4);
      return { instance, distance, point: [px, py, pz] };
    },
    /** How many instances the cooked world has. */
    worldInstances() { return x.ganbate_rt_world_instances(); },
    /** World instance `index`'s matrix (16 numbers) and whether it is shown. */
    worldInstance(index) {
      const p = out(16).byteOffset;
      const r = x.ganbate_rt_world_instance(index >>> 0, p);
      if (!r) throw fail('worldInstance');
      return { matrix: Array.from(new Float32Array(x.memory.buffer, p, 16)), visible: r === 1 };
    },
    /** Moves world instance `index` to `matrix` and shows or hides it (an object of a cooked
     * scene: `worldEntities()` lists each entity's instances). */
    setWorldInstance(index, matrix, visible = true) {
      const [p] = place(f32(matrix));
      ok('setWorldInstance', x.ganbate_rt_set_world_instance(index >>> 0, p, visible ? 1 : 0));
    },
    /** Moves world lamp `index` (a cooked point light) to `position`. */
    setWorldLight(index, position) {
      ok('setWorldLight', x.ganbate_rt_set_world_light(index >>> 0, position[0], position[1], position[2]));
    },
    /**
     * Starts (or restarts) simulating the cooked world's physics (its `rigid_body` and
     * `collider` components) from the authored state. Returns false when it has none, or when
     * they could not be built: `physicsError()` then says why.
     */
    physicsStart() { return x.ganbate_rt_physics_start() !== 0; },
    /**
     * Steps the simulation `steps` times (default 1, at most 64 a call) by `dt` seconds each (at
     * most 1/15), then moves the instances and lamps of the bodies that moved (with their
     * children's). Sleeping and still bodies cost nothing; several steps in one call move things
     * once. Steps past 64 are not run, so a host that fell further behind (or passes a frame
     * count or milliseconds) loses that time instead of freezing the page while it catches up.
     */
    physicsStep(dt, steps = 1) {
      const n = Math.floor(steps);
      // NaN and negative counts take no step; huge ones stay huge (the engine caps them at 64).
      return x.ganbate_rt_physics_step(dt, n > 0 ? Math.min(n, 0xffffffff) : 0) !== 0;
    },
    /** Stops simulating and puts every moved instance and lamp back where it was cooked. */
    physicsStop() { x.ganbate_rt_physics_stop(); },
    /**
     * Why the cooked world's physics components could not be built (the entity named), or null
     * when they were or it has none.
     */
    physicsError() {
      const len = x.ganbate_rt_physics_error(0, 0);
      if (!len) return null;
      const p = reserve(len);
      x.ganbate_rt_physics_error(p, len);
      return new TextDecoder().decode(new Uint8Array(x.memory.buffer, p, len));
    },
    /** The simulation's state hash as 16 hexadecimal digits (equal to a native run's), or null. */
    physicsHash() {
      const p = out(2).byteOffset;
      if (!x.ganbate_rt_physics_hash(p)) return null;
      const b = new Uint8Array(x.memory.buffer, p, 8);
      return Array.from(b).reverse().map((v) => v.toString(16).padStart(2, '0')).join('');
    },
    /** The cooked world's instance under a CSS pixel: `{instance, distance, point}` or null. */
    pickWorld(cssX, cssY) {
      const { sx, sy } = cssScale();
      const p = out(4).byteOffset;
      const hit = x.ganbate_rt_pick_world(cssX / sx, cssY / sy, p);
      if (!hit) return null;
      const [distance, px, py, pz] = new Float32Array(x.memory.buffer, p, 4);
      return { instance: hit - 1, distance, point: [px, py, pz] };
    },
    /** The cooked world's instance along a 3D world-space ray `(origin, direction)`:
     * `{instance, distance, point}` or `null`. */
    pickWorldRay(origin, direction) {
      if (typeof x.ganbate_rt_pick_world_ray !== 'function') return null;
      if (!origin || !direction || origin.length < 3 || direction.length < 3) return null;
      const p = out(4).byteOffset;
      const hit = x.ganbate_rt_pick_world_ray(
        origin[0], origin[1], origin[2],
        direction[0], direction[1], direction[2],
        p,
      );
      if (!hit) return null;
      const [distance, px, py, pz] = new Float32Array(x.memory.buffer, p, 4);
      return { instance: hit - 1, distance, point: [px, py, pz] };
    },
    /** The world's entity table, `{entities, data}`, or null when it was cooked without one. */
    worldEntities() {
      const len = x.ganbate_world_entities(0, 0);
      if (!len) return null;
      const p = reserve(len);
      x.ganbate_world_entities(p, len);
      return JSON.parse(new TextDecoder().decode(new Uint8Array(x.memory.buffer, p, len)));
    },
    ...graphCalls(x, { reserve, place }),
    /**
     * Loads a glTF model (GLB): `source` is a URL, ArrayBuffer or Uint8Array. Its images are
     * decoded by the browser into textures. Resolves with `{model, info, textures}`, where
     * `info` lists images, nodes (name, parent, mesh, skin), meshes (morph targets), clips
     * (name, duration) and materials by index.
     */
    async loadModel(source) {
      let bytes;
      if (typeof source === 'string') {
        const r = await fetch(source);
        if (!r.ok) throw new Error(`${source}: HTTP ${r.status} ${r.statusText}`);
        bytes = new Uint8Array(await r.arrayBuffer());
      } else {
        bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
      }
      const [p] = place(bytes);
      const parsed = ok('loadModel', x.ganbate_rt_model_parse(p, bytes.length));
      const fetchOut = (call) => {
        const len = call(0, 0);
        const at = reserve(len);
        call(at, len);
        return new Uint8Array(x.memory.buffer, at, len).slice();
      };
      const info = JSON.parse(new TextDecoder().decode(
        fetchOut((at, cap) => x.ganbate_rt_model_describe(parsed, at, cap))));
      // Copy every image out of wasm memory before awaiting the decoders.
      const encoded = info.images.map((im, i) => ({
        ...im,
        bytes: fetchOut((at, cap) => x.ganbate_rt_model_image(parsed, i, at, cap)),
      }));
      const textures = await Promise.all(encoded.map(async (im) => {
        if (im.mime === 'image/png') {
          // The engine's decoder keeps the colour of transparent pixels.
          const [p] = place(im.bytes);
          const t = x.ganbate_rt_texture_png(p, im.bytes.length, im.srgb ? 1 : 0);
          if (!t) onLog(1, `model image ${im.name || ''} not decoded: ${lastError}`);
          return t;
        }
        try {
          const bitmap = await createImageBitmap(new Blob([im.bytes], { type: im.mime }),
            { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
          const t = engine.createTexture(bitmap, { srgb: im.srgb });
          if (bitmap.close) bitmap.close();
          return t;
        } catch (e) {
          onLog(1, `model image ${im.name || ''} (${im.mime}) not decoded: ${e.message}`);
          return 0;
        }
      }));
      const [tp] = place(new Uint32Array(textures));
      const model = ok('loadModel', x.ganbate_rt_model_create(parsed, tp, textures.length));
      return { model, info, textures };
    },
    /** Destroys a model and its instances (its textures stay: destroy them separately). */
    destroyModel(model) { x.ganbate_rt_model_destroy(model); },
    /** The material made for the model's glTF material `index`, to restyle with updateMaterial. */
    modelMaterial(model, index) { return ok('modelMaterial', x.ganbate_rt_model_material(model, index)); },
    createModelInstance(model, matrix) {
      const [p] = place(f32(matrix));
      return ok('createModelInstance', x.ganbate_rt_model_instance(model, p));
    },
    setModelInstance(instance, matrix, visible = true) {
      const [p] = place(f32(matrix));
      ok('setModelInstance', x.ganbate_rt_model_instance_set(instance, p, visible ? 1 : 0));
    },
    destroyModelInstance(instance) { x.ganbate_rt_model_instance_destroy(instance); },
    /** Resets the pose to the model's rest pose. */
    modelRest(instance) { ok('modelRest', x.ganbate_rt_model_rest(instance)); },
    /** Snapshots `instance`'s current pose inside Rust WASM memory for zero-copy Hermite cross-fade. */
    snapshotModelPose(instance) { ok('snapshotModelPose', x.ganbate_rt_model_snapshot_pose(instance)); },
    /** Evaluates clips, cross-fade, Look-At, Two-Bone IK, bone offsets, morphs, and GPU commit in one WASM call. */
    poseModelClip(instance, packet) {
      const a = f32(packet);
      const [p] = place(a);
      ok('poseModelClip', x.ganbate_rt_pose_model_clip(instance, p, a.length));
    },
    /** Mixes clip `clip` (index into info.clips) at `time` seconds into the pose with
     * `weight` (1 replaces what it animates, less mixes toward it). */
    modelClip(instance, clip, time, weight = 1) {
      ok('modelClip', x.ganbate_rt_model_clip(instance, clip, time, weight));
    },
    /** A node's local transform: `{translation, rotation: [x, y, z, w], scale}`. */
    modelNode(instance, node) {
      const at = out(10).byteOffset;
      ok('modelNode', x.ganbate_rt_model_node_get(instance, node, at));
      const v = Array.from(new Float32Array(x.memory.buffer, at, 10));
      return { translation: v.slice(0, 3), rotation: v.slice(3, 7), scale: v.slice(7, 10) };
    },
    setModelNode(instance, node, { translation, rotation, scale }) {
      const [p] = place(f32([...translation, ...rotation, ...scale]));
      ok('setModelNode', x.ganbate_rt_model_node_set(instance, node, p));
    },
    modelMorph(instance, node, target, weight) {
      ok('modelMorph', x.ganbate_rt_model_morph(instance, node, target, weight));
    },
    /** Applies the pose (node transforms, skinning, morphs) to what is drawn. */
    updateModel(instance) { ok('updateModel', x.ganbate_rt_model_update(instance)); },
    /** A node's world matrix as of the last update (16 numbers, column-major). */
    modelNodeWorld(instance, node) {
      const at = out(16).byteOffset;
      ok('modelNodeWorld', x.ganbate_rt_model_node_world(instance, node, at));
      return new Float32Array(x.memory.buffer, at, 16).slice();
    },
    setModelNodeVisible(instance, node, visible) {
      ok('setModelNodeVisible', x.ganbate_rt_model_node_visible(instance, node, visible ? 1 : 0));
    },
    setModelTint(instance, tint) {
      ok('setModelTint', x.ganbate_rt_model_tint(instance, tint[0], tint[1], tint[2], tint[3] ?? 1));
    },
    /** The model instance and node a picked prop instance draws, or null. */
    modelOwner(propInstance) {
      const at = reserve(4);
      const instance = x.ganbate_rt_model_owner(propInstance, at);
      if (!instance) return null;
      return { instance, node: new Uint32Array(x.memory.buffer, at, 1)[0] };
    },
    /** Releases the engine's GPU objects. */
    destroy() {
      if (stopped) return;
      stopped = true;
      gate.stop();
      x.ganbate_rt_stop();
      if (settled) host.release();
      observer?.disconnect();
      drawn = null;
      freeScratch();
    },
  };
  return engine;
}
