import type { GanbateHostRuntime, WebGpuState } from './internal-types.ts';
import { createPassTiming, GL_IMPORTS } from './pass-timing.ts';
import { sourceOfSize } from './texture-source.ts';
import type { GanbateHostOptions } from './types.ts';
import type {
  WebGpuBuffer,
  WebGpuCanvasContext,
  WebGpuCommandEncoder,
  WebGpuNavigator,
  WebGpuRenderBundleEncoder,
  WebGpuRenderPassEncoder,
  WebGpuTexture,
} from './webgpu-globals.ts';

const FORMATS = [
  null,
  'rgba8unorm',
  'rgba8unorm-srgb',
  'bgra8unorm',
  'bgra8unorm-srgb',
  'rgba16float',
  'rg16float',
  'r16float',
  'r8unorm',
  'r32float',
  'rgba32float',
  'depth32float',
  'bc1-rgba-unorm',
  'bc1-rgba-unorm-srgb',
  'bc7-rgba-unorm',
  'bc7-rgba-unorm-srgb',
  'astc-4x4-unorm',
  'astc-4x4-unorm-srgb',
] as const;

const SURFACE = 0xffffffff;
const EMPTY_GROUP = 0xfffffffe;
const PUSH_RENDER = 0xfffffffd;
const PUSH_COMPUTE = 0xfffffffc;
const PUSH_SLOT = 256;
const VIEW_DIMENSIONS = ['2d', '2d-array', 'cube'] as const;
const SAMPLE_TYPES = ['float', 'unfilterable-float', 'depth', 'sint', 'uint'] as const;
const COMPARES = [
  undefined,
  'never',
  'less',
  'less-equal',
  'equal',
  'greater',
  'greater-equal',
  'not-equal',
  'always',
] as const;
const ADDRESS = ['repeat', 'clamp-to-edge', 'mirror-repeat'] as const;

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

interface TrackedGpuTexture {
  readonly texture: WebGpuTexture;
  readonly format: string;
  readonly compressed?: boolean;
  readonly views: Map<number, unknown>;
  readonly blitGroups?: Map<number, unknown>;
}

/** Creates the WebGPU device and, if `canvas` is given, configures its context. */
export async function createGpu(canvas?: HTMLCanvasElement): Promise<WebGpuState> {
  const nav = navigator as unknown as WebGpuNavigator;
  if (!nav.gpu) {
    throw new Error(
      'This browser has no WebGPU. Use Chrome or Edge 113+ or Safari 26+, served over https or from localhost.',
    );
  }
  const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU is available but no adapter was found.');
  const wanted = [
    'depth-clip-control',
    'float32-filterable',
    'float32-blendable',
    'texture-compression-bc',
    'texture-compression-astc',
    'timestamp-query',
  ];
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
    .filter(Boolean)
    .join(' ');
  let context: WebGpuCanvasContext | null = null;
  let format: string | null = null;
  if (canvas) {
    context = canvas.getContext('webgpu') as unknown as WebGpuCanvasContext;
    format = nav.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format,
      alphaMode: 'opaque',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
  }
  return {
    api: 'webgpu',
    adapter,
    device,
    adapterName,
    features: requiredFeatures,
    canvas,
    context,
    format,
  };
}

/** The import object for the wasm module and the host state behind it. */
export function createHost(
  gpu: WebGpuState,
  {
    onLog = () => {},
    onStatus = () => {},
    onError = () => {},
    settled = false,
    bundles = true,
  }: GanbateHostOptions = {},
): GanbateHostRuntime {
  const { device, canvas, context } = gpu;
  const objects: any[] = [null];
  const freeList: number[] = [];
  const errors: string[] = [];
  let lastError = '';
  let memory!: WebAssembly.Memory;
  let surfaceTexture: WebGpuTexture | null = null;
  let surfaceView: unknown = null;
  /** The image `gpu_copy_source` copies: set by the engine around one runtime call. */
  let source: unknown = null;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  const report = (message: string) => {
    errors.push(message);
    onError(message);
  };
  device.addEventListener('uncapturederror', (e) => report(`WebGPU: ${e.error.message}`));
  device.lost.then((i) => report(`WebGPU device lost (${i.reason}): ${i.message}`));

  const put = (obj: unknown): number => {
    const h = freeList.length ? freeList.pop()! : objects.length;
    objects[h] = obj;
    return h;
  };
  const u8 = (ptr: number, len: number) => new Uint8Array(memory.buffer, ptr >>> 0, len >>> 0);
  const u32 = (ptr: number, n: number) => new Uint32Array(memory.buffer, ptr >>> 0, n >>> 0);
  const str = (ptr: number, len: number) => decoder.decode(u8(ptr, len));
  const writeStr = (s: string, out: number, cap: number) => {
    const bytes = encoder.encode(s);
    const n = Math.min(bytes.length, cap >>> 0);
    u8(out, n).set(bytes.subarray(0, n));
    return n;
  };
  const guard =
    <A extends number[]>(what: string, f: (...args: A) => number) =>
    (...args: A): number => {
      try {
        return f(...args);
      } catch (e) {
        lastError = `${what}: ${e instanceof Error ? e.message : String(e)}`;
        return 0;
      }
    };

  // Settled (`createFrameGate`), frames are drawn into a texture of the canvas's size and copied
  // to the canvas once finished (`present`); one replaced by a resize goes once a newer is shown.
  let stage: WebGpuTexture | null = null;
  const retired: WebGpuTexture[] = [];
  const presentGroups = new WeakMap<object, Map<number, unknown>>();
  const currentTexture = (): WebGpuTexture => {
    if (surfaceTexture) return surfaceTexture;
    if (!settled) {
      surfaceTexture = context!.getCurrentTexture();
    } else {
      if (!stage || stage.width !== canvas!.width || stage.height !== canvas!.height) {
        if (stage) retired.push(stage);
        stage = device.createTexture({
          size: [canvas!.width, canvas!.height],
          format: gpu.format,
          usage:
            GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
      }
      surfaceTexture = stage;
    }
    return surfaceTexture;
  };
  const texture = (h: number): TrackedGpuTexture =>
    h >>> 0 === SURFACE
      ? { texture: currentTexture(), format: gpu.format || 'rgba8unorm', views: new Map() }
      : (objects[h >>> 0] as TrackedGpuTexture);
  const viewOf = (
    t: TrackedGpuTexture,
    dimension: string,
    baseMipLevel: number,
    mipLevelCount: number,
    baseArrayLayer: number,
    arrayLayerCount: number,
  ) => {
    const dim = dimension === '2d' ? 0 : dimension === '2d-array' ? 1 : 2;
    const key =
      dim +
      baseMipLevel * 4 +
      mipLevelCount * 128 +
      baseArrayLayer * 4096 +
      arrayLayerCount * 16777216;
    let v = t.views.get(key);
    if (!v) {
      v = t.texture.createView({ dimension, baseMipLevel, mipLevelCount, baseArrayLayer, arrayLayerCount });
      t.views.set(key, v);
    }
    return v;
  };
  const attachment = (h: number, mip: number, layer: number) => {
    if (h >>> 0 === SURFACE) {
      if (!surfaceView) surfaceView = currentTexture().createView();
      return surfaceView;
    }
    return viewOf(objects[h] as TrackedGpuTexture, '2d', mip, 1, layer, 1);
  };

  // Shared layouts and groups: the empty group and the push-constant ring.
  const emptyLayout = device.createBindGroupLayout({ entries: [] });
  const emptyGroup = device.createBindGroup({ layout: emptyLayout, entries: [] });
  const pushLayout = (visibility: number) =>
    device.createBindGroupLayout({
      entries: [{ binding: 0, visibility, buffer: { type: 'uniform', hasDynamicOffset: true } }],
    });
  const pushRenderLayout = pushLayout(GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT);
  const pushComputeLayout = pushLayout(GPUShaderStage.COMPUTE);

  interface BundleEntry {
    fmtKey: string;
    words: Uint32Array;
    bundle: unknown;
  }
  const bundleBuckets = new Map<number, BundleEntry[]>();
  let bundleEntries = 0;
  let bundleCompiled = 0;
  let bundleReplays = 0;
  const clearBundles = () => {
    bundleBuckets.clear();
    bundleEntries = 0;
    bundleCompiled = 0;
  };
  let ring!: { buffer: WebGpuBuffer; size: number; render: unknown; compute: unknown };
  const ensureRing = (len: number) => {
    if (ring && ring.size >= len) return;
    let size = 65536;
    while (size < len) size *= 2;
    if (ring) {
      ring.buffer.destroy();
      clearBundles();
    }
    const buffer = device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const makeGroup = (layout: unknown) =>
      device.createBindGroup({
        layout,
        entries: [{ binding: 0, resource: { buffer, offset: 0, size: PUSH_SLOT } }],
      });
    ring = { buffer, size, render: makeGroup(pushRenderLayout), compute: makeGroup(pushComputeLayout) };
  };
  ensureRing(1);
  const group = (h: number) => {
    switch (h >>> 0) {
      case EMPTY_GROUP:
        return emptyGroup;
      case PUSH_RENDER:
        return ring.render;
      case PUSH_COMPUTE:
        return ring.compute;
      default:
        return objects[h >>> 0];
    }
  };

  // Blits and mip generation are render passes.
  const blitModule = device.createShaderModule({ code: BLIT_WGSL });
  const blitPipelines = new Map<string, { id: number; pipeline: unknown; layout: unknown }>();
  let nextBlitId = 0;
  const blitSamplers = [
    device.createSampler({}),
    device.createSampler({ magFilter: 'linear', minFilter: 'linear' }),
  ];
  const blitPipeline = (format: string, linear: boolean | number) => {
    const isLinear = Boolean(linear);
    const key = `${format}/${isLinear}`;
    let p = blitPipelines.get(key);
    if (!p) {
      const layout = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            texture: { sampleType: isLinear ? 'float' : 'unfilterable-float' },
          },
          {
            binding: 1,
            visibility: GPUShaderStage.FRAGMENT,
            sampler: { type: isLinear ? 'filtering' : 'non-filtering' },
          },
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
  const blit = (
    enc: WebGpuCommandEncoder,
    src: number,
    srcMip: number,
    srcLayer: number,
    dst: number,
    dstMip: number,
    dstLayer: number,
    linear: number,
  ) => {
    const s = objects[src] as TrackedGpuTexture;
    const dstFormat = dst >>> 0 === SURFACE ? gpu.format! : (objects[dst] as TrackedGpuTexture).format;
    const p = blitPipeline(dstFormat, linear !== 0);
    const bgKey = p.id + srcMip * 256 + srcLayer * 8192;
    let bindGroup = s.blitGroups?.get(bgKey);
    if (!bindGroup) {
      bindGroup = device.createBindGroup({
        layout: p.layout,
        entries: [
          { binding: 0, resource: viewOf(s, '2d', srcMip, 1, srcLayer, 1) },
          { binding: 1, resource: blitSamplers[linear ? 1 : 0] },
        ],
      });
      s.blitGroups?.set(bgKey, bindGroup);
    }
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: attachment(dst, dstMip, dstLayer),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: [0, 0, 0, 0],
        },
      ],
    });
    pass.setPipeline(p.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  };

  const canBundle = bundles && typeof device.createRenderBundleEncoder === 'function';
  if (settled && gpu.format) blitPipeline(gpu.format, false);

  const timing = createPassTiming();
  let stampable: boolean | null = null;
  const STAMPED_PASSES = 128;
  let stamps: { set: unknown; resolve: WebGpuBuffer; read: WebGpuBuffer } | null = null;
  let stampLabels: string[] = [];
  let frameSent = 0;
  const stampWrites = (k: number) => {
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
  const wordsEqual = (saved: Uint32Array, c: Uint32Array, start: number, len: number) => {
    for (let k = 0; k < len; k++) {
      if (saved[k] !== c[start + k]) return false;
    }
    return true;
  };
  const recordBundleBody = (rbe: WebGpuRenderBundleEncoder, c: Uint32Array, start: number, end: number) => {
    const u64 = (idx: number) => c[idx] + c[idx + 1] * 4294967296;
    let p = start;
    while (p < end) {
      const bop = c[p++];
      switch (bop) {
        case 3:
          rbe.setPipeline(objects[c[p++]]);
          break;
        case 4: {
          const index = c[p];
          const g = c[p + 1];
          const n = c[p + 2];
          if (n === 0) rbe.setBindGroup(index, group(g));
          else rbe.setBindGroup(index, group(g), c, p + 3, n);
          p += 3 + n;
          break;
        }
        case 5:
          rbe.setVertexBuffer(c[p], objects[c[p + 1]], u64(p + 2));
          p += 4;
          break;
        case 6:
          rbe.setIndexBuffer(objects[c[p]], 'uint32', u64(p + 1));
          p += 3;
          break;
        case 9:
          rbe.draw(c[p], c[p + 1], c[p + 2], c[p + 3]);
          p += 4;
          break;
        case 10:
          rbe.drawIndexed(c[p], c[p + 1], c[p + 2], c[p + 3] | 0, c[p + 4]);
          p += 5;
          break;
        default:
          throw new Error(`unsupported bundle op ${bop}`);
      }
    }
  };

  const execute = (cmdPtr: number, words: number, pushPtr: number, pushLen: number) => {
    const c = u32(cmdPtr, words);
    const f = new Float32Array(memory.buffer, cmdPtr >>> 0, words >>> 0);
    const uPushLen = pushLen >>> 0;
    if (uPushLen) {
      ensureRing(uPushLen);
      device.queue.writeBuffer(ring.buffer, 0, memory.buffer, pushPtr >>> 0, uPushLen);
    }
    const enc = device.createCommandEncoder();
    let pass: WebGpuRenderPassEncoder | null = null;
    const u64 = (idx: number) => c[idx] + c[idx + 1] * 4294967296;
    const timed = timing.on();
    if (timed && stampable === null) stampable = device.features.has('timestamp-query');
    let passLabel = '';
    let passSize = '';
    let stamp = -1;
    let i = 0;
    while (i < c.length) {
      const op = c[i++];
      switch (op) {
        case 1: {
          // BEGIN_RENDER
          const colors = c[i++];
          const hasDepth = c[i++];
          if (timed) {
            const t = texture(c[i]).texture;
            passLabel = '';
            passSize = `${Math.max(1, t.width >> c[i + 1])}x${Math.max(1, t.height >> c[i + 1])}`;
            stamp = stampable && stampLabels.length < STAMPED_PASSES ? stampLabels.push('') - 1 : -1;
          }
          const colorAttachments: Record<string, unknown>[] = [];
          const colorFormats: string[] | null = canBundle ? [] : null;
          let h = (2166136261 ^ colors) >>> 0;
          for (let k = 0; k < colors; k++) {
            const th = c[i];
            const fmt = th >>> 0 === SURFACE ? gpu.format! : (objects[th] as TrackedGpuTexture).format;
            if (colorFormats) colorFormats.push(fmt);
            colorAttachments.push({
              view: attachment(th, c[i + 1], c[i + 2]),
              loadOp: c[i + 3] ? 'clear' : 'load',
              storeOp: c[i + 4] ? 'store' : 'discard',
              clearValue: [f[i + 5], f[i + 6], f[i + 7], f[i + 8]],
            });
            i += 9;
          }
          let depthStencilAttachment: Record<string, unknown> | undefined;
          let depthStencilFormat: string | undefined;
          if (hasDepth) {
            depthStencilFormat = (objects[c[i]] as TrackedGpuTexture).format;
            depthStencilAttachment = {
              view: attachment(c[i], c[i + 1], c[i + 2]),
              depthLoadOp: c[i + 3] ? 'clear' : 'load',
              depthStoreOp: c[i + 4] ? 'store' : 'discard',
              depthClearValue: f[i + 5],
            };
            i += 6;
          }
          if (canBundle && !timed && colorFormats) {
            const bodyStart = i;
            let scan = i;
            let eligible = true;
            let draws = 0;
            while (scan < c.length) {
              const bop = c[scan];
              if (bop === 2) break;
              switch (bop) {
                case 3:
                  scan += 2;
                  break;
                case 4:
                  scan += 4 + c[scan + 3];
                  break;
                case 5:
                  scan += 5;
                  break;
                case 6:
                  scan += 4;
                  break;
                case 9:
                  draws++;
                  scan += 5;
                  break;
                case 10:
                  draws++;
                  scan += 6;
                  break;
                case 7:
                case 8:
                  eligible = false;
                  scan += 5;
                  break;
                default:
                  eligible = false;
                  scan = c.length;
                  break;
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
              let entry: BundleEntry | null = null;
              if (bucket) {
                for (let b = 0; b < bucket.length; b++) {
                  const cand = bucket[b];
                  if (
                    cand.fmtKey === fmtKey &&
                    cand.words.length === bodyLen &&
                    wordsEqual(cand.words, c, bodyStart, bodyLen)
                  ) {
                    entry = cand;
                    break;
                  }
                }
              }
              if (entry) {
                if (!entry.bundle) {
                  const rbe = device.createRenderBundleEncoder!({
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
          pass!.end();
          pass = null;
          if (stamp >= 0) stampLabels[stamp] = `${passLabel || 'clear'} ${passSize}`;
          stamp = -1;
          break;
        case 3: {
          const p = objects[c[i++]];
          pass!.setPipeline(p);
          if (timed && !passLabel) passLabel = p.label || '';
          break;
        }
        case 4: {
          const index = c[i];
          const g = c[i + 1];
          const n = c[i + 2];
          if (n === 0) pass!.setBindGroup(index, group(g));
          else pass!.setBindGroup(index, group(g), c, i + 3, n);
          i += 3 + n;
          break;
        }
        case 5:
          pass!.setVertexBuffer(c[i], objects[c[i + 1]], u64(i + 2));
          i += 4;
          break;
        case 6:
          pass!.setIndexBuffer(objects[c[i]], 'uint32', u64(i + 1));
          i += 3;
          break;
        case 7:
          pass!.setViewport(f[i], f[i + 1], f[i + 2], f[i + 3], 0, 1);
          i += 4;
          break;
        case 8:
          pass!.setScissorRect(c[i], c[i + 1], c[i + 2], c[i + 3]);
          i += 4;
          break;
        case 9:
          pass!.draw(c[i], c[i + 1], c[i + 2], c[i + 3]);
          i += 4;
          break;
        case 10:
          pass!.drawIndexed(c[i], c[p_add(i, 1)], c[p_add(i, 2)], c[p_add(i, 3)] | 0, c[p_add(i, 4)]);
          i += 5;
          break;
        case 11: {
          // DISPATCH
          const cp = enc.beginComputePass();
          cp.setPipeline(objects[c[i]]);
          const n = c[i + 1];
          i += 2;
          for (let k = 0; k < n; k++) {
            const index = c[i];
            const g = c[i + 1];
            const no = c[i + 2];
            if (no === 0) cp.setBindGroup(index, group(g));
            else cp.setBindGroup(index, group(g), c, i + 3, no);
            i += 3 + no;
          }
          cp.dispatchWorkgroups(c[i], c[i + 1], c[i + 2]);
          i += 3;
          cp.end();
          break;
        }
        case 12:
          // COPY_BUFFER
          enc.copyBufferToBuffer(objects[c[i]], u64(i + 1), objects[c[i + 3]], u64(i + 4), u64(i + 6));
          i += 8;
          break;
        case 13: {
          // COPY_TEXTURE
          const t = texture(c[i]);
          enc.copyTextureToBuffer(
            { texture: t.texture, mipLevel: c[i + 1], origin: [0, 0, c[i + 2]] },
            { buffer: objects[c[i + 5]], offset: u64(i + 6), bytesPerRow: c[i + 8], rowsPerImage: c[i + 4] },
            [c[i + 3], c[i + 4], 1],
          );
          i += 9;
          break;
        }
        case 14:
          // BLIT
          blit(enc, c[i], c[i + 1], c[i + 2], c[i + 3], c[i + 4], c[i + 5], c[i + 6]);
          i += 7;
          break;
        case 15:
          // COPY_DEPTH
          enc.copyTextureToTexture(
            { texture: (objects[c[i]] as TrackedGpuTexture).texture, mipLevel: c[i + 1], origin: [0, 0, c[i + 2]] },
            {
              texture: (objects[c[i + 3]] as TrackedGpuTexture).texture,
              mipLevel: c[i + 4],
              origin: [0, 0, c[i + 5]],
            },
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

  function p_add(base: number, off: number): number {
    return base + off;
  }

  const imports: Record<string, WebAssembly.ImportValue> = {
    gpu_info(out: number, cap: number) {
      const L = device.limits;
      const features =
        (device.features.has('depth-clip-control') ? 1 : 0) |
        (device.features.has('float32-filterable') ? 2 : 0) |
        (device.features.has('float32-blendable') ? 4 : 0) |
        (device.features.has('texture-compression-bc') ? 8 : 0) |
        (device.features.has('texture-compression-astc') ? 16 : 0);
      const v = [
        L.maxTextureDimension2D,
        L.maxTextureArrayLayers,
        L.maxBindGroups,
        L.minUniformBufferOffsetAlignment,
        L.minStorageBufferOffsetAlignment,
        L.maxUniformBufferBindingSize,
        Math.min(L.maxStorageBufferBindingSize, 0xffffffff),
        Math.min(L.maxBufferSize, 0xffffffff),
        L.maxColorAttachments,
        features,
        L.maxComputeWorkgroupsPerDimension,
        L.maxVertexBufferArrayStride,
        L.maxVertexAttributes,
        0,
        0,
        0,
      ];
      const n = Math.min(v.length, cap >>> 0);
      u32(out, n).set(v.slice(0, n));
      return n;
    },
    gpu_adapter_name: (out: number, cap: number) => writeStr(gpu.adapterName, out, cap),
    gpu_last_error: (out: number, cap: number) => writeStr(lastError, out, cap),
    gpu_error_count: () => errors.length,
    gpu_error: (i: number, out: number, cap: number) => writeStr(errors[i >>> 0] || '', out, cap),
    gpu_canvas(out: number) {
      if (!canvas) return 0;
      const code = Math.max(1, (FORMATS as readonly (string | null)[]).indexOf(gpu.format));
      u32(out, 3).set([canvas.width, canvas.height, code]);
      return 1;
    },
    gpu_create_buffer: guard('createBuffer', (size: number, usage: number) =>
      put(device.createBuffer({ size, usage })),
    ),
    gpu_write_buffer(h: number, offset: number, ptr: number, len: number) {
      device.queue.writeBuffer(objects[h >>> 0], offset, memory.buffer, ptr >>> 0, len >>> 0);
    },
    gpu_create_texture: guard(
      'createTexture',
      (width: number, height: number, layers: number, mips: number, format: number, usage: number) =>
        put({
          texture: device.createTexture({
            size: [width, height, layers],
            mipLevelCount: mips,
            format: FORMATS[format],
            usage,
          }),
          format: FORMATS[format] || 'rgba8unorm',
          compressed: format >= 12,
          views: new Map(),
          blitGroups: new Map(),
        }),
    ),
    gpu_write_texture(
      h: number,
      mip: number,
      layer: number,
      width: number,
      height: number,
      bytesPerRow: number,
      ptr: number,
      _len: number,
    ) {
      const t = objects[h >>> 0] as TrackedGpuTexture;
      const rowsPerImage = t.compressed ? Math.max(1, height >>> 2) : height;
      device.queue.writeTexture(
        { texture: t.texture, mipLevel: mip, origin: [0, 0, layer] },
        memory.buffer,
        { offset: ptr >>> 0, bytesPerRow, rowsPerImage },
        [width, height, 1],
      );
    },
    gpu_copy_source: guard(
      'copyExternalImageToTexture',
      (h: number, mip: number, layer: number, width: number, height: number) => {
        device.queue.copyExternalImageToTexture(
          { source: sourceOfSize(source, width, height), flipY: false },
          {
            texture: (objects[h >>> 0] as TrackedGpuTexture).texture,
            mipLevel: mip,
            origin: [0, 0, layer],
            colorSpace: 'srgb',
            premultipliedAlpha: false,
          },
          [width, height],
        );
        return 1;
      },
    ),
    gpu_generate_mips(h: number) {
      const t = objects[h >>> 0] as TrackedGpuTexture;
      const enc = device.createCommandEncoder();
      for (let layer = 0; layer < t.texture.depthOrArrayLayers; layer++) {
        for (let mip = 1; mip < t.texture.mipLevelCount; mip++) {
          blit(enc, h, mip - 1, layer, h, mip, layer, 1);
        }
      }
      device.queue.submit([enc.finish()]);
    },
    gpu_create_sampler: guard('createSampler', (ptr: number, words: number) => {
      const d = u32(ptr, words);
      const lod = new Float32Array(memory.buffer, (ptr >>> 0) + 20, 1)[0];
      const desc: Record<string, unknown> = {
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
    gpu_create_shader: guard('createShaderModule', (ptr: number, len: number) => {
      const code = str(ptr, len);
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
    gpu_create_bind_group_layout: guard('createBindGroupLayout', (ptr: number, words: number) => {
      const w = u32(ptr, words);
      const entries: Record<string, unknown>[] = [];
      for (let i = 0; i < w.length; i += 6) {
        const [binding, visibility, type, a, b] = w.subarray(i, i + 5);
        const e: Record<string, unknown> = { binding, visibility };
        switch (type) {
          case 0:
            e.buffer = { type: 'uniform', minBindingSize: b };
            break;
          case 1:
            e.buffer = { type: 'read-only-storage' };
            break;
          case 2:
            e.buffer = { type: 'storage' };
            break;
          case 3:
            e.sampler = { type: a === 2 ? 'comparison' : a === 1 ? 'non-filtering' : 'filtering' };
            break;
          case 4:
            e.texture = { sampleType: SAMPLE_TYPES[a], viewDimension: VIEW_DIMENSIONS[b] };
            break;
          case 5:
            e.storageTexture = { access: 'write-only', format: FORMATS[a], viewDimension: VIEW_DIMENSIONS[b] };
            break;
          default:
            throw new Error(`binding type ${type}`);
        }
        entries.push(e);
      }
      return put(device.createBindGroupLayout({ entries }));
    }),
    gpu_create_pipeline_layout: guard('createPipelineLayout', (ptr: number, count: number) => {
      const bindGroupLayouts = Array.from(u32(ptr, count), (h) => {
        switch (h) {
          case 0:
            return emptyLayout;
          case PUSH_RENDER:
            return pushRenderLayout;
          case PUSH_COMPUTE:
            return pushComputeLayout;
          default:
            return objects[h];
        }
      });
      return put(device.createPipelineLayout({ bindGroupLayouts }));
    }),
    gpu_create_render_pipeline: guard('createRenderPipeline', (ptr: number, len: number) => {
      const d = JSON.parse(str(ptr, len));
      d.layout = objects[d.layout];
      d.vertex.module = objects[d.vertex.module];
      if (d.fragment) d.fragment.module = objects[d.fragment.module];
      d.label = (d.fragment || d.vertex).module.label;
      return put(device.createRenderPipeline(d));
    }),
    gpu_create_compute_pipeline: guard('createComputePipeline', (ptr: number, len: number) => {
      const d = JSON.parse(str(ptr, len));
      d.layout = objects[d.layout];
      d.compute.module = objects[d.compute.module];
      return put(device.createComputePipeline(d));
    }),
    gpu_create_bind_group: guard('createBindGroup', (layout: number, ptr: number, words: number) => {
      const w = u32(ptr, words);
      const entries: Record<string, unknown>[] = [];
      for (let i = 0; i < w.length; i += 8) {
        const [binding, kind, h, a, b, c, d, e] = w.subarray(i, i + 8);
        let resource: unknown;
        if (kind === 0) {
          resource = { buffer: objects[h], offset: a + b * 4294967296, size: c + d * 4294967296 };
        } else if (kind === 1) {
          resource = viewOf(objects[h] as TrackedGpuTexture, VIEW_DIMENSIONS[a], b, c, d, e);
        } else {
          resource = objects[h];
        }
        entries.push({ binding, resource });
      }
      return put(device.createBindGroup({ layout: objects[layout >>> 0], entries }));
    }),
    gpu_submit: guard('submit', (cmdPtr: number, words: number, pushPtr: number, pushLen: number) => {
      execute(cmdPtr, words, pushPtr, pushLen);
      return 1;
    }),
    gpu_release(h: number) {
      const uh = h >>> 0;
      const o = objects[uh];
      if (!o) return;
      clearBundles();
      if (o.texture && o.texture.destroy) o.texture.destroy();
      else if (typeof GPUBuffer !== 'undefined' && o instanceof GPUBuffer) o.destroy();
      else if (typeof o.destroy === 'function') o.destroy();
      objects[uh] = null;
      freeList.push(uh);
    },
    log: (level: number, ptr: number, len: number) => onLog(level, str(ptr, len)),
    status: (ptr: number, len: number) => onStatus(str(ptr, len)),
    gpu_api: () => 1,
  };
  const unavailable = () => {
    throw new Error('WebGL 2 call on the WebGPU host');
  };
  for (const name of GL_IMPORTS) imports[name] = unavailable;

  return {
    imports: { ganbate: imports },
    errors,
    attach(instance: WebAssembly.Instance) {
      memory = instance.exports.memory as WebAssembly.Memory;
    },
    setSource(image: unknown) {
      source = image;
    },
    nextFrame() {
      surfaceTexture = null;
      surfaceView = null;
    },
    frameTexture() {
      return surfaceTexture;
    },
    present(frame: WebGpuTexture) {
      const target = context!.getCurrentTexture();
      const linear = frame.width !== target.width || frame.height !== target.height;
      const p = blitPipeline(gpu.format!, linear);
      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [
          { view: target.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
        ],
      });
      pass.setPipeline(p.pipeline);
      let groups = presentGroups.get(frame);
      if (!groups) {
        groups = new Map();
        presentGroups.set(frame, groups);
      }
      let bg = groups.get(p.id);
      if (!bg) {
        bg = device.createBindGroup({
          layout: p.layout,
          entries: [
            { binding: 0, resource: frame.createView() },
            { binding: 1, resource: blitSamplers[linear ? 1 : 0] },
          ],
        });
        groups.set(p.id, bg);
      }
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
      device.queue.submit([enc.finish()]);
      if (frame === stage) {
        while (retired.length) retired.pop()!.destroy();
      }
    },
    release() {
      while (retired.length) retired.pop()!.destroy();
      stage?.destroy();
      stage = null;
    },
    bundleStats: () => ({ cached: bundleCompiled, candidates: bundleEntries, replays: bundleReplays }),
    timing,
    endFrame() {
      if (!timing.on()) return;
      if (!stampLabels.length) {
        const m = timing.mark('frame');
        m.sent = frameSent || m.sent;
        frameSent = 0;
        const frame = timing.take()!;
        const done = () => {
          m.done = performance.now();
          frame.finish();
        };
        device.queue.onSubmittedWorkDone().then(done, done);
        return;
      }
      const labels = stampLabels;
      stampLabels = [];
      for (const label of labels) timing.mark(label);
      const frame = timing.take()!;
      const bytes = labels.length * 16;
      const enc = device.createCommandEncoder();
      enc.resolveQuerySet(stamps!.set, 0, labels.length * 2, stamps!.resolve, 0);
      enc.copyBufferToBuffer(stamps!.resolve, 0, stamps!.read, 0, bytes);
      device.queue.submit([enc.finish()]);
      stamps!.read.mapAsync(GPUMapMode.READ, 0, bytes).then(() => {
        const t = new BigInt64Array(stamps!.read.getMappedRange(0, bytes));
        frame.marks.forEach((m, k) => {
          m.sent = Number(t[2 * k] - t[0]) / 1e6;
          m.done = Number(t[2 * k + 1] - t[0]) / 1e6;
        });
        stamps!.read.unmap();
        frame.finish();
      }, frame.finish);
    },
    setXrTarget: () => false,
  };
}
