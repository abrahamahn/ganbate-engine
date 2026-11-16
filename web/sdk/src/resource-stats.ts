import type { GanbateResourceStats, GanbateResourceTotals } from './types.ts';

type ResourceKind = 'buffers' | 'textures';
type MutableTotals = { -readonly [K in keyof GanbateResourceTotals]: number };

const emptyTotals = (): MutableTotals => ({
  liveCount: 0, peakCount: 0, allocations: 0,
  estimatedBytes: 0, peakEstimatedBytes: 0, allocatedEstimatedBytes: 0,
  unknownSizeCount: 0, unknownSizeAllocations: 0,
});

/** Update only on allocation/deletion. The ledger does not retain GPU objects. */
export function createResourceTracker() {
  const records = new WeakMap<object, { kind: ResourceKind; bytes: number | null }>();
  const buffers = emptyTotals();
  const textures = emptyTotals();
  const total = emptyTotals();
  const groups = { buffers, textures };
  let closed = false;

  return {
    allocate(kind: ResourceKind, resource: object | null | undefined, estimatedBytes: number | null): void {
      if (closed || !resource || records.has(resource)) return;
      const bytes = estimatedBytes !== null && Number.isSafeInteger(estimatedBytes) && estimatedBytes >= 0
        ? estimatedBytes : null;
      records.set(resource, { kind, bytes });
      for (const group of [groups[kind], total]) {
        group.liveCount++;
        group.allocations++;
        group.peakCount = Math.max(group.peakCount, group.liveCount);
        if (bytes === null) {
          group.unknownSizeCount++;
          group.unknownSizeAllocations++;
        } else {
          group.estimatedBytes += bytes;
          group.allocatedEstimatedBytes += bytes;
          group.peakEstimatedBytes = Math.max(group.peakEstimatedBytes, group.estimatedBytes);
        }
      }
    },
    release(resource: object | null | undefined): void {
      if (closed || !resource) return;
      const record = records.get(resource);
      if (!record) return;
      records.delete(resource);
      for (const group of [groups[record.kind], total]) {
        group.liveCount--;
        if (record.bytes === null) group.unknownSizeCount--;
        else group.estimatedBytes -= record.bytes;
      }
    },
    /** The device/context cannot recover within the same host generation. */
    close(): void {
      closed = true;
      for (const group of [buffers, textures, total]) {
        group.liveCount = group.estimatedBytes = group.unknownSizeCount = 0;
      }
    },
    snapshot(): GanbateResourceStats {
      return { buffers: { ...buffers }, textures: { ...textures }, total: { ...total } };
    },
  };
}

// Capture readback lives outside the host's WASM import table but belongs to the
// same session. Keep this seam internal, without exposing mutable counters in the SDK.
const hostTrackers = new WeakMap<object, ReturnType<typeof createResourceTracker>>();
export function registerResourceTracker<T extends object>(host: T, tracker: ReturnType<typeof createResourceTracker>): T {
  hostTrackers.set(host, tracker);
  return host;
}
export function hostResourceTracker(host: object): ReturnType<typeof createResourceTracker> | undefined {
  return hostTrackers.get(host);
}

export interface TextureAllocationSize {
  readonly format: string;
  readonly width: number;
  readonly height?: number;
  readonly depthOrArrayLayers?: number;
  readonly mipLevelCount?: number;
  readonly sampleCount?: number;
  readonly dimension?: '2d' | '3d';
}

const uncompressedBytes: Readonly<Record<string, number>> = {
  r8unorm: 1, r8snorm: 1, r8uint: 1, r8sint: 1,
  r16uint: 2, r16sint: 2, r16float: 2, r16unorm: 2, r16snorm: 2,
  rg8unorm: 2, rg8snorm: 2, rg8uint: 2, rg8sint: 2,
  r32uint: 4, r32sint: 4, r32float: 4,
  rg16uint: 4, rg16sint: 4, rg16float: 4, rg16unorm: 4, rg16snorm: 4,
  rgba8unorm: 4, 'rgba8unorm-srgb': 4, rgba8snorm: 4, rgba8uint: 4, rgba8sint: 4,
  bgra8unorm: 4, 'bgra8unorm-srgb': 4, rgb9e5ufloat: 4,
  rgb10a2uint: 4, rgb10a2unorm: 4, rg11b10ufloat: 4,
  rg32uint: 8, rg32sint: 8, rg32float: 8,
  rgba16uint: 8, rgba16sint: 8, rgba16float: 8, rgba16unorm: 8, rgba16snorm: 8,
  rgba32uint: 16, rgba32sint: 16, rgba32float: 16,
  stencil8: 1, depth16unorm: 2, depth32float: 4,
  // depth24plus and combined depth/stencil formats have implementation-dependent
  // storage. Report them as unknown instead of silently assuming a packed size.
};

function textureBlock(format: string): readonly [number, number, number] | null {
  const bytes = uncompressedBytes[format];
  if (bytes !== undefined) return [1, 1, bytes];
  // https://www.w3.org/TR/webgpu/#texture-format-caps
  if (/^bc(?:1-rgba-unorm(?:-srgb)?|4-r-(?:unorm|snorm))$/.test(format)) return [4, 4, 8];
  if (/^bc(?:[237]-rgba-unorm(?:-srgb)?|5-rg-(?:unorm|snorm)|6h-rgb-(?:ufloat|float))$/.test(format)) return [4, 4, 16];
  if (/^(?:etc2-rgb8(?:a1)?unorm(?:-srgb)?|eac-r11(?:unorm|snorm))$/.test(format)) return [4, 4, 8];
  if (/^(?:etc2-rgba8unorm(?:-srgb)?|eac-rg11(?:unorm|snorm))$/.test(format)) return [4, 4, 16];
  const astc = /^astc-(4x4|5x4|5x5|6x5|6x6|8x5|8x6|8x8|10x5|10x6|10x8|10x10|12x10|12x12)-unorm(?:-srgb)?$/.exec(format);
  if (astc) {
    const [width, height] = astc[1].split('x').map(Number);
    return [width, height, 16];
  }
  return null;
}

/** Logical texel/block storage only. No row alignment or opaque driver allocation overhead. */
export function estimateTextureBytes(size: TextureAllocationSize): number | null {
  const block = textureBlock(size.format);
  if (!block) return null;
  let width = size.width;
  let height = size.height ?? 1;
  let depth = size.depthOrArrayLayers ?? 1;
  const levels = size.mipLevelCount ?? 1;
  const samples = size.sampleCount ?? 1;
  if (![width, height, depth, levels, samples].every((value) => Number.isSafeInteger(value) && value > 0)) return null;
  const maxDimension = Math.max(width, height, size.dimension === '3d' ? depth : 1);
  if (levels > Math.floor(Math.log2(maxDimension)) + 1) return null;
  let bytes = 0;
  for (let level = 0; level < levels; level++) {
    bytes += Math.ceil(width / block[0]) * Math.ceil(height / block[1]) * block[2] * depth * samples;
    width = Math.max(1, Math.floor(width / 2));
    height = Math.max(1, Math.floor(height / 2));
    if (size.dimension === '3d') depth = Math.max(1, Math.floor(depth / 2));
  }
  return Number.isSafeInteger(bytes) ? bytes : null;
}
