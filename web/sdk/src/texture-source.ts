import type {
  GanbateGpuInfo,
  GanbateImageSource,
  GanbateQualityMode,
  GanbateStereoEye,
  GanbateStereoEyeName,
} from './types.ts';

interface SizedImageLike {
  readonly naturalWidth?: number;
  readonly naturalHeight?: number;
  readonly videoWidth?: number;
  readonly videoHeight?: number;
  readonly width?: number;
  readonly height?: number;
  readonly data?: ArrayLike<number> & ArrayBufferView;
}

/** An image source's size in pixels: an image's natural size, a video's frame size. */
export function sourceSize(source: unknown): [number, number] {
  const s = (source ?? {}) as SizedImageLike;
  return [
    s.naturalWidth || s.videoWidth || s.width || 0,
    s.naturalHeight || s.videoHeight || s.height || 0,
  ];
}

/** `source` when it is set and `width` x `height`; throws otherwise (a host's upload check). */
export function sourceOfSize<T>(source: T | null | undefined, width: number, height: number): T {
  if (!source) throw new Error('no source image is set for this call');
  const [w, h] = sourceSize(source);
  if (w !== width || h !== height) throw new Error(`the source image is ${w}x${h}, not ${width}x${height}`);
  return source;
}

/**
 * Whether the browser can copy `source` into a texture itself: canvases and videos, and on WebGPU
 * image bitmaps and images too. `{width, height, data}` stays pixels.
 */
export function copiesDirectly(source: unknown, webgpu: boolean): boolean {
  if (!source || typeof source !== 'object' || (source as SizedImageLike).data) return false;
  if (webgpu) return true;
  const bitmap = typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap;
  return !bitmap && !('naturalWidth' in source);
}

/** RGBA8 pixels of an image source: `{width, height, data}` passes through; canvases, image
 * bitmaps and images are drawn into a 2D canvas and read back (straight alpha). */
export function pixelsOf(source: GanbateImageSource): {
  readonly width: number;
  readonly height: number;
  readonly data: ArrayLike<number> & ArrayBufferView;
} {
  const raw = source as SizedImageLike;
  if (raw && raw.data && raw.width && raw.height) {
    return { width: raw.width, height: raw.height, data: raw.data };
  }
  const [width, height] = sourceSize(source);
  if (!width || !height) throw new Error('createTexture: the source has no size');
  const c =
    typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(width, height)
      : Object.assign(document.createElement('canvas'), { width, height });
  const ctx = c.getContext('2d', { willReadFrequently: true }) as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null;
  if (!ctx) throw new Error('createTexture: 2D canvas context unavailable');
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(source as CanvasImageSource, 0, 0, width, height);
  return { width, height, data: ctx.getImageData(0, 0, width, height).data };
}

/**
 * Maps a `quality` mode (`'realtime' | 'still' | 'hybrid' | 'low-cost' | boolean`) to the C-ABI
 * preset code (`0` = realtime, `1` = still, `2` = hybrid stage, `3` = low-cost fast-path), or
 * `null` when unrecognized.
 */
export function qualityCodeFromMode(mode: GanbateQualityMode | boolean | string | undefined): 0 | 1 | 2 | 3 | null {
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
export function resolveQualityCode(
  quality: GanbateQualityMode | boolean | string | undefined,
  still?: boolean,
): 0 | 1 | 2 | 3 {
  const explicit = qualityCodeFromMode(quality);
  if (explicit !== null) return explicit;
  return still === true ? 1 : 0;
}

/**
 * Whether the tone-map output should grade each pixel once on `gpu` (`createGpu` or `createGl`),
 * as `createEngine`'s `gradeOnce: 'auto'` decides: a WebGPU adapter whose vendor is 'arm', or a
 * WebGL 2 context whose renderer string names Mali and that renders 32-bit floats (`EXT_color_buffer_float`).
 */
export function gradeOnceFor(gpu: GanbateGpuInfo): boolean {
  if (gpu.api === 'webgl2') return Boolean(gpu.ext?.colorFloat) && /mali/i.test(gpu.renderer || '');
  return String(gpu.adapter?.info?.vendor || '').toLowerCase() === 'arm';
}

/** `createEngine`'s `gradeOnce` option (true, false or 'auto') for `gpu`. */
export function resolveGradeOnce(option: boolean | 'auto' | string | undefined, gpu: GanbateGpuInfo): boolean {
  return option === 'auto' ? gradeOnceFor(gpu) : option === true;
}

export function resolveStereoEyeCode(eye: GanbateStereoEyeName | GanbateStereoEye | undefined): -1 | 0 | 1 | null {
  if (eye === undefined || eye === 'mono' || eye === 0) return 0;
  if (eye === 'left' || eye === -1) return -1;
  if (eye === 'right' || eye === 1) return 1;
  return null;
}
