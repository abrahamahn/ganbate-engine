import type { AnyGpuState, GanbateHostRuntime, GanbateWasmExports } from './internal-types.ts';
import { appleWebKit } from './pass-timing.ts';
import { pixelsOf } from './texture-source.ts';
import type { GanbateApi, GanbateCapturedFrame } from './types.ts';
import { createGl, createGlHost } from './webgl-host.ts';
import type { WebGpuNavigator, WebGpuTexture } from './webgpu-globals.ts';
import { createGpu, createHost } from './webgpu-host.ts';

const KEY_CODES: Readonly<Record<string, number>> = {
  ArrowLeft: 1,
  ArrowRight: 2,
  ArrowUp: 3,
  ArrowDown: 4,
  ShiftLeft: 5,
  ShiftRight: 5,
  Space: 32,
};

export function keyCode(e: KeyboardEvent): number {
  if (e.code in KEY_CODES) return KEY_CODES[e.code];
  let m = /^Key([A-Z])$/.exec(e.code);
  if (m) return m[1].toLowerCase().charCodeAt(0);
  m = /^(?:Digit|Numpad)([0-9])$/.exec(e.code);
  if (m) return m[1].charCodeAt(0);
  return 0;
}

/**
 * Streams `url` into wasm memory from `ganbate_alloc`, reporting progress (0..1) and bytes.
 */
export async function fetchInto(
  url: string,
  exports: Pick<GanbateWasmExports, 'memory' | 'ganbate_alloc' | 'ganbate_free'>,
  onProgress: (progress: number, bytes: number) => void,
): Promise<{ ptr: number; len: number }> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status} ${response.statusText}`);
  const encoding = response.headers.get('content-encoding');
  const announced = Number(response.headers.get('content-length')) || 0;
  const reader = response.body!.getReader();
  let ptr = 0;
  let cap = 0;
  let offset = 0;
  let chunks: Uint8Array[] | null = null;
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

export const nextFrame = (): Promise<void> => new Promise((r) => requestAnimationFrame(() => r()));

/**
 * Decodes the loaded world's JPEG and WebP images with the browser's decoders and hands the
 * pixels to the engine. PNGs stay for the engine's own decoder.
 */
export async function decodeWorldImages(
  x: Pick<
    GanbateWasmExports,
    'memory' | 'ganbate_alloc' | 'ganbate_free' | 'ganbate_world_images' | 'ganbate_world_image' | 'ganbate_world_set_image'
  >,
  onLog: (level: number, text: string) => void,
): Promise<number> {
  const read = (call: (p: number, cap: number) => number) => {
    const len = call(0, 0);
    if (!len) return new Uint8Array(0);
    const p = x.ganbate_alloc(len) >>> 0;
    call(p, len);
    const out = new Uint8Array(x.memory.buffer, p, len).slice();
    x.ganbate_free(p, len);
    return out;
  };
  const list = (
    JSON.parse(
      new TextDecoder().decode(read((p, cap) => x.ganbate_world_images(p, cap))),
    ) as { readonly texture: number; readonly mime: string }[]
  ).filter((im) => im.mime !== 'image/png');
  const decoded = await Promise.all(
    list.map(async (im) => {
      const bytes = read((p, cap) => x.ganbate_world_image(im.texture, p, cap));
      try {
        const bitmap = await createImageBitmap(
          new Blob([bytes as unknown as BlobPart], { type: im.mime }),
          {
            premultiplyAlpha: 'none',
            colorSpaceConversion: 'none',
          },
        );
        const pixels = pixelsOf(bitmap);
        if (bitmap.close) bitmap.close();
        return { im, pixels };
      } catch (e) {
        onLog(
          1,
          `world image ${im.texture} (${im.mime}) not decoded: ${e instanceof Error ? e.message : String(e)}`,
        );
        return null;
      }
    }),
  );
  for (const d of decoded) {
    if (!d) continue;
    const { width, height, data } = d.pixels;
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const p = x.ganbate_alloc(bytes.length) >>> 0;
    new Uint8Array(x.memory.buffer, p, bytes.length).set(bytes);
    x.ganbate_world_set_image(d.im.texture, width, height, p, bytes.length);
    x.ganbate_free(p, bytes.length);
  }
  return list.length;
}

export interface BootOptions {
  readonly canvas: HTMLCanvasElement;
  readonly wasm: string | ArrayBuffer | ArrayBufferView | WebAssembly.Instance;
  readonly api?: GanbateApi;
  readonly present?: 'direct' | 'settled';
  readonly bundles?: boolean;
  readonly checkErrors?: boolean;
  readonly onPhase?: (text: string, progress: number) => void;
  readonly onLog?: (level: number, text: string) => void;
  readonly onStatus?: (text: string) => void;
  readonly onError?: (text: string) => void;
}

/** WebGPU/WebGL 2, the host and the wasm module; shared by the player, self-test and `createEngine`. */
export async function boot(options: BootOptions): Promise<{
  gpu: AnyGpuState;
  host: GanbateHostRuntime;
  x: GanbateWasmExports;
  errors: string[];
  fail: (text: string) => void;
  settled: boolean;
}> {
  const { canvas, onPhase = () => {}, onLog = () => {} } = options;
  const errors: string[] = [];
  const fail = (text: string) => {
    errors.push(text);
    (options.onError || (() => {}))(text);
  };
  const api = options.api || 'auto';
  let gpu: AnyGpuState | null = null;
  const nav = typeof navigator !== 'undefined' ? (navigator as unknown as WebGpuNavigator) : null;
  if (api !== 'webgl2' && nav && nav.gpu) {
    onPhase('Starting WebGPU', 0);
    try {
      gpu = await createGpu(canvas);
    } catch (e) {
      if (api === 'webgpu') throw e;
      onLog(1, `WebGPU unavailable (${e instanceof Error ? e.message : String(e)}); using WebGL 2`);
    }
  } else if (api === 'webgpu') {
    throw new Error(
      'This browser has no WebGPU. Use Chrome or Edge 113+ or Safari 26+, served over https or from localhost.',
    );
  }
  if (!gpu) {
    onPhase('Starting WebGL 2', 0);
    gpu = createGl(canvas);
  }
  onLog(0, `${gpu.adapterName}; features: ${gpu.features.join(', ') || 'none'}; canvas ${gpu.format}`);
  const settled = gpu.api === 'webgpu' && options.present === 'settled';
  const hostOptions = {
    onLog,
    onStatus: options.onStatus || (() => {}),
    onError: fail,
    checkErrors: !!options.checkErrors,
    settled,
    bundles: options.bundles ?? !appleWebKit(),
  };
  const host = gpu.api === 'webgl2' ? createGlHost(gpu, hostOptions) : createHost(gpu, hostOptions);
  onPhase('Loading the engine', 0);
  let instance: WebAssembly.Instance;
  if (
    options.wasm &&
    typeof options.wasm === 'object' &&
    'exports' in options.wasm &&
    options.wasm.exports
  ) {
    instance = options.wasm as WebAssembly.Instance;
  } else if (options.wasm instanceof ArrayBuffer || ArrayBuffer.isView(options.wasm)) {
    ({ instance } = await WebAssembly.instantiate(options.wasm as BufferSource, host.imports));
  } else {
    const wasmResponse = await fetch(options.wasm as string);
    if (!wasmResponse.ok) throw new Error(`${options.wasm}: HTTP ${wasmResponse.status}`);
    try {
      ({ instance } = await WebAssembly.instantiateStreaming(wasmResponse.clone(), host.imports));
    } catch {
      ({ instance } = await WebAssembly.instantiate(await wasmResponse.arrayBuffer(), host.imports));
    }
  }
  host.attach(instance);
  return { gpu, host, x: instance.exports as unknown as GanbateWasmExports, errors, fail, settled };
}

/** Reads `texture` (the canvas texture of the current frame) back as RGBA8 pixels. */
export async function readPixels(
  gpu: AnyGpuState,
  texture: unknown,
  host: GanbateHostRuntime,
): Promise<GanbateCapturedFrame> {
  if (gpu.api === 'webgl2') {
    const surface = host.readSurface?.();
    if (!surface) throw new Error('capture: no WebGL 2 surface rendered');
    return surface;
  }
  const gpuTex = texture as WebGpuTexture;
  const width = gpuTex.width;
  const height = gpuTex.height;
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
  const buffer = gpu.device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = gpu.device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: gpuTex }, { buffer, bytesPerRow }, [width, height, 1]);
  gpu.device.queue.submit([enc.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(buffer.getMappedRange());
  const rgba = new Uint8ClampedArray(width * height * 4);
  const bgra = (gpu.format || '').startsWith('bgra');
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const s = y * bytesPerRow + x * 4;
      const d = (y * width + x) * 4;
      rgba[d] = src[s + (bgra ? 2 : 0)];
      rgba[d + 1] = src[s + 1];
      rgba[d + 2] = src[s + (bgra ? 0 : 2)];
      rgba[d + 3] = 255;
    }
  }
  buffer.unmap();
  buffer.destroy();
  return { width, height, rgba };
}
