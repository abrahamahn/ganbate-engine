import { createFrameGate } from './pass-timing.ts';
import type {
  GanbateCapturedFrame,
  GanbatePlayerController,
  GanbateSelfTestResult,
  StartPlayerOptions,
  StartSelfTestOptions,
} from './types.ts';
import { boot, decodeWorldImages, fetchInto, keyCode, nextFrame, readPixels } from './wasm-boot.ts';

/**
 * Runs the GPU-layer self-test of the wasm module (compute, copies, mips, a blit into the
 * canvas) and resolves with the canvas pixels and any errors.
 */
export async function startSelfTest(options: StartSelfTestOptions): Promise<GanbateSelfTestResult> {
  const { gpu, host, x, errors, fail } = await boot({
    ...options,
    onPhase: options.onPhase ? (text) => options.onPhase!(text) : undefined,
    checkErrors: true,
    present: 'direct',
  });
  if (options.sizeCanvas) options.sizeCanvas();
  const pixels = await new Promise<GanbateCapturedFrame>((resolve, reject) =>
    requestAnimationFrame(() => {
      host.nextFrame();
      let ok = 0;
      try {
        ok = x.ganbate_selftest();
      } catch (e) {
        fail(`the self-test stopped: ${e instanceof Error ? e.message : String(e)}`);
      }
      const texture = host.frameTexture();
      if (!ok || !texture) {
        reject(new Error(errors[errors.length - 1] || 'the self-test presented nothing'));
        return;
      }
      readPixels(gpu, texture, host).then(resolve, reject);
    }),
  );
  if (gpu.api === 'webgpu') await gpu.device.queue.onSubmittedWorkDone();
  await nextFrame();
  return {
    adapter: gpu.adapterName,
    format: gpu.format || 'rgba8unorm',
    pixels,
    errors: () => [...errors, ...host.errors.filter((e) => !errors.includes(e))],
  };
}

/**
 * Loads the player: WebGPU, the wasm module, the world and the Look file; then runs frames.
 */
export async function startPlayer(options: StartPlayerOptions): Promise<GanbatePlayerController> {
  const { canvas, onPhase = () => {} } = options;
  const { gpu, host, x, errors, fail, settled } = await boot(options);
  const call = <T>(what: string, f: () => T): T => {
    try {
      return f();
    } catch (e) {
      throw new Error(
        `${what} stopped: ${e instanceof Error ? e.message : String(e)}${
          errors.length ? ` (${errors[errors.length - 1]})` : ''
        }`,
      );
    }
  };

  const world = await fetchInto(options.scene, x, (p, bytes) =>
    onPhase(`Downloading the world (${(bytes / 1048576).toFixed(1)} MB)`, p),
  );
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
  let views: string[] = [];
  try {
    views = ((JSON.parse(lookText || '{}').views || []) as { name?: unknown }[]).map((v) => String(v.name));
  } catch {
    // The engine reports Look file errors.
  }
  const toWasm = (text: string) => {
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

  const keys = new Set<number>();
  const key = (code: number, down: boolean) => {
    if (!code) return;
    if (down) keys.add(code);
    else keys.delete(code);
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
  window.addEventListener('blur', () => {
    keys.clear();
    x.ganbate_blur();
  });
  const pointers = new Map<number, { x: number; y: number }>();
  let pinch: { x: number; y: number } | null = null;
  const centroid = () => {
    let sx = 0;
    let sy = 0;
    for (const p of pointers.values()) {
      sx += p.x;
      sy += p.y;
    }
    return { x: sx / pointers.size, y: sy / pointers.size };
  };
  canvas.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch {
      // Not an active pointer (for example a synthetic event).
    }
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2) pinch = centroid();
    e.preventDefault();
  });
  canvas.addEventListener('pointermove', (e) => {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x;
    const dy = e.clientY - p.y;
    p.x = e.clientX;
    p.y = e.clientY;
    if (pointers.size === 1) {
      x.ganbate_look(
        dx * (e.pointerType === 'touch' ? 1.4 : 1),
        dy * (e.pointerType === 'touch' ? 1.4 : 1),
      );
    } else if (pointers.size === 2 && pinch) {
      const c = centroid();
      const scale = Math.min(canvas.clientWidth, canvas.clientHeight) * 0.25;
      x.ganbate_move((pinch.y - c.y) / scale, (c.x - pinch.x) / scale);
    }
  });
  const lift = (e: PointerEvent) => {
    pointers.delete(e.pointerId);
    if (pointers.size < 2 && pinch) {
      pinch = null;
      x.ganbate_move(0, 0);
    }
  };
  canvas.addEventListener('pointerup', lift);
  canvas.addEventListener('pointercancel', lift);

  let last = performance.now();
  let running = true;
  let captures: {
    resolve: (frame: GanbateCapturedFrame) => void;
    reject: (err: unknown) => void;
  }[] = [];
  const capture = (texture: unknown) => readPixels(gpu, texture, host);
  const gate = createFrameGate(gpu, host, settled);
  const tick = (now: number) => {
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
      fail(`the engine stopped: ${e instanceof Error ? e.message : String(e)}`);
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
      capture(texture).then(
        (img) => pending.forEach((r) => r.resolve(img)),
        (e) => pending.forEach((r) => r.reject(e)),
      );
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);

  return {
    adapter: gpu.adapterName,
    format: gpu.format || 'rgba8unorm',
    views,
    exports: x as unknown as GanbatePlayerController['exports'],
    errors: () => [...errors, ...host.errors.filter((e) => !errors.includes(e))],
    frames: () => x.ganbate_frames(),
    key,
    stop() {
      running = false;
      gate.stop();
    },
    capture: () => new Promise((resolve, reject) => captures.push({ resolve, reject })),
  };
}
