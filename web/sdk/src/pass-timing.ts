import type { AnyGpuState, GanbateHostRuntime, PassTimingMark, PassTimingTracker } from './internal-types.ts';
import type { WebGpuTexture } from './webgpu-globals.ts';

export const GPU_IMPORTS = [
  'gpu_info',
  'gpu_adapter_name',
  'gpu_last_error',
  'gpu_error_count',
  'gpu_error',
  'gpu_canvas',
  'gpu_create_buffer',
  'gpu_write_buffer',
  'gpu_create_texture',
  'gpu_write_texture',
  'gpu_copy_source',
  'gpu_generate_mips',
  'gpu_create_sampler',
  'gpu_create_shader',
  'gpu_create_bind_group_layout',
  'gpu_create_pipeline_layout',
  'gpu_create_render_pipeline',
  'gpu_create_compute_pipeline',
  'gpu_create_bind_group',
  'gpu_submit',
  'gpu_release',
] as const;

export const GL_IMPORTS = [
  'gl_info',
  'gl_renderer_name',
  'gl_last_error',
  'gl_error_count',
  'gl_error',
  'gl_canvas',
  'gl_create_buffer',
  'gl_write_buffer',
  'gl_read_buffer',
  'gl_create_texture',
  'gl_write_texture',
  'gl_write_texture_source',
  'gl_generate_mips',
  'gl_create_sampler',
  'gl_create_program',
  'gl_create_pipeline',
  'gl_create_vao',
  'gl_submit',
  'gl_release',
] as const;

/**
 * Each pass's GPU time on a device no profiler reaches (an iPhone; 2026-11-12). While it is on, the
 * engine draws one frame at a time (`busy`), so every frame starts on an idle GPU, and a host marks
 * each pass (`mark`) with when it started (`sent`) and finished (`done`): WebGL 2 sends the passes
 * one at a time, each followed by a fence; WebGPU writes timestamps where the GPU has them, and
 * otherwise marks the whole frame. A pass took from the later of its start and the mark before it
 * to its finish, so the time the GPU waited for the page is left out.
 */
export function createPassTiming(): PassTimingTracker {
  let on = false;
  let generation = 0;
  let marks: PassTimingMark[] = [];
  let pending = 0;
  let frames = 0;
  let busyMs = 0;
  let wallMs = 0;
  const sums = new Map<string, number>();
  return {
    on: () => on,
    set(value: boolean) {
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
    mark(label: string) {
      const m: PassTimingMark = { label, sent: performance.now(), done: 0 };
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
export function appleWebKit(): boolean {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent || '';
  const ios = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
  const macSafari =
    /Macintosh/.test(ua) && /Version\/[\d.]+.*Safari\//.test(ua) && !/Chrome|Chromium|Edg|Firefox|OPR/.test(ua);
  return ios || macSafari;
}

export interface FrameGate {
  full(): boolean;
  submitted(): void;
  show(): void;
  reshow(): void;
  stop(): void;
}

/**
 * How many frames are on the GPU at once and, on a settled host, when a finished frame is shown.
 * Direct (the default), the engine draws into the canvas with up to two frames on the GPU. Settled
 * (`present: 'settled'`), it draws offscreen with one frame on the GPU and copies a frame to the
 * canvas only once the GPU has finished it, ahead of the next.
 */
export function createFrameGate(gpu: AnyGpuState, host: GanbateHostRuntime, settled: boolean): FrameGate {
  let inFlight = 0;
  let ready: WebGpuTexture | null = null;
  let shown: WebGpuTexture | null = null;
  let wake = 0;
  let stopped = false;
  const show = () => {
    if (!ready || stopped) return;
    shown = ready;
    ready = null;
    host.present?.(shown);
  };
  return {
    full: () => inFlight >= (settled ? 1 : 2),
    /** Counts a submitted frame until the GPU has finished it; settled, it is then shown. */
    submitted() {
      if (gpu.api !== 'webgpu') return;
      inFlight++;
      const frame = settled ? (host.frameTexture() as WebGpuTexture | null) : null;
      const done = () => {
        inFlight--;
        if (!frame || stopped) return;
        ready = frame;
        // A table at rest draws no next frame to show this one before: an animation frame does.
        if (!wake) {
          wake = requestAnimationFrame(() => {
            wake = 0;
            show();
          });
        }
      };
      gpu.device.queue.onSubmittedWorkDone().then(done, done);
    },
    /** Shows the finished frame (settled), before the next is drawn into its texture. */
    show,
    /** A resized canvas starts empty: the last finished frame, stretched, until the next. */
    reshow() {
      if (!settled || inFlight || stopped) return;
      if (ready) show();
      else if (shown) host.present?.(shown);
    },
    /** Shows nothing more (the engine is destroyed). */
    stop() {
      stopped = true;
    },
  };
}
