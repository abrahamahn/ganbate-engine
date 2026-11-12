import type { GanbateApi } from '../../sdk/ganbate.js';
import type { FixedCanvasSize, ParsedPlayerUrlConfig } from './player-types.ts';

function parseApi(raw: string | null): GanbateApi {
  if (raw === 'webgpu' || raw === 'webgl2') return raw;
  return 'auto';
}

function parseFixedSize(raw: string | null): FixedCanvasSize | null {
  const match = /^(\d+)x(\d+)$/.exec(raw ?? '');
  if (!match) return null;
  return {
    width: Number(match[1]),
    height: Number(match[2]),
  };
}

export function parsePlayerUrlConfig(search: string, isDev: boolean): ParsedPlayerUrlConfig {
  const params = new URLSearchParams(search);
  const coarse = typeof window !== 'undefined' && window.matchMedia('(pointer: coarse)').matches;
  const maxDpr = Number(params.get('dpr')) || (coarse ? 1.5 : 2);
  const scale = Math.min(Math.max(Number(params.get('scale')) || 1, 0.25), 2);
  const fixed = parseFixedSize(params.get('size'));
  const scene = params.get('scene') ?? '';
  const selftest = params.get('selftest') === '1';
  const defaultWasm = isDev ? '/ganbate_web.wasm' : './ganbate_web.wasm';

  return {
    mode: selftest ? 'selftest' : scene ? 'player' : 'launcher',
    scene,
    look: params.get('look') ?? '',
    wasm: params.get('wasm') || defaultWasm,
    api: parseApi(params.get('api')),
    query: search,
    maxDpr,
    scale,
    fixed,
    forceTouch: coarse || params.get('touch') === '1',
  };
}

export function applyFixedCanvasStyle(
  canvas: HTMLCanvasElement,
  fixed: FixedCanvasSize | null,
): void {
  if (fixed) {
    const dpr = window.devicePixelRatio || 1;
    canvas.classList.add('fixed');
    canvas.style.width = `${fixed.width / dpr}px`;
    canvas.style.height = `${fixed.height / dpr}px`;
  } else {
    canvas.classList.remove('fixed');
    canvas.style.width = '';
    canvas.style.height = '';
  }
}

export function createCanvasSizer(
  canvas: HTMLCanvasElement,
  config: Pick<ParsedPlayerUrlConfig, 'fixed' | 'maxDpr' | 'scale'>,
): () => void {
  return () => {
    let w: number;
    let h: number;
    if (config.fixed) {
      w = config.fixed.width;
      h = config.fixed.height;
    } else {
      const dpr = Math.min(window.devicePixelRatio || 1, config.maxDpr) * config.scale;
      w = Math.max(1, Math.round(canvas.clientWidth * dpr));
      h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    }
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
  };
}
