import type {
  GanbateApi,
  GanbatePlayerController,
  GanbateSelfTestResult,
} from '../../sdk/ganbate.js';

export type BootMode = 'player' | 'selftest' | 'launcher';

export interface FixedCanvasSize {
  readonly width: number;
  readonly height: number;
}

export interface ParsedPlayerUrlConfig {
  readonly mode: BootMode;
  readonly scene: string;
  readonly look: string;
  readonly wasm: string;
  readonly api: GanbateApi;
  readonly query: string;
  readonly maxDpr: number;
  readonly scale: number;
  readonly fixed: FixedCanvasSize | null;
  readonly forceTouch: boolean;
}

export interface LaunchRequest {
  readonly mode: 'player' | 'selftest';
  readonly scene: string;
  readonly look: string;
  readonly api: GanbateApi;
  readonly query: string;
  readonly ownedBlobUrls: readonly string[];
}

export type WindowGanbateState =
  | GanbatePlayerController
  | { readonly selftest: GanbateSelfTestResult }
  | { readonly failed: string };

declare global {
  interface Window {
    ganbate?: WindowGanbateState;
  }
}
