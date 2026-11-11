/** One semantic scene graph shared live by an editor, a running game and a demo (graph-session.mjs). */
import type { GanbateGraph, GanbateGraphCommit, GanbateGraphOp, GanbateGraphSolver } from './ganbate.js';

export declare const GRAPH_CHANNEL_PREFIX: 'ganbate-graph:';

/** A channel between participants: a BroadcastChannel, or an adapter over a frame's postMessage. */
export interface GanbateGraphChannel {
  postMessage(message: unknown): void;
  addEventListener(type: 'message', listener: (event: { readonly data: unknown }) => void): void;
  removeEventListener(type: 'message', listener: (event: { readonly data: unknown }) => void): void;
  close(): void;
}

export interface GanbateGraphEvent {
  readonly kind: 'graph' | 'select';
  readonly graph: GanbateGraph;
  readonly digest: string;
  readonly selection: string | null;
  /** Who made the change: 'local', 'code', 'remote', 'sync', or a name a participant passed. */
  readonly source: string;
  /** Whether another participant made it. */
  readonly remote: boolean;
  readonly repairs?: readonly string[];
}

export interface GanbateGraphSession {
  /** Resolves once the session has a graph (at once when opened with one). */
  readonly ready: Promise<void>;
  /** The shared graph; null until a session opened without one has joined. */
  readonly graph: GanbateGraph | null;
  readonly digest: string | null;
  readonly selection: string | null;
  commit(patch: readonly GanbateGraphOp[] | string, options?: { readonly source?: string }): GanbateGraphCommit;
  replace(graph: GanbateGraph | string, options?: { readonly source?: string }): GanbateGraphCommit;
  undo(options?: { readonly source?: string }): boolean;
  redo(options?: { readonly source?: string }): boolean;
  select(id: string | null, options?: { readonly source?: string }): boolean;
  subscribe(listener: (event: GanbateGraphEvent) => void): () => void;
  close(): void;
}

export declare function createGraphSession(options: {
  readonly engine: GanbateGraphSolver;
  /** Omitted: join the graph another participant shares. */
  readonly graph?: GanbateGraph | string | null;
  readonly name?: string;
  readonly channel?: GanbateGraphChannel | false;
  readonly mode?: 'project' | 'reject';
  readonly historyLimit?: number;
}): GanbateGraphSession;

/** A channel over postMessage to `target` at `origin` (a frame, or the page embedding this one). */
export declare function windowChannel(
  target: { postMessage(message: unknown, targetOrigin: string): void },
  origin: string,
  name?: string,
  self?: {
    addEventListener(type: 'message', listener: (event: MessageEvent) => void): void;
    removeEventListener(type: 'message', listener: (event: MessageEvent) => void): void;
  },
): GanbateGraphChannel;
