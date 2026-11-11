/** One semantic scene graph shared live by an editor, a running game and a demo (graph-session.mjs). */
import type { GanbateGraph, GanbateGraphCommit, GanbateGraphOp } from './ganbate.js';

export declare const GRAPH_CHANNEL_PREFIX: 'ganbate-graph:';

/** What a session needs of the engine: its solver for patches and for whole graphs. */
export interface GanbateGraphSolver {
  graphCommit(graph: GanbateGraph | string, patch: readonly GanbateGraphOp[] | string, mode?: 'project' | 'reject'): GanbateGraphCommit;
  graphCheck(graph: GanbateGraph | string, previous: GanbateGraph | string | null, mode?: 'project' | 'reject'): GanbateGraphCommit;
}

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
  readonly graph: GanbateGraph;
  readonly digest: string;
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
  readonly graph: GanbateGraph | string;
  readonly name?: string;
  readonly channel?: GanbateGraphChannel | false;
  readonly mode?: 'project' | 'reject';
  readonly historyLimit?: number;
}): GanbateGraphSession;
