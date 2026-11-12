/**
 * Graph sessions: one semantic scene graph (`ganbate.semantic.graph.v2`) shared live by everything
 * that edits or shows it — an editor's panels, a running game's living canvas, a demo page — on
 * this page and on other pages of the same origin. Every change goes through the engine's solver
 * (`engine.graphCommit` for patches, `engine.graphCheck` for a whole graph written as text), so all
 * participants keep the same rules, and a graph's digest names the same state everywhere.
 */
import type { GanbateGraph, GanbateGraphCommit, GanbateGraphOp, GanbateGraphSolver } from './types.ts';

export const GRAPH_CHANNEL_PREFIX = 'ganbate-graph:';

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

interface GraphStateSnapshot {
  readonly graph: GanbateGraph;
  readonly digest: string;
}

/**
 * Opens a session on `graph` (an object or its text), checked by `engine` first, or, without one,
 * a session that joins the graph another participant shares (`ready` resolves then).
 */
export function createGraphSession({
  engine,
  graph,
  name = 'default',
  channel,
  mode = 'project',
  historyLimit = 64,
  validate = () => {},
}: {
  readonly engine: GanbateGraphSolver;
  /** Omitted: join the graph another participant shares. */
  readonly graph?: GanbateGraph | string | null;
  readonly name?: string;
  readonly channel?: GanbateGraphChannel | false;
  readonly mode?: 'project' | 'reject';
  readonly historyLimit?: number;
  /** The game's own rules: throw to refuse a graph. */
  readonly validate?: (graph: GanbateGraph) => void;
}): GanbateGraphSession {
  const solved = (result: GanbateGraphCommit): GanbateGraphCommit => {
    validate(result.graph);
    return result;
  };
  const opened = graph == null ? null : solved(engine.graphCheck(graph, null, mode));
  let current: GraphStateSnapshot | null = opened && { graph: opened.graph, digest: opened.digest };
  let joined: () => void = () => {};
  const ready = new Promise<void>((resolve) => {
    joined = resolve;
  });
  if (current) joined();
  let selection: string | null = null;
  let settled = false;
  const past: GraphStateSnapshot[] = [];
  const future: GraphStateSnapshot[] = [];
  const listeners = new Set<(event: GanbateGraphEvent) => void>();
  const link: GanbateGraphChannel | null =
    channel === false
      ? null
      : (channel ??
        (typeof BroadcastChannel === 'function'
          ? (new BroadcastChannel(GRAPH_CHANNEL_PREFIX + name) as unknown as GanbateGraphChannel)
          : null));

  const post = (message: unknown) => link?.postMessage(message);
  const emit = (event: {
    readonly kind: 'graph' | 'select';
    readonly source: string;
    readonly remote: boolean;
    readonly repairs?: readonly string[];
  }) => {
    if (!current) return;
    for (const listener of listeners) {
      listener({ ...event, graph: current.graph, digest: current.digest, selection });
    }
  };
  const announce = (source: string) =>
    current && post({ kind: 'graph', graph: current.graph, digest: current.digest, source });
  const adopt = (
    next: GraphStateSnapshot,
    source: string,
    repairs: readonly string[] = [],
    remote = false,
  ): boolean => {
    if (next.digest === current?.digest) return false;
    if (current) {
      past.push(current);
      if (past.length > historyLimit) past.shift();
    }
    future.length = 0;
    current = { graph: next.graph, digest: next.digest };
    joined();
    if (selection !== null && !current.graph.nodes.some((n) => n.id === selection)) selection = null;
    emit({ kind: 'graph', source, repairs, remote });
    return true;
  };
  const step = (from: GraphStateSnapshot[], to: GraphStateSnapshot[], source: string): boolean => {
    const target = from.pop();
    if (!target || !current) return false;
    settled = true;
    to.push(current);
    current = target;
    emit({ kind: 'graph', source, repairs: [], remote: false });
    announce(source);
    return true;
  };

  const onMessage = (event: { readonly data: unknown }) => {
    const message = event.data as Record<string, unknown> | null;
    if (!message || typeof message !== 'object') return;
    if (message.kind === 'hello') {
      announce('sync');
    } else if (message.kind === 'graph') {
      if (message.source === 'sync' && settled) return;
      if (message.digest === current?.digest) {
        settled = true;
        return;
      }
      try {
        adopt(
          solved(engine.graphCheck(message.graph as GanbateGraph | string, null, 'reject')),
          typeof message.source === 'string' ? message.source : 'remote',
          [],
          true,
        );
        settled = true;
      } catch {
        // A graph the rules refuse is not adopted; the sender keeps its own.
      }
    } else if (
      current &&
      message.kind === 'select' &&
      (message.id === null || typeof message.id === 'string')
    ) {
      selection = message.id;
      emit({
        kind: 'select',
        source: typeof message.source === 'string' ? message.source : 'remote',
        remote: true,
      });
    }
  };
  link?.addEventListener('message', onMessage);
  post({ kind: 'hello' });
  announce('sync');

  const joinedState = (): GraphStateSnapshot => {
    if (!current) throw new Error('graph session: no graph yet; wait for `ready`');
    return current;
  };
  const change = (result: GanbateGraphCommit, source: string): GanbateGraphCommit => {
    settled = true;
    if (adopt(result, source, result.repairs)) announce(source);
    return result;
  };

  return {
    ready,
    get graph() {
      return current?.graph ?? null;
    },
    get digest() {
      return current?.digest ?? null;
    },
    get selection() {
      return selection;
    },
    commit(patch: readonly GanbateGraphOp[] | string, { source = 'local' }: { readonly source?: string } = {}) {
      return change(solved(engine.graphCommit(joinedState().graph, patch, mode)), source);
    },
    replace(nextGraph: GanbateGraph | string, { source = 'code' }: { readonly source?: string } = {}) {
      return change(solved(engine.graphCheck(nextGraph, joinedState().graph, mode)), source);
    },
    undo({ source = 'local' }: { readonly source?: string } = {}) {
      return step(past, future, source);
    },
    redo({ source = 'local' }: { readonly source?: string } = {}) {
      return step(future, past, source);
    },
    select(id: string | null, { source = 'local' }: { readonly source?: string } = {}) {
      if (id !== null && !joinedState().graph.nodes.some((n) => n.id === id)) return false;
      selection = id;
      emit({ kind: 'select', source, remote: false });
      post({ kind: 'select', id, source });
      return true;
    },
    subscribe(listener: (event: GanbateGraphEvent) => void) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    close() {
      link?.removeEventListener('message', onMessage);
      if (channel === undefined) link?.close();
      listeners.clear();
    },
  };
}

/** A channel over postMessage to `target` at `origin` (a frame, or the page embedding this one). */
export function windowChannel(
  target: { postMessage(message: unknown, targetOrigin: string): void },
  origin: string,
  name = 'default',
  self: EventTarget = globalThis,
): GanbateGraphChannel {
  const wrapped = new Map<(event: { readonly data: unknown }) => void, EventListener>();
  return {
    postMessage(message: unknown) {
      target.postMessage({ ganbateGraph: name, message }, origin);
    },
    addEventListener(type: 'message', listener: (event: { readonly data: unknown }) => void) {
      if (type !== 'message' || wrapped.has(listener)) return;
      const hear: EventListener = (rawEvent) => {
        const event = rawEvent as MessageEvent<{ ganbateGraph?: string; message?: unknown }>;
        const data = event.data;
        if ((event.source as unknown) !== target || event.origin !== origin || data?.ganbateGraph !== name) {
          return;
        }
        listener({ data: data.message });
      };
      wrapped.set(listener, hear);
      self.addEventListener('message', hear);
    },
    removeEventListener(type: 'message', listener: (event: { readonly data: unknown }) => void) {
      const hear = wrapped.get(listener);
      if (type !== 'message' || !hear) return;
      self.removeEventListener('message', hear);
      wrapped.delete(listener);
    },
    close() {
      for (const hear of wrapped.values()) self.removeEventListener('message', hear);
      wrapped.clear();
    },
  };
}
