/**
 * Graph sessions: one semantic scene graph (`ganbate.semantic.graph.v2`) shared live by everything
 * that edits or shows it — an editor's panels, a running game's living canvas, a demo page — on
 * this page and on other pages of the same origin. Every change goes through the engine's solver
 * (`engine.graphCommit` for patches, `engine.graphCheck` for a whole graph written as text), so all
 * participants keep the same rules, and a graph's digest names the same state everywhere.
 *
 * Participants talk over a `BroadcastChannel` named `ganbate-graph:<name>` (or any object with
 * `postMessage`, `addEventListener('message')` and `close`, such as an adapter over a frame's
 * `postMessage`). Messages: `{kind: 'graph', graph, digest, source}` when the state changes,
 * `{kind: 'select', id, source}` when the selection does, and `{kind: 'hello'}` from a newcomer,
 * which every participant answers with its state.
 */

export const GRAPH_CHANNEL_PREFIX = 'ganbate-graph:';

/**
 * Opens a session on `graph` (an object or its text), checked by `engine` first.
 * `engine` needs `graphCommit(graph, patch, mode)` and `graphCheck(graph, previous, mode)`.
 * `mode` 'project' repairs what breaks the rules and reports it; 'reject' refuses the change.
 * `channel`: false for a page-local session, an object as above, or omitted for the default.
 */
export function createGraphSession({
  engine,
  graph,
  name = 'default',
  channel,
  mode = 'project',
  historyLimit = 64,
}) {
  const opened = engine.graphCheck(graph, null, mode);
  let current = { graph: opened.graph, digest: opened.digest };
  let selection = null;
  let edited = false;
  const past = [];
  const future = [];
  const listeners = new Set();
  const link =
    channel === false
      ? null
      : (channel ?? (typeof BroadcastChannel === 'function' ? new BroadcastChannel(GRAPH_CHANNEL_PREFIX + name) : null));

  const post = (message) => link?.postMessage(message);
  const emit = (event) => {
    for (const listener of listeners) listener({ ...event, graph: current.graph, digest: current.digest, selection });
  };
  const announce = (source) => post({ kind: 'graph', graph: current.graph, digest: current.digest, source });
  const adopt = (next, source, repairs = [], remote = false) => {
    if (next.digest === current.digest) return false;
    past.push(current);
    if (past.length > historyLimit) past.shift();
    future.length = 0;
    current = { graph: next.graph, digest: next.digest };
    if (selection !== null && !current.graph.nodes.some((n) => n.id === selection)) selection = null;
    emit({ kind: 'graph', source, repairs, remote });
    return true;
  };
  const step = (from, to, source) => {
    const target = from.pop();
    if (!target) return false;
    edited = true;
    to.push(current);
    current = target;
    emit({ kind: 'graph', source, repairs: [], remote: false });
    announce(source);
    return true;
  };

  const onMessage = (event) => {
    const message = event.data;
    if (!message || typeof message !== 'object') return;
    if (message.kind === 'hello') {
      announce('sync');
    } else if (message.kind === 'graph' && message.digest !== current.digest) {
      try {
        // A newcomer takes the state already shared; afterwards every change is adopted.
        if (message.source === 'sync' && edited) return;
        adopt(engine.graphCheck(message.graph, null, 'reject'), message.source ?? 'remote', [], true);
      } catch {
        // A graph the rules refuse is not adopted; the sender keeps its own.
      }
    } else if (message.kind === 'select' && (message.id === null || typeof message.id === 'string')) {
      selection = message.id;
      emit({ kind: 'select', source: message.source ?? 'remote', remote: true });
    }
  };
  link?.addEventListener('message', onMessage);
  post({ kind: 'hello' });

  const change = (result, source) => {
    edited = true;
    if (adopt(result, source, result.repairs)) announce(source);
    return result;
  };

  return {
    get graph() {
      return current.graph;
    },
    get digest() {
      return current.digest;
    },
    get selection() {
      return selection;
    },
    /** Applies graph operations; returns `{graph, digest, repairs}` or throws why they were refused. */
    commit(patch, { source = 'local' } = {}) {
      return change(engine.graphCommit(current.graph, patch, mode), source);
    },
    /** Takes a whole graph written as code; changed relations place nodes, moved nodes relate anew. */
    replace(graph, { source = 'code' } = {}) {
      return change(engine.graphCheck(graph, current.graph, mode), source);
    },
    undo({ source = 'local' } = {}) {
      return step(past, future, source);
    },
    redo({ source = 'local' } = {}) {
      return step(future, past, source);
    },
    /** Selects a node by id (null: nothing) for every participant. */
    select(id, { source = 'local' } = {}) {
      if (id !== null && !current.graph.nodes.some((n) => n.id === id)) return false;
      selection = id;
      emit({ kind: 'select', source, remote: false });
      post({ kind: 'select', id, source });
      return true;
    },
    /** Calls `listener({kind, graph, digest, selection, source, remote, repairs})` on every change; returns its remover. */
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    close() {
      link?.removeEventListener('message', onMessage);
      if (channel === undefined) link?.close();
      listeners.clear();
    },
  };
}
