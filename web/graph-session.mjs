/**
 * Graph sessions: one semantic scene graph (`ganbate.semantic.graph.v2`) shared live by everything
 * that edits or shows it — an editor's panels, a running game's living canvas, a demo page — on
 * this page and on other pages of the same origin. Every change goes through the engine's solver
 * (`engine.graphCommit` for patches, `engine.graphCheck` for a whole graph written as text), so all
 * participants keep the same rules, and a graph's digest names the same state everywhere.
 *
 * Participants talk over a `BroadcastChannel` named `ganbate-graph:<name>` within one origin, or
 * over `windowChannel` between a page and a frame of another origin. Messages:
 * `{kind: 'graph', graph, digest, source}` when the state changes, `{kind: 'select', id, source}`
 * when the selection does, and `{kind: 'hello'}` from a newcomer, which every participant with a
 * graph answers with its state. A participant that has not changed anything takes the state the
 * others answer with; one opened without a graph joins with the first state it hears.
 */

export const GRAPH_CHANNEL_PREFIX = 'ganbate-graph:';

/**
 * Opens a session on `graph` (an object or its text), checked by `engine` first, or, without one,
 * a session that joins the graph another participant shares (`ready` resolves then).
 * `engine` needs `graphCommit(graph, patch, mode)` and `graphCheck(graph, previous, mode)`, as the
 * engine and `createGraphSolver` provide.
 * `mode` 'project' repairs what breaks the rules and reports it; 'reject' refuses the change.
 * `validate(graph)` adds the game's own rules (a node it needs, a value it allows): it throws to
 * refuse a graph, which then neither becomes this session's nor is taken from another participant.
 * `channel`: false for a page-local session, an object as above, or omitted for the default.
 */
export function createGraphSession({
  engine,
  graph,
  name = 'default',
  channel,
  mode = 'project',
  historyLimit = 64,
  validate = () => {},
}) {
  const solved = (result) => {
    validate(result.graph);
    return result;
  };
  const opened = graph == null ? null : solved(engine.graphCheck(graph, null, mode));
  let current = opened && { graph: opened.graph, digest: opened.digest };
  let joined = () => {};
  const ready = new Promise((resolve) => {
    joined = resolve;
  });
  if (current) joined();
  let selection = null;
  // Settled once it has changed the graph or heard the shared one: from then on, a newcomer's
  // state (`sync`) no longer replaces its own.
  let settled = false;
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
  const announce = (source) => current && post({ kind: 'graph', graph: current.graph, digest: current.digest, source });
  const adopt = (next, source, repairs = [], remote = false) => {
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
  const step = (from, to, source) => {
    const target = from.pop();
    if (!target) return false;
    settled = true;
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
    } else if (message.kind === 'graph') {
      if (message.source === 'sync' && settled) return;
      if (message.digest === current?.digest) {
        settled = true;
        return;
      }
      try {
        adopt(solved(engine.graphCheck(message.graph, null, 'reject')), message.source ?? 'remote', [], true);
        settled = true;
      } catch {
        // A graph the rules refuse is not adopted; the sender keeps its own.
      }
    } else if (current && message.kind === 'select' && (message.id === null || typeof message.id === 'string')) {
      selection = message.id;
      emit({ kind: 'select', source: message.source ?? 'remote', remote: true });
    }
  };
  link?.addEventListener('message', onMessage);
  post({ kind: 'hello' });
  announce('sync');

  const joinedState = () => {
    if (!current) throw new Error('graph session: no graph yet; wait for `ready`');
    return current;
  };
  const change = (result, source) => {
    settled = true;
    if (adopt(result, source, result.repairs)) announce(source);
    return result;
  };

  return {
    /** Resolves once the session has a graph (at once when opened with one). */
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
    /** Applies graph operations; returns `{graph, digest, repairs}` or throws why they were refused. */
    commit(patch, { source = 'local' } = {}) {
      return change(solved(engine.graphCommit(joinedState().graph, patch, mode)), source);
    },
    /** Takes a whole graph written as code; changed relations place nodes, moved nodes relate anew. */
    replace(graph, { source = 'code' } = {}) {
      return change(solved(engine.graphCheck(graph, joinedState().graph, mode)), source);
    },
    undo({ source = 'local' } = {}) {
      return step(past, future, source);
    },
    redo({ source = 'local' } = {}) {
      return step(future, past, source);
    },
    /** Selects a node by id (null: nothing) for every participant. */
    select(id, { source = 'local' } = {}) {
      if (id !== null && !joinedState().graph.nodes.some((n) => n.id === id)) return false;
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

/**
 * A channel over `postMessage` to another window, such as a frame embedded from another origin or
 * the page embedding this one. Only messages from `target` at `origin` are heard, and messages go
 * to that origin only; `name` keeps sessions sharing a window apart. `self` is this window.
 */
export function windowChannel(target, origin, name = 'default', self = globalThis) {
  const wrapped = new Map();
  return {
    postMessage(message) {
      target.postMessage({ ganbateGraph: name, message }, origin);
    },
    addEventListener(type, listener) {
      if (type !== 'message' || wrapped.has(listener)) return;
      const hear = (event) => {
        const data = event.data;
        if (event.source !== target || event.origin !== origin || data?.ganbateGraph !== name) return;
        listener({ data: data.message });
      };
      wrapped.set(listener, hear);
      self.addEventListener('message', hear);
    },
    removeEventListener(type, listener) {
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
