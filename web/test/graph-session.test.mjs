import assert from 'node:assert/strict';
import test from 'node:test';
import { createGraphSession, windowChannel } from '../dist/graph-session.mjs';

// A stand-in for the engine's solver: graph-level operations only, the text as the digest.
const engine = {
  graphCheck(graph) {
    const g = typeof graph === 'string' ? JSON.parse(graph) : graph;
    return { graph: g, digest: JSON.stringify(g), repairs: [] };
  },
  graphCommit(graph, patch) {
    const g = structuredClone(graph);
    for (const op of patch) {
      if (op.op !== 'graph') throw new Error('graphCommit: refused');
      g[op.key] = op.value;
    }
    return { graph: g, digest: JSON.stringify(g), repairs: ['noted'] };
  },
};
const ROOM = { schema: 'ganbate.semantic.graph.v2', nodes: [{ id: 'desk', type: 'desk', position: [0, 0, 0] }], edges: [] };
const settle = () => new Promise((resolve) => setTimeout(resolve, 30));

test('participants share every change and the selection; a newcomer takes the shared state', async () => {
  const name = `test-${Math.random()}`;
  const editor = createGraphSession({ engine, graph: ROOM, name });
  const canvas = createGraphSession({ engine, graph: ROOM, name });
  const seen = [];
  canvas.subscribe((event) => seen.push([event.kind, event.source, event.remote]));
  try {
    const night = editor.commit([{ op: 'graph', key: 'environment', value: { preset: 'night' } }]);
    assert.deepEqual(night.repairs, ['noted']);
    await settle();
    assert.equal(canvas.digest, editor.digest);
    assert.deepEqual(canvas.graph.environment, { preset: 'night' });

    editor.select('desk');
    await settle();
    assert.equal(canvas.selection, 'desk');
    assert.equal(editor.select('nothing'), false);

    const demo = createGraphSession({ engine, graph: ROOM, name });
    await settle();
    assert.equal(demo.digest, editor.digest, 'a newcomer adopts the shared state');
    demo.close();

    assert.equal(canvas.undo(), true);
    await settle();
    assert.equal(editor.digest, JSON.stringify(ROOM));
    assert.deepEqual(seen, [
      ['graph', 'local', true],
      ['select', 'local', true],
      ['graph', 'local', false],
    ]);
    assert.equal(canvas.redo(), true);
    assert.equal(canvas.graph.environment.preset, 'night');
  } finally {
    editor.close();
    canvas.close();
  }
});

test('a refused change keeps the state, the history and the other participants unchanged', async () => {
  const session = createGraphSession({ engine, graph: JSON.stringify(ROOM), channel: false });
  assert.throws(() => session.commit([{ op: 'remove', id: 'desk' }]), /refused/);
  assert.equal(session.digest, JSON.stringify(ROOM));
  assert.equal(session.undo(), false);
  const code = session.replace({ ...ROOM, environment: { preset: 'dawn' } });
  assert.equal(code.graph.environment.preset, 'dawn');
  assert.equal(session.undo({ source: 'code' }), true);
  assert.equal(session.digest, JSON.stringify(ROOM));
  session.close();
});

// Two windows of different origins: `win.postMessage` delivers to `win` itself, from the other.
function windowPair(parentOrigin, frameOrigin) {
  const make = (origin) => Object.assign(new EventTarget(), { origin });
  const [parent, frame] = [make(parentOrigin), make(frameOrigin)];
  const deliver = (win, from) => {
    win.postMessage = (data, targetOrigin) => {
      if (targetOrigin !== win.origin) return;
      setTimeout(() => win.dispatchEvent(Object.assign(new Event('message'), { data, origin: from.origin, source: from })), 0);
    };
  };
  deliver(parent, frame);
  deliver(frame, parent);
  return { parent, frame };
}

test('a page joins the graph of the frame it embeds over postMessage, and nobody else is heard', async () => {
  const { parent, frame } = windowPair('https://ganbate.io', 'https://play.ganbate.io');
  // The embedded table opens with its graph; the page opens without one and joins.
  const table = createGraphSession({ engine, graph: ROOM, channel: windowChannel(parent, 'https://ganbate.io', 'yozakura', frame) });
  const page = createGraphSession({ engine, channel: windowChannel(frame, 'https://play.ganbate.io', 'yozakura', parent) });
  assert.equal(page.graph, null);
  assert.throws(() => page.commit([]), /no graph yet/);
  await page.ready;
  assert.equal(page.digest, table.digest);

  page.commit([{ op: 'graph', key: 'environment', value: { preset: 'sunset' } }]);
  await settle();
  assert.equal(table.graph.environment.preset, 'sunset');

  // A third window posing as the page is ignored.
  const stranger = Object.assign(new EventTarget(), { origin: 'https://evil.example' });
  const forged = { ganbateGraph: 'yozakura', message: { kind: 'graph', graph: ROOM, digest: 'x', source: 'remote' } };
  frame.dispatchEvent(Object.assign(new Event('message'), { data: forged, origin: 'https://evil.example', source: stranger }));
  await settle();
  assert.equal(table.graph.environment.preset, 'sunset');
  table.close();
  page.close();
});

test("the game's own rules refuse a graph here and from other participants", async () => {
  const name = `test-${Math.random()}`;
  const validate = (graph) => {
    if (!graph.nodes.some((n) => n.id === 'desk')) throw new Error('the desk stays');
  };
  const strict = createGraphSession({ engine, graph: ROOM, name, validate });
  const loose = createGraphSession({ engine, graph: ROOM, name });
  try {
    assert.throws(() => strict.replace({ ...ROOM, nodes: [] }), /the desk stays/);
    loose.replace({ ...ROOM, nodes: [] });
    await settle();
    assert.equal(strict.graph.nodes.length, 1);
  } finally {
    strict.close();
    loose.close();
  }
});

