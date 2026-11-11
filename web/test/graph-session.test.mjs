import assert from 'node:assert/strict';
import test from 'node:test';
import { createGraphSession } from '../graph-session.mjs';

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
