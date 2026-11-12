import assert from 'node:assert/strict';
import test from 'node:test';
import { bindCharacterShapes } from '../dist/character-customization.mjs';
test('named, signed controls commit all matching nodes once and preserve unrelated pose channels', () => {
  const edits = [], updates = [];
  const engine = { modelMorph: (...args) => edits.push(args), updateModel: id => updates.push(id) };
  const info = { nodes: [{mesh: 0}, {mesh: null}, {mesh: 1}], meshes: [
    { targetNames: ['width','smile','wind'], targets: 3 },
    { targetNames: ['smile','width'], targets: 2 },
  ] };
  const pose = bindCharacterShapes(engine, 27, info, { width: [-1,1], smile: [0,1] });
  assert.deepEqual(pose.available, ['width','smile']);
  assert.equal(pose.apply({ width: -1.5, smile: .75, wind: .1 }), true);
  assert.deepEqual(edits, [[27,0,0,-1],[27,2,1,-1],[27,0,1,.75],[27,2,0,.75]]);
  assert.deepEqual(updates, [27]);
  assert.equal(pose.apply({width: -1}), false);
  assert.throws(() => pose.apply({width: 0, smile: NaN}), TypeError);
  assert.equal(edits.length, 4);
  assert.equal(pose.apply({width: 0}), true);
  assert.equal(pose.apply({width: 0}, true), true);
  assert.deepEqual(updates, [27,27,27]);
  const second = bindCharacterShapes(engine, 28, info, {width: [-1,1]});
  second.apply({width: -.5});
  assert.deepEqual(edits.at(-1), [28,2,1,-.5]);
});
