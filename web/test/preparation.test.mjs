import test from 'node:test';
import assert from 'node:assert/strict';
import { mapBounded } from '../sdk/src/preparation.ts';

const tick = () => new Promise((resolve) => setTimeout(resolve, 10));

test('bounded decode jobs cap concurrency and preserve result order', async () => {
  let active = 0;
  let peak = 0;
  const values = Array.from({ length: 13 }, (_, index) => index);
  const results = await mapBounded(values, async (value) => {
    active++;
    peak = Math.max(peak, active);
    await tick();
    active--;
    return value * 2;
  }, 3);
  assert.equal(peak, 3);
  assert.equal(active, 0);
  assert.deepEqual(results, values.map((value) => value * 2));
});

test('failed decode stops new jobs and drains in-flight work before rejecting', async () => {
  const started = [];
  let finishSecond;
  let settled = false;
  const failure = new Error('decode failed');
  const work = mapBounded([0, 1, 2, 3], async (value) => {
    started.push(value);
    if (value === 0) throw failure;
    await new Promise((resolve) => { finishSecond = resolve; });
    return value;
  }, 2).finally(() => { settled = true; });
  // Attach rejection before manually completing another worker.
  const rejection = assert.rejects(work, (error) => error === failure);
  await tick();
  assert.deepEqual(started, [0, 1]);
  assert.equal(settled, false, 'caller must not destroy resources still used by a decoder');
  finishSecond();
  await rejection;
  assert.deepEqual(started, [0, 1]);
});
