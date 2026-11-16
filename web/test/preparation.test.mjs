import test from 'node:test';
import assert from 'node:assert/strict';
import { getEventListeners } from 'node:events';
import { mapBounded, preparationBoundary } from '../sdk/src/preparation.ts';

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

test('preparation boundaries remove every caller and host listener on completion or cancellation', async () => {
  const host = new AbortController();
  const heldWork = new Promise(() => {});
  for (let attempt = 0; attempt < 30; attempt++) {
    const caller = new AbortController();
    const cancelled = assert.rejects(preparationBoundary(heldWork, caller.signal, host.signal, caller.signal), { name: 'AbortError' });
    assert.equal(getEventListeners(caller.signal, 'abort').length, 1, 'duplicate signals share a listener');
    assert.equal(getEventListeners(host.signal, 'abort').length, 1);
    caller.abort(); await cancelled;
    assert.equal(getEventListeners(caller.signal, 'abort').length, 0);
    assert.equal(getEventListeners(host.signal, 'abort').length, 0);
  }
  const caller = new AbortController();
  assert.equal(await preparationBoundary(Promise.resolve(7), caller.signal, host.signal), 7);
  assert.equal(getEventListeners(caller.signal, 'abort').length, 0);
  assert.equal(getEventListeners(host.signal, 'abort').length, 0);
  const failure = new Error('host failed');
  const failed = assert.rejects(preparationBoundary(heldWork, caller.signal, host.signal), (error) => error === failure);
  host.abort(failure); await failed;
  assert.equal(getEventListeners(caller.signal, 'abort').length, 0);
  assert.equal(getEventListeners(host.signal, 'abort').length, 0);
});

test('a pre-aborted boundary still observes rejection of work already handed to it', async () => {
  const controller = new AbortController(); controller.abort();
  await assert.rejects(preparationBoundary(Promise.reject(new Error('late work failed')), controller.signal), { name: 'AbortError' });
  // An unobserved rejection here fails the Node test runner after the test completes.
  await tick();
});
