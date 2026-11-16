/** Await a preparation boundary without retaining aborted listeners. */
export async function preparationBoundary<T>(work: Promise<T>, ...signals: readonly (AbortSignal | undefined)[]): Promise<T> {
  const active = [...new Set(signals.filter((signal): signal is AbortSignal => signal !== undefined))];
  const reason = (signal: AbortSignal) => signal.reason ?? new DOMException('Cancelled', 'AbortError');
  const aborted = active.find((signal) => signal.aborted);
  if (aborted) {
    // The caller already started this work before passing its promise. Observe a late
    // rejection even when cancellation means we will never consume its result.
    void work.catch(() => {});
    throw reason(aborted);
  }
  if (!active.length) return work;
  const listeners: (() => void)[] = [];
  const cancelled = new Promise<never>((_, reject) => {
    for (const signal of active) {
      const abort = () => reject(reason(signal));
      listeners.push(abort);
      signal.addEventListener('abort', abort, { once: true });
    }
  });
  try {
    return await Promise.race([work, cancelled]);
  } finally {
    active.forEach((signal, index) => signal.removeEventListener('abort', listeners[index]));
  }
}

/** Bounds decoded images in flight; consumes each result before decoding another. */
export async function mapBounded<T, R>(
  values: readonly T[],
  run: (value: T, index: number) => Promise<R>,
  concurrency = 4,
): Promise<R[]> {
  const result = new Array<R>(values.length);
  let next = 0;
  let failure: unknown;
  let failed = false;
  await Promise.all(Array.from({ length: Math.min(values.length, Math.max(1, concurrency)) }, async () => {
    for (;;) {
      if (failed) return;
      const index = next++;
      if (index >= values.length) return;
      try { result[index] = await run(values[index], index); }
      catch (error) { failed = true; failure = error; return; }
      // Yield between image uploads; this does not claim to move synchronous WASM work off-thread.
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
  }));
  if (failed) throw failure;
  return result;
}
