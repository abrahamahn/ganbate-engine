/** Await a preparation boundary without retaining aborted listeners. */
export async function preparationBoundary<T>(work: Promise<T>, signal?: AbortSignal): Promise<T> {
  signal?.throwIfAborted();
  if (!signal) return work;
  let abort!: () => void;
  const cancelled = new Promise<never>((_, reject) => {
    abort = () => reject(signal.reason ?? new DOMException('Cancelled', 'AbortError'));
    signal.addEventListener('abort', abort, { once: true });
  });
  try {
    return await Promise.race([work, cancelled]);
  } finally {
    signal.removeEventListener('abort', abort);
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
