import type { GanbateWasmExports } from './internal-types.ts';
import type { GanbateGraph, GanbateGraphCommit, GanbateGraphOp, GanbateGraphSolver } from './types.ts';

export interface WasmScratch {
  reserve(len: number): number;
  place(...arrays: (ArrayBufferView | null | undefined)[]): number[];
  free(): void;
}

/**
 * One scratch allocation in the module's memory, reused: the module reads call arguments during
 * the call only. Its start is rounded up to 4 bytes so typed-array views of it are aligned.
 * `reserve(len)` returns room for `len` bytes; `place(...arrays)` copies byte arrays there, each at
 * a 4-byte boundary, and returns their pointers; `free()` gives the memory back.
 */
export function scratchFor(x: Pick<GanbateWasmExports, 'memory' | 'ganbate_alloc' | 'ganbate_free'>): WasmScratch {
  let scratch = 0;
  let scratchCap = 0;
  const reserve = (len: number) => {
    if (len + 4 > scratchCap) {
      if (scratch) x.ganbate_free(scratch, scratchCap);
      scratchCap = Math.max(len + 4, scratchCap * 2, 4096);
      scratch = x.ganbate_alloc(scratchCap) >>> 0;
    }
    return (scratch + 3) & ~3;
  };
  const place = (...arrays: (ArrayBufferView | null | undefined)[]) => {
    let total = 0;
    const offsets = arrays.map((a) => {
      const at = total;
      total += a ? (a.byteLength + 3) & ~3 : 0;
      return at;
    });
    const base = reserve(Math.max(total, 4));
    const mem = new Uint8Array(x.memory.buffer);
    return arrays.map((a, i) => {
      if (!a) return 0;
      mem.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), base + offsets[i]);
      return base + offsets[i];
    });
  };
  const free = () => {
    if (scratch) x.ganbate_free(scratch, scratchCap);
    scratch = 0;
    scratchCap = 0;
  };
  return { reserve, place, free };
}

/**
 * The graph solver's two calls on module exports `x` (see `crates/graph`): `graphCommit`
 * applies a patch, `graphCheck` takes a whole graph. Each returns `{graph, digest, repairs}` or
 * throws with the reason a change was refused.
 */
export function graphCalls(
  x: Pick<GanbateWasmExports, 'memory' | 'ganbate_graph_commit' | 'ganbate_graph_check' | 'ganbate_graph_result'>,
  { reserve, place }: Pick<WasmScratch, 'reserve' | 'place'>,
): GanbateGraphSolver {
  const encoder = new TextEncoder();
  const json = (v: unknown) =>
    v === null ? new Uint8Array(0) : encoder.encode(typeof v === 'string' ? v : JSON.stringify(v));
  const call = (
    name: 'ganbate_graph_commit' | 'ganbate_graph_check',
    label: string,
    first: unknown,
    second: unknown,
    mode: 'project' | 'reject',
  ): GanbateGraphCommit => {
    const [a, b] = [json(first), json(second)];
    const [ap, bp] = place(a, b);
    const len = x[name](ap, a.byteLength, bp, b.byteLength, mode === 'reject' ? 1 : 0);
    const out = reserve(len);
    x.ganbate_graph_result(out, len);
    const result = JSON.parse(new TextDecoder().decode(new Uint8Array(x.memory.buffer, out, len))) as GanbateGraphCommit & {
      readonly error?: string;
    };
    if (result.error) throw new Error(`${label}: ${result.error}`);
    return result;
  };
  return {
    graphCommit: (
      graph: GanbateGraph | string,
      patch: readonly GanbateGraphOp[] | string,
      mode: 'project' | 'reject' = 'project',
    ) => call('ganbate_graph_commit', 'graphCommit', graph, patch, mode),
    graphCheck: (
      graph: GanbateGraph | string,
      previous: GanbateGraph | string | null = null,
      mode: 'project' | 'reject' = 'project',
    ) => call('ganbate_graph_check', 'graphCheck', previous, graph, mode),
  };
}

/** The graph solver without a canvas, for pages and tools that edit a graph without drawing it. */
export async function createGraphSolver({
  wasm,
}: {
  readonly wasm: string | ArrayBuffer | Uint8Array | WebAssembly.Module;
}): Promise<GanbateGraphSolver> {
  const module =
    wasm instanceof WebAssembly.Module
      ? wasm
      : await WebAssembly.compile(
          typeof wasm === 'string' ? await (await fetch(wasm)).arrayBuffer() : (wasm as BufferSource),
        );
  const imports: WebAssembly.Imports = {};
  for (const i of WebAssembly.Module.imports(module)) {
    ((imports[i.module] as Record<string, WebAssembly.ImportValue> | undefined) ??
      (imports[i.module] = {}))[i.name] = () => 0;
  }
  const { exports } = await WebAssembly.instantiate(module, imports);
  const x = exports as unknown as GanbateWasmExports;
  return graphCalls(x, scratchFor(x));
}
