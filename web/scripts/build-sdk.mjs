#!/usr/bin/env node
// Builds the modular TypeScript sources under `web/sdk/src/` into flat, standalone ES modules
// and self-contained `.d.ts` / `.d.mts` bundles in `<outDir>` (default `web/dist`):
//   - ganbate.js + ganbate.d.ts
//   - character-customization.mjs + character-customization.d.mts
//   - graph-session.mjs + graph-session.d.mts

import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(__dirname, '..');
const tscBin = path.join(webRoot, 'node_modules', '.bin', process.platform === 'win32' ? 'tsc.cmd' : 'tsc');

function parseOutDir(argv) {
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--outDir' && argv[i + 1]) {
      return path.resolve(argv[i + 1]);
    }
    if (argv[i].startsWith('--outDir=')) {
      return path.resolve(argv[i].slice('--outDir='.length));
    }
  }
  return path.join(webRoot, 'dist');
}

function stripImports(dts) {
  return dts.replace(/^import\s+type\s+\{[\s\S]*?\}\s+from\s+['"][^'"]+['"];\r?\n?/gm, '');
}

function stripDeclareGlobal(dts) {
  return dts.replace(/\r?\ndeclare global \{[\s\S]*?\r?\n\}\r?\n?/g, '\n');
}

function demoteExportsToInternal(dts, keepNames = new Set()) {
  return dts.replace(/^export\s+(interface|type)\s+([A-Za-z0-9_]+)/gm, (full, kind, name) =>
    keepNames.has(name) ? full : `${kind} ${name}`,
  );
}

function normalizeExportFunctions(dts) {
  return dts.replace(/^export declare function /gm, 'export function ');
}

function extractExportedFunctions(dts, allowedNames) {
  const clean = stripImports(dts);
  const lines = clean.split(/\r?\n/);
  const blocks = [];
  let pendingComment = [];
  let current = null;
  let keepCurrent = false;

  const flush = () => {
    if (current && keepCurrent) {
      blocks.push([...pendingComment, ...current].join('\n'));
    }
    pendingComment = [];
    current = null;
    keepCurrent = false;
  };

  for (const line of lines) {
    if (current) {
      current.push(line);
      if ((line.endsWith(';') && !line.startsWith(' ')) || line === '};' || line === '}') {
        flush();
      }
      continue;
    }

    if (line.startsWith('/**') || line.startsWith(' *') || line.startsWith(' */')) {
      pendingComment.push(line);
      continue;
    }

    const fnMatch = /^export declare function ([A-Za-z0-9_]+)/.exec(line);
    if (fnMatch) {
      keepCurrent = allowedNames.has(fnMatch[1]);
      current = [line];
      if (line.endsWith(';')) {
        flush();
      }
      continue;
    }

    const otherDecl = /^export (?:interface|type|declare const) /.exec(line);
    if (otherDecl) {
      keepCurrent = false;
      current = [line];
      if (line.endsWith(';') || line.endsWith('{}')) {
        flush();
      }
      continue;
    }

    if (line.trim() === '') {
      pendingComment = [];
    }
  }
  flush();
  return normalizeExportFunctions(blocks.join('\n'));
}

async function bundleSdkJs(outDir) {
  const entries = [
    ['sdk/src/index.ts', 'ganbate.js'],
    ['sdk/src/character-customization.ts', 'character-customization.mjs'],
    ['sdk/src/graph-session.ts', 'graph-session.mjs'],
  ];

  for (const [entryRel, fileName] of entries) {
    await build({
      configFile: false,
      logLevel: 'error',
      root: webRoot,
      build: {
        outDir,
        emptyOutDir: false,
        target: 'es2022',
        minify: false,
        sourcemap: false,
        lib: {
          entry: path.join(webRoot, entryRel),
          formats: ['es'],
          fileName: () => fileName,
        },
      },
    });
  }
}

function emitSdkDeclarations(outDir) {
  const tmpDir = mkdtempSync(path.join(tmpdir(), 'ganbate-sdk-dts-'));
  try {
    execFileSync(
      tscBin,
      [
        '--ignoreConfig',
        '--target',
        'ES2022',
        '--lib',
        'ES2023,DOM,DOM.Iterable',
        '--module',
        'ESNext',
        '--moduleResolution',
        'bundler',
        '--allowImportingTsExtensions',
        '--strict',
        '--skipLibCheck',
        '--declaration',
        '--emitDeclarationOnly',
        '--outDir',
        tmpDir,
        path.join(webRoot, 'sdk/src/index.ts'),
        path.join(webRoot, 'sdk/src/character-customization.ts'),
        path.join(webRoot, 'sdk/src/graph-session.ts'),
      ],
      { stdio: 'inherit' },
    );

    const indexDts = readFileSync(path.join(tmpDir, 'index.d.ts'), 'utf8');
    const publicFns = new Set();
    for (const match of indexDts.matchAll(/export\s*\{([^}]+)\}/g)) {
      for (const item of match[1].split(',')) {
        const name = item.trim().split(/\s+as\s+/).pop()?.trim();
        if (name) publicFns.add(name);
      }
    }

    const typesDts = stripImports(readFileSync(path.join(tmpDir, 'types.d.ts'), 'utf8')).trim();
    const webgpuInternalDts = demoteExportsToInternal(
      stripDeclareGlobal(stripImports(readFileSync(path.join(tmpDir, 'webgpu-globals.d.ts'), 'utf8'))),
    ).trim();
    const internalTypesFull = stripImports(readFileSync(path.join(tmpDir, 'internal-types.d.ts'), 'utf8'));
    // Drop `GanbateWasmExports` since no public SDK function references it; keep host state types.
    const internalTypesTrimmed = demoteExportsToInternal(
      internalTypesFull.replace(/export interface GanbateWasmExports \{[\s\S]*?\r?\n\}\r?\n?/, ''),
      new Set(['WebGpuState', 'WebGlExtensions', 'WebGlState', 'GanbateHostRuntime']),
    ).trim();

    const fnModules = [
      'engine.d.ts',
      'graph-solver.d.ts',
      'player-runtime.d.ts',
      'texture-source.d.ts',
      'webgl-host.d.ts',
      'webgpu-host.d.ts',
    ];
    const fnBlocks = fnModules
      .map((file) => extractExportedFunctions(readFileSync(path.join(tmpDir, file), 'utf8'), publicFns))
      .filter(Boolean);

    const ganbateDts = [
      '// Generated from web/sdk/src/*.ts by web/scripts/build-sdk.mjs. Do not edit by hand.',
      '',
      typesDts,
      '',
      webgpuInternalDts,
      '',
      internalTypesTrimmed,
      '',
      ...fnBlocks,
      '',
    ].join('\n');
    writeFileSync(path.join(outDir, 'ganbate.d.ts'), ganbateDts, 'utf8');

    const charDts = normalizeExportFunctions(
      stripImports(readFileSync(path.join(tmpDir, 'character-customization.d.ts'), 'utf8')).trim(),
    );
    writeFileSync(
      path.join(outDir, 'character-customization.d.mts'),
      `// Generated from web/sdk/src/character-customization.ts by web/scripts/build-sdk.mjs.\n\n${charDts}\n`,
      'utf8',
    );

    const graphSessionRaw = readFileSync(path.join(tmpDir, 'graph-session.d.ts'), 'utf8')
      .replace(/from ['"]\.\/types\.ts['"]/g, "from './ganbate.js'")
      .trim();
    const graphSessionDmts = normalizeExportFunctions(graphSessionRaw);
    writeFileSync(
      path.join(outDir, 'graph-session.d.mts'),
      `// Generated from web/sdk/src/graph-session.ts by web/scripts/build-sdk.mjs.\n\n${graphSessionDmts}\n`,
      'utf8',
    );

    // Self-verify all emitted declaration files in isolation under strict TypeScript.
    execFileSync(
      tscBin,
      [
        '--ignoreConfig',
        '--target',
        'ES2022',
        '--lib',
        'ES2023,DOM,DOM.Iterable',
        '--module',
        'ESNext',
        '--moduleResolution',
        'bundler',
        '--strict',
        '--noEmit',
        path.join(outDir, 'ganbate.d.ts'),
        path.join(outDir, 'character-customization.d.mts'),
        path.join(outDir, 'graph-session.d.mts'),
      ],
      { stdio: 'inherit' },
    );
  } finally {
    rmSync(tmpDir, { recursive: true, force: true });
  }
}

async function main() {
  const outDir = parseOutDir(process.argv.slice(2));
  mkdirSync(outDir, { recursive: true });
  await bundleSdkJs(outDir);
  emitSdkDeclarations(outDir);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
