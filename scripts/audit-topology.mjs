#!/usr/bin/env node
// Verifies ganbate-engine's directory topology, crate naming (no redundant `ganbate-` folder
// prefixes under `crates/`), zero external Cargo dependencies, `web/` structure, and per-file
// source line-count budgets so monolithic files cannot regress.

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const ALLOWED_TOP_DIRS = new Set(['crates', 'docs', 'examples', 'scripts', 'src', 'tests', 'web']);
const IGNORED_TOP_DIRS = new Set(['.git', '.github', 'target', 'artifacts']);
const RETIRED_TOP_PATHS = ['samples', 'content', 'editor'];

const ALLOWED_TOP_FILES = new Set([
  '.git',
  '.gitignore',
  'AGENTS.md',
  'Cargo.lock',
  'Cargo.toml',
  'README.md',
]);

const EXPECTED_CRATES = [
  'assets',
  'audio',
  'core',
  'graph',
  'jobs',
  'physics',
  'pipeline',
  'scene',
  'seq',
  'shader',
  'vfx',
  'web',
];

const EXPECTED_WEB_DIRS = new Set(['player', 'scripts', 'sdk', 'test']);
const IGNORED_WEB_DIRS = new Set(['dist', 'node_modules']);
const ALLOWED_WEB_FILES = new Set([
  '.gitignore',
  'package-lock.json',
  'package.json',
  'tsconfig.json',
  'vite.config.ts',
]);

const errors = [];

function checkRootTopology() {
  for (const retired of RETIRED_TOP_PATHS) {
    if (existsSync(path.join(REPO_ROOT, retired))) {
      errors.push(`Retired top-level directory must not exist: ${retired}/`);
    }
  }
  for (const entry of readdirSync(REPO_ROOT, { withFileTypes: true })) {
    if (entry.isDirectory() || entry.isSymbolicLink()) {
      if (!ALLOWED_TOP_DIRS.has(entry.name) && !IGNORED_TOP_DIRS.has(entry.name)) {
        errors.push(`Unexpected top-level directory: ${entry.name}/`);
      }
    } else if (entry.isFile()) {
      if (!ALLOWED_TOP_FILES.has(entry.name)) {
        errors.push(`Unexpected root file: ${entry.name}`);
      }
    }
  }
  for (const req of ALLOWED_TOP_DIRS) {
    const abs = path.join(REPO_ROOT, req);
    if (!existsSync(abs) || !statSync(abs).isDirectory()) {
      errors.push(`Missing top-level directory: ${req}/`);
    }
  }
}

function checkCratesTopology() {
  const cratesDir = path.join(REPO_ROOT, 'crates');
  if (!existsSync(cratesDir)) return;

  const actual = readdirSync(cratesDir, { withFileTypes: true })
    .filter((e) => e.isDirectory())
    .map((e) => e.name)
    .sort();

  for (const name of actual) {
    if (name.startsWith('ganbate-')) {
      errors.push(`Crate folder must not carry 'ganbate-' prefix: crates/${name}`);
    }
    if (!EXPECTED_CRATES.includes(name)) {
      errors.push(`Unexpected crate folder: crates/${name}`);
    }
  }
  for (const expected of EXPECTED_CRATES) {
    if (!actual.includes(expected)) {
      errors.push(`Missing crate folder: crates/${expected}`);
    }
  }
}

function checkZeroExternalCargoDeps(manifestRelPath) {
  const full = path.join(REPO_ROOT, manifestRelPath);
  if (!existsSync(full)) {
    errors.push(`Missing Cargo manifest: ${manifestRelPath}`);
    return;
  }
  const text = readFileSync(full, 'utf8');
  let inDeps = false;
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.replace(/#.*$/, '').trim();
    if (!line) continue;
    if (line.startsWith('[') && line.endsWith(']')) {
      inDeps = line.includes('dependencies]');
      continue;
    }
    if (inDeps && line.includes('=') && !line.includes('path')) {
      errors.push(`External Cargo dependency forbidden in ${manifestRelPath}: ${line}`);
    }
  }
}

function checkWebTopology() {
  const webDir = path.join(REPO_ROOT, 'web');
  if (!existsSync(webDir)) return;

  for (const entry of readdirSync(webDir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      if (!EXPECTED_WEB_DIRS.has(entry.name) && !IGNORED_WEB_DIRS.has(entry.name)) {
        errors.push(`Unexpected web/ directory: web/${entry.name}/`);
      }
    } else if (entry.isFile()) {
      if (!ALLOWED_WEB_FILES.has(entry.name)) {
        errors.push(`Unexpected stray file in web/: web/${entry.name}`);
      }
    }
  }
  for (const req of EXPECTED_WEB_DIRS) {
    const abs = path.join(webDir, req);
    if (!existsSync(abs) || !statSync(abs).isDirectory()) {
      errors.push(`Missing web/${req}/ directory`);
    }
  }
  const sdkEntry = path.join(webDir, 'sdk', 'src', 'index.ts');
  if (!existsSync(sdkEntry)) {
    errors.push('Missing modular TypeScript SDK entry: web/sdk/src/index.ts');
  }
  const sdkDir = path.join(webDir, 'sdk');
  if (existsSync(sdkDir)) {
    for (const entry of readdirSync(sdkDir, { withFileTypes: true })) {
      if (entry.isFile() && /\.(m?js|d\.m?ts)$/.test(entry.name)) {
        errors.push(`Legacy unmodularized SDK file must not live in web/sdk/: web/sdk/${entry.name}`);
      }
    }
  }
}

const MAX_PROD_SOURCE_LINES = 1900;
const MAX_TEST_SOURCE_LINES = 2500;
const MAX_SHADER_LINES = 1000;
const SOURCE_EXT_RE = /\.(rs|ts|tsx|wgsl|mjs|cjs)$/;
const SKIP_WALK_DIRS = new Set(['.git', '.github', 'artifacts', 'dist', 'node_modules', 'target']);

function isTestSourcePath(relPosix) {
  return (
    relPosix.startsWith('tests/') ||
    relPosix.startsWith('web/test/') ||
    relPosix.includes('/tests/') ||
    relPosix.endsWith('/tests.rs') ||
    /\.test\.[a-z]+$/.test(relPosix)
  );
}

function checkSourceLineBudgets() {
  let checked = 0;
  function walk(absDir) {
    for (const entry of readdirSync(absDir, { withFileTypes: true })) {
      if (SKIP_WALK_DIRS.has(entry.name)) continue;
      const absPath = path.join(absDir, entry.name);
      if (entry.isDirectory()) {
        walk(absPath);
        continue;
      }
      if (!entry.isFile() || !SOURCE_EXT_RE.test(entry.name)) continue;
      checked += 1;
      const relPosix = path.relative(REPO_ROOT, absPath).split(path.sep).join('/');
      const text = readFileSync(absPath, 'utf8');
      const lines = text.split(/\r?\n/).length - (text.endsWith('\n') ? 1 : 0);
      if (entry.name.endsWith('.wgsl')) {
        if (lines > MAX_SHADER_LINES) {
          errors.push(`WGSL shader exceeds ${MAX_SHADER_LINES}-line ceiling (${lines} lines): ${relPosix}`);
        }
      } else if (isTestSourcePath(relPosix)) {
        if (lines > MAX_TEST_SOURCE_LINES) {
          errors.push(`Test file exceeds ${MAX_TEST_SOURCE_LINES}-line ceiling (${lines} lines): ${relPosix}`);
        }
      } else if (lines > MAX_PROD_SOURCE_LINES) {
        errors.push(`Source file exceeds ${MAX_PROD_SOURCE_LINES}-line ceiling (${lines} lines): ${relPosix}`);
      }
    }
  }
  walk(REPO_ROOT);
  return checked;
}

checkRootTopology();
checkCratesTopology();
checkZeroExternalCargoDeps('Cargo.toml');
for (const crate of EXPECTED_CRATES) {
  checkZeroExternalCargoDeps(`crates/${crate}/Cargo.toml`);
}
checkWebTopology();
const checkedFiles = checkSourceLineBudgets();

if (errors.length > 0) {
  console.error(`ganbate-engine topology audit failed (${errors.length} issue(s)):`);
  for (const err of errors) {
    console.error(`  - ${err}`);
  }
  process.exit(1);
}

console.log(
  `ganbate-engine topology audit passed (7 top-level pillars, ${EXPECTED_CRATES.length} clean crates, 0 external Cargo deps, React+TS web/ layout, ${checkedFiles} source files within LOC budgets).`,
);
