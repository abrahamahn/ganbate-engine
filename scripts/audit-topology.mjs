#!/usr/bin/env node
// Verifies ganbate-engine's directory topology, crate naming (no redundant `ganbate-` folder
// prefixes under `crates/`), zero external Cargo dependencies, and `web/` structure.

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const ALLOWED_TOP_DIRS = new Set(['crates', 'docs', 'examples', 'scripts', 'src', 'tests', 'web']);
const IGNORED_TOP_DIRS = new Set(['.git', 'target', 'artifacts']);
const RETIRED_TOP_PATHS = ['samples', 'content', 'editor'];

const ALLOWED_TOP_FILES = new Set([
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

const EXPECTED_WEB_DIRS = new Set(['player', 'sdk', 'test']);
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
}

checkRootTopology();
checkCratesTopology();
checkZeroExternalCargoDeps('Cargo.toml');
for (const crate of EXPECTED_CRATES) {
  checkZeroExternalCargoDeps(`crates/${crate}/Cargo.toml`);
}
checkWebTopology();

if (errors.length > 0) {
  console.error(`ganbate-engine topology audit failed (${errors.length} issue(s)):`);
  for (const err of errors) {
    console.error(`  - ${err}`);
  }
  process.exit(1);
}

console.log(
  `ganbate-engine topology audit passed (7 top-level pillars, ${EXPECTED_CRATES.length} clean crates, 0 external Cargo deps, React+TS web/ layout).`,
);
