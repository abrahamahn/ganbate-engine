import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig, type Plugin } from 'vite';

const webRoot = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(webRoot, '..');
const sdkEntry = resolve(webRoot, 'sdk/ganbate.js');

function ganbateSdkExternalPlugin(): Plugin {
  let isBuild = false;
  return {
    name: 'ganbate-sdk-external',
    configResolved(config) {
      isBuild = config.command === 'build';
    },
    resolveId(source, importer) {
      if (!isBuild || !importer) return null;
      const resolved = source.startsWith('.') ? resolve(dirname(importer), source) : null;
      if (resolved === sdkEntry) {
        return { id: './ganbate.js', external: true };
      }
      return null;
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const pathname = req.url?.split('?')[0];
        if (pathname === '/ganbate_web.wasm') {
          const candidates = [
            resolve(webRoot, 'dist/ganbate_web.wasm'),
            resolve(repoRoot, 'target/wasm32-unknown-unknown/release/ganbate_web.wasm'),
          ];
          const found = candidates.find((p) => existsSync(p));
          if (found) {
            res.statusCode = 200;
            res.setHeader('Content-Type', 'application/wasm');
            res.setHeader('Cache-Control', 'no-store');
            res.end(readFileSync(found));
            return;
          }
        }
        next();
      });
    },
  };
}

export default defineConfig({
  root: resolve(webRoot, 'player'),
  base: './',
  plugins: [ganbateSdkExternalPlugin(), react()],
  server: {
    host: '127.0.0.1',
    port: 5175,
    fs: {
      allow: [webRoot, repoRoot],
    },
  },
  build: {
    outDir: resolve(webRoot, 'dist'),
    emptyOutDir: false,
    target: 'es2022',
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        entryFileNames: 'player.js',
        chunkFileNames: 'chunks/[name]-[hash].js',
        assetFileNames: (assetInfo) => {
          if (assetInfo.names?.some((name) => name.endsWith('.css'))) {
            return 'player.css';
          }
          return 'assets/[name]-[hash][extname]';
        },
      },
    },
  },
});
