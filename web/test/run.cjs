// Headless browser test of the Ganbate web player.
//
//   node web/test/run.cjs --dist web/dist --out OUT_DIR [--ganbate-render PATH] [--selftest] [--api webgl2] \
//     --case name=NAME,scene=WORLD.gbew,look=LOOK.json,view=VIEW,time=HOURS,weather=PRESET,size=WxH
//     [--case ...]
//
// --api webgl2 runs everything on the WebGL 2 backend instead of WebGPU.
//
// --engine WORLD.gbew runs web/test/engine.html: library mode (createEngine) with the world plus
// props, checking drawing, texture orientation, projection, picking, hiding, batch moves,
// texture updates, overlays, blending, tints, pick layers and error reporting
// (OUT/engine-first.png, -second.png, -third.png).
//
// --selftest first runs the module's GPU-layer self-test (compute with push constants into a
// storage texture, copies, mip generation, a blit into the canvas, misuse reporting) and checks
// the gradient it presents (OUT/selftest-web.png).
//
// For each case it serves web/dist plus the case's world and Look file (from where they are,
// under /assets/), opens the player with the view, time and weather fixed and the animation
// clock paused at `seconds` (default 12, ganbate-render's default), waits for `frames` presented
// frames (default 3), and saves the canvas pixels (read back from the canvas texture) as
// OUT/NAME-web.png and a page screenshot as OUT/NAME-page.png. With --ganbate-render it first
// renders the same view natively (one sample, realtime quality) to OUT/NAME-native.png and
// compares the two: OUT/NAME-diff.png shows |web - native| x 4. Console messages and WebGPU
// errors are written to OUT/NAME-console.txt; any WebGPU error fails the case, and so does a
// mean absolute difference above `max_mean` (default 3 levels of 255). `gzip=1` serves the
// world gzip-compressed (Content-Length then counts compressed bytes). `interact=1` then drives
// the keyboard, mouse, a two-finger touch drag, an on-screen button and (with `size=auto`, where
// the canvas follows the window) a viewport resize, and checks that each changes the status line or the picture.
//
'use strict';
const fs = require('fs');
const http = require('http');
const path = require('path');
const { execFileSync } = require('child_process');
const zlib = require('zlib');
const { chromium } = require('playwright');

// WebGPU in headless Chromium. Without a GPU it runs on SwiftShader; canvas presentation then
// needs Vulkan compositing (with the default software compositing the GPU process fails to
// allocate the canvas swap-chain images and drops the WebGPU instance after a few frames).
// SwiftShader is slow, so tests wait for frame counts rather than for time.
const CHROMIUM_ARGS = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan',
  '--use-vulkan=swiftshader'];

function parseArgs(argv) {
  const opts = { cases: [], dist: path.join(__dirname, '..', 'dist'), out: 'web-test-out', ganbateRender: null, selftest: false, api: '', engine: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const val = () => argv[++i];
    if (a === '--dist') opts.dist = val();
    else if (a === '--out') opts.out = val();
    else if (a === '--ganbate-render') opts.ganbateRender = val();
    else if (a === '--selftest') opts.selftest = true;
    else if (a === '--api') opts.api = val();
    else if (a === '--engine') opts.engine = val();
    else if (a === '--case') {
      const c = { seconds: '12', frames: '3', size: '960x540', max_mean: '3' };
      for (const kv of val().split(',')) {
        const [k, ...v] = kv.split('=');
        c[k] = v.join('=');
      }
      if (!c.name || !c.scene) throw new Error(`a case needs name= and scene=: ${argv[i]}`);
      opts.cases.push(c);
    } else throw new Error(`unknown argument ${a}`);
  }
  if (!opts.cases.length && !opts.selftest && !opts.engine) throw new Error('no --case given');
  return opts;
}

const TYPES = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript',
  '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json',
  '.png': 'image/png', '.svg': 'image/svg+xml', '.gbew': 'application/octet-stream' };

function serve(dist, assets, gzipped) {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    let file = null;
    if (url === '/test/engine.html') {
      file = path.join(__dirname, 'engine.html');
    } else if (url.startsWith('/assets/')) {
      const name = url.slice('/assets/'.length);
      file = assets.get(name);
      if (file && gzipped.has(name)) {
        // Like a server that compresses: Content-Length is the compressed size.
        const body = zlib.gzipSync(fs.readFileSync(file));
        res.writeHead(200, { 'content-type': 'application/octet-stream', 'content-encoding': 'gzip',
          'content-length': body.length, 'cache-control': 'no-store' });
        res.end(body);
        return;
      }
    } else {
      const rel = url === '/' ? 'index.html' : url.slice(1);
      const p = path.resolve(dist, rel);
      if (p.startsWith(path.resolve(dist))) file = p;
    }
    if (!file || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
      res.writeHead(404);
      res.end('not found');
      return;
    }
    const size = fs.statSync(file).size;
    res.writeHead(200, {
      'content-type': TYPES[path.extname(file)] || 'application/octet-stream',
      'content-length': size,
      'cache-control': 'no-store',
    });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((r) => server.listen(0, '127.0.0.1', () => r(server)));
}

function native(opts, c, out) {
  const [w, h] = c.size.split('x');
  const args = ['--scene', c.scene, '--width', w, '--height', h, '--samples', '1',
    '--realtime-quality', '--seconds', c.seconds, '--out', out];
  if (c.look) args.push('--look', c.look);
  if (c.view) args.push('--view', c.view);
  if (c.time) args.push('--time', c.time);
  if (c.weather) args.push('--weather', c.weather);
  const started = Date.now();
  execFileSync(opts.ganbateRender, args, { stdio: ['ignore', 'inherit', 'inherit'] });
  return (Date.now() - started) / 1000;
}

async function runCase(browser, port, opts, c, assets) {
  const log = [];
  // With size=auto the canvas follows this window (one CSS pixel per device pixel).
  const viewport = c.size === 'auto' ? { width: 640, height: 360 } : { width: 1400, height: 900 };
  const page = await browser.newPage({ viewport, deviceScaleFactor: 1 });
  page.on('console', (m) => log.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => log.push(`[pageerror] ${e.message}`));
  const q = new URLSearchParams({ scene: `/assets/${c.name}.gbew`, dpr: '1', seconds: c.seconds,
    paused: '1' });
  if (c.size !== 'auto') q.set('size', c.size);
  if (opts.api) q.set('api', opts.api);
  if (c.look) q.set('look', `/assets/${c.name}.json`);
  for (const k of ['view', 'time', 'weather', 'quality']) if (c[k]) q.set(k, c[k]);
  const url = `http://localhost:${port}/index.html?${q}`;
  const started = Date.now();
  try {
    return await drive(page, url, started, opts, c, assets, log);
  } finally {
    fs.writeFileSync(path.join(opts.out, `${c.name}-console.txt`), `${log.join('\n')}\n`);
    await page.close();
  }
}

/**
 * Drives the controls like a person: keyboard (Y weather, T time, W forward), a mouse drag, a
 * synthetic two-finger touch drag, the on-screen weather button and a viewport resize. Each
 * step must change the status line or the image, with no errors.
 */
async function interact(page, c) {
  const status = () => page.evaluate(() => document.getElementById('status').textContent);
  // Waits until `count` more frames have been presented (input acts once per frame).
  const frames = async (count) => {
    const n = await page.evaluate(() => window.ganbate.frames());
    await page.waitForFunction(([m, k]) => window.ganbate.frames() >= m + k, [n, count],
      { timeout: 300000, polling: 50 });
  };
  const settle = () => frames(2);
  // Captures the next frame and keeps it for `changed`.
  const image = () => page.evaluate(async () => {
    window.lastInteract = await window.ganbate.capture();
    return { width: window.lastInteract.width, height: window.lastInteract.height };
  });
  // Whether the next frame differs from the last capture: a resize, or a mean absolute
  // difference above 1 level of 255 per channel.
  const changed = () => page.evaluate(async () => {
    const a = window.lastInteract;
    const b = await window.ganbate.capture();
    window.lastInteract = b;
    if (a.width !== b.width || a.height !== b.height) return true;
    let sum = 0;
    for (let i = 0; i < a.rgba.length; i += 4) {
      sum += Math.abs(a.rgba[i] - b.rgba[i]) + Math.abs(a.rgba[i + 1] - b.rgba[i + 1]) + Math.abs(a.rgba[i + 2] - b.rgba[i + 2]);
    }
    return sum / (a.width * a.height * 3) > 1;
  });
  const steps = {};
  await page.focus('canvas');
  const s0 = await status();
  await image();
  await page.keyboard.press('KeyY');
  await settle();
  const s1 = await status();
  steps.weatherKey = { before: s0, after: s1, ok: s1.split(' · ')[1] !== s0.split(' · ')[1] };
  await page.keyboard.down('KeyT');
  await frames(8);
  await page.keyboard.up('KeyT');
  await settle();
  const s2 = await status();
  steps.timeKey = { before: s1.split(' · ')[0], after: s2.split(' · ')[0], ok: s2.split(' · ')[0] !== s1.split(' · ')[0] };
  await image();
  await page.keyboard.down('KeyW');
  await frames(8);
  await page.keyboard.up('KeyW');
  await settle();
  steps.flyForward = { status: await status(), ok: (await status()).includes('free') && await changed() };
  await image();
  const box = await page.locator('canvas').boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 120, box.y + box.height / 2 + 30, { steps: 6 });
  await page.mouse.up();
  await settle();
  steps.mouseLook = { ok: await changed() };
  await image();
  await page.evaluate(() => {
    const cv = document.querySelector('canvas');
    const r = cv.getBoundingClientRect();
    const at = (id, x, y, type) => cv.dispatchEvent(new PointerEvent(type, {
      pointerId: id, pointerType: 'touch', isPrimary: id === 11, clientX: r.left + x, clientY: r.top + y, bubbles: true }));
    at(11, 200, 200, 'pointerdown'); at(12, 260, 200, 'pointerdown');
    for (let k = 1; k <= 5; k++) { at(11, 200, 200 - k * 12, 'pointermove'); at(12, 260, 200 - k * 12, 'pointermove'); }
  });
  await frames(8);
  await page.evaluate(() => {
    const cv = document.querySelector('canvas');
    const r = cv.getBoundingClientRect();
    for (const id of [11, 12]) cv.dispatchEvent(new PointerEvent('pointerup', { pointerId: id, pointerType: 'touch', clientX: r.left + 200, clientY: r.top + 140, bubbles: true }));
  });
  await settle();
  steps.twoFingerMove = { ok: await changed() };
  const s3 = await status();
  await page.evaluate(() => document.body.classList.add('touch'));
  await page.getByRole('button', { name: 'weather' }).click();
  await settle();
  const s4 = await status();
  steps.weatherButton = { before: s3.split(' · ')[1], after: s4.split(' · ')[1], ok: s4.split(' · ')[1] !== s3.split(' · ')[1] };
  if (c.size === 'auto') {
    await page.setViewportSize({ width: 800, height: 450 });
    await settle();
    await settle();
    const s5 = await status();
    steps.resize = { status: s5, ok: s5.endsWith('800x450') };
  }
  const errors = await page.evaluate(() => window.ganbate.errors());
  return { ok: Object.values(steps).every((s) => s.ok) && errors.length === 0, steps, errors };
}

async function drive(page, url, started, opts, c, assets, log) {
  await page.goto(url);
  const timeout = Number(c.timeout || 600) * 1000;
  await page.waitForFunction((n) => window.ganbate && (window.ganbate.failed || window.ganbate.frames() >= n
    || window.ganbate.errors().length > 0), Number(c.frames), { timeout, polling: 250 });
  const state = await page.evaluate(() => ({
    failed: window.ganbate.failed || null,
    frames: window.ganbate.failed ? 0 : window.ganbate.frames(),
    errors: window.ganbate.failed ? [] : window.ganbate.errors(),
    adapter: window.ganbate.adapter || '',
    wasmMemoryMB: window.ganbate.exports ? window.ganbate.exports.memory.buffer.byteLength / 1048576 : 0,
    status: document.getElementById('status').textContent,
    userAgent: navigator.userAgent,
  }));
  const result = { name: c.name, url, seconds: (Date.now() - started) / 1000, ...state };
  if (!state.failed) {
    // Pixels of one more frame, read back from the canvas texture.
    const png = await page.evaluate(async () => {
      const img = await window.ganbate.capture();
      const cv = document.createElement('canvas');
      cv.width = img.width; cv.height = img.height;
      cv.getContext('2d').putImageData(new ImageData(img.rgba, img.width, img.height), 0, 0);
      window.lastCapture = img;
      return cv.toDataURL('image/png').split(',')[1];
    });
    const webPng = path.join(opts.out, `${c.name}-web.png`);
    fs.writeFileSync(webPng, Buffer.from(png, 'base64'));
    result.web = webPng;
    if (c.interact === '1') result.interact = await interact(page, c);
    // Software rendering can take seconds per frame; stop the loop so the page can settle.
    await page.evaluate(() => window.ganbate.stop());
    if (c.native) {
      assets.set(`${c.name}-native.png`, c.native);
      const cmp = await page.evaluate(async (src) => {
        const im = new Image();
        im.src = src;
        await im.decode();
        const a = window.lastCapture;
        if (im.width !== a.width || im.height !== a.height) return { error: `size ${im.width}x${im.height} vs ${a.width}x${a.height}` };
        const cv = document.createElement('canvas');
        cv.width = a.width; cv.height = a.height;
        const g = cv.getContext('2d');
        g.drawImage(im, 0, 0);
        const b = g.getImageData(0, 0, a.width, a.height).data;
        const diff = new Uint8ClampedArray(a.rgba.length);
        let sum = 0; let sq = 0; let over8 = 0; let over32 = 0;
        const n = a.width * a.height;
        for (let i = 0; i < n; i++) {
          let worst = 0;
          for (let k = 0; k < 3; k++) {
            const d = Math.abs(a.rgba[i * 4 + k] - b[i * 4 + k]);
            sum += d; sq += d * d;
            worst = Math.max(worst, d);
            diff[i * 4 + k] = Math.min(255, d * 4);
          }
          diff[i * 4 + 3] = 255;
          if (worst > 8) over8++;
          if (worst > 32) over32++;
        }
        g.putImageData(new ImageData(diff, a.width, a.height), 0, 0);
        const mse = sq / (n * 3);
        return {
          mean: sum / (n * 3),
          psnr: mse > 0 ? 10 * Math.log10(255 * 255 / mse) : Infinity,
          over8: over8 / n,
          over32: over32 / n,
          png: cv.toDataURL('image/png').split(',')[1],
        };
      }, `/assets/${c.name}-native.png`);
      if (cmp.png) {
        fs.writeFileSync(path.join(opts.out, `${c.name}-diff.png`), Buffer.from(cmp.png, 'base64'));
        delete cmp.png;
      }
      result.compare = cmp;
    }
    // The page with its overlays, for people; best effort while the GPU is still busy.
    try {
      await page.screenshot({ path: path.join(opts.out, `${c.name}-page.png`), timeout: 180000 });
    } catch (e) {
      result.screenshot = `skipped: ${e.message.split('\n')[0]}`;
    }
    result.errors = await page.evaluate(() => window.ganbate.errors());
  }
  result.ok = !result.failed && result.frames >= Number(c.frames) && result.errors.length === 0
    && (!result.interact || result.interact.ok)
    && (!result.compare || (!result.compare.error && result.compare.mean <= Number(c.max_mean)));
  return result;
}

/** The GPU-layer self-test: checks the presented gradient (mip 1 of a 64x64 compute image). */
async function runSelfTest(browser, port, opts) {
  const log = [];
  const page = await browser.newPage({ viewport: { width: 600, height: 400 }, deviceScaleFactor: 1 });
  page.on('console', (m) => log.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => log.push(`[pageerror] ${e.message}`));
  const result = { name: 'selftest' };
  try {
    await page.goto(`http://localhost:${port}/index.html?selftest=1&size=256x256&dpr=1${opts.api ? `&api=${opts.api}` : ''}`);
    await page.waitForFunction(() => window.ganbate && (window.ganbate.failed || window.ganbate.selftest),
      null, { timeout: 120000, polling: 100 });
    const r = await page.evaluate(() => {
      if (window.ganbate.failed) return { failed: window.ganbate.failed };
      const { pixels, errors, adapter } = window.ganbate.selftest;
      const px = (x, y) => Array.from(pixels.rgba.slice((y * pixels.width + x) * 4, (y * pixels.width + x) * 4 + 3));
      const cv = document.createElement('canvas');
      cv.width = pixels.width; cv.height = pixels.height;
      cv.getContext('2d').putImageData(new ImageData(pixels.rgba, pixels.width, pixels.height), 0, 0);
      return {
        adapter, errors: errors(), size: [pixels.width, pixels.height],
        corners: { topLeft: px(2, 2), topRight: px(253, 2), bottomLeft: px(2, 253), bottomRight: px(253, 253), centre: px(128, 128) },
        png: cv.toDataURL('image/png').split(',')[1],
      };
    });
    if (r.png) {
      fs.writeFileSync(path.join(opts.out, 'selftest-web.png'), Buffer.from(r.png, 'base64'));
      delete r.png;
    }
    Object.assign(result, r);
    const c = r.corners || {};
    const near = (v, want, tol) => v !== undefined && Math.abs(v - want) <= tol;
    // Red grows to the right, green downwards, blue is 0.5 (128) everywhere.
    result.ok = !r.failed && r.errors.length === 0
      && [c.topLeft, c.topRight, c.bottomLeft, c.bottomRight, c.centre].every((p) => p && near(p[2], 128, 2))
      && near(c.topLeft[0], 4, 8) && near(c.topLeft[1], 4, 8)
      && near(c.topRight[0], 251, 8) && near(c.topRight[1], 4, 8)
      && near(c.bottomLeft[0], 4, 8) && near(c.bottomLeft[1], 251, 8)
      && near(c.bottomRight[0], 251, 8) && near(c.bottomRight[1], 251, 8)
      && near(c.centre[0], 128, 8) && near(c.centre[1], 128, 8);
  } catch (e) {
    result.ok = false;
    result.failed = e.message;
  } finally {
    fs.writeFileSync(path.join(opts.out, 'selftest-console.txt'), `${log.join('\n')}\n`);
    await page.close();
  }
  return result;
}

/** Library mode: web/test/engine.html drives createEngine and reports its checks. */
async function runEngineCheck(browser, port, opts) {
  const log = [];
  const page = await browser.newPage({ viewport: { width: 600, height: 400 }, deviceScaleFactor: 1 });
  page.on('console', (m) => log.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => log.push(`[pageerror] ${e.message}`));
  const result = { name: 'engine' };
  try {
    const query = new URLSearchParams({ world: '/assets/engine.gbew' });
    if (opts.api) query.set('api', opts.api);
    await page.goto(`http://localhost:${port}/test/engine.html?${query}`);
    await page.waitForFunction(() => window.engineCheck, null, { timeout: 300000, polling: 200 });
    const r = await page.evaluate(() => window.engineCheck);
    for (const k of ['first', 'second', 'third', 'fifth', 'sixth', 'seventh']) {
      if (r[k]) {
        fs.writeFileSync(path.join(opts.out, `engine-${k}.png`), Buffer.from(r[k], 'base64'));
        delete r[k];
      }
    }
    log.push(...(r.log || []));
    delete r.log;
    Object.assign(result, r);
  } catch (e) {
    result.ok = false;
    result.failed = e.message;
  } finally {
    fs.writeFileSync(path.join(opts.out, 'engine-console.txt'), `${log.join('\n')}\n`);
    await page.close();
  }
  return result;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  fs.mkdirSync(opts.out, { recursive: true });
  const assets = new Map();
  for (const c of opts.cases) {
    assets.set(`${c.name}.gbew`, path.resolve(c.scene));
    if (c.look) assets.set(`${c.name}.json`, path.resolve(c.look));
    if (opts.ganbateRender && c.native !== '0' && c.size !== 'auto') {
      const out = path.join(opts.out, `${c.name}-native.png`);
      const s = native(opts, c, out);
      console.log(`${c.name}: native reference ${out} (${s.toFixed(1)} s)`);
      c.native = out;
    } else if (c.native && c.native !== '0') {
      c.native = path.resolve(c.native);
    } else {
      delete c.native;
    }
  }
  if (opts.engine) assets.set('engine.gbew', path.resolve(opts.engine));
  const gzipped = new Set(opts.cases.filter((c) => c.gzip === '1').map((c) => `${c.name}.gbew`));
  const server = await serve(path.resolve(opts.dist), assets, gzipped);
  const port = server.address().port;
  const browser = await chromium.launch({ args: CHROMIUM_ARGS });
  console.log(`browser: Chromium ${browser.version()} (headless)`);
  const results = [];
  if (opts.selftest) {
    const r = await runSelfTest(browser, port, opts);
    results.push(r);
    console.log(JSON.stringify(r, null, 2));
  }
  if (opts.engine) {
    const r = await runEngineCheck(browser, port, opts);
    results.push(r);
    console.log(JSON.stringify(r, null, 2));
  }
  for (const c of opts.cases) {
    let r;
    try {
      r = await runCase(browser, port, opts, c, assets);
    } catch (e) {
      r = { name: c.name, ok: false, failed: e.message };
    }
    results.push(r);
    console.log(JSON.stringify(r, null, 2));
  }
  await browser.close();
  server.close();
  fs.writeFileSync(path.join(opts.out, 'results.json'), `${JSON.stringify(results, null, 2)}\n`);
  const failed = results.filter((r) => !r.ok).map((r) => r.name);
  console.log(failed.length ? `FAILED: ${failed.join(', ')}` : `all ${results.length} cases passed`);
  process.exit(failed.length ? 1 : 0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
