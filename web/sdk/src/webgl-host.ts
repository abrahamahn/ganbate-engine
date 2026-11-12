import type {
  GanbateHostRuntime,
  PassTimingBatch,
  WebGlExtensions,
  WebGlState,
} from './internal-types.ts';
import { createPassTiming, GPU_IMPORTS } from './pass-timing.ts';
import { sourceOfSize } from './texture-source.ts';
import type { GanbateHostOptions, GanbateXrTarget } from './types.ts';

const GL_SURFACE = 0xffffffff;
const ROW_TEXELS = 1024; // ganbate_shader::glsl::STORAGE_ROW_TEXELS
const ROW_BYTES = ROW_TEXELS * 16;
const BUF_DATA = 1;
const BUF_INDEX = 2;
const BUF_STORAGE = 4;
const BUF_READBACK = 8;

const BLIT_VS = `#version 300 es
out vec2 uv;
void main() {
  uv = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}`;
const blitFs = (array: boolean) => `#version 300 es
precision highp float;
precision highp sampler2DArray;
uniform highp ${array ? 'sampler2DArray' : 'sampler2D'} src;
uniform float layer;
in vec2 uv;
out vec4 color;
void main() { color = ${array ? 'texture(src, vec3(uv, layer))' : 'texture(src, uv)'}; }`;

interface GlFormatInfo {
  readonly internal: number;
  readonly format?: number;
  readonly type?: number;
  readonly size?: number;
  readonly bgra?: boolean;
  readonly channels?: number;
  readonly half?: boolean;
  readonly unorm?: boolean;
  readonly float?: boolean;
  readonly depth?: boolean;
  readonly compressed?: boolean;
}

interface GlTextureRecord {
  readonly tex: WebGLTexture | null;
  readonly target: number;
  readonly f: GlFormatInfo;
  readonly w: number;
  readonly h: number;
  readonly layers: number;
  readonly mips: number;
  base: number;
  count: number;
  readonly uid: number;
}

interface GlBufferRecord {
  readonly size: number;
  readonly index: boolean;
  gl: WebGLBuffer | null;
  shadow: Uint8Array | null;
  tex: WebGLTexture | null;
  host: Uint8Array | null;
}

interface GlFramebufferWithUids extends WebGLFramebuffer {
  uids?: number[];
}

/** Creates a WebGL 2 context on `canvas` (or an offscreen one) with the extensions Ganbate uses. */
export function createGl(canvas?: HTMLCanvasElement): WebGlState {
  const target =
    canvas ||
    (typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(1, 1) : document.createElement('canvas'));
  const gl = target.getContext('webgl2', {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
    powerPreference: 'high-performance',
  }) as WebGL2RenderingContext | null;
  if (!gl) {
    throw new Error('This browser has neither WebGPU nor WebGL 2.');
  }
  const ext: WebGlExtensions = {
    colorFloat: gl.getExtension('EXT_color_buffer_float'),
    floatLinear: gl.getExtension('OES_texture_float_linear'),
    floatBlend: gl.getExtension('EXT_float_blend'),
    aniso: gl.getExtension('EXT_texture_filter_anisotropic') as WebGlExtensions['aniso'],
    provoking: gl.getExtension('WEBGL_provoking_vertex') as WebGlExtensions['provoking'],
    indexed: gl.getExtension('OES_draw_buffers_indexed') as WebGlExtensions['indexed'],
    s3tc:
      gl.getExtension('WEBGL_compressed_texture_s3tc') || gl.getExtension('EXT_texture_compression_s3tc'),
    s3tcSrgb:
      gl.getExtension('WEBGL_compressed_texture_s3tc_srgb') ||
      gl.getExtension('EXT_texture_compression_s3tc_srgb'),
    bptc: gl.getExtension('EXT_texture_compression_bptc'),
    astc: gl.getExtension('WEBGL_compressed_texture_astc'),
  };
  if (ext.provoking) ext.provoking.provokingVertexWEBGL(ext.provoking.FIRST_VERTEX_CONVENTION_WEBGL);
  const debug = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = String(
    debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
  );
  const features = Object.entries(ext)
    .filter(([, v]) => v)
    .map(([k]) => k);
  return {
    api: 'webgl2',
    gl,
    ext,
    canvas,
    renderer,
    adapterName: `WebGL 2 ${renderer}`,
    features,
    format: 'rgba8unorm',
  };
}

/** The import object for the wasm module on WebGL 2 and the host state behind it. */
export function createGlHost(
  gpu: WebGlState,
  { onLog = () => {}, onStatus = () => {}, onError = () => {}, checkErrors = false }: GanbateHostOptions = {},
): GanbateHostRuntime {
  const { gl, ext, canvas } = gpu;
  const objects: any[] = [null];
  const freeList: number[] = [];
  const errors: string[] = [];
  let lastError = '';
  let memory!: WebAssembly.Memory;
  let source: unknown = null;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const report = (message: string) => {
    errors.push(message);
    onError(message);
  };
  if (canvas && canvas.addEventListener) {
    canvas.addEventListener('webglcontextlost', () => report('WebGL 2 context lost'));
  }
  const put = (obj: unknown): number => {
    const h = freeList.length ? freeList.pop()! : objects.length;
    objects[h] = obj;
    return h;
  };
  const u8 = (ptr: number, len: number) => new Uint8Array(memory.buffer, ptr >>> 0, len >>> 0);
  const u32 = (ptr: number, n: number) => new Uint32Array(memory.buffer, ptr >>> 0, n >>> 0);
  const str = (ptr: number, len: number) => decoder.decode(u8(ptr, len));
  const writeStr = (s: string, out: number, cap: number) => {
    const bytes = encoder.encode(s);
    const n = Math.min(bytes.length, cap >>> 0);
    u8(out, n).set(bytes.subarray(0, n));
    return n;
  };
  const guard =
    <A extends number[]>(what: string, f: (...args: A) => number) =>
    (...args: A): number => {
      try {
        return f(...args);
      } catch (e) {
        lastError = `${what}: ${e instanceof Error ? e.message : String(e)}`;
        return 0;
      }
    };
  const glErrors = (what: string) => {
    if (!checkErrors) return;
    for (let e = gl.getError(), n = 0; e !== gl.NO_ERROR && n < 8; e = gl.getError(), n++) {
      report(`WebGL error 0x${e.toString(16)} in ${what}`);
    }
  };

  const FORMATS: (GlFormatInfo | null)[] = [
    null,
    { internal: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE, size: 4 },
    { internal: gl.SRGB8_ALPHA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE, size: 4 },
    { internal: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE, size: 4, bgra: true },
    { internal: gl.SRGB8_ALPHA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE, size: 4, bgra: true },
    { internal: gl.RGBA16F, format: gl.RGBA, type: gl.HALF_FLOAT, size: 8, channels: 4, half: true },
    { internal: gl.RG16F, format: gl.RG, type: gl.HALF_FLOAT, size: 4, channels: 2, half: true },
    { internal: gl.R16F, format: gl.RED, type: gl.HALF_FLOAT, size: 2, channels: 1, half: true },
    { internal: gl.R8, format: gl.RED, type: gl.UNSIGNED_BYTE, size: 1, channels: 1, unorm: true },
    { internal: gl.R32F, format: gl.RED, type: gl.FLOAT, size: 4, channels: 1, float: true },
    { internal: gl.RGBA32F, format: gl.RGBA, type: gl.FLOAT, size: 16, channels: 4, float: true },
    { internal: gl.DEPTH_COMPONENT32F, format: gl.DEPTH_COMPONENT, type: gl.FLOAT, size: 4, depth: true },
    { internal: 0x83f1, compressed: true },
    { internal: 0x8c4f, compressed: true },
    { internal: 0x8e8c, compressed: true },
    { internal: 0x8e8d, compressed: true },
    { internal: 0x93b0, compressed: true },
    { internal: 0x93d0, compressed: true },
  ];
  const COMPARES = [
    0,
    gl.NEVER,
    gl.LESS,
    gl.LEQUAL,
    gl.EQUAL,
    gl.GREATER,
    gl.GEQUAL,
    gl.NOTEQUAL,
    gl.ALWAYS,
  ];
  const WRAP = [gl.REPEAT, gl.CLAMP_TO_EDGE, gl.MIRRORED_REPEAT];
  const MODES = [gl.TRIANGLES, gl.TRIANGLE_STRIP, gl.LINES];

  const scratchUnit = (gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS) as number) - 1;
  let uid = 0;
  const textures = new Set<GlTextureRecord>();
  const bindScratch = (t: GlTextureRecord) => {
    gl.activeTexture(gl.TEXTURE0 + scratchUnit);
    gl.bindTexture(t.target, t.tex);
  };
  const setLevels = (t: GlTextureRecord, base: number, count: number) => {
    if (t.base === base && t.count === count) return;
    gl.texParameteri(t.target, gl.TEXTURE_BASE_LEVEL, base);
    gl.texParameteri(t.target, gl.TEXTURE_MAX_LEVEL, base + count - 1);
    t.base = base;
    t.count = count;
  };
  const makeTexture = (
    w: number,
    h: number,
    layers: number,
    mips: number,
    f: GlFormatInfo,
    array: boolean,
  ): GlTextureRecord => {
    const target = array ? gl.TEXTURE_2D_ARRAY : gl.TEXTURE_2D;
    const t: GlTextureRecord = {
      tex: gl.createTexture(),
      target,
      f,
      w,
      h,
      layers,
      mips,
      base: 0,
      count: mips,
      uid: ++uid,
    };
    bindScratch(t);
    if (array) gl.texStorage3D(target, mips, f.internal, w, h, layers);
    else gl.texStorage2D(target, mips, f.internal, w, h);
    textures.add(t);
    return t;
  };

  let surface: GlTextureRecord | null = null;
  const dropTexture = (t: GlTextureRecord) => {
    for (const [k, f] of fbos) {
      if (f.uids?.includes(t.uid)) {
        gl.deleteFramebuffer(f);
        fbos.delete(k);
      }
    }
    gl.deleteTexture(t.tex);
    textures.delete(t);
  };
  const ensureSurface = (): GlTextureRecord | null => {
    if (!canvas) return null;
    const w = canvas.width;
    const h = canvas.height;
    if (w === 0 || h === 0) return null;
    if (!surface || surface.w !== w || surface.h !== h) {
      if (surface) dropTexture(surface);
      surface = makeTexture(w, h, 1, 1, FORMATS[1]!, false);
    }
    return surface;
  };
  const texture = (h: number): GlTextureRecord =>
    h >>> 0 === GL_SURFACE ? ensureSurface()! : (objects[h >>> 0] as GlTextureRecord);

  const fbos = new Map<string, GlFramebufferWithUids>();
  const attach = (point: number, t: GlTextureRecord, mip: number, layer: number) => {
    if (t.target === gl.TEXTURE_2D_ARRAY) gl.framebufferTextureLayer(gl.FRAMEBUFFER, point, t.tex, mip, layer);
    else gl.framebufferTexture2D(gl.FRAMEBUFFER, point, gl.TEXTURE_2D, t.tex, mip);
  };
  const framebuffer = (
    colors: { t: GlTextureRecord; mip: number; layer: number }[],
    depth: { t: GlTextureRecord; mip: number; layer: number } | null,
  ) => {
    const key = [...colors, depth].map((a) => (a ? `${a.t.uid}.${a.mip}.${a.layer}` : '-')).join('|');
    let f = fbos.get(key);
    if (!f) {
      f = gl.createFramebuffer() as GlFramebufferWithUids;
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      colors.forEach((a, i) => attach(gl.COLOR_ATTACHMENT0 + i, a.t, a.mip, a.layer));
      if (depth) attach(gl.DEPTH_ATTACHMENT, depth.t, depth.mip, depth.layer);
      gl.drawBuffers(colors.length ? colors.map((_, i) => gl.COLOR_ATTACHMENT0 + i) : [gl.NONE]);
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        gl.deleteFramebuffer(f);
        throw new Error(`framebuffer incomplete (0x${status.toString(16)}) for ${key}`);
      }
      f.uids = [...colors, depth].filter((a): a is NonNullable<typeof a> => Boolean(a)).map((a) => a.t.uid);
      fbos.set(key, f);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    }
    return f;
  };
  const readFbo = gl.createFramebuffer();

  const fixed: Record<string | number, unknown> = {
    [gl.CULL_FACE]: undefined,
    [gl.DEPTH_TEST]: undefined,
    [gl.POLYGON_OFFSET_FILL]: undefined,
    [gl.BLEND]: undefined,
    [gl.SCISSOR_TEST]: undefined,
    cullFace: undefined,
    frontFace: undefined,
    depthFunc: undefined,
    depthMask: undefined,
    bias: undefined,
    slope: undefined,
    drawBuffers: undefined,
    blendMode: undefined,
    colorMask: undefined,
  };
  const setFlag = (cap: number, on: boolean) => {
    if (fixed[cap] !== on) {
      if (on) gl.enable(cap);
      else gl.disable(cap);
      fixed[cap] = on;
    }
  };
  const resetFixed = () => {
    fixed[gl.CULL_FACE] = undefined;
    fixed[gl.DEPTH_TEST] = undefined;
    fixed[gl.POLYGON_OFFSET_FILL] = undefined;
    fixed[gl.BLEND] = undefined;
    fixed[gl.SCISSOR_TEST] = undefined;
    fixed.cullFace = undefined;
    fixed.frontFace = undefined;
    fixed.depthFunc = undefined;
    fixed.depthMask = undefined;
    fixed.bias = undefined;
    fixed.slope = undefined;
    fixed.drawBuffers = undefined;
    fixed.blendMode = undefined;
    fixed.colorMask = undefined;
  };
  let current: any = null;
  let currentProgram: WebGLProgram | null = null;
  const blendFunc = (set: (a: number, b: number, c: number, d: number) => void, mode: number) => {
    if (mode === 1) set(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    else if (mode === 2) set(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    else set(gl.ONE, gl.ONE, gl.ONE, gl.ONE);
  };
  const applyPipeline = (p: any) => {
    if (currentProgram !== p.program.program) {
      gl.useProgram(p.program.program);
      currentProgram = p.program.program;
    }
    current = p;
    setFlag(gl.CULL_FACE, p.cull !== 0);
    if (p.cull !== 0 && fixed.cullFace !== p.cull) {
      gl.cullFace(p.cull === 1 ? gl.BACK : gl.FRONT);
      fixed.cullFace = p.cull;
    }
    const front = p.frontCcw ? gl.CW : gl.CCW;
    if (fixed.frontFace !== front) {
      gl.frontFace(front);
      fixed.frontFace = front;
    }
    setFlag(gl.DEPTH_TEST, p.depthTest);
    if (p.depthTest) {
      if (fixed.depthFunc !== p.depthFunc) {
        gl.depthFunc(p.depthFunc);
        fixed.depthFunc = p.depthFunc;
      }
      if (fixed.depthMask !== p.depthWrite) {
        gl.depthMask(p.depthWrite);
        fixed.depthMask = p.depthWrite;
      }
    }
    const offset = p.bias !== 0 || p.slope !== 0;
    setFlag(gl.POLYGON_OFFSET_FILL, offset);
    if (offset && (fixed.bias !== p.bias || fixed.slope !== p.slope)) {
      gl.polygonOffset(p.slope, p.bias);
      fixed.bias = p.bias;
      fixed.slope = p.slope;
    }
    if (p.targets.length && fixed.drawBuffers !== p.drawBuffers) {
      gl.drawBuffers(p.targets.map((t: { mask: number }, i: number) => (t.mask ? gl.COLOR_ATTACHMENT0 + i : gl.NONE)));
      fixed.drawBuffers = p.drawBuffers;
    }
    if (p.indexed) {
      const x = ext.indexed!;
      p.targets.forEach((t: { blend: number; mask: number }, i: number) => {
        if (t.blend) x.enableiOES(gl.BLEND, i);
        else x.disableiOES(gl.BLEND, i);
        if (t.blend) blendFunc((a, b, c, d) => x.blendFuncSeparateiOES(i, a, b, c, d), t.blend);
        x.colorMaskiOES(i, !!(t.mask & 1), !!(t.mask & 2), !!(t.mask & 4), !!(t.mask & 8));
      });
      fixed[gl.BLEND] = undefined;
      fixed.blendMode = undefined;
      fixed.colorMask = undefined;
      return;
    }
    setFlag(gl.BLEND, p.blend !== 0);
    if (p.blend !== 0 && fixed.blendMode !== p.blend) {
      blendFunc((a, b, c, d) => gl.blendFuncSeparate(a, b, c, d), p.blend);
      gl.blendEquation(gl.FUNC_ADD);
      fixed.blendMode = p.blend;
    }
    if (fixed.colorMask !== p.mask) {
      gl.colorMask(!!(p.mask & 1), !!(p.mask & 2), !!(p.mask & 4), !!(p.mask & 8));
      fixed.colorMask = p.mask;
    }
  };

  const nearest = gl.createSampler()!;
  gl.samplerParameteri(nearest, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.samplerParameteri(nearest, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) {
    gl.samplerParameteri(nearest, w, gl.CLAMP_TO_EDGE);
  }
  const linear = gl.createSampler()!;
  gl.samplerParameteri(linear, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.samplerParameteri(linear, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) {
    gl.samplerParameteri(linear, w, gl.CLAMP_TO_EDGE);
  }

  const compile = (type: number, srcText: string) => {
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, srcText);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh) || '';
      gl.deleteShader(sh);
      const lines = srcText.split('\n');
      const where = [...log.matchAll(/ERROR: \d+:(\d+)/g)]
        .slice(0, 3)
        .map((m) => `  ${m[1]}: ${(lines[Number(m[1]) - 1] || '').trim()}`)
        .join('\n');
      throw new Error(
        `${type === gl.VERTEX_SHADER ? 'vertex' : 'fragment'} shader: ${log}${where ? `\n${where}` : ''}`,
      );
    }
    return sh;
  };
  const link = (vs: string, fs: string) => {
    const p = gl.createProgram()!;
    const v = compile(gl.VERTEX_SHADER, vs);
    const f = compile(gl.FRAGMENT_SHADER, fs);
    gl.attachShader(p, v);
    gl.attachShader(p, f);
    gl.linkProgram(p);
    gl.deleteShader(v);
    gl.deleteShader(f);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(p);
      gl.deleteProgram(p);
      throw new Error(`link: ${log}`);
    }
    return p;
  };
  const blitPrograms = [false, true].map((array) => {
    const program = link(BLIT_VS, blitFs(array));
    return {
      program,
      src: gl.getUniformLocation(program, 'src'),
      layer: gl.getUniformLocation(program, 'layer'),
    };
  });
  const emptyVao = gl.createVertexArray();

  const ring = gl.createBuffer();
  let ringSize = 0;
  let currentVao: WebGLVertexArrayObject | null = null;

  const bufferTarget = (b: GlBufferRecord) => (b.index ? gl.ELEMENT_ARRAY_BUFFER : gl.COPY_WRITE_BUFFER);
  const bindForWrite = (b: GlBufferRecord) => {
    if (b.index) {
      gl.bindVertexArray(null);
      currentVao = null;
    }
    gl.bindBuffer(bufferTarget(b), b.gl);
  };
  const uploadRows = (b: GlBufferRecord, offset: number, len: number) => {
    if (!len || !b.shadow) return;
    const first = Math.floor(offset / ROW_BYTES);
    const last = Math.floor((offset + len - 1) / ROW_BYTES);
    gl.activeTexture(gl.TEXTURE0 + scratchUnit);
    gl.bindTexture(gl.TEXTURE_2D, b.tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      first,
      ROW_TEXELS,
      last - first + 1,
      gl.RGBA_INTEGER,
      gl.UNSIGNED_INT,
      new Uint32Array(b.shadow.buffer, first * ROW_BYTES, (last - first + 1) * ROW_TEXELS * 4),
    );
  };
  const bytesOf = (b: GlBufferRecord, offset: number, size: number) => {
    if (b.shadow) return b.shadow.subarray(offset, offset + size);
    if (b.host) return b.host.subarray(offset, offset + size);
    const out = new Uint8Array(size);
    gl.bindBuffer(gl.COPY_READ_BUFFER, b.gl);
    gl.getBufferSubData(gl.COPY_READ_BUFFER, offset, out);
    return out;
  };
  const writeBytes = (b: GlBufferRecord, offset: number, data: Uint8Array) => {
    if (b.gl) {
      bindForWrite(b);
      gl.bufferSubData(bufferTarget(b), offset, data);
    }
    if (b.shadow) {
      b.shadow.set(data, offset);
      uploadRows(b, offset, data.length);
    }
    if (b.host) b.host.set(data, offset);
  };

  const f32 = new Float32Array(1);
  const i32 = new Uint32Array(f32.buffer);
  const toHalf = (v: number) => {
    f32[0] = v;
    const x = i32[0];
    const sign = (x >>> 16) & 0x8000;
    const e = (x >>> 23) & 0xff;
    const m = x & 0x7fffff;
    if (e === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
    const he = e - 127 + 15;
    if (he >= 0x1f) return sign | 0x7c00;
    if (he <= 0) {
      if (he < -10) return sign;
      const mm = (m | 0x800000) >> (1 - he);
      return sign | ((mm + 0x1000) >> 13);
    }
    return sign | (he << 10) | ((m + 0x1000) >> 13);
  };

  const timing = createPassTiming();
  let channel: MessageChannel | null = null;
  const waiting: PassTimingBatch[] = [];
  let passLabel = '';
  let passSize = '';
  let lastPipeline = '';
  let passDrew = false;
  const glMark = (label: string) => {
    const m = timing.mark(label);
    m.sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    m.sent = performance.now();
  };
  const poll = () => {
    while (waiting.length) {
      const frame = waiting[0];
      const m = frame.marks.find((x) => !x.done);
      if (!m) {
        waiting.shift();
        frame.finish();
        continue;
      }
      if (gl.clientWaitSync(m.sync!, 0, 0) === gl.TIMEOUT_EXPIRED) {
        channel!.port2.postMessage(0);
        return;
      }
      m.done = performance.now();
      gl.deleteSync(m.sync!);
    }
  };

  const blit = (
    src: number,
    srcMip: number,
    srcLayer: number,
    dst: number,
    dstMip: number,
    dstLayer: number,
    filterLinear: number,
  ) => {
    const s = texture(src);
    const d = texture(dst);
    const w = Math.max(1, d.w >> dstMip);
    const h = Math.max(1, d.h >> dstMip);
    framebuffer([{ t: d, mip: dstMip, layer: dstLayer }], null);
    const bp = blitPrograms[s.target === gl.TEXTURE_2D_ARRAY ? 1 : 0];
    gl.useProgram(bp.program);
    currentProgram = bp.program;
    gl.uniform1i(bp.src, 0);
    gl.uniform1f(bp.layer, srcLayer);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(s.target, s.tex);
    setLevels(s, srcMip, 1);
    gl.bindSampler(0, filterLinear ? linear : nearest);
    gl.bindVertexArray(emptyVao);
    currentVao = emptyVao;
    resetFixed();
    for (const cap of [gl.DEPTH_TEST, gl.BLEND, gl.CULL_FACE, gl.SCISSOR_TEST, gl.POLYGON_OFFSET_FILL]) {
      gl.disable(cap);
    }
    gl.colorMask(true, true, true, true);
    gl.viewport(0, 0, w, h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    current = null;
  };

  const execute = (cmdPtr: number, words: number, pushPtr: number, pushLen: number) => {
    const c = u32(cmdPtr, words);
    const f = new Float32Array(memory.buffer, cmdPtr >>> 0, words >>> 0);
    const uPushLen = pushLen >>> 0;
    if (uPushLen) {
      gl.bindBuffer(gl.UNIFORM_BUFFER, ring);
      if (uPushLen > ringSize) {
        ringSize = Math.max(uPushLen, 65536);
        gl.bufferData(gl.UNIFORM_BUFFER, ringSize, gl.DYNAMIC_DRAW);
      }
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, u8(pushPtr, uPushLen));
    }
    let discard: number[] = [];
    let i = 0;
    while (i < c.length) {
      const op = c[i++];
      switch (op) {
        case 1: {
          // BEGIN_PASS
          const colors = c[i++];
          const hasDepth = c[i++];
          const cs: {
            t: GlTextureRecord;
            mip: number;
            layer: number;
            load: number;
            store: number;
            clear: number[];
          }[] = [];
          for (let k = 0; k < colors; k++) {
            cs.push({
              t: texture(c[i]),
              mip: c[i + 1],
              layer: c[i + 2],
              load: c[i + 3],
              store: c[i + 4],
              clear: [f[i + 5], f[i + 6], f[i + 7], f[i + 8]],
            });
            i += 9;
          }
          let depth: {
            t: GlTextureRecord;
            mip: number;
            layer: number;
            load: number;
            store: number;
            clear: number;
          } | null = null;
          if (hasDepth) {
            depth = {
              t: texture(c[i]),
              mip: c[i + 1],
              layer: c[i + 2],
              load: c[i + 3],
              store: c[i + 4],
              clear: f[i + 5],
            };
            i += 6;
          }
          framebuffer(cs, depth);
          gl.drawBuffers(cs.length ? cs.map((_, k) => gl.COLOR_ATTACHMENT0 + k) : [gl.NONE]);
          fixed.drawBuffers = cs.map(() => 1).join('');
          const a = cs[0] || depth!;
          gl.viewport(0, 0, Math.max(1, a.t.w >> a.mip), Math.max(1, a.t.h >> a.mip));
          passLabel = '';
          passDrew = false;
          passSize = `${Math.max(1, a.t.w >> a.mip)}x${Math.max(1, a.t.h >> a.mip)}`;
          setFlag(gl.SCISSOR_TEST, false);
          const dontCare: number[] = [];
          cs.forEach((x, k) => {
            if (x.load === 1) {
              if (fixed.colorMask !== 15) {
                gl.colorMask(true, true, true, true);
                fixed.colorMask = 15;
              }
              gl.clearBufferfv(gl.COLOR, k, x.clear);
            } else if (x.load === 2) dontCare.push(gl.COLOR_ATTACHMENT0 + k);
          });
          if (depth) {
            if (depth.load === 1) {
              if (fixed.depthMask !== true) {
                gl.depthMask(true);
                fixed.depthMask = true;
              }
              gl.clearBufferfv(gl.DEPTH, 0, [depth.clear]);
            } else if (depth.load === 2) dontCare.push(gl.DEPTH_ATTACHMENT);
          }
          if (dontCare.length) gl.invalidateFramebuffer(gl.FRAMEBUFFER, dontCare);
          discard = cs.map((x, k) => (x.store ? null : gl.COLOR_ATTACHMENT0 + k)).filter((x): x is number => x !== null);
          if (depth && !depth.store) discard.push(gl.DEPTH_ATTACHMENT);
          break;
        }
        case 2:
          // END_PASS
          if (discard.length) gl.invalidateFramebuffer(gl.FRAMEBUFFER, discard);
          discard = [];
          if (timing.on()) glMark(`${passLabel || (passDrew ? lastPipeline : 'clear')} ${passSize}`);
          break;
        case 3: {
          const p = objects[c[i++]];
          applyPipeline(p);
          lastPipeline = p.program.name;
          if (!passLabel) passLabel = lastPipeline;
          break;
        }
        case 4:
          gl.viewport(Math.round(f[i]), Math.round(f[i + 1]), Math.round(f[i + 2]), Math.round(f[i + 3]));
          i += 4;
          break;
        case 5:
          setFlag(gl.SCISSOR_TEST, true);
          gl.scissor(c[i], c[i + 1], c[i + 2], c[i + 3]);
          i += 4;
          break;
        case 6:
          gl.bindBufferRange(gl.UNIFORM_BUFFER, c[i], (objects[c[i + 1]] as GlBufferRecord).gl, c[i + 2], c[i + 3]);
          i += 4;
          break;
        case 7:
          gl.bindBufferRange(gl.UNIFORM_BUFFER, 0, ring, c[i], c[i + 1]);
          i += 2;
          break;
        case 8: {
          // BIND_UNIT
          const t = texture(c[i + 1]);
          gl.activeTexture(gl.TEXTURE0 + c[i]);
          gl.bindTexture(t.target, t.tex);
          setLevels(t, c[i + 3], c[i + 4]);
          gl.bindSampler(c[i], c[i + 2] ? objects[c[i + 2]] : nearest);
          i += 5;
          break;
        }
        case 9:
          // BIND_STORAGE
          gl.activeTexture(gl.TEXTURE0 + c[i]);
          gl.bindTexture(gl.TEXTURE_2D, (objects[c[i + 1]] as GlBufferRecord).tex);
          gl.bindSampler(c[i], nearest);
          i += 2;
          break;
        case 10: {
          // SET_HIDDEN
          const h = current.program.hidden[c[i]];
          if (h && h.loc) {
            if (h.uint) gl.uniform1ui(h.loc, c[i + 1]);
            else gl.uniform1i(h.loc, c[i + 1] | 0);
          }
          i += 2;
          break;
        }
        case 11: {
          const v = objects[c[i++]].vao as WebGLVertexArrayObject;
          if (currentVao !== v) {
            gl.bindVertexArray(v);
            currentVao = v;
          }
          break;
        }
        case 12:
          gl.drawArraysInstanced(current.mode, c[i + 2], c[i], c[i + 1]);
          i += 3;
          passDrew = true;
          break;
        case 13:
          gl.drawElementsInstanced(current.mode, c[i], gl.UNSIGNED_INT, c[i + 2] * 4, c[i + 1]);
          i += 3;
          passDrew = true;
          break;
        case 14: {
          // COPY_BUFFER
          const s = objects[c[i]] as GlBufferRecord;
          const d = objects[c[i + 2]] as GlBufferRecord;
          const so = c[i + 1];
          const dO = c[i + 3];
          const size = c[i + 4];
          if (s.gl && d.gl && !d.shadow && !d.host) {
            gl.bindBuffer(gl.COPY_READ_BUFFER, s.gl);
            gl.bindBuffer(gl.COPY_WRITE_BUFFER, d.gl);
            gl.copyBufferSubData(gl.COPY_READ_BUFFER, gl.COPY_WRITE_BUFFER, so, dO, size);
          } else {
            writeBytes(d, dO, bytesOf(s, so, size).slice());
          }
          i += 5;
          break;
        }
        case 15: {
          // COPY_TEXTURE
          const t = texture(c[i]);
          const mip = c[i + 1];
          const layer = c[i + 2];
          const w = c[i + 3];
          const h = c[i + 4];
          const b = objects[c[i + 5]] as GlBufferRecord;
          const off = c[i + 6];
          gl.bindFramebuffer(gl.READ_FRAMEBUFFER, readFbo);
          if (t.target === gl.TEXTURE_2D_ARRAY) {
            gl.framebufferTextureLayer(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, t.tex, mip, layer);
          } else {
            gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.tex, mip);
          }
          gl.readBuffer(gl.COLOR_ATTACHMENT0);
          gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
          const fm = t.f;
          let out: Uint8Array;
          if (fm.type === gl.UNSIGNED_BYTE && !fm.unorm) {
            out = new Uint8Array(w * h * 4);
            gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, out);
            if (fm.bgra) {
              for (let k = 0; k < out.length; k += 4) {
                const r = out[k];
                out[k] = out[k + 2];
                out[k + 2] = r;
              }
            }
          } else if (fm.unorm) {
            const rgba = new Uint8Array(w * h * 4);
            gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
            out = new Uint8Array(w * h);
            for (let k = 0; k < w * h; k++) out[k] = rgba[k * 4];
          } else {
            const rgba = new Float32Array(w * h * 4);
            gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, rgba);
            const n = fm.channels!;
            if (fm.half) {
              const halfs = new Uint16Array(w * h * n);
              for (let k = 0; k < w * h; k++) {
                for (let ch = 0; ch < n; ch++) halfs[k * n + ch] = toHalf(rgba[k * 4 + ch]);
              }
              out = new Uint8Array(halfs.buffer);
            } else {
              const fl = new Float32Array(w * h * n);
              for (let k = 0; k < w * h; k++) {
                for (let ch = 0; ch < n; ch++) fl[k * n + ch] = rgba[k * 4 + ch];
              }
              out = new Uint8Array(fl.buffer);
            }
          }
          b.host!.set(out, off);
          gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
          i += 7;
          break;
        }
        case 16:
          // BLIT
          blit(c[i], c[i + 1], c[i + 2], c[i + 3], c[i + 4], c[i + 5], c[i + 6]);
          i += 7;
          if (timing.on()) glMark('blit');
          break;
        case 17: {
          // COPY_DEPTH
          const from = framebuffer([], { t: texture(c[i]), mip: c[i + 1], layer: c[i + 2] });
          framebuffer([], { t: texture(c[i + 3]), mip: c[i + 4], layer: c[i + 5] });
          gl.bindFramebuffer(gl.READ_FRAMEBUFFER, from);
          setFlag(gl.SCISSOR_TEST, false);
          gl.blitFramebuffer(
            0,
            0,
            c[i + 6],
            c[i + 7],
            0,
            0,
            c[i + 6],
            c[i + 7],
            gl.DEPTH_BUFFER_BIT,
            gl.NEAREST,
          );
          i += 8;
          if (timing.on()) glMark('copy depth');
          break;
        }
        default:
          throw new Error(`unknown command ${op} at word ${i - 1}`);
      }
    }
  };

  let xrTarget: GanbateXrTarget | null = null;
  const setXrTarget = (target: GanbateXrTarget | null): boolean => {
    if (target == null) {
      xrTarget = null;
      return true;
    }
    if (
      typeof target !== 'object' ||
      !Number.isFinite(target.x) ||
      !Number.isFinite(target.y) ||
      !Number.isFinite(target.width) ||
      !Number.isFinite(target.height) ||
      target.width <= 0 ||
      target.height <= 0
    ) {
      return false;
    }
    xrTarget = {
      framebuffer: target.framebuffer ?? null,
      x: Math.round(target.x),
      y: Math.round(target.y),
      width: Math.round(target.width),
      height: Math.round(target.height),
    };
    return true;
  };

  const present = () => {
    const s = ensureSurface();
    if (!s) return;
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, readFbo);
    gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, s.tex, 0);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    setFlag(gl.SCISSOR_TEST, false);
    gl.disable(gl.SCISSOR_TEST);
    if (xrTarget) {
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, xrTarget.framebuffer);
      const x0 = xrTarget.x;
      const y0 = xrTarget.y;
      const x1 = x0 + xrTarget.width;
      const y1 = y0 + xrTarget.height;
      const filter = s.w === xrTarget.width && s.h === xrTarget.height ? gl.NEAREST : gl.LINEAR;
      gl.blitFramebuffer(0, 0, s.w, s.h, x0, y1, x1, y0, gl.COLOR_BUFFER_BIT, filter);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    } else {
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
      gl.blitFramebuffer(0, 0, s.w, s.h, 0, s.h, s.w, 0, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    }
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    if (timing.on()) glMark(`present ${s.w}x${s.h}`);
  };

  const imports: Record<string, WebAssembly.ImportValue> = {
    gpu_api: () => 2,
    gl_info(out: number, cap: number) {
      const P = (p: number) => gl.getParameter(p) as number;
      const features =
        (ext.colorFloat ? 1 : 0) |
        (ext.floatLinear ? 2 : 0) |
        (ext.floatBlend ? 4 : 0) |
        (ext.indexed ? 8 : 0) |
        (ext.s3tc && ext.s3tcSrgb ? 16 : 0) |
        (ext.bptc ? 32 : 0) |
        (ext.astc ? 64 : 0);
      const aniso = ext.aniso ? P(ext.aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT) : 1;
      f32[0] = aniso;
      const v = [
        P(gl.MAX_TEXTURE_SIZE),
        P(gl.MAX_ARRAY_TEXTURE_LAYERS),
        P(gl.MAX_UNIFORM_BLOCK_SIZE),
        P(gl.UNIFORM_BUFFER_OFFSET_ALIGNMENT),
        scratchUnit,
        P(gl.MAX_TEXTURE_IMAGE_UNITS),
        P(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS),
        P(gl.MAX_DRAW_BUFFERS),
        P(gl.MAX_VERTEX_ATTRIBS),
        P(gl.MAX_UNIFORM_BUFFER_BINDINGS),
        features,
        i32[0],
        0,
        0,
        0,
        0,
      ];
      const n = Math.min(v.length, cap >>> 0);
      u32(out, n).set(v.slice(0, n));
      return n;
    },
    gl_renderer_name: (out: number, cap: number) => writeStr(gpu.adapterName, out, cap),
    gl_last_error: (out: number, cap: number) => writeStr(lastError, out, cap),
    gl_error_count: () => errors.length,
    gl_error: (i: number, out: number, cap: number) => writeStr(errors[i >>> 0] || '', out, cap),
    gl_canvas(out: number) {
      if (!canvas) return 0;
      ensureSurface();
      u32(out, 2).set([canvas.width, canvas.height]);
      return 1;
    },
    gl_create_buffer: guard('createBuffer', (size: number, kinds: number) => {
      const b: GlBufferRecord = {
        size,
        index: !!(kinds & BUF_INDEX),
        gl: null,
        shadow: null,
        tex: null,
        host: null,
      };
      if (kinds & (BUF_DATA | BUF_INDEX)) {
        b.gl = gl.createBuffer();
        bindForWrite(b);
        gl.bufferData(bufferTarget(b), size, gl.DYNAMIC_DRAW);
      }
      if (kinds & BUF_STORAGE) {
        const rows = Math.max(1, Math.ceil(size / ROW_BYTES));
        b.shadow = new Uint8Array(rows * ROW_BYTES);
        b.tex = gl.createTexture();
        gl.activeTexture(gl.TEXTURE0 + scratchUnit);
        gl.bindTexture(gl.TEXTURE_2D, b.tex);
        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32UI, ROW_TEXELS, rows);
        uploadRows(b, 0, rows * ROW_BYTES);
      }
      if (kinds & BUF_READBACK) b.host = new Uint8Array(size);
      return put(b);
    }),
    gl_write_buffer(h: number, offset: number, ptr: number, len: number) {
      writeBytes(objects[h >>> 0] as GlBufferRecord, offset, u8(ptr, len));
    },
    gl_read_buffer: guard('readBuffer', (h: number, out: number, len: number) => {
      const b = objects[h >>> 0] as GlBufferRecord;
      u8(out, len).set(b.host!.subarray(0, len));
      return 1;
    }),
    gl_create_texture: guard(
      'createTexture',
      (w: number, h: number, layers: number, mips: number, format: number, array: number) =>
        put(makeTexture(w, h, layers, mips, FORMATS[format]!, !!array)),
    ),
    gl_write_texture(h: number, mip: number, layer: number, w: number, hgt: number, ptr: number, len: number) {
      const t = objects[h >>> 0] as GlTextureRecord;
      const fm = t.f;
      bindScratch(t);
      if (fm.compressed) {
        const view = u8(ptr, len);
        if (t.target === gl.TEXTURE_2D_ARRAY) {
          gl.compressedTexSubImage3D(t.target, mip, 0, 0, layer, w, hgt, 1, fm.internal, view);
        } else {
          gl.compressedTexSubImage2D(t.target, mip, 0, 0, w, hgt, fm.internal, view);
        }
        return;
      }
      const rawBytes = fm.bgra || fm.type !== gl.UNSIGNED_BYTE ? u8(ptr, len).slice() : u8(ptr, len);
      if (fm.bgra) {
        for (let k = 0; k < rawBytes.length; k += 4) {
          const r = rawBytes[k];
          rawBytes[k] = rawBytes[k + 2];
          rawBytes[k + 2] = r;
        }
      }
      let data: ArrayBufferView = rawBytes;
      if (fm.type === gl.HALF_FLOAT) data = new Uint16Array(rawBytes.buffer, rawBytes.byteOffset, rawBytes.byteLength >> 1);
      else if (fm.type === gl.FLOAT) data = new Float32Array(rawBytes.buffer, rawBytes.byteOffset, rawBytes.byteLength >> 2);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      if (t.target === gl.TEXTURE_2D_ARRAY) {
        gl.texSubImage3D(t.target, mip, 0, 0, layer, w, hgt, 1, fm.format!, fm.type!, data);
      } else {
        gl.texSubImage2D(t.target, mip, 0, 0, w, hgt, fm.format!, fm.type!, data);
      }
    },
    gl_write_texture_source: guard(
      'texSubImage2D',
      (h: number, mip: number, layer: number, w: number, hgt: number) => {
        const image = sourceOfSize(source, w, hgt) as TexImageSource;
        const t = objects[h >>> 0] as GlTextureRecord;
        bindScratch(t);
        const unpack: [number, number | boolean][] = [
          [gl.UNPACK_FLIP_Y_WEBGL, false],
          [gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false],
          [gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.BROWSER_DEFAULT_WEBGL],
        ];
        const saved = unpack.map(([p]) => gl.getParameter(p));
        for (const [p, v] of unpack) gl.pixelStorei(p, v);
        try {
          if (t.target === gl.TEXTURE_2D_ARRAY) {
            gl.texSubImage3D(t.target, mip, 0, 0, layer, w, hgt, 1, t.f.format!, t.f.type!, image);
          } else {
            gl.texSubImage2D(t.target, mip, 0, 0, w, hgt, t.f.format!, t.f.type!, image);
          }
        } finally {
          unpack.forEach(([p], i) => gl.pixelStorei(p, saved[i]));
        }
        return 1;
      },
    ),
    gl_generate_mips(h: number) {
      const t = objects[h >>> 0] as GlTextureRecord;
      bindScratch(t);
      setLevels(t, 0, t.mips);
      gl.generateMipmap(t.target);
    },
    gl_create_sampler: guard('createSampler', (ptr: number, words: number) => {
      const d = u32(ptr, words);
      const lod = new Float32Array(memory.buffer, (ptr >>> 0) + 20, 1)[0];
      const s = gl.createSampler()!;
      const lin = d[0] !== 0;
      gl.samplerParameteri(s, gl.TEXTURE_MAG_FILTER, lin ? gl.LINEAR : gl.NEAREST);
      gl.samplerParameteri(
        s,
        gl.TEXTURE_MIN_FILTER,
        lin
          ? d[1]
            ? gl.LINEAR_MIPMAP_LINEAR
            : gl.LINEAR_MIPMAP_NEAREST
          : d[1]
            ? gl.NEAREST_MIPMAP_LINEAR
            : gl.NEAREST_MIPMAP_NEAREST,
      );
      for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) {
        gl.samplerParameteri(s, w, WRAP[d[2]]);
      }
      gl.samplerParameterf(s, gl.TEXTURE_MAX_LOD, lod);
      if (d[4]) {
        gl.samplerParameteri(s, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
        gl.samplerParameteri(s, gl.TEXTURE_COMPARE_FUNC, COMPARES[d[4]]);
      }
      if (ext.aniso && d[3] > 1) gl.samplerParameterf(s, ext.aniso.TEXTURE_MAX_ANISOTROPY_EXT, d[3]);
      return put(s);
    }),
    gl_create_program: guard(
      'createProgram',
      (vp: number, vl: number, fp: number, fl: number, mp: number, ml: number) => {
        const program = link(str(vp, vl), str(fp, fl));
        const entry = (srcText: string) => /from '([^']+)'/.exec(srcText)?.[1];
        const name = entry(str(fp, fl)) || entry(str(vp, vl)) || 'program';
        const hidden: { loc: WebGLUniformLocation | null; uint: boolean }[] = [];
        gl.useProgram(program);
        for (const line of str(mp, ml).split('\n')) {
          const [kind, uName, a, b] = line.split(' ');
          if (kind === 'b') {
            const index = gl.getUniformBlockIndex(program, uName);
            if (index !== gl.INVALID_INDEX) gl.uniformBlockBinding(program, index, Number(a));
          } else if (kind === 'u') {
            const loc = gl.getUniformLocation(program, uName);
            if (loc) gl.uniform1i(loc, Number(a));
          } else if (kind === 'h') {
            hidden[Number(a)] = { loc: gl.getUniformLocation(program, uName), uint: b === 'uint' };
          }
        }
        gl.useProgram(currentProgram);
        return put({ program, hidden, name });
      },
    ),
    gl_create_pipeline: guard('createPipeline', (program: number, ptr: number, words: number) => {
      const w = u32(ptr, words);
      const fl = new Float32Array(w.slice().buffer);
      const targets: { blend: number; mask: number }[] = [];
      let k = 9;
      for (let t = 0; t < w[8]; t++, k += 2) targets.push({ blend: w[k], mask: w[k + 1] });
      const stride = w[k];
      const count = w[k + 1];
      const attrs: [number, number, number][] = [];
      for (let a = 0; a < count; a++) attrs.push([w[k + 2 + a * 3], w[k + 3 + a * 3], w[k + 4 + a * 3]]);
      const written = targets.filter((t) => t.mask !== 0);
      const uniform = written.every((t) => t.blend === written[0].blend && t.mask === written[0].mask);
      return put({
        program: objects[program],
        mode: MODES[w[0]],
        cull: w[1],
        frontCcw: w[2] !== 0,
        depthTest: w[3] !== 0,
        depthFunc: COMPARES[w[4]],
        depthWrite: w[5] !== 0,
        bias: fl[6],
        slope: fl[7],
        targets,
        drawBuffers: targets.map((t) => (t.mask ? 1 : 0)).join(''),
        blend: written.length ? written[0].blend : 0,
        mask: written.length ? written[0].mask : 15,
        indexed: !uniform,
        stride,
        attrs,
      });
    }),
    gl_create_vao: guard('createVertexArray', (pipeline: number, buffer: number, offset: number, index: number) => {
      const p = objects[pipeline];
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      if (buffer) {
        gl.bindBuffer(gl.ARRAY_BUFFER, (objects[buffer] as GlBufferRecord).gl);
        for (const [loc, fmt, off] of p.attrs as [number, number, number][]) {
          gl.enableVertexAttribArray(loc);
          if (fmt <= 4) gl.vertexAttribPointer(loc, fmt, gl.FLOAT, false, p.stride, offset + off);
          else if (fmt === 5) gl.vertexAttribIPointer(loc, 1, gl.UNSIGNED_INT, p.stride, offset + off);
          else gl.vertexAttribPointer(loc, 4, gl.UNSIGNED_BYTE, true, p.stride, offset + off);
        }
      }
      if (index) gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, (objects[index] as GlBufferRecord).gl);
      gl.bindVertexArray(null);
      currentVao = null;
      return put({ vao });
    }),
    gl_submit: guard(
      'submit',
      (cmdPtr: number, words: number, pushPtr: number, pushLen: number, isPresent: number) => {
        execute(cmdPtr, words, pushPtr, pushLen);
        if (isPresent) present();
        glErrors(isPresent ? 'present' : 'submit');
        return 1;
      },
    ),
    gl_release(h: number) {
      const uh = h >>> 0;
      const o = objects[uh];
      if (!o) return;
      if (o.tex && o.target) dropTexture(o);
      else if (o.vao) gl.deleteVertexArray(o.vao);
      else if (o.program && o.hidden) gl.deleteProgram(o.program);
      else if (o.mode !== undefined) {
        /* a pipeline: its program is released on its own */
      } else if ('shadow' in o) {
        if (o.gl) gl.deleteBuffer(o.gl);
        if (o.tex) gl.deleteTexture(o.tex);
      } else if (typeof WebGLSampler !== 'undefined' && o instanceof WebGLSampler) {
        gl.deleteSampler(o);
      }
      objects[uh] = null;
      freeList.push(uh);
    },
    log: (level: number, ptr: number, len: number) => onLog(level, str(ptr, len)),
    status: (ptr: number, len: number) => onStatus(str(ptr, len)),
  };
  const unavailable = () => {
    throw new Error('WebGPU call on the WebGL 2 host');
  };
  for (const name of GPU_IMPORTS) imports[name] = unavailable;

  return {
    imports: { ganbate: imports },
    errors,
    attach(instance: WebAssembly.Instance) {
      memory = instance.exports.memory as WebAssembly.Memory;
    },
    setSource(image: unknown) {
      source = image;
    },
    nextFrame() {},
    timing,
    endFrame() {
      const frame = timing.take();
      if (!frame) return;
      if (!channel && typeof MessageChannel === 'function') {
        channel = new MessageChannel();
        channel.port1.onmessage = poll;
      }
      if (!channel) {
        for (const m of frame.marks) gl.deleteSync(m.sync!);
        timing.set(false);
        return;
      }
      waiting.push(frame);
      if (waiting.length === 1) channel.port2.postMessage(0);
    },
    frameTexture() {
      return surface;
    },
    bundleStats: () => ({ cached: 0, candidates: 0, replays: 0 }),
    setXrTarget,
    readSurface() {
      const s = surface;
      if (!s) return null;
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, readFbo);
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, s.tex, 0);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
      const rgba = new Uint8ClampedArray(s.w * s.h * 4);
      gl.readPixels(0, 0, s.w, s.h, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
      for (let k = 3; k < rgba.length; k += 4) rgba[k] = 255;
      return { width: s.w, height: s.h, rgba };
    },
  };
}
