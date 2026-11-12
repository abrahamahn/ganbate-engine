// Full-screen passes: ambient occlusion, air (fog, outlines, occlusion), bloom, tone mapping and
// grading (Stylized ACES, AgX, ACES Fitted, or baked 3D LUT), the painterly filter and final output.

#import common

struct Screen {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
};

@vertex fn vs_fullscreen(@builtin(vertex_index) i: u32) -> Screen {
  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: Screen;
  o.clip = vec4f(uv * 2.0 - vec2f(1.0), 0.0, 1.0);
  o.uv = uv;
  return o;
}

struct PostPush {
  a: vec4f,
  b: vec4f,
  c: vec4f,
  d: vec4f,
};
var<push_constant> pc: PostPush;

@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var aux_tex: texture_2d<f32>;
@group(0) @binding(3) var src_tex: texture_2d<f32>;
@group(0) @binding(4) var extra_tex: texture_2d<f32>;
@group(0) @binding(5) var lin_sampler: sampler;
@group(0) @binding(6) var lut_tex: texture_2d<f32>;

fn world_at(uv: vec2f, depth: f32) -> vec3f {
  let ndc = vec2f(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0);
  let p = frame.inv_view_proj * vec4f(ndc, max(depth, 1e-7), 1.0);
  return p.xyz / p.w;
}

fn depth_px(p: vec2i) -> f32 {
  let size = vec2i(frame.viewport.xy);
  return textureLoad(depth_tex, clamp(p, vec2i(0), size - vec2i(1)), 0);
}

fn linear_depth(d: f32) -> f32 {
  return frame.cascades.w / max(d, 1e-7);
}

struct Restored {
  @location(0) color: vec4f,
  @location(1) aux: vec4f,
};

// The cached world (`Renderer::world_cache`) back into the frame's colour and aux, pixel for pixel
// (its depth is copied), before the props, water, sky and weather are drawn over it.
@fragment fn fs_restore(in: Screen) -> Restored {
  let p = vec2i(in.clip.xy);
  var o: Restored;
  o.color = textureLoad(src_tex, p, 0);
  o.aux = textureLoad(aux_tex, p, 0);
  return o;
}

// Empties a region of the cached picture (`Renderer::world_cache`): nothing drawn, at the far
// plane, before what lies there is drawn again.
struct Cleared {
  @location(0) color: vec4f,
  @location(1) aux: vec4f,
  @location(2) shadow: vec4f,
  @builtin(frag_depth) depth: f32,
};

@fragment fn fs_clear() -> Cleared {
  var o: Cleared;
  o.color = vec4f(0.0);
  o.aux = vec4f(0.0);
  o.shadow = vec4f(1.0, 1.0, 1.0, 0.0);
  o.depth = 0.0;
  return o;
}

const KERNEL = array<vec3f, 16>(
  vec3f(0.5381, 0.1856, 0.4319), vec3f(0.1379, 0.2486, 0.4430), vec3f(0.3371, 0.5679, 0.0057),
  vec3f(-0.6999, -0.0451, 0.0019), vec3f(0.0689, -0.1598, 0.8547), vec3f(0.0560, 0.0069, 0.1843),
  vec3f(-0.0146, 0.1402, 0.0762), vec3f(0.0100, -0.1924, 0.0344), vec3f(-0.3577, -0.5301, 0.4358),
  vec3f(-0.3169, 0.1063, 0.0158), vec3f(0.0103, -0.5869, 0.0046), vec3f(-0.0897, -0.4940, 0.3287),
  vec3f(0.7119, -0.0154, 0.0918), vec3f(-0.0533, 0.0596, 0.5411), vec3f(0.0352, -0.0631, 0.5460),
  vec3f(-0.4776, 0.2847, 0.0271)
);

// pc.a = (radius metres, samples, ao width, ao height)
@fragment fn fs_ssao(in: Screen) -> @location(0) vec4f {
  let full = frame.viewport.xy;
  let px = vec2i(in.uv * full);
  let d = depth_px(px);
  let aux = textureLoad(aux_tex, clamp(px, vec2i(0), vec2i(full) - vec2i(1)), 0);
  if (d <= 0.0 || aux.w <= 0.0) {
    return vec4f(1.0);
  }
  let p = world_at(in.uv, d);
  let n = normalize(aux.xyz);
  let radius = pc.a.x * clamp(linear_depth(d) * 0.04, 0.6, 3.0);
  let spin = rand01(u32(in.clip.x) * 2749u + u32(in.clip.y) * 6151u + u32(frame.wind.w) * 15485863u) * 6.2831;
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.9) {
    up = vec3f(1.0, 0.0, 0.0);
  }
  let t0 = normalize(cross(up, n));
  let b0 = cross(n, t0);
  let t = t0 * cos(spin) + b0 * sin(spin);
  let b = cross(n, t);
  let count = i32(pc.a.y);
  var occlusion = 0.0;
  for (var i = 0; i < count; i++) {
    let k = KERNEL[i % 16];
    let s = p + (t * k.x + b * k.y + n * (k.z + 0.08)) * radius;
    let c = frame.view_proj * vec4f(s, 1.0);
    if (c.w <= 0.0) {
      continue;
    }
    let sdepth = c.z / c.w;
    let suv = c.xy / c.w * 0.5 + vec2f(0.5);
    let scene = depth_px(vec2i(suv * full));
    if (scene > sdepth * 1.0002) {
      let range = smoothstep(0.0, 1.0, radius / max(abs(linear_depth(scene) - linear_depth(sdepth)), 1e-3));
      occlusion += range;
    }
  }
  let ao = 1.0 - occlusion / f32(max(count, 1));
  return vec4f(ao, ao, ao, 1.0);
}

// Light shafts: march from the pixel towards the sun (or moon) on screen and gather the open
// sky seen past the scenery, so light streams through gaps in roofs, trees and railings.
fn light_shafts(uv: vec2f, frag: vec2f) -> vec3f {
  let strength = frame.paint.z;
  let g = frame.glow_dir.xyz;
  if (strength <= 0.0 || g.y < -0.08) {
    return vec3f(0.0);
  }
  let sp = frame.view_proj * vec4f(frame.camera.xyz + g * 1000.0, 1.0);
  if (sp.w <= 0.0) {
    return vec3f(0.0);
  }
  let suv = sp.xy / sp.w * 0.5 + vec2f(0.5);
  let delta = suv - uv;
  let aspect = frame.viewport.x * frame.viewport.w;
  let span = length(vec2f(delta.x * aspect, delta.y));
  let falloff = exp(-span * 2.2);
  if (falloff < 0.02) {
    return vec3f(0.0);
  }
  let jitter = rand01(u32(frag.x) * 7919u + u32(frag.y) * 104729u + u32(frame.wind.w) * 15485863u);
  var sum = 0.0;
  var weight = 1.0;
  var total = 0.0;
  for (var i = 0; i < 20; i++) {
    let f = (f32(i) + jitter) / 20.0;
    let q = uv + delta * f;
    if (all(q > vec2f(0.0)) && all(q < vec2f(1.0))) {
      if (depth_px(vec2i(q * frame.viewport.xy)) <= 0.0) {
        sum += weight;
      }
    }
    total += weight;
    weight *= 0.93;
  }
  // Strongest when the light is low (dawn, sunset, moonrise) and in mist.
  let low = smoothstep(0.75, 0.05, g.y);
  let haze = 0.6 + 0.8 * frame.fog2.z;
  return frame.glow.rgb * (sum / total * falloff * low * haze * strength);
}

// Air: occlusion, outlines, aerial perspective, mist and light shafts.
// pc.a = (ao strength, outline strength, outline width px, unused), pc.b = ao tint rgb,
// pc.c = outline colour rgb
@fragment fn fs_air(in: Screen) -> @location(0) vec4f {
  let px = vec2i(in.clip.xy);
  var col = textureLoad(src_tex, px, 0).rgb;
  let d = depth_px(px);
  let cam = frame.camera.xyz;
  let world = world_at(in.uv, d);
  let to = world - cam;
  let dist = length(to);
  let dir = to / max(dist, 1e-4);
  let shafts = light_shafts(in.uv, in.clip.xy);
  if (d <= 0.0) {
    // Sky: a mist veil near the horizon.
    let mist = frame.fog2.z;
    col = mix(col, fog_color(dir), mist * 0.5 * smoothstep(0.3, 0.0, dir.y));
    return vec4f(col + shafts * 0.5, 1.0);
  }
  let aux = textureLoad(aux_tex, px, 0);
  var ao = textureSampleLevel(extra_tex, lin_sampler, in.uv, 0.0).r;
  if (aux.w >= 0.99) {
    ao = min(ao, clamp(aux.w - 1.0, 0.0, 1.0));
  }
  let occl = mix(pc.b.rgb, vec3f(1.0), ao);
  col *= mix(vec3f(1.0), occl, pc.a.x);

  // Soft illustrated outlines on depth and normal breaks, fading with distance.
  let w = i32(max(pc.a.z, 1.0));
  let lin = linear_depth(d);
  var edge = 0.0;
  let offs = array<vec2i, 4>(vec2i(w, 0), vec2i(-w, 0), vec2i(0, w), vec2i(0, -w));
  for (var i = 0; i < 4; i++) {
    let q = px + offs[i];
    let dq = depth_px(q);
    let lq = select(linear_depth(dq), 1e6, dq <= 0.0);
    let rel = (lq - lin) / lin;
    edge = max(edge, smoothstep(0.04, 0.15, rel));
    let nq = textureLoad(aux_tex, clamp(q, vec2i(0), vec2i(frame.viewport.xy) - vec2i(1)), 0);
    if (nq.w > 0.0 && aux.w > 0.6) {
      edge = max(edge, smoothstep(0.75, 0.35, dot(normalize(nq.xyz), normalize(aux.xyz))) * 0.6);
    }
  }
  // Grass and flowers (aux.w = 0) get no outlines, and leaves and petals (aux.w = 0.5) only a
  // trace: a line round every leaf cluster of a canopy speckles it.
  let weight = select(1.0, 0.15, aux.w < 0.6) * step(0.01, aux.w);
  edge *= pc.a.y * smoothstep(90.0, 15.0, dist) * weight;
  col = mix(col, pc.c.rgb * (0.4 + 0.6 * luminance(col)), edge);

  // Aerial perspective, then low mist that pools over water and low ground.
  let fog = fog_amount(cam, world, dist);
  col = mix(col, fog_color(dir), fog);
  let mist = frame.fog2.z;
  if (mist > 0.0) {
    let low = exp(-max(world.y - frame.water.x - 0.3, 0.0) * 0.35);
    let m = mist * (1.0 - exp(-dist * 0.03)) * mix(0.35, 1.0, low);
    col = mix(col, fog_color(dir) * 1.05, clamp(m, 0.0, 0.92));
  }
  return vec4f(col + shafts, 1.0);
}

// Bloom. pc.a = (source texel w, source texel h, threshold, first pass)
@fragment fn fs_bloom_down(in: Screen) -> @location(0) vec4f {
  let t = pc.a.xy;
  let a = textureSampleLevel(src_tex, lin_sampler, in.uv + vec2f(-t.x, -t.y), 0.0).rgb;
  let b = textureSampleLevel(src_tex, lin_sampler, in.uv + vec2f(t.x, -t.y), 0.0).rgb;
  let c = textureSampleLevel(src_tex, lin_sampler, in.uv + vec2f(-t.x, t.y), 0.0).rgb;
  let d = textureSampleLevel(src_tex, lin_sampler, in.uv + vec2f(t.x, t.y), 0.0).rgb;
  if (pc.a.w > 0.5) {
    // Karis average (tames single-pixel sparkles) and a soft threshold.
    let wa = 1.0 / (1.0 + luminance(a));
    let wb = 1.0 / (1.0 + luminance(b));
    let wc = 1.0 / (1.0 + luminance(c));
    let wd = 1.0 / (1.0 + luminance(d));
    var col = (a * wa + b * wb + c * wc + d * wd) / (wa + wb + wc + wd);
    let br = max(col.r, max(col.g, col.b));
    let knee = pc.a.z * 0.5;
    let soft = clamp(br - pc.a.z + knee, 0.0, 2.0 * knee);
    let contrib = max(soft * soft / (4.0 * knee + 1e-4), br - pc.a.z) / max(br, 1e-4);
    col *= contrib;
    return vec4f(col, 1.0);
  }
  return vec4f((a + b + c + d) * 0.25, 1.0);
}

// pc.a = (source texel w, source texel h, radius, unused)
@fragment fn fs_bloom_up(in: Screen) -> @location(0) vec4f {
  let t = pc.a.xy * pc.a.z;
  var col = textureSampleLevel(src_tex, lin_sampler, in.uv, 0.0).rgb * 4.0;
  col += textureSampleLevel(src_tex, lin_sampler, in.uv + vec2f(t.x, 0.0), 0.0).rgb * 2.0;
  col += textureSampleLevel(src_tex, lin_sampler, in.uv - vec2f(t.x, 0.0), 0.0).rgb * 2.0;
  col += textureSampleLevel(src_tex, lin_sampler, in.uv + vec2f(0.0, t.y), 0.0).rgb * 2.0;
  col += textureSampleLevel(src_tex, lin_sampler, in.uv - vec2f(0.0, t.y), 0.0).rgb * 2.0;
  col += textureSampleLevel(src_tex, lin_sampler, in.uv + t, 0.0).rgb;
  col += textureSampleLevel(src_tex, lin_sampler, in.uv - t, 0.0).rgb;
  col += textureSampleLevel(src_tex, lin_sampler, in.uv + vec2f(t.x, -t.y), 0.0).rgb;
  col += textureSampleLevel(src_tex, lin_sampler, in.uv + vec2f(-t.x, t.y), 0.0).rgb;
  return vec4f(col / 16.0, 1.0);
}

fn aces(x: vec3f) -> vec3f {
  let a = x * (2.51 * x + vec3f(0.03));
  let b = x * (2.43 * x + vec3f(0.59)) + vec3f(0.14);
  return clamp(a / b, vec3f(0.0), vec3f(1.0));
}

// Troy Sobotka's AgX transform: sRGB inset -> log2 shaper -> 6th-order sigmoid -> linear outset.
fn tonemap_agx(x_in: vec3f) -> vec3f {
  let c = max(x_in, vec3f(1e-10));
  let inset = vec3f(
    dot(c, vec3f(0.84247906, 0.07843360, 0.07922375)),
    dot(c, vec3f(0.04232824, 0.87846864, 0.07916613)),
    dot(c, vec3f(0.04237565, 0.07843360, 0.87914297)),
  );
  let ev = clamp((log2(max(inset, vec3f(1e-10))) + vec3f(12.47393)) / 16.5, vec3f(0.0), vec3f(1.0));
  let x2 = ev * ev;
  let x4 = x2 * x2;
  let p = 15.5 * x4 * x2 - 40.14 * x4 * ev + 31.96 * x4 - 6.868 * x2 * ev + 0.4298 * x2 + 0.1191 * ev - vec3f(0.00232);
  let lin = pow(max(p, vec3f(0.0)), vec3f(2.2));
  let outset = vec3f(
    dot(lin, vec3f(1.19687901, -0.09802088, -0.09902974)),
    dot(lin, vec3f(-0.05289685, 1.15190313, -0.09896118)),
    dot(lin, vec3f(-0.05297164, -0.09804345, 1.15107367)),
  );
  return clamp(outset, vec3f(0.0), vec3f(1.0));
}

// Stephen Hill's sRGB -> AP1 RRT+ODT fitted ACES transform -> linear sRGB.
fn tonemap_aces_fitted(x_in: vec3f) -> vec3f {
  let c = max(x_in, vec3f(0.0));
  let v = vec3f(
    dot(c, vec3f(0.59719, 0.35458, 0.04823)),
    dot(c, vec3f(0.07600, 0.90834, 0.01566)),
    dot(c, vec3f(0.02840, 0.13383, 0.83777)),
  );
  let a = v * (v + vec3f(0.0245786)) - vec3f(0.000090537);
  let b = v * (0.983729 * v + vec3f(0.4329510)) + vec3f(0.238081);
  let rrt = a / b;
  let out = vec3f(
    dot(rrt, vec3f(1.60475, -0.53108, -0.07367)),
    dot(rrt, vec3f(-0.10208, 1.10813, -0.00605)),
    dot(rrt, vec3f(-0.00327, -0.07276, 1.07602)),
  );
  return clamp(out, vec3f(0.0), vec3f(1.0));
}

fn apply_tonemap(x: vec3f, mode: i32) -> vec3f {
  if (mode == 1) {
    return tonemap_agx(x);
  }
  if (mode == 2) {
    return tonemap_aces_fitted(x);
  }
  return aces(x * 0.8);
}

// Tone mapping and grading to display-referred linear colour.
// pc.a = (exposure, bloom, saturation, contrast), pc.b = lift rgb + vignette,
// pc.c = (gain rgb, tonemap operator: 0 stylized, 1 agx, 2 aces_fitted)
@fragment fn fs_tonemap(in: Screen) -> @location(0) vec4f {
  let px = vec2i(in.clip.xy);
  var col = textureLoad(src_tex, px, 0).rgb;
  let bloom = textureSampleLevel(extra_tex, lin_sampler, in.uv, 0.0).rgb;
  col = (col + bloom * pc.a.y) * pc.a.x;
  col = apply_tonemap(col, i32(pc.c.w + 0.5));
  let l = luminance(col);
  col = max(mix(vec3f(l), col, pc.a.z), vec3f(0.0));
  // Contrast around mid grey in perceptual space.
  let g = pow(col, vec3f(1.0 / 2.2));
  let c = clamp((g - vec3f(0.5)) * pc.a.w + vec3f(0.5), vec3f(0.0), vec3f(1.0));
  col = pow(c, vec3f(2.2));
  col = col * pc.c.rgb + pc.b.rgb * (vec3f(1.0) - col);
  let v = length(in.uv - vec2f(0.5)) * 1.35;
  col *= 1.0 - pc.b.w * pow(v, 2.6);
  return vec4f(clamp(col, vec3f(0.0), vec3f(1.0)), 1.0);
}

// Depth of field (pc.b = focus distance m, depth to the full blur m, blur radius px, on): how far
// a pixel's colour spreads, in pixels. Nothing nearer than the focus spreads; past it the spread
// grows to the full blur; the sky is as far as anything.
fn defocus_radius(p: vec2i) -> f32 {
  let d = depth_px(p);
  let z = select(linear_depth(d), 1e5, d <= 0.0);
  return pc.b.z * smoothstep(pc.b.x, pc.b.x + pc.b.y, z);
}

// A defocused pixel: a disc of taps as wide as its spread, each counted only where its own spread
// reaches back here, so what is in focus (a foreground subject) never smears into the soft scene
// behind it.
fn defocus(uv: vec2f, px: vec2i, r: f32) -> vec3f {
  let size = vec2i(frame.viewport.xy);
  var sum = textureLoad(src_tex, px, 0).rgb;
  var weight = 1.0;
  for (var i = 0; i < 24; i++) {
    let spread = sqrt((f32(i) + 0.5) / 24.0) * r;
    let angle = f32(i) * 2.39996323;
    let off = vec2f(cos(angle), sin(angle)) * spread;
    let q = clamp(px + vec2i(round(off)), vec2i(0), size - vec2i(1));
    let w = smoothstep(spread - 1.0, spread, defocus_radius(q));
    sum += textureSampleLevel(src_tex, lin_sampler, uv + off * frame.viewport.zw, 0.0).rgb * w;
    weight += w;
  }
  return sum / weight;
}

// Painterly filter: a four-sector Kuwahara whose radius grows with distance, in perceptual space,
// after the depth of field. pc.a = (radius px near, radius px far, mix, unused), pc.b as for
// `defocus_radius`.
@fragment fn fs_paint(in: Screen) -> @location(0) vec4f {
  let px = vec2i(in.clip.xy);
  let size = vec2i(frame.viewport.xy);
  if (pc.b.w > 0.5) {
    let spread = defocus_radius(px);
    if (spread >= 0.75) {
      return vec4f(defocus(in.uv, px, spread), 1.0);
    }
  }
  let centre = pow(textureLoad(src_tex, px, 0).rgb, vec3f(1.0 / 2.2));
  let d = depth_px(px);
  let far = select(smoothstep(8.0, 70.0, linear_depth(d)), 1.0, d <= 0.0);
  let r = i32(round(mix(pc.a.x, pc.a.y, far)));
  if (r < 1) {
    return vec4f(pow(centre, vec3f(2.2)), 1.0);
  }
  var mean0 = vec3f(0.0);
  var mean1 = vec3f(0.0);
  var mean2 = vec3f(0.0);
  var mean3 = vec3f(0.0);
  var sq0 = vec3f(0.0);
  var sq1 = vec3f(0.0);
  var sq2 = vec3f(0.0);
  var sq3 = vec3f(0.0);
  for (var j = -r; j <= r; j++) {
    for (var i = -r; i <= r; i++) {
      let q = clamp(px + vec2i(i, j), vec2i(0), size - vec2i(1));
      let c = pow(textureLoad(src_tex, q, 0).rgb, vec3f(1.0 / 2.2));
      let cc = c * c;
      if (i <= 0 && j <= 0) {
        mean0 += c;
        sq0 += cc;
      }
      if (i >= 0 && j <= 0) {
        mean1 += c;
        sq1 += cc;
      }
      if (i <= 0 && j >= 0) {
        mean2 += c;
        sq2 += cc;
      }
      if (i >= 0 && j >= 0) {
        mean3 += c;
        sq3 += cc;
      }
    }
  }
  let n = f32((r + 1) * (r + 1));
  mean0 /= n;
  mean1 /= n;
  mean2 /= n;
  mean3 /= n;
  let v0 = dot(sq0 / n - mean0 * mean0, vec3f(1.0));
  let v1 = dot(sq1 / n - mean1 * mean1, vec3f(1.0));
  let v2 = dot(sq2 / n - mean2 * mean2, vec3f(1.0));
  let v3 = dot(sq3 / n - mean3 * mean3, vec3f(1.0));
  var best = mean0;
  var bv = v0;
  if (v1 < bv) {
    best = mean1;
    bv = v1;
  }
  if (v2 < bv) {
    best = mean2;
    bv = v2;
  }
  if (v3 < bv) {
    best = mean3;
  }
  let out = mix(centre, best, pc.a.z);
  return vec4f(pow(out, vec3f(2.2)), 1.0);
}

// Final output with FXAA-style edge smoothing and sharpening.
// pc.a = (enable FXAA, encode sRGB, sharpen, unused)
@fragment fn fs_final(in: Screen) -> @location(0) vec4f {
  let px = vec2i(in.clip.xy);
  let size = vec2i(frame.viewport.xy);
  var col = textureLoad(src_tex, px, 0).rgb;
  // Sharpening crisps what is near and leaves the distance soft: sharpened, fine far detail (a
  // canopy's petals) turns to grit.
  let d = depth_px(px);
  let sharpen = pc.a.z * select(1.0 - smoothstep(6.0, 18.0, linear_depth(d)), 0.0, d <= 0.0);
  if (sharpen > 0.0) {
    let ring = textureLoad(src_tex, clamp(px + vec2i(0, -1), vec2i(0), size - vec2i(1)), 0).rgb
      + textureLoad(src_tex, clamp(px + vec2i(0, 1), vec2i(0), size - vec2i(1)), 0).rgb
      + textureLoad(src_tex, clamp(px + vec2i(1, 0), vec2i(0), size - vec2i(1)), 0).rgb
      + textureLoad(src_tex, clamp(px + vec2i(-1, 0), vec2i(0), size - vec2i(1)), 0).rgb;
    let detail = col - ring * 0.25;
    // Limit the boost so edges sharpen without halos.
    col = max(col + clamp(detail * sharpen, vec3f(-0.08), vec3f(0.08)), vec3f(0.0));
  }
  if (pc.a.x > 0.5) {
    let ln = luminance(textureLoad(src_tex, clamp(px + vec2i(0, -1), vec2i(0), size - vec2i(1)), 0).rgb);
    let ls = luminance(textureLoad(src_tex, clamp(px + vec2i(0, 1), vec2i(0), size - vec2i(1)), 0).rgb);
    let le = luminance(textureLoad(src_tex, clamp(px + vec2i(1, 0), vec2i(0), size - vec2i(1)), 0).rgb);
    let lw = luminance(textureLoad(src_tex, clamp(px + vec2i(-1, 0), vec2i(0), size - vec2i(1)), 0).rgb);
    let lc = luminance(col);
    let lo = min(lc, min(min(ln, ls), min(le, lw)));
    let hi = max(lc, max(max(ln, ls), max(le, lw)));
    let range = hi - lo;
    if (range > max(0.04, hi * 0.12)) {
      let horizontal = abs(ln + ls - 2.0 * lc) > abs(le + lw - 2.0 * lc);
      var across = vec2f(frame.viewport.z, 0.0);
      if (horizontal) {
        across = vec2f(0.0, frame.viewport.w);
      }
      let a = textureSampleLevel(src_tex, lin_sampler, in.uv + across * 0.5, 0.0).rgb;
      let b = textureSampleLevel(src_tex, lin_sampler, in.uv - across * 0.5, 0.0).rgb;
      let blend = clamp(range * 1.5, 0.0, 0.6);
      col = mix(col, (a + b) * 0.5, blend);
    }
  }
  if (pc.a.y > 0.5) {
    col = select(1.055 * pow(col, vec3f(1.0 / 2.4)) - vec3f(0.055), col * 12.92, col <= vec3f(0.0031308));
  }
  return vec4f(col, 1.0);
}

// The lit picture at pixel `p` tone-mapped, graded and vignetted, clamped to display range: what
// `fs_tonemap` writes there. pc.a = (exposure, bloom, saturation, contrast), pc.b = lift rgb +
// vignette, pc.c = (gain rgb, tonemap operator). `identity`: the grade is saturation 1, contrast
// 1, lift 0 and gain 1, which leave the tone-mapped colour as it is, held to [0, 1] (the
// saturation's floor, the contrast's ceiling): it is held there without them.
fn graded_at(p: vec2i, identity: bool) -> vec3f {
  let size = vec2i(frame.viewport.xy);
  let q = clamp(p, vec2i(0), size - vec2i(1));
  let uv = (vec2f(q) + vec2f(0.5)) * frame.viewport.zw;
  var col = textureLoad(src_tex, q, 0).rgb;
  let bloom = textureSampleLevel(extra_tex, lin_sampler, uv, 0.0).rgb;
  col = (col + bloom * pc.a.y) * pc.a.x;
  col = apply_tonemap(col, i32(pc.c.w + 0.5));
  if (identity) {
    col = clamp(col, vec3f(0.0), vec3f(1.0));
  } else {
    let l = luminance(col);
    col = max(mix(vec3f(l), col, pc.a.z), vec3f(0.0));
    let g = pow(col, vec3f(1.0 / 2.2));
    let c = clamp((g - vec3f(0.5)) * pc.a.w + vec3f(0.5), vec3f(0.0), vec3f(1.0));
    col = pow(c, vec3f(2.2));
    col = col * pc.c.rgb + pc.b.rgb * (vec3f(1.0) - col);
  }
  let v = length(uv - vec2f(0.5)) * 1.35;
  col *= 1.0 - pc.b.w * pow(v, 2.6);
  return clamp(col, vec3f(0.0), vec3f(1.0));
}

// `fs_tonemap` and `fs_final` in one pass, for when no painterly filter runs between them: the
// same sharpening and edge smoothing on the same tone-mapped pixels, computed here for the pixel
// and its four neighbours rather than written out and read back. pc.a..c as `graded_at`, pc.d =
// (sharpen, edge smoothing on, encode sRGB, unused).
fn tonemap_output(in: Screen, identity: bool) -> vec4f {
  let px = vec2i(in.clip.xy);
  let centre = graded_at(px, identity);
  var col = centre;
  let d = depth_px(px);
  let sharpen = pc.d.x * select(1.0 - smoothstep(6.0, 18.0, linear_depth(d)), 0.0, d <= 0.0);
  let smoothing = pc.d.y > 0.5;
  if (sharpen > 0.0 || smoothing) {
    let n = graded_at(px + vec2i(0, -1), identity);
    let s = graded_at(px + vec2i(0, 1), identity);
    let e = graded_at(px + vec2i(1, 0), identity);
    let w = graded_at(px + vec2i(-1, 0), identity);
    if (sharpen > 0.0) {
      let detail = col - (n + s + e + w) * 0.25;
      col = max(col + clamp(detail * sharpen, vec3f(-0.08), vec3f(0.08)), vec3f(0.0));
    }
    if (smoothing) {
      let ln = luminance(n);
      let ls = luminance(s);
      let le = luminance(e);
      let lw = luminance(w);
      let lc = luminance(col);
      let lo = min(lc, min(min(ln, ls), min(le, lw)));
      let hi = max(lc, max(max(ln, ls), max(le, lw)));
      let range = hi - lo;
      if (range > max(0.04, hi * 0.12)) {
        // Half a pixel either way across the edge: between this pixel and its neighbour.
        let horizontal = abs(ln + ls - 2.0 * lc) > abs(le + lw - 2.0 * lc);
        let a = (centre + select(e, s, horizontal)) * 0.5;
        let b = (centre + select(w, n, horizontal)) * 0.5;
        col = mix(col, (a + b) * 0.5, clamp(range * 1.5, 0.0, 0.6));
      }
    }
  }
  if (pc.d.z > 0.5) {
    col = select(1.055 * pow(col, vec3f(1.0 / 2.4)) - vec3f(0.055), col * 12.92, col <= vec3f(0.0031308));
  }
  return vec4f(col, 1.0);
}

@fragment fn fs_tonemap_output(in: Screen) -> @location(0) vec4f {
  return tonemap_output(in, false);
}

// `fs_tonemap_output` for a Look whose grade changes nothing (`graded_at`'s `identity`): its five
// taps a pixel skip the gamma round trip, the saturation and the lift and gain.
@fragment fn fs_tonemap_output_plain(in: Screen) -> @location(0) vec4f {
  return tonemap_output(in, true);
}

// Samples a 16x16x16 colour-grading LUT packed as a 256x16 2D strip, parameterized by
// `x / (1 + x)` so `[0, inf)` scene-linear HDR maps smoothly into `[0, 1]`.
fn sample_lut(hdr: vec3f) -> vec3f {
  let c = max(hdr, vec3f(0.0));
  let u = c / (vec3f(1.0) + c);
  let r = (u.r * 15.0 + 0.5) / 256.0;
  let g = (u.g * 15.0 + 0.5) / 16.0;
  let b = u.b * 15.0;
  let b0 = floor(b);
  let b1 = min(b0 + 1.0, 15.0);
  let f = b - b0;
  let c0 = textureSampleLevel(lut_tex, lin_sampler, vec2f(r + b0 / 16.0, g), 0.0).rgb;
  let c1 = textureSampleLevel(lut_tex, lin_sampler, vec2f(r + b1 / 16.0, g), 0.0).rgb;
  return mix(c0, c1, f);
}

// Fused single-pass tone mapping, 3D LUT colour grading, vignette, sharpening, and sRGB output
// when the painterly Kuwahara pass is disabled (such as `Quality::hybrid_stage()`).
// pc.a = (exposure, bloom, vignette, sharpen), pc.b = (encode sRGB, enable FXAA, unused, unused)
@fragment fn fs_tonemap_final(in: Screen) -> @location(0) vec4f {
  let px = vec2i(in.clip.xy);
  let size = vec2i(frame.viewport.xy);
  let bloom = textureSampleLevel(extra_tex, lin_sampler, in.uv, 0.0).rgb * pc.a.y;
  var hdr = (textureLoad(src_tex, px, 0).rgb + bloom) * pc.a.x;
  let d = depth_px(px);
  let sharpen = pc.a.w * select(1.0 - smoothstep(6.0, 18.0, linear_depth(d)), 0.0, d <= 0.0);
  if (sharpen > 0.0 || pc.b.y > 0.5) {
    let n0 = (textureLoad(src_tex, clamp(px + vec2i(0, -1), vec2i(0), size - vec2i(1)), 0).rgb + bloom) * pc.a.x;
    let s0 = (textureLoad(src_tex, clamp(px + vec2i(0, 1), vec2i(0), size - vec2i(1)), 0).rgb + bloom) * pc.a.x;
    let e0 = (textureLoad(src_tex, clamp(px + vec2i(1, 0), vec2i(0), size - vec2i(1)), 0).rgb + bloom) * pc.a.x;
    let w0 = (textureLoad(src_tex, clamp(px + vec2i(-1, 0), vec2i(0), size - vec2i(1)), 0).rgb + bloom) * pc.a.x;
    if (sharpen > 0.0) {
      let detail = hdr - (n0 + s0 + e0 + w0) * 0.25;
      hdr = max(hdr + clamp(detail * sharpen, vec3f(-0.12), vec3f(0.12)), vec3f(0.0));
    }
    if (pc.b.y > 0.5) {
      let ln = luminance(n0);
      let ls = luminance(s0);
      let le = luminance(e0);
      let lw = luminance(w0);
      let lc = luminance(hdr);
      let lo = min(lc, min(min(ln, ls), min(le, lw)));
      let hi = max(lc, max(max(ln, ls), max(le, lw)));
      if (hi - lo > max(0.05, hi * 0.14)) {
        hdr = mix(hdr, (n0 + s0 + e0 + w0) * 0.25, 0.35);
      }
    }
  }
  var col = sample_lut(hdr);
  let v = length(in.uv - vec2f(0.5)) * 1.35;
  col *= 1.0 - pc.a.z * pow(v, 2.6);
  col = clamp(col, vec3f(0.0), vec3f(1.0));
  if (pc.b.x > 0.5) {
    col = select(1.055 * pow(col, vec3f(1.0 / 2.4)) - vec3f(0.055), col * 12.92, col <= vec3f(0.0031308));
  }
  return vec4f(col, 1.0);
}
