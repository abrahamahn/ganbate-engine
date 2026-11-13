// Procedural surface detail presets (roof tiles, wood planks, plaster, stone, grass/terrain ground, lattice) and brush noise.
#define_import_path scene_detail
#import common

// Coordinates on a surface's dominant plane.
fn plane_uv(p: vec3f, n: vec3f) -> vec2f {
  let a = abs(n);
  if (a.x > a.y && a.x > a.z) {
    return p.zy;
  }
  if (a.z > a.y) {
    return p.xy;
  }
  return p.xz;
}

// Brush strokes in 0..1: long soft dabs on the surface, fixed in the world so they never swim.
fn brush_noise(p: vec3f, n: vec3f, size: f32) -> f32 {
  let q = plane_uv(p, n) / size;
  let r = vec2f(q.x * 0.8 + q.y * 0.6, q.y * 0.8 - q.x * 0.6);
  return noise2(vec2f(r.x * 0.45, r.y * 1.8)) * 0.6 + noise2(vec2f(r.x * 1.1, r.y * 4.2) + vec2f(7.3, 1.9)) * 0.4;
}

struct Detail {
  albedo: vec3f,
  n: vec3f,
  cavity: f32,  // darkening in gutters, joints and seams
  rough: f32,   // replacement roughness, or < 0 to keep the material's
};

fn no_detail(albedo: vec3f, n: vec3f) -> Detail {
  var o: Detail;
  o.albedo = albedo;
  o.n = n;
  o.cavity = 1.0;
  o.rough = -1.0;
  return o;
}

// How many pattern cells of `size` metres along `axis` one pixel covers.
fn cells_per_pixel(axis: vec3f, size: f32, dpx: vec3f, dpy: vec3f) -> f32 {
  return (abs(dot(dpx, axis)) + abs(dot(dpy, axis))) / size;
}

// Rounded roof tiles in columns running down the slope, lapped row over row.
fn roof_tiles(p: vec3f, geo: vec3f, n: vec3f, albedo: vec3f, dpx: vec3f, dpy: vec3f) -> Detail {
  var o = no_detail(albedo, n);
  if (geo.y < 0.12) {
    return o;
  }
  let h = vec2f(geo.x, geo.z);
  let hl = length(h);
  var down = vec3f(0.0, 0.0, 1.0);
  if (hl > 0.03) {
    down = vec3f(h.x / hl, 0.0, h.y / hl);
  }
  let e = vec3f(-down.z, 0.0, down.x);
  var s = normalize(cross(e, geo));
  if (s.y > 0.0) {
    s = -s;
  }
  let width = draw.detail0.y;
  let row = draw.detail0.z;
  let fade = 1.0 - smoothstep(0.25, 0.6, max(cells_per_pixel(e, width, dpx, dpy), cells_per_pixel(s, row, dpx, dpy)));
  let relief = draw.detail0.w * fade;
  let u = dot(p, e) / width;
  let v = dot(p, s) / row;
  let x = fract(u) * 2.0 - 1.0;
  let crest = sqrt(max(1.0 - x * x, 0.0));
  let fv = fract(v);
  let lap = smoothstep(0.0, 0.18, fv);
  let lip = smoothstep(0.78, 1.0, fv);
  o.n = normalize(n + e * (clamp(x / max(crest, 0.2), -2.5, 2.5) * 0.3 * relief) + s * ((lip * 0.5 - (1.0 - lap) * 0.35) * relief));
  let gutter = smoothstep(0.0, 0.45, crest);
  // Far away the pattern averages out to a slightly darker roof.
  o.cavity = mix(0.86, mix(0.5, 1.0, lap) * mix(0.62, 1.0, gutter), fade);
  let id = vec2f(floor(u), floor(v));
  let r = hash2(id) - 0.5;
  let r2 = hash2(id + vec2f(17.0, 3.0)) - 0.5;
  o.albedo = albedo * (1.0 + r * draw.detail1.x * 2.0 * fade) * (1.0 + vec3f(r2, r2 * 0.6, -r2) * 0.12 * fade);
  return o;
}

// Planks with staggered butt joints and grain, along world x (or z for faces looking along x).
fn wood(p: vec3f, geo: vec3f, n: vec3f, albedo: vec3f, dpx: vec3f, dpy: vec3f) -> Detail {
  var o = no_detail(albedo, n);
  let a = abs(geo);
  var along = vec3f(1.0, 0.0, 0.0);
  if (a.x > a.z && a.x > a.y * 0.5) {
    along = vec3f(0.0, 0.0, 1.0);
  }
  let across = normalize(cross(geo, along) + vec3f(0.0, 1e-4, 0.0));
  let width = draw.detail0.y;
  let length_m = draw.detail0.z;
  let fade = 1.0 - smoothstep(0.3, 0.7, cells_per_pixel(across, width, dpx, dpy));
  let u = dot(p, across) / width;
  let plank = floor(u);
  let w = dot(p, along) / length_m + hash2(vec2f(plank, 7.0));
  let id = vec2f(plank, floor(w));
  let r = hash2(id) - 0.5;
  let fu = fract(u);
  let seam = smoothstep(0.0, 0.07, fu) * smoothstep(1.0, 0.93, fu);
  let fw = fract(w);
  let butt = smoothstep(0.0, 0.012, fw) * smoothstep(1.0, 0.988, fw);
  let along_m = dot(p, along);
  let g = noise2(vec2f(along_m * 0.9 / width, fu * 7.0 + r * 31.0)) * 0.7 + noise2(vec2f(along_m * 3.1 / width, fu * 23.0 + r * 11.0)) * 0.3;
  let relief = draw.detail0.w * fade;
  o.cavity = mix(1.0, mix(0.55, 1.0, seam * butt), relief);
  o.albedo = albedo * (1.0 + r * draw.detail1.x * 2.0 * fade) * (1.0 + (g - 0.5) * 0.35 * relief);
  o.n = normalize(n + across * ((smoothstep(0.85, 1.0, fu) - smoothstep(0.15, 0.0, fu)) * 0.25 * relief));
  return o;
}

// Soft mottling, plus faint rain streaks on walls.
fn plaster(p: vec3f, geo: vec3f, n: vec3f, albedo: vec3f) -> Detail {
  var o = no_detail(albedo, n);
  let size = draw.detail0.y;
  let m = fbm2(plane_uv(p, geo) / size);
  var streak = 0.0;
  if (abs(geo.y) < 0.4) {
    let side = dot(p, normalize(vec3f(-geo.z, 0.0, geo.x) + vec3f(1e-4, 0.0, 0.0)));
    streak = smoothstep(0.55, 0.85, noise2(vec2f(side * 3.0 / size, p.y * 0.35)));
  }
  let k = draw.detail0.w;
  o.albedo = albedo * (1.0 + (m - 0.5) * draw.detail1.x * 2.0 * k - streak * 0.08 * k);
  return o;
}

// Irregular stones (a jittered cell pattern) with dark joints.
fn stone(p: vec3f, geo: vec3f, n: vec3f, albedo: vec3f, dp_len: f32) -> Detail {
  var o = no_detail(albedo, n);
  let size = draw.detail0.y;
  let fade = 1.0 - smoothstep(0.25, 0.6, dp_len / size);
  let uv = plane_uv(p, geo) / size;
  let c = floor(uv);
  let f = fract(uv);
  var d1 = 8.0;
  var d2 = 8.0;
  var id = vec2f(0.0);
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let cell = c + vec2f(f32(i), f32(j));
      let site = vec2f(f32(i), f32(j)) + vec2f(hash2(cell), hash2(cell + vec2f(31.0, 17.0))) * 0.8 + vec2f(0.1) - f;
      let d = dot(site, site);
      if (d < d1) {
        d2 = d1;
        d1 = d;
        id = cell;
      } else if (d < d2) {
        d2 = d;
      }
    }
  }
  let edge = sqrt(d2) - sqrt(d1);
  let joint = smoothstep(0.03, 0.14, edge);
  let relief = draw.detail0.w * fade;
  o.cavity = mix(1.0, mix(0.42, 1.0, joint), relief);
  o.albedo = albedo * (1.0 + (hash2(id) - 0.5) * draw.detail1.x * 2.0 * fade);
  return o;
}

// Grassy ground: the grass colour in large soft patches, so it meets the blades seamlessly.
fn grass_ground(p: vec3f, n: vec3f, albedo: vec3f) -> Detail {
  var o = no_detail(albedo, n);
  var base = albedo;
  if (draw.detail2.x > 0.5) {
    // Mostly the grass colour, keeping a little of the ground texture's markings.
    base = mix(albedo, draw.detail1.yzw, 0.75);
  }
  let m = fbm2(p.xz * 0.11);
  let m2 = noise2(p.xz * 0.9);
  let dry = vec3f(1.12, 1.06, 0.7);
  o.albedo = base * mix(vec3f(0.78), dry, smoothstep(0.35, 0.75, m)) * (0.92 + 0.16 * m2);
  return o;
}

// Terrain: grassy ground (as grass_ground) laid in broad meadow patches, giving way to beach
// sand near the water (wet and dark at the water line) and to bare rock in layers on steep
// slopes. Where sand and rock begin comes from terrain_cover, as in the ground map.
fn terrain_ground(p: vec3f, geo: vec3f, n: vec3f, albedo: vec3f, dp_len: f32) -> Detail {
  var o = grass_ground(p, n, albedo);
  let size = max(frame.terrain.w, 1.0);
  let broad = smoothstep(0.36, 0.66, fbm2(p.xz / size + vec2f(41.0, 13.0)));
  let meadow = frame.meadow.rgb * (0.9 + 0.2 * noise2(p.xz * 0.35));
  o.albedo = mix(o.albedo, meadow, broad * frame.sand.w);
  let c = terrain_cover(p, geo.y, frame.wet_sand.w, frame.terrain);
  if (c.x + c.y <= 0.0) {
    return o;
  }
  let rock_w = c.y;
  let above = p.y - frame.wet_sand.w;
  let dry = smoothstep(0.0, max(frame.terrain.x * 0.45, 0.05), above);
  let ripple = 0.94 + 0.06 * sin(dot(p.xz, vec2f(1.7, 0.6)) * 3.0 + noise2(p.xz * 0.8) * 4.0);
  let sand = mix(frame.wet_sand.rgb, frame.sand.rgb, dry) * ripple * (0.95 + 0.1 * noise2(p.xz * 2.3));
  o.albedo = mix(o.albedo, sand, c.x);
  // Rock: big angular facets (a jittered cell pattern on the face, stretched along it), each
  // its own tone, warmth and tilt, with thin dark cracks between them and faint sedimentary
  // bands; the cracks fade out where a facet is only a few pixels across.
  let face = plane_uv(p, geo);
  let q = vec2f(face.x / 6.5, (p.y + noise2(p.xz * 0.07) * 2.5) / 3.6);
  let cq = floor(q);
  let fq = fract(q);
  var d1 = 9.0;
  var d2 = 9.0;
  var id = vec2f(0.0);
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let cell = cq + vec2f(f32(i), f32(j));
      let site = vec2f(f32(i), f32(j)) + vec2f(hash2(cell), hash2(cell + vec2f(31.0, 17.0))) - fq;
      let d = dot(site, site);
      if (d < d1) {
        d2 = d1;
        d1 = d;
        id = cell;
      } else if (d < d2) {
        d2 = d;
      }
    }
  }
  let fade = 1.0 - smoothstep(0.08, 0.3, dp_len / 3.6);
  let crack = mix(1.0, smoothstep(0.006, 0.045, sqrt(d2) - sqrt(d1)), fade);
  let tone = hash2(id) - 0.5;
  let warm = hash2(id + vec2f(5.0, 9.0)) - 0.5;
  let band = sin(p.y * 0.85 + noise2(p.xz * 0.05) * 3.0) * 0.07;
  let grain = noise2(face * 1.7) * 0.6 + noise2(face * 5.3) * 0.4;
  let rock = frame.rock.rgb * (1.0 + tone * 0.32 + band) * (vec3f(1.0) + vec3f(warm, warm * 0.3, -warm) * 0.14) * (0.9 + 0.2 * grain);
  let tilt = vec3f(hash2(id + vec2f(1.3, 0.0)) - 0.5, (hash2(id + vec2f(0.0, 4.1)) - 0.5) * 0.6, hash2(id + vec2f(2.7, 0.0)) - 0.5) * 0.7;
  o.albedo = mix(o.albedo, rock, rock_w);
  o.cavity = mix(1.0, mix(0.62, 1.0, crack), rock_w);
  o.n = normalize(mix(o.n, normalize(n + tilt), rock_w));
  return o;
}

// A lattice of thin bars over the surface (a lantern's frame, paper ribs): cells `size u` wide
// and `size v` tall on the face's plane (a zero size leaves that direction's bars out), bars
// `strength` of a cell wide in the rule's colour (dark wood by default). The cavity term carries
// the bars, so they also shade a lamp's glow. Bars thinner than a pixel fade to their average
// cover rather than shimmer.
fn lattice(p: vec3f, geo: vec3f, n: vec3f, albedo: vec3f, dpx: vec3f, dpy: vec3f) -> Detail {
  var o = no_detail(albedo, n);
  let raw = vec2f(draw.detail0.y, draw.detail0.z);
  let used = vec2f(step(1e-4, raw.x), step(1e-4, raw.y));
  let size = max(raw, vec2f(1e-3));
  let q = plane_uv(p, geo) / size;
  let f = abs(fract(q) - vec2f(0.5));
  let half = 0.5 - draw.detail0.w * 0.5;
  // The pattern's footprint per pixel, from the world-space derivatives taken up front.
  let aa = max((abs(plane_uv(dpx, geo)) + abs(plane_uv(dpy, geo))) / size * 0.75, vec2f(1e-4));
  let bars = vec2f(smoothstep(half - aa.x, half + aa.x, f.x), smoothstep(half - aa.y, half + aa.y, f.y)) * used;
  // Where cells shrink under a few pixels the pattern averages to its coverage.
  let fade = 1.0 - smoothstep(0.15, 0.4, max(aa.x * used.x, aa.y * used.y));
  let average = draw.detail0.w * 0.6 * max(used.x, used.y);
  let cover = mix(average, max(bars.x, bars.y), fade);
  var wood = vec3f(0.06, 0.035, 0.02);
  if (draw.detail2.x > 0.5) {
    wood = draw.detail1.yzw;
  }
  o.albedo = mix(albedo, wood, cover);
  o.cavity = 1.0 - cover * 0.92;
  return o;
}

fn surface_detail(p: vec3f, geo: vec3f, n: vec3f, albedo: vec3f, dpx: vec3f, dpy: vec3f, dp_len: f32) -> Detail {
  let preset = i32(draw.detail0.x + 0.5);
  var base = albedo;
  if (draw.detail2.x > 0.5 && preset != 6 && preset != 7 && preset != 8) {
    base = draw.detail1.yzw;
  }
  if (preset == 1) {
    return roof_tiles(p, geo, n, base, dpx, dpy);
  }
  if (preset == 2) {
    return wood(p, geo, n, base, dpx, dpy);
  }
  if (preset == 3) {
    return plaster(p, geo, n, base);
  }
  if (preset == 4) {
    return stone(p, geo, n, base, dp_len);
  }
  if (preset == 6) {
    return grass_ground(p, n, albedo);
  }
  if (preset == 7) {
    return terrain_ground(p, geo, n, albedo, dp_len);
  }
  if (preset == 8) {
    return lattice(p, geo, n, albedo, dpx, dpy);
  }
  var o = no_detail(base, n);
  if (preset == 5) {
    o.rough = 0.12;
  }
  return o;
}

