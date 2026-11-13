// Unified surface response library for Ganbate Engine: art-directed stylized cel/character
// shading and physically-based microfacet BRDFs (GGX, ClearCoat, Charlie cloth, Subsurface skin)
// sharing identical scene lighting, shadows, tiled lamps, and material inputs.
#define_import_path brdf
#import common

// Compile-time specialization switches (`ganbate_shader::CompileOptions`).
const ENABLE_STYLIZED: bool = true;
const ENABLE_PHOTOREAL: bool = true;
const FIXED_SURFACE_MODEL: i32 = -1;

struct SurfaceInput {
  albedo: vec3f,
  n: vec3f,
  geo_n: vec3f,
  v: vec3f,
  l: vec3f,
  key: vec3f,
  hemi: vec3f,
  lamps: vec3f,
  sh: f32,
  rough: f32,
  metallic: f32,
  ao: f32,
  brush: f32,
  foliage: bool,
  model: i32,
  lobe_weight: f32, // < 0 uses Look / model default
  lobe_param: f32,  // < 0 uses Look / model default
};

fn d_ggx(nh: f32, a2: f32) -> f32 {
  let d = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-6);
}

fn v_smith_ggx(nv: f32, nl: f32, a2: f32) -> f32 {
  let gv = nl * sqrt(nv * nv * (1.0 - a2) + a2);
  let gl = nv * sqrt(nl * nl * (1.0 - a2) + a2);
  return 0.5 / max(gv + gl, 1e-5);
}

fn f_schlick(vh: f32, f0: vec3f) -> vec3f {
  let m = clamp(1.0 - vh, 0.0, 1.0);
  let m2 = m * m;
  return f0 + (vec3f(1.0) - f0) * (m2 * m2 * m);
}

fn f_schlick_roughness(nv: f32, f0: vec3f, rough: f32) -> vec3f {
  let m = clamp(1.0 - nv, 0.0, 1.0);
  let m2 = m * m;
  let hi = max(vec3f(1.0 - rough), f0);
  return f0 + (hi - f0) * (m2 * m2 * m);
}

fn fd_burley(nv: f32, nl: f32, lh: f32, rough: f32) -> f32 {
  let f90 = 0.5 + 2.0 * rough * lh * lh;
  let fv = 1.0 + (f90 - 1.0) * pow(clamp(1.0 - nv, 0.0, 1.0), 5.0);
  let fl = 1.0 + (f90 - 1.0) * pow(clamp(1.0 - nl, 0.0, 1.0), 5.0);
  return (fv * fl) / PI;
}

fn d_charlie(nh: f32, rough: f32) -> f32 {
  let inv_r = 1.0 / max(rough, 0.05);
  let sin2h = max(1.0 - nh * nh, 1e-5);
  return (2.0 + inv_r) * pow(sin2h, 0.5 * inv_r) / (2.0 * PI);
}

fn sky_ibl(r_dir: vec3f, rough: f32, hemi: vec3f) -> vec3f {
  let crisp = sky_base(r_dir);
  return mix(crisp, hemi, rough * rough * 0.82);
}

// Model 0: Ganbate soft-cel ramp, shadow tint, toon specular, and upright-gated fresnel rim.
fn eval_stylized_ramp(s: SurfaceInput) -> vec3f {
  let ndl = dot(s.n, s.l);
  let half_l = ndl * 0.5 + 0.5 + s.brush * 0.45;
  var lit = smoothstep(frame.ramp.x - frame.ramp.y, frame.ramp.x + frame.ramp.y, half_l) * s.sh;
  var glow = vec3f(0.0);
  if (s.foliage) {
    let wrap = frame.style.z;
    let soft = smoothstep(0.0, 1.0, (ndl + wrap) / (1.0 + wrap));
    lit = mix(lit, soft * mix(0.3, 1.0, s.sh), 0.65);
    let back = pow(clamp(dot(-s.l, s.v), 0.0, 1.0), 3.0);
    glow = s.albedo * s.key * (back * frame.style.w * mix(0.25, 1.0, s.sh));
  }
  let dark = mix(vec3f(1.0), frame.shadow_tint.rgb, frame.shadow_tint.w);
  var ambient = s.hemi;
  if (s.foliage) {
    ambient = s.hemi * (1.0 + frame.style.w);
  }
  let diffuse = s.key * lit + ambient * mix(dark, vec3f(1.0), lit);
  var col = s.albedo * (diffuse + s.lamps) * (1.0 - 0.5 * s.metallic) + glow;

  let hv = normalize(s.l + s.v);
  let nh = max(dot(s.n, hv), 0.0);
  let size = frame.style.y * s.rough;
  let spec_t = 1.0 - size * 0.12;
  let gloss = 1.0 - s.rough;
  let spec = smoothstep(spec_t - 0.004, spec_t + 0.004, nh) * frame.style.x * (gloss * gloss * gloss) * lit;
  col += mix(vec3f(1.0), s.albedo, s.metallic) * s.key * spec;

  let fres = pow(1.0 - clamp(dot(s.n, s.v), 0.0, 1.0), frame.ramp.w);
  let rim_col = mix(frame.amb_sky.rgb * 1.6, s.key, clamp(ndl * 0.5 + 0.5, 0.0, 1.0));
  let upright = 1.0 - smoothstep(0.3, 0.75, s.geo_n.y);
  col += rim_col * (fres * frame.ramp.z * (0.35 + 0.65 * s.sh) * upright);
  return col;
}

// Model 1: Art-directed character cel shading with warm terminator scatter and anisotropic ribbon highlight.
fn eval_stylized_character(s: SurfaceInput) -> vec3f {
  let ndl = dot(s.n, s.l);
  let half_l = ndl * 0.5 + 0.5 + s.brush * 0.2;
  let sharpness = select(0.32, clamp(1.0 - s.lobe_weight, 0.08, 1.0), s.lobe_weight >= 0.0);
  let band = max(frame.ramp.y * sharpness, 0.005);
  let lit = smoothstep(frame.ramp.x - band, frame.ramp.x + band, half_l) * s.sh;
  let term_in = smoothstep(frame.ramp.x - 0.14, frame.ramp.x, half_l);
  let term_out = 1.0 - smoothstep(frame.ramp.x, frame.ramp.x + 0.12, half_l);
  let sss_band = term_in * term_out * s.sh * max(frame.surface_mode.w, 0.35);

  let dark = mix(vec3f(1.0), frame.shadow_tint.rgb, frame.shadow_tint.w);
  let diffuse = s.key * (lit + frame.subsurface_color.rgb * sss_band * 0.55)
    + s.hemi * mix(dark, vec3f(1.0), lit) * s.ao;
  var col = s.albedo * (diffuse + s.lamps) * (1.0 - 0.45 * s.metallic);

  // Tangent-aligned ribbon highlight for hair, eyes, and lacquered trim.
  let hv = normalize(s.l + s.v);
  var up_axis = vec3f(0.0, 1.0, 0.0);
  if (abs(s.n.y) > 0.92) {
    up_axis = vec3f(1.0, 0.0, 0.0);
  }
  let tangent = normalize(cross(up_axis, s.n));
  let aniso = select(0.55, clamp(s.lobe_param, 0.0, 1.0), s.lobe_param >= 0.0);
  let th = dot(tangent, hv);
  let sin_th = sqrt(max(1.0 - th * th, 0.0));
  let nh_eff = mix(max(dot(s.n, hv), 0.0), sin_th, aniso * 0.65);
  let gloss = 1.0 - s.rough;
  let spec_edge = 1.0 - (0.02 + 0.14 * s.rough);
  let spec = smoothstep(spec_edge - 0.01, spec_edge + 0.01, nh_eff) * max(frame.style.x, 0.45) * gloss * gloss * lit;
  col += mix(vec3f(1.0), s.albedo, s.metallic) * s.key * spec;

  let nv = clamp(dot(s.n, s.v), 0.0, 1.0);
  let rim = smoothstep(0.55, 0.88, 1.0 - nv) * clamp(ndl * 0.65 + 0.35, 0.0, 1.0);
  col += mix(frame.amb_sky.rgb * 1.4, s.key, 0.65) * (rim * max(frame.ramp.z, 0.25) * (0.4 + 0.6 * s.sh));
  return col;
}

// Model 2: Energy-conserving microfacet GGX + height-correlated Smith G2 + Burley diffuse + sky IBL.
fn eval_standard_ggx(s: SurfaceInput, crisp_ibl: vec3f, horizon_occl: f32) -> vec3f {
  let r = clamp(s.rough, 0.045, 1.0);
  let a = r * r;
  let a2 = a * a;
  let nv = max(dot(s.n, s.v), 1e-4);
  let nl_raw = dot(s.n, s.l);
  let nl = max(nl_raw, 0.0);
  let hv = normalize(s.l + s.v);
  let nh = max(dot(s.n, hv), 0.0);
  let vh = max(dot(s.v, hv), 0.0);
  let lh = max(dot(s.l, hv), 0.0);

  let f0 = mix(vec3f(0.04), s.albedo, s.metallic);
  let f = f_schlick(vh, f0);
  let d = d_ggx(nh, a2);
  let vis = v_smith_ggx(nv, max(nl, 1e-4), a2);
  // Multi-scattering energy compensation for rough metals and dielectrics.
  let energy_comp = vec3f(1.0) + f0 * ((1.0 / max(1.0 - r * 0.42, 0.25)) - 1.0);
  let spec_brdf = d * vis * f * energy_comp;

  let kd = (vec3f(1.0) - f) * (1.0 - s.metallic);
  var diff_term = kd * s.albedo * fd_burley(nv, max(nl, 1e-4), lh, r);
  var key_vis = nl * s.sh;
  if (s.foliage) {
    let wrap = frame.style.z;
    key_vis = clamp((nl_raw + wrap) / (1.0 + wrap), 0.0, 1.0) * mix(0.25, 1.0, s.sh);
    let back = pow(clamp(dot(-s.l, s.v), 0.0, 1.0), 3.0) * frame.style.w * mix(0.25, 1.0, s.sh);
    diff_term += s.albedo * back * (1.0 / PI);
  }

  var acc = (diff_term + spec_brdf * step(0.0, nl_raw)) * s.key * key_vis;
  let f_ibl = f_schlick_roughness(nv, f0, r);
  let kd_ibl = (vec3f(1.0) - f_ibl) * (1.0 - s.metallic);
  acc += kd_ibl * s.albedo * s.hemi * s.ao;
  let spec_ao = clamp(pow(nv + s.ao, a) - 1.0 + s.ao, 0.0, 1.0) * horizon_occl * horizon_occl;
  acc += mix(crisp_ibl, s.hemi, r * r * 0.82) * f_ibl * (1.0 - r * 0.55) * spec_ao;
  return acc + (s.albedo * (1.0 - 0.75 * s.metallic) * s.lamps + f0 * s.lamps * (1.0 - r * 0.6));
}

// Model 3: Base GGX substrate under a smooth dielectric clear-coat layer (automotive, lacquer, ceramics).
fn eval_clearcoat_over(base: vec3f, s: SurfaceInput) -> vec3f {
  let coat_w = select(max(frame.surface_mode.y, 0.85), clamp(s.lobe_weight, 0.0, 1.0), s.lobe_weight >= 0.0);
  let coat_r = clamp(select(frame.subsurface_color.w, s.lobe_param, s.lobe_param >= 0.0), 0.03, 1.0);
  let ac = coat_r * coat_r;
  let ac2 = ac * ac;

  let cn = normalize(mix(s.geo_n, s.n, 0.25));
  let r_dir = reflect(-s.v, cn);
  let nv = max(dot(cn, s.v), 1e-4);
  let nl = max(dot(cn, s.l), 0.0);
  let hv = normalize(s.l + s.v);
  let nh = max(dot(cn, hv), 0.0);
  let vh = max(dot(s.v, hv), 0.0);

  let fc_ibl = f_schlick_roughness(nv, vec3f(0.04), coat_r).x * coat_w;
  let fc = f_schlick(vh, vec3f(0.04)).x * coat_w;
  let dc = d_ggx(nh, ac2);
  let vc = v_smith_ggx(nv, max(nl, 1e-4), ac2);
  var acc = base * (1.0 - fc_ibl * 0.85) + s.key * (dc * vc * fc * nl * s.sh);
  acc += sky_ibl(r_dir, coat_r, s.hemi) * fc_ibl * s.ao;
  return acc + s.lamps * (fc_ibl * 0.35);
}

// Model 4: Estevez & Kulla Charlie sheen BRDF for woven cloth, velvet, silk, and felt-like fabrics.
fn eval_charlie_cloth(s: SurfaceInput) -> vec3f {
  let r = clamp(s.rough, 0.08, 1.0);
  let nv = max(dot(s.n, s.v), 1e-4);
  let nl_raw = dot(s.n, s.l);
  let nl = max(nl_raw, 0.0);
  let hv = normalize(s.l + s.v);
  let nh = max(dot(s.n, hv), 0.0);

  let sheen_w = select(max(frame.surface_mode.z, 0.75), clamp(s.lobe_weight, 0.0, 1.0), s.lobe_weight >= 0.0);
  let sheen_tint = select(0.5, clamp(s.lobe_param, 0.0, 1.0), s.lobe_param >= 0.0);
  let lum = max(luminance(s.albedo), 0.04);
  let tint_col = s.albedo / lum;
  let sheen_col = mix(vec3f(1.0), tint_col, sheen_tint) * sheen_w;

  let dc = d_charlie(nh, r);
  let vc = 1.0 / max(4.0 * (nl + nv - nl * nv), 0.12);
  let cloth_spec = sheen_col * (dc * vc);

  // Soft fabric wrap diffuse + grazing fiber scatter.
  let wrap_nl = clamp((nl_raw + 0.22) / 1.22, 0.0, 1.0);
  let diff = s.albedo * (1.0 / PI) * wrap_nl * mix(0.2, 1.0, s.sh);
  let direct = s.key * (diff + cloth_spec * nl * s.sh);

  let graze = pow(clamp(1.0 - nv, 0.0, 1.0), 3.0);
  let env = (s.albedo * s.hemi + sheen_col * s.hemi * graze * 0.65) * s.ao;
  return direct + env + (s.albedo + sheen_col * 0.25) * s.lamps * (1.0 / PI * 2.2);
}

// Model 5: Dual-lobe specular + RGB wrapped subsurface profile + back-lit translucency (skin, wax, jade).
fn eval_subsurface_skin(s: SurfaceInput, crisp_ibl: vec3f) -> vec3f {
  let r = clamp(s.rough, 0.08, 1.0);
  let a1 = r * r;
  let a2_primary = a1 * a1;
  let r_lobe2 = clamp(r * 0.48, 0.05, 1.0);
  let a2_secondary = r_lobe2 * r_lobe2 * r_lobe2 * r_lobe2;

  let nv = max(dot(s.n, s.v), 1e-4);
  let nl_raw = dot(s.n, s.l);
  let nl = max(nl_raw, 0.0);
  let hv = normalize(s.l + s.v);
  let nh = max(dot(s.n, hv), 0.0);
  let vh = max(dot(s.v, hv), 0.0);

  let sss_w = select(max(frame.surface_mode.w, 0.65), clamp(s.lobe_weight, 0.0, 1.0), s.lobe_weight >= 0.0);
  let sss_col = clamp(frame.subsurface_color.rgb, vec3f(0.05), vec3f(1.0));

  // RGB-dependent wrap diffuse: red scatters furthest into the terminator, blue stays crisp.
  let wraps = vec3f(0.52, 0.24, 0.10) * sss_w;
  let nl_rgb = clamp((vec3f(nl_raw) + wraps) / (vec3f(1.0) + wraps), vec3f(0.0), vec3f(1.0));
  let shadow_bleed = mix(vec3f(s.sh), pow(vec3f(max(s.sh, 0.02)), vec3f(1.0) - sss_col * (0.55 * sss_w)), sss_w);
  let diff_light = nl_rgb * shadow_bleed * mix(vec3f(1.0), vec3f(1.0) + sss_col * 0.28, sss_w * (1.0 - nl));

  // Skin IOR ~ 1.38 => F0 = 0.028, dual specular lobe.
  let f = f_schlick(vh, vec3f(0.028));
  let d = mix(d_ggx(nh, a2_primary), d_ggx(nh, a2_secondary), 0.18);
  let vis = v_smith_ggx(nv, max(nl, 1e-4), a2_primary);
  let spec = d * vis * f * nl * s.sh;
  var acc = (s.albedo * (vec3f(1.0) - f) * (1.0 / PI) * diff_light * s.key) + spec * s.key;

  // Back-lit thin translucency.
  let back = pow(clamp(dot(-s.l, s.v), 0.0, 1.0), 2.5) * sss_w * mix(0.25, 1.0, s.sh);
  acc += s.albedo * sss_col * s.key * (back * 0.45);
  acc += s.albedo * s.hemi * mix(vec3f(s.ao), pow(vec3f(max(s.ao, 0.05)), vec3f(1.0) - sss_col * 0.45), sss_w);

  let f_ibl = f_schlick_roughness(nv, vec3f(0.028), r);
  acc += mix(crisp_ibl, s.hemi, r * r * 0.82) * f_ibl * (1.0 - r * 0.5) * s.ao;
  return acc + s.albedo * s.lamps * (1.0 + sss_w * 0.15);
}

fn evaluate_surface(s: SurfaceInput) -> vec3f {
  if (FIXED_SURFACE_MODEL == 0) {
    return eval_stylized_ramp(s);
  } else if (FIXED_SURFACE_MODEL == 1) {
    return eval_stylized_character(s);
  } else if (FIXED_SURFACE_MODEL == 2) {
    let rd = reflect(-s.v, s.n);
    return eval_standard_ggx(s, sky_base(rd), clamp(1.0 + dot(rd, s.geo_n), 0.0, 1.0));
  } else if (FIXED_SURFACE_MODEL == 3) {
    let rd = reflect(-s.v, s.n);
    return eval_clearcoat_over(eval_standard_ggx(s, sky_base(rd), clamp(1.0 + dot(rd, s.geo_n), 0.0, 1.0)), s);
  } else if (FIXED_SURFACE_MODEL == 4) {
    return eval_charlie_cloth(s);
  } else if (FIXED_SURFACE_MODEL == 5) {
    return eval_subsurface_skin(s, sky_base(reflect(-s.v, s.n)));
  }

  if (ENABLE_PHOTOREAL) {
    if (s.model == 4) {
      return eval_charlie_cloth(s);
    }
    if (s.model == 2 || s.model == 3 || s.model == 5) {
      let rd = reflect(-s.v, s.n);
      let ho = clamp(1.0 + dot(rd, s.geo_n), 0.0, 1.0);
      let crisp = sky_base(rd);
      if (s.model == 5) {
        return eval_subsurface_skin(s, crisp);
      }
      let base = eval_standard_ggx(s, crisp, ho);
      if (s.model == 2) {
        return base;
      }
      return eval_clearcoat_over(base, s);
    }
  }
  if (ENABLE_STYLIZED) {
    if (s.model == 1) {
      return eval_stylized_character(s);
    }
    return eval_stylized_ramp(s);
  }
  let rd = reflect(-s.v, s.n);
  return eval_standard_ggx(s, sky_base(rd), clamp(1.0 + dot(rd, s.geo_n), 0.0, 1.0));
}
