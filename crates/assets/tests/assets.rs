use ganbate_assets::{cooked, glb, json, mip, png, zlib};
use ganbate_core::{
    Vec3,
    world::{
        AlphaMode, Emission, EntityRecord, EntityTable, Mesh, Shading, Texture, Transform, World,
    },
};

fn glb_file(doc: &str, bytes: &[u8]) -> Vec<u8> {
    let mut j = doc.as_bytes().to_vec();
    while j.len() % 4 != 0 {
        j.push(b' ')
    }
    let mut b = bytes.to_vec();
    while b.len() % 4 != 0 {
        b.push(0)
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2_u32.to_le_bytes());
    out.extend_from_slice(&(28 + j.len() as u32 + b.len() as u32).to_le_bytes());
    for (id, chunk) in [(0x4e4f534au32, j), (0x004e4942, b)] {
        out.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        out.extend_from_slice(&id.to_le_bytes());
        out.extend_from_slice(&chunk);
    }
    out
}
const DOC: &str = r#"{"asset":{"version":"2.0"},"buffers":[{"byteLength":54}],"bufferViews":[{"buffer":0,"byteLength":48,"byteStride":16},{"buffer":0,"byteOffset":48,"byteLength":6}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],"nodes":[{"mesh":0,"translation":[1,0,0],"children":[1]},{"mesh":0,"translation":[0,2,0]}],"scenes":[{"nodes":[0]}],"scene":0}"#;
fn bytes() -> Vec<u8> {
    let mut b = Vec::new();
    for v in [[0., 0., 0., 9.], [1., 0., 0., 9.], [0., 1., 0., 9.]] {
        for f in v {
            b.extend_from_slice(&f32::to_le_bytes(f));
        }
    }
    for i in [0u16, 1, 2] {
        b.extend_from_slice(&i.to_le_bytes());
    }
    b
}
#[test]
fn source_to_cooked_roundtrip_preserves_hierarchy_and_sharing() {
    let w = glb::import(&glb_file(DOC, &bytes())).unwrap();
    assert_eq!(
        (w.meshes.len(), w.instances.len(), w.validate().unwrap()),
        (1, 2, 2)
    );
    assert_eq!(
        w.scene().unwrap().triangles[1].positions[2],
        Vec3(1., 3., 0.)
    );
    let b = cooked::encode(&w).unwrap();
    assert_eq!(cooked::version(&b), Some(3));
    let w2 = cooked::decode(&b).unwrap();
    assert_eq!(w, w2);
    assert_eq!(cooked::encode(&w2).unwrap(), b);
    for n in 0..b.len() {
        assert!(cooked::decode(&b[..n]).is_err())
    }
    let mut future = b.clone();
    future[4] = 5;
    assert!(cooked::decode(&future).is_err());
    let mut extra = b;
    extra.push(0);
    assert!(cooked::decode(&extra).is_err());
}
#[test]
fn malformed_input_and_unsupported_features_fail_before_rendering() {
    for doc in [
        DOC.replace("\"children\":[1]", "\"children\":[0]"),
        DOC.replace("\"byteLength\":48", "\"byteLength\":4"),
        DOC.replace("\"nodes\":[0]", "\"nodes\":[2]"),
        DOC.replace("\"version\":\"2.0\"", "\"version\":\"9.0\""),
    ] {
        assert!(glb::import(&glb_file(&doc, &bytes())).is_err());
    }
    let mut b = bytes();
    b[48] = 9;
    assert!(glb::import(&glb_file(DOC, &b)).is_err());
    let ext = DOC.replacen('{', r#"{"extensionsRequired":["UNKNOWN_VENDOR"],"#, 1);
    assert!(
        glb::import(&glb_file(&ext, &bytes()))
            .unwrap_err()
            .contains("UNKNOWN_VENDOR")
    );
    let message = json::parse(br#"{"name":"\uD83D\uDE80"}"#).unwrap();
    assert_eq!(message.get("name").unwrap().string().unwrap(), "🚀");
    for bad in [
        &br#"[1,]"#[..],
        br#"{"a":1,"a":2}"#,
        br#""\uDFFF""#,
        b"01",
        b"1e999",
    ] {
        assert!(json::parse(bad).is_err())
    }
}

/// The version 1 layout, written by the previous cooker.
fn encode_v1(w: &World) -> Vec<u8> {
    let mut v = b"GBEW".to_vec();
    let word = |v: &mut Vec<u8>, n: u32| v.extend_from_slice(&n.to_le_bytes());
    word(&mut v, 1);
    word(&mut v, w.meshes.len() as u32);
    word(&mut v, w.instances.len() as u32);
    for m in &w.meshes {
        word(&mut v, m.positions.len() as u32);
        word(&mut v, m.indices.len() as u32);
        for f in m
            .color
            .into_iter()
            .chain(m.positions.iter().flat_map(|p| [p.0, p.1, p.2]))
        {
            v.extend_from_slice(&f.to_le_bytes());
        }
        for &i in &m.indices {
            word(&mut v, i);
        }
    }
    for i in &w.instances {
        word(&mut v, i.mesh);
        for f in i.transform.0 {
            v.extend_from_slice(&f.to_le_bytes());
        }
    }
    v
}
#[test]
fn version_1_files_still_decode() {
    let w = glb::import(&glb_file(DOC, &bytes())).unwrap();
    let v1 = encode_v1(&w);
    assert_eq!(cooked::version(&v1), Some(1));
    let decoded = cooked::decode(&v1).unwrap();
    assert_eq!(decoded, w);
    assert!(decoded.meshes[0].normals.is_empty() && decoded.meshes[0].material.is_none());
    for n in 0..v1.len() {
        assert!(cooked::decode(&v1[..n]).is_err())
    }
    let mut extra = v1.clone();
    extra.push(0);
    assert!(cooked::decode(&extra).is_err());
    // Re-cooking upgrades to version 3.
    assert_eq!(cooked::version(&cooked::encode(&decoded).unwrap()), Some(3));
}

// Synthetic GLB with materials, textures and a light.

struct Builder {
    bin: Vec<u8>,
    views: Vec<String>,
}
impl Builder {
    fn view(&mut self, data: &[u8]) -> usize {
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        self.views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{}}}"#,
            self.bin.len(),
            data.len()
        ));
        self.bin.extend_from_slice(data);
        self.views.len() - 1
    }
}
fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|f| f.to_le_bytes()).collect()
}
fn checker(w: u32, h: u32, a: [u8; 4], b: [u8; 4]) -> Vec<u8> {
    (0..w * h)
        .flat_map(|i| if (i % w + i / w) % 2 == 0 { a } else { b })
        .collect()
}

/// Two primitives (a textured foliage quad and an emissive lamp triangle), a normal map, an
/// image used both as colour and as a normal map, and a point light under translated nodes.
fn scene_glb(material_patch: &str) -> Vec<u8> {
    let mut b = Builder {
        bin: vec![],
        views: vec![],
    };
    let quad_pos = b.view(&floats(&[0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.]));
    let quad_nrm = b.view(&floats(&[0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1.]));
    let quad_uv: Vec<u8> = [0u16, 0, 65535, 0, 65535, 65535, 0, 32768]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let quad_uv = b.view(&quad_uv);
    let quad_idx: Vec<u8> = [0u16, 1, 2, 0, 2, 3]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let quad_idx = b.view(&quad_idx);
    let tri_pos = b.view(&floats(&[0., 0., 1., 0.5, 0., 1., 0., 0.5, 1.]));
    let tri_uv = b.view(&floats(&[0., 0., 2., 0., 0., -1.5]));
    let tri_idx = b.view(&[2u8, 1, 0]);
    let colour = png::encode(4, 4, &checker(4, 4, [255, 64, 0, 255], [0, 32, 255, 128])).unwrap();
    let normal = png::encode(
        3,
        2,
        &checker(3, 2, [128, 128, 255, 255], [255, 128, 128, 255]),
    )
    .unwrap();
    let img0 = b.view(&colour);
    let img1 = b.view(&normal);
    let doc = format!(
        r#"{{"asset":{{"version":"2.0"}},
        "extensionsUsed":["KHR_lights_punctual","KHR_materials_emissive_strength"],
        "extensions":{{"KHR_lights_punctual":{{"lights":[{{"type":"point","color":[1,0.8,0.6],"intensity":12,"range":6}}]}}}},
        "buffers":[{{"byteLength":{len}}}],
        "bufferViews":[{views}],
        "accessors":[
            {{"bufferView":{quad_pos},"componentType":5126,"count":4,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}},
            {{"bufferView":{quad_nrm},"componentType":5126,"count":4,"type":"VEC3"}},
            {{"bufferView":{quad_uv},"componentType":5123,"normalized":true,"count":4,"type":"VEC2"}},
            {{"bufferView":{quad_idx},"componentType":5123,"count":6,"type":"SCALAR"}},
            {{"bufferView":{tri_pos},"componentType":5126,"count":3,"type":"VEC3"}},
            {{"bufferView":{tri_uv},"componentType":5126,"count":3,"type":"VEC2"}},
            {{"bufferView":{tri_idx},"componentType":5121,"count":3,"type":"SCALAR"}}],
        "samplers":[{{"magFilter":9728,"minFilter":9728,"wrapS":33071}}],
        "images":[{{"bufferView":{img0},"mimeType":"image/png","name":"checker"}},{{"bufferView":{img1},"mimeType":"image/png"}}],
        "textures":[{{"sampler":0,"source":0}},{{"source":1}},{{"source":0}}],
        "materials":[
            {{"name":"leaf","pbrMetallicRoughness":{{"baseColorFactor":[0.8,0.9,1,1],"baseColorTexture":{{"index":0}},"metallicFactor":0,"roughnessFactor":0.7}},
              "normalTexture":{{"index":1,"scale":0.5}},"alphaMode":"MASK","alphaCutoff":0.3,"doubleSided":true,"extras":{{"ganbate":{{"shading":"foliage"}}}}}},
            {{"name":"lamp","pbrMetallicRoughness":{{"baseColorFactor":[1,0.5,0.2,0.75]}},"emissiveFactor":[1,0.5,0.25],
              "extensions":{{"KHR_materials_emissive_strength":{{"emissiveStrength":4}}}},"emissiveTexture":{{"index":0}},
              "normalTexture":{{"index":2}},"alphaMode":"BLEND","extras":{{"ganbate":{{"emission":"lamp"}}}}{material_patch}}}],
        "meshes":[{{"primitives":[
            {{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2}},"indices":3,"material":0}},
            {{"attributes":{{"POSITION":4,"TEXCOORD_0":5}},"indices":6,"material":1}}]}}],
        "nodes":[
            {{"name":"root","translation":[10,0,0],"children":[1,2]}},
            {{"mesh":0}},
            {{"translation":[0,2,0],"rotation":[0,0.7071068,0,0.7071068],"extensions":{{"KHR_lights_punctual":{{"light":0}}}}}}],
        "scenes":[{{"nodes":[0]}}],"scene":0}}"#,
        len = b.bin.len(),
        views = b.views.join(","),
    );
    glb_file(&doc, &b.bin)
}

#[test]
fn materials_textures_and_lights_cook_exactly() {
    let w = glb::import(&scene_glb("")).unwrap();
    assert_eq!(
        (w.meshes.len(), w.instances.len(), w.materials.len()),
        (2, 2, 2)
    );
    assert_eq!((w.textures.len(), w.lights.len()), (3, 1));
    let [quad, tri] = [&w.meshes[0], &w.meshes[1]];
    assert_eq!(quad.normals, vec![Vec3(0., 0., 1.); 4]);
    assert_eq!(
        quad.uvs,
        vec![[0., 0.], [1., 0.], [1., 1.], [0., 32768. / 65535.]]
    );
    assert_eq!(quad.indices, vec![0, 1, 2, 0, 2, 3]);
    assert_eq!((quad.material, quad.color), (Some(0), [0.8, 0.9, 1.]));
    assert!(tri.normals.is_empty());
    assert_eq!(tri.uvs, vec![[0., 0.], [2., 0.], [0., -1.5]]);
    assert_eq!(
        (tri.indices.clone(), tri.material),
        (vec![2, 1, 0], Some(1))
    );
    for inst in &w.instances {
        assert_eq!(inst.transform.point(Vec3(0., 0., 0.)), Vec3(10., 0., 0.));
    }
    let [leaf, lamp] = [&w.materials[0], &w.materials[1]];
    assert_eq!(leaf.name, "leaf");
    assert_eq!(leaf.base_color, [0.8, 0.9, 1., 1.]);
    assert_eq!(
        (leaf.base_color_texture, leaf.normal_texture),
        (Some(0), Some(1))
    );
    assert_eq!(
        (leaf.normal_scale, leaf.metallic, leaf.roughness),
        (0.5, 0., 0.7)
    );
    assert_eq!(
        (leaf.alpha, leaf.double_sided),
        (AlphaMode::Mask(0.3), true)
    );
    assert_eq!(
        (leaf.shading, leaf.emission),
        (Shading::Foliage, Emission::Constant)
    );
    assert_eq!(leaf.emissive, [0.; 3]);
    assert_eq!(lamp.emissive, [4., 2., 1.]);
    assert_eq!(
        (lamp.emissive_texture, lamp.normal_texture),
        (Some(0), Some(2))
    );
    assert_eq!((lamp.metallic, lamp.roughness), (1., 1.));
    assert_eq!((lamp.alpha, lamp.double_sided), (AlphaMode::Blend, false));
    assert_eq!(
        (lamp.shading, lamp.emission),
        (Shading::Standard, Emission::Lamp)
    );
    // One image in two roles is stored twice: sRGB colour and linear normal map.
    let t = &w.textures;
    assert_eq!(
        t.iter()
            .map(|t| (t.width, t.height, t.srgb, t.levels.len()))
            .collect::<Vec<_>>(),
        vec![(4, 4, true, 3), (3, 2, false, 2), (4, 4, false, 3)]
    );
    assert_eq!(
        t[0].levels[0],
        checker(4, 4, [255, 64, 0, 255], [0, 32, 255, 128])
    );
    assert_eq!(t[0].levels[0], t[2].levels[0]);
    assert_ne!(t[0].levels[1], t[2].levels[1]);
    let l = w.lights[0];
    assert_eq!((l.color, l.intensity, l.range), ([1., 0.8, 0.6], 12., 6.));
    assert_eq!(l.position, Vec3(10., 2., 0.));

    let cooked = cooked::encode(&w).unwrap();
    let decoded = cooked::decode(&cooked).unwrap();
    assert_eq!(decoded, w);
    assert_eq!(cooked::encode(&decoded).unwrap(), cooked);
    for n in (0..cooked.len()).step_by(7) {
        assert!(cooked::decode(&cooked[..n]).is_err());
    }
    // Every enumeration and index stays validated on decode.
    let mut corrupt = w.clone();
    corrupt.meshes[1].material = Some(9);
    assert!(cooked::encode(&corrupt).is_err());
    let mut corrupt = w.clone();
    corrupt.textures[0].levels.pop();
    assert!(corrupt.validate().is_err(), "a partial chain");
    let mut top = w.clone();
    top.textures[0].levels.truncate(1);
    assert!(top.validate().is_ok(), "the top level alone");
    let mut corrupt = w.clone();
    corrupt.meshes[0].uvs.pop();
    assert!(corrupt.validate().is_err());
    let mut corrupt = w;
    corrupt.lights[0].range = f32::NAN;
    assert!(corrupt.validate().is_err());
}

#[test]
fn unsupported_gltf_features_are_named() {
    let cases = [
        (
            r#""textures":[{"sampler":0,"source":0}"#,
            r#""textures":[{"sampler":0,"extensions":{"EXT_texture_webp":{"source":0}}}"#,
            "EXT_texture_webp",
        ),
        (
            r#""baseColorTexture":{"index":0}"#,
            r#""baseColorTexture":{"index":0,"texCoord":1}"#,
            "TEXCOORD_1",
        ),
        (
            r#""baseColorTexture":{"index":0}"#,
            r#""baseColorTexture":{"index":0,"extensions":{"KHR_texture_transform":{}}}"#,
            "KHR_texture_transform",
        ),
        (
            r#""mimeType":"image/png","name":"checker""#,
            r#""mimeType":"image/jpeg","name":"checker""#,
            "image/jpeg",
        ),
        (
            r#""name":"checker""#,
            r#""name":"checker","uri":"checker.png""#,
            "URI",
        ),
        (r#""type":"point""#, r#""type":"spot""#, "spot light"),
        (
            r#""count":6,"type":"SCALAR"}"#,
            r#""count":6,"type":"SCALAR","sparse":{}}"#,
            "sparse",
        ),
        (
            r#""roughnessFactor":0.7"#,
            r#""roughnessFactor":0.7,"metallicRoughnessTexture":{"index":0}"#,
            "metallicRoughnessTexture",
        ),
        (
            r#""alphaMode":"BLEND""#,
            r#""alphaMode":"BLEND","occlusionTexture":{"index":1}"#,
            "occlusionTexture",
        ),
        (r#""TEXCOORD_0":5"#, r#""TEXCOORD_9":5"#, "no TEXCOORD_0"),
        (
            r#""indices":3,"material":0}"#,
            r#""indices":3,"material":0,"extensions":{"KHR_draco_mesh_compression":{}}}"#,
            "KHR_draco_mesh_compression",
        ),
        (
            r#""scenes":[{"nodes":[0]}]"#,
            r#""scenes":[{"nodes":[0]}],"animations":[{}]"#,
            "animation",
        ),
        (r#"{"mesh":0}"#, r#"{"mesh":0,"skin":0}"#, "skinned"),
        (
            r#""bufferViews":[{"buffer":0,"#,
            r#""bufferViews":[{"extensions":{"EXT_meshopt_compression":{}},"buffer":0,"#,
            "EXT_meshopt_compression",
        ),
        (r#""shading":"foliage""#, r#""shading":"glossy""#, "glossy"),
        (
            r#""extensionsUsed""#,
            r#""extensionsRequired":["KHR_lights_punctual","EXT_mesh_gpu_instancing"],"extensionsUsed""#,
            "EXT_mesh_gpu_instancing",
        ),
    ];
    for (from, to, message) in cases {
        let e = glb::import(&patch(&scene_glb(""), from, to)).unwrap_err();
        assert!(e.contains(message), "{e:?} does not mention {message}");
    }
    // Supported required extensions, unknown material keys and samplers are accepted.
    let required = patch(
        &scene_glb(r#","x":0"#),
        r#""extensionsUsed""#,
        r#""extensionsRequired":["KHR_lights_punctual"],"extensionsUsed""#,
    );
    assert!(glb::import(&required).is_ok());
}

/// Replace text in a GLB's JSON chunk, keeping the container valid.
fn patch(glb: &[u8], from: &str, to: &str) -> Vec<u8> {
    let len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
    let doc = String::from_utf8(glb[20..20 + len].to_vec()).unwrap();
    assert!(doc.contains(from), "{from} not in document");
    let bin_start = 20 + len;
    let bin_len = u32::from_le_bytes(glb[bin_start..bin_start + 4].try_into().unwrap()) as usize;
    glb_file(
        &doc.replacen(from, to, 1),
        &glb[bin_start + 8..bin_start + 8 + bin_len],
    )
}

// PNG.

fn rgba(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .flat_map(|(x, y)| f(x, y))
        .collect()
}
fn check(data: &[u8], w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) {
    let image = png::decode(data).unwrap();
    assert_eq!((image.width, image.height), (w, h));
    assert_eq!(image.rgba, rgba(w, h, f));
}
fn wide(v: u32) -> u8 {
    ((v + 128) / 257) as u8
}

#[test]
fn decodes_reference_pngs() {
    // Written by Pillow and Python's zlib: dynamic, fixed and stored deflate blocks.
    check(GRADIENT_RGB, 16, 16, |x, y| {
        [
            (x * 17) as u8,
            (y * 17) as u8,
            ((x * x + 3 * y * y + x * y) & 255) as u8,
            255,
        ]
    });
    check(PALETTE_4BIT, 13, 7, |x, y| {
        let i = (x * 3 + y * 5) % 10;
        let alpha = [0, 128].get(i as usize).copied().unwrap_or(255);
        [
            (i * 25) as u8,
            (255 - i * 20) as u8,
            ((i * 60) & 255) as u8,
            alpha,
        ]
    });
    check(GREY_1BIT, 19, 5, |x, y| {
        let v = if (x + y) % 3 == 0 { 255 } else { 0 };
        [v, v, v, 255]
    });
    check(GREY_16BIT, 6, 4, |x, y| {
        let v = wide((x * 9000 + y * 1234) & 65535);
        [v, v, v, 255]
    });
    check(GREY_ALPHA, 9, 9, |x, y| {
        let v = ((x * 28) & 255) as u8;
        [v, v, v, ((255 - y * 28) & 255) as u8]
    });
    check(RGBA_STORED, 8, 8, |x, y| {
        [
            (x * 32) as u8,
            (y * 32) as u8,
            (255 - x * 16) as u8,
            (128 + y * 8) as u8,
        ]
    });
    check(RGB_FIXED, 4, 4, |_, _| [200, 100, 50, 255]);
    check(RGBA_16BIT, 3, 2, |x, y| {
        [
            wide(x * 20000),
            wide(y * 30000 + 255),
            wide(65535 - x * 257),
            wide(32768 + x * 1000),
        ]
    });
    check(GREY_2BIT_KEY, 5, 3, |x, y| {
        let v = (x + 2 * y) % 4;
        [
            (v * 85) as u8,
            (v * 85) as u8,
            (v * 85) as u8,
            if v == 2 { 0 } else { 255 },
        ]
    });
    check(RGB_KEY, 4, 2, |x, y| {
        [
            (x * 50) as u8,
            (y * 50) as u8,
            7,
            if (x, y) == (1, 1) { 0 } else { 255 },
        ]
    });
}

/// Pseudo-random bytes from a fixed seed.
fn noise(seed: u64) -> impl FnMut() -> u32 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as u32
    }
}

#[test]
fn png_round_trips_every_encoder_colour_type() {
    let mut rand = noise(7);
    let cases: Vec<(u32, u32, Vec<u8>, u8, u8)> = vec![
        (
            5,
            3,
            rgba(5, 3, |x, y| {
                [(x * 40 + y) as u8; 3]
                    .map(|v| v)
                    .into_iter()
                    .chain([255])
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            }),
            0,
            8,
        ),
        (
            9,
            4,
            rgba(9, 4, |x, y| {
                if (x + y) % 2 == 0 {
                    [10, 200, 30, 255]
                } else {
                    [0, 0, 0, 0]
                }
            }),
            3,
            1,
        ),
        (
            7,
            5,
            rgba(7, 5, |x, y| {
                [
                    [1, 2, 3, 255],
                    [9, 9, 9, 40],
                    [200, 1, 1, 255],
                    [0, 5, 0, 0],
                ][((x + y) % 4) as usize]
            }),
            3,
            2,
        ),
        (
            11,
            3,
            rgba(11, 3, |x, y| {
                [
                    (x % 4 * 60) as u8,
                    (y * 70) as u8,
                    99,
                    if x == 0 { 17 } else { 255 },
                ]
            }),
            3,
            4,
        ),
        (
            40,
            9,
            rgba(40, 9, |x, y| {
                [
                    (x % 20 * 6) as u8,
                    (y * 25) as u8,
                    7,
                    255 - (x % 20 % 3) as u8,
                ]
            }),
            3,
            8,
        ),
        (
            32,
            32,
            rgba(32, 32, |x, y| {
                let v = (x * 8) as u8;
                [v, v, v, (y * 8) as u8]
            }),
            4,
            8,
        ),
        (
            64,
            48,
            rgba(64, 48, |x, y| {
                [(x * 4) as u8, (y * 5) as u8, (x ^ y) as u8, 255]
            }),
            2,
            8,
        ),
        (
            33,
            17,
            (0..33 * 17 * 4).map(|_| rand() as u8).collect(),
            6,
            8,
        ),
        (1, 1, vec![1, 2, 3, 4], 3, 1),
    ];
    for (w, h, pixels, colour, depth) in cases {
        let encoded = png::encode(w, h, &pixels).unwrap();
        assert_eq!((encoded[25], encoded[24]), (colour, depth), "{w}x{h}");
        let decoded = png::decode(&encoded).unwrap();
        assert_eq!((decoded.width, decoded.height), (w, h));
        assert_eq!(decoded.rgba, pixels, "{w}x{h} colour type {colour}");
    }
    // A larger image spans several deflate blocks and long matches, and stays compressed.
    let (w, h) = (300, 200);
    let big = rgba(w, h, |x, y| {
        [(x / 3) as u8, (y / 2) as u8, ((x * y) / 97) as u8, 255]
    });
    let encoded = png::encode(w, h, &big).unwrap();
    assert!(encoded.len() < big.len() / 8, "{} bytes", encoded.len());
    assert_eq!(png::decode(&encoded).unwrap().rgba, big);
    assert!(png::encode(0, 4, &[]).is_err());
    assert!(png::encode(2, 2, &[0; 15]).is_err());
}

#[test]
fn zlib_round_trips_and_bounds_output() {
    let mut rand = noise(3);
    let text: Vec<u8> = b"ganbate engine cooks worlds; "
        .iter()
        .cycle()
        .take(100_000)
        .copied()
        .collect();
    let mixed: Vec<u8> = (0..70_000u32)
        .map(|i| {
            if i % 1000 < 500 {
                rand() as u8
            } else {
                (i / 7) as u8
            }
        })
        .collect();
    for data in [vec![], vec![7], vec![0; 200_000], text, mixed] {
        let z = zlib::compress(&data);
        assert_eq!(zlib::decompress(&z, data.len()).unwrap(), data);
        if !data.is_empty() {
            assert!(zlib::decompress(&z, data.len() - 1).is_err());
        }
    }
    let mut garbage = noise(11);
    for _ in 0..2000 {
        let len = garbage() as usize % 64;
        let mut data: Vec<u8> = (0..len).map(|_| garbage() as u8).collect();
        if data.len() >= 2 {
            data[0] = 0x78;
            data[1] = 0x9c;
        }
        let _ = zlib::decompress(&data, 1 << 16);
    }
}

fn chunk(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = (body.len() as u32).to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    out.extend_from_slice(&png::crc32(&[kind, body]).to_be_bytes());
    out
}
fn png_file(w: u32, h: u32, depth: u8, colour: u8, interlace: u8, idat: &[u8]) -> Vec<u8> {
    let mut ihdr = w.to_be_bytes().to_vec();
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[depth, colour, 0, 0, interlace]);
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    out.extend(chunk(b"IHDR", &ihdr));
    out.extend(chunk(b"IDAT", idat));
    out.extend(chunk(b"IEND", &[]));
    out
}
/// Recompute every chunk CRC, so corruption reaches the decompressor.
fn fix_crcs(data: &mut [u8]) {
    let mut pos = 8;
    while pos + 12 <= data.len() {
        let len = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let Some(end) = pos.checked_add(12 + len).filter(|&e| e <= data.len()) else {
            return;
        };
        let crc = png::crc32(&[&data[pos + 4..end - 4]]);
        data[end - 4..end].copy_from_slice(&crc.to_be_bytes());
        pos = end;
    }
}

#[test]
fn hostile_pngs_fail_without_panicking() {
    let good = png::encode(
        16,
        9,
        &rgba(16, 9, |x, y| [x as u8 * 9, y as u8 * 20, 3, 255]),
    )
    .unwrap();
    assert!(png::decode(&good).is_ok());
    for n in 0..good.len() {
        assert!(png::decode(&good[..n]).is_err(), "prefix {n}");
    }
    let mut extra = good.clone();
    extra.push(0);
    assert!(png::decode(&extra).is_err());
    for at in [20, 30, 40, good.len() - 20] {
        let mut bad = good.clone();
        bad[at] ^= 1;
        assert!(png::decode(&bad).unwrap_err().contains("CRC"), "byte {at}");
    }
    let rows = zlib::compress(&[0; 5 * 5]);
    assert!(png::decode(&png_file(4, 5, 8, 0, 0, &rows)).is_ok());
    for (w, h) in [(100_000, 1), (1, 100_000), (16_384, 16_384), (0, 4)] {
        let e = png::decode(&png_file(w, h, 8, 6, 0, &rows)).unwrap_err();
        assert!(e.contains("outside limits"), "{e}");
    }
    // At the pixel limit, a tiny stream fails fast instead of allocating the image.
    assert!(png::decode(&png_file(16_384, 2_048, 16, 6, 0, &rows)).is_err());
    assert!(
        png::decode(&png_file(4, 5, 8, 0, 1, &rows))
            .unwrap_err()
            .contains("Adam7")
    );
    assert!(png::decode(&png_file(4, 5, 3, 0, 0, &rows)).is_err());
    assert!(
        png::decode(&png_file(4, 5, 8, 3, 0, &rows))
            .unwrap_err()
            .contains("PLTE")
    );
    // Decompression bomb: far more data than the header describes.
    let bomb = zlib::compress(&[0; 1 << 20]);
    assert!(
        png::decode(&png_file(4, 5, 8, 0, 0, &bomb))
            .unwrap_err()
            .contains("expected size")
    );
    let mut filtered = vec![0; 5 * 5];
    filtered[5] = 9;
    assert!(
        png::decode(&png_file(4, 5, 8, 0, 0, &zlib::compress(&filtered)))
            .unwrap_err()
            .contains("filter")
    );
    let mut adler = zlib::compress(&[0; 5 * 5]);
    *adler.last_mut().unwrap() ^= 1;
    assert!(
        png::decode(&png_file(4, 5, 8, 0, 0, &adler))
            .unwrap_err()
            .contains("Adler")
    );
    let mut dictionary = zlib::compress(&[0; 5 * 5]);
    dictionary[1] |= 0x20;
    assert!(png::decode(&png_file(4, 5, 8, 0, 0, &dictionary)).is_err());
    // Mutated streams with valid CRCs exercise the inflater and filters; none may panic.
    let mut rand = noise(5);
    for fixture in [GRADIENT_RGB, PALETTE_4BIT, GREY_16BIT, RGBA_STORED, &good] {
        for _ in 0..400 {
            let mut data = fixture.to_vec();
            for _ in 0..1 + rand() % 3 {
                let at = 8 + rand() as usize % (data.len() - 8);
                data[at] ^= 1 << (rand() % 8);
            }
            fix_crcs(&mut data);
            let _ = png::decode(&data);
        }
    }
}

#[test]
fn split_idat_and_ancillary_chunks_decode() {
    let pixels = rgba(20, 20, |x, y| [x as u8 * 12, y as u8 * 12, 77, 255]);
    let encoded = png::encode(20, 20, &pixels).unwrap();
    assert_eq!(encoded[25], 2);
    // Rebuild with the image data spread over three IDAT chunks and a private ancillary chunk.
    let ihdr_end = 8 + 25;
    let idat_len = u32::from_be_bytes(encoded[ihdr_end..ihdr_end + 4].try_into().unwrap()) as usize;
    let idat = &encoded[ihdr_end + 8..ihdr_end + 8 + idat_len];
    let mut out = encoded[..ihdr_end].to_vec();
    out.extend(chunk(b"teXt", b"Comment\0synthetic"));
    let third = idat.len() / 3;
    for part in [&idat[..third], &idat[third..2 * third], &idat[2 * third..]] {
        out.extend(chunk(b"IDAT", part));
    }
    out.extend(chunk(b"IEND", &[]));
    assert_eq!(png::decode(&out).unwrap().rgba, pixels);
    let mut critical = encoded[..ihdr_end].to_vec();
    critical.extend(chunk(b"XYZW", &[]));
    critical.extend_from_slice(&encoded[ihdr_end..]);
    assert!(png::decode(&critical).unwrap_err().contains("XYZW"));
}

// Mip chains.

#[test]
fn mip_chains_are_complete_and_filtered_in_the_right_space() {
    for (w, h, sizes) in [
        (5, 3, vec![(5, 3), (2, 1), (1, 1)]),
        (6, 1, vec![(6, 1), (3, 1), (1, 1)]),
        (1, 1, vec![(1, 1)]),
        (8, 2, vec![(8, 2), (4, 1), (2, 1), (1, 1)]),
    ] {
        let t = mip::texture(w, h, vec![200; (w * h * 4) as usize], mip::Filter::Srgb).unwrap();
        assert_eq!(t.levels.len(), Texture::level_count(w, h));
        for (level, (lw, lh)) in t.levels.iter().zip(sizes) {
            assert_eq!(level.len(), (lw * lh * 4) as usize);
            // A constant image stays constant at every level.
            assert!(level.iter().all(|&v| v == 200));
        }
    }
    let board = checker(2, 2, [0, 0, 0, 255], [255, 255, 255, 0]);
    let srgb = mip::texture(2, 2, board.clone(), mip::Filter::Srgb).unwrap();
    assert!(srgb.srgb);
    // Linear light: half of white encodes as sRGB 188, not 128. Alpha averages linearly.
    assert_eq!(srgb.levels[1], vec![188, 188, 188, 128]);
    let linear = mip::texture(2, 2, board, mip::Filter::Linear).unwrap();
    assert!(!linear.srgb);
    assert_eq!(linear.levels[1], vec![128, 128, 128, 128]);
    // Normal maps are averaged as vectors and renormalized.
    let normals = checker(2, 2, [255, 128, 128, 255], [128, 128, 255, 255]);
    let n = mip::texture(2, 2, normals, mip::Filter::Normal).unwrap();
    assert_eq!(n.levels[1], vec![218, 128, 218, 255]);
    // Odd sizes fold the leftover texel into the last one instead of dropping it.
    let row = mip::texture(
        3,
        1,
        vec![0, 0, 0, 255, 90, 90, 90, 255, 180, 180, 180, 255],
        mip::Filter::Linear,
    )
    .unwrap();
    assert_eq!(row.levels[1], vec![90, 90, 90, 255]);
    assert!(mip::texture(0, 1, vec![], mip::Filter::Srgb).is_err());
    assert!(mip::texture(2, 2, vec![0; 15], mip::Filter::Srgb).is_err());
    let mesh = Mesh::default();
    assert!(mesh.normals.is_empty() && mesh.material.is_none());
}

// Reference PNGs written by Pillow 1.x-compatible encoders (see the test above).
const GRADIENT_RGB: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 16, 0, 0, 0, 16, 8, 2,
    0, 0, 0, 144, 145, 104, 54, 0, 0, 1, 1, 73, 68, 65, 84, 120, 218, 141, 208, 203, 46, 131, 97,
    24, 133, 209, 69, 63, 122, 82, 245, 106, 171, 165, 45, 170, 122, 112, 26, 136, 72, 136, 129,
    68, 194, 128, 72, 48, 17, 137, 129, 251, 191, 10, 131, 170, 99, 254, 198, 120, 39, 43, 79, 246,
    28, 194, 92, 200, 133, 133, 144, 15, 197, 80, 14, 149, 80, 13, 17, 106, 161, 17, 154, 97, 61,
    180, 67, 55, 137, 28, 243, 36, 22, 201, 83, 160, 72, 137, 50, 75, 84, 88, 166, 202, 10, 193,
    106, 18, 69, 114, 44, 144, 167, 72, 153, 10, 213, 201, 76, 141, 58, 13, 214, 104, 210, 74, 162,
    58, 229, 11, 148, 88, 98, 153, 21, 86, 169, 179, 70, 139, 13, 58, 116, 217, 78, 162, 145, 193,
    215, 104, 208, 100, 157, 54, 93, 250, 244, 146, 104, 255, 143, 31, 179, 205, 78, 18, 189, 127,
    240, 71, 108, 209, 163, 159, 196, 104, 38, 127, 76, 135, 205, 9, 207, 110, 18, 71, 25, 252,
    233, 183, 250, 15, 158, 65, 18, 39, 63, 249, 179, 63, 245, 95, 60, 87, 73, 156, 83, 230, 34,
    251, 156, 9, 127, 195, 128, 81, 18, 151, 51, 207, 217, 228, 118, 202, 15, 25, 39, 113, 157,
    113, 206, 253, 175, 122, 70, 236, 37, 113, 247, 147, 127, 200, 168, 31, 242, 194, 126, 18, 143,
    60, 205, 172, 127, 253, 228, 57, 72, 226, 57, 187, 254, 237, 123, 61, 251, 28, 190, 3, 42, 136,
    27, 245, 175, 205, 154, 191, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];
const PALETTE_4BIT: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 13, 0, 0, 0, 7, 4, 3, 0,
    0, 0, 33, 94, 218, 41, 0, 0, 0, 48, 80, 76, 84, 69, 0, 255, 0, 25, 235, 60, 50, 215, 120, 75,
    195, 180, 100, 175, 240, 125, 155, 44, 150, 135, 104, 175, 115, 164, 200, 95, 224, 225, 75, 28,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 230, 205, 11, 250, 0, 0, 0, 2, 116, 82,
    78, 83, 0, 128, 155, 43, 78, 24, 0, 0, 0, 27, 73, 68, 65, 84, 120, 94, 99, 96, 206, 84, 109,
    116, 103, 78, 96, 136, 16, 41, 48, 155, 20, 33, 192, 64, 36, 31, 0, 109, 77, 13, 21, 9, 151,
    197, 162, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];
const GREY_1BIT: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 19, 0, 0, 0, 5, 1, 0, 0,
    0, 0, 143, 164, 234, 231, 0, 0, 0, 26, 73, 68, 65, 84, 120, 94, 99, 152, 228, 169, 192, 160,
    50, 201, 129, 209, 243, 118, 12, 139, 39, 195, 2, 16, 27, 0, 53, 140, 5, 86, 111, 200, 136,
    157, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];
const GREY_16BIT: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 6, 0, 0, 0, 4, 16, 0, 0,
    0, 0, 216, 255, 205, 220, 0, 0, 0, 38, 73, 68, 65, 84, 120, 94, 99, 100, 96, 80, 214, 128, 65,
    22, 150, 75, 44, 26, 172, 26, 172, 191, 64, 144, 133, 245, 18, 235, 47, 150, 75, 16, 46, 152,
    3, 227, 2, 0, 63, 71, 15, 95, 22, 237, 46, 119, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];
const GREY_ALPHA: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 9, 0, 0, 0, 9, 8, 4, 0,
    0, 0, 74, 152, 206, 155, 0, 0, 0, 23, 73, 68, 65, 84, 120, 156, 99, 100, 248, 47, 195, 128, 10,
    89, 24, 158, 48, 160, 129, 65, 33, 4, 0, 148, 90, 9, 33, 172, 204, 214, 213, 0, 0, 0, 0, 73,
    69, 78, 68, 174, 66, 96, 130,
];
const RGBA_STORED: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 8, 0, 0, 0, 8, 8, 6, 0,
    0, 0, 196, 15, 190, 139, 0, 0, 1, 19, 73, 68, 65, 84, 120, 1, 1, 8, 1, 247, 254, 1, 0, 0, 255,
    128, 32, 0, 240, 0, 32, 0, 240, 0, 32, 0, 240, 0, 32, 0, 240, 0, 32, 0, 240, 0, 32, 0, 240, 0,
    32, 0, 240, 0, 4, 0, 32, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 4, 0, 32, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 32, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 32, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 32, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 32, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 32, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 68, 178, 10, 37, 89, 240, 2, 30, 0, 0,
    0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];
const RGB_FIXED: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 4, 0, 0, 0, 4, 8, 2, 0,
    0, 0, 38, 147, 9, 41, 0, 0, 0, 21, 73, 68, 65, 84, 120, 1, 99, 60, 145, 98, 196, 0, 3, 76, 48,
    6, 8, 224, 230, 0, 0, 69, 246, 1, 102, 53, 251, 6, 99, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96,
    130,
];
const RGBA_16BIT: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 3, 0, 0, 0, 2, 16, 6, 0,
    0, 0, 205, 228, 186, 89, 0, 0, 0, 48, 73, 68, 65, 84, 120, 218, 99, 96, 96, 96, 248, 255, 255,
    127, 3, 131, 159, 2, 195, 255, 127, 255, 154, 95, 204, 113, 96, 248, 255, 247, 111, 251, 5,
    160, 120, 153, 62, 68, 162, 76, 31, 34, 81, 166, 15, 146, 0, 0, 103, 23, 25, 237, 31, 211, 217,
    48, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];
const GREY_2BIT_KEY: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 5, 0, 0, 0, 3, 2, 0, 0,
    0, 0, 52, 237, 130, 133, 0, 0, 0, 2, 116, 82, 78, 83, 0, 2, 152, 157, 172, 20, 0, 0, 0, 17, 73,
    68, 65, 84, 120, 218, 99, 144, 102, 96, 216, 216, 192, 32, 205, 0, 0, 6, 140, 1, 104, 33, 214,
    105, 24, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];
const RGB_KEY: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 4, 0, 0, 0, 2, 8, 2, 0,
    0, 0, 240, 202, 234, 52, 0, 0, 0, 6, 116, 82, 78, 83, 0, 50, 0, 50, 0, 7, 12, 204, 72, 42, 0,
    0, 0, 31, 73, 68, 65, 84, 120, 218, 5, 193, 49, 1, 0, 0, 12, 131, 48, 62, 252, 160, 169, 154,
    166, 119, 9, 96, 56, 60, 132, 44, 151, 151, 15, 34, 194, 3, 89, 200, 37, 16, 6, 0, 0, 0, 0, 73,
    69, 78, 68, 174, 66, 96, 130,
];

/// Version 2 as it was written: f32 everything, u32 indices, textures without a kind.
fn encode_v2(w: &World) -> Vec<u8> {
    let mut v = b"GBEW".to_vec();
    let word = |v: &mut Vec<u8>, n: u32| v.extend_from_slice(&n.to_le_bytes());
    let floats = |v: &mut Vec<u8>, f: &[f32]| {
        for x in f {
            v.extend_from_slice(&x.to_le_bytes());
        }
    };
    word(&mut v, 2);
    for n in [
        w.meshes.len(),
        w.instances.len(),
        w.materials.len(),
        w.textures.len(),
        w.lights.len(),
    ] {
        word(&mut v, n as u32);
    }
    for m in &w.meshes {
        word(&mut v, m.positions.len() as u32);
        word(&mut v, m.indices.len() as u32);
        word(
            &mut v,
            u32::from(!m.normals.is_empty()) | u32::from(!m.uvs.is_empty()) << 1,
        );
        floats(&mut v, &m.color);
        word(&mut v, m.material.unwrap_or(u32::MAX));
        for p in m.positions.iter().chain(&m.normals) {
            floats(&mut v, &[p.0, p.1, p.2]);
        }
        for uv in &m.uvs {
            floats(&mut v, uv);
        }
        for &i in &m.indices {
            word(&mut v, i);
        }
    }
    for i in &w.instances {
        word(&mut v, i.mesh);
        floats(&mut v, &i.transform.0);
    }
    for m in &w.materials {
        word(&mut v, m.name.len() as u32);
        v.extend_from_slice(m.name.as_bytes());
        v.resize(v.len().next_multiple_of(4), 0);
        floats(&mut v, &m.base_color);
        word(&mut v, m.base_color_texture.unwrap_or(u32::MAX));
        word(&mut v, m.normal_texture.unwrap_or(u32::MAX));
        floats(&mut v, &[m.normal_scale]);
        floats(&mut v, &m.emissive);
        word(&mut v, m.emissive_texture.unwrap_or(u32::MAX));
        floats(&mut v, &[m.metallic, m.roughness]);
        let (mode, cutoff) = match m.alpha {
            AlphaMode::Opaque => (0, 0.),
            AlphaMode::Mask(c) => (1, c),
            AlphaMode::Blend => (2, 0.),
        };
        word(&mut v, mode);
        floats(&mut v, &[cutoff]);
        word(&mut v, u32::from(m.double_sided));
        word(&mut v, m.shading as u32);
        word(&mut v, m.emission as u32);
    }
    for t in &w.textures {
        word(&mut v, t.width);
        word(&mut v, t.height);
        word(&mut v, u32::from(t.srgb));
        word(&mut v, t.levels.len() as u32);
        for l in &t.levels {
            v.extend_from_slice(l);
        }
    }
    for l in &w.lights {
        floats(&mut v, &[l.position.0, l.position.1, l.position.2]);
        floats(&mut v, &l.color);
        floats(&mut v, &[l.intensity, l.range]);
    }
    v
}

#[test]
fn version_2_files_still_decode() {
    let w = glb::import(&scene_glb("")).unwrap();
    let v2 = encode_v2(&w);
    assert_eq!(cooked::version(&v2), Some(2));
    assert_eq!(cooked::decode(&v2).unwrap(), w);
    for n in (0..v2.len()).step_by(5) {
        assert!(cooked::decode(&v2[..n]).is_err());
    }
}

#[test]
fn compact_worlds_code_meshes_within_tolerance_and_keep_encoded_images() {
    let mut w = glb::import(&scene_glb("")).unwrap();
    // A large mesh (a sea 1 km across, far off the origin) and a vertex no triangle uses.
    let mut sea = w.meshes[0].clone();
    for p in &mut sea.positions {
        p.0 = p.0 * 400.0 + 1237.25;
    }
    sea.positions.push(Vec3(9., 9., 9.));
    if !sea.normals.is_empty() {
        sea.normals.push(Vec3(0., 1., 0.));
    }
    if !sea.uvs.is_empty() {
        sea.uvs.push([0.5, 0.5]);
    }
    w.meshes.push(sea);
    w.instances.push(ganbate_core::world::Instance {
        mesh: 2,
        transform: ganbate_core::world::Transform::IDENTITY,
    });
    // An encoded image stays as it is.
    let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR".to_vec();
    png.extend_from_slice(&4u32.to_be_bytes());
    png.extend_from_slice(&4u32.to_be_bytes());
    png.extend_from_slice(b"rest of the file");
    w.textures[0].levels.clear();
    w.textures[0].encoded = Some(ganbate_core::world::EncodedImage {
        mime: "image/png".into(),
        data: png.clone(),
    });
    let exact = cooked::encode(&w).unwrap();
    assert_eq!(cooked::decode(&exact).unwrap(), w);
    let compact = cooked::encode_compact(&w).unwrap();
    assert!(
        compact.len() < exact.len(),
        "{} vs {}",
        compact.len(),
        exact.len()
    );
    let c = cooked::decode(&compact).unwrap();
    assert_eq!(c.textures, w.textures);
    assert_eq!(
        (c.materials.clone(), c.lights.clone(), c.instances.clone()),
        (w.materials.clone(), w.lights.clone(), w.instances.clone())
    );
    for (a, b) in w.meshes.iter().zip(&c.meshes) {
        assert_eq!(a.indices.len(), b.indices.len());
        assert_eq!((a.material, a.color), (b.material, b.color));
        // The unused vertex is gone; every corner keeps its attributes within tolerance.
        let used: std::collections::BTreeSet<u32> = a.indices.iter().copied().collect();
        assert_eq!(b.positions.len(), used.len());
        for (&i, &j) in a.indices.iter().zip(&b.indices) {
            let (p, q) = (a.positions[i as usize], b.positions[j as usize]);
            for (x, y) in [(p.0, q.0), (p.1, q.1), (p.2, q.2)] {
                assert!((x - y).abs() <= 0.00025 + x.abs() * 1e-6, "{x} vs {y}");
            }
            if let (Some(n), Some(m)) = (a.normals.get(i as usize), b.normals.get(j as usize)) {
                let dot = n.0 * m.0 + n.1 * m.1 + n.2 * m.2;
                assert!(dot > 0.99999, "{n:?} vs {m:?}");
            }
            if let (Some(u), Some(v)) = (a.uvs.get(i as usize), b.uvs.get(j as usize)) {
                assert!(
                    (u[0] - v[0]).abs() <= 1.0 / 16384.0 && (u[1] - v[1]).abs() <= 1.0 / 16384.0
                );
            }
        }
    }
    // Coding again gives the same bytes.
    assert_eq!(cooked::encode_compact(&c).unwrap(), compact);
    for n in (0..compact.len()).step_by(3) {
        assert!(cooked::decode(&compact[..n]).is_err());
    }
    // A corrupted stream is an error, never a panic.
    for at in (48..compact.len()).step_by(11) {
        let mut bad = compact.clone();
        bad[at] ^= 0x5A;
        let _ = cooked::decode(&bad);
    }
}

#[test]
fn octahedral_normals_survive_every_direction() {
    let mut w = World::default();
    let mut m = Mesh {
        color: [0.5; 3],
        ..Mesh::default()
    };
    // Directions over the whole sphere, including the axes and the octahedron's seams.
    for i in 0..400 {
        let t = i as f32 / 400.0;
        let z = 1.0 - 2.0 * t;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let phi = i as f32 * 2.399_963;
        m.normals.push(Vec3(r * phi.cos(), r * phi.sin(), z));
    }
    for axis in [
        Vec3(1., 0., 0.),
        Vec3(-1., 0., 0.),
        Vec3(0., 1., 0.),
        Vec3(0., -1., 0.),
        Vec3(0., 0., 1.),
        Vec3(0., 0., -1.),
        Vec3(0.6, 0., -0.8),
    ] {
        m.normals.push(axis);
    }
    m.positions = (0..m.normals.len())
        .map(|i| Vec3(i as f32 * 0.01, 0., 0.))
        .collect();
    m.indices = (0..(m.normals.len() as u32 / 3) * 3).collect();
    w.meshes.push(m);
    w.instances.push(ganbate_core::world::Instance {
        mesh: 0,
        transform: ganbate_core::world::Transform::IDENTITY,
    });
    let back = cooked::decode(&cooked::encode_compact(&w).unwrap()).unwrap();
    let (a, b) = (&w.meshes[0], &back.meshes[0]);
    for (&i, &j) in a.indices.iter().zip(&b.indices) {
        let (n, m) = (a.normals[i as usize], b.normals[j as usize]);
        let dot = n.0 * m.0 + n.1 * m.1 + n.2 * m.2;
        assert!(dot > 0.99999, "{n:?} vs {m:?}");
    }
}

const NAMED: &str = r#"{"asset":{"version":"2.0"},"buffers":[{"byteLength":54}],"bufferViews":[{"buffer":0,"byteLength":48,"byteStride":16},{"buffer":0,"byteOffset":48,"byteLength":6}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],"extensions":{"KHR_lights_punctual":{"lights":[{"type":"point","intensity":3,"range":5}]}},"nodes":[{"name":"Stand","mesh":0,"translation":[1,0,0],"children":[1,2]},{"name":"Top","mesh":0,"translation":[0,2,0]},{"name":"Lamp","translation":[0,3,0],"extensions":{"KHR_lights_punctual":{"light":0}}}],"scenes":[{"nodes":[0]}],"scene":0}"#;

#[test]
fn node_import_keeps_names_hierarchy_and_local_transforms() {
    let n = glb::import_nodes(&glb_file(NAMED, &bytes()), false).unwrap();
    assert_eq!(n.world.meshes.len(), 1);
    assert!(n.world.instances.is_empty() && n.world.lights.is_empty());
    let names: Vec<&str> = n.nodes.iter().map(|x| x.name.as_str()).collect();
    assert_eq!(names, ["Stand", "Top", "Lamp"]);
    assert_eq!(n.nodes[1].parent, Some(0));
    assert_eq!(n.nodes[1].local.0[13], 2.0);
    assert_eq!(n.nodes[0].meshes, 0..1);
    assert_eq!(n.nodes[2].meshes, 0..0);
    let lamp = n.nodes[2].light.unwrap();
    assert_eq!((lamp.intensity, lamp.range), (3.0, 5.0));
    // The flattening import of the same file places the same geometry.
    let w = glb::import(&glb_file(NAMED, &bytes())).unwrap();
    assert_eq!(w.instances.len(), 2);
    assert_eq!(w.lights[0].position, Vec3(1., 3., 0.));
}

#[test]
fn node_import_refuses_matrices_that_shear() {
    let with_matrix = |m: &str| {
        glb_file(
            &NAMED.replace(r#""translation":[1,0,0]"#, &format!(r#""matrix":{m}"#)),
            &bytes(),
        )
    };
    // x leans into y: no translation, rotation and scale make this.
    let sheared = with_matrix("[1,0,0,0,0.5,1,0,0,0,0,1,0,0,0,0,1]");
    let err = glb::import_nodes(&sheared, false).unwrap_err();
    assert!(err.contains("Stand") && err.contains("shear"), "{err}");
    // The flattening import keeps the matrix as it is.
    assert_eq!(
        glb::import(&sheared).unwrap().instances[0].transform.0[4],
        0.5
    );
    // Matrices that split into translation, rotation and scale import: turned and scaled,
    // mirrored, scaled to nothing along an axis, and with rounding noise.
    for m in [
        "[2,0,0,0,0,0,3,0,0,-1,0,0,4,5,6,1]",
        "[-1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]",
        "[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1]",
        "[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]",
        "[1,0.0000001,0,0,0,1,0.0000002,0,0.0000001,0,1,0,0,0,0,1]",
        "[1,0,0,0,0,0.0000001,0.0000002,0,0,0,1,0,0,0,0,1]",
    ] {
        let n = glb::import_nodes(&with_matrix(m), false).unwrap();
        assert_eq!(n.nodes.len(), 3, "{m}");
    }
}

#[test]
fn scenes_cook_with_their_entity_table() {
    let w = glb::import(&glb_file(NAMED, &bytes())).unwrap();
    let t = |x: f32| {
        let mut m = Transform::IDENTITY;
        m.0[12] = x;
        m
    };
    let table = EntityTable {
        entities: vec![
            EntityRecord {
                id: 0x0123_4567_89ab_cdef,
                name: "Stand".into(),
                parent: None,
                transform: t(1.0),
                instances: 0..1,
                lights: 0..0,
            },
            EntityRecord {
                id: 0xfedc_ba98_7654_3210,
                name: "Top 桜".into(),
                parent: Some(0),
                transform: t(1.0),
                instances: 1..2,
                lights: 0..1,
            },
        ],
        data: "{\"camera\": 1}\n".into(),
    };
    for compact in [false, true] {
        let b = cooked::encode_scene(&w, &table, compact).unwrap();
        assert_eq!(cooked::version(&b), Some(4));
        let (w2, t2) = cooked::decode_scene(&b).unwrap();
        assert_eq!(t2, table);
        assert_eq!(w2.instances, w.instances);
        // A version 3 reader's view: `decode` gives the same world.
        assert_eq!(cooked::decode(&b).unwrap(), w2);
        assert_eq!(cooked::encode_scene(&w2, &t2, compact).unwrap(), b);
        for n in [0, b.len() / 2, b.len() - 1] {
            assert!(cooked::decode_scene(&b[..n]).is_err());
        }
    }
    // Version 3 files have an empty table.
    let (_, empty) = cooked::decode_scene(&cooked::encode(&w).unwrap()).unwrap();
    assert_eq!(empty, EntityTable::default());
    // Bad tables are refused.
    let mut bad = table.clone();
    bad.entities[1].instances = 1..3;
    assert!(cooked::encode_scene(&w, &bad, false).is_err());
    let mut bad = table.clone();
    bad.entities[0].parent = Some(1);
    assert!(cooked::encode_scene(&w, &bad, false).is_err());
    let mut bad = table;
    bad.entities[1].id = bad.entities[0].id;
    assert!(cooked::encode_scene(&w, &bad, false).is_err());
}

#[test]
fn affine_inverses_undo_their_transforms() {
    let m = Transform::from_trs([1.0, -2.0, 3.0], [0.2, 0.5, -0.1, 0.83], [2.0, 0.5, 1.5]).unwrap();
    let inv = m.inverse().unwrap();
    let p = Vec3(0.3, 4.0, -7.0);
    let back = inv.point(m.point(p));
    for (a, b) in [(back.0, p.0), (back.1, p.1), (back.2, p.2)] {
        assert!((a - b).abs() < 1e-4, "{back:?} vs {p:?}");
    }
    let flat = Transform::from_trs([0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0]).unwrap();
    assert!(flat.inverse().is_none());
}
