use super::cache::*;
use super::*;

/// A batch's instances in view are drawn in runs, at most `MAX_INSTANCE_RUNS`; more runs, or
/// no frustum, draw the batch whole, and none in view draws nothing.
#[test]
fn instance_runs_draw_what_is_in_view_in_few_runs() {
    // Looking down -z from the origin: an instance 5 m ahead ('x') is in view, one 50 m
    // aside ('.') is not. The batch starts at instance 1.
    let view = math::mul(
        &math::perspective(1.0, 1.0, 0.1),
        &math::look_at([0.0; 3], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
    );
    let seen = Frustum::from_view_proj(&view);
    let at = |x: f32| ([x - 0.1, -0.1, -5.1], [x + 0.1, 0.1, -4.9]);
    let placed = |pattern: &str| -> Vec<(V3, V3)> {
        let x = |c| if c == 'x' { 0.0 } else { 50.0 };
        std::iter::once(at(50.0))
            .chain(pattern.chars().map(|c| at(x(c))))
            .collect()
    };
    let runs = |pattern: &str, within| {
        let (runs, n) = Renderer::instance_runs(&placed(pattern), 1, pattern.len() as u32, within);
        runs[..n].to_vec()
    };
    assert_eq!(runs("x.xx..x", None), [(1, 7)]);
    assert_eq!(runs("x.xx..x", Some(&seen)), [(1, 1), (3, 2), (7, 1)]);
    assert_eq!(
        runs("xx.x.x.x", Some(&seen)),
        [(1, 2), (4, 1), (6, 1), (8, 1)]
    );
    assert_eq!(runs("xx.x.x.x.x", Some(&seen)), [(1, 10)]);
    assert_eq!(runs("....", Some(&seen)), []);
}

#[test]
fn material_from_json() {
    let d = MaterialDesc::from_json(
        br#"{"name": "stone", "color": [0.1, 0.3, 0.2, 1], "roughness": 2,
             "shading": "unlit", "alphaCutoff": 0.5, "baseTexture": 7,
             "castsShadow": false, "lamp": true, "emissive": [1, 0.5, 0],
             "surface": "clearcoat", "clearcoat": 0.8, "clearcoatRoughness": 0.1}"#,
    )
    .unwrap();
    assert_eq!(d.name, "stone");
    assert_eq!(d.surface, Some(SurfaceModel::ClearCoat));
    assert_eq!((d.clearcoat, d.clearcoat_roughness), (Some(0.8), Some(0.1)));
    assert_eq!(d.base_color, [0.1, 0.3, 0.2, 1.0]);
    assert_eq!((d.roughness, d.shading), (1.0, Shading::Unlit));
    assert_eq!(
        (d.alpha_cutoff, d.base_texture),
        (Some(0.5), Some(PropTexture(7)))
    );
    assert!(!d.casts_shadow && d.lamp && d.emissive == [1.0, 0.5, 0.0]);
    assert_eq!(
        MaterialDesc::from_json(b"{}").unwrap(),
        MaterialDesc::default()
    );
    assert_eq!(
        MaterialDesc::from_json(br#"{"order": 44.5}"#)
            .unwrap()
            .order,
        44.5
    );
    let solid = MaterialDesc::from_json(br#"{"overlay": true, "depthWrite": true}"#).unwrap();
    assert!(solid.overlay && solid.depth_test && solid.depth_write);
    assert_eq!(depth_index(&solid), 2);
    for bad in [
        &b"[]"[..],
        br#"{"colour": [1, 1, 1, 1]}"#,
        br#"{"color": [1, 1, 1]}"#,
        br#"{"shading": "toon"}"#,
        br#"{"baseTexture": 0}"#,
        br#"{"castsShadow": 1}"#,
    ] {
        assert!(
            MaterialDesc::from_json(bad).is_err(),
            "{}",
            String::from_utf8_lossy(bad)
        );
    }
}

fn dummy_material_entry(desc: MaterialDesc) -> MaterialEntry {
    MaterialEntry {
        desc,
        data: MaterialData {
            base_color: [1.0; 4],
            emissive: [0.0; 4],
            params: [0.0; 4],
            flags: [0.0; 4],
            base: None,
            normal: None,
            emission: None,
        },
        group: BindGroupId::from_raw(1),
        shadow_group: None,
        detail: [0.0; 16],
        revision: 0,
    }
}

fn z_quad_mesh_entry(half: f32) -> MeshEntry {
    let positions = vec![
        [-half, -half, 0.0],
        [half, -half, 0.0],
        [half, half, 0.0],
        [-half, half, 0.0],
    ];
    let (min, max) = bounds(&positions);
    MeshEntry {
        vertices: BufferId::from_raw(1),
        indices: BufferId::from_raw(2),
        index_count: 6,
        positions,
        uvs: vec![[0.0, 0.0]; 4],
        triangles: vec![0, 1, 2, 0, 2, 3],
        min,
        max,
        skin: None,
        revision: 0,
        joint_boxes: vec![],
        shape: 0,
        drift: 0.0,
        morph: None,
    }
}

fn translation_matrix(x: f32, y: f32, z: f32) -> M4 {
    let mut m = math::IDENTITY;
    m[12] = x;
    m[13] = y;
    m[14] = z;
    m
}

#[test]
fn posed_corners_follow_the_joints_of_a_skinned_prop() {
    // Two vertices, one on each joint; joint 1 lifts its vertex 3 m.
    let mut mesh = z_quad_mesh_entry(0.5);
    mesh.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    mesh.skin = Some(SkinRange {
        first: 0,
        count: 4,
        max_joint: 1,
    });
    let records = [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ];
    mesh.joint_boxes = joint_boxes(&mesh.positions, &records);
    let model = translation_matrix(10.0, 0.0, 0.0);
    let inst = InstanceEntry {
        model,
        normal: normal_matrix(&model),
        ..instance_entry(PropMesh::from_raw(1), PropMaterial::from_raw(1))
    };
    let joints = [math::IDENTITY, translation_matrix(0.0, 3.0, 0.0)];
    let mut posed = vec![];
    assert_eq!(posed_corners(&inst, &joints, &mesh, &mut posed), 1.0);
    assert_eq!(bounds(&posed), ([10.0, 0.0, 0.0], [11.0, 3.0, 0.0]));
}

fn instance_entry(mesh: PropMesh, material: PropMaterial) -> InstanceEntry {
    InstanceEntry {
        mesh,
        material,
        model: math::IDENTITY,
        normal: math::IDENTITY,
        tint: [1.0; 4],
        visible: true,
        pick: 0,
        joints: None,
        dirty: true,
        stamp: [0; 4],
        sig: 0,
        still: 0,
        world_bounds: None,
        modelled: ([0.0; 3], [0.0; 3]),
        cached: None,
        shadow: ShadowTrack::default(),
    }
}

/// An instance to change, marked as set (as the renderer's setters mark it).
fn edit(props: &mut Props, id: u32) -> &mut InstanceEntry {
    let i = props.instances.get_mut(id).unwrap();
    i.dirty = true;
    i
}

#[test]
fn shadow_maps_are_drawn_again_once_a_caster_drifted_far_enough() {
    // Straight down on the origin, 4 m across in 1024 texels: 256 texels a metre.
    let down = math::look_at([0.0, 5.0, 0.0], [0.0; 3], [0.0, 0.0, -1.0]);
    let map: ShadowMap = (
        math::mul(&math::ortho(-2.0, 2.0, -2.0, 2.0, 0.1, 10.0), &down),
        1024,
    );
    let texel = 1.0 / 256.0;
    let unit = ([-0.5; 3], [0.5; 3]);
    assert!((texels_per_metre(&map, unit).unwrap() - 256.0).abs() < 1e-2);
    assert_eq!(
        texels_per_metre(&map, ([3.0, 0.0, 0.0], [4.0, 1.0, 1.0])),
        None
    );
    // A spot: ahead of it, the nearer the finer; behind it, outside; reaching it, unbounded.
    let spot = (
        math::mul(
            &math::perspective_depth(1.0, 1.0, 0.1, 20.0),
            &math::look_at([0.0, 5.0, 0.0], [0.0; 3], [0.0, 0.0, -1.0]),
        ),
        1024,
    );
    let near = texels_per_metre(&spot, ([-0.1, 3.0, -0.1], [0.1, 3.1, 0.1])).unwrap();
    let far = texels_per_metre(&spot, ([-0.1, 0.0, -0.1], [0.1, 0.1, 0.1])).unwrap();
    assert!(near > far && far > 0.0);
    assert_eq!(
        texels_per_metre(&spot, ([-0.1, 6.0, -0.1], [0.1, 7.0, 0.1])),
        None
    );
    assert_eq!(
        texels_per_metre(&spot, ([-0.1, 4.0, -0.1], [0.1, 6.0, 0.1])),
        Some(f32::INFINITY)
    );

    // A skinned quad (its lower edge on joint 0, its upper on joint 1) casting, and an opaque
    // prop that casts none.
    let mut props = Props::default();
    let mut quad = z_quad_mesh_entry(0.5);
    let on = |j: f32| [[j, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
    let records = [on(0.0), on(0.0), on(1.0), on(1.0)].concat();
    quad.joint_boxes = joint_boxes(&quad.positions, &records);
    quad.skin = Some(SkinRange {
        first: 0,
        count: 8,
        max_joint: 1,
    });
    let quad = PropMesh(props.meshes.insert(quad).unwrap());
    let plain = PropMesh(props.meshes.insert(z_quad_mesh_entry(0.5)).unwrap());
    let caster = PropMaterial(
        props
            .materials
            .insert(dummy_material_entry(MaterialDesc::default()))
            .unwrap(),
    );
    let shadeless = PropMaterial(
        props
            .materials
            .insert(dummy_material_entry(MaterialDesc {
                casts_shadow: false,
                ..MaterialDesc::default()
            }))
            .unwrap(),
    );
    let figure = props
        .instances
        .insert(instance_entry(quad, caster))
        .unwrap();
    props.share_joints(&[figure], &[math::IDENTITY; 2]).unwrap();
    let pillar = props
        .instances
        .insert(instance_entry(plain, shadeless))
        .unwrap();
    let eye = math::mul(
        &math::perspective(1.0, 1.0, 0.1),
        &math::look_at([0.0, 1.0, 5.0], [0.0; 3], [0.0, 1.0, 0.0]),
    );
    let view = CacheView {
        view_proj: &eye,
        size: (64, 64),
    };
    // Measured against half a texel, so a drift can be told from a move.
    let frame = |props: &mut Props| {
        props.pose_instances();
        track_cached_props(props, false, &view, &[map], 0.5)
    };
    let lift = |props: &mut Props, texels: f32| {
        let joints = [math::IDENTITY, translation_matrix(0.0, texels * texel, 0.0)];
        props.share_joints(&[figure], &joints).unwrap();
    };

    // A caster appearing counts; the maps then hold it.
    assert!(frame(&mut props).casters_moved);
    commit_prop_shadows(&mut props);
    let still = frame(&mut props);
    assert!(!still.casters_moved && still.moved.is_empty());

    // Its upper edge rises 0.3 texel a frame: measured from where the maps hold it, the
    // second frame's rise counts.
    lift(&mut props, 0.3);
    let small = frame(&mut props);
    assert!(!small.casters_moved && small.moved.len() == 1);
    props.pose_instances();
    let exact = track_cached_props(&mut props, false, &view, &[map], SHADOW_DRIFT_TEXELS);
    assert!(
        exact.casters_moved,
        "any drift counts as the renderer measures it"
    );
    lift(&mut props, 0.6);
    assert!(frame(&mut props).casters_moved);
    commit_prop_shadows(&mut props);
    lift(&mut props, 0.9);
    assert!(!frame(&mut props).casters_moved);
    commit_prop_shadows(&mut props);

    // A prop that casts no shadow moves freely.
    edit(&mut props, pillar).model = translation_matrix(1.0, 0.0, 0.0);
    let shadeless_move = frame(&mut props);
    assert!(!shadeless_move.casters_moved && shadeless_move.moved.is_empty());

    // Its mesh morphed 0.6 texel since the maps were drawn counts, and only until then.
    props.meshes.get_mut(quad.0).unwrap().drift = 0.6 * texel;
    assert!(frame(&mut props).casters_moved);
    commit_prop_shadows(&mut props);
    assert!(frame(&mut props).moved.is_empty());

    // Hidden, shown and destroyed, it counts each time; while the whole picture is drawn
    // again too.
    edit(&mut props, figure).visible = false;
    assert!(frame(&mut props).casters_moved);
    commit_prop_shadows(&mut props);
    assert!(!frame(&mut props).casters_moved);
    edit(&mut props, figure).visible = true;
    props.pose_instances();
    assert!(track_cached_props(&mut props, true, &view, &[map], 0.5).casters_moved);
    commit_prop_shadows(&mut props);
    props.remove_instance(figure);
    let gone = frame(&mut props);
    assert!(gone.casters_moved && gone.moved.len() == 1);
}

#[test]
fn pick_along_ray_normalizes_filters_layers_and_rejects_invalid_rays() {
    let mut props = Props::default();
    let mesh = PropMesh(props.meshes.insert(z_quad_mesh_entry(1.0)).unwrap());
    let opaque_mat = PropMaterial(
        props
            .materials
            .insert(dummy_material_entry(MaterialDesc {
                name: "panel".into(),
                ..MaterialDesc::default()
            }))
            .unwrap(),
    );
    let overlay_mat = PropMaterial(
        props
            .materials
            .insert(dummy_material_entry(MaterialDesc {
                name: "reticle".into(),
                overlay: true,
                depth_test: false,
                order: 2.0,
                ..MaterialDesc::default()
            }))
            .unwrap(),
    );

    let near_model = translation_matrix(0.0, 0.0, -2.5);
    let near_inst = PropInstance(
        props
            .instances
            .insert(InstanceEntry {
                model: near_model,
                normal: normal_matrix(&near_model),
                pick: 0b0001,
                ..instance_entry(mesh, opaque_mat)
            })
            .unwrap(),
    );

    let far_model = translation_matrix(0.0, 0.0, -6.0);
    let far_inst = PropInstance(
        props
            .instances
            .insert(InstanceEntry {
                model: far_model,
                normal: normal_matrix(&far_model),
                pick: 0b0010,
                ..instance_entry(mesh, opaque_mat)
            })
            .unwrap(),
    );

    // Unnormalized ray direction `[0, 0, -5]` must normalize cleanly and report world-metre distance `5.5`.
    let hit = props
        .pick_along_ray([0.25, -0.5, 3.0], [0.0, 0.0, -5.0], u32::MAX)
        .expect("hit near prop");
    assert_eq!(hit.instance, near_inst);
    assert!(hit.distance.is_finite() && (hit.distance - 5.5).abs() < 1e-5);
    assert!(hit.point.iter().all(|v| v.is_finite()));
    assert!((hit.point[0] - 0.25).abs() < 1e-5);
    assert!((hit.point[1] + 0.5).abs() < 1e-5);
    assert!((hit.point[2] + 2.5).abs() < 1e-5);

    // Layer mask filtering selects only matching layer bits and returns `None` on disjoint/zero masks.
    let layer2_hit = props
        .pick_along_ray([0.0, 0.0, 3.0], [0.0, 0.0, -1.0], 0b0010)
        .expect("hit layer 2 prop");
    assert_eq!(layer2_hit.instance, far_inst);
    assert!((layer2_hit.distance - 9.0).abs() < 1e-5);
    assert!(
        props
            .pick_along_ray([0.0, 0.0, 3.0], [0.0, 0.0, -1.0], 0b0100)
            .is_none()
    );
    assert!(
        props
            .pick_along_ray([0.0, 0.0, 3.0], [0.0, 0.0, -1.0], 0)
            .is_none()
    );

    // Overlay drawn on top (`overlay && !depth_test`) wins even when placed behind an opaque prop.
    let overlay_model = translation_matrix(0.0, 0.0, -8.0);
    let overlay_inst = PropInstance(
        props
            .instances
            .insert(InstanceEntry {
                model: overlay_model,
                normal: normal_matrix(&overlay_model),
                pick: 0b0100,
                ..instance_entry(mesh, overlay_mat)
            })
            .unwrap(),
    );
    let top_hit = props
        .pick_along_ray([0.0, 0.0, 3.0], [0.0, 0.0, -1.0], u32::MAX)
        .expect("overlay wins");
    assert_eq!(top_hit.instance, overlay_inst);
    assert!((top_hit.distance - 11.0).abs() < 1e-5);

    // NaN, Inf, and zero / near-zero (`len_sq < 1e-12`) rays must all return `None`.
    for (bad_origin, bad_dir) in [
        ([f32::NAN, 0.0, 3.0], [0.0, 0.0, -1.0]),
        ([0.0, f32::INFINITY, 3.0], [0.0, 0.0, -1.0]),
        ([0.0, 0.0, f32::NEG_INFINITY], [0.0, 0.0, -1.0]),
        ([0.0, 0.0, 3.0], [f32::NAN, 0.0, -1.0]),
        ([0.0, 0.0, 3.0], [0.0, f32::INFINITY, -1.0]),
        ([0.0, 0.0, 3.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, 3.0], [1e-7, -1e-7, 1e-7]),
    ] {
        assert!(
            props
                .pick_along_ray(bad_origin, bad_dir, u32::MAX)
                .is_none(),
            "expected None for origin={bad_origin:?}, dir={bad_dir:?}"
        );
    }
}

#[test]
fn renderer_pick_ray_and_world_ray_delegate_cleanly() {
    use crate::render::{Look, Output, Quality};
    use crate::world::{AlphaMode, Instance, Material, Mesh, Transform, World};
    use ganbate_core::Vec3;

    let Ok(mut device) = Device::headless() else {
        return;
    };

    let mut world = World::default();
    world.materials.push(Material {
        name: "wall".into(),
        base_color: [0.6, 0.6, 0.65, 1.0],
        roughness: 0.7,
        metallic: 0.0,
        alpha: AlphaMode::Opaque,
        shading: Shading::Standard,
        ..Material::default()
    });
    world.meshes.push(Mesh {
        positions: vec![
            Vec3(-2.0, -2.0, 0.0),
            Vec3(2.0, -2.0, 0.0),
            Vec3(2.0, 2.0, 0.0),
            Vec3(-2.0, 2.0, 0.0),
        ],
        normals: vec![Vec3(0.0, 0.0, 1.0); 4],
        uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        color: [1.0, 1.0, 1.0],
        indices: vec![0, 1, 2, 0, 2, 3],
        material: Some(0),
    });
    world.instances.push(Instance {
        mesh: 0,
        transform: Transform(translation_matrix(0.0, 0.0, -4.0)),
    });

    let mut renderer = Renderer::new(
        &mut device,
        &world,
        Look::default(),
        Quality::low_cost(),
        (100, 100),
        Output::Still,
    )
    .expect("headless renderer");

    let prop_mesh = renderer
        .create_prop_mesh(
            &mut device,
            &MeshData {
                positions: &[
                    [-0.5, -0.5, 0.0],
                    [0.5, -0.5, 0.0],
                    [0.5, 0.5, 0.0],
                    [-0.5, 0.5, 0.0],
                ],
                normals: &[[0.0, 0.0, 1.0]; 4],
                uvs: &[[0.0, 0.0]; 4],
                indices: &[0, 1, 2, 0, 2, 3],
            },
        )
        .expect("prop mesh");
    let prop_mat = renderer
        .create_prop_material(&mut device, &MaterialDesc::default())
        .expect("prop material");
    let prop_inst = renderer
        .create_prop_instance(prop_mesh, prop_mat, &translation_matrix(0.0, 0.0, -1.5))
        .expect("prop instance");
    renderer
        .set_prop_instance_pick(prop_inst, 0b0010)
        .expect("set pick mask");

    // World ray picking with unnormalized ray hits cooked instance 0 at z = -4 (distance 7.0 from z = 3).
    let world_hit = renderer
        .pick_world_along_ray([0.5, -0.25, 3.0], [0.0, 0.0, -3.0])
        .expect("world ray hit");
    assert_eq!(world_hit.instance, 0);
    assert!(world_hit.distance.is_finite() && (world_hit.distance - 7.0).abs() < 1e-5);
    assert!((world_hit.point[0] - 0.5).abs() < 1e-5);
    assert!((world_hit.point[1] + 0.25).abs() < 1e-5);
    assert!((world_hit.point[2] + 4.0).abs() < 1e-5);

    // Prop ray picking respects layer mask and matches centre-screen camera `pick`.
    assert!(
        renderer
            .pick_along_ray([0.0, 0.0, 3.0], [0.0, 0.0, -2.0], 0b0001)
            .is_none()
    );
    let ray_prop = renderer
        .pick_along_ray([0.0, 0.0, 3.0], [0.0, 0.0, -2.0], 0b0010)
        .expect("prop ray hit");
    assert_eq!(ray_prop.instance, prop_inst);
    assert!((ray_prop.distance - 4.5).abs() < 1e-5);

    let view = View {
        name: "centre".into(),
        eye: [0.0, 0.0, 3.0],
        target: [0.0, 0.0, 0.0],
        fov: 50.0,
        ..View::default()
    };
    let screen_prop = renderer
        .pick(&view, (100, 100), (50.0, 50.0), 0b0010)
        .expect("screen prop hit");
    assert_eq!(screen_prop.instance, ray_prop.instance);
    assert!((screen_prop.distance - ray_prop.distance).abs() < 1e-4);

    let screen_world = renderer
        .pick_world(&view, (100, 100), (50.0, 50.0))
        .expect("screen world hit");
    assert_eq!(screen_world.instance, 0);
    assert!((screen_world.distance - 7.0).abs() < 1e-4);

    // Degenerate / non-finite rays return None on both Renderer ray methods.
    for bad_dir in [
        [0.0, 0.0, 0.0],
        [f32::NAN, 0.0, -1.0],
        [0.0, f32::INFINITY, -1.0],
    ] {
        assert!(
            renderer
                .pick_along_ray([0.0, 0.0, 3.0], bad_dir, u32::MAX)
                .is_none()
        );
        assert!(
            renderer
                .pick_world_along_ray([0.0, 0.0, 3.0], bad_dir)
                .is_none()
        );
    }
    assert!(
        renderer
            .pick_world_along_ray([f32::NEG_INFINITY, 0.0, 3.0], [0.0, 0.0, -1.0])
            .is_none()
    );

    renderer.destroy_props(&mut device);
}

#[test]
fn canvas_clear_rounds_to_canvas_bytes_and_decodes_srgb() {
    assert_eq!(
        canvas_clear([0.2, 0.4, 0.8, 0.6], false),
        Ok([51.0 / 255.0, 102.0 / 255.0, 204.0 / 255.0, 153.0 / 255.0])
    );
    let srgb = canvas_clear([0.2, 0.4, 0.8, 0.6], true).unwrap();
    assert_eq!(srgb[..3], [51, 102, 204].map(math::srgb_to_linear));
    assert_eq!(srgb[3], 153.0 / 255.0, "alpha is never sRGB-encoded");
    assert_eq!(
        canvas_clear([-1.0, 2.0, 0.0, 1.0], false),
        Ok([0.0, 1.0, 0.0, 1.0])
    );
    assert!(canvas_clear([0.5, f32::NAN, 0.5, 1.0], true).is_err());
    assert!(canvas_clear([0.5, 0.5, 0.5, f32::INFINITY], false).is_err());
}

#[test]
fn instances_share_joint_sets_until_the_last_of_them_goes() {
    let mut props = Props::default();
    let mesh = PropMesh(props.meshes.insert(z_quad_mesh_entry(0.5)).unwrap());
    let mat = PropMaterial(
        props
            .materials
            .insert(dummy_material_entry(MaterialDesc::default()))
            .unwrap(),
    );
    let ids: Vec<u32> = (0..3)
        .map(|_| props.instances.insert(instance_entry(mesh, mat)).unwrap())
        .collect();
    let pose = |y: f32| [math::IDENTITY, translation_matrix(0.0, y, 0.0)];
    let set_of = |props: &Props, id: u32| props.instances.get(id).unwrap().joints;
    let users = |props: &Props, set: u32| props.joint_sets.get(set).map(|s| s.users);

    // One set for all three, written in place while it is theirs alone.
    props.share_joints(&ids, &pose(1.0)).unwrap();
    let set = set_of(&props, ids[0]).unwrap();
    props.share_joints(&ids, &pose(2.0)).unwrap();
    assert!(ids.iter().all(|&id| set_of(&props, id) == Some(set)));
    assert_eq!(users(&props, set), Some(3));

    // Given the same matrices of its own, an instance stays the same frame to frame.
    props.pose_instances();
    let still = props.instances.get(ids[0]).unwrap().still;
    props.share_joints(&ids[..1], &pose(2.0)).unwrap();
    props.pose_instances();
    assert_eq!(props.instances.get(ids[0]).unwrap().still, still + 1);
    let own = set_of(&props, ids[0]).unwrap();
    assert_ne!(own, set);
    assert_eq!(users(&props, set), Some(2));

    // Named twice, an instance counts once; a set goes with the last instance following it.
    props
        .share_joints(&[ids[1], ids[1], ids[2]], &pose(4.0))
        .unwrap();
    let shared = set_of(&props, ids[1]).unwrap();
    assert_eq!(users(&props, shared), Some(2));
    assert_eq!(users(&props, set), None);

    // One of two users named twice is not the set's every user: the other keeps its joints.
    let matrices = |props: &Props, id: u32| {
        joints_of(&props.joint_sets, props.instances.get(id).unwrap()).to_vec()
    };
    props.share_joints(&[ids[1], ids[1]], &pose(6.0)).unwrap();
    assert_eq!(matrices(&props, ids[1]), pose(6.0));
    assert_eq!(matrices(&props, ids[2]), pose(4.0));
    assert_eq!(set_of(&props, ids[2]), Some(shared));
    assert_eq!(users(&props, shared), Some(1));
    props.share_joints(&ids[1..], &pose(4.0)).unwrap();
    let shared = set_of(&props, ids[1]).unwrap();
    assert_eq!(users(&props, shared), Some(2));
    props.share_joints(&ids[..1], &[]).unwrap();
    assert_eq!((set_of(&props, ids[0]), users(&props, own)), (None, None));
    props.share_joints(&[], &pose(5.0)).unwrap();
    props.remove_instance(ids[1]);
    props.remove_instance(ids[2]);
    assert_eq!(props.joint_sets.iter().count(), 0);
}

/// Runs made for WebGL 2's storage texture rows send none of them twice and leave the buffer
/// holding what a whole write would; made for 4 bytes they are the runs joined across four
/// unchanged rows alone, as before granularity.
#[test]
fn changed_runs_send_each_unit_of_the_write_granularity_once() {
    const TEXTURE_ROW: u64 = 16384;
    let by_gap = |changed: &[bool]| {
        let mut runs = vec![];
        let mut r = 0;
        while r < changed.len() {
            if !changed[r] {
                r += 1;
                continue;
            }
            let (start, mut end) = (r, r + 1);
            r = end;
            while r < changed.len() && r - end <= 4 {
                if changed[r] {
                    end = r + 1;
                }
                r += 1;
            }
            runs.push(start..end);
        }
        runs
    };
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for trial in 0..400 {
        let rows = (next() % 400) as usize + 1;
        let row = [16, 40][trial % 2];
        let density = next() % 40 + 1;
        let changed: Vec<bool> = (0..rows).map(|_| next() % 100 < density).collect();
        let runs = |unit| -> Vec<_> { changed_runs(rows, row * 4, unit, |r| changed[r]).collect() };
        assert_eq!(runs(4), by_gap(&changed));
        let joined = runs(TEXTURE_ROW);
        let unit = |byte: usize| byte / TEXTURE_ROW as usize;
        for pair in joined.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            assert!(
                unit(a.end * row * 4 - 1) < unit(b.start * row * 4),
                "{joined:?}"
            );
        }
        assert!(
            joined
                .iter()
                .all(|r| changed[r.start] && changed[r.end - 1])
        );
        let held: Vec<f32> = (0..rows * row).map(|k| k as f32).collect();
        let data: Vec<f32> = (0..rows * row)
            .map(|k| {
                if changed[k / row] {
                    -1.0 - k as f32
                } else {
                    k as f32
                }
            })
            .collect();
        let mut buffer = held.clone();
        for r in &joined {
            let at = r.start * row..r.end * row;
            buffer[at.clone()].copy_from_slice(&data[at]);
        }
        assert_eq!(buffer, data);
    }
    // Instance rows (160 bytes) far apart in one texture row are one write; in two, two.
    let two = |a: usize, b: usize| -> Vec<_> {
        changed_runs(200, 160, TEXTURE_ROW, |r| r == a || r == b).collect()
    };
    assert_eq!(two(0, 50).as_slice(), std::slice::from_ref(&(0..51)));
    assert_eq!(two(0, 102).as_slice(), std::slice::from_ref(&(0..103)));
    assert_eq!(two(0, 103), [0..1, 103..104]);
    assert_eq!(two(101, 103).as_slice(), std::slice::from_ref(&(101..104)));
}

fn bound_bits((lo, hi): (V3, V3)) -> [u32; 6] {
    std::array::from_fn(|k| if k < 3 { lo[k] } else { hi[k - 3] }.to_bits())
}

#[test]
fn a_morph_finds_the_bounds_and_joint_boxes_a_whole_scan_gives() {
    let wave = |i: usize| {
        let t = i as f32;
        [(t * 0.37).sin(), (t * 1.3).cos() - 0.5, t * 0.01 - 0.2]
    };
    let points: Vec<V3> = (0..50).map(wave).collect();
    for (first, end) in [(0, 5), (0, 50), (17, 30), (45, 50)] {
        let morph = Morph::new(&points, first, end, None);
        let found = morph.bounds(&points, &points[first..end]);
        assert_eq!(
            bound_bits(found),
            bound_bits(bounds(&points)),
            "{first}..{end}"
        );
    }
    // A bound of 0 holds the zero the points hold; with both zeros there the whole scan
    // decides. A -0 inside the bounds is no matter.
    let left: Vec<V3> = points
        .iter()
        .map(|p| [-p[0].abs() - 0.1, p[1], p[2]])
        .collect();
    for signs in [[0.0, 0.0, -0.0], [0.0, -0.0, -0.0], [-0.0, -0.0, -0.0]] {
        let mut zeros = left.clone();
        zeros[3][0] = signs[0];
        zeros[40][0] = signs[1];
        zeros[12][1] = signs[2];
        for (first, end) in [(0, 5), (40, 41), (10, 20), (0, 50)] {
            let morph = Morph::new(&zeros, first, end, None);
            let found = morph.bounds(&zeros, &zeros[first..end]);
            assert_eq!(
                bound_bits(found),
                bound_bits(bounds(&zeros)),
                "{signs:?} {first}..{end}"
            );
        }
    }

    // Each vertex on two to four joints, one twice, some weighing nothing, one past the
    // boxes held; the boxes held missing one joint.
    let records: Vec<[f32; 4]> = (0..50)
        .flat_map(|v| {
            let j = (v % 6) as f32;
            [
                [j, j + 1.0, j, 7.0],
                [0.5, if v % 3 == 0 { 0.0 } else { 0.25 }, 0.25, 0.1],
            ]
        })
        .collect();
    let mut held = joint_boxes(&points, &records);
    held.truncate(7);
    held[2] = None;
    let moved: Vec<V3> = (0..50).map(|i| math::scale(wave(i + 3), 1.2)).collect();
    let mut expected = held.clone();
    let mut changes = 0;
    for (h, b) in expected.iter_mut().zip(joint_boxes(&moved, &records)) {
        if let Some(b) = b {
            let was = *h;
            let x = h.get_or_insert(b);
            *x = (
                std::array::from_fn(|k| x.0[k].min(b.0[k])),
                std::array::from_fn(|k| x.1[k].max(b.1[k])),
            );
            changes += u32::from(*h != was);
        }
    }
    let morph = Morph::new(&moved, 0, 50, Some(&records));
    assert_eq!(morph.widen(&mut held, &moved), changes);
    assert!(changes > 0);
    let bits = |b: &[Option<(V3, V3)>]| -> Vec<Option<[u32; 6]>> {
        b.iter().map(|b| b.map(bound_bits)).collect()
    };
    assert_eq!(bits(&held), bits(&expected));
}

/// A mesh morphed over a range frame after frame, over another range and whole keeps the
/// bounds a scan of all its vertices gives, to the bit.
#[test]
fn morphed_meshes_keep_the_bounds_of_all_their_vertices() {
    use crate::render::{Look, Output, Quality};
    use crate::world::World;

    let Ok(mut device) = Device::headless() else {
        return;
    };
    let mut r = Renderer::new(
        &mut device,
        &World::default(),
        Look::default(),
        Quality::low_cost(),
        (16, 16),
        Output::Still,
    )
    .expect("headless renderer");
    let positions: Vec<V3> = (0..40)
        .map(|i| [i as f32 * 0.1 - 2.0, (i % 7) as f32, -(i as f32)])
        .collect();
    let normals = vec![[0.0, 1.0, 0.0]; 40];
    let indices: Vec<u32> = (0..39).flat_map(|i| [i, i + 1, 0]).collect();
    let mesh = r
        .create_prop_mesh(
            &mut device,
            &MeshData {
                positions: &positions,
                normals: &normals,
                uvs: &[],
                indices: &indices,
            },
        )
        .expect("mesh");
    let check = |r: &Renderer, what: &str| {
        let m = r.props.meshes.get(mesh.0).unwrap();
        assert_eq!(
            bound_bits((m.min, m.max)),
            bound_bits(bounds(&m.positions)),
            "{what}"
        );
    };
    // The morphed range holds the lowest x at first, then pulls in, then reaches past
    // every other vertex, then rests on 0 beside a vertex at -0.
    for (k, shift) in [0.0, 0.5, 1.0, -3.0, 2.0].into_iter().enumerate() {
        let moved: Vec<V3> = positions[0..5]
            .iter()
            .map(|p| [p[0] + shift, p[1] * 2.0, p[2]])
            .collect();
        r.update_prop_mesh_range(&mut device, mesh, 0, &moved, &normals[0..5])
            .expect("morph");
        check(&r, &format!("morph {k}"));
    }
    let other: Vec<V3> = positions[10..14]
        .iter()
        .map(|p| [p[0], p[1] + 9.0, p[2]])
        .collect();
    r.update_prop_mesh_range(&mut device, mesh, 10, &other, &normals[10..14])
        .expect("another range");
    check(&r, "another range");
    let zeros: Vec<V3> = positions[0..5].iter().map(|p| [p[0], 0.0, p[2]]).collect();
    let mut whole = positions.clone();
    whole[20][1] = -0.0;
    r.update_prop_mesh(&mut device, mesh, &whole, &normals)
        .expect("whole");
    check(&r, "whole");
    for k in 0..2 {
        r.update_prop_mesh_range(&mut device, mesh, 0, &zeros, &normals[0..5])
            .expect("zeros");
        check(&r, &format!("zeros {k}"));
    }
    r.destroy_props(&mut device);
}

#[test]
fn cleared_prop_textures_hold_the_canvas_bytes_in_every_mip_and_take_updates() {
    use crate::render::{Look, Output, Quality};
    use crate::world::World;

    let Ok(mut device) = Device::headless() else {
        return;
    };
    let mut r = Renderer::new(
        &mut device,
        &World::default(),
        Look::default(),
        Quality::low_cost(),
        (16, 16),
        Output::Still,
    )
    .expect("headless renderer");
    let read = |device: &mut Device<'_>, r: &Renderer, t: PropTexture, mip: u32| {
        let id = r.prop_texture_id(Some(t), r.dummy).expect("texture");
        let (w, h) = device.texture_desc(id).expect("desc").mip_size(mip);
        let rb = device
            .create_buffer(&BufferDesc {
                size: u64::from(w * h * 4),
                usage: BufferUsage::COPY_DST,
                memory: Memory::Readback,
            })
            .expect("readback");
        let mut e = device.encoder();
        e.copy_texture_to_buffer(id, mip, 0, rb, 0);
        device.submit(e).expect("copy");
        let bytes = device.read_buffer(rb).expect("read");
        device.destroy_buffer(rb);
        bytes
    };
    let solid = |px: [u8; 4], n: usize| px.repeat(n);

    // 5x3 has mips 5x3, 2x1 and 1x1; every one holds the bytes a canvas of the colour would.
    let srgb = r
        .create_prop_texture_clear(&mut device, (5, 3), true, [0.2, 0.4, 0.8, 1.0])
        .expect("srgb clear");
    let unorm = r
        .create_prop_texture_clear(&mut device, (5, 3), false, [0.2, 0.4, 0.8, 0.6])
        .expect("unorm clear");
    for (mip, n) in [(0, 15), (1, 2), (2, 1)] {
        assert_eq!(
            read(&mut device, &r, srgb, mip),
            solid([51, 102, 204, 255], n)
        );
        assert_eq!(
            read(&mut device, &r, unorm, mip),
            solid([51, 102, 204, 153], n)
        );
    }

    // A placeholder takes the picture later at its size; a wrong size or a failed fill
    // leaves it and its edit count as they were.
    let edits = |r: &Renderer| r.props.texture_edits.get(&srgb.0).copied().unwrap_or(0);
    assert!(r.update_prop_texture(&mut device, srgb, &[9; 4]).is_err());
    let refused = r.update_prop_texture_with(&mut device, srgb, |_, _| Err("refused".into()));
    assert_eq!(refused, Err("refused".into()));
    assert_eq!(edits(&r), 0);
    assert_eq!(
        read(&mut device, &r, srgb, 0),
        solid([51, 102, 204, 255], 15)
    );
    r.update_prop_texture(&mut device, srgb, &solid([200, 10, 30, 255], 15))
        .expect("update");
    assert_eq!(edits(&r), 1);
    assert_eq!(
        read(&mut device, &r, srgb, 0),
        solid([200, 10, 30, 255], 15)
    );
    assert_eq!(read(&mut device, &r, srgb, 2), solid([200, 10, 30, 255], 1));

    // A refused fill or colour makes no prop texture.
    let count = |r: &Renderer| r.props.textures.iter().count();
    let before = count(&r);
    let made =
        r.create_prop_texture_with(&mut device, (4, 4), true, |_, _| Err("no source".into()));
    assert_eq!(made, Err("no source".into()));
    assert!(
        r.create_prop_texture_clear(&mut device, (4, 4), true, [f32::NAN, 0.0, 0.0, 1.0])
            .is_err()
    );
    assert_eq!(count(&r), before);

    r.destroy_props(&mut device);
}
