import assert from 'node:assert/strict';
import test from 'node:test';
import {
  bindCharacterShapes,
  createCharacterRigController,
  encodePoseCommandPacket,
  hermiteFade,
  poseModelClip,
  quatFromEuler,
  sampleBlendSpace1D,
  sampleBlendSpace2D,
} from '../../web/character-customization.mjs';
import {
  CHARACTER_ACADEMIC_SCENARIOS,
  CHARACTER_SCHEMA,
  CHARACTER_SEMANTIC_GRAPH_SCHEMA,
  applyCharacterSemanticPatch,
  buildBoneOffsetOps,
  buildPresetApplyOps,
  createDefaultCharacterDocument,
  evaluateCharacterPoseSpec,
  filterBoneTree,
  getCharacterSemanticSubgraph,
  groupMorphTargets,
  liftCharacterStageDragToSemanticGraph,
  mountCharacterStudio,
  projectCharacterIkConstraints,
  resolveTwoAgentCharacterConflict,
  summarizeCharacterForAi,
} from '../character-studio.js';

function createMiniDom() {
  class MiniElement {
    constructor(tag, ownerDocument) {
      this.tagName = String(tag).toUpperCase();
      this.ownerDocument = ownerDocument;
      this.className = '';
      this.textContent = '';
      this.value = '';
      this.type = '';
      this.dataset = {};
      this.style = {};
      this.children = [];
      this.listeners = new Map();
    }
    set innerHTML(val) {
      if (val === '') this.children = [];
    }
    get innerHTML() {
      return '';
    }
    appendChild(child) {
      this.children.push(child);
      return child;
    }
    addEventListener(type, fn) {
      const list = this.listeners.get(type) ?? [];
      list.push(fn);
      this.listeners.set(type, list);
    }
    dispatch(type, extra = {}) {
      for (const fn of this.listeners.get(type) ?? []) {
        fn({ target: this, ...extra });
      }
    }
    find(predicate) {
      if (predicate(this)) return this;
      for (const ch of this.children) {
        const hit = ch.find?.(predicate);
        if (hit) return hit;
      }
      return null;
    }
  }
  const doc = {
    createElement(tag) {
      return new MiniElement(tag, doc);
    },
  };
  return new MiniElement('div', doc);
}

test('character-customization single-call packet encoding, blend spaces, and rig controller', () => {
  assert.equal(hermiteFade(0, 0.2), 0);
  assert.equal(hermiteFade(0.1, 0.2), 0.5);
  assert.equal(hermiteFade(0.25, 0.2), 1);

  const bs1 = sampleBlendSpace1D(
    [
      { clip: 0, position: 0 },
      { clip: 1, position: 1 },
    ],
    0.25,
    0.5,
  );
  assert.equal(bs1.length, 2);
  assert.equal(bs1[0].weight, 1);
  assert.ok(Math.abs(bs1[1].weight - 0.25) < 1e-6);

  const bs2 = sampleBlendSpace2D(
    [
      { clip: 0, position: [0, 0] },
      { clip: 1, position: [1, 0] },
      { clip: 2, position: [0, 1] },
    ],
    [1, 0],
    0.4,
  );
  assert.equal(bs2.length, 1);
  assert.equal(bs2[0].clip, 1);

  let singleCalls = 0;
  let lastPacket = null;
  const wasmEngine = {
    snapshotModelPose() {},
    poseModelClip(instance, packet) {
      assert.equal(instance, 7);
      singleCalls++;
      lastPacket = packet;
    },
    modelClip() {
      throw new Error('should not fall back to per-joint JS calls when poseModelClip exists');
    },
    modelNode() {
      return { translation: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] };
    },
    setModelNode() {},
    modelMorph() {},
    updateModel() {},
  };

  const info = {
    nodes: [
      { name: 'hips', mesh: null },
      { name: 'neck_01', mesh: null },
      { name: 'head', mesh: 0 },
    ],
    meshes: [{ targetNames: ['blink', 'smile'], targets: 2 }],
    clips: [
      { name: 'idle_calm', duration: 2.0, nodes: [0, 1, 2] },
      { name: 'gesture_greet', duration: 1.5, nodes: [0, 1, 2] },
    ],
  };

  const rig = createCharacterRigController(wasmEngine, 7, info, { defaultCrossfadeSeconds: 0.2 });
  rig.evaluate({
    clockSeconds: 0,
    stateKey: 'idle',
    clip: 'idle_calm',
    time: 0.4,
    lookAt: { targetModel: [0.2, 1.6, 2.0], joints: [['neck_01', 0.35], ['head', 0.65]] },
    bonePoses: { hips: [0, 0.1, 0] },
    morphWeights: { blink: 0.8 },
  });
  rig.evaluate({
    clockSeconds: 0.05,
    stateKey: 'greet',
    clip: 'gesture_greet',
    time: 0.05,
  });
  assert.equal(singleCalls, 2);
  assert.ok(lastPacket instanceof Float32Array);
  assert.ok((Math.round(lastPacket[0]) & 4) !== 0);

  const morphCalls = [];
  const binder = bindCharacterShapes(
    {
      modelMorph: (...args) => morphCalls.push(args),
      updateModel: () => {},
    },
    3,
    info,
    { blink: [0, 1], smile: [0, 1] },
  );
  assert.deepEqual(binder.available, ['blink', 'smile']);
  assert.equal(binder.apply({ blink: 1.5 }), true);
  assert.deepEqual(morphCalls[0], [3, 2, 0, 1]);
  assert.equal(quatFromEuler(0, 0, 0)[3], 1);
  assert.ok(encodePoseCommandPacket({ clips: [{ clip: 0, time: 0.2, weight: 1 }] }).length >= 12);
});

test('Tri-Modal Articulated Semantic Graph: constraint projection (x_hat -> x*), inverse lifting, and 2-agent conflict solver', () => {
  const doc = createDefaultCharacterDocument('scholar_actor', 'Scholar Actor');

  // 1. Minimal-Perturbation Articulated Constraint Projection (`projectCharacterIkConstraints`)
  const invalidDoc = structuredClone(doc);
  invalidDoc.ikChains[0].enabled = true;
  invalidDoc.ikChains[0].target = [1.85, -0.35, 1.4]; // Exceeds arm reach manifold AND penetrates floor!
  invalidDoc.lookAt.target = [3.5, 2.9, -0.5]; // Behind head plane AND exceeds ±55° yaw / ±35° pitch cone!

  const proj = projectCharacterIkConstraints(invalidDoc);
  assert.equal(proj.verified, false);
  assert.ok(proj.violations.length >= 2);
  assert.ok(proj.hallucinationGhosts.length >= 2);

  const projectedTarget = proj.document.ikChains[0].target;
  assert.ok(projectedTarget[1] >= 0.05, 'floor constraint must clamp y >= 0.05m');
  const shoulder = [0.18, 1.38, 0.0];
  const distFromShoulder = Math.hypot(
    projectedTarget[0] - shoulder[0],
    projectedTarget[1] - shoulder[1],
    projectedTarget[2] - shoulder[2],
  );
  assert.ok(distFromShoulder <= proj.effectiveArmReachM + 1e-2, `expected dist ${distFromShoulder} <= ${proj.effectiveArmReachM}`);

  // 2. Inverse Semantic Lifting (`L_char: R^3 -> G_char`)
  const lifted = liftCharacterStageDragToSemanticGraph(doc, {
    kind: 'ik_target',
    chainIndex: 0,
    target: [0.36, 1.22, 0.34],
    enabled: true,
  });
  assert.equal(lifted.rejected, false);
  assert.equal(lifted.subgraph.schema, CHARACTER_SEMANTIC_GRAPH_SCHEMA);
  const reachEdge = lifted.liftedEdges.find((e) => e.rel === 'reaches_for');
  assert.ok(reachEdge, 'expected symbolic reaches_for edge');
  assert.ok(reachEdge.distanceM > 0 && reachEdge.distanceM <= 0.56);

  // 3. Natural-language prompt patching (`applyCharacterSemanticPatch`)
  const patched = applyCharacterSemanticPatch(
    doc,
    'Reach for [1.4, 1.2, 1.1] with right hand, look at [0.25, 1.55, 1.8], tall scholar with heterochromia',
  );
  assert.equal(patched.rejected, false);
  assert.ok(patched.projection.hallucinationGhosts.length >= 1, 'unreachable IK target [1.4, 1.2, 1.1] should project x_hat -> x*');
  assert.equal(patched.document.dna.skinIrisDecals.iris.heterochromia, true);

  // 4. 2-Agent Concurrent Character Edit Conflict Resolution
  const conflictResolved = resolveTwoAgentCharacterConflict(doc);
  assert.ok(conflictResolved.naiveConflictCount >= 1);
  assert.ok(conflictResolved.hallucinationGhosts.length >= 1);
  assert.equal(conflictResolved.subgraph.verification.verified, true);

  // 5. Forward Symbolic Edge Grounding (`G_char -> R^3 -> x*`)
  const edgeGrounded = applyCharacterSemanticPatch(doc, {
    edges: [
      { from: 'effector:scholar_actor:hand_r', rel: 'reaches_for', to: 'anchor:left_workspace' },
      { from: 'char:scholar_actor:head', rel: 'looks_at', to: 'anchor:work_surface_below' },
      { from: 'wardrobe:scholar_coat', rel: 'occludes_body_mask', to: 'char:scholar_actor' },
    ],
  });
  assert.equal(edgeGrounded.rejected, false);
  assert.equal(edgeGrounded.document.ikChains[0].enabled, true);
  assert.ok(edgeGrounded.document.ikChains[0].target[0] < 0, 'left_workspace edge should ground to negative X');
  assert.ok(edgeGrounded.document.lookAt.target[1] < 1.35, 'work_surface_below edge should ground to lower Y');
  assert.equal(edgeGrounded.document.activeOutfitId, 'scholar_coat');

  // 6. Academic Scenarios catalog
  assert.ok(CHARACTER_ACADEMIC_SCENARIOS.embodied_scholar_reach);
  assert.ok(CHARACTER_ACADEMIC_SCENARIOS.unreachable_ik_projection);
  assert.ok(CHARACTER_ACADEMIC_SCENARIOS.two_agent_posture_conflict);
  assert.ok(CHARACTER_ACADEMIC_SCENARIOS.wardrobe_layer_occlusion);

  // 7. Token-compact AI summary (`<120 tokens`)
  const digest = summarizeCharacterForAi(conflictResolved.resolvedDocument);
  assert.ok(digest.includes('CHAR[scholar_actor'));
  assert.ok(digest.includes('DNA['));
  assert.ok(digest.includes('ASM['));
  assert.ok(digest.split(/\s+/).length < 120);
});

test('mountCharacterStudio renders rig hierarchy, emits studioApi.edit ops, and exposes Tri-Modal hooks', () => {
  const container = createMiniDom();
  const edits = [];
  const packets = [];
  const doc = createDefaultCharacterDocument('hero_actor', 'Hero Actor');

  assert.equal(doc.schema, CHARACTER_SCHEMA);
  assert.equal(filterBoneTree(doc.skeleton.joints, 'arm').length, 2);
  assert.equal(groupMorphTargets(doc.morphTargets).expression.length, 2);
  assert.deepEqual(buildBoneOffsetOps('hips', [0, 0, 0], [0, 0, 0]), [{ remove: ['boneOffsets', 'hips'] }]);
  assert.equal(buildPresetApplyOps(doc, 'warm_smile').length, 2);

  const spec = evaluateCharacterPoseSpec(doc, {
    blendSpaceId: 'stance_blend_1d',
    blendParam: 0.5,
    timeS: 0.3,
  });
  assert.equal(spec.clips.length, 2);

  const handle = mountCharacterStudio(container, {
    getActiveDocument: () => ({ id: 'doc_hero', path: 'content/characters/hero.character.json', root: doc }),
    edit(id, ops, label) {
      edits.push({ id, ops, label });
    },
    getEngine: () => ({
      activeCharacterInstance: 11,
      poseModelClip(inst, pkt) {
        packets.push({ inst, pkt });
      },
    }),
  });

  assert.ok(packets.length >= 1);
  assert.equal(packets[0].inst, 11);

  const smileSlider = container.find((el) => el.dataset?.morph === 'smile');
  assert.ok(smileSlider, 'expected smile morph slider');
  smileSlider.value = '0.85';
  smileSlider.dispatch('input');
  assert.equal(edits.length, 1);
  assert.deepEqual(edits[0].ops, [{ set: ['morphWeights', 'smile'], value: 0.85 }]);

  const ikBtn = container.find((el) => el.dataset?.role === 'ik-toggle-0');
  assert.ok(ikBtn);
  ikBtn.dispatch('click');
  assert.equal(edits.length, 2);
  assert.deepEqual(edits[1].ops, [{ set: ['ikChains', '0', 'enabled'], value: true }]);

  // Verify Tri-Modal Hook Contract on mounted controller
  const subgraph = handle.getCharacterSemanticSubgraph();
  assert.equal(subgraph.schema, CHARACTER_SEMANTIC_GRAPH_SCHEMA);
  assert.ok(subgraph.nodes.some((n) => n.type === 'articulated_humanoid'));

  const conflictBtn = container.find((el) => el.dataset?.role === 'char-simulate-conflict');
  assert.ok(conflictBtn, 'expected 2-Agent Conflict button in Character Studio');
  conflictBtn.dispatch('click');
  assert.equal(edits.length, 3);

  const runScenarioBtn = container.find((el) => el.dataset?.role === 'char-run-scenario');
  assert.ok(runScenarioBtn, 'expected Run Academic Preset button');
  runScenarioBtn.dispatch('click');
  assert.equal(edits.length, 4);

  const applyGraphBtn = container.find((el) => el.dataset?.role === 'char-apply-graph-json');
  assert.ok(applyGraphBtn, 'expected Apply Graph JSON button');
  applyGraphBtn.dispatch('click');
  assert.equal(edits.length, 5);

  const outerWardrobeSelect = container.find((el) => el.dataset?.role === 'dna-wardrobe-outer');
  assert.ok(outerWardrobeSelect, 'expected Layer 4 outer wardrobe slot selector');
  outerWardrobeSelect.value = 'outer_pavilion_coat';
  outerWardrobeSelect.dispatch('change');
  assert.equal(edits.length, 6);

  const asmReachToggle = container.find((el) => el.dataset?.role === 'asm-param-reachActive');
  assert.ok(asmReachToggle, 'expected live ASM reachActive parameter toggle');
  asmReachToggle.dispatch('click');

  handle.onDeactivate();
  handle.onActivate();
  handle.destroy();
  assert.equal(container.children.length, 0);
});
