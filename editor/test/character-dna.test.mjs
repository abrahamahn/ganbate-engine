import assert from 'node:assert/strict';
import test from 'node:test';
import {
  ANIM_STATE_MACHINE_SCHEMA,
  asmToSemanticSubgraph,
  createAnimStateMachineController,
  createDefaultAnimStateMachine,
  liftContinuousPoseToAsmParams,
  selectMatchingTransition,
  summarizeAsmForAi,
} from '../anim-state-machine.js';
import {
  BODY_OCCLUSION_REGIONS,
  CHARACTER_DNA_SCHEMA,
  compileDnaToRigOverrides,
  createAvatarDnaController,
  createDefaultCharacterDna,
  generateDnaFromTraits,
  hitTestStageDragRegion,
  liftStageRegionDragToDna,
  solveWardrobeOcclusionConstraints,
  summarizeCharacterDnaForAi,
} from '../character-dna.js';

test('5-layer Parametric Character DNA, 3D stage inverse lifting, and wardrobe anti-clipping solver', () => {
  const dna = createDefaultCharacterDna();
  assert.equal(dna.schema, CHARACTER_DNA_SCHEMA);
  assert.equal(dna.bodyBuild.statureScale, 1.0);

  // Hit-testing 3D stage regions for direct sculpting
  assert.equal(hitTestStageDragRegion(0.5, 0.39, 'bust'), 'nose');
  assert.equal(hitTestStageDragRegion(0.5, 0.51, 'bust'), 'jaw');
  assert.equal(hitTestStageDragRegion(0.5, 0.32, 'body'), 'shoulders');

  // Inverse Semantic Lifting (`L_dna: R^3 / R^2 -> G_dna`)
  const liftedNose = liftStageRegionDragToDna(dna, 'nose', 0.25, -0.3);
  assert.equal(liftedNose.regionId, 'nose');
  assert.ok(liftedNose.dna.craniofacial.nose.bridgeHeight > dna.craniofacial.nose.bridgeHeight);
  assert.ok(liftedNose.dna.craniofacial.nose.nostrilWidth > dna.craniofacial.nose.nostrilWidth);
  assert.ok(liftedNose.updatedTraits.length >= 2);

  // Wardrobe Anti-Clipping Constraint Solver (`C_occ`)
  const thickInnerDna = createDefaultCharacterDna();
  thickInnerDna.wardrobeSlots.inner.itemId = 'inner_high_collar_knit'; // 3.2mm
  thickInnerDna.wardrobeSlots.inner.scaleOffsetMm = 2.0; // effective 5.2mm > top minInnerClearanceMm (2.0mm)
  const solvedOcc = solveWardrobeOcclusionConstraints(thickInnerDna);
  assert.ok((solvedOcc.bodyOcclusionMask & BODY_OCCLUSION_REGIONS.TORSO_UPPER) !== 0);
  assert.ok((solvedOcc.bodyOcclusionMask & BODY_OCCLUSION_REGIONS.NECK_COLLAR) !== 0);
  assert.ok((solvedOcc.bodyOcclusionMask & BODY_OCCLUSION_REGIONS.FEET) !== 0);
  assert.ok(solvedOcc.projections.length >= 1, 'expected inner/top garment clearance projection');
  assert.ok(solvedOcc.dna.wardrobeSlots.top.scaleOffsetMm > 0);

  // Semantic trait generation & headless controller
  const scholarDna = generateDnaFromTraits('tall slender scholar with sculpted jaw, heterochromia, and pavilion coat');
  assert.ok(scholarDna.bodyBuild.statureScale > 1.0);
  assert.equal(scholarDna.skinIrisDecals.iris.heterochromia, true);
  assert.equal(scholarDna.wardrobeSlots.outer.itemId, 'outer_haori_coat');

  const ctrl = createAvatarDnaController(scholarDna);
  let notifications = 0;
  const unsub = ctrl.subscribe(() => {
    notifications++;
  });
  ctrl.dragStageRegion('jaw', -0.2, 0.15);
  assert.equal(notifications, 1);
  const compiled = ctrl.compileRigOverrides();
  assert.ok(compiled.effectiveArmReachM > 0.56);
  const summary = summarizeCharacterDnaForAi(ctrl.getDna());
  assert.ok(summary.includes('DNA[stature='));
  assert.ok(summary.split(/\s+/).length < 40);
  unsub();
});

test('Circumstance-Driven Animation State Machine conditions, Hermite crossfade, and semantic subgraph', () => {
  const asm = createDefaultAnimStateMachine();
  assert.equal(asm.schema, ANIM_STATE_MACHINE_SCHEMA);

  const matchWalk = selectMatchingTransition(asm, 'Idle', { locomotionSpeed: 0.45 }, []);
  assert.ok(matchWalk);
  assert.equal(matchWalk.to, 'Walk');

  const ctrl = createAnimStateMachineController(asm);
  assert.equal(ctrl.getState().currentState, 'Idle');

  // Inverse lift continuous 3D IK target distance into ASM parameters -> transitions Idle -> Reach -> Interact
  const reachParams = liftContinuousPoseToAsmParams(asm, {
    shoulderOrigin: [0.18, 1.38, 0.0],
    ikTarget: [0.32, 1.15, 0.38],
    ikEnabled: true,
  });
  assert.equal(reachParams.reachActive, true);
  assert.ok(reachParams.interactionDistance < 0.58);

  const afterReach = ctrl.setParameters(reachParams);
  assert.equal(afterReach.currentState, 'Reach');
  assert.equal(afterReach.previousState, 'Idle');

  // Step halfway through crossfade and verify Hermite weight + next transition to Interact
  const midStep = ctrl.step(0.08);
  assert.equal(midStep.currentState, 'Interact');
  assert.ok(midStep.hermiteWeight >= 0 && midStep.hermiteWeight <= 1);

  // Trigger priority override (`applaud` -> `React_Positive`)
  const afterTrigger = ctrl.fireTrigger('applaud');
  assert.equal(afterTrigger.currentState, 'React_Positive');

  const subgraph = asmToSemanticSubgraph(asm, afterTrigger);
  assert.equal(subgraph.nodes.length, asm.states.length);
  assert.equal(subgraph.edges.length, asm.transitions.length);
  assert.ok(subgraph.nodes.some((n) => n.stateId === 'React_Positive' && n.active));

  const aiDigest = summarizeAsmForAi(asm, afterTrigger);
  assert.ok(aiDigest.includes('ASM[state=React_Positive'));
});
