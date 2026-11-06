/**
 * Circumstance-Driven Animation State Machine (`ganbate.character.asm/1`).
 *
 * Genre-agnostic declarative state graph (`Idle`, `Walk`, `Interact`, `Reach`, `Action`, `React_*`)
 * with rule-based transition conditions (`==`, `!=`, `>=`, `<=`, `>`, `<`, `onTrigger`),
 * Hermite (`3t^2 - 2t^3`) cross-fade evaluation, facial morph & Look-At overrides, and
 * bidirectional semantic subgraph projection (`nodes[]` + `edges[]`).
 */

import { hermiteFade } from '../web/character-customization.mjs';

export const ANIM_STATE_MACHINE_SCHEMA = 'ganbate.character.asm/1';

export function createDefaultAnimStateMachine() {
  return {
    schema: ANIM_STATE_MACHINE_SCHEMA,
    initialState: 'Idle',
    parameters: {
      locomotionSpeed: 0.0,
      interactionDistance: 1.2,
      circumstance: 'calm',
      focusIntensity: 0.5,
      reachActive: false,
    },
    states: [
      {
        id: 'Idle',
        label: 'Idle Poise',
        clipId: 'idle_calm',
        looping: true,
        speed: 1.0,
        morphOverrides: { blink: 0.04, smile: 0.12 },
        lookAtEnabled: true,
        ikEnabled: false,
      },
      {
        id: 'Walk',
        label: 'Walk Stride',
        clipId: 'walk_stride',
        blendSpaceId: 'stance_blend_1d',
        looping: true,
        speed: 1.0,
        morphOverrides: { blink: 0.02, smile: 0.05 },
        lookAtEnabled: true,
        ikEnabled: false,
      },
      {
        id: 'Reach',
        label: 'Reach & Grasp IK',
        clipId: 'gesture_greet',
        looping: false,
        speed: 1.0,
        morphOverrides: { blink: 0.08, smile: 0.18 },
        lookAtEnabled: true,
        ikEnabled: true,
      },
      {
        id: 'Interact',
        label: 'Precision Object Interact',
        clipId: 'gesture_greet',
        looping: false,
        speed: 1.0,
        morphOverrides: { blink: 0.1, smile: 0.35 },
        lookAtEnabled: true,
        ikEnabled: true,
      },
      {
        id: 'Action',
        label: 'Primary Presentation Action',
        clipId: 'gesture_greet',
        looping: false,
        speed: 1.15,
        morphOverrides: { blink: 0.06, smile: 0.42 },
        lookAtEnabled: true,
        ikEnabled: true,
      },
      {
        id: 'React_Positive',
        label: 'Warm Acknowledgment',
        clipId: 'gesture_greet',
        looping: false,
        speed: 1.0,
        morphOverrides: { blink: 0.12, smile: 0.78 },
        lookAtEnabled: true,
        ikEnabled: false,
      },
      {
        id: 'React_Alert',
        label: 'Attentive Focus Shift',
        clipId: 'idle_calm',
        looping: true,
        speed: 1.1,
        morphOverrides: { blink: 0.18, smile: 0.0 },
        lookAtEnabled: true,
        ikEnabled: false,
      },
    ],
    transitions: [
      {
        id: 'any_to_react_positive',
        from: '*',
        to: 'React_Positive',
        priority: 30,
        crossfadeMs: 180,
        conditions: [{ op: 'onTrigger', trigger: 'applaud' }],
      },
      {
        id: 'any_to_action',
        from: '*',
        to: 'Action',
        priority: 25,
        crossfadeMs: 160,
        conditions: [{ op: 'onTrigger', trigger: 'execute_action' }],
      },
      {
        id: 'idle_to_walk',
        from: 'Idle',
        to: 'Walk',
        priority: 20,
        crossfadeMs: 220,
        conditions: [{ param: 'locomotionSpeed', op: '>=', value: 0.15 }],
      },
      {
        id: 'walk_to_idle',
        from: 'Walk',
        to: 'Idle',
        priority: 20,
        crossfadeMs: 240,
        conditions: [{ param: 'locomotionSpeed', op: '<', value: 0.15 }],
      },
      {
        id: 'idle_to_reach',
        from: 'Idle',
        to: 'Reach',
        priority: 18,
        crossfadeMs: 200,
        conditions: [{ param: 'reachActive', op: '==', value: true }],
      },
      {
        id: 'reach_to_interact',
        from: 'Reach',
        to: 'Interact',
        priority: 19,
        crossfadeMs: 180,
        conditions: [
          { param: 'reachActive', op: '==', value: true },
          { param: 'interactionDistance', op: '<=', value: 0.58 },
        ],
      },
      {
        id: 'interact_to_idle',
        from: 'Interact',
        to: 'Idle',
        priority: 15,
        crossfadeMs: 260,
        conditions: [{ param: 'reachActive', op: '==', value: false }],
      },
      {
        id: 'reach_to_idle',
        from: 'Reach',
        to: 'Idle',
        priority: 15,
        crossfadeMs: 260,
        conditions: [{ param: 'reachActive', op: '==', value: false }],
      },
      {
        id: 'idle_to_alert',
        from: 'Idle',
        to: 'React_Alert',
        priority: 12,
        crossfadeMs: 200,
        conditions: [{ param: 'circumstance', op: '==', value: 'focused' }],
      },
      {
        id: 'alert_to_idle',
        from: 'React_Alert',
        to: 'Idle',
        priority: 12,
        crossfadeMs: 220,
        conditions: [{ param: 'circumstance', op: '==', value: 'calm' }],
      },
    ],
  };
}

export function normalizeAnimStateMachine(raw) {
  const fallback = createDefaultAnimStateMachine();
  if (!raw || typeof raw !== 'object') return fallback;

  const states = Array.isArray(raw.states) && raw.states.length
    ? raw.states.map((s, idx) => ({
        id: String(s.id ?? `State_${idx}`),
        label: String(s.label ?? s.id ?? `State ${idx}`),
        clipId: String(s.clipId ?? 'idle_calm'),
        blendSpaceId: s.blendSpaceId ? String(s.blendSpaceId) : undefined,
        looping: s.looping !== false,
        speed: Number.isFinite(s.speed) && s.speed > 0 ? Number(s.speed) : 1.0,
        morphOverrides: s.morphOverrides && typeof s.morphOverrides === 'object' ? { ...s.morphOverrides } : {},
        lookAtEnabled: s.lookAtEnabled !== false,
        ikEnabled: Boolean(s.ikEnabled),
      }))
    : fallback.states;

  const transitions = Array.isArray(raw.transitions)
    ? raw.transitions.map((t, idx) => ({
        id: String(t.id ?? `tr_${idx}`),
        from: String(t.from ?? '*'),
        to: String(t.to ?? states[0].id),
        priority: Number.isFinite(t.priority) ? Number(t.priority) : 10,
        crossfadeMs: Number.isFinite(t.crossfadeMs) && t.crossfadeMs >= 0 ? Number(t.crossfadeMs) : 200,
        conditions: Array.isArray(t.conditions) ? t.conditions.map((c) => ({ ...c })) : [],
      }))
    : fallback.transitions;

  return {
    schema: ANIM_STATE_MACHINE_SCHEMA,
    initialState: states.some((s) => s.id === raw.initialState) ? raw.initialState : states[0].id,
    parameters: {
      ...fallback.parameters,
      ...(raw.parameters && typeof raw.parameters === 'object' ? raw.parameters : {}),
    },
    states,
    transitions,
  };
}

export function evaluateTransitionCondition(cond, params = {}, pendingTriggers = new Set()) {
  if (!cond || typeof cond !== 'object') return false;
  if (cond.op === 'onTrigger') {
    return Boolean(cond.trigger && pendingTriggers.has(cond.trigger));
  }
  const actual = params[cond.param];
  const expected = cond.value;
  switch (cond.op) {
    case '==':
      return actual === expected;
    case '!=':
      return actual !== expected;
    case '>=':
      return Number(actual) >= Number(expected);
    case '<=':
      return Number(actual) <= Number(expected);
    case '>':
      return Number(actual) > Number(expected);
    case '<':
      return Number(actual) < Number(expected);
    default:
      return false;
  }
}

/**
 * Evaluates which transition (if any) should fire from `currentStateId` given `params` and `triggers`.
 */
export function selectMatchingTransition(rawAsm, currentStateId, params = {}, triggers = []) {
  const asm = normalizeAnimStateMachine(rawAsm);
  const triggerSet = new Set(Array.isArray(triggers) ? triggers : [triggers].filter(Boolean));
  const mergedParams = { ...asm.parameters, ...params };

  const candidates = asm.transitions
    .filter((tr) => (tr.from === '*' || tr.from === currentStateId) && tr.to !== currentStateId)
    .filter(
      (tr) =>
        tr.conditions.length > 0 &&
        tr.conditions.every((cond) => evaluateTransitionCondition(cond, mergedParams, triggerSet)),
    )
    .sort((a, b) => b.priority - a.priority);

  return candidates[0] ?? null;
}

/**
 * Inverse Semantic Lifting (`L_asm: R^3 -> G_asm`) from continuous 3D effector/target distances
 * to discrete Animation State Machine circumstance parameters (`reachActive`, `interactionDistance`).
 */
export function liftContinuousPoseToAsmParams(rawAsm, { shoulderOrigin = [0.18, 1.38, 0], ikTarget = null, ikEnabled = false } = {}) {
  const asm = normalizeAnimStateMachine(rawAsm);
  const nextParams = { ...asm.parameters };
  nextParams.reachActive = Boolean(ikEnabled && Array.isArray(ikTarget));

  if (Array.isArray(ikTarget) && ikTarget.length >= 3) {
    const dx = (Number(ikTarget[0]) || 0) - (Number(shoulderOrigin[0]) || 0);
    const dy = (Number(ikTarget[1]) || 0) - (Number(shoulderOrigin[1]) || 0);
    const dz = (Number(ikTarget[2]) || 0) - (Number(shoulderOrigin[2]) || 0);
    const dist = Math.hypot(dx, dy, dz);
    nextParams.interactionDistance = Number(dist.toFixed(3));
  }

  return nextParams;
}

/**
 * Projects the Animation State Machine into a semantic subgraph (`nodes[]` + `edges[]`)
 * for unified inspection and LLM reasoning.
 */
export function asmToSemanticSubgraph(rawAsm, runtimeState = {}) {
  const asm = normalizeAnimStateMachine(rawAsm);
  const activeStateId = runtimeState.currentState ?? asm.initialState;
  const nodes = asm.states.map((st) => ({
    id: `asm_state:${st.id}`,
    type: 'anim_state',
    stateId: st.id,
    label: st.label,
    clipId: st.clipId,
    active: st.id === activeStateId,
    ikEnabled: st.ikEnabled,
    lookAtEnabled: st.lookAtEnabled,
  }));

  const edges = asm.transitions.map((tr) => ({
    id: tr.id,
    from: tr.from === '*' ? 'asm_state:*' : `asm_state:${tr.from}`,
    rel: 'transitions_on',
    to: `asm_state:${tr.to}`,
    crossfadeMs: tr.crossfadeMs,
    conditionSummary: tr.conditions
      .map((c) => (c.op === 'onTrigger' ? `onTrigger(${c.trigger})` : `${c.param} ${c.op} ${c.value}`))
      .join(' && '),
  }));

  return { nodes, edges };
}

/**
 * Token-compact (`<50 tokens`) summary of the Circumstance Animation State Machine.
 */
export function summarizeAsmForAi(rawAsm, runtimeState = {}) {
  const asm = normalizeAnimStateMachine(rawAsm);
  const cur = runtimeState.currentState ?? asm.initialState;
  const prev = runtimeState.previousState ?? null;
  const fade = Number(runtimeState.hermiteWeight ?? 1).toFixed(2);
  const p = { ...asm.parameters, ...(runtimeState.parameters ?? {}) };
  return [
    `ASM[state=${cur}${prev ? `(from=${prev},w=${fade})` : ''}`,
    `speed=${Number(p.locomotionSpeed ?? 0).toFixed(2)}`,
    `dist=${Number(p.interactionDistance ?? 0).toFixed(2)}m`,
    `reach=${Boolean(p.reachActive)}`,
    `circumstance=${p.circumstance ?? 'calm'}`,
    `states=${asm.states.length} rules=${asm.transitions.length}]`,
  ].join(' ');
}

/**
 * Stateful Circumstance-Driven Animation State Machine controller with live condition simulator.
 */
export function createAnimStateMachineController(initialAsm) {
  let asm = normalizeAnimStateMachine(initialAsm);
  let currentStateId = asm.initialState;
  let previousStateId = null;
  let activeTransition = null;
  let crossfadeElapsedS = 0;
  let stateElapsedS = 0;
  let pendingTriggers = new Set();

  const getStateObj = (id) => asm.states.find((s) => s.id === id) ?? asm.states[0];

  const evaluateSnapshot = () => {
    const curState = getStateObj(currentStateId);
    const prevState = previousStateId ? getStateObj(previousStateId) : null;
    const fadeDurationS = activeTransition ? Math.max(0.001, activeTransition.crossfadeMs / 1000) : 0;
    const hWeight = activeTransition ? hermiteFade(crossfadeElapsedS, fadeDurationS) : 1.0;

    // Blend morphOverrides across active Hermite crossfade
    const blendedMorphs = {};
    if (prevState && hWeight < 1.0) {
      const keys = new Set([
        ...Object.keys(prevState.morphOverrides || {}),
        ...Object.keys(curState.morphOverrides || {}),
      ]);
      for (const k of keys) {
        const a = Number(prevState.morphOverrides?.[k] ?? 0);
        const b = Number(curState.morphOverrides?.[k] ?? 0);
        blendedMorphs[k] = Number((a * (1 - hWeight) + b * hWeight).toFixed(3));
      }
    } else {
      Object.assign(blendedMorphs, curState.morphOverrides || {});
    }

    return {
      currentState: currentStateId,
      previousState: previousStateId,
      clipId: curState.clipId,
      fromClipId: prevState && hWeight < 1.0 ? prevState.clipId : null,
      blendSpaceId: curState.blendSpaceId,
      blendParam: asm.parameters.locomotionSpeed ?? 0,
      crossfadeMs: activeTransition ? activeTransition.crossfadeMs : 220,
      crossfadeElapsedS,
      hermiteWeight: hWeight,
      timeS: stateElapsedS,
      morphOverrides: blendedMorphs,
      lookAtEnabled: curState.lookAtEnabled,
      ikEnabled: curState.ikEnabled,
      parameters: { ...asm.parameters },
      activeTransitionId: activeTransition?.id ?? null,
    };
  };

  const evaluateTransitionsNow = () => {
    const match = selectMatchingTransition(asm, currentStateId, asm.parameters, [...pendingTriggers]);
    pendingTriggers.clear();
    if (match) {
      previousStateId = currentStateId;
      currentStateId = match.to;
      activeTransition = match;
      crossfadeElapsedS = 0;
      stateElapsedS = 0;
      return match;
    }
    return null;
  };

  return {
    getDefinition: () => structuredClone(asm),
    setDefinition(nextAsm) {
      asm = normalizeAnimStateMachine(nextAsm);
      if (!asm.states.some((s) => s.id === currentStateId)) {
        currentStateId = asm.initialState;
      }
      return evaluateSnapshot();
    },
    getState: () => evaluateSnapshot(),
    setParameter(paramName, value) {
      asm.parameters[paramName] = value;
      evaluateTransitionsNow();
      return evaluateSnapshot();
    },
    setParameters(patch = {}) {
      Object.assign(asm.parameters, patch);
      evaluateTransitionsNow();
      return evaluateSnapshot();
    },
    fireTrigger(triggerName) {
      pendingTriggers.add(String(triggerName));
      evaluateTransitionsNow();
      return evaluateSnapshot();
    },
    forceState(targetStateId, crossfadeMs = 200) {
      if (!asm.states.some((s) => s.id === targetStateId)) return evaluateSnapshot();
      if (targetStateId !== currentStateId) {
        previousStateId = currentStateId;
        currentStateId = targetStateId;
        activeTransition = { id: `manual_${targetStateId}`, from: previousStateId, to: targetStateId, crossfadeMs };
        crossfadeElapsedS = 0;
        stateElapsedS = 0;
      }
      return evaluateSnapshot();
    },
    step(dtSeconds = 1 / 60) {
      const dt = Math.max(0, Number(dtSeconds) || 0);
      evaluateTransitionsNow();
      const curState = getStateObj(currentStateId);
      stateElapsedS += dt * (curState.speed ?? 1.0);
      if (activeTransition) {
        crossfadeElapsedS += dt;
        const fadeDurS = Math.max(0.001, activeTransition.crossfadeMs / 1000);
        if (crossfadeElapsedS >= fadeDurS) {
          activeTransition = null;
          previousStateId = null;
          crossfadeElapsedS = 0;
        }
      }
      return evaluateSnapshot();
    },
    toSemanticSubgraph: () => asmToSemanticSubgraph(asm, evaluateSnapshot()),
    summarizeForAi: () => summarizeAsmForAi(asm, evaluateSnapshot()),
  };
}
