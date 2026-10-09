const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const M = fs.existsSync(__dirname + "/measured-model.js")
  ? require("./measured-model.js")
  : {};
const episode = {
  duration_s: 1,
  period_s: 0.05,
  initial_delay: 1,
  initial_capture_s: -0.2,
  initial_feature: { feature_sequence: 0, feature_capture_s: -0.2 },
  feature_updates: [
    { t: 0.2, feature_sequence: 8, capture_s: 0.05 },
    { t: 0.7, feature_sequence: 9, capture_s: 0.6 },
  ],
  requests: [
    {
      role: "S2",
      start_s: 0.05,
      end_s: 0.18,
      completed_s: 0.18,
      published_s: 0.2,
      feature_sequence: 8,
    },
    {
      role: "S1",
      request_tick: 0,
      start_s: 0.1,
      end_s: 0.3,
      completed_s: 0.3,
      adopted_s: 0.4,
      use_s: 0.12,
      feature_sequence: 0,
      feature_capture_s: -0.2,
      state_capture_s: 0.08,
    },
    {
      role: "S1",
      request_tick: 4,
      start_s: 0.25,
      end_s: 0.35,
      completed_s: 0.35,
      use_s: 0.26,
      feature_sequence: 8,
      feature_capture_s: 0.05,
      state_capture_s: 0.24,
    },
    {
      role: "S1",
      request_tick: 6,
      start_s: 0.4,
      end_s: 0.5,
      completed_s: 0.5,
      use_s: 0.41,
      feature_sequence: 8,
      feature_capture_s: 0.05,
      state_capture_s: 0.39,
    },
  ],
  ticks: [
    {
      t: 0,
      tick: 0,
      executed_slot: { producer_kind: "bootstrap" },
      action_buffer: { slots: [] },
    },
    {
      t: 0.4,
      tick: 8,
      executed_slot: { producer_kind: "request", producer_request_tick: 0 },
      action_buffer: { slots: [] },
    },
    {
      t: 0.8,
      tick: 16,
      fallback: true,
      executed_slot: { producer_kind: "fallback" },
      action_buffer: { slots: [] },
    },
  ],
  camera_events: [
    { t: 0.3, camera_queue: { waiting_tick: 6 } },
    { t: 0.9, camera_queue: { waiting_tick: 18 } },
  ],
  delay_transitions: [{ t: 0.6, before: 1, after: 2 }],
};
test("dependency labels use authoritative feature sequence and repeated children", () => {
  assert.equal(typeof M.dependencies, "function");
  const d = M.dependencies(episode, 0, 1);
  assert.equal(d.parents.get(8).label, "S2-00");
  assert.equal(d.parents.get(0).label, "S2-pre01");
  assert.equal(d.actions.get(0).label, "S1[pre01].1");
  assert.equal(d.actions.get(4).label, "S1[00].1");
  assert.equal(d.actions.get(6).label, "S1[00].2");
});
test("current executed AoI follows producer request even after newer cache publication", () => {
  assert.equal(typeof M.stateAt, "function");
  const s = M.stateAt(episode, 0.75);
  assert.equal(s.cacheAgeMs, 150.00000000000003);
  assert.equal(s.imageAgeMs, 950);
  assert.equal(s.stateAgeMs, 670);
  assert.equal(s.executedRequest.request_tick, 0);
  assert.equal(s.camera.waiting_tick, 6);
  assert.equal(s.delay, 2);
  assert.equal(M.stateAt(episode, 0.5).delay, 1);
});
test("bootstrap uses original negative capture time and fallback stays unknown", () => {
  assert.equal(typeof M.stateAt, "function");
  assert.equal(M.stateAt(episode, 0.1).imageAgeMs, 300.00000000000006);
  assert.equal(M.stateAt(episode, 0.85).imageAgeMs, null);
  assert.equal(M.stateAt(episode, 0.85).stateAgeMs, null);
  assert.equal(
    M.stateAt({ ...episode, initial_capture_s: null, initial_feature: {} }, 0)
      .cacheAgeMs,
    null,
  );
});
test("ready and executing phases never leak a future completion or tick", () => {
  assert.equal(typeof M.stateAt, "function");
  const s = M.stateAt(episode, 0.2);
  assert.equal(s.inflight.length, 1);
  assert.equal(s.ready.length, 0);
  assert.equal(s.tick.tick, 0);
  assert.equal(s.camera.waiting_tick, undefined);
  assert.equal(M.stateAt(episode, 0.32).ready[0].request_tick, 0);
});
test("AoI series represent jumps and unknown fallback without zero substitution", () => {
  assert.equal(typeof M.aoiSeries, "function");
  const curves = M.aoiSeries(episode, 0, 1);
  assert.equal(curves.image.at(-1)[1], null);
  assert.equal(curves.cache.filter((p) => p[0] === 0.2).length, 2);
  assert.equal(curves.used[1].imageMs, 210.00000000000003);
});
test("calibration message uses solo means and exports only concise hardware fields", () => {
  assert.equal(typeof M.calibrationMessage, "function");
  const c = {
    id: "dual/example",
    layout: "dual",
    summary: { s1_solo_mean_ms: 20, s2_solo_mean_ms: 70, s1_rpc_mean_ms: 100 },
    hardware: { renderer_gpu: 3, command: ["/private/path"] },
  };
  const m = M.calibrationMessage(c, episode);
  assert.deepEqual(m.solo, { actionMs: 20, vlmMs: 70 });
  assert.equal(m.type, "pir2:load-solo-calibration");
  assert.equal(m.hardware.command, undefined);
});

test("unknown bootstrap image capture never inherits the state timestamp", () => {
  const missing = structuredClone(episode);
  missing.initial_feature.feature_capture_s = null;
  const snapshot = M.stateAt(missing, 0.05);
  assert.equal(snapshot.imageAgeMs, null);
  assert.equal(snapshot.stateAgeMs, 250);
});
