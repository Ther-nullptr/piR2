const test = require("node:test");
const assert = require("node:assert/strict");
const M = require("./model.js");
const S = require("./scenarios.js");

test("paper repeat-last keeps the original image and state ages despite newer VLM features", () => {
  const r = M.simulatePolicyQueues({
    executionPolicy: "paper",
    delayMode: "fixed",
    fixedDelay: 1,
    actionMs: 120,
    vlmMs: 20,
    seconds: 0.45,
  });
  const ages = S.simulationAoi(r),
    at = 325;
  const repeated = r.executions.find((e) => e.timeMs === 300);
  assert.equal(repeated.executionMode, "repeat-last");
  assert.equal(repeated.origin.requestId, 1);
  const source = r.actionRequests[repeated.origin.requestId];
  assert.equal(source.featureCaptureMs, 100);
  assert.equal(source.stateCaptureMs, 100);
  assert.equal(S.ageAt(ages.executionImage, at), 225);
  assert.equal(S.ageAt(ages.executionState, at), 225);
  assert.equal(S.ageAt(ages.cache, at), 25);
  assert.equal(S.ageAt(ages.executionImage, 75), null);
  assert.equal(S.ageAt(ages.executionState, 75), null);
  assert.equal(r.actionRequests[1].outputs[0].executedAtMs, 250);
});

test("GPU-only speedup preserves fixed host cost and explicit conversion cost", () => {
  const base = {
    ...M.queueDefaults,
    actionMs: 100,
    gpuShareA: 0.6,
    vlmMs: 200,
    gpuShareV: 0.8,
  };
  const before = structuredClone(base);
  const out = S.projectScenario(base, {
    actionSpeedup: 2,
    visionSpeedup: 2,
    actionOverheadMs: 5,
    visionOverheadMs: 10,
  });
  assert.equal(out.config.actionMs, 75);
  assert.equal(out.config.gpuShareA, 0.4);
  assert.equal(out.config.vlmMs, 130);
  assert.equal(out.config.gpuShareV, 80 / 130);
  assert.deepEqual(base, before);
});
test("precision names alone never imply a numerical speedup", () => {
  const base = { ...M.queueDefaults };
  const out = S.projectScenario(base, {
    visionPrecision: "INT4",
    actionPrecision: "INT8",
  });
  assert.equal(out.config.vlmMs, base.vlmMs);
  assert.equal(out.config.actionMs, base.actionMs);
  assert.throws(
    () => S.projectScenario(base, { visionSpeedup: 0 }),
    RangeError,
  );
});
test("four states use declared latency budgets, not adaptive d or success", () => {
  assert.equal(S.classify(80, 40, 100, 50).id, "both_met");
  assert.equal(S.classify(120, 40, 100, 50).id, "vision_late");
  assert.equal(S.classify(80, 80, 100, 50).id, "action_late");
  assert.equal(S.classify(120, 80, 100, 50).id, "both_late");
  assert.equal(S.classify(null, 40, 100, 50).id, "unknown");
});
test("execution AoI follows actual producing feature, never latest cache", () => {
  const r = M.simulatePolicyQueues({
    actionMs: 20,
    vlmMs: 20,
    periodMs: 50,
    cameraHz: 20,
    seconds: 0.3,
    delayMode: "fixed",
    fixedDelay: 1,
  });
  const a = S.simulationAoi(r);
  assert.equal(S.ageAt(a.cache, 100), 50);
  assert.equal(S.ageAt(a.executionImage, 100), 100);
  assert.equal(S.ageAt(a.executionState, 100), 50);
  assert.equal(S.ageAt(a.executionImage, 50), null); // initial-feature input is unknown
  assert.equal(S.ageAt(a.executionState, 0), null); // bootstrap actions predate recording
});
test("fallback breaks action AoI; time statistics do not fill gaps with zero", () => {
  const r = M.simulatePolicyQueues({
    actionMs: 100,
    vlmMs: 20,
    periodMs: 50,
    cameraHz: 20,
    seconds: 0.4,
    delayMode: "fixed",
    fixedDelay: 1,
    bootstrapSlots: 1,
  });
  const a = S.simulationAoi(r);
  assert.equal(S.ageAt(a.executionImage, 150), null);
  assert.equal(
    S.ageStatistics([{ startMs: 0, endMs: 100, generationMs: 0 }]).meanMs,
    50,
  );
  assert.equal(S.ageStatistics([]).meanMs, null);
});
test("local aliases retain actual VLM parent and label older sources as pre", () => {
  const r = M.simulatePolicyQueues({
    actionMs: 10,
    vlmMs: 60,
    periodMs: 50,
    cameraHz: 20,
    seconds: 0.5,
    delayMode: "fixed",
    fixedDelay: 1,
  });
  const names = S.dependencyAliases(r, 100, 300);
  const first = r.visualJobs.find((v) => v.startMs < 300 && v.finishMs > 100);
  assert.equal(names.vision(first.id), "S2-00");
  for (const job of r.actionRequests) {
    const name = names.action(job);
    assert.equal(name.parentFeature, job.featureVersion);
    if (job.featureVersion >= 0)
      assert.ok(name.label.includes(names.parent(job.featureVersion)));
  }
});
