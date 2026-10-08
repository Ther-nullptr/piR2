const test = require("node:test");
const assert = require("node:assert/strict");
const M = require("./model.js");

test("a missed deadline exactly at control end is counted independently of future output censoring", () => {
  const r = M.simulatePolicyQueues({
    actionMs: 80,
    periodMs: 50,
    delayMode: "fixed",
    fixedDelay: 1,
    seconds: 0.05,
  });
  assert.equal(r.metrics.deadlineMisses, 1);
  assert.equal(r.metrics.expiredOutputs, 0);
  assert.equal(r.actionRequests[0].outputs[0].status, "window-ended");
});

test("πR² 80ms requests use d=2 and supply two executable 50ms slots", () => {
  for (const vlmMs of [30, 140]) {
    const r = M.simulatePolicyQueues({ actionMs: 80, vlmMs, seconds: 0.6 });
    assert.equal(r.initialDelay, 2);
    assert.equal(r.finalDelay, 2);
    assert.equal(r.metrics.expiredOutputs, 0);
    assert.equal(r.metrics.fallbackTicks, 0);
    assert.deepEqual(
      r.actionRequests.map((x) => x.requestTick),
      [0, 2, 4, 6, 8, 10],
    );
    assert.deepEqual(
      r.executions.slice(0, 4).map((x) => x.origin.label),
      ["B0", "B1", "I-1a[0]", "I-1a[1]"],
    );
    assert.equal(r.metrics.executedOutputs, 10);
    assert.ok(
      r.actionRequests
        .filter((x) => x.publication)
        .every((x) => x.publication.installed === 2),
    );
  }
});

test("fixed d=1 is a deliberate mismatch: expire outputs, then exhaust bootstrap", () => {
  const r = M.simulatePolicyQueues({
    actionMs: 80,
    delayMode: "fixed",
    fixedDelay: 1,
    seconds: 3,
  });
  assert.equal(r.initialDelay, 1);
  assert.equal(r.finalDelay, 1);
  assert.equal(r.metrics.adoptedOutputs, 0);
  assert.equal(r.metrics.expiredOutputs, 30);
  assert.equal(r.metrics.expiredInWindow, 29);
  assert.equal(r.metrics.expiredAtWindowEnd, 1);
  assert.equal(r.metrics.fallbackTicks, 20);
  assert.ok(
    r.executions.slice(0, 40).every((x) => x.origin.kind === "bootstrap"),
  );
  assert.ok(r.executions.slice(40).every((x) => x.fallback));
  assert.ok(
    r.actionRequests.filter((x) => x.publication).every((x) => x.unused),
  );
});

test("late two-slot output only loses its expired prefix", () => {
  const r = M.simulatePolicyQueues({
    actionMs: 110,
    delayMode: "fixed",
    fixedDelay: 2,
    seconds: 0.6,
  });
  const first = r.actionRequests[0];
  assert.deepEqual(first.publication, {
    atMs: 150,
    consumedAtTick: 3,
    installed: 1,
    expired: 1,
    protected: 0,
  });
  assert.equal(first.outputs[0].unused, true);
  assert.equal(first.outputs[0].unusedAtMs, 150);
  assert.equal(first.outputs[1].executedAtMs, 150);
  assert.equal(first.unused, false);
  assert.equal(r.executions[3].origin.label, "I-1a[1]");
});

test("seeded calibration underestimation adapts after a partially late result", () => {
  const config = { actionMs: 75, jitter: 0.5, seed: 2, seconds: 4 };
  const r = M.simulatePolicyQueues(config);
  assert.equal(r.initialDelay, 2);
  assert.equal(r.finalDelay, 3);
  const late = r.actionRequests.find((x) => x.publication?.expired);
  assert.equal(late.publication.expired, 1);
  assert.equal(late.publication.installed, 1);
  assert.equal(r.actionRequests[late.id + 1].d, 3);
  assert.deepEqual(M.simulatePolicyQueues(config), r);
});

test("unused VLM features do not imply unused action outputs", () => {
  const r = M.simulatePolicyQueues({ vlmMs: 30, actionMs: 80, seconds: 0.6 });
  assert.deepEqual(
    r.visualJobs.filter((x) => x.unused).map((x) => x.id),
    [0, 2, 4, 6, 8, 10],
  );
  assert.equal(r.visualJobs[0].unusedAtMs, 80);
  assert.equal(r.visualJobs[1].usedByRequestIds[0], 1);
  assert.equal(r.actionRequests[1].label, "I1a");
  assert.ok(r.actionRequests.every((x) => !x.unused));
  assert.equal(r.metrics.expiredOutputs, 0);
});

test("external camera rate retains only one latest waiting image", () => {
  const r = M.simulatePolicyQueues({
    vlmMs: 140,
    actionMs: 20,
    cameraHz: 20,
    seconds: 0.6,
  });
  assert.deepEqual(
    r.visualJobs.slice(0, 4).map((v) => [v.captureMs, v.startMs]),
    [
      [0, 0],
      [100, 140],
      [250, 280],
      [400, 420],
    ],
  );
  assert.ok(
    r.frameDrops.some(
      (x) => x.id === 1 && x.replacementId === 2 && x.atMs === 100,
    ),
  );
  const independent = M.simulatePolicyQueues({
    vlmMs: 30,
    cameraHz: 40,
    periodMs: 50,
    seconds: 0.6,
  });
  assert.equal(independent.frames.length, 24);
  assert.equal(independent.executions.length, 12);
  const s = M.policyQueueSnapshot(r, 120);
  assert.equal(s.vision.running.label, "V0");
  assert.equal(s.vision.pending.label, "O2");
});

test("every request and execution preserves feature and target-tick causality", () => {
  const r = M.simulatePolicyQueues({
    actionMs: 90,
    vlmMs: 140,
    jitter: 0.4,
    seconds: 5,
  });
  for (const q of r.actionRequests) {
    assert.ok(q.featureReadyMs <= q.startMs);
    assert.ok(q.featureCaptureMs <= q.stateCaptureMs);
    assert.equal(q.stateCaptureMs, q.requestTick * r.config.periodMs);
    assert.deepEqual(
      q.outputs.map((o) => o.targetTick),
      Array.from({ length: q.d }, (_, k) => q.requestTick + q.d + k),
    );
    assert.ok(
      q.committedPrefix.every((s, k) => s.targetTick === q.requestTick + k),
    );
    if (q.id) assert.ok(q.startMs >= r.actionRequests[q.id - 1].finishMs);
  }
  for (const e of r.executions) {
    assert.equal(e.tick, e.origin.targetTick);
    if (e.origin.kind === "request") {
      const request = r.actionRequests[e.origin.requestId];
      assert.ok(request.finishMs <= e.timeMs);
      assert.equal(request.outputs[e.origin.offset].executedAtMs, e.timeMs);
    }
  }
});

test("same-time completion precedes controller consumption and feature read", () => {
  const r = M.simulatePolicyQueues({
    actionMs: 100,
    vlmMs: 50,
    marginMs: 0,
    seconds: 0.6,
  });
  assert.equal(r.actionRequests[0].publication.atMs, 100);
  assert.equal(r.actionRequests[0].publication.expired, 0);
  assert.equal(r.executions[2].origin.requestId, 0);
  assert.equal(r.actionRequests[1].featureVersion, 1);
  const order = r.events.filter((e) => e.timeMs === 100).map((e) => e.kind);
  assert.ok(order.indexOf("action-finish") < order.indexOf("publish"));
  assert.ok(order.indexOf("vision-finish") < order.indexOf("action-start"));
});

test("window end censors outstanding results and does not invent rejection", () => {
  const r = M.simulatePolicyQueues({ actionMs: 80, vlmMs: 140, seconds: 0.1 });
  assert.equal(r.actionRequests.length, 1);
  assert.equal(r.actionRequests[0].publication, null);
  assert.ok(
    r.actionRequests[0].outputs.every(
      (o) => o.status === "window-ended" && !o.unused,
    ),
  );
  assert.equal(r.visualJobs[0].status, "window-ended");
  assert.equal(r.visualJobs[0].unused, false);
  assert.equal(r.metrics.expiredOutputs, 0);
  assert.equal(r.executions.length, 2);
  assert.equal(M.policyQueueSnapshot(r, 1000).timeMs, 100);
});

test("native snapshots distinguish computing, completed queue and committed execution", () => {
  const r = M.simulatePolicyQueues({ actionMs: 80, vlmMs: 30, seconds: 0.6 });
  const before = M.policyQueueSnapshot(r, 79);
  assert.equal(before.action.running.label, "I-1a");
  assert.equal(before.action.ready, null);
  assert.equal(before.rolling.cells[2].kind, "becoming-clean");
  const ready = M.policyQueueSnapshot(r, 80);
  assert.equal(ready.action.running, null);
  assert.equal(ready.action.ready.outputs[0].targetTick, 2);
  assert.equal(ready.rolling.origin, 2);
  const published = M.policyQueueSnapshot(r, 100);
  assert.equal(published.action.ready, null);
  assert.equal(published.action.running.label, "I1a");
  assert.equal(published.execution.lastExecuted.origin.label, "I-1a[0]");
  assert.equal(published.execution.slots[0].targetTick, 3);
  assert.equal(published.execution.slots[0].committed, true);
  assert.equal(published.execution.slots[1].committed, false);
  const earlier = M.policyQueueSnapshot(r, 30);
  assert.equal(earlier.vision.latestFeature.unused, false);
  assert.deepEqual(earlier.vision.latestFeature.usedByRequestIds, []);
});

test("empty execution slots use explicit fallback only until predictions arrive", () => {
  const r = M.simulatePolicyQueues({
    bootstrapSlots: 0,
    actionMs: 80,
    seconds: 0.6,
  });
  assert.equal(r.metrics.fallbackTicks, 2);
  assert.ok(r.executions.slice(0, 2).every((x) => x.fallback));
  assert.ok(r.executions.slice(2).every((x) => !x.fallback));
});

test("queue inputs are finite, bounded and protect browser event count", () => {
  for (const bad of [
    null,
    [],
    { actionMs: NaN },
    { vlmMs: Infinity },
    { periodMs: 0 },
    { seconds: -1 },
    { jitter: 1 },
    { cameraHz: -1 },
    { fixedDelay: 1.5 },
    { bootstrapSlots: 1.5 },
    { maxDelay: 6 },
    { delayMode: "unknown" },
    { fixedDelay: 4, maxDelay: 3 },
    { seconds: 60, cameraHz: 240, periodMs: 5 },
  ]) {
    assert.throws(() => M.simulatePolicyQueues(bad));
  }
  const r = M.simulatePolicyQueues({
    cameraHz: 0,
    actionMs: 500,
    maxDelay: 5,
    seconds: 0.6,
  });
  assert.equal(r.visualJobs.length, 0);
  assert.equal(r.calibration.overBudget, true);
  assert.throws(() => M.policyQueueSnapshot(r, NaN));
});

test("snapshots hide future frame start and cannot mutate recorded executions", () => {
  const r = M.simulatePolicyQueues({ vlmMs: 140, seconds: 0.6 });
  const s = M.policyQueueSnapshot(r, 120);
  assert.equal(s.vision.pending.startedAtMs, null);
  s.execution.lastExecuted.origin.label = "changed";
  assert.notEqual(r.executions[2].origin.label, "changed");
});

test("end drain matches controller expiration while censoring genuine future slots", () => {
  const r = M.simulatePolicyQueues({
    actionMs: 80,
    delayMode: "fixed",
    fixedDelay: 1,
    seconds: 0.1,
  });
  const last = r.actionRequests[0];
  assert.equal(last.publication, null);
  assert.equal(last.drainedAtMs, 100);
  assert.equal(last.outputs[0].status, "expired-at-window-end");
  assert.equal(last.outputs[0].unusedAtMs, 100);
  assert.equal(r.metrics.expiredInWindow, 0);
  assert.equal(r.metrics.expiredAtWindowEnd, 1);
  assert.equal(r.metrics.expiredOutputs, 1);
  assert.equal(
    M.policyQueueSnapshot(r, 99).action.ready.outputs[0].status,
    "ready",
  );
  assert.equal(M.policyQueueSnapshot(r, 100).action.ready, null);

  const late = M.simulatePolicyQueues({
    actionMs: 180,
    delayMode: "fixed",
    fixedDelay: 2,
    seconds: 0.15,
  });
  const outputs = late.actionRequests[0].outputs;
  assert.equal(outputs[0].targetTick, 2);
  assert.equal(outputs[0].unusedAtMs, 180);
  assert.equal(outputs[1].targetTick, 3);
  assert.equal(outputs[1].status, "window-ended");
  assert.equal(outputs[1].unused, false);
  const snapshot = M.policyQueueSnapshot(late, 150);
  assert.equal(snapshot.action.running.outputs[0].status, "computing");
  assert.equal(snapshot.action.running.outputs[0].unused, undefined);
});

const closeTo = (actual, expected, tolerance = 1e-6) =>
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${actual} != ${expected}`,
  );
const resourcePair = {
  computeMode: "shared",
  actionMs: 20,
  vlmMs: 100,
  slowdownA: 1,
  slowdownV: 1,
  periodMs: 200,
  cameraHz: 1,
  seconds: 0.15,
};

test("shared GPU progress restores solo speed immediately after one finishes", () => {
  const r = M.simulatePolicyQueues(resourcePair);
  const a = r.actionRequests[0],
    v = r.visualJobs[0];
  closeTo(a.finishMs, 40);
  closeTo(v.finishMs, 120);
  closeTo(a.contentionDelayMs, 20);
  closeTo(v.contentionDelayMs, 20);
  closeTo(r.metrics.gpuOverlapMs, 40);
  assert.ok(
    r.resourceSegments.some(
      (s) =>
        s.startMs === 40 &&
        s.visionId === 0 &&
        s.actionId === null &&
        s.rateV === 1,
    ),
  );
  assert.equal(M.policyQueueSnapshot(r, 15).action.running.phase, "gpu-shared");
  assert.equal(M.policyQueueSnapshot(r, 50).vision.running.phase, "gpu-solo");
  assert.equal(r.calibration.scope, "isolated-full-request");
});

test("zero slowdown reproduces independent timing including host phases and jitter", () => {
  const config = {
    actionMs: 67,
    vlmMs: 121,
    gpuShareA: 0.6,
    gpuShareV: 0.8,
    jitter: 0.3,
    seed: 987,
    seconds: 2,
  };
  const baseline = M.simulatePolicyQueues({
    ...config,
    computeMode: "independent",
    slowdownA: 0,
    slowdownV: 0,
  });
  const shared = M.simulatePolicyQueues({
    ...config,
    computeMode: "shared",
    slowdownA: 0,
    slowdownV: 0,
  });
  assert.deepEqual(shared.actionRequests, baseline.actionRequests);
  assert.deepEqual(shared.visualJobs, baseline.visualJobs);
  assert.deepEqual(shared.executions, baseline.executions);
  assert.deepEqual(shared.metrics, baseline.metrics);
});

test("host time is not inflated and GPU overlap can begin partway through a job", () => {
  const r = M.simulatePolicyQueues({ ...resourcePair, gpuShareA: 0.5 });
  const a = r.actionRequests[0],
    v = r.visualJobs[0];
  closeTo(a.hostMs, 10);
  closeTo(a.gpuWorkMs, 10);
  closeTo(a.finishMs, 30);
  closeTo(v.finishMs, 110);
  closeTo(a.overlapMs, 20);
  assert.equal(a.segments[0].phase, "host");
  assert.equal(a.segments[0].endMs, 10);
  assert.equal(M.policyQueueSnapshot(r, 5).action.running.phase, "host");
  const cpu = M.simulatePolicyQueues({ ...resourcePair, gpuShareA: 0 });
  closeTo(cpu.actionRequests[0].finishMs, 20);
  closeTo(cpu.visualJobs[0].finishMs, 100);
  closeTo(cpu.metrics.gpuOverlapMs, 0);
  closeTo(cpu.actionRequests[0].contentionDelayMs, 0);
  assert.equal(cpu.actionRequests[0].gpuStartMs, null);
});

test("asymmetric slowdown and simultaneous completion conserve GPU work", () => {
  const asymmetric = M.simulatePolicyQueues({ ...resourcePair, slowdownA: 3 });
  closeTo(asymmetric.actionRequests[0].finishMs, 80);
  closeTo(asymmetric.visualJobs[0].finishMs, 140);
  const simultaneous = M.simulatePolicyQueues({ ...resourcePair, vlmMs: 20 });
  closeTo(simultaneous.actionRequests[0].finishMs, 40);
  closeTo(simultaneous.visualJobs[0].finishMs, 40);
  const at = M.policyQueueSnapshot(simultaneous, 40);
  assert.equal(at.resource.rateA, 0);
  assert.equal(at.resource.rateV, 0);
  assert.equal(at.action.ready.phase, "complete");
  const coincidentTick = M.simulatePolicyQueues({
    ...resourcePair,
    vlmMs: 20,
    periodMs: 40,
    seconds: 0.12,
  });
  assert.equal(coincidentTick.actionRequests[0].publication.expired, 0);
  assert.equal(coincidentTick.executions[1].origin.requestId, 0);
  assert.equal(coincidentTick.actionRequests[1].featureVersion, 0);
});

test("serial GPU scheduling is non-preemptive and prefers ready action at boundaries", () => {
  const r = M.simulatePolicyQueues({ ...resourcePair, computeMode: "serial" });
  closeTo(r.actionRequests[0].gpuStartMs, 0);
  closeTo(r.actionRequests[0].finishMs, 20);
  closeTo(r.visualJobs[0].gpuStartMs, 20);
  closeTo(r.visualJobs[0].gpuWaitMs, 20);
  closeTo(r.visualJobs[0].finishMs, 120);
  assert.ok(r.resourceSegments.every((s) => !(s.rateA > 0 && s.rateV > 0)));
  assert.equal(M.policyQueueSnapshot(r, 10).vision.running.phase, "gpu-wait");
  const waiting = M.simulatePolicyQueues({
    computeMode: "serial",
    actionMs: 70,
    gpuShareA: 0.5,
    vlmMs: 50,
    cameraHz: 40,
    seconds: 0.15,
  });
  closeTo(waiting.actionRequests[0].gpuStartMs, 50);
  closeTo(waiting.actionRequests[0].finishMs, 85);
  closeTo(waiting.visualJobs[1].gpuStartMs, 85);
});

test("GPU queue wait never refreshes the admitted request's state or feature", () => {
  const r = M.simulatePolicyQueues({
    ...resourcePair,
    computeMode: "serial",
    actionMs: 100,
    gpuShareA: 0.8,
    vlmMs: 140,
    seconds: 0.3,
  });
  const a = r.actionRequests[0];
  closeTo(a.gpuStartMs, 140);
  closeTo(a.gpuWaitMs, 120);
  closeTo(a.finishMs, 220);
  assert.equal(a.featureVersion, -1);
  assert.equal(a.stateCaptureMs, 0);
  const snapshot = M.policyQueueSnapshot(r, 150);
  assert.equal(snapshot.vision.latestFeature.id, 0);
  assert.equal(snapshot.action.running.featureVersion, -1);
});

test("renewed overlap delays unfinished work and adapts d from actual response", () => {
  const r = M.simulatePolicyQueues({
    ...resourcePair,
    actionMs: 80,
    vlmMs: 140,
    periodMs: 50,
    seconds: 0.25,
  });
  assert.equal(r.initialDelay, 2);
  assert.equal(M.policyQueueSnapshot(r, 175).delay, 2);
  assert.equal(M.policyQueueSnapshot(r, 175).action.ready.id, 0);
  closeTo(r.actionRequests[0].finishMs, 160);
  assert.equal(r.actionRequests[0].publication.expired, 2);
  assert.equal(r.actionRequests[1].d, 4);
  assert.equal(r.actionRequests[1].startMs, 200);
  closeTo(r.visualJobs[0].finishMs, 240);
  closeTo(r.actionRequests[1].finishMs, 300);
  assert.equal(r.finalDelay, 4);
  assert.equal(r.drainEndMs, 300);
});

test("window draining finishes admitted jobs without new arrivals or phantom contention", () => {
  const r = M.simulatePolicyQueues({
    ...resourcePair,
    actionMs: 80,
    vlmMs: 140,
    seconds: 0.05,
  });
  assert.equal(r.actionRequests.length, 1);
  assert.equal(r.visualJobs.length, 1);
  assert.equal(r.frames.length, 1);
  closeTo(r.actionRequests[0].finishMs, 160);
  closeTo(r.visualJobs[0].finishMs, 220);
  closeTo(r.drainEndMs, 220);
  closeTo(r.metrics.gpuOverlapMs, 50);
  closeTo(r.actionRequests[0].overlapMs, 160);
  assert.ok(
    r.resourceSegments
      .filter((s) => s.startMs >= 160)
      .every((s) => s.actionId === null && s.rateV === 1),
  );
  assert.ok(r.actionRequests[0].outputs.every((o) => !o.unused));
  const serial = M.simulatePolicyQueues({
    ...resourcePair,
    computeMode: "serial",
    seconds: 0.05,
  });
  closeTo(serial.visualJobs[0].finishMs, 120);
  closeTo(serial.drainEndMs, 120);
  assert.equal(serial.visualJobs.length, 1);
});

test("snapshots clip resource accounting and expose the actual active phase", () => {
  const r = M.simulatePolicyQueues(resourcePair);
  const at10 = M.policyQueueSnapshot(r, 10);
  closeTo(at10.action.running.contentionDelayMs, 5);
  closeTo(at10.action.running.overlapMs, 10);
  assert.equal(at10.action.running.responseMs, null);
  assert.ok(at10.action.running.segments.every((s) => s.endMs <= 10));
  at10.action.running.segments[0].rate = 999;
  assert.equal(r.actionRequests[0].segments[0].rate, 0.5);
});

test("bounded deterministic workloads conserve per-job work and output ownership", () => {
  for (let seed = 0; seed < 24; seed++) {
    const computeMode = ["independent", "shared", "serial"][seed % 3];
    const config = {
      computeMode,
      seed,
      actionMs: 10 + seed * 9,
      vlmMs: 17 + seed * 11,
      slowdownA: (seed % 5) / 2,
      slowdownV: (seed % 4) / 2,
      gpuShareA: (seed % 5) / 4,
      gpuShareV: (seed % 4) / 3,
      cameraHz: 7 + seed,
      periodMs: 20 + (seed % 4) * 10,
      jitter: 0.25,
      seconds: 0.6,
    };
    const r = M.simulatePolicyQueues(config);
    assert.deepEqual(M.simulatePolicyQueues(config), r);
    for (const job of [...r.actionRequests, ...r.visualJobs]) {
      closeTo(job.finishMs - job.startMs, job.responseMs);
      closeTo(
        job.responseMs,
        job.hostMs + job.gpuWorkMs + job.gpuWaitMs + job.contentionDelayMs,
      );
      closeTo(
        job.segments.reduce((sum, s) => sum + s.workMs, 0),
        job.gpuWorkMs,
      );
      assert.ok(job.segments.every((s) => s.endMs > s.startMs));
    }
    if (computeMode === "serial")
      assert.ok(r.resourceSegments.every((s) => !s.overlap));
    assert.ok(r.metrics.gpuOverlapMs <= r.windowEndMs + 1e-6);
    for (const q of r.actionRequests) {
      assert.ok(q.featureReadyMs <= q.startMs + 1e-6);
      for (const o of q.outputs)
        if (o.executedAtMs !== null) {
          assert.ok(o.executedAtMs >= q.finishMs - 1e-6);
          assert.equal(o.executedAtMs, o.targetTick * r.config.periodMs);
        }
    }
    assert.ok(r.actionRequests.every((q) => q.startMs < r.windowEndMs));
    assert.ok(r.visualJobs.every((q) => q.startMs < r.windowEndMs));
  }
});

test("resource configuration rejects invalid or unbounded values", () => {
  for (const config of [
    { computeMode: "preemptive" },
    { slowdownA: -1 },
    { slowdownV: Infinity },
    { slowdownA: 21 },
    { gpuShareA: -0.1 },
    { gpuShareV: 1.1 },
    { gpuShareA: NaN },
  ])
    assert.throws(() => M.simulatePolicyQueues(config));
});
