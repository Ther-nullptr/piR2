const test = require("node:test");
const assert = require("node:assert/strict");
const M = require("./model.js");

test("paper policy consumes only the seeded clean segment then adopts d clean outputs", () => {
  const r = M.simulatePolicyQueues({
    executionPolicy: "paper",
    delayMode: "fixed",
    fixedDelay: 2,
    actionMs: 80,
    seconds: 0.4,
  });
  assert.equal(r.bootstrap.length, 2);
  assert.deepEqual(
    r.executions.slice(0, 6).map((e) => e.executionMode),
    [
      "bootstrap",
      "bootstrap",
      "new-segment",
      "continue-segment",
      "new-segment",
      "continue-segment",
    ],
  );
  assert.deepEqual(
    r.actionRequests.slice(0, 3).map((x) => x.startMs),
    [0, 80, 160],
  );
  assert.deepEqual(
    r.executions.slice(2, 4).map((x) => [x.origin.requestId, x.origin.offset]),
    [
      [0, 0],
      [0, 1],
    ],
  );
  assert.equal(r.metrics.expiredOutputs, 0);
  assert.ok(r.executions.every((e) => !e.repeated));
});

test("paper lateness repeats the last emitted clean output without walking into the tail", () => {
  const r = M.simulatePolicyQueues({
    executionPolicy: "paper",
    delayMode: "fixed",
    fixedDelay: 1,
    actionMs: 120,
    vlmMs: 20,
    seconds: 0.45,
  });
  assert.equal(r.bootstrap.length, 1);
  assert.equal(r.executions[3].executionMode, "new-segment");
  const repeated = r.executions[4];
  assert.equal(repeated.executionMode, "repeat-last");
  assert.equal(repeated.repeated, true);
  assert.equal(repeated.origin.requestId, 0);
  assert.equal(repeated.origin.offset, 0);
  assert.equal(repeated.origin.targetTick, 4);
  assert.equal(repeated.origin.producerTargetTick, 3);
  assert.equal(r.actionRequests[0].outputs[0].executedAtMs, 150);
  assert.equal(r.metrics.expiredOutputs, 0);
  assert.equal(r.metrics.fallbackTicks, 0);
  assert.ok(r.metrics.repeatTicks > 0);
  assert.ok(
    r.executions
      .filter((x) => x.origin.kind === "request")
      .every((e) => e.origin.offset < r.actionRequests[e.origin.requestId].d),
  );
});

test("paper rolling full-request mean raises and lowers d at worker admission", () => {
  const r = M.simulatePolicyQueues({
    executionPolicy: "paper",
    delayWindow: 2,
    paperInitialDelay: 1,
    actionMs: 40,
    startupActionDelayMs: 100,
    periodMs: 50,
    marginMs: 250,
    seconds: 0.4,
  });
  assert.equal(r.initialDelay, 1);
  assert.deepEqual(
    r.actionRequests.slice(0, 4).map((a) => a.d),
    [1, 3, 2, 1],
  );
  assert.deepEqual(
    r.actionRequests.slice(0, 4).map((a) => a.startMs),
    [0, 140, 180, 220],
  );
  assert.deepEqual(
    r.actionRequests.slice(0, 4).map((a) => a.stateCaptureMs),
    [0, 100, 150, 200],
  );
  assert.equal(r.calibration.scope, "paper-reference-initial-seed");
  assert.equal(r.finalDelay, 1);
  const changes = r.events.filter((e) => e.kind === "delay");
  assert.deepEqual(
    changes.map((e) => e.delay),
    [3, 2, 1],
  );
  assert.equal(M.policyQueueSnapshot(r, 139).delay, 1);
  assert.equal(M.policyQueueSnapshot(r, 140).delay, 3);
});

test("paper delay uses round-to-even and records exceeding the simulator guard", () => {
  for (const [actionMs, expected] of [
    [125, 2],
    [175, 4],
  ]) {
    const r = M.simulatePolicyQueues({
      executionPolicy: "paper",
      actionMs,
      seconds: 0.5,
    });
    assert.equal(r.actionRequests[1].d, expected);
  }
  const r = M.simulatePolicyQueues({
    executionPolicy: "paper",
    actionMs: 300,
    maxDelay: 2,
    seconds: 0.8,
  });
  assert.equal(r.actionRequests[1].d, 2);
  assert.ok(r.metrics.overBudgetRequests > 0);
  assert.ok(r.events.some((e) => e.kind === "delay" && e.requiredDelay === 6));
});

test("paper continuous worker has one latest-ready result and causal snapshots", () => {
  const r = M.simulatePolicyQueues({
    executionPolicy: "paper",
    delayMode: "fixed",
    fixedDelay: 2,
    actionMs: 20,
    seconds: 0.2,
  });
  const before = M.policyQueueSnapshot(r, 35);
  assert.equal(before.action.ready.id, 0);
  assert.equal(before.action.running.id, 1);
  assert.ok(before.action.ready.outputs.every((o) => o.targetTick === null));
  assert.ok(
    before.action.ready.outputs.every(
      (o) => !o.unused && o.discardedAtMs == null,
    ),
  );
  assert.equal(r.actionRequests[0].discardedAtMs, 40);
  assert.equal(r.actionRequests[3].discardedAtMs, 100);
  assert.equal(r.actionRequests[4].publication.atMs, 100);
  assert.deepEqual(
    r.actionRequests[4].outputs.map((o) => o.targetTick),
    [2, 3],
  );
  assert.ok(
    M.policyQueueSnapshot(r, 99).action.running.outputs.every(
      (o) => o.targetTick === null,
    ),
  );
  const adopted = M.policyQueueSnapshot(r, 100);
  assert.equal(adopted.action.running.id, 5);
  assert.equal(adopted.action.ready, null);
  assert.equal(adopted.execution.activeSegment.requestId, 4);
  assert.equal(adopted.execution.activeSegment.remaining, 1);
  assert.equal(adopted.execution.lastExecuted.origin.requestId, 4);
  assert.ok(r.actionRequests.every((a) => a.startMs < r.windowEndMs));
  assert.equal(r.metrics.expiredOutputs, 0);
});

test("paper unadopted latest-ready and drained output are censored rather than expired", () => {
  const r = M.simulatePolicyQueues({
    executionPolicy: "paper",
    delayMode: "fixed",
    fixedDelay: 2,
    actionMs: 40,
    cameraHz: 0,
    seconds: 0.06,
  });
  assert.equal(r.actionRequests.length, 2);
  assert.equal(r.drainEndMs, 80);
  assert.equal(r.metrics.expiredOutputs, 0);
  for (const a of r.actionRequests)
    for (const o of a.outputs) {
      assert.equal(o.targetTick, null);
      assert.equal(o.status, "window-ended");
      assert.equal(o.unused, false);
      assert.equal(o.unusedAtMs, null);
    }
  const s = M.policyQueueSnapshot(r, 55);
  assert.equal(s.action.ready.id, 0);
  assert.equal(s.action.running.id, 1);
  assert.equal(s.action.running.finishMs, null);
});

test("paper empty cold start is unknown fallback, then holds a real producer", () => {
  const r = M.simulatePolicyQueues({
    executionPolicy: "paper",
    bootstrapSlots: 0,
    delayMode: "fixed",
    fixedDelay: 1,
    actionMs: 120,
    seconds: 0.3,
  });
  assert.deepEqual(
    r.executions.slice(0, 3).map((e) => e.executionMode),
    ["fallback", "fallback", "fallback"],
  );
  assert.equal(r.executions[3].executionMode, "new-segment");
  assert.equal(r.executions[4].executionMode, "repeat-last");
  assert.equal(r.metrics.fallbackTicks, 3);
});

test("paper policy validates execution parameters", () => {
  for (const c of [
    { executionPolicy: "unknown" },
    { delayWindow: 0 },
    { delayWindow: 1.2 },
    { paperInitialDelay: 0 },
    { paperInitialDelay: 3, maxDelay: 2 },
  ])
    assert.throws(() => M.simulatePolicyQueues(c), RangeError);
});

test("paper continuous resource scheduling conserves work without duplicate admission", () => {
  for (const computeMode of ["independent", "shared", "serial"])
    for (const hostMode of ["independent", "shared"])
      for (const gpuShare of [0, 0.6, 1]) {
        const r = M.simulatePolicyQueues({
          executionPolicy: "paper",
          actionMs: 43,
          vlmMs: 71,
          startupActionDelayMs: 7,
          delayWindow: 3,
          jitter: 0.2,
          seconds: 0.7,
          computeMode,
          hostMode,
          gpuShareA: gpuShare,
          gpuShareV: gpuShare,
        });
        for (const job of [...r.actionRequests, ...r.visualJobs]) {
          closeTo(
            job.segments.reduce((s, x) => s + x.workMs, 0),
            job.gpuWorkMs,
          );
          closeTo(
            job.segments.reduce((s, x) => s + x.hostWorkMs, 0),
            job.hostWorkMs,
          );
          closeTo(
            job.responseMs,
            job.soloMs +
              job.dispatchOverheadMs +
              job.hostWaitMs +
              job.gpuWaitMs +
              job.contentionDelayMs,
          );
          assert.ok(job.startMs < r.windowEndMs);
        }
        for (const [i, request] of r.actionRequests.entries()) {
          if (i) closeTo(request.startMs, r.actionRequests[i - 1].finishMs);
          assert.equal(request.outputs.length, request.d);
          if (request.publication) {
            assert.ok(request.publication.atMs >= request.finishMs);
            assert.equal(request.publication.installed, request.d);
          }
        }
        for (const e of r.executions)
          if (e.origin.kind === "request") {
            const source = r.actionRequests[e.origin.requestId];
            assert.ok(source.finishMs <= e.timeMs);
            assert.equal(e.origin.featureVersion, source.featureVersion);
            assert.ok(e.origin.offset < source.d);
            assert.equal(
              e.origin.producerTargetTick,
              source.outputs[e.origin.offset].targetTick,
            );
            if (e.repeated)
              assert.ok(
                source.outputs[e.origin.offset].executedAtMs < e.timeMs,
              );
          }
        if (computeMode === "serial") assert.equal(r.metrics.gpuOverlapMs, 0);
        if (hostMode === "shared")
          closeTo(
            r.metrics.hostBusyMs,
            r.metrics.hostBusyAMs + r.metrics.hostBusyVMs,
          );
      }
});

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
  assert.equal(r.executions[3].executionMode, "new-segment");
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

test("VLM admission cap preserves camera arrivals and selects the newest waiting frame", () => {
  const r = M.simulatePolicyQueues({
    vlmMs: 10,
    actionMs: 20,
    cameraHz: 100,
    vlmRateCapHz: 10,
    seconds: 0.35,
  });
  assert.equal(r.frames.length, 35);
  assert.deepEqual(
    r.visualJobs.map((v) => [v.captureMs, v.startMs]),
    [
      [0, 0],
      [100, 100],
      [200, 200],
      [300, 300],
    ],
  );
  assert.equal(r.metrics.replacedFrames, 30);
  closeTo(r.metrics.admittedVlmHz, 4 / 0.35);
  const at95 = M.policyQueueSnapshot(r, 95);
  assert.equal(at95.vision.running, null);
  assert.equal(at95.vision.pending.label, "O9");
  assert.equal(at95.vision.admissionReason, "rate-limit");
  assert.equal(at95.vision.nextAdmissionMs, 100);
  assert.equal(at95.vision.admittedJobs, 1);
  assert.equal(r.frameDrops.at(-1).reason, "window-ended");
});

test("rate limited vision waits for completion and respects the control cutoff", () => {
  const r = M.simulatePolicyQueues({
    vlmMs: 220,
    cameraHz: 100,
    vlmRateCapHz: 10,
    seconds: 0.4,
  });
  assert.deepEqual(
    r.visualJobs.map((v) => [v.captureMs, v.startMs]),
    [
      [0, 0],
      [220, 220],
    ],
  );
  assert.equal(M.policyQueueSnapshot(r, 150).vision.admissionReason, "running");
  assert.equal(r.drainEndMs, 440);
  const cutoff = M.simulatePolicyQueues({
    vlmMs: 10,
    vlmRateCapHz: 10,
    seconds: 0.1,
  });
  assert.equal(cutoff.visualJobs.length, 1);
  assert.equal(cutoff.frames.length, 2);
  assert.equal(
    M.policyQueueSnapshot(cutoff, 100).vision.admissionReason,
    "window-ended",
  );
});

test("shared host serializes prefixes with action winning simultaneous admission", () => {
  for (const computeMode of ["independent", "shared", "serial"]) {
    const r = M.simulatePolicyQueues({
      ...resourcePair,
      computeMode,
      hostMode: "shared",
      gpuShareA: 0.5,
      gpuShareV: 0.5,
    });
    const a = r.actionRequests[0],
      v = r.visualJobs[0];
    closeTo(a.hostStartMs, 0);
    closeTo(a.hostReadyMs, 10);
    closeTo(a.finishMs, 20);
    closeTo(v.hostStartMs, 10);
    closeTo(v.hostWaitMs, 10);
    closeTo(v.hostReadyMs, 60);
    closeTo(v.finishMs, 110);
    closeTo(v.contentionDelayMs, 0);
    closeTo(r.metrics.hostBusyMs, 60);
    assert.equal(M.policyQueueSnapshot(r, 5).vision.running.phase, "host-wait");
    assert.equal(M.policyQueueSnapshot(r, 5).vision.running.hostStartMs, null);
    closeTo(M.policyQueueSnapshot(r, 5).vision.running.hostWaitMs, 5);
    closeTo(M.policyQueueSnapshot(r, 5).action.running.hostRemainingMs, 5);
    assert.ok(
      r.resourceSegments.every(
        (s) => !(s.actionPhase === "host" && s.visionPhase === "host"),
      ),
    );
  }
});

test("a running shared host prefix is not preempted by a later action", () => {
  const r = M.simulatePolicyQueues({
    hostMode: "shared",
    actionMs: 10,
    gpuShareA: 0.5,
    vlmMs: 60,
    gpuShareV: 0,
    cameraHz: 1,
    marginMs: 0,
    seconds: 0.15,
  });
  const a = r.actionRequests[1],
    v = r.visualJobs[0];
  closeTo(v.hostStartMs, 5);
  closeTo(v.finishMs, 65);
  closeTo(a.startMs, 50);
  closeTo(a.hostStartMs, 65);
  closeTo(a.hostWaitMs, 15);
  closeTo(a.finishMs, 75);
  assert.equal(a.featureVersion, -1);
  assert.equal(a.stateCaptureMs, 50);
});

test("host and serial GPU waits are separate and admitted work drains after cutoff", () => {
  const r = M.simulatePolicyQueues({
    ...resourcePair,
    actionMs: 80,
    computeMode: "serial",
    hostMode: "shared",
    gpuShareA: 0.5,
    gpuShareV: 0.8,
    seconds: 0.05,
  });
  const a = r.actionRequests[0],
    v = r.visualJobs[0];
  closeTo(a.finishMs, 80);
  closeTo(v.hostWaitMs, 40);
  closeTo(v.hostReadyMs, 60);
  closeTo(v.gpuWaitMs, 20);
  closeTo(v.gpuStartMs, 80);
  closeTo(v.finishMs, 160);
  closeTo(v.responseMs, v.soloMs + v.hostWaitMs + v.gpuWaitMs);
  closeTo(v.contentionDelayMs, 0);
  closeTo(r.drainEndMs, 160);
  closeTo(r.metrics.hostBusyMs, 50);
  closeTo(r.metrics.hostWaitVMs, 40);
  closeTo(r.metrics.gpuWaitVMs, 20);
  assert.equal(r.visualJobs.length, 1);
  assert.equal(r.actionRequests.length, 1);
});

test("a small first dispatch delay misses one slot and permanently latches adaptive d", () => {
  const config = { actionMs: 48, marginMs: 0, cameraHz: 0, seconds: 0.5 };
  const baseline = M.simulatePolicyQueues(config);
  const r = M.simulatePolicyQueues({ ...config, startupActionDelayMs: 3 });
  assert.deepEqual(r.calibration, baseline.calibration);
  assert.equal(r.initialDelay, 1);
  assert.equal(r.finalDelay, 2);
  const first = r.actionRequests[0];
  closeTo(first.soloMs, 48);
  closeTo(first.hostMs, 0);
  closeTo(first.dispatchOverheadMs, 3);
  closeTo(first.hostWorkMs, 3);
  closeTo(first.gpuWorkMs, 48);
  closeTo(first.finishMs, 51);
  closeTo(first.contentionDelayMs, 0);
  assert.equal(first.publication.expired, 1);
  assert.equal(first.segments[0].phase, "dispatch");
  assert.equal(M.policyQueueSnapshot(r, 1).action.running.phase, "dispatch");
  assert.equal(M.policyQueueSnapshot(r, 99).delay, 1);
  assert.equal(M.policyQueueSnapshot(r, 100).delay, 2);
  assert.ok(
    r.actionRequests
      .slice(1)
      .every(
        (a) => a.d === 2 && a.dispatchOverheadMs === 0 && a.responseMs === 48,
      ),
  );
  assert.equal(baseline.metrics.expiredOutputs, 0);
});

test("dispatch retains the shared host until its ordinary prefix completes", () => {
  const r = M.simulatePolicyQueues({
    ...resourcePair,
    hostMode: "shared",
    gpuShareA: 0.5,
    gpuShareV: 0,
    startupActionDelayMs: 3,
  });
  const a = r.actionRequests[0],
    v = r.visualJobs[0];
  closeTo(a.hostReadyMs, 13);
  closeTo(a.finishMs, 23);
  closeTo(v.hostStartMs, 13);
  closeTo(v.hostWaitMs, 13);
  closeTo(v.finishMs, 113);
  assert.deepEqual(
    a.segments.map((s) => s.phase),
    ["dispatch", "host", "gpu"],
  );
});

test("admission and host configuration rejects invalid or unbounded values", () => {
  for (const config of [
    { vlmRateCapHz: -1 },
    { vlmRateCapHz: 241 },
    { vlmRateCapHz: NaN },
    { hostMode: "preemptive" },
    { startupActionDelayMs: -1 },
    { startupActionDelayMs: 1001 },
    { startupActionDelayMs: Infinity },
  ])
    assert.throws(() => M.simulatePolicyQueues(config));
});

test("host service, dispatch and GPU work remain conserved across queues and snapshots", () => {
  const hostPhase = (p) => p === "host" || p === "dispatch";
  for (let seed = 0; seed < 36; seed++) {
    const config = {
      seed,
      computeMode: ["independent", "shared", "serial"][seed % 3],
      hostMode: seed % 2 ? "independent" : "shared",
      actionMs: 13 + seed * 7,
      vlmMs: 17 + seed * 11,
      gpuShareA: (seed % 4) / 3,
      gpuShareV: (Math.floor(seed / 4) % 3) / 2,
      startupActionDelayMs: (seed % 5) * 3,
      vlmRateCapHz: seed % 3 ? 7 + seed : 0,
      cameraHz: 19 + seed,
      slowdownA: seed % 3,
      slowdownV: seed % 4,
      jitter: 0.2,
      seconds: 0.7,
    };
    const r = M.simulatePolicyQueues(config);
    assert.deepEqual(M.simulatePolicyQueues(config), r);
    for (const job of [...r.actionRequests, ...r.visualJobs]) {
      closeTo(
        job.responseMs,
        job.hostWorkMs +
          job.hostWaitMs +
          job.gpuWorkMs +
          job.gpuWaitMs +
          job.contentionDelayMs,
      );
      closeTo(job.hostWorkMs, job.hostMs + job.dispatchOverheadMs);
      closeTo(
        job.segments.reduce((total, s) => total + s.workMs, 0),
        job.gpuWorkMs,
      );
      closeTo(
        job.segments.reduce((total, s) => total + s.hostWorkMs, 0),
        job.hostWorkMs,
      );
      closeTo(
        job.segments.reduce((total, s) => total + s.dispatchWorkMs, 0),
        job.dispatchOverheadMs,
      );
      assert.ok(
        job.segments.every(
          (s) => s.endMs > s.startMs && s.workMs >= 0 && s.hostWorkMs >= 0,
        ),
      );
      assert.ok(job.startMs < r.windowEndMs);
      const at = Math.min(r.windowEndMs, job.startMs + job.responseMs / 2);
      const snapshot = M.policyQueueSnapshot(r, at);
      const visible = job.outputs
        ? snapshot.action.running
        : snapshot.vision.running;
      assert.equal(visible.id, job.id);
      closeTo(
        visible.hostRemainingMs +
          visible.segments.reduce((total, s) => total + s.hostWorkMs, 0),
        job.hostWorkMs,
      );
      closeTo(
        visible.gpuRemainingMs +
          visible.segments.reduce((total, s) => total + s.workMs, 0),
        job.gpuWorkMs,
      );
      closeTo(
        visible.elapsedMs,
        visible.segments.reduce((total, s) => total + s.endMs - s.startMs, 0),
      );
      assert.ok(visible.segments.every((s) => s.endMs <= at));
    }
    if (config.hostMode === "shared")
      assert.ok(
        r.resourceSegments.every(
          (s) => !(hostPhase(s.actionPhase) && hostPhase(s.visionPhase)),
        ),
      );
    if (config.computeMode === "serial")
      assert.ok(r.resourceSegments.every((s) => !(s.rateA > 0 && s.rateV > 0)));
    if (config.vlmRateCapHz)
      assert.ok(
        r.visualJobs
          .slice(1)
          .every(
            (job, i) =>
              job.startMs - r.visualJobs[i].startMs >=
              1000 / config.vlmRateCapHz - 1e-6,
          ),
      );
  }
});

test("default independent host prefixes preserve the existing event stream", () => {
  const r = M.simulatePolicyQueues({
    ...resourcePair,
    computeMode: "independent",
    gpuShareA: 0.5,
    gpuShareV: 0.5,
  });
  assert.deepEqual(
    r.events.filter((e) => e.timeMs === 0).map((e) => e.kind),
    ["camera", "vision-start", "reserve", "action-start", "execute"],
  );
  closeTo(r.actionRequests[0].finishMs, 20);
  closeTo(r.visualJobs[0].finishMs, 100);
});
