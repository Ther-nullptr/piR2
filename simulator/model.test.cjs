const test = require("node:test");
const assert = require("node:assert/strict");
const M = require("./model.js");

test("zero-wait deterministic updates have mean AoI = 1.5 service", () => {
  const r = M.simulate({
    ...M.defaults,
    seconds: 10,
    burnMs: 1000,
    fastEnabled: false,
    source: "zero",
    slowCap: 0,
    jitter: 0,
    kA: 0,
    kV: 0,
    serviceOverride: [5, 20],
  });
  assert.ok(Math.abs(r.aoiMean - 30) < 1e-6, r.aoiMean);
});
test("uncontended periodic S1 uses response time and meets its deadline", () => {
  const r = M.simulate({
    ...M.defaults,
    seconds: 2,
    burnMs: 0,
    cameraHz: 0,
    jitter: 0,
    serviceOverride: [10, 20],
    period: 40,
  });
  assert.equal(r.planned, 50);
  assert.equal(r.completed, 50);
  assert.equal(r.dropped, 0);
  assert.equal(r.misses, 0);
  assert.equal(r.responseP95, 10);
});
test("busy-drop counts conserve every S1 release, including final completion", () => {
  const r = M.simulate({
    ...M.defaults,
    seconds: 0.2,
    burnMs: 0,
    cameraHz: 0,
    jitter: 0,
    serviceOverride: [50, 20],
    period: 20,
  });
  assert.equal(r.planned, 10);
  assert.equal(r.completed + r.dropped, 10);
  assert.equal(r.misses, 10);
});
test("FIFO response includes time spent waiting behind a running S2 job", () => {
  const r = M.simulate({
    ...M.defaults,
    seconds: 0.4,
    burnMs: 0,
    cameraHz: 100,
    fastQueue: "fifo",
    policy: "serial",
    jitter: 0,
    serviceOverride: [5, 70],
    period: 40,
  });
  assert.ok(r.calls.some((x) => x.wait > 0 && x.response > x.service));
  assert.equal(r.completed + r.dropped, r.planned);
});
test("keep latest queued camera frame is fresher than dropping new frames", () => {
  const c = {
    ...M.defaults,
    seconds: 4,
    burnMs: 500,
    fastEnabled: false,
    cameraHz: 100,
    jitter: 0,
    serviceOverride: [5, 70],
  };
  const latest = M.simulate({ ...c, slowQueue: "latest" });
  const oldest = M.simulate({ ...c, slowQueue: "dropnew" });
  assert.ok(
    latest.aoiMean < oldest.aoiMean,
    `${latest.aoiMean}, ${oldest.aoiMean}`,
  );
});
test("roofline distinguishes weight-only from activation quantization", () => {
  const c = { ...M.defaults, ratioV: 4, overhead: 0 };
  assert.equal(
    M.serviceModel({ ...c, qV: "w4" }).S2.gpu,
    M.serviceModel({ ...c, qV: "bf16" }).S2.gpu,
  );
  assert.ok(
    M.serviceModel({ ...c, qV: "int8" }).S2.total <
      M.serviceModel({ ...c, qV: "bf16" }).S2.total,
  );
});
test("biased iterative correction need not approach zero error", () => {
  const rows = M.correctionError({
    alpha: 0.8,
    bias: 0.04,
    sigma: 0,
    rho: 0,
    initial: 0,
    steps: 100,
  });
  assert.ok(Math.abs(rows.at(-1).rmse - 0.2) < 1e-8);
});
test("disabled S1 has an undefined miss rate, not perfect deadline performance", () => {
  const r = M.simulate({ ...M.defaults, fastEnabled: false });
  assert.equal(r.planned, 0);
  assert.equal(r.missRate, null);
});
test("timeline reaches the end in high-throughput zero-wait settings", () => {
  const r = M.simulate({
    ...M.defaults,
    seconds: 30,
    source: "zero",
    hostScale: 0,
    compute: 4,
    bandwidth: 4,
    lowbitMath: 4,
    qA: "int8",
    qV: "int8",
    period: 10,
    jitter: 0,
  });
  assert.equal(r.timeline.at(-1).end, 30000);
});
