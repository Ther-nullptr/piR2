const test = require("node:test");
const assert = require("node:assert/strict");
let R;
try {
  R = require("./replay-model.js");
} catch {
  R = {};
}
const clean = {
  id: "a",
  scope: "prepared_model",
  comparison_key: "same",
  observation_id: "o",
  seed: 7,
  wall_ms: 3,
  clean_wall: true,
  instrumented: false,
  output: { status: "recorded" },
};

test("control intervals separate measured execution from pacing and physical time", () => {
  const episode = {
    id: "e",
    controlled_start_s: 100,
    controlled_end_s: 100.1,
    metadata: { control_steps: 2 },
    ticks: [
      {
        id: "t0",
        tick: 0,
        applied_s: 100.002,
        next_observation_computed_s: 100.026,
        physics_step_seconds: 0.05,
      },
      {
        id: "t1",
        tick: 1,
        applied_s: 100.051,
        next_observation_computed_s: 100.074,
        physics_step_seconds: 0.05,
      },
    ],
  };
  const spans = R.controlIntervals(episode);
  assert.deepEqual(
    spans.map((p) => p.kind),
    ["gap", "execution", "gap", "execution", "gap"],
  );
  const execute = spans.filter((p) => p.kind === "execution");
  assert.ok(Math.abs(execute[0].end_s - execute[0].start_s - 0.024) < 1e-10);
  assert.equal(execute[0].tick.physics_step_seconds, 0.05);
  assert.ok(Math.abs(spans[2].end_s - spans[2].start_s - 0.025) < 1e-10);
  assert.ok(Math.abs(spans.at(-1).end_s - 0.1) < 1e-10);
  const late = structuredClone(episode);
  late.ticks[0].next_observation_computed_s = 100.06;
  late.ticks[1].applied_s = 100.062;
  const delayed = R.controlIntervals(late);
  assert.ok(delayed.find((p) => p.kind === "execution").end_s > 0.05);
  assert.ok(Math.abs(delayed[2].end_s - delayed[2].start_s - 0.002) < 1e-10);
});

test("missing control completion or omitted ticks never turn into invented execution or waiting", () => {
  const episode = {
    id: "e",
    controlled_start_s: 100,
    controlled_end_s: 100.2,
    metadata: { control_steps: 4 },
    ticks: [
      { id: "t0", tick: 0, applied_s: 100.002 },
      {
        id: "t2",
        tick: 2,
        applied_s: 100.1,
        next_observation_computed_s: 100.12,
      },
    ],
  };
  const spans = R.controlIntervals(episode);
  assert.deepEqual(
    spans.map((p) => p.kind),
    ["gap", "execution", "execution"],
  );
  assert.equal(spans[1].end_s, null);
  assert.equal(spans[2].tick.physics_step_seconds, undefined);
  assert.deepEqual(
    R.controlIntervals({ ...episode, controlled_start_s: null }),
    [],
  );
});
test("policy identity uses worker metadata, never filenames or scheduling labels", () => {
  assert.equal(R.policyVariant({ label: "pir2 checkpoint-10000" }), null);
  assert.equal(R.policyVariant({ configuration: { variant: "pir2" } }), null);
  const run = {
    kind: "deployment",
    policy_variant: "pir2",
    policy_identity: {
      variant: "pir2",
      predicted_horizon: 40,
      decoded_horizon: 40,
      training: { global_step: 10000 },
    },
  };
  assert.equal(R.policyVariant(run), "pir2");
  assert.match(R.policyLabel(run), /πR².*10000/);
  assert.match(R.policyNote(run), /独立.*S2/);
  assert.doesNotMatch(R.policyNote(run), /Flow4|解码16/);
  assert.equal(R.policyVariant({ ...run, policy_variant: "flow" }), "invalid");
  assert.equal(
    R.policyVariant({ ...run, configuration: { variant: "flow" } }),
    "invalid",
  );
});
test("legacy and current stream projections cannot change the resolved policy cohort", () => {
  const descriptor = { id: "test", kind: "deployment", precision: "bf16" };
  const pir2 = { ...descriptor, policy_variant: "pir2" };
  const flow = { ...descriptor, policy_variant: "flow" };
  assert.equal(R.matchesDescriptor(pir2, descriptor, null, "pir2"), true);
  assert.equal(R.matchesDescriptor(flow, descriptor, null, "pir2"), false);
  assert.equal(
    R.matchesDescriptor(descriptor, descriptor, null, "pir2"),
    false,
  );
  assert.equal(R.matchesDescriptor(pir2, descriptor, null, null), false);
  assert.equal(R.matchesDescriptor(descriptor, descriptor, null, null), true);
  assert.equal(R.matchesDescriptor(flow, pir2), false);
  assert.equal(
    R.matchesDescriptor(pir2, { ...descriptor, policy_variant: null }),
    false,
  );
});
test("execution cadence selects an explicit matched set and treats legacy runs as paced", () => {
  assert.equal(typeof R.deploymentCadence, "function");
  assert.equal(typeof R.replayDescriptors, "function");
  const descriptors = [
    { id: "old", kind: "deployment", precision: "bf16" },
    {
      id: "ready-bf",
      kind: "deployment",
      precision: "bf16",
      execution_clock: "ready",
    },
    {
      id: "ready-w8",
      kind: "deployment",
      precision: "w8a8",
      execution_clock: "ready",
    },
    { id: "model", kind: "model", precision: "bf16" },
  ];
  assert.equal(R.deploymentCadence(descriptors[0]), "paced");
  assert.deepEqual(
    R.replayDescriptors(descriptors, "deployment", "ready").map((r) => r.id),
    ["ready-bf", "ready-w8"],
  );
  assert.deepEqual(
    R.replayDescriptors(descriptors, "deployment", "paced").map((r) => r.id),
    ["old"],
  );
  assert.deepEqual(
    R.replayDescriptors(descriptors, "model", "ready").map((r) => r.id),
    ["model"],
  );
});
test("aligns only known identical input, seed, and record type; keeps missing slots", () => {
  assert.equal(typeof R.alignModel, "function");
  const runs = [
    { precision: "bf16", clean_requests: [clean] },
    { precision: "w8a8", clean_requests: [{ ...clean, id: "b", seed: 8 }] },
    { precision: "w4a4", clean_requests: [] },
  ];
  const rows = R.alignModel(runs, "clean", clean);
  assert.equal(rows[0].record.id, "a");
  assert.equal(rows[1].record, null);
  assert.equal(rows[2].record, null);
  assert.equal(
    R.alignModel(
      [{ profile_requests: [{ ...clean, instrumented: true }] }],
      "profile",
      clean,
    )[0].record,
    null,
  );
});
test("clean statistics exclude profile, missing, nonfinite and instrumented rows", () => {
  assert.deepEqual(
    R.cleanStats({
      clean_requests: [
        clean,
        { ...clean, wall_ms: 5 },
        { ...clean, wall_ms: 900, instrumented: true },
        { ...clean, wall_ms: null },
        { ...clean, wall_ms: Infinity },
      ],
    }),
    { count: 2, p50: 4, p95: 4.9 },
  );
});
const episode = {
  controlled_start_s: 100,
  controlled_end_s: 102,
  robot: {
    initial_state: { position_xyz_m: [0, 0, 0] },
    frames: [
      { frame_id: "i", relative_s: -0.1, media_time_s: 0 },
      { frame_id: "a", relative_s: 0.2, media_time_s: 0.05 },
      { frame_id: "b", relative_s: 0.8, media_time_s: 0.1 },
    ],
    media: { image: { media_id: "own" } },
  },
  ticks: [
    {
      id: "t0",
      relative_s: 0.1,
      action: [1],
      robot_response: {
        after_computed_s: 100.2,
        after: { position_xyz_m: [1, 0, 0] },
      },
    },
    {
      id: "t1",
      relative_s: 0.6,
      action: [2],
      robot_response: {
        after_computed_s: 100.8,
        after: { position_xyz_m: [2, 0, 0] },
      },
    },
  ],
};
test("robot uses own exact frame mapping and never exposes a future state or command", () => {
  const early = R.robotAt(episode, 0.15, "image");
  assert.equal(early.frame.frame_id, "i");
  assert.deepEqual(early.robotState.position_xyz_m, [0, 0, 0]);
  assert.equal(early.tick.id, "t0");
  assert.equal(early.media.media_id, "own");
  assert.equal(R.robotAt(episode, 0.7, "image").frame.frame_id, "a");
  assert.deepEqual(
    R.robotAt(episode, 0.7, "image").robotState.position_xyz_m,
    [1, 0, 0],
  );
  assert.equal(R.robotAt(episode, 3, "wrist_image").media, null);
  assert.equal(R.robotAt(episode, 3, "image").ended, true);
  assert.equal(R.robotAt(episode, -0.2, "image").frame, null);
  assert.equal(R.robotAt(episode, -0.2, "image").tick, null);
});
test("state/action series retain actual timestamps and units independently", () => {
  const series = R.robotSeries(episode, "x");
  assert.deepEqual(
    series.map((r) => r[1]),
    [0, 1, 2],
  );
  assert.ok(Math.abs(series[1][0] - 0.2) < 1e-12);
  assert.ok(Math.abs(series[2][0] - 0.8) < 1e-12);
  assert.deepEqual(R.robotSeries(episode, "command:0"), [
    [0.1, 1],
    [0.6, 2],
  ]);
});
test("paired episode identity requires fixture and task/episode/seed, never same observations", () => {
  const a = {
    task_id: 0,
    episode_id: 0,
    seed: 1000,
    paired_fixture_key: "fixture",
  };
  assert.equal(R.pairedEpisode(a, { ...a }), true);
  assert.equal(R.pairedEpisode(a, { ...a, paired_fixture_key: null }), false);
  assert.equal(R.pairedEpisode(a, { ...a, task_id: 1 }), false);
});
test("model outputs remain unavailable until their own request completes", () => {
  assert.equal(R.outputAt(clean, 2).status, "pending");
  assert.equal(R.outputAt(clean, 3).status, "recorded");
  assert.equal(
    R.outputAt(
      {
        host_wall_ms: 4,
        output: { status: "unavailable", reason: "not recorded" },
      },
      8,
    ).status,
    "unavailable",
  );
});
test("latest loading transaction drops stale results including failures", async () => {
  const loader = R.latestLoader();
  let a, b;
  const first = loader.load(
    () =>
      new Promise((resolve) => {
        a = resolve;
      }),
  );
  const second = loader.load(
    () =>
      new Promise((resolve) => {
        b = resolve;
      }),
  );
  b("new");
  assert.deepEqual(await second, { current: true, value: "new" });
  a("old");
  assert.deepEqual(await first, { current: false });
});
test("unavailable run descriptors use counts, never fake request arrays", () => {
  assert.deepEqual(
    R.records({ clean_requests: 30, profile_requests: 3 }, "profile"),
    [],
  );
  assert.deepEqual(R.cleanStats({ clean_requests: 30 }), {
    count: 0,
    p50: null,
    p95: null,
  });
});

test("unknown observation or seed permits own inspection but never cross-run pairing", () => {
  for (const missing of [{ observation_id: null }, { seed: null }]) {
    const row = { ...clean, ...missing },
      own = { id: "own", clock_cohort: "clock", clean_requests: [row] },
      other = {
        id: "other",
        clock_cohort: "clock",
        clean_requests: [{ ...row, id: "other-row" }],
      };
    const aligned = R.alignModel([own, other], "clean", row, "own");
    assert.equal(aligned[0].record, row);
    assert.equal(aligned[0].comparable, false);
    assert.equal(aligned[1].record, null);
  }
});
test("late rejected request cannot replace a newer transaction", async () => {
  const loader = R.latestLoader();
  let reject;
  const stale = loader.load(
    () =>
      new Promise((_, no) => {
        reject = no;
      }),
  );
  const current = await loader.load(async () => "selected");
  assert.equal(current.value, "selected");
  reject(new Error("late 404"));
  assert.deepEqual(await stale, { current: false });
});

test("incomplete episodes preserve metadata identity and never imply playable control", () => {
  const episode = R.normalizeEpisode({
    status: "incomplete",
    reason: "missing controller end",
    metadata: { task_id: 4, episode_id: 2, seed: 1000, success: true },
    ticks: [],
    queue: null,
  });
  assert.deepEqual(
    [episode.task_id, episode.episode_id, episode.seed],
    [4, 2, 1000],
  );
  assert.equal(R.playableEpisode(episode), false);
  assert.equal(episode.reason, "missing controller end");
});

test("canonical comparison scopes never mix wall statistics or pair different boundaries", () => {
  const row = { ...clean, scope: "prepared_model" };
  const run = {
    id: "own",
    clock_cohort: "same-gpu",
    clean_requests: [
      { ...row, id: "policy", scope: "policy", wall_ms: 38 },
      { ...row, id: "prep", scope: "preparation_transfer", wall_ms: 8 },
      { ...row, id: "model", wall_ms: 27 },
    ],
  };
  assert.deepEqual(R.cleanStats(run, "policy"), { count: 1, p50: 38, p95: 38 });
  assert.deepEqual(R.cleanStats(run, "preparation_transfer"), {
    count: 1,
    p50: 8,
    p95: 8,
  });
  assert.deepEqual(R.cleanStats(run, "prepared_model"), {
    count: 1,
    p50: 27,
    p95: 27,
  });
  const foreign = {
    id: "other",
    clock_cohort: "same-gpu",
    clean_requests: [{ ...row, id: "other-policy", scope: "policy" }],
  };
  assert.equal(
    R.alignModel([run, foreign], "clean", run.clean_requests[2], "own")[1]
      .record,
    null,
  );
});

test("an exact own record remains inspectable without clock/input pairing metadata", () => {
  const row = {
    ...clean,
    id: "own/request",
    scope: "prepared_model",
    comparison_key: null,
    seed: null,
    observation_id: null,
  };
  const own = { id: "own", clock_cohort: null, clean_requests: [row] };
  const foreign = {
    id: "other",
    clock_cohort: null,
    clean_requests: [{ ...row, id: "other/request" }],
  };
  const selected = R.alignModel([own, foreign], "clean", row, "own");
  assert.equal(selected[0].record, row);
  assert.equal(selected[0].comparable, false);
  assert.equal(selected[1].record, null);
});

test("profile boundary includes preparation only when declared, never decoded policy output", () => {
  const profile = { instrumented: true };
  assert.equal(
    R.scopeOf(
      {
        profile_summary: {
          input_preparation_included: true,
          scope:
            "Instrumented preparation + full VLM and Flow4; output decoding excluded",
        },
      },
      profile,
    ),
    "preparation_model",
  );
  assert.equal(
    R.scopeOf(
      {
        profile_summary: {
          input_preparation_included: false,
          scope:
            "Instrumented prepared full VLM and Flow4; output decoding excluded",
        },
      },
      profile,
    ),
    "prepared_model",
  );
  assert.equal(R.scopeOf({}, profile), "unknown");
});

test("unknown instrumentation blocks cross-run pairing even with a matching key and cohort", () => {
  for (const instrumented of [undefined, null]) {
    const row = { ...clean, instrumented, clean_wall: false };
    const own = {
      id: "own",
      clock_cohort: "known-clock",
      clean_requests: [row],
    };
    const foreign = {
      id: "other",
      clock_cohort: "known-clock",
      clean_requests: [{ ...row, id: "other/request" }],
    };
    const aligned = R.alignModel([own, foreign], "clean", row, "own");
    assert.equal(aligned[0].record, row);
    assert.equal(aligned[0].comparable, false);
    assert.equal(aligned[1].record, null);
  }
});

test("coupling filters prevent mixing synchronous and asynchronous robot evidence", () => {
  const runs = [
    { id: "legacy", kind: "deployment", execution_clock: "paced" },
    {
      id: "sync",
      kind: "deployment",
      execution_clock: "paced",
      execution_coupling: "synchronous",
    },
    {
      id: "ready",
      kind: "deployment",
      execution_clock: "ready",
      execution_coupling: "synchronous",
    },
  ];
  assert.equal(typeof R.deploymentCoupling, "function");
  assert.equal(R.deploymentCoupling(runs[0]), "asynchronous");
  assert.deepEqual(
    R.replayDescriptors(runs, "deployment", "paced", "synchronous").map(
      (r) => r.id,
    ),
    ["sync"],
  );
  assert.deepEqual(
    R.replayDescriptors(runs, "deployment", "paced").map((r) => r.id),
    ["legacy"],
  );
});

test("synchronous phase and physics cursor follow recorded intervals and completed steps", () => {
  const measured = {
    ...episode,
    phases: [
      { phase: "inference", start_s: 0, end_s: 0.1 },
      { phase: "execution", start_s: 0.1, end_s: 0.9 },
      { phase: "inference", start_s: 0.9, end_s: 2 },
    ],
    ticks: episode.ticks.map((row, i) => ({
      ...row,
      next_observation_computed_s: i === 0 ? 100.2 : 100.8,
      simulated_elapsed_seconds: (i + 1) * 0.05,
    })),
  };
  assert.equal(typeof R.phaseAt, "function");
  assert.equal(typeof R.physicsAt, "function");
  assert.equal(R.phaseAt(measured, 0.05).phase, "inference");
  assert.equal(R.phaseAt(measured, 0.1).phase, "execution");
  assert.equal(R.phaseAt(measured, 0.95).phase, "inference");
  assert.equal(R.phaseAt(measured, 2), null);
  assert.equal(R.physicsAt(measured, 0.15), 0);
  assert.equal(R.physicsAt(measured, 0.21), 0.05);
  assert.equal(R.physicsAt(measured, 0.95), 0.1);
  assert.equal(R.robotAt(measured, 0.95, "image").frame.frame_id, "b");
  assert.equal(R.robotAt(measured, 1.95, "image").frame.frame_id, "b");
});

test("physics cursor distinguishes known zero before the first step from missing evidence", () => {
  const complete = {
    controlled_start_s: 100,
    ticks: [
      { next_observation_computed_s: 100.2, simulated_elapsed_seconds: 0.05 },
      { next_observation_computed_s: 100.4, simulated_elapsed_seconds: 0.1 },
    ],
  };
  assert.equal(R.physicsAt(complete, 0.1), 0);
  assert.equal(R.physicsAt(complete, 0.3), 0.05);
  assert.equal(R.physicsAt({ ...complete, ticks: [] }, 0.1), null);
  assert.equal(R.physicsAt({ controlled_start_s: 100 }, 0.1), null);
  for (const value of [undefined, null, NaN, Infinity]) {
    for (const field of [
      "next_observation_computed_s",
      "simulated_elapsed_seconds",
    ]) {
      const incomplete = {
        ...complete,
        ticks: complete.ticks.map((tick, i) =>
          i === 1 ? { ...tick, [field]: value } : tick,
        ),
      };
      assert.equal(R.physicsAt(incomplete, 0.1), null);
      assert.equal(R.physicsAt(incomplete, 0.3), null);
    }
  }
});

test("metric scaling preserves missing values while keeping measured zero", () => {
  assert.equal(typeof R.scaleMetric, "function");
  for (const value of [undefined, null, NaN, Infinity, false, "0"]) {
    assert.equal(R.scaleMetric(value, 1000), null);
    assert.equal(R.scaleMetric(value, 100), null);
  }
  assert.equal(R.scaleMetric(0, 1000), 0);
  assert.equal(R.scaleMetric(0.025, 1000), 25);
  assert.equal(R.scaleMetric(0.25, 100), 25);
  assert.equal(R.scaleMetric(Number.MAX_VALUE, 1000), null);
});

test("controller policy is a bound cohort dimension across precision and SSE", () => {
  assert.equal(typeof R.controllerPolicy, "function");
  const descriptor = {
    id: "b",
    kind: "deployment",
    precision: "bf16",
    policy_variant: "pir2",
    deployment_scheduling_policy: "responsive",
  };
  const other = {
    ...descriptor,
    id: "w",
    precision: "w8a8",
    deployment_scheduling_policy: "conservative",
  };
  assert.deepEqual(
    R.replayDescriptors(
      [descriptor, other],
      "deployment",
      "paced",
      "asynchronous",
      null,
      "responsive",
    ),
    [descriptor],
  );
  assert.equal(
    R.matchesDescriptor(
      { ...descriptor },
      descriptor,
      null,
      "pir2",
      "responsive",
    ),
    true,
  );
  assert.equal(
    R.matchesDescriptor(
      { ...descriptor, deployment_scheduling_policy: "conservative" },
      descriptor,
      null,
      "pir2",
      "responsive",
    ),
    false,
  );
  const legacy = { id: "old", kind: "deployment", precision: "bf16" };
  assert.equal(R.controllerPolicy(legacy), null);
  assert.equal(
    R.matchesDescriptor(
      { ...legacy, deployment_scheduling_policy: "responsive" },
      legacy,
      null,
      undefined,
      null,
    ),
    false,
  );
});
