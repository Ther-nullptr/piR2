const test = require("node:test");
const assert = require("node:assert/strict");
const R = require("./replay-model.js");
const T = require("./replay-timeline.js");

function fixture() {
  const requests = [0, 1, 2].flatMap((id) => [
    {
      id: `v${id}`,
      role: "S2",
      request_id: id,
      source_tick: Math.max(0, id - 1),
      target_tick: id,
      feature_sequence: id,
      timing_scope: "server_vlm_stage",
      start_s: id * 0.1,
      end_s: id * 0.1 + 0.02,
      status: id === 2 ? "unused" : "completed",
      disposition: id === 2 ? "unused" : "adopted",
    },
    {
      id: `a${id}`,
      role: "S1",
      request_id: id,
      request_tick: id,
      source_tick: Math.max(0, id - 1),
      target_tick: id,
      feature_sequence: id,
      timing_scope: "server_action_stage",
      start_s: id * 0.1 + 0.025,
      end_s: id * 0.1 + 0.06,
      adopted_s: id === 2 ? null : id * 0.1 + 0.065,
      status: id === 2 ? "unused" : "completed",
      disposition: id === 2 ? "unused" : "adopted",
    },
  ]);
  return {
    run: {
      id: "async",
      kind: "deployment",
      execution_coupling: "one_step_async",
      execution_clock: "ready",
      precision: "bf16",
    },
    episode: {
      controlled_start_s: 100,
      controlled_end_s: 100.24,
      work_end_s: 100.28,
      queue: {
        duration_s: 0.24,
        requests: requests.slice(0, 4),
        replay_requests: requests,
      },
      phases: [
        {
          id: "i0",
          phase: "inference",
          request_id: 0,
          request_tick: 0,
          source_tick: 0,
          target_tick: 0,
          start_s: 0,
          end_s: 0.065,
          simulation_paused: true,
        },
        {
          id: "x0",
          phase: "execution",
          request_id: 0,
          request_tick: 0,
          source_tick: 0,
          target_tick: 0,
          start_s: 0.09,
          end_s: 0.13,
          simulation_paused: false,
        },
        {
          id: "i1",
          phase: "inference",
          request_id: 1,
          request_tick: 1,
          source_tick: 0,
          target_tick: 1,
          start_s: 0.099,
          end_s: 0.165,
          simulation_paused: false,
        },
        {
          id: "w1",
          phase: "wait",
          request_id: 1,
          request_tick: 1,
          source_tick: 0,
          target_tick: 1,
          start_s: 0.13,
          end_s: 0.165,
          simulation_paused: true,
        },
        {
          id: "x1",
          phase: "execution",
          request_id: 1,
          request_tick: 1,
          source_tick: 0,
          target_tick: 1,
          start_s: 0.199,
          end_s: 0.239,
          simulation_paused: false,
        },
        {
          id: "i2",
          phase: "inference",
          request_id: 2,
          request_tick: 2,
          source_tick: 1,
          target_tick: 2,
          start_s: 0.199,
          end_s: 0.265,
          simulation_paused: false,
        },
        {
          id: "d2",
          phase: "drain",
          request_id: 2,
          request_tick: 2,
          source_tick: 1,
          target_tick: 2,
          start_s: 0.24,
          end_s: 0.275,
          simulation_paused: true,
        },
      ],
      ticks: [
        {
          tick: 0,
          relative_s: 0.09,
          next_observation_computed_s: 100.13,
          simulated_elapsed_seconds: 0.05,
        },
        {
          tick: 1,
          relative_s: 0.199,
          next_observation_computed_s: 100.239,
          simulated_elapsed_seconds: 0.1,
        },
      ],
      robot: {
        frames: [
          { frame_id: "initial", relative_s: 0, media_time_s: 0 },
          { frame_id: "first", relative_s: 0.13, media_time_s: 0.05 },
          { frame_id: "last", relative_s: 0.239, media_time_s: 0.1 },
        ],
        media: {},
      },
    },
  };
}

test("one-step selection stays independent from legacy async and sync across descriptor refresh", () => {
  const async = fixture().run;
  const runs = [
    async,
    { ...async, id: "legacy", execution_coupling: "asynchronous" },
    { ...async, id: "sync", execution_coupling: "synchronous" },
  ];
  for (const descriptors of [
    runs,
    [...runs, { ...async, id: "w8", precision: "w8a8" }],
  ]) {
    assert.ok(
      R.replayDescriptors(
        descriptors,
        "deployment",
        "ready",
        "one_step_async",
      ).every((r) => r.execution_coupling === "one_step_async"),
    );
    assert.equal(
      R.replayDescriptors(descriptors, "deployment", "ready", "synchronous")[0]
        .id,
      "sync",
    );
  }
});

test("complete work timeline retains unused tail and distinct bootstrap/prefetch source zero", () => {
  const row = fixture();
  const tracks = T.tracks([row], "deployment");
  assert.deepEqual(
    tracks.map((t) => t.title),
    ["VLM / S2", "DiT / S1", "Policy RPC", "Execute / Wait / Drain"],
  );
  assert.equal(tracks[0].blocks.length, 3);
  assert.notEqual(tracks[0].blocks[0].label, tracks[0].blocks[1].label);
  assert.match(tracks[0].blocks[1].description, /Obs 0 → action 1/);
  assert.equal(tracks[1].blocks[2].sourceId, "v2");
  assert.equal(T.phase(tracks[1].blocks[2], 0.27).key, "unused");
  assert.equal(tracks[2].blocks[1].basis, "async_policy_rpc");
  assert.equal(
    tracks[3].blocks.find((b) => b.phaseKind === "execution").basis,
    "env_step",
  );
  assert.match(T.basisLabel("env_step"), /无额外节拍等待/);
  assert.ok(tracks[2].blocks[1].description.includes("Overlap allowed"));
  assert.ok(Math.abs(R.episodeEnd(row.episode) - 0.28) < 1e-10);
  assert.ok(Math.abs(R.controlledEnd(row.episode) - 0.24) < 1e-10);
});

test("async statuses show simultaneous inference/execution, explicit waits and stopped robot tail", () => {
  const { episode } = fixture();
  assert.match(R.asyncStatus(episode, 0.01), /Bootstrap/);
  assert.equal(R.asyncStatus(episode, 0.12), "Inferring + executing");
  assert.match(R.asyncStatus(episode, 0.14), /Waiting/);
  assert.match(R.asyncStatus(episode, 0.25), /Robot stopped.*unused inference/);
  assert.match(R.asyncStatus(episode, 0.3), /all work complete/);
  assert.equal(R.activePhases(episode, 0.12).length, 2);
  assert.equal(R.robotAt(episode, 0.12, "image").frame.frame_id, "initial");
  assert.equal(R.robotAt(episode, 0.14, "image").frame.frame_id, "first");
  assert.equal(R.robotAt(episode, 0.16, "image").frame.frame_id, "first");
  assert.equal(R.robotAt(episode, 0.27, "image").frame.frame_id, "last");
  assert.equal(R.robotAt(episode, 0.27, "image").ended, true);
  assert.equal(R.physicsAt(episode, 0.27), 0.1);
});

test("execution highlights exact previous producer while next policy is running", () => {
  const row = fixture();
  const tracks = T.tracks([row], "deployment");
  assert.deepEqual(
    T.sourceBlocks(tracks, "x1").map((b) => b.id),
    ["a1", "v1"],
  );
  const concurrent = tracks.filter(
    (t) => T.laneState(t, 0.21).key === "running",
  );
  assert.ok(concurrent.some((t) => t.title === "Policy RPC"));
  assert.ok(concurrent.some((t) => t.title === "Execute / Wait / Drain"));
  const view = T.oneStepViewport([row], 0.23, 5);
  assert.ok(view.start <= 0.1);
  row.episode.phases.find((p) => p.id === "x1").request_tick = null;
  assert.deepEqual(T.sourceBlocks(T.tracks([row], "deployment"), "x1"), []);
});

test("ambiguous VLM lineage remains unknown and missing work boundaries never invent a drain", () => {
  const row = fixture();
  row.episode.queue.replay_requests.push({
    ...row.episode.queue.replay_requests[2],
    id: "duplicate",
  });
  const tracks = T.tracks([row], "deployment");
  assert.equal(tracks[1].blocks[1].sourceId, null);
  assert.equal(T.sourceBlocks(tracks, "x1").length, 1);
  delete row.episode.work_end_s;
  assert.equal(R.episodeEnd(row.episode), R.controlledEnd(row.episode));
  assert.equal(
    R.physicsAt({ ...row.episode, ticks: [{ tick: 0 }] }, 0.2),
    null,
  );
});

const grid = [
  [1, 0],
  [1, 1],
  [2, 0],
  [2, 1],
  [2, 2],
  [4, 0],
  [4, 1],
  [4, 2],
  [4, 4],
  [8, 0],
  [8, 1],
  [8, 4],
  [8, 8],
];
function matrixDescriptors() {
  return grid.flatMap(([k, d]) =>
    ["bf16", "w8a8", "w4a4"].map((precision) => ({
      id: `${precision}-k${k}-d${d}`,
      kind: "deployment",
      precision,
      execution_coupling: d ? "chunked_async" : "synchronous",
      execution_clock: "ready",
      execution_prefix_steps: k,
      async_overlap_steps: d,
    })),
  );
}

test("matrix choices expose exactly measured K/D pairs and select only one three-precision cohort", () => {
  const descriptors = matrixDescriptors();
  for (const [k, d] of grid) {
    const coupling = d ? "chunked_async" : "synchronous";
    const choices = R.scheduleChoices(descriptors, "ready", coupling);
    assert.ok(choices.some((s) => s.k === k && s.d === d));
    assert.deepEqual(R.selectSchedule(choices, { k, d }), { k, d });
    const cohort = R.replayDescriptors(
      descriptors,
      "deployment",
      "ready",
      coupling,
      { k, d },
    );
    assert.equal(cohort.length, 3);
    assert.deepEqual(
      cohort.map((r) => r.precision),
      ["bf16", "w8a8", "w4a4"],
    );
    assert.ok(
      cohort.every((r) => R.sameSchedule(R.deploymentSchedule(r), { k, d })),
    );
  }
  assert.equal(
    R.replayDescriptors(descriptors, "deployment", "ready", "chunked_async", {
      k: 4,
      d: 3,
    }).length,
    0,
  );
  assert.equal(
    R.replayDescriptors(descriptors, "deployment", "ready", "synchronous", {
      k: 4,
      d: 1,
    }).length,
    0,
  );
});

test("K change, coupling switch and catalog refresh preserve a valid deterministic measured schedule", () => {
  const descriptors = matrixDescriptors();
  const asyncChoices = R.scheduleChoices(descriptors, "ready", "chunked_async");
  assert.deepEqual(R.selectSchedule(asyncChoices, { k: 8, d: 2 }), {
    k: 8,
    d: 1,
  });
  assert.deepEqual(R.selectSchedule(asyncChoices, { k: 8, d: 4 }), {
    k: 8,
    d: 4,
  });
  const syncChoices = R.scheduleChoices(descriptors, "ready", "synchronous");
  assert.deepEqual(R.selectSchedule(syncChoices, { k: 8, d: 4 }), {
    k: 8,
    d: 0,
  });
  assert.deepEqual(R.selectSchedule(asyncChoices, { k: 8, d: 0 }), {
    k: 8,
    d: 1,
  });
  const removed = descriptors.filter((r) => r.execution_prefix_steps !== 8);
  assert.deepEqual(
    R.selectSchedule(R.scheduleChoices(removed, "ready", "chunked_async"), {
      k: 8,
      d: 4,
    }),
    { k: 1, d: 1 },
  );
  assert.equal(R.selectSchedule([], { k: 1, d: 1 }), null);
});

test("invalid or missing matrix dimensions never invent valid combinations; legacy modes remain separate", () => {
  const good = matrixDescriptors().find((r) => r.async_overlap_steps === 2);
  for (const bad of [
    { ...good, execution_prefix_steps: null },
    { ...good, async_overlap_steps: null },
    { ...good, async_overlap_steps: 8 },
    { ...good, async_overlap_steps: 0 },
    { ...good, execution_prefix_steps: "2" },
  ]) {
    assert.equal(R.deploymentSchedule(bad), null);
    assert.deepEqual(R.scheduleChoices([bad], "ready", "chunked_async"), []);
    assert.deepEqual(
      R.replayDescriptors([bad], "deployment", "ready", "chunked_async", null),
      [],
    );
  }
  const legacy = { ...good, id: "cached", execution_coupling: "asynchronous" };
  assert.equal(R.deploymentSchedule(legacy), null);
  assert.equal(
    R.replayDescriptors([legacy], "deployment", "ready", "asynchronous", null)
      .length,
    1,
  );
  assert.deepEqual(R.deploymentSchedule(fixture().run), { k: 1, d: 1 });
  assert.deepEqual(
    R.deploymentSchedule({
      kind: "deployment",
      execution_coupling: "synchronous",
      configuration: { synchronous_chunk_steps: 8 },
    }),
    { k: 8, d: 0 },
  );
});

test("a live projection cannot silently change selected precision, coupling, cadence or K/D", () => {
  const descriptor = matrixDescriptors().find(
    (r) => r.async_overlap_steps === 2,
  );
  assert.equal(R.matchesDescriptor({ ...descriptor }, descriptor), true);
  for (const mutation of [
    { execution_prefix_steps: 8 },
    { async_overlap_steps: 1 },
    { execution_clock: "paced" },
    { execution_coupling: "one_step_async" },
    { precision: "w4a4" },
    { id: "another" },
  ])
    assert.equal(
      R.matchesDescriptor({ ...descriptor, ...mutation }, descriptor),
      false,
    );
});

function chunkedFixture() {
  const row = fixture();
  row.run = {
    ...row.run,
    execution_coupling: "chunked_async",
    execution_prefix_steps: 4,
    async_overlap_steps: 2,
  };
  // Real fixture semantics: request 4 uses obs 2 for all native offsets;
  // terminal request 8 can remain unused beyond the last physical tick 6.
  row.episode.controlled_end_s = 100.75;
  row.episode.work_end_s = 100.85;
  row.episode.queue.replay_requests = [0, 4, 8].flatMap((id, index) => [
    {
      id: `v${id}`,
      role: "S2",
      request_id: id,
      source_tick: Math.max(0, id - 2),
      target_tick: id,
      feature_sequence: id,
      start_s: index * 0.3,
      end_s: index * 0.3 + 0.02,
      timing_scope: "server_vlm_stage",
    },
    {
      id: `a${id}`,
      role: "S1",
      request_id: id,
      request_tick: id,
      source_tick: Math.max(0, id - 2),
      target_tick: id,
      feature_sequence: id,
      start_s: index * 0.3 + 0.02,
      end_s: index * 0.3 + 0.04,
      timing_scope: "server_action_stage",
      disposition: id === 8 ? "unused" : "adopted",
      status: id === 8 ? "unused" : "completed",
      decoded_horizon: 16,
      accepted_prefix_steps: id === 8 ? 0 : 4,
      executed_prefix_steps: id === 8 ? 0 : id === 4 ? 3 : 4,
    },
  ]);
  row.episode.phases = [0, 1, 2].map((offset) => ({
    id: `request-4/execution-${4 + offset}`,
    phase: "execution",
    request_id: 4,
    request_tick: 4,
    target_tick: 4,
    source_tick: 2,
    execution_id: 4 + offset,
    executed_offset: offset,
    start_s: 0.4 + offset * 0.1,
    end_s: 0.5 + offset * 0.1,
    simulation_paused: false,
  }));
  row.episode.phases.push({
    id: "drain8",
    phase: "drain",
    request_id: 8,
    start_s: 0.75,
    end_s: 0.85,
    simulation_paused: true,
  });
  return row;
}

test("every native offset retains its exact shared producer and visible source at the end of a long prefix", () => {
  const row = chunkedFixture(),
    tracks = T.tracks([row], "deployment");
  assert.equal(tracks.length, 4);
  const actions = tracks
    .at(-1)
    .blocks.filter((b) => b.phaseKind === "execution");
  assert.equal(new Set(actions.map((b) => b.id)).size, 3);
  for (const [offset, block] of actions.entries()) {
    assert.deepEqual(
      T.sourceBlocks(tracks, block.id).map((b) => b.id),
      ["a4", "v4"],
    );
    assert.match(
      block.label,
      new RegExp(`Request 4.*Tick ${4 + offset}.*offset ${offset}`),
    );
    assert.match(
      block.description,
      new RegExp(`lag ${2 + offset} physics steps`),
    );
  }
  const view = T.oneStepViewport([row], 0.699, 5);
  assert.ok(view.start <= 0.3);
  assert.match(
    tracks[1].blocks[1].description,
    /native 16 \/ accepted 4 \/ executed 3/,
  );
  assert.equal(T.phase(tracks[1].blocks[2], 0.85).key, "unused");
  assert.equal(tracks.at(-1).blocks.at(-1).sourceId, null);
  assert.equal(R.episodeEnd(row.episode) > R.controlledEnd(row.episode), true);
});

test("prefix metrics distinguish native, accepted, terminal execution and missing fields", () => {
  const episode = {
    queue: {
      work_requests: [
        {
          request_id: 128,
          decoded_horizon: 16,
          accepted_prefix_steps: 4,
          executed_prefix_steps: 3,
        },
      ],
    },
  };
  assert.deepEqual(
    R.currentPrefix(episode, {
      request_id: 128,
      executed_offset: 2,
      observation_lag_ticks: 4,
    }),
    { native: 16, accepted: 4, executed: 3, offset: 2, lag: 4 },
  );
  assert.equal(R.currentPrefix(episode, null), null);
  assert.equal(R.currentPrefix(episode, { request_id: 132 }), null);
  delete episode.queue.work_requests[0].executed_prefix_steps;
  assert.equal(R.currentPrefix(episode, { request_id: 128 }).executed, null);
  assert.equal(R.currentPrefix(episode, { request_id: 128 }).offset, null);
  const sync = {
    cycles: [
      {
        cycle: 9,
        request_tick: 72,
        executed_steps: 2,
        audit: { decoded_horizon: 16, source_tick: 72 },
      },
    ],
    queue: {
      requests: [
        {
          role: "S1",
          request_tick: 72,
          valid_start_tick: 72,
          valid_end_tick: 80,
        },
      ],
    },
  };
  assert.deepEqual(R.currentPrefix(sync, { cycle: 9, tick: 73 }), {
    native: 16,
    accepted: 8,
    executed: 2,
    offset: 1,
    lag: 1,
  });
  delete sync.queue.requests[0].valid_end_tick;
  assert.equal(R.currentPrefix(sync, { cycle: 9, tick: 73 }).accepted, null);
});

test("known zero waits differ from missing or corrupt wait evidence", () => {
  assert.deepEqual(R.waitStats({}), { count: null, total: null, p95: null });
  assert.deepEqual(R.waitStats({ phases: [] }), {
    count: 0,
    total: 0,
    p95: null,
  });
  assert.deepEqual(
    R.waitStats({ phases: [{ phase: "wait", start_s: 1, end_s: 1.5 }] }),
    { count: 1, total: 0.5, p95: 0.5 },
  );
  for (const end_s of [null, undefined, 0, Infinity])
    assert.deepEqual(
      R.waitStats({ phases: [{ phase: "wait", start_s: 1, end_s }] }),
      { count: null, total: null, p95: null },
    );
});

test("missing physical identities never render a numerically invented chunk lag", () => {
  const row = chunkedFixture();
  row.episode.phases[0].source_tick = null;
  row.episode.phases[0].executed_offset = null;
  const block = T.tracks([row], "deployment").at(-1).blocks[0];
  assert.match(block.description, /lag \? physics steps/);
  assert.match(block.label, /offset \?/);
});

test("resolved legacy cohorts reject K/D drift on subsequent projections without fixing cached-worker K/D", () => {
  const descriptor = {
    id: "legacy",
    kind: "deployment",
    precision: "bf16",
    execution_coupling: "synchronous",
    execution_clock: "ready",
  };
  const initial = {
    ...descriptor,
    configuration: { synchronous_chunk_steps: 8 },
  };
  const changed = {
    ...descriptor,
    configuration: { synchronous_chunk_steps: 1 },
  };
  assert.equal(R.matchesDescriptor(initial, descriptor), true);
  assert.equal(R.matchesDescriptor(initial, descriptor, { k: 8, d: 0 }), true);
  assert.equal(R.matchesDescriptor(changed, descriptor, { k: 8, d: 0 }), false);
  assert.equal(R.matchesDescriptor(changed, descriptor, { k: 1, d: 0 }), true);
  assert.equal(R.matchesDescriptor(initial, descriptor, null), false);
  const cached = { ...descriptor, execution_coupling: "asynchronous" };
  assert.equal(
    R.matchesDescriptor(
      { ...cached, configuration: { synchronous_chunk_steps: 8 } },
      cached,
      null,
    ),
    true,
  );
  assert.equal(
    R.matchesDescriptor(
      { ...cached, configuration: { synchronous_chunk_steps: 1 } },
      cached,
      null,
    ),
    true,
  );
});
