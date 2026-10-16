const test = require("node:test");
const assert = require("node:assert/strict");
const T = require("./replay-timeline.js");

function cachedExecution() {
  return {
    run: {
      id: "r",
      precision: "bf16",
      execution_coupling: "asynchronous",
      policy_variant: "pir2",
    },
    episode: {
      id: "e",
      controlled_start_s: 100,
      controlled_end_s: 101.1,
      metadata: { control_steps: 22 },
      ticks: [
        {
          id: "tick21",
          tick: 21,
          applied_s: 101.051,
          next_observation_computed_s: 101.075,
          physics_step_seconds: 0.05,
          executed_slot: {
            producer_kind: "request",
            producer_request_tick: 20,
          },
        },
      ],
      queue: {
        duration_s: 1.1,
        requests: [
          {
            id: "v-old",
            role: "S2",
            start_s: 0.2,
            end_s: 0.24,
            feature_sequence: 1,
          },
          {
            id: "v-new",
            role: "S2",
            start_s: 0.9,
            end_s: 0.94,
            feature_sequence: 2,
          },
          {
            id: "a-source",
            role: "S1",
            start_s: 1.001,
            end_s: 1.025,
            request_tick: 20,
            feature_sequence: 1,
          },
          {
            id: "a-new",
            role: "S1",
            start_s: 1.052,
            end_s: 1.068,
            request_tick: 21,
            feature_sequence: 2,
          },
        ],
      },
    },
  };
}

test("cached policy execution has its own measured lane and exact DiT-to-VLM lineage", () => {
  const tracks = T.tracks([cachedExecution()], "deployment");
  const control = tracks.find((t) => t.stage === "control");
  assert.ok(control);
  const block = control.blocks.find((b) => b.phaseKind === "execution");
  assert.ok(Math.abs(block.start - 1.051) < 1e-10);
  assert.ok(Math.abs(block.end - 1.075) < 1e-10);
  assert.equal(block.basis, "env_step");
  assert.equal(block.sourceId, "a-source");
  assert.deepEqual(
    T.sourceBlocks(tracks, block.id).map((b) => b.id),
    ["a-source", "v-old"],
  );
  assert.equal(block.event.tick.tick, 21);
  assert.match(block.description, /50.*ms/);
  const gap = control.blocks.find((b) => b.phaseKind === "gap");
  assert.equal(gap.basis, "control_gap");
  assert.equal(gap.sourceId, null);
  const view = T.controlViewport(tracks, 1.07, 1.1);
  assert.ok(view.start <= 0.2 && view.end >= block.end);
});

test("bootstrap, fallback and ambiguous control producers never acquire a guessed model source", () => {
  for (const kind of ["bootstrap", "fallback", "request"]) {
    const row = cachedExecution();
    if (kind === "request")
      row.episode.queue.requests.push({
        ...row.episode.queue.requests[2],
        id: "duplicate",
      });
    else
      row.episode.ticks[0].executed_slot = {
        producer_kind: kind,
        producer_request_tick: null,
      };
    const block = T.tracks([row], "deployment")
      .find((t) => t.stage === "control")
      .blocks.find((b) => b.phaseKind === "execution");
    assert.equal(block.sourceId, null);
    assert.deepEqual(
      T.sourceBlocks(T.tracks([row], "deployment"), block.id),
      [],
    );
    if (kind !== "request")
      assert.match(block.description, new RegExp(kind, "i"));
  }
});

test("execution source stops at an ambiguous visual cache and keeps the recorded physical step", () => {
  const row = cachedExecution();
  row.episode.queue.requests.push({
    ...row.episode.queue.requests[0],
    id: "duplicate-visual",
  });
  row.episode.ticks[0].physics_step_seconds = 0.02;
  const tracks = T.tracks([row], "deployment");
  const block = tracks
    .find((t) => t.stage === "control")
    .blocks.find((b) => b.phaseKind === "execution");
  assert.deepEqual(
    T.sourceBlocks(tracks, block.id).map((b) => b.id),
    ["a-source"],
  );
  assert.equal(
    tracks.find((t) => t.stage === "action").blocks[0].label,
    "S1[?]",
  );
  assert.equal(block.sourceLabel, "S1[?]");
  assert.match(block.description, /20\.0 ms simulation/);
  assert.equal(
    T.controlViewport(tracks, 1.07, 1.1).start,
    T.viewport(1.07, 1.1, 0.3).start,
  );
});

test("DiT labels retain consumed VLM source even after a newer publication", () => {
  const requests = [
    { id: "v0", role: "S2", start_s: 0, end_s: 0.1, feature_sequence: 8 },
    { id: "v1", role: "S2", start_s: 0.12, end_s: 0.2, feature_sequence: 9 },
    {
      id: "a0",
      role: "S1",
      start_s: 0.21,
      end_s: 0.3,
      request_tick: 5,
      feature_sequence: 8,
    },
    {
      id: "a1",
      role: "S1",
      start_s: 0.31,
      end_s: 0.4,
      request_tick: 6,
      feature_sequence: 8,
    },
  ];
  const tracks = T.tracks(
    [
      {
        run: { id: "run", precision: "w8a8" },
        episode: { queue: { duration_s: 1, requests } },
      },
    ],
    "deployment",
  );
  assert.deepEqual(
    tracks.map((x) => x.stage),
    ["vlm", "action"],
  );
  assert.deepEqual(
    tracks[1].blocks.map((x) => x.label),
    ["S1[00].1", "S1[00].2"],
  );
  assert.equal(tracks[1].blocks[0].sourceLabel, "S2-00");
  assert.equal(tracks[1].blocks[0].basis, "rpc");
});

test("Clean wall without stages never invents a VLM and DiT split", () => {
  const tracks = T.tracks(
    [
      {
        run: { id: "r", precision: "bf16" },
        record: {
          id: "c",
          scope: "prepared_model",
          wall_ms: 18,
          instrumented: false,
        },
      },
    ],
    "model",
  );
  assert.equal(tracks.length, 2);
  assert.ok(tracks.every((x) => x.blocks.length === 0 && x.unavailable));
});

test("Profile spans come only from actual stage-attributed kernel intervals", () => {
  const row = {
    run: { id: "r", precision: "w8a8" },
    record: {
      id: "p",
      scope: "prepared_model",
      instrumented: true,
      timeline: [
        { stage: "vlm", start_ms: 2, duration_ms: 1 },
        { stage: "vlm", start_ms: 5, duration_ms: 2 },
        { stage: "action", start_ms: 8, duration_ms: 3 },
        { stage: "unknown", start_ms: 0, duration_ms: 99 },
      ],
    },
  };
  const tracks = T.tracks([row], "model");
  assert.equal(tracks[0].blocks[0].start, 2);
  assert.equal(tracks[0].blocks[0].end, 7);
  assert.equal(tracks[0].blocks[0].basis, "gpu_span");
  assert.equal(tracks[1].blocks[0].sourceLabel, null);
  assert.equal(T.phase(tracks[0].blocks[0], 4).key, "running");
  assert.equal(T.phase(tracks[0].blocks[0], 7).key, "completed");
});

test("an unrecorded completion has no invented duration or percentage", () => {
  const block = { start: 1, end: null, status: "incomplete" };
  assert.equal(T.phase(block, 0).key, "future");
  assert.equal(T.phase(block, 2).progress, null);
  assert.equal(T.phase(block, 2).key, "unknown");
});

test("an unknown RPC outcome remains unknown after the error was observed", () => {
  const block = { start: 1, end: 2, status: "unknown", basis: "rpc" };
  assert.equal(T.phase(block, 3).key, "unknown");
  assert.equal(T.phase(block, 3).progress, null);
});

test("canonical Profile records can declare their boundary in the run summary", () => {
  const rows = [
    {
      run: {
        id: "real-format",
        precision: "bf16",
        profile_summary: {
          scope:
            "Instrumented prepared full VLM and Flow4; output decoding excluded",
          input_preparation_included: false,
        },
      },
      record: {
        id: "profile",
        instrumented: true,
        timeline: [
          { stage: "vlm", start_ms: 2, duration_ms: 10 },
          { stage: "action", start_ms: 15, duration_ms: 20 },
        ],
      },
    },
  ];
  assert.deepEqual(
    T.tracks(rows, "model").map((track) => track.blocks.length),
    [1, 1],
  );
});

test("a following viewport moves completed blocks left while their duration stays fixed", () => {
  const first = T.viewport(4, 20, 2),
    next = T.viewport(4.5, 20, 2);
  const a = T.geometry({ start: 3.8, end: 4.2 }, first);
  const b = T.geometry({ start: 3.8, end: 4.2 }, next);
  assert.ok(b.left < a.left);
  assert.ok(Math.abs(b.width - a.width) < 1e-9);
});

function liveModel(spans, extra = {}) {
  return {
    run: { id: "live", precision: "w8a8" },
    record: {
      id: "profile",
      scope: "prepared_model",
      instrumented: true,
      host_wall_ms: null,
      wall_ms: null,
      stage_spans: spans.map((span) => ({
        domain: "host_submit",
        clock_id: "host-monotonic",
        ...span,
      })),
      ...extra,
    },
  };
}

test("live host submission retains open ends and their recorded status", () => {
  const row = liveModel([
    { stage: "vlm", start_ms: 3, end_ms: null, status: "running" },
  ]);
  const track = T.tracks([row], "model")[0];
  assert.equal(track.blocks.length, 1);
  const block = track.blocks[0];
  assert.equal(block.end, null);
  assert.equal(block.status, "running");
  assert.equal(block.clockId, "host-monotonic");
  assert.equal(T.phase(block, 10).key, "running");
  assert.match(T.phase(block, 10).label, /主机提交中/);
  assert.equal(T.phase(block, 10).progress, null);
  assert.equal(row.record.host_wall_ms, null);
  assert.equal(T.geometry(block, T.viewport(10, 10, 10)).open, true);
});

test("submitted host spans do not claim GPU completion", () => {
  const row = liveModel([
    { stage: "vlm", start_ms: 3, end_ms: 10, status: "submitted" },
  ]);
  const block = T.tracks([row], "model")[0].blocks[0];
  assert.match(T.phase(block, 5).label, /主机提交中/);
  assert.equal(T.phase(block, 10).key, "submitted");
  assert.match(T.phase(block, 10).label, /主机已提交.*GPU 完成未知/);
});

test("failed live spans retain failure with or without a recorded end", () => {
  for (const end of [null, 10]) {
    const row = liveModel([
      { stage: "action", start_ms: 3, end_ms: end, status: "failed" },
    ]);
    const track = T.tracks([row], "model")[1];
    assert.equal(track.blocks.length, 1);
    assert.equal(track.blocks[0].status, "failed");
    assert.equal(T.laneState(track, 10).key, "failed");
    assert.equal(
      T.phase(track.blocks[0], 10).progress,
      end === null ? null : 1,
    );
  }
});

test("incomplete host spans remain unknown even with a recorded stop", () => {
  const row = liveModel([
    { stage: "action", start_ms: 3, end_ms: 10, status: "incomplete" },
  ]);
  const block = T.tracks([row], "model")[1].blocks[0];
  assert.equal(T.phase(block, 10).key, "unknown");
  assert.match(T.phase(block, 10).label, /未完成/);
});

test("canonical GPU intervals replace host spans without mixing clock origins", () => {
  const row = liveModel(
    [
      { stage: "vlm", start_ms: 1, end_ms: 10, status: "submitted" },
      { stage: "action", start_ms: 11, end_ms: null, status: "running" },
    ],
    { timeline: [{ stage: "vlm", start_ms: 50, duration_ms: 5 }] },
  );
  const tracks = T.tracks([row], "model");
  assert.equal(tracks[0].blocks[0].basis, "gpu_span");
  assert.equal(tracks[0].blocks[0].start, 50);
  assert.equal(tracks[0].blocks[0].end, 55);
  assert.equal(tracks[1].blocks.length, 0);
});

test("model range follows recorded progress without inventing a request duration", () => {
  assert.equal(typeof T.observedModelEnd, "function");
  const row = liveModel(
    [
      { stage: "vlm", start_ms: 3, end_ms: 12, status: "submitted" },
      { stage: "action", start_ms: 15, end_ms: null, status: "running" },
    ],
    { host_progress_ms: 18 },
  );
  const before = JSON.stringify(row.record);
  assert.equal(T.observedModelEnd(row.record), 18);
  assert.equal(JSON.stringify(row.record), before);
  assert.equal(
    T.observedModelEnd({ ...row.record, host_progress_ms: null }),
    15,
  );
  assert.equal(T.observedModelEnd({ ...row.record, host_wall_ms: 22 }), 22);
  assert.equal(T.observedModelEnd({ wall_ms: 30 }), 30);
  assert.equal(
    T.observedModelEnd({ timeline: [{ start_ms: 50, duration_ms: 5 }] }),
    55,
  );
  assert.equal(T.observedModelEnd(null), 0);
  assert.equal(
    T.observedModelEnd({
      host_progress_ms: Infinity,
      host_wall_ms: null,
      wall_ms: "unknown",
      stage_spans: [{ start_ms: NaN, end_ms: null }],
    }),
    0,
  );
});

function synchronousRow() {
  return {
    run: { id: "sync", precision: "w8a8", execution_coupling: "synchronous" },
    episode: {
      queue: {
        duration_s: 3,
        requests: [
          {
            id: "v0",
            role: "S2",
            feature_sequence: 8,
            start_s: 0.01,
            end_s: 0.07,
            timing_scope: "server_vlm_stage",
          },
          {
            id: "a0",
            role: "S1",
            feature_sequence: 8,
            request_tick: 5,
            start_s: 0.07,
            end_s: 0.12,
            timing_scope: "server_action_stage",
          },
          // A newer unrelated action is closer in time to execution; it is not its source.
          {
            id: "a1",
            role: "S1",
            feature_sequence: 8,
            request_tick: 9,
            start_s: 0.13,
            end_s: 0.14,
            timing_scope: "server_action_stage",
          },
        ],
      },
      phases: [
        {
          id: "p0",
          phase: "inference",
          cycle: 0,
          request_tick: 5,
          start_s: 0,
          end_s: 0.15,
          simulation_paused: true,
        },
        {
          id: "p1",
          phase: "execution",
          cycle: 0,
          request_tick: 5,
          start_s: 0.16,
          end_s: 0.56,
          physics_start_tick: 5,
          physics_end_tick: 13,
        },
        {
          id: "p2",
          phase: "inference",
          cycle: 1,
          request_tick: 13,
          start_s: 0.6,
          end_s: 0.8,
          simulation_paused: true,
        },
      ],
    },
  };
}

test("synchronous task tracks preserve measured server stages and separate wall-control phases", () => {
  const row = synchronousRow(),
    before = JSON.stringify(row);
  const tracks = T.tracks([row], "deployment");
  assert.deepEqual(
    tracks.map((t) => t.stage),
    ["vlm", "action", "phase"],
  );
  assert.equal(tracks[0].blocks[0].basis, "server_stage");
  assert.equal(tracks[1].blocks[0].basis, "server_stage");
  assert.equal(tracks[1].blocks[0].timingScope, "server_action_stage");
  assert.equal(tracks[1].blocks[0].sourceId, "v0");
  assert.match(
    T.basisLabel(tracks[1].blocks[0].basis),
    /服务端.*非独立 RPC.*非 GPU/,
  );
  const phases = tracks[2].blocks;
  assert.deepEqual(
    phases.map((b) => [b.start, b.end]),
    [
      [0, 0.15],
      [0.16, 0.56],
      [0.6, 0.8],
    ],
  );
  assert.deepEqual(
    phases.map((b) => b.basis),
    ["inference_rpc", "execution_phase", "inference_rpc"],
  );
  assert.equal(tracks[2].unit, "s");
  assert.equal(tracks[2].clockId, "control_wall");
  assert.equal(T.phase(phases[0], 0.1).key, "running");
  assert.equal(T.phase(phases[1], 0.1).key, "future");
  assert.equal(T.phase(phases[0], 0.2).key, "completed");
  assert.equal(T.phase(phases[1], 0.2).key, "running");
  assert.equal(JSON.stringify(row), before);
});

test("execution phase source uses its recorded request identity, never the nearest task", () => {
  const tracks = T.tracks([synchronousRow()], "deployment");
  const phase = tracks[2].blocks[1];
  assert.equal(phase.sourceId, "a0");
  assert.equal(phase.sourceLabel, "S1[00].1");
  assert.deepEqual(
    T.sourceBlocks(tracks, phase.id).map((b) => b.id),
    ["a0", "v0"],
  );
  assert.deepEqual(
    T.sourceBlocks(tracks, "a1").map((b) => b.id),
    ["v0"],
  );
  assert.deepEqual(T.sourceBlocks(tracks, "absent"), []);
});

test("missing or ambiguous execution producer remains unknown without timestamp fallback", () => {
  for (const requestTick of [undefined, null, 99]) {
    const row = synchronousRow();
    row.episode.phases[1].request_tick = requestTick;
    const phase = T.tracks([row], "deployment")[2].blocks[1];
    assert.equal(phase.sourceId, null);
    assert.equal(phase.sourceLabel, null);
  }
  const row = synchronousRow();
  row.episode.queue.requests.push({
    ...row.episode.queue.requests[1],
    id: "duplicate",
  });
  assert.equal(T.tracks([row], "deployment")[2].blocks[1].sourceId, null);
});

test("missing synchronous phase evidence does not fabricate durations or RPC timings", () => {
  const row = synchronousRow();
  delete row.episode.phases;
  const tracks = T.tracks([row], "deployment");
  assert.equal(tracks[2].blocks.length, 0);
  assert.ok(tracks[2].unavailable);
  delete row.episode.queue.requests[0].timing_scope;
  assert.equal(T.tracks([row], "deployment")[0].blocks[0].basis, "unknown");
});

test("partial and invalid phase records do not invent completion", () => {
  const row = synchronousRow();
  row.episode.phases = [
    {
      id: "open",
      phase: "inference",
      start_s: 0,
      end_s: null,
      status: "running",
    },
    { id: "invalid", phase: "execution", start_s: 2, end_s: 1 },
    { id: "missing-start", phase: "inference", end_s: 3 },
  ];
  const blocks = T.tracks([row], "deployment")[2].blocks;
  assert.equal(blocks.length, 1);
  assert.equal(blocks[0].end, null);
  assert.equal(T.phase(blocks[0], 2).progress, null);
});

function completeSynchronousCycles() {
  const row = synchronousRow();
  row.episode.phases = [0, 1, 2].flatMap((cycle) => {
    const start = cycle * 0.45;
    return [
      {
        id: `i${cycle}`,
        cycle,
        phase: "inference",
        start_s: start,
        end_s: start + 0.04,
      },
      {
        id: `e${cycle}`,
        cycle,
        phase: "execution",
        start_s: start + 0.04,
        end_s: start + 0.44,
      },
    ];
  });
  return row;
}

test("synchronous auto window fits two recorded cycles and keeps the current execution producers visible", () => {
  const row = completeSynchronousCycles(),
    before = JSON.stringify(row);
  const span = T.synchronousSpan([row], 1.34);
  assert.ok(Math.abs(span - 0.89) < 1e-9);
  const start = T.viewport(0.1, 1.34, span);
  assert.ok(start.start <= 0 && start.end >= 0.89);
  const executing = T.viewport(0.7, 1.34, span);
  for (const block of [
    { start: 0.45, end: 0.47 }, // observed VLM producer
    { start: 0.47, end: 0.49 }, // observed DiT producer
    { start: 0.49, end: 0.89 }, // execution phase
  ]) {
    const geometry = T.geometry(block, executing);
    assert.equal(geometry.visible, true);
    assert.ok(geometry.left >= 0 && geometry.left + geometry.width <= 100);
  }
  assert.equal(JSON.stringify(row), before);
  assert.equal(T.synchronousSpan([row], 0.2), 0.2);
});

test("synchronous auto window falls back for missing or incomplete cycles without estimating their duration", () => {
  assert.equal(T.synchronousSpan([], 3), 0.3);
  for (const status of ["unknown", "incomplete", "running"]) {
    const row = completeSynchronousCycles();
    row.episode.phases[3].status = status;
    // Neither complete cycle on either side is adjacent to another complete one.
    assert.equal(T.synchronousSpan([row], 3), 0.3);
  }
  for (const end of [null, undefined, NaN, -1]) {
    const row = completeSynchronousCycles();
    row.episode.phases[3].end_s = end;
    assert.equal(T.synchronousSpan([row], 3), 0.3);
  }
  const row = completeSynchronousCycles();
  row.episode.phases = row.episode.phases.slice(0, 2);
  assert.equal(T.synchronousSpan([row], 3), 0.3);
  row.run.execution_coupling = "asynchronous";
  row.episode.phases = completeSynchronousCycles().episode.phases;
  assert.equal(T.synchronousSpan([row], 3), 0.3);
});

test("synchronous auto viewport retains the executing cycle sources near the end of unequal cycles", () => {
  const row = completeSynchronousCycles();
  row.episode.phases = [
    {
      id: "i0",
      cycle: 0,
      request_tick: 5,
      phase: "inference",
      start_s: 0,
      end_s: 0.15,
    },
    {
      id: "e0",
      cycle: 0,
      request_tick: 5,
      phase: "execution",
      start_s: 0.16,
      end_s: 0.56,
    },
    {
      id: "i1",
      cycle: 1,
      request_tick: 9,
      phase: "inference",
      start_s: 0.6,
      end_s: 0.7,
    },
    {
      id: "e1",
      cycle: 1,
      request_tick: 9,
      phase: "execution",
      start_s: 0.71,
      end_s: 3.71,
    },
  ];
  const before = JSON.stringify(row);
  for (const execution of row.episode.phases.filter(
    (p) => p.phase === "execution",
  )) {
    const cursor = execution.end_s - 1e-7;
    const view = T.synchronousViewport([row], cursor, 3.71);
    const inference = row.episode.phases.find(
      (p) => p.cycle === execution.cycle && p.phase === "inference",
    );
    assert.ok(view.start <= inference.start_s);
    assert.ok(view.end >= cursor && view.end <= 3.71);
    assert.ok(view.span >= cursor - inference.start_s);
  }
  assert.equal(JSON.stringify(row), before);
});

test("one recorded complete cycle anchors execution history without estimating a second cycle", () => {
  const row = synchronousRow();
  row.episode.phases = row.episode.phases.slice(0, 2);
  assert.equal(T.synchronousSpan([row], 0.56), 0.3);
  const cursor = 0.56 - 1e-7,
    view = T.synchronousViewport([row], cursor, 0.56);
  for (const block of T.sourceBlocks(T.tracks([row], "deployment"), "p1")) {
    assert.ok(view.start <= block.start);
    assert.ok(view.end >= block.end);
  }
  assert.equal(view.start, 0);
  assert.equal(view.end, 0.56);
  row.episode.phases[0].status = "unknown";
  assert.deepEqual(
    T.synchronousViewport([row], cursor, 0.56),
    T.viewport(cursor, 0.56, 0.3),
  );
  delete row.episode.phases;
  assert.deepEqual(
    T.synchronousViewport([row], cursor, 0.56),
    T.viewport(cursor, 0.56, 0.3),
  );
});
