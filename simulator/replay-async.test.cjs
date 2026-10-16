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
