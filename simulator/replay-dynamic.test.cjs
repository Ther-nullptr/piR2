const test = require("node:test");
const assert = require("node:assert/strict");
const D = require("./replay-dynamic.js");

const episode = {
  id: "episode-a",
  suite: "libero_spatial",
  task_id: 0,
  level: "l1",
  traj_index: 0,
  speed_mps: 0.06,
  seed: 0,
  init_state_id: 0,
  duration_s: 0.15,
  frames: [
    {
      frame_id: "f0",
      relative_s: 0.018,
      media_time_s: 0,
      phase: "inference_hold",
    },
    {
      frame_id: "f1",
      relative_s: 0.034,
      media_time_s: 1 / 60,
      phase: "inference_hold",
    },
    {
      frame_id: "f2",
      relative_s: 0.118,
      media_time_s: 2 / 60,
      phase: "action",
    },
  ],
  events: [
    {
      id: "hold",
      kind: "inference_hold",
      start_s: 0,
      end_s: 0.1,
      request_id: "r8",
    },
    { id: "act", kind: "action", start_s: 0.1, end_s: 0.15, request_id: "r8" },
  ],
  requests: [
    {
      id: "r8",
      request_sequence: 8,
      wall_stages: [
        {
          name: "Fresh VLM",
          wall_started_s: 100.02,
          wall_completed_s: 100.12,
          duration_ms: 100,
        },
        {
          name: "Flow DiT",
          wall_started_s: 100.12,
          wall_completed_s: 100.38,
          duration_ms: 260,
        },
      ],
    },
  ],
};

test("simulation playback seeks the latest measured frame, including moving inference holds", () => {
  assert.equal(D.at(episode, 0).frame, null);
  assert.equal(D.at(episode, 0.033).frame.frame_id, "f0");
  assert.equal(D.at(episode, 0.04).frame.frame_id, "f1");
  assert.equal(D.at(episode, 0.04).event.kind, "inference_hold");
  assert.equal(D.at(episode, 0.1).event.kind, "action");
  assert.equal(D.at(episode, 0.14).request.id, "r8");
  assert.equal(D.at(episode, 0.2).ended, true);
});

test("wall panel uses its measured origin and never stretches stages to a simulation hold", () => {
  const panel = D.wallPanel(episode.requests[0]);
  assert.equal(panel.origin_s, 100.02);
  assert.ok(Math.abs(panel.end_ms - 360) < 1e-7);
  assert.ok(Math.abs(panel.stages[1].start_ms - 100) < 1e-7);
  assert.equal(panel.stages[1].duration_ms, 260);
  assert.equal(D.wallPanel(null), null);
  assert.equal(
    D.wallPanel({
      wall_stages: [
        { name: "missing", wall_started_s: null, wall_completed_s: null },
      ],
    }).stages[0].start_ms,
    null,
  );
});

test("condition selection never silently pairs another trajectory, speed or seed", () => {
  const same = { ...episode, id: "episode-b" };
  const other = { ...episode, id: "episode-c", speed_mps: 0.09 };
  const runs = [
    {
      id: "bf",
      kind: "dynamic",
      precision: "bf16",
      episodes: [episode, other],
    },
    { id: "w8", kind: "dynamic", precision: "w8a8", episodes: [same] },
    { id: "old", kind: "deployment", precision: "bf16", episodes: [episode] },
  ];
  assert.equal(D.conditions(runs).length, 2);
  assert.equal(D.rows(runs, D.conditionKey(episode), "all", {}).length, 2);
  assert.equal(D.rows(runs, D.conditionKey(other), "all", {}).length, 1);
  assert.notEqual(
    D.conditionKey(episode),
    D.conditionKey({ ...episode, seed: 1 }),
  );
});

test("timeline axis closes before independently rendered lanes", () => {
  const html = D.timelineAxis(0, 1.5);
  const stack = [];
  for (const match of html.matchAll(/<\/?(div|span)\b[^>]*>/g)) {
    if (match[0].startsWith("</")) assert.equal(stack.pop(), match[1]);
    else stack.push(match[1]);
  }
  assert.deepEqual(stack, []);
  assert.ok(html.includes('class="task-scale"'));
});

test("missing evidence and failed episodes never claim executing actions", () => {
  assert.equal(
    D.phaseLabel({ event: null, ended: false }, { status: "ok" }),
    "无已记录阶段",
  );
  assert.equal(
    D.phaseLabel(
      { event: { kind: "action" }, ended: false },
      { status: "error" },
    ),
    "回合失败",
  );
  assert.equal(
    D.phaseLabel(
      { event: { kind: "inference_hold" }, ended: false },
      { status: "ok" },
    ),
    "等待中 · 目标移动",
  );
});
