const test = require("node:test");
const assert = require("node:assert/strict");
const D = require("./replay-dynamic.js");

test("NFE stays on the model stage when full adapter timing is present", () => {
  const request = { server_audit: { forward_counts: { dit: 5 } } };
  assert.equal(D.stageNfe(request, { name: "RPC" }), null);
  assert.equal(D.stageNfe(request, { name: "Full adapter call" }), null);
  assert.equal(D.stageNfe(request, { name: "DiT server plan" }), 5);
  assert.equal(D.stageNfe(request, { name: "Bootstrap VLM + Flow4" }), 5);
});

test("native timing labels preserve separate per-request S2/S1 ranges", () => {
  const label = D.timingLabel({
    protocol: "pir2_dynamic",
    timing_mode: "native",
    stage_latencies: null,
    requests: [
      { lane: "s2", service_duration_ms: 31.2 },
      { lane: "s2", service_duration_ms: 39.8 },
      { lane: "s1", service_duration_ms: 12.4 },
      { lane: "s1", service_duration_ms: 17.3 },
    ],
  });
  assert.ok(label.includes("逐请求实测"));
  assert.ok(label.includes("31.2–39.8"));
  assert.ok(label.includes("12.4–17.3"));
  assert.ok(!label.includes("反事实"));
  assert.ok(
    D.timingLabel({
      protocol: "pir2_dynamic",
      stage_latencies: { s2_s: 0.15, s1_s: 0.04 },
    }).includes("150 / S1 40"),
  );
});

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

test("same precision Flow and piR2 stay in independent columns with optional same-protocol comparison", () => {
  const runs = [
    {
      id: "flow",
      kind: "dynamic",
      precision: "bf16",
      protocol: "default_vla_flow4",
      episodes: [episode],
    },
    {
      id: "p1",
      kind: "dynamic",
      precision: "bf16",
      protocol: "pir2_dynamic",
      episodes: [episode],
    },
    {
      id: "p2",
      kind: "dynamic",
      precision: "bf16",
      protocol: "pir2_dynamic",
      episodes: [episode],
    },
  ];
  const key = D.conditionKey(episode);
  assert.deepEqual(
    D.rows(runs, key, "all", {}).map((r) => r.run.id),
    ["flow", "p1"],
  );
  const group = D.groupKey(runs[1]);
  const rows = D.rows(runs, key, "all", {}, [group]);
  assert.deepEqual(
    rows.map((r) => r.run.id),
    ["flow", "p1", "p2"],
  );
  assert.notEqual(rows[1].columnKey, rows[2].columnKey);
  assert.equal(D.candidates(runs, key, "bf16", "pir2_dynamic").length, 2);
});

test("streaming playback follows physics instead of the concurrent S2 lane and honors each camera map", () => {
  const e = {
    ...episode,
    protocol: "pir2_dynamic",
    events: [
      { id: "s2", kind: "s2", start_s: 0, end_s: 0.2, request_id: "v" },
      {
        id: "a",
        kind: "action",
        start_s: 0,
        end_s: 0.05,
        request_id: "s",
        source: { feature_age_s: 0.3 },
      },
    ],
    requests: [
      { id: "v", lane: "s2" },
      { id: "s", lane: "s1", parent_request_id: "v" },
    ],
    frames_by_camera: {
      image: episode.frames,
      wrist_image: [episode.frames[0]],
    },
  };
  assert.equal(D.at(e, 0.04).event.id, "a");
  assert.equal(D.at(e, 0.04, "wrist_image").frame.frame_id, "f0");
  assert.deepEqual(D.lineage(e, "s"), ["s", "v"]);
  assert.deepEqual(D.lineage(e, null), []);
  assert.equal(
    D.phaseLabel({ event: { kind: "bootstrap_hold" } }, e),
    "启动等待 · 目标移动",
  );
  assert.equal(
    D.phaseLabel(
      { event: { kind: "action", fallback: true } },
      { ...e, status: "ok" },
    ),
    "Fallback · 动作执行",
  );
});

test("streaming metadata distinguishes physical steps, committed delay and warmup groups", () => {
  const e = {
    protocol: "pir2_dynamic",
    decoded_horizon: 40,
    predicted_horizon: 40,
    delay_ticks_observed: [1, 2],
    episode_warmup_requests: 8,
    episode_warmup_recorded: 40,
    warmup_rpc_recorded: 120,
  };
  const label = D.modelContract(e);
  assert.match(label, /每步执行 1/);
  assert.match(label, /承诺 d=1, 2/);
  assert.match(label, /预热 8 轮 \/ 40 组 \(120 次 RPC\)/);
  assert.ok(!label.includes("校准"));
});

test("startup barrier is a physical wait with its own label and unmodified service", () => {
  const e = {
    protocol: "pir2_dynamic",
    timing_mode: "native",
    requested_startup_release_seconds: 0.4,
    startup_seconds: 0.4,
    startup_seconds_applied: 0.4,
    bootstrap_service_seconds: 0.125,
    startup_barrier_seconds: 0.275,
    startup_alignment_met: true,
    events: [
      { kind: "bootstrap_hold", start_s: 0, end_s: 0.125 },
      { kind: "startup_barrier", start_s: 0.125, end_s: 0.4 },
      { kind: "action", start_s: 0.4, end_s: 0.45 },
    ],
  };
  assert.equal(D.at(e, 0.3).event.kind, "startup_barrier");
  assert.match(D.phaseLabel(D.at(e, 0.3), e), /共同起点等待/);
  assert.match(D.startupLabel(e), /400\.0 ms.*125\.0 ms.*275\.0 ms/);
  assert.match(D.startupLabel(e), /对齐已达标/);
  assert.match(
    D.startupLabel({ ...e, startup_alignment_met: false }),
    /对齐未达标/,
  );
  assert.match(D.startupLabel({ startup_seconds: 0.12 }), /未设置共同起点/);
  assert.match(
    D.startupLabel({ ...e, startup_seconds_applied: 0.3 }),
    /对齐尚未完成/,
  );
});

test("average action rate includes inference and startup holds, not video frames", () => {
  assert.equal(
    D.averageActionHz({
      steps: 100,
      duration_s: 10,
      control_freq: 20,
      video_fps: 60,
    }),
    10,
  );
  assert.equal(
    D.averageActionHz({ steps: 100, duration_s: 5.4, control_freq: 20 }),
    100 / 5.4,
  );
  assert.equal(D.averageActionHz({ steps: 0, duration_s: 0 }), null);
  assert.equal(D.averageActionHz({ steps: 100, duration_s: null }), null);
});

test("S1 state tick and S2 image tick have distinct labels and RPC boundary", () => {
  assert.equal(
    D.observationLabel({ lane: "s1", request_tick: 9, feature_source_tick: 3 }),
    "状态 tick 9",
  );
  assert.equal(
    D.observationLabel({ lane: "s2", source_tick: 3 }),
    "图像 tick 3",
  );
  assert.match(D.rpcScope({ protocol: "pir2_dynamic" }), /不含客户端预处理/);
});

test("canonical streaming events resolve their request and legacy inline events remain readable", () => {
  const request = { id: "s1", lane: "s1", feature_sequence: 7 };
  const e = { requests: [request] };
  const canonical = { kind: "s1", request_id: "s1", id: "event" };
  assert.equal(D.resolveEvent(e, canonical).source, request);
  assert.equal(canonical.source, undefined);
  assert.equal(
    D.resolveEvent(e, { ...canonical, source: { feature_sequence: 7 } }).source,
    request,
  );
  const action = { kind: "action", source: { tick: 3 } };
  assert.equal(D.resolveEvent(e, action), action);
});

test("camera mappings prefer canonical image frames while retaining old duplicate layout support", () => {
  const image = [{ frame_id: "image" }],
    wrist = [{ frame_id: "wrist" }];
  assert.equal(
    D.cameraFrames(
      { frames: image, frames_by_camera: { wrist_image: wrist } },
      "image",
    ),
    image,
  );
  assert.equal(
    D.cameraFrames(
      { frames: image, frames_by_camera: { image, wrist_image: wrist } },
      "image",
    ),
    image,
  );
  assert.equal(
    D.cameraFrames(
      { frames_by_camera: { image, wrist_image: wrist } },
      "image",
    ),
    image,
  );
  assert.equal(
    D.cameraFrames(
      { frames: image, frames_by_camera: { wrist_image: wrist } },
      "wrist_image",
    ),
    wrist,
  );
  assert.deepEqual(
    D.cameraFrames({ frames: image, frames_by_camera: {} }, "wrist_image"),
    [],
  );
});

test("native Flow labels display varying actual calls and K instead of a scalar tau", () => {
  const episode = {
    protocol: "default_vla_flow4",
    timing_mode: "native",
    execution_prefix_steps: 2,
    decoded_horizon: 16,
    predicted_horizon: 40,
    requests: [{ service_duration_ms: 31.2 }, { service_duration_ms: 83.4 }],
    calibration: { kind: null, applied_latency_s: null },
    episode_warmup_recorded: 8,
    episode_warmup_requests: 8,
  };
  const label = D.timingLabel(episode);
  assert.ok(label.includes("逐请求实测"));
  assert.ok(label.includes("31.2–83.4"));
  assert.ok(label.includes("K 2"));
  assert.ok(!label.includes("τ"));
  const contract = D.modelContract(episode);
  assert.ok(contract.includes("decoded 16 / predicted 40"));
  assert.ok(contract.includes("K 2"));
  assert.ok(!contract.includes("校准保留"));
});

test("native Flow wall note states the full request boundary", () => {
  const scope = D.rpcScope({
    protocol: "default_vla_flow4",
    timing_mode: "native",
  });
  assert.match(scope, /NumPy/);
  assert.match(scope, /完整策略调用/);
  assert.match(scope, /RPC/);
});
