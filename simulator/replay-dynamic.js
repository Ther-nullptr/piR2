/* Physical simulation playback and an independent measured wall-clock panel. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.DynamicReplay = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const finite = (v) => typeof v === "number" && Number.isFinite(v);
  const MAX_ROWS = 4;
  const episodes = (run) => run.episode_summaries ?? run.episodes ?? [];
  function durationBounds(e, lane) {
    if (Array.isArray(e.requests)) {
      const requests =
        lane === "all" ? e.requests : e.requests.filter((r) => r.lane === lane);
      if (
        !requests.length ||
        requests.some(
          (r) => !finite(r.service_duration_ms) || r.service_duration_ms < 0,
        )
      )
        return null;
      return requests.reduce(
        (bounds, r) => ({
          min: Math.min(bounds.min, r.service_duration_ms),
          max: Math.max(bounds.max, r.service_duration_ms),
        }),
        { min: Infinity, max: -Infinity },
      );
    }
    const bounds = e.request_duration_bounds_ms?.[lane];
    return finite(bounds?.min) &&
      finite(bounds?.max) &&
      bounds.min >= 0 &&
      bounds.max >= bounds.min
      ? { min: bounds.min, max: bounds.max }
      : null;
  }
  function episodeSummary(e) {
    // Match the catalog allowlist; no request, media, trace or policy identity references.
    const summary = Object.fromEntries(
      [
        "id",
        "item_id",
        "condition_id",
        "parent_dynamic_condition_id",
        "template_condition_id",
        "suite",
        "task_id",
        "level",
        "traj_index",
        "speed_mps",
        "seed",
        "init_state_id",
        "speed_scope",
        "seed_scope",
        "scene_drive",
        "scene_drive_scope",
        "protocol",
        "timing_mode",
        "fixed_delay",
        "execution_prefix_steps",
      ].map((key) => [key, e[key] ?? null]),
    );
    for (const [key, fields] of [
      ["stage_latencies", ["bootstrap_s", "s1_s", "s2_s"]],
      [
        "calibration",
        [
          "kind",
          "protocol",
          "retained_requests",
          "warmup_requests",
          "raw_latency_s",
          "applied_latency_s",
        ],
      ],
    ])
      summary[key] =
        e[key] && typeof e[key] === "object"
          ? Object.fromEntries(
              fields
                .filter((field) => Object.hasOwn(e[key], field))
                .map((field) => [field, e[key][field]]),
            )
          : null;
    summary.request_duration_bounds_ms = Object.fromEntries(
      ["all", "s1", "s2"].map((lane) => [lane, durationBounds(e, lane)]),
    );
    return summary;
  }
  function conditionKey(e) {
    return JSON.stringify([
      e.suite,
      e.task_id,
      e.level,
      e.traj_index,
      e.speed_mps,
      e.seed,
      e.init_state_id,
    ]);
  }
  function conditions(runs) {
    const found = new Map();
    for (const run of runs.filter((r) => r.kind === "dynamic"))
      for (const e of episodes(run))
        if (!found.has(conditionKey(e))) found.set(conditionKey(e), e);
    return [...found].map(([key, episode]) => ({ key, episode }));
  }
  const protocol = (run) =>
    run.protocol === "pir2_dynamic" ? "pir2_dynamic" : "default_vla_flow4";
  const protocolLabel = (run) =>
    protocol(run) === "pir2_dynamic" ? "πR²" : "Flow4";
  const groupKey = (run) => `${run.precision}:${protocol(run)}`;
  function candidates(runs, key, precision, selectedProtocol = null) {
    return runs
      .filter(
        (r) =>
          r.kind === "dynamic" &&
          r.precision === precision &&
          (!selectedProtocol || protocol(r) === selectedProtocol),
      )
      .flatMap((run) =>
        episodes(run)
          .filter((e) => conditionKey(e) === key)
          .map((episode) => ({ run, episode })),
      );
  }
  function rows(runs, key, precision, selection, comparisons = []) {
    const groups = new Map();
    for (const run of runs.filter((r) => r.kind === "dynamic"))
      groups.set(groupKey(run), run);
    const primary = [...groups]
      .filter(([, run]) => precision === "all" || run.precision === precision)
      .flatMap(([columnKey, run]) => {
        const found = candidates(runs, key, run.precision, protocol(run));
        const selected =
          found.find(
            (r) =>
              r.run.id === (selection[columnKey] ?? selection[run.precision]),
          ) ?? found[0];
        if (!selected) return [];
        return [{ ...selected, columnKey }];
      });
    if (primary.length > MAX_ROWS)
      throw new Error("最多显示 4 列；请选择单一精度");
    let available = MAX_ROWS - primary.length;
    return primary.flatMap((selected) => {
      const { columnKey, run } = selected;
      const found = candidates(runs, key, run.precision, protocol(run));
      if (!available || !comparisons.includes(columnKey) || found.length < 2)
        return [selected];
      const extraKey = columnKey + ":compare";
      const extra =
        found.find((r) => r.run.id === selection[extraKey]) ??
        found.find((r) => r.run.id !== run.id);
      available--;
      return [selected, { ...extra, columnKey: extraKey, comparison: true }];
    });
  }
  function cameraFrames(episode, camera = "image") {
    if (camera === "image")
      return episode.frames ?? episode.frames_by_camera?.image ?? [];
    return episode.frames_by_camera
      ? (episode.frames_by_camera[camera] ?? [])
      : (episode.frames ?? []);
  }
  function resolveEvent(episode, event) {
    if (!["s1", "s2"].includes(event.kind)) return event;
    const source =
      (episode.requests ?? []).find((r) => r.id === event.request_id) ??
      event.source;
    return { ...event, source };
  }
  function at(episode, time, camera = "image") {
    const frame =
      cameraFrames(episode, camera)
        .filter((f) => f.relative_s <= time)
        .at(-1) ?? null;
    const physics = (episode.events ?? []).filter((e) =>
      [
        "action",
        "bootstrap_hold",
        "startup_barrier",
        "inference_hold",
      ].includes(e.kind),
    );
    const active = physics.find(
      (e) => e.start_s <= time && (!finite(e.end_s) || time < e.end_s),
    );
    const event =
      active ??
      physics.filter((e) => finite(e.end_s) && e.end_s <= time).at(-1) ??
      null;
    const request =
      (episode.requests ?? []).find((r) => r.id === event?.request_id) ?? null;
    return {
      frame,
      event,
      request,
      ended: finite(episode.duration_s) && time >= episode.duration_s,
    };
  }
  function lineage(episode, requestId) {
    const result = [];
    while (requestId && !result.includes(requestId)) {
      result.push(requestId);
      requestId = episode.requests.find(
        (r) => r.id === requestId,
      )?.parent_request_id;
    }
    return result;
  }
  function wallPanel(request) {
    if (!request) return null;
    const stages = request.wall_stages ?? [];
    const values = stages
      .flatMap((s) => [s.wall_started_s, s.wall_completed_s])
      .filter(finite);
    const origin = values.length ? Math.min(...values) : null;
    const end = values.length ? (Math.max(...values) - origin) * 1000 : null;
    return {
      origin_s: origin,
      end_ms: end,
      stages: stages.map((s) => ({
        ...s,
        start_ms: finite(s.wall_started_s)
          ? (s.wall_started_s - origin) * 1000
          : null,
        end_ms: finite(s.wall_completed_s)
          ? (s.wall_completed_s - origin) * 1000
          : null,
      })),
    };
  }
  function observationLabel(r) {
    return r.lane === "s1"
      ? `状态 tick ${r.request_tick ?? "—"}`
      : r.lane === "s2"
        ? `图像 tick ${r.source_tick ?? "—"}`
        : `观测 action step ${r.source_tick ?? "—"}`;
  }
  function rpcScope(e) {
    if (e.protocol !== "pir2_dynamic" && e.timing_mode === "native")
      return "完整策略调用包含输入准备、RPC、输出处理及 NumPy 结果规范化；完整调用只有实测时长，RPC/server 使用各自绝对边界";
    return e.protocol === "pir2_dynamic"
      ? "RPC端点调用 / 含序列化，不含客户端预处理"
      : "RPC与server计时以记录边界为准";
  }
  function modelContract(e) {
    const horizon = `decoded ${e.decoded_horizon ?? "—"} / predicted ${e.predicted_horizon ?? "—"}`;
    if (e.protocol === "pir2_dynamic")
      return `每步执行 1 · 承诺 d=${e.delay_ticks_observed?.join(", ") || "—"} · ${horizon} · 预热 ${e.episode_warmup_requests ?? "—"} 轮 / ${e.episode_warmup_recorded ?? "—"} 组 (${e.warmup_rpc_recorded ?? "—"} 次 RPC)`;
    if (e.timing_mode === "native")
      return `K ${e.execution_prefix_steps ?? "—"} / ${horizon} · 每回合热身 ${e.episode_warmup_recorded ?? "—"} / 配置 ${e.episode_warmup_requests ?? "—"} 次 · 每次完整 Flow4 调用独立计时`;
    return `K ${e.execution_prefix_steps ?? "—"} / ${horizon} · 校准保留 ${e.calibration.retained_requests ?? "—"} 次 / 校准热身 ${e.calibration.warmup_requests ?? "—"} 次 · 每回合热身 ${e.episode_warmup_recorded ?? "—"} / 配置 ${e.episode_warmup_requests ?? "—"} 次`;
  }
  function timingLabel(e) {
    if (e.protocol !== "pir2_dynamic") {
      if (e.timing_mode === "native") {
        const bounds = durationBounds(e, "all");
        const range = bounds
          ? `${bounds.min.toFixed(1)}–${bounds.max.toFixed(1)}`
          : "—";
        return `逐请求实测延迟 · 完整 Flow4 ${range} ms · K ${e.execution_prefix_steps ?? "—"}`;
      }
      const kind =
        e.calibration?.kind === "measured"
          ? "实测校准延迟"
          : e.calibration?.kind === "counterfactual"
            ? "反事实固定延迟"
            : `校准类型 ${e.calibration?.kind ?? "未知"}`;
      const tau = e.calibration?.applied_latency_s;
      return `${kind} · τ ${finite(tau) ? (tau * 1000).toFixed(1) : "—"} ms`;
    }
    const policy =
      e.fixed_delay == null ? "responsive" : `fixed d=${e.fixed_delay}`;
    if (e.timing_mode === "native") {
      const range = (lane) => {
        const bounds = durationBounds(e, lane);
        if (!bounds) return "—";
        const low = bounds.min,
          high = bounds.max;
        return low === high
          ? low.toFixed(1)
          : `${low.toFixed(1)}–${high.toFixed(1)}`;
      };
      return `逐请求实测延迟 · S2 ${range("s2")} / S1 ${range("s1")} ms · ${policy}`;
    }
    const ms = (v) => (finite(v) ? (v * 1000).toFixed(0) : "—");
    return `反事实独立通道 · S2 ${ms(e.stage_latencies?.s2_s)} / S1 ${ms(e.stage_latencies?.s1_s)} ms · ${policy}`;
  }
  function timelineAxis(start, end) {
    return `<div class="task-axis"><span>物理仿真时间 / s</span><div class="task-scale">${Array.from({ length: 6 }, (_, i) => `<span style="left:${i * 20}%">${(start + ((end - start) * i) / 5).toFixed(3)} s</span>`).join("")}</div></div>`;
  }
  function averageActionHz(e) {
    return Number.isInteger(e.steps) &&
      e.steps >= 0 &&
      finite(e.duration_s) &&
      e.duration_s > 0
      ? e.steps / e.duration_s
      : null;
  }
  function motionLabel(e) {
    return e.scene_drive === "frozen" ? "外驱冻结对照" : "外驱运动";
  }
  function initializationLabel(e) {
    const speed = !finite(e.speed_mps)
      ? "—"
      : Number(e.speed_mps.toFixed(3)) === e.speed_mps
        ? e.speed_mps.toFixed(3)
        : String(e.speed_mps);
    return `${e.suite} / task ${e.task_id} · ${e.level} / traj ${e.traj_index} · 初始化速度 ${speed} m/s${e.speed_scope === "custom" ? "（自定义）" : ""} · seed ${e.seed ?? "—"}${e.seed_scope === "custom" ? "（自定义）" : ""} · 初态 ${e.init_state_id ?? "—"}`;
  }
  function startupLabel(e) {
    const ms = (v) => (finite(v) ? (v * 1000).toFixed(1) : "—");
    if (!finite(e.requested_startup_release_seconds))
      return `启动等待 ${ms(e.startup_seconds ?? e.stage_latencies?.bootstrap_s)} ms · 未设置共同起点`;
    const result =
      e.startup_alignment_met === true
        ? finite(e.startup_seconds_applied) &&
          e.startup_seconds_applied >=
            e.requested_startup_release_seconds - 1e-8
          ? "启动对齐已达标"
          : "启动对齐尚未完成"
        : e.startup_alignment_met === false
          ? "启动对齐未达标"
          : "启动对齐结果未知";
    return `共同起点 ${ms(e.requested_startup_release_seconds)} ms · bootstrap 调用 ${ms(e.bootstrap_service_seconds)} ms + 额外等待 ${ms(e.startup_barrier_seconds)} ms · ${result}（实际已推进 ${ms(e.startup_seconds_applied)} ms）`;
  }
  function stageNfe(request, stage) {
    const count = request.server_audit?.forward_counts?.dit;
    return ["DiT server plan", "Bootstrap VLM + Flow4", "Flow DiT"].includes(
      stage.name,
    ) && finite(count)
      ? count
      : null;
  }
  function phaseLabel(now, episode) {
    if (episode.status && episode.status !== "ok") return "回合失败";
    if (!now.event) return "无已记录阶段";
    if (now.ended) return "已结束";
    if (
      episode.scene_drive === "frozen" &&
      ["bootstrap_hold", "startup_barrier", "inference_hold"].includes(
        now.event.kind,
      )
    )
      return "推理等待 · 外驱冻结";
    if (now.event.kind === "bootstrap_hold") return "启动等待 · 目标移动";
    if (now.event.kind === "startup_barrier") return "共同起点等待 · 目标移动";
    if (now.event.kind === "action" && now.event.fallback)
      return "Fallback · 动作执行";
    if (now.event.kind === "inference_hold") return "等待中 · 目标移动";
    return now.event.kind === "action" ? "动作执行" : "无已记录阶段";
  }
  return {
    MAX_ROWS,
    episodeSummary,
    protocol,
    protocolLabel,
    groupKey,
    cameraFrames,
    resolveEvent,
    lineage,
    conditionKey,
    conditions,
    candidates,
    rows,
    at,
    wallPanel,
    timelineAxis,
    modelContract,
    timingLabel,
    averageActionHz,
    motionLabel,
    initializationLabel,
    startupLabel,
    stageNfe,
    observationLabel,
    rpcScope,
    phaseLabel,
  };
});
