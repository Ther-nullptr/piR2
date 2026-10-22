/* Physical simulation playback and an independent measured wall-clock panel. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.DynamicReplay = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const finite = (v) => typeof v === "number" && Number.isFinite(v);
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
      for (const e of run.episodes ?? []) found.set(conditionKey(e), e);
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
        (run.episodes ?? [])
          .filter((e) => conditionKey(e) === key)
          .map((episode) => ({ run, episode })),
      );
  }
  function rows(runs, key, precision, selection, comparisons = []) {
    const groups = new Map();
    for (const run of runs.filter((r) => r.kind === "dynamic"))
      groups.set(groupKey(run), run);
    return [...groups]
      .filter(([, run]) => precision === "all" || run.precision === precision)
      .flatMap(([columnKey, run]) => {
        const found = candidates(runs, key, run.precision, protocol(run));
        const selected =
          found.find(
            (r) =>
              r.run.id === (selection[columnKey] ?? selection[run.precision]),
          ) ?? found[0];
        if (!selected) return [];
        const result = [{ ...selected, columnKey }];
        if (comparisons.includes(columnKey) && found.length > 1) {
          const extraKey = columnKey + ":compare";
          const extra =
            found.find((r) => r.run.id === selection[extraKey]) ??
            found.find((r) => r.run.id !== selected.run.id);
          result.push({ ...extra, columnKey: extraKey, comparison: true });
        }
        return result;
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
      ["action", "bootstrap_hold", "inference_hold"].includes(e.kind),
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
    return e.protocol === "pir2_dynamic"
      ? "RPC端点调用 / 含序列化，不含客户端预处理"
      : "RPC与server计时以记录边界为准";
  }
  function modelContract(e) {
    const horizon = `decoded ${e.decoded_horizon ?? "—"} / predicted ${e.predicted_horizon ?? "—"}`;
    if (e.protocol === "pir2_dynamic")
      return `每步执行 1 · 承诺 d=${e.delay_ticks_observed?.join(", ") || "—"} · ${horizon} · 预热 ${e.episode_warmup_requests ?? "—"} 轮 / ${e.episode_warmup_recorded ?? "—"} 组 (${e.warmup_rpc_recorded ?? "—"} 次 RPC)`;
    return `K ${e.execution_prefix_steps ?? "—"} / ${horizon} · 校准保留 ${e.calibration.retained_requests ?? "—"} 次 / 校准热身 ${e.calibration.warmup_requests ?? "—"} 次 · 每回合热身 ${e.episode_warmup_recorded ?? "—"} / 配置 ${e.episode_warmup_requests ?? "—"} 次`;
  }
  function timelineAxis(start, end) {
    return `<div class="task-axis"><span>物理仿真时间 / s</span><div class="task-scale">${Array.from({ length: 6 }, (_, i) => `<span style="left:${i * 20}%">${(start + ((end - start) * i) / 5).toFixed(3)} s</span>`).join("")}</div></div>`;
  }
  function phaseLabel(now, episode) {
    if (episode.status && episode.status !== "ok") return "回合失败";
    if (!now.event) return "无已记录阶段";
    if (now.ended) return "已结束";
    if (now.event.kind === "bootstrap_hold") return "启动等待 · 目标移动";
    if (now.event.kind === "action" && now.event.fallback)
      return "Fallback · 动作执行";
    if (now.event.kind === "inference_hold") return "等待中 · 目标移动";
    return now.event.kind === "action" ? "动作执行" : "无已记录阶段";
  }
  return {
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
    observationLabel,
    rpcScope,
    phaseLabel,
  };
});
