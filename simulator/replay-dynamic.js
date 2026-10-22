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
  function candidates(runs, key, precision) {
    return runs
      .filter((r) => r.kind === "dynamic" && r.precision === precision)
      .flatMap((run) =>
        (run.episodes ?? [])
          .filter((e) => conditionKey(e) === key)
          .map((episode) => ({ run, episode })),
      );
  }
  function rows(runs, key, precision, selection) {
    const precisions = [
      ...new Set(
        runs.filter((r) => r.kind === "dynamic").map((r) => r.precision),
      ),
    ];
    return precisions
      .filter((p) => precision === "all" || p === precision)
      .flatMap((p) => {
        const found = candidates(runs, key, p);
        const selected =
          found.find((r) => r.run.id === selection[p]) ?? found[0];
        return selected ? [selected] : [];
      });
  }
  function at(episode, time) {
    const frame =
      (episode.frames ?? []).filter((f) => f.relative_s <= time).at(-1) ?? null;
    const active = (episode.events ?? []).find(
      (e) => e.start_s <= time && (!finite(e.end_s) || time < e.end_s),
    );
    const event =
      active ??
      (episode.events ?? [])
        .filter((e) => finite(e.end_s) && e.end_s <= time)
        .at(-1) ??
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
  function timelineAxis(start, end) {
    return `<div class="task-axis"><span>物理仿真时间 / s</span><div class="task-scale">${Array.from({ length: 6 }, (_, i) => `<span style="left:${i * 20}%">${(start + ((end - start) * i) / 5).toFixed(3)} s</span>`).join("")}</div></div>`;
  }
  function phaseLabel(now, episode) {
    if (episode.status !== "ok") return "回合失败";
    if (!now.event) return "无已记录阶段";
    if (now.ended) return "已结束";
    if (now.event.kind === "inference_hold") return "等待中 · 目标移动";
    return now.event.kind === "action" ? "动作执行" : "无已记录阶段";
  }
  return {
    conditionKey,
    conditions,
    candidates,
    rows,
    at,
    wallPanel,
    timelineAxis,
    phaseLabel,
  };
});
