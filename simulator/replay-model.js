/* Evidence interpretation only. No synthetic kernels, robot motion or inference. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.QuantReplay = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  function scaleMetric(value, factor) {
    if (!finite(value) || !finite(factor)) return null;
    const scaled = value * factor;
    return finite(scaled) ? scaled : null;
  }
  function lastAt(rows, time, key) {
    let found = null;
    for (const row of rows ?? [])
      if (
        finite(row[key]) &&
        row[key] <= time &&
        (!found || row[key] >= found[key])
      )
        found = row;
    return found;
  }
  function records(run, type) {
    const value =
      run?.[type === "profile" ? "profile_requests" : "clean_requests"];
    return Array.isArray(value) ? value : [];
  }
  const scopes = {
    policy: {
      label: "policy · 原始观测 → 解码动作",
      note: "完整本地调用：包含输入准备、传输、VLM/DiT 与动作解码；不含机器人 RPC/环境。",
    },
    preparation_transfer: {
      label: "preparation_transfer · 输入准备与传输",
      note: "仅原始观测 → 已准备 GPU 输入，不执行 VLM/DiT。",
    },
    prepared_model: {
      label: "prepared_model · 常驻输入 → 模型",
      note: "仅已准备的常驻输入 → VLM/DiT，CUDA 完成；不含输入准备或动作解码。",
    },
    preparation_model: {
      label: "preparation_model · 准备 + 模型（不含解码）",
      note: "Instrumented 输入准备 + VLM/DiT；不含动作解码，不等于 policy 边界。",
    },
  };
  function scopeOf(run, record) {
    if (typeof record?.scope === "string" && record.scope) return record.scope;
    const summary = run?.profile_summary;
    if (record?.instrumented && summary?.scope?.includes("full VLM")) {
      if (summary.input_preparation_included === false) return "prepared_model";
      if (summary.input_preparation_included === true)
        return "preparation_model";
    }
    return "unknown";
  }
  function scopeInfo(scope) {
    return (
      scopes[scope] ?? {
        label: `未知边界 · ${scope}`,
        note: "测量边界未知；保留原始记录，不推断模型执行或参与跨精度比较。",
      }
    );
  }
  function scopedRecords(run, type, scope) {
    return records(run, type).filter(
      (record) => scopeOf(run, record) === scope,
    );
  }
  function alignModel(runs, type, selected, selectedRunId = null) {
    const owner =
      selectedRunId != null
        ? runs.find((run) => run.id === selectedRunId)
        : runs.find((run) => records(run, type).includes(selected));
    const scope = scopeOf(owner, selected);
    const known = Boolean(
      typeof selected?.comparison_key === "string" &&
        selected.comparison_key.length > 0 &&
        typeof selected.observation_id === "string" &&
        selected.observation_id.length &&
        Number.isInteger(selected.seed) &&
        typeof selected.instrumented === "boolean" &&
        scopes[scope] &&
        typeof owner?.clock_cohort === "string" &&
        owner.clock_cohort.length,
    );
    return runs.map((run) => {
      const isOwner = run === owner;
      const comparable = known && run.clock_cohort === owner.clock_cohort;
      const record =
        records(run, type).find((row) => {
          // Own records remain inspectable even when instrumentation is unknown.
          if (isOwner && row.id === selected?.id) return true;
          if (
            row.instrumented !== (type === "profile") ||
            selected?.instrumented !== (type === "profile")
          )
            return false;
          return (
            comparable &&
            scopeOf(run, row) === scope &&
            row.comparison_key === selected.comparison_key &&
            row.observation_id === selected.observation_id &&
            row.seed === selected.seed
          );
        }) ?? null;
      return { run, record, comparable: Boolean(record && comparable) };
    });
  }
  function quantile(values, p) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b),
      i = (sorted.length - 1) * p,
      low = Math.floor(i);
    return sorted[low] + (sorted[Math.ceil(i)] - sorted[low]) * (i - low);
  }
  function cleanStats(run, scope = "prepared_model") {
    const values = scopedRecords(run, "clean", scope)
      .filter(
        (r) =>
          r.clean_wall === true &&
          r.instrumented === false &&
          finite(r.wall_ms),
      )
      .map((r) => r.wall_ms);
    return {
      count: values.length,
      p50: quantile(values, 0.5),
      p95: quantile(values, 0.95),
    };
  }
  function stateSamples(episode) {
    const out = [],
      initial = episode?.robot?.initial_state,
      first =
        episode?.robot?.frames?.find((f) => f.phase === "initial") ??
        episode?.robot?.frames?.[0];
    if (initial && finite(first?.relative_s))
      out.push({ time: first.relative_s, value: initial });
    for (const row of episode?.ticks ?? []) {
      const response = row.robot_response;
      if (
        response?.after &&
        finite(response.after_computed_s) &&
        finite(episode.controlled_start_s)
      )
        out.push({
          time: response.after_computed_s - episode.controlled_start_s,
          value: response.after,
        });
    }
    return out.sort((a, b) => a.time - b.time);
  }
  function episodeEnd(episode) {
    return finite(episode?.controlled_end_s) &&
      finite(episode?.controlled_start_s)
      ? episode.controlled_end_s - episode.controlled_start_s
      : (episode?.queue?.duration_s ?? null);
  }
  function normalizeEpisode(episode) {
    const metadata = episode.metadata ?? {};
    return {
      ...episode,
      task_id: episode.task_id ?? metadata.task_id,
      episode_id: episode.episode_id ?? metadata.episode_id,
      seed: episode.seed ?? metadata.seed,
    };
  }
  function deploymentCadence(run) {
    return run.execution_clock ?? run.configuration?.execution_clock ?? "paced";
  }
  function deploymentCoupling(run) {
    return (
      run.execution_coupling ??
      run.configuration?.execution_coupling ??
      "asynchronous"
    );
  }
  function replayDescriptors(
    descriptors,
    kind,
    cadence,
    coupling = "asynchronous",
  ) {
    return descriptors.filter(
      (run) =>
        run.kind === kind &&
        (kind !== "deployment" ||
          (deploymentCadence(run) === cadence &&
            deploymentCoupling(run) === coupling)),
    );
  }
  function playableEpisode(episode) {
    return Boolean(
      episode &&
        !["incomplete", "unavailable"].includes(episode.status) &&
        episode.queue &&
        Array.isArray(episode.ticks) &&
        episode.ticks.length &&
        finite(episodeEnd(episode)) &&
        episodeEnd(episode) > 0,
    );
  }
  function robotAt(episode, time, camera) {
    const end = episodeEnd(episode),
      at = finite(end) ? Math.min(time, end) : time;
    return {
      frame: lastAt(episode?.robot?.frames, at, "relative_s"),
      tick: lastAt(episode?.ticks, at, "relative_s"),
      robotState: lastAt(stateSamples(episode), at, "time")?.value ?? null,
      media: episode?.robot?.media?.[camera] ?? null,
      ended: finite(end) && time >= end,
      end,
    };
  }
  function phaseAt(episode, time) {
    return (
      episode?.phases?.find(
        (phase) =>
          finite(phase.start_s) &&
          finite(phase.end_s) &&
          phase.start_s <= time &&
          time < phase.end_s,
      ) ?? null
    );
  }
  function physicsAt(episode, time) {
    const ticks = episode?.ticks;
    if (
      !finite(episode?.controlled_start_s) ||
      !finite(time) ||
      !Array.isArray(ticks) ||
      !ticks.length ||
      !ticks.every(
        (tick) =>
          finite(tick.next_observation_computed_s) &&
          finite(tick.simulated_elapsed_seconds),
      )
    )
      return null;
    const completed = ticks.filter(
      (tick) =>
        finite(tick.next_observation_computed_s) &&
        tick.next_observation_computed_s - episode.controlled_start_s <= time,
    );
    return completed.length
      ? (completed.at(-1).simulated_elapsed_seconds ?? null)
      : 0;
  }
  const channels = {
    x: ["position_xyz_m", 0, "m"],
    y: ["position_xyz_m", 1, "m"],
    z: ["position_xyz_m", 2, "m"],
    rx: ["rotation_rotvec_rad", 0, "rad (rotation vector)"],
    ry: ["rotation_rotvec_rad", 1, "rad (rotation vector)"],
    rz: ["rotation_rotvec_rad", 2, "rad (rotation vector)"],
    g0: ["gripper_qpos", 0, "joint qpos"],
    g1: ["gripper_qpos", 1, "joint qpos"],
  };
  function robotSeries(episode, channel) {
    if (channel.startsWith("command:"))
      return (episode?.ticks ?? []).map((r) => [
        r.relative_s,
        r.action?.[Number(channel.split(":")[1])] ?? null,
      ]);
    const info = channels[channel];
    return info
      ? stateSamples(episode).map((s) => [
          s.time,
          s.value[info[0]]?.[info[1]] ?? null,
        ])
      : [];
  }
  function pairedEpisode(a, b) {
    return Boolean(
      a?.paired_fixture_key &&
        a.paired_fixture_key === b?.paired_fixture_key &&
        ["task_id", "episode_id", "seed"].every(
          (k) => a[k] != null && a[k] === b[k],
        ),
    );
  }
  function outputAt(record, timeMs) {
    if (!record)
      return {
        status: "unavailable",
        reason: "No matching input / seed / record",
      };
    const end = record.instrumented ? record.host_wall_ms : record.wall_ms;
    if (finite(end) && timeMs < end)
      return {
        status: "pending",
        reason: "Request has not completed at this cursor",
      };
    return (
      record.output ?? { status: "unavailable", reason: "Output not recorded" }
    );
  }
  function latestLoader() {
    let generation = 0;
    return {
      invalidate() {
        generation++;
      },
      async load(fn) {
        const current = ++generation;
        try {
          const value = await fn();
          return current === generation
            ? { current: true, value }
            : { current: false };
        } catch (error) {
          return current === generation
            ? { current: true, error }
            : { current: false };
        }
      },
    };
  }
  return {
    scopeOf,
    scopeInfo,
    scopedRecords,
    normalizeEpisode,
    deploymentCadence,
    deploymentCoupling,
    phaseAt,
    physicsAt,
    replayDescriptors,
    playableEpisode,
    finite,
    scaleMetric,
    lastAt,
    records,
    alignModel,
    quantile,
    cleanStats,
    stateSamples,
    episodeEnd,
    robotAt,
    robotSeries,
    channels,
    pairedEpisode,
    outputAt,
    latestLoader,
  };
});
