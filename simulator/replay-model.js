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
  function controlledEnd(episode) {
    return finite(episode?.controlled_end_s) &&
      finite(episode?.controlled_start_s)
      ? episode.controlled_end_s - episode.controlled_start_s
      : (episode?.queue?.duration_s ?? null);
  }
  function episodeEnd(episode) {
    return finite(episode?.work_end_s) && finite(episode?.controlled_start_s)
      ? episode.work_end_s - episode.controlled_start_s
      : controlledEnd(episode);
  }
  function timelineQueue(episode) {
    if (!episode?.queue) return null;
    return {
      ...episode.queue,
      requests: episode.queue.replay_requests ?? episode.queue.requests,
      duration_s: episodeEnd(episode),
    };
  }
  function activePhases(episode, time) {
    return (episode?.phases ?? []).filter(
      (phase) =>
        finite(phase.start_s) &&
        finite(phase.end_s) &&
        phase.start_s <= time &&
        time < phase.end_s,
    );
  }
  function controlIntervals(episode) {
    const origin = episode?.controlled_start_s;
    if (!finite(origin)) return [];
    const stop = episode.controlled_end_s;
    const steps = (episode.ticks ?? [])
      .filter(
        (tick) =>
          finite(tick.applied_s) &&
          tick.applied_s >= origin &&
          (!finite(stop) || tick.applied_s <= stop),
      )
      .map((tick) => ({
        id: tick.id ?? `${episode.id}/tick-${tick.tick}`,
        kind: "execution",
        start_s: tick.applied_s - origin,
        end_s:
          finite(tick.next_observation_computed_s) &&
          tick.next_observation_computed_s >= tick.applied_s &&
          (!finite(stop) || tick.next_observation_computed_s <= stop)
            ? tick.next_observation_computed_s - origin
            : null,
        tick,
      }))
      .sort((a, b) => a.start_s - b.start_s);
    const intervals = [];
    function gap(start, end, before, after) {
      if (finite(start) && finite(end) && end > start)
        intervals.push({
          id: `${episode.id}/control-gap-${before ?? "initial"}`,
          kind: "gap",
          start_s: start,
          end_s: end,
          previous_tick: before,
          next_tick: after,
        });
    }
    for (const [index, step] of steps.entries()) {
      const previous = steps[index - 1];
      if (!previous && step.tick.tick === 0) gap(0, step.start_s, null, 0);
      else if (previous && step.tick.tick === previous.tick.tick + 1)
        gap(previous.end_s, step.start_s, previous.tick.tick, step.tick.tick);
      intervals.push(step);
    }
    const last = steps.at(-1);
    // Missing ticks/completions are unknown work, never inferred idle time.
    if (
      last &&
      finite(stop) &&
      Number.isInteger(episode.metadata?.control_steps) &&
      last.tick.tick + 1 === episode.metadata.control_steps
    )
      gap(last.end_s, stop - origin, last.tick.tick, null);
    return intervals;
  }
  function asyncStatus(episode, time) {
    const phases = activePhases(episode, time);
    const active = new Set(phases.map((p) => p.phase));
    const end = controlledEnd(episode),
      workEnd = episodeEnd(episode);
    if (finite(workEnd) && time >= workEnd)
      return "Robot stopped · all work complete";
    if (finite(end) && time >= end)
      return active.has("inference")
        ? "Robot stopped · unused inference / terminal drain"
        : "Robot stopped · terminal drain";
    if (phases.some((p) => p.phase === "inference" && p.request_id === 0))
      return "Bootstrap inference · simulation paused";
    if (active.has("inference") && active.has("execution"))
      return "Inferring + executing";
    if (active.has("wait"))
      return "Waiting for next result · simulation paused";
    if (active.has("execution")) return "Executing physical step";
    if (active.has("inference")) return "Inferring · between physical steps";
    return "Control transition";
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
  function policyVariant(run) {
    const claims = [];
    if (Object.hasOwn(run, "policy_variant")) claims.push(run.policy_variant);
    if (run.policy_identity && Object.hasOwn(run.policy_identity, "variant"))
      claims.push(run.policy_identity.variant);
    const config = run.configuration;
    if (config) {
      const workers = [config.checkpoint];
      if (deploymentCoupling(run) === "asynchronous")
        workers.push(config.slow_checkpoint);
      const values = [config.variant, ...workers.map((w) => w?.variant)];
      const known = values.filter((v) => v != null);
      if (known.some((v) => v !== known[0])) return "invalid";
      if (claims.some((v) => v != null && known.some((k) => k !== v)))
        return "invalid";
      if (values.every((v) => v != null)) claims.push(values[0]);
    }
    if (
      claims.some((v) => ![null, "pir2", "flow"].includes(v)) ||
      claims.some((v) => v !== claims[0])
    )
      return "invalid";
    return claims[0] ?? null;
  }
  function policyLabel(run) {
    const variant = policyVariant(run);
    if (variant === "pir2") {
      const step = run.policy_identity?.training?.global_step;
      return `πR²${Number.isInteger(step) && step > 0 ? ` · trained ${step} steps` : " · training provenance unknown"}`;
    }
    return variant === "flow" ? "GR00T Flow4" : "Policy identity unknown";
  }
  function policyNote(run) {
    if (policyVariant(run) !== "pir2") return policyLabel(run);
    const p = run.policy_identity ?? {};
    return `${policyLabel(run)}：独立 S2 VLM 更新视觉缓存，S1 使用实际消费的缓存与本轮新鲜本体状态，滚动缓冲常态每次 1 步 DiT 去噪并固定已承诺前缀。H${p.predicted_horizon ?? "?"} / 原生解码${p.decoded_horizon ?? "?"}；bootstrap 4 步与整缓冲恢复 5 步单独计数。K/D 完整策略预取不适用于此协议；轨道为实际 RPC 区间，不是 GPU 内核计时。`;
  }
  function isFullPolicyAsync(run) {
    return ["one_step_async", "chunked_async"].includes(
      deploymentCoupling(run),
    );
  }
  function deploymentSchedule(run) {
    const coupling = deploymentCoupling(run);
    if (coupling === "asynchronous") return null;
    const k =
      run.execution_prefix_steps ??
      run.configuration?.execution_prefix_steps ??
      run.configuration?.synchronous_chunk_steps ??
      (coupling === "one_step_async" ? 1 : null);
    const d =
      run.async_overlap_steps ??
      run.configuration?.async_overlap_steps ??
      (coupling === "synchronous"
        ? 0
        : coupling === "one_step_async"
          ? 1
          : null);
    if (
      !Number.isInteger(k) ||
      k < 1 ||
      !Number.isInteger(d) ||
      d < 0 ||
      d > k ||
      (coupling === "synchronous" && d !== 0) ||
      (coupling === "one_step_async" && (k !== 1 || d !== 1)) ||
      (coupling === "chunked_async" && d === 0)
    )
      return null;
    return { k, d };
  }
  function sameSchedule(a, b) {
    return a == null || b == null
      ? a == null && b == null
      : a.k === b.k && a.d === b.d;
  }
  function scheduleChoices(descriptors, cadence, coupling) {
    const choices = new Map();
    for (const run of replayDescriptors(
      descriptors,
      "deployment",
      cadence,
      coupling,
    )) {
      const schedule = deploymentSchedule(run);
      if (schedule) choices.set(`${schedule.k}/${schedule.d}`, schedule);
    }
    return [...choices.values()].sort((a, b) => a.k - b.k || a.d - b.d);
  }
  function selectSchedule(choices, preferred) {
    return (
      choices.find((s) => sameSchedule(s, preferred)) ??
      choices.find((s) => s.k === preferred?.k) ??
      choices[0] ??
      null
    );
  }
  function matchesDescriptor(
    run,
    descriptor,
    resolvedSchedule = undefined,
    resolvedPolicy = undefined,
  ) {
    if (
      run.id !== descriptor.id ||
      run.kind !== descriptor.kind ||
      run.precision !== descriptor.precision
    )
      return false;
    if (run.kind !== "deployment") return true;
    const expected = deploymentSchedule(descriptor);
    const policy = policyVariant(run),
      declaredPolicy = policyVariant(descriptor);
    const explicitPolicy =
      Object.hasOwn(descriptor, "policy_variant") ||
      Object.hasOwn(descriptor.policy_identity ?? {}, "variant") ||
      declaredPolicy != null;
    return (
      policy !== "invalid" &&
      declaredPolicy !== "invalid" &&
      (!explicitPolicy || policy === declaredPolicy) &&
      (resolvedPolicy === undefined || policy === resolvedPolicy) &&
      deploymentCoupling(run) === deploymentCoupling(descriptor) &&
      deploymentCadence(run) === deploymentCadence(descriptor) &&
      (expected == null || sameSchedule(deploymentSchedule(run), expected)) &&
      (resolvedSchedule === undefined ||
        sameSchedule(deploymentSchedule(run), resolvedSchedule))
    );
  }
  function currentPrefix(episode, tick) {
    if (!tick) return null;
    const work = episode.queue?.work_requests?.find(
      (r) =>
        Number.isInteger(tick.request_id) && r.request_id === tick.request_id,
    );
    if (work)
      return {
        native: work.decoded_horizon ?? null,
        accepted: work.accepted_prefix_steps ?? null,
        executed: work.executed_prefix_steps ?? null,
        offset: tick.executed_offset ?? null,
        lag: tick.observation_lag_ticks ?? null,
      };
    const cycle = episode.cycles?.find((r) => r.cycle === tick.cycle);
    if (!cycle) return null;
    const requests = (episode.queue?.requests ?? []).filter(
      (r) => r.role === "S1" && r.request_tick === cycle.request_tick,
    );
    const request = requests.length === 1 ? requests[0] : null;
    return {
      native: cycle.audit?.decoded_horizon ?? null,
      accepted:
        Number.isInteger(request?.valid_start_tick) &&
        Number.isInteger(request?.valid_end_tick)
          ? request.valid_end_tick - request.valid_start_tick
          : null,
      executed: cycle.executed_steps ?? null,
      offset:
        Number.isInteger(tick.tick) && Number.isInteger(cycle.request_tick)
          ? tick.tick - cycle.request_tick
          : null,
      lag:
        Number.isInteger(tick.tick) &&
        Number.isInteger(cycle.audit?.source_tick)
          ? tick.tick - cycle.audit.source_tick
          : null,
    };
  }
  function waitStats(episode) {
    const unknown = { count: null, total: null, p95: null };
    if (!Array.isArray(episode?.phases)) return unknown;
    const waits = episode.phases.filter((p) => p.phase === "wait");
    if (
      waits.some(
        (p) => !finite(p.start_s) || !finite(p.end_s) || p.end_s < p.start_s,
      )
    )
      return unknown;
    const durations = waits.map((p) => p.end_s - p.start_s);
    return {
      count: durations.length,
      total: durations.reduce((a, b) => a + b, 0),
      p95: quantile(durations, 0.95),
    };
  }
  function replayDescriptors(
    descriptors,
    kind,
    cadence,
    coupling = "asynchronous",
    schedule = undefined,
  ) {
    return descriptors.filter(
      (run) =>
        run.kind === kind &&
        (kind !== "deployment" ||
          (deploymentCadence(run) === cadence &&
            deploymentCoupling(run) === coupling &&
            (!isFullPolicyAsync(run) || deploymentSchedule(run) != null) &&
            (schedule === undefined ||
              sameSchedule(deploymentSchedule(run), schedule)))),
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
    const end = controlledEnd(episode),
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
    policyVariant,
    policyLabel,
    policyNote,
    deploymentSchedule,
    isFullPolicyAsync,
    sameSchedule,
    scheduleChoices,
    selectSchedule,
    matchesDescriptor,
    currentPrefix,
    waitStats,
    phaseAt,
    activePhases,
    asyncStatus,
    timelineQueue,
    controlIntervals,
    controlledEnd,
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
