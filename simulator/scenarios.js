/* Scenario transforms and information-age accounting; never model inference. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.PiR2Scenarios = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const defaults = Object.freeze({
    visionPrecision: "BF16",
    actionPrecision: "BF16",
    visionSpeedup: 1,
    actionSpeedup: 1,
    visionOverheadMs: 0,
    actionOverheadMs: 0,
  });
  function bounded(value, min, max, name) {
    if (
      typeof value !== "number" ||
      !Number.isFinite(value) ||
      value < min ||
      value > max
    )
      throw new RangeError(`${name} must be in [${min},${max}]`);
    return value;
  }
  function projectScenario(base, settings = {}) {
    const s = { ...defaults, ...settings },
      config = { ...base },
      components = {};
    for (const [side, prefix, key] of [
      ["A", "action", "actionMs"],
      ["V", "vision", "vlmMs"],
    ]) {
      const speed = bounded(s[prefix + "Speedup"], 0.1, 16, prefix + "Speedup");
      const overhead = bounded(
        s[prefix + "OverheadMs"],
        0,
        1000,
        prefix + "OverheadMs",
      );
      const solo = bounded(base[key], 0.1, 5000, key),
        share = bounded(base["gpuShare" + side], 0, 1, "gpuShare" + side);
      const host = solo * (1 - share),
        gpu = solo * share,
        newGpu = gpu / speed,
        total = host + overhead + newGpu;
      bounded(total, 0.1, 5000, key + " after scaling");
      config[key] = total;
      config["gpuShare" + side] = newGpu / total;
      components[prefix] = {
        baseMs: solo,
        baseHostMs: host,
        baseGpuMs: gpu,
        hostMs: host + overhead,
        gpuMs: newGpu,
        totalMs: total,
        speedupAssumption: speed,
        extraHostMs: overhead,
        precisionLabel: s[prefix + "Precision"],
      };
    }
    return { config, components, assumptions: s };
  }
  function classify(vision, action, visionBudget = 100, actionBudget = 50) {
    bounded(visionBudget, 0.1, 10000, "vision budget");
    bounded(actionBudget, 0.1, 10000, "action budget");
    if (!Number.isFinite(vision) || !Number.isFinite(action))
      return {
        id: "unknown",
        label: "样本不足",
        visionMet: null,
        actionMet: null,
      };
    const v = vision <= visionBudget,
      a = action <= actionBudget;
    const id = v
      ? a
        ? "both_met"
        : "action_late"
      : a
        ? "vision_late"
        : "both_late";
    return {
      id,
      label: {
        both_met: "两侧均满足预算",
        action_late: "VLM 达标 · DiT 超预算",
        vision_late: "VLM 超预算 · DiT 达标",
        both_late: "两侧均超预算",
      }[id],
      visionMet: v,
      actionMet: a,
    };
  }
  function ageSegments(updates, start, end) {
    const rows = updates.slice().sort((a, b) => a.timeMs - b.timeMs),
      out = [];
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i],
        a = Math.max(start, row.timeMs),
        b = Math.min(end, rows[i + 1]?.timeMs ?? end);
      if (b <= a || !Number.isFinite(row.generationMs)) continue;
      if (row.generationMs > a + 1e-6)
        throw new RangeError("Future source timestamp");
      out.push({ startMs: a, endMs: b, generationMs: row.generationMs });
    }
    return out;
  }
  function ageAt(segments, timeMs) {
    const row = segments.find((s) => s.startMs <= timeMs && timeMs < s.endMs);
    return row ? Math.max(0, timeMs - row.generationMs) : null;
  }
  function ageStatistics(segments) {
    const duration = segments.reduce((s, r) => s + r.endMs - r.startMs, 0);
    if (!duration)
      return { knownMs: 0, meanMs: null, p95Ms: null, maxMs: null };
    const mean =
      segments.reduce(
        (s, r) =>
          s +
          ((r.startMs + r.endMs) / 2 - r.generationMs) * (r.endMs - r.startMs),
        0,
      ) / duration;
    let lo = Math.min(...segments.map((r) => r.startMs - r.generationMs)),
      hi = Math.max(...segments.map((r) => r.endMs - r.generationMs));
    const max = hi;
    for (let i = 0; i < 55; i++) {
      const mid = (lo + hi) / 2,
        mass = segments.reduce(
          (s, r) =>
            s +
            Math.max(
              0,
              Math.min(r.endMs - r.startMs, mid - (r.startMs - r.generationMs)),
            ),
          0,
        );
      if (mass / duration < 0.95) lo = mid;
      else hi = mid;
    }
    return {
      knownMs: duration,
      meanMs: mean,
      p95Ms: (lo + hi) / 2,
      maxMs: max,
    };
  }
  function simulationAoi(result) {
    const stop = result.windowEndMs;
    const cacheUpdates = [
      { timeMs: 0, generationMs: null },
      ...result.visualJobs
        .filter((j) => j.finishMs <= stop)
        .map((j) => ({ timeMs: j.finishMs, generationMs: j.captureMs })),
    ];
    const image = [{ timeMs: 0, generationMs: null }],
      state = [{ timeMs: 0, generationMs: null }],
      uses = [];
    for (const r of result.actionRequests)
      if (r.startMs < stop && r.featureVersion >= 0)
        uses.push({
          timeMs: r.startMs,
          ageMs: r.startMs - r.featureCaptureMs,
          requestId: r.id,
        });
    for (const e of result.executions) {
      const request =
        e.origin.kind === "request"
          ? result.actionRequests[e.origin.requestId]
          : null;
      image.push({
        timeMs: e.timeMs,
        generationMs:
          request && request.featureVersion >= 0
            ? request.featureCaptureMs
            : null,
      });
      state.push({
        timeMs: e.timeMs,
        generationMs: request ? request.stateCaptureMs : null,
      });
    }
    const cache = ageSegments(cacheUpdates, 0, stop),
      executionImage = ageSegments(image, 0, stop),
      executionState = ageSegments(state, 0, stop);
    return {
      cache,
      executionImage,
      executionState,
      uses,
      summary: {
        cache: ageStatistics(cache),
        executionImage: ageStatistics(executionImage),
        executionState: ageStatistics(executionState),
      },
    };
  }
  function dependencyAliases(result, start, end) {
    const first =
      result.visualJobs.find((j) => j.startMs < end && j.finishMs > start) ||
      result.visualJobs.find((j) => j.startMs >= start);
    const base = first?.id ?? (result.visualJobs.at(-1)?.id ?? -1) + 1;
    const ordinals = new Map(),
      counts = new Map();
    for (const r of result.actionRequests) {
      const n = (counts.get(r.featureVersion) || 0) + 1;
      counts.set(r.featureVersion, n);
      ordinals.set(r.id, n);
    }
    const parent = (id) =>
      id < 0
        ? "boot"
        : id - base < 0
          ? `pre${String(base - id).padStart(2, "0")}`
          : String(id - base).padStart(2, "0");
    return {
      base,
      parent,
      vision: (id) => `S2-${parent(id)}`,
      action: (r) => ({
        label: `S1[${parent(r.featureVersion)}].${ordinals.get(r.id)}`,
        parentFeature: r.featureVersion,
        ordinal: ordinals.get(r.id),
      }),
    };
  }
  function safeSoloMessage(data) {
    if (
      !data ||
      data.type !== "pir2:load-solo-calibration" ||
      data.version !== 1 ||
      !["single", "dual"].includes(data.layout)
    )
      return null;
    try {
      const actionMs = bounded(data.solo?.actionMs, 0.1, 5000, "actionMs"),
        vlmMs = bounded(data.solo?.vlmMs, 0.1, 5000, "vlmMs");
      const periodMs = bounded(data.periodMs, 5, 1000, "periodMs"),
        cameraHz = bounded(data.cameraHz, 0, 240, "cameraHz");
      return {
        conditionId: String(data.conditionId || "").slice(0, 160),
        layout: data.layout,
        config: {
          actionMs,
          vlmMs,
          periodMs,
          cameraHz,
          computeMode: data.layout === "single" ? "shared" : "independent",
        },
      };
    } catch {
      return null;
    }
  }
  return {
    defaults,
    projectScenario,
    classify,
    ageSegments,
    ageAt,
    ageStatistics,
    simulationAoi,
    dependencyAliases,
    safeSoloMessage,
  };
});
