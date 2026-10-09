/* Pure replay interpretation. Timestamps are seconds relative to control_start. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.MeasuredReplay = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const age = (time, capture) =>
    capture == null ? null : (time - capture) * 1000;
  const latest = (rows, time) => rows.filter((r) => r.t <= time).at(-1);
  function dependencies(episode, a = 0, b = episode.duration_s) {
    const parents = new Map(),
      actions = new Map();
    const visual = episode.requests
      .filter((r) => r.role === "S2")
      .sort((x, y) => x.start_s - y.start_s);
    let visible = 0,
      prior = 0;
    for (const r of visual) {
      if (r.feature_sequence == null || r.start_s > b || r.end_s < a) continue;
      const index = String(visible++).padStart(2, "0");
      parents.set(r.feature_sequence, {
        label: `S2-${index}`,
        index,
        request: r,
      });
    }
    const initial = episode.initial_feature ?? {};
    for (const r of episode.requests
      .filter((r) => r.role === "S1")
      .sort((x, y) => x.start_s - y.start_s)) {
      const seq = r.feature_sequence;
      if (seq != null && !parents.has(seq)) {
        const index = `pre${String(++prior).padStart(2, "0")}`;
        const request = visual.find((v) => v.feature_sequence === seq);
        parents.set(seq, {
          label: `S2-${index}`,
          index,
          request,
          bootstrap: seq === initial.feature_sequence,
          historical: true,
        });
      }
      const parent = parents.get(seq);
      const k = parent ? (parent.children = (parent.children ?? 0) + 1) : null;
      actions.set(r.request_tick, {
        request: r,
        parent,
        label: parent ? `S1[${parent.index}].${k}` : "S1[?]",
        k,
      });
    }
    return { parents, actions };
  }
  function stateAt(episode, time) {
    const tick = latest(episode.ticks, time);
    const initial = episode.initial_feature ?? {};
    const feature = latest(episode.feature_updates ?? [], time) ?? {
      feature_sequence: initial.feature_sequence,
      source_tick: initial.feature_source_tick,
      capture_s: initial.feature_capture_s,
      t: 0,
    };
    const camera =
      latest(episode.camera_events ?? [], time)?.camera_queue ?? {};
    const requests = episode.requests.filter((r) => r.role === "S1");
    const slot = tick?.executed_slot;
    const executedRequest =
      slot?.producer_kind === "request"
        ? requests.find((r) => r.request_tick === slot.producer_request_tick)
        : null;
    let imageCapture = null,
      stateCapture = null;
    if (tick && !tick.fallback && !slot?.fallback) {
      if (slot?.producer_kind === "bootstrap") {
        imageCapture = initial.feature_capture_s ?? null;
        stateCapture = episode.initial_capture_s;
      } else if (executedRequest) {
        imageCapture = executedRequest.feature_capture_s;
        stateCapture = executedRequest.state_capture_s;
      }
    }
    const transition = latest(episode.delay_transitions ?? [], time);
    return {
      tick,
      feature,
      camera,
      executedRequest,
      cacheAgeMs: age(time, feature.capture_s),
      imageAgeMs: age(time, imageCapture),
      stateAgeMs: age(time, stateCapture),
      delay: transition ? transition.after : episode.initial_delay,
      inflight: requests.filter(
        (r) =>
          r.start_s <= time && (r.completed_s == null || r.completed_s > time),
      ),
      ready: requests.filter(
        (r) =>
          r.completed_s != null &&
          r.completed_s <= time &&
          (r.adopted_s == null || r.adopted_s > time),
      ),
    };
  }
  function aoiSeries(episode, a, b) {
    const times = [
      ...new Set([
        a,
        b,
        ...(episode.feature_updates ?? []).map((r) => r.t),
        ...episode.ticks.map((r) => r.t),
      ]),
    ]
      .filter((t) => t >= a && t <= b)
      .sort((x, y) => x - y);
    const out = { cache: [], image: [], state: [], used: [] };
    for (const t of times) {
      // Left/right limits preserve the genuine age drop at publication/execution.
      for (const sample of t === a ? [t] : [t - 1e-9, t]) {
        const s = stateAt(episode, sample);
        for (const [key, field] of [
          ["cache", "cacheAgeMs"],
          ["image", "imageAgeMs"],
          ["state", "stateAgeMs"],
        ])
          out[key].push([
            t,
            s[field] == null ? null : s[field] + (t - sample) * 1000,
          ]);
      }
    }
    out.used = episode.requests
      .filter(
        (r) =>
          r.role === "S1" && r.use_s != null && r.use_s >= a && r.use_s <= b,
      )
      .map((r) => ({
        t: r.use_s,
        request_tick: r.request_tick,
        imageMs: age(r.use_s, r.feature_capture_s),
        stateMs: age(r.use_s, r.state_capture_s),
      }));
    return out;
  }
  function calibrationMessage(condition, episode) {
    const s = condition.summary,
      h = condition.hardware;
    return {
      type: "pir2:load-solo-calibration",
      version: 1,
      conditionId: condition.id,
      layout: condition.layout,
      solo: { actionMs: s.s1_solo_mean_ms, vlmMs: s.s2_solo_mean_ms },
      periodMs: episode.period_s * 1000,
      cameraHz: 1 / episode.period_s,
      initialDelay: episode.initial_delay,
      hardware: {
        roleGpus: h.role_gpus ?? { S1: s.s1_gpu, S2: s.s2_gpu },
        rendererGpu: h.renderer_gpu,
        s1EffectiveSmMhz: s.s1_effective_sm_mhz,
        s2EffectiveSmMhz: s.s2_effective_sm_mhz,
        s1PowerLimitW: s.s1_power_limit_w,
        s2PowerLimitW: s.s2_power_limit_w,
      },
    };
  }
  return { dependencies, stateAt, aoiSeries, calibrationMessage };
});
