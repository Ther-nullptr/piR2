/* Phenomenological, event-driven teaching model; no robot-quality prediction. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.VLAModel = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const formats = {
    bf16: { label: "BF16", w: 16, a: 16, math: 1 },
    int8: { label: "INT8 · W8A8", w: 8, a: 8, math: 2 },
    fp8: { label: "FP8 · W8A8", w: 8, a: 8, math: 2 },
    w4: { label: "INT4 · W4A16", w: 4, a: 16, math: 1 },
  };
  const defaults = {
    qA: "bf16",
    qV: "bf16",
    group: 128,
    compute: 1,
    bandwidth: 1,
    hostScale: 1,
    ratioA: 0.5,
    ratioV: 2,
    overhead: 8,
    lowbitMath: 2,
    period: 40,
    cameraHz: 30,
    slowCap: 0,
    seconds: 12,
    burnMs: 1000,
    policy: "concurrent",
    fastQueue: "drop",
    slowQueue: "latest",
    source: "camera",
    kA: 0.55,
    kV: 0.94,
    contention: "fixed",
    jitter: 0.1,
    seed: 1234,
    fastEnabled: true,
  };
  // Solo complete-call medians and a separate diagnostic profile. Their
  // difference is a fitted residual, NOT a measured pure CPU component.
  const anchors = {
    S1: { total: 16.121612, gpu: 5.343662, linear: 4.092607, weightShare: 0.9 },
    S2: {
      total: 22.506812,
      gpu: 9.070782,
      linear: 5.093758,
      weightShare: 0.65,
    },
  };

  function serviceModel(settings) {
    const c = { ...defaults, ...settings };
    const result = {};
    for (const [role, anchor] of Object.entries(anchors)) {
      const q = formats[role === "S1" ? c.qA : c.qV];
      const ratio = role === "S1" ? c.ratioA : c.ratioV;
      const mathGain = q.a < 16 ? c.lowbitMath : 1;
      const scaleBytes = q.w < 16 ? 2 / c.group : 0;
      const weightRatio = (q.w / 8 + scaleBytes) / 2;
      const traffic =
        anchor.weightShare * weightRatio +
        ((1 - anchor.weightShare) * q.a) / 16;
      const computeTime =
        (anchor.linear * Math.min(1, ratio)) / (c.compute * mathGain);
      const memoryTime =
        (anchor.linear * Math.min(1, 1 / ratio) * traffic) / c.bandwidth;
      const linear = Math.max(computeTime, memoryTime);
      const other =
        (anchor.gpu - anchor.linear) / Math.sqrt(c.compute * c.bandwidth);
      const added =
        q.w < 16 ? (anchor.linear * c.overhead) / 100 / c.compute : 0;
      const residual = (anchor.total - anchor.gpu) * c.hostScale;
      result[role] = {
        linear,
        other,
        added,
        residual,
        gpu: linear + other + added,
        total: Math.max(0.001, linear + other + added + residual),
        traffic,
        limiting: computeTime >= memoryTime ? "计算分支" : "带宽分支",
        computeTime,
        memoryTime,
      };
    }
    return result;
  }

  function random(seed) {
    let state = seed >>> 0;
    return () => {
      state += 0x6d2b79f5;
      let x = state;
      x = Math.imul(x ^ (x >>> 15), x | 1);
      x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }
  function percentile(values, p) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b),
      k = (sorted.length - 1) * p;
    return (
      sorted[Math.floor(k)] * (1 - (k % 1)) + sorted[Math.ceil(k)] * (k % 1)
    );
  }
  const mean = (values) =>
    values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;

  function simulate(settings) {
    const c = { ...defaults, ...settings },
      service = serviceModel(c),
      rng = random(c.seed);
    const end = c.seconds * 1000,
      burn = Math.min(c.burnMs, end * 0.5),
      eps = 1e-7;
    const cost = c.serviceOverride || [service.S1.total, service.S2.total];
    const camPeriod = c.cameraHz > 0 ? 1000 / c.cameraHz : Infinity;
    const limitPeriod = c.slowCap > 0 ? 1000 / c.slowCap : 0;
    let time = 0,
      nextA = c.fastEnabled ? 0 : Infinity,
      nextCamera =
        c.source === "camera"
          ? camPeriod === Infinity
            ? Infinity
            : 0
          : Infinity;
    let nextSlow = 0,
      a = null,
      v = null,
      pending = null,
      capture = 0,
      generation = 0,
      idA = 0,
      idV = 0;
    let area = 0,
      populationArea = 0,
      overlapMs = 0,
      cameraCount = 0,
      replaced = 0,
      droppedNew = 0,
      iterations = 0;
    const queue = [],
      calls = [],
      drops = [],
      updates = [],
      timeline = [],
      age = [];
    const sampleCost = (i) =>
      Math.max(0.001, cost[i] * (1 + c.jitter * (2 * rng() - 1)));
    function startJobs() {
      if (!a && queue.length && (c.policy === "concurrent" || !v)) {
        const request = queue.shift(),
          work = sampleCost(0);
        a = {
          ...request,
          start: time,
          remaining: work,
          service: work,
          featureCapture: capture,
          version: generation,
        };
      }
      const haveFrame = c.source === "zero" || pending !== null;
      if (
        !v &&
        time < end - eps &&
        haveFrame &&
        time >= nextSlow - eps &&
        (c.policy === "concurrent" || !a)
      ) {
        const work = sampleCost(1);
        v = {
          id: idV++,
          start: time,
          capture: c.source === "zero" ? time : pending,
          remaining: work,
          service: work,
        };
        pending = null;
        nextSlow = time + limitPeriod;
      }
    }
    while (time < end - eps || a || queue.length || v) {
      if (++iterations > 300000)
        throw new Error("事件数超过上限，请降低到达频率或缩短模拟。");
      if (v && v.remaining <= eps) {
        capture = v.capture;
        generation++;
        updates.push({ ...v, finish: time, generation });
        v = null;
        if (time <= end) age.push([time, Math.max(0, time - capture)]);
      }
      if (a && a.remaining <= eps) {
        const response = time - a.release;
        calls.push({
          ...a,
          finish: time,
          response,
          wait: a.start - a.release,
          ageAtStart: a.start - a.featureCapture,
          ageAtReady: time - a.featureCapture,
          d: Math.max(1, Math.ceil(response / c.period - 1e-9)),
          miss: response > c.period + eps,
        });
        a = null;
      }
      while (nextA <= time + eps && nextA < end - eps) {
        const request = { id: idA++, release: nextA };
        if (c.fastQueue === "drop" && (a || queue.length))
          drops.push({ ...request, reason: "S1 忙" });
        else if (queue.length >= 8)
          drops.push({ ...request, reason: "等待队列已满" });
        else queue.push(request);
        nextA = idA * c.period;
      }
      while (nextCamera <= time + eps && nextCamera < end - eps) {
        cameraCount++;
        if (pending === null) pending = nextCamera;
        else if (c.slowQueue === "latest") {
          pending = nextCamera;
          replaced++;
        } else droppedNew++;
        nextCamera = cameraCount * camPeriod;
      }
      startJobs();
      if (time >= end - eps && !a && !v && !queue.length) break;
      const both = !!(a && v);
      const crossA = c.contention === "traffic" ? service.S2.traffic : 1;
      const crossV = c.contention === "traffic" ? service.S1.traffic : 1;
      const rateA = a ? 1 / (1 + (both ? c.kA * crossA : 0)) : 0;
      const rateV = v ? 1 / (1 + (both ? c.kV * crossV : 0)) : 0;
      const events = [];
      if (a) events.push(time + a.remaining / rateA);
      if (v) events.push(time + v.remaining / rateV);
      if (nextA < end - eps) events.push(nextA);
      if (nextCamera < end - eps) events.push(nextCamera);
      if (
        !v &&
        nextSlow > time + eps &&
        nextSlow < end &&
        (pending !== null || c.source === "zero")
      )
        events.push(nextSlow);
      if (time < end - eps) events.push(end);
      const next = Math.min(...events);
      if (!Number.isFinite(next) || next <= time)
        throw new Error("事件推进失败");
      const low = Math.max(time, burn),
        high = Math.min(next, end);
      if (high > low) {
        const dt = high - low;
        area += (low - capture) * dt + (dt * dt) / 2;
        populationArea += ((a ? 1 : 0) + queue.length) * dt;
        if (both) overlapMs += dt;
      }
      if (time < end) {
        const segment = {
          start: time,
          end: Math.min(next, end),
          a: a?.id ?? null,
          v: v?.id ?? null,
        };
        const previous = timeline.at(-1);
        if (
          previous &&
          previous.a === segment.a &&
          previous.v === segment.v &&
          previous.end === time
        )
          previous.end = segment.end;
        else timeline.push(segment);
      }
      if (time < end)
        age.push(
          [time, Math.max(0, time - capture)],
          [Math.min(next, end), Math.max(0, Math.min(next, end) - capture)],
        );
      if (a) a.remaining = Math.max(0, a.remaining - (next - time) * rateA);
      if (v) v.remaining = Math.max(0, v.remaining - (next - time) * rateV);
      time = next;
    }
    const eligible = calls.filter((x) => x.release >= burn - eps),
      rejected = drops.filter((x) => x.release >= burn - eps);
    const planned = eligible.length + rejected.length,
      misses = rejected.length + eligible.filter((x) => x.miss).length;
    const measure = end - burn;
    return {
      config: c,
      service,
      calls: eligible,
      drops: rejected,
      updates,
      timeline,
      age,
      planned,
      completed: eligible.length,
      dropped: rejected.length,
      misses,
      missRate: planned ? misses / planned : null,
      responseP50: percentile(
        eligible.map((x) => x.response),
        0.5,
      ),
      responseP95: percentile(
        eligible.map((x) => x.response),
        0.95,
      ),
      waitMean: mean(eligible.map((x) => x.wait)),
      dP95: percentile(
        eligible.map((x) => x.d),
        0.95,
      ),
      aoiMean: area / measure,
      actionAgeP95: percentile(
        eligible.map((x) => x.ageAtReady),
        0.95,
      ),
      cameraCount,
      replaced,
      droppedNew,
      updateHz:
        updates.filter((x) => x.finish >= burn && x.finish < end).length /
        (measure / 1000),
      overlapFraction: overlapMs / measure,
      populationMean: populationArea / measure,
      littleEstimate:
        (eligible.length / measure) *
        (mean(eligible.map((x) => x.response)) || 0),
      theoreticalAoi:
        c.source === "zero" &&
        !c.fastEnabled &&
        c.jitter === 0 &&
        c.slowCap === 0
          ? 1.5 * cost[1]
          : null,
    };
  }

  function correctionError({
    alpha = 0.8,
    bias = 0.03,
    sigma = 0.1,
    rho = 0.5,
    initial = 1,
    steps = 24,
  } = {}) {
    let variance = initial * initial,
      center = 0,
      covariance = 0;
    const rows = [{ step: 0, rmse: Math.sqrt(variance), bias: 0 }];
    for (let i = 1; i <= steps; i++) {
      variance =
        alpha * alpha * variance + sigma * sigma + 2 * alpha * rho * covariance;
      covariance = alpha * rho * covariance + sigma * sigma;
      center = alpha * center + bias;
      rows.push({
        step: i,
        rmse: Math.sqrt(Math.max(0, variance + center * center)),
        bias: center,
      });
    }
    return rows;
  }
  return {
    defaults,
    formats,
    anchors,
    serviceModel,
    simulate,
    correctionError,
    percentile,
  };
});
