/* Event-driven πR² queue and resource timing; no robot-quality prediction. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.VLAModel = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
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

  // Queue/timestamp model of scripts/libero_wallclock.py and
  // libero_protocol_scheduler.py. These are scheduling facts, not model quality.
  const queueDefaults = Object.freeze({
    vlmMs: 140,
    actionMs: 80,
    periodMs: 50,
    cameraHz: 20,
    seconds: 4,
    jitter: 0,
    seed: 1234,
    delayMode: "adaptive",
    fixedDelay: 1,
    maxDelay: 5,
    marginMs: 5,
    bootstrapSlots: 40,
    computeMode: "independent",
    slowdownA: 0.5,
    slowdownV: 0.5,
    gpuShareA: 1,
    gpuShareV: 1,
  });

  function queueConfig(settings) {
    if (!settings || typeof settings !== "object" || Array.isArray(settings))
      throw new TypeError("Queue settings must be an object");
    const c = { ...queueDefaults, ...settings };
    const limits = {
      vlmMs: [0.1, 5000],
      actionMs: [0.1, 5000],
      periodMs: [5, 1000],
      cameraHz: [0, 240],
      seconds: [0.05, 60],
      jitter: [0, 0.9],
      seed: [0, 4294967295],
      fixedDelay: [1, 5],
      maxDelay: [1, 5],
      marginMs: [0, 250],
      bootstrapSlots: [0, 200],
      slowdownA: [0, 20],
      slowdownV: [0, 20],
      gpuShareA: [0, 1],
      gpuShareV: [0, 1],
    };
    for (const [key, [low, high]] of Object.entries(limits)) {
      if (
        typeof c[key] !== "number" ||
        !Number.isFinite(c[key]) ||
        c[key] < low ||
        c[key] > high
      )
        throw new RangeError(`${key} must be finite and in [${low}, ${high}]`);
    }
    for (const key of ["seed", "fixedDelay", "maxDelay", "bootstrapSlots"])
      if (!Number.isInteger(c[key]))
        throw new RangeError(`${key} must be an integer`);
    if (!["adaptive", "fixed"].includes(c.delayMode))
      throw new RangeError("delayMode must be adaptive or fixed");
    if (!["independent", "shared", "serial"].includes(c.computeMode))
      throw new RangeError("computeMode must be independent, shared or serial");
    if (c.fixedDelay > c.maxDelay)
      throw new RangeError("fixedDelay cannot exceed maxDelay");
    if (
      Math.ceil((c.seconds * 1000) / c.periodMs) +
        Math.ceil(c.seconds * c.cameraHz) >
      12000
    )
      throw new RangeError(
        "Too many camera/control arrivals; reduce seconds or rates",
      );
    return c;
  }

  function simulatePolicyQueues(settings = {}) {
    const config = queueConfig(settings),
      c = config,
      eps = 1e-7;
    const windowEndMs = c.seconds * 1000;
    const sample = (base, rng) => base * (1 + c.jitter * (2 * rng() - 1));
    const actionRng = random(c.seed ^ 0x13a4),
      visualRng = random(c.seed ^ 0x5b71);
    const calibrationRng = random(c.seed ^ 0x7c29);
    // Approximate the wrapper's 12 warm requests, discard first 2, P95 + margin.
    // Samples represent the FULL request cost, not a DiT-only kernel duration.
    const warmSamples = Array.from({ length: 12 }, () =>
      sample(c.actionMs, calibrationRng),
    );
    const calibrationP95Ms = percentile(warmSamples.slice(2), 0.95);
    const budget = (duration) =>
      Math.max(1, Math.ceil((duration + c.marginMs) / c.periodMs - 1e-12));
    const requiredDelay = budget(calibrationP95Ms);
    const initialDelay =
      c.delayMode === "adaptive"
        ? Math.min(c.maxDelay, requiredDelay)
        : c.fixedDelay;
    let delay = initialDelay,
      committedUntil = 0,
      nextRequestTick = 0,
      bufferOrigin = 0;
    let time = 0,
      cameraIndex = 0,
      tickIndex = 0,
      iterations = 0;
    let runningVision = null,
      pendingFrame = null,
      outstandingAction = null;
    let workA = null,
      workV = null,
      windowEnded = false;
    const resourceSegments = [];
    const bootstrapFeature = {
      id: -1,
      label: "V-1",
      captureMs: 0,
      finishMs: 0,
      usedByRequestIds: [],
      status: "bootstrap",
      unused: false,
      unusedAtMs: null,
    };
    let latestFeature = bootstrapFeature;
    const visualJobs = [],
      actionRequests = [],
      executions = [],
      frameDrops = [],
      events = [];
    const frames = [],
      featureReads = new Map(),
      commands = new Map(),
      bootstrap = [];
    const record = (kind, data = {}) =>
      events.push({ timeMs: time, kind, ...data });
    const originCopy = (x) => ({ ...x });
    const newSlot = (kind, targetTick, extra = {}) => ({
      kind,
      targetTick,
      ...extra,
    });
    for (let k = 0; k < c.bootstrapSlots; k++) {
      const origin = newSlot("bootstrap", k, { label: `B${k}` });
      bootstrap.push({
        ...origin,
        status: "available",
        executedAtMs: null,
        unused: false,
        unusedAtMs: null,
      });
      commands.set(k, origin);
    }
    const markDisplaced = (origin, reason) => {
      if (origin.kind === "bootstrap") {
        Object.assign(bootstrap[origin.targetTick], {
          status: reason,
          unused: true,
          unusedAtMs: time,
        });
      } else if (origin.kind === "request") {
        const output = actionRequests[origin.requestId].outputs[origin.offset];
        Object.assign(output, {
          status: reason,
          unused: true,
          unusedAtMs: time,
        });
      }
    };
    const reserve = (tick, length) => {
      const slots = [];
      for (let k = tick; k < tick + length; k++) {
        if (!commands.has(k))
          commands.set(
            k,
            newSlot("fallback", k, { label: `H${k}`, originTick: tick }),
          );
        slots.push(originCopy(commands.get(k)));
      }
      committedUntil = Math.max(committedUntil, tick + length);
      record("reserve", { startTick: tick, length, committedUntil, slots });
      return slots;
    };
    const rollingState = (request, phase) => {
      const computing = phase === "computing",
        origin = request.requestTick + (computing ? 0 : request.d);
      return {
        horizon: 40,
        origin,
        slideSteps: request.d,
        phase,
        // These phases describe slot ownership/cleanliness, not neural tensors.
        cells: Array.from({ length: 40 }, (_, offset) => ({
          offset,
          targetTick: origin + offset,
          kind:
            offset < request.d
              ? computing
                ? "committed"
                : "clean"
              : computing && offset < 2 * request.d
                ? "becoming-clean"
                : offset >= 40 - request.d
                  ? "noise"
                  : "partially-denoised",
        })),
      };
    };
    const initWork = (job, role, soloMs) => {
      const share = role === "action" ? c.gpuShareA : c.gpuShareV;
      Object.assign(job, {
        soloMs,
        hostMs: soloMs * (1 - share),
        gpuWorkMs: soloMs * share,
        gpuStartMs: null,
        gpuFinishMs: null,
        finishMs: null,
        responseMs: null,
        overlapMs: 0,
        contentionDelayMs: 0,
        gpuWaitMs: 0,
        segments: [],
      });
      return {
        job,
        role,
        remaining: job.gpuWorkMs,
        phase: job.hostMs > 0 ? "host" : "gpu-wait",
        hostUntil: time + job.hostMs,
      };
    };
    const startVision = () => {
      if (runningVision || !pendingFrame) return;
      const frame = pendingFrame;
      pendingFrame = null;
      const job = {
        id: visualJobs.length,
        label: `V${visualJobs.length}`,
        sourceFrameId: frame.id,
        sourceTick: Math.floor((frame.captureMs + eps) / c.periodMs),
        sourceLabel: frame.label,
        captureMs: frame.captureMs,
        startMs: time,
        usedByRequestIds: [],
        status: "running",
        unused: false,
        unusedAtMs: null,
      };
      frame.startedAtMs = time;
      visualJobs.push(job);
      runningVision = job;
      workV = initWork(job, "vision", sample(c.vlmMs, visualRng));
      record("vision-start", { jobId: job.id, frameId: frame.id });
    };
    const publish = (request, tick) => {
      const publication = {
        atMs: time,
        consumedAtTick: tick,
        installed: 0,
        expired: 0,
        protected: 0,
      };
      const slots = [];
      for (const output of request.outputs) {
        const status =
          output.targetTick < tick
            ? "expired"
            : output.targetTick < committedUntil
              ? "protected"
              : "installed";
        publication[status]++;
        output.status = status;
        output.publicationAtMs = time;
        if (status === "installed") {
          if (commands.has(output.targetTick))
            markDisplaced(commands.get(output.targetTick), "superseded");
          const origin = newSlot("request", output.targetTick, {
            requestId: request.id,
            requestLabel: request.label,
            offset: output.offset,
            label: output.label,
            originTick: request.requestTick,
          });
          commands.set(output.targetTick, origin);
          output.adoptedAtMs = time;
          slots.push({ status, origin });
        } else {
          output.unused = true;
          output.unusedAtMs = time;
          slots.push({ status, origin: originCopy(output) });
        }
      }
      request.publication = publication;
      record("publish", {
        requestId: request.id,
        publication: { ...publication },
        slots,
      });
      if (
        c.delayMode === "adaptive" &&
        request.finishMs > request.deadlineMs + eps
      ) {
        delay = Math.max(
          delay,
          Math.min(c.maxDelay, budget(request.finishMs - request.startMs)),
        );
        record("delay", {
          delay,
          requiredDelay: budget(request.finishMs - request.startMs),
        });
      }
      record("action-consumed", { requestId: request.id });
    };
    const startAction = (tick) => {
      const prefix = reserve(tick, delay),
        version = latestFeature.id;
      const count = featureReads.get(version) || 0;
      featureReads.set(version, count + 1);
      const suffix =
        count < 26 ? String.fromCharCode(97 + count) : `_${count + 1}`;
      const request = {
        id: actionRequests.length,
        label: `I${version}${suffix}`,
        requestTick: tick,
        d: delay,
        startMs: time,
        deadlineMs: (tick + delay) * c.periodMs,
        featureVersion: version,
        featureCaptureMs: latestFeature.captureMs,
        featureReadyMs: latestFeature.finishMs,
        stateCaptureMs: time,
        committedPrefix: prefix,
        extraBufferShift: tick - bufferOrigin,
        bufferRestarted: tick - bufferOrigin >= 40,
        outputs: [],
        publication: null,
        completedInWindow: false,
        completed: false,
      };
      for (let offset = 0; offset < delay; offset++)
        request.outputs.push({
          id: `${request.id}:${offset}`,
          label: `${request.label}[${offset}]`,
          requestId: request.id,
          offset,
          targetTick: tick + delay + offset,
          status: "computing",
          adoptedAtMs: null,
          executedAtMs: null,
          unused: false,
          unusedAtMs: null,
        });
      actionRequests.push(request);
      outstandingAction = request;
      workA = initWork(request, "action", sample(c.actionMs, actionRng));
      latestFeature.usedByRequestIds.push(request.id);
      if (version >= 0) latestFeature.status = "used";
      nextRequestTick = tick + delay;
      record("action-start", {
        requestId: request.id,
        featureVersion: version,
        rolling: rollingState(request, "computing"),
      });
    };
    const finishWork = (role) => {
      const work = role === "action" ? workA : workV;
      const job = work.job;
      job.finishMs = time;
      job.responseMs = time - job.startMs;
      if (job.gpuWorkMs > 0) job.gpuFinishMs = time;
      // GPU slowdown and non-preemptive queue wait are distinct costs.
      job.contentionDelayMs = Math.max(
        0,
        job.responseMs - job.soloMs - job.gpuWaitMs,
      );
      if (job.contentionDelayMs < eps) job.contentionDelayMs = 0;
      if (role === "action") {
        workA = null;
        job.completed = true;
        job.completedInWindow = time < windowEndMs - eps;
        job.deadlineMiss = time > job.deadlineMs + eps;
        job.overBudget = budget(job.responseMs) > c.maxDelay;
        for (const output of job.outputs) output.status = "ready";
        bufferOrigin = job.requestTick + job.d;
        record("action-finish", {
          requestId: job.id,
          rolling: rollingState(job, "post-slide"),
        });
      } else {
        workV = null;
        const old = latestFeature;
        if (
          time < windowEndMs - eps &&
          old.id >= 0 &&
          !old.usedByRequestIds.length
        ) {
          old.status = "superseded-unused";
          old.unused = true;
          old.unusedAtMs = time;
        }
        latestFeature = job;
        latestFeature.status = "available";
        record("vision-finish", { jobId: job.id, replacedFeatureId: old.id });
        runningVision = null;
        // The latest waiting camera frame is not a GPU job until admitted.
        if (time < windowEndMs - eps) startVision();
      }
    };
    const settleWork = () => {
      // Vision completion is visible before same-time action admission.
      for (const role of ["vision", "action"]) {
        const work = role === "action" ? workA : workV;
        if (!work) continue;
        if (work.phase === "gpu" && work.remaining <= eps) finishWork(role);
        else if (work.phase === "host" && work.hostUntil <= time + eps) {
          if (work.job.gpuWorkMs === 0) finishWork(role);
          else {
            work.phase = "gpu-wait";
            record("gpu-ready", { role, jobId: work.job.id });
          }
        }
      }
    };
    const dispatchGpu = () => {
      // Non-preemptive serial arbitration occurs only after ALL simultaneous
      // arrivals and host completions. Ready action work wins a free boundary.
      if (
        c.computeMode === "serial" &&
        [workA, workV].some((w) => w?.phase === "gpu")
      )
        return;
      for (const work of [workA, workV]) {
        if (work?.phase !== "gpu-wait") continue;
        work.phase = "gpu";
        work.job.gpuStartMs = time;
        record("gpu-start", { role: work.role, jobId: work.job.id });
        if (c.computeMode === "serial") break;
      }
    };
    const rates = () => {
      const both = workA?.phase === "gpu" && workV?.phase === "gpu";
      return {
        overlap: Boolean(both),
        rateA:
          workA?.phase === "gpu"
            ? 1 / (1 + (both && c.computeMode === "shared" ? c.slowdownA : 0))
            : 0,
        rateV:
          workV?.phase === "gpu"
            ? 1 / (1 + (both && c.computeMode === "shared" ? c.slowdownV : 0))
            : 0,
      };
    };
    const nextWorkEvent = (work, rate) =>
      !work
        ? Infinity
        : work.phase === "host"
          ? work.hostUntil
          : work.phase === "gpu"
            ? time + work.remaining / rate
            : Infinity;
    const advanceWork = (next, current) => {
      const duration = next - time;
      const segment = {
        startMs: time,
        endMs: next,
        actionId: workA?.job.id ?? null,
        visionId: workV?.job.id ?? null,
        actionPhase: workA?.phase ?? null,
        visionPhase: workV?.phase ?? null,
        rateA: current.rateA,
        rateV: current.rateV,
        overlap: current.overlap,
        afterWindow: time >= windowEndMs - eps,
      };
      const previous = resourceSegments.at(-1);
      const same =
        previous &&
        [
          "actionId",
          "visionId",
          "actionPhase",
          "visionPhase",
          "rateA",
          "rateV",
          "overlap",
          "afterWindow",
        ].every((key) => previous[key] === segment[key]);
      if (same && Math.abs(previous.endMs - time) < eps) previous.endMs = next;
      else resourceSegments.push(segment);
      for (const [work, rate] of [
        [workA, current.rateA],
        [workV, current.rateV],
      ]) {
        if (!work) continue;
        const gpu = work.phase === "gpu";
        const workMs = gpu ? duration * rate : 0;
        const overlap = gpu && current.overlap;
        if (gpu) {
          work.remaining = Math.max(0, work.remaining - workMs);
          if (overlap) work.job.overlapMs += duration;
        } else if (work.phase === "gpu-wait") work.job.gpuWaitMs += duration;
        const item = {
          startMs: time,
          endMs: next,
          phase: work.phase,
          rate: gpu ? rate : 0,
          workMs,
          overlap: Boolean(overlap),
        };
        const last = work.job.segments.at(-1);
        if (
          last &&
          last.phase === item.phase &&
          last.rate === item.rate &&
          last.overlap === item.overlap &&
          Math.abs(last.endMs - time) < eps
        ) {
          last.endMs = next;
          last.workMs += workMs;
        } else work.job.segments.push(item);
      }
      time = next;
    };
    while (true) {
      if (++iterations > 100000)
        throw new RangeError("Queue event limit exceeded");
      settleWork();
      if (!windowEnded && time >= windowEndMs - eps) {
        windowEnded = true;
        if (pendingFrame)
          frameDrops.push({
            ...pendingFrame,
            atMs: time,
            reason: "window-ended",
            unused: false,
            unusedAtMs: null,
          });
        pendingFrame = null;
        record("window-end", {});
      }
      if (!windowEnded) {
        const cameraAt =
          c.cameraHz > 0 ? (cameraIndex * 1000) / c.cameraHz : Infinity;
        if (cameraAt <= time + eps) {
          const frame = {
            id: cameraIndex,
            label: `O${cameraIndex}`,
            captureMs: time,
            startedAtMs: null,
          };
          frames.push(frame);
          cameraIndex++;
          if (pendingFrame) {
            const dropped = {
              ...pendingFrame,
              atMs: time,
              replacementId: frame.id,
              reason: "replaced-pending",
              unused: true,
              unusedAtMs: time,
            };
            frameDrops.push(dropped);
            record("frame-drop", {
              frameId: pendingFrame.id,
              replacementId: frame.id,
            });
          }
          pendingFrame = frame;
          record("camera", { frameId: frame.id });
          startVision();
        }
        if (tickIndex * c.periodMs <= time + eps) {
          if (outstandingAction?.completedInWindow) {
            publish(outstandingAction, tickIndex);
            outstandingAction = null;
          }
          if (!outstandingAction && tickIndex >= nextRequestTick)
            startAction(tickIndex);
          if (!commands.has(tickIndex)) reserve(tickIndex, 1);
          committedUntil = Math.max(committedUntil, tickIndex + 1);
          const origin = commands.get(tickIndex);
          commands.delete(tickIndex);
          if (origin.kind === "request") {
            const output =
              actionRequests[origin.requestId].outputs[origin.offset];
            output.status = "executed";
            output.executedAtMs = time;
          } else if (origin.kind === "bootstrap") {
            bootstrap[origin.targetTick].status = "executed";
            bootstrap[origin.targetTick].executedAtMs = time;
          }
          const execution = {
            tick: tickIndex,
            timeMs: time,
            origin: originCopy(origin),
            fallback: origin.kind === "fallback",
            delay,
            committedUntil,
          };
          executions.push(execution);
          record("execute", { execution });
          tickIndex++;
        }
      }
      dispatchGpu();
      if (windowEnded && !workA && !workV) break;
      const current = rates();
      const next = Math.min(
        windowEnded ? Infinity : windowEndMs,
        windowEnded ? Infinity : tickIndex * c.periodMs,
        !windowEnded && c.cameraHz > 0
          ? (cameraIndex * 1000) / c.cameraHz
          : Infinity,
        nextWorkEvent(workA, current.rateA),
        nextWorkEvent(workV, current.rateV),
      );
      if (!Number.isFinite(next) || !(next > time))
        throw new Error("Queue event clock did not advance");
      advanceWork(next, current);
    }
    // The real controller drains its last outstanding result after control ends.
    // Count only slots whose target was already passed; future slots are censored.
    // Evidence cannot appear before that result completes, even after the window.
    let expiredAtWindowEnd = 0;
    if (outstandingAction) {
      const drainedAtMs = Math.max(windowEndMs, outstandingAction.finishMs);
      outstandingAction.drainedAtMs = drainedAtMs;
      for (const output of outstandingAction.outputs) {
        if (output.targetTick < tickIndex) {
          output.status = "expired-at-window-end";
          output.unused = true;
          output.unusedAtMs = drainedAtMs;
          expiredAtWindowEnd++;
        }
      }
    }
    for (const job of visualJobs)
      if (!job.usedByRequestIds.length && !job.unused)
        job.status = "window-ended";
    for (const request of actionRequests) {
      for (const output of request.outputs)
        if (!output.unused && output.executedAtMs === null)
          output.status = "window-ended";
      request.unused = request.outputs.every((output) => output.unused);
      request.unusedAtMs = request.unused
        ? Math.max(...request.outputs.map((output) => output.unusedAtMs))
        : null;
    }
    if (outstandingAction)
      events.push({
        timeMs: outstandingAction.drainedAtMs,
        kind: "action-drained",
        requestId: outstandingAction.id,
        expired: expiredAtWindowEnd,
        futureSlots: outstandingAction.outputs.length - expiredAtWindowEnd,
      });
    events.sort((a, b) => a.timeMs - b.timeMs);
    const publications = actionRequests
      .filter((r) => r.publication)
      .map((r) => r.publication);
    const outputs = actionRequests.flatMap((r) => r.outputs);
    const expiredInWindow = publications.reduce((n, p) => n + p.expired, 0);
    const windowResourceMs = (predicate) =>
      resourceSegments.reduce(
        (total, segment) =>
          total +
          (predicate(segment)
            ? Math.max(
                0,
                Math.min(windowEndMs, segment.endMs) - segment.startMs,
              )
            : 0),
        0,
      );
    const totalJobMs = (jobs, field) =>
      jobs.reduce((total, job) => total + job[field], 0);
    return {
      config,
      windowEndMs,
      drainEndMs: time,
      resourceSegments,
      initialDelay,
      finalDelay: delay,
      calibration: {
        scope: "isolated-full-request",
        samplesMs: warmSamples.slice(2),
        p95Ms: calibrationP95Ms,
        marginMs: c.marginMs,
        requiredDelay,
        overBudget: requiredDelay > c.maxDelay,
        method:
          "isolated seeded 12 full-request warm samples; discard 2; empirical P95 plus margin; online d adapts to contention",
      },
      bootstrapFeature,
      bootstrap,
      frames,
      visualJobs,
      actionRequests,
      executions,
      frameDrops,
      events,
      metrics: {
        controlTicks: executions.length,
        requests: actionRequests.length,
        adoptedOutputs: outputs.filter((o) => o.adoptedAtMs !== null).length,
        executedOutputs: outputs.filter((o) => o.executedAtMs !== null).length,
        // Total includes post-control drain, matching expired_action_slots.
        expiredOutputs: expiredInWindow + expiredAtWindowEnd,
        expiredInWindow,
        expiredAtWindowEnd,
        protectedOutputs: publications.reduce((n, p) => n + p.protected, 0),
        fallbackTicks: executions.filter((x) => x.fallback).length,
        unusedFeatures: visualJobs.filter((v) => v.unused).length,
        replacedFrames: frameDrops.filter((f) => f.unused).length,
        deadlineMisses: actionRequests.filter(
          (r) => r.deadlineMiss && r.deadlineMs <= windowEndMs + eps,
        ).length,
        overBudgetRequests: actionRequests.filter((r) => r.overBudget).length,
        // Resource utilization is measured ONLY within the control window.
        gpuOverlapMs: windowResourceMs((s) => s.overlap),
        gpuBusyMs: windowResourceMs((s) => s.rateA > 0 || s.rateV > 0),
        gpuBusyAMs: windowResourceMs((s) => s.rateA > 0),
        gpuBusyVMs: windowResourceMs((s) => s.rateV > 0),
        // Per-job latency statistics include the complete drain of admitted jobs.
        gpuWaitAMs: totalJobMs(actionRequests, "gpuWaitMs"),
        gpuWaitVMs: totalJobMs(visualJobs, "gpuWaitMs"),
        contentionDelayAMs: totalJobMs(actionRequests, "contentionDelayMs"),
        contentionDelayVMs: totalJobMs(visualJobs, "contentionDelayMs"),
        actionResponseP95Ms: percentile(
          actionRequests.map((r) => r.responseMs),
          0.95,
        ),
        visionResponseP95Ms: percentile(
          visualJobs.map((r) => r.responseMs),
          0.95,
        ),
        actionInflation: mean(
          actionRequests.map((r) => r.responseMs / r.soloMs),
        ),
        visionInflation: mean(visualJobs.map((r) => r.responseMs / r.soloMs)),
      },
    };
  }

  function policyQueueSnapshot(result, timeMs) {
    if (typeof timeMs !== "number" || !Number.isFinite(timeMs))
      throw new RangeError("Snapshot time must be finite");
    const at = Math.max(0, Math.min(result.windowEndMs, timeMs));
    const commands = new Map(
      result.bootstrap.map((b) => [
        b.targetTick,
        { kind: "bootstrap", targetTick: b.targetTick, label: b.label },
      ]),
    );
    let runningVision = null,
      pendingFrame = null,
      latestFeature = result.bootstrapFeature;
    let runningAction = null,
      readyAction = null,
      committedUntil = 0,
      lastExecuted = null;
    let nextTick = 0,
      delay = result.initialDelay;
    let rolling = {
      horizon: 40,
      origin: 0,
      slideSteps: delay,
      phase: "bootstrap",
      cells: Array.from({ length: 40 }, (_, offset) => ({
        offset,
        targetTick: offset,
        kind:
          offset < delay
            ? "clean"
            : offset >= 40 - delay
              ? "noise"
              : "partially-denoised",
      })),
    };
    for (const event of result.events) {
      if (event.timeMs > at + 1e-7) break;
      if (event.kind === "camera") pendingFrame = result.frames[event.frameId];
      else if (event.kind === "vision-start") {
        runningVision = result.visualJobs[event.jobId];
        pendingFrame = null;
      } else if (event.kind === "vision-finish") {
        latestFeature = result.visualJobs[event.jobId];
        runningVision = null;
      } else if (event.kind === "action-start") {
        runningAction = result.actionRequests[event.requestId];
        rolling = event.rolling;
      } else if (event.kind === "action-finish") {
        readyAction = result.actionRequests[event.requestId];
        runningAction = null;
        rolling = event.rolling;
      } else if (event.kind === "action-consumed") readyAction = null;
      else if (event.kind === "action-drained") {
        readyAction = null;
        runningAction = null;
      } else if (event.kind === "delay") delay = event.delay;
      else if (event.kind === "reserve") {
        for (const slot of event.slots)
          commands.set(slot.targetTick, { ...slot });
        committedUntil = event.committedUntil;
      } else if (event.kind === "publish") {
        for (const slot of event.slots)
          if (slot.status === "installed")
            commands.set(slot.origin.targetTick, { ...slot.origin });
      } else if (event.kind === "execute") {
        lastExecuted = event.execution;
        commands.delete(lastExecuted.tick);
        nextTick = lastExecuted.tick + 1;
        committedUntil = lastExecuted.committedUntil;
      } else if (event.kind === "window-end") pendingFrame = null;
    }
    const currentResource =
      result.resourceSegments.find(
        (s) => s.startMs <= at + 1e-7 && s.endMs > at + 1e-7,
      ) || null;
    const visibleService = (job) => {
      if (job.id === -1) return { phase: "complete" };
      const segments = job.segments
        .filter((s) => s.startMs < at)
        .map((s) => {
          const endMs = Math.min(s.endMs, at);
          return {
            ...s,
            endMs,
            workMs: s.phase === "gpu" ? (endMs - s.startMs) * s.rate : 0,
          };
        });
      const current = job.segments.find(
        (s) => s.startMs <= at + 1e-7 && s.endMs > at + 1e-7,
      );
      const sum = (predicate, value) =>
        segments.reduce((total, s) => total + (predicate(s) ? value(s) : 0), 0);
      return {
        soloMs: job.soloMs,
        hostMs: job.hostMs,
        gpuWorkMs: job.gpuWorkMs,
        gpuStartMs:
          job.gpuStartMs !== null && job.gpuStartMs <= at + 1e-7
            ? job.gpuStartMs
            : null,
        gpuFinishMs:
          job.gpuFinishMs !== null && job.gpuFinishMs <= at + 1e-7
            ? job.gpuFinishMs
            : null,
        responseMs: job.finishMs <= at + 1e-7 ? job.responseMs : null,
        elapsedMs: Math.max(0, Math.min(at, job.finishMs) - job.startMs),
        phase:
          job.finishMs <= at + 1e-7
            ? "complete"
            : current?.phase === "gpu"
              ? current.overlap && result.config.computeMode === "shared"
                ? "gpu-shared"
                : "gpu-solo"
              : current?.phase || "gpu-wait",
        gpuWaitMs: sum(
          (s) => s.phase === "gpu-wait",
          (s) => s.endMs - s.startMs,
        ),
        overlapMs: sum(
          (s) => s.overlap,
          (s) => s.endMs - s.startMs,
        ),
        contentionDelayMs: sum(
          (s) => s.phase === "gpu",
          (s) => (s.endMs - s.startMs) * (1 - s.rate),
        ),
        segments,
      };
    };
    // Snapshot records deliberately avoid future rejection/use/accounting knowledge.
    const visibleJob = (job) =>
      job
        ? {
            ...job,
            ...visibleService(job),
            usedByRequestIds: job.usedByRequestIds.filter(
              (id) => result.actionRequests[id].startMs <= at + 1e-7,
            ),
            unused: job.unused && job.unusedAtMs <= at + 1e-7,
            unusedAtMs:
              job.unusedAtMs !== null && job.unusedAtMs <= at + 1e-7
                ? job.unusedAtMs
                : null,
            status:
              job.id === -1
                ? "bootstrap"
                : job.finishMs > at + 1e-7
                  ? "running"
                  : "available",
          }
        : null;
    const visibleRequest = (request) =>
      request
        ? {
            id: request.id,
            label: request.label,
            requestTick: request.requestTick,
            d: request.d,
            startMs: request.startMs,
            finishMs: request.finishMs,
            deadlineMs: request.deadlineMs,
            featureVersion: request.featureVersion,
            featureCaptureMs: request.featureCaptureMs,
            ...visibleService(request),
            outputs: request.outputs.map((o) => ({
              id: o.id,
              label: o.label,
              targetTick: o.targetTick,
              offset: o.offset,
              status: request.finishMs <= at + 1e-7 ? "ready" : "computing",
            })),
          }
        : null;
    return {
      timeMs: at,
      delay,
      ended: at >= result.windowEndMs,
      resource: {
        computeMode: result.config.computeMode,
        segment: currentResource ? { ...currentResource } : null,
        rateA: currentResource?.rateA || 0,
        rateV: currentResource?.rateV || 0,
        overlap: currentResource?.overlap || false,
      },
      vision: {
        running: visibleJob(runningVision),
        pending: pendingFrame ? { ...pendingFrame, startedAtMs: null } : null,
        latestFeature: visibleJob(latestFeature),
      },
      action: {
        running: visibleRequest(runningAction),
        ready: visibleRequest(readyAction),
      },
      execution: {
        nextTick,
        committedUntil,
        lastExecuted: lastExecuted
          ? { ...lastExecuted, origin: { ...lastExecuted.origin } }
          : null,
        slots: [...commands.values()]
          .sort((a, b) => a.targetTick - b.targetTick)
          .map((slot) => ({
            ...slot,
            committed: slot.targetTick < committedUntil,
          })),
      },
      rolling: {
        ...rolling,
        cells: rolling.cells.map((cell) => ({ ...cell })),
      },
    };
  }

  return {
    percentile,
    queueDefaults,
    simulatePolicyQueues,
    policyQueueSnapshot,
  };
});
