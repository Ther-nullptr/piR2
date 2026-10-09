/* Queue simulation and rendering run locally; file:// is supported. */
"use strict";
const M = window.VLAModel,
  S = window.PiR2Scenarios,
  $ = (id) => document.getElementById(id);
const fmt = (value, digits = 1) =>
  value === null || !Number.isFinite(value)
    ? "—"
    : Number(value).toFixed(digits);
const svg = (width, height, body) =>
  `<svg viewBox="0 0 ${width} ${height}" role="img" xmlns="http://www.w3.org/2000/svg">${body}</svg>`;

// This view consumes the scheduling model's events and snapshots. It
// does not maintain a second controller, queue, or deadline implementation.
let queueConfig = { ...M.queueDefaults, executionPolicy: "paper" },
  queueResult,
  queueBaseline,
  queueProtocolReference,
  scenarioSettings = { ...S.defaults },
  scenarioProjection,
  scenarioReference,
  queueAoi,
  selectedDependency = null,
  calibrationSource = null,
  queueTime = 260,
  queueRunTimer,
  queueAnimation = null,
  queueLastFrame = null,
  queueFreshnessContext = { latestAdoptedId: null, latestCompletedId: null };
const queueColors = {
  vision: ["var(--s2-soft)", "var(--s2)"],
  inference: ["var(--s1-soft)", "var(--s1)"],
  action: ["#fff0d9", "#b77725"],
  wait: ["#fff8e8", "#b69a62"],
  bootstrap: ["#f3edfa", "#6b458f"],
  fallback: ["#fee1e4", "var(--red)"],
  repeat: ["#eee4f6", "#8055a0"],
  newsegment: ["var(--s1-soft)", "var(--s1)"],
};
const queueEscape = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (character) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        character
      ],
  );
const queueCross = '<span class="unused-mark" aria-label="未采用">×</span>';
const queueModeNames = {
  independent: "双卡 / 独立 GPU",
  shared: "单卡并发争用",
  serial: "单卡串行对照",
};
function queueExecutionName(execution) {
  if (execution?.repeated) return "重复末动作";
  return (
    {
      "new-segment": "采用新段",
      "continue-segment": "旧段续播",
      "repeat-last": "重复末动作",
      bootstrap: "启动动作",
      fallback: "无来源保持",
    }[execution?.executionMode] ||
    (execution?.fallback
      ? "无来源保持"
      : execution?.origin?.kind === "bootstrap"
        ? "启动动作"
        : "执行动作")
  );
}
function queueJobTiming(job) {
  return `独占 ${fmt(job.soloMs)} ms；实际 ${fmt(job.responseMs)} ms；共享慢化 +${fmt(job.contentionDelayMs)} ms；等 GPU ${fmt(job.gpuWaitMs)} ms`;
}
function queuePhase(job, resource, side) {
  if (!job) return "空闲";
  if (job.phase === "host-wait") return "等待共享主机服务";
  if (job.phase === "dispatch") return "首次派发额外开销";
  if (job.phase === "host") return "非 GPU 阶段";
  if (job.phase === "gpu-wait") return "等待 GPU（尚未执行）";
  const rate = resource?.segment?.[side === "a" ? "rateA" : "rateV"];
  return ["gpu", "gpu-shared", "gpu-solo"].includes(job.phase) &&
    Number.isFinite(rate)
    ? `${job.phase === "gpu-shared" ? "GPU 共享" : "GPU 执行"} · ${fmt(rate, 2)}×独占速度`
    : "请求进行中";
}
function queueUnused(item) {
  return (
    item.unused && item.unusedAtMs !== null && item.unusedAtMs <= queueTime
  );
}
function queueFreshness(item, options = {}) {
  const request =
    queueResult.actionRequests[options.requestId ?? item.requestId];
  if (!request) {
    const bootstrap = item.kind === "bootstrap";
    return {
      band: bootstrap ? "bootstrap" : "unknown",
      ageMs: null,
      ageLabel: bootstrap ? `状态龄 ≥${fmt(queueTime, 0)} ms` : "状态龄未知",
      status: item.repeated
        ? "重复末动作"
        : item.executedAtMs !== null &&
            item.executedAtMs !== undefined &&
            item.executedAtMs <= queueTime
          ? "已执行"
          : options.committed
            ? "已承诺"
            : bootstrap
              ? "初始化"
              : "fallback",
      latest: false,
      latestCompleted: false,
      unused: queueUnused(item),
      requestId: null,
      tooltip: bootstrap
        ? `${item.label}：图前初始化，采样时刻未知；状态龄至少 ${fmt(queueTime)} ms。`
        : `${item.label}：保持/备用动作，状态采样年龄未知。`,
      color: ["#eef1f3", "#71818b"],
    };
  }
  const output =
    request.outputs.find((candidate) => candidate.offset === item.offset) ||
    item;
  const started = request.startMs <= queueTime,
    ready = request.finishMs <= queueTime;
  const stateTime =
    request.stateCaptureMs ?? request.requestTick * queueConfig.periodMs;
  const ageMs = started ? queueTime - stateTime : null;
  const unused = queueUnused(output);
  const adopted =
    output.adoptedAtMs !== null && output.adoptedAtMs <= queueTime;
  const executed =
    output.executedAtMs !== null && output.executedAtMs <= queueTime;
  const committed =
    options.committed ??
    (adopted &&
      !executed &&
      output.targetTick < queueFreshnessContext.committedUntil);
  const status = item.repeated
    ? "重复末动作"
    : unused
      ? output.status.includes("expired")
        ? "已过期"
        : "未采用"
      : !ready
        ? "待生成"
        : executed
          ? "已执行"
          : committed
            ? "已承诺"
            : adopted
              ? "已发布"
              : "已完成待领取";
  const band = !ready
    ? "pending"
    : ageMs <= queueConfig.periodMs
      ? "recent"
      : ageMs <= 2 * queueConfig.periodMs
        ? "middle"
        : "older";
  const colors = {
    pending: ["#f5f7f8", "#8798a3"],
    recent: ["#fff4d7", "#b68024"],
    middle: ["#ffe0a2", "#a06b1a"],
    older: ["#efc199", "#87552b"],
  };
  const latest =
    ready && !unused && request.id === queueFreshnessContext.latestAdoptedId;
  const latestCompleted =
    ready &&
    !unused &&
    !adopted &&
    request.id === queueFreshnessContext.latestCompletedId;
  const visualAge =
    request.featureVersion < 0
      ? `≥${fmt(queueTime)} ms（F−1 图前初始化）`
      : `${fmt(queueTime - request.featureCaptureMs)} ms（F${request.featureVersion}）`;
  const completionAge = ready
    ? `${fmt(queueTime - request.finishMs)} ms`
    : "尚未完成";
  const waitMs = ready
    ? Math.max(
        0,
        (executed
          ? output.executedAtMs
          : unused
            ? output.unusedAtMs
            : queueTime) - request.finishMs,
      )
    : null;
  return {
    band,
    ageMs: ready ? ageMs : null,
    ageLabel: ready ? `状态龄 ${fmt(ageMs, 0)} ms` : "待生成",
    status,
    latest,
    latestCompleted,
    unused,
    requestId: request.id,
    visionAgeMs:
      started && request.featureVersion >= 0
        ? queueTime - request.featureCaptureMs
        : null,
    completionWaitMs: waitMs,
    color: colors[band],
    tooltip: `${output.label}；${status}；${started ? `状态采样 ${fmt(stateTime)} ms，当前状态龄 ${fmt(ageMs)} ms` : "请求尚未开始，未生成动作"}；视觉龄 ${started ? visualAge : "尚未读取"}；完成至今 ${completionAge}；完成后等待 ${ready ? `${fmt(waitMs)} ms${executed ? "（已执行）" : unused ? "（已停止等待）" : "（仍待执行）"}` : "尚未开始"}；${(queueConfig.executionPolicy === "paper" && !adopted) || output.targetTick === null ? "执行槽待换段时确定" : `目标 t${output.targetTick}`}。${latest ? "当前最新已采用批次。" : latestCompleted ? "当前最新已完成结果。" : ""}年龄不决定动作是否有效。`,
  };
}
function queueFreshnessAttributes(metadata) {
  return `data-request-id="${metadata.requestId ?? ""}" data-age-ms="${metadata.ageMs ?? ""}" data-vision-age-ms="${metadata.visionAgeMs ?? ""}" data-completion-wait-ms="${metadata.completionWaitMs ?? ""}" data-output-status="${queueEscape(metadata.status)}" data-latest-batch="${metadata.latest}" data-latest-completed="${metadata.latestCompleted}"`;
}
function queueActionSlot(item, options = {}) {
  const metadata = queueFreshness(item, options);
  const kind =
    item.kind === "bootstrap"
      ? "bootstrap"
      : item.kind === "fallback"
        ? "fallback"
        : "action";
  return `<div class="queue-slot queue-freshness-slot slot-${kind} freshness-${metadata.band}${metadata.latest ? " latest-batch" : ""}${options.committed ? " committed" : ""}" ${queueFreshnessAttributes(metadata)} title="${queueEscape(metadata.tooltip)}"><b>${queueEscape(item.label)}${metadata.unused ? queueCross : ""}</b><strong class="slot-state-age">${metadata.ageLabel}</strong><span class="slot-status">${metadata.status}</span>${metadata.latest ? '<em class="latest-batch-badge">最新批次</em>' : metadata.latestCompleted ? '<em class="latest-completed-badge">最新完成</em>' : ""}<small>${item.targetTick === null || item.targetTick === undefined ? "换段后确定执行 tick" : `目标 t${item.targetTick} · ${fmt(item.targetTick * queueConfig.periodMs, 0)} ms`}</small></div>`;
}
function queueSlot(label, caption, kind = "inference", options = {}) {
  return `<div class="queue-slot slot-${queueEscape(kind)}${options.committed ? " committed" : ""}"><b>${queueEscape(label)}${options.unused ? queueCross : ""}</b><small>${queueEscape(caption)}</small>${options.committed ? "<em>已承诺</em>" : ""}</div>`;
}
function queueEmpty(message = "空等待位") {
  return `<div class="queue-slot slot-empty"><b>空</b><small>${queueEscape(message)}</small></div>`;
}
function setQueueControls(preserveActive = false) {
  for (const input of document.querySelectorAll("[data-scenario-key]")) {
    if (!preserveActive || input !== document.activeElement)
      input.value = scenarioSettings[input.dataset.scenarioKey];
  }
  for (const input of document.querySelectorAll("[data-queue-key]")) {
    const value = queueConfig[input.dataset.queueKey];
    if (
      input.type === "range" &&
      ["vlmMs", "actionMs"].includes(input.dataset.queueKey)
    )
      input.max = Math.max(500, value);
    if (
      !preserveActive ||
      input !== document.activeElement ||
      input.type !== "number"
    )
      input.value = value;
  }
  $("queue-paper-controls").hidden = queueConfig.executionPolicy !== "paper";
  $("queue-marginMs").disabled = queueConfig.executionPolicy === "paper";
  $("queue-paperInitialDelay").max = queueConfig.maxDelay;
  $("queue-fixed-label").hidden = queueConfig.delayMode !== "fixed";
  $("queue-contention-controls").hidden = queueConfig.computeMode !== "shared";
  $("queue-resource-note").textContent =
    queueConfig.computeMode === "shared"
      ? "只有 GPU 阶段重叠时降速；非 GPU 阶段保持原耗时。"
      : queueConfig.computeMode === "serial"
        ? "非抢占：正在运行的 GPU 阶段不被打断，空闲边界优先动作请求。"
        : "两个 worker 使用独立 GPU；主机服务是否共享由上方选项决定。";
  $("queue-fixedDelay").max = queueConfig.maxDelay;
  $("queue-jitter-value").textContent = `±${fmt(queueConfig.jitter * 100, 0)}%`;
}
function readQueueControls(preserveActive = false) {
  for (const input of document.querySelectorAll("[data-scenario-key]")) {
    const key = input.dataset.scenarioKey;
    scenarioSettings[key] =
      typeof S.defaults[key] === "number"
        ? input.value.trim() === ""
          ? S.defaults[key]
          : Number(input.value)
        : input.value;
  }
  for (const key of Object.keys(M.queueDefaults)) {
    const input = $(`queue-${key}`);
    if (!input) continue;
    if (typeof M.queueDefaults[key] === "number") {
      let value =
        input.value.trim() === "" ? M.queueDefaults[key] : Number(input.value);
      if (!Number.isFinite(value)) value = M.queueDefaults[key];
      value = Math.min(Number(input.max), Math.max(Number(input.min), value));
      if (
        [
          "seed",
          "maxDelay",
          "fixedDelay",
          "bootstrapSlots",
          "delayWindow",
          "paperInitialDelay",
        ].includes(key)
      )
        value = Math.round(value);
      queueConfig[key] = value;
    } else queueConfig[key] = input.value;
  }
  queueConfig.fixedDelay = Math.min(
    queueConfig.fixedDelay,
    queueConfig.maxDelay,
  );
  queueConfig.paperInitialDelay = Math.min(
    queueConfig.paperInitialDelay,
    queueConfig.maxDelay,
  );
  setQueueControls(preserveActive);
}
function queueText(
  x,
  y,
  value,
  size = 11,
  color = "#64757c",
  anchor = "middle",
) {
  return `<text x="${x}" y="${y}" fill="${color}" font-size="${size}" text-anchor="${anchor}">${queueEscape(value)}</text>`;
}
function renderQueueTimeline() {
  if (!queueResult) return;
  const width = 1200,
    height = 637,
    left = 120,
    right = 1170,
    period = queueConfig.periodMs,
    span = Math.min(
      queueResult.windowEndMs,
      Number($("lab-timeline-span").value) || queueResult.windowEndMs,
    );
  const start = Math.max(
      0,
      Math.min(
        queueResult.windowEndMs - span,
        Math.floor((queueTime - span * 0.65) / period) * period,
      ),
    ),
    end = start + span,
    x = (time) => left + ((time - start) / span) * (right - left);
  const aliases = S.dependencyAliases(queueResult, start, end);
  $("queue-window-label").textContent = `${fmt(start, 0)}–${fmt(end, 0)} ms`;
  let body =
    '<defs><clipPath id="queue-time-clip"><rect x="120" y="25" width="1050" height="576"/></clipPath><pattern id="queue-sharing-pattern" width="6" height="6" patternUnits="userSpaceOnUse"><path d="M-1,1 L1,-1 M0,6 L6,0 M5,7 L7,5" stroke="#9364a0" stroke-width="1.5"/></pattern><pattern id="queue-gpu-wait-pattern" width="6" height="6" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.1" fill="#b58c4a"/></pattern><pattern id="queue-bootstrap-pattern" width="7" height="7" patternUnits="userSpaceOnUse"><rect width="7" height="7" fill="#f3edfa"/><path d="M0,7L7,0" stroke="#e2d3f0" stroke-width="2"/></pattern></defs>';
  const lanes = [
    ["VLM · S2", 48],
    ["图像等待位", 110],
    ["特征缓存", 172],
    ["DiT · S1", 234],
    ["完成待领取", 296],
    ["输出目标槽", 358],
    ["控制器执行", 470],
    ["GPU 活跃状态", 532],
  ];
  for (const [name, y] of lanes)
    body +=
      `<rect x="112" y="${y - 5}" width="1066" height="${name === "输出目标槽" ? 91 : 47}" rx="5" fill="${name === "输出目标槽" ? "#fff9ed" : "#f7f9f8"}"/>` +
      queueText(103, y + 21, name, 12, "#435b66", "end");
  const tickStride = Math.max(1, Math.ceil(span / period / 14));
  for (
    let tick = Math.ceil(start / period);
    tick * period <= end;
    tick += tickStride
  ) {
    const xx = x(tick * period);
    body += `<line x1="${xx}" x2="${xx}" y1="30" y2="590" stroke="#dce5e5" stroke-dasharray="3 5"/>`;
    body += queueText(xx, 19, fmt(tick * period, 0));
    body += queueText(xx, 618, `t${tick}`, 10);
  }
  body += queueText(1193, 19, "ms", 10);
  const block = (
    from,
    until,
    y,
    label,
    kind,
    tooltip,
    unused = false,
    readyAt = from,
    attributes = "",
  ) => {
    if (until <= start || from >= end || until <= from) return;
    const lo = Math.max(from, start),
      hi = Math.min(until, end),
      boxWidth = Math.max(0.8, x(hi) - x(lo) - 2),
      color = queueColors[kind],
      opacity = readyAt > queueTime ? 0.23 : 1;
    body += `<g opacity="${opacity}" ${attributes}><title>${queueEscape(tooltip)}</title><rect x="${x(lo) + 1}" y="${y}" width="${boxWidth}" height="31" rx="3" fill="${color[0]}" stroke="${color[1]}"/>`;
    if (boxWidth > label.length * 5.5 + (unused ? 12 : 0))
      body += queueText(x(lo) + boxWidth / 2 + 1, y + 20, label, 10, color[1]);
    if (unused)
      body += `<text class="unused-mark" x="${x(hi) - 6}" y="${y + 10}" fill="#ae524b" font-size="10" text-anchor="end">×</text>`;
    body += "</g>";
  };
  const resourceOverlay = (job, y) => {
    for (const segment of job.segments || []) {
      const lo = Math.max(start, segment.startMs),
        hi = Math.min(end, segment.endMs);
      if (hi <= lo) continue;
      const sharing = queueConfig.computeMode === "shared" && segment.overlap;
      const fill = sharing
        ? "url(#queue-sharing-pattern)"
        : segment.phase === "gpu-wait"
          ? "url(#queue-gpu-wait-pattern)"
          : ["host", "dispatch", "host-wait"].includes(segment.phase)
            ? segment.phase === "host-wait"
              ? "#dd9787"
              : "#b7c2c9"
            : null;
      if (fill)
        body += `<rect class="queue-phase-${sharing ? "shared" : segment.phase}" x="${x(lo)}" y="${y + 24}" width="${x(hi) - x(lo)}" height="7" fill="${fill}" opacity="${segment.startMs > queueTime ? 0.23 : 1}"><title>${queueEscape(`${job.label}：${sharing ? "GPU 共享慢化" : segment.phase === "gpu-wait" ? "等待 GPU" : "非 GPU 阶段"}；${fmt(lo)}–${fmt(hi)} ms`)}</title></rect>`;
    }
  };
  const outputBlock = (item) => {
    if (
      queueConfig.executionPolicy === "paper" &&
      item.requestId !== undefined &&
      (item.adoptedAtMs === null || item.adoptedAtMs > queueTime)
    )
      return;
    if (!Number.isInteger(item.targetTick)) return;
    const from = item.targetTick * period,
      until = from + period;
    if (until <= start || from >= end) return;
    const lo = Math.max(start, from),
      hi = Math.min(end, until),
      boxWidth = Math.max(1, x(hi) - x(lo) - 4),
      center = x(lo) + boxWidth / 2 + 2,
      metadata = queueFreshness(item),
      fill =
        metadata.band === "bootstrap"
          ? "url(#queue-bootstrap-pattern)"
          : metadata.color[0];
    body += `<g class="queue-output-slot freshness-${metadata.band}${metadata.latest ? " latest-batch" : ""}" ${queueFreshnessAttributes(metadata)}><title>${queueEscape(metadata.tooltip)}</title><rect x="${x(lo) + 2}" y="360" width="${boxWidth}" height="78" rx="4" fill="${fill}" stroke="${metadata.latest ? "#285a93" : metadata.color[1]}" stroke-width="${metadata.latest ? 2.8 : 1}"${metadata.band === "pending" ? ' stroke-dasharray="4 3"' : ""}/>`;
    if (boxWidth > 40) {
      body += queueText(center, 377, item.label, 10, metadata.color[1]);
      body += queueText(center, 398, metadata.ageLabel, 10, "#674b30");
      if (metadata.status !== metadata.ageLabel)
        body += queueText(center, 414, metadata.status, 9, metadata.color[1]);
      if (metadata.latest || metadata.latestCompleted)
        body += queueText(
          center,
          431,
          metadata.latest ? "最新批次" : "最新完成",
          9,
          "#285a93",
        );
    }
    if (metadata.unused)
      body += `<text class="unused-mark" x="${x(hi) - 7}" y="372" fill="#ae524b" font-size="10" text-anchor="end">×</text>`;
    body += "</g>";
  };
  body += '<g clip-path="url(#queue-time-clip)">';
  for (const job of queueResult.visualJobs) {
    block(
      job.startMs,
      job.finishMs,
      50,
      aliases.vision(job.id),
      "vision",
      `${aliases.vision(job.id)}（${job.label}）使用 ${job.sourceLabel}；${fmt(job.startMs)}–${fmt(job.finishMs)} ms；${queueJobTiming(job)}`,
      queueUnused(job),
      job.startMs,
      `data-vision-id="${job.id}" class="${selectedDependency?.featureVersion === job.id ? "lab-selected-job" : ""}"`,
    );
    resourceOverlay(job, 50);
  }
  const frameDrops = new Map(
    queueResult.frameDrops.map((frame) => [frame.id, frame]),
  );
  for (const frame of queueResult.frames) {
    const drop = frameDrops.get(frame.id),
      finish = frame.startedAtMs ?? drop?.atMs ?? queueResult.windowEndMs;
    block(
      frame.captureMs,
      finish,
      112,
      frame.label,
      "vision",
      `${frame.label} 等待 ${fmt(frame.captureMs)}–${fmt(finish)} ms${drop?.unused ? "；后来被新图像覆盖" : ""}`,
      drop ? queueUnused(drop) : false,
    );
  }
  const updates = [
    queueResult.bootstrapFeature,
    ...queueResult.visualJobs.filter(
      (job) => job.finishMs < queueResult.windowEndMs,
    ),
  ];
  updates.forEach((feature, index) => {
    const finish = updates[index + 1]?.finishMs ?? queueResult.windowEndMs;
    block(
      feature.finishMs,
      finish,
      174,
      aliases.vision(feature.id),
      "vision",
      `F${feature.id} 缓存；采集 ${fmt(feature.captureMs)} ms，${fmt(feature.finishMs)} ms 就绪`,
    );
  });
  for (const request of queueResult.actionRequests) {
    block(
      request.startMs,
      request.finishMs,
      236,
      aliases.action(request).label,
      "inference",
      `${aliases.action(request).label}（${request.label}）读取 ${aliases.vision(request.featureVersion)} / F${request.featureVersion} + s${request.requestTick}；d=${request.d}；${fmt(request.startMs)}–${fmt(request.finishMs)} ms；${queueJobTiming(request)}`,
      queueUnused(request),
      request.startMs,
      `data-action-id="${request.id}" data-parent-feature="${request.featureVersion}" class="${selectedDependency?.id === request.id ? "lab-selected-job" : ""}"`,
    );
    resourceOverlay(request, 236);
    block(
      request.finishMs,
      request.publication?.atMs ??
        request.discardedAtMs ??
        queueResult.windowEndMs,
      298,
      `${aliases.action(request).label} W`,
      "wait",
      `${request.label} 已完成；等待控制循环领取`,
      false,
    );
    for (const output of request.outputs) outputBlock(output);
  }
  const plannedTargets = new Set(
    queueResult.actionRequests.flatMap((request) =>
      request.outputs.map((output) => output.targetTick),
    ),
  );
  for (const bootstrap of queueResult.bootstrap)
    if (!plannedTargets.has(bootstrap.targetTick)) outputBlock(bootstrap);
  for (const execution of queueResult.executions) {
    const origin = execution.origin,
      producer =
        origin.kind === "request"
          ? queueResult.actionRequests[origin.requestId]
          : null,
      sourceLabel = producer
        ? `${aliases.action(producer).label}[${origin.offset}]`
        : origin.label,
      repeated =
        execution.repeated || execution.executionMode === "repeat-last",
      kind = repeated
        ? "repeat"
        : execution.executionMode === "new-segment"
          ? "newsegment"
          : origin.kind === "request"
            ? "action"
            : origin.kind;
    block(
      execution.timeMs,
      execution.timeMs + period,
      472,
      `${repeated ? "↻ " : ""}${sourceLabel}`,
      kind,
      `t${execution.tick} ${queueExecutionName(execution)} ${sourceLabel}；d=${execution.delay}${producer ? `；源状态 ${fmt(producer.stateCaptureMs)} ms；图像来源 ${producer.featureVersion < 0 ? "图前未知" : `${fmt(producer.featureCaptureMs)} ms`}` : "；启动或保持动作来源未知"}${repeated ? "；保留生产请求，不刷新 AoI" : ""}`,
      false,
      execution.timeMs,
      `data-execution-tick="${execution.tick}" data-execution-mode="${execution.executionMode || (execution.fallback ? "fallback" : "normal")}" data-origin-request="${producer?.id ?? ""}" data-origin-offset="${origin.offset ?? ""}" class="${producer && selectedDependency?.id === producer.id ? "lab-selected-job" : ""}"`,
    );
  }
  for (const segment of queueResult.resourceSegments || []) {
    const activeA = segment.actionPhase === "gpu",
      activeV = segment.visionPhase === "gpu";
    if (!activeA && !activeV) continue;
    const sharing = queueConfig.computeMode === "shared" && activeA && activeV;
    const label =
      activeA && activeV
        ? sharing
          ? "V + I 共享"
          : "V / I 独立"
        : activeA
          ? "I 独占"
          : "V 独占";
    block(
      segment.startMs,
      segment.endMs,
      534,
      label,
      activeA ? "inference" : "vision",
      `GPU 阶段：${label}；I 速度 ${fmt(segment.rateA, 2)}，V 速度 ${fmt(segment.rateV, 2)}`,
    );
    if (sharing)
      resourceOverlay(
        { label, segments: [{ ...segment, overlap: true, phase: "gpu" }] },
        534,
      );
  }
  for (const request of queueResult.actionRequests) {
    if (request.startMs < start || request.startMs >= end) continue;
    if (selectedDependency && selectedDependency.id !== request.id) continue;
    const parent = queueResult.visualJobs.find(
      (j) => j.id === request.featureVersion,
    );
    if (!parent) continue;
    const origin = Math.max(start, parent.finishMs),
      target = request.startMs;
    body += `<g class="lab-dependency" data-parent-feature="${parent.id}" data-child-request="${request.id}"><title>${aliases.vision(parent.id)} → ${aliases.action(request).label}：实际读取的特征，${fmt(parent.finishMs)} ms 已完成</title><path d="M${x(origin)},205 L${x(target)},232 l-3,-6 m3,6 l3,-6" fill="none" stroke="var(--s2)" stroke-width="${selectedDependency ? 2.5 : 1}" opacity="${selectedDependency ? 1 : 0.45}"${parent.finishMs < start ? ' stroke-dasharray="3 2"' : ""}/></g>`;
  }
  body += "</g>";
  if (queueTime >= start && queueTime <= end)
    body += `<line x1="${x(queueTime)}" x2="${x(queueTime)}" y1="29" y2="594" stroke="#b77725" stroke-width="2"/><circle cx="${x(queueTime)}" cy="29" r="4" fill="#b77725"/>`;
  $("queue-timeline").innerHTML = svg(width, height, body);
}
function renderQueueSnapshots(snapshot) {
  const { vision, action, execution, rolling } = snapshot;
  const executionScroll =
    $("queue-execution").querySelector(".queue-scroll")?.scrollLeft || 0;
  const runningVision = vision.running;
  $("queue-image").innerHTML =
    `<div class="queue-row-label">已启动的视觉请求</div><div class="queue-slots">${runningVision ? queueSlot(runningVision.label, `${runningVision.sourceLabel} · ${queuePhase(runningVision, snapshot.resource, "v")}`, "vision") : queueEmpty(vision.admissionReason === "rate-limit" ? `接纳限流至 ${fmt(vision.nextAdmissionMs)} ms` : "VLM 空闲")}</div><div class="queue-row-label">1 个图像等待位</div><div class="queue-slots">${vision.pending ? queueSlot(vision.pending.label, `采集 ${fmt(vision.pending.captureMs)} ms`, "vision") : queueEmpty("新帧到来时填入")}</div><p class="muted">${runningVision ? `当前请求已经过 ${fmt(queueTime - runningVision.startMs)} ms；` : ""}图像等待位的覆盖与请求内部等 GPU 分别处理。</p>`;
  const feature = vision.latestFeature,
    reader = action.running || action.ready;
  $("queue-feature").innerHTML =
    `<div class="queue-row-label">最新完成版本</div><div class="queue-slots">${queueSlot(`F${feature.id}`, feature.id < 0 ? "图前初始化" : `${feature.label} 于 ${fmt(feature.finishMs)} ms 完成`, "vision")}</div><div class="queue-row-label">动作请求已经读取的快照</div><div class="queue-slots">${reader ? queueSlot(`${reader.label} / F${reader.featureVersion}`, `状态 s${reader.requestTick} · 请求于 ${fmt(reader.startMs)} ms`, "inference") : queueEmpty("无未完成动作请求")}</div><p class="muted">年龄 ${feature.id < 0 ? "未知（图前初始化）" : `${fmt(queueTime - feature.captureMs)} ms`}；缓存更新不会改变已开始请求的输入。</p>`;
  $("queue-request").innerHTML =
    `<div class="queue-row-label">最多 1 个在途请求</div><div class="queue-slots">${action.running ? queueSlot(action.running.label, `d=${action.running.d} · ${queuePhase(action.running, snapshot.resource, "a")}`, "inference") : queueEmpty("计算 worker 空闲")}</div><div class="queue-row-label">完成结果的暂存位</div><div class="queue-slots">${action.ready ? action.ready.outputs.map((output) => queueActionSlot(output, { requestId: action.ready.id })).join("") : queueEmpty("完成后等待控制循环领取")}</div><p class="muted">${action.ready ? `${queueEscape(action.ready.label)} 于 ${fmt(action.ready.finishMs)} ms 完成，一份结果含 ${action.ready.outputs.length} 个动作。` : action.running ? `请求已经过 ${fmt(queueTime - action.running.startMs)} ms；GPU 等待包含在响应时间内。` : "忙时不排队保存旧状态；下次获准请求时读取新状态。"}</p>`;
  const last = execution.lastExecuted,
    slotMap = new Map(execution.slots.map((slot) => [slot.targetTick, slot]));
  const lastTick = Math.max(
    execution.nextTick + 11,
    ...execution.slots.map((slot) => slot.targetTick),
  );
  const slots = [];
  for (let tick = execution.nextTick; tick <= lastTick; tick++) {
    const slot = slotMap.get(tick);
    slots.push(
      `<div class="queue-target"><span>t${tick} · ${fmt(tick * queueConfig.periodMs, 0)} ms</span>${slot ? queueActionSlot(slot, { committed: slot.committed }) : queueEmpty("未来待补")}</div>`,
    );
  }
  const rejected = queueResult.actionRequests
    .flatMap((request) => request.outputs)
    .filter(queueUnused)
    .slice(-8);
  const lastFreshness = last
    ? queueFreshness({ ...last.origin, executedAtMs: last.timeMs })
    : null;
  $("queue-execution").innerHTML =
    `<div class="queue-executed"><span>刚执行</span><strong>${last ? queueEscape(last.origin.label) : "—"}</strong><small>${last ? `t${last.tick} · ${fmt(last.timeMs)} ms · ${queueExecutionName(last)}` : "尚未执行"}</small>${lastFreshness ? `<span class="executed-state-age" ${queueFreshnessAttributes(lastFreshness)} title="${queueEscape(lastFreshness.tooltip)}">${lastFreshness.ageLabel}${lastFreshness.latest ? " · 最新批次" : ""}</span>` : ""}</div><div class="queue-scroll" aria-label="未来执行槽，可横向滚动"><div class="queue-targets">${slots.join("")}</div></div><p class="muted">${queueConfig.executionPolicy === "paper" ? `${execution.slots.length} 个当前段的干净动作待执行；用完后在控制 tick 接纳新段，未就绪时重复最后动作。图中的未去噪尾部不可执行。` : `${execution.slots.length} 个未来动作已存储；承诺边界 t${execution.committedUntil}（此前的尚未执行格受保护）。虚线空格表示未来待补，不表示此刻已发生 fallback。`}</p>${rejected.length ? `<div class="queue-row-label">最近确认未采用的输出</div><div class="queue-slots">${rejected.map((output) => queueActionSlot(output)).join("")}</div>` : ""}`;
  $("queue-execution").querySelector(".queue-scroll").scrollLeft =
    executionScroll;
  const names = {
    committed: "已承诺前缀",
    clean: "干净",
    "becoming-clean": "本次去噪输出区",
    "partially-denoised": "后续继续细化",
    noise: "新噪声尾部",
  };
  $("queue-rolling").innerHTML =
    rolling.cells
      .map(
        (cell) =>
          `<div class="rolling-cell rolling-${cell.kind}" title="位置 ${cell.offset} / ${queueConfig.executionPolicy === "paper" ? "缓冲位置，非预定执行 tick" : `目标 t${cell.targetTick}`} / ${names[cell.kind]}"><b>${cell.offset}</b><small>${queueConfig.executionPolicy === "paper" ? `p${cell.offset}` : `t${cell.targetTick}`}</small></div>`,
      )
      .join("") +
    '<div class="rolling-key"><span>蓝：计算中的承诺前缀</span><span>橙：干净 / 本次输出区</span><span>浅灰：仍待细化</span><span>斜纹：新噪声尾部</span></div>';
  $("queue-buffer-label").textContent =
    `H=${rolling.horizon} · d=${rolling.slideSteps} · ${rolling.phase === "computing" ? "计算中" : rolling.phase === "post-slide" ? "完成后已左移" : "初始化"}`;
}
function queueEventDescription(event) {
  const request =
    event.requestId !== undefined
      ? queueResult.actionRequests[event.requestId]
      : null;
  switch (event.kind) {
    case "gpu-ready":
    case "gpu-start": {
      const job =
        event.role === "action"
          ? queueResult.actionRequests[event.jobId]
          : queueResult.visualJobs[event.jobId];
      return `${job.label} ${event.kind === "gpu-ready" ? "非 GPU 阶段完成，等待 GPU 准入" : "开始 GPU 阶段"}`;
    }
    case "camera":
      return `O${event.frameId} 到达图像等待位`;
    case "vision-start":
      return `${queueResult.visualJobs[event.jobId].label} 开始，读取 O${event.frameId}`;
    case "vision-finish":
      return `${queueResult.visualJobs[event.jobId].label} 完成，F${event.jobId} 成为最新缓存`;
    case "frame-drop":
      return `O${event.frameId} 被 O${event.replacementId} 覆盖，未进入 VLM`;
    case "action-start":
      return `${request.label} 开始；读取 F${event.featureVersion}，d=${request.d}`;
    case "action-finish":
      return `${request.label} 完成，${request.outputs.length} 个动作等待领取`;
    case "publish":
      return `${request.label} 被领取：采用 ${event.publication.installed}，过期 ${event.publication.expired}，受保护 ${event.publication.protected}`;
    case "reserve":
      return `承诺 t${event.startTick}–t${event.startTick + event.length - 1}：${event.slots.map((slot) => slot.label).join("、")}`;
    case "execute":
      return `t${event.execution.tick} ${queueExecutionName(event.execution)} ${event.execution.origin.label}`;
    case "delay":
      return `d 更新为 ${event.delay}；${queueConfig.executionPolicy === "paper" ? "按近期已完成响应的滚动均值，可升可降" : `本次所需 ${event.requiredDelay} tick，只升不降`}`;
    case "action-ready-replaced":
    case "action-superseded":
    case "ready-superseded":
      return `${request?.label || "旧结果"} 在接纳前被较新的完成结果覆盖`;
    case "window-end":
      return "观察窗口结束；未来输出未采用与否尚未判定";
    case "action-drained":
      return `${request.label} 收尾完成；${queueConfig.executionPolicy === "paper" ? "窗口结束后的输出不再接纳" : "按窗口末尾控制 tick 检查已过期输出"}`;
    default:
      return null;
  }
}
function renderQueue() {
  if (!queueResult) return;
  const snapshot = M.policyQueueSnapshot(queueResult, queueTime),
    metrics = queueResult.metrics;
  queueFreshnessContext = {
    latestAdoptedId: null,
    latestCompletedId: null,
    committedUntil: snapshot.execution.committedUntil,
  };
  for (const request of queueResult.actionRequests) {
    if (
      request.finishMs <= queueTime &&
      request.outputs.some((output) => !queueUnused(output))
    )
      queueFreshnessContext.latestCompletedId = request.id;
    if (
      request.outputs.some(
        (output) =>
          output.adoptedAtMs !== null && output.adoptedAtMs <= queueTime,
      )
    )
      queueFreshnessContext.latestAdoptedId = request.id;
  }
  $("queue-time").value = queueTime;
  $("queue-time-label").textContent = `${fmt(queueTime, 0)} ms`;
  $("queue-delay").textContent = snapshot.delay;
  $("queue-delay-note").textContent =
    `初始 ${queueResult.initialDelay} / 最终 ${queueResult.finalDelay}；${queueConfig.delayMode === "fixed" ? "固定预算" : queueConfig.executionPolicy === "paper" ? "滚动均值，可升可降" : "P95 初始标定，在线只升不降"}${queueResult.calibration.overBudget ? " · 标定已超过最大 d" : ""}`;
  $("queue-rates").textContent =
    `${fmt(metrics.requests / queueConfig.seconds)} / ${fmt(metrics.controlTicks / queueConfig.seconds)}`;
  $("queue-losses").textContent =
    `${metrics.expiredOutputs} / ${metrics.fallbackTicks}`;
  $("queue-losses-note").textContent =
    `整段模拟（含 ${metrics.expiredAtWindowEnd} 个收尾过期）；另有 ${metrics.protectedOutputs} 个受承诺保护；重复末动作 ${metrics.repeatTicks || 0} tick（来源保留）`;
  $("queue-age").textContent =
    snapshot.vision.latestFeature.id < 0
      ? "未知"
      : `${fmt(queueTime - snapshot.vision.latestFeature.captureMs)} ms`;
  const readAges = queueResult.actionRequests
    .filter((r) => r.featureVersion >= 0)
    .map((request) => request.startMs - request.featureCaptureMs);
  $("queue-age-note").textContent =
    `请求读入年龄 P95 ${fmt(M.percentile(readAges, 0.95))} ms`;
  renderQueueTimeline();
  renderQueueSnapshots(snapshot);
  const events = queueResult.events
    .filter(
      (event) => event.timeMs <= queueTime && event.kind !== "action-consumed",
    )
    .slice(-12);
  $("queue-events").innerHTML = events
    .map(
      (event) =>
        `<div class="queue-event${event.kind === "execute" ? " execution-event" : ""}"><time>${fmt(event.timeMs, 1)} ms</time><span>${queueEscape(queueEventDescription(event) || event.kind)}</span></div>`,
    )
    .join("");
  window.PiR2Lab.renderAoi(
    queueAoi,
    queueTime,
    queueResult.windowEndMs,
    queueResult,
  );
  window.PiR2Lab.renderPolicyState(
    queueResult,
    snapshot,
    queueTime,
    queueExecutionName(snapshot.execution.lastExecuted),
  );
  window.lastQueueSimulation = {
    config: { ...queueConfig },
    effectiveConfig: { ...scenarioProjection.config },
    scenario: { ...scenarioSettings },
    reference: scenarioReference,
    aoi: queueAoi,
    calibrationSource,
    result: queueResult,
    baseline: queueBaseline,
    protocolReference: queueProtocolReference,
    snapshot,
    freshness: { ...queueFreshnessContext },
    timeMs: queueTime,
  };
}
function renderQueueResourceSummary() {
  const metrics = queueResult.metrics,
    base = queueBaseline.metrics;
  $("queue-resource-mode").textContent =
    queueModeNames[queueConfig.computeMode];
  for (const [side, key] of [
    ["action", "actionResponseP95Ms"],
    ["vision", "visionResponseP95Ms"],
  ]) {
    $(`queue-${side}-response`).textContent = `${fmt(metrics[key])} ms`;
    const difference = metrics[key] - base[key];
    $(`queue-${side}-baseline`).textContent =
      `独立 GPU 对照（相同主机策略）${fmt(base[key])} ms；差值 ${difference >= 0 ? "+" : ""}${fmt(difference)} ms`;
  }
  $("queue-overlap").textContent =
    `${fmt((100 * metrics.gpuOverlapMs) / queueResult.windowEndMs)}%`;
  $("queue-overlap-note").textContent =
    `${fmt(metrics.gpuOverlapMs)} ms / ${fmt(queueResult.windowEndMs, 0)} ms 观察窗口`;
  $("queue-resource-costs").textContent =
    `累计共享慢化：I +${fmt(metrics.contentionDelayAMs)} / V +${fmt(metrics.contentionDelayVMs)} ms。累计等 GPU：I ${fmt(metrics.gpuWaitAMs)} / V ${fmt(metrics.gpuWaitVMs)} ms。P95 与累计成本包含窗口后收尾；分别统计两个 worker，不把二者之和当作总墙钟时间。`;
}
function stopQueuePlayback() {
  if (queueAnimation !== null) cancelAnimationFrame(queueAnimation);
  queueAnimation = null;
  queueLastFrame = null;
  $("queue-play").textContent = "播放";
  $("queue-play").setAttribute("aria-label", "播放时序");
}
function runQueue() {
  stopQueuePlayback();
  const before = performance.now();
  $("queue-error").hidden = true;
  try {
    scenarioProjection = S.projectScenario(queueConfig, scenarioSettings);
    queueResult = M.simulatePolicyQueues(scenarioProjection.config);
    selectedDependency = null;
    scenarioReference = M.simulatePolicyQueues(queueConfig);
    queueAoi = S.simulationAoi(queueResult);
    queueProtocolReference = M.simulatePolicyQueues({
      ...scenarioProjection.config,
      executionPolicy:
        queueConfig.executionPolicy === "paper" ? "libero" : "paper",
    });
    queueBaseline =
      queueConfig.computeMode === "independent"
        ? queueResult
        : M.simulatePolicyQueues({
            ...scenarioProjection.config,
            computeMode: "independent",
          });
    queueTime = Math.max(0, Math.min(queueTime, queueResult.windowEndMs));
    $("queue-time").max = queueResult.windowEndMs;
    renderQueue();
    renderQueueResourceSummary();
    window.PiR2Lab.renderComparison(
      scenarioProjection,
      scenarioReference,
      queueResult,
    );
    window.PiR2Lab.renderPolicyComparison(queueResult, queueProtocolReference);
    $("queue-runtime").textContent =
      `${queueResult.events.length} 个事件 · 种子 ${queueConfig.seed} · 生成 ${fmt(performance.now() - before, 0)} ms`;
  } catch (error) {
    $("queue-error").hidden = false;
    $("queue-error").textContent = `无法生成当前情景：${error.message}`;
    queueResult = null;
    window.lastQueueSimulation = null;
  }
}
$("queue-controls").addEventListener("submit", (event) =>
  event.preventDefault(),
);
$("queue-controls").addEventListener("input", (event) => {
  const key = event.target.dataset.queueKey;
  if (!key && !event.target.dataset.scenarioKey) return;
  const number = $(`queue-${key}`);
  if (key && event.target.type === "range" && number !== event.target)
    number.value = event.target.value;
  clearTimeout(queueRunTimer);
  queueRunTimer = setTimeout(() => {
    readQueueControls(true);
    runQueue();
  }, 100);
});
$("queue-controls").addEventListener("change", (event) => {
  if (event.target.type !== "number") return;
  clearTimeout(queueRunTimer);
  readQueueControls();
  runQueue();
});
$("queue-reset").addEventListener("click", () => {
  clearTimeout(queueRunTimer);
  resetScenario();
  queueConfig = {
    ...M.queueDefaults,
    executionPolicy: queueConfig.executionPolicy,
  };
  queueTime = Math.min(260, queueConfig.seconds * 1000);
  setQueueControls();
  runQueue();
});
for (const button of document.querySelectorAll("[data-queue-preset]"))
  button.addEventListener("click", () => {
    clearTimeout(queueRunTimer);
    resetScenario();
    const timings = { 1: [30, 20], 2: [140, 20], 3: [30, 80], 4: [140, 80] }[
      button.dataset.queuePreset
    ];
    queueConfig = {
      ...M.queueDefaults,
      executionPolicy: queueConfig.executionPolicy,
      vlmMs: timings[0],
      actionMs: timings[1],
    };
    setQueueControls();
    runQueue();
  });
$("queue-shared-preset").addEventListener("click", () => {
  clearTimeout(queueRunTimer);
  resetScenario();
  queueConfig = {
    ...M.queueDefaults,
    executionPolicy: queueConfig.executionPolicy,
    vlmMs: 100,
    actionMs: 20,
    computeMode: "shared",
    slowdownA: 1,
    slowdownV: 1,
  };
  queueTime = 25;
  setQueueControls();
  runQueue();
});
$("queue-time").addEventListener("input", () => {
  stopQueuePlayback();
  queueTime = Number($("queue-time").value);
  renderQueue();
});
$("queue-step").addEventListener("click", () => {
  if (!queueResult) return;
  stopQueuePlayback();
  queueTime = Math.min(
    queueResult.windowEndMs,
    (Math.floor(queueTime / queueConfig.periodMs) + 1) * queueConfig.periodMs,
  );
  renderQueue();
});
$("queue-play").addEventListener("click", () => {
  if (!queueResult) return;
  if (queueAnimation !== null) {
    stopQueuePlayback();
    return;
  }
  if (queueTime >= queueResult.windowEndMs) queueTime = 0;
  $("queue-play").textContent = "暂停";
  $("queue-play").setAttribute("aria-label", "暂停时序");
  const frame = (now) => {
    if (queueLastFrame !== null)
      queueTime = Math.min(
        queueResult.windowEndMs,
        queueTime + (now - queueLastFrame) * Number($("queue-speed").value),
      );
    queueLastFrame = now;
    renderQueue();
    if (queueTime >= queueResult.windowEndMs) stopQueuePlayback();
    else queueAnimation = requestAnimationFrame(frame);
  };
  queueAnimation = requestAnimationFrame(frame);
});
$("queue-export").addEventListener("click", () => {
  if (!queueResult) return;
  const payload = {
    schema: 3,
    kind: "pir2_policy_queue_simulation",
    generatedAt: new Date().toISOString(),
    config: queueResult.config,
    baseConfig: queueConfig,
    scenarioAssumptions: scenarioSettings,
    soloCalibrationSource: calibrationSource,
    samePolicyReference: {
      config: scenarioReference.config,
      metrics: scenarioReference.metrics,
    },
    aoi: queueAoi,
    otherSchedulingPolicy: {
      config: queueProtocolReference.config,
      metrics: queueProtocolReference.metrics,
    },
    assumptions:
      "Inputs are isolated full-request latencies split into pre-GPU host and GPU work. Independent, overlapping GPU slowdown, or nonpreemptive serial resource scheduling selected in config. executionPolicy=libero uses isolated P95 plus margin, monotone d, timestamped protected slots and zero-delta hold on exhaustion. executionPolicy=paper is a mechanism reference: continuously queried action worker, rolling completed-latency mean with round-to-even d, latest-ready result, clean-segment boundary swaps and last-emitted-action repetition. Its initial d and maximum d are explicit simulator assumptions; per-request clean width is frozen, not a bit-exact deployment reproduction. Repeated outputs retain their original producer and AoI. Optional shared nonpreemptive host server and VLM admission cap. Quantization scales only assumed GPU work, with optional added host work; precision labels are not benchmark results. AoI follows consumed features and producing actions; unknown bootstrap/fallback intervals are gaps. Synthetic scheduling only; no measured GPU, quantization accuracy or task success claim.",
    result: queueResult,
    independentReference: {
      config: queueBaseline.config,
      metrics: queueBaseline.metrics,
      calibration: queueBaseline.calibration,
    },
    snapshot: M.policyQueueSnapshot(queueResult, queueTime),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    }),
    url = URL.createObjectURL(blob),
    link = document.createElement("a");
  link.href = url;
  link.download =
    "pir2-queues-" + new Date().toISOString().replace(/[:.]/g, "-") + ".json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
function resetScenario() {
  scenarioSettings = { ...S.defaults };
  calibrationSource = null;
  selectedDependency = null;
  $("lab-calibration-source").textContent =
    "机制示例：当前参数为可编辑假设，未拟合硬件性能。";
}
function applyLabConfig(config, assumptions = {}) {
  clearTimeout(queueRunTimer);
  resetScenario();
  queueConfig = { ...M.queueDefaults, ...config };
  scenarioSettings = { ...S.defaults, ...assumptions };
  queueTime = 260;
  setQueueControls();
  runQueue();
}
$("lab-example-feedback").addEventListener("click", () => {
  applyLabConfig(
    {
      actionMs: 30,
      vlmMs: 200,
      gpuShareA: 1 / 3,
      gpuShareV: 0.8,
      periodMs: 50,
      cameraHz: 20,
      computeMode: "shared",
      slowdownA: 0,
      slowdownV: 0,
      hostMode: "shared",
      seconds: 12,
      marginMs: 5,
    },
    { visionSpeedup: 2, visionPrecision: "INT8 / W8A8" },
  );
});
$("lab-example-latch").addEventListener("click", () => {
  applyLabConfig({
    actionMs: 48,
    vlmMs: 30,
    periodMs: 50,
    marginMs: 0,
    startupActionDelayMs: 3,
    computeMode: "independent",
    seconds: 4,
  });
});
$("queue-executionPolicy").addEventListener("change", () => {
  clearTimeout(queueRunTimer);
  readQueueControls();
  runQueue();
});
$("lab-example-repeat").addEventListener("click", () => {
  applyLabConfig({
    executionPolicy: "paper",
    delayMode: "fixed",
    fixedDelay: 2,
    paperInitialDelay: 2,
    actionMs: 160,
    vlmMs: 30,
    periodMs: 50,
    seconds: 2,
    bootstrapSlots: 2,
  });
  queueTime = 450;
  renderQueue();
});
$("lab-example-recover-d").addEventListener("click", () => {
  applyLabConfig({
    executionPolicy: "paper",
    paperInitialDelay: 1,
    delayWindow: 2,
    actionMs: 40,
    vlmMs: 30,
    periodMs: 50,
    startupActionDelayMs: 100,
    seconds: 2,
  });
  queueTime = 300;
  renderQueue();
});
$("lab-compare-cap").addEventListener("click", () => {
  queueConfig.vlmRateCapHz = 5;
  setQueueControls();
  runQueue();
});
$("lab-timeline-span").addEventListener("change", renderQueueTimeline);
$("queue-timeline").addEventListener("click", (event) => {
  if (!queueResult) return;
  const node = event.target.closest(
    "[data-action-id], [data-vision-id], [data-execution-tick]",
  );
  if (!node) {
    selectedDependency = null;
    renderQueueTimeline();
    return;
  }
  selectedDependency = node.hasAttribute("data-execution-tick")
    ? node.dataset.originRequest === ""
      ? null
      : queueResult.actionRequests[Number(node.dataset.originRequest)]
    : node.hasAttribute("data-action-id")
      ? queueResult.actionRequests[Number(node.dataset.actionId)]
      : queueResult.actionRequests.find(
          (r) => r.featureVersion === Number(node.dataset.visionId),
        );
  renderQueueTimeline();
});
for (const id of ["lab-vision-budget", "lab-action-budget"])
  $(id).addEventListener("input", () => {
    if (queueResult)
      window.PiR2Lab.renderComparison(
        scenarioProjection,
        scenarioReference,
        queueResult,
      );
  });
window.addEventListener("message", (event) => {
  if (
    event.origin !== location.origin ||
    event.source !== $("lab-replay-frame").contentWindow
  )
    return;
  const source = S.safeSoloMessage(event.data);
  if (!source) return;
  applyLabConfig({ ...source.config, executionPolicy: "libero" });
  calibrationSource = source;
  $("lab-calibration-source").textContent =
    `已按本地 LIBERO 调度口径载入 ${source.conditionId} 的独占 RPC 均值：VLM ${fmt(source.config.vlmMs)} ms，DiT ${fmt(source.config.actionMs)} ms。完整 RPC 暂按 GPU 比例 1 建模；主机比例、争用系数与量化收益尚未拟合，请在高级参数中设置。频率不自动换算成速度。`;
  renderQueue();
  window.PiR2Lab.showTab("simulation");
});
setQueueControls();
runQueue();
window.PiR2Lab.initNavigation();
