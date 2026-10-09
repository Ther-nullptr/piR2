/* Presentation helpers for measured replay and explicit scheduling assumptions. */
"use strict";
window.PiR2Lab = (() => {
  const $ = (id) => document.getElementById(id),
    S = window.PiR2Scenarios;
  const fmt = (n, d = 1) =>
    n === null || !Number.isFinite(n) ? "—" : n.toFixed(d);
  const escape = (text) =>
    String(text).replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  function showTab(name) {
    const replay = name === "replay";
    $("lab-replay").hidden = !replay;
    $("queue-mode").hidden = replay;
    for (const tab of ["replay", "simulation"]) {
      $("lab-tab-" + tab).setAttribute("aria-pressed", String(tab === name));
      $("lab-tab-" + tab).classList.toggle("active", tab === name);
    }
  }
  async function initNavigation() {
    $("lab-tab-replay").addEventListener("click", () => showTab("replay"));
    $("lab-tab-simulation").addEventListener("click", () =>
      showTab("simulation"),
    );
    showTab("simulation");
    let available = false;
    if (location.protocol !== "file:") {
      try {
        const response = await fetch("/experiment-status.json");
        available = response.ok && (await response.json()).available === true;
      } catch {
        /* Offline simulation remains available. */
      }
    }
    $("lab-evidence-status").textContent = available
      ? "本地实验报告已连接"
      : "实测报告未连接 · 机制仿真可用";
    $("lab-replay-unavailable").hidden = available;
    $("lab-replay-frame").hidden = !available;
    if (available) {
      $("lab-replay-frame").src = "/experiment-report.html?embedded=1";
      showTab("replay");
    }
  }
  function renderComparison(projection, reference, result) {
    const rows = [
      ["VLM RPC P95（含收尾）", "visionResponseP95Ms", "ms"],
      ["DiT RPC P95（含收尾）", "actionResponseP95Ms", "ms"],
      ["实际 VLM 接纳率", "admittedVlmHz", "Hz"],
      ["DiT 请求率", (r) => r.metrics.requests / r.config.seconds, "Hz"],
      ["控制频率", (r) => r.metrics.controlTicks / r.config.seconds, "Hz"],
      ["实际截止期违约", "deadlineMisses", "次"],
      ["最终 d", (r) => r.finalDelay, "tick"],
      ["无来源保持", "fallbackTicks", "tick"],
      ["重复最后动作", (r) => r.metrics.repeatTicks || 0, "tick"],
      ["继续旧段后续动作", (r) => r.metrics.continuedSegmentTicks || 0, "tick"],
      [
        "主机有任务占用的时间",
        (r) => (100 * r.metrics.hostBusyMs) / r.windowEndMs,
        "%",
      ],
      ["DiT 主机累计等待", "hostWaitAMs", "ms"],
      ["VLM 主机累计等待", "hostWaitVMs", "ms"],
    ];
    const value = (r, key) =>
      typeof key === "function" ? key(r) : r.metrics[key];
    $("lab-compare-table").innerHTML =
      "<table><thead><tr><th>指标</th><th>原始耗时 + 当前调度</th><th>应用加速假设</th></tr></thead><tbody>" +
      rows
        .map(
          ([label, key, unit]) =>
            `<tr><td>${label}</td><td>${fmt(value(reference, key))} ${unit}</td><td>${fmt(value(result, key))} ${unit}</td></tr>`,
        )
        .join("") +
      "</tbody></table>";
    const describe = (role, c) =>
      `${role}：主机 ${fmt(c.baseHostMs)} + 新增 ${fmt(c.extraHostMs)} + GPU ${fmt(c.baseGpuMs)} / ${fmt(c.speedupAssumption, 2)} = ${fmt(c.totalMs)} ms（标签 ${escape(c.precisionLabel)}）`;
    $("lab-transform-note").innerHTML =
      describe("VLM", projection.components.vision) +
      "；" +
      describe("DiT", projection.components.action) +
      "。对照保留同一接纳上限、主机策略、首次开销与随机种子。";
    const h = result.metrics;
    $("lab-load-note").textContent =
      `窗口内主机工作：DiT ${fmt(h.hostBusyAMs)} ms，VLM ${fmt(h.hostBusyVMs)} ms。${result.config.hostMode === "shared" ? "共享主机服务排队；更多 VLM 请求可能延长 DiT 等待。" : "独立主机服务允许重叠，二者工作时长之和不能当作墙钟时间。"}接纳率按请求数 / 窗口统计，起点即时接纳可能使短窗平均值略高于限流值。首次额外开销只作用于首个 DiT 请求，LIBERO 独占标定不包含它；论文参考的在线滚动窗口包含它。`;
    const vb = Number($("lab-vision-budget").value),
      ab = Number($("lab-action-budget").value);
    if (!(vb >= 0.1 && vb <= 10000 && ab >= 0.1 && ab <= 10000)) {
      $("lab-state-class").textContent = "请输入 0.1–10000 ms 的固定预算";
      return;
    }
    const state = S.classify(
      h.visionResponseP95Ms,
      h.actionResponseP95Ms,
      vb,
      ab,
    );
    $("lab-state-class").dataset.state = state.id;
    $("lab-state-class").textContent =
      state.label +
      ` · 固定预算 VLM ${fmt(vb)} / DiT ${fmt(ab)} ms；与自适应 d 的截止期分别统计。`;
  }
  function renderAoi(aoi, time, end, result) {
    const series = [
      ["cache", "最新缓存", "var(--s2)"],
      ["executionImage", "执行动作的视觉来源", "var(--s1)"],
      ["executionState", "执行动作的状态来源", "var(--state)"],
    ];
    const W = 1200,
      H = 285,
      left = 70,
      right = 1175,
      top = 25,
      bottom = 227;
    const max = Math.max(
      50,
      ...series.map(([key]) => aoi.summary[key].maxMs || 0),
    );
    const x = (t) => left + (t / end) * (right - left),
      y = (v) => bottom - (v / max) * (bottom - top);
    let body = "";
    for (const event of result?.executions || []) {
      if (!event.repeated) continue;
      const stop = Math.min(end, event.timeMs + result.config.periodMs);
      body += `<rect class="lab-aoi-repeat" data-origin-request="${event.origin.requestId ?? ""}" x="${x(event.timeMs)}" y="${top}" width="${x(stop) - x(event.timeMs)}" height="${bottom - top}" fill="#8055a0" opacity=".08"><title>重复末动作；来源时间戳保持，AoI 不重置</title></rect>`;
    }
    for (let i = 0; i <= 4; i++) {
      const v = (max * i) / 4,
        t = (end * i) / 4;
      body += `<line x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}" stroke="#e1e7ea"/><text x="${left - 8}" y="${y(v) + 4}" text-anchor="end" font-size="11">${fmt(v, 0)}</text><text x="${x(t)}" y="250" text-anchor="middle" font-size="11">${fmt(t, 0)}</text>`;
    }
    body +=
      '<text x="12" y="17" font-size="12">AoI / ms</text><text x="1140" y="278" font-size="12">时间 / ms</text>';
    for (const [key, name, color] of series) {
      let prev = null,
        path = "";
      for (const row of aoi[key]) {
        const a = `${x(row.startMs)},${y(row.startMs - row.generationMs)}`,
          b = `${x(row.endMs)},${y(row.endMs - row.generationMs)}`;
        path += `${prev && Math.abs(prev.endMs - row.startMs) < 1e-6 ? " L" : " M"}${a} L${b}`;
        prev = row;
      }
      body += `<path class="lab-aoi-series" data-series="${key}" d="${path}" stroke="${color}" stroke-width="2" fill="none"><title>${name}；已知区间均值 ${fmt(aoi.summary[key].meanMs)} ms；P95 ${fmt(aoi.summary[key].p95Ms)} ms；已知覆盖 ${fmt((aoi.summary[key].knownMs / end) * 100)}%</title></path>`;
    }
    for (const use of aoi.uses)
      body += `<circle class="lab-aoi-use" cx="${x(use.timeMs)}" cy="${y(use.ageMs)}" r="2" fill="#99602b"><title>DiT ${use.requestId} 读取的特征龄 ${fmt(use.ageMs)} ms</title></circle>`;
    body += `<line x1="${x(time)}" x2="${x(time)}" y1="${top}" y2="${bottom}" stroke="#9b3e43" stroke-dasharray="4 3"/>`;
    $("lab-aoi-chart").innerHTML =
      `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="模拟 AoI 曲线">${body}</svg>`;
    $("lab-aoi-now").textContent = series
      .map(([key, name]) => `${name} ${fmt(S.ageAt(aoi[key], time))} ms`)
      .join(" · ");
  }
  function renderPolicyState(result, snapshot, time, executionName) {
    const paper = result.config.executionPolicy === "paper",
      fixed = result.config.delayMode === "fixed";
    const last = snapshot.execution.lastExecuted;
    $("lab-policy-live").textContent =
      `${paper ? "论文参考" : "本地 LIBERO"} · 当前 d=${snapshot.delay} · ${last ? `${executionName} ${last.origin.label}` : "尚未执行"}`;
    $("lab-policy-description").textContent = paper
      ? "论文参考：旧段续播，耗尽后重复末动作；滚动 d 可升可降。"
      : "本地 LIBERO：消费已有时间槽；无槽时保持。d 超时上调，不自动回落。";
    $("lab-delay-rule").textContent = fixed
      ? "固定 d 对照"
      : paper
        ? `近 ${result.config.delayWindow} 次均值估计，可升降（d≤${result.config.maxDelay}）`
        : "超时上调，不自动回落";
    const end = result.windowEndMs,
      W = 1200,
      H = 125,
      left = 70,
      right = 1175,
      top = 12,
      bottom = 85,
      max = result.config.maxDelay;
    const x = (t) => left + (t / end) * (right - left),
      y = (d) => bottom - ((d - 1) / Math.max(1, max - 1)) * (bottom - top);
    let body = "";
    for (let d = 1; d <= max; d++)
      body += `<line x1="${left}" x2="${right}" y1="${y(d)}" y2="${y(d)}" stroke="#e1e7ea"/><text x="${left - 12}" y="${y(d) + 4}" text-anchor="end" font-size="11">${d}</text>`;
    let path = `M${x(0)},${y(result.initialDelay)}`;
    for (const e of result.events)
      if (e.kind === "delay" && e.timeMs < end)
        path += ` H${x(e.timeMs)} V${y(e.delay)}`;
    path += ` H${x(end)}`;
    body += `<path data-series="delay" d="${path}" fill="none" stroke="#8055a0" stroke-width="2.5"/><line x1="${x(time)}" x2="${x(time)}" y1="${top}" y2="${bottom}" stroke="#9b3e43" stroke-dasharray="4 3"/><text x="12" y="17" font-size="12">d</text>`;
    for (let i = 0; i <= 4; i++)
      body += `<text x="${x((end * i) / 4)}" y="110" text-anchor="middle" font-size="11">${fmt((end * i) / 4, 0)} ms</text>`;
    $("lab-delay-chart").innerHTML =
      `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="d 随时间变化">${body}</svg>`;
  }
  function renderPolicyComparison(result, other) {
    const paper = result.config.executionPolicy === "paper" ? result : other;
    const libero = result.config.executionPolicy === "libero" ? result : other;
    const rows = [
      ["DiT 请求率 / Hz", (r) => r.metrics.requests / r.config.seconds],
      [
        "采用新动作段 / 次",
        (r) =>
          r.actionRequests.filter((q) => q.publication?.installed > 0).length,
      ],
      ["继续旧段后续动作 / tick", (r) => r.metrics.continuedSegmentTicks || 0],
      ["重复末动作 / tick", (r) => r.metrics.repeatTicks || 0],
      ["无来源保持 / tick", (r) => r.metrics.fallbackTicks],
      ["预算超时 / 请求", (r) => r.metrics.deadlineMisses],
      ["初始 d", (r) => r.initialDelay],
      ["最终 d", (r) => r.finalDelay],
      [
        "执行图像 AoI P95 / ms",
        (r) => S.simulationAoi(r).summary.executionImage.p95Ms,
      ],
    ];
    $("lab-policy-comparison").innerHTML =
      "<table><thead><tr><th>指标</th><th>论文机制参考</th><th>本地 LIBERO</th></tr></thead><tbody>" +
      rows
        .map(
          ([label, get]) =>
            `<tr><td>${label}</td><td>${fmt(get(paper))}</td><td>${fmt(get(libero))}</td></tr>`,
        )
        .join("") +
      "</tbody></table>";
  }
  return {
    showTab,
    initNavigation,
    renderComparison,
    renderAoi,
    renderPolicyState,
    renderPolicyComparison,
  };
})();
