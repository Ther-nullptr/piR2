/* All charts and event simulations run locally; file:// is supported. */
"use strict";
const M = window.VLAModel,
  $ = (id) => document.getElementById(id);
const palette = {
  linear: "#25826a",
  other: "#4675ac",
  added: "#d7934b",
  residual: "#b5bfc4",
  a: "#386da8",
  v: "#218c78",
  red: "#bd5965",
  gray: "#aab8bd",
  gold: "#d5a35c",
};
let config = { ...M.defaults },
  result,
  reference,
  timer;
const fmt = (v, d = 1) =>
  v === null || !Number.isFinite(v) ? "—" : Number(v).toFixed(d);
const text = (x, y, value, extra = "") =>
  `<text x="${x}" y="${y}" fill="#64757c" font-size="11" ${extra}>${value}</text>`;
const svg = (width, height, body) =>
  `<svg viewBox="0 0 ${width} ${height}" role="img" xmlns="http://www.w3.org/2000/svg">${body}</svg>`;
function linePath(points, x, y) {
  return points
    .filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]))
    .map(
      (p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)}`,
    )
    .join(" ");
}
function axes(w, h, xmax, ymax, xlabel = "时间 / ms") {
  const left = 45,
    top = 14,
    bottom = h - 34,
    right = w - 14;
  const x = (t) => left + (t / xmax) * (right - left),
    y = (t) => bottom - (t / ymax) * (bottom - top);
  let body = "";
  for (let i = 0; i <= 4; i++) {
    const v = (ymax * i) / 4,
      yy = y(v);
    body +=
      `<line x1="${left}" x2="${right}" y1="${yy}" y2="${yy}" stroke="#e8edeb"/>` +
      text(
        left - 7,
        yy + 4,
        fmt(v, ymax < 2 ? 2 : ymax < 20 ? 1 : 0),
        'text-anchor="end"',
      );
    body += text(
      x((xmax * i) / 4),
      h - 17,
      fmt((xmax * i) / 4, 0),
      'text-anchor="middle"',
    );
  }
  body += text(w / 2, h - 1, xlabel, 'text-anchor="middle" font-size="10"');
  return { x, y, body };
}
function formatControl(key, value) {
  if (["compute", "bandwidth", "hostScale", "lowbitMath"].includes(key))
    return `${fmt(value, 2)}×`;
  if (["period"].includes(key)) return `${value} ms`;
  if (key === "cameraHz") return `${value} Hz`;
  if (key === "slowCap") return value === 0 ? "不额外限频" : `${value} Hz`;
  if (key === "seconds") return `${value} s`;
  if (key === "jitter") return `±${Math.round(value * 100)}%`;
  if (key === "overhead") return `${value}%`;
  return fmt(value, 2);
}
function setControls() {
  for (const [key, value] of Object.entries(config)) {
    const input = $(key);
    if (!input) continue;
    if (input.type === "checkbox") input.checked = value;
    else input.value = value;
    const output = document.querySelector(`output[for="${key}"]`);
    if (output) output.textContent = formatControl(key, value);
  }
}
function readControls() {
  for (const key of Object.keys(M.defaults)) {
    const input = $(key);
    if (!input) continue;
    if (input.type === "checkbox") config[key] = input.checked;
    else if (typeof M.defaults[key] === "number") {
      let value = Number(input.value);
      if (!Number.isFinite(value)) value = M.defaults[key];
      if (input.hasAttribute("min")) value = Math.max(Number(input.min), value);
      if (input.hasAttribute("max")) value = Math.min(Number(input.max), value);
      config[key] = value;
    } else config[key] = input.value;
  }
  setControls();
}
function delta(id, value, base, unit = "ms") {
  const difference = value - base;
  $(id).textContent =
    value === null
      ? "没有完成请求"
      : `BF16 对照 ${fmt(base)} ${unit} · ${difference > 0 ? "+" : ""}${fmt(difference)} ${unit}`;
  $(id).className =
    difference < -0.01 ? "good" : difference > 0.01 ? "bad" : "";
}
function renderService() {
  const w = 860,
    h = 190,
    left = 145,
    right = 130,
    barWidth = w - left - right;
  const rows = [
    ["S1 · BF16 对照", reference.service.S1],
    ["S1 · 当前", result.service.S1],
    ["S2 · BF16 对照", reference.service.S2],
    ["S2 · 当前", result.service.S2],
  ];
  const maximum = Math.max(...rows.map((x) => x[1].total)) * 1.02;
  let body = "";
  rows.forEach(([label, s], i) => {
    const y = 15 + i * 39;
    body += text(left - 10, y + 17, label, 'text-anchor="end"');
    let cursor = left;
    for (const [key, title] of [
      ["linear", "GEMM"],
      ["other", "其他 GPU 工作"],
      ["added", "量化新增工作"],
      ["residual", "拟合剩余项"],
    ]) {
      const width = (s[key] / maximum) * barWidth;
      body += `<rect x="${cursor}" y="${y}" width="${width}" height="26" rx="2" fill="${palette[key]}"><title>${label} / ${title}: ${fmt(s[key], 3)} ms</title></rect>`;
      if (width > 50)
        body += `<text x="${cursor + width / 2}" y="${y + 17}" text-anchor="middle" fill="${key === "residual" ? "#34454c" : "white"}" font-size="10">${fmt(s[key])}</text>`;
      cursor += width;
    }
    body += text(
      cursor + 9,
      y + 17,
      `${fmt(s.total, 2)} ms`,
      'font-weight="600"',
    );
  });
  body += text(left, 183, "0");
  body += text(left + barWidth, 183, `${fmt(maximum)} ms`, 'text-anchor="end"');
  $("service-chart").innerHTML =
    svg(w, h, body) +
    `<div class="legend">${[
      ["linear", "GEMM"],
      ["other", "其他 GPU"],
      ["added", "新增量化成本"],
      ["residual", "拟合剩余项"],
    ]
      .map(
        ([k, n]) => `<span><i style="background:${palette[k]}"></i>${n}</span>`,
      )
      .join("")}</div>`;
  $("service-note").textContent =
    `当前 Roofline 假设：S1 由${result.service.S1.limiting}决定，S2 由${result.service.S2.limiting}决定。${config.qA === "w4" || config.qV === "w4" ? "W4A16 保持高精度乘法吞吐；若已由计算限制，压缩权重可能几乎不提速。" : "服务时间缩短并不保证响应时间按同样比例缩短。"}`;
}
function renderTimeline() {
  const width = 520,
    height = 150,
    start = Number($("window").value),
    duration = 800,
    left = 40,
    right = width - 10;
  const x = (t) => left + ((t - start) / duration) * (right - left);
  $("window-label").textContent = `${start}–${start + duration} ms`;
  let body = text(2, 47, "S1") + text(2, 100, "S2");
  for (let i = 0; i <= 4; i++) {
    const xx = x(start + (i * duration) / 4);
    body +=
      `<line x1="${xx}" x2="${xx}" y1="15" y2="115" stroke="#e5ece8"/>` +
      text(xx, 137, fmt(start + (i * duration) / 4, 0), 'text-anchor="middle"');
  }
  for (const r of result.timeline) {
    if (r.end <= start || r.start >= start + duration) continue;
    const lo = Math.max(start, r.start),
      hi = Math.min(start + duration, r.end);
    for (const [key, y, color] of [
      ["a", 30, palette.a],
      ["v", 83, palette.v],
    ])
      if (r[key] !== null)
        body += `<rect x="${x(lo)}" y="${y}" width="${Math.max(0.4, x(hi) - x(lo))}" height="24" rx="2" fill="${color}" opacity="${r[key] % 2 ? 0.8 : 1}"><title>${key === "a" ? "S1" : "S2"} 任务 #${r[key]}，片段 ${fmt(r.start)}–${fmt(r.end)} ms</title></rect>`;
  }
  for (const d of result.drops)
    if (d.release >= start && d.release <= start + duration)
      body += `<path d="M${x(d.release) - 3},20 l6,8 m-6,0 l6,-8" stroke="${palette.red}" stroke-width="2"><title>丢弃请求 #${d.id}: ${d.reason}</title></path>`;
  $("timeline").innerHTML = svg(width, height, body);
}
function renderAge() {
  const w = 520,
    h = 230,
    start = Number($("window").value),
    end = start + 800;
  const clean = (r) => {
    const ageAt = (t) =>
      t -
      r.updates.filter((u) => u.finish <= t).reduce((value, u) => u.capture, 0);
    return [
      [start, ageAt(start)],
      ...r.age.filter((p) => p[0] >= start && p[0] <= end),
      [end, ageAt(end)],
    ].map((p) => [p[0] - start, p[1]]);
  };
  const current = clean(result),
    base = clean(reference);
  const max =
    Math.max(1, ...current.map((p) => p[1]), ...base.map((p) => p[1])) * 1.05;
  const a = axes(w, h, 800, max, `窗口内时间 / ms（起点 ${start} ms）`);
  const body =
    a.body +
    `<path d="${linePath(base, a.x, a.y)}" fill="none" stroke="#b9c3c6" stroke-width="1.5"/><path d="${linePath(current, a.x, a.y)}" fill="none" stroke="${palette.v}" stroke-width="1.8"/>`;
  $("age-chart").innerHTML = svg(w, h, body);
  $("age-note").textContent =
    `曲线显示同左侧 800 ms 窗口，统计覆盖整段有效时间。慢环发布 ${fmt(result.updateHz)} Hz · 替换旧等待帧 ${result.replaced} 次 / 丢弃新帧 ${result.droppedNew} 次。${result.theoreticalAoi !== null ? `确定服务的理论均值 1.5S = ${fmt(result.theoreticalAoi, 3)} ms；模拟 ${fmt(result.aoiMean, 3)} ms。` : "当前条件不满足简单 Uniform(S, 2S) 的验证前提。"}`;
}
function renderDelay() {
  if (!result.planned) {
    $("delay-chart").innerHTML =
      '<p class="muted">S1 未启用，没有响应时间或延迟档位。</p>';
    $("buffer").innerHTML = "";
    $("queue-note").textContent =
      "无 S1 请求时，deadline miss rate 未定义；可以单独检查 S2 的 AoI。";
    return;
  }
  const counts = [0, 0, 0, 0];
  result.calls.forEach((r) => counts[Math.min(4, r.d) - 1]++);
  const w = 500,
    h = 128,
    max = Math.max(1, ...counts);
  let body = "";
  counts.forEach((n, i) => {
    const x = 35 + i * 115,
      y = 91 - (n / max) * 58;
    body +=
      `<rect x="${x}" y="${y}" width="65" height="${91 - y}" fill="${palette.a}" rx="4"/>` +
      text(x + 32, y - 6, `${n}`, 'text-anchor="middle"') +
      text(x + 32, 113, `d=${i === 3 ? "4+" : i + 1}`, 'text-anchor="middle"');
  });
  $("delay-chart").innerHTML = svg(w, h, body);
  const d = Math.max(1, Math.ceil(result.dP95 || 1)),
    compatible = d < 8;
  $("buffer").innerHTML =
    `<div class="buffer-cells">${Array.from({ length: 16 }, (_, i) => `<span style="background:${i < d ? palette.a : i < 2 * d ? palette.v : i >= 16 - d ? palette.gray : palette.gold}" title="位置 ${i}">${i}</span>`).join("")}</div><div class="buffer-legend">蓝：已在途 [0,d)　绿：新提交 [d,2d)　灰：噪声尾段</div>`;
  $("queue-note").textContent =
    `P95 延迟档位 d≈${d} · 平均排队 ${fmt(result.waitMean)} ms · 平均在途数 ${fmt(result.populationMean, 2)}。${compatible ? "H/d 仅是固定延迟、连续更新时的近似调用次数，不是噪声独立平均的证明。" : "当前 d 与 H=16 的三段调度不兼容；须改变 H/节奏，不能据此声称策略可运行。"}`;
}
function renderSweep() {
  const keys = Object.keys(M.formats),
    short = ["BF16", "INT8", "FP8", "W4A16"];
  let html =
    '<table class="sweep-table"><thead><tr><th>S1 ↓ / S2 →</th>' +
    short.map((s) => `<th>${s}</th>`).join("") +
    "</tr></thead><tbody>";
  keys.forEach((qa, i) => {
    html += `<tr><th>${short[i]}</th>`;
    keys.forEach((qv) => {
      const r = M.simulate({ ...config, qA: qa, qV: qv });
      const color =
        r.missRate === null
          ? "#f0f3f3"
          : `hsl(${145 - r.missRate * 140}, 28%, ${94 - r.missRate * 14}%)`;
      html += `<td><button data-qa="${qa}" data-qv="${qv}" style="background:${color}" class="${config.qA === qa && config.qV === qv ? "selected" : ""}" aria-label="S1 ${qa}, S2 ${qv}"><b>${r.missRate === null ? "无 S1" : fmt(r.missRate * 100) + "%"}</b>${fmt(r.aoiMean, 0)} ms</button></td>`;
    });
    html += "</tr>";
  });
  $("sweep").innerHTML = html + "</tbody></table>";
  $("sweep")
    .querySelectorAll("button")
    .forEach((button) =>
      button.addEventListener("click", () => {
        config.qA = button.dataset.qa;
        config.qV = button.dataset.qv;
        setControls();
        run();
      }),
    );
}
function renderNoise() {
  const p = {};
  for (const k of ["alpha", "rho", "bias"]) {
    p[k] = Number($(k).value);
    $(`${k}-out`).textContent = fmt(p[k], 3);
  }
  const rows = M.correctionError(p),
    clean = M.correctionError({ ...p, bias: 0, rho: 0 });
  const w = 920,
    h = 195,
    max = Math.max(1, ...rows.map((x) => x.rmse)) * 1.05;
  const a = axes(w, h, 24, max, "修正次数 n（不是毫秒）");
  $("noise-chart").innerHTML = svg(
    w,
    h,
    a.body +
      `<path d="${linePath(
        clean.map((r) => [r.step, r.rmse]),
        a.x,
        a.y,
      )}" fill="none" stroke="#aebbc0" stroke-width="2"/><path d="${linePath(
        rows.map((r) => [r.step, r.rmse]),
        a.x,
        a.y,
      )}" fill="none" stroke="${palette.red}" stroke-width="2.5"/>` +
      text(60, 24, "示意 RMSE") +
      text(
        520,
        24,
        `当前相关/偏置模型：${fmt(rows.at(-1).rmse, 3)}；独立无偏：${fmt(clean.at(-1).rmse, 3)}`,
      ),
  );
}
function run() {
  const before = performance.now();
  $("error").hidden = true;
  try {
    result = M.simulate(config);
    reference = M.simulate({ ...config, qA: "bf16", qV: "bf16" });
    $("latency").textContent = `${fmt(result.responseP95)} ms`;
    delta("latency-delta", result.responseP95, reference.responseP95);
    $("miss").textContent =
      result.missRate === null ? "—" : `${fmt(result.missRate * 100)}%`;
    $("miss-detail").textContent = result.planned
      ? `${result.misses} / ${result.planned} 次释放，含 ${result.dropped} 次丢弃`
      : "未启用 S1 / 无请求释放";
    $("aoi").textContent = `${fmt(result.aoiMean)} ms`;
    delta("aoi-delta", result.aoiMean, reference.aoiMean);
    $("used-age").textContent = `${fmt(result.actionAgeP95)} ms`;
    $("window").max = Math.max(0, config.seconds * 1000 - 800);
    if (Number($("window").value) > Number($("window").max))
      $("window").value = $("window").max;
    renderService();
    renderTimeline();
    renderAge();
    renderDelay();
    renderSweep();
    renderNoise();
    $("runtime").textContent =
      `模拟 ${config.seconds}s · 去除前 ${Math.min(config.burnMs, config.seconds * 500) / 1000}s · 种子 ${config.seed} · 计算 ${fmt(performance.now() - before, 0)} ms`;
    window.lastSimulation = { config: { ...config }, result, reference };
  } catch (error) {
    $("error").hidden = false;
    $("error").textContent = error.message;
    console.error(error);
  }
}
$("controls").addEventListener("submit", (e) => e.preventDefault());
$("controls").addEventListener("input", (event) => {
  if (event.target.id === "hardware") {
    const pair = {
      base: [1, 1],
      compute: [2, 1],
      bandwidth: [1, 2],
      edge: [0.25, 0.2],
    }[event.target.value];
    if (pair) {
      config.compute = pair[0];
      config.bandwidth = pair[1];
      setControls();
    }
  } else if (["compute", "bandwidth"].includes(event.target.id))
    $("hardware").value = "custom";
  readControls();
  clearTimeout(timer);
  timer = setTimeout(run, 90);
});
$("reset").addEventListener("click", () => {
  config = { ...M.defaults };
  $("hardware").value = "base";
  setControls();
  run();
});
$("window").addEventListener("input", () => {
  renderTimeline();
  renderAge();
});
for (const id of ["alpha", "rho", "bias"])
  $(id).addEventListener("input", renderNoise);
document.querySelectorAll("[data-preset]").forEach((button) =>
  button.addEventListener("click", () => {
    config = { ...M.defaults };
    $("hardware").value = "base";
    if (button.dataset.preset === "stress") config.period = 20;
    if (button.dataset.preset === "limited")
      Object.assign(config, { period: 20, slowCap: 12, qV: "int8" });
    if (button.dataset.preset === "memory") {
      Object.assign(config, { bandwidth: 0.5, qV: "w4", ratioV: 0.3 });
      $("hardware").value = "custom";
    }
    if (button.dataset.preset === "aoi")
      Object.assign(config, {
        fastEnabled: false,
        source: "zero",
        jitter: 0,
        kA: 0,
        kV: 0,
        seconds: 12,
      });
    setControls();
    run();
  }),
);
$("export").addEventListener("click", () => {
  const payload = {
    schema: 1,
    kind: "mechanism_simulation_not_hardware_prediction",
    generatedAt: new Date().toISOString(),
    config,
    assumptions:
      "Phenomenological state-dependent slowdown; roofline inputs are editable hypotheses; residual fitted from separate runs; no quality calibration.",
    measuredAnchors: M.anchors,
    result,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    }),
    url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download =
    "vla-scenario-" + new Date().toISOString().replace(/[:.]/g, "-") + ".json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
const e = window.VLAEvidence;
$("bounds-table").innerHTML =
  "<table><thead><tr><th>范围</th><th>可量化工作占比</th><th>局部快 2×</th><th>局部快 4×</th><th>目标耗时 → 0</th></tr></thead><tbody>" +
  e.bounds
    .filter((r) => r.variant === "baseline" || r.role === "S2")
    .map(
      (r) =>
        `<tr><td>${r.role}${r.variant === "rope_rms" ? " · 融合后" : ""}</td><td>${fmt(r.eligible_fraction * 100)}%</td><td>${fmt(r.scenarios[0].gpu_work_speedup, 2)}×</td><td>${fmt(r.scenarios[1].gpu_work_speedup, 2)}×</td><td>${fmt(r.zero_target_cost_ceiling, 2)}×</td></tr>`,
    )
    .join("") +
  "</tbody></table>";
$("evidence-table").innerHTML =
  "<table><thead><tr><th>实现</th><th>S2 独占 P50</th><th>同卡配对 P50</th><th>数值检查</th></tr></thead><tbody>" +
  e.fusion
    .map(
      (row) =>
        `<tr><td>${row.label}</td><td>${fmt(row.s2, 2)} ms</td><td>${fmt(row.pair, 2)} ms</td><td>${row.exact ? "已测样本 max_abs=0" : "参考"}</td></tr>`,
    )
    .join("") +
  "</tbody></table>";
setControls();
run();
