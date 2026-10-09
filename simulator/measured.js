/* Offline measured trace viewer. Host intervals never imply GPU occupancy. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const ns = "http://www.w3.org/2000/svg";
  const colors = { S1: "var(--s1)", S2: "var(--s2)", state: "var(--state)" };
  const fmt = (x, digits = 1) =>
    x == null || !Number.isFinite(Number(x))
      ? "未知"
      : Number(x).toFixed(digits);
  const esc = (x) =>
    String(x ?? "未知").replace(
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
  const M = MeasuredReplay;
  let condition,
    episode,
    time = 0,
    playing = false,
    lastFrame = null,
    selectedFeature = null,
    selectedRequest = null;
  function element(tag, attrs = {}, text = null, parent = null) {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attrs))
      node.setAttribute(key, value);
    if (text !== null) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }
  function options(id, entries, value) {
    $(id).replaceChildren(
      ...entries.map(([key, label]) => {
        const option = document.createElement("option");
        option.value = key;
        option.textContent = label;
        return option;
      }),
    );
    if (value != null && entries.some(([key]) => key === value))
      $(id).value = value;
  }
  function card(title, value, detail, cls = "") {
    return `<div class="card ${cls}"><h3>${esc(title)}</h3><strong>${esc(value)}</strong><p>${esc(detail)}</p></div>`;
  }
  function limits() {
    const span = Number($("span").value);
    if (!span || span >= episode.duration_s) return [0, episode.duration_s];
    const a = Math.max(0, Math.min(time - span / 2, episode.duration_s - span));
    return [a, a + span];
  }
  function dependencies() {
    return M.dependencies(episode, ...limits());
  }
  function selectDependency(feature, request = null) {
    selectedFeature = feature;
    selectedRequest = request;
    setTime(time);
  }
  function axes(svg, width, top, height, a, b, max, label, unit = "") {
    const left = 105,
      right = width - 20;
    const x = (t) => left + ((t - a) / Math.max(1e-8, b - a)) * (right - left),
      y = (v) => top + height - (v / max) * height;
    for (let i = 0; i < 5; i++) {
      const t = a + ((b - a) * i) / 4;
      element(
        "line",
        { x1: x(t), x2: x(t), y1: top, y2: top + height, class: "grid" },
        null,
        svg,
      );
      element(
        "text",
        { x: x(t), y: top + height + 16, "text-anchor": "middle" },
        fmt(t, 2) + " s",
        svg,
      );
    }
    element("text", { x: 3, y: top + 15, "font-weight": "650" }, label, svg);
    element("text", { x: 3, y: top + 31 }, unit, svg);
    for (const value of [0, max / 2, max])
      element(
        "text",
        { x: left - 8, y: y(value) + 4, "text-anchor": "end" },
        fmt(value, max < 2 ? 2 : 0),
        svg,
      );
    return { x, y, left, right };
  }
  function curve(svg, points, transform, color, stepped = false) {
    let d = "",
      connected = false;
    for (const [t, v] of points) {
      if (v == null || !Number.isFinite(v)) {
        connected = false;
        continue;
      }
      d += !connected
        ? `M${transform.x(t)},${transform.y(v)}`
        : stepped
          ? `H${transform.x(t)}V${transform.y(v)}`
          : `L${transform.x(t)},${transform.y(v)}`;
      connected = true;
    }
    element(
      "path",
      { d, fill: "none", stroke: color, "stroke-width": 1.8 },
      null,
      svg,
    );
  }
  function drawTimeline() {
    const svg = $("timeline");
    svg.replaceChildren();
    const [a, b] = limits(),
      x = (t) => 105 + ((t - a) / (b - a)) * 1075,
      d = dependencies();
    const defs = element("defs", {}, null, svg),
      marker = element(
        "marker",
        {
          id: "dependency-arrow",
          viewBox: "0 0 10 10",
          refX: 9,
          refY: 5,
          markerWidth: 5,
          markerHeight: 5,
          orient: "auto-start-reverse",
        },
        null,
        defs,
      );
    element("path", { d: "M0 0L10 5L0 10Z", fill: "#7c7792" }, null, marker);
    for (const [label, y] of [
      ["S2 视觉", 22],
      ["缓存发布", 77],
      ["S1 DiT", 129],
      ["待接纳", 186],
    ]) {
      element("text", { x: 3, y: y + 17, "font-weight": 600 }, label, svg);
      element(
        "line",
        { x1: 105, x2: 1180, y1: y + 29, y2: y + 29, class: "grid" },
        null,
        svg,
      );
    }
    // Arrows use only recorded publications and actual feature sequence.
    for (const item of d.actions.values()) {
      const r = item.request;
      if (r.use_s == null || r.use_s < a || r.use_s > b || !item.parent)
        continue;
      const parent = item.parent,
        publication = episode.feature_updates.find(
          (f) => f.feature_sequence === r.feature_sequence,
        );
      const from = publication?.t ?? (parent.bootstrap ? 0 : null);
      if (from == null || from > r.use_s) continue;
      const active = selectedFeature === r.feature_sequence;
      const path = element(
        "path",
        {
          d: `M${x(Math.max(a, from))},94L${x(r.use_s)},128`,
          fill: "none",
          stroke: active ? "#954f19" : "#a2a7b7",
          "stroke-width": active ? 2.5 : 1,
          opacity: selectedFeature == null ? 0.55 : active ? 1 : 0.12,
          "marker-end": "url(#dependency-arrow)",
        },
        null,
        svg,
      );
      element(
        "title",
        {},
        `${parent.label} 发布 ${fmt(from, 4)} s → ${item.label} 使用 ${fmt(r.use_s, 4)} s；feature_sequence=${r.feature_sequence}`,
        path,
      );
      path.style.cursor = "pointer";
      path.onclick = () => selectDependency(r.feature_sequence, r.request_tick);
    }
    function bar(r, start, end, y, color, label, ready = false) {
      if (end < a || start > b) return;
      const xa = x(Math.max(a, start)),
        xb = x(Math.min(b, end)),
        active =
          selectedFeature != null && selectedFeature === r.feature_sequence;
      const rect = element(
        "rect",
        {
          x: xa,
          y,
          width: Math.max(1, xb - xa),
          height: 25,
          rx: 2,
          fill: color,
          stroke: active ? "#e0ae2c" : r.censored ? "#14263e" : "none",
          "stroke-width": active ? 3 : 1,
          "stroke-dasharray": r.censored ? "3 2" : "none",
          opacity: selectedFeature == null || active ? 1 : 0.3,
          "data-feature": r.feature_sequence ?? "unknown",
          "data-request": r.request_tick ?? "",
        },
        null,
        svg,
      );
      element(
        "title",
        {},
        `${label} · ${ready ? "计算完成、待接纳" : "主机 RPC"} ${fmt((end - start) * 1000)} ms；实际 feature_sequence=${r.feature_sequence ?? "未知"}；${r.censored ? "结束截尾" : "完整"}`,
        rect,
      );
      rect.style.cursor = "pointer";
      rect.onclick = () =>
        selectDependency(r.feature_sequence, r.request_tick ?? null);
      if (xb - xa > 58)
        element(
          "text",
          {
            x: xa + 3,
            y: y + 17,
            style: `fill:${ready ? "#214b71" : "white"};font-size:10px;pointer-events:none`,
          },
          label,
          svg,
        );
    }
    for (const r of episode.requests) {
      const label =
        r.role === "S1"
          ? d.actions.get(r.request_tick)?.label
          : d.parents.get(r.feature_sequence)?.label;
      bar(
        r,
        r.start_s,
        r.end_s,
        r.role === "S1" ? 129 : 22,
        colors[r.role],
        label ?? "S2[?]",
      );
      if (
        r.role === "S1" &&
        r.completed_s != null &&
        r.completed_s <= episode.duration_s
      )
        bar(
          r,
          r.completed_s,
          r.adopted_s ?? episode.duration_s,
          186,
          "#c4dcf2",
          label,
          true,
        );
    }
    for (const f of episode.feature_updates.filter(
      (f) => f.t >= a && f.t <= b,
    )) {
      const parent = d.parents.get(f.feature_sequence);
      const node = element(
        "circle",
        {
          cx: x(f.t),
          cy: 94,
          r: selectedFeature === f.feature_sequence ? 5 : 3,
          fill: colors.S2,
        },
        null,
        svg,
      );
      element(
        "title",
        {},
        `${parent?.label ?? "S2[?]"} 缓存发布，采集 ${fmt(f.capture_s, 4)} s`,
        node,
      );
      node.style.cursor = "pointer";
      node.onclick = () => selectDependency(f.feature_sequence);
    }
    const historical = [...d.parents.values()].filter(
      (p) =>
        p.historical &&
        [...d.actions.values()].some(
          (i) =>
            i.parent === p && i.request.start_s >= a && i.request.start_s <= b,
        ),
    );
    if (historical.length)
      element(
        "text",
        { x: 108, y: 73, style: "font-size:10px" },
        "窗口前：" + historical.map((p) => p.label).join("、"),
        svg,
      );
    element(
      "line",
      { x1: x(time), x2: x(time), y1: 8, y2: 220, class: "cursor" },
      null,
      svg,
    );
    for (let i = 0; i < 5; i++)
      element(
        "text",
        { x: x(a + ((b - a) * i) / 4), y: 242, "text-anchor": "middle" },
        fmt(a + ((b - a) * i) / 4, 2) + " s",
        svg,
      );
    const item = d.actions.get(selectedRequest),
      parent = d.parents.get(selectedFeature);
    $("dependency-detail").textContent = item
      ? `${item.label} ← ${parent?.label ?? "未知"}；实际序列 ${item.request.feature_sequence ?? "未知"}；请求 tick ${item.request.request_tick}；图像采集 ${fmt(item.request.feature_capture_s, 4)} s；状态采集 ${fmt(item.request.state_capture_s, 4)} s；DiT 使用 ${fmt(item.request.use_s, 4)} s。`
      : parent
        ? `${parent.label}：feature_sequence=${selectedFeature}；实际连接 ${parent.children ?? 0} 个 DiT 请求。`
        : "点击请求或箭头追溯来源；缩小时间窗口可读完整编号。";
  }
  function drawQueues() {
    const svg = $("queues");
    svg.replaceChildren();
    const [a, b] = limits(),
      curves = M.aoiSeries(episode, a, b);
    const ageMax =
      Math.max(
        50,
        ...[...curves.cache, ...curves.image, ...curves.state].map(
          (p) => p[1] ?? 0,
        ),
        ...curves.used.flatMap((p) => [p.imageMs ?? 0, p.stateMs ?? 0]),
      ) * 1.05;
    const transform = axes(
      svg,
      1200,
      15,
      135,
      a,
      b,
      ageMax,
      "信息年龄",
      "AoI / ms",
    );
    curve(svg, curves.cache, transform, colors.S2);
    curve(svg, curves.image, transform, colors.S1);
    curve(svg, curves.state, transform, colors.state);
    for (const p of curves.used)
      for (const [key, color] of [
        ["imageMs", colors.S2],
        ["stateMs", colors.state],
      ]) {
        if (p[key] == null) continue;
        const attrs =
          key === "imageMs"
            ? { cx: transform.x(p.t), cy: transform.y(p[key]), r: 2.5 }
            : {
                x: transform.x(p.t) - 2.5,
                y: transform.y(p[key]) - 2.5,
                width: 5,
                height: 5,
                transform: `rotate(45 ${transform.x(p.t)} ${transform.y(p[key])})`,
              };
        const node = element(
          key === "imageMs" ? "circle" : "rect",
          { ...attrs, fill: color },
          null,
          svg,
        );
        element(
          "title",
          {},
          `请求 tick ${p.request_tick} 的 DiT 使用时${key === "imageMs" ? "图像" : "状态"}年龄 ${fmt(p[key])} ms`,
          node,
        );
      }
    const ticks = episode.ticks.filter((r) => r.t >= a && r.t <= b);
    const panels = [
      {
        top: 207,
        max: Math.max(5, ...ticks.map((r) => r.clean_slots)),
        label: "动作缓冲",
        unit: "非 fallback 槽",
        series: [
          [ticks.map((r) => [r.t, r.clean_slots]), "#91b3cf"],
          [ticks.map((r) => [r.t, r.fresh_slots]), colors.S1],
        ],
      },
      {
        top: 315,
        max: Math.max(3, ...ticks.map((r) => r.d ?? 0)),
        label: "延迟 d",
        unit: "控制 tick",
        series: [
          [
            [
              [a, M.stateAt(episode, a).delay],
              ...episode.delay_transitions
                .filter((r) => r.t >= a && r.t <= b)
                .map((r) => [r.t, r.after]),
              [b, M.stateAt(episode, b).delay],
            ],
            "#7f3c8d",
          ],
        ],
      },
      {
        top: 423,
        max: Math.max(6, ...ticks.map((r) => r.lateness_ms)),
        label: "控制迟到",
        unit: "ms；×请求超时",
        series: [[ticks.map((r) => [r.t, r.lateness_ms]), "#596c84"]],
      },
    ];
    for (const panel of panels) {
      const tr = axes(
        svg,
        1200,
        panel.top,
        50,
        a,
        b,
        panel.max,
        panel.label,
        panel.unit,
      );
      for (const [points, c] of panel.series) curve(svg, points, tr, c, true);
      if (panel.top === 423)
        for (const miss of episode.miss_events.filter(
          (r) => r.t >= a && r.t <= b,
        ))
          element(
            "path",
            {
              d: `M${tr.x(miss.t) - 4},${tr.y(0) - 8}l8,8m-8,0l8,-8`,
              stroke: "#b72f3b",
              "stroke-width": 2,
            },
            null,
            svg,
          );
    }
    element(
      "line",
      {
        x1: transform.x(time),
        x2: transform.x(time),
        y1: 15,
        y2: 473,
        class: "cursor",
      },
      null,
      svg,
    );
  }
  function renderState() {
    const s = M.stateAt(episode, time),
      d = dependencies(),
      tick = s.tick;
    const label = (r) => d.actions.get(r.request_tick)?.label ?? "S1[?]";
    $("state").innerHTML =
      card(
        "相机等待槽",
        s.camera.waiting_tick == null
          ? "空"
          : `相机帧 t${s.camera.waiting_tick}`,
        "新帧替换尚未开始的等待帧",
        "s2",
      ) +
      card(
        "S2 正在处理",
        s.camera.inflight_tick == null
          ? "空闲"
          : `相机帧 t${s.camera.inflight_tick}`,
        "相机队列按当前时间的真实快照更新",
        "s2",
      ) +
      card(
        "最新可用缓存",
        d.parents.get(s.feature.feature_sequence)?.label ?? "未知",
        `图像 AoI ${fmt(s.cacheAgeMs)} ms；序列 ${s.feature.feature_sequence ?? "未知"}`,
        "s2",
      ) +
      card(
        "S1 当前请求",
        s.inflight.length
          ? s.inflight.map(label).join("、")
          : s.ready.length
            ? "等待接纳"
            : "空闲",
        `计算中 ${s.inflight.length}；已完成待接纳 ${s.ready.length}；当前 d=${s.delay ?? "未知"}`,
        "s1",
      ) +
      card(
        "实际正在执行",
        tick
          ? `t${tick.tick} · ${tick.fallback ? "fallback" : s.executedRequest ? label(s.executedRequest) : tick.executed_slot?.producer_kind === "bootstrap" ? "启动动作" : "来源未知"}`
          : "尚未执行",
        `图像 AoI ${fmt(s.imageAgeMs)} ms；状态 AoI ${fmt(s.stateAgeMs)} ms`,
      );
    $("time-label").textContent =
      `${fmt(time, 3)} / ${fmt(episode.duration_s, 3)} s`;
    $("buffer-label").textContent = tick
      ? `快照 t=${fmt(tick.t, 4)} s · 已执行 tick ${tick.tick} · after_execute`
      : "首个执行 tick 之前";
    $("buffer").innerHTML = tick
      ? (tick.action_buffer.slots ?? [])
          .map((slot) => {
            const producer = d.actions.get(slot.producer_request_tick);
            return `<div class="slot ${slot.committed ? "committed" : ""} ${slot.producer_kind === "bootstrap" ? "bootstrap" : ""} ${slot.fallback ? "fallback" : ""} ${selectedRequest != null && selectedRequest === slot.producer_request_tick ? "selected-dependency" : ""}" title="${esc(JSON.stringify(slot))}">t${slot.tick}<small>${slot.producer_kind === "request" ? esc(producer?.label ?? "来源未知") : slot.producer_kind === "bootstrap" ? "启动动作" : slot.fallback ? "fallback" : "未知"}</small></div>`;
          })
          .join("")
      : '<p class="empty">尚无执行后的动作缓冲快照。</p>';
  }
  function setTime(value) {
    time = Math.max(0, Math.min(episode.duration_s, value));
    $("time").value = time;
    renderState();
    drawTimeline();
    drawQueues();
  }
  function drawScatter() {
    const svg = $("scatter");
    svg.replaceChildren();
    const xkey = $("x-axis").value,
      ykey = $("y-axis").value;
    const rows = REPORT.conditions
      .map((c) => c.summary)
      .filter((s) => s[xkey] != null && s[ykey] != null);
    if (!rows.length) {
      element("text", { x: 30, y: 60 }, "该坐标轴暂无实测值", svg);
      return;
    }
    let xmin = Math.min(...rows.map((r) => r[xkey])),
      xmax = Math.max(...rows.map((r) => r[xkey])),
      ymin = Math.min(...rows.map((r) => r[ykey])),
      ymax = Math.max(...rows.map((r) => r[ykey]));
    const dx = xmax - xmin || Math.max(1, xmax * 0.15),
      dy = ymax - ymin || Math.max(0.01, ymax * 0.15);
    xmin -= dx * 0.1;
    xmax += dx * 0.17;
    ymin = Math.max(0, ymin - dy * 0.1);
    ymax += dy * 0.2;
    const x = (v) => 70 + ((v - xmin) / (xmax - xmin)) * 650,
      y = (v) => 315 - ((v - ymin) / (ymax - ymin)) * 270;
    for (let i = 0; i < 5; i++) {
      const xx = xmin + ((xmax - xmin) * i) / 4,
        yy = ymin + ((ymax - ymin) * i) / 4;
      element(
        "line",
        { x1: x(xx), x2: x(xx), y1: 35, y2: 315, class: "grid" },
        null,
        svg,
      );
      element(
        "text",
        { x: x(xx), y: 337, "text-anchor": "middle" },
        fmt(xx, xmax < 2 ? 2 : 1),
        svg,
      );
      element(
        "line",
        { x1: 70, x2: 720, y1: y(yy), y2: y(yy), class: "grid" },
        null,
        svg,
      );
      element(
        "text",
        { x: 60, y: y(yy) + 4, "text-anchor": "end" },
        fmt(yy, ymax < 2 ? 2 : 1),
        svg,
      );
    }
    for (const row of rows) {
      const selected =
        row.layout === condition.layout &&
        row.condition === condition.condition;
      const node = element(
        row.layout === "single" ? "circle" : "rect",
        row.layout === "single"
          ? { cx: x(row[xkey]), cy: y(row[ykey]), r: selected ? 7 : 5 }
          : { x: x(row[xkey]) - 5, y: y(row[ykey]) - 5, width: 10, height: 10 },
        null,
        svg,
      );
      node.setAttribute(
        "fill",
        row.layout === "single" ? colors.S1 : colors.S2,
      );
      if (selected) {
        node.setAttribute("stroke", "#14263e");
        node.setAttribute("stroke-width", "2");
      }
      element(
        "title",
        {},
        `${row.layout}/${row.condition}: ${fmt(row[xkey], 3)}, ${fmt(row[ykey], 3)}`,
        node,
      );
      node.style.cursor = "pointer";
      node.addEventListener("click", () =>
        selectCondition(row.layout, row.condition),
      );
      element(
        "text",
        { x: x(row[xkey]) + 8, y: y(row[ykey]) - 8, style: "font-size:9px" },
        `${row.layout[0]}:${row.condition}`,
        svg,
      );
    }
    element(
      "text",
      { x: 395, y: 363, "text-anchor": "middle" },
      $("x-axis").selectedOptions[0].textContent,
      svg,
    );
    element(
      "text",
      { x: 70, y: 20 },
      $("y-axis").selectedOptions[0].textContent +
        " · 蓝圆点：单卡 / 橙方块：双卡",
      svg,
    );
  }
  function renderCondition() {
    const s = condition.summary;
    $("condition-kpis").innerHTML =
      card(
        "独占 S1 / S2 均值",
        `${fmt(s.s1_solo_mean_ms)} / ${fmt(s.s2_solo_mean_ms)} ms`,
        "仅作为模型无干扰服务时间的校准",
        "s1",
      ) +
      card(
        "并发运行 S1 / S2 p95",
        `${fmt(s.s1_rpc_p95_ms)} / ${fmt(s.s2_rpc_p95_ms)} ms`,
        "完整、窗口内 RPC；未完成请求截尾",
        "s2",
      ) +
      card(
        "执行图像 / 状态 AoI p95",
        `${fmt(s.executed_image_age_ms_p95)} / ${fmt(s.executed_state_age_ms_p95)} ms`,
        `DiT 使用图像 p95 ${fmt(s.used_cache_age_p95)} ms；按执行 tick 采样`,
      ) +
      card(
        "等待帧替换 / 平均 d",
        `${fmt(s.frame_replacement_fraction * 100)}% / ${fmt(s.d_mean, 2)}`,
        `${s.camera_replaced} / ${s.camera_offered} 帧被替换`,
      ) +
      card(
        "板卡能耗 / 平均功率",
        `${fmt(s.sampled_energy_j)} J / ${fmt(s.average_power_w)} W`,
        `遥测覆盖 ${fmt(s.energy_coverage * 100)}%；含同卡渲染`,
      );
    const roles = condition.hardware.role_gpus ?? {
      S1: s.s1_gpu,
      S2: s.s2_gpu,
    };
    $("hardware").innerHTML =
      `<p class="note">实际角色：S1 → GPU ${esc(roles.S1)}；S2 → GPU ${esc(roles.S2)}；渲染 → GPU ${esc(condition.hardware.renderer_gpu)}。申请条件 ${esc(condition.condition)}；以下为各角色真实遥测均值。</p><table><thead><tr><th>角色 / 设备</th><th>实际 SM MHz</th><th>实测板卡功率 W</th><th>功率上限 W</th></tr></thead><tbody>${[
        ["S1", "s1"],
        ["S2", "s2"],
      ]
        .map(
          ([role, key]) =>
            `<tr><td>${role} / GPU ${esc(roles[role])}</td><td>${fmt(s[key + "_effective_sm_mhz"])}</td><td>${fmt(s[key + "_board_power_w"])}</td><td>${fmt(s[key + "_power_limit_w"])}</td></tr>`,
        )
        .join("")}</tbody></table>`;
    $("hardware-json").textContent = JSON.stringify(
      condition.hardware,
      null,
      2,
    );
    const columns = [
      ["layout", "部署"],
      ["condition", "条件"],
      ["episodes", "回合"],
      ["control_seconds", "暴露秒数"],
      ["s1_rpc_p95_ms", "S1 RPC p95 ms"],
      ["s2_rpc_p95_ms", "S2 RPC p95 ms"],
      ["used_cache_age_p95", "DiT 图像 AoI p95 ms"],
      ["executed_image_age_ms_p95", "执行图像 AoI p95 ms"],
      ["executed_state_age_ms_p95", "执行状态 AoI p95 ms"],
      ["frame_replacement_fraction", "帧替换比例"],
      ["d_mean", "平均 d"],
      ["fallback_fraction", "fallback 比例"],
      ["expired_slot_fraction", "过期 / 应到槽"],
      ["sampled_energy_j", "板卡 J"],
    ];
    const table = (rows) =>
      `<thead><tr>${columns.map(([, l]) => `<th>${l}</th>`).join("")}</tr></thead><tbody>${rows.map((c) => `<tr class="${c === condition ? "selected" : ""}">${columns.map(([k]) => `<td>${typeof c.summary[k] === "number" ? fmt(c.summary[k], k.includes("fraction") ? 3 : 1) : esc(c.summary[k])}</td>`).join("")}</tr>`).join("")}</tbody>`;
    $("comparison").innerHTML = table(REPORT.conditions);
    const paired = REPORT.conditions.filter(
      (c) => c.condition === condition.condition,
    );
    $("paired-comparison").innerHTML =
      `<h3>同名条件对照（实际角色频率见上表，回合成功提前终止导致暴露时长不同）</h3><table>${table(paired)}</table>`;
    $("counterpart").disabled = !paired.some(
      (c) => c.layout !== condition.layout,
    );
    drawScatter();
  }
  function renderDelay() {
    const byRequest = new Map(
      episode.delay_transitions.map((r) => [r.request_tick, r]),
    );
    $("delay-transitions").innerHTML =
      "<thead><tr><th>请求 tick</th><th>截止 s</th><th>完成越界 ms</th><th>控制器消费 s</th><th>d 变化</th><th>所需 tick（含 5ms）</th></tr></thead><tbody>" +
      episode.miss_events
        .map((r) => {
          const tr = byRequest.get(r.request_tick);
          return `<tr><td>${r.request_tick}</td><td>${fmt(r.t, 4)}</td><td>${fmt(r.overshoot_ms, 3)}</td><td>${fmt(tr?.t, 4)}</td><td>${tr ? `${tr.before} → ${tr.after}` : "回合内未消费"}</td><td>${tr?.required_ticks ?? "未知"}</td></tr>`;
        })
        .join("") +
      "</tbody>";
    const tiny = condition.episodes
      .flatMap((e) => e.delay_transitions.map((r) => ({ ...r, episode: e.id })))
      .filter(
        (r) =>
          r.overshoot_ms != null && r.overshoot_ms < 1 && r.after > r.before,
      );
    $("delay-explanation").textContent = tiny.length
      ? `本条件出现小于 1 ms 的越界：${tiny.map((r) => `${r.episode} 超过 ${fmt(r.overshoot_ms, 3)} ms，d ${r.before}→${r.after}`).join("；")}。它们触发只增不减规则，后续 d 保持偏高不能单独当成硬件算力证据。`
      : episode.miss_events.length
        ? "下表包含每次请求超时；有超时不一定增加 d，且排空阶段未消费的结果不会反写控制期 d。"
        : "本回合无请求截止超时。";
  }
  function selectEpisode() {
    episode =
      condition.episodes.find((e) => e.id === $("episode").value) ||
      condition.episodes[0];
    $("time").max = episode.duration_s;
    playing = false;
    $("play").textContent = "播放";
    selectedFeature = null;
    selectedRequest = null;
    const m = episode.metadata;
    $("exposure").textContent =
      `${episode.id} · ${episode.metrics.ticks} 控制 tick / ${fmt(episode.duration_s, 3)} 秒 · 终止 ${episode.metrics.termination_reason ?? "未知"} · 严格控制时序有效 ${m.control_rate_valid ?? "未知"} · seed ${m.seed ?? "未知"}。仅选定回合的闭环记录，不能外推完整套件成功率。`;
    renderDelay();
    setTime(0);
  }
  function selectCondition(layoutName, conditionName) {
    const previousEpisode = episode?.id;
    if (layoutName) $("layout").value = layoutName;
    const list = REPORT.conditions.filter(
      (c) => c.layout === $("layout").value,
    );
    options(
      "condition",
      list.map((c) => [c.condition, c.condition]),
      conditionName,
    );
    condition =
      list.find((c) => c.condition === $("condition").value) || list[0];
    options(
      "episode",
      condition.episodes.map((e) => [
        e.id,
        `${e.id} · ${fmt(e.duration_s, 2)} 秒`,
      ]),
      previousEpisode,
    );
    selectEpisode();
    renderCondition();
  }
  function exportCalibration() {
    const message = M.calibrationMessage(condition, episode);
    if (!(message.solo.actionMs > 0 && message.solo.vlmMs > 0)) {
      $("calibration-status").textContent = " 缺少完整独占校准，无法建立情景。";
      return;
    }
    if (window.parent !== window) {
      window.parent.postMessage(
        message,
        location.origin === "null" ? "*" : location.origin,
      );
      $("calibration-status").textContent =
        " 已发送独占校准；共享干扰与抖动仍待设定。";
    } else {
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(message, null, 2)], {
          type: "application/json",
        }),
      );
      const link = document.createElement("a");
      link.href = url;
      link.download = "pir2-solo-calibration.json";
      link.click();
      URL.revokeObjectURL(url);
      $("calibration-status").textContent =
        " 已导出独占校准 JSON；可在假设情景中使用。";
    }
  }
  function frame(now) {
    if (playing) {
      if (lastFrame != null)
        setTime(time + ((now - lastFrame) / 1000) * Number($("speed").value));
      if (time >= episode.duration_s) {
        playing = false;
        $("play").textContent = "播放";
      }
    }
    lastFrame = now;
    requestAnimationFrame(frame);
  }
  try {
    if (new URLSearchParams(location.search).get("embedded") === "1")
      document.body.classList.add("embedded");
    options(
      "layout",
      [...new Set(REPORT.conditions.map((c) => c.layout))].map((x) => [
        x,
        x === "single" ? "单卡 S1+S2" : "双卡 S1 / S2",
      ]),
    );
    $("generated").textContent =
      `${REPORT.conditions.length} 个正式条件 · ${REPORT.conditions.reduce((n, c) => n + c.episodes.length, 0)} 个回合 · 离线快照`;
    $("methods").innerHTML = Object.entries(REPORT.method)
      .map(([k, v]) => `<p><b>${esc(k)}：</b>${esc(v)}</p>`)
      .join("");
    $("pairing").innerHTML = REPORT.paired_initial_states
      .map(
        (r) =>
          `<p class="note">任务 ${r.task_id} / 回合 ${r.episode_id} / seed ${r.seed}，${r.observations} 条件：${Object.entries(
            r.equal,
          )
            .map(([k, v]) => `${esc(k)} = ${v ? "一致" : "不同或未建立证据"}`)
            .join("；")}</p>`,
      )
      .join("");
    $("footer").textContent =
      `生成于 ${REPORT.generated_utc}。自包含 HTML；完整数据与汇总另存 report-data.json、summary.json、summary.csv。排除 ${REPORT.skipped_diagnostics.length} 个诊断条件。`;
    $("layout").onchange = () => selectCondition();
    $("condition").onchange = () =>
      selectCondition($("layout").value, $("condition").value);
    $("episode").onchange = selectEpisode;
    $("counterpart").onclick = () => {
      const c = REPORT.conditions.find(
        (c) =>
          c.condition === condition.condition && c.layout !== condition.layout,
      );
      if (c) selectCondition(c.layout, c.condition);
    };
    $("time").oninput = (e) => setTime(Number(e.target.value));
    $("span").onchange = () => setTime(time);
    $("play").onclick = () => {
      if (time >= episode.duration_s) setTime(0);
      playing = !playing;
      lastFrame = null;
      $("play").textContent = playing ? "暂停" : "播放";
    };
    for (const id of ["x-axis", "y-axis"]) $(id).onchange = drawScatter;
    $("load-calibration").onclick = exportCalibration;
    selectCondition();
    requestAnimationFrame(frame);
  } catch (error) {
    $("error").textContent = error.stack;
  }
})();
