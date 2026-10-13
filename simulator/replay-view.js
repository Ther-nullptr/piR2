/* Browser presentation over explicit evidence; the pure model owns time semantics. */
"use strict";
(() => {
  const R = window.QuantReplay,
    Q = window.MeasuredReplay,
    $ = (id) => document.getElementById(id);
  const colors = { bf16: "#315b8a", w8a8: "#087b6d", w4a4: "#a45b08" };
  const categories = {
    gemm: "#466c96",
    attention: "#538d81",
    quantization_prepare: "#c38c3b",
    producer_quantization: "#b29365",
    normalization_modulation: "#889f71",
    conversion_copy_ambiguous: "#b3a5be",
    reduction: "#8d9a70",
    convolution: "#658195",
    random: "#cab699",
    pointwise: "#b27b81",
    layout_copy: "#8c86ac",
    transfer: "#6e9ca7",
    unknown: "#a4aba5",
  };
  const state = {
    catalog: null,
    descriptors: [],
    runs: [],
    kind: "model",
    type: "clean",
    scope: "prepared_model",
    precision: "all",
    selection: new Map(),
    sample: null,
    rows: [],
    clock: 0,
    end: 1,
    playing: false,
    lastWall: null,
    events: [],
    hits: [],
    selected: null,
    mediaEpoch: 0,
    loading: false,
  };
  const loader = R.latestLoader();
  const esc = (v) =>
    String(v ?? "").replace(
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
  const fmt = (v, n = 2) => (R.finite(v) ? v.toFixed(n) : "—");
  const label = (p) => String(p ?? "unknown").toUpperCase();
  const options = (el, rows, value) => {
    el.innerHTML = rows
      .map(([v, t]) => `<option value="${esc(v)}">${esc(t)}</option>`)
      .join("");
    if (rows.some(([v]) => String(v) === String(value))) el.value = value;
  };
  const color = (p) => colors[p] ?? "#78669d";
  const notice = (value, bad = false) => {
    $("data-status").textContent = value;
    $("data-status").dataset.state = bad ? "error" : "ready";
  };
  function stop() {
    state.playing = false;
    state.lastWall = null;
    $("play").textContent =
      state.kind === "model" ? "播放模型时间线" : "播放机器人视频";
  }
  async function json(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error(`无法加载 HTTP ${response.status}`);
    return response.json();
  }
  function visibleRuns() {
    return state.runs.filter(
      (r) => state.precision === "all" || r.precision === state.precision,
    );
  }
  const secondaryControls = [
    "record-type",
    "scope",
    "sample",
    "precision",
    "camera",
    "stage",
    "category",
    "channel",
    "window",
    "seek",
    "play",
    "previous",
    "next",
    "download",
    "inspect-run",
  ];
  function loadingControls(loading) {
    state.loading = loading;
    for (const id of secondaryControls) $(id).disabled = loading;
    for (const kind of ["model", "deployment"])
      $("mode-" + kind).setAttribute(
        "aria-pressed",
        String(kind === state.kind),
      );
  }
  async function loadRuns() {
    stop();
    state.mediaEpoch++;
    state.rows = [];
    state.runs = [];
    loadingControls(true);
    state.events = [];
    state.selected = null;
    $("precision-cards").innerHTML = '<p class="loading">正在切换实验记录…</p>';
    $("timeline").replaceChildren();
    $("lineage").replaceChildren();
    $("curve").replaceChildren();
    $("projection").replaceChildren();
    $("event-list").replaceChildren();
    $("inspector").textContent = "加载中…";
    $("data-status").textContent = "加载记录…";
    $("data-status").dataset.state = "loading";
    const kind = state.kind;
    const descriptors = state.descriptors.filter((d) => d.kind === kind),
      precisions = [...new Set(descriptors.map((d) => d.precision))].sort(
        (a, b) =>
          ["bf16", "w8a8", "w4a4"].indexOf(a) -
          ["bf16", "w8a8", "w4a4"].indexOf(b),
      );
    const selected = precisions.map(
      (p) =>
        descriptors.find((d) => d.id === state.selection.get(kind + ":" + p)) ??
        descriptors.find((d) => d.precision === p),
    );
    const result = await loader.load(() =>
      Promise.all(
        selected.map(async (d) => {
          try {
            return {
              ...(await json("/api/replay/run/" + encodeURIComponent(d.id))),
              descriptor: d,
            };
          } catch (error) {
            return {
              ...d,
              error: error.message,
              clean_requests: [],
              profile_requests: [],
              episodes: [],
            };
          }
        }),
      ),
    );
    if (!result.current) return;
    if (result.error) {
      loadingControls(false);
      notice(result.error.message, true);
      return;
    }
    state.runs = result.value.map((run) => ({
      ...run,
      episodes: Array.isArray(run.episodes)
        ? run.episodes.map(R.normalizeEpisode)
        : [],
    }));
    loadingControls(false);
    state.runs.forEach((r) =>
      state.selection.set(kind + ":" + r.precision, r.id),
    );
    options(
      $("precision"),
      [["all", "同步比较"], ...precisions.map((p) => [p, label(p)])],
      state.precision,
    );
    state.precision = $("precision").value;
    configureSamples();
    notice(`${state.runs.length} 个精度 · 本地记录已连接`);
  }
  function configureSamples() {
    if (state.loading) return;
    const previous = state.sample;
    const runs = visibleRuns();
    let samples;
    if (state.kind === "model") {
      const availableScopes = [
        ...new Set(
          runs.flatMap((run) =>
            R.records(run, state.type).map((record) => R.scopeOf(run, record)),
          ),
        ),
      ];
      const preferred = availableScopes.includes(state.scope)
        ? state.scope
        : availableScopes.includes("prepared_model")
          ? "prepared_model"
          : availableScopes[0];
      options(
        $("scope"),
        availableScopes.map((scope) => [scope, R.scopeInfo(scope).label]),
        preferred,
      );
      state.scope = $("scope").value || "unknown";
      $("record-type").querySelector('option[value="clean"]').textContent =
        runs.some((run) =>
          R.records(run, "clean").some((record) => record.clean_wall !== true),
        )
          ? "计时记录 · 含插桩状态未知"
          : "Clean · 耗时与动作";
      const unique = new Map();
      for (const run of runs)
        for (const record of R.scopedRecords(run, state.type, state.scope)) {
          const key = state.scope + ":" + (record.comparison_key ?? record.id);
          if (!unique.has(key)) unique.set(key, record);
        }
      samples = [...unique.values()].sort(
        (a, b) =>
          (a.observation ?? 0) - (b.observation ?? 0) ||
          (a.round ?? 0) - (b.round ?? 0),
      );
      options(
        $("sample"),
        samples.map((r) => [
          r.id,
          `${label(runs.find((run) => R.records(run, state.type).includes(r))?.precision)} · ${state.scope} · Obs ${r.observation ?? "?"} · seed ${r.seed ?? "?"} · round ${r.round ?? "?"}`,
        ]),
        previous?.id,
      );
      state.sample = samples.find((r) => r.id === $("sample").value) ?? null;
    } else {
      const unique = new Map();
      for (const run of runs)
        for (const episode of run.episodes ?? []) {
          const key = [episode.task_id, episode.episode_id, episode.seed].join(
            "/",
          );
          if (!unique.has(key)) unique.set(key, episode);
        }
      samples = [...unique.entries()];
      options(
        $("sample"),
        samples.map(([key, e]) => [
          key,
          `Task ${e.task_id} · episode ${e.episode_id} · seed ${e.seed}`,
        ]),
        previous?.task_id != null
          ? [previous.task_id, previous.episode_id, previous.seed].join("/")
          : null,
      );
      state.sample = unique.get($("sample").value) ?? null;
    }
    $("record-label").hidden = state.kind !== "model";
    $("scope-label").hidden = state.kind !== "model";
    $("sample-label").textContent =
      state.kind === "model" ? "输入 / 随机种子" : "任务 / 回合";
    const modelChannels = [
      "x",
      "y",
      "z",
      "roll",
      "pitch",
      "yaw",
      "gripper",
    ].map((k) => [k, k]);
    const robotChannels = [
      ["x", "位置 X · m"],
      ["y", "位置 Y · m"],
      ["z", "位置 Z · m"],
      ["rx", "旋转向量 X · rad"],
      ["ry", "旋转向量 Y · rad"],
      ["rz", "旋转向量 Z · rad"],
      ["g0", "夹爪关节 0"],
      ["g1", "夹爪关节 1"],
      ...["x", "y", "z", "rx", "ry", "rz", "gripper"].map((k, i) => [
        "command:" + i,
        "命令 " + k + " · normalized",
      ]),
    ];
    options(
      $("channel"),
      state.kind === "model" ? modelChannels : robotChannels,
      $("channel").value,
    );
    selectSample();
  }
  function selectSample() {
    if (state.loading) return;
    stop();
    state.selected = null;
    state.mediaEpoch++;
    const runs = visibleRuns();
    if (state.kind === "model") {
      state.sample =
        runs
          .flatMap((r) => R.scopedRecords(r, state.type, state.scope))
          .find((r) => r.id === $("sample").value) ?? null;
      const owner = runs.find((run) =>
        R.records(run, state.type).includes(state.sample),
      );
      state.rows = R.alignModel(runs, state.type, state.sample, owner?.id);
      state.end = Math.max(
        1,
        ...state.rows.map(
          ({ record }) => record?.host_wall_ms ?? record?.wall_ms ?? 0,
        ),
      );
      $("evidence-note").textContent =
        R.scopeInfo(state.scope).note +
        (state.type === "profile"
          ? " 实际 Profile 请求有独立 seed。host wall、device span 与 GPU 活动合计分别记录；未保留输出保持未知。Clean P50/P95 只取同一边界的未插桩记录。"
          : " Clean P50/P95 只统计当前边界，跨精度必须同边界、同输入、同 seed、已知 clock cohort 与 comparison key；本运行未配对的记录仍可单独检查。");
    } else {
      state.rows = runs.map((run) => ({
        run,
        episode:
          (run.episodes ?? []).find(
            (e) =>
              [e.task_id, e.episode_id, e.seed].join("/") === $("sample").value,
          ) ?? null,
      }));
      state.end = Math.max(
        1,
        ...state.rows.map(({ episode }) => R.episodeEnd(episode) ?? 0),
      );
      const paired = state.rows.every(
        (r) => r.episode && R.pairedEpisode(state.rows[0].episode, r.episode),
      );
      $("evidence-note").textContent =
        `真实闭环 · ${paired ? "已验证相同初始场景" : "初始场景配对未验证"}。不同精度使用各自观测、动作与机器人轨迹，以实际控制墙钟对齐。当前为单个调试示例，不是套件成功率；记录相机拷贝会带来开销。`;
    }
    const canPlay =
      state.kind === "model"
        ? state.rows.some((row) => row.record)
        : state.rows.some((row) => R.playableEpisode(row.episode));
    for (const id of ["play", "seek", "previous", "next"])
      $(id).disabled = !canPlay;
    state.clock = Math.min(state.clock, state.end);
    $("seek").max = state.end;
    state.events = buildEvents();
    options(
      $("category"),
      [
        ["all", "全部类别"],
        ...[...new Set(state.events.map((e) => e.category).filter(Boolean))]
          .sort()
          .map((k) => [k, k]),
      ],
      $("category").value,
    );
    renderCards();
    $("inspector").textContent =
      "点击内核、请求或控制动作，查看来源与实际时间戳。";
    render();
  }
  function buildEvents() {
    const out = [];
    for (const { run, record, episode } of state.rows) {
      if (state.kind === "model" && record) {
        if (state.type === "profile")
          for (const [i, event] of (record.timeline ?? []).entries())
            out.push({
              ...event,
              id: record.id + "/kernel/" + i,
              precision: run.precision,
              run,
              record,
              kind: "kernel",
              time: event.start_ms,
              end: event.start_ms + event.duration_ms,
              name: run.kernels?.[event.kernel_id] ?? "Unknown kernel",
            });
        else
          out.push({
            id: record.id,
            precision: run.precision,
            run,
            record,
            kind: "request",
            time: 0,
            end: record.wall_ms ?? 0,
            stage: "all",
            name: `${R.scopeInfo(state.scope).label} · ${record.clean_wall === true ? "Clean wall" : "Recorded wall（插桩未验证）"} · 无内核时间线`,
          });
      } else if (episode) {
        for (const request of episode.queue?.requests ?? [])
          out.push({
            id: request.id,
            precision: run.precision,
            run,
            episode,
            request,
            kind: "rpc",
            time: request.start_s,
            end: request.end_s ?? request.completed_s ?? request.start_s,
            stage: request.role === "S2" ? "vlm" : "action",
            name: `${request.role} request ${request.request_tick ?? request.source_tick ?? request.feature_source_tick ?? "?"}`,
          });
        for (const tick of episode.ticks ?? [])
          out.push({
            id: tick.id,
            precision: run.precision,
            run,
            episode,
            tick,
            kind: "command",
            time: tick.relative_s,
            end: tick.relative_s,
            stage: "control",
            name: `执行动作 · tick ${tick.tick}`,
          });
      }
    }
    return out.sort(
      (a, b) => a.time - b.time || a.precision.localeCompare(b.precision),
    );
  }
  function renderCards() {
    $("precision-cards").style.gridTemplateColumns =
      state.rows.length === 1 ? "minmax(0,1fr)" : "";
    $("precision-cards").innerHTML =
      state.rows
        .map(
          ({ run }, i) =>
            `<article class="precision-card" data-precision="${esc(run.precision)}" data-boundary="${esc(state.kind === "model" ? state.scope : "deployment")}" style="--precision:${color(run.precision)}"><div class="card-heading"><h2>${esc(label(run.precision))}</h2><select class="run-select" data-index="${i}" aria-label="${esc(label(run.precision))} 运行配置">${state.descriptors
              .filter(
                (d) => d.kind === state.kind && d.precision === run.precision,
              )
              .map(
                (d) =>
                  `<option value="${esc(d.id)}" ${d.id === run.id ? "selected" : ""}>${esc(d.mode ?? d.label ?? d.id)}</option>`,
              )
              .join(
                "",
              )}</select></div><div class="media-stage"><span class="media-message">等待实际观测</span><div class="frame-meta"></div></div><div class="card-status"></div><div class="metrics"></div><p class="card-details"></p><button class="card-action" data-index="${i}">检查当前动作来源</button><a class="run-source" href="/api/replay/run/${encodeURIComponent(run.id)}" target="_blank" rel="noopener">Run JSON ↗</a></article>`,
        )
        .join("") ||
      '<p class="loading">此证据类型没有记录。已配置的其他视图仍可使用。</p>';
    for (const el of document.querySelectorAll(".run-select"))
      el.addEventListener("change", () => {
        const row = state.rows[Number(el.dataset.index)];
        state.selection.set(state.kind + ":" + row.run.precision, el.value);
        loadRuns();
      });
    for (const el of document.querySelectorAll(".card-action"))
      el.addEventListener("click", () => {
        const row = state.rows[Number(el.dataset.index)];
        if (state.kind === "deployment") {
          const tick = R.robotAt(
            row.episode,
            state.clock,
            $("camera").value,
          ).tick;
          inspect(
            state.events.find((e) => e.tick?.id === tick?.id) ?? {
              run: row.run,
              episode: row.episode,
            },
          );
        } else inspect({ run: row.run, record: row.record });
      });
  }
  function metricsHTML(rows) {
    return rows
      .map(
        ([title, value, unit]) =>
          `<div class="metric"><small>${esc(title)}</small><strong>${esc(typeof value === "string" ? value : fmt(value))}</strong><span>${esc(unit ?? "")}</span></div>`,
      )
      .join("");
  }
  function syncMedia(card, asset, target, caption, context = null) {
    window.ReplayMedia.update(
      card,
      asset,
      target,
      caption,
      context,
      state.mediaEpoch,
    );
  }
  function renderCardState() {
    const cards = [...document.querySelectorAll(".precision-card")];
    state.rows.forEach(({ run, record, episode, comparable }, i) => {
      const card = cards[i],
        status = card.querySelector(".card-status"),
        details = card.querySelector(".card-details"),
        metrics = card.querySelector(".metrics"),
        action = card.querySelector(".card-action");
      if (run.error) {
        status.textContent = run.error;
        status.dataset.status = "bad";
        details.textContent = run.id;
        return;
      }
      if (state.kind === "model") {
        const observation = (run.observations ?? []).find(
            (o) => o.observation_id === record?.observation_id,
          ),
          asset = observation?.assets?.find(
            (a) => a.camera === $("camera").value,
          ),
          stats = record
            ? R.cleanStats(run, state.scope)
            : { count: 0, p50: null, p95: null },
          output = R.outputAt(record, state.clock);
        syncMedia(
          card,
          asset,
          asset ? 0 : null,
          observation
            ? `固定输入 · 静态帧 · Obs ${record.observation ?? "?"} · ${observation.sha256?.slice(0, 12) ?? "hash 未知"}`
            : "没有匹配输入",
        );
        const warnings = [
          ...(run.warnings ?? []),
          ...(run.retained_outputs?.length
            ? [
                `${run.retained_outputs.length} 份未匹配保留输出，仅可在来源中检查`,
              ]
            : []),
        ];
        status.textContent = !record
          ? "缺少同输入 / 同 seed 的记录"
          : `${run.status ?? "状态未知"} · ${state.type === "profile" ? "Profile · " + (record.timeline_status ?? "未知") : "Clean · " + (output.status === "pending" ? "推理中" : output.status)}${warnings.length ? " · " + warnings.join("；") : ""}`;
        if (record && state.type === "clean" && record.clean_wall !== true)
          status.textContent += " · 插桩状态未知：原始 wall 不计入 Clean 统计";
        status.textContent += record
          ? ` · ${comparable ? "配对条件已知" : "未配对 / 比较条件未知"}`
          : "";
        status.dataset.status =
          !record || run.status === "failed" || run.status === "incomplete"
            ? "bad"
            : "";
        metrics.innerHTML = metricsHTML([
          ["Clean P50", stats.p50, "ms"],
          ["Clean P95", stats.p95, "ms"],
          ["Clean 样本", String(stats.count), ""],
          [
            state.type === "profile" ? "Profile host wall" : "当前请求 wall",
            record?.host_wall_ms ?? record?.wall_ms,
            "ms",
          ],
          ["GPU 活动合计", record?.gpu_work_ms, "ms"],
          ["Device span", record?.device_span_ms, "ms"],
        ]);
        details.textContent = `${state.scope} · seed ${record?.seed ?? "未知"} · 同精度验证 ${run.validation?.status ?? "未知"} · BF16 max |Δ| ${fmt(comparable ? record?.difference_from_bf16?.max_abs : null, 5)}\n${output.status === "unavailable" ? output.reason : output.status === "pending" ? "动作尚未产生；可拖动到请求完成后查看。" : "动作已记录；输出差异不是任务质量判据。"}`;
        action.textContent = record
          ? `检查请求 · ${record.request_id ?? record.id}`
          : "检查运行状态与保留输出";
      } else {
        if (!R.playableEpisode(episode)) {
          status.textContent = `记录 ${episode?.status ?? "unavailable"} · ${episode?.reason ?? "没有完整控制证据，无法播放本回合"}`;
          status.dataset.status = "bad";
          syncMedia(card, null, null, "无可播放的控制窗口");
          metrics.innerHTML = metricsHTML([["控制证据", "不可用", ""]]);
          details.textContent = `${run.id} · Task ${episode?.task_id ?? "?"} · episode ${episode?.episode_id ?? "?"} · seed ${episode?.seed ?? "?"}`;
          action.textContent = "检查缺失证据与来源";
          return;
        }
        const at = R.robotAt(episode, state.clock, $("camera").value),
          meta = episode?.metadata ?? {},
          queue = episode?.queue
            ? Q.stateAt(
                episode.queue,
                Math.min(state.clock, at.end ?? state.clock),
              )
            : null;
        const verdict =
          meta.success === true
            ? "成功"
            : meta.success === false
              ? "失败"
              : "结果未知";
        status.textContent = !episode
          ? "该任务没有回合记录"
          : `${at.ended ? "已结束" : "正在回放"} · 回合最终结果：${verdict} · ${meta.control_rate_valid === false ? "时序无效" : meta.control_rate_valid === true ? "控制时序有效" : "控制时序未知"} · ${meta.termination_reason ?? "原因未知"}${episode.robot?.error ? " · " + episode.robot.error : ""}`;
        status.dataset.status =
          meta.success === false || meta.control_rate_valid === false
            ? "bad"
            : meta.success === true
              ? "good"
              : "";
        syncMedia(
          card,
          at.media,
          at.frame ? at.frame.media_time_s + 0.0001 : null,
          at.frame
            ? `${at.frame.frame_id} · 采集 ${fmt(at.frame.relative_s, 3)} s · ${at.ended ? "末帧保持" : "本精度独立轨迹"}`
            : "此时尚无帧",
          {
            frame: at.frame,
            frames: episode.robot?.frames ?? [],
            cursor: state.clock,
            ended: at.ended,
          },
        );
        metrics.innerHTML = metricsHTML([
          ["S1 RPC P50", meta.action_rpc_seconds?.p50 * 1000, "ms"],
          ["S1 RPC P95", meta.action_rpc_seconds?.p95 * 1000, "ms"],
          ["实际控制率", meta.actual_control_hz, "Hz"],
          ["当前动作的图像年龄", queue?.imageAgeMs, "ms"],
          ["当前动作的状态年龄", queue?.stateAgeMs, "ms"],
          ["控制动作", at.tick ? String(at.tick.tick) : "—", "tick"],
        ]);
        metrics.title =
          "当前动作的信息年龄 = 当前控制墙钟游标 − 该已执行命令实际来源的采集时刻；不是执行瞬间的汇总 AoI。回合结束后固定在终止时刻，不虚构后续执行。";
        details.textContent = `位置 [m] ${at.robotState?.position_xyz_m?.map((v) => fmt(v, 4)).join(", ") ?? "未知"} · 缓存 AoI ${fmt(queue?.cacheAgeMs)} ms · d ${queue?.delay ?? "未知"} · ${episode?.robot?.recorded_frames ?? 0} 帧 / 丢弃 ${episode?.robot?.dropped_frames ?? 0}`;
        action.textContent = at.tick
          ? `检查动作 ${at.tick.tick} · ${at.tick.action?.map((v) => fmt(v, 3)).join(", ")}`
          : "游标前没有已执行命令";
      }
    });
  }
  function filteredEvents() {
    return state.events.filter(
      (e) =>
        ($("stage").value === "all" || e.stage === $("stage").value) &&
        ($("category").value === "all" || e.category === $("category").value),
    );
  }
  function windowBounds() {
    const fraction =
        $("window").value === "all" ? 1 : Number($("window").value),
      span = state.end * fraction,
      left = Math.max(0, Math.min(state.clock - span / 2, state.end - span));
    return [left, left + span];
  }
  function inspect(event) {
    if (!event) return;
    state.selected = event.id ?? null;
    let data;
    if (event.kind === "kernel")
      data = {
        precision: event.precision,
        request_id: event.record.id,
        observation_id: event.record.observation_id,
        seed: event.record.seed,
        stage: event.stage,
        category: event.category,
        kernel: event.name,
        stream: event.stream,
        device: event.device,
        start_ms: event.time,
        duration_ms: event.duration_ms,
        output: event.record.output,
        source: event.run.provenance,
      };
    else if (event.tick) {
      const q = event.episode.queue
        ? Q.stateAt(event.episode.queue, event.tick.relative_s)
        : null;
      data = {
        precision: event.precision,
        tick: event.tick.tick,
        control_wall_s: event.tick.relative_s,
        command_units: event.episode.robot?.command_units,
        command: event.tick.action,
        robot_response: event.tick.robot_response,
        request: q?.executedRequest ?? null,
        feature_id: q?.executedRequest?.feature_id ?? null,
        source_frame_tick: q?.executedRequest?.feature_source_tick ?? null,
        image_aoi_ms: q?.imageAgeMs,
        state_aoi_ms: q?.stateAgeMs,
        cache_aoi_ms: q?.cacheAgeMs,
        slot: event.tick.executed_slot,
        initial_feature:
          event.tick.executed_slot?.producer_kind === "bootstrap"
            ? event.episode.queue?.initial_feature
            : undefined,
      };
    } else if (event.request)
      data = {
        precision: event.precision,
        request: event.request,
        source: event.run.provenance,
      };
    else if (event.record)
      data = {
        precision: event.run.precision,
        ...event.record,
        timeline: event.record.timeline
          ? `${event.record.timeline.length} actual events (select event for kernel)`
          : undefined,
        output: R.outputAt(event.record, state.clock),
        validation: event.run.validation?.status,
        provenance: event.run.provenance,
      };
    else
      data = state.rows.map(({ run, episode }) => ({
        id: run.id,
        precision: run.precision,
        status: run.status,
        error: run.error,
        mode: run.mode,
        configuration: run.configuration,
        checkpoint: run.checkpoint,
        device: run.device,
        clock_cohort: run.clock_cohort,
        shared_config_sha256: run.shared_config_sha256,
        boundaries: run.boundaries,
        validation: run.validation,
        retained_outputs: run.retained_outputs,
        warnings: run.warnings,
        provenance: run.provenance,
        episode: episode
          ? {
              id: episode.id,
              status: episode.status,
              metadata: episode.metadata,
              robot: {
                status: episode.robot?.status,
                error: episode.robot?.error,
                recorded_frames: episode.robot?.recorded_frames,
                dropped_frames: episode.robot?.dropped_frames,
              },
              paired_fixture_key: episode.paired_fixture_key,
            }
          : null,
      }));
    $("inspector").textContent = JSON.stringify(data, null, 2);
    renderEventList();
    renderLineage();
  }
  function renderLineage() {
    if (state.loading) return;
    const selected = state.events.find((e) => e.id === state.selected);
    const graphs = state.rows.map((row) => ({
      ...row,
      graph:
        state.kind === "model"
          ? window.ReplayLineage.model(row.run, row.record, state.clock)
          : window.ReplayLineage.deployment(
              row.episode,
              Math.min(state.clock, R.episodeEnd(row.episode) ?? state.clock),
              selected?.precision === row.run.precision ? selected : null,
            ),
    }));
    $("lineage-title").textContent =
      state.kind === "deployment"
        ? "观测 → VLM → 缓存 → DiT → 动作 → 机器人"
        : state.scope === "preparation_transfer"
          ? "输入准备与传输 · 本次记录未执行 VLM / DiT"
          : ["policy", "prepared_model", "preparation_model"].includes(
                state.scope,
              )
            ? "本请求 · VLM → 视觉特征 → DiT → 动作"
            : "未知测量边界 · 不推断组件执行";
    $("lineage-note").textContent =
      state.kind === "model"
        ? `${R.scopeInfo(state.scope).note} 虚线只说明记录中已知的执行路径，不编造未记录组件耗时。`
        : selected
          ? "当前高亮所选事件的真实来源；点击“跟随时间游标”恢复当前执行动作。S1[n].k 中 n 是实际消费的 S2 特征。"
          : "跟随当前实际执行动作：连线只采用 source tick、feature sequence、缓存发布、实际读取与 executed_slot；不会把最新缓存误当成已消费缓存。";
    $("lineage").innerHTML = graphs
      .map(({ run, graph }, row) => {
        const marker = "lineage-arrow-" + row;
        const arrows = graph.edges
          .map((edge) => {
            const i = graph.nodes.findIndex((n) => n.key === edge.from),
              j = graph.nodes.findIndex((n) => n.key === edge.to);
            return `<path class="lineage-arrow" data-from="${edge.from}" data-to="${edge.to}" d="M${i * 200 + 178},47 H${j * 200 + 16}" marker-end="url(#${marker})" stroke="${color(run.precision)}" stroke-width="2" ${edge.declared ? 'stroke-dasharray="4 3"' : ""}><title>${esc(edge.basis)}</title></path>`;
          })
          .join("");
        return `<div class="lineage-row ${selected?.precision === run.precision ? "selected" : ""}" data-precision="${esc(run.precision)}" style="--precision:${color(run.precision)}"><strong class="lineage-precision">${esc(label(run.precision))}</strong><div class="lineage-track"><svg viewBox="0 0 1200 104" preserveAspectRatio="none" aria-hidden="true"><defs><marker id="${marker}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="${color(run.precision)}"/></marker></defs>${arrows}</svg>${graph.nodes.map((node, index) => `<button class="lineage-node ${node.known ? "known" : "unknown"}" data-lineage-row="${row}" data-node="${index}" data-kind="${node.key}"><strong>${esc(node.title)}</strong>${node.lines.map((text) => `<span>${esc(text)}</span>`).join("")}</button>`).join("")}</div></div>`;
      })
      .join("");
    for (const button of $("lineage").querySelectorAll(".lineage-node"))
      button.addEventListener("click", () => {
        const row = graphs[Number(button.dataset.lineageRow)],
          node = row.graph.nodes[Number(button.dataset.node)];
        const event = node.tick
          ? state.events.find((e) => e.tick?.id === node.tick.id)
          : node.request
            ? state.events.find(
                (e) =>
                  e.request?.id === node.request.id ||
                  (e.record?.id === node.request.id &&
                    (!node.stage || e.stage === node.stage)),
              )
            : null;
        if (event) inspect(event);
        else
          $("inspector").textContent = JSON.stringify(
            { precision: row.run.precision, ...node },
            null,
            2,
          );
      });
  }
  let timelineKey = "";
  function renderTimeline() {
    const events = filteredEvents(),
      [a, b] = windowBounds(),
      W = 1200,
      L = 130,
      right = 1180,
      rowH = 24,
      lanes = [];
    for (const row of state.rows) {
      const p = row.run.precision;
      if (state.kind === "model" && state.type === "clean")
        lanes.push({ p, stage: "all", name: label(p) + " · wall" });
      else
        for (const stage of [
          "vlm",
          "action",
          ...(state.kind === "deployment" ? ["control"] : []),
        ])
          lanes.push({
            p,
            stage,
            name:
              label(p) +
              " · " +
              { vlm: "VLM / S2", action: "Action / S1", control: "Control" }[
                stage
              ],
          });
    }
    const H = lanes.length * rowH + 45,
      x = (t) => L + ((t - a) / (b - a)) * (right - L);
    const key = JSON.stringify([
      state.mediaEpoch,
      state.kind,
      state.type,
      $("stage").value,
      $("category").value,
      a,
      b,
    ]);
    if (key === timelineKey && $("timeline").querySelector(".time-cursor")) {
      const cursor = $("timeline").querySelector(".time-cursor");
      cursor.setAttribute("x1", x(state.clock));
      cursor.setAttribute("x2", x(state.clock));
      return;
    }
    timelineKey = key;
    let svg = "";
    for (let i = 0; i <= 5; i++) {
      const t = a + ((b - a) * i) / 5;
      svg += `<line x1="${x(t)}" y1="8" x2="${x(t)}" y2="${H - 28}" stroke="#e5e9e3"/><text x="${x(t)}" y="${H - 9}" text-anchor="middle" font-size="10" fill="#68776c">${fmt(t, state.kind === "model" ? 1 : 2)} ${state.kind === "model" ? "ms" : "s"}</text>`;
    }
    lanes.forEach((lane, i) => {
      svg += `<text x="8" y="${i * rowH + 25}" font-size="10" fill="${color(lane.p)}">${esc(lane.name)}</text><line x1="${L}" y1="${i * rowH + 33}" x2="${right}" y2="${i * rowH + 33}" stroke="#edf0eb"/>`;
    });
    state.hits = [];
    for (const event of events) {
      if (event.end < a || event.time > b) continue;
      const lane = lanes.findIndex(
        (l) => l.p === event.precision && l.stage === event.stage,
      );
      if (lane < 0) continue;
      const index = state.hits.push(event) - 1,
        xx = x(Math.max(a, event.time)),
        width = Math.max(1.5, x(Math.min(b, event.end)) - xx),
        fill = event.category
          ? (categories[event.category] ?? "#9b8eab")
          : color(event.precision);
      svg += `<rect class="hit" data-hit="${index}" data-event-kind="${event.kind}" x="${xx}" y="${lane * rowH + 12}" width="${width}" height="17" rx="1" fill="${fill}" opacity=".86"><title>${esc(label(event.precision) + " " + event.name + " · " + fmt(event.time, 4) + " → " + fmt(event.end, 4) + (state.kind === "model" ? " ms" : " s"))}</title></rect>`;
    }
    svg += `<line class="time-cursor" x1="${x(state.clock)}" y1="6" x2="${x(state.clock)}" y2="${H - 25}" stroke="#b64d44" stroke-width="1.6"/>`;
    $("timeline").innerHTML =
      `<svg viewBox="0 0 ${W} ${Math.max(85, H)}" role="img" aria-label="实际请求与组件时间线">${svg}</svg>`;
    $("timeline-note").textContent =
      state.kind === "deployment"
        ? "S1 / S2 显示完整 RPC 区间；Control 是实际命令时刻。它们不是 GPU kernel 时间。"
        : state.type === "profile"
          ? "实际 CUDA 设备事件；同一行内的多 stream 可能重叠，点击查看。"
          : `${R.scopeInfo(state.scope).label} · 仅有请求 wall；本 Clean 调用未采集 GPU timeline。`;
    const used = [...new Set(events.map((e) => e.category).filter(Boolean))];
    $("legend").innerHTML = used
      .map(
        (k) =>
          `<span style="--key:${categories[k] ?? "#9b8eab"}">${esc(k)}</span>`,
      )
      .join("");
    $("timeline")
      .querySelector("svg")
      .addEventListener("click", (event) => {
        const target = event.target.closest("[data-hit]");
        if (target) inspect(state.hits[Number(target.dataset.hit)]);
        else {
          const rect = event.currentTarget.getBoundingClientRect(),
            t =
              a +
              ((((event.clientX - rect.left) / rect.width) * W - L) /
                (right - L)) *
                (b - a);
          setClock(t);
        }
      });
  }
  function plot(
    series,
    { xLabel = "", yLabel = "", cursor = null, square = false } = {},
  ) {
    const valid = series.flatMap((s) =>
      s.points.filter(([x, y]) => R.finite(x) && R.finite(y)),
    );
    if (!valid.length)
      return '<p class="placeholder">当前游标没有可用数据。未记录值保持未知。</p>';
    const W = 620,
      H = 210,
      L = 58,
      T = 14,
      B = 171,
      E = 604;
    let xmin = Math.min(...valid.map((p) => p[0])),
      xmax = Math.max(...valid.map((p) => p[0])),
      ymin = Math.min(...valid.map((p) => p[1])),
      ymax = Math.max(...valid.map((p) => p[1]));
    if (xmax === xmin) {
      xmin -= 0.5;
      xmax += 0.5;
    }
    if (ymax === ymin) {
      ymin -= 0.01;
      ymax += 0.01;
    }
    const pad = (ymax - ymin) * 0.08;
    ymin -= pad;
    ymax += pad;
    if (square) {
      const span = Math.max(xmax - xmin, ymax - ymin),
        xc = (xmin + xmax) / 2,
        yc = (ymin + ymax) / 2;
      xmin = xc - (span * (E - L)) / (B - T) / 2;
      xmax = xc + (span * (E - L)) / (B - T) / 2;
      ymin = yc - span / 2;
      ymax = yc + span / 2;
    }
    const x = (t) => L + ((t - xmin) / (xmax - xmin)) * (E - L),
      y = (v) => B - ((v - ymin) / (ymax - ymin)) * (B - T);
    let out = "";
    for (let i = 0; i <= 4; i++) {
      const xx = xmin + ((xmax - xmin) * i) / 4,
        yy = ymin + ((ymax - ymin) * i) / 4;
      out += `<line x1="${L}" y1="${y(yy)}" x2="${E}" y2="${y(yy)}" stroke="#e5e9e3"/><text x="${L - 7}" y="${y(yy) + 3}" text-anchor="end" font-size="10" fill="#657268">${fmt(yy, 3)}</text><text x="${x(xx)}" y="188" text-anchor="middle" font-size="10" fill="#657268">${fmt(xx, 2)}</text>`;
    }
    for (const s of series) {
      let d = "",
        previous = false;
      for (const point of s.points) {
        if (!point.every(R.finite)) {
          previous = false;
          continue;
        }
        d += `${previous ? "L" : "M"}${x(point[0])},${y(point[1])} `;
        previous = true;
      }
      out += `<path d="${d}" fill="none" stroke="${color(s.precision)}" stroke-width="1.8"/><text x="${L + series.indexOf(s) * 120}" y="207" font-size="10" fill="${color(s.precision)}">${esc(label(s.precision))}</text>`;
    }
    if (R.finite(cursor) && cursor >= xmin && cursor <= xmax)
      out += `<line x1="${x(cursor)}" y1="${T}" x2="${x(cursor)}" y2="${B}" stroke="#b64d44" stroke-dasharray="3 3"/>`;
    out += `<text x="8" y="10" font-size="10" fill="#657268">${esc(yLabel)}</text><text x="${E}" y="207" text-anchor="end" font-size="10" fill="#657268">${esc(xLabel)}</text>`;
    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(yLabel + " vs " + xLabel)}">${out}</svg>`;
  }
  function renderPlots() {
    const channel = $("channel").value;
    if (state.kind === "model") {
      const series = state.rows.map(({ run, record }) => {
        const out = R.outputAt(record, state.clock);
        return {
          precision: run.precision,
          points:
            out.status === "recorded"
              ? (out.actions?.[channel]?.values?.flat(Infinity) ?? []).map(
                  (v, i) => [i, v],
                )
              : [],
        };
      });
      $("curve-title").textContent = "动作输出 · 预测步";
      $("curve").innerHTML = plot(series, {
        xLabel: "decoded step",
        yLabel: channel + " · controller command",
      });
      $("curve-note").textContent =
        "这是模型预测序列，不是已执行的机器人轨迹。只有当前请求完成且保留输出时才显示。";
      $("projection-title").textContent = "输入与验证";
      $("projection").innerHTML =
        '<div class="provenance-lines">' +
        state.rows
          .map(({ run, record, comparable }) => {
            const observation = run.observations?.find(
              (o) => o.observation_id === record?.observation_id,
            );
            return `<strong style="color:${color(run.precision)}">${esc(label(run.precision))}</strong> · ${esc(record?.observation_id ?? "输入未知")}<br>seed ${esc(record?.seed ?? "未知")} · 同精度输出 / RNG 验证 ${esc(run.validation?.status ?? "未知")}<br>BF16 max |Δ| ${fmt(comparable ? record?.difference_from_bf16?.max_abs : null, 6)} · 保留未匹配输出 ${run.retained_outputs?.length ?? 0}<br>${esc(
              Object.values(observation?.language ?? {})
                .flat(Infinity)
                .join(" "),
            )}<br>`;
          })
          .join("<br>") +
        "</div>";
    } else {
      const series = state.rows.map(({ run, episode }) => ({
        precision: run.precision,
        points: R.robotSeries(episode, channel).filter(
          ([t]) => t <= state.clock,
        ),
      }));
      $("curve-title").textContent = "机器人状态 / 已执行命令";
      $("curve").innerHTML = plot(series, {
        xLabel: "control wall / s",
        yLabel: channel.startsWith("command:")
          ? "normalized command"
          : (R.channels[channel]?.[2] ?? ""),
        cursor: state.clock,
      });
      $("curve-note").textContent =
        "位置使用 m；旋转使用 axis-angle rotation vector（rad），不是 Euler 角；夹爪显示两个独立关节。状态按实际计算完成时间出现，命令按实际执行时间出现。";
      $("projection-title").textContent = "实际末端轨迹 · XY";
      $("projection").innerHTML = plot(
        state.rows.map(({ run, episode }) => ({
          precision: run.precision,
          points: R.stateSamples(episode)
            .filter((s) => s.time <= state.clock)
            .map((s) => s.value.position_xyz_m?.slice(0, 2) ?? [null, null]),
        })),
        { xLabel: "X / m", yLabel: "Y / m", square: true },
      );
    }
  }
  function renderEventList() {
    const [a, b] = windowBounds(),
      events = filteredEvents().filter((e) => e.time >= a && e.time <= b);
    let start = events.findIndex((e) => e.time >= state.clock);
    if (start < 0) start = events.length;
    start = Math.max(0, start - 40);
    const slice = events.slice(start, start + 140);
    $("event-count").textContent =
      `${events.length} 个 · 显示游标附近 ${slice.length} 个`;
    $("event-list").innerHTML =
      slice
        .map(
          (e, i) =>
            `<button class="event-row ${e.id === state.selected ? "selected" : ""}" data-list-index="${i}" style="--precision:${color(e.precision)}"><span class="precision">${esc(label(e.precision))}</span><span>${fmt(e.time, state.kind === "model" ? 3 : 2)} ${state.kind === "model" ? "ms" : "s"}</span><span title="${esc(e.name)}">${esc(e.name)}</span><span>${fmt((e.end - e.time) * (state.kind === "model" ? 1 : 1000), 3)} ms</span></button>`,
        )
        .join("") || '<p class="small-note">此窗口没有相应事件。</p>';
    for (const button of $("event-list").querySelectorAll("button"))
      button.addEventListener("click", () => {
        const event = slice[Number(button.dataset.listIndex)];
        setClock(event.time);
        inspect(event);
      });
  }
  function render() {
    if (state.loading) return;
    $("seek").value = state.clock;
    $("clock").textContent =
      `${fmt(state.clock, state.kind === "model" ? 2 : 3)} / ${fmt(state.end, state.kind === "model" ? 1 : 2)} ${state.kind === "model" ? "ms" : "s"}`;
    renderCardState();
    renderLineage();
    renderTimeline();
    renderPlots();
    renderEventList();
  }
  function setClock(t) {
    state.clock = Math.max(0, Math.min(state.end, t));
    render();
  }
  function step(direction) {
    stop();
    const times = [
      ...new Set([
        0,
        state.end,
        ...filteredEvents().flatMap((e) => [e.time, e.end]),
      ]),
    ].sort((a, b) => a - b);
    setClock(
      direction > 0
        ? (times.find((t) => t > state.clock + 1e-7) ?? state.end)
        : ([...times.filter((t) => t < state.clock - 1e-7)].at(-1) ?? 0),
    );
  }
  let lastDraw = 0;
  function animate(now) {
    if (state.playing && now - lastDraw >= 50) {
      lastDraw = now;
      if (state.lastWall != null) {
        state.clock +=
          ((now - state.lastWall) / (state.kind === "model" ? 1 : 1000)) *
          Number($("speed").value);
        if (state.clock >= state.end) {
          state.clock = state.end;
          stop();
        }
        render();
      }
      state.lastWall = now;
    }
    requestAnimationFrame(animate);
  }
  function bind() {
    $("lineage-follow").addEventListener("click", () => {
      state.selected = null;
      renderLineage();
      renderEventList();
    });
    for (const kind of ["model", "deployment"])
      $("mode-" + kind).addEventListener("click", () => {
        $("evidence-mode").value = kind;
        $("evidence-mode").dispatchEvent(new Event("change"));
      });
    $("evidence-mode").addEventListener("change", () => {
      state.kind = $("evidence-mode").value;
      state.clock = 0;
      state.sample = null;
      $("speed").value = state.kind === "model" ? "0.01" : "1";
      loadRuns();
    });
    $("scope").addEventListener("change", () => {
      state.scope = $("scope").value;
      configureSamples();
    });
    $("record-type").addEventListener("change", () => {
      state.type = $("record-type").value;
      configureSamples();
    });
    $("sample").addEventListener("change", selectSample);
    $("precision").addEventListener("change", () => {
      state.precision = $("precision").value;
      configureSamples();
    });
    $("camera").addEventListener("change", () => {
      state.mediaEpoch++;
      renderCardState();
    });
    for (const id of ["stage", "category", "window"])
      $(id).addEventListener("change", render);
    $("channel").addEventListener("change", renderPlots);
    $("seek").addEventListener("input", () => {
      stop();
      setClock(Number($("seek").value));
    });
    $("previous").addEventListener("click", () => step(-1));
    $("next").addEventListener("click", () => step(1));
    $("play").addEventListener("click", () => {
      if (state.loading || $("play").disabled) return;
      if (state.playing) stop();
      else {
        if (state.clock >= state.end) state.clock = 0;
        state.playing = true;
        state.lastWall = null;
        $("play").textContent = "暂停";
      }
    });
    $("inspect-run").addEventListener("click", () => inspect({}));
    $("download").addEventListener("click", () => {
      const data = {
        catalog_created_at: state.catalog?.created_at,
        kind: state.kind,
        record_type: state.type,
        cursor: state.clock,
        cursor_unit: state.kind === "model" ? "ms" : "control_wall_s",
        camera: $("camera").value,
        runs: state.rows.map(({ run, record, episode }) => ({
          id: run.id,
          precision: run.precision,
          error: run.error,
          source_url: "/api/replay/run/" + run.id,
          provenance: run.provenance,
          record_id: record?.id,
          observation_id: record?.observation_id,
          seed: record?.seed,
          comparison_key: record?.comparison_key,
          output: record ? R.outputAt(record, state.clock) : undefined,
          episode_id: episode?.id,
          robot: episode
            ? R.robotAt(episode, state.clock, $("camera").value)
            : undefined,
        })),
        inspector: JSON.parse(
          $("inspector").textContent.startsWith("{") ||
            $("inspector").textContent.startsWith("[")
            ? $("inspector").textContent
            : "null",
        ),
      };
      const url = URL.createObjectURL(
          new Blob([JSON.stringify(data, null, 2)], {
            type: "application/json",
          }),
        ),
        a = document.createElement("a");
      a.href = url;
      a.download = "quantization-replay-selection.json";
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    });
  }
  async function initialize() {
    bind();
    requestAnimationFrame(animate);
    try {
      const catalog = await json("/api/replay/catalog");
      if (catalog.schema !== "pir2.replay.catalog" || catalog.version !== 1)
        throw new Error("不支持的 replay catalog 版本");
      state.catalog = catalog;
      state.descriptors = catalog.runs ?? [];
      const requestedView = new URLSearchParams(location.search).get("view");
      state.kind =
        requestedView === "model"
          ? "model"
          : requestedView === "deployment" ||
              requestedView === "robot" ||
              state.descriptors.some(
                (r) => r.kind === "deployment" && r.episodes > 0,
              )
            ? "deployment"
            : "model";
      $("evidence-mode").value = state.kind;
      $("speed").value = state.kind === "model" ? "0.01" : "1";
      $("catalog-meta").textContent =
        `${catalog.created_at ?? "创建时间未知"} · ${catalog.clock_cohort ?? "clock cohort 未声明"}`;
      await loadRuns();
    } catch (error) {
      notice("未连接回放数据", true);
      $("error-banner").hidden = false;
      $("error-banner").textContent =
        `${error.message}。使用预览服务的 --replay 参数配置 catalog.json；队列实验台仍可使用。`;
    }
  }
  initialize();
})();
