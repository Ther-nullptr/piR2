/* LIBERO-Dynamic uses the existing page, transport and bounded media presenter. */
"use strict";
window.DynamicReplayView = (() => {
  const D = window.DynamicReplay,
    $ = (id) => document.getElementById(id);
  const esc = (value) =>
    String(value ?? "").replace(
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
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const ms = (value) => (finite(value) ? value * 1000 : null);
  const fmt = (value, digits = 2) =>
    finite(value) ? value.toFixed(digits) : "—";
  const color = (precision) =>
    ({ bf16: "#315b8a", w8a8: "#087b6d", w4a4: "#a45b08" })[precision] ??
    "#78669d";
  const state = {
    runs: [],
    rows: [],
    selection: {},
    comparisons: [],
    clock: 0,
    end: 1,
    playing: false,
    last: null,
    epoch: 0,
    selected: null,
    panelKey: null,
  };
  const option = (select, values, chosen) => {
    select.innerHTML = values
      .map(
        ([value, text]) =>
          `<option value="${esc(value)}">${esc(text)}</option>`,
      )
      .join("");
    if (values.some(([value]) => value === chosen)) select.value = chosen;
  };
  const result = (e) =>
    e.status !== "ok"
      ? `错误 · ${e.error ?? e.status ?? "未知"}`
      : e.success === true
        ? "任务成功"
        : e.success === false
          ? "任务未成功"
          : "结果未知";
  const conditionLabel = (e) =>
    `${e.suite} / task ${e.task_id} · ${e.level} / traj ${e.traj_index} · ${fmt(e.speed_mps, 3)} m/s · seed ${e.seed ?? "—"}`;
  const streaming = (e) => e.protocol === "pir2_dynamic";
  const timingLabel = D.timingLabel;
  const runLabel = ({ run, episode: e }) =>
    `${run.label} · ${timingLabel(e)} · ${streaming(e) ? "每步执行1" : `K${e.execution_prefix_steps ?? "?"}`}`;
  const name = (run) =>
    `${run.precision.toUpperCase()} · ${D.protocolLabel(run)}`;
  const eventLabel = (e) =>
    e.kind === "s2"
      ? `S2-${e.source.feature_sequence}`
      : e.kind === "s1"
        ? `S1[t${e.source.request_tick}] ${e.source.normal_or_recovery ?? ""}`
        : e.kind === "action"
          ? `A${e.action_step}${e.partial ? " · 中断物理步" : e.fallback ? " · Fallback" : ""}`
          : e.kind === "bootstrap_hold"
            ? "Bootstrap hold"
            : `Hold C${e.chunk_id}`;
  const dependencyLabel = (e) =>
    e.kind === "s1"
      ? `← ${e.source.feature_sequence === 0 ? "Bootstrap" : `S2-${e.source.feature_sequence}`} · valid [${e.source.server_audit?.valid_start_tick ?? "?"}, ${e.source.server_audit?.valid_end_tick ?? "?"})`
      : e.kind === "s2"
        ? `观测 tick ${e.source.source_tick}`
        : e.kind === "bootstrap_hold"
          ? "启动 · 物体持续运动"
          : e.producer_kind === "request"
            ? `← S1[t${e.source.source?.producer_request_tick}] · AoI ${fmt(ms(e.feature_age_s), 0)} ms`
            : e.producer_kind
              ? e.producer_kind
              : `← C${e.chunk_id}`;
  function navigate(view) {
    const url = new URL(location.href);
    url.searchParams.set("view", view);
    location.href = url;
  }
  function inspect(value) {
    $("inspector").textContent = JSON.stringify(value, null, 2);
  }
  function selectRequest(row, event) {
    state.selected = {
      runId: row.run.id,
      columnKey: row.columnKey,
      requestId: event.request_id,
      event,
    };
    inspect({
      clock: "rollout_sim_time_s",
      run_id: row.run.id,
      episode_id: row.episode.id,
      event,
      request: row.episode.requests.find((r) => r.id === event.request_id),
      provenance: row.episode.provenance,
    });
    render();
  }
  function configure() {
    state.playing = false;
    state.clock = 0;
    state.selected = null;
    state.epoch++;
    state.panelKey = null;
    state.rows = D.rows(
      state.runs,
      $("sample").value,
      $("precision").value,
      state.selection,
      state.comparisons,
    );
    state.end =
      Math.max(
        0,
        ...state.rows.map((r) => r.episode.duration_s).filter(finite),
      ) || 1;
    $("seek").max = state.end;
    const grid = $("precision-cards");
    grid.style.setProperty("--precision-count", state.rows.length || 1);
    grid.innerHTML =
      state.rows
        .map(
          (
            { run, episode: e },
            i,
          ) => `<article class="precision-card dynamic-card" data-precision="${esc(run.precision)}" data-boundary="dynamic_simulation" style="--precision:${color(run.precision)}">
      <div class="card-heading"><h2>${esc(name(run))}</h2><span class="coupling-badge">LIBERO-Dynamic</span></div>
      <select class="dynamic-run" data-index="${i}" aria-label="${esc(name(run))} 动态配置"></select><button class="dynamic-compare" data-index="${i}">${state.rows[i].comparison ? "移除对照列" : "添加同协议对照"}</button>
      <div class="media-stage"><span class="media-message">等待实际物理帧</span><div class="frame-meta"></div></div>
      <div class="card-status">${esc(result(e))}</div><p class="dynamic-language">${esc(e.language ?? "任务语言未记录")}</p>
      <div class="metrics"></div><p class="card-details">${esc(conditionLabel(e))}<br>${esc(timingLabel(e))} · ${esc(e.timing_rule ?? e.timing_domain ?? "未知")}${streaming(e) ? "<br>启动等待 " + fmt(ms(e.startup_seconds ?? e.stage_latencies?.bootstrap_s), 1) + " ms；真实 GPU RPC 串行计算，逻辑通道独立服务。" : ""}<br>${esc(D.modelContract(e))}<br>${esc(e.video_view_contract ?? "场景观察相机；不是模型预处理后的输入张量")}<br>${esc(e.warnings.join(" · "))}</p>
      <a class="run-source" href="/api/replay/run/${encodeURIComponent(run.id)}" target="_blank" rel="noopener">Run JSON ↗</a></article>`,
        )
        .join("") || '<p class="loading">此条件没有记录。</p>';
    for (const select of grid.querySelectorAll(".dynamic-run")) {
      const row = state.rows[Number(select.dataset.index)];
      option(
        select,
        D.candidates(
          state.runs,
          $("sample").value,
          row.run.precision,
          D.protocol(row.run),
        ).map((r) => [r.run.id, runLabel(r)]),
        row.run.id,
      );
      select.addEventListener("change", () => {
        state.selection[row.columnKey] = select.value;
        configure();
      });
    }
    for (const button of grid.querySelectorAll(".dynamic-compare")) {
      const row = state.rows[Number(button.dataset.index)],
        key = D.groupKey(row.run);
      button.hidden =
        !row.comparison &&
        (state.comparisons.includes(key) ||
          D.candidates(
            state.runs,
            $("sample").value,
            row.run.precision,
            D.protocol(row.run),
          ).length < 2);
      button.onclick = () => {
        state.comparisons = row.comparison
          ? state.comparisons.filter((k) => k !== key)
          : [...state.comparisons, key];
        configure();
      };
    }
    render();
  }
  function viewport() {
    const choice = $("window").value;
    const span =
      choice === "all"
        ? state.end
        : choice === "auto"
          ? Math.min(state.end, 1.5)
          : state.end * Number(choice);
    return window.ReplayTimeline.viewport(state.clock, state.end, span);
  }
  function renderTimeline() {
    const view = viewport(),
      start = view.start ?? view[0],
      end = view.end ?? view[1];
    const width = end - start,
      pct = (t) => (100 * (t - start)) / width;
    $("timeline-range").textContent = `${fmt(start, 3)}–${fmt(end, 3)} sim s`;
    let html = D.timelineAxis(start, end);
    for (const row of state.rows) {
      const { run, episode: e } = row;
      const selectedIds =
        state.selected?.columnKey === row.columnKey
          ? D.lineage(e, state.selected.requestId)
          : [];
      const lanes = streaming(e)
        ? [
            ["s2", "S2 · VLM"],
            ["s1", "S1 · DiT"],
            ["action", "动作执行 / 启动等待"],
          ]
        : [
            ["inference_hold", "推理等待 · 物体运动"],
            ["action", "动作执行 · 物理推进"],
          ];
      for (const [kind, label] of lanes) {
        const events = e.events.filter(
          (v) =>
            (v.kind === kind ||
              (kind === "action" && v.kind === "bootstrap_hold")) &&
            v.start_s < end &&
            (!finite(v.end_s) || v.end_s > start),
        );
        html += `<div class="task-lane" data-stage="${kind === "action" ? "control" : kind}" style="--precision:${color(run.precision)}"><header class="task-lane-label"><strong>${esc(name(run))}</strong><span>${label}</span><small class="task-phase">${kind === "action" ? `${fmt(e.control_freq, 0)} Hz · ${fmt(1000 / e.control_freq, 1)} ms/步` : esc(timingLabel(e))}</small></header><div class="task-rail"><div class="task-blocks">`;
        html += events
          .map((event) => {
            const v = D.resolveEvent(e, event);
            const active =
              v.start_s <= state.clock &&
              (!finite(v.end_s) || state.clock < v.end_s);
            const phase =
              state.clock < v.start_s
                ? "future"
                : active
                  ? "running"
                  : "completed";
            const selected =
              selectedIds.includes(v.request_id) ||
              (state.selected?.columnKey === row.columnKey &&
                state.selected.event.id === v.id);
            const publication = v.source?.publication;
            const evidence = publication
              ? ` · 入队 ${publication.installed_slots ?? "—"} / 过期 ${publication.expired_slots ?? "—"} / 保护 ${publication.protected_slots ?? "—"}`
              : "";
            return `<button class="task-block${selected ? " source-selected" : ""}${v.fallback ? " dynamic-fallback" : ""}" data-event="${esc(v.id)}" data-column="${esc(row.columnKey)}" data-run="${esc(run.id)}" data-phase="${phase}" data-phase-kind="${kind === "action" ? "execution" : "inference"}" style="left:${pct(v.start_s)}%;width:${(100 * ((v.end_s ?? end) - v.start_s)) / width}%" title="${esc(`${eventLabel(v)} · ${dependencyLabel(v)} · ${fmt(v.start_s, 6)}–${fmt(v.end_s, 6)} sim s${evidence}`)}"><strong>${esc(eventLabel(v))}</strong><small>${esc(dependencyLabel(v))}</small></button>`;
          })
          .join("");
        html += `</div><div class="task-playhead" style="left:${pct(state.clock)}%"></div></div></div>`;
      }
    }
    $("timeline").innerHTML = html;
  }
  function renderWall() {
    const panels = state.rows.flatMap((row) => {
      if (state.selected && state.selected.columnKey !== row.columnKey)
        return [];
      const request = state.selected
        ? row.episode.requests.find((r) => r.id === state.selected.requestId)
        : D.at(row.episode, state.clock).request;
      return [{ ...row, request, panel: D.wallPanel(request) }];
    });
    const key = JSON.stringify(panels.map((p) => [p.columnKey, p.request?.id]));
    if (key === state.panelKey) return;
    state.panelKey = key;
    $("curve").innerHTML = panels
      .map(({ run, episode: e, request: r, panel: p }) => {
        if (!r)
          return `<p class="placeholder">${esc(name(run))} · 当前没有已关联的模型请求${state.selected?.event.fallback ? "（fallback 动作）" : ""}。</p>`;
        const stageMarkup = p.stages
          .map(
            (s, i) =>
              `<div class="dynamic-wall-stage"><span>${esc(s.name)}${D.stageNfe(r, s) !== null ? ` · ${D.stageNfe(r, s)} NFE` : ""}</span><div class="dynamic-wall-rail">${finite(s.start_ms) && finite(s.end_ms) && p.end_ms > 0 ? `<span style="left:${(100 * s.start_ms) / p.end_ms}%;width:${(100 * (s.end_ms - s.start_ms)) / p.end_ms}%;background:${i ? "#766eab" : color(run.precision)}"></span>` : finite(s.duration_ms) ? "仅有实测时长；无绝对边界" : "未记录"}</div><strong>${fmt(s.duration_ms, 2)} ms</strong></div>`,
          )
          .join("");
        return `<section class="dynamic-wall" data-request-id="${esc(r.id)}"><h3>${esc(name(run))} · request ${r.request_sequence} · ${streaming(e) ? esc(r.lane.toUpperCase()) : "Fresh VLM → Flow DiT"}</h3><p>${esc(D.observationLabel(r))} · sim ${fmt(r.sim_time_s, 3)} s · RPC ${fmt(r.rpc_duration_ms, 2)} ms；${esc(timingLabel(e))}</p>${stageMarkup}${streaming(e) ? `<p>依赖 ${r.parent_request_id ? esc(e.requests.find((v) => v.id === r.parent_request_id)?.lane === "bootstrap" ? "Bootstrap feature 0" : `S2 feature ${r.feature_sequence}`) : "无上游请求"} · ${esc(r.normal_or_recovery ?? "")} · valid [${r.server_audit?.valid_start_tick ?? "—"}, ${r.server_audit?.valid_end_tick ?? "—"}) · 入队 ${r.publication?.installed_slots ?? "—"} / 过期 ${r.publication?.expired_slots ?? "—"} / 保护 ${r.publication?.protected_slots ?? "—"}<br>特征 AoI ${fmt(ms(r.feature_age_s), 1)} ms · 请求 delay ${r.delay_ticks ?? "—"} ticks；${e.timing_mode === "native" ? "本次完整调用实测" : "逻辑服务时长"} ${fmt(ms(r.logical_duration_s), 1)} ms。RPC / server wall 与逻辑时钟分别记录。</p>` : ""}<p class="small-note">独立 wall 轴：0–${fmt(p.end_ms, 2)} ms；原点 monotonic ${fmt(p.origin_s, 6)} s。${esc(streaming(e) || e.timing_mode === "native" ? D.rpcScope(e) : (r.server_audit?.timing_scope ?? "测量边界未记录"))}</p><table><thead><tr><th>Stage</th><th>Wall start / s</th><th>Wall end / s</th></tr></thead><tbody>${p.stages.map((s) => `<tr><td>${esc(s.name)}</td><td>${fmt(s.wall_started_s, 6)}</td><td>${fmt(s.wall_completed_s, 6)}</td></tr>`).join("")}</tbody></table></section>`;
      })
      .join("");
  }
  function renderMotion() {
    $("projection").innerHTML = state.rows
      .map(({ run, episode: e }) => {
        const sample = e.motion_events
          .filter((v) => v.rollout_sim_time_s <= state.clock)
          .at(-1);
        const targets = sample?.target_object_poses ?? {};
        return `<div class="provenance-lines"><strong>${esc(name(run))}</strong> · sim ${fmt(sample?.rollout_sim_time_s, 3)} s · ${esc(sample?.kind ?? "无记录")}<table><thead><tr><th>Target</th><th>X / m</th><th>Y / m</th><th>Z / m</th></tr></thead><tbody>${Object.entries(
          targets,
        )
          .map(
            ([name, p]) =>
              `<tr><td>${esc(name)}</td>${[0, 1, 2].map((i) => `<td>${fmt(p.body_pos?.[i], 4)}</td>`).join("")}</tr>`,
          )
          .join(
            "",
          )}</tbody></table><p>物体位姿来自物理记录；推理等待期间机器人保持，目标继续运动。source_tick 只计已执行动作，排除等待子步。</p></div>`;
      })
      .join("");
  }
  function renderEvents() {
    const events = state.rows.flatMap((row) =>
      row.episode.events
        .filter((e) => Math.abs(e.start_s - state.clock) <= 0.75)
        .map((event) => ({ row, event: D.resolveEvent(row.episode, event) })),
    );
    $("event-count").textContent = `游标附近 ${events.length} 条`;
    $("event-list").innerHTML = events
      .map(
        ({ row, event: e }) =>
          `<button class="event-row" data-event="${esc(e.id)}" data-run="${esc(row.run.id)}" data-column="${esc(row.columnKey)}">${esc(name(row.run))} · ${fmt(e.start_s, 3)} sim s · ${esc(eventLabel(e))} · ${esc(dependencyLabel(e))}</button>`,
      )
      .join("");
  }
  function render() {
    $("seek").value = state.clock;
    $("clock").textContent =
      `${fmt(state.clock, 3)} / ${fmt(state.end, 2)} sim s`;
    $("play").textContent = state.playing ? "暂停" : "播放机器人视频";
    const cards = $("precision-cards").querySelectorAll(".precision-card");
    state.rows.forEach(({ episode: e }, i) => {
      const now = D.at(e, state.clock, $("camera").value),
        frame = now.frame;
      window.ReplayMedia.update(
        cards[i],
        e.media[$("camera").value],
        frame ? frame.media_time_s + 0.0001 : null,
        "物理仿真采样 · 未生成插值帧",
        {
          frames: D.cameraFrames(e, $("camera").value),
          cursor: state.clock,
          ended: now.ended,
          timeLabel: "仿真采样",
        },
        state.epoch,
      );
      const done = e.events.filter(
        (v) =>
          v.kind === "action" &&
          !v.partial &&
          finite(v.end_s) &&
          v.end_s <= state.clock,
      ).length;
      const metrics = [
        ["当前物理阶段", D.phaseLabel(now, e)],
        ["已执行动作", `${done} / ${e.steps ?? "—"}`],
        ["真实请求 RPC", `${fmt(now.request?.rpc_duration_ms, 1)} ms`],
        [
          "视频 / 控制",
          `${fmt(e.video_fps, 0)} fps / ${fmt(e.control_freq, 0)} Hz`,
        ],
      ];
      if (streaming(e))
        metrics.push(
          ["动作视觉 AoI", `${fmt(ms(now.event?.feature_age_s), 1)} ms`],
          ["动作状态 AoI", `${fmt(ms(now.event?.state_age_s), 1)} ms`],
        );
      cards[i].querySelector(".metrics").innerHTML = metrics
        .map(
          ([name, value]) =>
            `<div class="metric"><small>${esc(name)}</small><strong>${esc(value)}</strong></div>`,
        )
        .join("");
    });
    renderTimeline();
    renderWall();
    renderMotion();
    renderEvents();
  }
  function tick(now) {
    if (state.playing && state.last != null) {
      state.clock = Math.min(
        state.end,
        state.clock + ((now - state.last) / 1000) * Number($("speed").value),
      );
      if (state.clock >= state.end) state.playing = false;
      render();
    }
    state.last = now;
    requestAnimationFrame(tick);
  }
  async function initialize() {
    document.body.dataset.view = "dynamic";
    for (const id of [
      "record-label",
      "scope-label",
      "execution-clock-label",
      "execution-coupling-label",
      "controller-policy-label",
      "execution-prefix-label",
      "async-overlap-label",
      "kernel-details",
      "stream-details",
    ])
      $(id).hidden = true;
    $("follow-latest").disabled = true;
    $("stage").closest("label").hidden = true;
    $("category").closest("label").hidden = true;
    $("channel").closest("label").hidden = true;
    $("mode-dynamic").setAttribute("aria-pressed", "true");
    $("mode-model").onclick = () => navigate("model");
    $("mode-deployment").onclick = () => navigate("deployment");
    $("speed").value = "1";
    $("sample-label").textContent = "动态场景 / 速度";
    option(
      $("camera"),
      [
        ["image", "主相机 + 腕部 · 双镜头"],
        ["wrist_image", "腕部相机"],
      ],
      "image",
    );
    $("evidence-note").textContent =
      "LIBERO-Dynamic：主时间轴采用物理仿真时间，包含推理等待期间目标物体的运动。模型使用本项目已有 GR00T 权重。各配置注明逐请求实测或固定延迟；Flow4 每次等待后执行 K 步，πR² 按 S2/S1 逻辑通道调度。实际 GPU RPC 串行计算，测量边界在独立 wall 面板展示。";
    $("timeline-note").textContent =
      "Hold 是应用到物理世界的 τ；执行动作按真实物理边界显示。视频逐帧依照 sample_sim_time_s 映射，60 fps 视频与 20 Hz 控制分别计数。πR² 首次启动等待付费，随后动作每步推进；点击 S1 关联 S2，点击动作关联实际生产请求；高亮只表示记录中的依赖。";
    document.querySelector(".timeline-section h2").textContent =
      "LIBERO-Dynamic · 物理时间与动作来源";
    $("curve-title").textContent = "真实 VLM → DiT · 独立 Wall 时间";
    $("projection-title").textContent = "移动目标 · 实测物理位姿";
    $("curve-note").textContent =
      "RPC 为实际客户端往返；server 区间按记录边界展示，未提供绝对边界时只显示时长。原生 Flow4 的完整策略调用含结果规范化，其逐请求耗时直接决定该次 Hold；固定延迟配置另行标注，绝对 wall 时间不与物理时间混算。";
    for (const id of ["sample", "precision"])
      $(id).addEventListener("change", configure);
    for (const id of ["camera", "window"])
      $(id).addEventListener("change", render);
    $("play").onclick = () => {
      if (state.clock >= state.end) state.clock = 0;
      state.playing = !state.playing;
      state.last = null;
      render();
    };
    $("seek").oninput = () => {
      state.clock = Number($("seek").value);
      state.playing = false;
      render();
    };
    for (const [id, direction] of [
      ["previous", -1],
      ["next", 1],
    ])
      $(id).onclick = () => {
        const times = [
          ...new Set(
            state.rows.flatMap((r) => r.episode.events.map((e) => e.start_s)),
          ),
        ].sort((a, b) => a - b);
        state.clock =
          direction < 0
            ? (times.filter((t) => t < state.clock - 1e-6).at(-1) ?? 0)
            : (times.find((t) => t > state.clock + 1e-6) ?? state.end);
        state.playing = false;
        render();
      };
    for (const id of ["timeline", "event-list"])
      $(id).addEventListener("click", (event) => {
        const target = event.target.closest("[data-event]");
        if (!target) return;
        const row = state.rows.find(
            (r) => r.columnKey === target.dataset.column,
          ),
          item = row?.episode.events.find((e) => e.id === target.dataset.event);
        if (item) selectRequest(row, item);
      });
    $("timeline-follow").onclick = () => {
      state.selected = null;
      state.panelKey = null;
      render();
    };
    $("inspect-run").onclick = () =>
      inspect(
        state.rows.map(({ run, episode }) => ({
          run_id: run.id,
          configuration: run.configuration,
          time_contract: run.time_contract,
          provenance: run.provenance,
          episode,
        })),
      );
    $("download").onclick = () => {
      const url = URL.createObjectURL(
        new Blob(
          [
            JSON.stringify(
              {
                view: "dynamic",
                clock: "rollout_sim_time_s",
                cursor_s: state.clock,
                selected: state.selected,
                rows: state.rows,
              },
              null,
              2,
            ),
          ],
          { type: "application/json" },
        ),
      );
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "dynamic-replay-evidence.json";
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    };
    try {
      const response = await fetch("/api/replay/catalog");
      if (!response.ok) throw new Error(`Catalog HTTP ${response.status}`);
      const catalog = await response.json();
      if (catalog.schema !== "pir2.replay.catalog" || catalog.version !== 1)
        throw new Error("不支持的 catalog");
      const descriptors = (catalog.runs ?? []).filter(
        (r) => r.kind === "dynamic",
      );
      state.runs = await Promise.all(
        descriptors.map(async (descriptor) => {
          const result = await fetch(
            `/api/replay/run/${encodeURIComponent(descriptor.id)}`,
          );
          if (!result.ok) throw new Error(`Run HTTP ${result.status}`);
          const run = await result.json();
          if (
            run.id !== descriptor.id ||
            run.kind !== "dynamic" ||
            run.precision !== descriptor.precision
          )
            throw new Error("Dynamic run identity mismatch");
          return run;
        }),
      );
      option(
        $("sample"),
        D.conditions(state.runs).map(({ key, episode }) => [
          key,
          conditionLabel(episode),
        ]),
      );
      option(
        $("precision"),
        [
          ["all", "已记录精度 · 同条件并排"],
          ...[...new Set(state.runs.map((r) => r.precision))].map((p) => [
            p,
            p.toUpperCase(),
          ]),
        ],
        "all",
      );
      $("data-status").textContent = `${state.runs.length} 个动态配置`;
      $("data-status").dataset.state = "ready";
      $("catalog-meta").textContent =
        `LIBERO-Dynamic · ${catalog.created_at ?? ""}`;
      configure();
      requestAnimationFrame(tick);
    } catch (error) {
      $("error-banner").hidden = false;
      $("error-banner").textContent = error.message;
      $("data-status").textContent = "动态证据加载失败";
    }
  }
  return { initialize };
})();
