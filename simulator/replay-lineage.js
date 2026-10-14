/* Pure source lineage, using the measured queue's S2/S1 dependency aliases. */
(function (root, factory) {
  if (typeof module === "object" && module.exports)
    module.exports = factory(
      require("./replay-model.js"),
      require("./measured-model.js"),
    );
  else root.ReplayLineage = factory(root.QuantReplay, root.MeasuredReplay);
})(typeof globalThis !== "undefined" ? globalThis : this, function (R, Q) {
  "use strict";
  const definitions = [
    ["observation", "实际观测"],
    ["vlm", "S2 · VLM"],
    ["cache", "视觉特征缓存"],
    ["action", "S1 · DiT"],
    ["output", "动作输出 / 缓冲"],
    ["control", "执行控制动作"],
  ];
  const time = (value) =>
    R.finite(value) ? value.toFixed(3) + " s" : "时间未记录";
  function empty() {
    return {
      nodes: definitions.map(([key, title]) => ({
        key,
        title,
        known: false,
        lines: ["来源未记录"],
      })),
      edges: [],
    };
  }
  function deployment(episode, cursor, selected = null) {
    const graph = empty(),
      q = episode?.queue;
    if (!q) return graph;
    const requests = q.requests ?? [],
      ticks = episode.ticks ?? [];
    const aliases = Q.dependencies(q, 0, q.duration_s ?? Infinity);
    let tick = selected?.tick ?? R.lastAt(ticks, cursor, "relative_s");
    let action = selected?.request?.role === "S1" ? selected.request : null;
    let vision = selected?.request?.role === "S2" ? selected.request : null;
    if (vision)
      action =
        requests
          .filter(
            (r) =>
              r.role === "S1" &&
              r.feature_sequence === vision.feature_sequence &&
              r.start_s <= cursor,
          )
          .at(-1) ?? null;
    if (
      !action &&
      !vision &&
      !tick?.fallback &&
      !tick?.executed_slot?.fallback &&
      tick?.executed_slot?.producer_kind === "request"
    )
      action =
        requests.find(
          (r) =>
            r.role === "S1" &&
            r.request_tick === tick.executed_slot.producer_request_tick,
        ) ?? null;
    if (selected?.request)
      tick = action
        ? (ticks
            .filter(
              (t) =>
                t.relative_s <= cursor &&
                t.executed_slot?.producer_kind === "request" &&
                t.executed_slot?.producer_request_tick === action.request_tick,
            )
            .at(-1) ?? null)
        : null;
    const bootstrap =
      !tick?.fallback &&
      !tick?.executed_slot?.fallback &&
      tick?.executed_slot?.producer_kind === "bootstrap";
    const sequence =
      action?.feature_sequence ??
      vision?.feature_sequence ??
      (bootstrap ? q.initial_feature?.feature_sequence : null);
    if (sequence != null && !vision)
      vision =
        requests.find(
          (r) => r.role === "S2" && r.feature_sequence === sequence,
        ) ?? null;
    const publication = (q.feature_updates ?? []).find(
      (f) => f.feature_sequence === sequence,
    );
    const pub = publication?.t ?? vision?.published_s;
    const initial =
      sequence != null && sequence === q.initial_feature?.feature_sequence;
    const capture =
      action?.feature_capture_s ??
      vision?.capture_s ??
      (initial ? q.initial_feature?.feature_capture_s : null);
    const sourceTick =
      action?.feature_source_tick ??
      vision?.source_tick ??
      (initial ? q.initial_feature?.feature_source_tick : null);
    const set = (key, values) =>
      Object.assign(
        graph.nodes.find((n) => n.key === key),
        values,
      );
    const edge = (from, to, basis) => graph.edges.push({ from, to, basis });
    graph.featureSequence = sequence;
    graph.request = action;
    if (R.finite(capture) && sourceTick != null)
      set("observation", {
        known: true,
        lines: [`image tick ${sourceTick}`, `采集 ${time(capture)}`],
        observation_id: vision?.observation_id ?? null,
        source_tick: sourceTick,
        capture_s: capture,
      });
    if (vision) {
      const parent = aliases.parents.get(sequence);
      set("vlm", {
        known: true,
        title: parent?.label ?? "S2 · VLM",
        lines: [
          `VLM · ${time(vision.start_s)} → ${time(vision.end_s)}`,
          `feature ${sequence}`,
        ],
        request: vision,
      });
      if (R.finite(capture) && sourceTick != null)
        edge("observation", "vlm", "Recorded S2 source_tick/capture_s");
    } else if (initial)
      set("vlm", {
        title: "S2 · bootstrap",
        lines: ["初始化视觉特征", "未记录该次 VLM 区间"],
      });
    const published = R.finite(pub) && pub <= cursor;
    if (sequence != null)
      set("cache", {
        known: published || initial,
        lines: [
          `feature ${sequence}`,
          initial
            ? "初始缓存 · 发布时间未知"
            : published
              ? `发布 ${time(pub)}`
              : "尚未发布 / 时间未知",
        ],
        feature_sequence: sequence,
        published_s: pub,
        feature_id: action?.feature_id ?? vision?.feature_id,
      });
    if (vision && published && R.finite(vision.end_s) && vision.end_s <= pub)
      edge(
        "vlm",
        "cache",
        "Recorded publication and matching feature_sequence",
      );
    if (action) {
      const alias = aliases.actions.get(action.request_tick);
      set("action", {
        known: true,
        title: alias?.label ?? "S1 · DiT",
        lines: [
          `DiT · 读取 ${time(action.use_s)}`,
          `state tick ${action.state_tick ?? "?"} · ${time(action.state_capture_s)}`,
        ],
        request: action,
      });
      if (
        R.finite(action.use_s) &&
        action.use_s <= cursor &&
        (initial || (published && pub <= action.use_s))
      )
        edge(
          "cache",
          "action",
          "Actual consumed feature_sequence at recorded use_s",
        );
    }
    const completed = action?.completed_s;
    if (R.finite(completed) && completed <= cursor) {
      set("output", {
        known: true,
        lines: [
          `完成 ${time(completed)}`,
          R.finite(action.adopted_s) && action.adopted_s <= cursor
            ? `接纳 ${time(action.adopted_s)}`
            : "尚未接纳 / 接纳时间未知",
        ],
        request: action,
      });
      edge("action", "output", "Actual request completion");
    } else if (bootstrap)
      set("output", {
        known: true,
        title: "Bootstrap 动作",
        lines: ["初始化动作缓冲", "具体模型区间未记录"],
      });
    if (tick && tick.relative_s <= cursor) {
      set("control", {
        known: true,
        lines: [
          `tick ${tick.tick} · ${time(tick.relative_s)}`,
          tick.fallback ? "fallback · 无模型来源" : "实际执行命令",
        ],
        tick,
      });
      if (
        !tick.fallback &&
        !tick.executed_slot?.fallback &&
        (bootstrap ||
          (action && R.finite(completed) && completed <= tick.relative_s))
      )
        edge("output", "control", "Actual executed_slot producer identity");
    }
    return graph;
  }
  function model(run, record, cursor) {
    const graph = empty();
    if (!record) return graph;
    const set = (key, values) =>
      Object.assign(
        graph.nodes.find((n) => n.key === key),
        values,
      );
    set("observation", {
      known: Boolean(record.observation_id),
      title: record.observation_id ? "固定真实输入" : "输入身份未知",
      lines: [
        `Obs ${record.observation ?? "?"}`,
        `seed ${record.seed ?? "未知"}`,
      ],
      observation_id: record.observation_id,
    });
    const scope = R.scopeOf(run, record);
    if (!["policy", "prepared_model", "preparation_model"].includes(scope)) {
      const preparation = scope === "preparation_transfer";
      set("vlm", {
        title: "VLM · 未执行 / 未知",
        lines: [preparation ? "输入准备不执行模型" : "边界未知，不推断执行"],
      });
      set("action", {
        title: "DiT · 未执行 / 未知",
        lines: [preparation ? "输入准备不执行模型" : "边界未知，不推断执行"],
      });
      set("cache", {
        known: preparation,
        title: preparation ? "输入准备 / 传输" : "边界未知",
        lines: [R.scopeInfo(scope).label],
      });
      set("output", {
        known: preparation,
        title: preparation ? "已准备输入" : "原始记录",
        lines: [preparation ? "不产生模型动作" : "检查该记录的原始输出"],
      });
      set("control", { title: "未执行机器人", lines: ["离线测量记录"] });
      return graph;
    }
    const profile = record.instrumented === true;
    const stages = Object.fromEntries(
      ["vlm", "action"].map((stage) => {
        const events = (record.timeline ?? []).filter((e) => e.stage === stage);
        return [
          stage,
          events.length
            ? {
                start: Math.min(...events.map((e) => e.start_ms)),
                end: Math.max(...events.map((e) => e.start_ms + e.duration_ms)),
                count: events.length,
              }
            : null,
        ];
      }),
    );
    for (const [key, name] of [
      ["vlm", "VLM"],
      ["action", "DiT / Action"],
    ]) {
      const stage = stages[key];
      set(key, {
        known: true,
        title: name,
        lines: stage
          ? [
              `${stage.start.toFixed(3)} → ${stage.end.toFixed(3)} ms`,
              `${stage.count} 设备事件 · ${cursor < stage.start ? "未开始" : cursor < stage.end ? "执行中" : "已结束"}`,
            ]
          : ["请求内的实际模型路径", "分段时间未记录"],
        stage: key,
        request: record,
      });
    }
    set("cache", {
      known: true,
      title: "本请求视觉特征",
      lines: ["VLM → Action 张量依赖", "张量 / 交接时间未单独记录"],
    });
    const output = R.outputAt(record, cursor);
    set("output", {
      known: output.status === "recorded",
      title: "Decoded actions",
      lines: [
        output.status === "recorded"
          ? "本请求动作已记录"
          : output.status === "pending"
            ? "请求尚未完成"
            : "本请求未保留动作",
        profile ? "Profile seed 独立" : "Clean retained output",
      ],
      request: record,
    });
    set("control", {
      title: "未执行机器人",
      lines: ["独立模型调用", "切换机器人视图看闭环"],
    });
    if (
      (run.boundaries?.[scope] || record.instrumented) &&
      record.observation_id
    )
      for (const [from, to] of [
        ["observation", "vlm"],
        ["vlm", "cache"],
        ["cache", "action"],
        ["action", "output"],
      ])
        graph.edges.push({
          from,
          to,
          basis:
            "Recorded full VLM + action prepared-model invocation; component handoff timestamps are not inferred",
          declared: true,
        });
    return graph;
  }
  return { deployment, model };
});
