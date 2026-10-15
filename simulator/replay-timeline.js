/* Measured task blocks: timestamps move the viewport, never an estimated GPU rate. */
(function (root, factory) {
  if (typeof module === "object" && module.exports)
    module.exports = factory(
      require("./measured-model.js"),
      require("./replay-model.js"),
    );
  else root.ReplayTimeline = factory(root.MeasuredReplay, root.QuantReplay);
})(typeof globalThis !== "undefined" ? globalThis : this, function (Q, R) {
  "use strict";
  const finite = (v) => typeof v === "number" && Number.isFinite(v);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const names = { vlm: "VLM / S2", action: "DiT / S1" };

  function basisLabel(basis) {
    return (
      {
        rpc: "完整请求区间（含准备/通信）",
        server_stage: "服务端分段（非独立 RPC，非 GPU kernel 时间）",
        inference_rpc: "完整同步推理 RPC · 控制墙钟 · 仿真暂停",
        execution_phase: "动作执行段 · 控制墙钟（包含执行段节拍等待）",
        gpu_span: "实际 CUDA 首末事件跨度（含间隙）",
        host_submit: "主机提交区间",
        host_observed_cuda_completion: "主机观察 CUDA 完成",
        cuda_event: "CUDA event 区间",
      }[basis] ?? "计时边界未记录"
    );
  }

  function phase(block, cursor) {
    if (cursor < block.start)
      return { key: "future", label: "尚未开始", progress: 0 };
    const submitting = block.basis === "host_submit";
    if (
      (!finite(block.end) || cursor >= block.end) &&
      ["failed", "error", "cancelled", "dropped", "rejected"].includes(
        block.status,
      )
    )
      return {
        key: "failed",
        label: submitting ? "主机提交失败 / 未完成" : "失败 / 未完成",
        progress: finite(block.end) ? 1 : null,
      };
    if (!finite(block.end))
      return {
        key: block.status === "running" ? "running" : "unknown",
        label:
          block.status === "running"
            ? submitting
              ? "主机提交中 · 结束待记录"
              : "运行中 · 结束待记录"
            : "结束时间未记录",
        progress: null,
      };
    if (cursor < block.end)
      return {
        key: "running",
        label: submitting ? "主机提交中" : "运行中",
        progress: clamp(
          (cursor - block.start) / (block.end - block.start),
          0,
          1,
        ),
      };
    if (["incomplete", "unknown"].includes(block.status))
      return {
        key: "unknown",
        label: block.status === "unknown" ? "结果未知" : "未完成",
        progress: null,
      };
    if (submitting)
      return {
        key: "submitted",
        label: "主机已提交 · GPU 完成未知",
        progress: 1,
      };
    if (finite(block.adopted) && block.adopted <= cursor)
      return { key: "adopted", label: "动作已接纳", progress: 1 };
    if (block.consumed?.some((t) => finite(t) && t <= cursor))
      return { key: "consumed", label: "特征已使用", progress: 1 };
    return { key: "completed", label: "已完成", progress: 1 };
  }

  function viewport(cursor, end, requestedSpan) {
    const span = Math.max(0.001, finite(requestedSpan) ? requestedSpan : end);
    const left = clamp(cursor - span * 0.42, 0, Math.max(0, end - span));
    return { start: left, end: left + span, span };
  }

  function geometry(block, view) {
    const stop = finite(block.end) ? block.end : view.end;
    return {
      left: ((block.start - view.start) / view.span) * 100,
      width: (Math.max(0, stop - block.start) / view.span) * 100,
      visible: stop >= view.start && block.start <= view.end,
      open: !finite(block.end),
    };
  }

  function deployment(row) {
    const queue = row.episode?.queue;
    const requests = queue?.requests ?? [];
    const aliases = queue
      ? Q.dependencies(queue, 0, queue.duration_s ?? Infinity)
      : null;
    const synchronous = R.deploymentCoupling(row.run) === "synchronous";
    const taskTracks = ["vlm", "action"].map((stage) => ({
      id: row.run.id + ":" + stage,
      precision: row.run.precision,
      stage,
      title: names[stage],
      unit: "s",
      basis: synchronous ? "server_stage" : "rpc",
      clockId: "control_wall",
      unavailable: !queue ? "本运行未记录模块时序" : null,
      blocks: requests
        .filter(
          (r) =>
            r.role === (stage === "vlm" ? "S2" : "S1") && finite(r.start_s),
        )
        .map((r) => {
          const parent = aliases.parents.get(r.feature_sequence);
          const label =
            stage === "vlm"
              ? (parent?.label ?? "S2[?]")
              : (aliases.actions.get(r.request_tick)?.label ?? "S1[?]");
          const end = finite(r.end_s)
            ? r.end_s
            : finite(r.completed_s)
              ? r.completed_s
              : null;
          return {
            id: r.id,
            label,
            stage,
            start: r.start_s,
            end,
            unit: "s",
            basis: r.timing_scope?.startsWith("server_")
              ? "server_stage"
              : synchronous
                ? "unknown"
                : "rpc",
            timingScope: r.timing_scope ?? null,
            clockId: "control_wall",
            status: r.status ?? (end == null ? "incomplete" : "completed"),
            featureSequence: r.feature_sequence ?? null,
            sourceLabel: stage === "action" ? (parent?.label ?? null) : null,
            sourceId: stage === "action" ? (parent?.request?.id ?? null) : null,
            adopted: r.adopted_s,
            consumed:
              stage === "vlm"
                ? requests
                    .filter(
                      (a) =>
                        a.role === "S1" &&
                        a.feature_sequence != null &&
                        a.feature_sequence === r.feature_sequence,
                    )
                    .map((a) => a.use_s)
                : [],
            event: {
              id: r.id,
              kind: r.timing_scope?.startsWith("server_")
                ? "server_stage"
                : "rpc",
              precision: row.run.precision,
              run: row.run,
              episode: row.episode,
              request: r,
            },
          };
        }),
    }));
    if (synchronous || row.episode?.phases?.length) {
      const blocks = (row.episode?.phases ?? [])
        .filter(
          (p) =>
            ["inference", "execution"].includes(p.phase) &&
            finite(p.start_s) &&
            (p.end_s == null || (finite(p.end_s) && p.end_s >= p.start_s)),
        )
        .map((p) => {
          // Recorded producer identity is authoritative. Temporal proximity is
          // insufficient: another completed DiT may not supply these actions.
          const producers =
            p.phase === "execution" && p.request_tick != null
              ? requests.filter(
                  (r) => r.role === "S1" && r.request_tick === p.request_tick,
                )
              : [];
          const producer = producers.length === 1 ? producers[0] : null;
          const action = producer
            ? taskTracks[1].blocks.find((b) => b.id === producer.id)
            : null;
          return {
            id: p.id,
            label: `Cycle ${p.cycle ?? "?"} · ${p.phase === "inference" ? "Infer" : "Execute"}`,
            stage: "phase",
            phaseKind: p.phase,
            start: p.start_s,
            end: p.end_s ?? null,
            unit: "s",
            basis:
              p.phase === "inference" ? "inference_rpc" : "execution_phase",
            clockId: "control_wall",
            status: p.status ?? (p.end_s == null ? "incomplete" : "completed"),
            sourceId: action?.id ?? null,
            sourceLabel: action?.label ?? null,
            description:
              p.phase === "inference"
                ? "Simulation paused"
                : "Executing actions",
            event: {
              id: p.id,
              kind: "phase",
              precision: row.run.precision,
              run: row.run,
              episode: row.episode,
              phase: p,
              source_request: producer,
            },
          };
        });
      taskTracks.push({
        id: row.run.id + ":phase",
        precision: row.run.precision,
        stage: "phase",
        title: "Inference / Execution",
        unit: "s",
        basis: "control_wall",
        clockId: "control_wall",
        blocks,
        unavailable: blocks.length ? null : "未记录推理 / 执行阶段",
      });
    }
    return taskTracks;
  }

  function sourceBlocks(tracks, selectedId) {
    const blocks = new Map(
      tracks.flatMap((t) => t.blocks.map((b) => [b.id, b])),
    );
    const visited = new Set([selectedId]),
      sources = [];
    let sourceId = blocks.get(selectedId)?.sourceId;
    while (sourceId != null && !visited.has(sourceId)) {
      visited.add(sourceId);
      const block = blocks.get(sourceId);
      if (!block) break;
      sources.push(block);
      sourceId = block.sourceId;
    }
    return sources;
  }

  function model(row) {
    const record = row.record;
    const scope = R.scopeOf(row.run, record);
    const neural = ["prepared_model", "policy", "preparation_model"].includes(
      scope,
    );
    const gpuTimeline =
      neural && record?.instrumented === true
        ? (record.timeline ?? []).filter(
            (x) =>
              finite(x.start_ms) && finite(x.duration_ms) && x.duration_ms >= 0,
          )
        : [];
    return ["vlm", "action"].map((stage) => {
      const intervals = gpuTimeline.filter((x) => x.stage === stage);
      // A canonical GPU trace has its own origin. Never fill its gaps with
      // host submission spans anchored to a different clock.
      const actualStages =
        neural && !gpuTimeline.length
          ? (record?.stage_spans ?? []).filter(
              (x) =>
                x.stage === stage &&
                finite(x.start_ms) &&
                (x.end_ms == null ||
                  (finite(x.end_ms) && x.end_ms >= x.start_ms)) &&
                [
                  "host_submit",
                  "host_observed_cuda_completion",
                  "cuda_event",
                ].includes(x.domain),
            )
          : [];
      const blocks = actualStages.length
        ? actualStages.map((x, i) => ({
            id: record.id + "/" + stage + "/" + i,
            label: stage === "vlm" ? "VLM / S2" : "DiT / S1",
            stage,
            start: x.start_ms,
            end: x.end_ms ?? null,
            unit: "ms",
            basis: x.domain,
            clockId: x.clock_id ?? null,
            status: x.status ?? (x.end_ms == null ? "incomplete" : "completed"),
            sourceLabel: x.feature_id ? "feature " + x.feature_id : null,
            event: {
              kind: "stage",
              run: row.run,
              record,
              stage,
              time: x.start_ms,
              end: x.end_ms ?? null,
              domain: x.domain,
              clock_id: x.clock_id ?? null,
              status: x.status ?? null,
            },
          }))
        : intervals.length
          ? [
              {
                id: record.id + "/" + stage,
                label: stage === "vlm" ? "VLM / S2" : "DiT / S1",
                stage,
                start: Math.min(...intervals.map((x) => x.start_ms)),
                end: Math.max(
                  ...intervals.map((x) => x.start_ms + x.duration_ms),
                ),
                unit: "ms",
                basis: "gpu_span",
                sourceLabel: null,
                kernels: intervals,
                event: { kind: "stage", run: row.run, record, stage },
              },
            ]
          : [];
      return {
        id: row.run.id + ":" + stage,
        precision: row.run.precision,
        stage,
        title: names[stage],
        unit: "ms",
        blocks,
        unavailable: blocks.length
          ? null
          : !record
            ? "缺少匹配请求"
            : !neural
              ? "该边界不包含神经推理"
              : "未采集该次模块时间 · 切换 Profile 查看",
      };
    });
  }

  function tracks(rows, kind) {
    return rows.flatMap((row) =>
      kind === "deployment" ? deployment(row) : model(row),
    );
  }

  function observedModelEnd(record) {
    const observed = [
      record?.host_progress_ms,
      record?.host_wall_ms,
      record?.wall_ms,
      ...(record?.stage_spans ?? []).flatMap((x) => [x.start_ms, x.end_ms]),
      ...(record?.timeline ?? []).flatMap((x) => [
        x.start_ms,
        finite(x.start_ms) && finite(x.duration_ms) && x.duration_ms >= 0
          ? x.start_ms + x.duration_ms
          : null,
      ]),
    ];
    // This is a viewport extent, never a completed request duration.
    return observed.reduce(
      (end, value) => (finite(value) ? Math.max(end, value) : end),
      0,
    );
  }

  function laneState(track, cursor) {
    if (track.unavailable)
      return { key: "unknown", label: "未记录", block: null };
    const running = track.blocks.find(
      (b) => b.start <= cursor && (!finite(b.end) || cursor < b.end),
    );
    if (running) return { ...phase(running, cursor), block: running };
    const completed = track.blocks
      .filter((b) => finite(b.end) && b.end <= cursor)
      .sort((a, b) => b.end - a.end)[0];
    return completed
      ? { ...phase(completed, cursor), block: completed }
      : { key: "idle", label: "等待首个任务", block: null };
  }

  return {
    tracks,
    phase,
    viewport,
    geometry,
    laneState,
    observedModelEnd,
    basisLabel,
    sourceBlocks,
  };
});
