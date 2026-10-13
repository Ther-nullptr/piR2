"""Optional Chinese contention plots; matplotlib is imported only on demand."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .analysis import digest, merged
from .labels import COLORS, LATENCY_FIELDS, MODE_NAMES


def plotting(font=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    if font is not None:
        candidates = [Path(font).expanduser()]
    else:
        candidates = [
            Path(entry.fname)
            for entry in font_manager.fontManager.ttflist
            if any(
                name in entry.name.casefold()
                for name in ("source han", "noto sans cjk", "noto serif cjk", "思源")
            )
        ]
    required = set(map(ord, "请求响应流跨度硬件测量让出未记录带宽时间并发活动"))
    selected = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            if required <= set(FT2Font(str(candidate)).get_charmap()):
                selected = candidate
                break
        except (OSError, RuntimeError):
            continue
    if selected is None:
        raise RuntimeError(
            "Chinese plots require a Source Han or Noto CJK font with Chinese glyphs. "
            "Install one, pass --font /path/to/font.ttf, or use --no-plots for JSON/CSV/HTML only."
        )
    font_manager.fontManager.addfont(str(selected))
    plt.rcParams.update(
        {
            "font.family": font_manager.FontProperties(fname=selected).get_name(),
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "font.size": 10,
            "axes.titlesize": 12,
        }
    )
    return plt


def save_figure(fig, base):
    files = []
    for suffix in (".png", ".svg"):
        target = base.with_suffix(suffix)
        fig.savefig(target, dpi=180, bbox_inches="tight", facecolor="white")
        files.append({"path": str(target.resolve()), "sha256": digest(target)})
    return files


def timeline(
    nsys, measurement, out, start_ms=0, duration_ms=300, metric_gpu_id=None, font=None
):
    plt = plotting(font)
    kernels = [k for k in nsys["kernels"] if k["component"] != "unassigned"]
    if not kernels:
        return []
    origin = min(
        [k["start_ns"] for k in kernels]
        + [
            request["s1_release_ns"]
            for request in nsys["requests"]
            if request["s1_release_ns"] is not None
        ]
    )
    xmin, xmax = origin + start_ms * 1e6, origin + (start_ms + duration_ms) * 1e6
    end = max(k["end_ns"] for k in kernels)
    xmax = min(xmax, end)
    gpu_ids = {m["gpu_id_from_type"] for m in nsys["metrics"]}
    if metric_gpu_id is None and len(gpu_ids) > 1:
        raise ValueError(
            f"Multiple metric GPUs {gpu_ids}; pass --metric-gpu-id explicitly"
        )
    selected = [
        m
        for m in nsys["metrics"]
        if metric_gpu_id is None or m["gpu_id_from_type"] == metric_gpu_id
    ]
    families = [
        (
            "SM 活动",
            lambda name: (
                "sm active" in name.lower()
                or "sms active" in name.lower()
                or "sm issue" in name.lower()
            ),
        ),
        ("Tensor 活动", lambda name: "tensor active" in name.lower()),
        (
            "DRAM 带宽 counter",
            lambda name: (
                "dram" in name.lower()
                and ("bandwidth" in name.lower() or "throughput" in name.lower())
            ),
        ),
    ]
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.7, 1, 1, 1]},
        constrained_layout=True,
    )
    for level, component in enumerate(("S2", "S1")):
        intervals = [
            (max(k["start_ns"], xmin), min(k["end_ns"], xmax))
            for k in kernels
            if k["component"] == component
            and k["end_ns"] > xmin
            and k["start_ns"] < xmax
        ]
        axes[0].broken_barh(
            [((a - origin) / 1e6, (b - a) / 1e6) for a, b in merged(intervals)],
            (level - 0.26, 0.52),
            facecolors=COLORS[component],
        )
    for request in nsys["requests"]:
        release = request["s1_release_ns"]
        if release is not None and xmin <= release <= xmax:
            axes[0].axvline(
                (release - origin) / 1e6,
                color=COLORS["S1"],
                linestyle=":",
                linewidth=0.8,
                alpha=0.7,
            )
    axes[0].set_yticks([0, 1], ["S2 / VLM", "S1 / DiT"])
    axes[0].set_ylim(1.7, -0.7)
    axes[0].set_title(
        "真实 CUDA kernel 时间线（实线：kernel 并集；竖虚线：DiT release）", loc="left"
    )
    for ax, (title, predicate) in zip(axes[1:], families):
        matches = [m for m in selected if predicate(m["name"])]
        for metric in matches:
            if not metric.get("quality", {}).get("interpretable", True):
                continue
            points = [
                (timestamp, value)
                for timestamp, value in metric["samples"]
                if xmin <= timestamp <= xmax
            ]
            if points:
                ax.plot(
                    [(t - origin) / 1e6 for t, _ in points],
                    [v for _, v in points],
                    linewidth=0.9,
                    label=metric["name"],
                )
        ax.set_title(title + "：整卡采样", loc="left")
        if ax.lines:
            ax.legend(fontsize=8, loc="upper right")
            units = {m["unit"] for m in matches}
            ax.set_ylabel(next(iter(units)) if len(units) == 1 else "见原始 counter 名")
        else:
            ax.text(
                0.5,
                0.5,
                "该硬件 counter 原始单位或范围异常"
                if matches
                else "本次未取得该硬件 counter",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("相对首个测量事件的时间（ms）")
    axes[-1].set_xlim(start_ms, max(start_ms + 0.001, (xmax - origin) / 1e6))
    label = f"{measurement.get('precision')} · {MODE_NAMES.get(measurement.get('mode'), measurement.get('mode'))} · offset={measurement.get('offset_ms')} ms"
    fig.suptitle(
        label
        + "\n并发整卡 counter 不能拆为两个模型的占用；活动比例不等于 FLOP/s 或 GB/s",
        fontsize=13,
    )
    files = save_figure(fig, out / "timeline")
    plt.close(fig)
    return files


def request_figure(nsys, measurement, out, font=None):
    plt = plotting(font)
    requests = nsys["requests"]
    if not requests:
        return []
    fields = [
        ("s1_release_to_first_kernel_ms", "DiT release→首 kernel"),
        ("s1_release_to_completion_ms", "DiT release→末 kernel"),
        ("s2_max_kernel_free_gap_ms", "VLM 最大自身 kernel 空隙"),
        ("pair_kernel_makespan_ms", "整轮 kernel makespan"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, (field, title) in zip(axes.flat, fields):
        points = [(r["iteration"], r[field]) for r in requests if r[field] is not None]
        if points:
            ax.plot(
                [p[0] for p in points], [p[1] for p in points], marker=".", linewidth=1
            )
        else:
            ax.text(
                0.5,
                0.5,
                "未记录必要的 release / kernel 证据",
                transform=ax.transAxes,
                ha="center",
            )
        ax.set_title(title)
        ax.set_xlabel("测量轮次")
        ax.set_ylabel("ms（Nsight 插桩运行）")
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"{measurement.get('precision')} · {measurement.get('mode')}：调度时序\nVLM 空隙含正常提交及依赖间隙，不能单独当作被抢占时间"
    )
    files = save_figure(fig, out / "requests")
    plt.close(fig)
    return files


def latency_comparison(rows, out, font=None):
    plt = plotting(font)
    fields = LATENCY_FIELDS
    fig, axes = plt.subplots(len(fields), 1, figsize=(15, 16), constrained_layout=True)
    labels = [
        f"{r['precision']}{'（硬件诊断）' if r['precision'] == 'w4a4' else ''}/{r['component']}\n{MODE_NAMES.get(r['mode'], r['mode'])}\nδ={r['offset_ms']}ms · {r['cohort_sha256'][:8]}"
        for r in rows
    ]
    for ax, (field, title) in zip(axes, fields):
        for i, row in enumerate(rows):
            data = row["latency"].get(field, {})
            if data.get("p50") is not None:
                ax.bar(
                    i,
                    data["p50"],
                    color=COLORS["S1"] if field.startswith("s1") else COLORS["S2"],
                    alpha=0.8,
                )
                ax.plot(i, data["p95"], marker="_", color="#222222", markersize=9)
            else:
                ax.text(
                    i,
                    0.02,
                    "未记录",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                    transform=ax.get_xaxis_transform(),
                )
        ax.set_xticks(range(len(rows)), labels, fontsize=7)
        ax.set_ylabel("ms")
        ax.set_title(title + "：柱 p50，黑横线 p95（无 profiler）", loc="left")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "πR² 单卡：先看请求响应，再看 CUDA 流跨度；缺失响应不以流跨度或 0 填充\nW4A4 仅为硬件诊断，既往闭环质量未通过"
    )
    save_figure(fig, out / "latency-comparison")
    plt.close(fig)


def scheduler_offsets(scheduler, out, font=None):
    plt = plotting(font)
    fields = [
        ("s1_release_to_first_kernel_ms", "DiT release→首 kernel"),
        ("s1_release_to_completion_ms", "DiT release→完成"),
        ("s2_max_kernel_free_gap_ms", "VLM 最大自身 kernel 空隙"),
        ("pair_kernel_makespan_ms", "整轮 kernel makespan"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    lines = defaultdict(list)
    for row in scheduler:
        lines[
            (
                row["cohort_sha256"][:8],
                row["precision"],
                row["component"],
                row["mode"],
            )
        ].append(row)
    for ax, (field, title) in zip(axes.flat, fields):
        for key, values in sorted(lines.items()):
            values = sorted(
                (v for v in values if v["metrics"][field]["p50"] is not None),
                key=lambda v: v["offset_ms"],
            )
            if values:
                ax.plot(
                    [v["offset_ms"] for v in values],
                    [v["metrics"][field]["p50"] for v in values],
                    marker="o",
                    label="/".join(key),
                )
        ax.set_title(title + "（p50）")
        ax.set_xlabel("DiT 释放相对 VLM 的 offset（ms）")
        ax.set_ylabel("ms（Nsight 插桩运行）")
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend(fontsize=7)
    fig.suptitle(
        "量化 × 调度优先级 × 释放时间：仅展示已实测点\nVLM 空隙不能独立证明被抢占；严格时延比较另见无 profiler 运行"
    )
    save_figure(fig, out / "scheduler-offset")
    plt.close(fig)
