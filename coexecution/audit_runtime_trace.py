"""Audit graph launches, remaining synchronization and real GPU overlap."""

import argparse
import bisect
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from coexecution.analyze_trace import duration, intersect, merge
from coexecution.fusion_trace import analyze
from coexecution.operator_analysis import assign_launches


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    report, kernels = analyze(
        args.database, args.output / "kernel-accounting.json", 2, return_entries=True
    )
    connection = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    ranges = list(
        connection.execute(
            "SELECT start,end,globalTid,text FROM NVTX_EVENTS WHERE text IS NOT NULL AND end IS NOT NULL"
        )
    )
    variants = [
        (a, b, label.split("/")[1])
        for a, b, _, label in ranges
        if label.startswith("VARIANT/")
    ]
    pairs = []
    for start, end, _, label in ranges:
        if label.startswith("PAIR/") and int(label.split("/")[2]) >= 2:
            variant = next(name for low, high, name in variants if low <= start < high)
            pairs.append((start, end, variant, label.split("/")[1]))
    pairs.sort()
    starts = [row[0] for row in pairs]
    counts = Counter((v, m) for _, _, v, m in pairs)
    runtimes = list(
        connection.execute(
            "SELECT rowid,start,end,globalTid,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    )
    owners = assign_launches(
        ranges,
        [(identifier, start, thread) for identifier, start, _, thread, _ in runtimes],
    )
    names = dict(connection.execute("SELECT id,value FROM StringIds"))
    grouped = defaultdict(list)
    for identifier, start, end, _, name in runtimes:
        index = bisect.bisect_right(starts, start) - 1
        if index < 0 or start >= pairs[index][1] or not owners[identifier]["worker"]:
            continue
        _, _, variant, mode = pairs[index]
        role = owners[identifier]["worker"].split("/")[0]
        api = names[name]
        if any(
            key in api
            for key in [
                "GraphLaunch",
                "LaunchKernel",
                "Synchronize",
                "Memcpy",
                "WaitEvent",
            ]
        ):
            grouped[(variant, mode, role, api)].append(end - start)
    api_rows = [
        {
            "variant": v,
            "mode": m,
            "role": r,
            "api": a,
            "calls_per_iteration": len(values) / counts[(v, m)],
            "host_api_ms_per_iteration": sum(values) / 1e6 / counts[(v, m)],
        }
        for (v, m, r, a), values in sorted(grouped.items())
    ]
    node_rows = list(
        connection.execute(
            "SELECT start,graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId IS NOT NULL AND graphNodeId!=0"
        )
    )
    nodes = Counter()
    for start, node in node_rows:
        index = bisect.bisect_right(starts, start) - 1
        if index >= 0 and start < pairs[index][1]:
            nodes[(pairs[index][2], pairs[index][3])] += 1
    graph_rows = [
        {"variant": v, "mode": m, "graph_node_kernels_per_pair": n / counts[(v, m)]}
        for (v, m), n in sorted(nodes.items())
    ]
    summaries = []
    for variant, mode in counts:
        rows = [
            row
            for row in report["overlap"]
            if row["variant"] == variant and row["mode"] == mode
        ]
        summaries.append(
            {
                "variant": variant,
                "mode": mode,
                "pairs": len(rows),
                "s1_active_ms": sum(row["s1_active_ms"] for row in rows) / len(rows),
                "s2_active_ms": sum(row["s2_active_ms"] for row in rows) / len(rows),
                "overlap_ms": sum(row["overlap_ms"] for row in rows) / len(rows),
            }
        )
    result = {
        "source": str(args.database),
        "device_context": report["device_context"],
        "api": api_rows,
        "graph_kernels": graph_rows,
        "kernel_overlap": summaries,
        "scope": "Six measured pairs per mode/recipe, no warmups. Host API durations are not additive GPU/wall breakdowns.",
    }
    (args.output / "audit.json").write_text(json.dumps(result, indent=2))
    plot(kernels, args.output / "timeline")
    print(json.dumps(result, indent=2))


def plot(kernels, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True)
    maximum = 0
    for ax, variant in zip(axes, ["rope", "combined"]):
        rows = [
            row
            for row in kernels
            if row["variant"] == variant
            and row["mode"] == "concurrent"
            and row["iteration"] == 2
        ]
        roles = {
            role: merge(
                [(row["start"], row["end"]) for row in rows if row["role"] == role]
            )
            for role in ["S1", "S2"]
        }
        roles["Both"] = intersect(roles["S1"], roles["S2"])
        origin = min(row["start"] for row in rows)
        for role, y, color in [
            ("S1", 2, "#386DA8"),
            ("S2", 1, "#218C78"),
            ("Both", 0, "#9B638D"),
        ]:
            bars = [((a - origin) / 1e6, (b - a) / 1e6) for a, b in roles[role]]
            ax.broken_barh(bars, (y - 0.28, 0.56), facecolors=color, edgecolors="none")
            if bars:
                maximum = max(maximum, max(a + b for a, b in bars))
        ax.set_yticks([0, 1, 2], ["Both active", "S2 kernels", "S1 kernels"])
        ax.set_ylim(-0.6, 2.6)
        ax.set_title(
            f"{variant}: real kernel overlap {duration(roles['Both']) / 1e6:.3f} ms",
            loc="left",
            weight="bold",
        )
        ax.grid(axis="x", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlim(0, maximum * 1.03)
    axes[-1].set_xlabel("Time from first GPU kernel in each selected pair (ms)")
    fig.suptitle(
        "S1/S2 device execution / one CUDA context: RoPE reference vs static + graph runtime\nProfiler timelines prove execution; separate unprofiled trials measure speed.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=1.8)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
