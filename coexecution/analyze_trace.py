"""Count real S1/S2 kernel overlap from Nsight's CUDA/NVTX correlations."""

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def intersect(first, second):
    result = []
    i = j = 0
    while i < len(first) and j < len(second):
        lo = max(first[i][0], second[j][0])
        hi = min(first[i][1], second[j][1])
        if hi > lo:
            result.append((lo, hi))
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1
    return result


def duration(intervals):
    return sum(end - start for start, end in intervals)


def analyze(database, output, warmup=2):
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    assignments = list(
        connection.execute("""
        SELECT DISTINCT k.rowid, k.deviceId, k.contextId, k.streamId, substr(n.text,1,2)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId
        JOIN NVTX_EVENTS n ON n.globalTid=r.globalTid AND n.start<=r.start AND r.start<n.end
        WHERE n.text LIKE 'S1/%' OR n.text LIKE 'S2/%'
    """)
    )
    assert assignments, "No CUDA launch/NVTX correlations found"
    roles = defaultdict(set)
    counts = Counter()
    contexts = set()
    for _, device, context, stream, role in assignments:
        contexts.add((device, context))
        roles[(device, context, stream)].add(role)
        counts[(stream, role)] += 1
    assert len(contexts) == 1, "S1 and S2 must share one CUDA device and context"
    assert all(len(value) == 1 for value in roles.values()), "Mixed roles on a stream"
    assert len(roles) == 2, "This analyzer expects one stream per system"
    role_streams = {next(iter(value)): key[2] for key, value in roles.items()}
    assert set(role_streams) == {"S1", "S2"} and len(set(role_streams.values())) == 2
    all_kernels = list(
        connection.execute(
            "SELECT start,end,deviceId,contextId,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
        )
    )
    all_pairs = list(
        connection.execute(
            "SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'PAIR/%' AND end IS NOT NULL ORDER BY start"
        )
    )
    pairs, plot_data = [], {}
    device, context = next(iter(contexts))
    for begin, end, text in all_pairs:
        _, mode, iteration = text.split("/")
        if int(iteration) < warmup:
            continue
        intervals = {role: [] for role in role_streams}
        unknown = 0
        for start, stop, kd, kc, stream in all_kernels:
            if stop <= begin or start >= end or (kd, kc) != (device, context):
                continue
            role = next((r for r, s in role_streams.items() if s == stream), None)
            if role is None:
                unknown += 1
                continue
            assert begin <= start <= stop <= end, (
                "Kernel crosses synchronized pair boundary"
            )
            intervals[role].append((start, stop))
        assert unknown == 0, "Unexpected compute stream within a measured pair"
        first, second = merge(intervals["S1"]), merge(intervals["S2"])
        common = intersect(first, second)
        pair = {
            "mode": mode,
            "iteration": int(iteration),
            "s1_kernel_count": len(intervals["S1"]),
            "s2_kernel_count": len(intervals["S2"]),
            "s1_active_ms": duration(first) / 1e6,
            "s2_active_ms": duration(second) / 1e6,
            "overlap_ms": duration(common) / 1e6,
            "gpu_union_ms": duration(merge(first + second)) / 1e6,
            "host_pair_range_ms": (end - begin) / 1e6,
        }
        pairs.append(pair)
        if mode not in plot_data:
            plot_data[mode] = {
                "origin_ns": min(first[0][0], second[0][0]),
                "S1": first,
                "S2": second,
                "overlap": common,
                "pair": pair,
            }
    summaries = {}
    for mode in sorted({p["mode"] for p in pairs}):
        rows = [p for p in pairs if p["mode"] == mode]
        first = sum(p["s1_active_ms"] for p in rows)
        second = sum(p["s2_active_ms"] for p in rows)
        common = sum(p["overlap_ms"] for p in rows)
        summaries[mode] = {
            "pairs": len(rows),
            "s1_active_ms": first,
            "s2_active_ms": second,
            "overlap_ms": common,
            "mean_overlap_ms_per_pair": common / len(rows),
            "overlap_fraction_of_s1_active": common / first,
            "overlap_fraction_of_s2_active": common / second,
        }
    assert summaries["serial"]["overlap_ms"] == 0
    result = {
        "source": str(database),
        "cuda_device_id": device,
        "cuda_context_id": context,
        "single_device_and_context": True,
        "stream_roles": role_streams,
        "assignment_kernel_counts": {f"{s}:{r}": n for (s, r), n in counts.items()},
        "warmup_pairs_excluded_per_mode": warmup,
        "summary": summaries,
        "pairs": pairs,
        "note": "Overlap is intersection of merged GPU kernel intervals, not CPU/NVTX or CUDA-event enclosing spans. Profiler timings are not the latency benchmark.",
    }
    run_metadata = json.loads((database.parent / "results.json").read_text())
    preflight = json.loads((database.parent / "gpu-preflight.json").read_text())
    result["physical_gpu"] = run_metadata["physical_visible_device"]
    result["physical_gpu_uuid"] = preflight["uuid"]
    result["worker_api_audit"] = [
        {"role": role, "api": api, "count": count, "host_api_duration_ms": elapsed}
        for role, api, count, elapsed in connection.execute("""
            SELECT substr(n.text,1,2), s.value, count(*), sum(r.end-r.start)/1e6
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
            JOIN NVTX_EVENTS n ON n.globalTid=r.globalTid AND n.start<=r.start AND r.start<n.end
            WHERE (n.text LIKE 'S1/%' OR n.text LIKE 'S2/%') AND
                (s.value LIKE '%Synchronize%' OR s.value LIKE '%Memcpy%' OR s.value LIKE '%WaitEvent%')
            GROUP BY 1,2 ORDER BY 1,2
        """)
    ]
    result["worker_copy_audit"] = [
        {
            "role": role,
            "copy_kind": kind,
            "count": count,
            "bytes": size,
            "largest_copy_bytes": largest,
        }
        for role, kind, count, size, largest in connection.execute("""
            SELECT substr(n.text,1,2), kind.label, count(*), sum(m.bytes), max(m.bytes)
            FROM CUPTI_ACTIVITY_KIND_MEMCPY m
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=m.correlationId
            JOIN NVTX_EVENTS n ON n.globalTid=r.globalTid AND n.start<=r.start AND r.start<n.end
            JOIN ENUM_CUDA_MEMCPY_OPER kind ON kind.id=m.copyKind
            WHERE n.text LIKE 'S1/%' OR n.text LIKE 'S2/%'
            GROUP BY 1,2 ORDER BY 1,2
        """)
    ]
    result["audit_scope"] = (
        "Worker API and copy totals include warmup pairs; host API durations must not be added to GPU kernel times."
    )
    assert run_metadata["cuda_device_count"] == 1
    for mode in summaries:
        assert summaries[mode]["pairs"] == sum(
            row["mode"] == mode for row in run_metadata["rows"]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    output.with_name("timeline-intervals.json").write_text(json.dumps(plot_data))
    connection.close()
    return result, plot_data


def plot_timeline(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 5.5), sharex=True, constrained_layout=True
    )
    max_time = 0.0
    for ax, mode in zip(axes, ["serial", "concurrent"]):
        entry = data[mode]
        origin = entry["origin_ns"]
        for role, y, color in [
            ("S1", 2, "#167D9A"),
            ("S2", 1, "#DB873A"),
            ("overlap", 0, "#8C4A85"),
        ]:
            bars = [((a - origin) / 1e6, (b - a) / 1e6) for a, b in entry[role]]
            ax.broken_barh(bars, (y - 0.28, 0.56), facecolors=color, edgecolors="none")
            if bars:
                max_time = max(max_time, max(a + b for a, b in bars))
        ax.set_yticks([0, 1, 2], ["Both active", "S2 kernels", "S1 kernels"])
        ax.set_ylim(-0.65, 2.75)
        ax.set_title(
            f"{mode.capitalize()} | actual kernel overlap: {entry['pair']['overlap_ms']:.3f} ms",
            loc="left",
        )
        ax.grid(axis="x", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlim(0, max_time * 1.02)
    axes[-1].set_xlabel("Time since first GPU kernel in each selected pair (ms)")
    fig.suptitle(
        "One physical GPU (selected device): S1 and S2 kernel execution\nNsight trace proves overlap; unprofiled runs measure latency",
        fontsize=12,
    )
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    report, examples = analyze(args.database, args.output, args.warmup)
    if args.plot:
        plot_timeline(examples, args.output.with_name("timeline"))
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, indent=2))
