"""Attribute device kernels to innermost module and ATen launch ranges."""

import argparse
import bisect
import csv
import heapq
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from coexecution.analyze_trace import duration, merge


def assign_launches(ranges, launches):
    """Join on thread and CPU launch time, never on GPU/NVTX time overlap.

    Inputs: ranges=(start,end,thread,text), launches=(id,start,thread).
    Nested ranges are exclusive for attribution: only the innermost wins.
    """
    by_thread = defaultdict(list)
    for start, end, thread, text in ranges:
        category = None
        if text.startswith(("S1/", "S2/")):
            category = "worker"
        elif text.startswith("MODULE/"):
            category = "module"
        elif text.startswith("OP/"):
            category = "operator"
        if category and end is not None:
            by_thread[thread].append((start, end, text, category))
    thread_launches = defaultdict(list)
    for identifier, start, thread in launches:
        thread_launches[thread].append((start, identifier))
    result = {}
    for thread, entries in thread_launches.items():
        scopes = sorted(by_thread[thread])
        cursor = 0
        active = {key: [] for key in ["worker", "module", "operator"]}
        for timestamp, identifier in sorted(entries):
            while cursor < len(scopes) and scopes[cursor][0] <= timestamp:
                start, end, text, category = scopes[cursor]
                heapq.heappush(active[category], (-start, end, text))
                cursor += 1
            result[identifier] = {}
            for category, heap in active.items():
                while heap and heap[0][1] <= timestamp:
                    heapq.heappop(heap)
                result[identifier][category] = heap[0][2] if heap else None
    return result


def stage_name(role, module):
    name = (module or "").removeprefix(f"MODULE/{role}/")
    if role == "S1":
        for prefix, stage in [
            ("vl_self_attention", "VL condition self-attention"),
            ("vlln", "VL condition normalization"),
            ("state_encoder", "State encoder"),
            ("action_encoder", "Action encoder"),
            ("action_decoder", "Action decoder"),
            ("model", "DiT"),
        ]:
            if name == prefix or name.startswith(prefix + "."):
                return stage
        return "Rolling schedule / glue"
    for prefix, stage in [
        ("visual.patch_embed", "Vision patch projection"),
        ("visual.blocks", "Vision transformer"),
        ("visual.deepstack_merger_list", "Vision token mergers"),
        ("visual.merger", "Vision token mergers"),
        ("visual", "Vision positions / glue"),
        ("language_model.layers", "Language transformer"),
        ("language_model.embed_tokens", "Text embedding"),
        ("language_model", "Language positions / norm"),
    ]:
        if name == prefix or name.startswith(prefix + "."):
            return stage
    return "VL merge / masks / cache"


def operator_family(operator):
    name = operator.split(".")[1] if operator.startswith("aten.") else operator
    if any(key in name for key in ["attention", "flash"]):
        return "SDPA attention"
    if name in {"linear", "mm", "addmm", "bmm", "baddbmm", "matmul"}:
        return "Linear / matrix multiplication"
    if "convolution" in name or name in {"conv1d", "conv2d", "conv3d"}:
        return "Convolution"
    if name in {"to", "_to_copy", "type_as"}:
        return "Dtype / device conversion"
    if "norm" in name:
        return "Normalization"
    if any(
        key in name
        for key in [
            "index",
            "gather",
            "scatter",
            "cat",
            "copy",
            "clone",
            "contiguous",
            "_to_copy",
        ]
    ):
        return "Index / layout / copy"
    if name in {
        "sum",
        "mean",
        "max",
        "min",
        "any",
        "all",
        "amax",
        "amin",
        "nonzero",
        "cumsum",
    }:
        return "Reduction / scan"
    if name in {"normal_", "randn", "rand", "uniform_"}:
        return "Random generation"
    if operator == "Unattributed":
        return "Unattributed"
    return "Elementwise / other"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(database, output, warmup):
    output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    ranges = list(
        connection.execute(
            "SELECT start,end,globalTid,text FROM NVTX_EVENTS WHERE text IS NOT NULL AND end IS NOT NULL"
        )
    )
    runtime = list(
        connection.execute(
            "SELECT correlationId,start,globalTid,nameId,end FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        )
    )
    assert len({row[0] for row in runtime}) == len(runtime), (
        "Ambiguous runtime correlation IDs"
    )
    owners = assign_launches(ranges, [row[:3] for row in runtime])
    strings = dict(connection.execute("SELECT id,value FROM StringIds"))
    pairs = sorted(
        (start, end, text.split("/")[1], int(text.split("/")[2]))
        for start, end, _, text in ranges
        if text.startswith("PAIR/") and int(text.split("/")[2]) >= warmup
    )
    starts = [entry[0] for entry in pairs]

    def measured_pair(timestamp):
        index = bisect.bisect_right(starts, timestamp) - 1
        return pairs[index] if index >= 0 and timestamp < pairs[index][1] else None

    counts = defaultdict(int)
    for _, _, mode, _ in pairs:
        counts[mode] += 1
    operator_ranges = [row for row in ranges if row[3].startswith("OP/")]
    operator_owners = assign_launches(
        ranges, [(index, row[0], row[2]) for index, row in enumerate(operator_ranges)]
    )
    call_counts = defaultdict(int)
    for index, (start, _, _, text) in enumerate(operator_ranges):
        owner = operator_owners[index]
        pair = measured_pair(start)
        if pair is None or owner["worker"] is None:
            continue
        role, mode = owner["worker"].split("/")[:2]
        operator, signature = text.split("/")[1:]
        call_counts[(mode, role, "operator", operator)] += 1
        call_counts[(mode, role, "signature_id", int(signature))] += 1
    kernels = []
    contexts = set()
    for start, end, corr, name, device, context, stream in connection.execute(
        "SELECT start,end,correlationId,demangledName,deviceId,contextId,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        pair = measured_pair(start)
        if pair is None:
            continue
        owner = owners.get(corr)
        assert owner is not None and owner["worker"] is not None, (
            "Unassigned measured kernel"
        )
        assert start >= pair[0] and end <= pair[1]
        role, mode = owner["worker"].split("/")[:2]
        assert mode == pair[2]
        contexts.add((device, context))
        operator = (
            owner["operator"].split("/")[1] if owner["operator"] else "Unattributed"
        )
        signature = int(owner["operator"].split("/")[2]) if owner["operator"] else -1
        kernels.append(
            {
                "mode": mode,
                "iteration": pair[3],
                "role": role,
                "stream": stream,
                "module": owner["module"] or "Unattributed",
                "stage": stage_name(role, owner["module"]),
                "operator": operator,
                "signature_id": signature,
                "family": operator_family(operator),
                "kernel": strings[name],
                "start_ns": start,
                "end_ns": end,
                "duration_ms": (end - start) / 1e6,
            }
        )
    assert len(contexts) == 1, "Expected one CUDA device/context"
    aggregates = {}
    for dimension in [
        "stage",
        "module",
        "operator",
        "family",
        "signature_id",
        "kernel",
    ]:
        groups = defaultdict(list)
        for row in kernels:
            groups[(row["mode"], row["role"], row[dimension])].append(row)
        if dimension in {"operator", "signature_id"}:
            for mode, role, kind, value in call_counts:
                if kind == dimension:
                    groups.setdefault((mode, role, value), [])
        entries = []
        for (mode, role, value), rows in groups.items():
            entries.append(
                {
                    "mode": mode,
                    "role": role,
                    dimension: value,
                    "kernel_count": len(rows),
                    "kernels_per_call": len(rows) / counts[mode],
                    "kernel_ms_per_call": sum(row["duration_ms"] for row in rows)
                    / counts[mode],
                }
            )
            if dimension in {"operator", "signature_id"}:
                entries[-1]["operator_calls_per_iteration"] = (
                    call_counts[(mode, role, dimension, value)] / counts[mode]
                )
        entries.sort(
            key=lambda row: (row["mode"], row["role"], -row["kernel_ms_per_call"])
        )
        aggregates[dimension] = entries
        write_csv(output / f"{dimension}.csv", entries)
    api_groups = defaultdict(list)
    for corr, start, _, name_id, end in runtime:
        owner = owners[corr]
        pair = measured_pair(start)
        if owner["worker"] and pair:
            role, mode = owner["worker"].split("/")[:2]
            operator = (
                owner["operator"].split("/")[1] if owner["operator"] else "Unattributed"
            )
            api_groups[
                (
                    mode,
                    role,
                    stage_name(role, owner["module"]),
                    operator,
                    strings[name_id],
                )
            ].append(end - start)
    api_rows = []
    for (mode, role, stage, operator, api), values in api_groups.items():
        api_rows.append(
            {
                "mode": mode,
                "role": role,
                "stage": stage,
                "operator": operator,
                "api": api,
                "calls_per_iteration": len(values) / counts[mode],
                "host_api_ms_per_iteration": sum(values) / 1e6 / counts[mode],
            }
        )
    api_rows.sort(
        key=lambda row: (row["mode"], row["role"], -row["host_api_ms_per_iteration"])
    )
    write_csv(output / "api.csv", api_rows)
    closure = []
    for start, end, mode, iteration in pairs:
        rows = [
            row
            for row in kernels
            if row["mode"] == mode and row["iteration"] == iteration
        ]
        busy = duration(merge([(row["start_ns"], row["end_ns"]) for row in rows])) / 1e6
        closure.append(
            {
                "mode": mode,
                "iteration": iteration,
                "host_pair_ms": (end - start) / 1e6,
                "gpu_kernel_union_ms": busy,
                "outside_kernel_union_ms": (end - start) / 1e6 - busy,
            }
        )
    write_csv(output / "pair_accounting.csv", closure)
    report = {
        "source": str(database),
        "measured_pairs": dict(counts),
        "device_context": list(contexts),
        "total_kernels": len(kernels),
        "unattributed_kernels": sum(
            row["operator"] == "Unattributed" for row in kernels
        ),
        "aggregates": aggregates,
        "api": api_rows,
        "pair_accounting": closure,
        "notes": [
            "Kernel durations from instrumented diagnostic trace, not clean latency.",
            "Outside kernel union includes CPU submission, synchronization, copies and profiler overhead; it is not pure CPU time.",
        ],
    }
    (output / "breakdown.json").write_text(json.dumps(report, indent=2))
    write_csv(output / "all-kernels.csv", kernels)
    connection.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    report = analyze(args.database, args.output, args.warmup)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"aggregates", "api", "pair_accounting"}
            },
            indent=2,
        )
    )
