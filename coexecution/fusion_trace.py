"""Compare fused chains using both CUDA runtime and driver launch correlations."""

import argparse
import bisect
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from coexecution.analyze_trace import duration, intersect, merge
from coexecution.operator_analysis import assign_launches


def analyze(database, output, warmup, return_entries=False):
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    ranges = list(
        connection.execute(
            "SELECT start,end,globalTid,text FROM NVTX_EVENTS WHERE text IS NOT NULL AND end IS NOT NULL"
        )
    )
    worker_ranges = [
        (
            start,
            end,
            thread,
            text.replace("CHAIN/", "MODULE/", 1) if text.startswith("CHAIN/") else text,
        )
        for start, end, thread, text in ranges
    ]
    strings = dict(connection.execute("SELECT id,value FROM StringIds"))
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    launches, correlation_ids = [], defaultdict(list)
    for table in ["CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"]:
        if table not in tables:
            continue
        for row_id, corr, start, thread, name_id in connection.execute(
            f"SELECT rowid,correlationId,start,globalTid,nameId FROM {table}"
        ):
            identifier = (table, row_id)
            launches.append((identifier, start, thread))
            correlation_ids[corr].append((identifier, strings[name_id]))
    owners = assign_launches(worker_ranges, launches)
    variants = [
        (start, end, text.split("/")[1])
        for start, end, _, text in ranges
        if text.startswith("VARIANT/")
    ]
    pairs = []
    for start, end, _, text in ranges:
        if not text.startswith("PAIR/"):
            continue
        _, mode, iteration = text.split("/")
        if int(iteration) < warmup:
            continue
        enclosing = [variant for lo, hi, variant in variants if lo <= start < hi]
        assert len(enclosing) == 1
        pairs.append((start, end, enclosing[0], mode, int(iteration)))
    pairs.sort()
    starts = [row[0] for row in pairs]
    entries = []
    contexts = set()
    for start, end, corr, name_id, device, context, stream in connection.execute(
        "SELECT start,end,correlationId,demangledName,deviceId,contextId,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        pair_index = bisect.bisect_right(starts, start) - 1
        if pair_index < 0 or start >= pairs[pair_index][1]:
            continue
        pair = pairs[pair_index]
        assert end <= pair[1]
        assignments = [
            (owners[identifier], api, identifier[0])
            for identifier, api in correlation_ids[corr]
            if owners[identifier]["worker"] is not None
        ]
        assert assignments, f"No launch correlation for {strings[name_id]}"
        roles = {owner["worker"].split("/")[0] for owner, _, _ in assignments}
        chains = {owner["module"] or "MODULE/other" for owner, _, _ in assignments}
        assert len(roles) == len(chains) == 1, "Ambiguous launch ownership"
        role, chain = roles.pop(), chains.pop().removeprefix("MODULE/")
        contexts.add((device, context))
        entries.append(
            {
                "variant": pair[2],
                "mode": pair[3],
                "iteration": pair[4],
                "role": role,
                "chain": chain,
                "kernel": strings[name_id],
                "start": start,
                "end": end,
                "stream": stream,
                "api_tables": sorted({table for _, _, table in assignments}),
            }
        )
    assert len(contexts) == 1
    counts = Counter((variant, mode) for _, _, variant, mode, _ in pairs)
    groups = defaultdict(list)
    for entry in entries:
        groups[(entry["variant"], entry["mode"], entry["role"], entry["chain"])].append(
            entry
        )
    breakdown = []
    for (variant, mode, role, chain), rows in sorted(groups.items()):
        n = counts[(variant, mode)]
        breakdown.append(
            {
                "variant": variant,
                "mode": mode,
                "role": role,
                "chain": chain,
                "kernel_count_per_call": len(rows) / n,
                "kernel_ms_per_call": sum(row["end"] - row["start"] for row in rows)
                / 1e6
                / n,
            }
        )
    overlap = []
    for _, _, variant, mode, iteration in pairs:
        selected = [
            row
            for row in entries
            if (row["variant"], row["mode"], row["iteration"])
            == (variant, mode, iteration)
        ]
        s1 = merge(
            [(row["start"], row["end"]) for row in selected if row["role"] == "S1"]
        )
        s2 = merge(
            [(row["start"], row["end"]) for row in selected if row["role"] == "S2"]
        )
        value = duration(intersect(s1, s2)) / 1e6
        if mode == "serial":
            assert value == 0
        overlap.append(
            {
                "variant": variant,
                "mode": mode,
                "iteration": iteration,
                "overlap_ms": value,
                "s1_active_ms": duration(s1) / 1e6,
                "s2_active_ms": duration(s2) / 1e6,
            }
        )
    result = {
        "source": str(database),
        "device_context": list(contexts),
        "total_measured_kernels": len(entries),
        "breakdown": breakdown,
        "overlap": overlap,
        "launch_api_table_counts": dict(
            Counter(table for row in entries for table in row["api_tables"])
        ),
        "note": "Kernel work from independent profile, not clean latency; both runtime and driver launch tables are supported.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    connection.close()
    return (result, entries) if return_entries else result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(analyze(args.database, args.output, args.warmup), indent=2))
