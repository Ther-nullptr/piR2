"""Group only accepted evidence with compatible source and timing semantics."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .analysis import attempt_acceptance, read_measurement, stats


def aggregate_measurements(root):
    paths = sorted(Path(root).rglob("measurement.json"))
    acceptance = {str(path): attempt_acceptance(path, root) for path in paths}
    measurements = [
        read_measurement(path) for path in paths if acceptance[str(path)]["accepted"]
    ]
    groups = defaultdict(list)
    for measurement in measurements:
        if measurement["unprofiled"]:
            groups[
                (
                    measurement["cohort_sha256"],
                    measurement["precision"],
                    measurement["component"],
                    measurement["mode"],
                    measurement["offset_ms"],
                )
            ].append(measurement)
    rows = []
    for (cohort, precision, component, mode, offset), items in sorted(
        groups.items(), key=lambda item: str(item[0])
    ):
        fields = {key for item in items for key in item["latency_samples"]}
        row = {
            "cohort_sha256": cohort,
            "precision": precision,
            "component": component,
            "mode": mode,
            "offset_ms": offset,
            "repeat_files": len(items),
            "source_files": [item["source"] for item in items],
            "source_sha256": [item["sha256"] for item in items],
            "latency": {
                field: stats(
                    [
                        value
                        for item in items
                        for value in item["latency_samples"].get(field, [])
                    ]
                )
                for field in fields
            },
        }
        rows.append(row)
    # Response ratios never fall back to CUDA stream spans for older schemas.
    for row in rows:
        row["slowdown_vs_isolated"] = {}
        row["stream_span_ratio_vs_isolated"] = {}
        for component, mode, prefix in (
            ("S1", "dit_only", "s1"),
            ("S2", "vlm_only", "s2"),
        ):
            candidates = [
                baseline
                for baseline in rows
                if baseline["cohort_sha256"] == row["cohort_sha256"]
                and baseline["mode"] == mode
                and baseline["precision"] == row["precision"]
                and baseline["component"] == row["component"]
                and baseline["offset_ms"] == 0
            ]
            for field, target in (
                (f"{prefix}_response_ms", "slowdown_vs_isolated"),
                (f"{prefix}_gpu_span_ms", "stream_span_ratio_vs_isolated"),
            ):
                baseline = (
                    candidates[0]["latency"].get(field, {}).get("p50")
                    if len(candidates) == 1
                    else None
                )
                observed = row["latency"].get(field, {}).get("p50")
                row[target][component] = (
                    observed / baseline if baseline and observed is not None else None
                )
    return measurements, rows, acceptance


def scheduler_records(root):
    evidence_paths = sorted(Path(root).rglob("evidence.json"))
    scheduler = []
    seen_nsys = set()
    for path in reversed(evidence_paths):
        if not attempt_acceptance(path, root)["accepted"]:
            continue
        evidence = json.loads(path.read_text())
        ns = evidence.get("nsys")
        if not ns or not ns.get("requests") or ns.get("sha256") in seen_nsys:
            continue
        seen_nsys.add(ns.get("sha256"))
        m = evidence["measurement"]
        fields = [
            "s1_release_to_launch_ms",
            "s1_launch_to_first_kernel_ms",
            "s1_release_to_first_kernel_ms",
            "s1_release_to_completion_ms",
            "s2_max_kernel_free_gap_ms",
            "pair_kernel_makespan_ms",
        ]
        scheduler.append(
            {
                "cohort_sha256": m["cohort_sha256"],
                "precision": m["precision"],
                "component": m["component"],
                "mode": m["mode"],
                "offset_ms": m["offset_ms"],
                "source": str(path.resolve()),
                "metrics": {
                    field: stats([request[field] for request in ns["requests"]])
                    for field in fields
                },
            }
        )
    return scheduler
