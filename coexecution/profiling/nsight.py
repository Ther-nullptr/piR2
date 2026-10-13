"""Read Nsight Systems SQLite exports without loading CUDA or model libraries."""

from __future__ import annotations

import bisect
import itertools
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from .analysis import digest, duration, intersection, merged, stats, weighted_samples

COMPONENT_RE = re.compile(r"^(S1/DIT|S2/VLM)(?:/|$)", re.IGNORECASE)
ITERATION_RE = re.compile(r"(?:^|/)iteration=(\d+)(?:/|$)")


def _rows(connection, table):
    return [dict(row) for row in connection.execute(f'SELECT * FROM "{table}"')]


def read_nsys(path):
    connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    strings = (
        {row["id"]: row["value"] for row in _rows(connection, "StringIds")}
        if "StringIds" in tables
        else {}
    )
    nvtx = _rows(connection, "NVTX_EVENTS") if "NVTX_EVENTS" in tables else []
    for row in nvtx:
        row["text"] = row.get("text") or strings.get(row.get("textId"), "")
    phases = [
        [row["start"], row["end"]]
        for row in nvtx
        if row["text"] == "pir2/phase=measurement" and row.get("end")
    ]
    component_ranges = defaultdict(list)
    pairs = []
    releases = {}
    for row in nvtx:
        text = row["text"]
        if phases and not any(start <= row["start"] <= end for start, end in phases):
            continue
        if text.startswith("PAIR/") and row.get("end"):
            parts = text.split("/")
            if parts[-1].isdigit():
                row["iteration"] = int(parts[-1])
                pairs.append(row)
        if text.startswith("S1_RELEASE/"):
            parts = text.split("/")
            iteration_match = ITERATION_RE.search(text)
            if iteration_match or parts[-1].isdigit():
                releases[
                    int(iteration_match.group(1) if iteration_match else parts[-1])
                ] = row["start"]
        match = COMPONENT_RE.match(text)
        iteration_match = ITERATION_RE.search(text)
        if match and iteration_match and row.get("end"):
            row["component"] = match.group(1).split("/")[0].upper()
            iteration_match = ITERATION_RE.search(text)
            row["iteration"] = (
                int(iteration_match.group(1)) if iteration_match else None
            )
            component_ranges[row["globalTid"]].append(row)
    for ranges in component_ranges.values():
        ranges.sort(key=lambda row: row["start"])
    starts_by_thread = {
        tid: [row["start"] for row in ranges]
        for tid, ranges in component_ranges.items()
    }
    launches = defaultdict(list)
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table not in tables:
            continue
        for row in _rows(connection, table):
            row["api_source_table"] = table
            tid, correlation = row.get("globalTid"), row.get("correlationId")
            if tid is None or correlation is None:
                continue
            ranges = component_ranges.get(tid, [])
            pos = bisect.bisect_right(starts_by_thread.get(tid, []), row["start"])
            owner = next(
                (
                    item
                    for item in reversed(ranges[:pos])
                    if item["start"] <= row["start"] <= item["end"]
                ),
                None,
            )
            launches[(tid & ~0xFFFFFF, correlation)].append((row, owner))
    process_ids = (
        {row["globalPid"]: row["pid"] for row in _rows(connection, "PROCESSES")}
        if "PROCESSES" in tables
        else {}
    )
    cuda_devices = (
        _rows(connection, "TARGET_INFO_CUDA_DEVICE")
        if "TARGET_INFO_CUDA_DEVICE" in tables
        else []
    )
    kernels = []
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    for row in _rows(connection, table) if table in tables else []:
        possibilities = launches.get(
            (row.get("globalPid"), row.get("correlationId")), []
        )
        matches = [
            (runtime, owner) for runtime, owner in possibilities if owner is not None
        ]
        # Runtime and driver can describe the same launch. Choose the closest
        # launch start before kernel execution; never join correlationId alone.
        before = [
            (runtime, owner)
            for runtime, owner in matches
            if runtime["start"] <= row["start"]
        ]
        runtime_candidates = [
            candidate
            for candidate in before
            if candidate[0]["api_source_table"] == "CUPTI_ACTIVITY_KIND_RUNTIME"
        ]
        match = max(
            runtime_candidates or before,
            key=lambda item: item[0]["start"],
            default=None,
        )
        runtime, owner = match if match else (None, None)
        item = {
            "start_ns": row["start"],
            "end_ns": row["end"],
            "device_id": row.get("deviceId"),
            "stream_id": row.get("streamId"),
            "process_id_encoded": row.get("globalPid"),
            "process_id": process_ids.get(row.get("globalPid")),
            "correlation_id": row.get("correlationId"),
            "name": strings.get(
                row.get("demangledName"), str(row.get("demangledName"))
            ),
            "component": owner["component"] if owner else "unassigned",
            "range": owner["text"] if owner else None,
            "iteration": owner.get("iteration") if owner else None,
            "launch_start_ns": runtime["start"] if runtime else None,
            "launch_api_source": runtime["api_source_table"] if runtime else None,
            "launch_end_ns": runtime["end"] if runtime else None,
        }
        if item["iteration"] is None and runtime:
            enclosing = [
                pair
                for pair in pairs
                if pair["start"] <= runtime["start"] <= pair["end"]
            ]
            if enclosing:
                item["iteration"] = min(
                    enclosing, key=lambda pair: pair["end"] - pair["start"]
                )["iteration"]
        physical = {
            device["gpuId"]
            for device in cuda_devices
            if device["pid"] == item["process_id"]
            and device["cudaId"] == item["device_id"]
        }
        item["physical_gpu_id"] = next(iter(physical)) if len(physical) == 1 else None
        kernels.append(item)
    metric_series = []
    if {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"} <= tables:
        metadata = {
            (row["typeId"], row["metricId"]): row
            for row in _rows(connection, "TARGET_INFO_GPU_METRICS")
        }
        grouped = defaultdict(list)
        for row in _rows(connection, "GPU_METRICS"):
            grouped[(row["typeId"], row["metricId"])].append(
                [row["timestamp"], row["value"]]
            )
        for key, samples in sorted(grouped.items()):
            meta = metadata.get(key, {})
            name = meta.get("metricName", f"unknown typeId={key[0]} metricId={key[1]}")
            # Units are metadata, never inferred from counter values.
            unit_match = re.search(r"\[([^\]]+)\]", name)
            metric_series.append(
                {
                    "type_id": key[0],
                    "metric_id": key[1],
                    "source_id": meta.get("sourceId"),
                    "gpu_id_from_type": key[0] & 0xFF,
                    "type_name": meta.get("typeName"),
                    "name": name,
                    "unit": unit_match.group(1)
                    if unit_match
                    else "not provided in metricName",
                    "samples": sorted(samples),
                }
            )
    gpu_metadata = (
        _rows(connection, "TARGET_INFO_GPU") if "TARGET_INFO_GPU" in tables else []
    )
    connection.close()
    associated = [kernel for kernel in kernels if kernel["component"] != "unassigned"]
    # Default window is all measured associated GPU activity; CPU NVTX may end
    # before async device work, so do not truncate device intervals at NVTX end.
    iteration_kernels = defaultdict(list)
    for kernel in associated:
        iteration_kernels[kernel["iteration"]].append(kernel)
    window = merged(
        [
            [min(k["start_ns"] for k in batch), max(k["end_ns"] for k in batch)]
            for batch in iteration_kernels.values()
        ]
    )
    s1 = [[k["start_ns"], k["end_ns"]] for k in associated if k["component"] == "S1"]
    s2 = [[k["start_ns"], k["end_ns"]] for k in associated if k["component"] == "S2"]
    request_metrics = []
    for iteration in sorted(
        {k["iteration"] for k in associated if k["iteration"] is not None}
    ):
        batch = [k for k in associated if k["iteration"] == iteration]
        s1_kernels = [k for k in batch if k["component"] == "S1"]
        s2_kernels = [k for k in batch if k["component"] == "S2"]
        one = merged([[k["start_ns"], k["end_ns"]] for k in s1_kernels])
        two = merged([[k["start_ns"], k["end_ns"]] for k in s2_kernels])
        release = releases.get(iteration)
        launch = min((k["launch_start_ns"] for k in s1_kernels), default=None)
        first = one[0][0] if one else None
        last = one[-1][1] if one else None
        gaps = [right[0] - left[1] for left, right in itertools.pairwise(two)]
        request_metrics.append(
            {
                "iteration": iteration,
                "s1_release_ns": release,
                "s1_first_launch_ns": launch,
                "s1_first_kernel_ns": first,
                "s1_last_kernel_ns": last,
                "s1_release_to_launch_ms": (launch - release) / 1e6
                if launch is not None and release is not None
                else None,
                "s1_launch_to_first_kernel_ms": (first - launch) / 1e6
                if first is not None and launch is not None
                else None,
                "s1_release_to_first_kernel_ms": (first - release) / 1e6
                if first is not None and release is not None
                else None,
                "s1_release_to_completion_ms": (last - release) / 1e6
                if last is not None and release is not None
                else None,
                "s1_kernel_span_ms": (one[-1][1] - one[0][0]) / 1e6 if one else None,
                "s2_kernel_span_ms": (two[-1][1] - two[0][0]) / 1e6 if two else None,
                "s2_max_kernel_free_gap_ms": max(gaps, default=0) / 1e6
                if two
                else None,
                "s2_internal_kernel_free_ms": sum(gaps) / 1e6 if two else None,
                "s1_kernel_union_ms": duration(one) / 1e6,
                "s2_kernel_union_ms": duration(two) / 1e6,
                "actual_kernel_overlap_ms": duration(intersection(one, two)) / 1e6,
                "pair_kernel_makespan_ms": (
                    max(k["end_ns"] for k in batch) - min(k["start_ns"] for k in batch)
                )
                / 1e6,
            }
        )
    for series in metric_series:
        values = [value for _, value in series["samples"]]
        quality_issues = []
        if "%" in series["unit"] and any(value < 0 or value > 100 for value in values):
            quality_issues.append(
                "percentage outside 0..100; do not interpret until raw export is verified"
            )
        if "Clock Frequency" in series["name"] and any(value < 0 for value in values):
            quality_issues.append(
                "negative frequency in raw export; invalid physical frequency"
            )
        if series["unit"] == "MHz" and any(abs(value) > 1000000 for value in values):
            quality_issues.append(
                "MHz label incompatible with raw magnitude; no automatic rescaling or overflow repair"
            )
        series["quality"] = {
            "interpretable": not quality_issues,
            "issues": quality_issues,
            "raw_min": min(values) if values else None,
            "raw_max": max(values) if values else None,
        }
        gaps = [
            b[0] - a[0]
            for a, b in zip(series["samples"], series["samples"][1:])
            if b[0] > a[0]
        ]
        max_gap = 3 * stats(gaps)["p50"] if gaps else None
        series["window_stats"] = weighted_samples(series["samples"], window, max_gap)
    return {
        "source": str(Path(path).resolve()),
        "sha256": digest(path),
        "timestamp_unit": "ns",
        "gpu_metadata": gpu_metadata,
        "cuda_device_mapping": cuda_devices,
        "window_ns": window,
        "phase_ranges_ns": phases,
        "kernels": kernels,
        "metrics": metric_series,
        "requests": request_metrics,
        "execution": {
            "s1_union_ms": duration(s1) / 1e6,
            "s2_union_ms": duration(s2) / 1e6,
            "overlap_ms": duration(intersection(s1, s2)) / 1e6,
            "both_union_ms": duration(s1 + s2) / 1e6,
            "associated_kernels": len(associated),
            "unassigned_kernels": len(kernels) - len(associated),
        },
        "limitations": [
            "NVTX marks host launches; CUDA correlation and PID determine kernel ownership.",
            "Device metrics describe the whole GPU; no division into per-model utilization.",
            "SM/Tensor activity percentages are not FLOP/s; DRAM percentage is not GB/s.",
            "Kernel overlap does not prove running-kernel preemption.",
            "S2 kernel-free gaps include normal host/launch/dependency gaps, not proven scheduler preemption.",
            "Launch-to-first-kernel includes launch overhead and dependency waits; not a pure hardware queue wait.",
        ],
    }
