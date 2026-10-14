"""Pure metadata and interval accounting for measured contention evidence."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merged(intervals):
    """Return the union, not sum, of possibly overlapping execution intervals."""
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(end - start for start, end in merged(intervals))


def intersection(left, right):
    left, right = merged(left), merged(right)
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            result.append([start, end])
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def stats(values):
    values = sorted(
        float(value) for value in values if value is not None and math.isfinite(value)
    )
    if not values:
        return {
            "n": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
        }

    def quantile(fraction):
        pos = fraction * (len(values) - 1)
        lower = int(pos)
        return values[lower] + (
            values[min(lower + 1, len(values) - 1)] - values[lower]
        ) * (pos - lower)

    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "min": values[0],
        "max": values[-1],
    }


def weighted_samples(samples, windows, max_gap_ns=None):
    """Time-weight sample values with a documented piecewise-constant convention.

    A value at t applies to [t, next_t), not beyond the final timestamp. Long
    gaps can be excluded explicitly. This is descriptive resampling, not an
    assertion about the hardware counter's internal integration boundary.
    """
    windows = merged(windows)
    samples = sorted(samples)
    covered, weighted = 0, 0.0
    for (start, value), (end, _) in itertools.pairwise(samples):
        if end <= start or not math.isfinite(value):
            continue
        if max_gap_ns is not None and end - start > max_gap_ns:
            continue
        width = duration(intersection([[start, end]], windows))
        covered += width
        weighted += width * value
    requested = duration(windows)
    return {
        "mean": weighted / covered if covered else None,
        "covered_ns": covered,
        "requested_ns": requested,
        "coverage_fraction": covered / requested if requested else None,
        "interpolation": "left sample held until next timestamp; no terminal extrapolation",
        "max_gap_ns": max_gap_ns,
    }


def attempt_acceptance(path, campaign_root):
    """Prevent completed worker files from bypassing a failed parent guard."""
    root = Path(campaign_root).resolve()
    current = Path(path).resolve().parent
    if current != root and root not in current.parents:
        return {"accepted": False, "reason": "measurement outside selected root"}
    # A caller may select a worker subdirectory as the report root. Its parent
    # attempt still owns validity, so extend the search to the campaign boundary.
    container = next(
        (
            parent
            for parent in (current, *current.parents)
            if parent.name in {"attempts", "rounds"}
        ),
        None,
    )
    inside_attempts = container is not None
    if container is not None and root.is_relative_to(container):
        root = container.parent
    while current == root or root in current.parents:
        inside_attempts |= current.name in {"attempts", "rounds"}
        result_path = current / "result.json"
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text())
            except (OSError, json.JSONDecodeError):
                return {
                    "accepted": False,
                    "reason": "unreadable attempt result",
                    "source": str(result_path),
                }
            if not isinstance(result, dict):
                return {
                    "accepted": False,
                    "reason": "attempt result is not an object",
                    "source": str(result_path),
                }
            valid = (
                result.get("state") == "complete"
                and result.get("measurements_valid") is True
            )
            return {
                "accepted": valid,
                "reason": "complete verified attempt"
                if valid
                else "attempt not complete and valid",
                "source": str(result_path),
                "state": result.get("state"),
            }
        if current == root:
            break
        current = current.parent
    return {
        "accepted": not inside_attempts,
        "reason": "attempt pending final validity result"
        if inside_attempts
        else "standalone evidence; no campaign attempt ancestor",
    }


def read_measurement(path):
    measurement = json.loads(Path(path).read_text())
    capture = measurement.get("measured_region", {}).get(
        "capture", measurement.get("protocol", {}).get("capture")
    )
    protocol = measurement.get("protocol", {})
    provenance = measurement.get("provenance", {})
    cohort = {
        "measurement_schema_version": measurement.get("schema_version"),
        "worker_revision": measurement.get("worker_revision"),
        "response_semantics": measurement.get("response_semantics"),
        "gpu": measurement.get("gpu"),
        "hardware": measurement.get("hardware"),
        "protocol": {
            k: v
            for k, v in protocol.items()
            if k not in {"capture", "iterations", "warmup", "dit_offset_ms"}
        },
        "provenance": {
            k: provenance.get(k)
            for k in (
                "checkpoint",
                "dataset",
                "worktree",
                "repository",
                "groot_root",
                "worker_sha256",
                "worker_modules_sha256",
                "adapter_sha256",
                "scheduling_sha256",
                "cpu_affinity",
                "fixtures",
                "bootstrap",
            )
        },
    }
    cohort_digest = hashlib.sha256(
        json.dumps(cohort, sort_keys=True).encode()
    ).hexdigest()
    result = {
        "measurement_schema_version": measurement.get("schema_version"),
        "worker_revision": measurement.get("worker_revision"),
        "response_semantics": measurement.get("response_semantics"),
        "latency_semantics": {
            "response_ms": "host-observed release to own CUDA event ready; includes cooperative permit wait, CPU submission, execution/queueing and host wake-up",
            "gpu_span_ms": "CUDA stream event span; excludes pre-start cooperative permit wait; not total request response or kernel busy time",
        },
        "cohort_sha256": cohort_digest,
        "cohort": cohort,
        "source": str(Path(path).resolve()),
        "sha256": digest(path),
        "precision": measurement.get("precision"),
        "mode": measurement.get("mode"),
        "component": measurement.get("component", "all"),
        "offset_ms": measurement.get(
            "offset_ms", measurement.get("protocol", {}).get("offset_ms", 0)
        ),
        "capture": capture,
        "measured_region": measurement.get("measured_region", {}),
        "unprofiled": capture == "none",
        "gpu": measurement.get("gpu"),
        "protocol": measurement.get("protocol", {}),
        "equivalence": measurement.get("equivalence"),
        "sample_count": len(measurement.get("samples", [])),
        "latency": {},
        "latency_samples": {},
    }
    samples = measurement.get("samples", [])
    fields = sorted(
        {key for sample in samples for key in sample if key.endswith("_ms")}
    )
    for field in fields:
        result["latency_samples"][field] = [sample.get(field) for sample in samples]
        result["latency"][field] = stats(result["latency_samples"][field])
    return result
