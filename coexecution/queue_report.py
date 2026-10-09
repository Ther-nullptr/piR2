"""Offline reports from real LIBERO queue traces and NVML telemetry.

Run ``python -m coexecution.queue_report --input RUN_ROOT --output REPORT_DIR``.
No model is loaded. The optional embedded queue model is explicitly illustrative.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import re
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def read_jsonl(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    result = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            # A running experiment may be writing its last line.
    return result


def summary(values):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(v))
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}

    def quantile(p):
        k = (len(values) - 1) * p
        return values[math.floor(k)] * (1 - k % 1) + values[math.ceil(k)] * (k % 1)

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "max": values[-1],
    }


def fraction(n, d):
    return n / d if d else None


def integrate_telemetry(samples, windows, gpu_ids):
    """Clip linear sample segments to each control window, without extrapolation.

    Energy is GPU board energy (includes renderer/background work on these GPUs).
    Counter differences and power integration are independent estimates. Coverage
    is GPU-seconds observed divided by requested GPU-seconds, so a missing device
    cannot silently appear to consume zero energy.
    """
    exposure = sum(b - a for a, b in windows)
    devices = []
    for gpu in gpu_ids:
        rows = sorted(
            (r for r in samples if r.get("gpu") == gpu), key=lambda r: r["t_s"]
        )
        energy = coverage = counter_energy = counter_coverage = 0.0
        weighted = defaultdict(float)
        weighted_coverage = defaultdict(float)
        throttle_time = 0.0
        max_gap = 0.0
        for left, right in pairwise(rows):
            dt = right["t_s"] - left["t_s"]
            if dt <= 0:
                continue
            for begin, end in windows:
                a, b = max(begin, left["t_s"]), min(end, right["t_s"])
                if b <= a:
                    continue
                max_gap = max(max_gap, dt)
                lo, hi = (a - left["t_s"]) / dt, (b - left["t_s"]) / dt
                width = b - a
                for key in [
                    "power_w",
                    "sm_mhz",
                    "memory_mhz",
                    "utilization",
                    "temperature_c",
                ]:
                    x, y = left.get(key), right.get(key)
                    if x is None or y is None:
                        continue
                    integral = ((x + (y - x) * lo) + (x + (y - x) * hi)) / 2 * width
                    weighted[key] += integral
                    weighted_coverage[key] += width
                    if key == "power_w":
                        energy += integral
                        coverage += width
                x, y = left.get("energy_mj"), right.get("energy_mj")
                if x is not None and y is not None and y >= x:
                    counter_energy += (y - x) * (hi - lo) / 1000
                    counter_coverage += width
                if left.get("throttle_reasons", 0):
                    throttle_time += width
        devices.append(
            {
                "gpu": gpu,
                "sampled_energy_j": energy if coverage else None,
                "counter_energy_j": counter_energy if counter_coverage else None,
                "coverage_fraction": fraction(coverage, exposure),
                "counter_coverage_fraction": fraction(counter_coverage, exposure),
                "covered_seconds": coverage,
                "average_power_w": fraction(energy, coverage),
                "sm_mhz_mean": fraction(
                    weighted["sm_mhz"], weighted_coverage["sm_mhz"]
                ),
                "memory_mhz_mean": fraction(
                    weighted["memory_mhz"], weighted_coverage["memory_mhz"]
                ),
                "utilization_mean": fraction(
                    weighted["utilization"], weighted_coverage["utilization"]
                ),
                "temperature_c_mean": fraction(
                    weighted["temperature_c"], weighted_coverage["temperature_c"]
                ),
                "throttle_time_s": throttle_time,
                "max_sample_gap_s": max_gap,
            }
        )
    total = sum(d["sampled_energy_j"] or 0 for d in devices)
    covered = sum(d["covered_seconds"] for d in devices)
    full_counters = all(d["counter_energy_j"] is not None for d in devices)
    return {
        "devices": devices,
        "sampled_energy_j": total if covered else None,
        "counter_energy_j": sum(d["counter_energy_j"] for d in devices)
        if devices and full_counters
        else None,
        "coverage_fraction": fraction(covered, exposure * len(gpu_ids)),
        # Partial telemetry cannot identify whole-window total board power.
        "average_power_w": fraction(total, exposure)
        if gpu_ids
        and exposure > 0
        and math.isclose(covered, exposure * len(gpu_ids), abs_tol=1e-9)
        else None,
        "controlled_seconds": exposure,
        "scope": "NVML board energy for inference GPUs, including any renderer/background GPU work; no baseline subtraction",
    }


def summarize_episode(events, metadata):
    """Separate controlled exposure from warmup/drain and future-output censoring."""
    events = sorted(events, key=lambda r: (r["t_s"], r.get("seq", 0)))
    begin = next((r for r in events if r["kind"] == "control_start"), None)
    end = next((r for r in events if r["kind"] == "control_end"), None)
    if begin is None or end is None:
        raise ValueError("A completed control_start/control_end window is required")
    start, stop = begin["t_s"], end["t_s"]
    inside = [r for r in events if start <= r["t_s"] <= stop]
    ticks = [r for r in inside if r["kind"] == "control_tick"]
    by_kind = defaultdict(list)
    for row in events:
        by_kind[row["kind"]].append(row)
    completion = {r["request_tick"]: r for r in by_kind["action_completed"]}
    consumed = {
        r.get("audit", {}).get("request_tick"): r
        for r in by_kind["request_completed"] + by_kind["unused_result_at_episode_end"]
    }

    def relative(value):
        return value - start if value is not None else None

    feature_updates = [
        {
            "t": relative(r["t_s"]),
            "feature_sequence": r.get("feature_sequence"),
            "source_tick": r.get("source_tick"),
            "capture_s": relative(r.get("capture_s")),
            "rpc_completed_s": relative(r.get("rpc_completed_s")),
        }
        for r in by_kind["feature_published"]
        if r["t_s"] <= stop
    ]
    # Join an RPC to its publication by the recorded capture identity and exact
    # completion timestamp, never by proximity or unrelated counters.
    publications = {
        (r.get("source_tick"), r.get("capture_s"), r.get("rpc_completed_s")): r
        for r in by_kind["feature_published"]
    }
    adoption = {
        r["request_tick"]: r for r in by_kind["action_adopted"] if r["t_s"] <= stop
    }
    started = {r["request_tick"]: r for r in by_kind["action_started"]}
    requests = []
    action_durations, vision_durations = [], []
    due = censored_future = 0
    steps = end.get("control_steps", len(ticks))
    for row in by_kind["action_submitted"]:
        if not start <= row["t_s"] < stop:
            continue
        k = row["request_tick"]
        done = completion.get(k, {})
        timing = consumed.get(k, {}).get("client_timing", {})
        actual = {**done, **timing}
        finish = done.get("t_s")
        censored = finish is None or finish > stop
        if not censored:
            action_durations.append(
                1000 * done.get("request_elapsed_s", finish - row["t_s"])
            )
        lower, upper = done.get("valid_start_tick"), done.get("valid_end_tick")
        if lower is not None and upper is not None:
            due += max(0, min(upper, steps) - lower)
            censored_future += max(0, upper - max(lower, steps))
        requests.append(
            {
                "role": "S1",
                "request_tick": k,
                "start_s": row["t_s"] - start,
                "worker_start_s": started.get(k, row)["t_s"] - start,
                "end_s": min(finish if finish is not None else stop, stop) - start,
                "completed_s": finish - start if finish is not None else None,
                "censored": censored,
                "adopted_s": adoption[k]["t_s"] - start if k in adoption else None,
                "deadline_s": row["deadline_s"] - start
                if "deadline_s" in row
                else None,
                "feature_source_tick": actual.get("feature_source_tick"),
                "feature_sequence": actual.get("feature_sequence"),
                "feature_capture_s": relative(actual.get("feature_capture_s")),
                "state_capture_s": relative(actual.get("state_capture_s")),
                "state_tick": actual.get("state_tick"),
                "use_s": relative(timing.get("plan_rpc_started_s")),
                "delay_ticks": row.get("delay_ticks"),
                "valid_start_tick": lower,
                "valid_end_tick": upper,
            }
        )
    for row in by_kind["vision_request"]:
        a, b = row["started_s"], row["completed_s"]
        if not start <= a < stop:
            continue
        censored = b > stop
        if not censored:
            vision_durations.append(1000 * row["rpc_seconds"])
        publication = publications.get(
            (row.get("source_tick"), row.get("capture_s"), b), {}
        )
        requests.append(
            {
                "role": "S2",
                "source_tick": row.get("source_tick"),
                "capture_s": relative(row.get("capture_s")),
                "feature_sequence": publication.get("feature_sequence"),
                "published_s": relative(publication.get("t_s")),
                "start_s": a - start,
                "end_s": min(b, stop) - start,
                "completed_s": b - start,
                "censored": censored,
            }
        )
    cache_samples = []
    for row in by_kind["request_completed"] + by_kind["unused_result_at_episode_end"]:
        timing = row.get("client_timing", {})
        when = timing.get("plan_rpc_started_s", timing.get("started_s"))
        age = timing.get("cache_age_at_request_ms")
        if when is not None and age is not None and start <= when < stop:
            cache_samples.append(
                {
                    "t": when - start,
                    "age_ms": age,
                    "request_tick": row.get("audit", {}).get("request_tick"),
                }
            )
    adopted = [r for r in inside if r["kind"] == "action_adopted"]
    expired = sum(r.get("expired_slots", 0) for r in adopted)
    admitted = {r["request_tick"] for r in requests if r["role"] == "S1"}
    expired += sum(
        r.get("expired_before_episode_end", 0)
        for r in by_kind["unused_result_at_episode_end"]
        if r.get("audit", {}).get("request_tick") in admitted
    )
    offered = sum(r["kind"] == "camera_offered" for r in inside)
    replaced = sum(r["kind"] == "camera_replaced" for r in inside)
    fallback = sum(bool(r.get("fallback")) for r in ticks)
    tick_rows = []
    by_request = {r["request_tick"]: r for r in requests if r["role"] == "S1"}
    for row in ticks:
        slots = row.get("action_buffer", {}).get("slots", [])
        t = row["t_s"] - start
        feature = next(
            (r for r in reversed(feature_updates) if r["t"] <= t),
            {"capture_s": relative(begin.get("feature_capture_s"))},
        )
        slot = row.get("executed_slot") or {}
        producer = by_request.get(slot.get("producer_request_tick"), {})
        image_capture = state_capture = None
        if not row.get("fallback") and not slot.get("fallback"):
            if slot.get("producer_kind") == "bootstrap":
                image_capture = relative(begin.get("feature_capture_s"))
                state_capture = relative(begin.get("initial_capture_s"))
            elif slot.get("producer_kind") == "request":
                image_capture = producer.get("feature_capture_s")
                state_capture = producer.get("state_capture_s")
        tick_rows.append(
            {
                "t": t,
                "latest_cache_age_ms": (t - feature["capture_s"]) * 1000
                if feature["capture_s"] is not None
                else None,
                "executed_image_age_ms": (t - image_capture) * 1000
                if image_capture is not None
                else None,
                "executed_state_age_ms": (t - state_capture) * 1000
                if state_capture is not None
                else None,
                "tick": row["tick"],
                "d": row.get("action_delay_budget_ticks", row.get("d")),
                "clean_slots": sum(not s.get("fallback", False) for s in slots),
                "fresh_slots": sum(
                    s.get("producer_kind") == "request" and not s.get("fallback")
                    for s in slots
                ),
                "future_slots": sum(
                    not s.get("committed") and not s.get("fallback") for s in slots
                ),
                "fallback": row.get("fallback", False),
                "lateness_ms": row.get("lateness_s", 0) * 1000,
                "command_age_ticks": row.get("command_age_ticks"),
                "action_buffer": row.get("action_buffer", {}),
                "executed_slot": row.get("executed_slot"),
            }
        )
    misses = [
        {
            "t": r["deadline_s"],
            "request_tick": r["request_tick"],
            "completed_s": r["completed_s"],
            "overshoot_ms": (r["completed_s"] - r["deadline_s"]) * 1000
            if r["completed_s"] is not None
            else None,
        }
        for r in requests
        if r["role"] == "S1"
        and r["deadline_s"] is not None
        and r["deadline_s"] <= stop - start
        and (r["completed_s"] is None or r["completed_s"] > r["deadline_s"])
    ]
    delay = begin.get("initial_delay_ticks")
    delay_transitions = []
    for row in by_kind["request_completed"]:
        if row["t_s"] > stop or not row.get("deadline_miss"):
            continue
        timing = row.get("client_timing", {})
        submitted, completed = timing.get("submitted_s"), timing.get("completed_s")
        deadline = timing.get("deadline_s")
        required = (
            max(
                1,
                math.ceil(
                    (completed - submitted + 0.005) / begin.get("period_s", 0.05)
                ),
            )
            if submitted is not None and completed is not None
            else None
        )
        after = (
            max(delay, min(5, required))
            if delay is not None and required is not None
            else None
        )
        delay_transitions.append(
            {
                "t": relative(row["t_s"]),
                "request_tick": row.get("audit", {}).get("request_tick"),
                "before": delay,
                "after": after,
                "required_ticks": required,
                "overshoot_ms": (completed - deadline) * 1000
                if completed is not None and deadline is not None
                else None,
            }
        )
        delay = after
    metrics = {
        "controlled_seconds": stop - start,
        "ticks": len(ticks),
        "success": metadata.get("success"),
        "termination_reason": end.get("reason"),
        "action_rpc_ms": summary(action_durations),
        "vision_rpc_ms": summary(vision_durations),
        "cache_age_ms": summary(r["age_ms"] for r in cache_samples),
        "clean_slots": summary(r["clean_slots"] for r in tick_rows),
        "fresh_slots": summary(r["fresh_slots"] for r in tick_rows),
        "delay_ticks": summary(r["d"] for r in tick_rows),
        "lateness_ms": summary(r["lateness_ms"] for r in tick_rows),
        "camera_offered": offered,
        "camera_replaced": replaced,
        "frame_replacement_fraction": fraction(replaced, offered),
        "fallback_ticks": fallback,
        "fallback_fraction": fraction(fallback, len(ticks)),
        "expired_slots": expired,
        "due_generated_slots": due,
        "expired_slot_fraction": fraction(expired, due),
        "censored_future_slots": censored_future,
        "deadline_misses": len(misses),
        "action_censored_requests": sum(
            r["censored"] for r in requests if r["role"] == "S1"
        ),
        "vision_censored_requests": sum(
            r["censored"] for r in requests if r["role"] == "S2"
        ),
    }
    for key in [
        "latest_cache_age_ms",
        "executed_image_age_ms",
        "executed_state_age_ms",
    ]:
        metrics[key] = summary(r[key] for r in tick_rows)
    queue_events = [
        {"t": r["t_s"] - start, "kind": r["kind"], "camera_queue": r["camera_queue"]}
        for r in inside
        if "camera_queue" in r
    ]
    provenance_fields = [
        "task_id",
        "episode_id",
        "seed",
        "task",
        "initial_state_sha256",
        "settled_sim_state_sha256",
        "initial_rgb_sha256",
        "fixture_model_sha256",
        "control_rate_valid",
        "first_success_tick",
    ]
    return {
        "start_s": start,
        "end_s": stop,
        "duration_s": stop - start,
        "period_s": begin.get("period_s", 0.05),
        "metrics": metrics,
        "initial_delay": begin.get("initial_delay_ticks"),
        "initial_capture_s": relative(begin.get("initial_capture_s")),
        "bootstrap_slots": begin.get("bootstrap_action_slots", 0),
        "initial_feature": {
            "feature_sequence": begin.get("feature_sequence"),
            "feature_source_tick": begin.get("feature_source_tick"),
            "feature_capture_s": relative(begin.get("feature_capture_s")),
        },
        "metadata": {k: metadata.get(k) for k in provenance_fields},
        "ticks": tick_rows,
        "requests": requests,
        "camera_events": queue_events,
        "cache_samples": cache_samples,
        "feature_updates": feature_updates,
        "delay_transitions": delay_transitions,
        "miss_events": misses,
    }


def condition_report(directory):
    hardware = read_json(directory / "hardware.json", {})
    metadata = read_jsonl(directory / "episodes.jsonl")
    by_key = {(r.get("task_id"), r.get("episode_id")): r for r in metadata}
    episodes = []
    for path in sorted(directory.glob("task*-episode*-trace.jsonl")):
        parts = path.stem.split("-")
        key = (int(parts[0][4:]), int(parts[1][7:]))
        events = read_jsonl(path)
        if not any(r.get("kind") == "control_end" for r in events):
            continue
        episode = summarize_episode(
            events, by_key.get(key, {"task_id": key[0], "episode_id": key[1]})
        )
        episode["id"] = path.stem.removesuffix("-trace")
        episodes.append(episode)
    if not episodes:
        return None
    calibration = read_json(directory / "calibration.json", {})
    warmup = read_json(directory / "slow-warmup.json", {})
    action_solo = calibration.get("action_rpc_seconds", [])
    vision_solo = [
        r["rpc_seconds"] for r in warmup.get("calls", []) if not r.get("excluded")
    ]
    samples = read_jsonl(directory / "telemetry.jsonl")
    windows = [(e["start_s"], e["end_s"]) for e in episodes]
    gpu_ids = hardware.get("inference_gpus", sorted({r["gpu"] for r in samples}))
    energy = integrate_telemetry(samples, windows, gpu_ids)
    for episode in episodes:
        a, b = episode["start_s"], episode["end_s"]
        episode["energy"] = integrate_telemetry(samples, [(a, b)], gpu_ids)
        episode["telemetry"] = [
            {**r, "t_s": r["t_s"] - a}
            for r in samples
            if a <= r["t_s"] <= b and r["gpu"] in gpu_ids
        ]
    metrics = [e["metrics"] for e in episodes]
    duration = sum(m["controlled_seconds"] for m in metrics)
    flat = {
        "layout": hardware.get("layout", directory.parent.name),
        "condition": hardware.get("condition", directory.name),
        "episodes": len(episodes),
        "successes": sum(m["success"] is True for m in metrics),
        "control_seconds": duration,
        "ticks": sum(m["ticks"] for m in metrics),
        "s1_solo_capacity_rps": fraction(1, summary(action_solo)["mean"]),
        "s2_solo_capacity_rps": fraction(1, summary(vision_solo)["mean"]),
        "s1_solo_samples": len(action_solo),
        "s2_solo_samples": len(vision_solo),
        "s1_solo_mean_ms": (summary(action_solo)["mean"] or 0) * 1000 or None,
        "s2_solo_mean_ms": (summary(vision_solo)["mean"] or 0) * 1000 or None,
        "sampled_energy_j": energy["sampled_energy_j"],
        "counter_energy_j": energy["counter_energy_j"],
        "average_power_w": energy["average_power_w"],
        "energy_coverage": energy["coverage_fraction"],
        "joules_per_control_tick": fraction(
            energy["sampled_energy_j"], sum(m["ticks"] for m in metrics)
        )
        if energy["sampled_energy_j"] is not None
        else None,
        "effective_sm_mhz": summary(d["sm_mhz_mean"] for d in energy["devices"])[
            "mean"
        ],
        "control_rate_valid_episodes": sum(
            e["metadata"].get("control_rate_valid") is True for e in episodes
        ),
    }
    role_gpus = hardware.get("role_gpus") or {
        "S1": gpu_ids[0] if gpu_ids else None,
        "S2": gpu_ids[-1] if gpu_ids else None,
    }
    for role, name in [("S1", "s1"), ("S2", "s2")]:
        device = next((d for d in energy["devices"] if d["gpu"] == role_gpus[role]), {})
        flat[f"{name}_gpu"] = role_gpus[role]
        flat[f"{name}_effective_sm_mhz"] = device.get("sm_mhz_mean")
        flat[f"{name}_board_power_w"] = device.get("average_power_w")
        flat[f"{name}_power_limit_w"] = (
            hardware.get("settings", {})
            .get(str(role_gpus[role]), {})
            .get("power_limit_w")
        )
        values = [
            (r["completed_s"] - r["start_s"]) * 1000
            for e in episodes
            for r in e["requests"]
            if r["role"] == role and not r["censored"]
        ]
        stats = summary(values)
        for p in ["count", "mean", "p50", "p95"]:
            flat[
                f"{name}_rpc_{p}_ms" if p != "count" else f"{name}_complete_requests"
            ] = stats[p]
        flat[f"{name}_censored_requests"] = sum(
            r["censored"] for e in episodes for r in e["requests"] if r["role"] == role
        )
    for output, values in [
        ("used_cache_age", [r["age_ms"] for e in episodes for r in e["cache_samples"]]),
        ("clean_slots", [r["clean_slots"] for e in episodes for r in e["ticks"]]),
        ("fresh_slots", [r["fresh_slots"] for e in episodes for r in e["ticks"]]),
        ("d", [r["d"] for e in episodes for r in e["ticks"]]),
        (
            "latest_cache_age_ms",
            [r["latest_cache_age_ms"] for e in episodes for r in e["ticks"]],
        ),
        (
            "executed_image_age_ms",
            [r["executed_image_age_ms"] for e in episodes for r in e["ticks"]],
        ),
        (
            "executed_state_age_ms",
            [r["executed_state_age_ms"] for e in episodes for r in e["ticks"]],
        ),
    ]:
        stats = summary(values)
        for p in ["mean", "p95", "max"]:
            flat[f"{output}_{p}"] = stats[p]
    for key in [
        "fallback_ticks",
        "expired_slots",
        "due_generated_slots",
        "censored_future_slots",
        "camera_offered",
        "camera_replaced",
        "deadline_misses",
    ]:
        flat[key] = sum(m[key] for m in metrics)
    for key, num, den in [
        ("fallback_fraction", "fallback_ticks", "ticks"),
        ("expired_slot_fraction", "expired_slots", "due_generated_slots"),
        ("frame_replacement_fraction", "camera_replaced", "camera_offered"),
    ]:
        flat[key] = fraction(flat[num], flat[den])
    return {
        "layout": flat["layout"],
        "condition": flat["condition"],
        "id": f"{flat['layout']}/{flat['condition']}",
        "summary": flat,
        "hardware": hardware,
        "energy": energy,
        "episodes": episodes,
        "solo": {
            "S1": summary(v * 1000 for v in action_solo),
            "S2": summary(v * 1000 for v in vision_solo),
        },
        "complete": hardware.get("returncode") == 0,
    }


def paired_initial_states(conditions):
    groups = defaultdict(list)
    fields = [
        "initial_state_sha256",
        "settled_sim_state_sha256",
        "initial_rgb_sha256",
        "fixture_model_sha256",
    ]
    for condition in conditions:
        for episode in condition["episodes"]:
            m = episode["metadata"]
            key = (m["task_id"], m["episode_id"], m["seed"])
            groups[key].append(
                {"condition": condition["id"], **{k: m.get(k) for k in fields}}
            )
    result = []
    for key, rows in sorted(groups.items()):
        equal = {
            field: len(rows) > 1
            and all(r[field] is not None for r in rows)
            and len({json.dumps(r[field], sort_keys=True) for r in rows}) == 1
            for field in fields
        }
        result.append(
            {
                "task_id": key[0],
                "episode_id": key[1],
                "seed": key[2],
                "observations": len(rows),
                "equal": equal,
                "conditions": rows,
            }
        )
    return result


def model_comparisons(conditions, model_path):
    inputs = []
    for condition in conditions:
        s = condition["summary"]
        if not s["s1_solo_mean_ms"] or not s["s2_solo_mean_ms"]:
            continue
        for mode in ["independent", "shared", "serial"]:
            inputs.append(
                {
                    "id": condition["id"],
                    "mode": mode,
                    "config": {
                        "actionMs": s["s1_solo_mean_ms"],
                        "vlmMs": s["s2_solo_mean_ms"],
                        "periodMs": condition["episodes"][0]["period_s"] * 1000,
                        "cameraHz": 1 / condition["episodes"][0]["period_s"],
                        "seconds": 10,
                        "jitter": 0,
                        "computeMode": mode,
                        "slowdownA": 0.5,
                        "slowdownV": 0.5,
                    },
                }
            )
    code = "const m=require(process.argv[1]);let b='';process.stdin.on('data',d=>b+=d);process.stdin.on('end',()=>console.log(JSON.stringify(JSON.parse(b).map(x=>({...x,metrics:m.simulatePolicyQueues(x.config).metrics})))));"
    try:
        result = subprocess.run(
            ["node", "-e", code, str(model_path)],
            input=json.dumps(inputs),
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        comparisons = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        return {"status": "unavailable", "reason": str(error), "inputs": inputs}
    return {
        "status": "illustrative_only",
        "scope": "Existing simulatePolicyQueues, 10 seconds of deterministic mean solo RPC service. Shared slowdown=0.5 is an assumption, not a fitted measurement. Its synthetic d calibration, resource overlap, and drain accounting are model outputs, not GPU measurements or quality predictions.",
        "comparisons": comparisons,
    }


def write_figures(conditions, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    colors = {"single": "#2166ac", "dual": "#b65f14"}
    panels = [
        (
            "s2_solo_capacity_rps",
            "used_cache_age_p95",
            "S2 solo capacity (requests/s)",
            "Used cache age p95 (ms)",
        ),
        (
            "s2_solo_capacity_rps",
            "frame_replacement_fraction",
            "S2 solo capacity (requests/s)",
            "Replaced / offered frames",
        ),
        (
            "s1_solo_capacity_rps",
            "fresh_slots_mean",
            "S1 solo capacity (requests/s)",
            "Mean buffer slots from S1",
        ),
        (
            "s1_solo_capacity_rps",
            "d_mean",
            "S1 solo capacity (requests/s)",
            "Mean delay d (ticks)",
        ),
        (
            "s1_solo_capacity_rps",
            "expired_slot_fraction",
            "S1 solo capacity (requests/s)",
            "Expired / due generated slots",
        ),
        (
            "s1_solo_capacity_rps",
            "fallback_fraction",
            "S1 solo capacity (requests/s)",
            "Fallback / control ticks",
        ),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (x, y, xlabel, ylabel) in zip(axes.flat, panels):
        for condition in conditions:
            s = condition["summary"]
            if s.get(x) is None or s.get(y) is None:
                continue
            ax.scatter(
                s[x],
                s[y],
                color=colors.get(s["layout"], "#444"),
                marker="o" if s["layout"] == "single" else "s",
                s=45,
            )
            annotate_highlights(ax, s, x, y)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.2)
    figure.suptitle(
        "Measured queue state versus measured solo capacity\nBlue circles: single GPU · orange squares: dual GPU · capacity = 1 / mean complete solo RPC",
        fontsize=13,
    )
    save_figure(figure, output / "capacity-queues")
    plt.close(figure)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (x, y, xlabel, ylabel) in zip(
        axes.flat,
        [
            (
                "s1_effective_sm_mhz",
                "s1_solo_capacity_rps",
                "S1 observed SM clock (MHz)",
                "S1 solo capacity (requests/s)",
            ),
            (
                "s2_effective_sm_mhz",
                "s2_solo_capacity_rps",
                "S2 observed SM clock (MHz)",
                "S2 solo capacity (requests/s)",
            ),
            (
                "average_power_w",
                "used_cache_age_p95",
                "Mean summed board power (W)",
                "Used cache age p95 (ms)",
            ),
            (
                "s1_solo_capacity_rps",
                "s1_rpc_p95_ms",
                "S1 solo capacity (requests/s)",
                "Concurrent S1 full RPC p95 (ms)",
            ),
            (
                "s2_solo_capacity_rps",
                "s2_rpc_p95_ms",
                "S2 solo capacity (requests/s)",
                "Concurrent S2 full RPC p95 (ms)",
            ),
            (
                "control_seconds",
                "sampled_energy_j",
                "Controlled exposure (s)",
                "Summed board energy (J)",
            ),
        ],
    ):
        for condition in conditions:
            s = condition["summary"]
            if s.get(x) is None or s.get(y) is None:
                continue
            ax.scatter(
                s[x],
                s[y],
                color=colors.get(s["layout"], "#444"),
                marker="o" if s["layout"] == "single" else "s",
                s=45,
            )
            annotate_highlights(ax, s, x, y)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.2)
    figure.suptitle(
        "Measured hardware, latency and controlled-window energy\nPer-role effective clock is sampled, not assumed from the requested setting",
        fontsize=13,
    )
    save_figure(figure, output / "hardware-energy")
    plt.close(figure)
    write_sweep_figures(conditions, output, plt, colors)
    for condition in conditions:
        e = condition["episodes"][0]
        figure, axes = plt.subplots(
            4, 1, figsize=(14, 8), sharex=True, constrained_layout=True
        )
        for role, y, color in [("S1", 1, "#2166ac"), ("S2", 0, "#b65f14")]:
            rows = [r for r in e["requests"] if r["role"] == role]
            axes[0].broken_barh(
                [(r["start_s"], r["end_s"] - r["start_s"]) for r in rows],
                (y, 0.65),
                facecolors=color,
            )
        axes[0].set(
            yticks=[0.3, 1.3],
            yticklabels=["S2 visual", "S1 action"],
            ylabel="Host RPCs",
        )
        axes[1].plot(
            [r["t"] for r in e["cache_samples"]],
            [r["age_ms"] for r in e["cache_samples"]],
            ".-",
            color="#b65f14",
            linewidth=1,
        )
        axes[1].set_ylabel("Used cache (ms)")
        for key, label, color in [
            ("clean_slots", "Non-fallback incl. bootstrap", "#6699bb"),
            ("fresh_slots", "From S1 request", "#2166ac"),
            ("d", "Delay d", "#7f3c8d"),
        ]:
            axes[2].step(
                [r["t"] for r in e["ticks"]],
                [r[key] for r in e["ticks"]],
                where="post",
                label=label,
                color=color,
            )
        axes[2].set_ylabel("Slots / ticks")
        axes[2].legend(loc="upper right", fontsize=8)
        axes[3].plot(
            [r["t"] for r in e["ticks"]],
            [r["lateness_ms"] for r in e["ticks"]],
            color="#555",
        )
        axes[3].scatter(
            [r["t"] for r in e["ticks"] if r["fallback"]],
            [0 for r in e["ticks"] if r["fallback"]],
            marker="x",
            color="#c23b3b",
            label="Fallback",
        )
        axes[3].axhline(5, color="#c23b3b", linestyle="--", linewidth=0.8)
        axes[3].set(
            xlabel="Seconds since control start",
            ylabel="Lateness (ms)",
            xlim=(0, e["duration_s"]),
        )
        for ax in axes:
            ax.grid(alpha=0.2)
        figure.suptitle(
            f"{condition['id']} · {e['id']} · real LIBERO trace\nHost RPC overlap does not establish CUDA kernel overlap",
            fontsize=13,
        )
        save_figure(
            figure, output / f"trace-{condition['layout']}-{condition['condition']}"
        )
        plt.close(figure)


def annotate_highlights(ax, row, x, y):
    if row["condition"] not in {
        "default",
        "default_repeat",
        "core210",
        "core300",
        "power100",
    }:
        return
    ax.annotate(
        f"{row['layout'][0]}:{row['condition']}",
        (row[x], row[y]),
        xytext=(4, 8 if row["layout"] == "single" else -13),
        textcoords="offset points",
        fontsize=7,
    )


def write_sweep_figures(conditions, output, plt, colors):
    panels = [
        ("s1_rpc_p95_ms", "Concurrent S1 complete RPC p95 (ms)"),
        ("s2_rpc_p95_ms", "Concurrent S2 complete RPC p95 (ms)"),
        ("used_cache_age_p95", "Used cache age p95 (ms)"),
        ("frame_replacement_fraction", "Replaced / offered frames"),
        ("average_power_w", "Mean summed board power (W)"),
        ("d_mean", "Mean delay d (ticks)"),
    ]
    for kind in ["core", "power"]:
        xkey = "s2_effective_sm_mhz" if kind == "core" else "s2_power_limit_w"
        xlabel = (
            "Measured S2 SM clock (MHz)"
            if kind == "core"
            else "Configured power limit per GPU (W)"
        )
        figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for ax, (key, ylabel) in zip(axes.flat, panels):
            for layout in sorted({c["layout"] for c in conditions}):
                rows = [c["summary"] for c in conditions if c["layout"] == layout]
                if kind == "core":
                    selected = [
                        s
                        for s in rows
                        if s["condition"].startswith("core")
                        or s["condition"] == "default"
                    ]
                else:
                    selected = [
                        s
                        for s in rows
                        if s["condition"].startswith("power")
                        or s["condition"] == "core2505"
                    ]
                selected = sorted(
                    (
                        s
                        for s in selected
                        if s.get(xkey) is not None and s.get(key) is not None
                    ),
                    key=lambda s: s[xkey],
                )
                ax.plot(
                    [s[xkey] for s in selected],
                    [s[key] for s in selected],
                    color=colors.get(layout, "#444"),
                    marker="o" if layout == "single" else "s",
                    markersize=5,
                    linewidth=1.3,
                    label=layout,
                )
                if kind == "core":
                    repeat = [s for s in rows if s["condition"] == "default_repeat"]
                    for s in repeat:
                        ax.scatter(
                            s[xkey],
                            s[key],
                            facecolors="white",
                            edgecolors=colors.get(layout, "#444"),
                            s=65,
                            marker="D",
                            linewidths=1.5,
                            label=f"{layout} baseline repeat",
                            zorder=4,
                        )
            ax.set(xlabel=xlabel, ylabel=ylabel)
            ax.grid(alpha=0.2)
            ax.margins(x=0.06, y=0.12)
        axes.flat[0].legend(fontsize=8)
        title = (
            "Core-frequency sweep: observed discrete conditions"
            if kind == "core"
            else "Power-limit sweep at requested 2505 MHz: observed discrete conditions"
        )
        detail = (
            "Lines only connect observations; hollow diamonds show baseline repeats. S1 clock may differ by layout."
            if kind == "core"
            else "300 W point uses core2505. Actual summed board power is measured; configured limits need not bind."
        )
        figure.suptitle(title + "\n" + detail, fontsize=12)
        save_figure(figure, output / f"{kind}-sweep")
        plt.close(figure)


def save_figure(figure, path):
    figure.savefig(path.with_suffix(".png"), dpi=180, facecolor="white")
    figure.savefig(path.with_suffix(".svg"), facecolor="white")


def export_report(input_path, output, include_diagnostics=False, no_figures=False):
    candidates = (
        [input_path]
        if (input_path / "hardware.json").exists()
        else sorted(p.parent for p in input_path.glob("*/*/hardware.json"))
    )
    conditions, skipped = [], []
    for directory in candidates:
        if not include_diagnostics and directory.name.startswith(
            ("smoke", "clock-probe", "profile")
        ):
            skipped.append(str(directory.relative_to(input_path)))
            continue
        condition = condition_report(directory)
        if condition is not None:
            conditions.append(condition)
    if not conditions:
        raise ValueError(
            "No completed control windows found (use --include-diagnostics for smoke/probe previews)"
        )
    output.mkdir(parents=True, exist_ok=True)
    template_dir = Path(__file__).resolve().parents[1] / "simulator"
    model = model_comparisons(conditions, template_dir / "model.js")
    report = {
        "schema_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Selected GR00T LIBERO Spatial episodes; not full-suite success evaluation.",
        "method": {
            "独占容量": "完整独占 RPC 校准样本算术均值的倒数；排除预热调用，不做频率到容量的线性外推。",
            "延迟统计": "仅统计控制开始至结束内接纳且完成的请求；跨越结束时刻的请求保留为截尾条，不进入主延迟分位数。",
            "使用缓存": "DiT plan RPC 开始时实际选中特征的年龄；包含最终未被接纳的已提交请求。",
            "执行年龄": "执行动作按 executed_slot.producer_request_tick 追溯生产请求，不能用最新缓存代替。启动动作使用真实初始采集时刻；fallback 或来源缺失保持未知。汇总按执行 tick 采样。",
            "依赖关系": "S1 只采用 action_completed / client_timing 的 feature_sequence；S2 按 source_tick、capture_s、rpc_completed_s 精确对应缓存发布，禁止按时间接近猜测。",
            "时间戳": "请求、初始采集和 feature_updates 使用相对控制起点的秒数；camera_queue 保留原始绝对采集时刻，但状态快照只选择不晚于游标的事件。",
            "延迟规则": "仅消费超时请求时 d=max(d,min(5,ceil((完整 RPC 秒数+0.005)/period)))；只增不减。微小越界可锁定更高 d，不能独立证明硬件吞吐跃迁。",
            "槽统计": "非 fallback 槽包含启动槽；S1 生产槽只计 request 来源。过期比例的分母为回合结束前应执行的生成槽，结束后的未来槽截尾而非过期。",
            "板卡能耗": "在每个控制窗口边界线性插值并单独积分；排除预热、重置和排空。NVML 板卡能耗包含同 GPU 渲染；报告覆盖率与能量计数器独立估计。",
            "配对比较": "成功终止使暴露时长不同；统计汇总真实请求和 tick，不能视为相同暴露或全套件成功率估计。比较前检查初始状态哈希和各角色实际频率。",
            "主机与设备": "时间线为主机 RPC 区间，不是 GPU 内核占用。并发运行表示在同一控制过程中发出请求，内核重叠需独立 profile 证据。",
        },
        "conditions": conditions,
        "paired_initial_states": paired_initial_states(conditions),
        "skipped_diagnostics": skipped,
        "model_reference": model,
    }
    profile_path = input_path / "profiles" / "kernel-overlap.json"
    if profile_path.exists():
        report["profile_evidence"] = read_json(profile_path)
        (output / "kernel-overlap.json").write_text(profile_path.read_text())
    compact = json.dumps(
        report, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    (output / "report-data.json").write_text(compact + "\n")
    summary_data = {k: v for k, v in report.items() if k != "conditions"}
    summary_data["conditions"] = [
        {
            "id": c["id"],
            "summary": c["summary"],
            "energy": c["energy"],
            "hardware": c["hardware"],
            "episodes": [
                {
                    "id": e["id"],
                    "metadata": e["metadata"],
                    "metrics": e["metrics"],
                    "energy": e["energy"],
                }
                for e in c["episodes"]
            ],
        }
        for c in conditions
    ]
    (output / "summary.json").write_text(
        json.dumps(summary_data, indent=2, allow_nan=False) + "\n"
    )
    (output / "simulation-comparison.json").write_text(
        json.dumps(model, indent=2, allow_nan=False) + "\n"
    )
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(conditions[0]["summary"]))
        writer.writeheader()
        writer.writerows(c["summary"] for c in conditions)
    if not no_figures:
        write_figures(conditions, output)
    html = (template_dir / "measured.html").read_text()
    styles = (template_dir / "shared.css").read_text()
    font = template_dir / "fonts" / "SourceHanSansCN-VF.ttf.woff2"
    font_url = 'url("fonts/SourceHanSansCN-VF.ttf.woff2")'
    if font.is_file():
        encoded_font = base64.b64encode(font.read_bytes()).decode("ascii")
        styles = styles.replace(
            font_url, f'url("data:font/woff2;base64,{encoded_font}")'
        )
        license_file = font.with_name("LICENSE.txt")
        if license_file.is_file():
            license_text = license_file.read_text()
            (output / "font-LICENSE.txt").write_text(license_text)
            styles = "/*\n" + license_text.replace("*/", "* /") + "\n*/\n" + styles
    else:
        # Reports remain standalone when only a system Source Han font is available.
        styles = re.sub(
            r",\s*" + re.escape(font_url) + r'\s*format\("woff2"\)', "", styles
        )
    html = html.replace("/*SHARED_STYLES*/", styles)
    html = html.replace(
        "/*REPORT_DATA*/", "const REPORT = " + compact.replace("</", "<\\/") + ";"
    )
    html = html.replace("/*QUEUE_MODEL*/", (template_dir / "model.js").read_text())
    html = html.replace(
        "/*MEASURED_MODEL*/", (template_dir / "measured-model.js").read_text()
    )
    html = html.replace("/*REPORT_APP*/", (template_dir / "measured.js").read_text())
    for name in ["core-sweep", "power-sweep"]:
        html = html.replace(
            f"<!--{name.upper()}-->",
            embedded_figure(output / f"{name}.svg") if not no_figures else "",
        )
    profile_image = input_path / "profiles" / "kernel-overlap.png"
    if not profile_image.exists():
        profile_image = profile_image.with_suffix(".svg")
    profile_html = ""
    if profile_image.exists() and not no_figures:
        profile_html = (
            '<section class="panel static-gallery"><h2>独立内核验证 / Separate kernel evidence</h2><p class="note">Nsight Systems instrumented runs establish actual S1/S2 kernel intersections. These include the final request drain and use separate profiled episodes. Their overlap fractions are not speedups, throughput comparisons, or unprofiled timing estimates. Device IDs in this evidence are process-local CUDA ordinals.</p>'
            + embedded_figure(profile_image)
            + "</section>"
        )
        (output / profile_image.name).write_bytes(profile_image.read_bytes())
    html = html.replace("<!--PROFILE_EVIDENCE-->", profile_html)
    (output / "index.html").write_text(html)
    return report


def embedded_figure(path):
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = "image/png" if path.suffix == ".png" else "image/svg+xml"
    return f'<img style="width:100%;height:auto" alt="{path.stem}" src="data:{mime};base64,{data}"/>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Write full data and interactive HTML without matplotlib or static images",
    )
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Include smoke*, clock-probe*, profile* runs; omitted from comparisons by default",
    )
    args = parser.parse_args()
    report = export_report(
        args.input.resolve(),
        args.output.resolve(),
        args.include_diagnostics,
        args.no_figures,
    )
    print(
        json.dumps(
            {
                "report": str(args.output / "index.html"),
                "conditions": len(report["conditions"]),
                "episodes": sum(len(c["episodes"]) for c in report["conditions"]),
            }
        )
    )


if __name__ == "__main__":
    main()
