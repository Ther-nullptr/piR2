"""Count actual feature consumption in a schema-2 measured queue report.

Run ``python -m coexecution.feature_usage --report REPORT --output DIRECTORY``;
``--raw-root`` additionally checks source identities against original trace files.
No model is loaded, and no particular GPU, task, or frequency is assumed.

Main utilization excludes bootstrap and each episode's last, right-censored
feature. Reads follow authoritative feature_sequence, never nearest timestamps:
cache selection precedes optional feature installation and plan RPC submission,
so a plan may legitimately start after its selected feature was overwritten.
Direct execution follows the last producer request only, not latent influence
through a rolling denoising buffer. Missing provenance is counted separately;
with incomplete provenance, observed counts are lower bounds even for fully
observed cache lifetimes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _field(record, key, kind):
    _check(isinstance(record, dict), "expected an object")
    value = record.get(key)
    _check(isinstance(value, kind), f"{key}: expected {kind.__name__}")
    return value


def _number(value, name, optional=False):
    if optional and value is None:
        return None
    _check(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value),
        f"{name}: expected a finite number",
    )
    return value


def _identifier(value, name, optional=False):
    if optional and value is None:
        return None
    _check(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{name}: expected a nonnegative integer",
    )
    return value


def _unique(rows, key, label):
    result = {}
    for row in rows:
        _check(isinstance(row, dict) and key in row, f"{label}: missing {key}")
        identity = row[key]
        _check(
            isinstance(identity, (str, int)) or identity is None,
            f"{label}: invalid {key}",
        )
        _check(identity not in result, f"duplicate {label}: {identity}")
        result[identity] = row
    return result


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _identity(condition, episode):
    meta = _field(episode, "metadata", dict)
    valid = meta.get("control_rate_valid")
    _check(
        valid is None or isinstance(valid, bool),
        "control_rate_valid: expected bool or null",
    )
    return {
        "layout": _field(condition, "layout", str),
        "condition": _field(condition, "condition", str),
        "condition_id": _field(condition, "id", str),
        "task_id": meta.get("task_id"),
        "episode_id": _field(episode, "id", str),
        "control_rate_valid": valid,
    }


def analyze_episode(condition, episode):
    """Return feature rows and a pooled-count summary for one observed window.

    Requests need known sequence, finite in-window use_s, and a completion record;
    completion after the window is allowed. Missing or unmatched requests are
    never assigned to the latest cache. selection_times_s stores the submission
    time as an approximate upper bound on cache selection, not a device event.
    """
    ident = _identity(condition, episode)
    duration = _number(episode.get("duration_s"), "duration_s")
    _check(duration >= 0, "duration_s must be nonnegative")
    updates = _field(episode, "feature_updates", list)
    transitions = _field(episode, "delay_transitions", list)
    initial = _field(episode, "initial_feature", dict)
    _check("feature_sequence" in initial, "initial_feature: missing feature_sequence")
    _identifier(initial["feature_sequence"], "initial feature_sequence", optional=True)
    _number(
        initial.get("feature_capture_s"), "initial feature_capture_s", optional=True
    )
    for update in updates:
        _check(isinstance(update, dict), "feature update must be an object")
        _identifier(update.get("feature_sequence"), "feature_sequence")
        published = _number(update.get("t"), "feature publication")
        _check(0 <= published <= duration, "feature publication outside control window")
        _number(update.get("capture_s"), "feature capture_s", optional=True)
    updates = sorted(updates, key=lambda update: update["t"])
    rows = []
    for index, update in enumerate([None, *updates]):
        bootstrap = update is None
        following = updates[index] if index < len(updates) else None
        rows.append(
            {
                **ident,
                "feature_sequence": initial["feature_sequence"]
                if bootstrap
                else update["feature_sequence"],
                "display_feature": "boot" if bootstrap else f"S2-{index - 1:02d}",
                "bootstrap": bootstrap,
                "source_tick": initial.get("feature_source_tick")
                if bootstrap
                else update.get("source_tick"),
                "capture_s": initial.get("feature_capture_s")
                if bootstrap
                else update.get("capture_s"),
                "published_s": None if bootstrap else update["t"],
                "observed_start_s": 0.0 if bootstrap else update["t"],
                "overwritten_s": following["t"] if following else None,
                "observed_end_s": following["t"] if following else duration,
                "right_censored": following is None,
                "n_reads": 0,
                "n_direct_requests": 0,
                "n_execution_ticks": 0,
                "read_request_ticks": [],
                "selection_times_s": [],
                "use_times_s": [],
                "read_request_delays": [],
                "direct_request_ticks": [],
                "execution_ticks": [],
                "execution_times_s": [],
                "n_use_after_overwrite": 0,
                "n_read_requests_finishing_after_window": 0,
            }
        )
    by_feature = _unique(rows, "feature_sequence", "feature sequence")
    requests = _field(episode, "requests", list)
    for request in requests:
        _check(
            isinstance(request, dict) and request.get("role") in {"S1", "S2"},
            "request role must be S1 or S2",
        )
    actions = [r for r in requests if r["role"] == "S1"]
    by_request = _unique(actions, "request_tick", "request tick")
    known_reads = unmatched_reads = missing_reads = outside_reads = 0
    qualified = set()
    for request in actions:
        tick = _identifier(request["request_tick"], "request_tick")
        start = _number(request.get("start_s"), "start_s")
        use = _number(request.get("use_s"), "use_s", optional=True)
        completed = _number(request.get("completed_s"), "completed_s", optional=True)
        sequence = _identifier(
            request.get("feature_sequence"), "request feature_sequence", optional=True
        )
        capture = _number(
            request.get("feature_capture_s"), "request feature_capture_s", optional=True
        )
        if use is None or sequence is None or completed is None:
            missing_reads += 1
            continue
        _check(
            start <= use <= completed,
            "request times must satisfy start_s <= use_s <= completed_s",
        )
        if not 0 <= use < duration:
            outside_reads += 1
            continue
        known_reads += 1
        row = by_feature.get(sequence)
        if row is None:
            unmatched_reads += 1
            continue
        source_tick = request.get("feature_source_tick")
        if source_tick is not None and row["source_tick"] is not None:
            _check(source_tick == row["source_tick"], "feature source_tick mismatch")
        if capture is not None and row["capture_s"] is not None:
            _check(abs(capture - row["capture_s"]) < 1e-8, "feature capture mismatch")
        row["n_reads"] += 1
        row["read_request_ticks"].append(tick)
        row["selection_times_s"].append(start)
        row["use_times_s"].append(use)
        row["read_request_delays"].append(request.get("delay_ticks"))
        row["n_use_after_overwrite"] += int(
            row["overwritten_s"] is not None and use >= row["overwritten_s"]
        )
        row["n_read_requests_finishing_after_window"] += int(completed > duration)
        qualified.add(tick)
    ticks = _field(episode, "ticks", list)
    _unique(ticks, "tick", "control tick")
    bootstrap_execution = fallback_execution = unmatched_execution = 0
    for tick in ticks:
        _identifier(tick["tick"], "control tick")
        when = _number(tick.get("t"), "control tick t")
        _check(0 <= when <= duration, "control tick outside control window")
        slot = tick.get("executed_slot") or {}
        _check(isinstance(slot, dict), "executed_slot must be an object or null")
        if tick.get("fallback") or slot.get("fallback"):
            fallback_execution += 1
            continue
        if slot.get("producer_kind") == "bootstrap":
            bootstrap_execution += 1
            continue
        producer = slot.get("producer_request_tick")
        if slot.get("producer_kind") != "request" or producer not in qualified:
            unmatched_execution += 1
            continue
        request = by_request[producer]
        row = by_feature[request["feature_sequence"]]
        row["n_execution_ticks"] += 1
        row["execution_ticks"].append(tick["tick"])
        row["execution_times_s"].append(when)
        row["direct_request_ticks"].append(producer)
    for row in rows:
        row["direct_request_ticks"] = sorted(set(row["direct_request_ticks"]))
        row["n_direct_requests"] = len(row["direct_request_ticks"])
        row["used_by_dit"] = row["n_reads"] > 0
        row["directly_executed"] = row["n_direct_requests"] > 0
    published = [row for row in rows if not row["bootstrap"]]
    bootstrap_reads = rows[0]["n_reads"]
    _check(
        sum(row["n_reads"] for row in published) + bootstrap_reads + unmatched_reads
        == known_reads,
        "read accounting mismatch",
    )
    _check(
        known_reads + missing_reads + outside_reads == len(actions),
        "request accounting mismatch",
    )
    _check(
        sum(row["n_execution_ticks"] for row in rows)
        + bootstrap_execution
        + fallback_execution
        + unmatched_execution
        == len(ticks),
        "execution accounting mismatch",
    )
    latest_tick = max(ticks, key=lambda tick: tick["t"]) if ticks else {}
    summary = {
        **ident,
        "duration_s": duration,
        "n_s1_submitted": len(actions),
        "n_known_inwindow_reads": known_reads,
        "n_bootstrap_reads": bootstrap_reads,
        "n_unmatched_reads": unmatched_reads,
        "n_missing_reads": missing_reads,
        "n_outside_window_reads": outside_reads,
        "n_bootstrap_execution_ticks": bootstrap_execution,
        "n_fallback_execution_ticks": fallback_execution,
        "n_unmatched_execution_ticks": unmatched_execution,
        "n_control_ticks": len(ticks),
        "final_d": latest_tick.get("d"),
        "n_d_changes": len(transitions),
        "read_accounting_complete": missing_reads == 0 and unmatched_reads == 0,
        "execution_accounting_complete": unmatched_execution == 0,
        "n_s2_started": sum(r["role"] == "S2" for r in requests),
        "n_s2_completed_after_window": sum(
            r["role"] == "S2"
            and r.get("completed_s") is not None
            and _number(r["completed_s"], "S2 completed_s") > duration
            for r in requests
        ),
    }
    summary.update(summarize_features(published))
    return rows, summary


def summarize_features(features):
    """Counts are pooled, never an unweighted average of episode percentages."""
    all_rows = [row for row in features if not row["bootstrap"]]
    mature = [row for row in all_rows if not row["right_censored"]]
    out = {
        "n_published": len(all_rows),
        "n_mature": len(mature),
        "n_right_censored": len(all_rows) - len(mature),
        "n_censored_used": sum(
            row["n_reads"] > 0 for row in all_rows if row["right_censored"]
        ),
        "n_published_reads": sum(row["n_reads"] for row in all_rows),
        "n_published_direct_requests": sum(
            row["n_direct_requests"] for row in all_rows
        ),
        "n_published_execution_ticks": sum(
            row["n_execution_ticks"] for row in all_rows
        ),
        "n_use_after_overwrite": sum(row["n_use_after_overwrite"] for row in all_rows),
        "n_read_requests_finishing_after_window": sum(
            row["n_read_requests_finishing_after_window"] for row in all_rows
        ),
    }
    for prefix, selected in [("mature", mature), ("all_published", all_rows)]:
        used = [row for row in selected if row["n_reads"] > 0]
        counts = Counter(row["n_reads"] for row in selected)
        out.update(
            {
                f"{prefix}_reads": sum(row["n_reads"] for row in selected),
                f"{prefix}_used_features": len(used),
                f"{prefix}_direct_features": sum(
                    row["n_direct_requests"] > 0 for row in selected
                ),
                f"{prefix}_utilization_read": _ratio(len(used), len(selected)),
                f"{prefix}_utilization_direct_observed": _ratio(
                    sum(row["n_direct_requests"] > 0 for row in selected), len(selected)
                ),
                f"{prefix}_mean_reads": mean(row["n_reads"] for row in selected)
                if selected
                else None,
                f"{prefix}_mean_reads_if_used": mean(row["n_reads"] for row in used)
                if used
                else None,
                f"{prefix}_n_zero": counts[0],
                f"{prefix}_n_one": counts[1],
                f"{prefix}_n_two": counts[2],
                f"{prefix}_n_three_plus": sum(n for k, n in counts.items() if k >= 3),
                f"{prefix}_read_count_distribution": dict(sorted(counts.items())),
            }
        )
        _check(
            sum(
                out[f"{prefix}_n_{key}"] for key in ["zero", "one", "two", "three_plus"]
            )
            == len(selected),
            "feature distribution accounting mismatch",
        )
    return out


def _same(a, b, label):
    equal = a == b
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        equal = abs(a - b) < 1e-8
    _check(equal, f"raw trace mismatch: {label}")


def audit_raw(condition, episode, root):
    """Verify source IDs/publications/executed producers against original logs."""
    root = Path(root).resolve()
    components = [condition["layout"], condition["condition"], episode["id"]]
    _check(
        all(p not in {"", ".", ".."} and Path(p).name == p for p in components),
        "raw trace components must be local path names",
    )
    path = (
        root / components[0] / components[1] / f"{components[2]}-trace.jsonl"
    ).resolve()
    _check(path.is_relative_to(root), "raw trace path escapes raw root")
    raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    base = _number(episode.get("start_s"), "raw audit start_s")
    end = _number(episode.get("end_s"), "raw audit end_s")
    _same(end - base, episode["duration_s"], "control duration")
    completions = _unique(
        [r for r in raw if r.get("kind") == "action_completed"],
        "request_tick",
        "raw completion",
    )
    consumed = [
        r
        for r in raw
        if r.get("kind") in {"request_completed", "unused_result_at_episode_end"}
    ]
    timings = _unique(
        [
            {
                "request_tick": _field(r, "audit", dict).get("request_tick"),
                **_field(r, "client_timing", dict),
            }
            for r in consumed
        ],
        "request_tick",
        "raw client timing",
    )
    publications = _unique(
        [
            r
            for r in raw
            if r.get("kind") == "feature_published"
            and _number(r.get("t_s"), "raw publication t_s") <= end
        ],
        "feature_sequence",
        "raw publication",
    )
    executions = _unique(
        [r for r in raw if r.get("kind") == "control_tick"], "tick", "raw control tick"
    )
    for request in episode["requests"]:
        if request["role"] != "S1":
            continue
        tick = request["request_tick"]
        completed = completions.get(tick, {})
        timing = timings.get(tick, {})
        if request.get("completed_s") is not None:
            _check(bool(completed), f"raw completion missing for request {tick}")
        for key in ["feature_sequence", "feature_source_tick", "feature_capture_s"]:
            if key in completed and key in timing:
                _same(completed[key], timing[key], f"completion/timing {key}")
        actual = {**completed, **timing}
        _same(
            request.get("feature_sequence"),
            actual.get("feature_sequence"),
            "feature sequence",
        )
        _same(
            request.get("feature_source_tick"),
            actual.get("feature_source_tick"),
            "feature source tick",
        )
        capture = actual.get("feature_capture_s")
        _same(
            request.get("feature_capture_s"),
            capture - base if capture is not None else None,
            "feature capture",
        )
        started = timing.get("plan_rpc_started_s")
        _same(
            request.get("use_s"),
            started - base if started is not None else None,
            "plan RPC start",
        )
    _same(len(publications), len(episode["feature_updates"]), "publication count")
    for update in episode["feature_updates"]:
        raw_update = publications.get(update["feature_sequence"])
        _check(raw_update is not None, "raw publication missing")
        _same(update["t"], raw_update["t_s"] - base, "publication time")
        _same(
            update.get("source_tick"),
            raw_update.get("source_tick"),
            "publication source tick",
        )
    _same(len(executions), len(episode["ticks"]), "control tick count")
    for tick in episode["ticks"]:
        raw_tick = executions.get(tick["tick"])
        _check(raw_tick is not None, "raw control tick missing")
        _same(
            tick.get("executed_slot"),
            raw_tick.get("executed_slot"),
            "executed producer",
        )
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_action_source_records": len(completions),
        "n_controlwindow_publications": len(publications),
        "n_control_ticks": len(executions),
    }


def analyze_report(report, raw_root=None):
    """Analyze a schema-2 report without assuming task IDs or frequency names.

    Passing raw_root enables identity checks against layout/condition trace files.
    Missing control_rate_valid is retained as unknown and excluded from valid_only.
    """
    _check(
        isinstance(report, dict) and report.get("schema_version") == 2,
        "expected schema_version=2 measured report",
    )
    input_conditions = _field(report, "conditions", list)
    _unique(input_conditions, "id", "condition id")
    features, episodes, conditions, raw_audits = [], [], [], []
    for condition in input_conditions:
        source_episodes = _field(condition, "episodes", list)
        _unique(source_episodes, "id", "episode id")
        condition_rows, condition_episodes = [], []
        for episode in source_episodes:
            rows, summary = analyze_episode(condition, episode)
            condition_rows.extend(rows)
            condition_episodes.append(summary)
            if raw_root is not None:
                raw_audits.append(audit_raw(condition, episode, raw_root))
        hardware = condition.get("hardware", {})
        source_summary = condition.get("summary", {})
        _check(
            isinstance(hardware, dict) and isinstance(source_summary, dict),
            "hardware and summary must be objects",
        )
        # Preserve reported metadata, never derive compute capacity from MHz or
        # encode a machine's default clocks in condition-name parsing.
        aggregate = {
            "layout": _field(condition, "layout", str),
            "condition": _field(condition, "condition", str),
            "condition_id": _field(condition, "id", str),
            "hardware": {
                k: hardware[k]
                for k in [
                    "requested_profile",
                    "role_gpus",
                    "renderer_gpu",
                    "inference_gpus",
                ]
                if k in hardware
            },
            "n_episodes": len(condition_episodes),
            "n_valid_episodes": sum(
                e["control_rate_valid"] is True for e in condition_episodes
            ),
            "n_unknown_validity_episodes": sum(
                e["control_rate_valid"] is None for e in condition_episodes
            ),
            "duration_s": sum(e["duration_s"] for e in condition_episodes),
            "final_d_by_episode": {
                e["episode_id"]: e["final_d"] for e in condition_episodes
            },
        }
        for field in ["s1_effective_sm_mhz", "s2_effective_sm_mhz", "s1_gpu", "s2_gpu"]:
            aggregate[field] = source_summary.get(field)
        for field in [
            "n_s1_submitted",
            "n_bootstrap_reads",
            "n_known_inwindow_reads",
            "n_unmatched_reads",
            "n_missing_reads",
            "n_outside_window_reads",
            "n_s2_started",
        ]:
            aggregate[field] = sum(e[field] for e in condition_episodes)
        aggregate["read_accounting_complete"] = all(
            e["read_accounting_complete"] for e in condition_episodes
        )
        aggregate["execution_accounting_complete"] = all(
            e["execution_accounting_complete"] for e in condition_episodes
        )
        aggregate.update(summarize_features(condition_rows))
        aggregate["valid_only"] = summarize_features(
            [r for r in condition_rows if r["control_rate_valid"] is True]
        )
        features.extend(condition_rows)
        episodes.extend(condition_episodes)
        conditions.append(aggregate)
    validation = {
        "episodes": len(episodes),
        "conditions": len(conditions),
        "features_including_bootstrap": len(features),
        "source_identity_raw_audits": raw_audits,
        "raw_logs_verified": len(raw_audits),
        "sum_checks": "published reads + bootstrap + unmatched = known reads; known + missing + outside = submitted; attributed action ticks + bootstrap + fallback + unmatched = all ticks",
    }
    for field in [
        "n_known_inwindow_reads",
        "n_unmatched_reads",
        "n_missing_reads",
        "n_outside_window_reads",
        "n_unmatched_execution_ticks",
        "n_use_after_overwrite",
        "n_read_requests_finishing_after_window",
    ]:
        validation[field] = sum(e[field] for e in episodes)
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_report_schema": report["schema_version"],
        "scope": "Observed feature-sequence lineage, not inferred timing or latent rolling-buffer influence. Main utilization excludes bootstrap and terminal right censoring; direct execution is observed only within the control window.",
        "features": features,
        "episodes": episodes,
        "conditions": conditions,
        "validation": validation,
    }


def _write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, required=True, help="Schema-2 measured report-data.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for derived JSON and CSV tables",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="Optional root of original layout/condition trace files",
    )
    args = parser.parse_args(argv)
    source = args.report.read_bytes()
    result = analyze_report(json.loads(source), raw_root=args.raw_root)
    result["source_report"] = args.report.name
    result["source_report_sha256"] = hashlib.sha256(source).hexdigest()
    filenames = ["feature-metrics.json", "per-feature.csv", "feature-summary.csv"]
    _check(
        all(
            (args.output / name).resolve() != args.report.resolve()
            for name in filenames
        ),
        "output would overwrite the input report",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / filenames[0]).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output / filenames[1], result["features"])
    _write_csv(args.output / filenames[2], result["conditions"])
    print(
        json.dumps(
            {
                key: value
                for key, value in result["validation"].items()
                if key != "source_identity_raw_audits"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
