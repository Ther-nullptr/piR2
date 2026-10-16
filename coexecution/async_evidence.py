"""Independent whole-work audit for the one-step lookahead protocol.

This module reads recorded host evidence only. Controlled-window queue statistics
and terminal work are different boundaries; neither can stand in for the other.
"""

import math
from collections import defaultdict
from itertools import pairwise

STAGES = (
    "vlm_started_s",
    "vlm_completed_s",
    "action_started_s",
    "action_completed_s",
)
IDENTITY = ("request_id", "request_tick", "source_tick", "target_tick")


def _require(condition, message):
    if not condition:
        raise ValueError("Lookahead async evidence: " + message)


def _number(value):
    _require(
        type(value) in (int, float) and math.isfinite(value), "nonfinite time/value"
    )
    return value


def _same(actual, expected, label):
    if isinstance(expected, dict):
        _require(isinstance(actual, dict), label + " must be an object")
        for key, value in expected.items():
            _require(key in actual, label + " missing " + key)
            _same(actual[key], value, label + "." + key)
    elif isinstance(expected, list):
        _require(
            isinstance(actual, list) and len(actual) == len(expected), label + " length"
        )
        for i, value in enumerate(expected):
            _same(actual[i], value, f"{label}[{i}]")
    elif type(expected) is float:
        _require(
            type(actual) in (int, float)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9),
            label + " mismatch",
        )
    else:
        _require(
            type(actual) is type(expected) and actual == expected, label + " mismatch"
        )


def _ordered(values, label):
    values = [_number(value) for value in values]
    _require(all(a <= b for a, b in pairwise(values)), label + " order")


def _index(rows, key, label):
    result = {}
    for row in rows:
        value = row.get(key)
        _require(type(value) is int and value >= 0, label + " invalid ID")
        _require(value not in result, label + " duplicate ID")
        result[value] = row
    return result


def _validate_lookahead_work(events, normalized=None, metadata=None, *, chunked=False):
    """Cross-check full normalized work against submissions, stages and execution.

    Returns audited relative work rows and absolute phases. Never clips terminal
    work to control_end. Missing or inconsistent evidence is rejected, not filled.
    """
    groups = defaultdict(list)
    for row in events:
        _number(row.get("t_s"))
        groups[row.get("kind")].append(row)
    for kind in (
        "control_error",
        "policy_request_failed",
        "drain_error",
        "worker_shutdown_error",
    ):
        _require(not groups[kind], "failed/incomplete trace: " + kind)
    for kind in ("control_start", "control_end", "async_work_end"):
        _require(len(groups[kind]) == 1, "requires exactly one " + kind)
    begin, end, finish = (
        groups[kind][0] for kind in ("control_start", "control_end", "async_work_end")
    )
    start, stop, work_end = begin["t_s"], end["t_s"], finish["t_s"]
    _ordered((start, stop, work_end), "work boundaries")
    coupling = "chunked_async" if chunked else "one_step_async"
    chunk = begin.get("execution_prefix_steps") if chunked else 1
    depth = begin.get("async_overlap_steps") if chunked else 1
    maximum = begin.get("max_steps")
    _require(
        type(chunk) is int and type(depth) is int and 1 <= depth <= chunk, "invalid K/D"
    )
    _require(type(maximum) is int and maximum >= 1, "invalid maximum")
    if chunked:
        _same(
            begin,
            {"protocol_version": 2, "synchronous_chunk_steps": chunk},
            "version2 protocol",
        )
    _same(
        begin,
        {
            "execution_coupling": coupling,
            "execution_clock": "ready",
            "execution_prefix_steps": chunk,
        },
        "protocol",
    )
    requests = _index(
        groups["chunked_async_request" if chunked else "one_step_async_request"],
        "request_id",
        "work",
    )
    ticks = _index(groups["control_tick"], "tick", "execution")
    _require(
        bool(ticks) and sorted(ticks) == list(range(len(ticks))),
        "noncontiguous physical ticks",
    )
    adopted_ids = set(range(0, len(ticks), chunk))
    expected_ids = {0} | {
        target for target in range(chunk, maximum, chunk) if target - depth < len(ticks)
    }
    _require(set(requests) == expected_ids, "missing/extra declared request targets")
    _require(len(ticks) <= maximum, "physical step cap exceeded")
    _same(end, {"control_steps": len(ticks)}, "control end")
    _same(
        finish,
        {
            "policy_calls": len(requests),
            "adopted_policy_calls": len(adopted_ids),
            "unused_policy_calls": len(requests) - len(adopted_ids),
        },
        "work end",
    )
    indexed = {}
    for kind in (
        "policy_request_submitted",
        "policy_rpc_started",
        "policy_rpc_completed",
        "vision_request",
        "feature_published",
        "action_submitted",
        "action_started",
        "action_completed",
    ):
        indexed[kind] = _index(groups[kind], "request_id", kind)
        _require(indexed[kind].keys() == requests.keys(), "missing/extra " + kind)
    received = _index(
        groups["request_completed"] + groups["unused_result_at_episode_end"],
        "request_id",
        "result disposition",
    )
    _require(received.keys() == requests.keys(), "missing result disposition")
    adopted = _index(groups["action_adopted"], "request_id", "adoption")
    executed = _index(
        groups["chunked_async_execution" if chunked else "one_step_async_execution"],
        "execution_id" if chunked else "request_id",
        "physical execution",
    )
    _require(
        set(adopted) == adopted_ids and executed.keys() == ticks.keys(),
        "adoption/execution IDs disagree",
    )
    phases = groups["cycle_phase"]
    _require(
        all(
            p.get("phase") in ("inference", "execution", "wait", "drain")
            for p in phases
        ),
        "unknown phase",
    )
    by_phase = {
        name: _index(
            [p for p in phases if p["phase"] == name],
            "execution_id" if chunked and name == "execution" else "request_id",
            name,
        )
        for name in ("inference", "execution", "wait", "drain")
    }
    _require(by_phase["inference"].keys() == requests.keys(), "missing inference phase")
    _require(by_phase["execution"].keys() == ticks.keys(), "missing execution phase")
    _require(set(by_phase["wait"]) <= adopted_ids - {0}, "invalid wait request")
    _require(
        set(by_phase["drain"]) == set(requests) - adopted_ids,
        "missing/extra terminal drain",
    )
    boundaries = (
        _index(groups["chunked_async_boundary"], "request_id", "boundary")
        if chunked
        else {}
    )
    if chunked:
        _require(set(boundaries) == adopted_ids - {0}, "missing/extra boundary check")
    work_rows = []
    for rid, request in sorted(requests.items()):
        source = max(0, rid - depth)
        prefix = min(chunk, maximum - rid)
        executed_steps = min(prefix, max(0, len(ticks) - rid))
        used = rid in adopted_ids
        identity = {
            "request_id": rid,
            "request_tick": rid,
            "source_tick": source,
            "target_tick": rid,
        }
        _same(request, identity, "request identity")
        for kind, rows in indexed.items():
            _same(rows[rid], identity, kind + " identity")
        capture, submitted, rpc_start, rpc_end = (
            request.get(key)
            for key in ("capture_s", "submitted_s", "rpc_started_s", "rpc_completed_s")
        )
        audit = request.get("audit", {})
        hashes = request.get("frame_hashes")
        _require(isinstance(hashes, dict) and bool(hashes), "missing frame hashes")
        _require(
            all(
                isinstance(v, str)
                and len(v) == 64
                and all(c in "0123456789abcdef" for c in v)
                for v in hashes.values()
            ),
            "invalid frame hash",
        )
        _same(
            audit,
            {
                "source_tick": source,
                "request_tick": rid,
                "capture_s": capture,
                "frame_hashes": hashes,
                "forward_counts": {"vlm": 1, "dit": 4},
            },
            "backend audit",
        )
        va, vb, aa, ab = (audit.get(key) for key in STAGES)
        accepted = request.get("received_s") if chunked else request["t_s"]
        _ordered(
            (
                start,
                capture,
                submitted,
                rpc_start,
                va,
                vb,
                aa,
                ab,
                rpc_end,
                accepted,
                work_end,
            ),
            "request/stage",
        )
        if rid:
            prior = ticks[source]
            _same(capture, prior["sensor_capture_s"], "prefetch snapshot capture")
            _ordered(
                (prior["sensor_capture_s"], capture, submitted, prior["applied_s"]),
                "prefetch before physical step",
            )
            if source > 0:
                _ordered(
                    (ticks[source - 1]["next_observation_computed_s"], capture),
                    "source observation availability",
                )
        for kind, moment in (
            ("policy_request_submitted", submitted),
            ("policy_rpc_started", rpc_start),
            ("policy_rpc_completed", rpc_end),
        ):
            _same(
                indexed[kind][rid],
                {
                    "t_s": moment,
                    "capture_s": capture,
                    "frame_hashes": hashes,
                    "submitted_s": submitted,
                },
                kind,
            )
        _same(
            indexed["policy_rpc_started"][rid],
            {"rpc_started_s": rpc_start},
            "rpc start",
        )
        _same(
            indexed["policy_rpc_completed"][rid],
            {"rpc_started_s": rpc_start, "rpc_completed_s": rpc_end},
            "rpc complete",
        )
        dependency = {
            "feature_sequence": rid,
            "feature_source_tick": source,
            "feature_capture_s": capture,
            "state_tick": source,
            "state_capture_s": capture,
        }
        _same(
            indexed["vision_request"][rid],
            {
                "capture_s": capture,
                "feature_sequence": rid,
                "t_s": vb,
                "started_s": va,
                "completed_s": vb,
                "rpc_seconds": vb - va,
                "timing_scope": "server_vlm_stage",
            },
            "VLM stage",
        )
        _same(
            indexed["feature_published"][rid],
            {
                "capture_s": capture,
                "feature_sequence": rid,
                "t_s": vb,
                "rpc_completed_s": vb,
            },
            "feature publication",
        )
        _same(
            indexed["action_submitted"][rid],
            {
                **dependency,
                "t_s": aa,
                "deadline_s": None,
                "delay_ticks": 0,
                "timing_scope": "server_action_stage",
            },
            "DiT submit",
        )
        _same(indexed["action_started"][rid], {"t_s": aa}, "DiT start")
        timing = {
            **dependency,
            "submitted_s": aa,
            "completed_s": ab,
            "policy_submitted_s": submitted,
            "policy_rpc_started_s": rpc_start,
            "policy_rpc_completed_s": rpc_end,
            "timing_scope": "server_action_stage",
            "server_plan_started_s": aa,
            "timing_version": 2,
            "cache_age_at_server_plan_ms": (aa - capture) * 1000,
        }
        _same(request.get("client_timing"), timing, "request timing")
        _same(
            indexed["action_completed"][rid],
            {
                **timing,
                "t_s": ab,
                "request_elapsed_s": ab - aa,
                "valid_start_tick": rid,
                "valid_end_tick": rid + prefix,
            },
            "DiT completion",
        )
        _same(
            request,
            {
                "disposition": "adopted" if used else "unused",
                "accepted_prefix_steps": prefix if used else 0,
                "adopted_s": accepted if used else None,
                "unused_reason": None if used else end.get("reason"),
            },
            "result",
        )
        _same(
            received[rid],
            {
                "kind": "request_completed" if used else "unused_result_at_episode_end",
                "target_tick": rid,
                "t_s": accepted,
                "audit": audit,
                "client_timing": timing,
                "unused_reason": request["unused_reason"],
            },
            "received result",
        )
        native = request.get("native_decoded_actions")
        _require(
            isinstance(native, list) and len(native) > 0, "missing decoded actions"
        )
        _require(
            all(
                isinstance(a, list)
                and len(a) == 7
                and all(type(v) in (int, float) and math.isfinite(v) for v in a)
                for a in native
            ),
            "invalid decoded actions",
        )
        _same(audit, {"decoded_horizon": len(native)}, "native horizon")
        _same(
            request,
            {"unused_plan_suffix_slots": len(native) - executed_steps},
            "unused suffix",
        )
        _same(
            by_phase["inference"][rid],
            {
                **identity,
                "cycle": rid,
                "start_s": rpc_start,
                "end_s": rpc_end,
                "t_s": rpc_end,
                "source_capture_s": capture,
                "frame_hashes": hashes,
                "submitted_s": submitted,
                "simulation_paused": rid == 0,
                "execution_coupling": coupling,
                "execution_clock": "ready",
            },
            "inference phase",
        )
        _require(len(native) >= chunk, "prefix exceeds native horizon")
        if chunked:
            _same(
                request,
                {
                    "protocol_version": 2,
                    "execution_prefix_steps": chunk,
                    "async_overlap_steps": depth,
                    "prefix_steps": prefix,
                    "executed_prefix_steps": executed_steps,
                    "unexecuted_accepted_prefix_steps": (prefix if used else 0)
                    - executed_steps,
                    "discarded_native_suffix_steps": len(native)
                    - (prefix if used else 0),
                    "finalized_s": request["t_s"],
                },
                "finalized request accounting",
            )
            _ordered(
                (accepted, request["finalized_s"], work_end), "request finalization"
            )
            if used:
                _ordered(
                    (
                        ticks[rid + executed_steps - 1]["next_observation_computed_s"],
                        request["finalized_s"],
                    ),
                    "finalize after actual prefix",
                )
            else:
                _ordered((stop, accepted), "unused response after control")
        if used:
            if rid:
                _ordered(
                    (ticks[rid - 1]["next_observation_computed_s"], accepted),
                    "adoption after committed prefix",
                )
            _same(
                adopted[rid],
                {
                    "request_tick": rid,
                    "target_tick": rid,
                    "t_s": accepted,
                    "installed_slots": prefix,
                    "expired_slots": 0,
                    "protected_slots": 0,
                    "ready_to_adopt_seconds": accepted - rpc_end,
                },
                "adoption",
            )
            for physical in range(rid, rid + executed_steps):
                tick = ticks[physical]
                offset = physical - rid
                applied, completed = (
                    tick.get("applied_s"),
                    tick.get("next_observation_computed_s"),
                )
                _ordered((accepted, applied, completed, stop), "action application")
                if physical:
                    _ordered(
                        (ticks[physical - 1]["next_observation_computed_s"], applied),
                        "physical execution",
                    )
                _ordered(
                    (start, tick.get("sensor_capture_s"), applied),
                    "current sensor timestamp",
                )
                _same(
                    tick,
                    {
                        **{k: v for k, v in identity.items() if k != "request_tick"},
                        **dependency,
                        "tick": physical,
                        "action": [
                            float(max(-1, min(1, v))) for v in native[offset][:6]
                        ]
                        + [float(native[offset][6] >= 0.5)],
                        "fallback": False,
                        "t_s": applied,
                        "command_ready_s": rpc_end,
                        "ready_to_apply_seconds": applied - rpc_end,
                        "first_application_of_request": offset == 0,
                        "command_age_ticks": offset,
                        "action_delay_budget_ticks": 0,
                        "action_request_pending": rid + chunk in requests
                        and physical >= rid + chunk - depth,
                        "execution_coupling": coupling,
                        "execution_clock": "ready",
                        "observation_lag_ticks": physical - source,
                        "environment_step_s": completed - applied,
                        "physics_step_seconds": 0.05,
                        "simulated_elapsed_seconds": (physical + 1) * 0.05,
                        "executed_image_age_ms": (applied - capture) * 1000,
                        "executed_state_age_ms": (applied - capture) * 1000,
                    },
                    "physical tick",
                )
                extra = (
                    {
                        "execution_id": physical,
                        "tick": physical,
                        "executed_offset": offset,
                    }
                    if chunked
                    else {}
                )
                if chunked:
                    _same(tick, {"executed_offset": offset}, "physical native offset")
                    if (
                        not (
                            rid + chunk in requests and physical == rid + chunk - depth
                        )
                        and physical
                    ):
                        # A fresh camera update is allowed only at bootstrap or
                        # the declared prefetch boundary; all inner steps reuse
                        # the preceding env.step observation.
                        _same(
                            tick["sensor_capture_s"],
                            ticks[physical - 1]["next_observation_computed_s"],
                            "post-step observation reuse",
                        )
                _same(
                    tick.get("executed_slot"),
                    {
                        "producer_kind": "request",
                        "producer_request_tick": rid,
                        "tick": physical,
                        "fallback": False,
                    },
                    "executed producer",
                )
                buffer = tick.get("action_buffer")
                _same(
                    buffer,
                    {"committed_until": rid + prefix if chunked else physical + 1},
                    "exact prefix",
                )
                slots = buffer.get("slots")
                _require(
                    isinstance(slots, list)
                    and len(slots) == rid + prefix - physical - 1,
                    "buffer prefix length",
                )
                for slot, future_tick in zip(slots, range(physical + 1, rid + prefix)):
                    _same(
                        slot,
                        {
                            "tick": future_tick,
                            "producer_kind": "request",
                            "producer_request_tick": rid,
                            "fallback": False,
                            "committed": chunked,
                            "status": "committed" if chunked else "future",
                        },
                        "buffer prefix producer",
                    )
                _same(
                    executed[physical],
                    {
                        **identity,
                        **extra,
                        "applied_s": applied,
                        "completed_s": completed,
                        "t_s": completed,
                        "executed_steps": 1,
                        "observation_lag_ticks": physical - source,
                    },
                    "execution record",
                )
                _same(
                    by_phase["execution"][physical],
                    {
                        **identity,
                        **extra,
                        "cycle": rid,
                        "start_s": applied,
                        "end_s": completed,
                        "t_s": completed,
                        "physics_start_tick": physical,
                        "physics_end_tick": physical + 1,
                        "simulation_paused": False,
                        "execution_clock": "ready",
                    },
                    "execution phase",
                )
        work_rows.append(
            {
                **identity,
                **{
                    key: request[key]
                    for key in (
                        "disposition",
                        "accepted_prefix_steps",
                        "unused_plan_suffix_slots",
                        "frame_hashes",
                        "unused_reason",
                    )
                },
                **{
                    key: request[key] - start if request[key] is not None else None
                    for key in (
                        "capture_s",
                        "submitted_s",
                        "rpc_started_s",
                        "rpc_completed_s",
                        "adopted_s",
                    )
                },
                "server_stages": {key: audit[key] - start for key in STAGES},
                "forward_counts": audit["forward_counts"],
                "decoded_horizon": audit["decoded_horizon"],
                "completed_after_control": rpc_end > stop,
                **(
                    {
                        "protocol_version": 2,
                        "execution_prefix_steps": chunk,
                        "async_overlap_steps": depth,
                        "executed_prefix_steps": executed_steps,
                        "unexecuted_accepted_prefix_steps": request[
                            "unexecuted_accepted_prefix_steps"
                        ],
                        "discarded_native_suffix_steps": request[
                            "discarded_native_suffix_steps"
                        ],
                        "prefix_steps": prefix,
                        "received_s": accepted - start,
                        "finalized_s": request["finalized_s"] - start,
                    }
                    if chunked
                    else {}
                ),
            }
        )
    for name in ("wait", "drain"):
        for rid, phase in by_phase[name].items():
            request = requests[rid]
            boundary_tick = len(ticks) if name == "drain" else rid
            _same(
                phase,
                {
                    **{key: request[key] for key in IDENTITY},
                    "cycle": rid,
                    "simulation_paused": True,
                    "physics_start_tick": boundary_tick,
                    "physics_end_tick": boundary_tick,
                    "execution_clock": "ready",
                    "t_s": phase.get("end_s"),
                },
                name + " phase",
            )
            a, b = phase.get("start_s"), phase.get("end_s")
            _ordered(
                (
                    ticks[boundary_tick - 1]["next_observation_computed_s"],
                    a,
                    request["received_s"] if chunked else request["t_s"],
                    b,
                    work_end,
                ),
                name + " time",
            )
            if name == "drain":
                _same(a, stop, "drain starts at control end")
            else:
                _ordered((b, ticks[rid]["applied_s"]), "wait precedes next action")
    if chunked:
        for rid, boundary in boundaries.items():
            _same(
                boundary,
                {"target_tick": rid, "physical_tick": rid},
                "boundary identity",
            )
            ready = boundary.get("result_ready")
            _require(type(ready) is bool, "boundary result readiness")
            _ordered(
                (
                    ticks[rid - 1]["next_observation_computed_s"],
                    boundary["t_s"],
                    requests[rid]["received_s"],
                ),
                "boundary receive",
            )
            _require(
                (rid in by_phase["wait"]) is (not ready), "missing/extra boundary wait"
            )
            if not ready:
                _same(
                    by_phase["wait"][rid]["start_s"],
                    boundary["t_s"],
                    "wait boundary start",
                )
            else:
                _ordered(
                    (requests[rid]["rpc_completed_s"], boundary["t_s"]),
                    "ready response before boundary",
                )
    if normalized is not None:
        _same(normalized, work_rows, "normalized full work")
    rpc_intervals = [
        (r["rpc_started_s"], r["rpc_completed_s"]) for r in requests.values()
    ]
    execution_intervals = [
        (t["applied_s"], t["next_observation_computed_s"]) for t in ticks.values()
    ]

    def overlap(intervals):
        return sum(
            max(0.0, min(b, d) - max(a, c))
            for a, b in intervals
            for c, d in execution_intervals
        )

    if metadata:
        for key, value in {
            "execution_coupling": coupling,
            "execution_clock": "ready",
            "control_steps": len(ticks),
            "controlled_start_s": start,
            "controlled_end_s": stop,
            "controlled_wall_seconds": stop - start,
            "actual_control_hz": len(ticks) / (stop - start) if stop > start else None,
            "physical_control_seconds": len(ticks) * 0.05,
            "simulated_elapsed_seconds": len(ticks) * 0.05,
            "total_policy_rpc_seconds": sum(b - a for a, b in rpc_intervals),
            "total_policy_request_seconds": sum(
                r["rpc_completed_s"] - r["submitted_s"] for r in requests.values()
            ),
            "measured_inference_execution_overlap_seconds": overlap(rpc_intervals),
            "measured_server_execution_overlap_seconds": {
                stage: overlap(
                    [
                        (
                            r["audit"][stage + "_started_s"],
                            r["audit"][stage + "_completed_s"],
                        )
                        for r in requests.values()
                    ]
                )
                for stage in ("vlm", "action")
            },
            "policy_calls": len(requests),
            "adopted_policy_calls": len(adopted_ids),
            "unused_policy_calls": len(requests) - len(adopted_ids),
            "forward_counts": {"vlm": len(requests), "dit": 4 * len(requests)},
            "scored_work_end_s": work_end,
            "scored_work_wall_seconds": work_end - start,
            "terminal_drain_seconds": work_end - stop,
            "post_step_wait_seconds": sum(
                p["end_s"] - p["start_s"] for p in by_phase["wait"].values()
            ),
            "unused_plan_suffix_slots": sum(
                r["unused_plan_suffix_slots"] for r in requests.values()
            ),
            **(
                {
                    "execution_prefix_steps": chunk,
                    "async_overlap_steps": depth,
                    "protocol_version": 2,
                    "accepted_prefix_steps": sum(
                        r["accepted_prefix_steps"] for r in requests.values()
                    ),
                    "executed_prefix_steps": len(ticks),
                    "unexecuted_accepted_prefix_steps": sum(
                        r["unexecuted_accepted_prefix_steps"] for r in requests.values()
                    ),
                    "discarded_native_suffix_steps": sum(
                        r["discarded_native_suffix_steps"] for r in requests.values()
                    ),
                }
                if chunked
                else {}
            ),
        }.items():
            if key in metadata:
                _same(metadata[key], value, "metadata " + key)
    return {
        "work_requests": work_rows,
        "phases": phases,
        "work_end_s": work_end,
        "control_end_s": stop,
        "start_s": start,
    }


def validate_one_step_async_work(events, normalized=None, metadata=None):
    """Validate original version1 recordings without changing their schema."""
    try:
        return _validate_lookahead_work(events, normalized, metadata, chunked=False)
    except ValueError as error:
        raise ValueError(
            str(error).replace("Lookahead async evidence", "One-step async evidence")
        ) from error


def validate_chunked_async_work(events, normalized=None, metadata=None):
    """Validate version2 native-prefix execution and lookahead evidence."""
    return _validate_lookahead_work(events, normalized, metadata, chunked=True)


def validate_async_work(events, normalized=None, metadata=None):
    """Select the declared protocol; reject unknown async evidence."""
    begins = [e for e in events if e.get("kind") == "control_start"]
    _require(len(begins) == 1, "requires exactly one control_start")
    coupling = begins[0].get("execution_coupling")
    _require(coupling in ("one_step_async", "chunked_async"), "unknown async protocol")
    validator = (
        validate_chunked_async_work
        if coupling == "chunked_async"
        else validate_one_step_async_work
    )
    return validator(events, normalized, metadata)
