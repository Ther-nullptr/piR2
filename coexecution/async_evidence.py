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
        raise ValueError("One-step async evidence: " + message)


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


def validate_one_step_async_work(events, normalized=None, metadata=None):
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
    _same(
        begin,
        {
            "execution_coupling": "one_step_async",
            "execution_clock": "ready",
            "execution_prefix_steps": 1,
        },
        "protocol",
    )
    requests = _index(groups["one_step_async_request"], "request_id", "work")
    ticks = _index(groups["control_tick"], "tick", "execution")
    _require(
        bool(ticks) and sorted(ticks) == list(range(len(ticks))),
        "noncontiguous physical ticks",
    )
    _require(
        sorted(requests) == list(range(len(requests))),
        "noncontiguous request IDs",
    )
    _require(
        len(requests) in (len(ticks), len(ticks) + 1),
        "invalid request/execution counts",
    )
    _same(end, {"control_steps": len(ticks)}, "control end")
    _same(
        finish,
        {
            "policy_calls": len(requests),
            "adopted_policy_calls": len(ticks),
            "unused_policy_calls": len(requests) - len(ticks),
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
        groups["one_step_async_execution"], "request_id", "physical execution"
    )
    _require(
        adopted.keys() == ticks.keys() == executed.keys(),
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
        name: _index([p for p in phases if p["phase"] == name], "request_id", name)
        for name in ("inference", "execution", "wait", "drain")
    }
    _require(by_phase["inference"].keys() == requests.keys(), "missing inference phase")
    _require(by_phase["execution"].keys() == ticks.keys(), "missing execution phase")
    _require(set(by_phase["wait"]) <= set(ticks) - {0}, "invalid wait request")
    _require(
        set(by_phase["drain"]) == set(requests) - set(ticks),
        "missing/extra terminal drain",
    )
    work_rows = []
    for rid, request in sorted(requests.items()):
        source = max(0, rid - 1)
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
        accepted = request["t_s"]
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
            prior = ticks[rid - 1]
            _same(capture, prior["sensor_capture_s"], "prefetch snapshot capture")
            _ordered(
                (prior["sensor_capture_s"], capture, submitted, prior["applied_s"]),
                "prefetch before physical step",
            )
            if rid > 1:
                _ordered(
                    (ticks[rid - 2]["next_observation_computed_s"], capture),
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
                "valid_end_tick": rid + 1,
            },
            "DiT completion",
        )
        used = rid in ticks
        _same(
            request,
            {
                "disposition": "adopted" if used else "unused",
                "accepted_prefix_steps": int(used),
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
            {"unused_plan_suffix_slots": len(native) - int(used)},
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
            },
            "inference phase",
        )
        if used:
            tick = ticks[rid]
            applied, completed = (
                tick.get("applied_s"),
                tick.get("next_observation_computed_s"),
            )
            _ordered((accepted, applied, completed, stop), "action application")
            if rid:
                _ordered(
                    (ticks[rid - 1]["next_observation_computed_s"], applied),
                    "physical execution",
                )
            _same(
                adopted[rid],
                {
                    "request_tick": rid,
                    "target_tick": rid,
                    "t_s": accepted,
                    "installed_slots": 1,
                    "expired_slots": 0,
                    "protected_slots": 0,
                    "ready_to_adopt_seconds": accepted - rpc_end,
                },
                "adoption",
            )
            _same(
                tick,
                {
                    **{k: v for k, v in identity.items() if k != "request_tick"},
                    **dependency,
                    "action": [float(max(-1, min(1, v))) for v in native[0][:6]]
                    + [float(native[0][6] >= 0.5)],
                    "fallback": False,
                    "t_s": applied,
                    "command_ready_s": rpc_end,
                    "ready_to_apply_seconds": applied - rpc_end,
                    "first_application_of_request": True,
                    "command_age_ticks": 0,
                    "action_delay_budget_ticks": 0,
                    "action_request_pending": rid + 1 < len(requests),
                    "execution_coupling": "one_step_async",
                    "execution_clock": "ready",
                    "observation_lag_ticks": rid - source,
                    "environment_step_s": completed - applied,
                    "physics_step_seconds": 0.05,
                    "simulated_elapsed_seconds": (rid + 1) * 0.05,
                    "executed_image_age_ms": (applied - capture) * 1000,
                    "executed_state_age_ms": (applied - capture) * 1000,
                },
                "physical tick",
            )
            _same(
                tick.get("executed_slot"),
                {
                    "producer_kind": "request",
                    "producer_request_tick": rid,
                    "tick": rid,
                    "fallback": False,
                },
                "executed producer",
            )
            _same(
                tick.get("action_buffer"),
                {"slots": [], "committed_until": rid + 1},
                "exact prefix",
            )
            _same(
                executed[rid],
                {
                    **identity,
                    "applied_s": applied,
                    "completed_s": completed,
                    "t_s": completed,
                    "executed_steps": 1,
                    "observation_lag_ticks": rid - source,
                },
                "execution record",
            )
            _same(
                by_phase["execution"][rid],
                {
                    **identity,
                    "start_s": applied,
                    "end_s": completed,
                    "t_s": completed,
                    "physics_start_tick": rid,
                    "physics_end_tick": rid + 1,
                    "simulation_paused": False,
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
            }
        )
    for name in ("wait", "drain"):
        for rid, phase in by_phase[name].items():
            request = requests[rid]
            _same(
                phase,
                {
                    **{key: request[key] for key in IDENTITY},
                    "cycle": rid,
                    "simulation_paused": True,
                    "physics_start_tick": rid,
                    "physics_end_tick": rid,
                    "execution_clock": "ready",
                    "t_s": phase.get("end_s"),
                },
                name + " phase",
            )
            a, b = phase.get("start_s"), phase.get("end_s")
            _ordered(
                (
                    ticks[rid - 1]["next_observation_computed_s"],
                    a,
                    request["t_s"],
                    b,
                    work_end,
                ),
                name + " time",
            )
            if name == "drain":
                _same(a, stop, "drain starts at control end")
            else:
                _ordered((b, ticks[rid]["applied_s"]), "wait precedes next action")
    if normalized is not None:
        _same(normalized, work_rows, "normalized full work")
    if metadata:
        for key, value in {
            "policy_calls": len(requests),
            "adopted_policy_calls": len(ticks),
            "unused_policy_calls": len(requests) - len(ticks),
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
