"""Native-prefix scheduling and full-work evidence; no neural inference."""

import copy
import hashlib
import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from coexecution.async_evidence import validate_chunked_async_work
from coexecution.libero.one_step_async import run_chunked_async_loop

GRID = [
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (2, 2),
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 4),
    (8, 0),
    (8, 1),
    (8, 4),
    (8, 8),
]


def scenario(
    *,
    chunk=4,
    depth=2,
    max_steps=10,
    terminal=None,
    done=False,
    faster=False,
    fail_at=None,
    step_failure=None,
):
    main = threading.get_ident()
    ids, captures, observed, executed, closed = [], [], [], [], []
    started = {t: threading.Event() for t in range(0, max_steps, chunk)}
    finished = {t: threading.Event() for t in started}
    stepping = [threading.Event() for _ in range(max_steps)]
    stepped = [threading.Event() for _ in range(max_steps)]
    physical = calls = 0
    obs = {"video.image": np.zeros((2, 2, 3), np.uint8), "state.x": np.zeros(1)}

    def native(target):
        return np.asarray(
            [[target * 0.001 + j * 0.01] * 7 for j in range(16)], np.float32
        )

    class Planner:
        def __init__(self):
            ids.append(threading.get_ident())

        def prepare(self, snapshot):
            ids.append(threading.get_ident())
            return snapshot

        def plan(self, snapshot, capture_s, source_tick):
            nonlocal calls
            ids.append(threading.get_ident())
            target = calls * chunk
            calls += 1
            a = time.monotonic()
            started[target].set()
            if target:
                assert stepping[source_tick].wait(2)
                last = (
                    min(
                        target,
                        terminal or max_steps,
                        (step_failure + 1) if step_failure is not None else max_steps,
                    )
                    - 1
                )
                if not faster:
                    assert stepped[last].wait(2)
                time.sleep(0.004)
            if fail_at == target:
                raise RuntimeError("model failure")
            assert not snapshot["video.image"].flags.writeable
            assert snapshot["state.x"][0] == source_tick
            assert (snapshot["video.image"] == source_tick).all()
            captures.append((target, source_tick))
            b = time.monotonic()
            finished[target].set()
            return {
                "actions": native(target),
                "audit": {
                    "source_tick": source_tick,
                    "capture_s": capture_s,
                    "frame_hashes": {
                        "image": hashlib.sha256(
                            snapshot["video.image"].tobytes()
                        ).hexdigest()
                    },
                    "vlm_started_s": a,
                    "vlm_completed_s": (a + b) / 2,
                    "action_started_s": (a + b) / 2,
                    "action_completed_s": b,
                    "decoded_horizon": 16,
                    "forward_counts": {"vlm": 1, "dit": 4},
                },
            }

        def close(self):
            ids.append(threading.get_ident())
            closed.append(True)

    def observe():
        assert threading.get_ident() == main
        observed.append(physical)
        return obs

    def step(action):
        nonlocal physical
        assert threading.get_ident() == main
        tick = physical
        target, offset = (tick // chunk) * chunk, tick % chunk
        expected = native(target)[offset].copy()
        expected[6] = float(expected[6] >= 0.5)
        np.testing.assert_array_equal(action, expected)
        future = target + chunk
        prefetch = future < max_steps and tick == future - depth
        if prefetch:
            assert started[future].wait(2)
        stepping[tick].set()
        # Mutate exactly the buffers owned by the simulator while worker reads.
        obs["video.image"][:] = tick + 1
        obs["state.x"][:] = tick + 1
        if prefetch and faster:
            assert finished[future].wait(2)
        time.sleep(0.002)
        stepped[tick].set()
        if step_failure == tick:
            raise RuntimeError("environment failure")
        executed.append(action.copy())
        physical += 1
        return (
            obs,
            bool(physical == terminal and not done),
            bool(physical == terminal and done),
        )

    trace = io.StringIO()
    try:
        result, _ = run_chunked_async_loop(
            step,
            observe,
            Planner,
            max_steps=max_steps,
            execution_prefix_steps=chunk,
            async_overlap_steps=depth,
            clock=SimpleNamespace(now=time.monotonic),
            trace=trace,
            request_timeout_seconds=3,
            shutdown_timeout_seconds=3,
        )
    except BaseException as error:
        error.events = [json.loads(row) for row in trace.getvalue().splitlines()]
        error.closed = closed
        raise
    events = [json.loads(row) for row in trace.getvalue().splitlines()]
    evidence = validate_chunked_async_work(events, metadata=result)
    return result, events, evidence, observed, captures, ids, closed, executed


@pytest.mark.parametrize("chunk,depth", [(k, d) for k, d in GRID if d])
@pytest.mark.parametrize("faster", [False, True])
def test_native_prefixes_overlap_and_boundaries(chunk, depth, faster):
    result, events, evidence, observed, captures, ids, closed, executed = scenario(
        chunk=chunk, depth=depth, max_steps=2 * chunk + 1, faster=faster
    )
    assert captures == [(0, 0), (chunk, chunk - depth), (2 * chunk, 2 * chunk - depth)]
    assert observed == [source for _, source in captures]
    assert len(set(ids)) == 1 and ids[0] != threading.get_ident()
    assert closed == [True]
    assert len(executed) == 2 * chunk + 1
    assert result["policy_calls"] == result["adopted_policy_calls"] == 3
    assert (
        result["executed_prefix_steps"]
        == result["accepted_prefix_steps"]
        == 2 * chunk + 1
    )
    assert result["unexecuted_accepted_prefix_steps"] == 0
    assert (
        result["discarded_native_suffix_steps"]
        == result["unused_plan_suffix_slots"]
        == 48 - (2 * chunk + 1)
    )
    assert result["measured_inference_execution_overlap_seconds"] > 0
    assert result["actual_control_hz"] == pytest.approx(
        len(executed) / result["controlled_wall_seconds"]
    )
    assert result["amortized_wall_seconds_per_action"] == pytest.approx(
        1 / result["actual_control_hz"]
    )
    assert result["boundary_gap_seconds"]["count"] == 2
    assert result["adjacent_action_gap_seconds"]["count"] == len(executed) - 1
    assert (
        result["post_step_wait_intervals"] is None
        if faster
        else result["post_step_wait_intervals"]["count"] == 2
    )
    ticks = [r for r in events if r["kind"] == "control_tick"]
    assert [t["executed_offset"] for t in ticks] == list(range(chunk)) * 2 + [0]
    assert [t["observation_lag_ticks"] for t in ticks] == list(range(chunk)) + list(
        range(depth, depth + chunk)
    ) + [depth]
    assert [r["accepted_prefix_steps"] for r in evidence["work_requests"]] == [
        chunk,
        chunk,
        1,
    ]
    for request in evidence["work_requests"][1:]:
        target = request["target_tick"]
        assert (
            request["adopted_s"] + evidence["start_s"]
            >= ticks[target - 1]["next_observation_computed_s"]
        )


@pytest.mark.parametrize(
    "depth,terminal",
    [(1, 1), (1, 7), (1, 8), (4, 3), (4, 5), (4, 7), (8, 1), (8, 9), (8, 11)],
)
@pytest.mark.parametrize("done", [False, True])
def test_terminal_prefix_counts_and_speculative_tail(depth, terminal, done):
    result, _events, evidence, *_ = scenario(
        chunk=8, depth=depth, max_steps=20, terminal=terminal, done=done
    )
    used = (terminal + 7) // 8
    expected_ids = [0] + [target for target in (8, 16) if target - depth < terminal]
    assert [r["request_id"] for r in evidence["work_requests"]] == expected_ids
    assert result["policy_calls"] == len(expected_ids)
    assert result["adopted_policy_calls"] == used
    assert result["unused_policy_calls"] == len(expected_ids) - used
    assert result["accepted_prefix_steps"] == 8 * used
    assert result["executed_prefix_steps"] == terminal
    assert result["unexecuted_accepted_prefix_steps"] == 8 * used - terminal
    assert result["unused_plan_suffix_slots"] == 16 * len(expected_ids) - terminal
    assert result["termination_reason"] == (
        "environment_terminated" if done else "success"
    )
    drains = [p for p in evidence["phases"] if p["phase"] == "drain"]
    assert len(drains) == result["unused_policy_calls"]
    assert all(
        p["physics_start_tick"] == p["physics_end_tick"] == terminal for p in drains
    )


def test_exact_cap_final_partial_prefix():
    result, _, evidence, *_ = scenario(chunk=8, depth=4, max_steps=12)
    last = evidence["work_requests"][-1]
    assert last["target_tick"] == 8 and last["accepted_prefix_steps"] == 4
    assert last["executed_prefix_steps"] == 4
    assert last["discarded_native_suffix_steps"] == 12
    assert last["unexecuted_accepted_prefix_steps"] == 0
    assert result["unused_policy_calls"] == 0


@pytest.mark.parametrize("fail_at,terminal", [(0, None), (4, None), (4, 3)])
def test_model_failure_preserves_partial_trace_and_owned_close(fail_at, terminal):
    with pytest.raises(RuntimeError, match="model failure") as exc:
        scenario(chunk=4, depth=2, fail_at=fail_at, terminal=terminal)
    assert exc.value.closed == [True]
    assert any(e["kind"] == "control_error" for e in exc.value.events)
    assert not any(e["kind"] == "async_work_end" for e in exc.value.events)


def test_simulator_failure_preserves_unused_future_and_actual_prefix():
    with pytest.raises(RuntimeError, match="environment failure") as exc:
        scenario(chunk=4, depth=2, step_failure=2)
    requests = [e for e in exc.value.events if e["kind"] == "chunked_async_request"]
    assert requests[0]["executed_prefix_steps"] == 2
    assert requests[0]["unexecuted_accepted_prefix_steps"] == 2
    assert requests[1]["disposition"] == "unused"
    assert requests[1]["unused_reason"] == "controller_error"
    assert exc.value.closed == [True]


@pytest.mark.parametrize("chunk,depth", GRID)
def test_all_thirteen_controls_and_plan_commands(chunk, depth):
    from coexecution.libero.experiments import build_commands, validate_plan
    from scripts.evaluate_libero_protocol import (
        build_parser,
        validate_deployment_controls,
    )

    coupling = "chunked_async" if depth else "synchronous"
    args = build_parser().parse_args(
        [
            "--output",
            "unused",
            "--protocol",
            "deployment",
            "--variant",
            "flow",
            "--execution-coupling",
            coupling,
            "--execution-clock",
            "ready",
            "--execution-prefix-steps",
            str(chunk),
            "--async-overlap-steps",
            str(depth),
        ]
    )
    validate_deployment_controls(args)
    plan = {
        "runtime_root": "/runtime",
        "model_python": "/model",
        "evaluation_python": "/eval",
        "upstream": "/upstream",
        "libero_config": "/config",
        "gpus": {"action": 0, "vlm": 0, "simulation": 0},
        "ports": {"action": 5570, "vlm": 5572},
        "episodes_per_task": 1,
        "max_steps": 300,
        "seed": 1000,
        "protocols": ["deployment"],
        "execution_clock": "ready",
        "execution_prefix_steps": 8,
        "async_overlap_steps": 8,
        "conditions": [
            {
                "name": "case",
                "checkpoint": "/weights",
                "variant": "flow",
                "flags": [],
                "execution_coupling": coupling,
                "execution_prefix_steps": chunk,
                "async_overlap_steps": depth,
            }
        ],
    }
    validate_plan(plan)
    command = build_commands(plan, plan["conditions"][0], "deployment", Path("/output"))
    assert set(command) == {"action", "evaluate"}
    argv = command["evaluate"]
    assert argv[argv.index("--synchronous-chunk-steps") + 1] == str(chunk)
    assert argv[argv.index("--async-overlap-steps") + 1] == str(depth)


@pytest.mark.parametrize(
    "extra",
    [
        ["--async-overlap-steps", "0"],
        ["--async-overlap-steps", "5"],
        ["--execution-clock", "paced"],
        ["--variant", "pir2"],
        ["--protocol", "algorithm"],
        ["--server-layout", "paired"],
        ["--feature-rpc", "combined"],
        ["--vision-max-hz", "5"],
        ["--deployment-fixed-delay", "1"],
        ["--artificial-compute-wait", "1"],
    ],
)
def test_chunked_controls_reject_incompatible_settings(extra):
    from scripts.evaluate_libero_protocol import (
        build_parser,
        validate_deployment_controls,
    )

    args = build_parser().parse_args(
        [
            "--output",
            "unused",
            "--protocol",
            "deployment",
            "--variant",
            "flow",
            "--execution-coupling",
            "chunked_async",
            "--execution-clock",
            "ready",
            "--execution-prefix-steps",
            "4",
            "--async-overlap-steps",
            "2",
            *extra,
        ]
    )
    with pytest.raises(ValueError):
        validate_deployment_controls(args)


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("chunked_async_request", "source_tick", 999),
        ("chunked_async_request", "executed_prefix_steps", 8),
        ("chunked_async_request", "unexecuted_accepted_prefix_steps", 0),
        ("chunked_async_request", "discarded_native_suffix_steps", 0),
        ("chunked_async_execution", "executed_offset", 7),
        ("cycle_phase", "execution_id", 999),
        ("policy_rpc_completed", "rpc_completed_s", -1),
        ("action_completed", "valid_end_tick", 999),
        ("chunked_async_boundary", "result_ready", True),
    ],
)
def test_corrupted_raw_evidence_rejected(kind, field, value):
    _, events, *_ = scenario(chunk=4, depth=2, terminal=6)
    candidates = [
        e
        for e in events
        if e["kind"] == kind and (kind != "cycle_phase" or e["phase"] == "execution")
    ]
    target = candidates[-1] if kind != "chunked_async_request" else candidates[1]
    target[field] = value
    with pytest.raises(ValueError):
        validate_chunked_async_work(events)


def test_duplicate_phase_missing_tail_and_repeated_offset_rejected():
    _, events, *_ = scenario(chunk=4, depth=4, terminal=5)
    variants = []
    duplicate = copy.deepcopy(events)
    duplicate.append(
        copy.deepcopy(
            next(
                e
                for e in duplicate
                if e["kind"] == "cycle_phase" and e["phase"] == "execution"
            )
        )
    )
    variants.append(duplicate)
    variants.append(
        [
            e
            for e in copy.deepcopy(events)
            if e["kind"] != "chunked_async_request" or e["disposition"] != "unused"
        ]
    )
    repeated = copy.deepcopy(events)
    ticks = [e for e in repeated if e["kind"] == "control_tick"]
    ticks[1]["action"] = ticks[0]["action"]
    variants.append(repeated)
    for changed in variants:
        with pytest.raises(ValueError):
            validate_chunked_async_work(changed)


def test_full_public_report_raw_normalized_integration_and_tampering(tmp_path):
    from coexecution.feature_usage import analyze_episode, audit_raw
    from coexecution.libero.experiments import deployment_report
    from coexecution.queue_report import summarize_episode

    result, events, *_ = scenario(chunk=8, depth=8, terminal=1)
    meta = {**result, "task_id": 0, "episode_id": 0}
    episode = summarize_episode(events, meta)
    episode["id"] = "task0-episode0"
    condition = {
        "layout": "chunked",
        "condition": "deployment",
        "id": "chunked/deployment",
    }
    directory = tmp_path / "chunked" / "deployment"
    directory.mkdir(parents=True)
    (directory / "task0-episode0-trace.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )
    (directory / "episodes.jsonl").write_text(json.dumps(meta) + "\n")
    _, usage = analyze_episode(condition, episode)
    assert usage["read_accounting_complete"] and usage["execution_accounting_complete"]
    audit_raw(condition, episode, tmp_path)
    report = deployment_report(
        {"conditions": [{"name": "chunked"}], "task_ids": [0], "episodes_per_task": 1},
        tmp_path,
    )
    assert (
        report["conditions"][0]["episodes"][0]["work_requests"]
        == episode["work_requests"]
    )
    for key in (
        "source_tick",
        "executed_prefix_steps",
        "discarded_native_suffix_steps",
        "received_s",
    ):
        bad = copy.deepcopy(episode)
        bad["work_requests"][-1][key] = 999
        with pytest.raises(ValueError):
            audit_raw(condition, bad, tmp_path)
    bad = copy.deepcopy(episode)
    bad["ticks"][0]["executed_offset"] = 1
    with pytest.raises(ValueError):
        audit_raw(condition, bad, tmp_path)


def test_metadata_work_measurements_and_protocol_cannot_be_forged():
    result, events, *_ = scenario(chunk=4, depth=2, max_steps=9)
    for key in (
        "actual_control_hz",
        "measured_inference_execution_overlap_seconds",
        "total_policy_rpc_seconds",
        "accepted_prefix_steps",
        "protocol_version",
    ):
        bad = {**result, key: 999}
        with pytest.raises(ValueError):
            validate_chunked_async_work(events, metadata=bad)
