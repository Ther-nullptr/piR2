"""Actual host concurrency and accounting contracts; no neural inference."""

import hashlib
import io
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest


def scenario(
    *,
    max_steps=3,
    success_at=None,
    done_at=None,
    faster=False,
    fail_at=None,
    step_failure=False,
):
    from coexecution.libero.one_step_async import run_one_step_async_loop

    main = threading.get_ident()
    ids, captures, actions, closed = [], [], [], []
    started = [threading.Event() for _ in range(max_steps + 1)]
    stepping = [threading.Event() for _ in range(max_steps)]
    stepped = [threading.Event() for _ in range(max_steps)]
    finished = [threading.Event() for _ in range(max_steps + 1)]
    physical = 0
    calls = 0
    # Deliberately alias environment storage to detect missing submission copies.
    observation = {"video.image": np.zeros((2, 2, 3), np.uint8), "state.x": np.zeros(1)}

    class Planner:
        def __init__(self):
            ids.append(threading.get_ident())

        def prepare(self, obs):
            ids.append(threading.get_ident())
            return obs

        def plan(self, obs, capture_s, source_tick):
            nonlocal calls
            ids.append(threading.get_ident())
            request = calls
            calls += 1
            a = time.monotonic()
            started[request].set()
            if request:
                assert stepping[request - 1].wait(2), "inference did not overlap step"
                if not faster:
                    assert stepped[request - 1].wait(2)
                time.sleep(0.005)
            if fail_at == request:
                raise RuntimeError("model request failed")
            assert not obs["video.image"].flags.writeable
            assert obs["state.x"][0] == source_tick
            assert (obs["video.image"] == source_tick).all()
            captures.append((request, source_tick))
            b = time.monotonic()
            finished[request].set()
            return {
                "actions": np.full((16, 7), 0.1 * request, np.float32),
                "audit": {
                    "source_tick": source_tick,
                    "capture_s": capture_s,
                    "frame_hashes": {
                        "image": hashlib.sha256(
                            obs["video.image"].tobytes()
                        ).hexdigest()
                    },
                    "vlm_started_s": a,
                    "vlm_completed_s": (a + b) / 2,
                    "action_started_s": (a + b) / 2,
                    "action_completed_s": b,
                    "decoded_horizon": 16,
                    "forward_counts": {"vlm": 1, "dit": 4},
                    "graph_execution": {"cold_capture": False},
                },
            }

        def close(self):
            ids.append(threading.get_ident())
            closed.append(True)

    def observe():
        assert threading.get_ident() == main
        return observation

    def step(action):
        nonlocal physical
        assert threading.get_ident() == main
        tick = physical
        actions.append(action.copy())
        if tick + 1 < max_steps:
            assert started[tick + 1].wait(2)
        stepping[tick].set()
        observation["video.image"][:] = tick + 1
        observation["state.x"][:] = tick + 1
        if faster and tick + 1 < max_steps:
            assert finished[tick + 1].wait(2)
        time.sleep(0.005)
        physical += 1
        stepped[tick].set()
        if step_failure:
            raise RuntimeError("environment step failed")
        return observation, physical == success_at, physical == done_at

    trace = io.StringIO()
    try:
        result, frames = run_one_step_async_loop(
            step,
            observe,
            Planner,
            max_steps=max_steps,
            clock=SimpleNamespace(now=time.monotonic),
            trace=trace,
            request_timeout_seconds=3,
            shutdown_timeout_seconds=3,
        )
    except BaseException as error:
        error.trace = [json.loads(row) for row in trace.getvalue().splitlines()]
        error.closed = closed
        error.actions = actions
        raise
    return (
        result,
        [json.loads(row) for row in trace.getvalue().splitlines()],
        actions,
        captures,
        ids,
        closed,
        frames,
    )


@pytest.mark.parametrize("faster", [False, True])
def test_real_overlap_exact_prefix_unique_source_and_frozen_snapshot(faster):
    result, events, actions, captures, ids, closed, _ = scenario(faster=faster)
    assert len(set(ids)) == 1 and ids[0] != threading.get_ident()
    assert closed == [True]
    assert captures == [(0, 0), (1, 0), (2, 1)]
    assert len(actions) == result["control_steps"] == 3
    assert result["policy_calls"] == result["adopted_policy_calls"] == 3
    assert result["forward_counts"] == {"vlm": 3, "dit": 12}
    assert result["unused_policy_calls"] == 0
    assert result["unused_plan_suffix_slots"] == 45
    assert result["simulated_elapsed_seconds"] == pytest.approx(0.15)
    assert result["measured_inference_execution_overlap_seconds"] > 0
    requests = [r for r in events if r["kind"] == "one_step_async_request"]
    assert [r["request_id"] for r in requests] == [0, 1, 2]
    assert [r["source_tick"] for r in requests] == [0, 0, 1]
    assert [r["target_tick"] for r in requests] == [0, 1, 2]
    ticks = [r for r in events if r["kind"] == "control_tick"]
    assert [r["observation_lag_ticks"] for r in ticks] == [0, 1, 1]
    assert [r["executed_slot"]["producer_request_tick"] for r in ticks] == [0, 1, 2]
    assert all(r["executed_image_age_ms"] >= 0 for r in ticks)
    assert all(r["action_buffer"]["slots"] == [] for r in ticks)
    phases = [r for r in events if r["kind"] == "cycle_phase"]
    assert len([r for r in phases if r["phase"] == "execution"]) == 3
    inference = [r for r in phases if r["phase"] == "inference"]
    assert [r["simulation_paused"] for r in inference] == [True, False, False]
    waits = [r for r in phases if r["phase"] == "wait"]
    assert len(waits) == (0 if faster else 2)


@pytest.mark.parametrize(
    "success,done,reason", [(1, None, "success"), (None, 1, "environment_terminated")]
)
def test_terminal_drain_accounts_unused_work_without_extra_actions(
    success, done, reason
):
    result, events, actions, captures, _, closed, _ = scenario(
        success_at=success, done_at=done
    )
    assert len(actions) == 1
    assert captures == [(0, 0), (1, 0)]
    assert closed == [True]
    assert result["termination_reason"] == reason
    assert result["success"] is bool(success)
    assert result["policy_calls"] == 2 and result["adopted_policy_calls"] == 1
    assert result["unused_policy_calls"] == 1
    assert result["unused_plan_suffix_slots"] == 31
    assert result["forward_counts"] == {"vlm": 2, "dit": 8}
    assert result["terminal_drain_seconds"] > 0
    assert result["scored_work_end_s"] > result["controlled_end_s"]
    assert result["scored_work_wall_seconds"] == pytest.approx(
        result["controlled_wall_seconds"] + result["terminal_drain_seconds"]
    )
    assert len([r for r in events if r["kind"] == "unused_result_at_episode_end"]) == 1
    assert len([r for r in events if r["kind"] == "action_adopted"]) == 1
    assert events[-1]["kind"] == "async_work_end"


def test_max_step_one_avoids_predictably_unused_call():
    result, _, actions, captures, _, closed, _ = scenario(max_steps=1)
    assert len(actions) == 1 and captures == [(0, 0)]
    assert result["policy_calls"] == 1
    assert result["unused_policy_calls"] == 0
    assert closed == [True]


@pytest.mark.parametrize("fail_at", [0, 1])
def test_rpc_failure_never_becomes_success_and_worker_closes(fail_at):
    with pytest.raises(RuntimeError, match="model request failed") as exc:
        scenario(fail_at=fail_at)
    error = exc.value
    assert error.closed == [True]
    assert len(error.actions) == fail_at
    assert any(r["kind"] == "control_error" for r in error.trace)
    assert not any(r["kind"] == "async_work_end" for r in error.trace)


def test_terminal_rpc_failure_preserves_failure_after_successful_env_step():
    with pytest.raises(RuntimeError, match="model request failed") as exc:
        scenario(fail_at=1, success_at=1)
    assert len(exc.value.actions) == 1
    assert any(r["kind"] == "control_error" for r in exc.value.trace)
    assert not any(r["kind"] == "async_work_end" for r in exc.value.trace)


def test_controls_and_prefix_alias_reject_incompatible_modes():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from evaluate_libero_protocol import build_parser, validate_deployment_controls

    parser = build_parser()
    base = [
        "--protocol",
        "deployment",
        "--variant",
        "flow",
        "--output",
        "unused",
        "--execution-coupling",
        "one_step_async",
        "--execution-clock",
        "ready",
        "--execution-prefix-steps",
        "1",
    ]
    args = parser.parse_args(base)
    validate_deployment_controls(args)
    assert args.synchronous_chunk_steps == 1
    for extra in [
        ["--execution-clock", "paced"],
        ["--synchronous-chunk-steps", "8"],
        ["--variant", "pir2"],
        ["--protocol", "algorithm"],
        ["--server-layout", "paired"],
        ["--feature-rpc", "combined"],
        ["--vision-max-hz", "5"],
        ["--action-min-service-ms", "10"],
        ["--deployment-fixed-delay", "1"],
        ["--artificial-compute-wait", "0.1"],
    ]:
        with pytest.raises(ValueError):
            validate_deployment_controls(parser.parse_args(base + extra))


@pytest.mark.parametrize("terminal", [False, True])
def test_queue_report_includes_full_work_and_passes_raw_lineage(tmp_path, terminal):
    from coexecution.feature_usage import analyze_episode, audit_raw
    from coexecution.queue_report import summarize_episode

    result, events, *_ = scenario(success_at=1 if terminal else None)
    episode = summarize_episode(events, {**result, "task_id": 0, "episode_id": 0})
    episode["id"] = "task0-episode0"
    condition = {
        "layout": "one-step",
        "condition": "deployment",
        "id": "one-step/deployment",
    }
    _, usage = analyze_episode(condition, episode)
    assert usage["read_accounting_complete"]
    assert usage["execution_accounting_complete"]
    assert all(
        t["executed_image_age_ms"] >= 0 and t["executed_state_age_ms"] >= 0
        for t in episode["ticks"]
    )
    assert episode["metrics"]["action_rpc_ms"]["count"] == 0
    assert episode["metrics"]["action_stage_ms"]["count"] > 0
    assert len(episode["work_requests"]) == result["policy_calls"]
    assert (
        len({r["request_id"] for r in episode["work_requests"]})
        == result["policy_calls"]
    )
    assert episode["work_end_s"] == result["scored_work_end_s"]
    assert (
        sum(r["disposition"] == "unused" for r in episode["work_requests"])
        == result["unused_policy_calls"]
    )
    directory = tmp_path / "one-step" / "deployment"
    directory.mkdir(parents=True)
    (directory / "task0-episode0-trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    audit_raw(condition, episode, tmp_path)
    (directory / "episodes.jsonl").write_text(
        json.dumps({**result, "task_id": 0, "episode_id": 0}) + "\n"
    )
    from coexecution.libero.experiments import deployment_report

    report = deployment_report(
        {
            "conditions": [{"name": "one-step"}],
            "task_ids": [0],
            "episodes_per_task": 1,
        },
        tmp_path,
    )
    assert (
        report["conditions"][0]["episodes"][0]["work_requests"]
        == episode["work_requests"]
    )


def test_environment_error_drains_pending_call_but_never_claims_execution():
    with pytest.raises(RuntimeError, match="environment step failed") as exc:
        scenario(step_failure=True)
    error = exc.value
    assert error.closed == [True]
    assert not any(
        r["kind"] in ("control_end", "async_work_end", "one_step_async_execution")
        for r in error.trace
    )
    unused = [r for r in error.trace if r["kind"] == "unused_result_at_episode_end"]
    assert len(unused) == 1 and unused[0]["unused_reason"] == "controller_error"


def test_worker_timeout_and_shutdown_are_bounded_and_owned():
    from coexecution.libero.one_step_async import OwnedPlanWorker

    release, running, closed = threading.Event(), threading.Event(), threading.Event()
    resource = SimpleNamespace(close=closed.set)
    worker = OwnedPlanWorker(lambda: resource, shutdown_timeout=0.02)

    def blocked(_resource):
        running.set()
        release.wait(2)

    future = worker.submit(blocked)
    assert running.wait(1)
    try:
        with pytest.raises(TimeoutError):
            future.result(timeout=0.01)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="did not stop"):
            worker.close()
        assert time.monotonic() - started < 0.5
    finally:
        release.set()
        worker.thread.join(timeout=1)
    assert closed.is_set() and not worker.thread.is_alive()


def test_measured_overlap_is_actual_rpc_and_stage_intersection():
    result, events, *_ = scenario()
    rpc = [r for r in events if r["kind"] == "one_step_async_request"]
    execution = [
        r for r in events if r["kind"] == "cycle_phase" and r["phase"] == "execution"
    ]
    overlap = sum(
        max(
            0,
            min(r["rpc_completed_s"], e["end_s"])
            - max(r["rpc_started_s"], e["start_s"]),
        )
        for r in rpc
        for e in execution
    )
    assert result["measured_inference_execution_overlap_seconds"] == pytest.approx(
        overlap
    )
    assert overlap < result["simulated_elapsed_seconds"]
    for stage, measured in result["measured_server_execution_overlap_seconds"].items():
        intersection = sum(
            max(
                0,
                min(r["audit"][f"{stage}_completed_s"], e["end_s"])
                - max(r["audit"][f"{stage}_started_s"], e["start_s"]),
            )
            for r in rpc
            for e in execution
        )
        assert measured == pytest.approx(intersection)


def test_episode_warmup_and_reset_use_only_owned_worker(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "zmq", SimpleNamespace(RCVTIMEO=1, SNDTIMEO=2))
    import coexecution.libero.one_step_async as module
    from scripts import evaluate_libero_protocol as evaluator

    obs = {"video.image": np.zeros((2, 2, 3), np.uint8), "state.x": np.zeros(1)}
    records = []

    class Socket:
        def __init__(self):
            self.options = {}

        def getsockopt(self, option):
            return self.options.get(option, 120000)

        def setsockopt(self, option, value):
            self.options[option] = value

    class Planner:
        def __init__(self, _port):
            self.client = SimpleNamespace(
                timeout_ms=120000, socket=Socket(), call_endpoint=self.reset
            )
            records.append(("create", threading.get_ident()))

        def reset(self, endpoint, data):
            assert endpoint == "reset" and data == {"options": {"seed": 4321}}
            records.append(("reset", threading.get_ident()))

        def prepare(self, observation):
            return observation

        def plan(self, observation, capture_s, source_tick):
            records.append((self.client.timeout_ms, threading.get_ident()))
            a = time.monotonic()
            return {
                "actions": np.zeros((16, 7)),
                "audit": {
                    "source_tick": source_tick,
                    "capture_s": capture_s,
                    "frame_hashes": {
                        "image": hashlib.sha256(
                            observation["video.image"].tobytes()
                        ).hexdigest()
                    },
                    "vlm_started_s": a,
                    "vlm_completed_s": a,
                    "action_started_s": a,
                    "action_completed_s": a,
                    "decoded_horizon": 16,
                    "forward_counts": {"vlm": 1, "dit": 4},
                },
            }

        def close(self):
            records.append(("close", threading.get_ident()))

    monkeypatch.setattr(module, "_PolicyPlanner", Planner)
    monkeypatch.setattr(
        evaluator,
        "checked_env_step",
        lambda env, action: (obs, 0, False, False, {"success": False}),
    )
    env = SimpleNamespace(
        _process_observation=lambda x: x,
        _env=SimpleNamespace(env=SimpleNamespace(_get_observations=lambda **kw: obs)),
    )
    args = SimpleNamespace(
        record_replay=False,
        record_video=False,
        synchronous_warmup_timeout_seconds=900,
        slow_warmup_calls=3,
        port=5570,
        max_steps=1,
    )
    result, _ = module.one_step_async_episode(env, obs, args, 4321, io.StringIO())
    assert [r[0] for r in records] == [
        "create",
        900000,
        900000,
        900000,
        "reset",
        120000,
        "close",
    ]
    assert len({r[1] for r in records}) == 1 and records[0][1] != threading.get_ident()
    assert len(result["warmup"]["calls"]) == 3
    assert result["policy_calls"] == 1
