"""Real controller contracts with a deterministic clock; no neural model is loaded."""

import io
import json
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest


class Clock:
    def __init__(self):
        self.value = 10.0
        self.sleeps = []

    def now(self):
        return self.value

    def sleep_until(self, target):
        self.sleeps.append(target)
        self.value = max(self.value, target)


def scenario(
    *,
    mode="paced",
    max_steps=10,
    chunk=8,
    success_at=None,
    done_at=None,
    fail=False,
    horizon=16,
):
    from coexecution.libero.synchronous import run_synchronous_loop

    clock = Clock()
    history, captures, plans = [], [], []
    physical_steps = 0

    def observe():
        captures.append(physical_steps)
        return {
            "state.x": np.array([physical_steps]),
            "video.image": np.full((2, 2, 3), physical_steps, np.uint8),
        }

    def plan(observation, capture_s, source_tick):
        assert physical_steps == source_tick == observation["state.x"][0]
        a = clock.now()
        history.append(("inference", physical_steps, a))
        if fail:
            raise RuntimeError("worker failed")
        clock.value += 0.03
        v = clock.now()
        clock.value += 0.07
        b = clock.now()
        result = {
            "actions": np.zeros((horizon, 7), dtype=np.float32),
            "audit": {
                "source_tick": source_tick,
                "capture_s": capture_s,
                "frame_hashes": {"image": str(physical_steps)},
                "vlm_started_s": a,
                "vlm_completed_s": v,
                "action_started_s": v,
                "action_completed_s": b,
                "decoded_horizon": horizon,
                "forward_counts": {"vlm": 1, "dit": 4},
                "graph_execution": {"cold_capture": False},
            },
        }
        plans.append(result)
        return result

    def step(action):
        nonlocal physical_steps
        history.append(("step", physical_steps, clock.now()))
        clock.value += 0.002
        physical_steps += 1
        return (
            {
                "state.x": np.array([physical_steps]),
                "video.image": np.full((2, 2, 3), physical_steps, np.uint8),
            },
            physical_steps == success_at,
            physical_steps == done_at,
        )

    trace = io.StringIO()
    result, _ = run_synchronous_loop(
        step,
        observe,
        plan,
        max_steps=max_steps,
        chunk_steps=chunk,
        execution_clock=mode,
        clock=clock,
        trace=trace,
    )
    return (
        result,
        [json.loads(line) for line in trace.getvalue().splitlines()],
        history,
        captures,
        clock,
        plans,
    )


def test_paced_cycles_are_mutually_exclusive_and_include_final_dwell():
    result, events, history, captures, clock, _plans = scenario()
    phases = [row for row in events if row["kind"] == "cycle_phase"]
    assert [row["phase"] for row in phases] == ["inference", "execution"] * 2
    assert captures == [0, 8]
    assert result["control_steps"] == 10
    assert result["controlled_wall_seconds"] == pytest.approx(0.7)
    assert result["simulated_elapsed_seconds"] == pytest.approx(0.5)
    assert result["actual_control_hz"] == pytest.approx(10 / 0.7)
    assert result["fixed_rate_timing_applicable"] is False
    assert result["control_rate_valid"] is None
    assert result["forward_counts"] == {"vlm": 2, "dit": 8}
    assert all(a["end_s"] <= b["start_s"] for a, b in pairwise(phases))
    for kind, tick, when in history:
        if kind == "step":
            assert all(
                not (p["start_s"] <= when < p["end_s"])
                for p in phases
                if p["phase"] == "inference"
            )
    assert clock.sleeps[-1] == pytest.approx(phases[-1]["end_s"])
    assert result["unused_plan_suffix_slots"] == 22
    ticks = [row for row in events if row["kind"] == "control_tick"]
    assert ticks[-1]["executed_slot"]["producer_request_tick"] == 8
    assert ticks[-1]["feature_source_tick"] == 8
    assert ticks[-1]["simulated_elapsed_seconds"] == pytest.approx(0.5)


def test_ready_executes_without_pacing_and_observes_again_after_chunk():
    result, _events, _history, captures, clock, _ = scenario(mode="ready", max_steps=9)
    assert not clock.sleeps
    assert result["controlled_wall_seconds"] == pytest.approx(0.218)
    assert captures == [0, 8]
    assert result["execution_coupling"] == "synchronous"
    assert result["control_deadline_misses_over5ms"] is None


@pytest.mark.parametrize(
    "success,done,reason", [(3, None, "success"), (None, 5, "environment_terminated")]
)
def test_early_stop_never_executes_remaining_suffix(success, done, reason):
    result, _events, _history, captures, _, _ = scenario(
        success_at=success, done_at=done
    )
    assert result["control_steps"] == (success or done)
    assert result["termination_reason"] == reason
    assert result["success"] is bool(success)
    assert captures == [0]
    assert result["unused_plan_suffix_slots"] == 16 - (success or done)


def test_worker_failure_never_advances_environment():
    with pytest.raises(RuntimeError, match="worker failed"):
        scenario(fail=True)


def test_native_decoded_horizon_must_cover_requested_chunk():
    with pytest.raises(ValueError, match="decoded horizon"):
        scenario(horizon=6)


def test_synchronous_controls_are_rejected_before_model_loading():
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
        "synchronous",
    ]
    args = parser.parse_args(base)
    validate_deployment_controls(args)
    assert args.synchronous_chunk_steps == 8
    for extra in [
        ["--variant", "pir2"],
        ["--protocol", "algorithm"],
        ["--server-layout", "paired"],
        ["--feature-rpc", "combined"],
        ["--vision-max-hz", "5"],
        ["--action-min-service-ms", "10"],
        ["--deployment-fixed-delay", "1"],
        ["--artificial-compute-wait", "0.1"],
        ["--synchronous-chunk-steps", "0"],
    ]:
        with pytest.raises(ValueError):
            validate_deployment_controls(parser.parse_args(base + extra))


def test_backend_synchronous_plan_preserves_rng_and_native_horizon():
    from coexecution.libero.backend import Backend

    backend = object.__new__(Backend)
    backend.variant = "flow"
    backend.role = "action"
    backend.paired = False
    backend.counts = {"vlm": 0, "dit": 0}
    backend.stream = SimpleNamespace(synchronize=lambda: None)
    backend.reset = lambda *_: pytest.fail("Scored plan must not reset the RNG")
    seen = []

    def vision(observation, capture_s, source_tick, return_features):
        assert return_features is False
        backend.counts["vlm"] += 1
        return {
            "meta": {
                "frame_hashes": {"image": "real"},
                "graph_execution": {"cold_capture": False},
            }
        }

    def action(observation, *, options, t_state_capture, period_ms):
        seen.append((options, t_state_capture, period_ms))
        backend.counts["dit"] += 4
        return {
            k: np.zeros((1, 16, 1))
            for k in ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        }, {}

    backend.vision = vision
    backend.core = SimpleNamespace(get_action_chunk_cached=action)
    assert hasattr(backend, "synchronous_plan")
    for tick in (0, 8):
        result = backend.synchronous_plan({"state": {"x": 1}}, 1.0, tick)
        assert result["actions"].shape == (16, 7)
        assert result["audit"]["forward_counts"] == {"vlm": 1, "dit": 4}
        assert result["audit"]["source_tick"] == tick
    assert seen == [({"num_inference_timesteps": 4}, 1.0, 50.0)] * 2


def test_trace_passes_existing_feature_usage_audit_without_invented_bootstrap(tmp_path):
    from coexecution.feature_usage import analyze_episode, audit_raw
    from coexecution.queue_report import summarize_episode

    result, events, *_ = scenario()
    metadata = {**result, "task_id": 0, "episode_id": 0}
    episode = summarize_episode(events, metadata)
    episode["id"] = "task0-episode0"
    condition = {"layout": "sync", "condition": "deployment", "id": "sync/deployment"}
    _rows, usage = analyze_episode(condition, episode)
    assert usage["read_accounting_complete"]
    assert usage["execution_accounting_complete"]
    assert episode["metrics"]["fixed_rate_timing_applicable"] is False
    assert all(row["executed_image_age_ms"] >= 0 for row in episode["ticks"])
    directory = tmp_path / "sync" / "deployment"
    directory.mkdir(parents=True)
    (directory / "task0-episode0-trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    audit_raw(condition, episode, tmp_path)


@pytest.mark.parametrize("change", ["capture", "source", "stage", "counts", "nan"])
def test_invalid_plan_evidence_fails_before_any_action(change):
    from coexecution.libero.synchronous import run_synchronous_loop

    clock = Clock()
    stepped = []

    def plan(_obs, capture_s, source_tick):
        start = clock.now()
        clock.value += 0.1
        audit = {
            "source_tick": source_tick,
            "capture_s": capture_s,
            "frame_hashes": {},
            "decoded_horizon": 8,
            "forward_counts": {"vlm": 1, "dit": 4},
            "vlm_started_s": start,
            "vlm_completed_s": start + 0.03,
            "action_started_s": start + 0.03,
            "action_completed_s": clock.now(),
        }
        actions = np.zeros((8, 7))
        if change == "capture":
            audit["capture_s"] -= 1
        elif change == "source":
            audit["source_tick"] += 1
        elif change == "stage":
            audit["vlm_completed_s"] = clock.now() + 1
        elif change == "counts":
            audit["forward_counts"]["vlm"] = 0
        else:
            actions[0, 0] = np.nan
        return {"actions": actions, "audit": audit}

    trace = io.StringIO()
    with pytest.raises(ValueError):
        run_synchronous_loop(
            stepped.append, dict, plan, max_steps=1, clock=clock, trace=trace
        )
    assert not stepped
    events = [json.loads(row) for row in trace.getvalue().splitlines()]
    assert events[-1]["kind"] == "control_error"
    assert not any(row["kind"] == "control_end" for row in events)


def test_synchronous_identity_rejects_stale_capability_or_small_native_chunk():
    from coexecution.libero.synchronous import validate_synchronous_identity

    identity = {
        "variant": "flow",
        "role": "action",
        "execution_endpoints": ["synchronous_plan"],
        "decoded_horizon": 16,
        "feature_transport": "numpy_fp32",
    }
    validate_synchronous_identity(identity, 8)
    for replacement in [
        {"execution_endpoints": []},
        {"decoded_horizon": 6},
        {"paired_same_process": True},
        {"feature_transport": "numpy_bf16"},
    ]:
        with pytest.raises(ValueError):
            validate_synchronous_identity(identity | replacement, 8)
