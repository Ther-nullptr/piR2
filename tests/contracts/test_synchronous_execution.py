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


def synchronous_recording(tmp_path):
    result, events, *_ = scenario()
    root = tmp_path / "bf16-sync" / "deployment"
    root.mkdir(parents=True)
    config = {
        "protocol": "deployment",
        "execution_coupling": "synchronous",
        "execution_clock": "paced",
        "checkpoint": {"inference_optimization": {"config": {"precision": "bf16"}}},
        "task_ids": [0],
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "episodes.jsonl").write_text(
        json.dumps({**result, "task_id": 0, "episode_id": 0}) + "\n"
    )
    (root / "task0-episode0-trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    return root, events


def test_replay_preserves_coupling_real_phases_and_lineage(tmp_path):
    from coexecution.replay.catalog import build_catalog

    root, _ = synchronous_recording(tmp_path)
    output = tmp_path / "catalog"
    catalog = build_catalog(
        model_runs=[],
        deployments=[root],
        output=output,
        default_execution_coupling="synchronous",
    )
    assert catalog["default_execution_coupling"] == "synchronous"
    assert catalog["runs"][0]["execution_coupling"] == "synchronous"
    run = json.loads((output / catalog["runs"][0]["path"]).read_text())
    assert run["execution_coupling"] == "synchronous"
    episode = run["episodes"][0]
    assert len(episode["phases"]) == 4
    assert episode["phases"][0]["start_s"] == 0
    assert episode["phases"][-1]["end_s"] == pytest.approx(0.7)
    assert episode["phases"][0]["simulation_paused"] is True
    assert episode["cycles"][1]["unused_plan_suffix_slots"] == 14
    assert episode["ticks"][-1]["request_id"].endswith("/S1-8")
    assert episode["queue"]["requests"][0]["timing_scope"] == "server_action_stage"


@pytest.mark.parametrize(
    "fault", ["overlap", "wrong_physics", "missing_phase", "config_disagreement"]
)
def test_replay_rejects_false_synchronous_phase_evidence(tmp_path, fault):
    from coexecution.replay.catalog import build_catalog

    root, events = synchronous_recording(tmp_path)
    phases = [row for row in events if row["kind"] == "cycle_phase"]
    if fault == "overlap":
        phases[1]["start_s"] = phases[0]["start_s"]
    elif fault == "wrong_physics":
        phases[0]["physics_end_tick"] = 1
    elif fault == "missing_phase":
        events.remove(phases[1])
    else:
        events[0]["execution_coupling"] = "asynchronous"
    (root / "task0-episode0-trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    with pytest.raises(ValueError, match="coupling|phase|physics"):
        build_catalog(model_runs=[], deployments=[root], output=tmp_path / "catalog")


def test_synchronous_report_does_not_call_server_stages_independent_rpcs(tmp_path):
    from coexecution.queue_report import condition_report

    root, _ = synchronous_recording(tmp_path)
    report = condition_report(root)
    metrics = report["episodes"][0]["metrics"]
    assert metrics["action_rpc_ms"]["count"] == 0
    assert metrics["vision_rpc_ms"]["count"] == 0
    assert metrics["action_stage_ms"]["p50"] == pytest.approx(70)
    assert metrics["vision_stage_ms"]["p50"] == pytest.approx(30)
    assert report["summary"]["s1_rpc_p50_ms"] is None
    assert report["summary"]["s1_stage_p50_ms"] == pytest.approx(70)
    assert report["summary"]["control_rate_valid_episodes"] is None


def _replace_trace(root, events):
    (root / "task0-episode0-trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("vision_request", "started_s", 10.2),
        ("vision_request", "completed_s", 10.23),
        ("vision_request", "source_tick", 5),
        ("vision_request", "capture_s", 9.0),
        ("feature_published", "feature_sequence", 7),
        ("feature_published", "rpc_completed_s", 10.24),
        ("action_submitted", "t_s", 10.2),
        ("action_submitted", "feature_source_tick", 5),
        ("action_started", "t_s", 10.2),
        ("action_completed", "t_s", 10.3),
        ("action_completed", "feature_capture_s", 9.0),
        ("action_completed", "valid_end_tick", 99),
        ("action_adopted", "installed_slots", 99),
        ("action_adopted", "t_s", 10.2),
    ],
)
def test_replay_rejects_stage_events_that_disagree_with_cycle_audit(
    tmp_path, kind, field, value
):
    from coexecution.replay.catalog import build_catalog

    root, events = synchronous_recording(tmp_path)
    row = next(row for row in events if row["kind"] == kind)
    row[field] = value
    _replace_trace(root, events)
    with pytest.raises(ValueError, match="Synchronous.*stage"):
        build_catalog(model_runs=[], deployments=[root], output=tmp_path / "catalog")


@pytest.mark.parametrize(
    "kind",
    [
        "vision_request",
        "feature_published",
        "action_submitted",
        "action_started",
        "action_completed",
        "request_completed",
        "action_adopted",
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_replay_requires_exactly_one_stage_event_per_cycle(tmp_path, kind, mutation):
    from coexecution.replay.catalog import build_catalog

    root, events = synchronous_recording(tmp_path)
    row = next(row for row in events if row["kind"] == kind)
    if mutation == "missing":
        events.remove(row)
    else:
        events.append(dict(row))
    _replace_trace(root, events)
    with pytest.raises(ValueError, match="Synchronous.*stage"):
        build_catalog(model_runs=[], deployments=[root], output=tmp_path / "catalog")


def test_replay_rejects_contradictory_stage_audit_and_executed_source(tmp_path):
    from coexecution.replay.catalog import build_catalog

    root, events = synchronous_recording(tmp_path)
    request = next(row for row in events if row["kind"] == "request_completed")
    request["audit"]["forward_counts"]["dit"] = 8
    _replace_trace(root, events)
    with pytest.raises(ValueError, match="Synchronous.*stage"):
        build_catalog(model_runs=[], deployments=[root], output=tmp_path / "catalog")


def test_full_synchronous_rpc_does_not_reuse_independent_action_rpc_start():
    _result, events, *_ = scenario()
    timing = next(
        row["client_timing"] for row in events if row["kind"] == "request_completed"
    )
    assert "plan_rpc_started_s" not in timing
    assert timing["timing_scope"] == "server_action_stage"
    assert timing["synchronous_rpc_started_s"] < timing["submitted_s"]
    assert timing["synchronous_rpc_completed_s"] >= timing["completed_s"]


def test_replay_normalizes_legacy_full_rpc_name_without_rewriting_source(tmp_path):
    from coexecution.replay.catalog import build_catalog

    root, events = synchronous_recording(tmp_path)
    for row in events:
        timing = row if row["kind"] == "action_completed" else row.get("client_timing")
        if timing is not None:
            timing["plan_rpc_started_s"] = timing.pop(
                "synchronous_rpc_started_s",
                10.0
                if row.get("request_tick", row.get("audit", {}).get("request_tick"))
                == 0
                else 10.5,
            )
            timing.pop("synchronous_rpc_completed_s", None)
            timing.pop("timing_scope", None)
    _replace_trace(root, events)
    path = root / "task0-episode0-trace.jsonl"
    raw = path.read_bytes()
    output = tmp_path / "catalog"
    catalog = build_catalog(model_runs=[], deployments=[root], output=output)
    run = json.loads((output / catalog["runs"][0]["path"]).read_text())
    request = run["episodes"][0]["queue"]["requests"][0]
    assert request["plan_rpc_started_s"] is None
    assert request["synchronous_rpc_started_s"] == 0
    assert request["synchronous_rpc_completed_s"] == pytest.approx(0.1)
    assert request["timing_scope"] == "server_action_stage"
    assert path.read_bytes() == raw


def test_replay_rejects_review_reproduction_of_vlm_inside_execution(tmp_path):
    from coexecution.replay.catalog import build_catalog

    root, events = synchronous_recording(tmp_path)
    vision = next(row for row in events if row["kind"] == "vision_request")
    vision.update(started_s=10.2, completed_s=10.23, t_s=10.23, rpc_seconds=0.03)
    _replace_trace(root, events)
    with pytest.raises(ValueError, match="Synchronous.*stage"):
        build_catalog(model_runs=[], deployments=[root], output=tmp_path / "catalog")


@pytest.mark.parametrize(
    "failure,replace_socket", [(False, False), (True, False), (True, True)]
)
def test_warmup_rpc_timeout_restores_client_and_current_socket(
    monkeypatch, failure, replace_socket
):
    import sys

    from coexecution.libero import synchronous

    zmq = SimpleNamespace(RCVTIMEO=27, SNDTIMEO=28)
    monkeypatch.setitem(sys.modules, "zmq", zmq)

    class Socket:
        def __init__(self, receive=120000, send=130000):
            self.options = {27: receive, 28: send}

        def getsockopt(self, option):
            return self.options[option]

        def setsockopt(self, option, value):
            self.options[option] = value

    original = Socket()
    client = SimpleNamespace(timeout_ms=120000, socket=original)
    assert hasattr(synchronous, "warmup_rpc_timeout")
    try:
        with synchronous.warmup_rpc_timeout(client, 900000):
            assert client.timeout_ms == 900000
            assert client.socket.options == {27: 900000, 28: 900000}
            if replace_socket:
                # PolicyClient recreates the invalid REQ socket using timeout_ms.
                client.socket = Socket(client.timeout_ms, client.timeout_ms)
            if failure:
                raise TimeoutError("warmup failed")
    except TimeoutError:
        assert failure
    assert client.timeout_ms == 120000
    assert client.socket.options == {27: 120000, 28: 130000}
    assert (client.socket is original) is not replace_socket


def test_warmup_timeout_control_is_bounded_and_only_sync_uses_it():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from evaluate_libero_protocol import build_parser, validate_deployment_controls

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
    args = build_parser().parse_args(base)
    assert args.synchronous_warmup_timeout_seconds == 900
    for value in ("0", "-1", "2147484"):
        with pytest.raises(ValueError, match="warmup.*timeout"):
            validate_deployment_controls(
                build_parser().parse_args(
                    base + ["--synchronous-warmup-timeout-seconds", value]
                )
            )
