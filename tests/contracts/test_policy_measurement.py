"""Measurement definitions and pairing; no model or GPU computations."""

import importlib
import importlib.util
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def measurement():
    assert importlib.util.find_spec("coexecution.libero.measurement") is not None
    return importlib.import_module("coexecution.libero.measurement")


def test_measurement_import_and_help_do_not_load_models():
    measurement()
    script = (
        "import sys; import coexecution.libero.measurement as m; "
        "assert 'torch' not in sys.modules; assert 'gr00t' not in sys.modules; "
        "m.build_parser().parse_args(['--help'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_new_runtime_measurement_modes_require_explicit_selection():
    module = measurement()
    action = next(
        action for action in module.build_parser()._actions if action.dest == "modes"
    )
    assert "images" in action.choices and "compiled" in action.choices
    assert "images" not in action.default and "compiled" not in action.default


@pytest.mark.parametrize(
    "modes, expected",
    [
        (["graph", "images", "compiled"], "graph"),
        (["compiled", "graph"], "compiled"),
        (["graph", "baseline", "compiled"], "baseline"),
        (["baseline"], "baseline"),
    ],
)
def test_explicit_modes_select_a_requested_reference(modes, expected):
    assert measurement().reference_mode(modes) == expected


@pytest.mark.parametrize("modes", [[], ["unknown"], ["graph", "graph"]])
def test_invalid_reference_selection_fails_before_model_loading(modes):
    with pytest.raises(ValueError):
        measurement().reference_mode(modes)


def test_documented_cli_modes_without_baseline_reach_the_runner(monkeypatch):
    module = measurement()
    received = []
    monkeypatch.setattr(module, "run", received.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measurement",
            "--checkpoint",
            "unused",
            "--dataset",
            "unused",
            "--output",
            "unused",
            "--modes",
            "graph",
            "images",
            "compiled",
        ],
    )
    module.main()
    assert len(received) == 1
    assert received[0].modes == ["graph", "images", "compiled"]


def test_prepared_forward_keeps_the_original_model_body_and_options():
    module = measurement()
    log, features, prediction, options = [], object(), object(), {"image_delay": 2}
    backbone, action = object(), object()

    def vision(inputs):
        log.append(("vision", inputs))
        return features

    def head(encoded, inputs, received_options):
        log.append(("action", encoded, inputs, received_options))
        return prediction

    model = SimpleNamespace(
        backbone=vision, action_head=SimpleNamespace(get_action=head)
    )
    assert module.forward_prepared(model, backbone, action, options) is prediction
    assert log == [("vision", backbone), ("action", features, action, options)]


def test_pair_plan_preserves_shared_inputs_and_seeds_in_each_round():
    module = measurement()
    modes = ["baseline", "prepared", "static", "graph"]
    first = module.round_plan(modes, observations=3, rounds=4, seed=127)
    assert first == module.round_plan(modes, observations=3, rounds=4, seed=127)
    assert len(first) == 4
    for row in first:
        assert sorted(row["modes"]) == sorted(modes)
        assert sorted(row["observations"]) == [0, 1, 2]
        assert len(row["seeds"]) == 3
    assert first != module.round_plan(modes, observations=3, rounds=4, seed=128)
    with pytest.raises(ValueError):
        module.round_plan(["baseline", "baseline"], 3, 4, 127)


def test_runtime_settings_are_precision_independent_and_layered():
    module = measurement()
    empty = module.runtime_settings("baseline")
    assert empty == dict.fromkeys(
        (
            "input_preparation",
            "tokenizer_cache",
            "static_vlm",
            "vlm_graph",
            "image_array_preparation",
            "compiled_host_guards",
        ),
        False,
    )
    assert module.runtime_settings("prepared") == {**empty, "input_preparation": True}
    assert module.runtime_settings("tokenized") == {
        **empty,
        "input_preparation": True,
        "tokenizer_cache": True,
    }
    assert module.runtime_settings("graph") == {
        **empty,
        "input_preparation": True,
        "tokenizer_cache": True,
        "static_vlm": True,
        "vlm_graph": True,
    }
    assert module.runtime_settings("images") == {
        **module.runtime_settings("graph"),
        "image_array_preparation": True,
    }
    assert all(module.runtime_settings("compiled").values())
    with pytest.raises(ValueError):
        module.runtime_settings("unknown")


def test_action_gate_checks_all_finite_keys_shapes_and_exact_values():
    module = measurement()
    reference = {"action": np.array([0.1, -0.2], dtype=np.float32)}
    assert module.action_difference(reference, reference)["passes"]
    changed = {"action": reference["action"].copy()}
    changed["action"][1] += 0.01
    assert not module.action_difference(reference, changed)["passes"]
    assert module.action_difference(reference, changed)["max_abs"] > 0
    assert not module.action_difference(reference, {"action": np.array([np.nan, 0])})[
        "passes"
    ]
    assert not module.action_difference(reference, {"other": reference["action"]})[
        "passes"
    ]
    assert not module.action_difference(reference, {"action": np.zeros(1)})["passes"]


def test_sample_summaries_keep_timing_boundaries_separate():
    module = measurement()
    rows = [
        {"mode": "bf16-baseline", "scope": scope, "wall_ms": value}
        for scope, values in (
            ("policy", (50, 52, 54)),
            ("prepared_model", (30, 32, 34)),
        )
        for value in values
    ]
    result = module.summarize_samples(rows)
    assert result["bf16-baseline"]["policy"]["p50_ms"] == 52
    assert result["bf16-baseline"]["prepared_model"]["p50_ms"] == 32
    assert result["bf16-baseline"]["policy"]["samples"] == 3


def test_steady_graph_audit_accepts_replay_only_and_disabled_paths():
    module = measurement()
    before = {"capture_count": 2, "replay_count": 8, "failure_count": 0}
    after = {**before, "replay_count": 12}
    result = module.graph_interval(before, after, expected_replays=4)
    assert result["passes"] and result["replay_delta"] == 4
    assert result["capture_delta"] == 0
    assert module.graph_interval(None, None, expected_replays=4)["passes"]


@pytest.mark.parametrize(
    "change",
    ({"capture_count": 3}, {"failure_count": 1}, {"replay_count": 11}),
)
def test_steady_graph_audit_rejects_capture_failure_or_eager_bypass(change):
    module = measurement()
    before = {"capture_count": 2, "replay_count": 8, "failure_count": 0}
    after = {**before, "replay_count": 12, **change}
    assert not module.graph_interval(before, after, expected_replays=4)["passes"]


def test_steady_graph_audit_rejects_counter_reset_and_runtime_switch():
    module = measurement()
    before = {"capture_count": 2, "replay_count": 8, "failure_count": 0}
    assert not module.graph_interval(before, None, expected_replays=4)["passes"]
    assert not module.graph_interval(
        before, dict.fromkeys(before, 0), expected_replays=4
    )["passes"]


def test_complete_model_audit_requires_one_vlm_and_four_flow_replays():
    module = measurement()
    before = {
        name: {"capture_count": 2, "replay_count": 10, "failure_count": 0}
        for name in ("dit", "vlm")
    }
    after = {
        name: {**values, "replay_count": values["replay_count"] + count}
        for (name, values), count in zip(before.items(), (4, 1))
    }
    result = module.complete_graph_interval(before, after)
    assert result["passes"] and result["vlm"]["replay_delta"] == 1
    after["vlm"]["replay_count"] -= 1
    assert not module.complete_graph_interval(before, after)["passes"]
    after["vlm"]["replay_count"] += 1
    after["vlm"]["capture_count"] += 1
    assert not module.complete_graph_interval(before, after)["passes"]


def test_graph_snapshots_follow_enabled_runtime_scope():
    module = measurement()
    dit = {"capture_count": 2, "replay_count": 10, "failure_count": 0}
    vlm = {"capture_count": 3, "replay_count": 12, "failure_count": 0}
    session = SimpleNamespace(
        graphs=SimpleNamespace(evidence=lambda: dict(dit)),
        static_vlm=SimpleNamespace(evidence=lambda: {"vlm_graph": dict(vlm)}),
        config=SimpleNamespace(vlm_graph=False),
    )
    assert module.graph_snapshot(session) == {"dit": dit, "vlm": None, "action": None}
    session.config.vlm_graph = True
    assert module.graph_snapshot(session) == {"dit": dit, "vlm": vlm, "action": None}


def test_model_timer_waits_for_completion_and_invokes_the_body_once(monkeypatch):
    module = measurement()
    pending, calls = [], []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            pass

        def elapsed_time(self, other):
            assert not pending, "CUDA elapsed time requires completed events"
            return 1.5

    cuda = SimpleNamespace(Event=Event, synchronize=lambda device: pending.clear())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    value = object()

    def body():
        calls.append(1)
        pending.append("unfinished GPU work")
        return value

    output, wall, detail = module.timed_model(body, "cuda:0")
    assert output is value and calls == [1] and not pending
    assert wall >= detail["host_return_ms"] >= 0
    assert detail["cuda_span_ms"] == 1.5
