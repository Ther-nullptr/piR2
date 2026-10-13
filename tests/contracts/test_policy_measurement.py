"""Measurement definitions and pairing; no model or GPU computations."""

import importlib
import importlib.util
import json
import subprocess
import sys
from contextlib import nullcontext
from io import StringIO
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

    output, wall, detail = module.timed_model(body, "cuda:0", cuda_events=True)
    assert output is value and calls == [1] and not pending
    assert wall >= detail["host_return_ms"] >= 0
    assert detail["cuda_span_ms"] == 1.5


def test_model_only_and_cuda_instrumentation_are_explicit_cli_options():
    parser = measurement().build_parser()
    base = ["--checkpoint", "unused", "--dataset", "unused", "--output", "unused"]
    defaults = parser.parse_args(base)
    assert defaults.measurement_boundary == "comparison"
    assert defaults.cuda_events is False
    args = parser.parse_args(
        [
            *base,
            "--measurement-boundary",
            "model-only",
            "--cuda-events",
            "--max-prepared-observations",
            "8",
        ]
    )
    assert args.measurement_boundary == "model-only"
    assert args.cuda_events is True and args.max_prepared_observations == 8


def test_model_only_oversized_cohort_fails_before_loading(monkeypatch):
    module = measurement()
    monkeypatch.setattr(
        module, "run", lambda args: pytest.fail("Model loading attempted")
    )
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
            "--measurement-boundary",
            "model-only",
            "--observations",
            "3",
            "--max-prepared-observations",
            "2",
        ],
    )
    with pytest.raises(SystemExit) as raised:
        module.main()
    assert raised.value.code == 2


def test_clean_model_timer_does_not_create_cuda_events(monkeypatch):
    module = measurement()
    calls = []
    cuda = SimpleNamespace(synchronize=lambda device: calls.append(("wait", device)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    value = object()

    def body():
        calls.append(("model",))
        return value

    output, wall, detail = module.timed_model(body, "cuda:0")
    assert output is value and wall >= detail["host_return_ms"] >= 0
    assert "cuda_span_ms" not in detail
    assert calls == [("wait", "cuda:0"), ("model",), ("wait", "cuda:0")]


@pytest.mark.parametrize("failure", [None, "output", "rng", "graph"])
def test_model_only_prepares_once_and_preserves_prior_evidence_on_failure(
    monkeypatch, tmp_path, failure
):
    module = measurement()
    assert callable(getattr(module, "_model_only_mode", None)), (
        "Independent runner missing"
    )
    log, state = [], {"rng": 0}
    engine_outputs = []
    raw_storage = np.zeros(1)

    def collate(observation):
        log.append("prepare")
        return {"inputs": observation}, {"offset": 10}

    def backbone(value):
        log.append("vlm")
        return value

    def head(features, action, options):
        assert options is None
        log.append("action")
        state["rng"] += 1
        result = {"action": np.array([features["pixel"] + action["state"]])}
        engine_outputs.append(result)
        return result

    def raw(observation):
        log.append("raw")
        state["rng"] += 1
        raw_storage[0] = observation["pixel"] + observation["state"] + 10
        return {"action": raw_storage}, {}

    def decode(result, decoder_state):
        log.append("decode")
        return {"action": result["action"] + decoder_state["offset"]}

    def reset(policy, seed, device):
        state["rng"] = seed

    def timer(body, device, *, cuda_events):
        assert cuda_events is False
        log.append("timer_start")
        result = body()
        log.append("timer_end")
        if failure == "output":
            result["action"] += 1
        elif failure == "rng":
            state["rng"] += 1
        return result, 2.0, {"host_return_ms": 1.0}

    fake_torch = SimpleNamespace(
        get_rng_state=lambda: state["rng"],
        cuda=SimpleNamespace(get_rng_state=lambda device: state["rng"]),
        equal=lambda a, b: a == b,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(module, "_seed_reset", reset)
    monkeypatch.setattr(module, "timed_model", timer)
    if failure == "graph":
        monkeypatch.setattr(
            module, "complete_graph_interval", lambda *args: {"passes": False}
        )
    policy = SimpleNamespace(
        _collate_observation=collate,
        _decode_action=decode,
        get_action=raw,
        model=SimpleNamespace(
            backbone=backbone,
            action_head=SimpleNamespace(get_action=head),
            prepare_input=lambda data: (data, data),
        ),
    )
    session = SimpleNamespace(
        graphs=None, action_graph=None, config=SimpleNamespace(vlm_graph=False)
    )
    args = SimpleNamespace(
        max_prepared_observations=2,
        warmup=1,
        device="cuda:0",
        cuda_events=False,
        output=tmp_path,
    )
    plan = module.round_plan(["graph"], 2, 2, 127)
    stream = StringIO()
    previous = {
        "gates": [
            {"mode": precision, "prepared_model": {"passes": True}}
            for precision in ("bf16", "w8a8")
        ],
        "graph_audits": [
            {"mode": precision, "prepared_model": {"passes": True}}
            for precision in ("bf16", "w8a8")
        ],
        "retained": {precision: np.array([1.0]) for precision in ("bf16", "w8a8")},
    }

    def replay():
        return module._model_only_mode(
            policy,
            session,
            args,
            [{"pixel": 1, "state": 2}, {"pixel": 4, "state": 5}],
            ["obs-a", "obs-b"],
            plan,
            "w4a4-graph",
            stream,
            previous=previous,
        )

    if failure:
        with pytest.raises(RuntimeError):
            replay()
        gates = json.loads((tmp_path / "validation.json").read_text())
        audits = json.loads((tmp_path / "graph-timing-audit.json").read_text())
        assert gates[:2] == previous["gates"]
        assert audits[:2] == previous["graph_audits"]
        assert audits[-1]["mode"] == "w4a4-graph"
        if failure == "graph":
            assert audits[-1]["prepared_model"]["passes"] is False
        else:
            assert gates[-1]["mode"] == "w4a4-graph"
            assert (
                not gates[-1]["rng_equal"] or not gates[-1]["prepared_model"]["passes"]
            )
        with np.load(tmp_path / "actions.npz") as actions:
            for precision, value in previous["retained"].items():
                np.testing.assert_array_equal(actions[precision], value)
            if failure != "graph":
                assert any(
                    "w4a4-graph/o1/policy/action" in key for key in actions.files
                )
                assert any(
                    "w4a4-graph/o1/prepared_model/action" in key
                    for key in actions.files
                )
        return
    result = replay()
    assert log.count("prepare") == 2
    first_timed = log.index("timer_start")
    assert "raw" not in log[first_timed:] and "prepare" not in log[first_timed:]
    for position, value in enumerate(log):
        if value == "timer_start":
            assert log[position : position + 4] == [
                "timer_start",
                "vlm",
                "action",
                "timer_end",
            ]
    assert len(result["samples"]) == 4
    assert len({row["request_id"] for row in result["samples"]}) == 4
    assert {row["observation_id"] for row in result["samples"]} == {"obs-a", "obs-b"}
    assert {row["scope"] for row in result["samples"]} == {"prepared_model"}
    assert all(
        row["prepared_model"]["passes"] and row["rng_equal"] for row in result["gates"]
    )
    assert len(engine_outputs) == 6  # Two warmups plus four fresh complete calls.


def test_model_only_mode_failure_preserves_failing_gate_and_both_outputs(
    monkeypatch, tmp_path
):
    module = measurement()
    values = {"bf16-graph": np.array([1.0]), "bf16-compiled": np.array([2.0])}

    def mode_result(
        policy, session, args, observations, ids, plan, label, stream, **kwargs
    ):
        return {
            "samples": [],
            "gates": [],
            "graph_audits": [],
            "outputs": {(0, 0): {"action": values[label]}},
            "retained": {f"r0/{label}/o0/prepared_model/action": values[label]},
        }

    monkeypatch.setattr(module, "_model_only_mode", mode_result)
    optimized = SimpleNamespace(set_runtime=lambda **kwargs: None, evidence=dict)
    args = SimpleNamespace(modes=["graph", "compiled"], output=tmp_path)
    with pytest.raises(RuntimeError, match="Model-only runtime output differs"):
        module._model_only_precision(
            object(),
            optimized,
            args,
            "bf16",
            [object()],
            [{"observation_id": "obs-0"}],
            [],
            StringIO(),
        )
    gates = json.loads((tmp_path / "validation.json").read_text())
    assert gates[-1]["mode"] == "bf16-compiled"
    assert gates[-1]["reference_mode"] == "bf16-graph"
    assert gates[-1]["same_precision"] == {
        "passes": False,
        "max_abs": 1.0,
        "reason": None,
    }
    with np.load(tmp_path / "actions.npz") as actions:
        for label, value in values.items():
            np.testing.assert_array_equal(
                actions[f"r0/{label}/o0/prepared_model/action"], value
            )


@pytest.mark.parametrize("fail_last", [False, True])
def test_model_only_artifacts_accumulate_all_precisions_on_success_or_failure(
    monkeypatch, tmp_path, fail_last
):
    module = measurement()
    output = tmp_path / "replay"
    args = module.build_parser().parse_args(
        [
            "--checkpoint",
            "unused",
            "--dataset",
            "unused",
            "--output",
            str(output),
            "--modes",
            "graph",
            "compiled",
            "--rounds",
            "1",
            "--observations",
            "1",
            "--measurement-boundary",
            "model-only",
        ]
    )
    model = SimpleNamespace(
        config=SimpleNamespace(streaming=False, action_horizon=40),
        action_head=SimpleNamespace(num_inference_timesteps=4),
    )

    class Session:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_runtime(self, **kwargs):
            pass

        def evidence(self):
            return {}

    def mode_result(
        policy, session, args, observations, ids, plan, label, stream, **kwargs
    ):
        value = np.array([2.0 if fail_last and label == "w4a4-compiled" else 1.0])
        return {
            "samples": [],
            "gates": [
                {
                    "round": 0,
                    "mode": label,
                    "observation": 0,
                    "prepared_model": {"passes": True},
                    "rng_equal": True,
                }
            ],
            "graph_audits": [
                {
                    "round": 0,
                    "mode": label,
                    "observation": 0,
                    "prepared_model": {"passes": True},
                }
            ],
            "outputs": {(0, 0): {"action": value}},
            "retained": {f"r0/{label}/o0/prepared_model/action": value},
        }

    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device=lambda _: nullcontext(),
            get_device_name=lambda _: "test-interface",
        ),
        device=lambda _: SimpleNamespace(type="cuda"),
        inference_mode=nullcontext,
        set_num_threads=lambda _: None,
        set_num_interop_threads=lambda _: None,
        __version__="interface-only",
        version=SimpleNamespace(cuda="none"),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(module, "_source_manifest", lambda _: {})
    monkeypatch.setattr(module, "_config", lambda *args: None)
    monkeypatch.setattr(
        module,
        "_observations",
        lambda *args: ([object()], [{"observation_id": "obs-0"}]),
    )
    monkeypatch.setattr(module, "_model_only_mode", mode_result)
    monkeypatch.setattr(
        importlib.import_module("coexecution.libero.service"),
        "load_core",
        lambda *args: SimpleNamespace(model=model),
    )
    monkeypatch.setattr(
        importlib.import_module("coexecution.optimization.session"),
        "GrootOptimizations",
        Session,
    )
    monkeypatch.setattr(
        importlib.import_module("scripts.libero_experiment_utils"),
        "checkpoint_identity",
        lambda _: {},
    )
    if fail_last:
        with pytest.raises(RuntimeError, match="Model-only runtime output differs"):
            module.run(args)
    else:
        module.run(args)
    gates = json.loads((output / "validation.json").read_text())
    audits = json.loads((output / "graph-timing-audit.json").read_text())
    expected = {
        f"{precision}-{mode}" for precision in args.precisions for mode in args.modes
    }
    assert len(gates) == 12 and {row["mode"] for row in gates} == expected
    assert len(audits) == 6 and {row["mode"] for row in audits} == expected
    assert gates[-1]["same_precision"]["passes"] is not fail_last
    with np.load(output / "actions.npz") as actions:
        assert set(actions.files) == {
            f"r0/{label}/o0/prepared_model/action" for label in expected
        }
    if not fail_last:
        result = json.loads((output / "results.json").read_text())
        assert result["validation"] == gates
        assert result["graph_timing_audit"] == audits
