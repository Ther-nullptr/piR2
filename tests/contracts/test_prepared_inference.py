"""Inference boundary contracts using objects, never CPU neural computation."""

import importlib
import importlib.util
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

MODULE = "coexecution.libero.inference"


def implementation():
    assert importlib.util.find_spec(MODULE) is not None, "Prepared engine is missing"
    return importlib.import_module(MODULE)


def test_import_does_not_load_frameworks_or_services():
    implementation()
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import sys; import {MODULE}; "
                "assert not any(k in sys.modules for k in ('torch', 'gr00t', 'zmq'))"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr


def test_engine_calls_only_components_and_preserves_options_outputs_and_fresh_inputs():
    module = implementation()
    calls = []
    features, prediction, options = object(), object(), {"image_delay": 2}

    def forbidden(*args, **kwargs):
        pytest.fail("Non-model work entered the prepared engine")

    def backbone(value):
        calls.append(("vlm", value))
        return features

    def head(encoded, value, received):
        calls.append(("action", encoded, value, received))
        return prediction

    model = SimpleNamespace(
        backbone=backbone,
        action_head=SimpleNamespace(get_action=head),
        prepare_input=forbidden,
        get_action=forbidden,
    )
    engine = module.PreparedInferenceEngine(model)
    first = module.PreparedInputs(object(), object(), object(), "frame-0")
    second = module.PreparedInputs(object(), object(), object(), "frame-1")
    assert engine.full(first, options) is prediction
    assert engine.full(second, options) is prediction
    assert calls == [
        ("vlm", first.backbone),
        ("action", features, first.action, options),
        ("vlm", second.backbone),
        ("action", features, second.action, options),
    ]
    assert engine.vlm(first.backbone) is features
    assert engine.action(features, first.action) is prediction
    assert calls[-1] == ("action", features, first.action, None)


def test_instrumentation_is_opt_in_and_component_exceptions_propagate():
    module = implementation()
    events = []

    @contextmanager
    def stage(name):
        events.append(("begin", name))
        try:
            yield
        finally:
            events.append(("end", name))

    failure = RuntimeError("CUDA launch failed")

    def head(*args):
        raise failure

    model = SimpleNamespace(
        backbone=lambda value: value, action_head=SimpleNamespace(get_action=head)
    )
    engine = module.PreparedInferenceEngine(model, stage=stage)
    with pytest.raises(RuntimeError) as raised:
        engine.full(module.PreparedInputs(object(), object(), None, "obs"))
    assert raised.value is failure
    assert events == [
        ("begin", "vlm"),
        ("end", "vlm"),
        ("begin", "action"),
        ("end", "action"),
    ]


@pytest.mark.parametrize("options", [False, 3, "delay", []])
def test_invalid_options_fail_before_any_model_execution(options):
    module = implementation()
    model = SimpleNamespace(backbone=lambda _: pytest.fail("VLM ran"))
    engine = module.PreparedInferenceEngine(model)
    with pytest.raises(TypeError, match="options"):
        engine.full(module.PreparedInputs(None, None, None, "obs"), options)


def test_preparation_owns_input_trees_and_decoding_is_a_separate_adapter():
    module = implementation()
    calls = []
    shared = {"pixels": [1], "state": [2]}
    state = {"state": [3]}

    def collate(observation):
        calls.append(("collate", observation))
        return {"inputs": shared}, state

    def prepare(value):
        calls.append(("prepare", value))
        return value, value

    prediction = object()
    policy = SimpleNamespace(
        _collate_observation=collate,
        model=SimpleNamespace(prepare_input=prepare),
        _decode_action=lambda value, states: (value, states),
    )
    first = module.prepare_observation(policy, "camera-0", observation_id="obs-0")
    shared["pixels"][0] = 9
    state["state"][0] = 10
    second = module.prepare_observation(policy, "camera-1", observation_id="obs-1")
    assert first.backbone["pixels"] == [1]
    assert first.decode_state["state"] == [3]
    assert second.backbone["pixels"] == [9]
    assert second.observation_id == "obs-1"
    assert calls == [
        ("collate", "camera-0"),
        ("prepare", first.backbone),
        ("collate", "camera-1"),
        ("prepare", second.backbone),
    ]
    assert module.decode_prediction(policy, prediction, first) == (
        prediction,
        first.decode_state,
    )


def test_prepared_cohort_is_bounded_and_unique_before_preparation():
    module = implementation()
    policy = SimpleNamespace(_collate_observation=lambda _: pytest.fail("prepared"))
    with pytest.raises(ValueError, match="limit"):
        module.prepare_observations(policy, [1, 2], ["a", "b"], limit=1)
    with pytest.raises(ValueError, match="unique"):
        module.prepare_observations(policy, [1, 2], ["a", "a"], limit=2)
    with pytest.raises(ValueError):
        module.prepare_observations(policy, [1], [], limit=2)


def test_preparation_failures_do_not_execute_a_model_or_return_partial_data():
    module = implementation()
    failure = ValueError("Malformed observation")

    def collate(_):
        raise failure

    with pytest.raises(ValueError) as raised:
        module.prepare_observation(
            SimpleNamespace(_collate_observation=collate), {}, observation_id="obs"
        )
    assert raised.value is failure


def test_owned_snapshot_precedes_preparation_metadata_registration():
    module = implementation()
    raw = {"pixels": [1], "state": [2]}
    registered = []

    def prepare(value):
        metadata = {"token": [7]}
        registered.append(metadata)
        return {"metadata": metadata, "pixels": value["pixels"]}, {
            "state": value["state"]
        }

    def backbone(value):
        assert any(value["metadata"] is item for item in registered), (
            "Prepared metadata identity was copied"
        )
        return value["pixels"]

    model = SimpleNamespace(
        prepare_input=prepare,
        backbone=backbone,
        action_head=SimpleNamespace(
            get_action=lambda features, action, options: (features, action)
        ),
    )
    policy = SimpleNamespace(
        model=model, _collate_observation=lambda value: ({"inputs": value}, None)
    )
    owned = module.prepare_observation(policy, raw, observation_id="frame-0")
    raw["pixels"][0] = 8
    raw["state"][0] = 9
    prediction = module.PreparedInferenceEngine(model).full(owned)
    assert prediction == ([1], {"state": [2]})
