"""Feature wire contracts use host arrays only; CUDA tests are explicit opt-in."""

import importlib
import sys
from copy import deepcopy
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest


def codec():
    return importlib.import_module("coexecution.libero.features")


def test_episode_vision_preparation_records_real_work_without_feature_publication():
    calls = []
    graph = {"capture_delta": {"vlm": 1}, "cold_capture": True}

    def call(endpoint, payload):
        calls.append((endpoint, payload))
        return {
            "meta": {
                "capture_s": 9.0,
                "source_tick": 0,
                "vlm_forward_calls": 1,
                "vlm_seconds": 0.3,
                "graph_execution": graph,
                "frame_hashes": {"image": "actual"},
            }
        }

    clock = iter([10.0, 10.31])
    record = codec().prepare_episode_vision(
        SimpleNamespace(call_endpoint=call),
        {"language": {"task": "new"}},
        9.0,
        clock=lambda: next(clock),
    )
    assert calls == [
        (
            "vision",
            {
                "observation": {"language": {"task": "new"}},
                "capture_s": 9.0,
                "source_tick": 0,
                "return_features": False,
            },
        )
    ]
    assert record["capture_s"] == 9.0
    assert record["started_s"] == 10.0 and record["completed_s"] == 10.31
    assert record["rpc_seconds"] == pytest.approx(0.31)
    assert record["vlm_forward_calls"] == 1
    assert record["graph_execution"] == graph
    assert "vl_embeds" not in record


@pytest.mark.parametrize(
    "change",
    [
        {"vlm_forward_calls": 0},
        {"vlm_forward_calls": 2},
        {"capture_s": 9.1},
        {"source_tick": 1},
    ],
)
def test_episode_preparation_rejects_wrong_compute_or_source_before_control(change):
    meta = {
        "capture_s": 9.0,
        "source_tick": 0,
        "vlm_forward_calls": 1,
        "vlm_seconds": 0.3,
        **change,
    }
    client = SimpleNamespace(call_endpoint=lambda *_: {"meta": meta})
    with pytest.raises(RuntimeError):
        codec().prepare_episode_vision(client, {}, 9.0)


@pytest.mark.parametrize(
    "layout,transport",
    [
        ("split", "numpy_fp32"),
        ("split", "numpy_bf16"),
        ("paired", "numpy_bf16"),
        ("paired", "cuda_local"),
    ],
)
def test_each_episode_prepares_s2_before_fresh_bootstrap_even_with_cached_calibration(
    monkeypatch,
    layout,
    transport,
):
    import io
    from unittest.mock import Mock

    import libero_wallclock as wallclock

    events, replies = [], []
    args = SimpleNamespace(
        control_hz=20,
        slow_warmup={"cached": True},
        calibration=(1, {"selected_delay_ticks": 1}),
        port=1,
        slow_port=2,
        max_steps=5,
        record_video=False,
        server_layout=layout,
        feature_transport=transport,
    )

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["port"] == 2
            self.socket = SimpleNamespace(setsockopt=lambda *args: None)
            self.context = SimpleNamespace(
                setsockopt=lambda *args: None, destroy=lambda **kwargs: None
            )

        def call_endpoint(self, name, payload):
            assert name == "vision" and payload["return_features"] is False
            assert "feature_session" not in payload
            language = payload["observation"]["language"]["language.task"][0][0]
            events.append(("prepare", language))
            return {
                "meta": {
                    "capture_s": payload["capture_s"],
                    "source_tick": 0,
                    "vlm_forward_calls": 1,
                    "vlm_seconds": 0.3,
                    "graph_execution": {
                        "capture_delta": {"vlm": 1},
                        "cold_capture": True,
                    },
                }
            }

    def bootstrap(name, payload):
        assert name == "bootstrap" and payload["delay_ticks"] == 1
        events.append(
            ("bootstrap", payload["observation"]["language"]["language.task"][0][0])
        )
        assert payload["capture_s"] >= replies[-1]["completed_s"]
        return {
            "actions": np.zeros((16, 7)),
            "bootstrap_seconds": 0.01,
            "forward_counts": {"vlm": 1, "dit": 4},
        }

    # Observe the real reusable preparation helper, rather than replace its RPC.
    original = codec().prepare_episode_vision

    def prepare(*args, **kwargs):
        record = original(*args, **kwargs)
        replies.append(record)
        return record

    monkeypatch.setitem(
        sys.modules, "gr00t.policy.server_client", SimpleNamespace(PolicyClient=Client)
    )
    monkeypatch.setitem(sys.modules, "zmq", SimpleNamespace(LINGER=17))
    monkeypatch.setattr(wallclock, "prepare_episode_vision", prepare)
    monkeypatch.setattr(wallclock, "ActionWorker", Mock())
    monkeypatch.setattr(wallclock, "VisionWorker", Mock())

    def control(_step, observation, *args, **kwargs):
        events.append(("control", observation["language.task"]))
        return {}, []

    monkeypatch.setattr(wallclock, "run_control_loop", control)
    for task in ("first", "second"):
        observation = {"language.task": task}

        def refresh(_task=task, _observation=observation, **kwargs):
            events.append(("refresh", _task))
            return _observation

        env = SimpleNamespace(
            _process_observation=lambda raw: raw,
            _env=SimpleNamespace(env=SimpleNamespace(_get_observations=refresh)),
        )
        result, _ = wallclock.deployment_episode(
            env,
            observation,
            SimpleNamespace(call_endpoint=bootstrap),
            args,
            1000,
            io.StringIO(),
        )
        assert result["vision_preparation"] == replies[-1]
        assert result["calibration"] is args.calibration[1]
    assert events == [
        (phase, task)
        for task in ("first", "second")
        for phase in ("prepare", "refresh", "bootstrap", "control")
    ]


def envelope():
    return {
        "schema": "libero.features",
        "version": 1,
        "encoding": "numpy_bf16",
        "fields": {
            "backbone_features": {
                "dtype": "bfloat16",
                "shape": [2, 3],
                "data": np.array(
                    [[0x0000, 0x8000, 0x7F80], [0xFF80, 0x7FC1, 0xFFFF]],
                    dtype=np.uint16,
                ),
            },
        },
    }


def test_explicit_bfloat16_tag_preserves_every_word_without_float_conversion():
    payload = envelope()
    words = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    payload["fields"]["backbone_features"].update(shape=list(words.shape), data=words)
    encoding, fields = codec().validate_features(payload)
    assert encoding == "numpy_bf16"
    dtype, actual = fields["backbone_features"]
    assert dtype == "bfloat16"
    assert actual.tobytes() == words.tobytes()
    evidence = codec().feature_evidence(payload)
    assert evidence["payload_nbytes"] == words.nbytes
    assert evidence["field_dtypes"] == {"backbone_features": "bfloat16"}


@pytest.mark.parametrize(
    "dtype", ["bool", "int8", "uint16", "int64", "float16", "float32", "float64"]
)
@pytest.mark.parametrize("shape", [(), (2, 3)])
def test_typed_numpy_roundtrip_preserves_dtype_shape_and_owns_source(dtype, shape):
    original = np.ones(shape, dtype=dtype)
    payload = codec().encode_features({"value": original}, "numpy_bf16")
    original[...] = 0
    _, fields = codec().validate_features(payload)
    declared, actual = fields["value"]
    assert declared == np.dtype(dtype).name
    assert actual.dtype == original.dtype
    assert actual.shape == original.shape
    assert np.all(actual == 1)
    assert not np.shares_memory(actual, original)


def test_legacy_uint16_is_never_inferred_to_be_bfloat16():
    value = np.array([0x3F80, 0x8000], dtype=np.uint16)
    encoding, fields = codec().validate_features({"value": value})
    assert encoding == "numpy_fp32"
    assert fields["value"][0] == "uint16"
    assert fields["value"][1].tobytes() == value.tobytes()


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "unknown"},
        {"version": 2},
        {"version": True},
        {"encoding": "guess"},
        {"fields": {}},
        {"extra": True},
    ],
)
def test_unknown_envelopes_are_rejected(change):
    payload = envelope()
    payload.update(change)
    with pytest.raises(ValueError):
        codec().validate_features(payload)


@pytest.mark.parametrize(
    "change",
    [
        {"dtype": "brainfloat"},
        {"dtype": "float32"},
        {"shape": [6]},
        {"shape": [-2, 3]},
        {"shape": [True, 3]},
        {"shape": [2.0, 3]},
        {"data": [1, 2, 3]},
        {"data": np.ones((2, 3), dtype=object)},
        {"data": np.ones((2, 3), dtype=">u2")},
        {"extra": "undeclared"},
    ],
)
def test_malformed_fields_are_rejected_before_a_decoder_can_mutate_state(change):
    payload = envelope()
    payload["fields"]["backbone_features"].update(change)
    with pytest.raises(ValueError):
        codec().validate_features(payload)


def test_legacy_arrays_and_scalars_keep_owned_wire_storage():
    original = {"value": np.ones(2, dtype=np.float32), "mask": np.bool_(True)}
    payload = codec().encode_features(original, "numpy_fp32")
    original["value"][:] = 2
    encoding, fields = codec().validate_features(payload)
    assert encoding == "numpy_fp32"
    assert np.all(fields["value"][1] == 1)
    assert fields["mask"][1].shape == ()
    assert fields["mask"][1].dtype == np.bool_


def test_causal_metadata_validation_does_not_modify_input():
    meta = {
        "capture_s": 10.0,
        "source_tick": 2,
        "requested_tick": 1,
        "frame_hashes": {"image": "abc"},
    }
    before = deepcopy(meta)
    codec().validate_feature_meta(meta, state_capture_s=10.0, request_tick=2)
    assert meta == before
    for changes in (
        {"capture_s": 10.001},
        {"source_tick": 3},
        {"capture_s": float("nan")},
        {"capture_s": float("inf")},
    ):
        with pytest.raises(ValueError):
            codec().validate_feature_meta(
                {**meta, **changes}, state_capture_s=10.0, request_tick=2
            )


def backend(tmp_path):
    from coexecution.libero.backend import Backend

    instance = Backend.__new__(Backend)
    instance.counts = {"vlm": 0, "dit": 0}
    instance.variant = "flow"
    instance.output = Path(tmp_path)
    instance.anchor = None
    instance.cache_meta = {"capture_s": 1.0, "source_tick": 0}
    instance.cache_install_timing = {}
    instance.feature_transport = "numpy_bf16"
    instance.stream = SimpleNamespace(synchronize=lambda: None)
    instance.torch = SimpleNamespace()
    instance.BatchFeature = dict
    core = SimpleNamespace(
        model=SimpleNamespace(device="cuda:0", dtype="bfloat16"),
        _vl_cache={"old": object()},
        _vl_cache_lock=Lock(),
        _vl_cache_id=3,
        _vl_cache_t_image_capture=1.0,
    )

    def prime(features, capture):
        core._vl_cache, core._vl_cache_t_image_capture = features, capture
        core._vl_cache_id += 1
        return core._vl_cache_id, capture

    def action(*args, **kwargs):
        instance.counts["dit"] += 4
        return {
            key: np.zeros((1, 16, 1))
            for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        }, {"image_delay_ticks": 0, "image_delay_ms": 0}

    core._prime_vlm_cache, core.get_action_chunk_cached = prime, action
    instance.core = core
    return instance


def test_install_validates_all_fields_before_decode_or_cache_change(
    tmp_path, monkeypatch
):
    from coexecution.libero import backend as module

    instance = backend(tmp_path)
    old_cache, old_meta = instance.core._vl_cache, instance.cache_meta
    payload = envelope()
    payload["fields"]["bad"] = {"dtype": "unknown", "shape": [], "data": np.array(1)}
    monkeypatch.setattr(
        module,
        "decode_features",
        lambda *a, **k: pytest.fail("Malformed wire transferred"),
    )
    with pytest.raises(ValueError):
        instance.install(payload, {"capture_s": 2.0, "source_tick": 1})
    assert instance.core._vl_cache is old_cache
    assert instance.cache_meta is old_meta


def test_decode_failure_keeps_previous_cache_and_metadata(tmp_path, monkeypatch):
    from coexecution.libero import backend as module

    instance = backend(tmp_path)
    old_cache, old_meta = instance.core._vl_cache, instance.cache_meta

    def fail(*args, **kwargs):
        raise RuntimeError("transfer failed")

    monkeypatch.setattr(module, "decode_features", fail)
    with pytest.raises(RuntimeError, match="transfer failed"):
        instance.install(envelope(), {"capture_s": 2.0, "source_tick": 1})
    assert instance.core._vl_cache is old_cache
    assert instance.cache_meta is old_meta


def test_combined_plan_installs_once_and_records_distinct_server_boundaries(
    tmp_path, monkeypatch
):
    from coexecution.libero import backend as module

    instance = backend(tmp_path)
    decoded = {"fresh": object()}
    monkeypatch.setattr(module, "decode_features", lambda *args, **kwargs: decoded)
    meta = {
        "capture_s": 2.0,
        "source_tick": 1,
        "requested_tick": 1,
        "frame_hashes": {"image": "abc"},
    }
    result = instance.plan(
        {"state": {}},
        1,
        1,
        np.zeros((1, 7)),
        2.0,
        vl_embeds=envelope(),
        feature_meta=meta,
    )
    audit = result["audit"]
    assert instance.core._vl_cache == decoded
    assert instance.core._vl_cache_id == 4
    assert audit["forward_counts"] == {"vlm": 0, "dit": 4}
    assert audit["feature_installed_in_request"]
    assert (
        audit["server_request_started_s"]
        <= audit["server_install_started_s"]
        <= audit["server_install_completed_s"]
        <= audit["server_plan_started_s"]
    )
    assert audit["cache"]["feature_wire"]["encoding"] == "numpy_bf16"
    assert audit["cache"]["frame_hashes"] == meta["frame_hashes"]
    meta["frame_hashes"]["image"] = "mutated"
    assert instance.cache_meta["frame_hashes"]["image"] == "abc"
    second = instance.plan({"state": {}}, 2, 1, np.zeros((1, 7)), 2.0)["audit"]
    assert not second["feature_installed_in_request"]
    assert second["server_plan_started_s"] >= audit["server_plan_started_s"]
    assert instance.core._vl_cache_id == 4


@pytest.mark.parametrize(
    "changes", [{"capture_s": 2.01}, {"source_tick": 2}, {"capture_s": float("nan")}]
)
def test_combined_future_or_invalid_feature_rejected_before_install(
    tmp_path, monkeypatch, changes
):
    instance = backend(tmp_path)
    monkeypatch.setattr(
        instance, "install", lambda *args: pytest.fail("Future feature installed")
    )
    with pytest.raises(ValueError):
        instance.plan(
            {"state": {}},
            1,
            1,
            np.zeros((1, 7)),
            2.0,
            vl_embeds=envelope(),
            feature_meta={"capture_s": 2.0, "source_tick": 1, **changes},
        )
    assert instance.core._vl_cache_id == 3
    assert instance.counts == {"vlm": 0, "dit": 0}


def test_compact_vision_snapshots_actual_cache_without_requesting_upcast(
    tmp_path, monkeypatch
):
    from coexecution.libero import backend as module

    instance = backend(tmp_path)
    actual = {"backbone_features": object()}
    calls = []

    def update(observation, return_features, t_image_capture):
        calls.append(return_features)
        instance.core._prime_vlm_cache(actual, t_image_capture)
        instance.counts["vlm"] += 1
        return {"cache_id": instance.core._vl_cache_id, "seq_len": 3}

    def encode(features, encoding, *, torch_module):
        assert features == actual
        assert encoding == "numpy_bf16"
        assert torch_module is instance.torch
        return envelope()

    instance.core.update_vlm_cache = update
    monkeypatch.setattr(module, "encode_features", encode)
    result = instance.vision({}, 2.0, 1)
    assert calls == [False]
    assert result["meta"]["feature_wire"]["source"] == "vlm_cache_raw_tensor"
    assert result["meta"]["feature_wire"]["payload_nbytes"] == 12
    assert result["meta"]["vlm_forward_calls"] == 1
    instance.vision({}, 3.0, 2, return_features=False)
    assert calls == [False, False]


@pytest.mark.parametrize(
    "mode, expected_calls",
    [("separate", ["install", "plan", "plan"]), ("combined", ["plan", "plan"])],
)
def test_action_worker_deduplicates_feature_and_uses_explicit_server_boundary(
    monkeypatch, mode, expected_calls
):
    from libero_wallclock import ActionWorker, Observation

    calls = []
    clock = SimpleNamespace(now=lambda: 10.0)

    class Client:
        def __init__(self, **kwargs):
            self.meta = {"capture_s": 9.0, "source_tick": 0}

        def call_endpoint(self, name, payload):
            calls.append((name, payload))
            if name == "install":
                self.meta = payload["meta"]
                return {}
            if "vl_embeds" in payload:
                self.meta = payload["feature_meta"]
            return {
                "actions": np.zeros((1, 7)),
                "audit": {
                    "cache": self.meta,
                    "valid_start_tick": 1,
                    "valid_end_tick": 2,
                    "server_seconds": 0.01,
                    "server_plan_started_s": 10.005,
                },
            }

    monkeypatch.setitem(
        sys.modules, "gr00t.policy.server_client", SimpleNamespace(PolicyClient=Client)
    )
    visual = {
        "sequence": 8,
        "vl_embeds": envelope(),
        "meta": {"capture_s": 9.0, "source_tick": 0},
    }
    actor = ActionWorker(1, feature_rpc=mode, clock=clock)
    try:
        for _ in range(2):
            result = actor.submit(
                Observation({}, 0, 9.0), 0, 1, np.zeros((1, 7)), visual, 10.05, 10.0
            ).result()
    finally:
        actor.close()
    assert [name for name, _ in calls] == expected_calls
    assert "vl_embeds" not in calls[-1][1]
    timing = result["client_timing"]
    assert timing["feature_sequence"] == 8
    assert timing["plan_rpc_started_s"] == 10.0
    assert timing["server_plan_started_s"] == 10.005
    assert timing["cache_age_at_server_plan_ms"] == pytest.approx(1005)
    assert timing["cache_age_at_request_ms"] == 1000
    assert timing["feature_use_boundary"] == "server_plan_started_s"


def test_combined_failed_reply_does_not_advance_client_feature_sequence(monkeypatch):
    from libero_wallclock import ActionWorker, Observation

    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def call_endpoint(self, name, payload):
            calls.append(payload)
            if len(calls) == 1:
                raise RuntimeError("reply failed")
            return {
                "actions": np.zeros((1, 7)),
                "audit": {
                    "cache": payload["feature_meta"],
                    "valid_start_tick": 1,
                    "valid_end_tick": 2,
                    "server_seconds": 0.01,
                },
            }

    monkeypatch.setitem(
        sys.modules, "gr00t.policy.server_client", SimpleNamespace(PolicyClient=Client)
    )
    actor = ActionWorker(1, feature_rpc="combined")
    visual = {
        "sequence": 8,
        "vl_embeds": envelope(),
        "meta": {"capture_s": 9.0, "source_tick": 0},
    }
    args = (Observation({}, 0, 9.0), 0, 1, np.zeros((1, 7)), visual, 10.05, 10.0)
    try:
        with pytest.raises(RuntimeError, match="reply failed"):
            actor.submit(*args).result()
        actor.submit(*args).result()
    finally:
        actor.close()
    assert len(calls) == 2
    assert all(payload["vl_embeds"] is visual["vl_embeds"] for payload in calls)


@pytest.mark.parametrize("mode", ["separate", "combined"])
def test_worker_rejects_future_feature_before_any_network_state_change(
    monkeypatch, mode
):
    from libero_wallclock import ActionWorker, Observation

    monkeypatch.setitem(
        sys.modules,
        "gr00t.policy.server_client",
        SimpleNamespace(
            PolicyClient=lambda **kwargs: SimpleNamespace(
                call_endpoint=lambda *args: pytest.fail("Future feature sent")
            )
        ),
    )
    actor = ActionWorker(1, feature_rpc=mode)
    visual = {
        "sequence": 8,
        "vl_embeds": envelope(),
        "meta": {"capture_s": 10.0, "source_tick": 0},
    }
    try:
        with pytest.raises(ValueError):
            actor.submit(
                Observation({}, 0, 9.0), 0, 1, np.zeros((1, 7)), visual, 10.05, 10.0
            ).result()
    finally:
        actor.close()


def test_feature_cli_defaults_and_deployment_guard():
    from evaluate_libero_protocol import build_parser, validate_deployment_controls

    parser = build_parser()
    common = ["--protocol", "deployment", "--variant", "flow", "--output", "unused"]
    defaults = parser.parse_args(common)
    assert defaults.feature_transport == "numpy_fp32"
    assert defaults.feature_rpc == "separate"
    for flags in (["--feature-transport", "numpy_bf16"], ["--feature-rpc", "combined"]):
        args = parser.parse_args([*common, *flags])
        validate_deployment_controls(args)
        args.protocol = "algorithm"
        with pytest.raises(ValueError, match="deployment"):
            validate_deployment_controls(args)


def test_feature_identity_mismatch_fails_before_evaluation():
    codec().validate_feature_identity({"feature_transport": "numpy_bf16"}, "numpy_bf16")
    codec().validate_feature_identity({}, "numpy_fp32")
    with pytest.raises(ValueError, match="transport"):
        codec().validate_feature_identity(
            {"feature_transport": "numpy_fp32"}, "numpy_bf16"
        )


def test_legacy_vision_keeps_upstream_export_and_reset_clears_install_evidence(
    tmp_path, monkeypatch
):
    from coexecution.libero import backend as module

    instance = backend(tmp_path)
    instance.feature_transport = "numpy_fp32"
    instance.cache_install_timing = {"server_install_completed_s": 8.0}
    calls, seeds = [], []

    def update(observation, return_features, t_image_capture):
        calls.append(return_features)
        instance.counts["vlm"] += 1
        return {"vl_embeds": {"value": np.ones(2, dtype=np.float32)}}

    instance.core.update_vlm_cache = update
    instance.core.reset_streaming_buffer = lambda: None
    instance.torch.manual_seed = seeds.append
    monkeypatch.setattr(
        module,
        "encode_features",
        lambda *a, **k: pytest.fail("Legacy exporter replaced"),
    )
    result = instance.vision({}, 2.0, 1)
    assert calls == [True]
    assert result["meta"]["feature_wire"]["source"] == "upstream_return_features"
    assert instance.cache_install_timing == {}
    instance.reset({"seed": 17})
    assert instance.core._vl_cache is None
    assert instance.core._vl_cache_t_image_capture is None
    assert instance.cache_meta == {}
    assert instance.cache_install_timing == {}
    assert seeds == [17]


def test_combined_bootstrap_feature_can_plan_without_any_feature_upload(monkeypatch):
    from libero_wallclock import ActionWorker, Observation

    payloads = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def call_endpoint(self, name, payload):
            assert name == "plan"
            payloads.append(payload)
            return {
                "actions": np.zeros((1, 7)),
                "audit": {
                    "cache": {"capture_s": 9.0, "source_tick": 0},
                    "valid_start_tick": 1,
                    "valid_end_tick": 2,
                    "server_seconds": 0.01,
                },
            }

    monkeypatch.setitem(
        sys.modules, "gr00t.policy.server_client", SimpleNamespace(PolicyClient=Client)
    )
    actor = ActionWorker(
        1,
        feature_rpc="combined",
        initial_feature={"sequence": 0, "meta": {"capture_s": 9.0, "source_tick": 0}},
    )
    try:
        result = actor.submit(
            Observation({}, 0, 9.0), 0, 1, np.zeros((1, 7)), None, 10.05, 10.0
        ).result()
    finally:
        actor.close()
    assert "vl_embeds" not in payloads[0]
    assert result["client_timing"]["feature_sequence"] == 0


def test_combined_calibration_installs_each_sample_on_one_owned_client_thread(
    monkeypatch,
):
    import threading

    from libero_wallclock import calibrate

    calls = []
    parent_thread = threading.get_ident()

    class Client:
        def __init__(self, **kwargs):
            self.owner = threading.get_ident()

        def call_endpoint(self, name, payload):
            assert threading.get_ident() == self.owner != parent_thread
            assert name == "plan"
            calls.append(payload)
            return {
                "actions": np.zeros((1, 7)),
                "audit": {
                    "cache": payload["feature_meta"],
                    "valid_start_tick": payload["request_tick"] + 1,
                    "valid_end_tick": payload["request_tick"] + 2,
                    "server_seconds": 0.01,
                },
            }

    monkeypatch.setitem(
        sys.modules, "gr00t.policy.server_client", SimpleNamespace(PolicyClient=Client)
    )
    boot = SimpleNamespace(call_endpoint=lambda *args: {"actions": np.zeros((16, 7))})
    _, result = calibrate(
        boot,
        {},
        9.0,
        1,
        0.05,
        {"vl_embeds": envelope(), "meta": {"capture_s": 9.0, "source_tick": 0}},
        1,
        feature_rpc="combined",
        feature_transport="numpy_bf16",
    )
    assert len(calls) == 12
    assert all("vl_embeds" in payload for payload in calls)
    assert result["sample_count"] == 10
    assert result["feature_rpc"] == "combined"
    assert result["feature_transport"] == "numpy_bf16"
