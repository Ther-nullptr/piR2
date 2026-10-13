"""Exclusive GPU activity accounting; CPU records never become GPU work."""

import importlib
import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


def test_leaf_device_work_and_active_union_are_distinct():
    module = importlib.import_module("coexecution.libero.profiling")
    trace = {
        "traceEvents": [
            {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 10000},
            {"ph": "X", "cat": "kernel", "name": "cublas_gemm", "ts": 100, "dur": 2000},
            {"ph": "X", "cat": "kernel", "name": "flash_fwd", "ts": 1100, "dur": 2000},
            {
                "ph": "X",
                "cat": "gpu_memcpy",
                "name": "Memcpy HtoD",
                "ts": 4000,
                "dur": 1000,
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "unknown_work",
                "ts": 6000,
                "dur": 500,
            },
        ]
    }
    report = module.summarize_trace(trace)
    assert report["device_event_count"] == 4
    assert report["gpu_work_ms"] == 5.5
    assert report["gpu_active_union_ms"] == 4.5
    assert sum(row["gpu_ms"] for row in report["categories"]) == 5.5
    assert {row["category"] for row in report["categories"]} == {
        "gemm",
        "attention",
        "transfer",
        "unknown",
    }


def test_stage_uses_launch_correlation_and_keeps_quant_pack_exclusive():
    module = importlib.import_module("coexecution.libero.profiling")
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": module.STAGE_PREFIX + "vlm",
                "ts": 0,
                "dur": 1000,
                "pid": 1,
                "tid": 1,
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaGraphLaunch",
                "ts": 100,
                "dur": 10,
                "pid": 1,
                "tid": 1,
                "args": {"correlation": 7},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "layernorm_pack",
                "ts": 3000,
                "dur": 500,
                "args": {"correlation": 7},
            },
        ]
    }
    report = module.summarize_trace(trace)
    assert report["by_stage"] == [{"stage": "vlm", "event_count": 1, "gpu_ms": 0.5}]
    assert report["categories"] == [
        {"category": "producer_quantization", "event_count": 1, "gpu_ms": 0.5}
    ]


def test_measured_norm_producers_are_separate_from_standalone_activation_packing():
    module = importlib.import_module("coexecution.libero.profiling")
    trace = {
        "traceEvents": [
            {"ph": "X", "cat": "kernel", "name": name, "ts": i * 1000, "dur": 500}
            for i, name in enumerate(
                (
                    "_prepare_norm_modulation",
                    "_prepare_residual_norm_modulation",
                    "void prepare_integer<8>(...)",
                )
            )
        ]
    }
    report = module.summarize_trace(trace)
    assert report["categories"] == [
        {"category": "producer_quantization", "event_count": 2, "gpu_ms": 1.0},
        {"category": "quantization_prepare", "event_count": 1, "gpu_ms": 0.5},
    ]
    assert report["gpu_work_ms"] == 1.5
    assert report["device_event_count"] == 3


def test_model_only_profile_prepares_before_capture_and_marks_actual_request_scopes(
    monkeypatch, tmp_path
):
    module = importlib.import_module("coexecution.libero.profiling")
    measurement = importlib.import_module("coexecution.libero.measurement")
    calls = []

    class Profile:
        def __enter__(self):
            calls.append("capture_start")
            return self

        def __exit__(self, *exc):
            calls.append("capture_end")

        def export_chrome_trace(self, path):
            with open(path, "w") as output:
                json.dump({"traceEvents": []}, output)

    @contextmanager
    def record(name):
        calls.append(name)
        yield

    def prepare(observation):
        calls.append("prepare")
        return {"inputs": observation}, None

    model = SimpleNamespace(
        prepare_input=lambda value: (value, value),
        backbone=lambda value: calls.append("vlm") or value,
        action_head=SimpleNamespace(get_action=lambda *args: calls.append("action")),
    )
    policy = SimpleNamespace(
        model=model,
        _collate_observation=prepare,
        get_action=lambda value: pytest.fail("Raw policy entered model-only profile"),
    )
    session = SimpleNamespace(
        graphs=None, action_graph=None, config=SimpleNamespace(vlm_graph=False)
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda device: None),
        profiler=SimpleNamespace(
            profile=lambda **kwargs: Profile(),
            record_function=record,
            ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(measurement, "_seed_reset", lambda *args: None)
    result = module.collect_model_profile(
        policy,
        [{"pixel": 1}, {"pixel": 2}],
        session,
        tmp_path / "trace.json",
        rounds=2,
        seed=3,
        device="cuda:0",
        model_only=True,
        observation_ids=["obs-a", "obs-b"],
    )
    assert calls.count("prepare") == 2
    assert "prepare" not in calls[calls.index("capture_start") :]
    assert calls.count("vlm") == 6 and calls.count("action") == 6
    assert result["input_preparation_included"] is False
    assert result["instrumented"] is True and result["calls"] == 4
    assert len({row["request_id"] for row in result["requests"]}) == 4
    assert {row["observation_id"] for row in result["requests"]} == {"obs-a", "obs-b"}
    for row in result["requests"]:
        assert row["annotation"] in calls
