"""Exclusive GPU activity accounting; CPU records never become GPU work."""

import importlib


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
