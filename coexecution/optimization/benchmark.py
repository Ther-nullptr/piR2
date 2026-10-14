"""Explicit CUDA-only integer GEMM/Linear timing on caller-supplied real inputs.

No model loading, automatic precision selection, profiler, power changes or
performance claim at import time. These timings are not action-request latency.
"""

import random
import statistics


def _measure(functions, device, *, samples, replays, seed):
    import torch

    if (
        type(samples) is not int
        or samples < 5
        or type(replays) is not int
        or replays < 1
    ):
        raise ValueError("Use at least five stable samples and positive replay count")
    caller = torch.cuda.current_stream(device)
    graphs, retained = {}, []
    with torch.inference_mode(), torch.cuda.device(device):
        for name, function in functions.items():
            for _ in range(3):
                retained.append(function())
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(caller)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(replays):
                    output = function()
            caller.wait_stream(stream)
            graphs[name] = graph
            retained.append(output)
        caller.synchronize()
        values = {name: [] for name in functions}
        rng = random.Random(seed)
        for _ in range(samples):
            order = list(functions)
            rng.shuffle(order)
            for name in order:
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record(caller)
                graphs[name].replay()
                end.record(caller)
                end.synchronize()
                values[name].append(start.elapsed_time(end) / replays)
    return {
        name: {"gpu_samples_ms": data, "median_gpu_ms": statistics.median(data)}
        for name, data in values.items()
    }


def benchmark_integer_tactics(
    q, packed, *, samples=12, replays=8, seed=0, minimum_gain=0.03
):
    """Time all eight existing integer tactics, retaining tactic0 unless >3% gain."""
    if (
        q.bits not in (4, 8)
        or packed.bits != q.bits
        or not packed.data.is_cuda
        or packed.logical_k != q.in_features
        or not 0 <= minimum_gain < 1
    ):
        raise ValueError("Compatible CUDA integer projection/pack and gain required")
    ops = getattr(q, "_reference_ops", q._ops)
    operation = ops.gemm_biasless if q.biasless_epilogue else ops.gemm
    functions = {
        str(t): lambda t=t: operation(
            packed.data,
            packed.scales,
            q.packed_weight,
            q.weight_scale,
            q.bias,
            q.bits,
            t,
        )
        for t in range(8)
    }
    timings = _measure(
        functions, packed.data.device, samples=samples, replays=replays, seed=seed
    )
    best = min(range(8), key=lambda t: timings[str(t)]["median_gpu_ms"])
    baseline = timings["0"]["median_gpu_ms"]
    selected = (
        best
        if timings[str(best)]["median_gpu_ms"] < baseline * (1 - minimum_gain)
        else 0
    )
    return {
        "bits": q.bits,
        "shape": [
            packed.data.shape[0],
            q.in_features,
            q.out_features,
            q.bias is not None,
        ],
        "tactic": selected,
        "minimum_gain": minimum_gain,
        "measurements": timings,
        "timing_boundary": "GPU elapsed of integer dot + existing scale/bias/BF16 epilogue; activation pack excluded",
        "schedule": "Three warmups then seeded interleaved Graph replays; compile/capture excluded",
    }


def benchmark_linear_pair(source, q, x, *, samples=12, replays=8, seed=0):
    """Compare original BF16 Linear against complete integer Linear including pack."""
    import torch

    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or not source.weight.is_cuda
        or source.weight.dtype != torch.bfloat16
        or q.bits not in (4, 8)
        or x.shape[-1] != q.in_features
        or source.weight.shape != (q.out_features, q.in_features)
    ):
        raise ValueError(
            "Matched BF16 CUDA source/input and integer projection required"
        )
    timings = _measure(
        {
            "bf16": lambda: torch.nn.functional.linear(x, source.weight, source.bias),
            "integer": lambda: q(x),
        },
        x.device,
        samples=samples,
        replays=replays,
        seed=seed,
    )
    return {
        "bits": q.bits,
        "input_shape": list(x.shape),
        "measurements": timings,
        "speedup": timings["bf16"]["median_gpu_ms"]
        / timings["integer"]["median_gpu_ms"],
        "timing_boundary": "Complete Linear GPU elapsed, including activation scaling/quantization/packing and output conversion; not a complete model request",
    }
