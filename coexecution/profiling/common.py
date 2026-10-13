"""Metadata and measurement helpers; importing this module does not load CUDA."""

import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_output_path(nominal, replay_root, *, pid=None, time_ns=None):
    """No filesystem or GPU effects; replay applications get disjoint outputs."""
    if replay_root is None:
        return Path(nominal)
    pid = os.getpid() if pid is None else pid
    time_ns = time.time_ns() if time_ns is None else time_ns
    return Path(replay_root) / f"pass-{pid}-{time_ns}"


def create_output(args):
    args.nominal_output = args.output
    args.output = choose_output_path(args.output, args.replay_output_root)
    args.output.mkdir(parents=True, exist_ok=False)
    args.output_created = True
    print(
        json.dumps(
            {
                "worker_output": str(args.output),
                "replay_output_root": str(args.replay_output_root)
                if args.replay_output_root is not None
                else None,
            }
        ),
        flush=True,
    )


@contextlib.contextmanager
def cuda_capture_range(enabled):
    if not enabled:
        yield
        return
    import torch

    runtime = torch.cuda.cudart()
    rc = runtime.cudaProfilerStart()
    if int(rc) != 0:
        raise RuntimeError(f"cudaProfilerStart failed: {rc}")
    try:
        yield
    finally:
        runtime.cudaProfilerStop()


def observe_ready(event):
    """Host-clock timestamp after this job's CUDA completion, never device-wide."""
    event.synchronize()
    return time.perf_counter_ns()


def response_ms(release_ns, ready_ns):
    """Host-observed GPU-output readiness, including any pre-submit permit wait."""
    if ready_ns < release_ns:
        raise ValueError("Readiness cannot precede release on the same host clock")
    return (ready_ns - release_ns) / 1e6


def stats(values):
    import numpy as np

    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(np.std(values)),
    }


def tensor_digest(tensor):
    import torch

    return hashlib.sha256(
        tensor.detach().contiguous().view(dtype=torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def launcher_state():
    result = {}
    for name, cache in (
        ("coexecution.fused_ops", "_LAUNCHERS"),
        ("coexecution.kernels.common.fused", "_LUTS"),
        ("coexecution.optimization.backends.integer.linear", "_LAUNCHERS"),
    ):
        module = sys.modules.get(name)
        if module is not None:
            result[f"{name}.{cache}"] = sorted(
                repr(key) for key in getattr(module, cache, {})
            )
    return result


def metadata(mapping):
    import torch

    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
        for name, value in mapping.items()
        if torch.is_tensor(value)
    }
