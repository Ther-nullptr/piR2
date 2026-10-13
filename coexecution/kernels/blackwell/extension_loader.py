"""Lazy, source-owned loading for the selected Thor FP8/FP4 operator APIs."""

import os
from pathlib import Path

from coexecution.kernels.build import build_extension

DECODE_POLICIES = {
    "fp8": {"balanced": 0, "low_clock": 1, "legacy": 2},
    "fp4": {"balanced": 0, "low_clock": 1, "mid_clock": 2, "legacy": 3},
}


def resolve_decode_policy(backend):
    """Preserve the reference clock-based policy with piR2-owned settings."""
    policies = DECODE_POLICIES[backend]
    prefix = "PIR2_" + backend.upper()
    name = prefix + "_DECODE_POLICY"
    requested = os.environ.get(name, "auto").strip().lower()
    requested = {
        "low-clock": "low_clock",
        "lowclock": "low_clock",
        "mid-clock": "mid_clock",
        "midclock": "mid_clock",
        "default": "balanced",
    }.get(requested, requested)
    if requested not in ("auto", *policies):
        raise ValueError(f"{name} must be one of {', '.join(('auto', *policies))}")
    low_name = prefix + "_LOW_CLOCK_THRESHOLD_HZ"
    low = int(
        os.environ.get(low_name, 954_000_000 if backend == "fp8" else 540_000_000)
    )
    mid = int(os.environ.get(prefix + "_MID_CLOCK_THRESHOLD_HZ", 1_152_000_000))
    if low <= 0 or (backend == "fp4" and mid <= low):
        raise ValueError("Decode clock thresholds must be positive and increasing")
    devfreq = Path(
        os.environ.get("PIR2_GPU_DEVFREQ_DIR", "/sys/class/devfreq/gpu-gpc-0")
    ).expanduser()
    try:
        frequency = int((devfreq / "max_freq").read_text().strip())
    except (OSError, ValueError):
        frequency = None
    policy = requested
    if requested == "auto":
        policy = "balanced"
        if frequency is not None and frequency <= low:
            policy = "low_clock"
        elif backend == "fp4" and frequency is not None and frequency <= mid:
            policy = "mid_clock"
    return {
        "policy": policy,
        "policy_id": policies[policy],
        "requested": requested,
        "max_freq_hz": frequency,
        "low_threshold_hz": low,
        "mid_threshold_hz": mid if backend == "fp4" else None,
        "source": "environment"
        if requested != "auto"
        else "auto:max_freq"
        if frequency is not None
        else "auto:fallback",
    }


def require_backend(backend, namespace, required_ops, device=None):
    """Validate hardware before compiling; never search external shared libraries."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"Owned CUTLASS {backend.upper()} requires CUDA")
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability != (11, 0):
        raise RuntimeError(
            f"Owned CUTLASS {backend.upper()} currently requires Thor SM110"
        )
    ops = getattr(torch.ops, namespace)
    # The common loader verifies even preexisting namespaces against this tree.
    build_extension(backend)
    missing = [name for name in required_ops if not hasattr(ops, name)]
    if missing:
        raise RuntimeError(
            f"{namespace} is missing required operators: {', '.join(missing)}"
        )
    if not ops.supports_device(110):
        raise RuntimeError(f"{namespace} rejects Thor SM110")
    policy = resolve_decode_policy(backend)
    ops.set_decode_policy(policy["policy_id"])
    if backend == "fp8":
        value = os.environ.get("PIR2_FP8_SWAP_AB", "1").strip().lower()
        if value not in ("0", "1", "false", "true", "no", "yes", "off", "on"):
            raise ValueError("PIR2_FP8_SWAP_AB must be a boolean value")
        ops.set_swap_ab_enabled(value in ("1", "true", "yes", "on"))
    return ops
