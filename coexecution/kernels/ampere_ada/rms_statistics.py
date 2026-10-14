"""Exact FP32 RMS statistics from BF16 rows without a square workspace.

The bounded Ada (CUDA capability 8.9) fast path instantiates the installed,
pinned PyTorch reduction header with a square-on-load operator. Other inputs
keep the original Triton square kernel and native FP32 mean. Load and warm
before CUDA Graph capture.
"""

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch

from coexecution.kernels.ampere_ada._candidate_support import is_ada_device

_TORCH_VERSION = "2.7.1+cu126"
_TORCH_GIT_VERSION = "e2d141dbde55c2a4370fac5165b0561b6af4798b"
_REDUCE_HEADER_SHA256 = (
    "afde4f3f5831bec60eacf9ebeaf2e4e0ac82e0a48b5fd8296520a67c9645a412"
)
_FAILED_BUILDS = set()


@lru_cache(maxsize=1)
def _pinned_reduce_header():
    header = Path(torch.__file__).parent / "include/ATen/native/cuda/Reduce.cuh"
    try:
        content = header.read_bytes()
    except OSError:
        return None
    return (
        content
        if hashlib.sha256(content).hexdigest() == _REDUCE_HEADER_SHA256
        else None
    )


def _pinned_runtime():
    return (
        torch.__version__ == _TORCH_VERSION
        and torch.version.git_version == _TORCH_GIT_VERSION
        and _pinned_reduce_header() is not None
    )


def _registered_extension(identity):
    extension = torch.ops.pir2_rms_statistics
    if (
        not hasattr(extension, "mean_square")
        or not hasattr(extension, "source_identity")
        or extension.source_identity() != identity
    ):
        raise RuntimeError(
            "Registered RMS extension source identity differs; restart the process"
        )
    return extension


@lru_cache(maxsize=1)
def load_statistics_extension():
    """Build the separate producer from verified installed reduction headers."""
    if not _pinned_runtime():
        raise RuntimeError(
            "RMS statistics require pinned PyTorch and Reduce.cuh SHA256"
        )
    wrapper = Path(__file__).parent / "csrc/rms_statistics.cu"
    identity = hashlib.sha256(
        _pinned_reduce_header()
        + b"\0"
        + wrapper.read_bytes()
        + b"\0"
        + _TORCH_VERSION.encode()
        + b"\0"
        + _TORCH_GIT_VERSION.encode()
    ).hexdigest()
    name = f"pir2_rms_statistics_{identity}_ext"
    if name in _FAILED_BUILDS:
        raise RuntimeError(
            "RMS extension build previously failed for this source identity; "
            "restart the process after fixing the build"
        )
    if hasattr(torch.ops.pir2_rms_statistics, "mean_square"):
        return _registered_extension(identity)
    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Load and warm RMS statistics before CUDA Graph capture")
    from torch.utils.cpp_extension import load

    previous_jobs = os.environ.get("MAX_JOBS")
    os.environ["MAX_JOBS"] = "2"
    source_define = f'-DPIR2_RMS_SOURCE_ID=\\"{identity}\\"'
    try:
        load(
            name=name,
            sources=[str(wrapper)],
            extra_cflags=["-O3", "-std=c++17", source_define],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "--fmad=false",
                "--ftz=false",
                "-gencode=arch=compute_89,code=sm_89",
                source_define,
            ],
            is_python_module=False,
            verbose=False,
        )
        return _registered_extension(identity)
    except Exception:
        # A failed JIT build has already advanced Torch's version bookkeeping;
        # retrying it could otherwise load an old binary without recompiling.
        _FAILED_BUILDS.add(name)
        raise
    finally:
        if previous_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = previous_jobs


def _validate(x):
    if not (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim >= 1
        and not x.requires_grad
    ):
        raise ValueError("RMS statistics require CUDA BF16 rows without autograd")


def eligible(x):
    """Whether this input has the same reduction geometry as native FP32 mean."""
    try:
        _validate(x)
        width = x.shape[-1]
        return (
            x.is_contiguous()
            and 128 <= width <= 8192
            and width % 4 == 0
            and x.numel() > 0
            and x.numel() <= (2**31 - 1) // 4
            and x.data_ptr() % 8 == 0
            and _pinned_runtime()
            and is_ada_device(x.device)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _reference_mean_square(x):
    """Keep the original square kernel and reduction, including fallback tails."""
    # Delayed imports avoid a cycle with rmsnorm, which calls mean_square.
    import triton

    from coexecution.fused_ops import _launch, _square_fp32_kernel

    rows = x.contiguous()
    squared = torch.empty_like(rows, dtype=torch.float32)
    _launch(
        _square_fp32_kernel,
        (triton.cdiv(rows.numel(), 256),),
        (rows, squared),
        (rows.numel(), 256),
    )
    return squared.mean(-1, keepdim=True)


def mean_square(x):
    """Return FP32 ``[*x.shape[:-1], 1]`` with exact original variance bytes.

    Contiguous aligned Ada CUDA BF16 rows of width 128..8192 divisible by four
    use the installed PyTorch 2.7.1 reduction tree, with separate FP32 square
    and accumulator additions. The hypothetical FP32 square must fit native
    32-bit iterator indexing. Other valid devices and layouts retain the
    existing square kernel followed by native mean, with a contiguous logical
    copy for strided inputs.
    """
    _validate(x)
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device)
        x.record_stream(stream)
        if x.numel() == 0:
            # Native mean also preserves NaN output for a zero-width reduction.
            return torch.empty_like(x, dtype=torch.float32).mean(-1, keepdim=True)
        if eligible(x):
            return load_statistics_extension().mean_square(x)
        return _reference_mean_square(x)
