"""Optional exact single-row GEMV for the pinned packed INT4 representation.

This independent CUDA extension reads symmetric-row-v1 bytes directly. It
reduces integer dot products before the reference FP32 scaling order and the
single BF16 conversion. Load and warm it before CUDA Graph capture.
"""

import hashlib
import math
import os
from functools import lru_cache
from pathlib import Path

import torch

from coexecution.kernels.ampere_ada._candidate_support import (
    is_ada_device,
    owned_build_identity,
    projection_for_rows,
    require_pre_capture_build,
)

_FAILED_BUILDS = set()


def _registered_extension(identity):
    extension = torch.ops.pir2_int4_gemv
    if (
        not hasattr(extension, "gemv")
        or not hasattr(extension, "source_identity")
        or extension.source_identity() != identity
    ):
        raise RuntimeError(
            "Registered gemv extension source identity differs; restart the process"
        )
    return extension


@lru_cache(maxsize=1)
def load_gemv_extension():
    """Load verified repository-owned source before CUDA Graph capture."""
    source = Path(__file__).parent / "csrc/int4_gemv.cu"
    identity = hashlib.sha256(
        source.read_bytes() + b"\0" + owned_build_identity()
    ).hexdigest()
    name = f"pir2_int4_gemv_{identity}_ext"
    if name in _FAILED_BUILDS:
        raise RuntimeError(
            "Gemv extension build previously failed for this source identity; "
            "restart the process after fixing the build"
        )
    if hasattr(torch.ops.pir2_int4_gemv, "gemv"):
        return _registered_extension(identity)
    require_pre_capture_build()
    from torch.utils.cpp_extension import load

    previous_jobs = os.environ.get("MAX_JOBS")
    os.environ["MAX_JOBS"] = "2"
    source_define = f'-DPIR2_GEMV_SOURCE_ID=\\"{identity}\\"'
    try:
        load(
            name=name,
            sources=[str(source)],
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
        _FAILED_BUILDS.add(name)
        raise
    finally:
        if previous_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = previous_jobs


def eligible_shape(rows, q):
    """Check the physical single-row INT4 shape without preparing buffers."""
    weight = q.packed_weight
    if rows != 1 or q.bits != 4 or weight.ndim != 2:
        return False
    n, stored_k = weight.shape
    k = stored_k * 2
    return 128 <= k <= 8192 and k % 128 == 0 and n > 0 and n % 8 == 0


def _validate_pack(packed, q):
    if (
        packed.bits != q.bits
        or packed.logical_k != q.in_features
        or packed.format_version != "symmetric-row-v1"
    ):
        raise ValueError("Incompatible activation pack")
    a, sa = packed.data, packed.scales
    b, sb = q.packed_weight, q.weight_scale
    storage = torch.uint8 if q.bits == 4 else torch.int8
    tensors = (a, sa, b, sb) if q.bias is None else (a, sa, b, sb, q.bias)
    if not (
        q.bits in (4, 8)
        and q.in_features > 0
        and a.ndim == b.ndim == 2
        and a.shape[1] == b.shape[1]
        and a.dtype == b.dtype == storage
        and sa.dtype == sb.dtype == torch.float32
        and sa.ndim == sb.ndim == 1
        and sa.shape[0] == a.shape[0]
        and sb.shape[0] == b.shape[0]
        and b.shape[0] > 0
        and b.shape[0] % 8 == 0
        and a.shape[1] * (2 if q.bits == 4 else 1) == (q.in_features + 127) // 128 * 128
        and math.prod(packed.shape) == a.shape[0]
        and all(
            tensor.is_cuda and tensor.device == a.device and tensor.is_contiguous()
            for tensor in tensors
        )
        and (
            q.bias is None
            or (
                q.bias.dtype == torch.bfloat16
                and q.bias.ndim == 1
                and q.bias.shape[0] == b.shape[0]
            )
        )
    ):
        raise ValueError("Invalid packed buffers for integer GEMV")
    # GEMV reads uint32 words; the CUTLASS fallback reads 128-bit operands.
    # Empty results read neither packed matrix.
    if a.shape[0]:
        native = is_ada_device(a.device) and eligible_shape(a.shape[0], q)
        alignment = 4 if native else 16
        if a.data_ptr() % alignment or b.data_ptr() % alignment:
            raise ValueError("Invalid packed buffers for integer GEMV")


def int4_gemv(packed, q):
    """Return padded, contiguous BF16 [M,N] using GEMV or q's existing op.

    The native kernel supports one INT4 row, padded 128<=K<=8192 with K%128=0,
    and padded N%8=0. The caller retains the reference crop/reshape behavior.
    Other valid shapes and INT8 use the existing packed operator. Empty rows
    return the padded empty result without launching an operator.
    """
    q = projection_for_rows(q, packed.data.shape[0])
    _validate_pack(packed, q)
    rows = packed.data.shape[0]
    if rows == 0:
        return torch.empty(
            (0, q.packed_weight.shape[0]),
            device=packed.data.device,
            dtype=torch.bfloat16,
        )
    arguments = (
        packed.data,
        packed.scales,
        q.packed_weight,
        q.weight_scale,
        q.bias,
        q.bits,
    )
    if not (is_ada_device(packed.data.device) and eligible_shape(rows, q)):
        reference = q._ops.gemm_biasless if q.biasless_epilogue else q._ops.gemm
        return reference(*arguments, q.tactic)
    return load_gemv_extension().gemv(*arguments, q.biasless_epilogue)
