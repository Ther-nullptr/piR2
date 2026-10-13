"""Optional INT4 CUTLASS tiles for the pinned packed small-M representation.

Only the reusable, SHA256-verified backend run template is materialized in
Torch's local build cache. The published wrapper owns buffer checks and tile
dispatch; upstream source and the reference integer registrations are untouched.
Load and warm the extension before CUDA Graph capture.
"""

import hashlib
import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import torch

from coexecution.kernels.ampere_ada._candidate_support import (
    is_ada_device,
    owned_build_identity,
    projection_for_rows,
    require_pre_capture_build,
)

_BACKEND_SHA256 = "78689a96089653d3cf2f9fabb1fa64eeb5bb7ee835b7d3316297c7ef884a431f"
_DISPATCH_MARKER = b"template <int Bits, bool WithBias>\nat::Tensor dispatch_tactic"
_CACHED_HEADER = "pir2_int4_small_m_backend.cuh"
_FAILED_BUILDS = set()


def _pinned_run_header(inputs):
    """Verify the run prefix and adapt only its swizzle and optional arguments."""
    try:
        source = Path(inputs["sources"][0]).read_bytes()
    except (OSError, KeyError, IndexError) as error:
        raise RuntimeError("Pinned integer GEMM source is unavailable") from error
    if hashlib.sha256(source).hexdigest() != _BACKEND_SHA256:
        raise RuntimeError("Integer GEMM source does not match the pinned SHA256")
    prefix, marker, _ = source.partition(_DISPATCH_MARKER)
    if not marker:
        raise RuntimeError("Pinned integer GEMM source has no run-template boundary")
    template_end = b"int WarpN, int Stages>"
    swizzle = b"cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>"
    argument_tail = b"0, 0, nullptr, nullptr, nullptr);"
    if (
        prefix.count(template_end) != 1
        or prefix.count(swizzle) != 1
        or prefix.count(argument_tail) != 1
    ):
        raise RuntimeError("Pinned integer GEMM source has no unique run boundaries")
    prefix = prefix.replace(swizzle, b"Swizzle", 1)
    prefix = prefix.replace(
        template_end,
        b"int WarpN, int Stages,\n          typename Swizzle = " + swizzle + b">",
        1,
    )
    # Identity Arguments defaults these gather/scatter pointers to nullptr;
    # Stream-K has no gather/scatter parameters and defaults avail_sms to -1.
    prefix = prefix.replace(argument_tail, b"0, 0);", 1)
    return b"#pragma once\n" + prefix + b"} // namespace\n"


def _cache_run_header(directory, content):
    """Replace the cached header atomically so concurrent builders see it whole."""
    header = directory / _CACHED_HEADER
    if header.exists() and header.read_bytes() == content:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as file:
            temporary = Path(file.name)
            file.write(content)
        os.replace(temporary, header)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _registered_extension(identity):
    extension = torch.ops.pir2_int4_small_m
    if (
        not hasattr(extension, "gemm")
        or not hasattr(extension, "source_identity")
        or extension.source_identity() != identity
    ):
        raise RuntimeError(
            "Registered small-M extension source identity differs; restart the process"
        )
    return extension


@lru_cache(maxsize=1)
def load_small_m_extension():
    """Load the separate Ada extension after verifying its pinned backend source."""
    from coexecution.kernels.ampere_ada.build import build_inputs

    inputs = build_inputs()
    header = _pinned_run_header(inputs)
    wrapper = Path(__file__).parent / "csrc/int4_small_m.cu"
    identity = hashlib.sha256(
        header + b"\0" + wrapper.read_bytes() + b"\0" + owned_build_identity()
    ).hexdigest()
    name = f"pir2_int4_small_m_{identity}_ext"
    if name in _FAILED_BUILDS:
        raise RuntimeError(
            "Small-M extension build previously failed for this source identity; "
            "restart the process after fixing the build"
        )
    if hasattr(torch.ops.pir2_int4_small_m, "gemm"):
        return _registered_extension(identity)
    require_pre_capture_build()
    from torch.utils.cpp_extension import _get_build_directory, load

    directory = Path(_get_build_directory(name, verbose=False))
    _cache_run_header(directory, header)
    previous_jobs = os.environ.get("MAX_JOBS")
    os.environ["MAX_JOBS"] = "2"
    source_define = f'-DPIR2_SMALL_M_SOURCE_ID=\\"{identity}\\"'
    try:
        load(
            name=name,
            sources=[str(wrapper)],
            build_directory=str(directory),
            extra_include_paths=[str(directory), *inputs["include_paths"]],
            extra_cflags=["-O3", "-std=c++17", source_define],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "--fmad=false",
                "--ftz=false",
                "-lineinfo",
                "-gencode=arch=compute_89,code=sm_89",
                source_define,
            ],
            is_python_module=False,
            verbose=False,
        )
        return _registered_extension(identity)
    except Exception:
        # Torch JIT records its source version before compilation succeeds.
        # Retrying that identity could skip the failed build and load an old .so.
        _FAILED_BUILDS.add(name)
        raise
    finally:
        if previous_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = previous_jobs


def eligible_shape(rows, q):
    """Check physical INT4 dimensions without preparing or reading CUDA buffers."""
    weight = q.packed_weight
    if q.bits != 4 or weight.ndim != 2:
        return False
    n, stored_k = weight.shape
    k = stored_k * 2
    return (
        0 < rows <= 64
        and 128 <= k <= 8192
        and k % 128 == 0
        and 0 < n <= 8192
        and n % 8 == 0
    )


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
        and q.out_features > 0
        and a.ndim == b.ndim == 2
        and a.shape[1] == b.shape[1]
        and a.dtype == b.dtype == storage
        and sa.dtype == sb.dtype == torch.float32
        and sa.ndim == sb.ndim == 1
        and sa.shape[0] == a.shape[0]
        and sb.shape[0] == b.shape[0]
        and b.shape[0] == (q.out_features + 7) // 8 * 8
        and a.shape[1] * (2 if q.bits == 4 else 1) == (q.in_features + 127) // 128 * 128
        and all(type(dimension) is int and dimension >= 0 for dimension in packed.shape)
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
        raise ValueError("Invalid packed buffers for small-M integer GEMM")
    # CUTLASS loads the packed operands and row broadcasts in 128-bit vectors.
    # Empty results consume none of these buffers.
    if a.shape[0]:
        aligned = (a, b, sb) if q.bias is None else (a, b, sb, q.bias)
        if any(tensor.data_ptr() % 16 for tensor in aligned):
            raise ValueError("Invalid packed buffers for small-M integer GEMM")


def small_m_gemm(packed, q, tactic):
    """Return padded contiguous BF16 [M,N] with tactic 0..5 or q's existing op.

    Tactics 0..3 use identity scheduling with tiles 16x64, 16x128, 32x64 and
    32x128. Tactics 4/5 use Stream-K scheduling with 64x64/16x64 tiles and
    INT32 partial reduction before the reference epilogue. All use K=128 and
    three stages. Native calls require INT4, 1<=M<=64, padded 128<=K<=8192 with
    K%128=0 and padded 8<=N<=8192 with N%8=0. Other valid shapes retain q's
    tactic and epilogue. Empty rows return without loading the extension.
    The caller retains the reference crop and reshape behavior.
    """
    if type(tactic) is not int or tactic not in (0, 1, 2, 3, 4, 5):
        raise ValueError("tactic must be one of 0, 1, 2, 3, 4 or 5")
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
        # The reference kernel uses the current input-device stream as well.
        # Record storage here because the reference dispatcher does not do so.
        stream = torch.cuda.current_stream(packed.data.device)
        tensors = arguments[:4] if q.bias is None else (*arguments[:4], q.bias)
        for tensor in tensors:
            tensor.record_stream(stream)
        reference = q._ops.gemm_biasless if q.biasless_epilogue else q._ops.gemm
        return reference(*arguments, q.tactic)
    return load_small_m_extension().gemm(*arguments, tactic, q.biasless_epilogue)
