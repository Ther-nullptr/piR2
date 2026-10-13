"""Optional exact split-K for the pinned packed INT8/INT4 representation.

The extension uses the backend's packaged CUTLASS headers without modifying
upstream source. Integer partials are reduced before the reference FP32 scale
order and the single BF16 conversion. Load and warm it before CUDA capture.
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

_CUTLASS_REVISION = "982748aa7356fa838c2ea4994ddcb0b2a4b4cefa"
_FAILED_BUILDS = set()


def _registered_extension(identity):
    extension = torch.ops.pir2_integer_split_k
    if (
        not hasattr(extension, "gemm")
        or not hasattr(extension, "source_identity")
        or extension.source_identity() != identity
    ):
        raise RuntimeError(
            "Registered split-K extension source identity differs; restart the process"
        )
    return extension


@lru_cache(maxsize=1)
def load_split_k_extension():
    """Build a separate Ada extension only when an eligible call needs it."""
    source = Path(__file__).parent / "csrc/integer_split_k.cu"
    identity = hashlib.sha256(
        source.read_bytes() + b"\0" + owned_build_identity()
    ).hexdigest()
    name = f"pir2_integer_split_k_{identity}_ext"
    if name in _FAILED_BUILDS:
        raise RuntimeError(
            "Split-K extension build previously failed for this source identity; "
            "restart the process after fixing the build"
        )
    from coexecution.kernels.ampere_ada.build import build_inputs

    inputs = build_inputs()
    if inputs.get("cutlass_revision") != _CUTLASS_REVISION:
        raise RuntimeError("CUTLASS revision does not match the pinned snapshot")
    if hasattr(torch.ops.pir2_integer_split_k, "gemm"):
        return _registered_extension(identity)
    require_pre_capture_build()
    from torch.utils.cpp_extension import _get_build_directory, load

    directory = _get_build_directory(name, verbose=False)
    previous_jobs = os.environ.get("MAX_JOBS")
    os.environ["MAX_JOBS"] = "2"
    source_define = f'-DPIR2_SPLIT_K_SOURCE_ID=\\"{identity}\\"'
    try:
        load(
            name=name,
            sources=[str(source)],
            build_directory=directory,
            extra_include_paths=inputs["include_paths"],
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
        # Torch JIT can otherwise reuse an old .so after marking a failed build
        # as the current source version within this process.
        _FAILED_BUILDS.add(name)
        raise
    finally:
        if previous_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = previous_jobs


def eligible_shape(rows, q):
    """Check the physical GEMM shape before preparing native Tensor inputs."""
    weight = q.packed_weight
    if q.bits not in (4, 8) or weight.ndim != 2:
        return False
    n, stored_k = weight.shape
    k = stored_k * (2 if q.bits == 4 else 1)
    return (
        0 < rows <= 64
        and 4096 <= k <= 8192
        and k % 128 == 0
        and 0 < n <= 8192
        and n % 8 == 0
    )


def eligible(packed, q):
    """Check physical packed-input dimensions; execution also validates buffers."""
    return (
        packed.data.ndim == 2
        and is_ada_device(packed.data.device)
        and eligible_shape(packed.data.shape[0], q)
        and packed.data.shape[1] == q.packed_weight.shape[1]
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
    storage = torch.int8 if q.bits == 8 else torch.uint8
    tensors = (a, sa, b, sb) if q.bias is None else (a, sa, b, sb, q.bias)
    if not (
        q.bits in (4, 8)
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
            t.is_cuda and t.device == a.device and t.is_contiguous() for t in tensors
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
        raise ValueError("Invalid packed buffers for integer GEMM")


def split_k_gemm(packed, q, splits, tile=0, residual=None):
    """Return padded, contiguous BF16 [M,N] using split-K or the reference op.

    The caller retains the reference crop/reshape behavior. Shapes outside
    M<=64, padded 4096<=K<=8192 and padded N<=8192 use q's existing packed op.
    Zero rows return the padded empty result without launching a GEMM.
    Tile 0 keeps 64x128/warp32x64/K8=64/stages3; tile 1 uses
    16x64/warp16x32/K8=64/stages3; tile 2 uses
    64x64/warp32x32/K8=128/stages2. INT4 doubles K8 for the physical K tile.
    Optional residual preserves the original BF16 GEMM result before adding
    BF16 residual values in FP32 and rounding again. Native residual layout is
    contiguous aligned CUDA BF16 [M,Npad]; logical [M,Nlogical] or
    [*packed.shape,Nlogical] residuals are reshaped/zero-padded in the existing
    packed-GEMM fallback. The default residual=None path is unchanged.
    """
    if type(splits) is not int or splits not in (2, 4, 8):
        raise ValueError("splits must be one of 2, 4 or 8")
    if type(tile) is not int or tile not in (0, 1, 2):
        raise ValueError("tile must be one of 0, 1 or 2")
    q = projection_for_rows(q, packed.data.shape[0])
    _validate_pack(packed, q)
    if residual is not None:
        from coexecution.kernels.ampere_ada.epilogue import (
            _native_residual_layout,
            _reference_with_residual,
            _validate_residual,
        )
        from coexecution.kernels.ampere_ada.epilogue import (
            _validate_pack as validate_epilogue_pack,
        )

        validate_epilogue_pack(packed, q)
        _validate_residual(packed, q, residual)
    if packed.data.shape[0] == 0:
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
    if residual is not None:
        if not eligible(packed, q) or not _native_residual_layout(packed, q, residual):
            return _reference_with_residual(packed, q, residual)
        return load_split_k_extension().gemm(
            *arguments, splits, q.biasless_epilogue, tile, residual
        )
    if not eligible(packed, q):
        reference = q._ops.gemm_biasless if q.biasless_epilogue else q._ops.gemm
        return reference(*arguments, q.tactic)
    return load_split_k_extension().gemm(*arguments, splits, q.biasless_epilogue, tile)
