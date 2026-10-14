"""Native-order non-affine BF16 LayerNorm directly into integer activation packs.

The optional CUDA extension follows the PyTorch 2.7.1 vectorized Welford path.
Other layouts retain native ``layer_norm`` and the existing activation packer.
Load and warm the extension before CUDA Graph capture.
"""

import hashlib
import math
import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

from coexecution.kernels.ampere_ada._candidate_support import (
    is_ada_device,
    owned_build_identity,
    require_pre_capture_build,
)
from coexecution.kernels.ampere_ada.integer import PackedActivation
from coexecution.optimization.backends.integer.linear import prepare

_FAILED_BUILDS = set()


def _registered_extension(identity):
    extension = torch.ops.pir2_layernorm_pack
    if (
        not hasattr(extension, "prepare")
        or not hasattr(extension, "source_identity")
        or extension.source_identity() != identity
    ):
        raise RuntimeError(
            "Registered norm extension source identity differs; restart the process"
        )
    return extension


@lru_cache(maxsize=1)
def load_norm_extension():
    """Load verified repository-owned source before CUDA Graph capture."""
    source = Path(__file__).parent / "csrc/layernorm_pack.cu"
    identity = hashlib.sha256(
        source.read_bytes() + b"\0" + owned_build_identity()
    ).hexdigest()
    name = f"pir2_layernorm_pack_{identity}_ext"
    if name in _FAILED_BUILDS:
        raise RuntimeError(
            "Norm extension build previously failed for this source identity; "
            "restart the process after fixing the build"
        )
    if hasattr(torch.ops.pir2_layernorm_pack, "prepare"):
        return _registered_extension(identity)
    require_pre_capture_build()
    from torch.utils.cpp_extension import load

    previous_jobs = os.environ.get("MAX_JOBS")
    os.environ["MAX_JOBS"] = "2"
    source_define = f'-DPIR2_NORM_SOURCE_ID=\\"{identity}\\"'
    try:
        load(
            name=name,
            sources=[str(source)],
            extra_cflags=["-O3", "-std=c++17", source_define],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
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


def _validate_layernorm(x, bits, eps, scale, shift):
    if type(bits) is not int or bits not in (4, 8):
        raise ValueError("LayerNorm packing requires bits=4 or 8")
    if (
        not isinstance(eps, (int, float))
        or isinstance(eps, bool)
        or not math.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError("LayerNorm epsilon must be finite and positive")
    if not (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim >= 2
        and 0 < x.shape[-1] <= 65536
        and not x.requires_grad
    ):
        raise ValueError("LayerNorm packing requires CUDA BF16 rows without autograd")
    if (scale is None) != (shift is None):
        raise ValueError("LayerNorm modulation requires both scale and shift")
    if scale is None:
        return
    if x.ndim != 3:
        raise ValueError("LayerNorm modulation requires B,T,D input")
    allowed_shapes = ((x.shape[0], 1, x.shape[-1]), tuple(x.shape))
    if (
        not all(
            isinstance(value, torch.Tensor)
            and tuple(value.shape) in allowed_shapes
            and value.dtype == x.dtype
            and value.device == x.device
            and not value.requires_grad
            for value in (scale, shift)
        )
        or scale.shape != shift.shape
    ):
        raise ValueError(
            "LayerNorm scale and shift must match B,1,D or B,T,D, dtype and device"
        )


def prepare_layernorm(x, bits, eps, scale=None, shift=None):
    """Return a reference-format pack after exact native BF16 LayerNorm.

    Aligned contiguous 2D/3D rows of width 128..8192, divisible by four, use
    the PyTorch 2.7.1 vectorized CUDA statistics order. Optional B,1,D or B,T,D
    modulation accepts strided chunks with unit feature stride, preserving
    separate BF16 normalization, factor, product and shift boundaries. Other
    valid layouts use native LayerNorm followed by reference packing.
    """
    _validate_layernorm(x, bits, eps, scale, shift)
    width = x.shape[-1]
    rows = x.numel() // width
    if rows == 0:
        pad = (width + 127) // 128 * 128
        data = torch.empty(
            (0, pad if bits == 8 else pad // 2),
            device=x.device,
            dtype=torch.int8 if bits == 8 else torch.uint8,
        )
        scales = torch.empty(0, device=x.device, dtype=torch.float32)
        return PackedActivation(data, scales, tuple(x.shape[:-1]), width, bits)
    native = (
        is_ada_device(x.device)
        and torch.__version__.split("+")[0] == "2.7.1"
        and x.ndim in (2, 3)
        and x.is_contiguous()
        and x.data_ptr() % 8 == 0
        and 128 <= width <= 8192
        and width % 4 == 0
        and (scale is None or (scale.stride(-1) == shift.stride(-1) == 1))
    )
    if native:
        data, scales = load_norm_extension().prepare(x, bits, float(eps), scale, shift)
        return PackedActivation(data, scales, tuple(x.shape[:-1]), width, bits)
    normalized = F.layer_norm(x, (width,), eps=eps)
    if scale is not None:
        normalized = normalized * (1 + scale) + shift
    return prepare(normalized, bits)
