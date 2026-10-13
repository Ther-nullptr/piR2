"""Producer packing that preserves native norm statistics and BF16 boundaries."""

import math

import torch
import triton
import triton.language as tl
from coexecution.kernels.ampere_ada.integer import PackedActivation
from coexecution.kernels.ampere_ada.modulation import (
    PackedActivations,
    prepare_modulation,
)
from triton.language.extra.cuda import libdevice

from coexecution.fused_ops import _launch
from coexecution.kernels.ampere_ada.rms_statistics import mean_square


@triton.jit
def _rms_tail_pack(
    X,
    VARIANCE,
    WEIGHT,
    Q,
    SCALE,
    K: tl.constexpr,
    PAD: tl.constexpr,
    EPS: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    index = tl.arange(0, BLOCK)
    valid = index < K
    x = tl.load(X + row * K + index, valid, 0).to(tl.float32)
    variance = tl.load(VARIANCE + row)
    inv = libdevice.rsqrt(variance + EPS)
    normalized = (x * inv).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(WEIGHT + index, valid, 0).to(tl.float32)
    value = (normalized * weight).to(tl.bfloat16).to(tl.float32)
    limit: tl.constexpr = (1 << (BITS - 1)) - 1
    maximum = tl.max(tl.abs(value), 0)
    scale = tl.where(maximum > 0, maximum * (1.0 / limit), 1.0)
    tl.store(SCALE + row, scale)
    quant = libdevice.nearbyint(tl.div_rn(value, scale))
    quant = tl.minimum(tl.maximum(quant, -limit), limit).to(tl.int32)
    if BITS == 8:
        tl.store(Q + row * PAD + index, quant.to(tl.int8), index < PAD)
    else:
        pairs = tl.reshape(quant & 15, (BLOCK // 2, 2))
        packed = tl.sum(pairs << (tl.arange(0, 2)[None, :] * 4), 1).to(tl.uint8)
        offsets = tl.arange(0, BLOCK // 2)
        tl.store(Q + row * (PAD // 2) + offsets, packed, offsets < PAD // 2)


def prepare_rmsnorm(x, weight, eps, bits):
    """Use exact FP32 RMS statistics and retain the existing integer pack tail."""
    if not (
        type(bits) is int
        and bits in (4, 8)
        and x.is_cuda
        and x.device == weight.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.ndim >= 2
        and x.is_contiguous()
        and weight.is_contiguous()
        and not x.requires_grad
        and not weight.requires_grad
        and 0 < x.shape[-1] <= 8192
        and weight.numel() == x.shape[-1]
        and math.isfinite(eps)
        and eps > 0
    ):
        raise ValueError("RMS producer requires contiguous CUDA BF16 rows and weights")
    variance = mean_square(x)
    k = x.shape[-1]
    rows, pad = x.numel() // k, triton.cdiv(k, 128) * 128
    data = torch.empty(
        (rows, pad if bits == 8 else pad // 2),
        device=x.device,
        dtype=torch.int8 if bits == 8 else torch.uint8,
    )
    scales = torch.empty(rows, device=x.device, dtype=torch.float32)
    _launch(
        _rms_tail_pack,
        (rows,),
        (x, variance, weight, data, scales),
        (k, pad, eps, bits, triton.next_power_of_2(pad)),
    )
    return PackedActivation(data, scales, tuple(x.shape[:-1]), k, bits)


def prepare_final_modulation(x, scale, shift, bits):
    """Input is already normalized; final DiT's shift/scale order stays upstream."""
    if x.ndim != 3 or scale.shape != shift.shape:
        raise ValueError("Final modulation requires B,T,D and matched scale/shift")
    if scale.ndim == 2:
        if scale.shape != (x.shape[0], x.shape[-1]):
            raise ValueError("Shared condition shape does not match final modulation")
        packed = prepare_modulation(x, scale[:, None], shift[:, None], bits)
    elif scale.ndim == 3 and scale.shape == (x.shape[0], 1, x.shape[-1]):
        packed = prepare_modulation(x, scale, shift, bits)
    elif scale.ndim == 3 and scale.shape == x.shape:
        width = x.shape[-1]
        packed = prepare_modulation(
            x.reshape(-1, 1, width),
            scale.reshape(-1, 1, width),
            shift.reshape(-1, 1, width),
            bits,
        )
    else:
        raise ValueError("Unsupported final conditioning shape")
    result = PackedActivations({bits: packed}).with_leading_shape(x.shape[:-1])
    result.shape = x.shape
    return result
