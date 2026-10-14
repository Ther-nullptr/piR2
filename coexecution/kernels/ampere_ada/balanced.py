"""Calibrated FP32 channel scaling fused into signed INT4 activation packing.

BF16 LUT and gated-product rounding precede balancing. The register route keeps
RN division and low-nibble-first bytes; it does not use the rejected reciprocal
division experiment. Quantization is lossy even when two packers agree bitwise.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from coexecution.kernels.ampere_ada.integer import PackedActivation, _activation_values

REGISTER_WIDTHS = (128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)


@triton.jit
def _pack(value, data, scales, row, PAD: tl.constexpr, BLOCK: tl.constexpr):
    maximum = tl.max(tl.abs(value), 0)
    scale = tl.where(maximum > 0, maximum * (1.0 / 7.0), 1.0)
    tl.store(scales + row, scale)
    q = libdevice.nearbyint(tl.div_rn(value, scale))
    q = tl.minimum(tl.maximum(q, -7), 7).to(tl.int32) & 15
    pairs = tl.reshape(q, (BLOCK // 2, 2))
    packed = tl.sum(pairs << (tl.arange(0, 2)[None, :] * 4), 1).to(tl.uint8)
    idx = tl.arange(0, BLOCK // 2)
    tl.store(data + row * (PAD // 2) + idx, packed, idx < PAD // 2)


@triton.jit
def _reference(
    X,
    UP,
    LUT,
    INV,
    DATA,
    SCALES,
    K: tl.constexpr,
    PAD: tl.constexpr,
    MODE: tl.constexpr,
    SX: tl.constexpr,
    SU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    value = _activation_values(X, UP, LUT, row, idx, K, MODE, SX, SU)
    inverse = tl.load(INV + idx, idx < K, 0)
    _pack(value * inverse, DATA, SCALES, row, PAD, BLOCK)


@triton.jit
def _modulation(
    X,
    SCALE,
    SHIFT,
    INV,
    DATA,
    SCALES,
    K: tl.constexpr,
    PAD: tl.constexpr,
    TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    valid = idx < K
    x = tl.load(X + row * K + idx, valid, 0)
    scale = tl.load(SCALE + (row // TOKENS) * K + idx, valid, 0).to(tl.float32)
    shift = tl.load(SHIFT + (row // TOKENS) * K + idx, valid, 0).to(tl.float32)
    factor = (1.0 + scale).to(x.dtype).to(tl.float32)
    product = (x.to(tl.float32) * factor).to(x.dtype).to(tl.float32)
    value = (product + shift).to(x.dtype).to(tl.float32)
    inverse = tl.load(INV + idx, valid, 0)
    _pack(value * inverse, DATA, SCALES, row, PAD, BLOCK)


def _check(x, inverse):
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim < 2
        or x.numel() == 0
        or not 1 <= x.shape[-1] <= 8192
        or x.stride(-1) != 1
    ):
        raise ValueError("Balanced INT4 expects nonempty CUDA BF16 rows, K=1..8192")
    if (
        inverse.device != x.device
        or inverse.dtype != torch.float32
        or inverse.shape != (x.shape[-1],)
        or not inverse.is_contiguous()
    ):
        raise ValueError("Balanced INT4 expects contiguous FP32 channel inverse")


def prepare(x, inverse, lut=None, up=None, *, route="register"):
    _check(x, inverse)
    if route not in ("register", "reference"):
        raise ValueError("Unknown balanced preparation route")
    k = x.shape[-1]
    rows = x.numel() // k
    pad = triton.cdiv(k, 128) * 128
    if lut is not None and (
        lut.device != x.device
        or lut.dtype != x.dtype
        or lut.numel() != 65536
        or not lut.is_contiguous()
    ):
        raise ValueError("Invalid BF16 activation LUT")
    if up is not None and (
        lut is None
        or up.shape != x.shape
        or up.device != x.device
        or up.dtype != x.dtype
        or up.stride(-1) != 1
    ):
        raise ValueError("Invalid gated activation")
    if route == "register":
        if pad not in REGISTER_WIDTHS:
            raise ValueError(
                "Unsupported register INT4 padded width; select reference explicitly"
            )
        from coexecution.kernels.build import build_extension

        ops = build_extension("int4_prepare")
        data, scales = ops.prepare(x, lut, up, inverse, 256)
    else:
        data = torch.empty((rows, pad // 2), dtype=torch.uint8, device=x.device)
        scales = torch.empty(rows, dtype=torch.float32, device=x.device)
        mode = 0 if lut is None else 2 if up is None else 1
        other = x if up is None else up
        with torch.cuda.device(x.device):
            _reference[(rows,)](
                x,
                other,
                x if lut is None else lut,
                inverse,
                data,
                scales,
                k,
                pad,
                mode,
                x.view(rows, k).stride(0),
                other.view(rows, k).stride(0),
                triton.next_power_of_2(pad),
                num_warps=4 if pad <= 2048 else 8,
                enable_fp_fusion=False,
            )
    return PackedActivation(data, scales, tuple(x.shape[:-1]), k, 4)


def linear(q, x, lut=None, up=None):
    if q.bits != 4 or x.shape[-1] != q.in_features:
        raise ValueError(
            "Balanced Linear requires INT4 and matching logical input width"
        )
    _check(x, q.inverse_balance)
    from coexecution.optimization.backends.integer.linear import selected_tactic

    if q.prepare_route == "reference":
        return q.forward_packed(
            prepare(x, q.inverse_balance, lut, up, route="reference")
        )
    # Installation builds this extension before warmup/capture; no source scan,
    # compile or load on the steady-state per-projection dispatch path.
    y = torch.ops.pir2_int4_prepare.linear(
        x,
        q.packed_weight,
        q.weight_scale,
        q.bias,
        lut,
        up,
        q.inverse_balance,
        256,
        selected_tactic(q, x.numel() // x.shape[-1]),
    )
    return y[:, : q.out_features].reshape(*x.shape[:-1], q.out_features)


def prepare_modulation(x, scale, shift, inverse):
    _check(x, inverse)
    if (
        x.ndim != 3
        or not x.is_contiguous()
        or scale.shape != shift.shape
        or scale.shape != (x.shape[0], 1, x.shape[-1])
    ):
        raise ValueError("Balanced modulation requires [B,T,K] and [B,1,K]")
    if any(v.device != x.device or v.dtype != x.dtype for v in (scale, shift)):
        raise ValueError("Modulation dtype/device differs from BF16 rows")
    b, t, k = x.shape
    pad = triton.cdiv(k, 128) * 128
    data = torch.empty((b * t, pad // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty(b * t, dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        _modulation[(b * t,)](
            x,
            scale.contiguous(),
            shift.contiguous(),
            inverse,
            data,
            scales,
            k,
            pad,
            t,
            triton.next_power_of_2(pad),
            num_warps=4 if pad <= 2048 else 8,
            enable_fp_fusion=False,
        )
    return PackedActivation(data, scales, (b, t), k, 4)
