"""Thor floating packing and BF16 producers for the pinned CUTLASS layouts.

FP8 needs a global amax and therefore two passes. FP4 normalizes each block of
16 values and writes the reference's swizzled UE4M3 scales in the producer.
"""

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class FloatingPacked:
    data: torch.Tensor
    scale: torch.Tensor
    shape: tuple
    precision: str


@triton.jit
def _store_fp4_row(value, row, Packed, Scales, K: tl.constexpr, BLOCK: tl.constexpr):
    # Same block-16 UE4M3 and scale addressing as the pinned FP4 backend.
    groups = tl.reshape(value, (BLOCK // 16, 16))
    maximum = tl.max(tl.abs(groups), 1)
    scale = tl.where(maximum > 0, tl.div_rn(maximum, 6.0), 1.0)
    encoded = tl.minimum(scale, 448.0).to(tl.float8e4nv)
    decoded = encoded.to(tl.float32)
    decoded = tl.where(decoded > 0, decoded, scale)
    values = groups * tl.div_rn(1.0, decoded)[:, None]
    values = tl.reshape(values, (BLOCK // 2, 2))
    lo, hi = tl.split(values)
    pairs = tl.inline_asm_elementwise(
        "{ .reg .b8 pair; cvt.rn.satfinite.e2m1x2.f32 pair, $2, $1; cvt.u32.u8 $0, pair; }",
        constraints="=r,f,f",
        args=[lo, hi],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)
    col = tl.arange(0, BLOCK // 2)
    tl.store(Packed + row * (K // 2) + col, pairs, col < K // 2)
    sf = tl.arange(0, BLOCK // 16)
    offset = (
        (row // 128) * tl.cdiv(K // 16, 4) * 512
        + (row % 32) * 16
        + (row % 128 // 32) * 4
        + (sf // 4) * 512
        + sf % 4
    )
    tl.store(Scales + offset, encoded.to(tl.uint8, bitcast=True), sf < K // 16)


@triton.jit
def _store_producer(
    value,
    row,
    Output,
    Auxiliary,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    FP4: tl.constexpr,
):
    value = value.to(tl.bfloat16).to(tl.float32)
    if FP4:
        _store_fp4_row(value, row, Output, Auxiliary, K, BLOCK)
    else:
        col = tl.arange(0, BLOCK)
        tl.store(Output + row * K + col, value, col < K)
        absolute = tl.abs(value)
        finite = (col < K) & (absolute < float("inf"))
        tl.store(Auxiliary + row, tl.max(tl.where(finite, absolute, 0.0), 0))


@triton.jit
def _swiglu_pack(
    Gate,
    Up,
    Lut,
    Output,
    Auxiliary,
    K: tl.constexpr,
    GS: tl.constexpr,
    US: tl.constexpr,
    BLOCK: tl.constexpr,
    FP4: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    g = tl.load(Gate + row * GS + col, col < K, 0)
    up = tl.load(Up + row * US + col, col < K, 0).to(tl.float32)
    # Preserve torch SiLU's BF16 output boundary before the BF16 multiply.
    activated = tl.load(Lut + g.to(tl.uint16, bitcast=True).to(tl.int32))
    value = (activated.to(tl.float32) * up).to(tl.bfloat16)
    _store_producer(value, row, Output, Auxiliary, K, BLOCK, FP4)


@triton.jit
def _norm_modulation_pack(
    X,
    Scale,
    Shift,
    Y,
    Residual,
    Output,
    Auxiliary,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SS: tl.constexpr,
    HS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    MODULATE: tl.constexpr,
    ADD: tl.constexpr,
    FP4: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < K
    x = tl.load(X + row * K + col, valid, 0)
    if ADD:
        other = tl.load(Y + row * K + col, valid, 0).to(tl.float32)
        x = (x.to(tl.float32) + other).to(tl.bfloat16)
        tl.store(Residual + row * K + col, x, valid)
    value = x.to(tl.float32)
    mean = tl.sum(value, 0) / K
    centered = tl.where(valid, value - mean, 0.0)
    variance = tl.sum(centered * centered, 0) / K
    # Native LayerNorm's reduction order differs; validate error explicitly.
    value = (centered * tl.rsqrt(variance + EPS)).to(tl.bfloat16).to(tl.float32)
    if MODULATE:
        s = tl.load(Scale + (row // TOKENS) * SS + col, valid, 0).to(tl.float32)
        h = tl.load(Shift + (row // TOKENS) * HS + col, valid, 0).to(tl.float32)
        factor = (1.0 + s).to(tl.bfloat16).to(tl.float32)
        value = (value * factor).to(tl.bfloat16).to(tl.float32)
        value = (value + h).to(tl.bfloat16)
    _store_producer(value, row, Output, Auxiliary, K, BLOCK, FP4)


def _producer_buffers(x, precision):
    if (
        precision not in ("fp8", "fp4")
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.numel() == 0
        or x.requires_grad
        or x.shape[-1] % 32
        or x.shape[-1] > 65536
    ):
        raise ValueError(
            "Floating producer requires nonempty aligned CUDA BF16 inference"
        )
    rows, k = x.numel() // x.shape[-1], x.shape[-1]
    if precision == "fp4":
        output = torch.empty((rows, k // 2), device=x.device, dtype=torch.uint8)
        auxiliary = torch.empty(
            (triton.cdiv(rows, 128) * 128, triton.cdiv(k // 16, 4) * 4),
            device=x.device,
            dtype=torch.uint8,
        )
    else:
        output = torch.empty((rows, k), device=x.device, dtype=x.dtype)
        auxiliary = torch.empty(rows, device=x.device, dtype=torch.float32)
    return output, auxiliary, rows, k


def _finish_producer(output, auxiliary, shape, precision):
    if precision == "fp4":
        return FloatingPacked(output, auxiliary, tuple(shape), precision)
    rows = output.shape[0]
    packed = torch.empty_like(output, dtype=torch.float8_e4m3fn)
    scale = torch.empty((), device=output.device, dtype=torch.float32)
    _fp8_scale_and_cast[(triton.cdiv(output.numel(), 4096),)](
        output,
        auxiliary,
        packed,
        scale,
        output.numel(),
        rows,
        triton.next_power_of_2(rows),
        4096,
        num_warps=4,
    )
    return FloatingPacked(packed, scale, tuple(shape), precision)


def prepare_swiglu(gate, up, lut, precision):
    """Fuse native BF16 SiLU lookup, multiply and floating packing."""
    shape = tuple(gate.shape)
    output, auxiliary, rows, k = _producer_buffers(gate, precision)
    if up.shape != gate.shape or up.device != gate.device or up.dtype != gate.dtype:
        raise ValueError("SwiGLU operands must have identical shape, device and dtype")
    if lut.shape != (65536,) or lut.device != gate.device or lut.dtype != gate.dtype:
        raise ValueError("SwiGLU requires the native BF16 SiLU lookup table")
    # Merged gate/up views have a contiguous last dimension but strided rows.
    gate, up = gate.reshape(-1, k), up.reshape(-1, k)
    if gate.stride(1) != 1 or up.stride(1) != 1:
        raise ValueError("SwiGLU producer requires contiguous columns")
    _swiglu_pack[(rows,)](
        gate,
        up,
        lut,
        output,
        auxiliary,
        k,
        gate.stride(0),
        up.stride(0),
        triton.next_power_of_2(k),
        precision == "fp4",
        num_warps=4 if k <= 2048 else 8,
        enable_fp_fusion=False,
    )
    return _finish_producer(output, auxiliary, shape, precision)


def prepare_norm(x, precision, *, eps, scale=None, shift=None, residual=None):
    """LayerNorm [+ AdaLN] [+ residual] directly to floating packed input.

    BF16 intermediate boundaries are preserved, but the reduction differs from
    native LayerNorm. The returned residual is the exact BF16 x + residual.
    """
    output, auxiliary, rows, k = _producer_buffers(x, precision)
    if x.ndim != 3 or not x.is_contiguous() or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Norm packing requires contiguous B,T,D and positive epsilon")
    tokens, ss, hs = 1, 0, 0
    if (scale is None) != (shift is None):
        raise ValueError("Modulation requires both scale and shift")
    if scale is not None:
        for value in (scale, shift):
            if (
                value.dtype != x.dtype
                or value.device != x.device
                or value.shape not in (x.shape, (x.shape[0], k))
                or value.stride(-1) != 1
            ):
                raise ValueError("Modulation must match BF16 B,D or per-token B,T,D")
        if scale.shape != shift.shape:
            raise ValueError("Modulation shapes must match")
        tokens = x.shape[1] if scale.ndim == 2 else 1
        scale, shift = scale.reshape(-1, k), shift.reshape(-1, k)
        ss, hs = scale.stride(0), shift.stride(0)
    summed = None
    if residual is not None:
        if (
            residual.shape != x.shape
            or residual.dtype != x.dtype
            or residual.device != x.device
            or not residual.is_contiguous()
        ):
            raise ValueError("Residual must match contiguous BF16 input")
        summed = torch.empty_like(x)
    _norm_modulation_pack[(rows,)](
        x,
        scale,
        shift,
        residual,
        summed,
        output,
        auxiliary,
        k,
        tokens,
        ss,
        hs,
        float(eps),
        triton.next_power_of_2(k),
        scale is not None,
        residual is not None,
        precision == "fp4",
        num_warps=4 if k <= 2048 else 8,
        enable_fp_fusion=False,
    )
    return _finish_producer(output, auxiliary, x.shape, precision), summed


@triton.jit
def _fp8_partial_amax(X, Partial, N: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    values = tl.abs(tl.load(X + offsets, offsets < N, other=0).to(tl.float32))
    # Match the reference amax's treatment of nonfinite values.
    values = tl.where(values < float("inf"), values, 0.0)
    tl.store(Partial + block, tl.max(values, 0))


@triton.jit
def _fp8_scale_and_cast(
    X,
    Partial,
    Packed,
    Scale,
    N: tl.constexpr,
    PARTS: tl.constexpr,
    REDUCE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    indices = tl.arange(0, REDUCE)
    maxima = tl.load(Partial + indices, indices < PARTS, other=0.0)
    scale = tl.maximum(tl.max(maxima, 0) * (1.0 / 448.0), 1.0e-12)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    values = values / scale
    values = tl.minimum(448.0, tl.maximum(-448.0, values))
    tl.store(Packed + offsets, values.to(tl.float8e4nv), offsets < N)
    if block == 0:
        tl.store(Scale, scale)


def prepare_fp8(matrix):
    """Return fresh E4M3 data and one FP32 scale; safe for CUDA Graph replay.

    The second pass reduces the small partial-max array independently in each CTA,
    then combines scale finalization and conversion. No global atomic or memset is
    needed. Both passes read the current input; no activation-dependent state is
    cached between calls.
    """
    if (
        not matrix.is_cuda
        or matrix.dtype != torch.bfloat16
        or not matrix.is_contiguous()
        or matrix.ndim != 2
        or matrix.numel() == 0
    ):
        raise ValueError("FP8 packing requires a nonempty contiguous CUDA BF16 matrix")
    count = matrix.numel()
    block = 4096
    parts = triton.cdiv(count, block)
    partial = torch.empty(parts, device=matrix.device, dtype=torch.float32)
    packed = torch.empty_like(matrix, dtype=torch.float8_e4m3fn)
    scale = torch.empty((), device=matrix.device, dtype=torch.float32)
    _fp8_partial_amax[(parts,)](matrix, partial, count, block, num_warps=4)
    _fp8_scale_and_cast[(parts,)](
        matrix,
        partial,
        packed,
        scale,
        count,
        parts,
        triton.next_power_of_2(parts),
        block,
        num_warps=4,
    )
    return packed, scale
