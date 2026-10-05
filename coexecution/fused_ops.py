"""Inference-only BF16 fusion kernels that preserve reference rounding steps.

No stream changes, device synchronization, input mutation or hidden workspaces.
Triton launches on the current PyTorch stream. The adapters allocate outputs.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_LAUNCHERS = {}


def _launch(kernel, grid, tensors, metadata):
    """Reuse the compiled launcher for this dtype/layout/alignment specialization.

    Tensor values and addresses are never cached. Alignment is included because
    Triton's compiled launcher may assume the pointer divisibility seen at JIT.
    """
    grid = tuple(grid) + (1,) * (3 - len(grid))
    pointers = tuple((t.dtype, t.device, t.data_ptr() % 16) for t in tensors)
    key = (kernel, grid, metadata, pointers)
    launcher = _LAUNCHERS.get(key)
    if launcher is None:
        compiled = kernel[grid](
            *tensors, *metadata, num_warps=4, enable_fp_fusion=False
        )
        _LAUNCHERS[key] = compiled[grid]
    else:
        launcher(*tensors, *metadata)


@triton.jit
def _rope_kernel(
    Q,
    K,
    C,
    S,
    OQ,
    OK,
    B: tl.constexpr,
    T: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    QS,
    KS,
    OQS,
    OKS,
    CS,
    SS,
    ROUND_PRODUCTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    is_query = tl.program_id(1) == 0
    dimension = offsets % D
    partner = (dimension + D // 2) % D
    if is_query:
        head = offsets // D % HQ
        position = offsets // (D * HQ) % T
        batch = offsets // (D * HQ * T)
        valid = batch < B
        base = batch * QS[0] + position * QS[1] + head * QS[2]
        value = tl.load(Q + base + dimension * QS[3], valid, 0).to(tl.float32)
        rotated = tl.load(Q + base + partner * QS[3], valid, 0).to(tl.float32)
        output_offset = (
            batch * OQS[0] + position * OQS[1] + head * OQS[2] + dimension * OQS[3]
        )
    else:
        head = offsets // D % HK
        position = offsets // (D * HK) % T
        batch = offsets // (D * HK * T)
        valid = batch < B
        base = batch * KS[0] + position * KS[1] + head * KS[2]
        value = tl.load(K + base + dimension * KS[3], valid, 0).to(tl.float32)
        rotated = tl.load(K + base + partner * KS[3], valid, 0).to(tl.float32)
        output_offset = (
            batch * OKS[0] + position * OKS[1] + head * OKS[2] + dimension * OKS[3]
        )
    cosine = tl.load(
        C + batch * CS[0] + position * CS[1] + dimension * CS[2], valid, 0
    ).to(tl.float32)
    sine = tl.load(
        S + batch * SS[0] + position * SS[1] + dimension * SS[2], valid, 0
    ).to(tl.float32)
    rotated = tl.where(dimension < D // 2, -rotated, rotated)
    direct_product = value * cosine
    rotated_product = rotated * sine
    if ROUND_PRODUCTS:
        # Text RoPE performs two separate BF16 multiplies before BF16 addition.
        direct_product = direct_product.to(tl.bfloat16).to(tl.float32)
        rotated_product = rotated_product.to(tl.bfloat16).to(tl.float32)
    output = direct_product + rotated_product
    if is_query:
        tl.store(OQ + output_offset, output, valid)
    else:
        tl.store(OK + output_offset, output, valid)


def _check_tensors(q, k, cos, sin):
    assert q.is_cuda and q.device == k.device == cos.device == sin.device
    assert q.dtype == k.dtype == torch.bfloat16
    assert cos.shape == sin.shape and q.shape[-1] == k.shape[-1] == cos.shape[-1]
    assert q.shape[-1] % 2 == 0


def vision_rope(q, k, cos, sin):
    """[tokens, heads, dim] Q/K, [tokens, dim] trig; supports packed QKV strides."""
    _check_tensors(q, k, cos, sin)
    assert q.ndim == k.ndim == 3 and cos.ndim == sin.ndim == 2
    assert q.shape == k.shape and q.shape[0] == cos.shape[0]
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    tokens, heads, dim = q.shape
    _launch(
        _rope_kernel,
        (triton.cdiv(q.numel(), 256), 2),
        (q, k, cos, sin, oq, ok),
        (
            1,
            tokens,
            heads,
            heads,
            dim,
            (0, *q.stride()),
            (0, *k.stride()),
            (0, *oq.stride()),
            (0, *ok.stride()),
            (0, *cos.stride()),
            (0, *sin.stride()),
            False,
            256,
        ),
    )
    return oq, ok


def text_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """[batch, heads, tokens, dim] Q/K, with grouped Q/K heads and BF16 trig."""
    _check_tensors(q, k, cos, sin)
    assert q.ndim == k.ndim == 4 and cos.ndim == sin.ndim == 3
    assert cos.dtype == sin.dtype == torch.bfloat16 and unsqueeze_dim == 1
    assert q.shape[0] == k.shape[0] == cos.shape[0]
    assert q.shape[2] == k.shape[2] == cos.shape[1]
    oq, ok = torch.empty_like(q), torch.empty_like(k)

    def bthd(tensor):
        strides = tensor.stride()
        return strides[0], strides[2], strides[1], strides[3]

    batch, heads, tokens, dim = q.shape
    _launch(
        _rope_kernel,
        (triton.cdiv(max(q.numel(), k.numel()), 256), 2),
        (q, k, cos, sin, oq, ok),
        (
            batch,
            tokens,
            heads,
            k.shape[1],
            dim,
            bthd(q),
            bthd(k),
            bthd(oq),
            bthd(ok),
            cos.stride(),
            sin.stride(),
            True,
            256,
        ),
    )
    return oq, ok


@triton.jit
def _square_fp32_kernel(X, SQUARE, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, 0).to(tl.float32)
    tl.store(SQUARE + offsets, values * values, offsets < N)


@triton.jit
def _rms_tail_kernel(
    X,
    VARIANCE,
    WEIGHT,
    OUTPUT,
    N: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, 0).to(tl.float32)
    variance = tl.load(VARIANCE + offsets // D, offsets < N, 0)
    inv = libdevice.rsqrt(variance + EPS)
    normalized = (values * inv).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(WEIGHT + offsets % D).to(tl.float32)
    tl.store(OUTPUT + offsets, weight * normalized, offsets < N)


def rmsnorm(x, weight, eps):
    """Fused cast/square and tail, retaining PyTorch's exact FP32 mean reduction."""
    assert x.is_cuda and x.device == weight.device
    assert x.dtype == weight.dtype == torch.bfloat16
    assert (
        x.is_contiguous() and weight.is_contiguous() and weight.numel() == x.shape[-1]
    )
    squared = torch.empty_like(x, dtype=torch.float32)
    _launch(
        _square_fp32_kernel,
        (triton.cdiv(x.numel(), 256),),
        (x, squared),
        (x.numel(), 256),
    )
    variance = squared.mean(-1, keepdim=True)
    output = torch.empty_like(x)
    _launch(
        _rms_tail_kernel,
        (triton.cdiv(x.numel(), 256),),
        (x, variance, weight, output),
        (x.numel(), x.shape[-1], eps, 256),
    )
    return output
