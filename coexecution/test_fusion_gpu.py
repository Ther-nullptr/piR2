"""Actual GPU numerical/layout tests for lossless candidate kernels."""

import pytest
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextRMSNorm,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
)

from coexecution.fused_ops import rmsnorm, text_rope, vision_rope


@pytest.mark.parametrize("tokens", [512, 517])
@torch.inference_mode()
def test_vision_rope_qkv_strides_exact(tokens):
    assert torch.cuda.device_count() == 1
    torch.manual_seed(713)
    packed = torch.randn(tokens, 3, 16, 64, device="cuda", dtype=torch.bfloat16)
    q, k, _ = packed.unbind(1)
    # Original VLM casts cos/sin to FP32 inside vision RoPE.
    angles = torch.randn(tokens, 64, device="cuda", dtype=torch.bfloat16)
    cos, sin = angles.cos(), angles.sin()
    expected = apply_rotary_pos_emb_vision(q, k, cos, sin)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = vision_rope(q, k, cos, sin)
    torch.cuda.current_stream().wait_stream(stream)
    for reference, candidate in zip(expected, actual):
        torch.testing.assert_close(reference, candidate, rtol=0, atol=0)
        assert reference.stride() == candidate.stride()
    # Reusing a compiled launcher must still consume new values and allocations.
    q.add_(0.03125)
    for reference, candidate in zip(
        apply_rotary_pos_emb_vision(q, k, cos, sin), vision_rope(q, k, cos, sin)
    ):
        torch.testing.assert_close(reference, candidate, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [141, 143])
@torch.inference_mode()
def test_text_rope_grouped_heads_and_bf16_rounding(tokens):
    torch.manual_seed(714)
    q = torch.randn(1, tokens, 16, 128, device="cuda", dtype=torch.bfloat16).transpose(
        1, 2
    )
    k = torch.randn(1, tokens, 8, 128, device="cuda", dtype=torch.bfloat16).transpose(
        1, 2
    )
    angles = torch.randn(1, tokens, 128, device="cuda", dtype=torch.bfloat16)
    cos, sin = angles.cos(), angles.sin()
    expected = apply_rotary_pos_emb(q, k, cos, sin)
    actual = text_rope(q, k, cos, sin)
    for reference, candidate in zip(expected, actual):
        torch.testing.assert_close(reference, candidate, rtol=0, atol=0)
        assert reference.stride() == candidate.stride()
    q.mul_(1.25)
    for reference, candidate in zip(
        apply_rotary_pos_emb(q, k, cos, sin), text_rope(q, k, cos, sin)
    ):
        torch.testing.assert_close(reference, candidate, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 141, 2048), (1, 141, 16, 128), (1, 141, 8, 128)])
@torch.inference_mode()
def test_rmsnorm_keeps_reference_reduction_and_rounding(shape):
    torch.manual_seed(715)
    module = Qwen3VLTextRMSNorm(shape[-1]).to(device="cuda", dtype=torch.bfloat16)
    module.weight.copy_(torch.randn_like(module.weight))
    values = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    for scale in [0.0, 1e-4, 1.0, 100.0]:
        x = values * scale
        reference = module(x)
        candidate = rmsnorm(x, module.weight, module.variance_epsilon)
        torch.testing.assert_close(reference, candidate, rtol=0, atol=0)
        assert reference.stride() == candidate.stride()
