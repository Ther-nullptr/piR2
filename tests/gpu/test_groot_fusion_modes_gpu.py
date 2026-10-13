"""Selective BF16 adapters must leave unrelated models and components intact."""

import os
from types import SimpleNamespace

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip("Set PIR2_GPU_TESTS=1 for CUDA fusion checks", allow_module_level=True)

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextRMSNorm,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
)

from coexecution.groot_fusion import NonGemmAdapters, same


class Qwen3VLVisionAttention(torch.nn.Module):
    def forward(self, q, k, cos, sin):
        return apply_rotary_pos_emb_vision(q, k, cos, sin)


class Qwen3VLTextAttention(torch.nn.Module):
    def forward(self, q, k, cos, sin):
        return apply_rotary_pos_emb(q, k, cos, sin)


def model():
    return (
        torch.nn.ModuleDict(
            {
                "vision": Qwen3VLVisionAttention(),
                "text": Qwen3VLTextAttention(),
                "rms": Qwen3VLTextRMSNorm(16),
            }
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )


@torch.inference_mode()
def test_selective_modes_and_instance_isolation():
    first, second = model(), model()
    vision = (
        torch.randn(17, 3, 16, device="cuda", dtype=torch.bfloat16),
        torch.randn(17, 3, 16, device="cuda", dtype=torch.bfloat16),
        torch.randn(17, 16, device="cuda"),
        torch.randn(17, 16, device="cuda"),
    )
    text = (
        torch.randn(1, 3, 17, 16, device="cuda", dtype=torch.bfloat16),
        torch.randn(1, 1, 17, 16, device="cuda", dtype=torch.bfloat16),
        torch.randn(1, 17, 16, device="cuda", dtype=torch.bfloat16),
        torch.randn(1, 17, 16, device="cuda", dtype=torch.bfloat16),
    )
    inputs = {"vision": vision, "text": text, "rms": (vision[0],)}
    expected = {key: first[key](*args) for key, args in inputs.items()}
    adapter = NonGemmAdapters(SimpleNamespace(model=first))
    adapter.validation = True
    selected = {
        "vision_rope": {"vision"},
        "text_rope": {"text"},
        "rope": {"vision", "text"},
        "rmsnorm": {"rms"},
        "rope_rms": {"vision", "text", "rms"},
        "baseline": set(),
    }
    try:
        for mode, changed in selected.items():
            adapter.set(mode)
            for key, args in inputs.items():
                assert ("forward" in first[key].__dict__) == (key in changed)
                assert "forward" not in second[key].__dict__
                assert same(first[key](*args), expected[key])
                assert same(second[key](*args), expected[key])
        adapter.set("vision_rope")
        forward = first["vision"].forward
        with pytest.raises(ValueError, match="Unknown non-GEMM fusion mode"):
            adapter.set("unsupported")
        assert first["vision"].forward is forward
    finally:
        adapter.close()
    assert all("forward" not in module.__dict__ for module in first.values())
    assert all("forward" not in module.__dict__ for module in second.values())
