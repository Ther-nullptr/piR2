"""Explicit GPU checks: PIR2_GPU_TESTS=1 pytest coexecution/test_groot_fusion_gpu.py.

Synthetic adapter checks do not establish GR00T policy or closed-loop quality.
"""

import os
from collections import Counter
from types import SimpleNamespace

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip("Set PIR2_GPU_TESTS=1 for CUDA fusion checks", allow_module_level=True)

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from coexecution.groot_fusion import NonGemmAdapters, modulation, same
from coexecution.groot_pointwise import PointwiseFusion, PreparedAttention


def pointwise_fixture():
    fusion = PointwiseFusion.__new__(PointwiseFusion)
    fusion.blocks = [SimpleNamespace(attn1=SimpleNamespace(heads=3))]
    fusion.validation = True
    fusion.calls = Counter()
    values = torch.arange(65536, device="cuda", dtype=torch.int32)
    fusion.lut = F.silu(values.to(torch.int16).view(torch.bfloat16))
    return fusion


@pytest.mark.parametrize("per_token", [False, True])
@torch.inference_mode()
def test_condition_span_and_masks(per_token):
    fusion = pointwise_fixture()
    shape = (2, 17, 37) if per_token else (2, 37)
    temb = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    image = torch.rand(2, 13, device="cuda") > 0.5
    valid = torch.rand(2, 13, device="cuda") > 0.2
    activated, im, nm = fusion.prepare(temb, image, valid)
    assert same(activated, F.silu(temb))
    for actual, selected in ((im, image & valid), (nm, ~image & valid)):
        expected = torch.where(selected, 0.0, -float("inf")).to(temb.dtype)
        expected = expected[:, None, None, :].expand(2, 3, 1, 13)
        assert same(actual, expected)


class AdaLayerNorm(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(width, 2 * width)

    def forward(self, x, temb=None):
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=-1)
        if scale.ndim == 2:
            scale, shift = scale[:, None], shift[:, None]
        return self.norm(x) * (1 + scale) + shift


@pytest.mark.parametrize("backend", ["triton", "native"])
@pytest.mark.parametrize("per_token", [False, True])
@torch.inference_mode()
def test_adaln_condition_layouts_and_restoration(backend, per_token):
    module = AdaLayerNorm(1536).to(device="cuda", dtype=torch.bfloat16).eval()
    policy = SimpleNamespace(model=torch.nn.Sequential(module).eval())
    x = torch.randn(1, 41, 1536, device="cuda", dtype=torch.bfloat16)
    shape = x.shape if per_token else (1, 1536)
    condition = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    expected = module(x, condition)
    adapter = NonGemmAdapters(policy, backend)
    try:
        adapter.set("all")
        assert same(module(x, condition), expected)
    finally:
        adapter.close()
    assert "forward" not in module.__dict__
    assert same(module(x, condition), expected)


@torch.inference_mode()
def test_modulation_rejects_unsupported_layout():
    x = torch.randn(2, 3, 16, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Modulation requires"):
        modulation(x.transpose(0, 1), scale, scale)


@pytest.mark.parametrize("cross", [False, True])
@torch.inference_mode()
def test_prepared_attention_matches_sdpa(cross):
    attention = (
        Attention(query_dim=16, heads=2, dim_head=8)
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    attention.set_processor(AttnProcessor2_0())
    x = torch.randn(2, 5, 16, device="cuda", dtype=torch.bfloat16)
    context = (
        torch.randn(2, 7, 16, device="cuda", dtype=torch.bfloat16) if cross else None
    )
    length = context.shape[1] if cross else x.shape[1]
    mask = torch.zeros(2, 2, 1, length, device="cuda", dtype=torch.bfloat16)
    mask[..., -1] = -float("inf")
    # The stock processor expands heads from its flattened [B*H, Q, K] input;
    # PreparedAttention receives the already prepared [B, H, Q, K] mask.
    expected = attention(
        x, encoder_hidden_states=context, attention_mask=mask.flatten(0, 1)
    )
    actual = PreparedAttention()(attention, x, context, mask)
    assert same(actual, expected)


@torch.inference_mode()
def test_silu_gate():
    fusion = pointwise_fixture()
    gate = torch.randn(2, 17, 37, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    assert same(fusion.gate(gate, up), F.silu(gate) * up)


@pytest.mark.parametrize("per_token", [False, True])
@torch.inference_mode()
def test_final_modulation_and_hidden_states(per_token):
    fusion = pointwise_fixture()

    class Block:
        attn1 = SimpleNamespace(heads=3)

        def __call__(self, x, **kwargs):
            return x + 0.125

    model = SimpleNamespace(
        timestep_encoder=torch.nn.Identity(),
        transformer_blocks=[Block() for _ in range(3)],
        attend_text_every_n_blocks=2,
        proj_out_1=torch.nn.Linear(16, 32).to(device="cuda", dtype=torch.bfloat16),
        norm_out=torch.nn.LayerNorm(16).to(device="cuda", dtype=torch.bfloat16),
        proj_out_2=torch.nn.Linear(16, 3).to(device="cuda", dtype=torch.bfloat16),
    )
    x = torch.randn(2, 5, 16, device="cuda", dtype=torch.bfloat16)
    context = torch.randn(2, 7, 16, device="cuda", dtype=torch.bfloat16)
    condition = torch.randn(
        x.shape if per_token else (2, 16), device="cuda", dtype=torch.bfloat16
    )
    image = torch.rand(2, 7, device="cuda") > 0.5
    valid = torch.ones_like(image)
    actual, history = fusion.dit(
        model,
        x,
        context,
        timestep=condition,
        image_mask=image,
        backbone_attention_mask=valid,
        return_all_hidden_states=True,
    )
    expected_history = [x]
    for _ in range(3):
        expected_history.append(expected_history[-1] + 0.125)
    assert len(history) == 4
    assert all(same(a, b) for a, b in zip(history, expected_history))
    shift, scale = model.proj_out_1(F.silu(condition)).chunk(2, dim=-1)
    if not per_token:
        scale, shift = scale[:, None], shift[:, None]
    expected = model.proj_out_2(
        model.norm_out(expected_history[-1]) * (1 + scale) + shift
    )
    assert same(actual, expected)
