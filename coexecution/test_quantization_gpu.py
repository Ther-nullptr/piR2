"""Opt-in integer adapter contracts; no checkpoints or task-quality evaluation.

Run with PIR2_GPU_TESTS=1 and the pinned robotics_kernels source installed:
python -m pytest -q coexecution/test_quantization_gpu.py
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip("Set PIR2_GPU_TESTS=1 for explicit CUDA tests", allow_module_level=True)

import torch
from robotics_kernels.ampere_ada.integer import prepare_activation
from robotics_kernels.common.fused import prepare_gelu

from coexecution.quantization import TransformerINT, inventory, prepare
from coexecution.quantization_connections import AdditionalConnections, DitGraphs


def equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def policy_for(blocks):
    return SimpleNamespace(
        model=SimpleNamespace(
            action_head=SimpleNamespace(
                model=SimpleNamespace(transformer_blocks=blocks)
            )
        )
    )


class Norm(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = torch.nn.LayerNorm(
            width, elementwise_affine=False, device="cuda", dtype=torch.bfloat16
        )
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(
            width, 2 * width, device="cuda", dtype=torch.bfloat16
        )

    def forward(self, x, temb):
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=-1)
        if scale.ndim == 2:
            scale, shift = scale[:, None], shift[:, None]
        return self.norm(x) * (1 + scale) + shift


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("native", [False, True])
@torch.inference_mode()
def test_integer_linear_matches_quantized_reference_and_restores(bits, native):
    torch.manual_seed(195)
    linear = torch.nn.Linear(130, 64, device="cuda", dtype=torch.bfloat16).eval()
    model = torch.nn.Sequential(linear)
    x = torch.randn(17, 130, device="cuda", dtype=torch.bfloat16)
    original = model(x)
    limit = 2 ** (bits - 1) - 1
    xs = x.float().abs().amax(dim=-1) / limit
    ws = linear.weight.float().abs().amax(dim=-1) / limit
    qx = torch.round(x.float() / xs[:, None]).clamp(-limit, limit)
    qw = torch.round(linear.weight.float() / ws[:, None]).clamp(-limit, limit)
    expected = ((qx @ qw.t()) * xs[:, None] * ws[None, :] + linear.bias.float()).to(
        torch.bfloat16
    )
    adapter = TransformerINT(model, {"0"}, native=native, expanded=True)
    try:
        adapter.set(f"w{bits}a{bits}")
        equal(model(x), expected)
    finally:
        adapter.close()
    assert "forward" not in linear.__dict__
    equal(model(x), original)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("tokens", [1, 41])
@torch.inference_mode()
def test_condition_group_reuses_pack_and_gemm_and_releases_inputs(bits, tokens):
    dit = torch.nn.Module()
    blocks = [torch.nn.Module(), torch.nn.Module()]
    for block in blocks:
        block.norm1 = Norm(64)
    dit.transformer_blocks = torch.nn.ModuleList(blocks)
    dit.proj_out_1 = torch.nn.Linear(64, 128, device="cuda", dtype=torch.bfloat16)
    model = torch.nn.Module()
    model.action_head = torch.nn.Module()
    model.action_head.model = dit
    sites = inventory(model, expanded=True)[0]
    modules = [site[0] for site in sites.values()]
    condition = torch.randn(
        (1, 64) if tokens == 1 else (1, tokens, 64), device="cuda", dtype=torch.bfloat16
    )
    adapter = TransformerINT(
        model, sites, native=True, expanded=True, condition_group=True
    )
    try:
        for _ in range(2):
            expected = [adapter.quant[bits][name](condition) for name in sites]
            adapter.set(f"w{bits}a{bits}")
            with patch.object(adapter, "dispatch", wraps=adapter.dispatch) as dispatch:
                actual = [module(condition) for module in modules]
                assert dispatch.call_count == 1
            for value, reference in zip(actual, expected):
                equal(value, reference)
            group = adapter.grouped[bits][0][1]
            assert group._input is None and group._output is None
            condition.add_(0.25)
        # A forward interrupted before the other projections must release its cache.
        modules[0](condition)
        assert group._input is condition
    finally:
        adapter.close()
    assert group._input is None and group._output is None
    assert not group._remaining
    assert all("forward" not in module.__dict__ for module in modules)


@pytest.mark.parametrize("bits", [8, 4])
@torch.inference_mode()
def test_fused_activation_pack_matches_reference(bits):
    x = torch.randn(17, 258, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(x)
    lut = prepare_gelu(x.device, x.dtype, "tanh")
    actual = prepare(x, bits, lut, up)
    expected = prepare_activation(x, bits, up=up, gelu="tanh", pack_reuse=True)
    equal(actual.data, expected.data)
    equal(actual.scales, expected.scales)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("per_token", [False, True])
@torch.inference_mode()
def test_modulation_pack_preserves_shared_and_per_token_inputs(bits, per_token):
    norm = Norm(64)
    x = torch.randn(2, 3, 64, device="cuda", dtype=torch.bfloat16)
    temb = torch.randn(
        (2, 3, 64) if per_token else (2, 64), device="cuda", dtype=torch.bfloat16
    )
    expected = prepare_activation(norm(x, temb), bits, pack_reuse=True)
    adapter = AdditionalConnections(policy_for([SimpleNamespace(norm1=norm)]), None, {})
    try:
        adapter.set(f"w{bits}a{bits}")
        actual = norm(x, temb).for_bits(bits)
        equal(actual.data, expected.data)
        equal(actual.scales, expected.scales)
        assert actual.shape == expected.shape
    finally:
        adapter.close()
    assert "forward" not in norm.__dict__


@pytest.mark.parametrize(
    "cross,selected,packed",
    [
        (False, ("to_q",), False),
        (False, ("to_q", "to_k", "to_v"), True),
        (True, ("to_q",), True),
        (True, ("to_k", "to_v"), False),
    ],
)
@torch.inference_mode()
def test_partial_coverage_only_packs_fully_quantized_consumers(cross, selected, packed):
    norm = Norm(64)
    attention = SimpleNamespace(
        is_cross_attention=cross, to_q=object(), to_k=object(), to_v=object()
    )
    block = SimpleNamespace(norm1=norm, attn1=attention)
    controller = SimpleNamespace(
        sites={name: (getattr(attention, name), "role", "dit") for name in selected}
    )
    adapter = AdditionalConnections(policy_for([block]), controller, {})
    x = torch.randn(1, 3, 64, device="cuda", dtype=torch.bfloat16)
    temb = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
    expected = norm(x, temb)
    try:
        adapter.set("w8a8")
        actual = norm(x, temb)
        assert getattr(actual, "_robotics_packed_input", False) is packed
        if not packed:
            equal(actual, expected)
    finally:
        adapter.close()


class Category(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(
            torch.randn(3, 128, 64, device="cuda", dtype=torch.bfloat16) * 0.02
        )
        self.b = torch.nn.Parameter(
            torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
        )

    def forward(self, x, ids):
        return torch.bmm(x, self.W[ids]) + self.b[ids, None, :]


@pytest.mark.parametrize("bits", [8, 4])
@torch.inference_mode()
def test_fixed_category_guard_and_restoration(bits):
    category = Category()
    adapter = AdditionalConnections(
        policy_for([]),
        SimpleNamespace(dispatch=lambda q, x: q(x)),
        {"category": (category, 2)},
    )
    x = torch.randn(1, 3, 128, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([2], device="cuda", dtype=torch.long)
    expected = adapter.quant[bits]["category"](x)
    try:
        adapter.set(f"w{bits}a{bits}", fuse=False)
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("CPU sync")):
            actual = category(x, ids)
        equal(actual, expected)
        # Reject a CPU ID so this negative test cannot poison the CUDA context.
        with pytest.raises(RuntimeError, match="category 2"):
            category(x, torch.tensor([1]))
        with pytest.raises(ValueError, match="B1"):
            category(x.expand(2, -1, -1), torch.tensor([2, 2]))
    finally:
        adapter.close()
    assert "forward" not in category.__dict__


class GraphModel(torch.nn.Module):
    def forward(
        self,
        x,
        context,
        timestep=None,
        encoder_attention_mask=None,
        return_all_hidden_states=False,
        image_mask=None,
        backbone_attention_mask=None,
    ):
        output = x + context
        for value in (
            timestep,
            encoder_attention_mask,
            image_mask,
            backbone_attention_mask,
        ):
            if value is not None:
                output = output + value
        return (output, [x, output]) if return_all_hidden_states else output


@torch.inference_mode()
def test_graph_masks_updates_shapes_history_and_restore():
    model = GraphModel()
    eager = model.forward
    adapter = DitGraphs(model)
    adapter.set("test")
    try:
        for width in (16, 24):
            x = torch.randn(1, width, device="cuda")
            for kwargs in ({}, {"encoder_attention_mask": torch.ones_like(x)}):
                previous = model(x, x, **kwargs)
                snapshot = previous.clone()
                x.add_(1)
                equal(model(x, x, **kwargs), eager(x, x, **kwargs))
                equal(previous, snapshot)
                output, history = model(x, x, return_all_hidden_states=True, **kwargs)
                equal(output, eager(x, x, **kwargs))
                assert len(history) == 2
        adapter.set(None)
        assert "forward" not in model.__dict__
    finally:
        adapter.close()
    assert not adapter.graphs


def test_graph_capture_failure_restores_original_forward(monkeypatch):
    from coexecution import quantization_connections

    class FailedGraph:
        def __init__(self, function):
            pass

        def __call__(self, *inputs):
            raise RuntimeError("capture failed")

    monkeypatch.setattr(quantization_connections, "CudaGraphCall", FailedGraph)
    model = GraphModel()
    adapter = DitGraphs(model)
    adapter.set("test")
    with pytest.raises(RuntimeError, match="capture failed"):
        model(torch.ones(1), torch.ones(1))
    assert "forward" not in model.__dict__ and not adapter.graphs
