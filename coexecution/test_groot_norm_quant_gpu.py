"""Opt-in norm/residual packing contracts; no checkpoint or task-quality claim."""

import os
from contextlib import closing
from types import SimpleNamespace

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip("Set PIR2_GPU_TESTS=1 for explicit CUDA tests", allow_module_level=True)

import torch
import torch.nn.functional as F
from robotics_kernels.ampere_ada.modulation import prepare_modulation

from coexecution import quantization_connections as connections
from coexecution import test_groot_streaming_gpu as streaming
from coexecution.groot_optimization import GrootOptimizations, OptimizationConfig
from coexecution.test_groot_streaming_gpu import (
    assert_equal,
    assert_integer_trace,
    rollout,
)
from coexecution.test_quantization_gpu import Norm, policy_for


@pytest.fixture
def streaming_policy():
    return streaming.policy_and_inputs.__wrapped__()


def unpack(packed):
    data = packed.data.to(torch.int32)
    if packed.bits == 4:
        data = torch.stack((data & 15, (data >> 4) & 15), dim=-1).flatten(1)
        data = torch.where(data >= 8, data - 16, data)
    return data.float() * packed.scales[:, None]


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("per_token", [False, True])
@torch.inference_mode()
def test_norm_pack_matches_native_within_quantization_step(bits, per_token):
    torch.manual_seed(317)
    norm = Norm(130).eval()
    x = torch.randn(2, 3, 130, device="cuda", dtype=torch.bfloat16)
    x[0, 0].zero_()  # Constant row, including padding, must stay finite.
    temb = torch.randn(
        (2, 3, 130) if per_token else (2, 130), device=x.device, dtype=x.dtype
    )
    original = norm(x, temb)
    with closing(
        connections.AdditionalConnections(
            policy_for([SimpleNamespace(norm1=norm)]), None, {}
        )
    ) as extra:
        extra.set(f"w{bits}a{bits}")
        reference = norm(x, temb).for_bits(bits)
        extra.set(f"w{bits}a{bits}", norm_quant=True)
        actual = norm(x, temb).for_bits(bits)
        assert actual.shape == reference.shape == x.shape[:-1]
        error = (unpack(actual) - unpack(reference)).abs()
        bound = 2 * torch.maximum(actual.scales, reference.scales)[:, None]
        assert torch.isfinite(unpack(actual)).all()
        assert torch.all(error <= bound + 1e-6)
    torch.testing.assert_close(norm(x, temb), original, rtol=0, atol=0)
    assert "forward" not in norm.__dict__


def native_norm_pack(x, scale, shift, bits, *, eps):
    return prepare_modulation(
        F.layer_norm(x, (x.shape[-1],), eps=eps), scale, shift, bits
    )


def native_residual_pack(x, y, gate, scale, shift, bits, *, eps):
    residual = x + y * gate
    return residual, native_norm_pack(residual, scale, shift, bits, eps=eps)


@pytest.mark.parametrize("precision", ["w8a8", "w4a4"])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("graph", [False, True])
@torch.inference_mode()
def test_norm_fusion_streaming_semantics_and_cleanup(
    streaming_policy, precision, residual, graph, monkeypatch, record_property
):
    policy, inputs = streaming_policy
    head = policy.model.action_head
    config = {"precision": precision, "fusion": True, "scope": "all", "category_id": 0}
    original = rollout(head, inputs)
    with GrootOptimizations(policy, OptimizationConfig(**config)):
        reference = rollout(head, inputs)
    candidate = OptimizationConfig(
        **config,
        norm_modulation_quant=True,
        residual_norm_quant=residual,
        dit_graph=graph,
    )
    # Native reduction isolates control-flow/rounding mistakes from the explicitly
    # different Triton LayerNorm reduction. It must exactly match the old path.
    with monkeypatch.context() as patch:
        patch.setattr(connections, "prepare_norm_modulation", native_norm_pack)
        patch.setattr(
            connections, "prepare_residual_norm_modulation", native_residual_pack
        )
        with GrootOptimizations(policy, candidate):
            assert_equal(rollout(head, inputs), reference)
    with GrootOptimizations(policy, candidate):
        actual = rollout(head, inputs)
        assert_equal(rollout(head, inputs), actual)
        for (a, _, t), (r, _, rt) in zip(actual[0], reference[0]):
            torch.testing.assert_close(t, rt, rtol=0, atol=0)
            assert torch.isfinite(a).all()
        assert torch.equal(actual[1], reference[1])
        error = torch.stack(
            [
                (a[0].float() - b[0].float()).abs()
                for a, b in zip(actual[0], reference[0])
            ]
        )
        record_property("same_precision_action_max_abs_error", error.max().item())
        record_property("same_precision_action_mean_abs_error", error.mean().item())
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as trace:
            rollout(head, inputs)
        assert_integer_trace(trace, 8 if precision == "w8a8" else 4)
        kernels = [
            e.name
            for e in trace.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert any("_prepare_norm_modulation" in name for name in kernels)
        assert (
            any("_prepare_residual_norm_modulation" in name for name in kernels)
            == residual
        )
    assert_equal(rollout(head, inputs), original)
    with (
        pytest.raises(RuntimeError, match="deliberate"),
        GrootOptimizations(policy, candidate),
    ):
        rollout(head, inputs)
        raise RuntimeError("deliberate body failure")
    assert all("forward" not in b.__dict__ for b in head.model.transformer_blocks)
    assert_equal(rollout(head, inputs), original)
