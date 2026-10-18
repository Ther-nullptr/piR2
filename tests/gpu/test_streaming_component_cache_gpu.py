"""Opt-in partial-forward cache recovery on the real CUDA streaming action head.

Run with PIR2_GPU_TESTS=1 and the pinned GR00T/kernel dependencies. Synthetic
features bypass the VLM; these contracts do not measure LIBERO task quality.
"""

import os

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip("Set PIR2_GPU_TESTS=1 for explicit CUDA tests", allow_module_level=True)

import torch
from gr00t.model.modules.dit import SelfAttentionTransformer

from coexecution.optimization.config import OptimizationConfig
from coexecution.optimization.session import GrootOptimizations
from tests.helpers.groot_streaming import assert_equal, rollout


@pytest.mark.parametrize("precision", ["w8a8", "w4a4"])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("component_cache_clear", [False, True])
@torch.inference_mode()
def test_dit_component_recovers_partial_group_without_exiting_scope(
    policy_and_inputs, precision, graph, component_cache_clear
):
    policy, inputs = policy_and_inputs
    head = policy.model.action_head
    # Preserve the complete policy's root inventory while bypassing the VLM.
    policy.model.backbone = torch.nn.Identity().eval()
    # The shared fixture defaults this stage to Identity. Exercise the actual
    # S1 VL-transformer inventory without changing any existing fixture users.
    torch.manual_seed(217)
    with torch.device("cuda"):
        head.vl_self_attention = SelfAttentionTransformer(
            num_attention_heads=4,
            attention_head_dim=32,
            num_layers=1,
            dropout=0.0,
            final_dropout=False,
            positional_embeddings=None,
        )
    head.vl_self_attention.bfloat16().eval().requires_grad_(False)
    common = {"precision": precision, "component": "dit"}
    with GrootOptimizations(policy, OptimizationConfig(**common)):
        reference = rollout(head, inputs)
    with GrootOptimizations(
        policy,
        OptimizationConfig(
            **common, dit_graph=graph, component_cache_clear=component_cache_clear
        ),
    ) as scope:
        selected = scope.coverage["linears"]
        assert all(name.startswith("action_head.") for name in selected)
        assert any(
            name.startswith("action_head.vl_self_attention.") for name in selected
        )
        cache_policy = scope.evidence()["cache_clear_policy"]
        if component_cache_clear:
            assert cache_policy["mode"] == "active_precision_per_root"
            assert cache_policy["precision"] == precision
            assert cache_policy["groups"]["vlm"] == 0
            assert cache_policy["groups"]["s1_feature"] > 0
            assert cache_policy["groups"]["dit"] > 0
        else:
            assert cache_policy == {"mode": "global"}

        def fail_after_partial_group(_module, _inputs):
            # Q/K have run, but V has not consumed its grouped cache. For Graph,
            # this fails the first capture/warmup and restores eager execution.
            raise RuntimeError("deliberate incomplete projection group")

        projection = head.model.transformer_blocks[0].attn1.to_v
        hook = projection.register_forward_pre_hook(fail_after_partial_group)
        try:
            with pytest.raises(RuntimeError, match="incomplete projection group"):
                rollout(head, inputs)
        finally:
            hook.remove()
        # The same live scope must release references, reset and consume fresh
        # inputs. Its root boundary must also survive Graph's eager restoration.
        assert_equal(rollout(head, inputs), reference)
        assert_equal(rollout(head, inputs), reference)
    assert "forward" not in head.model.__dict__
    assert "reset_streaming_buffer" not in head.__dict__
