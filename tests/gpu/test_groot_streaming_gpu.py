"""Real action-head CUDA contracts, without a checkpoint or learned-quality claim.

Prepare the pinned, patched GR00T source and robotics-kernels dependencies, then:
PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_groot_streaming_gpu.py

The small AlternateVLDiT runs the upstream bootstrap and rolling-buffer code.
Synthetic features bypass the VLM and do not measure LIBERO task success.
"""

import os
from contextlib import closing, nullcontext

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip(
        "Set PIR2_GPU_TESTS=1 for streaming CUDA checks", allow_module_level=True
    )

import torch
from transformers.feature_extraction_utils import BatchFeature

from coexecution.optimization.config import OptimizationConfig
from coexecution.optimization.execution.graph import DitGraphs
from coexecution.optimization.session import GrootOptimizations
from tests.helpers.groot_streaming import assert_equal, assert_integer_trace, rollout


@pytest.mark.parametrize("precision", ["bf16", "w8a8", "w4a4"])
@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "dit_graph"])
@torch.inference_mode()
def test_real_streaming_head_fusion_state_reset_and_integer_kernels(
    policy_and_inputs, precision, use_graph, record_property
):
    policy, inputs = policy_and_inputs
    head = policy.model.action_head
    original = rollout(head, inputs)
    config = {"precision": precision, "scope": "all", "category_id": 0}
    with GrootOptimizations(policy, OptimizationConfig(**config)):
        reference = rollout(head, inputs)
    with GrootOptimizations(policy, OptimizationConfig(**config, fusion=True)):
        fused_forward = head.model.forward
        # Observe capture/replay counters while checking the graph lifecycle.
        graph_scope = closing(DitGraphs(head.model)) if use_graph else nullcontext()
        with graph_scope as graphs:
            if use_graph:
                graphs.set(precision)
            # Warm the real kernels/capture before recording their execution.
            actual = rollout(head, inputs)
            assert_equal(actual, reference)
            baseline_actions = torch.stack([state[0] for state in original[0]]).float()
            actual_actions = torch.stack([state[0] for state in actual[0]]).float()
            error = (actual_actions - baseline_actions).abs()
            record_property("precision", precision)
            record_property("dit_graph", use_graph)
            record_property("bf16_action_max_abs_error", error.max().item())
            record_property("bf16_action_mean_abs_error", error.mean().item())
            record_property("same_precision_fusion_max_abs_error", 0.0)
            if use_graph:
                assert sum(g.captures for g in graphs.graphs.values()) == 1
                assert sum(g.replays for g in graphs.graphs.values()) == 7
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as trace:
                replay = rollout(head, inputs)
            assert_equal(replay, reference)
            if precision != "bf16":
                # The eager action encoder/decoder still dispatch real integer GEMMs.
                assert_integer_trace(trace, 8 if precision == "w8a8" else 4)
            if use_graph:
                assert sum(g.captures for g in graphs.graphs.values()) == 1
                assert sum(g.replays for g in graphs.graphs.values()) == 14

            changed = list(inputs)
            backbone, action = inputs[2]
            changed[2] = (
                BatchFeature(
                    data={**backbone, "backbone_features": -backbone.backbone_features}
                ),
                action,
            )
            fresh = rollout(head, changed)
            # Same bootstrap and first call; the second call must consume fresh VL data.
            for index in (0, 1):
                torch.testing.assert_close(
                    fresh[0][index][0], actual[0][index][0], rtol=0, atol=0
                )
            assert not torch.equal(fresh[0][2][0], actual[0][2][0])
            assert_equal(rollout(head, inputs), reference)

            if use_graph:
                # S=13 -> 17 -> 17 -> 13; both mandatory masks retain real values.
                # AlternateVLDiT requires image/valid masks; its optional encoder
                # mask is ignored upstream, so do not invent unsupported mask modes.
                varied = list(inputs)
                for index in (1, 2):
                    backbone, action = inputs[index]
                    varied[index] = (
                        BatchFeature(
                            data={
                                key: torch.cat([value, value[:, :4]], dim=1)
                                for key, value in backbone.items()
                            }
                        ),
                        action,
                    )
                graphs.set(None)
                varied_reference = rollout(head, varied)
                graphs.set(precision)
                before = sum(g.captures for g in graphs.graphs.values())
                assert_equal(rollout(head, varied), varied_reference)
                assert sum(g.captures for g in graphs.graphs.values()) == before + 2
                assert len(graphs.graphs) == 1
                record_property("graph_capture_count", before + 2)
        assert head.model.forward is fused_forward
        if use_graph:
            assert not graphs.graphs
        assert_equal(rollout(head, inputs), reference)
        if use_graph:
            with (
                pytest.raises(RuntimeError, match="deliberate graph body failure"),
                closing(DitGraphs(head.model)) as failed_graphs,
            ):
                failed_graphs.set(precision)
                assert_equal(rollout(head, inputs), reference)
                raise RuntimeError("deliberate graph body failure")
            assert head.model.forward is fused_forward and not failed_graphs.graphs
            assert_equal(rollout(head, inputs), reference)
    # Closing restores upstream execution and does not leave a precision cache behind.
    assert_equal(rollout(head, inputs), original)
    with GrootOptimizations(
        policy, OptimizationConfig(**config, fusion=True, dit_graph=use_graph)
    ):
        assert_equal(rollout(head, inputs), reference)
    assert_equal(rollout(head, inputs), original)
