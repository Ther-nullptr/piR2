"""Opt-in CUDA full-action/RNG/lifetime gates; never execute a model on CPU."""

import os

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip(
        "Set PIR2_GPU_TESTS=1 for whole action graph checks", allow_module_level=True
    )

import torch

from coexecution.optimization.config import OptimizationConfig
from coexecution.optimization.execution.cache import SerialForwardBoundary
from coexecution.optimization.execution.tensor_graph import TensorGraphCall
from coexecution.optimization.groot.action import ActionGraph
from coexecution.optimization.session import GrootOptimizations
from tests.helpers.groot_streaming import make_policy_and_inputs


@pytest.fixture
def flow():
    return make_policy_and_inputs(streaming=False)


def equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("kv_reuse", [False, True])
@torch.inference_mode()
def test_action_graph_fresh_inputs_noise_rng_and_owned_outputs(flow, kv_reuse):
    policy, inputs = flow
    head = policy.model.action_head
    original = head.get_action
    expected, rng = [], []
    options = [{"image_delay": value} for value in (0, 2, 0, 2)]
    torch.cuda.manual_seed(619)
    for (backbone, action), request in zip(inputs, options):
        expected.append(original(backbone, action, request))
        rng.append(torch.cuda.get_rng_state().clone())
    torch.cuda.manual_seed(619)
    retained = []
    with ActionGraph(policy, max_entries=2, kv_reuse=kv_reuse) as graph:
        for index, ((backbone, action), request) in enumerate(zip(inputs, options)):
            actual = head.get_action(backbone, action, request)
            equal(actual, expected[index])
            assert torch.equal(torch.cuda.get_rng_state(), rng[index])
            for output, saved in retained:
                equal(output, saved)
            retained.append(
                (actual, {key: value.clone() for key, value in actual.items()})
            )
        evidence = graph.evidence()
        assert evidence["capture_count"] == 2
        assert evidence["replay_count"] == 4
        assert evidence["logical_nfe_count"] == 16
        # Three warmups and one capture execute the Python Flow4 body per graph.
        assert evidence["verified_flow4_body_count"] == 8
        assert evidence["python_dit_forward_count"] == 32
    assert head.get_action == original
    assert not head.model._forward_hooks
    for output, saved in retained:
        equal(output, saved)


@pytest.mark.parametrize("kv_reuse", [False, True])
@torch.inference_mode()
def test_action_graph_stream_layout_eviction_and_hook_counts(flow, kv_reuse):
    policy, inputs = flow
    head = policy.model.action_head
    original = head.get_action
    stream = torch.cuda.Stream()
    calls = []

    def observe(_module, _inputs, _output):
        calls.append(1)

    observe.__pir2_execution_observer__ = True
    hook = head.model.register_forward_hook(observe)
    try:
        with ActionGraph(policy, max_entries=1, kv_reuse=kv_reuse) as graph:
            for index, (backbone, action) in enumerate(inputs):
                # A stepped sequence slice exercises owned input stride preservation.
                values = dict(backbone)
                for key, value in values.items():
                    values[key] = value[:, ::2] if index % 2 else value
                backbone = type(backbone)(data=values)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    torch.cuda.manual_seed(19 + index)
                    expected = original(backbone, action)
                    torch.cuda.manual_seed(19 + index)
                    equal(head.get_action(backbone, action), expected)
                torch.cuda.current_stream().wait_stream(stream)
            assert graph.evidence()["capture_count"] == 4
            assert graph.evidence()["eviction_count"] == 3
        assert hook.id in head.model._forward_hooks
        assert len(calls) == 4 * (4 + 16)
    finally:
        hook.remove()


@pytest.mark.parametrize(
    "mutation",
    ["weight", "training", "config", "method", "hook", "processor", "normalized_shape"],
)
@pytest.mark.parametrize("kv_reuse", [False, True])
@torch.inference_mode()
def test_frozen_execution_rejects_mutation_before_rng(flow, mutation, kv_reuse):
    policy, inputs = flow
    head = policy.model.action_head
    with ActionGraph(policy, kv_reuse=kv_reuse) as graph:
        head.get_action(*inputs[0])
        if mutation == "weight":
            next(head.parameters()).add_(1)
        elif mutation == "training":
            head.train()
        elif mutation == "config":
            head.config.add_pos_embed = not head.config.add_pos_embed
        elif mutation == "method":
            head.model.forward = lambda *args, **kwargs: None
        elif mutation == "processor":
            from diffusers.models.attention_processor import AttnProcessor2_0

            head.model.transformer_blocks[0].attn1.set_processor(AttnProcessor2_0())
        elif mutation == "normalized_shape":
            head.model.norm_out.normalized_shape = (41, 128)
        else:
            head.model.register_forward_hook(lambda *args: None)
        before = torch.cuda.get_rng_state()
        with pytest.raises(ValueError, match="execution changed"):
            head.get_action(*inputs[1])
        assert torch.equal(before, torch.cuda.get_rng_state())
        assert graph.evidence()["replay_count"] == 1


@torch.inference_mode()
def test_unknown_hooks_options_and_dtype_fail_without_rng_or_eager_fallback(flow):
    policy, inputs = flow
    head = policy.model.action_head
    before = torch.cuda.get_rng_state()
    hook = head.state_encoder.register_forward_hook(lambda *args: None)
    with (
        pytest.raises(ValueError, match="unknown submodule hooks"),
        ActionGraph(policy),
    ):
        pass
    assert torch.equal(before, torch.cuda.get_rng_state())
    hook.remove()
    with ActionGraph(policy) as graph:
        with pytest.raises(ValueError, match="unknown options"):
            head.get_action(*inputs[0], options={"inpaint": []})
        assert torch.equal(before, torch.cuda.get_rng_state())
        backbone, action = inputs[0]
        changed = type(action)(data={**action, "state": action.state.float()})
        with pytest.raises(ValueError, match="dtype"):
            head.get_action(backbone, changed)
        assert graph.evidence()["replay_count"] == 0
        assert torch.equal(before, torch.cuda.get_rng_state())


@pytest.mark.parametrize("kv_reuse", [False, True])
@torch.inference_mode()
def test_capture_failure_keeps_wrapper_and_restores_original_on_close(
    flow, monkeypatch, kv_reuse
):
    policy, inputs = flow
    head, failed = policy.model.action_head, []
    original = head.get_action

    def fail(*args):
        failed.append(1)
        raise RuntimeError("capture failed")

    with ActionGraph(policy, kv_reuse=kv_reuse) as graph:
        monkeypatch.setattr(TensorGraphCall, "__call__", fail)
        for _ in range(2):
            with pytest.raises(RuntimeError, match="capture failed"):
                head.get_action(*inputs[0])
        assert graph.evidence()["failure_count"] == 2
        assert graph.evidence()["logical_nfe_count"] == 0
        assert graph.evidence()["resident_signatures"] == 0
        assert head.get_action != original
    assert head.get_action == original and len(failed) == 2


@torch.inference_mode()
def test_action_graph_with_real_serial_boundary_and_session_restores(flow):
    policy, inputs = flow
    head = policy.model.action_head
    original = head.get_action, head.model.forward, head.vl_self_attention.forward
    with GrootOptimizations(
        policy, OptimizationConfig(action_graph=True, dit_graph=True)
    ) as session:
        assert session.graphs is None
        assert isinstance(session.boundary, SerialForwardBoundary)
        for value in inputs:
            head.get_action(*value)
        assert session.action_graph.evidence()["logical_nfe_count"] == 16
        assert session.evidence()["dit_graph_superseded"]
    assert (
        head.get_action,
        head.model.forward,
        head.vl_self_attention.forward,
    ) == original


@torch.inference_mode()
def test_close_and_overlapping_calls_reject_before_restoration(flow):
    policy, inputs = flow
    head = policy.model.action_head
    with ActionGraph(policy) as graph:
        wrapper = head.get_action
        assert graph.lock.acquire(blocking=False)
        try:
            with pytest.raises(RuntimeError, match="active call"):
                graph.close()
            with pytest.raises(RuntimeError, match="serial"):
                head.get_action(*inputs[0])
            assert head.get_action == wrapper
        finally:
            graph.lock.release()


@pytest.mark.parametrize("precision", ["bf16", "w8a8", "w4a4"])
@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("kv_reuse", [False, True])
@torch.inference_mode()
def test_integer_action_graph_matches_same_precision_with_visual_conditioning(
    flow, precision, fusion, kv_reuse
):
    from gr00t.model.modules.dit import SelfAttentionTransformer

    policy, inputs = flow
    head = policy.model.action_head
    with torch.device("cuda"):
        head.vl_self_attention = (
            SelfAttentionTransformer(
                num_attention_heads=4,
                attention_head_dim=32,
                num_layers=1,
                dropout=0.0,
                final_dropout=False,
                positional_embeddings=None,
            )
            .bfloat16()
            .eval()
            .requires_grad_(False)
        )
    config = {
        "precision": precision,
        "component": "dit",
        "scope": "all",
        "fusion": fusion,
        "action_graph": kv_reuse,
    }
    expected, rng = [], []
    with GrootOptimizations(policy, OptimizationConfig(**config)):
        torch.cuda.manual_seed(401)
        for value in inputs:
            expected.append(head.get_action(*value))
            rng.append(torch.cuda.get_rng_state().clone())
    with GrootOptimizations(
        policy,
        OptimizationConfig(
            **{**config, "action_graph": True},
            action_kv_reuse=kv_reuse,
            dit_graph=True,
        ),
    ) as session:
        torch.cuda.manual_seed(401)
        for index, value in enumerate(inputs):
            equal(head.get_action(*value), expected[index])
            assert torch.equal(torch.cuda.get_rng_state(), rng[index])
        assert session.action_graph.evidence()["logical_nfe_count"] == 16
        assert session.graphs is None


@torch.inference_mode()
def test_kv_reuse_same_pointer_refresh_and_captured_projection_proof(flow):
    policy, inputs = flow
    head = policy.model.action_head
    original = head.get_action
    backbone, action = inputs[0]
    expected, rng = [], []
    for index, (fresh_backbone, fresh_action) in enumerate(inputs):
        torch.cuda.manual_seed(811 + index)
        expected.append(original(fresh_backbone, fresh_action))
        rng.append(torch.cuda.get_rng_state().clone())
    with ActionGraph(policy, kv_reuse=True) as graph:
        for index, (fresh_backbone, fresh_action) in enumerate(inputs):
            for target, source in ((backbone, fresh_backbone), (action, fresh_action)):
                for key in target:
                    target[key].copy_(source[key])
            torch.cuda.manual_seed(811 + index)
            equal(head.get_action(backbone, action), expected[index])
            assert torch.equal(torch.cuda.get_rng_state(), rng[index])
            assert graph._kv_scope.retained_count == 0
        proof = graph.evidence()["kv_reuse"]
        assert proof["verified_body_count"] == 4
        assert proof["python_projection_compute_count"] == 16
        assert proof["python_projection_reuse_count"] == 48
        assert proof["expected_projection_count"] == 64
        assert proof["executed_projection_count"] == 16
        assert proof["avoided_projection_count"] == 48
        assert graph.evidence()["logical_nfe_count"] == 16


@torch.inference_mode()
def test_guard_is_frozen_at_scope_entry_before_first_capture(flow):
    policy, inputs = flow
    head = policy.model.action_head
    with ActionGraph(policy):
        head.train()
        before = torch.cuda.get_rng_state()
        with pytest.raises(ValueError, match="execution changed"):
            head.get_action(*inputs[0])
        assert torch.equal(before, torch.cuda.get_rng_state())


@pytest.mark.parametrize("kv_reuse", [False, True])
def test_closed_scope_preserves_new_owner_and_can_be_reentered(flow, kv_reuse):
    policy, _inputs = flow
    head = policy.model.action_head
    original = head.get_action
    old = ActionGraph(policy, kv_reuse=kv_reuse)
    with old:
        assert head.get_action != original
    with ActionGraph(policy):
        current = head.get_action
        old.close()
        assert head.get_action == current
    assert head.get_action == original
    with old:
        assert head.get_action != original
    assert head.get_action == original
