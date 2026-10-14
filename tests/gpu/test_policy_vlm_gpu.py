"""Exact real tiny CUDA Qwen3/GR00T VLM contracts, including full VLM capture.

PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_policy_vlm_gpu.py
Every neural constructor executes directly on CUDA. No LIBERO quality claim.
"""

import importlib
import importlib.util
import inspect
import os
from types import SimpleNamespace

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip(
        "Set PIR2_GPU_TESTS=1 for real CUDA VLM checks", allow_module_level=True
    )

import torch

MODULE = "coexecution.optimization.groot.vlm"
METADATA_KEYS = ("input_ids", "attention_mask", "image_grid_thw")


def implementation():
    assert importlib.util.find_spec(MODULE) is not None, (
        "Static policy VLM module is missing"
    )
    return importlib.import_module(MODULE).PolicyStaticS2


def test_static_policy_api_and_legacy_core_entrypoint():
    from coexecution.static_s2 import StaticS2

    assert callable(implementation())
    assert callable(getattr(StaticS2, "forward_inputs", None))
    assert "vision_rope" in inspect.signature(StaticS2).parameters


@pytest.fixture
def policy():
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
    from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    assert torch.cuda.device_count() == 1, "Use exactly one admitted clean CUDA card"
    torch.manual_seed(113)
    config = Gr00tN1d7Config(
        backbone_embedding_dim=128,
        hidden_size=128,
        input_embedding_dim=128,
        max_state_dim=8,
        max_action_dim=8,
        action_horizon=40,
        max_num_embodiments=1,
        max_seq_len=64,
        num_inference_timesteps=4,
        streaming=False,
        state_dropout_prob=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 4,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 128,
            "interleave_self_attention": True,
        },
    )
    qwen_config = Qwen3VLConfig(
        image_token_id=12,
        video_token_id=13,
        vision_start_token_id=10,
        vision_end_token_id=11,
        text_config={
            "vocab_size": 64,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "max_position_embeddings": 128,
            "rope_scaling": {"rope_type": "default", "mrope_section": [6, 5, 5]},
            "use_cache": False,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "patch_size": 4,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 128,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0, 1],
        },
        attn_implementation="sdpa",
    )
    with torch.device("cuda"):
        model = Gr00tN1d7.__new__(Gr00tN1d7)
        torch.nn.Module.__init__(model)
        model.config = config
        backbone = Qwen3Backbone.__new__(Qwen3Backbone)
        torch.nn.Module.__init__(backbone)
        backbone.model = Qwen3VLForConditionalGeneration(qwen_config)
        backbone.tune_llm = backbone.tune_visual = False
        model.backbone = backbone
        model.action_head = Gr00tN1d7ActionHead(config)
        model.to(dtype=torch.bfloat16).eval()
    return SimpleNamespace(model=model)


def inputs(*, pixels=None, token=20, padded=True, grid=(1, 4, 4), state=0.1):
    # Token/grid tensors are CPU collator metadata. Numeric images/states originate
    # on CUDA; no learned computation or model construction occurs on CPU.
    ids = torch.tensor([[4, 10, 12, 12, 12, 12, 11, token, 21, 0]], device="cpu")
    mask = torch.ones_like(ids)
    if padded:
        mask[0, -1] = 0
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "image_grid_thw": torch.tensor([grid], device="cpu"),
        "pixel_values": torch.randn(16, 96, device="cuda", dtype=torch.bfloat16)
        if pixels is None
        else pixels,
        "state": torch.full((1, 1, 8), state, device="cuda", dtype=torch.bfloat16),
        "embodiment_id": torch.tensor([0], device="cuda"),
    }


def prepare(policy, batch):
    return policy.model.backbone.prepare_input(batch).to("cuda", dtype=torch.bfloat16)


def snapshot(output):
    return {key: tensor.clone() for key, tensor in output.items()}


def exact(reference, actual):
    assert reference.keys() == actual.keys()
    for key in reference:
        torch.testing.assert_close(reference[key], actual[key], atol=0, rtol=0)
        assert reference[key].stride() == actual[key].stride()


@pytest.mark.parametrize("cuda_graph", [False, True])
@pytest.mark.parametrize("padded", [False, True])
@torch.inference_mode()
def test_features_match_pinned_hidden_states_after_last_deepstack(
    policy, cuda_graph, padded
):
    batch = inputs(padded=padded)
    backbone = policy.model.backbone
    reference = snapshot(backbone(prepare(policy, batch)))
    with implementation()(policy, cuda_graph=cuda_graph) as scope:
        exact(reference, backbone(prepare(policy, batch)))
        metadata = next(iter(scope.core.registrations.values())).metadata
        # Match the original helper's None/4-D decision, not an artificial mask.
        assert (metadata.causal_mask is None) == (not padded)
        if padded:
            assert metadata.causal_mask.dtype == torch.bool
        assert scope.evidence()["pixel_outputs_cached"] is False


@pytest.mark.parametrize("cuda_graph", [False, True])
@torch.inference_mode()
def test_full_seeded_gr00t_actions_and_fresh_image_state_are_exact(policy, cuda_graph):
    model = policy.model
    batch = inputs()
    changed = inputs(pixels=batch["pixel_values"] * 0.5, state=-0.2)

    def sample(values):
        torch.manual_seed(211)
        model.action_head.reset_streaming_buffer()
        return snapshot(model.get_action(values))

    references = [sample(values) for values in (batch, changed)]
    with implementation()(policy, cuda_graph=cuda_graph) as scope:
        for reference, values in zip(references, (batch, changed)):
            exact(reference, sample(values))
        assert scope.evidence()["resident_bindings"] == 1
        if cuda_graph:
            assert scope.evidence()["vlm_graph"]["capture_count"] == 1
            assert scope.evidence()["vlm_graph"]["replay_count"] == 2


@torch.inference_mode()
def test_static_scope_composes_with_input_separation_and_session_switching(policy):
    from coexecution.groot_optimization import GrootOptimizations, OptimizationConfig

    batch = inputs()

    def sample():
        torch.manual_seed(237)
        policy.model.action_head.reset_streaming_buffer()
        return snapshot(policy.model.get_action(batch))

    reference = sample()
    with GrootOptimizations(
        policy, OptimizationConfig(input_preparation=True)
    ) as scope:
        exact(reference, sample())
        scope.set_runtime(static_vlm=True)
        exact(reference, sample())
        scope.set_runtime(vlm_graph=True)
        exact(reference, sample())
        scope.set_runtime(static_vlm=False, vlm_graph=False)
        exact(reference, sample())
        assert scope.evidence()["input_preparation"]["fallbacks"] == {}


@pytest.mark.parametrize("cuda_graph", [False, True])
@pytest.mark.parametrize("changed", ["text", "mask", "equal_count_grid"])
@torch.inference_mode()
def test_same_shape_changed_metadata_cannot_reuse_the_wrong_binding(
    policy, cuda_graph, changed
):
    batch = inputs()
    other = inputs(pixels=batch["pixel_values"])
    if changed == "text":
        other["input_ids"][0, 7] = 22
    elif changed == "mask":
        other["attention_mask"][0, 8] = 0
    else:
        other["image_grid_thw"][0] = torch.tensor([1, 2, 8], device="cpu")
    backbone = policy.model.backbone
    reference = snapshot(backbone(prepare(policy, other)))
    with implementation()(policy, cuda_graph=cuda_graph) as scope:
        backbone(prepare(policy, batch))
        exact(reference, backbone(prepare(policy, other)))
        assert scope.evidence()["resident_bindings"] == 2
        if cuda_graph:
            assert scope.evidence()["vlm_graph"]["capture_count"] == 2


@torch.inference_mode()
def test_primed_replays_refresh_pixels_keep_outputs_and_do_not_recapture(
    policy, monkeypatch
):
    backbone = policy.model.backbone
    batch = inputs()
    changed = inputs(pixels=batch["pixel_values"] * 0.5)
    reference = snapshot(backbone(prepare(policy, changed)))
    with implementation()(policy, cuda_graph=True) as scope:
        first = backbone(prepare(policy, batch))
        saved = snapshot(first)
        # Returned masks and features are caller-owned; poisoning one output must
        # leave graph constants and later independently retained outputs intact.
        first["image_mask"].zero_()
        first["backbone_attention_mask"].zero_()
        first["backbone_features"].zero_()
        after = backbone(prepare(policy, changed))
        exact(reference, after)
        after_saved = snapshot(after)
        before = scope.evidence()["vlm_graph"]["capture_count"]

        def no_cpu(tensor, *args, **kwargs):
            if tensor.is_cuda:
                pytest.fail("A primed VLM replay must not read resident values on CPU")
            return original_cpu(tensor, *args, **kwargs)

        original_cpu = torch.Tensor.cpu
        monkeypatch.setattr(torch.Tensor, "cpu", no_cpu)
        for _ in range(3):
            backbone(prepare(policy, batch))
            exact(reference, backbone(prepare(policy, changed)))
        exact(after_saved, after)
        assert not torch.equal(saved["backbone_features"], after["backbone_features"])
        assert scope.evidence()["vlm_graph"]["capture_count"] == before
        assert scope.evidence()["vlm_graph"]["replay_count"] == 8


@torch.inference_mode()
def test_metadata_and_graph_cache_eviction_are_bounded_together(policy):
    backbone = policy.model.backbone
    batch = inputs()
    with implementation()(policy, cuda_graph=True, max_entries=2) as scope:
        for token in (20, 22, 23, 20):
            current = inputs(pixels=batch["pixel_values"], token=token)
            reference = snapshot(scope.original_forward(prepare(policy, current)))
            exact(reference, backbone(prepare(policy, current)))
        evidence = scope.evidence()
        assert evidence["resident_bindings"] <= 2
        assert evidence["metadata_entries"] <= 2
        assert evidence["vision_entries"] <= 2
        assert evidence["vlm_graph"]["resident_signatures"] <= 2
        assert evidence["vlm_graph"]["capture_count"] == 4
        assert evidence["binding_evictions"] == 2


@pytest.mark.parametrize("cuda_graph", [False, True])
@torch.inference_mode()
def test_owned_metadata_mutation_rejects_and_fresh_preparation_recovers(
    policy, cuda_graph
):
    batch = inputs()
    backbone = policy.model.backbone
    reference = snapshot(backbone(prepare(policy, batch)))
    with implementation()(policy, cuda_graph=cuda_graph) as scope:
        prepared = prepare(policy, batch)
        backbone(prepared)
        prepared["input_ids"][0, 7] = 25
        with pytest.raises(ValueError, match="metadata has changed"):
            backbone(prepared)
        exact(reference, backbone(prepare(policy, batch)))
        assert scope.evidence()["invalidations"] >= 1


@torch.inference_mode()
def test_graph_detects_weight_mutation_then_preparation_rebuilds(policy):
    batch = inputs()
    backbone = policy.model.backbone
    with implementation()(policy, cuda_graph=True) as scope:
        prepared = prepare(policy, batch)
        backbone(prepared)
        backbone.model.model.language_model.layers[0].self_attn.q_proj.weight.add_(0.01)
        with pytest.raises(ValueError, match="execution changed|prepare again"):
            backbone(prepared)
        prepared = prepare(policy, batch)
        reference = snapshot(scope.original_forward(prepared))
        exact(reference, backbone(prepared))
        assert scope.evidence()["vlm_graph"]["capture_count"] == 2


@pytest.mark.parametrize("cuda_graph", [False, True])
@pytest.mark.parametrize("producer", ["positions", "interpolation", "vision_rotary"])
@torch.inference_mode()
def test_changed_metadata_producer_rejects_old_inputs_and_fresh_preparation_is_exact(
    policy, cuda_graph, producer
):
    batch = inputs()
    backbone = policy.model.backbone
    inner = backbone.model.model
    owner, name = {
        "positions": (inner, "get_rope_index"),
        "interpolation": (inner.visual, "fast_pos_embed_interpolate"),
        "vision_rotary": (inner.visual, "rot_pos_emb"),
    }[producer]
    original = getattr(owner, name)
    had = name in vars(owner)
    try:
        with implementation()(policy, cuda_graph=cuda_graph) as scope:
            prepared = prepare(policy, batch)
            backbone(prepared)
            old_metadata = next(iter(scope.core.registrations.values())).metadata

            def changed(*args, **kwargs):
                result = original(*args, **kwargs)
                return (
                    tuple(value + 1 for value in result)
                    if isinstance(result, tuple)
                    else result + 0.125
                )

            setattr(owner, name, changed)
            with pytest.raises(ValueError, match="execution changed|prepare again"):
                backbone(prepared)
            prepared = prepare(policy, batch)
            reference = snapshot(scope.original_forward(prepared))
            exact(reference, backbone(prepared))
            new_key = scope._resident_keys[
                tuple(id(prepared[key]) for key in METADATA_KEYS)
            ]
            binding = scope.bindings[new_key]
            new_metadata = scope.core.registrations[id(binding)].metadata
            assert new_metadata is not old_metadata
            assert new_metadata.visual is not old_metadata.visual
            if cuda_graph:
                assert scope.evidence()["vlm_graph"]["capture_count"] == 2
    finally:
        if had:
            setattr(owner, name, original)
        else:
            delattr(owner, name)


@pytest.mark.parametrize("cuda_graph", [False, True])
@pytest.mark.parametrize("mutation", ["defaults", "closure"])
@torch.inference_mode()
def test_same_callable_metadata_producer_changes_rebuild_derived_input_and_vision_metadata(
    policy, cuda_graph, mutation
):
    batch = inputs()
    backbone = policy.model.backbone
    visual = backbone.model.model.visual
    original = visual.fast_pos_embed_interpolate
    offset = [0.0]

    def changed(grid, amount=0.0):
        return original(grid) + (amount if mutation == "defaults" else offset[0])

    visual.fast_pos_embed_interpolate = changed
    try:
        with implementation()(policy, cuda_graph=cuda_graph) as scope:
            prepared = prepare(policy, batch)
            backbone(prepared)
            old_metadata = next(iter(scope.core.registrations.values())).metadata
            if mutation == "defaults":
                changed.__defaults__ = (0.125,)
            else:
                offset[0] = 0.125
            with pytest.raises(ValueError, match="execution changed|prepare again"):
                backbone(prepared)
            prepared = prepare(policy, batch)
            reference = snapshot(scope.original_forward(prepared))
            exact(reference, backbone(prepared))
            key = scope._resident_keys[
                tuple(id(prepared[name]) for name in METADATA_KEYS)
            ]
            new_metadata = scope.core.registrations[id(scope.bindings[key])].metadata
            assert (
                new_metadata is not old_metadata
                and new_metadata.visual is not old_metadata.visual
            )
            assert not torch.equal(
                new_metadata.visual.pos_embeds, old_metadata.visual.pos_embeds
            )
            assert scope.core.metadata_epoch == 1
    finally:
        del visual.fast_pos_embed_interpolate


@torch.inference_mode()
def test_reentry_rebuilds_derived_metadata_after_closed_scope_producer_default_change(
    policy,
):
    batch = inputs()
    backbone = policy.model.backbone
    visual = backbone.model.model.visual
    original = visual.fast_pos_embed_interpolate

    def changed(grid, amount=0.0):
        return original(grid) + amount

    visual.fast_pos_embed_interpolate = changed
    scope = implementation()(policy, cuda_graph=True)
    try:
        with scope:
            backbone(prepare(policy, batch))
            old_metadata = next(iter(scope.core.registrations.values())).metadata
        changed.__defaults__ = (0.125,)
        reference = snapshot(backbone(prepare(policy, batch)))
        with scope:
            prepared = prepare(policy, batch)
            exact(reference, backbone(prepared))
            key = scope._resident_keys[
                tuple(id(prepared[name]) for name in METADATA_KEYS)
            ]
            new_metadata = scope.core.registrations[id(scope.bindings[key])].metadata
            assert new_metadata.visual is not old_metadata.visual
            assert scope.core.metadata_epoch == 1
    finally:
        del visual.fast_pos_embed_interpolate


@pytest.mark.parametrize("cuda_graph", [False, True])
@pytest.mark.parametrize("mutation", ["defaults", "closure"])
@torch.inference_mode()
def test_original_cold_mask_helper_mutation_rebuilds_effective_mask_and_exact_features(
    policy, cuda_graph, mutation, monkeypatch
):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

    original = qwen.create_causal_mask
    modified = [False]

    def helper(
        config,
        input_embeds,
        attention_mask,
        cache_position,
        past_key_values,
        position_ids=None,
        block_first_key=False,
    ):
        result = original(
            config=config,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        if block_first_key if mutation == "defaults" else modified[0]:
            assert result is not None
            result = result.clone()
            result[..., 0] = False
        return result

    monkeypatch.setattr(qwen, "create_causal_mask", helper)
    batch, backbone = inputs(padded=True), policy.model.backbone
    with implementation()(policy, cuda_graph=cuda_graph) as scope:
        prepared = prepare(policy, batch)
        backbone(prepared)
        old_metadata = next(iter(scope.core.registrations.values())).metadata
        if mutation == "defaults":
            helper.__defaults__ = (None, True)
        else:
            modified[0] = True
        with pytest.raises(ValueError, match="execution changed|prepare again"):
            backbone(prepared)
        prepared = prepare(policy, batch)
        reference = snapshot(scope.original_forward(prepared))
        exact(reference, backbone(prepared))
        key = scope._resident_keys[tuple(id(prepared[name]) for name in METADATA_KEYS)]
        metadata = scope.core.registrations[id(scope.bindings[key])].metadata
        assert metadata is not old_metadata
        assert not torch.equal(metadata.causal_mask, old_metadata.causal_mask)
        assert scope.core.metadata_epoch == 1


@pytest.mark.parametrize("cuda_graph", [False, True])
@pytest.mark.parametrize("field", ["_causal_mask_original", "vision_rope"])
@pytest.mark.parametrize("mutation", ["defaults", "closure"])
@torch.inference_mode()
def test_replaced_helpers_reprepare_then_track_replacement_mutation_and_exact_features(
    policy, cuda_graph, field, mutation, monkeypatch
):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

    batch, backbone = inputs(padded=True), policy.model.backbone
    offset = [0.0]
    with implementation()(policy, cuda_graph=cuda_graph) as scope:
        prepared = prepare(policy, batch)
        backbone(prepared)
        if field == "_causal_mask_original":
            original = scope.core._causal_mask_original

            def replacement(*args, amount=0.0, **kwargs):
                result = original(*args, **kwargs)
                if amount if mutation == "defaults" else offset[0]:
                    assert result is not None
                    result = result.clone()
                    result[..., 0] = False
                return result

        else:
            original = qwen.apply_rotary_pos_emb_vision

            def replacement(query, key, cos, sin, *, amount=0.0):
                query, key = original(query, key, cos, sin)
                value = amount if mutation == "defaults" else offset[0]
                return query + value, key + value

            # The independent eager reference uses the same chosen arithmetic;
            # its GPU vision/text layers remain the original pinned implementation.
            monkeypatch.setattr(qwen, "apply_rotary_pos_emb_vision", replacement)
        setattr(scope.core, field, replacement)
        prepared = prepare(policy, batch)
        reference = snapshot(scope.original_forward(prepared))
        exact(reference, backbone(prepared))
        if mutation == "defaults":
            replacement.__kwdefaults__["amount"] = 0.125
        else:
            offset[0] = 0.125
        with pytest.raises(ValueError, match="execution changed|prepare again"):
            backbone(prepared)
        prepared = prepare(policy, batch)
        reference = snapshot(scope.original_forward(prepared))
        exact(reference, backbone(prepared))
        assert scope.core.metadata_epoch == 2


@pytest.mark.parametrize("cuda_graph", [False, True])
@torch.inference_mode()
def test_training_and_replaced_components_are_rejected_before_neural_execution(
    policy, cuda_graph
):
    backbone = policy.model.backbone
    scope = implementation()(policy, cuda_graph=cuda_graph)
    with scope:
        backbone.model.model.language_model.train()
        with pytest.raises(ValueError, match="eval"):
            backbone.prepare_input(inputs())
        backbone.model.model.language_model.eval()
        policy.model.action_head.model.train()
        with pytest.raises(ValueError, match="eval"):
            backbone.prepare_input(inputs())
        policy.model.action_head.model.eval()
        original = backbone.model.model.visual
        # A host sentinel is sufficient: admission must reject the replacement
        # before accessing it or constructing any alternate neural module.
        del backbone.model.model.visual
        backbone.model.model.visual = SimpleNamespace()
        try:
            with pytest.raises(ValueError, match="replaced"):
                backbone.prepare_input(inputs())
        finally:
            del backbone.model.model.visual
            backbone.model.model.visual = original


@torch.inference_mode()
def test_direct_static_scope_rejects_fp32_cuda_model_before_installation(policy):
    policy.model.to(dtype=torch.float32)
    original = policy.model.backbone.forward
    with pytest.raises(ValueError, match="CUDA BF16"), implementation()(policy):
        pass
    assert policy.model.backbone.forward == original


@torch.inference_mode()
def test_metadata_ready_on_one_stream_can_capture_and_replay_on_another(policy):
    backbone = policy.model.backbone
    batch = inputs()
    reference = snapshot(backbone(prepare(policy, batch)))
    caller = torch.cuda.current_stream()
    worker = torch.cuda.Stream()
    with implementation()(policy, cuda_graph=True) as scope:
        prepared = prepare(policy, batch)
        worker.wait_stream(caller)
        with torch.cuda.stream(worker):
            first = backbone(prepared)
            for tensor in first.values():
                tensor.record_stream(worker)
        caller.wait_stream(worker)
        exact(reference, first)
        worker.wait_stream(caller)
        with torch.cuda.stream(worker):
            second = backbone(prepared)
            for tensor in second.values():
                tensor.record_stream(worker)
        caller.wait_stream(worker)
        exact(reference, second)
        assert scope.evidence()["vlm_graph"]["capture_count"] == 1
        assert scope.evidence()["vlm_graph"]["replay_count"] == 2


@torch.inference_mode()
def test_core_workload_interface_and_explicit_vision_rope_remain_exact(policy):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

    from coexecution.static_s2 import StaticS2

    batch = inputs()
    backbone = policy.model.backbone
    reference = snapshot(backbone(prepare(policy, batch)))
    workload = SimpleNamespace(
        model=policy.model,
        features_only=True,
        backbone_inputs=[prepare(policy, batch)],
        _pre_final_norm=None,
        slow=lambda index: backbone(prepare(policy, batch)),
    )
    calls = []

    def rope(*args):
        calls.append(1)
        return qwen.apply_rotary_pos_emb_vision(*args)

    def capture(_module, values):
        workload._pre_final_norm = values[0]

    hook = backbone.model.model.language_model.norm.register_forward_pre_hook(capture)
    try:
        with StaticS2(workload, vision_rope=rope) as scope:
            exact(reference, workload.slow(0))
            assert scope.evidence()["vision_rope_binding"] == "explicit callable"
        assert len(calls) == 2
    finally:
        hook.remove()


@pytest.mark.parametrize("hook_site", ["text_layer", "vision_layer", "norm"])
@torch.inference_mode()
def test_graph_rejects_unknown_captured_hooks_but_backbone_observer_is_allowed(
    policy, hook_site
):
    backbone = policy.model.backbone
    sites = {
        "text_layer": backbone.model.model.language_model.layers[0],
        "vision_layer": backbone.model.model.visual.blocks[0],
        "norm": backbone.model.model.language_model.norm,
    }
    hook = sites[hook_site].register_forward_hook(lambda *_args: None)
    try:
        with (
            pytest.raises(ValueError, match="hooks"),
            implementation()(policy, cuda_graph=True),
        ):
            pass
    finally:
        hook.remove()
    calls = []
    hook = backbone.register_forward_hook(lambda *_args: calls.append(1))
    try:
        with implementation()(policy, cuda_graph=True):
            batch = inputs()
            backbone(prepare(policy, batch))
            backbone(prepare(policy, batch))
        assert len(calls) == 2
    finally:
        hook.remove()


@torch.inference_mode()
def test_failed_capture_restores_eager_entry_and_reentry_clears_graphs(
    policy, monkeypatch
):
    from coexecution.kernels.common import graph

    backbone = policy.model.backbone
    batch = inputs()
    reference = snapshot(backbone(prepare(policy, batch)))
    original_forward, original_prepare = backbone.forward, backbone.prepare_input
    original_call = graph.CudaGraphCall.__call__
    scope = implementation()(policy, cuda_graph=True)
    with scope:

        def fail(*_args, **_kwargs):
            raise RuntimeError("intentional VLM capture failure")

        monkeypatch.setattr(graph.CudaGraphCall, "__call__", fail)
        with pytest.raises(RuntimeError, match="intentional VLM capture failure"):
            backbone(prepare(policy, batch))
        assert scope.evidence()["vlm_graph"]["failure_count"] == 1
        assert scope.evidence()["vlm_graph"]["resident_signatures"] == 0
        assert backbone.forward == original_forward
        monkeypatch.setattr(graph.CudaGraphCall, "__call__", original_call)
    assert (
        backbone.forward == original_forward
        and backbone.prepare_input == original_prepare
    )
    with scope:
        exact(reference, backbone(prepare(policy, batch)))
        assert scope.evidence()["vlm_graph"]["resident_signatures"] == 1
    assert scope.evidence()["vlm_graph"]["resident_signatures"] == 0
