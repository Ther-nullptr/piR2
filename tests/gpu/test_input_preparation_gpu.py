"""Real tiny Qwen3/GR00T preparation contracts; no learned-quality claim.

PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_input_preparation_gpu.py
All neural constructors and numerical inputs originate on CUDA.
"""

import importlib
import importlib.util
import os
from types import SimpleNamespace

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip(
        "Set PIR2_GPU_TESTS=1 for preparation CUDA checks", allow_module_level=True
    )

import torch

PREPARATION = "coexecution.optimization.execution.preparation"
BACKBONE_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
ACTION_KEYS = ("state", "embodiment_id", "action", "action_mask", "image_delay")


def adapter(policy):
    assert importlib.util.find_spec(PREPARATION) is not None, (
        "InputPreparation module is missing"
    )
    return importlib.import_module(PREPARATION).InputPreparation(policy)


def test_input_preparation_api():
    assert importlib.util.find_spec(PREPARATION) is not None, (
        "InputPreparation module is missing"
    )
    assert callable(importlib.import_module(PREPARATION).InputPreparation)


@pytest.fixture
def tiny_policy():
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
    from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.manual_seed(109)
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
        image_token_id=4,
        video_token_id=5,
        vision_start_token_id=2,
        vision_end_token_id=3,
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
            "deepstack_visual_indexes": [0],
        },
    )
    qwen_config._attn_implementation = "eager"
    # The exact pinned interfaces are used, avoiding any from_pretrained download.
    with torch.device("cuda"):
        model = Gr00tN1d7.__new__(Gr00tN1d7)
        torch.nn.Module.__init__(model)
        model.config = config
        backbone = Qwen3Backbone.__new__(Qwen3Backbone)
        torch.nn.Module.__init__(backbone)
        backbone.model = Qwen3VLForConditionalGeneration(qwen_config)
        backbone.tune_llm = False
        backbone.tune_visual = False
        model.backbone = backbone
        model.action_head = Gr00tN1d7ActionHead(config)
        model.to(dtype=torch.bfloat16).eval()
    return SimpleNamespace(model=model)


def inputs(*, optional=False):
    with torch.device("cuda"):
        batch = {
            "input_ids": torch.tensor([[2, 4, 4, 4, 4, 3, 20, 21]], dtype=torch.long),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
            "pixel_values": torch.randn(16, 96, dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long),
            "state": torch.randn(1, 1, 8, dtype=torch.float32),
            "embodiment_id": torch.zeros(1, dtype=torch.long),
        }
        if optional:
            batch.update(
                action=torch.randn(1, 40, 8, dtype=torch.float32),
                action_mask=torch.ones(1, 40, 8, dtype=torch.bool),
                image_delay=torch.tensor([2], dtype=torch.long),
            )
    return batch


@pytest.mark.parametrize("optional", [False, True])
@torch.inference_mode()
def test_real_prepared_values_and_each_field_transfers_once(
    tiny_policy, optional, monkeypatch, record_property
):
    from transformers.feature_extraction_utils import BatchFeature

    batch = inputs(optional=optional)
    saved = {key: value.clone() for key, value in batch.items()}
    original_to = torch.Tensor.to
    transfers = []

    def record_to(tensor, *args, **kwargs):
        transfers.append(tensor)
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", record_to)
    expected = tiny_policy.model.prepare_input(batch)
    baseline_calls = len(transfers)
    assert baseline_calls == 2 * len(batch)
    assert (
        expected[0]["pixel_values"].data_ptr() != expected[1]["pixel_values"].data_ptr()
    )
    transfers.clear()
    with adapter(tiny_policy):
        actual = tiny_policy.model.prepare_input(batch)
        separated_calls = len(transfers)
        assert separated_calls == len(batch)
        assert all(isinstance(part, BatchFeature) for part in actual)
        assert tuple(actual[0]) == BACKBONE_KEYS
        assert tuple(actual[1]) == (ACTION_KEYS if optional else ACTION_KEYS[:2])
        for index, part in enumerate(actual):
            for key, value in part.items():
                torch.testing.assert_close(value, expected[index][key], rtol=0, atol=0)
                assert value.device.type == "cuda"
        for key, value in batch.items():
            torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
    record_property("baseline_tensor_to_calls", baseline_calls)
    record_property("separated_tensor_to_calls", separated_calls)


@torch.inference_mode()
def test_real_complete_action_fresh_pixels_state_hooks_and_restoration(tiny_policy):
    model = tiny_policy.model
    batch = inputs(optional=True)
    options = {
        "force_nonstreaming": True,
        "inpaint": batch["action"][:, :2],
        "image_delay": 2,
    }
    original_prepare = model.prepare_input
    original_forward = model.backbone.forward
    features = []
    hook = model.backbone.register_forward_hook(
        lambda _module, _args, output: features.append(output["backbone_features"])
    )
    saved_hooks = dict(model.backbone._forward_hooks)
    torch.cuda.manual_seed(572)
    expected = model.get_action(batch, options)["action_pred"]
    changed = {
        **batch,
        "pixel_values": -batch["pixel_values"],
        "state": -batch["state"],
    }
    torch.cuda.manual_seed(572)
    changed_expected = model.get_action(changed, options)["action_pred"]
    assert not torch.equal(expected, changed_expected)
    with adapter(tiny_policy):
        assert model.backbone.forward == original_forward
        torch.cuda.manual_seed(572)
        actual = model.get_action(batch, options)["action_pred"]
        retained = actual.clone()
        torch.cuda.manual_seed(572)
        changed_actual = model.get_action(changed, options)["action_pred"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(changed_actual, changed_expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, retained, rtol=0, atol=0)
        assert not torch.equal(features[-2], features[-1])
        assert model.backbone._forward_hooks == saved_hooks
    assert model.prepare_input == original_prepare
    assert "prepare_input" not in vars(model)
    assert "prepare_input" not in vars(model.backbone)
    assert "prepare_input" not in vars(model.action_head)
    assert model.backbone.forward == original_forward
    hook.remove()


@torch.inference_mode()
def test_real_preparation_body_failure_restores_and_reentry_is_fresh(tiny_policy):
    model = tiny_policy.model
    first, second = inputs(), inputs()
    saved = model.prepare_input
    scope = adapter(tiny_policy)
    with pytest.raises(RuntimeError, match="deliberate body"), scope:
        previous = model.prepare_input(first)
        previous_values = tuple(
            {key: value.clone() for key, value in part.items()} for part in previous
        )
        raise RuntimeError("deliberate body failure")
    assert model.prepare_input == saved
    with scope:
        current = model.prepare_input(second)
        assert (
            current[0]["pixel_values"].data_ptr()
            != previous[0]["pixel_values"].data_ptr()
        )
        assert current[1]["state"].data_ptr() != previous[1]["state"].data_ptr()
        for index, part in enumerate(previous):
            for key, value in part.items():
                torch.testing.assert_close(
                    value, previous_values[index][key], rtol=0, atol=0
                )
    assert model.prepare_input == saved


@pytest.mark.parametrize("pre_hook", [False, True])
@torch.inference_mode()
def test_real_hooks_using_action_fields_keep_original_backbone_input(
    tiny_policy, pre_hook
):
    model = tiny_policy.model
    batch = inputs()
    states = []

    def observe(_module, args, *output):
        states.append(args[0]["state"].clone())

    register = (
        model.backbone.register_forward_pre_hook
        if pre_hook
        else model.backbone.register_forward_hook
    )
    hook = register(observe)
    try:
        reference = model.backbone(model.prepare_input(batch)[0])
        with adapter(tiny_policy) as preparation:
            backbone, action = model.prepare_input(batch)
            assert "state" in backbone and "pixel_values" not in action
            actual = model.backbone(backbone)
            torch.testing.assert_close(
                actual["backbone_features"],
                reference["backbone_features"],
                rtol=0,
                atol=0,
            )
            assert preparation.evidence()["fallbacks"]["backbone"] == [
                "forward_pre_hooks" if pre_hook else "forward_hooks"
            ]
        assert len(states) == 2
        torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
    finally:
        hook.remove()


@pytest.mark.parametrize("pre_hook", [False, True])
@torch.inference_mode()
def test_real_global_hooks_using_discarded_fields_preserve_inputs(
    tiny_policy, pre_hook
):
    registry = torch.nn.modules.module
    model = tiny_policy.model
    batch = inputs()
    reference = model.backbone(model.prepare_input(batch)[0])
    states = []
    kind = "global_forward_pre_hooks" if pre_hook else "global_forward_hooks"
    saved = dict(getattr(registry, "_" + kind))

    def observe(module, args, *output):
        if module is model.backbone:
            states.append(args[0]["state"].clone())

    with adapter(tiny_policy) as preparation:
        register = (
            registry.register_module_forward_pre_hook
            if pre_hook
            else registry.register_module_forward_hook
        )
        hook = register(observe)
        try:
            backbone, action = model.prepare_input(batch)
            actual = model.backbone(backbone)
            assert "state" in backbone and "pixel_values" in action
            torch.testing.assert_close(
                actual["backbone_features"],
                reference["backbone_features"],
                rtol=0,
                atol=0,
            )
            assert len(states) == 1
            torch.testing.assert_close(
                states[0], batch["state"].to(torch.bfloat16), rtol=0, atol=0
            )
            assert preparation.evidence()["fallbacks"] == {
                "backbone": [kind],
                "action_head": [kind],
            }
        finally:
            hook.remove()
        assert getattr(registry, "_" + kind) == saved
        separated = model.prepare_input(batch)
        assert tuple(separated[0]) == BACKBONE_KEYS
        assert tuple(separated[1]) == ACTION_KEYS[:2]
