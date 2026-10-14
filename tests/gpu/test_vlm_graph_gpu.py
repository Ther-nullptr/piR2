"""Explicit CUDA replay/ownership checks; no checkpoint or task-quality claim.

PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_vlm_graph_gpu.py
"""

import os
from types import SimpleNamespace

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip("Set PIR2_GPU_TESTS=1 for VLM CUDA checks", allow_module_level=True)

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

from coexecution.kernels.common.graph import CudaGraphCall
from coexecution.optimization.execution.vlm import VlmGraphs


def tiny_qwen():
    config = Qwen3VLConfig(
        vision_config={
            "depth": 2,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_heads": 4,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 128,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0, 1],
        },
        text_config={
            "vocab_size": 64,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [4, 4, 8],
                "mrope_interleaved": True,
            },
        },
    )
    with torch.device("cuda"):
        model = Qwen3VLModel._from_config(
            config, attn_implementation="flash_attention_2", dtype=torch.bfloat16
        )
    return model.eval().requires_grad_(False)


def test_tuple_output_refresh_retention_and_capture_key():
    x = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
    metadata = torch.randn_like(x)
    generation = [0]
    call = CudaGraphCall(
        lambda a: (a + metadata, a * metadata),
        tuple_output=True,
        retained_tensors=lambda: (metadata,),
        capture_key=lambda: generation[0],
    )
    with torch.inference_mode():
        old = call(x)
        saved = tuple(t.clone() for t in old)
        newer = call(x + 1)
        assert call.captures == 1
        for got, expected in zip(newer, (x + 1 + metadata, (x + 1) * metadata)):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
        for got, expected in zip(old, saved):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
        generation[0] += 1
        call(x)
        assert call.captures == 2
    torch.cuda.synchronize()
    call.close()
    assert call.graph is None and not call._constants


@pytest.mark.parametrize("fail_body", [False, True])
def test_vision_text_replay_cache_lifetime_and_restoration(fail_body):
    torch.manual_seed(219)
    qwen = tiny_qwen()
    backbone = SimpleNamespace(model=SimpleNamespace(model=qwen))
    pixels = torch.randn(16, 24, device="cuda", dtype=torch.bfloat16)
    grid = torch.tensor([[1, 4, 4]], device="cuda", dtype=torch.int64)
    hidden = torch.randn(1, 8, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(8, device="cuda").view(1, 1, 8).expand(3, 1, 8)
    originals = [(m, m.forward, "forward" in m.__dict__) for m in qwen.modules()]
    scope = VlmGraphs(backbone)
    with torch.inference_mode():
        reference_vision = qwen.visual(pixels, grid)
        reference_text = qwen.language_model(
            inputs_embeds=hidden,
            position_ids=positions,
            use_cache=True,
        )
        scope.install()
        try:
            vision = qwen.visual(pixels, grid)
            text = qwen.language_model(
                inputs_embeds=hidden,
                position_ids=positions,
                use_cache=True,
            )
            for a, b in zip(
                (vision[0], *vision[1]), (reference_vision[0], *reference_vision[1])
            ):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(
                text.last_hidden_state, reference_text.last_hidden_state, rtol=0, atol=0
            )
            kept = text.last_hidden_state.clone()
            kv = [
                (layer.keys.clone(), layer.values.clone())
                for layer in text.past_key_values.layers
            ]
            qwen.visual(pixels + 1, grid)
            qwen.language_model(
                inputs_embeds=hidden + 1, position_ids=positions, use_cache=True
            )
            assert scope.evidence()["captures"] == 3
            torch.testing.assert_close(text.last_hidden_state, kept, rtol=0, atol=0)
            for layer, (keys, values) in zip(text.past_key_values.layers, kv):
                torch.testing.assert_close(layer.keys, keys, rtol=0, atol=0)
                torch.testing.assert_close(layer.values, values, rtol=0, atol=0)
            with pytest.raises(ValueError, match="historical KV"):
                qwen.language_model(
                    inputs_embeds=hidden,
                    position_ids=positions,
                    past_key_values=text.past_key_values,
                )
            mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]], device="cuda")
            with pytest.raises(ValueError, match="padding masks"):
                qwen.language_model(
                    inputs_embeds=hidden, position_ids=positions, attention_mask=mask
                )
            if fail_body:
                raise RuntimeError("deliberate test body failure")
        except RuntimeError as error:
            if not fail_body or str(error) != "deliberate test body failure":
                raise
        finally:
            torch.cuda.synchronize()
            scope.close()
        for module, forward, had in originals:
            assert module.forward == forward
            assert ("forward" in module.__dict__) == had
        restored = qwen.visual(pixels, grid)
        torch.testing.assert_close(restored[0], reference_vision[0], rtol=0, atol=0)
        scope.close()
