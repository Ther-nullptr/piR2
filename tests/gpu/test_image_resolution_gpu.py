"""Pinned dual-camera processing and real tiny CUDA VLM/DiT shape checks.

PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_image_resolution_gpu.py
All neural construction/execution stays on CUDA. No task-quality claim.
"""

import os

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip("Set PIR2_GPU_TESTS=1 for resolution checks", allow_module_level=True)

import torch

from tests.gpu.test_policy_vlm_gpu import exact, snapshot
from tests.gpu.test_policy_vlm_gpu import policy as policy  # noqa: PLC0414
from tests.gpu.test_tokenization_gpu import processor as processor  # noqa: PLC0414


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@torch.inference_mode()
def test_real_pixels_grids_features_and_graph_actions_at_each_resolution(
    policy, processor, fast, streaming
):
    from copy import deepcopy

    import albumentations as A
    import cv2
    import numpy as np
    from gr00t.model.gr00t_n1d7.image_augmentations import apply_with_replay
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (
        Gr00tN1d7DataCollator,
        Gr00tN1d7Processor,
    )
    from PIL import Image
    from transformers import Qwen2VLImageProcessorFast

    from coexecution.libero.inference import (
        PreparedInferenceEngine,
        prepare_observation,
    )
    from coexecution.libero.preprocessing import (
        ImageResolutionConfig,
        ImageResolutionScope,
    )
    from coexecution.optimization.config import OptimizationConfig
    from coexecution.optimization.session import GrootOptimizations

    if fast:
        processor.image_processor = Qwen2VLImageProcessorFast(
            patch_size=4,
            temporal_patch_size=2,
            merge_size=2,
            min_pixels=64,
            max_pixels=1024,
        )
    processor.chat_template = "{% for message in messages %}{% for item in message['content'] %}{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ item['text'] }}{% endif %}{% endfor %}{% endfor %}"
    groot = object.__new__(Gr00tN1d7Processor)
    collator = object.__new__(Gr00tN1d7DataCollator)
    groot.processor, collator.processor = processor, deepcopy(processor)
    groot._collator = collator
    groot.use_albumentations = True
    groot.image_target_size = [32, 32]
    groot.image_crop_size = [24, 24]
    groot.eval_image_transform = A.Compose(
        [
            A.CenterCrop(24, 24),
            A.Resize(32, 32, interpolation=cv2.INTER_AREA),
        ]
    )
    policy.model.config.streaming = streaming
    policy.model.action_head.config.streaming = streaming
    policy.processor, policy.collate_fn = groot, collator
    images = [
        Image.fromarray(value)
        for value in np.random.default_rng(43).integers(
            0, 256, (2, 32, 40, 3), dtype=np.uint8
        )
    ]

    def collate(observation):
        transformed, _ = apply_with_replay(groot.eval_image_transform, observation)
        encoded = groot._apply_vlm_processing(
            torch.stack(transformed).numpy(), "move left"
        )
        batch = collator([encoded])
        # Align the existing tokenizer fixture's special IDs with the tiny model.
        ids = batch["inputs"]["input_ids"]
        mapped = ids.clone()
        for old, new in ((2, 10), (3, 11), (4, 12), (5, 13), (6, 20), (7, 21)):
            mapped[ids == old] = new
        batch["inputs"].update(
            input_ids=mapped,
            state=torch.zeros(1, 1, 8),
            embodiment_id=torch.tensor([0]),
        )
        return batch, {}

    policy._collate_observation = collate
    engine = PreparedInferenceEngine(policy.model)
    original = groot.eval_image_transform
    native, _ = apply_with_replay(original, images)
    lengths = []
    for size in (32, 24, 16):
        with ImageResolutionScope(policy, ImageResolutionConfig(size)) as resolution:
            transformed, _ = apply_with_replay(groot.eval_image_transform, images)
            for reference, actual in zip(native, transformed, strict=True):
                expected = cv2.resize(
                    reference.numpy().transpose(1, 2, 0),
                    (size, size),
                    interpolation=cv2.INTER_AREA,
                )
                np.testing.assert_array_equal(
                    actual.numpy(), expected.transpose(2, 0, 1)
                )
            prepared = prepare_observation(policy, images, observation_id="eager")
            features = snapshot(engine.vlm(prepared.backbone))
            resolution.record_features(features)
            lengths.append(features["backbone_features"].shape[1])

            def sequence(inputs):
                torch.manual_seed(67)
                policy.model.action_head.reset_streaming_buffer()
                bootstrap = engine.full(
                    inputs, {"force_nonstreaming": True, "image_delay": 0}
                )
                if streaming:
                    policy.model.action_head.seed_streaming_buffer(
                        bootstrap["action_pred"], slide_steps=1
                    )
                options = (
                    {
                        "slide_steps": 1,
                        "num_inference_steps_per_call": 1,
                        "image_delay": 0,
                    }
                    if streaming
                    else {}
                )
                return snapshot(bootstrap), snapshot(engine.full(inputs, options))

            reference = sequence(prepared)
            config = OptimizationConfig(
                input_preparation=True,
                image_array_preparation=True,
                static_vlm=True,
                vlm_graph=True,
                dit_graph=True,
            )
            with GrootOptimizations(policy, config) as optimized:
                prepared = prepare_observation(policy, images, observation_id="graph")
                for _ in range(2):
                    for eager, captured in zip(
                        reference, sequence(prepared), strict=True
                    ):
                        exact(eager, captured)
                assert (
                    optimized.evidence()["static_vlm"]["vlm_graph"]["capture_count"]
                    == 1
                )
            evidence = resolution.evidence()
            assert evidence["actual_grid_thw"] == [[1, size // 4, size // 4]] * 2
            assert evidence["visual_tokens_per_image"] == [(size // 8) ** 2] * 2
        assert groot.eval_image_transform is original
    assert lengths[0] > lengths[1] > lengths[2]
