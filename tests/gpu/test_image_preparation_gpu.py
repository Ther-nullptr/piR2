"""Pinned deployment preprocessing only; no neural CPU computation.

PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_image_preparation_gpu.py
"""

import os
from types import SimpleNamespace

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip(
        "Set PIR2_GPU_TESTS=1 for pinned processor checks", allow_module_level=True
    )

from tests.gpu.test_tokenization_gpu import processor  # noqa: F401


@pytest.mark.parametrize("fast", [False, True])
def test_pinned_gr00t_conversation_and_actual_qwen_pixels_are_exact(processor, fast):  # noqa: F811
    import numpy as np
    import torch
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (
        Gr00tN1d7DataCollator,
        Gr00tN1d7Processor,
    )

    from coexecution.optimization.execution.images import ImageArrayPreparation

    if fast:
        from transformers import Qwen2VLImageProcessorFast

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
    groot.processor = collator.processor = processor
    groot._collator = collator
    policy = SimpleNamespace(
        model=SimpleNamespace(), processor=groot, collate_fn=collator
    )
    images = np.random.default_rng(123).integers(0, 256, (2, 3, 16, 24), dtype=np.uint8)
    reference = groot._apply_vlm_processing(images, "move left")
    reference_batch = collator([reference])["inputs"]
    original = groot._apply_vlm_processing
    with ImageArrayPreparation(policy):
        prepared = groot._apply_vlm_processing(images, "move left")
        assert prepared["vlm_content"]["text"] == reference["vlm_content"]["text"]
        for value in prepared["vlm_content"]["images"]:
            assert value.flags.owndata and value.flags.c_contiguous
        images.fill(0)
        actual = collator([prepared])["inputs"]
        assert actual.keys() == reference_batch.keys()
        for key in actual:
            torch.testing.assert_close(
                actual[key], reference_batch[key], rtol=0, atol=0
            )
    assert groot._apply_vlm_processing == original
