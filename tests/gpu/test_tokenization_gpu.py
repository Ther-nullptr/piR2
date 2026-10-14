"""Pinned Hugging Face tokenizer/image contracts; no neural CPU computation.

PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_tokenization_gpu.py
Full VLM/action equality is validated by the root CUDA gates separately.
"""

import importlib
import importlib.util
import os
from types import SimpleNamespace

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip(
        "Set PIR2_GPU_TESTS=1 for pinned tokenization checks", allow_module_level=True
    )

TOKENIZATION = "coexecution.optimization.execution.tokenization"


def test_tokenizer_cache_api():
    assert importlib.util.find_spec(TOKENIZATION) is not None, (
        "Tokenizer cache module is missing"
    )
    assert callable(importlib.import_module(TOKENIZATION).TokenizerCacheScope)


@pytest.fixture
def processor():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        Qwen2TokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    vocabulary = {
        "[unk]": 0,
        "[pad]": 1,
        "<|vision_start|>": 2,
        "<|vision_end|>": 3,
        "<|image_pad|>": 4,
        "<|video_pad|>": 5,
        "move": 6,
        "left": 7,
        "right": 8,
    }
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[unk]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = Qwen2TokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[unk]",
        pad_token="[pad]",
        padding_side="left",
        additional_special_tokens=[
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
        ],
    )
    image = Qwen2VLImageProcessor(
        patch_size=4,
        temporal_patch_size=2,
        merge_size=2,
        min_pixels=64,
        max_pixels=1024,
    )
    return Qwen3VLProcessor(
        image_processor=image, tokenizer=fast, video_processor=Qwen3VLVideoProcessor()
    )


def test_actual_batch_encoding_cpu_tokens_masks_and_encoding_metadata_are_owned(
    processor,
):
    import torch
    from transformers.tokenization_utils_base import BatchEncoding

    policy = SimpleNamespace(
        model=object(),
        processor=SimpleNamespace(collator=SimpleNamespace(processor=processor)),
    )
    original = processor.tokenizer
    options = {"padding": True, "return_tensors": "pt", "return_offsets_mapping": True}
    expected = original(["move left", "right"], **options)
    with importlib.import_module(TOKENIZATION).TokenizerCacheScope(policy) as cache:
        first = processor.tokenizer(["move left", "right"], **options)
        first["input_ids"].fill_(0)
        second = processor.tokenizer(["move left", "right"], **options)
        third = processor.tokenizer(["move left", "right"], **options)
        assert type(second) is BatchEncoding
        assert second.encodings is not third.encodings
        assert second.tokens(0) == expected.tokens(0)
        assert second.word_ids(0) == expected.word_ids(0)
        for name in expected:
            assert second[name].device.type == "cpu"
            assert second[name].data_ptr() != third[name].data_ptr()
            torch.testing.assert_close(second[name], expected[name], rtol=0, atol=0)
        assert cache.evidence()["hits"] == 2
    assert processor.tokenizer is original


def test_actual_qwen_processor_always_refreshes_pixels_and_same_count_grid(processor):
    import numpy as np
    import torch

    policy = SimpleNamespace(
        model=object(),
        processor=SimpleNamespace(collator=SimpleNamespace(processor=processor)),
    )
    text = ["move <|vision_start|><|image_pad|><|vision_end|>"]
    images = [
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.full((8, 32, 3), 255, dtype=np.uint8),
    ]
    expected = [
        processor(text=text, images=[image], padding=True, return_tensors="pt")
        for image in images
    ]
    with importlib.import_module(TOKENIZATION).TokenizerCacheScope(policy) as cache:
        first = processor(
            text=text, images=[images[0]], padding=True, return_tensors="pt"
        )
        second = processor(
            text=text, images=[images[1]], padding=True, return_tensors="pt"
        )
        assert torch.equal(first["input_ids"], second["input_ids"])
        assert not torch.equal(first["image_grid_thw"], second["image_grid_thw"])
        assert not torch.equal(first["pixel_values"], second["pixel_values"])
        assert first["pixel_values"].data_ptr() != second["pixel_values"].data_ptr()
        for actual, reference in zip((first, second), expected):
            for key in reference:
                torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
        assert cache.evidence()["hits"] == 1
