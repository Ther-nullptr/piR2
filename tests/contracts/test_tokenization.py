"""Pure token/cache and processor lifecycle contracts; no model computation."""

import importlib
import importlib.util
import subprocess
import sys
import threading
from collections import UserDict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOKENIZATION = "coexecution.optimization.execution.tokenization"


class BatchEncoding(UserDict):
    """Container and encoding metadata, without tensor or neural operations."""

    def __init__(self, data, encodings):
        super().__init__(data)
        self.encodings = encodings


@pytest.fixture
def interface(monkeypatch):
    calls = []

    class PreTrainedTokenizerBase:
        is_fast = True
        padding_side = "left"
        truncation_side = "right"
        model_max_length = 64
        pad_token_id = 0
        pad_token = "[pad]"
        split_special_tokens = False

        def __init__(self):
            self.init_kwargs = {"padding_side": "left", "model_max_length": 64}
            self.special_tokens_map_extended = {"pad_token": "[pad]"}
            self._tokenizer = object()
            self.vocabulary_size = 64
            self.pause = None

        def __len__(self):
            return self.vocabulary_size

        def __call__(self, text, **kwargs):
            calls.append((self, text, dict(kwargs)))
            if self.pause is not None:
                self.pause()
            texts = [text] if isinstance(text, str) else text
            ids = [[ord(char) for char in value] for value in texts]
            if kwargs.get("truncation"):
                ids = [
                    value[: kwargs.get("max_length", self.model_max_length)]
                    for value in ids
                ]
            return BatchEncoding(
                {
                    "input_ids": ids,
                    "attention_mask": [[1] * len(value) for value in ids],
                },
                [{"text": value} for value in texts],
            )

        def decode(self, ids):
            return "".join(chr(value) for value in ids)

    class Qwen3VLProcessor:
        def __init__(self):
            self.tokenizer = PreTrainedTokenizerBase()
            self.image_calls = []

        def __call__(self, *, text, images, **kwargs):
            # Model the pinned processor order: fresh image/grid, expansion, tokenizer.
            self.image_calls.append(images)
            pixels, grid = object(), [tuple(image["grid"]) for image in images]
            expanded = list(text)
            image_index = 0
            for index, value in enumerate(expanded):
                while "[image]" in value:
                    t, h, w = grid[image_index]
                    value = value.replace(
                        "[image]", "[placeholder]" * (t * h * w // 4), 1
                    )
                    image_index += 1
                expanded[index] = value.replace("[placeholder]", "[image]")
            tokenized = self.tokenizer(expanded, **kwargs)
            return {**tokenized, "pixel_values": pixels, "image_grid_thw": grid}

    class Collator:
        def __init__(self, processor):
            self.processor = processor
            self.calls = 0

        def __call__(self, features):
            self.calls += 1
            content = [feature["vlm_content"] for feature in features]
            return self.processor(
                text=[value["text"] for value in content],
                images=[image for value in content for image in value["images"]],
                padding=True,
            )

    policy_collator = Collator(Qwen3VLProcessor())
    model_collator = Collator(Qwen3VLProcessor())

    class Model:
        def __init__(self):
            self.collator = model_collator

        def prepare_input(self, batch):
            if "vlm_content" in batch:
                return self.collator(
                    [{"vlm_content": value} for value in batch["vlm_content"]]
                )
            return batch

    model = Model()
    processor = SimpleNamespace(collator=policy_collator)
    policy = SimpleNamespace(
        model=model, processor=processor, collate_fn=policy_collator
    )
    module = ModuleType("transformers")
    module.Qwen3VLProcessor = Qwen3VLProcessor
    module.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    monkeypatch.setitem(sys.modules, "transformers", module)
    return SimpleNamespace(
        policy=policy,
        model=model,
        policy_processor=policy_collator.processor,
        model_processor=model_collator.processor,
        policy_collator=policy_collator,
        model_collator=model_collator,
        calls=calls,
    )


def scope(policy, **kwargs):
    assert importlib.util.find_spec(TOKENIZATION) is not None, (
        "Tokenizer cache module is missing"
    )
    return importlib.import_module(TOKENIZATION).TokenizerCacheScope(policy, **kwargs)


def content(text="move [image]", grid=(1, 4, 4)):
    return {"text": text, "images": [{"grid": grid, "rgb": object()}]}


def test_tokenization_import_is_metadata_only():
    assert importlib.util.find_spec(TOKENIZATION) is not None, (
        "Tokenizer cache module is missing"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import sys
from {TOKENIZATION} import TokenizerCacheScope
assert callable(TokenizerCacheScope)
assert not any(name.startswith(("torch", "transformers", "gr00t", "triton"))
               for name in sys.modules)
""",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_exact_expanded_text_hits_after_fresh_image_processing(interface):
    processor = interface.policy_processor
    original = processor.tokenizer
    with scope(interface.policy) as cache:
        first = processor(
            text=["move [image]"], images=content()["images"], padding=True
        )
        second = processor(
            text=["move [image]"], images=content()["images"], padding=True
        )
        assert first["input_ids"] == second["input_ids"]
        assert first["input_ids"] is not second["input_ids"]
        assert first["pixel_values"] is not second["pixel_values"]
        assert first["image_grid_thw"] is not second["image_grid_thw"]
        assert len(processor.image_calls) == 2 and len(interface.calls) == 1
        assert interface.calls[0][1] == ["move " + "[image]" * 4]
        assert cache.evidence()["hits"] == 1
        assert (
            processor.tokenizer.decode(first["input_ids"][0]) == "move " + "[image]" * 4
        )
    assert processor.tokenizer is original


def test_same_token_count_grid_geometry_remains_fresh(interface):
    processor = interface.policy_processor
    with scope(interface.policy):
        first = processor(text=["[image]"], images=content(grid=(1, 4, 4))["images"])
        second = processor(text=["[image]"], images=content(grid=(1, 2, 8))["images"])
        assert first["input_ids"] == second["input_ids"]
        assert first["image_grid_thw"] == [(1, 4, 4)]
        assert second["image_grid_thw"] == [(1, 2, 8)]
        assert first["pixel_values"] is not second["pixel_values"]
        assert len(interface.calls) == 1


def test_owned_encodings_and_nested_containers_survive_caller_mutation(interface):
    with scope(interface.policy):
        tokenizer = interface.policy_processor.tokenizer
        first = tokenizer(["ab", "c"], padding=True)
        first["input_ids"][0][0] = -1
        first["attention_mask"].append([0])
        first.encodings[0]["text"] = "changed"
        second = tokenizer(["ab", "c"], padding=True)
        third = tokenizer(["ab", "c"], padding=True)
        assert type(second) is BatchEncoding
        assert second["input_ids"] == [[97, 98], [99]]
        assert second["attention_mask"] == [[1, 1], [1]]
        assert second.encodings == [{"text": "ab"}, {"text": "c"}]
        assert second.encodings is not third.encodings
        second.encodings[0]["text"] = "mutated again"
        assert third.encodings[0]["text"] == "ab"
        assert len(interface.calls) == 1


def test_fast_backend_stamp_avoids_union_vocabulary_count_and_tracks_added_tokens(
    interface,
):
    class AddedToken:
        content = "[extra]"
        single_word = lstrip = rstrip = normalized = special = False

    class FastBackend:
        def __init__(self):
            self.added = {64000: AddedToken()}
            self.base_count_calls = 0

        def get_vocab_size(self, *, with_added_tokens):
            assert not with_added_tokens, "Expensive union vocabulary count was called"
            self.base_count_calls += 1
            return 64000

        def get_added_tokens_decoder(self):
            return dict(self.added)

    class FastTokenizer(type(interface.policy_processor.tokenizer)):
        def __len__(self):
            return self._tokenizer.get_vocab_size(with_added_tokens=True)

    tokenizer = FastTokenizer()
    backend = FastBackend()
    tokenizer._tokenizer = backend
    interface.policy_processor.tokenizer = tokenizer
    with scope(interface.policy) as cache:
        proxy = interface.policy_processor.tokenizer
        proxy(["a"])
        proxy(["a"])
        assert cache.evidence()["hits"] == 1
        assert backend.base_count_calls > 0
        backend.added[64000].content = "[changed]"
        with pytest.raises(RuntimeError, match="configuration.*changed"):
            proxy(["a"])
        assert cache.evidence()["entries"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"padding": "max_length", "max_length": 16},
        {"padding": True},
        {"truncation": True, "max_length": 2},
        {"add_special_tokens": False},
        {"return_attention_mask": False},
        {"return_offsets_mapping": True},
    ],
)
def test_effective_tokenization_kwargs_are_part_of_the_key(interface, kwargs):
    with scope(interface.policy):
        tokenizer = interface.policy_processor.tokenizer
        tokenizer(["abcdef"])
        tokenizer(["abcdef"], **kwargs)
        tokenizer(["abcdef"], **dict(reversed(tuple(kwargs.items()))))
        assert len(interface.calls) == 2


def test_batch_order_and_expanded_placeholder_counts_are_part_of_the_key(interface):
    with scope(interface.policy):
        tokenizer = interface.policy_processor.tokenizer
        tokenizer(["a", "b"])
        tokenizer(["b", "a"])
        processor = interface.policy_processor
        processor(text=["[image]"], images=content(grid=(1, 4, 4))["images"])
        processor(text=["[image]"], images=content(grid=(1, 4, 8))["images"])
        assert len(interface.calls) == 4


def test_lru_eviction_is_bounded_and_recent_hits_keep_the_entry(interface):
    with scope(interface.policy, max_entries=2) as cache:
        tokenizer = interface.policy_processor.tokenizer
        for text in ("a", "b", "a", "c", "a", "b"):
            tokenizer([text])
        assert len(interface.calls) == 4
        assert cache.evidence()["entries"] == 2
        assert cache.evidence()["evictions"] == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unknown_option": True},
        {"text_target": "target"},
        {"is_split_into_words": True},
    ],
)
def test_unsupported_tokenizer_calls_fall_through_without_memoization(
    interface, kwargs
):
    with scope(interface.policy) as cache:
        tokenizer = interface.policy_processor.tokenizer
        tokenizer(["a"], **kwargs)
        tokenizer(["a"], **kwargs)
        assert len(interface.calls) == 2
        assert cache.evidence()["entries"] == 0


def test_two_collator_roles_stay_conditional_and_aliases_are_deduplicated(interface):
    original = interface.policy_processor.tokenizer
    interface.model.collator = interface.policy_collator
    with scope(interface.policy) as cache:
        first = interface.policy.collate_fn([{"vlm_content": content()}])
        assert interface.model.prepare_input(first) is first
        interface.model.prepare_input({"vlm_content": [content()]})
        assert interface.policy_collator.calls == 2
        assert len(interface.calls) == 1
        assert cache.evidence()["processors"] == 1
    assert interface.policy_processor.tokenizer is original


def test_distinct_collator_processors_are_each_scoped(interface):
    originals = (
        interface.policy_processor.tokenizer,
        interface.model_processor.tokenizer,
    )
    with scope(interface.policy) as cache:
        for _ in range(2):
            interface.policy.collate_fn([{"vlm_content": content()}])
            interface.model.prepare_input({"vlm_content": [content()]})
        assert len(interface.calls) == 2
        assert cache.evidence()["processors"] == 2
    assert (
        interface.policy_processor.tokenizer,
        interface.model_processor.tokenizer,
    ) == originals


@pytest.mark.parametrize(
    "change",
    ["padding", "max_length", "special_token", "vocabulary", "backend", "init_kwargs"],
)
def test_detected_tokenizer_configuration_changes_clear_and_reject_until_reentry(
    interface, change
):
    adapter = scope(interface.policy)
    original = interface.policy_processor.tokenizer
    with adapter:
        proxy = interface.policy_processor.tokenizer
        proxy(["a"])
        if change == "padding":
            proxy.padding_side = "right"
        elif change == "max_length":
            original.model_max_length = 32
        elif change == "special_token":
            original.special_tokens_map_extended["pad_token"] = "[newpad]"
        elif change == "vocabulary":
            original.vocabulary_size += 1
        elif change == "backend":
            original._tokenizer = object()
        else:
            original.init_kwargs["model_max_length"] = 32
        with pytest.raises(RuntimeError, match="configuration.*changed"):
            proxy(["a"])
        assert adapter.evidence()["entries"] == 0
    with adapter:
        interface.policy_processor.tokenizer(["a"])
    assert len(interface.calls) == 2


@pytest.mark.parametrize(
    "change", ["model", "collator", "processor", "tokenizer", "consumer"]
)
def test_model_or_tokenization_consumer_replacement_is_rejected(
    interface, change, monkeypatch
):
    with scope(interface.policy):
        proxy = interface.policy_processor.tokenizer
        if change == "model":
            interface.policy.model = object()
        elif change == "collator":
            interface.policy.processor.collator = object()
        elif change == "processor":
            interface.policy_collator.processor = object()
        elif change == "tokenizer":
            interface.policy_processor.tokenizer = object()
        else:
            monkeypatch.setattr(
                type(interface.policy_processor), "__call__", lambda *args: None
            )
        with pytest.raises(RuntimeError, match="changed|replaced"):
            proxy(["a"])


def test_reentry_retains_valid_owned_tokens_but_old_proxy_stays_closed(interface):
    adapter = scope(interface.policy)
    with adapter:
        old = interface.policy_processor.tokenizer
        first = old(["a"])
    with pytest.raises(RuntimeError, match="closed"):
        old(["a"])
    with adapter:
        second = interface.policy_processor.tokenizer(["a"])
        assert first == second and first is not second
        with pytest.raises(RuntimeError, match="closed"):
            old(["a"])
    assert len(interface.calls) == 1


def test_close_and_overlap_are_rejected_before_restoration(interface):
    started, release = threading.Event(), threading.Event()
    original = interface.policy_processor.tokenizer
    errors = []

    def pause():
        started.set()
        assert release.wait(5)

    original.pause = pause
    adapter = scope(interface.policy)
    with adapter:
        proxy = interface.policy_processor.tokenizer

        def submit():
            try:
                proxy(["a"])
            except (RuntimeError, AssertionError) as error:
                errors.append(error)

        worker = threading.Thread(target=submit)
        worker.start()
        try:
            assert started.wait(5)
            with pytest.raises(RuntimeError, match="serial"):
                proxy(["a"])
            with pytest.raises(RuntimeError, match="active call"):
                adapter.close()
            assert interface.policy_processor.tokenizer is proxy
        finally:
            release.set()
            worker.join(5)
        assert not errors
    assert interface.policy_processor.tokenizer is original


def test_scope_body_failure_restores_both_tokenizers(interface):
    originals = (
        interface.policy_processor.tokenizer,
        interface.model_processor.tokenizer,
    )
    with pytest.raises(RuntimeError, match="body"), scope(interface.policy):
        interface.policy_processor.tokenizer(["a"])
        raise RuntimeError("body failure")
    assert (
        interface.policy_processor.tokenizer,
        interface.model_processor.tokenizer,
    ) == originals


@pytest.mark.parametrize("max_entries", [0, -1, True, 1.5])
def test_cache_bound_must_be_a_positive_integer(interface, max_entries):
    with pytest.raises(ValueError, match="positive integer"):
        scope(interface.policy, max_entries=max_entries)
