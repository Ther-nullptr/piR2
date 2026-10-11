"""Bounded token-only caching after the original Qwen image processing.

No tensor/model dependencies load at import time. Tokenizer internals are frozen
while the scope is active; visible configuration changes require close/re-entry.
The scope never retains images, grids, state or visual/text model features.
"""

import threading
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from copy import deepcopy

_OPTIONS = frozenset(
    (
        "add_special_tokens",
        "padding",
        "truncation",
        "max_length",
        "stride",
        "is_split_into_words",
        "pad_to_multiple_of",
        "padding_side",
        "return_tensors",
        "return_token_type_ids",
        "return_attention_mask",
        "return_overflowing_tokens",
        "return_special_tokens_mask",
        "return_offsets_mapping",
        "return_length",
        "verbose",
        "split_special_tokens",
    )
)
_TOKEN_FIELDS = frozenset(
    (
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "special_tokens_mask",
        "offset_mapping",
        "length",
        "overflowing_tokens",
        "num_truncated_tokens",
        "overflow_to_sample_mapping",
    )
)


def _freeze(value, *, configuration=False):
    if value is None or isinstance(value, (str, bool, int, float, bytes)):
        return type(value), value
    if isinstance(value, (tuple, list)):
        return type(value), tuple(
            _freeze(item, configuration=configuration) for item in value
        )
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return tuple(
            sorted(
                (key, _freeze(item, configuration=configuration))
                for key, item in value.items()
            )
        )
    # AddedToken properties affect normalization and special-token handling.
    if type(value).__name__ == "AddedToken":
        return "AddedToken", tuple(
            getattr(value, field)
            for field in (
                "content",
                "single_word",
                "lstrip",
                "rstrip",
                "normalized",
                "special",
            )
        )
    if configuration:
        return type(value), id(value)
    raise TypeError("Unsupported tokenization option")


def _function(method):
    return getattr(method, "__func__", method)


def _call_function(owner):
    return _function(owner.__call__) if callable(owner) else None


def _vocabulary_stamp(tokenizer):
    backend = getattr(tokenizer, "_tokenizer", None)
    size = getattr(backend, "get_vocab_size", None)
    added = getattr(backend, "get_added_tokens_decoder", None)
    if callable(size) and callable(added):
        # with_added_tokens=True builds a union of the complete vocabulary in the
        # pinned Rust tokenizer. Base size and compact added metadata avoid it.
        return size(with_added_tokens=False), tuple(
            sorted(
                (index, _freeze(token, configuration=True))
                for index, token in added().items()
            )
        )
    return len(tokenizer)


def _stamp(tokenizer):
    # Avoid serializing the vocabulary or backend on the per-observation path.
    # In-place backend vocabulary/normalizer mutations are forbidden in this scope.
    return (
        id(tokenizer),
        id(getattr(tokenizer, "_tokenizer", None)),
        _vocabulary_stamp(tokenizer),
        id(tokenizer.init_kwargs),
        _freeze(tokenizer.init_kwargs, configuration=True),
        _freeze(tokenizer.special_tokens_map_extended, configuration=True),
        tuple(
            _freeze(getattr(tokenizer, name, None), configuration=True)
            for name in (
                "padding_side",
                "truncation_side",
                "model_max_length",
                "pad_token_id",
                "bos_token_id",
                "eos_token_id",
                "unk_token_id",
                "sep_token_id",
                "cls_token_id",
                "mask_token_id",
                "split_special_tokens",
            )
        ),
    )


def _key(args, kwargs):
    if len(args) == 1 and "text" not in kwargs:
        text, options = args[0], kwargs
    elif not args and "text" in kwargs:
        text = kwargs["text"]
        options = {key: value for key, value in kwargs.items() if key != "text"}
    else:
        return None
    if not (
        isinstance(text, str)
        or isinstance(text, (tuple, list))
        and all(isinstance(item, str) for item in text)
    ):
        return None
    if set(options) - _OPTIONS or options.get("is_split_into_words", False):
        return None
    if options.get("return_tensors") not in (None, "pt", "np"):
        return None
    try:
        return _freeze(text), _freeze(options)
    except TypeError:
        return None


def _cpu_tokens(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, (tuple, list)):
        return all(_cpu_tokens(item) for item in value)
    if isinstance(value, Mapping):
        return all(_cpu_tokens(item) for item in value.values())
    device = getattr(value, "device", None)
    if device is not None:
        return getattr(device, "type", device) == "cpu"
    return type(value).__module__.startswith("numpy") and hasattr(value, "dtype")


def _restore(processor, original, had):
    if had:
        processor.tokenizer = original
    elif "tokenizer" in vars(processor):
        delattr(processor, "tokenizer")


class _TokenizerProxy:
    def __init__(self, scope, tokenizer, generation):
        object.__setattr__(self, "_scope", scope)
        object.__setattr__(self, "_original", tokenizer)
        object.__setattr__(self, "_generation", generation)

    def __getattr__(self, name):
        return getattr(self._original, name)

    def __setattr__(self, name, value):
        setattr(self._original, name, value)

    def __call__(self, *args, **kwargs):
        return self._scope._call(self._original, self._generation, args, kwargs)


class TokenizerCacheScope:
    """One serial policy, exact expanded text/options, owned CPU token results.

    Only the two existing collator processor tokenizer references are replaced.
    Original image processing, geometry metadata and conditional collation run
    every time. Unknown calls/results fall through. Close before replacing model,
    processors, consumers or tokenizer internals. Re-entry may retain valid owned
    tokens; changed tokenizer configuration is explicitly rebound on re-entry.
    """

    def __init__(self, policy, max_entries=16):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("Tokenizer max_entries must be a positive integer")
        self.policy, self.max_entries = policy, max_entries
        self.stack = ExitStack()
        self.lock = threading.Lock()
        self.closed = True
        self._close_owner = None
        self._generation = 0
        self._references, self._consumers, self._bindings = [], [], []
        self._stamps = {}
        self._entries = OrderedDict()
        self._hits = self._misses = self._evictions = self._bypasses = 0

    def __enter__(self):
        if not self.closed:
            raise RuntimeError("Tokenizer cache scope is already active")
        try:
            self._install()
        except BaseException:
            self.close()
            raise
        return self

    def _install(self):
        from transformers import PreTrainedTokenizerBase, Qwen3VLProcessor

        self._validate_references()
        model = self.policy.model
        references = [(self.policy, "model", model)]
        collators = []
        policy_processor = getattr(self.policy, "processor", None)
        if policy_processor is not None:
            references.append((self.policy, "processor", policy_processor))
            collator = policy_processor.collator
            references.append((policy_processor, "collator", collator))
            collators.append(collator)
            if hasattr(self.policy, "collate_fn"):
                if self.policy.collate_fn is not collator:
                    raise ValueError("Unsupported policy collator consumer")
                references.append((self.policy, "collate_fn", collator))
        if hasattr(model, "collator"):
            collators.append(model.collator)
            references.append((model, "collator", model.collator))
        bindings, consumers, seen = [], [], set()
        for collator in collators:
            processor = collator.processor
            references.append((collator, "processor", processor))
            if callable(collator):
                consumers.append((collator, _call_function(collator)))
            if id(processor) in seen:
                continue
            seen.add(id(processor))
            if type(processor) is not Qwen3VLProcessor:
                raise ValueError(
                    "Unsupported tokenizer processor; expected Qwen3VLProcessor"
                )
            tokenizer = processor.tokenizer
            if (
                not isinstance(tokenizer, PreTrainedTokenizerBase)
                or not tokenizer.is_fast
            ):
                raise ValueError(
                    "Tokenizer caching requires a Hugging Face fast tokenizer"
                )
            bindings.append((processor, tokenizer, "tokenizer" in vars(processor)))
            for owner in (processor, tokenizer):
                consumers.append((owner, _call_function(owner)))
        if not bindings:
            raise ValueError("Tokenizer caching selected no collator processors")
        stamps = {id(tokenizer): _stamp(tokenizer) for _, tokenizer, _ in bindings}
        if stamps != self._stamps:
            self._entries.clear()
        self._stamps = stamps
        self._references, self._consumers = references, consumers
        self._bindings = []
        self._generation += 1
        self.closed = False
        for processor, tokenizer, had in bindings:
            proxy = _TokenizerProxy(self, tokenizer, self._generation)
            self.stack.callback(_restore, processor, tokenizer, had)
            self._bindings.append((processor, proxy))
            processor.tokenizer = proxy

    def _validate_references(self):
        if any(
            getattr(owner, name, None) is not original
            for owner, name, original in self._references
        ):
            raise RuntimeError("Tokenizer model, collator or processor was replaced")
        if any(
            _call_function(owner) is not original for owner, original in self._consumers
        ):
            raise RuntimeError("Tokenization consumer changed inside the scope")

    def _call(self, tokenizer, generation, args, kwargs):
        if self.closed or generation != self._generation:
            raise RuntimeError("Tokenizer cache scope is closed")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Tokenizer caching requires serial submissions")
        try:
            if self.closed:
                raise RuntimeError("Tokenizer cache scope is closed")
            self._validate_references()
            if any(
                processor.tokenizer is not proxy for processor, proxy in self._bindings
            ):
                raise RuntimeError("Processor tokenizer changed inside the scope")
            if _stamp(tokenizer) != self._stamps[id(tokenizer)]:
                self._entries.clear()
                raise RuntimeError(
                    "Tokenizer configuration changed; close and re-enter the scope"
                )
            key = _key(args, kwargs)
            if key is None:
                self._bypasses += 1
                return tokenizer(*args, **kwargs)
            key = id(tokenizer), key
            if key in self._entries:
                self._hits += 1
                self._entries.move_to_end(key)
                return deepcopy(self._entries[key])
            result = tokenizer(*args, **kwargs)
            if (
                not isinstance(result, Mapping)
                or "input_ids" not in result
                or set(result) - _TOKEN_FIELDS
                or not _cpu_tokens(result)
            ):
                self._bypasses += 1
                return result
            owned = deepcopy(result)
            # A tokenizer call must not mutate the frozen effective configuration.
            if _stamp(tokenizer) != self._stamps[id(tokenizer)]:
                self._entries.clear()
                raise RuntimeError(
                    "Tokenizer configuration changed during tokenization"
                )
            self._misses += 1
            self._entries[key] = owned
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1
            return result
        finally:
            self.lock.release()

    def evidence(self):
        return {
            "processors": len(self._bindings),
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "bypasses": self._bypasses,
        }

    @contextmanager
    def close_guard(self):
        owner = threading.get_ident()
        if self.closed or self._close_owner == owner:
            yield
            return
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Cannot close tokenizer caching during an active call")
        self._close_owner = owner
        try:
            yield
        finally:
            self._close_owner = None
            self.lock.release()

    def close(self):
        with self.close_guard():
            self.closed = True
            self.stack.close()

    def __exit__(self, *exception):
        self.close()
