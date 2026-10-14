"""Scoped GR00T input separation without changing its collation or transfers.

Only entering the scope imports model dependencies. Tensor values stay owned by
the original preparation/transfer code; this adapter owns fresh input mappings.
"""

import inspect
import sys
import threading
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager

BACKBONE_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
ACTION_KEYS = ("state", "embodiment_id", "action", "action_mask", "image_delay")


def _function(method):
    return getattr(method, "__func__", method)


def _restore(module, name, original, had):
    if had:
        setattr(module, name, original)
    elif name in vars(module):
        delattr(module, name)


def _prepare_wraps(current, original):
    """Admit only an explicitly declared scoped chain ending at this wrapper."""
    seen = set()
    while current is not original:
        if id(current) in seen or not callable(current):
            return False
        seen.add(id(current))
        current = getattr(current, "__pir2_prepare_original__", None)
    return True


class InputPreparation:
    """One serial eval CUDA policy with fresh component input trees.

    The captured model method still runs its conditional collator and original
    dtype/device transfer. A scope lock covers that complete preparation call.
    Later scoped forward adapters may compose around the admitted consumers.
    Close the scope before replacing model weights, components or consumers.
    Unknown component or global hooks retain the original full input mapping for that
    component, so instrumentation may trade transfer reduction for compatibility.
    """

    def __init__(self, policy):
        self.policy = policy
        self.model = None
        self.backbone = None
        self.head = None
        self.stack = ExitStack()
        self.lock = threading.Lock()
        self.closed = True
        self._owner = None
        self._component_active = False
        self._close_owner = None
        self._methods = []
        self._consumers = []
        self._generation = 0
        self._fallbacks = {}

    def __enter__(self):
        if not self.closed:
            raise RuntimeError("Input preparation scope is already active")
        try:
            self._install()
        except BaseException:
            self.close()
            raise
        return self

    def _install(self):
        import torch
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
        from transformers.feature_extraction_utils import BatchFeature

        model = self.policy.model
        if type(model) is not Gr00tN1d7:
            raise ValueError("Unsupported input preparation model; expected Gr00tN1d7")
        backbone, head = model.backbone, model.action_head
        if type(backbone) is not Qwen3Backbone or type(head) is not Gr00tN1d7ActionHead:
            raise ValueError("Unsupported input preparation backbone or action head")
        self.model, self.backbone, self.head = model, backbone, head
        self._bf16 = torch.bfloat16
        self._validate_model()
        consumers = (
            (
                model,
                (
                    "prepare_input",
                    "forward",
                    "get_action",
                    "forward_vlm",
                    "forward_dit",
                    "forward_dit_action_only",
                ),
            ),
            (backbone, ("forward", "set_frozen_modules_to_eval_mode")),
            (
                head,
                (
                    "forward",
                    "get_action",
                    "_encode_features",
                    "get_action_with_features",
                    "_streaming_inference",
                ),
            ),
        )
        for module, names in consumers:
            for name in names:
                canonical = getattr(type(module), name, None)
                if not callable(canonical):
                    continue
                if _function(getattr(module, name)) is not _function(canonical):
                    raise ValueError(f"Unsupported input consumption override: {name}")
                self._consumers.append((type(module), name, canonical))
        self._generation += 1
        self._fallbacks = {}
        self.closed = False
        for module, keys in (
            (backbone, BACKBONE_KEYS),
            (head, ACTION_KEYS),
            (model, None),
        ):
            original = module.prepare_input
            had = "prepare_input" in vars(module)
            # Capture each restoration before its first mutation, including entry failures.
            self.stack.callback(_restore, module, "prepare_input", original, had)
            replacement = self._wrapper(module, original, keys, BatchFeature)
            self._methods.append((module, replacement))
            module.prepare_input = replacement

    def _validate_model(self):
        if any(module.training for module in (self.model, self.backbone, self.head)):
            raise ValueError("Input preparation requires eval-mode modules")
        parameter = next(self.model.parameters())
        if parameter.dtype != self._bf16:
            raise ValueError("Input preparation requires a BF16 model")
        if parameter.device.type != "cuda":
            raise ValueError("Input preparation requires CUDA")

    def _validate_call(self, generation):
        if self.closed or generation != self._generation:
            raise RuntimeError("Input preparation scope is closed")
        if (
            self.policy.model is not self.model
            or self.model.backbone is not self.backbone
            or self.model.action_head is not self.head
        ):
            raise RuntimeError("Input preparation model or component was replaced")
        self._validate_model()
        if any(
            not _prepare_wraps(module.prepare_input, wrapper)
            for module, wrapper in self._methods
        ):
            raise RuntimeError("Input preparation method changed inside the scope")
        if any(
            getattr(cls, name) is not original
            for cls, name, original in self._consumers
        ):
            raise RuntimeError("Input consumption method changed inside the scope")

    def _wrapper(self, module, original, keys, feature_class):
        input_name = next(iter(inspect.signature(original).parameters))
        generation = self._generation

        def prepare(*args, **kwargs):
            nested = keys is not None and self._owner == threading.get_ident()
            if nested and self._component_active:
                raise RuntimeError("Input preparation requires serial submissions")
            if not nested and not self.lock.acquire(blocking=False):
                raise RuntimeError("Input preparation requires serial submissions")
            try:
                self._validate_call(generation)
                if not nested:
                    self._owner = threading.get_ident()
                self._component_active = keys is not None
                if args:
                    batch = dict(args[0])
                    args = (batch, *args[1:])
                elif input_name in kwargs:
                    batch = dict(kwargs[input_name])
                    kwargs = {**kwargs, input_name: batch}
                else:
                    # Retain the original signature's error for missing arguments.
                    return original(*args, **kwargs)
                original_keys = set(batch)
                prepared = original(*args, **kwargs)
                if keys is None:
                    return prepared
                if not isinstance(prepared, Mapping):
                    raise TypeError("Unsupported prepared input; expected a mapping")
                unknown = set(prepared) - original_keys - set(keys)
                if unknown:
                    raise ValueError(
                        f"Unsupported prepared input fields: {sorted(unknown)}"
                    )
                hooks = [
                    kind
                    for kind in ("forward_hooks", "forward_pre_hooks")
                    if getattr(module, "_" + kind, {})
                ]
                registry = sys.modules.get("torch.nn.modules.module")
                hooks.extend(
                    "global_" + kind
                    for kind in ("forward_hooks", "forward_pre_hooks")
                    if getattr(registry, "_global_" + kind, {})
                )
                if hooks:
                    component = "backbone" if module is self.backbone else "action_head"
                    self._fallbacks.setdefault(component, set()).update(hooks)
                    return feature_class(data=dict(prepared))
                return feature_class(
                    data={key: prepared[key] for key in keys if key in prepared}
                )
            finally:
                self._component_active = False
                if not nested:
                    self._owner = None
                    self.lock.release()

        return prepare

    def evidence(self):
        return {
            "backbone_keys": list(BACKBONE_KEYS),
            "action_keys": list(ACTION_KEYS),
            "fallbacks": {key: sorted(value) for key, value in self._fallbacks.items()},
        }

    @contextmanager
    def close_guard(self):
        """Reject active preparation before any enclosing scope starts restoration."""
        owner = threading.get_ident()
        if self.closed or self._close_owner == owner:
            yield
            return
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Cannot close input preparation during an active call")
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
            self._methods.clear()
            self._consumers.clear()

    def __exit__(self, *exception):
        self.close()
