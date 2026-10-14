"""Opt-in whole S1 CUDA replay of the pinned, unchanged non-streaming Flow4.

Inputs/noise are refreshed on every call; Python scheduling, cache timestamps
and policy decoding remain outside. Importing admission helpers is GPU-free.
"""

import threading
from collections import Counter, OrderedDict
from contextlib import contextmanager, nullcontext
from types import MethodType

from coexecution.optimization.execution.guards import HostGuardPlan
from coexecution.optimization.execution.patching import clone_forward

OUTPUT_KEYS = ("action_pred", "backbone_features", "state_features")
BACKBONE_KEYS = ("backbone_features", "image_mask", "backbone_attention_mask")
ACTION_KEYS = ("state", "embodiment_id")
# The unseparated model prepare_input also passes VLM inputs and a training-only
# image_delay tensor. The pinned inference head never consumes these fields.
UNUSED_ACTION_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "image_delay",
}


def flow_options(head, options):
    """Reject unsupported behavior before allocating noise or entering capture."""
    if (
        head.config.streaming
        or head.config.action_horizon != 40
        or head.action_horizon != 40
        or head.num_inference_timesteps != 4
    ):
        raise ValueError("Action graph requires non-streaming Flow4 with H40")
    if options is not None and not isinstance(options, dict):
        raise ValueError("Action graph options must be a dict or None")
    options = {} if options is None else dict(options)
    if options.keys() - {"num_inference_timesteps", "image_delay"}:
        raise ValueError(
            "Action graph does not support streaming, inpaint or unknown options"
        )
    steps = options.get("num_inference_timesteps", 4)
    if type(steps) is not int or steps != 4:
        raise ValueError("Action graph requires unchanged Flow4 NFE=4")
    options["num_inference_timesteps"] = 4
    if "image_delay" in options:
        delay = options["image_delay"]
        if type(delay) is not int:
            raise ValueError("Action graph image_delay must be an integer")
        options["image_delay"] = max(0, min(delay, head.config.image_delay_max))
    return options


def action_graph_key(values, options, stream_id):
    return (
        stream_id,
        tuple(sorted(options.items())),
        tuple(
            None
            if value is None
            else (tuple(value.shape), value.dtype, value.device, tuple(value.stride()))
            for value in values
        ),
    )


def freeze_action_metadata(guard, modules, *, processor_types):
    """Bind the pinned normalization/attention metadata once, without recursion."""
    for module in modules:
        for name in (
            "normalized_shape",
            "compute_dtype",
            "spatial_norm",
            "group_norm",
            "norm_cross",
            "norm_q",
            "norm_k",
        ):
            if hasattr(module, name):
                guard.attribute(module, name)
        if hasattr(module, "processor"):
            processor = module.processor
            if type(processor) not in processor_types:
                raise ValueError(
                    "Action graph requires an admitted attention processor"
                )
            guard.attribute(module, "processor")
            guard.attribute(processor, "__call__", method=True)


class ExplicitNoise:
    """Local torch namespace for explicit noise and a capture-safe delay constant."""

    def __init__(self, torch):
        self.torch, self.noise, self.calls = torch, None, 0

    def __getattr__(self, name):
        return getattr(self.torch, name)

    @contextmanager
    def bind(self, noise):
        if self.noise is not None:
            raise RuntimeError("Action noise binding cannot overlap")
        self.noise, self.calls = noise, 0
        try:
            yield
            if self.calls != 1:
                raise RuntimeError("Pinned Flow must consume noise exactly once")
        finally:
            self.noise = None

    def randn(self, *, size, dtype, device):
        if self.noise is None:
            raise RuntimeError("Action noise requires an active binding")
        if self.calls:
            raise RuntimeError("Pinned Flow must consume noise exactly once")
        if (
            tuple(size) != tuple(self.noise.shape)
            or dtype != self.noise.dtype
            or device != self.noise.device
        ):
            raise ValueError("Pinned Flow requested incompatible action noise")
        self.calls += 1
        return self.noise

    def tensor(self, data, *, dtype, device):
        # The only reachable torch.tensor in the admitted Flow4 body constructs
        # [clamped_image_delay]. Fill on device instead of synchronously copying
        # a Python list from the host during CUDA graph capture.
        if (
            self.noise is None
            or type(data) is not list
            or len(data) != 1
            or type(data[0]) is not int
            or dtype != self.torch.long
            or device != self.noise.device
        ):
            raise ValueError("Unsupported action graph delay tensor construction")
        return self.torch.full((1,), data[0], dtype=dtype, device=device)


class ActionGraph:
    """Frozen serial CUDA action head. Close before changing weights or execution.

    Tracked mutations are rejected. Untracked .data/foreign writes are outside
    this contract. A failed capture never silently switches later calls to eager.
    Explicit close restores the original methods and hooks transactionally.
    """

    def __init__(self, policy, *, max_entries=8, guard_backend=None, kv_reuse=False):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("Action graph max_entries must be a positive integer")
        self.policy, self.model = policy, policy.model
        self.head = self.model.action_head
        self.max_entries = max_entries
        self.guard_backend = guard_backend
        if type(kv_reuse) is not bool:
            raise ValueError("Action K/V reuse must be a bool")
        self.kv_reuse = kv_reuse
        self._kv_scope = None
        self.graphs, self.counts = OrderedDict(), Counter()
        self.lock = threading.Lock()
        self._entered = False
        self._close_owner = None
        self._execution_guard = self._proof_hook = None
        self._original = None
        self._had_original = False
        self._constants = ()
        self._settings = None

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Action graph scope is already active")
        import torch
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
        from transformers.feature_extraction_utils import BatchFeature

        head = self.head
        flow_options(head, None)
        if type(head) is not Gr00tN1d7ActionHead:
            raise ValueError("Action graph only supports the pinned GR00T action head")
        for name in (
            "get_action",
            "_encode_features",
            "process_backbone_output",
            "get_action_with_features",
        ):
            if getattr(getattr(head, name), "__func__", None) is not getattr(
                Gr00tN1d7ActionHead, name
            ):
                raise ValueError(
                    "Action graph cannot capture an unknown action-head implementation"
                )
        parameter = next(head.parameters())
        if parameter.device.type != "cuda" or parameter.dtype != torch.bfloat16:
            raise ValueError("Action graph requires CUDA BF16 parameters")
        if any(module.training for module in head.modules()):
            raise ValueError("Action graph requires eval mode")
        self.torch, self.BatchFeature = torch, BatchFeature
        self.noise = ExplicitNoise(torch)
        self.flow = MethodType(
            clone_forward(
                head.get_action_with_features.__func__, {"torch": self.noise}
            ),
            head,
        )
        self._original, self._had_original = head.get_action, "get_action" in vars(head)

        def observe(_module, _args, _output):
            self.counts["python_dit_forward_count"] += 1

        observe.__pir2_execution_observer__ = True
        try:
            if self.kv_reuse:
                from gr00t.model.modules.dit import AlternateVLDiT

                from coexecution.optimization.groot.kv_reuse import FlowKVReuse

                if type(head.model) is not AlternateVLDiT:
                    raise ValueError("K/V reuse requires the pinned AlternateVLDiT")
                if self._kv_scope is None:
                    self._kv_scope = FlowKVReuse(head.model)
                self._kv_scope.__enter__()
            self._proof_hook = head.model.register_forward_hook(observe)
            head.get_action = self._call
            self._entered = True
            self._settings = self._execution_settings()
            self._check_execution()
            return self
        except BaseException:
            self.close()
            raise

    def _check_execution(self):
        if (
            self.policy.model is not self.model
            or self.model.action_head is not self.head
        ):
            raise ValueError("Action graph model was replaced; close and rebind")
        if self._execution_settings() != self._settings:
            raise ValueError(
                "Action graph execution settings changed; close and rebind"
            )
        if self._execution_guard is not None:
            if not self._execution_guard.unchanged():
                raise ValueError("Action graph execution changed; close and rebind")
            return
        from diffusers.models.attention_processor import AttnProcessor2_0
        from torch.nn.modules import module as torch_modules

        from coexecution.groot_pointwise import PreparedAttention
        from coexecution.optimization.groot.vlm import (
            _CAPTURED_FIELDS,
            _GLOBAL_BINDINGS,
            captured_tensors,
        )

        if (
            torch_modules._global_forward_pre_hooks
            or torch_modules._global_forward_hooks
        ):
            raise ValueError("Action graph cannot capture global hooks")
        observers = []
        for module in self.head.modules():
            if module._forward_pre_hooks or any(
                not getattr(hook, "__pir2_execution_observer__", False)
                for hook in module._forward_hooks.values()
            ):
                raise ValueError("Action graph cannot capture unknown submodule hooks")
            observers.extend(module._forward_hooks.values())
        self._execution_guard = HostGuardPlan(
            self.head,
            methods=(
                "forward",
                "get_action",
                "get_action_with_features",
                "_encode_features",
                "process_backbone_output",
                "prepare_attention_mask",
                "norm_encoder_hidden_states",
            ),
            fields=(
                *_CAPTURED_FIELDS,
                "action_horizon",
                "action_dim",
                "num_timestep_buckets",
                "num_inference_timesteps",
            ),
            global_bindings=_GLOBAL_BINDINGS,
            observer_hooks=observers,
            backend=self.guard_backend,
        )
        self._execution_guard.value(vars(self.head.config))
        for name in (
            "action_horizon",
            "action_dim",
            "num_timestep_buckets",
            "num_inference_timesteps",
        ):
            self._execution_guard.attribute(self.head, name)
        for name in ("_global_forward_pre_hooks", "_global_forward_hooks"):
            self._execution_guard.attribute(torch_modules, name)
        # Attention dimensions/scales, dropout probabilities and other public
        # scalar settings are constants in the captured module implementations.
        for module in self.head.modules():
            for name, value in vars(module).items():
                if not name.startswith("_") and isinstance(
                    value, (bool, int, float, str)
                ):
                    self._execution_guard.attribute(module, name)
        freeze_action_metadata(
            self._execution_guard,
            self.head.modules(),
            processor_types=(AttnProcessor2_0, PreparedAttention),
        )
        if self._kv_scope is not None:
            self._kv_scope.freeze(self._execution_guard)
        self._execution_guard.attribute(self, "kv_reuse")
        self._execution_guard.attribute(self, "_kv_scope")
        self._constants = captured_tensors(self.head)

    def _execution_settings(self):
        torch = self.torch
        backend = torch.backends.cuda
        return (
            backend.flash_sdp_enabled(),
            backend.mem_efficient_sdp_enabled(),
            backend.math_sdp_enabled(),
            backend.cudnn_sdp_enabled(),
            backend.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
        )

    def _inputs(self, backbone, action):
        torch, head = self.torch, self.head
        if (
            set(backbone) - set(BACKBONE_KEYS)
            or set(action) - set(ACTION_KEYS) - UNUSED_ACTION_KEYS
            or not set(ACTION_KEYS) <= set(action)
        ):
            raise ValueError("Action graph received unsupported backbone/action fields")
        if "backbone_features" not in backbone:
            raise ValueError("Action graph requires backbone_features")
        values = tuple(backbone.get(name) for name in BACKBONE_KEYS) + tuple(
            action[name] for name in ACTION_KEYS
        )
        parameter = next(head.parameters())
        for value in values:
            if value is not None and (
                not torch.is_tensor(value)
                or value.device != parameter.device
                or value.layout != torch.strided
                or any(
                    stride <= 0
                    for size, stride in zip(value.shape, value.stride())
                    if size > 1
                )
            ):
                raise ValueError(
                    "Action graph requires resident, non-overlapping CUDA tensors"
                )
        features, image_mask, attention_mask, state, category = values
        if (
            features.ndim != 3
            or features.dtype != parameter.dtype
            or features.shape[0] < 1
            or features.shape[1] < 1
            or state.dtype != parameter.dtype
            or state.ndim != 3
            or tuple(state.shape[:2])
            != (features.shape[0], head.config.state_history_length)
            or tuple(category.shape) != (features.shape[0],)
            or category.dtype != torch.long
        ):
            raise ValueError(
                "Action graph state/features/category shape or dtype is unsupported"
            )
        for value in (image_mask, attention_mask):
            if value is not None and tuple(value.shape) != tuple(features.shape[:2]):
                raise ValueError("Action graph masks must match the visual sequence")
        if head.config.use_alternate_vl_dit and (
            image_mask is None or attention_mask is None
        ):
            raise ValueError("Alternate VL DiT requires both dynamic masks")
        return values

    def _call(self, backbone_output, action_input, options=None):
        if not self._entered:
            raise RuntimeError("Action graph requires an active scope")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Action graph requires serial submissions")
        try:
            with self.torch.inference_mode():
                return self._run(backbone_output, action_input, options)
        finally:
            self.lock.release()

    def _run(self, backbone_output, action_input, options):
        options = flow_options(self.head, options)
        self._check_execution()
        values = self._inputs(backbone_output, action_input)
        features = values[0]
        stream = self.torch.cuda.current_stream(features.device)
        key = action_graph_key(values, options, stream.cuda_stream)
        graph = self.graphs.get(key)
        if graph is None:
            from coexecution.optimization.execution.tensor_graph import TensorGraphCall

            present = tuple(value is not None for value in values)

            def function(noise, *inputs):
                iterator = iter(inputs)
                unpacked = tuple(
                    next(iterator) if exists else None for exists in present
                )
                backbone = self.BatchFeature(
                    data={
                        name: value
                        for name, value in zip(BACKBONE_KEYS, unpacked[:3])
                        if value is not None
                    }
                )
                action = self.BatchFeature(data=dict(zip(ACTION_KEYS, unpacked[3:])))
                encoded = self.head._encode_features(backbone, action)
                before = self.counts["python_dit_forward_count"]
                with (
                    self.noise.bind(noise),
                    self._kv_scope.body()
                    if self._kv_scope is not None
                    else nullcontext(),
                ):
                    output = self.flow(
                        encoded.backbone_features,
                        encoded.state_features,
                        action.embodiment_id,
                        backbone,
                        action,
                        options,
                    )
                if self.counts["python_dit_forward_count"] - before != 4:
                    raise RuntimeError(
                        "Captured action graph did not execute exactly four DiT evaluations"
                    )
                self.counts["verified_flow4_body_count"] += 1
                return tuple(output[name] for name in OUTPUT_KEYS)

            graph = TensorGraphCall(function, retained_tensors=self._constants)
        # Exactly the original single call, including first capture. Warmup and
        # capture consume explicit noise, so they cannot advance the RNG again.
        noise = self.torch.randn(
            size=(
                features.shape[0],
                self.head.config.action_horizon,
                self.head.action_dim,
            ),
            dtype=features.dtype,
            device=features.device,
        )
        before = graph.captures, graph.replays
        try:
            outputs = graph(noise, *(value for value in values if value is not None))
        except BaseException:
            self.counts["failure_count"] += 1
            self.graphs.pop(key, None)
            raise
        finally:
            self.counts["capture_count"] += graph.captures - before[0]
            self.counts["replay_count"] += graph.replays - before[1]
        self.counts["logical_nfe_count"] += 4
        self.graphs[key] = graph
        self.graphs.move_to_end(key)
        while len(self.graphs) > self.max_entries:
            self.graphs.popitem(last=False)
            self.counts["eviction_count"] += 1
        return self.BatchFeature(data=dict(zip(OUTPUT_KEYS, outputs)))

    @contextmanager
    def close_guard(self):
        owner = threading.get_ident()
        if self._close_owner == owner:
            yield
            return
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Cannot close action graph during an active call")
        self._close_owner = owner
        try:
            yield
        finally:
            self._close_owner = None
            self.lock.release()

    def close(self):
        with self.close_guard():
            self.graphs.clear()
            if self._kv_scope is not None:
                self._kv_scope.close()
            if self._original is not None:
                if self._had_original:
                    self.head.get_action = self._original
                elif "get_action" in vars(self.head):
                    del self.head.get_action
                # This restoration belongs to one installation only. Repeated
                # close must not overwrite a method installed by a later scope.
                self._original = None
                self._had_original = False
            if self._proof_hook is not None:
                self._proof_hook.remove()
            self._proof_hook = self._execution_guard = None
            self._constants = ()
            self._entered = False

    def __exit__(self, *exception):
        self.close()

    def evidence(self):
        return {
            **{
                name: self.counts[name]
                for name in (
                    "capture_count",
                    "replay_count",
                    "failure_count",
                    "eviction_count",
                    "logical_nfe_count",
                    "python_dit_forward_count",
                    "verified_flow4_body_count",
                )
            },
            "resident_signatures": len(self.graphs),
            "max_entries": self.max_entries,
            "active": self._entered,
            "scope": "conditioning plus unchanged Flow4; explicit noise; owned outputs",
            "logical_nfe_per_replay": 4,
            "python_hooks": "warmup/capture only; graph replay bypasses Python hooks",
            "kv_reuse": (
                {"enabled": False}
                if self._kv_scope is None
                else self._kv_scope.evidence(replays=self.counts["replay_count"])
            ),
        }
