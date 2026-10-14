"""Scoped static Qwen3-VL metadata and optional complete VLM CUDA replay.

Only construction imports model dependencies. Images and every vision/language
layer execute for each refresh. Static token/grid metadata are private, bounded,
and value checked at CPU preparation; graph pixels are copied freshly on replay.
"""

import threading
from collections import Counter, OrderedDict
from contextlib import contextmanager
from types import SimpleNamespace

from coexecution.optimization.execution.guards import (
    HostGuardPlan,
    ModelTopologyGuard,
    tensor_version,
)
from coexecution.optimization.execution.patching import effective_callable_stamp

_CAPTURED_FIELDS = (
    "packed_weight",
    "weight_scale",
    "bias",
    "bits",
    "tactic",
    "shape_tactics",
    "linear",
    "silu_lut",
    "lut",
    "mode",
    "native",
    "scaled_dot_product_attention",
)
_GLOBAL_BINDINGS = (
    "create_causal_mask",
    "apply_rotary_pos_emb",
    "apply_rotary_pos_emb_vision",
    "eager_attention_forward",
    "ALL_ATTENTION_FUNCTIONS",
    "F",
)
_CAPTURED_METHODS = (
    "forward",
    "_deepstack_process",
    "get_image_features",
    "get_placeholder_mask",
    "fast_pos_embed_interpolate",
    "rot_pos_emb",
    "get_rope_index",
)


def _tensor_stamp(tensor):
    version = tensor_version(tensor)
    return (
        id(tensor),
        version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _captured_stamp(value, seen):
    """Small host stamps for packed constants/defaults; never scan tensor values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if id(value) in seen:
        return ("seen", id(value))
    seen.add(id(value))
    if all(hasattr(value, name) for name in ("shape", "dtype", "device", "stride")):
        return _tensor_stamp(value)
    if isinstance(value, (tuple, list)):
        return tuple(_captured_stamp(item, seen) for item in value)
    if isinstance(value, dict):
        return tuple((key, _captured_stamp(item, seen)) for key, item in value.items())
    function = getattr(value, "__func__", value)
    if hasattr(function, "__code__"):
        closure = getattr(function, "__closure__", ()) or ()
        return (
            id(function),
            function.__code__,
            _captured_stamp(getattr(value, "__self__", None), seen),
            _captured_stamp(function.__defaults__, seen),
            _captured_stamp(function.__kwdefaults__, seen),
            tuple(_captured_stamp(cell.cell_contents, seen) for cell in closure),
            tuple(
                (name, _captured_stamp(function.__globals__[name], seen))
                for name in _GLOBAL_BINDINGS
                if name in function.__code__.co_names and name in function.__globals__
            ),
        )
    if hasattr(value, "_global_mapping"):
        # The supported Qwen path uses exactly SDPA. Preserve its callable, not
        # just the unchanged string name in each module's configuration.
        return id(value), _captured_stamp(value["sdpa"], seen)
    # Packed/group controllers are captured in projection defaults. Transient
    # _input/_output caches intentionally never participate in this signature.
    return (
        id(value),
        tuple(
            (name, _captured_stamp(getattr(value, name), seen))
            for name in _CAPTURED_FIELDS
            if hasattr(value, name)
        ),
    )


def execution_stamp(model):
    """Captured weights, methods, SDPA choices and hooks, using host metadata only."""
    modules = []
    for name, module in model.named_modules():
        config = getattr(module, "config", None)
        modules.append(
            (
                name,
                id(module),
                module.training,
                getattr(config, "_attn_implementation", None),
                getattr(config, "use_cache", None),
                tuple(
                    (key, id(hook))
                    for key, hook in getattr(module, "_forward_hooks", {}).items()
                ),
                tuple(
                    (key, id(hook))
                    for key, hook in getattr(module, "_forward_pre_hooks", {}).items()
                ),
                tuple(
                    (method, _captured_stamp(getattr(module, method), set()))
                    for method in _CAPTURED_METHODS
                    if callable(getattr(module, method, None))
                ),
            )
        )
    return (
        tuple(modules),
        tuple(
            (name, _tensor_stamp(tensor)) for name, tensor in model.named_parameters()
        ),
        tuple((name, _tensor_stamp(tensor)) for name, tensor in model.named_buffers()),
    )


class CompiledExecutionGuard(HostGuardPlan):
    """The same frozen VLM state discovered once, with deduplicated hot checks."""

    def __init__(self, model, *, backend=None):
        super().__init__(
            model,
            methods=_CAPTURED_METHODS,
            fields=_CAPTURED_FIELDS,
            global_bindings=_GLOBAL_BINDINGS,
            backend=backend,
        )


def captured_tensors(model):
    """Keep registered and closure-owned packed constants alive on replay streams."""
    seen, result = set(), {}

    def visit(value):
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        if all(hasattr(value, name) for name in ("shape", "dtype", "device", "stride")):
            result[id(value)] = value
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        else:
            function = getattr(value, "__func__", value)
            if hasattr(function, "__code__"):
                visit(getattr(value, "__self__", None))
                visit(function.__defaults__)
                visit(function.__kwdefaults__)
                for cell in getattr(function, "__closure__", ()) or ():
                    visit(cell.cell_contents)
            else:
                for field in _CAPTURED_FIELDS:
                    if hasattr(value, field):
                        visit(getattr(value, field))

    for _, value in (*model.named_parameters(), *model.named_buffers()):
        visit(value)
    for _, module in model.named_modules():
        for method in _CAPTURED_METHODS:
            value = getattr(module, method, None)
            if callable(value):
                visit(value)
    return tuple(result.values())


def graph_bucket(metadata_key, pixels):
    """One helper per metadata/content and complete pixel layout signature."""
    return (
        metadata_key,
        tuple(pixels.shape),
        pixels.dtype,
        pixels.device,
        tuple(pixels.stride()),
    )


def check_policy_parameter(parameter, bf16):
    """Public direct scopes enforce the same device/dtype as session admission."""
    if parameter.device.type != "cuda" or parameter.dtype != bf16:
        raise ValueError("Static policy requires CUDA BF16 parameters")


class PolicyStaticS2:
    """One frozen serial eval CUDA Qwen3Backbone with bounded metadata/graphs.

    Close before changing components, parameters or attention implementations.
    Tracked changes require preparing again; .data/foreign untracked writes are
    outside this contract. A Python lock rejects overlapping submissions and
    teardown, but does not make arbitrary multi-stream callers safe.
    """

    def __init__(
        self,
        policy,
        *,
        cuda_graph=False,
        vision_rope=None,
        max_entries=32,
        guard_backend=None,
    ):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("Static policy max_entries must be a positive integer")
        if type(cuda_graph) is not bool:
            raise ValueError("Static policy cuda_graph must be a bool")
        import torch
        from transformers.feature_extraction_utils import BatchFeature
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

        from coexecution.static_s2 import METADATA_KEYS, StaticS2

        self.torch, self.BatchFeature, self.qwen = torch, BatchFeature, qwen
        self.metadata_keys = METADATA_KEYS
        self.policy, self.model = policy, policy.model
        self.backbone = self.model.backbone
        self.outer = self.backbone.model
        workload = SimpleNamespace(
            model=self.model,
            features_only=True,
            backbone_inputs=[],
            _pre_final_norm=None,
            slow=None,
        )
        self.core = StaticS2(workload, vision_rope=vision_rope)
        self.norm = self.core.inner.language_model.norm
        self.max_entries, self.cuda_graph = max_entries, cuda_graph
        self.guard_backend = guard_backend
        self.bindings, self.graphs = OrderedDict(), OrderedDict()
        self._cpu_metadata, self._resident_keys = {}, {}
        self.counts, self.restore = Counter(), []
        self._capture_hook = None
        self._entered = self._busy = False
        self.original_forward = self.original_prepare = None
        self.lock, self._close_owner = threading.Lock(), None
        self._graph_class = None
        self._execution_guard = None
        self._execution_epoch = 0
        self._policy_topology = None
        self._vlm_modules = ()

    def _check_model(self, *, rebuild=False):
        qwen, core = self.qwen, self.core
        if (
            self.policy.model is not self.model
            or self.model.backbone is not self.backbone
            or self.backbone.model is not self.outer
            or self.outer.model is not core.inner
            or core.inner.visual is not core.visual
            or core.inner.language_model.norm is not self.norm
        ):
            raise ValueError(
                "Static policy model was replaced; close and rebind the scope"
            )
        if (
            type(self.backbone).__name__ != "Qwen3Backbone"
            or type(self.outer) is not qwen.Qwen3VLForConditionalGeneration
            or type(core.inner) is not qwen.Qwen3VLModel
        ):
            raise ValueError("Static policy only supports Qwen3Backbone and Qwen3-VL")
        check_policy_parameter(next(self.model.parameters()), self.torch.bfloat16)
        if self._policy_topology is None:
            self._policy_topology = ModelTopologyGuard(
                self.model, backend=self.guard_backend
            )
            self._vlm_modules = tuple(core.inner.modules())
        if self._policy_topology.training_changed():
            raise ValueError("Static policy only supports eval/inference")
        if not self._policy_topology.unchanged():
            if not rebuild:
                raise ValueError("Static policy topology changed; prepare again")
            self._policy_topology = ModelTopologyGuard(
                self.model, backend=self.guard_backend
            )
            self._vlm_modules = tuple(core.inner.modules())
            if self._policy_topology.training_changed():
                raise ValueError("Static policy only supports eval/inference")
        if core.visual.pos_embed.weight.device.type != "cuda":
            raise ValueError("Static policy requires CUDA")
        if (
            core.visual.config._attn_implementation != "sdpa"
            or core.inner.language_model.config._attn_implementation != "sdpa"
        ):
            raise ValueError("Static policy only supports SDPA attention")
        from torch.nn.modules import module as torch_modules

        if (
            torch_modules._global_forward_pre_hooks
            or torch_modules._global_forward_hooks
            or any(
                module._forward_pre_hooks or module._forward_hooks
                for module in (self.outer, self.outer.lm_head)
            )
        ):
            raise ValueError("Static policy cannot skip outer model/lm_head hooks")
        if any(
            module._forward_pre_hooks or module._forward_hooks
            for module in (core.visual.pos_embed, core.visual.rotary_pos_emb)
        ):
            raise ValueError(
                "Static policy cannot cache vision metadata producer hooks"
            )
        allowed = set() if self._capture_hook is None else {self._capture_hook.id}
        if self.norm._forward_pre_hooks.keys() - allowed:
            raise ValueError(
                "Static policy cannot compose unknown final-norm pre-hooks"
            )
        if self.cuda_graph:
            for module in self._vlm_modules:
                own_pre_hooks = allowed if module is self.norm else set()
                if (
                    module._forward_hooks
                    or module._forward_pre_hooks.keys() - own_pre_hooks
                ):
                    raise ValueError("VLM graph cannot capture unknown submodule hooks")

    def _replace(self, target, name, replacement):
        existed = name in vars(target)
        self.restore.append((target, name, existed, getattr(target, name)))
        setattr(target, name, replacement)

    def _execution_stamp(self, *, rebuild=False):
        if self._execution_guard is None or not self._execution_guard.unchanged():
            if self._execution_guard is not None and not rebuild:
                raise ValueError("Static policy execution changed; prepare again")
            if self._execution_guard is not None or self._execution_epoch:
                # Defaults, closure contents or globals can change a metadata
                # producer without changing its function identity/code. A shared
                # epoch invalidates BOTH input and visual derived caches.
                self.core.metadata_epoch = getattr(self.core, "metadata_epoch", 0) + 1
            self._execution_guard = CompiledExecutionGuard(
                self.core.inner, backend=getattr(self, "guard_backend", None)
            )
            self._execution_guard.attribute(self.core, "vision_rope")
            self._execution_guard.attribute(self.core, "_causal_mask_original")
            self._execution_epoch += 1
        backend = self.torch.backends.cuda
        return (
            self.core._model_stamp(),
            self._execution_epoch,
            effective_callable_stamp(self.core.vision_rope),
            backend.flash_sdp_enabled(),
            backend.mem_efficient_sdp_enabled(),
            backend.math_sdp_enabled(),
            backend.cudnn_sdp_enabled(),
            backend.matmul.allow_tf32,
            self.torch.backends.cudnn.allow_tf32,
        )

    @contextmanager
    def _call_guard(self):
        if not self._entered:
            raise RuntimeError("Static policy requires an active scope")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Static policy supports one caller at a time")
        self._busy = True
        try:
            with self.torch.inference_mode():
                yield
        finally:
            self._busy = False
            self.lock.release()

    def _prune(self):
        while len(self.bindings) > self.max_entries:
            key, binding = self.bindings.popitem(last=False)
            self._drop_binding(key, binding)
            self.counts["binding_evictions"] += 1
        live_bindings = {id(binding) for binding in self.bindings.values()}
        self.core.registrations = {
            identity: value
            for identity, value in self.core.registrations.items()
            if identity in live_bindings
        }
        live_metadata = {
            id(value.metadata) for value in self.core.registrations.values()
        }
        self.core.metadata_cache = {
            key: value
            for key, value in self.core.metadata_cache.items()
            if id(value) in live_metadata
        }
        live_visual = {
            id(value.metadata.visual) for value in self.core.registrations.values()
        }
        self.core.vision_cache = {
            key: value
            for key, value in self.core.vision_cache.items()
            if id(value) in live_visual
        }
        for key in tuple(self.graphs):
            if key[0] not in self.bindings:
                self.graphs.pop(key)
                self.counts["graph_evictions"] += 1

    def _drop_binding(self, key, binding):
        # A same-content CPU reprepare may reuse the key after an owned GPU
        # tensor mutation. Its captured graph still closes over the old binding,
        # so remove every matching pixel signature before publishing replacement.
        for bucket in tuple(self.graphs):
            if bucket[0] == key:
                self.graphs.pop(bucket)
                self.counts["graph_evictions"] += 1
        self._cpu_metadata.pop(key, None)
        self._resident_keys.pop(
            tuple(id(binding[name]) for name in self.metadata_keys), None
        )
        self.core.registrations.pop(id(binding), None)

    def _prepare_input(self, batch):
        from coexecution.static_s2 import (
            metadata_content_key,
            ordinary_clone,
            tensor_stamp,
        )

        with self._call_guard():
            try:
                self._check_model(rebuild=True)
                prepared = self.original_prepare(batch)
                if not isinstance(prepared, (self.BatchFeature, dict)):
                    raise TypeError(
                        "Static policy prepare_input must return tensor metadata"
                    )
                if any(
                    prepared.get(key) is not None
                    for key in ("pixel_values_videos", "video_grid_thw")
                ):
                    raise ValueError("Static policy only supports image-only inputs")
                for name in self.metadata_keys:
                    if not self.torch.is_tensor(prepared.get(name)):
                        raise ValueError(
                            f"Static policy requires tensor CPU metadata: {name}"
                        )
                if prepared["attention_mask"].dtype not in (
                    self.torch.bool,
                    self.torch.int32,
                    self.torch.int64,
                ):
                    raise ValueError(
                        "Static policy requires a boolean/integer binary mask"
                    )
                self._check_pixels(prepared.get("pixel_values"), resident=False)
                device = self.core.visual.pos_embed.weight.device
                # Cache validity includes captured neural functions/weights, not
                # just metadata-producing vision embeddings. Changes reprepare.
                stamp = self._execution_stamp(rebuild=True)
                key = metadata_content_key(prepared, device, stamp)
                binding = self.bindings.get(key)
                if binding is not None:
                    registration = self.core.registrations.get(id(binding))
                    if registration is None or registration.stamps != tuple(
                        tensor_stamp(binding[name]) for name in self.metadata_keys
                    ):
                        self.bindings.pop(key)
                        self._drop_binding(key, binding)
                        self.core.counts["invalidations"] += 1
                        binding = None
                if binding is not None:
                    self.bindings.move_to_end(key)
                    self.counts["binding_cache_hits"] += 1
                else:
                    cpu_metadata = {
                        name: ordinary_clone(prepared[name])
                        for name in self.metadata_keys
                    }
                    with self.torch.inference_mode(False), self.torch.no_grad():
                        binding = {
                            name: tensor.to(device=device, copy=True)
                            for name, tensor in cpu_metadata.items()
                        }
                    # The core checks only metadata producers in its legacy path;
                    # policy bindings carry the complete captured execution stamp.
                    self.core._register(
                        binding, self.core._model_stamp(), cpu_metadata=cpu_metadata
                    )
                    self.bindings[key], self._cpu_metadata[key] = binding, cpu_metadata
                    self._resident_keys[
                        tuple(id(binding[name]) for name in self.metadata_keys)
                    ] = key
                    self._prune()
                return self.BatchFeature(data={**prepared, **binding})
            except BaseException:
                self._prune()
                raise

    def _check_pixels(self, pixels, *, resident=True):
        if (
            not self.torch.is_tensor(pixels)
            or not pixels.is_floating_point()
            or pixels.layout != self.torch.strided
            or not pixels.is_contiguous()
        ):
            raise ValueError("Static policy requires contiguous floating-point pixels")
        if resident and pixels.device != self.core.visual.pos_embed.weight.device:
            raise ValueError("Static policy requires resident CUDA pixels")

    def _graph_forward(self, key, binding, pixels, metadata):
        bucket = graph_bucket(key, pixels)
        graph = self.graphs.get(bucket)
        is_new = graph is None
        if is_new:
            if self._graph_class is None:
                from coexecution.kernels.common.graph import CudaGraphCall

                self._graph_class = CudaGraphCall

            def forward(fresh_pixels):
                return self.core.forward_inputs(binding, fresh_pixels)[
                    "backbone_features"
                ]

            constants = (
                *binding.values(),
                *metadata.tensors(),
                *captured_tensors(self.core.inner),
            )

            def retained():
                return constants

            graph = self._graph_class(forward, retained_tensors=retained)
        before = graph.captures, graph.replays
        try:
            from coexecution.optimization.execution.submission import cold_capture

            with cold_capture(graph.captures == 0):
                features = graph(pixels)
        except BaseException:
            self.counts["graph_failures"] += 1
            self.graphs.clear()
            # Keep the upstream eager entry usable and propagate the failure.
            # The outer SerialForwardBoundary adopts this restored target.
            self.backbone.forward = self.original_forward
            raise
        finally:
            self.counts["graph_captures"] += graph.captures - before[0]
            self.counts["graph_replays"] += graph.replays - before[1]
        self.graphs[bucket] = graph
        self.graphs.move_to_end(bucket)
        while len(self.graphs) > self.max_entries:
            self.graphs.popitem(last=False)
            self.counts["graph_evictions"] += 1
        return self.BatchFeature(
            {
                "backbone_features": features,
                "backbone_attention_mask": metadata.attention_mask,
                "image_mask": metadata.image_mask,
            }
        )

    def _forward(self, inputs):
        from coexecution.static_s2 import tensor_stamp

        with self._call_guard():
            rope_before = self.core.inner.rope_deltas
            try:
                self._check_model()
                if not isinstance(inputs, (self.BatchFeature, dict)) or any(
                    not self.torch.is_tensor(inputs.get(name))
                    for name in self.metadata_keys
                ):
                    raise ValueError("Static policy requires its prepared metadata")
                if any(
                    inputs.get(key) is not None
                    for key in ("pixel_values_videos", "video_grid_thw")
                ):
                    raise ValueError("Static policy only supports image-only inputs")
                key = self._resident_keys.get(
                    tuple(id(inputs[name]) for name in self.metadata_keys)
                )
                if key is None:
                    raise ValueError("Static policy requires its prepared metadata")
                binding = self.bindings[key]
                registration = self.core.registrations[id(binding)]
                if registration.stamps != tuple(
                    tensor_stamp(inputs[name]) for name in self.metadata_keys
                ):
                    raise ValueError(
                        "Static policy owned metadata has changed; prepare again"
                    )
                if key[1] != self._execution_stamp():
                    raise ValueError("Static policy execution changed; prepare again")
                pixels = inputs.get("pixel_values")
                self._check_pixels(pixels)
                self.bindings.move_to_end(key)
                self.counts["policy_forwards"] += 1
                metadata = registration.metadata
                stream = self.torch.cuda.current_stream(pixels.device)
                stream.wait_event(metadata.ready)
                for tensor in (*binding.values(), *metadata.tensors()):
                    tensor.record_stream(stream)
                pixels.record_stream(stream)
                outputs = (
                    self._graph_forward(key, binding, pixels, metadata)
                    if self.cuda_graph
                    else self.core.forward_inputs(binding, pixels)
                )
                return self.BatchFeature(
                    {
                        "backbone_features": outputs["backbone_features"],
                        "backbone_attention_mask": outputs[
                            "backbone_attention_mask"
                        ].clone(),
                        "image_mask": outputs["image_mask"].clone(),
                    }
                )
            except BaseException:
                self.core.inner.rope_deltas = rope_before
                raise
            finally:
                self.core.workload._pre_final_norm = None

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Static policy scopes cannot be entered twice")
        self._check_model(rebuild=True)
        self.original_forward, self.original_prepare = (
            self.backbone.forward,
            self.backbone.prepare_input,
        )
        self.graphs.clear()
        self._execution_guard = None
        try:
            self.core.__enter__()

            def capture(_module, inputs):
                self.core.workload._pre_final_norm = inputs[0]

            self._capture_hook = self.norm.register_forward_pre_hook(capture)

            def prepare(batch):
                return self._prepare_input(batch)

            prepare.__pir2_prepare_original__ = self.original_prepare
            self._replace(self.backbone, "prepare_input", prepare)
            self._replace(self.backbone, "forward", self._forward)
            self._entered = True
            return self
        except BaseException:
            self.close()
            raise

    @contextmanager
    def close_guard(self):
        owner = threading.get_ident()
        if not self._entered or self._close_owner == owner:
            yield
            return
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Cannot close static policy during an active call")
        self._close_owner = owner
        try:
            yield
        finally:
            self._close_owner = None
            self.lock.release()

    def close(self):
        with self.close_guard():
            self.graphs.clear()
            self._execution_guard = None
            for target, name, existed, original in reversed(self.restore):
                if existed:
                    setattr(target, name, original)
                elif name in vars(target):
                    delattr(target, name)
            self.restore.clear()
            if self._capture_hook is not None:
                self._capture_hook.remove()
                self._capture_hook = None
            self.core.__exit__()
            self.core.workload._pre_final_norm = None
            self._entered = self._busy = False

    def __exit__(self, *exception):
        self.close()

    def evidence(self):
        return {
            **self.core.evidence(),
            "policy_forwards": self.counts["policy_forwards"],
            "resident_bindings": len(self.bindings),
            "max_entries": self.max_entries,
            "binding_cache_hits": self.counts["binding_cache_hits"],
            "binding_evictions": self.counts["binding_evictions"],
            "active": self._entered,
            "metadata_validity": "owned CPU contents + tracked resident metadata + frozen captured execution",
            "guard_checks": 0
            if self._execution_guard is None
            else len(self._execution_guard.checks),
            "execution_epoch": self._execution_epoch,
            "vlm_graph": {
                "enabled": self.cuda_graph,
                "capture_count": self.counts["graph_captures"],
                "replay_count": self.counts["graph_replays"],
                "failure_count": self.counts["graph_failures"],
                "eviction_count": self.counts["graph_evictions"],
                "resident_signatures": len(self.graphs),
                "max_entries": self.max_entries,
                "scope": "whole features-only VLM; fresh pixels and fresh owned output",
            },
        }
