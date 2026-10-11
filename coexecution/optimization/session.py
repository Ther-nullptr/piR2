"""Transactional installation, evidence and restoration of GR00T optimizations.

GPU dependencies are loaded only when an enabled configuration enters its scope.
"""

import hashlib
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from pathlib import Path

from coexecution.optimization.capabilities import preflight_backend
from coexecution.optimization.config import OptimizationConfig, selected_component
from coexecution.optimization.execution.cache import (
    SerialForwardBoundary,
    clear_floating_projection_cache,
)
from coexecution.optimization.tactics import apply_tactics, read_json


class GrootOptimizations:
    """One serial, frozen BF16 CUDA policy; close before replacing its weights.

    This scope neither selects a checkpoint nor changes sampling, observations
    or the policy's VLM cache. No optimization is enabled by default.
    Vision Conv3D layout conversion requires the explicit vision_channels_last flag.
    """

    def __init__(self, policy, config=None):
        self.policy = policy
        self.config = OptimizationConfig() if config is None else config
        self.stack = ExitStack()
        self.runtime_stack = ExitStack()
        self.coverage = {}
        self.backend = {}
        self.boundary = None
        self.preparation = None
        self.tokenizer_cache = None
        self.graphs = None
        self.static_vlm = None
        self.fusion = None
        self._active = False
        self._model = None
        self._runtime_refs = []
        self._clear_cache = lambda: None
        self._roots = []
        self._resets = []

    def __enter__(self):
        if self._active:
            raise RuntimeError("Optimization session is already active")
        self._active = True
        if not self.config.enabled:
            return self
        try:
            self._install()
        except BaseException:
            self.close()
            raise
        return self

    def _install(self):
        import torch

        model = self.policy.model
        if model.training or next(model.parameters()).dtype != torch.bfloat16:
            raise ValueError("Optimizations require an eval-mode BF16 model")
        if next(model.parameters()).device.type != "cuda":
            raise ValueError("Optimizations require CUDA")
        self.config.validate_variant(
            "pir2" if getattr(model.config, "streaming", False) else "flow"
        )
        expanded = self.config.scope == "all"
        if self.config.precision != "bf16":
            from coexecution.optimization.groot.projections import inventory

            sites = inventory(model, expanded=expanded)[0]
            recorded = read_json(self.config.coverage, {}).get("linears", {})
            candidates = recorded if self.config.coverage is not None else sites
            executed = {
                name: value
                for name, value in candidates.items()
                if selected_component(name, self.config.component)
            }
            categories = {
                name: (module, self.config.category_id)
                for name, module in model.named_modules()
                if type(module).__name__ == "CategorySpecificLinear"
                and self.config.category_id is not None
                and selected_component(name, self.config.component)
            }
            if not sites.keys() & executed.keys() and not categories:
                raise ValueError("Quantization selected no matching projections")
        # Validate the selected backend/profile before modifying model methods
        # or weight layouts. Dependency loading is allowed, patching is not.
        self.backend = preflight_backend(self.config, next(model.parameters()).device)
        self._bind_runtime_model(model)

        def clear_cache():
            pass

        if self.config.vision_channels_last:
            proj = model.backbone.model.visual.patch_embed.proj
            self.stack.callback(setattr, proj.weight, "data", proj.weight.data)
            proj.weight.data = proj.weight.data.contiguous(
                memory_format=torch.channels_last_3d
            )
            hook = proj.register_forward_pre_hook(
                lambda _module, inputs: (
                    inputs[0].contiguous(memory_format=torch.channels_last_3d),
                )
            )
            self.stack.callback(hook.remove)
        if self.config.fusion:
            from coexecution.groot_fusion import NonGemmAdapters
            from coexecution.groot_pointwise import PointwiseFusion

            fusion = NonGemmAdapters(self.policy, "native")
            self.fusion = fusion
            self.stack.callback(fusion.close)
            fusion.set("all")
            pointwise = PointwiseFusion(self.policy)
            self.stack.callback(pointwise.close)
            pointwise.set("dit")
        if self.config.precision in ("fp8", "fp4"):
            from coexecution.optimization.groot.floating import FloatingProjections

            floating = FloatingProjections(
                model,
                executed,
                expanded=expanded,
                precision=self.config.precision,
                fast_fp8=self.config.fp8_fast_quant,
                shared_inputs=self.config.fp_shared_inputs,
                grouped=self.config.fp_grouped,
                swiglu=self.config.fp_swiglu,
            )
            self.stack.callback(floating.close)
            self.coverage = floating.coverage
            self.coverage["component"] = self.config.component

            def clear_cache():
                clear_floating_projection_cache(floating.groups)

            if self.config.norm_modulation_quant:
                from coexecution.optimization.groot.floating_connections import (
                    FloatingConnections,
                )

                connections = FloatingConnections(
                    self.policy, floating, residual=self.config.residual_norm_quant
                )
                self.stack.callback(connections.close)
        elif self.config.precision != "bf16":
            from coexecution.optimization.groot.integer import TransformerINT
            from coexecution.optimization.groot.integer_connections import (
                AdditionalConnections,
            )

            controller = TransformerINT(
                model,
                executed,
                native=True,
                expanded=expanded,
                condition_group=self.config.group_conditioning,
            )
            self.stack.callback(controller.close)
            extra = AdditionalConnections(self.policy, controller, categories)
            self.stack.callback(extra.close)
            apply_tactics(
                controller, extra, self.config, recorded, backend_evidence=self.backend
            )
            controller.set(self.config.precision)
            extra.set(
                self.config.precision,
                expanded=expanded,
                fuse=self.config.fusion,
                norm_quant=self.config.norm_modulation_quant,
                residual_quant=self.config.residual_norm_quant,
            )
            self.coverage = {
                "linears": sorted(controller.sites),
                "category_linears": sorted(categories),
                "scope": self.config.scope,
                "component": self.config.component,
                "category_id": self.config.category_id,
                "projection_groups": controller.groups,
                "tactic_policy": "Exact runtime shape, then explicit site/group, then 0",
            }
            clear_cache = controller.clear_projection_cache
        if self.config.dit_graph:
            from coexecution.optimization.execution.graph import DitGraphs

            graphs = DitGraphs(model.action_head.model)
            self.graphs = graphs
            self.stack.callback(graphs.close)
            graphs.set(self.config.precision)
        self._clear_cache = clear_cache
        self._install_runtime_adapters()
        self._install_boundary()

    def _bind_runtime_model(self, model):
        if self._model is not None:
            self._validate_runtime_model()
            return
        self._model = model
        head = getattr(model, "action_head", None)
        self._runtime_refs = [(self.policy, "model", model)]
        for owner, name in (
            (model, "backbone"),
            (model, "action_head"),
            (head, "model"),
        ):
            if owner is not None and hasattr(owner, name):
                self._runtime_refs.append((owner, name, getattr(owner, name)))
        self._roots = [
            root
            for root in (
                getattr(model, "backbone", None),
                getattr(head, "vl_self_attention", None),
                getattr(head, "model", None),
            )
            if root is not None
        ]
        self._resets = (
            [head] if callable(getattr(head, "reset_streaming_buffer", None)) else []
        )

    def _validate_runtime_model(self):
        if any(
            getattr(owner, name, None) is not original
            for owner, name, original in self._runtime_refs
        ):
            raise RuntimeError("Optimization model or component was replaced")

    def _admit_runtime_model(self):
        if self._model is not None:
            self._validate_runtime_model()
            return
        import torch

        model = self.policy.model
        parameter = next(model.parameters())
        if model.training or parameter.dtype != torch.bfloat16:
            raise ValueError("Optimizations require an eval-mode BF16 model")
        if parameter.device.type != "cuda":
            raise ValueError("Optimizations require CUDA")
        self.config.validate_variant(
            "pir2" if getattr(model.config, "streaming", False) else "flow"
        )
        self.backend = preflight_backend(self.config, parameter.device)
        self._bind_runtime_model(model)

    def _install_runtime_adapters(self):
        if self.config.input_preparation:
            from coexecution.optimization.execution.preparation import InputPreparation

            if self.preparation is None:
                self.preparation = InputPreparation(self.policy)
            self.runtime_stack.callback(self.preparation.close)
            self.preparation.__enter__()
        if self.config.tokenizer_cache:
            from coexecution.optimization.execution.tokenization import (
                TokenizerCacheScope,
            )

            if self.tokenizer_cache is None:
                self.tokenizer_cache = TokenizerCacheScope(self.policy)
            self.runtime_stack.callback(self.tokenizer_cache.close)
            self.tokenizer_cache.__enter__()
        if self.config.static_vlm:
            from coexecution.optimization.groot.vlm import PolicyStaticS2

            if (
                self.static_vlm is None
                or self.static_vlm.cuda_graph != self.config.vlm_graph
            ):
                self.static_vlm = PolicyStaticS2(
                    self.policy,
                    cuda_graph=self.config.vlm_graph,
                    vision_rope=None if self.fusion is None else self.fusion.vision,
                )
            self.runtime_stack.callback(self.static_vlm.close)
            self.static_vlm.__enter__()

    def _install_boundary(self):
        self.boundary = None
        if self.config.enabled:
            self.boundary = SerialForwardBoundary(
                self._roots, self._clear_cache, reset_modules=self._resets
            )
            self.runtime_stack.callback(self.boundary.close)

    @contextmanager
    def _runtime_guard(self):
        with ExitStack() as guards:
            for adapter in (self.preparation, self.tokenizer_cache, self.static_vlm):
                if adapter is not None:
                    guards.enter_context(adapter.close_guard())
            yield

    def set_runtime(
        self,
        *,
        input_preparation=None,
        tokenizer_cache=None,
        static_vlm=None,
        vlm_graph=None,
    ):
        """Switch common runtime scopes; retain fixed precision, layouts and DiT graphs."""
        if not self._active:
            raise RuntimeError(
                "Runtime switching requires an active optimization session"
            )
        for name, value in (
            ("input_preparation", input_preparation),
            ("tokenizer_cache", tokenizer_cache),
            ("static_vlm", static_vlm),
            ("vlm_graph", vlm_graph),
        ):
            if value is not None and type(value) is not bool:
                raise ValueError(f"Runtime {name} must be a bool or None")
        config = replace(
            self.config,
            **{
                name: value
                for name, value in (
                    ("input_preparation", input_preparation),
                    ("tokenizer_cache", tokenizer_cache),
                    ("static_vlm", static_vlm),
                    ("vlm_graph", vlm_graph),
                )
                if value is not None
            },
        )
        with self._runtime_guard():
            self._validate_runtime_model()
            if config == self.config:
                return self
            if config.enabled:
                self._admit_runtime_model()
            previous = self.config
            previous_boundary = self.boundary
            # Active-root rejection must happen before any adapter restoration.
            try:
                if previous_boundary is not None:
                    previous_boundary.close()
                self.runtime_stack.close()
                self.runtime_stack = ExitStack()
                self.config = config
                self._install_runtime_adapters()
                self._install_boundary()
            except BaseException:
                if previous_boundary is not None and not previous_boundary.closed:
                    # A live root rejected close without changing any methods.
                    raise
                try:
                    self.runtime_stack.close()
                    self.runtime_stack = ExitStack()
                    self.config = previous
                    self._install_runtime_adapters()
                    self._install_boundary()
                except BaseException as rollback:
                    self.close()
                    raise RuntimeError(
                        "Runtime rollback failed; optimization session closed"
                    ) from rollback
                raise
        return self

    def evidence(self):
        config = asdict(self.config)
        for key in ("coverage", "tactics", "group_tactics", "shape_tactics"):
            path = config[key]
            config[key] = (
                None
                if path is None
                else {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            )
        return {
            "config": config,
            "coverage": self.coverage,
            "backend": self.backend,
            "input_preparation": (
                None
                if self.preparation is None or not self.config.input_preparation
                else self.preparation.evidence()
            ),
            "tokenizer_cache": (
                None
                if self.tokenizer_cache is None or not self.config.tokenizer_cache
                else self.tokenizer_cache.evidence()
            ),
            "dit_graph": None if self.graphs is None else self.graphs.evidence(),
            "static_vlm": (
                None
                if self.static_vlm is None or not self.config.static_vlm
                else self.static_vlm.evidence()
            ),
            "scope": "Experimental serial inference; replay checks do not establish task quality",
        }

    def close(self):
        # Reject closing inside a live root before ExitStack starts restoring any
        # controller. Its duplicate boundary callback is intentionally idempotent.
        with self._runtime_guard():
            try:
                if self.boundary is not None:
                    self.boundary.close()
            finally:
                if self.boundary is None or self.boundary.closed:
                    try:
                        self.runtime_stack.close()
                    finally:
                        try:
                            self.stack.close()
                        finally:
                            self._active = False

    def __exit__(self, *exception):
        self.close()
