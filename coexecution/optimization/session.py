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
        self.operator_profile = None
        self.boundary = None
        self.preparation = None
        self.tokenizer_cache = None
        self.image_preparation = None
        self.guard_backend = None
        self.graphs = None
        self.action_graph = None
        self.static_vlm = None
        self.fusion = None
        self._active = False
        self._model = None
        self._runtime_refs = []
        self._clear_cache = lambda: None
        self._roots = []
        self._resets = []
        self._configuration_file_hashes = {}

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
        self._configuration_file_hashes = {
            key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for key in (
                "coverage",
                "tactics",
                "group_tactics",
                "shape_tactics",
                "operator_profile",
            )
            if (path := getattr(self.config, key)) is not None
        }
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
        if self.config.operator_profile is not None:
            from coexecution.optimization.operators import OperatorProfile

            data = self.backend["operator_profile"]
            self.operator_profile = OperatorProfile(
                data["architecture"],
                data["precision"],
                data["implementations"],
                data.get("parameters", {}),
            )
        profile = self.operator_profile
        implementations = {} if profile is None else dict(profile.implementations)
        pointwise = None

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

            fusion = NonGemmAdapters(
                self.policy,
                "native" if self.config.fusion_mode == "all" else "triton",
                implementations=implementations,
            )
            self.fusion = fusion
            self.stack.callback(fusion.close)
            fusion.set(self.config.fusion_mode)
            if self.config.fusion_mode == "all":
                pointwise = PointwiseFusion(self.policy)
                self.stack.callback(pointwise.close)
                pointwise.set("dit")
        if self.config.precision in ("fp8", "fp4"):
            from coexecution.optimization.groot.floating import FloatingProjections
            from coexecution.optimization.installation import selected_fp4_swap_ab

            floating = FloatingProjections(
                model,
                executed,
                expanded=expanded,
                precision=self.config.precision,
                fast_fp8=self.config.fp8_fast_quant,
                fp4_swap_ab=selected_fp4_swap_ab(self.config, profile),
                shared_inputs=self.config.fp_shared_inputs,
                grouped=self.config.fp_grouped,
                swiglu=self.config.fp_swiglu,
                smoothquant=self.config.fp_smoothquant,
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
            from coexecution.optimization.installation import (
                configure_integer_controller,
                fuse_integer_modulation,
                install_integer_adapters,
            )

            controller = TransformerINT(
                model,
                executed,
                native=implementations.get("activation_pack", "native") == "native",
                expanded=expanded,
                condition_group=self.config.group_conditioning,
            )
            self.stack.callback(controller.close)
            configure_integer_controller(controller, profile)
            extra = AdditionalConnections(self.policy, controller, categories)
            self.stack.callback(extra.close)
            apply_tactics(
                controller, extra, self.config, recorded, backend_evidence=self.backend
            )
            controller.set(self.config.precision)
            extra.set(
                self.config.precision,
                expanded=expanded,
                fuse=fuse_integer_modulation(self.config, profile),
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
            self.coverage["operator_adapters"] = install_integer_adapters(
                self, controller, pointwise, profile
            )
            clear_cache = controller.clear_projection_cache
        # The larger graph owns the four DiT calls; nested capture is unsupported.
        if self.config.dit_graph and not self.config.action_graph:
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
        self.guard_backend = None
        if self.config.compiled_host_guards:
            from coexecution.optimization.execution.compiled_guards import (
                load_compiled_guards,
            )

            self.guard_backend = load_compiled_guards()
        if self.config.image_array_preparation:
            from coexecution.optimization.execution.images import ImageArrayPreparation

            if self.image_preparation is None:
                self.image_preparation = ImageArrayPreparation(self.policy)
            self.runtime_stack.callback(self.image_preparation.close)
            self.image_preparation.__enter__()
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
                or self.static_vlm.guard_backend is not self.guard_backend
            ):
                self.static_vlm = PolicyStaticS2(
                    self.policy,
                    cuda_graph=self.config.vlm_graph,
                    vision_rope=(
                        self.fusion.vision
                        if self.fusion is not None
                        and self.config.fusion_mode
                        in ("vision_rope", "rope", "rope_rms", "all")
                        and self.fusion.implementations.get("vision_rope")
                        != "reference"
                        else None
                    ),
                    guard_backend=self.guard_backend,
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
        # Freeze S1 execution only after its submodule serial wrappers exist.
        # The action scope has its own submission lock; the child boundaries
        # continue clearing projection groups at each original model call.
        if self.config.action_graph:
            from coexecution.optimization.groot.action import ActionGraph

            if (
                self.action_graph is None
                or self.action_graph.guard_backend is not self.guard_backend
                or self.action_graph.kv_reuse != self.config.action_kv_reuse
            ):
                self.action_graph = ActionGraph(
                    self.policy,
                    guard_backend=self.guard_backend,
                    kv_reuse=self.config.action_kv_reuse,
                )
            self.runtime_stack.callback(self.action_graph.close)
            self.action_graph.__enter__()

    @contextmanager
    def _runtime_guard(self):
        with ExitStack() as guards:
            for adapter in (
                self.image_preparation,
                self.preparation,
                self.tokenizer_cache,
                self.static_vlm,
                self.action_graph,
            ):
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
        image_array_preparation=None,
        compiled_host_guards=None,
        action_kv_reuse=None,
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
            ("image_array_preparation", image_array_preparation),
            ("compiled_host_guards", compiled_host_guards),
            ("action_kv_reuse", action_kv_reuse),
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
                    ("image_array_preparation", image_array_preparation),
                    ("compiled_host_guards", compiled_host_guards),
                    ("action_kv_reuse", action_kv_reuse),
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
        for key in (
            "coverage",
            "tactics",
            "group_tactics",
            "shape_tactics",
            "operator_profile",
            "fp_smoothquant",
        ):
            path = config[key]
            actual = (
                None
                if path is None
                else hashlib.sha256(Path(path).read_bytes()).hexdigest()
            )
            if (
                key in self._configuration_file_hashes
                and actual != self._configuration_file_hashes[key]
            ):
                raise RuntimeError(
                    f"Installed optimization configuration file changed: {key}"
                )
            config[key] = None if path is None else {"sha256": actual}
        return {
            "config": config,
            "coverage": self.coverage,
            "backend": self.backend,
            "image_array_preparation": (
                None
                if self.image_preparation is None
                or not self.config.image_array_preparation
                else self.image_preparation.evidence()
            ),
            "compiled_host_guards": (
                None
                if self.guard_backend is None or not self.config.compiled_host_guards
                else self.guard_backend.evidence()
            ),
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
            "action_graph": None
            if self.action_graph is None
            else self.action_graph.evidence(),
            "dit_graph_superseded": self.config.action_graph and self.config.dit_graph,
            "static_vlm": (
                None
                if self.static_vlm is None or not self.config.static_vlm
                else self.static_vlm.evidence()
            ),
            "operator_selection": (
                None
                if self.operator_profile is None
                else self.operator_profile.to_dict()
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
