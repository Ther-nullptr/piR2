"""Transactional installation, evidence and restoration of GR00T optimizations.

GPU dependencies are loaded only when an enabled configuration enters its scope.
"""

import hashlib
from contextlib import ExitStack
from dataclasses import asdict
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
        self.coverage = {}
        self.backend = {}
        self.operator_profile = None
        self.boundary = None

    def __enter__(self):
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
        clear_cache = lambda: None
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
            self.stack.callback(fusion.close)
            fusion.set(self.config.fusion_mode)
            if self.config.fusion_mode == "all":
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
            clear_cache = lambda: clear_floating_projection_cache(floating.groups)
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
        if self.config.dit_graph:
            from coexecution.optimization.execution.graph import DitGraphs

            graphs = DitGraphs(model.action_head.model)
            self.stack.callback(graphs.close)
            graphs.set(self.config.precision)
        head = getattr(model, "action_head", None)
        roots = [
            root
            for root in (
                getattr(model, "backbone", None),
                getattr(head, "vl_self_attention", None),
                getattr(head, "model", None),
            )
            if root is not None
        ]
        resets = (
            [head] if callable(getattr(head, "reset_streaming_buffer", None)) else []
        )
        self.boundary = SerialForwardBoundary(roots, clear_cache, reset_modules=resets)
        self.stack.callback(self.boundary.close)

    def evidence(self):
        config = asdict(self.config)
        for key in (
            "coverage",
            "tactics",
            "group_tactics",
            "shape_tactics",
            "operator_profile",
        ):
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
        if self.boundary is None:
            self.stack.close()
            return
        try:
            self.boundary.close()
        finally:
            if self.boundary.closed:
                self.stack.close()

    def __exit__(self, *exception):
        self.close()
