"""Diagnostic component-owned submission boundaries for the piR2 runtime.

This module imports no tensor framework: its ownership rules can be tested without
running a model. Only the explicit worker installs it on a CUDA-loaded policy.
"""

from collections import Counter
from contextlib import contextmanager


def owner_of(name):
    if name.startswith("backbone."):
        return "vlm"
    if name.startswith("action_head."):
        return "dit"
    raise ValueError(f"Projection has no declared component owner: {name}")


def partition_groups(controllers):
    groups = {"vlm": [], "dit": []}
    ownership = []
    seen = set()
    for controller in controllers:
        for bits, records in controller.grouped.items():
            for names, group in records:
                owners = {owner_of(name) for name in names}
                if len(owners) != 1:
                    raise ValueError(f"Cross-component projection group: {names}")
                owner = owners.pop()
                if id(group) in seen:
                    raise ValueError("A mutable projection group is shared")
                seen.add(id(group))
                groups[owner].append(group)
                ownership.append({"owner": owner, "bits": bits, "names": list(names)})
    return groups, ownership


def clear_groups(groups):
    for group in groups:
        group._input = group._output = None
        group._remaining.clear()


class ComponentBoundaries:
    """Split root guards and mutable group caches while keeping both guards live."""

    def __init__(self, optimized, boundary_class):
        original = optimized.boundary
        if original is None:
            raise ValueError("The original guarded optimization session is required")
        if optimized.config.precision not in ("bf16", "w8a8", "w4a4"):
            raise ValueError(
                "Only BF16 and the audited integer controller are supported"
            )
        if (
            any(
                getattr(optimized.config, option, False)
                for option in (
                    "dit_graph",
                    "vlm_graph",
                    "norm_modulation_quant",
                    "residual_norm_quant",
                    "group_conditioning",
                    "operator_profile",
                )
            )
            or getattr(optimized.config, "category_id", None) is not None
        ):
            raise ValueError(
                "This diagnostic requires the ordinary transformer controller"
            )
        self.controllers = []
        if optimized.config.precision != "bf16":
            controller = getattr(original.clear, "__self__", None)
            if controller is None or not hasattr(controller, "grouped"):
                raise ValueError("Unknown original integer cache callback")
            self.controllers.append(controller)
        self.groups, ownership = partition_groups(self.controllers)
        model = optimized.policy.model
        head = model.action_head
        vlm_parameters = {id(value) for value in model.backbone.parameters()}
        dit_parameters = {id(value) for value in head.parameters()}
        if vlm_parameters & dit_parameters:
            raise ValueError("VLM and DiT unexpectedly share parameter objects")
        self.guards = []
        original.close()
        try:
            self.guards.append(
                boundary_class(
                    [model.backbone], lambda: clear_groups(self.groups["vlm"])
                )
            )
            self.guards.append(
                boundary_class(
                    [head.vl_self_attention, head.model],
                    lambda: clear_groups(self.groups["dit"]),
                    reset_modules=[head],
                )
            )
        except BaseException:
            self.close()
            raise
        self.evidence = {
            "adapter": "ComponentBoundaries-v1",
            "source_serial_guard_replaced": True,
            "guards_retained": {
                "vlm": ["backbone"],
                "dit": [
                    "action_head.vl_self_attention",
                    "action_head.model",
                    "action_head.reset_streaming_buffer",
                ],
            },
            "cross_component_parameter_aliases": 0,
            "projection_group_ownership": ownership,
            "mutable_projection_caches": "Each group has exactly one component owner; only its component clears it before/after forward.",
            "streaming_buffer_owner": "DiT thread only; restored by main thread only after both completion events finish.",
            "feature_cache_owner": "Two versioned GPU slots, completion-gated publish, reader lease held until DiT completion.",
            "rng_owner": "Only DiT consumes CUDA RNG; main restores the common state while both workers are idle. VLM RNG non-consumption is checked.",
            "kernel_workspace": "Native integer dispatch allocates its outputs and uses getCurrentCUDAStream; Triton output allocations are call-local. Shared LUT/packed weights are read-only.",
            "lazy_launcher_state": "All fixture shapes warmed on each execution stream before measurement; known shared launcher key sets must remain unchanged.",
            "scope": "Diagnostic-only adapter; the general model service still requires serial submissions.",
        }

    def assert_clear(self):
        for groups in self.groups.values():
            for group in groups:
                if (
                    group._input is not None
                    or group._output is not None
                    or group._remaining
                ):
                    raise RuntimeError("Projection cache survived its root boundary")

    def close(self):
        for guard in reversed(self.guards):
            guard.close()
        self.guards.clear()


@contextmanager
def executed_projections(controllers):
    """Count actual dispatch during validation/warmup; absent from timed region."""
    counters = Counter()
    saved = []
    for controller in controllers:
        original = controller.dispatch
        names_by_object = {}
        for bits, linears in controller.quant.items():
            for name, linear in linears.items():
                names_by_object[id(linear)] = (bits, [name])
            for names, group in controller.grouped[bits]:
                names_by_object[id(group.linear)] = (bits, names)

        def dispatch(
            q, x, lut=None, up=None, original=original, names_by_object=names_by_object
        ):
            bits, names = names_by_object[id(q)]
            for name in names:
                counters[(name, bits)] += 1
            return original(q, x, lut, up)

        saved.append((controller, original))
        controller.dispatch = dispatch
    try:
        yield counters
    finally:
        for controller, original in saved:
            controller.dispatch = original
