"""Connect explicit operator profiles to reversible model adapters.

Configuration validation is GPU-free. Numerical kernels and candidate adapters
are imported only when installing an explicitly selected implementation.
"""

FUSION_COMPONENTS = {
    "vision_rope": {"vision_rope"},
    "text_rope": {"text_rope"},
    "rope": {"vision_rope", "text_rope"},
    "rmsnorm": {"rmsnorm"},
    "rope_rms": {"vision_rope", "text_rope", "rmsnorm"},
    "all": {"vision_rope", "text_rope", "rmsnorm", "adaln"},
}


def validate_selection(config, profile):
    enabled = FUSION_COMPONENTS[config.fusion_mode] if config.fusion else set()
    for family in ("vision_rope", "text_rope", "rmsnorm", "adaln"):
        implementation = profile.get(family, "reference")
        if implementation != "reference" and family not in enabled:
            raise ValueError(f"{family} is outside the selected fusion components")
    dit = (
        "layernorm_pack",
        "integer_residual",
        "context_pack",
        "final_modulation_pack",
    )
    if any(profile.get(family, "reference") != "reference" for family in dit):
        if not config.fusion or config.fusion_mode != "all":
            raise ValueError("Selected producers require full DiT fusion")
        if config.component == "vlm":
            raise ValueError("Selected producers require the DiT component")
    if profile.get("rmsnorm_pack", "reference") != "reference":
        if "rmsnorm" not in enabled or config.component == "dit":
            raise ValueError("RMS packing requires selected text RMSNorm fusion")
    if "layernorm_pack" in profile.implementations and config.norm_modulation_quant:
        raise ValueError("Profile LayerNorm and norm_modulation_quant overlap")
    if "adaln" in profile.implementations and config.norm_modulation_quant:
        raise ValueError("Explicit AdaLN selection and norm_modulation_quant overlap")
    if (
        "adaln" in profile.implementations
        and profile.get("layernorm_pack", "reference") != "reference"
    ):
        raise ValueError("Explicit AdaLN selection and LayerNorm packing overlap")
    if (
        profile.get("rmsnorm_pack", "reference") == "native"
        and "rmsnorm" in profile.implementations
        and profile.get("rmsnorm") != "native"
    ):
        raise ValueError("Native RMS packing and selected RMS implementation overlap")
    if (
        profile.get("integer_residual", "reference") != "reference"
        and config.residual_norm_quant
    ):
        raise ValueError("Profile residual writeback and residual_norm_quant overlap")
    if (
        profile.get("layernorm_pack", "reference") != "reference"
        and profile.get("final_modulation_pack", "reference") != "reference"
    ):
        raise ValueError("Final LayerNorm packing and modulation packing overlap")


def fuse_integer_modulation(config, profile):
    """Keep legacy fused packing unless an explicit AdaLN choice owns the norm."""
    return (
        config.fusion
        and config.fusion_mode == "all"
        and (profile is None or "adaln" not in profile.implementations)
    )


def configure_integer_controller(controller, profile):
    if profile is None:
        return
    from coexecution.optimization.groot.integer_candidates import (
        IntegerCandidateDispatch,
    )

    parameters = profile.options("integer_gemm")
    controller.install_candidates(
        IntegerCandidateDispatch(
            controller,
            implementation=profile.get("integer_gemm", "reference"),
            small_m_tactic=parameters.get("tactic", 0),
            split_k_slices=parameters.get("slices", 2),
            split_k_tile=parameters.get("tile", 0),
        )
    )


def install_integer_adapters(scope, controller, pointwise, profile):
    if profile is None:
        return {}
    bits = {"w8a8": 8, "w4a4": 4}[scope.config.precision]
    evidence = {}
    if profile.get("group_writeback", "reference") == "native":
        from coexecution.optimization.groot.group_outputs import GroupOutputAdapters

        groups = GroupOutputAdapters(controller)
        scope.stack.callback(groups.close)
        groups.set(bits)
        if not groups.saved:
            raise ValueError("Group writeback selected no contiguous projection groups")
        evidence["group_writeback"] = {
            "requested": "native",
            "installed_groups": len(groups.saved),
            "runtime": "Shape/tactic guards may retain separate writes",
        }
    rms = profile.get("rmsnorm_pack", "reference") == "native"
    final = profile.get("final_modulation_pack", "reference") == "triton"
    if rms or final:
        from coexecution.optimization.groot.producers import QuantizationProducers

        producers = QuantizationProducers(scope.policy, controller, pointwise)
        scope.stack.callback(producers.close)
        producers.set(bits, rmsnorm=rms, final_modulation=final)
        evidence["rms_and_final_producers"] = producers.evidence()
    if profile.get("context_pack", "reference") == "shared":
        from coexecution.optimization.groot.producers import SharedContextQuantization

        context = SharedContextQuantization(pointwise, controller)
        scope.stack.callback(context.close)
        context.set(bits)
        evidence["context_pack"] = context.evidence()
    if profile.get("layernorm_pack", "reference") == "native":
        from coexecution.optimization.groot.producers import LayerNormQuantization

        norms = LayerNormQuantization(controller, pointwise)
        scope.stack.callback(norms.close)
        norms.set(bits, "all")
        evidence["layernorm_pack"] = norms.evidence()
    if profile.get("integer_residual", "reference") == "native":
        from coexecution.optimization.groot.residual import DitResidualAdapters

        residuals = DitResidualAdapters(controller, pointwise)
        scope.stack.callback(residuals.close)
        residuals.set(bits)
        evidence["integer_residual"] = residuals.evidence()
    return {
        "scope": "Installed adapters; not proof of per-call kernel execution or speed",
        "adapters": evidence,
    }
