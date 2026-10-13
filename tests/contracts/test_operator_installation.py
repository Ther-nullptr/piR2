"""Profiles must not silently select code excluded by the model configuration."""

import pytest

from coexecution.optimization.config import OptimizationConfig
from coexecution.optimization.operators import OperatorProfile


def validate(config, implementation):
    from coexecution.optimization.installation import validate_selection

    profile = OperatorProfile("sm_89", config.precision, implementation)
    validate_selection(config, profile)


def test_profile_does_not_enable_unrequested_bf16_components():
    config = OptimizationConfig(fusion=True, fusion_mode="rope")
    with pytest.raises(ValueError, match="rmsnorm.*selected fusion"):
        validate(config, {"rmsnorm": "native"})
    validate(config, {"vision_rope": "native"})


def test_dit_candidate_requires_dit_component_and_full_fusion():
    with pytest.raises(ValueError, match="full DiT fusion"):
        validate(OptimizationConfig(precision="w8a8"), {"layernorm_pack": "native"})
    with pytest.raises(ValueError, match="DiT component"):
        validate(
            OptimizationConfig(precision="w8a8", fusion=True, component="vlm"),
            {"integer_residual": "native"},
        )
    validate(
        OptimizationConfig(precision="w8a8", fusion=True),
        {"layernorm_pack": "native"},
    )


def test_two_norm_replacements_cannot_overlap():
    with pytest.raises(ValueError, match="overlap"):
        validate(
            OptimizationConfig(
                precision="w8a8", fusion=True, norm_modulation_quant=True
            ),
            {"layernorm_pack": "native"},
        )


def test_reference_profiles_do_not_require_extra_model_changes():
    validate(
        OptimizationConfig(precision="w8a8"),
        {"integer_gemm": "reference", "group_writeback": "reference"},
    )
