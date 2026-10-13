"""Operator choices must survive the existing CLI without enabling defaults."""

import argparse
import hashlib

import pytest

from coexecution.optimization.config import (
    OptimizationConfig,
    add_optimization_arguments,
    optimization_config,
)
from coexecution.optimization.session import GrootOptimizations


def parse(*arguments):
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    return optimization_config(parser.parse_args(arguments))


def test_selective_fusion_mode_reaches_configuration():
    config = parse("--operator-fusion", "--operator-fusion-mode", "rope")
    assert config.fusion and config.fusion_mode == "rope"
    assert config.precision == "bf16" and not config.dit_graph


def test_legacy_fusion_and_disabled_defaults_remain_distinct():
    disabled = parse()
    assert not disabled.enabled
    assert disabled.fusion_mode == "all" and disabled.operator_profile is None
    assert parse("--operator-fusion").fusion_mode == "all"


def test_selective_mode_cannot_be_silently_ignored():
    with pytest.raises(ValueError, match="requires.*fusion"):
        OptimizationConfig(fusion_mode="rope")
    with pytest.raises(ValueError, match="fusion mode"):
        OptimizationConfig(fusion=True, fusion_mode="unknown")


@pytest.mark.parametrize(
    "options",
    [
        {"norm_modulation_quant": True},
        {"group_conditioning": True, "scope": "all"},
    ],
)
def test_dit_producers_require_full_fusion_selection(options):
    with pytest.raises(ValueError, match="all.*fusion|fusion.*all"):
        OptimizationConfig(precision="w8a8", fusion=True, fusion_mode="rope", **options)


def test_profile_requires_an_enabled_optimization(tmp_path):
    with pytest.raises(ValueError, match="profile requires"):
        OptimizationConfig(operator_profile=tmp_path / "profile.json")


def test_profile_identity_is_content_based(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text('{"operator": "example"}\n')
    config = parse("--inference-precision", "w8a8", "--operator-profile", str(profile))
    assert config.operator_profile == profile
    evidence = GrootOptimizations(object(), config).evidence()
    assert evidence["config"]["operator_profile"] == {
        "sha256": hashlib.sha256(profile.read_bytes()).hexdigest()
    }
