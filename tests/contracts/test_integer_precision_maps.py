"""Integer protocol/configuration checks; no tensor or model execution."""

import argparse
import json
from itertools import pairwise

import pytest

from coexecution.optimization.config import (
    OptimizationConfig,
    add_optimization_arguments,
    optimization_config,
)
from coexecution.optimization.precision import (
    checked_precision_map,
    progressive_maps,
    read_precision_map,
)


def test_quantization_options_are_explicit_and_keep_uniform_defaults(tmp_path):
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    default = optimization_config(parser.parse_args([]))
    assert default.precision_map is None and default.int4_calibration is None
    config = optimization_config(
        parser.parse_args(
            [
                "--inference-precision",
                "w4a4",
                "--quantization-precision-map",
                str(tmp_path / "map.json"),
                "--int4-calibration",
                str(tmp_path / "maxima.pt"),
                "--int4-prepare",
                "reference",
            ]
        )
    )
    assert config.int4_balance_alpha == 0.5 and config.int4_prepare == "reference"
    assert config.precision == "w4a4"


@pytest.mark.parametrize("value", [True, -1, 1.1, float("nan"), float("inf"), "0.5"])
def test_invalid_balance_alpha_rejected(value):
    with pytest.raises(ValueError, match="alpha"):
        OptimizationConfig(precision="w4a4", int4_balance_alpha=value)


@pytest.mark.parametrize("precision", ["bf16", "fp8", "fp4"])
def test_integer_maps_and_calibration_reject_other_precisions(precision, tmp_path):
    with pytest.raises(ValueError, match="integer precision"):
        OptimizationConfig(precision=precision, precision_map=tmp_path / "map.json")
    with pytest.raises(ValueError, match="integer precision"):
        OptimizationConfig(precision=precision, int4_calibration=tmp_path / "maxima.pt")


@pytest.mark.parametrize(
    "mapping", [{"q": 4}, {"q": 4, "k": 8, "v": 4}, {"q": True, "k": 4, "v": 4}]
)
def test_partial_or_non_atomic_maps_rejected(mapping):
    with pytest.raises(ValueError):
        checked_precision_map(mapping, ["q", "k", "v"], [["q", "k", "v"]])


def test_serialized_map_has_exact_schema_and_candidates(tmp_path):
    p = tmp_path / "map.json"
    p.write_text(json.dumps({"schema_version": 1, "precision_map": {"a": 4, "b": 8}}))
    assert read_precision_map(p, ["a", "b"]) == {"a": 4, "b": 8}
    p.write_text(
        json.dumps({"schema_version": True, "precision_map": {"a": 4, "b": 8}})
    )
    with pytest.raises(ValueError, match="schema"):
        read_precision_map(p, ["a", "b"])


def test_fixed_cumulative_order_and_atomic_group_boundaries():
    names = [
        "backbone.model.model.visual.blocks.10.proj",
        "backbone.model.model.visual.blocks.2.proj",
        "backbone.model.model.language_model.layers.0.q_proj",
        "backbone.model.model.language_model.layers.0.k_proj",
        "backbone.model.model.language_model.layers.0.v_proj",
        "action_head.vl_self_attention.0.to_q",
        "action_head.model.transformer_blocks.0.to_q",
        "action_head.extra.proj",
        "action_head.category",
    ]
    groups = [names[2:5]]
    tiers = progressive_maps(names, groups, [names[-1]], divisions=9)
    assert len(tiers) == 10
    assert set(tiers[0]["precision_map"].values()) == {8}
    assert set(tiers[-1]["precision_map"].values()) == {4}
    assert tiers[1]["precision_map"][names[1]] == 4
    assert tiers[1]["precision_map"][names[0]] == 8
    assert tiers[3]["w4_count"] == 5
    for before, after in pairwise(tiers):
        assert {n for n, b in before["precision_map"].items() if b == 4} <= {
            n for n, b in after["precision_map"].items() if b == 4
        }
        assert after["w4_count"] + after["w8_count"] == len(names)
        assert len({after["precision_map"][n] for n in groups[0]}) == 1
