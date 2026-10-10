"""Configuration and transactional lifecycle contracts; no model/GPU imports."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from coexecution.groot_optimization import (
    GrootOptimizations,
    OptimizationConfig,
    add_optimization_arguments,
    apply_tactics,
    optimization_config,
)


def test_default_does_not_import_gpu_modules_or_touch_policy():
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    config = optimization_config(parser.parse_args([]))
    before = set(sys.modules)
    with GrootOptimizations(object(), config) as scope:
        assert not scope.config.enabled
        assert scope.evidence()["coverage"] == {}
    assert set(sys.modules) == before


def test_flags_are_independent_and_streaming_is_explicitly_unsupported():
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    config = optimization_config(
        parser.parse_args(
            ["--inference-precision", "w4a4", "--quantization-scope", "all"]
        )
    )
    assert config.precision == "w4a4"
    assert not config.fusion and not config.dit_graph
    config.validate_variant("flow")
    with pytest.raises(ValueError, match="streaming"):
        config.validate_variant("pir2")
    OptimizationConfig().validate_variant("pir2")


def test_category_requires_expanded_scope():
    with pytest.raises(ValueError, match="scope=all"):
        OptimizationConfig(category_id=2)
    with pytest.raises(ValueError, match="nonnegative"):
        OptimizationConfig(scope="all", category_id=-1)


def test_condition_grouping_requires_shared_condition_path():
    for options in ({}, {"precision": "w8a8"}, {"precision": "w8a8", "fusion": True}):
        with pytest.raises(ValueError, match="Condition grouping"):
            OptimizationConfig(group_conditioning=True, **options)
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    config = optimization_config(
        parser.parse_args(
            [
                "--inference-precision",
                "w4a4",
                "--operator-fusion",
                "--quantization-scope",
                "all",
                "--group-conditioning",
            ]
        )
    )
    assert config.group_conditioning


def test_empty_integer_coverage_is_rejected_before_gpu_install(tmp_path, monkeypatch):
    coverage = tmp_path / "coverage.json"
    coverage.write_text('{"linears": {"old.name": {}}}')
    model = SimpleNamespace(
        training=False,
        config=SimpleNamespace(streaming=False),
        parameters=lambda: iter(
            [SimpleNamespace(dtype="bf16", device=SimpleNamespace(type="cuda"))]
        ),
        named_modules=list,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(
        sys.modules,
        "coexecution.quantization",
        SimpleNamespace(
            inventory=lambda *a, **k: ({"new.name": object()}, [], [], []),
            TransformerINT=lambda *a, **k: pytest.fail("No sites to install"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "coexecution.quantization_connections",
        SimpleNamespace(AdditionalConnections=None),
    )
    with (
        pytest.raises(ValueError, match="no matching projections"),
        GrootOptimizations(
            SimpleNamespace(model=model),
            OptimizationConfig(precision="w8a8", coverage=coverage),
        ),
    ):
        pytest.fail("An empty selection cannot enter integer inference")


def test_failed_install_and_body_restore_in_reverse_order():
    calls = []

    class Partial(GrootOptimizations):
        def _install(self):
            self.stack.callback(calls.append, "fusion")
            self.stack.callback(calls.append, "quantization")
            raise RuntimeError("failed installation")

    with (
        pytest.raises(RuntimeError, match="installation"),
        Partial(object(), OptimizationConfig(fusion=True)),
    ):
        pytest.fail("A failed installation cannot enter the body")
    assert calls == ["quantization", "fusion"]

    class Installed(GrootOptimizations):
        def _install(self):
            self.stack.callback(calls.append, "quantization")
            self.stack.callback(calls.append, "graph")

    with (
        pytest.raises(RuntimeError, match="body"),
        Installed(object(), OptimizationConfig(fusion=True)),
    ):
        raise RuntimeError("body failure")
    assert calls[-2:] == ["graph", "quantization"]


def test_explicit_tactics_and_group_defaults(tmp_path):
    linears = tmp_path / "tactics.json"
    groups = tmp_path / "groups.json"
    linears.write_text(json.dumps({"8": {"q": 3}, "4": {"q": 6}}))
    groups.write_text(
        json.dumps([{"bits": 8, "members": ["q", "k", "v"], "tactic": 2}])
    )
    quant = {bits: {"q": SimpleNamespace(tactic=0)} for bits in (8, 4)}
    grouped = {
        bits: [
            (
                ["q", "k", "v"],
                SimpleNamespace(linear=SimpleNamespace(tactic=7)),
            )
        ]
        for bits in (8, 4)
    }
    controller = SimpleNamespace(quant=quant, grouped=grouped)
    extra = SimpleNamespace(quant={8: {}, 4: {}})
    config = OptimizationConfig(tactics=linears, group_tactics=groups)
    apply_tactics(controller, extra, config, {})
    assert [quant[b]["q"].tactic for b in (8, 4)] == [3, 6]
    assert [grouped[b][0][1].linear.tactic for b in (8, 4)] == [2, 0]
    groups.write_text(
        json.dumps([{"bits": 8, "members": ["q", "k", "v"], "tactic": 8}])
    )
    with pytest.raises(ValueError, match="tactic"):
        apply_tactics(controller, extra, config, {})


def test_group_shape_tactics_require_explicit_coverage(tmp_path):
    groups = tmp_path / "groups.json"
    groups.write_text(
        json.dumps([{"bits": 4, "shape": [41, 1536, 4608, True], "tactic": 3}])
    )
    grouped = {
        bits: [
            (
                ["q", "k", "v"],
                SimpleNamespace(
                    linear=SimpleNamespace(
                        in_features=1536, out_features=4608, bias=object(), tactic=0
                    )
                ),
            )
        ]
        for bits in (8, 4)
    }
    controller = SimpleNamespace(quant={8: {}, 4: {}}, grouped=grouped)
    extra = SimpleNamespace(quant={8: {}, 4: {}})
    config = OptimizationConfig(group_tactics=groups)
    apply_tactics(controller, extra, config, {"q": {"shape": [1, 41, 1536]}})
    assert grouped[4][0][1].linear.tactic == 3
    apply_tactics(controller, extra, config, {})
    assert grouped[4][0][1].linear.tactic == 0


def test_ada_preset_loads_through_public_tactic_interface():
    preset = (
        Path(__file__).resolve().parents[1] / "configs/quantization/groot-libero10-ada"
    )
    coverage = json.loads((preset / "coverage.json").read_text())["linears"]
    singles = json.loads((preset / "tactics.json").read_text())
    rows = json.loads((preset / "group-tactics.json").read_text())
    assert set(singles["8"]) == set(singles["4"])
    assert set(coverage) <= set(singles["8"])
    assert all(set(row["members"]) <= set(coverage) for row in rows)
    controller = SimpleNamespace(
        quant={
            bits: {name: SimpleNamespace(tactic=-1) for name in coverage}
            for bits in (8, 4)
        },
        grouped={
            bits: [
                (row["members"], SimpleNamespace(linear=SimpleNamespace(tactic=-1)))
                for row in rows
                if row["bits"] == bits
            ]
            for bits in (8, 4)
        },
    )
    extra = SimpleNamespace(
        quant={
            bits: {
                name: SimpleNamespace(tactic=-1)
                for name in singles[str(bits)]
                if name not in coverage
            }
            for bits in (8, 4)
        }
    )
    apply_tactics(
        controller,
        extra,
        OptimizationConfig(
            tactics=preset / "tactics.json", group_tactics=preset / "group-tactics.json"
        ),
        coverage,
    )
    for bits in (8, 4):
        actual = {**controller.quant[bits], **extra.quant[bits]}
        assert {n: q.tactic for n, q in actual.items()} == singles[str(bits)]
        expected = {
            tuple(row["members"]): row["tactic"] for row in rows if row["bits"] == bits
        }
        assert {
            tuple(n): g.linear.tactic for n, g in controller.grouped[bits]
        } == expected
