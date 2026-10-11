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
    read_shape_tactics,
)


def test_default_does_not_import_gpu_modules_or_touch_policy():
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    config = optimization_config(parser.parse_args([]))
    before = set(sys.modules)
    with GrootOptimizations(object(), config) as scope:
        assert not scope.config.enabled
        assert not config.vision_channels_last
        assert scope.evidence()["coverage"] == {}
    assert set(sys.modules) == before


@pytest.mark.parametrize("precision", ["w8a8", "w4a4"])
def test_norm_residual_fusion_flags_require_integer_fusion(precision):
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    config = optimization_config(
        parser.parse_args(
            [
                "--inference-precision",
                precision,
                "--operator-fusion",
                "--norm-modulation-quant",
                "--residual-norm-quant",
            ]
        )
    )
    assert config.norm_modulation_quant and config.residual_norm_quant
    config.validate_variant("pir2")
    assert not OptimizationConfig().norm_modulation_quant
    assert not OptimizationConfig().residual_norm_quant
    for options in ({}, {"fusion": True}, {"precision": precision}):
        with pytest.raises(ValueError, match="fusion and integer"):
            OptimizationConfig(norm_modulation_quant=True, **options)
    with pytest.raises(ValueError, match="requires norm_modulation_quant"):
        OptimizationConfig(precision=precision, fusion=True, residual_norm_quant=True)


def test_flags_are_independent_and_streaming_allows_eager_integer_inference():
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
    config.validate_variant("pir2")
    OptimizationConfig().validate_variant("pir2")


def test_vision_layout_is_independent_and_opt_in():
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    config = optimization_config(parser.parse_args(["--vision-channels-last"]))
    assert config.vision_channels_last and config.enabled
    assert config.precision == "bf16" and not config.fusion and not config.dit_graph
    config.validate_variant("pir2")
    config.validate_variant("flow")


@pytest.mark.parametrize(
    "training,dtype,device,message",
    [
        (True, "bf16", "cuda", "eval-mode BF16"),
        (False, "fp32", "cuda", "eval-mode BF16"),
        (False, "bf16", "cpu", "require CUDA"),
    ],
)
def test_vision_layout_preserves_model_guards(
    monkeypatch, training, dtype, device, message
):
    model = SimpleNamespace(
        training=training,
        parameters=lambda: iter(
            [SimpleNamespace(dtype=dtype, device=SimpleNamespace(type=device))]
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    with (
        pytest.raises(ValueError, match=message),
        GrootOptimizations(
            SimpleNamespace(model=model), OptimizationConfig(vision_channels_last=True)
        ),
    ):
        pytest.fail("Layout conversion cannot bypass the model guards")


def test_streaming_allows_serial_dit_graph_and_rejects_condition_grouping():
    OptimizationConfig(dit_graph=True).validate_variant("pir2")
    config = OptimizationConfig(
        precision="w8a8", fusion=True, scope="all", group_conditioning=True
    )
    config.validate_variant("flow")
    with pytest.raises(ValueError, match="Streaming piR2"):
        config.validate_variant("pir2")


def test_streaming_condition_guard_also_applies_to_direct_scope(monkeypatch):
    model = SimpleNamespace(
        training=False,
        config=SimpleNamespace(streaming=True),
        parameters=lambda: iter(
            [SimpleNamespace(dtype="bf16", device=SimpleNamespace(type="cuda"))]
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    with (
        pytest.raises(ValueError, match="Streaming piR2"),
        GrootOptimizations(
            SimpleNamespace(model=model),
            OptimizationConfig(
                precision="w8a8",
                fusion=True,
                scope="all",
                group_conditioning=True,
                vision_channels_last=True,
            ),
        ),
    ):
        pytest.fail("A direct scope cannot bypass the streaming condition restriction")


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


def test_shape_tactics_reach_singles_categories_and_groups(tmp_path, monkeypatch):
    path = tmp_path / "shapes.json"
    path.write_text(
        json.dumps([{"bits": 8, "shape": [41, 130, 64, True], "tactic": 2}])
    )
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    config = optimization_config(
        parser.parse_args(["--quantization-shape-tactics", str(path)])
    )
    installed = []
    monkeypatch.setitem(
        sys.modules,
        "coexecution.quantization",
        SimpleNamespace(
            install_shape_tactics=lambda q, shapes: installed.append((q, shapes))
        ),
    )
    single, category, group = [SimpleNamespace(tactic=0) for _ in range(3)]
    controller = SimpleNamespace(
        quant={8: {"q": single}, 4: {}},
        grouped={8: [(["k", "v"], SimpleNamespace(linear=group))], 4: []},
    )
    extra = SimpleNamespace(quant={8: {"category": category}, 4: {}})
    apply_tactics(controller, extra, config, {})
    assert [id(q) for q, _ in installed] == [id(single), id(category), id(group)]
    assert all(shapes == {(8, 41, 130, 64, True): 2} for _, shapes in installed)
    assert GrootOptimizations(object(), config).evidence()["config"]["shape_tactics"][
        "sha256"
    ]


@pytest.mark.parametrize(
    "row",
    [
        {"bits": True, "shape": [41, 130, 64, True], "tactic": 2},
        {"bits": 8, "shape": [True, 130, 64, True], "tactic": 2},
        {"bits": 8, "shape": [41, 0, 64, True], "tactic": 2},
        {"bits": 8, "shape": [41, 130, 64, 1], "tactic": 2},
        {"bits": 8, "shape": [41, 130, 64, True], "tactic": 8},
    ],
)
def test_shape_tactics_reject_invalid_signatures(tmp_path, row):
    path = tmp_path / "shapes.json"
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError):
        read_shape_tactics(path)


def test_shape_tactics_reject_duplicate_signatures(tmp_path):
    path = tmp_path / "shapes.json"
    row = {"bits": 8, "shape": [41, 130, 64, True], "tactic": 2}
    path.write_text(json.dumps([row, row]))
    with pytest.raises(ValueError, match="Duplicate"):
        read_shape_tactics(path)


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


@pytest.mark.parametrize("named_tactics", [False, True])
def test_name_only_coverage_does_not_require_legacy_shapes(tmp_path, named_tactics):
    groups = tmp_path / "groups.json"
    groups.write_text(
        json.dumps([{"bits": 8, "members": ["q", "k", "v"], "tactic": 2}])
    )
    grouped = {
        bits: [(["q", "k", "v"], SimpleNamespace(linear=SimpleNamespace(tactic=7)))]
        for bits in (8, 4)
    }
    controller = SimpleNamespace(quant={8: {}, 4: {}}, grouped=grouped)
    extra = SimpleNamespace(quant={8: {}, 4: {}})
    config = OptimizationConfig(group_tactics=groups if named_tactics else None)
    apply_tactics(controller, extra, config, {name: {} for name in ("q", "k", "v")})
    assert grouped[8][0][1].linear.tactic == (2 if named_tactics else 0)
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
