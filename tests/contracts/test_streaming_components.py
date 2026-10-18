"""Component selection and serial cache ownership; no tensor/model execution."""

import argparse
import json
import sys
from types import SimpleNamespace

import pytest

from coexecution.optimization import session
from coexecution.optimization.config import (
    OptimizationConfig,
    add_optimization_arguments,
    optimization_config,
    selected_component,
)
from coexecution.optimization.execution.cache import (
    SerialForwardBoundary,
    projection_cache_clearers,
)


class Root:
    def forward(self, value=None):
        return value() if callable(value) else value

    def reset_streaming_buffer(self):
        self.reset = True


def test_component_cli_default_and_explicit_selection():
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    assert optimization_config(parser.parse_args([])).component == "all"
    assert not OptimizationConfig(component="vlm").enabled
    for component in ("all", "vlm", "dit"):
        config = optimization_config(
            parser.parse_args(["--quantization-component", component])
        )
        assert config.component == component
        # Preserve the remote experimental Graph and W4 interfaces.
        OptimizationConfig(
            component=component, precision="w4a4", dit_graph=True
        ).validate_variant("pir2")
    with pytest.raises(ValueError, match="component"):
        OptimizationConfig(component="vision")


def test_component_cache_clear_is_explicit_integer_only_configuration():
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    assert not optimization_config(parser.parse_args([])).component_cache_clear
    selected = optimization_config(
        parser.parse_args(
            [
                "--inference-precision",
                "w8a8",
                "--component-cache-clear",
            ]
        )
    )
    assert selected.component_cache_clear
    selected.validate_variant("flow")
    with pytest.raises(ValueError, match="W8A8"):
        OptimizationConfig(component_cache_clear=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_flow_session_component_clears_preserve_default_full_cleanup(enabled):
    from coexecution.optimization.execution.cache import ProjectionCacheClearer

    model, head = SimpleNamespace(backbone=Root()), Root()
    head.model, head.vl_self_attention = Root(), Root()
    model.action_head = head
    calls = []

    class Remaining(set):
        def __init__(self, key):
            self.key = key
            super().__init__()

        def clear(self):
            calls.append(self.key)
            super().clear()

    class Controller:
        def __init__(self):
            self.grouped = {
                bits: [
                    (
                        [prefix + ".projection"],
                        SimpleNamespace(
                            _input=None,
                            _output=None,
                            _remaining=Remaining((bits, name)),
                        ),
                    )
                    for name, prefix in (
                        ("vlm", "backbone"),
                        ("s1_feature", "action_head.vl_self_attention"),
                        ("dit", "action_head.model"),
                    )
                ]
                for bits in (8, 4)
            }

        def clear_projection_cache(self):
            ProjectionCacheClearer(
                group for records in self.grouped.values() for _, group in records
            )()

        def projection_cache_clearers(self, precision):
            return projection_cache_clearers(self.grouped, precision)

    controller = Controller()
    scope = session.GrootOptimizations(
        SimpleNamespace(model=model),
        OptimizationConfig(precision="w8a8", component_cache_clear=enabled),
    )
    scope._bind_runtime_model(model)
    scope._clear_cache = controller.clear_projection_cache
    scope._install_boundary()
    try:
        assert model.backbone.forward("frame") == "frame"
        if enabled:
            assert calls == [(8, "vlm")] * 2
            assert scope.evidence()["cache_clear_policy"]["groups"] == {
                "vlm": 1,
                "s1_feature": 1,
                "dit": 1,
            }
        else:
            assert len(calls) == 12
        calls.clear()
        head.reset_streaming_buffer()
        assert len(calls) == 12 and sum(bits == 4 for bits, _ in calls) == 6
        calls.clear()
    finally:
        scope.close()
    assert len(calls) == 6 and sum(bits == 4 for bits, _ in calls) == 3


@pytest.mark.parametrize(
    "name,component,expected",
    [
        ("backbone.visual.qkv", "vlm", True),
        ("backbone.text.q_proj", "dit", False),
        ("action_head.vl_self_attention.to_q", "dit", True),
        ("action_head.vl_self_attention.to_q", "vlm", False),
        ("action_head.model.ffn", "dit", True),
        ("other.backbone.qkv", "vlm", False),
        ("other.projection", "all", True),
    ],
)
def test_component_follows_the_executing_system(name, component, expected):
    assert selected_component(name, component) is expected


@pytest.mark.parametrize(
    "options",
    [
        {"category_id": 0, "scope": "all"},
        {"group_conditioning": True, "scope": "all", "fusion": True},
        {"norm_modulation_quant": True, "fusion": True},
    ],
)
def test_vlm_selection_rejects_dit_only_options(options):
    with pytest.raises(ValueError, match="DiT component"):
        OptimizationConfig(precision="w8a8", component="vlm", **options)


def test_boundary_rejects_cross_root_and_reset_overlap_and_restores():
    first, second, head = Root(), Root(), Root()
    clear_calls = []
    boundary = SerialForwardBoundary(
        [first, second, first],
        lambda: clear_calls.append("clear"),
        reset_modules=[head],
    )
    first_call = first.forward

    def nested():
        with pytest.raises(RuntimeError, match="serial submissions"):
            second.forward()
        with pytest.raises(RuntimeError, match="serial submissions"):
            head.reset_streaming_buffer()
        return "completed"

    assert first_call("value") == "value"
    assert len(clear_calls) == 2
    assert first_call(nested) == "completed"
    assert len(clear_calls) == 4
    head.reset_streaming_buffer()
    assert head.reset
    boundary.close()
    boundary.close()
    assert "forward" not in first.__dict__
    assert "forward" not in second.__dict__
    assert "reset_streaming_buffer" not in head.__dict__


def test_boundary_cleans_failure_and_releases_lock_even_when_clear_fails():
    root = Root()
    calls = []

    def clear():
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError("cleanup failed")

    boundary = SerialForwardBoundary([root], clear)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        root.forward()
    assert root.forward("next") == "next"
    boundary.close()


def test_boundary_cleans_partial_forward_and_preserves_instance_override():
    root = Root()
    retained = []

    def failing():
        retained.append(object())
        raise RuntimeError("partial root")

    root.forward = failing
    boundary = SerialForwardBoundary([root], retained.clear)
    with pytest.raises(RuntimeError, match="partial root"):
        root.forward()
    assert not retained
    boundary.close()
    assert root.forward is failing


def test_boundary_rolls_back_partial_installation():
    root = Root()
    with pytest.raises(AttributeError):
        SerialForwardBoundary([root, object()], lambda: None)
    assert "forward" not in root.__dict__


def test_closed_boundary_rejects_saved_wrapper_without_resurrecting_patches():
    root = Root()
    boundary = SerialForwardBoundary([root], lambda: None)
    previous = root.forward
    boundary.close()
    with pytest.raises(RuntimeError, match="closed"):
        previous()
    boundary.close()
    assert "forward" not in root.__dict__


def test_boundary_cannot_close_inside_an_active_root():
    root = Root()
    boundary = SerialForwardBoundary([root], lambda: None)
    with pytest.raises(RuntimeError, match="active"):
        root.forward(boundary.close)
    assert root.forward("still open") == "still open"
    boundary.close()
    assert "forward" not in root.__dict__


@pytest.fixture
def fake_runtime(monkeypatch):
    events = []
    head = Root()
    head.vl_self_attention, head.model = Root(), Root()
    model = SimpleNamespace(
        training=False,
        config=SimpleNamespace(streaming=True),
        parameters=lambda: iter(
            [SimpleNamespace(dtype="bf16", device=SimpleNamespace(type="cuda"))]
        ),
        backbone=Root(),
        action_head=head,
        named_modules=list,
    )
    sites = {
        name: (object(), "role", "scope")
        for name in (
            "backbone.visual.qkv",
            "backbone.text.q_proj",
            "action_head.vl_self_attention.to_q",
            "action_head.model.ffn",
        )
    }
    state = SimpleNamespace(events=events, model=model, sites=sites)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(
        sys.modules,
        "coexecution.optimization.groot.projections",
        SimpleNamespace(inventory=lambda *a, **k: (sites, [], [], [])),
    )

    def preflight(config, device):
        events.append("preflight")
        return {"backend_id": "test", "quant_format": config.precision}

    monkeypatch.setattr(session, "preflight_backend", preflight)
    monkeypatch.setattr(session, "apply_tactics", lambda *a, **k: None)

    class Integer:
        def __init__(self, model, executed, **kwargs):
            events.append("integer-install")
            self.sites = {
                name: value for name, value in sites.items() if name in executed
            }
            self.groups = []
            state.controller = self

        def set(self, precision):
            events.append(precision)

        def clear_projection_cache(self):
            events.append("clear")

        def close(self):
            events.append("integer-close")

    class Extra:
        def __init__(self, *args):
            pass

        def set(self, *args, **kwargs):
            pass

        def close(self):
            events.append("extra-close")

    class Floating:
        def __init__(self, model, executed, **kwargs):
            events.append("floating-install")
            self.sites = {
                name: value for name, value in sites.items() if name in executed
            }
            self.groups = [SimpleNamespace(active=False, cache=None)]
            self.coverage = {"linears": [{"name": name} for name in self.sites]}
            state.controller = self

        def close(self):
            events.append("floating-close")

    monkeypatch.setitem(
        sys.modules,
        "coexecution.optimization.groot.integer",
        SimpleNamespace(TransformerINT=Integer),
    )
    monkeypatch.setitem(
        sys.modules,
        "coexecution.optimization.groot.integer_connections",
        SimpleNamespace(AdditionalConnections=Extra),
    )
    monkeypatch.setitem(
        sys.modules,
        "coexecution.optimization.groot.floating",
        SimpleNamespace(FloatingProjections=Floating),
    )
    return state


@pytest.mark.parametrize("precision", ["w8a8", "w4a4", "fp8", "fp4"])
@pytest.mark.parametrize("component", ["all", "vlm", "dit"])
def test_session_applies_the_same_component_filter_to_both_backends(
    fake_runtime, precision, component
):
    state = fake_runtime
    config = OptimizationConfig(precision=precision, component=component)
    with session.GrootOptimizations(
        SimpleNamespace(model=state.model), config
    ) as scope:
        selected = {name for name in state.sites if selected_component(name, component)}
        assert set(state.controller.sites) == selected
        assert scope.evidence()["coverage"]["component"] == component
        assert scope.evidence()["backend"]["backend_id"] == "test"
        assert state.events[0] == "preflight"
        state.model.action_head.vl_self_attention.forward()
        state.model.action_head.reset_streaming_buffer()
    assert "forward" not in state.model.action_head.model.__dict__


def test_coverage_intersection_cannot_quantize_the_other_component(
    fake_runtime, tmp_path
):
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"linears": {"action_head.model.ffn": {}}}))
    config = OptimizationConfig(precision="w8a8", component="vlm", coverage=path)
    with (
        pytest.raises(ValueError, match="no matching projections"),
        session.GrootOptimizations(SimpleNamespace(model=fake_runtime.model), config),
    ):
        pytest.fail("An empty component intersection cannot install projections")
    assert not fake_runtime.events


def test_preflight_failure_precedes_all_layout_and_forward_mutations(
    fake_runtime, monkeypatch
):
    def fail(*args):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(session, "preflight_backend", fail)
    config = OptimizationConfig(
        precision="w8a8", fusion=True, vision_channels_last=True
    )
    with (
        pytest.raises(RuntimeError, match="backend unavailable"),
        session.GrootOptimizations(SimpleNamespace(model=fake_runtime.model), config),
    ):
        pytest.fail("No patching is allowed after failed preflight")
    assert not fake_runtime.events
    assert "forward" not in fake_runtime.model.backbone.__dict__


def test_scope_close_during_root_does_not_restore_other_adapters(fake_runtime):
    state = fake_runtime
    with session.GrootOptimizations(
        SimpleNamespace(model=state.model), OptimizationConfig(precision="w8a8")
    ) as scope:
        with pytest.raises(RuntimeError, match="active"):
            state.model.action_head.model.forward(scope.close)
        assert "integer-close" not in state.events
        assert state.model.action_head.model.forward("still open") == "still open"
    assert "integer-close" in state.events


def test_scope_restores_adapters_even_if_boundary_cleanup_fails(fake_runtime):
    state = fake_runtime
    scope = session.GrootOptimizations(
        SimpleNamespace(model=state.model), OptimizationConfig(precision="w8a8")
    )
    scope.__enter__()

    def fail():
        raise RuntimeError("cleanup failed")

    scope.boundary.clear = fail
    with pytest.raises(RuntimeError, match="cleanup failed"):
        scope.close()
    assert "extra-close" in state.events
    assert "integer-close" in state.events
    assert "forward" not in state.model.action_head.model.__dict__


def test_graph_failure_keeps_outer_serial_boundary_and_eager_fallback(
    fake_runtime, monkeypatch
):
    state = fake_runtime

    class Graph:
        def __init__(self, model):
            self.model, self.original = model, model.forward

        def set(self, key):
            def fail(*args, **kwargs):
                self.model.forward = self.original
                raise RuntimeError("capture failed")

            self.model.forward = fail

        def close(self):
            state.events.append("graph-close")
            self.model.forward = self.original

    monkeypatch.setitem(
        sys.modules,
        "coexecution.optimization.execution.graph",
        SimpleNamespace(DitGraphs=Graph),
    )
    config = OptimizationConfig(precision="w8a8", dit_graph=True)
    with session.GrootOptimizations(SimpleNamespace(model=state.model), config):
        model = state.model.action_head.model
        with pytest.raises(RuntimeError, match="capture failed"):
            model.forward()
        before = state.events.count("clear")
        assert model.forward("fallback") == "fallback"
        assert state.events.count("clear") == before + 2
    assert state.events.index("graph-close") < state.events.index("integer-close")


def test_root_specific_clears_keep_one_guard_and_full_reset_close_cleanup():
    first, second, head = Root(), Root(), Root()
    calls = []
    boundary = SerialForwardBoundary(
        [first, second],
        lambda: calls.append("full"),
        reset_modules=[head],
        forward_clears=[
            (first, lambda: calls.append("first")),
            (second, lambda: calls.append("second")),
        ],
    )
    assert first.forward(1) == 1
    assert calls == ["first", "first"]
    with pytest.raises(RuntimeError, match="serial submissions"):
        first.forward(second.forward)
    assert calls[-2:] == ["first", "first"]
    second.forward(2)
    assert calls[-2:] == ["second", "second"]
    head.reset_streaming_buffer()
    assert calls[-2:] == ["full", "full"]
    boundary.close()
    assert calls[-1] == "full"


def test_root_specific_clear_failure_still_cleans_and_restores_guard():
    root = Root()
    calls = []

    def clear():
        calls.append("root")
        if len(calls) == 1:
            raise RuntimeError("clear failed")

    boundary = SerialForwardBoundary(
        [root], lambda: calls.append("full"), forward_clears=[(root, clear)]
    )
    with pytest.raises(RuntimeError, match="clear failed"):
        root.forward()
    assert calls == ["root", "root"]
    assert root.forward(3) == 3
    boundary.close()
    assert calls[-1] == "full"


@pytest.mark.parametrize("invalid", ["unknown", "duplicate", "noncallable"])
def test_root_clear_validation_precedes_module_modification(invalid):
    root, other = Root(), Root()
    entries = {
        "unknown": [(other, lambda: None)],
        "duplicate": [(root, lambda: None), (root, lambda: None)],
        "noncallable": [(root, None)],
    }[invalid]
    with pytest.raises((ValueError, TypeError)):
        SerialForwardBoundary([root], lambda: None, forward_clears=entries)
    assert "forward" not in root.__dict__


def test_projection_clearers_touch_only_active_precision_and_called_component():
    counts = {}

    class Remaining(set):
        def __init__(self, key):
            super().__init__([0, 1])
            self.key = key

        def clear(self):
            counts[self.key] = counts.get(self.key, 0) + 1
            super().clear()

    prefixes = {
        "vlm": "backbone.text",
        "s1_feature": "action_head.vl_self_attention.block",
        "dit": "action_head.model.block",
    }
    grouped, groups = {}, {}
    for bits in (8, 4):
        grouped[bits] = []
        for component, prefix in prefixes.items():
            group = SimpleNamespace(
                _input=object(),
                _output=object(),
                _remaining=Remaining((bits, component)),
            )
            grouped[bits].append(([prefix + ".q", prefix + ".k"], group))
            groups[bits, component] = group
    clearers = projection_cache_clearers(grouped, "w8a8")
    clearers["s1_feature"]()
    assert counts == {(8, "s1_feature"): 1}
    assert groups[8, "s1_feature"]._input is None
    assert groups[8, "s1_feature"]._output is None
    assert all(
        group._input is not None and group._output is not None and group._remaining
        for key, group in groups.items()
        if key != (8, "s1_feature")
    )
    projection_cache_clearers(grouped, "w4a4")["vlm"]()
    assert counts == {(8, "s1_feature"): 1, (4, "vlm"): 1}
    assert {name: len(clearer.groups) for name, clearer in clearers.items()} == {
        "vlm": 1,
        "s1_feature": 1,
        "dit": 1,
    }


@pytest.mark.parametrize("invalid", ["mixed", "unknown", "empty", "shared"])
def test_projection_clear_partition_rejects_ambiguous_ownership(invalid):
    group = SimpleNamespace()
    records = {
        "mixed": [(["backbone.q", "action_head.model.k"], group)],
        "unknown": [(["elsewhere.q"], group)],
        "empty": [([], group)],
        "shared": [(["backbone.q"], group), (["backbone.k"], group)],
    }[invalid]
    with pytest.raises(ValueError):
        projection_cache_clearers({8: records, 4: []}, "w8a8")
