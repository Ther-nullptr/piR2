"""Configuration and transactional lifecycle contracts; no model/GPU imports."""

import argparse
import json
import sys
from contextlib import contextmanager, nullcontext
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


def test_fp4_swap_ab_is_opt_in_and_precision_specific():
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    config = optimization_config(
        parser.parse_args(["--inference-precision", "fp4", "--fp4-swap-ab"])
    )
    assert config.fp4_swap_ab
    assert not OptimizationConfig(precision="fp4").fp4_swap_ab
    for precision in ("bf16", "fp8", "w8a8", "w4a4"):
        with pytest.raises(ValueError, match="Swap-AB requires precision=fp4"):
            OptimizationConfig(precision=precision, fp4_swap_ab=True)


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
def test_floating_precision_is_opt_in_and_accepts_serial_graph(precision):
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    config = optimization_config(
        parser.parse_args(["--inference-precision", precision, "--dit-cuda-graph"])
    )
    assert config.precision == precision and config.dit_graph
    config.validate_variant("pir2")


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
def test_shared_floating_inputs_are_explicit_and_precision_specific(precision):
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    config = optimization_config(
        parser.parse_args(["--inference-precision", precision, "--fp-shared-inputs"])
    )
    assert config.fp_shared_inputs
    assert not OptimizationConfig().fp_shared_inputs
    for incompatible in ("bf16", "w8a8", "w4a4"):
        with pytest.raises(ValueError, match="Shared floating inputs"):
            OptimizationConfig(precision=incompatible, fp_shared_inputs=True)


@pytest.mark.parametrize(
    "options",
    [
        {"category_id": 2, "scope": "all"},
        {"tactics": Path("integer.json")},
        {"group_tactics": Path("integer.json")},
        {"shape_tactics": Path("integer.json")},
        {"group_conditioning": True, "scope": "all", "fusion": True},
    ],
)
@pytest.mark.parametrize("precision", ["fp8", "fp4"])
def test_floating_precision_rejects_integer_only_options(options, precision):
    with pytest.raises(ValueError, match="Floating precision does not support integer"):
        OptimizationConfig(precision=precision, **options)


@pytest.mark.parametrize("precision", ["w8a8", "w4a4", "fp8", "fp4"])
def test_norm_residual_fusion_flags_require_lowbit_fusion(precision):
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
        with pytest.raises(ValueError, match="fusion and low-bit"):
            OptimizationConfig(norm_modulation_quant=True, **options)
    with pytest.raises(ValueError, match="requires norm_modulation_quant"):
        OptimizationConfig(precision=precision, fusion=True, residual_norm_quant=True)


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
def test_floating_group_and_swiglu_flags(precision):
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    config = optimization_config(
        parser.parse_args(
            [
                "--inference-precision",
                precision,
                "--fp-grouped",
                "--fp-swiglu",
            ]
        )
    )
    assert config.fp_grouped and config.fp_swiglu
    assert not OptimizationConfig().fp_grouped and not OptimizationConfig().fp_swiglu
    for incompatible in ("bf16", "w8a8", "w4a4"):
        for flag in ("fp_grouped", "fp_swiglu"):
            with pytest.raises(ValueError, match="Floating fusion requires"):
                OptimizationConfig(precision=incompatible, **{flag: True})


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


@pytest.mark.parametrize("precision", ["bf16", "w8a8", "w4a4", "fp8", "fp4"])
def test_input_preparation_is_opt_in_and_precision_independent(precision):
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    config = optimization_config(
        parser.parse_args(["--inference-precision", precision, "--input-preparation"])
    )
    assert config.input_preparation and config.enabled
    assert not OptimizationConfig().input_preparation
    assert not config.fusion and not config.dit_graph
    config.validate_variant("flow")
    config.validate_variant("pir2")
    assert GrootOptimizations(object(), config).evidence()["config"][
        "input_preparation"
    ]


@pytest.mark.parametrize("precision", ["bf16", "w8a8", "w4a4", "fp8", "fp4"])
def test_tokenizer_cache_is_opt_in_and_precision_independent(precision):
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    config = optimization_config(
        parser.parse_args(["--inference-precision", precision, "--tokenizer-cache"])
    )
    assert config.tokenizer_cache and config.enabled
    assert not OptimizationConfig().tokenizer_cache
    assert not config.fusion and not config.dit_graph
    config.validate_variant("flow")
    config.validate_variant("pir2")
    assert GrootOptimizations(object(), config).evidence()["config"]["tokenizer_cache"]


@pytest.fixture
def runtime_interface(monkeypatch):
    """Lifecycle sentinels; no real tensors, tokenizers, neural modules or kernels."""
    events, failures = [], []
    model = SimpleNamespace(
        training=False,
        config=SimpleNamespace(streaming=False),
        parameters=lambda: iter(
            [SimpleNamespace(dtype="bf16", device=SimpleNamespace(type="cuda"))]
        ),
        named_modules=lambda: iter([]),
        backbone=SimpleNamespace(forward=lambda inputs: inputs),
        action_head=SimpleNamespace(
            model=SimpleNamespace(forward=lambda inputs: inputs)
        ),
    )
    policy = SimpleNamespace(model=model)

    class Adapter:
        label = "preparation"

        def __init__(self, actual_policy, *, guard_backend=None):
            assert actual_policy is policy
            self.closed = True
            self.active_call = False
            self.guard_backend = guard_backend
            events.append("created " + self.label)

        def __enter__(self):
            events.append("entered " + self.label)
            if failures and failures[0] == self.label:
                failures.pop(0)
                raise RuntimeError(self.label + " installation failed")
            self.closed = False
            return self

        def close(self):
            if not self.closed:
                events.append("closed " + self.label)
                self.closed = True

        def evidence(self):
            return {"adapter": self.label}

        @contextmanager
        def close_guard(self):
            if self.active_call:
                raise RuntimeError("Cannot change runtime during an active call")
            yield

    class Tokenizer(Adapter):
        label = "tokenizer"

    class Images(Adapter):
        label = "images"

    class Static(Adapter):
        label = "static"

        def __init__(
            self, actual_policy, *, cuda_graph=False, vision_rope=None, **kwargs
        ):
            super().__init__(actual_policy, **kwargs)
            self.cuda_graph, self.vision_rope = cuda_graph, vision_rope

    class Action(Adapter):
        label = "action"

    class Boundary:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.active_call = False
            events.append("created boundary")

        def close(self):
            if self.active_call:
                raise RuntimeError("Cannot close boundary during an active call")
            if not self.closed:
                events.append("closed boundary")
                self.closed = True

    class Graphs:
        def __init__(self, head):
            self.cache = object()
            events.append("created graphs")

        def set(self, precision):
            events.append("set graphs " + precision)

        def close(self):
            events.append("closed graphs")

        def evidence(self):
            return {"captures": 0}

    class Floating:
        def __init__(self, *args, **kwargs):
            self.groups = []
            self.coverage = {}
            events.append("created floating")

        def close(self):
            events.append("closed floating")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    for name, exports in (
        (
            "coexecution.optimization.execution.preparation",
            {"InputPreparation": Adapter},
        ),
        (
            "coexecution.optimization.execution.tokenization",
            {"TokenizerCacheScope": Tokenizer},
        ),
        (
            "coexecution.optimization.execution.images",
            {"ImageArrayPreparation": Images},
        ),
        ("coexecution.optimization.groot.vlm", {"PolicyStaticS2": Static}),
        ("coexecution.optimization.groot.action", {"ActionGraph": Action}),
        ("coexecution.optimization.execution.graph", {"DitGraphs": Graphs}),
        ("coexecution.optimization.groot.floating", {"FloatingProjections": Floating}),
        (
            "coexecution.optimization.groot.projections",
            {"inventory": lambda *a, **k: ({"backbone.linear": {}}, [], [], [])},
        ),
    ):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**exports))
    monkeypatch.setattr(
        "coexecution.optimization.session.SerialForwardBoundary", Boundary
    )
    monkeypatch.setattr(
        "coexecution.optimization.session.preflight_backend", lambda *args: {}
    )
    return SimpleNamespace(policy=policy, model=model, events=events, failures=failures)


@pytest.mark.parametrize("flag", ["image_array_preparation", "compiled_host_guards"])
@pytest.mark.parametrize("precision", ["bf16", "w8a8", "w4a4"])
def test_common_host_optimizations_are_explicit_and_precision_independent(
    flag, precision
):
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    config = optimization_config(
        parser.parse_args(
            ["--inference-precision", precision, "--" + flag.replace("_", "-")]
        )
    )
    assert getattr(config, flag) and config.enabled
    assert not getattr(OptimizationConfig(), flag)


def test_runtime_image_preparation_switch_rollback_and_live_call(runtime_interface):
    with GrootOptimizations(
        runtime_interface.policy, OptimizationConfig(image_array_preparation=True)
    ) as session:
        assert session.evidence()["image_array_preparation"] == {"adapter": "images"}
        session.image_preparation.active_call = True
        with pytest.raises(RuntimeError, match="active call"):
            session.set_runtime(image_array_preparation=False)
        session.image_preparation.active_call = False
        runtime_interface.failures.append("tokenizer")
        with pytest.raises(RuntimeError, match="installation failed"):
            session.set_runtime(image_array_preparation=False, tokenizer_cache=True)
        assert (
            session.config.image_array_preparation
            and not session.image_preparation.closed
        )
        session.set_runtime(image_array_preparation=False)
        assert session.image_preparation.closed


def test_runtime_compiled_backend_reinstalls_guards_and_rolls_back(
    runtime_interface, monkeypatch
):
    from coexecution.optimization.execution import compiled_guards

    backend = SimpleNamespace(evidence=lambda: {"backend": "cython"})
    monkeypatch.setattr(compiled_guards, "load_compiled_guards", lambda: backend)
    with GrootOptimizations(
        runtime_interface.policy, OptimizationConfig(action_graph=True, static_vlm=True)
    ) as session:
        old_action, old_vlm = session.action_graph, session.static_vlm
        session.set_runtime(compiled_host_guards=True)
        assert old_action.closed and old_vlm.closed
        assert session.action_graph.guard_backend is backend
        assert session.static_vlm.guard_backend is backend
        assert session.evidence()["compiled_host_guards"] == {"backend": "cython"}
        session.set_runtime(compiled_host_guards=False)
        assert (
            session.action_graph.guard_backend is None
            and session.static_vlm.guard_backend is None
        )

        def unavailable():
            raise RuntimeError("native backend unavailable")

        monkeypatch.setattr(compiled_guards, "load_compiled_guards", unavailable)
        with pytest.raises(RuntimeError, match="native backend unavailable"):
            session.set_runtime(compiled_host_guards=True)
        assert not session.config.compiled_host_guards
        assert not session.action_graph.closed and not session.static_vlm.closed


def test_action_graph_installs_after_boundary_and_supersedes_nested_dit(
    runtime_interface,
):
    with GrootOptimizations(
        runtime_interface.policy,
        OptimizationConfig(action_graph=True, dit_graph=True),
    ) as session:
        assert session.graphs is None
        assert session.evidence()["dit_graph_superseded"]
        assert runtime_interface.events == [
            "created boundary",
            "created action",
            "entered action",
        ]
        assert session.evidence()["action_graph"] == {"adapter": "action"}
        runtime_interface.events.clear()
        session.set_runtime(input_preparation=True)
        assert runtime_interface.events == [
            "closed boundary",
            "closed action",
            "created preparation",
            "entered preparation",
            "created boundary",
            "entered action",
        ]
    assert runtime_interface.events[-3:] == [
        "closed boundary",
        "closed action",
        "closed preparation",
    ]


def test_live_action_call_prevents_runtime_switch_and_session_close(runtime_interface):
    with GrootOptimizations(
        runtime_interface.policy,
        OptimizationConfig(action_graph=True),
    ) as session:
        previous = list(runtime_interface.events)
        session.action_graph.active_call = True
        for operation in (
            session.close,
            lambda: session.set_runtime(input_preparation=True),
        ):
            with pytest.raises(RuntimeError, match="active call"):
                operation()
            assert runtime_interface.events == previous
            assert not session.boundary.closed
        session.action_graph.active_call = False


def test_action_install_failure_restores_runtime_transaction(runtime_interface):
    runtime_interface.failures.append("action")
    with (
        pytest.raises(RuntimeError, match="action installation failed"),
        GrootOptimizations(
            runtime_interface.policy,
            OptimizationConfig(action_graph=True, input_preparation=True),
        ),
    ):
        pass
    assert "closed boundary" in runtime_interface.events
    assert "closed preparation" in runtime_interface.events


def test_tokenizer_and_preparation_install_before_outer_boundary(runtime_interface):
    with GrootOptimizations(
        runtime_interface.policy,
        OptimizationConfig(
            input_preparation=True,
            tokenizer_cache=True,
        ),
    ) as session:
        assert runtime_interface.events == [
            "created preparation",
            "entered preparation",
            "created tokenizer",
            "entered tokenizer",
            "created boundary",
        ]
        assert session.evidence()["tokenizer_cache"] == {"adapter": "tokenizer"}
    assert runtime_interface.events[-3:] == [
        "closed boundary",
        "closed tokenizer",
        "closed preparation",
    ]


def test_runtime_switch_retains_fixed_precision_controller_and_dit_graph(
    runtime_interface,
):
    with GrootOptimizations(
        runtime_interface.policy,
        OptimizationConfig(
            precision="fp8",
            input_preparation=True,
            dit_graph=True,
        ),
    ) as session:
        graphs, graph_cache = session.graphs, session.graphs.cache
        public_cleanup = []
        session.stack.callback(public_cleanup.append, "fixed cleanup")
        for _ in range(3):
            session.set_runtime(
                input_preparation=False,
                tokenizer_cache=True,
                static_vlm=False,
                vlm_graph=False,
            )
            session.set_runtime(input_preparation=True, tokenizer_cache=False)
        assert session.graphs is graphs and graphs.cache is graph_cache
        assert session.config.precision == "fp8" and session.config.dit_graph
        assert session.evidence()["dit_graph"] == {"captures": 0}
        assert runtime_interface.events.count("created floating") == 1
        assert runtime_interface.events.count("created graphs") == 1
        assert "closed floating" not in runtime_interface.events
        assert "closed graphs" not in runtime_interface.events
        assert not public_cleanup
    assert public_cleanup == ["fixed cleanup"]
    assert runtime_interface.events.count("closed floating") == 1
    assert runtime_interface.events.count("closed graphs") == 1
    assert runtime_interface.events.count("closed preparation") == 4
    assert runtime_interface.events.count("closed tokenizer") == 3


def test_runtime_switch_closes_boundary_first_and_reinstalls_it_last(runtime_interface):
    with GrootOptimizations(
        runtime_interface.policy,
        OptimizationConfig(
            input_preparation=True,
        ),
    ) as session:
        runtime_interface.events.clear()
        session.set_runtime(input_preparation=False, tokenizer_cache=True)
        assert runtime_interface.events == [
            "closed boundary",
            "closed preparation",
            "created tokenizer",
            "entered tokenizer",
            "created boundary",
        ]


def test_runtime_switch_failure_rolls_back_to_previous_adapters(runtime_interface):
    with GrootOptimizations(
        runtime_interface.policy,
        OptimizationConfig(
            input_preparation=True,
        ),
    ) as session:
        runtime_interface.failures.append("tokenizer")
        with pytest.raises(RuntimeError, match="tokenizer installation failed"):
            session.set_runtime(input_preparation=False, tokenizer_cache=True)
        assert session.config.input_preparation and not session.config.tokenizer_cache
        assert not session.preparation.closed and not session.boundary.closed
        session.set_runtime(input_preparation=False, tokenizer_cache=True)
        assert session.config.tokenizer_cache and not session.tokenizer_cache.closed


def test_runtime_switch_restores_serial_boundary_after_partial_cleanup_failure(
    runtime_interface,
    monkeypatch,
):
    from coexecution.optimization.execution.cache import SerialForwardBoundary

    monkeypatch.setattr(
        "coexecution.optimization.session.SerialForwardBoundary", SerialForwardBoundary
    )
    with GrootOptimizations(
        runtime_interface.policy,
        OptimizationConfig(
            precision="fp8",
            input_preparation=True,
            dit_graph=True,
        ),
    ) as session:
        previous_boundary = session.boundary
        graphs, graph_cache = session.graphs, session.graphs.cache
        fail_once = True

        def clear():
            nonlocal fail_once
            if fail_once:
                fail_once = False
                raise RuntimeError("one-shot boundary cleanup failed")

        session._clear_cache = previous_boundary.clear = clear
        with pytest.raises(RuntimeError, match="one-shot boundary cleanup failed"):
            session.set_runtime(input_preparation=False, tokenizer_cache=True)
        assert previous_boundary.closed
        assert session.boundary is not previous_boundary and not session.boundary.closed
        assert session.config.input_preparation and not session.config.tokenizer_cache
        assert not session.preparation.closed
        assert session.graphs is graphs and graphs.cache is graph_cache
        assert runtime_interface.events.count("created floating") == 1
        assert "closed floating" not in runtime_interface.events
        assert "closed graphs" not in runtime_interface.events
        assert session.boundary.lock.acquire(blocking=False)
        try:
            with pytest.raises(RuntimeError, match="serial submissions"):
                runtime_interface.model.backbone.forward("overlap")
        finally:
            session.boundary.lock.release()
        assert (
            runtime_interface.model.backbone.forward("still guarded") == "still guarded"
        )


def test_runtime_switch_live_real_boundary_preserves_every_installed_adapter(
    runtime_interface,
    monkeypatch,
):
    from coexecution.optimization.execution.cache import SerialForwardBoundary

    monkeypatch.setattr(
        "coexecution.optimization.session.SerialForwardBoundary", SerialForwardBoundary
    )
    runtime_interface.model.backbone.forward = lambda callback: callback()
    with GrootOptimizations(
        runtime_interface.policy,
        OptimizationConfig(
            input_preparation=True,
            tokenizer_cache=True,
        ),
    ) as session:
        boundary = session.boundary
        before = list(runtime_interface.events)
        with pytest.raises(RuntimeError, match="active call"):
            runtime_interface.model.backbone.forward(
                lambda: session.set_runtime(
                    input_preparation=False,
                    tokenizer_cache=False,
                )
            )
        assert session.boundary is boundary and not boundary.closed
        assert session.config.input_preparation and session.config.tokenizer_cache
        assert not session.preparation.closed and not session.tokenizer_cache.closed
        assert runtime_interface.events == before


def test_failed_runtime_rollback_closes_fixed_controllers_and_session(
    runtime_interface,
):
    with GrootOptimizations(
        runtime_interface.policy,
        OptimizationConfig(
            precision="fp8",
            input_preparation=True,
            dit_graph=True,
        ),
    ) as session:
        runtime_interface.failures.extend(["tokenizer", "preparation"])
        with pytest.raises(RuntimeError, match="rollback.*closed"):
            session.set_runtime(input_preparation=False, tokenizer_cache=True)
        assert runtime_interface.events.count("closed floating") == 1
        assert runtime_interface.events.count("closed graphs") == 1
        with pytest.raises(RuntimeError, match="active"):
            session.set_runtime(input_preparation=True)


@pytest.mark.parametrize("active", ["preparation", "tokenizer_cache", "boundary"])
def test_runtime_switch_rejects_active_call_before_any_restoration(
    runtime_interface, active
):
    with GrootOptimizations(
        runtime_interface.policy,
        OptimizationConfig(
            input_preparation=True,
            tokenizer_cache=True,
        ),
    ) as session:
        target = getattr(session, active)
        target.active_call = True
        before = list(runtime_interface.events)
        try:
            with pytest.raises(RuntimeError, match="active call"):
                session.set_runtime(input_preparation=False, tokenizer_cache=False)
            assert runtime_interface.events == before
            assert session.config.input_preparation and session.config.tokenizer_cache
        finally:
            target.active_call = False


def test_default_active_session_can_switch_runtime_without_precision_install(
    runtime_interface,
):
    with GrootOptimizations(runtime_interface.policy) as session:
        assert not runtime_interface.events
        session.set_runtime(input_preparation=True)
        assert session.config.precision == "bf16" and session.config.input_preparation
        session.set_runtime(input_preparation=False)
        assert not session.config.enabled
        assert session.boundary is None


def test_runtime_switch_accepts_disabled_vlm_options_without_installing(
    runtime_interface,
):
    with GrootOptimizations(runtime_interface.policy) as session:
        assert session.set_runtime(static_vlm=False, vlm_graph=False) is session
        assert not runtime_interface.events


@pytest.mark.parametrize("precision", ["bf16", "w8a8", "w4a4", "fp8", "fp4"])
def test_static_vlm_and_graph_flags_are_opt_in_and_precision_independent(precision):
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    config = optimization_config(
        parser.parse_args(
            ["--inference-precision", precision, "--static-vlm", "--vlm-cuda-graph"]
        )
    )
    assert config.static_vlm and config.vlm_graph and config.enabled
    assert not OptimizationConfig().static_vlm and not OptimizationConfig().vlm_graph
    config.validate_variant("flow")
    config.validate_variant("pir2")
    assert GrootOptimizations(object(), config).evidence()["config"]["vlm_graph"]


def test_vlm_graph_requires_static_metadata_before_any_installation():
    with pytest.raises(ValueError, match="requires static_vlm"):
        OptimizationConfig(vlm_graph=True)


def test_static_vlm_installs_after_fixed_precision_before_outer_boundary(
    runtime_interface,
):
    with GrootOptimizations(
        runtime_interface.policy,
        OptimizationConfig(
            precision="fp8", dit_graph=True, static_vlm=True, vlm_graph=True
        ),
    ) as session:
        assert runtime_interface.events == [
            "created floating",
            "created graphs",
            "set graphs fp8",
            "created static",
            "entered static",
            "created boundary",
        ]
        assert session.static_vlm.cuda_graph
        assert session.evidence()["static_vlm"] == {"adapter": "static"}
    assert runtime_interface.events[-4:] == [
        "closed boundary",
        "closed static",
        "closed graphs",
        "closed floating",
    ]


@pytest.mark.parametrize(
    "mode", ["vision_rope", "text_rope", "rope", "rmsnorm", "rope_rms", "all"]
)
@pytest.mark.parametrize("implementation", [None, "reference", "native"])
def test_static_vlm_preserves_the_selected_vision_rope(
    runtime_interface, monkeypatch, tmp_path, mode, implementation
):
    events = runtime_interface.events
    profile = None
    if implementation is not None:
        data = {
            "architecture": "sm_89",
            "precision": "bf16",
            "implementations": {"vision_rope": implementation},
        }
        profile = tmp_path / "operators.json"
        profile.write_text(json.dumps(data))
        monkeypatch.setattr(
            "coexecution.optimization.session.preflight_backend",
            lambda *args: {"operator_profile": data},
        )

    class Fusion:
        def __init__(self, *args, implementations):
            self.implementations = implementations
            self.vision = lambda *values: values
            events.append("created fusion")

        def set(self, mode):
            events.append("set fusion " + mode)

        def close(self):
            events.append("closed fusion")

    class Pointwise:
        def __init__(self, *args):
            events.append("created pointwise")

        def set(self, mode):
            events.append("set pointwise " + mode)

        def close(self):
            events.append("closed pointwise")

    monkeypatch.setitem(
        sys.modules, "coexecution.groot_fusion", SimpleNamespace(NonGemmAdapters=Fusion)
    )
    monkeypatch.setitem(
        sys.modules,
        "coexecution.groot_pointwise",
        SimpleNamespace(PointwiseFusion=Pointwise),
    )
    with GrootOptimizations(
        runtime_interface.policy,
        OptimizationConfig(
            fusion=True, fusion_mode=mode, operator_profile=profile, static_vlm=True
        ),
    ) as session:
        expected = (
            session.fusion.vision
            if mode in ("vision_rope", "rope", "rope_rms", "all")
            and implementation != "reference"
            else None
        )
        assert session.static_vlm.vision_rope is expected
        if mode == "all":
            assert events.index("set pointwise dit") < events.index("entered static")
        else:
            assert "created pointwise" not in events
        fusion = session.fusion
        session.set_runtime(vlm_graph=True)
        assert session.fusion is fusion
        assert session.static_vlm.vision_rope is expected
        assert session.config.fusion_mode == mode
        selection = session.evidence()["operator_selection"]
        if implementation is None:
            assert selection is None
        else:
            assert selection["implementations"] == {"vision_rope": implementation}
        assert events.count("created fusion") == 1


def test_static_vlm_runtime_switch_retains_dit_graph_and_recreates_graph_mode(
    runtime_interface,
):
    with GrootOptimizations(
        runtime_interface.policy,
        OptimizationConfig(precision="fp8", dit_graph=True, static_vlm=True),
    ) as session:
        graphs, static = session.graphs, session.static_vlm
        session.set_runtime(vlm_graph=True)
        assert session.static_vlm.cuda_graph and static.closed
        session.set_runtime(static_vlm=False, vlm_graph=False)
        assert (
            not session.config.static_vlm and session.evidence()["static_vlm"] is None
        )
        session.set_runtime(static_vlm=True, vlm_graph=True)
        assert session.static_vlm.cuda_graph
        assert session.graphs is graphs
        assert runtime_interface.events.count("created floating") == 1
        assert runtime_interface.events.count("created graphs") == 1


def test_static_vlm_runtime_failure_rolls_back_all_four_adapters(runtime_interface):
    with GrootOptimizations(
        runtime_interface.policy,
        OptimizationConfig(input_preparation=True, tokenizer_cache=True),
    ) as session:
        runtime_interface.failures.append("static")
        with pytest.raises(RuntimeError, match="static installation failed"):
            session.set_runtime(static_vlm=True, vlm_graph=True)
        assert session.config.input_preparation and session.config.tokenizer_cache
        assert not session.config.static_vlm and not session.config.vlm_graph
        assert not session.preparation.closed and not session.tokenizer_cache.closed
        assert not session.boundary.closed


@pytest.mark.parametrize("operation", ["set_runtime", "close"])
def test_static_busy_guard_rejects_before_any_other_restoration(
    runtime_interface, operation
):
    with GrootOptimizations(
        runtime_interface.policy,
        OptimizationConfig(input_preparation=True, static_vlm=True),
    ) as session:
        session.static_vlm.active_call = True
        before = list(runtime_interface.events)
        try:
            with pytest.raises(RuntimeError, match="active call"):
                if operation == "close":
                    session.close()
                else:
                    session.set_runtime(static_vlm=False)
            assert runtime_interface.events == before
            assert not session.preparation.closed and not session.boundary.closed
        finally:
            session.static_vlm.active_call = False


def test_runtime_rejects_disabling_static_while_graph_remains_enabled(
    runtime_interface,
):
    with GrootOptimizations(
        runtime_interface.policy, OptimizationConfig(static_vlm=True, vlm_graph=True)
    ) as session:
        before = list(runtime_interface.events)
        with pytest.raises(ValueError, match="requires static_vlm"):
            session.set_runtime(static_vlm=False)
        assert runtime_interface.events == before


def test_runtime_switch_requires_active_scope_and_fixed_model(runtime_interface):
    session = GrootOptimizations(
        runtime_interface.policy, OptimizationConfig(input_preparation=True)
    )
    with pytest.raises(RuntimeError, match="active"):
        session.set_runtime(tokenizer_cache=True)
    with session:
        runtime_interface.policy.model = object()
        with pytest.raises(RuntimeError, match="replaced"):
            session.set_runtime(tokenizer_cache=True)
    with pytest.raises(RuntimeError, match="active"):
        session.set_runtime(tokenizer_cache=True)


def test_input_preparation_enters_before_boundary_and_closes_after_it(monkeypatch):
    calls = []
    model = SimpleNamespace(
        training=False,
        config=SimpleNamespace(streaming=False),
        parameters=lambda: iter(
            [SimpleNamespace(dtype="bf16", device=SimpleNamespace(type="cuda"))]
        ),
    )
    policy = SimpleNamespace(model=model)

    class Preparation:
        def __init__(self, actual_policy):
            assert actual_policy is policy
            calls.append("created preparation")

        def __enter__(self):
            calls.append("entered preparation")
            return self

        def close(self):
            calls.append("closed preparation")

        def evidence(self):
            return {"backbone_keys": ["pixel_values"]}

        def close_guard(self):
            return nullcontext()

    class Boundary:
        closed = False

        def __init__(self, *args, **kwargs):
            calls.append("created boundary")

        def close(self):
            if not self.closed:
                calls.append("closed boundary")
                self.closed = True

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(
        sys.modules,
        "coexecution.optimization.execution.preparation",
        SimpleNamespace(InputPreparation=Preparation),
    )
    monkeypatch.setattr(
        "coexecution.optimization.session.SerialForwardBoundary", Boundary
    )
    monkeypatch.setattr(
        "coexecution.optimization.session.preflight_backend", lambda *args: {}
    )
    with GrootOptimizations(
        policy, OptimizationConfig(input_preparation=True)
    ) as session:
        assert calls[:3] == [
            "created preparation",
            "entered preparation",
            "created boundary",
        ]
        assert session.evidence()["input_preparation"] == {
            "backbone_keys": ["pixel_values"]
        }
    assert calls[-2:] == ["closed boundary", "closed preparation"]


def test_preparation_entry_failure_is_registered_for_cleanup_immediately(monkeypatch):
    calls = []
    model = SimpleNamespace(
        training=False,
        config=SimpleNamespace(streaming=False),
        parameters=lambda: iter(
            [SimpleNamespace(dtype="bf16", device=SimpleNamespace(type="cuda"))]
        ),
    )

    class Preparation:
        def __init__(self, policy):
            pass

        def __enter__(self):
            raise RuntimeError("preparation entry failed")

        def close(self):
            calls.append("closed preparation")

        def close_guard(self):
            return nullcontext()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(
        sys.modules,
        "coexecution.optimization.execution.preparation",
        SimpleNamespace(InputPreparation=Preparation),
    )
    monkeypatch.setattr(
        "coexecution.optimization.session.preflight_backend", lambda *args: {}
    )
    with (
        pytest.raises(RuntimeError, match="preparation entry"),
        GrootOptimizations(
            SimpleNamespace(model=model),
            OptimizationConfig(input_preparation=True),
        ),
    ):
        pytest.fail("A failed preparation installation cannot enter the session")
    assert calls == ["closed preparation"]


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


def test_fast_fp8_is_opt_in_and_rejects_other_precisions():
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    config = optimization_config(
        parser.parse_args(["--inference-precision", "fp8", "--fp8-fast-quant"])
    )
    assert config.fp8_fast_quant
    assert not OptimizationConfig(precision="fp8").fp8_fast_quant
    for precision in ("bf16", "w8a8", "w4a4", "fp4"):
        with pytest.raises(ValueError, match="requires precision=fp8"):
            OptimizationConfig(precision=precision, fp8_fast_quant=True)


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
        "coexecution.optimization.groot.integer",
        SimpleNamespace(
            TransformerINT=lambda *a, **k: pytest.fail("No sites to install"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "coexecution.optimization.groot.projections",
        SimpleNamespace(inventory=lambda *a, **k: ({"new.name": object()}, [], [], [])),
    )
    monkeypatch.setitem(
        sys.modules,
        "coexecution.optimization.groot.integer_connections",
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
        "coexecution.optimization.backends.integer.linear",
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
        Path(__file__).resolve().parents[2] / "configs/quantization/groot-libero10-ada"
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
