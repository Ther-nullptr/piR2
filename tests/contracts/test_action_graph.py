"""Action graph admission, dynamic inputs and accounting; no neural execution."""

import argparse
import importlib
import importlib.util
import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from coexecution.optimization.config import (
    OptimizationConfig,
    add_optimization_arguments,
    optimization_config,
)
from coexecution.optimization.execution.guards import HostGuardPlan


def adapter():
    name = "coexecution.optimization.groot.action"
    assert importlib.util.find_spec(name) is not None, (
        "Missing whole action graph adapter"
    )
    return importlib.import_module(name)


def test_action_graph_is_opt_in_and_flow_bf16_w8_w4_only():
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    assert "--action-cuda-graph" in parser.format_help()
    assert not optimization_config(parser.parse_args([])).action_graph
    for precision in ("bf16", "w8a8", "w4a4"):
        config = optimization_config(
            parser.parse_args(
                ["--action-cuda-graph", "--inference-precision", precision]
            )
        )
        assert config.action_graph and config.enabled
        config.validate_variant("flow")
        with pytest.raises(ValueError, match="Flow4"):
            config.validate_variant("pir2")
    for precision in ("fp8", "fp4"):
        with pytest.raises(ValueError, match="BF16, W8A8 or W4A4"):
            OptimizationConfig(precision=precision, action_graph=True)


def head_metadata(**changes):
    config = SimpleNamespace(
        streaming=False,
        action_horizon=40,
        state_history_length=1,
        image_delay_max=5,
    )
    for key, value in changes.items():
        setattr(config, key, value)
    return SimpleNamespace(config=config, num_inference_timesteps=4, action_horizon=40)


def test_flow_options_are_explicit_and_delay_is_part_of_the_signature():
    module = adapter()
    head = head_metadata()
    assert module.flow_options(head, None) == {"num_inference_timesteps": 4}
    request = {"num_inference_timesteps": 4, "image_delay": 2}
    actual = module.flow_options(head, request)
    assert actual == request and actual is not request
    assert module.flow_options(head, {"image_delay": 99})["image_delay"] == 5
    assert module.flow_options(head, {"image_delay": -2})["image_delay"] == 0
    assert module.flow_options(head, {"image_delay": 1}) != actual


@pytest.mark.parametrize(
    "options",
    [
        {"num_inference_timesteps": 3},
        {"num_inference_timesteps": "4"},
        {"num_inference_timesteps": True},
        {"inpaint": []},
        {"slide_steps": 1},
        {"force_nonstreaming": True},
        {"unknown": 1},
        {"image_delay": 1.5},
        {"image_delay": True},
    ],
)
def test_unsupported_options_fail_before_any_graph_or_rng(options):
    with pytest.raises(ValueError):
        adapter().flow_options(head_metadata(), options)


@pytest.mark.parametrize("changes", [{"streaming": True}, {"action_horizon": 16}])
def test_unsupported_model_contract_is_rejected(changes):
    with pytest.raises(ValueError, match="Flow4.*H40"):
        adapter().flow_options(head_metadata(**changes), None)


class TensorMetadata:
    def __init__(self, shape=(1, 13, 128), dtype="bf16", strides=(1664, 128, 1)):
        self.shape, self.dtype, self.strides = shape, dtype, strides
        self.device = "cuda:0"

    def stride(self):
        return self.strides


def test_graph_key_tracks_values_layout_and_stream_but_not_tensor_identity():
    module = adapter()
    first, second = TensorMetadata(), TensorMetadata()
    key = module.action_graph_key((first,), {"image_delay": 1}, 7)
    assert module.action_graph_key((second,), {"image_delay": 1}, 7) == key
    assert module.action_graph_key((second,), {"image_delay": 2}, 7) != key
    assert module.action_graph_key((second,), {"image_delay": 1}, 8) != key
    second.strides = (3328, 256, 1)
    assert module.action_graph_key((second,), {"image_delay": 1}, 7) != key
    second.strides, second.dtype = first.strides, "fp32"
    assert module.action_graph_key((second,), {"image_delay": 1}, 7) != key
    assert module.action_graph_key((first, None), {"image_delay": 1}, 7) != key


def test_noise_binding_is_exactly_once_and_never_calls_the_global_rng():
    module = adapter()
    noise = TensorMetadata((1, 40, 8))
    marker = object()
    real_torch = SimpleNamespace(ones_like=marker)
    proxy = module.ExplicitNoise(real_torch)
    assert proxy.ones_like is marker
    with proxy.bind(noise):
        assert (
            proxy.randn(size=noise.shape, dtype=noise.dtype, device=noise.device)
            is noise
        )
        with pytest.raises(RuntimeError, match="once"):
            proxy.randn(size=noise.shape, dtype=noise.dtype, device=noise.device)
    with pytest.raises(RuntimeError, match="binding"):
        proxy.randn(size=noise.shape, dtype=noise.dtype, device=noise.device)
    with pytest.raises(RuntimeError, match="once"), proxy.bind(noise):
        pass
    with pytest.raises(ValueError, match="noise"), proxy.bind(noise):
        proxy.randn(size=(2, 40, 8), dtype=noise.dtype, device=noise.device)


def test_delay_constant_uses_capture_safe_device_fill_without_host_tensor_copy():
    module = adapter()
    calls, output = [], object()

    def full(shape, value, *, dtype, device):
        calls.append((shape, value, dtype, device))
        return output

    proxy = module.ExplicitNoise(SimpleNamespace(long="int64", full=full))
    noise = TensorMetadata((1, 40, 8))
    with proxy.bind(noise):
        proxy.randn(size=noise.shape, dtype=noise.dtype, device=noise.device)
        assert proxy.tensor([2], dtype="int64", device=noise.device) is output
        with pytest.raises(ValueError, match="delay"):
            proxy.tensor([2, 3], dtype="int64", device=noise.device)
        with pytest.raises(ValueError, match="delay"):
            proxy.tensor([2], dtype="float32", device=noise.device)
    assert calls == [((1,), 2, "int64", "cuda:0")]


def test_graph_accounting_preserves_physical_hooks_and_proves_logical_nfe():
    from coexecution.libero import backend

    assert hasattr(backend, "action_execution_counts")
    action_execution_counts = backend.action_execution_counts

    before = {"vlm": 2, "dit": 10}
    after = {"vlm": 3, "dit": 10}
    graph_before = {
        "logical_nfe_count": 12,
        "replay_count": 3,
        "capture_count": 1,
        "failure_count": 0,
    }
    graph_after = {**graph_before, "logical_nfe_count": 16, "replay_count": 4}
    audit = action_execution_counts(before, after, graph_before, graph_after)
    assert audit["forward_counts"] == {"vlm": 1, "dit": 4}
    assert audit["python_forward_counts"] == {"vlm": 1, "dit": 0}
    assert audit["action_graph"]["replay_delta"] == 1
    assert audit["dit_count_source"] == "captured_flow4_graph"
    eager = action_execution_counts(before, {"vlm": 3, "dit": 14}, None, None)
    assert eager["forward_counts"] == eager["python_forward_counts"]
    with pytest.raises(RuntimeError, match="graph"):
        action_execution_counts(
            before, after, graph_before, {**graph_after, "logical_nfe_count": 15}
        )


def test_measurement_audits_one_action_graph_replay_as_four_logical_evaluations():
    from coexecution.libero.measurement import complete_graph_interval

    graph = {
        "capture_count": 1,
        "replay_count": 2,
        "failure_count": 0,
        "logical_nfe_count": 8,
    }
    before = {"dit": None, "vlm": None, "action": graph}
    after = {**before, "action": {**graph, "replay_count": 3, "logical_nfe_count": 12}}
    audit = complete_graph_interval(before, after)
    assert "action" in audit
    assert audit["passes"] and audit["action"]["replay_delta"] == 1
    after["action"]["logical_nfe_count"] = 11
    assert not complete_graph_interval(before, after)["passes"]


def test_tuple_graph_refreshes_owned_buffers_preserves_stride_and_stream(monkeypatch):
    """Metadata sentinels exercise ownership/stream ordering, never tensor math."""
    from coexecution.optimization.execution.tensor_graph import TensorGraphCall

    calls = []
    current = [SimpleNamespace(cuda_stream=7)]
    current[0].wait_stream = lambda stream: calls.append(
        ("caller_wait", stream.cuda_stream)
    )

    class Tensor(TensorMetadata):
        is_cuda = True
        layout = "strided"

        def copy_(self, source):
            self.source = source
            calls.append(("copy", self, source))
            return self

        def record_stream(self, stream):
            calls.append(("record", self, stream.cuda_stream))

        def clone(self):
            calls.append(("clone", self))
            return Tensor(self.shape, self.dtype, self.strides)

    class Graph:
        def replay(self):
            calls.append(("replay",))

    @contextmanager
    def stream_context(stream):
        original = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = original

    stream = SimpleNamespace(
        cuda_stream=8,
        wait_stream=lambda caller: calls.append(("capture_wait", caller.cuda_stream)),
    )
    cuda = SimpleNamespace(
        device=lambda device: nullcontext(),
        current_stream=lambda device: current[0],
        Stream=lambda device: stream,
        stream=stream_context,
        CUDAGraph=Graph,
        graph=lambda graph, stream: stream_context(stream),
    )
    torch = SimpleNamespace(
        Tensor=Tensor,
        strided="strided",
        cuda=cuda,
        empty_strided=lambda shape, strides, dtype, device: Tensor(
            shape, dtype, strides
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    body_calls = []

    def identity(*args):
        body_calls.append(tuple(args))
        return args

    first = Tensor(strides=(3328, 256, 1))
    constant = Tensor()
    graph = TensorGraphCall(identity, retained_tensors=(constant,))
    output = graph(first)[0]
    assert output is not first and output is not graph.inputs[0]
    assert graph.inputs[0].stride() == first.stride()
    assert len(body_calls) == 4 and graph.captures == graph.replays == 1
    second = Tensor(strides=first.strides)
    calls.clear()
    fresh = graph(second)[0]
    assert fresh is not output
    assert len(body_calls) == 4 and graph.captures == 1 and graph.replays == 2
    assert calls[0] == ("copy", graph.inputs[0], second)
    assert calls[1] == ("replay",)
    assert ("record", constant, 7) in calls
    current[0].cuda_stream = 9
    with pytest.raises(ValueError, match="stream changed"):
        graph(second)


def test_frozen_guard_allows_only_declared_observer_counter_state():
    calls, neural = [], ["fixed"]

    def observe(*unused):
        calls.append("observed")

    observe.__pir2_execution_observer__ = True

    def forward():
        return neural

    model = SimpleNamespace(
        training=False,
        _forward_hooks={1: observe},
        _forward_pre_hooks={},
        forward=forward,
    )
    model.named_modules = lambda: iter([("", model)])
    model.named_parameters = lambda: iter([])
    model.named_buffers = lambda: iter([])
    guard = HostGuardPlan(
        model,
        methods=("forward",),
        fields=(),
        global_bindings=(),
        observer_hooks=(observe,),
    )
    assert guard.unchanged()
    observe()
    assert guard.unchanged(), (
        "Counter changes are observational, not captured neural state"
    )
    neural.append("changed")
    assert not guard.unchanged(), "Neural closure state must remain frozen"
    neural.pop()
    assert guard.unchanged()
    model._forward_hooks[1] = lambda *unused: None
    assert not guard.unchanged()
    model._forward_hooks[1] = observe
    assert guard.unchanged()
    observe.__pir2_execution_observer__ = False
    assert not guard.unchanged()
    observe.__pir2_execution_observer__ = True
    observe.__code__ = forward.__code__
    assert not guard.unchanged()


def test_action_metadata_freezes_processor_dispatch_tuple_and_none_fields(monkeypatch):
    module = adapter()

    class Processor:
        def __call__(self, *args):
            raise AssertionError("Metadata contract must never execute attention")

    processor = Processor()
    attention = SimpleNamespace(
        training=False,
        processor=processor,
        normalized_shape=(128,),
        spatial_norm=None,
        compute_dtype=object(),
    )
    attention.named_modules = lambda: iter([("", attention)])
    attention.named_parameters = lambda: iter([])
    attention.named_buffers = lambda: iter([])
    guard = HostGuardPlan(attention, methods=(), fields=(), global_bindings=())
    assert hasattr(module, "freeze_action_metadata")
    module.freeze_action_metadata(guard, (attention,), processor_types=(Processor,))
    assert guard.unchanged()
    attention.processor = Processor()
    assert not guard.unchanged()
    attention.processor = processor
    assert guard.unchanged()
    attention.normalized_shape = (41, 128)
    assert not guard.unchanged()
    attention.normalized_shape = (128,)
    assert guard.unchanged()
    attention.spatial_norm = "new execution branch"
    assert not guard.unchanged()
    attention.spatial_norm = None
    dtype = attention.compute_dtype
    attention.compute_dtype = object()
    assert not guard.unchanged()
    attention.compute_dtype = dtype
    assert guard.unchanged()
    monkeypatch.setattr(Processor, "__call__", lambda *args: None)
    assert not guard.unchanged()


def test_action_metadata_rejects_unadmitted_processor_implementations():
    module = adapter()
    attention = SimpleNamespace(processor=lambda *args: None)
    guard = SimpleNamespace(attribute=lambda *args, **kwargs: None)
    assert hasattr(module, "freeze_action_metadata")
    with pytest.raises(ValueError, match="processor"):
        module.freeze_action_metadata(guard, (attention,), processor_types=())


@pytest.mark.parametrize("instance_method", [False, True])
def test_inactive_action_scope_cannot_restore_over_a_new_owner(instance_method):
    module = adapter()

    class Head:
        def get_action(self):
            raise AssertionError("Metadata-only restoration contract")

    head = Head()
    if instance_method:
        head.get_action = lambda: None
    original = head.get_action
    graph = module.ActionGraph(SimpleNamespace(model=SimpleNamespace(action_head=head)))
    graph._original = original
    graph._had_original = instance_method
    graph._entered = True
    head.get_action = graph._call
    graph.close()
    assert head.get_action == original
    newer = lambda: "new owner"
    head.get_action = newer
    graph.close()
    assert head.get_action is newer
    assert graph._original is None
