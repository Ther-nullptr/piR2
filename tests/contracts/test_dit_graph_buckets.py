"""Shape retention and cache bookkeeping using tensor metadata only."""

import importlib
import sys
from types import ModuleType

import pytest


class TensorMetadata:
    def __init__(self, shape, dtype="bf16", device="cuda:0", strides=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        if strides is None:
            values, stride = [], 1
            for width in reversed(shape):
                values.append(stride)
                stride *= width
            strides = tuple(reversed(values))
        self.strides = strides

    def stride(self):
        return self.strides


class MetadataGraph:
    def __init__(self, function):
        self.function, self.signature = function, None
        self.captures = self.replays = 0

    def __call__(self, *inputs):
        signature = tuple((v.shape, v.dtype, v.device, v.stride()) for v in inputs)
        if signature != self.signature:
            self.captures += 1
            self.signature = signature
        value = self.function(*inputs)
        self.replays += 1
        return value


@pytest.fixture
def graph_module(monkeypatch):
    provider = ModuleType("coexecution.kernels.common.graph")
    provider.CudaGraphCall = MetadataGraph
    monkeypatch.setitem(sys.modules, "coexecution.kernels.common.graph", provider)
    module = importlib.import_module("coexecution.optimization.execution.graph")
    monkeypatch.setattr(module, "CudaGraphCall", MetadataGraph)
    return module


class Model:
    def __init__(self):
        self.calls, self.fail = [], False

    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        self.calls.append((hidden_states, encoder_hidden_states, kwargs))
        if self.fail:
            raise RuntimeError("intentional capture failure")
        return hidden_states


def inputs(length=153):
    return (TensorMetadata((1, 41, 1536)), TensorMetadata((1, length, 2048)))


def test_two_shapes_stay_resident_and_new_objects_do_not_recapture(graph_module):
    model = Model()
    graphs = graph_module.DitGraphs(model)
    graphs.set("bf16")
    for length in (153, 156, 153, 156):
        model.forward(*inputs(length))
    assert sum(graph.captures for graph in graphs.graphs.values()) == 2
    assert len(graphs.graphs) == 2
    assert graphs.evidence()["capture_count"] == 2
    assert graphs.evidence()["replay_count"] == 4


@pytest.mark.parametrize(
    "field", ["dtype", "device", "strides", "mask_shape", "presence"]
)
def test_complete_signature_includes_layout_type_device_and_optional_masks(
    graph_module, field
):
    model = Model()
    graphs = graph_module.DitGraphs(model)
    graphs.set("bf16")
    x, context = inputs()
    mask = TensorMetadata((1, 153), dtype="bool")
    model.forward(x, context, image_mask=mask)
    x2, context2 = inputs()
    if field == "dtype":
        context2.dtype = "float32"
    elif field == "device":
        context2.device = "cuda:1"
    elif field == "strides":
        context2.strides = (400000, 2048, 1)
    elif field == "mask_shape":
        mask = TensorMetadata((1, 156), dtype="bool")
    else:
        mask = None
    model.forward(x2, context2, image_mask=mask)
    assert len(graphs.graphs) == 2


def test_bounded_lru_and_counters_survive_eviction(graph_module):
    model = Model()
    graphs = graph_module.DitGraphs(model, max_entries=2)
    graphs.set("bf16")
    for length in (153, 156, 153, 160, 156):
        model.forward(*inputs(length))
    assert len(graphs.graphs) == 2
    assert graphs.evidence()["capture_count"] == 4
    assert graphs.evidence()["eviction_count"] == 2
    assert graphs.evidence()["replay_count"] == 5


def test_failed_new_signature_keeps_old_bucket_and_restores_eager(graph_module):
    model = Model()
    original = model.forward
    graphs = graph_module.DitGraphs(model, max_entries=1)
    graphs.set("bf16")
    model.forward(*inputs())
    model.fail = True
    with pytest.raises(RuntimeError, match="intentional"):
        model.forward(*inputs(156))
    assert model.forward == original
    assert len(graphs.graphs) == 1
    assert graphs.evidence()["failure_count"] == 1
    model.fail = False
    graphs.set("bf16")
    before = graphs.evidence()["capture_count"]
    model.forward(*inputs())
    assert graphs.evidence()["capture_count"] == before
    graphs.close()
    assert model.forward == original and not graphs.graphs


def test_history_api_is_eager_and_unmodified(graph_module):
    model = Model()
    graphs = graph_module.DitGraphs(model)
    graphs.set("bf16")
    x, context = inputs()
    mask = TensorMetadata((1, 153), dtype="bool")
    assert (
        model.forward(x, context, image_mask=mask, return_all_hidden_states=True) is x
    )
    assert not graphs.graphs
    assert model.calls[-1][2]["image_mask"] is mask
    assert model.calls[-1][2]["return_all_hidden_states"]
