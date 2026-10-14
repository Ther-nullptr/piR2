"""Explicit host-native compilation gates; no neural framework or GPU work.

PIR2_NATIVE_TESTS=1 python -m pytest -q tests/native/test_compiled_host_guards.py
"""

import os
from types import SimpleNamespace

import pytest

if os.environ.get("PIR2_NATIVE_TESTS") != "1":
    pytest.skip(
        "Set PIR2_NATIVE_TESTS=1 for native compilation", allow_module_level=True
    )

from coexecution.optimization.execution.compiled_guards import load_compiled_guards
from coexecution.optimization.execution.guards import HostGuardPlan, ModelTopologyGuard
from coexecution.optimization.groot.vlm import CompiledExecutionGuard
from tests.contracts.test_policy_vlm import ModelMetadata


@pytest.fixture(scope="module")
def backend():
    return load_compiled_guards()


@pytest.mark.parametrize(
    "mutation",
    ["weight", "config", "hook", "defaults", "closure", "owner", "registry", "forward"],
)
def test_native_and_python_guards_reject_the_same_mutations(backend, mutation):
    model = ModelMetadata()
    model._modules = {}
    settings = [0]

    def forward(values, amount=0):
        return values, amount, settings[0]

    model.forward = forward
    plain, native = (
        CompiledExecutionGuard(model),
        CompiledExecutionGuard(model, backend=backend),
    )
    assert plain.unchanged() and native.unchanged()
    if mutation == "weight":
        model.weight._version += 1
    elif mutation == "config":
        model.config._attn_implementation = "eager"
    elif mutation == "hook":
        model._forward_hooks[1] = lambda *args: None
    elif mutation == "defaults":
        forward.__defaults__ = (1,)
    elif mutation == "closure":
        settings[0] = 1
    elif mutation == "owner":
        model.__dict__ = dict(vars(model))
    elif mutation == "registry":
        model._modules["new"] = ModelMetadata()
    else:
        model.forward = lambda values: values
    assert not plain.unchanged() and not native.unchanged()


def test_topology_and_observer_exception_retain_python_semantics(backend):
    model = ModelMetadata()
    model._modules = {}
    model.modules = lambda: iter((model,))
    topology = ModelTopologyGuard(model, backend=backend)
    assert topology.unchanged() and not topology.training_changed()
    model.training = True
    assert topology.training_changed()
    model.training = False
    counter = [0]

    def observer(*args):
        counter[0] += 1

    observer.__pir2_execution_observer__ = True
    model._forward_hooks[1] = observer
    guard = HostGuardPlan(
        model,
        methods=("forward",),
        fields=(),
        global_bindings=(),
        observer_hooks=(observer,),
        backend=backend,
    )
    observer()
    assert guard.unchanged()
    observer.__pir2_execution_observer__ = False
    assert not guard.unchanged()
    model._modules["new"] = SimpleNamespace()
    assert not topology.unchanged()
