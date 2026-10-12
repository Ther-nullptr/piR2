"""Import compatibility and dependency boundaries without model execution."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_configuration_and_legacy_scope_import_without_gpu_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

from coexecution import groot_optimization as legacy
from coexecution.optimization import config, session, tactics
from coexecution.optimization.groot import action
from coexecution.optimization.execution import tensor_graph
from coexecution.optimization.execution import compiled_guards, images

assert callable(action.ActionGraph)
assert callable(tensor_graph.TensorGraphCall)

for source, names in (
    (config, ("OptimizationConfig", "add_optimization_arguments", "optimization_config")),
    (session, ("GrootOptimizations",)),
    (tactics, ("apply_tactics", "checked_tactic", "read_json", "read_shape_tactics")),
):
    for name in names:
        assert getattr(legacy, name) is getattr(source, name), name

with legacy.GrootOptimizations(object()) as scope:
    assert not scope.config.enabled
    assert scope.evidence()["coverage"] == {}

for name in sys.modules:
    assert not name.startswith(("torch", "triton", "robotics_kernels", "Cython", "setuptools", "numpy")), name
    assert not name.startswith("coexecution.optimization.backends"), name
""",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_shared_projection_and_graph_legacy_imports_preserve_identity():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from types import ModuleType

# Import-only sentinels: this contract must never execute a tensor or kernel.
sys.modules["torch"] = ModuleType("torch")
graph_dependency = ModuleType("robotics_kernels.common.graph")
graph_dependency.CudaGraphCall = object()
sys.modules["robotics_kernels.common.graph"] = graph_dependency

from coexecution.groot_graph import CudaGraphCall as legacy_call, DitGraphs as legacy_graph
from coexecution.groot_projections import inventory as legacy_inventory
from coexecution.optimization.execution.graph import DitGraphs
from coexecution.optimization.groot.projections import inventory

assert legacy_graph is DitGraphs
assert legacy_call is graph_dependency.CudaGraphCall
assert legacy_inventory is inventory
""",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
