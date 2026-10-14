"""VLM configuration and import boundaries; never execute a tensor/model."""

import argparse
import subprocess
import sys

import pytest

from coexecution.optimization.config import (
    OptimizationConfig,
    add_optimization_arguments,
    optimization_config,
)


def test_vlm_graph_is_opt_in_and_keeps_bf16_default():
    parser = argparse.ArgumentParser()
    add_optimization_arguments(parser)
    assert not optimization_config(parser.parse_args([])).vlm_graph
    config = optimization_config(parser.parse_args(["--vlm-cuda-graph"]))
    assert config.enabled and config.precision == "bf16" and config.vlm_graph
    assert not config.fusion and not config.dit_graph
    config.validate_variant("flow")


@pytest.mark.parametrize("precision", ["bf16", "w8a8", "w4a4"])
def test_graph_preserves_selected_precision_and_rejects_streaming(precision):
    config = OptimizationConfig(precision=precision, vlm_graph=True)
    assert config.precision == precision
    with pytest.raises(ValueError, match="standard Flow"):
        config.validate_variant("pir2")


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
def test_unvalidated_floating_graph_rejected(precision):
    with pytest.raises(ValueError, match="VLM Graph supports"):
        OptimizationConfig(precision=precision, vlm_graph=True)


def test_dit_only_configuration_cannot_install_vlm_graph():
    with pytest.raises(ValueError, match="VLM component"):
        OptimizationConfig(component="dit", vlm_graph=True)


def test_enabling_configuration_does_not_import_gpu_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from coexecution.optimization.config import OptimizationConfig
from coexecution.optimization.session import GrootOptimizations
config = OptimizationConfig(vlm_graph=True)
scope = GrootOptimizations(object(), config)
assert scope.evidence()['vlm_graph'] is None
assert not any(n.startswith(('torch', 'triton', 'transformers')) for n in sys.modules)
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
