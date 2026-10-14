"""Opt-in actual CUDA backbone/Flow4 equality and fresh input ownership gates.

PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_prepared_inference_gpu.py
Uses the pinned tiny Qwen3/GR00T model entirely on CUDA, not a quality benchmark.
"""

import os
from types import SimpleNamespace

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip(
        "Set PIR2_GPU_TESTS=1 for prepared CUDA checks", allow_module_level=True
    )

import torch

from coexecution.libero.inference import PreparedInferenceEngine, prepare_observations
from coexecution.optimization.config import OptimizationConfig
from coexecution.optimization.session import GrootOptimizations
from tests.gpu.test_policy_vlm_gpu import exact, inputs, snapshot
from tests.gpu.test_policy_vlm_gpu import (
    policy as policy,  # noqa: PLC0414 (fixture export)
)


@pytest.mark.parametrize("runtime", ["plain", "static", "graph"])
@torch.inference_mode()
def test_full_engine_matches_raw_output_rng_and_keeps_fresh_owned_inputs(
    policy, runtime
):
    config = OptimizationConfig(
        input_preparation=runtime != "plain",
        static_vlm=runtime != "plain",
        vlm_graph=runtime == "graph",
    )
    with GrootOptimizations(policy, config) as scope:
        prepared = _check_owned_inputs(policy)
        if runtime != "plain":
            assert scope.static_vlm.evidence()["resident_bindings"] == 2
            assert scope.evidence()["input_preparation"]["fallbacks"] == {}
            prepared[0].backbone["input_ids"].add_(1)
            with pytest.raises(ValueError, match="owned metadata has changed"):
                PreparedInferenceEngine(policy.model).full(prepared[0])


def _check_owned_inputs(policy):
    batches = [inputs(state=0.1), inputs(state=0.8), inputs(token=25, state=-0.2)]
    raw_pixels = [batch["pixel_values"].clone() for batch in batches]
    adapter = SimpleNamespace(
        model=policy.model,
        _collate_observation=lambda batch: ({"inputs": dict(batch)}, {"frame": 1}),
    )
    prepared = prepare_observations(adapter, batches, ["frame-0", "frame-1", "frame-2"])
    for batch in batches:
        batch["pixel_values"].zero_()
        batch["state"].zero_()
    engine = PreparedInferenceEngine(policy.model)
    retained = []
    for index, item in enumerate(prepared):
        assert torch.equal(item.backbone["pixel_values"], raw_pixels[index])
        assert item.backbone["pixel_values"].is_cuda
        raw = {
            **batches[index],
            "pixel_values": raw_pixels[index],
            "state": item.action["state"],
        }
        torch.manual_seed(151 + index)
        reference = snapshot(policy.model.get_action(raw, options={"image_delay": 0}))
        reference_rng = torch.cuda.get_rng_state().clone()
        reference_cpu_rng = torch.get_rng_state().clone()
        torch.manual_seed(151 + index)
        result = engine.full(item, {"image_delay": 0})
        exact(reference, result)
        assert torch.equal(reference_rng, torch.cuda.get_rng_state())
        assert torch.equal(reference_cpu_rng, torch.get_rng_state())
        retained.append((result, snapshot(result)))
    assert not torch.equal(prepared[0].action["state"], prepared[1].action["state"])
    assert not torch.equal(retained[0][1]["action_pred"], retained[1][1]["action_pred"])
    for original, saved in retained:
        exact(saved, original)
    return prepared
