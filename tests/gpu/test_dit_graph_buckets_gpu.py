"""Opt-in CUDA graph replay/layout/lifetime checks; no CPU neural model."""

import os

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip("Set PIR2_GPU_TESTS=1 for CUDA graph checks", allow_module_level=True)

import torch

from coexecution.optimization.execution.graph import DitGraphs


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep=None,
        encoder_attention_mask=None,
        return_all_hidden_states=False,
        image_mask=None,
        backbone_attention_mask=None,
    ):
        result = self.linear(hidden_states) + encoder_hidden_states[:, :1]
        if timestep is not None:
            result = result + timestep[:, None, None].to(result.dtype)
        if image_mask is not None:
            result = result + image_mask[:, :1, None].to(result.dtype)
        return (result, [result]) if return_all_hidden_states else result


def equal(actual, expected):
    assert torch.equal(
        actual.contiguous().view(torch.int16), expected.contiguous().view(torch.int16)
    )


@torch.inference_mode()
def test_alternating_cuda_shapes_replay_exactly_and_keep_outputs_alive():
    assert torch.cuda.device_count() == 1
    with torch.device("cuda"):
        model = Model().bfloat16().eval()
    eager = model.forward
    graphs = DitGraphs(model, max_entries=2)
    graphs.set("bf16")
    previous = []
    for length in (153, 156, 153, 156, 153, 156):
        x = torch.randn(1, 41, 32, device="cuda", dtype=torch.bfloat16)
        context = torch.randn(1, length, 32, device="cuda", dtype=torch.bfloat16)
        step = torch.randint(0, 3, (1,), device="cuda")
        mask = torch.ones(1, length, device="cuda", dtype=torch.bool)
        expected = eager(x, context, timestep=step, image_mask=mask)
        output = model(x, context, timestep=step, image_mask=mask)
        equal(output, expected)
        for retained, copy in previous:
            equal(retained, copy)
        previous.append((output, output.clone()))
    evidence = graphs.evidence()
    assert evidence["capture_count"] == 2 and evidence["replay_count"] == 6
    assert evidence["resident_signatures"] == 2
    graphs.close()
    assert model.forward == eager and not graphs.graphs


@torch.inference_mode()
def test_layout_eviction_and_eager_history_are_exact():
    with torch.device("cuda"):
        model = Model().bfloat16().eval()
    eager = model.forward
    graphs = DitGraphs(model, max_entries=1)
    graphs.set("bf16")
    for length in (153, 156, 153):
        x = torch.randn(1, 41, 32, device="cuda", dtype=torch.bfloat16)
        context = torch.randn(1, length * 2, 32, device="cuda", dtype=torch.bfloat16)[
            :, ::2
        ]
        equal(model(x, context), eager(x, context))
    assert graphs.evidence()["capture_count"] == 3
    assert graphs.evidence()["eviction_count"] == 2
    before = graphs.evidence()
    result, history = model(x, context, return_all_hidden_states=True)
    expected, _ = eager(x, context, return_all_hidden_states=True)
    equal(result, expected)
    assert history[0] is result and graphs.evidence() == before
    graphs.close()
