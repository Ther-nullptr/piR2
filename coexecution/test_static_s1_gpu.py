"""GPU regression tests for static S1 state handling and CUDA graph replay."""

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from coexecution.graph_dit import GraphCall
from coexecution.static_s1 import StaticS1
from coexecution.workload import PiR2Workload


@pytest.fixture(scope="module")
def workload():
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    return PiR2Workload("outputs/pir2-so100-smoke/checkpoint-10")


@torch.inference_mode()
def rollout(workload, count=12, slide=1):
    workload.reset()
    outputs = []
    for index in range(count):
        inputs = BatchFeature(
            workload.action_inputs[index % len(workload.action_inputs)]
        )
        output = workload.model.action_head.get_action(
            BatchFeature(workload.initial_features),
            inputs,
            options={
                "slide_steps": slide,
                "num_inference_steps_per_call": 1,
                "image_delay": index % 3,
            },
        )["action_pred"]
        outputs.append(output.clone())
    torch.cuda.synchronize()
    return (
        outputs,
        workload.model.action_head._stream_buf.clone(),
        workload.model.action_head._stream_buf_t.clone(),
        torch.cuda.get_rng_state(),
    )


def assert_exact(first, second):
    for left, right in zip(first[0], second[0]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(first[1:], second[1:]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("slide", [1, 2, 3])
def test_static_s1_preserves_actions_state_and_rng(workload, slide):
    reference = rollout(workload, slide=slide)
    original = workload.model.action_head._streaming_inference
    with StaticS1(workload) as adapter:
        actual = rollout(workload, slide=slide)
        assert_exact(reference, actual)
        assert adapter.evidence()["streaming_calls"] >= 12
    assert workload.model.action_head._streaming_inference == original


def test_graph_s1_preserves_reset_and_continuous_actions(workload):
    reference = rollout(workload, 20)
    with StaticS1(workload, graph=True) as adapter:
        actual = rollout(workload, 20)
        assert_exact(reference, actual)
        second = rollout(workload, 20)
        assert_exact(reference, second)
        assert adapter.evidence()["graph"]["replays"] >= 40


@torch.inference_mode()
def test_graph_uses_new_values_and_current_stream():
    assert torch.cuda.device_count() == 1
    weight = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)

    def operation(x):
        return (x @ weight).relu()

    graph = GraphCall(operation)
    first = torch.randn(17, 64, device="cuda", dtype=torch.bfloat16)
    expected = operation(first)
    output = graph(first)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        second = first + 1
        actual = graph(second)
        reference = operation(second)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert graph.evidence()["captures"] == 1
    graph.freeze()
    changed_shape = first[:5].clone()
    torch.testing.assert_close(
        graph(changed_shape), operation(changed_shape), rtol=0, atol=0
    )
    assert graph.evidence()["captures"] == 1
    assert graph.evidence()["fallbacks"] == 1


def test_failed_preparation_restores_model(workload, monkeypatch):
    head = workload.model.action_head
    before = (
        head._streaming_inference,
        head.model.forward,
        head.model.transformer_blocks[0].norm1.forward,
    )
    adapter = StaticS1(workload)

    def fail(*args):
        raise RuntimeError("injected preparation failure")

    monkeypatch.setattr(adapter, "prepare_constants", fail)
    try:
        with pytest.raises(RuntimeError, match="injected preparation failure"), adapter:
            pass
        assert before == (
            head._streaming_inference,
            head.model.forward,
            head.model.transformer_blocks[0].norm1.forward,
        )
    finally:
        # Keep the shared fixture intact when exercising the pre-fix failure.
        if adapter.restore:
            adapter.__exit__()


def test_exit_cuda_error_still_restores_model(workload, monkeypatch):
    head = workload.model.action_head
    before = (
        head._streaming_inference,
        head.model.forward,
        head.model.transformer_blocks[0].norm1.forward,
    )
    adapter = StaticS1(workload)
    adapter.__enter__()
    sync = torch.cuda.synchronize

    def fail():
        raise RuntimeError("injected synchronization failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "synchronize", fail)
            with pytest.raises(RuntimeError, match="injected synchronization failure"):
                adapter.__exit__()
        assert before == (
            head._streaming_inference,
            head.model.forward,
            head.model.transformer_blocks[0].norm1.forward,
        )
    finally:
        sync()
        if adapter.restore:
            adapter.__exit__()


@torch.inference_mode()
def test_wrong_buffer_shape_preserves_eager_rejection(workload):
    workload.reset()
    head = workload.model.action_head
    features = workload.initial_features["backbone_features"]
    state_features = torch.zeros(
        1, 1, head.input_embedding_dim, device="cuda", dtype=features.dtype
    )
    head._stream_buf = head._stream_buf[:, :-1]
    with (
        StaticS1(workload),
        pytest.raises(NotImplementedError, match="seed_streaming_buffer"),
    ):
        head._streaming_inference(
            features,
            state_features,
            workload.action_inputs[0]["embodiment_id"],
            BatchFeature(workload.initial_features),
            {"slide_steps": 1},
        )
    workload.reset()
