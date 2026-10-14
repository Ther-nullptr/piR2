"""Opt-in real CUDA ownership and independent paired-policy capture gates.

These tiny pinned GR00T/Qwen3 policies execute only on CUDA. They validate
mechanisms, not official-checkpoint behavior or LIBERO deployment quality.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip("Set PIR2_GPU_TESTS=1 for resident CUDA gates", allow_module_level=True)

import torch
from transformers import BatchFeature

from coexecution.libero.backend import Backend
from coexecution.libero.resident import FeatureLeasePool
from coexecution.optimization.config import OptimizationConfig
from coexecution.optimization.execution.submission import CudaSubmissionCoordinator
from coexecution.optimization.session import GrootOptimizations
from tests.gpu.test_policy_vlm_gpu import (
    inputs,
    policy,  # noqa: F401 -- reuse real CUDA fixture
)


@torch.inference_mode()
def test_owned_snapshot_is_fresh_exact_and_survives_release_and_producer_mutation():
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    exchange = FeatureLeasePool("cuda:0", max_entries=1)
    session = exchange.open_session()
    retained = []
    for frame in range(3):
        with torch.cuda.stream(producer):
            values = {
                "features": (
                    torch.arange(2048, device="cuda", dtype=torch.int32) + frame
                )
                .to(torch.uint16)
                .view(torch.bfloat16),
                "mask": torch.tensor([[True, False]], device="cuda"),
                "scalar": torch.tensor(frame, device="cuda", dtype=torch.int64),
            }
            expected = {key: value.clone() for key, value in values.items()}
            handle, meta = exchange.publish_cuda(
                session,
                values,
                {"capture_s": float(frame), "source_tick": frame},
                torch_module=torch,
                stream=producer,
            )
            for value in values.values():
                value.zero_()
        with (
            torch.cuda.stream(consumer),
            exchange.borrow_cuda(
                handle, meta, device="cuda:0", stream=consumer
            ) as owned,
        ):
            assert all(owned[key].data_ptr() != values[key].data_ptr() for key in owned)
            copies = {key: value.clone() for key, value in owned.items()}
            retained.append((owned, expected, copies))
        exchange.release(session, [handle])
        consumer.synchronize()
        assert exchange.evidence()["live_leases"] == 0
    for owned, expected, copies in retained:
        for key in expected:
            torch.testing.assert_close(
                owned[key], expected[key], rtol=0, atol=0, equal_nan=True
            )
            torch.testing.assert_close(
                copies[key], expected[key], rtol=0, atol=0, equal_nan=True
            )
    exchange.close_session(session)


@torch.inference_mode()
def test_consumer_wait_and_allocator_lifetime_outlive_registry_release():
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    exchange = FeatureLeasePool("cuda:0", max_entries=1)
    session = exchange.open_session()
    with torch.cuda.stream(producer):
        value = torch.full((1024, 1024), 7, device="cuda", dtype=torch.float32)
        value.mul_(3)
        ready = torch.cuda.Event()
        ready.record(producer)
        handle = exchange.publish(
            session,
            {"features": value},
            {
                "features": {
                    "dtype": "float32",
                    "shape": [1024, 1024],
                    "nbytes": 4 * 1024**2,
                }
            },
            {"capture_s": 1.0},
            ready=ready,
        )
    with torch.cuda.stream(consumer):
        with exchange.borrow_cuda(
            handle, {"capture_s": 1.0}, device="cuda:0", stream=consumer
        ) as owned:
            result = owned["features"].clone()
        exchange.release(session, [handle])
        del owned, value
    with torch.cuda.stream(producer):
        churn = [torch.zeros((1024, 1024), device="cuda") for _ in range(8)]
    consumer.synchronize()
    assert torch.equal(result, torch.full_like(result, 21))
    assert exchange.evidence()["live_leases"] == 0
    del churn
    exchange.close_session(session)


def test_independent_policies_cold_capture_concurrency_fresh_handoff_and_s2_reset_rng(
    policy,  # noqa: F811 -- imported CUDA fixture
):
    first, second = policy, deepcopy(policy)
    assert all(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(first.model.parameters(), second.model.parameters())
    )
    with torch.inference_mode():
        initial = inputs()
        batches = [
            initial,
            inputs(pixels=initial["pixel_values"] * 0.5),
            inputs(pixels=initial["pixel_values"] * -0.25),
        ]
        features = [second.model.forward_vlm(batch) for batch in batches]
        _, action = first.model.prepare_input(initial)
        torch.manual_seed(901)
        expected = [
            first.model.forward_dit_action_only(feature, action)["action_pred"].clone()
            for feature in features[:2]
        ]
        expected_rng = torch.cuda.get_rng_state().clone()
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    coordinator = CudaSubmissionCoordinator(quiesce=torch.cuda.synchronize)
    exchange = FeatureLeasePool("cuda:0", max_entries=2)
    session = exchange.open_session()
    with torch.inference_mode(), torch.cuda.stream(streams[1]):
        handle, meta = exchange.publish_cuda(
            session,
            features[0],
            {"capture_s": 0.0, "source_tick": 0},
            torch_module=torch,
            stream=streams[1],
        )
    config = OptimizationConfig(dit_graph=True, static_vlm=True, vlm_graph=True)
    scopes = []
    for index, current in enumerate((first, second)):
        with coordinator.capture(), torch.cuda.stream(streams[index]):
            scopes.append(GrootOptimizations(current, config).__enter__())
    # Use the actual Backend.reset policy for S2's deterministic reset. Its own
    # cache is independent of the exchange and S1, as for the paired CPU control.
    second.reset_streaming_buffer = second.model.action_head.reset_streaming_buffer
    second._vl_cache_lock = threading.Lock()
    resetter = Backend.__new__(Backend)
    resetter.core, resetter.torch = second, torch
    resetter.role, resetter.paired = "vlm", True
    resetter.feature_exchange = None
    barrier = threading.Barrier(2)

    def action_call(current_handle, current_meta):
        with (
            coordinator.submission(),
            torch.cuda.stream(streams[0]),
            torch.inference_mode(),
        ):
            barrier.wait(timeout=10)
            with exchange.borrow_cuda(
                current_handle, current_meta, device="cuda:0", stream=streams[0]
            ) as owned:
                result = first.model.forward_dit_action_only(
                    BatchFeature(owned), action
                )["action_pred"]
            exchange.release(session, [current_handle])
            streams[0].synchronize()
            return result

    def vision_call(index):
        with (
            coordinator.submission(),
            torch.cuda.stream(streams[1]),
            torch.inference_mode(),
        ):
            barrier.wait(timeout=10)
            resetter.reset({"seed": 99999})
            fresh = second.model.forward_vlm(batches[index])
            return exchange.publish_cuda(
                session,
                fresh,
                {"capture_s": float(index), "source_tick": index},
                torch_module=torch,
                stream=streams[1],
            )

    try:
        torch.manual_seed(901)
        with ThreadPoolExecutor(2) as workers:
            for index in range(2):
                action_future = workers.submit(action_call, handle, meta)
                vision_future = workers.submit(vision_call, index + 1)
                result = action_future.result(timeout=60)
                handle, meta = vision_future.result(timeout=60)
                torch.testing.assert_close(result, expected[index], rtol=0, atol=0)
        assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
        assert scopes[0].graphs.capture_count == 1
        assert scopes[0].graphs.replay_count == 8
        assert scopes[1].static_vlm.counts["graph_captures"] == 1
        assert scopes[1].static_vlm.counts["graph_replays"] == 2
        with exchange.borrow(handle, meta) as owned:
            for key, value in features[2].items():
                torch.testing.assert_close(owned.features[key], value, rtol=0, atol=0)
        assert not torch.equal(
            features[0]["backbone_features"], features[1]["backbone_features"]
        )
    finally:
        with coordinator.capture():
            exchange.close_session(session)
            for index, scope in reversed(list(enumerate(scopes))):
                with torch.cuda.stream(streams[index]):
                    scope.close()
