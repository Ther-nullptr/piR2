"""Real action-head CUDA contracts, without a checkpoint or learned-quality claim.

Prepare the pinned, patched GR00T source and robotics-kernels dependencies, then:
PIR2_GPU_TESTS=1 python -m pytest -q coexecution/test_groot_streaming_gpu.py

The small AlternateVLDiT runs the upstream bootstrap and rolling-buffer code.
Synthetic features bypass the VLM and do not measure LIBERO task success.
"""

import os
from contextlib import closing, nullcontext
from types import SimpleNamespace

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip(
        "Set PIR2_GPU_TESTS=1 for streaming CUDA checks", allow_module_level=True
    )

import torch
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from transformers.feature_extraction_utils import BatchFeature

from coexecution.groot_optimization import GrootOptimizations, OptimizationConfig
from coexecution.quantization_connections import DitGraphs


@pytest.fixture
def policy_and_inputs():
    torch.manual_seed(107)
    config = Gr00tN1d7Config(
        backbone_embedding_dim=128,
        hidden_size=128,
        input_embedding_dim=128,
        max_state_dim=8,
        max_action_dim=8,
        action_horizon=40,
        max_num_embodiments=1,
        max_seq_len=64,
        num_inference_timesteps=4,
        streaming=True,
        streaming_schedule_mode="pir2",
        streaming_chunk_size_max=5,
        image_delay_max=5,
        image_delay_embed_dim=64,
        state_dropout_prob=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 4,
            "num_attention_heads": 4,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 128,
            "interleave_self_attention": True,
        },
    )
    with torch.device("cuda"):
        model = torch.nn.Module()
        model.config = config
        model.action_head = Gr00tN1d7ActionHead(config)
    model = model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    # Zero-initialized delay weights would hide a stale image-delay input.
    model.action_head.delay_embedding.weight.normal_(std=0.05)
    inputs = []
    for index in range(4):
        image = (torch.arange(13, device="cuda") + index) % 2 == 0
        valid = torch.ones(1, 13, device="cuda", dtype=torch.bool)
        valid[:, -1 - index] = False
        backbone = BatchFeature(
            data={
                "backbone_features": torch.randn(
                    1, 13, 128, device="cuda", dtype=torch.bfloat16
                ),
                "backbone_attention_mask": valid,
                "image_mask": image.unsqueeze(0),
            }
        )
        action = BatchFeature(
            data={
                "state": torch.randn(1, 1, 8, device="cuda", dtype=torch.bfloat16),
                "embodiment_id": torch.zeros(1, device="cuda", dtype=torch.long),
            }
        )
        inputs.append((backbone, action))
    return SimpleNamespace(model=model), inputs


@torch.inference_mode()
def rollout(head, inputs):
    """Reset, bootstrap, then change slide size, delay, state and cached features."""
    head.reset_streaming_buffer()
    assert head._stream_buf is None and head._stream_buf_t is None
    torch.cuda.manual_seed(571)
    times, retained_outputs = [], []

    def record_time(module, args, kwargs):
        times.append(kwargs["timestep"].clone())

    def record_output(module, args, output):
        retained_outputs.append((output, output.clone()))

    handle = head.model.register_forward_pre_hook(record_time, with_kwargs=True)
    output_handle = head.model.register_forward_hook(record_output)
    snapshots = []

    def snapshot(output):
        snapshots.append(
            (
                output.clone(),
                head._stream_buf.clone(),
                head._stream_buf_t.clone(),
            )
        )

    try:
        clean = head.get_action(
            *inputs[0],
            options={"force_nonstreaming": True, "image_delay": 0},
        )["action_pred"]
        head.seed_streaming_buffer(clean, slide_steps=1)
        snapshot(clean)
        for index, slide in enumerate((1, 3, 5), start=1):
            output = head.get_action(
                *inputs[index],
                options={
                    "slide_steps": slide,
                    "num_inference_steps_per_call": 1,
                    "image_delay": (0, 2, 5)[index - 1],
                },
            )["action_pred"]
            torch.testing.assert_close(
                head._stream_buf[:, :-slide], output[:, slide:], rtol=0, atol=0
            )
            assert torch.count_nonzero(head._stream_buf_t[:, -slide:]) == 0
            snapshot(output)
    finally:
        handle.remove()
        output_handle.remove()
    assert len(times) == 7  # Four bootstrap evaluations plus three streaming calls.
    for time in times:
        assert time.shape == (1, 41)
        assert torch.count_nonzero(time[:, :1]) == 0
    for time in times[:4]:
        assert torch.equal(time[:, 1:], time[:, 1:2].expand(-1, 40))
    for time in times[4:]:
        assert torch.unique(time[:, 1:]).numel() > 1
    for output, saved_output in retained_outputs:
        torch.testing.assert_close(output, saved_output, rtol=0, atol=0)
    assert torch.isfinite(torch.stack([state[0] for state in snapshots])).all()
    return snapshots, torch.cuda.get_rng_state()


def assert_equal(first, second):
    assert len(first[0]) == len(second[0])
    for left, right in zip(first[0], second[0]):
        for actual, expected in zip(left, right):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(first[1], second[1])


def assert_integer_trace(trace, bits):
    events = trace.events()
    assert any(event.name == "robotics_integer::gemm_biasless" for event in events)
    names = [
        event.name.replace(" ", "")
        for event in events
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    element = "signedchar" if bits == 8 else "integer_subbyte<4,true>"
    instruction = f"GemmShape<16,8,{32 if bits == 8 else 64}>"
    assert any(element in name and instruction in name for name in names), names


@pytest.mark.parametrize("precision", ["bf16", "w8a8", "w4a4"])
@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "dit_graph"])
@torch.inference_mode()
def test_real_streaming_head_fusion_state_reset_and_integer_kernels(
    policy_and_inputs, precision, use_graph, record_property
):
    policy, inputs = policy_and_inputs
    head = policy.model.action_head
    original = rollout(head, inputs)
    config = {"precision": precision, "scope": "all", "category_id": 0}
    with GrootOptimizations(policy, OptimizationConfig(**config)):
        reference = rollout(head, inputs)
    with GrootOptimizations(policy, OptimizationConfig(**config, fusion=True)):
        fused_forward = head.model.forward
        # Observe capture/replay counters while checking the graph lifecycle.
        graph_scope = closing(DitGraphs(head.model)) if use_graph else nullcontext()
        with graph_scope as graphs:
            if use_graph:
                graphs.set(precision)
            # Warm the real kernels/capture before recording their execution.
            actual = rollout(head, inputs)
            assert_equal(actual, reference)
            baseline_actions = torch.stack([state[0] for state in original[0]]).float()
            actual_actions = torch.stack([state[0] for state in actual[0]]).float()
            error = (actual_actions - baseline_actions).abs()
            record_property("precision", precision)
            record_property("dit_graph", use_graph)
            record_property("bf16_action_max_abs_error", error.max().item())
            record_property("bf16_action_mean_abs_error", error.mean().item())
            record_property("same_precision_fusion_max_abs_error", 0.0)
            if use_graph:
                assert sum(g.captures for g in graphs.graphs.values()) == 1
                assert sum(g.replays for g in graphs.graphs.values()) == 7
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as trace:
                replay = rollout(head, inputs)
            assert_equal(replay, reference)
            if precision != "bf16":
                # The eager action encoder/decoder still dispatch real integer GEMMs.
                assert_integer_trace(trace, 8 if precision == "w8a8" else 4)
            if use_graph:
                assert sum(g.captures for g in graphs.graphs.values()) == 1
                assert sum(g.replays for g in graphs.graphs.values()) == 14

            changed = list(inputs)
            backbone, action = inputs[2]
            changed[2] = (
                BatchFeature(
                    data={**backbone, "backbone_features": -backbone.backbone_features}
                ),
                action,
            )
            fresh = rollout(head, changed)
            # Same bootstrap and first call; the second call must consume fresh VL data.
            for index in (0, 1):
                torch.testing.assert_close(
                    fresh[0][index][0], actual[0][index][0], rtol=0, atol=0
                )
            assert not torch.equal(fresh[0][2][0], actual[0][2][0])
            assert_equal(rollout(head, inputs), reference)

            if use_graph:
                # S=13 -> 17 -> 17 -> 13; both mandatory masks retain real values.
                # AlternateVLDiT requires image/valid masks; its optional encoder
                # mask is ignored upstream, so do not invent unsupported mask modes.
                varied = list(inputs)
                for index in (1, 2):
                    backbone, action = inputs[index]
                    varied[index] = (
                        BatchFeature(
                            data={
                                key: torch.cat([value, value[:, :4]], dim=1)
                                for key, value in backbone.items()
                            }
                        ),
                        action,
                    )
                graphs.set(None)
                varied_reference = rollout(head, varied)
                graphs.set(precision)
                before = sum(g.captures for g in graphs.graphs.values())
                assert_equal(rollout(head, varied), varied_reference)
                assert sum(g.captures for g in graphs.graphs.values()) == before + 2
                assert len(graphs.graphs) == 1
                record_property("graph_capture_count", before + 2)
        assert head.model.forward is fused_forward
        if use_graph:
            assert not graphs.graphs
        assert_equal(rollout(head, inputs), reference)
        if use_graph:
            with (
                pytest.raises(RuntimeError, match="deliberate graph body failure"),
                closing(DitGraphs(head.model)) as failed_graphs,
            ):
                failed_graphs.set(precision)
                assert_equal(rollout(head, inputs), reference)
                raise RuntimeError("deliberate graph body failure")
            assert head.model.forward is fused_forward and not failed_graphs.graphs
            assert_equal(rollout(head, inputs), reference)
    # Closing restores upstream execution and does not leave a precision cache behind.
    assert_equal(rollout(head, inputs), original)
    with GrootOptimizations(
        policy, OptimizationConfig(**config, fusion=True, dit_graph=use_graph)
    ):
        assert_equal(rollout(head, inputs), reference)
    assert_equal(rollout(head, inputs), original)
