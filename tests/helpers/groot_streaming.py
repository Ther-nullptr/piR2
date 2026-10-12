"""CUDA-only streaming model fixture and state-preservation assertions."""

from types import SimpleNamespace

import torch
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from transformers.feature_extraction_utils import BatchFeature


def make_policy_and_inputs(*, streaming=True):
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
        streaming=streaming,
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
