"""CUDA regression: the in-flight action prefix stays exactly clean under jitter."""

from types import SimpleNamespace

import torch
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

assert torch.cuda.is_available(), "CUDA is required"
config = SimpleNamespace(
    action_horizon=40,
    streaming_constant_weight=0.0,
    streaming_linear_weight=0.0,
    streaming_random_weight=0.0,
    streaming_chunk_wise_weight=1.0,
    streaming_rtc_weight=0.0,
    noise_s=0.999,
    num_timestep_buckets=1000,
    streaming_schedule_mode="pir2",
    streaming_chunk_size_max=5,
)
head = SimpleNamespace(config=config)
for seed in range(8):
    torch.manual_seed(seed)
    tau = Gr00tN1d7ActionHead.sample_time_per_position(
        head, 128, device="cuda", dtype=torch.float32
    )
    assert torch.all(tau[:, 0] == config.noise_s), (
        "Jitter corrupted the committed clean prefix"
    )
    d = head._last_clean_prefix_length
    assert 1 <= d <= 5
    assert torch.all(tau[:, :d] == config.noise_s)
    assert torch.all((tau >= 0) & (tau <= config.noise_s))
    assert torch.all(tau[:, 1:] <= tau[:, :-1])
print("CUDA clean-prefix schedule regression passed")
