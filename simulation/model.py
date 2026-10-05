"""Conditional U-Net flow policies with shared/per-position FiLM conditioning.

Architecture adapted from Stanford Diffusion Policy's ConditionalUnet1D (MIT),
see licenses/DIFFUSION_POLICY_LICENSE and THIRD_PARTY_NOTICES.md.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        frequencies = torch.exp(
            torch.arange(dim // 2) * (-math.log(10000) / (dim // 2 - 1))
        )
        self.register_buffer("frequencies", frequencies)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim)
        )

    def forward(self, times):
        phase = times[..., None] * 1000 * self.frequencies
        return self.mlp(torch.cat([phase.sin(), phase.cos()], dim=-1))


def conv_block(cin, cout, kernel=5):
    return nn.Sequential(
        nn.Conv1d(cin, cout, kernel, padding=kernel // 2),
        nn.GroupNorm(8, cout),
        nn.Mish(),
    )


class Residual(nn.Module):
    def __init__(self, cin, cout, cond=338):
        super().__init__()
        self.first = conv_block(cin, cout)
        self.second = conv_block(cout, cout)
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond, cout * 2))
        self.residual = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, conditioning):
        features = self.first(x)
        film = self.film(conditioning).transpose(1, 2)
        film = F.interpolate(film, size=x.shape[-1], mode="linear", align_corners=False)
        scale, bias = film.chunk(2, dim=1)
        return self.second(scale * features + bias) + self.residual(x)


class FlowUNet(nn.Module):
    def __init__(self, dims=(256, 512, 1024), action_dim=16, observation_dim=41):
        super().__init__()
        self.dims = tuple(dims)
        self.time = TimeEmbedding()
        cond_dim = 256 + observation_dim * 2
        pairs = list(zip([action_dim, *dims[:-1]], dims))
        self.down = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Residual(cin, cout, cond_dim),
                        Residual(cout, cout, cond_dim),
                        nn.Conv1d(cout, cout, 3, stride=2, padding=1)
                        if i < len(pairs) - 1
                        else nn.Identity(),
                    ]
                )
                for i, (cin, cout) in enumerate(pairs)
            ]
        )
        self.middle = nn.ModuleList(
            [Residual(dims[-1], dims[-1], cond_dim) for _ in range(2)]
        )
        self.up = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Residual(cout * 2, cin, cond_dim),
                        Residual(cin, cin, cond_dim),
                        nn.ConvTranspose1d(cin, cin, 4, stride=2, padding=1),
                    ]
                )
                for cin, cout in reversed(pairs[1:])
            ]
        )
        self.final = nn.Sequential(
            conv_block(dims[0], dims[0]), nn.Conv1d(dims[0], action_dim, 1)
        )

    def forward(self, actions, times, observations):
        batch, horizon, _ = actions.shape
        if times.ndim == 1:
            times = times[:, None].expand(batch, horizon)
        time_embedding = self.time(times)
        state_embedding = observations.flatten(1)[:, None].expand(-1, horizon, -1)
        condition = torch.cat([time_embedding, state_embedding], dim=-1)
        x = actions.transpose(1, 2)
        skips = []
        for first, second, down in self.down:
            x = second(first(x, condition), condition)
            skips.append(x)
            x = down(x)
        for block in self.middle:
            x = block(x, condition)
        for first, second, up in self.up:
            x = torch.cat([x, skips.pop()], dim=1)
            x = up(second(first(x, condition), condition))
        return self.final(x).transpose(1, 2)


def clamp_action(normalized, offset, scale):
    """Map the executed physical bounds back into normalized inpaint coordinates."""
    return ((normalized * scale + offset).clamp(-1, 1) - offset) / scale


def schedule(horizon, delay, device):
    if not 1 <= delay < horizon / 2:
        raise ValueError("delay must be positive and less than horizon / 2")
    positions = torch.arange(horizon, device=device, dtype=torch.float32)
    ramp = 1 - (positions - delay + 0.5) / (horizon - 2 * delay)
    return torch.where(
        positions < delay, 1.0, torch.where(positions >= horizon - delay, 0.0, ramp)
    )


@torch.no_grad()
def stream_step(model, buffer, times, observations, delay):
    batch, horizon, action_dim = buffer.shape
    target = schedule(horizon, delay, buffer.device)
    target = torch.cat([torch.ones(delay, device=buffer.device), target[:-delay]])
    advance = (target[None] - times).clamp(min=0)
    prediction = buffer + advance[..., None] * model(buffer, times, observations)
    shifted = torch.cat(
        [
            prediction[:, delay:],
            torch.randn(batch, delay, action_dim, device=buffer.device),
        ],
        dim=1,
    )
    shifted_times = torch.cat(
        [(times + advance)[:, delay:], torch.zeros(batch, delay, device=buffer.device)],
        dim=1,
    )
    return prediction, shifted, shifted_times


@torch.no_grad()
def flow_sample(model, observations, steps=15, prefix=None, horizon=16):
    actions = torch.randn(
        observations.shape[0], horizon, 16, device=observations.device
    )
    delay = 0 if prefix is None else prefix.shape[1]
    for i in range(steps):
        times = torch.full(actions.shape[:2], i / steps, device=actions.device)
        if delay:
            actions[:, :delay] = prefix
            times[:, :delay] = 1.0
        actions = actions + model(actions, times, observations) / steps
        if delay:
            actions[:, :delay] = prefix
    return actions
