"""Small CUDA projection fixtures, shared without importing test modules."""

from types import SimpleNamespace

import torch


def policy_for(blocks):
    return SimpleNamespace(
        model=SimpleNamespace(
            action_head=SimpleNamespace(
                model=SimpleNamespace(transformer_blocks=blocks)
            )
        )
    )


class Norm(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = torch.nn.LayerNorm(
            width, elementwise_affine=False, device="cuda", dtype=torch.bfloat16
        )
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(
            width, 2 * width, device="cuda", dtype=torch.bfloat16
        )

    def forward(self, x, temb):
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=-1)
        if scale.ndim == 2:
            scale, shift = scale[:, None], shift[:, None]
        return self.norm(x) * (1 + scale) + shift


class Qwen3VLTextAttention(torch.nn.Module):
    """Small inventory-compatible fixture with the real shared-input call order."""

    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(256, 128)
        self.k_proj = torch.nn.Linear(256, 64)
        self.v_proj = torch.nn.Linear(256, 64)
        self.o_proj = torch.nn.Linear(128, 256)

    def forward(self, x, *, fail=False, other=None):
        q = self.q_proj(x)
        if fail:
            raise RuntimeError("partial group")
        k = self.k_proj(x if other is None else other)
        return q, k, self.v_proj(x)
