"""Explicit Thor producer checks; action or task quality is a separate gate."""

import os

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip("Set PIR2_GPU_TESTS=1", allow_module_level=True)

import torch
import torch.nn.functional as F

from coexecution.kernels.common.fused import prepare_gelu
from coexecution.optimization.backends.thor.linear import FloatingLinear
from coexecution.optimization.backends.thor.packing import (
    prepare_gelu_packed,
    prepare_rmsnorm_packed,
)
from coexecution.optimization.backends.thor.smoothquant import (
    PROFILE_SCHEMA,
    UPSTREAM_REVISION,
    weight_digest,
)
from coexecution.optimization.groot.floating import FloatingProjections
from coexecution.optimization.groot.projections import inventory
from tests.gpu.test_floating_fusion_gpu import Qwen3VLTextMLP


@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("precision", ["fp8", "fp4"])
@torch.inference_mode()
def test_gelu_smoothing_preserves_native_rounding_and_graph(
    approximate, smooth, precision
):
    torch.manual_seed(71)
    x = torch.randn(1, 41, 1536, device="cuda", dtype=torch.bfloat16)
    linear = torch.nn.Linear(1536, 256, device="cuda", dtype=torch.bfloat16).eval()
    scale = torch.logspace(-1, 1, 1536, device="cuda") if smooth else None
    quantized = FloatingLinear(linear, "gelu", precision, smoothing_scale=scale)
    lut = prepare_gelu(x.device, x.dtype, approximate)

    def run():
        return quantized(prepare_gelu_packed(x, lut, precision, smoothing_scale=scale))

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for factor in (0, 0.01, 1, 8):
        x.normal_().mul_(factor)
        graph.replay()
        torch.testing.assert_close(
            captured, quantized(F.gelu(x, approximate=approximate)), rtol=0, atol=0
        )


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
@torch.inference_mode()
def test_rmsnorm_smoothing_replays_fresh_inputs(precision):
    torch.manual_seed(72)
    x = torch.randn(1, 153, 2048, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    scale = torch.logspace(-1, 1, 2048, device="cuda")
    layer = torch.nn.Linear(
        2048, 256, bias=False, device="cuda", dtype=torch.bfloat16
    ).eval()
    quantized = FloatingLinear(layer, "rms", precision, smoothing_scale=scale)

    def run():
        return quantized(
            prepare_rmsnorm_packed(
                x, weight, precision, eps=1e-6, smoothing_scale=scale
            )
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for factor in (0, 0.01, 1, 8):
        x.normal_().mul_(factor)
        graph.replay()
        norm = (
            x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        ).bfloat16() * weight
        torch.testing.assert_close(captured, quantized(norm), rtol=0.03, atol=0.03)
        torch.testing.assert_close(captured, run(), rtol=0, atol=0)


def profile_for(model, precision, path):
    sites = inventory(model)[0]
    records = {
        name: {
            "scale": torch.linspace(0.5, 2, module.in_features),
            "weight_sha256": weight_digest(module.weight),
        }
        for name, (module, _, _) in sites.items()
    }
    torch.save(
        {
            "schema": PROFILE_SCHEMA,
            "upstream_revision": UPSTREAM_REVISION,
            "precision": precision,
            "linears": records,
        },
        path,
    )
    return path


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
@torch.inference_mode()
def test_diffusers_gelu_dropout_and_restoration(tmp_path, precision):
    from diffusers.models.attention import BasicTransformerBlock

    torch.manual_seed(74)
    model = (
        BasicTransformerBlock(256, 4, 64, activation_fn="gelu").cuda().bfloat16().eval()
    )
    x = torch.randn(1, 41, 256, device="cuda", dtype=torch.bfloat16)
    original = model(x)
    profile = profile_for(model, precision, tmp_path / "sq.pt")
    controller = FloatingProjections(model, precision=precision, smoothquant=profile)
    try:
        reference = model(x)
    finally:
        controller.close()
    controller = FloatingProjections(
        model, precision=precision, smoothquant=profile, gelu=True
    )
    try:
        assert controller.coverage["gelu_producers"]
        torch.testing.assert_close(model(x), reference, rtol=0, atol=0)
        model.ff.net[1].train()
        with pytest.raises(ValueError, match="eval Dropout"):
            model(x)
        model.ff.net[1].eval()
    finally:
        controller.close()
    torch.testing.assert_close(model(x), original, rtol=0, atol=0)
    model.ff.net[1].train()
    with pytest.raises(ValueError, match="eval Dropout"):
        FloatingProjections(model, precision=precision, smoothquant=profile, gelu=True)
    assert "gelu" not in model.ff.net[0].__dict__
    assert all("forward" not in module.__dict__ for module in model.modules())


class Qwen3VLTextRMSNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(256))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        variance = x.float().square().mean(-1, keepdim=True)
        return (
            x.float() * torch.rsqrt(variance + self.variance_epsilon)
        ).bfloat16() * self.weight


class Qwen3VLTextDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm = Qwen3VLTextRMSNorm()
        self.mlp = Qwen3VLTextMLP()

    def forward(self, x):
        return self.mlp(self.post_attention_layernorm(x))


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
@pytest.mark.parametrize("partial", [False, True])
@torch.inference_mode()
def test_rmsnorm_requires_both_consumers_and_restores(tmp_path, precision, partial):
    model = Qwen3VLTextDecoderLayer().cuda().bfloat16().eval()
    x = torch.randn(1, 41, 256, device="cuda", dtype=torch.bfloat16)
    original = model(x)
    executed = {"mlp.gate_proj", "mlp.down_proj"} if partial else None
    profile = None if partial else profile_for(model, precision, tmp_path / "sq.pt")
    baseline = FloatingProjections(
        model, precision=precision, executed=executed, smoothquant=profile
    )
    try:
        reference = model(x)
    finally:
        baseline.close()
    controller = FloatingProjections(
        model, precision=precision, executed=executed, rmsnorm=True, smoothquant=profile
    )
    try:
        assert bool(controller.coverage["rmsnorm_producers"]) is not partial
        torch.testing.assert_close(model(x), reference, rtol=0.03, atol=0.03)
    finally:
        controller.close()
    torch.testing.assert_close(model(x), original, rtol=0, atol=0)
