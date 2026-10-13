"""Explicit Thor Swap-AB contracts; no learned-policy quality claim."""

import os

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip("Set PIR2_GPU_TESTS=1 for explicit CUDA tests", allow_module_level=True)

import torch

from coexecution.optimization.backends.thor.linear import FloatingLinear
from coexecution.optimization.groot.floating import FloatingProjections
from tests.helpers.projection_models import Qwen3VLTextAttention


@pytest.mark.parametrize("k", [256, 1536])
@torch.inference_mode()
def test_swap_ab_bias_dynamic_graph_and_restoration(k):
    torch.manual_seed(1000)
    layer = torch.nn.Linear(k, 1536, device="cuda", dtype=torch.bfloat16).eval()
    layer.bias.copy_(torch.linspace(-3, 3, 1536, device="cuda"))
    layer.requires_grad_(False)
    model = torch.nn.Sequential(layer)
    x = torch.randn(1, 41, k, device="cuda", dtype=torch.bfloat16)
    original = model(x).clone()
    original_x = x.clone()
    baseline = FloatingLinear(layer, "baseline", "fp4")
    controller = FloatingProjections(
        model, precision="fp4", expanded=True, fp4_swap_ab=True
    )
    try:
        assert controller.coverage["fp4_swap_ab"]["enabled"]
        for _ in range(3):
            model(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = model(x)
        for factor in (0.0, 0.125, -3.0, 1.0):
            x.copy_(original_x * factor)
            expected = baseline(x)
            actual = model(x)
            graph.replay()
            torch.testing.assert_close(captured, actual, rtol=0, atol=0)
            # CUTLASS epilogue ordering can differ at BF16 rounding boundaries.
            torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.001)
            if factor == 0:
                torch.testing.assert_close(
                    actual, layer.bias.expand_as(actual), rtol=0, atol=0
                )
        del graph, captured
        with pytest.raises(RuntimeError, match="body failure"):
            try:
                raise RuntimeError("body failure")
            finally:
                controller.close()
    finally:
        controller.close()
    assert "forward" not in layer.__dict__
    torch.testing.assert_close(model(original_x), original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "m,k,n,bias",
    [
        (40, 256, 1536, True),
        (41, 256, 128, True),
        (41, 256, 1536, False),
        (1, 256, 1536, True),
    ],
)
@torch.inference_mode()
def test_unsupported_signatures_use_original_operator(m, k, n, bias):
    layer = torch.nn.Linear(k, n, bias=bias, device="cuda", dtype=torch.bfloat16).eval()
    baseline = FloatingLinear(layer, "baseline", "fp4")
    candidate = FloatingLinear(layer, "candidate", "fp4", fp4_swap_ab=True)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(candidate(x), baseline(x), rtol=0, atol=0)


@torch.inference_mode()
def test_swap_ab_rejects_bad_shape_scale_and_bias():
    layer = torch.nn.Linear(256, 1536, device="cuda", dtype=torch.bfloat16).eval()
    q = FloatingLinear(layer, "test", "fp4", fp4_swap_ab=True)
    a, z = q.pack_input(torch.randn(41, 256, device="cuda", dtype=torch.bfloat16))
    run = q.ops.linear_forward_packed_gemm_bf16_swap_ab
    with pytest.raises(RuntimeError, match="supports only"):
        run(a[:40], z, q.packed_weight, q.weight_scale, q.bias)
    with pytest.raises(RuntimeError, match="scale buffer too small"):
        run(a, z.flatten()[:1], q.packed_weight, q.weight_scale, q.bias)
    with pytest.raises(RuntimeError, match="output-channel bias"):
        run(a, z, q.packed_weight, q.weight_scale, q.bias.float())
    with pytest.raises(RuntimeError, match="contiguous CUDA uint8"):
        run(a, z.float(), q.packed_weight, q.weight_scale, q.bias)


@torch.inference_mode()
def test_grouped_projection_propagates_swap_ab_and_restores():
    owner = Qwen3VLTextAttention()
    for name in ("q_proj", "k_proj", "v_proj"):
        setattr(owner, name, torch.nn.Linear(256, 512))
    model = torch.nn.Sequential(owner).cuda().bfloat16().eval()
    model.requires_grad_(False)
    x = torch.randn(1, 41, 256, device="cuda", dtype=torch.bfloat16)
    original = model(x)
    baseline = FloatingProjections(model, precision="fp4", grouped=True)
    try:
        expected = model(x)
    finally:
        baseline.close()
    candidate = FloatingProjections(
        model, precision="fp4", grouped=True, fp4_swap_ab=True
    )
    try:
        assert candidate.groups[0].linear.fp4_swap_ab
        for actual, reference in zip(model(x), expected):
            torch.testing.assert_close(actual, reference, rtol=0.008, atol=0.001)
    finally:
        candidate.close()
    assert all("forward" not in m.__dict__ for m in model.modules())
    for actual, reference in zip(model(x), original):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
