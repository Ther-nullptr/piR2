"""Thor FP8/FP4 smoothing contracts; no claim of robot task success."""

import os

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip("Set PIR2_GPU_TESTS=1 for explicit CUDA tests", allow_module_level=True)

import torch
import torch.nn.functional as F

from coexecution.optimization.backends.thor.linear import FloatingLinear
from coexecution.optimization.backends.thor.packing import prepare_norm, prepare_swiglu
from coexecution.optimization.backends.thor.smoothquant import (
    PROFILE_SCHEMA,
    UPSTREAM_REVISION,
    ActivationCalibration,
    channel_scales,
    load_profile,
    weight_digest,
)
from coexecution.optimization.groot.floating import FloatingProjections
from coexecution.optimization.groot.projections import inventory
from tests.gpu.test_floating_fusion_gpu import Qwen3VLTextMLP, valid_scales


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
@torch.inference_mode()
def test_scaled_gemm_matches_explicit_balancing_and_graph(precision):
    torch.manual_seed(42)
    linear = torch.nn.Linear(256, 128, device="cuda", dtype=torch.bfloat16).eval()
    original = linear.weight.clone()
    x = torch.randn(2, 256, 41, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    s = torch.logspace(-1, 1, 256, device="cuda")
    transformed = torch.nn.Linear(256, 128, device="cuda", dtype=torch.bfloat16).eval()
    transformed.weight.copy_((linear.weight.float() * s).bfloat16())
    transformed.bias.copy_(linear.bias)
    quantized = FloatingLinear(linear, "test", precision, smoothing_scale=s)
    explicit = FloatingLinear(transformed, "reference", precision, fast_fp8=True)

    def check(actual):
        reference = explicit((x.float() / s).bfloat16())
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(linear.weight, original, rtol=0, atol=0)

    check(quantized(x))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = quantized(x)
    for factor in (0.0, 0.125, 7.0):
        x.normal_().mul_(factor)
        graph.replay()
        check(captured)


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
@torch.inference_mode()
def test_smoothing_reaches_fused_norm_and_swiglu(precision):
    torch.manual_seed(43)
    x = torch.randn(1, 41, 256, device="cuda", dtype=torch.bfloat16)
    scale = torch.logspace(-1, 1, 256, device="cuda")
    linear = torch.nn.Linear(256, 128, device="cuda", dtype=torch.bfloat16).eval()
    quant = FloatingLinear(linear, "test", precision, smoothing_scale=scale)
    lut = F.silu(
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    packed = prepare_swiglu(x, x * 2, lut, precision, smoothing_scale=scale)
    torch.testing.assert_close(
        quant(packed), quant(F.silu(x) * (x * 2)), rtol=0, atol=0
    )
    # Dropping a scale marker must fail before an incorrect compensated GEMM.
    raw = prepare_swiglu(x, x, lut, precision)
    with pytest.raises(ValueError, match="does not match"):
        quant(raw)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result, residual = prepare_norm(
            x, precision, eps=1e-5, residual=x, smoothing_scale=scale
        )
        captured = quant(result)
    x.normal_().mul_(3)
    graph.replay()
    eager, expected_residual = prepare_norm(
        x, precision, eps=1e-5, residual=x, smoothing_scale=scale
    )
    torch.testing.assert_close(captured, quant(eager), rtol=0, atol=0)
    torch.testing.assert_close(residual, x + x, rtol=0, atol=0)
    torch.testing.assert_close(residual, expected_residual, rtol=0, atol=0)
    reference = F.layer_norm(x + x, (256,), eps=1e-5)
    torch.testing.assert_close(quant(eager), quant(reference), rtol=0.04, atol=0.05)
    if precision == "fp4":
        assert torch.equal(
            valid_scales(result.scale, 41, 256), valid_scales(eager.scale, 41, 256)
        )


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
@torch.inference_mode()
def test_profile_grouped_fusion_restores_and_checks_weights(tmp_path, precision):
    model = torch.nn.Sequential(Qwen3VLTextMLP()).cuda().bfloat16().eval()
    sites, groups, _, _ = inventory(model)
    x = torch.randn(1, 41, 256, device="cuda", dtype=torch.bfloat16)
    original = model(x)
    cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    with ActivationCalibration(sites, max_rows=8) as calibration:
        for factor in (1, 2, 3):
            model(x * factor)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    assert set(calibration.maxima) == set(sites)
    assert all(v.shape[0] <= 8 for v in calibration.samples.values())
    shared = channel_scales(
        calibration.maxima[groups[0][0]], [sites[n][0].weight for n in groups[0]], 0.5
    ).cpu()
    records = {}
    for name, (module, _, _) in sites.items():
        s = (
            shared
            if name in groups[0]
            else channel_scales(calibration.maxima[name], [module.weight], 0.5).cpu()
        )
        records[name] = {"scale": s, "weight_sha256": weight_digest(module.weight)}
    profile = {
        "schema": PROFILE_SCHEMA,
        "upstream_revision": UPSTREAM_REVISION,
        "precision": precision,
        "linears": records,
    }
    path = tmp_path / "profile.pt"
    torch.save(profile, path)
    controller = FloatingProjections(
        model, precision=precision, grouped=True, smoothquant=path
    )
    try:
        reference = model(x)
    finally:
        controller.close()
    controller = FloatingProjections(
        model, precision=precision, grouped=True, swiglu=True, smoothquant=path
    )
    try:
        torch.testing.assert_close(model(x), reference, rtol=0, atol=0)
    finally:
        controller.close()
    torch.testing.assert_close(model(x), original, rtol=0, atol=0)
    records[groups[0][0]]["weight_sha256"] = "wrong"
    torch.save(profile, path)
    with pytest.raises(ValueError, match="weight mismatch"):
        load_profile(path, sites, groups, precision)
    assert all(not m._forward_hooks for m in model.modules())
