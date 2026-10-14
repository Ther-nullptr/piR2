"""Explicit INT8/INT4 CUDA math, packing, maps and instance isolation checks.

PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_balanced_integer_gpu.py
"""

import os
from copy import deepcopy
from types import SimpleNamespace

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip("Set PIR2_GPU_TESTS=1 for integer CUDA checks", allow_module_level=True)

import torch

from coexecution.kernels.ampere_ada.balanced import prepare, prepare_modulation
from coexecution.kernels.ampere_ada.integer import PackedActivation
from coexecution.kernels.common.fused import prepare_gelu
from coexecution.optimization.backends.integer.linear import PreparedLinear
from coexecution.optimization.groot.calibration import (
    ActivationCalibration,
    install_balance,
)
from coexecution.optimization.groot.integer import TransformerINT


def reference_pack(value, bits):
    """Independent FP64 division oracle rounded to FP32 before integer rounding."""
    k = value.shape[-1]
    value = value.view(-1, k).float()
    limit = (1 << (bits - 1)) - 1
    maximum = value.abs().amax(-1)
    scale = torch.where(maximum > 0, maximum * (1.0 / limit), torch.ones_like(maximum))
    quotient = (value.double() / scale[:, None].double()).float()
    q = torch.round(quotient).clamp(-limit, limit).to(torch.int32)
    pad = (k + 127) // 128 * 128
    q = torch.nn.functional.pad(q, (0, pad - k))
    data = (
        q.to(torch.int8)
        if bits == 8
        else ((q[:, 0::2] & 15) | ((q[:, 1::2] & 15) << 4)).to(torch.uint8)
    )
    return data, scale


def unpack(data, bits):
    if bits == 8:
        return data.to(torch.int32)
    values = torch.stack((data & 15, (data >> 4) & 15), -1).flatten(-2).to(torch.int32)
    return torch.where(values >= 8, values - 16, values)


@pytest.mark.parametrize(
    "width", [130, 132, 256, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
)
@pytest.mark.parametrize("kind", ["plain", "gelu", "swiglu", "zero"])
def test_register_pack_matches_reference_and_independent_oracle(width, kind):
    torch.manual_seed(321 + width)
    # Feature-contiguous, non-contiguous row stride is a real grouped-output case.
    x = torch.randn(5, width * 2, device="cuda", dtype=torch.bfloat16)[:, :width]
    inverse = torch.rand(width, device="cuda", dtype=torch.float32) + 0.1
    lut = up = None
    value = x.float()
    if kind == "zero":
        x.zero_()
        value = x.float()
    elif kind in ("gelu", "swiglu"):
        lut = (
            prepare_gelu(x.device, x.dtype, "tanh")
            if kind == "gelu"
            else torch.nn.functional.silu(
                torch.arange(65536, device="cuda", dtype=torch.int32)
                .to(torch.int16)
                .view(torch.bfloat16)
            )
        )
        value = lut[(x.view(torch.int16).to(torch.int32) & 65535).long()].float()
        if kind == "swiglu":
            up = x
            value = (value * up.float()).to(torch.bfloat16).float()
    oracle, scales = reference_pack(value * inverse, 4)
    with torch.inference_mode():
        for route in ("reference", "register"):
            result = prepare(x, inverse, lut, up, route=route)
            torch.testing.assert_close(result.data, oracle, rtol=0, atol=0)
            torch.testing.assert_close(result.scales, scales, rtol=0, atol=0)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_real_integer_gemm_matches_integer_dot_scale_and_bias(bits, bias):
    torch.manual_seed(322)
    source = torch.nn.Linear(
        132, 24, bias=bias, device="cuda", dtype=torch.bfloat16
    ).eval()
    q = PreparedLinear.from_linear(
        source, bits=bits, pack_reuse=True, biasless_epilogue=True
    )
    x = torch.randn(5, 132, device="cuda", dtype=torch.bfloat16)
    data, scales = reference_pack(x, bits)
    packed = PackedActivation(data, scales, (5,), 132, bits)
    # FP64 integer-valued dot is exact at these dimensions; TF32 is not used.
    dot = (
        unpack(data, bits).double() @ unpack(q.packed_weight, bits).double().t()
    ).float()
    expected = dot * scales[:, None]
    expected = expected * q.weight_scale[None, :]
    if q.bias is not None:
        expected = expected + q.bias.float()[None, :]
    with torch.inference_mode():
        actual = q.forward_packed(packed)
    torch.testing.assert_close(
        actual, expected[:, :24].to(torch.bfloat16), rtol=0, atol=0
    )


def test_balanced_modulation_preserves_bf16_boundaries():
    torch.manual_seed(323)
    x = torch.randn(2, 7, 132, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(2, 1, 132, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn_like(scale)
    inverse = torch.rand(132, device="cuda", dtype=torch.float32) + 0.1
    value = (x * (1 + scale)) + shift
    expected, scales = reference_pack(value.float() * inverse, 4)
    actual = prepare_modulation(x, scale, shift, inverse)
    torch.testing.assert_close(actual.data, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.scales, scales, rtol=0, atol=0)


def test_calibration_maps_restore_and_do_not_patch_other_instances(tmp_path):
    torch.manual_seed(324)
    model = (
        torch.nn.Sequential(
            torch.nn.Linear(128, 128, device="cuda", dtype=torch.bfloat16),
            torch.nn.Linear(128, 8, device="cuda", dtype=torch.bfloat16),
        )
        .eval()
        .requires_grad_(False)
    )
    x = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16)
    other_model = deepcopy(model)
    policy = SimpleNamespace(model=model)
    originals = [(m, m.forward, "forward" in m.__dict__) for m in model]
    with torch.inference_mode():
        with ActivationCalibration(policy) as calibration:
            baseline = model(x)
        path = tmp_path / "maxima.pt"
        torch.save(calibration.statistics(), path)
        names = {"0": {}, "1": {}}
        controller = TransformerINT(model, names, native=True, expanded=True)
        other_controller = TransformerINT(
            other_model, names, native=True, expanded=True
        )
        other_controller.set("w4a4")
        other_output = other_model(x).clone()
        # This minimal CUDA fixture has no category/AdaLN producers.
        extra = SimpleNamespace(linears={}, quant={4: {}, 8: {}})
        try:
            controller.set("w8a8")
            fixed8 = model(x).clone()
            controller.set("bf16")
            install_balance(controller, extra, path)
            torch.testing.assert_close(other_model(x), other_output, rtol=0, atol=0)
            assert all(
                getattr(q, "inverse_balance", None) is None
                for q in other_controller.quant[4].values()
            )
            torch.testing.assert_close(model(x), baseline, rtol=0, atol=0)
            controller.set_mapping({"0": 8, "1": 8})
            torch.testing.assert_close(model(x), fixed8, rtol=0, atol=0)
            controller.set("w4a4")
            fixed4 = model(x).clone()
            controller.set_mapping({"0": 4, "1": 4})
            torch.testing.assert_close(model(x), fixed4, rtol=0, atol=0)
            controller.set_mapping({"0": 8, "1": 4})
            expected = controller.dispatch(
                controller.quant[4]["1"],
                controller.dispatch(controller.quant[8]["0"], x),
            )
            torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
            with pytest.raises(ValueError, match="exactly"):
                controller.set_mapping({"0": 4})
            torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        finally:
            torch.cuda.synchronize()
            controller.close()
            other_controller.close()
        torch.testing.assert_close(model(x), baseline, rtol=0, atol=0)
    for m, forward, had in originals:
        assert m.forward == forward and ("forward" in m.__dict__) == had
