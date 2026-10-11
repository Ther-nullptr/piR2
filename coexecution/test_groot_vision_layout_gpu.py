"""CUDA Conv3d layout contracts; no checkpoint or task-quality claim.

PIR2_GPU_TESTS=1 python -m pytest -q coexecution/test_groot_vision_layout_gpu.py
"""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip("Set PIR2_GPU_TESTS=1 for vision CUDA checks", allow_module_level=True)

import torch
import torch.nn.functional as F

from coexecution.groot_optimization import GrootOptimizations, OptimizationConfig


@pytest.mark.parametrize("channels_last_before", [False, True])
@pytest.mark.parametrize("fail_body", [False, True])
def test_vision_layout_and_restoration(
    channels_last_before, fail_body, record_property
):
    torch.manual_seed(108)
    with torch.device("cuda"):
        model = torch.nn.Module()
        model.backbone = torch.nn.Module()
        model.backbone.model = torch.nn.Module()
        model.backbone.model.visual = torch.nn.Module()
        model.backbone.model.visual.patch_embed = torch.nn.Module()
        model.backbone.model.visual.patch_embed.proj = torch.nn.Conv3d(
            3, 16, kernel_size=(2, 4, 4), stride=(2, 4, 4)
        )
    model.config = SimpleNamespace(streaming=True)
    model = model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    proj = model.backbone.model.visual.patch_embed.proj
    if channels_last_before:
        proj.weight.data = proj.weight.data.contiguous(
            memory_format=torch.channels_last_3d
        )
    original = proj.weight.data
    parameter = proj.weight
    values = original.clone()
    bias = proj.bias
    x = torch.randn(7, 3, 2, 8, 8, dtype=torch.bfloat16, device="cuda")
    input_values = x.clone()
    original_calls = []
    existing_hook = proj.register_forward_pre_hook(
        lambda _module, inputs: original_calls.append(inputs[0].data_ptr())
    )
    original_hooks = dict(proj._forward_pre_hooks)
    # Both operands originate in BF16; only the reference accumulation is FP32.
    with (
        torch.inference_mode(),
        torch.backends.cudnn.flags(
            enabled=True, benchmark=False, deterministic=True, allow_tf32=False
        ),
    ):
        reference = F.conv3d(x.float(), original.float(), bias.float(), proj.stride)
        scope = GrootOptimizations(
            SimpleNamespace(model=model), OptimizationConfig(vision_channels_last=True)
        )
        expected_exit = (
            pytest.raises(RuntimeError, match="deliberate body failure")
            if fail_body
            else nullcontext()
        )
        with expected_exit, scope:
            assert proj.weight is parameter and proj.bias is bias
            assert proj.weight.is_contiguous(memory_format=torch.channels_last_3d)
            torch.testing.assert_close(proj.weight, values, rtol=0, atol=0)
            observed_layouts = []
            observer = proj.register_forward_pre_hook(
                lambda _module, inputs: observed_layouts.append(
                    inputs[0].is_contiguous(memory_format=torch.channels_last_3d)
                )
            )
            output = proj(x)
            observer.remove()
            assert observed_layouts == [True]
            torch.testing.assert_close(x, input_values, rtol=0, atol=0)
            error = (output.float() - reference).abs()
            record_property("vision_conv_max_abs_vs_fp32", error.max().item())
            record_property("vision_conv_mean_abs_vs_fp32", error.mean().item())
            torch.testing.assert_close(output.float(), reference, rtol=0.02, atol=0.02)
            if fail_body:
                raise RuntimeError("deliberate body failure")
    assert proj.weight is parameter and proj.bias is bias
    assert proj.weight.data_ptr() == original.data_ptr()
    assert (
        proj.weight.untyped_storage().data_ptr()
        == original.untyped_storage().data_ptr()
    )
    assert proj.weight.stride() == original.stride()
    torch.testing.assert_close(proj.weight, values, rtol=0, atol=0)
    assert proj._forward_pre_hooks == original_hooks
    assert original_calls == [x.data_ptr()]
    scope.close()
    existing_hook.remove()
