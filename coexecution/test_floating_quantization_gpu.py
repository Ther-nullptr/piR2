"""Explicit Thor tests: PIR2_GPU_TESTS=1 and both floating backend SOs required.

These validate kernel/adapter contracts, not learned-policy quality.
"""

import os

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip("Set PIR2_GPU_TESTS=1 for explicit CUDA tests", allow_module_level=True)

import torch

from coexecution.floating_quantization import FloatingLinear, FloatingProjections


@torch.inference_mode()
def test_fp8_noncontiguous_inputs_and_quantized_reference():
    torch.manual_seed(1000)
    torch.backends.cuda.matmul.allow_tf32 = False
    linear = torch.nn.Linear(256, 128, device="cuda", dtype=torch.bfloat16).eval()
    q = FloatingLinear(linear, "test", "fp8")
    x = torch.randn(2, 256, 41, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    packed, scale = q.pack_input(x)
    reference = (packed.float() @ q.packed_weight.float().t()) * (
        scale * q.weight_scale
    ) + linear.bias.float()
    actual = q(x)
    assert packed.dtype == q.packed_weight.dtype == torch.float8_e4m3fn
    assert actual.shape == (2, 41, 128) and actual.dtype == torch.bfloat16
    torch.testing.assert_close(
        actual, reference.reshape_as(actual).to(torch.bfloat16), rtol=0.02, atol=0.01
    )


@pytest.mark.parametrize(
    "precision,fast", [("fp8", False), ("fp8", True), ("fp4", False)]
)
@torch.inference_mode()
def test_graph_refreshes_scale_and_adapter_restores_after_exception(precision, fast):
    torch.manual_seed(1000)
    linear = torch.nn.Linear(256, 256, device="cuda", dtype=torch.bfloat16).eval()
    model = torch.nn.Sequential(linear)
    x = torch.randn(41, 256, device="cuda", dtype=torch.bfloat16)
    original_input = x.clone()
    original = model(x).clone()
    original_weight = linear.weight.clone()
    rng = torch.cuda.get_rng_state().clone()
    controller = FloatingProjections(
        model, expanded=True, precision=precision, fast_fp8=fast
    )
    try:
        for _ in range(3):
            model(x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = model(x)
        for factor in (1.0, 9.0, 0.125):
            x.mul_(factor)
            expected = model(x)
            graph.replay()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        assert torch.equal(torch.cuda.get_rng_state(), rng)
        with pytest.raises(RuntimeError, match="body failure"):
            try:
                raise RuntimeError("body failure")
            finally:
                del graph, captured
                controller.close()
        assert "forward" not in linear.__dict__
        assert not controller.linears
        torch.testing.assert_close(linear.weight, original_weight, rtol=0, atol=0)
        x.copy_(original_input)
        torch.testing.assert_close(model(x), original, rtol=0, atol=0)
    finally:
        controller.close()


@pytest.mark.parametrize("factor", [0.0, 1.0e-14, 1.0, 100.0])
@torch.inference_mode()
def test_fast_fp8_packing_matches_reference(factor):
    from coexecution.floating_packing import prepare_fp8

    torch.manual_seed(1000)
    linear = torch.nn.Linear(256, 128, device="cuda", dtype=torch.bfloat16).eval()
    q = FloatingLinear(linear, "test", "fp8")
    x = torch.randn(41, 256, device="cuda", dtype=torch.bfloat16) * factor
    expected, expected_scale = q.pack_input(x)
    actual, scale = prepare_fp8(x)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 41])
@torch.inference_mode()
def test_fp4_preserves_nonzero_bias(rows):
    linear = torch.nn.Linear(256, 128, device="cuda", dtype=torch.bfloat16).eval()
    linear.bias.copy_(torch.linspace(-3, 3, 128, device="cuda"))
    q = FloatingLinear(linear, "biased", "fp4")
    actual = q(torch.zeros(rows, 256, device="cuda", dtype=torch.bfloat16))
    torch.testing.assert_close(actual, linear.bias.expand_as(actual), rtol=0, atol=0)


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
@torch.inference_mode()
def test_unaligned_projection_is_explicit_bf16_fallback(precision):
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 128, device="cuda", dtype=torch.bfloat16),
        torch.nn.Linear(128, 7, device="cuda", dtype=torch.bfloat16),
    ).eval()
    controller = FloatingProjections(model, expanded=True, precision=precision)
    try:
        assert set(controller.linears) == {"0"}
        assert controller.coverage["fallbacks"][0]["name"] == "1"
        assert "forward" not in model[1].__dict__
    finally:
        controller.close()


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


@pytest.mark.parametrize(
    "precision,fast", [("fp8", False), ("fp8", True), ("fp4", False)]
)
@torch.inference_mode()
def test_shared_inputs_are_exact_scoped_and_graph_safe(precision, fast):
    torch.manual_seed(1000)
    model = torch.nn.Sequential(Qwen3VLTextAttention()).cuda().bfloat16().eval()
    owner = model[0]
    x = torch.randn(1, 256, 41, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    original = owner(x)
    controller = FloatingProjections(
        model, precision=precision, fast_fp8=fast, shared_inputs=True
    )
    group = controller.groups[0]
    calls = []
    for index, linear in enumerate(group.linears):
        pack = linear.pack_input

        def counted(value, p=pack, i=index):
            calls.append(i)
            return p(value)

        linear.pack_input = counted

    def check(actual, value, other=None):
        expected = [
            q(value if i != 1 or other is None else other)
            for i, q in enumerate(group.linears)
        ]
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    try:
        assert len(controller.coverage["shared_input_groups"]) == 1
        assert len(controller.coverage["linears"]) == 4
        rng = torch.cuda.get_rng_state().clone()
        for factor in (1.0, 9.0, 0.125):
            x.mul_(factor)
            calls.clear()
            actual = owner(x)
            assert calls == [0]
            assert not group.active and group.cache is None
            check(actual, x)
        other = x * 2
        check(owner(x, other=other), x, other)
        with pytest.raises(RuntimeError, match="partial group"):
            owner(x, fail=True)
        assert not group.active and group.cache is None
        x.add_(1)
        check(owner(x), x)
        # A projection called outside its owner cannot leave reusable input state.
        owner.q_proj(x)
        x.mul_(2)
        torch.testing.assert_close(owner.k_proj(x), group.linears[1](x), rtol=0, atol=0)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = owner(x)
        for factor in (0.25, 7.0):
            x.mul_(factor)
            graph.replay()
            check(captured, x)
        del graph, captured
        assert torch.equal(torch.cuda.get_rng_state(), rng)
    finally:
        controller.close()
    assert not controller.groups and not controller.linears
    assert all("forward" not in m.__dict__ for m in model.modules())
    assert original[0].shape == owner(x)[0].shape


@torch.inference_mode()
def test_partial_floating_coverage_does_not_create_shared_group():
    model = torch.nn.Sequential(Qwen3VLTextAttention()).cuda().bfloat16().eval()
    controller = FloatingProjections(
        model, {"0.q_proj", "0.k_proj"}, shared_inputs=True
    )
    try:
        assert not controller.groups
        assert not controller.coverage["shared_input_groups"]
        assert "forward" not in model[0].__dict__
        assert "forward" not in model[0].v_proj.__dict__
    finally:
        controller.close()
