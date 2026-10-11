"""Explicit Thor producer/group tests; not policy task-quality acceptance."""

import os
from types import SimpleNamespace

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip("Set PIR2_GPU_TESTS=1 for explicit CUDA tests", allow_module_level=True)

import torch
import torch.nn.functional as F

from coexecution.floating_connections import FloatingConnections
from coexecution.floating_packing import prepare_norm, prepare_swiglu
from coexecution.floating_quantization import FloatingLinear, FloatingProjections
from coexecution.test_floating_quantization_gpu import Qwen3VLTextAttention


@pytest.mark.parametrize("residual", [False, True])
def test_norm_fusion_rejects_position_embeddings_before_patching(residual):
    def block_with_position_embeddings(enabled):
        block = torch.nn.Module()
        block.norm_type = "ada_norm"
        block.pos_embed = torch.nn.Identity() if enabled else None
        block.norm1 = torch.nn.Module()
        block.norm1.norm = torch.nn.LayerNorm(32, elementwise_affine=False)
        block.norm3 = torch.nn.LayerNorm(32, elementwise_affine=False)
        block.attn1 = torch.nn.Module()
        block.attn1.is_cross_attention = False
        for name in ("to_q", "to_k", "to_v"):
            setattr(block.attn1, name, torch.nn.Identity())
        projection = torch.nn.Module()
        projection.proj = torch.nn.Identity()
        block.ff = torch.nn.Module()
        block.ff.net = torch.nn.ModuleList([projection])
        return block

    # A later incompatible block must not leave earlier compatible norms patched.
    blocks = torch.nn.ModuleList(
        [block_with_position_embeddings(False), block_with_position_embeddings(True)]
    )
    policy = SimpleNamespace(
        model=SimpleNamespace(
            action_head=SimpleNamespace(
                model=SimpleNamespace(transformer_blocks=blocks)
            )
        )
    )
    selected = [
        projection
        for block in blocks
        for projection in (
            block.attn1.to_q,
            block.attn1.to_k,
            block.attn1.to_v,
            block.ff.net[0].proj,
        )
    ]
    controller = SimpleNamespace(
        precision="fp8",
        sites={str(i): (projection, "", "") for i, projection in enumerate(selected)},
        coverage={},
    )
    with pytest.raises(ValueError, match="Floating norm fusion requires no position"):
        FloatingConnections(policy, controller, residual=residual)
    assert all("forward" not in module.__dict__ for module in blocks.modules())
    assert controller.coverage == {}


def valid_scales(scales, rows, k):
    row = torch.arange(rows, device=scales.device)[:, None]
    sf = torch.arange(k // 16, device=scales.device)[None, :]
    offset = (
        (row // 128) * ((k // 16 + 3) // 4) * 512
        + row % 32 * 16
        + (row % 128 // 32) * 4
        + sf // 4 * 512
        + sf % 4
    )
    return scales.flatten()[offset]


def unpack(packed):
    if packed.precision == "fp8":
        return (packed.data.float() * packed.scale).reshape(packed.shape)
    k = packed.shape[-1]
    data = packed.data
    code = torch.stack((data & 15, data >> 4), dim=-1).flatten(1).long()
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=data.device)
    values = levels[code & 7] * torch.where(code < 8, 1, -1)
    scales = (
        valid_scales(packed.scale, data.shape[0], k)
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    return (values * scales.repeat_interleave(16, dim=-1)).reshape(packed.shape)


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
@pytest.mark.parametrize("rows,k", [(1, 256), (41, 1536), (153, 6144)])
@torch.inference_mode()
def test_swiglu_native_rounding_pack_and_graph(precision, rows, k):
    torch.manual_seed(18)
    joint = torch.randn(rows, 2 * k, device="cuda", dtype=torch.bfloat16)
    gate, up = joint.chunk(2, dim=-1)
    lut = F.silu(
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    linear = torch.nn.Linear(k, 128, device="cuda", dtype=torch.bfloat16).eval()
    quant = FloatingLinear(linear, "swiglu-test", precision, fast_fp8=True)

    def check(packed):
        native = F.silu(gate) * up
        expected, scale = quant.pack_input(native)
        assert torch.equal(packed.data.view(torch.uint8), expected.view(torch.uint8))
        if precision == "fp8":
            assert torch.equal(packed.scale, scale)
        else:
            assert torch.equal(
                valid_scales(packed.scale, rows, k), valid_scales(scale, rows, k)
            )
        torch.testing.assert_close(quant(packed), quant(native), rtol=0, atol=0)

    for factor in (1.0, 0.0, 100.0):
        joint.normal_().mul_(factor)
        check(prepare_swiglu(gate, up, lut, precision))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = prepare_swiglu(gate, up, lut, precision)
    for factor in (0.125, 9.0):
        joint.normal_().mul_(factor)
        graph.replay()
        check(captured)


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
@pytest.mark.parametrize(
    "per_token,residual", [(False, False), (True, False), (False, True)]
)
@torch.inference_mode()
def test_norm_residual_pack_rounding_and_graph(precision, per_token, residual):
    torch.manual_seed(317)
    x = torch.randn(2, 41, 1536, device="cuda", dtype=torch.bfloat16)
    y = torch.randn_like(x) if residual else None
    s = (
        torch.randn_like(x)
        if per_token
        else torch.randn(2, 1536, device=x.device, dtype=x.dtype)
    )
    h = torch.randn_like(s)
    x[0, 0].fill_(0.25)

    def run():
        return prepare_norm(
            x,
            precision,
            eps=1e-5,
            scale=None if residual else s,
            shift=None if residual else h,
            residual=y,
        )

    def check(result):
        packed, summed = result
        actual_x = x if y is None else x + y
        if residual:
            torch.testing.assert_close(summed, actual_x, rtol=0, atol=0)
        reference = F.layer_norm(actual_x, (1536,), eps=1e-5)
        if not residual:
            scale, shift = (s, h) if per_token else (s[:, None], h[:, None])
            reference = reference * (1 + scale) + shift
        actual = unpack(packed)
        relative_l2 = torch.linalg.vector_norm(
            actual - reference.float()
        ) / torch.linalg.vector_norm(reference.float())
        assert relative_l2 < (0.04 if precision == "fp8" else 0.15), relative_l2
        assert torch.isfinite(actual).all()

    check(run())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for factor in (0.0, 0.125, 9.0):
        x.normal_().mul_(factor)
        graph.replay()
        check(captured)
        eager = run()
        assert torch.equal(
            captured[0].data.view(torch.uint8), eager[0].data.view(torch.uint8)
        )


class Qwen3VLTextMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_act="silu")
        self.gate_proj = torch.nn.Linear(256, 512, bias=False)
        self.up_proj = torch.nn.Linear(256, 512, bias=False)
        self.down_proj = torch.nn.Linear(512, 256, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
@torch.inference_mode()
def test_merged_projection_scope_scale_policy_restoration(precision):
    torch.manual_seed(46)
    model = torch.nn.Sequential(Qwen3VLTextAttention()).cuda().bfloat16().eval()
    owner = model[0]
    owner.k_proj.weight.mul_(0.125)  # Unequal FP8 weight scales must be explicit.
    x = torch.randn(2, 41, 256, device="cuda", dtype=torch.bfloat16)
    original_input = x.clone()
    original = owner(x)
    controller = FloatingProjections(model, precision=precision, grouped=True)
    group = controller.groups[0]
    try:

        def check(value, other=None):
            actual = owner(value, other=other)
            for index, result in enumerate(actual):
                source = other if index == 1 and other is not None else value
                expected = group.linear(source).split(group.sizes, dim=-1)[index]
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
            assert group.cache is None and not group.active

        check(x)
        check(x, x * 0.125)
        with pytest.raises(RuntimeError, match="partial group"):
            owner(x, fail=True)
        assert group.cache is None and not group.active
        check(x * 9)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = owner(x)
        x.mul_(7)
        graph.replay()
        for a, b in zip(captured, owner(x)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        del graph, captured
    finally:
        controller.close()
    for a, b in zip(owner(original_input), original):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert all("forward" not in m.__dict__ for m in model.modules())


@pytest.mark.parametrize("precision", ["fp8", "fp4"])
@torch.inference_mode()
def test_grouped_swiglu_model_is_exact_to_unfused_activation(precision):
    model = torch.nn.Sequential(Qwen3VLTextMLP()).cuda().bfloat16().eval()
    x = torch.randn(1, 153, 256, device="cuda", dtype=torch.bfloat16)
    original = model(x)
    with_controller = FloatingProjections(model, precision=precision, grouped=True)
    reference = model(x)
    with_controller.close()
    controller = FloatingProjections(
        model, precision=precision, grouped=True, swiglu=True
    )
    try:
        torch.testing.assert_close(model(x), reference, rtol=0, atol=0)
        assert len(controller.coverage["swiglu_producers"]) == 1
    finally:
        controller.close()
    torch.testing.assert_close(model(x), original, rtol=0, atol=0)
