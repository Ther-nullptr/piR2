"""Opt-in numerical contracts for the owned Ada candidates; never timing claims."""

import os

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip(
        "Set PIR2_GPU_TESTS=1 for CUDA candidate checks", allow_module_level=True
    )

import torch

from coexecution.kernels.ampere_ada import epilogue, int4_gemv, int4_small_m, split_k
from coexecution.kernels.ampere_ada.integer import IntegerLinear, IntegerProjectionGroup
from coexecution.kernels.ampere_ada.norm import prepare_layernorm
from coexecution.kernels.ampere_ada.producers import prepare_rmsnorm
from coexecution.kernels.ampere_ada.rms_statistics import (
    _reference_mean_square,
    mean_square,
)
from coexecution.optimization.backends.integer.linear import prepare

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9),
    reason="Native candidate cubins target CUDA capability 8.9",
)


def _case(bits, rows, width=4096, outputs=128, bias=True):
    linear = torch.nn.Linear(
        width, outputs, bias=bias, device="cuda", dtype=torch.bfloat16
    ).eval()
    q = IntegerLinear.from_linear(
        linear, bits=bits, pack_reuse=True, biasless_epilogue=True
    )
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    return q, x, prepare(x, bits)


def _reference(q, packed):
    operation = q._ops.gemm_biasless if q.biasless_epilogue else q._ops.gemm
    return operation(
        packed.data,
        packed.scales,
        q.packed_weight,
        q.weight_scale,
        q.bias,
        q.bits,
        q.tactic,
    )


def _equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    integer = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(
        actual.contiguous().view(integer), expected.contiguous().view(integer)
    )


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("outputs", [15, 128, 4104])
@torch.inference_mode()
def test_int4_gemv_preserves_reference_rounding_and_padding(bias, outputs):
    q, _, packed = _case(4, 1, outputs=outputs, bias=bias)
    _equal(int4_gemv.int4_gemv(packed, q), _reference(q, packed))


@pytest.mark.parametrize("tactic", range(6))
@pytest.mark.parametrize("rows", [1, 17, 64])
@torch.inference_mode()
def test_small_m_tiles_keep_exact_reference_bf16(tactic, rows):
    q, _, packed = _case(4, rows)
    _equal(int4_small_m.small_m_gemm(packed, q, tactic), _reference(q, packed))


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("slices", [2, 4, 8])
@pytest.mark.parametrize("tile", [0, 1, 2])
@torch.inference_mode()
def test_split_k_reduces_integer_partials_before_scaling(bits, slices, tile):
    q, _, packed = _case(bits, 7)
    _equal(split_k.split_k_gemm(packed, q, slices, tile), _reference(q, packed))
    residual = torch.randn(7, q.out_features, device="cuda", dtype=torch.bfloat16)
    _equal(
        split_k.split_k_gemm(packed, q, slices, tile, residual=residual),
        _reference(q, packed) + residual,
    )


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("tactic", [0, 2, 7])
@torch.inference_mode()
def test_residual_epilogue_keeps_projection_bf16_boundary(bits, tactic):
    q, _, packed = _case(bits, 9)
    q.tactic = tactic
    residual = torch.randn(9, q.out_features, device="cuda", dtype=torch.bfloat16)
    _equal(
        epilogue.residual_gemm(packed, q, residual), _reference(q, packed) + residual
    )


@pytest.mark.parametrize("bits", [4, 8])
@torch.inference_mode()
def test_segmented_group_writeback_matches_reference_contiguous_slices(bits):
    projections = [_case(bits, 2, width=128, outputs=size)[0] for size in (24, 40, 64)]
    group = IntegerProjectionGroup(projections, contiguous_outputs=True)
    x = torch.randn(2, 3, 128, device="cuda", dtype=torch.bfloat16)
    packed = prepare(x, bits)
    outputs = epilogue.group_outputs(packed, group)
    expected = _reference(group.linear, packed)
    for actual, size, offset in zip(outputs, group.sizes, group.offsets):
        assert actual.is_contiguous()
        _equal(actual, expected[:, offset : offset + size].reshape(2, 3, size))


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("width", [128, 2048, 4097])
@torch.inference_mode()
def test_layernorm_pack_preserves_native_statistics_and_fallback(bits, width):
    x = torch.randn(2, 7, width, device="cuda", dtype=torch.bfloat16)
    scale, shift = torch.randn(
        2, 1, 2 * width, device="cuda", dtype=torch.bfloat16
    ).chunk(2, -1)
    actual = prepare_layernorm(x, bits, 1e-5, scale, shift)
    normalized = torch.nn.functional.layer_norm(x, (width,), eps=1e-5)
    expected = prepare(normalized * (1 + scale) + shift, bits)
    assert torch.equal(actual.data, expected.data)
    _equal(actual.scales, expected.scales)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("width", [128, 2048, 4097])
@torch.inference_mode()
def test_rms_statistics_and_pack_keep_bf16_boundaries(bits, width):
    x = torch.randn(2, 7, width, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)
    _equal(mean_square(x), _reference_mean_square(x))
    values = x.float()
    normalized = (
        values * torch.rsqrt((values * values).mean(-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    expected = prepare(normalized * weight, bits)
    actual = prepare_rmsnorm(x, weight, 1e-6, bits)
    assert torch.equal(actual.data, expected.data)
    _equal(actual.scales, expected.scales)


@torch.inference_mode()
def test_graph_replay_reads_fresh_activation_scales_and_residuals():
    q, x, packed = _case(4, 7)
    residual = torch.randn(7, q.out_features, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            epilogue.residual_gemm(packed, q, residual)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = epilogue.residual_gemm(packed, q, residual)
    fresh = prepare(torch.randn_like(x), 4)
    packed.data.copy_(fresh.data)
    packed.scales.copy_(fresh.scales)
    residual.normal_()
    graph.replay()
    _equal(result, _reference(q, packed) + residual)
