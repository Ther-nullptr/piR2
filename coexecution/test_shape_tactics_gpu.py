"""Opt-in exact integer dispatch and changing-shape CUDA Graph checks."""

import os
from types import SimpleNamespace

import pytest

if os.environ.get("PIR2_GPU_TESTS") != "1":
    pytest.skip("Set PIR2_GPU_TESTS=1 for explicit CUDA tests", allow_module_level=True)

import torch
from robotics_kernels.ampere_ada.integer import IntegerProjectionGroup
from robotics_kernels.common.graph import CudaGraphCall

from coexecution.quantization import (
    PreparedLinear,
    TransformerINT,
    install_shape_tactics,
    prepare,
    selected_tactic,
)
from coexecution.quantization_connections import packed_input
from coexecution.test_quantization_gpu import equal


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("native", [False, True])
@torch.inference_mode()
def test_shape_tactics_preserve_eager_packed_and_graph_outputs(
    bits, native, monkeypatch
):
    torch.manual_seed(415)
    model = torch.nn.Sequential(
        torch.nn.Linear(130, 64, device="cuda", dtype=torch.bfloat16)
    ).eval()
    adapter = TransformerINT(model, {"0"}, native=native, expanded=True)
    try:
        adapter.set(f"w{bits}a{bits}")
        q = adapter.quant[bits]["0"]
        inputs = {
            m: torch.randn(1, m, 130, device="cuda", dtype=torch.bfloat16)
            for m in (41, 153, 17)
        }
        expected = {m: model(x).clone() for m, x in inputs.items()}
        choices = {(bits, 41, 130, 64, True): 2, (bits, 153, 130, 64, True): 7}
        calls = []
        operation = (
            torch.ops.pir2_integer_dispatch.linear if native else q._ops.gemm_biasless
        )

        def recorded(*args):
            calls.append(args[-1])
            return operation(*args)

        if native:
            monkeypatch.setattr(torch.ops.pir2_integer_dispatch, "linear", recorded)
        else:
            q._ops = SimpleNamespace(gemm=q._ops.gemm, gemm_biasless=recorded)
        install_shape_tactics(q, choices)
        original = q._reference_ops
        install_shape_tactics(q, choices)
        assert q._reference_ops is original  # Reconfiguration must not nest wrappers.
        assert [selected_tactic(q, m) for m in inputs] == [2, 7, 0]
        graph = CudaGraphCall(model)
        for m in (41, 153, 17, 41):
            x = inputs[m]
            equal(model(x), expected[m])
            assert calls[-1] == {41: 2, 153: 7}.get(m, 0)
            equal(
                adapter.dispatch(q, packed_input(prepare(x, bits), x.shape)),
                expected[m],
            )
            equal(graph(x), expected[m])
            equal(graph(x + 0.125), model(x + 0.125))
        assert graph.captures == 4
        assert q.tactic == 0
    finally:
        adapter.close()


@pytest.mark.parametrize("bits", [8, 4])
@torch.inference_mode()
def test_grouped_shape_tactics_use_combined_width_and_one_gemm(bits):
    torch.manual_seed(517)
    linears = [
        PreparedLinear.from_linear(
            torch.nn.Linear(130, n, device="cuda", dtype=torch.bfloat16), bits=bits
        )
        for n in (64, 32, 32)
    ]
    group = IntegerProjectionGroup(linears)
    q = group.linear
    calls = []
    original = q._ops

    def recorded(a, sa, b, sb, bias, precision, tactic):
        calls.append(tactic)
        return original.gemm(a, sa, b, sb, bias, precision, tactic)

    q._ops = SimpleNamespace(gemm=recorded, gemm_biasless=original.gemm_biasless)
    install_shape_tactics(q, {(bits, 41, 130, 128, True): 2})
    for m in (41, 17):
        x = torch.randn(1, m, 130, device="cuda", dtype=torch.bfloat16)
        expected = [linear(x) for linear in linears]
        before = len(calls)
        actual = [group.project(x, i) for i in range(3)]
        assert len(calls) == before + 1
        assert calls[-1] == (2 if m == 41 else 0)
        for a, e in zip(actual, expected):
            equal(a, e)
