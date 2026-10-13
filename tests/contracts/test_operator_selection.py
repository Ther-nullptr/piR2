"""Operator metadata and selection contracts need no tensor/GPU dependencies."""

import dataclasses
import json
import subprocess
import sys

import pytest

from coexecution.optimization.operators import (
    CandidateMeasurement,
    MeasurementContext,
    OperatorProfile,
    inventory,
    read_measurements,
    read_operator_profile,
    select_measured,
    validate_profile,
)


def context(**changes):
    values = {
        "architecture": "sm_89",
        "precision": "w4a4",
        "shape": (41, 4096, 1536),
        "cohort": "same-device-randomized-round",
        "workload": "checkpoint-inputs-and-policy-settings-digest",
        "environment": "gpu-uuid-clocks-torch-cuda-digest",
        "boundary": "operator_chain",
        "timing": "cuda_event",
        "statistic": "p50",
        "numerical_gate": "same-quantized-parent-atol0-rtol0",
    }
    return MeasurementContext(**(values | changes))


def candidate(candidate_id="reference", latency_ms=1.0, **changes):
    values = {
        "candidate_id": candidate_id,
        "family": "integer_gemm",
        "implementation": "reference" if candidate_id == "reference" else "small_m",
        "context": context(),
        "latency_ms": latency_ms,
        "samples": 20,
        "numerical_passed": True,
        "source_revision": "measured-source-tree-sha256",
        "artifact": "retained-measurement-sha256",
    }
    return CandidateMeasurement(**(values | changes))


def test_import_does_not_load_gpu_packages():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import coexecution.optimization.operators; "
            "assert not {'torch', 'triton'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_inventory_distinguishes_integer_and_thor_and_declares_contracts():
    rows = inventory()
    families = {row.family for row in rows}
    assert {
        "vision_rope",
        "text_rope",
        "rmsnorm",
        "adaln",
        "integer_gemm",
        "activation_pack",
        "layernorm_pack",
        "integer_residual",
        "group_writeback",
        "floating_gemm",
        "context_pack",
        "rmsnorm_pack",
        "final_modulation_pack",
    } <= families
    assert all(
        row.source and row.shape_contract and row.numerical_contract for row in rows
    )
    thor = inventory("floating_gemm")
    assert all(row.architectures == ("sm_110",) for row in thor)
    assert all(set(row.precisions) <= {"fp8", "fp4"} for row in thor)


def test_explicit_partial_profile_preserves_unspecified_defaults(tmp_path):
    path = tmp_path / "operators.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "architecture": "sm_89",
                "precision": "w4a4",
                "implementations": {"integer_gemm": "split_k", "adaln": "triton"},
                "parameters": {"integer_gemm": {"slices": 4, "tile": 1}},
            }
        )
    )
    profile = read_operator_profile(path)
    validate_profile(profile, architecture="sm_89", precision="w4a4")
    assert profile.get("integer_gemm", "reference") == "split_k"
    assert profile.get("rmsnorm", "triton") == "triton"
    assert profile.options("integer_gemm") == {"slices": 4, "tile": 1}
    with pytest.raises(TypeError):
        profile.implementations["integer_gemm"] = "reference"


@pytest.mark.parametrize(
    "implementations,parameters",
    [
        ({"unknown": "native"}, {}),
        ({"text_rope": "native"}, {}),
        ({"integer_gemm": "small_m"}, {"integer_gemm": {"tactic": True}}),
        ({"integer_gemm": "small_m"}, {"integer_gemm": {"tactic": 6}}),
        ({"integer_gemm": "split_k"}, {"integer_gemm": {"slices": 16}}),
        ({"integer_gemm": "split_k"}, {"integer_gemm": {"tile": 3}}),
        ({"integer_gemm": "reference"}, {"integer_gemm": {"slices": 2}}),
        ({"adaln": "native"}, {"integer_gemm": {"tactic": 0}}),
    ],
)
def test_profile_rejects_unsupported_choices_and_parameters(
    implementations, parameters
):
    with pytest.raises(ValueError):
        OperatorProfile("sm_89", "w4a4", implementations, parameters)


@pytest.mark.parametrize("version", [True, 2, "1"])
def test_profile_json_version_is_strict(tmp_path, version):
    path = tmp_path / "operators.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": version,
                "architecture": "sm_89",
                "precision": "bf16",
                "implementations": {"adaln": "native"},
            }
        )
    )
    with pytest.raises(ValueError, match="schema_version"):
        read_operator_profile(path)


def test_profile_validates_actual_architecture_and_precision():
    profile = OperatorProfile("sm_89", "w4a4", {"integer_gemm": "small_m"})
    with pytest.raises(ValueError, match="architecture"):
        validate_profile(profile, architecture="sm_87", precision="w4a4")
    with pytest.raises(ValueError, match="precision"):
        validate_profile(profile, architecture="sm_89", precision="w8a8")
    with pytest.raises(ValueError, match="architecture"):
        OperatorProfile("sm_87", "w4a4", {"integer_gemm": "small_m"})
    with pytest.raises(ValueError, match="precision"):
        OperatorProfile("sm_89", "w8a8", {"integer_gemm": "small_m"})
    with pytest.raises(ValueError, match="precision"):
        OperatorProfile("sm_110", "fp8", {"integer_gemm": "reference"})


def test_measured_selection_uses_complete_boundary_and_gate():
    baseline = candidate()
    failed = candidate("failed-fastest", 0.1, numerical_passed=False)
    passed = candidate("measured-small-m", 0.8)
    choice = select_measured([baseline, failed, passed], baseline="reference")
    assert choice.winner == passed
    assert choice.rejected == ("failed-fastest",)
    assert choice.relative_gain == pytest.approx(0.2)
    assert choice.scope == "operator_chain"


@pytest.mark.parametrize(
    "field,value",
    [
        ("architecture", "sm_87"),
        ("precision", "w8a8"),
        ("shape", (1, 4096, 1536)),
        ("cohort", "another-physical-device-or-round"),
        ("workload", "another-checkpoint"),
        ("environment", "another-compiler"),
        ("boundary", "operator"),
        ("timing", "wall"),
        ("statistic", "p95"),
        ("numerical_gate", "unrelated-bf16-reference"),
    ],
)
def test_selection_rejects_mismatched_measurement_context(field, value):
    # Reference is portable, so context mismatches reach the comparison gate.
    other = candidate(
        "different-context",
        0.2,
        implementation="reference",
        context=context(**{field: value}),
    )
    with pytest.raises(ValueError, match=field):
        select_measured([candidate(), other], baseline="reference")


def test_deployment_winner_requires_full_policy_wall_latency():
    with pytest.raises(ValueError, match="full_policy.*wall"):
        select_measured(
            [candidate(), candidate("small", 0.5)],
            baseline="reference",
            deployment=True,
        )
    ctx = context(boundary="full_policy", timing="wall")
    choice = select_measured(
        [candidate(context=ctx), candidate("small", 0.5, context=ctx)],
        baseline="reference",
        deployment=True,
    )
    assert choice.winner.candidate_id == "small"
    assert choice.scope == "deployment_latency"


def test_ties_and_insufficient_gain_keep_baseline():
    choice = select_measured(
        [candidate(), candidate("small", 0.999)],
        baseline="reference",
        minimum_relative_gain=0.01,
    )
    assert choice.winner.candidate_id == "reference"
    choice = select_measured([candidate(), candidate("small")], baseline="reference")
    assert choice.winner.candidate_id == "reference"


@pytest.mark.parametrize("latency", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_latencies_are_not_measurements(latency):
    with pytest.raises(ValueError, match="latency_ms"):
        candidate(latency_ms=latency)


def test_failed_baseline_or_duplicate_records_are_rejected():
    with pytest.raises(ValueError, match="baseline.*numerical"):
        select_measured([candidate(numerical_passed=False)], baseline="reference")
    with pytest.raises(ValueError, match="candidate_id"):
        select_measured([candidate(), candidate()], baseline="reference")


def test_measurement_json_round_trip_preserves_comparison_identity(tmp_path):
    rows = [candidate(), candidate("small", 0.8, parameters={"tactic": 4})]
    path = tmp_path / "measurements.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidates": [dataclasses.asdict(row) for row in rows],
            }
        )
    )
    assert read_measurements(path) == tuple(rows)


def test_candidate_retains_bounded_implementation_parameters():
    record = candidate(
        "split-four", 0.8, implementation="split_k", parameters={"slices": 4}
    )
    assert record.parameters == {"slices": 4}
    with pytest.raises(ValueError, match="slices"):
        candidate("invalid", implementation="split_k", parameters={"slices": 16})
