"""Malformed metadata fails before loading models or executing numerical kernels."""

import json

import pytest

from coexecution.optimization.operators import (
    CandidateMeasurement,
    OperatorProfile,
    read_measurements,
    read_operator_profile,
)


@pytest.mark.parametrize(
    "payload,message",
    [
        ([], "Expected a JSON object"),
        (
            {
                "schema_version": 1,
                "architecture": "sm_89",
                "precision": "w4a4",
                "implementations": [],
            },
            "implementations/parameters must be objects",
        ),
        (
            {
                "schema_version": 1,
                "architecture": "sm_89",
                "precision": "w4a4",
                "implementations": {"integer_gemm": "small_m"},
                "parameters": {"integer_gemm": []},
            },
            "Parameters for integer_gemm must be an object",
        ),
    ],
)
def test_profile_rejects_wrong_container_types(tmp_path, payload, message):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(TypeError, match=message):
        read_operator_profile(path)


@pytest.mark.parametrize("candidates", [None, {}, "invalid"])
def test_measurements_require_a_candidate_list(tmp_path, candidates):
    path = tmp_path / "measurements.json"
    path.write_text(json.dumps({"schema_version": 1, "candidates": candidates}))
    with pytest.raises(TypeError, match="candidates must be a JSON list"):
        read_measurements(path)


@pytest.mark.parametrize("candidate", [None, [], {"context": []}])
def test_measurements_require_a_context_object(tmp_path, candidate):
    path = tmp_path / "measurements.json"
    path.write_text(json.dumps({"schema_version": 1, "candidates": [candidate]}))
    with pytest.raises(TypeError, match="explicit context object"):
        read_measurements(path)


def test_direct_candidate_requires_typed_context():
    with pytest.raises(TypeError, match="context must be MeasurementContext"):
        CandidateMeasurement(
            candidate_id="synthetic-candidate",
            family="integer_gemm",
            implementation="small_m",
            context={},
            latency_ms=1.0,
            samples=1,
            numerical_passed=True,
            source_revision="synthetic-revision",
            artifact="synthetic-artifact",
        )


def test_valid_profile_and_invalid_values_keep_existing_contract():
    profile = OperatorProfile(
        architecture="sm_89",
        precision="w4a4",
        implementations={"integer_gemm": "small_m"},
        parameters={"integer_gemm": {"tactic": 2}},
    )
    assert profile.options("integer_gemm") == {"tactic": 2}
    with pytest.raises(ValueError, match="Unsupported"):
        OperatorProfile(
            architecture="sm_89",
            precision="w4a4",
            implementations={"integer_gemm": "small_m"},
            parameters={"integer_gemm": {"tactic": 6}},
        )
