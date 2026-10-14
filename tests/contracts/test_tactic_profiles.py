"""Profile identities must match runtime evidence before tactic mutation."""

import json
from types import SimpleNamespace

import pytest

from coexecution.optimization.tactics import apply_tactics, read_shape_tactics

IDENTITY = {
    "backend_id": "pir2_integer",
    "backend_revision": "239b4a3ef2268048571c9f508ad5700f98398c2f",
    "device_arch": "sm_89",
    "quant_format": "w8a8-symmetric-row-v1",
}
ROW = {"bits": 8, "shape": [41, 128, 64, True], "tactic": 2}


def write_profile(tmp_path, **updates):
    profile = {"schema_version": 1, **IDENTITY, "rows": [ROW], **updates}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    return path


def test_matching_profile_keeps_exact_tactic(tmp_path):
    choices = read_shape_tactics(write_profile(tmp_path), backend_evidence=IDENTITY)
    assert choices == {(8, 41, 128, 64, True): 2}


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend_id", "pir2_cutlass_fp8"),
        ("backend_revision", "other"),
        ("device_arch", "sm_110"),
        ("quant_format", "fp8-e4m3-per-tensor-v1"),
    ],
)
def test_profile_rejects_different_backend_identity(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        read_shape_tactics(
            write_profile(tmp_path), backend_evidence={**IDENTITY, field: value}
        )


def test_profile_requires_known_runtime_identity(tmp_path):
    path = write_profile(tmp_path)
    with pytest.raises(ValueError, match="runtime backend evidence"):
        read_shape_tactics(path)
    with pytest.raises(ValueError, match="backend_revision"):
        read_shape_tactics(
            path, backend_evidence={**IDENTITY, "backend_revision": "unknown"}
        )


@pytest.mark.parametrize("version", [True, 0, 2, "1"])
def test_profile_schema_version_is_strict(tmp_path, version):
    with pytest.raises(ValueError, match="schema_version"):
        read_shape_tactics(
            write_profile(tmp_path, schema_version=version), backend_evidence=IDENTITY
        )


def test_profile_rejects_other_integer_format_rows(tmp_path):
    path = write_profile(tmp_path, rows=[{**ROW, "bits": 4}])
    with pytest.raises(ValueError, match="bits do not match"):
        read_shape_tactics(path, backend_evidence=IDENTITY)


def test_legacy_lists_preserve_mixed_bits_and_tactics(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps([ROW, {**ROW, "bits": 4, "tactic": 7}]))
    assert read_shape_tactics(path) == {
        (8, 41, 128, 64, True): 2,
        (4, 41, 128, 64, True): 7,
    }


def test_invalid_profile_does_not_mutate_site_or_group_tactics(tmp_path):
    single = SimpleNamespace(tactic=6)
    group = SimpleNamespace(linear=SimpleNamespace(tactic=5))
    controller = SimpleNamespace(
        quant={8: {"q": single}, 4: {}}, grouped={8: [(["q"], group)], 4: []}
    )
    config = SimpleNamespace(
        shape_tactics=write_profile(tmp_path), tactics=None, group_tactics=None
    )
    with pytest.raises(ValueError, match="device_arch"):
        apply_tactics(
            controller,
            SimpleNamespace(quant={8: {}, 4: {}}),
            config,
            {},
            backend_evidence={**IDENTITY, "device_arch": "sm_110"},
        )
    assert single.tactic == 6
    assert group.linear.tactic == 5
