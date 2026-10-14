"""Public runtime identity contracts; no model or GPU execution."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from coexecution.libero import runtime_receipt as receipts
from coexecution.optimization.config import OptimizationConfig


def fixture(monkeypatch, tmp_path):
    source = {
        "manifest_sha256": "a" * 64,
        "files": {"coexecution/test.py": "b" * 64},
        "loaded_modules": {},
        "git_commit": "c" * 40,
    }
    monkeypatch.setattr(receipts, "source_manifest", lambda root=None: deepcopy(source))
    evidence = {
        "config": asdict(OptimizationConfig(precision="w8a8", fusion=True)),
        "operator_selection": {
            "architecture": "sm_89",
            "precision": "w8a8",
            "implementations": {"layernorm_pack": "native"},
            "parameters": {},
        },
        "coverage": {
            "linears": [f"backbone.layer.{i}" for i in range(432)],
            "operator_adapters": {
                "norm": {
                    "enabled": True,
                    "source_path": "/private/hidden",
                    "password": "secret",
                }
            },
        },
        "dit_graph": {"capture_count": 1, "replay_count": 12},
        "backend": {
            "backend_id": "pir2_integer",
            "runtime": {
                "device_name": "NVIDIA RTX 6000 Ada Generation",
                "path": "/private/model",
                "auth_token": "hf_private",
            },
        },
    }
    model = SimpleNamespace(
        config=SimpleNamespace(streaming=False, action_horizon=40),
        action_head=SimpleNamespace(num_inference_timesteps=4),
        device="cuda:1",
    )
    session = SimpleNamespace(
        evidence=lambda: deepcopy(evidence), policy=SimpleNamespace(model=model)
    )
    identity = "d" * 64
    binary = tmp_path / f"pir2_integer_{identity}_ext.so"
    binary.write_bytes(b"fake binary bytes; no code is loaded")
    torch = SimpleNamespace(
        __version__="test",
        version=SimpleNamespace(cuda="12.6"),
        ops=SimpleNamespace(
            loaded_libraries={str(binary)},
            pir2_integer=SimpleNamespace(source_identity=lambda: identity),
        ),
        cuda=SimpleNamespace(
            get_device_properties=lambda device: SimpleNamespace(
                name="NVIDIA test GPU", major=8, minor=9, uuid="private-device-uuid"
            )
        ),
    )
    return session, torch, evidence, binary


def test_receipt_import_never_imports_torch_or_starts_an_experiment():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import coexecution.libero.runtime_receipt; assert 'torch' not in sys.modules; assert 'gr00t' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_receipt_records_installed_profile_coverage_binary_without_private_paths(
    monkeypatch, tmp_path
):
    session, torch, evidence, binary = fixture(monkeypatch, tmp_path)
    evidence["config"]["operator_profile"] = {"sha256": "e" * 64}
    result = receipts.capture_runtime_receipt(
        session,
        variant="flow",
        measurement_boundary="prepared_model",
        torch_module=torch,
    )
    assert result["schema"] == "pir2.runtime.receipt" and result["version"] == 1
    assert result["coverage"]["linear_count"] == 432
    assert result["operator_profile_sha256"] == "e" * 64
    assert result["operator_profile"]["implementations"]["layernorm_pack"] == "native"
    library = result["binaries"]["libraries"][0]
    assert library["sha256"] == receipts.file_digest(binary)
    assert library["identity_verified"]
    assert result["device"]["compute_capability"] == [8, 9]
    assert result["device"]["clock_sample"] is None
    assert result["validation"]["status"] == "pending"
    serialized = json.dumps(result)
    assert str(tmp_path) not in serialized
    assert "/private" not in serialized and "hf_private" not in serialized
    assert "password" not in serialized and "private-device-uuid" not in serialized


def test_source_config_binary_changes_identity_but_replay_counter_does_not(
    monkeypatch, tmp_path
):
    session, torch, evidence, binary = fixture(monkeypatch, tmp_path)

    def capture():
        return receipts.capture_runtime_receipt(
            session, variant="flow", measurement_boundary="policy", torch_module=torch
        )

    first = capture()
    evidence["dit_graph"]["replay_count"] = 99
    assert capture()["identity_sha256"] == first["identity_sha256"]
    binary.write_bytes(b"changed loaded binary")
    assert capture()["identity_sha256"] != first["identity_sha256"]
    evidence["config"]["input_preparation"] = True
    assert capture()["configuration_sha256"] != first["configuration_sha256"]


def test_receipt_does_not_accept_foreign_gpu_validation_or_model_variant(
    monkeypatch, tmp_path
):
    session, torch, _, _ = fixture(monkeypatch, tmp_path)
    options = {
        "variant": "flow",
        "measurement_boundary": "model-only",
        "torch_module": torch,
    }
    initial = receipts.capture_runtime_receipt(session, **options)
    with pytest.raises(ValueError, match="exact runtime"):
        receipts.capture_runtime_receipt(
            session,
            **options,
            validation={
                "status": "gpu_verified",
                "evidence_sha256": "e" * 64,
                "runtime_identity_sha256": "f" * 64,
            },
        )
    verified = receipts.capture_runtime_receipt(
        session,
        **options,
        validation={
            "status": "gpu_verified",
            "evidence_sha256": "e" * 64,
            "runtime_identity_sha256": initial["identity_sha256"],
        },
    )
    assert verified["validation"]["status"] == "gpu_verified"
    with pytest.raises(ValueError, match="loaded model"):
        receipts.capture_runtime_receipt(
            session, variant="pir2", measurement_boundary="policy"
        )


def test_loaded_registration_and_binary_source_must_agree(monkeypatch, tmp_path):
    _, torch, _, _ = fixture(monkeypatch, tmp_path)
    torch.ops.pir2_integer.source_identity = lambda: "e" * 64
    with pytest.raises(RuntimeError, match="disagree"):
        receipts.loaded_binaries(torch)


def test_wrong_runtime_root_rejected_before_source_scan():
    with pytest.raises(ValueError, match="runtime root"):
        receipts.source_manifest(Path("/not/the/imported/runtime"))


def test_measurement_receipt_cli_is_explicit():
    from coexecution.libero import measurement

    parser = measurement.build_parser()
    required = ["--checkpoint", "unused", "--dataset", "unused", "--output", "unused"]
    assert parser.parse_args(required).runtime_receipt is False
    assert parser.parse_args([*required, "--runtime-receipt"]).runtime_receipt is True


@pytest.mark.parametrize("flag", [None, False])
def test_disabled_measurement_receipt_does_not_import_capture_dependencies(
    monkeypatch, flag
):
    import builtins

    from coexecution.libero import measurement

    original = {"calls": 2}
    args = SimpleNamespace(measurement_boundary="model-only")
    if flag is not None:
        args.runtime_receipt = flag
    import_module = builtins.__import__

    def reject_receipt_import(name, *args, **kwargs):
        if name in ("torch", "coexecution.libero.runtime_receipt"):
            raise AssertionError("Disabled receipt imported capture dependencies")
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_receipt_import)
    result = measurement._mode_evidence(
        SimpleNamespace(evidence=lambda: original), args
    )
    assert result == original
    assert "runtime_receipt" not in result


def test_mode_evidence_attaches_explicit_receipt_without_rewriting_session_snapshot(
    monkeypatch,
):
    from coexecution.libero import measurement

    original = {"calls": 2}
    session = SimpleNamespace(evidence=lambda: original)
    torch = object()
    monkeypatch.setitem(sys.modules, "torch", torch)
    calls = []
    monkeypatch.setattr(
        receipts,
        "capture_runtime_receipt",
        lambda value, **kwargs: (
            calls.append((value, kwargs)) or {"validation": {"status": "pending"}}
        ),
    )
    args = SimpleNamespace(measurement_boundary="model-only", runtime_receipt=True)
    evidence = measurement._mode_evidence(session, args)
    assert evidence["runtime_receipt"]["validation"]["status"] == "pending"
    assert original == {"calls": 2}
    assert calls == [
        (
            session,
            {
                "variant": "flow",
                "measurement_boundary": "model-only",
                "torch_module": torch,
            },
        )
    ]
