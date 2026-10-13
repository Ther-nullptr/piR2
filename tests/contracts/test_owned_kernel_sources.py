"""Owned-source contracts; no tensors, CUDA compilation, or device execution."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from coexecution.kernels import build, sources
from coexecution.kernels.blackwell import extension_loader, fp4_linear, fp8_linear


def test_source_metadata_and_floating_loaders_import_without_gpu_libraries():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import coexecution.kernels.sources; "
            "import coexecution.kernels.build; "
            "import coexecution.kernels.blackwell.fp8_linear; "
            "import coexecution.kernels.blackwell.fp4_linear; "
            "assert not any(name.startswith(('torch', 'triton', 'robotics_kernels')) "
            "for name in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend", build.BACKENDS)
def test_all_native_build_inputs_are_owned(backend, monkeypatch):
    monkeypatch.setenv("ROBOTICS_CUTLASS_FP8_SO", "/unused/external-fp8.so")
    monkeypatch.setenv("ROBOTICS_CUTLASS_FP4_SO", "/unused/external-fp4.so")
    inputs = build.build_inputs(backend)
    paths = inputs["sources"] + inputs["headers"] + inputs["include_paths"]
    assert paths
    assert all(Path(path).resolve().is_relative_to(build.ROOT) for path in paths)
    assert inputs["namespace"].startswith("pir2_")
    if inputs["cutlass_revision"]:
        assert inputs["cutlass_revision"] == "982748aa7356fa838c2ea4994ddcb0b2a4b4cefa"


def test_build_rejects_source_symlink_escape(tmp_path, monkeypatch):
    owned = tmp_path / "owned"
    owned.mkdir()
    external = tmp_path / "external.cu"
    external.write_text("unowned source")
    (owned / "source.cu").symlink_to(external)
    monkeypatch.setattr(build, "ROOT", owned)
    monkeypatch.setitem(
        build.BACKENDS,
        "fixture",
        {"sources": ("source.cu",), "namespace": "pir2_fixture", "cutlass": False},
    )
    with pytest.raises(ValueError, match="escapes"):
        build.build_inputs("fixture")


def test_repository_manifest_accounts_for_the_current_owned_tree():
    identity = sources.source_identity(build)
    assert identity["kind"] == "repository-owned"
    assert identity["dirty"] is False
    assert identity["revision"] == identity["expected_revision"]
    assert identity["module_tracked"] is True
    assert identity["module_sha256"] != "unknown"


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    (tmp_path / "kernel.cu").write_bytes(b"original CUDA bytes")
    digest = hashlib.sha256(b"original CUDA bytes").hexdigest()
    manifest = {
        "upstream": {"url": "https://example.com/source", "revision": "upstream"},
        "files": {
            "kernel.cu": {
                "origin": "upstream",
                "origin_sha256": digest,
                "sha256": digest,
            }
        },
        "vendors": [],
    }
    (tmp_path / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(sources, "ROOT", tmp_path)
    monkeypatch.setattr(sources, "MANIFEST", tmp_path / "SOURCE_MANIFEST.json")
    return tmp_path, digest


def test_source_change_cannot_keep_a_clean_upstream_identity(source_tree):
    root, original = source_tree
    (root / "kernel.cu").write_bytes(b"adapted CUDA bytes")
    changed = sources.source_identity()
    assert changed["dirty"] is True
    assert changed["changed_sources"] == ["kernel.cu"]
    assert changed["expected_revision"] != changed["revision"]
    sources.refresh_manifest()
    clean = sources.source_identity()
    assert clean["dirty"] is False
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    assert manifest["files"]["kernel.cu"]["origin_sha256"] == original
    assert manifest["files"]["kernel.cu"]["sha256"] != original


def test_missing_or_unrecorded_sources_are_reported(source_tree):
    root, _ = source_tree
    (root / "kernel.cu").unlink()
    (root / "new_kernel.cu").write_text("new source")
    identity = sources.source_identity()
    assert identity["dirty"] is True
    assert identity["missing_sources"] == ["kernel.cu"]
    assert identity["unrecorded_sources"] == ["new_kernel.cu"]


def test_unrecorded_local_headers_change_source_identity(source_tree):
    root, _ = source_tree
    (root / "new_header.hpp").write_text("a new compiler input")
    identity = sources.source_identity()
    assert identity["dirty"] is True
    assert identity["unrecorded_sources"] == ["new_header.hpp"]


def vendor_tree(root):
    directory = root / "vendor/library"
    directory.mkdir(parents=True)
    (directory / "kernel.h").write_text("pinned header")
    hashes = {"kernel.h": hashlib.sha256(b"pinned header").hexdigest()}
    metadata = directory / "PROVENANCE.json"
    metadata.write_text(json.dumps({"files": hashes}))
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    manifest["vendors"] = [
        {
            "path": "vendor/library",
            "manifest": "vendor/library/PROVENANCE.json",
            "sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        }
    ]
    (root / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest))
    return directory, metadata


def test_unrecorded_vendor_headers_cannot_keep_a_clean_identity(source_tree):
    root, _ = source_tree
    directory, _ = vendor_tree(root)
    assert sources.source_identity()["dirty"] is False
    (directory / "new.h").write_text("unreviewed include")
    identity = sources.source_identity()
    assert identity["dirty"] is True
    assert identity["unrecorded_sources"] == ["vendor/library/new.h"]


def test_vendor_manifest_is_checked_before_external_read(source_tree, tmp_path):
    root, _ = source_tree
    _, metadata = vendor_tree(root)
    external = tmp_path.parent / (tmp_path.name + "-external.json")
    external.write_text("not even JSON; must never be read")
    metadata.unlink()
    metadata.symlink_to(external)
    with pytest.raises(ValueError, match="escapes"):
        sources.source_identity()


@pytest.mark.parametrize("target", ["manifest", "header"])
def test_refresh_cannot_bless_changed_vendor_inputs(source_tree, target):
    root, _ = source_tree
    directory, metadata = vendor_tree(root)
    changed = metadata if target == "manifest" else directory / "kernel.h"
    changed.write_bytes(changed.read_bytes() + b" ")
    original_manifest = (root / "SOURCE_MANIFEST.json").read_bytes()
    with pytest.raises(ValueError, match="Vendored"):
        sources.refresh_manifest()
    assert (root / "SOURCE_MANIFEST.json").read_bytes() == original_manifest


def test_external_or_unknown_modules_cannot_claim_owned_revision(source_tree, tmp_path):
    root, _ = source_tree
    external = tmp_path.parent / (tmp_path.name + "-external.py")
    external.write_text("external module")
    external_identity = sources.source_identity(SimpleNamespace(__file__=external))
    assert external_identity["revision"] == "unknown"
    assert external_identity["dirty"] == "unknown"
    assert (
        external_identity["module_sha256"]
        == hashlib.sha256(external.read_bytes()).hexdigest()
    )
    missing_identity = sources.source_identity(SimpleNamespace())
    assert missing_identity["revision"] == "unknown"
    assert sources.source_identity()["revision"] != "unknown"


def test_dirty_sources_cannot_be_built(monkeypatch):
    monkeypatch.setattr(
        sources,
        "source_identity",
        lambda **kwargs: {
            "source_sha256": "a" * 64,
            "dirty": True,
            "missing_sources": [],
        },
    )
    with pytest.raises(RuntimeError, match="manifest"):
        build.build_extension("integer")


def test_stale_registered_namespace_is_rejected_before_compiler_import(monkeypatch):
    monkeypatch.setattr(
        sources,
        "source_identity",
        lambda **kwargs: {
            "source_sha256": "a" * 64,
            "dirty": False,
            "missing_sources": [],
        },
    )
    namespace = SimpleNamespace(
        gemm=lambda: None,
        gemm_biasless=lambda: None,
        source_identity=lambda: "outdated-source-digest",
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_initialized=lambda: False),
            ops=SimpleNamespace(pir2_integer=namespace),
        ),
    )
    with pytest.raises(RuntimeError, match="source identity.*restart"):
        build.build_extension("integer")


@pytest.mark.parametrize("module", [fp8_linear, fp4_linear])
def test_wrong_device_never_triggers_floating_compilation(monkeypatch, module):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_capability=lambda device: (8, 9),
            )
        ),
    )
    monkeypatch.setattr(
        extension_loader,
        "build_extension",
        lambda backend: pytest.fail("Unsupported device cannot compile"),
    )
    with pytest.raises(RuntimeError, match="SM110"):
        extension_loader.require_backend(
            "fp8" if module is fp8_linear else "fp4",
            module.OP_NAMESPACE,
            module.REQUIRED_OPS,
        )


@pytest.mark.parametrize("module", [fp8_linear, fp4_linear])
def test_floating_loaders_build_owned_sources_despite_external_so_settings(
    monkeypatch, module
):
    backend = "fp8" if module is fp8_linear else "fp4"
    calls = []
    namespace = SimpleNamespace()
    monkeypatch.setenv("ROBOTICS_CUTLASS_FP8_SO", "/unused/external-fp8.so")
    monkeypatch.setenv("ROBOTICS_CUTLASS_FP4_SO", "/unused/external-fp4.so")
    monkeypatch.setenv(f"PIR2_{backend.upper()}_DECODE_POLICY", "balanced")
    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda device: (11, 0),
        ),
        ops=SimpleNamespace(**{module.OP_NAMESPACE: namespace}),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)

    def build_owned(selected):
        calls.append(selected)
        for operator in module.REQUIRED_OPS:
            setattr(namespace, operator, lambda *args: None)
        namespace.supports_device = lambda cc: cc == 110

    monkeypatch.setattr(extension_loader, "build_extension", build_owned)
    result = getattr(module, f"require_{backend}_backend")()
    assert result is namespace
    assert calls == [backend]
