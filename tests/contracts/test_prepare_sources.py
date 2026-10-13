"""Vendored preparation verifies owned bytes without Git, CUDA, or mutation."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import prepare_sources


def digest(content):
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def vendored(tmp_path, monkeypatch):
    root = tmp_path / "coexecution/kernels"
    vendor = root / "vendor/example"
    vendor.mkdir(parents=True)
    (root / "kernel.cu").write_bytes(b"owned CUDA source")
    (vendor / "header.hpp").write_bytes(b"owned header")
    vendor_manifest = {
        "revision": "vendor-pin",
        "files": {"header.hpp": digest(b"owned header")},
    }
    metadata = vendor / "PROVENANCE.json"
    metadata.write_text(json.dumps(vendor_manifest))
    manifest = {
        "schema_version": 1,
        "upstream": {"url": "https://example.com/source", "revision": "pin"},
        "files": {"kernel.cu": {"sha256": digest(b"owned CUDA source")}},
        "vendors": [
            {
                "path": "vendor/example",
                "manifest": "vendor/example/PROVENANCE.json",
                "revision": "vendor-pin",
                "sha256": digest(metadata.read_bytes()),
            }
        ],
    }
    manifest_path = root / "SOURCE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))
    spec = {
        "mode": "vendored",
        "path": "coexecution/kernels",
        "manifest": "coexecution/kernels/SOURCE_MANIFEST.json",
        "url": "https://example.com/source.git",
        "revision": "pin",
        "patches": [],
    }
    monkeypatch.setattr(prepare_sources, "ROOT", tmp_path)
    monkeypatch.setattr(
        prepare_sources.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "Vendored source preparation cannot run Git"
        ),
    )
    return root, manifest_path, manifest, spec


def test_vendored_preparation_is_read_only(vendored, capsys):
    root, _, _, spec = vendored
    before = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    prepare_sources.prepare(spec)
    after = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    assert before == after
    assert "Verified 3 owned source files" in capsys.readouterr().out


@pytest.mark.parametrize("relative", ["kernel.cu", "vendor/example/header.hpp"])
def test_modified_source_is_never_overwritten(vendored, relative):
    root, _, _, spec = vendored
    path = root / relative
    path.write_bytes(b"local work to preserve")
    with pytest.raises(RuntimeError, match="hash differs.*left untouched"):
        prepare_sources.prepare(spec)
    assert path.read_bytes() == b"local work to preserve"


def test_missing_vendored_tree_is_not_cloned_or_created(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_sources, "ROOT", tmp_path)
    monkeypatch.setattr(
        prepare_sources.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "Missing vendored sources cannot be cloned"
        ),
    )
    with pytest.raises(RuntimeError, match="missing or nonlocal"):
        prepare_sources.prepare({"mode": "vendored", "path": "absent"})
    assert not (tmp_path / "absent").exists()


def test_changed_vendor_manifest_cannot_redefine_expected_hashes(vendored):
    root, _, _, spec = vendored
    path = root / "vendor/example/PROVENANCE.json"
    value = json.loads(path.read_text())
    value["files"]["header.hpp"] = digest(b"tampered header")
    path.write_text(json.dumps(value))
    (root / "vendor/example/header.hpp").write_bytes(b"tampered header")
    with pytest.raises(RuntimeError, match="PROVENANCE.json"):
        prepare_sources.prepare(spec)


def test_manifest_cannot_verify_a_file_outside_its_owned_tree(vendored):
    root, manifest_path, manifest, spec = vendored
    external = root.parent / "external.cu"
    external.write_bytes(b"external bytes")
    manifest["files"] = {"../external.cu": {"sha256": digest(external.read_bytes())}}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="escapes"):
        prepare_sources.prepare(spec)


def test_manifest_cannot_use_an_external_symlink(vendored):
    root, _, _, spec = vendored
    external = root.parent / "external.cu"
    external.write_bytes(b"owned CUDA source")
    (root / "kernel.cu").unlink()
    (root / "kernel.cu").symlink_to(external)
    with pytest.raises(RuntimeError, match="escapes"):
        prepare_sources.prepare(spec)


@pytest.mark.parametrize("field", ["url", "revision"])
def test_manifest_origin_must_match_the_lock(vendored, field):
    _, manifest_path, manifest, spec = vendored
    manifest["upstream"][field] = "different"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="origin differs"):
        prepare_sources.prepare(spec)


def test_existing_git_source_with_wrong_revision_is_never_reset(tmp_path, monkeypatch):
    source = tmp_path / "upstream"
    source.mkdir()
    monkeypatch.setattr(prepare_sources, "ROOT", tmp_path)
    calls = []

    def git(path, *args, **kwargs):
        calls.append((Path(path), args))
        return SimpleNamespace(stdout="local-revision\n")

    monkeypatch.setattr(prepare_sources, "git", git)
    with pytest.raises(
        RuntimeError, match="Unexpected source revision.*left untouched"
    ):
        prepare_sources.prepare({"path": "upstream", "revision": "pin"})
    assert calls == [(source, ("rev-parse", "HEAD"))]
