"""GPU-free provenance and fingerprints for the owned operator source tree.

The manifest distinguishes original upstream bytes from adapted repository
bytes. CUTLASS retains its own unmodified per-file manifest. Source identity is
not binary identity or evidence that a particular kernel ran successfully.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "SOURCE_MANIFEST.json"
SOURCE_SUFFIXES = {
    ".py",
    ".cu",
    ".cuh",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".inl",
}
SOURCE_NOTICES = {
    "UPSTREAM_NOTICE.txt",
    "blackwell/LICENSE",
    "blackwell/UPSTREAM_PROVENANCE.json",
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_digest(files):
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _owned_path(name):
    path = ROOT / name
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"Kernel manifest input escapes owned source tree: {name}")
    return path


def _expected_sources():
    manifest = json.loads(_owned_path(MANIFEST).read_text())
    expected = {name: entry["sha256"] for name, entry in manifest["files"].items()}
    vendor_paths = []
    for vendor in manifest.get("vendors", []):
        directory = _owned_path(vendor["path"])
        metadata = _owned_path(vendor["manifest"])
        if not metadata.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Vendor manifest escapes its declared vendor directory")
        vendor_paths.append(directory)
        expected[vendor["manifest"]] = vendor["sha256"]
        if not metadata.is_file():
            continue
        provenance = json.loads(metadata.read_text())
        for name, digest in provenance["files"].items():
            relative = f"{vendor['path']}/{name}"
            path = _owned_path(relative)
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError(
                    f"Vendor source escapes its declared directory: {name}"
                )
            if relative in expected:
                raise ValueError(f"Duplicate kernel manifest source: {relative}")
            expected[relative] = digest
    return manifest, expected, vendor_paths


def _local_source_paths(root):
    paths = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            _owned_path(path)
        if (
            path.is_file()
            and (
                path.suffix in SOURCE_SUFFIXES or relative.as_posix() in SOURCE_NOTICES
            )
            and "vendor" not in relative.parts
        ):
            paths.append(path)
    return sorted(paths)


def _vendor_source_paths(directories):
    paths = []
    for directory in directories:
        for path in directory.rglob("*"):
            _owned_path(path)
            if path.is_file():
                paths.append(path)
    return paths


def source_identity(module=None, *, backend=None):
    """Hash declared local and vendored inputs without importing GPU libraries.

    ``revision`` and ``expected_revision`` identify the complete owned tree, while
    ``upstream_revision`` describes the source snapshot before local adaptations.
    ``dirty`` means bytes differ from this manifest, independently of Git status.
    An external, missing or unrecorded module never inherits the owned revision.
    """
    manifest, expected, vendor_paths = _expected_sources()
    observed = {}
    missing = []
    changed = []
    for name, digest in sorted(expected.items()):
        path = _owned_path(name)
        try:
            observed[name] = _sha256(path)
        except FileNotFoundError:
            observed[name] = "missing"
            missing.append(name)
        if observed[name] != digest:
            changed.append(name)
    unrecorded = []
    for path in [*_local_source_paths(ROOT), *_vendor_source_paths(vendor_paths)]:
        relative = path.relative_to(ROOT).as_posix()
        if relative not in expected:
            unrecorded.append(relative)
            observed[relative] = _sha256(path)
    actual = _tree_digest(observed)
    result = {
        "kind": "repository-owned",
        "expected_revision": "sha256:" + _tree_digest(expected),
        "revision": "sha256:" + actual,
        "dirty": bool(changed or unrecorded),
        "source_sha256": actual,
        "manifest_sha256": _sha256(MANIFEST),
        "url": manifest["upstream"]["url"],
        "upstream_revision": manifest["upstream"]["revision"],
        "module_tracked": False,
        "module_sha256": "unknown",
        "missing_sources": missing,
        "changed_sources": changed,
        "unrecorded_sources": unrecorded,
    }
    if backend is not None:
        from coexecution.kernels.build import build_inputs

        inputs = build_inputs(backend)
        result["backend"] = backend
        result["namespace"] = inputs["namespace"]
        result["cutlass_revision"] = inputs["cutlass_revision"]
    if module is None:
        return result
    filename = getattr(module, "__file__", None)
    result["revision"] = "unknown"
    result["dirty"] = "unknown"
    if filename is not None:
        path = Path(filename).resolve()
        try:
            result["module_sha256"] = _sha256(path)
        except OSError:
            pass
        if path.is_relative_to(ROOT.resolve()):
            relative = path.relative_to(ROOT.resolve()).as_posix()
            result["module_path"] = "coexecution/kernels/" + relative
            result["module_tracked"] = relative in expected
            if relative in expected and relative not in missing:
                result["revision"] = "sha256:" + actual
                result["dirty"] = bool(changed or unrecorded)
    return result


def verified_source_identity(*, backend=None):
    """Reject changed/unrecorded build inputs before loading a shared library."""
    identity = source_identity(backend=backend)
    if identity["missing_sources"]:
        raise FileNotFoundError("Owned kernel manifest has missing source files")
    if identity["dirty"]:
        raise RuntimeError(
            "Owned kernel source manifest differs; review and refresh local changes "
            "before building"
        )
    return identity


def refresh_manifest(*, origins=None):
    """Refresh adapted hashes after a reviewed edit, preserving original hashes.

    Callers importing another source must provide its provenance in ``origins``;
    new files without an origin record are identified as piR2 project code. This
    maintenance function performs no build, model import, or device inspection.
    """
    manifest, expected, vendor_paths = _expected_sources()
    vendor_names = {
        path.relative_to(ROOT).as_posix() for path in _vendor_source_paths(vendor_paths)
    }
    for vendor in manifest.get("vendors", []):
        prefix = vendor["path"].rstrip("/") + "/"
        declared = {name for name in expected if name.startswith(prefix)}
        if {name for name in vendor_names if name.startswith(prefix)} != declared:
            raise ValueError(
                "Vendored source inventory changed; update provenance explicitly"
            )
        for name in declared:
            if (
                not _owned_path(name).is_file()
                or _sha256(_owned_path(name)) != expected[name]
            ):
                raise ValueError(
                    "Vendored inputs changed; update provenance explicitly"
                )
    existing = manifest["files"]
    files = {}
    for path in _local_source_paths(ROOT):
        name = path.relative_to(ROOT).as_posix()
        entry = dict(existing.get(name, {"origin": "pir2-project"}))
        entry.update((origins or {}).get(name, {}))
        entry["sha256"] = _sha256(path)
        files[name] = entry
    manifest["files"] = files
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.refresh:
        refresh_manifest()
    print(json.dumps(source_identity(), indent=2))


if __name__ == "__main__":
    main()
