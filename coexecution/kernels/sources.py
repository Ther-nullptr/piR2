"""GPU-free provenance and fingerprints for the owned operator source tree.

The manifest distinguishes original upstream bytes from adapted repository
bytes. CUTLASS is a pinned Git submodule; its checkout identity is recorded
without hashing every upstream header. Source identity is not binary identity
or evidence that a particular kernel ran successfully.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
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
    if any("vendor" in Path(name).parts for name in expected):
        raise ValueError(
            "Vendor dependencies belong in submodules, not owned file hashes"
        )
    return manifest, expected


def _submodule_spec(name, manifest):
    entries = [row for row in manifest.get("submodules", []) if row.get("name") == name]
    if len(entries) != 1:
        raise ValueError(f"Expected one pinned submodule declaration for {name}")
    spec = entries[0]
    relative = Path(spec["path"])
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[:1] != ("vendor",)
    ):
        raise ValueError(
            "Kernel submodules must have a repository-relative vendor path"
        )
    _owned_path(relative)
    if not isinstance(spec.get("revision"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", spec["revision"]
    ):
        raise ValueError("Kernel submodules require a full pinned Git commit")
    if not isinstance(spec.get("url"), str) or not spec["url"]:
        raise ValueError("Kernel submodules require an upstream URL")
    scope = spec.get("scope")
    if (
        not isinstance(scope, list)
        or not scope
        or any(
            not isinstance(item, str)
            or not item
            or Path(item).is_absolute()
            or ".." in Path(item).parts
            for item in scope
        )
    ):
        raise ValueError("Kernel submodule scope must contain relative source paths")
    if name == "cutlass" and not {"include", "tools/util/include"} <= set(scope):
        raise ValueError("CUTLASS scope must include its header directories")
    return spec


def _git(directory, *arguments):
    # Ignore repository-routing environment variables: the declared local
    # checkout, never a caller's unrelated GIT_DIR/worktree, supplies headers.
    routing = {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    }
    environment = {
        key: value for key, value in os.environ.items() if key not in routing
    }
    return subprocess.check_output(
        [
            "git",
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(directory),
            *arguments,
        ],
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=20,
        env=environment,
    ).strip()


def _dependency_directory(spec):
    override = os.environ.get("CUTLASS_ROOT") if spec["name"] == "cutlass" else None
    return (
        Path(override).expanduser().resolve() if override else _owned_path(spec["path"])
    )


def _header_fingerprint(directory):
    files = {}
    for scope in ("include", "tools/util/include"):
        visited = set()
        for current, children, names in os.walk(directory / scope, followlinks=True):
            real = Path(current).resolve()
            if real in visited:
                children[:] = []
                continue
            visited.add(real)
            children[:] = [
                name for name in children if name not in {".git", "__pycache__"}
            ]
            for name in names:
                path = Path(current) / name
                if path.is_file():
                    files[path.relative_to(directory).as_posix()] = _sha256(path)
    return _tree_digest(files)


def _submodule_identity(spec):
    directory = _dependency_directory(spec)
    result = {
        "name": spec["name"],
        "path": spec["path"],
        "url": spec["url"],
        "scope": list(spec["scope"]),
        "expected_revision": spec["revision"],
        "revision": "unknown",
        "initialized": False,
        "dirty": "unknown",
        "missing_paths": [],
        "error": None,
        "source": "environment"
        if spec["name"] == "cutlass" and os.environ.get("CUTLASS_ROOT")
        else "submodule",
        "header_sha256": None,
    }
    for name in ("include/cutlass", "include/cute", "tools/util/include"):
        if not (directory / name).is_dir():
            result["missing_paths"].append(name)
    if result["missing_paths"]:
        result["error"] = "is missing required CUTLASS header directories"
        return result
    try:
        actual_root = Path(_git(directory, "rev-parse", "--show-toplevel")).resolve()
        if actual_root == directory.resolve():
            result["initialized"] = True
            result["revision"] = _git(
                directory, "rev-parse", "--verify", "HEAD^{commit}"
            )
            result["dirty"] = bool(
                _git(
                    directory,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "--ignored=matching",
                    "--ignore-submodules=none",
                    "--",
                    "include",
                    "tools/util/include",
                )
            )
    except (OSError, subprocess.SubprocessError):
        pass
    # Clean Git uses its actual commit. Modified or headers-only trees need
    # content identity so a second local edit cannot reuse the first binary.
    if result["revision"] == "unknown" or result["dirty"] is not False:
        result["header_sha256"] = _header_fingerprint(directory)
    result["matches_recommended_revision"] = result["revision"] == spec["revision"]
    return result


def _submodule_error(record):
    command = (
        "git submodule update --init --recursive coexecution/kernels/" + record["path"]
    )
    return RuntimeError(
        f"Kernel submodule {record['name']} {record['error']}. "
        f"Set CUTLASS_ROOT to a compatible header root, or from the repository root run: {command}"
    )


def require_submodule(name="cutlass"):
    """Resolve CUTLASS_ROOT or the default submodule; never initialize/fetch it.

    ``path`` stays relative to the kernel tree. ``directory`` is the absolute
    checkout path for native include flags. The pin is a recommended default,
    not a version gate; actual API compatibility is determined by compilation.
    A headers-only root does not require Git metadata or a LICENSE.txt file.
    """
    manifest, _ = _expected_sources()
    record = _submodule_identity(_submodule_spec(name, manifest))
    if record["error"] is not None:
        raise _submodule_error(record)
    return {
        **record,
        "directory": str(_dependency_directory(_submodule_spec(name, manifest))),
    }


def _local_source_paths(root):
    paths = []
    for directory, children, filenames in os.walk(root, followlinks=False):
        # Do not descend into a submodule, including its .git routing file.
        children[:] = [
            name for name in children if name not in {"vendor", "__pycache__", ".git"}
        ]
        for name in children:
            path = Path(directory) / name
            if path.is_symlink():
                _owned_path(path)
        for name in filenames:
            path = Path(directory) / name
            relative = path.relative_to(root)
            if path.suffix in SOURCE_SUFFIXES or relative.as_posix() in SOURCE_NOTICES:
                _owned_path(path)
                if path.is_file():
                    paths.append(path)
    return sorted(paths)


def _submodule_digest(record, *, expected=False):
    return _tree_digest(
        {
            "path": record["path"],
            "url": record["url"],
            "scope": record["scope"],
            "revision": record["expected_revision"] if expected else record["revision"],
            "initialized": True if expected else record["initialized"],
            "dirty": False if expected else record["dirty"],
            "missing_paths": [] if expected else record["missing_paths"],
            "available": True if expected else record["error"] is None,
            "header_sha256": None if expected else record["header_sha256"],
        }
    )


def source_identity(module=None, *, backend=None):
    """Hash owned inputs and record pinned/observed submodule checkout identity.

    ``revision`` and ``expected_revision`` identify the complete owned tree, while
    ``upstream_revision`` describes the source snapshot before local adaptations.
    ``dirty`` includes changed own bytes and unavailable/modified dependencies.
    An external, missing or unrecorded module never inherits the owned revision.
    A backend without CUTLASS uses only owned inputs; BF16 builds therefore do
    not require the optional header submodule to be initialized.
    """
    manifest, expected = _expected_sources()
    spec = None
    if backend is not None:
        from coexecution.kernels.build import BACKENDS

        if backend not in BACKENDS:
            raise ValueError(f"Unknown owned kernel backend: {backend}")
        spec = BACKENDS[backend]
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
    for path in _local_source_paths(ROOT):
        relative = path.relative_to(ROOT).as_posix()
        if relative not in expected:
            unrecorded.append(relative)
            observed[relative] = _sha256(path)
    owned_dirty = bool(changed or unrecorded)
    submodules = []
    if spec is None or spec["cutlass"]:
        names = [row["name"] for row in manifest.get("submodules", [])]
        if spec is not None and "cutlass" not in names:
            raise ValueError("CUTLASS backend requires a pinned cutlass submodule")
        for name in names:
            record = _submodule_identity(_submodule_spec(name, manifest))
            submodules.append(record)
            key = "@submodule/" + name
            expected[key] = _submodule_digest(record, expected=True)
            observed[key] = _submodule_digest(record)
            if record["error"] is not None:
                changed.append(record["path"])
                if not record["initialized"] or record["missing_paths"]:
                    missing.append(record["path"])
    actual = _tree_digest(observed)
    result = {
        "kind": "repository-owned",
        "expected_revision": "sha256:" + _tree_digest(expected),
        "revision": "sha256:" + actual,
        "dirty": owned_dirty
        or any(
            record["dirty"] is not False or record["error"] is not None
            for record in submodules
        ),
        "owned_dirty": owned_dirty,
        "source_sha256": actual,
        "manifest_sha256": _sha256(MANIFEST),
        "url": manifest["upstream"]["url"],
        "upstream_revision": manifest["upstream"]["revision"],
        "module_tracked": False,
        "module_sha256": "unknown",
        "missing_sources": missing,
        "changed_sources": changed,
        "unrecorded_sources": unrecorded,
        "submodules": submodules,
    }
    if backend is not None:
        result["backend"] = backend
        result["namespace"] = spec["namespace"]
        result["cutlass_revision"] = next(
            (
                record["revision"]
                for record in submodules
                if record["name"] == "cutlass"
            ),
            None,
        )
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
                result["dirty"] = owned_dirty or any(
                    record["dirty"] is not False or record["error"] is not None
                    for record in submodules
                )
    return result


def verified_source_identity(*, backend=None):
    """Reject changed/unrecorded build inputs before loading a shared library."""
    identity = source_identity(backend=backend)
    for record in identity["submodules"]:
        if record["error"] is not None:
            raise _submodule_error(record)
    if identity["missing_sources"]:
        raise FileNotFoundError("Owned kernel manifest has missing source files")
    if identity["owned_dirty"]:
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
    # A source refresh is an own-code maintenance edit. It does not inspect,
    # initialize, reset or repin any dependency checkout.
    manifest, _ = _expected_sources()
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
