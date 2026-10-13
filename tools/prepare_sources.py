"""Verify vendored sources or fetch pinned repositories; never reset local work."""

import argparse
import hashlib
import json
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(path, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(path), *args], check=check, capture_output=True, text=True
    )


def _owned_path(root, relative):
    """Keep manifest and source paths inside the declared source directory."""
    relative = Path(relative)
    path = root / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not path.resolve().is_relative_to(root.resolve())
        or path.is_symlink()
    ):
        raise RuntimeError(f"Vendored source path escapes its owned tree: {relative}")
    if not path.is_file():
        raise RuntimeError(f"Vendored source is missing: {relative}; left untouched")
    return path


def _verify_file(root, relative, expected):
    path = _owned_path(root, relative)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Vendored source hash differs: {relative}; left untouched")
    return path


def prepare_vendored(spec):
    """Verify owned files and resolve headers without model imports or CUDA work."""
    target = ROOT / spec["path"]
    if (
        not target.is_dir()
        or target.is_symlink()
        or not target.resolve().is_relative_to(ROOT.resolve())
    ):
        raise RuntimeError(
            "Vendored source directory is missing or nonlocal; left untouched"
        )
    manifest_path = _owned_path(ROOT, spec["manifest"])
    if not manifest_path.resolve().is_relative_to(target.resolve()):
        raise RuntimeError("Vendored manifest must reside inside its owned source tree")
    manifest = json.loads(manifest_path.read_text())
    upstream = manifest.get("upstream", {})
    if upstream.get("revision") != spec["revision"] or str(
        upstream.get("url", "")
    ).removesuffix(".git") != spec["url"].removesuffix(".git"):
        raise RuntimeError("Vendored manifest origin differs from sources.lock.json")
    files = manifest.get("files")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or not isinstance(files, dict)
        or not files
    ):
        raise RuntimeError("Invalid owned source manifest")
    for relative, record in files.items():
        _verify_file(target, relative, record["sha256"])
    count = len(files)
    for submodule in manifest.get("submodules", []):
        path = target / submodule["path"]
        if not path.resolve().is_relative_to(target.resolve()):
            raise RuntimeError("Submodule path escapes its owned source directory")
        resolve_cutlass()
    print(f"Verified {count} owned source files in {target}; no checkout or build")


def resolve_cutlass():
    """Resolve compatible headers without requiring the default Git revision."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from coexecution.kernels.sources import require_submodule

    dependency = require_submodule("cutlass")
    print(
        f"CUTLASS source: {dependency['directory']} "
        f"(revision: {dependency.get('revision', 'unknown')})"
    )
    return dependency


def prepare_submodule(spec):
    if os.environ.get("CUTLASS_ROOT"):
        resolve_cutlass()
        return
    target = ROOT / spec["path"]
    if target.is_symlink() or not target.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError("Submodule path escapes the repository")
    entry = git(ROOT, "ls-files", "--stage", "--", spec["path"]).stdout.strip()
    metadata, separator, name = entry.partition("\t")
    if (
        not separator
        or name != spec["path"]
        or metadata.split() != ["160000", spec["revision"], "0"]
    ):
        raise RuntimeError("Submodule gitlink differs from sources.lock.json")
    if not (target / ".git").exists():
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive", "--", spec["path"]],
            cwd=ROOT,
            check=True,
        )
    resolve_cutlass()
    print(f"Prepared CUTLASS dependency at {spec['path']}")


def prepare(spec):
    mode = spec.get("mode", "git")
    if mode == "vendored":
        prepare_vendored(spec)
        return
    if mode == "submodule":
        prepare_submodule(spec)
        return
    if mode != "git":
        raise ValueError(f"Unknown source preparation mode: {mode}")
    target = ROOT / spec["path"]
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", spec["url"], str(target)], check=True
        )
        git(target, "checkout", "--detach", spec["revision"])
    if git(target, "rev-parse", "HEAD").stdout.strip() != spec["revision"]:
        raise RuntimeError(f"Unexpected source revision in {target}; left untouched")
    if "submodule" in spec:
        module = target / spec["submodule"]
        if not (module / ".git").exists():
            git(target, "submodule", "update", "--init", "--", spec["submodule"])
        target = target / spec["submodule"]
        if (
            git(target, "rev-parse", "HEAD").stdout.strip()
            != spec["submodule_revision"]
        ):
            raise RuntimeError(f"Unexpected submodule revision in {target}")
    for patch in spec["patches"]:
        path = ROOT / patch
        if (
            git(
                target, "apply", "--reverse", "--check", str(path), check=False
            ).returncode
            == 0
        ):
            print(f"Already applied: {patch}")
        else:
            git(target, "apply", "--check", str(path))
            git(target, "apply", str(path))
            print(f"Applied: {patch}")
    print(f"Verified {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    specs = json.loads((ROOT / "sources.lock.json").read_text())["sources"]
    parser.add_argument("--component", choices=sorted(specs), required=True)
    args = parser.parse_args()
    prepare(specs[args.component])


if __name__ == "__main__":
    main()
