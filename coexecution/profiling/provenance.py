"""Freeze local source and checkpoint evidence without importing CUDA libraries."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def run_text(command):
    return subprocess.run(
        command, text=True, capture_output=True, check=True, timeout=30
    ).stdout.strip()


def tracked_files(work):
    output = subprocess.check_output(["git", "-C", str(work), "ls-files", "-z"])
    return [work / name.decode() for name in output.split(b"\0") if name]


def file_identity(path):
    stat = Path(path).stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def freeze_checkpoint(checkpoint):
    checkpoint = Path(checkpoint).absolute()
    index = checkpoint / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(read_json(index)["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
        index = checkpoint / "model.safetensors"
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in shards):
        raise ValueError("Checkpoint shard paths must stay within the checkpoint")
    if not shards:
        raise RuntimeError("Checkpoint weight map is empty")
    paths = {index, *(checkpoint / name for name in shards)}
    for name in (
        "config.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
    ):
        paths.add(checkpoint / name)
    paths.update((checkpoint / "experiment_cfg").glob("*.json"))
    paths.update((checkpoint / "experiment_cfg").glob("*.yaml"))
    files = {}
    for path in sorted(paths):
        before = file_identity(path)
        value = digest(path)
        if file_identity(path) != before:
            raise RuntimeError(f"Checkpoint changed while hashing: {path}")
        files[str(path)] = {"sha256": value, "identity": before}
    return {
        "path": str(checkpoint),
        "files": files,
        "verification": "Full SHA256 at prepare and load/check/run entry; size, mtime, ctime, inode, device checked at every source guard and stage boundary.",
    }


def verify_checkpoint(manifest, full=False):
    for name, expected in manifest["files"].items():
        if not Path(name).is_file() or file_identity(name) != expected["identity"]:
            raise RuntimeError(f"Frozen checkpoint identity changed: {name}")
        if full and digest(name) != expected["sha256"]:
            raise RuntimeError(f"Frozen checkpoint SHA256 changed: {name}")


def verify_sources(plan):
    mismatches = []
    for name, expected in plan["source_files"].items():
        path = Path(name)
        if not path.is_file() or digest(path) != expected:
            mismatches.append(name)
    if mismatches:
        raise RuntimeError(f"Frozen source changed: {mismatches}")
    actual_sources = [
        str(path)
        for path in source_paths(
            Path(plan["worktree"]),
            Path(plan["groot_root"]),
            [Path(path) for path in plan.get("extra_source_roots", [])],
        )
    ]
    if actual_sources != sorted(plan["source_files"]):
        raise RuntimeError("Frozen source inventory changed")
    work = Path(plan["worktree"])
    actual_commit = run_text(["git", "-C", str(work), "rev-parse", "HEAD"])
    if actual_commit != plan["source_commit"]:
        raise RuntimeError("Worktree HEAD differs from frozen plan")
    actual_tracked = sorted(str(path) for path in tracked_files(work) if path.is_file())
    if actual_tracked != plan["tracked_paths"]:
        raise RuntimeError("Worktree tracked source inventory changed")


SOURCE_SUFFIXES = {".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".c", ".cc", ".inl"}


def runtime_files(base):
    """Include actual runtime code, including local patches and new source files."""
    if not base.is_dir():
        raise FileNotFoundError(base)
    return {
        path.absolute()
        for path in base.rglob("*")
        if path.is_file()
        and path.suffix in SOURCE_SUFFIXES
        and not {".git", "__pycache__"}.intersection(path.relative_to(base).parts)
    }


def source_paths(work, groot_root, extra_roots=()):
    paths = {path.absolute() for path in tracked_files(work) if path.is_file()}
    # The Git manifest alone misses untracked additions which Python may import.
    paths.update(runtime_files(work / "coexecution"))
    paths.update(runtime_files(groot_root / "gr00t"))
    for base in extra_roots:
        paths.update(runtime_files(base))
    return sorted(paths)
