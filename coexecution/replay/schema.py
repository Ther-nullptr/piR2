"""Versioned, finite replay identities and time boundaries (standard library)."""

import hashlib
import json
import math
from pathlib import Path

VERSION = 1
PRECISIONS = frozenset(("bf16", "w8a8", "w4a4"))


def optional_number(value):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected a finite number or unknown (null)")
    return value


def interval(start, stop):
    start, stop = optional_number(start), optional_number(stop)
    if start is None or stop is None or stop < start:
        raise ValueError("Invalid timestamp order")
    return start, stop


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def comparison_key(
    *, kind, observation_sha256, seed, shared_config_sha256, clock_cohort
):
    if kind != "model" or any(
        value is None
        for value in (observation_sha256, seed, shared_config_sha256, clock_cohort)
    ):
        return None
    if (
        not observation_sha256
        or not shared_config_sha256
        or not clock_cohort
        or type(seed) is not int
    ):
        return None
    return digest([observation_sha256, seed, shared_config_sha256, clock_cohort])


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value {value} in {path}")

    return json.loads(Path(path).read_text(), parse_constant=invalid)


def read_jsonl(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value {value} in {path}")

    with Path(path).open() as stream:
        return [
            json.loads(line, parse_constant=invalid) for line in stream if line.strip()
        ]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")


def source(path):
    path = Path(path).resolve()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return {"path": str(path), "sha256": sha.hexdigest(), "bytes": path.stat().st_size}


def under_root(root, relative):
    """Explicit artifact names only; reject absolute paths, traversal and symlinks out."""
    root = Path(root).resolve()
    relative = Path(relative)
    path = (root / relative).resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not path.is_relative_to(root)
    ):
        raise ValueError("Replay artifact must remain under its declared root")
    return path
