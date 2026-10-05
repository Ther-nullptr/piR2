"""Stable checkpoint provenance shared by policy servers and evaluators."""

import hashlib
from pathlib import Path


def checkpoint_identity(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    files = sorted(checkpoint.glob("*.safetensors"))
    files += [
        checkpoint / name
        for name in [
            "config.json",
            "processor_config.json",
            "statistics.json",
            "embodiment_id.json",
        ]
        if (checkpoint / name).exists()
    ]
    hashes = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    return {"checkpoint": str(checkpoint), "sha256": hashes}
