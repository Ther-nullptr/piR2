"""Integer tactic parsing and projection assignment with unchanged fallbacks."""

import json
import math
from pathlib import Path


def read_json(path, default):
    return default if path is None else json.loads(Path(path).read_text())


def checked_tactic(value):
    if type(value) is not int or value not in range(8):
        raise ValueError("Integer tactic must be an integer in [0, 7]")
    return value


def read_shape_tactics(path, *, backend_evidence=None):
    choices = {}
    rows = read_json(path, [])
    profile_bits = None
    if isinstance(rows, dict):
        profile = rows
        fields = ("backend_id", "backend_revision", "device_arch", "quant_format")
        if (
            type(profile.get("schema_version")) is not int
            or profile["schema_version"] != 1
        ):
            raise ValueError("Shape tactic profile schema_version must be 1")
        if any(
            not isinstance(profile.get(key), str) or not profile[key] for key in fields
        ):
            raise ValueError(
                "Shape tactic profile requires backend/revision/architecture/format metadata"
            )
        if backend_evidence is None:
            raise ValueError("Shape tactic profiles require runtime backend evidence")
        for key in fields:
            actual = backend_evidence.get(key)
            if actual in (None, "unknown") or profile[key] != actual:
                raise ValueError(
                    f"Shape tactic profile {key} does not match the runtime backend"
                )
        if profile["backend_id"] != "pir2_integer":
            raise ValueError("Integer shape tactics require the pir2_integer backend")
        formats = {"w8a8-symmetric-row-v1": 8, "w4a4-symmetric-row-v1": 4}
        if profile["quant_format"] not in formats:
            raise ValueError("Unsupported integer shape tactic quant_format")
        profile_bits = formats[profile["quant_format"]]
        rows = profile.get("rows")
    if not isinstance(rows, list):
        raise TypeError("Shape tactics must contain a JSON row list")
    for row in rows:
        bits, shape = row["bits"], row["shape"]
        if (
            type(bits) is not int
            or bits not in (4, 8)
            or not isinstance(shape, list)
            or len(shape) != 4
            or any(type(size) is not int or size <= 0 for size in shape[:3])
            or type(shape[3]) is not bool
        ):
            raise ValueError(
                "Expected bits 4/8 and shape [positive M, K, N, bool bias]"
            )
        if profile_bits is not None and bits != profile_bits:
            raise ValueError("Shape tactic bits do not match the profile quant_format")
        key = (bits, *shape)
        if key in choices:
            raise ValueError("Duplicate integer shape tactic")
        choices[key] = checked_tactic(row["tactic"])
    return choices


def apply_tactics(controller, extra, config, coverage, *, backend_evidence=None):
    shapes = (
        None
        if config.shape_tactics is None
        else read_shape_tactics(config.shape_tactics, backend_evidence=backend_evidence)
    )
    singles = read_json(config.tactics, {})
    groups = read_json(config.group_tactics, [])
    for bits in (8, 4):
        models = {**controller.quant[bits], **extra.quant[bits]}
        legacy = [row for row in groups if row["bits"] == bits and "shape" in row]
        for name, tactic in singles.get(str(bits), {}).items():
            if name in models:
                models[name].tactic = checked_tactic(tactic)
        for names, group in controller.grouped[bits]:
            q = group.linear
            matching = [
                row
                for row in groups
                if row["bits"] == bits and row.get("members") == list(names)
            ]
            if not matching and legacy and names[0] in coverage:
                shape = coverage[names[0]]["shape"]
                signature = [
                    math.prod(shape[:-1]),
                    q.in_features,
                    q.out_features,
                    q.bias is not None,
                ]
                matching = [row for row in legacy if row["shape"] == signature]
            choices = {checked_tactic(row["tactic"]) for row in matching}
            if len(choices) > 1:
                raise ValueError("Conflicting grouped tactics")
            q.tactic = choices.pop() if choices else 0
    if config.shape_tactics is not None:
        from coexecution.optimization.backends.integer.linear import (
            install_shape_tactics,
        )

        for bits in (8, 4):
            models = [
                *controller.quant[bits].values(),
                *extra.quant[bits].values(),
                *(group.linear for _, group in controller.grouped[bits]),
            ]
            for q in models:
                install_shape_tactics(q, shapes)
