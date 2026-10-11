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


def read_shape_tactics(path):
    choices = {}
    rows = read_json(path, [])
    if not isinstance(rows, list):
        raise TypeError("Shape tactics must be a JSON list")
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
        key = (bits, *shape)
        if key in choices:
            raise ValueError("Duplicate integer shape tactic")
        choices[key] = checked_tactic(row["tactic"])
    return choices


def apply_tactics(controller, extra, config, coverage):
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

        shapes = read_shape_tactics(config.shape_tactics)
        for bits in (8, 4):
            models = [
                *controller.quant[bits].values(),
                *extra.quant[bits].values(),
                *(group.linear for _, group in controller.grouped[bits]),
            ]
            for q in models:
                install_shape_tactics(q, shapes)
