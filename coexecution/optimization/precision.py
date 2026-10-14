"""Explicit integer maps and fixed cumulative tiers; no GPU/model imports."""

import json
import math
import re
from pathlib import Path


def checked_precision_map(mapping, names, groups=()):
    names = set(names)
    if not isinstance(mapping, dict) or set(mapping) != names:
        raise ValueError("Precision map must cover every selected candidate exactly")
    if any(type(bits) is not int or bits not in (4, 8) for bits in mapping.values()):
        raise ValueError("Candidate precision must be integer 4 or 8")
    for group in groups:
        if not set(group) <= names or len({mapping[n] for n in group}) != 1:
            raise ValueError("Shared-input projection groups must switch atomically")
    return dict(mapping)


def read_precision_map(path, names, groups=()):
    data = json.loads(Path(path).read_text())
    if (
        not isinstance(data, dict)
        or type(data.get("schema_version")) is not int
        or data["schema_version"] != 1
    ):
        raise ValueError("Precision map schema_version must be 1")
    return checked_precision_map(data.get("precision_map"), names, groups)


def progressive_maps(names, groups=(), category_names=(), divisions=10):
    """Vision -> text -> visual transformer -> DiT -> additional -> category.

    The x-axis is candidate count, not parameter/FLOP coverage. Shared groups are
    indivisible, so actual fractions can exceed each nominal cumulative target.
    """
    names = set(names)
    groups = list(groups)
    categories = set(category_names)
    if (
        not names
        or not categories <= names
        or type(divisions) is not int
        or divisions < 1
    ):
        raise ValueError("Nonempty candidates and positive integer divisions required")

    def order(name):
        family = (
            5
            if name in categories
            else 0
            if ".visual." in name
            else 1
            if ".language_model." in name
            else 2
            if name.startswith("action_head.vl_self_attention.")
            else 3
            if name.startswith("action_head.model.transformer_blocks.")
            else 4
        )
        natural = tuple(
            (1, int(x)) if x.isdigit() else (0, x) for x in re.split(r"(\d+)", name)
        )
        return family, natural

    chunks, seen = [], set()
    for group in groups:
        group = set(group)
        if not group or not group <= names or seen & group:
            raise ValueError("Projection groups must be disjoint candidate subsets")
        seen |= group
        chunks.append(sorted(group, key=order))
    chunks.extend([[name] for name in names - seen])
    chunks.sort(key=lambda group: order(group[0]))
    result = []
    for tier in range(divisions + 1):
        target = math.ceil(len(names) * tier / divisions)
        selected = set()
        for group in chunks:
            if len(selected) >= target:
                break
            selected.update(group)
        mapping = {name: 4 if name in selected else 8 for name in sorted(names)}
        result.append(
            {
                "schema_version": 1,
                "tier": tier,
                "w4_count": len(selected),
                "w8_count": len(names) - len(selected),
                "w4_fraction": len(selected) / len(names),
                "precision_map": checked_precision_map(mapping, names, groups),
            }
        )
    return result
