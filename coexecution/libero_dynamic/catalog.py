"""Validate reviewed moving-object configurations and enumerate paired conditions.

This module performs metadata work only. It neither starts a simulator nor
implements evaluation. The source dictionaries retain their original speeds;
``speed_mps`` is a separate declared experimental override for a future runner.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LEVELS = ("l1", "l2", "l3")
SPEEDS_MPS = (0.03, 0.06, 0.09)
MANIFEST = Path(__file__).resolve().parents[2] / "configs/libero/dynamic-benchmark.json"


def read_reviewed_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Verify raw file bytes against an independently pinned digest, then parse."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Reviewed dynamic configuration is missing: {path}")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"Reviewed configuration SHA256 mismatch: {path}; "
            f"expected {expected_sha256}, found {actual}"
        )
    try:
        import yaml
    except ImportError as error:
        raise ValueError("Reading dynamic configurations requires PyYAML") from error
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid dynamic configuration YAML: {path}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Dynamic configuration must be a mapping: {path}")  # noqa: TRY004 -- invalid file content
    return data


def _speed_grid(speeds) -> tuple[float, ...]:
    speeds = tuple(speeds)
    if not speeds or any(
        isinstance(speed, bool)
        or not isinstance(speed, (int, float))
        or not math.isfinite(speed)
        or speed not in SPEEDS_MPS
        for speed in speeds
    ):
        raise ValueError(
            f"Benchmark speeds must be selected from {SPEEDS_MPS}; "
            "speed 0 does not disable upstream motion"
        )
    if len(set(speeds)) != len(speeds):
        raise ValueError("Duplicate benchmark speed")
    return tuple(sorted(speeds))


def _validate_configs(configs: dict[str, dict]) -> None:
    if set(configs) != set(LEVELS):
        raise ValueError(f"Level coverage must be exactly {LEVELS}")
    for level in LEVELS:
        config = configs[level]
        if not isinstance(config, dict) or config.get("enabled") is not True:
            raise ValueError(f"Disabled or malformed dynamic level: {level}")
        tasks = config.get("tasks", {})
        if not isinstance(tasks, dict) or set(tasks) != set(SUITES):
            raise ValueError(f"Suite coverage must be exactly {SUITES}: {level}")
        for suite in SUITES:
            by_task = tasks[suite]
            if (
                not isinstance(by_task, dict)
                or any(type(task_id) is not int for task_id in by_task)
                or set(by_task) != set(range(10))
            ):
                raise ValueError(
                    f"Incomplete or duplicate task coverage: {level}/{suite}"
                )
            for task_id, task in by_task.items():
                prefix = f"{suite}_task{task_id:02d}__{level}"
                if not isinstance(task, dict) or task.get("enabled", True) is not True:
                    raise ValueError(f"Disabled or malformed task: {prefix}")
                variants = task.get("activation_variants")
                if not isinstance(variants, list) or len(variants) != 5:
                    raise ValueError(f"Expected five reviewed variants: {prefix}")
                names = set()
                for index, variant in enumerate(variants):
                    if not isinstance(variant, dict):
                        raise ValueError(f"Malformed variant: {prefix}")  # noqa: TRY004 -- invalid config content
                    name = variant.get("variant_name")
                    if name not in {f"traj{i:02d}" for i in range(5)} or name in names:
                        raise ValueError(
                            f"Missing or duplicate trajectory variant: {prefix}"
                        )
                    names.add(name)
                    if name != f"traj{index:02d}":
                        raise ValueError(
                            f"Trajectory ordering disagrees with upstream indices: {prefix}"
                        )
                    if variant.get("variant_id") != f"{prefix}__{name}":
                        raise ValueError(
                            f"Variant identity disagrees with task: {prefix}"
                        )
                    motion = variant.get("motion", task.get("motion"))
                    if not isinstance(motion, dict) or not motion.get("type"):
                        raise ValueError(f"Missing actual motion type: {prefix}/{name}")
                    targets = variant.get("target_objects", task.get("target_objects"))
                    if targets is None:
                        target_name = variant.get(
                            "target_object_name", task.get("target_object_name")
                        )
                        if not target_name:
                            raise ValueError(
                                f"Missing target identity: {prefix}/{name}"
                            )
                        continue
                    if not isinstance(targets, list) or not targets:
                        raise ValueError(f"Missing target identities: {prefix}/{name}")
                    target_names = []
                    for target in targets:
                        if not isinstance(target, dict) or not target.get(
                            "target_object_name"
                        ):
                            raise ValueError(
                                f"Missing target identity: {prefix}/{name}"
                            )
                        target_names.append(target["target_object_name"])
                    if len(set(target_names)) != len(target_names):
                        raise ValueError(f"Duplicate target identity: {prefix}/{name}")


def expand_conditions(
    configs: dict[str, dict], *, speeds=SPEEDS_MPS, suites=None
) -> list[dict[str, Any]]:
    """Validate all 600 reviewed variants and expand an explicit benchmark subset.

    This accepts parsed dictionaries for callers and contract tests. Source-byte
    verification belongs to :func:`build_inventory`; this function alone does
    not attest to any upstream revision or config hash.
    """
    speeds = _speed_grid(speeds)
    selected = tuple(SUITES if suites is None else suites)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(SUITES)
    ):
        raise ValueError(f"Select distinct benchmark suites from {SUITES}")
    _validate_configs(configs)
    records = []
    for suite in SUITES:
        if suite not in selected:
            continue
        for task_id in range(10):
            for level in LEVELS:
                task = configs[level]["tasks"][suite][task_id]
                variants = sorted(
                    task["activation_variants"], key=lambda v: v["variant_name"]
                )
                for index, variant in enumerate(variants):
                    item_id = variant["variant_id"]
                    # Variant fields replace corresponding task fields upstream.
                    selected_fields = {
                        key: copy.deepcopy(variant.get(key, task.get(key)))
                        for key in (
                            "target_object_name",
                            "target_joint_name",
                            "target_body_name",
                            "target_kind",
                            "dynamic_role",
                            "motion",
                            "latency",
                            "target_objects",
                            "initial_scene_adjustments",
                        )
                        if key in variant or key in task
                    }
                    # Match upstream concrete_item_cfg: a multi-target variant
                    # uses its first target's motion as the effective top-level
                    # path. Raw variant geometry can differ and is separate.
                    selected_fields["raw_variant_motion"] = copy.deepcopy(
                        variant.get("motion")
                    )
                    if selected_fields.get("target_objects"):
                        selected_fields["motion"] = copy.deepcopy(
                            selected_fields["target_objects"][0].get("motion", {})
                        )
                    for speed in speeds:
                        records.append(
                            {
                                "condition_id": f"{item_id}__s{round(speed * 100):03d}",
                                "item_id": item_id,
                                "variant_id": item_id,
                                "variant_name": variant["variant_name"],
                                "suite": suite,
                                "task_id": task_id,
                                "level": level,
                                "traj_index": index,
                                "speed_mps": speed,
                                "seed": 0,
                                "init_state_id": 0,
                                "pre_policy_dynamic_steps": 40,
                                **copy.deepcopy(selected_fields),
                            }
                        )
    return records


def build_inventory(upstream: Path, *, suites=None) -> dict[str, Any]:
    """Enumerate conditions using this repository's pinned expected config hashes."""
    upstream = Path(upstream).resolve()
    manifest = json.loads(MANIFEST.read_text())
    configs = {
        level: read_reviewed_config(upstream / source["path"], source["sha256"])
        for level, source in manifest["configs"].items()
    }
    records = expand_conditions(configs, suites=suites)
    result = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    actual_revision = result.stdout.strip() if result.returncode == 0 else None
    if actual_revision is not None and actual_revision != manifest["upstream_revision"]:
        raise ValueError(f"Upstream revision mismatch: {actual_revision}")
    for record in records:
        record["config_source"] = copy.deepcopy(manifest["configs"][record["level"]])
    selected_suites = [
        suite for suite in SUITES if any(r["suite"] == suite for r in records)
    ]
    return {
        "schema": "pir2.libero_dynamic.inventory.v1",
        "status": "inventory_only",
        "evaluation_supported": False,
        "upstream": {
            "url": manifest["upstream_url"],
            "expected_revision": manifest["upstream_revision"],
            "actual_revision": actual_revision,
            "path": str(upstream),
            "verified_configs": manifest["configs"],
            "verification_scope": "config bytes and complete condition grid; not runtime evaluation",
        },
        "selection": {"suites": selected_suites, "speed_mps": list(SPEEDS_MPS)},
        "counts": {
            "tasks": len(selected_suites) * 10,
            "reviewed_variants": len(records) // len(SPEEDS_MPS),
            "conditions": len(records),
            "full_population_conditions": manifest["full_population_conditions"],
        },
        "seed": manifest["seed"],
        "init_state_id": manifest["init_state_id"],
        "pre_policy_dynamic_steps": manifest["pre_policy_dynamic_steps"],
        "speed_contract": manifest["speed_contract"],
        "phase_contract": manifest["phase_contract"],
        "conditions": records,
    }
