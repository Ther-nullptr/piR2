"""Offline replay catalog builder. Model array loading is lazy and untimed."""

import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .schema import VERSION, read_json, under_root, write_json

EPISODE_INDEX_FIELDS = (
    "id",
    "item_id",
    "condition_id",
    "parent_dynamic_condition_id",
    "template_condition_id",
    "suite",
    "task_id",
    "level",
    "traj_index",
    "speed_mps",
    "seed",
    "init_state_id",
    "speed_scope",
    "seed_scope",
    "scene_drive",
    "scene_drive_scope",
    "protocol",
    "timing_mode",
    "fixed_delay",
    "execution_prefix_steps",
)


def _duration_bounds(requests):
    values = [request.get("service_duration_ms") for request in requests]
    if not values or any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in values
    ):
        return None
    return {"min": min(values), "max": max(values)}


def dynamic_index(run):
    """Small selector metadata; measured bounds are unknown for partial evidence."""
    episodes = run.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError("Dynamic run needs recorded episodes")
    summaries, identifiers = [], set()
    for episode in episodes:
        identifier = episode.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
        ):
            raise ValueError("Dynamic episodes need unique identities")
        identifiers.add(identifier)
        summary = {key: episode.get(key) for key in EPISODE_INDEX_FIELDS}
        for key, fields in (
            ("stage_latencies", ("bootstrap_s", "s1_s", "s2_s")),
            (
                "calibration",
                (
                    "kind",
                    "protocol",
                    "retained_requests",
                    "warmup_requests",
                    "raw_latency_s",
                    "applied_latency_s",
                ),
            ),
        ):
            value = episode.get(key)
            summary[key] = (
                {field: value[field] for field in fields if field in value}
                if isinstance(value, dict)
                else None
            )
        requests = episode.get("requests", [])
        summary["request_duration_bounds_ms"] = {
            lane: _duration_bounds(
                requests
                if lane == "all"
                else [r for r in requests if r.get("lane") == lane]
            )
            for lane in ("all", "s1", "s2")
        }
        summaries.append(summary)
    return {"protocol": run.get("protocol"), "episode_summaries": summaries}


def refresh_index(catalog_path):
    """Refresh only catalog summaries, atomically, from its listed recorded runs."""
    path = Path(catalog_path)
    if path.is_dir():
        path /= "catalog.json"
    path = path.resolve(strict=True)
    catalog = read_json(path)
    if (
        catalog.get("schema") != "pir2.replay.catalog"
        or catalog.get("version") != VERSION
        or not isinstance(catalog.get("runs"), list)
    ):
        raise ValueError("Unsupported replay catalog schema/version")
    identifiers = set()
    for descriptor in catalog["runs"]:
        identifier = descriptor.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
        ):
            raise ValueError("Catalog needs unique run identities")
        identifiers.add(identifier)
        relative = descriptor.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("Catalog run needs a relative path")
        run = read_json(under_root(path.parent, relative))
        if (
            run.get("schema") != "pir2.replay.run"
            or run.get("version") != VERSION
            or any(
                not descriptor.get(key) or run.get(key) != descriptor[key]
                for key in ("id", "kind", "precision")
            )
        ):
            raise ValueError(f"Replay run identity mismatch: {identifier}")
        if run["kind"] == "dynamic":
            summary = dynamic_index(run)
            if "episodes" in descriptor and descriptor["episodes"] != len(
                summary["episode_summaries"]
            ):
                raise ValueError(f"Dynamic episode count mismatch: {identifier}")
            descriptor.update(summary)
        # Drop the full run before reading the next, even for a legacy catalog.
        del run
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".catalog-", delete=False
        ) as target:
            temporary = Path(target.name)
        write_json(temporary, catalog)
        temporary.chmod(path.stat().st_mode & 0o777)
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return catalog


def build_catalog(
    *,
    model_runs,
    deployments,
    output,
    dynamic_evaluations=(),
    clock_cohort=None,
    default_execution_clock=None,
    default_execution_coupling=None,
    default_execution_prefix_steps=None,
    default_async_overlap_steps=None,
):
    from .deployment import deployment_directories, deployment_run
    from .dynamic import dynamic_run
    from .media import MediaRegistry
    from .model import compare_outputs
    from .model import model_runs as import_models

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Replay output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    media = MediaRegistry(output)
    runs = []
    for path in model_runs:
        runs.extend(import_models(path, media, clock_cohort=clock_cohort))
    for path in deployments:
        directories = deployment_directories(path)
        if not directories:
            raise ValueError(f"No deployment evidence under {path}")
        runs.extend(deployment_run(directory, media) for directory in directories)
    for path in dynamic_evaluations:
        runs.append(dynamic_run(path, media))
    if not runs:
        raise ValueError(
            "At least one recorded model, deployment or dynamic run is required"
        )
    if len({run["id"] for run in runs}) != len(runs):
        raise ValueError("Duplicate replay run input")
    if default_execution_clock is not None and (
        default_execution_clock not in ("paced", "ready")
        or not any(
            run.get("execution_clock") == default_execution_clock
            for run in runs
            if run["kind"] == "deployment"
        )
    ):
        raise ValueError(
            "Requested default execution clock is not present in deployment evidence"
        )
    if default_execution_coupling is not None and (
        default_execution_coupling
        not in ("asynchronous", "synchronous", "one_step_async", "chunked_async")
        or not any(
            run.get("execution_coupling") == default_execution_coupling
            and (
                default_execution_clock is None
                or run.get("execution_clock") == default_execution_clock
            )
            for run in runs
            if run["kind"] == "deployment"
        )
    ):
        raise ValueError(
            "Requested execution coupling/clock is not present in deployment evidence"
        )
    schedule = (default_execution_prefix_steps, default_async_overlap_steps)
    if any(v is not None for v in schedule) and (
        type(schedule[0]) is not int
        or type(schedule[1]) is not int
        or not 0 <= schedule[1] <= schedule[0]
        or schedule[0] < 1
        or not any(
            run["kind"] == "deployment"
            and run.get("execution_prefix_steps") == schedule[0]
            and run.get("async_overlap_steps") == schedule[1]
            and (
                default_execution_clock is None
                or run.get("execution_clock") == default_execution_clock
            )
            and (
                default_execution_coupling is None
                or run.get("execution_coupling") == default_execution_coupling
            )
            for run in runs
        )
    ):
        raise ValueError("Requested default K/D schedule is not present in evidence")
    if schedule[0] is not None:
        selected = next(
            run
            for run in runs
            if run["kind"] == "deployment"
            and run.get("execution_prefix_steps") == schedule[0]
            and run.get("async_overlap_steps") == schedule[1]
            and (
                default_execution_clock is None
                or run["execution_clock"] == default_execution_clock
            )
            and (
                default_execution_coupling is None
                or run["execution_coupling"] == default_execution_coupling
            )
        )
        default_execution_clock = default_execution_clock or selected["execution_clock"]
        default_execution_coupling = (
            default_execution_coupling or selected["execution_coupling"]
        )
    compare_outputs(runs)
    index = []
    for run in runs:
        path = Path("runs") / (run["id"] + ".json")
        write_json(output / path, run)
        index.append(
            {key: run[key] for key in ("id", "kind", "label", "precision", "mode")}
            | {
                "path": str(path),
                "clean_requests": len(run.get("clean_requests", [])),
                "profile_requests": len(run.get("profile_requests", [])),
                "episodes": len(run.get("episodes", [])),
                "warnings": run["warnings"],
                **(dynamic_index(run) if run["kind"] == "dynamic" else {}),
                **(
                    {
                        "execution_clock": run["execution_clock"],
                        "execution_coupling": run["execution_coupling"],
                        "execution_prefix_steps": run["execution_prefix_steps"],
                        "async_overlap_steps": run["async_overlap_steps"],
                        "policy_variant": run["policy_variant"],
                        "deployment_scheduling_policy": run[
                            "deployment_scheduling_policy"
                        ],
                        "controller_identity": run["controller_identity"],
                    }
                    if run["kind"] == "deployment"
                    else {}
                ),
            }
        )
    write_json(
        output / "media.json",
        {
            "schema": "pir2.replay.media",
            "version": VERSION,
            "assets": list(media.assets.values()),
        },
    )
    catalog = {
        "schema": "pir2.replay.catalog",
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": index,
        "media_manifest": "media.json",
        "clock_cohort": clock_cohort,
        "default_execution_clock": default_execution_clock,
        "default_execution_coupling": default_execution_coupling,
        "default_execution_prefix_steps": default_execution_prefix_steps,
        "default_async_overlap_steps": default_async_overlap_steps,
        "notes": [
            "Independent model replay and closed-loop robot trajectories have separate measurement boundaries",
            "Missing evidence remains null or unavailable, never a fabricated zero",
            "No profile aggregate is converted into a synthetic timeline",
        ],
    }
    write_json(output / "catalog.json", catalog)
    return catalog
