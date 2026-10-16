"""Offline replay catalog builder. Model array loading is lazy and untimed."""

from datetime import datetime, timezone
from pathlib import Path

from .schema import VERSION, write_json


def build_catalog(
    *,
    model_runs,
    deployments,
    output,
    clock_cohort=None,
    default_execution_clock=None,
    default_execution_coupling=None,
    default_execution_prefix_steps=None,
    default_async_overlap_steps=None,
):
    from .deployment import deployment_directories, deployment_run
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
    if not runs:
        raise ValueError("At least one measured model or deployment run is required")
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
                **(
                    {
                        "execution_clock": run["execution_clock"],
                        "execution_coupling": run["execution_coupling"],
                        "execution_prefix_steps": run["execution_prefix_steps"],
                        "async_overlap_steps": run["async_overlap_steps"],
                        "policy_variant": run["policy_variant"],
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
