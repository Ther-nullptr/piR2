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
        not in ("asynchronous", "synchronous", "one_step_async")
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
        "notes": [
            "Independent model replay and closed-loop robot trajectories have separate measurement boundaries",
            "Missing evidence remains null or unavailable, never a fabricated zero",
            "No profile aggregate is converted into a synthetic timeline",
        ],
    }
    write_json(output / "catalog.json", catalog)
    return catalog
