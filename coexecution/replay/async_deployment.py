"""Display records for audited complete full-policy asynchronous work."""

from coexecution.async_evidence import validate_async_work


def async_replay(events, queue, episode_id):
    evidence = validate_async_work(events, normalized=queue["work_requests"])
    start = evidence["start_s"]
    phases = []
    for raw in evidence["phases"]:
        # Preserve absolute raw evidence separately; displayed coordinates share
        # only the normalized control-wall origin, never server/GPU clock guesses.
        phase = {key: value for key, value in raw.items() if not key.endswith("_s")}
        phase.update(
            {
                key: value - start
                for key, value in raw.items()
                if key.endswith("_s") and value is not None
            }
        )
        suffix = (
            f"execution-{raw['execution_id']}"
            if raw["phase"] == "execution" and "execution_id" in raw
            else raw["phase"]
        )
        phase.update(
            id=f"{episode_id}/request-{raw['request_id']}/{suffix}",
            raw=raw,
            status="completed",
            clock_id="control_wall",
        )
        phases.append(phase)
    tasks = []
    for request in evidence["work_requests"]:
        rid = request["request_id"]
        source = request["source_tick"]
        stages = request["server_stages"]
        shared = {
            **request,
            "feature_sequence": rid,
            "feature_id": f"{episode_id}/feature-{rid}",
            "observation_id": f"{episode_id}/observation-{source}",
            "status": "unused" if request["disposition"] == "unused" else "completed",
            "censored": False,
            "clock_id": "control_wall",
        }
        for role, prefix in (("S2", "vlm"), ("S1", "action")):
            task = {
                **shared,
                "id": f"{episode_id}/{role}-{rid}",
                "role": role,
                "timing_scope": f"server_{prefix}_stage",
                "start_s": stages[f"{prefix}_started_s"],
                "end_s": stages[f"{prefix}_completed_s"],
                "completed_s": stages[f"{prefix}_completed_s"],
            }
            if role == "S2":
                task.update(published_s=task["completed_s"])
            else:
                task.update(
                    feature_source_tick=source,
                    feature_capture_s=request["capture_s"],
                    state_tick=source,
                    state_capture_s=request["capture_s"],
                    use_s=task["start_s"],
                    use_boundary="server_plan_started_s",
                    timing_version=2,
                )
            tasks.append(task)
    return phases, tasks


# Public compatibility for callers importing the original v1 helper.
one_step_async_replay = async_replay
