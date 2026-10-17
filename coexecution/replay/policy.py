"""Policy identity and observed piR2 work, without importing model libraries.

Scheduling names and artifact labels do not identify a policy. Old records may
lack worker identities or counters; missing evidence remains unknown.
"""

from pathlib import Path

from .schema import read_json, source


def _consistent(values, name):
    known = [value for value in values if value is not None]
    if known and any(value != known[0] for value in known):
        raise ValueError(f"Policy {name} identities disagree")
    return known[0] if known else None


def policy_identity(config):
    coupling = config.get("execution_coupling", "asynchronous")
    roles = [("checkpoint", "action")]
    if coupling == "asynchronous":
        roles.append(("slow_checkpoint", "vlm"))
    workers = [config.get(key) or {} for key, _ in roles]
    claims = [config.get("variant"), *[w.get("variant") for w in workers]]
    declared = _consistent(claims, "variant")
    if declared not in (None, "pir2", "flow"):
        raise ValueError("Unknown policy variant")
    if declared == "pir2" and coupling != "asynchronous":
        raise ValueError("piR2 requires the independent cached-worker protocol")
    for worker, (_, role) in zip(workers, roles):
        if worker.get("role") not in (None, role):
            raise ValueError("Policy worker roles disagree")
    variant = declared if declared and all(v is not None for v in claims) else None
    predicted = _consistent(
        [w.get("predicted_horizon") for w in workers], "predicted horizon"
    )
    decoded = _consistent(
        [w.get("decoded_horizon") for w in workers], "decoded horizon"
    )
    for value in (predicted, decoded):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError("Invalid policy horizon")
    if predicted is not None and decoded is not None and decoded > predicted:
        raise ValueError("Decoded policy horizon exceeds predicted horizon")
    hashes = [w.get("sha256", {}) for w in workers]
    for key in {key for values in hashes for key in values}:
        _consistent([values.get(key) for values in hashes], f"checkpoint {key}")
    checkpoints = [
        {"role": role, "path": worker.get("checkpoint"), "sha256": worker.get("sha256")}
        for worker, (_, role) in zip(workers, roles)
    ]
    identity = {
        "variant": variant,
        "declared_variant": declared,
        "status": "verified_workers" if variant else "unknown",
        "predicted_horizon": predicted,
        "decoded_horizon": decoded,
        "checkpoints": checkpoints,
        "streaming": None,
        "schedule_mode": None,
        "training": None,
    }
    provenance = []
    # Optional, hash-bound checkpoint metadata. Never infer a training step from
    # a directory name, and never load pickle/weights just to render a replay.
    checkpoint = workers[0].get("checkpoint")
    if checkpoint and hashes[0].get("config.json"):
        path = Path(checkpoint) / "config.json"
        if path.is_file():
            receipt = source(path)
            if receipt["sha256"] != hashes[0]["config.json"]:
                raise ValueError(
                    "Recorded checkpoint config hash disagrees with local file"
                )
            model = read_json(path)
            streaming = model.get("streaming")
            schedule = model.get("streaming_schedule_mode")
            if streaming is not None and type(streaming) is not bool:
                raise ValueError("Invalid checkpoint streaming identity")
            if declared and streaming is not None and streaming != (declared == "pir2"):
                raise ValueError(
                    "Checkpoint streaming mode disagrees with worker variant"
                )
            if declared == "pir2" and schedule not in (None, "pir2"):
                raise ValueError("Checkpoint streaming schedule disagrees with piR2")
            _consistent([predicted, model.get("action_horizon")], "checkpoint horizon")
            identity.update(streaming=streaming, schedule_mode=schedule)
            provenance.append(receipt)
            training = path.with_name("trainer_state.json")
            if training.is_file():
                step = read_json(training).get("global_step")
                if step is not None and (type(step) is not int or step < 0):
                    raise ValueError("Invalid recorded training global_step")
                training_source = source(training)
                identity["training"] = {
                    "global_step": step,
                    "basis": "checkpoint trainer_state.json; metadata, not a quality claim",
                    "source": training_source,
                }
                provenance.append(training_source)
    return identity, provenance


def _counts(value, expected, name):
    if value is None:
        return None
    if not isinstance(value, dict) or any(
        type(value.get(key)) is not int or value[key] != number
        for key, number in expected.items()
    ):
        raise ValueError(f"piR2 {name} forward counts disagree with observed protocol")
    return dict(value)


def _unique(rows, key, name):
    result = {}
    for row in rows:
        identity = key(row)
        if identity in result:
            raise ValueError(f"Duplicate piR2 {name} evidence")
        result[identity] = row
    return result


def pir2_episode_evidence(events, metadata, queue, identity=None):
    """Attach raw count/source evidence to RPCs; do not invent stage intervals.

    Older VisionWorker records omit the VLM counter. A consumed cache can prove
    it only through the exact published source identity. Unconsumed requests
    remain unknown, and are not counted again each time a cache is reused.
    """
    submitted = _unique(
        [r for r in events if r.get("kind") == "action_submitted"],
        lambda r: r.get("request_tick"),
        "submission",
    )
    starts = [r for r in events if r.get("kind") == "control_start"]
    decoded = starts[0].get("bootstrap_action_slots") if len(starts) == 1 else None
    _consistent(
        [decoded, (identity or {}).get("decoded_horizon")], "bootstrap decoded horizon"
    )
    results = _unique(
        [
            r
            for r in events
            if r.get("kind") in ("request_completed", "unused_result_at_episode_end")
        ],
        lambda r: r.get("audit", {}).get("request_tick"),
        "result",
    )
    requests, caches = {}, {}
    if any(
        type(tick) is not int or tick < 0 for tick in submitted.keys() | results.keys()
    ):
        raise ValueError("piR2 request evidence requires a physical tick identity")
    for tick in sorted(submitted.keys() | results.keys()):
        result, submission = results.get(tick, {}), submitted.get(tick, {})
        audit, timing = result.get("audit", {}), result.get("client_timing", {})
        restarted = audit.get("buffer_restarted")
        if restarted is not None and type(restarted) is not bool:
            raise ValueError("Invalid piR2 buffer recovery evidence")
        expected = {"vlm": 0, "dit": 5 if restarted else 1}
        counts = (
            _counts(audit.get("forward_counts"), expected, "S1")
            if restarted is not None
            else None
        )
        category = ("recovery" if restarted else "normal") if counts else "unknown"
        start, end = (
            submission.get("committed_start_tick"),
            submission.get("committed_end_tick"),
        )
        prefix = None
        if start is not None and end is not None:
            if type(start) is not int or type(end) is not int or end < start:
                raise ValueError("Invalid piR2 committed prefix")
            _consistent(
                [end - start, submission.get("delay_ticks"), audit.get("d_request")],
                "committed prefix",
            )
            prefix = {"start_tick": start, "end_tick": end, "steps": end - start}
        _consistent(
            [submission.get("state_tick"), timing.get("state_tick")], "state source"
        )
        cache = audit.get("cache") or {}
        _consistent(
            [cache.get("source_tick"), timing.get("feature_source_tick")],
            "feature source tick",
        )
        _consistent(
            [cache.get("capture_s"), timing.get("feature_capture_s")], "feature capture"
        )
        cache_key = (
            cache.get("source_tick"),
            cache.get("capture_s"),
            timing.get("feature_sequence"),
        )
        if (
            all(v is not None for v in cache_key)
            and cache.get("vlm_forward_calls") is not None
        ):
            count = cache["vlm_forward_calls"]
            if type(count) is not int or count != 1:
                raise ValueError("piR2 consumed S2 VLM counter must equal one")
            caches[cache_key] = count
        requests[tick] = {
            "category": category,
            "forward_counts": counts,
            "disposition": "unused_final"
            if result.get("kind") == "unused_result_at_episode_end"
            else "consumed"
            if result
            else "unknown",
            "audit": audit or None,
            "client_timing": timing or None,
            "committed_prefix": prefix,
            "timing_basis": "recorded client RPC; no inferred VLM/DiT stage split",
        }
    publications = _unique(
        [r for r in events if r.get("kind") == "feature_published"],
        lambda r: (r.get("source_tick"), r.get("capture_s"), r.get("rpc_completed_s")),
        "feature publication",
    )
    vision = []
    for row in events:
        if row.get("kind") != "vision_request":
            continue
        key = (row.get("source_tick"), row.get("capture_s"))
        publication = publications.get((*key, row.get("completed_s")))
        direct = row.get("vlm_forward_calls")
        if direct is not None and (type(direct) is not int or direct != 1):
            raise ValueError("piR2 S2 VLM counter must equal one")
        count = (
            direct
            if direct is not None
            else caches.get((*key, publication.get("feature_sequence")))
            if publication
            else None
        )
        vision.append(
            {
                "source_tick": row.get("source_tick"),
                "capture_s": row.get("capture_s"),
                "started_s": row.get("started_s"),
                "completed_s": row.get("completed_s"),
                "feature_sequence": publication.get("feature_sequence")
                if publication
                else None,
                "vlm_forward_calls": count,
                "count_basis": "vision_request"
                if direct is not None
                else "consumed cache with exact source/publication identity"
                if count is not None
                else None,
                "graph_execution": row.get("graph_execution"),
            }
        )
    for recorded, actual in (
        (metadata.get("action_requests"), len(requests)),
        ((metadata.get("vlm_work") or {}).get("completed"), len(vision)),
    ):
        if recorded is not None and (type(recorded) is not int or recorded != actual):
            raise ValueError("piR2 request accounting disagrees with episode metadata")
    for request in queue.get("requests", []):
        if request["role"] == "S1":
            request["policy_evidence"] = requests.get(request.get("request_tick"))
        else:
            matches = [
                v
                for v in vision
                if v["source_tick"] == request.get("source_tick")
                and v["feature_sequence"] == request.get("feature_sequence")
            ]
            request["policy_evidence"] = matches[0] if len(matches) == 1 else None
    unknown = sum(r["category"] == "unknown" for r in requests.values())
    s2_unknown = sum(r["vlm_forward_calls"] is None for r in vision)
    return {
        "variant": "pir2",
        "scope": "recorded episode work including final unused requests; bootstrap separate; excludes calibration/warmup",
        "bootstrap": {
            "decoded_horizon": decoded,
            "forward_counts": _counts(
                metadata.get("bootstrap_forward_counts"),
                {"vlm": 1, "dit": 4},
                "bootstrap",
            ),
            "seconds": metadata.get("bootstrap_seconds"),
            "timing_basis": "aggregate only; no fabricated timeline placement",
        },
        "s1": {
            "requests": len(requests),
            "normal_requests": sum(
                r["category"] == "normal" for r in requests.values()
            ),
            "recovery_requests": sum(
                r["category"] == "recovery" for r in requests.values()
            ),
            "unknown_requests": unknown,
            "dit_forward_calls": None
            if unknown
            else sum(r["forward_counts"]["dit"] for r in requests.values()),
            "vlm_forward_calls": None
            if unknown
            else sum(r["forward_counts"]["vlm"] for r in requests.values()),
            "records": [
                {"request_tick": tick, **value} for tick, value in requests.items()
            ],
        },
        "s2": {
            "requests": len(vision),
            "unknown_requests": s2_unknown,
            "verified_vlm_forward_calls": sum(
                v["vlm_forward_calls"] or 0 for v in vision
            ),
            "vlm_forward_calls": None
            if s2_unknown
            else sum(v["vlm_forward_calls"] for v in vision),
            "records": vision,
        },
    }


def controller_identity(config, *records):
    """Separate controller scheduling from model identity; never infer by label."""
    claims, bases = [], []
    for index, record in enumerate((config, *records)):
        explicit = record.get("deployment_scheduling_policy")
        delay = record.get("delay_policy", record.get("action_delay_policy"))
        inferred = (
            "responsive"
            if delay == "responsive_rolling_p95"
            else "conservative"
            if delay in ("adaptive_nondecreasing", "fixed", "one_step_committed")
            else None
        )
        for value in (explicit, inferred):
            if value is not None:
                if value not in ("conservative", "responsive"):
                    raise ValueError("Invalid controller policy")
                claims.append(value)
        if explicit is not None:
            bases.append("configuration" if index == 0 else "recorded_controller")
        elif inferred is not None:
            bases.append("recorded_delay_policy")
    return {
        "policy": _consistent(claims, "controller"),
        "basis": sorted(set(bases)) or ["legacy_unspecified"],
    }


def attach_publication_evidence(events, queue, origin_s):
    """Join raw read/publication counts and exact executed-slot ownership.

    This creates derived replay fields only. Legacy action_adopted is a result
    read marker, not proof of installed actions. Missing counters remain null.
    """
    result_rows = [
        r
        for r in events
        if r.get("kind") in ("request_completed", "unused_result_at_episode_end")
    ]
    results = _unique(
        result_rows,
        lambda r: r.get("audit", {}).get("request_tick"),
        "publication result",
    )
    adopted = _unique(
        [r for r in events if r.get("kind") == "action_adopted"],
        lambda r: r.get("request_tick"),
        "publication read",
    )
    submitted = _unique(
        [r for r in events if r.get("kind") == "action_submitted"],
        lambda r: r.get("request_tick"),
        "publication submission",
    )
    executions = {}
    ticks = [r for r in events if r.get("kind") == "control_tick"]
    control_end = next((r for r in events if r.get("kind") == "control_end"), {})
    controlled_end = control_end.get("t_s", origin_s + queue.get("duration_s", 0))
    execution_complete = (
        bool(ticks)
        and sorted(r["tick"] for r in ticks) == list(range(len(ticks)))
        and control_end.get("control_steps", len(ticks)) == len(ticks)
    )
    for tick in ticks:
        slot = tick.get("executed_slot") or {}
        kind, producer = slot.get("producer_kind"), slot.get("producer_request_tick")
        if kind not in ("bootstrap", "fallback", "request") or (
            kind == "request" and type(producer) is not int
        ):
            execution_complete = False
        if kind == "request" and type(producer) is int:
            executions.setdefault(producer, []).append(
                {
                    "tick": tick["tick"],
                    "applied_s": tick.get("applied_s", tick["t_s"]) - origin_s,
                }
            )

    def relative(value):
        from .schema import optional_number

        value = optional_number(value)
        return value - origin_s if value is not None else None

    snapshots = []
    for request in queue.get("requests", []):
        if request.get("role") != "S1":
            continue
        key = request.get("request_tick")
        result, read, submission = (
            results.get(key, {}),
            adopted.get(key, {}),
            submitted.get(key, {}),
        )
        audit, timing = result.get("audit", {}), result.get("client_timing", {})
        unused = result.get("kind") == "unused_result_at_episode_end"
        counters = {}
        for field in ("installed_slots", "expired_slots", "protected_slots"):
            value = _consistent(
                [result.get(field), read.get(field)], "publication " + field
            )
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Invalid publication counter")
            counters[field] = value
        installed = counters["installed_slots"]
        status = (
            "unused_final"
            if unused
            else "unknown"
            if installed is None
            else "installed"
            if installed > 0
            else "discarded"
        )
        processed = (
            None
            if unused
            else relative(
                result.get("controller_processed_s", result.get("t_s", read.get("t_s")))
            )
        )
        actual = sorted(executions.get(key, []), key=lambda r: r["tick"])
        prefix = {
            "start_tick": submission.get("committed_start_tick"),
            "end_tick": submission.get("committed_end_tick"),
        }
        request["publication"] = {
            **counters,
            "status": status,
            "unused_at_s": relative(result.get("t_s")) if unused else None,
            "control_end_s": relative(control_end.get("t_s")),
            "processed_s": processed,
            "controller_ready_observed_s": relative(
                result.get("controller_ready_observed_s")
            ),
            "rpc_completed_s": relative(timing.get("completed_s")),
            "next_unexecuted_tick": result.get(
                "consumed_at_tick", read.get("adopted_at_tick")
            ),
            "valid_start_tick": audit.get(
                "valid_start_tick", read.get("valid_start_tick")
            ),
            "valid_end_tick": audit.get("valid_end_tick", read.get("valid_end_tick")),
            "committed_prefix": prefix
            if all(v is not None for v in prefix.values())
            else None,
            "executed_slots": len(actual) if execution_complete else None,
            "executions": actual,
            "execution_evidence_complete": execution_complete,
            "delay_policy_decision": result.get("delay_policy_decision"),
            "basis": "raw request_completed/action_adopted counters; exact control_tick.executed_slot producer",
        }
        if result.get("action_buffer") is not None and processed is not None:
            snapshots.append(
                {
                    "t": processed,
                    "request_tick": key,
                    "next_unexecuted_tick": result.get("consumed_at_tick"),
                    "action_buffer": result["action_buffer"],
                    "phase": result.get("action_buffer_phase"),
                    "basis": "recorded controller publication snapshot",
                }
            )
    queue["publication_snapshots"] = sorted(snapshots, key=lambda r: r["t"])
    decisions = [
        {**r, "t": r["t_s"] - origin_s}
        for r in events
        if r.get("kind") == "delay_policy_decision"
    ]
    queue["delay_policy_decisions"] = sorted(decisions, key=lambda r: r["t"])
    if decisions:
        queue["delay_transitions"] = [
            {**r, "before": r["previous_delay_ticks"], "after": r["chosen_delay_ticks"]}
            for r in queue["delay_policy_decisions"]
        ]
    # The robot stops at control_end. Completed terminal RPCs retain their real
    # wall endpoints in a separate drain tail; controlled metrics are unchanged.
    completed = [
        relative(r.get("completed_s"))
        for r in events
        if r.get("kind") in ("vision_request", "action_completed")
    ] + [relative(r.get("client_timing", {}).get("completed_s")) for r in result_rows]
    work_end = max(
        [
            queue.get("work_end_s", controlled_end),
            *[origin_s + value for value in completed if value is not None],
        ]
    )
    queue.update(
        work_end_s=work_end,
        work_duration_s=work_end - origin_s,
        terminal_drain_s=max(0, work_end - controlled_end),
    )
