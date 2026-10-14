"""Actual independent model calls and separately instrumented Chrome traces."""

import re
from collections import defaultdict
from pathlib import Path

from coexecution.libero.profiling import _union_duration, classify_trace

from .schema import (
    PRECISIONS,
    VERSION,
    comparison_key,
    digest,
    interval,
    optional_number,
    read_json,
    read_jsonl,
    source,
)


def _gate_status(values):
    decisions = []

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("passes", "rng_equal") and type(item) is bool:
                    decisions.append(item)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(values)
    return "failed" if False in decisions else "passed" if decisions else "unavailable"


def _shared_config(result, mode):
    arguments = result.get("arguments")
    checkpoint = result.get("checkpoint", {}).get("sha256")
    device = result.get("device")
    if (
        not arguments
        or not checkpoint
        or not device
        or not result.get("runtime_sources")
    ):
        return None
    # Quantized math/packing choices are intentionally precision-specific. Shared
    # execution boundaries and host/runtime options must remain matched.
    fields = (
        "measurement_boundary",
        "torch_threads",
        "operator_fusion",
        "dit_cuda_graph",
        "action_cuda_graph",
        "vision_channels_last",
        "cuda_events",
    )
    if any(name not in arguments for name in fields):
        return None
    common = {name: arguments[name] for name in fields}
    return digest(
        {
            "checkpoint": checkpoint,
            "mode": mode.split("-", 1)[1],
            "common": common,
            "device": device,
            "sources": result["runtime_sources"],
            "model_options": result.get("model_options"),
        }
    )


def model_run_id(root, label):
    """Share the persisted identity with live journals before a model is loaded."""
    return "model-" + digest([str(Path(root).resolve()), label])[:20]


def clean_request(row, run_id, output, *, pair_key=None):
    """Normalize one real clean invocation, shared by live and archive import."""
    return {
        **row,
        "id": model_request_id(run_id, "clean", row["request_id"]),
        "clean_wall": row.get("instrumented") is False,
        "comparison_key": pair_key,
        "output": output,
        "difference_from_bf16": None,
    }


def model_request_id(run_id, collection, request_id):
    return f"{run_id}/{collection}/{request_id}"


def encode_action_arrays(arrays, *, allow_nonfinite=False):
    """Normalize actual CPU action arrays; used by live and archived evidence."""
    import numpy as np

    output = {}
    for key, value in arrays.items():
        if value.dtype.kind not in "fiu":
            raise ValueError(f"Invalid model action array: {key}")
        finite = np.isfinite(value)
        if not finite.all():
            if not allow_nonfinite:
                raise ValueError(f"Invalid model action array: {key}")
            output[key] = {
                "shape": list(value.shape),
                "values": None,
                "status": "nonfinite",
                "nonfinite_count": int(value.size - finite.sum()),
            }
        else:
            output[key] = {"shape": list(value.shape), "values": value.tolist()}
    return output


def _arrays(path, *, allow_nonfinite=False):
    if not path.exists():
        return {}
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return encode_action_arrays(
            {key: archive[key] for key in archive.files},
            allow_nonfinite=allow_nonfinite,
        )


def _output(arrays, request):
    prefix = request + "/"
    actions = {
        key.removeprefix(prefix): value
        for key, value in arrays.items()
        if key.startswith(prefix)
    }
    return (
        {
            "status": "invalid"
            if any(value.get("status") == "nonfinite" for value in actions.values())
            else "recorded",
            "actions": actions,
            "axis": "decoded_action_horizon",
            "units": "model_decoded_action_fields",
        }
        if actions
        else {
            "status": "unavailable",
            "reason": "No retained action arrays for this exact invocation",
        }
    )


def _profiles(root, label, metadata, observations, run_id):
    if not metadata:
        return [], [], [], []
    path = root / "profiles" / f"{label}.trace.json"
    if not metadata.get("requests"):
        return (
            [],
            [],
            [source(path)] if path.is_file() else [],
            [
                "Profile aggregate has no per-request annotations; module timeline is unavailable. Aggregate measurements remain in profile_summary."
            ],
        )
    if not path.is_file():
        return (
            [],
            [],
            [],
            ["Raw profile trace unavailable; aggregate durations are not a timeline"],
        )
    trace = read_json(path)
    for event in trace.get("traceEvents", []):
        if event.get("ph") == "X":
            start, duration = (
                optional_number(event.get("ts")),
                optional_number(event.get("dur")),
            )
            if start is None or duration is None or duration < 0:
                raise ValueError("Invalid actual profile timestamp or duration")
    classified = classify_trace(trace)
    scoped = {}
    for row in classified["request_scopes"]:
        if row["request_id"] in scoped:
            raise ValueError("Duplicate host profile request identity")
        scoped[row["request_id"]] = row
    by_request = defaultdict(list)
    kernels, lookup = [], {}
    for row in classified["device_events"]:
        name = row["kernel"]
        if name not in lookup:
            lookup[name] = len(kernels)
            kernels.append(name)
        by_request[row["request_id"]].append(row)
    result, seen = [], set()
    for row in metadata.get("requests", []):
        key = row["request_id"]
        if key in seen:
            raise ValueError("Duplicate profile request metadata")
        seen.add(key)
        scope = scoped.get(key)
        if row.get("observation_id") not in observations:
            raise ValueError("Profile observation identity is unknown")
        if scope is None:
            result.append(
                {
                    **row,
                    "id": model_request_id(run_id, "profile", key),
                    "instrumented": True,
                    "timeline_status": "unavailable",
                    "timeline": [],
                    "host_wall_ms": None,
                    "device_span_ms": None,
                    "gpu_work_ms": None,
                    "gpu_active_union_ms": None,
                    "output": {
                        "status": "unavailable",
                        "reason": "This instrumented invocation did not retain outputs",
                    },
                }
            )
            continue
        anchor = scope["ts_us"]
        device = sorted(by_request[key], key=lambda event: event["ts_us"])
        intervals = [
            (event["ts_us"], event["ts_us"] + event["duration_us"]) for event in device
        ]
        result.append(
            {
                **row,
                "id": model_request_id(run_id, "profile", key),
                "instrumented": True,
                "timeline_status": "recorded",
                "trace_anchor_us": anchor,
                "host_wall_ms": scope["duration_us"] / 1000,
                "device_span_ms": (
                    max(stop for _, stop in intervals)
                    - min(start for start, _ in intervals)
                )
                / 1000
                if intervals
                else None,
                "gpu_work_ms": sum(event["duration_us"] for event in device) / 1000
                if device
                else None,
                "gpu_active_union_ms": _union_duration(intervals) / 1000
                if intervals
                else None,
                "timeline": [
                    {
                        "start_ms": (event["ts_us"] - anchor) / 1000,
                        "duration_ms": event["duration_us"] / 1000,
                        "stage": event["stage"],
                        "category": event["category"],
                        "kernel_id": lookup[event["kernel"]],
                        "stream": event["stream"],
                        "device": event["device"],
                    }
                    for event in device
                ],
                "output": {
                    "status": "unavailable",
                    "reason": "This instrumented invocation did not retain outputs; clean samples have different seeds and are never substituted",
                },
            }
        )
    warnings = []
    unlinked = sum(len(values) for key, values in by_request.items() if key not in seen)
    if unlinked:
        warnings.append(
            f"{unlinked} device activities lack unambiguous request correlation and are excluded from per-request lanes"
        )
    return result, kernels, [source(path)], warnings


def _load_evidence(root):
    """Successful results are authoritative; a failed run retains only sidecars."""
    completed = (root / "results.json").is_file()
    result = read_json(root / "results.json") if completed else {}
    provenance = [source(root / "results.json")] if completed else []
    warnings = (
        []
        if completed
        else [
            "No completed results.json; importing retained failure/incomplete sidecars. Unrecorded timing, configuration and comparison identity remain unknown."
        ]
    )
    samples = root / "samples.jsonl"
    rows = read_jsonl(samples) if samples.is_file() else []
    if samples.is_file():
        provenance.append(source(samples))
    for filename, key in (
        ("validation.json", "validation"),
        ("graph-timing-audit.json", "graph_timing_audit"),
        ("observations.json", "observations"),
        ("profiles.json", "profiles"),
    ):
        path = root / filename
        if path.is_file():
            provenance.append(source(path))
            data = read_json(path)
            if not completed:
                result[key] = data
            elif key in ("validation", "graph_timing_audit") and data != result.get(
                key
            ):
                warnings.append(
                    f"Legacy {filename} disagrees with results.json; results.json is authoritative and both sources are retained"
                )
    return result, rows, completed, provenance, warnings


def _retained_requests(arrays):
    """Archive keys carry actual invocation identity even when no timing row saved."""
    requests = {}
    for key in arrays:
        match = re.fullmatch(r"(r([0-9]+)/([^/]+)/o([0-9]+)/([^/]+))/.+", key)
        if match is not None:
            request, number, label, observation, scope = match.groups()
            requests[request] = {
                "request_id": request,
                "round": int(number),
                "mode": label,
                "observation": int(observation),
                "scope": scope,
            }
    return requests


def _normalize_legacy_identity(result, rows):
    """Derive labels only from stored hashes/indexes and canonical sample fields."""
    identities, normalized, derived = [], [], False
    for index, original in enumerate(result.get("observations", [])):
        identity = dict(original)
        if "observation_id" not in identity:
            sha = identity.get("sha256")
            if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
                raise ValueError(
                    "Legacy observation identity requires its persisted SHA-256"
                )
            identity["observation_id"] = f"obs-{index}-{sha[:16]}"
            identity["identity_derivation"] = {
                "observation_id": "Persisted observations list index and SHA-256"
            }
            derived = True
        identities.append(identity)
    for original in rows:
        row, basis = dict(original), {}
        label, index = row.get("mode", ""), row.get("observation")
        if "observation_id" not in row:
            if type(index) is not int or not 0 <= index < len(identities):
                raise ValueError(
                    "Legacy sample requires a valid persisted observation index"
                )
            row["observation_id"] = identities[index]["observation_id"]
            basis["observation_id"] = "Persisted sample observation index"
        if "request_id" not in row:
            number, scope = row.get("round"), row.get("scope")
            if (
                type(index) is not int
                or index < 0
                or type(number) is not int
                or number < 0
                or not isinstance(scope, str)
                or not re.fullmatch(r"[a-z_]+", scope)
            ):
                raise ValueError(
                    "Legacy request identity requires round/observation/scope"
                )
            row["request_id"] = f"r{number}/{label}/o{index}/{scope}"
            basis["request_id"] = (
                "Persisted round/mode/observation/scope; canonical archive key"
            )
        if "precision" not in row:
            row["precision"] = label.split("-", 1)[0]
            basis["precision"] = "Persisted mode label prefix"
        if "instrumented" not in row:
            row["timing_status"] = "instrumentation_unknown"
        if basis:
            row["identity_derivation"] = basis
            derived = True
        normalized.append(row)
    return identities, normalized, derived


def model_runs(root, media, *, clock_cohort=None):
    root = Path(root).resolve()
    result, rows, completed, provenance, warnings = _load_evidence(root)
    identities, rows, derived = _normalize_legacy_identity(result, rows)
    if derived:
        warnings.append(
            "Legacy labels were derived from persisted observation hashes/indexes and sample fields; each derivation is marked. Missing seeds, timing, input assets and execution settings remain unknown."
        )
    observations = {row["observation_id"]: row for row in identities}
    if len(observations) != len(identities):
        raise ValueError("Duplicate observation identity")
    if not completed:
        for row in rows:
            identity = row.get("observation_id")
            if identity is not None and identity not in observations:
                observations[identity] = {"observation_id": identity, "sha256": None}
    assets_path = root / "observations-replay.json"
    if assets_path.exists():
        provenance.append(source(assets_path))
        for row in read_json(assets_path):
            identity = observations.get(row.get("observation_id"))
            if identity is None or identity.get("sha256") != row.get("sha256"):
                raise ValueError(
                    "Replay asset observation hash differs from measured input"
                )
            copied = []
            for asset in row.get("assets", []):
                copied.append(
                    {
                        "camera": asset["camera"],
                        "frame_index": asset["frame_index"],
                        "media_id": media.add(root, asset["path"], mime=asset["mime"]),
                    }
                )
            observations[row["observation_id"]] = {
                **identity,
                "assets": copied,
                "state": row.get("state"),
            }
    arrays_path = root / "actions.npz"
    arrays = _arrays(arrays_path, allow_nonfinite=not completed)
    if arrays_path.exists():
        provenance.append(source(arrays_path))
    groups, seen = defaultdict(list), set()
    for row in rows:
        label, precision = row.get("mode", ""), row.get("precision")
        if (
            precision not in PRECISIONS
            or not label.startswith(precision + "-")
            or Path(label).name != label
        ):
            raise ValueError("Model precision and mode identity disagree")
        if row.get("request_id") in seen:
            raise ValueError("Duplicate clean request identity")
        seen.add(row.get("request_id"))
        if not isinstance(row.get("request_id"), str) or not row["request_id"]:
            raise ValueError("Explicit measured request identity is required")
        if row.get("observation_id") not in observations:
            raise ValueError("Unknown clean observation identity")
        for name in ("wall_ms", "host_return_ms", "cuda_span_ms"):
            value = optional_number(row.get(name))
            if value is not None:
                interval(0, value)
        if row.get("wall_ms") is None:
            raise ValueError("Measured request requires wall duration")
        groups[label].append(row)
    retained_requests = _retained_requests(arrays)
    if not completed:
        for row in [
            *result.get("validation", []),
            *result.get("graph_timing_audit", []),
            *retained_requests.values(),
        ]:
            label = row.get("mode", "")
            precision = label.split("-", 1)[0]
            if (
                precision not in PRECISIONS
                or not label.startswith(precision + "-")
                or Path(label).name != label
            ):
                raise ValueError("Failed evidence precision and mode identity disagree")
            groups.setdefault(label, [])
    runs = []
    for label, samples in groups.items():
        run_id = model_run_id(root, label)
        config_sha = _shared_config(result, label)
        clean = []
        for row in samples:
            identity = observations[row["observation_id"]]
            pair_config = (
                digest([config_sha, row.get("scope"), row.get("instrumented")])
                if config_sha is not None and type(row.get("instrumented")) is bool
                else None
            )
            clean.append(
                clean_request(
                    row,
                    run_id,
                    _output(arrays, row["request_id"]),
                    pair_key=comparison_key(
                        kind="model",
                        observation_sha256=identity.get("sha256"),
                        seed=row.get("seed"),
                        shared_config_sha256=pair_config,
                        clock_cohort=clock_cohort,
                    ),
                )
            )
        profiles, kernels, trace_sources, profile_warnings = _profiles(
            root, label, result.get("profiles", {}).get(label), observations, run_id
        )
        for request in profiles:
            request["comparison_key"] = comparison_key(
                kind="model",
                observation_sha256=observations[request["observation_id"]].get(
                    "sha256"
                ),
                seed=request.get("seed"),
                shared_config_sha256=digest([config_sha, "instrumented_profile"])
                if config_sha is not None
                else None,
                clock_cohort=clock_cohort,
            )
        gates = [
            row for row in result.get("validation", []) if row.get("mode") == label
        ]
        audits = [
            row
            for row in result.get("graph_timing_audit", [])
            if row.get("mode") == label
        ]
        retained_outputs = []
        sampled_ids = {row["request_id"] for row in clean}
        for key, retained in retained_requests.items():
            if retained["mode"] != label or key in sampled_ids:
                continue
            index = retained["observation"]
            identity = (
                identities[index].get("observation_id")
                if index < len(identities)
                else None
            )
            retained_outputs.append(
                {
                    **retained,
                    "id": f"{run_id}/retained/{key}",
                    "observation_id": identity,
                    "seed": None,
                    "wall_ms": None,
                    "instrumented": None,
                    "comparison_key": None,
                    "output": _output(arrays, key),
                    "timing_status": "unavailable",
                }
            )
        status = (
            "failed"
            if "failed" in (_gate_status(gates), _gate_status(audits))
            else "complete"
            if completed
            else "incomplete"
        )
        runs.append(
            {
                "schema": "pir2.replay.run",
                "version": VERSION,
                "id": run_id,
                "kind": "model",
                "label": root.parent.name + "/" + root.name + "/" + label,
                "precision": label.split("-", 1)[0],
                "mode": label,
                "status": status,
                "source_root": str(root),
                "clock_cohort": clock_cohort,
                "clock_cohort_basis": "explicit builder operator declaration"
                if clock_cohort
                else "unknown",
                "shared_config_sha256": config_sha,
                "same_input_comparison": bool(clean)
                and all(row["comparison_key"] is not None for row in clean),
                "boundaries": result.get("boundaries", {}),
                "configuration": result.get("arguments", {}),
                "checkpoint": result.get("checkpoint"),
                "device": result.get("device"),
                "mode_evidence": result.get("mode_evidence", {}).get(label),
                "runtime_receipt": (
                    result.get("mode_evidence", {}).get(label) or {}
                ).get("runtime_receipt"),
                "observations": list(observations.values()),
                "clean_requests": clean,
                "retained_outputs": retained_outputs,
                "profile_requests": profiles,
                "profile_summary": {
                    key: value
                    for key, value in result.get("profiles", {}).get(label, {}).items()
                    if key not in ("kernels", "requests")
                },
                "kernels": kernels,
                "validation": {
                    "status": _gate_status(gates),
                    "gates": gates,
                    "graph_status": _gate_status(audits),
                    "graph_audits": audits,
                },
                "provenance": provenance + trace_sources,
                "warnings": warnings + profile_warnings,
                "timing_note": "Clean wall, instrumented host wall, CUDA span and leaf GPU activity are distinct nonadditive measurements",
            }
        )
    return runs


def compare_outputs(runs):
    """Numerical debugging only; not an assertion of task-quality equivalence."""
    references = defaultdict(list)
    for run in runs:
        if run["kind"] == "model" and run["precision"] == "bf16":
            for row in run["clean_requests"]:
                if row["comparison_key"] is not None:
                    references[row["comparison_key"]].append(row)

    def flat(values):
        if isinstance(values, list):
            for item in values:
                yield from flat(item)
        else:
            yield values

    for run in runs:
        if run["kind"] != "model":
            continue
        for row in run["clean_requests"]:
            matches = references.get(row["comparison_key"], [])
            # Multiple identical seeded rounds are valid; select deterministically
            # only when all retained reference outputs agree.
            if not matches or any(
                other["output"] != matches[0]["output"] for other in matches[1:]
            ):
                continue
            reference = matches[0]
            left, right = (
                reference["output"].get("actions"),
                row["output"].get("actions"),
            )
            if (
                not left
                or not right
                or left.keys() != right.keys()
                or any(
                    left[key]["values"] is None or right[key]["values"] is None
                    for key in left
                )
                or any(left[key]["shape"] != right[key]["shape"] for key in left)
            ):
                continue
            fields, maximum = {}, 0.0
            for key in left:
                differences = [
                    candidate - base
                    for base, candidate in zip(
                        flat(left[key]["values"]), flat(right[key]["values"])
                    )
                ]
                maximum = max(maximum, max(map(abs, differences), default=0.0))
                fields[key] = {"shape": right[key]["shape"], "delta_flat": differences}
            row["difference_from_bf16"] = {
                "reference_request_id": reference["id"],
                "max_abs": maximum,
                "fields": fields,
                "scope": "Same recorded observation, seed, shared configuration and declared clock cohort; numerical difference only",
            }
