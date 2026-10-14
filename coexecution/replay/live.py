"""Explicit model recording into a local replay catalog while inference runs.

This module is an opt-in producer, never an HTTP handler. Model loading remains
in the existing measurement runner. Host stage events describe host submission;
only the final imported correlated profile supplies GPU kernel timestamps.
"""

import copy
import fcntl
import json
import math
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .media import MediaRegistry
from .model import (
    clean_request,
    compare_outputs,
    model_request_id,
    model_run_id,
    model_runs,
)
from .producer import HOST_CLOCK
from .schema import VERSION, read_json
from .stream import BoundedEventWriter, ReplayJournal

EVENT_BYTES = 8 * 1024 * 1024
PENDING_BYTES = 16 * 1024 * 1024


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(
                (
                    json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n"
                ).encode()
            )
            stream.flush()
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _occurrence():
    return {**HOST_CLOCK, "value": time.monotonic()}


def _host_elapsed_ms(start, occurrence):
    """Compare only explicit producer timestamps on the same host clock."""
    if any(
        value.get(key) != expected
        for value in (start, occurrence)
        for key, expected in HOST_CLOCK.items()
    ):
        return None
    values = [value.get("value") for value in (start, occurrence)]
    if any(
        type(value) not in (int, float) or not math.isfinite(value) for value in values
    ):
        return None
    elapsed = (values[1] - values[0]) * 1000
    return elapsed if elapsed >= 0 else None


def _initial_run(args, label, cohort):
    return {
        "schema": "pir2.replay.run",
        "version": VERSION,
        "id": model_run_id(args.output, label),
        "kind": "model",
        "label": args.output.parent.name + "/" + args.output.name + "/" + label,
        "precision": label.split("-", 1)[0],
        "mode": label,
        "status": "running",
        "source_root": str(args.output.resolve()),
        "clock_cohort": cohort,
        "clock_cohort_basis": "explicit recording operator declaration"
        if cohort
        else "unknown",
        "shared_config_sha256": None,
        "same_input_comparison": False,
        "boundaries": {},
        "configuration": {},
        "checkpoint": None,
        "device": None,
        "mode_evidence": None,
        "runtime_receipt": None,
        "observations": [],
        "clean_requests": [],
        "retained_outputs": [],
        "profile_requests": [],
        "profile_summary": {},
        "kernels": [],
        "provenance": [],
        "validation": {
            "status": "pending",
            "gates": [],
            "graph_status": "pending",
            "graph_audits": [],
        },
        "warnings": [
            "Recording in progress; runtime validation and final profile evidence are pending"
        ],
        "timing_note": "Live stage events are host submission spans; GPU kernel timestamps require the final correlated profile",
    }


class _Recording:
    def __init__(self, args, output, cohort):
        self.args, self.output, self.cohort = args, output, cohort
        self.media = MediaRegistry(output)
        path = output / "catalog.json"
        self.catalog = (
            read_json(path)
            if path.exists()
            else {
                "schema": "pir2.replay.catalog",
                "version": VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "runs": [],
                "streams": [],
                "media_manifest": "media.json",
                "clock_cohort": cohort,
                "default_view": "model",
                "default_record_type": "profile",
                "default_execution_clock": None,
                "notes": [
                    "Explicit model recording; opening the replay page never starts inference"
                ],
            }
        )
        if (self.catalog.get("schema"), self.catalog.get("version")) != (
            "pir2.replay.catalog",
            VERSION,
        ):
            raise ValueError("Existing output is not a replay catalog")
        if self.catalog.get("media_manifest") != "media.json":
            raise ValueError("Live registration requires the standard media manifest")
        if (output / "media.json").exists():
            manifest = read_json(output / "media.json")
            self.media.assets = {row["id"]: row for row in manifest["assets"]}
        self.runs = {
            f"{precision}-{args.modes[0]}": _initial_run(
                args, f"{precision}-{args.modes[0]}", cohort
            )
            for precision in args.precisions
        }
        existing = {row["id"] for row in self.catalog["runs"]}
        if existing.intersection(run["id"] for run in self.runs.values()):
            raise ValueError("Model run is already registered")
        self.writers = {}
        self.pending_observations = {}
        self.profile_invocations = {}
        try:
            for label, run in self.runs.items():
                journal = ReplayJournal(
                    output / "streams",
                    run,
                    {"host-monotonic": {"unit": "s", "domain": "host_monotonic"}},
                    max_event_bytes=EVENT_BYTES,
                )
                self.writers[label] = BoundedEventWriter(
                    journal, max_pending_bytes=PENDING_BYTES
                )
                self.catalog.setdefault("streams", []).append(
                    journal.registration(output)
                )
            self.persist()
        except BaseException:
            for writer in self.writers.values():
                writer.close()
            raise

    def persist(self):
        self._media_manifest()
        entries = {row["id"]: row for row in self.catalog["runs"]}
        for run in self.runs.values():
            path = Path("runs") / (run["id"] + ".json")
            _atomic_json(self.output / path, run)
            entries[run["id"]] = {
                key: run[key] for key in ("id", "kind", "label", "precision", "mode")
            }
            entries[run["id"]].update(
                path=str(path),
                clean_requests=len(run["clean_requests"]),
                profile_requests=len(run["profile_requests"]),
                episodes=0,
                warnings=run["warnings"],
            )
        self.catalog["runs"] = list(entries.values())
        _atomic_json(self.output / "catalog.json", self.catalog)

    def _media_manifest(self):
        _atomic_json(
            self.output / "media.json",
            {
                "schema": "pir2.replay.media",
                "version": VERSION,
                "assets": list(self.media.assets.values()),
            },
        )

    def _profile_event(self, label, kind, occurrence, sources, payload):
        run = self.runs[label]
        key = (label, sources.get("request_id"))
        if kind == "model.request.started" and payload.get("instrumented") is True:
            if key in self.profile_invocations:
                raise ValueError("Duplicate live profile request identity")
            request = sources["request_id"]
            row = {
                "id": model_request_id(run["id"], "profile", request),
                "request_id": request,
                "observation_id": sources["observation_id"],
                "mode": label,
                "seed": payload.get("seed"),
                "instrumented": True,
                # This follows collect_model_profile's actual input path;
                # its non-model-only path excludes output decoding.
                "scope": "prepared_model"
                if self.args.measurement_boundary == "model-only"
                else "preparation_model",
                "live_status": "started",
                "host_started": copy.deepcopy(occurrence),
                "host_progress_ms": _host_elapsed_ms(occurrence, occurrence),
                "stage_spans": [],
                "timeline_status": "unavailable",
                "timeline": [],
                "comparison_key": None,
                "host_wall_ms": None,
                "device_span_ms": None,
                "gpu_work_ms": None,
                "gpu_active_union_ms": None,
                "output": {
                    "status": "unavailable",
                    "reason": "This instrumented invocation did not retain outputs",
                },
            }
            self.profile_invocations[key] = row
            run["profile_requests"].append(row)
            return True
        row = self.profile_invocations.get(key)
        if row is None or kind not in (
            "model.stage.started",
            "model.stage.submitted",
            "model.stage.failed",
            "model.request.completed",
        ):
            return False
        elapsed = _host_elapsed_ms(row["host_started"], occurrence)
        if elapsed is not None:
            row["host_progress_ms"] = max(row["host_progress_ms"] or 0, elapsed)
        if kind == "model.request.completed":
            row.update(
                live_status="cuda_completion_observed",
                host_completed=copy.deepcopy(occurrence),
            )
            # An observed request completion cannot recover a missing stage end.
            for span in row["stage_spans"]:
                if span["status"] == "running":
                    span["status"] = "incomplete"
        else:
            stage = payload.get("stage")
            if (
                kind == "model.stage.started"
                and payload.get("timing_boundary") == "host_submission"
                and elapsed is not None
            ):
                row["stage_spans"].append(
                    {
                        "stage": stage,
                        "start_ms": elapsed,
                        "end_ms": None,
                        "domain": "host_submit",
                        "clock_id": occurrence["clock_id"],
                        "status": "running",
                        "feature_id": sources.get("feature_id"),
                    }
                )
            elif kind in ("model.stage.submitted", "model.stage.failed"):
                span = next(
                    (
                        span
                        for span in reversed(row["stage_spans"])
                        if span["stage"] == stage and span["status"] == "running"
                    ),
                    None,
                )
                if span is not None and (
                    kind == "model.stage.failed"
                    or payload.get("timing_boundary") == "host_submission"
                ):
                    span["status"] = (
                        "failed" if kind == "model.stage.failed" else "submitted"
                    )
                    if elapsed is not None and elapsed >= span["start_ms"]:
                        span["end_ms"] = elapsed
                if kind == "model.stage.failed":
                    row.update(
                        live_status="failed", host_failed=copy.deepcopy(occurrence)
                    )
        return True

    def emit(self, kind, *, occurrence, sources=None, payload=None):
        sources, payload = sources or {}, payload or {}
        label = payload.get("mode")
        if label is not None and label not in self.writers:
            raise ValueError("Live event mode does not belong to this recording")
        if label is None and not kind.startswith("observation."):
            raise ValueError("Model events require their actual precision/mode")
        labels = [label] if label is not None else list(self.writers)
        observation = None
        if kind == "observation.loaded":
            observation = {
                key: value
                for key, value in payload.items()
                if key not in ("timestamp_boundary", "live_camera_capture")
            }
            self.pending_observations[observation["observation_id"]] = observation
            if self.args.replay_assets:
                observation = None
        elif kind == "observation.assets_ready":
            observation = dict(payload["observation"])
            original = self.pending_observations.get(observation["observation_id"])
            if original is None or original.get("sha256") != observation.get("sha256"):
                raise ValueError(
                    "Live media identity differs from the actual loaded observation"
                )
            observation["assets"] = [
                {
                    "camera": row["camera"],
                    "frame_index": row["frame_index"],
                    "media_id": self.media.add(
                        self.args.output, row["path"], mime=row["mime"]
                    ),
                }
                for row in observation.get("assets", [])
            ]
            payload = {"observation": observation}
            self._media_manifest()
        accepted = True
        for current in labels:
            run, projection = self.runs[current], None
            if observation is not None:
                run["observations"].append(copy.deepcopy(observation))
                projection = {
                    "op": "append",
                    "collection": "observations",
                    "items": [observation],
                }
            elif kind == "model.request.completed" and "sample" in payload:
                row = clean_request(payload["sample"], run["id"], payload["output"])
                run["clean_requests"].append(copy.deepcopy(row))
                projection = {
                    "op": "append",
                    "collection": "clean_requests",
                    "items": [row],
                }
            elif self._profile_event(current, kind, occurrence, sources, payload):
                projection = {"op": "replace", "run": run}
            elif kind == "runtime.receipt":
                run["runtime_receipt"] = copy.deepcopy(payload["runtime_receipt"])
                projection = {"op": "replace", "run": run}
            admitted = self.writers[current].emit(
                kind,
                occurrence=occurrence,
                sources=sources,
                payload=payload,
                projection=projection,
            )
            accepted = admitted and accepted
        return accepted

    def finish(self, *, error=None, attest=None):
        # This finalization runs after model timing. The persisted canonical
        # normalizer is authoritative even when intermediate events were dropped.
        if error is None:
            imported = model_runs(
                self.args.output, self.media, clock_cohort=self.cohort
            )
            if {row["mode"] for row in imported} != set(self.runs):
                raise ValueError(
                    "Completed measurement did not retain every registered mode"
                )
            compare_outputs(imported)
            for run in imported:
                if attest is not None:
                    original = run.get("runtime_receipt")
                    if run["status"] != "complete" or original is None:
                        raise ValueError(
                            "Runtime attestation requires completed evidence and a receipt"
                        )
                    identity = original.get("identity_sha256")
                    receipt = attest(copy.deepcopy(run))
                    if (
                        not identity
                        or not isinstance(receipt, dict)
                        or {
                            key: value
                            for key, value in receipt.items()
                            if key != "validation"
                        }
                        != {
                            key: value
                            for key, value in original.items()
                            if key != "validation"
                        }
                    ):
                        raise ValueError(
                            "Runtime attestation must preserve the exact measured identity"
                        )
                    validation = receipt.get("validation") or {}
                    if (
                        validation.get("status") != "gpu_verified"
                        or validation.get("runtime_identity_sha256") != identity
                        or re.fullmatch(
                            r"[a-f0-9]{64}", str(validation.get("evidence_sha256", ""))
                        )
                        is None
                    ):
                        raise ValueError(
                            "Runtime attestation requires an exact identity GPU gate evidence SHA256"
                        )
                    run["runtime_receipt"] = receipt
                    run["mode_evidence"]["runtime_receipt"] = receipt
                self.runs[run["mode"]] = run
        else:
            for (label, _request), row in self.profile_invocations.items():
                run = self.runs[label]
                if row["live_status"] not in ("cuda_completion_observed", "failed"):
                    row["live_status"] = "failed"
                    for span in row["stage_spans"]:
                        if span["status"] == "running":
                            span["status"] = "incomplete"
                if row["id"] not in {item["id"] for item in run["profile_requests"]}:
                    run["profile_requests"].append(copy.deepcopy(row))
            # Failed gates can retain outputs before a timed row is admitted.
            # Reuse the archive importer so these exact invocations remain
            # visible, with missing latency/configuration explicitly unknown.
            try:
                partial = model_runs(
                    self.args.output, self.media, clock_cohort=self.cohort
                )
            except (ValueError, OSError, KeyError) as artifact_error:
                partial = []
                for run in self.runs.values():
                    run["warnings"].append(
                        f"Partial archive unavailable: {artifact_error}"
                    )
            for archived in partial:
                previous = self.runs.get(archived["mode"])
                if previous is None:
                    continue
                for collection, key in (
                    ("observations", "observation_id"),
                    ("clean_requests", "id"),
                    ("profile_requests", "id"),
                ):
                    records = {row[key]: row for row in previous[collection]}
                    for row in archived[collection]:
                        # An archive without actions.npz must not erase an
                        # actual decoded output already emitted by the runner.
                        if (
                            collection == "clean_requests"
                            and row["output"]["status"] == "unavailable"
                            and row[key] in records
                        ):
                            row = {**row, "output": records[row[key]]["output"]}
                        records[row[key]] = row
                    archived[collection] = list(records.values())
                if archived["runtime_receipt"] is None:
                    archived["runtime_receipt"] = previous["runtime_receipt"]
                self.runs[archived["mode"]] = archived
            for run in self.runs.values():
                known = {row["observation_id"] for row in run["observations"]}
                run["observations"].extend(
                    copy.deepcopy(row)
                    for key, row in self.pending_observations.items()
                    if key not in known
                )
                run["status"] = "failed"
                run["warnings"].append(
                    "Recording failed; only actually observed events and outputs are retained"
                )
        self.persist()
        for label, writer in self.writers.items():
            writer.flush()
            if not writer.emit(
                "run.canonical",
                occurrence=_occurrence(),
                payload={"authoritative": error is None},
                projection={"op": "replace", "run": self.runs[label]},
            ):
                raise RuntimeError(
                    "Canonical replay evidence exceeds the declared recording capacity"
                )

    def close(self, *, error=None):
        failures = []
        occurrence = _occurrence()
        for writer in self.writers.values():
            try:
                writer.close(
                    status="failed" if error is not None else "ended",
                    occurrence=occurrence,
                    payload={"error": repr(error)} if error is not None else {},
                )
            except BaseException as failure:  # noqa: BLE001 - close every writer before propagating
                failures.append(failure)
        if failures and error is None:
            raise failures[0]


def record_model(model_args, *, catalog_output, clock_cohort=None, attest=None):
    """Manually run one measurement mode and publish bounded actual evidence.

    ``attest`` is optional and runs after completed canonical evidence exists.
    It must return a receipt with the same measured identity after an external
    exact-identity gate; omission leaves the runtime validation status pending.
    No GPU selection or authorization is implied by catalog/page access.
    """
    from coexecution.libero import measurement

    # This explicit recording entry point retains source/operator identities;
    # ordinary measurement remains opt-in and the caller's arguments stay intact.
    model_args = copy.copy(model_args)
    model_args.runtime_receipt = True
    if len(model_args.modes) != 1:
        raise ValueError("Live model recording requires exactly one mode")
    measurement.validate_args(model_args)
    if model_args.output.exists():
        raise ValueError("Live model recording requires a new measurement output")
    if attest is not None and not callable(attest):
        raise TypeError("attest must be callable or None")
    output = Path(catalog_output).resolve()
    if output.is_relative_to(model_args.output.resolve()):
        raise ValueError(
            "Catalog output must remain outside the new measurement directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    # Registration is serialized across explicit producers. The HTTP reader is
    # read-only and observes atomic index replacements without taking this lock.
    with (output / ".record-model.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another model producer is recording this catalog"
            ) from error
        recording = _Recording(model_args, output, clock_cohort)
        try:
            measurement.run(model_args, event_sink=recording.emit)
            recording.finish(attest=attest)
        except BaseException as error:
            try:
                recording.finish(error=error)
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve the original model failure
                note = f"Partial replay finalization failed: {cleanup_error!r}"
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note(note)
                else:  # Python 3.10 retains the diagnostic without replacing the cause.
                    error.__notes__ = [*getattr(error, "__notes__", []), note]
            finally:
                recording.close(error=error)
            raise
        else:
            recording.close()
        return recording.catalog
