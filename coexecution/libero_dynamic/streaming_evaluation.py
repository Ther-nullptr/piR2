"""Existing πR² policies in the same pinned moving-object physical scenes.

The physical loop owns rendering, task success and one paid startup hold. The
core owns two logical service lanes. Native timing uses each full adapter call's
measured duration; fixed timing exposes explicit counterfactual controls.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .evaluation import (
    _effective_items,
    _jsonable,
    _load_runtime,
    _source_arguments,
    _speed,
    _write_json,
    add_model_arguments,
    add_scene_arguments,
    observation_fingerprint,
    prepare_conditions,
    scene_drive_metadata,
    seed_metadata,
    speed_metadata,
)
from .physical import make_physical_episode_type
from .streaming import (
    STARTUP_MEASUREMENTS,
    StageLatencies,
    StepResult,
    StreamingController,
    startup_configuration,
    timing_metadata,
)
from .streaming_rpc import Pir2RPC


def add_streaming_arguments(parser):
    add_scene_arguments(parser)
    add_model_arguments(parser)
    parser.add_argument("--action-port", type=int, required=True)
    parser.add_argument("--vision-port", type=int, required=True)
    parser.add_argument(
        "--timing-mode",
        choices=("native", "fixed"),
        default="native",
        help="Native uses each measured full adapter call; fixed requires all four stage durations",
    )
    for stage in ("bootstrap", "s1", "s2", "recovery"):
        parser.add_argument(
            f"--{stage}-latency-seconds",
            type=float,
            help=f"Fixed counterfactual {stage} service duration in seconds",
        )
    parser.add_argument("--fixed-delay", type=int, choices=range(1, 6))
    parser.add_argument(
        "--startup-release-seconds",
        type=float,
        help="Common physical startup release; extra waiting is recorded separately from native model service",
    )


def stage_latencies(options):
    # Old programmatic configurations retain fixed semantics. CLI defaults native.
    mode = getattr(options, "timing_mode", "fixed")
    values = {
        stage + "_s": getattr(options, stage + "_latency_seconds", None)
        for stage in ("bootstrap", "s1", "s2", "recovery")
    }
    if mode == "native":
        if any(value is not None for value in values.values()):
            raise ValueError("Native timing rejects fixed stage latency arguments")
        return None
    if mode != "fixed":
        raise ValueError("Timing mode must be native or fixed")
    if any(value is None for value in values.values()):
        raise ValueError("Fixed timing requires all four stage latency arguments")
    return StageLatencies(**values)


def validate_options(options):
    stage_latencies(options)
    startup_configuration(getattr(options, "startup_release_seconds", None))
    _speed(options.speed_mps, custom_speed=getattr(options, "custom_speed", False))
    for name in ("max_steps", "camera_size"):
        if type(getattr(options, name)) is not int or getattr(options, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(options.warmup_requests) is not int or options.warmup_requests < 0:
        raise ValueError("warmup_requests must be nonnegative")
    seed_metadata(options)
    scene_drive_metadata(options)
    if options.fixed_delay is not None and options.fixed_delay not in range(1, 6):
        raise ValueError("Fixed delay must be in 1..5")
    if (
        any(
            type(port) is not int or not 1 <= port <= 65535
            for port in (options.action_port, options.vision_port)
        )
        or options.action_port == options.vision_port
    ):
        raise ValueError("Independent valid S1/S2 ports are required")
    if not 1 <= options.rpc_timeout_seconds <= (2**31 - 1) // 1000:
        raise ValueError("RPC timeout must fit positive int32 milliseconds")
    if not math.isfinite(options.video_fps) or not 0 < options.video_fps <= 500:
        raise ValueError("Video FPS must be finite and in (0,500]")
    if not options.expected_checkpoint.expanduser().is_dir():
        raise ValueError("--expected-checkpoint must already exist locally")


@contextmanager
def _rpc_phase(rpc, phase):
    first = len(rpc.records)
    try:
        yield
    finally:
        for record in rpc.records[first:]:
            record["phase"] = phase


def warmup_streaming(
    rpc, observation, language, *, rounds, delays, seed, capture_s, records=None
):
    """Warm every eligible d on the actual initial input without any world access.

    Each pair starts from a fresh true VLM/Flow4 bootstrap, then a true S2 call
    and a true cached one-NFE S1 request. Caller must formally bootstrap again.
    The caller-owned records list preserves completed warmups on a later error.
    """
    records = [] if records is None else records
    with _rpc_phase(rpc, "warmup"):
        for round_index in range(rounds):
            for delay in delays:
                record = {
                    "phase": "warmup",
                    "round": round_index,
                    "delay_ticks": delay,
                    "status": "started",
                }
                records.append(record)
                bootstrap = rpc.bootstrap(
                    observation,
                    language,
                    capture_s=capture_s,
                    delay_ticks=delay,
                    seed=seed,
                )
                record["bootstrap"] = bootstrap
                visual = rpc.vision(
                    observation, language, capture_s=capture_s, source_tick=0
                )
                record["vision"] = {"meta": visual["meta"], "wall": visual["wall"]}
                plan = rpc.plan(
                    observation,
                    language,
                    request_tick=0,
                    delay_ticks=delay,
                    committed_actions=bootstrap["actions"][:delay],
                    state_capture_s=capture_s,
                    feature={**visual, "sequence": 1},
                )
                record.update(plan=plan, status="complete")
    return records


class PhysicsVideoWriter:
    """Two observer-camera streams sampled by the common physical substep hook."""

    def __init__(self, source, options, base, item_id):
        self.source, self.options = source, options
        self.paths = [
            base / "model_input_videos" / f"{item_id}--pir2-dual.mp4",
            base / "wrist_input_videos" / f"{item_id}--pir2-wrist.mp4",
        ]
        self.writers = []
        self.frame_counts = [0, 0]

    def open(self):
        if not self.options.save_videos:
            return
        if self.source.runtime.imageio is None:
            raise RuntimeError("imageio is required to record the physical world")
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.writers.append(
                self.source.runtime.imageio.get_writer(
                    path,
                    fps=self.options.video_fps,
                    macro_block_size=None,
                    codec="libx264",
                    pixelformat="yuv420p",
                )
            )

    def append(self, env, sample):
        names = ("dual", "wrist")
        sample["streams"] = {
            name: {
                "status": "not_attempted",
                "frame_index": self.frame_counts[index],
                "media_time_s": self.frame_counts[index] / self.options.video_fps,
            }
            for index, name in enumerate(names)
        }
        size = self.options.camera_size
        agent, wrist = (
            np.ascontiguousarray(
                np.asarray(env.sim.render(size, size, camera_name=name))[::-1]
            )
            for name in ("agentview", "robot0_eye_in_hand")
        )
        for index, frame in enumerate((np.concatenate([agent, wrist], axis=1), wrist)):
            stream = sample["streams"][names[index]]
            stream["status"] = "writing"
            try:
                self.writers[index].append_data(np.ascontiguousarray(frame))
            except BaseException as error:
                stream.update(status="error", error=f"{type(error).__name__}: {error}")
                raise
            stream["status"] = "written"
            self.frame_counts[index] += 1

    def close(self):
        errors = []
        for writer in self.writers:
            try:
                writer.close()
            except Exception as error:  # noqa: BLE001 -- retain every resource cleanup error
                errors.append(f"video close: {type(error).__name__}: {error}")
        return errors


def _partial_streaming(controller):
    """Copy only actual evidence after failure; never synthesize completed work."""
    if controller is None:
        return None
    pending = []
    for lane in ("s1", "s2"):
        future = getattr(controller, lane, None)
        if future is not None:
            record = copy.deepcopy(future[0])
            record["status"] = "unapplied_at_control_error"
            pending.append(record)
    trace = getattr(controller, "trace", None)
    timeline = getattr(controller, "timeline", None)
    return {
        "status": "incomplete",
        **timing_metadata(controller.timing_mode, controller.latencies),
        **controller.startup_metadata(),
        "bootstrap": controller.bootstrap_evidence,
        "logical_epoch_s": controller.epoch_s,
        "config": {
            **timing_metadata(controller.timing_mode, controller.latencies),
            **controller.startup_metadata(),
        },
        "logical_time_s": controller.logical_time_s,
        "requests": copy.deepcopy(getattr(controller, "requests", [])),
        "actions": copy.deepcopy(getattr(controller, "actions", [])),
        "events": [] if trace is None else trace.records(),
        "inflight_at_end": pending,
        "final_buffer": None if timeline is None else timeline.snapshot(),
    }


def make_streaming_episode_type(source, options):
    physical_type = make_physical_episode_type(
        source, options, callback_convention="physical_origin"
    )
    latencies = stage_latencies(options)
    mode = getattr(options, "timing_mode", "fixed")
    timing = timing_metadata(mode, latencies)

    class StreamingEpisode(physical_type):
        def __init__(self, item, args, *, rpc):
            super().__init__(item, args)
            self.rpc = rpc

        def run(self):
            base = options.output / self.item["level"] / self.item["suite"]
            audit_path = base / "policy_audits" / f"{self.item['item_id']}.json"
            manifest = base / "manifests" / f"{self.item['item_id']}.json"
            trace_path = base / "traces" / f"{self.item['item_id']}--pir2.jsonl"
            videos = PhysicsVideoWriter(source, options, base, self.item["item_id"])
            warmup, attempts, cleanup_errors = [], [], []
            controller = result = bootstrap = diagnostics = identities = initial = None
            language, status, error_text, error_traceback = "", "ok", None, None
            fatal = None
            setup_counts = {}
            diagnostic_summary = None
            scored_record_start = None
            start_physics = end_physics = None
            rollout = 0.0
            startup_seconds = None
            started = time.monotonic()
            try:
                self._configure_env_selection()
                suite, task, bddl = source.legacy.task_bddl(
                    self.item["suite"], self.item["task_id"]
                )
                language = task.language
                states = suite.get_task_init_states(self.item["task_id"])
                self.env = source.runtime.OffScreenRenderEnv(
                    bddl_file_name=str(bddl),
                    camera_heights=options.camera_size,
                    camera_widths=options.camera_size,
                    control_freq=20,
                )
                self.env.seed(options.seed)
                dt = float(self.env.sim.model.opt.timestep)
                if options.save_videos and options.video_fps > 1 / dt + 1e-9:
                    raise ValueError(
                        "Video FPS exceeds actual physics substep frequency"
                    )
                observation, settled, warmed, presteps, passengers = self._setup_world(
                    self.env, states[options.init_state_id]
                )
                setup_counts = {
                    "preview_compatible_settle_steps_done": settled,
                    "warmup_steps_done": warmed,
                    "pre_policy_dynamic_steps_done": presteps,
                    "fixture_passengers_added_count": len(passengers),
                }
                diagnostics = source.legacy.init_diag_state(self.env, self.args)
                initial = {
                    "observation_sha256": observation_fingerprint(
                        observation, language
                    ),
                    "physics_sim_time_s": float(self.env.sim.data.time),
                    "motion_phase_s": self._motion_phase_s(),
                    "pre_policy_dynamic_steps": 40,
                }
                identities = self.rpc.prepare()
                controller = StreamingController(
                    self.rpc,
                    latencies=latencies,
                    fixed_delay=options.fixed_delay,
                    timing_mode=mode,
                    startup_release_seconds=getattr(
                        options, "startup_release_seconds", None
                    ),
                )
                warmup_streaming(
                    self.rpc,
                    observation,
                    language,
                    rounds=options.warmup_requests,
                    delays=[options.fixed_delay]
                    if options.fixed_delay is not None
                    else list(range(1, 6)),
                    seed=options.seed,
                    capture_s=controller.epoch_s,
                    records=warmup,
                )
                if self.env.sim.data.time != initial["physics_sim_time_s"]:
                    raise RuntimeError("Model warmup advanced the physical world")
                videos.open()
                if options.save_videos:
                    self.frame_sink = videos.append
                # Timed model computation itself does not advance the simulator.
                # Exactly one explicit startup hold pays for the initial plan.
                start_physics = float(self.env.sim.data.time)
                self.control_started = True
                scored_record_start = len(self.rpc.records)
                controller.bootstrap(observation, language, seed=options.seed)
                # The controller's private successful receipt owns startup
                # timing; an external caller can only mutate detached copies.
                bootstrap = controller.bootstrap_evidence
                startup_seconds = bootstrap["startup_seconds"]
                self.hold_phase = "bootstrap_hold"
                self._apply_sync_latency(self.env, startup_seconds)
                if not math.isclose(
                    float(self.env.sim.data.time) - start_physics,
                    startup_seconds,
                    rel_tol=0,
                    abs_tol=1e-8,
                ):
                    raise RuntimeError(
                        "Actual startup physics differs from the declared paid hold"
                    )
                fresh = self.env.env._get_observations(force_update=True)
                self._motion_event(
                    "post_bootstrap_observation",
                    logical_capture_s=bootstrap["control_origin_s"],
                    observation_sha256=observation_fingerprint(fresh, language),
                )

                def step(action, tick):
                    attempt = {
                        "tick": tick,
                        "raw_action": action.copy(),
                        "status": "started",
                        "physics_started_s": float(self.env.sim.data.time),
                    }
                    attempts.append(attempt)
                    self.active_chunk_id = tick
                    try:
                        # Core already projected 0/1 gripper to native +1/-1.
                        obs, _, done, _ = self._step_env(self.env, action)
                        success = bool(done) or bool(self.env.check_success())
                        if self.use_preview:
                            source.legacy.update_preview_compatible_diagnostics(
                                self.env,
                                diagnostics,
                                self.preview_states,
                                self.preview_elapsed,
                            )
                        else:
                            source.legacy.update_diagnostics(self.env, diagnostics)
                        attempt.update(status="complete", success=success)
                        return StepResult(obs, success=success)
                    except BaseException:
                        attempt["status"] = "error"
                        raise
                    finally:
                        attempt["physics_completed_s"] = float(self.env.sim.data.time)
                        attempt["physics_elapsed_s"] = (
                            attempt["physics_completed_s"]
                            - attempt["physics_started_s"]
                        )
                        attempt["physics_complete"] = math.isclose(
                            attempt["physics_elapsed_s"],
                            self.preview_dt,
                            rel_tol=0,
                            abs_tol=1e-8,
                        )

                result = controller.run(
                    bootstrap,
                    initial_observation=fresh,
                    initial_capture_s=bootstrap["control_origin_s"],
                    language=language,
                    step_fn=step,
                    max_steps=options.max_steps,
                )
                measured = float(self.env.sim.data.time) - start_physics
                if not math.isclose(
                    measured, result["physical_seconds"], rel_tol=0, abs_tol=1e-7
                ):
                    raise RuntimeError(
                        "Logical physical_seconds differs from actual simulator advancement"
                    )
                if not math.isclose(
                    self.rollout_sim_time_s, measured, rel_tol=0, abs_tol=1e-7
                ):
                    raise RuntimeError("Physical hooks lost or duplicated world time")
                if (
                    options.save_videos
                    and videos.frame_counts != [len(self.video_samples)] * 2
                ):
                    raise RuntimeError(
                        "Camera frames differ from physical sample mapping"
                    )
            except BaseException as error:  # noqa: BLE001 -- persist partial artifacts before re-raising interrupts
                status = "error"
                error_text = f"{type(error).__name__}: {error}"
                error_traceback = traceback.format_exc()
                if not isinstance(error, Exception):
                    fatal = error
            finally:
                if scored_record_start is not None:
                    for record in self.rpc.records[scored_record_start:]:
                        record["phase"] = "scored"
                if result is None:
                    result = _partial_streaming(controller)
                if self.env is not None:
                    end_physics = float(self.env.sim.data.time)
                    if start_physics is not None:
                        rollout = end_physics - start_physics
                if diagnostics is not None:
                    try:
                        diagnostic_summary = source.legacy.summarize_diagnostics(
                            diagnostics, self.args
                        )
                    except Exception as error:  # noqa: BLE001 -- diagnostic failures must not prevent resource cleanup
                        cleanup_errors.append(
                            f"diagnostics: {type(error).__name__}: {error}"
                        )
                cleanup_errors.extend(videos.close())
                for name, resource in (("rpc", self.rpc), ("env", self.env)):
                    if resource is not None:
                        try:
                            resource.close()
                        except Exception as error:  # noqa: BLE001 -- save cleanup errors and close remaining resources
                            cleanup_errors.append(
                                f"{name} close: {type(error).__name__}: {error}"
                            )
            if cleanup_errors:
                status = "error"
                error_text = "; ".join([error_text or "", *cleanup_errors]).lstrip("; ")
            startup = (
                controller.startup_metadata()
                if controller is not None
                else {
                    **startup_configuration(
                        getattr(options, "startup_release_seconds", None)
                    ),
                    **{key: None for key in STARTUP_MEASUREMENTS},
                }
            )
            evidence = {
                "schema": "pir2.libero_dynamic.streaming_audit.v1",
                **timing,
                "item_id": self.item["item_id"],
                "identities": identities,
                "diagnostics": diagnostic_summary,
                "initial_observation": initial,
                "warmup": warmup,
                "streaming": result,
                "bootstrap": bootstrap,
                "rpc_records": self.rpc.records,
                "action_attempts": attempts,
                "motion_events": self.motion_events,
                **scene_drive_metadata(options),
                "drive_freeze": self.drive_freeze,
                "observation_registry": self.observed_objects,
                "latency_backend": self.latency_backend,
                "fixture_timing": None
                if self.fixture_timing is None
                else self.fixture_timing.records,
                "fixture_driver": None
                if self.fixture_timing is None
                else self.fixture_timing.driver_evidence,
                "video_samples": self.video_samples,
                "video_frame_counts": videos.frame_counts,
                "cleanup_errors": cleanup_errors,
                "clocks": {
                    "logical": "fixed positive epoch plus physical time since scored bootstrap capture",
                    "physical": "actual MuJoCo time since scored bootstrap capture; includes one startup hold",
                    "wall": "unmodified host/server monotonic timestamps; never subtracted from logical captures",
                },
            }
            _write_json(audit_path, evidence)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("w", encoding="utf-8") as stream:
                for event in (result or {}).get("events", []):
                    stream.write(
                        json.dumps(
                            _jsonable(event), ensure_ascii=False, allow_nan=False
                        )
                        + "\n"
                    )
            summary = {
                **{
                    key: self.item.get(key)
                    for key in (
                        "item_id",
                        "suite",
                        "task_id",
                        "level",
                        "traj_index",
                        "variant_id",
                    )
                },
                "evaluation_schema": "pir2.libero_dynamic.streaming.v1",
                "protocol": "pir2_dynamic",
                "status": status,
                "error": error_text,
                "error_traceback": error_traceback,
                "success": status == "ok"
                and bool((result or {}).get("success", False)),
                "steps": sum(attempt["physics_complete"] for attempt in attempts),
                "controller_recorded_steps": len((result or {}).get("actions", [])),
                "step_count_contract": "steps counts actual complete 50ms physical actions; controller_recorded_steps counts returned core records",
                "success_predicate": "source done or env.check_success",
                "diagnostics": diagnostic_summary,
                "language": language,
                "precision": options.precision,
                **speed_metadata(options),
                **seed_metadata(options),
                **scene_drive_metadata(options),
                **timing,
                "resource_model": "independent_stage_service",
                "compute_coupling": "serialized_gpu_simulation",
                "fixed_delay": options.fixed_delay,
                **startup,
                "startup_seconds_applied": 0.0
                if startup_seconds is None
                else min(rollout, startup_seconds),
                "rollout_sim_time_s": rollout,
                "physical_started_s": start_physics,
                "physical_completed_s": end_physics,
                "setup_counts": setup_counts,
                "drive_freeze": self.drive_freeze,
                "warmup_rounds": options.warmup_requests,
                "logical_epoch_s": None if controller is None else controller.epoch_s,
                "video_frame_count": min(videos.frame_counts),
                "video_sample_attempts": len(self.video_samples),
                "video_frame_counts": dict(
                    zip(("dual", "wrist"), videos.frame_counts, strict=True)
                ),
                "video_fps": options.video_fps,
                "video_duration_s": min(videos.frame_counts) / options.video_fps,
                "model_input_video": str(videos.paths[0])
                if options.save_videos and status == "ok"
                else None,
                "wrist_input_video": str(videos.paths[1])
                if options.save_videos and status == "ok"
                else None,
                "recorded_media": [
                    str(path) for path in videos.paths if path.is_file()
                ],
                "video_view_contract": "Physical observer camera renders; separate from officially rotated model inputs",
                "video_time_mapping": str(audit_path) + "#video_samples",
                "policy_audit": str(audit_path),
                "manifest": str(manifest),
                "step_trace": str(trace_path),
                "wall_duration_s": time.monotonic() - started,
                "cleanup_errors": cleanup_errors,
                "latency_backend": self.latency_backend,
            }
            _write_json(manifest, summary)
            if fatal is not None:
                raise fatal
            return summary

    return StreamingEpisode


def evaluate_streaming(options):
    validate_options(options)
    timing = timing_metadata(
        getattr(options, "timing_mode", "fixed"), stage_latencies(options)
    )
    inventory, conditions = prepare_conditions(options)
    options.output.mkdir(parents=True, exist_ok=False)
    evaluation = {
        "schema": "pir2.libero_dynamic.streaming_evaluation.v1",
        "status": "starting",
        "configuration": {
            **vars(options),
            **seed_metadata(options),
            **scene_drive_metadata(options),
        },
        "upstream": inventory["upstream"],
        "conditions": conditions,
        "protocol": "pir2_dynamic",
        **speed_metadata(options),
        **seed_metadata(options),
        **scene_drive_metadata(options),
        "phase_contract": inventory["phase_contract"],
        **timing,
        "resource_model": "independent_stage_service",
        **startup_configuration(getattr(options, "startup_release_seconds", None)),
        "action_budget_contract": "max_steps counts actual 50ms actions; startup hold adds physics once",
    }
    _write_json(options.output / "evaluation.json", evaluation)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    results = []
    try:
        source = _load_runtime(options.upstream)
        source_options = copy.copy(options)
        source_options.chunk_steps = 1
        args = _source_arguments(source, source_options)
        items = _effective_items(source, args, options, conditions)
        _write_json(options.output / "items.json", items)
        _write_json(options.output / "runtime-config.json", vars(args))
        episode_type = make_streaming_episode_type(source, options)
        with (options.output / "manifest.jsonl").open(
            "w", encoding="utf-8"
        ) as manifest:
            for index, item in enumerate(items):
                print(
                    f"[{index + 1}/{len(items)}] {item['item_id']} {options.precision} pir2_dynamic",
                    flush=True,
                )
                rpc = Pir2RPC(
                    action_port=options.action_port,
                    vision_port=options.vision_port,
                    expected_checkpoint=options.expected_checkpoint,
                    precision=options.precision,
                    host=options.model_host,
                    timeout_ms=options.rpc_timeout_seconds * 1000,
                )
                interrupted = None
                try:
                    result = episode_type(item, args, rpc=rpc).run()
                except BaseException as error:
                    # run() persists its interrupted episode before propagating.
                    # Read only this known item's manifest, register its identity
                    # in the batch, then propagate the original interruption.
                    interrupted_path = (
                        options.output
                        / item["level"]
                        / item["suite"]
                        / "manifests"
                        / f"{item['item_id']}.json"
                    )
                    if not interrupted_path.is_file():
                        raise
                    result = json.loads(interrupted_path.read_text())
                    if (
                        result.get("item_id") != item["item_id"]
                        or result.get("status") != "error"
                    ):
                        raise
                    interrupted = error
                result.update(
                    condition_id=conditions[index]["condition_id"],
                    upstream=inventory["upstream"],
                    **seed_metadata(options),
                    **scene_drive_metadata(options),
                )
                if "parent_dynamic_condition_id" in conditions[index]:
                    result["parent_dynamic_condition_id"] = conditions[index][
                        "parent_dynamic_condition_id"
                    ]
                if "template_condition_id" in conditions[index]:
                    result["template_condition_id"] = conditions[index][
                        "template_condition_id"
                    ]
                _write_json(Path(result["manifest"]), result)
                results.append(result)
                manifest.write(
                    json.dumps(_jsonable(result), ensure_ascii=False, allow_nan=False)
                    + "\n"
                )
                manifest.flush()
                print(
                    f"status={result['status']} success={result['success']} steps={result['steps']}",
                    flush=True,
                )
                _write_json(
                    options.output / "evaluation-summary.json",
                    {
                        "schema": "pir2.libero_dynamic.streaming_summary.v1",
                        "status": "running",
                        "protocol": "pir2_dynamic",
                        "precision": options.precision,
                        "suite": options.suite,
                        "conditions_planned": len(items),
                        "conditions_completed": len(results),
                        "successes": sum(row["success"] for row in results),
                        **timing,
                        **seed_metadata(options),
                        **scene_drive_metadata(options),
                    },
                )
                if interrupted is not None:
                    raise interrupted
                if result["status"] != "ok":
                    raise RuntimeError(
                        f"πR² episode failed: {result['error']}; retained {result['manifest']}"
                    )
        summary = json.loads((options.output / "evaluation-summary.json").read_text())
        summary.update(
            status="complete", success_rate=summary["successes"] / len(results)
        )
        _write_json(options.output / "evaluation-summary.json", summary)
        evaluation["status"] = "complete"
        _write_json(options.output / "evaluation.json", evaluation)
        return summary
    except BaseException as error:
        evaluation.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            completed_episodes=len(results),
        )
        _write_json(options.output / "evaluation.json", evaluation)
        _write_json(options.output / "failure.json", evaluation)
        _write_json(
            options.output / "evaluation-summary.json",
            {
                "schema": "pir2.libero_dynamic.streaming_summary.v1",
                "status": "failed",
                **timing,
                **seed_metadata(options),
                **scene_drive_metadata(options),
                "protocol": "pir2_dynamic",
                "conditions_completed": len(results),
                "successes": sum(row["success"] for row in results),
                "error": evaluation["error"],
            },
        )
        raise
