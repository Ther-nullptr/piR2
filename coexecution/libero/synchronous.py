"""Blocking Flow4 cycles: fresh perception, complete inference, then clean actions.

The environment never advances during inference. Optional wall pacing applies only
inside each execution phase, including the final command's dwell. No background
model worker or fixed global control deadline exists in this protocol.
"""

import hashlib
import math
from itertools import pairwise

import numpy as np

from coexecution.libero.scheduling import ExecutionCadence
from scripts.libero_protocol_scheduler import CommandTimeline, summarize_values
from scripts.libero_queue_trace import QueueTrace


def validate_synchronous_controls(args):
    coupling = getattr(args, "execution_coupling", "asynchronous")
    if coupling not in ("asynchronous", "synchronous"):
        raise ValueError("Unknown execution coupling")
    chunk = getattr(args, "synchronous_chunk_steps", 8)
    if type(chunk) is not int or chunk < 1:
        raise ValueError("Synchronous chunk steps must be a positive integer")
    if coupling != "synchronous":
        return
    if args.protocol != "deployment" or args.variant != "flow":
        raise ValueError("Synchronous execution requires deployment Flow")
    if (
        args.server_layout != "split"
        or args.feature_transport != "numpy_fp32"
        or args.feature_rpc != "separate"
        or args.action_min_service_ms
        or args.vision_min_service_ms
        or args.vision_max_hz
        or args.deployment_fixed_delay is not None
        or args.artificial_compute_wait
    ):
        raise ValueError(
            "Synchronous Flow requires one ordinary action endpoint without paired/resident "
            "feature exchange, asynchronous scheduling controls or artificial waits"
        )


def validate_synchronous_identity(identity, chunk_steps):
    if (
        identity.get("variant") != "flow"
        or identity.get("role") != "action"
        or identity.get("paired_same_process", False)
        or identity.get("server_layout", "split") != "split"
        or identity.get("feature_transport", "numpy_fp32") != "numpy_fp32"
        or "synchronous_plan" not in identity.get("execution_endpoints", [])
    ):
        raise ValueError(
            "Synchronous Flow requires a single synchronous_plan action endpoint"
        )
    horizon = identity.get("decoded_horizon")
    if type(horizon) is not int or chunk_steps > horizon:
        raise ValueError("Synchronous chunk exceeds the native decoded horizon")


def _validate_plan(result, *, capture_s, source_tick, started, completed, chunk_steps):
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("Synchronous plan must contain finite native (T,7) actions")
    if len(actions) < chunk_steps:
        raise ValueError("Synchronous chunk exceeds the native decoded horizon")
    audit = result["audit"]
    if audit["decoded_horizon"] != len(actions):
        raise ValueError("Decoded horizon evidence disagrees with returned actions")
    if audit["capture_s"] != capture_s or audit["source_tick"] != source_tick:
        raise ValueError("Synchronous plan must consume the current actual observation")
    times = (
        [started]
        + [
            audit[key]
            for key in (
                "vlm_started_s",
                "vlm_completed_s",
                "action_started_s",
                "action_completed_s",
            )
        ]
        + [completed]
    )
    if any(not math.isfinite(value) for value in times) or any(
        a > b for a, b in pairwise(times)
    ):
        raise ValueError("Invalid synchronous stage timestamps")
    if audit["forward_counts"] != {"vlm": 1, "dit": 4}:
        raise ValueError(
            "Each synchronous plan requires one real VLM and four DiT evaluations"
        )
    return actions, audit


def _plan_events(
    events, *, audit, tick, cycle, capture, submitted, completed, accepted
):
    """Reuse source-lineage events; stage boundaries are actual server timestamps."""
    va, vb, aa, ab = (
        audit[key]
        for key in (
            "vlm_started_s",
            "vlm_completed_s",
            "action_started_s",
            "action_completed_s",
        )
    )
    dependency = {
        "feature_sequence": cycle,
        "feature_source_tick": tick,
        "feature_capture_s": capture,
        "state_tick": tick,
        "state_capture_s": capture,
    }
    events.emit(
        "vision_request",
        t_s=vb,
        source_tick=tick,
        capture_s=capture,
        started_s=va,
        completed_s=vb,
        rpc_seconds=vb - va,
        timing_scope="server_vlm_stage",
        graph_execution=audit.get("vlm_graph_execution"),
    )
    events.emit(
        "feature_published",
        t_s=vb,
        source_tick=tick,
        capture_s=capture,
        rpc_completed_s=vb,
        feature_sequence=cycle,
    )
    events.emit(
        "action_submitted",
        t_s=aa,
        request_tick=tick,
        deadline_s=None,
        delay_ticks=0,
        timing_scope="server_action_stage",
        **dependency,
    )
    events.emit("action_started", t_s=aa, request_tick=tick)
    timing = {
        "submitted_s": aa,
        "completed_s": ab,
        "plan_rpc_started_s": submitted,
        "server_plan_started_s": aa,
        "timing_version": 2,
        "cache_age_at_server_plan_ms": (aa - capture) * 1000,
        **dependency,
    }
    events.emit(
        "action_completed",
        t_s=ab,
        request_tick=tick,
        request_elapsed_s=ab - aa,
        valid_start_tick=tick,
        valid_end_tick=tick + accepted,
        **timing,
    )
    events.emit(
        "request_completed",
        t_s=completed,
        audit={**audit, "request_tick": tick},
        client_timing=timing,
    )
    events.emit(
        "action_adopted",
        t_s=completed,
        request_tick=tick,
        installed_slots=accepted,
        expired_slots=0,
        protected_slots=0,
        ready_to_adopt_seconds=completed - ab,
    )


def run_synchronous_loop(
    step_fn,
    observe_fn,
    plan_fn,
    *,
    max_steps,
    chunk_steps=8,
    execution_clock="paced",
    period=0.05,
    clock,
    trace,
    recorder=None,
    record_video=False,
):
    """Run injected real environment/model calls; clocks can be faked in host tests.

    observe_fn must acquire a current physical observation without advancing the
    simulator. plan_fn must return only after its GPU stream and decode complete.
    Exceptions leave an incomplete trace and never become successful episodes.
    """
    if (
        type(max_steps) is not int
        or max_steps < 1
        or type(chunk_steps) is not int
        or chunk_steps < 1
    ):
        raise ValueError("Positive integer step and chunk bounds are required")
    if period != 0.05:
        raise ValueError("LIBERO physics steps must remain 0.05 seconds")
    ExecutionCadence(execution_clock, period, clock.now())
    events = QueueTrace(clock=clock.now, max_records=max_steps * 32 + 4096)
    base = clock.now()
    timeline = CommandTimeline(np.empty((0, 7), dtype=np.float32))
    ticks, frames, audits, durations, phase_durations = [], [], [], [], []
    success, reason, unused, cycle = False, "max_steps", 0, 0
    events.emit(
        "control_start",
        t_s=base,
        period_s=period,
        physics_step_seconds=period,
        execution_clock=execution_clock,
        execution_coupling="synchronous",
        fixed_rate_timing_applicable=False,
        max_steps=max_steps,
        synchronous_chunk_steps=chunk_steps,
        bootstrap_action_slots=0,
    )
    try:
        while len(ticks) < max_steps:
            tick = len(ticks)
            observation = observe_fn()
            capture = clock.now()
            if recorder is not None and cycle == 0:
                recorder.initial(observation, capture_s=capture)
            submitted = clock.now()
            result = plan_fn(observation, capture, tick)
            completed = clock.now()
            actions, audit = _validate_plan(
                result,
                capture_s=capture,
                source_tick=tick,
                started=submitted,
                completed=completed,
                chunk_steps=chunk_steps,
            )
            accepted = min(chunk_steps, max_steps - tick)
            timeline.publish(tick, tick, actions[:accepted], next_tick=tick)
            _plan_events(
                events,
                audit=audit,
                tick=tick,
                cycle=cycle,
                capture=capture,
                submitted=submitted,
                completed=completed,
                accepted=accepted,
            )
            events.emit(
                "cycle_phase",
                t_s=completed,
                cycle=cycle,
                request_tick=tick,
                phase="inference",
                start_s=submitted,
                end_s=completed,
                simulation_paused=True,
                physics_start_tick=tick,
                physics_end_tick=tick,
                source_capture_s=capture,
                frame_hashes=audit["frame_hashes"],
            )
            execute_start = clock.now()
            cadence = ExecutionCadence(execution_clock, period, execute_start)
            current, current_capture = observation, capture
            executed = 0
            for offset in range(accepted):
                target = cadence.wait(clock, offset)
                action, fallback, age = timeline.execute(len(ticks))
                if fallback:
                    raise RuntimeError(
                        "Synchronous execution cannot use fallback actions"
                    )
                applied = clock.now()
                after, success, terminated = step_fn(action)
                computed = clock.now()
                row = {
                    "tick": len(ticks),
                    "cycle": cycle,
                    "scheduled_s": target,
                    "applied_s": applied,
                    "lateness_s": None,
                    "phase_lateness_s": max(0, applied - target)
                    if target is not None
                    else None,
                    "sensor_capture_s": current_capture,
                    "sensor_release_s": applied,
                    "next_observation_computed_s": computed,
                    "execution_clock": execution_clock,
                    "execution_coupling": "synchronous",
                    "physics_step_seconds": period,
                    "simulated_elapsed_seconds": (len(ticks) + 1) * period,
                    "action": action.tolist(),
                    "fallback": False,
                    "command_age_ticks": age,
                    "action_delay_budget_ticks": 0,
                    "command_ready_s": completed,
                    "ready_to_apply_seconds": applied - completed,
                    "first_application_of_request": offset == 0,
                    "environment_step_s": computed - applied,
                    "executed_slot": timeline.last_execution,
                    "action_buffer": timeline.snapshot(),
                    "action_buffer_phase": "after_execute",
                    "feature_sequence": cycle,
                    "feature_source_tick": tick,
                    "feature_capture_s": capture,
                    "state_capture_s": capture,
                    "action_request_pending": False,
                }
                if recorder is not None:
                    row["robot_response"] = recorder.step(current, after, row)
                events.emit("control_tick", t_s=applied, **row)
                ticks.append(row)
                executed += 1
                current, current_capture = after, computed
                if record_video:
                    frames.append(after["video.image"].copy())
                # Every physically applied command owns its complete execution
                # interval, including the final command before success or replanning.
                cadence.wait(clock, offset + 1)
                if success or terminated:
                    reason = "success" if success else "environment_terminated"
                    break
            execute_end = clock.now()
            events.emit(
                "cycle_phase",
                t_s=execute_end,
                cycle=cycle,
                request_tick=tick,
                phase="execution",
                start_s=execute_start,
                end_s=execute_end,
                simulation_paused=False,
                physics_start_tick=tick,
                physics_end_tick=len(ticks),
                execution_clock=execution_clock,
                target_phase_hz=1 / period if cadence.paced else None,
            )
            suffix = len(actions) - executed
            unused += suffix
            events.emit(
                "synchronous_cycle",
                t_s=execute_end,
                cycle=cycle,
                request_tick=tick,
                submitted_s=submitted,
                completed_s=completed,
                audit=audit,
                native_decoded_actions=actions.tolist(),
                executed_steps=executed,
                unused_plan_suffix_slots=suffix,
                discarded_after_cycle=True,
            )
            audits.append(audit)
            durations.append(completed - submitted)
            phase_durations.append(execute_end - execute_start)
            cycle += 1
            if success or terminated:
                break
        end = clock.now()
        events.emit("control_end", t_s=end, control_steps=len(ticks), reason=reason)
    except BaseException as error:
        events.emit(
            "control_error", error_type=type(error).__name__, message=str(error)
        )
        raise
    finally:
        events.flush(trace)
    wall = end - base
    result = {
        "success": bool(success),
        "termination_reason": reason,
        "measurement_mode": "success_terminated_closed_loop",
        "execution_coupling": "synchronous",
        "execution_clock": execution_clock,
        "synchronous_chunk_steps": chunk_steps,
        "control_steps": len(ticks),
        "physics_step_seconds": period,
        "nominal_physics_hz": 1 / period,
        "phase_target_hz": 1 / period if execution_clock == "paced" else None,
        "target_control_hz": None,
        "fixed_rate_timing_applicable": False,
        "control_rate_valid": None,
        "mean_control_rate_valid": None,
        "control_deadline_valid": None,
        "control_period_valid": None,
        "control_deadline_misses_over5ms": None,
        "action_request_deadline_misses": None,
        "actual_control_hz": len(ticks) / wall if wall > 0 else None,
        "controlled_start_s": base,
        "controlled_end_s": end,
        "controlled_wall_seconds": wall,
        "physical_control_seconds": len(ticks) * period,
        "simulated_elapsed_seconds": len(ticks) * period,
        "real_time_factor": len(ticks) * period / wall if wall > 0 else None,
        "first_success_tick": len(ticks) - 1 if success else None,
        "policy_calls": cycle,
        "action_requests": cycle,
        "action_request_hz": cycle / wall if wall > 0 else None,
        "effective_plan_update_hz": cycle / wall if wall > 0 else None,
        "inference_wall_seconds": summarize_values(durations),
        "execution_phase_seconds": summarize_values(phase_durations),
        "inference_wait_fraction": sum(durations) / wall if wall > 0 else None,
        "action_rpc_seconds": None,
        "action_stage_seconds": summarize_values(
            [a["action_completed_s"] - a["action_started_s"] for a in audits]
        ),
        "vlm_stage_seconds": summarize_values(
            [a["vlm_completed_s"] - a["vlm_started_s"] for a in audits]
        ),
        "action_ready_to_first_apply_seconds": summarize_values(
            [
                t["ready_to_apply_seconds"]
                for t in ticks
                if t["first_application_of_request"]
            ]
        ),
        "forward_counts": {
            role: sum(a["forward_counts"][role] for a in audits)
            for role in ("vlm", "dit")
        },
        "scored_cold_capture_requests": sum(
            bool(a.get("graph_execution", {}).get("cold_capture")) for a in audits
        ),
        "unused_plan_suffix_slots": unused,
        "fallback_ticks": 0,
        "action_trace_sha256": hashlib.sha256(
            np.asarray([r["action"] for r in ticks], dtype=np.float32).tobytes()
        ).hexdigest(),
        "final_image_sha256": hashlib.sha256(
            current["video.image"].tobytes()
        ).hexdigest(),
        "replay_trace": events.status(),
    }
    if recorder is not None:
        result["replay_recording"] = recorder.status()
    return result, frames


def synchronous_episode(env, observation, client, args, seed, trace):
    """Existing evaluator adapter; warmup never advances physics or scored RNG."""
    from coexecution.replay.recording import ReplayRecorder
    from scripts.evaluate_libero_protocol import checked_env_step, nested
    from scripts.libero_wallclock import RealClock

    clock = RealClock()
    # Each task can change its text/image graph bucket. Warm it in this exact
    # endpoint, then reset the RNG only once before scored requests.
    warmup = []
    for _ in range(args.slow_warmup_calls):
        result = client.call_endpoint(
            "synchronous_plan",
            {
                "observation": nested(observation),
                "capture_s": clock.now(),
                "source_tick": 0,
            },
        )
        warmup.append(result["audit"])
    client.call_endpoint("reset", {"options": {"seed": seed}})
    recorder = (
        ReplayRecorder(max_frames=args.replay_max_frames)
        if args.record_replay
        else None
    )
    args.replay_recorder = recorder

    def observe():
        return env._process_observation(
            env._env.env._get_observations(force_update=True)
        )

    def plan(obs, capture_s, source_tick):
        return client.call_endpoint(
            "synchronous_plan",
            {
                "observation": nested(obs),
                "capture_s": capture_s,
                "source_tick": source_tick,
            },
        )

    def step(action):
        obs, _, done, truncated, info = checked_env_step(env, action)
        return obs, bool(info["success"]), bool(done or truncated)

    result, frames = run_synchronous_loop(
        step,
        observe,
        plan,
        max_steps=args.max_steps,
        chunk_steps=args.synchronous_chunk_steps,
        execution_clock=args.execution_clock,
        clock=clock,
        trace=trace,
        recorder=recorder,
        record_video=args.record_video,
    )
    result["warmup"] = {
        "scope": "before scored control and episode RNG reset; no physics steps",
        "calls": warmup,
    }
    return result, frames
