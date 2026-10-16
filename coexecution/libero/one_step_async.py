"""One-step lookahead with real policy/physics overlap and explicit waiting.

One owned worker computes a complete VLM + Flow4 request from the observation
before the current command. The simulator owner executes exactly one command,
then waits for the next result. There is no fallback or extra prefix consumption.
"""

import copy
import hashlib
import math
import queue
import threading
from concurrent.futures import Future

import numpy as np

from coexecution.libero.synchronous import _validate_plan, warmup_rpc_timeout
from scripts.libero_protocol_scheduler import CommandTimeline, summarize_values
from scripts.libero_queue_trace import QueueTrace


def snapshot_observation(observation):
    """Detach simulator-owned buffers before another thread can read them."""
    owned = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            value = np.array(value, copy=True, order="C")
            value.flags.writeable = False
        else:
            value = copy.deepcopy(value)
        owned[key] = value
    return owned


class OwnedPlanWorker:
    """Create, use and close the resource on one dedicated daemon thread.

    RPCs must themselves be bounded. Joining is bounded as a final failure guard;
    Python cannot forcibly interrupt arbitrary callbacks. A join timeout fails
    the experiment, never returns successful control or steals another socket.
    """

    def __init__(self, factory, *, startup_timeout=10, shutdown_timeout=270):
        self.jobs = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.ready, self.closed = Future(), Future()
        self.shutdown_timeout = shutdown_timeout
        self.thread = threading.Thread(
            target=self._run, args=(factory,), name="one-step-policy", daemon=True
        )
        self.thread.start()
        try:
            self.ready.result(timeout=startup_timeout)
        except BaseException:
            self.stop.set()
            self.thread.join(timeout=shutdown_timeout)
            raise

    def _run(self, factory):
        resource = None
        try:
            resource = factory()
            self.ready.set_result(None)
            while not self.stop.is_set() or not self.jobs.empty():
                try:
                    future, operation = self.jobs.get(timeout=0.05)
                except queue.Empty:
                    continue
                if future.set_running_or_notify_cancel():
                    try:
                        future.set_result(operation(resource))
                    except BaseException as error:  # noqa: BLE001 - propagate through Future
                        future.set_exception(error)
        except BaseException as error:  # noqa: BLE001 - propagate startup failure
            if not self.ready.done():
                self.ready.set_exception(error)
            self.closed.set_exception(error)
        finally:
            try:
                if resource is not None:
                    resource.close()
            except BaseException as error:  # noqa: BLE001 - propagate teardown failure
                if not self.closed.done():
                    self.closed.set_exception(error)
            if not self.closed.done():
                self.closed.set_result(None)

    def submit(self, operation):
        if self.stop.is_set() or not self.thread.is_alive():
            raise RuntimeError("Policy worker is closed")
        future = Future()
        self.jobs.put_nowait((future, operation))
        return future

    def call(self, operation, *, timeout):
        return self.submit(operation).result(timeout=timeout)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=self.shutdown_timeout)
        if self.thread.is_alive():
            raise TimeoutError("Owned policy worker did not stop within its bound")
        self.closed.result()


def _stage_events(events, request):
    """Server stages use unique request IDs, independent of observation ticks."""
    audit = request["audit"]
    identity = {
        key: request[key]
        for key in ("request_id", "request_tick", "target_tick", "source_tick")
    }
    dependency = {
        "feature_sequence": request["request_id"],
        "feature_source_tick": request["source_tick"],
        "feature_capture_s": request["capture_s"],
        "state_tick": request["source_tick"],
        "state_capture_s": request["capture_s"],
    }
    va, vb, aa, ab = (
        audit[key]
        for key in (
            "vlm_started_s",
            "vlm_completed_s",
            "action_started_s",
            "action_completed_s",
        )
    )
    events.emit(
        "vision_request",
        t_s=vb,
        **identity,
        capture_s=request["capture_s"],
        started_s=va,
        completed_s=vb,
        feature_sequence=request["request_id"],
        rpc_seconds=vb - va,
        timing_scope="server_vlm_stage",
        graph_execution=audit.get("vlm_graph_execution"),
    )
    events.emit(
        "feature_published",
        t_s=vb,
        **identity,
        capture_s=request["capture_s"],
        rpc_completed_s=vb,
        feature_sequence=request["request_id"],
    )
    events.emit(
        "action_submitted",
        t_s=aa,
        **identity,
        **dependency,
        deadline_s=None,
        delay_ticks=0,
        timing_scope="server_action_stage",
    )
    events.emit("action_started", t_s=aa, **identity)
    timing = {
        "submitted_s": aa,
        "completed_s": ab,
        "policy_submitted_s": request["submitted_s"],
        "policy_rpc_started_s": request["rpc_started_s"],
        "policy_rpc_completed_s": request["rpc_completed_s"],
        "timing_scope": "server_action_stage",
        "server_plan_started_s": aa,
        "timing_version": 2,
        "cache_age_at_server_plan_ms": (aa - request["capture_s"]) * 1000,
        **dependency,
    }
    events.emit(
        "action_completed",
        t_s=ab,
        **identity,
        **timing,
        request_elapsed_s=ab - aa,
        valid_start_tick=request["target_tick"],
        valid_end_tick=request["target_tick"] + 1,
    )
    return timing


def _intersection_seconds(inference, execution):
    return sum(
        max(0.0, min(b, d) - max(a, c)) for a, b in inference for c, d in execution
    )


def run_one_step_async_loop(
    step_fn,
    observe_fn,
    planner_factory,
    *,
    max_steps,
    clock,
    trace,
    recorder=None,
    record_video=False,
    initialize=None,
    request_timeout_seconds=270,
    shutdown_timeout_seconds=270,
):
    """Execute exactly one command per result, with actual concurrent inference.

    The factory, optional prepare(), plan() and close() run only on the owned
    worker. initialize(worker) may perform bounded warmup/reset calls before the
    scored window. The main thread alone calls observe_fn and step_fn.
    """
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if any(
        not math.isfinite(t) or t <= 0
        for t in (request_timeout_seconds, shutdown_timeout_seconds)
    ):
        raise ValueError("Worker timeouts must be positive and finite")
    events = QueueTrace(clock=clock.now, max_records=max_steps * 40 + 4096)
    worker = None
    pending = None
    ticks, frames, requests, execution, waits = [], [], [], [], []
    timeline = CommandTimeline(np.empty((0, 7), dtype=np.float32))
    base = controlled_end = None
    success, reason, warmup = False, "max_steps", None
    failed = True

    def submit(observation, capture_s, source_tick, target_tick):
        owned = observation
        request = {
            "request_id": target_tick,
            "request_tick": target_tick,
            "target_tick": target_tick,
            "source_tick": source_tick,
            "capture_s": capture_s,
            "frame_hashes": {
                key[6:]: hashlib.sha256(value.tobytes()).hexdigest()
                for key, value in owned.items()
                if key.startswith("video.")
            },
            "submitted_s": clock.now(),
        }
        events.emit("policy_request_submitted", t_s=request["submitted_s"], **request)

        def infer(planner):
            try:
                prepared = (
                    planner.prepare(owned) if hasattr(planner, "prepare") else owned
                )
                request["rpc_started_s"] = clock.now()
                events.emit(
                    "policy_rpc_started", t_s=request["rpc_started_s"], **request
                )
                result = planner.plan(prepared, capture_s, source_tick)
                request["rpc_completed_s"] = clock.now()
                events.emit(
                    "policy_rpc_completed", t_s=request["rpc_completed_s"], **request
                )
                actions, audit = _validate_plan(
                    result,
                    capture_s=capture_s,
                    source_tick=source_tick,
                    started=request["rpc_started_s"],
                    completed=request["rpc_completed_s"],
                    chunk_steps=1,
                )
                if audit["frame_hashes"] != request["frame_hashes"]:
                    raise ValueError(
                        "Policy frame hashes differ from the owned observation"
                    )
                request.update(
                    actions=actions.copy(), audit={**audit, "request_tick": target_tick}
                )
                request["client_timing"] = _stage_events(events, request)
                events.emit(
                    "cycle_phase",
                    t_s=request["rpc_completed_s"],
                    cycle=target_tick,
                    request_id=target_tick,
                    request_tick=target_tick,
                    source_tick=source_tick,
                    target_tick=target_tick,
                    phase="inference",
                    start_s=request["rpc_started_s"],
                    end_s=request["rpc_completed_s"],
                    submitted_s=request["submitted_s"],
                    simulation_paused=target_tick == 0,
                    source_capture_s=capture_s,
                    frame_hashes=audit["frame_hashes"],
                    execution_coupling="one_step_async",
                    execution_clock="ready",
                )
                return request
            except BaseException as error:
                events.emit(
                    "policy_request_failed",
                    request_id=target_tick,
                    request_tick=target_tick,
                    source_tick=source_tick,
                    target_tick=target_tick,
                    error_type=type(error).__name__,
                    message=str(error),
                )
                raise

        future = worker.submit(infer)
        return future

    def receive(future, *, adopted, unused_reason=None):
        request = future.result(timeout=request_timeout_seconds)
        accepted = clock.now()
        requests.append(request)
        events.emit(
            "request_completed" if adopted else "unused_result_at_episode_end",
            t_s=accepted,
            audit=request["audit"],
            client_timing=request["client_timing"],
            request_id=request["request_id"],
            target_tick=request["target_tick"],
            unused_reason=unused_reason,
        )
        if adopted:
            timeline.publish(
                request["request_id"],
                request["target_tick"],
                request["actions"][:1],
                next_tick=len(ticks),
            )
            events.emit(
                "action_adopted",
                t_s=accepted,
                request_tick=request["request_tick"],
                request_id=request["request_id"],
                target_tick=request["target_tick"],
                installed_slots=1,
                expired_slots=0,
                protected_slots=0,
                ready_to_adopt_seconds=accepted - request["rpc_completed_s"],
            )
        request["adopted_s"] = accepted if adopted else None
        events.emit(
            "one_step_async_request",
            t_s=accepted,
            **{key: value for key, value in request.items() if key != "actions"},
            native_decoded_actions=request["actions"].tolist(),
            accepted_prefix_steps=1 if adopted else 0,
            unused_plan_suffix_slots=len(request["actions"]) - int(adopted),
            disposition="adopted" if adopted else "unused",
            unused_reason=unused_reason,
        )
        return request

    try:
        worker = OwnedPlanWorker(
            planner_factory, shutdown_timeout=shutdown_timeout_seconds
        )
        if initialize is not None:
            warmup = initialize(worker)
        base = clock.now()
        events.emit(
            "control_start",
            t_s=base,
            period_s=0.05,
            physics_step_seconds=0.05,
            execution_coupling="one_step_async",
            execution_clock="ready",
            execution_prefix_steps=1,
            synchronous_chunk_steps=1,
            fixed_rate_timing_applicable=False,
            max_steps=max_steps,
            bootstrap_action_slots=0,
            protocol_version=1,
        )
        current = snapshot_observation(observe_fn())
        capture = clock.now()
        if recorder is not None:
            recorder.initial(current, capture_s=capture)
        pending = submit(current, capture, 0, 0)
        current_request = receive(pending, adopted=True)
        pending = None
        while len(ticks) < max_steps:
            tick = len(ticks)
            # Always capture before the atomic environment step; preparation and
            # transport on the worker may start later and are measured separately.
            current = snapshot_observation(observe_fn())
            current_capture = clock.now()
            if tick + 1 < max_steps:
                pending = submit(current, current_capture, tick, tick + 1)
            action, fallback, age = timeline.execute(tick)
            if fallback:
                raise RuntimeError("One-step async cannot execute fallback commands")
            applied = clock.now()
            after, success, terminated = step_fn(action)
            computed = clock.now()
            execution.append((applied, computed))
            row = {
                "tick": tick,
                "cycle": tick,
                "request_id": current_request["request_id"],
                "source_tick": current_request["source_tick"],
                "target_tick": tick,
                "scheduled_s": None,
                "applied_s": applied,
                "lateness_s": None,
                "sensor_capture_s": current_capture,
                "sensor_release_s": applied,
                "next_observation_computed_s": computed,
                "execution_clock": "ready",
                "execution_coupling": "one_step_async",
                "physics_step_seconds": 0.05,
                "simulated_elapsed_seconds": (tick + 1) * 0.05,
                "action": action.tolist(),
                "fallback": False,
                "command_age_ticks": age,
                "observation_lag_ticks": tick - current_request["source_tick"],
                "action_delay_budget_ticks": 0,
                "command_ready_s": current_request["rpc_completed_s"],
                "ready_to_apply_seconds": applied - current_request["rpc_completed_s"],
                "first_application_of_request": True,
                "environment_step_s": computed - applied,
                "executed_slot": timeline.last_execution,
                "action_buffer": timeline.snapshot(),
                "action_buffer_phase": "after_execute",
                "feature_sequence": current_request["request_id"],
                "feature_source_tick": current_request["source_tick"],
                "feature_capture_s": current_request["capture_s"],
                "state_tick": current_request["source_tick"],
                "state_capture_s": current_request["capture_s"],
                "executed_image_age_ms": (applied - current_request["capture_s"])
                * 1000,
                "executed_state_age_ms": (applied - current_request["capture_s"])
                * 1000,
                "action_request_pending": pending is not None,
            }
            if recorder is not None:
                row["robot_response"] = recorder.step(current, after, row)
            ticks.append(row)
            events.emit("control_tick", t_s=applied, **row)
            events.emit(
                "cycle_phase",
                t_s=computed,
                cycle=tick,
                phase="execution",
                request_id=current_request["request_id"],
                request_tick=current_request["request_tick"],
                source_tick=current_request["source_tick"],
                target_tick=tick,
                start_s=applied,
                end_s=computed,
                simulation_paused=False,
                physics_start_tick=tick,
                physics_end_tick=tick + 1,
                execution_clock="ready",
            )
            events.emit(
                "one_step_async_execution",
                t_s=computed,
                request_id=current_request["request_id"],
                request_tick=current_request["request_tick"],
                source_tick=current_request["source_tick"],
                target_tick=tick,
                applied_s=applied,
                completed_s=computed,
                executed_steps=1,
                observation_lag_ticks=row["observation_lag_ticks"],
            )
            if record_video:
                frames.append(after["video.image"].copy())
            current = after
            if success or terminated or len(ticks) == max_steps:
                reason = (
                    "success"
                    if success
                    else "environment_terminated"
                    if terminated
                    else "max_steps"
                )
                controlled_end = clock.now()
                events.emit(
                    "control_end",
                    t_s=controlled_end,
                    control_steps=len(ticks),
                    reason=reason,
                )
                if pending is not None:
                    receive(pending, adopted=False, unused_reason=reason)
                    pending = None
                    drained = clock.now()
                    events.emit(
                        "cycle_phase",
                        t_s=drained,
                        cycle=tick + 1,
                        phase="drain",
                        request_id=tick + 1,
                        request_tick=tick + 1,
                        source_tick=tick,
                        target_tick=tick + 1,
                        start_s=controlled_end,
                        end_s=drained,
                        simulation_paused=True,
                        physics_start_tick=len(ticks),
                        physics_end_tick=len(ticks),
                        execution_clock="ready",
                    )
                break
            waiting = not pending.done()
            waiting_start = clock.now()
            current_request = receive(pending, adopted=True)
            pending = None
            waiting_end = clock.now()
            if waiting:
                waits.append((waiting_start, waiting_end))
                events.emit(
                    "cycle_phase",
                    t_s=waiting_end,
                    cycle=tick + 1,
                    phase="wait",
                    request_id=tick + 1,
                    request_tick=tick + 1,
                    source_tick=tick,
                    target_tick=tick + 1,
                    start_s=waiting_start,
                    end_s=waiting_end,
                    simulation_paused=True,
                    physics_start_tick=len(ticks),
                    physics_end_tick=len(ticks),
                    execution_clock="ready",
                )
        work_end = clock.now()
        failed = False
    except BaseException as error:
        events.emit(
            "control_error", error_type=type(error).__name__, message=str(error)
        )
        # A simulator failure can leave valid work in flight. Drain and preserve
        # it without executing or replacing the original failure.
        if pending is not None:
            try:
                request = pending.result(timeout=request_timeout_seconds)
                if all(request is not existing for existing in requests):
                    receive(pending, adopted=False, unused_reason="controller_error")
            except BaseException as drain_error:  # noqa: BLE001 - preserve original failure
                events.emit(
                    "drain_error",
                    error_type=type(drain_error).__name__,
                    message=str(drain_error),
                )
        raise
    finally:
        try:
            if worker is not None:
                worker.close()
        except BaseException as error:
            events.emit(
                "worker_shutdown_error",
                error_type=type(error).__name__,
                message=str(error),
            )
            failed = True
            raise
        finally:
            if failed:
                events.flush(trace)
    wall = controlled_end - base
    audits = [r["audit"] for r in requests]
    rpc_intervals = [(r["rpc_started_s"], r["rpc_completed_s"]) for r in requests]
    overlap = _intersection_seconds(rpc_intervals, execution)
    result = {
        "success": bool(success),
        "termination_reason": reason,
        "measurement_mode": "success_terminated_closed_loop",
        "execution_coupling": "one_step_async",
        "execution_clock": "ready",
        "execution_prefix_steps": 1,
        "synchronous_chunk_steps": 1,
        "control_steps": len(ticks),
        "physics_step_seconds": 0.05,
        "nominal_physics_hz": 20,
        "phase_target_hz": None,
        "target_control_hz": None,
        "fixed_rate_timing_applicable": False,
        "control_rate_valid": None,
        "mean_control_rate_valid": None,
        "control_deadline_valid": None,
        "control_period_valid": None,
        "control_deadline_misses_over5ms": None,
        "action_request_deadline_misses": None,
        "actual_control_hz": len(ticks) / wall if wall else None,
        "controlled_start_s": base,
        "controlled_end_s": controlled_end,
        "controlled_wall_seconds": wall,
        "scored_work_end_s": work_end,
        "scored_work_wall_seconds": work_end - base,
        "terminal_drain_seconds": work_end - controlled_end,
        "physical_control_seconds": len(ticks) * 0.05,
        "simulated_elapsed_seconds": len(ticks) * 0.05,
        "real_time_factor": len(ticks) * 0.05 / wall if wall else None,
        "first_success_tick": len(ticks) - 1 if success else None,
        "policy_calls": len(requests),
        "action_requests": len(requests),
        "adopted_policy_calls": len(ticks),
        "unused_policy_calls": len(requests) - len(ticks),
        "action_request_hz": len(requests) / wall if wall else None,
        "effective_plan_update_hz": len(ticks) / wall if wall else None,
        "total_policy_rpc_seconds": sum(b - a for a, b in rpc_intervals),
        "total_policy_request_seconds": sum(
            r["rpc_completed_s"] - r["submitted_s"] for r in requests
        ),
        "inference_wall_seconds": summarize_values([b - a for a, b in rpc_intervals]),
        "inference_request_seconds": summarize_values(
            [r["rpc_completed_s"] - r["submitted_s"] for r in requests]
        ),
        "bootstrap_inference_seconds": requests[0]["rpc_completed_s"]
        - requests[0]["submitted_s"],
        "execution_phase_seconds": summarize_values([b - a for a, b in execution]),
        "post_step_wait_seconds": sum(b - a for a, b in waits),
        "post_step_wait_intervals": summarize_values([b - a for a, b in waits]),
        "measured_inference_execution_overlap_seconds": overlap,
        "measured_server_execution_overlap_seconds": {
            stage: _intersection_seconds(
                [(a[f"{stage}_started_s"], a[f"{stage}_completed_s"]) for a in audits],
                execution,
            )
            for stage in ("vlm", "action")
        },
        "action_rpc_seconds": None,
        "action_stage_seconds": summarize_values(
            [a["action_completed_s"] - a["action_started_s"] for a in audits]
        ),
        "vlm_stage_seconds": summarize_values(
            [a["vlm_completed_s"] - a["vlm_started_s"] for a in audits]
        ),
        "action_ready_to_first_apply_seconds": summarize_values(
            [t["ready_to_apply_seconds"] for t in ticks]
        ),
        "executed_image_age_ms": summarize_values(
            [t["executed_image_age_ms"] for t in ticks]
        ),
        "forward_counts": {
            role: sum(a["forward_counts"][role] for a in audits)
            for role in ("vlm", "dit")
        },
        "scored_cold_capture_requests": sum(
            bool(a.get("graph_execution", {}).get("cold_capture")) for a in audits
        ),
        "unused_plan_suffix_slots": sum(a["decoded_horizon"] for a in audits)
        - len(ticks),
        "fallback_ticks": 0,
        "action_trace_sha256": hashlib.sha256(
            np.asarray([r["action"] for r in ticks], np.float32).tobytes()
        ).hexdigest(),
        "final_image_sha256": hashlib.sha256(
            current["video.image"].tobytes()
        ).hexdigest(),
    }
    if recorder is not None:
        result["replay_recording"] = recorder.status()
    if warmup is not None:
        result["warmup"] = warmup
    events.emit(
        "async_work_end",
        t_s=work_end,
        policy_calls=len(requests),
        adopted_policy_calls=len(ticks),
        unused_policy_calls=len(requests) - len(ticks),
    )
    result["replay_trace"] = events.status()
    # Completion is published only after all requests and worker cleanup succeed.
    events.flush(trace)
    return result, frames


class _PolicyPlanner:
    """The socket, including abandoned timeout sockets, belongs to this thread."""

    def __init__(self, port):
        from gr00t.policy.server_client import PolicyClient

        self.client = PolicyClient(host="127.0.0.1", port=port, timeout_ms=120000)

    def prepare(self, observation):
        from scripts.evaluate_libero_protocol import nested

        return nested(observation)

    def plan(self, prepared, capture_s, source_tick):
        return self.client.call_endpoint(
            "synchronous_plan",
            {
                "observation": prepared,
                "capture_s": capture_s,
                "source_tick": source_tick,
            },
        )

    def close(self):
        # PolicyClient recreates REQ sockets on timeouts. destroy(linger=0)
        # closes all sockets in this exclusively owned context without hanging.
        self.client.context.destroy(linger=0)


def one_step_async_episode(env, observation, args, seed, trace):
    from coexecution.replay.recording import ReplayRecorder
    from scripts.evaluate_libero_protocol import checked_env_step
    from scripts.libero_wallclock import RealClock

    clock = RealClock()
    recorder = (
        ReplayRecorder(max_frames=args.replay_max_frames)
        if args.record_replay
        else None
    )
    args.replay_recorder = recorder
    timeout = args.synchronous_warmup_timeout_seconds
    initial = snapshot_observation(observation)

    def initialize(worker):
        calls = []

        def warm(planner):
            with warmup_rpc_timeout(planner.client, timeout * 1000):
                return planner.plan(planner.prepare(initial), clock.now(), 0)

        for _ in range(args.slow_warmup_calls):
            calls.append(worker.call(warm, timeout=2 * timeout + 30)["audit"])
        worker.call(
            lambda planner: planner.client.call_endpoint(
                "reset", {"options": {"seed": seed}}
            ),
            timeout=270,
        )
        return {
            "scope": "before scored control and one episode RNG reset; no physics steps",
            "rpc_timeout_seconds": timeout,
            "rpc_timeout_scope": "per-send/per-receive warmup socket timeout; scored RPC restores120s; worker waits bounded separately",
            "calls": calls,
        }

    def observe():
        return env._process_observation(
            env._env.env._get_observations(force_update=True)
        )

    def step(action):
        obs, _, done, truncated, info = checked_env_step(env, action)
        return obs, bool(info["success"]), bool(done or truncated)

    return run_one_step_async_loop(
        step,
        observe,
        lambda: _PolicyPlanner(args.port),
        max_steps=args.max_steps,
        clock=clock,
        trace=trace,
        recorder=recorder,
        record_video=args.record_video,
        initialize=initialize,
    )
