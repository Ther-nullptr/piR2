"""Independent control ticks and GPU workers; no inference wait in timed control."""

import gc
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from libero_protocol_scheduler import CommandTimeline, latency_budget, summarize_values


@dataclass(frozen=True)
class Observation:
    data: dict
    tick: int
    capture_s: float


def copy_observation(data):
    return {
        k: v if isinstance(v, str) else np.asarray(v).copy() for k, v in data.items()
    }


def to_nested(data):
    return {
        "state": {
            k[6:]: np.asarray(v, dtype=np.float32).reshape(1, 1, -1)
            for k, v in data.items()
            if k.startswith("state.")
        },
        "video": {
            k[6:]: np.asarray(v, dtype=np.uint8)[None, None]
            for k, v in data.items()
            if k.startswith("video.")
        },
        "language": {k: [[v]] for k, v in data.items() if isinstance(v, str)},
    }


class RealClock:
    @staticmethod
    def now():
        return time.monotonic()

    @staticmethod
    def sleep_until(deadline):
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)


class VisionWorker:
    def __init__(self, port):
        self.port = port
        self.condition = threading.Condition()
        self.pending = None
        self.latest = None
        self.stopping = False
        self.error = None
        self.submitted = 0
        self.completed = 0
        self.skipped_frames = 0
        self.latencies = []
        self.rpc_times = []
        self.stop_time = None
        self.dropped_at_stop = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def offer(self, observation):
        with self.condition:
            if self.pending is not None:
                self.skipped_frames += 1
            self.pending = observation
            self.condition.notify()

    def ready(self, observation):
        with self.condition:
            if self.error is not None:
                raise RuntimeError("VLM worker failed") from self.error
            result = self.latest
            if result and (
                result["meta"]["capture_s"] > observation.capture_s
                or result["meta"]["source_tick"] > observation.tick
            ):
                return None
            return result

    def _run(self):
        from gr00t.policy.server_client import PolicyClient

        client = PolicyClient(host="127.0.0.1", port=self.port, timeout_ms=120000)
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(
                        lambda: self.stopping or self.pending is not None
                    )
                    if self.stopping:
                        break
                    observation = self.pending
                    self.pending = None
                    self.submitted += 1
                start = time.monotonic()
                result = client.call_endpoint(
                    "vision",
                    {
                        "observation": to_nested(observation.data),
                        "capture_s": observation.capture_s,
                        "source_tick": observation.tick,
                        "return_features": True,
                    },
                )
                if result["meta"]["vlm_forward_calls"] != 1:
                    raise RuntimeError(
                        "A vision request must run the real VLM exactly once"
                    )
                finished = time.monotonic()
                self.latencies.append(finished - start)
                self.rpc_times.append(
                    {
                        "started_s": start,
                        "completed_s": finished,
                        "source_tick": observation.tick,
                        "capture_s": observation.capture_s,
                        "server_vlm_seconds": result["meta"]["vlm_seconds"],
                    }
                )
                with self.condition:
                    self.completed += 1
                    result["sequence"] = self.completed
                    self.latest = result
        except Exception as error:  # noqa: BLE001 -- propagate worker failures to the controller
            with self.condition:
                self.error = error
        finally:
            client.socket.close()
            client.context.term()

    def request_stop(self, cutoff=None):
        with self.condition:
            if self.stop_time is None:
                self.stop_time = time.monotonic() if cutoff is None else cutoff
            if self.pending is not None:
                self.dropped_at_stop += 1
            self.stopping = True
            self.pending = None
            self.condition.notify()

    def close(self):
        self.request_stop()
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("VLM worker failed") from self.error
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "skipped_sensor_frames": self.skipped_frames,
            "dropped_at_stop": self.dropped_at_stop,
            "started_after_control_end": sum(
                x["started_s"] > self.stop_time for x in self.rpc_times
            ),
            "completed_after_control_end": sum(
                x["completed_s"] > self.stop_time for x in self.rpc_times
            ),
            "rpc_seconds": list(self.latencies),
            "rpc_times": list(self.rpc_times),
        }


class ActionWorker:
    def __init__(self, port):
        self.port = port
        self.local = threading.local()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.completed = []

    def submit(
        self,
        observation,
        request_tick,
        delay_ticks,
        prefix,
        visual,
        deadline_s,
        submitted_s,
    ):
        return self.pool.submit(
            self._run,
            observation,
            request_tick,
            delay_ticks,
            prefix.copy(),
            visual,
            deadline_s,
            submitted_s,
        )

    def _run(self, observation, r, d, prefix, visual, deadline_s, submitted_s):
        from gr00t.policy.server_client import PolicyClient

        if not hasattr(self.local, "client"):
            self.local.client = PolicyClient(
                host="127.0.0.1", port=self.port, timeout_ms=120000
            )
            self.local.installed = -1
        client = self.local.client
        started = time.monotonic()
        if visual is not None and visual["sequence"] != self.local.installed:
            client.call_endpoint(
                "install", {"vl_embeds": visual["vl_embeds"], "meta": visual["meta"]}
            )
            self.local.installed = visual["sequence"]
        plan_rpc_start = time.monotonic()
        result = client.call_endpoint(
            "plan",
            {
                "observation": {"state": to_nested(observation.data)["state"]},
                "request_tick": r,
                "delay_ticks": d,
                "committed_actions": prefix,
                "state_capture_s": observation.capture_s,
                "output_scope": "native",
            },
        )
        result["client_timing"] = {
            "submitted_s": submitted_s,
            "started_s": started,
            "completed_s": time.monotonic(),
            "deadline_s": deadline_s,
            "cache_age_at_request_ms": max(
                0.0, (plan_rpc_start - result["audit"]["cache"]["capture_s"]) * 1000
            ),
            "state_age_at_request_ms": max(
                0.0, (plan_rpc_start - observation.capture_s) * 1000
            ),
        }
        self.completed.append(result)
        return result

    def close(self):
        self.pool.shutdown(wait=True)


def run_control_loop(
    step_fn,
    initial,
    initial_capture_s,
    initial_plan,
    actor,
    vision,
    initial_delay,
    period,
    max_steps,
    trace,
    record_video=False,
    clock=None,
):
    """The actor/vision interfaces are injectable so clock contracts can be tested."""
    clock = clock or RealClock()
    timeline = CommandTimeline(initial_plan)
    base = clock.now()
    next_data = copy_observation(initial)
    next_capture = initial_capture_s
    pending = None
    next_request = 0
    delay = initial_delay
    success = False
    reason = "time_limit"
    ticks = []
    request_records = []
    frames = []
    fallbacks = expired = protected = misses = over_budget = 0
    observation_events = []
    for tick in range(max_steps):
        target = base + tick * period
        clock.sleep_until(target)
        tick_started = clock.now()
        # A quickly computed o[k] is withheld until Tk. It is never exposed to
        # a GPU worker in the preceding physical interval.
        capture = next_capture if tick == 0 else tick_started
        current = Observation(next_data, tick, capture)
        observation_events.append((tick, tick_started, target))
        vision.offer(current)
        if pending is not None and pending.done():
            result = pending.result()
            pending = None
            audit = result["audit"]
            timing = result["client_timing"]
            publication = timeline.publish(
                audit["request_tick"],
                audit["valid_start_tick"],
                result["actions"],
                next_tick=tick,
            )
            expired += publication.expired
            protected += publication.protected
            late = timing["completed_s"] > timing["deadline_s"]
            misses += int(late)
            budget, ood = latency_budget(
                timing["completed_s"] - timing["submitted_s"] + 0.005, period
            )
            over_budget += int(ood)
            if late:
                delay = max(delay, budget)
            record = {
                "kind": "request_completed",
                "consumed_at_tick": tick,
                "expired_slots": publication.expired,
                "protected_slots": publication.protected,
                "installed_slots": publication.installed,
                "deadline_miss": late,
                "audit": audit,
                "client_timing": timing,
            }
            request_records.append(record)
            trace.write(json.dumps(record) + "\n")
        if pending is None and tick >= next_request:
            prefix = timeline.reserve(tick, delay)
            visual = vision.ready(current)
            submitted = clock.now()
            pending = actor.submit(
                current,
                tick,
                delay,
                prefix,
                visual,
                base + (tick + delay) * period,
                submitted,
            )
            next_request = tick + delay
        action, fallback, command_age = timeline.execute(tick)
        fallbacks += int(fallback)
        applied = clock.now()
        next_data, success, terminated = step_fn(action)
        computed = clock.now()
        next_data = copy_observation(next_data)
        next_capture = computed
        row = {
            "kind": "control_tick",
            "tick": tick,
            "scheduled_s": target,
            "applied_s": applied,
            "lateness_s": max(0, applied - target),
            "sensor_capture_s": capture,
            "sensor_release_s": tick_started,
            "next_observation_computed_s": computed,
            "next_observation_release_not_before_s": base + (tick + 1) * period,
            "environment_step_s": computed - applied,
            "action": action.tolist(),
            "fallback": fallback,
            "command_age_ticks": command_age,
            "action_delay_budget_ticks": delay,
            "action_request_pending": pending is not None and not pending.done(),
        }
        ticks.append(row)
        trace.write(json.dumps(row) + "\n")
        if record_video:
            frames.append(next_data["video.image"].copy())
        if success or terminated:
            reason = "success" if success else "environment_terminated"
            clock.sleep_until(base + (tick + 1) * period)
            break
    else:
        clock.sleep_until(base + max_steps * period)
    controlled_end = clock.now()
    vision.request_stop(controlled_end)
    # Waiting is allowed only after the timed control episode has ended.
    if pending is not None:
        result = pending.result()
        expired_at_end = max(
            0,
            min(
                len(result["actions"]), len(ticks) - result["audit"]["valid_start_tick"]
            ),
        )
        expired += expired_at_end
        trace.write(
            json.dumps(
                {
                    "kind": "unused_result_at_episode_end",
                    "expired_before_episode_end": expired_at_end,
                    "unneeded_future_slots": len(result["actions"]) - expired_at_end,
                    "audit": result["audit"],
                    "client_timing": result["client_timing"],
                }
            )
            + "\n"
        )
    actor.close()
    vlm_stats = vision.close()
    for timing, elapsed in zip(
        vlm_stats.get("rpc_times", []), vlm_stats.get("rpc_seconds", []), strict=True
    ):
        trace.write(
            json.dumps({"kind": "vision_request", "rpc_seconds": elapsed, **timing})
            + "\n"
        )
    metric_summary = summarize_values

    completed = actor.completed
    rpc = [
        x["client_timing"]["completed_s"] - x["client_timing"]["submitted_s"]
        for x in completed
    ]
    cache_age = [
        x["client_timing"].get("cache_age_at_request_ms", x["audit"]["image_delay_ms"])
        for x in completed
        if x["audit"]["image_delay_ms"] is not None
    ]
    compute = [x["audit"]["server_seconds"] for x in completed]
    extra = [x["audit"]["extra_buffer_shift"] for x in completed]
    applied = [x["applied_s"] for x in ticks]
    hz = (len(applied) - 1) / (applied[-1] - applied[0]) if len(applied) > 1 else None
    lateness = [x["lateness_s"] for x in ticks]
    mean_rate_valid = hz is not None and abs(hz - 1 / period) / (1 / period) <= 0.02
    intervals = np.diff(applied).tolist()
    deadline_valid = max(lateness) <= 0.005
    period_valid = bool(intervals) and max(abs(x - period) for x in intervals) <= 0.005
    valid = mean_rate_valid and deadline_valid and period_valid
    result = {
        "success": bool(success),
        "termination_reason": reason,
        "control_steps": len(ticks),
        "target_control_hz": 1 / period,
        "actual_control_hz": hz,
        "control_rate_valid": valid,
        "mean_control_rate_valid": mean_rate_valid,
        "control_deadline_valid": deadline_valid,
        "control_period_valid": period_valid,
        "actual_control_intervals_seconds": metric_summary(intervals),
        "controlled_wall_seconds": controlled_end - base,
        "control_lateness_seconds": metric_summary(lateness),
        "control_deadline_misses_over5ms": sum(x > 0.005 for x in lateness),
        "environment_step_seconds": metric_summary(
            [x["environment_step_s"] for x in ticks]
        ),
        "action_requests": len(completed),
        "action_request_hz": len(completed) / (controlled_end - base),
        "effective_plan_update_hz": sum(
            x["installed_slots"] > 0 for x in request_records
        )
        / (controlled_end - base),
        "action_request_deadline_misses": sum(
            x["client_timing"]["deadline_s"] <= controlled_end
            and x["client_timing"]["completed_s"] > x["client_timing"]["deadline_s"]
            for x in completed
        ),
        "action_rpc_seconds": metric_summary(rpc),
        "action_server_seconds": metric_summary(compute),
        "cache_age_ms": metric_summary(cache_age),
        "visual_state_timestamp_gap_ms": metric_summary(
            [x["audit"]["image_delay_ms"] for x in completed]
        ),
        "state_age_at_action_request_ms": metric_summary(
            [x["client_timing"].get("state_age_at_request_ms", 0.0) for x in completed]
        ),
        "expired_action_slots": expired,
        "protected_action_slots": protected,
        "fallback_ticks": fallbacks,
        "buffer_realignment_ticks": sum(extra),
        "buffer_restarts": sum(
            x["audit"].get("buffer_restarted", False) for x in completed
        ),
        "over_training_budget_events": over_budget,
        "rpc_over_training_budget_count": sum(
            latency_budget(seconds + 0.005, period)[1] for seconds in rpc
        ),
        "initial_action_delay_ticks": initial_delay,
        "final_action_delay_ticks": delay,
        "vlm_work": {
            k: v for k, v in vlm_stats.items() if k not in {"rpc_seconds", "rpc_times"}
        },
        "vlm_rpc_seconds": metric_summary(vlm_stats.get("rpc_seconds", [])),
        "forward_counts": {
            "dit": sum(x["audit"]["forward_counts"]["dit"] for x in completed),
            "action_gpu_vlm": sum(
                x["audit"]["forward_counts"]["vlm"] for x in completed
            ),
            "slow_gpu_vlm": vlm_stats["completed"],
        },
        "sensor_release_contract_verified": all(
            released >= target for _, released, target in observation_events
        ),
    }
    return result, frames


def calibrate(client, observation, capture_s, seed, period, visual, port):
    boot = client.call_endpoint(
        "bootstrap",
        {
            "observation": to_nested(observation),
            "capture_s": capture_s,
            "delay_ticks": 1,
            "seed": seed,
        },
    )
    queue = CommandTimeline(boot["actions"])
    samples = []
    actor = ActionWorker(port)
    try:
        for tick in range(12):
            prefix = queue.reserve(tick, 1)
            # Exercise the real cross-GPU cache transport on every calibration call.
            feature = {**visual, "sequence": tick + 1}
            submitted = time.monotonic()
            result = actor.submit(
                Observation(observation, tick, capture_s),
                tick,
                1,
                prefix,
                feature,
                submitted + period,
                submitted,
            ).result()
            elapsed = result["client_timing"]["completed_s"] - submitted
            if tick >= 2:
                samples.append(elapsed)
            queue.publish(
                tick,
                result["audit"]["valid_start_tick"],
                result["actions"],
                next_tick=tick,
            )
            queue.execute(tick)
    finally:
        actor.close()
    d, ood = latency_budget(float(np.percentile(samples, 95)) + 0.005, period)
    return d, {
        "sample_count": len(samples),
        "p95_rpc_seconds": float(np.percentile(samples, 95)),
        "margin_seconds": 0.005,
        "selected_delay_ticks": d,
        "over_training_budget": ood,
        "includes_cache_install_and_thread_queue": True,
        "scope": "warm GPU/RPC timing; reset and fresh sensor capture follow before scored control",
    }


def deployment_episode(env, observation, client, args, seed, trace):
    # Calibration is per condition; formal episodes always reset the policy RNG
    # and rolling buffer afterward. Simulation does not advance during calibration.
    period = 1 / args.control_hz
    capture = time.monotonic()
    if not hasattr(args, "slow_warmup"):
        from gr00t.policy.server_client import PolicyClient

        slow = PolicyClient(host="127.0.0.1", port=args.slow_port, timeout_ms=120000)
        records = []
        for _ in range(2):
            started = time.monotonic()
            visual = slow.call_endpoint(
                "vision",
                {
                    "observation": to_nested(observation),
                    "capture_s": capture,
                    "source_tick": 0,
                    "return_features": True,
                },
            )
            assert visual["meta"]["vlm_forward_calls"] == 1
            records.append(
                {
                    "rpc_seconds": time.monotonic() - started,
                    "vlm_seconds": visual["meta"]["vlm_seconds"],
                    "vlm_forward_calls": 1,
                }
            )
        del slow
        args.slow_warmup = {
            "scope": "outside scored control, repeated for each condition/process",
            "calls": records,
        }
        (args.output / "slow-warmup.json").write_text(
            json.dumps(args.slow_warmup, indent=2) + "\n"
        )
        calibration_path = args.output / "calibration.json"
        if calibration_path.exists():
            calibration = json.loads(calibration_path.read_text())
            d = calibration["selected_delay_ticks"]
        else:
            d, calibration = calibrate(
                client, observation, capture, seed, period, visual, args.port
            )
        args.calibration = (d, calibration)
        calibration_path.write_text(json.dumps(calibration, indent=2) + "\n")
    d, calibration = args.calibration
    gc.collect()
    observation = env._process_observation(
        env._env.env._get_observations(force_update=True)
    )
    capture = time.monotonic()
    boot = client.call_endpoint(
        "bootstrap",
        {
            "observation": to_nested(observation),
            "capture_s": capture,
            "delay_ticks": d,
            "seed": seed,
        },
    )
    actor = ActionWorker(args.port)
    vision = VisionWorker(args.slow_port)

    def advance(action):
        from evaluate_libero_protocol import checked_env_step

        obs, _, done, truncated, info = checked_env_step(env, action)
        return obs, bool(info["success"]), bool(done or truncated)

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        result, frames = run_control_loop(
            advance,
            observation,
            capture,
            boot["actions"],
            actor,
            vision,
            d,
            period,
            args.max_steps,
            trace,
            args.record_video,
        )
    finally:
        if gc_was_enabled:
            gc.enable()
        actor.close()
        vision.close()
    result["bootstrap_seconds"] = boot["bootstrap_seconds"]
    result["bootstrap_forward_counts"] = boot["forward_counts"]
    result["calibration"] = calibration
    result["cyclic_gc_disabled_during_control"] = True
    return result, frames
