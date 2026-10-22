"""πR² in a moving world with measured or fixed stage service times.

All GPU RPCs and env steps run serially on the caller thread. Logical futures
hide GPU results until the measured or declared completion; the next boundary
observation stays private while intervening logical events are processed. This
is an algorithm-time experiment, not a same-GPU concurrent deployment benchmark.
"""

import math
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, TypedDict

from coexecution.libero.scheduling import ResponsiveDelayBudget
from scripts.libero_protocol_scheduler import CommandTimeline, latency_budget
from scripts.libero_queue_trace import LatestFrameMailbox, QueueTrace


@dataclass(frozen=True)
class StageLatencies:
    """Fixed counterfactual stage durations; recovery never inherits normal1."""

    bootstrap_s: float
    s1_s: float
    s2_s: float
    recovery_s: float

    def __post_init__(self):
        for value in asdict(self).values():
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError("Stage durations must be finite and nonnegative")


@dataclass
class StepResult:
    """One real 50 ms physics step, with its freshly captured raw observation."""

    observation: Any
    success: bool = False
    terminated: bool = False


class StreamingResult(TypedDict):
    schema: str
    timing_mode: str
    stage_latencies: dict | None
    timing_domain: str
    resource_model: str
    compute_coupling: str
    latency_provenance: str
    logical_epoch_s: float
    control_origin_s: float
    startup_seconds: float
    requested_startup_release_seconds: float | None
    bootstrap_service_seconds: float
    startup_barrier_seconds: float
    startup_alignment_met: bool | None
    startup_policy: str
    steady_seconds: float
    physical_seconds: float
    action_steps: int
    success: bool
    terminated: bool
    events: list[dict]
    requests: list[dict]
    actions: list[dict]
    bootstrap: dict
    inflight_at_end: list[dict]
    final_buffer: dict
    config: dict


def timing_metadata(mode, latencies):
    """Describe service timing separately from physical execution coupling."""
    return {
        "timing_mode": mode,
        "timing_domain": "measured_request_discrete_event"
        if mode == "native"
        else "counterfactual_discrete_event",
        "latency_provenance": "measured_per_request"
        if mode == "native"
        else "fixed_counterfactual",
        "stage_latencies": None if latencies is None else asdict(latencies),
    }


def startup_configuration(release_seconds):
    """Validate an optional common physical release, separate from model timing."""
    if release_seconds is not None and (
        type(release_seconds) not in (int, float)
        or not math.isfinite(release_seconds)
        or release_seconds <= 0
    ):
        raise ValueError("Common startup release must be finite positive seconds")
    return {
        "requested_startup_release_seconds": None
        if release_seconds is None
        else float(release_seconds),
        "startup_policy": "paid_moving_world_hold"
        if release_seconds is None
        else "common_startup_release_moving_world_hold",
    }


STARTUP_MEASUREMENTS = (
    "bootstrap_service_seconds",
    "startup_barrier_seconds",
    "startup_seconds",
    "control_origin_s",
    "startup_alignment_met",
)


@dataclass(frozen=True)
class _Frame:
    observation: Any
    tick: int
    capture_s: float


class StreamingController:
    """Reuse command ownership/mailbox rules without wall-clock cross arithmetic.

    Call ``bootstrap`` on the scored initial observation. Then the runner must
    advance the real moving world for the receipt's ``startup_seconds`` holding the
    robot and recapture state. Pass that observation and its separate logical
    capture to ``run``. During ``run`` only step_fn advances the world; each call
    consumes exactly one 50 ms action, including real fallback commands.

    Completion at a control boundary is processed before the new observation is
    offered and before that tick executes. S2 completion precedes S1 completion
    at identical logical times; S1 submission occurs only at a control boundary.
    """

    period_s = 0.05

    def __init__(
        self,
        rpc,
        *,
        latencies=None,
        fixed_delay=None,
        epoch_s=1000.0,
        timing_mode="fixed",
        clock=None,
        startup_release_seconds=None,
    ):
        if timing_mode not in ("native", "fixed"):
            raise ValueError("Timing mode must be native or fixed")
        if timing_mode == "native" and latencies is not None:
            raise ValueError("Native timing rejects fixed stage latencies")
        if timing_mode == "fixed" and not isinstance(latencies, StageLatencies):
            raise TypeError("latencies must be StageLatencies")
        if (
            type(epoch_s) not in (float, int)
            or not math.isfinite(epoch_s)
            or epoch_s <= 0
        ):
            raise ValueError("Logical time requires a fixed positive epoch")
        if fixed_delay is not None and (
            type(fixed_delay) is not int or fixed_delay not in range(1, 6)
        ):
            raise ValueError("Fixed delay must be an integer in 1..5")
        self.rpc, self.latencies = rpc, latencies
        self._startup_configuration = startup_configuration(startup_release_seconds)
        self.timing_mode = timing_mode
        self.clock = time.monotonic if clock is None else clock
        self.fixed_delay, self.epoch_s = fixed_delay, float(epoch_s)
        initial, self.over_training_budget = (
            (1, False)
            if timing_mode == "native"
            else latency_budget(latencies.s1_s + 0.005, self.period_s)
        )
        self.initial_delay = fixed_delay if fixed_delay is not None else initial
        self.logical_time_s = self.epoch_s
        self._receipt = None
        self._ran = False
        self._generation = 0

    @property
    def bootstrap_evidence(self):
        """Detached successful receipt; callers cannot mutate private timing."""
        return deepcopy(self._receipt)

    def startup_metadata(self):
        receipt = self._receipt or {}
        return {
            **self._startup_configuration,
            **{key: receipt.get(key) for key in STARTUP_MEASUREMENTS},
        }

    def _call(self, method, *args, **kwargs):
        if self.timing_mode == "fixed":
            return method(*args, **kwargs), None
        started = self.clock()
        result = method(*args, **kwargs)
        completed = self.clock()
        if (
            any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in (started, completed)
            )
            or completed < started
            or not math.isfinite(completed - started)
        ):
            raise ValueError("Measured monotonic times must be finite and ordered")
        return result, {
            "clock": "host_monotonic",
            "started_s": started,
            "completed_s": completed,
            "duration_s": completed - started,
            "boundary": "full_adapter_call",
        }

    def bootstrap(self, raw_observation, language, *, seed=1000):
        # A bootstrap RPC can change server state even if its response/audit
        # fails. Revoke the previous run before making any new RPC.
        self._receipt = None
        self._ran = True
        self._generation += 1
        self.logical_time_s = self.epoch_s
        # A failed or late new bootstrap must not export the prior episode's
        # completed actions or pending requests as evidence for this attempt.
        self.trace = self.timeline = self.s1 = self.s2 = None
        self.requests, self.actions, self.by_tick = [], [], {}
        receipt, service_wall = self._call(
            self.rpc.bootstrap,
            raw_observation,
            language,
            capture_s=self.epoch_s,
            delay_ticks=self.initial_delay,
            seed=seed,
        )
        duration = (
            service_wall["duration_s"]
            if service_wall is not None
            else self.latencies.bootstrap_s
        )
        if service_wall is not None:
            receipt["service_wall"] = service_wall
        release = self._startup_configuration["requested_startup_release_seconds"]
        startup_seconds = duration if release is None else max(release, duration)
        receipt.update(
            logical_duration_s=duration,
            logical_completed_s=self.epoch_s + duration,
            timing_domain="logical_simulation",
            **self._startup_configuration,
            bootstrap_service_seconds=duration,
            startup_barrier_seconds=startup_seconds - duration,
            startup_seconds=startup_seconds,
            control_origin_s=self.epoch_s + startup_seconds,
            startup_alignment_met=None if release is None else duration <= release,
            generation=self._generation,
            receipt_id=uuid.uuid4().hex,
            language=language,
        )
        self._receipt = deepcopy(receipt)
        self._ran = False
        return deepcopy(self._receipt)

    def _emit(self, kind, **fields):
        self.trace.emit(kind, **fields)

    def _new_request(self, lane, result, duration, **metadata):
        wall = result["wall"]
        request = {
            "request_id": len(self.requests),
            "lane": lane,
            "logical_started_s": self.logical_time_s,
            "logical_completed_s": self.logical_time_s + duration,
            "logical_duration_s": duration,
            "status": "inflight",
            **deepcopy(wall),
            **metadata,
        }
        # Wall status describes the already completed physical RPC, not logical
        # availability. Preserve both explicitly in the exported record.
        request["wall_status"] = wall["status"]
        request["status"] = "inflight"
        self.requests.append(request)
        self._emit(
            "logical_request_started",
            **{
                k: v
                for k, v in request.items()
                if k not in {"server_audit", "model_actions"}
            },
        )
        return request

    def _start_s2(self):
        if self.s2 is not None or self.mailbox.pending is None:
            return
        frame = self.mailbox.take()
        result, service_wall = self._call(
            self.rpc.vision,
            frame.observation,
            self.language,
            capture_s=frame.capture_s,
            source_tick=frame.tick,
        )
        self.feature_sequence += 1
        result["sequence"] = self.feature_sequence
        request = self._new_request(
            "s2",
            result,
            self.latencies.s2_s if service_wall is None else service_wall["duration_s"],
            **({} if service_wall is None else {"service_wall": service_wall}),
            source_tick=frame.tick,
            capture_s=frame.capture_s,
            feature_sequence=result["sequence"],
            server_audit=deepcopy(result["meta"]),
        )
        self.s2 = (request, result)

    def _start_s1(self, frame):
        tick = frame.tick
        d = self.fixed_delay if self.fixed_delay is not None else self.budget.delay
        prefix = self.timeline.reserve(tick, d)
        feature = self.mailbox.latest
        age = frame.capture_s - feature["meta"]["capture_s"]
        result, service_wall = self._call(
            self.rpc.plan,
            frame.observation,
            self.language,
            request_tick=tick,
            delay_ticks=d,
            committed_actions=prefix.copy(),
            state_capture_s=frame.capture_s,
            feature=feature,
        )
        recovered = result["audit"]["buffer_restarted"]
        duration = (
            service_wall["duration_s"]
            if service_wall is not None
            else self.latencies.recovery_s
            if recovered
            else self.latencies.s1_s
        )
        request = self._new_request(
            "s1",
            result,
            duration,
            **({} if service_wall is None else {"service_wall": service_wall}),
            request_tick=tick,
            delay_ticks=d,
            state_capture_s=frame.capture_s,
            feature_sequence=feature["sequence"],
            feature_source_tick=feature["meta"]["source_tick"],
            feature_capture_s=feature["meta"]["capture_s"],
            feature_age_s=age,
            image_delay_ticks=max(0, round(age / self.period_s)),
            image_delay_embedding_ticks=min(5, max(0, round(age / self.period_s))),
            committed_actions=prefix.tolist(),
            server_audit=deepcopy(result["audit"]),
            model_actions=result["actions"].tolist(),
            action_buffer=self.timeline.snapshot(),
            normal_or_recovery="recovery5" if recovered else "normal1",
            over_training_budget=duration > 5 * self.period_s,
        )
        self.by_tick[tick] = request
        self.s1 = (request, result)
        self.next_request_tick = tick + d

    def _advance_events(self, boundary):
        while True:
            futures = [
                (future[0]["logical_completed_s"], order, lane, future)
                for order, (lane, future) in enumerate(
                    (("s2", self.s2), ("s1", self.s1))
                )
                if future is not None
            ]
            if not futures:
                break
            completed, _, lane, (request, result) = min(
                futures, key=lambda item: item[:2]
            )
            if completed > boundary + 1e-10:
                break
            self.logical_time_s = min(completed, boundary)
            request["status"] = "completed"
            if lane == "s2":
                self.s2 = None
                self.mailbox.publish(result, rpc_completed_s=None)
                request["publication"] = {"feature_sequence": result["sequence"]}
                self._emit(
                    "logical_request_completed",
                    request_id=request["request_id"],
                    lane=lane,
                )
                # At a boundary, run() must offer its fresh frame before the
                # next admission. Mid-step completions can consume only the
                # waiting historical frame; host-computed future state is private.
                # Deferral also avoids new work at the final control boundary.
                if not math.isclose(completed, boundary, rel_tol=0, abs_tol=1e-10):
                    self._start_s2()
            else:
                self.s1 = None
                audit = result["audit"]
                publication = self.timeline.publish(
                    audit["request_tick"],
                    audit["valid_start_tick"],
                    result["actions"],
                    self.next_unexecuted_tick,
                )
                request["publication"] = {
                    "installed_slots": publication.installed,
                    "expired_slots": publication.expired,
                    "protected_slots": publication.protected,
                    "next_unexecuted_tick": self.next_unexecuted_tick,
                    "logical_published_s": self.logical_time_s,
                }
                if self.fixed_delay is None:
                    request["budget_decision"] = self.budget.observe(
                        submitted_s=request["logical_started_s"],
                        completed_s=request["logical_completed_s"],
                        scheduled_s=request["logical_started_s"],
                        deadline_s=request["logical_started_s"]
                        + request["delay_ticks"] * self.period_s,
                    )
                self._emit(
                    "logical_request_completed",
                    request_id=request["request_id"],
                    lane=lane,
                    **request["publication"],
                    action_buffer=self.timeline.snapshot(),
                )
        self.logical_time_s = boundary

    def run(
        self,
        bootstrap,
        *,
        initial_observation,
        initial_capture_s,
        language,
        step_fn,
        max_steps,
    ) -> StreamingResult:
        """Consume one receipt capability, using its private successful snapshot.

        Only receipt_id, generation and language are read from the caller's
        receipt. Its other fields are detached evidence copies; mutating them
        cannot change the controller's actions, cache metadata or startup audit.
        """
        if self._receipt is None or self._ran:
            raise RuntimeError("Run requires a fresh bootstrap receipt")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps counts positive actual physical actions")
        startup_seconds = self._receipt["startup_seconds"]
        expected = self._receipt["control_origin_s"]
        if not math.isclose(initial_capture_s, expected, rel_tol=0, abs_tol=1e-10):
            raise ValueError(
                "Post-startup state capture must follow the paid startup hold"
            )
        if not isinstance(bootstrap, dict) or any(
            bootstrap.get(key) != self._receipt[key]
            for key in ("receipt_id", "generation")
        ):
            raise ValueError(
                "Bootstrap receipt is stale or belongs to another controller"
            )
        if (
            language != self._receipt["language"]
            or bootstrap.get("language") != language
        ):
            raise ValueError(
                "Run language must match the successful bootstrap language"
            )
        bootstrap = deepcopy(self._receipt)
        self._ran = True
        self.logical_time_s = expected
        self.control_origin_s, self.language = expected, language
        self.trace = QueueTrace(clock=lambda: self.logical_time_s)
        if bootstrap["startup_alignment_met"] is False:
            self._emit("startup_alignment_failed", **self.startup_metadata())
            raise RuntimeError(
                f"Bootstrap service {bootstrap['bootstrap_service_seconds']:.9g}s "
                f"exceeded requested startup release "
                f"{bootstrap['requested_startup_release_seconds']:.9g}s; "
                "actual hold preserved and no scored action admitted"
            )
        self.timeline = CommandTimeline(bootstrap["actions"])
        self.mailbox = LatestFrameMailbox(self.trace)
        self.mailbox.latest = {
            "sequence": 0,
            "meta": deepcopy(bootstrap["meta"]),
            "vl_embeds": None,
        }
        self.budget = ResponsiveDelayBudget(self.initial_delay, self.period_s)
        self.s1 = self.s2 = None
        self.next_request_tick = self.next_unexecuted_tick = self.feature_sequence = 0
        self.requests, self.actions, self.by_tick = [], [], {}
        observation = deepcopy(initial_observation)
        success = terminated = False
        self._emit(
            "control_started",
            startup_capture_s=bootstrap["capture_s"],
            state_capture_s=initial_capture_s,
            **self.startup_metadata(),
            action_buffer=self.timeline.snapshot(),
        )
        for tick in range(max_steps):
            boundary = self.control_origin_s + tick * self.period_s
            self._advance_events(boundary)
            frame = _Frame(deepcopy(observation), tick, boundary)
            self.mailbox.offer(frame)
            self._start_s2()
            self._advance_events(
                boundary
            )  # Zero-latency S2 is finite: one admitted frame.
            if self.s1 is None and tick >= self.next_request_tick:
                self._start_s1(frame)
                self._advance_events(boundary)
            action, fallback, queue_age = self.timeline.execute(tick)
            source = deepcopy(self.timeline.last_execution)
            producer = self.by_tick.get(source["producer_request_tick"])
            feature_capture = (
                bootstrap["capture_s"]
                if source["producer_kind"] == "bootstrap"
                else producer["feature_capture_s"]
                if producer
                else None
            )
            state_capture = (
                bootstrap["capture_s"]
                if source["producer_kind"] == "bootstrap"
                else producer["state_capture_s"]
                if producer
                else None
            )
            raw_action = action.copy()
            raw_action[6] = 1 - 2 * raw_action[6]
            record = {
                "tick": tick,
                "logical_started_s": boundary,
                "logical_completed_s": self.control_origin_s
                + (tick + 1) * self.period_s,
                "source": source,
                "fallback": fallback,
                "queue_age_ticks": queue_age,
                "feature_capture_s": feature_capture,
                "state_capture_s": state_capture,
                "feature_age_s": None
                if feature_capture is None
                else boundary - feature_capture,
                "state_age_s": None
                if state_capture is None
                else boundary - state_capture,
                "model_action": action.tolist(),
                "raw_action": raw_action.tolist(),
            }
            self._emit("action_started", **record)
            wall_start = time.monotonic()
            stepped = step_fn(raw_action.copy(), tick)
            wall_end = time.monotonic()
            if not isinstance(stepped, StepResult):
                raise TypeError(
                    "step_fn must return StepResult with fresh raw observation"
                )
            record.update(
                wall_started_s=wall_start,
                wall_completed_s=wall_end,
                wall_duration_s=wall_end - wall_start,
            )
            self.actions.append(record)
            self.next_unexecuted_tick = tick + 1
            # Run all mid-step events before publishing the newly computed state.
            self._advance_events(record["logical_completed_s"])
            self._emit(
                "action_completed",
                tick=tick,
                success=bool(stepped.success),
                terminated=bool(stepped.terminated),
            )
            observation = deepcopy(stepped.observation)
            success, terminated = bool(stepped.success), bool(stepped.terminated)
            if success or terminated:
                break
        inflight = []
        for future in (self.s1, self.s2):
            if future is not None:
                future[0]["status"] = "unapplied_at_control_end"
                inflight.append(deepcopy(future[0]))
        self.mailbox.drop_pending()
        self._emit(
            "control_ended",
            action_steps=len(self.actions),
            success=success,
            terminated=terminated,
            inflight_request_ids=[r["request_id"] for r in inflight],
        )
        steady = len(self.actions) * self.period_s
        return {
            "schema": "pir2.libero_dynamic.streaming.v1",
            **timing_metadata(self.timing_mode, self.latencies),
            "resource_model": "independent_stage_service",
            "compute_coupling": "serialized_gpu_simulation",
            "logical_epoch_s": self.epoch_s,
            **self.startup_metadata(),
            "steady_seconds": steady,
            "physical_seconds": startup_seconds + steady,
            "action_steps": len(self.actions),
            "success": success,
            "terminated": terminated,
            "events": self.trace.records(),
            "requests": deepcopy(self.requests),
            "actions": deepcopy(self.actions),
            "bootstrap": deepcopy(bootstrap),
            "inflight_at_end": inflight,
            "final_buffer": self.timeline.snapshot(),
            "config": {
                "stage_latencies": None
                if self.latencies is None
                else asdict(self.latencies),
                "period_s": self.period_s,
                "fixed_delay": self.fixed_delay,
                "initial_delay_ticks": self.initial_delay,
                "initial_over_training_budget": self.over_training_budget,
                **self.startup_metadata(),
                "boundary_order": "completion_then_offer_then_submit_then_execute",
                "equal_completion_order": "s2_then_s1",
            },
        }
