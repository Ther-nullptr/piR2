"""Host-only admission timing shared by LIBERO deployment and experiment plans."""

import math
from typing import ClassVar


def validate_execution_clock(
    mode, *, protocol="deployment", variant="flow", fixed_delay=None
):
    if mode not in ("paced", "ready"):
        raise ValueError("Execution clock must be paced or ready")
    if mode == "ready" and (
        protocol != "deployment" or variant != "flow" or fixed_delay is not None
    ):
        raise ValueError(
            "Ready execution requires deployment Flow without a fixed wall delay"
        )


class ExecutionCadence:
    """Keep the simulator's logical step separate from optional wall pacing.

    Ready mode does not predict a future actuator boundary from wall time. The
    controller can accept completed work only between atomic environment steps.
    """

    def __init__(self, mode, period_s, origin_s):
        validate_execution_clock(mode)
        if not math.isfinite(period_s) or period_s <= 0:
            raise ValueError("Physics step must be finite and positive")
        self.mode, self.period_s, self.origin_s = mode, period_s, origin_s

    @property
    def paced(self):
        return self.mode == "paced"

    def target(self, tick):
        return self.origin_s + tick * self.period_s if self.paced else None

    def wait(self, clock, tick):
        target = self.target(tick)
        if target is not None:
            clock.sleep_until(target)
        return target


def validate_vision_max_hz(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("Vision maximum Hz must be finite and nonnegative")


class AdmissionSchedule:
    """Space actual admissions without reserving observations or catch-up slots.

    The caller owns synchronization, wakeable waits and the monotonic clock.
    Zero disables the limit; the first admission is always immediately eligible.
    """

    def __init__(self, max_hz=0):
        validate_vision_max_hz(max_hz)
        self.max_hz = max_hz
        self.period_s = 1 / max_hz if max_hz else 0
        self.next_admission_s = None

    def remaining(self, now_s):
        if self.next_admission_s is None:
            return 0.0
        return max(0.0, self.next_admission_s - now_s)

    def record_admission(self, admitted_s):
        self.next_admission_s = admitted_s + self.period_s if self.max_hz else None


def validate_deployment_scheduling_policy(
    policy,
    *,
    protocol="deployment",
    variant="pir2",
    execution_coupling="asynchronous",
    execution_clock="paced",
    fixed_delay=None,
):
    if policy not in ("conservative", "responsive"):
        raise ValueError("Unknown deployment scheduling policy")
    if policy == "responsive" and (
        protocol != "deployment"
        or variant != "pir2"
        or execution_coupling != "asynchronous"
        or execution_clock != "paced"
        or fixed_delay is not None
    ):
        raise ValueError(
            "Responsive scheduling requires deployment piR2 with cached asynchronous "
            "workers, paced physics and no fixed delay override"
        )


class ResponsiveDelayBudget:
    """Per-episode budget from valid completed RPCs, all durations in seconds.

    The rolling P95 uses linear interpolation at (n-1)*0.95. Add 5 ms, then
    ceil(duration / physical period) and clamp to 1..5 ticks. Three consecutive
    full-window observations supporting a lower target lower only one tick.
    A missed deadline immediately raises using the latest response measured
    both from actual submission and from its scheduled physical boundary.
    This selects future commitments only; existing output validity is immutable.
    """

    config: ClassVar[dict] = {
        "window_size": 8,
        "percentile": 95,
        "margin_seconds": 0.005,
        "lower_observations": 3,
        "minimum_ticks": 1,
        "maximum_ticks": 5,
        "quantile_method": "linear",
        "rounding": "ceil(seconds / physical_period_seconds)",
    }

    def __init__(self, initial_delay, period_s):
        if type(initial_delay) is not int or initial_delay not in range(1, 6):
            raise ValueError("Initial delay must be an integer in 1..5")
        if not math.isfinite(period_s) or period_s <= 0:
            raise ValueError("Physical period must be finite and positive")
        self.delay = initial_delay
        self.period_s = period_s
        self.samples = ()
        self.lower_streak = 0

    def _required(self, seconds):
        # Tolerate floating point cancellation at exact tick boundaries only.
        return max(1, math.ceil(seconds / self.period_s - 1e-12))

    def observe(self, *, submitted_s, completed_s, scheduled_s, deadline_s):
        old = self.delay
        valid = (
            all(
                type(value) in (int, float) and math.isfinite(value)
                for value in (submitted_s, completed_s, scheduled_s, deadline_s)
            )
            and scheduled_s <= submitted_s <= completed_s
            and scheduled_s < deadline_s
        )
        decision = {
            "previous_delay_ticks": old,
            "sample_valid": valid,
            "sample_seconds": completed_s - submitted_s if valid else None,
            "scheduled_request_s": scheduled_s if valid else None,
            "deadline_s": deadline_s if valid else None,
            "deadline_miss": None,
            "candidate_delay_ticks": None,
            "latest_required_ticks": None,
            "scheduled_required_ticks": None,
            "over_training_budget": False,
            "reason": "invalid_sample",
        }
        if valid:
            margin = self.config["margin_seconds"]
            self.samples = (*self.samples, completed_s - submitted_s)[-8:]
            ordered = sorted(self.samples)
            index = (len(ordered) - 1) * 0.95
            low, high = math.floor(index), math.ceil(index)
            quantile = ordered[low] + (ordered[high] - ordered[low]) * (index - low)
            required = self._required(quantile + margin)
            candidate = min(5, required)
            latest = self._required(completed_s - submitted_s + margin)
            scheduled = self._required(completed_s - scheduled_s + margin)
            late = completed_s > deadline_s
            decision.update(
                deadline_miss=late,
                percentile_seconds=quantile,
                candidate_delay_ticks=candidate,
                latest_required_ticks=latest,
                scheduled_required_ticks=scheduled,
                over_training_budget=max(required, latest, scheduled) > 5,
                reason="hold",
            )
            if late:
                self.delay = min(5, max(old + 1, latest, scheduled))
                self.lower_streak = 0
                decision["reason"] = (
                    "raise_deadline_miss" if self.delay > old else "hold_at_maximum"
                )
            elif len(self.samples) == 8 and candidate < old:
                self.lower_streak += 1
                decision["reason"] = "hold_lower_hysteresis"
                if self.lower_streak >= 3:
                    self.delay = old - 1
                    self.lower_streak = 0
                    decision["reason"] = "lower_full_window_hysteresis"
            else:
                self.lower_streak = 0
                if len(self.samples) < 8:
                    decision["reason"] = "hold_window_incomplete"
        else:
            self.lower_streak = 0
        return {
            **decision,
            "samples_seconds": list(self.samples),
            "lower_streak": self.lower_streak,
            "chosen_delay_ticks": self.delay,
        }


class FutureSignal:
    """Wake the controller; callbacks capture only an Event, never controller state.

    clock.wait_for_signal(event, absolute_deadline_s) is injectable for fake-time
    tests. There is one bounded wait per pacing gap and no callback-side result
    reads, timeline mutations or inference submissions.
    """

    def __init__(self, future):
        from threading import Event

        signal = Event()
        future.add_done_callback(lambda _future: signal.set())
        self.future, self.signal = future, signal

    def wait_until(self, clock, deadline_s):
        if not self.future.done() and clock.now() < deadline_s:
            clock.wait_for_signal(self.signal, deadline_s)
        return self.future.done()


def publish_completed_result(timeline, result, *, next_tick, observed_s, clock):
    """Controller-thread admission; report actual installed ownership separately.

    The valid interval belongs to the backend and is never retimed. The
    snapshot is captured after publication, including between-tick changes.
    """
    audit, timing = result["audit"], result["client_timing"]
    publication = timeline.publish(
        audit["request_tick"], audit["valid_start_tick"], result["actions"], next_tick
    )
    processed_s = clock.now()
    return {
        "kind": "request_completed",
        "t_s": processed_s,
        "consumed_at_tick": next_tick,
        "controller_ready_observed_s": observed_s,
        "controller_processed_s": processed_s,
        "rpc_completed_s": timing["completed_s"],
        "ready_to_adopt_seconds": processed_s - timing["completed_s"],
        "expired_slots": publication.expired,
        "protected_slots": publication.protected,
        "installed_slots": publication.installed,
        "publication_status": "installed" if publication.installed else "discarded",
        "action_buffer": timeline.snapshot(),
        "action_buffer_phase": "after_publication_before_next_tick",
        "audit": audit,
        "client_timing": timing,
    }
