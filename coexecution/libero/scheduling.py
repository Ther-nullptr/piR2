"""Host-only admission timing shared by LIBERO deployment and experiment plans."""

import math


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
