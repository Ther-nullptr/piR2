"""Host-only admission timing shared by LIBERO deployment and experiment plans."""

import math


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
