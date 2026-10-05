"""Clock-independent command ownership and timestamp alignment for LIBERO."""

import math
from dataclasses import dataclass

import numpy as np


def summarize_values(values):
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def project_commands(commands):
    out = np.array(commands, dtype=np.float32, copy=True)
    if out.ndim != 2 or out.shape[1] != 7 or not np.isfinite(out).all():
        raise ValueError("Expected finite (T,7) commands")
    out[:, :6] = np.clip(out[:, :6], -1, 1)
    out[:, 6] = (out[:, 6] >= 0.5).astype(np.float32)
    return out


@dataclass
class Publication:
    installed: int
    expired: int
    protected: int


class CommandTimeline:
    """Single controller-thread owner; inference workers receive copies only."""

    def __init__(self, initial):
        initial = project_commands(initial)
        self.commands = {k: v.copy() for k, v in enumerate(initial)}
        self.origins = {k: 0 for k in self.commands}
        self.fallback_slots = set()
        self.committed_until = 0
        self.last = np.zeros(7, dtype=np.float32)
        self.last[-1] = 1.0

    def reserve(self, start, length):
        """Snapshot and commit [start,start+length), carrying gripper forward."""
        prefix = []
        previous = self.last.copy()
        for tick in range(start, start + length):
            if tick not in self.commands:
                hold = np.zeros(7, dtype=np.float32)
                hold[-1] = previous[-1]
                self.commands[tick] = hold
                self.origins[tick] = start
                self.fallback_slots.add(tick)
            previous = self.commands[tick]
            prefix.append(previous.copy())
        self.committed_until = max(self.committed_until, start + length)
        return np.stack(prefix)

    def publish(self, request_tick, start_tick, actions, next_tick):
        actions = project_commands(actions)
        installed = expired = protected = 0
        for offset, action in enumerate(actions):
            tick = start_tick + offset
            if tick < next_tick:
                expired += 1
                continue
            if tick < self.committed_until:
                protected += 1
                continue
            self.commands[tick] = action.copy()
            self.origins[tick] = request_tick
            self.fallback_slots.discard(tick)
            installed += 1
        return Publication(installed, expired, protected)

    def execute(self, tick):
        if tick not in self.commands:
            self.reserve(tick, 1)
        self.committed_until = max(self.committed_until, tick + 1)
        action = self.commands.pop(tick)
        origin = self.origins.pop(tick)
        fallback = tick in self.fallback_slots
        self.fallback_slots.discard(tick)
        self.last = action.copy()
        return action.copy(), fallback, tick - origin


def extra_buffer_shift(buffer_origin, request_tick):
    if request_tick < buffer_origin:
        raise ValueError("Control tick has not reached rolling-buffer origin")
    return request_tick - buffer_origin


def latency_budget(seconds, period, maximum=5):
    required = max(1, math.ceil(seconds / period))
    return min(required, maximum), required > maximum


def initial_frame_index(request_tick, visual_delay):
    wanted = request_tick - visual_delay
    return wanted, max(0, wanted), wanted < 0
