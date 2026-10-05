"""Test actual controller scheduling with fake time/transport, not model accuracy."""

import io
import unittest

import numpy as np
from libero_wallclock import run_control_loop


class Clock:
    def __init__(self):
        self.t = 10.0
        self.events = []

    def now(self):
        return self.t

    def sleep_until(self, t):
        self.t = max(self.t, t)


class Future:
    def __init__(self, actor, request):
        self.actor = actor
        self.request = request
        self.cached = None

    def done(self):
        return self.actor.clock.now() >= self.request["complete"]

    def result(self):
        if not self.done():
            self.actor.clock.events.append(("wait_action", self.actor.clock.now()))
        self.actor.clock.sleep_until(self.request["complete"])
        if self.cached is None:
            r = self.request
            d = r["d"]
            self.cached = {
                "actions": np.full((d, 7), 0.1),
                "audit": {
                    "request_tick": r["tick"],
                    "d_request": d,
                    "valid_start_tick": r["tick"] + d,
                    "valid_end_tick": r["tick"] + 2 * d,
                    "extra_buffer_shift": 0,
                    "image_delay_ms": 50.0,
                    "server_seconds": self.actor.latency,
                    "forward_counts": {"dit": 1, "vlm": 0},
                },
                "client_timing": {
                    "submitted_s": r["start"],
                    "started_s": r["start"],
                    "completed_s": r["complete"],
                    "deadline_s": r["deadline"],
                },
            }
            self.actor.completed.append(self.cached)
        return self.cached


class Actor:
    def __init__(self, clock, latency):
        self.clock = clock
        self.latency = latency
        self.completed = []

    def submit(self, obs, r, d, prefix, visual, deadline, submitted):
        return Future(
            self,
            {
                "tick": r,
                "d": d,
                "start": submitted,
                "complete": submitted + self.latency,
                "deadline": deadline,
            },
        )

    def close(self):
        pass


class Vision:
    def __init__(self, clock):
        self.clock = clock
        self.offers = []

    def offer(self, obs):
        self.offers.append((obs.tick, self.clock.now()))

    def ready(self, obs):
        return None

    def request_stop(self, cutoff=None):
        self.clock.events.append(("stop_vision", self.clock.now()))

    def close(self):
        return {"submitted": 0, "completed": 0, "rpc_seconds": []}


class ControlTiming(unittest.TestCase):
    def run_case(self, latency, slow_first_step=False, steps=12):
        clock = Clock()
        actor = Actor(clock, latency)
        vision = Vision(clock)
        calls = []

        def env_step(action):
            calls.append(clock.now())
            clock.t += 0.09 if slow_first_step and len(calls) == 1 else 0.01
            return {"state.x": [len(calls)]}, False, False

        result, _ = run_control_loop(
            env_step,
            {"state.x": [0]},
            clock.now(),
            np.zeros((40, 7)),
            actor,
            vision,
            1,
            0.05,
            steps,
            io.StringIO(),
            clock=clock,
        )
        self.last_clock = clock
        return result, calls, vision.offers

    def test_control_keeps20hz_when_inference_is_slow(self):
        result, ticks, _ = self.run_case(0.18)
        np.testing.assert_allclose(ticks, 10.0 + np.arange(12) * 0.05, atol=1e-9)
        self.assertGreater(result["action_request_deadline_misses"], 0)
        self.assertGreater(result["expired_action_slots"], 0)
        self.assertAlmostEqual(result["actual_control_hz"], 20)

    def test_future_simulator_observation_is_withheld(self):
        _, ticks, offers = self.run_case(0.02)
        for tick, released in offers:
            self.assertGreaterEqual(released, 10 + tick * 0.05)
        # Physics computed o1 at10.01, but VLM cannot receive it before10.05.
        self.assertEqual(offers[1][0], 1)
        self.assertAlmostEqual(offers[1][1], 10.05)
        self.assertAlmostEqual(ticks[1], 10.05)

    def test_average20hz_does_not_hide_a_long_control_gap(self):
        result, ticks, _ = self.run_case(0.02, slow_first_step=True)
        self.assertTrue(result["mean_control_rate_valid"])
        self.assertFalse(result["control_deadline_valid"])
        self.assertFalse(result["control_period_valid"])
        self.assertFalse(result["control_rate_valid"])
        self.assertAlmostEqual(ticks[1] - ticks[0], 0.09)

    def test_stop_vision_before_draining_a_late_action(self):
        result, _, _ = self.run_case(1.0, steps=2)
        names = [name for name, _ in self.last_clock.events]
        self.assertLess(names.index("stop_vision"), names.index("wait_action"))
        self.assertEqual(result["expired_action_slots"], 1)


if __name__ == "__main__":
    unittest.main()
