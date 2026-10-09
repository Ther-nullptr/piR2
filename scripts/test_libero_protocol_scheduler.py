"""Scheduler contract tests only; real model inference is validated separately on GPU."""

import unittest

import numpy as np
from libero_protocol_scheduler import (
    CommandTimeline,
    extra_buffer_shift,
    initial_frame_index,
    latency_budget,
)


class TimingContracts(unittest.TestCase):
    def test_snapshot_tracks_producer_without_changing_command_ownership(self):
        queue = CommandTimeline(np.zeros((6, 7)))
        queue.reserve(0, 2)
        queue.publish(3, 1, np.ones((4, 7)), next_tick=1)
        snapshot = queue.snapshot()
        slots = {slot["tick"]: slot for slot in snapshot["slots"]}
        self.assertEqual(snapshot["committed_until"], 2)
        self.assertEqual(slots[1]["status"], "committed")
        self.assertEqual(slots[1]["producer_kind"], "bootstrap")
        self.assertIsNone(slots[1]["producer_request_tick"])
        self.assertEqual(slots[2]["status"], "future")
        self.assertEqual(slots[2]["producer_request_tick"], 3)
        slots[1]["producer_kind"] = "mutated"
        queue.execute(1)
        self.assertEqual(queue.last_execution["producer_kind"], "bootstrap")
        self.assertNotIn(1, [slot["tick"] for slot in queue.snapshot()["slots"]])

    def test_fallback_snapshot_never_claims_a_model_producer(self):
        queue = CommandTimeline(np.zeros((1, 7)))
        queue.reserve(0, 3)
        slots = queue.snapshot()["slots"]
        self.assertEqual(slots[2]["status"], "fallback")
        self.assertTrue(slots[2]["committed"])
        self.assertTrue(slots[2]["fallback"])
        self.assertIsNone(slots[2]["producer_request_tick"])
        self.assertEqual(slots[2]["reserved_at_tick"], 0)
        queue.execute(5)
        self.assertTrue(queue.last_execution["fallback"])
        self.assertEqual(queue.last_execution["reserved_at_tick"], 5)

    def test_old_request_defines_which_actions_are_clean(self):
        queue = CommandTimeline(np.zeros((6, 7)))
        queue.reserve(0, 1)
        queue.execute(0)
        queue.execute(1)
        # piR2 request r=0,d=1 has exactly one new clean action at tick1.
        pub = queue.publish(0, 1, np.ones((1, 7)), next_tick=2)
        self.assertEqual((pub.installed, pub.expired), (0, 1))
        self.assertTrue(np.array_equal(queue.execute(2)[0], np.zeros(7)))

    def test_committed_prefix_is_never_overwritten(self):
        queue = CommandTimeline(np.zeros((6, 7)))
        before = queue.reserve(0, 3)
        pub = queue.publish(0, 0, np.ones((6, 7)), next_tick=0)
        self.assertEqual(pub.protected, 3)
        for k in range(3):
            self.assertTrue(np.array_equal(queue.execute(k)[0], before[k]))
        self.assertTrue(np.array_equal(queue.execute(3)[0], np.ones(7)))

    def test_fallback_gripper_follows_future_committed_state(self):
        initial = np.zeros((2, 7))
        initial[0, -1] = 1
        initial[1, -1] = 0
        queue = CommandTimeline(initial)
        prefix = queue.reserve(0, 4)
        self.assertEqual(prefix[:, -1].tolist(), [1, 0, 0, 0])
        self.assertTrue(queue.execute(2)[1])

    def test_no_double_slide_and_no_early_future_buffer(self):
        self.assertEqual(extra_buffer_shift(4, 4), 0)
        self.assertEqual(extra_buffer_shift(4, 6), 2)
        with self.assertRaises(ValueError):
            extra_buffer_shift(4, 3)

    def test_padding_is_explicit_and_budget_is_bounded(self):
        self.assertEqual(initial_frame_index(1, 3), (-2, 0, True))
        self.assertEqual(initial_frame_index(8, 3), (5, 5, False))
        self.assertEqual(latency_budget(0.12, 0.05), (3, False))
        self.assertEqual(latency_budget(0.7, 0.05), (5, True))


if __name__ == "__main__":
    unittest.main()
