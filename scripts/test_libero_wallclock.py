"""Test actual controller scheduling with fake time/transport, not model accuracy."""

import inspect
import io
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from libero_wallclock import run_control_loop


class ServiceFloors(unittest.TestCase):
    def test_floor_holds_until_anchor_and_retains_raw_completion(self):
        from libero_wallclock import release_service

        clock = Clock()
        clock.t = 10.025
        timing = release_service(10.0, clock.now(), 80, clock)
        self.assertAlmostEqual(clock.now(), 10.08)
        self.assertAlmostEqual(timing["raw_service_seconds"], 0.025)
        self.assertAlmostEqual(timing["added_wait_seconds"], 0.055)
        self.assertFalse(timing["floor_miss"])
        self.assertAlmostEqual(timing["completed_s"], 10.08)

    def test_compute_overrun_is_not_hidden_or_delayed_again(self):
        from libero_wallclock import release_service

        clock = Clock()
        clock.t = 10.095
        timing = release_service(10.0, clock.now(), 80, clock)
        self.assertAlmostEqual(clock.now(), 10.095)
        self.assertEqual(timing["added_wait_seconds"], 0)
        self.assertTrue(timing["floor_miss"])
        self.assertAlmostEqual(timing["raw_target_overrun_seconds"], 0.015)
        self.assertAlmostEqual(timing["target_overrun_seconds"], 0.015)

    def test_invalid_service_floors_rejected(self):
        from libero_wallclock import release_service

        for floor in (-1, float("nan"), float("inf")):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                release_service(10, 10, floor, Clock())

    def test_cli_controls_are_deployment_only(self):
        from evaluate_libero_protocol import build_parser, validate_deployment_controls

        parser = build_parser()
        common = ["--protocol", "deployment", "--variant", "pir2", "--output", "unused"]
        defaults = parser.parse_args(common)
        self.assertEqual(defaults.action_min_service_ms, 0)
        self.assertEqual(defaults.vision_min_service_ms, 0)
        self.assertEqual(defaults.vision_max_hz, 0)
        self.assertIsNone(defaults.deployment_fixed_delay)
        for controls in (
            ["--action-min-service-ms", "80"],
            ["--vision-min-service-ms", "160"],
            ["--vision-max-hz", "15"],
            ["--deployment-fixed-delay", "2"],
        ):
            args = parser.parse_args(common + controls)
            validate_deployment_controls(args)
            args.protocol = "algorithm"
            with self.assertRaisesRegex(ValueError, "deployment"):
                validate_deployment_controls(args)

    def test_calibration_uses_submission_floor_and_keeps_server_time_raw(self):
        from libero_wallclock import calibrate

        clock = Clock()
        endpoints = []

        class Client:
            def __init__(self, **kwargs):
                pass

            def call_endpoint(self, endpoint, payload):
                endpoints.append(endpoint)
                clock.t += 0.01
                if endpoint == "bootstrap":
                    return {"actions": np.zeros((40, 7))}
                if endpoint == "plan":
                    return {
                        "actions": np.zeros((1, 7)),
                        "audit": {
                            "cache": {"capture_s": 9.0, "source_tick": 0},
                            "valid_start_tick": payload["request_tick"] + 1,
                            "valid_end_tick": payload["request_tick"] + 2,
                            "server_seconds": 0.007,
                        },
                    }
                return {}

        visual = {"vl_embeds": [], "meta": {"capture_s": 9.0, "source_tick": 0}}
        with patch.dict(
            "sys.modules",
            {"gr00t.policy.server_client": SimpleNamespace(PolicyClient=Client)},
        ):
            delay, result = calibrate(
                Client(),
                {},
                9.0,
                1,
                0.05,
                visual,
                1,
                action_min_service_ms=85,
                clock=clock,
            )
        self.assertEqual(delay, 2)
        self.assertEqual(endpoints.count("plan"), 12)
        self.assertEqual(endpoints.count("bootstrap"), 1)
        np.testing.assert_allclose(result["action_rpc_seconds"], np.full(10, 0.085))
        self.assertTrue(
            all(
                abs(r["raw_service_seconds"] - 0.02) < 1e-8
                for r in result["raw_samples"]
            )
        )
        self.assertEqual(result["action_server_seconds"], [0.007] * 10)

    def test_deployment_bootstrap_and_worker_controls_use_fixed_delay(self):
        from libero_wallclock import deployment_episode

        args = SimpleNamespace(
            control_hz=20,
            action_min_service_ms=80,
            vision_min_service_ms=160,
            vision_max_hz=15,
            deployment_fixed_delay=2,
            slow_warmup={},
            calibration=(4, {"selected_delay_ticks": 4}),
            port=1,
            slow_port=2,
            max_steps=10,
            record_video=False,
        )
        boot_calls = []

        def boot(endpoint, payload):
            boot_calls.append((endpoint, payload))
            return {
                "actions": np.zeros((40, 7)),
                "bootstrap_seconds": 0.1,
                "forward_counts": {"dit": 8, "vlm": 1},
            }

        env = SimpleNamespace(
            _process_observation=lambda raw: raw,
            _env=SimpleNamespace(
                env=SimpleNamespace(_get_observations=lambda **kwargs: {})
            ),
        )
        with (
            patch("libero_wallclock.ActionWorker") as actor,
            patch("libero_wallclock.VisionWorker") as vision,
            patch("libero_wallclock.feature_rpc_client"),
            patch("libero_wallclock.prepare_episode_vision", return_value={}),
            patch("libero_wallclock.run_control_loop", return_value=({}, [])) as loop,
        ):
            result, _ = deployment_episode(
                env, {}, SimpleNamespace(call_endpoint=boot), args, 1, io.StringIO()
            )
        self.assertEqual(len(boot_calls), 1)
        self.assertEqual(boot_calls[0][0], "bootstrap")
        self.assertEqual(boot_calls[0][1]["delay_ticks"], 2)
        self.assertEqual(loop.call_args.args[6], 2)
        self.assertEqual(loop.call_args.kwargs["fixed_delay"], 2)
        self.assertEqual(actor.call_args.kwargs["min_service_ms"], 80)
        self.assertEqual(vision.call_args.kwargs["min_service_ms"], 160)
        self.assertEqual(vision.call_args.kwargs["max_hz"], 15)
        self.assertEqual(result["service_controls"]["vision_max_hz"], 15)
        self.assertEqual(result["service_controls"]["calibrated_delay_ticks"], 4)
        self.assertEqual(result["service_controls"]["effective_initial_delay_ticks"], 2)

    def test_action_future_and_feature_dependency_hidden_until_release(self):
        from libero_queue_trace import QueueTrace
        from libero_wallclock import ActionWorker, Observation

        clock = GatedClock()
        calls = []

        class Client:
            def __init__(self, **kwargs):
                pass

            def call_endpoint(self, endpoint, payload):
                calls.append(endpoint)
                clock.t += 0.01
                if endpoint == "plan":
                    return {
                        "actions": np.zeros((1, 7)),
                        "audit": {
                            "cache": {"capture_s": 9.0, "source_tick": 0},
                            "valid_start_tick": 1,
                            "valid_end_tick": 2,
                            "server_seconds": 0.007,
                        },
                    }
                return {}

        events = QueueTrace(clock=clock.now)
        visual = {
            "sequence": 8,
            "vl_embeds": [],
            "meta": {"capture_s": 9.0, "source_tick": 0},
        }
        with patch.dict(
            "sys.modules",
            {"gr00t.policy.server_client": SimpleNamespace(PolicyClient=Client)},
        ):
            actor = ActionWorker(1, events=events, min_service_ms=80, clock=clock)
            future = actor.submit(
                Observation({}, 0, 9.0), 0, 1, np.zeros((1, 7)), visual, 10.05, 10.0
            )
            try:
                self.assertTrue(clock.holding.wait(2))
                self.assertFalse(future.done())
                self.assertEqual(actor.completed, [])
                self.assertNotIn(
                    "action_completed", [r["kind"] for r in events.records()]
                )
            finally:
                clock.release.set()
                actor.close()
            result = future.result()
        self.assertEqual(calls, ["install", "plan"])
        self.assertEqual(result["client_timing"]["feature_sequence"], 8)
        self.assertAlmostEqual(result["client_timing"]["completed_s"], 10.08)
        self.assertAlmostEqual(result["client_timing"]["raw_service_seconds"], 0.02)
        self.assertEqual(result["audit"]["server_seconds"], 0.007)

    def test_vision_cache_is_not_published_until_release(self):
        from libero_queue_trace import QueueTrace
        from libero_wallclock import Observation, VisionWorker

        clock = GatedClock()
        calls = []

        class Client:
            def __init__(self, **kwargs):
                self.socket = SimpleNamespace(close=lambda: None)
                self.context = SimpleNamespace(term=lambda: None)

            def call_endpoint(self, endpoint, payload):
                calls.append(endpoint)
                clock.t += 0.02
                return {
                    "vl_embeds": [],
                    "meta": {
                        "capture_s": payload["capture_s"],
                        "source_tick": payload["source_tick"],
                        "vlm_forward_calls": 1,
                        "vlm_seconds": 0.015,
                    },
                }

        events = QueueTrace(clock=clock.now)
        observation = Observation({}, 0, 9.0)
        with patch.dict(
            "sys.modules",
            {"gr00t.policy.server_client": SimpleNamespace(PolicyClient=Client)},
        ):
            vision = VisionWorker(1, events=events, min_service_ms=80, clock=clock)
            try:
                vision.offer(observation)
                self.assertTrue(clock.holding.wait(2))
                self.assertIsNone(vision.ready(observation))
                self.assertEqual(vision.snapshot()["inflight_tick"], 0)
                self.assertNotIn(
                    "feature_published", [r["kind"] for r in events.records()]
                )
                vision.request_stop(10.04)
            finally:
                clock.release.set()
                stats = vision.close()
        self.assertEqual(calls, ["vision"])
        self.assertEqual(vision.ready(observation)["sequence"], 1)
        self.assertAlmostEqual(stats["rpc_times"][0]["completed_s"], 10.08)
        self.assertAlmostEqual(stats["rpc_times"][0]["raw_service_seconds"], 0.02)
        self.assertEqual(stats["completed_after_control_end"], 1)
        self.assertEqual(stats["rpc_times"][0]["server_vlm_seconds"], 0.015)


class VisionAdmission(unittest.TestCase):
    def run_worker(
        self, on_call, on_wait=None, *, max_hz=10, min_service_ms=0, on_event=None
    ):
        """Run the real worker loop with scripted transport and condition wakes."""
        from libero_queue_trace import QueueTrace
        from libero_wallclock import Observation, VisionWorker

        self.assertIn("max_hz", inspect.signature(VisionWorker).parameters)
        clock = Clock()

        class Trace(QueueTrace):
            def emit(self, kind, **fields):
                seq = super().emit(kind, **fields)
                if on_event is not None:
                    on_event(worker, clock, kind)
                return seq

        events = Trace(clock=clock.now)
        calls, waits = [], []

        class Client:
            def __init__(self, **kwargs):
                self.socket = SimpleNamespace(close=lambda: None)
                self.context = SimpleNamespace(term=lambda: None)

            def call_endpoint(self, endpoint, payload):
                self_test.assertEqual(endpoint, "vision")
                calls.append((clock.now(), payload))
                on_call(worker, clock, payload)
                return {
                    "vl_embeds": [],
                    "meta": {
                        "capture_s": payload["capture_s"],
                        "source_tick": payload["source_tick"],
                        "vlm_forward_calls": 1,
                        "vlm_seconds": 0.005,
                    },
                }

        def wait(timeout=None):
            waits.append(timeout)
            self.assertIsNotNone(on_wait, "Unexpected condition wait")
            self.assertIsNotNone(timeout)
            self.assertGreater(timeout, 0)
            on_wait(worker, clock, timeout)

        self_test = self
        with (
            patch.dict(
                "sys.modules",
                {"gr00t.policy.server_client": SimpleNamespace(PolicyClient=Client)},
            ),
            patch("threading.Thread.start"),
            patch("threading.Thread.join"),
        ):
            worker = VisionWorker(
                1,
                events=events,
                clock=clock,
                max_hz=max_hz,
                min_service_ms=min_service_ms,
            )
            worker.offer(Observation({"state.x": [0]}, 0, 9.95))
            with patch.object(worker.condition, "wait", side_effect=wait):
                worker._run()
            stats = worker.close()
        return worker, stats, events.records(), calls, waits

    def test_first_admission_is_immediate_and_publication_precedes_next_admission(self):
        from libero_wallclock import Observation

        def on_call(worker, clock, payload):
            clock.t += 0.01
            if payload["source_tick"] == 0:
                worker.offer(Observation({"state.x": [1]}, 1, clock.now()))
            else:
                worker.request_stop()

        worker, stats, rows, calls, waits = self.run_worker(
            on_call, lambda worker, clock, timeout: clock.sleep_until(clock.t + timeout)
        )
        np.testing.assert_allclose([t for t, _ in calls], [10, 10.1])
        self.assertEqual(stats["vision_max_hz"], 10)
        self.assertEqual(stats["admitted"], 2)
        self.assertEqual(stats["offered_sensor_frames"], 2)
        self.assertEqual(stats["submitted"], stats["completed"])
        np.testing.assert_allclose(stats["admission_starts_s"], [10, 10.1])
        self.assertAlmostEqual(stats["actual_admission_hz"], 10)
        np.testing.assert_allclose(stats["admission_wait_seconds"], [0, 0.09])
        self.assertEqual(len(waits), 1)
        published = [r for r in rows if r["kind"] == "feature_published"]
        np.testing.assert_allclose([r["t_s"] for r in published], [10.01, 10.11])
        self.assertLess(published[0]["t_s"], calls[1][0])
        for timing, (started, payload) in zip(stats["rpc_times"], calls, strict=True):
            self.assertEqual(timing["started_s"], started)
            self.assertLessEqual(timing["admitted_s"], started)
            self.assertEqual(timing["capture_s"], payload["capture_s"])
            self.assertEqual(timing["source_tick"], payload["source_tick"])
            self.assertEqual(timing["vision_max_hz"], 10)
            self.assertEqual(timing["added_wait_seconds"], 0)
        self.assertEqual(worker.snapshot()["feature_source_tick"], 1)

    def test_wait_keeps_latest_frame_pending_and_preserves_selected_source(self):
        from libero_wallclock import Observation

        def on_call(worker, clock, payload):
            if payload["source_tick"] == 0:
                clock.t = 10.01
                worker.offer(Observation({"state.x": [1]}, 1, clock.now()))
            else:
                self.assertEqual(payload["source_tick"], 3)
                self.assertEqual(payload["capture_s"], 10.08)
                self.assertEqual(payload["observation"]["state"]["x"].item(), 3)
                worker.request_stop()

        wakes = iter([(10.04, 2), (10.08, 3), (10.1, None)])

        def on_wait(worker, clock, timeout):
            self.assertIsNone(worker.snapshot()["inflight_tick"])
            self.assertEqual(worker.snapshot()["feature_source_tick"], 0)
            clock.t, tick = next(wakes)
            if tick is not None:
                worker.offer(Observation({"state.x": [tick]}, tick, clock.now()))

        worker, stats, rows, calls, waits = self.run_worker(on_call, on_wait)
        self.assertEqual([p["source_tick"] for _, p in calls], [0, 3])
        self.assertEqual(stats["skipped_sensor_frames"], 2)
        self.assertEqual(stats["offered_sensor_frames"], 4)
        self.assertEqual(stats["admitted"] + stats["skipped_sensor_frames"], 4)
        self.assertEqual(len(waits), 3)
        waiting = [r for r in rows if r["kind"] == "vision_admission_wait"]
        self.assertEqual(
            [r["camera_queue"]["waiting_tick"] for r in waiting], [1, 2, 3]
        )
        self.assertTrue(all(r["vision_max_hz"] == 10 for r in waiting))
        self.assertIsNone(worker.ready(Observation({}, 2, 10.06)))
        self.assertEqual(
            worker.ready(Observation({}, 3, 10.08))["meta"]["source_tick"], 3
        )

    def test_late_wake_anchors_next_slot_to_actual_admission_without_catchup(self):
        from libero_wallclock import Observation

        def on_call(worker, clock, payload):
            tick = payload["source_tick"]
            clock.t += 0.01
            if tick < 2:
                worker.offer(Observation({}, tick + 1, clock.now()))
            else:
                worker.request_stop()

        wakes = iter([10.37, 10.47])

        def on_wait(worker, clock, timeout):
            clock.t = next(wakes)

        _, stats, _, calls, waits = self.run_worker(on_call, on_wait)
        np.testing.assert_allclose([t for t, _ in calls], [10, 10.37, 10.47])
        np.testing.assert_allclose(stats["admission_starts_s"], [10, 10.37, 10.47])
        self.assertEqual(len(waits), 2)

    def test_slow_rpc_overrun_does_not_add_a_period_after_completion(self):
        from libero_wallclock import Observation

        def on_call(worker, clock, payload):
            tick = payload["source_tick"]
            clock.t += 0.25
            if tick == 0:
                worker.offer(Observation({}, 1, clock.now()))
            else:
                worker.request_stop()

        _, stats, _, calls, waits = self.run_worker(on_call)
        np.testing.assert_allclose([t for t, _ in calls], [10, 10.25])
        self.assertEqual(waits, [])
        self.assertEqual(stats["admission_wait_seconds"], [0, 0])

    def test_disabled_limit_preserves_immediate_next_request(self):
        from libero_wallclock import Observation

        def on_call(worker, clock, payload):
            clock.t += 0.01
            if payload["source_tick"] == 0:
                worker.offer(Observation({}, 1, clock.now()))
            else:
                worker.request_stop()

        _, stats, _, calls, waits = self.run_worker(on_call, max_hz=0)
        np.testing.assert_allclose([t for t, _ in calls], [10, 10.01])
        self.assertEqual(stats["admission_wait_seconds"], [0, 0])
        self.assertEqual(waits, [])

    def test_service_floor_remains_independent_of_admission_wait(self):
        from libero_wallclock import Observation

        def on_call(worker, clock, payload):
            clock.t += 0.01
            if payload["source_tick"] == 0:
                worker.offer(Observation({}, 1, clock.now()))
            else:
                worker.request_stop()

        _, stats, rows, calls, _ = self.run_worker(
            on_call,
            lambda worker, clock, timeout: clock.sleep_until(clock.t + timeout),
            min_service_ms=40,
        )
        np.testing.assert_allclose([t for t, _ in calls], [10, 10.1])
        np.testing.assert_allclose(stats["admission_wait_seconds"], [0, 0.06])
        published = [r["t_s"] for r in rows if r["kind"] == "feature_published"]
        np.testing.assert_allclose(published, [10.04, 10.14])
        np.testing.assert_allclose(
            [r["added_wait_seconds"] for r in stats["rpc_times"]], [0.03, 0.03]
        )

    def test_stop_waiting_drops_latest_and_preserves_backdated_control_cutoff(self):
        from libero_wallclock import Observation

        def on_call(worker, clock, payload):
            clock.t += 0.01
            worker.offer(Observation({}, 1, clock.now()))

        def on_wait(worker, clock, timeout):
            clock.t = 10.03
            worker.request_stop(9.99)
            worker.offer(Observation({}, 2, clock.now()))

        _, stats, _, calls, waits = self.run_worker(on_call, on_wait)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(waits), 1)
        self.assertEqual(stats["dropped_at_stop"], 2)
        self.assertEqual(stats["offered_sensor_frames"], 3)
        self.assertEqual(stats["stop_requested_s"], 10.03)
        self.assertEqual(stats["admitted_after_stop_request"], 0)
        self.assertEqual(stats["started_after_control_end"], 1)
        self.assertEqual(stats["admission_starts_s"], [10])

    def test_stop_inflight_publishes_owned_result_without_new_admission(self):
        from libero_wallclock import Observation

        def on_call(worker, clock, payload):
            self.assertEqual(worker.snapshot()["inflight_tick"], 0)
            clock.t += 0.01
            worker.offer(Observation({}, 1, clock.now()))
            worker.request_stop()
            clock.t += 0.2

        worker, stats, rows, calls, waits = self.run_worker(on_call)
        self.assertEqual(len(calls), 1)
        self.assertEqual(waits, [])
        self.assertEqual(stats["completed_after_control_end"], 1)
        self.assertEqual(stats["admitted_after_stop_request"], 0)
        self.assertEqual(stats["dropped_at_stop"], 1)
        self.assertEqual(worker.snapshot()["feature_source_tick"], 0)
        self.assertEqual(len([r for r in rows if r["kind"] == "feature_published"]), 1)

    def test_stop_between_admission_and_rpc_keeps_both_actual_timestamps(self):
        def on_event(worker, clock, kind):
            if kind == "vision_admitted":
                clock.t = 10.002
                worker.request_stop()
                clock.t = 10.004

        def on_call(worker, clock, payload):
            clock.t += 0.01

        _, stats, rows, calls, _ = self.run_worker(on_call, on_event=on_event)
        self.assertEqual(stats["admission_starts_s"], [10])
        self.assertEqual(stats["rpc_times"][0]["started_s"], 10.004)
        self.assertEqual(calls[0][0], 10.004)
        self.assertEqual(stats["stop_requested_s"], 10.002)
        self.assertEqual(stats["admitted_after_stop_request"], 0)
        self.assertEqual(stats["started_after_control_end"], 1)
        admitted = [r for r in rows if r["kind"] == "vision_admitted"]
        self.assertEqual(admitted[0]["t_s"], 10)

    def test_future_cutoff_does_not_postpone_observing_stop(self):
        def on_call(worker, clock, payload):
            clock.t += 0.01
            worker.request_stop(20.0)

        _, stats, _, calls, waits = self.run_worker(on_call)
        self.assertEqual(len(calls), 1)
        self.assertEqual(waits, [])
        self.assertEqual(stats["control_end_cutoff_s"], 20.0)
        self.assertEqual(stats["stop_requested_s"], 10.01)
        self.assertEqual(stats["admitted_after_stop_request"], 0)

    def test_real_condition_wakes_for_newest_frame_and_stop(self):
        from queue import Queue

        from libero_queue_trace import QueueTrace
        from libero_wallclock import Observation, VisionWorker

        self.assertIn("max_hz", inspect.signature(VisionWorker).parameters)
        waiting = Queue()
        clock = Clock()

        class Trace(QueueTrace):
            def emit(self, kind, **fields):
                seq = super().emit(kind, **fields)
                if kind == "vision_admission_wait":
                    waiting.put(fields["camera_queue"]["waiting_tick"])
                return seq

        class Client:
            def __init__(self, **kwargs):
                self.socket = SimpleNamespace(close=lambda: None)
                self.context = SimpleNamespace(term=lambda: None)

            def call_endpoint(self, endpoint, payload):
                worker.offer(Observation({}, 1, clock.now()))
                return {
                    "meta": {
                        **{k: payload[k] for k in ("capture_s", "source_tick")},
                        "vlm_forward_calls": 1,
                        "vlm_seconds": 0.005,
                    }
                }

        with patch.dict(
            "sys.modules",
            {"gr00t.policy.server_client": SimpleNamespace(PolicyClient=Client)},
        ):
            worker = VisionWorker(
                1, clock=clock, events=Trace(clock=clock.now), max_hz=0.01
            )
            try:
                worker.offer(Observation({}, 0, clock.now()))
                self.assertEqual(waiting.get(timeout=2), 1)
                worker.offer(Observation({}, 2, clock.now()))
                self.assertEqual(waiting.get(timeout=2), 2)
                worker.request_stop()
                worker.thread.join(timeout=2)
                self.assertFalse(worker.thread.is_alive())
            finally:
                stats = worker.close()
        self.assertEqual(stats["admitted"], 1)
        self.assertEqual(stats["skipped_sensor_frames"], 1)
        self.assertEqual(stats["dropped_at_stop"], 1)

    def test_invalid_worker_rates_fail_before_starting_thread(self):
        from libero_wallclock import VisionWorker

        self.assertIn("max_hz", inspect.signature(VisionWorker).parameters)
        with patch("threading.Thread.start") as start:
            for rate in (-1, float("nan"), float("inf"), -float("inf")):
                with (
                    self.subTest(rate=rate),
                    self.assertRaisesRegex(ValueError, "finite.*nonnegative"),
                ):
                    VisionWorker(1, max_hz=rate)
            start.assert_not_called()

    def test_invalid_cli_rates_and_algorithm_use_are_rejected(self):
        from evaluate_libero_protocol import build_parser, validate_deployment_controls

        parser = build_parser()
        common = ["--protocol", "deployment", "--variant", "flow", "--output", "unused"]
        self.assertIn("--vision-max-hz", parser.format_help())
        for rate in ("-1", "nan", "inf", "-inf"):
            with (
                self.subTest(rate=rate),
                self.assertRaisesRegex(ValueError, "finite.*nonnegative"),
            ):
                validate_deployment_controls(
                    parser.parse_args(common + [f"--vision-max-hz={rate}"])
                )
        defaults = parser.parse_args(common)
        defaults.protocol = "algorithm"
        validate_deployment_controls(defaults)


class QueueAccounting(unittest.TestCase):
    def test_server_restart_resumes_and_records_each_runtime_identity(self):
        from evaluate_libero_protocol import save_run_configuration

        identity = {
            "checkpoint": "checkpoint",
            "sha256": {"model.safetensors": "abc"},
            "variant": "pir2",
            "role": "action",
            "logical_device": "cuda:0",
            "cuda_visible_devices": "2,3",
            "pid": 100,
            "cuda_stream": 101,
        }
        config = {
            "protocol": "deployment",
            "checkpoint": identity,
            "slow_checkpoint": {**identity, "role": "vlm", "logical_device": "cuda:1"},
        }
        restarted = {
            **config,
            "checkpoint": {**identity, "pid": 200, "cuda_stream": 201},
            "slow_checkpoint": {
                **config["slow_checkpoint"],
                "pid": 200,
                "cuda_stream": 202,
            },
        }
        with TemporaryDirectory() as directory:
            output = Path(directory)
            first = save_run_configuration(output, config)
            saved = (output / "config.json").read_text()
            second = save_run_configuration(output, restarted)
            runtime = [
                json.loads(row)
                for row in (output / "runtime-identities.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual((output / "config.json").read_text(), saved)
        self.assertEqual(first, second)
        self.assertNotIn("pid", first["checkpoint"])
        self.assertNotIn("cuda_stream", first["slow_checkpoint"])
        self.assertEqual(
            [row["identities"]["checkpoint"]["pid"] for row in runtime], [100, 200]
        )
        self.assertEqual(
            runtime[1]["identities"]["slow_checkpoint"]["cuda_stream"], 202
        )
        self.assertEqual(identity["pid"], 100)

    def test_resume_rejects_checkpoint_or_layout_changes_without_recording_invocation(
        self,
    ):
        from evaluate_libero_protocol import save_run_configuration

        identity = {
            "sha256": {"model.safetensors": "abc"},
            "variant": "pir2",
            "role": "action",
            "logical_device": "cuda:0",
            "cuda_visible_devices": "2,3",
            "pid": 100,
            "cuda_stream": 101,
        }
        config = {
            "checkpoint": identity,
            "slow_checkpoint": {**identity, "role": "vlm"},
        }
        changes = {
            "sha256": {"model.safetensors": "different"},
            "variant": "flow",
            "role": "wrong",
            "logical_device": "cuda:1",
            "cuda_visible_devices": "3,2",
        }
        with TemporaryDirectory() as directory:
            output = Path(directory)
            save_run_configuration(output, config)
            for endpoint in ("checkpoint", "slow_checkpoint"):
                for key, value in changes.items():
                    with self.subTest(endpoint=endpoint, key=key):
                        changed = {**config, endpoint: {**config[endpoint], key: value}}
                        with self.assertRaisesRegex(RuntimeError, "changed on resume"):
                            save_run_configuration(output, changed)
            self.assertEqual(
                len((output / "runtime-identities.jsonl").read_text().splitlines()), 1
            )

    def test_mailbox_replaces_only_waiting_frame_and_preserves_inflight_owner(self):
        from libero_queue_trace import LatestFrameMailbox, QueueTrace

        events = QueueTrace(clock=lambda: 8.0)
        queue = LatestFrameMailbox(events)
        first = SimpleNamespace(tick=0, capture_s=7.0)
        second = SimpleNamespace(tick=1, capture_s=7.05)
        third = SimpleNamespace(tick=2, capture_s=7.1)
        self.assertIsNone(queue.offer(first))
        self.assertIs(queue.take(), first)
        queue.offer(second)
        self.assertIs(queue.offer(third), second)
        self.assertEqual(queue.snapshot()["inflight_tick"], 0)
        self.assertEqual(queue.snapshot()["waiting_tick"], 2)
        queue.publish({"sequence": 1, "meta": {"source_tick": 0, "capture_s": 7.0}})
        self.assertIsNone(queue.snapshot()["inflight_tick"])
        self.assertEqual(queue.snapshot()["feature_source_tick"], 0)
        self.assertEqual(queue.snapshot()["waiting_tick"], 2)
        dropped = queue.drop_pending()
        self.assertIs(dropped, third)
        records = events.records()
        replaced = [row for row in records if row["kind"] == "camera_replaced"]
        self.assertEqual(len(replaced), 1)
        self.assertEqual(replaced[0]["source_tick"], 1)
        self.assertEqual(replaced[0]["replacement_source_tick"], 2)

    def test_mailbox_rejects_feature_from_another_capture(self):
        from libero_queue_trace import LatestFrameMailbox, QueueTrace

        queue = LatestFrameMailbox(QueueTrace())
        queue.offer(SimpleNamespace(tick=3, capture_s=2.0))
        queue.take()
        with self.assertRaises(ValueError):
            queue.publish({"sequence": 1, "meta": {"source_tick": 4, "capture_s": 2.0}})
        self.assertEqual(queue.snapshot()["inflight_tick"], 3)
        self.assertIsNone(queue.latest)

    def test_feature_publication_time_is_when_it_becomes_available(self):
        from libero_queue_trace import LatestFrameMailbox, QueueTrace

        clock = Clock()
        events = QueueTrace(clock=clock.now)
        queue = LatestFrameMailbox(events)
        queue.offer(SimpleNamespace(tick=0, capture_s=clock.now()))
        queue.take()
        rpc_completed = clock.now() + 0.04
        clock.t += 0.06
        queue.publish(
            {"sequence": 1, "meta": {"source_tick": 0, "capture_s": 10.0}},
            rpc_completed_s=rpc_completed,
        )
        published = events.records()[-1]
        self.assertEqual(published["t_s"], clock.now())
        self.assertEqual(published["rpc_completed_s"], rpc_completed)

    def test_trace_serializes_concurrent_events_and_detaches_input_metadata(self):
        from libero_queue_trace import QueueTrace

        events = QueueTrace(clock=lambda: 4.0)
        details = {"slots": [1]}
        events.emit("initial", details=details)
        details["slots"].append(2)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda tick: events.emit("sample", tick=tick), range(40)))
        out = io.StringIO()
        events.flush(out)
        records = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual([row["seq"] for row in records], list(range(41)))
        self.assertEqual(records[0]["details"], {"slots": [1]})
        self.assertEqual({row["tick"] for row in records[1:]}, set(range(40)))

    def test_selected_tasks_control_summary_denominator(self):
        from evaluate_libero_protocol import save_summary

        records = [
            {"task_id": task, "episode_id": 0, "task": f"task{task}", "success": True}
            for task in (0, 3)
        ]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            save_summary(
                output,
                records,
                {"task_ids": [0, 3], "episodes_per_task": 1, "protocol": "algorithm"},
            )
            result = json.loads((output / "summary.json").read_text())
        self.assertEqual(result["expected_episodes"], 2)
        self.assertTrue(result["complete"])
        self.assertEqual(result["evaluation_scope"], "selected_task_subset")

    def test_subset_cli_keeps_full_suite_as_default(self):
        from evaluate_libero_protocol import build_parser

        common = ["--protocol", "deployment", "--variant", "pir2", "--output", "unused"]
        parser = build_parser()
        defaults = parser.parse_args(common)
        self.assertEqual(defaults.task_ids, list(range(10)))
        self.assertFalse(defaults.no_video)
        self.assertEqual(defaults.slow_warmup_calls, 12)
        subset = parser.parse_args(common + ["--task-ids", "0", "3", "--no-video"])
        self.assertEqual(subset.task_ids, [0, 3])
        self.assertTrue(subset.no_video)

    def test_calibration_reports_raw_samples_and_excludes_first_two(self):
        from libero_wallclock import calibrate

        class CalibrationActor:
            def __init__(self, port, **kwargs):
                self.count = 0

            def submit(self, obs, tick, delay, prefix, visual, deadline, submitted):
                elapsed = 9.0 if self.count < 2 else 0.06
                self.count += 1
                result = {
                    "client_timing": {"completed_s": submitted + elapsed},
                    "audit": {
                        "valid_start_tick": tick + 1,
                        "server_seconds": elapsed / 2,
                    },
                    "actions": np.zeros((1, 7)),
                }
                return SimpleNamespace(result=lambda: result)

            def close(self):
                pass

        client = SimpleNamespace(
            call_endpoint=lambda *args: {"actions": np.zeros((40, 7))}
        )
        with patch("libero_wallclock.ActionWorker", CalibrationActor):
            delay, result = calibrate(client, {}, 3.0, 1, 0.05, {}, 1)
        self.assertEqual(delay, 2)
        self.assertEqual(result["sample_count"], 10)
        self.assertEqual(len(result["raw_samples"]), 12)
        self.assertTrue(all(row["excluded"] for row in result["raw_samples"][:2]))
        self.assertTrue(all(not row["excluded"] for row in result["raw_samples"][2:]))
        np.testing.assert_allclose(result["action_rpc_seconds"], np.full(10, 0.06))


class Clock:
    def __init__(self):
        self.t = 10.0
        self.events = []

    def now(self):
        return self.t

    def sleep_until(self, t):
        self.t = max(self.t, t)


class GatedClock(Clock):
    def __init__(self):
        super().__init__()
        self.holding = threading.Event()
        self.release = threading.Event()

    def sleep_until(self, t):
        self.holding.set()
        if not self.release.wait(2):
            raise TimeoutError("Test did not release the service gate")
        super().sleep_until(t)


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
    def run_case(self, latency, slow_first_step=False, steps=12, fixed_delay=None):
        clock = Clock()
        actor = Actor(clock, latency)
        vision = Vision(clock)
        calls = []
        writes = []

        class DeferredOutput(io.StringIO):
            def write(self, text):
                writes.append(clock.now())
                return super().write(text)

        trace = DeferredOutput()

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
            trace,
            clock=clock,
            fixed_delay=fixed_delay,
        )
        self.last_clock = clock
        self.last_trace = [json.loads(line) for line in trace.getvalue().splitlines()]
        self.last_writes = writes
        return result, calls, vision.offers

    def test_trace_io_is_deferred_until_control_ends(self):
        result, _, _ = self.run_case(0.02)
        self.assertTrue(self.last_writes)
        self.assertGreaterEqual(
            min(self.last_writes), 10 + result["controlled_wall_seconds"]
        )

    def test_control_snapshot_links_installed_and_executed_action_owners(self):
        self.run_case(0.02)
        ticks = [row for row in self.last_trace if row["kind"] == "control_tick"]
        self.assertEqual(ticks[1]["executed_slot"]["producer_request_tick"], 0)
        self.assertEqual(ticks[0]["executed_slot"]["producer_kind"], "bootstrap")
        self.assertEqual(ticks[0]["action_buffer_phase"], "after_execute")
        self.assertNotIn(
            0, [slot["tick"] for slot in ticks[0]["action_buffer"]["slots"]]
        )
        submitted = [
            row for row in self.last_trace if row["kind"] == "action_submitted"
        ]
        adopted = [row for row in self.last_trace if row["kind"] == "action_adopted"]
        self.assertEqual(submitted[0]["state_tick"], 0)
        self.assertEqual(submitted[0]["state_capture_s"], 10.0)
        self.assertEqual(submitted[0]["feature_capture_s"], 10.0)
        self.assertEqual(adopted[0]["request_tick"], 0)
        self.assertEqual(adopted[0]["adopted_at_tick"], 1)

    def test_control_keeps20hz_when_inference_is_slow(self):
        result, ticks, _ = self.run_case(0.18)
        np.testing.assert_allclose(ticks, 10.0 + np.arange(12) * 0.05, atol=1e-9)
        self.assertGreater(result["action_request_deadline_misses"], 0)
        self.assertGreater(result["expired_action_slots"], 0)
        self.assertAlmostEqual(result["actual_control_hz"], 20)

    def test_fixed_delay_keeps_misses_and_expired_counts_without_adapting(self):
        adaptive, _, _ = self.run_case(0.18)
        fixed, _, _ = self.run_case(0.18, fixed_delay=1)
        self.assertGreater(adaptive["final_action_delay_ticks"], 1)
        self.assertEqual(fixed["final_action_delay_ticks"], 1)
        self.assertGreater(fixed["action_request_deadline_misses"], 0)
        self.assertGreater(fixed["expired_action_slots"], 0)
        ticks = [r for r in self.last_trace if r["kind"] == "control_tick"]
        self.assertTrue(all(r["action_delay_budget_ticks"] == 1 for r in ticks))

    def test_fixed_delay_cannot_disagree_with_bootstrap(self):
        with self.assertRaisesRegex(ValueError, "Bootstrap"):
            self.run_case(0.02, fixed_delay=2)

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


class ReplayRecordingContract(unittest.TestCase):
    def test_recording_is_optional_owned_and_maps_post_step_frame_to_computed_time(
        self,
    ):
        from coexecution.replay.recording import ReplayRecorder

        clock = Clock()
        actor, vision = Actor(clock, 0.02), Vision(clock)
        image = np.zeros((4, 4, 3), dtype=np.uint8)

        def observation(x):
            return {
                "state.x": np.array([x]),
                "state.y": [0],
                "state.z": [0],
                "video.image": image,
                "video.wrist_image": image,
            }

        count = 0

        def advance(action):
            nonlocal count
            count += 1
            image[:] = count
            clock.t += 0.01
            return observation(count), False, False

        recorder = ReplayRecorder(max_frames=3)
        stream = io.StringIO()
        result, frames = run_control_loop(
            advance,
            observation(0),
            clock.now() - 0.1,
            np.zeros((40, 7)),
            actor,
            vision,
            1,
            0.05,
            3,
            stream,
            clock=clock,
            replay_recorder=recorder,
        )
        ticks = [
            json.loads(line)
            for line in stream.getvalue().splitlines()
            if json.loads(line)["kind"] == "control_tick"
        ]
        self.assertEqual(
            ticks[0]["robot_response"]["before"]["position_xyz_m"], [0, 0, 0]
        )
        self.assertEqual(
            ticks[0]["robot_response"]["after"]["position_xyz_m"], [1, 0, 0]
        )
        self.assertEqual(
            recorder.mapping[1]["wall_s"], ticks[0]["next_observation_computed_s"]
        )
        self.assertEqual(recorder.frames["image"][0].max(), 0)
        self.assertEqual(result["replay_recording"]["status"], "overflow")
        self.assertEqual(frames, [])

    def test_bounded_debug_queue_reports_dropped_events(self):
        from libero_queue_trace import QueueTrace

        trace = QueueTrace(max_records=2)
        trace.emit("first", t_s=1)
        trace.emit("second", t_s=2)
        trace.emit("third", t_s=3)
        self.assertEqual(len(trace.records()), 2)
        self.assertEqual(trace.status()["dropped_records"], 1)


class ReplayFailureContinuation(unittest.TestCase):
    def test_second_camera_allocation_failure_does_not_abort_controller(self):
        from coexecution.replay.recording import ReplayRecorder

        clock = Clock()
        actor, vision = Actor(clock, 0.02), Vision(clock)
        images = {
            "video.image": np.zeros((4, 4, 3), dtype=np.uint8),
            "video.wrist_image": np.ones((4, 4, 3), dtype=np.uint8),
        }
        calls = []

        def advance(action):
            calls.append(clock.now())
            clock.t += 0.01
            return images, False, False

        original_array = np.array
        allocations = []

        def allocate(value, *args, **kwargs):
            if (
                isinstance(value, np.ndarray)
                and value.ndim == 3
                and kwargs.get("copy") is True
            ):
                allocations.append(value)
                if len(allocations) == 4:
                    raise MemoryError("second camera in first control step")
            return original_array(value, *args, **kwargs)

        recorder = ReplayRecorder(max_frames=4)
        trace = io.StringIO()
        with patch.object(np, "array", allocate):
            result, _ = run_control_loop(
                advance,
                images,
                clock.now(),
                np.zeros((40, 7)),
                actor,
                vision,
                1,
                0.05,
                3,
                trace,
                clock=clock,
                replay_recorder=recorder,
            )
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["control_steps"], 3)
        self.assertEqual(result["replay_recording"]["status"], "failed")
        self.assertEqual({len(frames) for frames in recorder.frames.values()}, {1})
        self.assertEqual(len(allocations), 4)
        self.assertTrue(
            any(
                json.loads(row)["kind"] == "control_end"
                for row in trace.getvalue().splitlines()
            )
        )
