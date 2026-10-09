"""Test actual controller scheduling with fake time/transport, not model accuracy."""

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
        self.assertIsNone(defaults.deployment_fixed_delay)
        for controls in (
            ["--action-min-service-ms", "80"],
            ["--vision-min-service-ms", "160"],
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
