"""Scheduling/ownership tests only; no torch, CUDA, model, or dataset execution."""

import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from coexecution.optimization.execution.cache import SerialForwardBoundary as Boundary
from coexecution.profiling import common, worker
from coexecution.profiling.runtime import (
    ComponentBoundaries,
    clear_groups,
    executed_projections,
    partition_groups,
)
from coexecution.profiling.scheduling import CooperativeGate


def parse_args(args):
    return worker.parser().parse_args(
        args + ["--checkpoint", "/tmp/checkpoint", "--dataset", "/tmp/dataset"]
    )


class FakeModule:
    def __init__(self):
        self.parameter = object()
        self.forward = lambda value: value
        self.reset_streaming_buffer = lambda: None

    def parameters(self):
        return [self.parameter]


class Tests(unittest.TestCase):
    def test_import_does_not_load_gpu_framework(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; from coexecution.profiling import worker; "
                    "assert not any(n.startswith(('torch', 'triton')) for n in sys.modules)"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_contract(self):
        args = parse_args(
            [
                "--mode",
                "cooperative_dit_priority",
                "--gpu",
                "3",
                "--output",
                "/tmp/test",
                "--slice-blocks",
                "2",
                "--dit-offset-ms",
                "5",
            ]
        )
        self.assertEqual(args.vlm_chunk_blocks, 2)
        self.assertEqual(args.dit_offset_ms, 5)
        self.assertEqual(args.capture, "none")
        self.assertEqual(args.frames, 8)

    def test_checkpoint_and_dataset_are_required(self):
        with self.assertRaises(SystemExit):
            worker.parser().parse_args(
                ["--mode", "serial", "--gpu", "5", "--output", "/tmp/test"]
            )

    def test_explicit_gpu_indices_are_portable(self):
        for index in (0, 5, 12):
            args = parse_args(
                ["--mode", "serial", "--gpu", str(index), "--output", "/tmp/test"]
            )
            self.assertEqual(args.gpu, index)

    def test_mask_mismatch_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "never-created"
            args = parse_args(
                ["--mode", "serial", "--gpu", "5", "--output", str(output)]
            )
            with (
                patch.dict(worker.os.environ, {"CUDA_VISIBLE_DEVICES": "5,6"}),
                self.assertRaisesRegex(RuntimeError, "exactly match"),
            ):
                worker.run(args)
            self.assertFalse(output.exists())

    def test_uuid_mask_is_bound_to_physical_index(self):
        args = parse_args(
            [
                "--mode",
                "serial",
                "--gpu",
                "5",
                "--gpu-uuid",
                "GPU-abc",
                "--output",
                "/tmp/test",
            ]
        )
        with (
            patch.dict(worker.os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-abc"}),
            patch.object(
                worker.subprocess,
                "run",
                return_value=SimpleNamespace(stdout="GPU-def\n"),
            ),
            self.assertRaisesRegex(RuntimeError, "UUID no longer match"),
        ):
            worker.validate_arguments(args)

    def test_nonfinite_offset_rejected_before_loading(self):
        for value in ("nan", "inf", "-1"):
            args = parse_args(
                [
                    "--mode",
                    "serial",
                    "--gpu",
                    "5",
                    "--output",
                    "/tmp/test",
                    "--dit-offset-ms",
                    value,
                ]
            )
            with (
                patch.dict(worker.os.environ, {"CUDA_VISIBLE_DEVICES": "5"}),
                self.assertRaisesRegex(ValueError, "offset"),
            ):
                worker.validate_arguments(args)

    def test_replay_outputs_are_process_and_pass_specific(self):
        first = common.choose_output_path("nominal", "/tmp/replays", pid=7, time_ns=100)
        second = common.choose_output_path(
            "nominal", "/tmp/replays", pid=7, time_ns=101
        )
        third = common.choose_output_path("nominal", "/tmp/replays", pid=8, time_ns=100)
        self.assertEqual(first, Path("/tmp/replays/pass-7-100"))
        self.assertEqual(len({first, second, third}), 3)
        self.assertEqual(common.choose_output_path("nominal", None), Path("nominal"))

    def test_replay_cli_and_whole_capture(self):
        args = parse_args(
            [
                "--mode",
                "serial",
                "--gpu",
                "3",
                "--output",
                "/tmp/nominal",
                "--replay-output-root",
                "/tmp/replays",
                "--capture-window",
                "whole",
            ]
        )
        self.assertEqual(args.replay_output_root, Path("/tmp/replays"))
        self.assertEqual(args.capture_window, "whole")

    def test_default_output_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), replay_output_root=None)
            with self.assertRaises(FileExistsError):
                common.create_output(args)
            self.assertFalse(getattr(args, "output_created", False))

    def test_replay_output_creation_preserves_nominal(self):
        with tempfile.TemporaryDirectory() as directory:
            nominal = Path(directory) / "nominal"
            root = Path(directory) / "passes"
            args = SimpleNamespace(output=nominal, replay_output_root=root)
            stdout = StringIO()
            with redirect_stdout(stdout):
                common.create_output(args)
            self.assertTrue(args.output.is_dir())
            self.assertEqual(args.output.parent, root)
            self.assertFalse(nominal.exists())
            self.assertIn(str(args.output), stdout.getvalue())
            self.assertTrue(args.output_created)

    def test_capture_stop_runs_after_failure(self):
        calls = []
        runtime = SimpleNamespace(
            cudaProfilerStart=lambda: calls.append("start") or 0,
            cudaProfilerStop=lambda: calls.append("stop") or 0,
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "torch": SimpleNamespace(
                        cuda=SimpleNamespace(cudart=lambda: runtime)
                    )
                },
            ),
            self.assertRaisesRegex(ValueError, "intentional"),
            common.cuda_capture_range(True),
        ):
            calls.append("body")
            raise ValueError("intentional")
        self.assertEqual(calls, ["start", "body", "stop"])

    def test_disabled_capture_needs_no_cuda(self):
        with common.cuda_capture_range(False):
            pass

    def test_response_includes_wait_before_cuda_submission(self):
        # DiT is released at 5 ms, waits for a cooperative permit until 17 ms,
        # submits until 19 ms, and becomes host-observed ready at 22 ms.
        release_ns = 5_000_000
        submit_start_ns = 17_000_000
        ready_ns = 22_000_000
        self.assertEqual(common.response_ms(release_ns, ready_ns), 17.0)
        self.assertGreater(
            common.response_ms(release_ns, ready_ns), (ready_ns - submit_start_ns) / 1e6
        )

    def test_ready_timestamp_is_taken_after_own_event_sync(self):
        calls = []
        clock = {"ns": 0}

        class Event:
            def synchronize(self):
                calls.append("own_event_sync")
                clock["ns"] = 22_000_000

        with patch.object(
            common.time, "perf_counter_ns", side_effect=lambda: clock["ns"]
        ):
            ready_ns = common.observe_ready(Event())
        self.assertEqual(calls, ["own_event_sync"])
        self.assertEqual(ready_ns, 22_000_000)
        self.assertEqual(common.response_ms(5_000_000, ready_ns), 17.0)

    def test_response_rejects_clock_reversal(self):
        with self.assertRaisesRegex(ValueError, "precede release"):
            common.response_ms(20, 19)

    def test_negative_gpu_rejected_by_parser(self):
        with self.assertRaises(SystemExit):
            parse_args(["--mode", "serial", "--gpu", "-1", "--output", "/tmp/test"])

    def test_mixed_group_rejected(self):
        c = SimpleNamespace(grouped={8: [(["backbone.a", "action_head.b"], object())]})
        with self.assertRaisesRegex(ValueError, "Cross-component"):
            partition_groups([c])

    def test_shared_mutable_group_rejected(self):
        g = object()
        c = SimpleNamespace(grouped={8: [(["backbone.a"], g), (["backbone.b"], g)]})
        with self.assertRaisesRegex(ValueError, "shared"):
            partition_groups([c])

    def test_clear_is_component_local(self):
        a = SimpleNamespace(_input=object(), _output=object(), _remaining={1})
        b = SimpleNamespace(_input=object(), _output=object(), _remaining={2})
        c = SimpleNamespace(grouped={8: [(["backbone.a"], a), (["action_head.b"], b)]})
        groups, _ = partition_groups([c])
        clear_groups(groups["vlm"])
        self.assertIsNone(a._input)
        self.assertIsNotNone(b._input)
        self.assertEqual(b._remaining, {2})

    def test_executed_coverage_counts_selected_bitwidth_and_restores_dispatch(self):
        q8, q4, grouped4 = object(), object(), object()
        calls = []
        original = lambda *args: calls.append(args) or "value"
        controller = SimpleNamespace(
            quant={8: {"backbone.a": q8}, 4: {"backbone.a": q4}},
            grouped={
                8: [],
                4: [
                    (
                        ["action_head.b", "action_head.c"],
                        SimpleNamespace(linear=grouped4),
                    )
                ],
            },
            dispatch=original,
        )
        with executed_projections([controller]) as counts:
            self.assertEqual(controller.dispatch(q4, "input"), "value")
            controller.dispatch(grouped4, "input")
        self.assertEqual(
            dict(counts),
            {("backbone.a", 4): 1, ("action_head.b", 4): 1, ("action_head.c", 4): 1},
        )
        self.assertIs(controller.dispatch, original)
        self.assertEqual(len(calls), 2)

    def test_unknown_integer_cache_callback_rejected(self):
        optimized = SimpleNamespace(
            boundary=SimpleNamespace(clear=lambda: None),
            config=SimpleNamespace(precision="w8a8"),
        )
        with self.assertRaisesRegex(ValueError, "Unknown original integer"):
            ComponentBoundaries(optimized, Boundary)

    def test_bound_integer_callback_partitions_cache_and_restores_methods(self):
        backbone, dit, vl, head = (FakeModule() for _ in range(4))
        head.model, head.vl_self_attention = dit, vl
        original_forward = backbone.forward
        a = SimpleNamespace(_input=None, _output=None, _remaining=set())
        b = SimpleNamespace(_input=None, _output=None, _remaining=set())

        class Controller:
            def __init__(self):
                self.grouped = {8: [(["backbone.a"], a), (["action_head.b"], b)], 4: []}

            def clear_projection_cache(self):
                clear_groups([a, b])

        controller = Controller()
        original = Boundary(
            [backbone, dit, vl], controller.clear_projection_cache, reset_modules=[head]
        )
        optimized = SimpleNamespace(
            boundary=original,
            config=SimpleNamespace(precision="w8a8"),
            policy=SimpleNamespace(
                model=SimpleNamespace(backbone=backbone, action_head=head)
            ),
        )
        adapter = ComponentBoundaries(optimized, Boundary)
        a._input, b._input = object(), object()
        backbone.forward("input")
        self.assertIsNone(a._input)
        self.assertIsNotNone(b._input)
        dit.forward("input")
        adapter.assert_clear()
        self.assertIs(adapter.controllers[0], controller)
        adapter.close()
        self.assertIs(backbone.forward, original_forward)
        original.close()

    def test_runtime_rejects_parameter_alias_before_removing_original_guard(self):
        backbone, dit, vl, head = (FakeModule() for _ in range(4))
        head.model, head.vl_self_attention = dit, vl
        head.parameter = backbone.parameter
        original = Boundary([backbone, dit, vl], lambda: None)
        optimized = SimpleNamespace(
            boundary=original,
            config=SimpleNamespace(precision="bf16"),
            policy=SimpleNamespace(
                model=SimpleNamespace(backbone=backbone, action_head=head)
            ),
        )
        with self.assertRaisesRegex(ValueError, "share parameter"):
            ComponentBoundaries(optimized, Boundary)
        self.assertFalse(original.closed)
        original.close()

    def test_distinct_boundaries_allow_two_components(self):
        backbone, dit, vl = FakeModule(), FakeModule(), FakeModule()
        head = FakeModule()
        head.model, head.vl_self_attention = dit, vl
        rendezvous = threading.Barrier(2)
        backbone.forward = lambda: rendezvous.wait(1)
        dit.forward = lambda: rendezvous.wait(1)
        original = Boundary([backbone, dit, vl], lambda: None, reset_modules=[head])
        optimized = SimpleNamespace(
            boundary=original,
            config=SimpleNamespace(precision="bf16"),
            policy=SimpleNamespace(
                model=SimpleNamespace(backbone=backbone, action_head=head)
            ),
        )
        adapter = ComponentBoundaries(optimized, Boundary)
        failures = []

        def call(module):
            try:
                module.forward()
            except (RuntimeError, TimeoutError) as e:
                failures.append(e)

        threads = [threading.Thread(target=call, args=(m,)) for m in (backbone, dit)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
            self.assertFalse(thread.is_alive())
        self.assertFalse(failures)
        adapter.close()

    def test_same_component_reentrant_guard_preserved(self):
        root = FakeModule()
        started, release = threading.Event(), threading.Event()

        def forward():
            started.set()
            release.wait(1)

        root.forward = forward
        boundary = Boundary([root], lambda: None)
        thread = threading.Thread(target=root.forward)
        thread.start()
        self.assertTrue(started.wait(1))
        with self.assertRaisesRegex(RuntimeError, "serial submissions"):
            root.forward()
        release.set()
        thread.join(2)
        boundary.close()

    def test_late_dit_arrival_after_vlm_completes(self):
        gate = CooperativeGate(timeout_s=1)
        gate.start_vlm()
        gate.complete_vlm()
        release = gate.release_dit(2, True)
        self.assertGreaterEqual(release - gate.vlm_release_ns, 2_000_000)
        gate.complete_dit()

    def test_cooperative_yield_and_resume_order(self):
        gate = CooperativeGate(timeout_s=1)
        gate.start_vlm()
        failures = []

        def dit():
            try:
                gate.release_dit(0, True)
                gate.complete_dit()
            except (RuntimeError, TimeoutError) as e:
                failures.append(e)

        thread = threading.Thread(target=dit)
        thread.start()
        self.assertTrue(gate.requested.wait(1))
        gate.boundary("test-layer")
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(failures)
        order = [row["event"] for row in gate.events]
        self.assertLess(order.index("dit_release"), order.index("vlm_yield"))
        self.assertLess(order.index("vlm_yield"), order.index("dit_complete"))
        self.assertLess(order.index("dit_complete"), order.index("vlm_resume"))

    def test_abort_releases_waiters(self):
        gate = CooperativeGate(timeout_s=1)
        gate.abort()
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            gate.release_dit(0, True)

    def test_timeout_aborts(self):
        gate = CooperativeGate(timeout_s=0.01)
        with self.assertRaises(TimeoutError):
            gate.release_dit(0, True)
        self.assertTrue(gate.aborted.is_set())

    def test_release_marker_precedes_cooperative_wait(self):
        gate = CooperativeGate(timeout_s=1)
        gate.start_vlm()
        seen = []

        def release():
            gate.release_dit(0, True, lambda: seen.append("release"))

        thread = threading.Thread(target=release)
        thread.start()
        self.assertTrue(gate.requested.wait(1))
        self.assertEqual(seen, ["release"])
        gate.permit.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
