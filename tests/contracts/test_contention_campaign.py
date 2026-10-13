"""CPU-only metadata, ownership and strict validation tests for the supervisor."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from coexecution.profiling import campaign as CAMPAIGN
from coexecution.profiling import collection as COLLECTION
from coexecution.profiling import hardware as HARDWARE
from coexecution.profiling import provenance as PROVENANCE


def valid_plan():
    return {
        "schema_version": 2,
        "gpu": {
            "candidates": [0],
            "max_memory_mib": 200,
            "max_utilization": 0,
            "telemetry_interval_s": 1,
            "clean_samples": 2,
            "wait_poll_s": 2,
        },
        "cases": CAMPAIGN.default_cases()[:1],
        "source_files": {"source": "sha"},
        "checkpoint_identity": {"path": "/checkpoint", "files": {"checkpoint": "sha"}},
        "groot_root": "/groot",
        "worker_module": "coexecution.profiling.worker",
        "worker_common": {
            "checkpoint": "/checkpoint",
            "groot_root": "/groot",
            "dataset": "/dataset",
            "frames": 8,
            "warmup": 10,
            "seed": 1000,
            "torch_threads": 2,
            "vlm_chunk_blocks": 1,
        },
        "stages": {"timing": {"iterations": 2, "repeats": 1, "capture_window": "pair"}},
    }


class CampaignTests(unittest.TestCase):
    def test_strict_clean_gate(self):
        sample = {"compute_processes": [], "utilization_gpu": 0, "memory_used_mib": 200}
        self.assertTrue(HARDWARE.is_clean(sample))
        for key, value in (
            ("compute_processes", [{"pid": 50}]),
            ("utilization_gpu", 1),
            ("memory_used_mib", 201),
        ):
            self.assertFalse(HARDWARE.is_clean({**sample, key: value}))

    def test_negative_gpu_rejected_before_nvml_call(self):
        with mock.patch.object(HARDWARE, "run_text") as query:
            with self.assertRaises(ValueError):
                HARDWARE.gpu_snapshot(-1)
            query.assert_not_called()

    def test_owned_tree_does_not_adopt_foreign_or_reused_pid(self):
        leader = {"pid": 10, "ppid": 1, "pgid": 10, "state": "S", "start_ticks": 100}
        child = {"pid": 11, "ppid": 10, "pgid": 10, "state": "S", "start_ticks": 101}
        foreign = {"pid": 12, "ppid": 1, "pgid": 12, "state": "S", "start_ticks": 101}
        known = {10: leader}
        owned = HARDWARE.owned_processes(
            leader, known, {10: leader, 11: child, 12: foreign}
        )
        self.assertEqual(set(owned), {10, 11})
        reused = {**leader, "start_ticks": 1000}
        self.assertEqual(
            HARDWARE.owned_processes(leader, known, {10: reused, 12: foreign}), {}
        )
        self.assertEqual(
            HARDWARE.foreign_pids(
                {"compute_processes": [{"pid": 11}, {"pid": 12}]}, owned
            ),
            [12],
        )

    def test_terminate_never_signals_reused_foreign_group(self):
        leader = {"pid": 10, "ppid": 1, "pgid": 10, "state": "S", "start_ticks": 100}
        reused = {**leader, "start_ticks": 200}
        with (
            mock.patch.object(HARDWARE, "process_snapshot", return_value={10: reused}),
            mock.patch.object(HARDWARE.os, "killpg") as kill,
        ):
            HARDWARE.terminate_owned_group(mock.Mock(), leader, {10: leader})
            kill.assert_not_called()

    def test_freeze_rejects_modified_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            path.write_text("a = 1\n")
            plan = {"source_files": {str(path): PROVENANCE.digest(path)}}
            path.write_text("a = 2\n")
            with self.assertRaisesRegex(RuntimeError, "Frozen source changed"):
                PROVENANCE.verify_sources(plan)

    def test_plan_hash_rejected_before_source_or_gpu_operations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text("{}")
            with mock.patch.object(CAMPAIGN, "verify_sources") as verify:
                with self.assertRaisesRegex(RuntimeError, "Plan SHA256"):
                    CAMPAIGN.load_plan(path, "incorrect")
                verify.assert_not_called()

    def test_ncu_replay_uses_unique_worker_root_and_retains_concurrency(self):
        plan = {
            "python": "/python",
            "worker_module": "coexecution.profiling.worker",
            "worker_common": {"frames": 8},
            "cpu_affinity": "40-47",
            "tools": {"ncu": "/ncu"},
            "stages": {
                "ncu": {
                    "iterations": 1,
                    "capture_window": "pair",
                    "metrics": list(COLLECTION.METRICS),
                }
            },
        }
        case = {
            "mode": "concurrent_equal",
            "precision": "w8a8",
            "component": "all",
            "dit_offset_ms": 2,
        }
        command = COLLECTION.stage_command(
            plan, case, "ncu", Path("/tmp/unique"), 3, "GPU-test"
        )
        self.assertEqual(command[command.index("--replay-mode") + 1], "app-range")
        self.assertNotIn("--profile-from-start", command)
        self.assertEqual(
            command[command.index("--replay-output-root") + 1],
            "/tmp/unique/worker-passes",
        )
        self.assertNotIn("--nvtx-include", command)
        self.assertEqual(command[command.index("--gpu") + 1], "3")
        case["vlm_chunk_blocks"] = 2
        command = COLLECTION.stage_command(
            plan, case, "ncu", Path("/tmp/unique"), 3, "GPU-test"
        )
        self.assertEqual(command.count("--vlm-chunk-blocks"), 1)
        self.assertEqual(command[command.index("--vlm-chunk-blocks") + 1], "2")

    def test_ncu_missing_or_unavailable_counter_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counters.csv"
            header = '"Metric Name","Metric Unit","Metric Value"\n'
            path.write_text(header + '"dram__bytes_read.sum","byte","1024"\n')
            result = COLLECTION.verify_counter_csv(path, ["dram__bytes_read.sum"])
            self.assertEqual(result["dram__bytes_read.sum"]["value"], 1024)
            with self.assertRaisesRegex(RuntimeError, "missing"):
                COLLECTION.verify_counter_csv(path, ["dram__bytes_write.sum"])
            path.write_text(header + '"dram__bytes_read.sum","byte","n/a"\n')
            with self.assertRaisesRegex(RuntimeError, "Unavailable"):
                COLLECTION.verify_counter_csv(path, ["dram__bytes_read.sum"])

    def test_equivalence_or_incomplete_worker_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "worker"
            root.mkdir()
            for name, value in (
                ("status.json", {"state": "complete"}),
                ("equivalence.json", {"passed": False}),
                ("measurement.json", {"equivalence": {"passed": False}}),
            ):
                (root / name).write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "valid equivalence"):
                COLLECTION.validate_workers(root.parent, "timing", {}, 30)

    def test_default_case_ids_are_unique_and_w4_is_not_silently_enabled(self):
        cases = CAMPAIGN.default_cases()
        self.assertEqual(len(cases), len({case["id"] for case in cases}))
        self.assertEqual({case["precision"] for case in cases}, {"bf16", "w8a8"})
        self.assertEqual({case["dit_offset_ms"] for case in cases}, {0, 2, 5, 10})

    def test_resume_requires_unchanged_completed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "measurement.json"
            evidence.write_text('{"valid": true}')
            campaign = CAMPAIGN.Campaign(root / "plan.json", "frozen", {}, root)
            record = {
                "state": "complete",
                "measurements_valid": True,
                "plan_sha256": "frozen",
                "output_sha256": {str(evidence): PROVENANCE.digest(evidence)},
            }
            self.assertTrue(campaign.job_valid(record))
            self.assertFalse(campaign.job_valid({**record, "state": "invalid"}))
            evidence.write_text('{"valid": false}')
            with self.assertRaisesRegex(RuntimeError, "Completed output changed"):
                campaign.job_valid(record)

    def test_pinned_campaign_waits_for_same_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {"gpu": {"candidates": [3, 2], "clean_samples": 2, "wait_poll_s": 0}}
            campaign = CAMPAIGN.Campaign(root / "plan.json", "frozen", plan, root)
            campaign.state["gpu"] = {"index": 3, "uuid": "gpu-three"}
            clean = {
                "index": 3,
                "uuid": "gpu-three",
                "compute_processes": [],
                "utilization_gpu": 0,
                "memory_used_mib": 100,
            }
            with (
                mock.patch.object(campaign, "verify"),
                mock.patch.object(CAMPAIGN, "gpu_snapshot", return_value=clean) as read,
            ):
                self.assertEqual(campaign.wait_clean(), 3)
                self.assertEqual(read.call_args_list, [mock.call(3), mock.call(3)])

    def test_checkpoint_hashes_and_stat_identity_reject_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "config.json",
                "processor_config.json",
                "statistics.json",
                "embodiment_id.json",
            ):
                (root / name).write_text("{}")
            shards = [f"model-{index}.safetensors" for index in range(3)]
            for name in shards:
                (root / name).write_bytes(b"tiny-metadata-fixture")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict(enumerate(shards))})
            )
            frozen = PROVENANCE.freeze_checkpoint(root)
            PROVENANCE.verify_checkpoint(frozen, full=True)
            changed = root / shards[0]
            changed.write_bytes(b"different-checkpoint")
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                PROVENANCE.verify_checkpoint(frozen)
            frozen["files"][str(changed)]["identity"] = PROVENANCE.file_identity(
                changed
            )
            with self.assertRaisesRegex(RuntimeError, "SHA256 changed"):
                PROVENANCE.verify_checkpoint(frozen, full=True)

    def test_global_ncu_name_matching_avoids_argument_false_positives(self):
        for name in ("/opt/nvidia/ncu", "nv-nsight-cu-cli", "ncu-ui"):
            self.assertTrue(HARDWARE.is_ncu_process_name(name))
        for name in ("python", "campaign.py", "nsys", "nvidia-smi"):
            self.assertFalse(HARDWARE.is_ncu_process_name(name))

    def test_nsys_pair_cannot_claim_multiple_measured_iterations(self):
        plan = valid_plan()
        plan["stages"] = {
            "nsys": {
                "iterations": 2,
                "repeats": 1,
                "capture_window": "pair",
                "sampling_hz": 1000,
            }
        }
        with self.assertRaisesRegex(ValueError, "NSYS pair capture"):
            CAMPAIGN.validate_plan(plan)
        plan["stages"]["nsys"]["capture_window"] = "whole"
        self.assertIs(CAMPAIGN.validate_plan(plan), plan)

    def test_wide_counter_csv_requires_all_requested_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wide.csv"
            path.write_text(
                '"ID","dram__bytes_read.sum","dram__bytes_write.sum"\n"","Gbyte","Mbyte"\n"0","5.2","300"\n'
            )
            values = COLLECTION.verify_counter_csv(
                path, ["dram__bytes_read.sum", "dram__bytes_write.sum"]
            )
            self.assertEqual(
                values["dram__bytes_read.sum"], {"value": 5.2, "unit": "Gbyte"}
            )
            with self.assertRaisesRegex(RuntimeError, "missing"):
                COLLECTION.verify_counter_csv(path, ["gpu__time_duration.sum"])

    def test_completed_but_invalid_or_empty_evidence_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = CAMPAIGN.Campaign(root / "plan.json", "frozen", {}, root)
            record = {"state": "complete", "plan_sha256": "frozen", "output_sha256": {}}
            self.assertFalse(campaign.job_valid(record))
            self.assertFalse(campaign.job_valid({**record, "measurements_valid": True}))

    def test_prelaunch_uuid_change_never_launches_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = CAMPAIGN.Campaign(root / "plan.json", "frozen", {}, root)
            campaign.state["gpu"] = {"index": 3, "uuid": "GPU-original"}
            sample = {
                "uuid": "GPU-other",
                "compute_processes": [],
                "utilization_gpu": 0,
                "memory_used_mib": 0,
            }
            with (
                mock.patch.object(campaign, "verify"),
                mock.patch.object(CAMPAIGN, "gpu_snapshot", return_value=sample),
                mock.patch.object(CAMPAIGN.subprocess, "Popen") as launch,
            ):
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    campaign.collect([], root, 3, {}, "job", "timing")
                launch.assert_not_called()

    def test_general_checkpoint_accepts_single_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "config.json",
                "processor_config.json",
                "statistics.json",
                "embodiment_id.json",
            ):
                (root / name).write_text("{}")
            (root / "model.safetensors").write_bytes(b"metadata only fixture")
            manifest = PROVENANCE.freeze_checkpoint(root)
            PROVENANCE.verify_checkpoint(manifest, full=True)
            self.assertIn(str(root / "model.safetensors"), manifest["files"])

    def test_prepare_requires_explicit_gpu_and_dataset(self):
        with mock.patch.object(CAMPAIGN, "prepare") as prepare:
            with self.assertRaises(SystemExit):
                CAMPAIGN.main(
                    ["prepare", "--plan", "plan.json", "--checkpoint", "checkpoint"]
                )
            prepare.assert_not_called()

    def test_nsys_metric_set_is_configurable_and_no_hardware_assumption(self):
        plan = {
            "python": "/python",
            "worker_module": "coexecution.profiling.worker",
            "worker_common": {},
            "cpu_affinity": None,
            "tools": {"nsys": "/nsys"},
            "stages": {
                "nsys": {"iterations": 1, "capture_window": "pair", "sampling_hz": 1000}
            },
        }
        case = {
            "mode": "concurrent_equal",
            "precision": "bf16",
            "component": "all",
            "dit_offset_ms": 0,
        }
        command = COLLECTION.stage_command(
            plan, case, "nsys", Path("/tmp/result"), 0, "GPU-test"
        )
        self.assertNotIn("--gpu-metrics-set=ad10x", command)
        self.assertIn("--gpu-uuid", command)
        plan["stages"]["nsys"]["metrics_set"] = "custom"
        command = COLLECTION.stage_command(
            plan, case, "nsys", Path("/tmp/result"), 0, "GPU-test"
        )
        self.assertIn("--gpu-metrics-set=custom", command)

    def test_common_inputs_reject_changed_dataset_or_bootstrap_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = CAMPAIGN.Campaign(root / "plan.json", "frozen", {}, root)
            records = [
                {
                    "files": {
                        "/one/fixtures.json": "same-input",
                        "/one/bootstrap.json": "same-bootstrap",
                    }
                }
            ]
            campaign.validate_inputs(records)
            campaign.validate_inputs(
                [
                    {
                        "files": {
                            "/two/fixtures.json": "same-input",
                            "/two/bootstrap.json": "same-bootstrap",
                        }
                    }
                ]
            )
            for name in ("fixtures.json", "bootstrap.json"):
                changed = {**records[0]["files"], f"/one/{name}": "changed"}
                with self.assertRaisesRegex(RuntimeError, "Common inputs"):
                    campaign.validate_inputs([{"files": changed}])

    def test_postprocess_binds_module_to_frozen_worktree_from_other_cwd(self):
        repo = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            plan = {
                "python": "/python",
                "worktree": str(repo),
                "analysis": {"enabled": True},
            }
            before = Path.cwd()
            try:
                os.chdir(directory)
                with mock.patch.object(COLLECTION.subprocess, "run") as run:
                    COLLECTION.postprocess(
                        plan,
                        {"index": 0},
                        "timing",
                        output,
                        [{"measurement": "/measurement.json"}],
                        {},
                    )
                    self.assertEqual(run.call_args.kwargs.get("cwd"), str(repo))
            finally:
                os.chdir(before)

    def test_plan_cannot_redirect_worker_away_from_frozen_model_and_source(self):
        plan = valid_plan()
        self.assertIs(CAMPAIGN.validate_plan(plan), plan)
        for field, value in (
            ("worker_module", "other.worker"),
            ("groot_root", "/other-groot"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                CAMPAIGN.validate_plan({**plan, field: value})
        for field in ("checkpoint", "groot_root"):
            changed = copy.deepcopy(plan)
            changed["worker_common"][field] = "/unfrozen"
            with self.subTest(field=field), self.assertRaises(ValueError):
                CAMPAIGN.validate_plan(changed)

    def test_malformed_worker_and_stage_parameters_rejected_before_gpu_admission(self):
        for field, value in (
            ("frames", 0),
            ("frames", True),
            ("warmup", -1),
            ("torch_threads", 0),
            ("seed", "1"),
            ("vlm_chunk_blocks", 3),
            ("gpu", 7),
        ):
            plan = valid_plan()
            plan["worker_common"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                CAMPAIGN.validate_plan(plan)
        for stages in (
            {},
            {"timing": {"iterations": 2, "repeats": 1, "capture_window": "wrong"}},
            {"timing": {"iterations": True, "repeats": 1, "capture_window": "pair"}},
        ):
            plan = valid_plan()
            plan["stages"] = stages
            with self.subTest(stages=stages), self.assertRaises(ValueError):
                CAMPAIGN.validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
