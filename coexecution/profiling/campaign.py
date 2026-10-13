"""Prepare, verify and resume source-frozen, single-GPU contention campaigns.

Preparing/checking a plan never starts a model or a CUDA workload. Each selected
GPU job is admitted only on a clean device, with hardware identity preserved
across the entire campaign and later resumes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .collection import METRICS, postprocess, stage_command, validate_workers
from .hardware import (
    foreign_pids,
    gpu_snapshot,
    is_clean,
    ncu_processes,
    owned_processes,
    process_identity,
    process_snapshot,
    terminate_owned_group,
)
from .provenance import (
    digest,
    freeze_checkpoint,
    now,
    read_json,
    run_text,
    source_paths,
    tracked_files,
    verify_checkpoint,
    verify_sources,
    write_json,
)

MODES = {
    "vlm_only",
    "dit_only",
    "serial",
    "concurrent_equal",
    "concurrent_dit_priority",
    "cooperative_dit_priority",
    "vlm_only_cooperative",
}


def default_cases():
    """Priority BF16/W8-all cases; root can provide an explicit JSON list."""
    cases = []
    for mode, offsets in [
        ("vlm_only", (0,)),
        ("dit_only", (0,)),
        ("serial", (0,)),
        ("concurrent_equal", (0, 2, 5, 10)),
        ("concurrent_dit_priority", (0, 2, 5, 10)),
        ("cooperative_dit_priority", (0, 2, 5, 10)),
        ("vlm_only_cooperative", (0,)),
    ]:
        for offset in offsets:
            for precision in ("bf16", "w8a8"):
                cases.append(
                    {
                        "id": f"{precision}-all-{mode}-offset{offset}",
                        "precision": precision,
                        "component": "all",
                        "mode": mode,
                        "dit_offset_ms": offset,
                    }
                )
    return cases


def validate_plan(plan):
    if plan.get("schema_version") != 2:
        raise ValueError("Unsupported plan schema")
    candidates = plan["gpu"]["candidates"]
    if (
        not candidates
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in candidates
        )
        or len(set(candidates)) != len(candidates)
    ):
        raise ValueError("GPU candidates must be distinct nonnegative physical indices")
    if plan["gpu"]["max_memory_mib"] != 200 or plan["gpu"]["max_utilization"] != 0:
        raise ValueError("The strict clean-GPU gate must not be relaxed")
    if plan["gpu"]["telemetry_interval_s"] != 1:
        raise ValueError("Telemetry interval must be one second")
    ids = set()
    for case in plan["cases"]:
        identifier = case["id"]
        if not identifier or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in identifier
        ):
            raise ValueError(f"Unsafe case id: {identifier!r}")
        if identifier in ids:
            raise ValueError(f"Duplicate case id: {identifier}")
        ids.add(identifier)
        if case["mode"] not in MODES or case["precision"] not in (
            "bf16",
            "w8a8",
            "w4a4",
        ):
            raise ValueError(f"Unsupported case {identifier}")
        offset = case["dit_offset_ms"]
        if case["component"] not in ("all", "vlm", "dit") or (
            isinstance(offset, bool)
            or not isinstance(offset, (int, float))
            or not math.isfinite(offset)
            or offset < 0
        ):
            raise ValueError(f"Invalid component/offset: {identifier}")
        if case.get("vlm_chunk_blocks", 1) not in (1, 2):
            raise ValueError(f"Invalid cooperative chunk size: {identifier}")
    if not ids or not plan["source_files"]:
        raise ValueError("Empty cases or source manifest")
    if not plan.get("checkpoint_identity", {}).get("files"):
        raise ValueError("Missing frozen checkpoint identity")
    if not plan["stages"]:
        raise ValueError("At least one collection stage is required")
    for stage, config in plan["stages"].items():
        if (
            type(config.get("iterations")) is not int
            or type(config.get("repeats")) is not int
            or config.get("capture_window") not in ("pair", "whole")
        ):
            raise ValueError("Invalid collection stage counts/capture window")
        if (
            stage not in ("timing", "nsys", "ncu")
            or config["iterations"] < 1
            or config["repeats"] < 1
        ):
            raise ValueError("Invalid collection stage")
        if stage == "ncu" and (
            config["iterations"] != 1 or config["capture_window"] != "pair"
        ):
            raise ValueError("NCU must collect exactly one complete pair per replay")
        if (
            stage == "nsys"
            and config["capture_window"] == "pair"
            and config["iterations"] != 1
        ):
            raise ValueError(
                "NSYS pair capture must use iterations=1; whole capture is a distinct protocol"
            )
    if plan.get("worker_module") != "coexecution.profiling.worker":
        raise ValueError("Worker module must be the frozen public contention worker")
    common = plan.get("worker_common", {})
    for key, frozen in (
        ("checkpoint", plan["checkpoint_identity"].get("path")),
        ("groot_root", plan.get("groot_root")),
    ):
        if not frozen or common.get(key) != frozen:
            raise ValueError(f"Worker {key} differs from its frozen provenance")
    allowed = {
        "checkpoint",
        "dataset",
        "groot_root",
        "frames",
        "warmup",
        "seed",
        "torch_threads",
        "vlm_chunk_blocks",
        "episode_position",
        "dataset_step",
    }
    if set(common) - allowed:
        raise ValueError(
            "Worker common options cannot override case/device/capture arguments"
        )
    for key, minimum in (("frames", 1), ("warmup", 0), ("torch_threads", 1)):
        if type(common.get(key)) is not int or common[key] < minimum:
            raise ValueError(f"Invalid worker {key}")
    if type(common.get("seed")) is not int:
        raise ValueError("Worker seed must be an integer")
    for key in ("episode_position", "dataset_step"):
        if key in common and (type(common[key]) is not int or common[key] < 0):
            raise ValueError(f"Invalid worker {key}")
    chunk = common.get("vlm_chunk_blocks", 1)
    if type(chunk) is not int or chunk not in (1, 2):
        raise ValueError("Invalid worker cooperative chunk size")
    if not common.get("dataset"):
        raise ValueError("Missing worker dataset")
    return plan


def load_plan(path, expected_sha):
    if not expected_sha or digest(path) != expected_sha:
        raise RuntimeError("Plan SHA256 does not match --plan-sha256")
    plan = validate_plan(read_json(path))
    verify_sources(plan)
    verify_checkpoint(plan["checkpoint_identity"], full=True)
    return plan


class Campaign:
    def __init__(self, plan_path, plan_sha, plan, output):
        self.plan_path, self.plan_sha, self.plan = plan_path, plan_sha, plan
        self.output = output
        self.state_path = output / "status.json"
        self.state = (
            read_json(self.state_path)
            if self.state_path.exists()
            else {"plan_sha256": plan_sha, "gpu": None, "jobs": {}}
        )
        if self.state["plan_sha256"] != plan_sha:
            raise RuntimeError("This output directory belongs to another frozen plan")
        self.active = None
        self.lock_acquired = False

    def status(self, state, **updates):
        self.state.update(
            {
                "state": state,
                "updated_at": now(),
                "supervisor_pid": os.getpid(),
                **updates,
            }
        )
        write_json(self.state_path, self.state)

    def verify(self):
        if digest(self.plan_path) != self.plan_sha:
            raise RuntimeError("Plan changed during campaign")
        verify_sources(self.plan)
        verify_checkpoint(self.plan["checkpoint_identity"])

    def wait_clean(self):
        pinned = self.state["gpu"]
        candidates = [pinned["index"]] if pinned else self.plan["gpu"]["candidates"]
        streak = dict.fromkeys(candidates, 0)
        while True:
            self.verify()
            samples = []
            for index in candidates:
                sample = gpu_snapshot(index)
                samples.append(sample)
                if pinned and sample["uuid"] != pinned["uuid"]:
                    raise RuntimeError("Pinned GPU identity changed")
                streak[index] = streak[index] + 1 if is_clean(sample) else 0
                if streak[index] >= self.plan["gpu"]["clean_samples"]:
                    if not pinned:
                        self.state["gpu"] = {"index": index, "uuid": sample["uuid"]}
                    self.status("gpu_admitted", last_gpu=sample)
                    return index
            self.status("waiting_clean_gpu", candidates=samples, clean_streak=streak)
            time.sleep(self.plan["gpu"]["wait_poll_s"])

    def collect(self, command, output, gpu, env, job_id, stage):
        # A final strict check closes the gap after a potentially expensive hash audit.
        self.verify()
        while True:
            profilers = ncu_processes() if stage == "ncu" else []
            if profilers:
                self.status(
                    "waiting_global_ncu", other_profilers=profilers, active_job=job_id
                )
                time.sleep(2)
                self.verify()
                continue
            sample = gpu_snapshot(gpu)
            if sample["uuid"] != self.state["gpu"]["uuid"]:
                raise RuntimeError("Pinned GPU identity changed before launch")
            if is_clean(sample):
                break
            self.status("waiting_clean_gpu", last_gpu=sample, admission_race=True)
            self.wait_clean()
            self.verify()
        metadata = {
            "job_id": job_id,
            "started_at": now(),
            "command": command,
            "environment_overrides": {
                **self.plan["environment"],
                "CUDA_VISIBLE_DEVICES": self.state["gpu"]["uuid"],
                **({"TMPDIR": env["TMPDIR"]} if stage == "ncu" else {}),
            },
            "inherited_runtime_environment": {
                key: env[key]
                for key in (
                    "PATH",
                    "LD_LIBRARY_PATH",
                    "PYTHONPATH",
                    "CUDA_DEVICE_MAX_CONNECTIONS",
                    "CUBLAS_WORKSPACE_CONFIG",
                )
                if key in env
            },
            "plan_sha256": self.plan_sha,
            "source_commit": self.plan["source_commit"],
            "checkpoint_identity": self.plan["checkpoint_identity"],
            "source_manifest_sha256": hashlib.sha256(
                json.dumps(self.plan["source_files"], sort_keys=True).encode()
            ).hexdigest(),
            "gpu": self.state["gpu"],
            "admission": sample,
            "note": "Secret-bearing environment variables are intentionally not recorded.",
        }
        with (output / "process.log").open("w") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=self.plan["worktree"],
                env=env,
                start_new_session=True,
            )
            leader = process_identity(process.pid)
            if leader is None:
                process.wait()
                raise RuntimeError(
                    "Launcher exited before process ownership could be recorded"
                )
            known = {process.pid: leader}
            self.active = (process, leader, known)
            metadata.update({"pid": process.pid, "process_identity": leader})
            write_json(output / "command.json", metadata)
            self.status(
                "running",
                active_job=job_id,
                active_pid=process.pid,
                active_output=str(output),
            )
            check_at = time.monotonic()
            exited_at = None
            try:
                with (output / "telemetry.jsonl").open("a") as telemetry:
                    while True:
                        tick_start = time.monotonic()
                        sample = gpu_snapshot(gpu)
                        all_processes = process_snapshot()
                        owned = owned_processes(leader, known, all_processes)
                        other_profilers = (
                            [
                                item
                                for item in ncu_processes(all_processes)
                                if item["pid"] not in owned
                            ]
                            if stage == "ncu"
                            else []
                        )
                        # NVML can briefly retain a just-exited worker PID.
                        allowed = {
                            **owned,
                            **{
                                pid: item
                                for pid, item in known.items()
                                if process_identity(pid) is None
                            },
                        }
                        foreign = foreign_pids(sample, allowed)
                        sample.update(
                            {
                                "owned_pids": sorted(owned),
                                "foreign_pids": foreign,
                                "foreign_ncu_profilers": other_profilers,
                            }
                        )
                        telemetry.write(json.dumps(sample) + "\n")
                        telemetry.flush()
                        self.status(
                            "running",
                            active_job=job_id,
                            active_pid=process.pid,
                            active_output=str(output),
                            last_gpu=sample,
                        )
                        if sample["uuid"] != self.state["gpu"]["uuid"]:
                            raise RuntimeError("GPU UUID changed during collection")
                        if foreign:
                            raise RuntimeError(
                                f"Foreign GPU compute processes appeared: {foreign}; this attempt is invalid"
                            )
                        if other_profilers:
                            raise RuntimeError(
                                f"Another NCU profiler appeared: {other_profilers}; stop only this job and preserve invalid evidence"
                            )
                        if time.monotonic() >= check_at:
                            self.verify()
                            check_at = time.monotonic() + 20
                        if process.poll() is not None:
                            still_running = [
                                item["pid"]
                                for item in sample["compute_processes"]
                                if item["pid"] in owned
                                and owned[item["pid"]]["state"] != "Z"
                            ]
                            if not still_running:
                                break
                            exited_at = exited_at or time.monotonic()
                            if time.monotonic() - exited_at > 30:
                                raise RuntimeError(
                                    "Launcher exited while its GPU children remained active"
                                )
                        time.sleep(
                            max(
                                0,
                                self.plan["gpu"]["telemetry_interval_s"]
                                - (time.monotonic() - tick_start),
                            )
                        )
                if process.returncode:
                    raise RuntimeError(
                        f"Collection exited with code {process.returncode}; inspect {output / 'process.log'}"
                    )
                self.verify()
            except BaseException:
                terminate_owned_group(process, leader, known)
                raise
            finally:
                metadata.update(
                    {
                        "finished_at": now(),
                        "returncode": process.poll(),
                        "known_owned_processes": list(known.values()),
                    }
                )
                write_json(output / "command.json", metadata)
                self.active = None

    def validate_inputs(self, records):
        """Reject changed observations or bootstrap state across cases and replays."""
        for record in records:
            fingerprint = {
                Path(name).name: value
                for name, value in record["files"].items()
                if Path(name).name in {"fixtures.json", "bootstrap.json"}
            }
            if set(fingerprint) != {"fixtures.json", "bootstrap.json"}:
                raise RuntimeError(
                    "Common inputs are missing fixture/bootstrap evidence"
                )
            expected = self.state.setdefault("input_fingerprint", fingerprint)
            if fingerprint != expected:
                raise RuntimeError(
                    "Common inputs changed across cases, replay passes or resume"
                )

    def job_valid(self, result):
        if (
            not result
            or result.get("state") != "complete"
            or result.get("plan_sha256") != self.plan_sha
            or result.get("measurements_valid") is not True
            or not result.get("output_sha256")
        ):
            return False
        for name, expected in result["output_sha256"].items():
            if not Path(name).is_file() or digest(name) != expected:
                raise RuntimeError(f"Completed output changed: {name}")
        return True

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / ".run.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_acquired = True
            self.status("checking_sources", active_pid=None)
            self.verify()
            for case in self.plan["cases"]:
                for stage, config in self.plan["stages"].items():
                    for repeat in range(config["repeats"]):
                        job_id = f"{case['id']}--{stage}--r{repeat + 1}"
                        if self.job_valid(self.state["jobs"].get(job_id)):
                            continue
                        gpu = self.wait_clean()
                        stamp = dt.datetime.now(dt.timezone.utc).strftime(
                            "%Y%m%dT%H%M%S"
                        )
                        output = (
                            self.output
                            / "attempts"
                            / job_id
                            / f"{stamp}-{time.time_ns()}"
                        )
                        output.mkdir(parents=True, exist_ok=False)
                        env = {
                            **os.environ,
                            **self.plan["environment"],
                            "CUDA_VISIBLE_DEVICES": self.state["gpu"]["uuid"],
                        }
                        if stage == "ncu":
                            private_tmp = output / "ncu-tmp"
                            private_tmp.mkdir(parents=True, exist_ok=True)
                            if private_tmp.stat().st_uid != os.getuid():
                                raise RuntimeError(
                                    "NCU TMPDIR must belong to this OS user"
                                )
                            env["TMPDIR"] = str(private_tmp)
                        command = stage_command(
                            self.plan,
                            case,
                            stage,
                            output,
                            gpu,
                            self.state["gpu"]["uuid"],
                        )
                        result = {
                            "state": "running",
                            "job_id": job_id,
                            "case": case,
                            "stage": stage,
                            "repeat": repeat + 1,
                            "output": str(output),
                            "plan_sha256": self.plan_sha,
                            "gpu": self.state["gpu"],
                            "started_at": now(),
                        }
                        self.state["jobs"][job_id] = result
                        write_json(output / "result.json", result)
                        try:
                            self.collect(command, output, gpu, env, job_id, stage)
                            records = validate_workers(
                                output, stage, case, config["iterations"]
                            )
                            self.validate_inputs(records)
                            hashes = postprocess(
                                self.plan,
                                self.state["gpu"],
                                stage,
                                output,
                                records,
                                case,
                            )
                            for record in records:
                                hashes.update(record["files"])
                            self.verify()
                            result.update(
                                {
                                    "state": "complete",
                                    "finished_at": now(),
                                    "worker_outputs": records,
                                    "output_sha256": hashes,
                                    "measurements_valid": True,
                                }
                            )
                        except BaseException as error:
                            result.update(
                                {
                                    "state": "invalid",
                                    "finished_at": now(),
                                    "error": repr(error),
                                    "measurements_valid": False,
                                }
                            )
                            write_json(output / "result.json", result)
                            self.state["jobs"][job_id] = result
                            self.status(
                                "stopped_error",
                                active_pid=None,
                                active_job=job_id,
                                error=repr(error),
                            )
                            raise
                        write_json(output / "result.json", result)
                        self.state["jobs"][job_id] = result
                        self.status("job_complete", active_pid=None, active_job=job_id)
            self.status(
                "complete",
                active_pid=None,
                active_job=None,
                completed_jobs=len(self.state["jobs"]),
            )


def executable(name):
    """Resolve a user-selected executable; never alter the system installation."""
    resolved = shutil.which(str(name))
    if resolved is None:
        raise FileNotFoundError(f"Executable not found: {name}")
    return str(Path(resolved).absolute())


def prepare(args):
    if args.plan.exists():
        raise FileExistsError(
            f"Keep frozen plans immutable; choose a new --plan: {args.plan}"
        )
    work = args.repo_root.resolve()
    groot_root = (args.groot_root or work / "upstream/learning/Isaac-GR00T").resolve()
    checkpoint = args.checkpoint.resolve()
    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    cases = read_json(args.cases_json) if args.cases_json else default_cases()
    selected_stages = tuple(dict.fromkeys(args.stages))
    tools = {
        name: executable(getattr(args, name) or name)
        for name in selected_stages
        if name in {"nsys", "ncu"}
    }
    extra_roots = []
    environment = {
        "OMP_NUM_THREADS": str(args.torch_threads),
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if args.offline:
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    if any(case["precision"] != "bf16" for case in cases):
        from coexecution.kernels.sources import require_submodule

        selected_cutlass = require_submodule("cutlass")
        extra_roots.append(Path(selected_cutlass["directory"]))
        environment["CUTLASS_ROOT"] = selected_cutlass["directory"]
    paths = source_paths(work, groot_root, extra_roots)
    stages = {
        "timing": {
            "iterations": args.iterations,
            "repeats": args.repeats,
            "capture_window": "pair",
        },
        "nsys": {
            "iterations": 1,
            "repeats": 1,
            "capture_window": "pair",
            "sampling_hz": args.nsys_sampling_hz,
            "metrics_set": args.nsys_metrics_set,
        },
        "ncu": {
            "iterations": 1,
            "repeats": 1,
            "capture_window": "pair",
            "metrics": args.ncu_metrics or list(METRICS),
            "replay_mode": "app-range",
            "global_profiler_guard": True,
        },
    }
    plan = {
        "schema_version": 2,
        "created_at": now(),
        "worktree": str(work),
        "groot_root": str(groot_root),
        "extra_source_roots": [str(path) for path in extra_roots],
        "source_commit": run_text(["git", "-C", str(work), "rev-parse", "HEAD"]),
        "tracked_paths": sorted(
            str(path) for path in tracked_files(work) if path.is_file()
        ),
        "source_files": {str(path): digest(path) for path in paths},
        "checkpoint_identity": freeze_checkpoint(checkpoint),
        "python": executable(args.python),
        "worker_module": "coexecution.profiling.worker",
        "tools": tools,
        "tool_versions": {
            name: run_text([path, "--version"]) for name, path in tools.items()
        },
        "gpu": {
            "candidates": args.gpus,
            "max_memory_mib": 200,
            "max_utilization": 0,
            "clean_samples": 2,
            "telemetry_interval_s": 1,
            "wait_poll_s": 2,
        },
        "worker_common": {
            "frames": args.frames,
            "warmup": args.warmup,
            "seed": args.seed,
            "torch_threads": args.torch_threads,
            "vlm_chunk_blocks": 1,
            "checkpoint": str(checkpoint),
            "dataset": str(dataset),
            "groot_root": str(groot_root),
        },
        "environment": environment,
        "cpu_affinity": args.cpu_set,
        "stages": {name: stages[name] for name in selected_stages},
        "analysis": {"enabled": args.analyze},
        "cases": cases,
        "ordering": "case-major; timing repeats and profiler runs are independent processes",
        "quality_note": "Precision cases measure hardware behavior, not LIBERO closed-loop task accuracy.",
    }
    validate_plan(plan)
    write_json(args.plan, plan)
    print(
        json.dumps(
            {
                "plan": str(args.plan),
                "plan_sha256": digest(args.plan),
                "source_files": len(paths),
                "cases": len(cases),
                "gpu_jobs": sum(stage["repeats"] for stage in plan["stages"].values())
                * len(cases),
                "gpu_jobs_started": 0,
            }
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser(
        "prepare", help="Freeze a new plan; starts no GPU programs"
    )
    create.add_argument("--plan", type=Path, required=True)
    create.add_argument("--checkpoint", type=Path, required=True)
    create.add_argument("--dataset", type=Path, required=True)
    create.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        required=True,
        help="Explicit physical GPU candidates, e.g. --gpus 2 3",
    )
    create.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    create.add_argument("--groot-root", type=Path)
    create.add_argument("--python", default=sys.executable)
    create.add_argument(
        "--cases-json",
        type=Path,
        help="List of mode/precision/component/dit_offset_ms case objects",
    )
    create.add_argument("--cpu-set", default=None)
    create.add_argument(
        "--stages",
        choices=("timing", "nsys", "ncu"),
        nargs="+",
        default=["timing", "nsys", "ncu"],
    )
    create.add_argument(
        "--nsys", help="Explicit Nsight Systems executable; otherwise search PATH"
    )
    create.add_argument(
        "--ncu", help="Explicit Nsight Compute executable; otherwise search PATH"
    )
    create.add_argument(
        "--nsys-metrics-set",
        help="Device-compatible metric set; omitted uses Nsight's default",
    )
    create.add_argument("--nsys-sampling-hz", type=int, default=10000)
    create.add_argument(
        "--ncu-metrics",
        nargs="+",
        help="Device-compatible requested counters; missing values invalidate capture",
    )
    create.add_argument("--warmup", type=int, default=10)
    create.add_argument("--iterations", type=int, default=30)
    create.add_argument("--repeats", type=int, default=3)
    create.add_argument("--frames", type=int, default=8)
    create.add_argument("--seed", type=int, default=1000)
    create.add_argument("--torch-threads", type=int, default=2)
    create.add_argument(
        "--offline", action="store_true", help="Use already cached model dependencies"
    )
    create.add_argument(
        "--analyze",
        action="store_true",
        help="Run metadata analysis for each collection",
    )
    for command in ("check", "run"):
        child = commands.add_parser(command)
        child.add_argument("--plan", type=Path, required=True)
        child.add_argument("--plan-sha256", required=True)
        if command == "run":
            child.add_argument(
                "--output",
                type=Path,
                required=True,
                help="Unique output directory; reuse only to resume this same plan",
            )
    args = parser.parse_args(argv)
    args.plan = args.plan.resolve()
    if args.command == "prepare":
        if (
            any(
                value < 1
                for value in (
                    args.iterations,
                    args.repeats,
                    args.frames,
                    args.torch_threads,
                    args.nsys_sampling_hz,
                )
            )
            or args.warmup < 0
        ):
            parser.error(
                "iterations/repeats/frames/threads/sampling must be positive and warmup nonnegative"
            )
        if any(gpu < 0 for gpu in args.gpus) or len(set(args.gpus)) != len(args.gpus):
            parser.error("--gpus requires distinct nonnegative physical indices")
        prepare(args)
        return
    plan = load_plan(args.plan, args.plan_sha256)
    if args.command == "check":
        print(
            json.dumps(
                {
                    "valid": True,
                    "plan_sha256": args.plan_sha256,
                    "source_files": len(plan["source_files"]),
                    "cases": len(plan["cases"]),
                    "gpu_jobs_started": 0,
                }
            )
        )
        return
    campaign = Campaign(args.plan, args.plan_sha256, plan, args.output.resolve())

    def interrupt(signum, _frame):
        raise InterruptedError(f"Supervisor received signal {signum}")

    previous = {
        sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        campaign.run()
    except BaseException as error:
        if campaign.active:
            terminate_owned_group(*campaign.active)
        if campaign.lock_acquired:
            campaign.status("stopped_error", active_pid=None, error=repr(error))
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
