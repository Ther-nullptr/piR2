"""Run declared LIBERO conditions with owned workers and paired task fixtures.

Model workers require CUDA. Import, command planning and fixture checks perform
no model computation. The runner owns only children it starts; it never adopts
or stops servers from another experiment's PID files.
"""

import argparse
import fcntl
import json
import math
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from coexecution.libero.features import validate_feature_options
from coexecution.libero.scheduling import (
    validate_execution_clock,
    validate_vision_max_hz,
)

FIXTURE_FIELDS = (
    "initial_state_sha256",
    "fixture_model_sha256",
    "settled_sim_state_sha256",
    "initial_rgb_sha256",
)


def validate_plan(plan):
    from coexecution.optimization.config import (
        add_optimization_arguments,
        optimization_config,
    )

    required = (
        "runtime_root",
        "model_python",
        "evaluation_python",
        "upstream",
        "libero_config",
        "gpus",
        "ports",
        "conditions",
        "protocols",
        "episodes_per_task",
        "max_steps",
        "seed",
    )
    if any(key not in plan for key in required):
        raise ValueError("Experiment plan is missing required fields")
    if any(
        type(plan[key]) is not int or plan[key] < 1
        for key in ("episodes_per_task", "max_steps")
    ):
        raise ValueError("Episode and step counts must be positive integers")
    tasks = plan.get("task_ids", list(range(10)))
    if (
        not isinstance(tasks, list)
        or not tasks
        or any(type(task) is not int or task not in range(10) for task in tasks)
        or len(set(tasks)) != len(tasks)
    ):
        raise ValueError("Declare unique task IDs in 0..9")
    if type(plan.get("record_replay", False)) is not bool:
        raise ValueError("record_replay must be boolean")
    if plan.get("record_replay") and set(plan["protocols"]) != {"deployment"}:
        raise ValueError("Replay recording requires deployment only")
    if (
        type(plan.get("replay_max_frames", 1024)) is not int
        or plan.get("replay_max_frames", 1024) < 1
    ):
        raise ValueError("Replay frame bound must be a positive integer")
    if type(plan["seed"]) is not int:
        raise ValueError("Experiment seed must be an integer")
    if set(plan["gpus"]) != {"action", "vlm", "simulation"} or any(
        type(value) is not int or value < 0 for value in plan["gpus"].values()
    ):
        raise ValueError("Declare physical CUDA devices for all three roles")
    if "gpu_uuids" in plan and (
        not isinstance(plan["gpu_uuids"], dict)
        or set(plan["gpu_uuids"]) != set(plan["gpus"])
        or any(
            not isinstance(value, str)
            or not re.fullmatch(
                r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value
            )
            for value in plan["gpu_uuids"].values()
        )
    ):
        raise ValueError("Declare a full GPU UUID for each role")
    if (
        set(plan["ports"]) != {"action", "vlm"}
        or any(
            type(value) is not int or not 1 <= value <= 65535
            for value in plan["ports"].values()
        )
        or len(set(plan["ports"].values())) != 2
    ):
        raise ValueError("Declare two distinct valid worker ports")
    if not plan["protocols"] or set(plan["protocols"]) - {"algorithm", "deployment"}:
        raise ValueError("Protocols must be algorithm and/or deployment")
    for settings in (plan, *plan["conditions"]):
        if type(settings.get("runtime_receipt", False)) is not bool:
            raise ValueError("runtime_receipt must be boolean")
        transport = settings.get("feature_transport", "numpy_fp32")
        rpc = settings.get("feature_rpc", "separate")
        validate_feature_options(transport, rpc)
        if (
            transport != "numpy_fp32" or rpc != "separate"
        ) and "deployment" not in plan["protocols"]:
            raise ValueError("Feature controls require a deployment protocol")
        rate = settings.get("vision_max_hz", 0)
        validate_vision_max_hz(rate)
        if rate and "deployment" not in plan["protocols"]:
            raise ValueError("Vision admission limits require a deployment protocol")
    names = [case["name"] for case in plan["conditions"]]
    if (
        not names
        or len(names) != len(set(names))
        or any(
            not name or Path(name).name != name or name in (".", "..") for name in names
        )
    ):
        raise ValueError("Conditions need unique directory names")
    parser = argparse.ArgumentParser(add_help=False)
    add_optimization_arguments(parser)
    for case in plan["conditions"]:
        for protocol in plan["protocols"]:
            from coexecution.libero.synchronous import validate_synchronous_controls

            defaults = {
                "execution_coupling": "asynchronous",
                "synchronous_chunk_steps": 8,
                "synchronous_warmup_timeout_seconds": 900,
                "server_layout": "split",
                "feature_transport": "numpy_fp32",
                "feature_rpc": "separate",
                "action_min_service_ms": 0,
                "vision_min_service_ms": 0,
                "vision_max_hz": 0,
                "deployment_fixed_delay": None,
                "artificial_compute_wait": 0,
            }
            validate_synchronous_controls(
                SimpleNamespace(
                    **{
                        key: case.get(key, plan.get(key, value))
                        for key, value in defaults.items()
                    },
                    protocol=protocol,
                    variant=case["variant"],
                )
            )
            validate_execution_clock(
                case.get("execution_clock", plan.get("execution_clock", "paced")),
                protocol=protocol,
                variant=case["variant"],
                fixed_delay=case.get(
                    "deployment_fixed_delay", plan.get("deployment_fixed_delay")
                ),
            )
        if case["variant"] not in ("flow", "pir2") or not case["checkpoint"]:
            raise ValueError("Declare each condition checkpoint and variant")
        if not isinstance(case["flags"], list) or any(
            not isinstance(flag, str) for flag in case["flags"]
        ):
            raise ValueError("Optimization flags must be an argument list")
        try:
            config = optimization_config(parser.parse_args(case["flags"]))
        except SystemExit as error:
            raise ValueError("Invalid optimization flags") from error
        config.validate_variant(case["variant"])
        from coexecution.libero.service import validate_server_layout

        layout = case.get("server_layout", plan.get("server_layout", "split"))
        paired_device = f"cuda:{plan['gpus']['vlm']}" if layout == "paired" else None
        validate_server_layout(
            layout,
            f"cuda:{plan['gpus']['action']}",
            paired_device,
            case.get("feature_transport", plan.get("feature_transport", "numpy_fp32")),
            case["variant"],
            config,
        )
        if layout == "paired" and plan["gpus"]["action"] != plan["gpus"]["vlm"]:
            raise ValueError(
                "Experiment paired layout currently requires the same physical GPU"
            )


def build_commands(plan, case, protocol, output):
    root = Path(plan["runtime_root"])
    commands = {}
    deployment_controls = []
    feature_transport = case.get(
        "feature_transport", plan.get("feature_transport", "numpy_fp32")
    )
    feature_rpc = case.get("feature_rpc", plan.get("feature_rpc", "separate"))
    layout = case.get("server_layout", plan.get("server_layout", "split"))
    coupling = case.get(
        "execution_coupling", plan.get("execution_coupling", "asynchronous")
    )
    feature_controls = []
    if "feature_transport" in case or "feature_transport" in plan:
        feature_controls = ["--feature-transport", feature_transport]
    if protocol == "deployment" and (
        "vision_max_hz" in case or "vision_max_hz" in plan
    ):
        rate = case.get("vision_max_hz", plan.get("vision_max_hz", 0))
        deployment_controls = ["--vision-max-hz", str(rate)]
    if protocol == "deployment":
        deployment_controls.extend(
            [
                "--execution-clock",
                case.get("execution_clock", plan.get("execution_clock", "paced")),
                "--execution-coupling",
                coupling,
                "--synchronous-chunk-steps",
                str(
                    case.get(
                        "synchronous_chunk_steps",
                        plan.get("synchronous_chunk_steps", 8),
                    )
                ),
                "--synchronous-warmup-timeout-seconds",
                str(
                    case.get(
                        "synchronous_warmup_timeout_seconds",
                        plan.get("synchronous_warmup_timeout_seconds", 900),
                    )
                ),
            ]
        )
        fixed_delay = case.get(
            "deployment_fixed_delay", plan.get("deployment_fixed_delay")
        )
        if fixed_delay is not None:
            deployment_controls.extend(["--deployment-fixed-delay", str(fixed_delay)])
    for role in ("action",) if coupling == "synchronous" else ("action", "vlm"):
        commands[role] = [
            str(plan["model_python"]),
            "-m",
            "coexecution.libero.service",
            "--checkpoint",
            str(case["checkpoint"]),
            "--variant",
            case["variant"],
            "--role",
            role,
            "--port",
            str(plan["ports"][role]),
            "--device",
            "cuda:0",
            "--output",
            str(output / (role + "-server")),
            *case["flags"],
            *feature_controls,
            *(
                ["--runtime-receipt"]
                if case.get("runtime_receipt", plan.get("runtime_receipt", False))
                else []
            ),
        ]
    if layout == "paired":
        commands["paired"] = commands.pop("action")
        del commands["vlm"]
        commands["paired"][commands["paired"].index("--output") + 1] = str(
            output / "paired-server"
        )
        commands["paired"].extend(
            [
                "--server-layout",
                "paired",
                "--vlm-device",
                "cuda:0",
                "--vlm-port",
                str(plan["ports"]["vlm"]),
            ]
        )
    commands["evaluate"] = [
        str(plan["evaluation_python"]),
        str(root / "scripts/evaluate_libero_protocol.py"),
        "--protocol",
        protocol,
        "--variant",
        case["variant"],
        "--output",
        str(output / protocol),
        "--port",
        str(plan["ports"]["action"]),
        "--slow-port",
        str(plan["ports"]["vlm"]),
        "--episodes-per-task",
        str(plan["episodes_per_task"]),
        "--max-steps",
        str(plan["max_steps"]),
        "--seed",
        str(plan["seed"]),
        "--control-hz",
        "20",
        "--visual-delay",
        "3",
        "--action-delay",
        "1",
        "--no-video",
        *deployment_controls,
        *(["--server-layout", layout] if layout != "split" else []),
        *(feature_controls if protocol == "deployment" else []),
        *(
            ["--feature-rpc", feature_rpc]
            if protocol == "deployment"
            and ("feature_rpc" in case or "feature_rpc" in plan)
            else []
        ),
        *(
            [
                "--record-replay",
                "--replay-max-frames",
                str(plan.get("replay_max_frames", 1024)),
            ]
            if plan.get("record_replay", False)
            else []
        ),
        "--task-ids",
        *(str(i) for i in plan.get("task_ids", range(10))),
    ]
    return commands


def paired_fixtures(conditions, *, episodes_per_task, task_ids=range(10)):
    expected = {
        (task, episode) for task in task_ids for episode in range(episodes_per_task)
    }
    mapped, problems = {}, []
    for name, rows in conditions.items():
        values = {(row["task_id"], row["episode_id"]): row for row in rows}
        if set(values) != expected or len(rows) != len(expected):
            problems.append(
                {"condition": name, "reason": "incomplete or duplicate episodes"}
            )
        mapped[name] = values
    if not mapped:
        problems.append({"reason": "no conditions"})
    else:
        reference_name, reference = next(iter(mapped.items()))
        for name, values in mapped.items():
            for key in expected & reference.keys() & values.keys():
                for field in FIXTURE_FIELDS:
                    if reference[key].get(field) is None or reference[key].get(
                        field
                    ) != values[key].get(field):
                        problems.append(
                            {
                                "condition": name,
                                "reference": reference_name,
                                "episode": key,
                                "field": field,
                            }
                        )
    return {
        "passed": not problems,
        "expected_per_condition": len(expected),
        "problems": problems,
    }


def _write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def validate_launch_paths(plan):
    root = Path(plan["runtime_root"])
    paths = [
        root / "coexecution/libero/service.py",
        root / "scripts/evaluate_libero_protocol.py",
        Path(plan["model_python"]),
        Path(plan["evaluation_python"]),
        Path(plan["libero_config"]),
        Path(plan["upstream"]) / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
        Path(plan["upstream"]) / "gr00t/policy/decoupled_policy.py",
        *(Path(case["checkpoint"]) / "config.json" for case in plan["conditions"]),
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if (
        plan.get("simulation_preload")
        and not Path(plan["simulation_preload"]).is_file()
    ):
        missing.append(str(plan["simulation_preload"]))
    if missing:
        raise ValueError(f"Declared launch paths are missing: {missing}")
    # The canonical evaluator hashes this pinned path under the runtime root.
    if (root / "upstream/learning/Isaac-GR00T").resolve() != Path(
        plan["upstream"]
    ).resolve():
        raise ValueError("Runtime upstream path does not match the declared source")


def environment(plan, role):
    env = os.environ.copy()
    root = Path(plan["runtime_root"])
    gpu = plan["gpus"][role]
    env.update(
        CUDA_VISIBLE_DEVICES=plan.get("gpu_uuids", {}).get(role, str(gpu)),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="2",
        MAX_JOBS="2",
        TOKENIZERS_PARALLELISM="false",
        NO_ALBUMENTATIONS_UPDATE="1",
        GROOT_HF_LOCAL_FIRST="1",
        GROOT_PATCH_MISTRAL="1",
        PYTHONPATH=os.pathsep.join(
            (
                str(root),
                str(root / "third_party/robotics-the-speedup-paradox/src"),
                str(plan["upstream"]),
            )
        ),
    )
    env.pop("HF_HUB_OFFLINE", None)
    if role == "simulation":
        env.update(
            MUJOCO_GL="egl",
            PYOPENGL_PLATFORM="egl",
            MUJOCO_EGL_DEVICE_ID=str(gpu),
            LIBERO_CONFIG_PATH=str(plan["libero_config"]),
        )
        preload = plan.get("simulation_preload") or env.get("PIR2_LIBERO_LD_PRELOAD")
        if preload:
            env["LD_PRELOAD"] = str(preload)
    return env


class OwnedProcesses:
    def __init__(self, plan):
        self.plan, self.children = plan, {}
        self.logs = {}

    def start(self, name, command, output, role):
        output.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns()}"
        log_path = output / f"{stamp}_{name}.log"
        with log_path.open("x") as log:
            process = subprocess.Popen(
                command,
                cwd=self.plan["runtime_root"],
                env=environment(self.plan, role),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.children[name] = process
        self.logs[name] = log_path
        job = {
            "pid": process.pid,
            "command": command,
            "gpu": self.plan["gpus"][role],
            "log": str(log_path),
        }
        _write(output / f"{stamp}_job.json", job)
        _write(output / "job.json", job)
        return process

    def ready(self, name, output, timeout=900, *, endpoints=1):
        process, deadline = self.children[name], time.monotonic() + timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"Worker {name} exited; see {self.logs[name]}")
            log = self.logs[name]
            if (
                log.exists()
                and log.read_text(errors="replace").count("TIMED_PROTOCOL_SERVER_READY")
                >= endpoints
            ):
                return
            if time.monotonic() > deadline:
                raise TimeoutError(f"Worker {name} did not become ready")
            time.sleep(1)

    def close(self):
        errors = []
        unique = {process.pid: process for process in self.children.values()}
        for process in reversed(tuple(unique.values())):
            if process.poll() is None:
                try:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired) as error:
                    errors.append((process.pid, str(error)))
        if errors:
            raise RuntimeError(f"Unable to stop owned workers: {errors}")
        self.children.clear()


def _admit_devices(plan):
    verified = {}
    if "gpu_uuids" in plan:
        observed = {}
        for gpu in set(plan["gpus"].values()):
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu),
                    "--query-gpu=index,uuid",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            index, uuid = (value.strip() for value in result.strip().split(","))
            if int(index) != gpu:
                raise RuntimeError("GPU UUID/index mismatch during admission")
            observed[gpu] = uuid
        for role, gpu in plan["gpus"].items():
            if observed[gpu].lower() != plan["gpu_uuids"][role].lower():
                raise RuntimeError(
                    f"GPU UUID/index mismatch for {role}: physical {gpu} is {observed[gpu]}"
                )
            verified[role] = {"physical_index": gpu, "uuid": observed[gpu]}
    for gpu in set(plan["gpus"].values()):
        # A completed condition can leave a briefly nonzero resource sample.
        # Recheck without launching anything; visible compute jobs reject at once.
        for attempt in range(5):
            status = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu),
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            memory, utilization = (
                int(value.strip()) for value in status.strip().split(",")
            )
            processes = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu),
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
            if memory <= 200 and not utilization and not processes:
                break
            if processes or attempt == 4:
                raise RuntimeError(
                    f"Declared GPU {gpu} is not clean before this experiment: "
                    f"memory={memory} MiB, utilization={utilization}%, "
                    f"compute_pids={processes or 'none'}"
                )
            time.sleep(0.5)
    for port in plan["ports"].values():
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    return verified


def deployment_report(plan, output, *, report_output=None):
    from coexecution.feature_usage import analyze_report
    from coexecution.queue_report import condition_report

    conditions = []
    expected = {
        (task, episode)
        for task in plan.get("task_ids", range(10))
        for episode in range(plan["episodes_per_task"])
    }
    for case in plan["conditions"]:
        directory = output / case["name"] / "deployment"
        condition = condition_report(directory)
        if (
            condition is None
            or len(condition["episodes"]) != len(expected)
            or {
                (episode["metadata"]["task_id"], episode["metadata"]["episode_id"])
                for episode in condition["episodes"]
            }
            != expected
        ):
            raise RuntimeError(f"Incomplete deployment control traces: {directory}")
        for episode in condition["episodes"]:
            for tick in episode["ticks"]:
                if tick["fallback"]:
                    continue
                for field in ("executed_image_age_ms", "executed_state_age_ms"):
                    age = tick.get(field)
                    if (
                        not isinstance(age, (int, float))
                        or not math.isfinite(age)
                        or age < 0
                    ):
                        raise RuntimeError(
                            f"Unknown or invalid executed observation age: {directory}/{field}"
                        )
        conditions.append(condition)
    report = {
        "schema_version": 2,
        "scope": (
            "Full Spatial suite"
            if set(plan.get("task_ids", range(10))) == set(range(10))
            else "Spatial task subset"
        )
        + ", paired declared initial states; actual executed image/state ages",
        "task_ids": list(plan.get("task_ids", range(10))),
        "conditions": conditions,
    }
    usage = analyze_report(report, raw_root=output)
    report_output = output if report_output is None else Path(report_output)
    report_output.mkdir(parents=True, exist_ok=True)
    _write(report_output / "deployment-aoi.json", report)
    _write(report_output / "feature-usage.json", usage)
    if not all(
        condition["read_accounting_complete"]
        and condition["execution_accounting_complete"]
        for condition in usage["conditions"]
    ):
        raise RuntimeError("Deployment feature/execution lineage is incomplete")
    return report


def run(plan, output):
    validate_plan(plan)
    validate_launch_paths(plan)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = output / "plan.json"
        if existing.exists() and json.loads(existing.read_text()) != plan:
            raise RuntimeError("Experiment plan changed on resume")
        _write(existing, plan)
        results, records = {}, {protocol: {} for protocol in plan["protocols"]}
        for case in plan["conditions"]:
            condition_output = output / case["name"]
            owned = OwnedProcesses(plan)
            try:
                verified_gpus = _admit_devices(plan)
                condition_output.mkdir(parents=True, exist_ok=True)
                _write(
                    condition_output / "gpu-admission.json",
                    {"verified_role_mapping": verified_gpus},
                )
                commands = build_commands(
                    plan, case, plan["protocols"][0], condition_output
                )
                for role in (name for name in commands if name != "evaluate"):
                    worker_output = condition_output / (role + "-server")
                    owned.start(
                        role,
                        commands[role],
                        worker_output,
                        "action" if role == "paired" else role,
                    )
                    owned.ready(
                        role, worker_output, endpoints=2 if role == "paired" else 1
                    )
                for protocol in plan["protocols"]:
                    commands = build_commands(plan, case, protocol, condition_output)
                    evaluation_output = condition_output / protocol
                    process = owned.start(
                        "evaluate",
                        commands["evaluate"],
                        evaluation_output,
                        "simulation",
                    )
                    _write(
                        output / "status.json",
                        {
                            "condition": case["name"],
                            "protocol": protocol,
                            "pid": process.pid,
                            "phase": "running",
                        },
                    )
                    print(case["name"], protocol, "started", process.pid, flush=True)
                    if process.wait():
                        raise RuntimeError(f"Evaluation failed: {evaluation_output}")
                    _write(
                        evaluation_output / "hardware.json",
                        {
                            "layout": case["name"],
                            "server_layout": case.get(
                                "server_layout", plan.get("server_layout", "split")
                            ),
                            "condition": protocol,
                            "role_gpus": {
                                "S1": plan["gpus"]["action"],
                                "S2": plan["gpus"]["vlm"]
                                if "vlm" in commands or "paired" in commands
                                else plan["gpus"]["action"],
                            },
                            "inference_gpus": sorted(
                                {
                                    plan["gpus"]["action"],
                                    *(
                                        [plan["gpus"]["vlm"]]
                                        if "vlm" in commands or "paired" in commands
                                        else []
                                    ),
                                }
                            ),
                            "renderer_gpu": plan["gpus"]["simulation"],
                            "verified_role_mapping": verified_gpus,
                            "returncode": process.returncode,
                        },
                    )
                    summary = json.loads(
                        (evaluation_output / "summary.json").read_text()
                    )
                    if not summary["complete"]:
                        raise RuntimeError("Full Spatial evaluation is incomplete")
                    results[case["name"] + "/" + protocol] = summary
                    records[protocol][case["name"]] = [
                        json.loads(row)
                        for row in (evaluation_output / "episodes.jsonl")
                        .read_text()
                        .splitlines()
                    ]
                    _write(output / "results.partial.json", results)
            except BaseException as error:
                _write(
                    output / "status.json",
                    {"phase": "failed", "condition": case["name"], "error": str(error)},
                )
                raise
            finally:
                owned.close()
        pairs = {
            protocol: paired_fixtures(
                rows,
                episodes_per_task=plan["episodes_per_task"],
                task_ids=plan.get("task_ids", range(10)),
            )
            for protocol, rows in records.items()
        }
        _write(output / "paired-initial-conditions.json", pairs)
        if not all(value["passed"] for value in pairs.values()):
            raise RuntimeError("Full-suite paired fixtures differ")
        if "deployment" in plan["protocols"]:
            deployment_report(plan, output)
        _write(output / "results.json", results)
        _write(output / "status.json", {"phase": "complete"})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def stop_requested(_number, _frame):
        raise KeyboardInterrupt("Experiment supervisor received a stop request")

    previous = signal.signal(signal.SIGTERM, stop_requested)
    try:
        run(json.loads(args.plan.read_text()), args.output)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
