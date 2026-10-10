"""Serialize fair algorithm/deployment comparisons on a fixed GPU layout."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/libero-protocols"


def acquire_supervisor_lock(kind):
    ART.mkdir(parents=True, exist_ok=True)
    lock = (ART / f".{kind}-supervisor.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(
            f"Another {kind} supervisor owns these experiment outputs"
        ) from None
    return lock


def write_status(path, phase, **extra):
    path.write_text(
        json.dumps(
            {
                "phase": phase,
                "supervisor_pid": os.getpid(),
                "updated_at_unix": time.time(),
                **extra,
            },
            indent=2,
        )
        + "\n"
    )
    print(phase, extra, flush=True)


def stop_job(path):
    if not path.exists():
        return
    job = json.loads(path.read_text())
    pid = job["pid"]
    proc = Path(f"/proc/{pid}/cmdline")
    if proc.exists() and b"serve_libero_protocol.py" in proc.read_bytes():
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while proc.exists() and proc.read_bytes():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Server{pid} did not stop")
            time.sleep(0.2)


def wait_idle(gpu, status):
    while True:
        data = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        memory, util = [int(x.strip()) for x in data.strip().split(",")]
        pids = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        if memory <= 200 and util == 0 and not pids:
            return
        write_status(
            status,
            "waiting_for_clean_gpu",
            gpu=gpu,
            memory_mib=memory,
            utilization=util,
        )
        time.sleep(30)


def environment(gpu, client=False):
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONPATH=os.pathsep.join(
            [str(ROOT), str(ROOT / "upstream/learning/Isaac-GR00T")]
        ),
        PYTHONUNBUFFERED="1",
        PYTHONHASHSEED="1000",
        OMP_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        NO_ALBUMENTATIONS_UPDATE="1",
        GROOT_HF_LOCAL_FIRST="1",
        GROOT_PATCH_MISTRAL="1",
        HF_HUB_OFFLINE="1",
    )
    if client:
        env.update(
            MUJOCO_EGL_DEVICE_ID=str(gpu),
            MUJOCO_GL="egl",
            PYOPENGL_PLATFORM="egl",
            LIBERO_CONFIG_PATH=str(ROOT / ".libero-config-pi05"),
        )
        if env.get("PIR2_LIBERO_LD_PRELOAD"):
            env["LD_PRELOAD"] = env["PIR2_LIBERO_LD_PRELOAD"]
    return env


def launch_server(checkpoint, variant, role, gpu, port, output, job_path, status):
    wait_idle(gpu, status)
    command = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/serve_libero_protocol.py"),
        "--checkpoint",
        str(checkpoint),
        "--variant",
        variant,
        "--role",
        role,
        "--port",
        str(port),
        "--output",
        str(output),
    ]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "server.log").open("a") as log:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment(gpu),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    job_path.write_text(
        json.dumps({"pid": proc.pid, "gpu": gpu, "command": command}, indent=2) + "\n"
    )
    return proc


def compare_initial_conditions(root):
    paired = {}
    mismatches = []
    for protocol in ["algorithm", "deployment"]:
        variants = {}
        for variant in ["pir2", "flow"]:
            rows = [
                json.loads(x)
                for x in (root / f"{protocol}-{variant}" / "episodes.jsonl")
                .read_text()
                .splitlines()
            ]
            variants[variant] = {(x["task_id"], x["episode_id"]): x for x in rows}
        for key in variants["pir2"]:
            a = variants["pir2"][key]
            b = variants["flow"][key]
            fields = [
                "initial_state_sha256",
                "fixture_model_sha256",
                "settled_sim_state_sha256",
                "initial_rgb_sha256",
            ]
            for field in fields:
                if a[field] != b[field]:
                    mismatches.append(
                        {
                            "protocol": protocol,
                            "task_id": key[0],
                            "episode_id": key[1],
                            "field": field,
                        }
                    )
        paired[protocol] = len(variants["pir2"])
    result = {
        "matched_episode_counts": paired,
        "mismatches": mismatches,
        "passed": not mismatches,
    }
    (root / "paired-initial-conditions.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    if mismatches:
        raise RuntimeError("Paired initial conditions differ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    supervisor_lock = acquire_supervisor_lock("evaluation")
    label = args.label or f"step{args.step}"
    root = ART / label
    root.mkdir(parents=True, exist_ok=True)
    status = ART / f"{label}-status.json"
    action_job = ART / "active-action-server.json"
    vision_job = ART / "active-vision-server.json"
    try:
        write_status(status, "preparing")
        for legacy in ["pir2-server-job.json", "vlm-server-job.json"]:
            stop_job(ART / legacy)
        stop_job(action_job)
        stop_job(vision_job)
        launch_server(
            ROOT / "models/GR00T-N1.7-LIBERO/libero_spatial",
            "flow",
            "vlm",
            2,
            5572,
            root / "vlm-server",
            vision_job,
            status,
        )
        results = {}
        for variant in ["pir2", "flow"]:
            checkpoint = (
                ROOT
                / f"outputs/libero-{variant}-seed1000/libero-{variant}-seed1000/checkpoint-{args.step}"
            )
            assert (checkpoint / "trainer_state.json").exists(), str(checkpoint)
            stop_job(action_job)
            launch_server(
                checkpoint,
                variant,
                "action",
                0,
                5570,
                root / f"{variant}-server",
                action_job,
                status,
            )
            for protocol in ["algorithm", "deployment"]:
                name = f"{protocol}-{variant}"
                out = root / name
                summary = out / "summary.json"
                # Even completed conditions pass the evaluator's exact config,
                # checkpoint and source-hash resume checks before reuse.
                wait_idle(3, status)
                command = [
                    str(ROOT / ".venv-pi05/bin/python"),
                    str(ROOT / "scripts/evaluate_libero_protocol.py"),
                    "--protocol",
                    protocol,
                    "--variant",
                    variant,
                    "--output",
                    str(out),
                    "--episodes-per-task",
                    str(args.episodes_per_task),
                    "--visual-delay",
                    "3",
                    "--action-delay",
                    "1",
                ]
                out.mkdir(parents=True, exist_ok=True)
                with (out / "evaluation.log").open("a") as log:
                    client = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=environment(3, client=True),
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    write_status(
                        status,
                        name,
                        client_pid=client.pid,
                        checkpoint=str(checkpoint),
                        command=command,
                    )
                    if client.wait():
                        raise RuntimeError(f"{name} failed; see{out}/evaluation.log")
                result = json.loads(summary.read_text())
                assert result["complete"]
                results[name] = result
            stop_job(action_job)
        compare_initial_conditions(root)
        (root / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
        subprocess.run(
            [
                str(ROOT / ".venv-pi05/bin/python"),
                str(ROOT / "scripts/report_libero_protocols.py"),
                str(root),
            ],
            cwd=ROOT,
            check=True,
        )
    except Exception as error:
        write_status(status, "failed", error=str(error))
        raise
    finally:
        stop_job(action_job)
        stop_job(vision_job)
        supervisor_lock.close()
    write_status(status, "complete", results=str(root / "comparison.json"))


if __name__ == "__main__":
    main()
