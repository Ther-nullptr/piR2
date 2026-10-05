"""Run the approved two-clock experiment and its matched training continuation."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from run_libero_protocols import (
    acquire_supervisor_lock,
    compare_initial_conditions,
    environment,
    stop_job,
    wait_idle,
    write_status,
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/libero-protocols"
STATE = ART / "pipeline-status.json"


def record(phase, **extra):
    STATE.write_text(
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


def alive(pid):
    p = Path(f"/proc/{pid}/cmdline")
    return p.exists() and bool(p.read_bytes())


def await_integration():
    record("waiting_for_protocol_integration")
    path = ART / "protocol-v4-integration-status.json"
    while True:
        try:
            status = json.loads(path.read_text())
        except (OSError, ValueError):
            time.sleep(10)
            continue
        if status["phase"] == "failed":
            raise RuntimeError(status.get("error", "Integration failed"))
        if status["phase"] == "complete" and not alive(status["supervisor_pid"]):
            break
        if not alive(status["supervisor_pid"]) and status["phase"] != "complete":
            raise RuntimeError("Integration supervisor exited without completion")
        time.sleep(10)
    assert json.loads((ART / "gpu-contracts.json").read_text())["passed"]
    assert json.loads(
        (ART / "protocol-v4-integration/paired-initial-conditions.json").read_text()
    )["passed"]


def run_comparison(step):
    record(f"protocol_comparison_step{step}")
    cmd = [
        "python3",
        str(ROOT / "scripts/run_libero_protocols.py"),
        "--step",
        str(step),
    ]
    with (ROOT / f"artifacts/logs/protocol-step{step}-runner.log").open("a") as log:
        child = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        record(f"protocol_comparison_step{step}", child_pid=child.pid, command=cmd)
        if child.wait():
            raise RuntimeError(f"Step{step} comparison failed")
    assert (
        json.loads((ART / f"step{step}-status.json").read_text())["phase"] == "complete"
    )


def start_training(variant):
    wait_idle(1, STATE)
    env = environment(1)
    env.update(CUDA_HOME="/usr/local/cuda-12.8", OMP_NUM_THREADS="4")
    env["PATH"] = "/usr/local/cuda-12.8/bin:" + env["PATH"]
    cmd = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/train_groot_spatial.py"),
        "--variant",
        variant,
        "--steps",
        "10000",
    ]
    with (ROOT / f"artifacts/logs/libero-{variant}-train.log").open("a") as log:
        child = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        record(f"{variant}_continuation_to_10000", training_pid=child.pid, command=cmd)
        (ART / f"{variant}-train-job.json").write_text(
            json.dumps(
                {
                    "pid": child.pid,
                    "gpu": 1,
                    "command": cmd,
                    "process_group": child.pid,
                },
                indent=2,
            )
            + "\n"
        )
    return child


def wait_training(child, variant):
    if child.wait():
        raise RuntimeError(f"{variant} training continuation failed")
    ckpt = (
        ROOT
        / f"outputs/libero-{variant}-seed1000/libero-{variant}-seed1000/checkpoint-10000"
    )
    assert (ckpt / "trainer_state.json").exists()
    assert json.loads((ckpt / "trainer_state.json").read_text())["global_step"] == 10000


def train(variant):
    wait_training(start_training(variant), variant)


def archive_stage500():
    for variant in ["pir2", "flow"]:
        source = (
            ROOT
            / f"outputs/libero-{variant}-seed1000/libero-{variant}-seed1000/checkpoint-500"
        )
        target = ROOT / f"outputs/libero-stage500-archive/{variant}"
        if not target.exists():
            shutil.copytree(source, target, copy_function=os.link)
        assert (target / "optimizer.pt").exists()
        assert (
            json.loads((target / "trainer_state.json").read_text())["global_step"]
            == 500
        )


def pause_training(child):
    if child.poll() is not None:
        if child.returncode:
            raise RuntimeError("Concurrent piR2 training failed")
        return False
    os.killpg(child.pid, signal.SIGSTOP)
    deadline = time.monotonic() + 60
    quiet = 0
    while quiet < 3:
        if time.monotonic() > deadline:
            os.killpg(child.pid, signal.SIGCONT)
            raise TimeoutError("Training GPU did not become quiet")
        status = Path(f"/proc/{child.pid}/status").read_text()
        stopped = any(x.startswith("State:") and "T" in x for x in status.splitlines())
        util = int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    "1",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        quiet = quiet + 1 if stopped and util == 0 else 0
        time.sleep(1)
    return True


def concurrent_continuation(client_pid):
    """Adopt the running fixed-delay client; reserve deployment time for evaluation."""
    with acquire_supervisor_lock("evaluation") as evaluation_lock:
        archive_stage500()
        root = ART / "step500"
        step_status = ART / "step500-status.json"
        training = start_training("pir2")
        record(
            "pir2_training_and_algorithm_flow",
            training_pid=training.pid,
            client_pid=client_pid,
        )
        write_status(step_status, "algorithm-flow", client_pid=client_pid, adopted=True)
        while alive(client_pid):
            if training.poll() not in (None, 0):
                raise RuntimeError("Concurrent piR2 training failed")
            time.sleep(2)
        summary = json.loads((root / "algorithm-flow/summary.json").read_text())
        if not summary["complete"]:
            raise RuntimeError("Adopted algorithm-flow exited before completion")
        paused = pause_training(training)
        pause_start = time.time()
        try:
            record(
                "deployment_flow_training_paused",
                training_pid=training.pid,
                paused=paused,
            )
            wait_idle(3, step_status)
            output = root / "deployment-flow"
            output.mkdir(parents=True, exist_ok=True)
            command = [
                str(ROOT / ".venv-pi05/bin/python"),
                str(ROOT / "scripts/evaluate_libero_protocol.py"),
                "--protocol",
                "deployment",
                "--variant",
                "flow",
                "--output",
                str(output),
                "--episodes-per-task",
                "20",
                "--visual-delay",
                "3",
                "--action-delay",
                "1",
            ]
            with (output / "evaluation.log").open("a") as log:
                client = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment(3, client=True),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                write_status(
                    step_status,
                    "deployment-flow",
                    client_pid=client.pid,
                    training_paused=paused,
                    training_pid=training.pid,
                )
                if client.wait():
                    raise RuntimeError("deployment-flow failed")
        finally:
            if paused:
                os.killpg(training.pid, signal.SIGCONT)
            (ART / "training-deployment-pause.json").write_text(
                json.dumps(
                    {
                        "training_pid": training.pid,
                        "paused": paused,
                        "start_unix": pause_start,
                        "end_unix": time.time(),
                        "reason": "same quiet-host condition for deployment timing",
                    },
                    indent=2,
                )
                + "\n"
            )
        results = {}
        for protocol in ["algorithm", "deployment"]:
            for variant in ["pir2", "flow"]:
                name = f"{protocol}-{variant}"
                result = json.loads((root / name / "summary.json").read_text())
                assert result["complete"]
                results[name] = result
        compare_initial_conditions(root)
        (root / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
        subprocess.run(
            [
                str(ROOT / ".venv-pi05/bin/python"),
                str(ROOT / "scripts/report_libero_protocols.py"),
                str(root),
            ],
            check=True,
            cwd=ROOT,
        )
        stop_job(ART / "active-action-server.json")
        stop_job(ART / "active-vision-server.json")
        write_status(step_status, "complete", results=str(root / "comparison.json"))
        evaluation_lock.close()
        record("pir2_continuation_to_10000", training_pid=training.pid)
        wait_training(training, "pir2")
        train("flow")
        run_comparison(10000)
        record("complete", results=str(ART / "step10000/comparison.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage500-only", action="store_true")
    parser.add_argument("--adopt-step500-client", type=int)
    args = parser.parse_args()
    pipeline_lock = acquire_supervisor_lock("pipeline")
    (ROOT / "artifacts/logs").mkdir(parents=True, exist_ok=True)
    try:
        if args.adopt_step500_client:
            concurrent_continuation(args.adopt_step500_client)
            return
        await_integration()
        run_comparison(500)
        if args.stage500_only:
            record("complete_stage500")
            return
        archive_stage500()
        train("pir2")
        train("flow")
        # No concurrent training during deployment timing measurements.
        run_comparison(10000)
        record("complete", results=str(ART / "step10000/comparison.json"))
    except Exception as error:
        record("failed", error=str(error))
        raise
    finally:
        pipeline_lock.close()


if __name__ == "__main__":
    main()
