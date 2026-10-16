"""Independent timing, Nsight Systems and Nsight Compute collection rounds."""

from __future__ import annotations

import math
import sqlite3
import subprocess

from .provenance import digest, read_json, write_json

METRICS = (
    "gpu__time_duration.sum",
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active_v2.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__bytes.sum.per_second",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
)


def stage_command(plan, case, stage, output, gpu, gpu_uuid):
    config = plan["stages"][stage]
    command = [
        plan["python"],
        "-m",
        plan["worker_module"],
        "--gpu",
        str(gpu),
        "--gpu-uuid",
        gpu_uuid,
        "--output",
        str(output / "worker"),
    ]
    for key, value in {
        **plan["worker_common"],
        "vlm_chunk_blocks": case.get(
            "vlm_chunk_blocks", plan["worker_common"].get("vlm_chunk_blocks", 1)
        ),
        "mode": case["mode"],
        "precision": case["precision"],
        "component": case["component"],
        "dit_offset_ms": case["dit_offset_ms"],
        "iterations": config["iterations"],
        "capture": "none" if stage == "timing" else "cuda",
        "capture_window": config["capture_window"],
    }.items():
        if key == "image_resolution" and value is None:
            continue
        command += ["--" + key.replace("_", "-"), str(value)]
    if stage == "ncu":
        command += ["--replay-output-root", str(output / "worker-passes")]
        command = [
            plan["tools"]["ncu"],
            "--replay-mode",
            "app-range",
            "--clock-control",
            "none",
            "--cache-control",
            "none",
            "--metrics",
            ",".join(config["metrics"]),
            "--export",
            str(output / "counters"),
        ] + command
    elif stage == "nsys":
        command = [
            plan["tools"]["nsys"],
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--gpu-metrics-devices=cuda-visible",
            *(
                [f"--gpu-metrics-set={config['metrics_set']}"]
                if config.get("metrics_set")
                else []
            ),
            f"--gpu-metrics-frequency={config['sampling_hz']}",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            f"--output={output / 'trace'}",
        ] + command
    if plan["cpu_affinity"]:
        command = ["taskset", "-c", plan["cpu_affinity"]] + command
    return command


def validate_workers(output, stage, case, iterations):
    roots = (
        sorted((output / "worker-passes").glob("*"))
        if stage == "ncu"
        else [output / "worker"]
    )
    roots = [path for path in roots if path.is_dir()]
    if not roots:
        raise RuntimeError("No worker output was produced")
    records = []
    for root in roots:
        status = read_json(root / "status.json")
        equivalent = read_json(root / "equivalence.json")
        measurement = read_json(root / "measurement.json")
        if (
            status.get("state") != "complete"
            or equivalent.get("passed") is not True
            or measurement.get("equivalence", {}).get("passed") is not True
        ):
            raise RuntimeError(f"Worker did not finish with valid equivalence: {root}")
        if (root / "failure.json").exists():
            raise RuntimeError(f"Worker left failure evidence: {root}")
        for field in ("mode", "precision", "component"):
            if measurement[field] != case[field]:
                raise RuntimeError(f"Worker case mismatch: {field}")
        if (
            measurement["offset_ms"] != case["dit_offset_ms"]
            or len(measurement["samples"]) != iterations
        ):
            raise RuntimeError("Worker offset/iteration mismatch")
        if measurement["protocol"]["iterations"] != iterations:
            raise RuntimeError("Worker protocol iteration count mismatch")
        files = [
            root / name
            for name in (
                "status.json",
                "equivalence.json",
                "measurement.json",
                "executed-coverage.json",
                "bootstrap.json",
                "fixtures.json",
                "streams.json",
            )
        ]
        records.append(
            {
                "root": str(root),
                "measurement": str(root / "measurement.json"),
                "files": {str(path): digest(path) for path in files},
            }
        )
    return records


def verify_counter_csv(path, required):
    from .ncu import parse_ncu_csv

    parsed = parse_ncu_csv(path)
    rows = parsed["rows"]
    if not rows:
        raise RuntimeError(f"NCU CSV has no metric table: {parsed.get('error')}")
    available = {}
    for row in rows:
        name = row.get("Metric Name")
        if name in required:
            try:
                value = float(row["numeric_value"])
            except (ValueError, TypeError, AttributeError) as error:
                raise RuntimeError(f"Unavailable NCU metric: {name}") from error
            if not math.isfinite(value) or value < 0:
                raise RuntimeError(f"Invalid NCU metric: {name}")
            if name == "gpu__time_duration.sum" and value <= 0:
                raise RuntimeError("NCU range duration is not positive")
            available[name] = {"value": value, "unit": row.get("Metric Unit")}
    missing = set(required) - set(available)
    if missing:
        raise RuntimeError(f"NCU metrics missing: {sorted(missing)}")
    return available


def postprocess(plan, gpu, stage, output, records, case):
    evidence = {}
    commands = []
    if stage == "nsys":
        report = output / "trace.nsys-rep"
        if not report.is_file():
            raise RuntimeError("Nsight Systems report is missing")
        command = [
            plan["tools"]["nsys"],
            "export",
            "--type=sqlite",
            f"--output={output / 'trace.sqlite'}",
            str(report),
        ]
        commands.append(command)
        with (output / "export.log").open("w") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                cwd=plan["worktree"],
            )
        with sqlite3.connect(f"file:{output / 'trace.sqlite'}?mode=ro", uri=True) as db:
            tables = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if not {
                "GPU_METRICS",
                "TARGET_INFO_GPU_METRICS",
                "CUPTI_ACTIVITY_KIND_KERNEL",
            }.issubset(tables):
                raise RuntimeError(
                    "Nsight trace lacks required GPU metrics/kernel tables; no zero-fill is permitted"
                )
            for table in ("GPU_METRICS", "CUPTI_ACTIVITY_KIND_KERNEL"):
                if db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] == 0:
                    raise RuntimeError(f"Nsight table is empty: {table}")
        evidence = {
            str(report): digest(report),
            str(output / "trace.sqlite"): digest(output / "trace.sqlite"),
        }
    elif stage == "ncu":
        report = output / "counters.ncu-rep"
        if not report.is_file():
            raise RuntimeError("Nsight Compute report is missing")
        command = [
            plan["tools"]["ncu"],
            "--import",
            str(report),
            "--csv",
            "--page",
            "raw",
        ]
        commands.append(command)
        with (
            (output / "counters.csv").open("w") as csv_file,
            (output / "export.log").open("w") as log,
        ):
            subprocess.run(
                command, stdout=csv_file, stderr=log, check=True, cwd=plan["worktree"]
            )
        counters = verify_counter_csv(
            output / "counters.csv", plan["stages"][stage]["metrics"]
        )
        write_json(output / "counter-validation.json", counters)
        evidence = {
            str(report): digest(report),
            str(output / "counters.csv"): digest(output / "counters.csv"),
        }
    if plan["analysis"]["enabled"]:
        command = [
            plan["python"],
            "-m",
            "coexecution.profiling.reporting",
            "analyze",
            "--measurement",
            records[0]["measurement"],
            "--out",
            str(output / "analysis"),
            "--no-plots",
        ]
        if stage == "nsys":
            command += [
                "--nsys",
                str(output / "trace.sqlite"),
                "--metric-gpu-id",
                str(gpu["index"]),
            ]
        elif stage == "ncu":
            command += [
                "--ncu",
                str(output / "counters.csv"),
                "--ncu-scope",
                "app-range",
                "--ncu-workloads",
                "1",
            ]
        commands.append(command)
        with (output / "analysis.log").open("w") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                cwd=plan["worktree"],
            )
    write_json(
        output / "postprocess.json",
        {
            "commands": commands,
            "case": case,
            "stage": stage,
            "counter_attribution": "entire GPU/profile range, not individual concurrent models",
        },
    )
    return evidence
