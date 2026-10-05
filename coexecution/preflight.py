"""Refuse performance measurements on a busy GPU; never changes device settings."""

import argparse
import csv
import datetime
import io
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fields = [
        "index",
        "uuid",
        "name",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "clocks.sm",
        "clocks.mem",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "driver_version",
    ]
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(args.gpu),
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    row = next(csv.reader(io.StringIO(text)))
    info = dict(zip(fields, [x.strip() for x in row]))
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    active = [
        r
        for r in csv.reader(io.StringIO(processes))
        if r and r[0].strip() == info["uuid"]
    ]
    info.update(
        {
            "compute_processes": active,
            "checked_at": datetime.datetime.now().astimezone().isoformat(),
        }
    )
    info["clean_empty"] = (
        not active
        and float(info["utilization.gpu"]) == 0
        and float(info["memory.used"]) <= 200
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(info, indent=2))
    if not info["clean_empty"]:
        raise RuntimeError(
            f"GPU {args.gpu} is not clean/idle; performance run stopped. See {args.output}"
        )
    print(f"GPU {args.gpu}: clean, idle, {info['name']}; single-device run only")


if __name__ == "__main__":
    main()
