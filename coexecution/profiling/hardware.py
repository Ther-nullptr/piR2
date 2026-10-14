"""Read-only GPU admission and lifetime-aware ownership of launched processes.

Only processes launched by this supervisor can be terminated. GPU clocks, power
limits and system profiler permissions are never modified here.
"""

from __future__ import annotations

import csv
import io
import os
import signal
import subprocess
import time
from pathlib import Path

from .provenance import now, run_text


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": int(pid),
            "ppid": int(fields[1]),
            "pgid": int(fields[2]),
            "state": fields[0],
            "start_ticks": int(fields[19]),
        }
    except (
        FileNotFoundError,
        ProcessLookupError,
        PermissionError,
        IndexError,
        ValueError,
    ):
        return None


def process_snapshot():
    return {
        int(path.name): identity
        for path in Path("/proc").iterdir()
        if path.name.isdecimal()
        and (identity := process_identity(int(path.name))) is not None
    }


def is_ncu_process_name(name):
    name = Path(name).name
    return name in {
        "ncu",
        "ncu-ui",
        "nv-nsight-cu",
        "nv-nsight-cu-cli",
    } or name.startswith("nv-nsight-cu-")


def ncu_processes(snapshot=None):
    result = []
    snapshot = process_snapshot() if snapshot is None else snapshot
    for pid, identity in snapshot.items():
        if identity["state"] == "Z":
            continue
        names = []
        try:
            names.append(os.readlink(f"/proc/{pid}/exe"))
        except (OSError, PermissionError):
            pass
        try:
            names.append(Path(f"/proc/{pid}/comm").read_text().strip())
        except (OSError, PermissionError):
            pass
        if any(is_ncu_process_name(name) for name in names):
            result.append({**identity, "names": names})
    return result


def owned_processes(leader, known, snapshot):
    """Track only the launched session's descendants, with PID reuse protection."""
    owned = {
        pid: identity
        for pid, identity in snapshot.items()
        if pid in known and identity["start_ticks"] == known[pid]["start_ticks"]
    }
    if (
        leader["pid"] in snapshot
        and snapshot[leader["pid"]]["start_ticks"] == leader["start_ticks"]
    ):
        owned[leader["pid"]] = snapshot[leader["pid"]]
    changed = True
    while changed:
        changed = False
        for pid, identity in snapshot.items():
            if pid not in owned and identity["ppid"] in owned:
                owned[pid] = identity
                changed = True
    # The launcher uses start_new_session=True. Same-pgid helpers survive reparenting.
    if any(identity["pgid"] == leader["pid"] for identity in owned.values()):
        owned.update(
            {
                pid: identity
                for pid, identity in snapshot.items()
                if identity["pgid"] == leader["pid"]
            }
        )
    known.update(owned)
    return owned


def gpu_snapshot(index):
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("Physical GPU index must be a nonnegative integer")
    rows = list(
        csv.reader(
            io.StringIO(
                run_text(
                    [
                        "nvidia-smi",
                        "-i",
                        str(index),
                        "--query-gpu=index,uuid,utilization.gpu,memory.used,clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ]
                )
            )
        )
    )
    if len(rows) != 1:
        raise RuntimeError("NVML returned an ambiguous device")
    values = [value.strip() for value in rows[0]]
    compute = []
    apps = run_text(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    for row in csv.reader(io.StringIO(apps)):
        if len(row) < 4:
            raise RuntimeError(f"Cannot interpret GPU compute-process query: {row}")
        compute.append(
            {
                "uuid": row[0].strip(),
                "pid": int(row[1]),
                "name": row[2].strip(),
                "memory_mib": row[3].strip(),
            }
        )
    if int(values[0]) != index:
        raise RuntimeError("NVML GPU index differs from requested physical GPU")
    return {
        "utc": now(),
        "monotonic_ns": time.monotonic_ns(),
        "index": int(values[0]),
        "uuid": values[1],
        "utilization_gpu": float(values[2]),
        "memory_used_mib": float(values[3]),
        "sm_clock_mhz": values[4],
        "memory_clock_mhz": values[5],
        "power_w": values[6],
        "power_limit_w": values[7],
        "temperature_c": values[8],
        "compute_processes": compute,
        "source": "nvidia-smi/NVML",
    }


def is_clean(sample):
    return (
        not sample["compute_processes"]
        and sample["utilization_gpu"] == 0
        and sample["memory_used_mib"] <= 200
    )


def foreign_pids(sample, owned):
    return [
        item["pid"] for item in sample["compute_processes"] if item["pid"] not in owned
    ]


def terminate_owned_group(process, leader, known):
    owned = owned_processes(leader, known, process_snapshot())
    if not any(item["pgid"] == leader["pid"] for item in owned.values()):
        return
    # Never signal a foreign PID; only the session created by this supervisor.
    try:
        os.killpg(leader["pid"], signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = owned_processes(leader, known, process_snapshot())
        if not any(
            item["pgid"] == leader["pid"] and item["state"] != "Z"
            for item in current.values()
        ):
            break
        time.sleep(0.1)
    current = owned_processes(leader, known, process_snapshot())
    if any(
        item["pgid"] == leader["pid"] and item["state"] != "Z"
        for item in current.values()
    ):
        try:
            os.killpg(leader["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
