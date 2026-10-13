"""Read native Nsight Compute CSV exports with raw cells and units preserved."""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

from .analysis import digest


def parse_ncu_csv(path):
    """Normalize native NCU long or wide CSV while retaining exact source cells.

    Wide raw-page exports have a header row, a units row, then range rows.
    Consumers share this parser instead of maintaining independent formats.
    """
    raw_rows = list(csv.reader(io.StringIO(Path(path).read_text(errors="replace"))))
    header_index = next(
        (
            index
            for index, row in enumerate(raw_rows)
            if "Metric Name" in row and "Metric Value" in row
        ),
        None,
    )
    long_format = header_index is not None
    if not long_format:
        header_index = next(
            (
                index
                for index, row in enumerate(raw_rows)
                if "ID" in row and any("__" in name for name in row)
            ),
            None,
        )
    if header_index is None:
        raise ValueError("No recognized native NCU long/wide CSV metric header")
    header = raw_rows[header_index]
    rows = []
    if long_format:
        for index in range(header_index + 1, len(raw_rows)):
            cells = raw_rows[index]
            if len(cells) != len(header) or cells == header:
                continue
            row = dict(zip(header, cells))
            if not row.get("Metric Name"):
                continue
            row["source_csv_row_index"] = index
            rows.append(row)
    else:
        if header_index + 1 >= len(raw_rows):
            raise ValueError("NCU wide CSV is missing its unit row")
        units = raw_rows[header_index + 1]
        if len(units) != len(header):
            raise ValueError("NCU wide unit row does not match header width")
        metric_columns = [i for i, name in enumerate(header) if "__" in name]
        metadata_columns = [i for i in range(len(header)) if i not in metric_columns]
        for index in range(header_index + 2, len(raw_rows)):
            cells = raw_rows[index]
            if not cells or all(not cell for cell in cells):
                continue
            if len(cells) != len(header):
                raise ValueError(
                    f"NCU wide row {index} has {len(cells)} columns, expected {len(header)}"
                )
            metadata = {header[i]: cells[i] for i in metadata_columns}
            for column in metric_columns:
                rows.append(
                    {
                        **metadata,
                        "Metric Name": header[column],
                        "Metric Unit": units[column],
                        "Metric Value": cells[column],
                        "source_csv_row_index": index,
                        "source_csv_column_index": column,
                    }
                )
    for row in rows:
        try:
            numeric = float(row["Metric Value"].replace(",", ""))
        except (TypeError, ValueError):
            numeric = None
        row["numeric_value"] = (
            numeric if numeric is not None and math.isfinite(numeric) else None
        )
    return {
        "format": "long" if long_format else "wide",
        "header_row_index": header_index,
        "raw_csv_rows": raw_rows,
        "rows": rows,
    }


def read_ncu(path, workloads=None, scope="unknown"):
    """Read NCU replay counters; GB/s is never unprofiled service bandwidth."""
    parsed = parse_ncu_csv(path)
    rows = parsed["rows"]
    wanted = {
        "dram__bytes_read.sum": [],
        "dram__bytes_write.sum": [],
        "dram__bytes.sum": [],
        "gpu__time_duration.sum": [],
        "dram__bytes.sum.per_second": [],
    }
    for row in rows:
        if row["Metric Name"] in wanted:
            wanted[row["Metric Name"]].append(row)
    scales = {
        "byte": 1,
        "Kbyte": 1e3,
        "Mbyte": 1e6,
        "Gbyte": 1e9,
        "nsecond": 1e-9,
        "usecond": 1e-6,
        "msecond": 1e-3,
        "second": 1,
        "ns": 1e-9,
        "us": 1e-6,
        "ms": 1e-3,
        "s": 1,
        "byte/s": 1,
        "Kbyte/s": 1e3,
        "Mbyte/s": 1e6,
        "Gbyte/s": 1e9,
    }

    def total(name):
        matches = wanted[name]
        if not matches or any(
            row["numeric_value"] is None or row.get("Metric Unit") not in scales
            for row in matches
        ):
            return None
        return sum(row["numeric_value"] * scales[row["Metric Unit"]] for row in matches)

    read, written = total("dram__bytes_read.sum"), total("dram__bytes_write.sum")
    byte_count = (
        read + written
        if read is not None and written is not None
        else total("dram__bytes.sum")
    )
    duration_s = total("gpu__time_duration.sum")
    # Multiple kernel rows cannot be treated as a concurrent range wall clock.
    range_scoped = len(wanted["gpu__time_duration.sum"]) == 1 and all(
        len(wanted[name]) <= 1
        for name in ("dram__bytes_read.sum", "dram__bytes_write.sum", "dram__bytes.sum")
    )
    replay_gb_per_s = (
        byte_count / duration_s / 1e9
        if byte_count is not None
        and duration_s
        and range_scoped
        and scope == "app-range"
        else None
    )
    reported_rate = (
        total("dram__bytes.sum.per_second")
        if len(wanted["dram__bytes.sum.per_second"]) == 1
        else None
    )
    return {
        "source": str(Path(path).resolve()),
        "sha256": digest(path),
        **parsed,
        "dram_read_bytes": read,
        "dram_write_bytes": written,
        "dram_total_bytes": byte_count,
        "workloads": workloads,
        "replay_scope": scope,
        "dram_bytes_per_workload": byte_count / workloads
        if byte_count is not None and workloads and scope == "app-range"
        else None,
        "duration_counter_seconds": duration_s,
        "single_range_or_single_kernel": range_scoped,
        "dram_gb_per_s": replay_gb_per_s,
        "replay_dram_gb_per_s": replay_gb_per_s,
        "replay_dram_gb_per_s_reported_counter": reported_rate / 1e9
        if reported_rate is not None
        else None,
        "normal_service_dram_gb_per_s": None,
        "bandwidth_label": "NCU重放窗口 GB/s",
        "interpretation": "Prioritize bytes/workload for traffic comparisons. NCU app-range instrumentation can greatly prolong the capture window; its GB/s describes that replay window, not normal unprofiled inference bandwidth. Normal-run Nsight Systems DRAM percentage curves remain separate. Never divide these bytes by a different run/device's latency and label the result measured bandwidth.",
        "limitations": [
            "NCU replay changes execution; latency comparisons must come from unprofiled measurement.json.",
            "Every GB/s value in this record is for the NCU replay window; normal-service GB/s is not measured here.",
            "Combined VLM/DiT range bytes are bytes per pair, not separately attributed to the models.",
            "Summed per-kernel time is not concurrent wall time; GB/s omitted for multiple timing rows.",
            "Confirm app-range capture from the launch manifest before interpreting a single row as a whole workload.",
        ],
    }
