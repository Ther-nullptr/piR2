"""Analyze captures or aggregate a measured hardware-contention campaign."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
from pathlib import Path

from .aggregation import aggregate_measurements, scheduler_records
from .analysis import read_measurement
from .figures import latency_comparison, request_figure, scheduler_offsets, timeline
from .labels import LATENCY_FIELDS, LATENCY_LABELS
from .ncu import read_ncu
from .nsight import read_nsys


def latency_summary_html(measurement):
    """Expose request response before stream spans without imputing old records."""
    label = (
        "无 profiler"
        if measurement.get("unprofiled")
        else "插桩运行，不能替代无 profiler 时延"
    )
    parts = [
        "<h2>请求响应与流跨度</h2>",
        f"<p>{label}；release→ready 包含 CPU 提交、CUDA 等待/执行及主机唤醒。缺失响应不以流跨度代替。</p>",
        "<table><thead><tr><th>口径</th><th>样本数</th><th>p50 (ms)</th><th>p95 (ms)</th></tr></thead><tbody>",
    ]
    for field, name in LATENCY_FIELDS:
        values = measurement.get("latency", {}).get(field, {})

        def display(key, snapshot=values):
            value = snapshot.get(key)
            return f"{value:.3f}" if value is not None else "未记录"

        parts.append(
            f"<tr><td>{html.escape(name)}</td><td>{values.get('n', 0)}</td><td>{display('p50')}</td><td>{display('p95')}</td></tr>"
        )
    parts.append("</tbody></table>")
    return "\n".join(parts)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def analyze(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    measurement = read_measurement(args.measurement)
    nsys = read_nsys(args.nsys) if args.nsys else None
    ncu = read_ncu(args.ncu, args.ncu_workloads, args.ncu_scope) if args.ncu else None
    if ncu:
        ncu["measurement_capture_region"] = measurement.get("measured_region", {})
        ncu["workload_denominator_note"] = (
            "Denominator counts captured worker workloads. If capture_includes_restore is true, bytes and GB/s include pre-pair restore; they are not model-only traffic."
        )
    evidence = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "measurement": measurement,
        "nsys": nsys,
        "ncu": ncu,
        "missing": [
            name
            for name, value in (
                ("Nsight Systems counters/timeline", nsys),
                ("NCU DRAM bytes and achieved GB/s", ncu),
            )
            if value is None
        ],
        "figures": [],
    }
    if nsys and not args.no_plots:
        evidence["figures"].extend(
            timeline(
                nsys,
                measurement,
                out,
                args.start_ms,
                args.duration_ms,
                args.metric_gpu_id,
                args.font,
            )
        )
        evidence["figures"].extend(request_figure(nsys, measurement, out, args.font))
    write_json(out / "evidence.json", evidence)
    index = [
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>πR² 硬件争用实测</title><style>body{max-width:1400px;margin:2em auto;font:16px sans-serif}img{width:100%}pre{white-space:pre-wrap}</style><h1>πR² 硬件争用实测</h1>'
    ]
    index.append(latency_summary_html(measurement))
    for figure in evidence["figures"]:
        if figure["path"].endswith(".svg"):
            index.append(f'<img src="{html.escape(Path(figure["path"]).name)}">')
    index.append(
        '<p><a href="evidence.json">完整数值、counter 原名、单位与源文件 SHA256</a></p>'
    )
    index.append(
        "<p>Counter 属于整卡；CUDA stream 优先级不证明正在执行的 kernel 被硬抢占。NCU replay 结果不能替代无 profiler 延迟。</p>"
    )
    index.append(
        "<pre>"
        + html.escape(
            json.dumps(
                {
                    "measurement": measurement,
                    "missing": evidence["missing"],
                    "ncu_summary": {
                        k: v for k, v in (ncu or {}).items() if k != "rows"
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        + "</pre></html>"
    )
    (out / "index.html").write_text("\n".join(index))
    print(
        json.dumps(
            {
                "output": str(out.resolve()),
                "figures": len(evidence["figures"]),
                "samples": measurement["sample_count"],
                "missing": evidence["missing"],
            },
            ensure_ascii=False,
        )
    )


def campaign(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    measurements, rows, acceptance = aggregate_measurements(args.root)
    write_json(
        out / "campaign.json",
        {
            "rows": rows,
            "attempt_acceptance": acceptance,
            "sources": [
                {k: v for k, v in item.items() if k != "latency_samples"}
                for item in measurements
            ],
            "comparison_scope": "unprofiled release-to-ready response first; includes cooperative permit wait and host wake-up. CUDA stream spans are separate and exclude pre-start permit wait. Missing response stays absent; schemas and worker source cohorts are never merged.",
            "precision_quality": "W4 hardware observations do not establish task accuracy",
            "profiled_runs_excluded_from_latency": sum(
                not item["unprofiled"] for item in measurements
            ),
        },
    )
    with (out / "latency.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cohort_sha256",
                "precision",
                "component",
                "mode",
                "offset_ms",
                "repeat_files",
                "metric",
                "metric_label",
                "n",
                "mean",
                "p50",
                "p95",
                "min",
                "max",
            ],
        )
        writer.writeheader()
        for row in rows:
            for field in sorted(
                row["latency"],
                key=lambda key: (
                    [item[0] for item in LATENCY_FIELDS].index(key)
                    if key in LATENCY_LABELS
                    else len(LATENCY_FIELDS),
                    key,
                ),
            ):
                values = row["latency"][field]
                writer.writerow(
                    {
                        key: row[key]
                        for key in (
                            "cohort_sha256",
                            "precision",
                            "component",
                            "mode",
                            "offset_ms",
                            "repeat_files",
                        )
                    }
                    | {
                        "metric": field,
                        "metric_label": LATENCY_LABELS.get(field, field),
                    }
                    | values
                )
    if rows and not args.no_plots:
        latency_comparison(rows, out, args.font)
    scheduler = scheduler_records(args.root)
    write_json(out / "scheduler.json", scheduler)
    if scheduler and not args.no_plots:
        scheduler_offsets(scheduler, out, args.font)
    print(
        json.dumps(
            {
                "measurements": len(measurements),
                "unprofiled_groups": len(rows),
                "scheduler_runs": len(scheduler),
                "output": str(out.resolve()),
            },
            ensure_ascii=False,
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("analyze")
    one.add_argument("--measurement", required=True)
    one.add_argument("--nsys")
    one.add_argument("--ncu")
    one.add_argument("--ncu-workloads", type=int)
    one.add_argument(
        "--ncu-scope", choices=("app-range", "kernel", "unknown"), default="unknown"
    )
    one.add_argument("--metric-gpu-id", type=int)
    one.add_argument("--start-ms", type=float, default=0)
    one.add_argument("--duration-ms", type=float, default=300)
    one.add_argument("--out", required=True)
    one.set_defaults(func=analyze)
    all_runs = commands.add_parser("campaign")
    all_runs.add_argument("--root", required=True)
    all_runs.add_argument("--out", required=True)
    all_runs.set_defaults(func=campaign)
    for command in (one, all_runs):
        command.add_argument(
            "--font",
            type=Path,
            help="Source Han / Noto CJK font file for Chinese plots",
        )
        command.add_argument(
            "--no-plots",
            action="store_true",
            help="Write numerical evidence without matplotlib or fonts",
        )
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
