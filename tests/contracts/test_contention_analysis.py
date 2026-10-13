"""Synthetic metadata tests; no model inference or GPU execution."""

import contextlib
import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from coexecution.profiling.analysis import (
    attempt_acceptance,
    duration,
    intersection,
    read_measurement,
    weighted_samples,
)
from coexecution.profiling.ncu import read_ncu
from coexecution.profiling.nsight import read_nsys


class EvidenceTests(unittest.TestCase):
    def test_union_and_real_overlap(self):
        self.assertEqual(duration([[0, 10], [5, 15], [20, 25]]), 20)
        self.assertEqual(intersection([[0, 10], [5, 15]], [[12, 17]]), [[12, 15]])

    def test_irregular_samples_weight_time_not_sample_count(self):
        result = weighted_samples([[0, 10.0], [10, 30.0], [30, 90.0]], [[5, 20]])
        self.assertAlmostEqual(result["mean"], 350 / 15)
        self.assertEqual(result["covered_ns"], 15)
        final = weighted_samples([[0, 10.0], [10, 30.0]], [[10, 20]])
        self.assertIsNone(final["mean"])
        missing = weighted_samples([[0, 10.0], [100, 30.0]], [[0, 100]], max_gap_ns=10)
        self.assertIsNone(missing["mean"])

    def test_sqlite_ownership_cross_thread_correlation_and_metric_type(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-contention-") as directory:
            path = Path(directory) / "synthetic.sqlite"
            db = sqlite3.connect(path)
            db.executescript("""
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE PROCESSES(globalPid INTEGER, pid INTEGER);
            CREATE TABLE TARGET_INFO_CUDA_DEVICE(gpuId INTEGER, cudaId INTEGER, pid INTEGER);
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, globalTid INTEGER, textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, globalPid INTEGER, correlationId INTEGER, demangledName INTEGER);
            CREATE TABLE GPU_METRICS(timestamp INTEGER, rawTimestamp INTEGER, typeId INTEGER, metricId INTEGER, value REAL);
            CREATE TABLE TARGET_INFO_GPU_METRICS(typeId INTEGER, sourceId INTEGER, typeName TEXT, metricId INTEGER, metricName TEXT);
            """)
            pid = 123 << 24
            other_pid = 321 << 24
            db.execute("INSERT INTO PROCESSES VALUES (?, ?)", (pid, 123))
            db.execute("INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (2, 0, 123)")
            db.executemany(
                "INSERT INTO StringIds VALUES (?,?)", [(1, "synthetic_kernel")]
            )
            db.executemany(
                "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)",
                [
                    (0, 100, "pir2/phase=warmup", pid + 1, None),
                    (
                        10,
                        None,
                        "S1_RELEASE/concurrent_equal/iteration=0",
                        pid + 1,
                        None,
                    ),
                    (11, 19, "S1/DIT/concurrent_equal/iteration=0", pid + 1, None),
                    (100, 300, "pir2/phase=measurement", pid + 1, None),
                    (110, 290, "PAIR/concurrent_equal/0", pid + 1, None),
                    (
                        115,
                        None,
                        "S1_RELEASE/concurrent_equal/iteration=0",
                        pid + 2,
                        None,
                    ),
                    (116, 140, "S1/DIT/concurrent_equal/iteration=0", pid + 2, None),
                    (110, 145, "S2/VLM/concurrent_equal/iteration=0", pid + 3, None),
                ],
            )
            db.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
                [
                    (12, 13, pid + 1, 0),
                    (120, 121, pid + 2, 1),
                    (121, 122, pid + 3, 2),
                    (123, 124, other_pid + 2, 1),
                ],
            )
            db.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
                [
                    (13, 50, 0, 1, pid, 0, 1),
                    (150, 200, 0, 1, pid, 1, 1),
                    (175, 220, 0, 2, pid, 2, 1),
                    (125, 270, 0, 1, other_pid, 1, 1),
                ],
            )
            db.executemany(
                "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?,?,?,?,?)",
                [
                    (258, 0, "GPU2", 1, "SMs Active [Throughput %]"),
                    (259, 1, "GPU3", 1, "DRAM Read Bandwidth [Throughput %]"),
                ],
            )
            db.executemany(
                "INSERT INTO GPU_METRICS VALUES (?,?,?,?,?)",
                [
                    (100, 900, 258, 1, 20),
                    (200, 1000, 258, 1, 50),
                    (300, 1100, 258, 1, 60),
                    (150, 900, 259, 1, 70),
                    (250, 1000, 259, 1, 80),
                ],
            )
            db.execute(
                "INSERT INTO NVTX_EVENTS VALUES (120,130,?, ?,NULL)",
                ("S2/VLM/cooperative_boundary/block0", pid + 3),
            )
            db.execute(
                "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (258,0,'GPU2',2,'GPC Clock Frequency [MHz]')"
            )
            db.executemany(
                "INSERT INTO GPU_METRICS VALUES (?,?,258,2,?)",
                [(100, 900, -1900000000), (200, 1000, -1800000000)],
            )
            db.commit()
            db.close()
            evidence = read_nsys(path)
            self.assertEqual(evidence["execution"]["associated_kernels"], 2)
            self.assertEqual(evidence["execution"]["unassigned_kernels"], 2)
            self.assertEqual(evidence["execution"]["overlap_ms"], 25 / 1e6)
            self.assertEqual(evidence["window_ns"], [[150, 220]])
            self.assertEqual(evidence["metrics"][0]["gpu_id_from_type"], 2)
            self.assertIn("SMs", evidence["metrics"][0]["name"])
            self.assertTrue(
                any("DRAM" in metric["name"] for metric in evidence["metrics"])
            )
            self.assertEqual(
                {
                    kernel["physical_gpu_id"]
                    for kernel in evidence["kernels"]
                    if kernel["component"] != "unassigned"
                },
                {2},
            )
            clock = next(
                metric
                for metric in evidence["metrics"]
                if "Clock Frequency" in metric["name"]
            )
            self.assertFalse(clock["quality"]["interpretable"])
            self.assertEqual(clock["samples"][0][1], -1900000000)
            request = evidence["requests"][0]
            self.assertEqual(request["s1_release_ns"], 115)
            self.assertEqual(request["s1_release_to_first_kernel_ms"], 35 / 1e6)
            self.assertEqual(request["s1_release_to_launch_ms"], 5 / 1e6)
            self.assertEqual(request["s1_launch_to_first_kernel_ms"], 30 / 1e6)

    def test_ncu_preserves_units_and_no_concurrent_kernel_time_sum(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-contention-") as directory:
            path = Path(directory) / "synthetic.csv"
            path.write_text(
                '==PROF== synthetic fixture\n"ID","Metric Name","Metric Unit","Metric Value"\n"0","dram__bytes_read.sum","Mbyte","100"\n"0","dram__bytes_write.sum","Mbyte","20"\n"0","gpu__time_duration.sum","msecond","10"\n'
            )
            one = read_ncu(path, workloads=2, scope="app-range")
            self.assertEqual(one["dram_bytes_per_workload"], 60e6)
            self.assertEqual(one["dram_gb_per_s"], 12)
            with path.open("a") as output:
                output.write('"1","gpu__time_duration.sum","msecond","10"\n')
            self.assertIsNone(
                read_ncu(path, workloads=2, scope="app-range")["dram_gb_per_s"]
            )

    def test_native_ncu_wide_header_units_and_raw_cells(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-ncu-wide-") as directory:
            path = Path(directory) / "range.csv"
            path.write_text(
                "==PROF== synthetic native raw-page fixture\n"
                '"ID","Kernel Name","dram__bytes_read.sum","dram__bytes_write.sum",'
                '"gpu__time_duration.sum","dram__bytes.sum.per_second"\n'
                '"","","Mbyte","Kbyte","usecond","Gbyte/s"\n'
                '"0","<app-range>","1,000","2,000","50,000","20.04"\n'
            )
            result = read_ncu(path, workloads=2, scope="app-range")
            self.assertEqual(result["format"], "wide")
            self.assertEqual(result["dram_total_bytes"], 1_002_000_000)
            self.assertEqual(result["dram_bytes_per_workload"], 501_000_000)
            self.assertAlmostEqual(result["replay_dram_gb_per_s"], 20.04)
            self.assertIsNone(result["normal_service_dram_gb_per_s"])
            read_row = result["rows"][0]
            self.assertEqual(read_row["Metric Value"], "1,000")
            self.assertEqual(read_row["Metric Unit"], "Mbyte")
            self.assertEqual(read_row["source_csv_row_index"], 3)
            self.assertEqual(read_row["source_csv_column_index"], 2)
            self.assertEqual(result["raw_csv_rows"][2][2], "Mbyte")
            kernel = read_ncu(path, workloads=2, scope="kernel")
            self.assertIsNone(kernel["dram_bytes_per_workload"])
            self.assertIsNone(kernel["replay_dram_gb_per_s"])

    def test_parser_imports_and_no_plot_cli_do_not_load_model_or_matplotlib(self):
        with tempfile.TemporaryDirectory(
            prefix="synthetic-parser-import-"
        ) as directory:
            root = Path(directory)
            source = root / "measurement.json"
            source.write_text(
                json.dumps({"samples": [], "protocol": {"capture": "none"}})
            )
            script = (
                "import sys\n"
                "from coexecution.profiling.reporting import main\n"
                f"main(['analyze', '--measurement', {str(source)!r}, '--out', {str(root / 'report')!r}, '--no-plots'])\n"
                "assert 'torch' not in sys.modules\n"
                "assert 'matplotlib' not in sys.modules\n"
            )
            subprocess.run(
                [sys.executable, "-c", script], check=True, capture_output=True
            )
            self.assertTrue((root / "report/evidence.json").is_file())

    def test_svg_labels_and_missing_counter_no_pixel_inspection(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("Matplotlib is not available in this Python environment")
        from coexecution.profiling.figures import plotting, timeline

        try:
            plotting()
        except RuntimeError as error:
            self.skipTest(str(error))

        synthetic = {
            "kernels": [
                {"component": "S2", "start_ns": 1000000, "end_ns": 3000000},
                {"component": "S1", "start_ns": 2000000, "end_ns": 4000000},
            ],
            "metrics": [],
            "requests": [{"s1_release_ns": 1500000}],
        }
        with tempfile.TemporaryDirectory(
            prefix="synthetic-contention-plot-"
        ) as directory:
            files = timeline(
                synthetic,
                {
                    "precision": "SYNTHETIC TEST",
                    "mode": "concurrent_equal",
                    "offset_ms": 1,
                },
                Path(directory),
            )
            self.assertEqual(len(files), 2)
            svg = Path(directory, "timeline.svg")
            labels = " ".join(ET.parse(svg).getroot().itertext())
            self.assertIn("SYNTHETIC TEST", labels)
            self.assertIn("S2 / VLM", labels)
            self.assertIn("S1 / DiT", labels)
            self.assertIn("本次未取得该硬件 counter", labels)
            self.assertIn("#d55e00", svg.read_text().lower())
            self.assertIn("#0072b2", svg.read_text().lower())

    def test_invalid_or_pending_attempt_cannot_enter_campaign(self):
        with tempfile.TemporaryDirectory(
            prefix="synthetic-contention-attempt-"
        ) as directory:
            root = Path(directory)
            attempt = root / "attempts" / "case" / "time"
            path = attempt / "worker" / "measurement.json"
            path.parent.mkdir(parents=True)
            path.write_text("{}")
            self.assertFalse(attempt_acceptance(path, root)["accepted"])
            result = attempt / "result.json"
            result.write_text(
                json.dumps({"state": "invalid", "measurements_valid": False})
            )
            self.assertFalse(attempt_acceptance(path, root)["accepted"])
            result.write_text(
                json.dumps({"state": "complete", "measurements_valid": True})
            )
            self.assertTrue(attempt_acceptance(path, root)["accepted"])
            self.assertTrue(
                attempt_acceptance(root / "pilot" / "measurement.json", root)[
                    "accepted"
                ]
            )
            pending_round = root / "rounds" / "case" / "time" / "measurement.json"
            self.assertFalse(attempt_acceptance(pending_round, root)["accepted"])

    def test_response_field_and_schema_isolation(self):
        from coexecution.profiling.reporting import latency_summary_html

        with tempfile.TemporaryDirectory(
            prefix="synthetic-response-metadata-"
        ) as directory:
            path = Path(directory) / "measurement.json"
            common = {
                "precision": "bf16",
                "mode": "cooperative_dit_priority",
                "gpu": 2,
                "protocol": {"capture": "none", "seed": 1000},
            }
            old = {**common, "schema_version": 1, "samples": [{"s1_gpu_span_ms": 2.0}]}
            path.write_text(json.dumps(old))
            before = read_measurement(path)
            new = {
                **common,
                "schema_version": 2,
                "worker_revision": 3,
                "response_semantics": "host release to own ready",
                "samples": [{"s1_gpu_span_ms": 2.0, "s1_response_ms": 5.0}],
            }
            path.write_text(json.dumps(new))
            after = read_measurement(path)
            self.assertNotEqual(before["cohort_sha256"], after["cohort_sha256"])
            self.assertNotIn("s1_response_ms", before["latency"])
            self.assertEqual(after["latency"]["s1_response_ms"]["p50"], 5.0)
            markup = latency_summary_html(after)
            self.assertLess(
                markup.index("DiT release"), markup.index("DiT CUDA 流跨度")
            )
            self.assertIn("5.000", markup)
            self.assertIn("未记录", latency_summary_html(before))

    def test_campaign_response_ratio_does_not_use_event_span(self):
        from coexecution.profiling.reporting import campaign

        with tempfile.TemporaryDirectory(
            prefix="synthetic-response-campaign-"
        ) as directory:
            root = Path(directory)
            base = {
                "schema_version": 2,
                "worker_revision": 3,
                "precision": "bf16",
                "component": "all",
                "offset_ms": 0,
                "gpu": 2,
                "protocol": {"capture": "none", "seed": 1000},
            }
            cases = {
                "baseline": {
                    **base,
                    "mode": "dit_only",
                    "samples": [
                        {"s1_response_ms": 4.0, "s1_gpu_span_ms": 2.0, "wall_ms": 5.0}
                    ],
                },
                "concurrent": {
                    **base,
                    "mode": "cooperative_dit_priority",
                    "samples": [
                        {"s1_response_ms": 10.0, "s1_gpu_span_ms": 3.0, "wall_ms": 11.0}
                    ],
                },
                "old": {
                    **base,
                    "schema_version": 1,
                    "worker_revision": None,
                    "mode": "cooperative_dit_priority",
                    "samples": [{"s1_gpu_span_ms": 3.0, "wall_ms": 11.0}],
                },
            }
            for name, record in cases.items():
                path = root / name / "measurement.json"
                path.parent.mkdir()
                path.write_text(json.dumps(record))
            with contextlib.redirect_stdout(io.StringIO()):
                campaign(
                    SimpleNamespace(
                        root=root, out=root / "report", no_plots=True, font=None
                    )
                )
            report = json.loads((root / "report/campaign.json").read_text())
            self.assertEqual(len(report["rows"]), 3)
            concurrent = next(
                row
                for row in report["rows"]
                if row["mode"] == "cooperative_dit_priority"
                and "s1_response_ms" in row["latency"]
            )
            self.assertEqual(concurrent["slowdown_vs_isolated"]["S1"], 2.5)
            self.assertEqual(concurrent["stream_span_ratio_vs_isolated"]["S1"], 1.5)
            old = next(
                row for row in report["rows"] if "s1_response_ms" not in row["latency"]
            )
            self.assertIsNone(old["slowdown_vs_isolated"]["S1"])
            self.assertIn("s1_response_ms", (root / "report/latency.csv").read_text())

    def test_measurement_capture_and_protocol_cohort(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-contention-") as directory:
            path = Path(directory) / "measurement.json"
            payload = {
                "precision": "bf16",
                "mode": "dit_only",
                "gpu": 3,
                "protocol": {"capture": "none", "seed": 1000, "iterations": 2},
                "samples": [{"s1_gpu_span_ms": 10.0}, {"s1_gpu_span_ms": 20.0}],
            }
            path.write_text(json.dumps(payload))
            one = read_measurement(path)
            self.assertTrue(one["unprofiled"])
            self.assertEqual(one["latency"]["s1_gpu_span_ms"]["p50"], 15)
            payload["protocol"]["capture"] = "cuda"
            payload["protocol"]["iterations"] = 1
            path.write_text(json.dumps(payload))
            two = read_measurement(path)
            self.assertFalse(two["unprofiled"])
            self.assertEqual(one["cohort_sha256"], two["cohort_sha256"])
            payload["gpu"] = 2
            path.write_text(json.dumps(payload))
            self.assertNotEqual(
                one["cohort_sha256"], read_measurement(path)["cohort_sha256"]
            )
            payload["provenance"] = {
                "worker_modules_sha256": {"execution.py": "first-revision"}
            }
            path.write_text(json.dumps(payload))
            modular = read_measurement(path)
            payload["provenance"]["worker_modules_sha256"]["execution.py"] = (
                "second-revision"
            )
            path.write_text(json.dumps(payload))
            self.assertNotEqual(
                modular["cohort_sha256"], read_measurement(path)["cohort_sha256"]
            )

    def test_nested_report_root_cannot_hide_attempt_validity(self):
        from coexecution.profiling.analysis import attempt_acceptance

        with tempfile.TemporaryDirectory(prefix="synthetic-attempt-root-") as directory:
            root = Path(directory)
            attempt = root / "attempts" / "job" / "run"
            worker = attempt / "worker-passes" / "pass-one"
            worker.mkdir(parents=True)
            measurement = worker / "measurement.json"
            measurement.write_text("{}")
            for result in (
                {"state": "invalid", "measurements_valid": False},
                {"state": "running", "measurements_valid": True},
                [],
                None,
            ):
                (attempt / "result.json").write_text(json.dumps(result))
                self.assertFalse(attempt_acceptance(measurement, worker)["accepted"])
            (attempt / "result.json").unlink()
            self.assertFalse(attempt_acceptance(measurement, worker)["accepted"])
            (attempt / "result.json").write_text(
                json.dumps({"state": "complete", "measurements_valid": True})
            )
            self.assertTrue(attempt_acceptance(measurement, worker)["accepted"])

    def test_campaign_excludes_failed_pending_and_profiled_latency(self):
        from coexecution.profiling.aggregation import aggregate_measurements

        with tempfile.TemporaryDirectory(
            prefix="synthetic-guarded-campaign-"
        ) as directory:
            root = Path(directory)
            common = {
                "precision": "bf16",
                "mode": "dit_only",
                "schema_version": 2,
                "protocol": {"capture": "none"},
                "provenance": {"worker_sha256": "one"},
                "samples": [{"s1_response_ms": 5.0}],
            }
            cases = {
                "valid": (common, {"state": "complete", "measurements_valid": True}),
                "invalid": (common, {"state": "invalid", "measurements_valid": False}),
                "pending": (common, None),
                "profiled": (
                    {**common, "protocol": {"capture": "cuda"}},
                    {"state": "complete", "measurements_valid": True},
                ),
                "other-source": (
                    {**common, "provenance": {"worker_sha256": "two"}},
                    {"state": "complete", "measurements_valid": True},
                ),
            }
            for name, (record, result) in cases.items():
                path = root / "rounds" / name / "measurement.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(record))
                if result is not None:
                    path.with_name("result.json").write_text(json.dumps(result))
            measurements, rows, acceptance = aggregate_measurements(root)
            self.assertEqual(len(measurements), 3)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row["cohort_sha256"] for row in rows}), 2)
            self.assertEqual(
                sum(row["latency"]["s1_response_ms"]["n"] for row in rows), 2
            )
            self.assertEqual(sum(item["accepted"] for item in acceptance.values()), 3)


if __name__ == "__main__":
    unittest.main()
